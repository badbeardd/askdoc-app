import sys
import types
import os
import re
import hashlib
import tempfile
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

# --- PATCH TORCH (Fixes Streamlit Cloud issues) ---
import torch
sys.modules['torch.classes'] = types.ModuleType('torch.classes')
torch.classes = sys.modules['torch.classes']

# --- IMPORTS ---
# Use the safe fallback for imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_together import Together
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="AskDoc – Conversational RAG", layout="wide")
st.title("📘 AskDoc – Smart Conversational PDF Q&A")

# 1. SETUP API KEY
try:
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
except Exception:
    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    st.error("❌ API Key missing! Please set TOGETHER_API_KEY in .env or Streamlit secrets.")
    st.stop()

# 2. SET THE MODEL
TOGETHER_MODEL = "ServiceNow-AI/Apriel-1.6-15b-Thinker"

# --- PROMPT TEMPLATES ---

# Prompt 1: Rewrite follow-up questions
condense_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_prompt_template)

# Prompt 2: Strict Answer Prompt (Stops Hallucinations)
qa_prompt_template = """You are a helpful AI assistant answering questions about a PDF document.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the context provided below.
2. Do NOT output Python code, Jupyter blocks, or variables like 'response ='.
3. Do NOT mention 'Tactic 2' or 'reference text'.
4. Just give the direct answer in plain text.
5. STOP immediately after answering. Do not analyze or critique your own answer.

Context:
{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=qa_prompt_template, input_variables=["context", "question"]
)

# --- HELPER FUNCTIONS ---

def clean_output(text: str) -> str:
    """
    Cleans the 'Thinking' logs and other junk from the Apriel/DeepSeek models.
    """
    # 1. Remove Apriel/DeepSeek <think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. Remove "Reasoning steps" text if it leaks
    text = re.sub(r'Here are my reasoning steps:.*', '', text, flags=re.DOTALL)
    
    # 3. Remove "System Prompt" leakage
    if "prompt =" in text:
        text = text.split("prompt =")[0]
        
    return text.strip()

def load_and_chunk(file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        if file.name.endswith(".pdf"):
            doc = fitz.open(tmp_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        else:
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()
    finally:
        os.remove(tmp_path)

    # Use the global import, don't re-import here
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks

@st.cache_resource
def get_embedding_model():
    # Cache to prevent reloading (Faster)
    return HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

def create_vectorstore(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    embedding_model = get_embedding_model()
    vectordb = FAISS.from_documents(documents, embedding_model)
    return vectordb

def create_qa_chain(vectordb):
    # FIXED INDENTATION HERE
    llm = Together(
        model=TOGETHER_MODEL,
        temperature=0.6,
        together_api_key=TOGETHER_API_KEY,
        # STOP SEQUENCES (The Kill Switch)
        stop=["<|eot_id|>", "<|eom_id|>", "prompt =", "User:", "Example:"]
    )
    
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=1000
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT} 
    )
    return qa_chain

# --- SESSION STATE ---
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# --- MAIN UI LOGIC ---
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    # Check if file changed
    file_obj = uploaded_file
    file_obj.seek(0)
    current_hash = hashlib.md5(file_obj.read()).hexdigest()
    file_obj.seek(0)

    if current_hash != st.session_state.file_hash:
        with st.spinner("Processing document..."):
            chunks = load_and_chunk(uploaded_file)
            vectordb = create_vectorstore(chunks)
            st.session_state.qa_chain = create_qa_chain(vectordb)
            st.session_state.file_hash = current_hash
            st.success(f"✅ Indexed {len(chunks)} chunks!")
    else:
        st.info("Using cached document.")

# Chat Interface
for msg in st.session_state.get("chat_history", []):
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Ask about your document..."):
    st.chat_message("user").write(question)
    
    if st.session_state.qa_chain:
        with st.spinner("Thinking..."):
            try:
                res = st.session_state.qa_chain({"question": question})
                answer = clean_output(res["answer"])
                st.chat_message("assistant").write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please upload a document first.")
