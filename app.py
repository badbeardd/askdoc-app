import sys
import types
import torch
print(torch.cuda.is_available())
# Patch torch.classes to prevent Streamlit from inspecting it
sys.modules['torch.classes'] = types.ModuleType('torch.classes')
torch.classes = sys.modules['torch.classes']
import hashlib
import streamlit as st
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import os
import textwrap
import tempfile
from dotenv import load_dotenv

from langchain_community.llms import Together
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant. Answer the user’s question based on the provided context. 
Be **concise** and **summarize key points** only. If the answer is too long, shorten it to fit within a short paragraph.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer (short and to-the-point):
""")


# 🔐 Load API Key
try:
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
except Exception:
    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

TOGETHER_MODEL = "meta-llama/Llama-3-8b-chat-hf"

# 📘 Streamlit UI
st.set_page_config(page_title="AskDoc – Conversational RAG", layout="wide")
st.title("📘 AskDoc – Smart Conversational PDF Q&A")

# 🧠 Cache uploaded file in session
def get_file_hash(file_obj):
    file_obj.seek(0)
    file_hash = hashlib.md5(file_obj.read()).hexdigest()
    file_obj.seek(0)
    return file_hash

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
question = st.text_input("💬 Ask something about the document")
submit = st.button("🔍 Ask")

# 🔧 Text splitting
def load_and_chunk(file):
    text = ""
    temp_fd, temp_path = tempfile.mkstemp()
    os.close(temp_fd)  # Close the file descriptor immediately

    with open(temp_path, "wb") as f:
        f.write(file.read())

    try:
        if file.name.endswith(".pdf"):
            doc = fitz.open(temp_path)
            for page in doc:
                text += page.get_text()
            doc.close()  # ✅ CLOSE the document before deleting
        else:
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read()
    finally:
        os.remove(temp_path)  # ✅ Safe to remove after doc is closed

    chunks = textwrap.wrap(text, width=500, break_long_words=False)
    return chunks


# 🔍 Vectorstore from chunks
def create_vectorstore(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    vectordb = FAISS.from_documents(documents, embedding_model)
    return vectordb

# 🧠 Setup LangChain QA chain
def create_qa_chain(vectordb):
    llm = Together(
        model=TOGETHER_MODEL,
        temperature=0.2,
        together_api_key=TOGETHER_API_KEY
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={
            "prompt": CONDENSE_QUESTION_PROMPT
        }
    )
    return qa_chain


# Session state setup
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# 📥 File processing + vectorstore caching
if uploaded_file is not None:
    current_hash = get_file_hash(uploaded_file)
    if current_hash != st.session_state.get("file_hash"):
        if "doc_loaded" not in st.session_state:
            chunks = load_and_chunk(uploaded_file)
            vectordb = create_vectorstore(chunks)
            st.session_state.vectorstore = vectordb
            st.session_state.qa_chain = create_qa_chain(vectordb)
            st.session_state.file_hash = current_hash
            st.session_state.doc_loaded = True
            st.success(f"✅ Document indexed with {len(chunks)} chunks.")
    else:
        st.info("📂 Same document detected. Skipping reprocessing.")

# 🧾 Ask Question
if submit and question:
    if not st.session_state.qa_chain:
        st.warning("⚠️ Please upload a document first.")
    else:
        with st.spinner("🤖 Thinking..."):
            answer = st.session_state.qa_chain.run(question)
        st.subheader("🟢 Answer:")
        st.markdown(f"> {answer}")
