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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Together
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
You are a helpful and intelligent AI assistant reading a document on behalf of the user.

Instructions:
- First, attempt to answer the question using the provided document context.
- If the answer is not fully present in the context, you may rely on your own general knowledge to complete the answer.
- If the document is a resume, extract key highlights such as skills, projects, education, and certifications.
- Use bullet points if the question asks for "points", "summary", or "understanding".
- Do NOT copy raw sentences from the document unless specifically asked to.
- Be concise, insightful, and avoid repeating irrelevant details.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Your response:
""")

# 📦 Required Libraries


# 🔐 Load API Key
try:
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
except Exception:
    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

TOGETHER_MODEL = "meta-llama/Llama-3-8b-chat-hf"

# ✅ Clean LLM output to remove markdown junk, repetition, or assistant tags
import re

def clean_output(text: str) -> str:
    """
    Cleans common garbage from LLM output like repetitive pipes, markdown junk, or AI assistant tags.
    """
    # Remove trailing pipe symbols and formatting junk
    text = re.sub(r"(\|\s*){3,}", "", text)

    # Remove repeated "AI Assistant" or signature-like tails
    text = re.sub(r"(AI Assistant\.?\s*){2,}", "", text, flags=re.IGNORECASE)

    # Remove long sequences of pipes/spaces/newlines at the end
    text = re.sub(r"\n?[\|\s]{20,}$", "", text)

    return text.strip()

def generate_answer(question: str, qa_chain=None, fallback_llm=None) -> str:
    """
    Run question through QA chain, fallback to base LLM if needed.
    """
    if qa_chain:
        try:
            answer = qa_chain.run(question)
            if not answer.strip() or len(answer.strip()) < 10:
                raise ValueError("Answer too short or empty, triggering fallback.")
        except Exception:
            if fallback_llm:
                answer = fallback_llm.invoke(question)
                answer += "\n\n_(Used general knowledge due to a fallback)_"
            else:
                answer = "⚠️ Could not retrieve an answer."
    else:
        if fallback_llm:
            answer = fallback_llm.invoke(question)
            answer += "\n\n_(No document provided. Used general knowledge.)_"
        else:
            answer = "⚠️ No model available to generate an answer."

    return clean_output(answer.strip())

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

# 🔁 Use cached file if uploader returns None on rerun
file = uploaded_file or st.session_state.get("uploaded_file")

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

    # ✅ Use better chunking
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
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
        temperature=0.7,
        together_api_key=TOGETHER_API_KEY
    )
    memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
    )

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
if "doc_loaded" not in st.session_state:    
    st.session_state.doc_loaded = False


# 🔁 Session defaults
for key, default in {
    "uploaded_file": None,
    "file_hash": None,
    "doc_loaded": False,
    "vectorstore": None,
    "qa_chain": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 📥 File processing
file = uploaded_file or st.session_state["uploaded_file"]

if file is not None:
    if st.session_state["uploaded_file"] is None:
        st.session_state["uploaded_file"] = file

    current_hash = get_file_hash(file)

    if current_hash != st.session_state["file_hash"]:
        chunks = load_and_chunk(file)
        vectordb = create_vectorstore(chunks)
        st.session_state.vectorstore = vectordb
        st.session_state.qa_chain = create_qa_chain(vectordb)
        st.session_state.file_hash = current_hash
        st.success(f"✅ Document indexed with {len(chunks)} chunks.")
    else:
        st.info("📂 Same document detected. Skipping reprocessing.")
else:
    st.info("📂 Please upload a PDF or TXT document.")


# 🧾 Ask Question
if submit and question:
    with st.spinner("🤖 Thinking..."):
        fallback_llm = Together(
            model=TOGETHER_MODEL,
            temperature=0.7,
            together_api_key=TOGETHER_API_KEY
        )
        answer = generate_answer(
            question=question,
            qa_chain=st.session_state.get("qa_chain"),
            fallback_llm=fallback_llm
        )

    st.subheader("🟢 Answer:")
    st.markdown(f"> {answer}")

