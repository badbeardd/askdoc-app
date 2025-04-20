import streamlit as st
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import faiss
import numpy as np
import os
import textwrap
import tempfile
import requests
from dotenv import load_dotenv

# 🔐 Load environment variable (locally only)
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "meta-llama/Llama-3-8b-instruct"  # safer version
@st.cache_resource
def load_model():
    return SentenceTransformer('intfloat/e5-large-v2')

model = load_model()

st.title("📘 AskDoc – Smart PDF Q&A")

uploaded_file = st.file_uploader("📄 Upload a PDF or TXT file", type=["pdf", "txt"])
question = st.text_input("💬 Ask a question about the document:")
submit = st.button("🔍 Get Answer")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.index = None

def load_and_chunk(file):
    text = ""
    temp_path = tempfile.mkstemp()[1]
    with open(temp_path, "wb") as f:
        f.write(file.read())
    if file.name.endswith(".pdf"):
        doc = fitz.open(temp_path)
        for page in doc:
            text += page.get_text()
    else:
        with open(temp_path, "r", encoding="utf-8") as f:
            text = f.read()
    os.remove(temp_path)
    chunks = textwrap.wrap(text, width=500, break_long_words=False)
    return chunks

def embed_and_index(chunks):
    vectors = model.encode(chunks, convert_to_numpy=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

def generate_answer_llama(context, question):
    prompt = f"""You are an AI assistant. Use the context below to answer the question accurately and concisely.

Context:
{context}

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": TOGETHER_MODEL,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["</s>"]
    }

    response = requests.post(
        "https://api.together.xyz/v1/completions",
        headers=headers,
        json=body
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['text'].strip()
    else:
        return f"Error: {response.text}"

if uploaded_file:
    chunks = load_and_chunk(uploaded_file)
    index, vectors = embed_and_index(chunks)
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.vectors = vectors
    st.success(f"✅ Indexed {len(chunks)} chunks!")

if submit and question:
    if st.session_state.index is None:
        st.warning("⚠️ Please upload a document first.")
    else:
        q_vector = model.encode([question], convert_to_numpy=True)
        D, I = st.session_state.index.search(q_vector, k=3)
        context_chunks = [st.session_state.chunks[i] for i in I[0]]
        context = "\n\n".join(context_chunks)

        with st.spinner("🤖 Generating answer using LLaMA3..."):
            answer = generate_answer_llama(context, question)

        st.subheader("🟢 Answer:")
        st.markdown(f"> {answer}")
