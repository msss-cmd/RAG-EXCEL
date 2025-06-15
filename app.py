import streamlit as st
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
import faiss
import numpy as np

# Title
st.title("Excel RAG Q&A App")

# API Key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

# Helper: Load and parse Excel
@st.cache_data(show_spinner=False)
def load_excel_content(file):
    xls = pd.ExcelFile(file)
    all_data = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df = df.dropna(how='all')
        all_data.append(f"Sheet: {sheet}\n{df.to_string(index=False)}")
    return all_data

# Helper: Embed chunks
@st.cache_resource(show_spinner=False)
def embed_chunks(chunks, api_key):
    openai.api_key = api_key
    return [get_embedding(chunk, engine="text-embedding-3-small") for chunk in chunks]

# Helper: Build FAISS index
@st.cache_resource(show_spinner=False)
def build_faiss_index(vectors):
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))
    return index

# Helper: Get top matching chunks
def get_top_chunks(query, chunks, index, api_key, k=3):
    query_vector = get_embedding(query, engine="text-embedding-3-small")
    D, I = index.search(np.array([query_vector]).astype("float32"), k)
    return [chunks[i] for i in I[0]]

# Helper: Generate GPT response
def generate_answer(prompt, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]

# Process uploaded file
if uploaded_file and api_key:
    st.info("Processing file and generating embeddings...")
    chunks = load_excel_content(uploaded_file)
    vectors = embed_chunks(chunks, api_key)
    index = build_faiss_index(vectors)
    st.session_state['chunks'] = chunks
    st.session_state['index'] = index
    st.success("Setup complete! You can now ask questions.")

# Ask question
if api_key and 'index' in st.session_state:
    query = st.text_input("Ask a question")
    if query:
        top_chunks = get_top_chunks(query, st.session_state['chunks'], st.session_state['index'], api_key)
        context = "\n\n".join(top_chunks)
        final_prompt = f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
        answer = generate_answer(final_prompt, api_key)
        st.markdown("### Answer")
        st.write(answer)
