import streamlit as st
import pandas as pd
import openai
import faiss
import numpy as np

# Title
st.title("Excel RAG Q&A App")

# API Key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

# OpenAI client setup
def get_openai_client(api_key):
    return openai.OpenAI(api_key=api_key)

# Helper: Load and parse Excel
def load_excel_content(file):
    xls = pd.ExcelFile(file)
    all_data = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df = df.dropna(how='all')
        all_data.append(f"Sheet: {sheet}\n{df.to_string(index=False)}")
    return all_data

# Helper: Embed chunks
def embed_chunks(chunks, client):
    embeddings = []
    for chunk in chunks:
        embedding = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding
        embeddings.append(embedding)
    return embeddings

# Helper: Build FAISS index
def build_faiss_index(vectors):
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))
    return index

# Helper: Get top matching chunks
def get_top_chunks(query, chunks, index, client, k=3):
    query_vector = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
    D, I = index.search(np.array([query_vector]).astype("float32"), k)
    return [chunks[i] for i in I[0]]

# Helper: Generate GPT response
def generate_answer(prompt, client):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Process uploaded file
if uploaded_file and api_key:
    st.info("Processing file and generating embeddings...")
    client = get_openai_client(api_key)
    chunks = load_excel_content(uploaded_file)
    vectors = embed_chunks(chunks, client)
    index = build_faiss_index(vectors)
    st.session_state['chunks'] = chunks
    st.session_state['index'] = index
    st.session_state['client'] = client
    st.success("Setup complete! You can now ask questions.")

# Ask question
if api_key and 'index' in st.session_state:
    query = st.text_input("Ask a question")
    if query:
        top_chunks = get_top_chunks(query, st.session_state['chunks'], st.session_state['index'], st.session_state['client'])
        context = "\n\n".join(top_chunks)
        final_prompt = f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
        answer = generate_answer(final_prompt, st.session_state['client'])
        st.markdown("### Answer")
        st.write(answer)
