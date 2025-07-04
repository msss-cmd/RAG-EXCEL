import streamlit as st
import pandas as pd
import os
import sqlite3
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import tempfile
from langchain_openai import ChatOpenAI

# Enhanced Page Configuration
st.set_page_config(
    page_title="Chat with Excel/CSV",
    page_icon=":bar_chart:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {background-color: white;}
    .sidebar .sidebar-content {
        background-color: #F1F5F9;
        color: black;
    }
    .sidebar .sidebar-content .stButton>button, .sidebar .sidebar-content h1, .sidebar .sidebar-content h2 {
        color: #1A202C;
    }
    .greeting-text {
        font-size: 3em;
        color: transparent;
        background-image: linear-gradient(90deg, #3b82f6, #ec4899);
        -webkit-background-clip: text;
        font-weight: 600;
        text-align: center;
    }
    .stTextInput > div > input {
        background-color: #F1F5F9;
        color: #1A202C;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
        width: 100%;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-size: 1em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to handle Q&A option
def code_for_chat(api_key):
    st.write('<div class="greeting-text">SSS Talk with Excel!</div>', unsafe_allow_html=True)
    st.sidebar.info("Ask any question about the uploaded Excel or CSV data.")
    st.sidebar.image("https://miro.medium.com/v2/resize:fit:786/format:webp/1*qUFgGhSERoWAa08MV6AVCQ.jpeg", use_container_width=True)

    uploaded_file = st.file_uploader("Upload Excel or CSV file:", type=["xlsx", "csv"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(tmp_file_path)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(tmp_file_path)

        st.write("### Uploaded Data:")
        st.dataframe(df.head(len(df)))

        question = st.text_input("Ask a question:")
        submit = st.button("Ask")

        if submit:
            st.subheader("Answer:")
            st.write("Please wait, answer is generating...")

            llm_1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

            with sqlite3.connect(f"{uploaded_file.name}.db") as conn:
                df.to_sql(f"{uploaded_file.name}s", conn, if_exists="replace")
                db = SQLDatabase.from_uri(f"sqlite:///{uploaded_file.name}.db")
                generate_query = create_sql_query_chain(llm_1, db)
                execute_query = QuerySQLDataBaseTool(db=db)

                answer_prompt = PromptTemplate.from_template(
                    """Given the following user question, SQL query, and SQL result, answer the question.
                    Question: {question}
                    SQL Query: {query}
                    SQL Result: {result}
                    Answer: """
                )

                rephrase_answer = answer_prompt | llm_1 | StrOutputParser()
                chain = (
                    RunnablePassthrough.assign(query=generate_query)
                    .assign(result=itemgetter("query") | execute_query)
                    | rephrase_answer
                )

                response = chain.invoke({"question": question})
                st.subheader(response)

# Main UI layout
def main():
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSZYA5fOIfm6K6v3Lrro3MXksMfO3SdglfSyg&s", use_container_width=True)
    st.title("DocTalk : Chat with Excel/CSV")
    st.sidebar.title("Options")

    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")

    if api_key:
        code_for_chat(api_key)
    else:
        st.sidebar.warning("Please enter your OpenAI API key to proceed.")

if __name__ == "__main__":
    main()
