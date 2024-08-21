import streamlit as st
import fitz  # PyMuPDF
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

import PyPDF2
import time
from docx import Document
import docx

# Streamlit interface
st.title('PDF-based Question Answering Chatbot')

# Initialize ChatOpenAI with your OpenAI API key
llm = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    openai_api_key="EMPTY",  # Replace with your actual OpenAI API key
    openai_api_base="https://api.openai.com/v1",
    temperature=1
)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
        if uploaded_file.type == "application/pdf":
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(uploaded_file.name)
            docs = loader.load()
        else:
            st.warning("Currently, only PDF files are supported.")

        st.session_state.uploaded_files = docs

# Question box
question = st.text_input("Ask a question based on the uploaded PDF:")
if question:
    if 'pdf_text' in st.session_state:
        # Use LangChain to answer the question based on the PDF text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(st.session_state.uploaded_files)

        vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
        retriever = vectorstore.as_retriever()

        rag_chain = create_retrieval_chain(retriever, question)
        results = rag_chain.invoke({"input": "Generate the quiz"})
        final_quiz = results['answer']
        st.write(final_quiz)
    else:
        st.warning("Please upload a PDF file first.")
