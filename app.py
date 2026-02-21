import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

st.set_page_config(page_title="Dynamic RAG Assistant", page_icon="📄")
st.title("📄 Upload & Ask - RAG Assistant")

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload 1 to 5 PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# Function to Build RAG
# -----------------------------
def build_rag_pipeline(files):

    docs = []

    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an assistant.

    Answer ONLY using the provided context.
    If the answer is not in context, say "I don't know."

    Context:
    {context}

    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# -----------------------------
# Run RAG if Files Uploaded
# -----------------------------
if uploaded_files and len(uploaded_files) <= 5:

    st.success(f"{len(uploaded_files)} document(s) uploaded successfully.")

    rag_chain = build_rag_pipeline(uploaded_files)

    user_query = st.text_input("Ask a question about the uploaded documents:")

    if user_query:
        with st.spinner("Analyzing documents..."):
            response = rag_chain.invoke(user_query)
            st.markdown("### 📌 Answer:")
            st.write(response)

elif uploaded_files and len(uploaded_files) > 5:
    st.warning("Please upload a maximum of 5 documents.")