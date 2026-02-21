# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application built using OpenAI GPT, LangChain, FAISS, and Streamlit.

This application allows users to upload multiple PDF documents and perform contextual question answering strictly based on the uploaded content.

---

## Features

- Upload multiple PDF documents
- Semantic search using FAISS vector database
- Context-aware answers using OpenAI GPT-4o-mini
- Grounded responses (no hallucination outside context)
- Streamlit-based web interface
- Secure API key management using environment variables

---

## Installation (Local Setup)

Clone the repository:

git clone https://github.com/YOUR_USERNAME/rag-document-assistant.git  
cd rag-document-assistant  

Create virtual environment:

python -m venv .venv  
.venv\Scripts\activate  

Install dependencies:

pip install -r requirements.txt  

Create a `.env` file:

OPENAI_API_KEY=your_openai_key_here  

Run the application:

streamlit run app.py  

---

## Use Cases

- Resume screening
- Policy document analysis
- Research document comparison
- Multi-document knowledge assistant

---

## Future Improvements

- Source citation display
- Conversational memory
- Persistent vector storage
- Cloud deployment

---

## Author

Venkata Reddy  
Data Science & Generative AI Enthusiast