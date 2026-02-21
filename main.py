import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()
# -----------------------------
# 1️⃣ Load Documents
# -----------------------------
docs = []
for file in os.listdir("documents"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("documents", file))
        docs.extend(loader.load())

# -----------------------------
# 2️⃣ Split Text
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
splits = splitter.split_documents(docs)

# -----------------------------
# 3️⃣ Embeddings
# -----------------------------
embeddings = OpenAIEmbeddings()

# -----------------------------
# 4️⃣ Vector Store
# -----------------------------
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# 5️⃣ LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",   # cheaper + strong
    temperature=0
)

# -----------------------------
# 6️⃣ Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Use the provided context to compare the documents.
If the answer can be inferred from context, respond clearly.

Context:
{context}

Question:
{question}
""")

# -----------------------------
# 7️⃣ Format Docs
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------
# 8️⃣ RAG Chain
# -----------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# 9️⃣ Run
# -----------------------------
while True:
    query = input("\nAsk: ")
    if query.lower() == "exit":
        break

    response = rag_chain.invoke(query)
    print("\nAnswer:\n", response)