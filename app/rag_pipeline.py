# ============================
# app/rag_pipeline.py
# ============================

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load embeddings
embeddings = OllamaEmbeddings(model="qwen2:0.5b")

# Load vector DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="qwen2:0.5b")

SYSTEM_PROMPT = """
You are a Mall AI Shopping Assistant.

STRICT RULES:
- Answer ONLY from provided context.
- If nothing matches, say:
  "I couldn't find that product in our mall database."
- Always include:
  Product Name
  Final Price
  Discount
  Store Name
  Floor and Location
- Do not hallucinate.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


def ask_mall_assistant(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content