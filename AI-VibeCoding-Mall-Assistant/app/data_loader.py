from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

products = [
    {
        "store": "TechZone",
        "category": "Electronics",
        "floor": "2",
        "location": "B12",
        "product": "Gaming Laptop XPro 15",
        "price": 95000,
        "discount": "10%",
        "final_price": 85500,
        "features": "16GB RAM, RTX 4060, 1TB SSD, 144Hz",
        "stock": "Available"
    },
    {
        "store": "SmartWorld",
        "category": "Electronics",
        "floor": "1",
        "location": "A04",
        "product": "Budget Laptop LiteBook 14",
        "price": 45000,
        "discount": "15%",
        "final_price": 38250,
        "features": "8GB RAM, 512GB SSD, Intel i5",
        "stock": "Available"
    }
]

documents = []

for p in products:
    content = f"""
Store: {p['store']}
Category: {p['category']}
Floor: {p['floor']}
Location: {p['location']}
Product: {p['product']}
Price: {p['price']}
Discount: {p['discount']}
Final Price: {p['final_price']}
Features: {p['features']}
Stock: {p['stock']}
"""
    documents.append(Document(page_content=content))

embeddings = OllamaEmbeddings(model="qwen2:0.5b")

Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("DB rebuilt cleanly.")