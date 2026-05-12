import chromadb
from chromadb.config import Settings

# Persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the collection
# We use get_or_create_collection to avoid errors if it already exists
collection = client.get_or_create_collection(name="documents")
