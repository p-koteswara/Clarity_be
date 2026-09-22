import chromadb

from app.embeddings import get_embedding

# Persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get the collection
# We use get_or_create_collection to avoid errors if it already exists
collection = client.get_or_create_collection(name="documents")


def list_documents() -> list[dict]:
    """Return unique documents currently stored in ChromaDB."""
    results = collection.get(include=["metadatas"])
    documents: dict[str, dict] = {}

    for meta in results.get("metadatas") or []:
        if not meta:
            continue

        # Prefer doc_id; fall back so legacy chunks without it still appear.
        doc_id = meta.get("doc_id") or meta.get("source") or "unknown"
        if doc_id not in documents:
            documents[doc_id] = {
                "doc_id": str(doc_id),
                "filename": meta.get("filename") or meta.get("source") or "unknown",
                "upload_time": meta.get("upload_time") or "",
                "chunk_count": 0,
            }
        documents[doc_id]["chunk_count"] += 1

    return list(documents.values())


def delete_document(doc_id: str) -> bool:
    """Remove every chunk belonging to doc_id. Returns False if none found."""
    existing = collection.get(where={"doc_id": doc_id})
    ids = existing.get("ids") or []
    if not ids:
        return False
    collection.delete(ids=ids)
    return True


def search_documents(query: str, doc_id: str | None = None, n_results: int = 3) -> list[str]:
    """Semantic search across all docs, or only the given doc_id when provided."""
    query_embedding = get_embedding(query, task_type="retrieval_query")
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if doc_id:
        query_kwargs["where"] = {"doc_id": doc_id}

    results = collection.query(**query_kwargs)
    if not results.get("documents"):
        return []
    return results["documents"][0] or []
