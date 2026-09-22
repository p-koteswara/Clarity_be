"""Routes for listing and deleting indexed documents."""

from fastapi import APIRouter, HTTPException

from app.database import delete_document, list_documents

router = APIRouter()


@router.get("/documents")
async def get_documents():
    """List all uploaded documents with filename, doc_id, upload_time, and chunk_count."""
    return {"documents": list_documents()}


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Delete every chunk stored for a specific document."""
    deleted = delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted", "doc_id": doc_id}
