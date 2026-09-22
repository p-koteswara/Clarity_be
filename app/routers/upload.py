from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid
from datetime import datetime, timezone
from pypdf import PdfReader
from docx import Document
from app.embeddings import chunk_text, get_embedding
from app.database import collection

router = APIRouter()


def _extract_text(upload_path: str, filename: str) -> str:
    """Read file contents based on extension."""
    text = ""
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        reader = PdfReader(upload_path)
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"
    elif extension == ".docx":
        doc = Document(upload_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif extension in [".txt", ".md"]:
        try:
            with open(upload_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(upload_path, "r", encoding="latin-1") as f:
                text = f.read()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")

    if not text.strip():
        raise HTTPException(status_code=400, detail=f"Could not extract text from document: {filename}")
    return text


def _index_file(file: UploadFile) -> dict:
    """Chunk, embed, and store one document without replacing others."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="File is missing a filename")

    os.makedirs("uploads", exist_ok=True)
    upload_path = f"uploads/{uuid.uuid4()}_{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = _extract_text(upload_path, file.filename)
        chunks = chunk_text(text)

        doc_id = str(uuid.uuid4())
        upload_time = datetime.now(timezone.utc).isoformat()

        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        embeddings = [get_embedding(chunk) for chunk in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": file.filename,
                "upload_time": upload_time,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return {
            "filename": file.filename,
            "doc_id": doc_id,
            "chunks": len(chunks),
        }
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)


@router.post("/upload")
async def upload_document(file: list[UploadFile] = File(...)):
    """Upload one or more documents. Existing documents are kept.

    In Swagger, use the file picker to select multiple files, then Execute.
    The original single-file `file` field name is unchanged for the frontend.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    indexed = [_index_file(item) for item in file]
    total_chunks = sum(doc["chunks"] for doc in indexed)

    return {
        "message": (
            "1 document indexed successfully"
            if len(indexed) == 1
            else f"{len(indexed)} documents indexed successfully"
        ),
        "total_chunks": total_chunks,
        "count": len(indexed),
        "documents": indexed,
    }
