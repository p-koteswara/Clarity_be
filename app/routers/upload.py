from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from pypdf import PdfReader
from docx import Document
from app.embeddings import chunk_text, get_embedding
from app.database import collection

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    # Save the file temporarily
    upload_path = f"uploads/{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract text based on file type
        text = ""
        extension = os.path.splitext(file.filename)[1].lower()

        if extension == ".pdf":
            reader = PdfReader(upload_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
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
            raise HTTPException(status_code=400, detail="Unsupported file type")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from document")

        # Chunk the text
        chunks = chunk_text(text)

        # Clear existing collection before indexing new document
        # ChromaDB delete with empty filter or all IDs
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

        # Embed and store chunks
        ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
        embeddings = [get_embedding(chunk) for chunk in chunks]
        metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return {
            "message": "Document indexed successfully",
            "chunks": len(chunks),
            "filename": file.filename
        }

    finally:
        # Cleanup: optionally remove the uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)
