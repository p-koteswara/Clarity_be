from fastapi import APIRouter
from pydantic import BaseModel
import google.generativeai as genai
from app.database import search_documents

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    doc_id: str | None = None

@router.post("/chat")
async def chat_with_document(request: ChatRequest):
    # Search one document when doc_id is set; otherwise search all indexed docs
    retrieved_chunks = search_documents(request.question, doc_id=request.doc_id)
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are Clarity, an AI assistant that answers questions based on the provided document.
Only answer based on the context below. If the answer isn't in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {request.question}
"""

    model = genai.GenerativeModel("gemini-flash-latest")
    response = model.generate_content(prompt)
    
    return {
        "answer": response.text,
        "sources": retrieved_chunks
    }
