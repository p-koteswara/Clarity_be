from fastapi import APIRouter
from pydantic import BaseModel
import google.generativeai as genai
from app.embeddings import get_embedding
from app.database import collection

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
async def chat_with_document(request: ChatRequest):
    # Embed the question
    question_embedding = get_embedding(request.question)

    # Query ChromaDB for top 3 chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    retrieved_chunks = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(retrieved_chunks)

    # Build the prompt
    prompt = f"""You are Clarity, an AI assistant that answers questions based on the provided document.
Only answer based on the context below. If the answer isn't in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {request.question}
"""

    # Call Gemini to generate answer
    try:
        # Trying a lite model which might have higher quota/rate limits
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        if "quota" in str(e).lower():
            answer = "The Gemini API quota for this model has been exceeded. Please check your API usage at https://aistudio.google.com/ or try again later. You can also try using a different API key."
        else:
            answer = f"An error occurred while generating the answer: {str(e)}"

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }
