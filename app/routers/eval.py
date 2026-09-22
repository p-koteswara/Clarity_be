from fastapi import APIRouter
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env is loaded
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_BACKEND_ROOT / ".env")

router = APIRouter()

class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]

class EvalResponse(BaseModel):
    faithfulness: float
    relevance: float
    completeness: float
    overall: float
    feedback: str

def get_llm():
    return ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )

@router.post("/eval", response_model=EvalResponse)
async def evaluate_response(request: EvalRequest):
    llm = get_llm()
    
    context_str = "\n\n".join(request.contexts)
    
    prompt = f"""You are an expert evaluator for RAG systems.
Evaluate the following answer based on the question and context.

Question: {request.question}

Context (source documents):
{context_str}

Answer to evaluate:
{request.answer}

Score each metric from 0.0 to 1.0:
- faithfulness: Is the answer grounded in the provided context? 
  (1.0 = fully grounded, 0.0 = hallucinated)
- relevance: Does the answer address the question asked?
  (1.0 = fully relevant, 0.0 = completely off topic)
- completeness: Does the answer fully address all parts 
  of the question?
  (1.0 = complete, 0.0 = very incomplete)

Respond ONLY with valid JSON, no extra text:
{{
  "faithfulness": <score>,
  "relevance": <score>,
  "completeness": <score>,
  "feedback": "<one sentence summary of answer quality>"
}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        # Clean response and parse JSON
        content = response.content.strip()
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        scores = json.loads(content)
        
        overall = round(
            (float(scores["faithfulness"]) + 
             float(scores["relevance"]) + 
             float(scores["completeness"])) / 3, 2
        )
        
        return EvalResponse(
            faithfulness=float(scores["faithfulness"]),
            relevance=float(scores["relevance"]),
            completeness=float(scores["completeness"]),
            overall=overall,
            feedback=str(scores["feedback"])
        )
    except Exception as e:
        return EvalResponse(
            faithfulness=0.0,
            relevance=0.0,
            completeness=0.0,
            overall=0.0,
            feedback=f"Evaluation failed: {str(e)}"
        )
