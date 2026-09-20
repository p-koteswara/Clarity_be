"""FastAPI router for the LangGraph agent chat endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.agent import run_agent

router = APIRouter()


class AgentChatRequest(BaseModel):
    """Incoming agent chat payload."""

    question: str = Field(..., min_length=1)
    thread_id: str = "default"


class AgentChatResponse(BaseModel):
    """Outgoing agent chat payload."""

    answer: str
    thread_id: str
    mode: str = "agent"


@router.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """Run one agent turn, scoped by thread_id so conversation memory persists."""
    try:
        answer = run_agent(request.question, request.thread_id)
        return AgentChatResponse(
            answer=answer,
            thread_id=request.thread_id,
            mode="agent",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
