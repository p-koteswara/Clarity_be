import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.agent.memory import get_config
from pydantic import BaseModel

try:
    from app.agent.agent import agent_executor
except ImportError:
    try:
        from app.agent.agent import _get_agent
        agent_executor = None
    except ImportError:
        agent_executor = None
        _get_agent = None

router = APIRouter()


async def stream_agent(question: str, thread_id: str):
    config = get_config(thread_id)
    executor = agent_executor if agent_executor is not None else (_get_agent() if _get_agent else None)
    if executor is None:
        yield f"data: {json.dumps({'error': 'Agent executor not initialized'})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        return

    try:
        async for event in executor.astream_events(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'token': chunk.content, 'done': False})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"


@router.post("/agent/stream")
async def stream_agent_response(request: dict):
    question = request.get("question", "")
    thread_id = request.get("thread_id", "default")
    return StreamingResponse(
        stream_agent(question, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class StreamRequest(BaseModel):
    question: str
    thread_id: str = "default"

@router.post("/agent/stream")
async def stream_agent_response(request: StreamRequest):
    return StreamingResponse(
        stream_agent(request.question, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )