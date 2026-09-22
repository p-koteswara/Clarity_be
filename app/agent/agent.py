"""LangGraph ReAct agent wired to Clarity tools and thread memory."""

from app.agent.tools import search_web
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agent.memory import get_config, memory
from app.agent.tools import calculate, get_current_datetime, search_clarity_docs

# Load .env from the backend root so OPENROUTER_API_KEY is found regardless of cwd.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_BACKEND_ROOT / ".env")

# Built on first use so a missing OpenRouter key does not crash the whole API.
_agent = None


def _get_agent():
    """Create (or reuse) the ReAct agent. Requires OPENROUTER_API_KEY."""
    global _agent
    if _agent is not None:
        return _agent

    # Re-read .env so a newly added key is picked up without restarting Uvicorn.
    load_dotenv(_BACKEND_ROOT / ".env", override=False)
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to clarity_backend/.env and restart."
        )

    # OpenRouter-hosted free model via the OpenAI-compatible ChatOpenAI client.
    llm = ChatOpenAI(
        model="nvidia/nemotron-3.5-lightning:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0,
    )

    # ReAct agent: LLM decides when to call tools; MemorySaver keeps history per thread.
    _agent = create_react_agent(
        llm,
        tools=[search_clarity_docs, calculate, get_current_datetime, search_web],
        checkpointer=memory,
    )
    return _agent


def run_agent(message: str, thread_id: str) -> str:
    """Invoke the agent for one user turn and return the last AI reply as text."""
    try:
        result = _get_agent().invoke(
            {"messages": [HumanMessage(content=message)]},
            config=get_config(thread_id),
        )
        messages = result.get("messages") or []
        # Walk backwards so we return the final assistant message, not a tool call.
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if isinstance(content, str):
                    return content
                return str(content)
        return "The agent did not return a response."
    except Exception as exc:
        return f"Agent error: {exc}"
