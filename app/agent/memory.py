"""Conversation memory for the LangGraph agent (thread-scoped checkpoints)."""

from langgraph.checkpoint.memory import MemorySaver

# Single global checkpointer shared by the agent so threads persist in-process.
memory = MemorySaver()


def get_config(thread_id: str) -> dict:
    """Return the LangGraph config dict that scopes state to a conversation thread."""
    return {"configurable": {"thread_id": thread_id}}
