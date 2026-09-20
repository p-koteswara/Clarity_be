"""Tools the Clarity LangGraph agent can call during a conversation."""

import ast
import operator
from datetime import datetime

from langchain_core.tools import tool

from app.database import collection
from app.embeddings import get_embedding

# Allowed operators for the calculate tool (no arbitrary eval).
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a math AST using only numeric literals and operators."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def search_clarity_docs(query: str) -> str:
    """Search through the indexed document to find relevant information. Use this for any question about the uploaded document."""
    try:
        query_embedding = get_embedding(query, task_type="retrieval_query")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )
        chunks = results["documents"][0] if results.get("documents") else []
        if not chunks:
            return "No relevant information found in the indexed document."
        return "\n\n".join(chunks)
    except Exception as exc:
        return f"Error searching documents: {exc}"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Example: '15 * 4 + 2'"""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as exc:
        return f"Invalid expression: {exc}"


@tool
def get_current_datetime() -> str:
    """Get the current date and time"""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
