"""Genie as a governed analytics tool (Modules 4, 5 and 7).

Uses the Genie Conversation API through the Databricks SDK. Pass a WorkspaceClient built with the caller's
credentials (on-behalf-of-user, Module 5) so Genie applies that user's permissions and row filters.
"""
from __future__ import annotations

import mlflow
from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool
from pydantic import BaseModel, Field

MAX_ROWS = 20


def ask_genie(space_id: str, question: str, w: WorkspaceClient | None = None) -> str:
    """Ask a Genie space one question; return its SQL, a few result rows and any text reply."""
    w = w or WorkspaceClient()
    msg = w.genie.start_conversation_and_wait(space_id, question)
    parts = []
    for att in msg.attachments or []:
        if getattr(att, "text", None) and att.text.content:
            parts.append(att.text.content)                     # clarification or explanation
        q = getattr(att, "query", None)
        if q:
            parts.append(f"Genie SQL ({q.description or 'query'}):\n{q.query}")
            try:
                res = w.genie.get_message_attachment_query_result(space_id, msg.conversation_id, msg.id, att.attachment_id)
            except AttributeError:                             # older SDKs
                res = w.genie.get_message_query_result(space_id, msg.conversation_id, msg.id)
            sr = res.statement_response
            cols = [c.name for c in sr.manifest.schema.columns]
            rows = (sr.result.data_array or [])[:MAX_ROWS] if sr.result else []
            parts.append(" | ".join(cols) + "\n" + "\n".join(" | ".join(map(str, r)) for r in rows))
    return "\n\n".join(parts) or "NOT_FOUND: Genie returned no answer"


class GenieQuestion(BaseModel):
    question: str = Field(min_length=5, description="A complete analytics question in plain English")


def make_genie_tool(space_id: str, description: str, w_factory=None):
    """w_factory: callable returning a WorkspaceClient per call (per-request identity for OBO)."""

    @tool("sales_analyst", args_schema=GenieQuestion, description=description)
    @mlflow.trace(span_type="TOOL", name="sales_analyst")
    def sales_analyst(question: str) -> str:
        try:
            return ask_genie(space_id, question, w_factory() if w_factory else None)
        except Exception as e:
            msg = str(e)
            if "PERMISSION" in msg.upper() or "403" in msg:
                return "PERMISSION_DENIED: you are not allowed to query this data. Tell the user to contact support."
            return f"TOOL_ERROR: Genie unavailable ({type(e).__name__})"

    return sales_analyst


GENIE_DESCRIPTION = ("Answers analytics questions about orders, revenue, customers, products, regions and late "
                     "deliveries using governed SQL over the sales tables. Use for counts, totals, trends and "
                     "comparisons. NOT for one specific order (use get_order_status) or for policies.")
