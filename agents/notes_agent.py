"""
Notes Agent – context memory specialist (local vector DB).
Returns unified AgentResult (answer, evidence, results_count, status).
"""

from langchain_core.retrievers import BaseRetriever

from agents.base_agent import run_rag
from schemas.evidence import AgentResult


def run(query: str, retriever: BaseRetriever) -> AgentResult:
    """Answer using RAG over local notes. Returns unified AgentResult."""
    docs = retriever.invoke(query)
    answer = run_rag(query, retriever, notes_mode=True)
    evidence = []
    for d in docs:
        content = (getattr(d, "page_content", None) or str(d))[:2000]
        evidence.append({
            "source_type": "notes",
            "source_title": getattr(d, "metadata", {}).get("title", "Note") if hasattr(d, "metadata") else "Note",
            "source_url": getattr(d, "metadata", {}).get("source", "") if hasattr(d, "metadata") else "",
            "extracted_evidence": content,
            "retrieval_reason": "Local notes vector match",
        })
    return {
        "answer": answer,
        "evidence": evidence,
        "results_count": len(evidence),
        "status": "ok",
    }
