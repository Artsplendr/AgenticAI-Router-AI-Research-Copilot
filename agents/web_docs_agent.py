"""
Web Docs Agent – practical knowledge specialist (Tavily web search).
Returns unified AgentResult (answer, evidence, results_count, status).
"""

from langchain_core.retrievers import BaseRetriever

from agents.base_agent import run_rag_with_context, _format_docs
from schemas.evidence import AgentResult
from tools.tavily_search import search_with_status


def run(query: str, retriever: BaseRetriever) -> AgentResult:
    """Answer using Tavily first, then vector store. Returns unified AgentResult."""
    context_parts = []
    n_results = 0
    status = "no_key"
    results_list = []

    tavily_context, n_results, status, results_list = search_with_status(
        query, max_results=10, search_depth="advanced", expand_query=True
    )
    if tavily_context:
        context_parts.append("Live web search results (primary):\n\n" + tavily_context)

    docs = retriever.invoke(query)
    store_context = _format_docs(docs)
    if store_context.strip():
        context_parts.append("Ingested docs (supplementary):\n\n" + store_context)

    combined_context = "\n\n---\n\n".join(context_parts) if context_parts else "(No context available.)"
    answer = run_rag_with_context(query, combined_context, allow_supplement=False)

    evidence = []
    for r in results_list:
        evidence.append({
            "source_type": "web_docs",
            "source_title": r.get("title", ""),
            "source_url": r.get("url", ""),
            "extracted_evidence": (r.get("content", "") or "")[:2000],
            "retrieval_reason": "Tavily web search match",
        })

    return {
        "answer": answer,
        "evidence": evidence,
        "results_count": n_results,
        "status": status,
    }
