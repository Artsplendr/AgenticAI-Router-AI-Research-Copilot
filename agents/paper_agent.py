"""
Paper Agent – arXiv-only academic specialist with relevance scoring and reranking.
Returns unified AgentResult (answer, evidence, results_count, status) plus paper quality metadata.
"""

from langchain_core.retrievers import BaseRetriever

from agents.base_agent import run_rag_with_context
from schemas.evidence import AgentResult
from tools.arxiv_search import search_with_status
from agents.paper_scoring import analyze_and_select_papers


def _build_context(selected_papers):
    """Build compact high-signal context from selected papers only."""
    parts = []
    for i, p in enumerate(selected_papers, 1):
        parts.append(
            f"""Paper {i}
Title: {p.get("title")}
Year: {p.get("year")}
URL: {p.get("url")}
Relevance score: {round(float(p.get("final_score", p.get("base_score", 0.0))), 3)}

Key abstract/content:
{(p.get("content") or "")[:1200]}"""
        )
    return "\n\n---\n\n".join(parts)


def run(query: str, retriever: BaseRetriever) -> AgentResult:
    """Answer using arXiv candidates, then relevance scoring/reranking for high-precision paper evidence."""
    status = "error"
    n_candidates = 0
    results_list = []

    # Pull broader candidate pool, then filter/rerank for precision.
    _, n_candidates, status, results_list = search_with_status(query, max_results=25)
    analysis = analyze_and_select_papers(query, results_list)
    selected = analysis.get("selected", []) or []
    paper_quality = float(analysis.get("confidence", 0.0) or 0.0)

    combined_context = _build_context(selected) if selected else "(No relevant academic papers found.)"
    answer = run_rag_with_context(query, combined_context, allow_supplement=False)

    evidence = []
    for p in selected:
        score = float(p.get("final_score", p.get("base_score", 0.0)) or 0.0)
        retrieval_reason = p.get("llm_reason") or p.get("retrieval_reason") or "arXiv match"
        evidence.append({
            "source_type": "paper",
            "source_title": p.get("title", ""),
            "source_url": p.get("url", ""),
            "source_authors": p.get("authors", ""),
            "source_year": p.get("year", ""),
            "extracted_evidence": (p.get("content", "") or "")[:2000],
            "retrieval_reason": f"{retrieval_reason} (relevance={score:.2f})",
            "relevance_score": round(score, 4),
            "score_breakdown": p.get("score_breakdown", {}),
        })

    retrieval_trace = {
        "candidates_count": n_candidates,
        "selected_count": len(selected),
        "paper_quality": round(paper_quality, 4),
        "top_score": max([float(p.get("final_score", 0.0) or 0.0) for p in selected], default=0.0),
        "avg_score": (
            sum(float(p.get("final_score", 0.0) or 0.0) for p in selected) / len(selected)
            if selected else 0.0
        ),
    }

    return {
        "answer": answer,
        # results_count now reflects selected (kept) papers, not raw candidates.
        "results_count": len(selected),
        "status": status,
        "evidence": evidence,
        # Extra metadata used by router/UI for quality-based fallback and transparency.
        "candidates_count": n_candidates,
        "selected_count": len(selected),
        "paper_quality": paper_quality,
        "retrieval_trace": retrieval_trace,
    }
