"""
arXiv search – fetch research paper metadata (title, abstract, URL) from https://arxiv.org.
Used by the Paper agent as the primary source; no API key required.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "to", "what",
    "when", "where", "which", "with", "compare", "comparison", "academic",
    "practical", "perspective", "perspectives", "explain", "summarize", "review",
    "do", "does", "did", "say", "says", "paper", "papers", "arxiv", "about",
}


def _extract_core_terms(query: str, max_terms: int = 6) -> List[str]:
    """Extract core topical terms from query for tighter arXiv search."""
    tokens = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9\-]+\b", (query or "").lower())
    terms = []
    seen = set()
    for tok in tokens:
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
        if len(terms) >= max_terms:
            break
    return terms


def _build_arxiv_query(query: str) -> str:
    """
    Build a stricter arXiv query.
    Prefer title/abstract constraints for core terms, fallback to raw query.
    """
    raw = (query or "").strip()
    if not raw:
        return "machine learning"
    core = _extract_core_terms(raw)
    lower = raw.lower()
    # Preserve strong domain phrases for retrieval quality.
    if "retrieval augmentation" in lower and "retrieval-augmented" not in core:
        core.insert(0, "retrieval-augmented")
    if "rag" in lower and "rag" not in core:
        core.insert(0, "rag")
    if len(core) < 2:
        return raw
    # Require strongest terms to appear in title/abstract for precision.
    parts = [f'(ti:"{t}" OR abs:"{t}")' for t in core[:5]]
    return " AND ".join(parts)


def search_with_status(
    query: str,
    max_results: int = 10,
) -> Tuple[str, int, str, List[dict]]:
    """
    Search arXiv and return (context_string, num_results, status, results_list).
    status: "ok" | "error". results_list: list of {title, url, content} for evidence.
    """
    try:
        import arxiv
    except ImportError:
        logger.warning("arxiv package not installed; run pip install arxiv")
        return "", 0, "error", []

    search_query = _build_arxiv_query(query)
    try:
        # arxiv package: Search then iterate (0.x/1.x: search.results(); 2.x: Client().results(search))
        search = arxiv.Search(query=search_query, max_results=max_results)
        if hasattr(arxiv, "Client"):
            raw = list(arxiv.Client().results(search))
        else:
            raw = list(search.results())
    except Exception as e:
        logger.warning("arXiv search failed: %s", e)
        return "", 0, "error", []

    results = []
    for r in raw:
        title = getattr(r, "title", None) or ""
        summary = getattr(r, "summary", None) or ""
        url = getattr(r, "entry_id", None) or getattr(r, "pdf_url", None) or ""
        if isinstance(summary, str) and "\n" in summary:
            summary = summary.replace("\n", " ")
        authors_list = getattr(r, "authors", None) or []
        try:
            authors_str = ", ".join(getattr(a, "name", str(a)) for a in authors_list[:5])
            if authors_list and len(authors_list) > 5:
                authors_str += " et al."
        except Exception:
            authors_str = ""
        published = getattr(r, "published", None)
        try:
            year = str(published.year) if published else ""
        except Exception:
            year = ""
        results.append({
            "title": title,
            "url": url,
            "content": summary,
            "authors": authors_str,
            "year": year,
        })

    if not results:
        logger.info("arXiv search returned 0 results for query=%r", search_query[:80])
        return "", 0, "ok", []

    # Build context string for the LLM (title, link, abstract)
    parts = [f"[{r['title']}]({r['url']})\n{r['content']}" for r in results]
    context = "\n\n---\n\n".join(parts)
    return context, len(results), "ok", results
