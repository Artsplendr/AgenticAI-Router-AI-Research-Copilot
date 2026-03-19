"""
Tavily search – live web search for the Web Docs agent.
Uses TAVILY_API_KEY from env; no-op if key is missing.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> list[dict]:
    """
    Run Tavily web search. Returns list of result dicts with 'title', 'url', 'content'.
    Returns [] if TAVILY_API_KEY is not set.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )
        raw = response.get("results", []) if isinstance(response, dict) else (getattr(response, "results", []) or [])
        out = []
        for r in raw:
            if isinstance(r, dict):
                out.append({"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")})
            elif hasattr(r, "title"):
                out.append({"title": getattr(r, "title", ""), "url": getattr(r, "url", ""), "content": getattr(r, "content", "")})
        return out
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return []


def _expand_query_for_search(query: str) -> str:
    """Expand comparison-style queries so Tavily returns more targeted results."""
    q = query.strip().lower()
    if any(x in q for x in ("compare", "vs", "versus", "difference between")):
        if "comparison" not in q and "differences" not in q:
            return query + " comparison differences advantages"
    return query


def search_as_context_string(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
    expand_query: bool = True,
) -> str:
    """Run Tavily search and return context string. See search_with_status for status/count."""
    context, _, _, _ = search_with_status(query, max_results=max_results, search_depth=search_depth, expand_query=expand_query)
    return context


def search_with_status(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
    expand_query: bool = True,
) -> tuple[str, int, str, list]:
    """
    Run Tavily search and return (context_string, num_results, status, results_list).
    status: "ok" | "no_key" | "error". results_list: list of {title, url, content} for evidence.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "", 0, "no_key", []
    search_query = _expand_query_for_search(query) if expand_query else query
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=search_query, max_results=max_results, search_depth=search_depth)
        # Handle both object and dict response (SDK may return either)
        raw = response.get("results", []) if isinstance(response, dict) else (getattr(response, "results", []) or [])
        results = []
        for r in raw:
            if isinstance(r, dict):
                results.append({"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")})
            elif hasattr(r, "title"):
                results.append({"title": getattr(r, "title", ""), "url": getattr(r, "url", ""), "content": getattr(r, "content", "")})
        # If advanced returned 0 results, retry with basic (sometimes more reliable or different quota)
        if not results and search_depth == "advanced":
            response = client.search(query=search_query, max_results=max_results, search_depth="basic")
            raw = response.get("results", []) if isinstance(response, dict) else (getattr(response, "results", []) or [])
            for r in raw:
                if isinstance(r, dict):
                    results.append({"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")})
                elif hasattr(r, "title"):
                    results.append({"title": getattr(r, "title", ""), "url": getattr(r, "url", ""), "content": getattr(r, "content", "")})
        if not results:
            logger.info("Tavily search returned 0 results for query=%r", search_query[:80])
            # One-off debug: inspect what the client returned so we can fix parsing if needed
            resp_type = type(response).__name__
            raw_len = len(raw)
            first_info = ""
            if raw:
                r0 = raw[0]
                first_info = " first_result_type=%r" % type(r0).__name__
                if isinstance(r0, dict):
                    first_info += " keys=%r" % (list(r0.keys()),)
                else:
                    first_info += " attrs=%r" % ([a for a in dir(r0) if not a.startswith("_")],)
            logger.info(
                "Tavily 0-results debug: response_type=%r response_is_dict=%s len(raw)=%s%s",
                resp_type, isinstance(response, dict), raw_len, first_info,
            )
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return "", 0, "error", []
    if not results:
        return "", 0, "ok", []
    parts = [f"[{r.get('title', '')}]({r.get('url', '')})\n{r.get('content', '')}" for r in results]
    return "\n\n---\n\n".join(parts), len(results), "ok", results
