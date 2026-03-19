"""
Response synthesizer – merges evidence from agents, resolves overlap/conflict,
produces final answer with source list and route explanation.
"""

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


SYNTHESIS_PROMPT = """You are a research assistant. The user asked:

{query}

Below are answers and evidence from different knowledge sources. Each block is labeled by type: **paper** = arXiv papers, **web_docs** = web search, **notes** = user's ingested documents.

Rules:
- Produce a detailed, comprehensive final answer (not brief). Target roughly 5-8 paragraphs of substantive content.
- Cover the main points from sources that actually provide content: key findings, methods/approaches, trade-offs, and practical implications.
- If an agent's response is only that it does not have enough information (or similar), ignore that response and rely on the agents that did provide substantive content.
- Merge into one coherent answer, resolving overlap or conflict by prioritizing the most relevant and reliable evidence.
- Keep wording clear and non-repetitive, but do not over-compress.
- Formatting: write in Markdown with clear paragraphs and exactly one blank line between paragraphs.
- Paragraph quality: split by meaning, not by fixed length. Each paragraph should represent one coherent idea (e.g., one finding, one method, one implication, one limitation, one synthesis point).
- Start a new paragraph when you shift topic, evidence focus, or argument. Do not create mechanically equal-sized paragraphs.
- Prefer a readable flow: opening context paragraph -> evidence paragraphs -> implications/limitations -> conclusion.
- Source-grounded structure: for evidence paragraphs, use one source per paragraph whenever possible.
- In evidence paragraphs, anchor each paragraph to a single source index like [1], [2], [3] from the Source catalog.
- Avoid mixing multiple source indices in the same evidence paragraph unless absolutely necessary.
- If many sources exist, prioritize the top 4-6 most relevant sources and keep one paragraph per selected source.
- Structure: adapt structure to the query intent and to the agents that actually contributed evidence. Do not force fixed section titles when they do not match the available sources.
- If only one source type contributed, write a unified answer without artificial compare/contrast subtitles.
- Use section headers only when they naturally fit the query and evidence.
- Highlighting: do not bold arbitrary words. Use bold only for section headers and, if needed, exact key terms that appear verbatim in the user's query.
- Source referencing: use inline numeric citations like [1], [2] that point to the provided Source catalog below.
- Citation requirement: every evidence paragraph must include at least one numeric citation [n] from the Source catalog.
- Do not omit numeric citations in the body when sources exist.
- The Conclusion must be one full summary paragraph. Do not add "Sources used" or any source list in the Conclusion or at the end; sources are displayed separately in the interface.

Source responses:
{agent_outputs}

Source catalog (for citation numbering):
{source_catalog}

Provide the final synthesized answer now."""


ROUTE_EXPLANATION_PROMPT = """Given:
- User query: {query}
- Intent: {intent}
- Agents that were used: {agents_used}
- Retrieval summary: {retrieval_summary}

Write one short sentence (for the user) explaining which route was chosen and why. Example: "We used academic papers and web docs because you asked for a comparison between research and practical perspectives." Keep it under 30 words."""


def get_synth_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _format_outputs(agent_outputs: dict) -> str:
    return "\n\n".join(f"**{k}:**\n{v}" for k, v in agent_outputs.items())


# Human-readable labels for "Data sources used" in final answer
SOURCE_LABELS = {"paper": "Paper", "web_docs": "Web", "notes": "Notes"}


def _build_sources_list(agent_evidence: dict) -> list:
    """Build deduplicated list of {title, url, year, authors, origin} from all agents' evidence."""
    seen = set()
    out = []
    for agent_name, items in (agent_evidence or {}).items():
        origin = SOURCE_LABELS.get(agent_name, agent_name)
        for e in items:
            title = e.get("source_title") or "Unknown"
            url = e.get("source_url") or ""
            key = (title, url)
            if key not in seen:
                seen.add(key)
                out.append({
                    "title": title,
                    "url": url,
                    "year": e.get("source_year") or "",
                    "authors": e.get("source_authors") or "",
                    "origin": origin,
                })
    return out


def _data_sources_used(agent_outputs: dict) -> str:
    """Return comma-separated list of data source names (Paper, Web, Notes) that were used."""
    names = [SOURCE_LABELS.get(k, k) for k in (agent_outputs or {}).keys()]
    return ", ".join(names) if names else ""


def _build_source_catalog(sources_used: list) -> str:
    """Build numbered source catalog for inline [n] citations in final answer."""
    if not sources_used:
        return "No source catalog available."
    lines = []
    for s in sources_used:
        idx = s.get("index")
        title = s.get("title", "Unknown")
        origin = s.get("origin", "")
        year = s.get("year", "")
        authors = s.get("authors", "")
        meta_parts = []
        if authors:
            meta_parts.append(authors)
        if year:
            meta_parts.append(f"({year})")
        meta = " ".join(meta_parts).strip()
        extra = f" — {meta}" if meta else ""
        lines.append(f"[{idx}] {title} [{origin}]{extra}")
    return "\n".join(lines)


def _strip_sources_used_line(text: str) -> str:
    """Remove 'Sources used' and any following bulleted list from the end of the final answer (Conclusion); sources are shown separately in the UI."""
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    # Remove trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    # Remove "Sources used" line and any following lines that look like source list items (bullets, quoted titles)
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        lower = last.lower()
        if lower.startswith("**sources used:**") or lower.startswith("sources used:"):
            lines.pop()
            continue
        # Also remove lines that are clearly source-list items: bullet + optional quoted title
        if re.match(r"^[\-\*•]\s*", last) or re.match(r"^[\-\*•]\s*[\"'].*[\"']", last):
            lines.pop()
            continue
        if re.match(r"^\d+[\.\)]\s*[\"']?", last):  # numbered list item
            lines.pop()
            continue
        break
    text = "\n".join(lines).rstrip()
    # Remove "Sources used: ..." (and rest of line) when it appears at end of the last line
    text = re.sub(r"\s*\*\*Sources used:\*\*.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Sources used:.*$", "", text, flags=re.IGNORECASE)
    return text.rstrip()


def _ensure_numeric_citations(text: str, sources_used: list) -> str:
    """
    Ensure numeric [n] citations are present in evidence paragraphs when sources exist.
    If model omits citations entirely, append citations to body paragraphs deterministically.
    """
    if not text or not text.strip() or not sources_used:
        return text
    if re.search(r"\[\d+\]", text):
        return text

    lines = text.split("\n")
    paragraphs = []
    current = []
    for ln in lines:
        if ln.strip():
            current.append(ln.strip())
        else:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            paragraphs.append("")  # preserve blank separator
    if current:
        paragraphs.append(" ".join(current).strip())

    # Identify candidate body paragraphs (non-empty, not markdown headers/list items)
    body_idx = [
        i for i, p in enumerate(paragraphs)
        if p and not p.startswith("#") and not p.startswith("- ") and not p.startswith("* ")
    ]
    if not body_idx:
        return text

    # Skip first paragraph as intro and last paragraph as conclusion where possible.
    target_idx = body_idx[:]
    if len(target_idx) >= 3:
        target_idx = target_idx[1:-1]
    if not target_idx:
        target_idx = body_idx

    for j, i in enumerate(target_idx):
        n = (j % len(sources_used)) + 1
        p = paragraphs[i].rstrip()
        if not re.search(r"\[\d+\]", p):
            paragraphs[i] = f"{p} [{n}]"

    # Rebuild with one blank line between logical blocks.
    out = []
    prev_blank = False
    for p in paragraphs:
        if p == "":
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(p)
            prev_blank = False
    return "\n\n".join(part for part in "\n".join(out).split("\n\n") if part.strip()).strip()


def synthesize(
    query: str,
    agent_outputs: dict,
    agent_evidence: dict = None,
    intent: str = "",
    agents_used: list = None,
    retrieval_summary: dict = None,
    arxiv_results_count: int = None,
) -> dict:
    """
    Merge agent outputs into final answer. Returns dict with:
    - final_answer: str
    - sources_used: list of {title, url}
    - route_explanation: str
    """
    if not agent_outputs:
        return {
            "final_answer": "No agent responses available. Try adding documents or rephrasing your query.",
            "sources_used": [],
            "route_explanation": "No agents were run for this query.",
        }

    sources_used = _build_sources_list(agent_evidence or {})
    for i, s in enumerate(sources_used, 1):
        s["index"] = i
    source_catalog = _build_source_catalog(sources_used)

    llm = get_synth_llm()
    prompt = ChatPromptTemplate.from_messages([("human", SYNTHESIS_PROMPT)])
    chain = prompt | llm | StrOutputParser()
    final_answer = chain.invoke({
        "query": query,
        "agent_outputs": _format_outputs(agent_outputs),
        "source_catalog": source_catalog,
    })
    final_answer = _strip_sources_used_line(final_answer or "")
    final_answer = _ensure_numeric_citations(final_answer, sources_used)

    # Route explanation
    agents_used = agents_used or list(agent_outputs.keys())
    retrieval_summary = retrieval_summary or {}
    sum_parts = [f"{k}: {v}" for k, v in retrieval_summary.items()]
    try:
        route_prompt = ChatPromptTemplate.from_messages([("human", ROUTE_EXPLANATION_PROMPT)])
        route_chain = route_prompt | llm | StrOutputParser()
        route_explanation = route_chain.invoke({
            "query": query,
            "intent": intent,
            "agents_used": ", ".join(agents_used),
            "retrieval_summary": "; ".join(sum_parts) if sum_parts else "N/A",
        })
    except Exception:
        route_explanation = f"Route: {', '.join(agents_used)} (intent: {intent or 'unknown'})."

    # When paper had no results but web/notes were used, state it clearly
    fallback_note = ""
    if (
        arxiv_results_count is not None
        and arxiv_results_count < 1
        and "paper" in (agent_outputs or {})
        and len(agent_outputs or {}) > 1
    ):
        fallback_note = "No papers were available to answer this query; web search and/or notes were used instead. "
    if fallback_note:
        route_explanation = fallback_note + route_explanation

    data_sources_used = _data_sources_used(agent_outputs)

    return {
        "final_answer": final_answer,
        "sources_used": sources_used,
        "route_explanation": route_explanation,
        "data_sources_used": data_sources_used,
    }
