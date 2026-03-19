"""
paper_scoring.py

Advanced paper scoring, filtering, reranking, and confidence estimation
for the arXiv-only paper agent.

Goals:
- work for arbitrary user queries, not just one demo query
- separate task terms from topic terms
- score papers using multiple signals
- filter weak matches
- rerank top candidates with an LLM
- promote diversity among selected papers
- expose score breakdown and confidence for routing / UI trace
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

TOP_K_INITIAL = 12          # candidates kept after heuristic scoring
TOP_K_LLM = 6               # candidates sent to LLM reranker
TOP_K_FINAL = 5             # final selected papers

MIN_BASE_SCORE = 0.28       # discard obviously weak papers
MIN_FINAL_SCORE = 0.40      # keep only sufficiently relevant final papers

TITLE_WEIGHT = 0.28
ABSTRACT_WEIGHT = 0.24
SEMANTIC_WEIGHT = 0.28
PHRASE_WEIGHT = 0.10
INTENT_WEIGHT = 0.05
RECENCY_WEIGHT = 0.05

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


# -------------------------------------------------------------------
# Query analysis vocabulary
# -------------------------------------------------------------------

GENERAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "if", "in", "into", "is", "it", "its", "may", "might", "of", "on",
    "or", "our", "should", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your", "than", "then", "about",
    "using", "use", "used"
}

TASK_TERMS = {
    "compare", "comparison", "explain", "summarize", "summary", "describe",
    "tutorial", "guide", "academic", "practical", "contextual", "research",
    "papers", "paper", "studies", "study", "evidence", "survey", "review"
}

PROTECTED_TERMS = {
    "rag", "retrieval", "reranking", "ranking", "re-ranking", "llm", "llms",
    "gpt", "bert", "transformer", "transformers", "multimodal", "embedding",
    "embeddings", "vector", "vectors", "faiss", "arxiv", "cross-encoder",
    "bi-encoder", "dense", "sparse", "hybrid", "retriever", "retrievers",
    "query", "queries", "document", "documents", "search", "retrieval-augmented",
    "lora", "qlora", "attention", "fine-tuning", "finetuning", "agent", "agents"
}

ACADEMIC_CUE_TERMS = {
    "benchmark", "benchmarks", "dataset", "datasets", "evaluation", "evaluations",
    "method", "methods", "architecture", "architectures", "model", "models",
    "experiment", "experiments", "metric", "metrics", "analysis", "theory",
    "theoretical", "empirical", "ablation", "retrieval", "ranking", "reranking"
}

GENERIC_NOISE_TERMS = {
    "best", "practices", "workflow", "organization", "education", "teaching",
    "healthcare", "psychology", "ethics", "leadership", "software registries",
    "repositories"
}


# -------------------------------------------------------------------
# Model singletons
# -------------------------------------------------------------------

_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


# -------------------------------------------------------------------
# Basic text helpers
# -------------------------------------------------------------------

def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9\-]+\b", _clean_text(text))


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # map from [-1, 1] to [0, 1]
    return _clip01((dot / (norm_a * norm_b) + 1.0) / 2.0)


def _extract_phrases(tokens: List[str]) -> List[str]:
    phrases: List[str] = []
    for i in range(len(tokens) - 1):
        phrases.append(f"{tokens[i]} {tokens[i + 1]}")
    for i in range(len(tokens) - 2):
        phrases.append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")
    return _dedupe_keep_order(phrases)


def _token_overlap_score(query_terms: List[str], text: str) -> float:
    if not query_terms:
        return 0.0
    tokens = set(_tokenize(text))
    if not tokens:
        return 0.0
    matched = sum(1 for term in query_terms if term in tokens)
    return matched / max(len(query_terms), 1)


def _phrase_overlap_score(phrases: List[str], text: str) -> float:
    if not phrases:
        return 0.0
    clean = _clean_text(text)
    matched = sum(1 for phrase in phrases if phrase in clean)
    return matched / max(len(phrases), 1)


def _jaccard_text_similarity(text_a: str, text_b: str) -> float:
    set_a = set(_tokenize(text_a))
    set_b = set(_tokenize(text_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def _extract_year(paper: Dict[str, Any]) -> int | None:
    raw = str(paper.get("year", "")).strip()
    match = re.search(r"(19|20)\d{2}", raw)
    return int(match.group(0)) if match else None


# -------------------------------------------------------------------
# Query analysis
# -------------------------------------------------------------------

def extract_query_terms(query: str) -> Dict[str, Any]:
    """
    Query-agnostic analysis.

    Returns:
    - clean_query
    - tokens
    - task_terms: intent/task words
    - keywords: topic-heavy terms for matching
    - phrases: important 2-gram/3-gram phrases
    - academic_focus: whether the query is strongly academic
    """
    clean_query = _clean_text(query)
    tokens = _tokenize(clean_query)

    task_terms = [tok for tok in tokens if tok in TASK_TERMS]

    keywords: List[str] = []
    for token in tokens:
        if token in PROTECTED_TERMS:
            keywords.append(token)
        elif token not in GENERAL_STOPWORDS and token not in TASK_TERMS and len(token) > 2:
            keywords.append(token)

    keywords = _dedupe_keep_order(keywords)

    all_phrases = _extract_phrases(tokens)
    important_phrases = [
        phrase for phrase in all_phrases
        if any(term in phrase for term in PROTECTED_TERMS) or
           all(part not in GENERAL_STOPWORDS for part in phrase.split())
    ]
    important_phrases = _dedupe_keep_order(important_phrases)[:12]

    academic_focus = any(
        tok in TASK_TERMS or tok in ACADEMIC_CUE_TERMS or tok == "arxiv"
        for tok in tokens
    )

    return {
        "raw": query,
        "clean_query": clean_query,
        "tokens": tokens,
        "task_terms": _dedupe_keep_order(task_terms),
        "keywords": keywords,
        "phrases": important_phrases,
        "academic_focus": academic_focus,
    }


# -------------------------------------------------------------------
# Heuristic scoring
# -------------------------------------------------------------------

def _intent_alignment_score(query_info: Dict[str, Any], paper: Dict[str, Any]) -> float:
    """
    Light bonus if the paper looks method / evaluation / research oriented.
    """
    title = _clean_text(paper.get("title", ""))
    content = _clean_text(paper.get("content", ""))
    combined = f"{title} {content}"

    academic_hits = sum(1 for term in ACADEMIC_CUE_TERMS if term in combined)
    base = min(academic_hits / 6.0, 1.0)

    if query_info.get("academic_focus"):
        return _clip01(0.5 + 0.5 * base)
    return base


def _recency_bonus(paper: Dict[str, Any]) -> float:
    """
    Mild recency preference, without overpowering relevance.
    """
    year = _extract_year(paper)
    if year is None:
        return 0.1
    if year >= 2024:
        return 1.0
    if year >= 2022:
        return 0.7
    if year >= 2019:
        return 0.4
    return 0.1


def _topic_mismatch_penalty(query_info: Dict[str, Any], paper: Dict[str, Any]) -> float:
    """
    Penalize generic best-practice papers from unrelated domains.
    """
    title = _clean_text(paper.get("title", ""))
    content = _clean_text((paper.get("content", "") or "")[:1200])
    combined = f"{title} {content}"

    topic_keywords = set(query_info.get("keywords", []))
    has_core_ir_topic = any(
        term in combined
        for term in {"retrieval", "reranking", "ranking", "search", "document", "query"}
    )
    generic_noise_hits = sum(1 for term in GENERIC_NOISE_TERMS if term in combined)

    if topic_keywords and not has_core_ir_topic and generic_noise_hits >= 2:
        return 0.18
    if generic_noise_hits >= 3:
        return 0.10
    return 0.0


def _semantic_similarity_score(query: str, paper: Dict[str, Any]) -> float:
    """
    Embedding similarity on title + first part of abstract/content.
    """
    title = paper.get("title", "") or ""
    content = (paper.get("content", "") or "")[:1400]
    paper_text = f"{title}. {content}".strip()
    if not paper_text:
        return 0.0

    try:
        q_emb = _embeddings.embed_query(query)
        p_emb = _embeddings.embed_query(paper_text)
        return _cosine_similarity(q_emb, p_emb)
    except Exception:
        return 0.0


def score_paper(query_info: Dict[str, Any], paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score one paper with interpretable sub-scores.
    """
    title = paper.get("title", "") or ""
    content = paper.get("content", "") or ""

    title_overlap = _token_overlap_score(query_info["keywords"], title)
    abstract_overlap = _token_overlap_score(query_info["keywords"], content)
    phrase_overlap = _phrase_overlap_score(query_info["phrases"], f"{title} {content}")
    semantic_similarity = _semantic_similarity_score(query_info["raw"], paper)
    intent_alignment = _intent_alignment_score(query_info, paper)
    recency_bonus = _recency_bonus(paper)
    mismatch_penalty = _topic_mismatch_penalty(query_info, paper)

    base_score = (
        TITLE_WEIGHT * title_overlap +
        ABSTRACT_WEIGHT * abstract_overlap +
        SEMANTIC_WEIGHT * semantic_similarity +
        PHRASE_WEIGHT * phrase_overlap +
        INTENT_WEIGHT * intent_alignment +
        RECENCY_WEIGHT * recency_bonus
        - mismatch_penalty
    )
    base_score = _clip01(base_score)

    retrieval_reason_parts: List[str] = []
    if semantic_similarity >= 0.72:
        retrieval_reason_parts.append("strong semantic match")
    elif semantic_similarity >= 0.58:
        retrieval_reason_parts.append("good semantic match")

    if title_overlap >= 0.5:
        retrieval_reason_parts.append("title matches key query terms")
    if abstract_overlap >= 0.5:
        retrieval_reason_parts.append("abstract matches topic terms")
    if phrase_overlap > 0:
        retrieval_reason_parts.append("matches important query phrase")
    if intent_alignment >= 0.65:
        retrieval_reason_parts.append("appears method/evaluation oriented")
    if mismatch_penalty > 0:
        retrieval_reason_parts.append("penalized for generic/off-topic wording")

    retrieval_reason = "; ".join(retrieval_reason_parts) if retrieval_reason_parts else "initial arXiv match"

    scored = dict(paper)
    scored["base_score"] = round(base_score, 4)
    scored["score"] = round(base_score, 4)
    scored["score_breakdown"] = {
        "title_overlap": round(title_overlap, 4),
        "abstract_overlap": round(abstract_overlap, 4),
        "phrase_overlap": round(phrase_overlap, 4),
        "semantic_similarity": round(semantic_similarity, 4),
        "intent_alignment": round(intent_alignment, 4),
        "recency_bonus": round(recency_bonus, 4),
        "mismatch_penalty": round(mismatch_penalty, 4),
    }
    scored["retrieval_reason"] = retrieval_reason
    return scored


# -------------------------------------------------------------------
# LLM reranking
# -------------------------------------------------------------------

def _llm_rerank_one(query: str, paper: Dict[str, Any]) -> Tuple[float, str]:
    """
    Ask the LLM for a relevance score and reason.

    Expected format:
    SCORE: 0.82
    REASON: concise explanation
    """
    title = paper.get("title", "") or ""
    abstract = (paper.get("content", "") or "")[:1800]

    prompt = f"""
You are evaluating whether an arXiv paper is relevant to a user's academic query.

User query:
{query}

Paper title:
{title}

Paper abstract/content snippet:
{abstract}

Instructions:
- Judge topical relevance to the query, not general quality.
- Favor papers directly about the queried technical topic.
- Penalize papers that only match generic words such as "best practices" but are from unrelated domains.
- Return exactly two lines in this format:

SCORE: <number from 0.00 to 1.00>
REASON: <one concise sentence>
""".strip()

    try:
        response = _llm.invoke(prompt)
        text = getattr(response, "content", str(response)).strip()

        score_match = re.search(r"SCORE:\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

        score = _safe_float(score_match.group(1), default=0.5) if score_match else 0.5
        score = _clip01(score)
        reason = reason_match.group(1).strip() if reason_match else "LLM judged this paper relevant."
        return score, reason

    except Exception:
        return paper.get("base_score", 0.5), "LLM rerank unavailable; fell back to heuristic score."


def llm_rerank(query: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank the top papers with an LLM and combine with heuristic score.
    """
    reranked: List[Dict[str, Any]] = []

    for paper in papers[:TOP_K_LLM]:
        llm_score, llm_reason = _llm_rerank_one(query, paper)

        final_score = _clip01(
            0.55 * _safe_float(paper.get("base_score", 0.0)) +
            0.45 * llm_score
        )

        item = dict(paper)
        item["llm_score"] = round(llm_score, 4)
        item["llm_reason"] = llm_reason
        item["final_score"] = round(final_score, 4)
        reranked.append(item)

    # keep any remaining papers without LLM pass, slightly downweighted
    for paper in papers[TOP_K_LLM:]:
        item = dict(paper)
        item["llm_score"] = None
        item["llm_reason"] = "Not sent to LLM reranker."
        item["final_score"] = round(_safe_float(paper.get("base_score", 0.0)) * 0.95, 4)
        reranked.append(item)

    reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return reranked


# -------------------------------------------------------------------
# Diversity selection
# -------------------------------------------------------------------

def _select_diverse_top_k(papers: List[Dict[str, Any]], k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    """
    Avoid selecting near-duplicate papers with very similar title/abstract wording.
    """
    selected: List[Dict[str, Any]] = []

    for paper in papers:
        if len(selected) >= k:
            break

        title = paper.get("title", "") or ""
        content = (paper.get("content", "") or "")[:1000]
        paper_text = f"{title} {content}"

        too_similar = False
        for chosen in selected:
            chosen_text = f"{chosen.get('title', '')} {(chosen.get('content', '') or '')[:1000]}"
            title_sim = _jaccard_text_similarity(title, chosen.get("title", "") or "")
            text_sim = _jaccard_text_similarity(paper_text, chosen_text)

            if title_sim >= 0.72 or text_sim >= 0.82:
                too_similar = True
                break

        if not too_similar:
            selected.append(paper)

    return selected


# -------------------------------------------------------------------
# Main filtering pipeline
# -------------------------------------------------------------------

def filter_and_rerank_papers(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Full paper selection pipeline:
    1. analyze query
    2. heuristic score all arXiv results
    3. remove weak candidates
    4. keep top initial candidates
    5. LLM rerank top candidates
    6. diversity selection
    7. final thresholding
    """
    if not results:
        return []

    query_info = extract_query_terms(query)

    scored = [score_paper(query_info, paper) for paper in results]
    scored.sort(key=lambda x: x.get("base_score", 0.0), reverse=True)

    filtered = [paper for paper in scored if paper.get("base_score", 0.0) >= MIN_BASE_SCORE]
    if not filtered:
        filtered = scored[: min(3, len(scored))]

    initial = filtered[:TOP_K_INITIAL]
    reranked = llm_rerank(query, initial)

    diverse = _select_diverse_top_k(reranked, k=TOP_K_FINAL)

    final_selected = [
        paper for paper in diverse
        if paper.get("final_score", paper.get("base_score", 0.0)) >= MIN_FINAL_SCORE
    ]

    # Do not force weak papers into the final set; return empty to trigger honest fallback.
    if not final_selected:
        return []

    return final_selected


# -------------------------------------------------------------------
# Confidence
# -------------------------------------------------------------------

def compute_paper_confidence(selected: List[Dict[str, Any]], query_info: Dict[str, Any] | None = None) -> float:
    """
    Confidence for router fallback logic.

    Factors:
    - strongest paper score
    - average selected score
    - count of strong papers
    - light coverage bonus if selected papers seem aligned to query terms
    """
    if not selected:
        return 0.0

    scores = [
        _safe_float(p.get("final_score", p.get("base_score", 0.0)))
        for p in selected
    ]
    top_score = max(scores)
    avg_score = sum(scores) / len(scores)
    strong_count = sum(1 for s in scores if s >= 0.7)
    strong_count_score = min(strong_count / 3.0, 1.0)

    coverage = 0.6
    if query_info:
        query_terms = query_info.get("keywords", [])
        if query_terms:
            coverages: List[float] = []
            for paper in selected:
                combined = f"{paper.get('title', '')} {paper.get('content', '')}"
                coverages.append(_token_overlap_score(query_terms, combined))
            coverage = sum(coverages) / max(len(coverages), 1)

    confidence = (
        0.40 * top_score +
        0.30 * avg_score +
        0.20 * strong_count_score +
        0.10 * coverage
    )
    return round(_clip01(confidence), 4)


# -------------------------------------------------------------------
# Optional convenience helper
# -------------------------------------------------------------------

def analyze_and_select_papers(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience wrapper if the caller wants both query analysis and selected papers.
    """
    query_info = extract_query_terms(query)
    selected = filter_and_rerank_papers(query, results)
    confidence = compute_paper_confidence(selected, query_info=query_info)

    return {
        "query_info": query_info,
        "selected": selected,
        "confidence": confidence,
        "selected_count": len(selected),
    }