"""
Preprocessor – runs before intent classification to improve routing accuracy.
Performs: query cleanup, optional type hints, keyword extraction, ambiguity detection.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class PreprocessorResult:
    """Result of preprocessing the user query."""
    cleaned_query: str
    type_hints: List[str]  # e.g. ["explain", "tutorial"]
    keywords: List[str]  # significant terms for retrieval
    is_ambiguous: bool


# Phrases that suggest query type (for type_hints)
TYPE_PATTERNS = {
    "explain": re.compile(r"\b(explain|what is|how does|define|meaning of)\b", re.I),
    "tutorial": re.compile(r"\b(tutorial|how to|guide|step-by-step|implementation)\b", re.I),
    "compare": re.compile(r"\b(compare|vs|versus|difference between|contrast)\b", re.I),
    "academic": re.compile(r"\b(papers?|arxiv|research|academic|evidence|study|studies)\b", re.I),
    "practical": re.compile(r"\b(practical|tutorial|best practice|in practice|real-world)\b", re.I),
    "contextual": re.compile(r"\b(my notes|my docs|stored|saved|what did i|relate to my)\b", re.I),
}

# Short or vague queries are ambiguous
MIN_LENGTH_FOR_CLEAR = 10
AMBIGUOUS_PATTERNS = re.compile(r"^\s*(it|this|that|something)\s*$", re.I)


def _extract_keywords(query: str, max_keywords: int = 8) -> List[str]:
    """Extract likely significant terms (skip stopwords)."""
    stop = {"the", "a", "an", "is", "are", "what", "how", "why", "when", "do", "does", "say", "say about"}
    words = re.findall(r"\b[a-zA-Z0-9]+\b", query)
    seen = set()
    out = []
    for w in words:
        wl = w.lower()
        if wl in stop or len(wl) < 2 or wl in seen:
            continue
        seen.add(wl)
        out.append(w)
        if len(out) >= max_keywords:
            break
    return out


def preprocess(query: str) -> PreprocessorResult:
    """
    Clean query and compute type hints, keywords, and ambiguity flag.
    Use cleaned_query for downstream classification.
    """
    cleaned = " ".join(query.strip().split())
    type_hints = []
    for hint, pat in TYPE_PATTERNS.items():
        if pat.search(cleaned):
            type_hints.append(hint)
    keywords = _extract_keywords(cleaned)
    is_ambiguous = (
        len(cleaned) < MIN_LENGTH_FOR_CLEAR
        or bool(AMBIGUOUS_PATTERNS.match(cleaned))
    )
    return PreprocessorResult(
        cleaned_query=cleaned,
        type_hints=type_hints,
        keywords=keywords,
        is_ambiguous=is_ambiguous,
    )
