"""
Evidence schema – common structure for agent outputs before synthesis.
Used for normalizer/synthesizer and source display in UI.

All agents return the same AgentResult contract so the router and synthesizer
never handle special cases.
"""

from typing import List, TypedDict


class EvidenceItem(TypedDict, total=False):
    source_type: str   # "paper" | "web_docs" | "notes"
    source_title: str
    source_url: str
    source_authors: str
    source_year: str
    extracted_evidence: str
    retrieval_reason: str


class AgentResult(TypedDict):
    """Unified contract for every agent. Router and synthesizer always consume this shape."""
    answer: str
    evidence: List[dict]
    results_count: int
    status: str       # "ok" | "no_key" | "error"
