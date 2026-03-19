"""
Intent classifier – source-aware routing labels.
Intents: academic → paper | practical → web_docs | contextual → notes | comparative → planner | general → web_docs (with fallback).
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import json

_PROMPT_HEAD = """Classify the user's research query into exactly one intent. Use source-oriented labels.

Intents:
- academic: research-heavy, evidence from papers, arXiv, "what do papers say", academic comparison
- practical: explain simply, tutorial, how-to, best practice, implementation, web docs and tutorials
- contextual: personal or project context, "my notes", "what did I store", saved docs
- comparative: compare multiple perspectives, sources, or domains (e.g. academic vs practical, papers vs tutorials)
- general: broad or unclear; prefer practical (web_docs) first

User query: """
_PROMPT_TAIL = """

Reply with a single JSON object with one key "intent" and value exactly one of: academic, practical, contextual, comparative, general. No other text."""


def get_classifier_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def classify_intent(query: str) -> str:
    """Return one of: academic, practical, contextual, comparative, general."""
    llm = get_classifier_llm()
    text = _PROMPT_HEAD + query + _PROMPT_TAIL
    chain = llm | StrOutputParser()
    raw = chain.invoke([HumanMessage(content=text)]).strip()
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()
    try:
        out = json.loads(raw)
        return out.get("intent", "general")
    except json.JSONDecodeError:
        return "general"


def plan_agents_for_comparative(query: str) -> list:
    """
    Rule-based planner for comparative intent. Conservative: do not over-select.
    - Default: exactly 2 agents (paper + web_docs) for vague "compare X and Y" questions.
    - Add notes only when the user explicitly asks for their stored/notes context.
    Returns list of agent names, length 2 or 3.
    """
    q = query.lower().strip()
    notes_trigger_phrases = (
        "my notes", "my note", "what do my notes", "stored about", "saved about",
        "my docs", "my documents", "relate to my", "relate to notes", "and my notes",
        "whether my notes", "notes align", "notes say", "my stored", "saved notes",
    )
    add_notes = any(phrase in q for phrase in notes_trigger_phrases)
    agents = ["paper", "web_docs"]
    if add_notes:
        agents.append("notes")
    return agents
