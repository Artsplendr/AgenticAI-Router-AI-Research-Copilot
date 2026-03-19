"""
Router graph – LangGraph orchestration with preprocessor, source-aware intents, planner, and results-based fallback.
All agents return the same AgentResult contract; the router and synthesizer never handle special cases.
"""

from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END

from router.preprocessor import preprocess
from router.intent_classifier import classify_intent, plan_agents_for_comparative
from tools.vector_store import get_or_create_vector_store, get_retriever
from agents.paper_agent import run as run_paper
from agents.web_docs_agent import run as run_web_docs
from agents.notes_agent import run as run_notes
from synthesizer.response_synthesizer import synthesize


Intent = Literal["academic", "practical", "contextual", "comparative", "general"]


class RouterState(TypedDict):
    query: str
    cleaned_query: str
    type_hints: list
    is_ambiguous: bool
    intent: str
    planner_decision: list
    agent_outputs: dict
    agent_evidence: dict
    final_answer: str
    sources_used: list
    route_explanation: str
    data_sources_used: str
    tavily_status: str
    tavily_results_count: int
    arxiv_status: str
    arxiv_results_count: int
    arxiv_candidates_count: int
    arxiv_quality: float
    retrieval_summary: dict


# Unified contract: every agent returns AgentResult. Labels for retrieval_summary only.
RETRIEVAL_LABELS = {"paper": "arXiv source(s)", "web_docs": "Tavily result(s)", "notes": "note chunk(s)"}


def _apply_agent_result(
    state: RouterState,
    agent_name: str,
    out: dict,
    *,
    set_arxiv: bool = False,
    set_tavily: bool = False,
) -> RouterState:
    """Merge one AgentResult into state. Single code path for all agents."""
    answer = out.get("answer", "")
    evidence = out.get("evidence", [])
    results_count = out.get("results_count", 0)
    status = out.get("status", "ok")
    label = RETRIEVAL_LABELS.get(agent_name, "source(s)")
    updates = {
        **state,
        "agent_outputs": {**state.get("agent_outputs", {}), agent_name: answer},
        "agent_evidence": {**state.get("agent_evidence", {}), agent_name: evidence},
        "retrieval_summary": {**state.get("retrieval_summary", {}), agent_name: f"{results_count} {label}"},
    }
    if set_arxiv:
        updates["arxiv_status"] = status
        updates["arxiv_results_count"] = results_count
        updates["arxiv_candidates_count"] = int(out.get("candidates_count", results_count) or 0)
        updates["arxiv_quality"] = float(out.get("paper_quality", 0.0) or 0.0)
        # Make retrieval summary for paper explicit: kept N of M
        updates["retrieval_summary"]["paper"] = (
            f"{results_count} selected arXiv source(s) "
            f"(from {updates['arxiv_candidates_count']} candidates)"
        )
    if set_tavily:
        updates["tavily_status"] = status
        updates["tavily_results_count"] = results_count
    return updates


def _get_store(source: str):
    return get_or_create_vector_store(source)


def _route_after_classify(state: RouterState) -> str:
    """Route by intent: single agent or planner for comparative."""
    intent = state["intent"]
    if intent == "comparative":
        return "planner"
    if intent == "academic":
        return "paper_agent"
    if intent == "practical":
        return "web_docs_agent"
    if intent == "contextual":
        return "notes_agent"
    return "web_docs_agent"


def node_preprocess(state: RouterState) -> RouterState:
    """Preprocessor: cleanup, type hints, ambiguity."""
    prepped = preprocess(state["query"])
    return {
        **state,
        "cleaned_query": prepped.cleaned_query,
        "type_hints": prepped.type_hints,
        "is_ambiguous": prepped.is_ambiguous,
    }


def node_classify(state: RouterState) -> RouterState:
    """Classify intent from cleaned query."""
    intent = classify_intent(state.get("cleaned_query") or state["query"])
    return {**state, "intent": intent}


def node_planner(state: RouterState) -> RouterState:
    """For comparative intent: decide which agents to run (1–3 sources)."""
    agents = plan_agents_for_comparative(state["query"])
    return {**state, "planner_decision": agents}


def node_paper(state: RouterState) -> RouterState:
    out = run_paper(state["query"], get_retriever(_get_store("paper"), k=4))
    return _apply_agent_result(state, "paper", out, set_arxiv=True)


def node_web_docs(state: RouterState) -> RouterState:
    out = run_web_docs(state["query"], get_retriever(_get_store("web_docs"), k=4))
    return _apply_agent_result(state, "web_docs", out, set_tavily=True)


def node_notes(state: RouterState) -> RouterState:
    out = run_notes(state["query"], get_retriever(_get_store("notes"), k=4))
    return _apply_agent_result(state, "notes", out)


def node_planned_parallel(state: RouterState) -> RouterState:
    """Run only the agents chosen by the planner. All return AgentResult; merge via _apply_agent_result."""
    query = state["query"]
    decision = state.get("planner_decision") or ["paper", "web_docs"]
    s = dict(state)
    for agent_name in decision:
        if agent_name == "paper":
            out = run_paper(query, get_retriever(_get_store("paper"), k=3))
            s = _apply_agent_result(s, "paper", out, set_arxiv=True)
        elif agent_name == "web_docs":
            out = run_web_docs(query, get_retriever(_get_store("web_docs"), k=3))
            s = _apply_agent_result(s, "web_docs", out, set_tavily=True)
        elif agent_name == "notes":
            out = run_notes(query, get_retriever(_get_store("notes"), k=3))
            s = _apply_agent_result(s, "notes", out)
    return s


def _route_after_paper(state: RouterState) -> str:
    """Quality-based fallback: add web_docs when paper retrieval is weak."""
    selected = int(state.get("arxiv_results_count", 0) or 0)
    quality = float(state.get("arxiv_quality", 0.0) or 0.0)
    # Require at least one selected paper and reasonable overall paper quality.
    if selected >= 1 and quality >= 0.55:
        return "synthesize"
    return "web_docs_agent"


def node_synthesize(state: RouterState) -> RouterState:
    syn = synthesize(
        state["query"],
        state.get("agent_outputs", {}),
        agent_evidence=state.get("agent_evidence", {}),
        intent=state.get("intent", ""),
        agents_used=list(state.get("agent_outputs", {}).keys()),
        retrieval_summary=state.get("retrieval_summary", {}),
        arxiv_results_count=state.get("arxiv_results_count"),
    )
    return {
        **state,
        "final_answer": syn["final_answer"],
        "sources_used": syn.get("sources_used", []),
        "route_explanation": syn.get("route_explanation", ""),
        "data_sources_used": syn.get("data_sources_used", ""),
    }


def create_graph() -> StateGraph:
    graph = StateGraph(RouterState)

    graph.add_node("preprocess", node_preprocess)
    graph.add_node("classify", node_classify)
    graph.add_node("planner", node_planner)
    graph.add_node("paper_agent", node_paper)
    graph.add_node("web_docs_agent", node_web_docs)
    graph.add_node("notes_agent", node_notes)
    graph.add_node("planned_parallel", node_planned_parallel)
    graph.add_node("synthesize", node_synthesize)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "classify")
    graph.add_conditional_edges("classify", _route_after_classify)
    graph.add_edge("planner", "planned_parallel")
    graph.add_edge("planned_parallel", "synthesize")
    graph.add_edge("web_docs_agent", "synthesize")
    graph.add_edge("notes_agent", "synthesize")
    graph.add_conditional_edges("paper_agent", _route_after_paper, {"synthesize": "synthesize", "web_docs_agent": "web_docs_agent"})
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_router(query: str) -> dict:
    """Run the router graph and return full state."""
    app = create_graph()
    initial: RouterState = {
        "query": query,
        "cleaned_query": "",
        "type_hints": [],
        "is_ambiguous": False,
        "intent": "",
        "planner_decision": [],
        "agent_outputs": {},
        "agent_evidence": {},
        "final_answer": "",
        "sources_used": [],
        "route_explanation": "",
        "data_sources_used": "",
        "tavily_status": "",
        "tavily_results_count": 0,
        "arxiv_status": "",
        "arxiv_results_count": 0,
        "arxiv_candidates_count": 0,
        "arxiv_quality": 0.0,
        "retrieval_summary": {},
    }
    return app.invoke(initial)
