"""
Streamlit app – interactive UI for research queries, routing, synthesized response, and LangSmith traces.
"""

import os
import re
import sys
import html
from pathlib import Path

# Run from project root so imports resolve
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Normalize LangSmith API key (strip quotes/curly quotes) so tracing works even if .env has key="value"
_langsmith_key = (os.getenv("LANGCHAIN_API_KEY") or "").strip().strip('"').strip("'").strip("\u201c\u201d")
if _langsmith_key:
    os.environ["LANGCHAIN_API_KEY"] = _langsmith_key

from router.router_graph import run_router


def langsmith_enabled() -> bool:
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")
        and bool(os.getenv("LANGCHAIN_API_KEY", "").strip())
    )


def _parse_langsmith_run(run, depth=0):
    """Recursively flatten a LangSmith Run into a list of step dicts with name, type, latency_ms, tokens."""
    steps = []
    name = getattr(run, "name", None) or str(getattr(run, "id", ""))[:8]
    run_type = getattr(run, "run_type", "") or "unknown"
    latency_ms = getattr(run, "latency_ms", None)
    if latency_ms is None and hasattr(run, "start_time") and hasattr(run, "end_time") and run.start_time and run.end_time:
        try:
            delta = (run.end_time - run.start_time).total_seconds() * 1000
            latency_ms = int(delta)
        except Exception:
            pass
    total_tokens = getattr(run, "total_tokens", None) or 0
    status = getattr(run, "status", "") or ""
    err = getattr(run, "error", None) or ""
    steps.append({
        "depth": depth,
        "name": name,
        "run_type": run_type,
        "latency_ms": latency_ms,
        "total_tokens": total_tokens,
        "status": status,
        "error": err[:200] if err else "",
    })
    for child in getattr(run, "child_runs", None) or []:
        steps.extend(_parse_langsmith_run(child, depth + 1))
    return steps


def _fetch_langsmith_trace(trace_id=None):
    """Fetch a root run from LangSmith by trace_id only. Returns parsed summary or None. Requires trace_id so metrics are for this session's run only."""
    if not os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1") or not os.getenv("LANGCHAIN_API_KEY", "").strip():
        return None
    if not trace_id or not str(trace_id).strip():
        return None
    try:
        from langsmith import Client
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        client = Client()
        runs = list(client.list_runs(project_name=project, trace_id=trace_id, is_root=True, limit=1))
        if not runs:
            return None
        root = runs[0]
        full = client.read_run(root.id, load_child_runs=True)
        steps = _parse_langsmith_run(full)
        # Use only the root run's metrics for this trace (no summing children to avoid aggregation/double-count)
        total_latency = getattr(full, "latency_ms", None)
        if (total_latency is None or total_latency == 0) and getattr(full, "start_time", None) and getattr(full, "end_time", None):
            try:
                total_latency = int((full.end_time - full.start_time).total_seconds() * 1000)
            except Exception:
                total_latency = None
        if total_latency is None or total_latency == 0:
            # Fallback: sum direct children (depth 1) for this trace only when root has no duration (e.g. async flush)
            direct_children = [s for s in steps if s.get("depth") == 1]
            total_latency = sum(s.get("latency_ms") or 0 for s in direct_children) or None
        total_tokens = getattr(full, "total_tokens", None)
        if total_tokens is None or total_tokens == 0:
            # Fallback: sum only direct children (depth 1) for this trace only, to avoid double-counting nested runs
            direct_children = [s for s in steps if s.get("depth") == 1]
            total_tokens = sum(s.get("total_tokens") or 0 for s in direct_children) or None
        return {
            "trace_name": getattr(full, "name", "Research Copilot") or "Research Copilot",
            "trace_id": str(getattr(full, "id", "")),
            "status": getattr(full, "status", "") or "unknown",
            "total_latency_ms": total_latency,
            "total_tokens": total_tokens,
            "steps": steps,
            "error": getattr(full, "error", None) or "",
        }
    except Exception:
        return None


def _interpret_langsmith_trace(parsed):
    """Generate a short plain-language interpretation of the trace data using an LLM."""
    if not parsed or not parsed.get("steps"):
        return ""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        summary_lines = [
            f"Trace: {parsed.get('trace_name', 'N/A')}",
            f"Status: {parsed.get('status', 'N/A')}",
            f"Last trace time: {parsed.get('total_latency_ms') or 0} ms",
            f"Last trace tokens: {parsed.get('total_tokens') or 0}",
            "Steps: " + ", ".join(s.get("name", "?") for s in parsed["steps"][:15]),
        ]
        if parsed.get("error"):
            summary_lines.append(f"Error: {parsed['error'][:150]}")
        prompt = ChatPromptTemplate.from_messages([(
            "human",
            "Summarize this LangSmith trace in 2–4 short sentences for a non-technical user. "
            "Explain what the system did (e.g. preprocessed the query, classified intent, ran which agents, synthesized the answer) and whether it succeeded or had issues. "
            "Be concise and clear.\n\nTrace summary:\n{summary}"
        )])
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"summary": "\n".join(summary_lines)})
    except Exception:
        return ""


def _flow_steps(result: dict | None = None, running: bool = False) -> list[str]:
    """Build ordered flow steps for running/completed status strip."""
    if running or not result:
        return [
            "Preprocessed",
            "Classifying intent",
            "Planning agents",
            "Synthesizing with citations",
        ]
    intent = (result.get("intent") or "").strip() or "unknown"
    outputs = result.get("agent_outputs") or {}
    planner = result.get("planner_decision") or []
    used_agents = list(outputs.keys())
    used_agents_txt = ", ".join(used_agents) if used_agents else "—"
    route_step = (
        f"Planned agents: {', '.join(planner) if planner else used_agents_txt}"
        if intent == "comparative"
        else f"Routed agents: {used_agents_txt}"
    )
    steps = [
        "Preprocessed",
        f"Classified as {intent}",
        route_step,
        "Synthesized with citations",
    ]
    paper_quality = float(result.get("arxiv_quality", 0.0) or 0.0)
    if ((result.get("arxiv_results_count", 0) or 0) < 1 or paper_quality < 0.55) and outputs.get("web_docs"):
        steps.append("Fallback to web_docs")
    return steps


def _render_flow_strip(steps: list[str], running: bool) -> str:
    """Render compact flow chips with dynamic gray->green transitions."""
    chips = []
    for i, step in enumerate(steps):
        if running:
            chips.append(
                f'<span class="flow-chip flow-running" style="animation-delay: {i * 0.85:.2f}s">{step}</span>'
            )
        else:
            chips.append(f'<span class="flow-chip flow-done">{step}</span>')
    arrows = '<span class="flow-arrow">→</span>'
    joined = f" {arrows} ".join(chips)
    return f"""
<div class="flow-wrap {'flow-wrap-running' if running else 'flow-wrap-done'}">
  {joined}
</div>
<style>
  .flow-wrap {{
    margin: 0.15rem 0 0.55rem 0;
    padding: 0.4rem 0.58rem;
    border-radius: 0.5rem;
    font-size: 0.84rem;
    line-height: 1.2;
    white-space: normal;
  }}
  .flow-wrap-running {{
    border: 1px solid rgba(120,120,120,0.28);
    background: rgba(120,120,120,0.06);
  }}
  .flow-wrap-done {{
    border: 1px solid rgba(46,125,50,0.35);
    background: rgba(46,125,50,0.08);
  }}
  .flow-chip {{
    display: inline-block;
    padding: 0.18rem 0.45rem;
    border-radius: 999px;
    border: 1px solid transparent;
    font-weight: 600;
  }}
  .flow-arrow {{
    color: #8a8a8a;
    font-weight: 700;
    margin: 0 0.16rem;
  }}
  .flow-running {{
    color: #5f6368;
    background: rgba(128,128,128,0.12);
    border-color: rgba(128,128,128,0.26);
    animation: flowDone 0.45s ease-out forwards;
  }}
  .flow-done {{
    color: #1b5e20;
    background: rgba(46,125,50,0.14);
    border-color: rgba(46,125,50,0.34);
  }}
  @keyframes flowDone {{
    from {{
      color: #5f6368;
      background: rgba(128,128,128,0.12);
      border-color: rgba(128,128,128,0.26);
    }}
    to {{
      color: #1b5e20;
      background: rgba(46,125,50,0.14);
      border-color: rgba(46,125,50,0.34);
    }}
  }}
</style>
"""


def _build_sidebar_mermaid(result: dict | None = None, running: bool = False) -> tuple[str, bool]:
    """Build sidebar mermaid code + running flag for dynamic flow highlighting."""
    mermaid_lines = [
        "flowchart TD",
        "  Q[Query] --> P[Preprocessor]",
        "  P --> C[Classify intent]",
        "  C --> I1[academic]",
        "  C --> I2[practical, general]",
        "  C --> I3[contextual]",
        "  C --> I4[comparative]",
        "  I1 --> PA[paper]",
        "  I2 --> W[web_docs]",
        "  I3 --> N[notes]",
        "  I4 --> PP[\"paper + web_docs ± notes\"]",
        "  subgraph agents[\" \"]",
        "    PA",
        "    W",
        "    N",
        "    PP",
        "  end",
        "  PA --> S[Synthesizer]",
        "  W --> S",
        "  N --> S",
        "  PP --> S",
        "  S --> F[Final answer + citations]",
    ]

    # Mermaid link order based on declarations above
    # 0 Q->P, 1 P->C, 2 C->I1, 3 C->I2, 4 C->I3, 5 C->I4, 6 I1->PA, 7 I2->W, 8 I3->N,
    # 9 I4->PP, 10 PA->S, 11 W->S, 12 N->S, 13 PP->S, 14 S->F
    done_links = []
    done_nodes = []

    if running:
        # Keep diagram neutral while running; highlight only when completed result is available.
        return "\n".join(mermaid_lines), True

    if result:
        intent = (result.get("intent") or "").strip()
        outputs = result.get("agent_outputs") or {}

        done_links.extend([0, 1, 14])
        done_nodes.extend(["Q", "P", "C", "S", "F"])

        intent_edge = {"academic": 2, "practical": 3, "general": 3, "contextual": 4, "comparative": 5}.get(intent)
        if intent_edge is not None:
            done_links.append(intent_edge)

        if intent == "academic":
            done_links.append(6)
            done_nodes.extend(["I1", "PA"])
        elif intent in ("practical", "general"):
            done_links.append(7)
            done_nodes.extend(["I2", "W"])
        elif intent == "contextual":
            done_links.append(8)
            done_nodes.extend(["I3", "N"])
        elif intent == "comparative":
            done_links.extend([9, 13])
            done_nodes.extend(["I4", "PP"])

        if outputs.get("paper"):
            done_links.append(10)
            done_nodes.append("PA")
        if outputs.get("web_docs"):
            done_links.append(11)
            done_nodes.append("W")
        if outputs.get("notes"):
            done_links.append(12)
            done_nodes.append("N")

        done_links = sorted(set(done_links))
        done_nodes = sorted(set(done_nodes))
        if done_nodes:
            for nid in done_nodes:
                mermaid_lines.append(
                    f"  style {nid} fill:#e7f7e7,stroke:#2e7d32,stroke-width:2.2px,color:#1b5e20;"
                )
        for idx in done_links:
            mermaid_lines.append(
                f"  linkStyle {idx} stroke:#2e7d32,stroke-width:3px,color:#2e7d32,opacity:0.96;"
            )

    return "\n".join(mermaid_lines), False


st.set_page_config(page_title="AI Research Copilot", layout="wide")

# Hide Streamlit's Deploy button and main menu; compact vertical spacing (title → description → divider → query)
st.markdown("""
<style>
  #MainMenu { visibility: hidden !important; }
  .stDeployButton { display: none !important; }
  [data-testid="stDeployButton"] { display: none !important; }
  [data-testid="baseButton-header"] { display: none !important; visibility: hidden !important; }
  button[kind="header"] { display: none !important; }
  a[href*="streamlit.io"][target="_blank"] { display: none !important; }
  footer { visibility: hidden !important; }
  [data-testid="stAppViewContainer"] .block-container { padding-top: 0.75rem !important; }
  /* Tighten vertical flow: title, description, divider (~40–50% less space) */
  [data-testid="stAppViewContainer"] .block-container h1 { margin-bottom: 0.35em !important; }
  [data-testid="stAppViewContainer"] .block-container .stMarkdown { margin-top: 0.05rem !important; margin-bottom: 0.2rem !important; }
  [data-testid="stAppViewContainer"] .block-container .stMarkdown p { line-height: 1.4 !important; margin: 0.15em 0 !important; }
  [data-testid="stAppViewContainer"] .block-container hr { margin: 0.35rem 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("AI Research Copilot (Multi-Source Router)")
st.markdown(
    "Queries are preprocessed and classified by intent, then routed to one or more specialist agents: **Paper** (arXiv), **Web Docs** (Tavily), and **Notes** (local vector store). "
    "For comparative questions, a planner selects which agents to run in parallel; a synthesizer merges their evidence into a single answer with citations and a route explanation."
)
st.divider()

# Sidebar: compact but readable spacing
st.markdown("""
<style>
  [data-testid="stSidebar"] .stCaption { margin: 0.2rem 0 !important; line-height: 1.4 !important; padding: 0 !important; }
  [data-testid="stSidebar"] .stMarkdown { margin: 0.25rem 0 !important; line-height: 1.45 !important; padding: 0 !important; }
  [data-testid="stSidebar"] .stMarkdown p { margin: 0.2rem 0 !important; }
  [data-testid="stSidebar"] .element-container { margin-bottom: 0.35rem !important; }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { margin-bottom: 0 !important; min-height: 0 !important; }
  [data-testid="stSidebar"] h2 { margin: 0.6rem 0 0.3rem 0 !important; padding: 0 !important; line-height: 1.3 !important; }
  [data-testid="stSidebar"] .stAlert { margin: 0.4rem 0 !important; padding: 0.4rem 0.5rem !important; }
  [data-testid="stSidebar"] button[kind="secondary"], [data-testid="stSidebar"] .stLinkButton { margin: 0.35rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# LangSmith status and link in sidebar
with st.sidebar:
    st.subheader("Observability")
    if langsmith_enabled():
        st.success("LangSmith tracing is **on**")
        st.caption("Runs are sent to your project. Open the dashboard to see the latest traces.")
        st.link_button("Open LangSmith", "https://smith.langchain.com", type="secondary")
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        st.caption(f"Project: **{project}**")
    else:
        st.info("LangSmith tracing is **off**")
        st.caption("Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to enable.")
    st.subheader("Tavily (web search)")
    if os.getenv("TAVILY_API_KEY", "").strip():
        st.caption("API key is set. Web Docs agent uses live search as primary source.")
    else:
        st.caption("Not configured. Set `TAVILY_API_KEY` in `.env` for web search.")
    st.subheader("arXiv (papers)")
    st.caption("Paper agent searches [arxiv.org](https://arxiv.org) by default. No API key required.")
    st.subheader("Notes agent")
    st.caption("Agents use your ingested notes. If they say \"not enough information\", add data:")
    st.caption("**Web docs:** `python scripts/ingest_web_docs.py <url1> <url2> ...`")
    st.caption("See README → Data ingestion for paper/notes.")

    st.subheader("Agent flow")
    _sidebar_result = st.session_state.get("result") if isinstance(st.session_state.get("result"), dict) else None
    _sidebar_running = bool(st.session_state.get("run_in_progress"))
    mermaid_code, mermaid_running = _build_sidebar_mermaid(_sidebar_result, _sidebar_running)
    running_edge_css = ""
    running_anim_js = ""
    mermaid_html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid@9/dist/mermaid.min.js"></script>
    <style>
      /* Agents container: give the cluster title more vertical room */
      .mermaid .cluster rect {{ min-height: 80px !important; }}
      /* Make the subgraph title ("Agents") more prominent */
      .mermaid .cluster .cluster-label,
      .mermaid .cluster .label span,
      .mermaid .cluster .label div,
      .mermaid .cluster [class*="label"] {{
        font-weight: 700 !important;
        font-size: 13px !important;
        text-align: center !important;
      }}
      .mermaid .cluster .label-container {{
        transform: translate(0, 10px);
        margin-bottom: 4px;
      }}
      .mermaid .cluster .label {{
        padding-bottom: 6px !important;
      }}
      /* Glow highlighted (green) edges after route is known */
      .mermaid .edgePath path[style*="#2e7d32"],
      .mermaid .edgePath path[style*="2e7d32"],
      .mermaid .edgePath path[style*="rgb(46,125,50)"],
      .mermaid .edgePath path[style*="rgb(46, 125, 50)"] {{
        filter: drop-shadow(0 0 5px rgba(76,175,80,0.88)) !important;
      }}
      /* Highlight involved nodes with light green + soft glow */
      .mermaid .node rect[style*="#2e7d32"],
      .mermaid .node rect[style*="2e7d32"],
      .mermaid .node rect[style*="rgb(46,125,50)"],
      .mermaid .node rect[style*="rgb(46, 125, 50)"] {{
        fill: #e7f7e7 !important;
        stroke: #2e7d32 !important;
        stroke-width: 2.2px !important;
        filter: drop-shadow(0 0 6px rgba(76,175,80,0.58));
      }}
      .mermaid .node rect[style*="#2e7d32"] + .label,
      .mermaid .node rect[style*="2e7d32"] + .label,
      .mermaid .node rect[style*="rgb(46,125,50)"] + .label,
      .mermaid .node rect[style*="rgb(46, 125, 50)"] + .label,
      .mermaid .node rect[style*="#2e7d32"] ~ text,
      .mermaid .node rect[style*="2e7d32"] ~ text {{
        fill: #1b5e20 !important;
      }}
      /* Base edge style (JS updates it stage-by-stage during run) */
      .mermaid .edgePath path, .mermaid .edgePaths .edgePath path {{
        stroke: #9aa0a6;
        stroke-width: 1.6px;
        opacity: 0.60;
      }}
      {running_edge_css}
    </style>
    <script>
      mermaid.initialize({{
        startOnLoad: true,
        theme: 'neutral',
        flowchart: {{ useMaxWidth: true, htmlLabels: true, nodeSpacing: 12, rankSpacing: 18, curve: 'basis', subGraphPadding: 28 }}
      }});
      {running_anim_js}
    </script>
    <pre class="mermaid" style="font-size: 10px;">
    {mermaid_code.strip()}
    </pre>
    """
    components.html(mermaid_html, height=420, scrolling=True)

# Single row: research query (fills width) + Run button (same baseline, consistent height/padding)
st.caption("Research query")
col_query, col_btn = st.columns([4, 1])
with col_query:
    query = st.text_area(
        "Research query",
        placeholder="e.g. Explain RAG in simple terms / What do arXiv papers say about retrieval augmentation? / Compare academic and practical views on RAG",
        height=100,
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("Run", key="run_btn", use_container_width=True)

# One-row layout: input + button on same baseline, consistent padding, button as natural action
st.markdown("""
<style>
  /* Hide non-working "Press ⌘+Enter to apply" hint under text_area */
  [data-testid="InputInstructions"] { display: none !important; }
  /* Single row container: top-aligned so input and button share the same vertical baseline */
  [data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
    gap: 0.75rem !important;
  }
  /* Left column: minimal gap between "Research query" label and input */
  [data-testid="stHorizontalBlock"] > div:first-child {
    flex: 1 1 auto !important;
    min-width: 0 !important;
  }
  [data-testid="stHorizontalBlock"] > div:first-child .stCaption,
  [data-testid="stHorizontalBlock"] > div:first-child [data-testid="stCaption"] {
    margin-bottom: 0.1rem !important;
    margin-top: 0 !important;
    padding-bottom: 0 !important;
  }
  [data-testid="stHorizontalBlock"] > div:last-child {
    display: flex !important;
    align-items: flex-start !important;
    justify-content: flex-end !important;
    padding-top: 0.25rem !important;
  }
  /* Run button: red-coral #F84F4F, white text #FFFFFF, hover #e04545 */
  [data-testid="stHorizontalBlock"] > div:last-child button[kind="primary"],
  [data-testid="stHorizontalBlock"] > div:last-child button {
    background-color: #F84F4F !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 0.5rem 1.25rem !important;
    min-height: 2.25rem !important;
    font-weight: 500 !important;
  }
  [data-testid="stHorizontalBlock"] > div:last-child button[kind="primary"]:hover,
  [data-testid="stHorizontalBlock"] > div:last-child button:hover {
    background-color: #e04545 !important;
    color: #FFFFFF !important;
  }
</style>
""", unsafe_allow_html=True)

# When Run is clicked: clear previous result and rerun so tabs disappear; then we'll show only spinner until done
if run_clicked:
    if not query or not query.strip():
        st.warning("Enter a query.")
    else:
        st.session_state["result"] = None
        st.session_state["run_in_progress"] = True
        st.rerun()

# Run the router in a separate pass so the UI shows only the spinner (no old tabs)
if st.session_state.get("run_in_progress"):
    try:
        st.markdown(_render_flow_strip(_flow_steps(running=True), running=True), unsafe_allow_html=True)
        if langsmith_enabled():
            from langsmith import trace
            with trace(name="Research Copilot", run_type="chain") as run:
                with st.spinner("Routing and generating..."):
                    result = run_router(query.strip())
            # Store trace_id so the LangSmith tab shows this run (avoids "pending" from a different/latest run)
            try:
                st.session_state["last_trace_id"] = str(getattr(run, "trace_id", None) or getattr(run, "id", None) or "")
            except Exception:
                st.session_state["last_trace_id"] = ""
        else:
            with st.spinner("Routing and generating..."):
                result = run_router(query.strip())
            st.session_state["last_trace_id"] = ""
    except Exception as e:
        st.error(f"Error: {e}")
        result = {}
        st.session_state["last_trace_id"] = ""
    st.session_state["result"] = result
    st.session_state["run_in_progress"] = False
    st.rerun()

if st.session_state.get("result"):
    result = st.session_state["result"]
    if result:
            outputs = result.get("agent_outputs") or {}
            agent_evidence = result.get("agent_evidence") or {}
            sources = result.get("sources_used") or []
            intent = result.get("intent", "")
            planner = result.get("planner_decision", [])
            retrieval = result.get("retrieval_summary", {})
            route_expl = result.get("route_explanation", "")
            agents_used = list(outputs.keys())

            SOURCE_LABELS = {"paper": "Paper", "web_docs": "Web", "notes": "Notes"}

            def _origin_for_source(source: dict, agent_evidence: dict) -> str:
                """Get origin label (Paper/Web/Notes) for a source; infer from agent_evidence if missing."""
                origin = source.get("origin", "").strip()
                if origin:
                    return origin
                title = source.get("title") or ""
                url = source.get("url") or ""
                for agent_name, items in (agent_evidence or {}).items():
                    for e in items:
                        if (e.get("source_title") or "").strip() == title and (e.get("source_url") or "").strip() == url:
                            return SOURCE_LABELS.get(agent_name, agent_name)
                return "—"

            def _format_citation(e):
                title = e.get("source_title") or "Unknown"
                url = e.get("source_url") or ""
                authors = e.get("source_authors") or ""
                year = e.get("source_year") or ""
                parts = [p for p in [authors, f"({year})" if year else ""] if p]
                cite = " ".join(parts) + ". " if parts else ""
                if url:
                    return f"- {cite}[{title}]({url})"
                return f"- {cite}{title}"

            def _format_final_answer_text(text: str) -> str:
                """Normalize and improve paragraph structure for readability in the Final answer tab."""
                if not text:
                    return ""
                normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                raw_lines = [ln.strip() for ln in normalized.split("\n")]

                def _is_structured_line(ln: str) -> bool:
                    return (
                        ln.startswith("#")
                        or ln.startswith("- ")
                        or ln.startswith("* ")
                        or ln.startswith("> ")
                        or (len(ln) > 2 and ln[0].isdigit() and ln[1] in (".", ")"))
                    )

                def _split_prose_block(block: str) -> list[str]:
                    # Preserve model meaning: no mechanical splitting. Keep prose block as one paragraph.
                    block = (block or "").strip()
                    return [block] if block else []

                blocks = []
                prose_acc = []

                def _flush_prose():
                    nonlocal prose_acc
                    prose_text = " ".join(p for p in prose_acc if p).strip()
                    if prose_text:
                        blocks.extend(_split_prose_block(prose_text))
                    prose_acc = []

                for ln in raw_lines:
                    if not ln:
                        _flush_prose()
                        continue
                    if _is_structured_line(ln):
                        _flush_prose()
                        blocks.append(ln)
                    else:
                        prose_acc.append(ln)
                _flush_prose()

                # Keep model/newline structure and apply only sentence-boundary fallback for dense prose.
                return "\n\n".join(b for b in blocks if b).strip()

            def _render_source_line(source: dict, origin: str):
                """Render one source item consistently."""
                idx = source.get("index")
                title = source.get("title", "Unknown")
                url = source.get("url", "")
                year = source.get("year", "")
                authors = source.get("authors", "")
                parts = []
                if authors:
                    parts.append(authors)
                if year:
                    parts.append(f"({year})")
                cite = " ".join(parts) + ". " if parts else ""
                idx_tag = f"[{idx}] " if idx is not None else ""
                if url:
                    st.markdown(f"- {idx_tag}[**{origin}**] {cite}[{title}]({url})")
                else:
                    st.markdown(f"- {idx_tag}[**{origin}**] {cite}{title}")

            def _md_inline_to_html(text: str) -> str:
                """Minimal inline markdown conversion for bold and links."""
                escaped = html.escape(text)
                escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
                escaped = re.sub(
                    r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
                    r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
                    escaped,
                )
                return escaped

            def _render_final_answer_html(text: str) -> str:
                """Render final answer with reliable UI-level paragraph indentation."""
                if not text or not text.strip():
                    return ""
                blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
                parts = [
                    '<div class="final-answer-indented">',
                    "<style>",
                    ".final-answer-indented p { margin: 0 0 0.88rem 0; line-height: 1.62; text-indent: 1.35em; }",
                    ".final-answer-indented h1,.final-answer-indented h2,.final-answer-indented h3,.final-answer-indented h4 { margin: 0.7rem 0 0.45rem 0; text-indent: 0; }",
                    ".final-answer-indented ul,.final-answer-indented ol { margin: 0 0 0.85rem 1.25rem; }",
                    ".final-answer-indented li { margin: 0.22rem 0; }",
                    "</style>",
                ]
                for block in blocks:
                    if block.startswith("### "):
                        parts.append(f"<h4>{_md_inline_to_html(block[4:].strip())}</h4>")
                    elif block.startswith("## "):
                        parts.append(f"<h3>{_md_inline_to_html(block[3:].strip())}</h3>")
                    elif block.startswith("# "):
                        parts.append(f"<h2>{_md_inline_to_html(block[2:].strip())}</h2>")
                    else:
                        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
                        is_ul = lines and all(ln.startswith("- ") or ln.startswith("* ") for ln in lines)
                        is_ol = lines and all(len(ln) > 2 and ln[0].isdigit() and ln[1] in (".", ")") for ln in lines)
                        if is_ul:
                            parts.append("<ul>")
                            for ln in lines:
                                parts.append(f"<li>{_md_inline_to_html(ln[2:].strip())}</li>")
                            parts.append("</ul>")
                        elif is_ol:
                            parts.append("<ol>")
                            for ln in lines:
                                parts.append(f"<li>{_md_inline_to_html(ln[2:].strip())}</li>")
                            parts.append("</ol>")
                        else:
                            parts.append(f"<p>{_md_inline_to_html(block)}</p>")
                parts.append("</div>")
                return "".join(parts)

            # Compact status strip under the query box
            agents_used_text = " + ".join(SOURCE_LABELS.get(a, a) for a in agents_used) if agents_used else "—"
            paper_quality = float(result.get("arxiv_quality", 0.0) or 0.0)
            arxiv_selected = int(result.get("arxiv_results_count", 0) or 0)
            arxiv_candidates = int(result.get("arxiv_candidates_count", 0) or 0)
            fallback_used = ((arxiv_selected < 1) or (paper_quality < 0.55)) and bool(outputs.get("web_docs"))
            status_strip = (
                f"Intent: {intent or '—'} | "
                f"Agents used: {agents_used_text} | "
                f"Fallback: {'yes' if fallback_used else 'no'} | "
                f"Sources: {len(sources)}"
            )
            flow_strip = _render_flow_strip(_flow_steps(result=result), running=False)
            st.markdown(
                f"""
<div style="
  margin: 0.15rem 0 0.55rem 0;
  padding: 0.42rem 0.62rem;
  border: 1px solid rgba(120,120,120,0.25);
  border-radius: 0.5rem;
  background: rgba(120,120,120,0.06);
  font-size: 0.86rem;
  line-height: 1.25;
">
  {status_strip}
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown(flow_strip, unsafe_allow_html=True)

            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Final answer + citations",
                "Paper agent",
                "Web docs agent",
                "Notes agent",
                "Planner / comparative",
                "Route trace",
                "LangSmith tracing",
            ])

            with tab1:
                st.subheader("Final answer")
                data_sources = result.get("data_sources_used", "")
                if data_sources:
                    st.markdown("**Data sources used:** " + data_sources)
                if fallback_used and (outputs.get("web_docs") or outputs.get("notes")):
                    st.info(
                        "Paper retrieval quality was low for this query; web search and/or notes were used instead."
                    )
                formatted_answer = _format_final_answer_text(result.get("final_answer", ""))
                st.markdown(_render_final_answer_html(formatted_answer), unsafe_allow_html=True)
                if sources:
                    st.subheader("Sources used")
                    st.caption("**Paper** = arXiv papers · **Web** = web search · **Notes** = your ingested documents")
                    if arxiv_candidates > 0:
                        st.caption(
                            f"Paper filtering: kept **{arxiv_selected}** of **{arxiv_candidates}** arXiv candidates "
                            f"(quality score: **{paper_quality:.2f}**)."
                        )
                    grouped = {"Paper": [], "Web": [], "Notes": [], "Other": []}
                    for s in sources:
                        origin = _origin_for_source(s, agent_evidence)
                        if origin in grouped:
                            grouped[origin].append(s)
                        else:
                            grouped["Other"].append(s)

                    if grouped["Paper"]:
                        with st.expander(f"Paper sources ({len(grouped['Paper'])})", expanded=False):
                            for s in grouped["Paper"]:
                                _render_source_line(s, "Paper")
                    if grouped["Web"]:
                        with st.expander(f"Web sources ({len(grouped['Web'])})", expanded=False):
                            for s in grouped["Web"]:
                                _render_source_line(s, "Web")
                    if grouped["Notes"]:
                        with st.expander(f"Notes sources ({len(grouped['Notes'])})", expanded=False):
                            for s in grouped["Notes"]:
                                _render_source_line(s, "Notes")
                    if grouped["Other"]:
                        with st.expander(f"Other sources ({len(grouped['Other'])})", expanded=False):
                            for s in grouped["Other"]:
                                _render_source_line(s, _origin_for_source(s, agent_evidence))

            with tab2:
                st.markdown(
                    "The **Paper agent** answers research-heavy and academic questions using **arXiv.org** as its primary source. "
                    "It retrieves relevant papers, summarizes evidence, and returns an answer with citations that include title, year, and authors when available. "
                    "This agent is used when the query intent is classified as *academic*."
                )
                text = outputs.get("paper", "")
                if text:
                    st.write(text)
                    evidence = agent_evidence.get("paper", [])
                    if evidence:
                        with st.expander(f"Citations ({len(evidence)})", expanded=False):
                            for e in evidence:
                                st.markdown(_format_citation(e))
                else:
                    st.caption("Paper agent was not used for this run.")

            with tab3:
                st.markdown(
                    "The **Web docs agent** answers practical and how-to questions using **Tavily** web search and, when configured, your ingested web documents. "
                    "It retrieves up-to-date articles, tutorials, and documentation, then summarizes the evidence and returns an answer with citations. "
                    "This agent is used when the query intent is classified as *practical* or *general*, or as a fallback when the Paper agent has no or few results."
                )
                text = outputs.get("web_docs", "")
                if text:
                    st.write(text)
                    evidence = agent_evidence.get("web_docs", [])
                    if evidence:
                        with st.expander(f"Citations ({len(evidence)})", expanded=False):
                            for e in evidence:
                                st.markdown(_format_citation(e))
                else:
                    st.caption("Web docs agent was not used for this run.")

            with tab4:
                st.markdown(
                    "The **Notes agent** answers questions using your **ingested notes and documents** stored in a local vector store. "
                    "It retrieves relevant chunks from your saved context, then summarizes the evidence and returns an answer with citations. "
                    "This agent is used when the query intent is classified as *contextual* (e.g. “what do my notes say”, or when you want answers grounded in your own data)."
                )
                text = outputs.get("notes", "")
                if text:
                    st.write(text)
                else:
                    st.caption("Notes agent was not used for this run.")

            with tab5:
                st.markdown(
                    "The **Planner** is used only when the query intent is **comparative** (e.g. “compare academic and practical views”, “contrast X with Y”). "
                    "It decides which agents (one to three) to run in parallel—typically *paper* and *web_docs*, with *notes* added when the query explicitly refers to your own data. "
                    "The planner is rule-based and conservative; the chosen agents then run in parallel and their outputs are merged by the synthesizer."
                )
                st.caption("**Intent**")
                st.code(intent, language=None)
                if intent == "comparative" and planner:
                    st.caption("**Planner decision (agents used)**")
                    st.code(" + ".join(planner), language=None)
                else:
                    st.caption("Planner is only used for comparative intent.")
                    if intent != "comparative":
                        st.caption(f"This run used intent: **{intent}**.")

            with tab6:
                st.subheader("Route trace")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("**Query received**")
                    st.code(result.get("query", "")[:200], language=None)
                with col2:
                    st.caption("**Intent classified**")
                    st.code(intent, language=None)
                if planner:
                    st.caption("**Planner decision (comparative)**")
                    st.code(" + ".join(planner), language=None)
                if retrieval:
                    st.caption("**Retrieval status**")
                    st.json(retrieval)
                st.caption("**Synthesizer completed**")
                if route_expl:
                    st.info("**Why this route:** " + route_expl)
                ts = result.get("tavily_status", "")
                tc = result.get("tavily_results_count", 0)
                if ts == "ok":
                    st.success(f"**Tavily search:** {tc} result(s) used for web_docs agent.")
                elif ts == "no_key":
                    st.info("**Tavily search:** Not configured (set `TAVILY_API_KEY` in `.env`).")
                elif ts == "error":
                    st.warning("**Tavily search:** Error (check terminal logs).")
                elif ts == "" and tc == 0:
                    st.caption("**Tavily search:** Not used this run.")
                arxiv_s = result.get("arxiv_status", "")
                arxiv_c = int(result.get("arxiv_results_count", 0) or 0)
                arxiv_total_c = int(result.get("arxiv_candidates_count", 0) or 0)
                arxiv_q = float(result.get("arxiv_quality", 0.0) or 0.0)
                if arxiv_s == "ok":
                    if arxiv_total_c > 0:
                        st.success(
                            f"**arXiv:** {arxiv_c} selected from {arxiv_total_c} candidate(s) "
                            f"(quality score {arxiv_q:.2f})."
                        )
                    else:
                        st.success(f"**arXiv:** {arxiv_c} result(s) used for paper agent.")
                elif arxiv_s == "error":
                    st.warning("**arXiv:** Error (check terminal logs).")
                elif arxiv_s == "" and arxiv_c == 0:
                    st.caption("**arXiv:** Not used this run.")
                if fallback_used and outputs.get("web_docs"):
                    st.caption("_Fallback: paper retrieval quality was low; web_docs was also used._")
                if langsmith_enabled():
                    st.caption("Trace sent to LangSmith — open the dashboard (sidebar) to view it.")

            with tab7:
                st.subheader("LangSmith trace data")
                if not langsmith_enabled():
                    st.info("LangSmith tracing is **off**. Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env`, then run a query to see trace data here.")
                    st.link_button("Open LangSmith dashboard", "https://smith.langchain.com", type="secondary")
                else:
                    trace_id = st.session_state.get("last_trace_id") or None
                    if not trace_id:
                        st.caption("Run a query above to see trace data for **this session only**. Status, total tokens, and total time will reflect that run only.")
                        st.link_button("Open LangSmith dashboard", "https://smith.langchain.com", type="secondary")
                        parsed = None
                    else:
                        with st.spinner("Loading trace from LangSmith…"):
                            parsed = _fetch_langsmith_trace(trace_id)
                    if not parsed:
                        if trace_id:
                            st.caption("Trace not found or still syncing. Try again in a moment or run a new query.")
                            st.link_button("Open LangSmith dashboard", "https://smith.langchain.com", type="secondary")
                    else:
                        # If we have a successful result from this run but trace still shows "pending" (async flush), treat as success
                        display_parsed = dict(parsed)
                        if result and parsed.get("status") == "pending" and not parsed.get("error"):
                            display_parsed["status"] = "success"
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Status", display_parsed.get("status", "—"))
                        with col2:
                            tok = display_parsed.get("total_tokens")
                            st.metric("Last trace tokens", tok if (tok is not None and tok > 0) else "—")
                        with col3:
                            lat_ms = display_parsed.get("total_latency_ms")
                            st.metric("Last trace time", f"{lat_ms} ms" if lat_ms is not None and lat_ms > 0 else "—")
                        st.caption(f"Trace ID: `{display_parsed.get('trace_id', '')[:8]}…`")
                        st.caption("_Metrics are for the last run in this session only (this trace). The LangSmith dashboard shows aggregates over all runs in the project._")
                        steps_list = display_parsed.get("steps", [])
                        with st.expander(f"**Steps in this trace** ({len(steps_list)} steps)", expanded=False):
                            for s in steps_list:
                                indent = "  " * (s.get("depth", 0))
                                lat = s.get("latency_ms")
                                tok = s.get("total_tokens")
                                lat_str = f" — {lat} ms" if lat is not None else ""
                                tok_str = f", {tok} tokens" if tok else ""
                                st.code(f"{indent}{s.get('name', '?')} ({s.get('run_type', '')}{lat_str}{tok_str})", language=None)
                                if s.get("error"):
                                    st.caption(f"Error: {s['error'][:150]}")
                        interpretation = _interpret_langsmith_trace(display_parsed)
                        if interpretation:
                            st.subheader("Current status")
                            st.info(interpretation)
                        st.link_button("Open in LangSmith", "https://smith.langchain.com", type="secondary")
