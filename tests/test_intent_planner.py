from router.intent_classifier import plan_agents_for_comparative


def test_plan_agents_for_comparative_defaults_to_two_agents():
    agents = plan_agents_for_comparative("Compare RAG in papers versus tutorials")
    assert agents == ["paper", "web_docs"]


def test_plan_agents_for_comparative_adds_notes_when_requested():
    agents = plan_agents_for_comparative("Compare papers, web docs, and my notes on RAG")
    assert agents == ["paper", "web_docs", "notes"]
