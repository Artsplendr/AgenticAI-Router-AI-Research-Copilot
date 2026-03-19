import importlib


def test_router_graph_import_and_compile_smoke(monkeypatch):
    # Prevent import-time failures in environments without secrets.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    module = importlib.import_module("router.router_graph")
    graph = module.create_graph()
    assert graph is not None
