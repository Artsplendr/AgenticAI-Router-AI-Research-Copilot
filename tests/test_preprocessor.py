from router.preprocessor import preprocess


def test_preprocess_cleans_query_and_extracts_hints():
    result = preprocess("   Compare academic vs practical RAG evaluation   ")
    assert result.cleaned_query == "Compare academic vs practical RAG evaluation"
    assert "compare" in result.type_hints
    assert "academic" in result.type_hints
    assert "practical" in result.type_hints
    assert result.is_ambiguous is False


def test_preprocess_marks_short_query_as_ambiguous():
    result = preprocess("this")
    assert result.is_ambiguous is True
