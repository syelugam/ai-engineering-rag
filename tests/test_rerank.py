"""Reranking tests. The cross-encoder is faked, so nothing is downloaded."""
from unittest.mock import MagicMock, patch

import pytest

from rag_service import COSINE_MIN_SCORE, RERANK_MIN_SCORE, RAGService

CANDIDATES = [
    {"text": "Lambda Functions\nPublish a version.", "score": 0.62, "section": "Lambda"},
    {"text": "Service Layout\nLambda is triggered by SQS.", "score": 0.40, "section": "Layout"},
    {"text": "Monitoring\nAlerts fire on errors.", "score": 0.35, "section": "Monitoring"},
]


def make_service(tmp_path, **kwargs):
    document = tmp_path / "guide.md"
    document.write_text("# Guide\nSome text.\n", encoding="utf-8")
    with patch("rag_service.SentenceTransformer"):
        return RAGService(document_path=str(document), **kwargs)


@pytest.fixture
def reranking_service(tmp_path):
    service = make_service(tmp_path, reranker_model_name="fake-cross-encoder")
    service.vector_search = MagicMock(return_value=list(CANDIDATES))
    service._reranker = MagicMock()
    # Raw logits: the reranker prefers the second candidate, then the first.
    service._reranker.predict.return_value = [1.0, 4.0, -6.0]
    return service


def test_rerank_reorders_by_cross_encoder_score(reranking_service):
    results = reranking_service.retrieve("What triggers the Lambda functions?")

    assert [r["section"] for r in results] == ["Layout", "Lambda", "Monitoring"]


def test_rerank_keeps_vector_score_and_returns_probabilities(reranking_service):
    top = reranking_service.retrieve("What triggers the Lambda functions?")[0]

    assert top["vector_score"] == 0.40
    assert 0.98 < top["score"] < 0.99          # sigmoid(4.0)


def test_rerank_casts_a_wider_net_then_trims_to_top_k(reranking_service):
    results = reranking_service.retrieve("question", top_k=2)

    reranking_service.vector_search.assert_called_once_with(
        "question", top_k=10, source=None
    )
    assert len(results) == 2


def test_rerank_threshold_filters_low_relevance(reranking_service):
    results = reranking_service.retrieve_relevant("question", min_score=0.5)

    assert [r["section"] for r in results] == ["Layout", "Lambda"]


def test_without_reranker_no_cross_encoder_is_loaded(tmp_path):
    service = make_service(tmp_path)
    service.vector_search = MagicMock(return_value=list(CANDIDATES))

    with patch("sentence_transformers.CrossEncoder", create=True) as cross_encoder:
        results = service.retrieve("question")

    cross_encoder.assert_not_called()
    assert results == CANDIDATES


def test_each_mode_gets_its_own_default_threshold(tmp_path):
    assert make_service(tmp_path).min_score == COSINE_MIN_SCORE
    assert (
        make_service(tmp_path, reranker_model_name="x").min_score
        == RERANK_MIN_SCORE
    )
    assert make_service(tmp_path, min_score=0.7).min_score == 0.7


def test_database_mode_searches_the_store_with_the_source_filter(tmp_path):
    store = MagicMock()
    store.search.return_value = list(CANDIDATES)
    with patch("rag_service.SentenceTransformer") as model_class:
        service = RAGService(store=store)
    model_class.return_value.encode.return_value = [0.1, 0.2]

    results = service.retrieve("question", top_k=3, source="platform_runbook.md")

    store.search.assert_called_once_with(
        [0.1, 0.2], top_k=3, source="platform_runbook.md"
    )
    assert results == CANDIDATES


def test_source_filter_without_database_is_rejected(tmp_path):
    service = make_service(tmp_path)
    with pytest.raises(ValueError):
        service.retrieve("question", source="platform_runbook.md")
