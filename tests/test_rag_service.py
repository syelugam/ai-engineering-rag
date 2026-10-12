
from unittest.mock import MagicMock, patch

import pytest

from rag_service import RAGService


@pytest.fixture
def rag_service(tmp_path):
    document = tmp_path / "architecture.txt"
    document.write_text(
        "# Deployment\n"
        "We use blue-green deployments to release new versions.\n\n"
        "# Security\n"
        "AWS IAM roles follow least privilege.\n",
        encoding="utf-8",
    )

    with patch("rag_service.SentenceTransformer") as mock_model:
        model = mock_model.return_value

        # Use simple deterministic fake embeddings for testing.
        model.encode.side_effect = [
            [[1.0, 0.0], [0.0, 1.0]],
        ]

        service = RAGService(document_path=str(document))
        service.embedding_model = MagicMock()

        yield service


def test_retrieve_returns_passages(rag_service):
    rag_service.embedding_model.encode.return_value = [1.0, 0.0]

    with patch("rag_service.util.cos_sim") as mock_cos:
        mock_cos.return_value = MagicMock()
        mock_cos.return_value.__getitem__.return_value = MagicMock()
        mock_cos.return_value.__getitem__.return_value.topk.return_value = (
            MagicMock(
                values=MagicMock(tolist=lambda: [0.9]),
                indices=MagicMock(tolist=lambda: [0]),
            )
        )

        results = rag_service.retrieve("How do we deploy?")

    assert len(results) == 1
    assert "blue-green" in results[0]["text"]


def test_retrieve_relevant_filters_low_scores(rag_service):
    rag_service.retrieve = MagicMock(return_value=[
        {"text": "Relevant passage", "score": 0.75},
        {"text": "Unrelated passage", "score": 0.20},
    ])

    results = rag_service.retrieve_relevant(
        "How do we deploy?",
        min_score=0.40,
    )

    assert len(results) == 1
    assert results[0]["text"] == "Relevant passage"


def test_retrieve_relevant_returns_empty_when_all_scores_are_low(
    rag_service,
):
    rag_service.retrieve = MagicMock(return_value=[
        {"text": "Unrelated passage", "score": 0.20},
    ])

    results = rag_service.retrieve_relevant(
        "What database do we use?",
        min_score=0.40,
    )

    assert results == []
