"""Tests for the full answer path. Ollama is faked, so these run offline."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag_service import RAGService

PASSAGE = {
    "text": "Deployment\nWe use blue-green deployments.",
    "score": 0.82,
    "source": "guide.md",
    "section": "Deployment",
    "chunk_id": "guide.md#0",
}


@pytest.fixture
def rag_service(tmp_path):
    document = tmp_path / "guide.md"
    document.write_text(
        "# Deployment\nWe use blue-green deployments.\n", encoding="utf-8"
    )
    with patch("rag_service.SentenceTransformer"):
        yield RAGService(document_path=str(document))


def fake_ollama(reply=None, error=None):
    """Stand-in for httpx.AsyncClient that returns `reply` or raises `error`."""
    response = MagicMock()
    response.json.return_value = {"message": {"content": reply}}

    client = MagicMock()
    client.post = AsyncMock(return_value=response, side_effect=error)

    client_class = MagicMock()
    client_class.return_value.__aenter__ = AsyncMock(return_value=client)
    client_class.return_value.__aexit__ = AsyncMock(return_value=False)
    return client_class, client


def ask(service, question="How do we deploy?"):
    return asyncio.run(
        service.answer(question, "http://localhost:11434", "test-model")
    )


def test_no_relevant_passages_skips_the_model(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=[])
    client_class, client = fake_ollama(reply="should never be used")

    with patch("rag_service.httpx.AsyncClient", client_class):
        result = ask(rag_service, "What database do we use?")

    assert result["sources"] == []
    assert "don't know" in result["answer"]
    client.post.assert_not_called()


def test_answer_sends_labelled_context_and_returns_sources(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=[PASSAGE])
    client_class, client = fake_ollama(reply="Blue-green. [guide.md#0]")

    with patch("rag_service.httpx.AsyncClient", client_class):
        result = ask(rag_service)

    prompt = client.post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "[guide.md#0]" in prompt
    assert "blue-green deployments" in prompt
    assert result["answer"] == "Blue-green. [guide.md#0]"
    assert result["sources"] == [PASSAGE]


def test_ollama_failure_is_raised_for_the_api_to_handle(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=[PASSAGE])
    client_class, _ = fake_ollama(error=httpx.ConnectError("Ollama is down"))

    with patch("rag_service.httpx.AsyncClient", client_class):
        with pytest.raises(httpx.HTTPError):
            ask(rag_service)
