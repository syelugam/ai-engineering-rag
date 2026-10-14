"""Tests for the full answer path. Ollama is faked, so these run offline."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag_service import ModelOutputError, RAGService, parse_model_answer

PASSAGES = [
    {"text": "Deployment\nWe use blue-green deployments.", "score": 0.82,
     "source": "guide.md", "section": "Deployment", "chunk_id": "guide.md#0"},
    {"text": "Security\nLeast privilege.", "score": 0.40,
     "source": "guide.md", "section": "Security", "chunk_id": "guide.md#1"},
]


@pytest.fixture
def rag_service(tmp_path):
    document = tmp_path / "guide.md"
    document.write_text("# Deployment\nWe use blue-green deployments.\n", encoding="utf-8")
    with patch("rag_service.SentenceTransformer"):
        yield RAGService(document_path=str(document))


def reply(answerable=True, answer="", citations=()):
    return json.dumps(
        {"answerable": answerable, "answer": answer, "citations": list(citations)}
    )


def fake_ollama(*replies, error=None):
    """Stand-in for httpx.AsyncClient returning each reply in turn, or raising error."""
    responses = []
    for content in replies:
        response = MagicMock()
        response.json.return_value = {"message": {"content": content}}
        responses.append(response)

    client = MagicMock()
    client.post = AsyncMock(side_effect=error or responses)

    client_class = MagicMock()
    client_class.return_value.__aenter__ = AsyncMock(return_value=client)
    client_class.return_value.__aexit__ = AsyncMock(return_value=False)
    return client_class, client


def ask(service, question="How do we deploy?"):
    return asyncio.run(service.answer(question, "http://localhost:11434", "test-model"))


def test_no_relevant_passages_skips_the_model(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=[])
    client_class, client = fake_ollama(reply(answer="never used"))

    with patch("rag_service.httpx.AsyncClient", client_class):
        result = ask(rag_service, "What database do we use?")

    assert result["answerable"] is False
    assert result["refusal_reason"] == "no_relevant_passages"
    client.post.assert_not_called()


def test_only_cited_passages_are_returned_as_sources(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, client = fake_ollama(
        reply(answer="Blue-green deployments.", citations=["guide.md#0"])
    )

    with patch("rag_service.httpx.AsyncClient", client_class):
        result = ask(rag_service)

    assert result["answerable"] is True
    assert result["answer"] == "Blue-green deployments."
    assert result["citations"] == ["guide.md#0"]
    assert result["sources"] == [PASSAGES[0]]      # Security passage not cited


def test_request_asks_for_json_schema_with_labelled_context(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, client = fake_ollama(reply(answer="x", citations=["guide.md#0"]))

    with patch("rag_service.httpx.AsyncClient", client_class):
        ask(rag_service)

    body = client.post.call_args.kwargs["json"]
    # answerable comes last so the model drafts the answer before judging it
    assert list(body["format"]["properties"]) == ["answer", "citations", "answerable"]
    assert body["options"]["temperature"] == 0
    assert body["think"] is False
    assert "[guide.md#0]" in body["messages"][0]["content"]


def test_model_saying_unanswerable_returns_no_sources(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, _ = fake_ollama(reply(answerable=False))

    with patch("rag_service.httpx.AsyncClient", client_class):
        result = ask(rag_service, "What Kubernetes version?")

    assert result["sources"] == []
    assert result["refusal_reason"] == "model_said_unanswerable"


def test_invented_citations_are_dropped(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, _ = fake_ollama(
        reply(answer="Blue-green.", citations=["guide.md#0", "made_up.md#9"])
    )

    with patch("rag_service.httpx.AsyncClient", client_class):
        result = ask(rag_service)

    assert result["citations"] == ["guide.md#0"]
    assert result["invented_citations"] == ["made_up.md#9"]


def test_answer_with_no_valid_citation_is_refused(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, _ = fake_ollama(reply(answer="Kubernetes 1.30.", citations=["nope#1"]))

    with patch("rag_service.httpx.AsyncClient", client_class):
        result = ask(rag_service)

    assert result["answerable"] is False
    assert result["refusal_reason"] == "no_valid_citations"
    assert "Kubernetes" not in result["answer"]


def test_broken_json_is_retried_once(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, client = fake_ollama(
        "Sure! The answer is blue-green.",
        reply(answer="Blue-green.", citations=["guide.md#0"]),
    )

    with patch("rag_service.httpx.AsyncClient", client_class):
        result = ask(rag_service)

    assert client.post.call_count == 2
    assert result["answerable"] is True
    first, second = (call.kwargs["json"] for call in client.post.call_args_list)
    assert "format" in first           # schema-constrained first
    assert "format" not in second      # free-form fallback


def test_parse_finds_json_surrounded_by_text():
    parsed = parse_model_answer('Here you go:\n{"answerable": false, "answer": "", "citations": []}\nThanks')

    assert parsed is not None and parsed.answerable is False


def test_error_message_includes_the_bad_reply(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, _ = fake_ollama("garbage one", "garbage two")

    with patch("rag_service.httpx.AsyncClient", client_class):
        with pytest.raises(ModelOutputError, match="garbage two"):
            ask(rag_service)


def test_broken_json_twice_raises(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, _ = fake_ollama("not json", "still not json")

    with patch("rag_service.httpx.AsyncClient", client_class):
        with pytest.raises(ModelOutputError):
            ask(rag_service)


def test_ollama_failure_is_raised_for_the_api_to_handle(rag_service):
    rag_service.retrieve_relevant = MagicMock(return_value=PASSAGES)
    client_class, _ = fake_ollama(error=httpx.ConnectError("Ollama is down"))

    with patch("rag_service.httpx.AsyncClient", client_class):
        with pytest.raises(httpx.HTTPError):
            ask(rag_service)


def test_parse_accepts_code_fenced_json():
    parsed = parse_model_answer('```json\n{"answerable": true, "answer": "a", "citations": []}\n```')

    assert parsed is not None and parsed.answer == "a"
