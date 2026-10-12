
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_rejects_empty_question():
    response = client.post(
        "/chat",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_chat_rejects_missing_question():
    response = client.post(
        "/chat",
        json={},
    )

    assert response.status_code == 422

def test_chat_returns_model_answer():
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "message": {
            "content": "Dependency injection supplies dependencies."
        }
    }

    fake_client = AsyncMock()
    fake_client.post.return_value = fake_response

    with patch("main.httpx.AsyncClient") as mock_client_class:
        mock_client_class.return_value.__aenter__.return_value = (
            fake_client
        )

        response = client.post(
            "/chat",
            json={"question": "Explain dependency injection."},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Dependency injection supplies dependencies.",
        "model": "gemma4:e2b",
    }
