from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, UserPromptPart

from harness import api as agent_api

client = TestClient(agent_api.app)


def test_health_endpoint_is_available_without_credentials() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_missing_api_key() -> None:
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 401


def test_chat_rejects_incorrect_api_key() -> None:
    response = client.post(
        "/chat",
        headers={"X-API-Key": "wrong"},
        json={"message": "hello"},
    )

    assert response.status_code == 401


def test_chat_rejects_empty_message() -> None:
    response = client.post(
        "/chat",
        headers={"X-API-Key": agent_api.api_key},
        json={"message": ""},
    )

    assert response.status_code == 422


def test_chat_reuses_session_history_and_can_clear_session(monkeypatch) -> None:
    calls = []

    async def fake_run(message, *, message_history):
        calls.append((message, list(message_history)))
        new_message = ModelRequest(parts=[UserPromptPart(content=message)])
        return SimpleNamespace(
            output=f"reply: {message}",
            all_messages=lambda: [*message_history, new_message],
        )

    monkeypatch.setattr(agent_api.agent, "run", fake_run)
    headers = {"X-API-Key": agent_api.api_key}

    first = client.post(
        "/chat",
        headers=headers,
        json={"message": "first", "session_id": "test-session"},
    )
    second = client.post(
        "/chat",
        headers=headers,
        json={"message": "second", "session_id": "test-session"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["output"] == "reply: second"
    assert len(calls[0][1]) == 0
    assert len(calls[1][1]) == 1

    deleted = client.delete("/sessions/test-session", headers=headers)

    assert deleted.status_code == 204
    assert "test-session" not in agent_api.sessions
