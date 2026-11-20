import asyncio
import os
import secrets
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from harness.history import compact_history

api_key = os.environ.get("AGENT_API_KEY", "local-dev-change-me")
litellm_api_key = os.environ.get("LITELLM_API_KEY", "sk-local-dev-change-me")
litellm_base_url = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
model_name = os.environ.get("AGENT_MODEL", "agent-main")

model = OpenAIChatModel(
    model_name,
    provider=OpenAIProvider(base_url=litellm_base_url, api_key=litellm_api_key),
)
agent = Agent(
    model,
    system_prompt="You are a concise, helpful assistant. Do not claim tools or capabilities you do not have.",
    capabilities=[ProcessHistory(compact_history)],
)

app = FastAPI(title="Agent Harness", version="0.1.0")
sessions: dict[str, list[ModelMessage]] = {}
session_locks: dict[str, asyncio.Lock] = {}
session_locks_guard = asyncio.Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    session_id: str
    output: str


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def lock_for_session(session_id: str) -> asyncio.Lock:
    async with session_locks_guard:
        lock = session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            session_locks[session_id] = lock
        return lock


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    lock = await lock_for_session(session_id)
    async with lock:
        history = sessions.get(session_id, [])
        result = await agent.run(request.message, message_history=history)
        sessions[session_id] = compact_history(result.all_messages())
    return ChatResponse(session_id=session_id, output=str(result.output))


@app.delete(
    "/sessions/{session_id}",
    status_code=204,
    dependencies=[Depends(require_api_key)],
)
async def delete_session(session_id: str) -> None:
    async with session_locks_guard:
        lock = session_locks.get(session_id)
    if lock is None:
        sessions.pop(session_id, None)
        return
    async with lock:
        sessions.pop(session_id, None)
