# Agent Harness

A local development scaffold for the agent stack described in [stack.md](./stack.md). It provides a FastAPI/Pydantic-AI chat API, a LiteLLM gateway with `agent-main` and `agent-worker` aliases, and a configurable OpenAI-compatible connection to a separately hosted `gemini-web2api-go` endpoint.

The Go proxy is intentionally external: its repository, image, authentication, cookies, and network policy are deployment-specific. The scaffold does not automate browser sessions or provision upstream credentials. Pydantic-AI history processing masks oversized older tool results; code-mode/MCP discovery and delegated workers remain integration points, not included services.

## Run locally with Docker Compose

1. Install Docker Desktop with Compose support and start it.
2. Copy `.env.example` to `.env`. Replace the sample keys and set `GEMINI_WEB2API_BASE_URL` and `GEMINI_WEB2API_API_KEY` for your OpenAI-compatible proxy.
3. If the proxy runs directly on this Windows host, the example `host.docker.internal` address is suitable with Docker Desktop. Otherwise use a URL reachable from the Compose network.
4. Start the stack:

   ```powershell
   docker compose up --build
   ```

5. Check readiness and send a request:

   ```powershell
   Invoke-RestMethod http://localhost:8000/health
   Invoke-RestMethod -Method Post http://localhost:8000/chat `
     -Headers @{ "X-API-Key" = "your-AGENT_API_KEY" } `
     -ContentType "application/json" `
     -Body '{"message":"Say hello"}'
   ```

The API is bound to localhost. Chat requests require `X-API-Key`; sessions are in-memory and are cleared when the API container restarts. Use `DELETE /sessions/{session_id}` to clear an individual conversation. The gateway disables message-content logging; avoid enabling request/response logging in deployments that may handle sensitive data.

The configured upstream model is `gemini-2.5-flash`. Change both aliases in `litellm_config.yaml` if the proxy exposes a different model name. The worker alias is reserved for delegated workloads and currently points to the same upstream model.

## Run tests without Docker or upstream credentials

With Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

Tests cover context compaction, preservation of stored history, health checks, and API authentication/validation. They do not prove that the configured proxy, upstream model, cookies, or any MCP/code-mode services are reachable.
