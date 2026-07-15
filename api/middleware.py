"""Request-ID correlation middleware and optional API-key authentication.

- ``RequestIdMiddleware`` assigns every request a correlation id (honoring an
  inbound ``X-Request-ID`` or minting a uuid4), stores it in the logging context
  var so every log line for the request carries it, and echoes it on the
  response.
- ``verify_api_key`` is a FastAPI dependency that enforces an ``X-API-Key``
  header only when ``RAG_API_KEY`` is configured — so local development stays
  open while a deployment can lock the mutating/expensive endpoints down.
"""

import uuid

from fastapi import Header, HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.config import get_settings
from core.logging import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"
_REQUEST_ID_HEADER_LOWER = REQUEST_ID_HEADER.lower().encode()


class RequestIdMiddleware:
    """Assign/propagate a per-request correlation id (pure ASGI).

    Pure ASGI (rather than BaseHTTPMiddleware) so the request_id contextvar set
    here is visible to every log record emitted while handling the request, and
    the id is stashed on ``scope["state"]`` so the global exception handler can
    read it even after this middleware unwinds.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        inbound = headers.get(_REQUEST_ID_HEADER_LOWER, b"").decode()
        request_id = inbound or uuid.uuid4().hex

        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"].append((_REQUEST_ID_HEADER_LOWER, request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            request_id_var.reset(token)


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require a valid X-API-Key header when RAG_API_KEY is configured.

    No-op when RAG_API_KEY is unset (open local development). Raises 401 when a
    key is configured and the header is missing or wrong.

    Args:
        x_api_key: The inbound X-API-Key header value, if any.

    Raises:
        HTTPException: 401 if a key is required and not correctly supplied.
    """
    configured = get_settings().rag_api_key
    if configured and x_api_key != configured:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
