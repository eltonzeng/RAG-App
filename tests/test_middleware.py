"""Tests for request-ID correlation, API-key auth, and the global 500 handler.

Uses httpx ASGITransport against the real app with a mocked pool; no live DB
or network.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.middleware import API_KEY_HEADER, REQUEST_ID_HEADER
from core.config import get_settings


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"cnt": 7})
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=conn)
    return pool


@pytest.fixture(autouse=True)
def inject_pool(mock_pool):
    app.state.pool = mock_pool


@pytest_asyncio.fixture
async def client():
    # raise_app_exceptions=False so the global exception handler's 500 response
    # is returned to the test instead of the exception propagating out.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestRequestId:
    @pytest.mark.asyncio
    async def test_generates_request_id_when_absent(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.headers.get(REQUEST_ID_HEADER)  # non-empty

    @pytest.mark.asyncio
    async def test_echoes_inbound_request_id(self, client: AsyncClient) -> None:
        resp = await client.get("/health", headers={REQUEST_ID_HEADER: "trace-123"})
        assert resp.headers[REQUEST_ID_HEADER] == "trace-123"


class TestApiKeyAuth:
    @pytest.mark.asyncio
    async def test_auth_off_when_unset(self, client: AsyncClient) -> None:
        """With no RAG_API_KEY configured, /ask needs no key (open dev mode)."""
        with (
            patch("api.routes.rewrite_query", new_callable=AsyncMock) as rw,
            patch("api.routes.hybrid_retrieve", new_callable=AsyncMock) as ret,
            patch("api.routes.rerank", new_callable=AsyncMock) as rr,
            patch("api.routes.generate", new_callable=AsyncMock) as gen,
        ):
            from api.models import MetadataFilters, QueryRewriteResult

            rw.return_value = QueryRewriteResult(queries=["q"], filters=MetadataFilters())
            ret.return_value = []
            rr.return_value = ([], False, False)
            gen.return_value = ("No info.", [])
            resp = await client.post("/ask", json={"query": "hi"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_401_without_key_when_configured(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAG_API_KEY", "s3cret")
        get_settings.cache_clear()
        resp = await client.post("/ask", json={"query": "hi"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_200_with_correct_key(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAG_API_KEY", "s3cret")
        get_settings.cache_clear()
        with (
            patch("api.routes.rewrite_query", new_callable=AsyncMock) as rw,
            patch("api.routes.hybrid_retrieve", new_callable=AsyncMock) as ret,
            patch("api.routes.rerank", new_callable=AsyncMock) as rr,
            patch("api.routes.generate", new_callable=AsyncMock) as gen,
        ):
            from api.models import MetadataFilters, QueryRewriteResult

            rw.return_value = QueryRewriteResult(queries=["q"], filters=MetadataFilters())
            ret.return_value = []
            rr.return_value = ([], False, False)
            gen.return_value = ("No info.", [])
            resp = await client.post(
                "/ask", json={"query": "hi"}, headers={API_KEY_HEADER: "s3cret"}
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_never_requires_key(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RAG_API_KEY", "s3cret")
        get_settings.cache_clear()
        resp = await client.get("/health")
        assert resp.status_code == 200


class TestGlobalExceptionHandler:
    @pytest.mark.asyncio
    async def test_unhandled_error_returns_correlated_500(self, client: AsyncClient) -> None:
        """An unexpected error yields a JSON 500 with the request id, no traceback."""
        with patch("api.routes.rewrite_query", new_callable=AsyncMock) as rw:
            rw.side_effect = RuntimeError("boom")
            resp = await client.post(
                "/ask", json={"query": "hi"}, headers={REQUEST_ID_HEADER: "trace-err"}
            )
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Internal server error"
        assert body["request_id"] == "trace-err"


class TestCors:
    @pytest.mark.asyncio
    async def test_preflight_allows_configured_origin(self, client: AsyncClient) -> None:
        resp = await client.options(
            "/ask",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code in (200, 204)
        assert resp.headers["access-control-allow-origin"] == "http://localhost:8501"
