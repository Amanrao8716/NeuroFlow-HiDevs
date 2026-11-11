
"""Async HTTP client for the NeuroFlow RAG API."""

from __future__ import annotations

import asyncio
import json as _json
import logging
from collections.abc import AsyncIterator, Generator
from pathlib import Path
from typing import Any, cast

import httpx
from httpx._types import RequestFiles

from neuroflow.exceptions import (
    AuthenticationError,
    IngestionError,
    NeuroFlowError,
    RateLimitError,
)
from neuroflow.exceptions import TimeoutError as NFTimeoutError
from neuroflow.models import (
    Citation,
    Document,
    EvaluationResult,
    Pipeline,
    QueryResult,
)

logger = logging.getLogger("neuroflow.sdk")

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_POLL_INTERVAL = 2.0
_DEFAULT_POLL_TIMEOUT = 300.0
_MAX_RETRIES = 3


class QueryResponseHandler:
    """Handle awaiting a query response and iterating over tokens."""

    def __init__(
        self,
        client: NeuroFlowClient,
        query: str,
        pipeline_id: str,
        stream: bool,
        **kwargs: object,
    ) -> None:
        self._client = client
        self._query = query
        self._pipeline_id = pipeline_id
        self._stream = stream
        self._kwargs = kwargs

    def __await__(
        self,
    ) -> Generator[Any, None, QueryResult | AsyncIterator[str]]:
        return self._client._execute_query(
            self._query,
            self._pipeline_id,
            stream=self._stream,
            **cast(dict[str, Any], self._kwargs),
        ).__await__()

    def __aiter__(self) -> AsyncIterator[str]:
        if not self._stream:
            raise TypeError(
                "Cannot async-iterate over query when stream=False. "
                "Use 'await client.query(...)' or pass 'stream=True'."
            )
        return self._client._stream_query_tokens(
            self._query,
            self._pipeline_id,
            **cast(dict[str, Any], self._kwargs),
        )


class NeuroFlowClient:
    """Async client for interacting with a NeuroFlow API server.

    Usage::

        async with NeuroFlowClient("http://localhost:8000", token="...") as client:
            doc = await client.ingest_file("report.pdf", pipeline_id="...")
            result = await client.query("What is X?", pipeline_id="...")
            print(result.answer)

            async for token in client.query("Stream this", pipeline_id="...", stream=True):
                print(token, end="", flush=True)
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        poll_timeout: float = _DEFAULT_POLL_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the NeuroFlow client.

        Args:
            base_url: Root URL of the NeuroFlow API.
            token: Pre-existing JWT bearer token.
            client_id: Client ID for automatic token exchange.
            client_secret: Client secret for automatic token exchange.
            timeout: Default HTTP request timeout in seconds.
            poll_interval: Seconds between status polls.
            poll_timeout: Maximum time to wait when polling.
            max_retries: Maximum retry attempts for HTTP 429 responses.
            transport: Custom HTTP transport for testing.
        """
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._max_retries = max_retries
        self._transport = transport
        self._http: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                transport=self._transport,
            )
        return self._http

    async def __aenter__(self) -> NeuroFlowClient:
        await self._ensure_client()
        if self._token is None and self._client_id and self._client_secret:
            await self._authenticate()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def _authenticate(self) -> None:
        """Exchange client credentials for a JWT token."""
        http = await self._ensure_client()
        resp = await http.post(
            "/auth/token",
            json={"client_id": self._client_id, "client_secret": self._client_secret},
        )
        if resp.status_code in (401, 403):
            raise AuthenticationError("Invalid client credentials.")
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
        files: RequestFiles | None = None,
        retry_on_429: bool = True,
    ) -> httpx.Response:
        """Send an HTTP request with automatic 429 retry and error mapping."""
        http = await self._ensure_client()
        attempts = 0
        while True:
            attempts += 1
            resp = await http.request(
                method,
                path,
                json=json,
                params=params,
                data=data,
                files=files,
                headers=self._auth_headers(),
            )

            if resp.status_code == 429 and retry_on_429 and attempts <= self._max_retries:
                retry_after_hdr = resp.headers.get("Retry-After", "1")
                try:
                    retry_after = float(retry_after_hdr)
                except (ValueError, TypeError):
                    retry_after = 1.0
                logger.warning(
                    "Rate limited (429). Retry %d/%d after %.1fs",
                    attempts,
                    self._max_retries,
                    retry_after,
                )
                await asyncio.sleep(retry_after)
                continue

            if resp.status_code == 429:
                retry_after_val = resp.headers.get("Retry-After")
                retry_float = None
                if retry_after_val:
                    try:
                        retry_float = float(retry_after_val)
                    except ValueError:
                        pass
                raise RateLimitError(retry_after=retry_float)

            if resp.status_code in (401, 403):
                raise AuthenticationError(resp.text)

            if resp.status_code >= 400:
                raise NeuroFlowError(
                    f"HTTP {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                )

            return resp

    async def _poll_document(self, document_id: str, pipeline_id: str) -> Document:
        """Poll until document processing leaves queued or processing status."""
        elapsed = 0.0
        while elapsed < self._poll_timeout:
            resp = await self._request(
                "GET",
                f"/documents/{document_id}",
                params={"pipeline_id": pipeline_id},
            )
            body = resp.json()
            status = body.get("status", "")
            if status not in ("queued", "processing"):
                if status == "failed":
                    raise IngestionError(
                        f"Document {document_id} processing failed.",
                        status_code=500,
                    )
                return Document(
                    document_id=body.get("document_id", document_id),
                    status=status,
                    filename=body.get("filename"),
                    source_type=body.get("source_type"),
                    chunk_count=body.get("chunk_count", 0),
                    metadata=body.get("metadata", {}),
                    created_at=body.get("created_at"),
                )
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

        raise NFTimeoutError(
            f"Document {document_id} did not finish within {self._poll_timeout}s."
        )

    async def ingest_file(
        self,
        file_path: str | Path,
        pipeline_id: str | None = None,
    ) -> Document:
        """Upload a local file for ingestion and poll until processing completes."""
        if not pipeline_id:
            raise IngestionError("pipeline_id must be provided for document ingestion.")

        path = Path(file_path)
        is_file = await asyncio.to_thread(path.is_file)

        if not is_file:
            raise IngestionError(f"File not found: {path}")

        file_content = await asyncio.to_thread(path.read_bytes)

        resp = await self._request(
            "POST",
            "/ingest",
            files={"file": (path.name, file_content)},
            data={"pipeline_id": pipeline_id},
            retry_on_429=False,
        )

        body = resp.json()
        doc_id = body["document_id"]

        if body.get("duplicate"):
            return Document(
                document_id=doc_id,
                status=body.get("status", "completed"),
                duplicate=True,
            )

        return await self._poll_document(doc_id, pipeline_id)

    async def ingest_url(
        self,
        url: str,
        pipeline_id: str | None = None,
    ) -> Document:
        """Submit a URL for ingestion and poll until processing completes."""
        if not pipeline_id:
            raise IngestionError("pipeline_id must be provided for URL ingestion.")

        resp = await self._request(
            "POST",
            "/ingest/url",
            json={"url": url, "pipeline_id": pipeline_id},
        )
        body = resp.json()
        doc_id = body["document_id"]

        if body.get("duplicate"):
            return Document(
                document_id=doc_id,
                status=body.get("status", "completed"),
                duplicate=True,
            )

        return await self._poll_document(doc_id, pipeline_id)

    def query(
        self,
        query: str,
        pipeline_id: str,
        stream: bool = False,
        **kwargs: object,
    ) -> QueryResponseHandler:
        """Execute a RAG query."""
        return QueryResponseHandler(
            self,
            query=query,
            pipeline_id=pipeline_id,
            stream=stream,
            **kwargs,
        )

    async def _execute_query(
        self,
        query: str,
        pipeline_id: str,
        *,
        stream: bool = False,
        top_k: int = 10,
        rerank: bool = True,
        **kwargs: object,
    ) -> QueryResult | AsyncIterator[str]:
        if stream:
            return self._stream_query_tokens(
                query=query,
                pipeline_id=pipeline_id,
                top_k=top_k,
                rerank=rerank,
                **kwargs,
            )

        payload: dict[str, object] = {
            "query": query,
            "pipeline_id": pipeline_id,
            "stream": False,
            "top_k": top_k,
            "rerank": rerank,
            **kwargs,
        }
        resp = await self._request("POST", "/query", json=payload)
        body = resp.json()
        citations = [Citation(**c) for c in body.get("citations", [])]
        return QueryResult(
            run_id=body["run_id"],
            query=body.get("query", query),
            answer=body.get("answer", ""),
            query_type=body.get("query_type", ""),
            citations=citations,
            sources=body.get("sources", []),
            chunks_used=body.get("chunks_used", 0),
            input_tokens=body.get("input_tokens", 0),
            output_tokens=body.get("output_tokens", 0),
            latency_ms=body.get("latency_ms", 0),
            model_used=body.get("model_used", ""),
            status=body.get("status", "completed"),
        )

    async def _stream_query_tokens(
        self,
        query: str,
        pipeline_id: str,
        *,
        top_k: int = 10,
        rerank: bool = True,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        payload: dict[str, object] = {
            "query": query,
            "pipeline_id": pipeline_id,
            "stream": True,
            "top_k": top_k,
            "rerank": rerank,
            **kwargs,
        }
        resp = await self._request("POST", "/query", json=payload)
        body = resp.json()
        stream_url = body.get("stream_url", f"/query/{body['run_id']}/stream")
        async for token in self._stream_tokens(stream_url):
            yield token

    async def _stream_tokens(self, stream_url: str) -> AsyncIterator[str]:
        """Yield token strings from an SSE stream."""
        http = await self._ensure_client()
        async with http.stream(
            "GET",
            stream_url,
            headers=self._auth_headers(),
            timeout=None,
        ) as resp:
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.rstrip("\r")

                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            return

                        try:
                            event = _json.loads(data_str)
                        except _json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")
                        if event_type == "token":
                            content = event.get("content", "")
                            if content:
                                yield content
                        elif event_type == "error":
                            raise NeuroFlowError(
                                event.get("message", "Stream error"),
                            )

    async def get_evaluation(
        self,
        run_id: str,
        wait: bool = True,
        *,
        pipeline_id: str | None = None,
    ) -> EvaluationResult:
        """Fetch the evaluation result for a pipeline run."""
        params: dict[str, str] = {}
        if pipeline_id:
            params["pipeline_id"] = pipeline_id

        elapsed = 0.0
        while True:
            try:
                resp = await self._request(
                    "GET",
                    f"/runs/{run_id}/evaluation",
                    params=params or None,
                )
                body = resp.json()
                return EvaluationResult(
                    id=body.get("id", ""),
                    run_id=body.get("run_id", run_id),
                    faithfulness=body.get("faithfulness"),
                    answer_relevance=body.get("answer_relevance"),
                    context_precision=body.get("context_precision"),
                    context_recall=body.get("context_recall"),
                    overall_score=body.get("overall_score"),
                    judge_model=body.get("judge_model"),
                    user_rating=body.get("user_rating"),
                    calibration_needed=body.get("calibration_needed", False),
                )
            except NeuroFlowError as exc:
                if not wait or exc.status_code != 404:
                    raise
                if elapsed >= self._poll_timeout:
                    raise NFTimeoutError(
                        f"Evaluation for run {run_id} not ready within {self._poll_timeout}s."
                    ) from exc
                await asyncio.sleep(self._poll_interval)
                elapsed += self._poll_interval

    async def list_pipelines(self) -> list[Pipeline]:
        """Return all active pipelines."""
        resp = await self._request("GET", "/pipelines")
        pipelines = []
        for item in resp.json():
            pipelines.append(
                Pipeline(
                    id=item["id"],
                    name=item["name"],
                    description=item.get("description"),
                    version=item.get("version", 1),
                    status=item.get("status", "active"),
                    config=item.get("config", {}),
                    created_at=item.get("created_at"),
                    updated_at=item.get("updated_at"),
                )
            )
        return pipelines

    async def create_pipeline(
        self,
        config: dict[str, Any] | str,
        *,
        description: str | None = None,
        **kwargs: object,
    ) -> Pipeline:
        """Create a new RAG pipeline."""
        if isinstance(config, str):
            payload: dict[str, Any] = {"name": config}
            if description is not None:
                payload["description"] = description
            payload.update(kwargs)
        elif isinstance(config, dict):
            payload = dict(config)
            if description is not None and "description" not in payload:
                payload["description"] = description
            payload.update(kwargs)
        else:
            raise TypeError("config must be a dictionary or pipeline name string.")

        resp = await self._request("POST", "/pipelines", json=payload)
        body = resp.json()
        return Pipeline(
            id=body["id"],
            name=body["name"],
            description=body.get("description"),
            version=body.get("version", 1),
            status=body.get("status", "active"),
            config=body.get("config", {}),
            created_at=body.get("created_at"),
            updated_at=body.get("updated_at"),
        )
