
"""API endpoints for asynchronous document ingestion."""

import hashlib
import json
import logging
from pathlib import Path
from uuid import UUID

from arq import create_pool
from arq.connections import RedisSettings
from config import settings
from db.pool import pipeline_connection
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, HttpUrl

from resilience.backpressure import BackpressureGuard, QueueStatus

router = APIRouter(tags=["ingestion"])
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 100 * 1024 * 1024

SUPPORTED_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".csv": "csv",
}


class URLIngestRequest(BaseModel):
    """URL ingestion request payload."""

    url: HttpUrl
    pipeline_id: UUID


async def _enqueue(
    request: Request,
    document_id: UUID,
    pipeline_id: UUID,
) -> None:
    """Queue a document-processing job using ARQ."""
    del request  # ARQ manages its own Redis connection.

    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        database=0,
    )

    pool = await create_pool(redis_settings)

    try:
        job = await pool.enqueue_job(
            "process_document",
            str(document_id),
            str(pipeline_id),
        )

        if job is None:
            logger.info(
                "Document %s already has a queued job.",
                document_id,
            )
    finally:
        await pool.aclose()


@router.post("/ingest", status_code=202)
async def ingest_file(
    request: Request,
    pipeline_id: UUID = Form(...),
    file: UploadFile = File(...),
):
    """Accept a file, deduplicate it, and queue background processing."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    source_type = SUPPORTED_TYPES.get(extension)

    if source_type is None:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload PDF, DOCX, image, or CSV.",
        )

    file_bytes = await file.read(MAX_FILE_SIZE + 1)

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 100 MB.",
        )

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    content_hash = hashlib.sha256(file_bytes).hexdigest()

    async with pipeline_connection(pipeline_id) as conn:
        existing = await conn.fetchrow(
            """
            SELECT id, status
            FROM documents
            WHERE content_hash = $1 AND pipeline_id = $2
            """,
            content_hash,
            pipeline_id,
        )

        if existing:
            return {
                "document_id": str(existing["id"]),
                "status": existing["status"],
                "duplicate": True,
            }

        document = await conn.fetchrow(
            """
            INSERT INTO documents
                (filename, source_type, content_hash, pipeline_id,
                 source_bytes, status, metadata)
            VALUES ($1, $2, $3, $4, $5, 'queued', $6::jsonb)
            RETURNING id, status
            """,
            filename,
            source_type,
            content_hash,
            pipeline_id,
            file_bytes,
            json.dumps({"original_filename": filename}),
        )

        document_id = document["id"]

    # ── Backpressure check (before enqueueing) ───────────────────────
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is not None:
        bp = BackpressureGuard(redis_client)
        bp_result = await bp.check()
        if bp_result.should_reject:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={
                    "error": "ingestion_queue_full",
                    "queue_depth": bp_result.queue_depth,
                    "retry_after": bp_result.retry_after,
                    "document_id": str(document_id),
                    "status": "rejected",
                },
            )
    else:
        bp_result = None

    try:
        await _enqueue(request, document_id, pipeline_id)

    except Exception as exc:
        logger.exception(
            "Failed to enqueue document %s",
            document_id,
        )

        async with pipeline_connection(pipeline_id) as conn:
            await conn.execute(
                """
                UPDATE documents
                SET status = 'failed',
                    metadata = metadata || $2::jsonb
                WHERE id = $1
                """,
                document_id,
                json.dumps(
                    {"error": "Failed to enqueue ingestion job"}
                ),
            )

        raise HTTPException(
            status_code=503,
            detail="Could not queue document for processing.",
        ) from exc

    response_body: dict = {
        "document_id": str(document_id),
        "status": "queued",
        "duplicate": False,
    }
    if bp_result is not None and bp_result.status == QueueStatus.WARNING:
        response_body["warning"] = bp_result.warning
        response_body["estimated_wait_minutes"] = bp_result.estimated_wait_minutes
    return response_body


@router.post("/ingest/url", status_code=202)
async def ingest_url(
    payload: URLIngestRequest,
    request: Request,
):
    """Accept a URL and queue background processing."""
    normalized_url = str(payload.url)
    content_hash = hashlib.sha256(
        normalized_url.encode("utf-8")
    ).hexdigest()
    filename = (
        Path(normalized_url.split("?", 1)[0]).name or "webpage"
    )

    async with pipeline_connection(payload.pipeline_id) as conn:
        existing = await conn.fetchrow(
            """
            SELECT id, status
            FROM documents
            WHERE content_hash = $1 AND pipeline_id = $2
            """,
            content_hash,
            payload.pipeline_id,
        )

        if existing:
            return {
                "document_id": str(existing["id"]),
                "status": existing["status"],
                "duplicate": True,
            }

        document = await conn.fetchrow(
            """
            INSERT INTO documents
                (filename, source_type, content_hash, pipeline_id,
                 source_bytes, status, metadata)
            VALUES ($1, 'url', $2, $3, NULL, 'queued', $4::jsonb)
            RETURNING id, status
            """,
            filename,
            content_hash,
            payload.pipeline_id,
            json.dumps({"url": normalized_url}),
        )

        document_id = document["id"]

    # ── Backpressure check (before enqueueing) ───────────────────────
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is not None:
        bp = BackpressureGuard(redis_client)
        bp_result = await bp.check()
        if bp_result.should_reject:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={
                    "error": "ingestion_queue_full",
                    "queue_depth": bp_result.queue_depth,
                    "retry_after": bp_result.retry_after,
                    "document_id": str(document_id),
                    "status": "rejected",
                },
            )
    else:
        bp_result = None

    try:
        await _enqueue(request, document_id, payload.pipeline_id)

    except Exception as exc:
        logger.exception(
            "Failed to enqueue URL document %s",
            document_id,
        )

        async with pipeline_connection(payload.pipeline_id) as conn:
            await conn.execute(
                """
                UPDATE documents
                SET status = 'failed',
                    metadata = metadata || $2::jsonb
                WHERE id = $1
                """,
                document_id,
                json.dumps(
                    {"error": "Failed to enqueue ingestion job"}
                ),
            )

        raise HTTPException(
            status_code=503,
            detail="Could not queue URL for processing.",
        ) from exc

    response_body: dict = {
        "document_id": str(document_id),
        "status": "queued",
        "duplicate": False,
    }
    if bp_result is not None and bp_result.status == QueueStatus.WARNING:
        response_body["warning"] = bp_result.warning
        response_body["estimated_wait_minutes"] = bp_result.estimated_wait_minutes
    return response_body


@router.get("/documents/{document_id}")
async def get_document(
    document_id: UUID,
    pipeline_id: UUID,
    request: Request,
):
    """Return document status and chunk count within its tenant."""
    del request  # Redis is not needed for status lookup.

    async with pipeline_connection(pipeline_id) as conn:
        document = await conn.fetchrow(
            """
            SELECT id, filename, source_type, status, chunk_count,
                   metadata, created_at
            FROM documents
            WHERE id = $1 AND pipeline_id = $2
            """,
            document_id,
            pipeline_id,
        )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "document_id": str(document["id"]),
        "filename": document["filename"],
        "source_type": document["source_type"],
        "status": document["status"],
        "chunk_count": document["chunk_count"] or 0,
        "metadata": document["metadata"],
        "created_at": document["created_at"].isoformat(),
    }
