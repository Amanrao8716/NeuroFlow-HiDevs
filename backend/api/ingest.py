
"""API endpoints for asynchronous document ingestion."""

import hashlib
import json
import logging
from pathlib import Path
from uuid import UUID
import zipfile
from io import BytesIO

from arq import create_pool
from arq.connections import RedisSettings
from config import settings
from db.pool import pipeline_connection
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, HttpUrl

from resilience.backpressure import BackpressureGuard, QueueStatus

from backend.security.auth import get_current_user, require_scope
from backend.security.validators import validate_public_url
from backend.security.secrets import redact_secrets
from backend.security.validators import sanitize_text

router = APIRouter(tags=["ingestion"], dependencies=[Depends(get_current_user)])
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = settings.max_upload_size_bytes

SUPPORTED_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".csv": "csv",
}


def _validate_document(content_type: str | None, extension: str, data: bytes) -> None:
    """Check declared MIME and bounded file signatures before persistence."""
    mime_by_extension = {
        ".pdf": {"application/pdf"},
        ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        ".csv": {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"},
        ".jpg": {"image/jpeg"}, ".jpeg": {"image/jpeg"},
        ".png": {"image/png"}, ".webp": {"image/webp"},
    }
    if content_type not in mime_by_extension.get(extension, set()):
        raise HTTPException(status_code=415, detail="Declared MIME type does not match the file type.")
    valid = {
        ".pdf": data.startswith(b"%PDF-") and b"/JavaScript" not in data and b"/JS" not in data,
        ".docx": data.startswith(b"PK\x03\x04") and b"[Content_Types].xml" in data[:65536],
        ".csv": not data.startswith((b"MZ", b"\x7fELF", b"PK\x03\x04")) and b"\x00" not in data[:8192],
        ".jpg": data.startswith(b"\xff\xd8\xff"), ".jpeg": data.startswith(b"\xff\xd8\xff"),
        ".png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        ".webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
    }
    if not valid.get(extension, False):
        raise HTTPException(status_code=415, detail="File signature does not match a supported document.")
    if extension == ".docx":
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > 1000 or not any(item.filename == "[Content_Types].xml" for item in entries):
                    raise ValueError("Invalid archive layout.")
                expanded = 0
                for item in entries:
                    path = Path(item.filename)
                    if path.is_absolute() or ".." in path.parts or item.filename.lower().endswith((".zip", ".7z", ".rar")):
                        raise ValueError("Unsafe archive member.")
                    expanded += item.file_size
                    if item.file_size > 0 and item.compress_size == 0:
                        raise ValueError("Invalid compression ratio.")
                    if item.compress_size and item.file_size / item.compress_size > 100:
                        raise ValueError("Excessive compression ratio.")
                    if expanded > 200 * 1024 * 1024:
                        raise ValueError("Archive expands beyond limit.")
                bad_member = archive.testzip()
                if bad_member:
                    raise ValueError("Corrupt archive member.")
        except (zipfile.BadZipFile, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=415, detail="DOCX archive is malformed or unsafe.") from exc


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
    _scope=Depends(require_scope("ingest")),
):
    """Accept a file, deduplicate it, and queue background processing."""
    filename = sanitize_text(Path(file.filename or "").name)
    filename, detected_filename_secrets = redact_secrets(filename)
    if detected_filename_secrets:
        logger.warning("Sensitive value redacted from uploaded filename.")
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

    _validate_document(file.content_type, extension, file_bytes)

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
    _scope=Depends(require_scope("ingest")),
):
    """Accept a URL and queue background processing."""
    normalized_url = str(payload.url)
    try:
        validate_public_url(normalized_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="URL is not an allowed public destination.") from exc
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
    _scope=Depends(require_scope("query")),
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


@router.get("/documents")
async def list_documents(
    pipeline_id: UUID,
    limit: int = 100,
    offset: int = 0,
    _scope=Depends(require_scope("query")),
):
    """List documents for a given pipeline with status, chunk count, and metadata."""
    async with pipeline_connection(pipeline_id) as conn:
        rows = await conn.fetch(
            """
            SELECT id, filename, source_type, status, chunk_count,
                   metadata, created_at
            FROM documents
            WHERE pipeline_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            pipeline_id,
            limit,
            offset,
        )

    return [
        {
            "id": str(r["id"]),
            "document_id": str(r["id"]),
            "filename": r["filename"],
            "source_type": r["source_type"],
            "status": r["status"],
            "chunk_count": r["chunk_count"] or 0,
            "metadata": r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"] or "{}"),
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: UUID,
    pipeline_id: UUID | None = None,
    limit: int = 100,
    _scope=Depends(require_scope("query")),
):
    """Retrieve all chunks belonging to a document."""
    from db.pool import connection
    if connection is None:
        raise HTTPException(status_code=503, detail="Database pool unavailable")

    async with connection() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.document_id, c.content, c.chunk_index, c.token_count, c.metadata, c.created_at,
                   d.filename, d.source_type
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.document_id = $1
            ORDER BY c.chunk_index ASC
            LIMIT $2
            """,
            document_id,
            limit,
        )

    return [
        {
            "chunk_id": str(r["id"]),
            "document_id": str(r["document_id"]),
            "content": r["content"],
            "chunk_index": r["chunk_index"],
            "token_count": r["token_count"],
            "metadata": r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"] or "{}"),
            "filename": r["filename"],
            "source_type": r["source_type"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]


@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: UUID, _scope=Depends(require_scope("query"))):
    """Retrieve chunk details, content, and parent document metadata."""
    from db.pool import connection
    if connection is None:
        raise HTTPException(status_code=503, detail="Database pool unavailable")

    async with connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.id, c.document_id, c.content, c.chunk_index, c.token_count, c.metadata, c.created_at,
                   d.filename, d.source_type, d.pipeline_id
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = $1
            """,
            chunk_id,
        )

    if not row:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return {
        "chunk_id": str(row["id"]),
        "document_id": str(row["document_id"]),
        "content": row["content"],
        "chunk_index": row["chunk_index"],
        "token_count": row["token_count"],
        "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
        "filename": row["filename"],
        "source_type": row["source_type"],
        "pipeline_id": str(row["pipeline_id"]) if row["pipeline_id"] else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }
