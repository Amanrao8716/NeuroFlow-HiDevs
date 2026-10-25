"""Training Data Extraction, Validation, and SFT/DPO Formatter."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from db.pool import connection
from evaluation.metrics.faithfulness import FaithfulnessMetric
from pipelines.retrieval.context_assembler import TokenCounter

logger = logging.getLogger("neuroflow.finetuning.extractor")

# Strict citation regex requiring [Source N] where N >= 1
CITATION_PATTERN = re.compile(r"\[Source\s+([1-9]\d*)\]")

# Carefully tested PII regular expressions
# 1. Email pattern
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# 2. Phone number patterns (handles US/international variations without false-positive matching on years or short numbers)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?"            # Country code: +1, +44, etc.
    r"(?:\(\d{2,4}\)|\d{2,4})[-.\s]?"    # Area code: (555), 555-, 020
    r"\d{3,4}[-.\s]?\d{3,4}\b|"         # 7-8 digits separated by - or space
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"     # 10 digit plain/dashed: 555-123-4567, 5551234567
)


def contains_pii(text: str) -> bool:
    """Detect presence of email addresses or telephone numbers in text without leaking contents."""
    if not text:
        return False

    if EMAIL_PATTERN.search(text):
        return True

    # Check phone pattern, excluding simple 4-digit years like 2023, 2024
    for match in PHONE_PATTERN.finditer(text):
        matched = match.group(0).strip()
        # Count actual digits
        digits = re.sub(r"\D", "", matched)
        if len(digits) >= 10:  # Valid phone numbers have at least 10 digits
            return True
        if len(digits) in (7, 8) and ("-" in matched or "." in matched or " " in matched):
            return True

    return False


def has_valid_citation(text: str) -> bool:
    """Check if the text contains at least one [Source N] citation where N is a positive integer."""
    if not text:
        return False
    return bool(CITATION_PATTERN.search(text))


def extract_context_from_prompt(prompt: str | None) -> str:
    """Extract context block from stored prompt text, handling <context> tags or markdown."""
    if not prompt:
        return ""

    # Check for <context>...</context>
    match = re.search(r"<context>\s*([\s\S]*?)\s*</context>", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Check for [Context]...[Question]
    match2 = re.search(r"\[Context\]\s*([\s\S]*?)\s*(?:\[Question\]|User question:|$)", prompt, re.IGNORECASE)
    if match2:
        return match2.group(1).strip()

    return ""


@dataclass
class ValidatedTrainingPair:
    """Represents a validated training example ready for JSONL serialization."""

    id: UUID
    run_id: UUID
    system_prompt: str
    user_message: str
    context: str
    assistant_message: str
    quality_score: float
    faithfulness: float
    user_rating: int | None
    created_at: datetime
    token_count: int

    def to_openai_sft_message(self) -> dict[str, Any]:
        """Format as OpenAI SFT conversational JSON object."""
        if self.context:
            user_content = f"[Context]\n{self.context}\n\n[Question]\n{self.user_message}"
        else:
            user_content = f"[Context]\nNo context available.\n\n[Question]\n{self.user_message}"

        sys_content = self.system_prompt or (
            "You are a precise research assistant. Answer the user's question using ONLY "
            "the provided context. If the context does not contain enough information to answer "
            "fully, say so explicitly. For every factual claim, include a citation in the format "
            "[Source N]."
        )

        return {
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": self.assistant_message},
            ]
        }


class TrainingDataExtractor:
    """Extracts, validates, filters, and formats eligible training pairs from PostgreSQL."""

    def __init__(
        self,
        min_quality_score: float = 0.82,
        min_faithfulness: float = 0.80,
        min_tokens: int = 50,
        max_tokens: int = 2000,
        token_counter: TokenCounter | None = None,
        faithfulness_scorer: FaithfulnessMetric | None = None,
    ):
        self.min_quality_score = min_quality_score
        self.min_faithfulness = min_faithfulness
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.token_counter = token_counter or TokenCounter()
        self.faithfulness_scorer = faithfulness_scorer or FaithfulnessMetric()

    async def fetch_candidate_pairs(
        self,
        pipeline_id: UUID | None = None,
        db_conn: Any = None,
    ) -> list[dict[str, Any]]:
        """Fetch potential training pairs satisfying baseline database-level criteria."""
        query = """
            SELECT 
                tp.id,
                tp.run_id,
                tp.system_prompt,
                tp.user_message,
                tp.assistant_message,
                tp.quality_score,
                tp.included_in_job,
                tp.created_at,
                r.query as run_query,
                r.prompt as run_prompt,
                r.status as run_status,
                r.error_message as run_error,
                COALESCE(r.user_rating, e.user_rating) as user_rating,
                e.faithfulness as eval_faithfulness
            FROM training_pairs tp
            JOIN pipeline_runs r ON r.id = tp.run_id
            LEFT JOIN LATERAL (
                SELECT faithfulness, user_rating
                FROM evaluations
                WHERE run_id = tp.run_id
                ORDER BY evaluated_at DESC
                LIMIT 1
            ) e ON true
            WHERE tp.quality_score >= $1
              AND tp.included_in_job IS NULL
              AND (r.status IS NULL OR r.status != 'failed')
              AND (r.error_message IS NULL)
              AND (COALESCE(r.user_rating, e.user_rating) IS NULL OR COALESCE(r.user_rating, e.user_rating) >= 4)
        """
        params: list[Any] = [self.min_quality_score]

        if pipeline_id is not None:
            query += " AND r.pipeline_id = $2"
            params.append(pipeline_id)

        query += " ORDER BY tp.created_at ASC"

        if db_conn is not None:
            rows = await db_conn.fetch(query, *params)
        else:
            async with connection() as conn:
                rows = await conn.fetch(query, *params)

        return [dict(r) for r in rows]

    async def validate_and_filter_pairs(
        self,
        candidate_rows: list[dict[str, Any]],
    ) -> tuple[list[ValidatedTrainingPair], dict[str, int]]:
        """Run comprehensive in-memory validation and re-evaluation rules on candidates."""
        accepted: list[ValidatedTrainingPair] = []
        rejection_reasons: dict[str, int] = {
            "pii_detected": 0,
            "missing_citation": 0,
            "token_length_out_of_bounds": 0,
            "low_faithfulness": 0,
            "rejected_rating": 0,
        }

        for row in candidate_rows:
            user_msg = row.get("user_message") or row.get("run_query") or ""
            assistant_msg = row.get("assistant_message") or ""

            # 1. PII detection on query/prompt
            if contains_pii(user_msg):
                rejection_reasons["pii_detected"] += 1
                continue

            # 2. Citation check
            if not has_valid_citation(assistant_msg):
                rejection_reasons["missing_citation"] += 1
                continue

            # 3. Token length boundaries
            token_count = self.token_counter.count(assistant_msg)
            if token_count < self.min_tokens or token_count > self.max_tokens:
                rejection_reasons["token_length_out_of_bounds"] += 1
                continue

            # 4. User rating validation (ensure no rating <= 2 or < 4)
            rating = row.get("user_rating")
            if rating is not None and rating < 4:
                rejection_reasons["rejected_rating"] += 1
                continue

            # 5. Faithfulness verification
            faithfulness = row.get("eval_faithfulness")
            context_text = extract_context_from_prompt(row.get("run_prompt"))

            if faithfulness is None:
                # Must re-evaluate using existing evaluation infrastructure
                try:
                    score_res = await self.faithfulness_scorer.score(
                        answer=assistant_msg,
                        context=context_text,
                        query=user_msg,
                    )
                    faithfulness = score_res.score
                except Exception as eval_exc:
                    logger.warning("Faithfulness re-evaluation failed: %s", eval_exc)
                    faithfulness = 0.0

            if faithfulness is None or faithfulness <= self.min_faithfulness:
                rejection_reasons["low_faithfulness"] += 1
                continue

            accepted.append(
                ValidatedTrainingPair(
                    id=row["id"],
                    run_id=row["run_id"],
                    system_prompt=row.get("system_prompt") or "",
                    user_message=user_msg,
                    context=context_text,
                    assistant_message=assistant_msg,
                    quality_score=float(row["quality_score"]),
                    faithfulness=float(faithfulness),
                    user_rating=rating,
                    created_at=row["created_at"],
                    token_count=token_count,
                )
            )

        return accepted, rejection_reasons

    async def extract_for_job(
        self,
        job_id: UUID,
        pipeline_id: UUID | None = None,
        output_dir: str = "training_data",
        db_conn: Any = None,
    ) -> tuple[Path, dict[str, Any], list[ValidatedTrainingPair]]:
        """Extract, validate, persist JSONL, and return report and pairs (without marking yet)."""
        candidates = await self.fetch_candidate_pairs(pipeline_id=pipeline_id, db_conn=db_conn)
        eligible_count = len(candidates)

        accepted_pairs, rejection_reasons = await self.validate_and_filter_pairs(candidates)
        accepted_count = len(accepted_pairs)
        rejected_count = eligible_count - accepted_count

        # Compute date range and average quality score
        date_range = None
        avg_quality = None
        if accepted_pairs:
            dates = [p.created_at for p in accepted_pairs]
            min_date = min(dates).strftime("%Y-%m-%d")
            max_date = max(dates).strftime("%Y-%m-%d")
            date_range = f"{min_date} to {max_date}" if min_date != max_date else min_date
            avg_quality = sum(p.quality_score for p in accepted_pairs) / accepted_count

        # Create output directory safely and write JSONL
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{job_id}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            for pair in accepted_pairs:
                line = json.dumps(pair.to_openai_sft_message(), ensure_ascii=False)
                f.write(line + "\n")

        report = {
            "eligible_count": eligible_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "rejection_reasons": rejection_reasons,
            "average_quality_score": round(avg_quality, 4) if avg_quality is not None else None,
            "date_range": date_range,
        }

        return file_path, report, accepted_pairs

    async def mark_pairs_included_in_job(
        self,
        job_id: UUID,
        pair_ids: list[UUID],
        db_conn: Any = None,
    ) -> int:
        """Mark accepted training pairs as consumed by job in a safe database transaction."""
        if not pair_ids:
            return 0

        query = """
            UPDATE training_pairs
            SET included_in_job = $1
            WHERE id = ANY($2::uuid[])
              AND included_in_job IS NULL
        """
        if db_conn is not None:
            res = await db_conn.execute(query, job_id, pair_ids)
        else:
            async with connection() as conn:
                async with conn.transaction():
                    res = await conn.execute(query, job_id, pair_ids)

        # parse updated count from 'UPDATE <n>'
        try:
            return int(str(res).split()[-1])
        except Exception:
            return len(pair_ids)

    async def preview_eligible_pairs(
        self,
        pipeline_id: UUID | None = None,
        limit: int = 5,
        db_conn: Any = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Read-only preview of eligible training pairs without mutating state or writing files."""
        candidates = await self.fetch_candidate_pairs(pipeline_id=pipeline_id, db_conn=db_conn)
        accepted_pairs, _ = await self.validate_and_filter_pairs(candidates)

        preview_items: list[dict[str, Any]] = []
        for pair in accepted_pairs[:limit]:
            preview_items.append({
                "id": str(pair.id),
                "run_id": str(pair.run_id),
                "user_message": pair.user_message,
                "assistant_message": pair.assistant_message,
                "quality_score": round(pair.quality_score, 4) if pair.quality_score is not None else None,
                "faithfulness": round(pair.faithfulness, 4) if pair.faithfulness is not None else None,
                "user_rating": pair.user_rating,
                "token_count": pair.token_count,
                "citations_count": len(CITATION_PATTERN.findall(pair.assistant_message)),
                "has_valid_citations": True,
            })

        return preview_items, len(accepted_pairs)

    async def extract_dpo_pairs(
        self,
        pipeline_id: UUID | None = None,
        output_dir: str = "training_data",
        job_id: UUID | None = None,
        db_conn: Any = None,
    ) -> list[dict[str, str]]:
        """Extract DPO preference pairs (rating >= 4 as chosen, rating <= 2 as rejected) for matching queries."""
        query = """
            SELECT 
                r.query,
                r.generation,
                COALESCE(r.user_rating, e.user_rating) as rating,
                e.overall_score
            FROM pipeline_runs r
            LEFT JOIN LATERAL (
                SELECT user_rating, overall_score
                FROM evaluations
                WHERE run_id = r.id
                ORDER BY evaluated_at DESC
                LIMIT 1
            ) e ON true
            WHERE (r.status IS NULL OR r.status != 'failed')
              AND r.generation IS NOT NULL
              AND COALESCE(r.user_rating, e.user_rating) IS NOT NULL
        """
        params: list[Any] = []
        if pipeline_id is not None:
            query += " AND r.pipeline_id = $1"
            params.append(pipeline_id)

        query += " ORDER BY r.created_at ASC"

        if db_conn is not None:
            rows = await db_conn.fetch(query, *params)
        else:
            async with connection() as conn:
                rows = await conn.fetch(query, *params)

        # Group by normalized query: query -> list of (generation, rating)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            raw_query = (row["query"] or "").strip()
            if not raw_query or contains_pii(raw_query):
                continue
            norm_q = raw_query.lower()
            grouped.setdefault(norm_q, []).append({
                "original_prompt": raw_query,
                "generation": row["generation"],
                "rating": int(row["rating"]),
            })

        dpo_pairs: list[dict[str, str]] = []
        seen_prompts: set[str] = set()

        for norm_q, items in grouped.items():
            chosens = [it for it in items if it["rating"] >= 4 and has_valid_citation(it["generation"])]
            rejecteds = [it for it in items if it["rating"] <= 2]

            if not chosens or not rejecteds:
                continue

            if norm_q in seen_prompts:
                continue
            seen_prompts.add(norm_q)

            # Pick highest rated/best chosen and lowest rejected
            best_chosen = max(chosens, key=lambda x: x["rating"])
            worst_rejected = min(rejecteds, key=lambda x: x["rating"])

            # Never pair identical content
            if best_chosen["generation"].strip() == worst_rejected["generation"].strip():
                continue

            dpo_pairs.append({
                "prompt": best_chosen["original_prompt"],
                "chosen": best_chosen["generation"],
                "rejected": worst_rejected["generation"],
            })

        if job_id and dpo_pairs:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            dpo_file = out_path / f"{job_id}_dpo.jsonl"
            with open(dpo_file, "w", encoding="utf-8") as f:
                for pair in dpo_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        return dpo_pairs
