
"""Check and apply database schema migrations."""

from pathlib import Path

from db.pool import get_pool


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


async def apply_migrations() -> None:
    """Apply schema and RLS SQL files if the schema is missing."""
    pool = get_pool()

    async with pool.acquire() as conn:
        schema_exists = await conn.fetchval(
            "SELECT to_regclass('public.documents') IS NOT NULL"
        )

        if schema_exists:
            # PostgreSQL initialization scripts may already have applied
            # the schema and RLS policies.
            rls_enabled = await conn.fetchval(
                """
                SELECT COALESCE(bool_and(relrowsecurity), FALSE)
                FROM pg_class
                WHERE oid IN (
                    to_regclass('public.documents'),
                    to_regclass('public.chunks'),
                    to_regclass('public.pipelines'),
                    to_regclass('public.pipeline_runs'),
                    to_regclass('public.evaluations'),
                    to_regclass('public.training_pairs'),
                    to_regclass('public.finetune_jobs')
                )
                """
            )

            if rls_enabled:
                print("Database schema and RLS policies already exist.")
                return

            rls_file = MIGRATIONS_DIR / "002_rls.sql"

            if not rls_file.is_file():
                raise FileNotFoundError(
                    f"RLS migration not found: {rls_file}"
                )

            async with conn.transaction():
                await conn.execute(
                    rls_file.read_text(encoding="utf-8")
                )

            print("RLS policies applied.")
            return

        schema_file = MIGRATIONS_DIR / "001_schema.sql"
        rls_file = MIGRATIONS_DIR / "002_rls.sql"

        for migration_file in (schema_file, rls_file):
            if not migration_file.is_file():
                raise FileNotFoundError(
                    f"Migration file not found: {migration_file}"
                )

        async with conn.transaction():
            await conn.execute(
                schema_file.read_text(encoding="utf-8")
            )
            await conn.execute(
                rls_file.read_text(encoding="utf-8")
            )

        print("Database schema and RLS policies applied.")
