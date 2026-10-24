
"""Check and apply database schema migrations."""

from pathlib import Path

from db.pool import get_pool


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


async def apply_migrations() -> None:
    """Apply schema, RLS, and subsequent numbered migrations."""
    pool = get_pool()

    async with pool.acquire() as conn:
        schema_exists = await conn.fetchval(
            "SELECT to_regclass('public.documents') IS NOT NULL"
        )

        if not schema_exists:
            for filename in ("001_schema.sql", "002_rls.sql"):
                migration_file = MIGRATIONS_DIR / filename

                if not migration_file.is_file():
                    raise FileNotFoundError(
                        f"Migration file not found: {migration_file}"
                    )

                async with conn.transaction():
                    await conn.execute(
                        migration_file.read_text(encoding="utf-8")
                    )

            print("Database schema and RLS policies applied.")

        else:
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

            if not rls_enabled:
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

        # Apply numbered migrations after the initial schema.
        # Existing databases can safely run migrations that are
        # idempotent, such as 003_ingestion_status.sql.
        for migration_file in sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql")):
            if migration_file.name in (
                "001_schema.sql",
                "002_rls.sql",
            ):
                continue

            async with conn.transaction():
                await conn.execute(
                    migration_file.read_text(encoding="utf-8")
                )

            print(f"Applied migration: {migration_file.name}")
