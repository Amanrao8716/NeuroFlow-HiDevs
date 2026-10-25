
"""arq configuration for the NeuroFlow ingestion worker."""

from arq.connections import RedisSettings

from config import settings
from worker.tasks import evaluate_pipeline_run, process_document, shutdown, startup


class WorkerSettings:
    """Configure background document processing and pipeline evaluation."""

    functions = [process_document, evaluate_pipeline_run]
    on_startup = startup
    on_shutdown = shutdown

    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        database=0,
    )

    max_jobs = 5
    job_timeout = 1800
    max_tries = 2
