"""NeuroFlow background worker entry point."""

import asyncio

from arq import run_worker

from worker.settings import WorkerSettings


if __name__ == "__main__":
    asyncio.run(run_worker(WorkerSettings))