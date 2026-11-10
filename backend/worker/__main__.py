"""NeuroFlow background worker entry point."""

from arq import run_worker
from worker.settings import WorkerSettings

if __name__ == "__main__":
    run_worker(WorkerSettings)
