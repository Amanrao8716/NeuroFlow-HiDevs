
"""NeuroFlow background worker."""

import asyncio
import logging
import signal

import redis.asyncio as redis

from config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("neuroflow.worker")


async def main() -> None:
    """Wait for jobs from Redis and shut down cleanly."""
    client = redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Some platforms do not support asyncio signal handlers.
            pass

    try:
        await client.ping()
        logger.info("Worker connected to Redis.")

        while not stop_event.is_set():
            try:
                result = await client.blpop("neuroflow:jobs", timeout=2)
                if result:
                    _, job = result
                    logger.info("Received job: %s", job)
                    logger.warning(
                        "Job processing is not implemented yet; "
                        "the job was removed from the queue."
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker error; retrying shortly.")
                await asyncio.sleep(2)
    finally:
        await client.aclose()
        logger.info("Worker stopped.")


if __name__ == "__main__":
    asyncio.run(main())
