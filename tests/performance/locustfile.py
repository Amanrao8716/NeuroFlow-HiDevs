"""Authenticated, bounded RAG load scenario for a dedicated test deployment.

Required environment: NEUROFLOW_LOAD_CLIENT_ID, NEUROFLOW_LOAD_CLIENT_SECRET,
NEUROFLOW_LOAD_PIPELINE_ID. Set these in the shell/secret manager, never in Git.
"""

from __future__ import annotations

import os
from uuid import UUID

from locust import HttpUser, between, task


class NeuroFlowUser(HttpUser):
    """Exercise query generation and progressive streaming on a known pipeline."""

    # Keep aggregate request volume below the app's per-IP query throttle at
    # the documented 10-user profile. Tune only against a dedicated deployment.
    wait_time = between(20, 25)

    def on_start(self) -> None:
        client_id = os.environ.get("NEUROFLOW_LOAD_CLIENT_ID", "")
        client_secret = os.environ.get("NEUROFLOW_LOAD_CLIENT_SECRET", "")
        pipeline_id = os.environ.get("NEUROFLOW_LOAD_PIPELINE_ID", "")
        if not client_id or not client_secret:
            raise RuntimeError("Set dedicated NEUROFLOW_LOAD_CLIENT_ID and _SECRET values.")
        if not pipeline_id:
            raise RuntimeError("Set NEUROFLOW_LOAD_PIPELINE_ID to an active test pipeline.")
        self.pipeline_id = str(UUID(pipeline_id))
        response = self.client.post(
            "/auth/token",
            json={"client_id": client_id, "client_secret": client_secret},
            name="/auth/token",
        )
        if response.status_code != 200:
            raise RuntimeError(f"Test client authentication failed: HTTP {response.status_code}")
        self.client.headers.update(
            {"Authorization": f"Bearer {response.json()['access_token']}"}
        )

    @task(3)
    def query(self) -> None:
        with self.client.post(
            "/query",
            json={
                "query": "Summarize the key points in the indexed deployment test document.",
                "pipeline_id": self.pipeline_id,
                "stream": False,
            },
            name="/query (complete)",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            body = response.json()
            if not body.get("run_id") or not body.get("answer"):
                response.failure("Response missing run_id or answer")

    @task(1)
    def stream_query(self) -> None:
        with self.client.post(
            "/query",
            json={
                "query": "List one important fact from the indexed deployment test document.",
                "pipeline_id": self.pipeline_id,
                "stream": True,
            },
            name="/query (stream init)",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            run_id = response.json().get("run_id")
        if not run_id:
            return
        with self.client.get(
            f"/query/{run_id}/stream",
            name="/query/{run_id}/stream",
            stream=True,
            catch_response=True,
            timeout=180,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            saw_event = False
            for line in response.iter_lines():
                if line and line.startswith(b"data:"):
                    saw_event = True
                if line and b'"type": "done"' in line:
                    break
            if not saw_event:
                response.failure("No SSE data events received")
