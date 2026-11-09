"""MLflow Experiment Tracking and Artifact Logging for Fine-Tuning Jobs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from config import settings

logger = logging.getLogger("neuroflow.finetuning.tracker")


class MLflowTracker:
    """Manages MLflow runs, parameters, metrics, and artifact tracking for fine-tuning jobs."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "neuroflow-finetuning",
    ):
        self.tracking_uri = (tracking_uri or settings.mlflow_tracking_uri).rstrip("/")
        self.experiment_name = experiment_name
        self._client: Any = None
        self._experiment_id: str | None = None

    def _get_client(self) -> Any:
        """Lazy-initialize and return MlflowClient."""
        if self._client is None:
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(self.tracking_uri)
                self._client = MlflowClient(tracking_uri=self.tracking_uri)
            except Exception as exc:
                logger.error("Failed to initialize MLflow client with tracking URI %s: %s", self.tracking_uri, exc)
                raise RuntimeError(f"MLflow service unavailable at {self.tracking_uri}: {exc}") from exc
        return self._client

    def _ensure_experiment(self) -> str:
        """Retrieve existing experiment or create a new one."""
        if self._experiment_id:
            return self._experiment_id

        client = self._get_client()
        try:
            exp = client.get_experiment_by_name(self.experiment_name)
            if exp is not None:
                self._experiment_id = exp.experiment_id
            else:
                self._experiment_id = client.create_experiment(self.experiment_name)
        except Exception as exc:
            logger.error("Error creating/retrieving MLflow experiment '%s': %s", self.experiment_name, exc)
            raise RuntimeError(f"MLflow experiment setup failed: {exc}") from exc

        return self._experiment_id

    def create_run(
        self,
        job_id: UUID,
        base_model: str,
        training_pair_count: int,
        average_quality_score: float | None,
        date_range: str | None,
        training_file_path: Path | str,
    ) -> tuple[str, str]:
        """Create an MLflow run named 'finetune-{job_id}', log parameters, and log JSONL artifact.

        Returns:
            Tuple of (mlflow_run_id, mlflow_run_url).
        """
        client = self._get_client()
        experiment_id = self._ensure_experiment()

        try:
            run = client.create_run(
                experiment_id=experiment_id,
                run_name=f"finetune-{job_id}",
                tags={
                    "job_id": str(job_id),
                    "pipeline_type": "fine-tuning",
                    "base_model": base_model,
                },
            )
            run_id = run.info.run_id

            # 1. Log required parameters
            client.log_param(run_id, "base_model", base_model)
            client.log_param(run_id, "accepted_training_pair_count", str(training_pair_count))
            client.log_param(
                run_id,
                "average_quality_score",
                f"{average_quality_score:.4f}" if average_quality_score is not None else "N/A",
            )
            client.log_param(run_id, "training_data_date_range", date_range or "N/A")

            # 2. Log JSONL training file as artifact
            p = Path(training_file_path)
            if p.exists():
                client.log_artifact(run_id, str(p), artifact_path="training_data")

            # 3. Format run URL for API inspection
            run_url = f"{self.tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"

            logger.info("Created MLflow run %s for job %s at %s", run_id, job_id, run_url)
            return run_id, run_url

        except Exception as exc:
            logger.error("Failed to create MLflow run for job %s: %s", job_id, exc)
            raise RuntimeError(f"MLflow tracking failed: {exc}") from exc

    def log_completion_metrics(
        self,
        run_id: str,
        metrics: dict[str, Any] | None = None,
        trained_tokens: int | None = None,
        status: str = "FINISHED",
    ) -> None:
        """Log training metrics (loss, tokens) and set final run status.

        Args:
            run_id: The MLflow run ID.
            metrics: Optional dictionary containing metrics (training_loss, validation_loss).
            trained_tokens: Total tokens trained during fine-tuning.
            status: Terminal status: 'FINISHED' or 'FAILED'.
        """
        if not run_id:
            return

        client = self._get_client()
        try:
            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        client.log_metric(run_id, k, float(v))

            if trained_tokens is not None:
                client.log_metric(run_id, "trained_tokens", float(trained_tokens))

            # Set terminal status
            client.set_terminated(run_id, status=status)
            logger.info("Updated MLflow run %s with status %s and metrics %s", run_id, status, metrics)
        except Exception as exc:
            logger.warning("Failed to log metrics to MLflow run %s: %s", run_id, exc)
            # Re-raise or log explicitly so failures are not silently swallowed
            raise RuntimeError(f"Failed to log metrics to MLflow run {run_id}: {exc}") from exc

    def register_model_artifact(
        self,
        run_id: str,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> dict[str, Any]:
        """Attempt to register a logged model artifact if a real model artifact exists in MLflow.

        Note: When using cloud API fine-tuning (e.g. OpenAI), weights are hosted by the provider
        and no local model artifact binary is downloaded. In that case, we report that limitation
        explicitly and record the provider model identifier separately.
        """
        client = self._get_client()
        try:
            # Check if artifact exists
            artifacts = client.list_artifacts(run_id, path=artifact_path)
            if not artifacts:
                msg = (
                    "No local model weight artifact found in MLflow run. "
                    "For cloud API fine-tuning, model is managed remotely by the provider; "
                    "provider model identifier is recorded in the model registry."
                )
                logger.info(msg)
                return {"registered": False, "reason": msg}

            reg_name = registered_model_name or f"neuroflow-{run_id}"
            import mlflow
            model_uri = f"runs:/{run_id}/{artifact_path}"
            mv = mlflow.register_model(model_uri, reg_name)
            return {"registered": True, "name": reg_name, "version": mv.version}
        except Exception as exc:
            logger.warning("Could not register model artifact in MLflow: %s", exc)
            return {"registered": False, "error": str(exc)}
