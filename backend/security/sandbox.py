"""Short-lived Docker runner for document parsing with bounded resources."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from pipelines.ingestion.models import ExtractedPage

MAX_INPUT_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_BYTES = 50 * 1024 * 1024
SANDBOX_TIMEOUT_SECONDS = 60


class DockerExtractionSandbox:
    def __init__(self, image: str | None = None, runner: Callable = subprocess.run):
        self.image = image or os.environ.get("NEUROFLOW_EXTRACTION_IMAGE", "neuroflow-api:latest")
        self.runner = runner

    def extract(self, source_type: str, data: bytes) -> list[ExtractedPage]:
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("Document exceeds the extraction input limit.")
        if source_type not in {"pdf", "docx", "csv", "image"}:
            raise ValueError("Unsupported sandbox document type.")
        with tempfile.TemporaryDirectory(prefix="neuroflow-extract-") as temp:
            root = Path(temp)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            os.chmod(root, 0o755)
            os.chmod(input_dir, 0o755)
            os.chmod(output_dir, 0o777)
            input_path = input_dir / "source.bin"
            output_path = output_dir / "pages.json"
            input_path.write_bytes(data)
            os.chmod(input_path, 0o444)
            command = [
                "docker", "run", "--rm", "--network=none", "--memory=256m",
                "--cpus=1", "--pids-limit=64", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--user=65534:65534",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "--mount", f"type=bind,source={input_dir},target=/input,readonly",
                "--mount", f"type=bind,source={output_dir},target=/output",
                self.image, "python", "-m", "worker.extract_document",
                "--source-type", source_type,
            ]
            try:
                self.runner(command, check=True, capture_output=True, timeout=SANDBOX_TIMEOUT_SECONDS)
                if not output_path.is_file() or output_path.stat().st_size > MAX_OUTPUT_BYTES:
                    raise ValueError("Document extraction produced invalid or oversized output.")
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                return [ExtractedPage(**item) for item in payload]
            except subprocess.TimeoutExpired as exc:
                raise ValueError("Document extraction exceeded its time limit.") from exc
            except Exception as exc:
                # Do not reveal host paths, container output, or runtime details.
                raise ValueError("Isolated document extraction failed.") from exc
