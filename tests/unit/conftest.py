from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Tests use synthetic connection settings and must not depend on a local .env.
os.environ.setdefault("POSTGRES_PASSWORD", "unit-test-postgres-password")
os.environ.setdefault("REDIS_PASSWORD", "unit-test-redis-password")
os.environ.setdefault("NEUROFLOW_JWT_SECRET", "unit-test-signing-key-at-least-32-characters")
os.environ.setdefault("NEUROFLOW_AUTH_CLIENTS", "{}")
