# NeuroFlow — Modular RAG Platform

**NeuroFlow** is a modular, observable Retrieval-Augmented Generation (RAG) platform that supports document ingestion, hybrid retrieval, grounded LLM generation, asynchronous evaluation, pipeline optimization, and evidence-driven fine-tuning.

The platform combines FastAPI, PostgreSQL with pgvector, Redis, MLflow, and multiple LLM providers to provide a scalable foundation for building reliable AI-powered knowledge systems.

## Key Features

### 1. Document Ingestion
- Supports PDF, DOCX, images, CSV, and web content ingestion.
- Extracts and processes document content.
- Implements configurable document chunking.
- Generates embeddings and stores document metadata.
- Supports asynchronous ingestion workflows.
- Includes document validation and ingestion status tracking.

### 2. Hybrid Retrieval
- Dense vector similarity search.
- Lexical and keyword-based retrieval.
- Metadata filtering.
- Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking.
- Embedding caching and retrieval optimization.

### 3. Grounded Generation
- Multiple LLM provider integrations.
- Provider selection and model routing.
- Context-aware prompt construction.
- Source-grounded responses with citation tracking.
- Streaming responses using Server-Sent Events (SSE).
- Prompt-injection protection.

### 4. Evaluation Framework
- Faithfulness evaluation.
- Answer relevance evaluation.
- Context precision and recall.
- Evaluation dataset management.
- Calibration and quality reporting.
- Pipeline-level evaluation metrics.
- Evaluation result tracking.

### 5. Fine-Tuning and Optimization
- Training data preparation and validation.
- JSONL dataset export.
- Fine-tuning job management.
- MLflow experiment tracking integration.
- Pipeline configuration optimization.
- Evidence-based model promotion principles.

### 6. Backend API
- FastAPI application architecture.
- REST API endpoints for ingestion, retrieval, querying, pipelines, evaluations, and fine-tuning.
- OpenAPI documentation.
- Request validation and structured responses.
- Authentication and authorization.
- Tenant-aware access controls.

### 7. Security and Reliability
- Authentication and scoped authorization.
- Secret management and sensitive-data redaction.
- SSRF and URL validation protections.
- Prompt-injection defenses.
- Rate limiting and timeout management.
- Circuit breakers and backpressure handling.
- Database access controls.
- Structured logging and observability.

### 8. Background Processing
- ARQ-based asynchronous workers.
- Redis-backed task processing.
- Document processing jobs.
- Evaluation workers.
- Fine-tuning job monitoring.
- Data retention tasks.

### 9. Python SDK
- Python client for interacting with NeuroFlow APIs.
- API request handling.
- Structured response models.
- Developer-friendly integration.
- SDK documentation and tests.

## Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, FastAPI, Pydantic |
| Database | PostgreSQL, pgvector |
| Cache / Queue | Redis |
| AI / LLM | OpenAI, Anthropic |
| Retrieval | Vector Search, RRF, Cross-Encoder |
| Evaluation | RAGAS-style evaluation metrics |
| Experiment Tracking | MLflow |
| Background Jobs | ARQ |
| Testing | Pytest |
| Code Quality | Ruff, Mypy |
| Load Testing | Locust |
| Containerization | Docker, Docker Compose |
| SDK | Python |

## Project Structure

```text
NeuroFlow/
│
├── backend/
│   ├── api/
│   ├── config.py
│   ├── db/
│   ├── models/
│   ├── observability/
│   ├── providers/
│   ├── resilience/
│   ├── security/
│   ├── services/
│   ├── tasks/
│   ├── worker/
│   └── tests/
│
├── frontend/
│
├── pipelines/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── evaluation/
│   └── training/
│
├── evaluation/
│   ├── calibration_results.json
│   ├── improvement_log.md
│   └── tune_retrieval.py
│
├── infra/
│   ├── docker-compose.yml
│   └── init/
│
├── sdk/
│
├── docs/
│
├── tests/
│   ├── unit/
│   ├── sdk/
│   └── performance/
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.11 or later, subject to dependency compatibility.
- Docker Desktop.
- Git.
- PostgreSQL and Redis for infrastructure-backed workflows.

### 1. Clone the repository

```bash
git clone https://github.com/Amanrao8716/NeuroFlow-HiDevs.git
cd NeuroFlow-HiDevs
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install project development dependencies if configured:

```bash
pip install -e ".[dev]"
```

### 4. Configure environment

Create a local `.env` file using the provided example when available.

Configure the required database, Redis, LLM provider, and security settings. Never commit real credentials or API keys.

### 5. Start infrastructure

If the Docker Compose configuration and required environment variables are available:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Check running containers:

```bash
docker compose -f infra/docker-compose.yml ps
```

### 6. Run the backend

Use the project's configured application entry point. For the standard FastAPI entry point:

```bash
uvicorn backend.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Verify the health endpoint:

```text
http://127.0.0.1:8000/health
```

## Testing

Run the unit tests:

```bash
python -m pytest tests/unit/ -v
```

Run SDK tests:

```bash
python -m pytest tests/sdk/ -v
```

Run backend tests:

```bash
python -m pytest -o pythonpath="backend ." backend/tests/ -v
```

Run code quality checks:

```bash
python -m ruff check backend/
python -m ruff format --check backend/
```

Run load tests using the configured Locust scenarios and an explicitly isolated test deployment.

## Quality Verification

The latest reported verification results:

| Check | Result |
|---|---:|
| Unit tests | 62 passed |
| SDK tests | 15 passed |
| Backend tests | 185 passed |
| Total tests | 262 passed |
| Ruff lint | Passed |
| Ruff formatting | Passed |
| OpenAPI tests | 5 passed (included in unit tests) |

**Note:** These are the reported results of the latest verification run. Live infrastructure-dependent quality metrics require running PostgreSQL, Redis, MLflow, and the relevant evaluation providers. No unmeasured quality scores are claimed.

## Documentation

Project documentation is maintained in the `docs/` directory. It includes architecture decisions, API contracts, data models, developer guidance, and operational documentation.

## Development Principles

- Keep API contracts versioned and documented.
- Use modular, reusable pipeline components.
- Protect tenant data through scoped authorization.
- Treat retrieved documents and model output as untrusted input.
- Apply bounded retries, timeouts, and concurrency controls.
- Track evaluation results and optimization decisions.
- Promote models only when measured evidence supports the change.
- Never commit credentials, private keys, or local environment files.

## Project Status

NeuroFlow has progressed from its initial architecture and API contracts through backend implementation, retrieval, generation, evaluation, security, reliability, and Python SDK development.

The repository is under incremental development. Infrastructure-dependent integrations and live evaluation metrics should be considered verified only when tested against their required running services.

## Author

**Aman Rao**

GitHub: https://github.com/Amanrao8716

