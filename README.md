# NeuroFlow

**NeuroFlow** is a modular, production-oriented Retrieval-Augmented Generation (RAG) platform designed to build reliable, observable, secure, and scalable AI-powered knowledge systems.

It combines document ingestion, hybrid retrieval, grounded generation, evaluation, fine-tuning workflows, and observability into a unified platform.

## Key Features

### 1. Document Ingestion
- Supports document ingestion through configurable ingestion pipelines.
- Extracts and processes document content.
- Provides chunking and preprocessing workflows.
- Supports metadata extraction and document organization.
- Designed for extensible ingestion sources.

### 2. Hybrid Retrieval
- Combines semantic and keyword-based retrieval.
- Supports metadata-aware filtering.
- Uses retrieval fusion techniques to improve relevant results.
- Supports configurable retrieval strategies.
- Provides reusable retrieval components.

### 3. Grounded Generation
- Generates answers using retrieved context.
- Supports configurable LLM providers.
- Uses structured prompt construction.
- Includes source-grounding and citation-oriented generation.
- Supports streaming-compatible API workflows.

### 4. Evaluation Framework
- Evaluates RAG output quality using multiple metrics.
- Includes faithfulness and answer relevance evaluation.
- Supports context precision and context recall evaluation.
- Provides calibration and retrieval-tuning workflows.
- Stores evaluation artifacts for analysis.

### 5. Fine-Tuning Workflows
- Includes fine-tuning data preparation workflows.
- Supports dataset validation and processing.
- Integrates experiment tracking workflows.
- Provides configurable optimization components.

### 6. API and Backend
- Built using FastAPI.
- Provides modular REST API endpoints.
- Includes request validation and structured response models.
- Provides OpenAPI documentation.
- Separates API, service, provider, and worker responsibilities.

### 7. Security and Reliability
- Authentication and authorization components.
- Input validation and prompt-injection protections.
- Rate limiting and request controls.
- Circuit breaker and timeout management.
- Backpressure and resilience mechanisms.
- Secure configuration and secret-management utilities.

### 8. Background Processing
- Supports asynchronous task processing.
- Uses Redis-backed infrastructure for background workflows.
- Separates long-running jobs from API request handling.

### 9. Python SDK
- Provides a Python SDK for interacting with NeuroFlow.
- Simplifies API access from Python applications.
- Includes SDK tests and developer documentation.

---

## Technology Stack

| Component | Technologies |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Database | PostgreSQL |
| Vector Search | pgvector |
| Cache / Queue | Redis |
| AI Integration | LLM provider abstractions |
| Evaluation | Custom RAG evaluation framework |
| Experiment Tracking | MLflow integration |
| Testing | Pytest |
| Code Quality | Ruff, Mypy |
| API Documentation | OpenAPI / Swagger |
| Containerization | Docker |

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
│   ├── tests/
│   └── worker/
│
├── pipelines/
│   ├── ingestion/
│   ├── retrieval/
│   └── generation/
│
├── evaluation/
│   ├── calibration_results.json
│   └── tune_retrieval.py
│
├── infra/
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

- Python 3.11+
- Git
- PostgreSQL
- Redis
- Docker (recommended for infrastructure setup)

### 1. Clone the Repository

```bash
git clone https://github.com/Amanrao8716/NeuroFlow-HiDevs.git
cd NeuroFlow-HiDevs
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a local `.env` file using `.env.example` as the reference.

```powershell
Copy-Item .env.example .env
```

Configure the required database, Redis, provider, and security settings before starting the application.

**Never commit `.env` or real credentials to GitHub.**

### 5. Start Infrastructure

If the Docker Compose configuration is available:

```bash
docker compose -f infra/docker-compose.yml up -d
```

### 6. Run the Backend

```bash
uvicorn backend.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
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
python -m pytest backend/tests/ -v
```

Run performance tests:

```bash
python -m pytest tests/performance/ -v
```

## Code Quality

Run Ruff:

```bash
python -m ruff check backend/
```

Check formatting:

```bash
python -m ruff format --check backend/
```

Run Mypy on supported modules:

```bash
python -m mypy backend/main.py
```

## Development Progress

NeuroFlow has been developed incrementally through its implementation tasks, covering:

- Backend architecture and API development.
- Document ingestion and processing.
- Retrieval and generation workflows.
- Evaluation and quality optimization.
- Fine-tuning and experiment tracking.
- Security and reliability improvements.
- Background processing.
- Python SDK and developer documentation.
- Testing and code-quality improvements.

## Quality Verification

The latest reported verification included:

| Check | Result |
|---|---|
| Unit tests | 62 passed |
| SDK tests | 15 passed |
| Backend tests | 185 passed |
| Total reported tests | 262 passed |
| Ruff lint | Passed |
| Ruff formatting | Passed |
| Mypy | Passed for selected modules |

*These results reflect the reported local test run. They do not represent a live production performance or deployment guarantee.*

## Contributing

1. Create a separate branch for your task.
2. Follow the existing project structure and coding conventions.
3. Add or update tests for your changes.
4. Run the relevant quality checks.
5. Submit changes through a pull request.

## License

Add the applicable license information before distributing this project.

---

**NeuroFlow — Building reliable, observable, and scalable RAG systems.**