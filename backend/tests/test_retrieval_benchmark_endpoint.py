"""Test-only retrieval ranking endpoint remains disabled unless explicitly enabled."""

from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient


def test_benchmark_ranking_endpoint_is_disabled_by_default(monkeypatch):
    from main import app
    import api.retrieve as retrieve_api

    monkeypatch.delenv("NEUROFLOW_TEST_BENCHMARK_ENDPOINT", raising=False)
    monkeypatch.setattr(retrieve_api._retriever, "retrieve_parallel", AsyncMock())
    response = TestClient(app).post("/__test__/retrieval-rankings", json={
        "query": "offline validation question",
        "pipeline_id": str(uuid4()),
        "top_k": 10,
    })

    assert response.status_code == 404
    retrieve_api._retriever.retrieve_parallel.assert_not_awaited()


def test_benchmark_ranking_endpoint_returns_all_four_actual_methods(monkeypatch):
    from main import app
    from pipelines.retrieval.query_processor import ProcessedQuery
    from pipelines.retrieval.retriever import RetrievalResult
    import api.retrieve as retrieve_api

    monkeypatch.setenv("NEUROFLOW_TEST_BENCHMARK_ENDPOINT", "true")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    dense = [RetrievalResult("dense-id", "doc", "dense text", 0.9, retrieval_method="dense", rank=1)]
    sparse = [RetrievalResult("sparse-id", "doc", "sparse text", 1.1, retrieval_method="sparse", rank=1)]
    fused = [RetrievalResult("fused-id", "doc", "fused text", 0.03, retrieval_method="rrf_fusion", rank=1)]
    monkeypatch.setattr(
        retrieve_api._processor,
        "process",
        AsyncMock(return_value=ProcessedQuery(query="offline validation question")),
    )
    monkeypatch.setattr(retrieve_api._retriever, "retrieve_parallel", AsyncMock(return_value=(dense, sparse, [])))
    monkeypatch.setattr(retrieve_api._reranker, "rerank", AsyncMock(return_value=fused))

    response = TestClient(app).post("/__test__/retrieval-rankings", json={
        "query": "offline validation question",
        "pipeline_id": str(uuid4()),
        "top_k": 10,
    })

    assert response.status_code == 200
    assert response.json() == {
        "dense_only": ["dense-id"],
        "sparse_only": ["sparse-id"],
        "hybrid_rrf": ["dense-id", "sparse-id"],
        "hybrid_reranked": ["fused-id"],
    }
    retrieve_api._retriever.retrieve_parallel.assert_awaited_once()
    retrieve_api._reranker.rerank.assert_awaited_once()


def test_benchmark_ranking_endpoint_refuses_online_model_loading(monkeypatch):
    from main import app
    import api.retrieve as retrieve_api

    monkeypatch.setenv("NEUROFLOW_TEST_BENCHMARK_ENDPOINT", "true")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    retrieve_mock = AsyncMock()
    monkeypatch.setattr(retrieve_api._retriever, "retrieve_parallel", retrieve_mock)

    response = TestClient(app).post("/__test__/retrieval-rankings", json={
        "query": "offline validation question",
        "pipeline_id": str(uuid4()),
        "top_k": 10,
    })

    assert response.status_code == 503
    retrieve_mock.assert_not_awaited()
