"""Keep legacy endpoint tests focused on their behavior under new auth defaults."""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def authenticated_test_client(monkeypatch):
    import api.compare as compare_api
    import api.query as query_api
    import api.retrieve as retrieve_api
    from main import app

    from backend.security.auth import ClientProfile, get_current_user

    async def test_user():
        return ClientProfile("test-suite", frozenset({"query", "ingest", "admin"}))

    app.dependency_overrides[get_current_user] = test_user
    monkeypatch.setattr(query_api, "classify_query", AsyncMock(return_value=False))
    monkeypatch.setattr(retrieve_api, "classify_query", AsyncMock(return_value=False))
    monkeypatch.setattr(compare_api, "classify_query", AsyncMock(return_value=False))
    yield
    app.dependency_overrides.pop(get_current_user, None)
