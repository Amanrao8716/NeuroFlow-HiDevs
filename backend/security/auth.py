"""JWT authentication backed by a server-configured client registry."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

JWT_ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 3600
bearer = HTTPBearer(auto_error=False)
router = APIRouter(prefix="/auth", tags=["authentication"])


@dataclass(frozen=True)
class ClientProfile:
    client_id: str
    scopes: frozenset[str]


class TokenRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=128)
    client_secret: str = Field(min_length=1, max_length=4096)


def hash_client_secret(secret: str, *, salt: bytes | None = None) -> str:
    """Return a scrypt hash suitable for NEUROFLOW_AUTH_CLIENTS provisioning."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(secret.encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt${}${}".format(
        base64.urlsafe_b64encode(salt).decode().rstrip("="),
        base64.urlsafe_b64encode(digest).decode().rstrip("="),
    )


def _check_secret(secret: str, encoded: str) -> bool:
    try:
        scheme, salt_text, digest_text = encoded.split("$", 2)
        if scheme != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        expected = base64.urlsafe_b64decode(digest_text + "=" * (-len(digest_text) % 4))
        actual = hashlib.scrypt(secret.encode(), salt=salt, n=2**14, r=8, p=1)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _client_registry() -> dict[str, dict]:
    raw = os.environ.get("NEUROFLOW_AUTH_CLIENTS", "{}")
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _jwt_secret() -> str:
    secret = os.environ.get("NEUROFLOW_JWT_SECRET", "")
    if len(secret.encode()) < 32:
        raise RuntimeError("NEUROFLOW_JWT_SECRET must contain at least 32 bytes.")
    return secret


def issue_token(client_id: str, scopes: set[str]) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": client_id, "scopes": sorted(scopes), "iat": now, "exp": now + TOKEN_TTL_SECONDS},
        _jwt_secret(),
        algorithm=JWT_ALGORITHM,
    )


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> ClientProfile:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    try:
        payload = jwt.decode(
            credentials.credentials,
            _jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "scopes", "iat", "exp"]},
        )
        client_id = payload["sub"]
        scopes = payload["scopes"]
        registry_entry = _client_registry().get(client_id)
        if (
            not isinstance(client_id, str)
            or not isinstance(scopes, list)
            or not registry_entry
            or set(scopes) != set(registry_entry.get("scopes", []))
        ):
            raise _unauthorized()
        return ClientProfile(client_id, frozenset(scopes))
    except HTTPException:
        raise
    except Exception as exc:
        raise _unauthorized() from exc


def require_scope(scope: str):
    async def dependency(user: ClientProfile = Depends(get_current_user)) -> ClientProfile:
        if scope not in user.scopes and "admin" not in user.scopes:
            raise HTTPException(status_code=403, detail="Insufficient scope.")
        return user

    return dependency


async def authenticate_token(payload: TokenRequest) -> dict[str, str | int]:
    registry_entry = _client_registry().get(payload.client_id)
    candidate_hash = registry_entry.get("secret_hash", "") if registry_entry else ""
    valid_secret = _check_secret(payload.client_secret, candidate_hash)
    # Perform comparable hash work even for an unknown client.
    if not registry_entry:
        _check_secret(payload.client_secret, hash_client_secret("dummy-client-secret"))
    if not registry_entry or not valid_secret:
        raise _unauthorized()
    scopes = set(registry_entry.get("scopes", []))
    return {
        "access_token": issue_token(payload.client_id, scopes),
        "token_type": "bearer",
        "expires_in": TOKEN_TTL_SECONDS,
    }


@router.post("/token")
async def create_access_token(payload: TokenRequest):
    return await authenticate_token(payload)
