"""Bearer-token auth. Client sends `Authorization: Bearer <key>`; server
hashes the incoming key and constant-time-compares against STABL_API_KEY_HASH.
"""
import hashlib
import secrets

from fastapi import Header, HTTPException

from . import settings


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def require_bearer(authorization: str = Header(default="")) -> None:
    if not settings.API_KEY_HASH:
        raise HTTPException(500, "server misconfigured: STABL_API_KEY_HASH not set")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.split(None, 1)[1].strip()
    if not secrets.compare_digest(_hash(token), settings.API_KEY_HASH):
        raise HTTPException(403, "invalid token")
