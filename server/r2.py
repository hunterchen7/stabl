"""Cloudflare R2 client. Used for big-file uploads — client gets a presigned
PUT URL, uploads directly to R2 (skipping our tunnel's body limit), then
hits /v1/upload/r2/finalize so the server can pull it down to UPLOADS_DIR.

R2 is S3-compatible so boto3 works as-is with the right endpoint URL.
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

import boto3
from botocore.config import Config


def configured() -> bool:
    return bool(os.environ.get("R2_BUCKET") and os.environ.get("R2_ACCOUNT_ID"))


@lru_cache(maxsize=1)
def _client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def bucket() -> str:
    return os.environ["R2_BUCKET"]


def presign_put(key: str, expires: int = 3600) -> str:
    return _client().generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket(), "Key": key},
        ExpiresIn=expires,
    )


def head(key: str) -> Optional[dict]:
    try:
        return _client().head_object(Bucket=bucket(), Key=key)
    except Exception:
        return None


def download(key: str, local_path: str) -> None:
    _client().download_file(bucket(), key, local_path)
