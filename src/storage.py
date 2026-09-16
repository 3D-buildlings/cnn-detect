"""MinIO (S3-compatible) storage helpers backed by boto3.

Responsibilities:
  * ensure the pipeline bucket exists
  * seed ``dataset/`` and ``hlp/`` prefixes from a local source directory
  * download those prefixes into a per-run workspace
  * upload the whole per-run output tree to ``runs/<dag_run_id>/``
  * write small JSON manifests
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import Config as BotocoreConfig
from botocore.exceptions import ClientError

from src import config

logger = logging.getLogger("skyscrapper.storage")


class StorageError(RuntimeError):
    """Raised for fatal storage problems (missing seed, upload failures)."""


class MinIOClient:
    """Thin wrapper around a boto3 S3 client pointed at MinIO."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> None:
        self.endpoint = (endpoint or config.minio_endpoint()).rstrip("/")
        self.access_key = access_key or config.minio_access_key()
        self.secret_key = secret_key or config.minio_secret_key()
        self.bucket = bucket or config.minio_bucket()
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.access_key or not self.secret_key:
                raise StorageError(
                    "Missing MinIO credentials. Set MINIO_ACCESS_KEY / MINIO_SECRET_KEY "
                    "(or export them in docker-compose / Airflow)."
                )
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="us-east-1",
                config=BotocoreConfig(
                    connect_timeout=10,
                    read_timeout=60,
                    retries={"max_attempts": 4, "mode": "standard"},
                ),
            )
        return self._client

    def ensure_bucket(self) -> None:
        try:
            self.client.create_bucket(Bucket=self.bucket)
            logger.info("Created MinIO bucket '%s'.", self.bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists", "409"}:
                return
            raise StorageError(f"Could not create bucket '{self.bucket}': {exc}") from exc

    def list_objects(self, prefix: str = "") -> List[str]:
        keys: List[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def prefix_empty(self, prefix: str) -> bool:
        keys = self.list_objects(prefix)
        return not keys

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def put_json(self, key: str, payload: Dict[str, Any]) -> None:
        self.put_bytes(key, json.dumps(payload, indent=2, default=str).encode("utf-8"),
                       content_type="application/json")

    def upload_file(self, local_path: str, key: str) -> None:
        self.client.upload_file(local_path, self.bucket, key)

    def upload_dir(self, local_dir: str, prefix: str) -> List[str]:
        root = Path(local_dir)
        if not root.is_dir():
            raise StorageError(f"Local directory not found: {local_dir}")

        uploaded: List[str] = []
        for file in sorted(root.rglob("*")):
            if not file.is_file():
                continue
            rel = file.relative_to(root).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
            self.upload_file(str(file), key)
            uploaded.append(key)
        logger.info("Uploaded %d files under '%s/' -> s3://%s/%s", len(uploaded), local_dir, self.bucket, prefix)
        return uploaded

    def download_prefix(self, prefix: str, local_dir: str) -> List[str]:
        keys = self.list_objects(prefix)
        if not keys:
            raise StorageError(f"No objects found under 's3://{self.bucket}/{prefix}'.")

        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        written: List[str] = []
        for key in keys:
            rel = key[len(prefix):].lstrip("/") if prefix else key
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, key, str(dest))
            written.append(str(dest))
        logger.info("Downloaded %d objects from 's3://%s/%s' -> %s",
                    len(written), self.bucket, prefix, local_dir)
        return written


def default_client() -> MinIOClient:
    client = MinIOClient()
    client.ensure_bucket()
    return client


def seed_inputs_from_local(local_root: str, force: bool = False) -> Dict[str, Any]:
    client = default_client()
    result: Dict[str, Any] = {}
    for folder, prefix in [
        ("dataset", config.DATASET_PREFIX),
        ("hlp", config.HLP_PREFIX),
    ]:
        source = Path(local_root) / folder
        if not source.is_dir():
            raise StorageError(f"Seed source folder not found: {source}")

        files = [p for p in sorted(source.rglob("*")) if p.is_file()]
        existing = set(client.list_objects(f"{prefix}/"))
        uploaded, skipped = 0, 0
        for file in files:
            rel = file.relative_to(source).as_posix()
            key = f"{prefix}/{rel}"
            if not force and key in existing:
                skipped += 1
                continue
            client.upload_file(str(file), key)
            uploaded += 1
        result[folder] = {"found": len(files), "uploaded": uploaded, "skipped": skipped}
        logger.info("seed '%s': found=%d uploaded=%d skipped=%d",
                    folder, len(files), uploaded, skipped)
    return result


def write_run_manifest(client: MinIOClient, run_id: str, payload: Dict[str, Any]) -> str:
    key = f"{config.s3_run_prefix(run_id)}/_manifest.json"
    client.put_json(key, payload)
    logger.info("Run manifest written to s3://%s/%s", client.bucket, key)
    return key
