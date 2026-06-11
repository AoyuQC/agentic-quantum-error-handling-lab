"""Artifact store for large run outputs (plots, arrays, audit logs).

In the cloud wrap, big artifacts live in S3 and the agent response carries only
small references (S3 URIs). Off-cloud — local dev, tests — the same interface
writes to a local directory, so nothing else has to change.

Usage:
    store = make_artifact_store()                  # local, ./runs/<run_id>/
    store = make_artifact_store("s3://bucket/pre") # S3-backed
    uri = store.put_json("zne.json", figure_dict)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class ArtifactStore(ABC):
    """Writes named artifacts and returns a stable URI/reference for each."""

    @abstractmethod
    def put_json(self, name: str, obj: Any) -> str:
        """Serialize ``obj`` to JSON under ``name``; return its URI."""

    @abstractmethod
    def put_text(self, name: str, text: str) -> str:
        """Write ``text`` under ``name``; return its URI."""


class LocalArtifactStore(ArtifactStore):
    """Writes artifacts under a local directory (default ``runs/<run_id>``)."""

    def __init__(self, root: str | Path = "runs", run_id: str = "local"):
        self.base = Path(root) / run_id
        self.base.mkdir(parents=True, exist_ok=True)

    def put_json(self, name: str, obj: Any) -> str:
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=str))
        return str(path)

    def put_text(self, name: str, text: str) -> str:
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return str(path)


class S3ArtifactStore(ArtifactStore):
    """Writes artifacts to ``s3://<bucket>/<prefix>/<run_id>/<name>``."""

    def __init__(self, bucket: str, prefix: str = "aqem", run_id: str = "run", region: Optional[str] = None):
        import boto3

        self.bucket = bucket
        self.prefix = f"{prefix.rstrip('/')}/{run_id}"
        self._s3 = boto3.client("s3", region_name=region)

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def _uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def put_json(self, name: str, obj: Any) -> str:
        key = self._key(name)
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(obj, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        return self._uri(key)

    def put_text(self, name: str, text: str) -> str:
        key = self._key(name)
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=text.encode("utf-8"))
        return self._uri(key)


def make_artifact_store(
    destination: Optional[str] = None, run_id: str = "run", region: Optional[str] = None
) -> ArtifactStore:
    """Build an artifact store from a destination string.

    Args:
        destination: ``s3://bucket[/prefix]`` for S3, or a local path / None for
            a :class:`LocalArtifactStore`.
        run_id: groups one run's artifacts under a sub-path.
        region: AWS region for the S3 client.
    """
    if destination and destination.startswith("s3://"):
        rest = destination[len("s3://"):]
        bucket, _, prefix = rest.partition("/")
        return S3ArtifactStore(bucket=bucket, prefix=prefix or "aqem", run_id=run_id, region=region)
    return LocalArtifactStore(root=destination or "runs", run_id=run_id)
