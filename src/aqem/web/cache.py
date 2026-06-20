"""Record/replay cache for adaptive-loop web runs.

A "live run" served by :mod:`aqem.web.server` is just an ordered, time-stamped
sequence of Server-Sent-Event frames (``progress`` events — including live
``llm_delta`` LLM tokens — then a terminal ``result``, then ``done``). The first
run of a given experiment setup is *recorded* as those frames plus their
relative timing; a repeat of the same setup is *replayed* by re-emitting the
recorded frames with the same pacing, so the UI animates exactly like a live
run (no engine work, no Bedrock calls).

The cache is keyed by the full :class:`~aqem.web.server.RunRequest` setup, so any
field change is a different setup (a miss). Storage mirrors the destination
convention of :mod:`aqem.cloud.artifacts`: ``s3://bucket/prefix`` is S3-backed
(shared across the autoscaled Fargate tasks), anything else is a local
directory. Build one with :func:`make_run_cache`.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

# Recording schema version, bumped if the frame format changes incompatibly.
RECORDING_VERSION = 1


def cache_key(request: dict[str, Any]) -> str:
    """Stable hex key for an experiment setup (a ``RunRequest`` as a dict).

    Canonical JSON (sorted keys) → sha256, so the key is independent of field
    order and any field change yields a different key (a cache miss).
    """
    blob = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RunCache(ABC):
    """Stores one recording (a JSON dict) per setup key."""

    @abstractmethod
    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return the recording for ``key``, or ``None`` on a miss."""

    @abstractmethod
    def put(self, key: str, recording: dict[str, Any]) -> None:
        """Store ``recording`` under ``key`` (overwrites any prior one)."""

    @abstractmethod
    def clear(self) -> int:
        """Remove all recordings; return how many were removed."""

    @abstractmethod
    def keys(self) -> list[str]:
        """Return the keys of all stored recordings."""


class LocalRunCache(RunCache):
    """Stores one ``<key>.json`` recording per setup under a local directory."""

    def __init__(self, root: str | Path = "runs/cache"):
        self.base = Path(root)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base / f"{key}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # A corrupt/partial recording is treated as a miss, not a crash.
            return None

    def put(self, key: str, recording: dict[str, Any]) -> None:
        self._path(key).write_text(json.dumps(recording, default=str))

    def clear(self) -> int:
        removed = 0
        for path in self.base.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def keys(self) -> list[str]:
        return [p.stem for p in self.base.glob("*.json")]


class S3RunCache(RunCache):
    """Stores recordings at ``s3://<bucket>/<prefix>/<key>.json``.

    Mirrors the boto3 usage in :class:`aqem.cloud.artifacts.S3ArtifactStore`, so
    the same task role grant (``grantReadWrite`` + ``List*``) covers replay,
    record, and clear.
    """

    def __init__(self, bucket: str, prefix: str = "aqem-cache", region: Optional[str] = None):
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._s3 = boto3.client("s3", region_name=region)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=self._key(key))
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception:
            return None
        try:
            return json.loads(obj["Body"].read())
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, recording: dict[str, Any]) -> None:
        self._s3.put_object(
            Bucket=self.bucket,
            Key=self._key(key),
            Body=json.dumps(recording, default=str).encode("utf-8"),
            ContentType="application/json",
        )

    def clear(self) -> int:
        paginator = self._s3.get_paginator("list_objects_v2")
        to_delete: list[dict[str, str]] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".json"):
                    to_delete.append({"Key": obj["Key"]})
        removed = 0
        # delete_objects takes at most 1000 keys per call.
        for i in range(0, len(to_delete), 1000):
            batch = to_delete[i : i + 1000]
            self._s3.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
            removed += len(batch)
        return removed

    def keys(self) -> list[str]:
        paginator = self._s3.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}/"):
            for obj in page.get("Contents", []):
                name = obj["Key"].rsplit("/", 1)[-1]
                if name.endswith(".json"):
                    out.append(name[: -len(".json")])
        return out


def make_run_cache(
    destination: Optional[str] = None, region: Optional[str] = None
) -> RunCache:
    """Build a run cache from a destination string (env-driven by default).

    Resolution order for ``destination``:
        explicit arg → ``AQEM_CACHE`` → ``AQEM_CACHE_DIR`` → ``runs/cache``.

    ``s3://bucket[/prefix]`` selects :class:`S3RunCache`; anything else is a
    local directory.
    """
    dest = destination or os.environ.get("AQEM_CACHE")
    if dest and dest.startswith("s3://"):
        rest = dest[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        return S3RunCache(bucket=bucket, prefix=prefix or "aqem-cache", region=region)
    root = dest or os.environ.get("AQEM_CACHE_DIR") or "runs/cache"
    return LocalRunCache(root=root)
