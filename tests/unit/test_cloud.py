"""Unit tests for the cloud wrap (offline — no AWS).

Covers the artifact store factory + local writes, the Guardrail no-op path, and
the artifact-store destination parsing. The Runtime handler is exercised in
tests/integration/test_runtime.py (needs the local simulator).
"""

import json

from aqem.cloud.artifacts import (
    LocalArtifactStore,
    S3ArtifactStore,
    make_artifact_store,
)
from aqem.cloud.guardrails import Guardrail


def test_local_artifact_store_writes_json(tmp_path):
    store = LocalArtifactStore(root=tmp_path, run_id="r1")
    uri = store.put_json("x.json", {"a": 1})
    assert json.loads(open(uri).read()) == {"a": 1}
    assert "r1" in uri


def test_make_artifact_store_local_by_default(tmp_path):
    store = make_artifact_store(str(tmp_path), run_id="r2")
    assert isinstance(store, LocalArtifactStore)
    uri = store.put_text("note.txt", "hi")
    assert open(uri).read() == "hi"


def test_make_artifact_store_parses_s3_uri(monkeypatch):
    # Avoid a real boto3 client: stub S3ArtifactStore.__init__.
    created = {}

    def fake_init(self, bucket, prefix="aqem", run_id="run", region=None):
        created.update(bucket=bucket, prefix=prefix, run_id=run_id)

    monkeypatch.setattr(S3ArtifactStore, "__init__", fake_init)
    store = make_artifact_store("s3://my-bucket/some/prefix", run_id="r3")
    assert isinstance(store, S3ArtifactStore)
    assert created == {"bucket": "my-bucket", "prefix": "some/prefix", "run_id": "r3"}


def test_guardrail_disabled_is_noop():
    g = Guardrail(guardrail_id=None)
    assert not g.enabled
    result = g.check("anything", source="INPUT")
    assert result.allowed
    assert result.action == "DISABLED"
