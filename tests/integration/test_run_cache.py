"""Record → replay → clear roundtrip for the web run cache.

The first run of a setup is recorded; an identical setup then replays the same
frames (signalled by a leading ``cache_status cached=true`` event) without
running the engine again. Clearing the cache makes the next run fresh. Runs with
the VLM disabled so it is fully offline/deterministic, and a tiny replay gap so
the replay doesn't actually sleep.
"""

import json

import pytest

from aqem.web import server
from aqem.web.cache import LocalRunCache, cache_key

pytestmark = pytest.mark.integration


def _parse(frames):
    """Parse a list of raw SSE strings into (event, data) tuples."""
    out = []
    for frame in frames:
        event, data = "message", ""
        for line in frame.strip().split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data += line[len("data:"):].strip()
        out.append((event, json.loads(data) if data else {}))
    return out


def _drain(req):
    """Run the SSE generator to completion and return parsed (event, data)."""
    return _parse(list(server._run_stream(req)))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the server at a fresh local cache and disable replay sleeps."""
    c = LocalRunCache(root=tmp_path / "cache")
    monkeypatch.setattr(server, "_CACHE", c)
    monkeypatch.setattr(server, "_REPLAY_MAX_GAP", 0.0)
    return c


def test_record_replay_clear_roundtrip(cache):
    req = server.RunRequest(
        qubits=2,
        target_accuracy=0.06,
        device="qd_readout_2",
        use_vlm=False,
        compare_baseline=False,
        budget_shots=200_000,
        seed=7,
    )
    key = cache_key(req.model_dump())

    # --- fresh run: records, signals not-cached ---
    first = _drain(req)
    events = [e for e, _ in first]
    assert first[0] == ("progress", {"event": "cache_status", "cached": False})
    assert events[-1] == "done"
    assert "result" in events
    assert cache.get(key) is not None  # recording persisted

    fresh_result = next(d for e, d in first if e == "result")

    # --- replay: signals cached, yields the identical result payload ---
    second = _drain(req)
    assert second[0] == ("progress", {"event": "cache_status", "cached": True})
    assert second[-1][0] == "done"
    replay_result = next(d for e, d in second if e == "result")
    assert replay_result == fresh_result

    # The progress frames (minus the cache_status header) match the recording.
    recorded = [f["data"] for f in cache.get(key)["frames"] if f["event"] == "progress"]
    replayed = [d for e, d in second[1:] if e == "progress"]
    assert replayed == recorded

    # --- clear: removes the recording; next run is fresh again ---
    assert cache.clear() == 1
    assert cache.get(key) is None
    third = _drain(req)
    assert third[0] == ("progress", {"event": "cache_status", "cached": False})


def test_setup_change_is_a_miss(cache):
    base = dict(
        qubits=2, target_accuracy=0.06, device="qd_readout_2",
        use_vlm=False, compare_baseline=False, budget_shots=200_000, seed=7,
    )
    _drain(server.RunRequest(**base))
    # A different seed is a different setup -> different key -> not cached.
    other = _drain(server.RunRequest(**{**base, "seed": 8}))
    assert other[0] == ("progress", {"event": "cache_status", "cached": False})
