"""Unit tests for the DAG engine: validation, topo order, invalidation cascade,
and the retry loop — using lightweight fake nodes (no simulator)."""

import pytest

from aqem.dag.engine import DAGEngine, _topological_order, _validate_dag
from aqem.dag.node import Node
from aqem.models import Decision, DecisionAction, NodeResult


class _Recorder(Node):
    """A trivial node that records each time it runs."""

    def __init__(self, node_id, deps=(), invalidates=(), log=None):
        self.node_id = node_id
        self.dependencies = deps
        self.invalidates = invalidates
        self._log = log if log is not None else []

    def run(self, ctx):
        self._log.append(self.node_id)
        ctx.put(self.node_id, {"ran": True})
        return NodeResult(node_id=self.node_id, outputs={"ran": True})


def test_validate_dag_detects_cycle():
    a = _Recorder("a", deps=("b",))
    b = _Recorder("b", deps=("a",))
    with pytest.raises(ValueError, match="cycle"):
        _validate_dag({"a": a, "b": b})


def test_validate_dag_detects_unknown_dependency():
    a = _Recorder("a", deps=("ghost",))
    with pytest.raises(ValueError, match="unknown node"):
        _validate_dag({"a": a})


def test_topological_order_respects_dependencies():
    nodes = {
        "a": _Recorder("a"),
        "b": _Recorder("b", deps=("a",)),
        "c": _Recorder("c", deps=("b",)),
    }
    order = _topological_order(nodes)
    assert order.index("a") < order.index("b") < order.index("c")


def test_invalidation_set_follows_invalidates_edges():
    # Mirror the real pipeline's invalidation edges.
    nodes = [
        _Recorder("strategy_select", invalidates=("readout_calibrate", "execute", "post_process")),
        _Recorder("readout_calibrate", deps=("strategy_select",), invalidates=("execute", "post_process")),
        _Recorder("execute", deps=("readout_calibrate",), invalidates=("post_process",)),
        _Recorder("post_process", deps=("execute",), invalidates=("validate",)),
        _Recorder("validate", deps=("post_process",)),
    ]
    engine = DAGEngine(nodes, terminal_node_id=None, max_iterations=1)

    shots_dirty = engine.invalidation_set(DecisionAction.RETRY_SHOTS.value)
    assert "execute" in shots_dirty and "post_process" in shots_dirty
    assert "readout_calibrate" not in shots_dirty

    cal_dirty = engine.invalidation_set(DecisionAction.RETRY_CALIBRATION.value)
    assert {"readout_calibrate", "execute", "post_process"} <= cal_dirty
    assert "strategy_select" not in cal_dirty

    strat_dirty = engine.invalidation_set(DecisionAction.RETRY_STRATEGY.value)
    assert {"strategy_select", "readout_calibrate", "execute", "post_process"} <= strat_dirty


class _Validate(Node):
    """Validate node emitting a scripted sequence of decisions."""

    node_id = "validate"
    dependencies = ("post_process",)

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self._i = 0

    def run(self, ctx):
        decision = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        ctx.put(self.node_id, {"decision": decision.to_dict()})
        return NodeResult(node_id=self.node_id, outputs={"decision": decision.to_dict()})


def test_engine_retries_then_stops_and_reruns_only_dirty(make_ctx):
    log = []
    retry = Decision(action=DecisionAction.RETRY_SHOTS.value, reason="more shots")
    stop = Decision(action=DecisionAction.STOP.value, reason="target met")

    nodes = [
        _Recorder("strategy_select", invalidates=("execute", "post_process"), log=log),
        _Recorder("execute", deps=("strategy_select",), invalidates=("post_process",), log=log),
        _Recorder("post_process", deps=("execute",), invalidates=("validate",), log=log),
        _Validate([retry, stop]),
        _Recorder("report", deps=("validate",), log=log),
    ]
    engine = DAGEngine(nodes, max_iterations=5)
    ctx = make_ctx()

    record = engine.run(ctx)

    assert record.status == "stopped"
    assert record.iterations == 2
    # strategy_select runs once; execute/post_process run twice (retry_shots).
    assert log.count("strategy_select") == 1
    assert log.count("execute") == 2
    assert log.count("post_process") == 2
    # report runs exactly once at the end.
    assert log.count("report") == 1


def test_engine_exhausts_on_persistent_retry(make_ctx):
    log = []
    retry = Decision(action=DecisionAction.RETRY_SHOTS.value, reason="never satisfied")
    nodes = [
        _Recorder("execute", invalidates=("post_process",), log=log),
        _Recorder("post_process", deps=("execute",), invalidates=("validate",), log=log),
        _Validate([retry]),
    ]
    engine = DAGEngine(nodes, terminal_node_id=None, max_iterations=3)
    record = engine.run(make_ctx())
    assert record.status == "exhausted"
    assert record.iterations == 3
