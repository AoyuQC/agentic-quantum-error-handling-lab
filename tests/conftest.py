"""Shared pytest configuration and fixtures for the AQEM test suite."""

import pytest

from aqem.dag import RunContext
from aqem.models import Budget, Problem
from aqem.policy import AuditLog, Policy


@pytest.fixture
def make_ctx():
    """Factory for a lightweight RunContext (no real device/circuit).

    Used by DAG-engine tests that drive fake nodes — those nodes don't touch
    ``device`` or ``circuit``, so ``None`` placeholders are fine.
    """

    def _make(shots_total=1_000_000, config=None):
        problem = Problem(num_qubits=2, observable=[(1.0, "ZI"), (1.0, "IZ")])
        policy = Policy(Budget(shots_total=shots_total), AuditLog())
        return RunContext(
            problem=problem,
            circuit=None,
            device=None,
            policy=policy,
            config=dict(config or {}),
        )

    return _make
