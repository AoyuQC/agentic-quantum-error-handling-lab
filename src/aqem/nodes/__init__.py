"""The eight QEM DAG stages and a factory assembling them in order."""

from .circuit_generate import CircuitGenerateNode
from .empirical_probe import EmpiricalProbeNode
from .execute import ExecuteNode
from .post_process import PostProcessNode
from .readout_calibrate import ReadoutCalibrateNode
from .report import ReportNode
from .strategy_select import StrategySelectNode
from .validate import ValidateNode


def default_nodes() -> list:
    """The standard QEM pipeline:

    empirical_probe -> strategy_select -> readout_calibrate -> circuit_generate
    -> execute -> post_process -> validate -> report
    """
    return [
        EmpiricalProbeNode(),
        StrategySelectNode(),
        ReadoutCalibrateNode(),
        CircuitGenerateNode(),
        ExecuteNode(),
        PostProcessNode(),
        ValidateNode(),
        ReportNode(),
    ]


__all__ = [
    "EmpiricalProbeNode",
    "StrategySelectNode",
    "ReadoutCalibrateNode",
    "CircuitGenerateNode",
    "ExecuteNode",
    "PostProcessNode",
    "ValidateNode",
    "ReportNode",
    "default_nodes",
]
