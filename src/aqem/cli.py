"""Command-line interface for the Agentic QEM Lab.

    aqem baseline   run the static full-stack Mitiq baseline
    aqem run        run the adaptive VLM-guided loop
    aqem report     run both and print the efficiency comparison

All commands run locally on a Braket LocalSimulator noise model (no AWS). The
adaptive loop uses the VLM only if ``--vlm`` is given; otherwise it runs the
deterministic rules path. ``report`` is the headline demo: it shows the adaptive
loop reaching the target with fewer shots than the blind full-stack baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

from .baseline.full_stack import BaselineConfig, run_full_stack_baseline
from .config import load_config, resolve_device, run_config
from .loop import run_adaptive_loop
from .models import Budget
from .problems import default_problem, ideal_expectation
from .reporting.efficiency import accuracy_point, compare

# Printed immediately before a JSON blob so callers can split it out cleanly.
_JSON_MARKER = "---JSON---"


def _build_problem(args):
    return default_problem(num_qubits=args.qubits, target_accuracy=args.target)


def _build_vlm(cfg: dict[str, Any], enabled: bool):
    if not enabled:
        return None
    from .vlm import get_vlm_client

    return get_vlm_client(cfg.get("vlm", {}))


def _baseline_config(cfg: dict[str, Any]) -> BaselineConfig:
    b = cfg.get("baseline", {})
    return BaselineConfig(
        shot_per_base=b.get("shot_per_base", 4000),
        overhead=b.get("overhead", 3),
        scale_factors=b.get("zne_scale_factors", [1, 3, 7]),
        num_twirls=b.get("twirl_count", 16),
        rem_twirls=b.get("rem_twirls", 50),
        zne_factory=b.get("zne_factory", "Exp"),
    )


def cmd_baseline(args) -> int:
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.get("noise", {}).get("device", "qd_total"))
    problem, circuit = _build_problem(args)
    ideal = ideal_expectation(circuit, problem.observable)

    estimate = run_full_stack_baseline(problem, circuit, device, _baseline_config(cfg))
    point = accuracy_point("baseline", estimate, ideal)

    print(f"[baseline] {problem.description}")
    print(f"  ideal        = {ideal:.4f}")
    print(f"  estimate     = {estimate.value:.4f} +/- {estimate.error_bar:.4f}")
    print(f"  error        = {point.error:.4f}  (target {problem.target_accuracy})")
    print(f"  techniques   = {estimate.techniques}")
    print(f"  shots used   = {estimate.shots_used}")
    if args.json:
        print(json.dumps({"estimate": estimate.to_dict(), "ideal": ideal, "error": point.error}, indent=2))
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.get("noise", {}).get("device", "qd_total"))
    problem, circuit = _build_problem(args)
    ideal = ideal_expectation(circuit, problem.observable)
    budget = Budget(shots_total=args.budget or cfg.get("budget", {}).get("shots_total"))
    vlm = _build_vlm(cfg, args.vlm)

    record = run_adaptive_loop(
        problem, circuit, device, budget,
        config=run_config(cfg), vlm=vlm, audit_path=args.audit, seed=args.seed,
    )
    est = record.final_outputs.get("estimate")
    print(f"[adaptive] {problem.description}  (status={record.status}, iterations={record.iterations})")
    print(f"  ideal        = {ideal:.4f}")
    if est:
        err = abs(est["value"] - ideal)
        print(f"  estimate     = {est['value']:.4f} +/- {est['error_bar']:.4f}")
        print(f"  error        = {err:.4f}  (target {problem.target_accuracy})")
        print(f"  techniques   = {est['techniques']}")
    print(f"  shots used   = {record.final_outputs.get('shots_used')}")
    print(f"  decision     = {(record.final_outputs.get('decision') or {}).get('reason', '')}")
    if args.json:
        print(json.dumps(record.to_dict(), indent=2, default=str))
    return 0


def cmd_report(args) -> int:
    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg.get("noise", {}).get("device", "qd_total"))
    problem, circuit = _build_problem(args)
    ideal = ideal_expectation(circuit, problem.observable)
    budget = Budget(shots_total=args.budget or cfg.get("budget", {}).get("shots_total"))
    vlm = _build_vlm(cfg, args.vlm)

    print("Running static full-stack baseline ...")
    baseline_est = run_full_stack_baseline(problem, circuit, device, _baseline_config(cfg))

    print("Running adaptive loop ...")
    record = run_adaptive_loop(
        problem, circuit, device, budget,
        config=run_config(cfg), vlm=vlm, audit_path=args.audit, seed=args.seed,
    )
    adaptive_est = _record_estimate(record)
    if adaptive_est is None:
        print("adaptive loop produced no estimate; aborting report", file=sys.stderr)
        return 1

    cmp = compare(adaptive_est, baseline_est, ideal, problem.target_accuracy)

    print()
    print(f"=== Efficiency comparison: {problem.description} ===")
    print(f"  ideal reference        : {ideal:.4f}   (target accuracy {problem.target_accuracy})")
    print(f"  {'':18}{'shots':>10}{'error':>10}{'techniques':>22}")
    print(f"  {'baseline':18}{cmp.baseline.shots:>10}{cmp.baseline.error:>10.4f}{str(cmp.baseline.techniques):>22}")
    print(f"  {'adaptive':18}{cmp.adaptive.shots:>10}{cmp.adaptive.error:>10.4f}{str(cmp.adaptive.techniques):>22}")
    print()
    print(f"  adaptive meets target  : {cmp.adaptive_meets_target}")
    print(f"  shots saved            : {cmp.shots_saved}")
    if cmp.shot_ratio:
        print(f"  shot ratio (base/adapt): {cmp.shot_ratio:.2f}x")
    print(f"  EFFICIENCY GAIN SHOWN   : {cmp.efficiency_gain_demonstrated}")

    if args.out:
        _write_figures(args.out, cmp)
        print(f"\n  figures + comparison written to {args.out}/")
    if args.json:
        print(_JSON_MARKER)
        print(json.dumps(cmp.to_dict(), indent=2))
    return 0


def _record_estimate(record):
    from .models import Estimate

    est = record.final_outputs.get("estimate")
    return Estimate.from_dict(est) if est else None


def _write_figures(out_dir: str, cmp) -> None:
    import os

    from .reporting.plots import accuracy_vs_shots_figure

    os.makedirs(out_dir, exist_ok=True)
    fig = accuracy_vs_shots_figure(cmp)
    with open(os.path.join(out_dir, "accuracy_vs_shots.json"), "w") as fh:
        json.dump(fig, fh, indent=2)
    with open(os.path.join(out_dir, "comparison.json"), "w") as fh:
        json.dump(cmp.to_dict(), fh, indent=2)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to config YAML (default: config/default.yaml)")
    p.add_argument("--device", default=None, help="noise-model device name (e.g. qd_readout_2, qd_total)")
    p.add_argument("--qubits", type=int, default=2, help="number of qubits")
    p.add_argument("--target", type=float, default=0.06, help="target absolute accuracy")
    p.add_argument("--json", action="store_true", help="also emit JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aqem", description="Agentic Quantum Error Mitigation Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("baseline", help="run the static full-stack Mitiq baseline")
    _add_common(p_base)
    p_base.set_defaults(func=cmd_baseline)

    p_run = sub.add_parser("run", help="run the adaptive VLM-guided loop")
    _add_common(p_run)
    p_run.add_argument("--budget", type=int, default=None, help="total shot budget")
    p_run.add_argument("--vlm", action="store_true", help="enable the VLM (default: rules-only)")
    p_run.add_argument("--audit", default=None, help="path to write the audit JSONL")
    p_run.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="run baseline + adaptive and compare efficiency")
    _add_common(p_rep)
    p_rep.add_argument("--budget", type=int, default=None, help="total shot budget for the adaptive loop")
    p_rep.add_argument("--vlm", action="store_true", help="enable the VLM for the adaptive loop")
    p_rep.add_argument("--audit", default=None, help="path to write the adaptive audit JSONL")
    p_rep.add_argument("--seed", type=int, default=7, help="RNG seed (default 7 for a reproducible demo)")
    p_rep.add_argument("--out", default=None, help="directory to write comparison figures/JSON")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
