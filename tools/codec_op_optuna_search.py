#!/usr/bin/env python3
"""Optuna TPE wrapper as alternative to CMA-ES on integer-only param spaces.

Per the cmaes-adjacent-libraries hypernerd verdict (memo
``feedback_cmaes_adjacent_libraries_hypernerd_verdict_20260507.md``),
Optuna's TPE (Tree-structured Parzen Estimator) is native on integer
distributions and typically beats CMA-ES on integer-heavy parameter
spaces (brotli quality, lgwin, lgblock are all int).

This is a parallel implementation to ``tools/codec_op_cma_search.py``.
Same CodecOp protocol, same atom-ledger schema, same custody fields per
codex dispatch-gate contract, but uses Optuna's ``IntDistribution`` and
``TPESampler`` instead of cmaes's continuous CMA + integer rounding.

CLAUDE.md compliance: pure CPU + torch + Optuna; no scorer load; no
archive mutation. Outputs are planning evidence only, not contest score
claims.

Usage::

    .venv/bin/python tools/codec_op_optuna_search.py \\
        --module tac.codec_pipeline \\
        --class Op1_PR101SplitBrotli \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --param-spec '{"brotli_quality": {"type": "int", "low": 1, "high": 11},
                       "brotli_lgwin": {"type": "int", "low": 10, "high": 24},
                       "brotli_lgblock": {"type": "int", "low": 16, "high": 24}}' \\
        --max-evals 100 \\
        --seed 42 \\
        --output reports/optuna_pr101_search.json

The output schema matches ``codec_op_cma_search`` so downstream consumers
(atom ledger, dispatch advisor, Pareto frontier ranker) work unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TOOL_NAME = "tools/codec_op_optuna_search.py"
REPORT_SCHEMA = "codec_op_optuna_search_report_v1"
LEDGER_ROW_SCHEMA = "codec_op_optuna_search_eval_v1"
EVIDENCE_SEMANTICS = "cpu_codec_op_search_forensic"
EVIDENCE_GRADE = "[CPU-prep+optuna_tpe]"
TARGET_MODES = ["contest_exact_eval_planning"]
DEPLOYMENT_TARGET = "desktop_research"
DISPATCH_BLOCKERS = [
    "cpu_only_codec_op_search",
    "no_archive_substitution_performed",
    "no_score_affecting_payload_change_proof",
    "missing_byte_closed_archive_manifest",
    "missing_exact_cuda_auth_eval",
]
FAILED_FITNESS_PENALTY = 1.0e30


@dataclass
class ParamSpec:
    name: str
    type: str  # "int" or "float"
    low: float
    high: float
    log: bool = False


@dataclass
class Evaluation:
    eval_idx: int
    params: dict[str, Any]
    bytes_out: int
    reconstruction_rms: float | None
    fitness: float | None
    timestamp_utc: str
    error: str | None = None


@dataclass
class SearchReport:
    schema: str
    tool: str
    op_module: str
    op_class: str
    n_evaluations: int
    n_successful: int
    n_failed: int
    best_eval: Evaluation | None
    all_evaluations: list[Evaluation] = field(default_factory=list)
    optimizer: str = "optuna_tpe"
    seed: int = 0
    sampler_kind: str = "TPESampler"
    evidence_semantics: str = EVIDENCE_SEMANTICS
    target_modes: list[str] = field(default_factory=lambda: list(TARGET_MODES))
    deployment_target: str = DEPLOYMENT_TARGET
    ready_for_exact_eval_dispatch: bool = False
    score_claim: bool = False
    dispatch_attempted: bool = False
    score_affecting_payload_changed: bool = False
    charged_bits_changed: bool = False
    dispatch_blockers: list[str] = field(default_factory=lambda: list(DISPATCH_BLOCKERS))
    parameter_space: list[dict[str, Any]] = field(default_factory=list)


def _import_codec_op(module: str, class_name: str):
    mod = importlib.import_module(module)
    if not hasattr(mod, class_name):
        raise SystemExit(f"module {module!r} has no class {class_name!r}")
    return getattr(mod, class_name)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise SystemExit(f"state_dict path does not exist: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return obj
    raise SystemExit(f"loaded {path} is not a dict")


def _parse_param_spec(spec_json: str) -> list[ParamSpec]:
    raw = json.loads(spec_json)
    specs: list[ParamSpec] = []
    for name, body in raw.items():
        param_type = str(body.get("type", "float"))
        if param_type not in {"int", "float"}:
            raise SystemExit(f"--param-spec entry {name!r} has unsupported type {param_type!r}")
        specs.append(ParamSpec(
            name=name,
            type=param_type,
            low=float(body["low"]),
            high=float(body["high"]),
            log=bool(body.get("log", False)),
        ))
    return specs


def _evaluate(
    op_cls,
    params: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    eval_idx: int,
) -> Evaluation:
    """Instantiate CodecOp with params, encode/decode, return metrics."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        op = op_cls(**params)
        result = op.encode(state_dict, context={})
        bytes_out = int(result.bytes_out)
        if bytes_out < 0:
            raise ValueError(f"CodecOp returned negative bytes_out={bytes_out}")
        decoded = op.decode(
            result.blob,
            op_state=result.op_state,
            context={},
        )
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        rms_sum = 0.0
        rms_count = 0
        for k, original in state_dict.items():
            if not isinstance(original, torch.Tensor):
                continue
            recon = decoded.get(k)
            if not isinstance(recon, torch.Tensor):
                continue
            if recon.shape != original.shape:
                continue
            diff = (recon.float() - original.float()).flatten()
            rms_sum += float((diff * diff).mean().item())
            rms_count += 1
        if rms_count == 0:
            raise ValueError("no matching tensor outputs in decode")
        rms = math.sqrt(rms_sum / rms_count)
        fitness = float(bytes_out) + 1e6 * rms
        return Evaluation(
            eval_idx=eval_idx,
            params=dict(params),
            bytes_out=bytes_out,
            reconstruction_rms=rms,
            fitness=fitness,
            timestamp_utc=timestamp,
        )
    except Exception as exc:
        return Evaluation(
            eval_idx=eval_idx,
            params=dict(params),
            bytes_out=-1,
            reconstruction_rms=None,
            fitness=None,
            timestamp_utc=timestamp,
            error=f"{type(exc).__name__}: {exc}",
        )


def optuna_tpe_search(
    op_cls,
    state_dict: dict[str, torch.Tensor],
    specs: list[ParamSpec],
    max_evals: int,
    seed: int = 0,
) -> list[Evaluation]:
    """TPE-based Bayesian optimization over CodecOp param space."""
    if max_evals <= 0:
        return []
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    evaluations: list[Evaluation] = []
    eval_counter = [0]  # closure-mutable

    def objective(trial):
        idx = eval_counter[0]
        eval_counter[0] += 1
        params: dict[str, Any] = {}
        for spec in specs:
            if spec.type == "int":
                v = trial.suggest_int(spec.name, int(spec.low), int(spec.high))
            else:
                v = trial.suggest_float(
                    spec.name, spec.low, spec.high, log=spec.log
                )
            params[spec.name] = v
        ev = _evaluate(op_cls, params, state_dict, idx)
        evaluations.append(ev)
        return (
            float(ev.fitness)
            if ev.fitness is not None and math.isfinite(ev.fitness)
            else FAILED_FITNESS_PENALTY
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=max_evals)
    return evaluations


def _build_search_report(
    evaluations: list[Evaluation],
    *,
    op_module: str,
    op_class: str,
    seed: int,
    specs: list[ParamSpec],
) -> SearchReport:
    successful = [e for e in evaluations if e.error is None and e.fitness is not None]
    failed = [e for e in evaluations if e.error is not None or e.fitness is None]
    best = None
    if successful:
        best = min(successful, key=lambda e: e.fitness if e.fitness is not None else float("inf"))
    return SearchReport(
        schema=REPORT_SCHEMA,
        tool=TOOL_NAME,
        op_module=op_module,
        op_class=op_class,
        n_evaluations=len(evaluations),
        n_successful=len(successful),
        n_failed=len(failed),
        best_eval=best,
        all_evaluations=evaluations,
        optimizer="optuna_tpe",
        seed=seed,
        parameter_space=[asdict(s) for s in specs],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, help="Module containing CodecOp class")
    parser.add_argument("--class", dest="class_name", required=True, help="CodecOp class name")
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--state-dict-key", default=None,
                        help="If state_dict is a single tensor, key under which to wrap it")
    parser.add_argument("--param-spec", required=True,
                        help="JSON dict: {param_name: {type, low, high, [log]}}")
    parser.add_argument("--max-evals", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atom-ledger-output", type=Path, default=None,
                        help="Optional JSONL path for atom-ledger rows (one per eval)")
    args = parser.parse_args(argv)

    op_cls = _import_codec_op(args.module, args.class_name)
    state_dict = _load_state_dict(args.state_dict_path)
    specs = _parse_param_spec(args.param_spec)

    evaluations = optuna_tpe_search(
        op_cls, state_dict, specs, max_evals=args.max_evals, seed=args.seed
    )
    report = _build_search_report(
        evaluations, op_module=args.module, op_class=args.class_name,
        seed=args.seed, specs=specs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {args.output} (n_evals={report.n_evaluations}, "
        f"n_failed={report.n_failed}, optimizer={report.optimizer})"
    )
    if report.best_eval is not None:
        b = report.best_eval
        print(
            f"best: eval_idx={b.eval_idx} bytes_out={b.bytes_out} "
            f"rms={b.reconstruction_rms:.6f} fitness={b.fitness:.2f} "
            f"params={b.params}"
        )

    if args.atom_ledger_output:
        args.atom_ledger_output.parent.mkdir(parents=True, exist_ok=True)
        with args.atom_ledger_output.open("w", encoding="utf-8") as fp:
            for ev in evaluations:
                row = {
                    "schema": LEDGER_ROW_SCHEMA,
                    "tool": TOOL_NAME,
                    "evidence_grade": EVIDENCE_GRADE,
                    "evidence_semantics": EVIDENCE_SEMANTICS,
                    "target_modes": list(TARGET_MODES),
                    "deployment_target": DEPLOYMENT_TARGET,
                    "score_affecting_payload_changed": False,
                    "charged_bits_changed": False,
                    "dispatch_blockers": list(DISPATCH_BLOCKERS),
                    "op_module": args.module,
                    "op_class": args.class_name,
                    "eval_idx": ev.eval_idx,
                    "params": ev.params,
                    "bytes_out": ev.bytes_out,
                    "reconstruction_rms": ev.reconstruction_rms,
                    "fitness": ev.fitness,
                    "timestamp_utc": ev.timestamp_utc,
                    "error": ev.error,
                }
                fp.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"appended {len(evaluations)} rows to {args.atom_ledger_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
