#!/usr/bin/env python3
"""CMA-ES (or random-search fallback) over CodecOp parameter space.

Implements the WIRE verdict from the codegen/genetic/evolutionary
research subagent (memo
``feedback_codegen_genetic_evolutionary_synthesis_VERDICT_20260507.md``).

The CodecOp sweep manifest tool (``codec_op_param_sweep_manifest.py``)
takes a discrete grid of parameter values. This tool generalizes to
**continuous black-box search** over numeric parameter ranges using
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) when the
``cmaes`` library is available, OR a deterministic random-search
fallback otherwise.

Each evaluation:
  1. Samples a parameter vector from the optimizer
  2. Instantiates the CodecOp with those parameters
  3. Runs encode() on the supplied state_dict
  4. Records bytes_out + reconstruction RMS as the fitness signal
  5. Annotates the local byte/RMS Pareto frontier
  6. Optionally appends one planning-only row per eval to the bilevel
     atom ledger

Termination: budget-based (max number of evaluations) or
convergence-based (when sigma drops below a threshold for CMA-ES).

Predicted impact [predicted-band only]: -0.0005 to -0.003 score off
PR106 frontier per single-op tuning pass per the prior subagent's
synthesis. Pilot 2-D before committing to 4-D+ to verify the
landscape is genuinely non-convex.

AGENTS.md compliance:
  - Strict-scorer-rule: pure CPU + torch + CodecOp encode/decode. No
    scorer load and no archive mutation.
  - Random-search fallback is fully deterministic given a seed.
  - Outputs are planning/forensic evidence only. They are not contest
    score claims, dispatch readiness, or charged-bit-change proofs.

Usage::

    .venv/bin/python tools/codec_op_cma_search.py \\
        --module tac.codec_pipeline_kl_pose \\
        --class Op_KLPoseStream \\
        --state-dict-path experiments/.../poses.pt \\
        --param-spec '{"n_components": {"type": "int", "low": 1, "high": 6, "init": 4}, \\
                       "brotli_quality": {"type": "int", "low": 1, "high": 11, "init": 11}}' \\
        --max-evals 50 \\
        --output reports/cma_search.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TOOL_NAME = "tools/codec_op_cma_search.py"
REPORT_SCHEMA = "codec_op_cma_search_report_v1"
LEDGER_ROW_SCHEMA = "codec_op_cma_search_eval_v1"
EVIDENCE_SEMANTICS = "cpu_codec_op_search_forensic"
EVIDENCE_GRADE = "[CPU-prep+cma_search]"
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
    """One parameter's search-space definition."""
    name: str
    type: str  # "int" or "float"
    low: float
    high: float
    init: float
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
    pareto_frontier: bool = False
    pareto_dominated_by: list[int] = field(default_factory=list)


@dataclass
class SearchReport:
    schema: str
    tool: str
    op_module: str
    op_class: str
    n_evaluations: int
    n_successful: int
    n_failed: int
    pareto_frontier_count: int
    best_eval: Evaluation | None
    all_evaluations: list[Evaluation] = field(default_factory=list)
    optimizer: str = "random_search"
    seed: int = 0
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
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise SystemExit(f"could not import module {module!r}: {exc}") from None
    if not hasattr(mod, class_name):
        raise SystemExit(f"module {module!r} has no class {class_name!r}")
    cls = getattr(mod, class_name)
    for required in ("encode", "decode"):
        if not hasattr(cls, required):
            raise SystemExit(
                f"{module}.{class_name} missing CodecOp method {required!r}"
            )
    return cls


def _load_state_dict(path: Path, key: str | None) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise SystemExit(f"state_dict path does not exist: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for name, value in obj.items():
            if not isinstance(name, str):
                raise SystemExit(
                    f"state_dict key {name!r} is not a string"
                )
            if not isinstance(value, torch.Tensor):
                raise SystemExit(
                    f"state_dict value for {name!r} is not a Tensor "
                    f"(got {type(value).__name__})"
                )
        return obj
    if key is None:
        raise SystemExit(
            f"loaded {path} is a tensor not a dict; pass --state-dict-key"
        )
    if not isinstance(obj, torch.Tensor):
        raise SystemExit(
            f"loaded {path} is neither dict nor Tensor "
            f"(got {type(obj).__name__})"
        )
    return {key: obj}


def _parse_param_spec(spec_json: str) -> list[ParamSpec]:
    try:
        raw = json.loads(spec_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--param-spec is not valid JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise SystemExit("--param-spec must be a JSON dict")
    if not raw:
        raise SystemExit("--param-spec must contain at least one parameter")
    specs: list[ParamSpec] = []
    for name, body in raw.items():
        if not isinstance(name, str) or not name:
            raise SystemExit("--param-spec parameter names must be non-empty strings")
        if not isinstance(body, dict):
            raise SystemExit(f"--param-spec entry {name!r} must be a dict")
        param_type = str(body.get("type", "float"))
        if param_type not in {"int", "float"}:
            raise SystemExit(
                f"--param-spec entry {name!r} has unsupported type "
                f"{param_type!r}; expected 'int' or 'float'"
            )
        try:
            low = float(body["low"])
            high = float(body["high"])
            init = float(body.get("init", (low + high) / 2))
        except KeyError as exc:
            raise SystemExit(
                f"--param-spec entry {name!r} missing required key {exc.args[0]!r}"
            ) from None
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"--param-spec entry {name!r} low/high/init must be numeric: {exc}"
            ) from None
        if not (math.isfinite(low) and math.isfinite(high) and math.isfinite(init)):
            raise SystemExit(f"--param-spec entry {name!r} bounds/init must be finite")
        if high < low:
            raise SystemExit(f"--param-spec entry {name!r} high must be >= low")
        log = bool(body.get("log", False))
        if log and (low <= 0.0 or high <= 0.0):
            raise SystemExit(f"--param-spec entry {name!r} log range must be > 0")
        specs.append(ParamSpec(
            name=name,
            type=param_type,
            low=low,
            high=high,
            init=init,
            log=log,
        ))
    return specs


def _coerce(value: float, spec: ParamSpec) -> Any:
    """Clamp + cast a continuous sample to the parameter's actual type."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{spec.name}: sampled value must be finite")
    clamped = max(spec.low, min(spec.high, value))
    if spec.type == "int":
        return round(clamped)
    if spec.type != "float":
        raise ValueError(f"{spec.name}: unsupported parameter type {spec.type!r}")
    return float(clamped)


def _evaluate(
    op_cls,
    params: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    eval_idx: int,
) -> Evaluation:
    """Run one encode/decode cycle and record fitness."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        op = op_cls(**params)
        result = op.encode(state_dict, context={})
        bytes_out = int(result.bytes_out)
        if bytes_out < 0:
            raise ValueError(f"CodecOp returned negative bytes_out={bytes_out}")
        decoded = op.decode(
            result.blob, op_state=result.op_state, context={}
        )
        if not isinstance(decoded, dict):
            raise ValueError(
                f"CodecOp decode returned {type(decoded).__name__}, expected dict"
            )
        # Reconstruction RMS over all tensors that survive the roundtrip
        rms_sum = 0.0
        rms_count = 0
        for k, original in state_dict.items():
            if not isinstance(original, torch.Tensor):
                continue
            if k not in decoded:
                continue
            recon = decoded[k]
            if not isinstance(recon, torch.Tensor):
                continue
            if recon.shape != original.shape:
                continue
            diff = (recon.float() - original.float()).flatten()
            rms_sum += float((diff * diff).mean().item())
            rms_count += 1
        if rms_count == 0:
            raise ValueError("CodecOp decode produced no matching tensor outputs")
        rms = math.sqrt(rms_sum / rms_count)
        # Fitness: bytes_out + 1e6 * RMS (penalty for high reconstruction
        # error; the codec must roundtrip cleanly to be useful). Lower
        # is better.
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


def random_search(
    op_cls,
    state_dict: dict[str, torch.Tensor],
    specs: list[ParamSpec],
    max_evals: int,
    seed: int = 0,
) -> list[Evaluation]:
    """Deterministic random-search fallback.

    Used when the ``cmaes`` library is not available. Samples each
    parameter independently from a uniform distribution over its
    [low, high] range (or log-uniform if ``log=True``).
    """
    if max_evals <= 0:
        return []
    rng = random.Random(seed)
    evaluations: list[Evaluation] = []
    # First evaluate at the init point (sanity baseline)
    init_params = {
        spec.name: _coerce(spec.init, spec) for spec in specs
    }
    evaluations.append(_evaluate(op_cls, init_params, state_dict, 0))
    for i in range(1, max_evals):
        params: dict[str, Any] = {}
        for spec in specs:
            if spec.log and spec.low > 0:
                # log-uniform sample
                log_low, log_high = math.log(spec.low), math.log(spec.high)
                value = math.exp(rng.uniform(log_low, log_high))
            else:
                value = rng.uniform(spec.low, spec.high)
            params[spec.name] = _coerce(value, spec)
        evaluations.append(_evaluate(op_cls, params, state_dict, i))
    return evaluations


def cma_es_search(
    op_cls,
    state_dict: dict[str, torch.Tensor],
    specs: list[ParamSpec],
    max_evals: int,
    seed: int = 0,
) -> list[Evaluation]:
    """CMA-ES search (when ``cmaes`` library is available).

    Falls back to random-search if the library is missing.
    """
    if max_evals <= 0:
        return []
    try:
        from cmaes import CMA  # type: ignore[import-not-found]
    except ImportError:
        return random_search(op_cls, state_dict, specs, max_evals, seed=seed)

    import numpy as np

    bounds = np.array([[s.low, s.high] for s in specs], dtype=np.float64)
    init_mean = np.array([s.init for s in specs], dtype=np.float64)
    sigma = 0.5 * float(np.mean(bounds[:, 1] - bounds[:, 0])) / 4.0
    optimizer = CMA(
        mean=init_mean,
        sigma=sigma,
        bounds=bounds,
        seed=seed,
    )
    evaluations: list[Evaluation] = []
    eval_idx = 0
    while eval_idx < max_evals and not optimizer.should_stop():
        # cmaes.CMA.tell() requires exactly popsize-length solutions, so always
        # fill a full generation even if it slightly exceeds max_evals (better
        # than crashing on a short tell()). The slight overshoot is bounded by
        # population_size and serves the user's budget intent.
        popsize = optimizer.population_size
        solutions = []
        for _ in range(popsize):
            x = optimizer.ask()
            params = {
                spec.name: _coerce(x[i], spec)
                for i, spec in enumerate(specs)
            }
            ev = _evaluate(op_cls, params, state_dict, eval_idx)
            evaluations.append(ev)
            objective = (
                float(ev.fitness)
                if ev.fitness is not None and math.isfinite(ev.fitness)
                else FAILED_FITNESS_PENALTY
            )
            solutions.append((x, objective))
            eval_idx += 1
        # Always a full popsize batch; never short.
        optimizer.tell(solutions)
    return evaluations


def _valid_for_pareto(ev: Evaluation) -> bool:
    return (
        ev.error is None
        and ev.bytes_out >= 0
        and ev.reconstruction_rms is not None
        and math.isfinite(ev.reconstruction_rms)
        and ev.fitness is not None
        and math.isfinite(ev.fitness)
    )


def annotate_pareto_frontier(evaluations: list[Evaluation]) -> None:
    """Mark local non-dominated evaluations over (bytes_out, RMS).

    This is a CPU planning frontier only. It ranks CodecOp parameter
    settings for follow-up without implying archive readiness or score truth.
    """
    valid = [ev for ev in evaluations if _valid_for_pareto(ev)]
    for ev in evaluations:
        ev.pareto_frontier = False
        ev.pareto_dominated_by = []
    for ev in valid:
        dominated_by: list[int] = []
        assert ev.reconstruction_rms is not None
        for other in valid:
            if other.eval_idx == ev.eval_idx:
                continue
            assert other.reconstruction_rms is not None
            no_worse = (
                other.bytes_out <= ev.bytes_out
                and other.reconstruction_rms <= ev.reconstruction_rms
            )
            strictly_better = (
                other.bytes_out < ev.bytes_out
                or other.reconstruction_rms < ev.reconstruction_rms
            )
            if no_worse and strictly_better:
                dominated_by.append(other.eval_idx)
        ev.pareto_dominated_by = sorted(dominated_by)
        ev.pareto_frontier = not dominated_by


def _evaluation_sort_key(ev: Evaluation) -> tuple[Any, ...]:
    return (
        ev.error is not None,
        not ev.pareto_frontier,
        ev.fitness if ev.fitness is not None else FAILED_FITNESS_PENALTY,
        ev.bytes_out if ev.bytes_out >= 0 else sys.maxsize,
        (
            ev.reconstruction_rms
            if ev.reconstruction_rms is not None
            else float("inf")
        ),
        ev.eval_idx,
    )


def _param_space_payload(specs: list[ParamSpec]) -> list[dict[str, Any]]:
    return [asdict(spec) for spec in specs]


def append_atom_ledger_rows(
    ledger_path: Path,
    evaluations: list[Evaluation],
    *,
    op_module: str,
    op_class: str,
    substrate_label: str,
) -> None:
    """Append one row per evaluation to the bilevel atom ledger.

    Rows are intentionally planning-only. A CodecOp parameter trial has
    not created a substituted archive, changed charged contest bytes, or
    passed exact CUDA auth eval, so it must not advertise dispatch readiness.
    """
    annotate_pareto_frontier(evaluations)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as f:
        for ev in evaluations:
            blockers = list(DISPATCH_BLOCKERS)
            if ev.error is not None:
                blockers.append("evaluation_failed")
            row = {
                "schema": LEDGER_ROW_SCHEMA,
                "tool": TOOL_NAME,
                "timestamp_utc": ev.timestamp_utc,
                "phase": None,
                "atom_id": f"{substrate_label}/eval_{ev.eval_idx}",
                "family": "codec_op_param_search",
                "family_group": f"codec_op_param_search:{op_class}",
                "pareto_scope": f"{op_module}.{op_class}",
                "substrate_label": f"{substrate_label}/eval_{ev.eval_idx}",
                "cathedral_op": f"{op_module}.{op_class}",
                "op_params": ev.params,
                "byte_delta": 0,
                "bytes_out": ev.bytes_out,
                "reconstruction_rms": ev.reconstruction_rms,
                "fitness": ev.fitness,
                "pareto_frontier": ev.pareto_frontier,
                "pareto_dominated_by": list(ev.pareto_dominated_by),
                "error": ev.error,
                "evidence_grade": EVIDENCE_GRADE,
                "evidence_semantics": EVIDENCE_SEMANTICS,
                "target_modes": list(TARGET_MODES),
                "deployment_target": DEPLOYMENT_TARGET,
                "ready_for_exact_eval_dispatch": False,
                "field_selection_ready_for_exact_eval_dispatch": False,
                "dispatchable": False,
                "dispatch_attempted": False,
                "score_claim": False,
                "score_affecting_payload_changed": False,
                "charged_bits_changed": False,
                "dispatch_blockers": blockers,
                "planning_objectives": {
                    "bytes_out": ev.bytes_out,
                    "reconstruction_rms": ev.reconstruction_rms,
                    "fitness": ev.fitness,
                },
                "notes": (
                    "CMA-ES/random CodecOp parameter-search trial; CPU-only "
                    "forensic planning row, no archive substitution or score claim."
                ),
            }
            f.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument(
        "--param-spec", required=True,
        help="JSON dict mapping param name to "
        "{type: int|float, low, high, init, log?}",
    )
    parser.add_argument("--max-evals", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizer", choices=["cma_es", "random"], default="cma_es",
        help="Optimizer choice. cma_es falls back to random if cmaes "
        "library is not installed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--atom-ledger-output", type=Path, default=None,
        help="If set, append one row per eval to this JSONL ledger.",
    )
    parser.add_argument(
        "--substrate-label", default="cma_search",
        help="Prefix for atom-ledger substrate_label.",
    )
    args = parser.parse_args(argv)
    if args.max_evals < 1:
        raise SystemExit("--max-evals must be >= 1")

    op_cls = _import_codec_op(args.module, args.class_name)
    state_dict = _load_state_dict(args.state_dict_path, args.state_dict_key)
    specs = _parse_param_spec(args.param_spec)

    if args.optimizer == "cma_es":
        evaluations = cma_es_search(
            op_cls, state_dict, specs, args.max_evals, seed=args.seed,
        )
    else:
        evaluations = random_search(
            op_cls, state_dict, specs, args.max_evals, seed=args.seed,
        )
    annotate_pareto_frontier(evaluations)

    optimizer_name = (
        "cma_es" if args.optimizer == "cma_es"
        and _cmaes_available() else "random_search"
    )
    valid_evals = [e for e in evaluations if _valid_for_pareto(e)]
    best = min(valid_evals, key=_evaluation_sort_key, default=None)
    report = SearchReport(
        schema=REPORT_SCHEMA,
        tool=TOOL_NAME,
        op_module=args.module,
        op_class=args.class_name,
        n_evaluations=len(evaluations),
        n_successful=len(valid_evals),
        n_failed=len([e for e in evaluations if e.error is not None]),
        pareto_frontier_count=len([e for e in evaluations if e.pareto_frontier]),
        best_eval=best,
        all_evaluations=evaluations,
        optimizer=optimizer_name,
        seed=args.seed,
        parameter_space=_param_space_payload(specs),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(report), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} (n_evals={len(evaluations)}, "
          f"n_failed={report.n_failed}, optimizer={optimizer_name})")
    if best:
        print(
            f"best: eval_idx={best.eval_idx} bytes_out={best.bytes_out} "
            f"rms={best.reconstruction_rms:.6f} fitness={best.fitness:.2f} "
            f"params={best.params}"
        )

    if args.atom_ledger_output:
        append_atom_ledger_rows(
            args.atom_ledger_output, evaluations,
            op_module=args.module, op_class=args.class_name,
            substrate_label=args.substrate_label,
        )
        print(f"appended {len(evaluations)} rows to {args.atom_ledger_output}")
    return 0


def _cmaes_available() -> bool:
    try:
        import cmaes  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
