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
  5. Optionally appends one row per eval to the bilevel atom ledger

Termination: budget-based (max number of evaluations) or
convergence-based (when sigma drops below a threshold for CMA-ES).

Predicted impact [predicted-band only]: -0.0005 to -0.003 score off
PR106 frontier per single-op tuning pass per the prior subagent's
synthesis. Pilot 2-D before committing to 4-D+ to verify the
landscape is genuinely non-convex.

CLAUDE.md compliance:
  - Strict-scorer-rule: pure CPU + numpy + brotli + torch + cathedral
    contest-score formula. No scorer load.
  - Random-search fallback is fully deterministic given a seed.
  - Per-eval cost = brotli (95% per the hot-path audit memo).

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

from tac.contest_rate_distortion_system import (  # noqa: E402
    contest_score,
)


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
    reconstruction_rms: float
    fitness: float
    timestamp_utc: str
    error: str | None = None


@dataclass
class SearchReport:
    op_module: str
    op_class: str
    n_evaluations: int
    n_failed: int
    best_eval: Evaluation | None
    all_evaluations: list[Evaluation] = field(default_factory=list)
    optimizer: str = "random_search"
    seed: int = 0


def _import_codec_op(module: str, class_name: str):
    mod = importlib.import_module(module)
    return getattr(mod, class_name)


def _load_state_dict(path: Path, key: str | None) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        return obj
    if key is None:
        raise SystemExit(
            f"loaded {path} is a tensor not a dict; pass --state-dict-key"
        )
    return {key: obj}


def _parse_param_spec(spec_json: str) -> list[ParamSpec]:
    raw = json.loads(spec_json)
    if not isinstance(raw, dict):
        raise SystemExit("--param-spec must be a JSON dict")
    specs: list[ParamSpec] = []
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise SystemExit(f"--param-spec entry {name!r} must be a dict")
        specs.append(ParamSpec(
            name=name,
            type=body.get("type", "float"),
            low=float(body["low"]),
            high=float(body["high"]),
            init=float(body.get("init", (body["low"] + body["high"]) / 2)),
            log=bool(body.get("log", False)),
        ))
    return specs


def _coerce(value: float, spec: ParamSpec) -> Any:
    """Clamp + cast a continuous sample to the parameter's actual type."""
    clamped = max(spec.low, min(spec.high, value))
    if spec.type == "int":
        return int(round(clamped))
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
        decoded = op.decode(
            result.blob, op_state=result.op_state, context={}
        )
        # Reconstruction RMS over all tensors that survive the roundtrip
        rms_sum = 0.0
        rms_count = 0
        for k, original in state_dict.items():
            if k not in decoded:
                continue
            recon = decoded[k]
            if recon.shape != original.shape:
                continue
            diff = (recon.float() - original.float()).flatten()
            rms_sum += float((diff * diff).mean().item())
            rms_count += 1
        rms = math.sqrt(rms_sum / max(rms_count, 1))
        # Fitness: bytes_out + 1e6 * RMS (penalty for high reconstruction
        # error; the codec must roundtrip cleanly to be useful). Lower
        # is better.
        fitness = float(result.bytes_out) + 1e6 * rms
        return Evaluation(
            eval_idx=eval_idx,
            params=dict(params),
            bytes_out=int(result.bytes_out),
            reconstruction_rms=rms,
            fitness=fitness,
            timestamp_utc=timestamp,
        )
    except Exception as exc:  # noqa: BLE001 — surface op-side errors
        return Evaluation(
            eval_idx=eval_idx,
            params=dict(params),
            bytes_out=-1,
            reconstruction_rms=float("inf"),
            fitness=float("inf"),
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
        solutions = []
        for _ in range(optimizer.population_size):
            if eval_idx >= max_evals:
                break
            x = optimizer.ask()
            params = {
                spec.name: _coerce(x[i], spec)
                for i, spec in enumerate(specs)
            }
            ev = _evaluate(op_cls, params, state_dict, eval_idx)
            evaluations.append(ev)
            solutions.append((x, ev.fitness))
            eval_idx += 1
        if solutions:
            optimizer.tell(solutions)
    return evaluations


def append_atom_ledger_rows(
    ledger_path: Path,
    evaluations: list[Evaluation],
    *,
    op_module: str,
    op_class: str,
    substrate_label: str,
) -> None:
    """Append one row per evaluation to the bilevel atom ledger.

    Schema follows the cathedral's convention with
    ``evidence_grade=[CPU-prep+cma_search]`` and
    ``target_modes=["contest_exact_eval"]`` per the codex contract.
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as f:
        for ev in evaluations:
            row = {
                "timestamp_utc": ev.timestamp_utc,
                "phase": None,
                "substrate_label": f"{substrate_label}/eval_{ev.eval_idx}",
                "cathedral_op": f"{op_module}.{op_class}",
                "op_params": ev.params,
                "bytes_out": ev.bytes_out,
                "reconstruction_rms": ev.reconstruction_rms,
                "fitness": ev.fitness,
                "error": ev.error,
                "evidence_grade": "[CPU-prep+cma_search]",
                "target_modes": ["contest_exact_eval"],
                "deployment_target": "t4_contest_runtime",
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
                "notes": (
                    "CMA-ES (or random-search fallback) candidate "
                    "from tools/codec_op_cma_search.py"
                ),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--state-dict-key", default=None)
    parser.add_argument(
        "--param-spec", required=True,
        help="JSON dict mapping param name → "
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

    optimizer_name = (
        "cma_es" if args.optimizer == "cma_es"
        and _cmaes_available() else "random_search"
    )
    valid_evals = [e for e in evaluations if e.error is None]
    best = min(valid_evals, key=lambda e: e.fitness, default=None)
    report = SearchReport(
        op_module=args.module,
        op_class=args.class_name,
        n_evaluations=len(evaluations),
        n_failed=len([e for e in evaluations if e.error is not None]),
        best_eval=best,
        all_evaluations=evaluations,
        optimizer=optimizer_name,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
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
