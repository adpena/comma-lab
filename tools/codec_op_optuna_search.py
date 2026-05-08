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
import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
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
MAX_EVAL_SEMANTICS = "hard_cap_no_overshoot"
PARTIAL_DECODE_WAIVER_BLOCKER = "partial_decode_coverage_waived"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ParamSpec:
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
    expected_tensor_count: int = 0
    matched_tensor_count: int = 0
    missing_tensor_keys: list[str] = field(default_factory=list)
    non_tensor_decoded_keys: list[str] = field(default_factory=list)
    shape_mismatch_tensor_keys: list[str] = field(default_factory=list)
    decoded_tensor_keys: list[str] = field(default_factory=list)
    matched_tensor_keys: list[str] = field(default_factory=list)
    decode_coverage_required: bool = True
    partial_decode_waived: bool = False
    partial_decode_waiver_reason: str | None = None
    decode_coverage_status: str = "unknown"
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
    optimizer: str = "optuna_tpe"
    seed: int = 0
    sampler_kind: str = "TPESampler"
    optuna_version: str | None = None
    generated_at_utc: str = ""
    evidence_grade: str = EVIDENCE_GRADE
    evidence_semantics: str = EVIDENCE_SEMANTICS
    target_modes: list[str] = field(default_factory=lambda: list(TARGET_MODES))
    deployment_target: str = DEPLOYMENT_TARGET
    ready_for_exact_eval_dispatch: bool = False
    field_selection_ready_for_exact_eval_dispatch: bool = False
    dispatchable: bool = False
    promotion_eligible: bool = False
    score_claim: bool = False
    dispatch_attempted: bool = False
    score_affecting_payload_changed: bool = False
    charged_bits_changed: bool = False
    exact_cuda_auth_eval: bool = False
    cuda_auth_eval_artifact: str | None = None
    archive_path: str | None = None
    archive_sha256: str | None = None
    archive_bytes: int | None = None
    dispatch_blockers: list[str] = field(default_factory=lambda: list(DISPATCH_BLOCKERS))
    parameter_space: list[dict[str, Any]] = field(default_factory=list)
    requested_max_evals: int | None = None
    max_eval_semantics: str = MAX_EVAL_SEMANTICS
    baseline_required: bool = True
    baseline_eval_idx: int | None = None
    baseline_params: dict[str, Any] = field(default_factory=dict)
    baseline_status: str = "not_evaluated"
    state_dict_path: str | None = None
    state_dict_sha256: str | None = None
    state_dict_bytes: int | None = None
    state_dict_key: str | None = None
    tensor_contract: list[dict[str, Any]] = field(default_factory=list)
    decode_coverage_required: bool = True
    partial_decode_waived: bool = False
    partial_decode_waiver_reason: str | None = None


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_contract(state_dict: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    contract: list[dict[str, Any]] = []
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        contract.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
            }
        )
    return contract


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
        if param_type == "int" and (
            not low.is_integer() or not high.is_integer()
        ):
            raise SystemExit(
                f"--param-spec entry {name!r} int bounds must be integers"
            )
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
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{spec.name}: sampled value must be finite")
    clamped = max(spec.low, min(spec.high, value))
    if spec.type == "int":
        return round(clamped)
    if spec.type != "float":
        raise ValueError(f"{spec.name}: unsupported parameter type {spec.type!r}")
    return float(clamped)


def _init_params(specs: list[ParamSpec]) -> dict[str, Any]:
    return {spec.name: _coerce(spec.init, spec) for spec in specs}


def _validate_partial_decode_waiver(
    allow_partial_decode: bool,
    waiver_reason: str | None,
) -> str | None:
    normalized = waiver_reason.strip() if waiver_reason else None
    if allow_partial_decode and not normalized:
        raise SystemExit(
            "--partial-decode-waiver-reason is required with "
            "--allow-partial-decode"
        )
    if normalized and not allow_partial_decode:
        raise SystemExit(
            "--partial-decode-waiver-reason requires --allow-partial-decode"
        )
    return normalized


def _dispatch_blockers(partial_decode_waived: bool) -> list[str]:
    blockers = list(DISPATCH_BLOCKERS)
    if partial_decode_waived:
        blockers.append(PARTIAL_DECODE_WAIVER_BLOCKER)
    return blockers


def _evaluate(
    op_cls,
    params: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    eval_idx: int,
    *,
    require_full_decode: bool = True,
    partial_decode_waiver_reason: str | None = None,
) -> Evaluation:
    """Instantiate CodecOp with params, encode/decode, return metrics."""
    timestamp = _utc_now()
    partial_decode_waived = not require_full_decode
    if partial_decode_waived and not partial_decode_waiver_reason:
        return Evaluation(
            eval_idx=eval_idx,
            params=dict(params),
            bytes_out=-1,
            reconstruction_rms=None,
            fitness=None,
            timestamp_utc=timestamp,
            decode_coverage_required=require_full_decode,
            partial_decode_waived=True,
            decode_coverage_status="waiver_invalid",
            error="ValueError: partial decode waiver requires a reason",
        )
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
        if not isinstance(decoded, dict):
            raise ValueError(
                f"CodecOp decode returned {type(decoded).__name__}, expected dict"
            )
        expected_keys = [
            k for k, original in state_dict.items()
            if isinstance(original, torch.Tensor)
        ]
        rms_sum = 0.0
        rms_count = 0
        missing_keys: list[str] = []
        non_tensor_keys: list[str] = []
        shape_mismatch_keys: list[str] = []
        matched_keys: list[str] = []
        decoded_tensor_keys = [
            k for k, value in decoded.items()
            if isinstance(k, str) and isinstance(value, torch.Tensor)
        ]
        for k, original in state_dict.items():
            if not isinstance(original, torch.Tensor):
                continue
            if k not in decoded:
                missing_keys.append(k)
                continue
            recon = decoded[k]
            if not isinstance(recon, torch.Tensor):
                non_tensor_keys.append(k)
                continue
            if recon.shape != original.shape:
                shape_mismatch_keys.append(k)
                continue
            diff = (recon.float() - original.float()).flatten()
            rms_sum += float((diff * diff).mean().item())
            rms_count += 1
            matched_keys.append(k)
        coverage_failed = (
            rms_count == 0
            or missing_keys
            or non_tensor_keys
            or shape_mismatch_keys
            or rms_count != len(expected_keys)
        )
        if coverage_failed and require_full_decode:
            return Evaluation(
                eval_idx=eval_idx,
                params=dict(params),
                bytes_out=bytes_out,
                reconstruction_rms=None,
                fitness=None,
                timestamp_utc=timestamp,
                expected_tensor_count=len(expected_keys),
                matched_tensor_count=rms_count,
                missing_tensor_keys=missing_keys,
                non_tensor_decoded_keys=non_tensor_keys,
                shape_mismatch_tensor_keys=shape_mismatch_keys,
                decoded_tensor_keys=decoded_tensor_keys,
                matched_tensor_keys=matched_keys,
                decode_coverage_required=require_full_decode,
                partial_decode_waived=partial_decode_waived,
                partial_decode_waiver_reason=partial_decode_waiver_reason,
                decode_coverage_status="failed",
                error=(
                    "ValueError: CodecOp decode did not reconstruct every "
                    "input tensor key"
                ),
            )
        if coverage_failed and rms_count == 0:
            return Evaluation(
                eval_idx=eval_idx,
                params=dict(params),
                bytes_out=bytes_out,
                reconstruction_rms=None,
                fitness=None,
                timestamp_utc=timestamp,
                expected_tensor_count=len(expected_keys),
                matched_tensor_count=rms_count,
                missing_tensor_keys=missing_keys,
                non_tensor_decoded_keys=non_tensor_keys,
                shape_mismatch_tensor_keys=shape_mismatch_keys,
                decoded_tensor_keys=decoded_tensor_keys,
                matched_tensor_keys=matched_keys,
                decode_coverage_required=require_full_decode,
                partial_decode_waived=partial_decode_waived,
                partial_decode_waiver_reason=partial_decode_waiver_reason,
                decode_coverage_status="failed",
                error=(
                    "ValueError: CodecOp partial decode waiver matched zero "
                    "input tensor keys"
                ),
            )
        rms = math.sqrt(rms_sum / rms_count)
        fitness = float(bytes_out) + 1e6 * rms
        return Evaluation(
            eval_idx=eval_idx,
            params=dict(params),
            bytes_out=bytes_out,
            reconstruction_rms=rms,
            fitness=fitness,
            timestamp_utc=timestamp,
            expected_tensor_count=len(expected_keys),
            matched_tensor_count=rms_count,
            decoded_tensor_keys=decoded_tensor_keys,
            matched_tensor_keys=matched_keys,
            decode_coverage_required=require_full_decode,
            partial_decode_waived=partial_decode_waived,
            partial_decode_waiver_reason=partial_decode_waiver_reason,
            decode_coverage_status=(
                "partial_waived" if coverage_failed else "full"
            ),
        )
    except Exception as exc:
        return Evaluation(
            eval_idx=eval_idx,
            params=dict(params),
            bytes_out=-1,
            reconstruction_rms=None,
            fitness=None,
            timestamp_utc=timestamp,
            decode_coverage_required=require_full_decode,
            partial_decode_waived=partial_decode_waived,
            partial_decode_waiver_reason=partial_decode_waiver_reason,
            decode_coverage_status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def optuna_tpe_search(
    op_cls,
    state_dict: dict[str, torch.Tensor],
    specs: list[ParamSpec],
    max_evals: int,
    seed: int = 0,
    *,
    require_full_decode: bool = True,
    partial_decode_waiver_reason: str | None = None,
) -> list[Evaluation]:
    """TPE-based Bayesian optimization over CodecOp param space."""
    if max_evals <= 0:
        return []
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna is required for tools/codec_op_optuna_search.py; "
            "install the project lock or run via uv"
        ) from exc
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
            elif spec.type == "float":
                v = trial.suggest_float(
                    spec.name, spec.low, spec.high, log=spec.log
                )
            else:
                raise ValueError(f"unsupported parameter type {spec.type!r}")
            params[spec.name] = v
        ev = _evaluate(
            op_cls,
            params,
            state_dict,
            idx,
            require_full_decode=require_full_decode,
            partial_decode_waiver_reason=partial_decode_waiver_reason,
        )
        evaluations.append(ev)
        return (
            float(ev.fitness)
            if ev.fitness is not None and math.isfinite(ev.fitness)
            else FAILED_FITNESS_PENALTY
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(_init_params(specs))
    study.optimize(objective, n_trials=max_evals)
    return evaluations


def _valid_for_pareto(ev: Evaluation) -> bool:
    full_decode_ok = (
        ev.expected_tensor_count > 0
        and ev.matched_tensor_count == ev.expected_tensor_count
        and not ev.missing_tensor_keys
        and not ev.non_tensor_decoded_keys
        and not ev.shape_mismatch_tensor_keys
    )
    partial_decode_ok = (
        not ev.decode_coverage_required
        and ev.partial_decode_waived
        and bool(ev.partial_decode_waiver_reason)
        and ev.matched_tensor_count > 0
    )
    return (
        ev.error is None
        and ev.bytes_out >= 0
        and ev.reconstruction_rms is not None
        and math.isfinite(ev.reconstruction_rms)
        and ev.fitness is not None
        and math.isfinite(ev.fitness)
        and (full_decode_ok or partial_decode_ok)
    )


def annotate_pareto_frontier(evaluations: list[Evaluation]) -> None:
    """Mark CPU-planning non-dominated trials over bytes and RMS."""
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


def _optuna_version() -> str | None:
    try:
        return version("optuna")
    except PackageNotFoundError:
        return None


def _baseline_status(evaluations: list[Evaluation]) -> str:
    baseline = next((ev for ev in evaluations if ev.eval_idx == 0), None)
    if baseline is None:
        return "not_evaluated"
    return "failed" if baseline.error else "evaluated"


def _build_search_report(
    evaluations: list[Evaluation],
    *,
    op_module: str,
    op_class: str,
    seed: int,
    specs: list[ParamSpec],
    requested_max_evals: int,
    state_dict_path: Path | None = None,
    state_dict_key: str | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
    decode_coverage_required: bool = True,
    partial_decode_waiver_reason: str | None = None,
) -> SearchReport:
    annotate_pareto_frontier(evaluations)
    successful = [e for e in evaluations if _valid_for_pareto(e)]
    failed = [e for e in evaluations if not _valid_for_pareto(e)]
    best = min(successful, key=_evaluation_sort_key, default=None)
    state_dict_bytes = None
    state_dict_sha256 = None
    state_dict_path_payload = None
    if state_dict_path is not None:
        state_dict_path_payload = str(state_dict_path)
        state_dict_bytes = state_dict_path.stat().st_size
        state_dict_sha256 = _sha256_file(state_dict_path)
    partial_decode_waived = not decode_coverage_required
    return SearchReport(
        schema=REPORT_SCHEMA,
        tool=TOOL_NAME,
        op_module=op_module,
        op_class=op_class,
        n_evaluations=len(evaluations),
        n_successful=len(successful),
        n_failed=len(failed),
        pareto_frontier_count=len([e for e in evaluations if e.pareto_frontier]),
        best_eval=best,
        all_evaluations=evaluations,
        optimizer="optuna_tpe",
        seed=seed,
        generated_at_utc=_utc_now(),
        optuna_version=_optuna_version(),
        dispatch_blockers=_dispatch_blockers(partial_decode_waived),
        parameter_space=[asdict(s) for s in specs],
        requested_max_evals=requested_max_evals,
        baseline_eval_idx=0 if evaluations else None,
        baseline_params=_init_params(specs) if evaluations else {},
        baseline_status=_baseline_status(evaluations),
        state_dict_path=state_dict_path_payload,
        state_dict_sha256=state_dict_sha256,
        state_dict_bytes=state_dict_bytes,
        state_dict_key=state_dict_key,
        tensor_contract=_tensor_contract(state_dict) if state_dict is not None else [],
        decode_coverage_required=decode_coverage_required,
        partial_decode_waived=partial_decode_waived,
        partial_decode_waiver_reason=partial_decode_waiver_reason,
    )


def append_atom_ledger_rows(
    ledger_path: Path,
    evaluations: list[Evaluation],
    *,
    op_module: str,
    op_class: str,
    substrate_label: str,
) -> None:
    """Append one planning-only row per Optuna evaluation."""
    annotate_pareto_frontier(evaluations)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fp:
        for ev in evaluations:
            blockers = _dispatch_blockers(ev.partial_decode_waived)
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
                "expected_tensor_count": ev.expected_tensor_count,
                "matched_tensor_count": ev.matched_tensor_count,
                "missing_tensor_keys": list(ev.missing_tensor_keys),
                "non_tensor_decoded_keys": list(ev.non_tensor_decoded_keys),
                "shape_mismatch_tensor_keys": list(ev.shape_mismatch_tensor_keys),
                "decoded_tensor_keys": list(ev.decoded_tensor_keys),
                "matched_tensor_keys": list(ev.matched_tensor_keys),
                "decode_coverage_required": ev.decode_coverage_required,
                "decode_coverage_status": ev.decode_coverage_status,
                "partial_decode_waived": ev.partial_decode_waived,
                "partial_decode_waiver_reason": ev.partial_decode_waiver_reason,
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
                "promotion_eligible": False,
                "dispatch_attempted": False,
                "score_claim": False,
                "score_affecting_payload_changed": False,
                "charged_bits_changed": False,
                "exact_cuda_auth_eval": False,
                "archive_sha256": None,
                "archive_bytes": None,
                "dispatch_blockers": blockers,
                "baseline_role": (
                    "init_baseline" if ev.eval_idx == 0 else "optimizer_trial"
                ),
                "planning_objectives": {
                    "bytes_out": ev.bytes_out,
                    "reconstruction_rms": ev.reconstruction_rms,
                    "fitness": ev.fitness,
                    "matched_tensor_count": ev.matched_tensor_count,
                    "expected_tensor_count": ev.expected_tensor_count,
                    "decode_coverage_status": ev.decode_coverage_status,
                },
                "notes": (
                    "Optuna TPE CodecOp parameter-search trial; CPU-only "
                    "forensic planning row, no archive substitution or score claim."
                ),
            }
            fp.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


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
    parser.add_argument("--emit-evidence", type=Path, default=None,
                        help="If set, append a TechniqueEvidence row for the best eval "
                             "to this JSONL file (cathedral_autopilot feedback-loop hook)")
    parser.add_argument("--evidence-technique-name", default=None,
                        help="Override the technique name in the emitted evidence row "
                             "(default: '<module>.<class>')")
    parser.add_argument("--atom-ledger-output", type=Path, default=None,
                        help="Optional JSONL path for atom-ledger rows (one per eval)")
    parser.add_argument(
        "--substrate-label", default="optuna_search",
        help="Prefix for atom-ledger substrate_label.",
    )
    parser.add_argument(
        "--allow-partial-decode",
        action="store_true",
        help=(
            "Permit CodecOp decode output to cover only a subset of input "
            "tensor keys. Requires --partial-decode-waiver-reason and keeps "
            "the report planning-only/non-dispatchable."
        ),
    )
    parser.add_argument(
        "--partial-decode-waiver-reason",
        default=None,
        help="Required custody note when --allow-partial-decode is used.",
    )
    args = parser.parse_args(argv)
    if args.max_evals < 1:
        raise SystemExit("--max-evals must be >= 1")
    partial_decode_waiver_reason = _validate_partial_decode_waiver(
        args.allow_partial_decode,
        args.partial_decode_waiver_reason,
    )
    require_full_decode = not args.allow_partial_decode

    op_cls = _import_codec_op(args.module, args.class_name)
    state_dict = _load_state_dict(args.state_dict_path, args.state_dict_key)
    specs = _parse_param_spec(args.param_spec)

    evaluations = optuna_tpe_search(
        op_cls,
        state_dict,
        specs,
        max_evals=args.max_evals,
        seed=args.seed,
        require_full_decode=require_full_decode,
        partial_decode_waiver_reason=partial_decode_waiver_reason,
    )
    report = _build_search_report(
        evaluations, op_module=args.module, op_class=args.class_name,
        seed=args.seed, specs=specs,
        requested_max_evals=args.max_evals,
        state_dict_path=args.state_dict_path,
        state_dict_key=args.state_dict_key,
        state_dict=state_dict,
        decode_coverage_required=require_full_decode,
        partial_decode_waiver_reason=partial_decode_waiver_reason,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(report), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
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
        append_atom_ledger_rows(
            args.atom_ledger_output,
            evaluations,
            op_module=args.module,
            op_class=args.class_name,
            substrate_label=args.substrate_label,
        )
        print(f"appended {len(evaluations)} rows to {args.atom_ledger_output}")

    if args.emit_evidence and report.best_eval is not None:
        # Append a TechniqueEvidence row to cathedral_autopilot's feed.
        import datetime as _dt_emit
        ev_path = args.emit_evidence
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        with ev_path.open("a", encoding="utf-8") as f:
            evidence_row = {
                "technique": args.evidence_technique_name or f"{args.module}.{args.class_name}",
                "empirical_archive_bytes": int(report.best_eval.bytes_out),
                "source": (
                    f"[CPU-prep optuna best of {report.n_evaluations} evals] {args.output} "
                    f"params={report.best_eval.params}"
                ),
                "timestamp": _dt_emit.datetime.now(_dt_emit.UTC).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
            f.write(json.dumps(evidence_row) + "\n")
        print(f"emitted evidence row to {ev_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
