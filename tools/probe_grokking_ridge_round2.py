#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable $0 grokking autopsy of the committed Round-2 ridge head.

The probe reuses all 480 exact-label training sufficient statistics.  It calls
the frozen CPU SegNet only for the 120 held-out states, persisting compact
per-state dot/norm statistics so interrupted work resumes without repeating a
completed teacher call.  No live run, evaluator, GPU, or provider is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.scorer_surrogate.frozen_replay_convex_head import (  # noqa: E402
    StateSufficientStatistics,
    aggregate_sufficient_statistics,
    array_sha256,
    deterministic_replay_assignments,
    fit_cached_convex_head,
    frozen_feature_matrix,
)
from tac.witness_dsl.frozen_replay_convex_head_policy import (  # noqa: E402
    FrozenReplayConvexHeadPolicy,
)
from tools import probe_frozen_replay_convex_head as base  # noqa: E402

SCHEMA = "grokking_ridge_round2_refit.v1"
LANE_ID = "lane_grokking_ridge_bounds_20260713"
AXIS = "[macOS-CPU advisory; numpy-fp32 training-gradient evidence; no score authority]"
SOURCE_RUN = REPO / "experiments/results/frozen_replay_convex_head_95kill_n600_20260713"
SOURCE_RECEIPT = SOURCE_RUN / "measurement_receipt.json"
SOURCE_WEIGHTS = SOURCE_RUN / "fit/convex_head_weights.npz"
PAPER_URL = "https://arxiv.org/abs/2601.19791v3"
RIDGE_RATIOS = (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)


class ProbeError(RuntimeError):
    """The local measurement contract failed closed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_training_statistics(
    assignments: tuple[Any, ...],
) -> Any:
    records: list[StateSufficientStatistics] = []
    for assignment in assignments:
        if assignment.split != "train":
            continue
        path = SOURCE_RUN / "train_cache" / f"pair_{assignment.pair_index:04d}.npz"
        if not path.is_file():
            raise ProbeError(f"missing committed train statistic: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["pair_index"]) != assignment.pair_index:
                raise ProbeError(f"train assignment drift: {path}")
            records.append(
                StateSufficientStatistics(
                    gram=np.ascontiguousarray(payload["gram"], dtype=np.float32),
                    rhs=np.ascontiguousarray(payload["rhs"], dtype=np.float32),
                    target_square_sum=float(payload["target_square_sum"]),
                    row_count=int(payload["row_count"]),
                    feature_sha256=str(payload["feature_sha256"]),
                    target_sha256=str(payload["target_sha256"]),
                )
            )
    if len(records) != 480:
        raise ProbeError(f"expected 480 committed train records, observed {len(records)}")
    return aggregate_sufficient_statistics(records)


def _ridge_weights(
    gram_mean: np.ndarray,
    rhs_mean: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    hessian = gram_mean + ridge_lambda * np.eye(gram_mean.shape[0], dtype=np.float64)
    if ridge_lambda == 0.0:
        # The registered chart has one exact redundancy: bias equals the sum
        # of the three checkpoint indicators.  The realized authority matrix
        # is fp32, so a much smaller fp64 cutoff would invert accumulation
        # noise and manufacture an enormous null-mode coefficient.  Use the
        # same fp32-scale rank floor reported by the autopsy diagnostics.
        inverse = np.linalg.pinv(
            hessian,
            rcond=128.0 * np.finfo(np.float32).eps,
            hermitian=True,
        )
        weights = inverse @ rhs_mean
        if not np.isfinite(weights).all():
            raise ProbeError("zero-ridge fp32-scale pseudoinverse is non-finite")
        return np.ascontiguousarray(weights)
    return np.ascontiguousarray(np.linalg.solve(hessian, rhs_mean))


def _candidate_weights(stats: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_15 = fit_cached_convex_head(stats, epochs=15)
    original_150 = fit_cached_convex_head(stats, epochs=150)
    with np.load(SOURCE_WEIGHTS, allow_pickle=False) as payload:
        committed = np.ascontiguousarray(payload["weights"], dtype=np.float32)
    gram_mean = np.ascontiguousarray(
        stats.gram / np.float32(stats.row_count), dtype=np.float32
    ).astype(np.float64)
    rhs_mean = np.ascontiguousarray(
        stats.rhs / np.float32(stats.row_count), dtype=np.float32
    ).astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(gram_mean)
    data_lmax = float(eigenvalues[-1])
    rank_threshold = float(
        max(gram_mean.shape) * np.finfo(np.float32).eps * max(1.0, data_lmax)
    )
    numerical_rank = int(np.count_nonzero(eigenvalues > rank_threshold))
    candidates: list[dict[str, Any]] = [
        {
            "name": "paper_direction_spectral_ridge_gd15_zero_init",
            "kind": "deterministic_full_batch_gd",
            "ridge_ratio_to_data_lmax": 1.0,
            "ridge_lambda": original_15.certificate.ridge_lambda,
            "learning_rate": original_15.certificate.step_size_eta,
            "steps": 15,
            "init_variance_nu2": 0.0,
            "weights": original_15.weights,
        },
        {
            "name": "paper_direction_spectral_ridge_gd150_zero_init",
            "kind": "deterministic_full_batch_gd_long_control",
            "ridge_ratio_to_data_lmax": 1.0,
            "ridge_lambda": original_150.certificate.ridge_lambda,
            "learning_rate": original_150.certificate.step_size_eta,
            "steps": 150,
            "init_variance_nu2": 0.0,
            "weights": original_150.weights,
        },
    ]
    for ratio in RIDGE_RATIOS:
        ridge = ratio * data_lmax
        candidates.append(
            {
                "name": f"closed_form_ridge_ratio_{ratio:.0e}".replace("+", ""),
                "kind": "closed_form_exact_optimum_control",
                "ridge_ratio_to_data_lmax": ratio,
                "ridge_lambda": ridge,
                "learning_rate": None,
                "steps": 0,
                "init_variance_nu2": None,
                "weights": _ridge_weights(gram_mean, rhs_mean, ridge_lambda=ridge),
            }
        )
    for candidate in candidates:
        candidate["weights_array_sha256"] = array_sha256(candidate["weights"])
    spectral = original_15.certificate
    one_minus_eta_lambda = 1.0 - spectral.step_size_eta * spectral.ridge_lambda
    diagnostics = {
        "feature_dimension_m": int(stats.gram.shape[0]),
        "scalar_training_rows_n": int(stats.row_count),
        "output_coordinates": 3,
        "state_count": int(stats.state_count),
        "data_eigenvalue_min": float(eigenvalues[0]),
        "data_eigenvalue_max": data_lmax,
        "numerical_rank_threshold": rank_threshold,
        "numerical_rank": numerical_rank,
        "empirical_null_dimension": int(stats.gram.shape[0] - numerical_rank),
        "paper_overparameterization_m_minus_n": int(stats.gram.shape[0] - stats.row_count),
        "actual_initialization_variance_nu2": 0.0,
        "spectral_ridge_lambda": spectral.ridge_lambda,
        "spectral_learning_rate_eta": spectral.step_size_eta,
        "eta_times_lambda": spectral.step_size_eta * spectral.ridge_lambda,
        "one_minus_eta_lambda": one_minus_eta_lambda,
        "null_mode_half_life_steps": math.log(0.5) / math.log(abs(one_minus_eta_lambda)),
        "steps_to_shrink_nonzero_null_mode_by_1e7": math.ceil(
            math.log(1e-7) / math.log(abs(one_minus_eta_lambda))
        ),
        "contraction_gamma": spectral.contraction_gamma,
        "gd15_parameter_residual": original_15.actual_parameter_residual,
        "gd15_residual_parameter_bound": original_15.residual_parameter_bound,
        "gd15_terminal_gradient_norm": original_15.terminal_gradient_norm,
        "gd15_objective_gap": original_15.actual_objective_gap,
        "gd15_equals_committed_weights_bitwise": bool(np.array_equal(original_15.weights, committed)),
        "gd150_equals_gd15_weights_bitwise": bool(
            np.array_equal(original_150.weights, original_15.weights)
        ),
        "gd150_max_abs_weight_delta_from_gd15": float(
            np.max(np.abs(original_150.weights.astype(np.float64) - original_15.weights))
        ),
        "committed_weights_array_sha256": array_sha256(committed),
    }
    return candidates, diagnostics


def _heldout_path(output_dir: Path, pair_index: int) -> Path:
    return output_dir / "heldout_sufficient_statistics" / f"pair_{pair_index:04d}.json"


def _measure_heldout(
    *,
    output_dir: Path,
    assignments: tuple[Any, ...],
    candidates: list[dict[str, Any]],
    labels: np.memmap,
    margins: np.memmap,
    segnet: Any,
    yopo: Any,
) -> tuple[list[dict[str, Any]], int]:
    import torch

    candidate_names = [row["name"] for row in candidates]
    weight_stack = np.ascontiguousarray(
        np.concatenate([row["weights"] for row in candidates], axis=1), dtype=np.float64
    )
    ledger = output_dir / "teacher_calls.jsonl"
    new_teacher_calls = 0
    for checkpoint_index, (checkpoint_name, checkpoint_path, _epoch) in enumerate(base.CHECKPOINTS):
        cohort = [
            row
            for row in assignments
            if row.split == "heldout" and row.checkpoint_index == checkpoint_index
        ]
        pending = [row for row in cohort if not _heldout_path(output_dir, row.pair_index).is_file()]
        if not pending:
            continue
        renderer, code, model, _dash = yopo._load_renderer(checkpoint_path)
        if model["n_pairs"] != 600 or code.shape[0] != 1200:
            raise ProbeError(f"checkpoint geometry drift: {checkpoint_name}")
        for assignment in pending:
            pair_index = assignment.pair_index
            label = np.array(labels[pair_index], dtype=np.int64, copy=True)
            margin = np.array(margins[pair_index], dtype=np.float32, copy=True)
            frame = base._render_state_nchw(renderer, pair_index)
            start = time.perf_counter()
            _append_jsonl(
                ledger,
                {
                    "event": "exact_teacher_state_call_started",
                    "pair_index": pair_index,
                    "checkpoint_name": checkpoint_name,
                    "timestamp_utc": _utc_now(),
                },
            )
            exact_costate, teacher_metrics, teacher_seconds = yopo._capture_exact_teacher_costate(
                segnet=segnet,
                frame_nchw=frame,
                labels=torch.as_tensor(label[None], dtype=torch.long),
            )
            features = frozen_feature_matrix(
                frame.detach().cpu().numpy(),
                label,
                margin,
                checkpoint_index=checkpoint_index,
                checkpoint_count=3,
                stride=1,
            ).astype(np.float64)
            target = (
                exact_costate.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .transpose(0, 2, 3, 1)
                .reshape(-1, 3)
            )
            # Torch may leave stale IEEE flags after the exact teacher.  As in
            # the committed Round-2 module, suppress stale status flags around
            # BLAS and then refuse actual non-finite arrays explicitly.
            with np.errstate(all="ignore"):
                predictions = (features @ weight_stack).reshape(
                    features.shape[0], len(candidates), 3
                )
                reference_square = float(np.sum(target * target))
                dots = np.einsum("nkj,nj->k", predictions, target, optimize=True)
                prediction_squares = np.einsum(
                    "nkj,nkj->k", predictions, predictions, optimize=True
                )
            if not (
                np.isfinite(predictions).all()
                and math.isfinite(reference_square)
                and np.isfinite(dots).all()
                and np.isfinite(prediction_squares).all()
            ):
                raise ProbeError(f"non-finite heldout reduction at pair {pair_index}")
            if reference_square <= 0.0:
                raise ProbeError(f"zero heldout target norm at pair {pair_index}")
            metrics = {}
            for index, name in enumerate(candidate_names):
                dot = float(dots[index])
                pred_square = float(prediction_squares[index])
                cosine = None
                if pred_square > 0.0:
                    cosine = dot / math.sqrt(reference_square * pred_square)
                residual_square = reference_square + pred_square - 2.0 * dot
                metrics[name] = {
                    "dot": dot,
                    "prediction_square_norm": pred_square,
                    "cosine_similarity": cosine,
                    "relative_l2_error": math.sqrt(max(0.0, residual_square) / reference_square),
                }
            row = {
                "schema": "grokking_ridge_round2_heldout_state.v1",
                "completed_at_utc": _utc_now(),
                "assignment": assignment.to_dict(),
                "authority": AXIS,
                "frame_sha256": array_sha256(frame.detach().cpu().numpy()),
                "label_sha256": array_sha256(label),
                "margin_sha256": array_sha256(margin),
                "exact_costate_sha256": array_sha256(exact_costate.detach().cpu().numpy()),
                "reference_square_norm": reference_square,
                "compared_elements": int(target.size),
                "teacher_metrics": teacher_metrics,
                "teacher_seconds": teacher_seconds,
                "wall_seconds": time.perf_counter() - start,
                "candidate_metrics": metrics,
            }
            _atomic_json(_heldout_path(output_dir, pair_index), row)
            _append_jsonl(
                ledger,
                {
                    "event": "exact_teacher_state_call_completed",
                    "pair_index": pair_index,
                    "checkpoint_name": checkpoint_name,
                    "timestamp_utc": _utc_now(),
                    "record_sha256": _sha256(_heldout_path(output_dir, pair_index)),
                },
            )
            new_teacher_calls += 1
    rows = []
    for assignment in assignments:
        if assignment.split != "heldout":
            continue
        path = _heldout_path(output_dir, assignment.pair_index)
        if not path.is_file():
            raise ProbeError(f"missing heldout record after stage: {path}")
        row = json.loads(path.read_text())
        if set(row["candidate_metrics"]) != set(candidate_names):
            raise ProbeError(f"candidate-set drift in {path}")
        rows.append(row)
    if len(rows) != 120:
        raise ProbeError(f"expected 120 heldout records, observed {len(rows)}")
    return rows, new_teacher_calls


def _aggregate_heldout(
    rows: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    reference_square = sum(float(row["reference_square_norm"]) for row in rows)
    compared_elements = sum(int(row["compared_elements"]) for row in rows)
    for candidate in candidates:
        name = candidate["name"]
        dot = sum(float(row["candidate_metrics"][name]["dot"]) for row in rows)
        pred_square = sum(
            float(row["candidate_metrics"][name]["prediction_square_norm"])
            for row in rows
        )
        residual_square = reference_square + pred_square - 2.0 * dot
        output[name] = {
            "dot": dot,
            "reference_norm": math.sqrt(reference_square),
            "candidate_norm": math.sqrt(max(0.0, pred_square)),
            "cosine_similarity": (
                None if pred_square <= 0.0 else dot / math.sqrt(reference_square * pred_square)
            ),
            "relative_l2_error": math.sqrt(max(0.0, residual_square) / reference_square),
            "compared_elements": compared_elements,
            "state_count": len(rows),
            "reduction_dtype": "float64",
        }
    return output


def run(*, output_dir: Path, resume: bool) -> dict[str, Any]:
    import torch

    receipt_path = output_dir / "measurement_receipt.json"
    if receipt_path.is_file():
        if not resume:
            raise ProbeError(f"completed output exists; pass --resume: {output_dir}")
        return json.loads(receipt_path.read_text())
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ProbeError(f"partial output exists; pass --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_receipt_sha_before = _sha256(SOURCE_RECEIPT)
    source_tree_inputs = base._verify_input_custody()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(455)
    torch.use_deterministic_algorithms(True)
    policy = FrozenReplayConvexHeadPolicy(teacher_batch_size=1)
    assignments = deterministic_replay_assignments(
        n_pairs=600,
        checkpoint_names=tuple(row[0] for row in base.CHECKPOINTS),
        holdout_period=policy.holdout_period,
        seed=policy.seed,
    )
    stats = _load_training_statistics(assignments)
    candidates, diagnostics = _candidate_weights(stats)
    contract_path = output_dir / "run_contract.json"
    contract = {
        "schema": "grokking_ridge_round2_run_contract.v1",
        "created_at_utc": _utc_now(),
        "lane_id": LANE_ID,
        "paper": {"arxiv": "2601.19791v3", "url": PAPER_URL},
        "authority": AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "source_run_read_only": True,
        "live_run_touched": False,
        "paid_dispatch": False,
        "git_head": _git_head(),
        "input_custody": source_tree_inputs,
        "source_receipt": {
            "path": str(SOURCE_RECEIPT.relative_to(REPO)),
            "bytes": SOURCE_RECEIPT.stat().st_size,
            "sha256": source_receipt_sha_before,
        },
        "source_tool": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "candidate_names": [row["name"] for row in candidates],
        "resume_contract": "one atomic compact record per completed heldout teacher call",
    }
    if contract_path.is_file():
        prior = json.loads(contract_path.read_text())
        for key in ("lane_id", "paper", "source_receipt", "source_tool", "candidate_names"):
            if prior[key] != contract[key]:
                raise ProbeError(f"resume contract drift: {key}")
        contract = prior
    else:
        _atomic_json(contract_path, contract)
    labels = base._stored_npy_memmap(base.GT_CACHE, "lstars.npy")
    margins = base._stored_npy_memmap(base.GT_CACHE, "margins.npy")
    yopo = base._load_tool_module(
        "_grokking_round2_committed_yopo", "tools/probe_yopo_first_layer_costate.py"
    )
    segnet = base._load_cpu_segnet()
    rows, new_teacher_calls = _measure_heldout(
        output_dir=output_dir,
        assignments=assignments,
        candidates=candidates,
        labels=labels,
        margins=margins,
        segnet=segnet,
        yopo=yopo,
    )
    aggregate = _aggregate_heldout(rows, candidates)
    public_candidates = []
    for candidate in candidates:
        public_candidates.append({key: value for key, value in candidate.items() if key != "weights"})
    best_cosine_name = max(
        aggregate,
        key=lambda name: -math.inf
        if aggregate[name]["cosine_similarity"] is None
        else aggregate[name]["cosine_similarity"],
    )
    best_rel_l2_name = min(aggregate, key=lambda name: aggregate[name]["relative_l2_error"])
    source_receipt_sha_after = _sha256(SOURCE_RECEIPT)
    if source_receipt_sha_after != source_receipt_sha_before:
        raise ProbeError("source Round-2 receipt mutated during the local refit")
    receipt = {
        "schema": SCHEMA,
        "completed_at_utc": _utc_now(),
        "lane_id": LANE_ID,
        "authority": AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "paper_mapping": {
            "paper_arxiv": "2601.19791v3",
            "paper_slow_mode": "random nonzero component orthogonal to the empirical data span shrinks as (1-eta*lambda)^t",
            "round2_theorem_applicability": "REFUSED",
            "refusal_reasons": [
                "m=31 is not greater than n=1474560 scalar training rows",
                "actual initialization variance is nu^2=0, so the paper slow component starts at exactly zero",
                "the 31-feature target is not established realizable",
                "spectral ridge gives eta*lambda approximately 2/3, outside the paper small-eta-lambda experimental simplification",
            ],
        },
        "round2_diagnostics": diagnostics,
        "candidates": public_candidates,
        "heldout_fidelity": aggregate,
        "best_heldout_cosine": {
            "candidate": best_cosine_name,
            **aggregate[best_cosine_name],
        },
        "best_heldout_relative_l2": {
            "candidate": best_rel_l2_name,
            **aggregate[best_rel_l2_name],
        },
        "teacher_call_accounting": {
            "source_train_exact_labels_reused": 480,
            "new_heldout_exact_teacher_calls_this_invocation": new_teacher_calls,
            "heldout_exact_teacher_calls_total": 120,
            "repeated_completed_teacher_calls": 0,
            "synthetic_data_used": False,
        },
        "source_custody": {
            "source_run": str(SOURCE_RUN.relative_to(REPO)),
            "source_receipt_sha256_before": source_receipt_sha_before,
            "source_receipt_sha256_after": source_receipt_sha_after,
            "source_run_mutated": False,
        },
        "verdict": {
            "undertrained_fixed_objective": False,
            "feature_poverty_scope": "31-feature linear chart across the measured ridge ladder, not nonlinear or richer-feature families",
            "plateau_transfer_authority": False,
            "pointer_delta": "NONE",
        },
        "run_contract": {
            "path": str(contract_path.relative_to(output_dir)),
            "sha256": _sha256(contract_path),
        },
    }
    _atomic_json(receipt_path, receipt)
    _atomic_json(
        output_dir / "complete.json",
        {
            "schema": "grokking_ridge_round2_complete.v1",
            "completed_at_utc": receipt["completed_at_utc"],
            "receipt": {
                "path": "measurement_receipt.json",
                "bytes": receipt_path.stat().st_size,
                "sha256": _sha256(receipt_path),
            },
            "cleanup": "COMPLETE_NO_BULK; only compact JSON sufficient statistics retained",
        },
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    receipt = run(output_dir=args.output_dir.resolve(), resume=args.resume)
    print(
        json.dumps(
            {
                "receipt": str((args.output_dir / "measurement_receipt.json").resolve()),
                "best_heldout_cosine": receipt["best_heldout_cosine"],
                "best_heldout_relative_l2": receipt["best_heldout_relative_l2"],
                "diagnostics": receipt["round2_diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
