#!/usr/bin/env python3
"""ddm_et3 -- solve-within null-basis phase field realized through DK1 CVP.

Axis: [macOS-CPU frozen-scorer advisory]. score_claim=false.

This runner composes the two P0 legs that et3 was chartered to measure:
SW1's in-null c-space solver for the et1/et2 block16 phase field, followed by
DK1's lattice-native private-support CVP/Babai realizer.  It never claims a
contest score and only fires the full-n600 byte-close follow-on by writing a
fire-order verdict; MAIN adjudicates promotion.
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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "experiments", REPO / "upstream"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from ddm_et1_ph1_block16_on_our_vehicle import translate_blocks  # noqa: E402
from ddm_et2_projected_phase_field import (  # noqa: E402
    BASELINE_ARCHIVE_SHA256,
    BASELINE_BYTES,
    BASELINE_D_POSE,
    BASELINE_D_SEG,
    BASELINE_S,
    forward,
    load_models,
    raw_memmap,
    score_from_components,
)
from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_pose_null_constrained_paint import snap_band_to_blocks, yuv6_shift  # noqa: E402
from ddm_sq1_stage_decomposition_and_solved_paint import confusion, resize_to_scorer  # noqa: E402
from ddm_sw1_null_basis_phase_solve import (  # noqa: E402
    block_mask_from_band,
    coeffs_from_delta_euclidean,
    coeffs_to_delta_chw,
    chw_to_block_vectors,
    metric_weights_from_saliency,
    null_coordinate_basis,
    pose_constraint_matrix,
)
from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.margin_saliency_map import compute_margin_saliency_map  # noqa: E402
from tac.optimization.lattice_native_pose_null_realizer import (  # noqa: E402
    add_private_delta_to_frame,
    build_default_operator,
    cvp_integer_realize,
    extract_private_camera_block,
    private_block_geometry,
    project_scorer_delta_to_pose_null,
)
from tac.optimization.trajectory_stopping import build_cap_stop_receipt  # noqa: E402


AXIS = "[macOS-CPU frozen-scorer advisory]"
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
DEN = 37_545_489
RATE_PER_BYTE = 25.0 / DEN
S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as exc:  # pragma: no cover - provenance fallback only
        return f"UNKNOWN:{type(exc).__name__}:{exc}"


def sha256_file(path: Path, chunk_size: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=1, default=jsonable, allow_nan=False)
        fh.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=jsonable, allow_nan=False))
        fh.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def parse_cap_ladder(text: str | None, *, fallback: int) -> tuple[int, ...]:
    raw = text if text is not None and text.strip() else str(int(fallback))
    out: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        cap = int(item)
        if cap < 1:
            raise ValueError("cap ladder entries must be positive")
        out.append(cap)
    if not out:
        raise ValueError("cap ladder must contain at least one cap")
    return tuple(out)


def _curve_last_step(curve: Any) -> int | None:
    if not isinstance(curve, list):
        return None
    steps = [int(row["step"]) for row in curve if isinstance(row, dict) and "step" in row]
    return max(steps) if steps else None


def cap_receipt_from_diagnostics(diagnostics: dict[str, Any], *, cap: int) -> dict[str, Any]:
    selected = diagnostics.get("selected", {})
    start = selected.get("start") if isinstance(selected, dict) else None
    start_diag: dict[str, Any] = {}
    for row in diagnostics.get("starts", []):
        if isinstance(row, dict) and row.get("start") == start:
            start_diag = row
            break
    stop_reason = str(
        selected.get("stop_reason")
        or start_diag.get("stop_reason")
        or "UNKNOWN_STOP_REASON"
    )
    steps_run = selected.get("steps_run") or start_diag.get("steps_run")
    if steps_run is None:
        steps_run = _curve_last_step(selected.get("curve")) or _curve_last_step(start_diag.get("curve"))
    steps = int(steps_run) if steps_run is not None else 0
    if stop_reason.startswith("iteration_cap"):
        still_descending = stop_reason == "iteration_cap_best_at_cap"
        receipt = build_cap_stop_receipt(
            stop_reason="cap_bound",
            steps_run=max(steps, int(cap)),
            cap=int(cap),
            still_descending=still_descending,
        )
        return receipt.to_payload()
    if stop_reason == "plateau_no_proxy_improvement":
        receipt = build_cap_stop_receipt(
            stop_reason="converged",
            steps_run=steps,
            cap=int(cap),
            still_descending=False,
        )
        return receipt.to_payload()
    receipt = build_cap_stop_receipt(
        stop_reason="failed",
        steps_run=steps,
        cap=int(cap),
        still_descending=None,
    )
    return receipt.to_payload()


def solve_within_null_basis_delta(
    segnet: Any,
    dec_f1: np.ndarray,
    gt_f1: np.ndarray,
    target_labels: np.ndarray,
    block_mask: np.ndarray,
    objective_weights: np.ndarray,
    basis_t: torch.Tensor,
    constraint_np: np.ndarray,
    *,
    steps: int,
    lr: float,
    eval_every: int,
    convergence_patience_evals: int,
    convergence_min_improvement: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = resize_to_scorer(dec_f1)
    truth = resize_to_scorer(gt_f1)
    target = torch.from_numpy(target_labels.astype(np.int64))[None]
    weights = torch.from_numpy(objective_weights.astype(np.float32))[None]
    block_mask_t = torch.from_numpy(block_mask.astype(np.float32))[None, :, :, None]
    truth_delta = truth - base
    truth_init = coeffs_from_delta_euclidean(truth_delta, basis_t) * block_mask_t
    starts = (("dec", torch.zeros_like(truth_init)), ("truth_null", truth_init.detach()))
    best: tuple[int, int, str, torch.Tensor, float, int] | None = None
    diagnostics: dict[str, Any] = {
        "starts": [],
        "parameterization": "per-2x2-block c in R6, delta=N@c, no projection step",
        "loss": "diagonal margin-saliency weighted cross_entropy against translated phase target",
        "target_delta_for_realizer": "unclipped float N@c delta; DK1 handles uint8 bounds",
    }
    with torch.enable_grad():
        for start_name, init in starts:
            coeffs = init.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([coeffs], lr=lr)
            start_diag: dict[str, Any] = {"start": start_name, "curve": [], "stop_reason": "iteration_cap"}
            start_best: tuple[int, int] | None = None
            evals_since_best = 0
            for step in range(steps + 1):
                live_coeffs = coeffs * block_mask_t
                delta = coeffs_to_delta_chw(live_coeffs, basis_t)
                preclip = base + delta
                cur = torch.clamp(preclip, 0.0, 255.0)
                if step % eval_every == 0 or step == steps:
                    q = torch.round(cur).detach()
                    with torch.no_grad():
                        lam = segnet(q).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                    bad = int((lam != target_labels).sum())
                    blocks = chw_to_block_vectors(delta.detach())[0].cpu().numpy()
                    a_delta = np.einsum("ad,hwd->hwa", constraint_np, blocks)
                    max_a = float(np.abs(a_delta).max())
                    clipped = int(((preclip < 0.0) | (preclip > 255.0)).sum().item())
                    start_diag["curve"].append(
                        {
                            "step": int(step),
                            "proxy_phase_target_flips": bad,
                            "max_abs_A_delta_float": max_a,
                            "clipped_channel_values": clipped,
                        }
                    )
                    improved = (
                        start_best is None
                        or int(start_best[0]) - bad >= max(1, int(convergence_min_improvement))
                    )
                    if improved:
                        start_best = (bad, int(step))
                        evals_since_best = 0
                    else:
                        evals_since_best += 1
                    if best is None or bad < best[0]:
                        best = (
                            bad,
                            int(step),
                            start_name,
                            live_coeffs.detach().clone(),
                            max_a,
                            clipped,
                        )
                    if (
                        convergence_patience_evals > 0
                        and evals_since_best >= convergence_patience_evals
                        and step < steps
                    ):
                        start_diag["stop_reason"] = "plateau_no_proxy_improvement"
                        break
                if step == steps:
                    start_diag["stop_reason"] = (
                        "iteration_cap_best_at_cap"
                        if start_best is not None and start_best[1] == step
                        else "iteration_cap_before_plateau"
                    )
                    break
                logits = segnet(cur)
                per_pixel = torch.nn.functional.cross_entropy(logits, target, reduction="none")
                loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1e-9)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            start_diag["best_proxy_phase_target_flips"] = start_best[0] if start_best else None
            start_diag["best_step"] = start_best[1] if start_best else None
            diagnostics["starts"].append(start_diag)
    if best is None:
        raise RuntimeError("null-basis solve produced no iterate")
    best_bad, best_step, best_start, best_coeffs, best_max_a, best_clipped = best
    best_delta = coeffs_to_delta_chw(best_coeffs, basis_t)[0].permute(1, 2, 0).detach().cpu().numpy()
    diagnostics["selected"] = {
        "start": best_start,
        "best_step": best_step,
        "best_proxy_phase_target_flips": best_bad,
        "max_abs_A_delta_float": best_max_a,
        "clipped_channel_values_before_realizer": best_clipped,
    }
    for row in diagnostics["starts"]:
        if row.get("start") == best_start:
            diagnostics["selected"]["stop_reason"] = row.get("stop_reason")
            break
    return np.asarray(best_delta, dtype=np.float64), diagnostics


def realize_cvp_delta(
    *,
    camera_frame: np.ndarray,
    target_delta_hwc: np.ndarray,
    block_mask: np.ndarray,
    metric_weights_hw: np.ndarray,
    cvp_tap_radius: int,
    cvp_max_channel_candidates: int,
    cvp_max_pixel_candidates: int,
    cvp_max_combinations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = np.asarray(camera_frame)
    if frame.dtype != np.uint8 or frame.shape != (CAM_H, CAM_W, 3):
        raise RuntimeError(f"bad camera frame for CVP: dtype={frame.dtype} shape={frame.shape}")
    delta = np.asarray(target_delta_hwc, dtype=np.float64)
    if delta.shape != (SEG_H, SEG_W, 3):
        raise RuntimeError(f"bad target delta shape: {delta.shape}")
    blocks = np.asarray(block_mask).astype(bool)
    if blocks.shape != (SEG_H // 2, SEG_W // 2):
        raise RuntimeError(f"bad block mask shape: {blocks.shape}")

    coords = [(int(by), int(bx)) for by, bx in zip(*np.nonzero(blocks), strict=False)]
    operator = build_default_operator()
    out = frame.copy()
    sampled: list[dict[str, Any]] = []
    candidate_scope_counts: Counter[str] = Counter()
    aggregate = {
        "blocks_total_requested_by_mask": len(coords),
        "blocks_realized": 0,
        "pose_leakage_sq_sum": 0.0,
        "seg_discrepancy_sum": 0.0,
        "changed_camera_values_sum": 0,
        "exact_declared_scope_count": 0,
        "global_integer_optimum_claim": False,
    }

    for by, bx in coords:
        sy = by * 2
        sx = bx * 2
        block_delta = project_scorer_delta_to_pose_null(delta[sy : sy + 2, sx : sx + 2])
        geom = private_block_geometry(operator, sy, sx)
        base_block = extract_private_camera_block(out, geom)
        weights = metric_weights_hw[sy : sy + 2, sx : sx + 2].reshape(4)
        s_metric = np.repeat(weights, 3).astype(np.float64)
        result = cvp_integer_realize(
            block_delta,
            geom,
            base_block=base_block,
            tap_radius=int(cvp_tap_radius),
            max_channel_candidates=int(cvp_max_channel_candidates),
            max_pixel_candidates=int(cvp_max_pixel_candidates),
            max_combinations=int(cvp_max_combinations),
            s_metric=s_metric,
        )
        out = add_private_delta_to_frame(out, geom, result.camera_delta)
        payload = result.to_dict()
        diag = payload.get("diagnostics", {})
        if isinstance(diag, dict):
            candidate_scope_counts[str(diag.get("candidate_scope", "UNKNOWN"))] += 1
            if bool(diag.get("exact_declared_scope")):
                aggregate["exact_declared_scope_count"] += 1
        aggregate["blocks_realized"] += 1
        aggregate["pose_leakage_sq_sum"] += float(result.pose_leakage_sq)
        aggregate["seg_discrepancy_sum"] += float(result.seg_discrepancy)
        aggregate["changed_camera_values_sum"] += int(result.changed_camera_values)
        if len(sampled) < 16:
            sampled.append(
                {
                    "scorer_row": sy,
                    "scorer_col": sx,
                    "geometry": {
                        "denominator": int(geom.denominator),
                        "assumes_uniform_025": False,
                    },
                    "result": payload,
                }
            )

    aggregate["candidate_scope_counts"] = dict(sorted(candidate_scope_counts.items()))
    return out, {
        "schema": "ddm_et3_dk1_cvp_realizer_receipt.v1",
        "method": "dk1_cvp_babai_private_support",
        "cvp_tap_radius": int(cvp_tap_radius),
        "cvp_max_channel_candidates": int(cvp_max_channel_candidates),
        "cvp_max_pixel_candidates": int(cvp_max_pixel_candidates),
        "cvp_max_combinations": int(cvp_max_combinations),
        "exact_d_weights": True,
        "uniform_025_assumption": False,
        "s_metric_source": "same diagonal margin-saliency scorer-grid weights repeated over RGB channels",
        "aggregate": aggregate,
        "sampled_block_receipts": sampled,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": PROMOTION_ELIGIBLE,
    }


def score_pair(
    *,
    segnet: Any,
    posenet: Any,
    dec_f0: np.ndarray,
    cam_f1: np.ndarray,
    pose_gt: np.ndarray,
    lgt: np.ndarray,
    flips0_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pair = np.stack([dec_f0, cam_f1]).astype(np.uint8)
    cells, pose = forward(segnet, posenet, pair[None])
    lam = cells[0]
    after = lam != lgt
    d_pose_after = float(np.square(pose[0] - pose_gt).sum() / 6.0)
    return lam, pose[0], {
        "flips_after": int(after.sum()),
        "net_flip_reduction": int(flips0_map.sum()) - int(after.sum()),
        "fixed_global": int((flips0_map & ~after).sum()),
        "introduced_global": int((~flips0_map & after).sum()),
        "C_after": confusion(lgt, lam).tolist(),
        "d_pose_after": d_pose_after,
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    bar: float,
    parent_d_pose: float,
    pose_max_threshold: float,
    pose_median_abs_tol: float,
    n4_eta: float | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "ddm_et3_solve_within_cvp_aggregate.v1",
        "n_rows": len(rows),
        "breakeven_eta_bar": float(bar),
        "pose_max_threshold": float(pose_max_threshold),
        "pose_median_abs_tol": float(pose_median_abs_tol),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": PROMOTION_ELIGIBLE,
    }
    if not rows:
        return out
    before = int(sum(r["flips_before"] for r in rows))
    after = int(sum(r["cvp_realized"]["flips_after"] for r in rows))
    denom = int(sum(r["label_ceiling_net_fixed"] for r in rows))
    eta = (before - after) / denom if denom else None
    ratios = np.asarray([r["cvp_realized"]["d_pose_ratio"] for r in rows], dtype=np.float64)
    pose_sse_delta = sum(
        (r["cvp_realized"]["d_pose_after"] - r["d_pose_before"]) * 6.0 for r in rows
    )
    seg_delta_s = (after - before) * S_PER_FLIP
    pose_delta_s = math.sqrt(10.0 * (parent_d_pose + pose_sse_delta / (N_PAIRS_TOTAL * 6))) - math.sqrt(
        10.0 * parent_d_pose
    )
    eta_pass = bool(eta is not None and eta > bar)
    pose_median = float(np.median(ratios))
    # Scorer-convention aggregate: upstream/evaluate.py averages d_pose ITSELF
    # across pairs before sqrt(10*.), so the pose VERDICT leg gates on the
    # subset ratio-of-means (same tolerance constant), never on the median of
    # per-pair ratios -- which can disagree with the verdict IN SIGN under
    # skew (rt1 measured x1.809 vs x0.431 on the same pairs; law:
    # pose_aggregation_is_mean_of_dpose_never_mean_of_ratios_20260816).  The
    # max-ratio leg stays as the per-pair blow-up guard; the median stays a
    # distribution diagnostic in the output.
    pose_before_sum = float(sum(r["d_pose_before"] for r in rows))
    pose_after_sum = float(sum(r["cvp_realized"]["d_pose_after"] for r in rows))
    pose_agg_ratio = (
        pose_after_sum / pose_before_sum if pose_before_sum > 0 else float("inf")
    )
    pose_pass = bool(
        abs(pose_agg_ratio - 1.0) <= pose_median_abs_tol
        and float(ratios.max()) <= pose_max_threshold
    )
    if eta_pass and pose_pass:
        verdict = "GREEN_N32_FIRE_ORDER_2_QUEUED_FOR_MAIN_ADJUDICATION"
        fire_order_2 = "QUEUED_FULL_N600_BYTE_CLOSE"
        verdict_scope = "FORMULATION: solve-within null-basis + DK1 CVP on block16 phase field, n32 advisory gate"
    elif not eta_pass:
        verdict = "FOLDED_FAMILY_ON_THIS_PARENT_N32_ALL_THREE_FORMS_MEASURED"
        fire_order_2 = "NOT_FIRED_ETA_NOT_ABOVE_BAR"
        verdict_scope = "FAMILY-on-this-parent: projection-E, projection-M, and solve-within+CVP all measured at n32 below priced eta bar"
    else:
        verdict = "HELD_POSE_BOUND_FAIL_NO_FIRE_ORDER_2"
        fire_order_2 = "NOT_FIRED_POSE_BOUND_FAIL"
        verdict_scope = "FORMULATION: solve-within+CVP n32 eta cleared bar but pose bound failed"
    out.update(
        {
            "flips_before_subset": before,
            "cvp_flips_after_subset": after,
            "cvp_net_flip_reduction_subset": before - after,
            "label_ceiling_net_fixed_subset": denom,
            "solve_within_cvp_eta_realized": eta,
            "eta_over_bar": eta / bar if eta is not None and bar else None,
            "eta_clears_bar": eta_pass,
            "pose_ratio_min": float(ratios.min()),
            "pose_ratio_p25": float(np.quantile(ratios, 0.25)),
            "pose_ratio_median": pose_median,
            "pose_ratio_p75": float(np.quantile(ratios, 0.75)),
            "pose_ratio_max": float(ratios.max()),
            "pose_ratio_mean": float(ratios.mean()),
            "pose_agg_ratio_scorer_convention": pose_agg_ratio,
            "pose_ratio_stats_are_diagnostic_only": True,
            "pose_bound_basis": "agg_ratio_of_means_scorer_convention_plus_max_ratio_guard",
            "pose_bound_pass": pose_pass,
            "subset_cvp_seg_delta_S_no_rate": seg_delta_s,
            "subset_cvp_pose_delta_S_against_parent": pose_delta_s,
            "subset_cvp_joint_delta_S_no_rate_against_parent": seg_delta_s + pose_delta_s,
            "n4_to_n32_stability": {
                "n4_solve_within_eta": n4_eta,
                "n32_solve_within_cvp_eta": eta,
                "retention_factor_n32_over_n4": (eta / n4_eta) if eta is not None and n4_eta else None,
                "shrink_factor_n4_over_n32": (n4_eta / eta) if eta not in (None, 0.0) and n4_eta else None,
                "projection_reference_from_charter": "projected-static eta shrank about 2.3x from n4 to n32",
            },
            "fire_order_2": fire_order_2,
            "verdict": verdict,
            "verdict_scope": verdict_scope,
        }
    )
    return out


def element_grade_vector(args: argparse.Namespace, *, basis_cert: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ddm_et3_element_grade_vector.v1",
        "chain": "block16_phase_field -> solve-within null-basis -> DK1 CVP/Babai -> frozen scorer",
        "solve": {
            "grade": "OPTIMAL-RECEIPT",
            "note": "SW1 c-space form: per-2x2 c in R6, delta=N@c; no post-solve projection",
            "basis_shape": basis_cert["basis_shape"],
            "max_abs_A_times_N": basis_cert["max_abs_A_times_N"],
        },
        "metric_source_lambda": {
            "grade": "MEASURED-SUBSTITUTE",
            "note": "diagonal margin-saliency weights from tac.margin_saliency_map; full MS4D is not claimed",
            "lambda_saliency": float(args.lambda_saliency),
            "outside_weight": float(args.outside_weight),
            "saliency_clip": float(args.saliency_clip),
        },
        "realization": {
            "grade": "OPTIMAL-RECEIPT",
            "note": "DK1 private-support CVP/Babai over exact D weights; no naive scorer-lattice round path",
            "cvp_tap_radius": int(args.cvp_tap_radius),
            "cvp_max_channel_candidates": int(args.cvp_max_channel_candidates),
            "cvp_max_pixel_candidates": int(args.cvp_max_pixel_candidates),
            "global_integer_optimum_claim": False,
        },
        "stopping_rules": {
            "grade": "OPTIMAL-RECEIPT",
            "note": "typed cap-stop receipts per pair and cap",
            "cap_ladder": parse_cap_ladder(args.cap_ladder, fallback=args.steps),
        },
        "pair_set": {
            "grade": "BOUNDED-RECEIPT",
            "note": "fixed ET1/ET2 n32 same-set comparability pair list; source JSON has no random/stratified provenance",
            "selection": selection,
        },
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
    }


def build_payload(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    parent_archive_sha: str,
    parent_score: dict[str, Any],
    field_current: dict[str, Any],
    basis_cert: dict[str, Any],
    scorer_custody: dict[str, Any],
    selection: dict[str, Any],
    n4_eta: float | None,
    recall_evidence: dict[str, Any],
) -> dict[str, Any]:
    aggregate = aggregate_rows(
        rows,
        bar=float(field_current["breakeven_eta"]),
        parent_d_pose=float(parent_score["d_pose"]),
        pose_max_threshold=args.pose_max_threshold,
        pose_median_abs_tol=args.pose_median_abs_tol,
        n4_eta=n4_eta,
    )
    return {
        "schema": "ddm_et3_solve_within_cvp_phase_field_summary.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "promotion_eligible": PROMOTION_ELIGIBLE,
        "parent": {
            "archive": str(args.parent_archive),
            "archive_sha256": parent_archive_sha,
            "archive_bytes": int(args.parent_archive.stat().st_size),
            "expected_own_vehicle_frontier": {
                "S": BASELINE_S,
                "d_seg": BASELINE_D_SEG,
                "d_pose": BASELINE_D_POSE,
                "bytes": BASELINE_BYTES,
            },
            "parent_score": parent_score,
        },
        "phase_field": field_current,
        "selection": selection,
        "solver": {
            "form": "solve-within-null-basis",
            "steps": int(args.steps),
            "cap_ladder": list(parse_cap_ladder(args.cap_ladder, fallback=args.steps)),
            "lr": float(args.lr),
            "eval_every": int(args.eval_every),
            "convergence_patience_evals": int(args.convergence_patience_evals),
            "convergence_min_improvement": int(args.convergence_min_improvement),
        },
        "metric_source": {
            "producer": "tac.margin_saliency_map.compute_margin_saliency_map",
            "lambda_saliency": float(args.lambda_saliency),
            "outside_weight": float(args.outside_weight),
            "saliency_clip": float(args.saliency_clip),
            "scope": "diagonal scorer-grid margin-saliency weights; MS4D full row-Gram is not claimed",
        },
        "realizer": {
            "producer": "tac.optimization.lattice_native_pose_null_realizer.cvp_integer_realize",
            "method": "CVP/Babai private-window integer realization",
            "cvp_tap_radius": int(args.cvp_tap_radius),
            "cvp_max_channel_candidates": int(args.cvp_max_channel_candidates),
            "cvp_max_pixel_candidates": int(args.cvp_max_pixel_candidates),
            "cvp_max_combinations": int(args.cvp_max_combinations),
            "exact_d_weights": True,
            "global_integer_optimum_claim": False,
        },
        "basis": basis_cert,
        "scorer_custody": scorer_custody,
        "element_grade_vector": element_grade_vector(args, basis_cert=basis_cert, selection=selection),
        "aggregate": aggregate,
        "rows": rows,
        "recall_evidence": recall_evidence,
        "boundaries": [
            "n32 frozen-scorer advisory only",
            "no archive build",
            "no upstream/evaluate.py row",
            "no contest-CPU/CUDA or public promotion claim",
            "DK1 CVP exactness is declared finite kept-scope only, not global MIQP optimality",
        ],
    }


def execute(args: argparse.Namespace) -> int:
    if args.limit < 0:
        raise RuntimeError("--limit must be non-negative")
    parent_archive_sha = sha256_file(args.parent_archive)
    if parent_archive_sha != BASELINE_ARCHIVE_SHA256:
        raise RuntimeError(f"parent archive SHA drifted: {parent_archive_sha}")
    args.bulk_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.bulk_dir / "et3_solve_within_cvp_rows.jsonl"
    summary_path = args.bulk_dir / "et3_solve_within_cvp_summary.json"
    receipt_summary_path = args.receipt_dir / "et3_solve_within_cvp_summary.json"

    raw = raw_memmap(args.parent_raw)
    parent_lstars = np.load(args.parent_argmax, mmap_mode="r")
    current_offsets = np.load(args.current_offsets, mmap_mode="r")
    gt_labels = open_stored_npy_memmap(args.gt_cache, "lstars")
    parent_score = json.loads(args.parent_score.read_text())
    field_payload = json.loads(args.phase_field_summary.read_text())
    field_current = field_payload["current"]
    basis_np, basis_cert = null_coordinate_basis()
    basis_t = torch.from_numpy(basis_np.astype(np.float32))
    constraint_np = pose_constraint_matrix()
    segnet, posenet, scorer_custody = load_models(args.upstream_root, threads=args.threads)
    pairs_all = [int(p) for p in json.loads(args.et1_n32_json.read_text())["pairs"]]
    pairs = pairs_all[: args.limit] if args.limit else pairs_all
    selection = {
        "mode": "fixed_et1_et2_n32_pair_set",
        "source": str(args.et1_n32_json),
        "pairs": pairs,
        "n_pairs": len(pairs),
        "population": N_PAIRS_TOTAL,
        "m88_note": "same-set comparability governs this A/B/C read; source has no random/stratified provenance and is not a contiguous prefix",
    }
    sw1_summary = json.loads(args.sw1_summary.read_text()) if args.sw1_summary.exists() else {}
    n4_eta = (
        sw1_summary.get("aggregate", {})
        .get("solve_within_null_basis", {})
        .get("eta_realized")
    )
    recall_evidence = {
        "sources_read": [
            ".omx/tmp/codex_runs/et3_prompt.md",
            ".omx/tmp/codex_runs/_common_contract.md",
            "PROGRAM.md",
            "CLAUDE.md/AGENTS.md (identical by cmp)",
            "docs/operating_manual_craft_handoff.md",
            ".omx/state/main_hot_state.md",
            ".omx/research/ddm_et2_20260806/RECEIPT.md",
            ".omx/research/ddm_sw1_20260806/RECEIPT.md",
            ".omx/research/ddm_dk1_20260806/RECEIPT.md",
            ".omx/research/ddm_rw1_20260806/RECEIPT.md",
            ".omx/research/ddm_na2_negative_audit_20260803.md",
            "canonical equations registry via tools/list_canonical_equations.py --json",
        ],
        "queries": [
            "solve-within|null-basis|null basis|CVP|Babai|phase-field|block16|pose-null|lattice-native|m88",
            "ddm_et2|ddm_sw1|ddm_dk1|eta_bar|breakeven_eta",
            "pose_null_subspace_is_ac_only|ddm_ms2_scorer_metric_second_order_action|dynamic_quantum_calibration",
        ],
        "beyond_charter_findings_that_changed_plan": [
            "RW1 already smoke-tested DK1-CVP on q3x but only as n1/block-limited project-after; ET3 keeps it as precedent, not verdict.",
            "NA2/m88 requires explicit selection-mode wording; this n32 set is fixed ET1/ET2 same-set comparability, not a bankable population claim.",
            "Canonical pose-null AC-only law keeps DC paint out of the null subspace; ET3 therefore preserves SW1's per-2x2 N@c parameterization and does not add DC repair.",
            "DK1 CVP is pruned finite-scope, not global integer optimality; ET3 records CVP bounds and exact_declared_scope counts.",
        ],
    }

    existing = load_jsonl(rows_path) if args.resume else []
    done = {int(row["pair"]) for row in existing}
    todo = [pair for pair in pairs if pair not in done]
    wanted: set[int] = set()
    for pair in todo:
        wanted.update({seq_len * pair, seq_len * pair + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted) if wanted else {}
    rows = list(existing)
    current_payload = build_payload(
        args,
        rows,
        parent_archive_sha=parent_archive_sha,
        parent_score=parent_score,
        field_current=field_current,
        basis_cert=basis_cert,
        scorer_custody=scorer_custody,
        selection=selection,
        n4_eta=float(n4_eta) if n4_eta is not None else None,
        recall_evidence=recall_evidence,
    )
    write_json_atomic(summary_path, current_payload)
    write_json_atomic(receipt_summary_path, current_payload)
    print(f"[et3] ready rows={len(rows)} remaining={len(todo)} pairs={pairs}", flush=True)

    caps = parse_cap_ladder(args.cap_ladder, fallback=args.steps)
    for pair in todo:
        started = time.time()
        dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]]).astype(np.uint8)
        cells, pose_base = forward(segnet, posenet, dec[None])
        gt_cells, pose_gt_all = forward(segnet, posenet, gt[None])
        lstar = cells[0]
        lgt = gt_cells[0]
        pose_gt = pose_gt_all[0]
        cached_parent = np.asarray(parent_lstars[pair])
        cached_gt = np.asarray(gt_labels[pair], dtype=np.uint8)
        if not np.array_equal(lstar, cached_parent):
            raise RuntimeError(f"C2 failed for pair {pair}: decoded parent argmax != cached parent")
        if not np.array_equal(lgt, cached_gt):
            raise RuntimeError(f"C3 failed for pair {pair}: canonical GT decode argmax != GT cache")

        target = translate_blocks(lstar, np.asarray(current_offsets[pair]), args.block)
        band = target != lstar
        snapped = snap_band_to_blocks(band)
        block_mask = block_mask_from_band(snapped)
        flips0_map = lstar != lgt
        flips0 = int(flips0_map.sum())
        label_after = target != lgt
        label_ceiling_net_fixed = flips0 - int(label_after.sum())
        d_pose_before = float(np.square(pose_base[0] - pose_gt).sum() / 6.0)
        base = resize_to_scorer(dec[1])
        base_sc_u8 = torch.round(base)[0].permute(1, 2, 0).numpy().astype(np.uint8)

        with torch.enable_grad():
            sal = compute_margin_saliency_map(
                segnet,
                torch.from_numpy(np.ascontiguousarray(dec[1])).permute(2, 0, 1).float(),
                flip_pixel_mask=torch.from_numpy(snapped.astype(bool)),
            )
        weights, weight_stats = metric_weights_from_saliency(
            sal.saliency.cpu().numpy(),
            snapped,
            lambda_saliency=args.lambda_saliency,
            outside_weight=args.outside_weight,
            clip=args.saliency_clip,
        )

        attempts: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for cap in caps:
            delta_hwc, diag = solve_within_null_basis_delta(
                segnet,
                dec[1],
                gt[1],
                target,
                block_mask,
                weights,
                basis_t,
                constraint_np,
                steps=int(cap),
                lr=args.lr,
                eval_every=args.eval_every,
                convergence_patience_evals=args.convergence_patience_evals,
                convergence_min_improvement=args.convergence_min_improvement,
            )
            selected = diag["selected"]
            cap_receipt = cap_receipt_from_diagnostics(diag, cap=int(cap))
            attempt = {
                "cap": int(cap),
                "proxy_phase_target_flips": int(selected["best_proxy_phase_target_flips"]),
                "selected": selected,
                "cap_stop_receipt": cap_receipt,
            }
            attempts.append(attempt)
            if best is None or attempt["proxy_phase_target_flips"] < best["attempt"]["proxy_phase_target_flips"]:
                best = {"attempt": attempt, "delta_hwc": delta_hwc, "diagnostics": diag}
            if cap_receipt["stop_reason"] == "converged":
                break
        if best is None:
            raise RuntimeError("cap ladder produced no solve-within result")

        cam_cvp, cvp_receipt = realize_cvp_delta(
            camera_frame=dec[1],
            target_delta_hwc=best["delta_hwc"],
            block_mask=block_mask,
            metric_weights_hw=weights,
            cvp_tap_radius=args.cvp_tap_radius,
            cvp_max_channel_candidates=args.cvp_max_channel_candidates,
            cvp_max_pixel_candidates=args.cvp_max_pixel_candidates,
            cvp_max_combinations=args.cvp_max_combinations,
        )
        cvp_lam, _cvp_pose, scored = score_pair(
            segnet=segnet,
            posenet=posenet,
            dec_f0=dec[0],
            cam_f1=cam_cvp,
            pose_gt=pose_gt,
            lgt=lgt,
            flips0_map=flips0_map,
        )
        d_pose_after = float(scored["d_pose_after"])
        pose_sse_delta = float((d_pose_after - d_pose_before) * 6.0)
        seg_delta_s = (int(scored["flips_after"]) - flips0) * S_PER_FLIP
        pose_delta_s = math.sqrt(
            10.0 * (float(parent_score["d_pose"]) + pose_sse_delta / (N_PAIRS_TOTAL * 6))
        ) - math.sqrt(10.0 * float(parent_score["d_pose"]))
        cvp_scorer_u8 = torch.round(resize_to_scorer(cam_cvp))[0].permute(1, 2, 0).numpy().astype(np.uint8)
        rec = {
            "schema": "ddm_et3_solve_within_cvp_pair.v1",
            "pair": int(pair),
            "flips_before": flips0,
            "label_ceiling_flips_left": int(label_after.sum()),
            "label_ceiling_net_fixed": label_ceiling_net_fixed,
            "label_ceiling_fixed": int((flips0_map & ~label_after).sum()),
            "label_ceiling_broken": int(((~flips0_map) & label_after).sum()),
            "band_px": int(band.sum()),
            "band_snapped_px": int(snapped.sum()),
            "band_snap_tax": float(snapped.sum() / max(1, band.sum())),
            "d_pose_before": d_pose_before,
            "metric_weights": weight_stats,
            "controls": {
                "C2_parent_argmax_matches_cache": True,
                "C3_gt_argmax_matches_cache": True,
                "offset_shape": list(np.asarray(current_offsets[pair]).shape),
                "null_basis_max_abs_A_times_N": basis_cert["max_abs_A_times_N"],
            },
            "solve_within": {
                "cap_ladder_attempts": attempts,
                "selected_attempt": best["attempt"],
                "diagnostics": best["diagnostics"],
            },
            "cvp_realized": {
                **scored,
                "eta_realized": (
                    (flips0 - int(scored["flips_after"])) / label_ceiling_net_fixed
                    if label_ceiling_net_fixed
                    else None
                ),
                "d_pose_before": d_pose_before,
                "d_pose_ratio": d_pose_after / d_pose_before if d_pose_before else None,
                "pose_sse_delta": pose_sse_delta,
                "seg_delta_S_no_rate": seg_delta_s,
                "pose_delta_S_against_parent": pose_delta_s,
                "joint_delta_S_no_rate_against_parent_pose": seg_delta_s + pose_delta_s,
                "changed_scorer_pixels": int((cvp_scorer_u8 != base_sc_u8).any(axis=2).sum()),
                "changed_scorer_channel_values": int((cvp_scorer_u8 != base_sc_u8).sum()),
                "yuv6_residual": yuv6_shift(base_sc_u8, cvp_scorer_u8),
                "realizer_receipt": cvp_receipt,
                "C_after": confusion(lgt, cvp_lam).tolist(),
            },
            "elapsed_s": time.time() - started,
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
            "promotion_eligible": PROMOTION_ELIGIBLE,
        }
        rows.append(rec)
        append_jsonl(rows_path, rec)
        current_payload = build_payload(
            args,
            rows,
            parent_archive_sha=parent_archive_sha,
            parent_score=parent_score,
            field_current=field_current,
            basis_cert=basis_cert,
            scorer_custody=scorer_custody,
            selection=selection,
            n4_eta=float(n4_eta) if n4_eta is not None else None,
            recall_evidence=recall_evidence,
        )
        write_json_atomic(summary_path, current_payload)
        write_json_atomic(receipt_summary_path, current_payload)
        agg = current_payload["aggregate"]
        print(
            f"[et3] pair {pair:3d} ({len(rows)}/{len(pairs)}) "
            f"eta_pair={rec['cvp_realized']['eta_realized']:+.4f} "
            f"pose={rec['cvp_realized']['d_pose_ratio']:.4f}x "
            f"agg_eta={agg.get('solve_within_cvp_eta_realized')} "
            f"verdict={agg.get('verdict')} "
            f"[{time.time()-started:.1f}s]",
            flush=True,
        )

    final = build_payload(
        args,
        rows,
        parent_archive_sha=parent_archive_sha,
        parent_score=parent_score,
        field_current=field_current,
        basis_cert=basis_cert,
        scorer_custody=scorer_custody,
        selection=selection,
        n4_eta=float(n4_eta) if n4_eta is not None else None,
        recall_evidence=recall_evidence,
    )
    write_json_atomic(summary_path, final)
    write_json_atomic(receipt_summary_path, final)
    print(json.dumps(final["aggregate"], indent=1, default=jsonable), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-raw", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/submission/inflated/0.raw"))
    ap.add_argument("--parent-archive", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes"))
    ap.add_argument("--parent-argmax", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy"))
    ap.add_argument("--parent-score", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/aggregate.json"))
    ap.add_argument("--current-offsets", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/tq1c_block16_offsets.npy"))
    ap.add_argument("--phase-field-summary", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806/phase_field/phase_field_rederive_summary.json"))
    ap.add_argument("--gt-cache", type=Path, default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--gt-mkv", type=Path, default=REPO / "upstream/videos/0.mkv")
    ap.add_argument("--upstream-root", type=Path, default=REPO / "upstream")
    ap.add_argument("--et1-n32-json", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et1_20260803/et1_b16_realization_n32.json"))
    ap.add_argument("--sw1-summary", type=Path, default=REPO / ".omx/research/ddm_sw1_20260806/sw1_null_basis_summary.json")
    ap.add_argument("--bulk-dir", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_et3_20260806"))
    ap.add_argument("--receipt-dir", type=Path, default=REPO / ".omx/research/ddm_et3_20260806")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--cap-ladder", default="15")
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--convergence-patience-evals", type=int, default=0)
    ap.add_argument("--convergence-min-improvement", type=int, default=1)
    ap.add_argument("--lambda-saliency", type=float, default=1.0)
    ap.add_argument("--outside-weight", type=float, default=0.02)
    ap.add_argument("--saliency-clip", type=float, default=20.0)
    ap.add_argument("--cvp-tap-radius", type=int, default=0)
    ap.add_argument("--cvp-max-channel-candidates", type=int, default=9)
    ap.add_argument("--cvp-max-pixel-candidates", type=int, default=16)
    ap.add_argument("--cvp-max-combinations", type=int, default=250000)
    ap.add_argument("--pose-max-threshold", type=float, default=1.04)
    ap.add_argument("--pose-median-abs-tol", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
