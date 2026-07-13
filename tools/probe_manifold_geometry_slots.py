#!/usr/bin/env python3
"""$0 manifold-slot probe: row companding, Fisher margin, and orbit curvature.

This is a read-only measurement over committed/cached artifacts.  It does not
invoke either scorer, train a model, mutate a run directory, or make a score
claim.  The two stage receipts are written atomically and reused when their
input fingerprints match, so interruption after S1/S2 loses no completed work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROW_ARTIFACT = REPO / ".omx/research/dseg_reducibility_gt_margin_n600_20260623.json"
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUT = REPO / ".omx/research/manifold_geometry_slots_probe_receipt_20260713.json"
SCHEMA = "manifold_geometry_slots_probe.v1"
HORIZON_ROW = 174.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    view = np.ascontiguousarray(array).view(np.uint8)
    digest = hashlib.sha256()
    step = 64 * 1024 * 1024
    for start in range(0, view.size, step):
        digest.update(memoryview(view[start : start + step]))
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("density must have positive finite mass")
    return values / total


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(p)
    q = _normalize(q)
    mid = 0.5 * (p + q)
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    p_pos = p > 0.0
    q_pos = q > 0.0
    p_term[p_pos] = p[p_pos] * np.log(p[p_pos] / mid[p_pos])
    q_term[q_pos] = q[q_pos] * np.log(q[q_pos] / mid[q_pos])
    return float(0.5 * (p_term.sum() + q_term.sum()))


def _density_metrics(target: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    target = _normalize(target)
    candidate = _normalize(candidate)
    n_top = max(1, math.ceil(0.10 * target.size))
    chosen = np.argsort(candidate)[-n_top:]
    return {
        "js_divergence_nats": _js_divergence(target, candidate),
        "cdf_l1_rows": float(np.abs(np.cumsum(target) - np.cumsum(candidate)).sum()),
        "flip_mass_captured_by_top_10pct_allocated_rows": float(target[chosen].sum()),
        "max_row_allocation_fraction": float(candidate.max()),
    }


def fisher_pairwise_wall_distance(logit_gap: np.ndarray | float) -> np.ndarray:
    """Fisher--Rao distance to p1=p2 after renormalizing the top-two pair.

    The sqrt map has unit-sphere coordinates sqrt(p); the categorical Fisher
    metric is the radius-two sphere metric.  This is an exact two-class
    projection, not the unrecoverable five-class distance from a gap alone.
    """

    gap = np.asarray(logit_gap, dtype=np.float64)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(gap, -700.0, 700.0)))
    p2 = 1.0 - p1
    argument = np.abs(np.sqrt(p1) - np.sqrt(p2)) / math.sqrt(2.0)
    return 2.0 * np.arcsin(np.clip(argument, 0.0, 1.0))


def _stage_s1_s2(row_artifact: Path, row_sha: str) -> dict[str, Any]:
    source = json.loads(row_artifact.read_text(encoding="utf-8"))
    if int(source["n_pairs_scored"]) != 600:
        raise ValueError("S1/S2 load-bearing probe requires the settled n600 artifact")

    per_row = sorted(source["per_row_full"], key=lambda item: int(item["row"]))
    rows = np.asarray([float(item["row"]) for item in per_row], dtype=np.float64)
    flip_rate = np.asarray([float(item["flip_rate"]) for item in per_row], dtype=np.float64)
    ground_mask = rows > HORIZON_ROW
    ground_rows = rows[ground_mask]
    ground_flip = _normalize(flip_rate[ground_mask])
    distance = ground_rows - HORIZON_ROW

    candidate_density = {
        "identity_uniform": np.ones_like(distance),
        "hyperbolic_log_depth_unshifted": 1.0 / distance,
        "projective_inverse_depth_unshifted": 1.0 / np.square(distance),
    }
    comparisons = {
        name: _density_metrics(ground_flip, density)
        for name, density in candidate_density.items()
    }

    shifted_fits: dict[str, dict[str, float]] = {}
    offsets = np.geomspace(0.25, 256.0, 2000)
    for name, exponent in (("shifted_hyperbolic_log_depth", 1.0), ("shifted_projective_inverse_depth", 2.0)):
        best: tuple[float, float, np.ndarray] | None = None
        for offset in offsets:
            density = np.power(distance + float(offset), -exponent)
            divergence = _js_divergence(ground_flip, density)
            if best is None or divergence < best[0]:
                best = (divergence, float(offset), density)
        assert best is not None
        shifted_fits[name] = {
            "exponent": exponent,
            "fitted_softening_offset_rows": best[1],
            **_density_metrics(ground_flip, best[2]),
        }

    flip_quantiles = source["margin_quantiles"]["flip"]
    quantile_rows: dict[str, dict[str, float]] = {}
    for name, raw_gap in flip_quantiles.items():
        gap = float(raw_gap)
        fr = float(fisher_pairwise_wall_distance(gap))
        flat = 0.5 * gap
        quantile_rows[name] = {
            "logit_gap": gap,
            "pairwise_fisher_wall_distance": fr,
            "flat_shadow_gap_over_2": flat,
            "relative_error_vs_gap_over_2": (fr / flat - 1.0) if flat > 0.0 else 0.0,
        }

    hist = source["margin_histogram"]
    edges = np.asarray(hist["edges"], dtype=np.float64)
    counts = np.asarray(hist["flip_counts"], dtype=np.float64)
    finite = np.isfinite(edges[:-1]) & np.isfinite(edges[1:]) & (edges[1:] < 1e8) & (counts > 0)
    mid = 0.5 * (edges[:-1][finite] + edges[1:][finite])
    weights = counts[finite]
    fr_mid = fisher_pairwise_wall_distance(mid)
    slope = float(np.sum(weights * mid * fr_mid) / np.sum(weights * np.square(mid)))
    rel_rmse = float(
        np.sqrt(np.sum(weights * np.square(fr_mid - slope * mid)) / np.sum(weights * np.square(fr_mid)))
    )

    return {
        "schema": f"{SCHEMA}.stage_s1_s2",
        "input": {
            "path": str(row_artifact.relative_to(REPO)),
            "sha256": row_sha,
            "n_pairs": 600,
            "authority_tier": source.get("authority_tier"),
        },
        "S1_input_chart": {
            "probe_scope": (
                "all-class measured flip density restricted to image rows v>174; a ground-class-only row ledger "
                "does not exist, so class routing remains a named confound"
            ),
            "horizon_row": HORIZON_ROW,
            "ground_rows": int(ground_rows.size),
            "measured_flip_mass_below_horizon_fraction": float(flip_rate[ground_mask].sum() / flip_rate.sum()),
            "measured_peak_row_below_horizon": int(ground_rows[int(np.argmax(ground_flip))]),
            "measured_flip_mass_rows_175_through_210": float(ground_flip[ground_rows <= 210.0].sum()),
            "candidate_comparisons": comparisons,
            "shifted_family_fits": shifted_fits,
            "nonparametric_equal_flip_metric": {
                "law": "sqrt(g_vv) = rho(v) proportional to measured flip density w(v)",
                "js_divergence_nats": 0.0,
                "interpretation": "uniform metric arc-length bins equalize first-order flip mass per bin",
            },
            "existing_chart_delta": (
                "GroundFrameChart #194 is a temporal xi-homography precomposition, not a row-density compander. "
                "Its projective skeleton does not implement the measured equal-flip metric; unshifted inverse-depth "
                "is therefore a comparison model, not a claim about the chart's learned capacity allocation."
            ),
        },
        "S2_head_simplex": {
            "probe_scope": (
                "exact top-two renormalized Fisher--Rao distance computed from the n600 logit-gap distribution; "
                "the full five-class Fisher distance is not identifiable from a top1-top2 gap alone"
            ),
            "formula": "d_FR,pair(delta)=2 asin(|sqrt(sigmoid(delta))-sqrt(sigmoid(-delta))|/sqrt(2))",
            "flip_margin_quantiles": quantile_rows,
            "weighted_linear_fit_on_nonempty_flip_histogram_bins": {
                "slope_through_origin": slope,
                "relative_rmse": rel_rmse,
                "counts_used": int(weights.sum()),
            },
            "ordering": "strictly monotone in nonnegative logit gap; it cannot change the argmax decision cells",
        },
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def _participation_rank(centered: np.ndarray) -> tuple[float, int]:
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    if float(energy.sum()) <= 0.0:
        return 0.0, 0
    pr = float(np.square(energy.sum()) / np.square(energy).sum())
    cumulative = np.cumsum(energy) / energy.sum()
    rank90 = int(np.searchsorted(cumulative, 0.90) + 1)
    return pr, rank90


def _menger_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    ab = b - a
    ac = c - a
    bc = c - b
    lab = float(np.linalg.norm(ab))
    lac = float(np.linalg.norm(ac))
    lbc = float(np.linalg.norm(bc))
    denom = lab * lac * lbc
    if denom <= 1e-12:
        return None
    gram = max(float(np.dot(ab, ab) * np.dot(ac, ac) - np.dot(ab, ac) ** 2), 0.0)
    return float(2.0 * math.sqrt(gram) / denom)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p10": None, "median": None, "p90": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(array.mean()),
    }


def _stage_s5(gt_cache: Path, archive_sha: str) -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "src"))
    from tac.boundary_math.lane_ground_factorization import (
        fit_frame_ground_lanes,
        robust_dim_scales,
        track_ground_lanes,
    )

    with np.load(gt_cache, allow_pickle=False) as archive:
        n_pairs = int(archive["n_pairs"])
        if n_pairs != 600:
            raise ValueError("S5 load-bearing probe requires gt_n600")
        lstars = np.asarray(archive["lstars"])
    lstars_sha = _sha256_array(lstars)
    per_frame = [fit_frame_ground_lanes(lstars[index], index) for index in range(n_pairs)]
    tracks = track_ground_lanes(per_frame)
    scales = robust_dim_scales(tracks)

    all_coeffs = np.concatenate([track.coeffs for track in tracks], axis=0)
    center = np.median(all_coeffs, axis=0)
    normalized_all = (all_coeffs - center) / scales
    global_pr, global_rank90 = _participation_rank(normalized_all - normalized_all.mean(axis=0))

    local_pr: dict[int, list[float]] = {5: [], 11: [], 21: []}
    local_rank90: dict[int, list[float]] = {5: [], 11: [], 21: []}
    curvatures: list[float] = []
    path_chord_ratios: list[float] = []
    for track in tracks:
        coords = (track.coeffs - center) / scales
        for window in local_pr:
            if coords.shape[0] < window:
                continue
            half = window // 2
            for index in range(half, coords.shape[0] - half):
                patch = coords[index - half : index + half + 1]
                pr, rank90 = _participation_rank(patch - patch.mean(axis=0))
                local_pr[window].append(pr)
                local_rank90[window].append(float(rank90))
        for index in range(1, coords.shape[0] - 1):
            if int(track.frames[index] - track.frames[index - 1]) > 2:
                continue
            if int(track.frames[index + 1] - track.frames[index]) > 2:
                continue
            curvature = _menger_curvature(coords[index - 1], coords[index], coords[index + 1])
            if curvature is not None and np.isfinite(curvature):
                curvatures.append(curvature)
        if coords.shape[0] >= 5:
            for start in range(0, coords.shape[0] - 4):
                patch = coords[start : start + 5]
                path = float(np.linalg.norm(np.diff(patch, axis=0), axis=1).sum())
                chord = float(np.linalg.norm(patch[-1] - patch[0]))
                if chord > 1e-12:
                    path_chord_ratios.append(path / chord)

    return {
        "schema": f"{SCHEMA}.stage_s5",
        "input": {
            "path": str(gt_cache.relative_to(REPO)),
            "archive_sha256": archive_sha,
            "member": "lstars",
            "member_sha256": lstars_sha,
            "n_pairs": n_pairs,
        },
        "lane_orbit_sampling": {
            "tracks": len(tracks),
            "observations": int(all_coeffs.shape[0]),
            "ambient_coefficient_dimension": int(all_coeffs.shape[1]),
            "global_pooled_participation_rank": global_pr,
            "global_pooled_rank90": global_rank90,
            "robust_dim_scales": [float(value) for value in scales],
        },
        "local_track_windows": {
            str(window): {
                "participation_rank": _summary(values),
                "rank90": _summary(local_rank90[window]),
            }
            for window, values in local_pr.items()
        },
        "extrinsic_only_diagnostics": {
            "normalized_menger_curvature": _summary(curvatures),
            "five_observation_path_over_chord": _summary(path_chord_ratios),
            "units": "dimensionless after per-coordinate robust scaling",
        },
        "intrinsic_curvature_verdict": {
            "status": "NOT_IDENTIFIABLE_FROM_THIS_CACHE",
            "reason": (
                "each tracked lane observation supplies a time-ordered one-parameter curve in R^8; "
                "one-dimensional Riemannian manifolds have identically zero intrinsic curvature, while the "
                "reported Menger values are embedding curvature. Pooling 29 tracks mixes strata and does not "
                "create the independent tangent 2-planes required for sectional-curvature estimation"
            ),
            "whitney_consequence": (
                "none: Whitney's generic 2d+1 embedding bound depends on intrinsic/topological dimension, "
                "not curvature; curvature could only inform reach/chart count after multi-direction neighborhood sampling"
            ),
        },
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def _stage_path(out: Path, suffix: str) -> Path:
    return out.with_name(f"manifold_geometry_slots_probe_{suffix}_20260713.json")


def _load_reusable(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key, value in expected.items():
        if payload.get("input", {}).get(key) != value:
            return None
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-artifact", type=Path, default=DEFAULT_ROW_ARTIFACT)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    row_artifact = args.row_artifact.resolve()
    gt_cache = args.gt_cache.resolve()
    out = args.out.resolve()
    row_sha = _sha256_file(row_artifact)
    gt_sha = _sha256_file(gt_cache)

    s1_s2_path = _stage_path(out, "s1_s2")
    s1_s2 = _load_reusable(s1_s2_path, {"sha256": row_sha, "n_pairs": 600})
    if s1_s2 is None:
        s1_s2 = _stage_s1_s2(row_artifact, row_sha)
        _atomic_json(s1_s2_path, s1_s2)
        print(f"wrote stage S1/S2: {s1_s2_path.relative_to(REPO)}", flush=True)
    else:
        print(f"reused stage S1/S2: {s1_s2_path.relative_to(REPO)}", flush=True)

    s5_path = _stage_path(out, "s5")
    s5 = _load_reusable(
        s5_path,
        {"archive_sha256": gt_sha, "n_pairs": 600},
    )
    if s5 is None:
        s5 = _stage_s5(gt_cache, gt_sha)
        _atomic_json(s5_path, s5)
        print(f"wrote stage S5: {s5_path.relative_to(REPO)}", flush=True)
    else:
        print(f"reused stage S5: {s5_path.relative_to(REPO)}", flush=True)

    bundle = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU numpy advisory; n600 cached artifacts; no scorer/evaluator; non-promotable]",
        "provenance": {
            "argv": [str(value) for value in sys.argv],
            "canonical_command": ".venv/bin/python tools/probe_manifold_geometry_slots.py",
            "config": {
                "row_artifact": str(row_artifact.relative_to(REPO)),
                "gt_cache": str(gt_cache.relative_to(REPO)),
                "out": str(out.relative_to(REPO)),
                "horizon_row": HORIZON_ROW,
            },
            "git_head": _git_head(),
            "tool": {
                "path": str(Path(__file__).resolve().relative_to(REPO)),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "inputs": {
                "row_artifact_sha256": row_sha,
                "gt_cache_archive_sha256": gt_sha,
                "gt_cache_member_lstars_sha256": s5["input"]["member_sha256"],
            },
            "stage_receipt_sha256": {
                "S1_S2": _sha256_file(s1_s2_path),
                "S5": _sha256_file(s5_path),
            },
            "determinism": {
                "rng": "none",
                "seed": "not_applicable_no_rng",
                "stage_reuse_requires_input_fingerprint": True,
                "writes": "atomic_tmp_plus_os_replace",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "stages": {
            "S1_S2": str(s1_s2_path.relative_to(REPO)),
            "S5": str(s5_path.relative_to(REPO)),
        },
        "S1_input_chart": s1_s2["S1_input_chart"],
        "S2_head_simplex": s1_s2["S2_head_simplex"],
        "S3_temporal_spacetime": {
            "status": "DERIVED_ONLY_IN_THIS_PROBE",
            "measurement_design": (
                "compare a fixed quantized separatrix representation encoded (a) per frame and (b) as initial "
                "curve plus xi transport, event marks, receiver phase, and event residuals; require identical decoded "
                "world-sheets before attributing byte savings"
            ),
            "settled_law_source": "tac.canonical_equations.rate_law_ladder_20260713.TEMPORAL_CHAIN",
        },
        "S4_optimizer_manifolds": {
            "status": "SOURCE_AUDIT_ONLY",
            "source": "tac.canonical_equations.witness_modular_norm_assignment_20260713.MODULE_NORM_ASSIGNMENTS",
            "stiefel_film_owner": "muon_round2_wire (assessment-only here)",
        },
        "S5_witness_manifold": s5,
        "false_authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "scorer_calls": 0,
            "evaluator_calls": 0,
            "paid_dispatches": 0,
        },
    }
    _atomic_json(out, bundle)
    print(f"wrote bundle: {out.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
