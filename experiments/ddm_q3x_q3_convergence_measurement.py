#!/usr/bin/env python
"""ddm_q3x -- Q3 convergence measurement for task #837.

Question measured here
----------------------
Take the sq1 seg-only solved-paint realizer, project its scorer-lattice frame_1
delta onto Q3 (the frame_1 yuv6-null subspace, rank 6/12 per 2x2 scorer block),
realize that projected delta back through camera-resolution uint8, and score
through the real frozen CPU DistortionNet path.

This is deliberately not the same as sq1's null-constrained re-solve. It answers
the charter's retention question: how much of the seg-only realizer's already
measured effect survives a Q3 projection?

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.
score_claim=false, promotion_eligible=false.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "experiments"):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

from ddm_sq1_eta_seg_realization import (
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
    label_boundary_band,
    seq_len,
)
from ddm_sq1_pose_null_constrained_paint import (
    pose_null_projector,
    project_null,
    snap_band_to_blocks,
    yuv6_shift,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (
    confusion,
    realize_scorer_paint_to_camera,
    resize_to_scorer,
    solve_margin_optimal_paint,
)

from tac.optimization.rw1_true_domain_instruments import (
    block_mask_from_scorer_mask,
    cap_receipt_from_solver_diagnostics,
    element_grade_vector,
    parse_cap_ladder,
    realize_q3_delta_lattice_native,
)
from tac.subset_selection import MODE_STRIDED, governing_ratio, select

ARGMAX_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
DEFAULT_SUB_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_gt3_20260803")
DEFAULT_GT3_RECEIPT = Path("/Volumes/VertigoDataTier/pact/ddm_gt3_20260803/gt3_free_basis_n600.json")
DEFAULT_PZ1_DPOSE = REPO / ".omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json"
DEFAULT_OUT = REPO / ".omx/research/ddm_q3x_q3_convergence_measurement_20260803.json"
DEFAULT_GT_MKV = REPO / "upstream/videos/0.mkv"
REALIZERS = ("dk1-cvp", "dk1-dykstra", "dk1-naive", "naive-round")
SOLVER_FORMS = ("project-after", "solve-within-null-basis")

RATE_PER_BYTE = 25.0 / 37_545_489.0
S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance fallback only
        return f"UNKNOWN:{exc}"


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=1, default=_jsonable)
    tmp.replace(path)


def _load_gt3_threshold(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    deciding = data["waterfill_indicator_THE_DECIDING_SCHEME"]
    seg_only = deciding["eta_seg_only_0.7895"]
    pose_neutral = deciding["eta_pose_neutral_0.5406"]
    eta_seg_only = float(seg_only["eta"])
    break_even_eta = float(pose_neutral["min_break_even_eta_over_bins"])
    threshold = break_even_eta / eta_seg_only
    return {
        "break_even_eta": break_even_eta,
        "eta_seg_only_source": eta_seg_only,
        "retained_fraction_threshold": threshold,
        "gt3_bytes_unchanged": float(seg_only["bytes_total"]),
        "gt3_flips_in_taken": float(seg_only["flips_in_taken"]),
        "gt3_dS_at_eta_seg_only": float(seg_only["dS_total"]),
        "gt3_gap": float(data["baseline"]["gap"]),
        "gt3_archive_bytes": float(data["baseline"]["archive_bytes"]),
        "gt3_S_per_byte": float(data["baseline"]["S_per_byte"]),
        "gt3_S_per_flip": float(data["baseline"]["S_per_flip"]),
    }


def _load_pose_population(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    vals: list[float | None] = [None] * N_PAIRS_TOTAL
    for row in data.get("rows", []):
        p = int(row["pair"])
        vals[p] = float(row["d_pose_base"])
    if any(v is None for v in vals):
        return None
    return [float(v) for v in vals]


def _per_pair_flips(gt_cache: np.ndarray, rd_cache: np.ndarray) -> np.ndarray:
    out = np.zeros((N_PAIRS_TOTAL,), dtype=np.int64)
    for i in range(N_PAIRS_TOTAL):
        out[i] = int((np.asarray(gt_cache[i]) != np.asarray(rd_cache[i])).sum())
    return out


def _make_selection(args: argparse.Namespace, flips: np.ndarray, pose_pop: list[float] | None) -> dict[str, Any]:
    sel = select(
        args.n,
        N_PAIRS_TOTAL,
        mode=MODE_STRIDED,
        stride=args.stride,
        offset=args.offset,
        governing=flips.tolist(),
        governing_name="flips_per_pair",
        n_bootstrap=args.selection_bootstrap,
    )
    extra_ratios = []
    if pose_pop is not None:
        extra_ratios.append(
            governing_ratio(
                sel.indices,
                pose_pop,
                name="d_pose_base",
                seed=0,
                n_bootstrap=args.selection_bootstrap,
            ).as_dict()
        )
    return {
        "indices": list(sel.indices),
        "provenance": sel.provenance(),
        "summary": sel.summary(),
        "extra_governing_ratios": extra_ratios,
    }


def _project_solved_delta_to_q3(base: torch.Tensor, paint_hwc: np.ndarray, band: np.ndarray, P: torch.Tensor) -> np.ndarray:
    paint = torch.from_numpy(np.ascontiguousarray(paint_hwc)).permute(2, 0, 1)[None].float()
    mask = torch.from_numpy(band)[None, None].float()
    delta = (paint - base) * mask
    projected = project_null(delta, P)
    cur = torch.clamp(base + projected, 0.0, 255.0)
    return torch.round(cur)[0].permute(1, 2, 0).numpy().astype(np.uint8)


def _score_pair(sc: Scorer, pose_gt: Any, lgt: np.ndarray, flips0: np.ndarray, dec_f0: np.ndarray, cam_f1: np.ndarray) -> dict[str, Any]:
    pair = np.stack([dec_f0, cam_f1])
    lam = sc.seg_argmax(pair)
    after = lam != lgt
    return {
        "flips_after": int(after.sum()),
        "fixed": int((flips0 & ~after).sum()),
        "introduced": int((~flips0 & after).sum()),
        "C_after": confusion(lgt, lam).tolist(),
        "d_pose_after": sc.d_pose(pose_gt, sc.pose_out(pair)),
    }


def _solve_within_null_basis(
    segnet: Any,
    dec_f1: np.ndarray,
    gt_f1: np.ndarray,
    band_snapped: np.ndarray,
    lgt: np.ndarray,
    *,
    steps: int,
    lr: float,
    eval_every: int,
    convergence_patience_evals: int,
    convergence_min_improvement: int,
) -> tuple[int, np.ndarray, str, dict[str, Any]]:
    from ddm_sw1_null_basis_phase_solve import (  # noqa: WPS433 - optional rw1 arm
        block_mask_from_band,
        null_coordinate_basis,
        solve_within_null_basis,
    )
    from ddm_sw1_null_basis_phase_solve import (
        pose_constraint_matrix as sw1_pose_constraint_matrix,
    )

    basis_np, basis_cert = null_coordinate_basis()
    basis_t = torch.from_numpy(basis_np.astype(np.float32))
    constraint_t = torch.from_numpy(sw1_pose_constraint_matrix().astype(np.float32))
    weights = np.ones((SEG_H, SEG_W), dtype=np.float32)
    block_mask = block_mask_from_band(band_snapped)
    paint, diagnostics = solve_within_null_basis(
        segnet,
        dec_f1,
        gt_f1,
        lgt,
        block_mask,
        weights,
        basis_t,
        constraint_t,
        steps=steps,
        lr=lr,
        eval_every=eval_every,
        convergence_patience_evals=convergence_patience_evals,
        convergence_min_improvement=convergence_min_improvement,
    )
    selected = diagnostics["selected"]
    proxy = int(selected["best_proxy_phase_target_flips"])
    tag = f"{selected['start']}@{selected['best_step']}"
    diagnostics["basis_cert"] = basis_cert
    diagnostics["selected"]["stop_reason"] = next(
        (
            row["stop_reason"]
            for row in diagnostics.get("starts", [])
            if row.get("start") == selected.get("start")
        ),
        "UNKNOWN_STOP_REASON",
    )
    return proxy, paint, tag, diagnostics


def _solve_once(
    args: argparse.Namespace,
    segnet: Any,
    dec_f1: np.ndarray,
    gt_f1: np.ndarray,
    band: np.ndarray,
    band_snapped: np.ndarray,
    lgt: np.ndarray,
    *,
    steps: int,
) -> tuple[int, np.ndarray, str, dict[str, Any]]:
    if args.solver_form == "project-after":
        return solve_margin_optimal_paint(
            segnet,
            dec_f1,
            gt_f1,
            band,
            lgt,
            steps=steps,
            lr=args.lr,
            eval_every=args.eval_every,
            convergence_patience_evals=args.convergence_patience_evals,
            convergence_min_improvement=args.convergence_min_improvement,
        )
    if args.solver_form == "solve-within-null-basis":
        return _solve_within_null_basis(
            segnet,
            dec_f1,
            gt_f1,
            band_snapped,
            lgt,
            steps=steps,
            lr=args.lr,
            eval_every=args.eval_every,
            convergence_patience_evals=args.convergence_patience_evals,
            convergence_min_improvement=args.convergence_min_improvement,
        )
    raise RuntimeError(f"unknown solver form: {args.solver_form}")


def _solve_cap_ladder(
    args: argparse.Namespace,
    segnet: Any,
    dec_f1: np.ndarray,
    gt_f1: np.ndarray,
    band: np.ndarray,
    band_snapped: np.ndarray,
    lgt: np.ndarray,
) -> dict[str, Any]:
    ladder = parse_cap_ladder(args.cap_ladder, fallback=args.steps)
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cap in ladder:
        nbad, paint, tag, diagnostics = _solve_once(
            args,
            segnet,
            dec_f1,
            gt_f1,
            band,
            band_snapped,
            lgt,
            steps=cap,
        )
        receipt = cap_receipt_from_solver_diagnostics(diagnostics, cap=cap)
        attempt = {
            "cap": int(cap),
            "proxy_flips": int(nbad),
            "tag": tag,
            "diagnostics": diagnostics,
            "cap_stop_receipt": receipt.to_payload(),
        }
        attempts.append(attempt)
        if best is None or int(nbad) < int(best["proxy_flips"]):
            best = {
                "proxy_flips": int(nbad),
                "paint": paint,
                "tag": tag,
                "diagnostics": diagnostics,
                "cap": int(cap),
                "cap_stop_receipt": receipt.to_payload(),
            }
        if receipt.stop_reason == "converged":
            break
    if best is None:
        raise RuntimeError("cap ladder produced no q3x solve attempt")
    ladder_public = [
        {
            "cap": row["cap"],
            "proxy_flips": row["proxy_flips"],
            "tag": row["tag"],
            "cap_stop_receipt": row["cap_stop_receipt"],
            "selected": row["diagnostics"].get("selected"),
        }
        for row in attempts
    ]
    best["ladder_attempts"] = ladder_public
    return best


def _q3_grade_vector(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    realization_grade = "NAIVE-NAMED" if args.realizer == "naive-round" else "OPTIMAL-RECEIPT"
    projection_grade = (
        "NAIVE-NAMED"
        if args.realizer == "naive-round" or args.solver_form == "project-after"
        else "OPTIMAL-RECEIPT"
    )
    return element_grade_vector(
        chain_name="q3x_q3_convergence_realizer",
        overrides={
            "init": ("OPTIMAL-RECEIPT", "solver uses inherited dec/truth multi-start starts"),
            "step_rule": ("NAIVE-NAMED", f"{args.solver_form} Adam step rule, not separately optimized in rw1"),
            "stopping_rule": ("OPTIMAL-RECEIPT", f"CA1 cap-stop receipts over ladder {args.cap_ladder or args.steps}"),
            "metric": ("NAIVE-NAMED", "proxy SegNet cross-entropy/argmax objective retained from source solver"),
            "subset": ("OPTIMAL-RECEIPT", f"bounded {MODE_STRIDED} selection {selection['provenance']}"),
            "realization": (realization_grade, args.realizer),
            "projection": (projection_grade, args.solver_form),
            "tie_breaks": ("NAIVE-NAMED", "best proxy flips then first observed best iterate"),
            "seed": ("UNKNOWN", "no stochastic seed consumed by this small-n CPU advisory path"),
            "caches": ("OPTIMAL-RECEIPT", "argmax cache C2/C3 checked per selected pair"),
        },
    )


def _aggregate(rows: list[dict[str, Any]], gt3: dict[str, float], pose_small_ratio: float) -> dict[str, Any]:
    total_described = int(sum(r["described_in_band"] for r in rows))
    before = int(sum(r["flips_before"] for r in rows))
    orig_after = int(sum(r["seg_only"]["flips_after"] for r in rows))
    proj_after = int(sum(r["q3_projected"]["flips_after"] for r in rows))
    orig_net = before - orig_after
    proj_net = before - proj_after
    orig_eta = orig_net / total_described if total_described else None
    proj_eta = proj_net / total_described if total_described else None
    retained = proj_net / orig_net if orig_net else None

    pose_before = float(np.mean([r["d_pose_before"] for r in rows])) if rows else None
    pose_orig = float(np.mean([r["seg_only"]["d_pose_after"] for r in rows])) if rows else None
    pose_proj = float(np.mean([r["q3_projected"]["d_pose_after"] for r in rows])) if rows else None
    pose_ratio = pose_proj / pose_before if pose_before and pose_before > 0 else None
    pose_delta = pose_proj - pose_before if pose_proj is not None and pose_before is not None else None

    projected_eta_for_gt3 = (
        gt3["eta_seg_only_source"] * retained if retained is not None else None
    )
    priced_row = None
    if projected_eta_for_gt3 is not None:
        dS = (
            gt3["gt3_bytes_unchanged"] * RATE_PER_BYTE
            - gt3["gt3_flips_in_taken"] * projected_eta_for_gt3 * S_PER_FLIP
        )
        priced_row = {
            "bytes_unchanged_from_gt3_scheme": gt3["gt3_bytes_unchanged"],
            "flips_in_taken_from_gt3_scheme": gt3["gt3_flips_in_taken"],
            "projected_eta_for_gt3_join": projected_eta_for_gt3,
            "dS_total_at_projected_eta": dS,
            "pct_of_gap_at_projected_eta": -dS / gt3["gt3_gap"] * 100.0,
        }

    crosses = bool(retained is not None and retained >= gt3["retained_fraction_threshold"])
    pose_small = bool(pose_ratio is not None and pose_ratio <= pose_small_ratio)
    if crosses and pose_small:
        outcome = "FIRED"
    elif not crosses:
        outcome = "FOLDED"
    else:
        outcome = "QUEUED_WITH_FIRE_ORDER"

    return {
        "n_rows": len(rows),
        "total_described_in_band": total_described,
        "flips_before": before,
        "seg_only_flips_after": orig_after,
        "q3_projected_flips_after": proj_after,
        "seg_only_net_flip_reduction": orig_net,
        "q3_projected_net_flip_reduction": proj_net,
        "seg_only_eta_net": orig_eta,
        "q3_projected_eta_net": proj_eta,
        "retained_seg_fraction": retained,
        "retained_fraction_threshold": gt3["retained_fraction_threshold"],
        "crosses_retained_fraction_threshold": crosses,
        "d_pose_before_mean": pose_before,
        "d_pose_seg_only_after_mean": pose_orig,
        "d_pose_q3_projected_after_mean": pose_proj,
        "d_pose_q3_projected_delta_mean": pose_delta,
        "d_pose_q3_projected_ratio": pose_ratio,
        "pose_small_threshold_ratio": pose_small_ratio,
        "pose_small": pose_small,
        "instrument_capacity": {
            "one_flip_retained_fraction_quantum": (1.0 / orig_net) if orig_net else None,
            "one_flip_eta_quantum": (1.0 / total_described) if total_described else None,
            "scorer_sites_per_pair": SEG_H * SEG_W,
            "sample_pairs": len(rows),
            "projector_rank_per_2x2_block": 6,
            "projector_dof_per_2x2_block": 12,
        },
        "priced_row_spec": priced_row,
        "outcome": outcome,
    }


def _payload(args: argparse.Namespace, selection: dict[str, Any], gt3: dict[str, float], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "ddm_q3x_q3_convergence_measurement.v1",
        "utc": _utc(),
        "git": _git_hash(),
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "selection_mode": MODE_STRIDED,
        "selection": selection,
        "inputs": {
            "sub_dir": str(args.sub_dir),
            "gt_mkv": str(args.gt_mkv),
            "argmax_cache": str(args.argmax_cache),
            "gt3_receipt": str(args.gt3_receipt),
            "pz1_dpose": str(args.pz1_dpose),
        },
        "gt3_threshold": gt3,
        "solver": {
            "seg_only_source": "ddm_sq1_stage_decomposition.solve_margin_optimal_paint",
            "solver_form": args.solver_form,
            "steps": args.steps,
            "cap_ladder": list(parse_cap_ladder(args.cap_ladder, fallback=args.steps)),
            "lr": args.lr,
            "eval_every": args.eval_every,
            "convergence_patience_evals": args.convergence_patience_evals,
            "convergence_min_improvement": args.convergence_min_improvement,
            "starts": ["dec", "truth"],
            "projector": "ddm_sq1_pose_null_constrained_paint.pose_null_projector",
            "realizer": args.realizer,
            "dk1_dykstra_iterations": args.dk1_dykstra_iterations,
            "dk1_cvp_tap_radius": args.dk1_cvp_tap_radius,
            "dk1_max_blocks": args.dk1_max_blocks,
            "snap_to_2x2_blocks": True,
        },
        "element_grade_vector": _q3_grade_vector(args, selection),
        "aggregate": _aggregate(rows, gt3, args.pose_small_ratio),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-mode", choices=[MODE_STRIDED], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--selection-bootstrap", type=int, default=500)
    ap.add_argument("--sub-dir", type=Path, default=DEFAULT_SUB_DIR)
    ap.add_argument("--gt-mkv", type=Path, default=DEFAULT_GT_MKV)
    ap.add_argument("--argmax-cache", type=Path, default=ARGMAX_CACHE)
    ap.add_argument("--gt3-receipt", type=Path, default=DEFAULT_GT3_RECEIPT)
    ap.add_argument("--pz1-dpose", type=Path, default=DEFAULT_PZ1_DPOSE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cap-ladder", default="25,50,100")
    ap.add_argument("--lr", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--convergence-patience-evals", type=int, default=0)
    ap.add_argument("--convergence-min-improvement", type=int, default=1)
    ap.add_argument("--realizer", choices=REALIZERS, default="dk1-cvp")
    ap.add_argument("--solver-form", choices=SOLVER_FORMS, default="project-after")
    ap.add_argument("--dk1-dykstra-iterations", type=int, default=8)
    ap.add_argument("--dk1-cvp-tap-radius", type=int, default=1)
    ap.add_argument("--dk1-max-blocks", type=int, default=None,
                    help="bounded smoke valve; default realizes every snapped Q3 block")
    ap.add_argument("--pose-small-ratio", type=float, default=1.05)
    ap.add_argument("--max-chunk-pairs", type=int, default=120)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.selection_mode != MODE_STRIDED:
        raise RuntimeError("only mode=strided is implemented for ddm_q3x")
    if args.n > args.max_chunk_pairs:
        raise RuntimeError(
            f"n={args.n} exceeds max chunk {args.max_chunk_pairs}; split the measurement"
        )

    t0 = time.time()
    gt3 = _load_gt3_threshold(args.gt3_receipt)
    gt_cache = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    rd_cache = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    flips_pop = _per_pair_flips(gt_cache, rd_cache)
    pose_pop = _load_pose_population(args.pz1_dpose)
    selection = _make_selection(args, flips_pop, pose_pop)
    pairs = selection["indices"]

    raw_path = args.sub_dir / "inflated" / "0.raw"
    raw = np.memmap(raw_path, dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))

    rows: list[dict[str, Any]] = []
    if args.resume and args.out.exists():
        existing = json.loads(args.out.read_text())
        rows = list(existing.get("rows", []))
        done = {int(r["pair"]) for r in rows}
        pairs = [p for p in pairs if int(p) not in done]
        print(f"[q3x] resume: {len(rows)} rows loaded, {len(pairs)} remaining", flush=True)

    wanted = set()
    for p in pairs:
        wanted.update({seq_len * int(p), seq_len * int(p) + 1})
    print(
        f"[q3x] mode={args.selection_mode} n_total={selection['provenance']['n']} "
        f"remaining={len(pairs)} stride={args.stride} threshold="
        f"{gt3['retained_fraction_threshold']:.6f}",
        flush=True,
    )
    gt_frames = decode_gt_frames(args.gt_mkv, wanted) if wanted else {}
    sc = Scorer(args.threads)
    segnet = sc.net.segnet
    P = pose_null_projector()
    print(f"[q3x] scorer ready t={time.time()-t0:.1f}s", flush=True)

    for p in pairs:
        p = int(p)
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        if not (lstar == np.asarray(rd_cache[p])).all():
            raise RuntimeError(f"C2 failed: decoded argmax does not match cache for pair {p}")
        if not (lgt == np.asarray(gt_cache[p])).all():
            raise RuntimeError(f"C3 failed: GT argmax does not match cache for pair {p}")

        band = label_boundary_band(lstar, 1)
        band_snapped = snap_band_to_blocks(band)
        flips0 = lstar != lgt
        described = int((flips0 & band).sum())
        pose_gt = sc.pose_out(gt)
        d_pose_before = sc.d_pose(pose_gt, sc.pose_out(dec))

        solve = _solve_cap_ladder(args, segnet, dec[1], gt[1], band, band_snapped, lgt)
        nbad = int(solve["proxy_flips"])
        paint = solve["paint"]
        tag = str(solve["tag"])
        cam_seg_only = realize_scorer_paint_to_camera(dec[1], band, paint)
        seg_only = _score_pair(sc, pose_gt, lgt, flips0, dec[0], cam_seg_only)

        base = resize_to_scorer(dec[1])
        if args.realizer == "naive-round":
            projected_paint = _project_solved_delta_to_q3(base, paint, band, P)
            cam_projected = realize_scorer_paint_to_camera(dec[1], band_snapped, projected_paint)
            realizer_receipt = {
                "schema": "ddm_rw1_q3x_legacy_realizer_receipt.v1",
                "method": "naive-round",
                "projector": "euclidean Q3 projector then scorer-lattice uint8 rounding",
                "snap_to_2x2_blocks": True,
                "score_claim": False,
                "promotion_eligible": False,
            }
        else:
            method = args.realizer.removeprefix("dk1-")
            block_mask = block_mask_from_scorer_mask(band_snapped)
            cam_projected, realizer_receipt = realize_q3_delta_lattice_native(
                camera_frame=dec[1],
                base_scorer=base,
                target_paint_hwc=paint,
                block_mask=block_mask,
                method=method,
                dykstra_iterations=args.dk1_dykstra_iterations,
                cvp_tap_radius=args.dk1_cvp_tap_radius,
                max_blocks=args.dk1_max_blocks,
            )
            projected_paint = (
                torch.round(resize_to_scorer(cam_projected))[0]
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8)
            )
        q3_projected = _score_pair(sc, pose_gt, lgt, flips0, dec[0], cam_projected)
        base_sc = torch.round(base)[0].permute(1, 2, 0).numpy().astype(np.uint8)

        rec = {
            "pair": p,
            "flips_before": int(flips0.sum()),
            "described_in_band": described,
            "band_frac": float(band.mean()),
            "band_frac_snapped": float(band_snapped.mean()),
            "snap_tax": float(band_snapped.sum() / max(1, band.sum())),
            "d_pose_before": d_pose_before,
            "C_before": confusion(lgt, lstar).tolist(),
            "seg_only": {
                "tag": tag,
                "proxy_flips_scorer_lattice": int(nbad),
                "solve_cap": int(solve["cap"]),
                "solve_cap_stop_receipt": solve["cap_stop_receipt"],
                "solve_cap_ladder": solve["ladder_attempts"],
                "solve_diagnostics_selected": solve["diagnostics"].get("selected"),
                **seg_only,
                "eta_net": (
                    (int(flips0.sum()) - int(seg_only["flips_after"])) / max(1, described)
                ),
            },
            "q3_projected": {
                **q3_projected,
                "realizer": args.realizer,
                "realizer_receipt": realizer_receipt,
                "eta_net": (
                    (int(flips0.sum()) - int(q3_projected["flips_after"])) / max(1, described)
                ),
                "yuv6_residual": yuv6_shift(base_sc, projected_paint),
                "changed_scorer_pixels": int((projected_paint != base_sc).any(axis=2).sum()),
                "changed_scorer_channel_values": int((projected_paint != base_sc).sum()),
            },
        }
        rows.append(rec)
        payload = _payload(args, selection, gt3, rows)
        _write_json(args.out, payload)
        agg = payload["aggregate"]
        print(
            f"[q3x] pair {p:3d} ({len(rows)}/{selection['provenance']['n']}) "
            f"orig_eta={rec['seg_only']['eta_net']:+.4f} "
            f"q3_eta={rec['q3_projected']['eta_net']:+.4f} "
            f"retained={agg['retained_seg_fraction']} "
            f"dpose={d_pose_before:.6g}->{rec['q3_projected']['d_pose_after']:.6g} "
            f"[{time.time()-tp:.1f}s]",
            flush=True,
        )

    payload = _payload(args, selection, gt3, rows)
    _write_json(args.out, payload)
    print(
        f"[q3x] DONE outcome={payload['aggregate']['outcome']} "
        f"retained={payload['aggregate']['retained_seg_fraction']} "
        f"threshold={payload['aggregate']['retained_fraction_threshold']} "
        f"t={time.time()-t0:.1f}s -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
