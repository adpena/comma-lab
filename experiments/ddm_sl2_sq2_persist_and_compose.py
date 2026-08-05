#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""SL2: persist SQ2 solved frame_1 bytes, then run terminal pose on them.

This wrapper deliberately reuses the SQ1 solved-paint and EG1/PB1 terminal-pose
surfaces instead of rebuilding either mechanism.  It is bounded to the selected
n32 SQ1 pair set and emits advisory, non-promotable measurements only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "4")

import numpy as np
import torch

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "experiments", REPO / "upstream"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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
from ddm_sq1_stage_decomposition_and_solved_paint import (
    SQ1_TRAJECTORY_STOP_CONFIG,
    realize_scorer_paint_to_camera,
    resize_to_scorer,
)

from tac.optimization.terminal_pose_gn import (
    CandidateArtifactScope,
    PoseAuthorityMode,
    PoseJointEvaluation,
    TerminalPoseCandidateArtifact,
    TerminalPoseGNConfig,
    TerminalPosePacketV1,
    serialize_terminal_pose_packet,
    solve_terminal_pose_gn,
)
from tac.optimization.trajectory_stopping import (
    TrajectoryPoint,
    evaluate_trajectory_stop,
)

DEFAULT_OUT_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_sl2_20260805")
DEFAULT_SUB_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2")
DEFAULT_ARGMAX_CACHE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
DEFAULT_PAIRS_NPY: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_selected_pairs.npy"
)
DEFAULT_GT_MKV: Final = REPO / "upstream/videos/0.mkv"
SEED: Final = 20260728
BASIS_SELECTOR: Final = "eg1_generic_low_frequency_six_v1"
AMPLITUDE_Q8: Final = 512
RANK: Final = 6
AUTHORITY_MARKER: Final = "SL2_N32_INCREMENTAL_EXACT_MACOS_ADVISORY"
AXIS: Final = "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
FRONTIER_LINE: Final = (
    "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; "
    "contest pointer borrowed/unmoved."
)


class SL2Error(ValueError):
    """SL2 fail-closed invariant error."""


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _pose6_from_out(out: dict[str, torch.Tensor]) -> np.ndarray:
    pose = out["pose"][0, :6].detach().cpu().numpy().astype(np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise SL2Error("PoseNet returned malformed pose6")
    return pose


def _pose6(sc: Scorer, pair_u8: np.ndarray) -> np.ndarray:
    return _pose6_from_out(sc.pose_out(pair_u8))


def _d_pose(sc: Scorer, pair_u8: np.ndarray, target_pose6: np.ndarray) -> float:
    pose = _pose6(sc, pair_u8)
    return float(np.mean((pose - target_pose6) ** 2, dtype=np.float64))


def _render_basis(seed: int, basis_selector: str, shape: tuple[int, int, int]) -> np.ndarray:
    if seed != SEED or basis_selector != BASIS_SELECTOR:
        raise SL2Error("terminal pose basis packet key differs")
    height, width, channels = shape
    if channels != 3:
        raise SL2Error("terminal pose basis expects RGB")
    x = np.cos(2.0 * np.pi * (np.arange(width, dtype=np.float64) + 0.5) / width)
    y = np.cos(2.0 * np.pi * (np.arange(height, dtype=np.float64) + 0.5) / height)
    fields = np.zeros((RANK, height, width, channels), dtype=np.float32)
    for channel in range(3):
        fields[channel, :, :, channel] = x[None, :]
        fields[channel + 3, :, :, channel] = y[:, None]
    return fields


def _terminal_packet(coefficients: np.ndarray) -> bytes:
    return serialize_terminal_pose_packet(
        TerminalPosePacketV1(
            seed=SEED,
            basis_selector=BASIS_SELECTOR,
            amplitude_q8=AMPLITUDE_Q8,
            coefficients=np.asarray(coefficients, dtype=np.int16),
        )
    )


def _solve_margin_dec_start(
    segnet,
    dec_f1: np.ndarray,
    band: np.ndarray,
    target_labels: np.ndarray,
    *,
    steps: int,
    lr: float,
    eval_every: int,
    convergence_patience_evals: int,
    convergence_min_improvement: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """SQ1 solved paint, restricted to the dec start used by all SQ2 n32 rows."""

    base = resize_to_scorer(dec_f1)
    tgt = torch.from_numpy(target_labels.astype(np.int64))[None]
    mask = torch.from_numpy(band)[None, None].float()
    x = base.clone().detach()
    best: tuple[int, np.ndarray, int] | None = None
    curve: list[dict[str, int]] = []
    evals_since_best = 0
    stop_reason = "iteration_cap_no_convergence_test"
    trajectory_payload = None

    with torch.enable_grad():
        delta = torch.zeros_like(base, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        for step in range(steps + 1):
            cur = torch.clamp(base * (1.0 - mask) + (x + delta) * mask, 0.0, 255.0)
            if step % eval_every == 0 or step == steps:
                quantized = torch.round(cur).detach()
                with torch.no_grad():
                    labels = segnet(quantized).argmax(dim=1)[0].numpy().astype(np.uint8)
                bad = int((labels != target_labels).sum())
                curve.append({"step": int(step), "proxy_flips": bad})
                improved = (
                    best is None
                    or int(best[0]) - bad >= max(1, int(convergence_min_improvement))
                )
                if improved:
                    paint = quantized[0].permute(1, 2, 0).numpy().astype(np.uint8)
                    best = (bad, paint, int(step))
                    evals_since_best = 0
                else:
                    evals_since_best += 1
                if len(curve) >= SQ1_TRAJECTORY_STOP_CONFIG.min_fit_points:
                    decision = evaluate_trajectory_stop(
                        [
                            TrajectoryPoint(
                                compute=float(point["step"]),
                                objective=float(point["proxy_flips"]),
                            )
                            for point in curve
                        ],
                        SQ1_TRAJECTORY_STOP_CONFIG,
                        safety_bound_compute=float(steps),
                    )
                    trajectory_payload = decision.to_payload()
                    if (
                        decision.should_stop
                        and decision.stop_reason in {"converged_projected", "marginal_below_bar"}
                        and step < steps
                    ):
                        stop_reason = str(decision.stop_reason)
                        break
                if (
                    convergence_patience_evals > 0
                    and evals_since_best >= convergence_patience_evals
                    and step < steps
                ):
                    stop_reason = "plateau_no_proxy_improvement"
                    break
            if step == steps:
                if best is not None and best[2] == steps:
                    stop_reason = "iteration_cap_best_at_cap"
                elif convergence_patience_evals > 0:
                    stop_reason = "iteration_cap_before_plateau"
                break
            loss = torch.nn.functional.cross_entropy(segnet(cur), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    if best is None:
        raise SL2Error("SQ2 solve produced no evaluated iterate")
    diagnostics = {
        "start": "dec",
        "best_step": int(best[2]),
        "best_proxy_flips": int(best[0]),
        "stop_reason": stop_reason,
        "steps_run": int(curve[-1]["step"]) if curve else 0,
        "curve": curve,
        "trajectory_stop": trajectory_payload,
    }
    return best[1], diagnostics


def run_stage1(args: argparse.Namespace, pairs: list[int], sc: Scorer) -> dict[str, Any]:
    out_dir = args.out_dir
    frames_dir = out_dir / "sq2_persisted_frames"
    out_path = out_dir / "sl2_sq2_persist_n32.json"
    rows: list[dict[str, Any]] = []
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        rows = list(existing.get("rows", []))
        done = {int(row["pair"]) for row in rows}
        pairs = [pair for pair in pairs if pair not in done]
        print(f"[sl2] stage1 resume: {len(rows)} rows on disk, {len(pairs)} remaining", flush=True)

    raw = np.memmap(
        args.sub_dir / "inflated/0.raw",
        dtype=np.uint8,
        mode="r",
        shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3),
    )
    cx1 = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gtc = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    wanted_frames = {seq_len * pair + offset for pair in pairs for offset in (0, 1)}
    gt_frames = decode_gt_frames(args.gt_mkv, wanted_frames) if pairs else {}

    def flush() -> None:
        cap_bound = [
            row for row in rows
            if str(row["sq2_solve"]["stop_reason"]).startswith("iteration_cap")
        ]
        payload = {
            "schema": "ddm_sl2_sq2_persist_n32.v1",
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "utc": _utc(),
            "frontier": FRONTIER_LINE,
            "sub_dir": str(args.sub_dir),
            "pairs_npy": str(args.pairs_npy),
            "pairs": [int(pair) for pair in np.load(args.pairs_npy).tolist()],
            "solver": {
                "seg_resource_step_bound": int(args.seg_resource_step_bound),
                "seg_lr": float(args.seg_lr),
                "eval_every": int(args.seg_eval_every),
                "convergence_patience_evals": int(args.seg_convergence_patience_evals),
                "convergence_min_improvement": int(args.seg_convergence_min_improvement),
                "start": "dec",
                "start_recall": "all 32 SQ2 uncap100 selected rows used dec as best start",
            },
            "cap_stop": {
                "bound_rows": len(cap_bound),
                "total_rows": len(rows),
                "bound": len(cap_bound) > 0,
                "note": "iteration_cap_* rows are CapStop rows, not convergence certificates",
            },
            "rows": rows,
        }
        if rows:
            payload["aggregate"] = _stage1_aggregate(rows)
        _atomic_write_json(out_path, payload)

    for index, pair in enumerate(pairs):
        t0 = time.time()
        dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]]).astype(np.uint8)
        current = sc.seg_argmax(dec)
        target = sc.seg_argmax(gt)
        if not np.array_equal(current, np.asarray(cx1[pair], dtype=np.uint8)):
            raise SL2Error(f"cx1 cache mismatch for pair {pair}")
        if not np.array_equal(target, np.asarray(gtc[pair], dtype=np.uint8)):
            raise SL2Error(f"GT argmax cache mismatch for pair {pair}")
        band = label_boundary_band(current, 1)
        flips_before_map = current != target
        paint, solve_diag = _solve_margin_dec_start(
            sc.net.segnet,
            dec[1],
            band,
            target,
            steps=args.seg_resource_step_bound,
            lr=args.seg_lr,
            eval_every=args.seg_eval_every,
            convergence_patience_evals=args.seg_convergence_patience_evals,
            convergence_min_improvement=args.seg_convergence_min_improvement,
        )
        edited_f1 = realize_scorer_paint_to_camera(dec[1], band, paint)
        solved_pair = np.stack([dec[0], edited_f1])
        solved_argmax = sc.seg_argmax(solved_pair)
        flips_after_map = solved_argmax != target
        target_pose6 = _pose6(sc, gt)
        d_pose_before = _d_pose(sc, dec, target_pose6)
        d_pose_solved = _d_pose(sc, solved_pair, target_pose6)
        flat = np.flatnonzero(band.reshape(-1)).astype(np.int64)
        frame_npz = frames_dir / f"pair_{pair:04d}_sq2_solved_frame.npz"
        frame_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            frame_npz,
            pair=np.asarray([pair], dtype=np.int16),
            frame0_base=dec[0],
            frame1_solved=edited_f1,
            target_pose6=target_pose6.astype(np.float64),
            band_flat=flat,
            paint_rgb=np.ascontiguousarray(paint.reshape(-1, 3)[flat].astype(np.uint8)),
        )
        row = {
            "pair": int(pair),
            "local_ordinal": len(rows),
            "flips_before": int(flips_before_map.sum()),
            "flips_after_solved": int(flips_after_map.sum()),
            "d_seg_solved": float(flips_after_map.mean()),
            "fixed": int((flips_before_map & ~flips_after_map).sum()),
            "introduced": int((~flips_before_map & flips_after_map).sum()),
            "described_in_band": int((flips_before_map & band).sum()),
            "band_px": int(band.sum()),
            "d_pose_before": d_pose_before,
            "d_pose_solved_pre_terminal": d_pose_solved,
            "target_pose6_sha256": _sha256_bytes(target_pose6.astype("<f8").tobytes()),
            "sq2_solve": solve_diag,
            "persisted_frame_npz": {
                "path": str(frame_npz),
                "bytes": int(frame_npz.stat().st_size),
                "sha256": _sha256_file(frame_npz),
            },
            "elapsed_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        flush()
        print(
            f"[sl2] sq2 pair {pair:3d} ({index+1}/{len(pairs)}) "
            f"flips {row['flips_before']}->{row['flips_after_solved']} "
            f"d_pose {d_pose_before:.6g}->{d_pose_solved:.6g} "
            f"{solve_diag['stop_reason']} best@{solve_diag['best_step']} "
            f"[{row['elapsed_s']:.1f}s]",
            flush=True,
        )
    return json.loads(out_path.read_text())


def _stage1_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sites = len(rows) * SEG_H * SEG_W
    flips_before = sum(int(row["flips_before"]) for row in rows)
    flips_after = sum(int(row["flips_after_solved"]) for row in rows)
    described = sum(int(row["described_in_band"]) for row in rows)
    fixed = sum(int(row["fixed"]) for row in rows)
    introduced = sum(int(row["introduced"]) for row in rows)
    return {
        "n_pairs": len(rows),
        "sites": sites,
        "flips_before": flips_before,
        "flips_after_solved": flips_after,
        "d_seg_before": flips_before / sites if sites else None,
        "d_seg_solved": flips_after / sites if sites else None,
        "eta_net_pooled": (flips_before - flips_after) / described if described else None,
        "fixed": fixed,
        "introduced": introduced,
        "described_in_band": described,
        "d_pose_solved_pre_terminal_mean": float(
            np.mean([row["d_pose_solved_pre_terminal"] for row in rows], dtype=np.float64)
        ),
        "pose_term_pre_terminal": float(
            math.sqrt(
                10.0
                * np.mean([row["d_pose_solved_pre_terminal"] for row in rows], dtype=np.float64)
            )
        ),
    }


def run_terminal_pose(args: argparse.Namespace, stage1: dict[str, Any], sc: Scorer) -> dict[str, Any]:
    rows = list(stage1.get("rows", []))
    if not rows:
        raise SL2Error("cannot compose terminal pose without persisted SQ2 rows")
    out_path = args.out_dir / "sl2_composed_terminal_pose_n32.json"
    progress_path = args.out_dir / "sl2_terminal_pose_progress.json"
    packet_path = args.out_dir / "sl2_terminal_pose_n32.tpgn"
    n_pairs = len(rows)
    coeffs = np.zeros((n_pairs, RANK), dtype=np.int16)
    base_dposes = np.asarray([row["d_pose_solved_pre_terminal"] for row in rows], dtype=np.float64)
    solved_dposes = np.full(n_pairs, np.nan, dtype=np.float64)
    done: set[int] = set()
    pose_rows: list[dict[str, Any]] = []
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("stage1_sha256") != _sha256_file(args.out_dir / "sl2_sq2_persist_n32.json"):
            raise SL2Error("terminal pose progress stage1 binding differs")
        coeffs = np.asarray(progress["coefficients"], dtype=np.int16)
        solved_dposes = np.asarray(progress["solved_dposes"], dtype=np.float64)
        done = {int(x) for x in progress["done_ordinals"]}
        pose_rows = list(progress.get("rows", []))
        print(f"[sl2] terminal resume: {len(done)} rows on disk", flush=True)

    d_seg_solved = float(stage1["aggregate"]["d_seg_solved"])

    def flush() -> None:
        effective = np.where(np.isfinite(solved_dposes), solved_dposes, base_dposes)
        packet = _terminal_packet(coeffs)
        packet_path.write_bytes(packet)
        progress_payload = {
            "schema": "ddm_sl2_terminal_pose_progress.v1",
            "axis": AXIS,
            "stage1_sha256": _sha256_file(args.out_dir / "sl2_sq2_persist_n32.json"),
            "coefficients": coeffs.tolist(),
            "base_dposes": base_dposes.tolist(),
            "solved_dposes": solved_dposes.tolist(),
            "done_ordinals": sorted(done),
            "rows": pose_rows,
        }
        _atomic_write_json(progress_path, progress_payload)
        payload = {
            "schema": "ddm_sl2_composed_terminal_pose_n32.v1",
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "utc": _utc(),
            "frontier": FRONTIER_LINE,
            "stage1_receipt": {
                "path": str(args.out_dir / "sl2_sq2_persist_n32.json"),
                "sha256": progress_payload["stage1_sha256"],
            },
            "terminal_pose": {
                "basis_selector": BASIS_SELECTOR,
                "seed": SEED,
                "amplitude_q8": AMPLITUDE_Q8,
                "relinearizations": int(args.pose_relinearizations),
                "line_search": [1.0, 0.5, 0.25],
                "authority_mode": PoseAuthorityMode.STALE_REHEARSAL.value,
                "authority_marker": AUTHORITY_MARKER,
            },
            "terminal_packet": {
                "path": str(packet_path),
                "bytes": len(packet),
                "sha256": _sha256_bytes(packet),
                "scope": "terminal-section-only; not a full archive",
            },
            "aggregate": {
                "n_pairs": n_pairs,
                "d_seg_solved": d_seg_solved,
                "d_pose_pre_terminal_mean": float(np.mean(base_dposes)),
                "d_pose_composed_effective_mean": float(np.mean(effective)),
                "pose_term_pre_terminal": float(math.sqrt(10.0 * np.mean(base_dposes))),
                "pose_term_composed_effective": float(math.sqrt(10.0 * np.mean(effective))),
                "terminal_pairs_done": len(done),
                "terminal_pairs_total": n_pairs,
                "all_terminal_done": len(done) == n_pairs,
            },
            "rows": pose_rows,
        }
        _atomic_write_json(out_path, payload)

    for ordinal, row in enumerate(rows):
        if ordinal in done:
            continue
        t0 = time.time()
        frame_npz = Path(row["persisted_frame_npz"]["path"])
        loaded = np.load(frame_npz)
        parent = np.stack([loaded["frame0_base"], loaded["frame1_solved"]]).astype(np.uint8)
        target = np.asarray(loaded["target_pose6"], dtype=np.float64)
        effective_now = np.where(np.isfinite(solved_dposes), solved_dposes, base_dposes)
        pose_sum_others = float(effective_now.sum() - effective_now[ordinal])

        def artifact_for(codes: np.ndarray, _ordinal: int = ordinal) -> TerminalPoseCandidateArtifact:
            matrix = coeffs.copy()
            matrix[_ordinal] = np.asarray(codes, dtype=np.int16)
            packet = _terminal_packet(matrix)
            return TerminalPoseCandidateArtifact(
                outer_archive=packet,
                terminal_packet=packet,
                scope=CandidateArtifactScope.TERMINAL_SECTION_ONLY,
            )

        parent_frame1 = parent[1].copy()

        def score_candidate(
            realized_pair: np.ndarray,
            artifact: TerminalPoseCandidateArtifact,
            _target: np.ndarray = target,
            _others: float = pose_sum_others,
            _parent_frame1: np.ndarray = parent_frame1,
        ) -> PoseJointEvaluation:
            if not np.array_equal(realized_pair[1], _parent_frame1):
                raise SL2Error("terminal pose changed persisted frame_1")
            pose6 = _pose6(sc, realized_pair)
            d_pose_i = float(np.mean((pose6 - _target) ** 2, dtype=np.float64))
            d_pose_mean = (_others + d_pose_i) / n_pairs
            return PoseJointEvaluation(
                pose6=pose6,
                d_seg=d_seg_solved,
                d_pose=d_pose_mean,
                archive_bytes=artifact.archive_bytes,
                archive_sha256=artifact.archive_sha256,
                sample_count=n_pairs,
                authority_marker=AUTHORITY_MARKER,
                custody_digest=None,
                realized=True,
            )

        result = solve_terminal_pose_gn(
            parent,
            target,
            _render_basis,
            artifact_for,
            score_candidate,
            seed=SEED,
            basis_selector=BASIS_SELECTOR,
            config=TerminalPoseGNConfig(
                relinearizations=args.pose_relinearizations,
                amplitude_q8=AMPLITUDE_Q8,
                authority_mode=PoseAuthorityMode.STALE_REHEARSAL,
            ),
            initial_coefficients=coeffs[ordinal],
            pair_index=ordinal,
        )
        coeffs[ordinal] = np.asarray(result.final_coefficients, dtype=np.int16)
        solved_dposes[ordinal] = float(result.pose_mse_final)
        done.add(ordinal)
        pose_row = {
            "pair": int(row["pair"]),
            "local_ordinal": ordinal,
            "d_pose_initial_pair": float(result.pose_mse_initial),
            "d_pose_final_pair": float(result.pose_mse_final),
            "joint_action_initial": float(result.initial_evaluation.joint_action),
            "joint_action_final": float(result.final_evaluation.joint_action),
            "final_population_d_pose": float(result.final_evaluation.d_pose),
            "steps": [step.to_payload() for step in result.steps],
            "final_coefficients": result.final_coefficients.tolist(),
            "strict_realized_improvement": bool(result.strict_realized_improvement),
            "frame1_sha256": _sha256_bytes(parent[1].tobytes()),
            "elapsed_s": round(time.time() - t0, 1),
        }
        pose_rows.append(pose_row)
        flush()
        print(
            f"[sl2] terminal pair {row['pair']:3d} ({len(done)}/{n_pairs}) "
            f"d_pose {pose_row['d_pose_initial_pair']:.6g}->{pose_row['d_pose_final_pair']:.6g} "
            f"steps {len(result.steps)} [{pose_row['elapsed_s']:.1f}s]",
            flush=True,
        )
    flush()
    return json.loads(out_path.read_text())


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sub-dir", type=Path, default=DEFAULT_SUB_DIR)
    parser.add_argument("--argmax-cache", type=Path, default=DEFAULT_ARGMAX_CACHE)
    parser.add_argument("--pairs-npy", type=Path, default=DEFAULT_PAIRS_NPY)
    parser.add_argument("--gt-mkv", type=Path, default=DEFAULT_GT_MKV)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seg-resource-step-bound", type=int, default=100)
    parser.add_argument("--seg-lr", type=float, default=4.0)
    parser.add_argument("--seg-eval-every", type=int, default=5)
    parser.add_argument("--seg-convergence-patience-evals", type=int, default=3)
    parser.add_argument("--seg-convergence-min-improvement", type=int, default=1)
    parser.add_argument("--pose-relinearizations", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-terminal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = build_args()
    if str(args.out_dir).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise SL2Error("persisted evidence must not be under /tmp")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [int(pair) for pair in np.load(args.pairs_npy).tolist()]
    if args.limit:
        pairs = pairs[: args.limit]
    torch.set_num_threads(args.threads)
    sc = Scorer(args.threads)
    if args.skip_stage1:
        stage1 = json.loads((args.out_dir / "sl2_sq2_persist_n32.json").read_text())
    else:
        stage1 = run_stage1(args, pairs, sc)
    if not args.skip_terminal:
        run_terminal_pose(args, stage1, sc)
    print(f"[sl2] done -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
