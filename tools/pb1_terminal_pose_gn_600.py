#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb1 P3 — terminal six-equation pose GN over ALL 600 pairs (eg1 E3 production shape).

Per the pose-terminal law (pose AFTER frozen seg, on the conditioned base) and
rv1-R1's #400 dxi-pose-polish leg: runs the committed
``tac.optimization.terminal_pose_gn.solve_terminal_pose_gn`` once per pair on
the FROZEN final seg endpoint frames.  Frame 1 is the receiver's rendered
camera frame (byte-identical throughout, asserted by the module); frame 0 is
the receiver's generic base policy (``zeros`` = current inert-stub semantics,
or ``copy`` = copy-predict frame0:=frame1, a zero-counted-byte generic
receiver program per the locked copy-PREDICT law).

Basis = the committed eg1 generic six-field cosine basis
(seed 20260728, ``eg1_generic_low_frequency_six_v1``, amplitude_q8 512).
Counted payload = TerminalPosePacketV1 [600,6] int16 coefficients, carried as
a third STORED member of the composed outer archive (constant bytes during
the solve; P5 re-prices a Brotli-compressed member at composition).

Pose solving is per-pair separable (PoseNet reads one pair; coefficients are
per-pair rows), so a partial sweep banks partial improvement exactly and the
progress file makes the sweep resumable at pair granularity.

Authority: STALE_REHEARSAL mode (module default) because ContestAxis has no
advisory member; evaluations attest sample_count=600 with the honest marker
``FULL_N600_INCREMENTAL_EXACT_MACOS_ADVISORY`` (d_pose mean recomputed
exactly from the per-pair cache at every candidate; d_seg frozen — frame 1
untouched).  [macOS-CPU advisory]; score_claim=false.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_pb1_p3_terminal_pose_receipt.v1"
SEED = 20260728
BASIS_SELECTOR = "eg1_generic_low_frequency_six_v1"
AMPLITUDE_Q8 = 512
RANK = 6
N_PAIRS = 600
MARKER = "FULL_N600_INCREMENTAL_EXACT_MACOS_ADVISORY"
BYTE_PRICE = 25.0 / 37_545_489.0
COMPOSED_MEMBERS = ("manifest.json", "state/tr1.ddt1", "state/pose.tpgn")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)


def deterministic_stored_zip(members: dict[str, bytes]) -> bytes:
    if tuple(members) != COMPOSED_MEMBERS:
        raise SystemExit("composed member order differs")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED,
                         allowZip64=False) as archive:
        for name in COMPOSED_MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(info, members[name])
    return stream.getvalue()


def render_basis(seed: int, basis_selector: str,
                 shape: tuple[int, int, int]) -> np.ndarray:
    """The committed eg1 generic low-frequency six-field cosine basis."""
    if seed != SEED or basis_selector != BASIS_SELECTOR:
        raise RuntimeError("terminal pose basis packet key differs")
    height, width, channels = shape
    if channels != 3:
        raise RuntimeError("expects RGB frame0")
    x = np.cos(2.0 * np.pi * (np.arange(width, dtype=np.float64) + 0.5) / width)
    y = np.cos(2.0 * np.pi * (np.arange(height, dtype=np.float64) + 0.5) / height)
    fields = np.zeros((RANK, height, width, channels), dtype=np.float32)
    for channel in range(3):
        fields[channel, :, :, channel] = x[None, :]
        fields[channel + 3, :, :, channel] = y[:, None]
    return fields


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path,
                    help="frozen final seg endpoint TR1 archive")
    ap.add_argument("--pose-targets", required=True, type=Path)
    ap.add_argument("--endpoint-dseg", required=True, type=float,
                    help="frozen endpoint full-n600 d_seg mean (frame1 fixed)")
    ap.add_argument("--frame0-policy", choices=("zeros", "copy"),
                    required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=N_PAIRS)
    ap.add_argument("--relinearizations", type=int, default=2)
    ap.add_argument("--max-wall-minutes", type=float, default=200.0)
    ap.add_argument("--probe-pairs", type=int, default=0,
                    help="if >0: measure base d_pose for BOTH frame0 policies"
                         " on this many pairs, write the probe receipt, exit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    sys.path.insert(0, str(REPO / "experiments"))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path
    from train_witness_realized_through_R_mlx import _cpu_pose_raw

    from tac.optimization import ddm_tr1_runtime as rt
    from tac.optimization.terminal_pose_gn import (
        CandidateArtifactScope,
        PoseJointEvaluation,
        TerminalPoseCandidateArtifact,
        TerminalPoseGNConfig,
        TerminalPosePacketV1,
        serialize_terminal_pose_packet,
        solve_terminal_pose_gn,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tr1_bytes = args.archive.read_bytes()
    parsed = rt.parse_archive(tr1_bytes)
    packet_bytes = rt.reemit_packet(parsed.packet)
    rows = json.loads(args.pose_targets.read_text())["rows"]
    targets = [np.asarray(r["center"], dtype=np.float64) for r in rows]
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p in posenet_cpu.parameters():
        p.requires_grad = False

    def frame0_for(f1: np.ndarray) -> np.ndarray:
        if args.frame0_policy == "zeros":
            return np.zeros_like(f1)
        return f1.copy()

    if args.probe_pairs > 0:
        rows_out = []
        for i in range(args.probe_pairs):
            f1 = rt.render_frame1_camera_uint8(parsed.packet, i)
            dz = float(np.mean((_cpu_pose_raw(posenet_cpu,
                                              np.zeros_like(f1), f1)
                                - targets[i]) ** 2))
            dc = float(np.mean((_cpu_pose_raw(posenet_cpu, f1.copy(), f1)
                                - targets[i]) ** 2))
            rows_out.append({"pair": i, "d_pose_zeros": dz,
                             "d_pose_copy": dc})
        probe = {
            "schema": "ddm_pb1_p3_frame0_policy_probe.v1",
            "n_pairs": args.probe_pairs,
            "mean_zeros": float(np.mean([r["d_pose_zeros"] for r in rows_out])),
            "mean_copy": float(np.mean([r["d_pose_copy"] for r in rows_out])),
            "rows": rows_out,
            "evidence_axis": "[macOS-CPU advisory]",
        }
        out = args.out_dir / "p3_frame0_policy_probe.json"
        _atomic_write(out, json.dumps(probe, indent=1, sort_keys=True) + "\n")
        print(json.dumps({k: probe[k] for k in
                          ("n_pairs", "mean_zeros", "mean_copy")}))
        return

    progress_path = args.out_dir / "p3_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        coeffs = np.asarray(progress["coefficients"], dtype=np.int16)
        base_dposes = np.asarray(progress["base_dposes"], dtype=np.float64)
        solved_dposes = np.asarray(progress["solved_dposes"],
                                   dtype=np.float64)
        done = set(progress["done_pairs"])
        if progress["frame0_policy"] != args.frame0_policy:
            raise SystemExit("progress file frame0 policy differs")
    else:
        coeffs = np.zeros((N_PAIRS, RANK), dtype=np.int16)
        base_dposes = np.full(N_PAIRS, np.nan)
        solved_dposes = np.full(N_PAIRS, np.nan)
        done = set()

    def compose(coeff_matrix: np.ndarray) -> tuple[bytes, bytes]:
        packet = TerminalPosePacketV1(
            seed=SEED, basis_selector=BASIS_SELECTOR,
            amplitude_q8=AMPLITUDE_Q8,
            coefficients=np.asarray(coeff_matrix, dtype=np.int16))
        pose_payload = serialize_terminal_pose_packet(packet)
        manifest = {
            "schema": "ddm_pb1_composed_archive.v1",
            "members": list(COMPOSED_MEMBERS),
            "tr1_sha256": _sha(tr1_bytes),
            "pose_sha256": _sha(pose_payload),
            "research_only": True,
            "score_claim": False,
            "pointer_moved": False,
        }
        outer = deterministic_stored_zip({
            "manifest.json": json.dumps(manifest, sort_keys=True,
                                        separators=(",", ":")).encode(),
            "state/tr1.ddt1": packet_bytes,
            "state/pose.tpgn": pose_payload,
        })
        return outer, pose_payload

    t_start = time.time()
    # Precompute EVERY pair's base d_pose once so the attested incremental
    # full-population mean is EXACT from the first candidate onward.
    if not np.isfinite(base_dposes).all():
        from train_witness_realized_through_R_mlx import (
            cpu_verdict_d_pose_batch,
        )

        print("[base] measuring all-pair base d_pose for policy "
              f"{args.frame0_policy}", flush=True)
        for b0 in range(0, N_PAIRS, 6):
            b1 = min(b0 + 6, N_PAIRS)
            idxs = [j for j in range(b0, b1)
                    if not np.isfinite(base_dposes[j])]
            if not idxs:
                continue
            f1s = [rt.render_frame1_camera_uint8(parsed.packet, j)
                   for j in idxs]
            f0s = [frame0_for(f) for f in f1s]
            dp = cpu_verdict_d_pose_batch(
                posenet_cpu, f0s, f1s, [targets[j] for j in idxs])
            for j, v in zip(idxs, dp, strict=True):
                base_dposes[j] = float(v)
        print(f"[base] mean base d_pose {np.mean(base_dposes):.5f} "
              f"({time.time()-t_start:.0f}s)", flush=True)

    wall_cap = args.max_wall_minutes * 60.0
    solved_this_run = 0
    for i in range(args.start, args.end):
        if i in done:
            continue
        if time.time() - t_start > wall_cap:
            print(f"[wall-cap] stopping before pair {i}", flush=True)
            break
        f1 = rt.render_frame1_camera_uint8(parsed.packet, i)
        f0 = frame0_for(f1)
        parent_pair = np.stack([f0, f1], axis=0)
        target = targets[i]

        effective_now = np.where(np.isfinite(solved_dposes),
                                 solved_dposes, base_dposes)
        pose_sum_others = float(effective_now.sum() - effective_now[i])

        def artifact_for_coefficients(codes: np.ndarray,
                                      _i: int = i) -> TerminalPoseCandidateArtifact:
            matrix = coeffs.copy()
            matrix[_i] = np.asarray(codes, dtype=np.int16)
            outer, pose_payload = compose(matrix)
            return TerminalPoseCandidateArtifact(
                outer_archive=outer,
                terminal_packet=pose_payload,
                scope=CandidateArtifactScope.FULL_OUTER_ARCHIVE,
            )

        def score_realized_candidate(realized_pair: np.ndarray,
                                     artifact: TerminalPoseCandidateArtifact,
                                     _i: int = i,
                                     _others: float = pose_sum_others):
            pose6 = _cpu_pose_raw(posenet_cpu, realized_pair[0],
                                  realized_pair[1])
            d_pose_i = float(np.mean((pose6 - targets[_i]) ** 2))
            d_pose_mean = (_others + d_pose_i) / N_PAIRS
            return PoseJointEvaluation(
                pose6=pose6,
                d_seg=args.endpoint_dseg,
                d_pose=d_pose_mean,
                archive_bytes=artifact.archive_bytes,
                archive_sha256=artifact.archive_sha256,
                sample_count=N_PAIRS,
                authority_marker=MARKER,
                custody_digest=None,
                realized=True,
            )

        t0 = time.time()
        result = solve_terminal_pose_gn(
            parent_pair,
            target,
            render_basis,
            artifact_for_coefficients,
            score_realized_candidate,
            seed=SEED,
            basis_selector=BASIS_SELECTOR,
            config=TerminalPoseGNConfig(
                relinearizations=args.relinearizations,
                amplitude_q8=AMPLITUDE_Q8,
            ),
            initial_coefficients=coeffs[i],
            pair_index=i,
        )
        wall = time.time() - t0
        coeffs[i] = np.asarray(result.final_coefficients, dtype=np.int16)
        solved_dposes[i] = float(result.pose_mse_final)
        done.add(i)
        solved_this_run += 1
        if solved_this_run % 5 == 0 or wall > 30:
            print(f"[pair {i}] mse {result.pose_mse_initial:.4f} -> "
                  f"{result.pose_mse_final:.4f} steps {len(result.steps)} "
                  f"{wall:.1f}s", flush=True)
        if solved_this_run % 10 == 0:
            _atomic_write(progress_path, json.dumps({
                "schema": "ddm_pb1_p3_progress.v1",
                "frame0_policy": args.frame0_policy,
                "endpoint_archive_sha256": _sha(tr1_bytes),
                "coefficients": coeffs.tolist(),
                "base_dposes": base_dposes.tolist(),
                "solved_dposes": solved_dposes.tolist(),
                "done_pairs": sorted(done),
            }))

    _atomic_write(progress_path, json.dumps({
        "schema": "ddm_pb1_p3_progress.v1",
        "frame0_policy": args.frame0_policy,
        "endpoint_archive_sha256": _sha(tr1_bytes),
        "coefficients": coeffs.tolist(),
        "base_dposes": base_dposes.tolist(),
        "solved_dposes": solved_dposes.tolist(),
        "done_pairs": sorted(done),
    }))

    solved_mask = np.isfinite(solved_dposes)
    effective = np.where(solved_mask, solved_dposes, base_dposes)
    outer, pose_payload = compose(coeffs)
    receipt = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "frame0_policy": args.frame0_policy,
        "endpoint_archive_sha256": _sha(tr1_bytes),
        "endpoint_dseg": args.endpoint_dseg,
        "pairs_done": len(done),
        "base_d_pose_mean": (float(np.nanmean(base_dposes))
                             if np.isfinite(base_dposes).any() else None),
        "solved_d_pose_mean_effective": (float(np.mean(effective))
                                         if np.isfinite(effective).all()
                                         else None),
        "composed_outer_bytes_stored": len(outer),
        "pose_member_bytes": len(pose_payload),
        "composed_outer_sha256": _sha(outer),
        "wall_seconds": time.time() - t_start,
        "authority_note": ("STALE_REHEARSAL module mode; honest marker "
                           + MARKER + "; d_seg frozen (frame1 untouched,"
                           " module-asserted); acceptance = pose-MSE AND"
                           " joint-action strict decrease per candidate"),
        "generated_by": "tools/pb1_terminal_pose_gn_600.py",
    }
    out = args.out_dir / "p3_terminal_pose_receipt.json"
    _atomic_write(out, json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    (args.out_dir / "p3_composed_outer.zip").write_bytes(outer)
    print(f"[done] pairs {len(done)}/{N_PAIRS} receipt {out}", flush=True)


if __name__ == "__main__":
    main()
