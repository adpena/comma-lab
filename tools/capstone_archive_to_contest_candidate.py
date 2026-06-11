# SPDX-License-Identifier: MIT
"""Capstone archive -> contest-CPU candidate closure pipeline (the critical-path tool).

This is the step between "trained vehicle" and "moved pointer": it turns a
byte-closed capstone ``archive.zip`` into a banked contest-CPU candidate by
chaining the EXISTING, separately-verified pieces (reuse, not reinvent):

  1. ARCHIVE -> ACTUAL numpy inflate (``tac.capstone_vq_nerv.inflate``): the real
     contest decode path — ``decode_archive`` + ``render_all_camera_frames``
     (bicubic A3 camera upscale to 874x1164) -> camera-resolution uint8 frames
     written to disk, EXACTLY as ``inflate.sh`` produces for the evaluator.
  2. INFLATED FRAMES -> CONTEST-FAITHFUL LOCAL SCORE: read the on-disk camera
     frames back, feed each pair into the frozen ``TorchScorerBridge`` (which
     resizes camera->scorer 384x512 and applies eval_roundtrip = A2), recompute
     ``d_seg`` / ``d_pose`` from the REAL reloaded int8/fp16 frames, and recompute
     ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_zip_bytes/37_545_489`` using
     the ACTUAL archive.zip file size (the denominator the contest charges).
     This is the contest-faithful LOCAL number (audit A2+A3 closed): it scores
     the same camera-resolution uint8 frames the contest scorer reads off disk.
  3. LOCAL SCORE -> PREDICTED contest-Linux-x86_64-CPU SCORE: apply the
     ``tac.optimization.local_cpu_contest_drift`` calibration (RUNG-B macOS->Linux
     bias +1.05e-5, SegNet-rounding; the near-exact same-device-mode simulator).
     Emit BOTH the point projection and the CONSERVATIVE projection (subtract
     bias, add guard) + the submit rule (conservative < frontier_CPU => candidate).
  4. -> ARMED (un-fired) PAIRED-EVAL DISPATCH PACKET: the lane-claim command + the
     Modal Linux-x86_64-CPU eval command on the EXACT archive bytes + the cost
     estimate (~$0.12). Gated on the candidate's conservative projection being a
     real sub-frontier (or operator greenlight). NOTHING is fired here.

NO-FAKE discipline (highest emphasis):
  * The numpy-inflate score is the contest-faithful LOCAL number, tagged
    ``[macOS-CPU advisory]`` — NON-PROMOTABLE. macOS is NOT 1:1 contest CPU.
  * The drift-simulator output is a PREDICTION, tagged ``[predicted contest-CPU]``
    — NOT a score claim. The ONLY authority is the actual contest eval (the armed
    packet). The simulator is the PREDICTOR + the trigger for the real eval.
  * No score is claimed from the simulator; no MPS anywhere; pure numpy decode of
    the REAL archive bytes; torch-CPU scorer only.

CLI::

    .venv/bin/python tools/capstone_archive_to_contest_candidate.py \
        --archive experiments/results/<run>/archive.zip \
        --targets-cache experiments/results/capstone_gt_targets_cache \
        --frontier-cpu-score 0.191099824 \
        --out-dir experiments/results/capstone_closure/<run> \
        [--batch-size 8] [--device cpu] [--candidate-id <id>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Contest constants (verified against upstream/evaluate.py:63-92 +
# tac.capstone_vq_nerv.export.CapstoneArchiveAccount.rate).
UNCOMPRESSED_SIZE_BYTES = 37_545_489  # sum(upstream/videos/*) — the rate denominator
RATE_SCALE = 25.0
SEG_SCALE = 100.0
POSE_SCALE = 10.0
CAMERA_H, CAMERA_W = 874, 1164
SCORER_H, SCORER_W = 384, 512

# RUNG-B drift offset (macOS-torch-CPU -> Linux-x86_64-CPU), from the drift ladder
# memo (.omx/research/local_to_contest_scorer_drift_ladder_and_correction_20260611.md
# section 2 + tac.optimization.local_cpu_contest_drift fit): +1.05e-5 bias, 3e-6 guard.
# These are HNeRV-medal-band-class-bounded projection PRIORS, never score claims.
DEFAULT_BIAS_LOCAL_MINUS_CONTEST = 1.05e-5
DEFAULT_GUARD_BAND = 3.0e-6

LOCAL_AXIS_TAG = "[macOS-CPU advisory]"
PREDICTED_AXIS_TAG = "[predicted contest-CPU]"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ContestFaithfulLocalScore:
    """The contest-faithful LOCAL score recomputed from the on-disk inflated frames.

    ``[macOS-CPU advisory]`` — NON-PROMOTABLE. The honest local predictor of the
    ``inflate.sh -> evaluate.py`` path: same camera-resolution uint8 frames the
    contest scorer reads, same eval_roundtrip, same component formula.
    """

    d_seg: float
    d_pose: float
    rate_unscaled: float
    archive_zip_bytes: int
    num_pairs: int
    score_seg_contribution: float
    score_pose_contribution: float
    score_rate_contribution: float
    score: float
    pose_enabled: bool
    full_600_pair: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": LOCAL_AXIS_TAG,
            "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "avg_segnet_dist": self.d_seg,
            "avg_posenet_dist": self.d_pose,
            "rate_unscaled": self.rate_unscaled,
            "archive_zip_bytes": self.archive_zip_bytes,
            "n_pairs": self.num_pairs,
            "score_seg_contribution": self.score_seg_contribution,
            "score_pose_contribution": self.score_pose_contribution,
            "score_rate_contribution": self.score_rate_contribution,
            "score_recomputed_from_components": self.score,
            "pose_enabled": self.pose_enabled,
            "full_600_pair_faithful": self.full_600_pair,
        }


def run_numpy_inflate_to_disk(archive_path: Path, dst_raw: Path) -> dict[str, Any]:
    """Step 1: run the ACTUAL numpy inflate on disk (the real contest decode path).

    Reuses ``tac.capstone_vq_nerv.inflate`` exactly (no reimplementation): reads
    the archive + config, decodes the int8/fp16 render basis, renders every pair
    through the pure-numpy reference with the bicubic A3 camera upscale, and writes
    the flat ``(N, 874, 1164, 3)`` uint8 tensor file the evaluator's
    ``TensorVideoDataset`` reads. Returns the decoded config + frame metadata.
    """
    from tac.capstone_vq_nerv.inflate import (
        _read_archive_and_config,
        decode_archive,
        render_all_camera_frames,
    )

    archive_bytes, config = _read_archive_and_config(archive_path)
    decoded = decode_archive(archive_bytes, config)
    frames = render_all_camera_frames(decoded)  # (num_pairs*2, 874, 1164, 3) uint8
    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    dst_raw.write_bytes(frames.tobytes(order="C"))
    return {
        "config": config,
        "num_pairs": int(decoded["num_pairs"]),
        "frames_shape": list(frames.shape),
        "raw_bytes": dst_raw.stat().st_size,
    }


def _build_bridge(num_pairs: int, targets_cache: Path, device: str):
    """Build the frozen ``TorchScorerBridge`` exactly as the capstone trainer does.

    Reuses ``run_capstone_campaign._load_or_build_targets`` semantics inline (the
    GT seg/pose targets cache + ``load_frozen_distortion_net`` + ``TorchScorerBridge``
    construction) so the re-score is byte-faithful to the live advisory's scorer.
    """
    import torch

    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge
    from tac.score_aware_loop.targets import (
        build_gt_targets,
        load_frozen_distortion_net,
    )

    targets_cache.mkdir(parents=True, exist_ok=True)
    cache = targets_cache / f"gt_targets_n{num_pairs}.pt"
    net = load_frozen_distortion_net(device=device)
    if cache.exists():
        blob = torch.load(cache, map_location=device, weights_only=False)
        seg_t, pose_t = blob["seg"], blob["pose"]
    else:
        seg_t, pose_t, _ = build_gt_targets(net, max_pairs=num_pairs, device=device)
        torch.save({"seg": seg_t, "pose": pose_t, "n": num_pairs}, cache)
    bridge = TorchScorerBridge(
        net,
        seg_targets_hard=seg_t,
        pose_targets=pose_t,
        eval_roundtrip=True,
    )
    return bridge


def score_inflated_frames_on_disk(
    raw_path: Path,
    num_pairs: int,
    archive_zip_bytes: int,
    targets_cache: Path,
    *,
    batch_size: int = 8,
    device: str = "cpu",
) -> ContestFaithfulLocalScore:
    """Step 2: recompute d_seg/d_pose from the on-disk inflated camera frames.

    Reads the flat ``(N, 874, 1164, 3)`` uint8 tensor back from disk (the SAME bytes
    the contest evaluator reads), reshapes to per-pair ``(b, 2, 3, 874, 1164)``, and
    feeds each batch into the frozen bridge. The bridge bilinear-downsamples
    camera->scorer (384x512, matching the evaluator) and applies eval_roundtrip
    (A2), so the score is the honest contest predictor on the REAL reloaded frames.
    """
    import torch

    frames = np.fromfile(raw_path, dtype=np.uint8).reshape(
        num_pairs * 2, CAMERA_H, CAMERA_W, 3
    )
    bridge = _build_bridge(num_pairs, targets_cache, device)
    pose_enabled = getattr(bridge, "pose_enabled", False)

    d_seg_total = 0.0
    d_pose_total = 0.0
    n = 0
    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        b = end - start
        idx_np = np.arange(start, end)
        # per-pair (b, 2, 3, H, W) float32 from the on-disk camera frames
        batch = frames[start * 2 : end * 2].astype(np.float32)  # (b*2, H, W, 3)
        batch = batch.reshape(b, 2, CAMERA_H, CAMERA_W, 3)
        render_n2chw = np.transpose(batch, (0, 1, 4, 2, 3))  # (b, 2, 3, H, W)
        idx_t = torch.from_numpy(idx_np.astype(np.int64))
        d_seg_total += bridge.exact_d_seg(render_n2chw, idx_t) * b
        if pose_enabled:
            d_pose_total += bridge.exact_d_pose(render_n2chw, idx_t) * b
        n += b

    denom = max(n, 1)
    d_seg = d_seg_total / denom
    d_pose = (d_pose_total / denom) if pose_enabled else 0.0
    rate_unscaled = archive_zip_bytes / UNCOMPRESSED_SIZE_BYTES
    seg_c = SEG_SCALE * d_seg
    pose_c = math.sqrt(POSE_SCALE * d_pose) if pose_enabled else 0.0
    rate_c = RATE_SCALE * rate_unscaled
    return ContestFaithfulLocalScore(
        d_seg=d_seg,
        d_pose=d_pose,
        rate_unscaled=rate_unscaled,
        archive_zip_bytes=archive_zip_bytes,
        num_pairs=num_pairs,
        score_seg_contribution=seg_c,
        score_pose_contribution=pose_c,
        score_rate_contribution=rate_c,
        score=seg_c + pose_c + rate_c,
        pose_enabled=pose_enabled,
        full_600_pair=(num_pairs == 600),
    )


@dataclass(frozen=True)
class ContestCPUPrediction:
    """Step 3: the PREDICTED contest-Linux-x86_64-CPU score (drift-corrected).

    ``[predicted contest-CPU]`` — a PREDICTION, NOT a score claim. The only
    authority is the actual contest eval (the armed packet).
    """

    local_score: float
    bias_local_minus_contest: float
    guard_band: float
    point_projection: float
    conservative_projection: float
    frontier_cpu_score: float
    conservative_beats_frontier: bool
    margin_vs_frontier: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": PREDICTED_AXIS_TAG,
            "score_claim": False,
            "promotion_eligible": False,
            "authority": "false_authority_prediction_only_exact_eval_is_arbiter",
            "local_score": self.local_score,
            "bias_local_minus_contest": self.bias_local_minus_contest,
            "guard_band": self.guard_band,
            "predicted_contest_cpu_point": self.point_projection,
            "predicted_contest_cpu_conservative": self.conservative_projection,
            "frontier_cpu_score": self.frontier_cpu_score,
            "conservative_beats_frontier": self.conservative_beats_frontier,
            "margin_vs_frontier": self.margin_vs_frontier,
        }


def predict_contest_cpu(
    local_score: float,
    frontier_cpu_score: float,
    *,
    bias: float = DEFAULT_BIAS_LOCAL_MINUS_CONTEST,
    guard: float = DEFAULT_GUARD_BAND,
) -> ContestCPUPrediction:
    """Step 3: apply the RUNG-B drift correction (macOS-CPU -> Linux-x86_64-CPU).

    Lower score is better, so the conservative projection subtracts the fitted
    bias (macOS reads slightly high) and adds the guard band back. This mirrors
    ``DriftCalibration.conservative_projected_contest_score``. The candidate
    qualifies for a paid exact eval iff the conservative projection beats the CPU
    frontier (per the drift memo section 3.2 submit rule).
    """
    point = local_score - bias
    conservative = point + guard
    beats = conservative < frontier_cpu_score
    return ContestCPUPrediction(
        local_score=local_score,
        bias_local_minus_contest=bias,
        guard_band=guard,
        point_projection=point,
        conservative_projection=conservative,
        frontier_cpu_score=frontier_cpu_score,
        conservative_beats_frontier=beats,
        margin_vs_frontier=frontier_cpu_score - conservative,
    )


def build_eval_packet(
    *,
    archive_path: Path,
    archive_sha256: str,
    archive_zip_bytes: int,
    prediction: ContestCPUPrediction,
    candidate_id: str,
    out_dir: Path,
    est_cost_usd: float = 0.12,
) -> dict[str, Any]:
    """Step 4: ARM (do NOT fire) the paired contest-CPU eval dispatch packet.

    Emits the lane-claim command + the Modal Linux-x86_64-CPU eval command on the
    EXACT archive bytes + the cost estimate. ``armed`` is True iff the conservative
    projection beats the frontier (a real sub-frontier candidate); otherwise the
    packet is recorded but flagged ``observe_only`` (operator greenlight still
    overrides). NOTHING is dispatched here (estimate-first; HARVEST-OR-LOSE only
    applies once the operator fires it).
    """
    eval_lane = f"lane_capstone_{candidate_id}_contest_cpu_eval_20260611"
    claim_cmd = (
        ".venv/bin/python tools/claim_lane_dispatch.py claim "
        f"--lane-id {eval_lane} --platform modal --agent claude:capstone_closure "
        "--instance modal_cpu --status active "
        f'--notes "contest-CPU exact eval of capstone candidate {candidate_id} '
        f'sha {archive_sha256[:12]} ({archive_zip_bytes} B); HARVEST-OR-LOSE within 24h"'
    )
    eval_cmd = (
        "PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach "
        "experiments/modal_auth_eval_cpu.py "
        f"--archive {archive_path} "
        f"--expected-archive-sha256 {archive_sha256} "
        "--inflate-sh src/tac/capstone_vq_nerv/runtime/inflate.sh "
        f"--output-dir {out_dir / 'modal_cpu_eval'} "
        f"--lane-id {eval_lane} --claim-agent claude:capstone_closure "
        "--detach --provider-detach-ack"
    )
    armed = prediction.conservative_beats_frontier
    return {
        "schema": "capstone_contest_cpu_eval_packet.v1",
        "candidate_id": candidate_id,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_zip_bytes": archive_zip_bytes,
        "estimated_cost_usd": est_cost_usd,
        "estimated_wall_clock_min": "60-120",
        "hardware": "Linux x86_64 CPU (Modal container) — contest leaderboard axis",
        "armed": armed,
        "fire_gate": (
            "conservative_projection < frontier_cpu_score (real sub-frontier) "
            "OR explicit operator greenlight"
        ),
        "recommended_action": (
            "dispatch_contest_cpu_exact_eval" if armed else "observe_only"
        ),
        "fired": False,
        "score_claim": False,
        "lane_claim_command": claim_cmd,
        "modal_cpu_eval_command": eval_cmd,
        "harvest_note": (
            "After firing: harvest within 24h (Modal .spawn result-cache TTL). "
            "The contest-CPU result_json is the ONLY score authority."
        ),
    }


def run_closure_pipeline(
    *,
    archive_path: Path,
    targets_cache: Path,
    frontier_cpu_score: float,
    out_dir: Path,
    batch_size: int = 8,
    device: str = "cpu",
    candidate_id: str = "",
    bias: float = DEFAULT_BIAS_LOCAL_MINUS_CONTEST,
    guard: float = DEFAULT_GUARD_BAND,
) -> dict[str, Any]:
    """Chain steps 1-4 and return the full closure record."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_zip_bytes = archive_path.stat().st_size
    archive_sha256 = _sha256_file(archive_path)
    if not candidate_id:
        candidate_id = f"{archive_path.parent.name}_{archive_sha256[:8]}"

    # Step 1: ACTUAL numpy inflate -> camera-resolution frames on disk.
    raw_path = out_dir / "inflated" / "0.raw"
    inflate_meta = run_numpy_inflate_to_disk(archive_path, raw_path)
    num_pairs = inflate_meta["num_pairs"]

    # Step 2: recompute the contest-faithful LOCAL score from the on-disk frames.
    local = score_inflated_frames_on_disk(
        raw_path,
        num_pairs,
        archive_zip_bytes,
        targets_cache,
        batch_size=batch_size,
        device=device,
    )

    # Step 3: drift-correct -> PREDICTED contest-CPU score.
    prediction = predict_contest_cpu(
        local.score, frontier_cpu_score, bias=bias, guard=guard
    )

    # Step 4: ARM (un-fired) the paired-eval dispatch packet.
    packet = build_eval_packet(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_zip_bytes=archive_zip_bytes,
        prediction=prediction,
        candidate_id=candidate_id,
        out_dir=out_dir,
    )

    record = {
        "schema": "capstone_closure_pipeline_record.v1",
        "candidate_id": candidate_id,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_zip_bytes": archive_zip_bytes,
        "config_carrier": inflate_meta["config"].get("carrier"),
        "config_decoder_dtype": inflate_meta["config"].get("decoder_dtype"),
        "config_base_channels": inflate_meta["config"].get("base_channels"),
        "num_pairs": num_pairs,
        "inflate": inflate_meta | {"raw_path": str(raw_path)},
        "contest_faithful_local_score": local.as_dict(),
        "predicted_contest_cpu": prediction.as_dict(),
        "eval_packet": packet,
        "provenance": {
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "device": device,
            "no_mps": True,
            "scorer": "torch-CPU bridge (frozen DistortionNet)",
        },
        "wall_s": time.time() - t0,
    }
    (out_dir / "closure_record.json").write_text(json.dumps(record, indent=2))
    return record


def _print_summary(record: dict[str, Any]) -> None:
    local = record["contest_faithful_local_score"]
    pred = record["predicted_contest_cpu"]
    pkt = record["eval_packet"]
    faithful = local["full_600_pair_faithful"]
    print("=" * 78)
    print(f"CAPSTONE CLOSURE PIPELINE — candidate {record['candidate_id']}")
    print(f"  archive: {record['archive_path']}")
    print(f"  sha256={record['archive_sha256'][:16]}  zip_bytes={record['archive_zip_bytes']}")
    print(
        f"  carrier={record['config_carrier']} dtype={record['config_decoder_dtype']} "
        f"base_ch={record['config_base_channels']} num_pairs={record['num_pairs']}"
        + ("" if faithful else "  [PIPELINE-VERIFICATION ONLY: num_pairs != 600, NOT a faithful score]")
    )
    print(f"  [1] numpy inflate -> {record['inflate']['frames_shape']} uint8 on disk")
    print(
        f"  [2] contest-faithful LOCAL {local['axis']}: "
        f"S={local['score_recomputed_from_components']:.6f} "
        f"(seg={local['score_seg_contribution']:.6f} "
        f"pose={local['score_pose_contribution']:.6f} "
        f"rate={local['score_rate_contribution']:.6f})"
    )
    print(
        f"  [3] {pred['axis']}: point={pred['predicted_contest_cpu_point']:.6f} "
        f"conservative={pred['predicted_contest_cpu_conservative']:.6f} "
        f"(bias={pred['bias_local_minus_contest']:.2e} guard={pred['guard_band']:.2e})"
    )
    print(
        f"      frontier_CPU={pred['frontier_cpu_score']:.6f}  "
        f"conservative_beats_frontier={pred['conservative_beats_frontier']}  "
        f"margin={pred['margin_vs_frontier']:+.6f}"
    )
    print(f"  [4] eval packet: armed={pkt['armed']}  action={pkt['recommended_action']}  est=${pkt['estimated_cost_usd']}")
    print(f"      lane-claim: {pkt['lane_claim_command']}")
    print(f"      modal-cpu : {pkt['modal_cpu_eval_command']}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, help="capstone archive.zip path")
    ap.add_argument(
        "--targets-cache",
        default="experiments/results/capstone_gt_targets_cache",
        help="GT seg/pose targets cache dir (built once, reused)",
    )
    ap.add_argument(
        "--frontier-cpu-score",
        type=float,
        default=0.191099824,
        help="current contest-CPU frontier (the submit-rule threshold)",
    )
    ap.add_argument("--out-dir", required=True, help="closure output dir")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="torch scorer device; NEVER mps")
    ap.add_argument("--candidate-id", default="")
    ap.add_argument("--bias", type=float, default=DEFAULT_BIAS_LOCAL_MINUS_CONTEST)
    ap.add_argument("--guard", type=float, default=DEFAULT_GUARD_BAND)
    args = ap.parse_args(argv)

    if args.device == "mps":
        print("FATAL: MPS is NEVER a valid scorer axis (CLAUDE.md non-negotiable).", file=sys.stderr)
        return 2

    record = run_closure_pipeline(
        archive_path=Path(args.archive),
        targets_cache=Path(args.targets_cache),
        frontier_cpu_score=args.frontier_cpu_score,
        out_dir=Path(args.out_dir),
        batch_size=args.batch_size,
        device=args.device,
        candidate_id=args.candidate_id,
        bias=args.bias,
        guard=args.guard,
    )
    _print_summary(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
