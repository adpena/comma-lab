#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""VALIDATE the realized witness harness == the contest oracle (>=6 decimals).

Measurement-integrity foundation (operator paranoia 2026-06-26,
``.omx/research/reaudit_refounding_and_md_decoupling_20260626.md``). R3's harness
audit flagged that NO negative verdict should stand until the ONE trusted realized
harness is PROVEN identical to the contest oracle. This script is that proof.

THE CLAIM PROVEN HERE
---------------------
The trainer's verdict path (``cpu_verdict_d_seg`` / ``cpu_verdict_d_pose`` +
``segnet_argmax_and_margin`` + ``_cpu_pose_raw``, git 8cceaa072) computes the
EXACT same d_seg / d_pose as ``upstream/evaluate.py`` (which calls
``upstream/modules.py`` ``DistortionNet.compute_distortion`` ->
``SegNet.compute_distortion`` argmax-disagreement / ``PoseNet.compute_distortion``
pose-MSE). ``experiments/contest_auth_eval.py`` wraps ``upstream/evaluate.py``,
so verdict-path == contest_auth_eval on the recon->score mapping.

METHOD ($0, CPU-light, REAL frames — NO synthetic-fixture trap)
--------------------------------------------------------------
* Decode N REAL pairs from ``upstream/videos/0.mkv`` (the contest source).
* Build a deterministic "comp" reconstruction (perturbed frames) so d_seg / d_pose
  are NONZERO and meaningful (a constant comp would trivially agree at 0).
* ORACLE:  ``DistortionNet.compute_distortion(gt_pair, comp_pair)`` (the literal
  evaluate.py call) -> per-pair (d_pose, d_seg).
* VERDICT: the trainer's functions on the SAME frames -> per-pair (d_seg, d_pose).
* ASSERT every per-pair AND mean |oracle - verdict| <= 1e-6.
* SECONDARY: assert the trainer's REAL loader ``load_real_segnet`` produces a
  byte-identical argmax to the oracle's ``dn.segnet`` (the loader is faithful too).

Evidence tag: ``[contest-CPU advisory]`` — this is a HARNESS-EQUIVALENCE proof,
not a score claim. It validates the measurement apparatus; it does not move the
frontier. Writes the proof to ``reports/realized_harness_vs_oracle_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_comp_recon(f0: np.ndarray, f1: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic perturbation -> a non-trivial 'reconstruction' (uint8 camera).

    Small additive structured noise so the SegNet argmax flips on a real (small)
    fraction of pixels and the PoseNet pose MSE is nonzero — exercising the
    SAME math both paths must agree on.
    """

    rng = np.random.default_rng(seed)
    f0 = np.asarray(f0, dtype=np.float64)
    f1 = np.asarray(f1, dtype=np.float64)
    # +/- ~6 levels of structured noise + a mild contrast shift on f1 (flips argmax
    # at class boundaries) and a smaller shift on f0 (perturbs pose).
    n1 = rng.normal(0.0, 6.0, size=f1.shape)
    n0 = rng.normal(0.0, 3.0, size=f0.shape)
    c1 = np.clip(np.round(1.03 * f1 - 2.0 + n1), 0, 255).astype(np.uint8)
    c0 = np.clip(np.round(f0 + n0), 0, 255).astype(np.uint8)
    return c0, c1


def run(n_pairs: int, tol: float, out_path: Path) -> dict:
    import torch

    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream oracle
    from tac.boundary_math.seg_core import (
        decode_gt_frame1_pairs,
        load_real_segnet,
        segnet_argmax_and_margin,
    )

    # The trainer's ACTUAL verdict functions (the path under test).
    from train_witness_realized_through_R_mlx import (
        _cpu_pose_raw,
        cpu_verdict_d_pose,
        cpu_verdict_d_seg,
    )

    torch.manual_seed(0)

    # ----- ORACLE scorers (exactly what evaluate.py constructs) -----
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for p in dn.parameters():
        p.requires_grad = False

    # ----- the trainer's REAL loader (secondary faithfulness check) -----
    seg_cpu_trainer = load_real_segnet("cpu")

    rows: list[dict] = []
    loader_argmax_mismatch_px = 0
    loader_total_px = 0

    for idx, f0, f1 in decode_gt_frame1_pairs(n_pairs=n_pairs):
        f0 = np.asarray(f0)
        f1 = np.asarray(f1)
        c0, c1 = _make_comp_recon(f0, f1, seed=1000 + int(idx))

        gt_pair = torch.from_numpy(np.stack([f0, f1], axis=0)[None]).double().float()
        comp_pair = torch.from_numpy(np.stack([c0, c1], axis=0)[None]).double().float()
        with torch.inference_mode():
            d_pose_oracle_t, d_seg_oracle_t = dn.compute_distortion(gt_pair, comp_pair)
        d_pose_oracle = float(d_pose_oracle_t.item())
        d_seg_oracle = float(d_seg_oracle_t.item())

        # ----- VERDICT path (trainer functions), oracle scorers, SAME frames -----
        gt_argmax, _margin = segnet_argmax_and_margin(dn.segnet, f1)
        gt_argmax = np.asarray(gt_argmax).astype(np.int64)
        d_seg_verdict = cpu_verdict_d_seg(dn.segnet, c1, gt_argmax)

        gt_pose = _cpu_pose_raw(dn.posenet, f0, f1)
        d_pose_verdict = cpu_verdict_d_pose(dn.posenet, c0, c1, gt_pose)

        # ----- SECONDARY: the trainer's REAL loader produces identical GT argmax -----
        gt_argmax_trainer_loader, _ = segnet_argmax_and_margin(seg_cpu_trainer, f1)
        gt_argmax_trainer_loader = np.asarray(gt_argmax_trainer_loader).astype(np.int64)
        loader_argmax_mismatch_px += int(np.count_nonzero(gt_argmax_trainer_loader != gt_argmax))
        loader_total_px += int(gt_argmax.size)

        rows.append(
            {
                "pair_idx": int(idx),
                "d_seg_oracle": d_seg_oracle,
                "d_seg_verdict": d_seg_verdict,
                "d_seg_abs_diff": abs(d_seg_oracle - d_seg_verdict),
                "d_pose_oracle": d_pose_oracle,
                "d_pose_verdict": d_pose_verdict,
                "d_pose_abs_diff": abs(d_pose_oracle - d_pose_verdict),
            }
        )

    seg_diffs = [r["d_seg_abs_diff"] for r in rows]
    pose_diffs = [r["d_pose_abs_diff"] for r in rows]
    max_seg_diff = max(seg_diffs) if seg_diffs else 0.0
    max_pose_diff = max(pose_diffs) if pose_diffs else 0.0
    mean_seg_oracle = float(np.mean([r["d_seg_oracle"] for r in rows])) if rows else 0.0
    mean_seg_verdict = float(np.mean([r["d_seg_verdict"] for r in rows])) if rows else 0.0
    mean_pose_oracle = float(np.mean([r["d_pose_oracle"] for r in rows])) if rows else 0.0
    mean_pose_verdict = float(np.mean([r["d_pose_verdict"] for r in rows])) if rows else 0.0

    def _decimals(diff: float) -> int:
        if diff <= 0.0:
            return 17
        import math

        return max(0, int(-math.floor(math.log10(diff))))

    seg_ok = max_seg_diff <= tol
    pose_ok = max_pose_diff <= tol
    loader_ok = loader_argmax_mismatch_px == 0
    passed = bool(seg_ok and pose_ok and loader_ok and rows)

    result = {
        "schema": "realized_harness_vs_oracle_validation.v1",
        "validated_at_utc": _utc(),
        "authority": "[contest-CPU advisory] harness-equivalence proof (NOT a score claim)",
        "promotion_eligible": False,
        "score_claim": False,
        "claim": (
            "trainer verdict path (cpu_verdict_d_seg/d_pose, git 8cceaa072) == "
            "upstream/evaluate.py DistortionNet.compute_distortion (modules.py:84,112) "
            "== experiments/contest_auth_eval.py oracle, on identical uint8 camera frames"
        ),
        "n_pairs": len(rows),
        "tolerance": tol,
        "real_input_source": "upstream/videos/0.mkv (contest source; NO synthetic fixture)",
        "d_seg": {
            "max_abs_diff": max_seg_diff,
            "agreement_decimals": _decimals(max_seg_diff),
            "mean_oracle": mean_seg_oracle,
            "mean_verdict": mean_seg_verdict,
            "pass": seg_ok,
        },
        "d_pose": {
            "max_abs_diff": max_pose_diff,
            "agreement_decimals": _decimals(max_pose_diff),
            "mean_oracle": mean_pose_oracle,
            "mean_verdict": mean_pose_verdict,
            "pass": pose_ok,
        },
        "trainer_loader_faithful": {
            "argmax_mismatch_px": loader_argmax_mismatch_px,
            "total_px": loader_total_px,
            "pass": loader_ok,
        },
        "passed": passed,
        "per_pair": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-pairs", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument(
        "--out",
        type=str,
        default=str(REPO / "reports" / "realized_harness_vs_oracle_validation.json"),
    )
    args = ap.parse_args(argv)

    result = run(int(args.num_pairs), float(args.tol), Path(args.out))
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "d_seg_max_abs_diff": result["d_seg"]["max_abs_diff"],
                "d_seg_agreement_decimals": result["d_seg"]["agreement_decimals"],
                "d_pose_max_abs_diff": result["d_pose"]["max_abs_diff"],
                "d_pose_agreement_decimals": result["d_pose"]["agreement_decimals"],
                "trainer_loader_faithful": result["trainer_loader_faithful"]["pass"],
                "out": str(args.out),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
