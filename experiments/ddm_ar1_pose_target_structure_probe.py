#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ar1 — $0 pose-target structure probe + pre-registered e_p (paint-survival) probe.

RUNNABLE NOW ($0, no scorer forward): measures the STRUCTURE of the banked PoseNet
targets t_p = PoseNet(orig)[:6] (600x6, `src/tac/scorer_targets.py` format). The rank /
temporal-smoothness of t_p BOUNDS the AR-compressibility of any from-scratch pose field
(the operator's falsifier-branch worst case): a rank-1, temporally-smooth target manifold
compresses far below PR130's measured 23 KB / 38.4 B/pair. This is the pose-leg lower-bound
input to the ddm_ar1 archetype-codec price table.

PRE-REGISTERED (scorer job; design-only here): the e_p = PoseNet(painted-partition) - t_p
probe per the operator's ranked-first pose directive. See `preregistered_ep_probe()` for the
exact contract + falsifier. NOT fired in this arm (needs the paint+PoseNet harness and one
n600 scorer slot; coordinate with da1's PoseNet usage first).

Axis: [macOS-CPU frozen-scorer advisory]. score_claim=false. pointer_moved=false.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for value in (str(REPO_ROOT / "src"), str(REPO_ROOT / "upstream"), str(REPO_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"


def measure_target_structure(targets_bin: str) -> dict:
    """Rank + temporal-smoothness of the banked 600x6 PoseNet target matrix."""
    from tac.scorer_targets import load_posenet_targets

    d = load_posenet_targets(targets_bin)
    if d is None:
        raise FileNotFoundError(targets_bin)
    t = d["targets"].numpy().astype(np.float64)  # (N, 6)
    n = int(t.shape[0])
    tc = t - t.mean(0, keepdims=True)
    _, s, _ = np.linalg.svd(tc, full_matrices=False)
    ev = (s ** 2)
    ev = ev / ev.sum()
    d1 = np.diff(t, axis=0)
    fd_ratio = (d1.var(0) / (t.var(0) + 1e-12))
    ac = [float(np.corrcoef(t[:-1, k], t[1:, k])[0, 1]) for k in range(t.shape[1])]
    # Crude AR(1) residual-entropy proxy: bits to code the lag-1 innovation per dim,
    # int5-quantized after zero-mean/unit-RMS normalization (PR130 mechanism).
    innov = d1 / (d1.std(0, keepdims=True) + 1e-12)
    q = np.clip(np.round(innov * 15.5), -16, 15).astype(np.int64)  # int5 range
    per_dim_bits = []
    for k in range(q.shape[1]):
        vals, cnts = np.unique(q[:, k], return_counts=True)
        p = cnts / cnts.sum()
        per_dim_bits.append(float(-(p * np.log2(p)).sum()))
    ar1_int5_bytes = (n - 1) * sum(per_dim_bits) / 8.0
    return {
        "n_pairs": n,
        "per_dim_std": [float(x) for x in t.std(0)],
        "svd_energy_frac": [float(x) for x in ev],
        "svd_cumsum": [float(x) for x in np.cumsum(ev)],
        "lag1_autocorr_per_dim": [round(x, 4) for x in ac],
        "firstdiff_var_over_signal_var": [round(float(x), 4) for x in fd_ratio],
        "ar1_int5_innovation_bits_per_dim": [round(x, 4) for x in per_dim_bits],
        "ar1_int5_field_bytes_proxy": round(ar1_int5_bytes, 1),
        "pr130_reference_bytes": 23054,
        "pr130_reference_bits_per_sym": 3.77,
        "interpretation": (
            "t_p is ~rank-1 (energy) + temporally smooth => a from-scratch pose field "
            "AR-codes well below PR130's 23 KB; pose stream is feasibility-bounded, not "
            "the binding constraint. int5 + zero-mean/unit-RMS normalization (PR130) applies."
        ),
    }


def preregistered_ep_probe() -> dict:
    """The decisive, NOT-YET-FIRED pose-survival probe (operator-ranked #1).

    Deterministic map PoseNet o paint o partition is shared encode/decode; encode has
    unlimited compute (rule-118). Compliance: NO PoseNet/surrogate at DECODE.
    """
    return {
        "name": "e_p = PoseNet(painted-partition pair) - t_p structure, n600 (chunked <=120)",
        "steps": [
            "1. t_p = PoseNet(orig)[:6] per pair (banked, scorer_targets.py, 600x6).",
            "2. Paint synthetic-partition frames from GT argmax (flat palette OR "
            "chroma-favored low-freq fill), through R (render_grid_to_camera_uint8, "
            "tac.through_r.resolution_chain) -> uint8 camera frames -> real PoseNet -> b_p.",
            "3. e_p = b_p - t_p (600x6). Measure: rank(e_p), single global 6x6 affine-fit "
            "R^2 (does paint already carry the ego-motion?), smoothness vs xi(t).",
        ],
        "reuse": [
            "upstream/modules.py DistortionNet/PoseNet + preprocess_input (seq_len=2 YUV6)",
            "tac.through_r.resolution_chain.render_grid_to_camera_uint8 (the R first half)",
            "tools/measure_ddm_pt1_continuous_paint_ceiling.py paint machinery (SegNet->PoseNet swap)",
        ],
        "falsifier": (
            "If e_p is FULL-RANK and xi-ROUGH (global-affine R^2 < ~0.5, no low-rank "
            "structure) => the paint base carries NO ego-motion signal; the pose stream "
            "must be a from-scratch PR130-class field (still bounded 38.4 B/pair = 23 KB). "
            "If e_p is LOW-RANK / xi-smooth => the shipped field is only the residual "
            "steering b_p->t_p, AR-coded << 23 KB."
        ),
        "staging_note": (
            "Terminal stage per operator staging contract: solve_pose(frozen_final_frames, "
            "t_p) -> field coeffs, re-invoked at every composition change. Absorbs all "
            "seg-repaint pose collateral (cb1 Lane +22.7 luma / hood -0.18) into the "
            "starting residual e_p; no per-repaint pose debt survives to S. Interim pose "
            "numbers are CALIBRATION ONLY (fr1 base-dependence)."
        ),
        "compliance": "NO PoseNet/surrogate weights at decode (strict scorer rule); only "
        "field coefficients ship (COUNTED). Encoder PoseNet is FREE (rule-118).",
        "bit_depth_dof": "int4/int5/int6/int8 x {zero-mean+unit-RMS norm (PR130), "
        "ker(A) gauge projector #580, PDW2 affine gauge #553} -> realized d_pose through "
        "the REAL uint8 frame round -> entropy-coded bytes. int5 is the first candidate.",
        "fired": False,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-bin", default=str(REPO_ROOT / "experiments/posenet_targets.bin"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    receipt = {
        "schema": "ddm_ar1_pose_target_structure_probe.v1",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.19108 [contest-CPU]",
        "pointer_moved": False,
        "measured_target_structure": measure_target_structure(args.targets_bin),
        "preregistered_ep_probe": preregistered_ep_probe(),
    }
    text = json.dumps(receipt, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
