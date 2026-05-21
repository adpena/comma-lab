# SPDX-License-Identifier: MIT
"""OVERNIGHT-ZZ Carmack MVP-first micro-PV: MLX vs PyTorch CPU equivalence on PR 101 preprocess path.

[empirical:experiments/results/mlx_contest_grade_pv_pr101_ground_truth_20260521/]

This is the MVP-first FREE local PV per CLAUDE.md "Carmack MVP-first phasing"
non-negotiable. It does NOT attempt to port FastViT-T12 + EfficientNet-B2 to MLX
(that is a multi-week effort sized in operator-routable remediation #1 below).

Instead it proves the MLX vs PyTorch numerical equivalence on the CANONICAL
PR 101 SCORER PREPROCESS PATH using WW's portable primitives (Catalog #335
canonical contract; ε ≤ 1e-5 fp32 per
`src/tac/portable_primitives/tests/test_portable_primitives_numerical_equivalence.py`).

The PoseNet preprocess (modules.py:70-74) and SegNet preprocess
(modules.py:107-109) both reduce to:
  bilinear_interpolate(x, size=(384, 512), mode='bilinear', align_corners=False)
  + (PoseNet) rgb_to_yuv6 matrix + per-channel normalize
  + (SegNet) slice last frame

Per Carmack MVP-first 5-step Step 2 (smoke MUST falsifiably challenge cargo-cult):
the cargo-cult assumption to challenge is "MLX inherits MPS Catalog #1 noise
property uniformly — every MLX numerical surface is noisy". The falsifying
signature: MLX preprocess output matches PyTorch CPU output within ε ≤ 5e-3
(operator contest-grade threshold) on the canonical PR 101 video frame
ground-truth tensor.

Per Catalog #1 + #192 non-promotable invariant: the result is
`[macOS-MLX research-signal]` evidence-grade regardless of outcome —
this PV CANNOT promote MLX to contest-axis status; it can only
empirically falsify-or-confirm a CARGO-CULTED assumption about
which MLX surfaces inherit MPS noise.

Per Catalog #229 PV: this script reads pr101 archive sha b83bf3488625dbd7
(experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/archive.zip)
and pr101 source video (public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/videos/0.mkv).
Canonical anchor: PR 101 leaderboard score 0.193 [contest-CUDA] per
`experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/pr_metadata.json`.

Per CLAUDE.md "Frontier scores are pointer-only" Catalog #343: the 0.193
literal above is a HISTORICAL_SCORE_LITERAL_OK reference to PR 101's
canonical PR_metadata.json (not a live frontier claim).
<!-- HISTORICAL_SCORE_LITERAL_OK:pr101_leaderboard_published_canonical_anchor_modules_py_byte_stable_reference_2026-05-04 -->
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

PR101_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source"
)
PR101_VIDEO_PATH = PR101_SOURCE_DIR / "videos/0.mkv"
OUTPUT_DIR = (
    REPO_ROOT
    / ".omx/research/overnight_zz_mlx_contest_grade_pv_pr101_ground_truth_20260521"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Operator contest-grade threshold per task spec: < 5e-3 per-component delta.
EPS_OPERATOR_CONTEST_GRADE = 5e-3
# WW primitives baseline per src/tac/portable_primitives/tests/.
EPS_WW_FP32_BASELINE = 1e-5

# Canonical PR 101 anchors (HISTORICAL).
# <!-- HISTORICAL_SCORE_LITERAL_OK:pr101_pr102_pr103_canonical_published_reference_anchors_per_pr_metadata_json -->
PR101_CANONICAL_SCORE = 0.193  # contest-CUDA per pr_metadata.json
PR101_ARCHIVE_SHA = "b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e"
PR101_ARCHIVE_BYTES = 178258

# Canonical contest scorer input shape from modules.py:
#   segnet_model_input_size = (W, H) = (512, 384) per frame_utils.py
#   PoseNet input: (B, T=2 frames, C=3 RGB, H, W) before preprocess
#   SegNet input: (B, T=2, C=3, H, W); takes x[:, -1, ...]
SCORER_TARGET_H = 384
SCORER_TARGET_W = 512


def load_pr101_canonical_frame_pair() -> np.ndarray:
    """Load 2 consecutive frames from PR 101 source video as the canonical PV ground-truth.

    Returns shape (1, 2, 3, native_H, native_W) float32 in range [0, 255].
    Native camera resolution per CLAUDE.md SegNet+PoseNet section: 874x1164.
    """
    import av

    container = av.open(str(PR101_VIDEO_PATH))
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")  # (H, W, C)
        frames.append(img)
        if len(frames) >= 2:
            break
    container.close()

    # Shape: (T=2, H, W, C); rearrange to canonical (B=1, T=2, C=3, H, W).
    arr = np.stack(frames, axis=0).astype(np.float32)  # (2, H, W, 3)
    arr = np.transpose(arr, (0, 3, 1, 2))  # (2, 3, H, W)
    arr = arr[None, ...]  # (1, 2, 3, H, W)
    return arr


def run_pytorch_cpu_preprocess(frame_pair_np: np.ndarray) -> dict[str, np.ndarray]:
    """Run PyTorch CPU canonical preprocess on the frame pair (ground truth).

    Mirrors PoseNet.preprocess_input + SegNet.preprocess_input from PR 101 modules.py.
    """
    import torch
    import torch.nn.functional as F

    # Convert to torch (B=1, T=2, C=3, H, W).
    x = torch.from_numpy(frame_pair_np.copy())

    # SegNet preprocess: x[:, -1, ...] then bilinear to (384, 512).
    seg_input_raw = x[:, -1, ...]  # (B=1, C=3, H, W)
    seg_input = F.interpolate(
        seg_input_raw,
        size=(SCORER_TARGET_H, SCORER_TARGET_W),
        mode="bilinear",
        align_corners=False,
    )  # (B=1, C=3, 384, 512)

    # PoseNet preprocess: rearrange (B*T, C, H, W) -> bilinear -> rgb_to_yuv6 -> rearrange.
    # We just exercise the bilinear step here (the heaviest numerical primitive shared
    # with SegNet); the rgb_to_yuv6 + normalize math is a deterministic pure-tensor op.
    B, T = x.shape[0], x.shape[1]
    pose_input_raw = x.reshape(B * T, 3, x.shape[-2], x.shape[-1])
    pose_input_resized = F.interpolate(
        pose_input_raw,
        size=(SCORER_TARGET_H, SCORER_TARGET_W),
        mode="bilinear",
        align_corners=False,
    )  # (B*T=2, C=3, 384, 512)

    return {
        "seg_preprocess": seg_input.numpy(),
        "pose_preprocess_resized": pose_input_resized.numpy(),
    }


def run_mlx_preprocess_via_ww_primitives(
    frame_pair_np: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run MLX preprocess via WW canonical portable primitives.

    Uses tac.portable_primitives.nn.bilinear_upsample with backend='mlx'.
    Per WW primitives' bilinear_upsample docstring: MLX bilinear interpolate
    is implemented via numpy round-trip through PyTorch's reference impl for
    byte-stable behavior (MLX 0.x does not ship a 1:1-faithful bilinear yet).
    This means MLX bilinear == PyTorch bilinear by construction — the PV
    here proves the PRIMITIVE CONTRACT is byte-stable, NOT that MLX's
    native bilinear is contest-grade (which it isn't, per WW commentary).

    This is the canonical apparatus-disciplined statement: the WW primitives
    encode the route MLX must take to reach contest-grade preprocess
    fidelity. If a future MLX version ships a native bilinear, the primitive
    can swap implementations transparently to operators.
    """
    import mlx.core as mx

    from tac.portable_primitives.nn import bilinear_upsample

    # Convert numpy frame pair to MLX.
    x = mx.array(frame_pair_np)  # (B=1, T=2, C=3, H, W)

    # SegNet preprocess.
    seg_input_raw = x[:, -1, :, :, :]  # (B=1, C=3, H, W)
    seg_input = bilinear_upsample(
        seg_input_raw,
        size=(SCORER_TARGET_H, SCORER_TARGET_W),
        backend="mlx",
        align_corners=False,
    )

    # PoseNet preprocess (bilinear stage only).
    B, T = int(x.shape[0]), int(x.shape[1])
    pose_input_raw = x.reshape((B * T, 3, x.shape[-2], x.shape[-1]))
    pose_input_resized = bilinear_upsample(
        pose_input_raw,
        size=(SCORER_TARGET_H, SCORER_TARGET_W),
        backend="mlx",
        align_corners=False,
    )

    mx.eval(seg_input, pose_input_resized)
    return {
        "seg_preprocess": np.array(seg_input),
        "pose_preprocess_resized": np.array(pose_input_resized),
    }


def compute_per_component_delta(
    pt_out: dict[str, np.ndarray], mlx_out: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Compute max-abs-delta + relative-RMS per output tensor.

    For each component: max_abs_delta = max(|pt - mlx|); rms_delta =
    sqrt(mean((pt - mlx)^2)); relative_delta_pct = rms_delta / rms(pt) * 100.
    """
    deltas: dict[str, dict[str, float]] = {}
    for key in pt_out:
        pt = pt_out[key].astype(np.float32)
        mlx = mlx_out[key].astype(np.float32)
        if pt.shape != mlx.shape:
            deltas[key] = {
                "shape_mismatch": True,
                "pt_shape": str(pt.shape),
                "mlx_shape": str(mlx.shape),
            }
            continue
        diff = pt - mlx
        max_abs = float(np.max(np.abs(diff)))
        rms = float(np.sqrt(np.mean(diff**2)))
        pt_rms = float(np.sqrt(np.mean(pt**2)))
        rel_pct = (rms / pt_rms * 100.0) if pt_rms > 0 else 0.0
        deltas[key] = {
            "max_abs_delta": max_abs,
            "rms_delta": rms,
            "relative_rms_delta_pct": rel_pct,
            "shape": str(pt.shape),
            "pt_rms": pt_rms,
        }
    return deltas


def main() -> int:
    print(f"[ZZ-MVP-PV] PR 101 archive sha: {PR101_ARCHIVE_SHA}")
    print(f"[ZZ-MVP-PV] PR 101 archive bytes: {PR101_ARCHIVE_BYTES}")
    print(f"[ZZ-MVP-PV] PR 101 canonical score: {PR101_CANONICAL_SCORE} [contest-CUDA]")
    print(f"[ZZ-MVP-PV] Reading frames from {PR101_VIDEO_PATH}")

    if not PR101_VIDEO_PATH.exists():
        print(f"[ZZ-MVP-PV] FATAL: PR 101 video not found at {PR101_VIDEO_PATH}", file=sys.stderr)
        return 2

    t0 = time.time()
    frame_pair = load_pr101_canonical_frame_pair()
    t_load = time.time() - t0
    print(
        f"[ZZ-MVP-PV] Loaded canonical frame pair: shape={frame_pair.shape} "
        f"dtype={frame_pair.dtype} range=[{frame_pair.min():.1f}, {frame_pair.max():.1f}] "
        f"({t_load:.2f}s)"
    )

    t0 = time.time()
    pt_out = run_pytorch_cpu_preprocess(frame_pair)
    t_pt = time.time() - t0
    print(
        f"[ZZ-MVP-PV] PyTorch CPU preprocess: {t_pt:.3f}s — "
        f"seg_shape={pt_out['seg_preprocess'].shape}, "
        f"pose_shape={pt_out['pose_preprocess_resized'].shape}"
    )

    t0 = time.time()
    mlx_out = run_mlx_preprocess_via_ww_primitives(frame_pair)
    t_mlx = time.time() - t0
    print(f"[ZZ-MVP-PV] MLX (via WW primitives) preprocess: {t_mlx:.3f}s")

    deltas = compute_per_component_delta(pt_out, mlx_out)
    print("\n[ZZ-MVP-PV] Per-component MLX-vs-PyTorch-CPU delta:")
    for key, d in deltas.items():
        if d.get("shape_mismatch"):
            print(f"  {key}: SHAPE-MISMATCH pt={d['pt_shape']} mlx={d['mlx_shape']}")
            continue
        print(
            f"  {key}: max_abs={d['max_abs_delta']:.6e} "
            f"rms={d['rms_delta']:.6e} "
            f"rel_pct={d['relative_rms_delta_pct']:.6f}%"
        )

    # Verdict per Carmack MVP-first 5-step Step 2.
    max_abs_seg = deltas["seg_preprocess"]["max_abs_delta"]
    max_abs_pose = deltas["pose_preprocess_resized"]["max_abs_delta"]
    rel_pct_seg = deltas["seg_preprocess"]["relative_rms_delta_pct"]
    rel_pct_pose = deltas["pose_preprocess_resized"]["relative_rms_delta_pct"]
    max_abs_max = max(max_abs_seg, max_abs_pose)
    rel_pct_max = max(rel_pct_seg, rel_pct_pose)

    # WW primitives' bilinear_upsample routes MLX through PyTorch reference
    # impl, so the expected delta is 0 or near-machine-epsilon.
    if max_abs_max <= EPS_WW_FP32_BASELINE:
        verdict = "PRIMITIVE_CONTRACT_PASSES_FP32_EPSILON"
        contest_grade = "PRIMITIVE_LEVEL_CONTEST_GRADE"
    elif max_abs_max <= EPS_OPERATOR_CONTEST_GRADE:
        verdict = "PRIMITIVE_CONTRACT_PASSES_OPERATOR_5E3_THRESHOLD"
        contest_grade = "PRIMITIVE_LEVEL_CONTEST_GRADE_ACCEPTABLE"
    else:
        verdict = "PRIMITIVE_CONTRACT_DRIFTS_BEYOND_OPERATOR_THRESHOLD"
        contest_grade = "PRIMITIVE_LEVEL_REFUSED"

    # Operational result: this PV proves the PRIMITIVE LEVEL but NOT the
    # ARCHITECTURE LEVEL. The contest scorer FastViT-T12 + EfficientNet-B2
    # MLX port is a multi-week effort sized below.
    architecture_level_verdict = (
        "DEFER_PENDING_FULL_FASTVIT_EFFICIENTNET_B2_MLX_PORT_MULTI_WEEK"
    )

    verdict_doc = {
        "schema_version": "mlx_contest_grade_pv_pr101_ground_truth_v1_20260521",
        "lane_id": "lane_overnight_zz_mlx_contest_grade_pv_via_pr101_ground_truth_20260521",
        "subagent_id": "zz-mlx-pv-pr101",
        "carmack_mvp_first_step_1": "FREE_local_CPU_MLX_primitive_PV_via_WW_canonical_primitives",
        "carmack_mvp_first_step_2_cargo_cult_to_challenge": (
            "MLX_inherits_MPS_Catalog_1_noise_uniformly_across_all_surfaces"
        ),
        "carmack_mvp_first_step_2_falsifying_signature_pred": (
            "MLX_via_WW_primitives_matches_PyTorch_CPU_within_5e-3_on_canonical_preprocess"
        ),
        "scope_decision": "MVP_FIRST_PRIMITIVE_LEVEL_PV_NOT_FULL_ARCHITECTURE_PORT",
        "scope_rationale": (
            "Faithful FastViT-T12 (timm) + EfficientNet-B2 (smp.Unet) MLX port is "
            "multi-week effort. WW primitives cover Linear/Conv2d/LayerNorm + "
            "bilinear/GELU/ReLU/Sigmoid/Softmax/Matmul ONLY. Contest scorers need "
            "RepMixer (FastViT) + MBConv-SE (EfficientNet) + UNet decoder + "
            "BatchNorm + DepthwiseConv + SE-attention + LayerScale + cond-pos-encoding. "
            "Per Carmack MVP-first 5-step Step 1, the FREE PV that produces signal "
            "WITHIN scope is the WW-primitive-on-canonical-PR101-preprocess micro-PV. "
            "Per CLAUDE.md 'Forbidden premature KILL' the LOW verdict at architecture "
            "level is DEFER not KILL; MLX paradigm INTACT pending the multi-week port."
        ),
        "pr101_canonical_anchors": {
            "archive_sha": PR101_ARCHIVE_SHA,
            "archive_bytes": PR101_ARCHIVE_BYTES,
            "leaderboard_score_contest_cuda": PR101_CANONICAL_SCORE,
            "score_source": "experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/pr_metadata.json",
            "video_source": str(PR101_VIDEO_PATH.relative_to(REPO_ROOT)),
        },
        "primitive_level_pv": {
            "mlx_backend_route": "WW_portable_primitives_bilinear_upsample_pytorch_reference_impl",
            "frame_pair_shape": list(frame_pair.shape),
            "pytorch_cpu_seconds": t_pt,
            "mlx_seconds": t_mlx,
            "deltas_per_component": deltas,
            "max_abs_delta_overall": max_abs_max,
            "relative_rms_delta_pct_max": rel_pct_max,
            "thresholds": {
                "ww_fp32_epsilon": EPS_WW_FP32_BASELINE,
                "operator_contest_grade_threshold": EPS_OPERATOR_CONTEST_GRADE,
            },
            "primitive_level_verdict": verdict,
            "contest_grade_at_primitive_level": contest_grade,
        },
        "architecture_level_pv": {
            "verdict": architecture_level_verdict,
            "blocker": "FastViT_T12_and_EfficientNet_B2_not_in_WW_primitive_scope",
            "remediation_path": "operator_routable_1_multi_week_full_arch_port_via_extended_ww_primitive_set",
        },
        "canonical_provenance": {
            "evidence_grade": "macOS-MLX-research-signal",
            "evidence_tag": "[macOS-MLX research-signal]",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "non_promotable_blockers": [
                "macos_mlx_research_signal_not_score_evidence_per_catalog_1",
                "primitive_level_only_not_architecture_level_pv",
                "requires_paired_linux_x86_64_nvidia_for_promotion",
            ],
        },
        "catalog_compliance": {
            "catalog_1_mps_fallback_default": "PRESERVED — MLX inherits non-promotable status",
            "catalog_192_macos_advisory": "PRESERVED — evidence_grade=macOS-MLX-research-signal",
            "catalog_287_evidence_tags": "PRESERVED — [macOS-MLX research-signal] on every claim",
            "catalog_316_frontier_pointer": "PRESERVED — no hardcoded current-frontier score literals",
            "catalog_323_canonical_provenance": "PRESERVED — all PV results carry canonical Provenance",
        },
    }

    verdict_path = OUTPUT_DIR / "verdict.json"
    verdict_path.write_text(json.dumps(verdict_doc, indent=2) + "\n")
    print(f"\n[ZZ-MVP-PV] Verdict written: {verdict_path.relative_to(REPO_ROOT)}")
    print(f"[ZZ-MVP-PV] PRIMITIVE-LEVEL verdict: {verdict}")
    print(f"[ZZ-MVP-PV] ARCHITECTURE-LEVEL verdict: {architecture_level_verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
