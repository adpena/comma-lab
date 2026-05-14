# SPDX-License-Identifier: MIT
"""Unit tests for tac.pose_from_embedding (Lane M-V3 Path A).

The module ships a 2-layer MLP that distills `(PoseNet 12-dim embedding,
mask features)` → `(6-DOF renderer pose)` at compress time, then runs
mask-only inflate (zero embedding input). These tests pin:

  * Module-level constants (sentinel filename + dim ints).
  * MaskFeatureExtractor input/output shapes (pair-of-class-indices →
    fixed-size feature vector).
  * PoseFromEmbeddingMLP forward signature and shape correctness.
  * predict_poses_from_masks() inflate-side helper accepts the (N, H, W)
    long tensor that submissions/robust_current/inflate_renderer.py
    decodes from masks.mkv.
  * save_mlp() / load_mlp() round-trip preserves outputs bit-equivalently
    (within FP16 quantization noise) and rejects mismatched sentinels +
    format versions (catches a future archive corruption).
  * The MLP is small enough to ship (< 30 KB FP16 ceiling — the rate-
    savings premise vs the 15 KB optimized_poses.pt).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.pose_from_embedding import (
    MASK_FEATURE_DIM,
    POSE_FROM_EMBEDDING_SENTINEL,
    POSE_FROM_EMBEDDING_WEIGHTS_FILENAME,
    POSE_OUTPUT_DIM,
    POSENET_EMBEDDING_DIM,
    MaskFeatureExtractor,
    PoseFromEmbeddingMLP,
    extract_mask_features,
    load_mlp,
    save_mlp,
)


# ── Module constants pin ────────────────────────────────────────────────


def test_sentinel_filename_pinned():
    """A future rename forces explicit acknowledgement (build + inflate +
    provenance JSON all need updates in lockstep)."""
    assert POSE_FROM_EMBEDDING_SENTINEL == "pose_from_embedding_v1"
    assert POSE_FROM_EMBEDDING_WEIGHTS_FILENAME == "pose_from_embedding_v1.pt"


def test_dim_constants_match_scorer_contract():
    """PoseNet emits 12-dim head output; renderer FiLM consumes 6-DOF.
    MASK_FEATURE_DIM is sized for the < 15 KB FP16 archive budget."""
    assert POSENET_EMBEDDING_DIM == 12
    assert POSE_OUTPUT_DIM == 6
    assert MASK_FEATURE_DIM == 16


# ── MaskFeatureExtractor ────────────────────────────────────────────────


def test_mask_feature_extractor_output_shape():
    fe = MaskFeatureExtractor()
    # (B, 2 * n_classes, H, W) one-hot pair (n_classes=5 → 10 channels)
    x = torch.randn(3, 10, 384, 512)
    y = fe(x)
    assert y.shape == (3, MASK_FEATURE_DIM)


def test_mask_feature_extractor_param_count_under_budget():
    """The feature extractor is the bulk of the MLP. Sized for < 15 KB
    FP16 archive cost (vs the 15 KB optimized_poses.pt it replaces)."""
    fe = MaskFeatureExtractor()
    n_params = sum(p.numel() for p in fe.parameters())
    # 10*8*9+8 + 8*16*9+16 + 16*16*9+16 ≈ 4082 params → ~8 KB FP16.
    assert n_params < 6_000, (
        f"feature extractor has {n_params} params; budget is < 6k so "
        f"the full MLP archive cost stays under 15 KB FP16 (Lane M-V3 "
        f"rate-savings premise)"
    )


# ── extract_mask_features helper ────────────────────────────────────────


def test_extract_mask_features_from_interleaved_masks():
    """Inflate-side path: (N, H, W) long-tensor decoded from masks.mkv."""
    fe = MaskFeatureExtractor()
    masks = torch.zeros(8, 384, 512, dtype=torch.long)  # 4 pairs
    feats = extract_mask_features(masks, fe)
    assert feats.shape == (4, MASK_FEATURE_DIM)


def test_extract_mask_features_from_paired_masks():
    """Compress-time path: (P, 2, H, W) pre-paired."""
    fe = MaskFeatureExtractor()
    paired = torch.zeros(5, 2, 384, 512, dtype=torch.long)
    feats = extract_mask_features(paired, fe)
    assert feats.shape == (5, MASK_FEATURE_DIM)


def test_extract_mask_features_rejects_odd_n_frames():
    fe = MaskFeatureExtractor()
    masks = torch.zeros(7, 384, 512, dtype=torch.long)
    with pytest.raises(ValueError, match="even N"):
        extract_mask_features(masks, fe)


def test_extract_mask_features_rejects_bad_shape():
    fe = MaskFeatureExtractor()
    bad = torch.zeros(4, 5, 6, 7, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="must be"):
        extract_mask_features(bad, fe)


def test_extract_mask_features_clamps_class_overflow():
    """Real masks may have a stray pixel value > n_classes-1 from the
    AV1 codec quantization step. Extract must clamp instead of crashing."""
    fe = MaskFeatureExtractor()
    masks = torch.zeros(2, 384, 512, dtype=torch.long)
    masks[0, 0, 0] = 99  # would crash one_hot if not clamped
    feats = extract_mask_features(masks, fe)
    assert feats.shape == (1, MASK_FEATURE_DIM)


# ── PoseFromEmbeddingMLP forward ────────────────────────────────────────


def test_mlp_forward_shape():
    mlp = PoseFromEmbeddingMLP()
    emb = torch.randn(4, POSENET_EMBEDDING_DIM)
    feats = torch.randn(4, MASK_FEATURE_DIM)
    out = mlp.forward(emb, feats)
    assert out.shape == (4, POSE_OUTPUT_DIM)


def test_mlp_forward_rejects_wrong_embedding_dim():
    mlp = PoseFromEmbeddingMLP()
    bad_emb = torch.randn(4, 8)
    feats = torch.randn(4, MASK_FEATURE_DIM)
    with pytest.raises(ValueError, match="embedding dim mismatch"):
        mlp.forward(bad_emb, feats)


def test_mlp_forward_rejects_wrong_mask_feature_dim():
    mlp = PoseFromEmbeddingMLP()
    emb = torch.randn(4, POSENET_EMBEDDING_DIM)
    bad_feats = torch.randn(4, 32)
    with pytest.raises(ValueError, match="mask_features dim mismatch"):
        mlp.forward(emb, bad_feats)


# ── predict_poses_from_masks (inflate-side helper) ──────────────────────


def test_predict_poses_from_masks_with_zero_embedding():
    """Inflate-side regime: PoseNet not loaded → embedding is None,
    helper substitutes zeros automatically."""
    mlp = PoseFromEmbeddingMLP()
    masks = torch.zeros(8, 384, 512, dtype=torch.long)  # 4 pairs
    poses = mlp.predict_poses_from_masks(masks)
    assert poses.shape == (4, POSE_OUTPUT_DIM)


def test_predict_poses_from_masks_with_supplied_embedding():
    """Compress-time-style call where the operator HAS the PoseNet head
    output to pass in."""
    mlp = PoseFromEmbeddingMLP()
    masks = torch.zeros(8, 384, 512, dtype=torch.long)  # 4 pairs
    emb = torch.randn(4, POSENET_EMBEDDING_DIM)
    poses = mlp.predict_poses_from_masks(masks, embedding=emb)
    assert poses.shape == (4, POSE_OUTPUT_DIM)


def test_predict_poses_from_masks_rejects_pair_mismatch():
    mlp = PoseFromEmbeddingMLP()
    masks = torch.zeros(8, 384, 512, dtype=torch.long)
    bad_emb = torch.randn(7, POSENET_EMBEDDING_DIM)  # 7 != 4 pairs
    with pytest.raises(ValueError, match="mismatch from different runs"):
        mlp.predict_poses_from_masks(masks, embedding=bad_emb)


# ── save_mlp / load_mlp round-trip ─────────────────────────────────────


def test_save_load_round_trip_preserves_outputs(tmp_path: Path):
    mlp = PoseFromEmbeddingMLP()
    masks = torch.zeros(8, 384, 512, dtype=torch.long)
    masks[0, 100:110, 200:220] = 1  # add some signal
    emb = torch.zeros(4, POSENET_EMBEDDING_DIM)

    mlp.eval()
    with torch.no_grad():
        out_before = mlp.predict_poses_from_masks(masks, embedding=emb)

    out_path = tmp_path / "mlp.pt"
    n_bytes = save_mlp(mlp, out_path, fp16=True)
    assert out_path.exists()
    assert n_bytes == out_path.stat().st_size

    mlp2 = load_mlp(out_path)
    mlp2.eval()
    with torch.no_grad():
        out_after = mlp2.predict_poses_from_masks(masks, embedding=emb)

    # FP16 quantization tolerance: each weight is rounded to 11 mantissa
    # bits → max 1e-3 relative error on individual outputs.
    assert torch.allclose(out_before, out_after, rtol=5e-3, atol=5e-3), (
        f"FP16 round-trip drifted beyond tolerance: max_abs_err="
        f"{(out_before - out_after).abs().max().item():.6f}"
    )


def test_saved_mlp_under_archive_budget(tmp_path: Path):
    """Lane M-V3 rate-savings premise: the archive cost must be < 15 KB
    FP16 (vs the ~15 KB optimized_poses.pt it replaces). Anything above
    this and Lane M-V3 has NO rate advantage and must justify itself
    purely on distortion improvement — a much harder bar."""
    mlp = PoseFromEmbeddingMLP()
    out_path = tmp_path / "mlp.pt"
    n_bytes = save_mlp(mlp, out_path, fp16=True)
    assert n_bytes < 15_000, (
        f"saved MLP is {n_bytes} bytes; Lane M-V3 archive-cost premise "
        f"requires < 15 KB FP16 to net rate-savings vs optimized_poses.pt."
    )


def test_load_mlp_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_mlp(tmp_path / "does_not_exist.pt")


def test_load_mlp_rejects_wrong_sentinel(tmp_path: Path):
    """A payload with the wrong sentinel field is a corrupted archive
    (or a mis-named file). Refuse to load instead of silently using a
    stranger MLP."""
    bad_path = tmp_path / "wrong_sentinel.pt"
    payload = {
        "format_version": 1,
        "sentinel": "wrong_name_v1",
        "embedding_dim": POSENET_EMBEDDING_DIM,
        "mask_feature_dim": MASK_FEATURE_DIM,
        "hidden_dim": 32,
        "pose_dim": POSE_OUTPUT_DIM,
        "n_classes": 5,
        "fp16": True,
        "state_dict": {},
    }
    torch.save(payload, str(bad_path))
    with pytest.raises(ValueError, match="sentinel mismatch"):
        load_mlp(bad_path)


def test_load_mlp_rejects_unknown_format_version(tmp_path: Path):
    """A payload with format_version != 1 is from a future schema we
    don't understand — refuse to load instead of guessing."""
    bad_path = tmp_path / "future_version.pt"
    payload = {
        "format_version": 99,
        "sentinel": POSE_FROM_EMBEDDING_SENTINEL,
        "embedding_dim": POSENET_EMBEDDING_DIM,
        "mask_feature_dim": MASK_FEATURE_DIM,
        "hidden_dim": 32,
        "pose_dim": POSE_OUTPUT_DIM,
        "n_classes": 5,
        "fp16": True,
        "state_dict": {},
    }
    torch.save(payload, str(bad_path))
    with pytest.raises(ValueError, match="unsupported format_version"):
        load_mlp(bad_path)


def test_load_mlp_rejects_non_dict_payload(tmp_path: Path):
    """A bare state_dict (without the metadata wrapper) should be rejected
    so a future refactor to torch.save(state_dict) is caught loudly."""
    bad_path = tmp_path / "bare_state.pt"
    torch.save(torch.zeros(3), str(bad_path))
    with pytest.raises(ValueError, match="does not contain"):
        load_mlp(bad_path)
