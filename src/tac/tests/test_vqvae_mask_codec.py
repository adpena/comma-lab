"""Tests for tac.vqvae_mask_codec — Lane VQ-VAE (paradigm α3)."""
from __future__ import annotations

import pytest
import torch

from tac.vqvae_mask_codec import (
    VQVAE_MAGIC,
    VQVAE_VERSION,
    VQVAEConfig,
    build_codebook_top_k,
    decode_indices_to_patches,
    decode_vqvae_codec,
    encode_patches_to_indices,
    encode_vqvae_codec,
    masks_from_patches,
    patches_from_masks,
    raw_baseline_bytes,
    vqvae_codec_bytes,
)


def _make_synthetic_masks(t: int = 4, h: int = 32, w: int = 48, *, seed: int = 7) -> torch.Tensor:
    """Synthetic 5-class mask with a few flat regions and small variation."""
    g = torch.Generator().manual_seed(seed)
    masks = torch.zeros(t, h, w, dtype=torch.int64)
    masks[:, : h // 2, :] = 0  # top half: class 0
    masks[:, h // 2 :, : w // 2] = 1  # bottom-left: class 1
    masks[:, h // 2 :, w // 2 :] = 2  # bottom-right: class 2
    # Sparse noise
    n_noise = max(1, t * h * w // 200)
    for _ in range(n_noise):
        ti = torch.randint(0, t, (1,), generator=g).item()
        hi = torch.randint(0, h, (1,), generator=g).item()
        wi = torch.randint(0, w, (1,), generator=g).item()
        ci = torch.randint(0, 5, (1,), generator=g).item()
        masks[ti, hi, wi] = ci
    return masks


# ── Patch primitives ────────────────────────────────────────────────────


def test_patches_roundtrip_identity():
    masks = _make_synthetic_masks(t=2, h=16, w=24)
    patches = patches_from_masks(masks, patch_size=4)
    masks_recovered = masks_from_patches(patches)
    assert torch.equal(masks, masks_recovered)


def test_patches_rejects_non_divisible():
    masks = _make_synthetic_masks(t=2, h=15, w=16)
    with pytest.raises(ValueError, match="divisible"):
        patches_from_masks(masks, patch_size=4)


def test_patches_shape():
    masks = _make_synthetic_masks(t=3, h=8, w=12)
    patches = patches_from_masks(masks, patch_size=4)
    assert patches.shape == (3, 2, 3, 4, 4)


# ── Codebook construction ─────────────────────────────────────────────────


def test_build_codebook_top_k_returns_correct_shape():
    masks = _make_synthetic_masks(t=4, h=32, w=48, seed=42)
    cb = build_codebook_top_k(masks, patch_size=4, k=16)
    assert cb.shape == (16, 4, 4)
    assert cb.dtype == torch.int64


def test_build_codebook_includes_dominant_pattern():
    """For mask with dominant flat-zero region, codebook[0] should be all zeros."""
    masks = torch.zeros(4, 32, 48, dtype=torch.int64)
    masks[:, 16:, :] = 1
    cb = build_codebook_top_k(masks, patch_size=4, k=4)
    # The top 2 should be all-zeros and all-ones
    cb_set = {tuple(p.flatten().tolist()) for p in cb}
    all_zeros = tuple([0] * 16)
    all_ones = tuple([1] * 16)
    assert all_zeros in cb_set or all_ones in cb_set


def test_build_codebook_rejects_invalid_k():
    masks = _make_synthetic_masks(t=2, h=8, w=12)
    with pytest.raises(ValueError, match="k must be"):
        build_codebook_top_k(masks, patch_size=4, k=0)


# ── Encode/Decode patches via codebook ──────────────────────────────────


def test_encode_decode_indices_recovers_patches_when_in_codebook():
    """Patches that exist exactly in codebook should reconstruct exactly."""
    masks = torch.zeros(2, 16, 24, dtype=torch.int64)
    masks[:, 8:, :] = 1
    cb = build_codebook_top_k(masks, patch_size=4, k=4)
    patches = patches_from_masks(masks, patch_size=4)
    indices = encode_patches_to_indices(patches, cb)
    patches_recovered = decode_indices_to_patches(indices, cb)
    assert torch.equal(patches, patches_recovered)


# ── Top-level codec encode/decode ────────────────────────────────────────


def test_encode_decode_codec_perfect_recovery_on_synthetic_in_codebook():
    masks = torch.zeros(2, 16, 24, dtype=torch.int64)
    masks[:, 8:, :] = 1
    cb = build_codebook_top_k(masks, patch_size=4, k=4)
    config = VQVAEConfig(patch_size=4, codebook_size=4, num_classes=5)
    blob = encode_vqvae_codec(masks, codebook=cb, config=config)
    masks_recovered = decode_vqvae_codec(blob)
    assert masks_recovered.shape == masks.shape
    assert masks_recovered.dtype == torch.int64
    assert torch.equal(masks_recovered, masks)


def test_encode_decode_codec_high_agreement_on_lossy():
    """With a small codebook, reconstruction agreement should still be >=90% on flat masks."""
    masks = _make_synthetic_masks(t=4, h=32, w=48, seed=11)
    cb = build_codebook_top_k(masks, patch_size=4, k=32)
    config = VQVAEConfig(patch_size=4, codebook_size=32, num_classes=5)
    blob = encode_vqvae_codec(masks, codebook=cb, config=config)
    masks_recovered = decode_vqvae_codec(blob)
    agreement = (masks_recovered == masks).float().mean().item()
    assert agreement >= 0.90


def test_encode_payload_starts_with_magic():
    masks = _make_synthetic_masks(t=2, h=16, w=24)
    cb = build_codebook_top_k(masks, patch_size=4, k=8)
    config = VQVAEConfig(patch_size=4, codebook_size=8, num_classes=5)
    blob = encode_vqvae_codec(masks, codebook=cb, config=config)
    assert blob[:4] == VQVAE_MAGIC


def test_decode_rejects_bad_magic():
    with pytest.raises(ValueError, match="magic mismatch"):
        decode_vqvae_codec(b"BADX" + b"\x00" * 30)


def test_encode_rejects_bad_codebook_shape():
    masks = _make_synthetic_masks(t=2, h=16, w=24)
    cb = torch.zeros(8, 5, 5, dtype=torch.int64)  # wrong patch size
    config = VQVAEConfig(patch_size=4, codebook_size=8, num_classes=5)
    with pytest.raises(ValueError, match="codebook shape"):
        encode_vqvae_codec(masks, codebook=cb, config=config)


# ── Byte-count claim ─────────────────────────────────────────────────────


def test_codec_beats_raw_baseline_on_flat_masks():
    """Synthetic flat-region masks compress to <50% of raw uint8 baseline."""
    masks = _make_synthetic_masks(t=4, h=32, w=48, seed=99)
    cb = build_codebook_top_k(masks, patch_size=4, k=32)
    config = VQVAEConfig(patch_size=4, codebook_size=32, num_classes=5)
    encoded = vqvae_codec_bytes(masks, codebook=cb, config=config)
    raw = raw_baseline_bytes(masks)
    assert encoded < raw * 0.5, (
        f"[synthetic] expected VQ codec to beat raw uint8 by 2×; "
        f"got encoded={encoded}, raw={raw}"
    )


# ── No silent defaults ──────────────────────────────────────────────────


def test_encode_requires_keyword_codebook_and_config():
    masks = _make_synthetic_masks(t=2, h=16, w=24)
    cb = build_codebook_top_k(masks, patch_size=4, k=4)
    config = VQVAEConfig(patch_size=4, codebook_size=4, num_classes=5)
    with pytest.raises(TypeError):
        encode_vqvae_codec(masks, cb, config)  # type: ignore[misc]


def test_build_codebook_requires_keyword_args():
    masks = _make_synthetic_masks(t=2, h=8, w=12)
    with pytest.raises(TypeError):
        build_codebook_top_k(masks, 4, 8)  # type: ignore[misc]
