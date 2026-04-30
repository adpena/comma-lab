"""Lane 17 — IMPS archive tests.

Per Council design (.omx/research/council_lane_17_imp_design_20260430.md):
- Magic byte b"IMPS" Q6 10/10
- Per-tensor sparsity gate 78% (sparse-CSR breakeven + safety margin)
- numel cap 65535 (uint16 idx)
- FP16 fallback for ineligible / low-sparsity / large-numel tensors

All claims tagged [synthetic]. The real-archive empirical smoke is in the
sister test ``test_imp_real_archive_smoke.py``.

CLAUDE.md non-negotiables verified:
- No scorer load (decode is pure-math byte→tensor).
- No silent defaults (encode/decode require model + device explicitly).
- Deterministic CPU-only.
"""

from __future__ import annotations

import json
import pytest
import torch
import torch.nn as nn

from tac.imps_renderer_archive import (
    IMPS_ARCHIVE_MAGIC,
    IMPS_ARCHIVE_VERSION,
    IMPS_PER_TENSOR_SPARSITY_GATE,
    IMPS_PER_TENSOR_NUMEL_CAP,
    IMPSArchiveError,
    decode_imps_archive,
    encode_imps_archive,
    _eligible_for_sparse_csr,
)
from tac.iterative_magnitude_pruning import (
    apply_mask_to_model,
    iter_prunable_parameters,
    prune_lowest_magnitude,
)


def _build_renderer():
    """Build the canonical AsymmetricPairGenerator (Lane G v3 arch)."""
    from tac.renderer import build_renderer

    return build_renderer(
        num_classes=5,
        embed_dim=6,
        base_ch=36,
        mid_ch=60,
        motion_hidden=32,
        depth=1,
        pose_dim=6,
        use_zoom_flow=False,
        padding_mode="zeros",
    )


# ── magic-byte / format invariants ──────────────────────────────────────


def test_imps_magic_byte_is_imps_synthetic() -> None:
    """[synthetic] Magic byte is exactly b"IMPS"."""
    assert IMPS_ARCHIVE_MAGIC == b"IMPS"
    assert len(IMPS_ARCHIVE_MAGIC) == 4


def test_imps_version_is_one_synthetic() -> None:
    """[synthetic] Wire-format version starts at 1."""
    assert IMPS_ARCHIVE_VERSION == 1


def test_imps_sparsity_gate_default_above_breakeven_synthetic() -> None:
    """[synthetic] Default gate must exceed the 50% naive breakeven; we use
    78% which is the empirical sparse-CSR-vs-FP4 breakeven + safety margin.
    """
    assert IMPS_PER_TENSOR_SPARSITY_GATE >= 0.50
    assert IMPS_PER_TENSOR_SPARSITY_GATE <= 0.99


def test_imps_numel_cap_matches_uint16_synthetic() -> None:
    """[synthetic] numel cap matches the sparse-CSR codec's uint16 idx cap."""
    assert IMPS_PER_TENSOR_NUMEL_CAP == 65535


# ── eligibility logic ──────────────────────────────────────────────────


def test_eligible_for_sparse_csr_below_gate_returns_false_synthetic() -> None:
    """[synthetic] At 50% sparsity (below 78% gate), eligible returns False."""
    w = torch.randn(8, 16, 3, 3)
    mask = w.abs() > w.abs().median().item()  # 50% sparsity exactly
    assert not _eligible_for_sparse_csr(w, mask)


def test_eligible_for_sparse_csr_above_gate_returns_true_synthetic() -> None:
    """[synthetic] At 89% sparsity (target Lane 17 final), eligible returns True."""
    torch.manual_seed(0)
    w = torch.randn(8, 16, 3, 3)
    threshold = w.abs().flatten().kthvalue(int(w.numel() * 0.89)).values.item()
    mask = w.abs() > threshold
    assert _eligible_for_sparse_csr(w, mask)


def test_eligible_for_sparse_csr_no_mask_returns_false_synthetic() -> None:
    """[synthetic] mask=None → not eligible (FP16 fallback)."""
    w = torch.randn(8, 16, 3, 3)
    assert not _eligible_for_sparse_csr(w, None)


def test_eligible_for_sparse_csr_oversized_returns_false_synthetic() -> None:
    """[synthetic] numel above the uint16 cap → not eligible (FP16 fallback)."""
    # 200×200×3×3 = 360_000 weights → above cap.
    w = torch.randn(200, 200, 3, 3)
    mask = torch.zeros_like(w, dtype=torch.bool)
    assert not _eligible_for_sparse_csr(w, mask)


# ── encode/decode roundtrip ─────────────────────────────────────────────


def test_imps_encode_dense_baseline_no_masks_synthetic() -> None:
    """[synthetic] Encoding a dense renderer with no masks → archive is
    valid, all layers FP16-fallback, decode produces the same renderer.
    """
    torch.manual_seed(42)
    model = _build_renderer()
    blob = encode_imps_archive(model=model, masks={})
    assert blob[:4] == IMPS_ARCHIVE_MAGIC

    decoded = decode_imps_archive(data=blob, device="cpu")
    # All Conv2d weights should match exactly within FP16 round-trip noise.
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), decoded.named_parameters()
    ):
        assert n1 == n2
        if p1.dim() == 4:
            # FP16 round-trip: max diff is 1 LSB of FP16, scale-dependent.
            diff = (p1.detach().cpu().float() - p2.detach().cpu().float()).abs()
            assert diff.max().item() < 0.05, (
                f"{n1}: FP16 roundtrip diff {diff.max().item()} too large"
            )


def test_imps_encode_high_sparsity_uses_sparse_csr_synthetic() -> None:
    """[synthetic] Encoding with 89% sparsity masks → at least one
    layer encoded as imps_conv (the gate fires for the largest convs).
    """
    torch.manual_seed(7)
    model = _build_renderer()
    # Apply IMP-style pruning at ~89% sparsity per tensor.
    masks = {}
    for name, p in iter_prunable_parameters(model):
        if p.numel() > IMPS_PER_TENSOR_NUMEL_CAP:
            continue  # skip oversized convs (FP16 fallback)
        flat = p.detach().abs().flatten()
        threshold = flat.kthvalue(int(flat.numel() * 0.89)).values.item()
        masks[name] = (p.detach().abs() > threshold).cpu()
    apply_mask_to_model(model, masks)

    blob = encode_imps_archive(model=model, masks=masks)
    # Parse header to count imps_conv layers.
    import struct as _struct

    offset = 4
    (header_len,) = _struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    header = json.loads(blob[offset:offset + header_len].decode("utf-8"))
    n_imps = sum(1 for layer in header["layers"] if layer["kind"] == "imps_conv")
    assert n_imps >= 1, (
        f"expected at least one imps_conv layer at 89% sparsity, "
        f"got 0 (kinds: {[l['kind'] for l in header['layers']]})"
    )


def test_imps_roundtrip_preserves_pruned_zeros_synthetic() -> None:
    """[synthetic] After IMP pruning + IMPS encode/decode, pruned positions
    in the decoded model are EXACTLY zero (no FP4 quantization noise on
    pruned positions — the codec stores only survivors).
    """
    torch.manual_seed(99)
    model = _build_renderer()
    masks = {}
    for name, p in iter_prunable_parameters(model):
        if p.numel() > IMPS_PER_TENSOR_NUMEL_CAP:
            continue
        flat = p.detach().abs().flatten()
        threshold = flat.kthvalue(int(flat.numel() * 0.85)).values.item()
        masks[name] = (p.detach().abs() > threshold).cpu()
    apply_mask_to_model(model, masks)

    blob = encode_imps_archive(model=model, masks=masks)
    decoded = decode_imps_archive(data=blob, device="cpu")

    # Map decoded layer name → param tensor.
    dec_params = dict(decoded.named_parameters())
    for name, p in iter_prunable_parameters(model):
        if name not in masks:
            continue  # FP16-fallback layer; skip strict-zero check
        # Only checked for sparse-CSR-eligible tensors (small enough numel).
        if p.numel() > IMPS_PER_TENSOR_NUMEL_CAP:
            continue
        # Look up in decoded model
        if name not in dec_params:
            continue
        m = masks[name]
        sparsity = 1.0 - float(m.sum().item()) / m.numel()
        if sparsity < IMPS_PER_TENSOR_SPARSITY_GATE:
            continue  # FP16-fallback path; values may be ~0 but not exactly
        dec_p = dec_params[name].detach().cpu()
        # Pruned positions must be exactly zero (sparse-CSR doesn't store them).
        pruned_vals = dec_p[~m]
        assert torch.all(pruned_vals == 0), (
            f"{name}: {(pruned_vals != 0).sum().item()} pruned positions "
            f"are nonzero after IMPS roundtrip (sparsity={sparsity:.2f})"
        )


# ── error handling ─────────────────────────────────────────────────────


def test_decode_imps_rejects_missing_data_synthetic() -> None:
    """[synthetic] decode with data=None raises (no silent default)."""
    with pytest.raises(IMPSArchiveError, match="data is required"):
        decode_imps_archive(data=None, device="cpu")


def test_decode_imps_rejects_missing_device_synthetic() -> None:
    """[synthetic] decode with device=None raises (no silent default)."""
    blob = b"IMPS" + b"\x00" * 100  # malformed but past the data-None check
    with pytest.raises(IMPSArchiveError, match="device is required"):
        decode_imps_archive(data=blob, device=None)


def test_decode_imps_rejects_bad_magic_synthetic() -> None:
    """[synthetic] decode with wrong magic byte raises with clear message."""
    blob = b"FAKE" + b"\x00" * 100
    with pytest.raises(IMPSArchiveError, match="bad/missing magic"):
        decode_imps_archive(data=blob, device="cpu")


def test_decode_imps_rejects_truncated_blob_synthetic() -> None:
    """[synthetic] decode with too-short blob raises."""
    with pytest.raises(IMPSArchiveError, match="bad/missing magic"):
        decode_imps_archive(data=b"IMPS", device="cpu")


def test_encode_imps_rejects_none_model_synthetic() -> None:
    """[synthetic] encode with model=None raises (no silent default)."""
    with pytest.raises(IMPSArchiveError, match="model is required"):
        encode_imps_archive(model=None)


def test_decode_imps_rejects_unsupported_version_synthetic() -> None:
    """[synthetic] Future-version archive header is rejected, not silently
    interpreted at the v1 schema.
    """
    import struct as _struct

    fake_header = json.dumps({
        "version": 99,  # not yet supported
        "format": "imps_renderer_archive_v99",
        "arch": {},
        "layers": [],
        "scalar_params": {},
        "body_len": 0,
    }, separators=(",", ":")).encode("utf-8")
    blob = (
        b"IMPS"
        + _struct.pack("<I", len(fake_header))
        + fake_header
        + _struct.pack("<I", 0)
    )
    with pytest.raises(IMPSArchiveError, match="unsupported version"):
        decode_imps_archive(data=blob, device="cpu")


# ── codec magic registry sanity ──────────────────────────────────────────


def test_imps_registered_in_codec_magic_registry_synthetic() -> None:
    """[synthetic] IMPS magic is registered in tac.codec_magic_registry so
    sniffers can discover it without importing imps_renderer_archive.
    """
    from tac.codec_magic_registry import find_by_magic

    entry = find_by_magic(b"IMPS")
    assert entry is not None
    assert entry.name == "Lane 17 IMP"
    assert "imps_renderer_archive" in entry.decode_module
