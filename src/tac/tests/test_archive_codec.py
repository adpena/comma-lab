# SPDX-License-Identifier: MIT
"""Roundtrip tests for ``tac.archive_codec``.

Check 46 contract: any quantizer/encoder must demonstrate
``decode(encode(x)) ≈ x`` to a known tolerance. The codecs in
``archive_codec.py`` use 4-bit quantization (TextureAtomCodebook),
struct-packed FP32 (MotionFieldCodec.serialize), and run-length-coded
sparse corrections (ScorerCorrectionTargets). Each gets its own
roundtrip test.
"""

from __future__ import annotations

import io
import zipfile

import torch

from tac.archive_codec import (
    MotionFieldCodec,
    ScorerCorrectionTargets,
    TextureAtomCodebook,
    build_minimal_archive,
)


# ── TextureAtomCodebook (4-bit quantized) ──────────────────────────────────


def test_texture_atom_codebook_roundtrip_within_4bit_tolerance() -> None:
    """``deserialize(serialize(codebook))`` recovers atoms within 4-bit step.

    4-bit quantization step is ``255 / 15 ≈ 17.0``; max absolute error per
    value is ≤ 0.5 step ≈ 8.5. The serialized atoms come back rounded to
    the nearest 4-bit grid point, which is what ``orig_q`` precomputes.
    """
    torch.manual_seed(0)
    codebook = TextureAtomCodebook(num_atoms=16, atom_size=8, num_channels=3)

    blob = codebook.serialize()
    restored = TextureAtomCodebook.deserialize(blob)

    # Quantize the original atoms to the same 4-bit grid the encoder used,
    # then assert exact equality vs. the restored atoms (loss is in the
    # quantize step, not the deserialize step).
    orig_q = (
        codebook.atoms.detach().clamp(0.0, 255.0) / 255.0 * 15.0
    ).round() * (255.0 / 15.0)
    assert torch.allclose(orig_q, restored.atoms.detach(), atol=1e-5), (
        "deserialize(serialize(...)) must reproduce the 4-bit-quantized "
        "atom grid exactly; only the quantize step may lose precision."
    )

    # And against the float original, the error stays within the 4-bit
    # quantization step (~8.5 per channel value).
    err = (codebook.atoms.detach() - restored.atoms.detach()).abs().max().item()
    assert err < 17.0, f"4-bit roundtrip error {err:.3f} exceeds quantizer step"


def test_texture_atom_codebook_metadata_preserved() -> None:
    codebook = TextureAtomCodebook(num_atoms=8, atom_size=4, num_channels=3)
    restored = TextureAtomCodebook.deserialize(codebook.serialize())
    assert restored.num_atoms == codebook.num_atoms
    assert restored.atom_size == codebook.atom_size
    assert restored.num_channels == codebook.num_channels


# ── MotionFieldCodec (struct-packed FP32) ──────────────────────────────────


def test_motion_field_codec_roundtrip_within_fp16_tolerance() -> None:
    """``deserialize(serialize(params))`` recovers params within FP16 tolerance.

    MotionFieldCodec serializes affine params as FP16 (per the implementation
    in ``serialize()``: ``.half().numpy().tobytes()``). FP16 has ~3 decimal
    digits of precision; max absolute error is ~ 1e-3 for inputs in
    [-1, 1].
    """
    torch.manual_seed(1)
    motion = MotionFieldCodec(num_classes=5, num_affine_params=6)
    params = torch.randn(8, 5, 6)  # (T-1, num_classes, 6) — affine entries

    blob = motion.serialize(params)
    restored = motion.deserialize(blob)

    assert restored.shape == params.shape
    # FP16 roundtrip tolerance: 2^(-10) for the mantissa ≈ 1e-3.
    assert torch.allclose(restored, params, atol=1e-3), (
        "MotionFieldCodec is FP16-packed; roundtrip should preserve to 1e-3."
    )


# ── ScorerCorrectionTargets (sparse + 8-bit, ENCODE-ONLY) ─────────────────


def test_scorer_corrections_serialize_format_invariants() -> None:
    """ScorerCorrectionTargets has no public deserialize at compile time;
    decode is implicit at inflate via ``apply_corrections``. We verify the
    serialize byte layout is consistent (count header + 7 bytes/correction)
    so the inflate-side reader has a stable contract.
    """
    import struct as _struct

    torch.manual_seed(2)
    corrections = ScorerCorrectionTargets(max_corrections_per_frame=32)
    B, H, W = 1, 32, 32
    rendered = torch.rand(B, 3, H, W) * 255.0
    target = torch.rand(B, 3, H, W) * 255.0
    fragility = torch.rand(B, H, W)

    indices, values, valid_mask = corrections.compute_corrections(
        rendered, target, fragility,
    )
    blob = corrections.serialize(
        indices[0], values[0], valid_mask=valid_mask[0],
    )

    # Header: uint16 count
    count = _struct.unpack_from("<H", blob, 0)[0]
    expected_count = int(valid_mask[0].sum().item())
    assert count == expected_count, (
        f"serialize() count header {count} ≠ valid_mask.sum() {expected_count}"
    )
    # Body: 7 bytes per correction (uint16 y, uint16 x, int8 r, int8 g, int8 b)
    expected_size = 2 + 7 * count
    assert len(blob) == expected_size, (
        f"serialize() byte length {len(blob)} ≠ {expected_size} "
        f"(2-byte header + 7 bytes/correction)"
    )

    # Manually decode and verify roundtrip on the FIRST correction (proves
    # the byte format documented in the docstring is what's emitted).
    if count > 0:
        y, x, r, g, b_val = _struct.unpack_from("<HHbbb", blob, 2)
        valid_idx = indices[0][valid_mask[0]]
        valid_val = values[0][valid_mask[0]]
        assert (y, x) == (
            int(valid_idx[0, 0].item()),
            int(valid_idx[0, 1].item()),
        ), "uint16 indices must roundtrip exactly"
        # int8 quantization tolerance per channel = 1
        for ch_i, ch_v in enumerate((r, g, b_val)):
            orig = float(valid_val[0, ch_i].item())
            err = abs(ch_v - max(-128, min(127, int(orig))))
            assert err < 2.0, (
                f"int8 channel {ch_i} roundtrip err {err:.3f} > step"
            )


# ── build_minimal_archive (zip container) ──────────────────────────────────


def test_build_minimal_archive_roundtrip_codebook_via_zip() -> None:
    """Codebook bytes inside the assembled zip must roundtrip identically."""
    torch.manual_seed(3)
    codebook = TextureAtomCodebook(num_atoms=8, atom_size=4)
    archive_bytes = build_minimal_archive(
        codebook=codebook,
        motion_params=None,
        motion_codec=None,
        correction_data=None,
        corrections_codec=None,
    )

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        codebook_bytes = zf.read("codebook.bin")

    restored = TextureAtomCodebook.deserialize(codebook_bytes)
    # Match the 4-bit grid as in the serialize roundtrip test above.
    orig_q = (codebook.atoms.detach().clamp(0.0, 255.0) / 255.0 * 15.0).round() * (
        255.0 / 15.0
    )
    assert torch.allclose(orig_q, restored.atoms.detach(), atol=1e-5)
