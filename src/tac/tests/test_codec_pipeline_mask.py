"""Tests for :mod:`tac.codec_pipeline_mask` - alpha-paradigm mask-encoder bakeoff.

Coverage:
    * Each Op satisfies the :class:`tac.codec_pipeline.CodecOp` Protocol.
    * Each Op's ``validate()`` reports its readiness state correctly:
        - alpha-NeRV: PASS only when ``context['pretrained_nerv_codec']`` is set.
        - alpha-Wavelet / alpha-VQ-VAE: PASS when shape compatible.
        - alpha-grayscale-LUT: PASS unconditionally on a valid mask.
        - alpha-AV1-baseline: PASS only when ffmpeg is on PATH.
    * For every ready codec: encode -> decode roundtrip on a synthetic
      16-frame mask volume.
    * Bakeoff helper picks the smallest valid output across ready candidates;
      AV1 baseline is treated as a control.
    * Manifest persisted to ``experiments/results/lane_codec_pipeline_mask_<UTC>/``.

Stub-tolerant: tests that require an underlying codec to be ready will
``pytest.skip()`` cleanly when ``validate()`` returns ``passed=False``.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest
import torch

from tac.codec_pipeline import CodecOp, CodecPipeline
from tac.codec_pipeline_mask import (
    Op_AV1BaselineMask,
    Op_GrayscaleLutMask,
    Op_NerVMaskCodec,
    Op_VqvaeMaskCodec,
    Op_WaveletMaskCodec,
    pick_smallest_mask_codec,
    run_bakeoff_and_write_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_mask_input(
    *, t: int = 16, h: int = 32, w: int = 32, seed: int = 2026
) -> dict[str, torch.Tensor]:
    """Build a synthetic (T, H, W) int64 mask tensor with values in [0, 5).

    The pattern mixes a gradient + per-frame offset to avoid degenerate
    all-zeros masks (which would compress to ~0 bytes regardless of codec).
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, 5, (t, h, w), generator=g, dtype=torch.int64)
    return {"masks": base}


def _output_dir() -> Path:
    """Per-test-session output dir under ``experiments/results/`` (NOT /tmp)."""
    repo_root = Path(__file__).resolve().parents[3]
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = repo_root / "experiments" / "results" / f"lane_codec_pipeline_mask_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Protocol satisfaction (5 tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op,expected_name",
    [
        (Op_NerVMaskCodec(), "alpha_nerv_mask"),
        (Op_WaveletMaskCodec(), "alpha_wavelet_mask"),
        (Op_VqvaeMaskCodec(), "alpha_vqvae_mask"),
        (Op_GrayscaleLutMask(), "alpha_grayscale_lut_mask"),
        (Op_AV1BaselineMask(), "alpha_av1_baseline_mask"),
    ],
)
def test_op_satisfies_codec_op_protocol(op, expected_name) -> None:
    assert isinstance(op, CodecOp)
    assert op.name == expected_name


# ---------------------------------------------------------------------------
# Validate readiness (5 tests)
# ---------------------------------------------------------------------------

def test_nerv_validate_fails_without_pretrained_codec() -> None:
    """alpha-NeRV requires a pretrained codec via context; without one, validate
    must FAIL with a finding pointing at the readiness audit."""
    sd = _synthetic_mask_input()
    op = Op_NerVMaskCodec()
    rep = op.validate(sd, context={})
    assert not rep.passed
    assert any("not yet L2 ready" in f or "pretrained_nerv_codec" in f for f in rep.findings)


def test_wavelet_validate_passes_on_compatible_shape() -> None:
    sd = _synthetic_mask_input(t=8, h=32, w=32)  # divisible by 2^2=4
    op = Op_WaveletMaskCodec(levels=2)
    rep = op.validate(sd, context={})
    assert rep.passed, rep.findings


def test_wavelet_validate_fails_on_incompatible_shape() -> None:
    sd = _synthetic_mask_input(t=4, h=30, w=30)  # NOT divisible by 2^2
    op = Op_WaveletMaskCodec(levels=2)
    rep = op.validate(sd, context={})
    assert not rep.passed
    assert any("divisible" in f for f in rep.findings)


def test_vqvae_validate_passes_on_compatible_shape() -> None:
    sd = _synthetic_mask_input(t=4, h=16, w=16)  # divisible by patch_size=4
    op = Op_VqvaeMaskCodec(patch_size=4, codebook_size=64)
    rep = op.validate(sd, context={})
    assert rep.passed, rep.findings


def test_grayscale_lut_validate_always_passes_on_valid_mask() -> None:
    sd = _synthetic_mask_input()
    op = Op_GrayscaleLutMask()
    rep = op.validate(sd, context={})
    assert rep.passed, rep.findings


def test_av1_baseline_validate_reports_ffmpeg_state() -> None:
    sd = _synthetic_mask_input()
    op = Op_AV1BaselineMask()
    rep = op.validate(sd, context={})
    if shutil.which("ffmpeg") is None:
        assert not rep.passed
        assert any("ffmpeg" in f for f in rep.findings)
    else:
        assert rep.passed, rep.findings


# ---------------------------------------------------------------------------
# Roundtrip (4 tests, skip-cleanly on stubs)
# ---------------------------------------------------------------------------

def test_grayscale_lut_roundtrip_exact() -> None:
    """Grayscale-LUT roundtrip must be EXACT (nearest-neighbour matches the
    Selfcomp class-target table by construction)."""
    sd = _synthetic_mask_input()
    op = Op_GrayscaleLutMask()
    rep = op.validate(sd, context={})
    if not rep.passed:
        pytest.skip(f"alpha-grayscale-LUT not ready: {rep.findings}")
    result = op.encode(sd, context={})
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    assert "masks" in decoded
    assert torch.equal(decoded["masks"], sd["masks"])


def test_wavelet_roundtrip_or_skip() -> None:
    """Wavelet roundtrip - exact when quantization is fine enough that the
    inverse DWT + argmax recovers the same one-hot encoding. step_ll=step_detail=0.5
    is the canonical safe setting for synthetic 5-class masks."""
    sd = _synthetic_mask_input(t=8, h=32, w=32)
    op = Op_WaveletMaskCodec(levels=2, step_ll=0.5, step_detail=0.5)
    rep = op.validate(sd, context={})
    if not rep.passed:
        pytest.skip(f"alpha-wavelet not ready: {rep.findings}")
    result = op.encode(sd, context={})
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    assert "masks" in decoded
    assert decoded["masks"].shape == sd["masks"].shape
    # Lossy via DWT quantization; some pixels may flip class boundaries.
    # Match-rate should be high (>50%) for any working wavelet codec.
    match_rate = float((decoded["masks"] == sd["masks"]).float().mean())
    assert match_rate >= 0.50, f"wavelet match_rate={match_rate:.4f} too low"


def test_vqvae_roundtrip_or_skip() -> None:
    sd = _synthetic_mask_input(t=4, h=16, w=16)
    op = Op_VqvaeMaskCodec(patch_size=4, codebook_size=64)
    rep = op.validate(sd, context={})
    if not rep.passed:
        pytest.skip(f"alpha-VQ-VAE not ready: {rep.findings}")
    result = op.encode(sd, context={})
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    assert "masks" in decoded
    assert decoded["masks"].shape == sd["masks"].shape


def test_av1_baseline_roundtrip_or_skip() -> None:
    sd = _synthetic_mask_input()
    op = Op_AV1BaselineMask(crf=20)
    rep = op.validate(sd, context={})
    if not rep.passed:
        pytest.skip(f"alpha-AV1-baseline not ready: {rep.findings}")
    result = op.encode(sd, context={})
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    assert "masks" in decoded
    assert decoded["masks"].shape == sd["masks"].shape


def test_nerv_validate_fails_on_missing_masks_key() -> None:
    """Every Op must reject a state_dict missing the canonical 'masks' key."""
    op = Op_NerVMaskCodec()
    rep = op.validate({}, context={})
    assert not rep.passed
    assert any("masks" in f for f in rep.findings)


# ---------------------------------------------------------------------------
# Bakeoff helper (3 tests)
# ---------------------------------------------------------------------------

def test_bakeoff_picks_smallest_winner() -> None:
    """Bakeoff over the default 5-op candidate list returns a valid winner."""
    sd = _synthetic_mask_input(t=8, h=32, w=32)
    winner, winner_result, entries = pick_smallest_mask_codec(sd)
    assert isinstance(winner, CodecOp)
    assert winner_result.bytes_out > 0
    assert winner_result.op_name == winner.name
    # At least one candidate must be ready (grayscale-LUT always is).
    ready_entries = [e for e in entries if e.ready]
    assert len(ready_entries) >= 1
    # Winner's bytes_out is the minimum across ready entries.
    ready_bytes = [e.bytes_out for e in ready_entries if e.bytes_out is not None]
    assert winner_result.bytes_out == min(ready_bytes)


def test_bakeoff_records_unready_findings() -> None:
    """Unready candidates must appear in the entries list with findings or
    error so operators can audit which codecs are stub vs ready."""
    sd = _synthetic_mask_input()
    _, _, entries = pick_smallest_mask_codec(sd)
    # NeRV is always unready without context['pretrained_nerv_codec'].
    nerv_entry = next(e for e in entries if e.op_name == "alpha_nerv_mask")
    assert not nerv_entry.ready
    assert nerv_entry.bytes_out is None
    assert nerv_entry.findings or nerv_entry.error


def test_bakeoff_writes_manifest_under_experiments_results() -> None:
    """Manifest must land under ``experiments/results/`` with the canonical
    schema, NEVER under /tmp (CLAUDE.md non-negotiable)."""
    sd = _synthetic_mask_input(t=8, h=32, w=32)
    out_dir = _output_dir()
    manifest_path = run_bakeoff_and_write_manifest(sd, output_dir=out_dir)
    assert manifest_path.exists()
    assert "/tmp" not in str(manifest_path)
    assert "experiments/results" in str(manifest_path)
    data = json.loads(manifest_path.read_text())
    assert data["schema_version"] == 1
    assert data["contract"] == "alpha_mask_bakeoff_manifest_v1"
    assert data["score_claim"] is False
    assert data["score_evidence_grade"] == "empirical"
    assert data["winner"]["bytes_out"] > 0
    assert len(data["candidates"]) == 5  # all 5 in default_alpha_mask_ops


# ---------------------------------------------------------------------------
# CodecPipeline composability (2 tests)
# ---------------------------------------------------------------------------

def test_grayscale_lut_in_codec_pipeline() -> None:
    """The alpha-grayscale-LUT op must wire into a single-op
    :class:`CodecPipeline` and produce a CPL1-wrapped blob that round-trips."""
    sd = _synthetic_mask_input()
    pipeline = CodecPipeline([Op_GrayscaleLutMask()])
    blob, manifest = pipeline.encode(sd)
    assert blob[:4] == b"CPL1"
    assert manifest.final_bytes == len(blob)
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["alpha_grayscale_lut_mask"]
    assert torch.equal(decoded["masks"], sd["masks"])


def test_wavelet_in_codec_pipeline_or_skip() -> None:
    sd = _synthetic_mask_input(t=8, h=32, w=32)
    pipeline = CodecPipeline([Op_WaveletMaskCodec(levels=2, step_ll=0.5, step_detail=0.5)])
    op_rep = pipeline.ops[0].validate(sd, context={})
    if not op_rep.passed:
        pytest.skip(f"alpha-wavelet not ready: {op_rep.findings}")
    blob, manifest = pipeline.encode(sd)
    assert blob[:4] == b"CPL1"
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["alpha_wavelet_mask"]
    assert decoded["masks"].shape == sd["masks"].shape


# ---------------------------------------------------------------------------
# Input contract enforcement
# ---------------------------------------------------------------------------

def test_op_rejects_wrong_dtype() -> None:
    op = Op_GrayscaleLutMask()
    bad = {"masks": torch.zeros((2, 4, 4), dtype=torch.float32)}
    rep = op.validate(bad, context={})
    assert not rep.passed
    assert any("int64" in f for f in rep.findings)


def test_op_rejects_out_of_range_classes() -> None:
    op = Op_GrayscaleLutMask()
    bad_masks = torch.full((2, 4, 4), 7, dtype=torch.int64)
    bad = {"masks": bad_masks}
    rep = op.validate(bad, context={})
    assert not rep.passed
    assert any("[0, 5)" in f or "values must be in" in f for f in rep.findings)
