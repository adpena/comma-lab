"""Tests for :mod:`tac.codec_pipeline_apogee_int`.

Coverage:
  - Protocol satisfaction (CodecOp Protocol)
  - validate refuses bits=5 (DEFERRED-pending-research)
  - validate accepts bits=6 (basin-parity PASSED) and bits=7
  - validate rejects bits outside supported range
  - validate rejects missing schema tensor
  - validate WARNS at bits=4 (FALSIFIED but not killed) but passes
  - encode/decode roundtrip int6
  - encode/decode roundtrip int7
  - byte-faithful determinism (two encodes with same bits + state_dict
    produce identical bytes)
  - composition with Op1_PR101SplitBrotli ([Op3, Op1] roundtrip)
  - composition with Op2_PR103ArithmeticCodec ([Op3, Op2] roundtrip)
  - empirical: [Op3, Op1] smaller than [Op1] alone (substrate-transform
    multiplicative gain)
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest
import torch

from tac.codec_pipeline import (
    CodecOp,
    CodecPipeline,
    Op1_PR101SplitBrotli,
    Op2_PR103ArithmeticCodec,
)
from tac.codec_pipeline_apogee_int import (
    REFUSED_BITS,
    SAFE_BITS,
    WARN_BITS,
    Op3_ApogeeIntN_Substrate,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA


def _synthetic_state_dict(seed: int = 0, scale: float = 0.1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * scale
        for name, shape in FIXED_STATE_SCHEMA
    }


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------

def test_op3_satisfies_codec_op_protocol() -> None:
    op = Op3_ApogeeIntN_Substrate(bits=6)
    assert isinstance(op, CodecOp)
    assert op.name == "apogee_intN_substrate"
    assert op.bits == 6


def test_op3_default_bits_is_int6() -> None:
    """Default bits=6 because basin-parity PASSED at int6 per
    project_apogee_int6_basin_parity_PASSED_20260507."""
    op = Op3_ApogeeIntN_Substrate()
    assert op.bits == 6


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

def test_op3_validate_refuses_bits5_deferred() -> None:
    """bits=5 must be REFUSED - apogee_int5 DEFERRED-pending-research
    (basin-parity FAIL: pose_dist_delta 2.26x threshold)."""
    op = Op3_ApogeeIntN_Substrate(bits=5)
    sd = _synthetic_state_dict()
    rep = op.validate(sd, context={})
    assert rep.passed is False
    assert any("DEFERRED-pending-research" in f for f in rep.findings)
    assert any("basin-parity" in f.lower() for f in rep.findings)
    assert 5 in REFUSED_BITS


def test_op3_validate_accepts_bits6_basin_parity_passed() -> None:
    """bits=6 is SAFE - basin-parity PASSED (pose_dist_delta +1.08e-4,
    30x below 1e-3 threshold)."""
    op = Op3_ApogeeIntN_Substrate(bits=6)
    sd = _synthetic_state_dict()
    rep = op.validate(sd, context={})
    assert rep.passed is True
    assert rep.findings == []  # no warnings at int6
    assert 6 in SAFE_BITS


def test_op3_validate_accepts_bits7_safe() -> None:
    """bits=7 is in SAFE_BITS (Track G empirical PASSED)."""
    op = Op3_ApogeeIntN_Substrate(bits=7)
    sd = _synthetic_state_dict()
    rep = op.validate(sd, context={})
    assert rep.passed is True
    assert 7 in SAFE_BITS


def test_op3_validate_warns_bits4_but_passes() -> None:
    """bits=4 is in WARN_BITS (FALSIFIED at score 1.4287 [contest-CUDA T4])
    but passed=True so substrate-transform composition is permitted for
    forensics / Pareto-mapping. NOT killed - DEFERRED-pending-research at
    different config (QAT/LSQ/per-channel)."""
    op = Op3_ApogeeIntN_Substrate(bits=4)
    sd = _synthetic_state_dict()
    rep = op.validate(sd, context={})
    assert rep.passed is True
    assert any("FALSIFIED" in f for f in rep.findings)
    assert 4 in WARN_BITS


def test_op3_validate_rejects_unsupported_bits() -> None:
    """bits outside {3, 4, 5, 6, 7, 8} must be rejected."""
    op = Op3_ApogeeIntN_Substrate(bits=2)
    sd = _synthetic_state_dict()
    rep = op.validate(sd, context={})
    assert rep.passed is False
    assert any("not supported" in f for f in rep.findings)


def test_op3_validate_rejects_missing_schema_tensor() -> None:
    """If a FIXED_STATE_SCHEMA tensor is missing, validate fails."""
    op = Op3_ApogeeIntN_Substrate(bits=6)
    sd = _synthetic_state_dict()
    del sd["stem.bias"]
    rep = op.validate(sd, context={})
    assert rep.passed is False
    assert any("missing tensors" in f for f in rep.findings)
    assert any("stem.bias" in f for f in rep.findings)


def test_op3_validate_rejects_shape_mismatch() -> None:
    """If a schema tensor has the wrong shape, validate fails."""
    op = Op3_ApogeeIntN_Substrate(bits=6)
    sd = _synthetic_state_dict()
    sd["stem.bias"] = torch.zeros(99)  # wrong shape
    rep = op.validate(sd, context={})
    assert rep.passed is False
    assert any("shape mismatches" in f for f in rep.findings)


# ---------------------------------------------------------------------------
# Encode / decode roundtrip
# ---------------------------------------------------------------------------

def test_op3_alone_int6_roundtrip_via_pipeline() -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op3_ApogeeIntN_Substrate(bits=6)])
    blob, manifest = pipeline.encode(sd)
    assert blob[:4] in (b"CPL1", b"CPL2")  # CPL2 is canonical default 2026-05-08
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["apogee_intN_substrate"]
    assert set(decoded.keys()) == set(sd.keys())
    # Per-tensor shape preserved.
    for name, shape in FIXED_STATE_SCHEMA:
        assert tuple(decoded[name].shape) == shape


def test_op3_int7_roundtrip_via_pipeline() -> None:
    """Track G basin-parity PASSED at int7."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op3_ApogeeIntN_Substrate(bits=7)])
    blob, _ = pipeline.encode(sd)
    decoded, _ = pipeline.decode(blob)
    assert set(decoded.keys()) == set(sd.keys())
    for name, shape in FIXED_STATE_SCHEMA:
        assert tuple(decoded[name].shape) == shape


def test_op3_int6_decoded_close_to_original() -> None:
    """Sanity: int6 quantization rel_err ~ 1.55% so dequantized weights
    are reasonably close (not exactly equal - lossy)."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op3_ApogeeIntN_Substrate(bits=6)])
    blob, _ = pipeline.encode(sd)
    decoded, _ = pipeline.decode(blob)
    for name, _ in FIXED_STATE_SCHEMA:
        t_orig = sd[name].to(torch.float32)
        t_back = decoded[name].to(torch.float32)
        # Allow up to 5% relative error per-tensor (int6 with block-FP).
        denom = max(t_orig.abs().max().item(), 1e-8)
        rel_err = (t_orig - t_back).abs().max().item() / denom
        assert rel_err <= 0.05, (
            f"{name}: rel_err {rel_err:.4f} > 0.05 at int6 (basin-parity "
            f"safety guard)"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_op3_byte_deterministic() -> None:
    """Two encodes with the same bits + same state_dict produce identical
    CPL1 bytes (substrate-transform reproducibility invariant)."""
    sd = _synthetic_state_dict()
    pipeline_a = CodecPipeline([Op3_ApogeeIntN_Substrate(bits=6)])
    pipeline_b = CodecPipeline([Op3_ApogeeIntN_Substrate(bits=6)])
    blob_a, manifest_a = pipeline_a.encode(sd)
    blob_b, manifest_b = pipeline_b.encode(sd)
    assert blob_a == blob_b
    assert manifest_a.final_blob_sha256 == manifest_b.final_blob_sha256


# ---------------------------------------------------------------------------
# Composition with Op 1
# ---------------------------------------------------------------------------

def test_op3_op1_composition_roundtrip() -> None:
    """[Op3(int6), Op1] pipeline encodes then decodes via CPL1.

    The CPL1 wrapper carries both ops' blobs + op_state; decode replays
    them in order. The final ``decoded`` state_dict comes from the LAST
    op's decode (Op 1 in this case), per CodecPipeline's contract.
    """
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op1_PR101SplitBrotli(auto_select=False),
    ])
    blob, manifest = pipeline.encode(sd)
    assert blob[:4] in (b"CPL1", b"CPL2")  # CPL2 is canonical default 2026-05-08
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["apogee_intN_substrate", "pr101_split_brotli"]
    assert set(decoded.keys()) == set(sd.keys())


def test_op3_op1_op_state_includes_bits_and_tensor_names() -> None:
    """Op 3's op_state must include ``bits`` and ``tensor_names`` so the
    decoder can deserialize the per-tensor blob stream."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op1_PR101SplitBrotli(auto_select=False),
    ])
    _, manifest = pipeline.encode(sd)
    op3_state = manifest.op_results[0].op_state
    assert op3_state["bits"] == 6
    assert op3_state["block_size"] == 128
    schema_names = [n for n, _ in FIXED_STATE_SCHEMA]
    assert op3_state["tensor_names"] == schema_names


# ---------------------------------------------------------------------------
# Composition with Op 2
# ---------------------------------------------------------------------------

def test_op3_op2_composition_roundtrip() -> None:
    """[Op3(int6), Op2] pipeline encodes then decodes via CPL1."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op2_PR103ArithmeticCodec(),
    ])
    blob, manifest = pipeline.encode(sd)
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["apogee_intN_substrate", "pr103_arithmetic_codec"]
    assert set(decoded.keys()) == set(sd.keys())


# ---------------------------------------------------------------------------
# Empirical: substrate-transform multiplicative gain
# ---------------------------------------------------------------------------

def test_op3_op1_substrate_transform_multiplicative_gain(tmp_path: Path) -> None:
    """[Op3(int6), Op1] should produce a SMALLER final blob than [Op1] alone
    on synthetic int6-quantizable state_dict - the substrate-transform
    multiplicative codec-engineering effect.

    NOTE: the CPL1 wrapper for [Op3, Op1] carries BOTH ops' blobs. So the
    multiplicative effect we measure is "Op 1's blob shrinks when Op 3
    has pre-quantized the state_dict to int6" - not "the final CPL1 blob
    shrinks" (that one might grow because we now carry Op 3's blob too).

    The empirical assertion: Op1 alone bytes_out (its raw split-Brotli
    blob, recorded in manifest.op_results[0].bytes_out) is GREATER THAN
    Op1's bytes_out when run after Op3-int6 quantizes the state_dict
    (recorded in manifest.op_results[1].bytes_out of the [Op3, Op1]
    pipeline). The dequantized state_dict has lower entropy and Brotli
    compresses it further.
    """
    sd = _synthetic_state_dict()

    # Baseline: Op 1 alone.
    op1_alone = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    _, manifest_alone = op1_alone.encode(sd)
    op1_alone_bytes = manifest_alone.op_results[0].bytes_out

    # Substrate-transform: [Op 3, Op 1].
    op3_then_op1 = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op1_PR101SplitBrotli(auto_select=False),
    ])
    _, manifest_stack = op3_then_op1.encode(sd)
    op1_after_op3_bytes = manifest_stack.op_results[1].bytes_out

    # Op 1's blob is smaller after Op 3 pre-quantizes. The substrate
    # multiplied Op 1's gain.
    assert op1_after_op3_bytes < op1_alone_bytes, (
        f"Substrate-transform multiplicative gain not observed: "
        f"Op1-alone={op1_alone_bytes}B, "
        f"Op1-after-Op3={op1_after_op3_bytes}B (must be smaller)"
    )

    # Record the empirical manifest for the lane.
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = tmp_path / f"lane_codec_pipeline_apogee_int_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": "[empirical:substrate-transform-codec-engineering]",
        "evidence_grade": "predicted",
        "score_claim": False,
        "op1_alone_bytes": op1_alone_bytes,
        "op1_after_op3_int6_bytes": op1_after_op3_bytes,
        "delta_bytes": op1_after_op3_bytes - op1_alone_bytes,
        "savings_pct": (
            (op1_alone_bytes - op1_after_op3_bytes) / op1_alone_bytes * 100.0
        ),
        "synthetic_input": "FIXED_STATE_SCHEMA seed=0 scale=0.1",
        "note": "Synthetic state_dict; not a contest-CUDA score claim.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2))


def test_op3_op1_op_byte_savings_recorded_in_manifest() -> None:
    """The pipeline manifest records per-op byte impact for empirical
    composability reasoning."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op1_PR101SplitBrotli(auto_select=False),
    ])
    _, manifest = pipeline.encode(sd)
    d = manifest.to_dict()
    assert len(d["ops"]) == 2
    assert d["ops"][0]["name"] == "apogee_intN_substrate"
    assert d["ops"][1]["name"] == "pr101_split_brotli"
    # Both ops compress vs raw fp32 input.
    assert d["ops"][0]["delta_bytes"] < 0
    assert d["ops"][1]["delta_bytes"] < 0


# ---------------------------------------------------------------------------
# Pipeline-level Contrarian gate routes through Op 3's validate
# ---------------------------------------------------------------------------

def test_pipeline_aborts_when_op3_refuses_bits5() -> None:
    """When Op 3 is configured with bits=5, the pipeline must abort encode
    via the Contrarian gate (no DEFERRED-pending-research dispatch slips
    through)."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op3_ApogeeIntN_Substrate(bits=5)])
    with pytest.raises(ValueError, match="DEFERRED-pending-research"):
        pipeline.encode(sd)


# ---------------------------------------------------------------------------
# Bug-hunter v2 (re-opened MEDIUM): partial-substrate refusal at decode
# ---------------------------------------------------------------------------

def test_op3_decode_refuses_partial_substrate_under_skip_validate() -> None:
    """When ``pipeline.encode(skip_validate=True)`` is used with a state_dict
    missing some FIXED_STATE_SCHEMA names, Op3 previously emitted a partial
    blob that decode happily reconstructed -- silently corrupting the
    substrate handed to downstream Op1/Op2. Per re-opened bug-hunter v2
    finding, decode now refuses partial blobs at the substrate boundary."""
    full_sd = _synthetic_state_dict()
    # Drop one schema tensor to provoke the bug.
    schema_names = [name for name, _ in FIXED_STATE_SCHEMA]
    dropped = schema_names[0]
    partial_sd = {n: t for n, t in full_sd.items() if n != dropped}

    op = Op3_ApogeeIntN_Substrate(bits=6)
    # encode does not check schema completeness (only validate does); under
    # skip_validate=True the partial encode succeeds and produces a blob
    # missing the dropped tensor.
    res = op.encode(partial_sd, context={})
    # Decode used to silently return a partial state_dict; now it must raise.
    with pytest.raises(ValueError, match="partial substrate refused"):
        op.decode(res.blob, op_state=res.op_state, context={})


def test_op3_pipeline_skip_validate_partial_state_dict_fails_at_decode() -> None:
    """Pipeline-level guard: under skip_validate=True the encode succeeds but
    the substrate-transform `current_state = op.decode(...)` step fails loudly
    with the partial-substrate ValueError (not a downstream KeyError)."""
    full_sd = _synthetic_state_dict()
    schema_names = [name for name, _ in FIXED_STATE_SCHEMA]
    dropped = schema_names[-1]
    partial_sd = {n: t for n, t in full_sd.items() if n != dropped}

    pipeline = CodecPipeline([
        Op3_ApogeeIntN_Substrate(bits=6),
        Op1_PR101SplitBrotli(auto_select=False),
    ])
    with pytest.raises(ValueError, match="partial substrate refused"):
        pipeline.encode(partial_sd, skip_validate=True)


def test_op3_full_state_dict_skip_validate_still_roundtrips() -> None:
    """Sanity counterpart: skip_validate=True with a COMPLETE state_dict
    must still roundtrip cleanly through Op3."""
    sd = _synthetic_state_dict()
    op = Op3_ApogeeIntN_Substrate(bits=6)
    res = op.encode(sd, context={})
    decoded = op.decode(res.blob, op_state=res.op_state, context={})
    schema_names = {name for name, _ in FIXED_STATE_SCHEMA}
    assert set(decoded.keys()) == schema_names
