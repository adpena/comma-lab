"""Tests for :mod:`tac.codec_pipeline`.

Coverage:
- ``CodecOp`` Protocol satisfaction for landed ops.
- ``CodecPipeline`` encode → decode roundtrip on synthetic state_dict.
- CPL1 wire format is deterministic given identical inputs + explicit overrides.
- Pipeline aborts on a validation failure (Contrarian gate).
- Manifest records per-op byte impact for empirical reasoning about composability.
- Wrong pipeline rejects a wrapper with mismatched op count.
"""

from __future__ import annotations

import pytest
import torch

from tac.codec_pipeline import (
    CodecOp,
    CodecPipeline,
    EncodeResult,
    Op1_PR101SplitBrotli,
    PipelineManifest,
    ValidationReport,
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

def test_op1_satisfies_codec_op_protocol() -> None:
    op = Op1_PR101SplitBrotli()
    assert isinstance(op, CodecOp)
    assert op.name == "pr101_split_brotli"


def test_pipeline_rejects_non_codec_op() -> None:
    class Fake:
        name = "fake"

    with pytest.raises(TypeError, match="does not satisfy CodecOp protocol"):
        CodecPipeline([Fake()])  # type: ignore[list-item]


def test_pipeline_rejects_empty_op_list() -> None:
    with pytest.raises(ValueError, match="at least one op"):
        CodecPipeline([])


# ---------------------------------------------------------------------------
# Encode/decode roundtrip
# ---------------------------------------------------------------------------

def test_pipeline_encode_decode_roundtrip_one_op() -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    blob, manifest = pipeline.encode(sd)
    assert isinstance(blob, bytes)
    assert blob[:4] == b"CPL1"
    assert manifest.final_bytes == len(blob)
    assert len(manifest.op_results) == 1
    assert manifest.op_results[0].op_name == "pr101_split_brotli"

    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["pr101_split_brotli"]
    assert set(decoded.keys()) == set(sd.keys())
    for name, shape in FIXED_STATE_SCHEMA:
        assert tuple(decoded[name].shape) == shape


def test_pipeline_explicit_overrides_byte_deterministic() -> None:
    """Byte-deterministic encode requires explicit overrides (auto_select=False)
    + identical state_dict. Two encodes with the same overrides must produce
    bit-identical wrappers."""
    sd = _synthetic_state_dict()
    overrides = {0: "negzig", 5: "off", 13: "twos"}
    op = Op1_PR101SplitBrotli(auto_select=False, explicit_overrides=overrides)
    pipeline = CodecPipeline([op])
    blob_a, manifest_a = pipeline.encode(sd)
    blob_b, manifest_b = pipeline.encode(sd)
    assert blob_a == blob_b
    assert manifest_a.final_blob_sha256 == manifest_b.final_blob_sha256


def test_pipeline_decode_after_explicit_override_encode() -> None:
    """Encode with explicit overrides → decode must read the overrides back
    from the wrapper's op_state and reconstruct correctly."""
    sd = _synthetic_state_dict()
    overrides = {0: "negzig", 5: "off"}
    pipeline = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False, explicit_overrides=overrides),
    ])
    blob, _ = pipeline.encode(sd)
    decoded, _ = pipeline.decode(blob)
    assert set(decoded.keys()) == set(sd.keys())


def test_pipeline_decode_rejects_bad_magic() -> None:
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    with pytest.raises(ValueError, match="bad magic"):
        pipeline.decode(b"NOPE" + b"\x00" * 100)


def test_pipeline_decode_rejects_op_count_mismatch() -> None:
    """Decode pipeline shape must equal encode pipeline shape — guards against
    operator using the wrong pipeline class to decode a blob."""
    sd = _synthetic_state_dict()
    encode_pipeline = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False),
    ])
    blob, _ = encode_pipeline.encode(sd)

    # Try to decode with a 2-op pipeline.
    decode_pipeline = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False),
        Op1_PR101SplitBrotli(auto_select=False),
    ])
    with pytest.raises(ValueError, match="mismatch"):
        decode_pipeline.decode(blob)


# ---------------------------------------------------------------------------
# Validation gate (Contrarian)
# ---------------------------------------------------------------------------

def test_pipeline_validate_rejects_missing_tensor() -> None:
    sd = _synthetic_state_dict()
    del sd["stem.bias"]  # break the schema
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    with pytest.raises(ValueError, match="validation"):
        pipeline.encode(sd)


def test_pipeline_skip_validate_runs_through() -> None:
    """skip_validate=True is the operator's escape hatch; documented but
    discouraged. Confirm the path exists for emergencies."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    blob, _ = pipeline.encode(sd, skip_validate=True)
    assert len(blob) > 0


# ---------------------------------------------------------------------------
# Manifest empirical reasoning
# ---------------------------------------------------------------------------

def test_manifest_records_per_op_byte_savings() -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    _, manifest = pipeline.encode(sd)
    op_dict = manifest.to_dict()["ops"][0]
    assert op_dict["name"] == "pr101_split_brotli"
    assert op_dict["bytes_in"] > op_dict["bytes_out"]  # codec compresses
    assert op_dict["delta_bytes"] < 0  # negative = bytes saved


def test_manifest_score_claim_default_false() -> None:
    """Pipeline manifest must NOT claim a contest-CUDA score by default.
    Per CLAUDE.md "Forbidden score claims" — every score must be explicitly
    tagged ``[contest-CUDA]`` and ``score_claim`` must default False."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    _, manifest = pipeline.encode(sd)
    assert manifest.score_claim is False
    assert manifest.score_evidence_grade == "predicted"
