"""Tests for :mod:`tac.codec_pipeline_sensitivity`.

Coverage:

* :class:`Op_SensitivityPreprocess` satisfies the :class:`CodecOp` Protocol.
* Identity transform: encode -> decode produces the input state_dict
  bit-faithful when ``low_threshold=-1.0`` (every tensor -> mid band).
* ``sensitivity_source='fisher'`` rejects when no artifact is provided.
* ``sensitivity_source='fisher'`` succeeds with a per-channel artifact in
  ``context['sensitivity_artifact']``.
* Stubbed per-tensor sensitivity dict path classifies tensors and applies
  the correct bit-allocation (low-sens -> int4 quantized, high-sens -> fp16).
* Composition with ``Op1_PR101SplitBrotli``: the Op1 sub-blob in
  ``[beta_uniform, Op1]`` is bit-equal to the Op1 sub-blob in ``[Op1]``
  (beta leaves state_dict untouched along the linear-over-state_dict pipeline
  contract).
* Composition with stubbed low-sensitivity dict produces a SMALLER beta blob
  than the uniform-identity beta blob on a synthetic state_dict (pre-quantized
  tensors compress better).
* Validation rejects unknown ``sensitivity_source``.
* Bytes-deterministic when ``sensitivity_source`` is fully specified.
* Identity factory ``Op_SensitivityPreprocess.identity()`` configures
  passthrough behaviour.
"""
from __future__ import annotations

import pytest
import torch

from tac.codec_pipeline import (
    CodecOp,
    CodecPipeline,
    Op1_PR101SplitBrotli,
)
from tac.codec_pipeline_sensitivity import (
    SUPPORTED_SENSITIVITY_SOURCES,
    Op_SensitivityPreprocess,
    SensitivityPreprocessError,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _synthetic_state_dict(seed: int = 0, scale: float = 0.1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * scale
        for name, shape in FIXED_STATE_SCHEMA
    }


def _stub_scores_mostly_low(state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Build a stub sensitivity dict where most tensors are LOW (-> int4)
    and a small number of tensors are HIGH (-> fp16 protect)."""
    scores: dict[str, float] = dict.fromkeys(state_dict.keys(), 0.01)
    # Protect a handful of tensors so the test exercises the high path too.
    protect = {"stem.weight", "rgb_0.weight", "rgb_1.weight"}
    for name in protect:
        if name in scores:
            scores[name] = 0.95
    return scores


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------

def test_op_satisfies_codec_op_protocol() -> None:
    op = Op_SensitivityPreprocess()
    assert isinstance(op, CodecOp)
    assert op.name == "sensitivity_preprocess"


# ---------------------------------------------------------------------------
# Identity transform - bit-faithful roundtrip
# ---------------------------------------------------------------------------

def test_identity_is_bit_faithful_roundtrip() -> None:
    sd = _synthetic_state_dict()
    op = Op_SensitivityPreprocess.identity()
    res = op.encode(sd, context={})
    decoded = op.decode(res.blob, op_state=res.op_state, context={})

    assert set(decoded.keys()) == set(sd.keys())
    for name, tensor in sd.items():
        assert torch.equal(decoded[name], tensor.detach()), (
            f"identity transform must preserve {name!r} bit-faithfully"
        )

    # All tensors land in the mid band.
    classes = res.op_state["classes"]
    assert all(c == "mid" for c in classes.values()), (
        f"identity transform must classify every tensor as 'mid'; got {set(classes.values())}"
    )
    assert res.op_state["n_high"] == 0
    assert res.op_state["n_mid"] == len(sd)
    assert res.op_state["n_low"] == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_rejects_unknown_sensitivity_source() -> None:
    op = Op_SensitivityPreprocess(sensitivity_source="bogus")
    sd = _synthetic_state_dict()
    rep = op.validate(sd)
    assert rep.passed is False
    assert any("unknown sensitivity_source" in f for f in rep.findings)


def test_validate_rejects_fisher_without_artifact() -> None:
    op = Op_SensitivityPreprocess(sensitivity_source="fisher")
    sd = _synthetic_state_dict()
    rep = op.validate(sd, context={})
    assert rep.passed is False
    assert any("sensitivity_artifact" in f for f in rep.findings)


def test_validate_accepts_uniform_with_no_context() -> None:
    op = Op_SensitivityPreprocess.identity()
    sd = _synthetic_state_dict()
    rep = op.validate(sd, context={})
    assert rep.passed is True
    assert rep.findings == []


def test_validate_supported_sources_set() -> None:
    """Documented contract: SUPPORTED_SENSITIVITY_SOURCES is the truth source."""
    assert frozenset(
        {"uniform", "fisher", "imp", "stub"}
    ) == SUPPORTED_SENSITIVITY_SOURCES


# ---------------------------------------------------------------------------
# Fisher artifact path
# ---------------------------------------------------------------------------

def test_fisher_with_per_channel_artifact_succeeds() -> None:
    """``sensitivity_source='fisher'`` reduces a per-channel artifact to a
    per-tensor scalar via max."""
    sd = _synthetic_state_dict()

    # Build a synthetic per-channel artifact: high sensitivity on stem.weight
    # (output channels = 1728), low everywhere else.
    artifact: dict[str, torch.Tensor] = {}
    for name, shape in FIXED_STATE_SCHEMA:
        # Sensitivity dim = output-channel dim = shape[0].
        if name == "stem.weight":
            artifact[name] = torch.full((shape[0],), 0.99)
        else:
            artifact[name] = torch.full((shape[0],), 0.01)

    op = Op_SensitivityPreprocess(sensitivity_source="fisher")
    rep = op.validate(sd, context={"sensitivity_artifact": artifact})
    assert rep.passed is True

    res = op.encode(sd, context={"sensitivity_artifact": artifact})
    classes = res.op_state["classes"]
    assert classes["stem.weight"] == "high"
    # Most other tensors land in 'low' (0.01 < low_threshold=0.1).
    n_low = sum(1 for c in classes.values() if c == "low")
    assert n_low > 20, f"expected >20 tensors classified low, got {n_low}"


def test_fisher_rejects_when_no_artifact_provided_at_encode() -> None:
    """Even bypassing validate(), encode must reject missing artifact."""
    op = Op_SensitivityPreprocess(sensitivity_source="fisher")
    sd = _synthetic_state_dict()
    with pytest.raises(SensitivityPreprocessError, match="sensitivity_artifact"):
        op.encode(sd, context={})


# ---------------------------------------------------------------------------
# Stub path - bit-allocation logic
# ---------------------------------------------------------------------------

def test_stub_classifies_low_to_int4_high_to_fp16() -> None:
    sd = _synthetic_state_dict()
    scores = _stub_scores_mostly_low(sd)
    op = Op_SensitivityPreprocess(sensitivity_source="stub")
    res = op.encode(sd, context={"sensitivity_scores": scores})
    classes = res.op_state["classes"]

    # stem.weight scored 0.95 -> high (>= high_threshold=0.5).
    assert classes["stem.weight"] == "high"
    assert classes["rgb_0.weight"] == "high"
    # Most tensors scored 0.01 -> low (< low_threshold=0.1).
    assert classes["blocks.0.weight"] == "low"
    assert classes["blocks.0.bias"] == "low"

    # Decode + verify the transformed tensors actually match the
    # bit-allocation decision.
    decoded = op.decode(res.blob, op_state=res.op_state, context={})

    # High-sens tensor should equal the fp16 round-trip of the original.
    expected_stem = sd["stem.weight"].to(torch.float16).to(torch.float32)
    assert torch.equal(decoded["stem.weight"], expected_stem)

    # Low-sens tensor should sit on the int4 quant grid (<= 15 unique values
    # in absolute terms; symmetric quant uses {-7..7} -> 15 distinct values
    # before the absmax-zero edge case, plus possibly 0).
    low_tensor = decoded["blocks.0.weight"]
    unique_count = torch.unique(low_tensor).numel()
    assert unique_count <= 16, (
        f"int4 quantized tensor should have <= 16 unique values; got {unique_count}"
    )


def test_stub_low_sens_blob_smaller_than_identity_blob() -> None:
    """Pre-quantizing low-sensitivity tensors to int4 should produce a more
    compressible beta blob than identity transform on the same state_dict."""
    sd = _synthetic_state_dict()
    scores = _stub_scores_mostly_low(sd)

    op_identity = Op_SensitivityPreprocess.identity()
    res_identity = op_identity.encode(sd, context={})

    op_stub = Op_SensitivityPreprocess(sensitivity_source="stub")
    res_stub = op_stub.encode(sd, context={"sensitivity_scores": scores})

    assert res_stub.bytes_out < res_identity.bytes_out, (
        f"stub-low-sensitivity beta blob ({res_stub.bytes_out}B) should be smaller "
        f"than identity beta blob ({res_identity.bytes_out}B); "
        f"int4 substrate was supposed to give us empirical byte savings"
    )


# ---------------------------------------------------------------------------
# Composition with Op1_PR101SplitBrotli (substrate-transform mode)
# ---------------------------------------------------------------------------

def test_uniform_identity_does_not_change_op1_subblob() -> None:
    """Pipeline ``[beta_uniform_identity, Op1]`` must produce an Op1 sub-blob
    that is bit-equal to the Op1 sub-blob in pipeline ``[Op1]``.

    Per the v1 :class:`CodecPipeline` contract (linear-over-state_dict), every
    op sees the original state_dict; therefore Op1 sees the same input in
    both pipelines, and its blob is byte-identical. beta's presence is a
    free-standing forensic record in the wrapper.
    """
    sd = _synthetic_state_dict()
    overrides = {0: "negzig", 5: "off"}

    pipe_op1_only = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False, explicit_overrides=overrides),
    ])
    pipe_beta_op1 = CodecPipeline([
        Op_SensitivityPreprocess.identity(),
        Op1_PR101SplitBrotli(auto_select=False, explicit_overrides=overrides),
    ])

    _, manifest_op1_only = pipe_op1_only.encode(sd)
    _, manifest_beta_op1 = pipe_beta_op1.encode(sd)

    op1_blob_alone = manifest_op1_only.op_results[0].blob
    op1_blob_in_beta_pipeline = manifest_beta_op1.op_results[1].blob
    assert op1_blob_alone == op1_blob_in_beta_pipeline, (
        "Op1 sub-blob must be byte-equal across [Op1] and [beta_uniform_identity, Op1] "
        "pipelines (beta identity is non-mutating)"
    )


def test_beta_op1_pipeline_decodes_via_op1() -> None:
    """End-to-end roundtrip through the 2-op pipeline. The CPL1 wrapper
    decodes both ops; the final returned state_dict is from Op1 (the last op
    in the chain produces the canonical decoded view per
    :meth:`CodecPipeline.decode`)."""
    sd = _synthetic_state_dict()
    pipe = CodecPipeline([
        Op_SensitivityPreprocess.identity(),
        Op1_PR101SplitBrotli(auto_select=False),
    ])
    blob, _ = pipe.encode(sd)
    decoded, replayed = pipe.decode(blob)
    assert replayed == ["sensitivity_preprocess", "pr101_split_brotli"]
    assert set(decoded.keys()) == set(sd.keys())
    for name, shape in FIXED_STATE_SCHEMA:
        assert tuple(decoded[name].shape) == shape


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_byte_deterministic_when_source_fully_specified() -> None:
    """Two encodes with the same identity config must produce byte-identical
    blobs."""
    sd = _synthetic_state_dict()
    op = Op_SensitivityPreprocess.identity()
    res_a = op.encode(sd, context={})
    res_b = op.encode(sd, context={})
    assert res_a.blob == res_b.blob
    assert res_a.op_state == res_b.op_state


def test_byte_deterministic_with_stub_scores() -> None:
    sd = _synthetic_state_dict()
    scores = _stub_scores_mostly_low(sd)
    op = Op_SensitivityPreprocess(sensitivity_source="stub")
    res_a = op.encode(sd, context={"sensitivity_scores": dict(scores)})
    res_b = op.encode(sd, context={"sensitivity_scores": dict(scores)})
    assert res_a.blob == res_b.blob


# ---------------------------------------------------------------------------
# IMP source - per-tensor scalar fallback path
# ---------------------------------------------------------------------------

def test_imp_source_with_per_tensor_scalar_fallback() -> None:
    """``sensitivity_source='imp'`` accepts ``context['sensitivity_scores']``
    (per-tensor scalar) when a full per-channel artifact isn't available."""
    sd = _synthetic_state_dict()
    scores = _stub_scores_mostly_low(sd)
    op = Op_SensitivityPreprocess(sensitivity_source="imp")
    rep = op.validate(sd, context={"sensitivity_scores": scores})
    assert rep.passed is True

    res = op.encode(sd, context={"sensitivity_scores": scores})
    assert res.op_state["classes"]["stem.weight"] == "high"
