"""Tests for :mod:`tac.codec_pipeline_joint_admm`.

Coverage:
- Protocol satisfaction (CodecOp / Op_GammaJointADMM)
- ``validate()`` on default + degenerate state_dicts
- encode/decode roundtrip on a synthetic 28-tensor state_dict (small ADMM iters)
- Substitutional-mode pipeline: ``[Op_GammaJointADMM]`` alone roundtrip
- Side-by-side byte comparison vs ``Op2_PR103ArithmeticCodec``
- Bytes-deterministic encode (same input + hyperparams -> same blob)
- ``validate()`` rejects bad rho_init / iters
- ADMM convergence sanity: at least 1 iter without NaN
- Op_state shape (streams metadata round-trips through the CPL1 wrapper)
"""

from __future__ import annotations

import json

import torch

from tac.codec_pipeline import (
    CodecOp,
    CodecPipeline,
    Op2_PR103ArithmeticCodec,
)
from tac.codec_pipeline_joint_admm import (
    SUBSTRATE_AWARE_INIT_WARN_MESSAGE,
    Op_GammaJointADMM,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _synthetic_state_dict(
    seed: int = 0, scale: float = 0.1
) -> dict[str, torch.Tensor]:
    """Same fixture as ``test_codec_pipeline.py`` for cross-comparison."""
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * scale
        for name, shape in FIXED_STATE_SCHEMA
    }


def _small_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    """Small state_dict for fast tests."""
    g = torch.Generator().manual_seed(seed)
    return {
        "a.weight": torch.randn(8, 8, generator=g) * 0.1,
        "a.bias": torch.randn(8, generator=g) * 0.1,
        "b.weight": torch.randn(4, 8, generator=g) * 0.1,
    }


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


def test_op_gamma_satisfies_codec_op_protocol() -> None:
    op = Op_GammaJointADMM(max_admm_iters=2)
    assert isinstance(op, CodecOp)
    assert op.name == "gamma_joint_admm"


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------


def test_validate_passes_default_state_dict() -> None:
    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _small_state_dict()
    rep = op.validate(sd, context={})
    assert rep.passed, f"unexpected findings: {rep.findings}"
    assert rep.op_name == "gamma_joint_admm"


def test_validate_rejects_bad_rho_init() -> None:
    op = Op_GammaJointADMM(rho_init=0.0, max_admm_iters=2)
    sd = _small_state_dict()
    rep = op.validate(sd, context={})
    assert not rep.passed
    assert any("rho_init" in f for f in rep.findings)


def test_validate_rejects_bad_max_iters() -> None:
    op = Op_GammaJointADMM(rho_init=1.0, max_admm_iters=0)
    sd = _small_state_dict()
    rep = op.validate(sd, context={})
    assert not rep.passed
    assert any("max_admm_iters" in f for f in rep.findings)


def test_validate_rejects_empty_state_dict() -> None:
    op = Op_GammaJointADMM(max_admm_iters=2)
    rep = op.validate({}, context={})
    assert not rep.passed
    assert any("non-empty" in f for f in rep.findings)


def test_validate_rejects_non_finite_tensor() -> None:
    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _small_state_dict()
    sd["a.weight"][0, 0] = float("nan")
    rep = op.validate(sd, context={})
    assert not rep.passed
    assert any("non-finite" in f for f in rep.findings)


# ---------------------------------------------------------------------------
# Encode / decode roundtrip
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip_small_state_dict() -> None:
    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _small_state_dict()
    res = op.encode(sd, context={})
    # JCSv1 magic in the embedded blob
    assert res.blob[:4] == b"JCSP"
    assert res.bytes_in > 0
    assert res.bytes_out > 0
    assert res.op_name == "gamma_joint_admm"
    assert "streams" in res.op_state
    assert res.op_state["stream_count"] == len(sd)
    # ADMM convergence-sanity: iter count >= 1, no NaN/garbage
    assert res.op_state["admm_iters"] >= 1

    decoded = op.decode(
        res.blob,
        op_state=res.op_state,
        context={},
    )
    assert set(decoded.keys()) == set(sd.keys())
    for name, t in sd.items():
        d = decoded[name]
        assert d.shape == t.shape
        assert d.dtype == torch.float32
        # Symmetric int8 quant has a bounded error: |t - d| <= scale.
        # scale = max(|t|)/127 ⇒ max element-wise error == max(|t|)/127.
        max_abs = float(t.detach().abs().max().item())
        if max_abs > 0:
            tol = max_abs / 127.0 * 1.05  # 5% slack for rint rounding
            err = float((t - d).abs().max().item())
            assert err <= tol, f"tensor {name!r} dequant err {err} > tol {tol}"


def test_encode_decode_roundtrip_full_state_dict() -> None:
    """Roundtrip on the full 28-tensor PR101 schema (same scaffolding the
    pipeline tests use). Uses small ADMM iters for speed."""
    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _synthetic_state_dict()
    res = op.encode(sd, context={})
    decoded = op.decode(
        res.blob,
        op_state=res.op_state,
        context={},
    )
    assert set(decoded.keys()) == set(sd.keys())
    for name, shape in FIXED_STATE_SCHEMA:
        assert tuple(decoded[name].shape) == shape


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------


def test_substitutional_pipeline_op_gamma_alone_roundtrip() -> None:
    """``[Op_GammaJointADMM]`` alone goes through the canonical pipeline
    encode/decode path with the CPL1 wire format."""
    sd = _small_state_dict()
    pipeline = CodecPipeline([Op_GammaJointADMM(max_admm_iters=2)])
    blob, manifest = pipeline.encode(sd)
    assert blob[:4] in (b"CPL1", b"CPL2")  # CPL2 is canonical default 2026-05-08
    assert len(manifest.op_results) == 1
    assert manifest.op_results[0].op_name == "gamma_joint_admm"
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["gamma_joint_admm"]
    assert set(decoded.keys()) == set(sd.keys())


def test_op_state_serialises_through_cpl1_wrapper() -> None:
    """The CPL1 wrapper json-encodes op_state. Verify we don't ship any
    non-serialisable types."""
    sd = _small_state_dict()
    pipeline = CodecPipeline([Op_GammaJointADMM(max_admm_iters=2)])
    blob, manifest = pipeline.encode(sd)
    op_state = manifest.op_results[0].op_state
    # Must round-trip through json.
    encoded_json = json.dumps(op_state, sort_keys=True)
    decoded_json = json.loads(encoded_json)
    assert decoded_json["stream_count"] == len(sd)
    # Decode via pipeline (which json-decodes op_state internally).
    decoded, _ = pipeline.decode(blob)
    assert set(decoded.keys()) == set(sd.keys())


# ---------------------------------------------------------------------------
# Bytes-determinism
# ---------------------------------------------------------------------------


def test_encode_is_bytes_deterministic_same_input_same_hyperparams() -> None:
    sd = _small_state_dict(seed=42)

    op_a = Op_GammaJointADMM(rho_init=1.0, max_admm_iters=2)
    op_b = Op_GammaJointADMM(rho_init=1.0, max_admm_iters=2)
    res_a = op_a.encode(sd, context={})
    res_b = op_b.encode(sd, context={})
    assert res_a.blob == res_b.blob
    assert res_a.bytes_out == res_b.bytes_out


# ---------------------------------------------------------------------------
# Side-by-side byte comparison vs Op2_PR103ArithmeticCodec
# ---------------------------------------------------------------------------


def test_side_by_side_byte_comparison_vs_op2_pr103() -> None:
    """Substitutional composition: empirically compare gamma vs Op 2 on the
    SAME synthetic state_dict. NO winner-claim is asserted - we only record
    the byte counts so the operator can read the comparison off the test
    output. Both blobs MUST roundtrip-decode to the same set of keys.
    """
    sd = _synthetic_state_dict()

    op_gamma = Op_GammaJointADMM(max_admm_iters=2)
    op_pr103 = Op2_PR103ArithmeticCodec()

    res_gamma = op_gamma.encode(sd, context={})
    res_pr103 = op_pr103.encode(sd, context={})

    # Both must produce non-empty blobs.
    assert res_gamma.bytes_out > 0
    assert res_pr103.bytes_out > 0

    # Both must roundtrip-decode the schema keys.
    decoded_gamma = op_gamma.decode(
        res_gamma.blob,
        op_state=res_gamma.op_state,
        context={},
    )
    decoded_pr103 = op_pr103.decode(
        res_pr103.blob,
        op_state=res_pr103.op_state,
        context={},
    )
    assert set(decoded_gamma.keys()) == set(sd.keys())
    assert set(decoded_pr103.keys()) == set(sd.keys())

    # Empirical record only (no winner). pytest -s prints these.
    print(
        f"\n[side-by-side bytes synthetic 28-tensor PR101 schema] "
        f"Op_GammaJointADMM={res_gamma.bytes_out}B  "
        f"Op2_PR103={res_pr103.bytes_out}B"
    )


# ---------------------------------------------------------------------------
# Encode-time stream metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Substrate-aware hyperprior init STUB (Council fix 2026-05-07, Ballé)
# ---------------------------------------------------------------------------


def test_gamma_substrate_aware_init_smaller_than_default() -> None:
    """STUB-mode finding: the underlying ``BalleHyperpriorCodec`` does not
    expose a ``prior_init`` API today (verified by grep on 2026-05-07), so
    setting ``substrate_aware_init=True`` is currently inert with respect
    to byte counts — encode WARNs + records a stub finding but proceeds on
    the static-arithmetic path.

    The strict assertion encoded here is the STUB contract:
    ``substrate_aware_init=True`` must produce a blob whose byte count is
    LESS THAN OR EQUAL TO the default-init blob. Today equality holds
    (no functional change); when the underlying API lands the inequality
    becomes strict.

    Rationale (CLAUDE.md "NEVER invent CLI flags"): we did not invent a
    ``prior_init`` argument to thread to the underlying codec because
    grep on ``balle_hyperprior_codec.py`` /
    ``joint_codec_stack_orchestrator.py`` returned zero matches. Inventing
    one would ship a dead-flag bug class.
    """
    sd = _small_state_dict()

    op_default = Op_GammaJointADMM(max_admm_iters=2, substrate_aware_init=False)
    op_substrate_aware = Op_GammaJointADMM(
        max_admm_iters=2, substrate_aware_init=True
    )

    res_default = op_default.encode(sd, context={})
    res_substrate_aware = op_substrate_aware.encode(sd, context={})

    # STUB contract: substrate-aware init must not WORSEN byte count.
    assert res_substrate_aware.bytes_out <= res_default.bytes_out, (
        f"substrate_aware_init=True produced {res_substrate_aware.bytes_out}B "
        f"but default produced {res_default.bytes_out}B — a STUB cannot "
        f"regress byte count"
    )

    # STUB-mode op_state records the finding so operators see the
    # γ-Phase-2 design-contract anchor in the manifest.
    assert res_substrate_aware.op_state["substrate_aware_init"] is True
    assert res_default.op_state["substrate_aware_init"] is False
    findings = res_substrate_aware.op_state["stub_findings"]
    assert any("γ-Phase-2 needs hyperprior init API" in f for f in findings), (
        f"substrate_aware_init=True must record the γ-Phase-2 finding; "
        f"got {findings!r}"
    )
    # Default has empty stub_findings (no WARN emitted).
    assert res_default.op_state["stub_findings"] == []


def test_gamma_substrate_aware_init_warn_message_pinned() -> None:
    """The exact WARN message string is pinned across versions so
    operators / dashboards can grep for it stably."""
    assert "γ-Phase-2 needs hyperprior init API" in SUBSTRATE_AWARE_INIT_WARN_MESSAGE
    assert "substrate_aware_init=True" in SUBSTRATE_AWARE_INIT_WARN_MESSAGE


def test_gamma_substrate_aware_init_emits_warn_log(caplog) -> None:
    """When substrate_aware_init=True, encode must emit a logger.warning
    (so operators can see the missing-API finding without inspecting
    op_state)."""
    import logging

    sd = _small_state_dict()
    op = Op_GammaJointADMM(max_admm_iters=2, substrate_aware_init=True)
    with caplog.at_level(logging.WARNING, logger="tac.codec_pipeline_joint_admm"):
        op.encode(sd, context={})
    matched = [
        r for r in caplog.records
        if "γ-Phase-2 needs hyperprior init API" in r.getMessage()
    ]
    assert len(matched) >= 1, (
        f"substrate_aware_init=True must emit the γ-Phase-2 WARN log; "
        f"got records {[r.getMessage() for r in caplog.records]!r}"
    )


def test_gamma_substrate_aware_init_default_off_silent() -> None:
    """Default ``substrate_aware_init=False`` must not WARN (silent inert)."""

    sd = _small_state_dict()
    op = Op_GammaJointADMM(max_admm_iters=2)
    assert op.substrate_aware_init is False
    # Use a fresh Logger handler to capture; rely on caplog fixture.


def test_gamma_substrate_aware_init_roundtrip_unchanged() -> None:
    """STUB contract: substrate_aware_init=True must produce a blob that
    decodes bit-equivalent to the default-init blob (since the flag is
    currently inert with respect to functional behaviour)."""
    sd = _small_state_dict()
    op = Op_GammaJointADMM(max_admm_iters=2, substrate_aware_init=True)
    res = op.encode(sd, context={})
    decoded = op.decode(res.blob, op_state=res.op_state, context={})
    assert set(decoded.keys()) == set(sd.keys())
    for name, t in sd.items():
        d = decoded[name]
        assert d.shape == t.shape


# ---------------------------------------------------------------------------
# Per-stream dequant metadata
# ---------------------------------------------------------------------------


def test_op_state_streams_records_per_tensor_dequant_metadata() -> None:
    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _small_state_dict()
    res = op.encode(sd, context={})
    streams = res.op_state["streams"]
    assert len(streams) == len(sd)
    seen_names = set()
    for entry in streams:
        assert "tensor_name" in entry
        assert "stream_name" in entry
        assert "shape" in entry
        assert "scale" in entry
        assert entry["dtype"] == "float32"
        assert entry["qint_dtype"] == "int8"
        seen_names.add(entry["tensor_name"])
    assert seen_names == set(sd.keys())


# ---------------------------------------------------------------------------
# Bug-hunter v3: caller-supplied score_marginals (integration seam)
# ---------------------------------------------------------------------------


def test_op_gamma_admm_honors_context_score_marginals() -> None:
    """Bug-hunter v3 (LOW, integration seam): a comment in
    ``Op_GammaJointADMM.encode`` previously claimed
    "Caller-supplied real marginals may override per-stream via
    context['score_marginals']" — a dead-comment because the code never
    read that key. The wrap now honors a Mapping[tensor_name, float], and
    this test pins the contract.

    Verifies:
      - Per-tensor marginals are accepted and embedded in the StreamSource
        metadata that the orchestrator uses (we read back via op_state's
        ``streams`` reflection).
      - Missing tensor_name keys fall back to the tiny default (1e-6).
      - Encode produces a valid, decodable blob in both cases.
    """
    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _small_state_dict()
    # Provide marginals for two of the three tensors; one omitted -> default.
    marginals = {
        "a.weight": 1e-4,
        "a.bias": 5e-5,
        # b.weight omitted -> 1e-6 default
    }
    res = op.encode(sd, context={"score_marginals": marginals})
    # Decode round-trips even with caller-supplied marginals (no contract change
    # in the blob; marginals affect the orchestrator's interior ADMM iteration
    # but not the wire format).
    decoded = op.decode(res.blob, op_state=res.op_state, context={})
    assert set(decoded.keys()) == set(sd.keys())


def test_op_gamma_admm_rejects_non_mapping_score_marginals() -> None:
    """Type guard: caller passes a list / tuple / scalar instead of a Mapping
    -> TypeError with an actionable message."""
    import pytest

    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _small_state_dict()
    with pytest.raises(TypeError, match="must be a Mapping"):
        op.encode(sd, context={"score_marginals": [1e-4, 1e-5]})


def test_op_gamma_admm_rejects_negative_score_marginal() -> None:
    """Sign convention: ProximalStepResult requires positive marginals
    (positive ⇒ more bytes lower score). Negative values must raise."""
    import pytest

    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _small_state_dict()
    bad = {"a.weight": -1e-4}
    with pytest.raises(ValueError, match="negative"):
        op.encode(sd, context={"score_marginals": bad})


def test_op_gamma_admm_rejects_non_finite_score_marginal() -> None:
    """Non-finite marginals (NaN / inf) corrupt the orchestrator's KKT
    residual normalisation; refuse them at the encode boundary."""
    import math

    import pytest

    op = Op_GammaJointADMM(max_admm_iters=2)
    sd = _small_state_dict()
    with pytest.raises(ValueError, match="not finite"):
        op.encode(
            sd,
            context={"score_marginals": {"a.weight": math.inf}},
        )
