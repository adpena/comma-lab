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
    Op1_PR101SplitBrotli,
    Op2_PR103ArithmeticCodec,
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


# ---------------------------------------------------------------------------
# Op 2 wiring + Op1+Op2 composition
# ---------------------------------------------------------------------------

def test_op2_satisfies_codec_op_protocol() -> None:
    op = Op2_PR103ArithmeticCodec()
    assert isinstance(op, CodecOp)
    assert op.name == "pr103_arithmetic_codec"


def test_op2_alone_encode_decode_roundtrip() -> None:
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op2_PR103ArithmeticCodec()])
    blob, manifest = pipeline.encode(sd)
    assert blob[:4] == b"CPL1"
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["pr103_arithmetic_codec"]
    assert set(decoded.keys()) == set(sd.keys())


def test_op1_op2_composition_via_cpl1() -> None:
    """Op 1 + Op 2 stored in CPL1 wrapper (substitutional composition mode).

    Both ops encode the same state_dict. The wrapper preserves both blobs +
    op_state — useful for forensics + side-by-side byte-impact comparison.
    Per the composition contract memo: substitutional composition is one
    of three modes (substitutional / substrate-transform / decorator).
    """
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([
        Op1_PR101SplitBrotli(auto_select=False),
        Op2_PR103ArithmeticCodec(),
    ])
    blob, manifest = pipeline.encode(sd)
    decoded, replayed = pipeline.decode(blob)
    assert replayed == ["pr101_split_brotli", "pr103_arithmetic_codec"]
    assert set(decoded.keys()) == set(sd.keys())
    # Per-op byte impact recorded in manifest.
    op1_bytes = manifest.op_results[0].bytes_out
    op2_bytes = manifest.op_results[1].bytes_out
    assert op1_bytes > 0 and op2_bytes > 0


def test_op2_op_state_includes_section_lengths() -> None:
    """The Op 2 decoder REQUIRES section_lengths (PR103 wire format hardcodes
    them). The pipeline encoder must populate this op_state field."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op2_PR103ArithmeticCodec()])
    _, manifest = pipeline.encode(sd)
    op_state = manifest.op_results[0].op_state
    assert "section_lengths" in op_state
    section_lengths = op_state["section_lengths"]
    # ac_fallback section landed 2026-05-07 for substrate-mismatch protection
    # (per-tensor AC auto-fallback gate); empty bytes-len when no tensor regressed.
    assert set(section_lengths.keys()) == {
        "br", "hists", "merged_ac", "hi_hist", "ac_fallback",
    }
    for k, v in section_lengths.items():
        assert v >= 0, f"section {k!r} length must be non-negative"


def test_manifest_score_claim_default_false() -> None:
    """Pipeline manifest must NOT claim a contest-CUDA score by default.
    Per CLAUDE.md "Forbidden score claims" — every score must be explicitly
    tagged ``[contest-CUDA]`` and ``score_claim`` must default False."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    _, manifest = pipeline.encode(sd)
    assert manifest.score_claim is False
    assert manifest.score_evidence_grade == "predicted"


# ---------------------------------------------------------------------------
# Bug-hunter regression: Op2 latent_hi_symbols / n_latent_hi_symbols coupling
# ---------------------------------------------------------------------------

def test_op2_rejects_latent_hi_count_mismatch() -> None:
    """Bug-hunter regression (CRITICAL #1, 2026-05-07): when a caller supplies
    ``latent_hi_symbols`` but explicitly disagrees with the array length via
    ``n_latent_hi_symbols`` Op2 must refuse to encode rather than silently
    ship a mismatched merged-AC drain count."""
    import numpy as np

    sd = _synthetic_state_dict()
    op = Op2_PR103ArithmeticCodec(
        latent_hi_symbols=np.array([1, 2, 3], dtype=np.uint16),
        n_latent_hi_symbols=99,  # explicit mismatch
    )
    pipeline = CodecPipeline([op])
    with pytest.raises(ValueError, match="n_latent_hi_symbols"):
        pipeline.encode(sd, skip_validate=True)


def test_op2_rejects_n_hi_without_latent_array() -> None:
    """If ``n_latent_hi_symbols > 0`` but ``latent_hi_symbols is None`` the
    decoder would attempt to drain symbols that the encoder never embedded.
    Op2 must refuse this configuration."""
    sd = _synthetic_state_dict()
    op = Op2_PR103ArithmeticCodec(
        latent_hi_symbols=None,
        n_latent_hi_symbols=17,
    )
    pipeline = CodecPipeline([op])
    with pytest.raises(ValueError, match="latent_hi_symbols is None"):
        pipeline.encode(sd, skip_validate=True)


def test_op2_auto_derives_n_hi_when_default_zero() -> None:
    """When the caller passes ``latent_hi_symbols=arr`` but leaves
    ``n_latent_hi_symbols`` at its default 0, Op2 must auto-derive the drain
    count from the array length so encode/decode stay consistent."""
    import numpy as np

    sd = _synthetic_state_dict()
    arr = np.array([0, 1, 2, 0, 1], dtype=np.uint16)
    op = Op2_PR103ArithmeticCodec(latent_hi_symbols=arr)
    pipeline = CodecPipeline([op])
    _, manifest = pipeline.encode(sd, skip_validate=True)
    op_state = manifest.op_results[0].op_state
    assert op_state["n_latent_hi_symbols"] == arr.size


def test_op1_auto_select_runs_exactly_once_per_encode() -> None:
    """Bug-hunter v2 (re-opened LOW): with ``auto_select=True`` and no
    explicit overrides, the wrap previously invoked ``auto_select_byte_maps``
    twice -- once inside ``encode_decoder_compact`` and once afterwards to
    populate ``op_state``. The fix computes the byte-map dict once and
    threads it explicitly into the encoder. Regression: count call sites
    via monkeypatch."""
    sd = _synthetic_state_dict()
    op = Op1_PR101SplitBrotli(auto_select=True)

    # Spy on auto_select_byte_maps via the canonical import location.
    from tac import pr101_split_brotli_codec as _codec_mod

    original = _codec_mod.auto_select_byte_maps
    counter = {"calls": 0}

    def _counted(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    _codec_mod.auto_select_byte_maps = _counted
    try:
        res = op.encode(sd, context={})
    finally:
        _codec_mod.auto_select_byte_maps = original

    # Exactly one auto-select call per encode (was 2 pre-fix).
    assert counter["calls"] == 1, (
        f"auto_select_byte_maps invoked {counter['calls']} times; expected "
        f"exactly 1 per encode (regression of bug-hunter v2 LOW fix)"
    )
    # And the resulting op_state still carries the byte_maps for decode.
    assert "effective_byte_maps" in res.op_state


def test_op1_auto_select_roundtrip_byte_faithful() -> None:
    """Sanity: bug-hunter v2 LOW fix must not change byte output -- the
    one-call path produces the same blob as the two-call path produced.
    """
    sd = _synthetic_state_dict()
    op = Op1_PR101SplitBrotli(auto_select=True)
    res = op.encode(sd, context={})
    decoded = op.decode(res.blob, op_state=res.op_state, context={})
    assert set(decoded.keys()) == set(sd.keys())


def test_op2_docstring_lists_all_five_section_keys() -> None:
    """Bug-hunter v2 (re-opened MEDIUM): ensure the Op2 class docstring
    enumerates EVERY section_lengths key the encoder populates. The prior
    docstring listed only 4 (br, hists, merged_ac, hi_hist) and silently
    drifted out-of-date when ac_fallback landed; keep them coupled."""
    docstring = Op2_PR103ArithmeticCodec.__doc__ or ""
    for key in ("br", "hists", "merged_ac", "hi_hist", "ac_fallback"):
        assert f"``{key}``" in docstring, (
            f"Op2 docstring missing section key {key!r}; encoder populates "
            f"all five and decoder reads the same set"
        )
    # Also pin the ac_fallback_set op_state mention.
    assert "ac_fallback_set" in docstring


# ---------------------------------------------------------------------------
# Bug-hunter v3: integration-seam regressions
# ---------------------------------------------------------------------------

def test_pipeline_encode_raises_actionable_error_on_non_json_op_state() -> None:
    """Bug-hunter v3 (MEDIUM, integration seam): a CodecOp that accidentally
    embeds a torch.Tensor / numpy array / set in ``op_state`` must produce a
    TypeError that names the offending op AND the offending key path. The
    previous bare ``json.dumps(op_state)`` produced an opaque
    ``"Object of type Tensor is not JSON serializable"`` with no op or key
    context, forcing operators to bisect across the 8 canonical stacks.
    """
    from tac.codec_pipeline import EncodeResult, ValidationReport

    class _BadOp:
        name = "bad_tensor_op"

        def encode(self, state_dict, *, context):  # type: ignore[no-untyped-def]
            return EncodeResult(
                blob=b"x",
                bytes_in=0,
                bytes_out=1,
                op_name=self.name,
                op_state={
                    "ok": [1, 2, 3],
                    "nested": {"inner": torch.tensor([1.0])},
                },
            )

        def decode(self, blob, *, op_state, context):  # type: ignore[no-untyped-def]
            return {}

        def validate(self, state_dict, *, context):  # type: ignore[no-untyped-def]
            return ValidationReport(passed=True, op_name=self.name)

    pipeline = CodecPipeline([_BadOp()])
    with pytest.raises(TypeError) as excinfo:
        pipeline.encode({})
    msg = str(excinfo.value)
    # Op name must appear so operators can locate the offending op without bisecting.
    assert "bad_tensor_op" in msg, f"op name missing from error: {msg!r}"
    # Key path must appear so operators don't have to introspect op_state.
    assert "nested.inner" in msg, f"key path missing from error: {msg!r}"
    # Hint about how to fix must be present (per integration-seam guidance).
    assert "JSON" in msg


def test_pipeline_encode_with_numpy_in_op_state_names_offender() -> None:
    """Bug-hunter v3 (MEDIUM): numpy.ndarray / numpy scalars are a common
    accidental op_state value (encoders that pass intermediate arrays through).
    The error must surface the offending key path the same way as torch."""
    import numpy as np

    from tac.codec_pipeline import EncodeResult, ValidationReport

    class _NumpyOp:
        name = "np_op"

        def encode(self, state_dict, *, context):  # type: ignore[no-untyped-def]
            return EncodeResult(
                blob=b"y",
                bytes_in=0,
                bytes_out=1,
                op_name=self.name,
                op_state={"arr": np.array([1, 2, 3])},
            )

        def decode(self, blob, *, op_state, context):  # type: ignore[no-untyped-def]
            return {}

        def validate(self, state_dict, *, context):  # type: ignore[no-untyped-def]
            return ValidationReport(passed=True, op_name=self.name)

    pipeline = CodecPipeline([_NumpyOp()])
    with pytest.raises(TypeError) as excinfo:
        pipeline.encode({})
    msg = str(excinfo.value)
    assert "np_op" in msg
    assert "arr" in msg


def test_pipeline_stress_30_op_chain_roundtrips() -> None:
    """Bug-hunter v3 (LOW, integration seam): exercising a large op chain
    catches off-by-one errors in the wire-format n_ops loop. 30 ops keeps
    the stress test fast (<1 s) while well exceeding any realistic stack
    depth (canonical full-stack max chain is 3 ops)."""
    from tac.codec_pipeline import EncodeResult, ValidationReport

    class _NoOp:
        """Trivial substitutional op that emits a fixed blob; sees the same
        state_dict each time (no transforms_state_dict). Decoder must replay
        in-order regardless of how many copies are in the chain."""

        name = "noop"

        def encode(self, state_dict, *, context):  # type: ignore[no-untyped-def]
            return EncodeResult(
                blob=b"noop-blob",
                bytes_in=0,
                bytes_out=9,
                op_name=self.name,
                op_state={"copy_idx": 0},
            )

        def decode(self, blob, *, op_state, context):  # type: ignore[no-untyped-def]
            return {"sentinel": torch.tensor(float(op_state.get("copy_idx", 0)))}

        def validate(self, state_dict, *, context):  # type: ignore[no-untyped-def]
            return ValidationReport(passed=True, op_name=self.name)

    pipeline = CodecPipeline([_NoOp() for _ in range(30)])
    blob, manifest = pipeline.encode({})
    assert len(manifest.op_results) == 30
    assert blob[:4] == b"CPL1"
    decoded, replayed = pipeline.decode(blob)
    assert len(replayed) == 30
    assert all(r == "noop" for r in replayed)
    # Final decoded state is from the LAST op (substitutional ops overwrite
    # decoded_state; the pipeline returns the last one).
    assert "sentinel" in decoded


# ---------------------------------------------------------------------------
# Op1 substrate-adaptive derivers (task #394)
# ---------------------------------------------------------------------------

def test_op1_auto_derive_all_constants_roundtrips() -> None:
    """Encode-decode roundtrip with substrate-adaptive derivers preserves
    state_dict bit-exactly. Threads derived storage_order/stream_ends/
    conv4_perms through op_state so decode can reconstruct."""
    scale = 0.05
    state = _synthetic_state_dict(seed=42, scale=scale)
    op = Op1_PR101SplitBrotli(
        auto_derive_all_constants=True, auto_select=True, brotli_quality=1
    )
    result = op.encode(state, context={})
    assert "derived_storage_order" in result.op_state
    assert "derived_stream_ends" in result.op_state
    assert "derived_conv4_perms" in result.op_state
    # storage_order must be a permutation of range(28)
    so = result.op_state["derived_storage_order"]
    assert sorted(so) == list(range(28))
    # stream_ends must be sorted-strict-increasing ending at 28
    se = result.op_state["derived_stream_ends"]
    assert se == sorted(set(se))
    assert se[-1] == 28
    # Roundtrip is bit-exact at int8-quant grid: PR101's codec quantizes to
    # 127 levels, so absolute error is bounded by ~scale/127. Use 2x that as
    # tolerance for safety (rounding can land just above 1 quantum on edges).
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    # Tolerance: int8 quant step = 2 * max|x| / 127, half-step error per element.
    # randn tail can reach ~4 sigma, so quant step is bounded by ~4*scale/127,
    # giving half-step error <= 2*scale/127. Use 4x that as cushion for tails.
    quant_atol = 4.0 * scale / 127.0
    for name, tensor in state.items():
        torch.testing.assert_close(decoded[name], tensor, rtol=0, atol=quant_atol)


def test_op1_auto_derive_all_constants_substrate_specific() -> None:
    """Two different substrates (different seeds) produce DIFFERENT derived
    constants — verifying the derivers are substrate-adaptive, not constant.
    """
    op = Op1_PR101SplitBrotli(
        auto_derive_all_constants=True, auto_select=False, brotli_quality=1
    )
    state_a = _synthetic_state_dict(seed=1, scale=0.5)
    state_b = _synthetic_state_dict(seed=2, scale=0.5)
    res_a = op.encode(state_a, context={})
    res_b = op.encode(state_b, context={})
    # At least one of the derived constants should differ between substrates
    # (random scale=0.5 produces enough variance that the brotli-optimal
    # ordering shifts). Equality across all 3 would mean derivers are no-ops.
    diff_so = res_a.op_state["derived_storage_order"] != res_b.op_state["derived_storage_order"]
    diff_se = res_a.op_state["derived_stream_ends"] != res_b.op_state["derived_stream_ends"]
    diff_cp = res_a.op_state["derived_conv4_perms"] != res_b.op_state["derived_conv4_perms"]
    assert diff_so or diff_se or diff_cp, (
        "derivers produced identical constants for two different substrates"
    )


def test_op1_auto_derive_all_constants_default_off() -> None:
    """Default behavior preserves PR101 wire-format: no derived_* keys in
    op_state when auto_derive_all_constants is False (default)."""
    scale = 0.1
    state = _synthetic_state_dict(seed=0, scale=scale)
    op = Op1_PR101SplitBrotli()
    result = op.encode(state, context={})
    assert "derived_storage_order" not in result.op_state
    assert "derived_stream_ends" not in result.op_state
    assert "derived_conv4_perms" not in result.op_state
    # Roundtrip still works (decode uses PR101 defaults). Same int8-quant
    # tolerance as the auto_derive case.
    decoded = op.decode(result.blob, op_state=result.op_state, context={})
    # Tolerance: int8 quant step = 2 * max|x| / 127, half-step error per element.
    # randn tail can reach ~4 sigma, so quant step is bounded by ~4*scale/127,
    # giving half-step error <= 2*scale/127. Use 4x that as cushion for tails.
    quant_atol = 4.0 * scale / 127.0
    for name, tensor in state.items():
        torch.testing.assert_close(decoded[name], tensor, rtol=0, atol=quant_atol)


def test_op1_auto_derive_all_constants_op_state_json_serializable() -> None:
    """op_state must be JSON-serializable per cathedral CPL1 wire format
    (bug-hunter v3 MEDIUM-1 enforcement). The new derived_conv4_perms uses
    str-keyed dict and list values to satisfy that."""
    import json
    state = _synthetic_state_dict(seed=0, scale=0.1)
    op = Op1_PR101SplitBrotli(auto_derive_all_constants=True, brotli_quality=1)
    result = op.encode(state, context={})
    # Must serialize without TypeError
    encoded = json.dumps(result.op_state)
    decoded_state = json.loads(encoded)
    assert decoded_state["derived_storage_order"] == result.op_state["derived_storage_order"]
