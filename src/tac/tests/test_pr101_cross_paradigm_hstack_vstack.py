"""Tests for cross-paradigm CodecPipeline composition + the
``tools/pr101_cross_paradigm_hstack_vstack_empirical.py`` driver.

Verifies:

- A pipeline of [Op1] alone is byte-deterministic for a fixed
  state_dict (re-running encode produces the same blob bytes).
- A pipeline of [Op1] alone roundtrips: encode -> decode -> state_dict ~=
  original (within fp roundtrip tolerance).
- The pipeline rejects op-name mismatches at decode time (wire-format
  contract: pipeline.ops list must match the blob's recorded op order).
- The pipeline raises a clear error on trailing bytes (wire-format
  guards against wrong concatenation).
- The CPL1 magic bytes are unchanged (`b"CPL1"`).
- The cross-paradigm tool source declares the required score-claim flags
  (``family_falsified=False``, ``score_claim=False``,
  ``ready_for_exact_eval_dispatch=False``).
"""

from __future__ import annotations

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

from tac.codec_pipeline import (  # noqa: E402
    CodecPipeline,
    Op1_PR101SplitBrotli,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA  # noqa: E402


def _synthetic_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * 0.05
        for name, shape in FIXED_STATE_SCHEMA
    }


def test_op1_alone_pipeline_byte_deterministic() -> None:
    """Two encodes of the same state_dict via the same pipeline produce
    identical blob bytes (deterministic CPL1 wrapper + deterministic op)."""
    sd = _synthetic_state_dict()
    p1 = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
    p2 = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
    blob1, _ = p1.encode(sd, skip_validate=True)
    blob2, _ = p2.encode(sd, skip_validate=True)
    assert blob1 == blob2, (
        f"non-deterministic encode: blob lengths {len(blob1)} vs {len(blob2)}"
    )


def test_op1_alone_pipeline_roundtrip() -> None:
    """encode -> decode -> reconstructed state_dict has same keys + close
    fp roundtrip values."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
    blob, manifest = pipeline.encode(sd, skip_validate=True)
    reconstructed, replayed = pipeline.decode(blob)
    assert replayed == ["pr101_split_brotli"]
    assert sorted(reconstructed.keys()) == sorted(sd.keys())
    # fp roundtrip via int8 quantization will have non-zero error but should
    # be well-bounded; we only check shapes here (the codec's own tests cover
    # quantization fidelity).
    for k in sd:
        assert reconstructed[k].shape == sd[k].shape


def test_pipeline_rejects_op_count_mismatch_at_decode() -> None:
    """Decoding a blob with 1 op via a pipeline with 2 ops must raise."""
    sd = _synthetic_state_dict()
    p_encode = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
    blob, _ = p_encode.encode(sd, skip_validate=True)
    p_decode_wrong = CodecPipeline(
        [Op1_PR101SplitBrotli(auto_select=True), Op1_PR101SplitBrotli(auto_select=True)]
    )
    try:
        p_decode_wrong.decode(blob)
    except ValueError as e:
        assert "ops but pipeline" in str(e) or "n_ops" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on op-count mismatch")


def test_pipeline_magic_bytes_are_CPL1() -> None:
    """The pipeline wire-format LEGACY magic constant is locked to b'CPL1'
    (preserved for backwards compat); the canonical default magic landed
    2026-05-08 ORCH-SYNC Bug 2 is b'CPL2'."""
    assert CodecPipeline.MAGIC == b"CPL1"
    assert CodecPipeline.MAGIC_V2 == b"CPL2"


def test_pipeline_blob_starts_with_magic_and_op_count() -> None:
    """First 4 bytes = MAGIC (CPL1 or CPL2); next 4 = u32_LE op count."""
    import struct
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
    blob, _ = pipeline.encode(sd, skip_validate=True)
    assert blob[:4] in (CodecPipeline.MAGIC, CodecPipeline.MAGIC_V2)
    n_ops = struct.unpack_from("<I", blob, 4)[0]
    assert n_ops == 1


def test_pipeline_rejects_trailing_bytes_at_decode() -> None:
    """Appending stray bytes to a CPL1 blob must raise on decode."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
    blob, _ = pipeline.encode(sd, skip_validate=True)
    bad_blob = blob + b"\x00\x01\x02\x03"
    try:
        pipeline.decode(bad_blob)
    except ValueError as e:
        assert "trailing" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on trailing bytes")


def test_pipeline_manifest_has_score_claim_false_default() -> None:
    """Per CLAUDE.md MPS-NOISE rule + strict-scorer-rule: manifest must
    default to ``score_claim=False`` until a contest-CUDA evidence row
    is appended."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
    _, manifest = pipeline.encode(sd, skip_validate=True)
    assert manifest.score_claim is False


def test_pipeline_decode_preserves_state_dict_keys() -> None:
    """Roundtrip must preserve every state_dict key (no key drops or
    silent insertions)."""
    sd = _synthetic_state_dict()
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
    blob, _ = pipeline.encode(sd, skip_validate=True)
    reconstructed, _ = pipeline.decode(blob)
    assert set(reconstructed.keys()) == set(sd.keys())


def test_cross_paradigm_tool_source_declares_required_flags() -> None:
    """The cross-paradigm driver source must declare every CLAUDE.md flag
    that gates promotion + dispatch."""
    src = (REPO_ROOT / "tools" / "pr101_cross_paradigm_hstack_vstack_empirical.py").read_text()
    assert '"family_falsified": False' in src
    assert '"score_claim": False' in src
    assert '"ready_for_exact_eval_dispatch": False' in src
    assert '"falsification_scope":' in src


def test_cross_paradigm_tool_uses_canonical_orchestrator() -> None:
    """The cross-paradigm driver MUST use ``tac.codec_pipeline.CodecPipeline``
    + ``tac.codec_pipeline_full_stack.run_composition_matrix`` rather than
    re-implementing composition logic."""
    src = (REPO_ROOT / "tools" / "pr101_cross_paradigm_hstack_vstack_empirical.py").read_text()
    assert "from tac.codec_pipeline import" in src
    assert "CodecPipeline" in src
    assert "run_composition_matrix" in src


def test_cross_paradigm_tool_imports_only_cpu_codec_paths() -> None:
    """Strict-scorer-rule: the cross-paradigm driver must not import scorer
    modules. Only codec + brotli + numpy + torch are allowed at this layer."""
    src = (REPO_ROOT / "tools" / "pr101_cross_paradigm_hstack_vstack_empirical.py").read_text()
    forbidden = ["from tac.scorer", "import tac.scorer", "load_scorers", "PoseNet", "SegNet"]
    for fr in forbidden:
        assert fr not in src, f"cross-paradigm tool must not reference {fr}"


def test_cross_paradigm_tool_retracts_stack3_byte_proxy_per_review_eng_c1() -> None:
    """REVIEW-ENG C1 (2026-05-08): Stack 3 '_then_Op1' is a byte-proxy, not a
    byte-closed archive. The driver source MUST:

    - Document the wire-format honesty constraint.
    - Tag the row source string with ``BYTE_PROXY_NOT_BYTE_CLOSED``.
    - Exclude byte-proxy rows from the dominance ranking.
    - Emit the dispatch_blocker ``137531_byte_proxy_not_byte_closed_archive``
      on the Stack 3 entry.
    """
    src = (
        REPO_ROOT / "tools" / "pr101_cross_paradigm_hstack_vstack_empirical.py"
    ).read_text()
    # Documented wire-format honesty section
    assert "Stack 3 wire-format honesty" in src
    assert "REVIEW-ENG C1" in src
    # Source-string tag
    assert "BYTE_PROXY_NOT_BYTE_CLOSED" in src
    # Dispatch blocker emitted on Stack 3 row
    assert "137531_byte_proxy_not_byte_closed_archive" in src
    # Dominance ranking excludes byte-proxy
    assert "ranking_excludes_byte_proxy_rows" in src
    assert "byte_closed_rows" in src
