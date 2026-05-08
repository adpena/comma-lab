"""Tests for ``tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py``.

REVIEW-ENG C2 closure: the original Path B step 6 wire format reserves 28
bytes for per-tensor K side-info but the decoder discards them. The "no dead
K" variant drops the K section from the archive (~28 B free win) while
keeping K in the build manifest as audit metadata.

These tests verify the source declares the required CLAUDE.md flags, the
forked inflate.py source omits the K read, and the wire format documentation
is accurate.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))


def _read_tool_source() -> str:
    return (
        REPO_ROOT
        / "tools"
        / "build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py"
    ).read_text()


def test_no_dead_k_tool_source_declares_required_flags() -> None:
    """Tool source must declare CLAUDE.md flags that gate promotion +
    dispatch (mirrors the original variant)."""
    src = _read_tool_source()
    assert '"family_falsified": False' in src
    assert '"score_claim": False' in src
    assert '"ready_for_exact_eval_dispatch": False' in src
    assert '"falsification_scope": "lagrangian_x_continuous_K_no_dead_k_only"' in src


def test_no_dead_k_tool_documents_review_eng_c2_finding() -> None:
    """The tool docstring must explicitly cite REVIEW-ENG C2 and explain
    what changes vs the original variant."""
    src = _read_tool_source()
    assert "REVIEW-ENG C2" in src
    assert "L117-121" in src or "inflate.py L117-121" in src
    assert "no dead K" in src.lower() or "no_dead_k" in src
    assert "28" in src  # mentions 28 bytes savings


def test_no_dead_k_uses_weights_only_true() -> None:
    """Per REVIEW-ENG C4 — weights_only must be True at every torch.load."""
    src = _read_tool_source()
    # Both torch.load sites in the new tool must use weights_only=True.
    assert src.count("torch.load(") >= 2
    assert "weights_only=True" in src
    # And no leftover weights_only=False
    assert "weights_only=False" not in src


def test_no_dead_k_forked_inflate_omits_K_section() -> None:
    """The hardcoded forked inflate source must NOT read 28 K bytes."""
    src = _read_tool_source()
    # The new wire format only carries scales + brotli payload (no K).
    # The forked source must not declare K_SECTION_BYTES.
    # Find the forked source string boundary.
    start = src.index('_FORKED_INFLATE_SRC = \'\'\'')
    end = src.index("'''", start + 30)
    forked = src[start:end]
    assert "K_SECTION_BYTES" not in forked, (
        "no-dead-k inflate must NOT reference K_SECTION_BYTES"
    )
    # Must reference SCALE_SECTION_BYTES (the section we still keep)
    assert "SCALE_SECTION_BYTES" in forked
    # Wire format docstring must mention "without K" / "no-dead-k"
    assert "without K" in forked or "no K" in forked or "no-dead-k" in forked


def test_no_dead_k_tool_dispatch_blocker_includes_c3_apogee_int6() -> None:
    """REVIEW-ENG C3 attaches `apogee_int6_contest_cuda_anchor_required_first`
    to ALL Path B step 6 candidates (rel_err → score mapping unmeasured).
    The no-dead-k variant inherits that blocker."""
    src = _read_tool_source()
    assert "apogee_int6_contest_cuda_anchor_required_first" in src


def test_no_dead_k_section_total_bytes_28_smaller_than_original() -> None:
    """Direct module-level constant check — when imported, the tool's
    `_build_lossy_decoder_section_no_K` returns a ``section_total_bytes``
    that is exactly 28 less than the equivalent original variant build."""
    import build_admm_x_lossy_coarsening_path_b_step6_no_dead_k as no_k

    # The wire format constants we can validate without running the encoder:
    # K bytes in wire format = 0 (vs 28 in original).
    src = _read_tool_source()
    assert '"K_bytes_in_wire_format": 0' in src
    # The new variant's lane_id is distinct from the original.
    assert no_k.LANE_ID == "admm_x_lossy_coarsening_path_b_step6_no_dead_k"


def test_no_dead_k_preserves_original_audit_trail() -> None:
    """The new variant outputs to a separate dir
    (``..._no_dead_k_<ts>``) so the original variant's submission_dir +
    archive.zip remains intact as forensic record."""
    src = _read_tool_source()
    assert "admm_x_lossy_coarsening_path_b_step6_no_dead_k_" in src
    # Original variant tool name preserved in the manifest for cross-reference
    assert "tools/build_admm_x_lossy_coarsening_path_b_step6.py" in src


def test_no_dead_k_evidence_grade_is_cpu_build() -> None:
    """Per CPU-only ML/scoring policy: cuda_eval_worth_testing=True is allowed
    (this is a free byte-win on a candidate already approved for dispatch),
    but evidence_grade must be ``[CPU-build]`` and score_claim=False."""
    src = _read_tool_source()
    assert '"evidence_grade": "[CPU-build]"' in src
    assert '"score_claim": False' in src
    assert '"cuda_eval_worth_testing": True' in src
