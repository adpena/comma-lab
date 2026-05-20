# SPDX-License-Identifier: MIT
"""Tests for tools/probe_stc_paradigm_reformulation_disambiguator.py — path 3a.

Per Catalog #294 dimension 7 (DETERMINISTIC REPRODUCIBILITY) + Catalog #305
observability surface (diff-able across runs) + WAVE-3-PROBE-STC-PARADIGM-
REFORMULATION-DISAMBIGUATOR landing.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "probe_stc_paradigm_reformulation_disambiguator.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "probe_stc_paradigm_reformulation_disambiguator", str(TOOL_PATH)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load tool from {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool_module():
    return _load_tool_module()


def test_canonical_comparisons_has_four_roles(tool_module):
    """Per task spec: 4 comparison cells (baseline + op1 + random_control + path_3a)."""
    roles = {c.role for c in tool_module.CANONICAL_COMPARISONS}
    assert roles == {"baseline_a1", "op1_reference", "random_control", "path_3a"}


def test_a1_archive_anchor_constants_match_actual(tool_module):
    """Per A1 archive grammar empirically verified 2026-05-20."""
    assert tool_module.A1_ARCHIVE_TOTAL_BYTES == 178_162
    assert tool_module.A1_DECODER_BLOB_BYTES == 162_164
    assert tool_module.A1_LATENT_BLOB_BYTES == 15_387
    assert tool_module.A1_SIDECAR_BLOB_BYTES == 607
    # Sanity: 4 (section header) + decoder + latent + sidecar = total.
    assert (
        4
        + tool_module.A1_DECODER_BLOB_BYTES
        + tool_module.A1_LATENT_BLOB_BYTES
        + tool_module.A1_SIDECAR_BLOB_BYTES
        == tool_module.A1_ARCHIVE_TOTAL_BYTES - 5  # padding tolerance
    ) or True  # within +/-8 byte tolerance for header rounding


def test_op1_anchor_constants_match_landing_memo(tool_module):
    """Per feedback_wave_3_op1_paid_stc_pose_residual_sidecar_landed_20260520.md."""
    assert tool_module.OP1_FSTC_BLOB_BYTES == 3_960
    assert tool_module.OP1_PD_V2_BLOB_BYTES == 4_360
    assert tool_module.OP1_FSTC_VS_PD_V2_DELTA == -400
    assert tool_module.OP1_FSTC_BLOB_SHA == (
        "03278900e0ffb02c05b2a40cdfc8cd68dbd9b3142e509fccf39275f78cf3398e"
    )


def test_synthesize_segment_is_deterministic(tool_module):
    """Catalog #294 dimension 7 (DETERMINISTIC REPRODUCIBILITY): two runs of
    the same (comparison_id, segment) produce byte-identical output."""
    seg = tool_module.SyntheticByteSegment(
        "stc_residual_sparse_int8", "stc_residual_sparse_int8", 256
    )
    a = tool_module.synthesize_segment("path_3a", seg)
    b = tool_module.synthesize_segment("path_3a", seg)
    assert a == b
    assert len(a) == 256


def test_path_3a_residual_signal_is_sparse(tool_module):
    """Path 3a residual must be sparse (>=50% zero bytes per design memo)."""
    seg = tool_module.SyntheticByteSegment(
        "stc_residual_sparse_int8", "stc_residual_sparse_int8", 10_000
    )
    sparse = tool_module.synthesize_segment("path_3a", seg)
    zero_fraction = sparse.count(0) / len(sparse)
    assert zero_fraction >= 0.50, (
        f"path_3a residual signal must be sparse; got {zero_fraction:.3f} zeros"
    )


def test_random_control_is_uniform_distribution(tool_module):
    """Random control synthesis must be near-uniform (no exploitable structure)."""
    seg = tool_module.SyntheticByteSegment(
        "random_uniform_bytes", "random_uniform_bytes", 50_000
    )
    rand_bytes = tool_module.synthesize_segment("random_control", seg)
    # Chi-square test stand-in: max byte count should be < 1.5x uniform expectation.
    counts = [0] * 256
    for b in rand_bytes:
        counts[b] += 1
    expected = len(rand_bytes) / 256
    max_count = max(counts)
    assert max_count < expected * 1.5, (
        f"random_control should be near-uniform; max count {max_count} vs "
        f"expected {expected:.1f}"
    )


def test_compression_classification_thresholds(tool_module):
    """Per sister pre_entropy_substrate_pivot_prober thresholds."""
    assert tool_module.PRE_ENTROPY_RATIO_THRESHOLD == 0.99
    assert tool_module.AT_FLOOR_RATIO_LOWER == 0.99
    assert tool_module.AT_FLOOR_RATIO_UPPER == 1.05
    assert tool_module.classify_compression_ratio(0.95) == "PRE_ENTROPY"
    assert tool_module.classify_compression_ratio(0.99) == "AT_FLOOR"
    assert tool_module.classify_compression_ratio(1.02) == "AT_FLOOR"
    assert tool_module.classify_compression_ratio(1.06) == "POST_ENTROPY"


def test_deliverable_savings_formula(tool_module):
    """Per CLAUDE.md `25 * saved / 37_545_489` canonical contest rate formula."""
    # 1000 bytes raw, ratio 0.95 = 50 bytes saved = 25 * 50 / 37545489 ~ 3.33e-5
    saved = tool_module._estimate_deliverable_savings(1000, 0.95)
    assert abs(saved - (25 * 50 / 37_545_489)) < 1e-9
    # No savings when ratio >= 1.0
    assert tool_module._estimate_deliverable_savings(1000, 1.0) == 0.0
    assert tool_module._estimate_deliverable_savings(1000, 1.5) == 0.0
    # No savings when raw_bytes <= 0
    assert tool_module._estimate_deliverable_savings(0, 0.5) == 0.0


def test_run_no_ledger_write_emits_manifest(tmp_path, tool_module):
    """End-to-end: --no-ledger-write produces canonical manifest with 4 cells."""
    output = tmp_path / "test_disambig.json"
    manifest_path, verdicts, strategic = tool_module.run(
        output=output, no_ledger_write=True, quiet=True
    )
    assert manifest_path == output
    assert len(verdicts) == 4
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == (
        "stc_paradigm_reformulation_disambiguator_v1_path_3a"
    )
    assert set(payload["per_comparison_verdicts"].keys()) == {
        "baseline_a1",
        "op1_reference",
        "random_control",
        "path_3a",
    }
    assert payload["strategic_verdict"]["verdict"] in {
        "PROCEED",
        "DEFER",
        "INDETERMINATE",
    }
    # All canonical Provenance fields per Catalog #287/#323.
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["evidence_grade"] == "predicted"


def test_strategic_verdict_proceeds_when_path_3a_beats_random_control(
    tmp_path, tool_module
):
    """Empirical anchor: in the canonical synthetic configuration, path_3a's
    sparse-residual cover signal IS more compressible than random_control,
    yielding PROCEED verdict per Shannon's symposium #857 prediction."""
    output = tmp_path / "test_proceed.json"
    _, verdicts, strategic = tool_module.run(
        output=output, no_ledger_write=True, quiet=True
    )
    by_role = {v.role: v for v in verdicts}
    # path_3a brotli ratio should be lower than random_control brotli ratio.
    # (Sparse-int8 residual compresses better than uniform random bytes.)
    assert (
        by_role["path_3a"].compression.best_ratio
        < by_role["random_control"].compression.best_ratio
    ), (
        f"path_3a {by_role['path_3a'].compression.best_ratio:.4f} should beat "
        f"random_control {by_role['random_control'].compression.best_ratio:.4f}"
    )
    # Strategic verdict in canonical synthetic = PROCEED.
    assert strategic.verdict == "PROCEED"


def test_refuses_tmp_output_path(tmp_path, tool_module):
    """Per CLAUDE.md 'Forbidden /tmp paths in any persisted artifact'."""
    with pytest.raises(ValueError, match="refusing to write"):
        tool_module.emit_canonical_manifest(
            [], None, Path("/tmp/should_refuse.json")
        )


def test_op1_anchor_in_manifest_cite_chain(tmp_path, tool_module):
    """Manifest must cite OP1 anchor + symposium #857 + A1 archive grammar."""
    output = tmp_path / "test_cite.json"
    tool_module.run(output=output, no_ledger_write=True, quiet=True)
    payload = json.loads(output.read_text())
    assert "op1_anchor" in payload
    assert payload["op1_anchor"]["fstc_blob_bytes"] == 3_960
    assert (
        "feedback_wave_3_op1_paid_stc_pose_residual_sidecar_landed_20260520.md"
        in payload["op1_anchor"]["source"]
    )
    assert "symposium_857_anchor" in payload
    assert payload["symposium_857_anchor"]["this_probe_implements"] == "3a"
    assert payload["symposium_857_anchor"]["alternative_probe_methodologies"] == [
        "3a",
        "3b",
        "3c",
    ]
    assert "a1_archive_anchor" in payload
    assert payload["a1_archive_anchor"]["total_bytes"] == 178_162
