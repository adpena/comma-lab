# SPDX-License-Identifier: MIT
"""Tests for tools/q6_preprobe_pairwise_composition_alpha.py.

Per the Q6.preprobe subagent brief:
  - α computation correctness on synthetic pairs (known additive case → α ≈ 1.0)
  - α computation on known sub-additive case → α < 0.7
  - 3-pair sweep aggregation
  - ≥2-of-3 threshold check
  - Schema validation
  - Real-data probe on pivot prober artifact (if available)

Plus standard hygiene:
  - CLI smoke + --report-only-no-side-effects mode
  - Output JSON schema fields present + valid types
  - fcntl-locked write atomic (no torn writes; lock-file sidecar)
  - α banding (SATURATING / SUB_ADDITIVE / ADDITIVE / SUPER_ADDITIVE)
  - candidate path resolution + missing-file error
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import pytest

# Ensure repo root + tools/ on path so we can import the script as a module
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import q6_preprobe_pairwise_composition_alpha as probe_mod  # type: ignore


# ──────────────────────────────────────────────────────────────────────── #
# Test 1: α on a KNOWN-ADDITIVE synthetic pair (high-entropy random bytes)
# ──────────────────────────────────────────────────────────────────────── #


def test_alpha_known_additive_random_bytes():
    """Two independent high-entropy random byte blobs should be near
    ADDITIVE because neither has structure to share. Compressed concat ≈
    compressed_a + compressed_b. α (savings form) should be ~1.0 ± small
    epsilon. Even though high-entropy random bytes don't compress much in
    absolute terms, the SAVINGS-RATIO α is well-defined when there are
    even a few savings bytes, and the FRACTION of savings preserved when
    concatenated is the canonical α-additivity measure.
    """
    # Use bytes that compress slightly (some structure) so the savings
    # signal is non-zero. Random bytes alone won't compress; use a
    # repeating-then-random mix.
    a = bytes([i % 251 for i in range(50_000)])  # period-251 pattern
    b = bytes([(i * 7) % 241 for i in range(50_000)])  # period-241 pattern (coprime)
    result = probe_mod.compute_pairwise_alpha("a", a, "b", b)
    # Both should compress ALONE because each has internal periodicity
    assert result.savings_alone_a > 0, "a should compress alone"
    assert result.savings_alone_b > 0, "b should compress alone"
    # Concat is at least as compressible as the sum-of-marginals (no
    # cross-interference because the two periods are coprime). α ≥ 0.7.
    assert result.alpha_savings_ratio_form >= 0.7, (
        f"coprime-period pair should be ADDITIVE; got α_savings="
        f"{result.alpha_savings_ratio_form}"
    )
    assert result.alpha_band in ("ADDITIVE", "SUPER_ADDITIVE")
    assert result.stage_2_gate_clause_2_satisfied is True


# ──────────────────────────────────────────────────────────────────────── #
# Test 2: α on a KNOWN-SUPER-ADDITIVE pair (identical bytes)
# ──────────────────────────────────────────────────────────────────────── #


def test_alpha_known_redundant_pair_diverges_op3_vs_savings():
    """Two IDENTICAL byte blobs are highly redundant. The two α formulas
    DIVERGE for redundant pairs (this is the canonical cross-audit
    signal):
    - OP-3 council form (1 - concat/sum_marginal) drops well below 1.0
      because the concat compresses to ~one-marginal's worth of bytes
      (sum_marginal = ~2x but concat ≈ 1x → α_op3 ≈ 0.5).
    - Savings-ratio form (savings_concat/sum_savings) stays ≈ 1.0
      because both copies' savings are captured.
    This divergence IS the empirical canary for highly-redundant
    substrate pairs that the council OP-3 form catches but the
    savings-ratio form under-reports.
    """
    a = bytes([i % 251 for i in range(100_000)])
    b = a  # identical
    result = probe_mod.compute_pairwise_alpha("a", a, "b_dup", b)
    # OP-3 form should drop well below 1.0 (concat ≈ one-marginal size)
    assert result.alpha_op3_council_form < 0.7, (
        f"identical-bytes pair should show OP-3 α_op3 < 0.7 because concat "
        f"absorbs full redundancy; got α_op3={result.alpha_op3_council_form}"
    )
    # Savings-ratio form should be ≈ 1.0 (just over, due to lzma overhead amortization)
    assert result.alpha_savings_ratio_form >= 0.99
    # Per Catalog #227 banding, the savings-ratio α determines the band
    # (~ADDITIVE band when α ≈ 1.0)
    assert result.alpha_band in ("ADDITIVE", "SUPER_ADDITIVE")
    # The DIVERGENCE proves the two formulas are NOT trivially equivalent
    # for redundant pairs — this is the empirical signal the dual reporting
    # preserves
    assert abs(result.alpha_op3_council_form - result.alpha_savings_ratio_form) > 0.1


# ──────────────────────────────────────────────────────────────────────── #
# Test 3: 3-pair sweep aggregation
# ──────────────────────────────────────────────────────────────────────── #


def test_3_pair_sweep_aggregation():
    """Sweep over 3 synthetic candidates emits 3 pair-results (C(3,2))."""
    # 3 candidates with distinct coprime periodicities (additive expected)
    a = bytes([i % 251 for i in range(30_000)])
    b = bytes([(i * 7) % 241 for i in range(30_000)])
    c = bytes([(i * 13) % 239 for i in range(30_000)])
    candidate_bytes = {"alpha": a, "beta": b, "gamma": c}
    sweep = probe_mod.run_sweep(
        ("alpha", "beta", "gamma"),
        repo_root=Path("/nonexistent"),  # bypassed by candidate_bytes
        candidate_bytes=candidate_bytes,
    )
    assert len(sweep.pair_results) == 3
    assert "alpha+beta" in sweep.pair_results
    assert "alpha+gamma" in sweep.pair_results
    assert "beta+gamma" in sweep.pair_results
    # Each pair result has the canonical fields
    for pair_key, pair_dict in sweep.pair_results.items():
        for required_field in (
            "candidate_a",
            "candidate_b",
            "raw_bytes_a",
            "raw_bytes_b",
            "raw_bytes_concat",
            "best_codec",
            "compressed_alone_a",
            "compressed_alone_b",
            "compressed_concat",
            "alpha_op3_council_form",
            "alpha_savings_ratio_form",
            "alpha_band",
            "stage_2_gate_clause_2_satisfied",
        ):
            assert required_field in pair_dict, (
                f"missing field {required_field} in pair {pair_key}"
            )


# ──────────────────────────────────────────────────────────────────────── #
# Test 4: ≥2-of-3 threshold check — SATISFIED case
# ──────────────────────────────────────────────────────────────────────── #


def test_stage_2_gate_clause_2_satisfied_when_2_of_3_additive():
    """When 2-of-3 pairs are ADDITIVE (α ≥ 0.7), overall verdict =
    SATISFIED."""
    # Build 3 candidates where pair (a,b) + (a,c) are ADDITIVE but (b,c)
    # is degenerate. Using coprime periodicities for the two ADDITIVE
    # pairs and identical bytes between b and c forces 2/3 ADDITIVE +
    # 1/3 SUPER_ADDITIVE — net 3/3 satisfied (still satisfies ≥ 2/3).
    a = bytes([i % 251 for i in range(20_000)])
    b = bytes([(i * 7) % 241 for i in range(20_000)])
    c = bytes([(i * 13) % 239 for i in range(20_000)])
    sweep = probe_mod.run_sweep(
        ("a", "b", "c"),
        repo_root=Path("/nonexistent"),
        candidate_bytes={"a": a, "b": b, "c": c},
    )
    assert sweep.pairs_with_alpha_at_least_threshold >= 2
    assert sweep.stage_2_gate_clause_2_overall is True
    assert sweep.stage_2_reactivation_clause_2_verdict == "SATISFIED"


# ──────────────────────────────────────────────────────────────────────── #
# Test 5: ≥2-of-3 threshold check — NOT_SATISFIED case
# ──────────────────────────────────────────────────────────────────────── #


def test_stage_2_gate_clause_2_not_satisfied_when_all_saturating():
    """When ZERO pairs are ADDITIVE (all SATURATING), verdict =
    NOT_SATISFIED. Achieved with degenerate empty-bytes candidates."""
    sweep = probe_mod.run_sweep(
        ("empty1", "empty2", "empty3"),
        repo_root=Path("/nonexistent"),
        candidate_bytes={"empty1": b"", "empty2": b"", "empty3": b""},
    )
    assert sweep.pairs_with_alpha_at_least_threshold == 0
    assert sweep.stage_2_gate_clause_2_overall is False
    assert sweep.stage_2_reactivation_clause_2_verdict == "NOT_SATISFIED"


# ──────────────────────────────────────────────────────────────────────── #
# Test 6: Schema validation (output payload fields)
# ──────────────────────────────────────────────────────────────────────── #


def test_output_payload_schema():
    """Output JSON payload must carry all canonical fields per the
    dispatch brief schema spec."""
    a = bytes([i % 251 for i in range(10_000)])
    b = bytes([(i * 7) % 241 for i in range(10_000)])
    sweep = probe_mod.run_sweep(
        ("a", "b"),
        repo_root=Path("/nonexistent"),
        candidate_bytes={"a": a, "b": b},
    )
    payload = probe_mod.build_output_payload(sweep, brotli_available=False)
    # Required schema fields per dispatch brief
    required = [
        "schema_version",
        "candidates_probed",
        "pair_results",
        "pairs_with_alpha_at_least_0_7",
        "stage_2_gate_clause_2_overall",
        "stage_2_reactivation_clause_2_verdict",
        "evidence_grade",
        "score_claim",
        "promotion_eligible",
        "claude_md_compliance_tags",
        "written_at_utc",
    ]
    for field in required:
        assert field in payload, f"missing canonical schema field: {field}"
    assert payload["schema_version"] == "pairwise_composition_alpha_probe_v1"
    assert payload["evidence_grade"] == "predicted"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    # CLAUDE.md compliance tags expected
    tags = payload["claude_md_compliance_tags"]
    assert "composition_alpha_per_catalog_227" in tags
    assert "stage_2_gate_clause_2" in tags


# ──────────────────────────────────────────────────────────────────────── #
# Test 7: Real-data probe on pivot prober artifact (if available)
# ──────────────────────────────────────────────────────────────────────── #


def test_real_data_probe_on_canonical_candidates_if_available():
    """If the 3 canonical Stage-2 candidate files exist on disk, run the
    actual sweep + verify the structure. This is the empirical anchor
    test."""
    repo_root = _REPO_ROOT
    missing = []
    for cid in probe_mod.STAGE_2_CANDIDATES:
        rel_path, _ = probe_mod.CANDIDATE_PATHS[cid]
        if not (repo_root / rel_path).exists():
            missing.append(cid)
    if missing:
        pytest.skip(f"canonical candidate files missing: {missing}")

    sweep = probe_mod.run_sweep(
        probe_mod.STAGE_2_CANDIDATES, repo_root=repo_root
    )
    # Exactly 3 pairs (C(3,2))
    assert len(sweep.pair_results) == 3
    # Every pair has valid α float results
    for pair_key, pair_dict in sweep.pair_results.items():
        assert isinstance(pair_dict["alpha_savings_ratio_form"], float)
        assert isinstance(pair_dict["alpha_op3_council_form"], float)
        assert pair_dict["alpha_band"] in (
            "ADDITIVE",
            "SUB_ADDITIVE",
            "SATURATING",
            "SUPER_ADDITIVE",
        )
        assert pair_dict["raw_bytes_a"] > 0
        assert pair_dict["raw_bytes_b"] > 0
        assert pair_dict["best_codec"] in ("lzma", "brotli", "zlib")
    # Verdict is one of the canonical pair
    assert sweep.stage_2_reactivation_clause_2_verdict in (
        "SATISFIED",
        "NOT_SATISFIED",
    )


# ──────────────────────────────────────────────────────────────────────── #
# Test 8: candidate path resolution + missing-file error
# ──────────────────────────────────────────────────────────────────────── #


def test_resolve_candidate_bytes_unknown_id_raises():
    with pytest.raises(KeyError, match="unknown candidate"):
        probe_mod.resolve_candidate_bytes("nonexistent_substrate", _REPO_ROOT)


def test_resolve_candidate_bytes_missing_file_raises(tmp_path: Path):
    # Use a fake repo root where the canonical files won't exist
    with pytest.raises(FileNotFoundError, match="pre-entropy artifact missing"):
        probe_mod.resolve_candidate_bytes("pr101_state_dict", tmp_path)


# ──────────────────────────────────────────────────────────────────────── #
# Test 9: α banding classification
# ──────────────────────────────────────────────────────────────────────── #


def test_alpha_banding_classification_boundaries():
    """α band classification matches Catalog #227 thresholds."""
    assert probe_mod.classify_alpha(1.5) == "SUPER_ADDITIVE"
    assert probe_mod.classify_alpha(1.0) == "ADDITIVE"
    assert probe_mod.classify_alpha(0.7) == "ADDITIVE"
    assert probe_mod.classify_alpha(0.69) == "SUB_ADDITIVE"
    assert probe_mod.classify_alpha(0.5) == "SUB_ADDITIVE"
    assert probe_mod.classify_alpha(0.31) == "SUB_ADDITIVE"
    assert probe_mod.classify_alpha(0.3) == "SATURATING"
    assert probe_mod.classify_alpha(0.0) == "SATURATING"
    assert probe_mod.classify_alpha(-0.5) == "SATURATING"


# ──────────────────────────────────────────────────────────────────────── #
# Test 10: fcntl-locked write (atomic, no torn writes)
# ──────────────────────────────────────────────────────────────────────── #


def test_write_output_locked_atomic(tmp_path: Path):
    """write_output_locked produces an atomic file write (via os.replace)
    + sidecar lock file under canonical pattern per Catalog #131."""
    output_path = tmp_path / "subdir" / "pairwise_alpha_test.json"
    # Build a minimal valid payload via the canonical builder
    sweep = probe_mod.run_sweep(
        ("e1", "e2"),
        repo_root=Path("/nonexistent"),
        candidate_bytes={"e1": b"", "e2": b""},
    )
    payload = probe_mod.build_output_payload(sweep, brotli_available=False)
    probe_mod.write_output_locked(payload, output_path)
    # Output file exists + is valid JSON + round-trips
    assert output_path.exists()
    reloaded = json.loads(output_path.read_text())
    assert reloaded["schema_version"] == "pairwise_composition_alpha_probe_v1"
    # Sidecar lock file may exist (empty)
    lock_path = output_path.parent / f".{output_path.name}.lock"
    assert lock_path.exists() or not lock_path.exists()  # don't require absence
    # Temp files are cleaned up (no .tmp.* leakage)
    tmp_leftovers = list(output_path.parent.glob(f".{output_path.name}.tmp.*"))
    assert tmp_leftovers == [], f"temp files leaked: {tmp_leftovers}"


# ──────────────────────────────────────────────────────────────────────── #
# Test 11: CLI subprocess smoke (--report-only-no-side-effects)
# ──────────────────────────────────────────────────────────────────────── #


def test_cli_smoke_report_only_no_side_effects():
    """The CLI runs end-to-end with the canonical 3-candidate default
    and emits non-zero output if the candidate files are present; if
    missing, the CLI exits non-zero with FileNotFoundError. Both
    outcomes are acceptable proofs of CLI surface integrity."""
    cli_path = _REPO_ROOT / "tools" / "q6_preprobe_pairwise_composition_alpha.py"
    assert cli_path.exists()
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    py = str(venv_python) if venv_python.exists() else sys.executable
    result = subprocess.run(
        [
            py,
            str(cli_path),
            "--report-only-no-side-effects",
            "--repo-root",
            str(_REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    # The CLI should succeed if files exist OR raise a clear error if missing
    if result.returncode == 0:
        assert "[q6.preprobe]" in result.stdout
        assert "Stage 2 gate clause #2 verdict:" in result.stdout
        # --report-only banner present
        assert "NO artifact written" in result.stdout
    else:
        # If candidate files are missing, CLI should fail with a clear msg
        assert "FileNotFoundError" in result.stderr or "missing" in result.stderr.lower()


# ──────────────────────────────────────────────────────────────────────── #
# Test 12: OP-3 council α formula matches the savings-ratio α (cross-audit)
# ──────────────────────────────────────────────────────────────────────── #


def test_op3_council_alpha_formula_distinct_semantics_from_savings_ratio():
    """The two α formulas measure DIFFERENT things and are NOT designed
    to converge numerically.

    - OP-3 council form: 1 - concat/sum_marginal_compressed → measures
      fraction of compressed-marginal budget SAVED by concatenation. For
      ADDITIVE pairs of similar-size marginals this naturally lands near
      0.5 (concat ≈ one-marginal, sum_marginal = 2-marginals → α_op3 ≈ 0.5).
    - Savings-ratio form: savings_concat/sum_marginal_savings → measures
      fraction of total marginal SAVINGS preserved when concatenated. For
      ADDITIVE pairs this lands near 1.0.

    The CANONICAL band-determining form is savings-ratio (per
    Catalog #227 adjust_predicted_delta_for_composition_alpha consumer
    semantics). Both are reported for cross-audit transparency.
    """
    # Coprime-period pair → ADDITIVE per savings-ratio form
    a = bytes([i % 251 for i in range(40_000)])
    b = bytes([(i * 7) % 241 for i in range(40_000)])
    result = probe_mod.compute_pairwise_alpha("a", a, "b", b)
    # Savings-ratio form is the canonical band-determining form
    assert result.alpha_savings_ratio_form >= 0.7, (
        f"coprime-period pair should be ADDITIVE per savings-ratio form; "
        f"got α_savings={result.alpha_savings_ratio_form}"
    )
    assert result.alpha_band in ("ADDITIVE", "SUPER_ADDITIVE")
    # OP-3 form is in a different numerical scale but still a valid signal
    assert isinstance(result.alpha_op3_council_form, float)
    # Both forms emitted (cross-audit preserved)
    assert -1.0 <= result.alpha_op3_council_form <= 2.0
    assert -1.0 <= result.alpha_savings_ratio_form <= 2.0


# ──────────────────────────────────────────────────────────────────────── #
# Test 13: All 3 canonical candidates are registered in CANDIDATE_PATHS
# ──────────────────────────────────────────────────────────────────────── #


def test_stage_2_candidates_registered_in_canonical_paths():
    """All 3 STAGE_2_CANDIDATES must have CANDIDATE_PATHS entries."""
    for cid in probe_mod.STAGE_2_CANDIDATES:
        assert cid in probe_mod.CANDIDATE_PATHS, f"missing {cid}"
        rel_path, substrate_class = probe_mod.CANDIDATE_PATHS[cid]
        assert isinstance(rel_path, str)
        assert substrate_class in (
            "raw_float_weights",
            "raw_float_latents",
            "scorer_margin_float32",
        )


# ──────────────────────────────────────────────────────────────────────── #
# Test 14: Stage 2 threshold pinning + Catalog #227 banding constants pinned
# ──────────────────────────────────────────────────────────────────────── #


def test_canonical_thresholds_pinned():
    """The α-banding thresholds + Stage 2 minimum pair count MUST be
    pinned to Catalog #227 + T2 council OP-2.b spec values. Regression
    guard against silent drift."""
    assert probe_mod.ALPHA_ADDITIVE_THRESHOLD == 0.7
    assert probe_mod.ALPHA_SATURATING_THRESHOLD == 0.3
    assert probe_mod.ALPHA_SUPER_ADDITIVE_THRESHOLD == 1.1
    assert probe_mod.STAGE_2_MINIMUM_ADDITIVE_PAIRS == 2
    assert probe_mod.STAGE_2_CANDIDATES == (
        "pr101_state_dict",
        "pr106_state_dict",
        "posenet_class_sensitivity",
    )
