# SPDX-License-Identifier: MIT
"""Tests for tools/probe_stc_3a_a1_residual_entropy.py.

Per OVERNIGHT-Y lane `lane_overnight_y_stc_3a_a1_residual_entropy_probe_build_local_cpu_run_20260521`
+ Catalog #229 PV + Catalog #287 evidence-tag discipline + Catalog #323 canonical
Provenance + Catalog #344 canonical equation #359-sister IN-DOMAIN reference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from probe_stc_3a_a1_residual_entropy import (  # noqa: E402
    CANONICAL_A1_ARCHIVE_BYTES,
    CANONICAL_A1_ARCHIVE_SHA256,
    HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL,
    HIGH_SPARSITY_THRESHOLD,
    MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL,
    MEDIUM_SPARSITY_THRESHOLD,
    STCResidualEntropyProbeVerdict,
    build_canonical_provenance_for_report,
    classify_verdict_tier,
    compute_predicted_delta_s_band,
    compute_stc_residual_entropy_on_a1,
    five_tuple_sparsity,
    main,
    shannon_entropy_bits_per_symbol,
    synthetic_residuals_for_test,
    verify_a1_archive_sha256,
)


# =====================================================================
# Test 1: Synthetic uniform random → high entropy (~8 bits/symbol)
# =====================================================================
def test_uniform_random_high_entropy_low_sparsity() -> None:
    """Uniform-random residuals should produce ~8 bits/symbol entropy + low sparsity."""
    residuals = synthetic_residuals_for_test(pattern="uniform_random", n_symbols=50_000)
    entropy = shannon_entropy_bits_per_symbol(residuals)
    sparsity = five_tuple_sparsity(residuals)
    # Uniform over [-127, 128] → entropy near log2(255) ≈ 7.99
    assert entropy > 7.5, f"expected high entropy, got {entropy}"
    assert sparsity < 0.05, f"expected low sparsity, got {sparsity}"


# =====================================================================
# Test 2: Synthetic predictable (sparse spikes) → low entropy + high sparsity
# =====================================================================
def test_compressible_low_entropy_high_sparsity() -> None:
    """Sparse-spike residuals should produce low entropy + high sparsity."""
    residuals = synthetic_residuals_for_test(pattern="compressible", n_symbols=50_000)
    entropy = shannon_entropy_bits_per_symbol(residuals)
    sparsity = five_tuple_sparsity(residuals)
    # Mostly zeros + 10% spikes → entropy < 2 + sparsity > 0.85
    assert entropy < 2.0, f"expected low entropy, got {entropy}"
    assert sparsity > 0.85, f"expected high sparsity, got {sparsity}"


# =====================================================================
# Test 3: Verdict tier HIGH classification
# =====================================================================
def test_verdict_tier_high() -> None:
    """HIGH tier requires BOTH entropy>=2.5 AND sparsity>=0.40."""
    tier, rationale = classify_verdict_tier(
        residual_entropy=3.0,
        sparsity=0.50,
    )
    assert tier == "HIGH"
    assert "HIGH-ENTROPY-RESIDUAL-PRESENT" in rationale
    assert "$5.20" in rationale


def test_verdict_tier_medium_via_entropy() -> None:
    """MEDIUM tier via entropy-only path: entropy in [1.5, 2.5)."""
    tier, _ = classify_verdict_tier(
        residual_entropy=2.0,
        sparsity=0.10,
    )
    assert tier == "MEDIUM"


def test_verdict_tier_medium_via_sparsity() -> None:
    """MEDIUM tier via sparsity-only path: sparsity in [0.2, 0.4) (even with low entropy)."""
    tier, _ = classify_verdict_tier(
        residual_entropy=1.0,
        sparsity=0.30,
    )
    assert tier == "MEDIUM"


def test_verdict_tier_medium_when_only_one_meets_high() -> None:
    """MEDIUM tier when entropy is HIGH but sparsity is below MEDIUM."""
    # sparsity=0.05 (below MEDIUM threshold 0.20) + entropy=5.0 (above HIGH threshold 2.5)
    # entropy meets MEDIUM OR sparsity → branch is MEDIUM
    tier, _ = classify_verdict_tier(
        residual_entropy=5.0,
        sparsity=0.05,
    )
    assert tier == "MEDIUM"


def test_verdict_tier_low() -> None:
    """LOW tier requires BOTH entropy<1.5 AND sparsity<0.20."""
    tier, rationale = classify_verdict_tier(
        residual_entropy=1.0,
        sparsity=0.10,
    )
    assert tier == "LOW"
    assert "Catalog #307" in rationale
    assert "DEFER substrate" in rationale


# =====================================================================
# Test 4: Canonical Provenance respect (score_claim=False; promotable=False)
# =====================================================================
def test_verdict_dataclass_invariants() -> None:
    """STCResidualEntropyProbeVerdict enforces non-promotable invariants."""
    verdict = STCResidualEntropyProbeVerdict(
        residual_entropy_bits_per_symbol=3.0,
        five_tuple_sparsity_ratio=0.5,
        predicted_delta_s_rate_only=0.000271,
        predicted_delta_s_band=(-0.005, 0.001),
        verdict_tier="HIGH",
        canonical_equation_id="procedural_predictor_plus_residual_correction_savings_v1",
        verdict_rationale="test verdict",
        sample_pairs_decoded=16,
        sample_pairs_total_residuals=100_000,
        a1_archive_sha256=CANONICAL_A1_ARCHIVE_SHA256,
        a1_archive_bytes=CANONICAL_A1_ARCHIVE_BYTES,
    )
    assert verdict.score_claim is False
    assert verdict.promotable is False
    assert verdict.axis_tag == "[macOS-CPU advisory]"


def test_verdict_dataclass_rejects_score_claim_true() -> None:
    """Invariant: score_claim must always be False per Catalog #192."""
    with pytest.raises(ValueError, match="score_claim must be False"):
        STCResidualEntropyProbeVerdict(
            residual_entropy_bits_per_symbol=3.0,
            five_tuple_sparsity_ratio=0.5,
            predicted_delta_s_rate_only=0.000271,
            predicted_delta_s_band=(-0.005, 0.001),
            verdict_tier="HIGH",
            canonical_equation_id="x",
            verdict_rationale="x",
            sample_pairs_decoded=0,
            sample_pairs_total_residuals=0,
            a1_archive_sha256="0" * 64,
            a1_archive_bytes=0,
            score_claim=True,  # FORBIDDEN
        )


def test_verdict_dataclass_rejects_promotable_true() -> None:
    """Invariant: promotable must always be False per Catalog #192."""
    with pytest.raises(ValueError, match="promotable must be False"):
        STCResidualEntropyProbeVerdict(
            residual_entropy_bits_per_symbol=3.0,
            five_tuple_sparsity_ratio=0.5,
            predicted_delta_s_rate_only=0.000271,
            predicted_delta_s_band=(-0.005, 0.001),
            verdict_tier="HIGH",
            canonical_equation_id="x",
            verdict_rationale="x",
            sample_pairs_decoded=0,
            sample_pairs_total_residuals=0,
            a1_archive_sha256="0" * 64,
            a1_archive_bytes=0,
            promotable=True,  # FORBIDDEN
        )


def test_verdict_dataclass_rejects_invalid_tier() -> None:
    """Invariant: verdict_tier must be HIGH/MEDIUM/LOW."""
    with pytest.raises(ValueError, match="verdict_tier must be"):
        STCResidualEntropyProbeVerdict(
            residual_entropy_bits_per_symbol=3.0,
            five_tuple_sparsity_ratio=0.5,
            predicted_delta_s_rate_only=0.000271,
            predicted_delta_s_band=(-0.005, 0.001),
            verdict_tier="WRONG",
            canonical_equation_id="x",
            verdict_rationale="x",
            sample_pairs_decoded=0,
            sample_pairs_total_residuals=0,
            a1_archive_sha256="0" * 64,
            a1_archive_bytes=0,
        )


# =====================================================================
# Test 5: Deterministic output (sha256 of report deterministic for same input)
# =====================================================================
def test_deterministic_synthetic_output() -> None:
    """Same synthetic seed should produce identical residual array."""
    a = synthetic_residuals_for_test(pattern="high", n_symbols=10_000, seed=42)
    b = synthetic_residuals_for_test(pattern="high", n_symbols=10_000, seed=42)
    assert np.array_equal(a, b)


# =====================================================================
# Test 6: Graceful failure on missing A1 archive
# =====================================================================
def test_missing_a1_archive_raises_filenotfound(tmp_path: Path) -> None:
    """verify_a1_archive_sha256 raises FileNotFoundError for missing path."""
    missing = tmp_path / "missing_archive.zip"
    with pytest.raises(FileNotFoundError, match="A1 archive not found"):
        verify_a1_archive_sha256(missing)


# =====================================================================
# Test 7: A1 archive sha256 verification refuses unknown bytes
# =====================================================================
def test_a1_sha256_mismatch_raises(tmp_path: Path) -> None:
    """verify_a1_archive_sha256 raises ValueError on sha mismatch."""
    fake = tmp_path / "fake_archive.zip"
    fake.write_bytes(b"not the real archive")
    with pytest.raises(ValueError, match="sha mismatch"):
        verify_a1_archive_sha256(fake)


def test_a1_sha256_canonical_passes() -> None:
    """The canonical A1 archive at submissions/a1/archive.zip MUST pass sha check."""
    archive_path = REPO_ROOT / "submissions" / "a1" / "archive.zip"
    if not archive_path.exists():
        pytest.skip("canonical A1 archive not present")
    sha = verify_a1_archive_sha256(archive_path)
    assert sha == CANONICAL_A1_ARCHIVE_SHA256


# =====================================================================
# Test 8: Shannon entropy edge cases
# =====================================================================
def test_shannon_entropy_empty_returns_zero() -> None:
    """Empty array should return 0 entropy."""
    assert shannon_entropy_bits_per_symbol(np.array([], dtype=np.int16)) == 0.0


def test_shannon_entropy_constant_returns_zero() -> None:
    """All-same-value array should return 0 entropy (perfect predictability)."""
    arr = np.full(1000, 5, dtype=np.int16)
    assert shannon_entropy_bits_per_symbol(arr) == 0.0


def test_shannon_entropy_two_equally_likely_returns_one() -> None:
    """50/50 binary distribution should produce 1 bit/symbol."""
    arr = np.array([0, 1] * 500, dtype=np.int16)
    entropy = shannon_entropy_bits_per_symbol(arr)
    assert abs(entropy - 1.0) < 0.01


# =====================================================================
# Test 9: Sparsity edge cases
# =====================================================================
def test_sparsity_empty_returns_zero() -> None:
    """Empty array should return 0 sparsity."""
    assert five_tuple_sparsity(np.array([], dtype=np.int16)) == 0.0


def test_sparsity_all_zero_returns_one() -> None:
    """All-zero residuals should produce sparsity == 1.0."""
    arr = np.zeros(1000, dtype=np.int16)
    assert five_tuple_sparsity(arr) == 1.0


def test_sparsity_all_large_returns_zero() -> None:
    """All-large residuals (|r| > threshold) should produce sparsity == 0."""
    arr = np.full(1000, 100, dtype=np.int16)
    assert five_tuple_sparsity(arr) == 0.0


# =====================================================================
# Test 10: Predicted delta S band per OVERNIGHT-W §10
# =====================================================================
def test_predicted_band_high_ev() -> None:
    """HIGH-EV (entropy>=2.5 + sparsity>=0.40): band [-0.005, +0.001]."""
    band = compute_predicted_delta_s_band(residual_entropy=3.0, sparsity=0.5)
    assert band == (-0.005, 0.001)


def test_predicted_band_medium_ev() -> None:
    """MEDIUM-EV: band [-0.001, +0.001]."""
    band = compute_predicted_delta_s_band(residual_entropy=2.0, sparsity=0.10)
    assert band == (-0.001, 0.001)


def test_predicted_band_low_ev() -> None:
    """LOW-EV: rate-penalty-only band."""
    band = compute_predicted_delta_s_band(residual_entropy=1.0, sparsity=0.10)
    assert band[0] > 0  # rate penalty
    assert band[1] > band[0]


# =====================================================================
# Test 11: compute_stc_residual_entropy_on_a1 end-to-end on synthetic residuals
# =====================================================================
def test_compute_stc_residual_entropy_on_a1_synthetic_high() -> None:
    """End-to-end: synthetic high-entropy residuals → MEDIUM verdict (entropy=5, sparsity~0.3)."""
    residuals = synthetic_residuals_for_test(pattern="high", n_symbols=10_000)
    verdict = compute_stc_residual_entropy_on_a1(residuals)
    assert verdict.verdict_tier in {"MEDIUM", "HIGH"}
    assert verdict.canonical_equation_id == "procedural_predictor_plus_residual_correction_savings_v1"
    assert verdict.predicted_delta_s_rate_only > 0  # rate penalty per eq #359-sister
    assert verdict.score_claim is False
    assert verdict.promotable is False
    assert verdict.axis_tag == "[macOS-CPU advisory]"


# =====================================================================
# Test 12: Canonical Provenance builder integration
# =====================================================================
def test_canonical_provenance_for_report_returns_dict() -> None:
    """build_canonical_provenance_for_report returns serializable dict."""
    verdict = STCResidualEntropyProbeVerdict(
        residual_entropy_bits_per_symbol=3.0,
        five_tuple_sparsity_ratio=0.5,
        predicted_delta_s_rate_only=0.000271,
        predicted_delta_s_band=(-0.005, 0.001),
        verdict_tier="HIGH",
        canonical_equation_id="x",
        verdict_rationale="x",
        sample_pairs_decoded=0,
        sample_pairs_total_residuals=0,
        a1_archive_sha256=CANONICAL_A1_ARCHIVE_SHA256,
        a1_archive_bytes=CANONICAL_A1_ARCHIVE_BYTES,
    )
    prov = build_canonical_provenance_for_report(verdict)
    assert isinstance(prov, dict)
    # Verify JSON-serializable
    json.dumps(prov, default=str)


# =====================================================================
# Test 13: CLI synthetic test mode produces parseable JSON report
# =====================================================================
def test_cli_synthetic_mode_writes_report(tmp_path: Path) -> None:
    """CLI in --synthetic-test-mode writes a parseable JSON report."""
    report_path = tmp_path / "report.json"
    rc = main([
        "--synthetic-test-mode",
        "--synthetic-pattern", "high",
        "--skip-ledger-registration",
        "--output-report-json", str(report_path),
    ])
    assert rc == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["schema"] == "stc_3a_a1_residual_entropy_probe_v1"
    assert payload["lane_id"] == "lane_overnight_y_stc_3a_a1_residual_entropy_probe_build_local_cpu_run_20260521"
    assert payload["verdict"]["verdict_tier"] in {"HIGH", "MEDIUM", "LOW"}
    assert payload["verdict"]["score_claim"] is False
    assert payload["verdict"]["promotable"] is False
    assert payload["verdict"]["axis_tag"] == "[macOS-CPU advisory]"
    assert payload["canonical_equation_id"] == "procedural_predictor_plus_residual_correction_savings_v1"


# =====================================================================
# Test 14: canonical equation IN-DOMAIN check (Catalog #344 + #359)
# =====================================================================
def test_canonical_equation_359_sister_in_domain() -> None:
    """The probe's canonical context MUST be IN-DOMAIN per Catalog #359."""
    from tac.canonical_equations.procedural_codebook_savings import (
        is_residual_hybrid_context,
    )

    assert is_residual_hybrid_context(
        "stc_predictor_plus_residual_a1_per_pair_correction"
    ) is True


# =====================================================================
# Test 15: Canonical thresholds match OVERNIGHT-W §9 spec
# =====================================================================
def test_canonical_thresholds_match_overnight_w_spec() -> None:
    """Verify the canonical threshold constants match the OVERNIGHT-W spec."""
    assert HIGH_ENTROPY_THRESHOLD_BITS_PER_SYMBOL == 2.5
    assert MEDIUM_ENTROPY_THRESHOLD_BITS_PER_SYMBOL == 1.5
    assert HIGH_SPARSITY_THRESHOLD == 0.40
    assert MEDIUM_SPARSITY_THRESHOLD == 0.20
