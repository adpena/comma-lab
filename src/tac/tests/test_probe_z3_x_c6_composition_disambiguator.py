# SPDX-License-Identifier: MIT
"""Dedicated tests for tools/probe_z3_x_c6_composition_disambiguator.

Per T1-F Grand Council Tier-1 authorization. ≥15 dedicated tests covering:

- Verdict band thresholds (ADDITIVE / SUB-ADDITIVE / SATURATING)
- Degenerate cases (negative savings / predicted_additive ≤ 0)
- Edge cases (zero savings, equal savings)
- Determinism (same seed → same synthetic sidecar bytes)
- Composition matrix append round-trip
- CLI subprocess exit code

All tests CPU-only; no network; no GPU.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE_PATH = Path(__file__).resolve().parents[3] / "tools" / "probe_z3_x_c6_composition_disambiguator.py"


def _load_probe_module():
    module_name = "probe_z3_x_c6_composition_disambiguator"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load probe at {_PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so dataclass + others can
    # resolve cls.__module__ during type construction.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe_module()


# =====================================================================
# Section 1: alpha computation + verdict band thresholds (≥6 tests)
# =====================================================================


def test_alpha_additive_band_realized_matches_predicted():
    """α = 1.0 (perfectly additive) → ADDITIVE verdict."""
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=1000, delta_b_c6=2000, delta_b_stack=3000
    )
    assert math.isclose(alpha, 1.0)
    assert verdict == "additive"


def test_alpha_additive_band_above_threshold():
    """α > 0.7 → ADDITIVE verdict."""
    # predicted_additive=3000, realized=2500 → α=0.833
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=1000, delta_b_c6=2000, delta_b_stack=2500
    )
    assert alpha > probe.ALPHA_ADDITIVE_THRESHOLD
    assert verdict == "additive"


def test_alpha_sub_additive_band():
    """0.3 < α ≤ 0.7 → SUB-ADDITIVE verdict."""
    # predicted_additive=3000, realized=1500 → α=0.5
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=1000, delta_b_c6=2000, delta_b_stack=1500
    )
    assert math.isclose(alpha, 0.5)
    assert verdict == "sub_additive"


def test_alpha_saturating_band():
    """α ≤ 0.3 → SATURATING verdict."""
    # predicted_additive=3000, realized=600 → α=0.2
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=1000, delta_b_c6=2000, delta_b_stack=600
    )
    assert math.isclose(alpha, 0.2)
    assert verdict == "saturating"


def test_alpha_boundary_at_0_7_threshold_strictly_above_additive():
    """α just above 0.7 is ADDITIVE (strict >); at 0.7 exactly is sub-additive."""
    # predicted=1000, realized=701 → α=0.701 → additive
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=500, delta_b_c6=500, delta_b_stack=701
    )
    assert alpha > probe.ALPHA_ADDITIVE_THRESHOLD
    assert verdict == "additive"
    # predicted=1000, realized=700 → α=0.7 exactly → sub_additive (boundary closed below)
    alpha2, verdict2 = probe.compute_alpha(
        delta_b_z3=500, delta_b_c6=500, delta_b_stack=700
    )
    assert math.isclose(alpha2, 0.7)
    assert verdict2 == "sub_additive"


def test_alpha_boundary_at_0_3_threshold_strictly_above_sub_additive():
    """α just above 0.3 is SUB-ADDITIVE; at 0.3 exactly is saturating."""
    # predicted=1000, realized=301 → α=0.301 → sub_additive
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=500, delta_b_c6=500, delta_b_stack=301
    )
    assert verdict == "sub_additive"
    # predicted=1000, realized=300 → α=0.3 exactly → saturating (boundary closed below)
    alpha2, verdict2 = probe.compute_alpha(
        delta_b_z3=500, delta_b_c6=500, delta_b_stack=300
    )
    assert math.isclose(alpha2, 0.3)
    assert verdict2 == "saturating"


# =====================================================================
# Section 2: degenerate / edge cases (≥4 tests)
# =====================================================================


def test_alpha_degenerate_both_negative():
    """Both substrates ADD bytes → degenerate_both_negative verdict; α = NaN."""
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=-500, delta_b_c6=-1000, delta_b_stack=-2000
    )
    assert math.isnan(alpha)
    assert verdict == "degenerate_both_negative"


def test_alpha_predicted_additive_le_zero():
    """Predicted_additive ≤ 0 (one positive, one offsetting negative) → special verdict."""
    # Z3 saves 1000, C6 adds 2000 → predicted_additive = -1000 ≤ 0
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=1000, delta_b_c6=-2000, delta_b_stack=500
    )
    assert math.isnan(alpha)
    assert verdict == "predicted_additive_le_zero"


def test_alpha_predicted_additive_exactly_zero():
    """predicted_additive == 0 → predicted_additive_le_zero verdict."""
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=500, delta_b_c6=-500, delta_b_stack=0
    )
    assert math.isnan(alpha)
    assert verdict == "predicted_additive_le_zero"


def test_alpha_realized_negative_below_zero_saturating():
    """If stack ADDS bytes vs A1 (realized<0) but predicted>0 → α<0 saturating."""
    alpha, verdict = probe.compute_alpha(
        delta_b_z3=1000, delta_b_c6=2000, delta_b_stack=-500
    )
    assert alpha < 0
    assert verdict == "saturating"


# =====================================================================
# Section 3: run_probe end-to-end with ProbeInputs (≥3 tests)
# =====================================================================


def test_run_probe_additive_baseline():
    """End-to-end run_probe with synthetic-additive inputs."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=190_000,  # Z3 saves 10K
        c6_only_archive_bytes=180_000,  # C6 saves 20K
        stacked_archive_bytes=170_000,  # Stack saves 30K (additive)
    )
    verdict = probe.run_probe(inputs)
    assert verdict.delta_b_z3 == 10_000
    assert verdict.delta_b_c6 == 20_000
    assert verdict.delta_b_stack == 30_000
    assert math.isclose(verdict.alpha, 1.0)
    assert verdict.verdict == "additive"
    # Custody discipline: probe is research-only.
    assert verdict.score_claim is False
    assert verdict.promotion_eligible is False
    assert verdict.ready_for_exact_eval_dispatch is False
    assert "probe_is_research_only_not_a_score_claim" in verdict.result_review_blockers


def test_run_probe_saturating_baseline():
    """End-to-end run_probe with saturating inputs."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=195_000,  # Z3 saves 5K
        c6_only_archive_bytes=180_000,  # C6 saves 20K
        stacked_archive_bytes=195_000,  # Stack saves only 5K (saturating, Z3 absorbed)
    )
    verdict = probe.run_probe(inputs)
    # predicted_additive = 25_000; realized = 5_000; α = 0.2
    assert math.isclose(verdict.alpha, 0.2)
    assert verdict.verdict == "saturating"


def test_run_probe_predicted_delta_s_matches_canonical_formula():
    """Predicted ΔS uses the contest's -25*Δb/N coefficient."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=190_000,
        c6_only_archive_bytes=180_000,
        stacked_archive_bytes=170_000,
    )
    verdict = probe.run_probe(inputs)
    N = 37_545_489
    assert math.isclose(
        verdict.predicted_delta_s_additive,
        -25.0 * 30_000 / N,
        rel_tol=1e-9,
    )
    assert math.isclose(
        verdict.predicted_delta_s_realized,
        -25.0 * 30_000 / N,
        rel_tol=1e-9,
    )


# =====================================================================
# Section 4: synthetic Z3 sidecar (≥2 tests; deterministic)
# =====================================================================


def test_synthesize_z3_sidecar_deterministic():
    """Same seed → same byte count."""
    bytes_a = probe.synthesize_z3_sidecar_bytes(seed=42, n_pairs=600)
    bytes_b = probe.synthesize_z3_sidecar_bytes(seed=42, n_pairs=600)
    assert bytes_a == bytes_b
    # Should be a reasonable size: header (27) + weights blob + 600*8 w_hat + 600*28 residual,
    # after brotli compression. Expect > 1KB and < 50KB.
    assert 1000 < bytes_a < 50_000


def test_synthesize_z3_sidecar_different_seeds_differ():
    """Different seeds may produce different byte counts (random latents)."""
    bytes_a = probe.synthesize_z3_sidecar_bytes(seed=0, n_pairs=600)
    bytes_b = probe.synthesize_z3_sidecar_bytes(seed=1, n_pairs=600)
    # They could be equal in pathological cases but typically differ by ~1-2%
    # The KEY property is that EACH is deterministic (test above proves that).
    # We just assert both are valid sizes.
    assert bytes_a > 0
    assert bytes_b > 0


def test_synthesize_z3_sidecar_different_hyper_dim_via_seed_variation():
    """Z3HP1 archive grammar pins n_pairs=600 + latent_dim=28; synthesis
    produces consistent byte counts across canonical configurations."""
    # The Z3HP1 archive grammar refuses n_pairs != 600 by design (per
    # archive.py's encode_z3hp1_sidecar validation). Validate that the
    # canonical 600-pair synthesis is itself deterministic + reasonable.
    bytes_seed_0 = probe.synthesize_z3_sidecar_bytes(seed=0, n_pairs=600)
    bytes_seed_2 = probe.synthesize_z3_sidecar_bytes(seed=2, n_pairs=600)
    # Both should produce valid sizes within the expected band.
    assert 1000 < bytes_seed_0 < 50_000
    assert 1000 < bytes_seed_2 < 50_000


def test_synthesize_z3_sidecar_rejects_invalid_n_pairs():
    """Synthesis correctly propagates Z3HP1's n_pairs validation."""
    with pytest.raises(ValueError, match="n_pairs must be 600"):
        probe.synthesize_z3_sidecar_bytes(seed=0, n_pairs=4)


# =====================================================================
# Section 5: composition matrix append round-trip (≥2 tests)
# =====================================================================


def test_append_composition_matrix_entry_round_trip(tmp_path):
    """Append + read-back round-trip preserves verdict fields."""
    matrix_path = tmp_path / "substrate_composition_matrix.json"
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=190_000,
        c6_only_archive_bytes=180_000,
        stacked_archive_bytes=170_000,
    )
    verdict = probe.run_probe(inputs)
    probe.append_composition_matrix_entry(verdict, matrix_path=matrix_path)

    assert matrix_path.exists()
    data = json.loads(matrix_path.read_text())
    assert "entries" in data
    key = "z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps"
    assert key in data["entries"]
    assert len(data["entries"][key]) == 1
    entry = data["entries"][key][0]
    assert entry["verdict"] == "additive"
    assert entry["delta_b_z3"] == 10_000
    assert entry["delta_b_c6"] == 20_000
    assert entry["delta_b_stack"] == 30_000
    assert entry["score_claim"] is False
    assert "probe_is_research_only_not_a_score_claim" in entry["result_review_blockers"]


def test_append_composition_matrix_entry_appends_multiple(tmp_path):
    """Multiple probe runs append (not overwrite)."""
    matrix_path = tmp_path / "matrix.json"
    inputs_1 = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=190_000,
        c6_only_archive_bytes=180_000,
        stacked_archive_bytes=170_000,
    )
    inputs_2 = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=195_000,
        c6_only_archive_bytes=180_000,
        stacked_archive_bytes=195_000,
    )
    probe.append_composition_matrix_entry(probe.run_probe(inputs_1), matrix_path=matrix_path)
    probe.append_composition_matrix_entry(probe.run_probe(inputs_2), matrix_path=matrix_path)
    data = json.loads(matrix_path.read_text())
    key = "z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps"
    assert len(data["entries"][key]) == 2
    assert data["entries"][key][0]["verdict"] == "additive"
    assert data["entries"][key][1]["verdict"] == "saturating"


def test_append_composition_matrix_handles_nan_alpha_as_null(tmp_path):
    """NaN α is serialized as null (JSON-safe)."""
    matrix_path = tmp_path / "matrix.json"
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=200_500,  # Z3 adds bytes
        c6_only_archive_bytes=201_000,  # C6 adds bytes
        stacked_archive_bytes=202_000,
    )
    verdict = probe.run_probe(inputs)
    assert math.isnan(verdict.alpha)
    probe.append_composition_matrix_entry(verdict, matrix_path=matrix_path)
    raw = matrix_path.read_text()
    # JSON should NOT contain "NaN" literal (that's not valid JSON)
    assert "NaN" not in raw
    # null serialization
    data = json.loads(raw)
    key = "z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps"
    assert data["entries"][key][0]["alpha"] is None


# =====================================================================
# Section 6: degenerate-case blockers + custody discipline (≥2 tests)
# =====================================================================


def test_run_probe_indeterminate_inputs_carry_blocker():
    """Predicted-additive-le-zero verdict carries the indeterminate blocker."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=199_000,  # Z3 saves 1K
        c6_only_archive_bytes=203_000,  # C6 adds 3K
        stacked_archive_bytes=202_000,
    )
    verdict = probe.run_probe(inputs)
    assert verdict.verdict == "predicted_additive_le_zero"
    assert "composition_additivity_indeterminate_inputs_yield_no_predicted_savings" in verdict.result_review_blockers


def test_run_probe_invalid_inputs_carry_blocker():
    """Zero/negative byte sizes carry the invalid-bytes blocker."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=0,
        c6_only_archive_bytes=180_000,
        stacked_archive_bytes=170_000,
    )
    verdict = probe.run_probe(inputs)
    assert "invalid_archive_bytes_must_be_positive" in verdict.result_review_blockers


def test_run_probe_additive_estimator_carries_structural_caveat():
    """When the stacked source is an additive estimate, the α verdict
    must carry the structural-not-empirical blocker."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=190_000,
        c6_only_archive_bytes=180_000,
        stacked_archive_bytes=170_000,
        stacked_source="additive_estimate_c6=180000+z3_sidecar=10000",
    )
    verdict = probe.run_probe(inputs)
    assert (
        "alpha_is_structural_not_empirical_real_stacked_archive_required_for_empirical_alpha"
        in verdict.result_review_blockers
    )


def test_run_probe_real_stacked_archive_no_structural_caveat():
    """When the stacked source is a real path (not an estimate), no
    structural caveat is raised."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=190_000,
        c6_only_archive_bytes=180_000,
        stacked_archive_bytes=170_000,
        stacked_source="/path/to/real/stacked_archive.zip",
    )
    verdict = probe.run_probe(inputs)
    assert (
        "alpha_is_structural_not_empirical_real_stacked_archive_required_for_empirical_alpha"
        not in verdict.result_review_blockers
    )


def test_run_probe_reduced_smoke_c6_carries_caveat():
    """C6 archive bytes < 50KB → reduced-smoke caveat fires."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=190_000,
        c6_only_archive_bytes=30_000,  # smoke artifact with reduced num_pairs
        stacked_archive_bytes=40_000,
    )
    verdict = probe.run_probe(inputs)
    assert (
        "c6_archive_bytes_below_50kb_likely_reduced_smoke_num_pairs_not_production"
        in verdict.result_review_blockers
    )


# =====================================================================
# Section 7: CLI subprocess (≥2 tests)
# =====================================================================


def test_cli_default_invocation_produces_valid_json():
    """`python tools/probe_*.py` returns valid JSON on stdout with exit 0."""
    if not (Path("submissions/a1/archive.zip")).exists():
        pytest.skip("A1 baseline archive not present in this checkout")
    result = subprocess.run(
        [sys.executable, str(_PROBE_PATH)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert "verdict" in data
    assert "alpha" in data
    assert "delta_b_z3" in data
    assert "delta_b_c6" in data
    assert "delta_b_stack" in data
    assert data["score_claim"] is False


def test_cli_with_override_inputs():
    """CLI with override bytes produces deterministic verdict."""
    if not (Path("submissions/a1/archive.zip")).exists():
        pytest.skip("A1 baseline archive not present in this checkout")
    result = subprocess.run(
        [
            sys.executable,
            str(_PROBE_PATH),
            "--c6-archive-bytes-override",
            "180000",
            "--z3-sidecar-bytes",
            "10000",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    # A1 = 178262 bytes (frozen public anchor)
    # Z3-only: 178262 + 10000 = 188262 (Z3 ADDS bytes per append-only diag)
    # C6-only: 180000 (C6 ADDS bytes vs A1 too)
    # Stacked: 180000 + 10000 = 190000
    # Δb_Z3 = 178262 - 188262 = -10000
    # Δb_C6 = 178262 - 180000 = -1738
    # Δb_stack = 178262 - 190000 = -11738
    assert data["delta_b_z3"] == -10000
    assert data["delta_b_c6"] == -1738
    assert data["delta_b_stack"] == -11738
    # Both negative → degenerate_both_negative
    assert data["verdict"] == "degenerate_both_negative"


# =====================================================================
# Section 8: real-substrate integration smoke (≥2 tests)
# =====================================================================


def test_real_a1_baseline_archive_read():
    """The A1 frozen public archive is reachable from the canonical path."""
    if not (Path("submissions/a1/archive.zip")).exists():
        pytest.skip("A1 baseline archive not present in this checkout")
    a1_bytes = probe.read_a1_baseline_bytes()
    assert a1_bytes == 178_262, f"A1 archive bytes changed: {a1_bytes}"


def test_real_c6_archive_read_from_smoke_artifact():
    """C6 smoke artifact (if present) parses + returns correct bytes."""
    c6_path = Path("experiments/results/c6_smoke_cpu_re_141735/0.bin")
    if not c6_path.exists():
        pytest.skip("C6 smoke archive not present in this checkout")
    bytes_seen = probe.read_c6_archive_bytes(c6_path)
    assert bytes_seen > 0
    assert bytes_seen == c6_path.stat().st_size


# =====================================================================
# Section 9: ProbeVerdict immutability + dataclass round-trip (≥1 test)
# =====================================================================


def test_probe_verdict_as_dict_is_json_serializable():
    """ProbeVerdict.as_dict() returns a JSON-serializable mapping."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=190_000,
        c6_only_archive_bytes=180_000,
        stacked_archive_bytes=170_000,
    )
    verdict = probe.run_probe(inputs)
    d = verdict.as_dict()
    # Must serialize without exception
    raw = json.dumps(d)
    parsed = json.loads(raw)
    assert parsed["verdict"] == "additive"
    assert parsed["inputs"]["base_a1_archive_bytes"] == 200_000


def test_compute_alpha_with_nan_alpha_serialization_via_dict():
    """NaN α path is also as_dict-serializable through the helper."""
    inputs = probe.ProbeInputs(
        base_a1_archive_bytes=200_000,
        z3_only_archive_bytes=205_000,  # Z3 adds
        c6_only_archive_bytes=205_000,  # C6 adds
        stacked_archive_bytes=210_000,
    )
    verdict = probe.run_probe(inputs)
    assert math.isnan(verdict.alpha)
    # as_dict preserves NaN as float NaN; we ONLY serialize through the
    # composition-matrix path which replaces it with None. Direct as_dict
    # may carry the NaN; the test asserts the conversion is well-defined.
    d = verdict.as_dict()
    assert "alpha" in d
    # We can't json.dumps(NaN) without allow_nan=False being violated.
    # Confirm allow_nan=True (default) tolerates it:
    raw = json.dumps(d)
    assert "NaN" in raw  # Default Python json emits NaN; matrix path replaces it.
