# SPDX-License-Identifier: MIT
"""Tests for Q6.preprobe OP-3 extension (10-candidate sweep).

Per `feedback_pre_entropy_substrate_pivot_prober_landed_20260517.md` op-routable #3:
extend pairwise composition_alpha probe to 5+ OTHER PRE_ENTROPY substrate
candidates (besides pr101 / pr106 / posenet) to surface alternative Stage-2
stacking topologies.

This subagent extends the candidate set to 10 (C(10,2) = 45 pairs) per the
briefing math.

Cross-refs:
- `tools/q6_preprobe_pairwise_composition_alpha.py` (the extended canonical helper)
- `tools/pre_entropy_substrate_pivot_prober.py` (sister prober with ranked deliverable savings)
- `.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_20260517T210723.json` (ranking artifact)
- CLAUDE.md Catalog #227 substrate composition matrix
"""

from __future__ import annotations

import importlib.util
import json
import sys
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
Q6_TOOL_PATH = REPO_ROOT / "tools" / "q6_preprobe_pairwise_composition_alpha.py"


def _load_q6_module():
    """Load the q6 preprobe tool as a module (it's a script, not a package)."""
    if "q6_preprobe" in sys.modules:
        return sys.modules["q6_preprobe"]
    spec = importlib.util.spec_from_file_location("q6_preprobe", Q6_TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so dataclass-frozen decorators work (need sys.modules entry)
    sys.modules["q6_preprobe"] = mod
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────────────────────────────────── #
# Test 1: extended candidate set has exactly 10 entries                         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_extended_candidate_set_has_ten_members():
    """STAGE_2_CANDIDATES_EXTENDED MUST have exactly 10 members (C(10,2) = 45 pairs)."""
    mod = _load_q6_module()
    assert hasattr(mod, "STAGE_2_CANDIDATES_EXTENDED")
    assert len(mod.STAGE_2_CANDIDATES_EXTENDED) == 10


def test_extended_candidate_set_includes_canonical_3():
    """The original 3 canonical candidates MUST be a subset of the extended set
    so the new 3-pair entries are deterministically reproducible regression
    fixtures.
    """
    mod = _load_q6_module()
    canonical = set(mod.STAGE_2_CANDIDATES)
    extended = set(mod.STAGE_2_CANDIDATES_EXTENDED)
    assert canonical.issubset(extended), (
        f"canonical 3 ({canonical}) must be subset of extended ({extended})"
    )


def test_extended_candidates_all_have_canonical_paths():
    """Every extended candidate MUST have a registered (path, class) in CANDIDATE_PATHS."""
    mod = _load_q6_module()
    for cand in mod.STAGE_2_CANDIDATES_EXTENDED:
        assert cand in mod.CANDIDATE_PATHS, (
            f"extended candidate {cand!r} missing from CANDIDATE_PATHS"
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Test 2: C(10,2) = 45 pair combinations                                        #
# ──────────────────────────────────────────────────────────────────────────── #


def test_extended_sweep_produces_45_pair_entries(tmp_path):
    """run_sweep over the 10-candidate extended set produces exactly C(10,2) = 45 pair entries."""
    from itertools import combinations

    mod = _load_q6_module()
    # Synthesize candidate_bytes inline to avoid filesystem dependency
    rng_seeds = list(range(10))
    candidate_bytes = {
        c: bytes([(s * 13 + i) % 256 for i in range(1024)])
        for c, s in zip(mod.STAGE_2_CANDIDATES_EXTENDED, rng_seeds, strict=True)
    }
    sweep = mod.run_sweep(
        mod.STAGE_2_CANDIDATES_EXTENDED,
        repo_root=tmp_path,
        candidate_bytes=candidate_bytes,
    )
    assert len(sweep.pair_results) == 45
    # All keys should match the canonical "<a>+<b>" format with a < b alphabetically
    # via combinations order
    expected_pairs = set()
    for a, b in combinations(mod.STAGE_2_CANDIDATES_EXTENDED, 2):
        expected_pairs.add(f"{a}+{b}")
    assert set(sweep.pair_results.keys()) == expected_pairs


# ──────────────────────────────────────────────────────────────────────────── #
# Test 3: every pair entry has alpha + band classification                      #
# ──────────────────────────────────────────────────────────────────────────── #


def test_every_pair_entry_has_alpha_and_band(tmp_path):
    """Every entry in the 45-pair result dict MUST have alpha_band + alpha_savings_ratio_form."""
    mod = _load_q6_module()
    candidate_bytes = {
        c: bytes([(seed * 13 + i) % 256 for i in range(512)])
        for seed, c in enumerate(mod.STAGE_2_CANDIDATES_EXTENDED)
    }
    sweep = mod.run_sweep(
        mod.STAGE_2_CANDIDATES_EXTENDED,
        repo_root=tmp_path,
        candidate_bytes=candidate_bytes,
    )
    valid_bands = {"ADDITIVE", "SUB_ADDITIVE", "SATURATING", "SUPER_ADDITIVE"}
    for pair_key, pair_dict in sweep.pair_results.items():
        assert pair_dict["alpha_band"] in valid_bands, (
            f"unknown band for {pair_key}: {pair_dict['alpha_band']}"
        )
        assert "alpha_savings_ratio_form" in pair_dict
        assert "alpha_op3_council_form" in pair_dict
        assert "stage_2_gate_clause_2_satisfied" in pair_dict


# ──────────────────────────────────────────────────────────────────────────── #
# Test 4: Stage 2 gate clause #2 verdict computed across wider set              #
# ──────────────────────────────────────────────────────────────────────────── #


def test_stage_2_verdict_computed_across_extended_set(tmp_path):
    """The Stage 2 gate clause #2 verdict should be SATISFIED when ≥2-of-45 pairs
    have α ≥ 0.7 (the same threshold applies; the wider sweep is for surface
    discovery)."""
    mod = _load_q6_module()
    candidate_bytes = {
        c: bytes([(seed * 13 + i) % 256 for i in range(512)])
        for seed, c in enumerate(mod.STAGE_2_CANDIDATES_EXTENDED)
    }
    sweep = mod.run_sweep(
        mod.STAGE_2_CANDIDATES_EXTENDED,
        repo_root=tmp_path,
        candidate_bytes=candidate_bytes,
    )
    assert isinstance(sweep.pairs_with_alpha_at_least_threshold, int)
    assert sweep.stage_2_reactivation_clause_2_verdict in ("SATISFIED", "NOT_SATISFIED")
    # With random distinct bytes, lzma generally yields ADDITIVE pairs (codecs
    # cannot find cross-substrate redundancy in random bytes) so all 45 should
    # satisfy clause #2.
    assert sweep.pairs_with_alpha_at_least_threshold >= mod.STAGE_2_MINIMUM_ADDITIVE_PAIRS


# ──────────────────────────────────────────────────────────────────────────── #
# Test 5: original 3-pair entries land in extended sweep (deterministic regression) #
# ──────────────────────────────────────────────────────────────────────────── #


def test_original_three_pairs_present_in_extended_sweep(tmp_path):
    """The 3 original pair entries (pr101+pr106, pr101+posenet, pr106+posenet) MUST
    appear in the 45-pair extended sweep with deterministic α values when run
    against the same synthetic bytes."""
    mod = _load_q6_module()
    seed_map = {c: i for i, c in enumerate(mod.STAGE_2_CANDIDATES_EXTENDED)}
    candidate_bytes = {
        c: bytes([(seed_map[c] * 13 + i) % 256 for i in range(2048)])
        for c in mod.STAGE_2_CANDIDATES_EXTENDED
    }

    # Run 3-candidate sweep first
    sweep_3 = mod.run_sweep(
        mod.STAGE_2_CANDIDATES,
        repo_root=tmp_path,
        candidate_bytes={c: candidate_bytes[c] for c in mod.STAGE_2_CANDIDATES},
    )
    # Run extended 10-candidate sweep
    sweep_10 = mod.run_sweep(
        mod.STAGE_2_CANDIDATES_EXTENDED,
        repo_root=tmp_path,
        candidate_bytes=candidate_bytes,
    )

    # Every key from sweep_3 MUST be in sweep_10 with identical pair contents.
    for key, val_3 in sweep_3.pair_results.items():
        assert key in sweep_10.pair_results, f"pair key {key} missing from extended"
        val_10 = sweep_10.pair_results[key]
        # α should be deterministic: same input bytes + same codec → same compression bytes
        assert val_3["alpha_op3_council_form"] == val_10["alpha_op3_council_form"], (
            f"α drift in {key}: 3-pair={val_3['alpha_op3_council_form']} vs 10-pair={val_10['alpha_op3_council_form']}"
        )
        assert val_3["alpha_savings_ratio_form"] == val_10["alpha_savings_ratio_form"]
        assert val_3["best_codec"] == val_10["best_codec"]


# ──────────────────────────────────────────────────────────────────────────── #
# Test 6: schema version differentiates extended from canonical                 #
# ──────────────────────────────────────────────────────────────────────────── #


def test_schema_version_extended_distinct_from_canonical():
    """The two schema versions MUST be distinct strings so consumers can route on it."""
    mod = _load_q6_module()
    assert mod.SCHEMA_VERSION != mod.SCHEMA_VERSION_EXTENDED
    assert "extended" in mod.SCHEMA_VERSION_EXTENDED
    assert "10c" in mod.SCHEMA_VERSION_EXTENDED


def test_payload_extended_flag_emits_extended_schema(tmp_path):
    """build_output_payload(extended=True) emits SCHEMA_VERSION_EXTENDED."""
    mod = _load_q6_module()
    candidate_bytes = {
        c: bytes([(i * 7 + j) % 256 for j in range(256)])
        for i, c in enumerate(mod.STAGE_2_CANDIDATES_EXTENDED)
    }
    sweep = mod.run_sweep(
        mod.STAGE_2_CANDIDATES_EXTENDED,
        repo_root=tmp_path,
        candidate_bytes=candidate_bytes,
    )
    payload_canonical = mod.build_output_payload(sweep, brotli_available=False, extended=False)
    payload_extended = mod.build_output_payload(sweep, brotli_available=False, extended=True)
    assert payload_canonical["schema_version"] == mod.SCHEMA_VERSION
    assert payload_extended["schema_version"] == mod.SCHEMA_VERSION_EXTENDED


# ──────────────────────────────────────────────────────────────────────────── #
# Test 7: --extended CLI flag conflicts with --candidate-a/--candidate-b         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_extended_flag_conflicts_with_explicit_candidates():
    """Passing --extended AND --candidate-a/--candidate-b should error (mutually exclusive)."""
    result = subprocess.run(
        [
            sys.executable,
            str(Q6_TOOL_PATH),
            "--extended",
            "--candidate-a",
            "pr101_state_dict",
            "--candidate-b",
            "pr106_state_dict",
            "--report-only-no-side-effects",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    # argparse errors exit with rc=2
    assert result.returncode != 0
    assert "extended" in (result.stderr.lower() + result.stdout.lower())


# ──────────────────────────────────────────────────────────────────────────── #
# Test 8: --extended produces 45 pair entries via CLI subprocess                 #
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.mark.timeout(600)
def test_extended_cli_writes_45_pair_artifact(tmp_path):
    """Run the tool's --extended CLI and confirm the emitted JSON has 45 pair entries.

    Note: this is a slow test (>60s default pytest timeout) because lzma
    compression on the 20MB posenet_class_sensitivity tensor + 3.5MB
    distill_v2_best + 7 smaller candidates takes ~3-5 minutes for the full
    45-pair sweep. Bumped to 600s.
    """
    output_path = tmp_path / "extended_45.json"
    # Run from repo root so candidate paths resolve
    result = subprocess.run(
        [
            sys.executable,
            str(Q6_TOOL_PATH),
            "--extended",
            "--output",
            str(output_path),
            "--repo-root",
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        timeout=540,  # lzma on 20MB tensor is slow; pair count 45 vs 3 → ~15× compute
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        # If files are missing on this dev machine, skip gracefully (the artifact
        # paths reference experiments/results/* which may not all exist locally)
        if "missing at" in result.stderr or "FileNotFoundError" in result.stderr:
            pytest.skip(f"some extended candidates missing locally: {result.stderr[:200]}")
        else:
            pytest.fail(f"CLI failed: rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}")
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data["schema_version"] == "pairwise_composition_alpha_probe_v1_extended_10c"
    assert len(data["pair_results"]) == 45
    assert len(data["candidates_probed"]) == 10


# ──────────────────────────────────────────────────────────────────────────── #
# Test 9: extended pair entries cover all 10 candidates                          #
# ──────────────────────────────────────────────────────────────────────────── #


def test_every_candidate_appears_in_at_least_one_pair(tmp_path):
    """Each of the 10 candidates MUST appear in at least 9 of the 45 pairs (C(9,1)=9)."""
    mod = _load_q6_module()
    candidate_bytes = {
        c: bytes([(i * 11 + j) % 256 for j in range(512)])
        for i, c in enumerate(mod.STAGE_2_CANDIDATES_EXTENDED)
    }
    sweep = mod.run_sweep(
        mod.STAGE_2_CANDIDATES_EXTENDED,
        repo_root=tmp_path,
        candidate_bytes=candidate_bytes,
    )
    for cand in mod.STAGE_2_CANDIDATES_EXTENDED:
        appearances = sum(
            1 for k in sweep.pair_results if cand in k.split("+")
        )
        assert appearances == 9, (
            f"candidate {cand!r} appears in {appearances} pairs; expected 9 (C(9,1))"
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Test 10: surfacing alternative stacking topologies                             #
# ──────────────────────────────────────────────────────────────────────────── #


def test_extended_sweep_distinct_alpha_bands_surface_topologies(tmp_path):
    """The PURPOSE of the OP-3 extension is to surface alternative stacking
    topologies. Confirm the per-pair α distribution is non-degenerate (more
    than 1 distinct α value across 45 pairs) when bytes vary by candidate.
    """
    mod = _load_q6_module()
    candidate_bytes = {
        c: bytes([(i * 17 + j * 3) % 256 for j in range(1024)])
        for i, c in enumerate(mod.STAGE_2_CANDIDATES_EXTENDED)
    }
    sweep = mod.run_sweep(
        mod.STAGE_2_CANDIDATES_EXTENDED,
        repo_root=tmp_path,
        candidate_bytes=candidate_bytes,
    )
    # Collect distinct α_savings values
    alpha_values = {
        round(p["alpha_savings_ratio_form"], 6) for p in sweep.pair_results.values()
    }
    # Non-degenerate distribution proves the extended sweep is informative
    assert len(alpha_values) > 1, (
        "all 45 pairs collapsed to same α — extended sweep is not surfacing variation"
    )
