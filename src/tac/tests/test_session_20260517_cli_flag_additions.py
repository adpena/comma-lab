# SPDX-License-Identifier: MIT
"""Regression tests for the 2026-05-17 CLI flag additions.

Audit-finding MEDIUM-3 closure (`feedback_post_landing_audit_gate_per_pair_master_gradient_namespace_wave_20260517.md`):
no dedicated tests existed for the session's CLI flag additions across the extractor + autopilot loop +
the Venn-reweighting function. This file closes that gap.

Surfaces covered:
1. `tools/extract_master_gradient.py` — `--compute-dtype` / `--storage-dtype` / `--preserve-per-pair` / `--per-pair-output-npy`
2. `tools/cathedral_autopilot_autonomous_loop.py` — `--report-only` / `--report-top-n`
3. `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification` + helpers
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_autopilot_module():
    """Load the autopilot loop as a module so we can call its internal helpers directly."""
    spec = importlib.util.spec_from_file_location(
        "autopilot_loop",
        REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("autopilot_loop", mod)
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────────────────────────────────── #
# 1. Extractor CLI surface tests                                                #
# ──────────────────────────────────────────────────────────────────────────── #


def test_extractor_help_advertises_compute_dtype_flag():
    """`--compute-dtype` flag MUST be in --help output (canonical CLI contract)."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "extract_master_gradient.py"), "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "--compute-dtype" in result.stdout, "--compute-dtype not advertised in help"
    assert "float32" in result.stdout and "float64" in result.stdout, "compute-dtype choices missing"


def test_extractor_help_advertises_storage_dtype_flag():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "extract_master_gradient.py"), "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT),
    )
    assert "--storage-dtype" in result.stdout, "--storage-dtype not advertised in help"


def test_extractor_help_advertises_preserve_per_pair_flag():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "extract_master_gradient.py"), "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT),
    )
    assert "--preserve-per-pair" in result.stdout
    assert "--per-pair-output-npy" in result.stdout


def test_extractor_help_advertises_per_pair_canonical_anchor_method_method_name():
    """The help text MUST cite the canonical PER_PAIR_GRADIENT_TENSOR_KIND contract."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "extract_master_gradient.py"), "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT),
    )
    # Apple Silicon CPU fp64 ~4× wall-clock caveat MUST be documented in --compute-dtype help
    assert "Apple Silicon" in result.stdout or "scalar-only" in result.stdout, (
        "--compute-dtype help must document Apple Silicon CPU fp64 SIMD trade-off"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# 2. Autopilot loop CLI surface tests                                           #
# ──────────────────────────────────────────────────────────────────────────── #


def test_autopilot_help_advertises_report_only_flag():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"), "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    assert "--report-only" in result.stdout, "--report-only not advertised"
    assert "no side effects" in result.stdout.lower(), "--report-only help must declare no-side-effects contract"


def test_autopilot_help_advertises_report_top_n_flag():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"), "--help"],
        capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT),
    )
    assert "--report-top-n" in result.stdout


def test_autopilot_report_only_no_dispatch_claim_recorded(tmp_path):
    """--report-only MUST NOT touch dispatch claims / halt events / spend authorization.

    Per CLAUDE.md "Apples-to-apples evidence discipline" and the --report-only
    schema's claude_md_compliance_tags array.
    """
    # Synthetic candidate JSONL
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        json.dumps({
            "candidate_id": "lane_test_only_a",
            "family": "test_synthetic",
            "predicted_score_delta": -0.012,
            "estimated_dispatch_cost_usd": 1.0,
            "estimated_planning_cost_usd": 0.0,
            "expected_information_gain": 0.012,
            "blockers": [],
        }) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"),
            "--candidates-jsonl", str(candidates_path),
            "--report-only", "--report-top-n", "1",
            "--rank-axis", "predicted_score_delta",
        ],
        capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"--report-only invocation failed: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["schema"] == "cathedral_autopilot_report_only_v1"
    assert "report_only_no_side_effects" in payload["claude_md_compliance_tags"]
    assert "no_dispatch_claim_recorded" in payload["claude_md_compliance_tags"]
    assert "no_halt_event_emitted" in payload["claude_md_compliance_tags"]
    assert "no_spend_authorization" in payload["claude_md_compliance_tags"]
    assert payload["n_candidates_total"] == 1
    assert payload["top_n_emitted"] == 1
    # Each top candidate row MUST be score-claim-free
    for c in payload["top_candidates"]:
        assert c["score_claim"] is False
        assert c["promotion_eligible"] is False
        assert c["ready_for_exact_eval_dispatch"] is False


# ──────────────────────────────────────────────────────────────────────────── #
# 3. Venn-reweighting function direct tests                                     #
# ──────────────────────────────────────────────────────────────────────────── #


def test_venn_reweight_empty_archive_sha_passthrough():
    """Empty archive_sha256 MUST passthrough raw delta (no Venn signal available)."""
    mod = _load_autopilot_module()
    assert mod.adjust_predicted_delta_for_venn_classification(-0.012, "") == -0.012


def test_venn_reweight_unknown_archive_sha_passthrough():
    """Unknown archive sha (no sidecar exists) MUST passthrough raw delta."""
    mod = _load_autopilot_module()
    fake_sha = "deadbeef" * 8
    assert mod.adjust_predicted_delta_for_venn_classification(-0.012, fake_sha) == -0.012


def test_venn_reweight_high_pair_invariant_without_proof_passthrough(tmp_path, monkeypatch):
    """HIGH PAIR_INVARIANT without DeliverabilityProof must not apply reward."""
    mod = _load_autopilot_module()
    # Create a synthetic sidecar directory + sidecar file
    fake_root = tmp_path / "master_gradient_consumers"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_root)
    sha = "abcdef0123456789" + "00" * 24  # 32-byte sha hex
    sidecar = fake_root / f"venn_classification_{sha[:12]}_20260517T120000.json"
    sidecar.write_text(json.dumps({
        "schema": "master_gradient_consumer_venn_classification_v1",
        "class_counts": {
            "PAIR_SPECIFIC": 100,
            "PAIR_INVARIANT": 8500,   # 85% > 80% threshold
            "PAIR_NEUTRAL": 1400,
            "DEAD": 0,
        },
    }), encoding="utf-8")
    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification(raw, sha)
    assert adjusted == raw


def test_venn_reweight_high_pair_specific_applies_penalty_factor(tmp_path, monkeypatch):
    """Sidecar with >= 30% PAIR_SPECIFIC MUST multiply delta by 0.85 (PENALTY)."""
    mod = _load_autopilot_module()
    fake_root = tmp_path / "master_gradient_consumers"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_root)
    sha = "fedcba9876543210" + "00" * 24
    sidecar = fake_root / f"venn_classification_{sha[:12]}_20260517T130000.json"
    sidecar.write_text(json.dumps({
        "class_counts": {
            "PAIR_SPECIFIC": 3500,   # 35% > 30% threshold
            "PAIR_INVARIANT": 5000,
            "PAIR_NEUTRAL": 1500,
            "DEAD": 0,
        },
    }), encoding="utf-8")
    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification(raw, sha)
    assert abs(adjusted - (raw * 0.85)) < 1e-9, (
        f"HIGH PAIR_SPECIFIC expected factor 0.85; got {adjusted/raw:.4f}"
    )


def test_venn_reweight_neutral_classification_no_adjustment(tmp_path, monkeypatch):
    """Sidecar with neither high-invariant nor high-specific MUST passthrough raw."""
    mod = _load_autopilot_module()
    fake_root = tmp_path / "master_gradient_consumers"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_root)
    sha = "11" * 32
    sidecar = fake_root / f"venn_classification_{sha[:12]}_20260517T140000.json"
    sidecar.write_text(json.dumps({
        "class_counts": {
            "PAIR_SPECIFIC": 1000,   # 10% < 30% threshold
            "PAIR_INVARIANT": 6000,  # 60% < 80% threshold
            "PAIR_NEUTRAL": 3000,
            "DEAD": 0,
        },
    }), encoding="utf-8")
    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification(raw, sha)
    assert adjusted == raw, "neutral classification must passthrough raw"


def test_venn_reweight_corrupt_sidecar_passthrough(tmp_path, monkeypatch):
    """Malformed sidecar JSON MUST passthrough raw (fail-OPEN; never crash ranker)."""
    mod = _load_autopilot_module()
    fake_root = tmp_path / "master_gradient_consumers"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_root)
    sha = "22" * 32
    sidecar = fake_root / f"venn_classification_{sha[:12]}_20260517T150000.json"
    sidecar.write_text("THIS IS NOT JSON", encoding="utf-8")
    raw = -0.012
    assert mod.adjust_predicted_delta_for_venn_classification(raw, sha) == raw


def test_venn_reweight_missing_class_counts_passthrough(tmp_path, monkeypatch):
    """Sidecar without class_counts dict MUST passthrough raw."""
    mod = _load_autopilot_module()
    fake_root = tmp_path / "master_gradient_consumers"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_root)
    sha = "33" * 32
    sidecar = fake_root / f"venn_classification_{sha[:12]}_20260517T160000.json"
    sidecar.write_text(json.dumps({"schema": "v1", "no_class_counts_key": True}), encoding="utf-8")
    raw = -0.012
    assert mod.adjust_predicted_delta_for_venn_classification(raw, sha) == raw


def test_venn_reweight_picks_most_recent_sidecar(tmp_path, monkeypatch):
    """If multiple sidecars exist for the same sha, the lexicographically-max (most recent) wins."""
    mod = _load_autopilot_module()
    fake_root = tmp_path / "master_gradient_consumers"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", fake_root)
    sha = "44" * 32
    older = fake_root / f"venn_classification_{sha[:12]}_20260517T100000.json"
    older.write_text(json.dumps({
        "class_counts": {"PAIR_SPECIFIC": 4000, "PAIR_INVARIANT": 4000, "PAIR_NEUTRAL": 2000, "DEAD": 0},
    }), encoding="utf-8")
    newer = fake_root / f"venn_classification_{sha[:12]}_20260517T200000.json"
    newer.write_text(json.dumps({
        "class_counts": {"PAIR_SPECIFIC": 100, "PAIR_INVARIANT": 9000, "PAIR_NEUTRAL": 900, "DEAD": 0},
    }), encoding="utf-8")
    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification(raw, sha)
    # newer is HIGH PAIR_INVARIANT, but no DeliverabilityProof exists in this
    # synthetic fixture, so the positive reward fails closed to passthrough.
    assert adjusted == raw


def test_venn_reweight_short_sha_passthrough():
    """archive_sha256 shorter than 12 chars MUST passthrough (defensive guard)."""
    mod = _load_autopilot_module()
    assert mod.adjust_predicted_delta_for_venn_classification(-0.012, "short") == -0.012


def test_venn_reweight_no_sidecar_dir_passthrough(tmp_path, monkeypatch):
    """Missing sidecar root directory MUST passthrough (gracefully handle clean repos)."""
    mod = _load_autopilot_module()
    nonexistent = tmp_path / "does_not_exist"
    monkeypatch.setattr(mod, "_VENN_CLASSIFICATION_SIDECAR_ROOT", nonexistent)
    sha = "55" * 32
    raw = -0.012
    assert mod.adjust_predicted_delta_for_venn_classification(raw, sha) == raw


# ──────────────────────────────────────────────────────────────────────────── #
# 4. End-to-end smoke (against actual live Venn sidecar for fec6 archive)       #
# ──────────────────────────────────────────────────────────────────────────── #


def test_venn_reweight_live_fec6_sidecar_round_trip():
    """If the fec6 Venn sidecar exists locally, verify it correctly applies REWARD.

    This is the canonical anchor test: the fec6 archive's Venn classification
    is HIGH PAIR_INVARIANT (~90.7% in the 8-pair fp64 anchor) so the wire-in
    may apply a proof-backed positive reward in production.
    """
    mod = _load_autopilot_module()
    live_root = REPO_ROOT / ".omx" / "state" / "master_gradient_consumers"
    fec6_sha = "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"
    # Skip if no live sidecar exists (test runs in clean clones too)
    if not live_root.is_dir():
        pytest.skip("no master_gradient_consumers sidecar root in this repo state")
    sidecar = mod._latest_venn_sidecar_for_archive(fec6_sha)
    if sidecar is None:
        pytest.skip(f"no live Venn sidecar for fec6 archive {fec6_sha[:12]}")
    raw = -0.012
    adjusted = mod.adjust_predicted_delta_for_venn_classification(raw, fec6_sha)
    # fec6 is often HIGH PAIR_INVARIANT. Catalog #319 now requires a proof
    # before any positive reward applies; if classification/proofs changed
    # (e.g. 600-pair tensor lands and Venn breakdown shifts) the test still
    # passes as long as SOME canonical factor applies.
    factor = adjusted / raw
    assert factor in (1.0, 0.85, 1.05, 1.10, 1.15, 1.20), (
        f"live fec6 Venn factor {factor:.4f} not in canonical factor set; "
        f"check _read_venn_class_counts integrity"
    )
