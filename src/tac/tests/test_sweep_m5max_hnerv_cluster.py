# SPDX-License-Identifier: MIT
"""Tests for tools/sweep_m5max_hnerv_cluster.py and tools/m5max_sweep_to_atoms.py.

Pure-logic tests that do NOT actually invoke upstream/evaluate.py (which
requires the full upstream snapshot + 0.mkv + ~5 min wall-clock per eval).
The integration smoke is:

    .venv/bin/python tools/sweep_m5max_hnerv_cluster.py \\
        --candidates-jsonl <small ledger> --output-dir <tmp under experiments/results>

(operator-driven, runs in-place when CLAUDE.md non-negotiables are satisfied)
"""
from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SWEEP_TOOL = REPO_ROOT / "tools" / "sweep_m5max_hnerv_cluster.py"
ATOMS_TOOL = REPO_ROOT / "tools" / "m5max_sweep_to_atoms.py"


def _load(tool_path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep_mod():
    return _load(SWEEP_TOOL, "sweep_m5max_hnerv_cluster")


@pytest.fixture(scope="module")
def atoms_mod():
    return _load(ATOMS_TOOL, "m5max_sweep_to_atoms")


# ────────────────────────────────────────────────────────────────────────────
# Architecture class normalization + tag policy
# ────────────────────────────────────────────────────────────────────────────


def test_normalize_arch_class_hnerv_aliases(sweep_mod) -> None:
    for alias in ["hnerv", "HNeRV", "hnerv_lc_v2", "hnerv-microcodec",
                  "hnerv_ft_microcodec", "ff_packed_brotli_hnerv"]:
        assert sweep_mod._normalize_arch_class(alias) == "hnerv", alias


def test_normalize_arch_class_non_hnerv_unchanged(sweep_mod) -> None:
    assert sweep_mod._normalize_arch_class("h3_grayscale") == "h3_grayscale"
    assert sweep_mod._normalize_arch_class("balle_scale_hyperprior") == "balle_scale_hyperprior"
    assert sweep_mod._normalize_arch_class("") == "unknown"


def test_calibrated_tag_hnerv_only(sweep_mod) -> None:
    assert sweep_mod._calibrated_tag("hnerv") == "[macOS-CPU calibrated]"
    assert sweep_mod._calibrated_tag("hnerv_ft_microcodec") == "[macOS-CPU calibrated]"
    # Non-HNeRV must NOT be tagged calibrated (per memo discipline rule #2)
    assert sweep_mod._calibrated_tag("h3_grayscale") == "[macOS-CPU advisory only]"
    assert sweep_mod._calibrated_tag("av1_high_pose") == "[macOS-CPU advisory only]"
    assert sweep_mod._calibrated_tag("balle_scale_hyperprior") == "[macOS-CPU advisory only]"


# ────────────────────────────────────────────────────────────────────────────
# Promotion classification (the auto-promote-to-GHA rule)
# ────────────────────────────────────────────────────────────────────────────


def test_promotion_thresholds_match_silver_band(sweep_mod) -> None:
    # PR102 silver = 0.19538; auto-promote anything below 0.190 (sub-medal).
    assert sweep_mod._classify_promotion(0.180) == "AUTO_PROMOTE_GHA"
    assert sweep_mod._classify_promotion(0.189) == "AUTO_PROMOTE_GHA"
    # In [0.190, 0.195) → operator decision (silver-band proximity)
    assert sweep_mod._classify_promotion(0.190) == "OPERATOR_DECISION"
    assert sweep_mod._classify_promotion(0.193) == "OPERATOR_DECISION"
    assert sweep_mod._classify_promotion(0.1949) == "OPERATOR_DECISION"
    # In [0.195, 0.200) → log only
    assert sweep_mod._classify_promotion(0.195) == "LOG_ONLY"
    assert sweep_mod._classify_promotion(0.199) == "LOG_ONLY"
    # >= 0.200 → log only
    assert sweep_mod._classify_promotion(0.200) == "LOG_ONLY"
    assert sweep_mod._classify_promotion(0.250) == "LOG_ONLY"
    # None → EVAL_FAILED
    assert sweep_mod._classify_promotion(None) == "EVAL_FAILED"


# ────────────────────────────────────────────────────────────────────────────
# Drift flag detection
# ────────────────────────────────────────────────────────────────────────────


def test_drift_flag_within_tolerance_is_ok(sweep_mod) -> None:
    # Within 5e-5 of prior anchor — OK
    flag = sweep_mod._compute_drift_flag(0.19664189, "hnerv", prior_anchor=0.19664189)
    assert flag == "OK"
    flag = sweep_mod._compute_drift_flag(0.19665, "hnerv", prior_anchor=0.19664189)
    assert flag == "OK"


def test_drift_flag_outside_tolerance_fires(sweep_mod) -> None:
    # 1e-3 deviation triggers POTENTIAL_NEW_CLASS
    flag = sweep_mod._compute_drift_flag(0.19764189, "hnerv", prior_anchor=0.19664189)
    assert flag == "POTENTIAL_NEW_CLASS"


def test_drift_flag_na_for_uncalibrated_class(sweep_mod) -> None:
    # Non-HNeRV class has no calibration anchor; flag is NA
    flag = sweep_mod._compute_drift_flag(0.5, "h3_grayscale", prior_anchor=0.4)
    assert flag == "NA"


def test_drift_flag_na_when_no_prior_anchor(sweep_mod) -> None:
    flag = sweep_mod._compute_drift_flag(0.19664189, "hnerv", prior_anchor=None)
    assert flag == "NA"


# ────────────────────────────────────────────────────────────────────────────
# HNeRV ε-band prediction (the calibration math)
# ────────────────────────────────────────────────────────────────────────────


def test_hnerv_predicted_contest_cpu_uses_pr107_bias(sweep_mod) -> None:
    # PR107: macOS=0.19664189, GHA=0.19663589 → bias=+6e-6
    # An archive that scores 0.19664189 on M5 Max should predict 0.19663589 on GHA
    point, lo, hi = sweep_mod._hnerv_predicted_contest_cpu(0.19664189)
    assert abs(point - 0.19663589) < 1e-12
    assert lo < point < hi
    # Band width = 2 × ε
    assert abs((hi - lo) - 2 * sweep_mod.PR107_EPSILON_BOUND) < 1e-12


def test_hnerv_predicted_contest_cpu_preserves_relative_score(sweep_mod) -> None:
    # An A1-shaped archive at 0.19284757 macOS-CPU should predict ~0.19284 on GHA
    # (bias is per-archive empirical; here we use the PR107 bias as approximation)
    point, lo, hi = sweep_mod._hnerv_predicted_contest_cpu(0.19284757)
    bias = sweep_mod.PR107_MACOS_CPU_SCORE - sweep_mod.PR107_GHA_CPU_SCORE
    assert abs(point - (0.19284757 - bias)) < 1e-12


# ────────────────────────────────────────────────────────────────────────────
# Apple Silicon guard (defensive — never run this on x86_64)
# ────────────────────────────────────────────────────────────────────────────


def test_apple_silicon_guard_passes_on_arm64_macos(sweep_mod) -> None:
    if platform.system().lower() == "darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        # Should not raise
        sweep_mod._verify_running_on_apple_silicon()
    else:
        with pytest.raises(SystemExit):
            sweep_mod._verify_running_on_apple_silicon()


# ────────────────────────────────────────────────────────────────────────────
# Candidate loader: directory mode
# ────────────────────────────────────────────────────────────────────────────


def test_load_candidates_from_dir_direct_zips(sweep_mod, tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_bytes(b"PK\x03\x04stub")
    (tmp_path / "b.zip").write_bytes(b"PK\x03\x04stub")
    specs = sweep_mod._load_candidates_from_dir(tmp_path, "hnerv")
    ids = sorted(s.candidate_id for s in specs)
    assert ids == ["a", "b"]
    for s in specs:
        assert s.architecture_class == "hnerv"


def test_load_candidates_from_dir_nested_subdirs(sweep_mod, tmp_path: Path) -> None:
    (tmp_path / "cand1").mkdir()
    (tmp_path / "cand1" / "archive.zip").write_bytes(b"PK\x03\x04stub")
    (tmp_path / "cand2").mkdir()
    (tmp_path / "cand2" / "archive.zip").write_bytes(b"PK\x03\x04stub")
    specs = sweep_mod._load_candidates_from_dir(tmp_path, "hnerv")
    ids = sorted(s.candidate_id for s in specs)
    assert ids == ["cand1", "cand2"]


def test_load_candidates_from_dir_empty_raises(sweep_mod, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="found no archive.zip"):
        sweep_mod._load_candidates_from_dir(tmp_path, "hnerv")


# ────────────────────────────────────────────────────────────────────────────
# Candidate loader: JSONL mode
# ────────────────────────────────────────────────────────────────────────────


def test_load_candidates_from_jsonl(sweep_mod, tmp_path: Path) -> None:
    arch = tmp_path / "a.zip"
    arch.write_bytes(b"PK\x03\x04stub")
    ledger = tmp_path / "candidates.jsonl"
    ledger.write_text(
        json.dumps({"candidate_id": "c1", "archive_path": str(arch),
                    "architecture_class": "hnerv"}) + "\n"
        + json.dumps({"candidate_id": "c2", "archive_path": str(arch),
                      "architecture_class": "h3_grayscale"}) + "\n"
        # Comment line should be ignored
        "# this is a comment\n"
        + json.dumps({"candidate_id": "c3", "archive_path": str(arch),
                      "architecture_class": "hnerv_lc_v2"}) + "\n"
    )
    specs = sweep_mod._load_candidates_from_jsonl(ledger)
    assert [s.candidate_id for s in specs] == ["c1", "c2", "c3"]
    assert specs[0].architecture_class == "hnerv"
    assert specs[1].architecture_class == "h3_grayscale"


def test_load_candidates_from_jsonl_missing_keys_raises(
    sweep_mod, tmp_path: Path
) -> None:
    ledger = tmp_path / "bad.jsonl"
    ledger.write_text(json.dumps({"candidate_id": "x"}) + "\n")
    with pytest.raises(SystemExit, match="missing required keys"):
        sweep_mod._load_candidates_from_jsonl(ledger)


# ────────────────────────────────────────────────────────────────────────────
# Atom emitter
# ────────────────────────────────────────────────────────────────────────────


def test_atom_for_calibrated_auto_promote_candidate(atoms_mod) -> None:
    row = {
        "candidate_id": "cand_x",
        "archive_path": "/p/archive.zip",
        "archive_sha256": "deadbeef",
        "archive_size_bytes": 178000,
        "architecture_class": "hnerv",
        "macos_cpu_score": 0.185,
        "macos_cpu_avg_segnet_dist": 0.0005,
        "macos_cpu_avg_posenet_dist": 3e-5,
        "compression_rate": 0.0047,
        "n_samples": 600,
        "macos_cpu_calibrated_tag": "[macOS-CPU calibrated]",
        "epsilon_band_low": 0.184999,
        "epsilon_band_high": 0.185001,
        "predicted_contest_cpu_gha": 0.184994,
        "drift_flag": "OK",
        "promotion_verdict": "AUTO_PROMOTE_GHA",
    }
    atom = atoms_mod._row_to_atom(row, "/p/results.jsonl")
    assert atom["atom_kind"] == "m5max_sweep_result"
    assert atom["macos_cpu_calibrated_tag"] == "[macOS-CPU calibrated]"
    assert atom["evidence_grade"] == "macOS-CPU-calibrated"
    assert atom["ready_for_gha_dispatch"] is True
    assert atom["score_claim"] is False
    assert atom["promotion_eligible"] is False
    assert atom["rank_or_kill_eligible"] is False
    assert atom["ready_for_exact_eval_dispatch"] is False
    assert atom["next_unblock_action"] == \
        "dispatch_GHA_cpu_eval_via_dispatch_cpu_eval_via_github_actions_py"
    assert atom["predicted_contest_cpu_gha_tag"] == \
        "[predicted; macos_x86_64_calibration_pr107]"


def test_atom_for_advisory_log_only_candidate(atoms_mod) -> None:
    row = {
        "candidate_id": "cand_y",
        "architecture_class": "h3_grayscale",
        "macos_cpu_score": 0.25,
        "macos_cpu_calibrated_tag": "[macOS-CPU advisory only]",
        "promotion_verdict": "LOG_ONLY",
        "drift_flag": "NA",
    }
    atom = atoms_mod._row_to_atom(row, "/p/results.jsonl")
    assert atom["evidence_grade"] == "macOS-CPU-advisory"
    assert atom["ready_for_gha_dispatch"] is False
    assert atom["next_unblock_action"] == "log_only_no_dispatch"
    assert atom["predicted_contest_cpu_gha_tag"] is None


def test_atom_for_failed_eval(atoms_mod) -> None:
    row = {
        "candidate_id": "cand_fail",
        "architecture_class": "hnerv",
        "macos_cpu_score": None,
        "macos_cpu_calibrated_tag": "[macOS-CPU calibrated]",
        "promotion_verdict": "EVAL_FAILED",
        "drift_flag": "NA",
    }
    atom = atoms_mod._row_to_atom(row, "/p/results.jsonl")
    assert atom["evidence_grade"] == "eval_failed"
    assert atom["ready_for_gha_dispatch"] is False
    assert atom["next_unblock_action"] == "investigate_eval_failure_then_resweep"


def test_atom_for_operator_decision_calibrated(atoms_mod) -> None:
    """Calibrated + OPERATOR_DECISION → still ready_for_gha_dispatch (silver-band)."""
    row = {
        "candidate_id": "cand_silver",
        "architecture_class": "hnerv",
        "macos_cpu_score": 0.193,
        "macos_cpu_calibrated_tag": "[macOS-CPU calibrated]",
        "promotion_verdict": "OPERATOR_DECISION",
        "drift_flag": "OK",
    }
    atom = atoms_mod._row_to_atom(row, "/p/results.jsonl")
    assert atom["ready_for_gha_dispatch"] is True
    assert atom["next_unblock_action"] == "operator_review_silver_band_proximity"


def test_atom_for_operator_decision_advisory_blocks_gha(atoms_mod) -> None:
    """Advisory + OPERATOR_DECISION → NOT ready_for_gha (no calibration anchor)."""
    row = {
        "candidate_id": "cand_silver_uncal",
        "architecture_class": "balle_scale_hyperprior",
        "macos_cpu_score": 0.193,
        "macos_cpu_calibrated_tag": "[macOS-CPU advisory only]",
        "promotion_verdict": "OPERATOR_DECISION",
        "drift_flag": "NA",
    }
    atom = atoms_mod._row_to_atom(row, "/p/results.jsonl")
    assert atom["ready_for_gha_dispatch"] is False


# ────────────────────────────────────────────────────────────────────────────
# /tmp guard (transient-evidence-trap)
# ────────────────────────────────────────────────────────────────────────────


def test_atoms_tool_rejects_tmp_output(atoms_mod, tmp_path: Path) -> None:
    src = tmp_path / "results.jsonl"
    src.write_text("")
    # Simulate CLI args
    sys_argv_backup = sys.argv
    try:
        sys.argv = ["m5max_sweep_to_atoms.py",
                    "--results-jsonl", str(src),
                    "--output-jsonl", "/tmp/atoms.jsonl"]
        with pytest.raises(SystemExit, match="/tmp paths are FORBIDDEN"):
            atoms_mod.main()
    finally:
        sys.argv = sys_argv_backup


# ────────────────────────────────────────────────────────────────────────────
# Round-trip: results.jsonl → atoms.jsonl
# ────────────────────────────────────────────────────────────────────────────


def test_atoms_roundtrip_writes_jsonl(atoms_mod, tmp_path: Path) -> None:
    src = tmp_path / "results.jsonl"
    src.write_text(
        json.dumps({"candidate_id": "a", "architecture_class": "hnerv",
                    "macos_cpu_score": 0.18,
                    "macos_cpu_calibrated_tag": "[macOS-CPU calibrated]",
                    "promotion_verdict": "AUTO_PROMOTE_GHA",
                    "drift_flag": "OK"}) + "\n"
        + json.dumps({"candidate_id": "b", "architecture_class": "hnerv",
                      "macos_cpu_score": 0.21,
                      "macos_cpu_calibrated_tag": "[macOS-CPU calibrated]",
                      "promotion_verdict": "LOG_ONLY",
                      "drift_flag": "OK"}) + "\n"
    )
    dst = tmp_path / "atoms.jsonl"
    sys_argv_backup = sys.argv
    try:
        sys.argv = ["m5max_sweep_to_atoms.py",
                    "--results-jsonl", str(src),
                    "--output-jsonl", str(dst)]
        atoms_mod.main()
    finally:
        sys.argv = sys_argv_backup
    rows = [json.loads(ln) for ln in dst.read_text().splitlines() if ln.strip()]
    assert len(rows) == 2
    assert rows[0]["candidate_id"] == "a"
    assert rows[0]["next_unblock_action"] == \
        "dispatch_GHA_cpu_eval_via_dispatch_cpu_eval_via_github_actions_py"
    assert rows[1]["next_unblock_action"] == "log_only_no_dispatch"


# ────────────────────────────────────────────────────────────────────────────
# Sweep-tool /tmp guard
# ────────────────────────────────────────────────────────────────────────────


def test_sweep_tool_rejects_tmp_output(sweep_mod) -> None:
    sys_argv_backup = sys.argv
    try:
        sys.argv = ["sweep_m5max_hnerv_cluster.py",
                    "--archives-dir", "/some/dir",
                    "--output-dir", "/tmp/sweep"]
        # Should reject before even reaching candidate loading
        if platform.system().lower() == "darwin" and platform.machine().lower() in ("arm64", "aarch64"):
            with pytest.raises(SystemExit, match="/tmp paths are FORBIDDEN"):
                sweep_mod.main()
        else:
            # Apple Silicon guard fires first on non-darwin or non-arm64
            with pytest.raises(SystemExit):
                sweep_mod.main()
    finally:
        sys.argv = sys_argv_backup


# ────────────────────────────────────────────────────────────────────────────
# Inflate.sh discovery
# ────────────────────────────────────────────────────────────────────────────


def test_discover_inflate_sh_adjacent(sweep_mod, tmp_path: Path) -> None:
    arch = tmp_path / "archive.zip"
    arch.write_bytes(b"stub")
    inf = tmp_path / "inflate.sh"
    inf.write_text("#!/bin/bash\n")
    found = sweep_mod._discover_inflate_sh(arch)
    assert found == inf


def test_discover_inflate_sh_submission_dir(sweep_mod, tmp_path: Path) -> None:
    arch = tmp_path / "archive.zip"
    arch.write_bytes(b"stub")
    sub = tmp_path / "submission_dir"
    sub.mkdir()
    inf = sub / "inflate.sh"
    inf.write_text("#!/bin/bash\n")
    found = sweep_mod._discover_inflate_sh(arch)
    assert found == inf


def test_discover_inflate_sh_returns_none(sweep_mod, tmp_path: Path) -> None:
    arch = tmp_path / "archive.zip"
    arch.write_bytes(b"stub")
    found = sweep_mod._discover_inflate_sh(arch)
    assert found is None
