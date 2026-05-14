# SPDX-License-Identifier: MIT
"""Tests for tools/check_lane_smoke_signal_nontrivial.py (PCC9).

The CRITICAL test reproduces the apogee_int4-class failure mode from the
forensic audit: a lane registry entry with `real_archive_empirical: true`
where the evidence points to a build_metadata.json with all-zero corrections.
PCC9 must flag it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PCC9 = REPO_ROOT / "tools" / "check_lane_smoke_signal_nontrivial.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_lane_smoke_test", PCC9)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_test_repo(tmp_path: Path, registry_lanes: list[dict], smoke_dirs: dict[str, dict]) -> Path:
    """Create a synthetic repo layout with a lane registry + smoke metadata."""
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "lane_registry.json").write_text(json.dumps({
        "schema_version": 1,
        "lanes": registry_lanes,
    }))
    for smoke_subpath, meta in smoke_dirs.items():
        smoke_dir = tmp_path / smoke_subpath
        smoke_dir.mkdir(parents=True, exist_ok=True)
        (smoke_dir / "build_metadata.json").write_text(json.dumps(meta))
    return tmp_path


def test_pcc9_catches_zero_delta_diagnostics_schema(tmp_path: Path) -> None:
    """Schema 1 (latent_sidecar): diagnostics.delta_q_min == max == mean == 0."""
    mod = _load_module()
    repo = _make_test_repo(
        tmp_path,
        registry_lanes=[{
            "id": "lane_test_zero_delta",
            "level": 2,
            "gates": {
                "real_archive_empirical": {
                    "status": True,
                    "evidence": "experiments/results/lane_test_zero_delta_smoke_v1/build_metadata.json bit-exact PR106",
                },
            },
        }],
        smoke_dirs={
            "experiments/results/lane_test_zero_delta_smoke_v1": {
                "diagnostics": {
                    "delta_q_min": 0,
                    "delta_q_max": 0,
                    "delta_q_mean": 0.0,
                    "scorer_available": False,
                },
            },
        },
    )
    violations = mod.scan_registry(repo)
    assert len(violations) == 1
    assert violations[0].lane_id == "lane_test_zero_delta"
    assert "zero_delta_smoke" in violations[0].reason


def test_pcc9_catches_search_mode_zero_schema(tmp_path: Path) -> None:
    """Schema 2 (yshift/lrl1): search_mode == 'zero'."""
    mod = _load_module()
    repo = _make_test_repo(
        tmp_path,
        registry_lanes=[{
            "id": "lane_test_search_mode_zero",
            "level": 2,
            "gates": {
                "real_archive_empirical": {
                    "status": True,
                    "evidence": "experiments/results/lane_test_search_mode_zero_smoke/build_metadata.json",
                },
            },
        }],
        smoke_dirs={
            "experiments/results/lane_test_search_mode_zero_smoke": {
                "search_mode": "zero",
                "n_pairs": 600,
            },
        },
    )
    violations = mod.scan_registry(repo)
    assert len(violations) == 1
    assert "zero_search_mode" in violations[0].reason


def test_pcc9_catches_advisory_proposal_combo(tmp_path: Path) -> None:
    """Schema 3 (any): tag=[advisory only] + score_claim=False + council_status=PROPOSAL."""
    mod = _load_module()
    repo = _make_test_repo(
        tmp_path,
        registry_lanes=[{
            "id": "lane_test_advisory_proposal",
            "level": 2,
            "gates": {
                "real_archive_empirical": {
                    "status": True,
                    "evidence": "experiments/results/lane_test_advisory_smoke/build_metadata.json",
                },
            },
        }],
        smoke_dirs={
            "experiments/results/lane_test_advisory_smoke": {
                "tag": "[advisory only]",
                "score_claim": False,
                "council_status": "PROPOSAL — pre-registered at L1",
                "search_mode": "gradient",  # NOT zero — ensure detection via tag+council
            },
        },
    )
    violations = mod.scan_registry(repo)
    assert len(violations) == 1
    assert "advisory_proposal_only" in violations[0].reason


def test_pcc9_does_not_flag_legitimate_empirical(tmp_path: Path) -> None:
    """A lane with non-zero corrections + non-advisory metadata must NOT be flagged."""
    mod = _load_module()
    repo = _make_test_repo(
        tmp_path,
        registry_lanes=[{
            "id": "lane_test_legit",
            "level": 2,
            "gates": {
                "real_archive_empirical": {
                    "status": True,
                    "evidence": "experiments/results/lane_test_legit_smoke/build_metadata.json",
                },
            },
        }],
        smoke_dirs={
            "experiments/results/lane_test_legit_smoke": {
                "diagnostics": {
                    "delta_q_min": -127,
                    "delta_q_max": 124,
                    "delta_q_mean": 0.32,
                    "scorer_available": True,
                },
                "search_mode": "gradient",
                "tag": "[empirical:contest-CUDA]",
                "score_claim": True,
                "council_status": "VALIDATED",
            },
        },
    )
    violations = mod.scan_registry(repo)
    assert violations == []


def test_pcc9_does_not_flag_unset_real_archive_empirical(tmp_path: Path) -> None:
    """Lanes without real_archive_empirical:true are out of scope."""
    mod = _load_module()
    repo = _make_test_repo(
        tmp_path,
        registry_lanes=[{
            "id": "lane_test_l1",
            "level": 1,
            "gates": {
                "real_archive_empirical": {"status": False, "evidence": ""},
            },
        }],
        smoke_dirs={},
    )
    violations = mod.scan_registry(repo)
    assert violations == []


def test_pcc9_strict_exits_nonzero(tmp_path: Path) -> None:
    """--strict mode exits 1 when violations exist."""
    mod = _load_module()
    repo = _make_test_repo(
        tmp_path,
        registry_lanes=[{
            "id": "lane_test_strict",
            "gates": {
                "real_archive_empirical": {
                    "status": True,
                    "evidence": "experiments/results/lane_test_strict_smoke/build_metadata.json",
                },
            },
        }],
        smoke_dirs={
            "experiments/results/lane_test_strict_smoke": {
                "diagnostics": {"delta_q_min": 0, "delta_q_max": 0, "delta_q_mean": 0.0},
            },
        },
    )
    rc = mod.main(["--repo-root", str(repo), "--strict"])
    assert rc == 1


def test_pcc9_against_live_registry_has_no_unresolved_smoke_promotions() -> None:
    """Live integration: current registry must not carry unresolved smoke promotions."""
    mod = _load_module()
    violations = mod.scan_registry(REPO_ROOT)
    assert violations == []
