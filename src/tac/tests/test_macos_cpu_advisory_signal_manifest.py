"""Tests for ``tac.optimization.macos_cpu_advisory_signal``.

Per CLAUDE.md operator routing 2026-05-13 + the cascade reframe directive:
macOS-CPU is wired as a free first-class advisory proxy. These tests pin
the non-promotability invariants, the autopilot ranking-atom shape, the
sister-subagent calibration model auto-discovery + fallback semantics, the
hardware substrate detection contract, and the JSONL append guard rails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.macos_cpu_advisory_signal import (
    ALLOWED_USES,
    DISPATCH_BLOCKERS,
    EVIDENCE_GRADE,
    EVIDENCE_TAG,
    FORBIDDEN_USES,
    PR107_PLACEHOLDER_CALIBRATION,
    SCHEMA_VERSION,
    MacOSCPUAdvisorySignalError,
    append_manifest_row_to_jsonl,
    build_macos_cpu_advisory_signal_manifest,
    detect_macos_cpu_hardware_substrate,
    is_running_on_macos_arm64,
    json_text,
    load_calibration_model,
    load_observations,
)

# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _observations() -> list[dict[str, object]]:
    return [
        {
            "family": "pr106_hnerv_cluster",
            "variant_id": "r2",
            "archive_bytes": 186_822,
            "score": 0.1966,
            "d_seg": 0.067,
            "d_pose": 0.000034,
            "archive_sha256": "a" * 64,
            "samples_evaluated": 600,
        },
        {
            "family": "pr101_lossy_coarsening",
            "variant_id": "blocks4_7bit",
            "archive_bytes": 177_903,
            "score": 0.2024,
            "d_seg": 0.070,
            "d_pose": 0.000040,
            "archive_sha256": "b" * 64,
            "samples_evaluated": 600,
        },
    ]


# ----------------------------------------------------------------------------
# 1) Non-promotability invariants
# ----------------------------------------------------------------------------


def test_manifest_is_permanently_non_promotable() -> None:
    manifest = build_macos_cpu_advisory_signal_manifest(
        _observations(),
        source="fixture",
        run_id="r1",
    )
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ranking_only"] is True
    assert manifest["evidence_grade"] == EVIDENCE_GRADE
    assert manifest["evidence_tag"] == EVIDENCE_TAG
    assert manifest["schema"] == SCHEMA_VERSION


def test_per_row_invariants_are_set() -> None:
    manifest = build_macos_cpu_advisory_signal_manifest(
        _observations(),
        source="fixture",
        run_id="r2",
    )
    for row in manifest["rows"]:
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["ranking_only"] is True
        assert row["evidence_grade"] == EVIDENCE_GRADE
        assert row["evidence_tag"] == EVIDENCE_TAG
        assert row["proxy_evidence"] == "macos_cpu_advisory"


def test_atoms_are_rankable_but_not_promotable() -> None:
    manifest = build_macos_cpu_advisory_signal_manifest(
        _observations(),
        source="fixture",
        run_id="r3",
    )
    atoms = manifest["ranking_atoms"]
    assert len(atoms) == 2
    for atom in atoms:
        # Rankable=True is the WHOLE POINT — macOS-CPU participates in ranking.
        assert atom["rankable"] is True
        # But the row is still non-promotable.
        assert atom["promotion_eligible"] is False
        assert atom["ready_for_exact_eval_dispatch"] is False
        assert atom["score_claim"] is False
        assert atom["dispatchable"] is False
        # Proxy_evidence semantic tag.
        assert atom["proxy_evidence"] == "macos_cpu_advisory"


def test_device_contract_allowed_vs_forbidden_uses() -> None:
    manifest = build_macos_cpu_advisory_signal_manifest(
        _observations(),
        source="fixture",
        run_id="r4",
    )
    contract = manifest["device_contract"]
    assert contract["device_family"] == "macos_cpu"
    assert "autopilot_dispatch_ranking_pre_gpu_spend" in contract["allowed_uses"]
    assert "promotion_to_contest_cpu" in contract["forbidden_uses"]
    assert contract["promotion_requires_paired_linux_x86_64"] is True
    # Sanity: tuples match constants.
    assert set(contract["allowed_uses"]) == set(ALLOWED_USES)
    assert set(contract["forbidden_uses"]) == set(FORBIDDEN_USES)


def test_dispatch_blockers_propagate_to_manifest() -> None:
    manifest = build_macos_cpu_advisory_signal_manifest(
        _observations(),
        source="fixture",
        run_id="r5",
    )
    assert "macos_cpu_advisory_not_score_evidence" in manifest["dispatch_blockers"]
    assert "requires_paired_contest_cpu_gha_linux_x86_64_before_score_claim" in manifest["dispatch_blockers"]
    # Each atom inherits the same blockers.
    for atom in manifest["ranking_atoms"]:
        for blocker in DISPATCH_BLOCKERS:
            assert blocker in atom["dispatch_blockers"]


# ----------------------------------------------------------------------------
# 2) Projected contest-CPU score band
# ----------------------------------------------------------------------------


def test_projected_contest_cpu_score_band_uses_pr107_placeholder() -> None:
    manifest = build_macos_cpu_advisory_signal_manifest(
        _observations(),
        source="fixture",
        run_id="r6",
    )
    row = manifest["rows"][0]
    assert row["projected_contest_cpu_score_p50"] == pytest.approx(0.1966)
    # PR107 placeholder: drift_p90_abs=6.1e-6 * high_variance_multiplier=100.
    expected_half_width = PR107_PLACEHOLDER_CALIBRATION["drift_p90_abs"] * PR107_PLACEHOLDER_CALIBRATION["high_variance_multiplier"]
    assert row["advisory_band_half_width"] == pytest.approx(expected_half_width)
    assert row["projected_contest_cpu_score_low"] == pytest.approx(0.1966 - expected_half_width)
    assert row["projected_contest_cpu_score_high"] == pytest.approx(0.1966 + expected_half_width)


def test_custom_calibration_model_supersedes_placeholder() -> None:
    custom = {
        "schema": "macos_cpu_to_contest_cpu_drift_calibration.empirical.v1",
        "source": "sister_subagent_lane",
        "anchor_count": 12,
        "drift_p50_abs": 1e-5,
        "drift_p90_abs": 5e-5,
        "drift_p99_abs": 1e-4,
        "high_variance_multiplier": 3.0,
        "calibration_status": "loaded_from_sister_subagent_empirical_table",
    }
    manifest = build_macos_cpu_advisory_signal_manifest(
        _observations(),
        source="fixture",
        run_id="r7",
        calibration_model=custom,
    )
    assert manifest["calibration_model"]["anchor_count"] == 12
    assert manifest["calibration_model"]["evidence_grade"] == EVIDENCE_GRADE
    assert manifest["calibration_model"]["promotable"] is False
    row = manifest["rows"][0]
    assert row["advisory_band_half_width"] == pytest.approx(5e-5 * 3.0)


# ----------------------------------------------------------------------------
# 3) Hardware substrate detection
# ----------------------------------------------------------------------------


def test_hardware_substrate_default_for_macos() -> None:
    # On the actual Darwin ARM64 host we expect the canonical prefix.
    substrate = detect_macos_cpu_hardware_substrate()
    if is_running_on_macos_arm64():
        assert substrate.startswith("darwin_arm64_"), substrate
    else:
        # On non-macOS CI runners the function returns a non_* prefix.
        assert substrate.startswith("non_macos_arm64_"), substrate


def test_hardware_substrate_override_threads_through() -> None:
    manifest = build_macos_cpu_advisory_signal_manifest(
        _observations(),
        source="fixture",
        run_id="r8",
        hardware_substrate="darwin_arm64_apple_m5_max_cpu",
    )
    assert manifest["hardware_substrate"] == "darwin_arm64_apple_m5_max_cpu"
    for row in manifest["rows"]:
        assert row["hardware_substrate"] == "darwin_arm64_apple_m5_max_cpu"


# ----------------------------------------------------------------------------
# 4) Malformed observation rejection
# ----------------------------------------------------------------------------


def test_row_missing_family_is_rejected() -> None:
    with pytest.raises(MacOSCPUAdvisorySignalError):
        build_macos_cpu_advisory_signal_manifest(
            [{"variant_id": "v1", "archive_bytes": 10, "score": 0.1}],
            source="x",
            run_id="r9",
        )


def test_row_with_neither_score_nor_components_is_rejected() -> None:
    with pytest.raises(MacOSCPUAdvisorySignalError):
        build_macos_cpu_advisory_signal_manifest(
            [{"family": "a", "variant_id": "v1", "archive_bytes": 10}],
            source="x",
            run_id="r10",
        )


def test_row_with_negative_archive_bytes_is_rejected() -> None:
    with pytest.raises(MacOSCPUAdvisorySignalError):
        build_macos_cpu_advisory_signal_manifest(
            [{"family": "a", "variant_id": "v1", "archive_bytes": 0, "score": 0.1}],
            source="x",
            run_id="r11",
        )


def test_row_with_negative_d_pose_is_rejected() -> None:
    with pytest.raises(MacOSCPUAdvisorySignalError):
        build_macos_cpu_advisory_signal_manifest(
            [{"family": "a", "variant_id": "v1", "archive_bytes": 10, "d_pose": -0.001}],
            source="x",
            run_id="r12",
        )


# ----------------------------------------------------------------------------
# 5) Calibration model auto-discovery
# ----------------------------------------------------------------------------


def test_load_calibration_model_falls_back_to_placeholder(tmp_path: Path) -> None:
    # Empty directory -> placeholder.
    result = load_calibration_model(search_root=tmp_path)
    assert result["calibration_status"] == PR107_PLACEHOLDER_CALIBRATION["calibration_status"]
    assert result["promotable"] is False


def test_load_calibration_model_picks_up_sister_subagent_output(tmp_path: Path) -> None:
    sister_dir = tmp_path / "lane_macos_cpu_proxy_empirical_validation_20260513_20260513T120000Z"
    sister_dir.mkdir(parents=True)
    sister_payload = {
        "schema": "macos_cpu_to_contest_cpu_drift_calibration.empirical.v1",
        "source": "sister_subagent",
        "anchor_count": 6,
        "drift_p50_abs": 2e-6,
        "drift_p90_abs": 8e-6,
        "drift_p99_abs": 1.2e-5,
        "high_variance_multiplier": 5.0,
        "calibration_status": "loaded_from_sister_subagent_empirical_table",
    }
    (sister_dir / "calibration_model.json").write_text(json.dumps(sister_payload))
    result = load_calibration_model(search_root=tmp_path)
    assert result["anchor_count"] == 6
    assert result["calibration_status"] == "loaded_from_sister_subagent_empirical_table"
    assert "calibration_source_path" in result


def test_load_calibration_model_picks_latest_when_multiple_exist(tmp_path: Path) -> None:
    import time
    dir_a = tmp_path / "lane_macos_cpu_proxy_empirical_validation_20260513_20260513T100000Z"
    dir_a.mkdir(parents=True)
    (dir_a / "calibration_model.json").write_text(json.dumps({"anchor_count": 1, "drift_p90_abs": 1e-5}))
    time.sleep(0.05)
    dir_b = tmp_path / "lane_macos_cpu_proxy_empirical_validation_20260513_20260513T110000Z"
    dir_b.mkdir(parents=True)
    (dir_b / "calibration_model.json").write_text(json.dumps({"anchor_count": 9, "drift_p90_abs": 5e-5}))
    result = load_calibration_model(search_root=tmp_path)
    assert result["anchor_count"] == 9


def test_load_calibration_model_malformed_falls_back(tmp_path: Path) -> None:
    sister_dir = tmp_path / "lane_macos_cpu_proxy_empirical_validation_20260513_x"
    sister_dir.mkdir(parents=True)
    (sister_dir / "calibration_model.json").write_text("not valid json {{{")
    result = load_calibration_model(search_root=tmp_path)
    assert result["calibration_status"] == PR107_PLACEHOLDER_CALIBRATION["calibration_status"]


# ----------------------------------------------------------------------------
# 6) Observation file loading
# ----------------------------------------------------------------------------


def test_load_observations_json_list(tmp_path: Path) -> None:
    p = tmp_path / "obs.json"
    p.write_text(json.dumps(_observations()))
    rows = load_observations(p)
    assert len(rows) == 2
    assert rows[0]["family"] == "pr106_hnerv_cluster"


def test_load_observations_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "obs.jsonl"
    with p.open("w") as f:
        for row in _observations():
            f.write(json.dumps(row) + "\n")
    rows = load_observations(p)
    assert len(rows) == 2


def test_load_observations_dict_wrapper(tmp_path: Path) -> None:
    p = tmp_path / "obs.json"
    p.write_text(json.dumps({"observations": _observations()}))
    rows = load_observations(p)
    assert len(rows) == 2


def test_load_observations_invalid_payload_raises(tmp_path: Path) -> None:
    p = tmp_path / "obs.json"
    p.write_text(json.dumps("not a list or dict with rows"))
    with pytest.raises(MacOSCPUAdvisorySignalError):
        load_observations(p)


# ----------------------------------------------------------------------------
# 7) JSONL appender guard rails
# ----------------------------------------------------------------------------


def test_append_manifest_row_refuses_tmp_paths() -> None:
    row = {"family": "x", "variant_id": "v1", "evidence_grade": EVIDENCE_GRADE}
    with pytest.raises(ValueError, match="/tmp"):
        append_manifest_row_to_jsonl(row, output_path=Path("/tmp/x.jsonl"))


def test_append_manifest_row_refuses_promoted_rows(tmp_path: Path) -> None:
    row = {
        "family": "x",
        "variant_id": "v1",
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "score_claim": True,
    }
    with pytest.raises(MacOSCPUAdvisorySignalError):
        append_manifest_row_to_jsonl(row, output_path=tmp_path / "out.jsonl")


def test_append_manifest_row_writes_non_promoted_row(tmp_path: Path) -> None:
    row = {
        "family": "x",
        "variant_id": "v1",
        "archive_bytes": 1234,
        "score_macos_cpu": 0.5,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
    }
    out = tmp_path / "subdir" / "out.jsonl"
    append_manifest_row_to_jsonl(row, output_path=out)
    text = out.read_text(encoding="utf-8").strip()
    assert text
    parsed = json.loads(text)
    assert parsed["score_claim"] is False
    assert parsed["promotion_eligible"] is False
    assert parsed["ready_for_exact_eval_dispatch"] is False
    assert parsed["ranking_only"] is True


# ----------------------------------------------------------------------------
# 8) JSON text formatting
# ----------------------------------------------------------------------------


def test_json_text_is_deterministic_sorted_keys() -> None:
    obj = {"b": 2, "a": 1, "c": 3}
    text = json_text(obj)
    # Sorted keys: a, b, c
    assert text.find("\"a\":") < text.find("\"b\":") < text.find("\"c\":")
    assert text.endswith("\n")
