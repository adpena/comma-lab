from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.repo_io import write_json
from tools.probe_pr103_arithmetic_retarget import _estimate_global_combo_state_count


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools/plan_pr103_arithmetic_transform.py"
RETARGET_SCRIPT = REPO / "tools/probe_pr103_arithmetic_retarget.py"
MATERIALIZE_SCRIPT = REPO / "tools/materialize_pr103_arithmetic_histogram_candidate.py"


def test_plan_pr103_arithmetic_transform_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    manifest = tmp_path / "schema_manifest.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    write_json(manifest, _manifest())

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--target-label",
            "blocks.1.weight",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["target_stream"]["label"] == "blocks.1.weight"
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["tool_run_manifest"]["ready_for_exact_eval_dispatch"] is False
    assert "candidate_runtime_adapter_missing" in plan["readiness_blockers"]
    assert "blocks.1.weight" in md_out.read_text(encoding="utf-8")


def test_plan_pr103_arithmetic_transform_cli_fails_closed_for_unknown_target(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "schema_manifest.json"
    write_json(manifest, _manifest())

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--target-label",
            "missing.weight",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "target label not found" in proc.stderr


def test_probe_pr103_arithmetic_retarget_cli_real_manifest_if_available(
    tmp_path: Path,
) -> None:
    manifest = (
        REPO
        / "experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json"
    )
    if not manifest.exists():
        pytest.skip("local PR103 schema refresh manifest is not present")
    json_out = tmp_path / "probe.json"
    md_out = tmp_path / "probe.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(RETARGET_SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--target-label",
            "stem.weight",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["target_stream"]["label"] == "stem.weight"
    assert report["score_claim"] is False
    assert report["ready_for_archive_preflight"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    assert "stem.weight" in md_out.read_text(encoding="utf-8")


def test_probe_pr103_arithmetic_coordinate_search_cli_real_manifest_if_available(
    tmp_path: Path,
) -> None:
    manifest = (
        REPO
        / "experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json"
    )
    if not manifest.exists():
        pytest.skip("local PR103 schema refresh manifest is not present")
    json_out = tmp_path / "coordinate.json"
    md_out = tmp_path / "coordinate.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(RETARGET_SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--target-label",
            "stem.weight",
            "--probe-mode",
            "coordinate-search",
            "--top-symbols",
            "2",
            "--deltas=-1,1",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["schema"] == "pr103_arithmetic_histogram_coordinate_probe_v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["search_config"]["candidate_count"] > 0
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    assert "Coordinate Probe" in md_out.read_text(encoding="utf-8")


def test_probe_pr103_arithmetic_beam_search_cli_real_manifest_if_available(
    tmp_path: Path,
) -> None:
    manifest = (
        REPO
        / "experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json"
    )
    if not manifest.exists():
        pytest.skip("local PR103 schema refresh manifest is not present")
    json_out = tmp_path / "beam.json"
    md_out = tmp_path / "beam.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(RETARGET_SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--target-label",
            "stem.weight",
            "--probe-mode",
            "beam-search",
            "--top-symbols",
            "2",
            "--deltas=-1,1",
            "--rounds",
            "2",
            "--beam-width",
            "2",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["schema"] == "pr103_arithmetic_histogram_beam_probe_v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["search_config"]["evaluated_candidate_count"] > 0
    assert report["best_candidate"]["moves"]
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    assert "Beam Probe" in md_out.read_text(encoding="utf-8")


def test_probe_pr103_arithmetic_global_combo_cli_real_manifest_if_available(
    tmp_path: Path,
) -> None:
    manifest = (
        REPO
        / "experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json"
    )
    beam0 = (
        REPO
        / ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_0_weight_beam_probe.json"
    )
    beam1 = (
        REPO
        / ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_1_weight_beam_probe.json"
    )
    if not manifest.exists() or not beam0.exists() or not beam1.exists():
        pytest.skip("local PR103 schema refresh and beam reports are not present")
    json_out = tmp_path / "global_combo.json"
    md_out = tmp_path / "global_combo.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(RETARGET_SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--probe-mode",
            "global-combo-search",
            "--beam-probe-report",
            str(beam0),
            "--beam-probe-report",
            str(beam1),
            "--top-per-stream",
            "3",
            "--beam-width",
            "8",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["schema"] == "pr103_arithmetic_histogram_global_combo_probe_v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "source_probe_delta_sum" in report["best_candidate"]
    assert "non_additivity_delta" in report["best_candidate"]
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    assert "Global Combo Probe" in md_out.read_text(encoding="utf-8")


def test_probe_pr103_arithmetic_global_combo_cli_refuses_slow_serial_search(
    tmp_path: Path,
) -> None:
    manifest = (
        REPO
        / "experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json"
    )
    beam_reports = [
        REPO
        / f".omx/research/pr103_arithmetic_transform_plans_20260510_codex/{name}_beam_probe.json"
        for name in (
            "blocks_0_weight",
            "blocks_1_weight",
            "blocks_2_weight",
            "blocks_3_weight",
            "stem_weight",
            "latent_hi_bytes",
        )
    ]
    if not manifest.exists() or not all(path.exists() for path in beam_reports):
        pytest.skip("local PR103 schema refresh and beam reports are not present")
    json_out = tmp_path / "slow.json"
    command = [
        sys.executable,
        str(RETARGET_SCRIPT),
        "--schema-manifest",
        str(manifest),
        "--probe-mode",
        "global-combo-search",
        "--top-per-stream",
        "25",
        "--beam-width",
        "512",
        "--global-combo-workers",
        "1",
        "--json-out",
        str(json_out),
    ]
    for report in beam_reports:
        command.extend(["--beam-probe-report", str(report)])

    proc = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "states per worker" in proc.stderr
    assert not json_out.exists()


def test_estimate_global_combo_state_count_matches_known_frontier() -> None:
    assert (
        _estimate_global_combo_state_count(
            stream_count=6,
            top_per_stream=20,
            beam_width=256,
        )
        == 21_966
    )
    assert _estimate_global_combo_state_count(
        stream_count=6,
        top_per_stream=25,
        beam_width=512,
    ) > 25_000


def test_materialize_pr103_arithmetic_histogram_candidate_cli_real_manifest_if_available(
    tmp_path: Path,
) -> None:
    manifest = (
        REPO
        / "experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.json"
    )
    beam = (
        REPO
        / ".omx/research/pr103_arithmetic_transform_plans_20260510_codex/blocks_0_weight_beam_probe.json"
    )
    if not manifest.exists() or not beam.exists():
        pytest.skip("local PR103 schema refresh and beam reports are not present")
    archive_out = tmp_path / "candidate.zip"
    json_out = tmp_path / "candidate.json"
    md_out = tmp_path / "candidate.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(MATERIALIZE_SCRIPT),
            "--schema-manifest",
            str(manifest),
            "--beam-probe-report",
            str(beam),
            "--output-archive",
            str(archive_out),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert archive_out.is_file()
    assert report["schema"] == "pr103_arithmetic_histogram_candidate_v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["candidate_archive"]["sha256"]
    assert report["candidate_archive"]["sha256"] != report["source_archive"]["sha256"]
    assert "candidate_runtime_adapter_missing" in report["readiness_blockers"]
    assert "Histogram Candidate" in md_out.read_text(encoding="utf-8")


def _manifest() -> dict:
    return {
        "planning_only": True,
        "score_claim": False,
        "ready_for_schema_review": True,
        "source_archive": {
            "path": "experiments/results/pr103/archive.zip",
            "bytes": 178223,
            "sha256": "a" * 64,
            "member_name": "x",
            "member_bytes": 178111,
            "member_sha256": "b" * 64,
        },
        "merged_arithmetic_stream": {
            "source_bytes": 153856,
            "source_sha256": "c" * 64,
            "decoded_symbol_count": 237561,
            "decoder_maybe_exhausted": True,
            "reencoded_byte_identical": True,
        },
        "next_arithmetic_schema_targets": [
            {
                "label": "stem.weight",
                "role": "ac_weight_tensor",
                "schema_index": 0,
                "symbol_count": 48384,
                "alphabet_size": 256,
                "decoded_symbols_sha256": "d" * 64,
                "observed_entropy_bytes_floor": 31582,
                "model_cross_entropy_bytes_floor": 31627,
                "model_gap_bytes_estimate": 46,
            },
            {
                "label": "blocks.1.weight",
                "role": "ac_weight_tensor",
                "schema_index": 4,
                "symbol_count": 46656,
                "alphabet_size": 256,
                "decoded_symbols_sha256": "e" * 64,
                "observed_entropy_bytes_floor": 32434,
                "model_cross_entropy_bytes_floor": 32478,
                "model_gap_bytes_estimate": 45,
            },
        ],
    }
