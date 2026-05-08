from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD = REPO_ROOT / "tools" / "score_dashboard.py"


def _load_dashboard_module():
    spec = importlib.util.spec_from_file_location("score_dashboard_under_test", DASHBOARD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dashboard_ingests_cpu_and_cuda_as_separate_score_axes(tmp_path: Path) -> None:
    dashboard = _load_dashboard_module()
    results = tmp_path / "experiments" / "results"
    cpu_dir = results / "cpu_eval"
    cuda_dir = results / "cuda_eval"
    cpu_dir.mkdir(parents=True)
    cuda_dir.mkdir(parents=True)

    (cpu_dir / "contest_auth_eval.adjudicated.json").write_text(
        json.dumps(
            {
                "canonical_score": 0.1966358879,
                "final_score": 0.20,
                "archive_size_bytes": 178_981,
                "archive_sha256": "a" * 64,
                "avg_segnet_dist": 0.00057599,
                "avg_posenet_dist": 0.00003460,
                "compression_rate": 0.004767,
                "n_samples": 600,
                "provenance": {
                    "device": "cpu",
                    "platform_system": "Linux",
                    "platform_machine": "x86_64",
                },
            }
        ),
        encoding="utf-8",
    )
    (cuda_dir / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                "canonical_score": 0.2283908312,
                "final_score": 0.23,
                "archive_size_bytes": 178_981,
                "archive_sha256": "a" * 64,
                "avg_segnet_dist": 0.00067565,
                "avg_posenet_dist": 0.00017347,
                "rate_unscaled": 0.004767,
                "n_samples": 600,
                "promotion_eligible": True,
                "score_claim_valid": True,
                "evidence_grade": "A++",
                "provenance": {
                    "device": "cuda",
                    "gpu_t4_match": True,
                },
            }
        ),
        encoding="utf-8",
    )

    rows = dashboard.scan(tmp_path, scan_root=results)
    by_axis = {row.score_axis: row for row in rows}

    assert set(by_axis) == {"contest_cpu", "contest_cuda"}
    assert by_axis["contest_cpu"].score == 0.1966358879
    assert by_axis["contest_cpu"].promotion_eligible is False
    assert by_axis["contest_cpu"].cpu_leaderboard_reproduction_eligible is True
    assert by_axis["contest_cuda"].promotion_eligible is True
    assert by_axis["contest_cuda"].rank_or_kill_eligible is True

    payload = json.loads(dashboard._format_json(rows))
    assert payload["axis_counts"] == {"contest_cpu": 1, "contest_cuda": 1}
    assert payload["axis_sort_policy"] == "ranked_within_score_axis_no_global_mixed_axis_rank"
    assert [row["score_axis"] for row in payload["rows"]] == ["contest_cuda", "contest_cpu"]
    assert [row["axis_rank"] for row in payload["rows"]] == [1, 1]
    json_cpu = next(row for row in payload["rows"] if row["score_axis"] == "contest_cpu")
    assert json_cpu["score"] == 0.1966358879
    assert json_cpu["cpu_leaderboard_reproduction_eligible"] is True
    assert json_cpu["promotion_eligible"] is False

    table = dashboard._format_table(rows)
    assert "axis_rank" in table
    assert "no global mixed-axis rank" in table
    assert table.index("contest_cuda") < table.index("contest_cpu")


def test_dashboard_surfaces_hardware_blockers_for_misdeclared_cpu_axis(tmp_path: Path) -> None:
    dashboard = _load_dashboard_module()
    results = tmp_path / "experiments" / "results"
    macos_dir = results / "macos_cpu_eval"
    macos_dir.mkdir(parents=True)

    (macos_dir / "contest_auth_eval.adjudicated.json").write_text(
        json.dumps(
            {
                "canonical_score": 0.1966358879,
                "archive_size_bytes": 178_981,
                "n_samples": 600,
                "score_axis": "contest_cpu",
                "evidence_grade": "contest-CPU-1to1",
                "promotion_eligible": True,
                "rank_or_kill_eligible": True,
                "provenance": {
                    "device": "cpu",
                    "platform_system": "Darwin",
                    "platform_machine": "arm64",
                },
            }
        ),
        encoding="utf-8",
    )

    rows = dashboard.scan(tmp_path, scan_root=results)

    assert len(rows) == 1
    row = rows[0]
    assert row.score_axis == "cpu_advisory"
    assert row.evidence_grade == "macOS-CPU advisory"
    assert row.promotion_eligible is False
    assert row.rank_or_kill_eligible is False
    assert row.hardware_compliance_blocker == "contest_cpu_requires_linux_x86_64"

    payload = json.loads(dashboard._format_json(rows))
    assert payload["axis_counts"] == {"cpu_advisory": 1}
    assert payload["rows"][0]["hardware_compliance_blocker"] == "contest_cpu_requires_linux_x86_64"
    assert "contest_cpu_requires_linux_x86_64" in dashboard._format_table(rows)


def test_dashboard_ignores_forged_top_level_hardware_fields(tmp_path: Path) -> None:
    dashboard = _load_dashboard_module()
    results = tmp_path / "experiments" / "results"
    forged_cpu_dir = results / "forged_cpu"
    forged_cuda_dir = results / "forged_cuda"
    forged_cpu_dir.mkdir(parents=True)
    forged_cuda_dir.mkdir(parents=True)

    (forged_cpu_dir / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                "canonical_score": 0.1966,
                "archive_size_bytes": 178_981,
                "n_samples": 600,
                "score_axis": "contest_cpu",
                "device": "cpu",
                "platform_system": "Linux",
                "platform_machine": "x86_64",
                "provenance": {
                    "device": "cpu",
                    "platform_system": "Darwin",
                    "platform_machine": "arm64",
                },
            }
        ),
        encoding="utf-8",
    )
    (forged_cuda_dir / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                "canonical_score": 0.2283,
                "archive_size_bytes": 178_981,
                "n_samples": 600,
                "device": "cuda",
                "gpu_t4_match": True,
                "promotion_eligible": True,
                "score_claim_valid": True,
                "evidence_grade": "A++",
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )

    rows = dashboard.scan(tmp_path, scan_root=results)
    by_dir = {Path(row.path).parent.name: row for row in rows}

    forged_cpu = by_dir["forged_cpu"]
    assert forged_cpu.score_axis == "cpu_advisory"
    assert forged_cpu.cpu_leaderboard_reproduction_eligible is False
    assert forged_cpu.hardware_compliance_blocker == "contest_cpu_requires_linux_x86_64"

    forged_cuda = by_dir["forged_cuda"]
    assert forged_cuda.score_axis == "unknown"
    assert forged_cuda.promotion_eligible is False
    assert forged_cuda.rank_or_kill_eligible is False
