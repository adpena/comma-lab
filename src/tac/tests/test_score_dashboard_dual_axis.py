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
                "device": "cpu",
                "n_samples": 600,
                "platform_system": "Linux",
                "platform_machine": "x86_64",
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
                "device": "cuda",
                "n_samples": 600,
                "gpu_t4_match": True,
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
