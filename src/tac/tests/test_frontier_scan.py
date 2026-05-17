# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.frontier_scan import (
    Anchor,
    build_frontier_scan_payload,
    render_frontier_scan_text,
    scan_frontier_citation_surface,
)
from tac.preflight import (
    PreflightError,
    check_reports_latest_md_not_stale_vs_canonical_frontier,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_frontier_fixture(repo_root: Path) -> None:
    _write_json(
        repo_root / ".omx/state/continual_learning_posterior.json",
        {
            "anchors": [
                {
                    "score_value": 0.192051316881,
                    "axis": "contest_cpu",
                    "archive_sha256": "6" * 64,
                    "hardware_substrate": "linux_x86_64_cpu",
                    "lane_id": "lane_cpu_frontier",
                },
                {
                    "score_value": 0.226210021693,
                    "axis": "contest_cuda",
                    "archive_sha256": "9" * 64,
                    "hardware_substrate": "linux_x86_64_t4",
                    "lane_id": "lane_cuda_frontier",
                },
                {
                    "score_value": 0.101,
                    "axis": "contest_cpu",
                    "archive_sha256": "a" * 64,
                    "hardware_substrate": "macos_cpu",
                    "lane_id": "lane_macos_advisory",
                },
            ]
        },
    )
    reports = repo_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "latest.md").write_text(
        "\n".join(
            [
                "Current: 0.193 [contest-CPU]",
                "| Axis | Best score |",
                "|---|---|",
                "| **`[contest-CUDA T4]`** | **0.3000000000** |",
                "- PR101: 0.193 [contest-CPU] -> we beat by 0.00095.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_frontier_scan_payload_uses_qualifying_best_per_axis(tmp_path: Path) -> None:
    _write_frontier_fixture(tmp_path)

    payload = build_frontier_scan_payload(tmp_path)

    assert payload["best_per_axis"]["contest_cpu"]["score"] == 0.192051316881
    assert payload["best_per_axis"]["contest_cuda"]["score"] == 0.226210021693
    assert payload["scan_stats"] == {
        "total_anchors": 3,
        "qualifying": 2,
        "excluded": 1,
    }
    drift_axes = {row["axis"] for row in payload["drift"]}
    assert drift_axes == {"contest_cpu", "contest_cuda"}
    text = render_frontier_scan_text(payload)
    assert "0.1920513169" in text
    assert "lane_cpu_frontier" in text
    assert "reports/latest.md" in text


def test_scan_best_anchor_cli_delegates_to_canonical_module(tmp_path: Path) -> None:
    _write_frontier_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/scan_best_anchor_per_axis.py"),
            "--repo-root",
            str(tmp_path),
            "--format",
            "json",
            "--check-drift",
        ],
        check=False,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "pact_frontier_scan_v1"
    assert payload["best_per_axis"]["contest_cpu"]["archive_sha256"] == "6" * 64


def test_preflight_frontier_scan_gate_blocks_stale_report(
    tmp_path: Path,
) -> None:
    _write_frontier_fixture(tmp_path)

    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        repo_root=tmp_path
    )

    assert len(violations) == 2
    assert "contest_cpu" in violations[0]
    with pytest.raises(PreflightError, match="Catalog #316"):
        check_reports_latest_md_not_stale_vs_canonical_frontier(
            strict=True,
            repo_root=tmp_path,
        )


def test_preflight_frontier_scan_gate_honors_header_waiver(
    tmp_path: Path,
) -> None:
    _write_frontier_fixture(tmp_path)
    reports_latest = tmp_path / "reports/latest.md"
    reports_latest.write_text(
        "<!-- FRONTIER_DRIFT_OK: paper snapshot frozen before rerun -->\n"
        + reports_latest.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert (
        check_reports_latest_md_not_stale_vs_canonical_frontier(
            strict=True,
            repo_root=tmp_path,
        )
        == []
    )


def test_preflight_frontier_scan_gate_blocks_state_doc_drift(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / ".omx/state/continual_learning_posterior.json",
        {
            "anchors": [
                {
                    "score_value": 0.192051316881,
                    "axis": "contest_cpu",
                    "archive_sha256": "6" * 64,
                    "hardware_substrate": "linux_x86_64_cpu",
                }
            ]
        },
    )
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    (reports / "latest.md").write_text("0.192051316881 [contest-CPU]\n")
    state = tmp_path / ".omx/state"
    (state / "current_focus.md").write_text(
        "A1 stale control language:\n"
        "`0.19284757743677347`\n"
        "`[contest-CPU; GHA Linux x86_64]`\n",
        encoding="utf-8",
    )

    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        repo_root=tmp_path
    )

    assert len(violations) == 1
    assert ".omx/state/current_focus.md" in violations[0]


def test_autopilot_frontier_threshold_uses_canonical_scan(monkeypatch) -> None:
    tools_path = Path(__file__).resolve().parents[3] / "tools"
    if str(tools_path) not in sys.path:
        sys.path.insert(0, str(tools_path))
    import cathedral_autopilot_autonomous_loop as loop
    import tac.frontier_scan as frontier_scan

    anchors = [
        Anchor(
            score=0.1915,
            axis="contest_cpu",
            archive_sha256="b" * 64,
            hardware_substrate="linux_x86_64_cpu",
            source_path=".omx/state/synthetic.json",
        )
    ]
    monkeypatch.setattr(frontier_scan, "collect_all_anchors", lambda _root: anchors)

    assert loop._resolve_canonical_frontier_threshold_cpu(default=0.192) == 0.1915


def test_frontier_citation_parser_reads_state_doc_split_axis(tmp_path: Path) -> None:
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    (state / "current_focus.md").write_text(
        "- Best local public-axis anchor:\n"
        "  `0.1920513168811056`\n"
        "  `[contest-CPU; GHA Linux x86_64]`.\n"
        "- Best CUDA anchor: `0.20533002902019143` `[contest-CUDA T4]`.\n"
        "- PR101: 0.193 [contest-CPU] -> we beat by 0.00095.\n",
        encoding="utf-8",
    )

    cited = scan_frontier_citation_surface(
        tmp_path,
        ".omx/state/current_focus.md",
    )

    assert cited["contest_cpu"] == 0.1920513168811056
    assert cited["contest_cuda"] == 0.20533002902019143
