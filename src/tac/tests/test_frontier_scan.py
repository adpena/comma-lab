# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.frontier_scan import (
    Anchor,
    build_cpu_axis_optimal_payload,
    build_frontier_scan_payload,
    cpu_axis_family_for_anchor,
    refresh_frontier_citation_surfaces,
    render_frontier_scan_text,
    scan_frontier_citation_surface,
    select_cpu_optimal_per_family,
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
    assert payload["g1_cpu_axis_optimization"]["improvement_found"] is False
    assert payload["g1_cpu_axis_optimization"]["new_score_claim_valid"] is False
    assert (
        payload["g1_cpu_axis_optimization"]["score_claim_kind"]
        == "existing_anchor_rerank_no_new_score_claim"
    )
    assert payload["g1_cpu_axis_optimization"]["qualifying_cpu_anchor_count"] == 1
    assert "G1 CPU-AXIS OPTIMIZATION" in text


def test_g1_cpu_axis_selector_groups_by_family_and_filters_advisory() -> None:
    anchors = [
        Anchor(
            score=0.193,
            axis="contest_cpu",
            archive_sha256="1" * 64,
            hardware_substrate="linux_x86_64_cpu",
            source_path="fixture",
            extra={"lane_id": "lane_pr101_old"},
        ),
        Anchor(
            score=0.192,
            axis="contest_cpu",
            archive_sha256="2" * 64,
            hardware_substrate="linux_x86_64_cpu",
            source_path="fixture",
            extra={"lane_id": "lane_pr101_new"},
        ),
        Anchor(
            score=0.180,
            axis="contest_cpu",
            archive_sha256="3" * 64,
            hardware_substrate="macos_arm64",
            source_path="fixture",
            extra={"lane_id": "lane_pr101_macos_advisory"},
        ),
        Anchor(
            score=0.196,
            axis="contest_cpu",
            archive_sha256="4" * 64,
            hardware_substrate="linux_x86_64_cpu",
            source_path="fixture",
            extra={"lane_id": "lane_pr107_public_axis"},
        ),
        Anchor(
            score=0.197,
            axis="contest_cpu",
            archive_sha256="7" * 64,
            hardware_substrate="linux_x86_64_cpu",
            source_path="fixture",
            extra={"lane_id": "lane_pr106_component_prefix16_pr101grammar"},
        ),
        Anchor(
            score=0.205,
            axis="contest_cuda",
            archive_sha256="5" * 64,
            hardware_substrate="linux_x86_64_t4",
            source_path="fixture",
            extra={"lane_id": "lane_pr106_cuda_only"},
        ),
        Anchor(
            score=0.199,
            axis="contest_cpu",
            archive_sha256="6" * 64,
            hardware_substrate="linux_x86_64_cpu",
            source_path="fixture",
            extra={},
        ),
    ]

    assert cpu_axis_family_for_anchor(anchors[0]) == "pr101"
    assert cpu_axis_family_for_anchor(anchors[-1]) == "other"
    assert cpu_axis_family_for_anchor(anchors[4]) == "pr106"
    per_family = select_cpu_optimal_per_family(anchors)
    assert per_family["pr101"].archive_sha256 == "2" * 64
    assert per_family["pr106"].archive_sha256 == "7" * 64
    assert per_family["pr107"].archive_sha256 == "4" * 64
    assert per_family["other"].archive_sha256 == "6" * 64


def test_continual_learning_loader_reads_live_accepted_anchor_history(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / ".omx/state/continual_learning_posterior.json",
        {
            "schema": "continual_learning_posterior_v1",
            "accepted_anchor_history": [
                {
                    "score_value": 0.19284757743677347,
                    "axis": "cpu",
                    "archive_sha256": "8" * 64,
                    "hardware_substrate": "linux_x86_64_gha_cpu",
                    "architecture_class": "pr101_lossy_coarsening",
                    "evidence_tag": "[contest-CPU GHA Linux x86_64]",
                    "observed_at_utc": "2026-05-09T02:03:11+00:00",
                    "archive_bytes": 178262,
                },
                {
                    "score_value": 0.2066181354574151,
                    "axis": "cuda",
                    "archive_sha256": "9" * 64,
                    "hardware_substrate": "linux_x86_64_t4",
                    "architecture_class": "lane_pr106_latent_sidecar_r2_pr101_grammar",
                    "evidence_tag": "[contest-CUDA]",
                    "observed_at_utc": "2026-05-11T18:09:15.320614+00:00",
                    "archive_bytes": 186780,
                },
            ],
        },
    )

    payload = build_frontier_scan_payload(tmp_path)

    assert payload["scan_stats"] == {
        "total_anchors": 2,
        "qualifying": 2,
        "excluded": 0,
    }
    assert payload["best_per_axis"]["contest_cpu"]["archive_sha256"] == "8" * 64
    assert payload["best_per_axis"]["contest_cuda"]["archive_sha256"] == "9" * 64
    buckets = payload["g1_cpu_axis_optimization"]["per_metadata_bucket_optimal"]
    assert buckets["pr101"]["archive_sha256"] == "8" * 64


def test_g1_cpu_axis_payload_reports_improvement_only_when_cpu_best_beats_frontier() -> None:
    anchors = [
        Anchor(
            score=0.192,
            axis="contest_cpu",
            archive_sha256="a" * 64,
            hardware_substrate="linux_x86_64_cpu",
            source_path="fixture",
            extra={"lane_id": "lane_pr101_frontier"},
        ),
        Anchor(
            score=0.189,
            axis="contest_cpu",
            archive_sha256="b" * 64,
            hardware_substrate="linux_x86_64_cpu",
            source_path="fixture",
            extra={"lane_id": "lane_pr107_better_cpu"},
        ),
    ]

    payload = build_cpu_axis_optimal_payload(anchors, current_frontier_cpu=0.192)

    assert payload["overall_cpu_optimal"]["archive_sha256"] == "b" * 64
    assert payload["delta_vs_current_frontier"] == pytest.approx(-0.003)
    assert payload["improvement_found"] is True
    assert payload["existing_anchor_selection_valid"] is True
    assert payload["new_score_claim_valid"] is False


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
        capture_output=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "pact_frontier_scan_v1"
    assert payload["best_per_axis"]["contest_cpu"]["archive_sha256"] == "6" * 64
    assert payload["g1_cpu_axis_optimization"]["schema"] == "g1_cpu_axis_optimal_archive_v1"


def test_refresh_frontier_citation_surfaces_updates_all_mirrors(
    tmp_path: Path,
) -> None:
    _write_frontier_fixture(tmp_path)
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True, exist_ok=True)
    for name in ("current_focus.md", "next_experiments.md"):
        (state / name).write_text(
            "\n".join(
                [
                    "# Mirror",
                    "",
                    "## Frontier",
                    "",
                    "- Canonical scanner-derived best CPU anchor:",
                    "  `0.199`",
                    "  `[contest-CPU; Linux x86_64 1:1]`, archive",
                    "  `" + "1" * 64 + "`,",
                    "  lane `stale_cpu_lane`.",
                    "  Refresh from `reports/latest.md` and",
                    "  `.omx/state/canonical_frontier_pointer.json`; this file is a mirror, not a",
                    "  frontier source of truth.",
                    "- Canonical scanner-derived best CUDA anchor:",
                    "  `0.300`",
                    "  `[contest-CUDA T4]`, archive",
                    "  `" + "2" * 64 + "`,",
                    "  lane `stale_cuda_lane`.",
                    "- A1 remains the control arm.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    refresh = refresh_frontier_citation_surfaces(
        tmp_path,
        checked_at_utc="2026-05-28T00:00:00Z",
    )

    assert refresh["changed"]["reports/latest.md"]["changed"] is True
    assert refresh["changed"][".omx/state/current_focus.md"]["changed"] is True
    assert refresh["changed"][".omx/state/next_experiments.md"]["changed"] is True
    payload = build_frontier_scan_payload(tmp_path)
    assert payload["drift"] == []
    assert all(not rows for rows in payload["frontier_citation_surface_drift"].values())
    latest = (tmp_path / "reports/latest.md").read_text(encoding="utf-8")
    assert "2026-05-28T00:00:00Z" in latest
    assert "0.1920513169" in latest
    focus = (state / "current_focus.md").read_text(encoding="utf-8")
    assert "0.192051316881" in focus
    assert "lane_cpu_frontier" in focus
    assert "this file is a mirror, not a\n  frontier source of truth" in focus


def test_scan_best_anchor_cli_refreshes_citation_surfaces(tmp_path: Path) -> None:
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
            "--refresh-citation-surfaces",
            "--checked-at-utc",
            "2026-05-28T00:00:00Z",
            "--check-drift",
        ],
        check=False,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["drift"] == []
    assert payload["citation_surface_refresh"]["schema"] == "frontier_citation_surface_refresh_v1"


def test_cpu_axis_optimal_archive_selector_cli(tmp_path: Path) -> None:
    _write_frontier_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/cpu_axis_optimal_archive_selector.py"),
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
        check=False,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "g1_cpu_axis_optimal_archive_v1"
    assert payload["overall_cpu_optimal"]["archive_sha256"] == "6" * 64
    assert payload["qualifying_cpu_anchor_count"] == 1


def test_probe_g1_cpu_axis_re_rank_cli_reports_axis_gap(tmp_path: Path) -> None:
    _write_frontier_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/probe_g1_cpu_axis_re_rank.py"),
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
        check=False,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "g1_cpu_axis_re_rank_probe_v1"
    assert payload["g1_cpu_axis_optimization"]["schema"] == "g1_cpu_axis_optimal_archive_v1"
    assert payload["verdict"] == "FRONTIER_STABLE_VIA_RE_RANK"
    assert payload["score_claim_valid"] is False
    assert payload["axis_rank_cpu"][0]["archive_sha256"] == "6" * 64
    assert payload["predicted_cpu_score_pr101_lc_v2"] is None

    report_dir = tmp_path / "experiments/results/g1_cpu_axis_re_rank_fixture"
    report_run = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/probe_g1_cpu_axis_re_rank.py"),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(report_dir),
        ],
        check=False,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert report_run.returncode == 0
    assert report_run.stdout.strip() == "experiments/results/g1_cpu_axis_re_rank_fixture/report.json"
    report_payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["verdict"] == "FRONTIER_STABLE_VIA_RE_RANK"


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
