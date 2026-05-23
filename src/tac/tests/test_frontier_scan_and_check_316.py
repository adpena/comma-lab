"""Tests for tac.frontier_scan + Catalog #316 anti-signal-loss gate.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
the fix (scan tool) + the self-protection (STRICT preflight Catalog #316)
+ the canonical helper (tac.frontier_scan) each need dedicated test
coverage so the bug class extincts at every surface.

Per Catalog #229 premise verification: tests pin the canonical-helper
contract (Anchor dataclass + collect/best_per_axis/detect_drift) AND
the gate's drift/signal-loss detection AND the gate's waiver mechanism
so future linters cannot regress any surface unobserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.frontier_scan import (
    QUALIFYING_HARDWARE,
    Anchor,
    best_per_axis,
    detect_drift,
    load_active_lane_dispatch_claims_anchors,
    load_continual_learning_anchors,
    scan_reports_latest_md,
)
from tac.preflight import (
    PreflightError,
    check_reports_latest_md_not_stale_vs_canonical_frontier,
)

# ----------------------------------------------------------------------------
# Anchor + canonical-helper contract
# ----------------------------------------------------------------------------


def test_anchor_canonical_axis_normalizes_short_form():
    a = Anchor(
        score=0.19,
        axis="cpu",
        archive_sha256="a" * 64,
        hardware_substrate="linux_x86_64_cpu",
        source_path="x",
    )
    assert a.canonical_axis() == "contest_cpu"
    b = Anchor(
        score=0.20,
        axis="cuda",
        archive_sha256="b" * 64,
        hardware_substrate="linux_x86_64_t4",
        source_path="x",
    )
    assert b.canonical_axis() == "contest_cuda"


def test_anchor_is_qualifying_filters_non_contest_hardware():
    bad_hw = Anchor(
        score=0.19,
        axis="cpu",
        archive_sha256="a" * 64,
        hardware_substrate="macos_arm64",
        source_path="x",
    )
    assert not bad_hw.is_qualifying()
    bad_axis = Anchor(
        score=0.19,
        axis="advisory",
        archive_sha256="a" * 64,
        hardware_substrate="linux_x86_64_cpu",
        source_path="x",
    )
    assert not bad_axis.is_qualifying()
    bad_score = Anchor(
        score=0.0,
        axis="cpu",
        archive_sha256="a" * 64,
        hardware_substrate="linux_x86_64_cpu",
        source_path="x",
    )
    assert not bad_score.is_qualifying()


def test_qualifying_hardware_pins_canonical_set():
    """Sanity-check the hardware set so a linter rename surfaces here."""
    expected = {
        "linux_x86_64_cpu",
        "linux_x86_64_t4",
        "linux_x86_64_a10g",
        "linux_x86_64_a100",
        "linux_x86_64_4090",
        "linux_x86_64_h100",
        "linux_x86_64_l40s",
        "linux_x86_64_gha_cpu",
    }
    assert set(QUALIFYING_HARDWARE) == expected


# ----------------------------------------------------------------------------
# best_per_axis + detect_drift
# ----------------------------------------------------------------------------


def _make_anchor(score, axis, sha_seed, hw="linux_x86_64_cpu"):
    seed_str = str(sha_seed)
    sha = (seed_str * ((64 // len(seed_str)) + 1))[:64].ljust(64, "0")
    return Anchor(
        score=score,
        axis=axis,
        archive_sha256=sha,
        hardware_substrate=hw,
        source_path="fixture",
    )


def test_best_per_axis_sorts_ascending_within_axis():
    anchors = [
        Anchor(0.20, "cpu", "a" * 64, "linux_x86_64_cpu", "x"),
        Anchor(0.19, "cpu", "b" * 64, "linux_x86_64_cpu", "x"),
        Anchor(0.25, "cuda", "c" * 64, "linux_x86_64_t4", "x"),
    ]
    best = best_per_axis(anchors)
    assert best["contest_cpu"][0].score == 0.19
    assert best["contest_cpu"][1].score == 0.20
    assert best["contest_cuda"][0].score == 0.25


def test_best_per_axis_excludes_non_qualifying_hardware():
    anchors = [
        Anchor(0.18, "cpu", "a" * 64, "macos_arm64", "x"),  # advisory
        Anchor(0.19, "cpu", "b" * 64, "linux_x86_64_cpu", "x"),  # qualifying
    ]
    best = best_per_axis(anchors)
    assert best["contest_cpu"][0].score == 0.19
    assert all(a.score != 0.18 for a in best.get("contest_cpu", []))


def test_detect_drift_flags_state_beats_cited():
    best = {
        "contest_cpu": [
            Anchor(0.19205, "cpu", "x" * 64, "linux_x86_64_cpu", "src"),
        ],
    }
    cited = {"contest_cpu": 0.198}
    drift = detect_drift(best, cited)
    assert len(drift) == 1
    assert drift[0].axis == "contest_cpu"
    assert drift[0].cited_score == pytest.approx(0.198)
    assert drift[0].best_score == pytest.approx(0.19205)
    assert drift[0].delta > 0


def test_detect_drift_tolerates_within_epsilon():
    best = {
        "contest_cpu": [
            Anchor(0.19200, "cpu", "x" * 64, "linux_x86_64_cpu", "src"),
        ],
    }
    cited = {"contest_cpu": 0.19200}  # exact match
    assert detect_drift(best, cited) == []
    # 1e-7 should still be tolerated (default 1e-6 epsilon)
    cited_tiny = {"contest_cpu": 0.192_000_5}
    assert detect_drift(best, cited_tiny) == []


def test_detect_drift_silent_when_no_citation():
    best = {"contest_cpu": [Anchor(0.19, "cpu", "x" * 64, "linux_x86_64_cpu", "x")]}
    assert detect_drift(best, {}) == []


# ----------------------------------------------------------------------------
# Source loaders (synthetic fixtures)
# ----------------------------------------------------------------------------


def test_load_continual_learning_anchors_parses_canonical_schema(tmp_path):
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    payload = {
        "anchors": [
            {
                "score_value": 0.19205,
                "axis": "cpu",
                "archive_sha256": "a" * 64,
                "hardware_substrate": "linux_x86_64_cpu",
                "lane_id": "lane_test",
                "evidence_grade": "[contest-CPU]",
            }
        ]
    }
    (state / "continual_learning_posterior.json").write_text(json.dumps(payload))
    anchors = load_continual_learning_anchors(tmp_path)
    assert len(anchors) == 1
    assert anchors[0].score == pytest.approx(0.19205)
    assert anchors[0].canonical_axis() == "contest_cpu"
    assert anchors[0].extra["lane_id"] == "lane_test"


def test_load_active_lane_dispatch_claims_anchors_parses_canonical_row(tmp_path):
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    row = (
        "| 2026-05-15T08:43:02Z | codex:harvest | lane_pr101_test | modal | "
        "pr101_test_modal_t4 | 2026-05-15T08:43:02Z | completed_recovered | "
        "Modal auth eval recovered; passed=True; archive_sha=" + ("c" * 64) + "; "
        "archive_bytes=178517; score_recomputed=0.19205; axis=cpu; "
        "hardware_substrate=linux_x86_64_cpu; posterior_update=accepted; "
        "posterior_n=63; posterior_reason=accepted |"
    )
    (state / "active_lane_dispatch_claims.md").write_text(row + "\n")
    anchors = load_active_lane_dispatch_claims_anchors(tmp_path)
    assert len(anchors) == 1
    assert anchors[0].score == pytest.approx(0.19205)
    assert anchors[0].canonical_axis() == "contest_cpu"
    assert anchors[0].extra.get("lane_id") == "lane_pr101_test"


# ----------------------------------------------------------------------------
# scan_reports_latest_md (citation-surface parser)
# ----------------------------------------------------------------------------


def test_scan_reports_latest_md_extracts_axis_tagged_citations(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text(
        "# Report\n"
        "Best so far: 0.19205 [contest-CPU]\n"
        "CUDA: 0.22636 [contest-CUDA]\n"
    )
    cited = scan_reports_latest_md(tmp_path)
    # parser keeps the LOWEST cited score per axis (best-of-all-matches)
    assert cited["contest_cpu"] == pytest.approx(0.19205)
    assert cited["contest_cuda"] == pytest.approx(0.22636)


def test_scan_reports_latest_md_prefers_current_best_table_over_prose_deltas(
    tmp_path,
):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text(
        "# Report\n\n"
        "### Current best - last rechecked 2026-05-22T18:51Z\n\n"
        "| Axis | Best score | Archive sha256 (first 12) | Hardware | Lane |\n"
        "|---|---|---|---|---|\n"
        "| **`[contest-CPU Linux x86_64]`** | **0.1920282830** | "
        "`7a0da5d0fc32` | linux_x86_64_cpu | `lane_cpu` |\n"
        "| **`[contest-CUDA T4]`** | **0.2053300290** | "
        "`9cb989cef519` | linux_x86_64_t4 | `lane_cuda` |\n\n"
        "A local advisory row was `-0.0000010605785158157577` lower while "
        "mentioning the Linux x86_64 `[contest-CPU]` frontier.\n"
        "A compact DQS1 row was `0.000022368065015737626` below a prior "
        "`[contest-CPU]` frontier and had an exact `[contest-CUDA T4]` "
        "replay note.\n",
        encoding="utf-8",
    )

    cited = scan_reports_latest_md(tmp_path)

    assert cited == {
        "contest_cpu": pytest.approx(0.1920282830),
        "contest_cuda": pytest.approx(0.2053300290),
    }


def test_scan_reports_latest_md_does_not_fallback_when_current_best_is_malformed(
    tmp_path,
):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text(
        "# Report\n\n"
        "### Current best - last rechecked 2026-05-22T18:51Z\n\n"
        "| Axis | Best score |\n"
        "|---|---|\n"
        "| missing axis label | missing score |\n\n"
        "Historical prose says `0.000001` `[contest-CPU]` and "
        "`0.000022` `[contest-CUDA T4]`, but that is not the generated "
        "current-frontier table.\n",
        encoding="utf-8",
    )

    assert scan_reports_latest_md(tmp_path) == {}


def test_scan_reports_latest_md_empty_on_no_axis_tag(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text("# Report\nScore was 0.19; no axis tag.\n")
    assert scan_reports_latest_md(tmp_path) == {}


# ----------------------------------------------------------------------------
# Catalog #316 STRICT gate — drift detection
# ----------------------------------------------------------------------------


def _stage_drifted_state(tmp_path: Path) -> Path:
    """Stage a fixture where state has 0.19205 but reports/latest.md cites
    a worse 0.198 — gate must flag."""
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    (state / "continual_learning_posterior.json").write_text(
        json.dumps({
            "anchors": [
                {
                    "score_value": 0.19205,
                    "axis": "cpu",
                    "archive_sha256": "a" * 64,
                    "hardware_substrate": "linux_x86_64_cpu",
                    "lane_id": "lane_test_frontier",
                }
            ]
        })
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text(
        "# Report\n"
        "Best so far: 0.198 [contest-CPU]  (stale)\n"
    )
    return tmp_path


def test_catalog_316_gate_warns_on_drift(tmp_path):
    repo = _stage_drifted_state(tmp_path)
    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        strict=False, verbose=False, repo_root=repo
    )
    assert len(violations) == 1
    assert "0.198" in violations[0] or "0.19205" in violations[0]


def test_catalog_316_gate_raises_in_strict_mode_on_drift(tmp_path):
    repo = _stage_drifted_state(tmp_path)
    with pytest.raises(PreflightError):
        check_reports_latest_md_not_stale_vs_canonical_frontier(
            strict=True, verbose=False, repo_root=repo
        )


def test_catalog_316_gate_clean_when_citation_matches_state(tmp_path):
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    (state / "continual_learning_posterior.json").write_text(
        json.dumps({
            "anchors": [
                {
                    "score_value": 0.19205,
                    "axis": "cpu",
                    "archive_sha256": "a" * 64,
                    "hardware_substrate": "linux_x86_64_cpu",
                }
            ]
        })
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text("# Report\n0.19205 [contest-CPU]\n")
    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_catalog_316_gate_flags_missing_citation_as_signal_loss(tmp_path):
    """No citation at all when state has qualifying anchors → signal loss."""
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    (state / "continual_learning_posterior.json").write_text(
        json.dumps({
            "anchors": [
                {
                    "score_value": 0.19205,
                    "axis": "cpu",
                    "archive_sha256": "a" * 64,
                    "hardware_substrate": "linux_x86_64_cpu",
                }
            ]
        })
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text("# Report\nNo score citations at all.\n")
    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "NO recognizable" in violations[0] or "signal loss" in violations[0]


def test_catalog_316_gate_honors_waiver(tmp_path):
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    (state / "continual_learning_posterior.json").write_text(
        json.dumps({
            "anchors": [
                {
                    "score_value": 0.19205,
                    "axis": "cpu",
                    "archive_sha256": "a" * 64,
                    "hardware_substrate": "linux_x86_64_cpu",
                }
            ]
        })
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text(
        "<!-- FRONTIER_DRIFT_OK: paper-frozen snapshot 2026-05-17 -->\n"
        "# Report\n"
        "0.198 [contest-CPU]  intentionally stale per paper revision\n"
    )
    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_catalog_316_gate_rejects_placeholder_waiver(tmp_path):
    """Placeholder <rationale> literal must NOT count as a real waiver."""
    state = tmp_path / ".omx/state"
    state.mkdir(parents=True)
    (state / "continual_learning_posterior.json").write_text(
        json.dumps({
            "anchors": [
                {
                    "score_value": 0.19205,
                    "axis": "cpu",
                    "archive_sha256": "a" * 64,
                    "hardware_substrate": "linux_x86_64_cpu",
                }
            ]
        })
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latest.md").write_text(
        "<!-- FRONTIER_DRIFT_OK: <rationale> -->\n"
        "# Report\n"
        "0.198 [contest-CPU] stale\n"
    )
    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        strict=False, verbose=False, repo_root=tmp_path
    )
    # Placeholder rejected → drift surfaces
    assert len(violations) == 1


def test_catalog_316_gate_silent_when_repo_missing_state(tmp_path):
    """No state files → gate fail-OPEN per design (tac.frontier_scan returns empty)."""
    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_catalog_316_gate_live_repo_regression_guard():
    """Live-repo guard: this session's reports/latest.md FRONTIER section
    must be in sync with state. Bound at 0 (operator's permanent fix)."""
    from tac.preflight import REPO_ROOT
    violations = check_reports_latest_md_not_stale_vs_canonical_frontier(
        strict=False, verbose=False, repo_root=REPO_ROOT
    )
    assert violations == [], f"Catalog #316 live-repo drift: {violations}"
