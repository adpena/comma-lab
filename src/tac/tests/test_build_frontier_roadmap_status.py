from __future__ import annotations

from pathlib import Path

from tools.build_frontier_roadmap_status import (
    build_roadmap_status,
    dirty_paths_for_row,
    render_markdown,
)

REPO = Path(__file__).resolve().parents[3]


def test_dirty_paths_for_row_matches_exact_and_nested_paths() -> None:
    row = {
        "code_paths": ["src/tac/hnerv_wavelet_apply_transform.py"],
        "evidence_paths": ["experiments/results/example/manifest.json"],
    }

    assert dirty_paths_for_row(
        row,
        [
            "src/tac/hnerv_wavelet_apply_transform.py",
            "experiments/results/example",
            "src/tac/unrelated.py",
        ],
    ) == [
        "experiments/results/example",
        "src/tac/hnerv_wavelet_apply_transform.py",
    ]


def test_frontier_roadmap_status_is_non_dispatching_and_dirty_aware() -> None:
    payload = build_roadmap_status(
        repo_root=REPO,
        dirty_paths=[
            "src/tac/hnerv_wavelet_apply_transform.py",
            "src/tac/sensitivity_map.py",
        ],
    )
    rows = {row["key"]: row for row in payload["rows"]}

    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["row_count"] == 12
    assert payload["dirty_path_count"] == 2
    assert "requires_exact_cuda_auth_eval" in payload["dispatch_blockers"]
    assert payload["next_comprehensive_tranche"]["score_claim"] is False
    assert payload["next_comprehensive_tranche"]["dispatch_attempted"] is False
    assert payload["next_comprehensive_tranche"]["ready_for_exact_eval_dispatch"] is False
    assert payload["next_comprehensive_tranche"]["schema"] == "next_comprehensive_tranche_v1"
    assert "requires_lane_dispatch_claim" in payload["next_comprehensive_tranche"]["dispatch_blockers"]

    wr01 = rows["hnerv_wavelet_wr01_apply"]
    assert wr01["role"] == "stacker_scorer_changing"
    assert wr01["missing_code_path_count"] == 0
    assert wr01["missing_evidence_path_count"] == 0
    assert wr01["readiness_stage"] == "blocked_by_dirty_worktree"
    assert wr01["safe_to_touch_now"] is False
    assert wr01["dirty_path_blockers"] == ["src/tac/hnerv_wavelet_apply_transform.py"]

    sensitivity = rows["sensitivity_omega_w_v3"]
    assert sensitivity["readiness_stage"] == "blocked_by_dirty_worktree"
    assert sensitivity["safe_to_touch_now"] is False
    assert sensitivity["dirty_path_blockers"] == ["src/tac/sensitivity_map.py"]

    categorical = rows["categorical_qma9_clade_spade_openpilot"]
    assert categorical["safe_to_touch_now"] is True
    assert categorical["readiness_stage"] == "needs_byte_closed_candidate_or_fixture"

    tranche = payload["next_comprehensive_tranche"]
    pools = tranche["candidate_pools"]
    assert "hnerv_wavelet_wr01_apply" not in pools["exact_eval_or_review"]
    assert "categorical_qma9_clade_spade_openpilot" in pools["needs_byte_closed_candidate"]
    workstreams = {stream["id"]: stream for stream in tranche["workstreams"]}
    assert workstreams["scorer_changing_mask_payload"]["keys"] == [
        "hnerv_wavelet_wr01_apply",
        "categorical_qma9_clade_spade_openpilot",
        "cmg3_predictive_mask_grammar",
    ]
    assert "no remote/GPU dispatch without an active lane claim" in tranche["global_acceptance_gates"]


def test_frontier_roadmap_status_markdown_is_operator_briefing() -> None:
    payload = build_roadmap_status(repo_root=REPO, dirty_paths=[])
    markdown = render_markdown(payload)

    assert "Frontier Roadmap Status" in markdown
    assert "Live-safe operator roadmap" in markdown
    assert "Next Comprehensive Tranche" in markdown
    assert "`rate_frontier_closure`" in markdown
    assert "`field_meta_selection`" in markdown
    assert "`hnerv_wavelet_wr01_apply`" in markdown
    assert "`categorical_qma9_clade_spade_openpilot`" in markdown
    assert "`stacker_scorer_changing`" in markdown
    assert "evidence" in markdown
    assert "next_unblocked_keys" in markdown
