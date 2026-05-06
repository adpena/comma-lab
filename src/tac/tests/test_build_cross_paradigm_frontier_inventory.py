from __future__ import annotations

import json
from pathlib import Path

from tools.build_cross_paradigm_frontier_inventory import (
    STATIC_ROWS,
    build_inventory,
    render_markdown,
)


REPO = Path(__file__).resolve().parents[3]


def test_cross_paradigm_inventory_is_deterministic_and_non_dispatching() -> None:
    first = build_inventory(repo_root=REPO)
    second = build_inventory(repo_root=REPO)

    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["score_claim"] is False
    assert first["dispatch_attempted"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    assert "requires_exact_cuda_auth_eval" in first["dispatch_blockers"]


def test_cross_paradigm_inventory_pins_required_score_path_rows() -> None:
    payload = build_inventory(repo_root=REPO)
    rows = {row["key"]: row for row in payload["rows"]}

    for key in (
        "categorical_qma9_clade_spade_openpilot",
        "lapose_motion_atom_allocator",
        "meta_lagrangian_cross_paradigm_allocator",
        "telescopic_foveation_field",
        "hnerv_per_tensor_context_entropy",
    ):
        assert key in rows
        assert rows[key]["score_claim"] is False
        assert rows[key]["ready_for_exact_eval_dispatch"] is False
        assert rows[key]["next_patch"]
        assert rows[key]["blockers"]

    categorical = rows["categorical_qma9_clade_spade_openpilot"]
    assert "categorical_masks" in categorical["paradigms"]
    assert "openpilot_priors" in categorical["paradigms"]
    assert categorical["status"] == "contract_and_candidate_readiness_landed_needs_byte_closed_candidate"
    assert "src/tac/categorical_candidate_readiness.py" in categorical["code_paths"]
    assert "src/tac/pr91_hpm1_readiness.py" in categorical["code_paths"]
    assert "tools/audit_categorical_candidate_readiness.py" in categorical["code_paths"]
    assert "tools/audit_pr91_hpm1_readiness.py" in categorical["code_paths"]
    assert "tools/build_categorical_candidate_fixture.py" in categorical["code_paths"]
    assert (
        "experiments/results/pr91_hpm1_readiness_20260506_codex/readiness.json"
        in categorical["evidence_paths"]
    )

    lapose = rows["lapose_motion_atom_allocator"]
    assert lapose["role"] == "proposal_allocator"
    assert "meta_lagrangian" in lapose["paradigms"]


def test_cross_paradigm_inventory_paths_are_current_on_main() -> None:
    payload = build_inventory(repo_root=REPO)

    assert payload["row_count"] == len(STATIC_ROWS)
    assert payload["missing_code_path_count"] == 0
    assert payload["missing_evidence_path_count"] == 0
    for row in payload["rows"]:
        assert row["path_audit"]["code"]["missing"] == []
        assert row["path_audit"]["evidence"]["missing"] == []


def test_cross_paradigm_inventory_markdown_is_operator_briefing() -> None:
    payload = build_inventory(repo_root=REPO)
    markdown = render_markdown(payload)

    assert "Cross-Paradigm Frontier Inventory" in markdown
    assert "Inventory-only orchestration artifact" in markdown
    assert "`categorical_qma9_clade_spade_openpilot`" in markdown
    assert "`meta_lagrangian_cross_paradigm_allocator`" in markdown
