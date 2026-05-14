from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ALL_LANES = REPO / "tools" / "all_lanes_preflight.py"


def _load_all_lanes_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "all_lanes_operator_briefing_gate_test",
        ALL_LANES,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _base_briefing_payload() -> dict[str, object]:
    return {
        "dispatch_claim_summary": {
            "active_count": 0,
            "stale_nonterminal_count": 0,
            "unparsable_timestamp_count": 0,
            "invalid_lane_id_count": 0,
        },
        "exact_eval_packets": [],
        "non_dispatchable_readiness_artifacts": [
            {
                "kind": "fixture_blocked_readiness",
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
            }
        ],
    }


def test_operator_briefing_dispatch_gate_accepts_blocked_score_plausible_rows() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [
            {
                "lane_id": "lane_pr106_stacked",
                "one_liner": "--env STACKED_LATENT_ARCHIVE=<path/to/archive.zip>",
                "gate_condition": "requires sister empirical landings",
                "score_target_routing": {"active": True},
                "dispatch_routing": {
                    "active": False,
                    "status": "dispatch_gate_blocked",
                    "blockers": [
                        "gate_condition_not_satisfied",
                        "operator_one_liner_has_unresolved_placeholders",
                    ],
                },
                "ready_for_operator_dispatch": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "active_composition_lanes": [],
    }

    assert module._operator_briefing_dispatch_failures(payload) == []


def test_operator_briefing_dispatch_gate_rejects_placeholder_active_row() -> None:
    module = _load_all_lanes_module()
    row = {
        "lane_id": "lane_pr106_stacked",
        "one_liner": "--env STACKED_LATENT_ARCHIVE=<path/to/archive.zip>",
        "dispatch_routing": {"active": True, "status": "dispatch_active", "blockers": []},
        "ready_for_operator_dispatch": True,
        "ready_for_exact_eval_dispatch": False,
    }
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [row],
        "active_composition_lanes": [row],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert "composition_lanes:lane_pr106_stacked:active_with_operator_placeholder" in failures
    assert "active_composition_lanes:lane_pr106_stacked:active_row_has_placeholder" in failures


def test_operator_briefing_dispatch_gate_rejects_active_list_mismatch() -> None:
    module = _load_all_lanes_module()
    row = {
        "lane_id": "lane_pr106_stacked",
        "one_liner": "echo ready",
        "dispatch_routing": {"active": False, "status": "dispatch_gate_blocked", "blockers": ["x"]},
        "ready_for_operator_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
    }
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [row],
        "active_composition_lanes": [row],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert "active_composition_lanes:lane_pr106_stacked:active_row_not_dispatch_active" in failures
    assert "active_composition_lanes:lane_pr106_stacked:active_row_not_operator_ready" in failures
    assert any(
        failure.startswith("active_composition_lanes_does_not_match_dispatch_routing")
        for failure in failures
    )


def test_operator_briefing_dispatch_gate_rejects_invalid_claim_summary() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "dispatch_claim_summary": {
            "active_count": 0,
            "stale_nonterminal_count": 0,
            "unparsable_timestamp_count": 0,
            "invalid_lane_id_count": 1,
        },
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert "dispatch_claim_summary:invalid_lane_id_count:1" in failures


def test_operator_briefing_dispatch_gate_rejects_terminal_packet_commands() -> None:
    module = _load_all_lanes_module()
    payload = {
        **_base_briefing_payload(),
        "supplementary_lanes": [],
        "active_supplementary_lanes": [],
        "gated_lanes": [],
        "active_gated_lanes": [],
        "composition_lanes": [],
        "active_composition_lanes": [],
        "exact_eval_packets": [
            {
                "lane_id": "terminal_packet",
                "terminal_exact_eval_evidence_blockers": ["same_lane_terminal_negative"],
                "repeat_dispatch_allowed": True,
                "ready_for_submit": True,
                "commands": {"claim": "claim", "submit": "submit", "harvest": "harvest"},
            }
        ],
    }

    failures = module._operator_briefing_dispatch_failures(payload)

    assert (
        "exact_eval_packets:terminal_packet:terminal_evidence_not_suppressing_repeat_dispatch"
        in failures
    )
    assert "exact_eval_packets:terminal_packet:terminal_evidence_ready_for_submit" in failures
    assert "exact_eval_packets:terminal_packet:terminal_evidence_commands_not_suppressed" in failures
