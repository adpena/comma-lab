# SPDX-License-Identifier: MIT
"""Regression tests for the DQS1 drop-many greedy disambiguator tool."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "probe_dqs1_drop_many_greedy_independent_disambiguator.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "probe_dqs1_drop_many_greedy_independent_disambiguator_under_test",
        TOOL,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_greedy_ranking_prefers_empirical_anchor_over_rate_only_tie() -> None:
    mod = _load_tool_module()

    empirical = mod.PerPairIndependentDelta(
        pair_index=371,
        predicted_delta_score=mod.CANONICAL_RATE_DELTA_PER_BYTE + 8.3e-18,
        empirical_source="continual_learning_posterior",
    )
    rate_only = mod.PerPairIndependentDelta(
        pair_index=26,
        predicted_delta_score=mod.CANONICAL_RATE_DELTA_PER_BYTE,
        empirical_source="predicted_rate_only",
    )

    ranked = mod.rank_pairs_by_greedy_independent([rate_only, empirical])

    assert ranked[0].pair_index == 371
    assert ranked[0].empirical_source == "continual_learning_posterior"


def test_greedy_tool_json_keeps_drop_many_defer_false_authority(
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_tool_module()
    base_score = 0.19202894881608987
    empirical_best = base_score + mod.CANONICAL_RATE_DELTA_PER_BYTE + 8.3e-18
    posterior = _write_json(
        tmp_path / "posterior.json",
        {
            "accepted_anchor_history": [
                {
                    "architecture_class": "lane_dqs1_top32_gap_uleb",
                    "axis": "cpu",
                    "score_value": base_score,
                    "archive_sha256": "0" * 64,
                },
                {
                    "architecture_class": "lane_dqs1_drop_one_rank021_pair0371",
                    "axis": "cpu",
                    "score_value": empirical_best,
                    "archive_sha256": "1" * 64,
                },
                {
                    "architecture_class": "lane_dqs1_drop_one_rank010_pair0327",
                    "axis": "cpu",
                    "score_value": base_score + 3.34e-7,
                    "archive_sha256": "2" * 64,
                },
                {
                    "architecture_class": "lane_dqs1_pairset_diversity_k002",
                    "axis": "cpu",
                    "score_value": base_score + 2.0e-5,
                    "archive_sha256": "3" * 64,
                },
            ],
        },
    )
    frontier = _write_json(
        tmp_path / "frontier.json",
        {"our_local_frontier_contest_cpu": {"score": empirical_best}},
    )
    acquisition = _write_json(
        tmp_path / "acquisition.json",
        {
            "candidates": [
                {"acquisition_operation": {"op": "drop_one", "dropped_pair_index": 26}},
                {"acquisition_operation": {"op": "drop_one", "dropped_pair_index": 371}},
            ],
        },
    )

    rc = mod.main(
        [
            "--posterior",
            str(posterior),
            "--frontier-pointer",
            str(frontier),
            "--acquisition-plan",
            str(acquisition),
            "--output-dir",
            str(tmp_path / "out"),
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["build_1c_final_verdict"].startswith("NEGATIVE_COLLAPSE_TO_K1")
    refinement = payload["canonical_equation_candidate_refinement"][
        "refinement_field_proposed"
    ]
    assert refinement["empirical_k1_best_drop_one_pair_index"] == 371
    assert payload["greedy_top_k_sweep"][0]["selected_empirical_sources"] == [
        "continual_learning_posterior"
    ]
    assert payload["canonical_provenance"]["score_claim"] is False
    assert payload["canonical_provenance"]["ready_for_exact_eval_dispatch"] is False
    assert payload["catalog_313_probe_outcomes_row"]["verdict"] == "DEFER"
