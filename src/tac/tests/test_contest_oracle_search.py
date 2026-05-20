# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.optimization.contest_oracle_search import build_lfv1_pair_queue


def test_lfv1_pair_queue_interleaves_byte_staircase() -> None:
    lattice = {
        "schema": "contest_atom_lattice_v1",
        "atom_count": 10,
        "pair_signal_overlap": {
            "top_pairs": [
                {
                    "pair_index": idx,
                    "venn_signature": "pair_component&xray_pair&xray_pixel",
                    "signals": ["pair_component", "xray_pair", "xray_pixel"],
                    "scope_kinds": ["pair", "frame", "pixel_region"],
                    "score_mass_sum": 100.0 - idx,
                    "max_waterfill_priority": 10.0 - idx,
                    "atom_ids": [f"pair:{idx}:mode:none"],
                }
                for idx in range(8)
            ]
        },
    }

    plan = build_lfv1_pair_queue(
        lattice,
        max_archive_delta_bytes=248,
        max_candidates=16,
        alpha_grid=[0.000005, 0.00001],
        radius_scale_grid=[0.45],
        power_grid=[0.8],
        origin_y_frac_grid=[0.38],
        min_pairs=1,
        max_pairs=8,
    )

    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    pair_counts = [row["archive_delta_budget"]["pair_count"] for row in plan["candidates"]]
    alphas = [row["params"]["alpha"] for row in plan["candidates"]]
    assert pair_counts == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    assert alphas == [0.000005, 0.00001] * 8
    assert plan["candidates"][0]["archive_delta_budget"]["estimated_archive_delta_bytes"] == 157
    assert plan["candidates"][14]["archive_delta_budget"]["estimated_archive_delta_bytes"] == 248


def test_lfv1_pair_queue_reports_grid_coverage_under_truncation() -> None:
    lattice = {
        "schema": "contest_atom_lattice_v1",
        "atom_count": 4,
        "pair_signal_overlap": {
            "top_pairs": [
                {
                    "pair_index": idx,
                    "venn_signature": "pair_component&xray_pair&xray_pixel",
                    "score_mass_sum": 10.0 - idx,
                    "max_waterfill_priority": 1.0,
                    "atom_ids": [f"pair:{idx}:mode:none"],
                }
                for idx in range(2)
            ]
        },
    }

    plan = build_lfv1_pair_queue(
        lattice,
        max_archive_delta_bytes=248,
        max_candidates=3,
        alpha_grid=[0.000005, 0.00001],
        radius_scale_grid=[0.45, 0.70],
        power_grid=[0.8],
        origin_y_frac_grid=[0.38],
        min_pairs=1,
        max_pairs=2,
    )

    coverage = plan["grid_coverage"]
    assert coverage["declared_param_combo_count"] == 4
    assert coverage["covered_param_combo_count"] == 3
    assert len(coverage["omitted_param_combos_after_truncation"]) == 1
    assert {row["grid"]["grid_index"] for row in plan["candidates"]} == {0, 1, 2}


def test_lfv1_pair_queue_stratifies_parameter_values_before_truncation() -> None:
    lattice = {
        "schema": "contest_atom_lattice_v1",
        "atom_count": 4,
        "pair_signal_overlap": {
            "top_pairs": [
                {
                    "pair_index": idx,
                    "venn_signature": "pair_component&xray_pair&xray_pixel",
                    "score_mass_sum": 10.0 - idx,
                    "max_waterfill_priority": 1.0,
                    "atom_ids": [f"pair:{idx}:mode:none"],
                }
                for idx in range(2)
            ]
        },
    }

    plan = build_lfv1_pair_queue(
        lattice,
        max_archive_delta_bytes=248,
        max_candidates=4,
        alpha_grid=[0.000005, 0.00001, 0.00002, 0.00004],
        radius_scale_grid=[0.45, 0.70],
        power_grid=[0.8],
        origin_y_frac_grid=[0.38],
        min_pairs=1,
        max_pairs=2,
    )

    assert {row["params"]["alpha"] for row in plan["candidates"]} == {
        0.000005,
        0.00001,
        0.00002,
        0.00004,
    }
    assert {row["params"]["radius_scale"] for row in plan["candidates"]} == {0.45, 0.70}
