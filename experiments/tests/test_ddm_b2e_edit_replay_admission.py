"""Tests for the B2E edit-replay admission harness.

The load-bearing ones are :func:`test_prune_ladders_are_nested_and_row_structured`
(the edit algebra must match the shipped receiver) and the adjudication tests (the
pre-registered bar must not silently drift).

Tests here are scorer-free and payload-free: they exercise the construction and
adjudication logic on synthetic states, so they run anywhere.  The exact
reproduction of the ns1 section-A calibration against the real frontier archive is
recorded in the b2e landing memo, not asserted here, because it needs the SSD
payloads.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.ddm_b2e_edit_replay_admission import (  # noqa: E402
    CALIBRATION,
    EDIT_NAMES,
    REQUIRED_COLLAPSE_FACTOR,
    B2EError,
    _build_parser,
    _prune_rows,
    adjudicate,
    bias_tag,
    bounded_pair_ids,
    build_edit,
    weight_space_delta,
)
from tac.pr130_lift.editability_levers import FILM_ROW_FAMILY, film_row_order  # noqa: E402


def _synthetic_state() -> OrderedDict[str, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(20260816)
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    state["token_embed.weight"] = torch.randn(5, 32, generator=generator)
    state["frame_embed.weight"] = torch.randn(600, 8, generator=generator)
    state["coord_mix.weight"] = torch.randn(32, 40, 1, 1, generator=generator)
    state["coord_mix.bias"] = torch.randn(32, generator=generator)
    for index in range(4):
        state[f"blocks.{index}.dw.weight"] = torch.randn(32, 1, 3, 3, generator=generator)
        state[f"blocks.{index}.pw.weight"] = torch.randn(32, 32, 1, 1, generator=generator)
        state[f"blocks.{index}.film.weight"] = torch.randn(64, 8, generator=generator)
        state[f"blocks.{index}.film.bias"] = torch.randn(64, generator=generator)
    state["head.weight"] = torch.randn(3, 32, 3, 3, generator=generator)
    state["head.bias"] = torch.randn(3, generator=generator)
    return state


# ---------------------------------------------------------------------------
# edit construction
# ---------------------------------------------------------------------------


def test_edit_names_cover_the_three_calibrated_edits() -> None:
    assert set(EDIT_NAMES) == set(CALIBRATION)
    assert len(EDIT_NAMES) == 3


def test_prune_ladders_are_nested_and_row_structured() -> None:
    """keep87 must remove a strict superset of rows vs the keep75 marginal band."""
    state = _synthetic_state()
    keep87 = _prune_rows(state, drop="keep87")
    marginal = _prune_rows(state, drop="keep75_minus_keep87")

    for name in FILM_ROW_FAMILY:
        base = state[name]
        rows = int(base.shape[0])
        order = film_row_order(base)
        expected_keep87 = max(1, round(rows * 87 / 100.0))
        expected_keep75 = max(1, round(rows * 75 / 100.0))

        zeroed_87 = {i for i in range(rows) if bool(keep87[name][i].eq(0).all())}
        zeroed_marginal = {i for i in range(rows) if bool(marginal[name][i].eq(0).all())}

        assert zeroed_87 == set(order[expected_keep87:])
        assert zeroed_marginal == set(order[expected_keep75:expected_keep87])
        assert not (zeroed_87 & zeroed_marginal)


def test_prune_leaves_non_film_tensors_untouched() -> None:
    state = _synthetic_state()
    pruned = _prune_rows(state, drop="keep87")
    for name, value in state.items():
        if name in FILM_ROW_FAMILY:
            continue
        torch.testing.assert_close(pruned[name], value)


def test_build_edit_rejects_unknown_edit() -> None:
    with pytest.raises(B2EError):
        build_edit("not_an_edit", _synthetic_state())


def test_prune_rows_rejects_unknown_ladder() -> None:
    with pytest.raises(B2EError):
        _prune_rows(_synthetic_state(), drop="keep99")


# ---------------------------------------------------------------------------
# weight-space delta reporting
# ---------------------------------------------------------------------------


def test_weight_space_delta_is_zero_for_identity() -> None:
    state = _synthetic_state()
    report = weight_space_delta(state, OrderedDict((k, v.clone()) for k, v in state.items()))
    assert report["delta_l2"] == pytest.approx(0.0)
    assert report["tensors_touched"] == 0
    assert report["global_base_l2"] > 0.0


def test_weight_space_delta_flags_pose_critical_tensors() -> None:
    state = _synthetic_state()
    edited = _prune_rows(state, drop="keep87")
    report = weight_space_delta(state, edited)
    assert report["tensors_touched"] == len(FILM_ROW_FAMILY)
    assert set(report["pose_critical_touched"]) == set(FILM_ROW_FAMILY)
    assert 0.0 < report["relative_l2_percent"] < 100.0


def test_weight_space_delta_rejects_mismatched_tensor_sets() -> None:
    state = _synthetic_state()
    edited = OrderedDict(list(state.items())[:-1])
    with pytest.raises(B2EError):
        weight_space_delta(state, edited)


# ---------------------------------------------------------------------------
# bounded pair selection
# ---------------------------------------------------------------------------


def test_bounded_pairs_are_stratified_not_a_prefix() -> None:
    ids = bounded_pair_ids(seed=20260816, strata=8, pairs_per_stratum=4)
    assert len(ids) == 32
    assert len(set(ids)) == 32
    assert ids == sorted(ids)
    assert max(ids) > 500, "a stratified n32 must reach the final strata, not a prefix"
    assert min(ids) < 100


def test_bounded_pairs_are_seed_deterministic() -> None:
    first = bounded_pair_ids(seed=11, strata=10, pairs_per_stratum=3)
    second = bounded_pair_ids(seed=11, strata=10, pairs_per_stratum=3)
    third = bounded_pair_ids(seed=12, strata=10, pairs_per_stratum=3)
    assert first == second
    assert first != third


def test_bias_tag_forbids_granting_admission_on_a_subset() -> None:
    tag = bias_tag(32)
    assert tag["score_claim"] is False
    assert "never" in tag["admissible_use"].lower()
    assert "refute" in tag["admissible_use"].lower()


# ---------------------------------------------------------------------------
# adjudication against the pre-registered bar
# ---------------------------------------------------------------------------


def test_calibration_excesses_match_the_pinned_mp2_ratios() -> None:
    """Guards the pinned calibration against silent edits."""
    expected = {
        "mixed_q3q4": 4.9585,
        "film_row_prune_keep87": 4.6376,
        "film_row_prune_keep75_minus_keep87": 3.7669,
    }
    for name, ratio in expected.items():
        row = CALIBRATION[name]
        assert row["edited_pose"] / row["base_pose"] == pytest.approx(ratio, abs=1e-3)


def test_unmeasured_edits_are_pending_not_admitted() -> None:
    verdict = adjudicate({})
    assert verdict["overall"] == "PENDING_MEASUREMENT"
    assert verdict["admitted_count"] == 0
    for row in verdict["edits"].values():
        assert row["status"] == "NOT_MEASURED"
        assert row["verdict"] == "PENDING"


def test_insufficient_collapse_is_refused() -> None:
    # 10x collapse against a bar of 50x.
    calibration_excess = (
        CALIBRATION["film_row_prune_keep87"]["edited_pose"]
        / CALIBRATION["film_row_prune_keep87"]["base_pose"]
        - 1.0
    )
    excess = calibration_excess / 10.0
    verdict = adjudicate(
        {"film_row_prune_keep87": {"base_pose": 1e-4, "edited_pose": 1e-4 * (1 + excess)}}
    )
    row = verdict["edits"]["film_row_prune_keep87"]
    assert row["verdict"] == "REFUSED"
    assert row["collapse_factor"] == pytest.approx(10.0, rel=1e-6)
    assert verdict["overall"] == "REGIME_THESIS_INSTANCE_REFUTED"


def test_sufficient_collapse_is_admitted() -> None:
    calibration_excess = (
        CALIBRATION["mixed_q3q4"]["edited_pose"] / CALIBRATION["mixed_q3q4"]["base_pose"] - 1.0
    )
    excess = calibration_excess / 80.0
    verdict = adjudicate(
        {"mixed_q3q4": {"base_pose": 2e-4, "edited_pose": 2e-4 * (1 + excess)}}
    )
    assert verdict["edits"]["mixed_q3q4"]["verdict"] == "ADMITTED"
    assert verdict["overall"] == "REGIME_THESIS_SUPPORTED"


def test_edit_that_does_not_damage_pose_is_admitted_with_infinite_collapse() -> None:
    verdict = adjudicate({"mixed_q3q4": {"base_pose": 2e-4, "edited_pose": 1.8e-4}})
    row = verdict["edits"]["mixed_q3q4"]
    assert row["verdict"] == "ADMITTED"
    assert row["collapse_factor"] == float("inf")


def test_mixed_outcomes_report_partial() -> None:
    verdict = adjudicate(
        {
            "mixed_q3q4": {"base_pose": 1e-4, "edited_pose": 1.0001e-4},
            "film_row_prune_keep87": {"base_pose": 1e-4, "edited_pose": 4e-4},
        }
    )
    assert verdict["overall"] == "REGIME_THESIS_PARTIAL"
    assert verdict["admitted_count"] == 1
    assert verdict["measured_count"] == 2


def test_adjudicate_rejects_non_positive_base_pose() -> None:
    with pytest.raises(B2EError):
        adjudicate({"mixed_q3q4": {"base_pose": 0.0, "edited_pose": 1e-4}})


def test_required_collapse_bar_is_fifty() -> None:
    """The charter's pre-registered bar; changing it needs a new charter."""
    assert REQUIRED_COLLAPSE_FACTOR == 50.0


def test_adjudication_is_never_a_score_claim() -> None:
    verdict = adjudicate({"mixed_q3q4": {"base_pose": 1e-4, "edited_pose": 1e-4}})
    assert verdict["score_claim"] is False


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_parser_exposes_the_three_stages() -> None:
    parser = _build_parser()
    args = parser.parse_args(["replay"])
    assert args.stage == "replay"
    assert list(args.edits) == list(EDIT_NAMES)
    assert args.checkpoint is None


def test_parser_rejects_unknown_edit() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["replay", "--edits", "bogus"])


def test_admit_stage_requires_measured_input(tmp_path: Path) -> None:
    from experiments.ddm_b2e_edit_replay_admission import main

    with pytest.raises(B2EError):
        main(["admit"])


def test_admit_stage_round_trips_a_measured_file(tmp_path: Path, capsys) -> None:
    from experiments.ddm_b2e_edit_replay_admission import main

    measured = tmp_path / "measured.json"
    measured.write_text(
        json.dumps(
            {
                "axis": "[macOS-CPU advisory n32 stratified]",
                "measured": {"mixed_q3q4": {"base_pose": 1e-4, "edited_pose": 1.0001e-4}},
            }
        ),
        encoding="utf-8",
    )
    main(["admit", "--measured", str(measured)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "admit"
    assert payload["score_claim"] is False
    assert payload["adjudication"]["edits"]["mixed_q3q4"]["verdict"] == "ADMITTED"
    assert payload["measured_input_sha256"]
