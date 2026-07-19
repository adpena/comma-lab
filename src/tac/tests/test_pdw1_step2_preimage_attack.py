from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator

_WORKTREE = Path(__file__).resolve().parents[3]
_DRIVER = _WORKTREE / "experiments/pdw1_fp32_realization_first_inbox_point.py"
_SPEC = importlib.util.spec_from_file_location("pdw1_step2_driver", _DRIVER)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_reference_aimed_preimage_preserves_exact_target_numerators() -> None:
    operator = DisjointResizeOperator.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)
    target = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3) + 96
    reference = (np.arange(8 * 10 * 3, dtype=np.uint16).reshape(8, 10, 3) * 17 % 256).astype(np.uint8)
    selected = np.zeros((3, 4), dtype=bool)
    selected[0, 1] = True
    selected[2, 3] = True

    frame, proof = _MODULE.reference_aimed_exact_target_preimage(
        operator,
        target,
        reference,
        selected,
        max_nodes_per_block=100_000,
    )
    numerators, denominator = operator.apply_numerators(frame)

    assert np.array_equal(numerators, target.astype(np.int64) * denominator)
    assert proof["selected_scorer_cells"] == 2
    assert proof["selected_channel_blocks"] == 6
    assert proof["status_counts"]["FEASIBLE_EXACT"] == 6
    assert proof["diophantine_infeasible_channel_blocks"] == 0
    assert proof["solver_budget_channel_blocks"] == 0
    assert proof["exact_rational_target_numerators"] is True


def test_reference_aimed_preimage_rejects_nonboolean_selection() -> None:
    operator = DisjointResizeOperator.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)
    target = np.full((3, 4, 1), 127, dtype=np.uint8)
    reference = np.full((8, 10, 1), 127, dtype=np.uint8)

    with pytest.raises(ValueError, match="selected_cells"):
        _MODULE.reference_aimed_exact_target_preimage(
            operator,
            target,
            reference,
            np.zeros((3, 4), dtype=np.uint8),
        )


def test_step2_ev_rank_uses_rank4_cost_and_refuses_interior() -> None:
    ledger = SimpleNamespace(
        total_flips=3,
        deficit=np.asarray([1.0, 1.0, 0.01], dtype=np.float32),
        annulus_dist=np.asarray([0.0, 0.0, 9.0], dtype=np.float32),
        c_wrong=np.asarray([0, 0, 0], dtype=np.int16),
        c_gt=np.asarray([1, 2, 1], dtype=np.int16),
        y=np.asarray([0, 0, 0], dtype=np.int16),
        x=np.asarray([0, 0, 0], dtype=np.int16),
    )
    composite = SimpleNamespace(
        down_col=np.asarray([[0.5, 0.5]], dtype=np.float64),
        down_row=np.asarray([[0.5, 0.5]], dtype=np.float64),
    )

    rank = _MODULE.build_step2_ev_rank(ledger, composite)

    # Equal margins: Road-Lane has the larger frozen head normal, hence lower
    # exact feature-space flip cost than Road-Undrivable.  The cheaper third
    # item is still refused because it lies beyond the #149 resize wall.
    assert rank.order.tolist() == [0, 1]
    assert rank.feature_flip_cost[0] < rank.feature_flip_cost[1]
    assert rank.eligible.tolist() == [True, True, False]
    assert rank.fisher_trace[0] == pytest.approx(rank.fisher_trace[1])


def test_secant_plan_keeps_only_probe_hard_accept_and_applies_prefix() -> None:
    operator = DisjointResizeOperator.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)
    base = [np.zeros((8, 10, 3), dtype=np.uint8)]
    reference = [np.full((8, 10, 3), 100, dtype=np.uint8)]
    ledger = SimpleNamespace(
        deficit=np.asarray([1.0, 1.0], dtype=np.float32),
        c_gt=np.asarray([1, 1], dtype=np.int16),
        pair_idx=np.asarray([0, 0], dtype=np.int32),
        y=np.asarray([0, 1], dtype=np.int16),
        x=np.asarray([0, 1], dtype=np.int16),
    )

    plan = _MODULE.build_secant_repair_plan(
        base,
        reference,
        ledger,
        np.asarray([0, 1], dtype=np.int64),
        operator,
        probe_signed_margin=np.asarray([1.0, 1.0]),
        probe_winner=np.asarray([1, 0]),
        safety_fraction=0.1,
    )
    repaired = _MODULE.apply_secant_repair_prefix(base, ledger, operator, plan, top_k=1)

    assert plan.order.tolist() == [0]
    assert plan.alpha.tolist() == pytest.approx([0.55])
    assert plan.probe_hard_accept.tolist() == [True, False]
    rs = operator.row_supports[0]
    cs = operator.col_supports[0]
    assert np.all(repaired[0][np.ix_(rs.indices, cs.indices, range(3))] == 55)
    assert np.count_nonzero(repaired[0]) == len(rs.indices) * len(cs.indices) * 3


def test_flip_patch_roundtrip_is_strict_and_deterministic() -> None:
    base = [np.zeros((4, 5, 3), dtype=np.uint8)]
    candidate = [base[0].copy()]
    candidate[0][1, 2] = (7, 8, 9)

    payload = _MODULE.encode_flip_patch(base, candidate)

    assert payload == _MODULE.encode_flip_patch(base, candidate)
    assert np.array_equal(_MODULE.apply_flip_patch(base, payload)[0], candidate[0])
    with pytest.raises(ValueError, match="length/trailer"):
        _MODULE.apply_flip_patch(base, payload + b"x")
