import numpy as np

from tac.optimization.rw1_true_domain_instruments import (
    block_mask_from_scorer_mask,
    cap_receipt_from_solver_diagnostics,
    element_grade_vector,
    parse_cap_ladder,
)


def test_parse_cap_ladder_requires_positive_entries():
    assert parse_cap_ladder("25, 50,100", fallback=7) == (25, 50, 100)
    assert parse_cap_ladder("", fallback=7) == (7,)


def test_cap_receipt_marks_best_at_cap_as_cap_bound():
    diagnostics = {
        "selected": {"start": "dec", "stop_reason": "iteration_cap_best_at_cap"},
        "starts": [{"start": "dec", "curve": [{"step": 0}, {"step": 25}]}],
    }
    receipt = cap_receipt_from_solver_diagnostics(diagnostics, cap=25)
    assert receipt.stop_reason == "cap_bound"
    assert receipt.steps_run == 25
    assert receipt.still_descending is True


def test_cap_receipt_marks_plateau_as_converged():
    diagnostics = {
        "selected": {"start": "truth"},
        "starts": [
            {
                "start": "truth",
                "stop_reason": "plateau_no_proxy_improvement",
                "curve": [{"step": 0}, {"step": 10}],
            }
        ],
    }
    receipt = cap_receipt_from_solver_diagnostics(diagnostics, cap=25)
    assert receipt.stop_reason == "converged"
    assert receipt.steps_run == 10
    assert receipt.still_descending is False


def test_element_grade_vector_defaults_missing_to_unknown():
    vector = element_grade_vector(
        chain_name="unit",
        overrides={"init": ("OPTIMAL-RECEIPT", "receipt")},
    )
    assert vector["elements"]["init"]["grade"] == "OPTIMAL-RECEIPT"
    assert vector["elements"]["seed"]["grade"] == "UNKNOWN"
    assert vector["status"] == "PARTIALLY-CURED"


def test_block_mask_from_scorer_mask_uses_any_within_2x2_block():
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 2] = True
    mask[2, 5] = True
    out = block_mask_from_scorer_mask(mask)
    assert out.shape == (2, 3)
    assert out[0, 1]
    assert out[1, 2]
    assert int(out.sum()) == 2
