from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tools.audit_ddm_m5_multicoefficient_solve_coverage import (
    certificate_eligibility,
    transition_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_transition_rows_separates_helpful_from_collateral() -> None:
    labels = np.asarray([[[0, 0, 1, 1, 2, 2]]], dtype=np.uint8)
    control = np.asarray([[[1, 0, 0, 1, 1, 2]]], dtype=np.uint8)
    candidate = np.asarray([[[0, 1, 1, 1, 1, 2]]], dtype=np.uint8)
    rows = transition_rows(
        labels, control, candidate, ("Road", "Lane", "Undrivable")
    )
    assert rows["Road"] == {
        "class_id": 0,
        "sites": 2,
        "control_errors": 1,
        "helpful_flips": 1,
        "harmful_off_target_flips": 1,
        "persistent_errors": 0,
        "candidate_errors": 1,
        "zero_off_target_collateral": False,
    }
    assert rows["Lane"]["control_errors"] == 1
    assert rows["Lane"]["helpful_flips"] == 1
    assert rows["Lane"]["harmful_off_target_flips"] == 0
    assert rows["Undrivable"]["persistent_errors"] == 1


def test_transition_rows_refuses_out_of_contract_labels() -> None:
    labels = np.asarray([[[3]]], dtype=np.uint8)
    with pytest.raises(DirectDescriptionError, match="outside"):
        transition_rows(labels, labels, labels, ("Road", "Lane", "Undrivable"))


def test_certificate_eligibility_requires_every_universal_negative_leg() -> None:
    eligible, missing = certificate_eligibility(
        numeric_byte_box=True,
        finite_reachable_set_manifest=False,
        exhaustive_enumeration=False,
        exact_receiver_replay=True,
        isolated_per_stratum_solutions=False,
    )
    assert eligible is False
    assert missing == [
        "finite_reachable_set_manifest",
        "exhaustive_enumeration",
        "isolated_per_stratum_solutions",
    ]
    eligible, missing = certificate_eligibility(
        numeric_byte_box=True,
        finite_reachable_set_manifest=True,
        exhaustive_enumeration=True,
        exact_receiver_replay=True,
        isolated_per_stratum_solutions=True,
    )
    assert eligible is True
    assert missing == []


def test_committed_n600_receipt_preserves_transition_not_net_accounting() -> None:
    receipt_path = (
        REPO_ROOT
        / ".omx/research/ddm_m5_multicoefficient_solve_coverage_20260723T103457Z"
        / "ddm_m5_multicoefficient_solve_coverage_receipt.json"
    )
    receipt = json.loads(receipt_path.read_bytes())
    measurement = receipt["measurement"]
    aggregate = measurement["aggregate"]
    rows = measurement["per_stratum"]
    assert measurement["pair_count"] == 600
    assert measurement["all_pairs_checkpointed"] is True
    assert aggregate["helpful_flips"] == 232_540
    assert aggregate["harmful_off_target_flips"] == 129_218
    assert aggregate["net_errors_closed"] == 103_322
    assert aggregate["net_errors_closed"] == (
        aggregate["helpful_flips"] - aggregate["harmful_off_target_flips"]
    )
    assert all(not row["zero_off_target_collateral"] for row in rows.values())
    assert receipt["certification"]["catalog_366_true_scope_errors"] is None
