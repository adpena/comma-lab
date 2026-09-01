from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from experiments import ddm_sfp1_scmdl_field_proposal_prep as sfp1


def _proposal(selector: sfp1.SelectorSpec) -> sfp1.ProposalSpec:
    return sfp1.ProposalSpec(
        proposal_id="test_proposal",
        rank=1,
        rank_status="PROJECTION",
        x_edit="assign realized terminal argmax where selector is true",
        assignment_source="mst1_cuda_terminal_argmax",
        selector=selector,
        g_edit=sfp1.GEditSpec(
            operation="refit_cross_group_causal_schedule",
            transition_order=(),
            contexts=("source_class", "target_class"),
            refit_required=True,
        ),
        refit_required=True,
        source_laws=("realized scorer cells",),
        prior_cell_relation="fresh",
    )


def test_schema_rejects_forbidden_assignment_source() -> None:
    invalid = replace(_proposal(sfp1.SelectorSpec(None, 1)), assignment_source="token_gt")
    with pytest.raises(ValueError, match="forbidden proposal material"):
        sfp1.validate_proposal(invalid)


def test_emitted_specs_are_projection_only_and_side_stream_free() -> None:
    specs = sfp1.proposal_specs()
    assert len(specs) == 3
    for spec in specs:
        sfp1.validate_proposal(spec)
        assert spec.rank_status == "PROJECTION"
        assert spec.assignment_source == "mst1_cuda_terminal_argmax"
        assert spec.g_edit is not None
        assert spec.g_edit.refit_required is True
        assert spec.g_edit.stored_side_stream is False


def test_null_control_writes_byte_identical_field(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sfp1, "SHAPE", (2, 2, 2))
    monkeypatch.setattr(sfp1, "PIXELS", 8)
    x = np.asarray([[[0, 1], [2, 3]], [[4, 3], [2, 1]]], dtype=np.uint8)
    target = np.flip(x, axis=2).copy()
    boundary = np.ones_like(x, dtype=np.uint8)
    output = tmp_path / "null.u8"

    result = sfp1._write_field(output, x, target, boundary, None, [0, 1])

    expected = x.tobytes()
    assert output.read_bytes() == expected
    assert result["changed_sites"] == 0
    assert result["sha256"] == hashlib.sha256(expected).hexdigest()


def test_real_edit_changes_only_selected_realized_disagreements(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sfp1, "SHAPE", (2, 2, 2))
    monkeypatch.setattr(sfp1, "PIXELS", 8)
    x = np.asarray([[[0, 0], [1, 1]], [[2, 2], [3, 3]]], dtype=np.uint8)
    target = np.asarray([[[0, 4], [1, 3]], [[4, 2], [0, 3]]], dtype=np.uint8)
    boundary = np.asarray([[[1, 1], [2, 1]], [[1, 1], [1, 2]]], dtype=np.uint8)
    spec = _proposal(sfp1.SelectorSpec(pair_rank_limit=1, boundary_distance_max=1))
    output = tmp_path / "candidate.u8"

    result = sfp1._write_field(output, x, target, boundary, spec, [1, 0])
    actual = np.frombuffer(output.read_bytes(), dtype=np.uint8).reshape((2, 2, 2))

    expected = x.copy()
    expected[1, 0, 0] = 4
    expected[1, 1, 0] = 0
    assert np.array_equal(actual, expected)
    assert result["changed_sites"] == 2
    assert result["transition_counts"] == {"2->4": 1, "3->0": 1}


def test_fold_table_does_not_admit_prior_or_gt_fields() -> None:
    folded = {row["law"]: row["disposition"] for row in sfp1.LAW_FOLD_TABLE}
    assert folded["WJ1 cost-times-error positions"] == "FOLD"
    assert folded["BHW2/JF2 benefit field"] == "CONTROL_ONLY"
    assert folded["FCD1 same-field diagonal"] == "FOLD"
    assert folded["WWC1 token-GT cone"] == "FOLD"


def test_resume_blocks_unverified_preexisting_payload(tmp_path) -> None:
    output = tmp_path / "candidate.u8"
    output.write_bytes(b"unowned-or-corrupt")
    with pytest.raises(ValueError, match="preserve and adjudicate"):
        sfp1._resume_or_write(output, None, lambda: {"should_not": "run"})


def test_delta_subset_checks_membership_and_assignments() -> None:
    base = np.zeros((2, 2, 2), dtype=np.uint8)
    smaller = base.copy()
    smaller[0, 0, 0] = 1
    larger = smaller.copy()
    larger[1, 1, 1] = 2
    assert sfp1.delta_is_subset(base, smaller, larger) == (True, True, 1, 2)

    conflicting = larger.copy()
    conflicting[0, 0, 0] = 3
    assert sfp1.delta_is_subset(base, smaller, conflicting)[:2] == (True, False)
