from __future__ import annotations

import pytest

from tac.witness_dsl.int8_training_rungs_policy import (
    Int8TeacherForwardProposal,
    Int8WitnessQATProposal,
)


def test_int8_stubs_are_default_off_and_emit_no_invented_argv() -> None:
    for proposal in (Int8TeacherForwardProposal(), Int8WitnessQATProposal()):
        display = proposal.to_display_dict()
        assert display["enabled"] is False
        assert display["state"] == "off_unwired"
        assert display["wired"] is False
        assert display["live_trainer_argv"] == []
        assert display["required_quality_pairs"] == 600


def test_unwired_int8_stubs_refuse_enablement() -> None:
    with pytest.raises(ValueError, match="default-OFF"):
        Int8TeacherForwardProposal(enabled=True)
    with pytest.raises(ValueError, match="default-OFF"):
        Int8WitnessQATProposal(enabled=True)


def test_qat_is_stage_boundary_only() -> None:
    with pytest.raises(ValueError, match="stage boundary"):
        Int8WitnessQATProposal(stage="per_step")
