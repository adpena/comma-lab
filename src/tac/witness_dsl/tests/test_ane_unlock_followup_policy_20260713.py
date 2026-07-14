from __future__ import annotations

import pytest

from tac.witness_dsl.ane_unlock_followup_policy_20260713 import (
    compile_ane_trainer_concurrency_ticket,
)


def test_packet_is_held_and_only_sidecar_is_treatment() -> None:
    solo = compile_ane_trainer_concurrency_ticket("trainer_solo")
    treatment = compile_ane_trainer_concurrency_ticket("trainer_plus_ane_sidecar")
    assert solo.base_config == treatment.base_config
    assert solo.n_pairs == treatment.n_pairs == 24
    assert solo.epochs == treatment.epochs == 4
    assert solo.sidecar_enabled is False
    assert treatment.sidecar_enabled is True
    assert solo.operator_go_required is True
    assert solo.score_claim is False


def test_packet_refuses_actuation_or_resize() -> None:
    with pytest.raises(PermissionError, match="prepare-only"):
        compile_ane_trainer_concurrency_ticket("trainer_solo", operator_go=True)
    with pytest.raises(ValueError, match="sealed"):
        compile_ane_trainer_concurrency_ticket("trainer_solo", n_pairs=600)
