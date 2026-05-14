# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SARC_REMOTE = REPO / "scripts" / "remote_lane_substrate_sar_coherent_pose_pairs.sh"


def test_sarc_remote_refuses_cuda_done_tag_when_trainer_failed() -> None:
    text = SARC_REMOTE.read_text(encoding="utf-8")
    assert 'if [ "$TRAIN_RC" -eq 0 ]; then' in text
    assert "auth_eval_artifact_present_but_trainer_failed" in text
    assert "refusing [contest-CUDA] completion tag" in text
    done_index = text.index("LANE_SARC_DONE [contest-CUDA]")
    rc_guard_index = text.index('if [ "$TRAIN_RC" -eq 0 ]; then')
    assert rc_guard_index < done_index


def test_sarc_remote_refuses_silent_green_when_auth_eval_missing() -> None:
    text = SARC_REMOTE.read_text(encoding="utf-8")
    assert "trainer returned rc=0 but auth eval artifact is missing" in text
    assert "TRAIN_RC=31" in text


def test_sarc_remote_verifies_active_claim_before_spend() -> None:
    text = SARC_REMOTE.read_text(encoding="utf-8")
    assert "claim_lane_dispatch.py\" summary" in text
    assert "dispatch_claim_summary.json" in text
    assert "payload.get(\"active\", [])" in text
    assert "no active dispatch claim" in text
    claim_index = text.index("stage_0_dispatch_claim_verified")
    nvdec_index = text.index("stage_0b_nvdec_probe_begin")
    assert claim_index < nvdec_index


def test_sarc_remote_closes_terminal_claim_on_exit() -> None:
    text = SARC_REMOTE.read_text(encoding="utf-8")
    assert "append_terminal_claim()" in text
    assert "completed_sarc_remote_driver" in text
    assert "failed_sarc_remote_driver_rc_" in text
    assert "failed_sarc_claim_verification_rc_" in text
    assert "--status \"$status\"" in text
    assert "trap cleanup EXIT" in text
    trap_index = text.index("trap cleanup EXIT")
    summary_index = text.index("claim_lane_dispatch.py\" summary")
    assert trap_index < summary_index
