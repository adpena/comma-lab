from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_staged_t1_script_does_not_open_phantom_claim() -> None:
    text = (REPO_ROOT / "scripts/staged_phase2_t1_balle_endtoend_dispatch.sh").read_text()

    assert "tools/claim_lane_dispatch.py claim" not in text
    assert "No lane claim opened" in text
    assert "experiments/modal_t1_balle_endtoend.py --execute" in text
    assert "experiments/modal_train_lane.py" not in text


def test_staged_phase3_script_does_not_open_phantom_claim() -> None:
    text = (REPO_ROOT / "scripts/staged_phase3_joint_scorer_renderer_codec_dispatch.sh").read_text()

    assert "tools/claim_lane_dispatch.py claim" not in text
    assert "No lane claim opened" in text
    assert "command-not-wired" in text
