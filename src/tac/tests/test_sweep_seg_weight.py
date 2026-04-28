from __future__ import annotations

import json
from pathlib import Path

from experiments import sweep_seg_weight


class MockTrainingState:
    def __init__(self) -> None:
        self.last_good_checkpoint = Path("trial/checkpoint_good.pt")
        self.reverted_to: Path | None = None

    def revert_to_checkpoint(self, checkpoint: Path) -> None:
        self.reverted_to = checkpoint


def test_posenet_floor_auto_revert_logic() -> None:
    state = MockTrainingState()

    decision = sweep_seg_weight.maybe_revert_for_posenet_floor(
        state,
        current_posenet_loss=0.021,
        baseline_posenet_loss=0.010,
    )

    assert decision.triggered is True
    assert state.reverted_to == state.last_good_checkpoint
    assert decision.skip_remaining_epochs is True


def test_sweep_produces_result_json_with_all_5_trial_entries(tmp_path: Path) -> None:
    def fake_trial(seg_weight: int, trial_dir: Path, epochs: int) -> dict:
        return {
            "seg_weight": seg_weight,
            "status": "completed",
            "epochs_requested": epochs,
            "checkpoint": str(trial_dir / "renderer_best.pt"),
            "auth_score": 1.0 + seg_weight / 1000.0,
        }

    result_path = sweep_seg_weight.run_sweep(
        output_dir=tmp_path,
        trial_runner=fake_trial,
        epochs=250,
    )

    payload = json.loads(result_path.read_text())
    assert payload["seg_weights"] == [120, 150, 200, 300, 500]
    assert len(payload["trials"]) == 5
    assert [t["seg_weight"] for t in payload["trials"]] == [120, 150, 200, 300, 500]


def test_best_trial_selection_picks_minimum_auth_score() -> None:
    trials = [
        {"seg_weight": 120, "auth_score": 1.14},
        {"seg_weight": 150, "auth_score": 1.09},
        {"seg_weight": 200, "auth_score": 1.12},
    ]

    best = sweep_seg_weight.select_best_trial(trials)

    assert best["seg_weight"] == 150
    assert best["auth_score"] == 1.09


def test_claude_md_override_is_documented() -> None:
    src = Path(sweep_seg_weight.__file__).read_text()

    assert "INTENTIONAL_OVERRIDE" in src
    assert "segnet_loss_weight > 100" in src
