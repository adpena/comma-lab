from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from experiments import ddm_js3_learned_implicit_conditioning as js3
from experiments import ddm_js4_pose_null_projected_conditioning as js4
from experiments import ddm_js5_projector_distilled_conditioning as js5


def _alpha_row(alpha: float, pose_pass: bool, robust: int) -> dict[str, object]:
    return {
        "alpha": alpha,
        "uint8_total_gate_pass": pose_pass,
        "metrics": {"projected_n600_robust_delta_flips": robust},
    }


def test_endpoint_budget_allocates_exact_endpoint_guard() -> None:
    base = -10e-6
    budgets = [js5.endpoint_budget(base, step, 4) for step in range(1, 5)]
    assert budgets == pytest.approx([-7e-6, -4e-6, -1e-6, 2e-6])
    assert all(later > earlier for earlier, later in itertools.pairwise(budgets))
    assert js5.accept_realized_pose(-1.1e-6, budgets[2])
    assert not js5.accept_realized_pose(-0.9e-6, budgets[2])


def test_alpha_decision_uses_largest_realized_useful_point() -> None:
    rows = [
        _alpha_row(1.0, False, -300),
        _alpha_row(0.5, False, -100),
        _alpha_row(0.25, True, 0),
        _alpha_row(0.125, True, -20),
        _alpha_row(0.0625, True, -3),
    ]
    decision = js5.choose_alpha(rows)
    assert decision.alpha == 0.125
    assert decision.qualifying_alphas == (0.125, 0.0625)
    assert decision.relinearize_every_accepted == 8


def test_alpha_decision_falls_back_without_claiming_robust_overlap() -> None:
    rows = [_alpha_row(1.0, False, -10), _alpha_row(0.5, True, 0), _alpha_row(0.25, True, 4)]
    decision = js5.choose_alpha(rows)
    assert decision.alpha == 0.5
    assert decision.qualifying_alphas == ()
    assert "no robust" in decision.reason


def test_projector_removes_active_row_component() -> None:
    correction = torch.zeros((1, 3, js3.H, js3.W), dtype=torch.float32)
    correction[:, 0] = 3.0
    basis = torch.zeros((1, js4.PROJECTOR_ROWS, 3 * js3.H * js3.W), dtype=torch.float32)
    basis[0, 0, : js3.H * js3.W] = 1.0 / np.sqrt(js3.H * js3.W)
    projected = js4.project_with_row_basis(correction, basis)
    assert torch.max(torch.abs(projected[:, 0])).item() < 2e-5
    assert torch.equal(projected[:, 1:], correction[:, 1:])


def test_bare_export_bakes_amplitude_and_never_serializes_projector(tmp_path: Path) -> None:
    torch.manual_seed(js3.SEED)
    model = js3.build_model(torch, functional, hidden=4, max_delta=6.0, qat=False)
    with torch.no_grad():
        model.head.weight.fill_(0.01)
        model.head.bias.fill_(0.02)
    value = torch.randn((1, js3.CHANNELS, 8, 8), dtype=torch.float32)
    expected = model(value) * 0.125
    exported = js5.export_bare_module(model, 0.125, tmp_path / "retained")
    assert exported["projector_payload_bytes"] == 0
    assert exported["scorer_at_decode"] is False
    assert exported["baked_max_delta"] == pytest.approx(0.75)
    coded_path = Path(exported["selected"]["coded"]["path"])
    decoded = js3.build_model(torch, functional, hidden=4, max_delta=0.75, qat=False)
    js3.load_decoded_state(decoded, js3.parse_module(coded_path.read_bytes()), torch)
    actual = decoded(value)
    assert torch.max(torch.abs(actual - expected)).item() < 0.01


def test_restore_training_state_rolls_back_model_optimizer_and_rng() -> None:
    torch.manual_seed(17)
    np.random.seed(17)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    snapshot = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state().clone(),
        "numpy_rng": np.random.get_state(),
    }
    expected_torch = torch.rand(3)
    expected_numpy = np.random.random(3)
    loss = model(torch.ones((1, 2))).sum()
    loss.backward()
    optimizer.step()
    js5._restore_training_state(model, optimizer, snapshot, torch)
    assert all(torch.equal(value, snapshot["model"][name]) for name, value in model.state_dict().items())
    assert torch.equal(torch.rand(3), expected_torch)
    assert np.array_equal(np.random.random(3), expected_numpy)


def test_training_projector_clones_inference_cache_for_autograd() -> None:
    class Router:
        active_basis: torch.Tensor | None = None

        def activate(self, _tokens: torch.Tensor) -> None:
            with torch.inference_mode():
                self.active_basis = torch.ones((1, 2, 3))

    router = Router()
    js5.activate_training_projector(router, torch.zeros((1, 1)))  # type: ignore[arg-type]
    assert router.active_basis is not None
    assert not router.active_basis.is_inference()
    value = torch.ones((1, 2, 3), requires_grad=True)
    (router.active_basis * value).sum().backward()
    assert value.grad is not None


def test_checkpoint_retains_live_ema_optimizer_rng_and_controller(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    state = {
        "accepted_steps": 3,
        "proposals": 5,
        "projector_manifest": {"path": "durable", "bytes": 1, "sha256": "a" * 64},
        "history": [{"accepted": True}],
        "config": {"alpha": 0.125},
    }
    record = js5._checkpoint(tmp_path, "stage", model, optimizer, ema, torch, state)
    checkpoint_path = Path(record["checkpoint"]["path"])
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert loaded["schema"] == "ddm_js5_checkpoint.v1"
    assert loaded["state"] == state
    assert set(loaded) >= {"model", "optimizer", "ema", "torch_rng", "numpy_rng"}
    latest = json.loads((tmp_path / "checkpoints/LATEST.json").read_text())
    assert latest["checkpoint"] == js3.file_record(checkpoint_path)


def test_relinearization_cadence_rejects_invalid_amplitude() -> None:
    assert js5.relinearization_cadence(1.0) == 1
    assert js5.relinearization_cadence(0.125) == 8
    with pytest.raises(ValueError):
        js5.relinearization_cadence(0.0)


def test_cached_stage_does_not_admit_pose_safe_zero_output() -> None:
    zero = {
        "kind": "ema",
        "bare": {
            "pose_guard_pass": True,
            "projected_n600_robust_delta_flips": 0,
            "projected_n600_delta_flips": 0,
        },
        "module_brotli_q11_bytes": 800,
    }
    stage = js5._normalize_stage_admission({"rows": [zero], "selected": zero})
    assert stage["selected"]["bare_pose_gate_pass"] is True
    assert stage["selected"]["bare_robust_movement_pass"] is False
    assert stage["selected"]["bare_admission_pass"] is False


def test_rung_seed_is_content_independent_and_hidden_specific() -> None:
    args = argparse.Namespace(
        stage_steps=(4, 12),
        shrink_ladder=(1.0, 0.5),
        lr=0.02,
        max_delta=6.0,
        ema_decay=0.99,
        grad_clip=5.0,
        checkpoint_every=4,
    )
    decision = js5.AlphaDecision(0.0625, 16, (), "test")
    hidden4 = js5._rung_config_for_hidden(args, decision, 4)
    hidden8 = js5._rung_config_for_hidden(args, decision, 8)
    assert hidden4["rung_seed"] == js3.SEED + 4 * 1_009
    assert hidden8["rung_seed"] == js3.SEED + 8 * 1_009
    assert hidden4["rung_seed"] != hidden8["rung_seed"]
