import pytest

torch = pytest.importorskip("torch")

from tac.cuda_levelset_training import CudaLevelSetConfig, TorchLevelSetWitness
from tac.cuda_v9_optimizers import build_torch_muon_adamw


def _flags():
    return {
        "--muon-lr": 0.002,
        "--muon-adamw-lr": 0.0001,
        "--muon-momentum": 0.95,
        "--muon-ns-steps": 5,
        "--muon-lr-final-frac": 0.1,
        "--muon-warm-start-momentum": True,
        "--stage-transition-rewarmup-epochs": 8,
        "--stage-transition-rewarmup-floor": 0.1,
        "--adam-beta2": 0.999,
        "--weight-decay": 0.01,
    }


def _model(seed=3):
    cfg = CudaLevelSetConfig(
        n_pairs=2, in_feat=7, hidden_dim=8, n_hidden=2, mod_dim=5
    )
    return TorchLevelSetWitness.build(cfg, seed=seed)


def test_muon_transition_warm_starts_hidden_matrices_and_rewarms_lrs():
    model = _model()
    outgoing = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    outgoing.step()
    outgoing.zero_grad(set_to_none=True)

    runtime, row = build_torch_muon_adamw(
        model, _flags(), total_epochs=20, start_epoch=5, outgoing_adamw=outgoing
    )
    assert row["n_muon_params"] > 0
    assert row["muon_warm_seeded_leaves"] == row["n_muon_params"]
    assert runtime.muon.param_groups[0]["lr"] == pytest.approx(0.0002)
    assert runtime.adamw.param_groups[0]["lr"] == pytest.approx(0.00001)

    mid = runtime.set_epoch(13)
    assert mid["rewarmup_multiplier"] == pytest.approx(1.0)
    end = runtime.set_epoch(20)
    assert end["muon_lr"] == pytest.approx(0.0002)
    assert end["adamw_lr"] == pytest.approx(0.0001)


def test_muon_split_step_and_checkpoint_roundtrip_are_real():
    model = _model(seed=7)
    runtime, _ = build_torch_muon_adamw(
        model, _flags(), total_epochs=12, start_epoch=4
    )
    runtime.set_epoch(7)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    sum(parameter.square().mean() for parameter in model.parameters()).backward()
    runtime.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))

    state = runtime.state_dict()
    clone = _model(seed=7)
    restored, _ = build_torch_muon_adamw(
        clone, _flags(), total_epochs=12, start_epoch=4
    )
    restored.load_state_dict(state)
    assert restored.current_epoch == 7
    assert restored.state_dict()["muon"]["state"]
    assert restored.state_dict()["adamw"]["state"]
    assert [group["lr"] for group in restored.param_groups] == pytest.approx(
        [group["lr"] for group in runtime.param_groups]
    )
