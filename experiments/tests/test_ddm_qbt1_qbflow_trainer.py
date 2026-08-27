from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qbt1_qbflow_trainer as qbt1


def _initial_model() -> qbt1.QBFLOWTorch:
    return qbt1.load_initial_model(torch.device("cpu"))


def test_torch_twin_matches_frozen_numpy_receiver() -> None:
    model = _initial_model().eval()
    pair_id = 31
    height, width = 6, 7
    params, boundary, interior = model.packet_state()
    expected = qbf1.reference_forward(
        params,
        boundary[pair_id],
        interior[pair_id],
        pair_id=pair_id,
        num_pairs=qbt1.N,
        height=height,
        width=width,
    )
    with torch.no_grad():
        actual = model(torch.tensor([pair_id]), height=height, width=width)
    rgb = (
        actual["rgb_pair_01"][0]
        .permute(2, 3, 0, 1)
        .reshape(height, width, 6)
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(
        actual["signed_interfaces"][0].numpy(), expected["signed_interfaces"], rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        actual["class_logits"][0].numpy(), expected["class_logits"], rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(rgb, expected["rgb_pair"], rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(actual["pose12"][0].numpy(), expected["pose12"], rtol=5e-4, atol=5e-4)


def test_chunk_ceiling_is_structural() -> None:
    chunks = qbt1.pair_chunks(tuple(range(32)), qbt1.MAX_CHUNK_PAIRS)
    assert tuple(map(len, chunks)) == (30, 2)
    with pytest.raises(qbt1.QBT1Error, match="hard ceiling"):
        qbt1.pair_chunks((0,), qbt1.MAX_CHUNK_PAIRS + 1)


def test_sealed_training_chunks_have_equal_no2_mass() -> None:
    chunks = qbt1.training_chunks(qbt1.SELECTION_IDS, qbt1.REAL_TRAIN_CHUNK_PAIRS)
    weights = dict(zip(qbt1.SELECTION_IDS, qbt1.SELECTION_WEIGHTS, strict=True))
    assert tuple(map(len, chunks)) == (16, 16)
    assert [sum(weights[pair_id] for pair_id in chunk) for chunk in chunks] == [300.0, 300.0]


def test_expected_flip_margin_rewards_positive_margin() -> None:
    target = torch.zeros((1, 1, 1), dtype=torch.long)
    losing = torch.tensor([[[[-1.0]], [[1.0]], [[0.0]], [[0.0]], [[0.0]]]])
    winning = torch.tensor([[[[3.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]])
    assert qbt1.expected_flip_margin_loss(winning, target, 0.1) < qbt1.expected_flip_margin_loss(
        losing, target, 0.1
    )


def test_role_prequantization_preserves_frozen_tensor_shapes() -> None:
    model = _initial_model()
    baseline = model.state_dict()
    candidate = qbt1.prequantize_role(baseline, "boundary_flow", 8)
    assert set(candidate) == set(baseline)
    assert {name: tuple(value.shape) for name, value in candidate.items()} == {
        name: tuple(value.shape) for name, value in baseline.items()
    }
    assert any(
        not torch.equal(candidate[name], baseline[name])
        for name in baseline
        if qbt1.state_tensor_role(name) == "boundary_flow"
    )
    params, boundary, interior = model.packet_state(candidate)
    qbf1.validate_param_shapes(params)
    assert boundary.shape == (qbt1.N, qbf1.BOUNDARY_LATENT_DIM)
    assert interior.shape == (qbt1.N, qbf1.INTERIOR_LATENT_DIM)


def test_checkpoint_restores_live_optimizer_rng_and_ema(tmp_path) -> None:
    qbt1.seed_everything(qbt1.SEED)
    model = _initial_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ema = qbt1.EMA(model, decay=0.9, warmup=True)
    config = {"schema": qbt1.SCHEMA, "action": "smoke", "resume_from": None}
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    receipt = qbt1.save_checkpoint(
        tmp_path / "stage.pt",
        model=model,
        optimizer=optimizer,
        ema=ema,
        config=config,
        step=3,
        stage="test",
        history=[{"step": 3}],
    )
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    step, restored_ema, history, payload = qbt1.load_checkpoint(
        tmp_path / "stage.pt", model=model, optimizer=optimizer, config=copy.deepcopy(config)
    )
    assert receipt["bytes"] > 0
    assert step == 3 and history == [{"step": 3}]
    assert payload["rng"] is not None
    assert restored_ema._num_updates == ema._num_updates
    for name, value in original.items():
        assert torch.equal(model.state_dict()[name], value)


def test_no2_gate_refuses_missing_control_and_accepts_real_same_budget_control() -> None:
    rows = [{"pair_id": pair_id, "d_seg": 0.0, "d_pose": 0.0} for pair_id in qbt1.SELECTION_IDS]
    refused = qbt1.no2_gate(pair_rows=rows, archive_bytes=100_000, b_hat=100_000, control=None)
    assert refused["control_status"] == "REFUSED_MISSING_REAL_SAME_BUDGET_QBW1_CONTROL"
    assert refused["admitted"] is False
    control = {
        "schema": qbt1.CONTROL_SCHEMA,
        "score_claim": False,
        "family": "QBW1_discrete_boundary_quotient",
        "custody_verified": True,
        "archive_bytes": 100_000,
        "pair_ids": list(qbt1.SELECTION_IDS),
        "all_payloads_retained": True,
        "S_hat": 1.0,
    }
    admitted = qbt1.no2_gate(pair_rows=rows, archive_bytes=100_000, b_hat=100_000, control=control)
    assert admitted["control_status"] == "PASS_REAL_SAME_BUDGET_CONTROL"
    assert admitted["admitted"] is True


def test_train_config_can_be_compiled_as_unclaimed_draft(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(qbt1, "verify_pins", lambda: {})
    config = qbt1.compile_config(
        action="train",
        output=tmp_path,
        pair_ids=qbt1.SELECTION_IDS,
        steps=2,
        device="mps",
    )
    assert config["launch_authorized"] is False
    assert config["chunk_pairs"] == qbt1.REAL_TRAIN_CHUNK_PAIRS
    assert [
        sum(qbt1.no2_sample_weights(chunk, torch.device("cpu")).tolist())
        for chunk in qbt1.training_chunks(config["pair_ids"], config["chunk_pairs"])
    ] == [300.0, 300.0]
    with pytest.raises(qbt1.QBT1Error, match="not authorized"):
        qbt1.validate_config(config)
    qbt1.validate_config(config, require_launch_authority=False)
