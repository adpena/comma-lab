from __future__ import annotations

import copy
import tarfile

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


def test_realized_ce_birth_keeps_pose_active() -> None:
    camera = torch.zeros((2, 2, 3, 4, 4), requires_grad=True)
    logits = torch.randn((2, 5, 2, 2), requires_grad=True)
    pose = torch.randn((2, 6), requires_grad=True)
    target_argmax = torch.tensor([[[0, 1], [2, 3]], [[4, 3], [2, 1]]])
    target_pose = torch.zeros((2, 6))
    total, parts = qbt1.realized_ce_birth_objective(
        camera, pose, logits, target_argmax, target_pose
    )
    total.backward()
    assert parts["seg_ce_realized"] > 0
    assert parts["pose_mse_realized"] > 0
    assert logits.grad is not None and pose.grad is not None


def test_balanced_class_weights_keep_weighted_mean_normalized() -> None:
    torch.manual_seed(11)
    camera = torch.zeros((2, 2, 3, 4, 4))
    logits = torch.randn((2, 5, 2, 2))
    pose = torch.zeros((2, 6))
    target_argmax = torch.tensor([[[0, 1], [2, 3]], [[4, 3], [2, 1]]])
    target_pose = torch.zeros((2, 6))
    unweighted, _parts = qbt1.realized_ce_birth_objective(
        camera, pose, logits, target_argmax, target_pose
    )
    uniform = torch.ones(qbf1.N_CLASSES)
    uniform_total, _uniform_parts = qbt1.realized_ce_birth_objective(
        camera, pose, logits, target_argmax, target_pose, class_weights=uniform
    )
    assert torch.allclose(unweighted, uniform_total, atol=1.0e-6)
    balanced = torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0])
    _total, parts = qbt1.realized_ce_birth_objective(
        camera, pose, logits, target_argmax, target_pose, class_weights=balanced
    )
    per_pixel = torch.nn.functional.cross_entropy(
        logits, target_argmax.long(), weight=balanced, reduction="none"
    )
    pixel_weight = balanced[target_argmax.long()]
    expected = (per_pixel.sum(dim=(1, 2)) / pixel_weight.sum(dim=(1, 2))).mean()
    assert torch.allclose(parts["seg_ce_realized"], expected, atol=1.0e-6)


def test_derive_balanced_class_weights_from_real_targets(monkeypatch) -> None:
    full = torch.tensor([[[0, 1], [2, 3]], [[4, 0], [0, 0]]])
    monkeypatch.setattr(qbt1, "_target_arrays", lambda ids, device: (full, torch.zeros((2, 6))))
    weights = qbt1.derive_balanced_class_weights((0, 1), torch.device("cpu"))
    counts = torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(weights, counts.sum() / (5.0 * counts))
    missing_lane = torch.where(full == 1, torch.zeros_like(full), full)
    monkeypatch.setattr(
        qbt1, "_target_arrays", lambda ids, device: (missing_lane, torch.zeros((2, 6)))
    )
    with pytest.raises(qbt1.QBT1Error, match="every class present"):
        qbt1.derive_balanced_class_weights((0, 1), torch.device("cpu"))


def test_birth_class_weight_mode_is_config_gated(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(qbt1, "verify_pins", lambda: {})
    initialization = tmp_path / "initialized.pt"
    initialization.write_bytes(b"custody-only validation payload")
    balanced = qbt1.compile_qbt2b_config(
        action="smoke",
        output=tmp_path / "balanced",
        pair_ids=(qbt1.SELECTION_IDS[0],),
        device="cpu",
        initialization_state=initialization,
        birth_max_steps=3,
        margin_steps=7,
        birth_class_weight_mode="balanced",
    )
    assert balanced["birth_class_weight_mode"] == "balanced"
    qbt1.validate_config(balanced)
    default = qbt1.compile_qbt2b_config(
        action="smoke",
        output=tmp_path / "default",
        pair_ids=(qbt1.SELECTION_IDS[0],),
        device="cpu",
        initialization_state=initialization,
        birth_max_steps=3,
        margin_steps=7,
    )
    assert default["birth_class_weight_mode"] == "none"
    legacy = dict(default)
    legacy.pop("birth_class_weight_mode")
    qbt1.validate_config(legacy)
    bogus = dict(default)
    bogus["birth_class_weight_mode"] = "area_sqrt"
    with pytest.raises(qbt1.QBT1Error, match="class-weight mode"):
        qbt1.validate_config(bogus)


def test_birth_gate_uses_derived_threshold_and_all_five_classes() -> None:
    rows = [
        {
            "class_id": class_id,
            "class_name": name,
            "predicted_pixels": 1,
            "predicted_pixel_share": 0.2,
            "within_class_error": 0.19,
        }
        for class_id, name in enumerate(qbt1.PALETTE_CLASSES)
    ]
    assert qbt1.birth_gate_from_table(rows)["all_five_classes_pass"] is True
    rows[1]["within_class_error"] = 0.20
    assert qbt1.birth_gate_from_table(rows)["all_five_classes_pass"] is False


def test_data_dependent_readout_fit_changes_only_last_frame_values() -> None:
    model = _initial_model()
    baseline_w = model.params["render_out_w"].detach().clone()
    baseline_b = model.params["render_out_b"].detach().clone()
    rng = np.random.default_rng(17)
    classes = np.arange(260, dtype=np.uint8) % qbf1.N_CLASSES
    states = rng.normal(0.0, 1.0e-4, size=(260, qbf1.COARSE_DIM)).astype(np.float32)
    states[np.arange(260), classes] = 1.0
    samples = {
        "render_state_f32": states,
        "native_class_u8": classes,
        "pair_id_u16": np.zeros(260, dtype=np.uint16),
    }
    palette = np.asarray(
        [[20, 30, 40], [60, 70, 80], [100, 110, 120], [140, 150, 160], [180, 190, 200]],
        dtype=np.float32,
    )
    receipt, retained = qbt1.fit_inherited_palette_readout(
        model, palette, (0,), samples=samples
    )
    assert receipt["degenerate"] is False
    assert set(receipt["changed_tensors"]) == {"params.render_out_w", "params.render_out_b"}
    assert torch.equal(model.params["render_out_w"][:, :3], baseline_w[:, :3])
    assert torch.equal(model.params["render_out_b"][:3], baseline_b[:3])
    assert np.isfinite(retained["fit_coefficients_f64"]).all()


def test_reencode_consolidation_keeps_verified_single_tar(tmp_path) -> None:
    root = tmp_path / "reencode"
    first = qbt1.atomic_bytes(root / "sections/a.bin", b"alpha")
    second = qbt1.atomic_bytes(root / "packet.qbf", b"beta")
    manifest = {"schema": "test", "archive": first, "packet": second}
    qbt1.atomic_json(root / "REENCODE_MANIFEST.json", manifest)
    consolidated = qbt1.consolidate_reencode_payloads(root, manifest)
    assert consolidated["retention_mode"] == "ONE_DETERMINISTIC_TAR_PER_REENCODE"
    assert sorted(path.name for path in root.iterdir()) == [
        "REENCODE_MANIFEST.json",
        "reencode_payloads.tar",
    ]
    with tarfile.open(root / "reencode_payloads.tar", "r") as archive:
        assert {member.name for member in archive.getmembers()} == {"packet.qbf", "sections/a.bin"}


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
        curriculum_state={"phase": "stage_03a_ce_class_birth", "birth_step": 3},
    )
    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)
    step, restored_ema, history, payload = qbt1.load_checkpoint(
        tmp_path / "stage.pt", model=model, optimizer=optimizer, config=copy.deepcopy(config)
    )
    assert receipt["bytes"] > 0
    assert step == 3 and history == [{"step": 3}]
    assert payload["rng"] is not None
    assert payload["curriculum_state"] == {
        "phase": "stage_03a_ce_class_birth",
        "birth_step": 3,
    }
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


def test_qbt2b_schema_is_additive_and_ema_uses_total_schedule(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(qbt1, "verify_pins", lambda: {})
    initialization = tmp_path / "initialized.pt"
    initialization.write_bytes(b"custody-only validation payload")
    legacy = qbt1.compile_config(
        action="smoke", output=tmp_path / "legacy", pair_ids=(qbt1.SELECTION_IDS[0],), steps=1, device="cpu"
    )
    qbt1.validate_config(legacy)
    assert "curriculum_mode" not in legacy
    qbt2b = qbt1.compile_qbt2b_config(
        action="smoke",
        output=tmp_path / "qbt2b",
        pair_ids=(qbt1.SELECTION_IDS[0],),
        device="cpu",
        initialization_state=initialization,
        birth_max_steps=3,
        margin_steps=7,
    )
    assert qbt2b["steps"] == 10
    assert qbt2b["ema"] == qbt1.resolve_ema_law(10)
    qbt1.validate_config(qbt2b)
