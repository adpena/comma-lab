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


def test_margin_constraint_default_off_is_bit_identical() -> None:
    torch.manual_seed(7)
    outputs = {"class_logits": torch.randn((2, 2, 2, 5))}
    camera = torch.zeros((2, 2, 3, 4, 4))
    pose = torch.randn((2, 6))
    scorer_logits = torch.randn((2, 5, 2, 2))
    target_argmax = torch.tensor([[[0, 1], [2, 3]], [[4, 3], [2, 1]]])
    target_pose = torch.zeros((2, 6))
    legacy_total, legacy_parts = qbt1.joint_objective(
        outputs,
        camera,
        pose,
        scorer_logits,
        target_argmax,
        target_pose,
        0.1,
    )
    off_total, off_parts = qbt1.joint_objective(
        outputs,
        camera,
        pose,
        scorer_logits,
        target_argmax,
        target_pose,
        0.1,
        margin_constraint_lambdas=None,
    )
    assert torch.equal(legacy_total, off_total)
    assert legacy_parts.keys() == off_parts.keys()
    assert all(torch.equal(legacy_parts[name], off_parts[name]) for name in legacy_parts)


def test_margin_constraint_dual_ascent_rises_and_decays_to_zero() -> None:
    bounds = {"Lane": 0.12, "Movable": 0.009}
    risen = qbt1.dual_ascent_margin_constraints(
        {"Lane": 0.0, "Movable": 0.0},
        {"Lane": 0.62, "Movable": 0.109},
        bounds,
        eta_lambda=1.0,
    )
    assert risen == pytest.approx({"Lane": 0.5, "Movable": 0.1})
    decayed = qbt1.dual_ascent_margin_constraints(
        risen,
        {"Lane": 0.0, "Movable": 0.0},
        bounds,
        eta_lambda=20.0,
    )
    assert decayed == {"Lane": 0.0, "Movable": 0.0}


def test_margin_constraint_penalty_is_live_on_realized_class_pixels() -> None:
    target = torch.tensor([[[1, 1], [3, 3]]])
    logits = torch.zeros((1, 5, 2, 2), requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 0.2
    outputs = {"class_logits": logits.permute(0, 2, 3, 1)}
    total, parts = qbt1.joint_objective(
        outputs,
        torch.zeros((1, 2, 3, 4, 4)),
        torch.zeros((1, 6)),
        logits,
        target,
        torch.zeros((1, 6)),
        1.0,
        margin_constraint_lambdas={"Lane": 0.5, "Movable": 0.25},
    )
    total.backward()
    assert parts["margin_constraint_penalty_score"] > 0
    assert parts["margin_constraint_penalty_score_Lane"] > 0
    assert parts["margin_constraint_penalty_score_Movable"] > 0
    assert logits.grad is not None and bool(torch.any(logits.grad != 0))


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


def test_birth_event_mode_pins_mode_threshold_pair(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(qbt1, "verify_pins", lambda: {})
    initialization = tmp_path / "initialized.pt"
    initialization.write_bytes(b"custody-only validation payload")
    kwargs = {
        "action": "smoke",
        "pair_ids": (qbt1.SELECTION_IDS[0],),
        "device": "cpu",
        "initialization_state": initialization,
        "birth_max_steps": 3,
        "margin_steps": 7,
    }
    default = qbt1.compile_qbt2b_config(output=tmp_path / "default", **kwargs)
    assert default["birth_event_mode"] == "accuracy_020"
    assert default["birth_within_class_error_max"] == qbt1.BIRTH_WITHIN_CLASS_ERROR_MAX
    qbt1.validate_config(default)
    legacy = dict(default)
    legacy.pop("birth_event_mode")
    qbt1.validate_config(legacy)
    existence = qbt1.compile_qbt2b_config(
        output=tmp_path / "existence", birth_event_mode="existence_majority", **kwargs
    )
    assert existence["birth_within_class_error_max"] == qbt1.BIRTH_EXISTENCE_ERROR_MAX
    qbt1.validate_config(existence)
    with pytest.raises(qbt1.QBT1Error, match="birth event mode"):
        qbt1.compile_qbt2b_config(
            output=tmp_path / "bogus", birth_event_mode="werr_040", **kwargs
        )
    for mode, wrong_threshold in (
        ("accuracy_020", qbt1.BIRTH_EXISTENCE_ERROR_MAX),
        ("existence_majority", qbt1.BIRTH_WITHIN_CLASS_ERROR_MAX),
    ):
        inconsistent = dict(default)
        inconsistent["birth_event_mode"] = mode
        inconsistent["birth_within_class_error_max"] = wrong_threshold
        with pytest.raises(qbt1.QBT1Error, match="event/retention law"):
            qbt1.validate_config(inconsistent)


def test_margin_constraint_mode_pins_mode_bounds_eta_group(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(qbt1, "verify_pins", lambda: {})
    initialization = tmp_path / "initialized.pt"
    initialization.write_bytes(b"custody-only validation payload")
    kwargs = {
        "action": "smoke",
        "pair_ids": (qbt1.SELECTION_IDS[0],),
        "device": "cpu",
        "initialization_state": initialization,
        "birth_max_steps": 3,
        "margin_steps": 7,
    }
    default = qbt1.compile_qbt2b_config(output=tmp_path / "default", **kwargs)
    assert default["margin_constraint_mode"] == qbt1.MARGIN_CONSTRAINT_UNCONSTRAINED
    assert default["margin_constraint_bounds"] == {}
    assert default["margin_constraint_eta_lambda"] == 0.0
    qbt1.validate_config(default)
    legacy = dict(default)
    for name in (
        "margin_constraint_mode",
        "margin_constraint_bounds",
        "margin_constraint_eta_lambda",
    ):
        legacy.pop(name)
    qbt1.validate_config(legacy)
    constrained = qbt1.compile_qbt2b_config(
        output=tmp_path / "constrained",
        margin_constraint_mode=qbt1.MARGIN_CONSTRAINT_LANE_MOVABLE,
        **kwargs,
    )
    assert constrained["margin_constraint_bounds"] == {
        "Lane": qbt1.MARGIN_CONSTRAINT_LANE_BOUND,
        "Movable": qbt1.MARGIN_CONSTRAINT_MOVABLE_BOUND,
    }
    assert constrained["margin_constraint_eta_lambda"] == qbt1.MARGIN_CONSTRAINT_ETA_LAMBDA
    qbt1.validate_config(constrained)
    mode_only = copy.deepcopy(default)
    mode_only["margin_constraint_mode"] = qbt1.MARGIN_CONSTRAINT_LANE_MOVABLE
    with pytest.raises(qbt1.QBT1Error, match="mode/bounds/eta group"):
        qbt1.validate_config(mode_only)
    values_only = copy.deepcopy(default)
    values_only["margin_constraint_bounds"] = copy.deepcopy(
        qbt1.MARGIN_CONSTRAINT_MODE_PINS[qbt1.MARGIN_CONSTRAINT_LANE_MOVABLE]["bounds"]
    )
    values_only["margin_constraint_eta_lambda"] = qbt1.MARGIN_CONSTRAINT_ETA_LAMBDA
    with pytest.raises(qbt1.QBT1Error, match="mode/bounds/eta group"):
        qbt1.validate_config(values_only)
    wrong_eta = copy.deepcopy(constrained)
    wrong_eta["margin_constraint_eta_lambda"] *= 2.0
    with pytest.raises(qbt1.QBT1Error, match="mode/bounds/eta group"):
        qbt1.validate_config(wrong_eta)
    assert qbt1.config_identity(default) != qbt1.config_identity(constrained)


def test_existence_gate_majority_threshold_and_accuracy_watch() -> None:
    rows = [
        {
            "class_id": class_id,
            "class_name": name,
            "predicted_pixels": 1,
            "predicted_pixel_share": 0.2,
            "within_class_error": 0.30,
        }
        for class_id, name in enumerate(qbt1.PALETTE_CLASSES)
    ]
    rows[0]["within_class_error"] = 0.10
    gate = qbt1.birth_gate_from_table(
        rows, within_class_error_max=qbt1.BIRTH_EXISTENCE_ERROR_MAX
    )
    assert gate["all_five_classes_pass"] is True
    assert gate["accuracy_watch"]["classes_passing"] == 1
    assert "existence" in gate["derived_from"]
    rows[2]["within_class_error"] = 0.50
    refused = qbt1.birth_gate_from_table(
        rows, within_class_error_max=qbt1.BIRTH_EXISTENCE_ERROR_MAX
    )
    assert refused["all_five_classes_pass"] is False
    legacy_gate = qbt1.birth_gate_from_table(rows)
    assert "accuracy_watch" not in legacy_gate
    assert "DEFAULT_TAU_PERSIST" in legacy_gate["derived_from"]


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


def test_checkpoint_refuses_cross_margin_constraint_mode_resume(tmp_path) -> None:
    qbt1.seed_everything(qbt1.SEED)
    model = _initial_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ema = qbt1.EMA(model, decay=0.9, warmup=True)
    off_config = {
        "schema": qbt1.SCHEMA,
        "action": "smoke",
        "resume_from": None,
        "margin_constraint_mode": qbt1.MARGIN_CONSTRAINT_UNCONSTRAINED,
        "margin_constraint_bounds": {},
        "margin_constraint_eta_lambda": 0.0,
    }
    qbt1.save_checkpoint(
        tmp_path / "off.pt",
        model=model,
        optimizer=optimizer,
        ema=ema,
        config=off_config,
        step=1,
        stage="test",
        history=[{"step": 1}],
    )
    constrained_config = copy.deepcopy(off_config)
    constrained_config.update(
        {
            "margin_constraint_mode": qbt1.MARGIN_CONSTRAINT_LANE_MOVABLE,
            "margin_constraint_bounds": copy.deepcopy(
                qbt1.MARGIN_CONSTRAINT_MODE_PINS[qbt1.MARGIN_CONSTRAINT_LANE_MOVABLE]["bounds"]
            ),
            "margin_constraint_eta_lambda": qbt1.MARGIN_CONSTRAINT_ETA_LAMBDA,
        }
    )
    assert "margin_constraint_mode" not in {
        "action",
        "resume_from",
        "launch_authorized",
        "scorer_lane",
        "metal_lane",
    }
    with pytest.raises(qbt1.QBT1Error, match="config identity differs"):
        qbt1.load_checkpoint(
            tmp_path / "off.pt",
            model=model,
            optimizer=optimizer,
            config=constrained_config,
        )


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


def test_storage_projection_birth_verdict_selection_survives_margin_tail() -> None:
    history = [
        {"step": 5, "birth_verdict": {"pair_ids": [62]}},
        {"step": 10, "birth_verdict": {"pair_ids": [62]}},
        {"step": 11, "margin_constraint": {"binding": {"Lane": True}}},
    ]
    assert qbt1.latest_birth_verdict_pair_ids(history) == (62,)
    with pytest.raises(qbt1.QBT1Error, match="lacks a retained birth verdict"):
        qbt1.latest_birth_verdict_pair_ids(history[-1:])


def test_r8_config_geometry_and_identity_excludes_dispatch_fields(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(qbt1, "verify_pins", lambda: {})
    initialization = tmp_path / "initialized_r8.pt"
    initialization.write_bytes(b"custody-only validation payload")
    config = qbt1.compile_qbt2b_config(
        action="train",
        output=tmp_path / "governed_n32_r8",
        pair_ids=qbt1.SELECTION_IDS,
        device="mps",
        initialization_state=initialization,
        birth_max_steps=20,
        margin_steps=qbt1.R8_MARGIN_STEPS,
        birth_class_weight_mode="balanced",
        birth_event_mode="existence_majority",
        margin_constraint_mode=qbt1.MARGIN_CONSTRAINT_LANE_MOVABLE,
    )
    assert qbt1.R8_MARGIN_STEPS == 15_000
    assert config["steps"] == 15_020
    assert config["margin_constraint_mode"] == qbt1.MARGIN_CONSTRAINT_LANE_MOVABLE
    qbt1.validate_config(config, require_launch_authority=False)
    identity = qbt1.canonical_sha256(qbt1.config_identity(config))
    claimed = copy.deepcopy(config)
    claimed["launch_authorized"] = True
    claimed["scorer_lane"] = {"claimed": True, "claim_id": "test"}
    claimed["metal_lane"] = {"claimed": True, "claim_id": "test"}
    assert qbt1.canonical_sha256(qbt1.config_identity(claimed)) == identity


def test_checkpoint_cadence_law_scales_with_steps_and_refuses_past_ceiling(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(qbt1, "verify_pins", lambda: {})
    initialization = tmp_path / "initialized_r8.pt"
    initialization.write_bytes(b"custody-only validation payload")
    kwargs = {
        "action": "train",
        "output": tmp_path / "governed_n32_r8",
        "pair_ids": qbt1.SELECTION_IDS,
        "device": "mps",
        "initialization_state": initialization,
        "birth_max_steps": 20,
        "margin_steps": qbt1.R8_MARGIN_STEPS,
        "birth_class_weight_mode": "balanced",
        "birth_event_mode": "existence_majority",
        "margin_constraint_mode": qbt1.MARGIN_CONSTRAINT_LANE_MOVABLE,
    }
    derived = (20 + qbt1.R8_MARGIN_STEPS) // qbt1.CHECKPOINT_CRASH_LOSS_DENOMINATOR
    assert derived == 50
    config = qbt1.compile_qbt2b_config(**kwargs, checkpoint_every_steps=derived)
    assert config["checkpoint_every_steps"] == 50
    qbt1.validate_config(config, require_launch_authority=False)
    with pytest.raises(qbt1.QBT1Error, match="checkpoint cadence"):
        qbt1.compile_qbt2b_config(**kwargs, checkpoint_every_steps=derived + 1)
    legacy = qbt1.compile_qbt2b_config(**kwargs)
    assert legacy["checkpoint_every_steps"] == 5
    qbt1.validate_config(legacy, require_launch_authority=False)
    short = qbt1.compile_qbt2b_config(
        action="smoke",
        output=tmp_path / "short",
        pair_ids=(qbt1.SELECTION_IDS[0],),
        device="cpu",
        initialization_state=initialization,
        birth_max_steps=3,
        margin_steps=7,
    )
    with pytest.raises(qbt1.QBT1Error, match="checkpoint cadence"):
        mutated = copy.deepcopy(short)
        mutated["checkpoint_every_steps"] = 6
        qbt1.validate_config(mutated, require_launch_authority=False)


def test_build_r8_init_refuses_non_stage3_end_source(tmp_path) -> None:
    source = tmp_path / "wrong.pt"
    torch.save({"stage": "stage_03a_ce_class_birth"}, source)
    with pytest.raises(qbt1.QBT1Error, match="stage-03 end"):
        qbt1.build_r8_initialized_state(source, tmp_path / "out.pt")


def test_build_r8_init_emits_loader_schema_and_strict_round_trips(tmp_path) -> None:
    reference = _initial_model()
    shadow = {
        name: value.detach().clone() for name, value in reference.state_dict().items()
    }
    source = tmp_path / "stage_03_end.pt"
    torch.save(
        {
            "stage": "stage_03_joint_boundary_interior_birth_end",
            "step": 5020,
            "ema": {"shadow": shadow},
        },
        source,
    )
    out = tmp_path / "initialized_r8.pt"
    fact = qbt1.build_r8_initialized_state(source, out)
    assert "sha256" in fact
    state = torch.load(out, map_location="cpu", weights_only=False)
    assert state["schema"] == "ddm_qbt2b_initialized_qbf1_state.v1"
    assert state["provenance"]["basis"] == "ema_shadow"
    assert state["provenance"]["source_step"] == 5020
    fresh = _initial_model()
    fresh.load_state_dict(state["state_dict"], strict=True)
