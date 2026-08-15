from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments import ddm_wd2_width_distillation_build as wd2_build
from experiments import ddm_wd3_scorer_aware_width_distillation as wd3
from experiments import ddm_wd3_student_receiver as receiver
from tac.witness_control.resume_registry import ResumeRegistry


def tiny_model() -> receiver.StudentSemanticRenderer:
    torch.manual_seed(7)
    return receiver.StudentSemanticRenderer(receiver.StudentSpec("test_flat_w8_d1", "flattened", 8, 1))


def fake_pin_facts() -> dict[str, dict[str, object]]:
    return {
        name: {"path": str(path), "bytes": size, "sha256": digest} for name, (path, size, digest) in wd3.PINS.items()
    }


def valid_config() -> dict[str, object]:
    config = wd3.blocked_config_template()
    config["teacher_cache_result"] = "/tmp/cache.json"
    config["resume_from"] = str(wd3.WARM_CHECKPOINT)
    config["scorer_lane"] = {
        "claimed": True,
        "claim_id": "scorer-wd3",
        "agent": "MAIN",
        "platform": "macos-cpu",
    }
    config["metal_lane"] = {
        "claimed": True,
        "claim_id": "metal-wd3",
        "agent": "MAIN",
        "platform": "macos-mps",
    }
    config["launch_authorized"] = True
    config["r5_exit_verified"] = True
    return config


def test_adaptive_packet_matches_fake_quant_grid_and_is_byte_idempotent() -> None:
    model = tiny_model()
    bits = receiver.uniform_allocation(model, 4, selection_sha256="a" * 64)
    rows = {name: list(values) for name, values in bits.bits.items()}
    rows[sorted(rows)[0]][0] = 3
    allocation = receiver.AdaptiveQuantizationAllocation(
        bits={name: tuple(values) for name, values in rows.items()},
        selection_sha256="a" * 64,
        policy="test_adaptive",
    )
    packet = receiver.pack_student(model, allocation)
    assert len(packet) == receiver.serialized_bytes_for_allocation(model, allocation)
    parsed = receiver.unpack_student(packet)
    assert receiver.pack_student(parsed, allocation) == packet
    fake = receiver.fake_quantize_state(model, allocation)
    for name, value in parsed.state_dict().items():
        torch.testing.assert_close(value, fake[name], rtol=0, atol=0)
    assert receiver.packet_allocation(packet) == allocation
    assert receiver.allocation_telemetry(model, allocation)["packet_bytes"] == len(packet)


def test_payload_code_mutation_changes_learned_state_or_is_rejected() -> None:
    model = tiny_model()
    allocation = receiver.uniform_allocation(model, 4)
    packet = bytearray(receiver.pack_student(model, allocation))
    metadata_bytes = receiver.HEADER.unpack_from(packet)[6]
    first_code_byte = receiver.HEADER.size + metadata_bytes + 2
    packet[first_code_byte] ^= 1
    try:
        changed = receiver.unpack_student(bytes(packet))
    except receiver.WD3ReceiverError:
        return
    assert any(
        not torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), changed.state_dict().values(), strict=True)
    )


def test_paired_receiver_orders_fixed_frame0_and_quantized_student_frame1() -> None:
    model = tiny_model()
    allocation = receiver.uniform_allocation(model, 4)
    tokens = torch.zeros((1, receiver.EVAL_H, receiver.EVAL_W), dtype=torch.long)
    fixed = torch.full((1, 3, receiver.CAMERA_H, receiver.CAMERA_W), 17.0, dtype=torch.float32)
    pair, frame1 = wd3.paired_receiver_tensor(
        model=model,
        allocation=allocation,
        tokens=tokens,
        pair_indices=torch.tensor([0]),
        fixed_frame0=fixed,
    )
    assert pair.shape == (1, 2, 3, receiver.CAMERA_H, receiver.CAMERA_W)
    assert torch.equal(pair[:, 0], fixed)
    assert torch.equal(pair[:, 1], frame1)
    assert torch.equal(frame1, frame1.round())
    frame1.mean().backward()
    assert model.head.weight.grad is not None
    assert torch.count_nonzero(model.head.weight.grad)


class DummyPose(nn.Module):
    def preprocess_input(self, pair: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(pair.flatten(1, 2), (2, 2))

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = value.mean(dim=(1, 2, 3), keepdim=False)
        return {"pose": mean[:, None].repeat(1, 12)}


class DummySeg(nn.Module):
    def preprocess_input(self, pair: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(pair[:, 1], (3, 4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = value.mean(dim=1, keepdim=True)
        return torch.cat([base + offset for offset in range(5)], dim=1)


def test_both_scorers_are_active_and_pose_gradient_reaches_frame1() -> None:
    frame0 = torch.zeros((2, 3, 8, 8))
    frame1 = torch.ones((2, 3, 8, 8), requires_grad=True)
    pair = torch.stack((frame0, frame1), dim=1)
    pose6, logits = wd3.scorer_forward(pair, DummyPose(), DummySeg())
    assert pose6.shape == (2, 6)
    assert logits.shape == (2, 5, 3, 4)
    (pose6.sum() + logits.sum()).backward()
    assert frame1.grad is not None
    assert torch.count_nonzero(frame1.grad)


def test_score_native_terms_and_one_sided_duals_match_hand_values() -> None:
    student_logits = torch.tensor(
        [[[[2.0, 0.0]], [[0.0, 2.0]], [[-1.0, -1.0]], [[-2.0, -2.0]], [[-3.0, -3.0]]]],
        requires_grad=True,
    )
    teacher_logits = student_logits.detach() + 0.1
    target = torch.tensor([[[0, 1]]])
    teacher_margin = torch.full((1, 1, 2), 0.5)
    student_pose = torch.tensor([[0.1] * 6], requires_grad=True)
    zero_pose = torch.zeros((1, 6))
    student_frame = torch.full((1, 3, 2, 2), 3.0, requires_grad=True)
    teacher_frame = torch.ones_like(student_frame)
    thresholds = wd3.StageThresholds(0.25, 0.0, 0.0, 1.0)
    duals = wd3.DualState(2.0, 3.0, 4.0, 5.0)
    total, components = wd3.score_native_objective(
        student_logits=student_logits,
        student_pose6=student_pose,
        student_frame1=student_frame,
        teacher_logits=teacher_logits,
        teacher_argmax=target,
        teacher_margin=teacher_margin,
        teacher_pose6=zero_pose,
        original_argmax=target,
        original_pose6=zero_pose,
        teacher_frame1=teacher_frame,
        selected_cells=torch.ones_like(target, dtype=torch.bool),
        thresholds=thresholds,
        duals=duals,
    )
    probs = student_logits.softmax(dim=1)
    hand_soft = (1 - probs.gather(1, target[:, None]).squeeze(1)).mean()
    torch.testing.assert_close(components["seg_axis_train_loss_proxy"], hand_soft)
    torch.testing.assert_close(components["seg_axis_stage_calibrated_score_proxy"], 25.0 * hand_soft)
    torch.testing.assert_close(components["pose_exact_nonlinear_score_train_quantity"], torch.sqrt(torch.tensor(0.1)))
    assert components["decode_mse_uint8_train_quantity"].item() == 4.0
    total.backward()
    assert student_logits.grad is not None and student_pose.grad is not None
    assert student_frame.grad is not None
    updated = wd3.DualState(1.0, 2.0, 3.0, 4.0).update(
        margin_violation=-4.0,
        teacher_kl_violation=2.0,
        decode_violation=0.5,
        teacher_pose_violation=0.25,
        step_size=0.25,
    )
    assert updated == wd3.DualState(1.0, 2.5, 3.125, 4.0625)


def test_selective_cells_are_mismatch_and_boundary_derived_with_road_lane_edges() -> None:
    original = torch.zeros((1, 3, 4), dtype=torch.long)
    original[:, :, 2:] = 1
    teacher = original.clone()
    student = original.clone()
    student[0, 1, 1] = 1
    selected = wd3.derive_selective_cell_mask(student, teacher, original)
    assert selected[0, 1, 1]
    assert selected[0, :, 1:3].all()
    assert not selected[0, 0, 0]
    telemetry = wd3.cell_edge_telemetry(student, original)
    assert telemetry["road_lane_flips"] == 1
    assert telemetry["per_edge_flips"] == {"Lane<->Road": 1}
    assert telemetry["per_target_cell_flips"] == {"Road": 1}


def test_validation_subsets_are_deterministic_strided_stratified_and_nonprefix() -> None:
    n60 = wd3.evenly_strided_indices()
    assert n60.tolist() == list(range(0, 600, 10))
    labels = np.arange(600) % 5
    first = wd3.stratified_random_indices(labels)
    second = wd3.stratified_random_indices(labels)
    assert np.array_equal(first, second)
    assert first.size == np.unique(first).size == 120
    assert not np.array_equal(first, np.arange(120))
    assert Counter(labels[first]) == Counter(dict.fromkeys(range(5), 24))


def test_gradient_waterfill_is_adaptive_and_cheapest_re_score_is_measured() -> None:
    model = tiny_model()
    sensitivity = {}
    for name, value in model.state_dict().items():
        if value.ndim < 2:
            continue
        axis = value.ndim - 1 if name.endswith("embed.weight") else 0
        rows = []
        for group in range(value.shape[axis]):
            errors = {str(bit): 1.0 / (bit + group + 1) for bit in range(2, 9)}
            bytes_by_bit = {str(bit): bit + 2 for bit in range(2, 9)}
            rows.append({"group": group, "errors": errors, "bytes": bytes_by_bit})
        sensitivity[name] = rows
    allocation = wd3.adaptive_allocation_from_sensitivity(
        model,
        sensitivity,
        maximum_predicted_error=sum(row["errors"]["4"] for rows in sensitivity.values() for row in rows),
        selection_sha256="b" * 64,
    )
    depths = {bit for values in allocation.bits.values() for bit in values}
    assert min(depths) >= 2 and max(depths) <= 8
    rows = [
        {
            "allocation_id": "projected",
            "packet_bytes": 1,
            "hard_cell_gate_pass": True,
            "road_lane_gate_pass": True,
            "pose_gate_pass": True,
            "parse_back_exact": True,
            "retained_payload": True,
            "measured": False,
        }
    ]
    with pytest.raises(wd3.WD3Error, match="projected"):
        wd3.choose_cheapest_passing_quantization(rows)
    for name, size in (("large", 20), ("small", 10)):
        rows.append(
            {
                "allocation_id": name,
                "packet_bytes": size,
                "hard_cell_gate_pass": True,
                "road_lane_gate_pass": True,
                "pose_gate_pass": True,
                "parse_back_exact": True,
                "retained_payload": True,
                "measured": True,
            }
        )
    assert wd3.choose_cheapest_passing_quantization(rows[1:])["allocation_id"] == "small"


def test_resume_registry_and_checkpoint_restore_all_controller_state(tmp_path: Path) -> None:
    model = tiny_model()
    allocation = receiver.uniform_allocation(model, 4, selection_sha256="c" * 64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    ema = wd2_build.DeploymentEMA(model, 0.9)
    generator = torch.Generator().manual_seed(17)
    selection = tmp_path / "selection.u8"
    selection.write_bytes(b"select")
    selection_record = wd3.file_record(selection)
    controller = wd3.WD3ResumeController(
        duals=wd3.DualState(1, 2, 3, 4),
        thresholds=wd3.StageThresholds(0.5, 0.2, 0.3),
        epoch=7,
        batch_cursor=11,
        selection_sha256=selection_record["sha256"],
        allocation_sha256=wd3.canonical_sha256(allocation.as_dict()),
    )
    registry = ResumeRegistry()
    wd3.register_resume_controller(registry, controller)
    state = registry.state_arrays()
    assert any(key.startswith("__wd3_") for key in state)
    config = {"schema": "unit"}
    checkpoint = tmp_path / "checkpoint.pt"
    wd3.save_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        controller=controller,
        allocation=allocation,
        selection_record=selection_record,
        subset_ids={"controller_n60": [0], "negative_n120": [1]},
        config=config,
        history=[{"epoch": 7}],
        stage="unit",
    )
    restored_model = tiny_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=2)
    restored_ema = wd2_build.DeploymentEMA(restored_model, 0.9)
    payload, restored, restored_allocation = wd3.load_checkpoint(
        checkpoint,
        model=restored_model,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        generator=torch.Generator(),
        expected_config=config,
    )
    assert restored.epoch == 7 and restored.batch_cursor == 11
    assert restored.duals == wd3.DualState(1, 2, 3, 4)
    assert restored.selection_sha256 == selection_record["sha256"]
    assert restored_allocation == allocation
    assert payload["scaler"] == {"enabled": False, "state_dict": {}}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg.update(base_receipt="/tmp/wrong.json"), "base receipt"),
        (lambda cfg: cfg.update(teacher_cache_result=None), "cache receipt"),
        (lambda cfg: cfg["subsets"].update(prefix=True), "prefix"),
        (lambda cfg: cfg.update(retain_all_payloads=False), "non-retaining"),
        (lambda cfg: cfg.update(resume_from=None), "resume"),
        (lambda cfg: cfg.update(chunk_pairs=121), "chunk"),
        (lambda cfg: cfg["scorer_lane"].update(claimed=False), "scorer lane"),
        (lambda cfg: cfg.update(launch_authorized=False), "authorization"),
        (lambda cfg: cfg.update(invented_flag=True), "unknown"),
    ],
)
def test_typed_compiler_rejects_every_g5_shape_before_heavy_load(mutation, message: str) -> None:
    config = valid_config()
    mutation(config)
    calls = []

    def exists(path: Path) -> bool:
        calls.append(path)
        return True

    with pytest.raises(wd3.WD3Error, match=message):
        wd3.validate_compiled_config(config, facts=fake_pin_facts(), path_exists=exists)
    assert len(calls) <= 2


def test_valid_compiler_and_arm_order_gates() -> None:
    config = valid_config()
    result = wd3.validate_compiled_config(config, facts=fake_pin_facts(), path_exists=lambda _path: True)
    assert result["status"] == "PASS"
    bad = copy.deepcopy(config)
    bad["arm"] = "D56"
    with pytest.raises(wd3.WD3Error, match="order"):
        wd3.validate_compiled_config(bad, facts=fake_pin_facts(), path_exists=lambda _p: True)
    fresh = copy.deepcopy(config)
    fresh["arm"] = "fresh"
    with pytest.raises(wd3.WD3Error, match="both W0"):
        wd3.validate_compiled_config(fresh, facts=fake_pin_facts(), path_exists=lambda _p: True)
    w96 = copy.deepcopy(config)
    w96["arm"] = "W96_factorized"
    w96["completed_arms"] = list(wd3.ARM_ORDER)
    with pytest.raises(wd3.WD3Error, match="capacity pressure"):
        wd3.validate_compiled_config(w96, facts=fake_pin_facts(), path_exists=lambda _p: True)


def test_scorer_free_arm_birth_is_typed_but_carries_no_launch_authority() -> None:
    config = valid_config()
    config.update(
        action="prepare_arm_birth",
        arm="D56",
        completed_arms=["W0_warm", "W0_reset"],
        device="cpu",
        teacher_cache_result=None,
        resume_from=None,
        launch_authorized=False,
        r5_exit_verified=False,
    )
    config["scorer_lane"] = {
        "claimed": False,
        "claim_id": None,
        "agent": "MAIN",
        "platform": "macos-cpu",
    }
    config["metal_lane"] = {
        "claimed": False,
        "claim_id": None,
        "agent": "MAIN",
        "platform": "macos-mps",
    }
    result = wd3.validate_compiled_config(config, facts=fake_pin_facts())
    assert result["status"] == "PASS"
    fire = wd3.compile_fire_order(config, facts=fake_pin_facts())
    assert fire["disposition"] == "READY_TO_MATERIALIZE_BUILD"


def test_n120_negative_and_n600_admission_are_hard_typed() -> None:
    ids = wd3.stratified_random_indices(np.arange(600) % 5).tolist()
    common = {
        "schema": "ddm_wd3_retained_subset_evaluation.v1",
        "pair_ids": ids,
        "n_pairs": 120,
        "all_payloads_retained": True,
        "evaluation_binding": {
            "pair_ids_sha256": wd3.canonical_sha256(ids),
            "cache_surface_sha256": "a" * 64,
        },
        "packet_archive": {
            "archive_bytes": wd3.BASE_BYTES,
            "receiver_parse_back_exact": True,
        },
    }
    baseline = {**common, "hard_d_seg": 0.0004, "d_pose": 0.0001}
    candidate = {**common, "hard_d_seg": 0.0005, "d_pose": 0.0001}
    verdict = wd3.compile_n120_negative_confirmation(
        arm="W0_warm",
        candidate=candidate,
        matched_baseline=baseline,
        expected_pair_ids=ids,
    )
    assert verdict["disposition"] == "INSTANCE_NEGATIVE"
    assert verdict["family_killed"] is False

    admission = wd3.compile_same_instrument_admission(
        {
            "n_pairs": 600,
            "hard_d_seg": 0.0003,
            "d_pose": 0.0001,
            "packet_archive": {
                "archive_bytes": wd3.BASE_BYTES,
                "receiver_parse_back_exact": True,
                "archive_repeat_byte_identical": True,
            },
            "all_payloads_retained": True,
            "same_instrument_base_sha256": wd3.PINS["base_receipt"][2],
            "authority_axis": "contest-CUDA",
        }
    )
    assert admission["disposition"] == "ADMIT"


def test_surgical_handoff_never_fabricates_qs_completion() -> None:
    candidate = {
        "schema": "ddm_wd3_retained_subset_evaluation.v1",
        "all_payloads_retained": True,
    }
    edge = {"per_edge_flips": {"Lane<->Road": 3}}
    blocked = wd3.compile_surgical_finish_handoff(candidate_receipt=candidate, residual_edge_map=edge, qs5_receipt=None)
    assert blocked["disposition"] == "BLOCKED"
    assert blocked["edits_materialized"] is False
    ready = wd3.compile_surgical_finish_handoff(
        candidate_receipt=candidate,
        residual_edge_map=edge,
        qs5_receipt={
            "schema": "ddm_qs5_receipt.v1",
            "repeat_identical": True,
            "pose_held_below_base": True,
            "receiver_consumed": True,
        },
    )
    assert ready["disposition"] == "READY_FOR_QS2_QS5_COMPILE"
    assert ready["edits_materialized"] is False


def test_runtime_patch_keeps_old_dispatch_and_adds_wd3(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    receipt = receiver.patch_runtime_tree(wd2_build.SOURCE_RUNTIME, destination)
    assert receipt["inactive_and_wd2_branches_retained"] is True
    residual = (destination / "runtime/residual_archive.py").read_text()
    f26 = (destination / "runtime/f26_inflate.py").read_text()
    assert 'startswith((b"WD2S", b"WD3Q"))' in residual
    assert '"wd3_receiver.py" if parts.semantic_blob.startswith(b"WD3Q")' in f26
    assert (destination / "cpr1/wd2_receiver.py").is_file()
    assert (destination / "cpr1/wd3_receiver.py").is_file()


def test_cache_repeat_finalization_is_byte_identical_on_tiny_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = {
        "a": (np.dtype("u1"), (4, 2)),
        "b": (np.dtype("<f4"), (4, 1)),
    }
    monkeypatch.setattr(wd3, "CACHE_FIELDS", fields)
    monkeypatch.setattr(receiver, "N", 4)
    for repeat in range(2):
        root = tmp_path / f"repeat_{repeat}" / "chunks"
        for start in (0, 2):
            wd3._atomic_npz(
                root / f"pairs_{start:04d}_{start + 2:04d}.npz",
                a=np.arange(start * 2, (start + 2) * 2, dtype=np.uint8).reshape(2, 2),
                b=np.arange(start, start + 2, dtype=np.float32).reshape(2, 1),
            )
        wd3._aggregate_cache_repeat(tmp_path / f"repeat_{repeat}", chunk_pairs=2)
    first = wd3._aggregate_cache_repeat(tmp_path / "repeat_0", chunk_pairs=2)
    second = wd3._aggregate_cache_repeat(tmp_path / "repeat_1", chunk_pairs=2)
    assert {name: row["sha256"] for name, row in first.items()} == {name: row["sha256"] for name, row in second.items()}


def test_build_verification_retains_payloads_and_stays_blocked(tmp_path: Path) -> None:
    receipt = wd3.verify_build(tmp_path)
    assert receipt["complete"] is True
    assert receipt["scorer_invocations"] == 0
    assert receipt["metal_invocations"] == 0
    assert receipt["dry_run_fire_order"]["disposition"] == "BLOCKED_NOT_LAUNCHABLE"
    assert receipt["payloads"]["uniform_packet"]["bytes"] > 0
    loaded = json.loads((tmp_path / "BUILD_RECEIPT.json").read_text())
    assert loaded["frontier_moved"] is False


def test_nonmonotone_rung_byte_cost_is_data_not_a_crash() -> None:
    """Governor-incident regression (2026-08-15): real coder measurements can code a
    higher-precision rung to the same or fewer bytes. A dominant (error-reducing) free
    rung must be TAKEN, a useless one SKIPPED — neither may abort the allocation."""
    model = tiny_model()
    sensitivity = {}
    first_name = None
    for name, value in model.state_dict().items():
        if value.ndim < 2:
            continue
        if first_name is None:
            first_name = name
        axis = value.ndim - 1 if name.endswith("embed.weight") else 0
        rows = []
        for group in range(value.shape[axis]):
            errors = {str(bit): 1.0 / (bit + group + 1) for bit in range(2, 9)}
            bytes_by_bit = {str(bit): bit + 2 for bit in range(2, 9)}
            rows.append({"group": group, "errors": errors, "bytes": bytes_by_bit})
        sensitivity[name] = rows
    # Dominant free rung: 2->3 bits on group 0 costs ZERO extra bytes yet reduces error.
    sensitivity[first_name][0]["bytes"]["3"] = sensitivity[first_name][0]["bytes"]["2"]
    # Useless free rung: 3->4 bits costs fewer bytes AND saves no error.
    sensitivity[first_name][0]["bytes"]["4"] = sensitivity[first_name][0]["bytes"]["3"] - 1
    sensitivity[first_name][0]["errors"]["4"] = sensitivity[first_name][0]["errors"]["3"]
    ceiling = sum(row["errors"]["4"] for rows in sensitivity.values() for row in rows)
    allocation = wd3.adaptive_allocation_from_sensitivity(
        model,
        sensitivity,
        maximum_predicted_error=ceiling,
        selection_sha256="b" * 64,
    )
    depths = {bit for values in allocation.bits.values() for bit in values}
    assert min(depths) >= 2 and max(depths) <= 8
    # The dominant free rung was taken: group 0 sits at >= 3 bits.
    assert allocation.bits[first_name][0] >= 3
