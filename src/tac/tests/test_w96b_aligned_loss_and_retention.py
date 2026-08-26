# SPDX-License-Identifier: MIT
"""Exact W96B aligned-law and lossless content-addressed retention contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import ddm_wd3_scorer_aware_width_distillation as wd3
from tac import content_addressed_retention as cas
from tac.witness_dsl import lever_registry


def _objective_inputs() -> dict[str, object]:
    student_logits = torch.tensor(
        [[[[2.0, -0.5]], [[0.0, 1.5]], [[-1.0, 0.5]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[[[1.5, 0.0]], [[0.0, 1.0]], [[-0.5, 0.5]]]], dtype=torch.float64
    )
    student_pose = torch.tensor([[0.25, -0.1, 0.2, 0.0, 0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    return {
        "student_logits": student_logits,
        "student_pose6": student_pose,
        "student_frame1": torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64),
        "teacher_logits": teacher_logits,
        "teacher_argmax": teacher_logits.argmax(dim=1),
        "teacher_margin": torch.tensor([[[1.0, 0.5]]], dtype=torch.float64),
        "teacher_pose6": torch.zeros((1, 6), dtype=torch.float64),
        "original_argmax": torch.tensor([[[0, 1]]], dtype=torch.long),
        "original_pose6": torch.zeros((1, 6), dtype=torch.float64),
        "teacher_frame1": torch.zeros((1, 1, 1, 2), dtype=torch.float64),
        "selected_cells": torch.tensor([[[True, True]]]),
        "thresholds": wd3.StageThresholds(
            calibration_scale=0.37,
            margin_ceiling=100.0,
            teacher_kl_ceiling=100.0,
            decode_ceiling=100.0,
        ),
        "duals": wd3.DualState(),
    }


def test_default_off_is_the_exact_legacy_target_probability_law() -> None:
    inputs = _objective_inputs()
    total, components = wd3.score_native_objective(**inputs)
    logits = inputs["student_logits"]
    target = inputs["original_argmax"]
    target_probability = logits.softmax(dim=1).gather(1, target[:, None]).squeeze(1)
    soft = (1.0 - target_probability)[inputs["selected_cells"]].mean()
    pose_mse = inputs["student_pose6"].square().mean()
    expected = 100.0 * 0.37 * soft + torch.sqrt(torch.clamp(10.0 * pose_mse, min=1e-20))

    assert torch.equal(total, expected)
    assert torch.equal(components["seg_axis_train_loss_proxy"], soft)
    assert "seg_axis_expected_flip_margin_score" not in components
    assert "seg_axis_expected_flip_tau" not in components


def test_aligned_law_is_exact_target_vs_best_other_expected_flip() -> None:
    from tac.pr130_lift.lifted.semantic_renderer_oracle import target_margin as ce1_target_margin

    inputs = _objective_inputs()
    tau = 0.125
    total, components = wd3.score_native_objective(
        **inputs,
        seg_loss_law=wd3.SEG_LOSS_EXPECTED_FLIP_MARGIN,
        expected_flip_temperature=tau,
    )
    logits = inputs["student_logits"]
    target = inputs["original_argmax"]
    target_logits = logits.gather(1, target[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, target[:, None], -1.0e9)
    margin = target_logits - other.amax(dim=1)
    expected_seg = 100.0 * torch.sigmoid(-margin[inputs["selected_cells"]] / tau).mean()
    pose = torch.sqrt(10.0 * inputs["student_pose6"].square().mean())

    assert torch.equal(wd3.expected_flip_target_margin(logits, target), margin)
    assert torch.equal(wd3.expected_flip_target_margin(logits, target), ce1_target_margin(logits, target).squeeze(1))
    assert torch.equal(components["seg_axis_expected_flip_margin_score"], expected_seg)
    assert torch.equal(total, expected_seg + pose)


def test_pose_supervision_is_live_at_aligned_step_zero() -> None:
    inputs = _objective_inputs()
    total, _ = wd3.score_native_objective(
        **inputs,
        seg_loss_law=wd3.SEG_LOSS_EXPECTED_FLIP_MARGIN,
        expected_flip_temperature=wd3.expected_flip_tau(0, 65 * 600),
    )
    total.backward()

    assert inputs["student_pose6"].grad is not None
    assert torch.count_nonzero(inputs["student_pose6"].grad).item() > 0


def test_tau_schedule_has_exact_full_window_endpoints_and_linear_resume_identity() -> None:
    total_steps = 65 * 600
    split = 19 * 600 + 17

    assert wd3.expected_flip_tau(0, total_steps) == 0.15
    assert wd3.expected_flip_tau(total_steps - 1, total_steps) == 0.05
    assert wd3.expected_flip_tau(split, total_steps) == pytest.approx(
        0.15 - 0.10 * split / (total_steps - 1), abs=0.0
    )
    assert wd3.expected_flip_tau(split, total_steps) == wd3.expected_flip_tau(split, total_steps)


def _scheduler_config(*, aligned: bool) -> dict[str, object]:
    objective = {
        "scoreaware": True,
        "seg_score_coefficient": 100.0,
        "pose_exact_nonlinear": True,
        "temperature": 2.0,
        "adaptive_duals": True,
        "decode_mse_ceiling": wd3.DECODE_MSE_CEILING,
        "packet_quantizer_in_loop": True,
    }
    if aligned:
        objective.update(
            {
                "seg_loss_law": wd3.SEG_LOSS_EXPECTED_FLIP_MARGIN,
                "expected_flip_tau_start": 0.15,
                "expected_flip_tau_end": 0.05,
                "full_window_epochs": 65,
                "pose_start_step": 0,
            }
        )
    return {
        "arm": "W96_flattened",
        "epochs": 65,
        "objective": objective,
        "optimizer": {
            "lr": 2.0e-5,
            "weight_decay": 1.0e-4,
            "grad_clip": 1.0,
            "dual_step": 1.0e-3,
            "reset_ramp_divisor": 3.16,
        },
    }


def test_cosine_floor_is_one_percent_only_for_aligned_law() -> None:
    aligned_model = torch.nn.Linear(2, 1)
    off_model = copy.deepcopy(aligned_model)
    _, aligned = wd3._new_optimizer_scheduler(aligned_model, _scheduler_config(aligned=True))
    _, off = wd3._new_optimizer_scheduler(off_model, _scheduler_config(aligned=False))

    assert aligned.eta_min == pytest.approx(2.0e-7)
    assert off.eta_min == pytest.approx(4.0e-7)


def test_resume_identity_masks_path_and_builder_but_not_loss_law_or_tau() -> None:
    base = _scheduler_config(aligned=True) | {
        "resume_from": "/before.pt",
        "expected_builder_sha256": "a" * 64,
    }
    repointed = copy.deepcopy(base)
    repointed["resume_from"] = "/after.pt"
    repointed["expected_builder_sha256"] = "b" * 64
    assert wd3._resume_config_identity(base) == wd3._resume_config_identity(repointed)

    for field, value in (("expected_flip_tau_end", 0.051), ("seg_loss_law", "renamed_surrogate")):
        drifted = copy.deepcopy(base)
        drifted["objective"][field] = value
        assert wd3._resume_config_identity(base) != wd3._resume_config_identity(drifted)


def test_aligned_profile_refuses_surrogate_rename_tau_pose_or_missing_cas() -> None:
    base = _scheduler_config(aligned=True)
    for field, value in (
        ("seg_loss_law", "aligned_target_probability"),
        ("expected_flip_tau_start", 0.2),
        ("expected_flip_tau_end", 0.04),
        ("pose_start_step", 1),
    ):
        drifted = copy.deepcopy(base)
        drifted["objective"][field] = value
        with pytest.raises(wd3.WD3Error, match="aligned expected-flip"):
            wd3.objective_profile(drifted)

    assert wd3.evaluation_retention_config(base) is None
    retained = copy.deepcopy(base)
    retained["evaluation_retention"] = {
        "schema": "ddm_w96b_evaluation_retention.v1",
        "mode": "content_addressed_chunks_v1",
        "cas_root": str(wd3.ALIGNED_CAS_ROOT),
        "compact_after_verify": True,
    }
    assert wd3.evaluation_retention_config(retained) == retained["evaluation_retention"]


def _write_evaluation_fixture(root: Path, *, variable: int) -> dict[str, str]:
    root.mkdir(parents=True)
    fixed = bytes([7]) * cas.CAMERA_FRAME_BYTES
    moving = bytes([variable]) * cas.CAMERA_FRAME_BYTES
    (root / "receiver_pairs.rgb.u8").write_bytes(fixed + moving + fixed + moving)
    shared = np.arange(128, dtype="<f4")
    np.savez(root / "scorer_chunk.npz", shared=shared, unique=np.asarray([variable], dtype="<i8"))
    np.savez(root / "scorer_outputs.npz", shared=shared, unique=np.asarray([variable + 1], dtype="<i8"))
    (root / "receipt.json").write_text(json.dumps({"variable": variable}) + "\n", encoding="utf-8")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_content_addressed_retention_compacts_and_restores_every_payload(tmp_path: Path) -> None:
    logical = tmp_path / "logical"
    expected = _write_evaluation_fixture(logical, variable=11)
    store = tmp_path / "cas"
    manifest_path = logical / "CAS_RETENTION_MANIFEST.json"

    manifest = cas.retain_tree(
        logical,
        store=store,
        manifest_path=manifest_path,
        compact=True,
        exclude_relative=(manifest_path.name,),
    )
    assert manifest["logical_bytes"] > manifest["unique_object_bytes_within_tree"]
    assert sorted(path.name for path in logical.iterdir()) == [manifest_path.name]
    cas.verify_manifest(manifest_path, deep=True)

    one_file = cas.restore_logical_file(manifest_path, "scorer_outputs.npz", tmp_path / "one_file.npz")
    assert one_file["sha256"] == expected["scorer_outputs.npz"]
    assert one_file["symlink"] is False

    destination = tmp_path / "restored"
    receipt = cas.restore_tree(manifest_path, destination)
    assert receipt["all_files_byte_identical"] is True
    assert receipt["symlinks_used"] is False
    assert {
        path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.rglob("*")
        if path.is_file()
    } == expected


def test_inventory_measures_cross_tree_dedup_without_changing_sources(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hashes = _write_evaluation_fixture(first, variable=21)
    second_hashes = _write_evaluation_fixture(second, variable=22)

    inventory = cas.inventory_trees((first, second))

    assert inventory["tree_count"] == 2
    assert inventory["post_dedup_allocated_bytes"] < inventory["logical_allocated_bytes"]
    assert inventory["all_payloads_recoverable_by_manifest"] is True
    assert _write_hashes(first) == first_hashes
    assert _write_hashes(second) == second_hashes


def _write_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_retention_refuses_symlink_payloads(tmp_path: Path) -> None:
    root = tmp_path / "logical"
    root.mkdir()
    target = root / "payload.bin"
    target.write_bytes(b"retained")
    (root / "alias.bin").symlink_to(target)

    with pytest.raises(cas.ContentAddressedRetentionError, match="symlink payload is forbidden"):
        cas.scan_tree(root)


def test_retention_detects_corrupt_object_before_source_compaction(tmp_path: Path) -> None:
    logical = tmp_path / "logical"
    _write_evaluation_fixture(logical, variable=31)
    store = tmp_path / "cas"
    manifest_path = logical / "CAS_RETENTION_MANIFEST.json"
    cas.retain_tree(
        logical,
        store=store,
        manifest_path=manifest_path,
        compact=False,
        exclude_relative=(manifest_path.name,),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk = manifest["files"][0]["chunks"][0]
    object_path = store / "objects" / "sha256" / chunk["sha256"][:2] / chunk["sha256"]
    payload = bytearray(object_path.read_bytes())
    payload[0] ^= 0xFF
    object_path.write_bytes(payload)

    with pytest.raises(cas.ContentAddressedRetentionError, match="CAS object bytes differ"):
        cas.verify_manifest(manifest_path, deep=True)
    assert (logical / "receiver_pairs.rgb.u8").is_file()


def test_w96b_dsl_factories_are_real_compiled_config_levers() -> None:
    rows = [
        row
        for row in lever_registry.package_lever_factories()
        if row.module == "w96b_aligned_loss_levers_20260826.py"
    ]

    assert {row.factory for row in rows} == {
        "lever_w96b_expected_flip_seed_20260815",
        "lever_w96b_expected_flip_seed_20260816",
    }
    assert all(row.flags == ("--compiled-config",) for row in rows)
    assert all(not row.missing_flags and row.trainer_declared for row in rows)
