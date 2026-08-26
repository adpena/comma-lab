# SPDX-License-Identifier: MIT
"""Exact archive, target-binding, and launch-law tests for RB1."""

from __future__ import annotations

import copy
import json

import brotli
import numpy as np
import pytest
import torch

from experiments import ddm_rb1_born_small_receiver as rb1
from experiments import ddm_wd3_scorer_aware_width_distillation as wd3
from experiments import ddm_wd3_student_receiver as receiver


def test_rb1_complete_archive_is_deterministic_and_section_closed() -> None:
    body = b"exact-bs3-body" * 17
    semantic = b"real-coded-renderer" * 11
    first = rb1.pack_archive_bytes(body, semantic)
    second = rb1.pack_archive_bytes(body, semantic)

    assert first == second
    assert rb1.parse_archive_bytes(first) == (body, semantic)

    changed = bytearray(first)
    changed[31 + rb1.HEADER.size + 5] ^= 0x01
    with pytest.raises((rb1.RB1ReceiverError, ValueError)):
        rb1.parse_archive_bytes(bytes(changed))


def test_rb1_real_coder_renderer_roundtrip_is_byte_idempotent() -> None:
    torch.manual_seed(20260826)
    spec = receiver.StudentSpec("rb1_test_d8", "dense", 8, 1)
    model = receiver.StudentSemanticRenderer(spec)
    allocation = receiver.uniform_allocation(model, 4, selection_sha256="2" * 64)
    packet = receiver.pack_student(model, allocation)
    stream = brotli.compress(packet, mode=brotli.MODE_GENERIC, quality=11)

    restored = rb1.unpack_renderer(stream)

    assert receiver.pack_student(restored, allocation) == packet


def test_rb1_training_tokens_are_the_bound_parseback_memmap(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tokens.u8"
    expected = np.arange(receiver.N * receiver.EVAL_H * receiver.EVAL_W, dtype=np.uint8)
    expected.tofile(path)
    target = {"parsed_tokens": str(path)}
    monkeypatch.setattr(wd3, "target_object_binding", lambda _config: target)

    observed = wd3._load_training_tokens({"target_object": target})

    assert observed.shape == (receiver.N, receiver.EVAL_H, receiver.EVAL_W)
    assert observed.dtype == torch.uint8
    assert torch.equal(observed[0, :2, :3], torch.from_numpy(expected.reshape(observed.shape)[0, :2, :3]))


def test_rb1_aligned_retention_is_forced_under_its_own_cas_root(monkeypatch) -> None:
    monkeypatch.setattr(wd3, "target_object_binding", lambda _config: {"schema": "test"})
    config = {
        "evaluation_retention": {
            "schema": "ddm_w96b_evaluation_retention.v1",
            "mode": "content_addressed_chunks_v1",
            "cas_root": str(wd3.RB1_CAS_ROOT),
            "compact_after_verify": True,
        }
    }
    assert wd3.evaluation_retention_config(config) == config["evaluation_retention"]

    drifted = copy.deepcopy(config)
    drifted["evaluation_retention"]["cas_root"] = str(wd3.ALIGNED_CAS_ROOT)
    with pytest.raises(wd3.WD3Error, match="retention contract differs"):
        wd3.evaluation_retention_config(drifted)


def test_rb1_arm_gate_does_not_inherit_old_object_verdict_state(monkeypatch) -> None:
    target = {"admitted_arms": ["D56", "F64"]}
    monkeypatch.setattr(wd3, "target_object_binding", lambda _config: target)
    config = {
        "action": "train",
        "arm": "D56",
        "completed_arms": [],
        "negative_confirmed_arms": [],
        "capacity_pressure_confirmed": False,
        "real_coder_override_dense_w96": False,
    }
    wd3._validate_arm(config)

    drifted = copy.deepcopy(config)
    drifted["completed_arms"] = ["W0_warm"]
    with pytest.raises(wd3.WD3Error, match="must not inherit"):
        wd3._validate_arm(drifted)


def test_teacher_master_recovers_after_final_rename_before_receipt(tmp_path, monkeypatch) -> None:
    tokens = tmp_path / "tokens.u8"
    semantic = tmp_path / "semantic.bin"
    destination = tmp_path / "teacher_master.u8"
    receipt_path = tmp_path / "teacher_master.json"
    tokens.write_bytes(b"tokens")
    semantic.write_bytes(b"semantic")
    destination.write_bytes(bytes(range(18)))
    target = {
        "parsed_tokens": str(tokens),
        "semantic_renderer": str(semantic),
        "semantic_renderer_sha256": "a" * 64,
        "gb1_receiver_files": {"receiver.py": {"sha256": "b" * 64}},
        "teacher_master": str(destination),
        "teacher_master_receipt": str(receipt_path),
    }
    monkeypatch.setattr(wd3, "target_object_binding", lambda _config: target)
    monkeypatch.setattr(receiver, "N", 1)
    monkeypatch.setattr(receiver, "CAMERA_H", 2)
    monkeypatch.setattr(receiver, "CAMERA_W", 3)
    binding = {
        "schema": "ddm_rb1_teacher_master_binding.v1",
        "target_object_sha256": wd3.canonical_sha256(target),
        "tokens": wd3.file_record(tokens),
        "semantic_renderer": wd3.file_record(semantic),
        "gb1_receiver_files": target["gb1_receiver_files"],
        "geometry": [1, 3, 2, 3],
        "rounding": "bilinear_align_corners_false_clamp_0_255_round_uint8",
    }
    progress_path = destination.with_name(f".{destination.name}.progress.json")
    progress_path.write_text(
        json.dumps(
            {
                "schema": "ddm_rb1_teacher_master_progress.v1",
                "binding": binding,
                "completed_pairs": 1,
                "complete": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = wd3._materialize_target_teacher_master(
        {"target_object": target},
        device=torch.device("cpu"),
    )

    assert receipt is not None
    assert receipt["recovered_after_atomic_rename"] is True
    assert receipt["payload"] == wd3.file_record(destination)
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["complete"] is True
