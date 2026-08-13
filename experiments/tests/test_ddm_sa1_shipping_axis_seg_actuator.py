from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional

from experiments import ddm_js3_learned_implicit_conditioning as js3
from experiments import ddm_sa1_modal_t4_sign_gate as dispatcher
from experiments import ddm_sa1_shipping_axis_seg_actuator as sa1
from experiments import ddm_sa1_t4_sign_gate_worker as worker
from experiments.ddm_sa1_runtime import sa1_conditioner as receiver
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_counted_runtime_matches_training_model(tmp_path: Path) -> None:
    torch.manual_seed(js3.SEED)
    trained = js3.build_model(torch, functional, 4, 6.0, qat=True).eval()
    with torch.no_grad():
        trained.head.weight.fill_(0.125)
    exported = js3.serialize_module(trained, "int8", tmp_path / "module")
    decoded = js3.parse_module(exported.coded)
    js3.load_decoded_state(trained, decoded, torch)
    tokens = torch.zeros((1, js3.H, js3.W), dtype=torch.long)
    tokens[:, :, js3.W // 2 :] = 1
    pre_r = torch.full((1, 3, js3.H, js3.W), 127.5)
    with torch.inference_mode():
        expected = pre_r + trained(js3.fixed_context(torch, functional, tokens, pre_r))
        observed = receiver.apply_conditioner(exported.coded, tokens, pre_r)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_candidate_archive_keeps_pose_member_and_is_deterministic(tmp_path: Path) -> None:
    base = tmp_path / "base.zip"
    with zipfile.ZipFile(base, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("p", b"pose carrier")
    first = sa1.deterministic_archive(base, b"conditioner")
    second = sa1.deterministic_archive(base, b"conditioner")
    assert first == second
    candidate = tmp_path / "candidate.zip"
    candidate.write_bytes(first)
    with zipfile.ZipFile(candidate) as archive:
        assert archive.namelist() == ["p", "sa1_conditioner.br"]
        assert archive.read("p") == b"pose carrier"
        assert archive.read("sa1_conditioner.br") == b"conditioner"


def test_t4_gate_requires_seg_sign_and_joint_break_even(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "BASE_FLIPS", 2)
    monkeypatch.setattr(worker, "DENOMINATOR", 4)
    gt = np.zeros((1, 2, 2), dtype=np.uint8)
    candidate = gt.copy()
    candidate[0, 0, 0] = 1
    admitted = worker.adjudicate(
        candidate,
        gt,
        candidate_archive_bytes=worker.BASE_BYTES,
        local_pose_delta=-0.01,
    )
    assert admitted["status"] == "ADMITTED_FOR_FULL_EXACT_ROW"
    rejected = worker.adjudicate(
        candidate,
        gt,
        candidate_archive_bytes=worker.BASE_BYTES,
        local_pose_delta=0.1,
    )
    assert rejected["status"] == "REJECTED_BY_T4_SIGN_GATE"


def test_modal_gate_is_one_archive_and_fits_30_minutes() -> None:
    result = dispatcher.k_arithmetic()
    assert result["k_archives"] == 1
    assert result["scorer_passes"] == 1
    assert result["fits_30_minutes"] is True
    assert result["headroom_seconds"] > 900.0


def test_local_run_lock_refuses_a_duplicate_writer(tmp_path: Path) -> None:
    with (
        sa1.exclusive_run_lock(tmp_path),
        pytest.raises(sa1.SA1Error, match="another SA1 runner owns"),
        sa1.exclusive_run_lock(tmp_path),
    ):
        pass


def test_sa1_sources_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_sa1_shipping_axis_seg_actuator.py",
            "experiments/ddm_sa1_runtime/sa1_conditioner.py",
            "experiments/ddm_sa1_t4_sign_gate_worker.py",
            "experiments/ddm_sa1_modal_t4_sign_gate.py",
            "experiments/tests/test_ddm_sa1_shipping_axis_seg_actuator.py",
        ),
    )
    assert findings == []
