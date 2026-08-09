from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from tac.pr130_lift.pose import mps_port
from tac.pr130_lift.pose import train_pose_carrier_full_resumable as pose_train
from tac.pr130_lift.pose.source_loader import load_lifted_module
from tac.pr130_lift.pose.train_pose_carrier_full_resumable import (
    _load_full_state,
    _save_full_state,
)


def _args(save: Path) -> argparse.Namespace:
    return argparse.Namespace(
        target_cache="target.pt",
        master_checkpoint="semantic.pt",
        init_carrier="carrier.pt",
        master_cache="masters.pt",
        reuse_master_cache=True,
        cache_masters_on_device=False,
        steps=10,
        batch_size=2,
        eval_batch_size=2,
        lr_basis=0.01,
        lr_coeff=0.02,
        basis_freeze_fraction=0.0,
        basis_train_until_fraction=1.0,
        qat_fraction=0.5,
        coeff_qat_fraction=0.5,
        metric_loss_after_basis=False,
        always_metric_loss=True,
        metric_normalized_weight=0.0,
        hard_mining_power=0.0,
        hard_mining_max=8.0,
        basis_bits=6,
        coeff_bits=12,
        amplitude=64.0,
        master_carrier_amplitude=0.0,
        carrier_base="gray",
        zero_init_coeff=False,
        seed=123,
        device="cpu",
        row_local_mode="reference-sparse",
        smoke_pairs=4,
        save=save,
    )


def test_full_state_round_trip_restores_training_state(tmp_path: Path) -> None:
    args = _args(tmp_path / "carrier.pt")
    raw_basis = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    coeff = torch.nn.Embedding(4, 2, sparse=False)
    with torch.no_grad():
        coeff.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2))
    basis_optimizer = torch.optim.Adam([raw_basis], lr=args.lr_basis)
    coeff_optimizer = torch.optim.Adam([coeff.weight], lr=args.lr_coeff)
    basis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(basis_optimizer, T_max=10)
    coeff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(coeff_optimizer, T_max=10)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.tensor([3, 1, 2, 0])
    sampling_weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    active_pair_ids = torch.tensor([0, 200, 399, 599])
    best = {
        "mean": 1.25,
        "basis": raw_basis.detach().clone(),
        "coeff": coeff.weight.detach().clone(),
    }
    history = [{"step": 3, "mean": 1.25}]
    execution_provenance = {
        "schema": "ddm_fx2_pose_optimizer_provenance.v1",
        "score_claim": False,
        "optimizer_class": "test.Adam",
        "row_local_mode": "reference-sparse",
        "gradient_representation": "sparse",
        "fallback_event": "none",
        "torch_version": torch.__version__.split("+", 1)[0],
        "git_sha": "test-only",
        "argv": ["test"],
        "native_probe_receipt": {"expected_sha256": "test-only"},
    }

    path = _save_full_state(
        args=args,
        step=3,
        raw_basis=raw_basis,
        coeff=coeff,
        basis_optimizer=basis_optimizer,
        coeff_optimizer=coeff_optimizer,
        basis_scheduler=basis_scheduler,
        coeff_scheduler=coeff_scheduler,
        generator=generator,
        order=order,
        cursor=2,
        sampling_weights=sampling_weights,
        history=history,
        best=best,
        active_pair_ids=active_pair_ids,
        execution_provenance=execution_provenance,
    )

    saved_payload = torch.load(path, map_location="cpu", weights_only=False)
    latest_payload = torch.load(
        pose_train._latest_state_path(args.save),
        map_location="cpu",
        weights_only=False,
    )
    assert saved_payload["execution_provenance"] == execution_provenance
    assert latest_payload["execution_provenance"] == execution_provenance
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))

    with torch.no_grad():
        raw_basis.zero_()
        coeff.weight.zero_()
    start_step, loaded_order, cursor, loaded_weights, loaded_history, loaded_best = _load_full_state(
        path,
        args=args,
        raw_basis=raw_basis,
        coeff=coeff,
        basis_optimizer=basis_optimizer,
        coeff_optimizer=coeff_optimizer,
        basis_scheduler=basis_scheduler,
        coeff_scheduler=coeff_scheduler,
        generator=generator,
        active_pair_ids=active_pair_ids,
        device=torch.device("cpu"),
    )

    assert start_step == 4
    assert cursor == 2
    assert torch.equal(loaded_order, order)
    assert torch.equal(loaded_weights, sampling_weights)
    assert loaded_history == history
    assert loaded_best["mean"] == 1.25
    assert torch.equal(raw_basis, best["basis"])
    assert torch.equal(coeff.weight, best["coeff"])


def test_execution_provenance_names_actual_reference_mechanism(tmp_path: Path) -> None:
    args = _args(tmp_path / "carrier.pt")
    lifted = load_lifted_module("train_pose_carrier_full")
    coefficients, optimizer = mps_port.build_row_local_coefficients(
        num_embeddings=8,
        embedding_dim=3,
        device=torch.device("cpu"),
        lr=args.lr_coeff,
        sparse_optimizer_type=lifted.RowLocalSparseAdam,
        mode=args.row_local_mode,
    )
    provenance = pose_train._execution_provenance(
        args=args,
        coefficients=coefficients,
        optimizer=optimizer,
        argv=["trainer.py", "--row-local-mode", "reference-sparse"],
    )

    assert provenance["score_claim"] is False
    assert provenance["optimizer_class"].endswith(".RowLocalSparseAdam")
    assert provenance["row_local_mode"] == "reference-sparse"
    assert provenance["gradient_representation"] == "sparse"
    assert provenance["selection_event"] == "reference_default"
    assert provenance["fallback_event"] == "none"
    assert provenance["fallback_policy"] == "automatic_fallback_forbidden"
    assert provenance["torch_version"] == torch.__version__.split("+", 1)[0]
    assert len(provenance["git_sha"]) == 40
    assert provenance["argv"] == [
        "trainer.py",
        "--row-local-mode",
        "reference-sparse",
    ]
    receipt = provenance["native_probe_receipt"]
    assert receipt["status"] == "verified_at_run"
    assert receipt["expected_sha256"] == pose_train.NATIVE_SPARSE_RECEIPT_SHA256
    assert receipt["observed_sha256"] == pose_train.NATIVE_SPARSE_RECEIPT_SHA256


def test_legacy_mps_resume_requires_explicit_dense_adapter(tmp_path: Path) -> None:
    args = _args(tmp_path / "carrier.pt")
    args.device = "mps:0"
    active_pair_ids = torch.tensor([0, 200, 399, 599])
    prior_args = pose_train._jsonable_args(args)
    prior_args.pop("row_local_mode")
    payload = {
        "schema": "ddm_mx2_pose_carrier_full_state.v1",
        "args": prior_args,
        "active_pair_ids": active_pair_ids.tolist(),
    }

    with pytest.raises(ValueError, match="optimizer mode mismatch"):
        pose_train._check_resume_compatibility(payload, args, active_pair_ids)

    args.row_local_mode = "dense-adapter"
    pose_train._check_resume_compatibility(payload, args, active_pair_ids)


def _pose_cli(tmp_path: Path) -> list[str]:
    return [
        "train_pose_carrier_full_resumable.py",
        "--challenge-root",
        str(tmp_path / "challenge"),
        "--target-cache",
        str(tmp_path / "target.pt"),
        "--master-checkpoint",
        str(tmp_path / "master.pt"),
        "--init-carrier",
        str(tmp_path / "carrier.pt"),
        "--out",
        str(tmp_path / "out.json"),
        "--save",
        str(tmp_path / "save.pt"),
    ]


def test_pose_trainer_refuses_raw_when_admission_enforced(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    monkeypatch.delenv("TAC_GOVERNED_ADMISSION", raising=False)
    monkeypatch.delenv("TAC_ADMISSION_BYPASS_OK", raising=False)
    monkeypatch.setattr(sys, "argv", _pose_cli(tmp_path))

    with pytest.raises(SystemExit) as excinfo:
        pose_train.main()

    assert excinfo.value.code == 7


def test_pose_trainer_governed_env_passes_guard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    monkeypatch.setenv("TAC_GOVERNED_ADMISSION", "1")
    monkeypatch.setattr(sys, "argv", _pose_cli(tmp_path))
    monkeypatch.setattr(pose_train, "lifted_script_path", lambda: nullcontext())
    called: dict[str, argparse.Namespace] = {}

    def fake_run(args: argparse.Namespace) -> dict[str, bool]:
        called["args"] = args
        return {"ok": True}

    monkeypatch.setattr(pose_train, "run", fake_run)

    pose_train.main()

    assert called["args"].save == tmp_path / "save.pt"
    assert called["args"].row_local_mode == "reference-sparse"
