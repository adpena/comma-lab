# SPDX-License-Identifier: MIT
"""Tests for the canonical PACT-NeRV score-aware training-loop helper.

PACT-NERV-FULL-MAIN-IMPLEMENTATION-WAVE 2026-05-27. Exercises the
substrate-AGNOSTIC training loop on CPU with a tiny synthetic torch model +
mock loss callback so the loop logic (train/val split, EMA shadow,
best-checkpoint selection, NaN watchdog, runtime emission, deterministic zip)
is validated without GPU per the $0 MLX-local directive.

Per CLAUDE.md "MPS portable-local-substrate authority" + Catalog #127/#192:
these tests assert NON-PROMOTABLE behavior (no score claim); they validate
the AGNOSTIC scaffold only.
"""
from __future__ import annotations

import zipfile

import pytest

torch = pytest.importorskip("torch")

from tac.substrates._shared.pact_nerv_full_main import (  # noqa: E402
    CONTEST_NORMALIZER,
    EVAL_HW,
    N_PAIRS_FULL,
    PactNervTrainingResult,
    build_archive_zip,
    closed_form_weight_byte_proxy,
    run_pact_nerv_score_aware_training,
    write_contest_runtime,
)


class _TinyModel(torch.nn.Module):
    """Minimal substrate-shaped model: forward(idx) -> (rgb_0, rgb_1)."""

    def __init__(self, num_pairs: int = 16, h: int = 4, w: int = 4):
        super().__init__()
        self.num_pairs = num_pairs
        self.h = h
        self.w = w
        self.latents = torch.nn.Parameter(torch.randn(num_pairs, 8) * 0.02)
        self.proj = torch.nn.Linear(8, 3 * h * w * 2)

    def forward(self, idx: torch.Tensor):
        z = self.latents[idx]
        out = torch.sigmoid(self.proj(z))
        b = idx.shape[0]
        out = out.view(b, 2, 3, self.h, self.w)
        return out[:, 0], out[:, 1]


def _make_pairs(num_pairs: int = 16, h: int = 4, w: int = 4):
    return torch.rand(num_pairs, 2, 3, h, w) * 255.0


def _mse_compute_loss_factory():
    """Mock compute_loss: MSE proxy + a closed-form rate term (no scorers)."""

    def _compute_loss(
        model, idx, gt_0, gt_1, abp, *, gt_pose_batch, gt_seg_batch, gt_seg_already_probs
    ):
        del gt_pose_batch, gt_seg_batch, gt_seg_already_probs
        rgb_0, rgb_1 = model(idx)
        mse = torch.mean((rgb_0 * 255.0 - gt_0) ** 2) + torch.mean(
            (rgb_1 * 255.0 - gt_1) ** 2
        )
        rate = 25.0 * abp / CONTEST_NORMALIZER
        loss = mse + rate
        return loss, {"mse": mse.detach(), "rate": rate.detach()}

    return _compute_loss


def test_constants_canonical():
    assert CONTEST_NORMALIZER == 37_545_489.0
    assert EVAL_HW == (384, 512)
    assert N_PAIRS_FULL == 600


def test_closed_form_weight_byte_proxy_fp16():
    model = _TinyModel()
    proxy = closed_form_weight_byte_proxy(model)
    total = sum(p.numel() for p in model.parameters())
    assert float(proxy.item()) == pytest.approx(total * 2.0)
    assert proxy.dtype == torch.float32


def test_closed_form_weight_byte_proxy_fp32_doubles():
    model = _TinyModel()
    p16 = float(closed_form_weight_byte_proxy(model, fp16=True).item())
    p32 = float(closed_form_weight_byte_proxy(model, fp16=False).item())
    assert p32 == pytest.approx(2.0 * p16)


def test_training_loop_runs_and_returns_ema_state(tmp_path):
    torch.manual_seed(0)
    model = _TinyModel()
    pairs = _make_pairs()
    proxy = closed_form_weight_byte_proxy(model)
    result = run_pact_nerv_score_aware_training(
        model=model,
        pair_tensor=pairs,
        compute_loss=_mse_compute_loss_factory(),
        archive_bytes_proxy=proxy,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        substrate_tag="tinytest",
        epochs=4,
        batch_size=4,
        lr=1e-2,
        val_pair_count=4,
        val_every_epochs=2,
    )
    assert isinstance(result, PactNervTrainingResult)
    assert result.n_pairs == 16
    assert result.n_train_pairs + result.n_val_pairs == 16
    assert result.epochs_completed == 4
    assert "latents" in result.best_ema_state_dict
    assert (tmp_path / "best.pt").is_file()
    # EMA shadow state_dict matches model param names.
    assert set(result.best_ema_state_dict.keys()) == set(
        dict(model.state_dict()).keys()
    )


def test_training_loop_best_checkpoint_improves(tmp_path):
    torch.manual_seed(1)
    model = _TinyModel()
    pairs = _make_pairs()
    proxy = closed_form_weight_byte_proxy(model)
    result = run_pact_nerv_score_aware_training(
        model=model,
        pair_tensor=pairs,
        compute_loss=_mse_compute_loss_factory(),
        archive_bytes_proxy=proxy,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        substrate_tag="tinytest",
        epochs=6,
        batch_size=4,
        lr=2e-2,
        val_pair_count=4,
        val_every_epochs=2,
    )
    # A real improving checkpoint should have been observed (not fallback).
    assert result.best_epoch >= 0
    assert not result.used_end_of_training_fallback
    assert result.best_val_lagrangian == result.best_val_lagrangian  # not NaN


def test_nan_watchdog_aborts(tmp_path):
    model = _TinyModel()
    pairs = _make_pairs()
    proxy = closed_form_weight_byte_proxy(model)

    def _nan_loss(model, idx, gt_0, gt_1, abp, **kw):
        rgb_0, rgb_1 = model(idx)
        bad = rgb_0.sum() * float("nan")
        return bad, {"total": bad.detach()}

    with pytest.raises(RuntimeError, match="NaN watchdog"):
        run_pact_nerv_score_aware_training(
            model=model,
            pair_tensor=pairs,
            compute_loss=_nan_loss,
            archive_bytes_proxy=proxy,
            device=torch.device("cpu"),
            output_dir=tmp_path,
            substrate_tag="tinytest",
            epochs=2,
            batch_size=4,
            lr=1e-2,
            val_pair_count=4,
            val_every_epochs=1,
            max_nan_strikes=3,
        )


def test_stage_log_accumulates(tmp_path):
    model = _TinyModel()
    pairs = _make_pairs()
    proxy = closed_form_weight_byte_proxy(model)
    log: list = []
    run_pact_nerv_score_aware_training(
        model=model,
        pair_tensor=pairs,
        compute_loss=_mse_compute_loss_factory(),
        archive_bytes_proxy=proxy,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        substrate_tag="tinytest",
        epochs=2,
        batch_size=4,
        lr=1e-2,
        val_pair_count=4,
        val_every_epochs=1,
        stage_log=log,
    )
    stages = [r["stage"] for r in log]
    assert any(s.startswith("ema_wired_decay_") for s in stages)
    assert any(s.startswith("train_complete_elapsed_") for s in stages)


def test_write_contest_runtime_emits_compliant_pair(tmp_path):
    sub = tmp_path / "submission"
    write_contest_runtime(
        sub,
        substrate_pkg_name="pact_nerv_ia3",
        repo_root=__import__("pathlib").Path(__file__).resolve().parents[5],
    )
    sh = (sub / "inflate.sh").read_text()
    assert "set -euo pipefail" in sh
    assert '"$1"' in sh and '"$2"' in sh and '"$3"' in sh
    py = (sub / "inflate.py").read_text()
    # No scorer imports (strict-scorer-rule).
    assert "PoseNet" not in py and "SegNet" not in py
    assert "inflate_one_video" in py
    assert "sys.argv" in py
    # Vendored substrate package present (Catalog #295 self-containment).
    assert (sub / "src" / "tac" / "substrates" / "pact_nerv_ia3" / "inflate.py").is_file()


def test_build_archive_zip_deterministic(tmp_path):
    sub = tmp_path / "submission"
    write_contest_runtime(
        sub,
        substrate_pkg_name="pact_nerv_ia3",
        repo_root=__import__("pathlib").Path(__file__).resolve().parents[5],
    )
    bin_bytes = b"\x00\x01\x02PACTNERV"
    zpath = tmp_path / "archive.zip"
    build_archive_zip(zpath, bin_bytes=bin_bytes, submission_dir=sub)
    assert zpath.is_file()
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        assert "0.bin" in names
        assert "inflate.sh" in names
        assert "inflate.py" in names
        assert zf.read("0.bin") == bin_bytes
        # Fixed timestamp for byte-determinism (Catalog #19).
        for zi in zf.infolist():
            assert zi.date_time == (2026, 1, 1, 0, 0, 0)


def test_build_archive_zip_byte_stable_across_runs(tmp_path):
    sub = tmp_path / "submission"
    write_contest_runtime(
        sub,
        substrate_pkg_name="pact_nerv_ia3",
        repo_root=__import__("pathlib").Path(__file__).resolve().parents[5],
    )
    bin_bytes = b"deterministic"
    z1 = tmp_path / "a.zip"
    z2 = tmp_path / "b.zip"
    build_archive_zip(z1, bin_bytes=bin_bytes, submission_dir=sub)
    build_archive_zip(z2, bin_bytes=bin_bytes, submission_dir=sub)
    assert z1.read_bytes() == z2.read_bytes()
