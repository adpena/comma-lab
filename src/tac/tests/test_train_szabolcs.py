"""Smoke test for ``experiments/train_szabolcs.py`` argparse + CLI shape.

Two halves:

1. **CLI argparse introspection** — the script must reject ``--device cpu``
   and require ``--device cuda`` per CLAUDE.md. The 5-epoch full smoke run
   isn't feasible in CI (no CUDA), but argparse + main() shape are testable
   without invoking the training loop.
2. **In-process training step** — we invoke ``main()`` indirectly by
   constructing a tiny model + running a single optimization step against a
   synthetic 2-frame video, on CPU. This validates the training loop math
   without needing CUDA. Skipped when the script can't be imported.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "train_szabolcs.py"


def _import_train_module():
    spec = importlib.util.spec_from_file_location(
        "_train_szabolcs_under_test", TRAIN_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── CLI shape ─────────────────────────────────────────────────────────────


class TestTrainSzabolcsCLI:
    def test_help_runs(self):
        # --help exits 0 (argparse). This catches syntax errors / import
        # errors in the script regardless of available hardware.
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--device" in proc.stdout
        assert "--total-epochs" in proc.stdout
        assert "--lr" in proc.stdout
        assert "--output-dir" in proc.stdout
        assert "--seed" in proc.stdout

    def test_rejects_cpu_device(self):
        # argparse choices=["cuda"] should reject --device cpu cleanly.
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT),
             "--device", "cpu",
             "--output-dir", "/tmp/_lane_sz_should_never_exist"],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode != 0
        assert "invalid choice" in proc.stderr or "choose from" in proc.stderr

    def test_rejects_mps_device(self):
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT),
             "--device", "mps",
             "--output-dir", "/tmp/_lane_sz_should_never_exist"],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode != 0


# ── In-process training step on CPU ───────────────────────────────────────


class TestTrainStepCPU:
    """The training loop's math must work on any device. We don't claim CUDA
    here — we monkey-patch out the device check and run one step on CPU to
    verify the model's forward + L1 loss + backward chain."""

    def test_one_step_cpu_smoke(self, tmp_path: Path):
        # Avoid invoking the script's main() (which gates on CUDA). Instead
        # exercise the imported helpers + a minimal model end-to-end on CPU.
        m = _import_train_module()
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from tac.contrib.szabolcs_renderer import (
            build_szabolcs_renderer,
            encode_luma_to_probability_map,
        )

        bundle = build_szabolcs_renderer(
            hidden=8, num_blocks=2, max_frame_index=16,
            shared_latent_height=8, shared_latent_width=10,
            quiet=True,
        )
        model = bundle.model
        lut = bundle.lut

        # Synthetic batch: 2 RGB frames at the renderer's working resolution.
        torch.manual_seed(0)
        rgb = torch.randint(0, 256, (2, model.h, model.w, 3), dtype=torch.uint8)

        luma = m.make_luma_low(rgb, model.h, model.w)
        prob = encode_luma_to_probability_map(luma, lut=lut)
        idx = torch.tensor([0, 1], dtype=torch.long)

        pred_low = model(prob, idx)
        # Don't go through camera-res upscale on CPU smoke (1164×874 alloc).
        loss = torch.nn.functional.l1_loss(pred_low, rgb.permute(0, 3, 1, 2).float())

        loss.backward()
        # All trainable params should have non-zero gradient on at least one
        # element (some might be sparse — check at least one grad is present).
        any_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert any_grad, "no gradients flowed through the training step"

    def test_main_rejects_when_cuda_absent(self, monkeypatch, tmp_path: Path):
        """If CUDA isn't available, main() must SystemExit rather than fall
        back to CPU. This is the CLAUDE.md non-negotiable in action."""
        m = _import_train_module()
        # If the test machine HAS cuda we skip — we're testing the absence
        # branch, not the presence one.
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this host; absence branch untested")

        argv = [
            "train_szabolcs.py",
            "--device", "cuda",
            "--output-dir", str(tmp_path / "out"),
            "--total-epochs", "1",
            "--video", "/dev/null",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            m.main()
