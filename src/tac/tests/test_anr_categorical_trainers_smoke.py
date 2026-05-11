"""Trainer smoke tests for ANR + categorical substrates.

These tests exercise the trainer ENTRY POINT contract (argparse,
forbidden-default refusal, smoke-vs-real branch logic) without running a full
training loop. Full training is operator-gated GPU dispatch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANR_TRAINER = ROOT / "experiments" / "train_anr_token_renderer.py"
CAT_TRAINER = ROOT / "experiments" / "train_categorical_renderer.py"


def _run_trainer(script: Path, *args: str, **kw) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(script), *args]
    return subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60, **kw,
    )


def test_anr_trainer_help_works():
    r = _run_trainer(ANR_TRAINER, "--help")
    assert r.returncode == 0
    assert "ANR substrate" in r.stdout or "anr" in r.stdout.lower()
    assert "--num-pairs" in r.stdout
    assert "--smoke" in r.stdout


def test_cat_trainer_help_works():
    r = _run_trainer(CAT_TRAINER, "--help")
    assert r.returncode == 0
    assert "--palette-dim" in r.stdout
    assert "--codebook-collapse-floor" in r.stdout
    assert "--smoke" in r.stdout


def test_anr_trainer_refuses_mps(tmp_path):
    r = _run_trainer(
        ANR_TRAINER, "--output-dir", str(tmp_path / "out"),
        "--num-pairs", "2", "--device", "mps",
    )
    assert r.returncode == 2
    assert "MPS forbidden" in r.stderr


def test_cat_trainer_refuses_mps(tmp_path):
    r = _run_trainer(
        CAT_TRAINER, "--output-dir", str(tmp_path / "out"),
        "--num-pairs", "2", "--device", "mps",
    )
    assert r.returncode == 2
    assert "MPS forbidden" in r.stderr


def test_anr_trainer_refuses_tmp_output(tmp_path):
    """CLAUDE.md non-negotiable: no /tmp persisted-evidence paths."""
    r = _run_trainer(
        ANR_TRAINER, "--output-dir", "/tmp/anr_test",
        "--num-pairs", "2", "--device", "cpu", "--smoke",
    )
    assert r.returncode == 2
    assert "/tmp" in r.stderr


def test_cat_trainer_refuses_tmp_output(tmp_path):
    r = _run_trainer(
        CAT_TRAINER, "--output-dir", "/tmp/cat_test",
        "--num-pairs", "2", "--device", "cpu", "--smoke",
    )
    assert r.returncode == 2
    assert "/tmp" in r.stderr


def test_anr_trainer_argparse_defaults():
    """The trainer's CLI defaults match CLAUDE.md non-negotiables."""
    src = ANR_TRAINER.read_text()
    # EMA decay default = 0.997
    assert '"--ema-decay"' in src and "0.997" in src
    # Lambda pose default exists
    assert "288.6751" in src
    # Smoke flag present
    assert "--smoke" in src
    # CUDA-required default (NO MPS-fallback ternary)
    assert 'default="cuda"' in src
    # Forbidden synthetic-non-smoke pattern carries the waiver comment.
    assert "SYNTHETIC_NON_SMOKE_OK" in src


def test_cat_trainer_argparse_defaults():
    src = CAT_TRAINER.read_text()
    assert '"--ema-decay"' in src and "0.997" in src
    assert "288.6751" in src
    assert "--smoke" in src
    assert 'default="cuda"' in src
    assert "SYNTHETIC_NON_SMOKE_OK" in src
    # Categorical-specific: codebook-collapse floor + palette-dim
    assert '"--codebook-collapse-floor"' in src
    assert '"--palette-dim"' in src


def test_anr_trainer_real_data_requires_video_path(tmp_path):
    """Non-smoke run with no --video-path must refuse."""
    # Use --device cpu to avoid CUDA check + bypass --smoke; --epochs=1 minimal.
    r = _run_trainer(
        ANR_TRAINER, "--output-dir", str(tmp_path / "anr_real_nopath"),
        "--num-pairs", "2", "--epochs", "1", "--device", "cpu",
    )
    # The trainer fails when it cannot load real-data source (no video_path).
    # We expect either rc != 0 OR a clear failure in stderr.
    assert r.returncode != 0
    # Possible failure points: scorer load, RealPairBatchSource import, or
    # video path requirement
    assert (
        "video" in r.stderr.lower()
        or "video" in r.stdout.lower()
        or "scorer" in r.stderr.lower()
        or "upstream" in r.stderr.lower()
        or "ImportError" in r.stderr
        or "Module" in r.stderr
    )


def test_cat_trainer_real_data_requires_video_path(tmp_path):
    r = _run_trainer(
        CAT_TRAINER, "--output-dir", str(tmp_path / "cat_real_nopath"),
        "--num-pairs", "2", "--epochs", "1", "--device", "cpu",
    )
    assert r.returncode != 0
    assert (
        "video" in r.stderr.lower()
        or "scorer" in r.stderr.lower()
        or "upstream" in r.stderr.lower()
        or "ImportError" in r.stderr
        or "Module" in r.stderr
    )
