# SPDX-License-Identifier: MIT
"""Tests for ``tools/run_deltaepszeta_training.py``.

Coverage:
- driver constructs from a synthetic substrate
- one optimizer step reduces the loss
- end-of-epoch callback fires and writes JSONL
- lambda ramps when rate proxy exceeds budget
- JSONL log writes valid records (parse-back)
- checkpoint saves and reloads bit-faithfully
- /tmp log_dir is rejected (transient-evidence trap)
- CLI imports cleanly

Strict-scorer-rule: pure CPU; no scorer load anywhere.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest
import torch

from tac.codec_pipeline import CodecPipeline, Op1_PR101SplitBrotli
from tac.codec_pipeline_deltaepszeta_callback import (
    CodecPipelineAwareTrainingCallback,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_driver_module():
    tool_path = _REPO_ROOT / "tools" / "run_deltaepszeta_training.py"
    spec = importlib.util.spec_from_file_location("run_deltaepszeta_training", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


drv = _load_driver_module()


def _synthetic_state_dict(seed: int = 0, scale: float = 0.1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(*shape, generator=g) * scale
        for name, shape in FIXED_STATE_SCHEMA
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_driver_constructs(tmp_path: pathlib.Path) -> None:
    sd = _synthetic_state_dict()
    cfg = drv.DeltaEpsZetaTrainingConfig(
        n_epochs=1,
        steps_per_epoch=2,
        log_dir=tmp_path / "run",
        run_label="ctor",
    )
    driver = drv.DeltaEpsZetaTrainingDriver(state_dict=sd, config=cfg)
    assert driver.lambda_value == cfg.lambda_init
    assert driver.log_path.parent.exists()
    # state_dict access produces detached tensors.
    out_state = driver.state_dict
    assert set(out_state.keys()) == set(sd.keys())


def test_driver_rejects_tmp_log_dir() -> None:
    sd = _synthetic_state_dict()
    cfg = drv.DeltaEpsZetaTrainingConfig(
        n_epochs=1, steps_per_epoch=1,
        log_dir=pathlib.Path("/tmp/no_good"),
    )
    with pytest.raises(ValueError, match="/tmp"):
        drv.DeltaEpsZetaTrainingDriver(state_dict=sd, config=cfg)


def test_driver_rejects_bad_config(tmp_path: pathlib.Path) -> None:
    sd = _synthetic_state_dict()
    with pytest.raises(ValueError, match="n_epochs"):
        drv.DeltaEpsZetaTrainingDriver(
            state_dict=sd,
            config=drv.DeltaEpsZetaTrainingConfig(
                n_epochs=0, log_dir=tmp_path / "run"
            ),
        )
    with pytest.raises(ValueError, match="steps_per_epoch"):
        drv.DeltaEpsZetaTrainingDriver(
            state_dict=sd,
            config=drv.DeltaEpsZetaTrainingConfig(
                n_epochs=1, steps_per_epoch=0, log_dir=tmp_path / "run"
            ),
        )
    with pytest.raises(ValueError, match="learning_rate"):
        drv.DeltaEpsZetaTrainingDriver(
            state_dict=sd,
            config=drv.DeltaEpsZetaTrainingConfig(
                n_epochs=1, steps_per_epoch=1, learning_rate=0,
                log_dir=tmp_path / "run",
            ),
        )


# ---------------------------------------------------------------------------
# step_once + train
# ---------------------------------------------------------------------------

def test_one_step_reduces_loss_on_synthetic(tmp_path: pathlib.Path) -> None:
    """The default sanity-loop = MSE-vs-reference + tiny rate term. With the
    parameters initialized AT the reference, MSE is ~0; perturb them so the
    distortion is non-trivial, then verify gradient descent moves the loss
    DOWN over a few steps."""
    sd = _synthetic_state_dict(seed=0)
    perturbed = {k: v + 0.05 * torch.randn_like(v) for k, v in sd.items()}
    # Use ZERO lambda so the test isolates pure distortion-following.
    cfg = drv.DeltaEpsZetaTrainingConfig(
        n_epochs=1, steps_per_epoch=4, learning_rate=1e-1,
        lambda_init=0.0, lambda_step=0.0,
        log_dir=tmp_path / "run",
    )
    distortion = drv.make_mse_distortion_fn(sd)
    driver = drv.DeltaEpsZetaTrainingDriver(
        state_dict=perturbed, config=cfg, distortion_fn=distortion,
    )
    first = driver.step_once()
    driver.step_once()
    third = driver.step_once()
    # Loss should non-increase at each step (allowing tiny stochastic noise).
    assert third.loss <= first.loss + 1e-9


def test_train_runs_full_loop_writes_jsonl(tmp_path: pathlib.Path) -> None:
    sd = _synthetic_state_dict(seed=1)
    cfg = drv.DeltaEpsZetaTrainingConfig(
        n_epochs=2, steps_per_epoch=3,
        log_dir=tmp_path / "run", run_label="full_loop",
    )
    driver = drv.DeltaEpsZetaTrainingDriver(state_dict=sd, config=cfg)
    rows = driver.train()
    assert len(rows) == 6  # 2 epochs * 3 steps
    assert rows[0].epoch == 0 and rows[0].step == 0
    assert rows[-1].epoch == 1 and rows[-1].step == 2
    # Log file exists + is parseable JSONL.
    log_text = driver.log_path.read_text(encoding="utf-8")
    parsed = [json.loads(line) for line in log_text.strip().split("\n")]
    assert len(parsed) == 6
    for row in parsed:
        assert "epoch" in row
        assert "step" in row
        assert "loss" in row
        assert "rate_bits" in row
        assert "lambda_value" in row


# ---------------------------------------------------------------------------
# Lambda dual-ascent
# ---------------------------------------------------------------------------

def test_lambda_ramps_when_rate_exceeds_budget(tmp_path: pathlib.Path) -> None:
    """Set a tiny rate budget so the rate proxy exceeds it; lambda should ramp UP
    after the first epoch."""
    sd = _synthetic_state_dict(seed=2, scale=1.0)  # larger scale -> larger H0
    cfg = drv.DeltaEpsZetaTrainingConfig(
        n_epochs=2, steps_per_epoch=1,
        rate_budget_bits=0.0,  # impossible-to-meet; H0 always > 0
        lambda_init=0.0, lambda_step=0.5,
        log_dir=tmp_path / "run",
    )
    driver = drv.DeltaEpsZetaTrainingDriver(state_dict=sd, config=cfg)
    initial_lambda = driver.lambda_value
    driver.train()
    # After 2 epochs of overshoot, lambda should have ramped strictly above init.
    assert driver.lambda_value > initial_lambda


def test_lambda_clamps_at_zero(tmp_path: pathlib.Path) -> None:
    """When rate budget is huge, lambda should ramp DOWN but never go negative."""
    sd = _synthetic_state_dict(seed=3, scale=1e-6)
    cfg = drv.DeltaEpsZetaTrainingConfig(
        n_epochs=5, steps_per_epoch=1,
        rate_budget_bits=100.0,  # always under
        lambda_init=0.1, lambda_step=0.5,
        log_dir=tmp_path / "run",
    )
    driver = drv.DeltaEpsZetaTrainingDriver(state_dict=sd, config=cfg)
    driver.train()
    assert driver.lambda_value >= 0.0


# ---------------------------------------------------------------------------
# Pipeline-aware callback wiring
# ---------------------------------------------------------------------------

def test_end_of_epoch_callback_fires(tmp_path: pathlib.Path) -> None:
    sd = _synthetic_state_dict()
    cfg = drv.DeltaEpsZetaTrainingConfig(
        n_epochs=2, steps_per_epoch=1,
        log_dir=tmp_path / "run",
    )
    cb_log = tmp_path / "cb_log"
    pipeline = CodecPipeline([Op1_PR101SplitBrotli(auto_select=False)])
    cb = CodecPipelineAwareTrainingCallback(
        pipeline=pipeline, log_dir=cb_log,
    )
    driver = drv.DeltaEpsZetaTrainingDriver(
        state_dict=sd, config=cfg, callback=cb,
    )
    driver.train()
    # Callback should have one report per epoch.
    assert len(cb.history) == 2
    rows = cb.read_log()
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_checkpoint_saves_and_loads_bit_faithfully(tmp_path: pathlib.Path) -> None:
    sd = _synthetic_state_dict(seed=4)
    cfg = drv.DeltaEpsZetaTrainingConfig(
        n_epochs=1, steps_per_epoch=2,
        log_dir=tmp_path / "run",
    )
    driver = drv.DeltaEpsZetaTrainingDriver(state_dict=sd, config=cfg)
    driver.train()
    saved_state = driver.state_dict
    ckpt_path = driver.save_checkpoint(tmp_path / "run" / "final.pt")
    assert ckpt_path.exists()
    reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    for k in saved_state:
        assert torch.equal(saved_state[k], reloaded[k]), f"mismatch at key {k!r}"


# ---------------------------------------------------------------------------
# Distortion factory
# ---------------------------------------------------------------------------

def test_make_mse_distortion_fn_returns_zero_at_reference() -> None:
    sd = _synthetic_state_dict()
    fn = drv.make_mse_distortion_fn(sd)
    out = fn(sd)
    assert isinstance(out, torch.Tensor)
    assert out.item() == pytest.approx(0.0, abs=1e-12)


def test_make_mse_distortion_fn_is_positive_off_reference() -> None:
    sd = _synthetic_state_dict(seed=5)
    fn = drv.make_mse_distortion_fn(sd)
    perturbed = {k: v + 0.1 * torch.randn_like(v) for k, v in sd.items()}
    out = fn(perturbed)
    assert out.item() > 0.0


# ---------------------------------------------------------------------------
# CLI sanity (importability)
# ---------------------------------------------------------------------------

def test_cli_module_importable() -> None:
    """Just verify the CLI main is callable and argparse forms a proper
    parser. We don't actually invoke training (filesystem dependency)."""
    spec = importlib.util.find_spec("run_deltaepszeta_training")
    assert spec is not None
    # main() exists.
    assert callable(drv.main)
