"""Smoke tests for experiments/train_scpp_self_compression.py.

These tests exercise the trainer in --smoke mode with very short stage
counts. They verify:

* CLI parser argument arity (per catalog #12 preflight_arity)
* CUDA-required default + --device cpu opt-in with banner
* /tmp path refusal (per CLAUDE.md "Forbidden /tmp paths")
* All 5 stages execute end-to-end
* EMA shadow is saved at end
* Archive is built from EMA shadow + decodes back to a valid substrate

The trainer is imported as a module; main() is invoked with argv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Make experiments/ importable
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

# Import via the module path the script supports
import train_scpp_self_compression as trainer  # type: ignore[import-not-found]


def test_parser_includes_required_flags():
    parser = trainer.build_parser()
    actions = {a.dest for a in parser._actions}
    # Must include the canonical flags
    required = {
        "output_dir", "device", "video_path", "target_archive_bytes",
        "stage_1_epochs", "stage_2_epochs", "stage_3_epochs",
        "stage_4_epochs", "stage_5_iters",
        "base_lr", "ema_decay", "rho_pose", "rho_seg", "rho_rate",
        "rho_distill", "latent_dim", "base_channels", "n_pairs",
        "eval_height", "eval_width", "eval_roundtrip",
        "auth_eval_on_best", "smoke", "seed",
    }
    missing = required - actions
    assert not missing, f"Parser missing required flags: {missing}"


def test_parser_device_default_is_cuda():
    parser = trainer.build_parser()
    args = parser.parse_args(
        ["--output-dir", "experiments/results/scpp_test"]
    )
    assert args.device == "cuda"


def test_parser_smoke_flag_is_optional():
    parser = trainer.build_parser()
    args = parser.parse_args(
        ["--output-dir", "experiments/results/scpp_test"]
    )
    assert args.smoke is False
    args2 = parser.parse_args(
        ["--output-dir", "experiments/results/scpp_test", "--smoke"]
    )
    assert args2.smoke is True


def test_assert_cuda_or_cpu_raises_without_cuda_default(monkeypatch):
    """No CUDA available + default --device cuda → RuntimeError per CLAUDE.md."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    parser = trainer.build_parser()
    args = parser.parse_args(["--output-dir", "experiments/results/test"])
    with pytest.raises(RuntimeError, match="CUDA-REQUIRED"):
        trainer.assert_cuda_or_explicit_cpu(args)


def test_assert_cpu_with_banner(monkeypatch, capsys):
    parser = trainer.build_parser()
    args = parser.parse_args(
        ["--output-dir", "experiments/results/test", "--device", "cpu"]
    )
    device = trainer.assert_cuda_or_explicit_cpu(args)
    assert device == torch.device("cpu")
    err = capsys.readouterr().err
    assert "BANNER" in err
    assert "DIFFER" in err


def test_main_refuses_tmp_output_dir(monkeypatch):
    """Per CLAUDE.md "Forbidden /tmp paths"."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        trainer.main([
            "--output-dir", "/tmp/scpp_test",
            "--device", "cpu",
            "--smoke",
        ])


def test_smoke_end_to_end_cpu(tmp_path):
    """Run a tiny smoke training pass on CPU + verify archive + metrics."""
    out_dir = tmp_path / "scpp_smoke"
    rc = trainer.main([
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--smoke",
        "--stage-1-epochs", "2",
        "--stage-2-epochs", "2",
        "--stage-3-epochs", "1",
        "--stage-4-epochs", "1",
        "--stage-5-iters", "3",
        "--n-pairs", "8",
        "--latent-dim", "8",
        "--base-channels", "16",
        "--seed", "0",
    ])
    assert rc == 0

    # Verify artifacts
    archive_path = out_dir / "scpp_substrate.bin"
    ema_path = out_dir / "ema_shadow.pt"
    metrics_path = out_dir / "stage_metrics.json"

    assert archive_path.exists()
    assert ema_path.exists()
    assert metrics_path.exists()

    # All 5 stages should be in metrics
    metrics = json.loads(metrics_path.read_text())
    assert len(metrics) == 5
    stage_names = [m["stage_name"] for m in metrics]
    assert stage_names == [
        "stage_1_anchor",
        "stage_2_finetune_distill",
        "stage_3_joint",
        "stage_4_qat",
        "stage_5_mdl_tto",
    ]
    # Every stage ran at least 1 iteration
    for m in metrics:
        assert m["n_iters"] >= 1


def test_smoke_archive_round_trips_through_decoder(tmp_path):
    """The trained archive must decode back to a valid substrate."""
    out_dir = tmp_path / "scpp_smoke"
    trainer.main([
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--smoke",
        "--stage-1-epochs", "1",
        "--stage-2-epochs", "1",
        "--stage-3-epochs", "1",
        "--stage-4-epochs", "1",
        "--stage-5-iters", "1",
        "--n-pairs", "4",
        "--latent-dim", "8",
        "--base-channels", "16",
        "--seed", "0",
    ])

    archive_path = out_dir / "scpp_substrate.bin"
    archive_bytes = archive_path.read_bytes()

    from tac.scpp_substrate import decode_scpp_substrate
    sd, lat, cfg = decode_scpp_substrate(archive_bytes)
    # Substrate state dict has expected keys (proj + 4 blocks + pair head)
    assert any("proj" in k for k in sd.keys())
    assert any("block1" in k for k in sd.keys())
    assert any("pair_head" in k for k in sd.keys())
    # Latents have the right shape
    assert lat.shape == (cfg.n_pairs, cfg.latent_dim)
    # Config preserves the input
    assert cfg.n_pairs == 4
    assert cfg.latent_dim == 8


def test_score_proxy_refuses_without_eval_roundtrip():
    """score_domain_lagrangian asserts eval_roundtrip_applied=True."""
    rendered = torch.zeros(1, 2, 3, 32, 32)
    target = torch.zeros_like(rendered)
    with pytest.raises(RuntimeError, match="without eval_roundtrip"):
        trainer.score_domain_lagrangian(
            rendered_pairs=rendered,
            target_pairs=target,
            rho_seg=1.0,
            rho_pose=1.0,
            eval_roundtrip_applied=False,
        )


def test_ema_default_decay_is_0_997():
    """Per CLAUDE.md "EMA — NON-NEGOTIABLE": default decay 0.997 for weights."""
    import torch.nn as nn
    model = nn.Linear(4, 8)
    ema = trainer.EMA(model)
    assert ema.decay == 0.997


def test_ema_apply_restores_shadow():
    import torch.nn as nn
    model = nn.Linear(4, 8)
    initial_weight = model.weight.data.clone()
    ema = trainer.EMA(model)

    # Mutate model
    with torch.no_grad():
        model.weight.data.add_(1.0)

    # Apply shadow (should restore close-to-initial via EMA)
    ema.apply(model)
    # EMA is initialised to current state, so apply restores initial state
    assert torch.allclose(model.weight.data, initial_weight)


def test_real_video_batch_source_refuses_missing_video():
    """Outside --smoke, real video required; missing video raises FileNotFoundError."""
    src = trainer.RealVideoBatchSource(
        video_path="/nonexistent/path/video.mkv",
        n_pairs=10,
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        src.get_batch(batch_size=2)


def test_synthetic_batch_source_smoke_only_shape():
    src = trainer._SyntheticBatchSource(n_pairs=4, eval_h=32, eval_w=32)
    batch = src.get_batch(batch_size=2)
    assert batch.shape == (2, 2, 3, 32, 32)
