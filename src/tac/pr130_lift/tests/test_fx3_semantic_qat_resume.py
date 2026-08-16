from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from tac.pr130_lift import train_semantic_quantized_resumable as trainer
from tac.training import EMA


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        challenge_root=tmp_path / "upstream",
        cache=None,
        input_cache=tmp_path / "input.pt",
        target_cache=tmp_path / "target.pt",
        master_cache=None,
        distill_weight=0.0,
        distill_max_seg=4e-4,
        init=tmp_path / "init.pt",
        bits=4,
        steps=6,
        batch_size=2,
        eval_batch_size=2,
        eval_every=2,
        checkpoint_every=1,
        resume_from=None,
        lr=0.01,
        float_warmup_steps=1,
        ce_fraction=0.5,
        softplus_fraction=0.8,
        ema_target_seed_fraction=0.01,
        parity_pairs=1,
        smoke_pairs=None,
        seed=123,
        device="cpu",
        disable_tf32=False,
        fixed_zero_mask=False,
        out=tmp_path / "out.json",
        save=tmp_path / "save.pt",
    )


def test_ema_decay_is_lawref_resolved_and_fallback_is_absence_only() -> None:
    policy = trainer.resolve_ema_policy(6_000)
    assert policy["equation_id"] == "ema_decay_run_geometry_v1"
    assert policy["decay"] == pytest.approx(0.01 ** (1.0 / 6_000))
    assert policy["fallback_used"] is False
    assert not {
        "resolved_at",
        "resolved_at_utc",
    } & policy["resolved_manifest"].keys()
    fallback = trainer.resolve_ema_policy(None)
    assert fallback["decay"] == 0.997
    assert fallback["fallback_used"] is True
    with pytest.raises(ValueError, match="positive"):
        trainer.resolve_ema_policy(0)


def test_ema_eval_scope_restores_live_weights_and_training_mode() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(5.0)
    live = model.weight.detach().clone()
    model.train()
    with trainer.ema_eval_scope(model, ema):
        assert torch.equal(model.weight, torch.full_like(model.weight, 2.0))
        assert model.training is False
    assert torch.equal(model.weight, live)
    assert model.training is True


def test_phase_boundaries_preserve_every_actual_stage(tmp_path: Path) -> None:
    args = _args(tmp_path)
    assert trainer.phase_end_steps(args) == {
        1: "float_ce",
        3: "ce",
        5: "softplus_margin",
        6: "expected_flip",
    }


def _step_linear(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: EMA,
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()
    ema.update(model)
    scheduler.step()


def test_full_state_resume_matches_uninterrupted_trajectory(tmp_path: Path) -> None:
    args = _args(tmp_path)
    architecture = {
        "width": 1,
        "blocks": 1,
        "frame_dim": 1,
        "num_pairs": 600,
        "num_tokens": 5,
        "phase_y": 1,
        "phase_x": 1,
        "temporal_radius": 0,
    }
    artifacts = {
        "init": {"path": "init.pt", "bytes": 1, "sha256": "a" * 64},
        "input_cache": {"path": "input.pt", "bytes": 1, "sha256": "b" * 64},
        "target_cache": {"path": "target.pt", "bytes": 1, "sha256": "c" * 64},
    }
    ema_policy = trainer.resolve_ema_policy(args.steps)
    torch.manual_seed(args.seed)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    ema = EMA(model, decay=ema_policy["decay"], warmup=True)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(6, generator=generator)
    x = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    y = torch.tensor([[0.25], [-0.5]])
    for _ in range(3):
        _step_linear(model, optimizer, scheduler, ema, x, y)
    best_state = ema.state_dict()
    payload = trainer._checkpoint_payload(
        args=args,
        architecture_config=architecture,
        artifacts=artifacts,
        ema_policy=ema_policy,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        order=order,
        cursor=2,
        step=3,
        phase="ce",
        checkpoint_kind="periodic",
        history=[{"step": 2}],
        best_key=(0.1,),
        best_seg=0.1,
        # ddm_av3 F3: which step actually won, and what the init measured.  Both
        # are REQUIRED (never defaulted) -- a checkpoint that cannot say whether
        # its "best" is its own input is exactly the defect these fields close.
        best_step=2,
        init_seg=0.2,
        best_rgb=None,
        best_state=best_state,
        deployment_state=best_state,
    )
    assert payload["training_state"]["best_step"] == 2
    assert payload["training_state"]["init_seg"] == 0.2
    checkpoint = tmp_path / "resume.pt"
    trainer._atomic_torch_save(payload, checkpoint)
    assert checkpoint.exists()
    assert not list(tmp_path.glob("*.tmp"))

    for _ in range(3):
        _step_linear(model, optimizer, scheduler, ema, x, y)
    uninterrupted_live = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    uninterrupted_ema = ema.state_dict()
    uninterrupted_lr = scheduler.get_last_lr()

    initial_state = {
        key: value.detach().clone() for key, value in payload["training_state"]["model_state_dict"].items()
    }
    resumed_model = torch.nn.Linear(2, 1)
    resumed_model.load_state_dict(initial_state)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=args.lr, weight_decay=0.0
    )
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer, T_max=args.steps
    )
    resumed_ema = EMA(resumed_model, decay=ema_policy["decay"], warmup=True)
    resumed_generator = torch.Generator().manual_seed(args.seed)
    restored = trainer._load_resume_state(
        checkpoint,
        args=args,
        architecture_config=architecture,
        artifacts=artifacts,
        ema_policy=ema_policy,
        model=resumed_model,
        ema=resumed_ema,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        generator=resumed_generator,
        device=torch.device("cpu"),
    )
    assert restored["start_step"] == 4
    assert torch.equal(restored["order"], order)
    assert restored["cursor"] == 2
    for _ in range(3):
        _step_linear(
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            resumed_ema,
            x,
            y,
        )
    for key, value in uninterrupted_live.items():
        assert torch.equal(resumed_model.state_dict()[key], value)
    for key, value in uninterrupted_ema.items():
        assert torch.equal(resumed_ema.state_dict()[key], value)
    assert resumed_scheduler.get_last_lr() == uninterrupted_lr


def test_deployed_int4_path_is_argmax_identical_for_ema_shadow() -> None:
    qat = trainer._load_lifted_qat()
    torch.manual_seed(19)
    architecture = {
        "width": 8,
        "blocks": 1,
        "frame_dim": 4,
        "num_pairs": 600,
        "num_tokens": 5,
        "phase_y": 1,
        "phase_x": 1,
        "temporal_radius": 0,
    }
    model = qat.SemanticTokenRenderer(**architecture)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.add_(0.25)
    segnet = torch.nn.Conv2d(3, 5, kernel_size=1).eval()
    tokens = torch.randint(0, 5, (600, 8, 8), dtype=torch.long)
    parity = trainer.deployed_argmax_parity(
        qat=qat,
        model=model,
        segnet=segnet,
        conditioning_tokens=tokens,
        bits=4,
        pair_ids=[17],
        batch_size=1,
        architecture_config=architecture,
        deployment_state=model.state_dict(),
        device=torch.device("cpu"),
    )
    assert parity["passed"] is True, f"parity={parity!r}"
    assert parity["argmax_diff_pixels"] == 0
    assert parity["frame_max_abs_delta"] == 0.0
    assert parity["semantic_blob_bytes"] > 0
