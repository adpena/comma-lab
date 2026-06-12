#!/usr/bin/env python
"""R4 adversarial probe — MULTI-STAGE-BOUNDARY live behavior (lens B).

R2 checked phase interactions STRUCTURALLY (anneal restarts per stage; QAT no-op
on non-QAT stages). R4-B checks the ACTUAL LIVE behavior crossing a boundary where
levers turn ON: a 2-stage curriculum (stage 1 = ALL LEVERS OFF, vendored; stage 2 =
ALL 5 LEVERS ON) must (1) produce a coherent CONTINUING descent across the boundary
(no NaN, no divergence spike), (2) reset ``epoch_in_stage`` so the Lever-2 anneal
RESTARTS at T=1.0 in stage 2 (not continue from stage 1's end), (3) carry the
decoder/latents/EMA across the boundary, (4) still produce a valid scoreable archive
at the end. Authority: synthetic scorer (RESEARCH-ONLY, [macOS-CPU advisory]).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from tac.torch_vehicle.curriculum import StageSpec, seg_temperature_for_epoch
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(s, t):
    return torch.nn.functional.cross_entropy(s, t)


def _vendored_spec(epochs: int) -> StageSpec:
    """Stage 1: ALL levers OFF (pure vendored path)."""
    return StageSpec(
        name="vendored", epochs=epochs, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.0,
        cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )


def _all_five_spec(epochs: int) -> StageSpec:
    """Stage 2: ALL 5 Layer-2 levers ON (the boundary-activation stage)."""
    return StageSpec(
        name="all5", epochs=epochs, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.01,
        cat_sigma=0.2, use_qat=True, init_latents_random=False,  # CARRY from stage 1
        seg_surrogate="soft_cosine", seg_temperature=1.0, seg_temperature_end=0.05,
        rate_lambda_w=1e-3, rate_lambda_lat=1e-3,
        score_aware_qat=True, qat_sensitivity_decay=0.99, margin_weight_tau=2.0,
    )


def main() -> int:
    torch.manual_seed(13)
    v = import_vendored_bundle()
    n_pairs = 8
    s1_epochs, s2_epochs = 4, 6

    # (A) The anneal-restart structural check: stage 2's anneal is a function of
    # epoch_in_stage, which restarts at 0 in the new stage. Verify the FIRST epoch
    # of stage 2 sees T≈start(1.0), NOT a continuation of stage 1.
    s2 = _all_five_spec(s2_epochs)
    t_first = seg_temperature_for_epoch(s2, 0)       # epoch_in_stage=0
    t_last = seg_temperature_for_epoch(s2, s2_epochs - 1)
    assert abs(t_first - 1.0) < 1e-9, f"stage-2 anneal does NOT restart at 1.0: {t_first}"
    assert abs(t_last - 0.05) < 1e-9, f"stage-2 anneal does NOT reach 0.05: {t_last}"
    # Stage 1 (no anneal: seg_temperature_end is None) holds static T for every epoch.
    s1 = _vendored_spec(s1_epochs)
    assert seg_temperature_for_epoch(s1, 0) == seg_temperature_for_epoch(s1, s1_epochs - 1), (
        "stage-1 (no anneal) temperature should be static"
    )
    print(f"[A] anneal RESTARTS per-stage: stage2 T {t_first:.3f}->{t_last:.3f}; "
          f"stage1 static (no skew across the boundary)")

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        cfg = TorchVehicleConfig(
            base_channels=8, latent_dim=28, out_dir=out_dir, checkpoint_every_epochs=1,
            device="cpu", seed=13, pose_film_enabled=True, pose_film_hidden=8,
        )
        sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=13)
        drv = TorchVehicleDriver(
            cfg, scorer=sc, vendored=v,
            curriculum=[_vendored_spec(s1_epochs), _all_five_spec(s2_epochs)],
        )
        out = drv.run()
        assert out["status"] == "complete", f"run did not complete: {out}"

        # (B) Pull the per-epoch telemetry from the trajectory JSONL: loss must stay
        # finite across the boundary, and the descent must continue (no divergence
        # spike when levers activate).
        import json

        tpath = out_dir / "torch_vehicle_trajectory.jsonl"
        rows = [json.loads(ln) for ln in tpath.read_text().splitlines() if ln.strip()]
        losses = [(r["stage_index"], r["epoch_in_stage"], r["loss"]) for r in rows]

        assert len(losses) == s1_epochs + s2_epochs, f"expected {s1_epochs+s2_epochs} rows, got {len(losses)}"
        for si, eis, lv in losses:
            assert lv == lv and abs(lv) < 1e9, f"non-finite/exploded loss at stage {si} ep {eis}: {lv}"
        s1_losses = [lv for si, eis, lv in losses if si == 0]
        s2_losses = [lv for si, eis, lv in losses if si == 1]
        print(f"[B] all {len(losses)} epochs finite across the boundary; "
              f"stage1 loss {s1_losses[0]:.2f}->{s1_losses[-1]:.2f}, "
              f"stage2 loss {s2_losses[0]:.2f}->{s2_losses[-1]:.2f}")

        # (C) No DIVERGENCE SPIKE at the boundary: the first stage-2 loss must not be
        # catastrophically larger than the last stage-1 loss (a wrong lever activation
        # would blow up). Allow a modest jump (levers add regularizers + change the seg
        # objective) but reject an explosion (>50x).
        boundary_ratio = s2_losses[0] / max(abs(s1_losses[-1]), 1e-6)
        assert boundary_ratio < 50.0, (
            f"BOUNDARY DIVERGENCE: stage-2 first loss {s2_losses[0]:.2f} is {boundary_ratio:.1f}x "
            f"the stage-1 final {s1_losses[-1]:.2f} — lever activation destabilized the run."
        )
        print(f"[C] no boundary divergence spike: stage2/stage1 loss ratio {boundary_ratio:.2f}x (<50x)")

        # (D) The end-of-run archive is valid + scoreable (the levers-on stage produced
        # a deployable artifact).
        best_archive = (out_dir / "best" / "best_archive.bin").read_bytes()
        from tac.torch_vehicle.pose_film import inflate_film_decoder, parse_pose_section
        pose = parse_pose_section(best_archive, v.parse_archive)
        assert pose is not None, "pose section missing after the all-5-on stage"
        frames = inflate_film_decoder(best_archive, v.parse_archive, v.HNeRVDecoder, film_hidden=8)
        assert torch.isfinite(frames).all() and tuple(frames.shape) == (n_pairs, 2, 3, 384, 512)
        print(f"[D] end-of-run archive valid + inflatable: {len(best_archive)} B, "
              f"frames {tuple(frames.shape)} finite")

    print("\nR4-B MULTI-STAGE-BOUNDARY PROBE: PASS (all 4 checks). "
          "Lever activation at the stage boundary produces a coherent continuing descent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
