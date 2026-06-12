#!/usr/bin/env python
"""R6 A/B — MAGNITUDE of the QAT-sensitivity-EMA reset-at-stage-boundary seam.

Probe [3] of probe_r6_integration_multistage.py MEASURED that the Lever-4 sensitivity
EMA is reset-to-empty at a QAT->QAT stage boundary (stage1 END=15 tensors -> stage2
START=0). This A/B MEASURES the MAGNITUDE: does carrying the EMA across the boundary
(so stage 2's QAT starts score-aware instead of uniform-127) produce a measurably
DIFFERENT descent + final archive, or is it below the codec quant-noise floor?

Two identical multi-QAT-stage runs from the same seed:
  A = CURRENT behavior (EMA reset to empty at the QAT->QAT boundary; uniform-127 for
      the first steps of stage 2).
  B = CARRIED behavior (a hooked driver injects the stage1-END EMA into stage2's
      runtime at build time -> score-aware grid from step 0 of stage 2).

Compare: stage2 first-batch quant grids + per-epoch loss trajectory + final archive
bytes. If A != B in a non-trivial way, the reset is a real (MED/HIGH) integration
defect; if A == B (or within quant noise), it is LOW/benign.

Authority: synthetic scorer, RESEARCH-ONLY, [macOS-CPU advisory] NON-PROMOTABLE.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(s, t):
    return torch.nn.functional.cross_entropy(s, t)


def _spec(name: str, epochs: int, *, use_qat: bool, init_random: bool) -> StageSpec:
    return StageSpec(
        name=name, epochs=epochs, seg_loss_fn=_ce, eval_every=1, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0, grad_clip_muon=1.0,
        lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0, cat_lambda=0.01,
        cat_sigma=0.2, use_qat=use_qat, init_latents_random=init_random,
        seg_surrogate="soft_cosine", seg_temperature=1.0, seg_temperature_end=0.05,
        rate_lambda_w=1e-3, rate_lambda_lat=1e-3,
        score_aware_qat=True, qat_sensitivity_decay=0.99, margin_weight_tau=2.0,
    )


def _curriculum() -> list[StageSpec]:
    return [
        _spec("pre_qat", 3, use_qat=False, init_random=True),
        _spec("qat_a", 4, use_qat=True, init_random=False),
        _spec("qat_b", 5, use_qat=True, init_random=False),
    ]


def _run(out_dir: Path, n_pairs: int, *, carry_ema: bool) -> dict:
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=out_dir, checkpoint_every_epochs=1,
        device="cpu", seed=41, pose_film_enabled=True, pose_film_hidden=8,
    )
    sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=41)
    v = import_vendored_bundle()
    drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=_curriculum())

    if carry_ema:
        # Hook: copy the PRIOR stage's END sensitivity EMA into the NEW stage runtime
        # at build time (the candidate FIX behavior — EMA carries like the weight EMA).
        orig_build = drv._build_stage_runtime
        prev_holder: dict[str, object] = {}

        def hooked_build(spec, **kw):
            rt = orig_build(spec, **kw)
            if spec.use_qat and "rt" in prev_holder:
                prev = prev_holder["rt"]
                prior = dict(prev.tensor_sensitivity_ema)  # type: ignore[attr-defined]
                if prior:
                    rt.tensor_sensitivity_ema.update(prior)
            prev_holder["rt"] = rt
            return rt

        drv._build_stage_runtime = hooked_build  # type: ignore[method-assign]

    out = drv.run()
    assert out["status"] == "complete", out
    return out


def _losses(out_dir: Path) -> list[tuple[int, int, float]]:
    tpath = out_dir / "torch_vehicle_trajectory.jsonl"
    rows = [json.loads(ln) for ln in tpath.read_text().splitlines() if ln.strip()]
    return [(r["stage_index"], r["epoch_in_stage"], r["loss"]) for r in rows]


def main() -> int:
    n_pairs = 8
    with tempfile.TemporaryDirectory() as td:
        a_dir = Path(td) / "A_reset"
        b_dir = Path(td) / "B_carry"
        _run(a_dir, n_pairs, carry_ema=False)
        _run(b_dir, n_pairs, carry_ema=True)

        a_arch = (a_dir / "best" / "best_archive.bin").read_bytes()
        b_arch = (b_dir / "best" / "best_archive.bin").read_bytes()
        a_loss = _losses(a_dir)
        b_loss = _losses(b_dir)

        # Stage 2 (qat_b, index 2) early-epoch loss divergence is the seam signal.
        a_s2 = [lv for si, _, lv in a_loss if si == 2]
        b_s2 = [lv for si, _, lv in b_loss if si == 2]
        early_delta = abs(a_s2[0] - b_s2[0])
        max_delta = max(abs(x - y) for x, y in zip(a_s2, b_s2, strict=True))

        bytes_equal = a_arch == b_arch
        print(f"A (reset)  final archive: {len(a_arch)} B")
        print(f"B (carry)  final archive: {len(b_arch)} B")
        print(f"archive bytes IDENTICAL A==B: {bytes_equal}  (delta {len(b_arch)-len(a_arch):+d} B)")
        print(f"stage2(qat_b) first-epoch loss: A={a_s2[0]:.6f}  B={b_s2[0]:.6f}  |delta|={early_delta:.6e}")
        print(f"stage2(qat_b) max per-epoch |loss delta| over the stage: {max_delta:.6e}")

        # Verdict signal: if the trajectory or the archive differs, the reset is a real
        # behavioral seam (the EMA carry changes the descent). If identical, benign.
        if bytes_equal and max_delta < 1e-9:
            print("\nSEAM MAGNITUDE: BENIGN — reset vs carry produce a BIT-IDENTICAL "
                  "run (the EMA reset has no behavioral effect at this operating point).")
        else:
            print("\nSEAM MAGNITUDE: BEHAVIORAL — reset vs carry produce a DIFFERENT "
                  f"descent (stage2 loss delta {max_delta:.2e}, archive "
                  f"{'differs' if not bytes_equal else 'same'}). The QAT->QAT EMA "
                  "reset is a real integration defect (same class as R2's resume "
                  "defect, manifesting at the normal stage boundary).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
