# SPDX-License-Identifier: MIT
"""R5 adversarial review probe — determinism / seed-reproducibility (lens A) +
optimizer(Muon)×lever interaction (lens B) for the 5 Layer-2 levers.

R1-R4 covered: static NO-FAKE (R1), runtime/resume (R2), gradient-direction +
double-counting (R3), deployed-archive + multi-stage-boundary (R4). NONE ran a
TWO-FRESH-RUNS-FROM-SAME-SEED determinism check with the LEVERS ON (the existing
``test_all_default_driver_run_is_deterministic_and_byte_identical`` is all-DEFAULT,
``use_muon=False``; R4 only checked all-5-on+Muon for *no-NaN*, NOT for determinism).

This probe MEASURES, on the REAL synthetic-scorer driver:

  A1. all-5-levers-ON (AdamW) — two fresh runs, same seed → bit-identical archive?
  A2. all-5-levers-ON + MUON  — two fresh runs, same seed → bit-identical archive?
       (Muon's Newton-Schulz orthogonalization is the new nondeterminism surface
        the lever paths could perturb.)
  B1. Muon partition covers EVERY FiLM param under all-5-on (0 dropped) + the
       rate-gradient + QAT both flow through BOTH param groups (no mis-route).
  C1. the QAT ``_rank_normalize`` argsort tie-break is deterministic on TIED
       sensitivities (the one ``argsort``-on-possible-ties path in the levers).

Authority: every number is ``[macOS-CPU advisory]`` NON-PROMOTABLE (synthetic
scorer, RESEARCH-ONLY). No GPU, no daemon, no Cool-Chic touched.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(seg_logits, targets_hard):
    return torch.nn.functional.cross_entropy(seg_logits, targets_hard)


def _all_five_spec(*, use_muon: bool, epochs: int = 3) -> StageSpec:
    """ALL FIVE levers ON (matches the compose-all-five test config + the
    ``use_muon`` toggle the existing tests never exercise with levers on)."""
    return StageSpec(
        name="r5_all_five",
        epochs=epochs,
        seg_loss_fn=_ce,
        eval_every=1,
        batch_size=3,
        ema_decay=0.999,
        use_muon=use_muon,
        adamw_lr=(1e-5 if use_muon else 1e-3),
        muon_lr=2e-4,
        muon_weight_decay=0.0,
        latent_lr_mult=10.0,
        grad_clip=1e9,
        grad_clip_muon=1e9,
        lr_floor_ratio=5e-6,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.01,
        cat_sigma=0.2,
        use_qat=True,
        init_latents_random=True,
        # ---- the 5 levers ----
        rate_lambda_w=0.05,
        rate_lambda_lat=0.02,
        seg_surrogate="soft_cosine",
        seg_temperature=1.0,
        seg_temperature_end=0.2,
        score_aware_qat=True,
        qat_sensitivity_decay=0.9,
        margin_weight_tau=2.0,
    )


def _run(out: Path, *, use_muon: bool, n_pairs: int = 6, seed: int = 0) -> bytes:
    cfg = TorchVehicleConfig(
        base_channels=20,
        latent_dim=28,
        out_dir=out,
        checkpoint_every_epochs=1,
        device="cpu",
        seed=seed,
        pose_film_enabled=True,
        pose_film_hidden=8,
    )
    sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=seed)
    driver = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(),
        curriculum=[_all_five_spec(use_muon=use_muon)],
    )
    summary = driver.run()
    assert summary["status"] == "complete", f"run did not complete: {summary}"
    arch = (out / "best" / "best_archive.bin").read_bytes()
    return arch


def probe_a1_all_five_adamw_determinism() -> bool:
    """A1: two fresh all-5-on (AdamW) runs, same seed → bit-identical archive."""
    with tempfile.TemporaryDirectory(prefix="r5_a1_") as td:
        root = Path(td)
        a = _run(root / "a", use_muon=False)
        b = _run(root / "b", use_muon=False)
    ok = a == b
    print(f"[A1] all-5-on AdamW two-run determinism: "
          f"{'BIT-IDENTICAL' if ok else 'DIVERGED'} (a={len(a)}B b={len(b)}B)")
    return ok


def probe_a2_all_five_muon_determinism() -> bool:
    """A2: two fresh all-5-on + MUON runs, same seed → bit-identical archive.
    Muon's Newton-Schulz orthogonalization is the new surface; if a lever
    gradient perturbs it nondeterministically this diverges."""
    with tempfile.TemporaryDirectory(prefix="r5_a2_") as td:
        root = Path(td)
        a = _run(root / "a", use_muon=True)
        b = _run(root / "b", use_muon=True)
    ok = a == b
    print(f"[A2] all-5-on + MUON two-run determinism: "
          f"{'BIT-IDENTICAL' if ok else 'DIVERGED'} (a={len(a)}B b={len(b)}B)")
    return ok


def probe_b1_muon_partition_covers_film_and_routes_grads() -> bool:
    """B1: under all-5-on + Muon, (i) the Muon/AdamW partition covers EVERY
    trainable param of the FiLM-wrapped decoder (0 dropped), and (ii) after a
    real all-5-on backward, the rate-gradient + QAT-sensitivity gradient reach
    BOTH param groups (Muon-routed 2D weights AND AdamW-routed biases/1D)."""
    import torch.nn.functional as F

    v = import_vendored_bundle()
    sc = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=Path(tempfile.mkdtemp(prefix="r5_b1_")),
        device="cpu", seed=0, pose_film_enabled=True, pose_film_hidden=8,
    )
    spec = _all_five_spec(use_muon=True)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])
    torch.manual_seed(31)
    decoder = driver._new_decoder(device=torch.device("cpu"))
    decoder.set_stored_pose(sc.pose_targets[:6])

    # (i) partition coverage — every trainable param accounted for, 0 uncovered.
    muon_params, adamw_params = v.partition_params_for_muon(decoder)
    muon_ids = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    all_trainable = [p for p in decoder.parameters() if p.requires_grad]
    covered = sum(1 for p in all_trainable if id(p) in muon_ids or id(p) in adamw_ids)
    overlap = muon_ids & adamw_ids
    n_total = len(all_trainable)
    film_params = [
        (n, p) for n, p in decoder.named_parameters() if n.startswith("pose_film.")
    ]
    film_covered = sum(
        1 for _, p in film_params if id(p) in muon_ids or id(p) in adamw_ids
    )
    cover_ok = (covered == n_total) and (len(overlap) == 0) and (
        film_covered == len(film_params)
    )

    # (ii) real all-5-on backward → grads reach BOTH groups.
    latents = torch.nn.Parameter(torch.randn(6, 28) * 0.1)
    idx = torch.arange(6)
    # forward through the score-aware QAT fake-quant (Lever 4) like the driver does.
    from tac.torch_vehicle.score_aware_qat import (
        apply_score_aware_qat,
        restore_score_aware_qat,
    )
    originals = apply_score_aware_qat(decoder, None)  # empty-EMA = uniform-127
    decoded_pair = decoder(latents[idx], idx)
    restore_score_aware_qat(decoder, originals)
    flat = decoded_pair.reshape(12, 3, 384, 512)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(6, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    dc = decoded_bhwc.clamp(0, 255)
    decoded_bhwc = dc + (dc.round() - dc).detach()
    seg_out, pose_pred6 = sc.seg_pose_forward(decoded_bhwc)
    from tac.torch_vehicle.driver import _seg_loss_for_spec
    seg_l = _seg_loss_for_spec(spec, seg_out, sc.seg_targets_hard[idx], temperature=1.0)
    pose_l = torch.sqrt(10.0 * F.mse_loss(pose_pred6, sc.pose_targets[idx]) + 1e-12)
    loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
    # + the rate regularizer (Lever 1) — backprops straight to weights/latents.
    reg = driver._weight_regularizers(decoder, latents, spec)
    if reg is not None:
        loss = loss + reg
    loss.backward()

    muon_with_grad = sum(1 for p in muon_params if p.grad is not None)
    adamw_with_grad = sum(
        1 for p in adamw_params if p.grad is not None and p.grad.abs().sum() >= 0
    )
    # FiLM fc2 is zero-init → its render contribution is zero at init, but the rate
    # surrogate (Lever 1) regularizes the FiLM weights too, so fc1 (2D → Muon) MUST
    # carry a gradient. Confirm at least one FiLM weight in the Muon group has grad.
    film_muon_grad = any(
        p.grad is not None and id(p) in muon_ids for _, p in film_params if p.ndim == 2
    )
    grads_ok = (muon_with_grad == len(muon_params)) and (
        adamw_with_grad == len(adamw_params)
    ) and film_muon_grad

    ok = cover_ok and grads_ok
    print(
        f"[B1] Muon×lever partition+routing: "
        f"{'PASS' if ok else 'FAIL'} "
        f"(covered {covered}/{n_total}, overlap={len(overlap)}, "
        f"FiLM covered {film_covered}/{len(film_params)}, "
        f"Muon-grad {muon_with_grad}/{len(muon_params)}, "
        f"AdamW-grad {adamw_with_grad}/{len(adamw_params)}, "
        f"FiLM-fc1-Muon-grad={film_muon_grad})"
    )
    return ok


def probe_c1_qat_rank_normalize_tie_break_deterministic() -> bool:
    """C1: ``_rank_normalize`` uses ``torch.argsort`` — on EXACTLY-TIED
    sensitivities the tie-break is the one possibly-nondeterministic op in the
    lever paths. Verify it is stable across repeated calls AND that the per-tensor
    level MAP is identical run-to-run on a tied sensitivity dict (so the QAT grid
    — hence the archive — does not silently change on a tie)."""
    from tac.torch_vehicle.score_aware_qat import (
        _rank_normalize,
        per_tensor_levels_from_sensitivity,
    )

    # All-tied vector → must collapse to all-0.5 (the uniform fallback), every call.
    tied = torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    r1 = _rank_normalize(tied)
    r2 = _rank_normalize(tied)
    tied_ok = torch.equal(r1, r2) and bool((r1 == 0.5).all().item())

    # PARTIALLY-tied dict (two tensors share a value) → the level map must be
    # identical across repeated calls (argsort tie-break stable run-to-run).
    names = ["blocks.0", "blocks.1", "blocks.2", "blocks.3"]
    sens = {"blocks.0": 1.0, "blocks.1": 2.0, "blocks.2": 2.0, "blocks.3": 9.0}
    maps = [per_tensor_levels_from_sensitivity(dict(sens), names) for _ in range(8)]
    map_ok = all(m == maps[0] for m in maps)

    ok = tied_ok and map_ok
    print(
        f"[C1] QAT rank-normalize tie-break determinism: "
        f"{'STABLE' if ok else 'UNSTABLE'} "
        f"(all-tied→uniform={tied_ok}, partial-tie map stable over 8 calls={map_ok})"
    )
    if not map_ok:
        print(f"     map0={maps[0]}  divergent={[m for m in maps if m != maps[0]]}")
    return ok


def main() -> int:
    print("=== R5 determinism (lens A) + Muon×lever (lens B) + tie-break (lens C) ===")
    results = {
        "A1_adamw_determinism": probe_a1_all_five_adamw_determinism(),
        "A2_muon_determinism": probe_a2_all_five_muon_determinism(),
        "B1_muon_partition_routing": probe_b1_muon_partition_covers_film_and_routes_grads(),
        "C1_qat_tie_break": probe_c1_qat_rank_normalize_tie_break_deterministic(),
    }
    print("\n--- SUMMARY ---")
    all_ok = True
    for k, vok in results.items():
        print(f"  {k:30s} {'PASS' if vok else 'FAIL'}")
        all_ok = all_ok and vok
    print(f"\nOVERALL: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
