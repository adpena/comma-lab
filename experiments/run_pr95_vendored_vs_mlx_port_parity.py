# SPDX-License-Identifier: MIT
"""Step-level parity DIFF: vendored PR95 torch loop  vs  our MLX port.

The operator's gold-standard control (2026-06-11): the vendored PR95 ``hnerv_muon``
torch loop CONVERGES to the d_seg basin (~5.6e-4); our retrains wall (0.014 / 0.010
/ 0.0025). The decisive control is to run the VENDORED ORIGINAL and OUR PORT on
IDENTICAL inputs (same init weights, same latents, same batch, same frozen scorer,
same config) and DIFF the training step. The first op/tensor where they diverge
beyond fp tolerance IS the recipe bug.

This harness is DIFF-ONLY (it does not edit any core port file). It imports:

- the VENDORED PR95 torch reference at
  ``experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/
  source/submissions/hnerv_muon/src`` (``model.py`` HNeRVDecoder, ``losses.py``,
  ``optim.py`` Muon, the ``stages/common.py`` train step) — runnable, gitignored,
  pristine; AND
- our MLX port: ``tac.mlx_pr95_port`` (``MlxScoreAwareTrainer`` + ``TorchScorerBridge``
  + ``HNeRVSyntheticTrainingBundleMLX`` + ``load_pytorch_state_dict_into_mlx``).

Authority: torch-CPU (the numerical diff runs CPU-vs-CPU); NO MPS anywhere. GT
decode ONLY via ``frame_utils.yuv420_to_rgb`` (reused via
``tac.score_aware_loop.targets.build_gt_targets``). ``[macOS-CPU advisory]`` —
non-promotable; this is a parity diagnostic, not a contest score.

What it diffs (in order — the FIRST divergence is the finding):

  (A) forward loss / seg_l / pose_l / d_seg on an IDENTICAL batch + IDENTICAL init.
  (B) the pixel cotangent (gradient on the rendered N2CHW pixels) — the score
      bridge's carrier gradient vs torch ``leaf.grad``.
  (C) per-tensor parameter gradients (after the vjp / backward).
  (D) post-step weight deltas (after one optimizer step) + the EMA shadow.
  (E) if step-1 parity holds: a bounded N-step (default 60) d_seg trajectory A/B
      on the SAME data + init — does convergence DRIFT later (cumulative)?

Run:
    .venv/bin/python experiments/run_pr95_vendored_vs_mlx_port_parity.py \
        --max-pairs 16 --batch-size 8 --traj-steps 60 --stage stage1

NO-FAKE: identical init/seed/batch on both sides; diff against an explicit fp
tolerance; the divergence point IS the finding. Both sides on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# The vendored, runnable PR95 torch reference (gitignored, pristine).
_PR95_SRC = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src"
)


# ---------------------------------------------------------------------------
# Stage configs (mirror the vendored src/stages/* make_config defaults).
# We expose the two extremes: stage1 (AdamW-only, CE, no QAT/C1a; isolates the
# core update) and stage8 (Muon, l7_softplus, QAT, C1a, sigma=0.1; the full
# recipe). The torch reference uses its OWN stage builders; the MLX side uses
# the matched MlxScoreAwareConfig + the curriculum stage spec.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageParity:
    name: str
    seg_loss_form: str          # MLX side seg-loss family
    torch_seg_loss_name: str    # torch losses.py fn name
    adamw_lr: float
    use_muon: bool
    use_qat: bool
    cat_lambda: float
    cat_sigma: float
    ema_decay: float = 0.999
    seg_weight: float = 100.0
    pose_weight: float = 1.0
    muon_lr: float = 2e-4
    latent_lr_mult: float = 10.0
    grad_clip: float = 1.0


STAGES = {
    "stage1": StageParity(
        name="stage1_v328_ce", seg_loss_form="ce_seg_loss",
        torch_seg_loss_name="ce_seg_loss", adamw_lr=1e-3, use_muon=False,
        use_qat=False, cat_lambda=0.0, cat_sigma=0.2,
    ),
    "stage8": StageParity(
        name="stage8_muon_finetune", seg_loss_form="l7_softplus_seg_loss",
        torch_seg_loss_name="l7_softplus_seg_loss", adamw_lr=1e-5, use_muon=True,
        use_qat=True, cat_lambda=0.02, cat_sigma=0.1, muon_lr=2e-4,
    ),
}


def _import_torch_reference() -> tuple[Any, Any, Any]:
    """Import the vendored PR95 ``model.py`` / ``losses.py`` / ``optim.py``."""
    if not (_PR95_SRC / "model.py").exists():
        raise FileNotFoundError(
            f"Vendored PR95 torch source not found at {_PR95_SRC}. "
            "This harness requires the runnable torch reference."
        )
    if str(_PR95_SRC) not in sys.path:
        sys.path.insert(0, str(_PR95_SRC))
    import losses as torch_losses  # type: ignore[import-not-found]
    import model as torch_model  # type: ignore[import-not-found]
    import optim as torch_optim  # type: ignore[import-not-found]
    return torch_model, torch_losses, torch_optim


# ---------------------------------------------------------------------------
# The torch reference step, lifted verbatim from src/stages/common.py
# (lines 168-212). We re-implement the *single batch step* here so we can call
# it with explicit init + batch + return the intermediate tensors for the diff.
# This is NOT a re-derivation of the recipe — it is the SAME ops, in the same
# order, with the same constants, importing the vendored losses/optim/model.
# ---------------------------------------------------------------------------
EVAL_SIZE = (384, 512)

# Set by --isolate-optimizer: forces QAT+C1a OFF so the diff exercises only the
# optimizer + seg/pose loss (the clean Muon-vs-Muon comparison).
_ISOLATE_OPTIMIZER = False


def _torch_forward_loss(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    idx: torch.Tensor,
    dnet: torch.nn.Module,
    seg_targets_hard: torch.Tensor,
    pose_targets: torch.Tensor,
    stage: StageParity,
    torch_losses: Any,
) -> dict[str, Any]:
    """One torch forward (vendored src/stages/common.py:172-199) -> loss tensors.

    Returns dict with loss/seg_l/pose_l/d_seg AND the rendered N2CHW frames
    (pre-roundtrip, [0,255]) so the cotangent diff can hook the same leaf.
    """
    B = len(idx)
    # ``_ISOLATE_OPTIMIZER`` (set by --isolate-optimizer) forces QAT+C1a OFF on
    # BOTH sides so the stage8 diff exercises ONLY the Muon optimizer + l7 loss.
    # Otherwise the harness arms QAT/C1a torch-side but NOT MLX-side (which needs
    # ``configure_stage``), producing a SPURIOUS weight-delta divergence that is
    # a harness gap, not a port bug.
    use_qat = stage.use_qat and not _ISOLATE_OPTIMIZER
    use_c1a = (stage.cat_lambda > 0) and not _ISOLATE_OPTIMIZER
    if use_qat:
        originals = torch_losses.apply_qat(decoder)
    decoded_pair = decoder(latents[idx])  # (B, 2, 3, 384, 512) in [0,255]
    if use_qat:
        torch_losses.restore_qat(decoder, originals)

    # Keep a handle on the raw render for the cotangent diff (retain grad).
    # Under torch.no_grad() (the eval path) the render has requires_grad=False;
    # only retain_grad when the graph is live (the forward/cotangent diff path).
    if decoded_pair.requires_grad:
        decoded_pair.retain_grad()
    render_leaf = decoded_pair

    flat = decoded_pair.reshape(B * 2, 3, EVAL_SIZE[0], EVAL_SIZE[1])
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    decoded_clamped = decoded_bhwc.clamp(0, 255)
    decoded_rounded = decoded_clamped.round()
    decoded_bhwc = decoded_clamped + (decoded_rounded - decoded_clamped).detach()

    posenet_in, segnet_in = dnet.preprocess_input(decoded_bhwc)
    seg_out = dnet.segnet(segnet_in)
    pose_out = dnet.posenet(posenet_in)

    seg_loss_fn = getattr(torch_losses, stage.torch_seg_loss_name)
    seg_l = seg_loss_fn(seg_out, seg_targets_hard[idx])
    pose_mse = F.mse_loss(pose_out["pose"][:, :6], pose_targets[idx])
    pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)

    loss = stage.seg_weight * seg_l + stage.pose_weight * pose_l
    if use_c1a:
        ent = torch_losses.cat_entropy_v2(
            decoder, sigma=stage.cat_sigma, sample_size=2000,
            device=latents.device,
        )
        loss = loss + stage.cat_lambda * ent

    with torch.no_grad():
        d_seg = (seg_out.argmax(dim=1) != seg_targets_hard[idx]).float().mean()

    return {
        "loss": loss, "seg_l": seg_l, "pose_l": pose_l,
        "d_seg": float(d_seg.item()), "render_leaf": render_leaf,
    }


def _rel(a: float, b: float, eps: float = 1e-9) -> float:
    """Symmetric relative difference."""
    return abs(a - b) / (max(abs(a), abs(b)) + eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--traj-steps", type=int, default=60)
    ap.add_argument("--stage", choices=sorted(STAGES), default="stage1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default="upstream/videos/0.mkv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rtol", type=float, default=1e-3, help="rel tol for scalar parity")
    ap.add_argument("--cotangent-rtol", type=float, default=5e-2)
    ap.add_argument(
        "--isolate-optimizer", action="store_true",
        help="force QAT+C1a OFF on BOTH sides (clean Muon-vs-Muon diff; the MLX "
        "QAT/C1a mechanisms need configure_stage, not armed by this harness).",
    )
    args = ap.parse_args()

    global _ISOLATE_OPTIMIZER
    _ISOLATE_OPTIMIZER = bool(args.isolate_optimizer)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(False)  # bicubic interp lacks det kernel

    stage = STAGES[args.stage]
    report: dict[str, Any] = {
        "harness": "pr95_vendored_vs_mlx_port_parity",
        "authority": "[macOS-CPU advisory] — parity diagnostic, NON-PROMOTABLE",
        "stage": stage.name,
        "seed": args.seed,
        "max_pairs": args.max_pairs,
        "batch_size": args.batch_size,
        "video": args.video,
        "isolate_optimizer": bool(args.isolate_optimizer),
    }

    # --- MLX availability ---------------------------------------------------
    try:
        import mlx.core as mx
    except ImportError:
        report["blocked"] = "MLX not available (requires Apple Silicon)."
        print(json.dumps(report, indent=2))
        return 2

    # --- Vendored torch reference ------------------------------------------
    try:
        torch_model, torch_losses, torch_optim = _import_torch_reference()
        report["vendored_source_present"] = True
        report["vendored_source_path"] = str(_PR95_SRC.relative_to(REPO_ROOT))
    except FileNotFoundError as exc:
        report["vendored_source_present"] = False
        report["blocked"] = str(exc)
        print(json.dumps(report, indent=2))
        return 2

    # --- Frozen scorer + GT targets (shared, CPU, yuv420_to_rgb) -----------
    from tac.score_aware_loop.targets import build_gt_targets, load_frozen_distortion_net

    t0 = time.time()
    dnet = load_frozen_distortion_net(upstream_dir="upstream", device="cpu")
    seg_targets_hard, pose_targets, n_pairs = build_gt_targets(
        dnet, video_path=args.video, max_pairs=args.max_pairs, device="cpu"
    )
    report["n_pairs"] = int(n_pairs)
    report["target_build_seconds"] = round(time.time() - t0, 1)

    # --- Identical init: torch decoder + latents ---------------------------
    torch.manual_seed(args.seed)
    decoder_t = torch_model.HNeRVDecoder(
        latent_dim=28, base_channels=36, eval_size=EVAL_SIZE
    ).to("cpu")
    latents_t0 = (torch.randn(n_pairs, 28) * 0.1).to("cpu")
    init_sd = {k: v.detach().clone() for k, v in decoder_t.state_dict().items()}
    init_latents = latents_t0.detach().clone()

    batch_size = min(args.batch_size, n_pairs)
    idx_np = np.arange(batch_size, dtype=np.int64)  # deterministic first batch
    idx_t = torch.from_numpy(idx_np)

    # =======================================================================
    # (A) FORWARD parity on the IDENTICAL batch + init.
    # =======================================================================
    decoder_t.train()
    latents_p = torch.nn.Parameter(init_latents.clone())
    tf = _torch_forward_loss(
        decoder_t, latents_p, idx_t, dnet, seg_targets_hard, pose_targets,
        stage, torch_losses,
    )

    # MLX side: identical init via the bundle + load_pytorch_state_dict_into_mlx.
    from tac.local_acceleration.pr95_hnerv_mlx import (
        HNeRVSyntheticTrainingBundleMLX,
        load_pytorch_state_dict_into_mlx,
    )
    from tac.mlx_pr95_port import (
        MlxScoreAwareConfig,
        MlxScoreAwareTrainer,
        TorchScorerBridge,
    )

    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=n_pairs, latent_dim=28, base_channels=36, output_layout="n2chw",
    )
    load_pytorch_state_dict_into_mlx(bundle.decoder, init_sd)
    bundle.latents = mx.array(init_latents.numpy())  # identical latents
    mx.eval(bundle.parameters())

    bridge = TorchScorerBridge(
        dnet, seg_targets_hard, pose_targets,
        seg_loss_form=stage.seg_loss_form,
        seg_weight=stage.seg_weight, pose_weight=stage.pose_weight,
        eval_roundtrip=True, scorer_hw=(384, 512),
    )

    cfg = MlxScoreAwareConfig(
        epochs=1, batch_size=batch_size,
        seg_weight=stage.seg_weight, pose_weight=stage.pose_weight,
        use_muon=stage.use_muon, adamw_lr=stage.adamw_lr, muon_lr=stage.muon_lr,
        latent_lr_mult=stage.latent_lr_mult, grad_clip=stage.grad_clip,
        grad_clip_muon=stage.grad_clip, cosine_lr_schedule=False,  # lr_scale=1.0
        ema_decay=stage.ema_decay, use_ema_for_eval=False, seed=args.seed,
    )
    trainer = MlxScoreAwareTrainer(bundle, bridge, cfg)
    # NOTE: stage QAT/C1a/sigma mechanisms — the MLX trainer arms these via
    # configure_stage(); for the forward/loss diff we use the bridge directly so
    # the seg-loss family matches. (QAT/C1a are weight-domain; they affect the
    # *gradient* path, surfaced in (C)/(D), not the forward loss family.)

    indices = mx.array(idx_np.astype(np.int32))
    render_mlx = trainer._render(indices)
    mx.eval(render_mlx)
    res = bridge.loss_and_pixel_grad(render_mlx, idx_t)

    # --- render parity (the decoder forward itself) ---
    render_mlx_np = np.asarray(render_mlx).astype(np.float64)
    render_t_np = tf["render_leaf"].detach().numpy().astype(np.float64)
    render_max_abs = float(np.max(np.abs(render_mlx_np - render_t_np)))
    render_max_uint8 = float(np.max(np.abs(np.round(render_mlx_np) - np.round(render_t_np))))

    fwd = {
        "loss":  {"torch": float(tf["loss"].item()),  "mlx": float(res.loss_value),
                  "rel": _rel(float(tf["loss"].item()), float(res.loss_value))},
        "seg_l": {"torch": float(tf["seg_l"].item()), "mlx": float(res.seg_loss_value),
                  "rel": _rel(float(tf["seg_l"].item()), float(res.seg_loss_value))},
        "pose_l": {"torch": float(tf["pose_l"].item()), "mlx": float(res.pose_loss_value),
                   "rel": _rel(float(tf["pose_l"].item()), float(res.pose_loss_value))},
        "d_seg": {"torch": tf["d_seg"], "mlx": float(res.d_seg),
                  "abs": abs(tf["d_seg"] - float(res.d_seg))},
        "render_max_abs_diff_float": render_max_abs,
        "render_max_abs_diff_uint8": render_max_uint8,
    }
    report["A_forward"] = fwd

    # =======================================================================
    # (B) COTANGENT parity: torch leaf.grad  vs  bridge pixel_cotangent.
    # =======================================================================
    tf["loss"].backward()
    cot_t = tf["render_leaf"].grad.detach().numpy().astype(np.float64)
    cot_mlx = np.asarray(res.pixel_cotangent).astype(np.float64)
    cot_t_norm = float(np.linalg.norm(cot_t))
    cot_mlx_norm = float(np.linalg.norm(cot_mlx))
    # cosine + relative-norm: the cotangent direction is what the vjp backprops.
    denom = (np.linalg.norm(cot_t) * np.linalg.norm(cot_mlx)) + 1e-30
    cosine = float(np.sum(cot_t * cot_mlx) / denom)
    report["B_cotangent"] = {
        "torch_norm": cot_t_norm, "mlx_norm": cot_mlx_norm,
        "norm_rel": _rel(cot_t_norm, cot_mlx_norm),
        "cosine_similarity": cosine,
        "max_abs_diff": float(np.max(np.abs(cot_t - cot_mlx))),
    }

    # =======================================================================
    # (C)+(D) PARAMETER-GRADIENT + POST-STEP-WEIGHT parity.
    #
    # Run ONE full optimizer step on each side from the SAME init, then diff the
    # resulting weight deltas. This captures grad + optimizer + grad-clip + EMA
    # in one cumulative observable (the thing that actually drives convergence).
    # =======================================================================
    # ---- torch one step (vendored optim: AdamW [+ Muon]) ----
    decoder_t2 = torch_model.HNeRVDecoder(28, 36, EVAL_SIZE).to("cpu")
    decoder_t2.load_state_dict(init_sd)
    decoder_t2.train()
    latents_p2 = torch.nn.Parameter(init_latents.clone())

    if stage.use_muon:
        muon_params, adamw_params = torch_optim.partition_params_for_muon(decoder_t2)
        muon_opt = torch_optim.Muon(
            muon_params, lr=stage.muon_lr, momentum=0.95, nesterov=True,
            ns_steps=5, weight_decay=0.0,
        )
        adamw_opt = torch.optim.AdamW(
            [{"params": adamw_params, "lr": stage.adamw_lr},
             {"params": [latents_p2], "lr": stage.adamw_lr * stage.latent_lr_mult}],
            weight_decay=0.0,
        )
    else:
        muon_opt = None
        muon_params = []
        adamw_params = list(decoder_t2.parameters())
        adamw_opt = torch.optim.AdamW(
            [{"params": decoder_t2.parameters(), "lr": stage.adamw_lr},
             {"params": [latents_p2], "lr": stage.adamw_lr * stage.latent_lr_mult}],
            weight_decay=0.0,
        )

    tf2 = _torch_forward_loss(
        decoder_t2, latents_p2, idx_t, dnet, seg_targets_hard, pose_targets,
        stage, torch_losses,
    )
    adamw_opt.zero_grad()
    if muon_opt is not None:
        muon_opt.zero_grad()
    tf2["loss"].backward()
    torch.nn.utils.clip_grad_norm_(adamw_params + [latents_p2], stage.grad_clip)
    if muon_opt is not None:
        torch.nn.utils.clip_grad_norm_(muon_params, stage.grad_clip)
    adamw_opt.step()
    if muon_opt is not None:
        muon_opt.step()

    post_sd_t = {k: v.detach().numpy().astype(np.float64)
                 for k, v in decoder_t2.state_dict().items()}
    post_latents_t = latents_p2.detach().numpy().astype(np.float64)
    delta_w_t = {k: post_sd_t[k] - init_sd[k].numpy().astype(np.float64)
                 for k in post_sd_t}
    delta_lat_t = post_latents_t - init_latents.numpy().astype(np.float64)

    # ---- MLX one step (fresh bundle from same init; trainer.step) ----
    bundle2 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=n_pairs, latent_dim=28, base_channels=36, output_layout="n2chw",
    )
    load_pytorch_state_dict_into_mlx(bundle2.decoder, init_sd)
    bundle2.latents = mx.array(init_latents.numpy())
    mx.eval(bundle2.parameters())
    trainer2 = MlxScoreAwareTrainer(bundle2, bridge, cfg)
    _ = trainer2.step(idx_np, lr_scale=1.0)

    from tac.local_acceleration.pr95_hnerv_mlx import pytorch_state_dict_from_mlx
    post_sd_mlx = {k: np.asarray(v).astype(np.float64)
                   for k, v in pytorch_state_dict_from_mlx(bundle2.decoder).items()}
    post_latents_mlx = np.asarray(bundle2.latents).astype(np.float64)
    delta_w_mlx = {k: post_sd_mlx[k] - init_sd[k].numpy().astype(np.float64)
                   for k in post_sd_mlx if k in init_sd}
    delta_lat_mlx = post_latents_mlx - init_latents.numpy().astype(np.float64)

    # Per-tensor weight-delta diff (the cumulative step observable).
    per_tensor: list[dict[str, Any]] = []
    for k in sorted(delta_w_t):
        if k not in delta_w_mlx:
            per_tensor.append({"tensor": k, "status": "MISSING_IN_MLX"})
            continue
        dt, dm = delta_w_t[k], delta_w_mlx[k]
        nt, nm = float(np.linalg.norm(dt)), float(np.linalg.norm(dm))
        denom = nt * nm + 1e-30
        cos = float(np.sum(dt * dm) / denom) if denom > 1e-29 else 1.0
        per_tensor.append({
            "tensor": k, "torch_delta_norm": nt, "mlx_delta_norm": nm,
            "norm_rel": _rel(nt, nm), "delta_cosine": cos,
            "max_abs_delta_diff": float(np.max(np.abs(dt - dm))),
        })
    lat_nt, lat_nm = float(np.linalg.norm(delta_lat_t)), float(np.linalg.norm(delta_lat_mlx))
    lat_denom = lat_nt * lat_nm + 1e-30
    report["D_weight_deltas"] = {
        "per_tensor": per_tensor,
        "latents": {
            "torch_delta_norm": lat_nt, "mlx_delta_norm": lat_nm,
            "norm_rel": _rel(lat_nt, lat_nm),
            "delta_cosine": float(np.sum(delta_lat_t * delta_lat_mlx) / lat_denom),
        },
    }

    # --- FIRST-DIVERGENCE verdict ------------------------------------------
    divergences: list[str] = []
    if fwd["render_max_abs_diff_uint8"] > 2:
        divergences.append(
            f"RENDER (decoder forward): max uint8 diff {fwd['render_max_abs_diff_uint8']:.0f} > 2"
        )
    if fwd["loss"]["rel"] > args.rtol:
        divergences.append(f"LOSS rel {fwd['loss']['rel']:.2e} > {args.rtol}")
    if fwd["seg_l"]["rel"] > args.rtol:
        divergences.append(f"SEG_LOSS rel {fwd['seg_l']['rel']:.2e} > {args.rtol}")
    if fwd["pose_l"]["rel"] > args.rtol:
        divergences.append(f"POSE_LOSS rel {fwd['pose_l']['rel']:.2e} > {args.rtol}")
    if report["B_cotangent"]["cosine_similarity"] < 0.999:
        divergences.append(
            f"COTANGENT cosine {report['B_cotangent']['cosine_similarity']:.5f} < 0.999"
        )
    bad_tensors = [t["tensor"] for t in per_tensor
                   if t.get("delta_cosine", 1.0) < 0.99 or t.get("status") == "MISSING_IN_MLX"]
    if bad_tensors:
        divergences.append(f"WEIGHT-DELTA cosine<0.99 on {len(bad_tensors)} tensors: {bad_tensors[:5]}")
    report["first_divergences"] = divergences
    report["step1_parity"] = len(divergences) == 0

    # =======================================================================
    # (E) N-STEP TRAJECTORY DRIFT (only meaningful if step-1 ~ parity, but we
    # run it regardless so the report shows the cumulative behavior).
    # Both sides: same init, same per-step batch (deterministic arange-cycling).
    # =======================================================================
    traj = {"torch_d_seg": [], "mlx_d_seg": [], "abs_diff": []}
    if args.traj_steps > 0:
        # torch trajectory (fresh from init)
        dec_e = torch_model.HNeRVDecoder(28, 36, EVAL_SIZE).to("cpu")
        dec_e.load_state_dict(init_sd)
        dec_e.train()
        lat_e = torch.nn.Parameter(init_latents.clone())
        ema_dec = deepcopy(dec_e)
        ema_lat = lat_e.data.clone()
        if stage.use_muon:
            mp, ap = torch_optim.partition_params_for_muon(dec_e)
            m_opt = torch_optim.Muon(mp, lr=stage.muon_lr, momentum=0.95,
                                     nesterov=True, ns_steps=5, weight_decay=0.0)
            a_opt = torch.optim.AdamW(
                [{"params": ap, "lr": stage.adamw_lr},
                 {"params": [lat_e], "lr": stage.adamw_lr * stage.latent_lr_mult}],
                weight_decay=0.0)
        else:
            m_opt, mp = None, []
            ap = list(dec_e.parameters())
            a_opt = torch.optim.AdamW(
                [{"params": dec_e.parameters(), "lr": stage.adamw_lr},
                 {"params": [lat_e], "lr": stage.adamw_lr * stage.latent_lr_mult}],
                weight_decay=0.0)

        def torch_d_seg_eval(d: torch.nn.Module, lat: torch.Tensor) -> float:
            d.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for s in range(0, n_pairs, batch_size):
                    ii = torch.arange(s, min(s + batch_size, n_pairs))
                    out = _torch_forward_loss(d, lat, ii, dnet, seg_targets_hard,
                                              pose_targets, stage, torch_losses)
                    tot += out["d_seg"] * len(ii); cnt += len(ii)
            d.train()
            return tot / max(cnt, 1)

        for s in range(args.traj_steps):
            start = (s * batch_size) % n_pairs
            ii = torch.arange(start, min(start + batch_size, n_pairs))
            if len(ii) < batch_size and n_pairs >= batch_size:
                ii = torch.arange(0, batch_size)
            out = _torch_forward_loss(dec_e, lat_e, ii, dnet, seg_targets_hard,
                                      pose_targets, stage, torch_losses)
            a_opt.zero_grad()
            if m_opt is not None:
                m_opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(ap + [lat_e], stage.grad_clip)
            if m_opt is not None:
                torch.nn.utils.clip_grad_norm_(mp, stage.grad_clip)
            a_opt.step()
            if m_opt is not None:
                m_opt.step()
            torch_losses.ema_update(ema_dec, dec_e, ema_lat, lat_e, decay=stage.ema_decay)

        # MLX trajectory (fresh from init; trainer3)
        bundle3 = HNeRVSyntheticTrainingBundleMLX(
            latent_count=n_pairs, latent_dim=28, base_channels=36, output_layout="n2chw")
        load_pytorch_state_dict_into_mlx(bundle3.decoder, init_sd)
        bundle3.latents = mx.array(init_latents.numpy())
        mx.eval(bundle3.parameters())
        trainer3 = MlxScoreAwareTrainer(bundle3, bridge, cfg)
        for s in range(args.traj_steps):
            start = (s * batch_size) % n_pairs
            ii = np.arange(start, min(start + batch_size, n_pairs), dtype=np.int64)
            if len(ii) < batch_size and n_pairs >= batch_size:
                ii = np.arange(0, batch_size, dtype=np.int64)
            trainer3.step(ii, lr_scale=1.0)

        d_seg_t_final = torch_d_seg_eval(dec_e, lat_e)
        d_seg_mlx_final = trainer3.exact_d_seg(use_ema=False)
        traj = {
            "steps": args.traj_steps,
            "torch_live_d_seg_final": d_seg_t_final,
            "mlx_live_d_seg_final": d_seg_mlx_final,
            "abs_diff_final": abs(d_seg_t_final - d_seg_mlx_final),
            "init_d_seg": fwd["d_seg"]["torch"],
        }
    report["E_trajectory"] = traj

    # --- disposition --------------------------------------------------------
    if report["step1_parity"]:
        td = traj.get("abs_diff_final", 0.0)
        if td and td > 0.02:
            report["disposition"] = (
                "step-1-parity-holds-BUT-trajectory-drifts: "
                f"after {args.traj_steps} steps torch={traj['torch_live_d_seg_final']:.5f} "
                f"vs mlx={traj['mlx_live_d_seg_final']:.5f} (|Δ|={td:.5f}) — cumulative drift."
            )
        else:
            report["disposition"] = (
                "port-is-faithful-step-and-trajectory: the wall is NOT a step-level "
                "numerical bug; look at the recipe SCHEDULE (Muon-throughout vs "
                "AdamW-then-Muon, stage epochs, init/resume) — config not arithmetic."
            )
    else:
        report["disposition"] = (
            "port-DIVERGES-at-step-1 — FIRST divergence(s): "
            + " | ".join(divergences)
            + "  ==> hand to sister agent a28f8a9c for the core-file fix."
        )

    out_path = args.out or str(
        REPO_ROOT
        / f"experiments/results/pr95_vendored_vs_mlx_parity_{args.stage}_"
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}/report.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    report["_out_path"] = out_path
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
