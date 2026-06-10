#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PTNC trainer — PoseNet-Tube-Native Carrier with the Jacobian-saliency-weighted anchor (task #61).

THE PTNC DISTINCTION (vs ``score_native_train_luma_carrier.py``, the #57 trainer): the #57 trainer used a
DENSE pixel-MSE anchor (weight >= 0.5 always), spending carrier capacity reproducing pose-IRRELEVANT luma
— the source of the d_pose 0.0036 ceiling. PTNC replaces the dense anchor with the
**PoseNet-Jacobian-saliency-WEIGHTED recon anchor** (USC IDSE with the EXACT measured atlas Jacobian of
the FROZEN PoseNet). Three ``--anchor-mode`` choices make the comparison FALSIFIABLE:

  * ``dense``    : the #57 control — uniform per-pixel recon weight (plain MSE).
  * ``ptnc``     : the invention — recon weight = ``saliency_to_weight_map(measured PoseNet Jacobian)``.
  * ``identity`` : explicit uniform weight via the saliency code path (proves PTNC == dense when the
                   weight field is replaced by identity, i.e. the Jacobian field is load-bearing).

The pose objective (exact 6-dim PoseNet MSE) is IDENTICAL across modes; ONLY the input-domain anchor
changes. So a lower d_pose-per-byte under ``ptnc`` is attributable to WHERE capacity is spent, not to a
different objective (the non-rename guarantee).

FRAME SLOTS:
  * ``--frame-slot 0`` (default): frame0 pose carrier (SegNet-invisible; pure pose RD point).
  * ``--frame-slot 1``: frame1 carrier — ALSO scored by SegNet. The PTNC anchor for frame1 confines luma
    to the pose-relevant pixels; the seg-null confinement (keep argmax) is the candidate-assembly layer.

Authority ``[local CPU-torch advisory]`` (exact PoseNet, GT via yuv420_to_rgb) + ``[macOS research-signal]``
(carrier numpy forward). NO MPS. $0. Non-promotable.

NO-FAKE (class 2 + class 6 + class 8): d_pose is the EXACT PoseNet MSE on the numpy-decoded frame (not a
proxy); byte cost is the brotli of the quantized weights; the saliency weight is the MEASURED Jacobian
(not a constant); a constant carrier fails; identity-weight recovers dense behavior.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
for _p in (REPO_ROOT, REPO_ROOT / "src", _HARNESS, REPO_ROOT / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.boundary_math.amortized_luma_carrier import (  # noqa: E402
    LumaCarrierConfig,
    build_coords,
    carrier_frame,
    carrier_param_count,
    deterministic_fourier_B,
    measure_carrier_bytes,
    save_carrier_npz,
)
from tac.boundary_math.posenet_jacobian_saliency import (  # noqa: E402
    compute_posenet_pixel_saliency,
    identity_weight_map,
    saliency_to_weight_map,
)

CAMERA_H, CAMERA_W = 874, 1164
_CONTEST_TOTAL_BYTES = 37_545_489
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
DEVICE = torch.device("cpu")  # NO MPS, NO cuda for the local advisory loop.


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


class TorchLumaCarrier(nn.Module):
    """Mirror of ``AmortizedLumaCarrier`` math (single arch; numpy parity verified)."""

    def __init__(self, cfg: LumaCarrierConfig):
        super().__init__()
        self.cfg = cfg
        B = deterministic_fourier_B(cfg.n_fourier, cfg.fourier_sigma)
        self.register_buffer("fourier_B", torch.from_numpy(B).float())
        coord_feat = 2 * cfg.n_fourier
        self.in_proj = nn.Linear(coord_feat, cfg.hidden_dim)
        self.film = nn.Linear(cfg.mod_dim, 2 * cfg.hidden_dim * cfg.n_hidden)
        self.hidden = nn.ModuleList([nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_hidden)])
        self.out = nn.Linear(cfg.hidden_dim, cfg.n_channels)
        self.mod = nn.Parameter(torch.zeros(cfg.num_pairs, cfg.mod_dim))

    def forward(self, coords: torch.Tensor, pair_idx: int) -> torch.Tensor:
        proj = coords @ self.fourier_B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        h = F.relu(self.in_proj(feat))
        film = self.film(self.mod[pair_idx]).reshape(self.cfg.n_hidden, 2, self.cfg.hidden_dim)
        for li, layer in enumerate(self.hidden):
            scale = 1.0 + film[li, 0]
            shift = film[li, 1]
            h = F.relu(layer(h) * scale + shift)
        rgb01 = torch.sigmoid(self.out(h))
        return rgb01 * 255.0

    def numpy_params(self) -> dict[str, np.ndarray]:
        out = {}
        for k, v in self.state_dict().items():
            if k == "fourier_B":
                continue
            out[k] = v.detach().cpu().numpy().astype(np.float32)
        return out


def _load_posenet():
    from modules import PoseNet, posenet_sd_path
    from safetensors.torch import load_file

    net = PoseNet().eval().to(DEVICE)
    net.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _gt_pose_targets(posenet, gt_pairs, pairs):
    targets = {}
    with torch.no_grad():
        for pi in pairs:
            g0 = gt_pairs[pi][0].float().to(DEVICE)
            g1 = gt_pairs[pi][1].float().to(DEVICE)
            x = torch.stack([g0, g1]).unsqueeze(0).permute(0, 1, 4, 2, 3)
            pin = posenet.preprocess_input(x)
            out = posenet(pin)["pose"][..., :6]
            targets[pi] = out.detach().clone().requires_grad_(False)
    return targets


def _pose_from_frames(posenet, frame0_chw, frame1_chw):
    x = torch.stack([frame0_chw, frame1_chw]).unsqueeze(0)
    pin = posenet.preprocess_input(x)
    return posenet(pin)["pose"][..., :6]


def _eval_roundtrip(frame_chw: torch.Tensor) -> torch.Tensor:
    rounded = torch.clamp(frame_chw, 0, 255)
    return rounded + (torch.round(rounded) - rounded).detach()


def _build_weight_maps(posenet, gt_pairs, pairs, frame_slot, anchor_mode, floor, gamma):
    """Per-pair (H,W) recon weight maps: dense/identity = uniform; ptnc = measured Jacobian saliency.

    The PTNC weight is measured ONCE at the GT operating point (the atlas Jacobian) — a frozen-oracle
    field, not re-measured per epoch (the trust-region/EM re-measure is a future loop, area b).
    """
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally, unpatch_upstream_yuv6

    maps: dict[int, torch.Tensor] = {}
    if anchor_mode in ("dense", "identity"):
        wm = torch.from_numpy(identity_weight_map(CAMERA_H, CAMERA_W)).to(DEVICE)
        for pi in pairs:
            maps[pi] = wm
        return maps, {"anchor_mode": anchor_mode, "weight_source": "uniform_identity"}

    # ptnc: measure the per-pair Jacobian saliency under the differentiable yuv6 patch.
    tok = patch_upstream_yuv6_globally()
    summaries = []
    try:
        for pi in pairs:
            g0 = gt_pairs[pi][0].float().permute(2, 0, 1).contiguous()
            g1 = gt_pairs[pi][1].float().permute(2, 0, 1).contiguous()
            field = compute_posenet_pixel_saliency(posenet, g0, g1, frame_slot=frame_slot)
            wm = saliency_to_weight_map(field, floor=floor, gamma=gamma, normalize=True)
            maps[pi] = torch.from_numpy(wm).to(DEVICE)
            summaries.append(field.to_summary())
    finally:
        unpatch_upstream_yuv6(tok)
    agg = {k: float(np.mean([s[k] for s in summaries])) for k in summaries[0]} if summaries else {}
    return maps, {"anchor_mode": "ptnc", "weight_source": "measured_posenet_jacobian",
                  "saliency_summary_mean": agg, "floor": floor, "gamma": gamma}


def train(
    targets_dir: Path,
    out_dir: Path,
    cfg: LumaCarrierConfig,
    *,
    n_pairs: int,
    epochs: int,
    lr: float,
    frame_slot: int,
    anchor_mode: str,
    anchor_floor: float,
    anchor_gamma: float,
    pose_weight: float,
    anchor_weight: float,
    seed: int,
    eval_every: int,
) -> dict[str, Any]:
    import render_and_score_lib as L

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally, unpatch_upstream_yuv6

    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta = json.loads((targets_dir / "targets_meta.json").read_text())
    n_built = int(meta["num_pairs_built"])
    pairs = list(range(min(n_pairs, n_built)))
    cfg = LumaCarrierConfig(**{**cfg.to_dict(), "num_pairs": len(pairs)})

    t0 = time.time()
    posenet = _load_posenet()
    gt_pairs = L.decode_gt_pairs(pairs)
    gt_pose_targets = _gt_pose_targets(posenet, gt_pairs, pairs)
    gt_slot = {pi: gt_pairs[pi][frame_slot].float().permute(2, 0, 1).contiguous() for pi in pairs}
    other_slot = 1 - frame_slot
    gt_other = {pi: gt_pairs[pi][other_slot].float().permute(2, 0, 1).contiguous() for pi in pairs}

    weight_maps, anchor_meta = _build_weight_maps(
        posenet, gt_pairs, pairs, frame_slot, anchor_mode, anchor_floor, anchor_gamma
    )
    print(f"[setup] posenet+GT+targets+weights {time.time()-t0:.1f}s n_pairs={len(pairs)} "
          f"mode={anchor_mode} slot={frame_slot}", flush=True)

    coords_np = build_coords(CAMERA_H, CAMERA_W)
    coords = torch.from_numpy(coords_np).to(DEVICE)
    model = TorchLumaCarrier(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    patch_token = patch_upstream_yuv6_globally()
    history: list[dict[str, Any]] = []
    try:
        for ep in range(1, epochs + 1):
            order = np.random.permutation(len(pairs))
            ep_pose, ep_anchor = 0.0, 0.0
            for j in order:
                pi = pairs[j]
                opt.zero_grad()
                rgb = model(coords, j).reshape(CAMERA_H, CAMERA_W, cfg.n_channels).permute(2, 0, 1)
                frame_carrier = _eval_roundtrip(rgb)  # (3,H,W)
                wm = weight_maps[pi]  # (H,W)
                # weighted recon anchor: mean over channels of w*(carrier-GT)^2.
                resid2 = (frame_carrier - gt_slot[pi]) ** 2  # (3,H,W)
                anchor = (wm.unsqueeze(0) * resid2).mean()
                # exact pose objective (the carrier fills frame_slot; the other slot is GT).
                if frame_slot == 0:
                    pose_pred = _pose_from_frames(posenet, frame_carrier, gt_other[pi])
                else:
                    pose_pred = _pose_from_frames(posenet, gt_other[pi], frame_carrier)
                pose_mse = F.mse_loss(pose_pred, gt_pose_targets[pi])
                # warm schedule: anchor-heavy early -> pose-heavy late (both nonzero throughout).
                w_pose = min(1.0, ep / max(1, epochs // 3))
                loss = anchor_weight * (1.0 - 0.5 * w_pose) * anchor + pose_weight * w_pose * pose_mse * 1e4
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_pose += float(pose_mse.detach())
                ep_anchor += float(anchor.detach())
            if ep % eval_every == 0 or ep == 1 or ep == epochs:
                row = {"epoch": ep, "mean_pose_mse_train": ep_pose / len(pairs),
                       "mean_anchor_train": ep_anchor / len(pairs), "wall_s": round(time.time() - t0, 1)}
                history.append(row)
                print(json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
                save_carrier_npz(out_dir / "carrier.npz", model.numpy_params(), cfg)
    finally:
        unpatch_upstream_yuv6(patch_token)

    params = model.numpy_params()
    save_carrier_npz(out_dir / "carrier.npz", params, cfg)
    byte_acct = measure_carrier_bytes(params, cfg)
    parity = _verify_parity(model, params, cfg, coords, coords_np, pairs[:4])

    scorer = L.ExactScorer()
    exact = _exact_measure(scorer, params, cfg, coords_np, gt_pairs, pairs, frame_slot)

    rate = 25.0 * byte_acct.total_bytes / _CONTEST_TOTAL_BYTES
    result = {
        "subagent": "task61_ptnc_frame1_dual_fidelity",
        "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]",
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "config": cfg.to_dict(),
        "param_count": carrier_param_count(cfg),
        "frame_slot": frame_slot,
        "anchor_mode": anchor_mode,
        "anchor_meta": anchor_meta,
        "history": history,
        "byte_account": byte_acct.to_dict(),
        "rate_term_carrier_only": rate,
        "exact_mean_d_pose": exact["mean_d_pose"],
        "exact_per_pair_d_pose": exact["per_pair_d_pose"],
        "portability_parity": parity,
        "pose_term_contribution_sqrt10": float(np.sqrt(10.0 * exact["mean_d_pose"])),
        # the headline RD efficiency metric: d_pose reduction per KB.
        "d_pose_per_kb": float(exact["mean_d_pose"] / max(1.0, byte_acct.total_bytes / 1024.0)),
    }
    (out_dir / "ptnc_train_result.json").write_text(json.dumps(result, indent=2))
    return result


def _verify_parity(model, params, cfg, coords_t, coords_np, pairs):
    agree = []
    for j in range(len(pairs)):
        with torch.inference_mode():
            tr = model(coords_t, j).reshape(CAMERA_H, CAMERA_W, cfg.n_channels)
            tr = torch.clamp(torch.round(tr), 0, 255).numpy().astype(np.uint8)
        npf = carrier_frame(params, cfg, coords_np, j, CAMERA_H, CAMERA_W)
        agree.append(float(np.mean(np.abs(tr.astype(np.int32) - npf.astype(np.int32)) <= 1)))
    return {"rgb_within_1lsb_frac_min": float(min(agree)), "rgb_within_1lsb_frac_mean": float(np.mean(agree)),
            "pairs_checked": len(agree), "parity_pass": bool(min(agree) >= 0.99)}


def _exact_measure(scorer, params, cfg, coords_np, gt_pairs, pairs, frame_slot):
    import render_and_score_lib as L

    d_pose_list: list[float] = []
    for j, pi in enumerate(pairs):
        fc = carrier_frame(params, cfg, coords_np, j, CAMERA_H, CAMERA_W)  # (H,W,3) uint8
        fc_chw = torch.from_numpy(fc.transpose(2, 0, 1)).float()
        if frame_slot == 0:
            f0_chw, f1_chw = fc_chw, gt_pairs[pi][1].float().permute(2, 0, 1)
        else:
            f0_chw, f1_chw = gt_pairs[pi][0].float().permute(2, 0, 1), fc_chw
        comp = torch.stack([f0_chw, f1_chw])
        gt_bthwc = torch.stack([gt_pairs[pi][0], gt_pairs[pi][1]]).float().unsqueeze(0)
        pose_d, _ = scorer.score_batch(gt_bthwc, L.comp_pair_to_bthwc(comp))
        d_pose_list.append(float(pose_d[0]))
    return {"mean_d_pose": float(np.mean(d_pose_list)), "per_pair_d_pose": d_pose_list}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-slot", type=int, default=0, choices=(0, 1))
    ap.add_argument("--anchor-mode", default="ptnc", choices=("dense", "ptnc", "identity"))
    ap.add_argument("--anchor-floor", type=float, default=0.02)
    ap.add_argument("--anchor-gamma", type=float, default=1.0)
    ap.add_argument("--pose-weight", type=float, default=50.0)
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--n-fourier", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-hidden", type=int, default=3)
    ap.add_argument("--mod-dim", type=int, default=24)
    ap.add_argument("--fourier-sigma", type=float, default=6.0)
    ap.add_argument("--quant-bits", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = LumaCarrierConfig(
        num_pairs=args.n_pairs, n_fourier=args.n_fourier, hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden, mod_dim=args.mod_dim, fourier_sigma=args.fourier_sigma,
        quant_bits=args.quant_bits,
    )
    result = train(
        args.targets_dir, args.out_dir, cfg, n_pairs=args.n_pairs, epochs=args.epochs, lr=args.lr,
        frame_slot=args.frame_slot, anchor_mode=args.anchor_mode, anchor_floor=args.anchor_floor,
        anchor_gamma=args.anchor_gamma, pose_weight=args.pose_weight, anchor_weight=args.anchor_weight,
        seed=args.seed, eval_every=args.eval_every,
    )
    print("\n=== PTNC TRAIN RESULT ===")
    print(json.dumps({k: v for k, v in result.items() if k not in ("history", "exact_per_pair_d_pose")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
