#!/usr/bin/env python3
"""R14 hardening: the Lever-4↔variable-level-export UNIFICATION NET-effect on the REAL
frozen scorer (the contest-validity check the operator mandated: "completely engineer
+ implement + adversarially review + harden").

The unification (``cfg.lever4_variable_level_export_enabled``) routes Lever-4's online
``||∂S/∂w||`` sensitivity into a VARIABLE per-tensor INT8 EXPORT grid (reverse-
waterfill). The BYTE saving is already proven (regression test). This probe answers the
DECISIVE contest question: is the byte saving NET-POSITIVE — i.e. does the sensitivity-
guided coarsening PRESERVE d_seg (coarsen only score-irrelevant tensors), so the saved
bytes are a real contest-score win and not a d_seg regression?

It runs on a REAL decoder + a REAL frozen-scorer ``||∂S/∂w||`` sensitivity (one
backward), builds BOTH the vendored uniform export AND the unification variable export,
parses BOTH back, and measures the REAL d_seg of each on the real SegNet. The verdict:
  - byte_saving > 0 (the variable export is smaller), AND
  - d_seg_variable <= d_seg_uniform + tol  (coarsening did not regress d_seg) =>
    NET-POSITIVE on the contest axis (advisory; the authoritative claim still needs the
    600-pair byte-closed dual CPU/CUDA eval).

Authority: real frozen scorer (SegNet argmax d_seg), CPU-TRUSTED, tiny slice =>
[contest-CPU advisory] NON-PROMOTABLE (byte-effect + d_seg-direction; NOT a score claim).
NO daemon touched (writes only .omx/tmp/r14_*).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.torch_vehicle.driver import (
    _EVAL_H,
    _EVAL_W,
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import RealScorerContext

_VIDEO = "upstream/videos/0.mkv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--out", default=".omx/tmp/r14_unif_net")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    sc = RealScorerContext(
        _VIDEO, device="cpu", max_pairs=args.n_pairs,
        targets_cache=str(out / "targets_cache"),
    )
    v = import_vendored_bundle()
    n = int(args.n_pairs)
    idx = torch.arange(min(4, n))

    # A real (lightly-trained) decoder so the sensitivity is meaningful, not random.
    cfg = TorchVehicleConfig(
        base_channels=20, latent_dim=28, out_dir=str(out / "run"), device="cpu", seed=0,
    )
    drv = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[])
    dec = drv._new_vendored_decoder(device=torch.device("cpu"))
    latents = torch.nn.Parameter(torch.randn(n, 28) * 0.1)

    def render(d, lat):
        dp = d(lat[idx])
        B = len(idx)
        flat = dp.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
        dc = bhwc.clamp(0, 255)
        return dc + (dc.round() - dc).detach()

    # Lightly train so weights are non-random (cheap: a few steps of CE seg).
    opt = torch.optim.Adam(list(dec.parameters()) + [latents], lr=3e-3)
    for _ in range(15):
        opt.zero_grad()
        seg_out, _pose = sc.seg_pose_forward(render(dec, latents))
        loss = F.cross_entropy(seg_out, sc.seg_targets_hard[idx])
        loss.backward()
        opt.step()

    # One REAL backward of the score-domain seg loss to seed ||dS/dw|| per tensor.
    opt.zero_grad()
    seg_out, _pose = sc.seg_pose_forward(render(dec, latents))
    seg_loss = F.cross_entropy(seg_out, sc.seg_targets_hard[idx])
    seg_loss.backward()
    sensitivity = {}
    for name, m in dec.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) and getattr(m, "weight", None) is not None:
            if m.weight.grad is not None:
                sensitivity[f"{name}.weight"] = float(m.weight.grad.detach().norm())

    sd = {k: t.detach().clone() for k, t in dec.state_dict().items()}
    lat = latents.detach().clone()
    meta = {"n_pairs": n, "latent_dim": 28, "base_channels": 20, "eval_size": [_EVAL_H, _EVAL_W]}

    # vendored uniform export.
    drv_off = TorchVehicleDriver(
        TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=str(out / "off"),
                           device="cpu", lever4_variable_level_export_enabled=False),
        scorer=sc, vendored=v, curriculum=[],
    )
    arc_u, eval_dec_u, lat_u = drv_off._build_archive_and_eval_decoder(sd, lat, dict(meta))

    # unification variable export from the REAL sensitivity.
    drv_on = TorchVehicleDriver(
        TorchVehicleConfig(base_channels=20, latent_dim=28, out_dir=str(out / "on"),
                           device="cpu", lever4_variable_level_export_enabled=True),
        scorer=sc, vendored=v, curriculum=[],
    )
    arc_v, eval_dec_v, lat_v = drv_on._build_archive_and_eval_decoder(
        sd, lat, dict(meta), sensitivity=sensitivity
    )

    # measure REAL d_seg of each parse-back decoder on the real SegNet.
    def real_d_seg(d, latt):
        with torch.no_grad():
            so, _ = sc.seg_pose_forward(render(d, latt.to(torch.device("cpu"))))
            return float((so.argmax(dim=1) != sc.seg_targets_hard[idx]).float().mean())

    d_seg_u = real_d_seg(eval_dec_u, lat_u)
    d_seg_v = real_d_seg(eval_dec_v, lat_v)
    byte_saving = len(arc_u) - len(arc_v)
    # contest-score deltas (the rate term + the 100*d_seg term).
    rate_delta = 25.0 * byte_saving / 37_545_489
    seg_delta = 100.0 * (d_seg_v - d_seg_u)
    net_score_delta = seg_delta - rate_delta  # POSITIVE = the variable export is WORSE

    verdict = {
        "scorer_class": type(sc).__name__,
        "n_pairs": n,
        "uniform_archive_bytes": len(arc_u),
        "variable_archive_bytes": len(arc_v),
        "byte_saving": byte_saving,
        "byte_saving_pct": round(100.0 * byte_saving / max(len(arc_u), 1), 2),
        "real_d_seg_uniform": d_seg_u,
        "real_d_seg_variable": d_seg_v,
        "d_seg_delta_variable_minus_uniform": d_seg_v - d_seg_u,
        "contest_rate_delta_from_bytes": rate_delta,
        "contest_seg_delta_100x": seg_delta,
        "net_score_delta_advisory": net_score_delta,
        "n_tensors_with_sensitivity": len(sensitivity),
    }
    # NET-POSITIVE iff bytes saved AND the net score delta is <= 0 (the rate win is not
    # eaten by a d_seg regression). Tolerance: tiny-slice d_seg is noisy.
    verdict["byte_saving_real"] = bool(byte_saving > 0)
    verdict["net_positive_advisory"] = bool(byte_saving > 0 and net_score_delta <= 1e-6)
    (out / "r14_unif_net_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
