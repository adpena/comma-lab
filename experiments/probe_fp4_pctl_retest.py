#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ADVERSARIAL re-test of the FP4 d_seg-hold NO-GO (gate 1) — is it a SCALING artifact?

The gate-1 smoke used max-abs per-channel E2M1 scaling (scale = |t|.max()/6). max-abs is
outlier-sensitive: one big weight inflates the scale → the 8 FP4 levels spread too wide →
the bulk rounds coarsely. So the +56% d_seg spill MIGHT be a naive-scaling artifact, not FP4
fundamentally. This re-test compares:
  - fp4_mixed_maxabs  : reproduce the gate-1 NO-GO (interior FP4 / heads int8, max-abs)
  - fp4_mixed_pctl    : SAME partition, PERCENTILE-clip scaling (99.9th pct → tighter grid)
  - fp4_sensguided    : gate-2 partition (keep the d_seg-critical rgb_1+skips+blocks.4 int8,
                        FP4 the insensitive bulk) + percentile scaling — the OPTIMAL post-hoc

If fp4_sensguided d_seg ≤ ~0.0038 → the post-hoc NO-GO is REVERSED (a free FP4 win exists);
else → the NO-GO is robust (the big low-res tensors genuinely need QAT). AUTHORITY:
`[contest-CPU advisory] NON-PROMOTABLE`. CPU only; does not disturb the live MPS run.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

_BASE = Path("experiments/results/from0_ab_v2_n96/control/best")
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# gate-2 d_seg-critical tensors (keep int8 in the sensitivity-guided partition)
_SENS_KEEP = ("rgb_1.weight", "skips.2.weight", "skips.3.weight", "skips.4.weight", "blocks.4.weight")
# gate-1 "all heads" int8-keep
_HEAD_KEEP = ("skips.", "refine.", "rgb_0.", "rgb_1.")


def _chan_scale(t, level_max, pctl):
    flat = t.abs().flatten(1) if t.dim() >= 2 else t.abs().reshape(1, -1)
    if pctl is None:
        s = flat.amax(dim=1)
    else:
        s = torch.quantile(flat, pctl, dim=1)
    return (s / level_max).clamp_min(1e-12).view([-1] + [1] * (t.dim() - 1))


def _e2m1_qdq(t, pctl):
    s = _chan_scale(t, 6.0, pctl)
    tn = (t / s).clamp(-6.0, 6.0)
    idx = (tn.abs().unsqueeze(-1) - _E2M1.to(t.dtype)).abs().argmin(dim=-1)
    return tn.sign() * _E2M1.to(t.dtype)[idx] * s


def _int8_qdq(t):
    s = (t.abs().max() / 127.0).clamp_min(1e-12)
    return (t / s).round().clamp_(-127, 127) * s


def main() -> int:
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    import_vendored_bundle()
    model_mod = import_vendored("model")
    vp = import_vendored("data").get_default_video_path()
    sd = torch.load(_BASE / "best_ema_decoder.pt", map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
    lat = torch.load(_BASE / "best_ema_latents.pt", map_location="cpu", weights_only=False)
    if isinstance(lat, dict):
        lat = lat.get("latents", next(iter(lat.values())))
    ctx = RealScorerContext(vp, device="cpu", train_device="cpu", split_by_head=False,
                            max_pairs=96, targets_cache=Path("experiments/results/capstone_gt_targets_cache"))

    def is_head(k):
        return any(k.startswith(p) for p in _HEAD_KEEP)

    def requant(mode):
        out = {}
        for k, v in sd.items():
            if not torch.is_tensor(v) or v.dim() < 2:
                out[k] = v.clone(); continue
            if mode == "fp32":
                out[k] = v.clone()
            elif mode == "int8":
                out[k] = _int8_qdq(v)
            elif mode == "fp4_mixed_maxabs":
                out[k] = _int8_qdq(v) if is_head(k) else _e2m1_qdq(v, None)
            elif mode == "fp4_mixed_pctl":
                out[k] = _int8_qdq(v) if is_head(k) else _e2m1_qdq(v, 0.999)
            elif mode == "fp4_sensguided":
                out[k] = _int8_qdq(v) if k in _SENS_KEEP else _e2m1_qdq(v, 0.999)
            else:
                raise ValueError(mode)
        return out

    def build(mode):
        d = model_mod.HNeRVDecoder(latent_dim=28, base_channels=20).eval()
        d.load_state_dict(requant(mode)); return d

    rows = {}
    for mode in ("fp32", "int8", "fp4_mixed_maxabs", "fp4_mixed_pctl", "fp4_sensguided"):
        r = ctx.exact_eval(build(mode), lat, 76592)
        rows[mode] = {"d_seg": r["seg_distortion"], "d_pose": r["pose_distortion"]}
        print(f"[{mode:18}] d_seg={r['seg_distortion']:.6f}  d_pose={r['pose_distortion']:.6f}  "
              f"[contest-CPU advisory]", flush=True)

    i8 = rows["int8"]["d_seg"]
    print("\n=== ADVERSARIAL FP4-SCALING RE-TEST ===")
    for m in ("fp4_mixed_maxabs", "fp4_mixed_pctl", "fp4_sensguided"):
        rel = (rows[m]["d_seg"] - i8) / max(i8, 1e-9)
        print(f"  {m:18} d_seg={rows[m]['d_seg']:.6f}  Δ{rel*100:+.1f}% vs int8")
    best = min(("fp4_mixed_pctl", "fp4_sensguided"), key=lambda m: rows[m]["d_seg"])
    rel_best = (rows[best]["d_seg"] - i8) / max(i8, 1e-9)
    verdict = (f"REVERSED: {best} holds d_seg (Δ{rel_best*100:+.1f}% ≤10%) → post-hoc FP4 win EXISTS"
               if rel_best <= 0.10 else
               f"NO-GO ROBUST: best post-hoc FP4 ({best}) still spills d_seg Δ{rel_best*100:+.1f}% → QAT required")
    print(f"  VERDICT: {verdict}")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/fp4_pctl_retest.json").write_text(json.dumps(
        {"rows": rows, "verdict": verdict, "authority": "contest-CPU advisory NON-PROMOTABLE"}, indent=2))
    print("  wrote reports/fp4_pctl_retest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
