#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""FP4 d_seg-HOLD smoke (WS-B research gate 1) — does mixed-precision FP4 hold d_seg?

The micro→macro audit (`.omx/research/small_basis_optimization_register_20260615.md`) found the
decoder ships INT8 (~7.35 b/param) while FP4 (~4 b/param) would cut ~22KB (Δrate ≈ −0.022) — the
sub-0.15 *maker*. The OPEN RISK: does 4-bit interior-conv quant cost d_seg? The macro μ4 bridge
predicts NO (d_seg is blind to HF detail >~192×256, FP4 noise is HF), but it was unmeasured.

This $0 CPU smoke re-quantizes the EXISTING control best EMA checkpoint (no retrain) four ways and
measures the REAL SegNet argmax-flip d_seg under each via the authority exact_eval path:
  FP32 (baseline) · INT8-all (current ship) · FP4-all (aggressive) · FP4-mixed (interior FP4 + heads INT8).

GO criterion: d_seg(FP4-mixed) ≈ d_seg(INT8) (within ~10%) ⇒ FP4 is d_seg-safe ⇒ the −0.022 rate win is
real (combine with the macro byte measurement: int8 76.6KB → mixed 42.9KB). NO-GO: FP4 spills d_seg →
needs FP4-QAT (train the decoder to tolerate 4-bit) before the win is bankable.

AUTHORITY: `[contest-CPU advisory] NON-PROMOTABLE` — in-loop CPU eval on the 96-pair subset; not the
byte-closed exact archive. CPU only (NO MPS); does not disturb the live stack_ce_early MPS run.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

_RATE_DENOM = 37_545_489
_BASE = Path("experiments/results/from0_ab_v2_n96/control/best")
_E2M1_ABS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # FP4 E2M1 magnitudes
# the d_seg-sensitive OUTPUT-PROXIMAL heads stay INT8 (macro Φ3: the 384×512 band where flips live)
_HEAD_PREFIXES = ("skips.", "refine.", "rgb_0.", "rgb_1.")


def _int8_qdq(t: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric INT8 quantize→dequantize (mirrors the shipped codec)."""
    s = t.abs().max() / 127.0
    if float(s) == 0.0:
        return t.clone()
    return (t / s).round().clamp_(-127, 127) * s


def _intN_qdq(t: torch.Tensor, nbits: int) -> torch.Tensor:
    """Per-tensor symmetric int-N quantize→dequantize (free post-hoc; finds the bit floor)."""
    qmax = 2 ** (nbits - 1) - 1
    s = t.abs().max() / qmax
    if float(s) == 0.0:
        return t.clone()
    return (t / s).round().clamp_(-qmax, qmax) * s


def _e2m1_qdq(t: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1 per-output-channel quantize→dequantize (real 4-bit grid, not a marker)."""
    grid = _E2M1_ABS.to(t.dtype)
    if t.dim() >= 2:
        red = tuple(range(1, t.dim()))
        s = t.abs().amax(dim=red, keepdim=True) / 6.0
    else:
        s = t.abs().max() / 6.0
    s = s.clamp_min(1e-12)
    tn = t / s
    idx = (tn.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
    return tn.sign() * grid[idx] * s


def _is_head(key: str) -> bool:
    return any(key.startswith(p) for p in _HEAD_PREFIXES)


def _requant(dec_sd: dict, mode: str) -> dict:
    """mode ∈ {fp32, int8, fp4_all, fp4_mixed}. Only weight tensors (dim>=2) are quantized;
    biases keep fp32 (output-proximal, codec keeps precision)."""
    out = {}
    for k, v in dec_sd.items():
        if not torch.is_tensor(v) or v.dim() < 2:
            out[k] = v.clone()
            continue
        if mode == "fp32":
            out[k] = v.clone()
        elif mode == "int8":
            out[k] = _int8_qdq(v)
        elif mode in ("int6", "int5", "int4u"):
            out[k] = _intN_qdq(v, {"int6": 6, "int5": 5, "int4u": 4}[mode])
        elif mode == "fp4_all":
            out[k] = _e2m1_qdq(v)
        elif mode == "fp4_mixed":
            out[k] = _int8_qdq(v) if _is_head(k) else _e2m1_qdq(v)
        else:
            raise ValueError(mode)
    return out


def main() -> int:
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    v = import_vendored_bundle()
    model_mod = import_vendored("model")
    video_path = import_vendored("data").get_default_video_path()

    dec_sd = torch.load(_BASE / "best_ema_decoder.pt", map_location="cpu", weights_only=False)
    dec_sd = dec_sd.get("state_dict", dec_sd) if isinstance(dec_sd, dict) and "state_dict" in dec_sd else dec_sd
    latents = torch.load(_BASE / "best_ema_latents.pt", map_location="cpu", weights_only=False)
    if isinstance(latents, dict):
        latents = latents.get("latents", next(iter(latents.values())))
    archive_bytes_int8 = (_BASE / "best_archive.bin").stat().st_size  # 76,592

    # CPU AUTHORITY scorer (NO mps → does not contend with the live MPS run on the GPU).
    ctx = RealScorerContext(
        video_path, device="cpu", train_device="cpu", split_by_head=False,
        max_pairs=96, targets_cache=Path("experiments/results/capstone_gt_targets_cache"),
    )

    def build(mode: str):
        d = model_mod.HNeRVDecoder(latent_dim=28, base_channels=20).eval()
        d.load_state_dict(_requant(dec_sd, mode))
        return d

    # rate-term bytes: int8 = measured archive; FP4 = macro-measured; intN = ROUGH (nbits/8
    # scaling of the 73,604B decoder blob + 2,923B latents/meta — ignores brotli interplay).
    _dec, _other = 73_604, archive_bytes_int8 - 73_604
    bytes_for = {"fp32": archive_bytes_int8, "int8": archive_bytes_int8,
                 "int6": round(_dec * 6 / 8) + _other, "int5": round(_dec * 5 / 8) + _other,
                 "int4u": round(_dec * 4 / 8) + _other, "fp4_all": 38_261, "fp4_mixed": 42_913}
    rows = {}
    for mode in ("fp32", "int8", "int6", "int5", "int4u", "fp4_mixed", "fp4_all"):
        res = ctx.exact_eval(build(mode), latents, bytes_for[mode])
        d_seg = res["seg_distortion"]; d_pose = res["pose_distortion"]
        rate = 25.0 * bytes_for[mode] / _RATE_DENOM
        score = 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + rate
        rows[mode] = {"d_seg": d_seg, "d_pose": d_pose, "bytes": bytes_for[mode],
                      "rate_term": rate, "score": score}
        print(f"[{mode:9}] d_seg={d_seg:.6f}  d_pose={d_pose:.6f}  bytes={bytes_for[mode]:>6}  "
              f"rate={rate:.4f}  S={score:.4f}  [contest-CPU advisory]", flush=True)

    base = rows["fp32"]["d_seg"]; i8 = rows["int8"]["d_seg"]
    mix = rows["fp4_mixed"]["d_seg"]; allf = rows["fp4_all"]["d_seg"]
    rel_mixed = (mix - i8) / max(i8, 1e-9)
    verdict = ("GO: FP4-mixed holds d_seg (Δ vs int8 within 10%) → the −0.022 rate win is real"
               if rel_mixed <= 0.10 else
               "NO-GO: FP4-mixed spills d_seg >10% vs int8 → needs FP4-QAT before the win banks")
    print("\n=== FP4 d_seg-HOLD VERDICT ===")
    print(f"  d_seg: fp32={base:.6f}  int8={i8:.6f} (Δ{ (i8-base)/max(base,1e-9)*100:+.1f}%)  "
          f"fp4_mixed={mix:.6f} (Δ{rel_mixed*100:+.1f}% vs int8)  fp4_all={allf:.6f}")
    print(f"  ΔS int8→fp4_mixed: {rows['fp4_mixed']['score']-rows['int8']['score']:+.4f} "
          f"(rate −{rows['int8']['rate_term']-rows['fp4_mixed']['rate_term']:.4f}, "
          f"seg {100*(mix-i8):+.4f})")
    print(f"  VERDICT: {verdict}")
    # FREE post-hoc floor: lowest int-N (no QAT) whose d_seg stays within 10% of int8.
    floor = None
    for m in ("int4u", "int5", "int6", "int8"):
        if m in rows and (rows[m]["d_seg"] - i8) / max(i8, 1e-9) <= 0.10:
            floor = m; break
    if floor:
        fr = rows[floor]
        print(f"  FREE POST-HOC FLOOR: {floor} holds d_seg ({fr['d_seg']:.6f}, "
              f"Δ{(fr['d_seg']-i8)/max(i8,1e-9)*100:+.1f}% vs int8) → S≈{fr['score']:.4f} "
              f"(ΔS vs int8 {fr['score']-rows['int8']['score']:+.4f}); a NO-QAT partial rate win")
    else:
        print("  FREE POST-HOC FLOOR: none below int8 holds d_seg → all sub-8-bit needs QAT")
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/fp4_dseg_hold_smoke.json")
    out.write_text(json.dumps({"rows": rows, "rel_mixed_vs_int8": rel_mixed, "verdict": verdict,
                               "authority": "contest-CPU advisory NON-PROMOTABLE"}, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
