# SPDX-License-Identifier: MIT
"""R3 review — cross-lever double-counting / self-cancellation probe (lens C).

Lever-1 (codec-scan-order conditional weight entropy ``H(W|W_prev)``) + the vendored
C1a (``cat_entropy_v2``, marginal ``H(W)``) + Lever-4 score-aware QAT all touch the
decoder-weight rate. R1 claimed "no double-count between Lever-1 and C1a (marginal vs
conditional)". R3 STRESSES that claim now that Lever-1 uses codec-scan-order:

  * Measure the COSINE SIMILARITY of the C1a gradient vs the Lever-1 gradient on the
    real basin decoder. cos>0 → they pull the same way (coherent, possibly redundant);
    cos<0 → they FIGHT (self-cancelling); cos≈0 → orthogonal (distinct structure).
  * Measure whether the COMBINED objective (C1a + Lever-1) reduces real decoder bytes
    MORE than, LESS than, or about the same as either alone — the deploy-faithful test
    of whether stacking them helps, hurts, or is redundant.

Authority: $0 / local / torch-CPU. Every number ``[macOS-CPU advisory]`` NON-PROMOTABLE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tac.losses.cat_entropy_v2 import cat_entropy_v2  # noqa: E402
from tac.losses.rate_surrogate import (  # noqa: E402
    RateSurrogateConfig,
    conditional_weight_entropy,
)
from tac.torch_vehicle.vendored_imports import import_vendored  # noqa: E402

_BASIN = (
    _ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
    / "torch_vehicle_checkpoint_state.pt"
)


def _flat_grad(decoder) -> torch.Tensor:
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in decoder.parameters()
    ])


def main() -> int:
    codec = import_vendored("codec")
    model = import_vendored("model")
    ck = torch.load(_BASIN, map_location="cpu", weights_only=False)
    sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}

    def build():
        d = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
        d.load_state_dict(sd)
        return d

    def dec_bytes(decoder) -> int:
        return len(codec.encode_decoder(codec.quantize_state_dict(decoder.state_dict())))

    cfg = RateSurrogateConfig(codec_scan_order=True)

    # --- gradient cosine at the basin point ----------------------------------
    dec = build()
    for p in dec.parameters():
        p.grad = None
    c1a = cat_entropy_v2(dec, device="cpu")
    c1a.backward()
    g_c1a = _flat_grad(dec).clone()

    for p in dec.parameters():
        p.grad = None
    h1 = conditional_weight_entropy(dec, cfg)
    h1.backward()
    g_l1 = _flat_grad(dec).clone()

    cos = float(
        torch.nn.functional.cosine_similarity(g_c1a.unsqueeze(0), g_l1.unsqueeze(0)).item()
    )
    norm_c1a = float(g_c1a.norm().item())
    norm_l1 = float(g_l1.norm().item())

    # --- deploy-faithful descent: C1a alone / L1 alone / both ----------------
    def descend(use_c1a: bool, use_l1: bool, steps: int = 30, lr: float = 5e-3) -> int:
        d = build()
        opt = torch.optim.SGD(d.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.zeros((), dtype=torch.float32)
            if use_c1a:
                loss = loss + cat_entropy_v2(d, device="cpu")
            if use_l1:
                loss = loss + conditional_weight_entropy(d, cfg)
            loss.backward()
            opt.step()
        return dec_bytes(d)

    b_start = dec_bytes(build())
    b_c1a = descend(True, False)
    b_l1 = descend(False, True)
    b_both = descend(True, True)

    # Self-cancellation check: if the combined descent reduces bytes by LESS than the
    # better of the two alone, the gradients are partially fighting. If it reduces by
    # AT LEAST as much as the better single, they are coherent.
    best_single = min(b_c1a, b_l1)
    combined_helps = b_both <= best_single
    coherent = cos > 0.0

    result = {
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "gradient_cosine_c1a_vs_lever1": round(cos, 4),
        "grad_norm_c1a": round(norm_c1a, 6),
        "grad_norm_lever1": round(norm_l1, 6),
        "real_decoder_bytes": {
            "start": b_start,
            "c1a_alone_30step": b_c1a,
            "lever1_alone_30step": b_l1,
            "both_30step": b_both,
            "best_single": best_single,
        },
        "byte_deltas_vs_start": {
            "c1a_alone": b_c1a - b_start,
            "lever1_alone": b_l1 - b_start,
            "both": b_both - b_start,
        },
        "coherent_gradients": coherent,
        "combined_descent_not_worse_than_best_single": combined_helps,
        "verdict": (
            "COHERENT" if (coherent and combined_helps)
            else ("SELF-CANCELLING" if cos < -0.1 else "PARTIALLY-COHERENT")
        ),
    }
    print(json.dumps(result, indent=2))
    print("DOUBLECOUNT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
