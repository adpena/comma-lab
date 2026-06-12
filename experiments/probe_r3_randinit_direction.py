# SPDX-License-Identifier: MIT
"""R3 review — Lever-1 gradient-direction from a HIGH-ENTROPY start (clear rate
headroom). The basin EMA sits at a near-rate-minimum where per-step byte movement is
dominated by codec quant noise (±tens of bytes on 73.5KB); to expose the TRUE gradient
direction we descend from a random-init decoder (uses most of the 127-symbol alphabet),
where there is real headroom, and confirm the REAL decoder bytes go DOWN monotonically.

Authority: ``[macOS-CPU advisory]`` NON-PROMOTABLE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tac.losses.rate_surrogate import RateSurrogateConfig, conditional_weight_entropy  # noqa: E402
from tac.torch_vehicle.vendored_imports import import_vendored  # noqa: E402


def main() -> int:
    codec = import_vendored("codec")
    model = import_vendored("model")
    cfg = RateSurrogateConfig(codec_scan_order=True)

    def db(d):
        return len(codec.encode_decoder(codec.quantize_state_dict(d.state_dict())))

    torch.manual_seed(0)
    dec = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
    b0 = db(dec)
    h0 = float(conditional_weight_entropy(dec, cfg).item())
    opt = torch.optim.SGD(dec.parameters(), lr=1e-2)
    traj = [b0]
    for i in range(40):
        opt.zero_grad()
        h = conditional_weight_entropy(dec, cfg)
        h.backward()
        opt.step()
        if (i + 1) % 10 == 0:
            traj.append(db(dec))
    h1 = float(conditional_weight_entropy(dec, cfg).item())
    print(f"LEVER-1a from RANDOM-INIT (40 steps lr=1e-2):", flush=True)
    print(f"  surrogate: {h0:.4f} -> {h1:.4f}  ({'DOWN' if h1 < h0 else 'UP'})", flush=True)
    print(f"  real_decoder_bytes trajectory (start, @10,@20,@30,@40): {traj}", flush=True)
    print(
        f"  net byte delta: {traj[-1] - traj[0]:+d}  "
        f"({'DOWN-correct' if traj[-1] < traj[0] else 'UP-WRONGWAY'})",
        flush=True,
    )
    print("RANDINIT_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
