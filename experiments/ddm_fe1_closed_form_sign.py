#!/usr/bin/env python3
"""ddm_fe1 CLOSED-FORM-FIRST: the sign of d(boundary)/d(frame code), per dim.

The FiLM path is affine in the frame embedding -- ``film`` is a ``Linear(8, 2*width)``
whose output scales and shifts each block's residual (``cpr1/inflate.py:81,84-86``) --
so before any search this arm linearizes the WHOLE realized chain (renderer -> resample
-> SegNet) about the shipped embedding and reads the sign of the flip response per
dimension.  The linearization is exact except at one place: the receiver rounds the
874x1164 master to uint8 (``cpr1/inflate.py:323-324``) and ``round`` has zero derivative
almost everywhere, so the linearization uses the straight-through surrogate (identity)
there.  That surrogate is DECLARED, not hidden, and the derivation's whole claim is
tested against the realized search on the same pairs.

Per pixel let ``s = logit[top1] - logit[top2]`` (>= 0).  A code step ``delta`` moves the
logits by ``delta * dlogits/dframe_d``, so to first order

* a RIGHT pixel is BROKEN when its top1-vs-top2 margin goes negative;
* a WRONG pixel is REPAIRED when its top1-vs-GT margin goes negative -- the test is
  against the GT logit directly, not against the runner-up, because the pixel only
  counts as repaired if it lands on GT.  Both tests ignore the chance that a THIRD
  class overtakes instead; that is the model's known optimism and it is why the
  realized search, not this predictor, decides.

The predicted flip change is the difference of those two counts.  Eight forward-mode
JVPs give ``dlogits/dframe_d`` for every pixel at once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_fe1_frame_embedding_search as fe1
import ddm_jg1_seg_solve as jg1


def _differentiable_logits(body, pair: int, frame: Any):
    """SegNet logits as a differentiable function of ONE pair's frame embedding.

    Mirrors ``jg1.render_frame1`` + ``jg1.argmax_from_camera_frames`` exactly, with the
    single documented exception of the ``.round()`` at camera resolution.
    """
    import torch
    import torch.nn.functional as functional

    semantic = body.semantic
    tokens = torch.from_numpy(
        np.ascontiguousarray(body.tokens[int(pair)])[None]
    ).long()
    value = semantic.token_embed(tokens).permute(0, 3, 1, 2)
    value = semantic.coord_mix(
        torch.cat(
            [
                value,
                semantic.coordinates(value.shape[0], value.device, value.dtype),
            ],
            dim=1,
        )
    )
    for block in semantic.blocks:
        residual = block.norm(block.pw(block.dw(value)))
        scale, shift = block.film(frame[None]).chunk(2, dim=1)
        residual = residual * (1.0 + scale[:, :, None, None])
        residual = residual + shift[:, :, None, None]
        value = value + functional.gelu(residual)
    rgb = torch.sigmoid(semantic.head(functional.gelu(value))) * 255.0
    master = functional.interpolate(
        rgb, size=(jg1.CAMERA_H, jg1.CAMERA_W), mode="bilinear", align_corners=False
    ).clamp(0.0, 255.0)
    # straight-through for the receiver's uint8 round
    seg_in = body.net.preprocess_input(master.unsqueeze(1))
    return body.net(seg_in)[0]


def linearize_pair(body, pair: int) -> dict[str, Any]:
    """Base logits plus ``dlogits/dframe_d`` for all eight dims, at one pair."""
    import torch
    from torch.func import jvp

    frame0 = torch.from_numpy(
        body.section.codes[int(pair)].astype(np.float32) * body.section.scales
    )

    def forward(frame):
        return _differentiable_logits(body, pair, frame)

    with torch.no_grad():
        base = forward(frame0)
    jac = torch.empty((fe1.FRAME_DIM, *tuple(base.shape)), dtype=base.dtype)
    for dim in range(fe1.FRAME_DIM):
        tangent = torch.zeros(fe1.FRAME_DIM)
        tangent[dim] = 1.0
        _out, tangent_out = jvp(forward, (frame0,), (tangent,))
        jac[dim] = tangent_out.detach()
    return {"base": base.detach(), "jac": jac}


def predict_moves(body, pair: int, linear: dict[str, Any]) -> list[dict[str, Any]]:
    """First-order predicted flip change for every single-code move at one pair."""
    import torch

    base = linear["base"]  # (5, 384, 512)
    jac = linear["jac"]  # (8, 5, 384, 512)
    gt = torch.from_numpy(body.gt[int(pair)].astype(np.int64))
    top2 = base.topk(2, dim=0)
    top1_class = top2.indices[0]
    margin = top2.values[0] - top2.values[1]
    correct = top1_class == gt
    gt_logit = base.gather(0, gt[None])[0]
    # For a WRONG pixel the repair candidate is the GT class: the pixel flips to GT once
    # the GT logit overtakes the current top-1.
    wrong_margin = base.gather(0, top1_class[None])[0] - gt_logit

    base_row = body.section.codes[int(pair)].astype(np.int64)
    rows = []
    for dim in range(fe1.FRAME_DIM):
        scale = float(body.section.scales[dim])
        d_margin = jac[dim].gather(0, top1_class[None])[0] - jac[dim].gather(
            0, top2.indices[1][None]
        )[0]
        d_wrong = jac[dim].gather(0, top1_class[None])[0] - jac[dim].gather(
            0, gt[None]
        )[0]
        for code in range(fe1.CODE_MIN, fe1.CODE_MAX + 1):
            if code == int(base_row[dim]):
                continue
            delta = (code - int(base_row[dim])) * scale
            repaired = int(
                ((~correct) & ((wrong_margin + delta * d_wrong) < 0)).sum()
            )
            broken = int((correct & ((margin + delta * d_margin) < 0)).sum())
            rows.append(
                {
                    "dim": dim,
                    "old": int(base_row[dim]),
                    "new": int(code),
                    "delta_frame": round(delta, 6),
                    "predicted_repaired": repaired,
                    "predicted_broken": broken,
                    "predicted_delta": broken - repaired,
                }
            )
    return rows


def cmd_sign(args) -> int:
    fe1._set_threads(args.threads)
    body = fe1.load_body(with_raw=False, verify_shas=not args.no_sha)
    pairs = [int(p) for p in args.pairs.split(",")]
    out_rows = []
    started = time.time()
    for pair in pairs:
        linear = linearize_pair(body, pair)
        predicted = predict_moves(body, pair, linear)
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        # per-dim sign summary: the best predicted direction for each dim
        signs = []
        for dim in range(fe1.FRAME_DIM):
            here = [r for r in predicted if r["dim"] == dim]
            best = min(here, key=lambda r: r["predicted_delta"])
            signs.append(
                {
                    "dim": dim,
                    "best_new": best["new"],
                    "direction": int(np.sign(best["new"] - best["old"])),
                    "predicted_delta": best["predicted_delta"],
                }
            )
        out_rows.append(
            {
                "pair": pair,
                "base_flips": base_flips,
                "base_row": body.section.codes[pair].tolist(),
                "per_dim_sign": signs,
                "predicted": predicted,
            }
        )
        print(
            f"pair {pair}: base {base_flips} best predicted delta "
            f"{min(r['predicted_delta'] for r in predicted)}",
            flush=True,
        )
    result = {
        "schema": "ddm_fe1_sign.v1",
        "axis": "[macOS-CPU advisory, jg1/sj1 instrument, DALI GT lineage]",
        "score_claim": False,
        "linearization": (
            "exact through renderer + bilinear resample + SegNet; straight-through "
            "surrogate for the receiver's uint8 round at 874x1164"
        ),
        "pairs": pairs,
        "elapsed_s": round(time.time() - started, 1),
        "rows": out_rows,
        "receipts": body.receipts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sign = sub.add_parser("sign", help="linearize and predict every single-code move")
    sign.add_argument("--pairs", default="118,238,452")
    sign.add_argument("--threads", type=int, default=3)
    sign.add_argument("--no-sha", action="store_true")
    sign.add_argument(
        "--out", default=str(fe1.WORK / "probe/SIGN.json")
    )
    sign.set_defaults(func=cmd_sign)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
