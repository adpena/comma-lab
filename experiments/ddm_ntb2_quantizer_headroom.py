"""How much of the 3-bit damage is the SCALE RULE, and how much is the bit budget?

The shipped renderer section quantizes each tensor with `quantized_components`: one
fp16 per-output-axis scale set to `max|w| / limit`. A max-absolute scale is optimal for
the WORST element and bad for the mean when the row is heavy-tailed, so a 3-bit verdict
drawn on it would be a verdict on that scale rule, not on three bits.

This probe measures, per 4-bit tensor, the relative weight-space error of
  (a) the SHIPPED 4-bit max-absolute representation -- the reference the render already
      tolerates,
  (b) 3-bit max-absolute -- what ntb2's retained packets contain,
  (c) 3-bit with a per-row clip ratio chosen to MINIMISE that row's squared error.
(c) is the best a scale-rule change can do at three bits without touching the receiver
format or re-fitting any weight.

This is WEIGHT SPACE, a proxy for the render: it bounds how much headroom the scale rule
has, and it may NOT be read as d_seg, d_pose, or score. The verdict instrument remains
`ddm_ntb2_renderer_score.py` at n600.

Axis: [weight-space proxy] -- headroom only. score_claim=false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "src")]

import ddm_jg1_seg_solve as jg1

from experiments import ddm_ntb2_control as control
from experiments import ddm_ntb2_renderer_score as score


def relative_error(original, restored) -> float:
    import torch

    num = torch.linalg.vector_norm(restored - original).item()
    den = torch.linalg.vector_norm(original).item()
    return num / den if den else 0.0


def best_clip_quantize(value, bits: int, ratios):
    """Per-row scale = ratio * max|row| / limit, ratio chosen per row by squared error.

    CAVEAT, and it is load-bearing for two rows of the table: `quantized_components`
    reduces over the LEADING dims for any tensor whose name ends in ``embed.weight``
    (a per-column scale) and over the trailing dims otherwise (a per-row scale). This
    helper always uses the per-ROW axis, so for ``token_embed.weight`` and
    ``frame_embed.weight`` column (c) is NOT the same scale family as column (b) and
    can legitimately come out WORSE. Those two tensors are worth 16 B and 0 B on this
    lever, so the axis is not fixed here -- the discrepancy is declared instead.
    """
    import torch

    limit = (1 << (bits - 1)) - 1
    flat = value.detach().cpu().float().reshape(value.shape[0], -1)
    peak = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    best = None
    best_err = None
    for ratio in ratios:
        scales = (peak * ratio / limit).to(torch.float16).float().clamp_min(1e-8)
        codes = (flat / scales).round().clamp(-limit, limit)
        restored = codes * scales
        err = ((restored - flat) ** 2).sum(dim=1, keepdim=True)
        if best is None:
            best, best_err = restored, err
        else:
            take = err < best_err
            best = torch.where(take, restored, best)
            best_err = torch.where(take, err, best_err)
    return best.reshape(value.shape)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=score.STORE / "QUANTIZER_HEADROOM.json")
    args = parser.parse_args(argv)

    import torch

    from experiments.ddm_sm3_semantic_representation import quantized_components

    torch.set_num_threads(2)
    tree = control.ROOT / "source_runtime"
    base_sha = score.sha256_file(tree / "archive.zip")
    if base_sha != control.BASE_SHA:
        raise SystemExit(f"base runtime archive moved: {base_sha}")
    semantic = jg1.load_semantic_renderer(
        archive_path=tree / "archive.zip", runtime_dir=tree / "runtime"
    )
    shipped = semantic.state_dict()

    ratios = [round(0.30 + 0.02 * i, 2) for i in range(36)]  # 0.30 .. 1.00
    rows = []
    for name, value in shipped.items():
        if value.ndim < 2:
            continue
        # The SHIPPED weights are already the dequantized 4-bit values, so re-quantizing
        # them at 4 bits is a NEAR-IDENTITY and its residual is the fp16-scale rounding
        # only.  The honest reference for "what the render tolerates" is therefore the
        # 3-bit error measured against these same shipped values.
        q4, _, _ = quantized_components(name, value, bits=4)
        q3, _, _ = quantized_components(name, value, bits=3)
        q3_best = best_clip_quantize(value, 3, ratios)
        rows.append(
            {
                "tensor": name,
                "numel": int(value.numel()),
                "rel_err_q4_maxabs": relative_error(value, q4),
                "rel_err_q3_maxabs": relative_error(value, q3),
                "rel_err_q3_best_clip": relative_error(value, q3_best),
                "scale_rule_gain": (
                    relative_error(value, q3) / relative_error(value, q3_best)
                    if relative_error(value, q3_best)
                    else None
                ),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    result = {
        "schema": "ddm_ntb2_quantizer_headroom.v1",
        "axis": "[weight-space proxy] headroom only",
        "score_claim": False,
        "promotion_eligible": False,
        "status": "PROXY -- bounds the scale-rule headroom; never a distortion or score row",
        "base_archive_sha256": base_sha,
        "clip_ratios": ratios,
        "producer": score.fact(Path(__file__)),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    score.retain(args.out, (json.dumps(result, indent=1, sort_keys=True) + "\n").encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
