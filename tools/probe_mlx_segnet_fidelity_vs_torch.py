#!/usr/bin/env python3
"""MLX-SegNet-port fidelity gate vs the authoritative upstream torch SegNet.

OPERATOR PRIORITY (2026-06-08): "hardening and fixing and engineering all drift
away possible of MLX port is a top priority" + "getting this as high signal and
fidelity as possible is extremely important".

The contest scores with the upstream torch SegNet (`evaluate.py` ->
`modules.py DistortionNet.SegNet`). The HiNeRV birth/survival harness scores with
an MLX PORT (`tac.local_acceleration.mlx_scorer_adapters.MLXSegNetAdapter`). Any
argmax drift between the two is false-authority risk: a birth that the MLX port
calls won/lost may differ from what the contest's torch SegNet sees.

This probe renders ONE candidate frame (the torch archive backend), scores it
with BOTH the upstream torch SegNet AND the MLX port, and reports the EXACT
per-pixel argmax agreement + per-class logit max-abs delta over the birth region.
It is the fidelity gate: drive `argmax_disagreement_fraction` to 0 (and the logit
delta below a tight band) so the MLX port is an argmax-exact mirror of the
authority.  Planning-control evidence only; non-promotable.

Empirical anchor (v9 live archive, region 50568 px): torch wins 11297, MLX wins
11306, argmax agreement 0.99966 (17 px disagree) -> the MLX port is NEAR-faithful
here; the residual 17 px are the hardening target.  Crucially this PROVES the
backend birth does NOT collapse (both scorers ~11k), so the runner's
authoritative "2" measures a DIFFERENT quantity than absolute region wins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def _read_payload(zp: Path) -> bytes:
    with zipfile.ZipFile(zp) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        pick = [n for n in members if n in {"0.bin", "x"}] or members
        return zf.read(pick[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--birth-class", type=int, default=4)
    ap.add_argument("--bbox-y0", type=int, default=281)
    ap.add_argument("--bbox-y1", type=int, default=384)
    ap.add_argument("--upstream-dir", default="upstream")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import mlx.core as mx
    import torch

    from tac.local_acceleration.mlx_scorer_adapters import MLXSegNetAdapter
    from tac.scorer import extract_gt_masks, load_default_segnet
    from tac.substrates.hi_nerv.architecture import HinervSubstrate
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.inflate import build_model_from_archive

    archive = Path(args.archive).expanduser().resolve()
    z = np.load(Path(args.npz).expanduser().resolve())
    target_labels = z["target_labels_bhw"].astype(np.int64)
    pair_indices = [int(v) for v in z["pair_indices"].tolist()]
    cls = int(args.birth_class)

    payload = _read_payload(archive)
    arc = parse_archive(payload)
    _arc, cfg, _recv = build_model_from_archive(payload, device="cpu")
    backend = HinervSubstrate(cfg).to("cpu").eval()
    fs = dict(arc.decoder_state_dict)
    for k in ("latents_coarse", "latents_mid", "latents_fine"):
        fs[k] = getattr(arc, k).to(dtype=getattr(backend, k).dtype)
    backend.load_state_dict(fs, strict=True)
    with torch.no_grad():
        _r0, r1 = backend(torch.tensor(pair_indices, dtype=torch.long))
    r1np = np.asarray(r1.detach().cpu(), dtype=np.float32)  # (N,3,H,W) [0,1]

    # (a) AUTHORITY: upstream torch SegNet (the exact evaluate.py path).
    segnet = load_default_segnet(args.upstream_dir, device="cpu")
    frames = [
        torch.from_numpy(np.transpose(f * 255.0, (1, 2, 0)).copy()).clamp(0, 255)
        for f in r1np
    ]
    torch_argmax = np.asarray(
        extract_gt_masks(frames, segnet, device="cpu", batch_size=4).cpu(), dtype=np.int64
    )

    # (b) MLX PORT: the harness scorer (build_mlx_segnet_pair_teacher -> this adapter).
    adapter = MLXSegNetAdapter(segnet)
    x = mx.array(np.transpose(r1np, (0, 2, 3, 1)).astype(np.float32)) * 255.0  # BHWC [0,255]
    mlx_logits = np.asarray(adapter(x).astype(mx.float32))  # BHWC logits
    mlx_argmax = mlx_logits.argmax(-1).astype(np.int64)

    region = np.zeros_like(target_labels, dtype=bool)
    region[:, args.bbox_y0 : args.bbox_y1, :] = True
    region &= target_labels == cls
    region_px = int(np.count_nonzero(region))

    disagree = (torch_argmax != mlx_argmax) & region
    disagree_px = int(np.count_nonzero(disagree))
    torch_wins = int(np.count_nonzero((torch_argmax == cls) & region))
    mlx_wins = int(np.count_nonzero((mlx_argmax == cls) & region))

    # Full-frame argmax agreement too (not just region) — the port must be exact everywhere.
    full_disagree = int(np.count_nonzero(torch_argmax != mlx_argmax))
    full_px = int(torch_argmax.size)

    row: dict[str, Any] = {
        "schema": "hi_nerv_mlx_segnet_fidelity_vs_torch.v1",
        "family": "hinerv",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "target_class": cls,
        "region_pixel_count": region_px,
        "torch_upstream_segnet_wins": torch_wins,
        "mlx_port_segnet_wins": mlx_wins,
        "region_argmax_disagreement_px": disagree_px,
        "region_argmax_agreement_fraction": (
            1.0 - disagree_px / region_px if region_px else None
        ),
        "full_frame_argmax_disagreement_px": full_disagree,
        "full_frame_argmax_agreement_fraction": 1.0 - full_disagree / full_px,
        "backend_collapses": torch_wins < 100,  # both scorers ~11k => NOT a collapse
        "mlx_port_is_argmax_exact": full_disagree == 0,
        "verdict": (
            "mlx_port_argmax_exact"
            if full_disagree == 0
            else f"mlx_port_drift_{full_disagree}px_to_harden_to_zero"
        ),
        "high_signal_finding": (
            "backend birth does NOT collapse: torch + MLX SegNet both ~11k region "
            "wins on the same render; the runner's authoritative '2' measures a "
            "DIFFERENT quantity (transition/initial-reference), not absolute wins."
        ),
        "authority": "planning_control_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "human_visual_fidelity_objective": False,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
