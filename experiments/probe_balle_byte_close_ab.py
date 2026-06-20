# SPDX-License-Identifier: MIT
"""BYTE-CLOSE A/B for the Ballé weight-entropy lever — the empirical bit-spend
proof (Catalog #304). $0, CPU-only, separate temp dirs. Does NOT touch the live run.

THE QUESTION (the #1 NO-FAKE deliverable): the lever measurably lowers the
ORDER-0 (Shannon, exact-histogram) symbol entropy H. But the deployed coder in
this torch-vehicle is the vendored ``codec.build_archive`` = zigzag(int8) +
brotli quality 11 on the WHOLE concatenated decoder state-dict (NOT a per-tensor
order-0 range coder). brotli is LZ77 + context-modeled Huffman, so it can exploit
structure BEYOND order-0. Therefore the translation "lower order-0 H ⇒ fewer
archive bytes" is an OPEN empirical question that ONLY the real archive answers.

This script runs two driver runs (bit-shared init: same seed, same arch),
differing ONLY by λ, and measures:

  * the REAL ``best/best_archive.bin`` (the EMA-shadow score surface) byte size
    AND its decoder-blob byte size (the 3-section grammar's middle section)
  * order-0 H (bits/weight) of the EMA shadow + the LIVE final decoder
  * a LIVE-decoder archive built via the SAME vendored codec (the weights the
    penalty directly shaped)
  * byte-closed d_seg / d_pose / score from ``best_meta.json``

The headline number is ΔTotalBytes(λ>0 − λ=0) and ΔS at equal d_seg/d_pose.
If the order-0 H cut does NOT translate to fewer brotli bytes, that is a real
finding that downgrades the lever.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext
from tac.torch_vehicle.weight_entropy_penalty import (
    measure_decoder_weight_symbol_entropy,
)


def _ce(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage(epochs: int, lr: float, ema_decay: float = 0.999) -> StageSpec:
    return StageSpec(
        name="byte_close_ab",
        epochs=epochs,
        seg_loss_fn=_ce,
        eval_every=max(1, epochs // 4),
        batch_size=8,
        ema_decay=ema_decay,
        use_muon=False,
        adamw_lr=lr,
        muon_lr=2e-4,
        muon_weight_decay=0.0,
        latent_lr_mult=10.0,
        grad_clip=1e9,
        grad_clip_muon=1e9,
        lr_floor_ratio=5e-6,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=False,
        init_latents_random=True,
    )


def _decoder_blob_bytes(archive: bytes) -> int:
    buf = io.BytesIO(archive)
    meta_len = struct.unpack("<I", buf.read(4))[0]
    buf.read(meta_len)
    return struct.unpack("<I", buf.read(4))[0]


def _run(lam: float, *, n_pairs: int, epochs: int, lr: float, seed: int,
         ema_decay: float = 0.999, waterfill: bool = False):
    out = Path(tempfile.mkdtemp(prefix=f"balle_byteclose_lam{int(lam)}_"))
    cfg = TorchVehicleConfig(
        base_channels=20,
        latent_dim=28,
        out_dir=out,
        device="cpu",
        seed=seed,
        ema_decay=ema_decay,
        weight_entropy_penalty_lambda=lam,
        weight_entropy_penalty_stage_min=0,
        weight_entropy_penalty_waterfill=waterfill,
    )
    sc = SyntheticScorerContext(n_pairs=n_pairs, device="cpu", seed=seed)
    drv = TorchVehicleDriver(
        cfg, scorer=sc, vendored=import_vendored_bundle(),
        curriculum=[_stage(epochs, lr, ema_decay=ema_decay)]
    )
    summary = drv.run()
    return drv, sc, out, summary


def _measure(lam, drv, sc, out: Path, summary):
    best_dir = out / "best"
    best_archive = (best_dir / "best_archive.bin").read_bytes()
    best_meta = json.loads((best_dir / "best_meta.json").read_text()) if (best_dir / "best_meta.json").exists() else {}
    ema_sd = torch.load(best_dir / "best_ema_decoder.pt", map_location="cpu", weights_only=False)
    ema_latents = torch.load(best_dir / "best_ema_latents.pt", map_location="cpu", weights_only=False)

    # order-0 H of the EMA shadow (the score surface) — rebuild a vendored decoder to host the sd.
    ema_dec = drv._new_vendored_decoder(device=torch.device("cpu"))
    ema_dec.load_state_dict({k: v.to("cpu") for k, v in ema_sd.items()})
    h_ema = measure_decoder_weight_symbol_entropy(ema_dec)

    # order-0 H + a LIVE-decoder archive (the weights the penalty directly shaped).
    live = drv._final_decoder
    h_live = measure_decoder_weight_symbol_entropy(live)
    live_sd = {k: t.detach().cpu().clone() for k, t in live.state_dict().items()}
    live_archive = drv.v.build_archive(live_sd, ema_latents.detach().cpu(), meta_dict={})

    return {
        "best_score": summary.get("best_score"),
        "best_archive_total_bytes": len(best_archive),
        "best_archive_decoder_bytes": _decoder_blob_bytes(best_archive),
        "ema_order0_H": h_ema,
        "live_order0_H": h_live,
        "live_archive_total_bytes": len(live_archive),
        "live_archive_decoder_bytes": _decoder_blob_bytes(live_archive),
        "d_seg": best_meta.get("seg_distortion", best_meta.get("d_seg")),
        "d_pose": best_meta.get("pose_distortion", best_meta.get("d_pose")),
        "score": best_meta.get("score"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambdas", type=str, default="0,50")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--waterfill", action="store_true")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    rows: dict[str, dict] = {}
    for lam in lambdas:
        drv, sc, out, summary = _run(
            lam, n_pairs=args.n_pairs, epochs=args.epochs, lr=args.lr, seed=args.seed,
            ema_decay=args.ema_decay, waterfill=(args.waterfill and lam > 0),
        )
        m = _measure(lam, drv, sc, out, summary)
        rows[f"lam{lam:g}"] = m
        print(f"[lam={lam:g}] {json.dumps(m)}")

    base = rows.get("lam0")
    print("\n=== BYTE-CLOSE A/B (vendored zigzag+brotli q11 codec) ===")
    if base:
        for k, r in rows.items():
            if k == "lam0":
                continue
            print(f"\n{k} vs lam0:")
            print(f"  Δema_order0_H        = {r['ema_order0_H'] - base['ema_order0_H']:+.4f} bits/wt")
            print(f"  Δlive_order0_H       = {r['live_order0_H'] - base['live_order0_H']:+.4f} bits/wt")
            dDec = r["best_archive_decoder_bytes"] - base["best_archive_decoder_bytes"]
            dTot = r["best_archive_total_bytes"] - base["best_archive_total_bytes"]
            print(f"  Δbest_decoder_bytes  = {dDec:+d}  ({base['best_archive_decoder_bytes']} -> {r['best_archive_decoder_bytes']})")
            print(f"  Δbest_total_bytes    = {dTot:+d}  ({base['best_archive_total_bytes']} -> {r['best_archive_total_bytes']})")
            dLiveDec = r["live_archive_decoder_bytes"] - base["live_archive_decoder_bytes"]
            print(f"  Δlive_decoder_bytes  = {dLiveDec:+d}  ({base['live_archive_decoder_bytes']} -> {r['live_archive_decoder_bytes']})")
            if r.get("d_seg") is not None and base.get("d_seg") is not None:
                print(f"  Δd_seg               = {r['d_seg'] - base['d_seg']:+.6f}")
                print(f"  Δd_pose              = {r['d_pose'] - base['d_pose']:+.6f}")
            if r.get("score") is not None and base.get("score") is not None:
                print(f"  Δscore               = {r['score'] - base['score']:+.6f}")
            print(f"  VERDICT: best-archive H-cut translates to fewer bytes = {dDec < 0}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
