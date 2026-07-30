#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_qp1 QA05 — renderer rank-1 STRUCTURED bias-edit probe (ru1 op-routable 3;
gc6 §4 row 11; Assumption-Adversary).

QUESTION (the "token-lattice-only" hole): is the seg residual (tier-2) addressable
ONLY per-pair via token edits, or does a GLOBAL rank-1 renderer-level bias — a
single spatial×color pattern applied to the rendered frame_1 across ALL pairs at
~0 marginal bytes — reduce realized d_seg flips? A rank-1 renderer-section edit is
proxied here in OUTPUT space (add the bias to the rendered uint8 frame_1). Output
space is the MOST permissive form (a renderer-section edit is a strict subset of
expressible output changes), so a null here is an UPPER-BOUND falsifier: if even a
free output-space rank-1 bias cannot net-reduce flips beyond the noise floor, a
constrained renderer edit cannot either.

DISTINCT from the P2b diagonal-ES class (retyped variance-artifact/INSTANCE): this
uses STRUCTURED directions derived from the ru1 atlas flip-mass (where flips
concentrate) + a finite-difference color search, NOT random-noise ES on the
renderer stream. It is scored against a matched RANDOM-CONTROL noise floor (the
ES-noise floor analogue): K random rank-1 patterns at matched amplitude/eval
budget.

FALSIFIER (pre-registered): best STRUCTURED net-flips <= the random-control noise
floor (max over controls, and mean+2sigma) => tier-2 is token-only at
INSTANCE(rank-1-bias) scope. best structured net > noise floor => a global
renderer-level seg lever EXISTS (tier-2 NOT token-only) at ~0 B/flip.

Realized through the real render + STE-uint8 + frozen CPU-torch SegNet argmax on a
BOUNDED top-flip pair subset (<=120, chunked). Base = pfs1 D1 seg endpoint
(p2c_aimed, b9a7983b).

Axis: [macOS-CPU advisory]; score_claim=false; promotion_eligible=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SEG_PX = 196608
SEG_H, SEG_W = 384, 512     # atlas y,x live here
SEG_BASE = Path("/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2c_aimed_archive.zip")
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
ATLAS = Path("/Volumes/VertigoDataTier/pact/ddm_ru1_20260729/atlas_flat.npz")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def top_flip_pairs(k: int) -> list[int]:
    a = np.load(ATLAS)
    pair = a["pair"].astype(np.int64)
    counts = np.bincount(pair, minlength=600)
    return [int(p) for p in np.argsort(-counts)[:k]]


def flip_mass_mask(cam_h: int, cam_w: int) -> np.ndarray:
    """Spatial flip-density mask at camera res (max-normalized to 1), built from the
    ru1 atlas (y,x in SegNet 384x512) and nearest-upsampled to camera res."""
    a = np.load(ATLAS)
    y = a["y"].astype(np.int64)
    x = a["x"].astype(np.int64)
    hist = np.zeros((SEG_H, SEG_W), dtype=np.float64)
    np.add.at(hist, (y, x), 1.0)
    # light smoothing (3x3 box) so the mask is not a spike field
    k = np.ones((3, 3)) / 9.0
    hp = np.pad(hist, 1, mode="edge")
    sm = sum(hp[i:i + SEG_H, j:j + SEG_W] * k[i, j]
             for i in range(3) for j in range(3))
    sm /= max(sm.max(), 1e-9)
    yi = (np.arange(cam_h) * SEG_H // cam_h).clip(0, SEG_H - 1)
    xi = (np.arange(cam_w) * SEG_W // cam_w).clip(0, SEG_W - 1)
    return sm[np.ix_(yi, xi)]  # (cam_h, cam_w) in [0,1]


def build_patterns(cam_h: int, cam_w: int, n_random: int, seed: int) -> dict:
    """rank-1 patterns pat(H,W,3) normalized so max|pat| == 1 (amp = peak bias in
    uint8 levels). Structured = atlas flip-mass x color dirs; generic = const/ramp;
    random controls = smooth-random spatial x random unit color."""
    mask = flip_mass_mask(cam_h, cam_w)[:, :, None]  # (H,W,1)
    colors = {
        "white": np.array([1.0, 1.0, 1.0]),
        "r": np.array([1.0, 0.0, 0.0]),
        "g": np.array([0.0, 1.0, 0.0]),
        "b": np.array([0.0, 0.0, 1.0]),
        "lum": np.array([0.299, 0.587, 0.114]),
        "rb": np.array([1.0, 0.0, -1.0]),
    }
    pats: dict[str, np.ndarray] = {}
    # STRUCTURED: atlas flip-mass mask x each color direction
    for cn, cv in colors.items():
        p = mask * cv[None, None, :]
        pats[f"atlas_{cn}"] = p / max(np.abs(p).max(), 1e-9)
    # GENERIC spatial controls (the existing qa05 family)
    yv = np.linspace(-1.0, 1.0, cam_h)[:, None, None]
    xv = np.linspace(-1.0, 1.0, cam_w)[None, :, None]
    ones = np.ones((cam_h, cam_w, 1))
    pats["gen_const_all"] = (ones * colors["white"][None, None, :])
    pats["gen_vramp"] = (yv * ones * colors["white"][None, None, :])
    pats["gen_vramp"] /= max(np.abs(pats["gen_vramp"]).max(), 1e-9)
    pats["gen_hramp"] = (xv * ones * colors["white"][None, None, :])
    pats["gen_hramp"] /= max(np.abs(pats["gen_hramp"]).max(), 1e-9)
    pats["gen_const_all"] /= max(np.abs(pats["gen_const_all"]).max(), 1e-9)
    # RANDOM CONTROLS (ES-noise floor analogue): smooth random spatial x random color
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        # low-freq smooth field via random Fourier-ish coarse grid upsample
        coarse = rng.standard_normal((6, 8))
        yi = (np.arange(cam_h) * 6 // cam_h).clip(0, 5)
        xi = (np.arange(cam_w) * 8 // cam_w).clip(0, 7)
        sf = coarse[np.ix_(yi, xi)][:, :, None]
        cv = rng.standard_normal(3)
        cv /= max(np.linalg.norm(cv), 1e-9)
        p = sf * cv[None, None, :]
        pats[f"rand_{i:02d}"] = p / max(np.abs(p).max(), 1e-9)
    return pats


def run(args) -> None:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "upstream"))
    sys.path.insert(0, str(REPO / "tools"))
    import torch
    torch.set_num_threads(int(args.threads))
    from sb1_seg_batch import SegRuntime

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "qa05_candidates.jsonl"

    rtp = SegRuntime(SEG_BASE, GT_CACHE)
    pairs = top_flip_pairs(args.subset)
    base_frames = {p: np.asarray(rtp.render_uint8(p), dtype=np.int16) for p in pairs}
    gts = {p: np.asarray(rtp.lstars[p], dtype=np.int64) for p in pairs}
    cam_h, cam_w, _ = base_frames[pairs[0]].shape

    def subset_flips(bias: np.ndarray) -> int:
        frames = [np.clip(base_frames[p] + bias, 0, 255).astype(np.uint8) for p in pairs]
        gl = [gts[p] for p in pairs]
        ds = rtp._verdict(rtp.seg, frames, gl)
        return round(sum(float(x) for x in ds) * SEG_PX)

    pats = build_patterns(cam_h, cam_w, args.n_random, args.seed)
    amps = args.amps
    processed = set()
    if jsonl.exists():
        for ln in jsonl.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                processed.add((r["pattern"], r["amp"]))
        print(f"[resume] {len(processed)} candidates", flush=True)

    t0 = time.time()
    base = subset_flips(np.zeros((cam_h, cam_w, 3), dtype=np.int16))
    print(f"[base] subset {len(pairs)} pairs, {base} flips ({time.time()-t0:.0f}s)", flush=True)
    for name, pat in pats.items():
        for amp in amps:
            if (name, amp) in processed:
                continue
            bias = np.round(pat * amp).astype(np.int16)
            f = subset_flips(bias)
            row = {"pattern": name, "amp": amp, "subset_base_flips": base,
                   "subset_flips": f, "net": base - f,
                   "kind": ("structured" if name.startswith("atlas_") else
                            "generic" if name.startswith("gen_") else "random")}
            with jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{name}@{amp:+d}] net {base - f:+d} (base {base}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

    rows = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]
    struct = [r for r in rows if r["kind"] == "structured"]
    generic = [r for r in rows if r["kind"] == "generic"]
    rand = [r for r in rows if r["kind"] == "random"]
    rand_net = np.array([r["net"] for r in rand], dtype=np.float64)
    noise_max = float(rand_net.max()) if len(rand_net) else 0.0
    noise_mean = float(rand_net.mean()) if len(rand_net) else 0.0
    noise_std = float(rand_net.std()) if len(rand_net) else 0.0
    noise_floor = noise_mean + 2.0 * noise_std
    best_struct = max(struct, key=lambda r: r["net"]) if struct else None
    best_generic = max(generic, key=lambda r: r["net"]) if generic else None
    beats = bool(best_struct and best_struct["net"] > max(noise_max, noise_floor))
    verdict = ("GLOBAL_RENDERER_SEG_LEVER_EXISTS_tier2_NOT_token_only" if beats
               else "tier2_token_only_CONFIRMED_at_INSTANCE_rank1_bias")
    receipt = {
        "schema": "ddm_qp1_qa05_structured_rank1_probe.v1",
        "item": "QA05 renderer rank-1 STRUCTURED bias-edit probe (ru1 op-routable 3)",
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "base_archive_sha256": _sha(SEG_BASE.read_bytes()),
        "subset_pairs": len(pairs), "subset_base_flips": base,
        "amps": amps, "n_random_controls": args.n_random,
        "best_structured": best_struct, "best_generic": best_generic,
        "noise_floor": {"max": noise_max, "mean": noise_mean, "std": noise_std,
                        "mean_plus_2sigma": noise_floor},
        "structured_beats_noise_floor": beats,
        "verdict": verdict,
        "b_per_flip_note": ("a rank-1 renderer bias is ~0 marginal bytes (renderer.sec "
                            "already counted); if net>0 the lever is ~0 B/flip. net<=0 => "
                            "no lever at any price on this form."),
        "verdict_scope": "INSTANCE (rank-1-bias output-space proxy, this subset/scorer/endpoint)",
        "wall_seconds": time.time() - t0,
        "generated_by": "tools/qp1_qa05_structured.py",
    }
    _atomic_write(out / "qa05_receipt.json", json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    bs = best_struct["net"] if best_struct else 0
    print(f"[QA05 done] best_structured_net {bs:+d} noise_floor(max {noise_max:+.0f} "
          f"m+2s {noise_floor:+.1f}) beats={beats} => {verdict} ({time.time()-t0:.0f}s)",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--subset", type=int, default=96)
    ap.add_argument("--amps", type=int, nargs="+", default=[-4, -2, -1, 1, 2, 4])
    ap.add_argument("--n-random", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
