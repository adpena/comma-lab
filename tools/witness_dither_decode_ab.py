#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""P-DITHER instrument: $0 A/B of the B19 decode-side seeded dither on a byte-close packet.

LAW + BAND PROVENANCE (T5 crucible, DRAFT_OPTIMAL_STACK_v6 SS7c, pre-registered — honored
exactly): B19 gate = dither the EXISTING mod32cap ep650 byte-close decode (seeded,
OFF-identical), n600 verdict A/B vs undithered. **fire: Delta d_seg <= -1e-5** (>= 0.56x
crossing margin in S) at unchanged bytes; **kill THIS FORM: Delta d_seg >= 0** (decode-side
zeroth-order dither stays unadmitted; trained-with dither = the named reformulation, run-2).
Pre-registered A/B config (probe_tau2_dither_20260708.md): mode=bayer8, amp=1.0 quantum,
seed=0xB19 (tac.witness_control.decode_dither defaults).

AUTHORITY (NO-FAKE): the dithered arm runs the PACKET'S OWN inflate.py (`_setup` /
`_init_worker` / `_render_pair` — the op-for-op shipped decode body, same pattern as the R6
parity driver) with EXACTLY ONE change: `_R` gains the additive pre-round dither term
(`tac.witness_control.decode_dither.dither_offset`). amp=0 reproduces the undithered path
bit-for-bit (OFF-identical by construction: +0.0 then identical round/clamp). The verdict is
the frozen CPU-torch `cpu_verdict_d_seg_argmax_batch` (the trainer's own n600 verdict
authority). Every number [macOS-CPU advisory]; NO score claims; archive bytes UNCHANGED by
construction (decode-side; rate sizes archive.zip only).

KEYING: this packet has pose_carrier=None (verified), so `_R` fires EXACTLY twice per
`_render_pair(pi)` — fk=0 then fk=1 — making the (pair, frame) call-counter keying exact.
The driver REFUSES packets whose manifest carries a pose_carrier (keying would be ambiguous).

Phases (all foreground, chunked + resumable — the harness kills ~5-min calls):
  decode  : dithered decode over a pair range -> <out>/dithered.raw (state jsonl resume)
  verdict : SegNet d_seg + realized argmax over a pair range for one arm (rows jsonl +
            argmax int8 memmap; resume by skipping recorded pairs)
  compare : Delta d_seg (full precision), per-class-pair fixed/created split, far-range-lane
            row band (176-224), GT-margin bands, undithered-vs-R6 bit-for-bit validation

Usage (chunked):
  .venv/bin/python tools/witness_dither_decode_ab.py decode  --packet <pkt> --out <dir> --start 0 --end 150
  .venv/bin/python tools/witness_dither_decode_ab.py verdict --arm dithered   --raw <dir>/dithered.raw --out <dir> --gt-cache <npz> --start 0 --end 96
  .venv/bin/python tools/witness_dither_decode_ab.py verdict --arm undithered --raw <pkt>/inflated/0.raw --out <dir> --gt-cache <npz> --start 0 --end 96
  .venv/bin/python tools/witness_dither_decode_ab.py compare --out <dir> --gt-cache <npz> --r6-rows <pkt>/inflated/r6_verdict_pairs.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.witness_control.decode_dither import (  # noqa: E402
    DEFAULT_AMP,
    DEFAULT_MODE,
    DEFAULT_SEED,
    dither_offset,
)

N_CLASSES = 5
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
LANE_ROWS = (176, 224)  # P-DZ census far-range-lane concentration band (scorer 384-row space)
ADVISORY = "[macOS-CPU advisory . NON-PROMOTABLE]"

# worker globals (spawn-safe: set by _worker_init in each worker process)
_W: dict = {}


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_packet_module(packet_dir: Path):
    p = packet_dir / "inflate.py"
    spec = importlib.util.spec_from_file_location("pkt_inflate_dither_ab", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_dither(mod, amp: float, mode: str, seed: int):
    """Replace mod._R with the dithered form (op-for-op + the additive pre-round term)."""
    import torch

    if mod._G.get("m", {}).get("pose_carrier") is not None:
        raise SystemExit("REFUSE: packet has a pose_carrier — (pair,frame) call-counter keying "
                         "is ambiguous for this manifest; extend the keying before dithering.")

    ctx = {"pi": -1, "k": 0}
    orig_render_pair = mod._render_pair

    def _R_dithered(rgb, rh, rw, ch, cw):
        x = torch.from_numpy(np.ascontiguousarray(rgb.reshape(rh, rw, 3))).permute(2, 0, 1)[None].float()
        with torch.inference_mode():
            up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
            off = dither_offset(ch, cw, amp=amp, mode=mode, seed=seed,
                                pair_idx=ctx["pi"], frame_idx=ctx["k"])
            up = up + torch.from_numpy(off).permute(2, 0, 1)[None]
            up = torch.clamp(torch.round(up), 0.0, 255.0)
        ctx["k"] += 1
        return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)

    def _render_pair_dithered(pi):
        ctx["pi"] = pi
        ctx["k"] = 0
        return orig_render_pair(pi)

    mod._R = _R_dithered
    mod._render_pair = _render_pair_dithered


def _worker_init(packet_dir: str, src: str, dst: str, amp: float, mode: str, seed: int):
    mod = _load_packet_module(Path(packet_dir))
    mod._setup(src)
    mod._G["dst"] = dst
    _patch_dither(mod, amp, mode, seed)
    _W["mod"] = mod


def _worker_render(pi: int) -> int:
    return _W["mod"]._render_pair(pi)


def cmd_decode(args):
    import multiprocessing as mp

    packet_dir = args.packet.resolve()
    src = str(packet_dir / "archive" / "0.bin")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "dithered.raw"
    state = out / "decode_state.jsonl"

    probe = _load_packet_module(packet_dir)
    probe._setup(src)
    m = probe._G["m"]
    n_pairs, fb = int(m["n_pairs"]), int(probe._G["framebytes"])
    if not dst.exists() or dst.stat().st_size != 2 * n_pairs * fb:
        with open(dst, "wb") as f:
            f.truncate(2 * n_pairs * fb)
    done = set()
    if state.exists():
        for line in state.read_text().splitlines():
            r = json.loads(line)
            done.update(range(r["range"][0], r["range"][1]))
    todo = [pi for pi in range(args.start, min(args.end, n_pairs)) if pi not in done]
    if not todo:
        print(f"[{_utc()}] decode range [{args.start},{args.end}) already done", flush=True)
        return
    t0 = time.time()
    nworkers = max(1, min(len(todo), args.workers))
    ctx = mp.get_context("spawn")
    with ctx.Pool(nworkers, initializer=_worker_init,
                  initargs=(str(packet_dir), src, str(dst), args.amp, args.mode, args.seed)) as pool:
        for _ in pool.imap_unordered(_worker_render, todo, chunksize=1):
            pass
    wall = round(time.time() - t0, 1)
    with open(state, "a") as f:
        f.write(json.dumps({"range": [args.start, args.end], "n": len(todo), "wall_s": wall,
                            "workers": nworkers, "amp": args.amp, "mode": args.mode,
                            "seed": args.seed}) + "\n")
    print(f"[{_utc()}] decoded pairs [{args.start},{args.end}) n={len(todo)} wall={wall}s "
          f"(amp={args.amp} mode={args.mode} seed={hex(args.seed)})", flush=True)


def cmd_verdict(args):
    from train_witness_realized_through_R_mlx import cpu_verdict_d_seg_argmax_batch

    from tac.boundary_math.seg_core import load_real_segnet

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"verdict_{args.arm}.jsonl"
    am_path = out / f"argmax_{args.arm}.i8"
    z = np.load(args.gt_cache, mmap_mode="r")
    n_pairs = int(z["lstars"].shape[0])
    h, w = int(z["lstars"].shape[1]), int(z["lstars"].shape[2])
    if not am_path.exists() or am_path.stat().st_size != n_pairs * h * w:
        with open(am_path, "wb") as f:
            f.truncate(n_pairs * h * w)
    am_mm = np.memmap(am_path, dtype=np.int8, mode="r+", shape=(n_pairs, h, w))
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            done.add(json.loads(line)["pi"])
    raw = np.memmap(args.raw, dtype=np.uint8, mode="r").reshape(2 * n_pairs, args.cam_h, args.cam_w, 3)
    seg = load_real_segnet("cpu")
    t0 = time.time()
    todo = [pi for pi in range(args.start, min(args.end, n_pairs)) if pi not in done]
    with open(rows_path, "a") as f:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            f1s = [np.asarray(raw[2 * pi + 1]) for pi in chunk]
            ls = [np.asarray(z["lstars"][pi]).astype(np.int64) for pi in chunk]
            d_seg, realized = cpu_verdict_d_seg_argmax_batch(seg, f1s, ls)
            for j, pi in enumerate(chunk):
                am_mm[pi] = realized[j].astype(np.int8)
                f.write(json.dumps({"pi": pi, "d_seg": d_seg[j]}) + "\n")
            f.flush()
            print(f"[{_utc()}] [{args.arm}] pairs {chunk[0]}..{chunk[-1]} "
                  f"mean_d_seg={float(np.mean(d_seg)):.8f} ({time.time() - t0:.0f}s)", flush=True)
    am_mm.flush()
    print(f"[{_utc()}] verdict[{args.arm}] range [{args.start},{args.end}) done", flush=True)


def _pair_key(g, wcls):
    return f"{CLASS_NAMES[int(g)]}->{CLASS_NAMES[int(wcls)]}"


def cmd_compare(args):
    out = args.out.resolve()
    z = np.load(args.gt_cache, mmap_mode="r")
    n_pairs = int(z["lstars"].shape[0])
    h, w = int(z["lstars"].shape[1]), int(z["lstars"].shape[2])
    am_u = np.memmap(out / "argmax_undithered.i8", dtype=np.int8, mode="r", shape=(n_pairs, h, w))
    am_d = np.memmap(out / "argmax_dithered.i8", dtype=np.int8, mode="r", shape=(n_pairs, h, w))

    def _rows(name):
        rows = {}
        for line in (out / f"verdict_{name}.jsonl").read_text().splitlines():
            r = json.loads(line)
            rows[r["pi"]] = r["d_seg"]
        return rows

    du, dd = _rows("undithered"), _rows("dithered")
    if len(du) != n_pairs or len(dd) != n_pairs:
        raise SystemExit(f"incomplete verdicts: undithered {len(du)}/{n_pairs}, dithered {len(dd)}/{n_pairs}")
    d_seg_u = float(np.mean([du[i] for i in range(n_pairs)]))
    d_seg_d = float(np.mean([dd[i] for i in range(n_pairs)]))
    delta = d_seg_d - d_seg_u

    # R6 bit-for-bit validation of the undithered arm (same helper, same raw).
    r6 = {}
    if args.r6_rows and Path(args.r6_rows).exists():
        for line in Path(args.r6_rows).read_text().splitlines():
            r = json.loads(line)
            r6[r["pi"]] = r["d_seg"]
    r6_match = bool(r6) and all(du[i] == r6.get(i) for i in range(n_pairs))

    fixed_cp: dict[str, int] = {}
    created_cp: dict[str, int] = {}
    n_fixed = n_created = 0
    band_fixed = band_created = 0
    mband_edges = [0.0, 0.1, 0.5, 2.0, np.inf]
    mband_fixed = np.zeros(len(mband_edges) - 1, np.int64)
    mband_created = np.zeros(len(mband_edges) - 1, np.int64)
    lane_fixed = lane_created = 0
    r0, r1 = LANE_ROWS
    for pi in range(n_pairs):
        g = np.asarray(z["lstars"][pi]).astype(np.int64)
        gm = np.asarray(z["margins"][pi], np.float32)
        u = am_u[pi].astype(np.int64)
        d = am_d[pi].astype(np.int64)
        fu = u != g
        fd = d != g
        fixed = fu & ~fd
        created = ~fu & fd
        n_fixed += int(fixed.sum())
        n_created += int(created.sum())
        rowmask = np.zeros_like(fu)
        rowmask[r0:r1, :] = True
        band_fixed += int((fixed & rowmask).sum())
        band_created += int((created & rowmask).sum())
        lane_fixed += int((fixed & (g == 1)).sum())
        lane_created += int((created & (g == 1)).sum())
        for k in range(len(mband_edges) - 1):
            mm = (gm >= mband_edges[k]) & (gm < mband_edges[k + 1])
            mband_fixed[k] += int((fixed & mm).sum())
            mband_created[k] += int((created & mm).sum())
        for gc, wc in zip(g[fixed].ravel(), u[fixed].ravel()):
            key = _pair_key(gc, wc)
            fixed_cp[key] = fixed_cp.get(key, 0) + 1
        for gc, wc in zip(g[created].ravel(), d[created].ravel()):
            key = _pair_key(gc, wc)
            created_cp[key] = created_cp.get(key, 0) + 1

    px_per_pair = h * w
    total_px = n_pairs * px_per_pair
    fire = bool(delta <= -1e-5)
    kill = bool(delta >= 0.0)
    verdict = "FIRE" if fire else ("KILL_THIS_FORM" if kill else "BETWEEN_STAYS_GATED")
    summary = {
        "advisory": ADVISORY,
        "generated": _utc(),
        "probe": "P-DITHER (B19 gate; v6 SS7c pre-registered: fire dseg<=-1e-5, kill dseg>=0)",
        "config": {"mode": DEFAULT_MODE, "amp": DEFAULT_AMP, "seed": hex(DEFAULT_SEED)},
        "n_pairs": n_pairs,
        "d_seg_undithered": d_seg_u,
        "d_seg_dithered": d_seg_d,
        "delta_d_seg": delta,
        "delta_S_units": 100.0 * delta,
        "undithered_matches_r6_bitforbit": r6_match,
        "flips_fixed": n_fixed,
        "flips_created": n_created,
        "net_flips": n_created - n_fixed,
        "delta_from_counts": (n_created - n_fixed) / total_px,
        "far_lane_rows_176_224": {"fixed": band_fixed, "created": band_created},
        "gt_lane_class": {"fixed": lane_fixed, "created": lane_created},
        "gt_margin_bands": {
            f"[{mband_edges[k]},{mband_edges[k+1]})": {"fixed": int(mband_fixed[k]),
                                                        "created": int(mband_created[k])}
            for k in range(len(mband_edges) - 1)},
        "fixed_class_pairs_top": dict(sorted(fixed_cp.items(), key=lambda kv: -kv[1])[:10]),
        "created_class_pairs_top": dict(sorted(created_cp.items(), key=lambda kv: -kv[1])[:10]),
        "verdict": verdict,
    }
    out_json = out / "pdither_compare.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[{_utc()}] wrote {out_json}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--packet", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--start", type=int, required=True)
    d.add_argument("--end", type=int, required=True)
    d.add_argument("--workers", type=int, default=10)
    d.add_argument("--amp", type=float, default=DEFAULT_AMP)
    d.add_argument("--mode", default=DEFAULT_MODE)
    d.add_argument("--seed", type=lambda s: int(s, 0), default=DEFAULT_SEED)
    d.set_defaults(fn=cmd_decode)
    v = sub.add_parser("verdict")
    v.add_argument("--arm", choices=("dithered", "undithered"), required=True)
    v.add_argument("--raw", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--gt-cache", type=Path, required=True)
    v.add_argument("--start", type=int, required=True)
    v.add_argument("--end", type=int, required=True)
    v.add_argument("--batch", type=int, default=32)
    v.add_argument("--cam-h", type=int, default=874)
    v.add_argument("--cam-w", type=int, default=1164)
    v.set_defaults(fn=cmd_verdict)
    c = sub.add_parser("compare")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--gt-cache", type=Path, required=True)
    c.add_argument("--r6-rows", type=Path, default=None)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
