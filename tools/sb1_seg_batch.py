#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_sb1 — the $0 seg/feasibility batch: resumable terminal-seg measurements.

Reuses (NOT rebuilds) the pb1/ru1 instruments through the shared tr1.ddt1 runtime:
  * ``tac.optimization.ddm_tr1_runtime`` — parse_archive / render_frame1_camera_uint8 /
    _encode_tokens / build_packet / _archive_manifest / _deterministic_stored_zip
    (the exact receiver path pb1 P2c + ru1 atlas used);
  * ``tac.boundary_math.seg_core.load_real_segnet`` — the frozen CPU-torch SegNet;
  * ``train_witness_realized_through_R_mlx.cpu_verdict_d_seg_batch`` — the realized
    argmax d_seg verdict (whole-pair, ERF collateral captured);
  * ru1 ``atlas_flat.npz`` — per-flip typed rows (aim; verified bit-identical to pb1 P1).

Lineage: ``tools/pb1_aimed_cell_edits.py`` (the P2c round-1 single-quantum greedy).
This tool ADDS: (a) resumability with per-instance JSONL caches + a live-scorer-slot
refusal (charter: ONE n600 scorer job at a time, reaper-safe sub-5-min chunks);
(b) multi-quantum damped-Newton line search per cell (the QA03 escalation beyond
round-1's single quantum); (c) a flicker-aimed random attack search (QA04) and a
global rank-1 output-bias probe (QA05).

Eval speed: token edits are applied IN PLACE to the parsed packet's decoded token
grid (``decode_token_grid`` reads it fresh per render), so each candidate eval is
one render + one SegNet forward for ONE pair — no Brotli re-encode/re-parse in the
loop. The full re-encode is done ONCE at the end for byte pricing (advisory; the
true shipping price is the r7 SMEVR coder — a cross-arm handoff, not priced here).

d_seg is the primary authority; every net-flip figure is whole-pair realized on the
CURRENT endpoint bytes (aim is advisory).

Axis: [macOS-CPU advisory]; score_claim=false; promotion_eligible=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:  # canonical argv-role helper (repo-root import, then script-dir fallback)
    from tools.argv_role import process_table_entrypoint_holders
except ImportError:  # pragma: no cover - exercised when run as `python tools/<this>.py`
    from argv_role import process_table_entrypoint_holders  # type: ignore

REPO = Path("/Users/adpena/projects/pact")
LEVELS = 16
SHAPE = (600, 24, 32, 4)
GRID_H, GRID_W = 24, 32
CELL = 16
SEG_PX = 196608  # 512*384 SegNet internal resolution (flips / SEG_PX = per-pair d_seg)
BYTE_PRICE = 25.0 / 37_545_489.0
SEG_AXIS_GAP = 0.389011  # #404: the seg contribution of the operating row (pfs1 D1)
CEIL_TIER2_FLIPS = 162_500  # ru1 m_def<0.25 band == the -0.138 ceiling
BAND_TIER1_FLIPS = 54_600   # ru1 m_def<kappa band == the -0.046 Contrarian band
# Processes that indicate another n600 scorer/verdict/trainer job holds the slot.
SLOT_TOKENS = ("pb1_receiver_realized_verdict", "train_levelset_witness",
               "train_witness_realized", "ddm_lv1_s2_nullspace",
               "contest_auth_eval", "evaluate.py", "ru1_endpoint_residual")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def slot_holders() -> list[str]:
    """Command lines of processes that actually RUN a scorer/verdict/trainer entrypoint.

    ddm_gh1 #829: this used to be `any(tok in line ...)` — a bare SUBSTRING scan of the process
    table, so an unrelated background `grep`/`rg`/editor whose argv merely CONTAINED a token (or a
    `--out` path with the token in a directory component) made the tool REFUSE A LAUNCH nothing
    was blocking. Routed through the canonical `tools.argv_role`, which requires the token to
    appear in the BASENAME of a PATH-SHAPED argument of a non-reader program.
    """
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True,
                         text=True, check=False).stdout
    return process_table_entrypoint_holders(out, SLOT_TOKENS, self_tokens=("sb1_seg_batch",))


def slot_is_live() -> bool:
    """True if another scorer/verdict/trainer job appears in the process table."""
    return bool(slot_holders())


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


class SegRuntime:
    """Shared parse -> in-place edit -> render -> realized d_seg loop.

    The working token state is ``self.codes`` (a writable view into the parsed
    packet's decoded grid). Mutating it and calling ``pair_flips`` re-renders the
    edited pair with NO re-encode. ``price_bytes`` does the one full re-encode.
    """

    def __init__(self, archive_path: Path, gt_cache: Path):
        sys.path.insert(0, str(REPO / "src"))
        sys.path.insert(0, str(REPO / "experiments"))
        sys.path.insert(0, str(REPO / "upstream"))
        from train_witness_realized_through_R_mlx import cpu_verdict_d_seg_batch

        from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
        from tac.boundary_math.seg_core import load_real_segnet
        from tac.optimization import ddm_tr1_runtime as rt

        self.rt = rt
        self._verdict = cpu_verdict_d_seg_batch
        self.base_archive = archive_path.read_bytes()
        self.parsed = rt.parse_archive(self.base_archive)
        self.packet = self.parsed.packet
        codes = np.asarray(self.packet.token_codes)
        # make a writable working copy and rebind it into the packet so renders
        # read the mutated state (ParsedTR1Packet is frozen -> object.__setattr__)
        self.codes = codes.astype(codes.dtype, copy=True)
        self.codes.setflags(write=True)
        object.__setattr__(self.packet, "token_codes", self.codes)
        self.orig_codes = codes.astype(np.int64, copy=True)  # immutable reference
        self.lstars = open_stored_npy_memmap(gt_cache, "lstars")
        self.seg = load_real_segnet("cpu")

    def pair_flips(self, p: int) -> int:
        """Whole-pair realized flip count for pair p at the CURRENT self.codes."""
        f1 = self.rt.render_frame1_camera_uint8(self.packet, p)
        gt = np.asarray(self.lstars[p], dtype=np.int64)
        d_seg = float(self._verdict(self.seg, [f1], [gt])[0])
        return round(d_seg * SEG_PX)

    def render_uint8(self, p: int):
        return self.rt.render_frame1_camera_uint8(self.packet, p)

    def price_bytes(self) -> tuple[int, str]:
        """One full re-encode of the current token state -> (bytes, sha256)."""
        rt = self.rt
        payloads = {
            "tokens": rt._encode_tokens(self.codes.astype(np.uint8)),
            "lotto_renderer": self.parsed.packet.section_payloads[1],
            "selector": self.parsed.packet.section_payloads[2],
            "pose_stub": self.parsed.packet.section_payloads[3],
        }
        packet = rt.build_packet(self.parsed.packet.metadata, payloads)
        manifest = rt._archive_manifest(packet)
        archive = rt._deterministic_stored_zip({
            "manifest.json": rt._canonical_json(dict(manifest)),
            "state/tr1.ddt1": packet,
        })
        return len(archive), _sha(archive)


# --------------------------------------------------------------------------- #
# QA03 — full-population GN (multi-quantum damped-Newton) terminal seg solve
# --------------------------------------------------------------------------- #

def _top_instances(atlas_path: Path, top_k: int, flicker_only: bool = False,
                   m_def_max: float | None = None):
    atlas = np.load(atlas_path)
    keep = np.ones(atlas["pair"].shape, dtype=bool)
    if flicker_only:
        keep &= atlas["gt_flicker"].astype(bool)
    if m_def_max is not None:
        keep &= atlas["m_def"].astype(np.float64) < m_def_max
    pair = atlas["pair"].astype(np.int64)[keep]
    cy = atlas["y"].astype(np.int64)[keep] // CELL
    cx = atlas["x"].astype(np.int64)[keep] // CELL
    key = (pair * GRID_H + cy) * GRID_W + cx
    uniq, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return [(int(uniq[o] // (GRID_H * GRID_W)),
             int((uniq[o] // GRID_W) % GRID_H),
             int(uniq[o] % GRID_W),
             int(counts[o])) for o in order[:top_k]]


def _best_single_quantum(rtp: SegRuntime, p: int, gy: int, gx: int, cur: int):
    """Try all 8 (4 channels x +/-1) single-quantum moves in place; return the
    (flips, ch, sign) that most reduces whole-pair flips (None-move restored)."""
    best = None
    for ch in range(4):
        code = int(rtp.codes[p, gy, gx, ch])
        for sign in (-1, 1):
            nc = code + sign
            if nc < 0 or nc >= LEVELS:
                continue
            rtp.codes[p, gy, gx, ch] = nc
            f = rtp.pair_flips(p)
            rtp.codes[p, gy, gx, ch] = code  # restore
            if best is None or f < best[0]:
                best = (f, ch, sign)
    return best


def qa03_gn_solve(args) -> None:
    rtp = SegRuntime(args.archive, args.gt_cache)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "qa03_instances.jsonl"
    instances = _top_instances(args.atlas, args.top_k)

    # ---- resume: replay accepted edits, skip processed instances ----
    base_flip_cache: dict[int, int] = {}
    processed: set[tuple[int, int, int]] = set()
    net_flips = 0
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            processed.add((r["pair"], r["cell"][0], r["cell"][1]))
            # work state replays all edits; the current whole-pair flips is the
            # LAST recorded final for this pair (rows are in processing order).
            base_flip_cache[r["pair"]] = r["pair_final_flips"]
            for ch, sign, _q in r["accepted_steps"]:
                rtp.codes[r["pair"], r["cell"][0], r["cell"][1], ch] += sign
            net_flips += r["net_flips"]
        print(f"[resume] {len(processed)} instances, net so far {net_flips:+d}",
              flush=True)

    t0 = time.time()
    for (p, gy, gx, nflips) in instances:
        if (p, gy, gx) in processed:
            continue
        if not args.skip_slot_check and slot_is_live():
            raise SystemExit(f"[refuse] scorer slot live before p{p} ({gy},{gx})")
        if p not in base_flip_cache:
            base_flip_cache[p] = rtp.pair_flips(p)
        cur = base_flip_cache[p]
        pair_base = cur
        accepted_steps = []
        # damped-Newton line search: repeat best single quantum while it helps.
        for _step in range(args.max_quanta):
            best = _best_single_quantum(rtp, p, gy, gx, cur)
            if best is None or best[0] >= cur:
                break
            f, ch, sign = best
            rtp.codes[p, gy, gx, ch] += sign  # commit the accepted quantum
            accepted_steps.append([ch, sign, int(rtp.codes[p, gy, gx, ch])])
            cur = f
        pair_net = pair_base - cur
        net_flips += pair_net
        base_flip_cache[p] = cur
        row = {"pair": p, "cell": [gy, gx], "atlas_flips": nflips,
               "pair_base_flips": int(pair_base), "pair_final_flips": int(cur),
               "net_flips": int(pair_net), "accepted_steps": accepted_steps}
        with jsonl.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[p{p} ({gy},{gx}) atlas {nflips}] net {pair_net:+d} "
              f"steps {len(accepted_steps)} cum {net_flips:+d} "
              f"({time.time()-t0:.0f}s)", flush=True)

    composed_bytes, composed_sha = rtp.price_bytes()
    d_seg_delta = -net_flips / (SHAPE[0] * SEG_PX)
    s_delta_seg = 100.0 * d_seg_delta
    byte_delta = composed_bytes - len(rtp.base_archive)
    receipt = {
        "schema": "ddm_sb1_qa03_gn_solve.v1",
        "item": "QA03 full-population GN (multi-quantum damped-Newton) terminal seg solve",
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "base_archive_sha256": _sha(rtp.base_archive),
        "base_archive_bytes": len(rtp.base_archive),
        "composed_archive_sha256": composed_sha,
        "composed_archive_bytes": composed_bytes,
        "byte_delta_tr1_reencode": int(byte_delta),
        "byte_delta_note": ("tr1.ddt1 re-encode; token edits byte-neutral at first "
                            "order (ru1); true shipping price = r7 SMEVR coder (xi1 handoff)"),
        "top_k": args.top_k, "max_quanta_per_cell": args.max_quanta,
        "instances_processed": len(instances),
        "net_flips_total": int(net_flips),
        "d_seg_delta": d_seg_delta, "seg_S_delta": s_delta_seg,
        "frac_of_minus0138_ceiling": net_flips / CEIL_TIER2_FLIPS,
        "frac_of_minus0046_band": net_flips / BAND_TIER1_FLIPS,
        "frac_of_seg_axis_gap": abs(s_delta_seg) / SEG_AXIS_GAP,
        "verdict_scope": "INSTANCE (this endpoint b9a7983b, this scorer, this atlas aim)",
        "wall_seconds": time.time() - t0,
        "generated_by": "tools/sb1_seg_batch.py qa03",
    }
    _atomic_write_text(out_dir / "qa03_receipt.json",
                       json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"[QA03 done] net {net_flips:+d} flips  dS_seg {s_delta_seg:+.6f}  "
          f"frac_ceiling {receipt['frac_of_minus0138_ceiling']:.4f}  "
          f"bytes {byte_delta:+d}", flush=True)


# --------------------------------------------------------------------------- #
# QA04 — flicker-aimed random attack search (SparseRS/Square-style, round-2)
# --------------------------------------------------------------------------- #

def qa04_attack_search(args) -> None:
    rtp = SegRuntime(args.archive, args.gt_cache)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "qa04_instances.jsonl"
    # aim: flicker AND m_def<0.25, ranked by (pair,cell) flicker count (of1/ru1)
    instances = _top_instances(args.atlas, args.top_k, flicker_only=True,
                               m_def_max=0.25)
    rng = np.random.default_rng(args.seed)

    processed: set[tuple[int, int, int]] = set()
    net_flips = 0
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            processed.add((r["pair"], r["cell"][0], r["cell"][1]))
            for ch, sign in r["accepted"]:
                rtp.codes[r["pair"], r["cell"][0], r["cell"][1], ch] += sign
            net_flips += r["net_flips"]
        print(f"[resume] {len(processed)} instances net {net_flips:+d}", flush=True)

    t0 = time.time()
    evals = 0
    for (p, gy, gx, nflips) in instances:
        if (p, gy, gx) in processed:
            continue
        if not args.skip_slot_check and slot_is_live():
            raise SystemExit(f"[refuse] scorer slot live before p{p} ({gy},{gx})")
        base = rtp.pair_flips(p)
        # Square-style single-coordinate proposals over the 4-channel cell.
        moves = [(ch, sign) for ch in range(4) for sign in (-1, 1)]
        rng.shuffle(moves)
        best = (base, None)
        for (ch, sign) in moves[:args.n_prop]:
            code = int(rtp.codes[p, gy, gx, ch])
            nc = code + sign
            if nc < 0 or nc >= LEVELS:
                continue
            rtp.codes[p, gy, gx, ch] = nc
            f = rtp.pair_flips(p)
            rtp.codes[p, gy, gx, ch] = code  # restore
            evals += 1
            if f < best[0]:
                best = (f, (ch, sign))
            if evals >= args.max_evals:
                break
        accepted = []
        if best[1] is not None and best[0] < base:
            ch, sign = best[1]
            rtp.codes[p, gy, gx, ch] += sign
            accepted = [[ch, sign]]
        pair_net = base - best[0] if accepted else 0
        net_flips += pair_net
        row = {"pair": p, "cell": [gy, gx], "flicker_flips": nflips,
               "base_flips": int(base), "net_flips": int(pair_net),
               "accepted": accepted}
        with jsonl.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[p{p} ({gy},{gx}) flk {nflips}] net {pair_net:+d} cum {net_flips:+d} "
              f"evals {evals}", flush=True)
        if evals >= args.max_evals:
            print(f"[QA04] eval budget {args.max_evals} reached", flush=True)
            break

    d_seg_delta = -net_flips / (SHAPE[0] * SEG_PX)
    receipt = {
        "schema": "ddm_sb1_qa04_attack_search.v1",
        "item": "QA04 flicker-aimed random attack search (round-2)",
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "base_archive_sha256": _sha(rtp.base_archive),
        "aim": "atlas gt_flicker==1 AND m_def<0.25, ranked by (pair,cell) flicker count",
        "top_k": args.top_k, "n_prop": args.n_prop, "max_evals": args.max_evals,
        "evals_used": evals, "net_flips_total": int(net_flips),
        "d_seg_delta": d_seg_delta, "seg_S_delta": 100.0 * d_seg_delta,
        "frac_of_seg_axis_gap": abs(100.0 * d_seg_delta) / SEG_AXIS_GAP,
        "round1_reference_net_flips": -155,
        "verdict_scope": "INSTANCE (this endpoint, this scorer, this aim/seed)",
        "wall_seconds": time.time() - t0,
        "generated_by": "tools/sb1_seg_batch.py qa04",
    }
    _atomic_write_text(out_dir / "qa04_receipt.json",
                       json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"[QA04 done] net {net_flips:+d} flips  dS_seg {100.0*d_seg_delta:+.6f}",
          flush=True)


# --------------------------------------------------------------------------- #
# QA05 — global rank-1 output-bias probe (is tier-2 token-only?)
# --------------------------------------------------------------------------- #

def qa05_renderer_rank1(args) -> None:
    """Probe whether a GLOBAL (all-pair) rank-1 additive bias on the rendered
    frame_1 can reduce net d_seg. net>0 for any candidate falsifies 'the token
    lattice is the only actuation surface' (ru1 op-routable 3 / Assumption-
    Adversary); net<=0 for all confirms tier-2 is per-pair token-only. Output-
    space proxy for a renderer-section rank-1 edit; bounded subset verdict."""
    rtp = SegRuntime(args.archive, args.gt_cache)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "qa05_candidates.jsonl"

    pairs = list(range(0, SHAPE[0], SHAPE[0] // args.subset))[:args.subset]
    base_frames = {p: np.asarray(rtp.render_uint8(p), dtype=np.int16) for p in pairs}
    H, W, _C = base_frames[pairs[0]].shape

    def subset_flips(bias) -> int:
        tot = 0
        for p in pairs:
            fr = np.clip(base_frames[p] + bias, 0, 255).astype(np.uint8)
            gt = np.asarray(rtp.lstars[p], dtype=np.int64)
            tot += round(float(rtp._verdict(rtp.seg, [fr], [gt])[0]) * SEG_PX)
        return tot

    yv = np.linspace(-1.0, 1.0, H)[:, None, None]
    xv = np.linspace(-1.0, 1.0, W)[None, :, None]
    patterns = {
        "const_all": np.ones((H, W, 3)),
        "const_r": np.array([1.0, 0.0, 0.0])[None, None, :] * np.ones((H, W, 1)),
        "const_g": np.array([0.0, 1.0, 0.0])[None, None, :] * np.ones((H, W, 1)),
        "const_b": np.array([0.0, 0.0, 1.0])[None, None, :] * np.ones((H, W, 1)),
        "vramp": yv * np.ones((1, W, 3)),
        "hramp": xv * np.ones((H, 1, 3)),
    }
    processed = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                processed.add(tuple(json.loads(line)["cand"]))

    t0 = time.time()
    base = subset_flips(np.zeros((H, W, 3), dtype=np.int16))
    for name, pat in patterns.items():
        for amp in args.amps:
            if (name, amp) in processed:
                continue
            if not args.skip_slot_check and slot_is_live():
                raise SystemExit(f"[refuse] scorer slot live before {name}@{amp}")
            bias = np.round(pat * amp).astype(np.int16)
            f = subset_flips(bias)
            row = {"cand": [name, amp], "subset_base_flips": base,
                   "subset_flips": f, "net": base - f}
            with jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{name}@{amp:+d}] net {base - f:+d} (base {base})", flush=True)

    all_rows = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]
    best = max(all_rows, key=lambda r: r["net"]) if all_rows else None
    receipt = {
        "schema": "ddm_sb1_qa05_rank1_probe.v1",
        "item": "QA05 global rank-1 output-bias probe (is tier-2 token-only?)",
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "base_archive_sha256": _sha(rtp.base_archive),
        "subset_pairs": len(pairs), "subset_base_flips": base,
        "amps": args.amps, "n_candidates": len(all_rows), "best": best,
        "interpretation": ("net>0 for any candidate => a global renderer-level lever "
                           "exists (tier-2 NOT token-only); net<=0 for all => tier-2 "
                           "is per-pair token-only on this vehicle (probe null)"),
        "verdict_scope": "INSTANCE (output-space proxy, this subset/scorer/endpoint)",
        "wall_seconds": time.time() - t0,
        "generated_by": "tools/sb1_seg_batch.py qa05",
    }
    _atomic_write_text(out_dir / "qa05_receipt.json",
                       json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"[QA05 done] best net {best['net'] if best else 0:+d} "
          f"({best['cand'] if best else None})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--archive", required=True, type=Path)
    common.add_argument("--gt-cache", required=True, type=Path)
    common.add_argument("--out-dir", required=True, type=Path)
    common.add_argument("--skip-slot-check", action="store_true")

    a = sub.add_parser("qa03", parents=[common])
    a.add_argument("--atlas", required=True, type=Path)
    a.add_argument("--top-k", type=int, default=120)
    a.add_argument("--max-quanta", type=int, default=4)

    b = sub.add_parser("qa04", parents=[common])
    b.add_argument("--atlas", required=True, type=Path)
    b.add_argument("--top-k", type=int, default=200)
    b.add_argument("--n-prop", type=int, default=8)
    b.add_argument("--max-evals", type=int, default=800)
    b.add_argument("--seed", type=int, default=20260729)

    c = sub.add_parser("qa05", parents=[common])
    c.add_argument("--subset", type=int, default=120)
    c.add_argument("--amps", type=int, nargs="+", default=[-2, -1, 1, 2])

    args = ap.parse_args()
    if args.cmd == "qa03":
        qa03_gn_solve(args)
    elif args.cmd == "qa04":
        qa04_attack_search(args)
    elif args.cmd == "qa05":
        qa05_renderer_rank1(args)


if __name__ == "__main__":
    main()
