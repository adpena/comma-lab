#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#307 CONTOUR-STRING FLIP CODING — $0/CPU n600 measurement of chain-code flip-residual rate.

MEANS. pointer 0.19110 UNMOVED. Authority: [macOS-CPU advisory] NON-PROMOTABLE — a coder-rate
measurement, never a score. NO-FAKE: real frozen witness checkpoint (int8-dequant weights = the
byte-closed archive values), real #202 byte-close render authority (self-orient fixed point +
analytic-lane-band composite + contest R), real frozen CPU-torch SegNet verdict, real in-tree
RangeEncoder streams that are DECODED back and verified bit-exact against every flip map.

WHAT THIS MEASURES (FEED-07b row 6): the #280 Lever-D Stage-0 gate measured min(b)=0.876 B/flip
(bz2 full-grid, 32 pairs) vs the pre-registered GO bar b < 0.65 B/flip. Published contour/chain
coding reaches ~1-1.5 bits/contour-px (0.125-0.19 B) on coherent strings — a 4-6x gap IF the flip
set is string-coherent. This tool codes the REAL n600 flip set as connected-component contour
strings (8-dir chain codes, digital-straightness contexts, adaptive range coding) and reports
bytes/flip + the coherent-vs-singleton decomposition. BOTH branches signal:
  GO   (b < 0.65): Lever-D reactivates (positions become affordable; the residual coder economics
       re-open per leverd_flicker_residual_reactivation_economics_v1).
  NO-GO (b >= 0.65): quantifies the incoherent/flicker fraction (singleton anchors dominate the
       byte cost -> the residual is temporally-flickering salt-and-pepper, not codeable strings).

DATA PATH (reuse, not re-derive): the #280 leverd_stage0 harness path VERBATIM — render frame1
per pair through the byte-close authority, frozen CPU SegNet argmax, flip = (realized != gt lstar).
Measured on BOTH surfaces: witness-alone (matches the live trainer's verdict d_seg) and
composed (witness + analytic lane band — the as-shipped decomposition; the band removes the most
string-coherent lane flips, so the two surfaces BRACKET the coder's real-world b).

RESUMABLE (non-negotiable): stage A (extraction) writes one .npz per pair into --flips-cache-dir
and skips existing files on relaunch; stage B (coding) re-reads the cache. A memory self-guard
checks psutil available RAM every pair and exits rc=7 (resumable) below --mem-floor-mb — the probe
never endangers the live trainer (run_probe_guarded.sh watchdog pattern, P0 machine-crash gate).

Usage (mod32cap live-run snapshot; self-orient freq-along=8 per its launch.sh):
  .venv/bin/python tools/measure_contour_string_flip_coding.py \
      --ckpt-dir <scratch snapshot dir> --npz-name snapshot_ema_BEST.npz \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --n 600 --so-freq-along 8.0 \
      --flips-cache-dir <scratch>/flips --out <scratch>/contour_flip_coding_n600.json
"""
from __future__ import annotations

import argparse
import bz2
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

N_CLASSES = 5
RATE_DENOM = 37_545_489.0
GO_BAR_B = 0.65  # pre-registered (#280 / FEED-07b row 6)


def _rss_str() -> str:
    """Instrumentation (2026-07-07 daemon-death diagnosis): current + peak RSS of THIS process.
    The #307/#336 daemons were safe_run rc=137 SIGKILLed on the group --rss-mb cap at a fixed
    step count under 10 AND 16/20 GiB caps — the spike must be MEASURED, not guessed. Printed
    per progress block so the last log line before any kill localizes the growth."""
    import resource

    import psutil
    cur = psutil.Process().memory_info().rss / 2**30
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    return f"rss_cur={cur:.2f}GiB rss_peak={peak:.2f}GiB"

# 8-dir chain code, fixed canonical indexing (E,SE,S,SW,W,NW,N,NE)
_DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_SYM_BACKTRACK = 8
_SYM_END = 9
_CHAIN_ALPHABET = 10
_CTX_START = 10  # chain context sentinel for "no previous symbol"


# ---------------------------------------------------------------------------
# adaptive range-coded streams (encoder + decoder share the identical model)
# ---------------------------------------------------------------------------
class AdaptiveStream:
    """One adaptive-context range-coded stream. Laplace(+1) per-context counts updated after each
    symbol — the decoder replays the identical model, so the byte length is a REAL decodable
    stream length (carries true integer-rounding + flush overhead; NO-FAKE)."""

    def __init__(self, alphabet: int):
        from tac.lossless.range_coder import RangeEncoder
        self.alphabet = int(alphabet)
        self.enc = RangeEncoder()
        self.counts: dict[int, np.ndarray] = {}
        self.n_symbols = 0

    def _ctx_counts(self, ctx: int) -> np.ndarray:
        c = self.counts.get(ctx)
        if c is None:
            c = np.ones(self.alphabet, dtype=np.int64)
            self.counts[ctx] = c
        return c

    def encode(self, sym: int, ctx: int) -> None:
        from tac.lossless.range_coder import cumulative_frequencies
        c = self._ctx_counts(int(ctx))
        cum, total = cumulative_frequencies(c.tolist())
        self.enc.encode(symbol=int(sym), cumulative=cum, total=total)
        c[int(sym)] += 1
        self.n_symbols += 1

    def finish(self) -> bytes:
        return self.enc.finish() if self.n_symbols else b""


class AdaptiveStreamDecoder:
    """Mirror of :class:`AdaptiveStream` for the bit-exact verification decode."""

    def __init__(self, data: bytes, alphabet: int):
        from tac.lossless.range_coder import RangeDecoder
        self.alphabet = int(alphabet)
        self.dec = RangeDecoder(data) if data else None
        self.counts: dict[int, np.ndarray] = {}

    def decode(self, ctx: int) -> int:
        from bisect import bisect_right

        from tac.lossless.range_coder import cumulative_frequencies
        assert self.dec is not None, "decoding from an empty stream"
        c = self.counts.get(int(ctx))
        if c is None:
            c = np.ones(self.alphabet, dtype=np.int64)
            self.counts[int(ctx)] = c
        cum, total = cumulative_frequencies(c.tolist())
        target = self.dec.target(total)
        sym = bisect_right(cum, target) - 1
        self.dec.update(low_count=cum[sym], high_count=cum[sym + 1], total=total)
        c[sym] += 1
        return int(sym)


def _varint_bytes(v: int) -> list[int]:
    """Little-endian 7-bit varint bytes (high bit = continuation)."""
    out = []
    v = int(v)
    assert v >= 0
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return out


def _read_varint(dec: AdaptiveStreamDecoder) -> int:
    v, shift, i = 0, 0, 0
    while True:
        b = dec.decode(min(i, 2))
        v |= (b & 0x7F) << shift
        if not (b & 0x80):
            return v
        shift += 7
        i += 1


def _write_varint(stream: AdaptiveStream, v: int) -> None:
    for i, b in enumerate(_varint_bytes(v)):
        stream.encode(b, min(i, 2))


# ---------------------------------------------------------------------------
# contour-string (chain-code) coder — encoder + verifying decoder
# ---------------------------------------------------------------------------
def _chain_ctx(last_sym: int, was_straight: bool) -> int:
    return (int(last_sym) << 1) | int(bool(was_straight))


def _straight_order(last_dir: int | None) -> tuple[int, ...]:
    """Neighbor probe order: straight-ahead first (the digital-straightness prior), then +-1,
    +-2, +-3, reverse. First step (no direction yet) probes canonically 0..7."""
    if last_dir is None:
        return tuple(range(8))
    d = int(last_dir)
    return (d, (d + 1) % 8, (d - 1) % 8, (d + 2) % 8, (d - 2) % 8,
            (d + 3) % 8, (d - 3) % 8, (d + 4) % 8)


def contour_encode_frames(flip_maps: list[np.ndarray], class_maps: list[np.ndarray]) -> dict:
    """Encode every frame's flip set as connected-component contour strings.

    Streams (each ONE adaptive range-coded stream across ALL frames — 4 flush overheads total):
      counts : n_components per frame (varint)
      anchor : per frame, gaps between sorted component-anchor linear indices (varint; first gap =
               anchor index itself + 1 so gaps are >= 1... actually raw first index, then deltas)
      chain  : DFS walk symbols 0-7 (absolute directions), 8 = backtrack, 9 = end-of-component,
               ctx = (last symbol, was-straight bit)
      cls    : GT class at anchor + at each newly visited pixel, ctx = class of the walk's current
               pixel (class coherence within a component -> near-free after adaptation)

    Encoder walk: at the current pixel probe the 8 neighbors straightest-first and step to the
    first unvisited flip pixel (emitting its ABSOLUTE direction); else emit backtrack/end. The
    decoder needs no probe rule — it just follows the emitted absolute directions, so the walk is
    replayable by construction (verified by :func:`contour_decode_frames`).
    """
    from scipy.ndimage import label as cc_label

    t0 = time.time()
    s_counts = AdaptiveStream(256)
    s_anchor = AdaptiveStream(256)
    s_chain = AdaptiveStream(_CHAIN_ALPHABET)
    s_cls = AdaptiveStream(N_CLASSES)
    structure = np.ones((3, 3), dtype=np.int64)  # 8-connectivity

    n_flips_total = 0
    n_comps_total = 0
    comp_size_hist: dict[str, int] = {"1": 0, "2-3": 0, "4-7": 0, "8-15": 0, "16+": 0}
    flips_by_size: dict[str, int] = {"1": 0, "2-3": 0, "4-7": 0, "8-15": 0, "16+": 0}
    chain_syms = 0
    backtracks = 0

    for fm, cm in zip(flip_maps, class_maps, strict=False):
        h, w = fm.shape
        lab, nlab = cc_label(fm, structure=structure)
        n_flips_total += int(fm.sum())
        n_comps_total += int(nlab)
        _write_varint(s_counts, int(nlab))
        if nlab == 0:
            continue
        # component anchors = raster-first pixel of each component, sorted ascending
        flat = lab.reshape(-1)
        first_idx = np.full(nlab + 1, flat.size, dtype=np.int64)
        # first occurrence per label (vectorized: reverse-scan assign)
        idx_all = np.where(flat > 0)[0]
        np.minimum.at(first_idx, flat[idx_all], idx_all)
        anchors = np.sort(first_idx[1:])
        prev = 0
        for a in anchors:
            _write_varint(s_anchor, int(a) - prev)  # first = absolute index, then gaps (>=1)
            prev = int(a)

        visited = np.zeros_like(fm, dtype=bool)
        for a in anchors:
            ay, ax = divmod(int(a), w)
            comp_px = 1
            visited[ay, ax] = True
            s_cls.encode(int(cm[ay, ax]), N_CLASSES - 1)  # anchor class ctx = sentinel (=4 reused)
            stack = [(ay, ax)]
            dir_in: list[int | None] = [None]
            last_sym: int = _CTX_START
            prev_move: int | None = None
            last_move: int | None = None
            while stack:
                cy, cx = stack[-1]
                step = None
                for d in _straight_order(dir_in[-1]):
                    ny, nx = cy + _DIRS[d][0], cx + _DIRS[d][1]
                    if 0 <= ny < h and 0 <= nx < w and fm[ny, nx] and not visited[ny, nx]:
                        step = (d, ny, nx)
                        break
                if step is not None:
                    d, ny, nx = step
                    was_straight = (last_move is not None and prev_move == last_move)
                    s_chain.encode(d, _chain_ctx(last_sym, was_straight))
                    chain_syms += 1
                    s_cls.encode(int(cm[ny, nx]), int(cm[cy, cx]))
                    visited[ny, nx] = True
                    stack.append((ny, nx))
                    dir_in.append(d)
                    prev_move, last_move = last_move, d
                    last_sym = d
                    comp_px += 1
                elif len(stack) > 1:
                    was_straight = (last_move is not None and prev_move == last_move)
                    s_chain.encode(_SYM_BACKTRACK, _chain_ctx(last_sym, was_straight))
                    chain_syms += 1
                    backtracks += 1
                    stack.pop()
                    dir_in.pop()
                    last_sym = _SYM_BACKTRACK
                else:
                    was_straight = (last_move is not None and prev_move == last_move)
                    s_chain.encode(_SYM_END, _chain_ctx(last_sym, was_straight))
                    chain_syms += 1
                    stack.pop()
                    dir_in.pop()
            key = ("1" if comp_px == 1 else "2-3" if comp_px <= 3 else
                   "4-7" if comp_px <= 7 else "8-15" if comp_px <= 15 else "16+")
            comp_size_hist[key] += 1
            flips_by_size[key] += comp_px

    parts = {"counts": s_counts.finish(), "anchor": s_anchor.finish(),
             "chain": s_chain.finish(), "cls": s_cls.finish()}
    total = sum(len(v) for v in parts.values())
    denom = max(1, n_flips_total)
    return {
        "streams": parts,
        "stream_bytes": {k: len(v) for k, v in parts.items()},
        "total_bytes": int(total),
        "n_flips": int(n_flips_total),
        "n_components": int(n_comps_total),
        "b_contour": float(total / denom),
        "chain_symbols": int(chain_syms),
        "backtracks": int(backtracks),
        "comp_size_hist": comp_size_hist,
        "flips_by_comp_size": flips_by_size,
        "singleton_flip_frac": float(flips_by_size["1"] / denom),
        "coherent_ge4_flip_frac": float(
            (flips_by_size["4-7"] + flips_by_size["8-15"] + flips_by_size["16+"]) / denom),
        "encode_s": round(time.time() - t0, 2),
    }


def contour_decode_frames(streams: dict[str, bytes], n_frames: int, h: int, w: int
                          ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Replay the four streams and reconstruct (flip_maps, class_maps). Used ONLY to verify the
    encoder's streams are genuinely decodable + bit-exact (NO-FAKE byte accounting)."""
    d_counts = AdaptiveStreamDecoder(streams["counts"], 256)
    d_anchor = AdaptiveStreamDecoder(streams["anchor"], 256)
    d_chain = AdaptiveStreamDecoder(streams["chain"], _CHAIN_ALPHABET)
    d_cls = AdaptiveStreamDecoder(streams["cls"], N_CLASSES)
    flips_out: list[np.ndarray] = []
    cls_out: list[np.ndarray] = []
    for _f in range(n_frames):
        fm = np.zeros((h, w), dtype=bool)
        cm = np.full((h, w), -1, dtype=np.int64)
        nlab = _read_varint(d_counts)
        anchors = []
        prev = 0
        for _ in range(nlab):
            prev = prev + _read_varint(d_anchor)
            anchors.append(prev)
        for a in anchors:
            ay, ax = divmod(int(a), w)
            fm[ay, ax] = True
            cm[ay, ax] = d_cls.decode(N_CLASSES - 1)
            stack = [(ay, ax)]
            last_sym: int = _CTX_START
            prev_move: int | None = None
            last_move: int | None = None
            while stack:
                cy, cx = stack[-1]
                was_straight = (last_move is not None and prev_move == last_move)
                sym = d_chain.decode(_chain_ctx(last_sym, was_straight))
                if sym < 8:
                    ny, nx = cy + _DIRS[sym][0], cx + _DIRS[sym][1]
                    fm[ny, nx] = True
                    cm[ny, nx] = d_cls.decode(int(cm[cy, cx]))
                    stack.append((ny, nx))
                    prev_move, last_move = last_move, sym
                    last_sym = sym
                elif sym == _SYM_BACKTRACK:
                    stack.pop()
                    last_sym = _SYM_BACKTRACK
                else:  # END
                    stack.pop()
                    assert not stack, "END symbol decoded with a non-empty walk stack"
        flips_out.append(fm)
        cls_out.append(cm)
    return flips_out, cls_out


def bz2_fullgrid_baseline(flip_maps: list[np.ndarray], class_maps: list[np.ndarray]) -> dict:
    """The #280 full-grid frame-major baseline (min(b)=0.876 on the ep300 32-pair set): one byte
    per (frame, pixel) = 0 | class+1, bz2 -9. Cheap; run on the SAME n600 set for the honest
    apples-to-apples comparison."""
    stacks = []
    n_flips = 0
    for fm, cm in zip(flip_maps, class_maps, strict=False):
        n_flips += int(fm.sum())
        stacks.append(np.where(fm, cm + 1, 0).astype(np.uint8).reshape(-1))
    byte_stack = np.concatenate(stacks)
    nb = len(bz2.compress(byte_stack.tobytes(), 9))
    return {"bz2_bytes": int(nb), "n_flips": int(n_flips),
            "b_bz2": float(nb / max(1, n_flips))}


# ---------------------------------------------------------------------------
# stage A: flip extraction through the #280/#202 byte-close render authority
# ---------------------------------------------------------------------------
def build_render_ctx(bc, params_dq, code_dq, manifest, lane_pairs):
    rh, rw = int(manifest["render_h"]), int(manifest["render_w"])
    ch, cw = int(manifest["camera_h"]), int(manifest["camera_w"])
    from tac.boundary_math.analytic_lane_render_band import render_config_from_header
    coords = bc._canon_coords_grid(rh, rw)
    B = bc._canon_curvelet_B(
        manifest["bank_n_scales"], manifest["bank_n_orient0"], manifest["bank_f0"],
        manifest["bank_base"], manifest["bank_n_iso"], manifest["max_bank_freq"])
    curv = bc._canon_curvelet_feats(coords, B)
    lr = manifest.get("lane_render_band")
    band_on = bool(lr is not None and lane_pairs is not None)
    lane_cfg = render_config_from_header({**lr, "pairs": []}) if band_on else None
    fwd_kw = {
        "n_hidden": int(manifest["n_hidden"]), "hidden_dim": int(manifest["hidden_dim"]),
        "n_classes": int(manifest["n_classes"]), "activation": str(manifest["activation"]),
        "softmax_temp": float(manifest["softmax_temp"]), "wire_w0": float(manifest["wire_w0"]),
        "wire_s0": float(manifest["wire_s0"]), "hosc_beta": float(manifest["hosc_beta"]),
        "hosc_omega": float(manifest["hosc_omega"]), "chroma": bool(manifest["chroma"])}
    return {"rh": rh, "rw": rw, "ch": ch, "cw": cw, "coords": coords, "curv": curv, "band_on": band_on,
                "lane_cfg": lane_cfg, "fwd_kw": fwd_kw, "manifest": manifest, "params": params_dq,
                "code": code_dq, "lane_pairs": lane_pairs}


def render_frame1_both(bc, ctx, pi):
    """(rgb_alone, rgb_composed) at render res for frame1 of pair pi — self-orient fixed point +
    optional analytic-lane-band composite, op-for-op the #280 leverd_stage0 path (which mirrors
    bc.numpy_oracle_reference_frames fk==1)."""
    from tac.boundary_math.analytic_lane_render_band import (
        composite_band_on_render,
        rasterize_lane_coverage_range_dependent,
        witness_uncertainty_mask,
    )
    from tac.boundary_math.lever_b_levelset_generator import (
        levelset_band_forward_numpy,
        levelset_rgb_forward_numpy,
    )

    m, curv, coords, fwd_kw = ctx["manifest"], ctx["curv"], ctx["coords"], ctx["fwd_kw"]
    params, code, rh, rw = ctx["params"], ctx["code"], ctx["rh"], ctx["rw"]
    if bool(m["self_orient"]):
        ndf = int(m["n_dir_freqs"])
        dirf = np.zeros((curv.shape[0], 4 * ndf), np.float32)
        prev_am = None
        for _ in range(int(m["so_iters"])):
            feats = np.concatenate([curv, dirf], axis=-1)
            _rgb, phi = levelset_rgb_forward_numpy(params, feats, code[2 * pi + 1], **fwd_kw)
            am = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
            if prev_am is not None and np.array_equal(am, prev_am):
                break
            dirf = bc._canon_dir_feats(coords, am, ndf, float(m["so_freq_along"]),
                                       float(m["so_freq_across"]), float(m["so_tau"]))
            prev_am = am
        feats = np.concatenate([curv, dirf], axis=-1)
    else:
        feats = curv
    if ctx["band_on"] and ctx["lane_pairs"] is not None and pi < len(ctx["lane_pairs"]):
        lc = ctx["lane_cfg"]
        cov = rasterize_lane_coverage_range_dependent(
            ctx["lane_pairs"][pi], h=rh, w=rw, softness=lc.softness, dash_gate=lc.dash_gate,
            dash_forward_max_m=lc.dash_forward_max_m, v_h=lc.v_h, cx=lc.cx).reshape(-1)
        rgb, phi, lane_rgb, margin = levelset_band_forward_numpy(
            params, feats, code[2 * pi + 1], lane_cls=lc.lane_cls, **fwd_kw)
        um = (witness_uncertainty_mask(margin, tau=lc.u_mask_tau, eps=lc.u_mask_eps)
              if lc.u_mask_enabled else None)
        rgb_comp = composite_band_on_render(rgb, lane_rgb, cov, um, lc.weight)
        return np.asarray(rgb, np.float32), np.asarray(rgb_comp, np.float32)
    rgb, _phi = levelset_rgb_forward_numpy(params, feats, code[2 * pi + 1], **fwd_kw)
    rgb = np.asarray(rgb, np.float32)
    return rgb, rgb


def segnet_argmax(seg_cpu, frame1_cam_uint8):
    """Frozen CPU-torch SegNet argmax on ONE camera frame -> (h,w) int64 (verdict path mirror)."""
    import torch

    r = np.asarray(frame1_cam_uint8)
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = seg_cpu.preprocess_input(xp)
        logits = seg_cpu(seg_in)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)


def extract_flips(args) -> tuple[Path, dict]:
    """Stage A: per-pair flip maps for BOTH surfaces, written incrementally (resume = skip
    existing). Returns (cache_dir, extraction_report)."""
    import brotli
    import levelset_byte_close_and_eval as bc
    import psutil
    import torch

    from tac.boundary_math.analytic_lane_render_band import (
        LaneBandRenderConfig,
        deserialize_lane_band_any,
    )
    from tac.boundary_math.lever_b_levelset_generator import _int8_symmetric
    from tac.boundary_math.seg_core import load_real_segnet

    torch.set_num_threads(max(1, int(args.torch_threads)))
    cache = Path(args.flips_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    N = int(args.n)

    params, cfg = bc._load_levelset_ckpt(Path(args.ckpt_dir), args.npz_name)
    so = bc.detect_self_orient(cfg, {
        "freq_across": float(args.so_freq_across), "freq_along": float(args.so_freq_along),
        "tau": float(args.so_tau), "iters": int(args.so_iters)})
    print(f"[#307] ckpt cfg: render={cfg['render_h']}x{cfg['render_w']} n_pairs={cfg['n_pairs']} "
          f"self_orient={so['self_orient']} n_dir_freqs={so.get('n_dir_freqs')} "
          f"activation={cfg['activation']} chroma={cfg['chroma']}", flush=True)

    # int8-dequantize weights + code = the BYTE-CLOSED archive values (#202 authority)
    params_dq = {}
    for k, v in params.items():
        if k == "code":
            continue
        q, s = _int8_symmetric(np.asarray(v, np.float32))
        params_dq[k] = (q.astype(np.float32) * s)
    code = np.asarray(params["code"], np.float32)
    qc, cs = _int8_symmetric(code)
    code_dq = (qc.astype(np.float32) * cs).reshape(code.shape)

    lane_cfg = LaneBandRenderConfig(
        softness=1.0, dash_gate=True, dash_forward_max_m=55.0, weight=1.0, lane_cls=1,
        u_mask_enabled=True, u_mask_tau=0.85, u_mask_eps=0.35)
    # lane-fit cache: the RD fit over N pairs is deterministic in (gt_cache, N, cfg) and costs
    # ~40 s — cache the SERIALIZED (counted) bytes in the flips cache dir so chunked-foreground
    # resume calls skip the refit. Decode path unchanged (same deserialize as the fresh fit).
    lane_cache_f = cache / f"lane_band_n{N}.bin"
    lane_report_f = cache / f"lane_band_n{N}.report.json"
    if lane_cache_f.exists() and lane_report_f.exists():
        lane_bytes = lane_cache_f.read_bytes()
        lane_report = json.loads(lane_report_f.read_text())
        lane_manifest = lane_report["_manifest_cache"]
    else:
        lane_bytes, lane_manifest, lane_report = bc.build_lane_band_section(
            args.gt_cache, N, lane_cfg, rd=True)
        lane_cache_f.write_bytes(lane_bytes)
        lane_report_f.write_text(json.dumps(
            {**lane_report, "_manifest_cache": lane_manifest}, default=str))
    lane_pairs, _hdr = deserialize_lane_band_any(brotli.decompress(lane_bytes))
    print(f"[#307] lane-band: {lane_report['n_pairs_fit']} pairs fit, "
          f"COUNTED={lane_report['counted_brotli_bytes']}B", flush=True)

    manifest = {
        "n_pairs": int(cfg["n_pairs"]), "n_classes": int(cfg["n_classes"]),
        "hidden_dim": int(cfg["hidden_dim"]), "n_hidden": int(cfg["n_hidden"]),
        "mod_dim": int(cfg["mod_dim"]), "activation": str(cfg["activation"]),
        "softmax_temp": float(cfg["softmax_temp"]), "chroma": bool(cfg["chroma"]),
        "wire_w0": float(cfg["wire_w0"]), "wire_s0": float(cfg["wire_s0"]),
        "hosc_beta": float(cfg["hosc_beta"]), "hosc_omega": float(cfg["hosc_omega"]),
        "bank_n_scales": int(cfg["bank_n_scales"]), "bank_n_orient0": int(cfg["bank_n_orient0"]),
        "bank_f0": float(cfg["bank_f0"]), "bank_base": float(cfg["bank_base"]),
        "bank_n_iso": int(cfg["bank_n_iso"]), "max_bank_freq": cfg["max_bank_freq"],
        "render_h": int(cfg["render_h"]), "render_w": int(cfg["render_w"]),
        "camera_h": bc.CAMERA_H, "camera_w": bc.CAMERA_W,
        "self_orient": bool(so["self_orient"]), "n_dir_freqs": int(so.get("n_dir_freqs", 0)),
        "so_freq_across": float(so.get("freq_across", 0.0)),
        "so_freq_along": float(so.get("freq_along", 0.0)),
        "so_tau": float(so.get("tau", 4.0)), "so_iters": int(so.get("iters", 0)),
        "lane_render_band": lane_manifest,
    }
    ctx = build_render_ctx(bc, params_dq, code_dq, manifest, lane_pairs)

    z = np.load(args.gt_cache, allow_pickle=False)
    lstars_all = z["lstars"]
    seg_cpu = load_real_segnet("cpu")
    rh, rw, ch, cw = ctx["rh"], ctx["rw"], ctx["ch"], ctx["cw"]

    n_done = 0
    t0 = time.time()
    for pi in range(N):
        f = cache / f"flips_{pi:04d}.npz"
        if f.exists():
            continue
        avail_mb = psutil.virtual_memory().available // 1048576
        if avail_mb < int(args.mem_floor_mb):
            print(f"[#307] MEM-GUARD: available {avail_mb}MB < floor {args.mem_floor_mb}MB — "
                  "exiting resumable (rc=7); relaunch to continue.", flush=True)
            sys.exit(7)
        rgb_alone, rgb_comp = render_frame1_both(bc, ctx, pi)
        gt = np.asarray(lstars_all[pi], np.int64)
        fa = segnet_argmax(seg_cpu, bc._torch_R_reference(rgb_alone, rh, rw, ch, cw)) != gt
        if rgb_comp is rgb_alone:
            fc = fa
        else:
            fc = segnet_argmax(seg_cpu, bc._torch_R_reference(rgb_comp, rh, rw, ch, cw)) != gt
        tmp = cache / f".tmp_flips_{pi:04d}.npz"
        np.savez_compressed(tmp, alone=np.packbits(fa), comp=np.packbits(fc),
                            shape=np.array([rh, rw]))
        tmp.rename(f)
        n_done += 1
        if n_done % 10 == 1 or pi == N - 1:
            el = time.time() - t0
            print(f"  [pair {pi}] d_seg_alone={fa.mean():.6f} d_seg_comp={fc.mean():.6f} "
                  f"({n_done} new, {el:.0f}s, {el / max(1, n_done):.1f}s/pair) {_rss_str()}",
                  flush=True)
        if int(getattr(args, "stop_after_new", 0)) and n_done >= int(args.stop_after_new):
            print(f"[#307] BOUNDED-VERIFY: banked {n_done} new pairs — exiting resumable "
                  f"(rc=7). {_rss_str()}", flush=True)
            sys.exit(7)
    return cache, {"lane_report_counted_bytes": int(lane_report["counted_brotli_bytes"]),
                   "manifest_self_orient": {k: manifest[k] for k in
                                            ("self_orient", "n_dir_freqs", "so_freq_across",
                                             "so_freq_along", "so_tau", "so_iters")}}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", required=True, help="dir holding the FROZEN checkpoint snapshot "
                    "(NEVER the live run dir — snapshot first; containment).")
    ap.add_argument("--npz-name", required=True)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n", type=int, default=600, help="pairs (600 = the evidence bar; smaller "
                    "values are smoke-only and the report is labeled accordingly).")
    ap.add_argument("--flips-cache-dir", required=True, help="stage-A per-pair cache (resume).")
    ap.add_argument("--stop-after-new", type=int, default=0,
                    help="bounded verification: exit rc=7 (resumable) after banking this many NEW "
                    "pairs (0 = no bound). Used to reproduce/verify the rc=137 death point in a "
                    "watched foreground run without committing to the full extraction.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=8.0,
                    help="mod32cap launch.sh trained --freq-along 8 (byte-close CLI default is 4; "
                    "pass the RUN'S trained value — never-invent-flags applies to values too).")
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--torch-threads", type=int, default=4,
                    help="cap SegNet CPU threads (protect the live trainer).")
    ap.add_argument("--mem-floor-mb", type=int, default=6144,
                    help="P0 self-guard: exit rc=7 (resumable) when available RAM < floor.")
    ap.add_argument("--ckpt-provenance", default="", help="free-form provenance string (epoch, "
                    "best d_seg, source run) stamped into the report.")
    args = ap.parse_args()

    t_start = time.time()
    cache, extraction = extract_flips(args)

    # stage B: load the cached flip maps + GT classes, code both surfaces
    N = int(args.n)
    z = np.load(args.gt_cache, allow_pickle=False)
    lstars_all = z["lstars"]
    surfaces: dict[str, dict] = {}
    for surf in ("alone", "comp"):
        flips, classes = [], []
        for pi in range(N):
            f = np.load(cache / f"flips_{pi:04d}.npz")
            h, w = (int(x) for x in f["shape"])
            fm = np.unpackbits(f[surf], count=h * w).astype(bool).reshape(h, w)
            flips.append(fm)
            classes.append(np.asarray(lstars_all[pi], np.int64))
        enc = contour_encode_frames(flips, classes)
        # NO-FAKE: decode-verify every frame bit-exact before reporting the byte count
        t0 = time.time()
        dec_f, dec_c = contour_decode_frames(enc["streams"], N, flips[0].shape[0],
                                             flips[0].shape[1])
        for pi in range(N):
            assert np.array_equal(dec_f[pi], flips[pi]), f"{surf} pair {pi}: flip map mismatch"
            assert np.array_equal(dec_c[pi][flips[pi]], classes[pi][flips[pi]]), \
                f"{surf} pair {pi}: class mismatch"
        verify_s = round(time.time() - t0, 2)
        base = bz2_fullgrid_baseline(flips, classes)
        enc.pop("streams")
        d_seg = float(np.mean([f.mean() for f in flips]))
        surfaces[surf] = {
            "d_seg_subset_mean": d_seg,
            "contour": {**enc, "decode_verified_bit_exact": True, "verify_s": verify_s},
            "bz2_fullgrid_baseline": base,
            "rate_S_contour": 25.0 * enc["total_bytes"] / RATE_DENOM,
            "go_bar": GO_BAR_B,
            "go": bool(enc["b_contour"] < GO_BAR_B),
        }
        print(f"[#307] surface={surf}: n_flips={enc['n_flips']} comps={enc['n_components']} "
              f"b_contour={enc['b_contour']:.4f} B/flip (bz2 baseline {base['b_bz2']:.4f}) "
              f"streams={surfaces[surf]['contour']['stream_bytes']} "
              f"singleton_frac={enc['singleton_flip_frac']:.3f} "
              f"coherent_ge4={enc['coherent_ge4_flip_frac']:.3f}", flush=True)

    verdict = "GO" if any(s["go"] for s in surfaces.values()) else "NO-GO"
    result = {
        "task": "#307 CONTOUR-STRING FLIP CODING (FEED-07b row 6)",
        "authority": (f"[macOS-CPU advisory] NON-PROMOTABLE — n={N} "
                      + ("(the n600 evidence bar)" if N >= 600 else "SMOKE-ONLY (below the n600 bar)")),
        "pointer": "0.19110 UNMOVED (MEANS)",
        "utc": datetime.now(UTC).isoformat(),
        "ckpt": {"dir": str(args.ckpt_dir), "npz": args.npz_name,
                 "provenance": args.ckpt_provenance},
        "n_pairs": N,
        "extraction": extraction,
        "surfaces": surfaces,
        "gate": {"threshold_b": GO_BAR_B, "verdict": verdict,
                 "prior_280_min_b": 0.876,
                 "note": ("GO -> Lever-D reactivates (positions affordable). NO-GO -> the "
                          "singleton/incoherent fraction quantifies the flicker share that no "
                          "contour coder can rescue.")},
        "elapsed_s": round(time.time() - t_start, 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n[#307] VERDICT: {verdict} (bar b<{GO_BAR_B}); wrote {args.out} "
          f"({result['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
