#!/usr/bin/env python3
"""FLOW (Tab 3) n600 VIDEO sequence renderer + WITNESS (Tab 2) hardest-pair selector.

Why this exists (operator 2026-07-03): Tab 3 FLOW must be the FULL n600 drive on a
TEMPORAL timeline (frame index 0->599, play = animate the segment), NOT a 3-pair
dropdown + epoch scrubber. Tab 2 WITNESS must show the HARDEST + most-DIVERSE pairs
(labelled by failure mode), NOT three near-identical consecutive frames.

ONE heavy governed pass over all 600 pairs from the BEST checkpoint feeds BOTH tabs:
  * the WITNESS's OWN render (INR forward -> R) + its OWN 5-class argmax partition are
    CHEAP (no SegNet) -> the smooth video base layers, all 600.
  * the SegNet-of-render argmax + per-pair d_seg (what the SCORER sees + hardness) come
    from the frozen CPU-torch SegNet forward on each rendered frame1 -> heavier, but
    per-frame bounded memory (chunk-of-1 loop; NEVER a 600-wide batch that spikes +66GiB).
  * GT SegNet argmax (lstars) comes from gt_n600.npz via MMAP (per-pair slice; the 5 GB
    cache is never fully resident).

DE-ORPHAN, DON'T REBUILD: the render is the canonical witness-through-R primitives
(tools.render_comma_baseline_vs_ours_viz.build_witness_render_context +
tac.local_acceleration.torch_levelset_inflate.torch_* -- the SAME forward the inflate/
verdict path runs); the SegNet argmax+margin is tac.boundary_math.seg_core via
render_comma_baseline_vs_ours_viz._load_scorers; the Tab-2 6-panel is
tools.witness_dashboard_panels.WitnessPanelRenderer. Palette = the canonical comma10k
SEG_PALETTE (Yousfi/comma reads these colors instantly) from render_comma_baseline_vs_ours_viz.

GOVERNED + CONTROL-PLANE-SAFE BY CONSTRUCTION: this runs as a DETACHED subprocess in its
OWN process group (own tools/safe_run.py RSS/timeout cap), so its ~2.6 GB torch+SegNet
footprint is NEVER summed into the lean dashboard's safe_run group -> it can NEVER crash
the dashboard or the machine. It yields to the live #205 GPU run under a memory floor
(--min-free-gib) and logs #205 liveness before + after. NEVER MPS (macOS-CPU authority),
NEVER competes for the GPU.

AUTHORITY: everything emitted is ``[macOS-CPU advisory · NON-PROMOTABLE]`` imagery. A viz
moves NO pointer (0.19110, UNMOVED); the exact row is byte-closed contest-CPU/CUDA. MEANS.

CLI (independently runnable / testable; start small with --n-pairs 24):
    .venv/bin/python tools/dashboard_flow_sequence.py \
        --ckpt-dir experiments/results/levelset_n600_witness_20260703T120444Z \
        --npz-name levelset_witness_ema_BEST.npz \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
        --out-stem .omx/tmp/dash_flow_seq/probe --n-pairs 24 --epoch 999
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# self-orient render overrides == the trainer defaults == the #205 launch (freq_across 32
# freq_along 4). detect_self_orient does NOT persist these; the defaults reproduce the
# trained forward faithfully (our d_seg through R matches the run verdict).
_SO_OVERRIDES = {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4}
from tac import witness_run_artifacts as _wra  # noqa: E402  (after sys.path setup)
_DEFAULT_BEST = _wra.EMA_BEST_NPZ
_FALLBACK_EMA = _wra.EMA_NPZ
_FLOW_FPS = 12  # temporal-timeline playback rate (~real-ish, loops)


def _log(**kw) -> None:
    kw.setdefault("stage", "flow_sequence")
    print(json.dumps(kw), flush=True)


def _free_gib() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)  # RAW_VM_BASIS_OK:dashboard telemetry display, not a refuse/admit guard
    except Exception:
        return None


def _run_alive(token: str | None) -> bool | None:
    """Is the live #205 run still alive? pgrep -f <token>. None if no token / pgrep absent."""
    if not token:
        return None
    try:
        r = subprocess.run(["pgrep", "-f", token], capture_output=True, text=True, timeout=5)  # subprocess-no-check-OK: pgrep rc=1 = no match, a valid answer; failure degrades to None
        return bool(r.stdout.strip())
    except Exception:
        return None


# ---- palette / labels (canonical comma10k; imported, never invented) ----
def _seg_palette():
    from tools.render_comma_baseline_vs_ours_viz import SEG_LABELS, SEG_PALETTE_HEX
    return SEG_LABELS, SEG_PALETTE_HEX


# ---------------------------------------------------------------------------
# failure-mode classification (honest; derived from the disagreement composition)
# ---------------------------------------------------------------------------
def classify_failure_mode(our_lstar: np.ndarray, gt_lstar: np.ndarray) -> tuple[str, dict]:
    """Tag a pair's dominant d_seg failure mode from the FULL-res disagreement map.
    Honest: the tag names the dominant GT class of the flipped pixels + their spatial
    signature -- never fabricated. Canonical classes: 0 Road 1 Lane 2 Undrivable
    3 Movable 4 MyCar (comma10k order)."""
    dis = our_lstar != gt_lstar
    H, W = our_lstar.shape
    tot = int(dis.sum())
    if tot == 0:
        return "clean (no flips)", {"area": 0.0}
    gt_at = gt_lstar[dis]
    fr = {c: float((gt_at == c).mean()) for c in range(5)}
    rows = np.where(dis)[0]
    mean_row = float(rows.mean()) / H  # 0 = top/far, 1 = bottom/near
    area = tot / (H * W)
    if fr[3] >= 0.15:
        mode = "adjacent / movable vehicle"
    elif fr[1] >= 0.22:
        mode = "lane-marking dash (rare-class)"
    elif area < 0.004 and mean_row < 0.45:
        mode = "distant small object"
    elif fr[2] >= 0.40:
        mode = "undrivable / sky boundary"
    else:
        mode = "road-edge boundary flips"
    info = {"fr": {k: round(v, 3) for k, v in fr.items()}, "mean_row": round(mean_row, 3),
            "area": round(area, 5)}
    return mode, info


def select_hard_diverse(per_pair_dseg: list[float], per_pair_mode: list[str],
                        k: int = 6) -> list[int]:
    """Pick k pairs that are HARD, spread across the whole drive, and cover distinct
    failure modes where they genuinely occur.

    1. TIMELINE SPREAD: the hardest pair in each of k equal timeline bins -> never
       clustered, each a real local worst-case.
    2. MODE DIVERSITY: if that set covers few failure modes but other modes exist among
       genuinely-hard pairs, swap the WEAKEST duplicate-mode pick for the hardest pair of
       a MISSING mode (keeping a timeline gap so we do not re-cluster). Honest: nothing is
       injected unless a genuinely-hard pair of that mode exists."""
    n = len(per_pair_dseg)
    if n == 0:
        return []
    k = max(1, min(k, n))
    edges = np.linspace(0, n, k + 1).astype(int)
    picks: list[int] = []
    for b in range(k):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        picks.append(max(range(lo, hi), key=lambda i: per_pair_dseg[i]))
    if len(picks) < k:  # some bin empty (n<k) -> backfill with next-hardest unpicked
        for i in sorted(range(n), key=lambda i: -per_pair_dseg[i]):
            if i not in picks:
                picks.append(i)
            if len(picks) >= k:
                break
    picks = sorted(set(picks))
    # mode-diversity injection (bounded, honest)
    if picks:
        thr = float(np.median([per_pair_dseg[i] for i in picks]))
        min_gap = max(1, n // (k * 3))
        order = sorted(range(n), key=lambda i: -per_pair_dseg[i])
        for i in order:
            m = per_pair_mode[i]
            if per_pair_dseg[i] < thr * 0.6:
                break  # remaining candidates too easy to be worth injecting
            present = {per_pair_mode[p] for p in picks}
            if m in present:
                continue
            dups = [p for p in picks
                    if sum(1 for q in picks if per_pair_mode[q] == per_pair_mode[p]) > 1
                    and abs(p - i) >= min_gap]
            if not dups:
                continue
            weakest = min(dups, key=lambda p: per_pair_dseg[p])
            picks.remove(weakest)
            picks.append(i)
    return sorted(set(picks))


# ---------------------------------------------------------------------------
# the witness render (INR forward -> R) + its OWN 5-class argmax partition.
# reuses the canonical torch primitives; captures phi1.argmax (free, same forward).
# ---------------------------------------------------------------------------
class _WitnessForward:
    def __init__(self, ckpt_dir: Path, npz_name: str | None):
        from tools.render_comma_baseline_vs_ours_viz import build_witness_render_context
        self.manifest, params, self.code_np = build_witness_render_context(
            ckpt_dir, npz_name, _SO_OVERRIDES)
        import torch
        from tac.local_acceleration.torch_levelset_inflate import (
            coords_grid, curvelet_B, curvelet_feats,
        )
        m = self.manifest
        self.rh, self.rw = int(m["render_h"]), int(m["render_w"])
        self.ch, self.cw = int(m["camera_h"]), int(m["camera_w"])
        self.coords = coords_grid(self.rh, self.rw)
        B = curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"],
                       m["bank_n_iso"], m["max_bank_freq"])
        self.curv = curvelet_feats(self.coords, B)
        self.P = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in params.items()}
        self.code = torch.as_tensor(self.code_np, dtype=torch.float32)

    @property
    def n_pairs(self) -> int:
        return int(self.manifest["n_pairs"])

    def render_pair(self, pi: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (f1_camera_uint8 [ch,cw,3], wit_partition_argmax [rh,rw] uint8) for pair pi.
        wit_partition = the witness's OWN 5-class argmax (phi1) at render res -- CHEAP, from
        the same forward, no SegNet."""
        import torch
        from tac.local_acceleration.torch_levelset_inflate import (
            dir_feats, torch_in_proj_h0, torch_outputs_from_h0, torch_R,
        )
        m = self.manifest
        rh, rw = self.rh, self.rw
        if m["self_orient"]:
            dirf = np.zeros((self.curv.shape[0], 4 * int(m["n_dir_freqs"])), np.float32)
            prev = None
            for _ in range(int(m["so_iters"])):
                ft = torch.as_tensor(np.concatenate([self.curv, dirf], -1), dtype=torch.float32)
                phi, _ = torch_outputs_from_h0(self.P, torch_in_proj_h0(self.P, ft, m),
                                               self.code[2 * pi + 1], m, False)
                am = phi.argmax(-1).reshape(rh, rw).cpu().numpy().astype(np.int64)
                if prev is not None and np.array_equal(am, prev):
                    break
                dirf = dir_feats(self.coords, am, m["n_dir_freqs"], m["so_freq_along"],
                                 m["so_freq_across"], m["so_tau"])
                prev = am
            feats = np.concatenate([self.curv, dirf], -1)
        else:
            feats = self.curv
        h0 = torch_in_proj_h0(self.P, torch.as_tensor(feats, dtype=torch.float32), m)
        phi1, rgb1 = torch_outputs_from_h0(self.P, h0, self.code[2 * pi + 1], m, True)
        wit = phi1.argmax(-1).reshape(rh, rw).cpu().numpy().astype(np.uint8)
        f1 = np.asarray(torch_R(rgb1, rh, rw, self.ch, self.cw), np.uint8)
        return f1, wit


def _rgb_png_b64(wit: np.ndarray, seg: np.ndarray, gt: np.ndarray) -> str:
    """Pack the three class-index fields into ONE OPAQUE RGB PNG: R=witness partition,
    G=SegNet argmax, B=GT argmax (all class 0..4). OPAQUE (no alpha) so the browser's decode
    never premultiplies and corrupts the exact class indices; PNG crushes the low entropy."""
    from PIL import Image
    rgb = np.stack([wit, seg, gt], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gray_png_b64(frag: np.ndarray) -> str:
    """The margin-fragility field as an OPAQUE grayscale (L) PNG (0..255, pre-quantized for
    compressibility). Separate from the class-index RGB so neither is alpha-premultiplied."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frag, np.uint8), "L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_flow_sequence(
    ckpt_dir: Path, npz_name: str | None, gt_cache: str, epoch: int | None,
    n_pairs: int | None = None, downsample: int = 4, jpeg_quality: int = 62,
    frag_levels: int = 32, hard_k: int = 6, min_free_gib: float = 5.0,
    run_token: str | None = None, progress_every: int = 25,
    stop_cb: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """The heavy governed 600-pass. Returns {"flow": <Tab3 payload>, "witness_pairs":
    [selected pair dicts], "meta": {...}}. Raises RuntimeError('low_free_ram') to abort
    (caller retries later)."""
    from PIL import Image

    from tools.render_comma_baseline_vs_ours_viz import _load_scorers

    t0 = time.time()
    ckpt_dir = Path(ckpt_dir)
    ema = ckpt_dir / (npz_name or _DEFAULT_BEST)
    if not ema.exists():
        ema = ckpt_dir / _FALLBACK_EMA
    npz_used = ema.name
    _log(event="start", ckpt=str(ema), gt_cache=gt_cache, run205_alive=_run_alive(run_token))

    fwd = _WitnessForward(ckpt_dir, npz_used)
    seg, seg_argmax, _ = _load_scorers(False)
    SEG_LABELS, SEG_PALETTE_HEX = _seg_palette()

    z = np.load(gt_cache, mmap_mode="r")  # 5 GB -> mmap; per-pair slice only
    lstars = z["lstars"]
    n_avail = min(int(z["n_pairs"]) if "n_pairs" in z.files else lstars.shape[0], fwd.n_pairs)
    n = n_avail if n_pairs is None else min(int(n_pairs), n_avail)

    ds = max(1, int(downsample))
    frames: list[dict[str, Any]] = []
    per_pair_dseg: list[float] = []
    per_pair_mode: list[str] = []
    per_pair_info: list[dict] = []
    fw = fh = 0
    fq = max(2, int(frag_levels))
    qstep = 256 // fq

    for pi in range(n):
        if stop_cb is not None and stop_cb():
            raise RuntimeError("stopped")
        if pi % progress_every == 0:
            free = _free_gib()
            if free is not None and free < min_free_gib:
                _log(event="abort_low_free_ram", free_gib=round(free, 2), pair=pi)
                raise RuntimeError("low_free_ram")
            _log(event="progress", pair=pi, n=n, free_gib=(round(free, 2) if free else None),
                 secs=round(time.time() - t0, 1))
        f1, wit = fwd.render_pair(pi)                        # camera-res render + own partition
        our_lstar, our_margin = seg_argmax(seg, f1)          # SegNet-of-render (heavy, per-frame)
        our_lstar = np.asarray(our_lstar, np.int64)
        gt_lstar = np.asarray(lstars[pi], np.int64)
        dseg = float((our_lstar != gt_lstar).mean())         # authoritative full-res per-pair d_seg
        mode, info = classify_failure_mode(our_lstar, gt_lstar)
        per_pair_dseg.append(round(dseg, 6))
        per_pair_mode.append(mode)
        per_pair_info.append(info)

        # downsample fields (nearest for class indices; margin->fragility u8, quantized)
        wd = np.asarray(wit, np.uint8)[::ds, ::ds]
        sd = np.asarray(our_lstar, np.uint8)[::ds, ::ds]
        gd = np.asarray(gt_lstar, np.uint8)[::ds, ::ds]
        vmax = float(np.percentile(our_margin, 96)) or 1.0
        frag = np.clip(1.0 - np.asarray(our_margin, np.float32) / (vmax + 1e-9), 0.0, 1.0)
        fd = (frag[::ds, ::ds] * 255.0).astype(np.uint8)
        fd = (fd // qstep * qstep).astype(np.uint8)          # quantize for PNG compressibility
        fh, fw = sd.shape
        # crop all to the common min grid (guards odd-size off-by-one)
        H0 = min(wd.shape[0], sd.shape[0], gd.shape[0], fd.shape[0])
        W0 = min(wd.shape[1], sd.shape[1], gd.shape[1], fd.shape[1])
        wd, sd, gd, fd = wd[:H0, :W0], sd[:H0, :W0], gd[:H0, :W0], fd[:H0, :W0]
        fh, fw = H0, W0
        # render layer: downsample camera-res f1 to the field grid (bilinear photo) -> JPEG
        rimg = Image.fromarray(f1).resize((W0, H0), Image.BILINEAR)
        frames.append({
            "i": pi,
            "dseg": round(dseg, 6),
            "mode": mode,
            "render_b64": base64.b64encode(_pil_jpeg(rimg, jpeg_quality)).decode("ascii"),
            "fields_b64": _rgb_png_b64(wd, sd, gd),   # RGB: R=witness partition, G=SegNet, B=GT
            "frag_b64": _gray_png_b64(fd),            # L: margin fragility 0..255
        })

    z.close()
    hard = select_hard_diverse(per_pair_dseg, per_pair_mode, k=hard_k)
    mean_dseg = round(float(np.mean(per_pair_dseg)), 6) if per_pair_dseg else None
    flow = {
        "ok": True, "kind": "flow_sequence",
        "epoch": epoch, "ckpt_npz": npz_used,
        "n": n, "w": int(fw), "h": int(fh), "downsample": ds, "fps": _FLOW_FPS,
        "classes": [{"i": i, "label": SEG_LABELS[i], "hex": SEG_PALETTE_HEX[i]}
                    for i in range(len(SEG_LABELS))],
        "mean_dseg": mean_dseg,
        "hardest": [{"i": i, "dseg": per_pair_dseg[i], "mode": per_pair_mode[i]} for i in hard],
        "per_pair_dseg": per_pair_dseg,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "macOS-CPU advisory · NON-PROMOTABLE",
        "render_secs": round(time.time() - t0, 1),
        "frames": frames,
    }
    witness_pairs = [{"pair_idx": i, "our_dseg": per_pair_dseg[i], "mode": per_pair_mode[i],
                      "info": per_pair_info[i]} for i in hard]
    _log(event="done", n=n, secs=round(time.time() - t0, 1), mean_dseg=mean_dseg,
         hard=hard, run205_alive=_run_alive(run_token))
    return {"flow": flow, "witness_pairs": witness_pairs,
            "meta": {"epoch": epoch, "n": n, "ckpt_npz": npz_used, "mean_dseg": mean_dseg}}


def _pil_jpeg(rimg, quality: int) -> bytes:
    buf = io.BytesIO()
    rimg.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def render_witness_panels(ckpt_dir: Path, npz_name: str | None, gt_cache: str,
                          epoch: int | None, witness_pairs: list[dict], dpi: int = 80) -> dict:
    """Render the Tab-2 6-panel + tribute figures for the SELECTED hard/diverse pairs,
    labelled with per-pair d_seg + failure-mode tag. Reuses WitnessPanelRenderer against
    gt_n600 (MMAP) + the BEST checkpoint."""
    from tools.witness_dashboard_panels import WitnessPanelRenderer
    pair_idx = [int(p["pair_idx"]) for p in witness_pairs]
    labels = {int(p["pair_idx"]): p.get("mode", "") for p in witness_pairs}
    r = WitnessPanelRenderer(gt_cache, pair_idx)
    payload = r.render(ckpt_dir, npz_name, epoch, dpi, emit_fields=False, labels=labels)
    return payload


def run_and_write(args) -> int:
    """Full pass: render flow sequence + witness panels, write <stem>.flow.json +
    <stem>.witness.json + <stem>.done (atomic tmp+rename, done last)."""
    stem = Path(args.out_stem)
    if str(stem.resolve()).startswith(("/tmp", "/private/tmp", "/var/tmp")):
        raise ValueError("--out-stem must be durable, not system /tmp (CLAUDE.md)")
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.set_num_threads(max(1, int(args.torch_threads)))
    except Exception:
        pass
    npz = args.npz_name
    ckpt_dir = Path(args.ckpt_dir)
    try:
        out = render_flow_sequence(
            ckpt_dir, npz, args.gt_cache, args.epoch, n_pairs=args.n_pairs,
            downsample=args.downsample, jpeg_quality=args.jpeg_quality,
            frag_levels=args.frag_levels, hard_k=args.hard_k, min_free_gib=args.min_free_gib,
            run_token=args.run_token, progress_every=args.progress_every)
    except RuntimeError as exc:
        _log(event="render_aborted", reason=str(exc))
        return 3 if "ram" in str(exc) else 4
    # Tab-2 panels for the selected pairs
    try:
        wit = render_witness_panels(ckpt_dir, npz, args.gt_cache, args.epoch,
                                    out["witness_pairs"], dpi=args.dpi)
    except Exception as exc:
        _log(event="witness_panels_error", err=str(exc))
        wit = {"ok": False, "err": str(exc), "status": "error"}

    def _atomic(path: Path, obj) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, path)

    flow_path = stem.with_suffix(".flow.json")
    wit_path = stem.with_suffix(".witness.json")
    done_path = stem.with_suffix(".done")
    _atomic(flow_path, out["flow"])
    _atomic(wit_path, wit)
    _atomic(done_path, {"epoch": args.epoch, "n": out["meta"]["n"],
                        "ckpt_npz": out["meta"]["ckpt_npz"],
                        "mean_dseg": out["meta"]["mean_dseg"],
                        "flow_bytes": flow_path.stat().st_size,
                        "witness_bytes": wit_path.stat().st_size,
                        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    _log(event="written", flow=str(flow_path), flow_bytes=flow_path.stat().st_size,
         witness=str(wit_path), witness_bytes=wit_path.stat().st_size)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--npz-name", default=_DEFAULT_BEST)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--out-stem", required=True,
                    help="output path stem -> <stem>.flow.json / .witness.json / .done")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--n-pairs", type=int, default=None, help="default: all available")
    ap.add_argument("--downsample", type=int, default=4, help="field downsample (384x512 -> 96x128)")
    ap.add_argument("--jpeg-quality", type=int, default=62)
    ap.add_argument("--frag-levels", type=int, default=32)
    ap.add_argument("--hard-k", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--min-free-gib", type=float, default=5.0)
    ap.add_argument("--run-token", default=None, help="pgrep token for #205 liveness logging")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--torch-threads", type=int, default=3)
    a = ap.parse_args(argv)
    return run_and_write(a)


if __name__ == "__main__":
    raise SystemExit(main())
