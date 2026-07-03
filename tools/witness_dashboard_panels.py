#!/usr/bin/env python3
"""Witness dashboard Tab-2 panel renderer — the canonical comma10k-baseline 6-panel
multipane PLUS the Yousfi/Fridrich inverse-steganalysis tribute triptych, rendered
FROM THE LIVE #205 checkpoint, as self-contained ``data:image/png`` URIs for the
CSP-locked live dashboard (tools/dashboard_server.py Tab 2 "WITNESS").

DE-ORPHAN, DON'T REBUILD (CLAUDE.md "velocity-driven orphaning"): every heavy piece
is REUSED from the canonical machinery —
  * witness load + render THROUGH the contest R operator  ->
    tools.render_comma_baseline_vs_ours_viz.{build_witness_render_context, render_our_pairs}
    (which wraps tools.levelset_byte_close_and_eval._load_levelset_ckpt + the canonical
     tac.local_acceleration.torch_levelset_inflate torch decode — float EMA shadow).
  * frozen CPU-torch SegNet argmax + top1-top2 MARGIN ->
    tac.boundary_math.seg_core.segnet_argmax_and_margin (the SAME authority L* is built with;
     NEVER MPS — macOS-CPU is ADVISORY, NON-PROMOTABLE).
  * REAL S-UNIWARD distortion (Holub-Fridrich-Denemark 2014, directional-Haar residual
    reciprocal-energy — the same approximation Yousfi 2017 uses) ->
    tac.uniward_delta.compute_uniward_cost_map.
  * GT frames + GT SegNet argmax (lstars) + GT margin field come from the shared GT cache
    (tools/build_shared_gt_cache_for_mlx_fleet.py; frame_utils.yuv420_to_rgb decode -- NEVER
     PyAV rgb24). gt_n6.npz is byte-identical to gt_n600[0:6] (verified) so the 6-panel uses
     the EXACT frames the scorer verdict uses, at ~50 MB (vs the 5 GB n600) -- light enough to
     co-exist with the live #205 GPU run.

The three tribute panels (Row C) are the campaign thesis made visible: this contest IS
inverse steganalysis. UNIWARD undetectability (Fridrich) -> SegNet detection (Yousfi) ->
our witness that SURVIVES the detector where it is most sensitive (the small-margin
codim-1 separatrix where d_seg lives).

AUTHORITY: everything here is ``[macOS-CPU advisory]`` NON-PROMOTABLE imagery. A viz moves
NO pointer (0.19110, UNMOVED); the exact row is byte-closed contest-CPU/CUDA. MEANS, not an end.

CLI (independently runnable / testable):
    .venv/bin/python tools/witness_dashboard_panels.py \
        --ckpt-dir experiments/results/levelset_n600_witness_20260703T120444Z \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n6.npz \
        --pairs 0,2,4 --out .omx/tmp/witness_panels_probe.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# self-orient render overrides == the trainer defaults == the #205 launch (--freq-across 32
# --freq-along 4). detect_self_orient does NOT persist these; the defaults reproduce the
# trained forward faithfully (our_dseg through R matches the run's verdict d_seg).
_SO_OVERRIDES = {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4}

_DEFAULT_EMA = "levelset_witness_ema_mlx.npz"


def _png_data_uri(fig, dpi: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.10)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


class WitnessPanelRenderer:
    """Lazy-loads + CACHES the frozen SegNet scorer and the GT cache once (they never change),
    and reloads only the witness checkpoint each render (the weights DO change every checkpoint
    -- that is the progress we visualize). Thread-safe for the dashboard executor: uses
    matplotlib's object API (Figure), never the pyplot global state."""

    def __init__(self, gt_cache: str | os.PathLike, pair_indices: list[int]):
        self.gt_cache = str(gt_cache)
        self.pair_indices = list(pair_indices)
        self._seg = None
        self._seg_argmax = None
        self._gt = None  # dict of the arrays we need
        # palette + labels reused from the canonical comma viz (DRY)
        from tools.render_comma_baseline_vs_ours_viz import (
            SEG_LABELS,
            SEG_PALETTE_HEX,
            colorize_seg,
        )
        self._SEG_LABELS = SEG_LABELS
        self._SEG_PALETTE_HEX = SEG_PALETTE_HEX
        self._colorize_seg = colorize_seg

    # -- lazy load the fixed pieces (once) --
    def _ensure_loaded(self) -> None:
        if self._seg is None:
            from tools.render_comma_baseline_vs_ours_viz import _load_scorers
            self._seg, self._seg_argmax, _ = _load_scorers(False)  # SegNet only (pose not needed)
        if self._gt is None:
            z = np.load(self.gt_cache)
            n = int(z["gt_f1"].shape[0])
            idx = [i for i in self.pair_indices if 0 <= i < n]
            if not idx:
                raise ValueError(f"no valid pair indices {self.pair_indices} for gt cache with {n} pairs")
            self.pair_indices = idx
            self._gt = {
                "n": n,
                "gt_f1": {i: np.asarray(z["gt_f1"][i], np.uint8) for i in idx},
                "lstars": {i: np.asarray(z["lstars"][i], np.int64) for i in idx},
                "margins": ({i: np.asarray(z["margins"][i], np.float64) for i in idx}
                            if "margins" in z.files else {}),
            }
            z.close()

    # -- one 3x3 comma multipane + tribute figure for a single pair --
    def _pair_figure(self, pi: int, gt_f1, our_f1, gt_lstar, our_lstar, our_margin,
                     gt_margin, uni, our_dseg: float, epoch, dpi: int) -> str:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.patches import Patch

        BG = "#1b1e24"
        FG = "#d8dde6"
        MUT = "#8b93a3"

        fig = Figure(figsize=(11.0, 8.7), facecolor=BG)

        def _panel(r, c, title, tcolor=FG):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1)
            ax.set_facecolor(BG)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#2a2f39")
            ax.set_title(title, color=tcolor, fontsize=9.0, pad=4, loc="left")
            return ax

        # ---- Row A: comma lineage — GT | ours | pixel error ----
        _panel(0, 0, "GT frame1 (comma decode)").imshow(gt_f1)
        _panel(0, 1, "our witness render (through R)").imshow(our_f1)
        err = np.abs(gt_f1.astype(np.float64) - our_f1.astype(np.float64)).mean(-1)
        _panel(0, 2, "pixel error |GT - ours|").imshow(
            err, cmap="hot", vmin=0, vmax=max(1.0, np.percentile(err, 99)))

        # ---- Row B: the scorer's eye — GT masks | our masks | disagreement (d_seg pixels) ----
        _panel(1, 0, "GT SegNet argmax (L*)").imshow(self._colorize_seg(gt_lstar))
        _panel(1, 1, "our render SegNet argmax").imshow(self._colorize_seg(our_lstar))
        dis = (our_lstar != gt_lstar)
        dis_rgb = np.zeros((*dis.shape, 3), np.uint8)
        dis_rgb[..., 0] = np.where(dis, 255, 20)  # red where disagree, near-black elsewhere
        dis_rgb[..., 1] = np.where(dis, 40, 22)
        dis_rgb[..., 2] = np.where(dis, 40, 26)
        _panel(1, 2, f"disagreement = d_seg pixels ({our_dseg*100:.3f}%)", "#ff8b8b").imshow(dis_rgb)

        # ---- Row C: the TRIBUTE triptych — margin field | UNIWARD cost | class legend ----
        # rho_seg: the SegNet top1-top2 margin. SMALL margin = fragile = the codim-1 separatrix
        # where the argmax can flip = where d_seg lives. We show FRAGILITY (bright = small margin).
        mfield = gt_margin if gt_margin is not None else our_margin
        vmax = float(np.percentile(mfield, 96)) or 1.0
        frag = np.clip(1.0 - mfield / (vmax + 1e-9), 0.0, 1.0)
        _panel(2, 0, "SegNet margin field ρ_seg (bright = fragile separatrix)", "#ffb454").imshow(
            frag, cmap="inferno", vmin=0, vmax=1)
        # rho_uniward: REAL S-UNIWARD cost on OUR render (Holub-Fridrich 2014). High = textured =
        # undetectable-to-embed (Fridrich principle #1). log1p + percentile stretch so the
        # textured regions (lane markings, lights, tree/edge structure) pop against the smooth
        # (low-texture, high-embedding-cost) background of a night scene.
        uni_log = np.log1p(uni)
        _panel(2, 1, "UNIWARD cost ρ_uniward on our render", "#7fe0a0").imshow(
            uni_log, cmap="viridis",
            vmin=float(np.percentile(uni_log, 40)), vmax=float(np.percentile(uni_log, 99.5)))
        # class legend (9th cell)
        c2 = _panel(2, 2, "comma10k classes", MUT)
        c2.set_frame_on(False)
        handles = [Patch(facecolor=self._SEG_PALETTE_HEX[i], edgecolor="#2a2f39",
                         label=self._SEG_LABELS[i]) for i in range(len(self._SEG_LABELS))]
        c2.legend(handles=handles, loc="center", frameon=False, fontsize=9,
                  labelcolor=FG, handlelength=1.3, borderpad=0.6, labelspacing=0.7)

        fig.suptitle(
            f"pair {pi}  ·  epoch {epoch if epoch is not None else '?'}  "
            f"·  d_seg {our_dseg:.5f}  ·  [macOS-CPU advisory · NON-PROMOTABLE]",
            color=FG, fontsize=10.5, y=0.995)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.12)
        return _png_data_uri(fig, dpi)

    def render(self, ckpt_dir: str | os.PathLike, npz_name: str | None = None,
               epoch: int | None = None, dpi: int = 80) -> dict[str, Any]:
        """Render the witness panels for the configured pairs from the given checkpoint dir.
        Returns a payload dict (JSON-safe) with per-pair ``data:image/png`` URIs + metadata."""
        import torch

        from tac.uniward_delta import compute_uniward_cost_map
        from tools.render_comma_baseline_vs_ours_viz import (
            build_witness_render_context,
            render_our_pairs,
        )

        t0 = time.time()
        self._ensure_loaded()
        ckpt_dir = Path(ckpt_dir)
        ema = ckpt_dir / (npz_name or _DEFAULT_EMA)
        ckpt_mtime = ema.stat().st_mtime if ema.exists() else None

        manifest, params, code_np = build_witness_render_context(ckpt_dir, npz_name, _SO_OVERRIDES)
        idx = sorted(self.pair_indices)
        lo, hi = idx[0], idx[-1] + 1
        if manifest["n_pairs"] < hi:
            idx = [i for i in idx if i < manifest["n_pairs"]]
            if not idx:
                raise ValueError(f"witness has {manifest['n_pairs']} pairs < requested {self.pair_indices}")
            lo, hi = idx[0], idx[-1] + 1
        rendered = render_our_pairs(manifest, params, code_np, lo, hi)  # contiguous [lo,hi)

        pairs_out: list[dict[str, Any]] = []
        dsegs: list[float] = []
        for pi in idx:
            our_f0, our_f1 = rendered[pi - lo]
            gt_f1 = self._gt["gt_f1"][pi]
            gt_lstar = self._gt["lstars"][pi]
            gt_margin = self._gt["margins"].get(pi)
            our_lstar, our_margin = self._seg_argmax(self._seg, our_f1)
            our_lstar = np.asarray(our_lstar, np.int64)
            our_dseg = float((our_lstar != gt_lstar).mean())
            dsegs.append(our_dseg)
            fr = torch.from_numpy(our_f1.astype(np.float32)).permute(2, 0, 1)[None]  # (1,3,H,W)
            with torch.inference_mode():
                uni = compute_uniward_cost_map(fr).squeeze(0).numpy()
            uri = self._pair_figure(pi, gt_f1, our_f1, gt_lstar, our_lstar, our_margin,
                                    gt_margin, uni, our_dseg, epoch, dpi)
            pairs_out.append({"pair_idx": pi, "our_dseg": round(our_dseg, 6), "panel": uri})

        return {
            "ok": True,
            "epoch": epoch,
            "ckpt_dir": str(ckpt_dir),
            "ckpt_npz": ema.name,
            "ckpt_mtime": ckpt_mtime,
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gt_cache": os.path.basename(self.gt_cache),
            "n_pairs_shown": len(pairs_out),
            "mean_dseg": (round(float(np.mean(dsegs)), 6) if dsegs else None),
            "render_secs": round(time.time() - t0, 2),
            "authority": "macOS-CPU advisory · NON-PROMOTABLE",
            "pairs": pairs_out,
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--npz-name", default=None)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n6.npz")
    ap.add_argument("--pairs", default="0,2,4", help="comma-separated pair indices")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--out", default=None, help="write the payload JSON here (durable path; NOT /tmp)")
    a = ap.parse_args(argv)

    pairs = [int(x) for x in str(a.pairs).split(",") if x.strip() != ""]
    r = WitnessPanelRenderer(a.gt_cache, pairs)
    payload = r.render(a.ckpt_dir, a.npz_name, a.epoch, a.dpi)
    sizes = [len(p["panel"]) for p in payload["pairs"]]
    print(json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=2))
    print(f"panel data-uri sizes (bytes): {sizes}  total={sum(sizes)}  "
          f"pairs={[p['pair_idx'] for p in payload['pairs']]}  "
          f"dsegs={[p['our_dseg'] for p in payload['pairs']]}", flush=True)
    if a.out:
        outp = Path(a.out)
        # Refuse SYSTEM temp (/tmp, /private/tmp, /var/tmp) per CLAUDE.md — but ALLOW the
        # sanctioned repo-local ``.omx/tmp/`` ephemeral scratch (resolve first so a substring
        # like ``.omx/tmp/`` is not mistaken for system /tmp).
        if str(outp.resolve()).startswith(("/tmp", "/private/tmp", "/var/tmp")):
            raise ValueError("--out must be a durable path, not system /tmp (CLAUDE.md)")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload))
        print(f"wrote {outp} ({outp.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
