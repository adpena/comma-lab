#!/usr/bin/env python3
"""ORACLE dashboard tab renderer — "the detector I built, and the world it reads".

The ORACLE tab is the FIRST panel of Yousfi's detection-game arc: the frozen
contest scorer + the openpilot physical priors + the detectability field — the
world the witness must survive in, BEFORE we look at the witness itself.

DE-ORPHAN, DON'T REBUILD (CLAUDE.md "velocity-driven orphaning"): every physical
prior is the REAL in-tree module, wired here for the first time to a viz —

  * openpilot lane-polynomial -> analytic band (the d_seg physical prior) ->
    tac.boundary_math.analytic_lane_render_band.build_analytic_lane_band_prior
    (openpilot deg-3 centerline fit to the GT class-1 argmax + AA-SDF
    range-dependent dash coverage; the FREE inflate-time rasterizer, rule 118).
  * ego-motion screw xi (the d_pose physical prior, DUAL-USE with d_seg) ->
    tac.boundary_math.ego_xi_trajectory.LaneOptimalEgoEstimator over
    analytic_lane_render_band._pack_pairs_to_matrix (the SE(3) twist that warps
    the partition AND is the pose — Chasles). se(3) engine = tac.lie.
  * static-class ground-plane structure ->
    tac.boundary_math.road_horizon_component.{classify_segnet_regions,
    fit_horizon_line, rasterize_sky_above_line} (road/sky horizon) +
    tac.boundary_math.hood_static_component.compute_static_hood_mask (ego hood).
  * SegNet detectability field (the detector's own sensitivity map) = the
    top1-top2 argmax MARGIN, bright where the argmax can flip = where d_seg lives.

The message: openpilot is the UNIFIED FREE physical prior for BOTH scored axes
(lane band -> d_seg, ego-xi -> d_pose). Memory
``openpilot_unified_physical_prior_both_scored_axes``.

Everything is read from the shared GT cache (gt_n6.npz: gt_f1 frames + lstars GT
SegNet argmax + margins + gt_poses) — the EXACT frames the scorer verdict uses,
~48 MB (vs the 5 GB n600), light enough to co-exist with the live #205 GPU run.
NO SegNet load, NO witness checkpoint, NO torch, NO GPU — pure numpy + matplotlib.

AUTHORITY: everything here is ``[macOS-CPU advisory]`` NON-PROMOTABLE imagery. A
viz moves NO pointer (0.19110, UNMOVED). The exact row is byte-closed on
contest-CPU/CUDA. MEANS, not an end.

CLI (independently runnable / testable — the governed detached path runs THIS):
    .venv/bin/python tools/oracle_dashboard_panels.py \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n6.npz \
        --frames 0,2,4 --out .omx/tmp/dash_oracle/oracle_probe.json
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

# comma10k palette (the colors Yousfi + comma read instantly) — reused from the canonical viz.
from tools.render_comma_baseline_vs_ours_viz import SEG_LABELS, SEG_PALETTE_HEX  # noqa: E402

BG = "#1b1e24"
FG = "#d8dde6"
MUT = "#8b93a3"
ACC = "#5ab0ff"
LANE = "#7fe0a0"
POSE = "#ffb454"


def _png_data_uri(fig, dpi: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.10)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _resize_rgb_to(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor RGB resize (H,W,3)->(h,w,3) — no scipy/PIL dep; display-only."""
    h, w = hw
    H, W = frame.shape[:2]
    ri = (np.linspace(0, H - 1, h)).astype(np.int64)
    ci = (np.linspace(0, W - 1, w)).astype(np.int64)
    return frame[ri][:, ci]


class OracleRenderer:
    """Renders the ORACLE physical-prior atlas from the shared GT cache. Loads the
    cache ONCE; role detection (road/lane/sky/movable/hood — NO hardcoded indices)
    runs over ALL cached frames for a stable, self-detected class assignment. Uses
    matplotlib's Figure object API (thread/process-safe, never pyplot globals)."""

    def __init__(self, gt_cache: str | os.PathLike):
        self.gt_cache = str(gt_cache)
        self._gt: dict | None = None
        self._roles = None
        self._xi = None
        self._lane_stats: dict = {}

    def _ensure_loaded(self) -> None:
        if self._gt is not None:
            return
        z = np.load(self.gt_cache)
        try:
            gt = {
                "n": int(z["gt_f1"].shape[0]),
                "gt_f1": np.asarray(z["gt_f1"], np.uint8),
                "lstars": np.asarray(z["lstars"], np.int64),
                "margins": (np.asarray(z["margins"], np.float32)
                            if "margins" in z.files else None),
            }
        finally:
            z.close()
        self._gt = gt

    # -- physical-prior computations (all REAL in-tree modules) --
    def _detect_roles(self):
        if self._roles is None:
            from tac.boundary_math.road_horizon_component import classify_segnet_regions
            self._roles = classify_segnet_regions(self._gt["lstars"])
        return self._roles

    def _ego_xi(self):
        """LANE-OPTIMAL per-pair ego trajectory over the cached sequence — the SE(3)
        screw twist that is BOTH the partition-warp (d_seg) and the pose (d_pose)."""
        if self._xi is None:
            from tac.boundary_math.analytic_lane_render_band import (
                LaneBandRenderConfig,
                _pack_pairs_to_matrix,
                build_lane_band_pairs_from_lstars,
            )
            from tac.boundary_math.ego_xi_trajectory import LaneOptimalEgoEstimator
            roles = self._detect_roles()
            cfg = LaneBandRenderConfig(lane_cls=int(roles.lane))
            pairs_lines, self._lane_stats = build_lane_band_pairs_from_lstars(
                list(self._gt["lstars"]), cfg)
            M, presence, K = _pack_pairs_to_matrix(pairs_lines)
            self._xi = LaneOptimalEgoEstimator().estimate(M, presence, K)
        return self._xi

    def _lane_band(self, frame_idx: int):
        from tac.boundary_math.analytic_lane_render_band import build_analytic_lane_band_prior
        roles = self._detect_roles()
        return build_analytic_lane_band_prior(
            self._gt["lstars"][frame_idx], lane_cls=int(roles.lane))

    def _ground_plane(self, frame_idx: int):
        from tac.boundary_math.hood_static_component import compute_static_hood_mask
        from tac.boundary_math.road_horizon_component import (
            fit_horizon_line,
            rasterize_sky_above_line,
        )
        roles = self._detect_roles()
        lstar = self._gt["lstars"][frame_idx]
        h, w = lstar.shape
        line = fit_horizon_line(lstar, sky_cls=int(roles.sky))
        sky = (rasterize_sky_above_line(line, h=h, w=w) if line is not None
               else np.zeros((h, w), bool))
        hood = compute_static_hood_mask(self._gt["lstars"], hood_cls=int(roles.hood)).mask
        us = np.arange(w, dtype=np.float64)
        hrow = (np.clip(line.horizon_row(us), 0, h) if line is not None
                else np.full(w, h * 0.4))
        return sky, hood, hrow, (line.rms_row if line is not None else float("nan"))

    # -- the 2x2 physical-prior atlas for one representative frame --
    def _atlas_figure(self, pi: int, dpi: int) -> dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        frame_full = self._gt["gt_f1"][pi]
        lstar = self._gt["lstars"][pi]
        h, w = lstar.shape
        frame = _resize_rgb_to(frame_full, (h, w))

        prior = self._lane_band(pi)
        cov = prior.coverage
        sky, hood, hrow, hrms = self._ground_plane(pi)
        margin = (self._gt["margins"][pi] if self._gt["margins"] is not None
                  else np.zeros((h, w), np.float32))
        xi = self._ego_xi()
        ds = float(xi.ds[pi])
        dy = float(xi.dy[pi])
        dpsi = float(xi.dpsi[pi])

        fig = Figure(figsize=(11.0, 6.4), facecolor=BG)

        def _ax(r, c, title, tcolor=FG):
            ax = fig.add_subplot(2, 2, r * 2 + c + 1)
            ax.set_facecolor(BG)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#2a2f39")
            ax.set_title(title, color=tcolor, fontsize=9.0, pad=4, loc="left")
            return ax

        # (0,0) openpilot lane band -> d_seg physical prior
        a00 = _ax(0, 0, f"openpilot lane band -> d_seg  ·  recall {prior.band_recall:.2f}"
                        f"  ·  {prior.n_lines} line(s)", LANE)
        a00.imshow(frame)
        band = np.zeros((h, w, 4), np.float32)
        band[..., 1] = 0.88                              # openpilot green
        band[..., 2] = 0.55
        band[..., 3] = np.clip(cov, 0, 1) * 0.72
        a00.imshow(band)

        # (0,1) ground-plane static structure (road / sky horizon / ego hood)
        a01 = _ax(0, 1, f"ground plane -> static classes  ·  horizon rms {hrms:.1f}px", ACC)
        a01.imshow(frame)
        ov = np.zeros((h, w, 4), np.float32)
        ov[sky] = (0.35, 0.55, 1.0, 0.30)               # sky / undrivable (blue)
        ov[hood] = (0.80, 0.0, 1.0, 0.34)               # ego hood (magenta)
        a01.imshow(ov)
        a01.plot(np.arange(w), hrow, color="#ffd24a", lw=1.4)   # road/sky horizon

        # (1,0) SegNet detectability field (the detector's sensitivity map)
        a10 = _ax(1, 0, "detectability field ρ_seg (bright = fragile separatrix)", POSE)
        vmax = float(np.percentile(margin, 96)) or 1.0
        frag = np.clip(1.0 - margin / (vmax + 1e-9), 0.0, 1.0)
        a10.imshow(frag, cmap="inferno", vmin=0, vmax=1)

        # (1,1) ego-xi screw -> d_pose physical prior (dual-use with d_seg)
        a11 = _ax(1, 1, f"ego-ξ screw -> d_pose  ·  ds {ds:+.2f}m  dψ {dpsi:+.3f}rad", POSE)
        a11.imshow(frame)
        # vanishing point ~ horizon center; arrow = forward (ds) + yaw (dpsi)
        vp_u = float(w) / 2.0 + dpsi * (w * 0.9)         # yaw shifts the heading laterally
        vp_v = float(np.clip(hrow[w // 2], 0, h))
        base_u, base_v = float(w) / 2.0, float(h) * 0.94
        a11.annotate("", xy=(vp_u, vp_v), xytext=(base_u, base_v),
                     arrowprops={"arrowstyle": "-|>", "color": "#ffd24a", "lw": 2.2,
                                 "shrinkA": 0, "shrinkB": 0})
        a11.scatter([vp_u], [vp_v], s=26, c="#ffd24a", zorder=5)

        fig.suptitle(
            f"ORACLE  ·  pair {pi}  ·  the world the detector reads  "
            f"·  [macOS-CPU advisory · NON-PROMOTABLE]",
            color=FG, fontsize=10.5, y=0.995)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01,
                            wspace=0.05, hspace=0.14)
        return {
            "frame_idx": pi,
            "lane_recall": round(float(prior.band_recall), 4)
            if not np.isnan(prior.band_recall) else None,
            "n_lines": int(prior.n_lines),
            "ds": round(ds, 4), "dy": round(dy, 4), "dpsi": round(dpsi, 5),
            "atlas": _png_data_uri(fig, dpi),
        }

    # -- the ego-xi screw-twist sequence chart across the whole cached segment --
    def _xi_chart(self, dpi: int) -> str:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        xi = self._ego_xi()
        t = np.arange(xi.n_pairs)
        fig = Figure(figsize=(11.0, 2.9), facecolor=BG)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color("#2a2f39")
        ax.tick_params(colors=MUT, labelsize=8)
        ax.plot(t, xi.ds, color=LANE, lw=1.6, marker="o", ms=3, label="ds  forward (m)")
        ax.plot(t, xi.dy, color=ACC, lw=1.4, marker="o", ms=3, label="dy  lateral (m)")
        ax2 = ax.twinx()
        ax2.set_facecolor(BG)
        for s in ax2.spines.values():
            s.set_color("#2a2f39")
        ax2.tick_params(colors=POSE, labelsize=8)
        ax2.plot(t, xi.dpsi, color=POSE, lw=1.4, marker="o", ms=3, label="dψ  yaw (rad)")
        ax.set_xlabel("pair (segment timeline)", color=MUT, fontsize=8.5)
        ax.set_ylabel("translation (m)", color=FG, fontsize=8.5)
        ax2.set_ylabel("yaw (rad)", color=POSE, fontsize=8.5)
        ax.set_title(f"ego-ξ SE(3) screw twist across the segment  ·  estimator "
                     f"{xi.estimator_id}  ·  the SAME ξ warps d_seg AND is d_pose (Chasles)",
                     color=FG, fontsize=9.5, loc="left", pad=6)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False,
                  fontsize=8, labelcolor=FG)
        fig.subplots_adjust(left=0.07, right=0.93, top=0.85, bottom=0.18)
        return _png_data_uri(fig, dpi)

    def render(self, frames: list[int], dpi: int = 80) -> dict[str, Any]:
        t0 = time.time()
        self._ensure_loaded()
        n = self._gt["n"]
        idx = sorted({int(i) for i in frames if 0 <= int(i) < n}) or [0]
        roles = self._detect_roles()
        xi = self._ego_xi()
        panels = [self._atlas_figure(pi, dpi) for pi in idx]
        xi_chart = self._xi_chart(dpi)
        mean_speed = float(np.mean(np.abs(xi.ds[1:]))) if xi.n_pairs > 1 else 0.0
        return {
            "ok": True,
            "gt_cache": os.path.basename(self.gt_cache),
            "n_frames": int(n),
            "frames_shown": idx,
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "render_secs": round(time.time() - t0, 2),
            "authority": "macOS-CPU advisory · NON-PROMOTABLE",
            "roles": {"road": int(roles.road), "lane": int(roles.lane),
                      "sky": int(roles.sky), "movable": int(roles.movable),
                      "hood": int(roles.hood)},
            "lane_band": {
                "band_recall_mean": (round(float(self._lane_stats.get("band_recall_mean")), 4)
                                     if self._lane_stats.get("band_recall_mean") is not None
                                     and not np.isnan(self._lane_stats.get("band_recall_mean", np.nan))
                                     else None),
                "total_lines": int(self._lane_stats.get("total_lines", 0)),
            },
            "xi": {
                "estimator_id": xi.estimator_id,
                "ds": [round(float(v), 4) for v in xi.ds],
                "dy": [round(float(v), 4) for v in xi.dy],
                "dpsi": [round(float(v), 5) for v in xi.dpsi],
                "mean_speed_ms": round(mean_speed, 4),
            },
            "classes": [{"i": i, "label": SEG_LABELS[i], "hex": SEG_PALETTE_HEX[i]}
                        for i in range(len(SEG_LABELS))],
            "panels": panels,
            "xi_chart": xi_chart,
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n6.npz")
    ap.add_argument("--frames", default="0,2,4", help="comma-separated frame indices")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--out", default=None, help="write the payload JSON here (durable path; NOT /tmp)")
    ap.add_argument("--done", default=None, help="write a small <done>.json marker on success")
    a = ap.parse_args(argv)

    frames = [int(x) for x in str(a.frames).split(",") if x.strip() != ""]
    r = OracleRenderer(a.gt_cache)
    payload = r.render(frames, a.dpi)
    sizes = [len(p["atlas"]) for p in payload["panels"]]
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("panels", "xi_chart")}, indent=2))
    print(f"atlas data-uri sizes (bytes): {sizes}  total={sum(sizes)}  "
          f"xi_chart={len(payload['xi_chart'])}  frames={payload['frames_shown']}", flush=True)
    if a.out:
        outp = Path(a.out)
        if str(outp.resolve()).startswith(("/tmp", "/private/tmp", "/var/tmp")):
            raise ValueError("--out must be a durable path, not system /tmp (CLAUDE.md)")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload))
        print(f"wrote {outp} ({outp.stat().st_size} bytes)", flush=True)
        if a.done:
            Path(a.done).write_text(json.dumps({
                "ok": True, "built_at_utc": payload["built_at_utc"],
                "frames_shown": payload["frames_shown"],
                "render_secs": payload["render_secs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
