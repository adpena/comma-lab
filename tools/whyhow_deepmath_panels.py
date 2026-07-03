#!/usr/bin/env python3
"""WHY/HOW deep-math museum — single-frame FIELD BUNDLE renderer (PASS 1).

Feeds the WHY/HOW tab's two Pass-1 plates (design
``.omx/research/whyhow_deepmath_museum_design_20260703.md`` §I.1 + §I.4) with the
RAW, co-registered scalar fields a client-side WebGPU renderer needs — NOT baked
matplotlib PNGs. ONE representative scored frame, three readings of the SAME
geometry:

  * rho_seg      SegNet top1-top2 argmax MARGIN, read as FRAGILITY (bright = small
                 margin = the codim-1 separatrix where the argmax can flip = where
                 d_seg lives). Yousfi's detector's own sensitivity. From the GT
                 cache ``margins`` (the EXACT margins the scorer verdict uses).
  * rho_uniward  REAL S-UNIWARD embedding cost (Holub-Fridrich-Denemark 2014) via
                 the CANONICAL tac.uniward_delta.compute_uniward_cost_map (NOT a JS
                 proxy). High = textured = undetectable-to-embed (Fridrich #1).
  * rho_sens     "our distortion sensitivity" = the separatrix-proximity field
                 (decayed distance from the argmax label-change boundary): where a
                 pixel flip changes the DECISION. Derived from the real partition;
                 grounded in the measured "~100% of d_seg lives on the 1-px annulus".

The THESIS (design §I.4, the tribute's heart): steganography (UNIWARD), steganalysis
(SegNet margin), and our loss are ONE geometry -- ``1 / ||d(detector)/d(pixel)||``,
the Fisher metric read three ways. Canonical measured anchor: Fisher curvature
<-> (-margin) Pearson **0.978** (memory ``unified-variational-levelset-flow``). We
ALSO report the live per-frame Pearson between the served fields, honestly labelled.

DE-ORPHAN, DON'T REBUILD (CLAUDE.md "velocity-driven orphaning"): every field is a
REAL in-tree computation wired to a viz for the first time -- the GT cache the ORACLE
tab already proves co-exists with the live #205 run (gt_n6.npz, 6 pairs, ~48 MB), the
canonical S-UNIWARD the RESIDUAL tab already renders, the comma10k palette from
render_comma_baseline_vs_ours_viz. NO witness checkpoint, NO SegNet forward (the GT
argmax + margins are already cached) -- torch is imported ONLY for the S-UNIWARD
wavelet cost on the one frame.

GOVERNED + CONTROL-PLANE-SAFE BY CONSTRUCTION: this runs as a DETACHED safe_run
subprocess in its OWN process group (own RSS/timeout cap), so its torch footprint is
NEVER summed into the lean dashboard's safe_run group -- it can never crash the
dashboard or the machine, and it yields to #205 under the memory floor (same pattern
as ORACLE + FLOW).

AUTHORITY: everything here is ``[macOS-CPU advisory]`` NON-PROMOTABLE imagery. A viz
moves NO pointer (0.19110, UNMOVED). The exact row is byte-closed on contest-CPU/CUDA.
MEANS, not an end.

CLI (independently runnable / testable -- the governed detached path runs THIS):
    .venv/bin/python tools/whyhow_deepmath_panels.py \
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n6.npz \
        --out .omx/tmp/dash_whyhow/whyhow_probe.json
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

# comma10k palette (the colors Yousfi + comma read instantly) -- reused from the canonical viz.
from tools.render_comma_baseline_vs_ours_viz import SEG_LABELS, SEG_PALETTE_HEX  # noqa: E402

# Canonical MEASURED anchor: Fisher curvature <-> (-margin) Pearson 0.978 (memory
# ``unified-variational-levelset-flow``; the margin field IS the Fisher surrogate).
CANONICAL_FISHER_MARGIN_PEARSON = 0.978


def _l_png_b64(field_u8: np.ndarray) -> str:
    """A single-channel field as an OPAQUE grayscale (L) PNG (exact 0..255 bytes, no colormap)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(field_u8, np.uint8), "L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _rgb_png_b64(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> str:
    """Pack three scalar fields into ONE OPAQUE RGB PNG (exact bytes; PNG never premultiplies
    and corrupts the channels). R/G/B carried separately, unpacked client-side."""
    from PIL import Image
    rgb = np.ascontiguousarray(np.stack([r, g, b], axis=-1), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _jpeg_b64(rgb_u8: np.ndarray, quality: int = 78) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb_u8, np.uint8), "RGB").save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resize_rgb_to(frame: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor RGB resize (H,W,3)->(h,w,3) -- no scipy/PIL dep; display-only."""
    h, w = hw
    H, W = frame.shape[:2]
    ri = (np.linspace(0, H - 1, h)).astype(np.int64)
    ci = (np.linspace(0, W - 1, w)).astype(np.int64)
    return frame[ri][:, ci]


def _frag_field(margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SegNet margin -> FRAGILITY in [0,1] (bright = small margin = fragile separatrix). Matches
    the ORACLE/RESIDUAL convention (1 - margin / p96). Returns (float in [0,1], uint8 0..255)."""
    m = np.asarray(margin, np.float32)
    vmax = float(np.percentile(m, 96)) or 1.0
    frag = np.clip(1.0 - m / (vmax + 1e-9), 0.0, 1.0)
    return frag, (frag * 255.0 + 0.5).astype(np.uint8)


def _uniward_field(frame_rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """REAL S-UNIWARD embedding cost on the frame via the CANONICAL torch module (NOT a proxy).
    log1p + percentile stretch (matching the RESIDUAL panel) so textured boundary structure pops.
    Returns (float in [0,1], uint8 0..255). torch imported HERE only (isolated subprocess)."""
    import torch

    from tac.uniward_delta import compute_uniward_cost_map
    fr = torch.from_numpy(frame_rgb_u8.astype(np.float32)).permute(2, 0, 1)[None]  # (1,3,H,W)
    with torch.inference_mode():
        uni = compute_uniward_cost_map(fr).squeeze(0).numpy()
    uni_log = np.log1p(np.asarray(uni, np.float32))
    lo = float(np.percentile(uni_log, 40))
    hi = float(np.percentile(uni_log, 99.5))
    span = (hi - lo) or 1.0
    norm = np.clip((uni_log - lo) / span, 0.0, 1.0)
    return norm, (norm * 255.0 + 0.5).astype(np.uint8)


def _boundary_sensitivity(lstar: np.ndarray, scale: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """our distortion sensitivity ~ separatrix proximity: a decayed distance from the argmax
    label-change boundary (where a pixel flip changes the DECISION). Derived from the real
    partition; grounded in the measured "~100% of d_seg on the 1-px annulus". Returns
    (float in [0,1], uint8 0..255)."""
    lab = np.asarray(lstar, np.int64)
    b = np.zeros(lab.shape, bool)
    diff_h = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= diff_h
    b[:, 1:] |= diff_h
    diff_v = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= diff_v
    b[1:, :] |= diff_v
    try:
        from scipy.ndimage import distance_transform_edt
        d = distance_transform_edt(~b)
        s = np.exp(-np.asarray(d, np.float32) / float(scale))
    except Exception:  # pure-numpy fallback: iterative decayed dilation from the boundary
        s = b.astype(np.float32)
        cur = b.copy()
        w = 1.0
        for _ in range(max(2, int(scale * 2))):
            w *= 0.78
            nxt = cur.copy()
            nxt[:, :-1] |= cur[:, 1:]
            nxt[:, 1:] |= cur[:, :-1]
            nxt[:-1, :] |= cur[1:, :]
            nxt[1:, :] |= cur[:-1, :]
            newly = nxt & ~cur
            s[newly] = np.maximum(s[newly], w)
            cur = nxt
    s = np.clip(s, 0.0, 1.0).astype(np.float32)
    return s, (s * 255.0 + 0.5).astype(np.uint8)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return None
    x = x[m] - x[m].mean()
    y = y[m] - y[m].mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 0.0:
        return None
    return float((x * y).sum() / denom)


class WhyHowRenderer:
    """Renders the WHY/HOW single-frame field bundle from the shared GT cache. Loads the cache
    ONCE; picks the frame with the RICHEST separatrix (longest argmax boundary) so the hero
    exhibit has the most codim-1 structure. NO witness checkpoint, NO SegNet forward."""

    def __init__(self, gt_cache: str | os.PathLike):
        self.gt_cache = str(gt_cache)
        self._gt: dict | None = None

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
        if gt["margins"] is None:
            raise ValueError(f"GT cache {self.gt_cache} lacks 'margins' (required for rho_seg)")
        self._gt = gt

    def _pick_frame(self) -> int:
        lst = self._gt["lstars"]
        best_i, best_edges = 0, -1
        for i in range(lst.shape[0]):
            lab = lst[i]
            e = int((lab[:, :-1] != lab[:, 1:]).sum() + (lab[:-1, :] != lab[1:, :]).sum())
            if e > best_edges:
                best_edges, best_i = e, i
        return best_i

    def render(self, frame: int | None = None, jpeg_quality: int = 80) -> dict[str, Any]:
        t0 = time.time()
        self._ensure_loaded()
        n = self._gt["n"]
        pi = self._pick_frame() if frame is None else max(0, min(n - 1, int(frame)))

        lstar = self._gt["lstars"][pi]
        margin = self._gt["margins"][pi]
        h, w = lstar.shape
        frame_rgb = _resize_rgb_to(self._gt["gt_f1"][pi], (h, w))

        frag_f, frag_u8 = _frag_field(margin)
        uni_f, uni_u8 = _uniward_field(frame_rgb)
        sens_f, sens_u8 = _boundary_sensitivity(lstar)

        argmax_u8 = np.clip(np.asarray(lstar, np.int64), 0, 255).astype(np.uint8)

        return {
            "ok": True,
            "gt_cache": os.path.basename(self.gt_cache),
            "frame_idx": int(pi),
            "n_frames": int(n),
            "w": int(w), "h": int(h),
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "render_secs": round(time.time() - t0, 2),
            "authority": "macOS-CPU advisory · NON-PROMOTABLE",
            "classes": [{"i": i, "label": SEG_LABELS[i], "hex": SEG_PALETTE_HEX[i]}
                        for i in range(len(SEG_LABELS))],
            # RAW fields (client decodes to Uint8Array; exact bytes, no colormap):
            "render_b64": _jpeg_b64(frame_rgb, jpeg_quality),          # RGB scene backdrop
            "tri_b64": _rgb_png_b64(frag_u8, uni_u8, sens_u8),         # R=rho_seg G=rho_uniward B=rho_sens
            "argmax_b64": _l_png_b64(argmax_u8),                       # SegNet argmax label field 0..4
            # measured correlations (the Unity):
            "pearson_fisher_margin": CANONICAL_FISHER_MARGIN_PEARSON,  # canonical measured anchor
            "pearson_seg_uniward_frame": _pearson(frag_f, uni_f),      # this frame, live
            "pearson_seg_sens_frame": _pearson(frag_f, sens_f),
            "pearson_uni_sens_frame": _pearson(uni_f, sens_f),
            "fields": {
                "seg": {"label": "ρ_seg — SegNet margin (Yousfi)",
                        "note": "small margin = fragile separatrix = where d_seg lives"},
                "uni": {"label": "ρ_uniward — S-UNIWARD cost (Fridrich)",
                        "note": "real Holub–Fridrich–Denemark 2014; high = textured = undetectable"},
                "sens": {"label": "our distortion sensitivity",
                         "note": "separatrix proximity — where a pixel flip changes the decision"},
            },
            "uniward_source": "tac.uniward_delta.compute_uniward_cost_map (real S-UNIWARD)",
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n6.npz")
    ap.add_argument("--frame", default=None, help="frame index (default: richest-separatrix auto-pick)")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--out", default=None, help="write the payload JSON here (durable path; NOT /tmp)")
    ap.add_argument("--done", default=None, help="write a small <done>.json marker on success")
    a = ap.parse_args(argv)

    frame = None if a.frame is None else int(a.frame)
    r = WhyHowRenderer(a.gt_cache)
    payload = r.render(frame, a.jpeg_quality)
    sizes = {k: len(payload[k]) for k in ("render_b64", "tri_b64", "argmax_b64")}
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("render_b64", "tri_b64", "argmax_b64")}, indent=2))
    print(f"field b64 sizes (bytes): {sizes}  total={sum(sizes.values())}  "
          f"frame={payload['frame_idx']}  {payload['h']}x{payload['w']}", flush=True)
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
                "frame_idx": payload["frame_idx"], "render_secs": payload["render_secs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
