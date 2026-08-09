# SPDX-License-Identifier: MIT
"""#906 CHROMA SITING — $0 local sensitivity probe (the Modal-dispatch pricer).

WHY (2026-08-09, operator: "our using pyav was likely a source of confounding
perhaps we should be optimizing against upstream").  ``upstream/evaluate.py``
picks the dataset BY DEVICE: CUDA -> ``DaliVideoDataset`` (nvdec), else
``AVVideoDataset`` (PyAV + ``frame_utils.yuv420_to_rgb``).  The contest
leaderboard runs CUDA, so the AUTHORITY decodes through nvdec while every local
GT cache we own decodes through PyAV.  ``yuv420_to_rgb``'s docstring claims it
matches nvdec, but that is an INTENT, not proven bit-identity.

The named mechanism: ``frame_utils.py:175`` upsamples chroma with
``F.interpolate(..., mode='bilinear', align_corners=False)``.  For a x2 upscale
that maps output index ``i`` to source coordinate ``i/2 - 0.25`` in BOTH axes =
CENTERED (JPEG / MPEG-1) chroma siting.  H.264/HEVC 4:2:0 -- what nvdec decodes
-- is horizontally CO-SITED (left / MPEG-2): chroma sample k aligns with luma
column 2k, vertical stays centered.  So the candidate discrepancy is a
half-luma-pixel HORIZONTAL shift of the U and V planes.

This probe does NOT need CUDA.  It renders each GT frame_1 under BOTH siting
conventions and runs the SAME frozen CPU-torch SegNet on both, reporting the
argmax disagreement fraction in units directly comparable to a d_seg.

Why it must run BEFORE any Modal dispatch (CLAUDE.md "Carmack MVP-first
phasing"): it converts an unpriced confound into a priced one at $0.  Reading
of the result:
  * disagreement << 2.86e-4  -> the siting mechanism cannot explain a seg term
    at PR130 scale; the DALI/AV confound is bounded small and the CUDA dispatch
    drops priority.
  * disagreement ~>= 2.86e-4 -> the confound is potentially the size of the
    entire remaining seg term; the one qualifying Modal job (build the official
    GT cache at --dataset dali AND --dataset av on one host and diff) becomes
    urgent, and every seg measurement taken against AV-GT inherits the error.

HONESTY: this measures the SENSITIVITY of the frozen SegNet argmax to the
siting convention.  It does NOT prove which convention nvdec actually emits --
that requires CUDA+DALI and is exactly what the Modal job buys.  A large
sensitivity makes the dispatch decisive; a small one makes it cheap to skip.

Sampling: stratified-random over the pair index, seeded, n>=120 by default --
never a prefix (m88/m96: a prefix of a temporally-correlated population is a
different population).

Axis label: [macOS-CPU advisory].  Not a score.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the repo/SSD tier per CLAUDE.md.")


def _planes(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (y, u, v) uint8 planes exactly as upstream ``yuv420_to_rgb`` does."""
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, : W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, : W // 2]
    return y.copy(), u.copy(), v.copy()


def _yuv_to_rgb_uint8(y_t, u_up, v_up):
    """BT.601 limited-range conversion -- byte-identical to upstream's tail."""
    import torch  # noqa: F401  (imported by caller; kept local for clarity)

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)
    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    import torch as _torch

    return _torch.stack([r, g, b], dim=-1).round().to(_torch.uint8)


def _upsample_centered(plane_t, H: int, W: int):
    """Upstream's convention: bilinear align_corners=False in BOTH axes.

    For x2 this maps output ``i`` -> source ``i/2 - 0.25`` = CENTERED siting
    (JPEG / MPEG-1).  Reproduces ``frame_utils.yuv420_to_rgb`` exactly.
    """
    import torch.nn.functional as F

    return F.interpolate(plane_t, size=(H, W), mode="bilinear", align_corners=False).squeeze()


def _upsample_left_sited(plane_t, H: int, W: int):
    """nvdec / H.264 / HEVC convention: horizontally CO-SITED, vertically centered.

    Vertical leg is identical to upstream (centered).  Horizontal leg maps
    output column ``2k`` -> chroma sample ``k`` exactly, and ``2k+1`` -> the
    midpoint ``(u[k] + u[k+1]) / 2`` with the right edge clamped.  That is a
    +0.25 chroma-sample (= +0.5 luma-pixel) shift relative to the centered form.
    """
    import torch
    import torch.nn.functional as F

    # (1,1,H/2,W/2) -> vertical-only centered upsample -> (1,1,H,W/2)
    half_w = plane_t.shape[-1]
    vert = F.interpolate(plane_t, size=(H, half_w), mode="bilinear", align_corners=False)
    vert = vert.squeeze(0).squeeze(0)  # (H, W/2)

    out = torch.empty((H, W), dtype=vert.dtype)
    out[:, 0::2] = vert                                   # even cols: co-sited, exact
    out[:, 1:-1:2] = 0.5 * (vert[:, :-1] + vert[:, 1:])   # odd cols: midpoint
    out[:, -1] = vert[:, -1]                              # right edge: clamp (replicate)
    return out


def _rgb_variants(frame):
    """Return (centered_rgb_uint8_HWC, left_sited_rgb_uint8_HWC) as numpy arrays."""
    import torch

    H, W = frame.height, frame.width
    y, u, v = _planes(frame)
    y_t = torch.from_numpy(y).float()
    u_t = torch.from_numpy(u).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0)

    rgb_centered = _yuv_to_rgb_uint8(
        y_t, _upsample_centered(u_t, H, W), _upsample_centered(v_t, H, W)
    )
    rgb_left = _yuv_to_rgb_uint8(
        y_t, _upsample_left_sited(u_t, H, W), _upsample_left_sited(v_t, H, W)
    )
    return rgb_centered.numpy(), rgb_left.numpy()


def _stratified_pair_indices(n_total: int, n_sample: int, seed: int) -> set[int]:
    """One uniformly-random pick per equal-width stratum -- NEVER a prefix."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n_total, n_sample + 1).astype(int)
    picks: set[int] = set()
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        hi = max(hi, lo + 1)
        picks.add(int(rng.integers(lo, hi)))
    return picks


def run(n_sample: int, seed: int, n_pairs_total: int, video_path: Path) -> dict:
    import av
    import torch
    from frame_utils import seq_len  # upstream

    from tac.boundary_math.seg_core import load_real_segnet, segnet_argmax_and_margin

    wanted = _stratified_pair_indices(n_pairs_total, n_sample, seed)
    seg_cpu = load_real_segnet("cpu")

    # POSITIVE CONTROL (fail-closed): our CENTERED reimplementation must be
    # BYTE-IDENTICAL to upstream's yuv420_to_rgb.  Without this the comparison
    # would measure OUR reimplementation error, not the siting convention.
    control: dict = {"checked": False}

    rows: list[dict] = []
    t0 = time.time()
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    buf: list[object] = []
    pair_idx = 0
    try:
        for frame in container.decode(stream):
            buf.append(frame)
            if len(buf) < seq_len:
                continue
            frame1 = buf[-1]
            buf = []
            if pair_idx in wanted:
                rgb_c, rgb_l = _rgb_variants(frame1)
                if not control["checked"]:
                    from frame_utils import yuv420_to_rgb  # upstream, READ-ONLY

                    ref = np.asarray(yuv420_to_rgb(frame1))
                    if not np.array_equal(ref, rgb_c):
                        raise AssertionError(
                            "centered-path positive control FAILED: our reimplementation is not "
                            f"byte-identical to upstream yuv420_to_rgb (max|delta|="
                            f"{int(np.abs(ref.astype(np.int32) - rgb_c.astype(np.int32)).max())}). "
                            "The siting comparison would be confounded by reimplementation error."
                        )
                    control = {
                        "checked": True,
                        "centered_path_byte_identical_to_upstream": True,
                        "control_pair_idx": pair_idx,
                    }
                rgb_delta = np.abs(rgb_c.astype(np.int32) - rgb_l.astype(np.int32))
                with torch.inference_mode():
                    lstar_c, _ = segnet_argmax_and_margin(seg_cpu, rgb_c.astype(np.float64))
                    lstar_l, _ = segnet_argmax_and_margin(seg_cpu, rgb_l.astype(np.float64))
                lc = np.asarray(lstar_c)
                ll = np.asarray(lstar_l)
                disagree = int((lc != ll).sum())
                rows.append(
                    {
                        "pair_idx": pair_idx,
                        "argmax_pixels": int(lc.size),
                        "argmax_disagree_pixels": disagree,
                        "argmax_disagree_frac": disagree / float(lc.size),
                        "rgb_max_abs_delta": int(rgb_delta.max()),
                        "rgb_mean_abs_delta": float(rgb_delta.mean()),
                        "rgb_changed_frac": float((rgb_delta > 0).mean()),
                    }
                )
            pair_idx += 1
            if pair_idx >= n_pairs_total:
                break
    finally:
        container.close()

    fracs = np.array([r["argmax_disagree_frac"] for r in rows], dtype=np.float64)
    total_px = int(sum(r["argmax_pixels"] for r in rows))
    total_dis = int(sum(r["argmax_disagree_pixels"] for r in rows))
    return {
        "probe": "chroma_siting_argmax_sensitivity",
        "axis_label": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "video": str(video_path),
        "seed": seed,
        "n_sampled_pairs": len(rows),
        "n_pairs_total": n_pairs_total,
        "sampling": "stratified_random_one_per_stratum",
        "positive_control": control,
        "elapsed_seconds": round(time.time() - t0, 1),
        # THE headline: pooled disagreement, directly comparable to a d_seg.
        "pooled_argmax_disagree_frac": (total_dis / total_px) if total_px else None,
        "mean_argmax_disagree_frac": float(fracs.mean()) if fracs.size else None,
        "max_argmax_disagree_frac": float(fracs.max()) if fracs.size else None,
        "min_argmax_disagree_frac": float(fracs.min()) if fracs.size else None,
        "pairs_with_any_disagreement": int((fracs > 0).sum()),
        "reference_pr130_d_seg": 2.8609e-4,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--num-sample-pairs", type=int, default=120, help="stratified sample size (>=120 per m88/m96)")
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--n-pairs-total", type=int, default=600)
    ap.add_argument("--video", type=Path, default=REPO / "upstream" / "videos" / "0.mkv")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO / ".omx" / "research" / "ddm_rm1_20260808" / "chroma_siting_sensitivity.json",
    )
    args = ap.parse_args(argv)
    _refuse_tmp(args.out_json)
    info = run(args.num_sample_pairs, args.seed, args.n_pairs_total, args.video)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(info, indent=2) + "\n")
    headline = {k: v for k, v in info.items() if k != "rows"}
    print(json.dumps(headline, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
