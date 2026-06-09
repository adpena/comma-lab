#!/usr/bin/env python3
"""Scorer spectral-sensitivity analyzer — the empirical scorer TRANSFER FUNCTION.

THE ARBITRARINESS CURE (operator 2026-06-09): the carrier hand-sets a single global
SIREN frequency w=30. That is one symptom of the arbitrariness class — a magic
constant by convention, not derived/measured/learned. This tool MEASURES which
spatial frequencies the FROZEN contest scorers (SegNet argmax on frame1, PoseNet on
the two-frame YUV6 motion) actually react to, so the carrier's frequency budget can
be DERIVED from the scorer instead of guessed.

METHOD (linear-response transfer function, $0 frozen-scorer measurement):
  1. decode source -> camera-res uint8 .raw (GT).
  2. for each radial frequency band k in [0,1] (DC -> Nyquist), B bands:
       build a band-limited perturbation delta_k (2D-FFT radial mask of white noise),
       normalized to a FIXED relative energy eps of each frame's std (equal energy
       per band -> H(k) is pure spectral sensitivity, not band area/energy);
       perturb ONLY frame1 of every pair (frame1 drives SegNet AND shifts the
       inter-frame motion PoseNet reads), write perturbed .raw;
       score perturbed-vs-source through the EXACT DistortionNet (ladder.score_pairs).
  3. H_seg(k) = mean d_seg(k); H_pose(k) = mean d_pose(k). The band(s) with the
     largest H are where the scorer is most sensitive -> where the carrier should
     place its frequency content (the DERIVED initializer for a learnable omega /
     Fourier-feature bandwidth / water-fill levels).

Authority: [macOS-CPU advisory] / exact_pair_scorer (uses the EXACT DistortionNet) ->
mechanism_update_eligible ONLY. It measures the scorer's sensitivity (a mechanism
fact that directs the frequency-basis design); it is NOT a candidate score. NOT
promotable; does NOT update the score roadmap. Cross-ref
`.omx/research/principled_frequency_basis_synthesis_20260609.md`.

Run (fast; a few min for N~24 x B~8 on the frozen scorer):
  .venv/bin/python tools/measure_scorer_spectral_sensitivity.py \\
    --n-pairs 24 --n-bands 8 --rel-energy 0.05 \\
    --work-dir /Volumes/VertigoDataTier/pact/scorer_spectral_sensitivity_<utc> \\
    --out <work>/scorer_spectral_sensitivity.v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _utc() -> str:
    import subprocess

    return subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
    ).stdout.strip()


def _band_limited_perturbation(
    shape: tuple[int, int, int], r_lo: float, r_hi: float, rng: Any
) -> Any:
    """Real band-limited field on the radial-frequency annulus [r_lo, r_hi] (r in
    [0,1], 0=DC, 1=Nyquist corner), unit std. One independent draw per channel."""
    import numpy as np

    h, w, c = shape
    fy = np.fft.fftfreq(h)[:, None]  # [-0.5, 0.5)
    fx = np.fft.fftfreq(w)[None, :]
    # normalize radius so 1.0 == the Nyquist corner (max |f| = 0.5 on each axis).
    radius = np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2) / np.sqrt(2.0)
    band = ((radius >= r_lo) & (radius < r_hi)).astype(np.float64)
    if band.sum() == 0:
        return np.zeros(shape, dtype=np.float64)
    out = np.empty(shape, dtype=np.float64)
    for ch in range(c):
        noise = rng.standard_normal((h, w))
        spec = np.fft.fft2(noise) * band
        field = np.real(np.fft.ifft2(spec))
        s = field.std()
        out[..., ch] = field / s if s > 0 else field
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=24, help="pairs to average per band")
    ap.add_argument("--n-bands", type=int, default=8, help="radial frequency bands DC->Nyquist")
    ap.add_argument(
        "--rel-energy",
        type=float,
        default=0.05,
        help="perturbation std as a fraction of each frame's std (equal per band)",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    import hi_nerv_renderer_sanity_ladder as ladder
    import numpy as np

    work = args.work_dir.resolve()
    if "/tmp" in str(work):
        raise SystemExit("work-dir must not be under /tmp (durable SSD only)")
    work.mkdir(parents=True, exist_ok=True)
    H, W, C = ladder._frame_dims()
    N = int(args.n_pairs)
    B = int(args.n_bands)
    eps = float(args.rel_energy)

    # 1. GT camera-res .raw (first 2N frames).
    src_raw = work / "source.raw"
    ladder.decode_source_to_raw(_REPO / "upstream" / "videos" / "0.mkv", src_raw, max_frames=2 * N)
    frames = np.memmap(src_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    navail = frames.shape[0] // 2
    N = min(N, navail)
    indices = list(range(N))

    # sanity: source-vs-source must be ~0 distortion (the measurement baseline).
    base = ladder.score_pairs(src_raw, src_raw, indices, device=args.device)
    base_rows = base["pairs"]
    base_seg = float(np.mean([r["d_seg"] for r in base_rows]))
    base_pose = float(np.mean([r["d_pose"] for r in base_rows]))

    rng = np.random.default_rng(args.seed)
    bands: list[dict[str, Any]] = []
    pert_raw = work / "perturbed.raw"
    for k in range(B):
        r_lo, r_hi = k / B, (k + 1) / B
        # Build perturbed frames: copy, perturb ONLY frame1 (odd index) of each pair.
        pert = np.array(frames[: 2 * N], dtype=np.float64)
        for i in range(N):
            f1 = pert[2 * i + 1]
            field = _band_limited_perturbation((H, W, C), r_lo, r_hi, rng)
            delta = eps * float(f1.std()) * field  # equal relative energy per band
            pert[2 * i + 1] = np.clip(f1 + delta, 0.0, 255.0)
        pert.astype(np.uint8).tofile(pert_raw)
        res = ladder.score_pairs(pert_raw, src_raw, indices, device=args.device)
        rows = res["pairs"]
        d_seg = float(np.mean([r["d_seg"] for r in rows]))
        d_pose = float(np.mean([r["d_pose"] for r in rows]))
        bands.append(
            {
                "band_index": k,
                "r_lo": r_lo,
                "r_hi": r_hi,
                "H_seg": d_seg - base_seg,  # transfer function (response above baseline)
                "H_pose": d_pose - base_pose,
                "d_seg_raw": d_seg,
                "d_pose_raw": d_pose,
            }
        )
        print(
            f"[spectral] band {k} r[{r_lo:.3f},{r_hi:.3f}] "
            f"H_seg={d_seg - base_seg:+.5f} H_pose={d_pose - base_pose:+.4f}"
        )

    seg_peak = max(bands, key=lambda b: b["H_seg"])
    pose_peak = max(bands, key=lambda b: b["H_pose"])
    artifact: dict[str, Any] = {
        "schema": "scorer_spectral_sensitivity.v1",
        "utc": _utc(),
        "authority_tier": "exact_cpu_advisory",
        "metric_family": "exact_pair_scorer",
        "score_roadmap_update_eligible": False,
        "mechanism_update_eligible": True,
        "promotable": False,
        "n_pairs": N,
        "n_bands": B,
        "rel_energy": eps,
        "baseline_d_seg": base_seg,
        "baseline_d_pose": base_pose,
        "bands": bands,
        "seg_peak_band": seg_peak,
        "pose_peak_band": pose_peak,
        "derived_frequency_target": {
            "note": (
                "H_seg / H_pose are the scorer's per-band spectral sensitivity (Δdistortion "
                "per equal-energy perturbation). The band(s) with peak H are where the carrier "
                "should place frequency content — the DERIVED (not hand-tuned) target for a "
                "learnable omega / Fourier-feature bandwidth / per-band water-fill level. "
                "r is normalized radial frequency (0=DC, 1=Nyquist corner)."
            ),
            "seg_peak_band_index": seg_peak["band_index"],
            "seg_peak_r_center": 0.5 * (seg_peak["r_lo"] + seg_peak["r_hi"]),
            "pose_peak_band_index": pose_peak["band_index"],
            "pose_peak_r_center": 0.5 * (pose_peak["r_lo"] + pose_peak["r_hi"]),
        },
        "arbitrariness_note": (
            "This measurement converts the arbitrary SIREN w from a convention into a "
            "scorer-derived quantity (non-arbitrariness principle). Sister design: "
            ".omx/research/principled_frequency_basis_synthesis_20260609.md"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print("=== scorer spectral-sensitivity transfer function ===")
    print(f"  baseline d_seg={base_seg:.5f} d_pose={base_pose:.4f} (source-vs-source)")
    print(f"  SEG  peak band {seg_peak['band_index']} r~{0.5*(seg_peak['r_lo']+seg_peak['r_hi']):.3f}  H_seg={seg_peak['H_seg']:+.5f}")
    print(f"  POSE peak band {pose_peak['band_index']} r~{0.5*(pose_peak['r_lo']+pose_peak['r_hi']):.3f}  H_pose={pose_peak['H_pose']:+.4f}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
