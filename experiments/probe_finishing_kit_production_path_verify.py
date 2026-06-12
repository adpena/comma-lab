# SPDX-License-Identifier: MIT
"""Finishing-kit PRODUCTION-PATH verification (the ≤1-LSB post-round gap, LENS-2b).

The convergence re-validation (``probe_finishing_kit_convergence_revalidation.py``)
fits + measures via the CAMERA-FLOAT path (pre-round, ``_measure_bias_affine``) — the
SAME path the original n=24 mid-basin probe used. But PRODUCTION inflate applies the
kit on POST-ROUND uint8 frames (``apply_distortion_kit_to_raw_frames``, via
``driver.kit_aware_exact_eval``). LENS-2b measured the two paths differ by ≤1 LSB
(mean 0.21) — a double-round artifact the driver docstring itself flags ("the ≤1 ULP
gap matters for a ±1-bias fit").

This probe re-measures the fitted-constant gain through ``kit_aware_exact_eval`` (the
production-faithful path) on the CONVERGED ep2120 decoder, so the banked/not-banked
verdict is on the number the FINISHED contest packet actually produces — not a
pre-round approximation. It reads the daemon's fitted constants from the
re-validation RESULT JSON (slice-1 T10 affine = the full kit).

Authority: ``[contest-CPU advisory] NON-PROMOTABLE``. Real frozen scorer + real
``yuv420_to_rgb`` GT; no MPS; no byte-closed ``evaluate.py`` row; frontier UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ADVISORY = "[contest-CPU advisory] NON-PROMOTABLE"
_CONVERGED_ARCHIVE = Path(
    "experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best/best_archive.bin"
)
_RESULT_JSON = Path(".omx/research/finishing_kit_convergence_revalidation_RESULT.json")
_TOTAL_VIDEO_BYTES = 37_545_489  # the contest rate-term denominator (unused for the distortion delta)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pairs", type=int, default=24, help="pairs for the production-path eval")
    p.add_argument("--result-json", type=Path, default=_RESULT_JSON,
                   help="the re-validation RESULT JSON to read fitted constants from")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.distortion_finishing_kit import DistortionKitConfig
    from tac.torch_vehicle.driver import import_vendored_bundle, kit_aware_exact_eval
    from tac.torch_vehicle.vendored_imports import import_vendored

    t0 = time.time()
    reval = json.loads(args.result_json.read_text())
    s1 = reval["converged_slice1_refit"]
    pr98_bias = np.asarray(s1["pr98_refit"]["best_bias_frame_channel"], dtype=np.float64)
    aff_scale = np.asarray(s1["t10_affine"]["best_scale_frame_channel"], dtype=np.float64)
    aff_bias = np.asarray(s1["t10_affine"]["best_bias_frame_channel"], dtype=np.float64)

    # Build the two production kits (PR98-only and full PR98+T10).
    kit_pr98 = DistortionKitConfig.from_pr98_bias(pr98_bias)
    kit_full = DistortionKitConfig.from_affine(aff_scale, aff_bias)
    kit_off = DistortionKitConfig(enabled=False)

    # Load the converged decoder via the SAME vendored path the driver uses.
    v = import_vendored_bundle()
    arch = _CONVERGED_ARCHIVE.read_bytes()
    dec_sd, latents, meta = v.parse_archive(arch)
    dec = v.HNeRVDecoder(
        latent_dim=meta.get("latent_dim", 28),
        base_channels=meta.get("base_channels", 20),
        eval_size=tuple(meta.get("eval_size", (384, 512))),
    )
    dec.load_state_dict({k: vv for k, vv in dec_sd.items()})
    dec.eval()
    lat = latents[: args.n_pairs]

    net = load_frozen_distortion_net(device="cpu")
    data = import_vendored("data")
    video_path = data.get_default_video_path()
    archive_bytes = len(arch)
    print(f"[prodpath] loaded in {time.time()-t0:.1f}s; n={args.n_pairs} via kit_aware_exact_eval", flush=True)

    def _eval(kit: Any) -> dict[str, float]:
        return kit_aware_exact_eval(
            dec, lat, net, video_path, distortion_kit=kit,
            archive_bytes=archive_bytes, total_video_bytes=_TOTAL_VIDEO_BYTES, device="cpu",
        )

    off = _eval(kit_off)
    print(f"[prodpath] OFF (production no-op): d_seg={off['seg_distortion']:.6f} d_pose={off['pose_distortion']:.6f}", flush=True)
    pr98 = _eval(kit_pr98)
    print(f"[prodpath] PR98: d_seg={pr98['seg_distortion']:.6f} d_pose={pr98['pose_distortion']:.6f}", flush=True)
    full = _eval(kit_full)
    print(f"[prodpath] FULL: d_seg={full['seg_distortion']:.6f} d_pose={full['pose_distortion']:.6f}", flush=True)

    def _dscore(d: dict[str, float]) -> float:
        return 100.0 * d["seg_distortion"] + (10.0 * d["pose_distortion"]) ** 0.5

    off_s, pr98_s, full_s = _dscore(off), _dscore(pr98), _dscore(full)
    # Camera-float gain reported by the re-validation slice-1 (for the path comparison).
    camfloat_full_delta = s1.get("full_kit_delta_vs_base")
    result = {
        "authority": _ADVISORY,
        "n_pairs": args.n_pairs,
        "converged_archive": str(_CONVERGED_ARCHIVE),
        "fitted_constants_from": str(args.result_json),
        "production_path": "driver.kit_aware_exact_eval (post-round uint8, apply_distortion_kit_to_raw_frames)",
        "off_distortion_score": off_s,
        "pr98_distortion_score": pr98_s,
        "full_distortion_score": full_s,
        "pr98_delta_vs_off_production": pr98_s - off_s,
        "full_delta_vs_off_production": full_s - off_s,
        "off_d_seg": off["seg_distortion"], "off_d_pose": off["pose_distortion"],
        "full_d_seg": full["seg_distortion"], "full_d_pose": full["pose_distortion"],
        "camera_float_full_delta_slice1": camfloat_full_delta,
        "production_vs_camera_float_gap": (
            (full_s - off_s) - camfloat_full_delta if camfloat_full_delta is not None else None
        ),
        "note": (
            "production_vs_camera_float_gap is the ≤1-LSB post-round penalty/bonus on the "
            "fitted gain (LENS-2b). If the production gain is materially weaker than the "
            "camera-float gain, the ±1-bias fit is partly a pre-round artifact."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = args.out or Path(
        f".omx/research/finishing_kit_production_path_verify_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[prodpath] production full_delta={full_s-off_s:.6f} vs camera-float {camfloat_full_delta} "
          f"-> wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
