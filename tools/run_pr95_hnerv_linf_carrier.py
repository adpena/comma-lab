# SPDX-License-Identifier: MIT
"""Run the PR95-HNeRV inverse-steganalysis carrier head-to-head row (advisory).

Binds the §7-proven L-inf margin-budget OBJECTIVE onto the RESOLVED PR95-HNeRV
CARRIER and emits the advisory head-to-head row:

  rate_term (cheap-by-construction --modelsize budget)
    x advisory d_seg/d_pose (carrier render vs REAL 0.mkv gt, bit-exact CPU mirror)
    x Z8-falsification (PR95-HNeRV rate << Z8 rate at comparable distortion)
    + L-inf-vs-L2 latent allocation (the §7 objective in the carrier's 28-d domain)

$0 macOS-CPU/MLX-local ONLY. NO paid dispatch, NO cloud GPU, NO PR. The carrier
render is ``[macOS-MLX research-signal]``; the d_seg/d_pose CPU-mirror measurement
is ``[macOS-CPU advisory]`` (frozen weights, Apple-Silicon CPU, NOT contest
GHA-Linux-x86_64). NON-PROMOTABLE per Catalog #341/#192/#127/#323. Paired CPU+CUDA
(Catalog #246) reserved for operator authorization.

Usage::

    .venv/bin/python tools/run_pr95_hnerv_linf_carrier.py \
        --archive experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip \
        --num-pairs 4 --pair-stride 64 --latent-bits-per-coeff 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.pr95_hnerv_linf_carrier import (  # noqa: E402
    allocate_latent_linf_vs_l2,
    build_head_to_head_row,
    carrier_rate_term,
    load_carrier_decoder,
    measure_carrier_distortion,
    push_pixel_saliency_to_latent,
    z8_falsification,
)


def _utc_now() -> str:
    import datetime

    return (
        datetime.datetime.now(datetime.UTC)
        .strftime("%Y%m%dT%H%M%SZ")
    )


def run(
    *,
    archive: str,
    upstream_dir: str,
    video_path: str,
    num_pairs: int,
    pair_stride: int,
    latent_bits_per_coeff: float,
    measure_distortion: bool,
    measure_latent_allocation: bool,
    fd_eps: float,
) -> dict[str, object]:
    """Build the advisory head-to-head row for a REAL PR95-HNeRV carrier archive."""
    rt = carrier_rate_term(archive)
    fz = z8_falsification(rt)

    distortion = None
    latent_alloc = None
    decoder = None
    latents = None
    diag: dict[str, object] = {}

    if measure_distortion or measure_latent_allocation:
        from tac.analysis.score_exact_saliency import (
            compute_s_seg_flip_risk,
            decode_real_pairs,
            load_score_exact_scorers,
        )

        decoder, latents, _ = load_carrier_decoder(archive)
        n_pairs = min(int(num_pairs), int(rt.n_pairs))
        # Map carrier latent rows to gt pairs at the SAME stride (1:1 ordering).
        pair_indices = [min(i * int(pair_stride), int(rt.n_pairs) - 1) for i in range(n_pairs)]
        gt_pairs = decode_real_pairs(
            video_path, n_pairs, pair_stride=int(pair_stride), start_pair=0, device="cpu"
        )
        posenet, segnet = load_score_exact_scorers(upstream_dir, device="cpu")

        if measure_distortion:
            distortion = measure_carrier_distortion(
                decoder, latents, gt_pairs, posenet, segnet, pair_indices=pair_indices
            )

        if measure_latent_allocation:
            # Push the oracle pixel saliency on pair 0 into the carrier's latent domain
            # and run the §7-proven L-inf-vs-L2 allocation at equal latent rate.
            sseg = compute_s_seg_flip_risk(segnet, gt_pairs[0:1])
            sp = sseg.flip_risk.detach().cpu().numpy()  # (384, 512)
            lat_sal = push_pixel_saliency_to_latent(
                decoder, latents[pair_indices[0]], sp, frame_slot=1, eps=float(fd_eps)
            )
            target_bits = float(rt.latent_dim) * float(latent_bits_per_coeff)
            latent_alloc = allocate_latent_linf_vs_l2(
                lat_sal.s_latent, latents[pair_indices[0]], target_bits=target_bits
            )
            diag["latent_saliency_nonzero"] = int((lat_sal.s_latent > 0).sum())
            diag["latent_saliency_method"] = lat_sal.method
            diag["latent_fd_eps"] = float(fd_eps)

    row = build_head_to_head_row(rt, distortion, fz, latent_allocation=latent_alloc)
    row["generated_at_utc"] = _utc_now()
    row["video_path"] = video_path
    row["num_pairs_requested"] = int(num_pairs)
    row["pair_stride"] = int(pair_stride)
    row["diagnostics"] = diag
    return row


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive",
        default=(
            "experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/"
            "archive.pr95_repacked.zip"
        ),
        help="REAL PR95-HNeRV public archive .zip (the carrier instance).",
    )
    p.add_argument("--upstream-dir", default="upstream")
    p.add_argument("--video-path", default="upstream/videos/0.mkv")
    p.add_argument("--num-pairs", type=int, default=4)
    p.add_argument("--pair-stride", type=int, default=64)
    p.add_argument("--latent-bits-per-coeff", type=float, default=4.0)
    p.add_argument("--fd-eps", type=float, default=1.0e-2)
    p.add_argument(
        "--skip-distortion",
        action="store_true",
        help="Skip the carrier render + advisory d_seg/d_pose (rate + Z8 row only).",
    )
    p.add_argument(
        "--skip-latent-allocation",
        action="store_true",
        help="Skip the L-inf-vs-L2 latent allocation probe.",
    )
    p.add_argument("--output-json", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    # Force CPU; never MPS authority per CLAUDE.md "MPS auth eval is NOISE".
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = _build_arg_parser().parse_args(argv)
    t0 = time.perf_counter()
    row = run(
        archive=args.archive,
        upstream_dir=args.upstream_dir,
        video_path=args.video_path,
        num_pairs=args.num_pairs,
        pair_stride=args.pair_stride,
        latent_bits_per_coeff=args.latent_bits_per_coeff,
        measure_distortion=not args.skip_distortion,
        measure_latent_allocation=not args.skip_latent_allocation,
        fd_eps=args.fd_eps,
    )
    row["wall_clock_seconds"] = float(time.perf_counter() - t0)

    out = args.output_json
    if out is None:
        out = f".omx/research/pr95_hnerv_linf_carrier_head_to_head_{row['generated_at_utc']}.json"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    resolved = out_path.as_posix()
    if resolved.startswith("/tmp/") or resolved.startswith("/private/tmp/") or resolved.startswith("/var/tmp/"):
        raise SystemExit("refusing system /tmp evidence path per CLAUDE.md transient-evidence trap")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")

    fz = row["z8_falsification"]
    print(
        "[PR95-HNeRV head-to-head] "
        f"rate={row['rate_term']:.4f} ({row['carrier_archive_bytes']:,} B cheap-by-construction)"
    )
    if row.get("advisory_d_seg") is not None:
        print(
            f"[advisory] d_seg={row['advisory_d_seg']:.4f} d_pose={row['advisory_d_pose']:.3e} "
            f"distortion_score={row['advisory_distortion_only_score']:.3f} "
            f"full_score_est={row['advisory_full_score_estimate']:.3f} "
            f"{row.get('measure_axis_tag')}"
        )
    if "latent_linf_vs_l2" in row:
        la = row["latent_linf_vs_l2"]
        print(
            f"[objective §7] L-inf bits={la['linf_bits']:.1f} L2 bits={la['l2_bits']:.1f} "
            f"differ={la['allocations_differ']}"
        )
    print(
        f"[Z8-falsification] PR95-HNeRV {row['carrier_archive_bytes']:,} B "
        f"<< Z8 {fz['z8_archive_bytes']:,} B = {fz['z8_over_pr95_byte_ratio']:.0f}x heavier; "
        f"Z8-disease={fz['z8_disease_confirmed']}"
    )
    print(f"[anchor] {out_path}  (NON-PROMOTABLE, score_claim={row['score_claim']})")
    print(f"[wall] {row['wall_clock_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
