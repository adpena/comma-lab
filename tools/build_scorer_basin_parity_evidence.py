#!/usr/bin/env python3
"""Build scorer-basin-parity readiness-evidence JSON for apogee_intN candidates.

This is one of three accepted evidence types for ``predispatch_sanity.py``
Gate 6 (apogee_evidence_semantics) - see the gate definition lines 50-70 of
that file. Specifically: ``scorer_basin_parity_gate``.

The tool inflates two archives (the candidate quantized archive and the
lossless PR106 reference), runs the upstream PoseNet + SegNet on
reconstructions from both, and emits a JSON manifest that Gate 6 will
accept when ``--readiness-evidence-json`` is pointed at it.

Per CLAUDE.md per-tag discipline, the evidence is tagged
``[scorer-basin-parity:CPU]`` (or ``...:CUDA`` if --device cuda) - NEVER
``[contest-CUDA]``. The score numbers in this report are NOT scores, they
are basin-geometry deltas suitable only for predispatch readiness.

Reusable for apogee_int4/5/6/7/8 - pass any candidate archive and a
matching lossless reference. The decoder schema is currently the PR106
HNeRV layout (latent_dim=28, base_channels=36, eval_size=(384, 512)).

Example
-------

.. code-block:: bash

    .venv/bin/python tools/build_scorer_basin_parity_evidence.py \\
        --candidate-archive experiments/results/apogee_int6_repack_*/apogee_int6_archive.zip \\
        --lossless-archive experiments/results/public_pr106_belt_and_suspenders_intake_*/archive.zip \\
        --output-json experiments/results/apogee_int6_basin_parity_*/parity_evidence.json

Then::

    .venv/bin/python tools/predispatch_sanity.py \\
        --archive <candidate> \\
        --readiness-evidence-json <output-json> \\
        --predicted-low ... --predicted-high ... --rel-err-pct ... \\
        --lane-class apogee_intN
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import sha256_bytes, sha256_file, write_json  # noqa: E402
from tac.scorer_basin_parity import (  # noqa: E402
    DEFAULT_HESSIAN_LOG_RATIO_TOLERANCE,
    DEFAULT_POSE_DIST_DELTA_THRESHOLD,
    DEFAULT_SEG_DIST_DELTA_THRESHOLD,
    ParityReport,
    compute_scorer_basin_parity,
)

logger = logging.getLogger("scorer_basin_parity_evidence")


# ---------------------------------------------------------------------------
# Archive parsing - dispatches based on magic byte
# ---------------------------------------------------------------------------


def _load_apogee_intn_decoder_module() -> tuple:
    """Load ``submissions/apogee_intN`` codec/model modules dynamically.

    Returns (parse_apogee_intn_archive, parse_packed_archive, HNeRVDecoder).
    """

    apogee_root = REPO_ROOT / "submissions" / "apogee_intN"
    src_dir = apogee_root / "src"

    # Add src to path so codec.py can import its peers
    sys.path.insert(0, str(src_dir))

    inflate_path = apogee_root / "inflate.py"
    spec = importlib.util.spec_from_file_location(
        "_apogee_intn_inflate_module", inflate_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load apogee_intN inflate module at {inflate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # codec module is now in sys.modules under name 'codec'
    codec_module = sys.modules.get("codec")
    if codec_module is None:
        raise RuntimeError("apogee_intN codec module did not register in sys.modules")

    return (
        module.parse_apogee_intn_archive,
        codec_module.parse_packed_archive,
        sys.modules["model"].HNeRVDecoder,
    )


def _read_zip_payload(archive_zip: Path) -> bytes:
    """Read the single ``0.bin`` member from the archive ZIP."""

    with zipfile.ZipFile(archive_zip) as zf:
        names = zf.namelist()
        # Standard apogee/PR106 layout has only one .bin member.
        bin_names = [n for n in names if n.endswith(".bin")]
        if not bin_names:
            raise ValueError(f"no .bin member in {archive_zip}: {names}")
        return zf.read(bin_names[0])


def _parse_archive_to_state(
    archive_zip: Path,
    parse_apogee_intn_archive,
    parse_packed_archive,
):
    """Dispatch on first byte of inner .bin payload.

    Returns (state_dict, latents, meta).
    """

    payload = _read_zip_payload(archive_zip)
    if not payload:
        raise ValueError(f"empty .bin payload in {archive_zip}")
    magic = payload[0]
    if magic == 0xFF:
        # PR106 packed-archive layout
        return parse_packed_archive(payload)
    if (magic & 0xF0) == 0xA0:
        # apogee_intN
        return parse_apogee_intn_archive(payload)
    raise ValueError(
        f"{archive_zip}: unsupported magic byte 0x{magic:02X} - expected 0xFF (PR106) or 0xA[4-8] (apogee_intN)"
    )


# ---------------------------------------------------------------------------
# Ground-truth frame extraction (CPU-friendly via PyAV)
# ---------------------------------------------------------------------------


def _decode_first_n_frames(video_path: Path, n: int) -> torch.Tensor:
    """Decode the first ``n`` frames of ``video_path`` to (n, 874, 1164, 3) uint8.

    Uses the same yuv420->RGB BT.601 conversion as upstream's AVVideoDataset.
    """

    sys.path.insert(0, str(REPO_ROOT / "upstream"))
    import av  # type: ignore
    from frame_utils import yuv420_to_rgb

    frames: list[torch.Tensor] = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if len(frames) >= n:
                break
            frames.append(yuv420_to_rgb(frame))  # (H, W, 3) uint8
    finally:
        container.close()

    if len(frames) < n:
        raise ValueError(
            f"{video_path}: decoded only {len(frames)} frames; need {n}"
        )
    return torch.stack(frames, dim=0)  # (n, H, W, 3)


def build_gt_pairs(video_path: Path, n_pairs: int) -> tuple[torch.Tensor, list[str]]:
    """Return (n_pairs, 2, H, W, 3) uint8 tensor + per-frame SHA256 tags."""

    needed = 2 * n_pairs
    flat = _decode_first_n_frames(video_path, needed)
    pairs = flat.reshape(n_pairs, 2, *flat.shape[1:])
    shas: list[str] = []
    for i in range(n_pairs):
        # SHA over each pair's bytes (joining both frames) for forensic reference.
        shas.append(sha256_bytes(pairs[i].numpy().tobytes())[:16])
    return pairs, shas


# ---------------------------------------------------------------------------
# Upstream scorer instantiation
# ---------------------------------------------------------------------------


def _load_upstream_scorers(
    posenet_sd_path: Path,
    segnet_sd_path: Path,
    device: torch.device,
):
    """Load upstream PoseNet + SegNet onto ``device``."""

    sys.path.insert(0, str(REPO_ROOT / "upstream"))
    from modules import PoseNet, SegNet
    from safetensors.torch import load_file

    posenet = PoseNet().eval().to(device)
    segnet = SegNet().eval().to(device)

    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False

    return posenet, segnet


# ---------------------------------------------------------------------------
# Evidence-JSON emission
# ---------------------------------------------------------------------------


def emit_evidence_json(
    *,
    output_json_path: Path,
    candidate_archive: Path,
    candidate_archive_sha: str,
    lossless_archive: Path,
    lossless_archive_sha: str,
    report: ParityReport,
    extra_notes: str = "",
) -> None:
    """Write a Gate-6-compatible readiness-evidence JSON file.

    Schema is the union of:
      - candidate_archive_sha256 / archive_sha256 (Gate 6 looks for either)
      - evidence_semantics = "scorer_basin_parity_gate"
      - status = "pass" | "fail" (semantically same as basin_parity_passed)
      - scorer_basin_parity_status = "pass" | "fail" (Gate 6 verifies this directly)
      - ready_for_exact_eval_dispatch = bool
      - evidence_grade = "ready" | "negative"
      - notes = human-readable summary including the [scorer-basin-parity:CPU] tag
      - pose_dist_delta / seg_dist_delta = numeric deltas
      - basin_parity_passed = bool
      - parity_report = full ParityReport.to_dict() for forensics
    """

    status = "pass" if report.basin_parity_passed else "fail"
    evidence_grade = "ready" if report.basin_parity_passed else "negative"

    note_tag = f"[scorer-basin-parity:{report.device.upper()}]"
    notes_lines: list[str] = [
        note_tag,
        f"computed_utc={report.computed_utc}",
        f"n_probes={report.n_probes}, n_hessian_samples={report.n_hessian_samples}",
        f"pose_dist_lossless={report.pose_dist_lossless:.3e} -> "
        f"pose_dist_quantized={report.pose_dist_quantized:.3e} "
        f"(delta {report.pose_dist_delta:+.3e}, threshold {report.pose_threshold:.3e})",
        f"seg_dist_lossless={report.seg_dist_lossless:.3e} -> "
        f"seg_dist_quantized={report.seg_dist_quantized:.3e} "
        f"(delta {report.seg_dist_delta:+.3e}, threshold {report.seg_threshold:.3e})",
        f"hessian_trace_lossless={report.hessian_trace_lossless:.3e}, "
        f"hessian_trace_quantized={report.hessian_trace_quantized:.3e}, "
        f"log10_ratio={report.hessian_log_ratio:+.2f} "
        f"(tolerance |{report.hessian_log_ratio_tolerance:.2f}|)",
    ]
    if report.failure_reasons:
        notes_lines.append("FAIL: " + "; ".join(report.failure_reasons))
    if extra_notes:
        notes_lines.append(extra_notes)
    notes = " | ".join(notes_lines)

    payload: dict = {
        "candidate_archive_sha256": candidate_archive_sha,
        "archive_sha256": candidate_archive_sha,  # alias
        "candidate_archive_path": str(candidate_archive),
        "lossless_archive_sha256": lossless_archive_sha,
        "lossless_archive_path": str(lossless_archive),
        "evidence_semantics": "scorer_basin_parity_gate",
        "status": status,
        "scorer_basin_parity_status": status,
        "ready_for_exact_eval_dispatch": bool(report.basin_parity_passed),
        "evidence_grade": evidence_grade,
        "pose_dist_delta": float(report.pose_dist_delta),
        "seg_dist_delta": float(report.seg_dist_delta),
        "basin_parity_passed": bool(report.basin_parity_passed),
        "notes": notes,
        "parity_report": report.to_dict(),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json_path, payload)
    logger.info("Wrote readiness evidence to %s", output_json_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--candidate-archive",
        required=True,
        type=Path,
        help="path to the apogee_intN candidate archive.zip",
    )
    parser.add_argument(
        "--lossless-archive",
        required=True,
        type=Path,
        help="path to the PR106 lossless reference archive.zip",
    )
    parser.add_argument(
        "--scorer-posenet",
        type=Path,
        default=REPO_ROOT / "upstream" / "models" / "posenet.safetensors",
        help="path to upstream/models/posenet.safetensors",
    )
    parser.add_argument(
        "--scorer-segnet",
        type=Path,
        default=REPO_ROOT / "upstream" / "models" / "segnet.safetensors",
        help="path to upstream/models/segnet.safetensors",
    )
    parser.add_argument(
        "--anchor-video",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="GT video to extract anchor frame pairs from (default upstream 0.mkv)",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="destination JSON path (a Gate-6 compatible readiness evidence file)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="default cpu - this is a basin-geometry probe, not a contest-CUDA score",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=10,
        help="number of latent pairs / GT pairs used for the distortion-delta axis",
    )
    parser.add_argument(
        "--n-hessian-samples",
        type=int,
        default=4,
        help="number of Rademacher samples per Hutchinson trace estimate",
    )
    parser.add_argument(
        "--n-hessian-pairs",
        type=int,
        default=2,
        help="number of latent pairs used in the Hessian probe (kept tiny for CPU)",
    )
    parser.add_argument(
        "--pose-threshold",
        type=float,
        default=DEFAULT_POSE_DIST_DELTA_THRESHOLD,
        help=f"pose_dist_delta failure threshold (default {DEFAULT_POSE_DIST_DELTA_THRESHOLD})",
    )
    parser.add_argument(
        "--seg-threshold",
        type=float,
        default=DEFAULT_SEG_DIST_DELTA_THRESHOLD,
        help=f"seg_dist_delta failure threshold (default {DEFAULT_SEG_DIST_DELTA_THRESHOLD})",
    )
    parser.add_argument(
        "--hessian-log-ratio-tolerance",
        type=float,
        default=DEFAULT_HESSIAN_LOG_RATIO_TOLERANCE,
        help=("|log10(trace_quantized / trace_lossless)| tolerance "
              f"(default {DEFAULT_HESSIAN_LOG_RATIO_TOLERANCE})"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for Rademacher samples + numpy",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="emit DEBUG log lines",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    candidate_archive = args.candidate_archive.resolve()
    lossless_archive = args.lossless_archive.resolve()

    if not candidate_archive.is_file():
        logger.error("candidate archive not found: %s", candidate_archive)
        return 2
    if not lossless_archive.is_file():
        logger.error("lossless archive not found: %s", lossless_archive)
        return 2
    if not args.scorer_posenet.is_file():
        logger.error("posenet weights not found: %s", args.scorer_posenet)
        return 2
    if not args.scorer_segnet.is_file():
        logger.error("segnet weights not found: %s", args.scorer_segnet)
        return 2
    if not args.anchor_video.is_file():
        logger.error("anchor video not found: %s", args.anchor_video)
        return 2

    candidate_sha = sha256_file(candidate_archive)
    lossless_sha = sha256_file(lossless_archive)
    logger.info("candidate sha256: %s", candidate_sha)
    logger.info("lossless  sha256: %s", lossless_sha)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.error("requested --device cuda but torch.cuda.is_available() is False")
        return 2

    # ---- 1. Parse archives -> state_dicts + latents -----------------------
    parse_apogee_intn_archive, parse_packed_archive, HNeRVDecoder = (
        _load_apogee_intn_decoder_module()
    )

    cand_sd, cand_latents, cand_meta = _parse_archive_to_state(
        candidate_archive, parse_apogee_intn_archive, parse_packed_archive
    )
    loss_sd, loss_latents, loss_meta = _parse_archive_to_state(
        lossless_archive, parse_apogee_intn_archive, parse_packed_archive
    )

    logger.info(
        "candidate parsed: %d weight tensors, %s latents, meta=%s",
        len(cand_sd),
        tuple(cand_latents.shape),
        cand_meta,
    )
    logger.info(
        "lossless parsed: %d weight tensors, %s latents, meta=%s",
        len(loss_sd),
        tuple(loss_latents.shape),
        loss_meta,
    )

    # Sanity: both archives must produce compatible decoder state-dicts
    if set(cand_sd.keys()) != set(loss_sd.keys()):
        logger.error(
            "decoder state-dict key mismatch: candidate_only=%s, lossless_only=%s",
            set(cand_sd.keys()) - set(loss_sd.keys()),
            set(loss_sd.keys()) - set(cand_sd.keys()),
        )
        return 3

    # Use lossless latents as the shared reference (they're frame-pair-aligned
    # GT-encodings; the apogee repacker preserves them bit-exact, so cand and
    # loss latents should match).
    latents = loss_latents.to(device)

    # ---- 2. Build decoder shell ------------------------------------------
    decoder = HNeRVDecoder(
        latent_dim=loss_meta["latent_dim"],
        base_channels=loss_meta["base_channels"],
        eval_size=tuple(loss_meta["eval_size"]),
    ).to(device)

    # Move state-dicts to device
    cand_sd_dev = {k: v.to(device) for k, v in cand_sd.items()}
    loss_sd_dev = {k: v.to(device) for k, v in loss_sd.items()}

    # ---- 3. Decode anchor GT pairs ---------------------------------------
    logger.info("decoding %d GT pairs from %s", args.n_probes, args.anchor_video)
    gt_pairs, gt_shas = build_gt_pairs(args.anchor_video, args.n_probes)
    gt_pairs = gt_pairs.to(device)
    logger.info(
        "GT pairs decoded: shape=%s, first-pair-sha=%s",
        tuple(gt_pairs.shape),
        gt_shas[0],
    )

    # ---- 4. Load scorers --------------------------------------------------
    logger.info("loading upstream scorers (PoseNet + SegNet) on %s", device)
    posenet, segnet = _load_upstream_scorers(
        args.scorer_posenet, args.scorer_segnet, device
    )

    # ---- 5. Run parity probe ---------------------------------------------
    logger.info("running scorer-basin-parity probe (this is the slow step)...")
    report = compute_scorer_basin_parity(
        quantized_state_dict=cand_sd_dev,
        lossless_state_dict=loss_sd_dev,
        decoder=decoder,
        posenet=posenet,
        segnet=segnet,
        latents=latents,
        gt_frames=gt_pairs,
        n_probes=args.n_probes,
        pose_threshold=args.pose_threshold,
        seg_threshold=args.seg_threshold,
        hessian_log_ratio_tolerance=args.hessian_log_ratio_tolerance,
        n_hessian_samples=args.n_hessian_samples,
        n_hessian_pairs=args.n_hessian_pairs,
        anchor_frame_shas=gt_shas,
        device=str(device),
        seed=args.seed,
    )

    logger.info(
        "parity result: passed=%s pose_delta=%.3e seg_delta=%.3e log_ratio=%+0.2f",
        report.basin_parity_passed,
        report.pose_dist_delta,
        report.seg_dist_delta,
        report.hessian_log_ratio,
    )
    for reason in report.failure_reasons:
        logger.warning("FAIL reason: %s", reason)

    # ---- 6. Emit JSON ----------------------------------------------------
    extra_notes = (
        f"candidate={candidate_archive.name} "
        f"lossless={lossless_archive.name} "
        f"meta={loss_meta}"
    )
    emit_evidence_json(
        output_json_path=args.output_json,
        candidate_archive=candidate_archive,
        candidate_archive_sha=candidate_sha,
        lossless_archive=lossless_archive,
        lossless_archive_sha=lossless_sha,
        report=report,
        extra_notes=extra_notes,
    )

    # Exit code 0 on PASS, 1 on FAIL - the JSON is written either way so
    # the operator can attach the negative evidence to a DEFER decision.
    return 0 if report.basin_parity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
