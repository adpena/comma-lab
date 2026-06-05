# SPDX-License-Identifier: MIT
"""Run the MLX scorer-prefilter directly on a SNeRV receiver archive.

This is the non-training comparison path for SNeRV packet experiments: consume
the actual receiver packet from ``archive.zip``/``0.bin``, decode with the
portable SNeRV receiver, wrap the decoded frames as the shared MLX
``RendererBundle``, and let the canonical MLX prefilter report frame-1 SegNet,
two-frame PoseNet, and rate components.  The output is false-authority local
MLX evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES
from tac.local_acceleration.mlx_renderer_prefilter_profile import (
    FALSE_AUTHORITY,
    write_mlx_renderer_prefilter_profile,
)
from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SNERV_ARCHIVE_MAGIC,
    SNERV_ARCHIVE_MAGIC_V2,
    decode_snerv_archive_frames,
)

DEFAULT_REQUIRED_NON_RATE_DROP = 0.22948096374507201
DEFAULT_ZIP_MEMBER = "0.bin"


class StaticSnervDecodedFramesModel:
    """MLX renderer adapter for decoded receiver frames.

    The shared prefilter expects ``model(idx) -> (B, 2, 3, H, W)`` in byte
    domain.  SNeRV receiver replay already returns exactly that layout, so this
    model is only an indexed view over the decoded packet tensor.
    """

    def __init__(self, frames_b2chw255: np.ndarray) -> None:
        import mlx.core as mx

        frames = np.asarray(frames_b2chw255, dtype=np.float32)
        if frames.ndim != 5 or frames.shape[1] != 2 or frames.shape[2] != 3:
            raise ValueError(
                "decoded SNeRV frames must have shape (pairs, 2, 3, H, W); "
                f"got {tuple(frames.shape)}"
            )
        self._frames = mx.array(frames)
        self.num_pairs = int(frames.shape[0])
        self.height = int(frames.shape[3])
        self.width = int(frames.shape[4])

    def __call__(self, idx: Any) -> Any:
        import mlx.core as mx

        return mx.take(self._frames, idx, axis=0)


def read_snerv_packet_from_archive_or_raw(
    path: str | Path,
    *,
    member: str = DEFAULT_ZIP_MEMBER,
) -> tuple[bytes, dict[str, Any]]:
    """Read a SNeRV SNAR packet from raw packet bytes or ``archive.zip``."""

    source = Path(path).expanduser().resolve(strict=True)
    blob = source.read_bytes()
    if blob.startswith((SNERV_ARCHIVE_MAGIC, SNERV_ARCHIVE_MAGIC_V2)):
        return blob, {
            "schema": "snerv_packet_input.v1",
            "input_path": source.as_posix(),
            "input_kind": "raw_snar_packet",
            "archive_member": None,
            "input_bytes": len(blob),
            "input_sha256": hashlib.sha256(blob).hexdigest(),
            "packet_bytes": len(blob),
            "packet_sha256": hashlib.sha256(blob).hexdigest(),
        }

    if not zipfile.is_zipfile(source):
        raise ValueError(
            f"{source}: expected raw SNAR1/SNAR2 packet or archive.zip containing {member!r}"
        )
    with zipfile.ZipFile(source) as zf:
        names = zf.namelist()
        selected_member = member
        if selected_member not in names:
            selected_member = _find_snar_member(zf, names)
        packet = zf.read(selected_member)
    if not packet.startswith((SNERV_ARCHIVE_MAGIC, SNERV_ARCHIVE_MAGIC_V2)):
        raise ValueError(
            f"{source}:{selected_member}: expected SNAR1/SNAR2 packet, got "
            f"first bytes {packet[:8]!r}"
        )
    return packet, {
        "schema": "snerv_packet_input.v1",
        "input_path": source.as_posix(),
        "input_kind": "archive_zip",
        "archive_member": selected_member,
        "archive_member_requested": member,
        "archive_members": names,
        "input_bytes": source.stat().st_size,
        "input_sha256": hashlib.sha256(blob).hexdigest(),
        "packet_bytes": len(packet),
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
    }


def _find_snar_member(zf: zipfile.ZipFile, names: list[str]) -> str:
    for name in names:
        blob = zf.read(name)
        if blob.startswith((SNERV_ARCHIVE_MAGIC, SNERV_ARCHIVE_MAGIC_V2)):
            return name
    raise ValueError(
        "archive member does not contain a SNAR1/SNAR2 packet; "
        f"members={names!r}"
    )


def build_snerv_archive_prefilter_bundle(
    *,
    packet: bytes,
    source_video_path: str | Path,
    num_pairs: int | None = None,
) -> tuple[RendererBundle, dict[str, Any]]:
    """Decode a receiver packet and build the shared MLX prefilter bundle."""

    frames = decode_snerv_archive_frames(packet)
    if frames.ndim != 5 or frames.shape[1] != 2 or frames.shape[2] != 3:
        raise ValueError(
            "SNeRV receiver decode returned unexpected shape "
            f"{tuple(frames.shape)}; expected (pairs, 2, 3, H, W)"
        )
    available_pairs = int(frames.shape[0])
    selected_pairs = available_pairs if num_pairs is None else int(num_pairs)
    if selected_pairs < 1 or selected_pairs > available_pairs:
        raise ValueError(
            f"num_pairs must be in [1, {available_pairs}], got {selected_pairs}"
        )
    frames = np.asarray(frames[:selected_pairs], dtype=np.float32)
    height = int(frames.shape[3])
    width = int(frames.shape[4])
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        source_video_path,
        num_pairs=selected_pairs,
        output_height=height,
        output_width=width,
    )
    bundle = RendererBundle(
        model=StaticSnervDecodedFramesModel(frames),
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=selected_pairs,
        forward_convention="call_b2chw_255",
        substrate_artifact_metadata={
            "schema": "snerv_archive_prefilter_bundle.v1",
            "receiver_packet_consumed": True,
            "decoded_receiver_frame_shape": [int(v) for v in frames.shape],
            "false_authority_local_mlx_only": True,
        },
    )
    return bundle, {
        "schema": "snerv_archive_prefilter_bundle_summary.v1",
        "available_pairs": available_pairs,
        "selected_pairs": selected_pairs,
        "decoded_receiver_frame_shape": [int(v) for v in frames.shape],
        "height": height,
        "width": width,
    }


def compare_prefilter_profiles(
    *,
    baseline_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
    required_non_rate_drop: float = DEFAULT_REQUIRED_NON_RATE_DROP,
) -> dict[str, Any]:
    """Compare scalar/shared SNeRV MLX-prefilter profiles in contest units."""

    base = _components(baseline_profile)
    cand = _components(candidate_profile)
    baseline_bytes = int(
        baseline_profile.get("archive_bytes")
        or baseline_profile.get("archive_size_bytes")
        or base["archive_bytes"]
    )
    candidate_bytes = int(
        candidate_profile.get("archive_bytes")
        or candidate_profile.get("archive_size_bytes")
        or cand["archive_bytes"]
    )
    byte_delta = candidate_bytes - baseline_bytes
    rate_delta = 25.0 * float(byte_delta) / float(ORIGINAL_VIDEO_BYTES)
    seg_delta = cand["avg_segnet_dist"] - base["avg_segnet_dist"]
    pose_delta = cand["avg_posenet_dist"] - base["avg_posenet_dist"]
    seg_term_delta = cand["seg_term"] - base["seg_term"]
    pose_term_delta = cand["pose_term"] - base["pose_term"]
    non_rate_delta = seg_term_delta + pose_term_delta
    score_delta = non_rate_delta + rate_delta
    non_rate_drop = -non_rate_delta
    return {
        "schema": "snerv_skip_high_prefilter_profile_comparison.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "authority": "macOS MLX local prefilter only; no contest score claim.",
        **FALSE_AUTHORITY,
        "contest_lagrangian": {
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/original_video_bytes",
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "byte_price": 25.0 / float(ORIGINAL_VIDEO_BYTES),
            "segnet_domain": "pair frame 1 only",
            "posenet_domain": "both frames through YUV6 pair input",
        },
        "baseline": {
            "archive_bytes": baseline_bytes,
            "avg_segnet_dist": base["avg_segnet_dist"],
            "avg_posenet_dist": base["avg_posenet_dist"],
            "seg_term": base["seg_term"],
            "pose_term": base["pose_term"],
            "rate_term": 25.0 * baseline_bytes / float(ORIGINAL_VIDEO_BYTES),
            "canonical_score": base["canonical_score"],
        },
        "candidate": {
            "archive_bytes": candidate_bytes,
            "avg_segnet_dist": cand["avg_segnet_dist"],
            "avg_posenet_dist": cand["avg_posenet_dist"],
            "seg_term": cand["seg_term"],
            "pose_term": cand["pose_term"],
            "rate_term": 25.0 * candidate_bytes / float(ORIGINAL_VIDEO_BYTES),
            "canonical_score": cand["canonical_score"],
        },
        "deltas_candidate_minus_baseline": {
            "archive_bytes": byte_delta,
            "rate_score": rate_delta,
            "frame1_segnet_avg_dist": seg_delta,
            "two_frame_posenet_avg_dist": pose_delta,
            "seg_term": seg_term_delta,
            "pose_term": pose_term_delta,
            "non_rate_score": non_rate_delta,
            "canonical_score": score_delta,
        },
        "admission": {
            "required_non_rate_drop": float(required_non_rate_drop),
            "observed_non_rate_drop": non_rate_drop,
            "observed_score_delta": score_delta,
            "passes_required_drop": bool(non_rate_drop >= float(required_non_rate_drop)),
            "passes_actual_byte_price": bool(score_delta < 0.0),
            "margin_vs_required_drop": non_rate_drop - float(required_non_rate_drop),
            "margin_vs_actual_byte_price": -score_delta,
        },
    }


def profile_snerv_archive(
    *,
    archive_path: str | Path,
    output_dir: str | Path,
    source_video_path: str | Path,
    upstream_dir: str | Path,
    archive_member: str = DEFAULT_ZIP_MEMBER,
    num_pairs: int | None = None,
    scorer_device: str = "gpu",
    scorer_batch_pairs: int = 1,
    progress_every: int = 25,
    required_pairs: int = 600,
    run_id: str | None = None,
    archive_bytes: int | None = None,
    archive_sha256: str | None = None,
    baseline_profile_path: str | Path | None = None,
    required_non_rate_drop: float = DEFAULT_REQUIRED_NON_RATE_DROP,
) -> dict[str, Any]:
    """Profile a SNeRV receiver archive and write profile/comparison artifacts."""

    archive = Path(archive_path).expanduser().resolve(strict=True)
    out_dir = Path(output_dir).expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    packet, packet_info = read_snerv_packet_from_archive_or_raw(
        archive,
        member=archive_member,
    )
    archive_size = int(archive_bytes) if archive_bytes is not None else archive.stat().st_size
    archive_hash = archive_sha256 or hashlib.sha256(archive.read_bytes()).hexdigest()
    bundle, bundle_info = build_snerv_archive_prefilter_bundle(
        packet=packet,
        source_video_path=source_video_path,
        num_pairs=num_pairs,
    )
    actual_run_id = run_id or f"snerv_archive_prefilter_{archive_hash[:12]}"
    profile_path = out_dir / "local_mlx_prefilter_profile.json"
    progress_path = out_dir / "local_mlx_prefilter_progress.jsonl"
    profile = write_mlx_renderer_prefilter_profile(
        bundle=bundle,
        output_path=profile_path,
        archive_bytes=archive_size,
        archive_sha256=archive_hash,
        upstream_dir=upstream_dir,
        scorer_device=scorer_device,
        scorer_batch_pairs=scorer_batch_pairs,
        required_pairs=required_pairs,
        run_id=actual_run_id,
        source_video_path=source_video_path,
        progress_jsonl_path=progress_path,
        progress_every=progress_every,
    )
    comparison_path = None
    comparison = None
    if baseline_profile_path is not None:
        baseline_path = Path(baseline_profile_path).expanduser().resolve(strict=True)
        baseline_profile = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = compare_prefilter_profiles(
            baseline_profile=baseline_profile,
            candidate_profile=profile,
            required_non_rate_drop=required_non_rate_drop,
        )
        comparison.update(
            {
                "baseline_profile_path": baseline_path.as_posix(),
                "candidate_profile_path": profile_path.as_posix(),
            }
        )
        comparison_path = out_dir / "snerv_skip_high_prefilter_comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    manifest_payload = {
        "schema": "snerv_archive_mlx_prefilter_manifest.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/profile_snerv_archive_mlx_prefilter.py",
        "run_id": actual_run_id,
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "archive_path": archive.as_posix(),
        "archive_bytes": archive_size,
        "archive_sha256": archive_hash,
        "packet_input": packet_info,
        "bundle": bundle_info,
        "profile_path": profile_path.as_posix(),
        "progress_jsonl_path": progress_path.as_posix(),
        "comparison_path": comparison_path.as_posix() if comparison_path else None,
        "score_components": profile.get("score_components"),
        "blockers": profile.get("blockers", []),
        **FALSE_AUTHORITY,
        "authority": (
            "macOS MLX archive prefilter only. Use receiver-closed exact CPU/CUDA "
            "replay before score, promotion, rank, or kill claims."
        ),
    }
    manifest_path = out_dir / "snerv_archive_mlx_prefilter_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_path": manifest_path.as_posix(),
        "profile_path": profile_path.as_posix(),
        "progress_jsonl_path": progress_path.as_posix(),
        "comparison_path": comparison_path.as_posix() if comparison_path else None,
        "profile": profile,
        "comparison": comparison,
    }


def _components(profile: dict[str, Any]) -> dict[str, float]:
    components = profile.get("score_components")
    if not isinstance(components, dict):
        raise ValueError("profile missing score_components")
    seg = float(components["avg_segnet_dist"])
    pose = float(components["avg_posenet_dist"])
    seg_term = float(components.get("seg_term", 100.0 * seg))
    pose_term = float(components.get("pose_term", math.sqrt(10.0 * pose)))
    archive_bytes = float(
        components.get("archive_bytes")
        or profile.get("archive_bytes")
        or profile.get("archive_size_bytes")
        or 0.0
    )
    canonical_score = float(
        components.get(
            "canonical_score",
            seg_term
            + pose_term
            + 25.0 * archive_bytes / float(ORIGINAL_VIDEO_BYTES),
        )
    )
    return {
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "archive_bytes": archive_bytes,
        "canonical_score": canonical_score,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-video-path", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--archive-member", default=DEFAULT_ZIP_MEMBER)
    parser.add_argument("--num-pairs", type=int)
    parser.add_argument("--required-pairs", type=int, default=600)
    parser.add_argument("--scorer-device", default="gpu", choices=("cpu", "gpu"))
    parser.add_argument("--scorer-batch-pairs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--run-id")
    parser.add_argument("--archive-bytes", type=int)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--baseline-profile", type=Path)
    parser.add_argument(
        "--required-non-rate-drop",
        type=float,
        default=DEFAULT_REQUIRED_NON_RATE_DROP,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = profile_snerv_archive(
        archive_path=args.archive,
        output_dir=args.output_dir,
        source_video_path=args.source_video_path,
        upstream_dir=args.upstream_dir,
        archive_member=args.archive_member,
        num_pairs=args.num_pairs,
        scorer_device=args.scorer_device,
        scorer_batch_pairs=args.scorer_batch_pairs,
        progress_every=args.progress_every,
        required_pairs=args.required_pairs,
        run_id=args.run_id,
        archive_bytes=args.archive_bytes,
        archive_sha256=args.archive_sha256,
        baseline_profile_path=args.baseline_profile,
        required_non_rate_drop=args.required_non_rate_drop,
    )
    print(json.dumps({k: v for k, v in result.items() if k.endswith("_path")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
