#!/usr/bin/env python3
"""Measure an exact n600 C2 receiver control with hard CPU-Torch scoring.

This wrapper is deliberately fail-closed about R1b authority.  It derives a
full production-receiver control row from exact bytes, but marks the boundary
candidate absent unless a counted packet is actually in the parsed archive.
Large decode scratch lives on the SSD tier and is removed only after an atomic
machine-readable receipt certifies deterministic reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tac.boundary_math.integer_plane_emitter_byte_close import (  # noqa: E402
    LOGICAL_PAIR_COUNT,
    archive_receipt,
    decode_counted_archive,
    parse_counted_archive,
)
from tac.boundary_math.shared_receiver_admission import (  # noqa: E402
    MAX_ARCHIVE_BYTES,
    MAX_D_SEG,
    SCORE_BYTES_NORMALIZER,
)

SCHEMA: Final = "r1b_boundary_generator_n600_measurement.v1"
FIXED_C1_CAP_BYTES: Final = 216_223
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
FRAME_COUNT: Final = 1200
CHANNELS: Final = 3
EXPECTED_RAW_BYTES: Final = FRAME_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
SCORER_FILES: Final = (
    ("modules.py", "modules.py"),
    ("frame_utils.py", "frame_utils.py"),
    ("posenet.safetensors", "models/posenet.safetensors"),
    ("segnet.safetensors", "models/segnet.safetensors"),
)


class R1BMeasurementError(RuntimeError):
    """Fail-closed custody, storage, decoder, or scorer error."""


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compose_score(*, d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, float]:
    if not all(math.isfinite(value) and value >= 0.0 for value in (d_seg, d_pose)):
        raise R1BMeasurementError("distortions must be finite and non-negative")
    if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes <= 0:
        raise R1BMeasurementError("archive_bytes must be a positive integer")
    seg_component = 100.0 * d_seg
    pose_component = math.sqrt(10.0 * d_pose)
    rate_component = 25.0 * archive_bytes / SCORE_BYTES_NORMALIZER
    return {
        "seg_component": seg_component,
        "pose_component": pose_component,
        "rate_component": rate_component,
        "score": seg_component + pose_component + rate_component,
    }


def gate_summary(*, archive_bytes: int, d_seg: float) -> dict[str, Any]:
    bytes_gate = archive_bytes <= MAX_ARCHIVE_BYTES
    fixed_cap = archive_bytes <= FIXED_C1_CAP_BYTES
    distortion_gate = d_seg <= MAX_D_SEG
    return {
        "task_archive_gate_bytes": MAX_ARCHIVE_BYTES,
        "archive_bytes": archive_bytes,
        "archive_margin_bytes": MAX_ARCHIVE_BYTES - archive_bytes,
        "archive_gate_pass": bytes_gate,
        "fixed_c1_cap_bytes": FIXED_C1_CAP_BYTES,
        "fixed_c1_margin_bytes": FIXED_C1_CAP_BYTES - archive_bytes,
        "fixed_c1_cap_pass": fixed_cap,
        "d_seg_gate": MAX_D_SEG,
        "d_seg": d_seg,
        "d_seg_margin": MAX_D_SEG - d_seg,
        "d_seg_gate_pass": distortion_gate,
        "joint_task_gate_pass": bytes_gate and distortion_gate,
        "joint_fixed_c1_gate_pass": fixed_cap and distortion_gate,
    }


def storage_preflight(scratch_root: Path) -> dict[str, Any]:
    resolved = scratch_root.expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in SSD_ROOTS):
        raise R1BMeasurementError(
            "scratch_root must be under /Volumes/VertigoDataTier/pact or /Volumes/APDataStore/pact"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(resolved)
    required = 2 * EXPECTED_RAW_BYTES + 2_000_000_000
    result = {
        "scratch_root": str(resolved),
        "expected_output_raw_bytes": EXPECTED_RAW_BYTES,
        "required_free_bytes": required,
        "free_bytes": usage.free,
        "ok": usage.free >= required,
    }
    if not result["ok"]:
        raise R1BMeasurementError(
            f"storage preflight refused: {usage.free} B free < {required} B required"
        )
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise R1BMeasurementError(f"receipt overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if partial.exists():
            partial.unlink()


def _git_custody() -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {"head": sha, "dirty_paths": status}


def _source_hashes(upstream: Path) -> dict[str, str]:
    paths = {
        label: (upstream / relative).resolve(strict=True)
        for label, relative in SCORER_FILES
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _score_raw_cpu(
    *,
    raw_path: Path,
    upstream: Path,
    batch_size: int,
    cpu_threads: int,
    seed: int,
) -> dict[str, Any]:
    if raw_path.name != "0.raw":
        raise R1BMeasurementError("TensorVideoDataset custody requires decoded path named 0.raw")
    if raw_path.stat().st_size != EXPECTED_RAW_BYTES:
        raise R1BMeasurementError("decoded raw does not contain exactly n600 frame pairs")
    if batch_size != 16:
        raise R1BMeasurementError("hard CPU row requires canonical evaluator batch_size=16")
    if cpu_threads <= 0:
        raise R1BMeasurementError("cpu_threads must be positive")

    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    upstream_string = str(upstream)
    if upstream_string not in sys.path:
        sys.path.insert(0, upstream_string)
    try:
        from frame_utils import AVVideoDataset, TensorVideoDataset
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path
    except ImportError as exc:
        raise R1BMeasurementError("failed to import the explicit upstream scorer") from exc

    device = torch.device("cpu")
    scorer = DistortionNet().eval().to(device=device)
    scorer.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    ground_truth = AVVideoDataset(
        ["0.mkv"],
        data_dir=upstream / "videos",
        batch_size=batch_size,
        device=device,
        num_threads=1,
        seed=seed,
        prefetch_queue_depth=1,
    )
    candidate = TensorVideoDataset(
        ["0.mkv"],
        data_dir=raw_path.parent,
        batch_size=batch_size,
        device=device,
        num_threads=1,
        seed=seed,
        prefetch_queue_depth=1,
    )
    ground_truth.prepare_data()
    candidate.prepare_data()
    gt_loader = torch.utils.data.DataLoader(ground_truth, batch_size=None, num_workers=0)
    candidate_loader = torch.utils.data.DataLoader(candidate, batch_size=None, num_workers=0)
    pose_sum = torch.zeros([], device=device)
    seg_sum = torch.zeros([], device=device)
    sample_count = 0
    batch_count = 0
    with torch.inference_mode():
        for gt_row, candidate_row in zip(gt_loader, candidate_loader, strict=True):
            batch_gt = gt_row[2].to(device)
            batch_candidate = candidate_row[2].to(device)
            if batch_gt.shape != batch_candidate.shape:
                raise R1BMeasurementError("ground-truth/candidate batch geometry mismatch")
            pose_dist, seg_dist = scorer.compute_distortion(batch_gt, batch_candidate)
            if pose_dist.shape != (batch_gt.shape[0],) or seg_dist.shape != (
                batch_gt.shape[0],
            ):
                raise R1BMeasurementError("scorer distortion output geometry mismatch")
            pose_sum += pose_dist.sum()
            seg_sum += seg_dist.sum()
            sample_count += int(batch_gt.shape[0])
            batch_count += 1
    if sample_count != LOGICAL_PAIR_COUNT:
        raise R1BMeasurementError(
            f"hard scorer covered {sample_count} pairs, expected {LOGICAL_PAIR_COUNT}"
        )
    return {
        "pair_count": sample_count,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "device": "cpu",
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads(),
        "seed": seed,
        "d_pose": float((pose_sum / sample_count).item()),
        "d_seg": float((seg_sum / sample_count).item()),
    }


def _base_packet_pose_custody(base_packet: bytes) -> dict[str, Any]:
    magic = b"LVLS1\x00"
    if not base_packet.startswith(magic) or len(base_packet) < len(magic) + 4:
        raise R1BMeasurementError("base packet is not canonical LVLS1")
    manifest_size = int.from_bytes(
        base_packet[len(magic) : len(magic) + 4], "little", signed=False
    )
    start = len(magic) + 4
    end = start + manifest_size
    try:
        manifest = json.loads(base_packet[start:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R1BMeasurementError("base LVLS1 manifest is invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("n_pairs") != LOGICAL_PAIR_COUNT:
        raise R1BMeasurementError("base LVLS1 manifest lacks n600 custody")
    return {
        "has_pose_sidecar": manifest.get("has_pose_sidecar"),
        "pose_carrier_manifest": manifest.get("pose_carrier"),
        "xi_receiver_present": bool(
            manifest.get("has_pose_sidecar") or manifest.get("pose_carrier")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--base-decoder", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    archive = args.archive.expanduser().resolve(strict=True)
    decoder = args.base_decoder.expanduser().resolve(strict=True)
    upstream = args.upstream.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if args.decode_workers <= 0:
        raise R1BMeasurementError("decode_workers must be positive")
    preflight = storage_preflight(args.scratch_root)
    scratch_root = Path(preflight["scratch_root"])
    parsed = parse_counted_archive(archive)
    if parsed.manifest["pair_count"] != LOGICAL_PAIR_COUNT:
        raise R1BMeasurementError("production archive must declare exactly n600")
    scorer_hashes = _source_hashes(upstream)
    source_video = (upstream / "videos" / "0.mkv").resolve(strict=True)
    source_bytes = source_video.stat().st_size
    if source_bytes != SCORE_BYTES_NORMALIZER:
        raise R1BMeasurementError(
            f"source byte denominator drift: {source_bytes} != {SCORE_BYTES_NORMALIZER}"
        )

    run_dir = scratch_root / f"r1b_n600_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    inflated = run_dir / "inflated"
    inflated.mkdir(parents=True, exist_ok=False)
    raw_path = inflated / "0.raw"
    cleanup_certified = False
    try:
        decode_started = time.time()
        decode_receipt = decode_counted_archive(
            archive=archive,
            base_decoder=decoder,
            scratch_root=run_dir / "decode_scratch",
            pair_cap=LOGICAL_PAIR_COUNT,
            output_raw=raw_path,
            workers=args.decode_workers,
        )
        decode_seconds = time.time() - decode_started
        score_started = time.time()
        scorer_row = _score_raw_cpu(
            raw_path=raw_path,
            upstream=upstream,
            batch_size=args.batch_size,
            cpu_threads=args.cpu_threads,
            seed=args.seed,
        )
        score_seconds = time.time() - score_started
        score = compose_score(
            d_seg=scorer_row["d_seg"],
            d_pose=scorer_row["d_pose"],
            archive_bytes=parsed.archive_bytes,
        )
        gates = gate_summary(
            archive_bytes=parsed.archive_bytes, d_seg=scorer_row["d_seg"]
        )
        receipt = {
            "schema": SCHEMA,
            "verdict": "MEASURED_N600_PRODUCTION_RECEIVER_CONTROL_R1B_CANDIDATE_ABSENT",
            "verdict_scope": (
                "exact existing C2 receiver control only; no counted boundary-coordinate packet, "
                "no R1b candidate, no family negative"
            ),
            "authority": {
                "axis": "[macOS-CPU advisory]",
                "hard_cpu_torch": True,
                "through_r": True,
                "batch_geometry": "upstream_AVVideoDataset_vs_TensorVideoDataset_batch16",
                "contest_score_claim": False,
                "pointer_mutation": False,
            },
            "candidate": {
                "role": "production_receiver_control_no_r1b_packet",
                "boundary_coordinate_packet_present": False,
                "r1b_candidate_measured": False,
                "pdw2_conditioning_counted": True,
                "pose_custody": _base_packet_pose_custody(parsed.base_packet),
            },
            "row": {
                "archive_bytes": parsed.archive_bytes,
                "archive_sha256": parsed.archive_sha256,
                **scorer_row,
                **score,
            },
            "gates": gates,
            "archive": archive_receipt(parsed),
            "decode": decode_receipt,
            "runtime": {
                "decode_workers": args.decode_workers,
                "decode_seconds": decode_seconds,
                "hard_score_seconds": score_seconds,
                "total_seconds": time.time() - started,
                "python": sys.version,
                "platform": platform.platform(),
                "argv": sys.argv,
            },
            "custody": {
                "git": _git_custody(),
                "base_decoder": {"path": str(decoder), "sha256": sha256_file(decoder)},
                "upstream": str(upstream),
                "scorer_hashes": scorer_hashes,
                "source_video": {
                    "path": str(source_video),
                    "bytes": source_bytes,
                    "sha256": sha256_file(source_video),
                },
                "decoded_raw": {
                    "original_path": str(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "sha256": decode_receipt["decoded_raw_sha256"],
                    "rebuild_command": sys.argv,
                    "rebuildable": True,
                    "cleanup_reason": (
                        "success-only deterministic decode scratch; archive, decoder, source, argv, "
                        "environment, bytes, and hashes retained in this receipt"
                    ),
                },
            },
            "blockers": [
                "R1B_N600_COUNTED_BOUNDARY_PACKET_ABSENT_FROM_PRODUCTION_ARCHIVE",
                "R1B_FIRST_ORDER_SECANT_QP_JACOBIAN_AND_NULL_PROJECTOR_CUSTODY_ABSENT",
                "R1B_XI_POSE_RECEIVER_ABSENT_FROM_THIS_CONTROL_ARCHIVE"
                if not _base_packet_pose_custody(parsed.base_packet)["xi_receiver_present"]
                else "R1B_XI_BYTES_NOT_JOINTLY_CHARGED_WITH_BOUNDARY_PACKET",
            ],
            "next_coordinate": (
                "land the full-real-linear null projector, derive batch16 winner-rival first-order "
                "plus realized secant Jacobians on the production C2 scorer planes, compile the "
                "selected localized coefficients and xi into one strict archive, then rerun this n600 row"
            ),
            "storage_preflight": preflight,
            "scratch_cleanup": {
                "path": str(run_dir),
                "policy": "delete only after atomic receipt fsync",
                "certified_rebuildable": True,
            },
            "pointer": "0.19108 [contest-CPU] UNMOVED",
        }
        _atomic_json(output, receipt)
        cleanup_certified = True
    finally:
        if cleanup_certified:
            shutil.rmtree(run_dir)
        elif run_dir.exists():
            print(
                f"BLOCKER: uncertified scratch retained after failure: {run_dir}",
                file=sys.stderr,
            )
    print(json.dumps({"receipt": str(output), "verdict": receipt["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
