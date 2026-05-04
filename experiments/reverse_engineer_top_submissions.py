#!/usr/bin/env python3
"""Canonical reverse-engineering anatomy for public top submissions.

The output is deterministic for a fixed pair of PR checkouts: no timestamps,
no host paths, and no local temp directories.  It records byte custody,
container segmentation, and QZS3 decode validation without making score
claims for our own archives.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import torch

from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import decode_qzs3_state_dict

REPO_URL = "https://github.com/commaai/comma_video_compression_challenge.git"


@dataclass(frozen=True)
class PullSpec:
    label: str
    pr_number: int
    expected_commit: str
    submission_dir: str
    archive_name: str
    expected_archive_sha256: str
    expected_archive_bytes: int
    container_member: str
    segmentation: str


PR_SPECS = (
    PullSpec(
        label="pr67_qpose14_qzs3_filmq9g_slsb1_r55",
        pr_number=67,
        expected_commit="696d4a1e64a7f2d9aada3e3833be3c91ad394c21",
        submission_dir="submissions/qpose14_qzs3_filmq9g_slsb1_r55",
        archive_name="archive.zip",
        expected_archive_sha256="a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765",
        expected_archive_bytes=276564,
        container_member="p",
        segmentation="pr67_fixed_qpose14",
    ),
    PullSpec(
        label="pr65_henosis_qz_n3z_r25_clean",
        pr_number=65,
        expected_commit="a8b53b5280ee8f05db65740cd48cf7c321a55497",
        submission_dir="submissions/henosis_qz_n3z_r25_clean",
        archive_name="archive.zip",
        expected_archive_sha256="b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68",
        expected_archive_bytes=284425,
        container_member="x",
        segmentation="pr65_24bit_length_table",
    ),
)


@dataclass(frozen=True)
class ExternalArchiveSpec:
    label: str
    pr_number: int
    expected_commit: str
    archive_url: str
    expected_archive_sha256: str
    expected_archive_bytes: int
    container_member: str
    segmentation: str
    reported_score: float
    reported_posenet_dist: float
    reported_segnet_dist: float


CURRENT_FLOOR_SPECS = (
    ExternalArchiveSpec(
        label="pr63_qpose14",
        pr_number=63,
        expected_commit="17a3474eb2a7bbd648b850ec6c8338ef8ba15a65",
        archive_url="https://github.com/user-attachments/files/27206870/archive.zip",
        expected_archive_sha256="e012ebeffcc1e1655f4d674d0a779c1bf4cd41cfa82746de2ff6f73692e82a66",
        expected_archive_bytes=287_573,
        container_member="p",
        segmentation="pr63_fixed_qpose14",
        reported_score=0.32,
        reported_posenet_dist=0.00052154,
        reported_segnet_dist=0.00061261,
    ),
    ExternalArchiveSpec(
        label="pr64_unified_brotli",
        pr_number=64,
        expected_commit="8e5437d923d664d88382a93be8110a25b1966348",
        archive_url="https://github.com/avocardio/comma_video_compression_challenge/releases/download/v1-unified_brotli/archive.zip",
        expected_archive_sha256="7e48da0be75f915d6a4cf76a4679f8c1fbe689f82d69c7559f6ba1c2cb1e981d",
        expected_archive_bytes=287_165,
        container_member="p",
        segmentation="pr64_single_brotli_len_table",
        reported_score=0.33,
        reported_posenet_dist=0.00061622,
        reported_segnet_dist=0.00061261,
    ),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def _fetch_pr(spec: PullSpec, root: Path, *, repo_url: str) -> Path:
    checkout = root / f"pr{spec.pr_number}"
    checkout.mkdir(parents=True, exist_ok=False)
    _run(["git", "init", "-q"], cwd=checkout)
    _run(["git", "remote", "add", "origin", repo_url], cwd=checkout)
    _run(["git", "fetch", "--depth=1", "origin", f"refs/pull/{spec.pr_number}/head"], cwd=checkout)
    observed = _run(["git", "rev-parse", "FETCH_HEAD"], cwd=checkout)
    if observed != spec.expected_commit:
        raise ValueError(
            f"PR #{spec.pr_number} commit drift: expected {spec.expected_commit}, got {observed}"
        )
    _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=checkout)
    return checkout


def _verify_pr_head(spec: ExternalArchiveSpec, *, repo_url: str) -> str:
    stdout = _run(["git", "ls-remote", repo_url, f"refs/pull/{spec.pr_number}/head"])
    observed = stdout.split()[0] if stdout.split() else ""
    if observed != spec.expected_commit:
        raise ValueError(
            f"PR #{spec.pr_number} commit drift: expected "
            f"{spec.expected_commit}, got {observed}"
        )
    return observed


def _download_external_archive(spec: ExternalArchiveSpec, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{spec.label}_archive.zip"
    with urllib.request.urlopen(spec.archive_url, timeout=60) as response:
        data = response.read()
    archive.write_bytes(data)
    actual_bytes = archive.stat().st_size
    actual_sha = _sha256_path(archive)
    if actual_bytes != spec.expected_archive_bytes:
        raise ValueError(
            f"{spec.label} archive byte drift: expected "
            f"{spec.expected_archive_bytes}, got {actual_bytes}"
        )
    if actual_sha != spec.expected_archive_sha256:
        raise ValueError(
            f"{spec.label} archive SHA drift: expected "
            f"{spec.expected_archive_sha256}, got {actual_sha}"
        )
    return archive


def _git_commit(checkout: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=checkout)


def _zip_member_anatomy(archive: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    members: dict[str, bytes] = {}
    anatomy: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename)
            if not info.filename or info.filename.startswith("/") or ".." in path.parts:
                raise ValueError(f"unsafe zip member: {info.filename!r}")
            data = zf.read(info)
            members[info.filename] = data
            anatomy.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "sha256": _sha256_bytes(data),
                }
            )
    return anatomy, members


def _pr67_segments(payload: bytes) -> dict[str, Any]:
    mask_len = 219_472
    if 276_430 <= len(payload) <= 276_470:
        model_len = 56_093
    elif 276_550 <= len(payload) <= 276_610:
        model_len = 56_221
    elif 278_100 <= len(payload) <= 278_130:
        model_len = 57_757
    elif 277_400 <= len(payload) <= 277_430:
        model_len = 57_053
    elif 277_350 <= len(payload) <= 277_399:
        model_len = 57_031
    elif len(payload) == 281_240:
        model_len = 60_880
    else:
        model_len = 61_147
    pieces = [
        ("mask_obu_br", payload[:mask_len]),
        ("model_qzs3_br", payload[mask_len:mask_len + model_len]),
        ("pose_qp1_br", payload[mask_len + model_len:]),
    ]
    decoded: list[dict[str, Any]] = []
    qzs3_payload = b""
    for name, data in pieces:
        raw = brotli.decompress(data)
        if name == "model_qzs3_br":
            qzs3_payload = raw
        decoded.append(
            {
                "name": name,
                "compressed_bytes": len(data),
                "compressed_sha256": _sha256_bytes(data),
                "raw_bytes": len(raw),
                "raw_sha256": _sha256_bytes(raw),
                "raw_head_hex": raw[:16].hex(),
            }
        )
    state = decode_qzs3_state_dict(qzs3_payload, device="cpu")
    return {
        "fixed_segment_lengths": {
            "mask_obu_br": mask_len,
            "model_qzs3_br": model_len,
            "pose_qp1_br": len(payload) - mask_len - model_len,
        },
        "decoded_segments": decoded,
        "qzs3_decode_validation": {
            "state_dict_key_count": len(state),
            "parameter_count": int(sum(t.numel() for t in state.values())),
            "first_key": next(iter(state)),
            "last_key": next(reversed(state)),
            "all_tensors_finite": all(bool(t.isfinite().all().item()) for t in state.values() if t.is_floating_point()),
        },
    }


def _pr65_segments(payload: bytes) -> dict[str, Any]:
    lengths: list[int] = []
    offset = 0
    while offset + 3 <= len(payload) and len(lengths) < 32:
        lengths.append(int.from_bytes(payload[offset:offset + 3], "little"))
        offset += 3
    return {"first_32_24bit_lengths": lengths}


def _torch_quantized_payload_summary(raw: bytes) -> dict[str, Any]:
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {"top_keys": [type(payload).__name__]}
    quantized = payload.get("quantized", {})
    dense_fp16 = payload.get("dense_fp16", {})
    if not isinstance(quantized, dict):
        quantized = {}
    if not isinstance(dense_fp16, dict):
        dense_fp16 = {}

    weight_kind_counts: dict[str, int] = {}
    fp4_module_count = 0
    dense_weight_module_count = 0
    parameter_like_count = 0
    for rec in quantized.values():
        if not isinstance(rec, dict):
            continue
        kind = str(rec.get("weight_kind"))
        weight_kind_counts[kind] = weight_kind_counts.get(kind, 0) + 1
        if kind == "fp4_packed":
            fp4_module_count += 1
            shape = rec.get("weight_shape")
            if shape is not None:
                count = 1
                for dim in shape:
                    count *= int(dim)
                parameter_like_count += count
        elif rec.get("weight_fp16") is not None:
            dense_weight_module_count += 1
            parameter_like_count += int(rec["weight_fp16"].numel())
        if rec.get("bias_fp16") is not None:
            parameter_like_count += int(rec["bias_fp16"].numel())
    for tensor in dense_fp16.values():
        if hasattr(tensor, "numel"):
            parameter_like_count += int(tensor.numel())

    return {
        "top_keys": sorted(str(key) for key in payload),
        "quantized_module_count": len(quantized),
        "dense_fp16_key_count": len(dense_fp16),
        "weight_kind_counts": weight_kind_counts,
        "fp4_module_count": fp4_module_count,
        "dense_weight_module_count": dense_weight_module_count,
        "parameter_like_count": parameter_like_count,
        "first_quantized_key": next(iter(quantized), None),
        "last_quantized_key": next(reversed(quantized), None) if quantized else None,
    }


def _pr63_segments(payload: bytes) -> dict[str, Any]:
    mask_len = 219_472
    model_len = 66_841
    pieces = [
        ("mask_obu_br", payload[:mask_len]),
        ("model_torch_quantized_br", payload[mask_len:mask_len + model_len]),
        ("pose_qpose14_uint16_br", payload[mask_len + model_len:]),
    ]
    decoded: list[dict[str, Any]] = []
    for name, data in pieces:
        raw = brotli.decompress(data)
        record: dict[str, Any] = {
            "name": name,
            "compressed_bytes": len(data),
            "compressed_sha256": _sha256_bytes(data),
            "raw_bytes": len(raw),
            "raw_sha256": _sha256_bytes(raw),
            "raw_head_hex": raw[:16].hex(),
        }
        if name == "model_torch_quantized_br":
            record["torch_payload"] = _torch_quantized_payload_summary(raw)
        elif name == "pose_qpose14_uint16_br":
            values = struct.unpack("<" + "H" * (len(raw) // 2), raw)
            record["pose_shape_inferred"] = [len(values) // 6, 6]
            record["velocity_word_min"] = int(min(values[0::6]))
            record["velocity_word_max"] = int(max(values[0::6]))
        decoded.append(record)
    return {
        "fixed_segment_lengths": {
            "mask_obu_br": mask_len,
            "model_torch_quantized_br": model_len,
            "pose_qpose14_uint16_br": len(payload) - mask_len - model_len,
        },
        "decoded_segments": decoded,
    }


def _pr64_segments(payload: bytes) -> dict[str, Any]:
    raw = brotli.decompress(payload)
    if len(raw) < 12:
        raise ValueError("PR64 payload too short for length table")
    n_mask, n_model, n_pose = struct.unpack_from("<III", raw, 0)
    offset = 12
    pieces: list[tuple[str, bytes]] = []
    for name, n_bytes in (
        ("mask_obu", n_mask),
        ("model_torch_quantized", n_model),
        ("pose_velocity_delta_uint16_int16", n_pose),
    ):
        end = offset + n_bytes
        if end > len(raw):
            raise ValueError(f"PR64 segment {name} overruns payload")
        pieces.append((name, raw[offset:end]))
        offset = end
    if offset != len(raw):
        raise ValueError(f"PR64 payload has {len(raw) - offset} trailing bytes")

    decoded: list[dict[str, Any]] = []
    for name, data in pieces:
        record: dict[str, Any] = {
            "name": name,
            "raw_bytes": len(data),
            "raw_sha256": _sha256_bytes(data),
            "raw_head_hex": data[:16].hex(),
        }
        if name == "model_torch_quantized":
            record["torch_payload"] = _torch_quantized_payload_summary(data)
        elif name == "pose_velocity_delta_uint16_int16":
            if len(data) < 2 or (len(data) - 2) % 2:
                raise ValueError("PR64 velocity delta payload has invalid length")
            velocity_anchor = struct.unpack_from("<H", data, 0)[0]
            deltas = struct.unpack("<" + "h" * ((len(data) - 2) // 2), data[2:])
            record.update(
                {
                    "velocity_anchor_uint16": int(velocity_anchor),
                    "delta_count": len(deltas),
                    "delta_min": int(min(deltas)) if deltas else 0,
                    "delta_max": int(max(deltas)) if deltas else 0,
                }
            )
        decoded.append(record)

    return {
        "payload_compressed_bytes": len(payload),
        "payload_raw_bytes": len(raw),
        "length_table": {
            "mask_obu": n_mask,
            "model_torch_quantized": n_model,
            "pose_velocity_delta_uint16_int16": n_pose,
        },
        "decoded_segments": decoded,
    }


def inspect_checkout(spec: PullSpec, checkout: Path) -> dict[str, Any]:
    commit = _git_commit(checkout)
    if commit != spec.expected_commit:
        raise ValueError(
            f"PR #{spec.pr_number} checkout drift: expected {spec.expected_commit}, got {commit}"
        )
    archive = checkout / spec.submission_dir / spec.archive_name
    if not archive.exists():
        raise FileNotFoundError(f"archive not found for {spec.label}: {archive}")
    archive_bytes = archive.stat().st_size
    archive_sha = _sha256_path(archive)
    if archive_bytes != spec.expected_archive_bytes:
        raise ValueError(
            f"{spec.label} archive byte drift: expected {spec.expected_archive_bytes}, got {archive_bytes}"
        )
    if archive_sha != spec.expected_archive_sha256:
        raise ValueError(
            f"{spec.label} archive SHA drift: expected {spec.expected_archive_sha256}, got {archive_sha}"
        )
    zip_members, member_data = _zip_member_anatomy(archive)
    if spec.container_member not in member_data:
        raise ValueError(f"{spec.label} missing container member {spec.container_member!r}")
    payload = member_data[spec.container_member]
    result: dict[str, Any] = {
        "label": spec.label,
        "pr_number": spec.pr_number,
        "source": {
            "repo_url": REPO_URL,
            "pull_ref": f"refs/pull/{spec.pr_number}/head",
            "commit": commit,
            "submission_dir": spec.submission_dir,
            "archive_name": spec.archive_name,
        },
        "archive": {
            "bytes": archive_bytes,
            "sha256": archive_sha,
        },
        "zip_members": zip_members,
        "container": {
            "member": spec.container_member,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "head_hex": payload[:16].hex(),
            "segmentation": spec.segmentation,
        },
    }
    if spec.segmentation == "pr67_fixed_qpose14":
        result["container"].update(_pr67_segments(payload))
    elif spec.segmentation == "pr65_24bit_length_table":
        result["container"].update(_pr65_segments(payload))
    else:  # pragma: no cover - specs are static
        raise AssertionError(spec.segmentation)
    return result


def inspect_external_archive(
    spec: ExternalArchiveSpec,
    archive: Path,
    *,
    observed_commit: str,
) -> dict[str, Any]:
    archive_bytes = archive.stat().st_size
    archive_sha = _sha256_path(archive)
    if archive_bytes != spec.expected_archive_bytes:
        raise ValueError(
            f"{spec.label} archive byte drift: expected "
            f"{spec.expected_archive_bytes}, got {archive_bytes}"
        )
    if archive_sha != spec.expected_archive_sha256:
        raise ValueError(
            f"{spec.label} archive SHA drift: expected "
            f"{spec.expected_archive_sha256}, got {archive_sha}"
        )
    zip_members, member_data = _zip_member_anatomy(archive)
    if spec.container_member not in member_data:
        raise ValueError(f"{spec.label} missing container member {spec.container_member!r}")
    payload = member_data[spec.container_member]
    result: dict[str, Any] = {
        "label": spec.label,
        "pr_number": spec.pr_number,
        "source": {
            "repo_url": REPO_URL,
            "pull_ref": f"refs/pull/{spec.pr_number}/head",
            "commit": observed_commit,
            "archive_url": spec.archive_url,
        },
        "reported_score": {
            "score": spec.reported_score,
            "posenet_dist": spec.reported_posenet_dist,
            "segnet_dist": spec.reported_segnet_dist,
            "archive_size_bytes": spec.expected_archive_bytes,
        },
        "archive": {
            "bytes": archive_bytes,
            "sha256": archive_sha,
        },
        "zip_members": zip_members,
        "container": {
            "member": spec.container_member,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "head_hex": payload[:16].hex(),
            "segmentation": spec.segmentation,
        },
    }
    if spec.segmentation == "pr63_fixed_qpose14":
        result["container"].update(_pr63_segments(payload))
    elif spec.segmentation == "pr64_single_brotli_len_table":
        result["container"].update(_pr64_segments(payload))
    else:  # pragma: no cover - specs are static
        raise AssertionError(spec.segmentation)
    return result


def build_report(
    checkouts: dict[int, Path],
    *,
    current_floor_archives: dict[int, Path],
    current_floor_commits: dict[int, str],
) -> dict[str, Any]:
    template = build_quantizr_faithful_renderer()
    return {
        "schema_version": 2,
        "tool": "experiments/reverse_engineer_top_submissions.py",
        "score_claim": False,
        "evidence_grade": "external_plus_empirical_byte_anatomy",
        "determinism": {
            "no_timestamps": True,
            "no_host_paths": True,
            "pinned_pr_commits": True,
            "expected_archive_hashes_enforced": True,
        },
        "local_jointframegenerator_reference": {
            "module": "src/tac/quantizr_faithful_renderer.py",
            "state_dict_key_count": len(template.state_dict()),
            "parameter_count": int(sum(p.numel() for p in template.parameters())),
        },
        "items": [
            inspect_checkout(spec, checkouts[spec.pr_number])
            for spec in PR_SPECS
        ],
        "current_floor_items": (
            [
                inspect_external_archive(
                    spec,
                    current_floor_archives[spec.pr_number],
                    observed_commit=current_floor_commits[spec.pr_number],
                )
                for spec in CURRENT_FLOOR_SPECS
            ]
            if current_floor_archives
            else []
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--pr65-dir", type=Path, default=None)
    parser.add_argument("--pr67-dir", type=Path, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--current-floor-archive-dir",
        type=Path,
        default=None,
        help="Optional directory containing predownloaded current-floor archives "
        "named pr63_qpose14_archive.zip and pr64_unified_brotli_archive.zip.",
    )
    parser.add_argument(
        "--skip-current-floor",
        action="store_true",
        help="Only inspect the pinned PR65/PR67 in-repo archives.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit = {65: args.pr65_dir, 67: args.pr67_dir}
    if any(explicit.values()) and not all(explicit.values()):
        raise SystemExit("--pr65-dir and --pr67-dir must be supplied together")

    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    owned_work_dir = False
    if all(explicit.values()):
        checkouts = {pr: path.resolve() for pr, path in explicit.items() if path is not None}
    else:
        if args.work_dir is None:
            temp_ctx = tempfile.TemporaryDirectory(prefix="pact_topsubs_")
            work_root = Path(temp_ctx.name)
            owned_work_dir = True
        else:
            work_root = args.work_dir.resolve()
            if work_root.exists():
                raise SystemExit(f"--work-dir already exists: {work_root}")
            work_root.mkdir(parents=True)
            owned_work_dir = not args.keep_work_dir
        checkouts = {
            spec.pr_number: _fetch_pr(spec, work_root, repo_url=args.repo_url)
            for spec in PR_SPECS
        }

    try:
        current_floor_archives: dict[int, Path] = {}
        current_floor_commits: dict[int, str] = {}
        if not args.skip_current_floor:
            if args.current_floor_archive_dir is None:
                floor_root = Path(tempfile.mkdtemp(prefix="pact_current_floor_"))
                cleanup_floor_root = True
            else:
                floor_root = args.current_floor_archive_dir.resolve()
                cleanup_floor_root = False
            try:
                for spec in CURRENT_FLOOR_SPECS:
                    current_floor_commits[spec.pr_number] = _verify_pr_head(
                        spec,
                        repo_url=args.repo_url,
                    )
                    if args.current_floor_archive_dir is None:
                        archive = _download_external_archive(spec, floor_root)
                    else:
                        archive = floor_root / f"{spec.label}_archive.zip"
                        if not archive.exists():
                            raise FileNotFoundError(
                                f"missing predownloaded current-floor archive: {archive}"
                            )
                    current_floor_archives[spec.pr_number] = archive
                report = build_report(
                    checkouts,
                    current_floor_archives=current_floor_archives,
                    current_floor_commits=current_floor_commits,
                )
            finally:
                if cleanup_floor_root:
                    shutil.rmtree(floor_root)
        else:
            report = build_report(
                checkouts,
                current_floor_archives={},
                current_floor_commits={},
            )
            report["current_floor_items"] = []
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "archive_anatomy.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "wrote": str(out),
                    "items": len(report["items"]),
                    "current_floor_items": len(report["current_floor_items"]),
                },
                sort_keys=True,
            )
        )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
        elif owned_work_dir and args.work_dir is not None and args.work_dir.exists():
            shutil.rmtree(args.work_dir)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
