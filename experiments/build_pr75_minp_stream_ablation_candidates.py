#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic PR75-minp stream ablation archives.

This is a local candidate builder only.  It mixes decoded/runtime-equivalent
streams from the current C089 frontier and the public PR75 minp archive, emits
stored ZIP archives, and validates each payload through the robust unpacker.
The outputs are byte/component hypotheses only until exact CUDA auth eval runs
on the exact archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_C089_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_PUBLIC_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr75_minp_stream_ablation_20260503_codex"
TOOL = "experiments/build_pr75_minp_stream_ablation_candidates.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
BASELINE_SCORE = 0.3154707273953505  # [external: PR-75 contest-CUDA T4 anchor (== PR-65 frontier)]
BASELINE_BYTES = 276_342


@dataclass(frozen=True)
class EncodedStreams:
    mask_br: bytes
    renderer_br: bytes
    actions_br: bytes
    pose_br: bytes


@dataclass(frozen=True)
class SourceArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_format: str
    encoded: EncodedStreams
    decoded: dict[str, bytes]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("pr75_minp_ablation_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_single_p(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if names != ["p"]:
            raise ValueError(f"{path} must contain exactly member 'p'; got {names!r}")
        return zf.read("p")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("p"), payload)


def _parse_encoded_streams(payload: bytes) -> EncodedStreams:
    if payload.startswith(b"P3"):
        cursor = 2 + struct.calcsize("<IHH")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
    elif payload.startswith(b"P6"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, renderer_len, actions_len, _record_count = struct.unpack_from(
            "<IHHH",
            payload,
            2,
        )
    elif len(payload) == 276_381:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_756, 255
    elif len(payload) == 276_379:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_756, 253
    elif len(payload) == 276_520:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_914, 236
    elif len(payload) == 276_641:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 56_034, 236
    else:
        raise ValueError(f"unsupported PR75 ablation payload form: prefix={payload[:4]!r} len={len(payload)}")
    if min(mask_len, renderer_len, actions_len) <= 0:
        raise ValueError("empty encoded stream in source payload")
    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    actions_end = renderer_end + actions_len
    if actions_end >= len(payload):
        raise ValueError("encoded stream lengths leave no pose payload")
    return EncodedStreams(
        mask_br=payload[cursor:mask_end],
        renderer_br=payload[mask_end:renderer_end],
        actions_br=payload[renderer_end:actions_end],
        pose_br=payload[actions_end:],
    )


def _best_brotli(raw: bytes, *, source: bytes | None = None) -> bytes:
    candidates = [source] if source is not None else []
    for quality, mode, lgwin, lgblock in (
        (11, 0, 24, 0),
        (11, 0, 19, 17),
        (9, 0, 10, 0),
        (7, 0, 10, 0),
        (5, 0, 10, 0),
    ):
        candidates.append(
            brotli.compress(raw, quality=quality, mode=mode, lgwin=lgwin, lgblock=lgblock)
        )
    best = min((item for item in candidates if item is not None), key=lambda item: len(item))
    if brotli.decompress(best) != raw:
        raise ValueError("selected Brotli stream failed round-trip")
    return best


def _build_p3_payload(streams: EncodedStreams) -> bytes:
    if len(streams.mask_br) > 0xFFFF_FFFF:
        raise ValueError("mask stream too large for P3")
    for name, data in (
        ("renderer", streams.renderer_br),
        ("actions", streams.actions_br),
    ):
        if len(data) > 0xFFFF:
            raise ValueError(f"{name} stream too large for P3")
    return (
        b"P3"
        + struct.pack("<IHH", len(streams.mask_br), len(streams.renderer_br), len(streams.actions_br))
        + streams.mask_br
        + streams.renderer_br
        + streams.actions_br
        + streams.pose_br
    )


def _load_source(label: str, path: Path, unpacker: Any) -> SourceArchive:
    payload = _read_single_p(path)
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    encoded = _parse_encoded_streams(payload)
    return SourceArchive(
        label=label,
        path=path.resolve(),
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_format=str(header.get("payload_format")),
        encoded=encoded,
        decoded=decoded,
    )


def _compressed_action_stream(source: SourceArchive) -> bytes:
    raw = source.decoded["seg_tile_actions.bin"]
    if source.label == "public_pr75_minp":
        # The observed public SG2 stream is already smaller than raw-record
        # Brotli while decoding to the same runtime record table.
        return source.encoded.actions_br
    return _best_brotli(raw)


def _validate_candidate(payload: bytes, expected_decoded: dict[str, bytes], unpacker: Any) -> dict[str, Any]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    if set(decoded) != set(expected_decoded):
        raise ValueError(f"decoded keys changed: {sorted(decoded)} vs {sorted(expected_decoded)}")
    for name, expected in expected_decoded.items():
        if decoded[name] != expected:
            raise ValueError(
                f"decoded member {name} mismatch: expected={_sha256_bytes(expected)} actual={_sha256_bytes(decoded[name])}"
            )
    return {
        "payload_format": str(header.get("payload_format")),
        "members": header.get("members", []),
    }


def build_candidates(
    *,
    c089_archive: Path,
    public_archive: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    unpacker = _load_unpacker()
    c089 = _load_source("c089", c089_archive, unpacker)
    public = _load_source("public_pr75_minp", public_archive, unpacker)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_br = {
        "c089": _compressed_action_stream(c089),
        "public": _compressed_action_stream(public),
    }
    sources = {"c089": c089, "public": public}
    plans = [
        ("c089_p3_repack", "c089", "c089", "c089"),
        ("public_actions_only", "c089", "c089", "public"),
        ("public_pose_only", "c089", "public", "c089"),
        ("public_renderer_only", "public", "c089", "c089"),
        ("public_actions_pose", "c089", "public", "public"),
        ("public_renderer_actions", "public", "c089", "public"),
        ("public_renderer_pose", "public", "public", "c089"),
        ("public_all_p3", "public", "public", "public"),
    ]
    rows: list[dict[str, Any]] = []
    for candidate_id, renderer_source, pose_source, action_source in plans:
        mask_source = "c089"
        streams = EncodedStreams(
            mask_br=sources[mask_source].encoded.mask_br,
            renderer_br=sources[renderer_source].encoded.renderer_br,
            actions_br=action_br[action_source],
            pose_br=sources[pose_source].encoded.pose_br,
        )
        expected_decoded = {
            "masks.mkv": sources[mask_source].decoded["masks.mkv"],
            "renderer.bin": sources[renderer_source].decoded["renderer.bin"],
            "seg_tile_actions.bin": sources[action_source].decoded["seg_tile_actions.bin"],
            "optimized_poses.qp1": sources[pose_source].decoded["optimized_poses.qp1"],
        }
        payload = _build_p3_payload(streams)
        validation = _validate_candidate(payload, expected_decoded, unpacker)
        candidate_dir = output_dir / candidate_id
        archive_path = candidate_dir / "archive.zip"
        if archive_path.exists() and not force:
            raise FileExistsError(f"{archive_path} exists; pass --force")
        _write_archive(archive_path, payload)
        archive_bytes = archive_path.stat().st_size
        manifest = {
            "candidate_id": candidate_id,
            "canonical_score_source_required": "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
            "evidence_grade": "empirical_stream_ablation_until_exact_cuda",
            "output_archive": {
                "bytes": archive_bytes,
                "path": str(archive_path),
                "sha256": _sha256_file(archive_path),
            },
            "payload": {
                "bytes": len(payload),
                "format": validation["payload_format"],
                "sha256": _sha256_bytes(payload),
            },
            "promotion_eligible": False,
            "runtime_parse_validation": validation,
            "score_claim": False,
            "selected_sources": {
                "mask": mask_source,
                "renderer": renderer_source,
                "actions": action_source,
                "pose": pose_source,
            },
            "stream_bytes": {
                "mask_br": len(streams.mask_br),
                "renderer_br": len(streams.renderer_br),
                "actions_br": len(streams.actions_br),
                "pose_br": len(streams.pose_br),
            },
            "decoded_members": {
                name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
                for name, data in sorted(expected_decoded.items())
            },
            "delta_vs_c089": {
                "archive_bytes": archive_bytes - c089.archive_bytes,
                "formula_only_rate_score": (archive_bytes - c089.archive_bytes) * RATE_SCORE_PER_BYTE,
                "baseline_score_if_components_unchanged": BASELINE_SCORE
                + (archive_bytes - BASELINE_BYTES) * RATE_SCORE_PER_BYTE,
            },
            "schema": "pr75_minp_stream_ablation_manifest_v1",
            "tool": TOOL,
        }
        manifest_path = candidate_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        rows.append(
            {
                "archive_bytes": archive_bytes,
                "archive_path": str(archive_path),
                "archive_sha256": manifest["output_archive"]["sha256"],
                "candidate_id": candidate_id,
                "delta_bytes_vs_c089": archive_bytes - c089.archive_bytes,
                "formula_only_rate_score_delta_vs_c089": manifest["delta_vs_c089"][
                    "formula_only_rate_score"
                ],
                "manifest_path": str(manifest_path),
                "payload_bytes": len(payload),
                "selected_sources": manifest["selected_sources"],
                "stream_bytes": manifest["stream_bytes"],
            }
        )

    summary = {
        "candidates": sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"])),
        "c089_archive": {
            "bytes": c089.archive_bytes,
            "path": str(c089.path),
            "payload_format": c089.payload_format,
            "sha256": c089.archive_sha256,
        },
        "public_archive": {
            "bytes": public.archive_bytes,
            "path": str(public.path),
            "payload_format": public.payload_format,
            "sha256": public.archive_sha256,
        },
        "evidence_grade": "empirical_stream_ablation_until_exact_cuda",
        "promotion_eligible": False,
        "schema": "pr75_minp_stream_ablation_matrix_v1",
        "score_claim": False,
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c089-archive", type=Path, default=DEFAULT_C089_ARCHIVE)
    parser.add_argument("--public-archive", type=Path, default=DEFAULT_PUBLIC_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_candidates(
        c089_archive=args.c089_archive,
        public_archive=args.public_archive,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "candidate_count": len(summary["candidates"]),
                "best_by_bytes": summary["candidates"][0],
                "output_dir": str(Path(args.output_dir).resolve()),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
