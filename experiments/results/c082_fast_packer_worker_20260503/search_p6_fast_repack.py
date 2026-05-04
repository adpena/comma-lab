#!/usr/bin/env python3
"""Local P6 stream-resweep screen for PR75/QZS3/QP1 archives.

This result-local tool only builds deterministic archive-byte candidates.  It
does not dispatch GPU work and every output remains non-promotable until exact
CUDA auth eval runs on the exact archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import brotli


REPO_ROOT = Path(__file__).resolve().parents[3]
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)


@dataclass(frozen=True)
class P6Slices:
    mask_br: bytes
    model_br: bytes
    actions_br: bytes
    pose_br: bytes
    record_count: int


@dataclass(frozen=True)
class BrotliChoice:
    data: bytes
    params: tuple[int, int, int, int] | None
    source_bytes: int

    @property
    def improved(self) -> bool:
        return len(self.data) < self.source_bytes

    def manifest(self) -> dict[str, Any]:
        params: dict[str, int] | str
        if self.params is None:
            params = "source"
        else:
            quality, mode, lgwin, lgblock = self.params
            params = {
                "quality": quality,
                "mode": mode,
                "lgwin": lgwin,
                "lgblock": lgblock,
            }
        return {
            "bytes": len(self.data),
            "delta_bytes_vs_source_stream": len(self.data) - self.source_bytes,
            "improved": self.improved,
            "params": params,
            "sha256": _sha256_bytes(self.data),
            "source_bytes": self.source_bytes,
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("c082_fast_packer_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_label(label: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not label or any(char not in allowed for char in label):
        raise ValueError(f"unsafe label {label!r}")
    return label


def _read_single_member_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain single member {MEMBER_NAME!r}; got {names!r}")
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path} member {MEMBER_NAME!r} must be ZIP_STORED")
        return zf.read(info)


def _zip_info(name: str, compress_type: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _zip_bytes(payload: bytes, *, compress_type: int = zipfile.ZIP_STORED) -> bytes:
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {}
    if compress_type == zipfile.ZIP_DEFLATED:
        kwargs["compresslevel"] = 9
    with zipfile.ZipFile(buffer, "w", **kwargs) as zf:
        zf.writestr(_zip_info(MEMBER_NAME, compress_type), payload)
    return buffer.getvalue()


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_zip_bytes(payload))


def _parse_p6_payload(payload: bytes) -> P6Slices:
    if not payload.startswith(b"P6"):
        raise ValueError(f"source payload must be P6; got {payload[:2]!r}")
    header_size = 2 + struct.calcsize("<IHHH")
    if len(payload) <= header_size:
        raise ValueError("P6 payload too short")
    mask_len, model_len, actions_len, record_count = struct.unpack_from("<IHHH", payload, 2)
    cursor = header_size
    if min(mask_len, model_len, actions_len, record_count) <= 0:
        raise ValueError("P6 payload contains empty required slices or record count")
    if cursor + mask_len + model_len + actions_len >= len(payload):
        raise ValueError("P6 payload slice lengths leave no pose stream")
    mask_br = payload[cursor : cursor + mask_len]
    cursor += mask_len
    model_br = payload[cursor : cursor + model_len]
    cursor += model_len
    actions_br = payload[cursor : cursor + actions_len]
    cursor += actions_len
    return P6Slices(
        mask_br=mask_br,
        model_br=model_br,
        actions_br=actions_br,
        pose_br=payload[cursor:],
        record_count=record_count,
    )


def _build_p6_payload(slices: P6Slices) -> bytes:
    return (
        b"P6"
        + struct.pack(
            "<IHHH",
            len(slices.mask_br),
            len(slices.model_br),
            len(slices.actions_br),
            slices.record_count,
        )
        + slices.mask_br
        + slices.model_br
        + slices.actions_br
        + slices.pose_br
    )


def _large_param_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    for lgwin in range(16, 25):
        for lgblock in (0, 16, 17, 18, 19, 20):
            if lgblock and lgblock > lgwin:
                continue
            params.append((11, 0, lgwin, lgblock))
    for quality in (10, 9):
        for lgwin in (16, 19, 22, 24):
            params.append((quality, 0, lgwin, 0))
    return list(dict.fromkeys(params))


def _small_param_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    for quality in range(11, -1, -1):
        for mode in (0, 1, 2):
            for lgwin in range(10, 25):
                for lgblock in (0, 16, 17, 18, 19, 20, 21, 22, 23, 24):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def _outer_param_grid() -> list[tuple[int, int, int, int]]:
    return [
        (quality, 0, lgwin, 0)
        for quality in (0, 1, 4, 9, 11)
        for lgwin in (18, 19, 22, 24)
    ]


def _compress(raw: bytes, params: tuple[int, int, int, int]) -> bytes:
    quality, mode, lgwin, lgblock = params
    return brotli.compress(raw, quality=quality, mode=mode, lgwin=lgwin, lgblock=lgblock)


def _best_brotli(
    *,
    raw: bytes,
    source: bytes,
    cache: dict[str, BrotliChoice],
) -> BrotliChoice:
    raw_sha = _sha256_bytes(raw)
    cached = cache.get(raw_sha)
    if cached is not None and len(cached.data) <= len(source):
        return BrotliChoice(cached.data, cached.params, len(source))

    params = _small_param_grid() if len(raw) <= 8192 else _large_param_grid()
    best = BrotliChoice(source, None, len(source))
    for param in params:
        candidate = _compress(raw, param)
        if len(candidate) < len(best.data):
            best = BrotliChoice(candidate, param, len(source))
    if brotli.decompress(best.data) != raw:
        raise ValueError("selected Brotli stream failed round-trip")
    cache[raw_sha] = BrotliChoice(best.data, best.params, len(best.data))
    return best


def _decoded_summary(decoded: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        for name, data in sorted(decoded.items())
    }


def _assert_decoded_parity(
    *,
    payload: bytes,
    source_decoded: dict[str, bytes],
    unpacker: Any,
) -> tuple[str, dict[str, bytes]]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    if set(decoded) != set(source_decoded):
        raise ValueError(
            "decoded member set changed: "
            f"source={sorted(source_decoded)} candidate={sorted(decoded)}"
        )
    for name, expected in source_decoded.items():
        if decoded[name] != expected:
            raise ValueError(
                f"decoded member {name} changed: "
                f"source={_sha256_bytes(expected)} candidate={_sha256_bytes(decoded[name])}"
            )
    return str(header.get("payload_format")), decoded


def _best_outer_brotli(payload: bytes) -> BrotliChoice:
    best = BrotliChoice(payload, None, len(payload))
    for param in _outer_param_grid():
        candidate = _compress(payload, param)
        if len(candidate) < len(best.data):
            best = BrotliChoice(candidate, param, len(payload))
    if best.improved and brotli.decompress(best.data) != payload:
        raise ValueError("outer Brotli candidate failed round-trip")
    return best


def _pose_safety_status(source_archive: Path) -> dict[str, Any] | None:
    path = source_archive.parent / "pose_safety_preflight.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return {
        "failure_class": payload.get("failure_class"),
        "path": str(path),
        "safe_for_exact_eval_dispatch": bool(payload.get("safe_for_exact_eval_dispatch")),
        "sha256": _sha256_file(path),
    }


def _build_manifest(
    *,
    label: str,
    source_archive: Path,
    source_payload: bytes,
    source_payload_format: str,
    source_decoded: dict[str, bytes],
    candidate_dir: Path,
    candidate_payload: bytes,
    stream_choices: dict[str, BrotliChoice],
    output_archive: Path,
    output_payload_format: str,
    pose_safety: dict[str, Any] | None,
    no_op: bool,
) -> dict[str, Any]:
    source_archive_bytes = source_archive.stat().st_size
    output_archive_bytes = output_archive.stat().st_size
    archive_delta = output_archive_bytes - source_archive_bytes
    manifest = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidate_id": candidate_dir.name,
        "decoded_stream_parity": True,
        "decoded_stream_parity_detail": {
            "members_compared": sorted(source_decoded),
            "source_decoded_members": _decoded_summary(source_decoded),
            "status": "passed",
        },
        "determinism": {
            "member_name": MEMBER_NAME,
            "zip_compress_type": "ZIP_STORED",
            "zip_permissions": "0644",
            "zip_timestamp": list(FIXED_ZIP_TIMESTAMP),
        },
        "evidence_grade": "empirical_lossless_byte_transform",
        "formula_only_rate_score_delta_vs_source": archive_delta * RATE_SCORE_PER_BYTE,
        "noop": no_op,
        "noop_status": "byte_identical_to_source_payload" if no_op else "not_noop_repacked_payload",
        "output_archive": {
            "bytes": output_archive_bytes,
            "path": str(output_archive),
            "sha256": _sha256_file(output_archive),
        },
        "payload": {
            "bytes": len(candidate_payload),
            "format": output_payload_format,
            "member": MEMBER_NAME,
            "sha256": _sha256_bytes(candidate_payload),
        },
        "pose_safety_preflight": pose_safety,
        "promotion_eligible": False,
        "schema": "c082_fast_p6_repack_manifest_v1",
        "score_claim": False,
        "source_archive": {
            "bytes": source_archive_bytes,
            "path": str(source_archive),
            "sha256": _sha256_file(source_archive),
        },
        "source_payload": {
            "bytes": len(source_payload),
            "format": source_payload_format,
            "sha256": _sha256_bytes(source_payload),
        },
        "source_preservation": {
            "candidate_payload_sha256": _sha256_bytes(candidate_payload),
            "decoded_streams_byte_identical": True,
            "payload_byte_identical_to_source": candidate_payload == source_payload,
            "source_payload_sha256": _sha256_bytes(source_payload),
            "status": (
                "byte_identical_noop"
                if candidate_payload == source_payload
                else "lossless_decoded_stream_preserving_repack"
            ),
        },
        "stream_choices": {
            name: choice.manifest()
            for name, choice in sorted(stream_choices.items())
        },
        "tool": str(Path(__file__).relative_to(REPO_ROOT)),
    }
    if not math.isfinite(float(manifest["formula_only_rate_score_delta_vs_source"])):
        raise ValueError("non-finite formula-only rate delta")
    return manifest


def screen_source(
    *,
    label: str,
    source_archive: Path,
    output_dir: Path,
    cache: dict[str, BrotliChoice],
    unpacker: Any,
) -> dict[str, Any]:
    label = _safe_label(label)
    source_archive = source_archive.resolve()
    source_payload = _read_single_member_payload(source_archive)
    source_header, source_decoded = unpacker._parse_payload(source_payload)  # noqa: SLF001
    source_payload_format = str(source_header.get("payload_format"))
    source_slices = _parse_p6_payload(source_payload)
    raw_streams = {
        "masks.mkv": brotli.decompress(source_slices.mask_br),
        "renderer.bin": brotli.decompress(source_slices.model_br),
        "seg_tile_actions.delta_varint": brotli.decompress(source_slices.actions_br),
        "optimized_poses.qp1": brotli.decompress(source_slices.pose_br),
    }
    choices = {
        "masks.mkv": _best_brotli(raw=raw_streams["masks.mkv"], source=source_slices.mask_br, cache=cache),
        "renderer.bin": _best_brotli(raw=raw_streams["renderer.bin"], source=source_slices.model_br, cache=cache),
        "seg_tile_actions.delta_varint": _best_brotli(
            raw=raw_streams["seg_tile_actions.delta_varint"],
            source=source_slices.actions_br,
            cache=cache,
        ),
        "optimized_poses.qp1": _best_brotli(
            raw=raw_streams["optimized_poses.qp1"],
            source=source_slices.pose_br,
            cache=cache,
        ),
    }
    candidate_slices = P6Slices(
        mask_br=choices["masks.mkv"].data,
        model_br=choices["renderer.bin"].data,
        actions_br=choices["seg_tile_actions.delta_varint"].data,
        pose_br=choices["optimized_poses.qp1"].data,
        record_count=source_slices.record_count,
    )
    candidate_payload = _build_p6_payload(candidate_slices)
    candidate_payload_format, _candidate_decoded = _assert_decoded_parity(
        payload=candidate_payload,
        source_decoded=source_decoded,
        unpacker=unpacker,
    )

    candidate_id = f"{label}_p6_stream_resweep"
    candidate_dir = output_dir / candidate_id
    output_archive = candidate_dir / "archive.zip"
    _write_archive(output_archive, candidate_payload)
    pose_safety = _pose_safety_status(source_archive)
    manifest = _build_manifest(
        label=label,
        source_archive=source_archive,
        source_payload=source_payload,
        source_payload_format=source_payload_format,
        source_decoded=source_decoded,
        candidate_dir=candidate_dir,
        candidate_payload=candidate_payload,
        stream_choices=choices,
        output_archive=output_archive,
        output_payload_format=candidate_payload_format,
        pose_safety=pose_safety,
        no_op=candidate_payload == source_payload,
    )
    _write_json(candidate_dir / "manifest.json", manifest)

    outer_choice = _best_outer_brotli(candidate_payload)
    zip_deflated_bytes = _zip_bytes(candidate_payload, compress_type=zipfile.ZIP_DEFLATED)
    negative_checks = {
        "outer_brotli": {
            "best_bytes": len(outer_choice.data),
            "delta_bytes_vs_stream_resweep_payload": len(outer_choice.data) - len(candidate_payload),
            "params": outer_choice.manifest()["params"],
            "would_write_candidate": outer_choice.improved,
        },
        "zip_deflated_member": {
            "archive_bytes": len(zip_deflated_bytes),
            "delta_bytes_vs_stored_archive": len(zip_deflated_bytes) - output_archive.stat().st_size,
            "would_write_candidate": len(zip_deflated_bytes) < output_archive.stat().st_size,
        },
    }

    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_delta_bytes_vs_source": (
            manifest["output_archive"]["bytes"] - manifest["source_archive"]["bytes"]
        ),
        "archive_path": manifest["output_archive"]["path"],
        "archive_sha256": manifest["output_archive"]["sha256"],
        "candidate_id": candidate_id,
        "decoded_stream_parity": True,
        "formula_only_rate_score_delta_vs_source": manifest[
            "formula_only_rate_score_delta_vs_source"
        ],
        "manifest_path": str(candidate_dir / "manifest.json"),
        "negative_container_checks": negative_checks,
        "noop": manifest["noop"],
        "noop_status": manifest["noop_status"],
        "payload_bytes": manifest["payload"]["bytes"],
        "payload_delta_bytes_vs_source": manifest["payload"]["bytes"] - len(source_payload),
        "payload_format": manifest["payload"]["format"],
        "pose_safety_preflight": pose_safety,
        "score_claim": False,
        "source_archive": manifest["source_archive"],
        "source_payload_format": source_payload_format,
        "source_preservation_status": manifest["source_preservation"]["status"],
        "stream_deltas": {
            name: choice.manifest()["delta_bytes_vs_source_stream"]
            for name, choice in sorted(choices.items())
        },
    }


def parse_source_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--source must be LABEL=PATH")
    label, path = raw.split("=", 1)
    return _safe_label(label), Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", type=parse_source_arg, required=True)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unpacker = _load_unpacker()
    cache: dict[str, BrotliChoice] = {}
    rows = [
        screen_source(
            label=label,
            source_archive=path,
            output_dir=output_dir,
            cache=cache,
            unpacker=unpacker,
        )
        for label, path in args.source
    ]
    summary = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidates": sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"])),
        "evidence_grade": "empirical_lossless_byte_transform",
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "schema": "c082_fast_p6_repack_screen_v1",
        "score_claim": False,
        "source_count": len(rows),
        "tool": str(Path(__file__).relative_to(REPO_ROOT)),
    }
    _write_json(output_dir / "p6_repack_screen_summary.json", summary)
    print(
        json.dumps(
            {
                "best_by_bytes": summary["candidates"][0],
                "output_dir": str(output_dir),
                "score_claim": False,
                "source_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
