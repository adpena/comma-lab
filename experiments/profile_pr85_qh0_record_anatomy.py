#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile PR85 QH0 renderer records without making score claims.

The PR85 model segment is already tight under outer Brotli.  This profiler
looks one layer deeper: it expands the charged PR85 bundle, parses the decoded
QH0 renderer payload with the same record order as the runtime loader, and
emits record-level byte/compressibility anatomy for future deterministic
serializer or repacker work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr85_bundle import expand_pr85_bundle_to_runtime_members  # noqa: E402
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer  # noqa: E402


SCHEMA = "pr85_qh0_record_anatomy_profile_v1"
TOOL = "experiments/profile_pr85_qh0_record_anatomy.py"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_qh0_record_anatomy_20260504_codex"
DEFAULT_JSON_OUT = DEFAULT_OUT_DIR / "pr85_qh0_record_anatomy_profile.json"
DEFAULT_MARKDOWN_OUT = DEFAULT_OUT_DIR / "pr85_qh0_record_anatomy_profile.md"
ORIGINAL_VIDEO_BYTES = 37_545_489


class QH0AnatomyError(RuntimeError):
    """Raised when the QH0 payload cannot be profiled safely."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = float(len(data))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _probe_compression(data: bytes) -> dict[str, Any]:
    probes: dict[str, int] = {
        "zlib_9": len(zlib.compress(data, 9)),
        "lzma_preset9": len(lzma.compress(data, preset=9)),
    }
    try:
        import brotli  # type: ignore

        probes["brotli_q11"] = len(brotli.compress(data, quality=11))
    except Exception as exc:  # pragma: no cover - depends on optional wheel
        return {
            "available": False,
            "error": repr(exc),
            "probes": probes,
            "best_probe": None,
        }
    best_name, best_bytes = min(probes.items(), key=lambda item: item[1])
    return {
        "available": True,
        "input_bytes": len(data),
        "probes": probes,
        "best_probe": {"codec": best_name, "bytes": best_bytes},
        "best_delta_vs_input": best_bytes - len(data),
        "best_ratio": round(best_bytes / max(1, len(data)), 12),
    }


def _read_archive_member(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise QH0AnatomyError(f"archive not found: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise QH0AnatomyError(f"{_rel(path)}: expected one member, got {len(infos)}")
        info = infos[0]
        if info.filename.startswith("/") or ".." in Path(info.filename).parts:
            raise QH0AnatomyError(f"unsafe member path: {info.filename!r}")
        raw = zf.read(info)
    return (
        {
            "archive_path": _rel(path),
            "archive_bytes": path.stat().st_size,
            "archive_sha256": _sha256_file(path),
            "member_name": info.filename,
            "member_bytes": len(raw),
            "member_sha256": _sha256_bytes(raw),
            "zip_stored": info.compress_type == zipfile.ZIP_STORED,
        },
        raw,
    )


def _module_weight_order(model: Any) -> list[tuple[str, Any]]:
    import torch

    ordered: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            ordered.append((name, module))
    return ordered


def _slice(raw: bytes, start: int, nbytes: int, label: str) -> bytes:
    end = start + nbytes
    if start < 0 or nbytes < 0 or end > len(raw):
        raise QH0AnatomyError(
            f"QH0 payload truncated while reading {label}: "
            f"start={start} nbytes={nbytes} payload={len(raw)}"
        )
    return raw[start:end]


def _record_stats(
    *,
    name: str,
    kind: str,
    category: str,
    raw: bytes,
    offset: int,
    nbytes: int,
    tensor_shape: tuple[int, ...],
    element_count: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = _slice(raw, offset, nbytes, name)
    probes = _probe_compression(data)
    best = probes.get("best_probe") or {}
    best_bytes = int(best.get("bytes", nbytes)) if isinstance(best, dict) else nbytes
    payload: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "category": category,
        "offset": offset,
        "end_offset": offset + nbytes,
        "bytes": nbytes,
        "sha256": _sha256_bytes(data),
        "tensor_shape": list(tensor_shape),
        "element_count": int(element_count),
        "entropy_bits_per_byte": round(_entropy_bits_per_byte(data), 12),
        "zero_fraction": round(data.count(0) / max(1, len(data)), 12),
        "compression_probe": probes,
        "best_probe_delta_vs_record_bytes": best_bytes - nbytes,
        "rate_score_if_record_removed_formula_only": round(nbytes * 25.0 / ORIGINAL_VIDEO_BYTES, 12),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def parse_qh0_records(raw: bytes) -> dict[str, Any]:
    """Parse QH0/QM0 records using the runtime loader's byte contract."""


    if len(raw) < 3:
        raise QH0AnatomyError("QH0 payload shorter than magic")
    magic = raw[:3]
    if magic not in (b"QH0", b"QM0"):
        raise QH0AnatomyError(f"unsupported renderer magic: {magic!r}")
    hilo_split = magic == b"QH0"
    model = build_quantizr_faithful_renderer()
    pos = 3
    records: list[dict[str, Any]] = []
    covered: set[str] = set()

    for module_name, module in _module_weight_order(model):
        kind_pos = pos
        kind = int(_slice(raw, pos, 1, f"{module_name}.weight.kind")[0])
        pos += 1
        shape = tuple(int(x) for x in module.weight.shape)
        numel = int(module.weight.numel())
        if kind == 1:
            block_size = 32
            blocks = (numel + block_size - 1) // block_size
            packed_len = (blocks * block_size + 1) // 2
            packed_offset = pos
            pos += packed_len
            scales_offset = pos
            pos += blocks * 2
            records.append(
                _record_stats(
                    name=f"{module_name}.weight",
                    kind="fp4_hilo" if hilo_split else "fp4_packed",
                    category="module_weight",
                    raw=raw,
                    offset=kind_pos,
                    nbytes=1 + packed_len + blocks * 2,
                    tensor_shape=shape,
                    element_count=numel,
                    extra={
                        "kind_byte": kind,
                        "block_size": block_size,
                        "block_count": blocks,
                        "packed_payload_offset": packed_offset,
                        "packed_payload_bytes": packed_len,
                        "scale_offset": scales_offset,
                        "scale_bytes": blocks * 2,
                    },
                )
            )
        elif kind == 0:
            nbytes = numel * 2
            pos += nbytes
            records.append(
                _record_stats(
                    name=f"{module_name}.weight",
                    kind="fp16_unsplit" if hilo_split else "fp16",
                    category="module_weight",
                    raw=raw,
                    offset=kind_pos,
                    nbytes=1 + nbytes,
                    tensor_shape=shape,
                    element_count=numel,
                    extra={"kind_byte": kind, "payload_bytes": nbytes},
                )
            )
        else:
            raise QH0AnatomyError(f"bad weight kind {kind} for {module_name}.weight")
        covered.add(f"{module_name}.weight")

        bias = getattr(module, "bias", None)
        if bias is not None:
            bias_shape = tuple(int(x) for x in bias.shape)
            bias_bytes = int(bias.numel()) * 2
            bias_offset = pos
            pos += bias_bytes
            records.append(
                _record_stats(
                    name=f"{module_name}.bias",
                    kind="fp16_unsplit_implicit" if hilo_split else "fp16_implicit",
                    category="module_bias",
                    raw=raw,
                    offset=bias_offset,
                    nbytes=bias_bytes,
                    tensor_shape=bias_shape,
                    element_count=int(bias.numel()),
                )
            )
            covered.add(f"{module_name}.bias")

    for key, tensor in model.state_dict().items():
        if key in covered:
            continue
        kind_pos = pos
        kind = int(_slice(raw, pos, 1, f"{key}.kind")[0])
        pos += 1
        shape = tuple(int(x) for x in tensor.shape)
        numel = int(tensor.numel())
        if kind == 2:
            rows = shape[0] if len(shape) >= 2 else 1
            q_offset = pos
            pos += numel
            scale_offset = pos
            pos += rows * 2
            records.append(
                _record_stats(
                    name=key,
                    kind="int8_row_scale_unsplit" if hilo_split else "int8_row_scale",
                    category="dense_tensor",
                    raw=raw,
                    offset=kind_pos,
                    nbytes=1 + numel + rows * 2,
                    tensor_shape=shape,
                    element_count=numel,
                    extra={
                        "kind_byte": kind,
                        "q_offset": q_offset,
                        "q_bytes": numel,
                        "scale_offset": scale_offset,
                        "scale_bytes": rows * 2,
                        "row_count": rows,
                    },
                )
            )
        elif kind == 0:
            nbytes = numel * 2
            pos += nbytes
            records.append(
                _record_stats(
                    name=key,
                    kind="fp16_unsplit" if hilo_split else "fp16",
                    category="dense_tensor",
                    raw=raw,
                    offset=kind_pos,
                    nbytes=1 + nbytes,
                    tensor_shape=shape,
                    element_count=numel,
                    extra={"kind_byte": kind, "payload_bytes": nbytes},
                )
            )
        else:
            raise QH0AnatomyError(f"bad dense kind {kind} for {key}")

    if pos != len(raw):
        raise QH0AnatomyError(f"trailing QH0 bytes: consumed={pos} total={len(raw)}")

    by_category: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for record in records:
        by_category[record["category"]] = by_category.get(record["category"], 0) + int(record["bytes"])
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + int(record["bytes"])

    recompressible = [
        record
        for record in records
        if int(record.get("best_probe_delta_vs_record_bytes", 0)) < 0 and int(record["bytes"]) >= 32
    ]
    return {
        "magic": magic.decode("ascii"),
        "payload_bytes": len(raw),
        "payload_sha256": _sha256_bytes(raw),
        "hilo_split": hilo_split,
        "consumed_bytes": pos,
        "record_count": len(records),
        "byte_accounting": {
            "records_total_bytes": sum(int(record["bytes"]) for record in records),
            "magic_bytes": 3,
            "records_plus_magic_bytes": 3 + sum(int(record["bytes"]) for record in records),
            "by_category": dict(sorted(by_category.items())),
            "by_kind": dict(sorted(by_kind.items())),
        },
        "top_records_by_bytes": sorted(records, key=lambda r: int(r["bytes"]), reverse=True)[:24],
        "top_recompressible_records": sorted(
            recompressible,
            key=lambda r: int(r["best_probe_delta_vs_record_bytes"]),
        )[:24],
        "records": records,
    }


def build_profile(archive: Path) -> dict[str, Any]:
    archive_meta, member = _read_archive_member(archive)
    expansion = expand_pr85_bundle_to_runtime_members(member)
    renderer = expansion.members.get("renderer.bin")
    if renderer is None:
        raise QH0AnatomyError("PR85 expansion did not produce renderer.bin")
    anatomy = parse_qh0_records(renderer)
    model_meta = expansion.manifest["runtime_members"]["renderer.bin"]
    recompressible_bytes = sum(
        -int(record["best_probe_delta_vs_record_bytes"])
        for record in anatomy["top_recompressible_records"]
        if int(record["best_probe_delta_vs_record_bytes"]) < 0
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "evidence_grade": "planning_only_qh0_record_anatomy",
        "archive": archive_meta,
        "renderer_member": {
            "bytes": len(renderer),
            "sha256": _sha256_bytes(renderer),
            "source_encoded_bytes": model_meta.get("source_encoded_bytes"),
            "source_encoded_sha256": model_meta.get("source_encoded_sha256"),
            "codec": model_meta.get("codec"),
        },
        "score_rate_formula": {
            "formula_only": True,
            "rate_score_per_byte": 25.0 / ORIGINAL_VIDEO_BYTES,
            "renderer_encoded_rate_score": round(
                float(model_meta.get("source_encoded_bytes", 0)) * 25.0 / ORIGINAL_VIDEO_BYTES,
                12,
            ),
            "score_claim_from_this_profile": False,
        },
        "anatomy": anatomy,
        "recommendations": [
            {
                "action": "build_qh0_record_serializer_before_any_model_byte_dispatch",
                "basis": "outer Brotli is already tight; only record-level rewrites can target model bytes without arbitrary byte surgery",
                "required_gates": [
                    "decode_qh0_state_dict tensor parity",
                    "deterministic encode_qh0_state_dict byte closure",
                    "runtime output parity on PR85 archive",
                    "dispatch claim before exact CUDA eval",
                ],
                "planning_only": True,
                "score_claim": False,
            },
            {
                "action": "prioritize_top_recompressible_records_only_after_serializer_exists",
                "basis": "record-local generic compression is a search map, not a directly composable archive format",
                "top_recompressible_record_count": len(anatomy["top_recompressible_records"]),
                "summed_top_recompressible_probe_bytes": recompressible_bytes,
                "planning_only": True,
                "score_claim": False,
            },
        ],
    }


def write_markdown(profile: Mapping[str, Any], path: Path) -> None:
    anatomy = profile["anatomy"]
    lines = [
        "# PR85 QH0 Record Anatomy",
        "",
        f"- planning_only: `{profile['planning_only']}`",
        f"- score_claim: `{profile['score_claim']}`",
        f"- renderer bytes: `{profile['renderer_member']['bytes']}`",
        f"- record count: `{anatomy['record_count']}`",
        "",
        "## Top Records By Bytes",
        "",
    ]
    for record in anatomy["top_records_by_bytes"][:12]:
        lines.append(
            f"- `{record['name']}` `{record['kind']}`: `{record['bytes']}` bytes, "
            f"best probe delta `{record['best_probe_delta_vs_record_bytes']}`"
        )
    lines.extend(["", "## Top Recompressible Records", ""])
    if anatomy["top_recompressible_records"]:
        for record in anatomy["top_recompressible_records"][:12]:
            lines.append(
                f"- `{record['name']}`: `{record['bytes']}` bytes, "
                f"best probe delta `{record['best_probe_delta_vs_record_bytes']}`"
            )
    else:
        lines.append("- none under record-local generic probes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--write-markdown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_profile(args.archive)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_bytes(_json_bytes(profile))
    if args.write_markdown:
        write_markdown(profile, args.output_markdown)
    print(f"wrote {_rel(args.output_json)}")
    if args.write_markdown:
        print(f"wrote {_rel(args.output_markdown)}")


if __name__ == "__main__":
    main()
