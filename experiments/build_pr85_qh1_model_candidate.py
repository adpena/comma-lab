#!/usr/bin/env python3
"""Build lossless PR85 QH1 model-repack candidates for byte screening.

QH1 is a lossless wrapper around PR85's decoded QH0 renderer bytes.  The
candidate replaces selected record slices with zeroes in a base template, stores
compressed patches for those slices, and reconstructs the exact original QH0
bytes before the existing tensor decoder runs.  This tool is build/local only:
it does not run scorers, dispatch GPUs, or claim score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr85_bundle import expand_pr85_bundle_to_runtime_members, pack_pr85_bundle, parse_pr85_bundle  # noqa: E402
from tac.qh0_renderer_codec import QH1_HEADER_STRUCT, QH1_SCHEMA, reconstruct_qh1_payload  # noqa: E402
from tac.submission_archive import write_deterministic_zip_member  # noqa: E402

from experiments.profile_pr85_qh0_record_anatomy import parse_qh0_records  # noqa: E402


TOOL = "experiments/build_pr85_qh1_model_candidate.py"
SCHEMA = "pr85_qh1_model_candidates_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_qh1_model_candidates_20260504_codex"
ORIGINAL_VIDEO_BYTES = 37_545_489
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class QH1CandidateError(RuntimeError):
    """Raised when a QH1 candidate cannot be built safely."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_single_x(path: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise QH1CandidateError(f"PR85 source archive must contain exactly ['x']; got {names!r}")
        raw = zf.read(infos[0])
    return (
        {
            "path": _rel(path),
            "archive_bytes": path.stat().st_size,
            "archive_sha256": _sha256_file(path),
            "member_name": "x",
            "member_bytes": len(raw),
            "member_sha256": _sha256(raw),
        },
        raw,
    )


def _compress(codec: str, data: bytes) -> bytes:
    if codec == "raw":
        return data
    if codec == "zlib":
        return zlib.compress(data, 9)
    if codec == "lzma":
        return lzma.compress(data, preset=9)
    if codec == "brotli":
        import brotli  # type: ignore

        return brotli.compress(data, quality=11, lgwin=24)
    raise QH1CandidateError(f"unsupported codec {codec!r}")


def _best_codec(data: bytes, codecs: Iterable[str]) -> tuple[str, bytes]:
    options = [(codec, _compress(codec, data)) for codec in codecs]
    return min(options, key=lambda item: (len(item[1]), item[0]))


def build_qh1_payload(
    source_qh0: bytes,
    records: list[Mapping[str, Any]],
    *,
    max_records: int,
    min_record_saving: int,
    patch_codecs: tuple[str, ...] = ("brotli", "zlib", "lzma"),
    base_codecs: tuple[str, ...] = ("brotli", "zlib", "lzma"),
) -> tuple[bytes, dict[str, Any]]:
    selected = [
        record
        for record in sorted(
            records,
            key=lambda row: int(row.get("best_probe_delta_vs_record_bytes", 0)),
        )
        if -int(record.get("best_probe_delta_vs_record_bytes", 0)) >= min_record_saving
    ][:max_records]
    base = bytearray(source_qh0)
    patch_payloads: list[bytes] = []
    patch_records: list[dict[str, Any]] = []
    used: list[tuple[int, int]] = []
    for record in selected:
        offset = int(record["offset"])
        nbytes = int(record["bytes"])
        end = offset + nbytes
        if any(not (end <= old_start or old_end <= offset) for old_start, old_end in used):
            raise QH1CandidateError(f"overlapping selected QH0 record at offset {offset}")
        used.append((offset, end))
        original = bytes(source_qh0[offset:end])
        base[offset:end] = b"\x00" * nbytes
        codec, encoded = _best_codec(original, patch_codecs)
        patch_payloads.append(encoded)
        patch_records.append(
            {
                "name": record.get("name"),
                "offset": offset,
                "nbytes": nbytes,
                "codec": codec,
                "encoded_bytes": len(encoded),
                "encoded_sha256": _sha256(encoded),
                "decoded_sha256": _sha256(original),
                "source_probe_delta_vs_record_bytes": int(
                    record.get("best_probe_delta_vs_record_bytes", 0)
                ),
            }
        )

    base_codec, base_encoded = _best_codec(bytes(base), base_codecs)
    header = {
        "schema": QH1_SCHEMA,
        "source_bytes": len(source_qh0),
        "source_sha256": _sha256(source_qh0),
        "base_codec": base_codec,
        "base_encoded_bytes": len(base_encoded),
        "base_encoded_sha256": _sha256(base_encoded),
        "base_decoded_sha256": _sha256(bytes(base)),
        "records": patch_records,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    qh1 = b"QH1" + QH1_HEADER_STRUCT.pack(len(header_bytes)) + header_bytes + base_encoded + b"".join(patch_payloads)
    if reconstruct_qh1_payload(qh1) != source_qh0:
        raise QH1CandidateError("QH1 reconstruction did not match source QH0 bytes")
    return qh1, {
        "selected_record_count": len(selected),
        "selected_records": patch_records,
        "base_codec": base_codec,
        "base_encoded_bytes": len(base_encoded),
        "header_bytes": len(header_bytes),
        "qh1_bytes": len(qh1),
        "qh1_sha256": _sha256(qh1),
        "source_qh0_bytes": len(source_qh0),
        "source_qh0_sha256": _sha256(source_qh0),
        "reconstructs_exact_qh0": True,
    }


def _write_single_x_archive(path: Path, x_payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        write_deterministic_zip_member(
            zf,
            "x",
            x_payload,
            compress_type=zipfile.ZIP_STORED,
            compresslevel=None,
        )


def build_candidates(
    archive: Path,
    out_dir: Path,
    *,
    max_records_values: tuple[int, ...],
    min_record_saving: int,
) -> dict[str, Any]:
    source_meta, x_raw = _read_single_x(archive)
    bundle = parse_pr85_bundle(x_raw)
    expansion = expand_pr85_bundle_to_runtime_members(x_raw)
    source_qh0 = expansion.members["renderer.bin"]
    anatomy = parse_qh0_records(source_qh0)
    records = anatomy["records"]
    candidates: list[dict[str, Any]] = []
    import brotli  # type: ignore

    for max_records in max_records_values:
        qh1, qh1_meta = build_qh1_payload(
            source_qh0,
            records,
            max_records=max_records,
            min_record_saving=min_record_saving,
        )
        segments = dict(bundle.segments)
        segments["model"] = brotli.compress(qh1, quality=11, lgwin=24)
        x_candidate = pack_pr85_bundle(segments, header_mode=bundle.format == "pr85_explicit_30byte_lengths" and "explicit_30" or "v5")
        candidate_id = f"qh1_top{max_records}_minsaving{min_record_saving}"
        candidate_dir = out_dir / candidate_id
        archive_path = candidate_dir / "archive.zip"
        _write_single_x_archive(archive_path, x_candidate)
        model_delta = len(segments["model"]) - len(bundle.segments["model"])
        archive_delta = archive_path.stat().st_size - int(source_meta["archive_bytes"])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "archive_path": _rel(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": _sha256_file(archive_path),
                "archive_delta_bytes_vs_source": archive_delta,
                "model_segment_bytes": len(segments["model"]),
                "model_segment_delta_bytes_vs_source": model_delta,
                "rate_score_delta_if_components_identical_formula_only": archive_delta
                * 25.0
                / ORIGINAL_VIDEO_BYTES,
                "qh1": qh1_meta,
                "runtime_compatibility": {
                    "single_x_requires_qh1_aware_pr85_runtime": True,
                    "current_public_pr85_inflate_unmodified": False,
                    "robust_current_qh1_loader": True,
                    "dispatchable_now": False,
                    "blocker": "QH1 single-x candidate needs a scored runtime tree that parses PR85 x and QH1; byte screen only.",
                },
            }
        )
        manifest = {"schema": "pr85_qh1_model_candidate_v1", "source_archive": source_meta, "candidate": candidates[-1]}
        (candidate_dir / "manifest.json").write_bytes(_json_bytes(manifest))

    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "source_archive": source_meta,
        "source_model_segment_bytes": len(bundle.segments["model"]),
        "source_renderer_qh0_bytes": len(source_qh0),
        "source_renderer_qh0_sha256": _sha256(source_qh0),
        "candidate_count": len(candidates),
        "best_candidate": min(candidates, key=lambda row: int(row["archive_delta_bytes_vs_source"]))
        if candidates
        else None,
        "candidates": candidates,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_summary.json").write_bytes(_json_bytes(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-records", default="1,2,4,8,16,24")
    parser.add_argument("--min-record-saving", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_records = tuple(int(part) for part in args.max_records.split(",") if part.strip())
    summary = build_candidates(
        args.archive,
        args.out_dir,
        max_records_values=max_records,
        min_record_saving=args.min_record_saving,
    )
    print(json.dumps(summary["best_candidate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
