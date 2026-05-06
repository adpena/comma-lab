#!/usr/bin/env python3
"""Byte-screen PR73-style flatpack transfer on PR79/C102 fixed-slice archives.

This is contest-faithful local screening only.  It does not dispatch GPU jobs
and does not claim scores.  The only emitted archive form is the currently
runtime-closed ``p = Brotli(RPK1(...))`` container accepted by
``submissions/robust_current/unpack_renderer_payload.py``.
"""
from __future__ import annotations

import argparse
import brotli
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
PACKER_PATH = REPO_ROOT / "experiments/build_renderer_packed_payload_archive.py"
DEFAULT_C102_ARCHIVE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/"
    "archive.zip"
)
DEFAULT_PR79_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)
DEFAULT_PR79_PROFILE = DEFAULT_PR79_ARCHIVE.parent / "pr79_minp_grammar_profile.json"
DEFAULT_PR79_PARITY = (
    DEFAULT_PR79_ARCHIVE.parent / "raw_output_parity_pairs_cpu/pr75_raw_output_parity.json"
)
DEFAULT_EXISTING_MATRIX = (
    REPO_ROOT / "experiments/results/pr79_action_subset_worker_20260503/candidate_matrix.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr79_flatpack_transfer_worker_20260503"

TOOL = "experiments/build_pr79_flatpack_transfer_candidates.py"
SCHEMA = "pr79_flatpack_transfer_candidate_matrix_v1"
MANIFEST_SCHEMA = "pr79_flatpack_transfer_candidate_manifest_v1"
MEMBER_NAME = "p"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
C102_SCORE = 0.31514430182167497
C102_BYTES = 276_485
C102_SHA256 = "79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8"
PR79_SHA256 = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"
TARGET_SCORE = 0.31
RPK1_MEMBER_ORDER = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.bin")
SOURCE_POSE_ALIASES = ("optimized_poses.qp1", "optimized_poses.bin")
CUDA_AUTH_EVAL_REQUIRED = (
    "No dispatch from this worker. Future dispatch requires a lane claim with "
    "tools/claim_lane_dispatch.py claim and exact CUDA auth eval on identical "
    "archive bytes via archive.zip -> inflate.sh -> upstream/evaluate.py."
)


@dataclass(frozen=True)
class LoadedArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    decoded: dict[str, bytes]
    runtime_members: dict[str, dict[str, Any]]
    rpk1_members: dict[str, bytes]


@dataclass(frozen=True)
class BrotliChoice:
    payload: bytes
    params: dict[str, int]
    order: tuple[str, ...]
    rpk1_bytes: int
    rpk1_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_unpacker() -> Any:
    return _load_module(UNPACKER_PATH, "pr79_flatpack_unpacker")


def _load_packer() -> Any:
    return _load_module(PACKER_PATH, "pr79_flatpack_rpk1_packer")


def _safe_zip_member_name(name: str) -> str:
    path = Path(name)
    if (
        not name
        or name.startswith("/")
        or ".." in path.parts
        or len(path.parts) != 1
        or name.startswith(".")
        or name.startswith("__MACOSX/")
        or name.startswith("._")
    ):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return name


def _read_single_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [_safe_zip_member_name(info.filename) for info in infos]
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly member {MEMBER_NAME!r}; got {names!r}")
        return zf.read(MEMBER_NAME)


def _zip_info() -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(MEMBER_NAME, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(), payload)


def _member_summary(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in header.get("members", []):
        if not isinstance(item, Mapping):
            continue
        name = str(item["name"])
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item.get("codec", "raw")),
            "decoded_bytes": int(item.get("decoded_bytes", item["bytes"])),
            "decoded_sha256": str(item.get("decoded_sha256", item["sha256"])),
            "sha256": str(item["sha256"]),
        }
    return out


def _decoded_summary(decoded: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(decoded.items())
    }


def _normalise_decoded(decoded: Mapping[str, bytes]) -> dict[str, bytes]:
    out = dict(decoded)
    if "optimized_poses.bin" in out and "optimized_poses.qp1" not in out:
        out["optimized_poses.qp1"] = out["optimized_poses.bin"]
    if "optimized_poses.qp1" in out and "optimized_poses.bin" not in out:
        out["optimized_poses.bin"] = out["optimized_poses.qp1"]
    return out


def _rpk1_members_from_decoded(decoded: Mapping[str, bytes]) -> dict[str, bytes]:
    normal = _normalise_decoded(decoded)
    required = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin")
    missing = [name for name in required if name not in normal]
    if not any(name in normal for name in SOURCE_POSE_ALIASES):
        missing.append("optimized_poses.qp1")
    if missing:
        raise ValueError(f"archive is missing required runtime members: {missing}")
    return {
        "masks.mkv": normal["masks.mkv"],
        "renderer.bin": normal["renderer.bin"],
        "seg_tile_actions.bin": normal["seg_tile_actions.bin"],
        "optimized_poses.bin": normal["optimized_poses.bin"],
    }


def load_archive(label: str, path: Path, unpacker: Any) -> LoadedArchive:
    path = path.resolve()
    payload = _read_single_payload(path)
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    return LoadedArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        decoded=decoded,
        runtime_members=_member_summary(header),
        rpk1_members=_rpk1_members_from_decoded(decoded),
    )


def default_brotli_param_grid() -> list[dict[str, int]]:
    params: list[dict[str, int]] = []
    for quality in (11, 10, 9):
        for mode in (0, 1, 2):
            for lgwin in (18, 20, 22, 24):
                for lgblock in (0, 18, 20):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append(
                        {"quality": quality, "mode": mode, "lgwin": lgwin, "lgblock": lgblock}
                    )
    return params


def fast_brotli_param_grid() -> list[dict[str, int]]:
    return [{"quality": 11, "mode": 0, "lgwin": 22, "lgblock": 0}]


def _build_rpk1_for_order(
    *,
    source: LoadedArchive,
    order: tuple[str, ...],
    packer: Any,
) -> tuple[bytes, dict[str, Any]]:
    ordered = [(name, source.rpk1_members[name]) for name in order]
    return packer.build_renderer_payload(
        ordered,
        source_archive_sha256=source.archive_sha256,
        pose_codec="raw",
    )


def _best_brotli_rpk1(
    *,
    source: LoadedArchive,
    params: Iterable[Mapping[str, int]],
    packer: Any,
) -> BrotliChoice:
    best: BrotliChoice | None = None
    for order in itertools.permutations(RPK1_MEMBER_ORDER):
        rpk1_payload, _header = _build_rpk1_for_order(source=source, order=order, packer=packer)
        rpk1_sha = _sha256_bytes(rpk1_payload)
        for param in params:
            compressed = brotli.compress(rpk1_payload, **dict(param))
            if brotli.decompress(compressed) != rpk1_payload:
                raise ValueError("Brotli RPK1 round-trip failed")
            choice = BrotliChoice(
                payload=compressed,
                params=dict(param),
                order=tuple(order),
                rpk1_bytes=len(rpk1_payload),
                rpk1_sha256=rpk1_sha,
            )
            if best is None or len(choice.payload) < len(best.payload):
                best = choice
    if best is None:
        raise ValueError("no Brotli RPK1 candidate generated")
    return best


def _validate_rpk1_candidate(
    *,
    compressed_payload: bytes,
    source: LoadedArchive,
    unpacker: Any,
) -> dict[str, Any]:
    raw_rpk1 = brotli.decompress(compressed_payload)
    header, decoded = unpacker._parse_payload(raw_rpk1)  # noqa: SLF001
    normal_decoded = _normalise_decoded(decoded)
    expected = _normalise_decoded(source.rpk1_members)
    mismatches = []
    for name, expected_data in expected.items():
        actual = normal_decoded.get(name)
        if actual != expected_data:
            mismatches.append(
                {
                    "actual_sha256": _sha256_bytes(actual) if actual is not None else None,
                    "expected_sha256": _sha256_bytes(expected_data),
                    "name": name,
                }
            )
    missing = sorted(set(expected) - set(normal_decoded))
    extra = sorted(set(normal_decoded) - set(expected))
    status = "passed" if not missing and not extra and not mismatches else "failed"
    if status != "passed":
        raise ValueError(f"RPK1 decoded parity failed for {source.label}: {mismatches}")
    return {
        "decoded_parity_status": status,
        "members": _member_summary(header),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
        "payload_format": str(header.get("payload_format", "rpk1_json")),
        "runtime_parser": _repo_rel(UNPACKER_PATH),
        "status": status,
    }


def _break_even(candidate_bytes: int, source_bytes: int) -> dict[str, Any]:
    delta_vs_c102 = candidate_bytes - C102_BYTES
    delta_vs_source = candidate_bytes - source_bytes
    rate_delta_vs_c102 = delta_vs_c102 * RATE_SCORE_PER_BYTE
    score_if_components_unchanged_vs_c102 = C102_SCORE + rate_delta_vs_c102
    return {
        "component_score_delta_needed_to_tie_c102": rate_delta_vs_c102,
        "component_score_improvement_needed_to_tie_c102": max(0.0, rate_delta_vs_c102),
        "component_score_improvement_needed_for_target_0_31": max(
            0.0,
            score_if_components_unchanged_vs_c102 - TARGET_SCORE,
        ),
        "delta_bytes_vs_c102": delta_vs_c102,
        "delta_bytes_vs_source": delta_vs_source,
        "rate_score_delta_vs_c102": rate_delta_vs_c102,
        "rate_score_delta_vs_source": delta_vs_source * RATE_SCORE_PER_BYTE,
        "score_if_components_unchanged_vs_c102": score_if_components_unchanged_vs_c102,
    }


def _source_summary(source: LoadedArchive) -> dict[str, Any]:
    return {
        "archive_bytes": source.archive_bytes,
        "archive_path": _repo_rel(source.path),
        "archive_sha256": source.archive_sha256,
        "decoded_members": _decoded_summary(source.decoded),
        "payload_bytes": len(source.payload),
        "payload_format": source.payload_format,
        "payload_sha256": source.payload_sha256,
        "runtime_members": source.runtime_members,
    }


def _selected_frame_parity(source: LoadedArchive, parity_path: Path) -> dict[str, Any]:
    if source.archive_sha256 == PR79_SHA256 and parity_path.exists():
        try:
            report = json.loads(parity_path.read_text())
            render = report.get("render_parity", {})
            return {
                "artifact": _repo_rel(parity_path),
                "pair_indices": report.get("pair_indices", []),
                "selected_raw_after_actions_exact_equal": bool(
                    render.get("public_vs_robust_current", {})
                    .get("selected_raw_after_actions", {})
                    .get("exact_equal")
                ),
                "status": "reused_public_pr79_selected_pair_raw_parity",
            }
        except json.JSONDecodeError:
            return {"artifact": _repo_rel(parity_path), "status": "parity_artifact_invalid_json"}
    return {
        "reason": "generated RPK1 candidate is decoded-byte identical to its source archive; selected-frame render parity was not rerun in this byte screen",
        "status": "not_rerun_for_source_preserving_container_transform",
    }


def _candidate_recommendation(row: Mapping[str, Any]) -> dict[str, Any]:
    if bool(row["noop"]):
        return {"dispatch": False, "reason": "no-op archive"}
    if int(row["delta_bytes_vs_source"]) >= 0:
        return {
            "dispatch": False,
            "reason": "byte-regressive versus its source; exact eval needs non-byte component rationale",
        }
    return {
        "dispatch": False,
        "reason": "no remote dispatch from this worker; claim lane and rerun exact CUDA only if operator promotes this byte-closed candidate",
    }


def _emit_candidate(
    *,
    source: LoadedArchive,
    choice: BrotliChoice,
    output_dir: Path,
    unpacker: Any,
    force: bool,
    parity_path: Path,
) -> dict[str, Any]:
    candidate_id = f"{source.label}_rpk1_single_brotli_flatpack"
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    _write_archive(archive_path, choice.payload)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256_file(archive_path)
    validation = _validate_rpk1_candidate(
        compressed_payload=choice.payload,
        source=source,
        unpacker=unpacker,
    )
    noop = archive_bytes == source.archive_bytes and archive_sha == source.archive_sha256
    break_even = _break_even(archive_bytes, source.archive_bytes)
    row = {
        "archive_bytes": archive_bytes,
        "archive_path": _repo_rel(archive_path),
        "archive_sha256": archive_sha,
        "candidate_id": candidate_id,
        "container": "brotli_rpk1_single_stream_permuted",
        "decode_parity_status": validation["status"],
        "delta_bytes_vs_c102": break_even["delta_bytes_vs_c102"],
        "delta_bytes_vs_source": break_even["delta_bytes_vs_source"],
        "dispatch_recommendation": None,
        "exact_break_even_required_component_delta": {
            "vs_c102": break_even["component_score_improvement_needed_to_tie_c102"],
            "for_target_0_31_from_c102_anchor": break_even[
                "component_score_improvement_needed_for_target_0_31"
            ],
        },
        "manifest_path": _repo_rel(manifest_path),
        "noop": noop,
        "payload_bytes": len(choice.payload),
        "payload_changed_vs_source": choice.payload != source.payload,
        "payload_sha256": _sha256_bytes(choice.payload),
        "rpk1_bytes": choice.rpk1_bytes,
        "rpk1_sha256": choice.rpk1_sha256,
        "score_claim": False,
        "source_archive_sha256": source.archive_sha256,
        "source_id": source.label,
        "source_payload_format": source.payload_format,
    }
    row["dispatch_recommendation"] = _candidate_recommendation(row)
    manifest = {
        "break_even": break_even,
        "brotli_params": choice.params,
        "candidate": row,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "determinism": {
            "zip_compress_type": "ZIP_STORED",
            "zip_member": MEMBER_NAME,
            "zip_timestamp": list(FIXED_ZIP_TIMESTAMP),
        },
        "evidence_grade": "empirical_byte_screen_only",
        "manifest_schema": MANIFEST_SCHEMA,
        "member_order": list(choice.order),
        "promotion_eligible": False,
        "runtime_parse_validation": validation,
        "score_claim": False,
        "selected_frame_parity": _selected_frame_parity(source, parity_path),
        "source_archive": _source_summary(source),
        "tool": TOOL,
    }
    _write_json(manifest_path, manifest)
    return row


def _load_existing_matrix_sources(path: Path, *, max_sources: int) -> list[tuple[str, Path]]:
    if max_sources <= 0 or not path.exists():
        return []
    matrix = json.loads(path.read_text())
    rows = matrix.get("candidates", [])
    if not isinstance(rows, list):
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("expected_proxy_minus_required", -1e9)),
            int(row.get("archive_bytes", 10**12)),
            str(row.get("candidate_id", "")),
        ),
    )
    out: list[tuple[str, Path]] = []
    seen_sha: set[str] = set()
    for row in ranked:
        archive_path = row.get("archive_path")
        archive_sha = str(row.get("archive_sha256", ""))
        candidate_id = str(row.get("candidate_id", "candidate"))
        if not archive_path or archive_sha in seen_sha:
            continue
        seen_sha.add(archive_sha)
        out.append((candidate_id, Path(str(archive_path))))
        if len(out) >= max_sources:
            break
    return out


def build_candidates(
    *,
    c102_archive: Path = DEFAULT_C102_ARCHIVE,
    pr79_archive: Path = DEFAULT_PR79_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    existing_matrix: Path = DEFAULT_EXISTING_MATRIX,
    pr79_profile: Path = DEFAULT_PR79_PROFILE,
    pr79_parity: Path = DEFAULT_PR79_PARITY,
    max_existing_sources: int = 4,
    force: bool = False,
    params: Iterable[Mapping[str, int]] | None = None,
    unpacker: Any | None = None,
    packer: Any | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unpacker = unpacker or _load_unpacker()
    packer = packer or _load_packer()
    param_grid = list(params or default_brotli_param_grid())

    source_specs = [
        ("c102_anchor", c102_archive),
        ("pr79_public_fixed_slices", pr79_archive),
        *_load_existing_matrix_sources(existing_matrix, max_sources=max_existing_sources),
    ]
    sources: list[LoadedArchive] = []
    seen_archive_sha: set[str] = set()
    for label, path in source_specs:
        source = load_archive(label, path, unpacker)
        if source.archive_sha256 in seen_archive_sha:
            continue
        seen_archive_sha.add(source.archive_sha256)
        sources.append(source)

    c102 = sources[0]
    pr79 = next(source for source in sources if source.label == "pr79_public_fixed_slices")
    if c102.archive_sha256 != C102_SHA256:
        raise ValueError(f"C102 archive SHA mismatch: expected {C102_SHA256}, got {c102.archive_sha256}")
    if pr79.archive_sha256 != PR79_SHA256:
        raise ValueError(f"PR79 archive SHA mismatch: expected {PR79_SHA256}, got {pr79.archive_sha256}")

    rows: list[dict[str, Any]] = []
    for source in sources:
        choice = _best_brotli_rpk1(source=source, params=param_grid, packer=packer)
        rows.append(
            _emit_candidate(
                source=source,
                choice=choice,
                output_dir=output_dir,
                unpacker=unpacker,
                force=force,
                parity_path=pr79_parity,
            )
        )

    rows.sort(key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])))
    summary = {
        "archive_anatomy_comparison": {
            "c102": _source_summary(c102),
            "pr79": _source_summary(pr79),
            "stream_sha_equalities": {
                name: _normalise_decoded(c102.decoded).get(name)
                == _normalise_decoded(pr79.decoded).get(name)
                for name in RPK1_MEMBER_ORDER
            },
        },
        "candidate_count": len(rows),
        "candidates": rows,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "compressor_attempts": {
            "brotli": {
                "attempted": True,
                "runtime_closed": True,
                "reason": "top-level p is Brotli-decompressed by robust_current before RPK1 parse",
            },
            "lzma2": {
                "attempted": False,
                "runtime_closed": False,
                "reason": "no current top-level p LZMA2 inflate closure in robust_current",
            },
            "zstd": {
                "attempted": False,
                "runtime_closed": False,
                "reason": "no current top-level p Zstd inflate closure in robust_current",
            },
        },
        "evidence_grade": "empirical_byte_screen_only",
        "existing_matrix": _repo_rel(existing_matrix),
        "max_existing_sources": max_existing_sources,
        "no_remote_dispatch_performed": True,
        "pr79_profile": _repo_rel(pr79_profile),
        "promotion_eligible": False,
        "schema": SCHEMA,
        "score_claim": False,
        "source_count": len(sources),
        "target_score": TARGET_SCORE,
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c102-archive", type=Path, default=DEFAULT_C102_ARCHIVE)
    parser.add_argument("--pr79-archive", type=Path, default=DEFAULT_PR79_ARCHIVE)
    parser.add_argument("--existing-matrix", type=Path, default=DEFAULT_EXISTING_MATRIX)
    parser.add_argument("--pr79-profile", type=Path, default=DEFAULT_PR79_PROFILE)
    parser.add_argument("--pr79-parity", type=Path, default=DEFAULT_PR79_PARITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-existing-sources", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fast-brotli-grid",
        action="store_true",
        help="Use one Brotli setting for fast tests instead of the focused screening grid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_candidates(
        c102_archive=args.c102_archive,
        pr79_archive=args.pr79_archive,
        existing_matrix=args.existing_matrix,
        pr79_profile=args.pr79_profile,
        pr79_parity=args.pr79_parity,
        output_dir=args.output_dir,
        max_existing_sources=args.max_existing_sources,
        force=bool(args.force),
        params=fast_brotli_param_grid() if args.fast_brotli_grid else default_brotli_param_grid(),
    )
    print(
        json.dumps(
            {
                "best_by_bytes": summary["candidates"][0] if summary["candidates"] else None,
                "candidate_count": summary["candidate_count"],
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
