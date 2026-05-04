#!/usr/bin/env python3
"""Build local-only PR85 QH0/QM0 serializer candidates.

The tool only rewrites the PR85 model segment when the decoded model payload is
still accepted by an existing runtime path. It never edits runtime files, runs
CUDA, dispatches jobs, or claims score evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    Pr85BundleError,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)
from tac.qh0_record_serializer import (  # noqa: E402
    build_serialized_variants,
    choose_byte_win_candidates,
    prove_decoded_tensor_parity,
    record_set_summary,
    sha256_bytes,
)


TOOL = "experiments/build_pr85_qh0_serializer_candidates.py"
SCHEMA = "pr85_qh0_serializer_candidates_v1"
MANIFEST_SCHEMA = "pr85_qh0_serializer_candidate_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_qh0_serializer_candidates_20260504_codex"
DEFAULT_REPLAY_INFLATE = (
    REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/replay_submission/inflate.py"
)
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
ORIGINAL_VIDEO_BYTES = 37_545_489
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
KNOWN_PUBLIC_PR85 = {
    "archive_bytes": 236_328,
    "archive_sha256": "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e",
}


class QH0SerializerCandidateError(RuntimeError):
    """Raised when a PR85 serializer candidate cannot be built safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _read_single_x_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise QH0SerializerCandidateError(f"source archive not found: {_repo_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise QH0SerializerCandidateError(
                f"PR85 source archive must contain exactly one member 'x'; got {names!r}"
            )
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    archive_sha = _sha256_file(path)
    return (
        {
            "path": _repo_rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": archive_sha,
            "known_public_pr85_v5_match": {
                "matches": (
                    int(path.stat().st_size) == KNOWN_PUBLIC_PR85["archive_bytes"]
                    and archive_sha == KNOWN_PUBLIC_PR85["archive_sha256"]
                ),
                "expected_archive_bytes": KNOWN_PUBLIC_PR85["archive_bytes"],
                "expected_archive_sha256": KNOWN_PUBLIC_PR85["archive_sha256"],
            },
            "member_name": info.filename,
            "member_file_size": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_crc32_hex": f"{info.CRC:08x}",
            "member_sha256": sha256_bytes(raw),
            "zip_compress_type": int(info.compress_type),
        },
        raw,
    )


def _brotli_compress(data: bytes, *, quality: int, lgwin: int) -> bytes:
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - environment guard
        raise QH0SerializerCandidateError("brotli is required for PR85 model recode") from exc
    return brotli.compress(data, quality=int(quality), lgwin=int(lgwin))


def _compression_grid(data: bytes, *, qualities: Sequence[int], lgwins: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for quality in qualities:
        for lgwin in lgwins:
            encoded = _brotli_compress(data, quality=quality, lgwin=lgwin)
            key = (sha256_bytes(encoded), len(encoded))
            rows.append(
                {
                    "codec": "brotli",
                    "quality": int(quality),
                    "lgwin": int(lgwin),
                    "bytes": len(encoded),
                    "sha256": sha256_bytes(encoded),
                    "payload": encoded,
                    "duplicate_stream": key in seen,
                }
            )
            seen.add(key)
    return rows


def _source_header_mode(bundle_format: str) -> str:
    return "explicit_30" if bundle_format == "pr85_explicit_30byte_lengths" else "v5"


def _zip_info_x() -> zipfile.ZipInfo:
    info = zipfile.ZipInfo("x", FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_single_x_archive(path: Path, x_payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(_zip_info_x(), x_payload, compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if [info.filename for info in infos] != ["x"]:
            raise QH0SerializerCandidateError("deterministic candidate archive wrote wrong members")
        member = zf.read("x")
        info = infos[0]
    return {
        "path": _repo_rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": "x",
        "member_file_size": int(info.file_size),
        "member_compress_size": int(info.compress_size),
        "member_crc32_hex": f"{info.CRC:08x}",
        "member_sha256": sha256_bytes(member),
        "zip_compress_type": int(info.compress_type),
    }


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def runtime_compatibility(
    magic: str,
    *,
    replay_inflate_py: Path = DEFAULT_REPLAY_INFLATE,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
) -> dict[str, Any]:
    """Static no-edit runtime support check for a serialized model magic."""

    replay_text = _text(replay_inflate_py)
    robust_inflate_renderer = _text(robust_current_dir / "inflate_renderer.py")
    robust_unpacker = _text(robust_current_dir / "unpack_renderer_payload.py")
    magic_token = f'b"{magic}"'
    replay_model_loader = (
        "get_decoded_state_dict_custom" in replay_text
        and magic_token in replay_text
        and "load_compact_archive_bundle" in replay_text
        and 'path = data_dir / "x"' in replay_text
    )
    robust_renderer_member_loader = magic_token in robust_inflate_renderer
    robust_single_x_unpacker = (
        'path = data_dir / "x"' in robust_unpacker
        or 'ARCHIVE_DIR/x' in robust_unpacker
        or '"/x"' in robust_unpacker
    )
    blockers: list[str] = []
    if not replay_model_loader:
        blockers.append(f"public_pr85_replay_missing_{magic}_model_loader")
    if not robust_renderer_member_loader:
        blockers.append(f"robust_current_missing_{magic}_renderer_member_loader")
    if not robust_single_x_unpacker:
        blockers.append("robust_current_missing_pr85_single_x_unpacker")
    runtime_can_decode_without_edits = replay_model_loader
    return {
        "magic": magic,
        "runtime_can_decode_without_edits": runtime_can_decode_without_edits,
        "dispatch_unlocked": runtime_can_decode_without_edits,
        "public_pr85_replay_single_x_can_decode": replay_model_loader,
        "public_pr85_replay_inflate_py": _repo_rel(replay_inflate_py),
        "robust_current_renderer_member_can_decode": robust_renderer_member_loader,
        "robust_current_single_x_can_unpack": robust_single_x_unpacker,
        "robust_current_dir": _repo_rel(robust_current_dir),
        "blockers": blockers,
        "blocker_class": None if runtime_can_decode_without_edits else "runtime_incompatibility",
        "minimal_runtime_implementation_needed": (
            None
            if runtime_can_decode_without_edits
            else "Add a no-sidecar PR85 single-x parser plus QH0/QM0 model loader to the scored runtime path."
        ),
    }


def _best_screened_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            int(row.get("candidate_model_delta_bytes_vs_source", 0)),
            str(row.get("candidate_id", "")),
        ),
    )


def build_candidates(
    archive: Path,
    out_dir: Path,
    *,
    qualities: Sequence[int],
    lgwins: Sequence[int],
    replay_inflate_py: Path = DEFAULT_REPLAY_INFLATE,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
) -> dict[str, Any]:
    source_archive, x_raw = _read_single_x_archive(archive)
    bundle = parse_pr85_bundle(x_raw)
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - environment guard
        raise QH0SerializerCandidateError("brotli is required for PR85 model decode") from exc
    source_model_segment = bytes(bundle.segments["model"])
    source_renderer = brotli.decompress(source_model_segment)
    record_set, variants = build_serialized_variants(source_renderer)
    header_mode = _source_header_mode(bundle.format)

    screened: list[dict[str, Any]] = []
    parity_by_variant: dict[str, dict[str, Any]] = {}
    for variant in variants:
        parity = prove_decoded_tensor_parity(source_renderer, variant.payload, device="cpu")
        parity_by_variant[variant.variant_id] = parity
        runtime = runtime_compatibility(
            variant.magic,
            replay_inflate_py=replay_inflate_py,
            robust_current_dir=robust_current_dir,
        )
        streams = _compression_grid(variant.payload, qualities=qualities, lgwins=lgwins)
        if variant.same_as_source:
            streams.append(
                {
                    "codec": "source_passthrough",
                    "quality": None,
                    "lgwin": None,
                    "bytes": len(source_model_segment),
                    "sha256": sha256_bytes(source_model_segment),
                    "payload": source_model_segment,
                    "duplicate_stream": False,
                }
            )
        for stream in streams:
            candidate_id = (
                f"{variant.variant_id}_source_passthrough"
                if stream["codec"] == "source_passthrough"
                else f"{variant.variant_id}_brq{stream['quality']}_lg{stream['lgwin']}"
            )
            model_delta = int(stream["bytes"]) - len(source_model_segment)
            screened.append(
                {
                    "candidate_id": candidate_id,
                    "serializer_variant": variant.variant_id,
                    "serializer_magic": variant.magic,
                    "serializer_payload_bytes": variant.payload_bytes,
                    "serializer_payload_sha256": variant.payload_sha256,
                    "candidate_model_codec": stream["codec"],
                    "candidate_model_brotli_quality": stream["quality"],
                    "candidate_model_brotli_lgwin": stream["lgwin"],
                    "candidate_model_bytes": int(stream["bytes"]),
                    "candidate_model_sha256": stream["sha256"],
                    "candidate_model_delta_bytes_vs_source": model_delta,
                    "candidate_archive_delta_bytes_vs_source_formula": model_delta,
                    "rate_score_delta_if_components_identical_formula_only": model_delta
                    * 25.0
                    / ORIGINAL_VIDEO_BYTES,
                    "decoded_tensor_parity": parity["decoded_tensor_parity"],
                    "runtime_compatibility": runtime,
                    "duplicate_stream": bool(stream.get("duplicate_stream", False)),
                    "_candidate_model_payload": stream["payload"],
                    "_serializer_payload": variant.payload,
                }
            )

    buildable = choose_byte_win_candidates(screened, require_runtime_compatible=True)
    built_candidates: list[dict[str, Any]] = []
    for row in sorted(buildable, key=lambda item: (int(item["candidate_model_bytes"]), item["candidate_id"])):
        if not bool(row.get("decoded_tensor_parity", False)):
            continue
        candidate_id = str(row["candidate_id"])
        candidate_dir = out_dir / candidate_id
        segments = dict(bundle.segments)
        segments["model"] = bytes(row["_candidate_model_payload"])
        candidate_x = pack_pr85_bundle(segments, header_mode=header_mode)
        candidate_archive = _write_single_x_archive(candidate_dir / "archive.zip", candidate_x)
        archive_delta = int(candidate_archive["archive_bytes"]) - int(source_archive["archive_bytes"])
        if archive_delta >= 0:
            continue
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "tool": TOOL,
            "candidate_id": candidate_id,
            "score_claim": False,
            "dispatch_performed": False,
            "remote_gpu_dispatch_performed": False,
            "source_archive": source_archive,
            "source_model_segment": {
                "bytes": len(source_model_segment),
                "sha256": sha256_bytes(source_model_segment),
                "decoded_qh0_bytes": len(source_renderer),
                "decoded_qh0_sha256": sha256_bytes(source_renderer),
            },
            "candidate_model_segment": {
                "bytes": int(row["candidate_model_bytes"]),
                "sha256": row["candidate_model_sha256"],
                "delta_bytes_vs_source": int(row["candidate_model_delta_bytes_vs_source"]),
                "codec": row["candidate_model_codec"],
                "brotli_quality": row["candidate_model_brotli_quality"],
                "brotli_lgwin": row["candidate_model_brotli_lgwin"],
                "decoded_serializer_magic": row["serializer_magic"],
                "decoded_serializer_bytes": row["serializer_payload_bytes"],
                "decoded_serializer_sha256": row["serializer_payload_sha256"],
            },
            "decoded_tensor_parity_proof": parity_by_variant[str(row["serializer_variant"])],
            "runtime_compatibility": row["runtime_compatibility"],
            "candidate_archive": candidate_archive,
            "candidate_archive_delta_bytes_vs_source": archive_delta,
            "exact_eval_eligibility": {
                "eligible": True,
                "dispatch_unlocked": True,
                "requires_lane_claim_before_remote_eval": True,
                "score_claim_from_this_artifact": False,
            },
        }
        (candidate_dir / "manifest.json").write_bytes(_json_bytes(manifest))
        built_candidates.append(manifest)

    best_screened = _strip_private(_best_screened_row(screened))
    best_built = (
        min(built_candidates, key=lambda item: int(item["candidate_archive_delta_bytes_vs_source"]))
        if built_candidates
        else None
    )
    if built_candidates:
        blocker_class = None
        blocker = None
        dispatch_unlocked = True
    elif best_screened is None:
        blocker_class = "no_serializer_candidates"
        blocker = "no QH0/QM0 serializer variants were produced"
        dispatch_unlocked = False
    elif not any(
        bool(row.get("runtime_compatibility", {}).get("runtime_can_decode_without_edits"))
        for row in screened
    ):
        blocker_class = "runtime_incompatibility"
        blocker = "no serializer variant is accepted by the existing no-edit PR85 runtime path"
        dispatch_unlocked = False
    else:
        blocker_class = "no_real_byte_win"
        blocker = "all runtime-compatible serialized model streams were byte-neutral or byte-negative"
        dispatch_unlocked = False

    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": not bool(built_candidates),
        "score_claim": False,
        "dispatch_performed": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": source_archive,
        "source_model_segment": {
            "bytes": len(source_model_segment),
            "sha256": sha256_bytes(source_model_segment),
            "decoded_qh0_bytes": len(source_renderer),
            "decoded_qh0_sha256": sha256_bytes(source_renderer),
        },
        "record_set": record_set_summary(record_set),
        "screened_candidate_count": len(screened),
        "built_candidate_count": len(built_candidates),
        "dispatch_unlocked": dispatch_unlocked,
        "blocker_class": blocker_class,
        "blocker": blocker,
        "best_screened_candidate": best_screened,
        "best_built_candidate": _candidate_summary(best_built) if best_built else None,
        "candidate_manifests": [
            {
                "candidate_id": candidate["candidate_id"],
                "manifest_path": _repo_rel(out_dir / candidate["candidate_id"] / "manifest.json"),
                "archive_path": candidate["candidate_archive"]["path"],
                "archive_bytes": candidate["candidate_archive"]["archive_bytes"],
                "archive_sha256": candidate["candidate_archive"]["archive_sha256"],
                "archive_delta_bytes_vs_source": candidate["candidate_archive_delta_bytes_vs_source"],
                "candidate_model_bytes": candidate["candidate_model_segment"]["bytes"],
                "candidate_model_sha256": candidate["candidate_model_segment"]["sha256"],
                "candidate_model_delta_bytes_vs_source": candidate["candidate_model_segment"][
                    "delta_bytes_vs_source"
                ],
            }
            for candidate in built_candidates
        ],
        "screened_candidates": [_strip_private(row) for row in sorted(
            screened,
            key=lambda item: (
                int(item["candidate_model_delta_bytes_vs_source"]),
                str(item["candidate_id"]),
            ),
        )[:48]],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_summary.json").write_bytes(_json_bytes(summary))
    return summary


def _strip_private(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _candidate_summary(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate["candidate_id"],
        "archive_path": candidate["candidate_archive"]["path"],
        "archive_bytes": candidate["candidate_archive"]["archive_bytes"],
        "archive_sha256": candidate["candidate_archive"]["archive_sha256"],
        "archive_delta_bytes_vs_source": candidate["candidate_archive_delta_bytes_vs_source"],
        "candidate_model_bytes": candidate["candidate_model_segment"]["bytes"],
        "candidate_model_sha256": candidate["candidate_model_segment"]["sha256"],
        "candidate_model_delta_bytes_vs_source": candidate["candidate_model_segment"][
            "delta_bytes_vs_source"
        ],
    }


def _parse_int_csv(text: str, *, label: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} must be comma-separated ints") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return tuple(values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--replay-inflate-py", type=Path, default=DEFAULT_REPLAY_INFLATE)
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    parser.add_argument("--qualities", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--lgwins", default="18,20,22,24")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_candidates(
        args.archive,
        args.out_dir,
        qualities=_parse_int_csv(args.qualities, label="qualities"),
        lgwins=_parse_int_csv(args.lgwins, label="lgwins"),
        replay_inflate_py=args.replay_inflate_py,
        robust_current_dir=args.robust_current_dir,
    )
    print(
        json.dumps(
            {
                "summary_path": _repo_rel(args.out_dir / "candidate_summary.json"),
                "built_candidate_count": summary["built_candidate_count"],
                "dispatch_unlocked": summary["dispatch_unlocked"],
                "blocker_class": summary["blocker_class"],
                "best_screened_candidate": summary["best_screened_candidate"],
                "best_built_candidate": summary["best_built_candidate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
