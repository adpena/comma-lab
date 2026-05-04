#!/usr/bin/env python3
"""Build PR75/C088-preserving renderer shrink candidates.

This is a local archive builder only. It preserves every non-renderer logical
payload member from a source archive, rewrites only ``renderer.bin``, and emits
deterministic byte-closed archives plus manifests. It does not dispatch remote
work and does not make score claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import build_blockfp_c067_archive as blockfp  # noqa: E402
from experiments import build_renderer_packed_payload_archive as packer  # noqa: E402
from submissions.robust_current.unpack_renderer_payload import (  # noqa: E402
    unpack_renderer_payload,
)
from tac.quantizr_qzs3_codec import decode_qzs3_state_dict, encode_qzs3_state_dict  # noqa: E402


SCHEMA = "renderer_shrink_candidate_builder_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/archive.zip"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/renderer_shrink_pr75_c088_20260503_worker"
)
RENDERER_MEMBER = "renderer.bin"
MASK_MEMBER = "masks.mkv"
POSE_MEMBERS = ("optimized_poses.bin", "optimized_poses.qp1")
PR75_PAYLOAD_MEMBER = "p"
PR75_MASK_LEN = 219_472
PR75_MODEL_LEN = 56_034
PR75_ACTIONS_LEN = 236
PR75_FIXED_MIN_LEN = PR75_MASK_LEN + PR75_MODEL_LEN + PR75_ACTIONS_LEN


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def parse_int_tuple(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    out: list[int] = []
    for item in items:
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer {item!r}") from exc
        if parsed <= 0 or parsed > 4096:
            raise argparse.ArgumentTypeError(
                f"integer entries must be in [1, 4096], got {parsed}"
            )
        if parsed not in out:
            out.append(parsed)
    return tuple(out)


def _read_single_payload(source_archive: Path) -> bytes | None:
    try:
        with zipfile.ZipFile(source_archive, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if len(infos) != 1 or infos[0].filename != PR75_PAYLOAD_MEMBER:
                return None
            return zf.read(PR75_PAYLOAD_MEMBER)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid zip archive: {source_archive}") from exc


def _parse_pr75_slices(payload: bytes) -> dict[str, Any] | None:
    if payload.startswith(b"P3"):
        header_size = 2 + struct.calcsize("<IHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        dict_len = 0
        record_count = None
        cursor = header_size
        format_name = "P3"
    elif payload.startswith(b"P4"):
        header_size = 2 + struct.calcsize("<IHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, dict_len, actions_len = struct.unpack_from("<IHHH", payload, 2)
        record_count = None
        cursor = header_size
        format_name = "P4"
    elif payload.startswith(b"P5"):
        header_size = 2 + struct.calcsize("<IHHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, dict_len, actions_len, record_count = struct.unpack_from(
            "<IHHHH",
            payload,
            2,
        )
        cursor = header_size
        format_name = "P5"
    elif payload.startswith(b"P6"):
        header_size = 2 + struct.calcsize("<IHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, actions_len, record_count = struct.unpack_from(
            "<IHHH",
            payload,
            2,
        )
        dict_len = 0
        cursor = header_size
        format_name = "P6"
    elif len(payload) > PR75_FIXED_MIN_LEN:
        mask_len = PR75_MASK_LEN
        model_len = PR75_MODEL_LEN
        actions_len = PR75_ACTIONS_LEN
        dict_len = 0
        record_count = None
        cursor = 0
        format_name = "fixed_pr75"
    else:
        return None

    if min(mask_len, model_len, actions_len) <= 0 or dict_len < 0:
        return None
    mask_start = cursor
    mask_end = mask_start + mask_len
    model_end = mask_end + model_len
    dict_end = model_end + dict_len
    actions_end = dict_end + actions_len
    if actions_end >= len(payload):
        return None
    return {
        "format": format_name,
        "mask": payload[mask_start:mask_end],
        "model": payload[mask_end:model_end],
        "action_dict": payload[model_end:dict_end],
        "actions": payload[dict_end:actions_end],
        "pose": payload[actions_end:],
        "record_count": record_count,
    }


def _validate_pr75_slices(
    slices: dict[str, Any],
    *,
    expected_members: dict[str, bytes],
) -> bool:
    """Return whether raw PR75 slices decode to the expected logical members."""

    import brotli

    try:
        mask = brotli.decompress(bytes(slices["mask"]))
        renderer = brotli.decompress(bytes(slices["model"]))
        pose = brotli.decompress(bytes(slices["pose"]))
        actions_wire = brotli.decompress(bytes(slices["actions"]))
    except Exception:
        return False
    if mask != expected_members.get(MASK_MEMBER):
        return False
    if renderer != expected_members.get(RENDERER_MEMBER):
        return False
    pose_expected = next(
        (expected_members[name] for name in POSE_MEMBERS if name in expected_members),
        None,
    )
    if pose_expected is None or pose != pose_expected:
        return False
    if "seg_tile_actions.bin" in expected_members:
        fmt = str(slices.get("format"))
        try:
            if fmt == "P6":
                from submissions.robust_current.unpack_renderer_payload import (
                    _decode_delta_varint_seg_tile_actions,
                )

                decoded_actions = _decode_delta_varint_seg_tile_actions(
                    actions_wire,
                    record_count=int(slices.get("record_count") or 0),
                )
            else:
                from submissions.robust_current.unpack_renderer_payload import (
                    _decode_seg_tile_actions,
                )

                decoded_actions = _decode_seg_tile_actions(actions_wire)
        except Exception:
            decoded_actions = actions_wire
        if decoded_actions != expected_members["seg_tile_actions.bin"]:
            return False
    return True


def _build_logical_rebrotli_pr75_slices(
    source_members: dict[str, bytes],
    *,
    brotli_quality: int,
) -> dict[str, Any] | None:
    """Build deterministic P3 slices from decoded logical runtime members.

    Public PR79 fixed-slice payloads do not carry self-describing lengths that
    this local builder can safely reuse after replacing only the renderer.  For
    those archives, rebuild the non-renderer logical streams into a conservative
    P3 payload and let the runtime unpacker prove byte closure.
    """

    import brotli

    pose_name = next((name for name in POSE_MEMBERS if name in source_members), None)
    if (
        MASK_MEMBER not in source_members
        or RENDERER_MEMBER not in source_members
        or pose_name is None
        or "seg_tile_actions.bin" not in source_members
    ):
        return None
    return {
        "format": "logical_rebrotli_p3",
        "mask": brotli.compress(source_members[MASK_MEMBER], quality=brotli_quality, lgwin=24),
        "model": brotli.compress(source_members[RENDERER_MEMBER], quality=brotli_quality, lgwin=24),
        "action_dict": b"",
        "actions": brotli.compress(
            source_members["seg_tile_actions.bin"],
            quality=brotli_quality,
            lgwin=24,
        ),
        "pose": brotli.compress(source_members[pose_name], quality=brotli_quality, lgwin=24),
        "record_count": None,
        "rebuilt_from_logical_members": True,
        "logical_pose_member": pose_name,
    }


def load_pr75_slices_for_renderer_shrink(
    source_archive: Path,
    source_members: dict[str, bytes],
    *,
    brotli_quality: int,
) -> dict[str, Any] | None:
    """Load or rebuild source slices for deterministic PR75/PR79-style output."""

    source_payload = _read_single_payload(source_archive)
    if source_payload is not None:
        parsed = _parse_pr75_slices(source_payload)
        if parsed is not None and _validate_pr75_slices(
            parsed,
            expected_members=source_members,
        ):
            return parsed
    return _build_logical_rebrotli_pr75_slices(
        source_members,
        brotli_quality=brotli_quality,
    )


def _build_pr75_payload(
    source_slices: dict[str, Any],
    *,
    renderer_bytes: bytes,
    brotli_quality: int,
) -> tuple[bytes, dict[str, Any]]:
    import brotli

    model = brotli.compress(renderer_bytes, quality=brotli_quality, lgwin=24)
    if len(model) > 65_535:
        raise ValueError(f"compressed renderer does not fit PR75 u16 model_len: {len(model)}")
    mask = bytes(source_slices["mask"])
    action_dict = bytes(source_slices["action_dict"])
    actions = bytes(source_slices["actions"])
    pose = bytes(source_slices["pose"])
    if len(mask) > 0xFFFF_FFFF or len(action_dict) > 65_535 or len(actions) > 65_535:
        raise ValueError("PR75 slice length does not fit self-describing header")

    source_format = str(source_slices["format"])
    if source_format in {"P4", "P5"} and action_dict:
        record_count = int(source_slices.get("record_count") or 0)
        if source_format == "P5":
            header = b"P5" + struct.pack(
                "<IHHHH",
                len(mask),
                len(model),
                len(action_dict),
                len(actions),
                record_count,
            )
        else:
            header = b"P4" + struct.pack(
                "<IHHH",
                len(mask),
                len(model),
                len(action_dict),
                len(actions),
            )
        payload_format = f"pr75_{source_format.lower()}_preserve_slices"
        payload = header + mask + model + action_dict + actions + pose
    elif source_format == "P6":
        record_count = int(source_slices.get("record_count") or 0)
        header = b"P6" + struct.pack(
            "<IHHH",
            len(mask),
            len(model),
            len(actions),
            record_count,
        )
        payload_format = "pr75_p6_preserve_slices"
        payload = header + mask + model + actions + pose
    else:
        header = b"P3" + struct.pack("<IHH", len(mask), len(model), len(actions))
        payload_format = (
            "pr75_p3_logical_rebrotli_slices"
            if source_format == "logical_rebrotli_p3"
            else "pr75_p3_preserve_slices"
        )
        payload = header + mask + model + actions + pose
    return payload, {
        "payload_format": payload_format,
        "source_payload_format": source_format,
        "rebuilt_from_logical_members": bool(
            source_slices.get("rebuilt_from_logical_members")
        ),
        "logical_pose_member": source_slices.get("logical_pose_member"),
        "mask_slice_bytes": len(mask),
        "renderer_slice_bytes": len(model),
        "action_dict_slice_bytes": len(action_dict),
        "actions_slice_bytes": len(actions),
        "pose_slice_bytes": len(pose),
        "record_count": source_slices.get("record_count"),
        "renderer_slice_sha256": _sha256_bytes(model),
    }


def _build_rpk1_payload(
    source_members: dict[str, bytes],
    *,
    renderer_bytes: bytes,
    brotli_quality: int,
) -> tuple[bytes, dict[str, Any]]:
    import brotli

    members = dict(source_members)
    members[RENDERER_MEMBER] = renderer_bytes
    pose_present = [name for name in POSE_MEMBERS if name in members]
    if MASK_MEMBER not in members or not pose_present:
        raise ValueError("RPK1 renderer shrink requires masks.mkv and an optimized pose payload")
    preferred = [
        MASK_MEMBER,
        RENDERER_MEMBER,
        "seg_tile_action_dict.bin",
        "seg_tile_actions.bin",
        *POSE_MEMBERS,
    ]
    ordered_names = [name for name in preferred if name in members]
    ordered_names.extend(sorted(name for name in members if name not in ordered_names))
    ordered = [(name, members[name]) for name in ordered_names]
    payload, header = packer.build_renderer_payload(ordered, pose_codec="raw")
    compressed = brotli.compress(payload, quality=brotli_quality, lgwin=24)
    return compressed, {
        "payload_format": "rpk1_json_brotli",
        "member_order": ordered_names,
        "payload_raw_bytes": len(payload),
        "payload_header": header,
    }


def _write_single_member_archive(output_archive: Path, payload: bytes) -> None:
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    packer.write_deterministic_payload_archive(
        output_archive,
        payload,
        payload_member_name=PR75_PAYLOAD_MEMBER,
    )


def _verify_archive(
    output_archive: Path,
    *,
    expected_renderer: bytes,
    expected_non_renderer_members: dict[str, bytes],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="renderer_shrink_verify_") as tmpdir:
        archive_dir = Path(tmpdir)
        with zipfile.ZipFile(output_archive, "r") as zf:
            names = zf.namelist()
            if names != [PR75_PAYLOAD_MEMBER]:
                raise ValueError(f"output archive must contain only p; got {names}")
            (archive_dir / PR75_PAYLOAD_MEMBER).write_bytes(zf.read(PR75_PAYLOAD_MEMBER))
        summary = unpack_renderer_payload(archive_dir)
        renderer = (archive_dir / RENDERER_MEMBER).read_bytes()
        if renderer != expected_renderer:
            raise ValueError("runtime unpacker did not reconstruct transformed renderer")
        for name, expected in expected_non_renderer_members.items():
            actual = (archive_dir / name).read_bytes()
            if actual != expected:
                raise ValueError(f"runtime unpacker changed non-renderer member {name}")
    return summary


def _member_meta(members: dict[str, bytes]) -> dict[str, Any]:
    return {
        name: {
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        for name, data in sorted(members.items())
    }


def build_renderer_shrink_candidates(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    renderer_export: Path | None = None,
    qzs3_block_sizes: tuple[int, ...] = (48, 64, 96, 128),
    brotli_quality: int = 11,
    force: bool = False,
) -> dict[str, Any]:
    """Build deterministic renderer-only shrink candidates."""

    if not 0 <= brotli_quality <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")
    source_archive = source_archive.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is non-empty; pass --force to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = source_archive.read_bytes()
    source_sha = _sha256_bytes(source_bytes)
    source_members, source_packaging = blockfp.extract_runtime_members(source_archive)
    if RENDERER_MEMBER not in source_members:
        raise ValueError(f"source archive missing {RENDERER_MEMBER}")
    pose_present = [name for name in POSE_MEMBERS if name in source_members]
    if MASK_MEMBER not in source_members or not pose_present:
        raise ValueError("source archive must contain masks.mkv and an optimized pose payload")
    source_renderer = source_members[RENDERER_MEMBER]
    renderer_input = source_renderer
    renderer_input_meta: dict[str, Any] = {
        "role": "source_renderer",
        "path": str(source_archive),
        "bytes": len(source_renderer),
        "sha256": _sha256_bytes(source_renderer),
        "same_as_source_renderer": True,
    }
    if renderer_export is not None:
        renderer_export = renderer_export.resolve()
        renderer_input = renderer_export.read_bytes()
        renderer_input_meta = {
            "role": "external_renderer_export",
            "path": str(renderer_export),
            "bytes": len(renderer_input),
            "sha256": _sha256_bytes(renderer_input),
            "same_as_source_renderer": renderer_input == source_renderer,
        }
    if not renderer_input.startswith(b"QZS3"):
        raise ValueError(
            "renderer shrink builder currently supports QZS3 renderer inputs only; "
            f"got {renderer_input[:4]!r}"
        )
    state = decode_qzs3_state_dict(renderer_input, device="cpu")
    pr75_slices = load_pr75_slices_for_renderer_shrink(
        source_archive,
        source_members,
        brotli_quality=brotli_quality,
    )
    output_layout = "pr75_preserved_slices" if pr75_slices is not None else "rpk1_json"
    non_renderer_members = {
        name: data for name, data in source_members.items() if name != RENDERER_MEMBER
    }

    candidates: list[dict[str, Any]] = []
    renderer_variants: list[tuple[str, bytes, int | None, str]] = []
    if renderer_export is not None:
        renderer_variants.append(
            ("external_qzs3_direct", renderer_input, None, "external_export_direct")
        )
    for block_size in qzs3_block_sizes:
        prefix = "external_qzs3" if renderer_export is not None else "qzs3"
        renderer_variants.append(
            (
                f"{prefix}_b{block_size:04d}",
                encode_qzs3_state_dict(state, block_size=block_size),
                block_size,
                "qzs3_reencoded_state",
            )
        )

    for variant_prefix, renderer, block_size, transform_kind in renderer_variants:
        candidate_id = f"{variant_prefix}_{output_layout}"
        candidate_dir = output_dir / candidate_id
        archive_path = candidate_dir / "archive.zip"
        if pr75_slices is not None:
            payload, payload_meta = _build_pr75_payload(
                pr75_slices,
                renderer_bytes=renderer,
                brotli_quality=brotli_quality,
            )
        else:
            payload, payload_meta = _build_rpk1_payload(
                source_members,
                renderer_bytes=renderer,
                brotli_quality=brotli_quality,
            )
        _write_single_member_archive(archive_path, payload)
        runtime_unpack = _verify_archive(
            archive_path,
            expected_renderer=renderer,
            expected_non_renderer_members=non_renderer_members,
        )
        archive_bytes = archive_path.stat().st_size
        archive_sha = _sha256_file(archive_path)
        manifest = {
            "schema": SCHEMA,
            "candidate_id": candidate_id,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical_archive_candidate_until_pose_safety_and_exact_cuda",
            "source_archive": {
                "path": str(source_archive),
                "bytes": len(source_bytes),
                "sha256": source_sha,
                **source_packaging,
            },
            "output_archive": {
                "path": str(archive_path),
                "bytes": archive_bytes,
                "sha256": archive_sha,
                "delta_bytes_vs_source_archive": archive_bytes - len(source_bytes),
                "formula_only_rate_delta_vs_source_archive": (
                    25.0 * (archive_bytes - len(source_bytes)) / ORIGINAL_VIDEO_BYTES
                ),
            },
            "renderer_transform": {
                "codec": "qzs3",
                "source_block_size": int.from_bytes(source_renderer[4:6], "little"),
                "input_block_size": int.from_bytes(renderer_input[4:6], "little"),
                "output_block_size": int(block_size) if block_size is not None else None,
                "transform_kind": transform_kind,
                "renderer_input": renderer_input_meta,
                "source_bytes": len(source_renderer),
                "source_sha256": _sha256_bytes(source_renderer),
                "output_bytes": len(renderer),
                "output_sha256": _sha256_bytes(renderer),
                "output_same_as_source_renderer": renderer == source_renderer,
                "output_same_as_renderer_input": renderer == renderer_input,
            },
            "non_renderer_preservation": {
                "all_non_renderer_members_preserved": True,
                "members": _member_meta(non_renderer_members),
            },
            "payload": {
                "member_name": PR75_PAYLOAD_MEMBER,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "brotli_quality": brotli_quality,
                **payload_meta,
            },
            "runtime_contract": {
                "byte_closed": True,
                "single_payload_member": True,
                "runtime_unpack_verified": True,
                "runtime_unpack_summary": runtime_unpack,
                "renderer_only_transplant": True,
                "canonical_score_source_required": (
                    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                    "experiments/contest_auth_eval.py --device cuda"
                ),
                "pose_safety_preflight_required_before_dispatch": True,
                "remote_gpu_dispatch_performed": False,
            },
        }
        manifest_path = candidate_dir / "build_manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        candidates.append(
            {
                "candidate_id": candidate_id,
                "archive": str(archive_path),
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha,
                "manifest": str(manifest_path),
                "qzs3_block_size": int(block_size) if block_size is not None else None,
                "delta_bytes_vs_source_archive": archive_bytes - len(source_bytes),
                "renderer_bytes": len(renderer),
                "renderer_sha256": _sha256_bytes(renderer),
                "payload_format": payload_meta["payload_format"],
                "renderer_input_role": renderer_input_meta["role"],
                "renderer_transform_kind": transform_kind,
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
    best = min(candidates, key=lambda item: (item["archive_bytes"], item["qzs3_block_size"]))
    summary = {
        "schema": "renderer_shrink_candidate_summary_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": {
            "path": str(source_archive),
            "bytes": len(source_bytes),
            "sha256": source_sha,
        },
        "renderer_input": renderer_input_meta,
        "source_members": _member_meta(source_members),
        "source_payload_layout": output_layout,
        "qzs3_block_sizes": list(qzs3_block_sizes),
        "candidate_count": len(candidates),
        "best_by_archive_bytes": best,
        "candidates": candidates,
    }
    (output_dir / "summary.json").write_bytes(_json_bytes(summary))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--renderer-export",
        type=Path,
        default=None,
        help=(
            "Optional external QZS3 trained renderer export to transplant while "
            "preserving all non-renderer runtime members from --source-archive."
        ),
    )
    parser.add_argument("--qzs3-block-sizes", type=parse_int_tuple, default=(48, 64, 96, 128))
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_renderer_shrink_candidates(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        renderer_export=args.renderer_export,
        qzs3_block_sizes=args.qzs3_block_sizes,
        brotli_quality=args.brotli_quality,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
