#!/usr/bin/env python3
"""Build a deterministic Alpha grayscale mask replacement archive.

This is the first contest-evaluable Alpha archive shape:

    renderer.bin
    grayscale.mkv
    optimized_poses.bin

It intentionally omits ``masks.mkv`` so ``inflate.sh`` auto-dispatches the
existing ``renderer_grayscale`` path. The builder consumes an
``alpha_mask_candidate_builder`` manifest, verifies artifact SHA/bytes against
disk, verifies the anchor archive custody, and writes a deterministic ZIP plus
an adjacent provenance manifest. It does not run scorers and makes no score
claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
import zlib


REPO_ROOT = Path(__file__).resolve().parents[1]
ALPHA_CANDIDATE_BUILDER = REPO_ROOT / "experiments" / "alpha_mask_candidate_builder.py"
DEFAULT_ANCHOR = (
    REPO_ROOT
    / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "experiments/results/alpha_mask_candidate_builder_pfp16_20260501_compact_crf_sweep"
    / "crf_63/alpha_mask_candidate_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/alpha_mask_replacement_pfp16_crf63_grayscale_only/archive.zip"
)
SCHEMA = "alpha_mask_replacement_archive_v1"
FIXED_DT = (1980, 1, 1, 0, 0, 0)
REQUIRED_ANCHOR_MEMBERS = ("renderer.bin", "optimized_poses.bin")
REPAIR_MEMBER_RAW = "alpha4_residual_repair.amr1"
REPAIR_COMPRESSORS = ("raw", "zlib", "lzma_xz", "brotli")


def _load_alpha_builder_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "alpha_mask_candidate_builder_for_archive_repair",
        ALPHA_CANDIDATE_BUILDER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Alpha candidate builder from {ALPHA_CANDIDATE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _safe_member_name(name: str) -> str:
    if not name or name.startswith("/"):
        raise ValueError(f"unsafe archive member path: {name!r}")
    parts = Path(name).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    if "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    if any(part.startswith("._") for part in parts) or "__MACOSX" in parts or ".DS_Store" in parts:
        raise ValueError(f"hidden/system archive member: {name!r}")
    return name


def _read_archive_members(archive: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    if not archive.is_file():
        raise FileNotFoundError(f"anchor archive not found: {archive}")
    members: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        seen: set[str] = set()
        for info in zf.infolist():
            name = _safe_member_name(info.filename)
            if name in seen:
                raise ValueError(f"duplicate archive member: {name!r}")
            seen.add(name)
            if info.is_dir():
                raise ValueError(f"unexpected archive directory member: {name!r}")
            data = zf.read(info)
            members[name] = data
            inventory.append(
                {
                    "name": name,
                    "size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(data),
                }
            )
    return members, inventory


def _resolve_manifest_path(path_text: str, *, manifest_dir: Path) -> Path:
    path = Path(path_text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([REPO_ROOT / path, manifest_dir / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"manifest artifact path not found: {path_text!r}")


def _load_candidate_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != "alpha_mask_candidate_builder_v1":
        raise ValueError(f"unexpected candidate manifest schema: {payload.get('schema')!r}")
    candidate = payload.get("candidate") or {}
    readiness = candidate.get("candidate_archive_readiness") or {}
    if readiness.get("full_sequence_candidate") is not True:
        raise ValueError("candidate manifest is not a full-sequence candidate")
    if readiness.get("ready_for_exact_eval_finalist_archive_assembly") is not True:
        raise ValueError("candidate manifest is not marked ready for finalist archive assembly")
    return payload


def _find_grayscale_artifact(candidate_manifest: dict[str, Any], *, manifest_dir: Path) -> dict[str, Any]:
    artifacts = candidate_manifest.get("candidate", {}).get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("candidate manifest missing candidate.artifacts list")
    matches = [
        item
        for item in artifacts
        if item.get("role") == "alpha4_grayscale_lut_video"
        and item.get("candidate_archive_member") in {"grayscale.mkv", "masks.alpha4.mkv"}
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one grayscale artifact, found {len(matches)}")
    record = dict(matches[0])
    path = _resolve_manifest_path(str(record["path"]), manifest_dir=manifest_dir)
    actual_size = path.stat().st_size
    actual_sha = _sha256_file(path)
    if int(record.get("size_bytes", -1)) != actual_size:
        raise ValueError("grayscale artifact size mismatch against manifest")
    if str(record.get("sha256")) != actual_sha:
        raise ValueError("grayscale artifact sha256 mismatch against manifest")
    record["resolved_path"] = str(path)
    return record


def _find_repair_artifact(candidate_manifest: dict[str, Any], *, manifest_dir: Path) -> dict[str, Any]:
    artifacts = candidate_manifest.get("candidate", {}).get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("candidate manifest missing candidate.artifacts list")
    matches = [
        item
        for item in artifacts
        if item.get("role") == "alpha4_residual_repair_payload"
        and item.get("candidate_archive_member") == REPAIR_MEMBER_RAW
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one residual repair artifact, found {len(matches)}")
    record = dict(matches[0])
    path = _resolve_manifest_path(str(record["path"]), manifest_dir=manifest_dir)
    actual_size = path.stat().st_size
    actual_sha = _sha256_file(path)
    if int(record.get("size_bytes", -1)) != actual_size:
        raise ValueError("residual repair artifact size mismatch against manifest")
    if str(record.get("sha256")) != actual_sha:
        raise ValueError("residual repair artifact sha256 mismatch against manifest")
    record["resolved_path"] = str(path)
    return record


def _candidate_source_mask_sha(candidate_manifest: dict[str, Any]) -> str | None:
    source = candidate_manifest.get("source")
    if not isinstance(source, dict):
        return None
    mask_member = source.get("mask_member")
    if not isinstance(mask_member, dict):
        return None
    sha = mask_member.get("sha256")
    return str(sha) if sha else None


def _parse_repair_policy(policy: str) -> dict[str, Any]:
    normalized = policy.strip().lower()
    if normalized in {"none", "omit", "omit_repair_payload"}:
        return {
            "kind": "none",
            "name": "none",
            "classes": (),
            "frame_group_count": None,
            "pair_indices": (),
        }
    if normalized in {"full", "full_repair", "all"}:
        return {
            "kind": "full",
            "name": "full",
            "classes": (),
            "frame_group_count": None,
            "pair_indices": (),
        }
    if normalized.startswith("class_prefix_"):
        suffix = normalized.removeprefix("class_prefix_")
        classes = tuple(int(token) for token in suffix.split("_") if token != "")
        if not classes:
            raise ValueError(f"repair policy {policy!r} did not name any classes")
        if any(class_id < 0 or class_id > 4 for class_id in classes):
            raise ValueError(f"repair policy {policy!r} includes class outside [0,4]")
        if len(set(classes)) != len(classes):
            raise ValueError(f"repair policy {policy!r} repeats a class")
        return {
            "kind": "class_prefix",
            "name": "class_prefix_" + "_".join(str(value) for value in classes),
            "classes": classes,
            "frame_group_count": None,
            "pair_indices": (),
        }
    match = re.fullmatch(r"top_residual_frame_groups_(\d+)(?:_of_\d+)?", normalized)
    if match:
        count = int(match.group(1))
        if count <= 0:
            raise ValueError(f"repair policy {policy!r} has nonpositive frame-group count")
        return {
            "kind": "top_residual_frame_groups",
            "name": normalized,
            "classes": (),
            "frame_group_count": count,
            "pair_indices": (),
        }
    if normalized.startswith("pair_indices_"):
        suffix = normalized.removeprefix("pair_indices_")
        pair_indices = tuple(int(token) for token in suffix.split("_") if token != "")
        if not pair_indices:
            raise ValueError(f"repair policy {policy!r} did not name any pair indices")
        if any(pair_index < 0 or pair_index >= 600 for pair_index in pair_indices):
            raise ValueError(f"repair policy {policy!r} includes pair index outside [0,599]")
        if len(set(pair_indices)) != len(pair_indices):
            raise ValueError(f"repair policy {policy!r} repeats a pair index")
        return {
            "kind": "pair_indices",
            "name": "pair_indices_" + "_".join(str(value) for value in pair_indices),
            "classes": (),
            "frame_group_count": None,
            "pair_indices": pair_indices,
        }
    raise ValueError(
        "repair policy must be one of none, full, class_prefix_<ids>, or "
        "top_residual_frame_groups_<count>_of_<total>, or pair_indices_<ids>, "
        f"got {policy!r}"
    )


def _compress_repair_payload(raw_payload: bytes, compressor: str) -> tuple[str, bytes]:
    if compressor not in REPAIR_COMPRESSORS:
        raise ValueError(f"unknown repair compressor {compressor!r}")
    if compressor == "raw":
        return REPAIR_MEMBER_RAW, raw_payload
    if compressor == "zlib":
        return f"{REPAIR_MEMBER_RAW}.zlib", zlib.compress(raw_payload, level=9)
    if compressor == "lzma_xz":
        return f"{REPAIR_MEMBER_RAW}.xz", lzma.compress(
            raw_payload,
            format=lzma.FORMAT_XZ,
            preset=9 | lzma.PRESET_EXTREME,
        )
    try:
        import brotli  # type: ignore
    except Exception as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("brotli repair compression requested but brotli is unavailable") from exc
    return f"{REPAIR_MEMBER_RAW}.br", brotli.compress(raw_payload, quality=11, lgwin=24)


def _build_selected_repair_member(
    *,
    candidate_manifest: dict[str, Any],
    manifest_dir: Path,
    repair_policy: str,
    repair_compressor: str,
    repair_frame_group_size: int = 50,
) -> tuple[str, bytes, dict[str, Any]] | None:
    policy = _parse_repair_policy(repair_policy)
    policy_kind = str(policy["kind"])
    policy_classes = tuple(int(value) for value in policy["classes"])
    policy_pair_indices = tuple(int(value) for value in policy["pair_indices"])
    if policy_kind == "none":
        return None
    if repair_frame_group_size <= 0:
        raise ValueError(f"repair_frame_group_size must be positive, got {repair_frame_group_size}")

    alpha_builder = _load_alpha_builder_module()
    repair_record = _find_repair_artifact(candidate_manifest, manifest_dir=manifest_dir)
    full_payload = Path(repair_record["resolved_path"]).read_bytes()
    full_header, full_runs = alpha_builder._decode_repair_payload(full_payload)
    total_pixels = int(full_header.get("selection", {}).get("total_residual_pixels", 0))
    total_runs = len(full_runs)

    if policy_kind == "full":
        selected_runs = list(full_runs)
        policy_name = "full"
        policy_details: dict[str, Any] = {}
    elif policy_kind == "class_prefix":
        selected_classes = set(policy_classes)
        selected_runs = [run for run in full_runs if int(run.class_id) in selected_classes]
        policy_name = str(policy["name"])
        policy_details = {"source_classes": [int(value) for value in policy_classes]}
    elif policy_kind == "top_residual_frame_groups":
        frame_group_count = int(policy["frame_group_count"])
        groups: dict[int, list[Any]] = {}
        for run in full_runs:
            group_id = int(run.frame_index) // repair_frame_group_size
            groups.setdefault(group_id, []).append(run)
        group_rank = sorted(
            groups,
            key=lambda group_id: (
                -sum(int(run.length) for run in groups[group_id]),
                group_id,
            ),
        )
        selected_group_ids = group_rank[:frame_group_count]
        selected_group_set = set(selected_group_ids)
        selected_runs = [
            run
            for run in full_runs
            if int(run.frame_index) // repair_frame_group_size in selected_group_set
        ]
        policy_name = str(policy["name"])
        policy_details = {
            "frame_group_size": int(repair_frame_group_size),
            "selected_group_count": int(len(selected_group_ids)),
            "requested_group_count": int(frame_group_count),
            "total_group_count": int(len(group_rank)),
            "selected_group_ids": [int(value) for value in selected_group_ids],
            "selected_group_ranges": [
                {
                    "start_frame": int(group_id * repair_frame_group_size),
                    "end_frame_exclusive": int((group_id + 1) * repair_frame_group_size),
                }
                for group_id in selected_group_ids
            ],
        }
    elif policy_kind == "pair_indices":
        selected_pairs = set(policy_pair_indices)
        selected_runs = [
            run
            for run in full_runs
            if int(run.frame_index) // 2 in selected_pairs
        ]
        policy_name = str(policy["name"])
        selected_frames = sorted({int(run.frame_index) for run in selected_runs})
        policy_details = {
            "selected_pair_count": int(len(selected_pairs)),
            "selected_pair_indices": [int(value) for value in policy_pair_indices],
            "selected_frame_count": int(len(selected_frames)),
            "selected_frame_indices": selected_frames,
        }
    else:  # pragma: no cover - guarded by parser
        raise AssertionError(f"unexpected repair policy kind {policy_kind!r}")

    selected_pixels = sum(int(run.length) for run in selected_runs)
    selection_meta = {
        "strategy": "alpha_mask_replacement_archive_selected_repair_policy_v1",
        "policy_name": policy_name,
        "policy_kind": policy_kind,
        "policy_classes": [int(value) for value in policy_classes],
        "policy_details": policy_details,
        "total_residual_pixels": total_pixels,
        "total_residual_runs": total_runs,
        "selected_repair_pixels": int(selected_pixels),
        "selected_repair_runs": int(len(selected_runs)),
        "residual_pixel_coverage": 1.0 if total_pixels == 0 else round(selected_pixels / total_pixels, 12),
        "partial_repair": bool(selected_pixels != total_pixels),
        "fail_on_partial_repair": False,
        "source_manifest_selection": full_header.get("selection", {}),
    }
    raw_payload = alpha_builder._encode_repair_payload(
        selected_runs,
        shape=tuple(int(value) for value in full_header["shape"]),
        source_mask_sha256=str(full_header["source_mask_u8_sha256"]),
        candidate_mask_sha256=str(full_header["candidate_mask_u8_sha256"]),
        selection_meta=selection_meta,
    )
    member_name, member_bytes = _compress_repair_payload(raw_payload, repair_compressor)
    meta = {
        "source_artifact": repair_record,
        "policy": selection_meta,
        "raw_amr1_size_bytes": int(len(raw_payload)),
        "raw_amr1_sha256": _sha256_bytes(raw_payload),
        "archive_member": member_name,
        "compressor": repair_compressor,
        "compressed_size_bytes": int(len(member_bytes)),
        "compressed_sha256": _sha256_bytes(member_bytes),
    }
    return member_name, member_bytes, meta


def _zip_info(name: str) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(_safe_member_name(name))
    zi.date_time = FIXED_DT
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = (0o644 & 0xFFFF) << 16
    return zi


def _write_archive(output: Path, members: list[tuple[str, bytes]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in members:
            zf.writestr(_zip_info(name), data)


def build_archive(
    *,
    anchor_archive: Path,
    candidate_manifest_path: Path,
    output: Path,
    provenance_json: Path | None,
    repair_policy: str = "none",
    repair_compressor: str = "raw",
    repair_frame_group_size: int = 50,
) -> dict[str, Any]:
    candidate_manifest_path = candidate_manifest_path.resolve()
    candidate_manifest = _load_candidate_manifest(candidate_manifest_path)
    grayscale = _find_grayscale_artifact(
        candidate_manifest,
        manifest_dir=candidate_manifest_path.parent,
    )
    anchor_members, anchor_inventory = _read_archive_members(anchor_archive)
    for required in REQUIRED_ANCHOR_MEMBERS:
        if required not in anchor_members:
            raise FileNotFoundError(f"anchor archive missing required member {required!r}")
    if "masks.mkv" not in anchor_members:
        raise FileNotFoundError("anchor archive missing masks.mkv custody baseline")
    expected_mask_sha = _candidate_source_mask_sha(candidate_manifest)
    actual_mask_sha = _sha256_bytes(anchor_members["masks.mkv"])
    if expected_mask_sha is not None and expected_mask_sha != actual_mask_sha:
        raise ValueError(
            "candidate manifest source mask sha256 does not match anchor masks.mkv: "
            f"{expected_mask_sha} != {actual_mask_sha}"
        )

    repair_member = _build_selected_repair_member(
        candidate_manifest=candidate_manifest,
        manifest_dir=candidate_manifest_path.parent,
        repair_policy=repair_policy,
        repair_compressor=repair_compressor,
        repair_frame_group_size=repair_frame_group_size,
    )

    grayscale_bytes = Path(grayscale["resolved_path"]).read_bytes()
    output_members = [
        ("renderer.bin", anchor_members["renderer.bin"]),
        ("grayscale.mkv", grayscale_bytes),
    ]
    if repair_member is not None:
        output_members.append((repair_member[0], repair_member[1]))
    output_members.append(("optimized_poses.bin", anchor_members["optimized_poses.bin"]))
    _write_archive(output, output_members)

    output_inventory = [
        {
            "name": name,
            "size_bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        for name, data in output_members
    ]
    output_sha = _sha256_file(output)
    output_bytes = output.stat().st_size
    anchor_sha = _sha256_file(anchor_archive)
    anchor_bytes = anchor_archive.stat().st_size
    mask_bytes = len(anchor_members["masks.mkv"])
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "experiments/build_alpha_mask_replacement_archive.py",
        "score_claim": False,
        "promotion_eligible": False,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "archive": {
            "path": str(output.resolve()),
            "size_bytes": output_bytes,
            "sha256": output_sha,
            "members": output_inventory,
            "delta_vs_anchor_archive_bytes": output_bytes - anchor_bytes,
            "rate_term_delta_vs_anchor": 25.0 * (output_bytes - anchor_bytes) / 37_545_489,
        },
        "anchor": {
            "archive_path": str(anchor_archive.resolve()),
            "archive_size_bytes": anchor_bytes,
            "archive_sha256": anchor_sha,
            "member_inventory": anchor_inventory,
            "reused_members": [
                {"name": "renderer.bin", "sha256": _sha256_bytes(anchor_members["renderer.bin"])},
                {"name": "optimized_poses.bin", "sha256": _sha256_bytes(anchor_members["optimized_poses.bin"])},
            ],
            "replaced_member": {
                "name": "masks.mkv",
                "size_bytes": mask_bytes,
                "sha256": _sha256_bytes(anchor_members["masks.mkv"]),
            },
        },
        "candidate": {
            "manifest_path": str(candidate_manifest_path),
            "manifest_sha256": _sha256_file(candidate_manifest_path),
            "grayscale_artifact": grayscale,
            "residual_repair": repair_member[2] if repair_member is not None else None,
            "agreement_before_repair": candidate_manifest.get("candidate", {})
            .get("alpha4", {})
            .get("agreement_before_repair"),
            "crf": candidate_manifest.get("candidate", {}).get("alpha4", {}).get("crf"),
        },
        "runtime_contract": {
            "archive_members": [name for name, _ in output_members],
            "inflate_auto_dispatch": (
                "submissions/robust_current/inflate.sh sets "
                "PYTHON_INFLATE=renderer_grayscale when grayscale.mkv exists "
                "and masks.mkv is absent"
            ),
            "optional_residual_repair": (
                "inflate_renderer_grayscale.py applies alpha4_residual_repair.amr1 "
                "or supported compressed variants before re-encoding legacy masks.mkv"
            ),
            "expected_inflate_mode": "renderer_grayscale",
        },
    }
    if provenance_json is None:
        provenance_json = output.with_name("alpha_mask_replacement_archive_manifest.json")
    provenance_json.parent.mkdir(parents=True, exist_ok=True)
    provenance_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-archive", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance-json", type=Path, default=None)
    parser.add_argument(
        "--repair-policy",
        default="none",
        help="Residual repair policy: none, full, or class_prefix_<ids> (for example class_prefix_2).",
    )
    parser.add_argument(
        "--repair-compressor",
        choices=REPAIR_COMPRESSORS,
        default="raw",
        help="Compression for selected residual repair payload.",
    )
    parser.add_argument(
        "--repair-frame-group-size",
        type=int,
        default=50,
        help="Frame-group size for top_residual_frame_groups_* repair policies.",
    )
    args = parser.parse_args(argv)

    payload = build_archive(
        anchor_archive=args.anchor_archive,
        candidate_manifest_path=args.candidate_manifest,
        output=args.output,
        provenance_json=args.provenance_json,
        repair_policy=args.repair_policy,
        repair_compressor=args.repair_compressor,
        repair_frame_group_size=args.repair_frame_group_size,
    )
    archive = payload["archive"]
    candidate = payload["candidate"]
    print(
        "[alpha-mask-replacement] "
        f"wrote {archive['path']} "
        f"({archive['size_bytes']:,}B sha256={archive['sha256']}) "
        f"delta_vs_anchor={archive['delta_vs_anchor_archive_bytes']:+,}B "
        f"crf={candidate.get('crf')} "
        f"agreement={candidate.get('agreement_before_repair', {}).get('argmax_agreement')} "
        f"repair_policy={args.repair_policy} repair_compressor={args.repair_compressor}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
