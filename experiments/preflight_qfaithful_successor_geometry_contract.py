#!/usr/bin/env python3
"""Local Q-FAITHFUL successor pose/geometry dispatch gate.

This preflight is intentionally local-only. It inspects candidate provenance
and archive bytes to decide whether a future long Q-FAITHFUL training burn is
allowed by the pose/geometry contract. It never runs scorers, never dispatches
remote work, and never makes a score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/preflight_qfaithful_successor_geometry_contract.py"
SCHEMA = "qfaithful_successor_geometry_contract_preflight_v1"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/qfaithful_successor_geometry_contract_20260503/"
    "qfaithful_successor_geometry_contract_preflight.json"
)
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
ACCEPTED_RENDERER_MAGICS = {b"QFAI": "QFAI", b"QZS3": "QZS3"}
PACKED_PAYLOAD_CONTAINERS = ("p", "renderer_payload.bin", "renderer_payload.bin.br")
GEOMETRY_MEMBERS = ("zoom_scalars.bin", "foveation_params.bin")
POSE_DIM = 6
POSE_ROW_BYTES = POSE_DIM * 2


class PreflightError(RuntimeError):
    """Classified local preflight failure."""


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise PreflightError(f"{path} did not contain a JSON object")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in value
    )


def _walk(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = [(path, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            out.extend(_walk(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            out.extend(_walk(child, f"{path}[{idx}]"))
    return out


def _dicts(payloads: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for payload in payloads:
        for path, value in _walk(payload):
            if isinstance(value, dict):
                rows.append((path, value))
    return rows


def _bool_values(payloads: list[dict[str, Any]], keys: set[str]) -> list[tuple[str, bool]]:
    values: list[tuple[str, bool]] = []
    for payload in payloads:
        for path, value in _walk(payload):
            if path.rsplit(".", 1)[-1] in keys and isinstance(value, bool):
                values.append((path, value))
    return values


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _shape_pair_count(value: Any) -> int | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _int_or_none(value[0])
    return None


def _shape_pose_dim(value: Any) -> int | None:
    if isinstance(value, (list, tuple)) and value:
        return _int_or_none(value[-1])
    return None


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_qfaithful_successor_preflight_unpacker",
        UNPACKER_PATH,
    )
    if spec is None or spec.loader is None:
        raise PreflightError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_zip_name(name: str) -> str | None:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts:
        return f"unsafe_archive_member:{name}"
    if name.startswith("__MACOSX") or any(part.startswith(".") for part in path.parts):
        return f"hidden_archive_member:{name}"
    return None


def _validate_renderer_wire(data: bytes) -> dict[str, Any]:
    magic = data[:4]
    wire_format = ACCEPTED_RENDERER_MAGICS.get(magic)
    result: dict[str, Any] = {
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "magic_hex": magic.hex(),
        "wire_format": wire_format,
        "runtime_readable_wire_contract": False,
        "blockers": [],
    }
    blockers: list[str] = result["blockers"]
    if wire_format is None:
        blockers.append("renderer_wire_format_not_qfai_or_qzs3")
        return result
    if wire_format == "QZS3":
        if len(data) < 6:
            blockers.append("qzs3_payload_too_short")
            return result
        block_size = int.from_bytes(data[4:6], "little")
        result["qzs3_block_size"] = block_size
        if block_size <= 0:
            blockers.append("qzs3_block_size_invalid")
            return result
    elif wire_format == "QFAI":
        if len(data) < 8:
            blockers.append("qfai_header_missing")
            return result
        header_len = struct.unpack_from("<I", data, 4)[0]
        header_end = 8 + header_len
        if header_len <= 0 or header_end > len(data):
            blockers.append("qfai_header_length_invalid")
            return result
        try:
            header = json.loads(data[8:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            blockers.append("qfai_header_unreadable")
            return result
        result["qfai_header"] = header
        if header.get("format") != "QFAI":
            blockers.append("qfai_header_format_invalid")
            return result
    result["runtime_readable_wire_contract"] = True
    return result


def _pose_bytes_summary(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "pose_dim": POSE_DIM,
        "pair_count": None,
        "nonzero_elements": 0,
        "all_zero": True,
        "finite": False,
        "promotable_pose_stream": False,
        "blockers": [],
    }
    blockers: list[str] = result["blockers"]
    if not data:
        blockers.append("archive_pose_stream_empty")
        return result
    if len(data) % POSE_ROW_BYTES != 0:
        blockers.append("archive_pose_stream_not_fp16_6d")
        return result
    values: list[float] = []
    try:
        for (value,) in struct.iter_unpack("<e", data):
            values.append(float(value))
    except struct.error as exc:
        blockers.append(f"archive_pose_stream_unreadable:{exc}")
        return result
    nonzero = sum(1 for value in values if value != 0.0)
    finite = all(math.isfinite(value) for value in values)
    result.update(
        {
            "pair_count": len(data) // POSE_ROW_BYTES,
            "numel": len(values),
            "nonzero_elements": nonzero,
            "all_zero": nonzero == 0,
            "finite": finite,
        }
    )
    if result["pair_count"] <= 0:
        blockers.append("archive_pose_pair_count_missing")
    if nonzero == 0:
        blockers.append("archive_pose_stream_all_zero")
    if not finite:
        blockers.append("archive_pose_stream_nonfinite")
    result["promotable_pose_stream"] = not blockers
    return result


def inspect_archive(archive: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _repo_rel(archive),
        "present": archive is not None and archive.is_file(),
        "archive_sha256": None,
        "archive_bytes": None,
        "zip_members": {},
        "logical_members": {},
        "packed_payload_container": None,
        "renderer_wire": None,
        "pose_stream": None,
        "geometry_members": {},
        "runtime_readable_output_contract": False,
        "blockers": [],
    }
    blockers: list[str] = result["blockers"]
    if archive is None or not archive.is_file():
        blockers.append("archive_missing")
        return result
    result["archive_sha256"] = _sha256_file(archive)
    result["archive_bytes"] = archive.stat().st_size

    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                blockers.append("archive_duplicate_member_names")
            for name in names:
                failure = _validate_zip_name(name)
                if failure:
                    blockers.append(failure)
            zip_members = {name: zf.read(name) for name in names if name in set(names)}
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        blockers.append(f"archive_unreadable:{exc}")
        return result

    result["zip_members"] = {
        name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(zip_members.items())
    }
    containers = [name for name in PACKED_PAYLOAD_CONTAINERS if name in zip_members]
    if len(containers) > 1:
        blockers.append("multiple_packed_renderer_payload_containers")
    if containers and "renderer.bin" in zip_members:
        blockers.append("packed_and_direct_renderer_members_ambiguous")

    logical_members = dict(zip_members)
    if len(containers) == 1 and not blockers:
        result["packed_payload_container"] = containers[0]
        try:
            with tempfile.TemporaryDirectory(prefix="qfaithful_preflight_unpack_") as tmp_raw:
                tmp = Path(tmp_raw)
                for name, data in zip_members.items():
                    failure = _validate_zip_name(name)
                    if failure:
                        raise PreflightError(failure)
                    (tmp / name).write_bytes(data)
                unpack_summary = _load_unpacker().unpack_renderer_payload(tmp)
                result["unpack_summary"] = unpack_summary
                for member in ("renderer.bin", "masks.mkv", "grayscale.mkv", "optimized_poses.bin"):
                    member_path = tmp / member
                    if member_path.exists():
                        logical_members[member] = member_path.read_bytes()
        except Exception as exc:
            blockers.append(f"packed_payload_unreadable:{exc}")

    result["logical_members"] = {
        name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(logical_members.items())
    }
    result["geometry_members"] = {
        name: {"bytes": len(logical_members[name]), "sha256": _sha256_bytes(logical_members[name])}
        for name in GEOMETRY_MEMBERS
        if name in logical_members
    }

    renderer = logical_members.get("renderer.bin")
    if renderer is None:
        blockers.append("renderer_bin_missing_after_unpack")
    else:
        result["renderer_wire"] = _validate_renderer_wire(renderer)
        blockers.extend(result["renderer_wire"]["blockers"])

    poses = logical_members.get("optimized_poses.bin")
    if poses is None:
        blockers.append("optimized_poses_bin_missing_after_unpack")
    else:
        result["pose_stream"] = _pose_bytes_summary(poses)
        blockers.extend(result["pose_stream"]["blockers"])

    result["runtime_readable_output_contract"] = not blockers
    return result


def check_no_score_claim(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    score_claims = _bool_values(payloads, {"score_claim", "promotable_score_claim"})
    promotion_flags = _bool_values(payloads, {"promotion_eligible"})
    blockers: list[str] = []
    if any(value for _path, value in score_claims):
        blockers.append("score_claim_true")
    if any(value for _path, value in promotion_flags):
        blockers.append("promotion_eligible_true")
    if not any(path.endswith(".score_claim") and value is False for path, value in score_claims):
        blockers.append("score_claim_false_not_proven")
    return {
        "passed": not blockers,
        "score_claim_values": [{"path": path, "value": value} for path, value in score_claims],
        "promotion_values": [{"path": path, "value": value} for path, value in promotion_flags],
        "blockers": blockers,
    }


def check_eval_roundtrip(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    values = _bool_values(payloads, {"eval_roundtrip", "eval_roundtrip_required"})
    blockers: list[str] = []
    if any(value is False for _path, value in values):
        blockers.append("eval_roundtrip_false")
    if not any(value is True for _path, value in values):
        blockers.append("eval_roundtrip_true_not_proven")
    return {
        "passed": not blockers,
        "values": [{"path": path, "value": value} for path, value in values],
        "blockers": blockers,
    }


def _pose_contract_candidates(payloads: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    wanted = {
        "training_pose_contract",
        "qfaithful_training_pose_contract",
        "checkpoint_training_pose_contract",
        "snapshot_training_pose_contract",
        "pose_tensor_contract",
    }
    candidates: list[tuple[str, dict[str, Any]]] = []
    for path, mapping in _dicts(payloads):
        if any(part in wanted for part in path.split(".")):
            candidates.append((path, mapping))
            nested = mapping.get("contract")
            if isinstance(nested, dict):
                candidates.append((f"{path}.contract", nested))
        elif {
            "pose_sha256",
            "pose_source_sha256",
            "training_uses_nonzero_pose_stream",
            "zero_pose_fallback_allowed",
        } & set(mapping):
            candidates.append((path, mapping))
    return candidates


def _contract_uses_nonzero_pose(contract: dict[str, Any]) -> bool:
    for key in (
        "training_uses_nonzero_pose_stream",
        "uses_nonzero_pose_stream",
        "training_uses_deployed_pose_stream",
    ):
        if contract.get(key) is True:
            return True
    if _int_or_none(contract.get("nonzero_elements")) and contract.get("all_zero") is False:
        return True
    return False


def _contract_zero_fallback_forbidden(contract: dict[str, Any]) -> bool:
    for key in ("zero_pose_fallback_allowed", "silent_zero_fallback_allowed"):
        if key in contract:
            return contract.get(key) is False
    return False


def _pose_contract_summary(path: str, contract: dict[str, Any]) -> dict[str, Any]:
    shape = _first_present(contract, ("shape", "pose_shape"))
    pose_dim = _int_or_none(
        _first_present(contract, ("pose_dim", "required_pose_dim"))
    )
    if pose_dim is None:
        pose_dim = _shape_pose_dim(shape)
    pair_count = _int_or_none(
        _first_present(
            contract,
            ("pair_count", "pose_pair_count", "rows", "n_pairs", "num_pairs"),
        )
    )
    if pair_count is None:
        pair_count = _shape_pair_count(shape)
    pose_sha = _first_present(
        contract,
        (
            "pose_sha256",
            "pose_source_sha256",
            "training_pose_sha256",
            "deployed_pose_sha256",
            "optimized_poses_bin_sha256",
            "sha256",
        ),
    )
    blockers: list[str] = []
    if pose_dim != POSE_DIM:
        blockers.append("training_pose_dim_not_6")
    if pair_count is None or pair_count <= 0:
        blockers.append("training_pose_pair_count_missing")
    if not _is_sha256(pose_sha):
        blockers.append("training_pose_sha256_missing")
    if not _contract_uses_nonzero_pose(contract):
        blockers.append("training_nonzero_pose_stream_not_proven")
    if not _contract_zero_fallback_forbidden(contract):
        blockers.append("zero_pose_fallback_not_forbidden")
    return {
        "source_path": path,
        "pose_dim": pose_dim,
        "pair_count": pair_count,
        "pose_sha256": pose_sha if _is_sha256(pose_sha) else None,
        "training_uses_nonzero_pose_stream": _contract_uses_nonzero_pose(contract),
        "zero_pose_fallback_allowed": False
        if _contract_zero_fallback_forbidden(contract)
        else None,
        "passed": not blockers,
        "blockers": blockers,
        "contract": contract,
    }


def check_training_pose_contract(
    payloads: list[dict[str, Any]],
    archive_info: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        _pose_contract_summary(path, contract)
        for path, contract in _pose_contract_candidates(payloads)
    ]
    candidates.sort(key=lambda item: (len(item["blockers"]), item["source_path"]))
    selected = candidates[0] if candidates else None
    blockers: list[str] = []
    if selected is None:
        blockers.append("training_pose_contract_missing")
    else:
        blockers.extend(selected["blockers"])

    archive_pose = archive_info.get("pose_stream") or {}
    archive_pose_sha = archive_pose.get("sha256")
    if archive_pose and archive_pose.get("promotable_pose_stream") is not True:
        blockers.append("archive_pose_stream_not_promotable")
    if selected and selected.get("pose_sha256") and archive_pose_sha:
        if selected["pose_sha256"] != archive_pose_sha:
            blockers.append("training_pose_sha_does_not_match_archive_pose_stream")
    if selected and selected.get("pair_count") and archive_pose.get("pair_count"):
        if int(selected["pair_count"]) != int(archive_pose["pair_count"]):
            blockers.append("training_pose_pair_count_does_not_match_archive")

    return {
        "passed": not blockers,
        "selected_contract": selected,
        "candidate_count": len(candidates),
        "archive_pose_sha256": archive_pose_sha,
        "archive_pair_count": archive_pose.get("pair_count"),
        "blockers": sorted(set(blockers)),
    }


def _mask_frame_contract(payloads: list[dict[str, Any]]) -> str:
    for _payload in payloads:
        for path, value in _walk(_payload):
            if path.endswith(".mask_frame_contract"):
                if isinstance(value, str):
                    return value.lower()
                if isinstance(value, dict) and isinstance(value.get("contract"), str):
                    return str(value["contract"]).lower()
    return "unknown"


def _configured_geometry_names(
    payloads: list[dict[str, Any]],
    archive_info: dict[str, Any],
) -> list[str]:
    names: set[str] = set(archive_info.get("geometry_members", {}))
    for payload in payloads:
        for _path, value in _walk(payload):
            if isinstance(value, str):
                if value in GEOMETRY_MEMBERS:
                    names.add(value)
                lowered = value.lower()
                if "zoom" in lowered:
                    names.add("zoom_scalars.bin")
                if "foveation" in lowered:
                    names.add("foveation_params.bin")
            elif isinstance(value, dict):
                member_name = value.get("member_name") or value.get("name")
                if member_name in GEOMETRY_MEMBERS:
                    names.add(str(member_name))
    return sorted(names)


def _any_true(payloads: list[dict[str, Any]], keys: set[str]) -> bool:
    return any(value is True for _path, value in _bool_values(payloads, keys))


def check_geometry_contract(
    payloads: list[dict[str, Any]],
    archive_info: dict[str, Any],
) -> dict[str, Any]:
    mask_contract = _mask_frame_contract(payloads)
    configured = _configured_geometry_names(payloads, archive_info)
    archive_geometry = archive_info.get("geometry_members", {})
    blockers: list[str] = []

    if mask_contract == "half" and not configured:
        blockers.append("half_frame_masks_without_zoom_warp_geometry")
    for name in configured:
        if name not in archive_geometry:
            blockers.append(f"{name}_not_charged_in_archive")

    zoom_consumed = _any_true(
        payloads,
        {
            "geometry_consumed_by_runtime",
            "geometry_consumption_proven",
            "runtime_consumes_zoom_warp_for_mask_expansion",
            "renderer_consumes_ego_flow",
            "consumes_zoom_warp",
            "zoom_warp_consumed_by_runtime",
        },
    )
    foveation_consumed = _any_true(
        payloads,
        {
            "geometry_consumed_by_runtime",
            "geometry_consumption_proven",
            "consumes_foveation_params",
            "foveation_consumed_by_runtime",
        },
    )
    if "zoom_scalars.bin" in configured and not zoom_consumed:
        blockers.append("zoom_warp_geometry_not_consumed_by_runtime")
    if "foveation_params.bin" in configured and not foveation_consumed:
        blockers.append("foveation_geometry_not_consumed_by_runtime")

    return {
        "passed": not blockers,
        "mask_frame_contract": mask_contract,
        "configured_geometry_members": configured,
        "archive_geometry_members": archive_geometry,
        "zoom_geometry_consumed": zoom_consumed,
        "foveation_geometry_consumed": foveation_consumed,
        "blockers": sorted(set(blockers)),
    }


def build_report(
    *,
    provenance_paths: tuple[Path, ...],
    archive: Path | None,
    label: str,
) -> dict[str, Any]:
    provenance_payloads = [_read_json(path) for path in provenance_paths]
    archive_info = inspect_archive(archive)
    checks = {
        "no_score_claim": check_no_score_claim(provenance_payloads),
        "eval_roundtrip": check_eval_roundtrip(provenance_payloads),
        "training_pose_contract": check_training_pose_contract(
            provenance_payloads,
            archive_info,
        ),
        "geometry_contract": check_geometry_contract(provenance_payloads, archive_info),
        "archive_output_contract": {
            "passed": archive_info.get("runtime_readable_output_contract") is True,
            "blockers": list(archive_info.get("blockers", [])),
        },
    }
    blockers = sorted(
        {
            blocker
            for check in checks.values()
            for blocker in check.get("blockers", [])
        }
    )
    allowed = not blockers
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "label": label,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "training_dispatch_allowed": allowed,
        "verdict": "local_contract_pass_no_score_claim" if allowed else "blocked_no_h100_dispatch",
        "required_next_step_if_allowed": (
            "claim lane before any training/eval dispatch; exact CUDA auth eval remains "
            "required before any score claim"
            if allowed
            else None
        ),
        "provenance_paths": [_repo_rel(path) for path in provenance_paths],
        "archive": archive_info,
        "checks": checks,
        "blockers": blockers,
    }


def write_report(
    output: Path,
    *,
    provenance_paths: tuple[Path, ...],
    archive: Path | None,
    label: str,
) -> dict[str, Any]:
    report = build_report(
        provenance_paths=provenance_paths,
        archive=archive,
        label=label,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance",
        action="append",
        type=Path,
        required=True,
        help="Candidate provenance JSON. May be supplied multiple times.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Exact candidate archive.zip to inspect.",
    )
    parser.add_argument("--label", default="qfaithful_successor_geometry_contract")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = write_report(
        args.output,
        provenance_paths=tuple(args.provenance),
        archive=args.archive,
        label=args.label,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
