#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Alpha INR/TinyNeRV readiness audit.

This is a deterministic, local-only scaffold for the next Alpha mask
replacement step. It validates archive custody and decoded-mask source identity,
then records byte/shape accounting for an untrained TinyNeRV/INR prototype.

It does not train, build a score archive, load scorer networks, launch remote
jobs, or make a score/promotion claim. Any future candidate still requires the
canonical CUDA path:

    archive.zip -> inflate.sh -> upstream/evaluate.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.nerv_mask_codec import (
    NERV_VERSION,
    NeRVMaskCodec,
    encode_nerv_codec,
    raw_fp16_baseline_bytes,
)

SCHEMA = "alpha_inr_readiness_v1"
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this readiness audit. A deterministic candidate "
    "archive and exact CUDA auth eval are required before any score claim, "
    "promotion, ranking, retirement, or stack math."
)
DEFAULT_BASELINE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
)
DEFAULT_OUTPUT = REPO_ROOT / "experiments/results/alpha_inr_readiness/alpha_inr_readiness.json"
REQUIRED_BASELINE_MEMBERS = ("renderer.bin", "masks.mkv", "optimized_poses.bin")
CLASS_IDS = (0, 1, 2, 3, 4)


class AlphaINRReadinessError(ValueError):
    """Fail-closed readiness audit error."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _safe_member_parts(name: str) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name:
        raise AlphaINRReadinessError(f"unsafe archive member path: {name!r}")
    member_path = PurePosixPath(name)
    parts = member_path.parts
    if member_path.is_absolute() or ".." in parts:
        raise AlphaINRReadinessError(f"unsafe archive member path: {name!r}")
    if not parts or any(part in ("", ".") for part in parts):
        raise AlphaINRReadinessError(f"unsafe archive member path: {name!r}")
    first = parts[0]
    if len(first) == 2 and first[1] == ":":
        raise AlphaINRReadinessError(f"unsafe archive member path: {name!r}")
    return parts


def _reject_hidden_or_system_member(name: str, parts: tuple[str, ...]) -> None:
    if "__MACOSX" in parts:
        raise AlphaINRReadinessError(f"hidden/system archive member: {name!r}")
    if any(part in {".DS_Store", "Thumbs.db"} for part in parts):
        raise AlphaINRReadinessError(f"hidden/system archive member: {name!r}")
    if any(part.startswith(".") or part.startswith("._") for part in parts):
        raise AlphaINRReadinessError(f"hidden/system archive member: {name!r}")


def _audit_archive(archive: Path, *, required_members: tuple[str, ...]) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"baseline archive not found: {archive}")
    archive_meta = _file_meta(archive)
    members: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        seen: set[str] = set()
        for info in zf.infolist():
            name = info.filename
            if name in seen:
                raise AlphaINRReadinessError(f"duplicate archive member: {name!r}")
            seen.add(name)
            if info.is_dir():
                raise AlphaINRReadinessError(f"unexpected archive directory member: {name!r}")
            parts = _safe_member_parts(name)
            _reject_hidden_or_system_member(name, parts)
            data = zf.read(info)
            record = {
                "name": name,
                "size_bytes": int(info.file_size),
                "compressed_size_bytes": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "sha256": _sha256_bytes(data),
            }
            members[name] = record
            inventory.append(record)

    missing = [name for name in required_members if name not in members]
    if missing:
        raise AlphaINRReadinessError(
            "baseline archive missing required member(s): " + ", ".join(missing)
        )
    archive_meta.update(
        {
            "validated_zip_safety": True,
            "required_members": list(required_members),
            "member_inventory": sorted(inventory, key=lambda item: item["name"]),
            "members": members,
        }
    )
    return archive_meta


def _load_tensor_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        for key in ("masks", "decoded_masks", "mask_classes", "class_ids", "candidate_masks"):
            if key in data:
                return data[key]
        raise AlphaINRReadinessError(
            f"{path} has no masks/decoded_masks/mask_classes/class_ids/candidate_masks array"
        )
    if suffix in {".pt", ".pth"}:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            for key in ("masks", "decoded_masks", "mask_classes", "class_ids", "candidate_masks"):
                value = obj.get(key)
                if isinstance(value, torch.Tensor | np.ndarray):
                    return value
        raise AlphaINRReadinessError(f"{path} did not contain a decoded mask tensor")
    raise AlphaINRReadinessError(
        f"unsupported decoded mask source suffix {path.suffix!r}; use .npy, .npz, .pt, or .pth"
    )


def _normalize_masks(value: Any, *, num_classes: int) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise AlphaINRReadinessError(f"decoded masks must have shape (T,H,W); got {tuple(array.shape)}")
    if array.size == 0:
        raise AlphaINRReadinessError("decoded masks must be non-empty")
    if np.issubdtype(array.dtype, np.floating):
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise AlphaINRReadinessError("decoded masks contain non-integer floating values")
        array = rounded
    if not np.issubdtype(array.dtype, np.integer):
        raise AlphaINRReadinessError(f"decoded masks must be integer class IDs; got {array.dtype}")
    min_value = int(array.min())
    max_value = int(array.max())
    if min_value < 0 or max_value >= int(num_classes):
        raise AlphaINRReadinessError(
            f"decoded masks class IDs must be in [0,{int(num_classes) - 1}], got [{min_value},{max_value}]"
        )
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def _mask_shape_dtype_raw_sha256(masks: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(int(v) for v in masks.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(masks.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(np.ascontiguousarray(masks)))
    return digest.hexdigest()


def _load_decoded_masks_source(path: Path, *, num_classes: int) -> tuple[np.ndarray, dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"decoded mask source not found: {path}")
    masks = _normalize_masks(_load_tensor_file(path), num_classes=num_classes)
    t, h, w = (int(v) for v in masks.shape)
    meta = {
        **_file_meta(path),
        "source_kind": "decoded_masks_source",
        "shape": [t, h, w],
        "dtype": str(masks.dtype),
        "num_classes": int(num_classes),
        "num_pixels": int(masks.size),
        "raw_u8_bytes": int(masks.nbytes),
        "class_id_u8_sha256": _sha256_bytes(masks.tobytes()),
        "shape_dtype_raw_sha256": _mask_shape_dtype_raw_sha256(masks),
        "class_histogram": {
            str(class_id): int(np.count_nonzero(masks == class_id)) for class_id in range(int(num_classes))
        },
    }
    return masks, meta


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AlphaINRReadinessError(f"{path}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AlphaINRReadinessError(f"{path}: expected a JSON object")
    return payload


def _require_false(payload: dict[str, Any], path: tuple[str, ...]) -> None:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            raise AlphaINRReadinessError(f"manifest missing {'.'.join(path)}")
        cursor = cursor[key]
    if cursor is not False:
        raise AlphaINRReadinessError(f"manifest {'.'.join(path)} must be false")


def _load_candidate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    payload = _read_json(path)
    if payload.get("schema") != "alpha_mask_candidate_builder_v1":
        raise AlphaINRReadinessError(
            f"candidate manifest schema must be 'alpha_mask_candidate_builder_v1'; got {payload.get('schema')!r}"
        )
    _require_false(payload, ("score_claim",))
    _require_false(payload, ("promotion_eligible",))
    _require_false(payload, ("scorer_network_loaded",))
    _require_false(payload, ("candidate", "score_claim"))
    _require_false(payload, ("candidate", "promotion_eligible"))
    if payload.get("evidence_grade") != EVIDENCE_GRADE:
        raise AlphaINRReadinessError(f"candidate manifest evidence_grade must be {EVIDENCE_GRADE!r}")
    if "contest_auth_eval.py --device cuda" not in payload.get("canonical_score_source_required", ""):
        raise AlphaINRReadinessError("candidate manifest must require exact CUDA auth eval")
    return payload, {
        **_file_meta(path),
        "schema": payload.get("schema"),
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
    }


def _field(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _gate(
    *,
    passed: bool,
    required_for: str,
    evidence: str,
    expected: Any = None,
    observed: Any = None,
    skipped: bool = False,
    blocker: bool | None = None,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "required_for": required_for,
        "evidence": evidence,
        "expected": expected,
        "observed": observed,
        "skipped": bool(skipped),
        "blocker": (not passed and not skipped) if blocker is None else bool(blocker),
    }


def _validate_candidate_manifest_identity(
    *,
    manifest: dict[str, Any],
    archive_meta: dict[str, Any],
    decoded_meta: dict[str, Any],
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise AlphaINRReadinessError("candidate manifest missing source object")

    manifest_archive_sha = source.get("archive_sha256")
    gates["source_archive_sha256_match"] = _gate(
        passed=manifest_archive_sha == archive_meta["sha256"],
        required_for="training_source_identity",
        evidence="candidate manifest source.archive_sha256 vs baseline archive file",
        expected=archive_meta["sha256"],
        observed=manifest_archive_sha,
    )
    if not gates["source_archive_sha256_match"]["passed"]:
        raise AlphaINRReadinessError("candidate manifest source archive sha256 mismatch")

    manifest_archive_size = source.get("archive_size_bytes")
    gates["source_archive_size_match"] = _gate(
        passed=int(manifest_archive_size) == int(archive_meta["size_bytes"]),
        required_for="training_source_identity",
        evidence="candidate manifest source.archive_size_bytes vs baseline archive file",
        expected=int(archive_meta["size_bytes"]),
        observed=manifest_archive_size,
    )
    if not gates["source_archive_size_match"]["passed"]:
        raise AlphaINRReadinessError("candidate manifest source archive size mismatch")

    mask_member = source.get("mask_member")
    if not isinstance(mask_member, dict):
        raise AlphaINRReadinessError("candidate manifest missing source.mask_member object")
    member_name = str(mask_member.get("name", ""))
    archive_member = archive_meta["members"].get(member_name)
    gates["source_mask_member_present"] = _gate(
        passed=archive_member is not None,
        required_for="training_source_identity",
        evidence="candidate manifest source.mask_member.name exists in baseline archive",
        expected=sorted(archive_meta["members"]),
        observed=member_name,
    )
    if archive_member is None:
        raise AlphaINRReadinessError("candidate manifest source mask member not present in archive")

    for field_name in ("size_bytes", "sha256"):
        expected = archive_member[field_name]
        observed = mask_member.get(field_name)
        gates[f"source_mask_member_{field_name}_match"] = _gate(
            passed=observed == expected,
            required_for="training_source_identity",
            evidence=f"candidate manifest source.mask_member.{field_name} vs archive member",
            expected=expected,
            observed=observed,
        )
        if not gates[f"source_mask_member_{field_name}_match"]["passed"]:
            raise AlphaINRReadinessError(f"candidate manifest source mask member {field_name} mismatch")

    manifest_decoded = source.get("decoded_masks")
    if not isinstance(manifest_decoded, dict):
        raise AlphaINRReadinessError("candidate manifest missing source.decoded_masks object")
    manifest_shape = manifest_decoded.get("shape")
    gates["decoded_mask_shape_match_manifest"] = _gate(
        passed=manifest_shape == decoded_meta["shape"],
        required_for="training_source_identity",
        evidence="candidate manifest source.decoded_masks.shape vs decoded mask source",
        expected=decoded_meta["shape"],
        observed=manifest_shape,
    )
    if not gates["decoded_mask_shape_match_manifest"]["passed"]:
        raise AlphaINRReadinessError("decoded mask source shape mismatch against candidate manifest")

    manifest_sha = manifest_decoded.get("class_id_u8_sha256")
    gates["decoded_mask_u8_sha256_match_manifest"] = _gate(
        passed=manifest_sha == decoded_meta["class_id_u8_sha256"],
        required_for="training_source_identity",
        evidence="candidate manifest source.decoded_masks.class_id_u8_sha256 vs decoded mask source",
        expected=decoded_meta["class_id_u8_sha256"],
        observed=manifest_sha,
    )
    if not gates["decoded_mask_u8_sha256_match_manifest"]["passed"]:
        raise AlphaINRReadinessError("decoded mask source sha256 mismatch against candidate manifest")

    readiness = _field(manifest, ("candidate", "candidate_archive_readiness")) or {}
    gates["candidate_manifest_has_exact_eval_builder_gate"] = _gate(
        passed=bool(readiness.get("exact_eval_archive_builder_required")) is True
        and "contest_auth_eval.py --device cuda" in str(readiness.get("exact_cuda_auth_eval_required", "")),
        required_for="promotion_only",
        evidence="candidate manifest preserves exact archive builder and CUDA auth-eval gates",
        expected="exact_eval_archive_builder_required=true and exact_cuda_auth_eval_required contains --device cuda",
        observed=readiness,
        blocker=False,
    )
    return gates


def _validate_expected_shape(
    decoded_meta: dict[str, Any],
    *,
    expected_frames: int | None,
    expected_height: int | None,
    expected_width: int | None,
) -> dict[str, Any]:
    expected = [expected_frames, expected_height, expected_width]
    if all(value is None for value in expected):
        return _gate(
            passed=True,
            required_for="training_source_identity",
            evidence="expected shape not supplied; observed shape recorded for byte accounting",
            expected=None,
            observed=decoded_meta["shape"],
            skipped=True,
            blocker=False,
        )
    if any(value is None for value in expected):
        raise AlphaINRReadinessError(
            "--expected-frames, --expected-height, and --expected-width must be supplied together"
        )
    expected_shape = [int(expected_frames), int(expected_height), int(expected_width)]
    return _gate(
        passed=decoded_meta["shape"] == expected_shape,
        required_for="training_source_identity",
        evidence="decoded mask source shape vs explicit expected scorer geometry",
        expected=expected_shape,
        observed=decoded_meta["shape"],
    )


def _prototype_byte_accounting(
    *,
    shape: list[int],
    num_classes: int,
    num_freqs: int,
    hidden_dim: int,
    depth: int,
    seed: int,
) -> dict[str, Any]:
    frames, height, width = (int(v) for v in shape)
    codec = NeRVMaskCodec(
        num_freqs=int(num_freqs),
        hidden_dim=int(hidden_dim),
        num_classes=int(num_classes),
        depth=int(depth),
        seed=int(seed),
    )
    payloads = []
    for weight_dtype in ("fp16", "int8"):
        blob = encode_nerv_codec(codec, weight_dtype=weight_dtype, version=NERV_VERSION)
        payloads.append(
            {
                "weight_dtype": weight_dtype,
                "wire_format": f"NRV{NERV_VERSION}",
                "encoded_payload_bytes": len(blob),
                "encoded_payload_sha256": _sha256_bytes(blob),
                "untrained_prototype_only": True,
                "not_candidate_archive_payload": True,
            }
        )
    return {
        "prototype_kind": "untrained_tiny_nerv_coordinate_mlp",
        "architecture": {
            "num_freqs": int(num_freqs),
            "hidden_dim": int(hidden_dim),
            "depth": int(depth),
            "num_classes": int(num_classes),
            "seed": int(seed),
            "param_count": int(codec.num_params()),
        },
        "decoded_mask_shape": [frames, height, width],
        "raw_decoded_class_id_u8_bytes": int(frames * height * width),
        "raw_one_hot_fp16_logit_bytes": raw_fp16_baseline_bytes(frames, height, width, int(num_classes)),
        "prototype_payloads": payloads,
        "training_performed": False,
        "scorer_network_loaded": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _validate_auth_eval_json(
    *,
    contest_auth_eval_json: Path,
    candidate_archive: Path,
) -> dict[str, Any]:
    payload = _read_json(contest_auth_eval_json.resolve())
    archive_meta = _file_meta(candidate_archive.resolve())
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise AlphaINRReadinessError("contest_auth_eval JSON missing provenance object")
    errors: list[str] = []
    device = provenance.get("device")
    if device != "cuda":
        errors.append(f"provenance.device expected 'cuda', observed {device!r}")
    if provenance.get("cuda_available") is not True:
        errors.append("provenance.cuda_available must be true")
    observed_sha = provenance.get("archive_sha256") or payload.get("archive_sha256")
    if observed_sha != archive_meta["sha256"]:
        errors.append("archive_sha256 mismatch against candidate archive")
    observed_size = payload.get("archive_size_bytes") or payload.get("archive_bytes")
    if int(observed_size or -1) != int(archive_meta["size_bytes"]):
        errors.append("archive_size_bytes mismatch against candidate archive")
    if errors:
        raise AlphaINRReadinessError("contest auth eval evidence failed: " + "; ".join(errors))
    return {
        "path": str(contest_auth_eval_json.resolve()),
        "size_bytes": int(contest_auth_eval_json.stat().st_size),
        "sha256": _sha256_file(contest_auth_eval_json),
        "device": "cuda",
        "cuda_available": True,
        "archive": archive_meta,
        "validated_exact_cuda_custody": True,
    }


def audit_alpha_inr_readiness(
    *,
    baseline_archive: Path,
    decoded_masks_source: Path,
    output_json: Path | None,
    candidate_manifest: Path | None = None,
    candidate_archive: Path | None = None,
    contest_auth_eval_json: Path | None = None,
    expected_frames: int | None = None,
    expected_height: int | None = None,
    expected_width: int | None = None,
    num_classes: int = 5,
    num_freqs: int = 8,
    hidden_dim: int = 64,
    depth: int = 4,
    seed: int = 2026,
    command: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if int(num_classes) != len(CLASS_IDS):
        raise AlphaINRReadinessError(f"num_classes must be {len(CLASS_IDS)} for current Alpha masks")
    archive_meta = _audit_archive(Path(baseline_archive), required_members=REQUIRED_BASELINE_MEMBERS)
    masks, decoded_meta = _load_decoded_masks_source(Path(decoded_masks_source), num_classes=int(num_classes))
    shape_gate = _validate_expected_shape(
        decoded_meta,
        expected_frames=expected_frames,
        expected_height=expected_height,
        expected_width=expected_width,
    )
    if not shape_gate["passed"] and not shape_gate["skipped"]:
        raise AlphaINRReadinessError("decoded mask source shape does not match expected geometry")

    candidate_manifest_payload: dict[str, Any] | None = None
    candidate_manifest_meta: dict[str, Any] | None = None
    manifest_identity_gates: dict[str, Any] = {}
    if candidate_manifest is not None:
        candidate_manifest_payload, candidate_manifest_meta = _load_candidate_manifest(Path(candidate_manifest))
        manifest_identity_gates = _validate_candidate_manifest_identity(
            manifest=candidate_manifest_payload,
            archive_meta=archive_meta,
            decoded_meta=decoded_meta,
        )

    if contest_auth_eval_json is not None and candidate_archive is None:
        raise AlphaINRReadinessError("--candidate-archive is required when --contest-auth-eval-json is supplied")
    auth_eval_evidence = None
    if contest_auth_eval_json is not None and candidate_archive is not None:
        auth_eval_evidence = _validate_auth_eval_json(
            contest_auth_eval_json=Path(contest_auth_eval_json),
            candidate_archive=Path(candidate_archive),
        )

    prototype = _prototype_byte_accounting(
        shape=decoded_meta["shape"],
        num_classes=int(num_classes),
        num_freqs=int(num_freqs),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        seed=int(seed),
    )
    mask_member = archive_meta["members"]["masks.mkv"]
    byte_accounting = {
        "baseline_archive_size_bytes": int(archive_meta["size_bytes"]),
        "baseline_masks_member_size_bytes": int(mask_member["size_bytes"]),
        "baseline_masks_member_compressed_size_bytes": int(mask_member["compressed_size_bytes"]),
        "decoded_mask_raw_u8_bytes": int(decoded_meta["raw_u8_bytes"]),
        "raw_one_hot_fp16_logit_bytes": int(prototype["raw_one_hot_fp16_logit_bytes"]),
        "prototype_payloads": prototype["prototype_payloads"],
        "archive_delta_estimates_are_derivation_only": True,
    }

    gates: dict[str, Any] = {
        "baseline_archive_custody_validated": _gate(
            passed=True,
            required_for="training_source_identity",
            evidence="zip-slip safe archive inventory with required renderer/masks/poses members",
            expected=list(REQUIRED_BASELINE_MEMBERS),
            observed=sorted(archive_meta["members"]),
        ),
        "decoded_mask_source_loaded": _gate(
            passed=True,
            required_for="training_source_identity",
            evidence="decoded mask tensor source loaded from .npy/.npz/.pt/.pth without scorer networks",
            expected="integer (T,H,W) class IDs",
            observed={"shape": decoded_meta["shape"], "dtype": decoded_meta["dtype"]},
        ),
        "decoded_mask_expected_shape": shape_gate,
        "candidate_manifest_identity": _gate(
            passed=candidate_manifest is not None,
            required_for="training_source_identity",
            evidence="candidate manifest ties decoded masks to exact source archive and masks.mkv member",
            expected="alpha_mask_candidate_builder_v1 manifest",
            observed=candidate_manifest_meta["path"] if candidate_manifest_meta else None,
            skipped=candidate_manifest is None,
            blocker=candidate_manifest is not None and False,
        ),
        "no_score_claim": _gate(
            passed=True,
            required_for="all_outputs",
            evidence="top-level readiness output and prototype records are explicitly non-promotable",
            expected=False,
            observed=False,
        ),
        "runtime_integration_required": _gate(
            passed=False,
            required_for="candidate_archive",
            evidence="inflate runtime must consume a trained INR/NRV payload from archive bytes before exact eval",
            expected="trained masks.nrv or equivalent archive member plus fixed runtime path",
            observed="untrained byte-accounting prototype only",
            blocker=True,
        ),
        "exact_cuda_auth_eval_required_before_score_claim": _gate(
            passed=auth_eval_evidence is not None,
            required_for="promotion_only",
            evidence=CUDA_AUTH_EVAL_PATH,
            expected="matching contest_auth_eval.json with provenance.device='cuda' and archive custody",
            observed=auth_eval_evidence,
            blocker=True,
        ),
    }
    gates.update(manifest_identity_gates)

    local_training_blockers = [
        name
        for name, gate in gates.items()
        if gate["required_for"] == "training_source_identity" and gate["blocker"]
    ]
    promotion_blockers = [
        name
        for name, gate in gates.items()
        if gate["required_for"] in {"candidate_archive", "promotion_only"} and gate["blocker"]
    ]

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "score_claim": False,
        "score_claim_eligible": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_readiness_only": True,
        "training_performed": False,
        "remote_job_launched": False,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "purpose": (
            "Audit whether decoded-baseline Alpha masks are ready for a local "
            "TinyNeRV/INR training prototype, without making archive or score claims."
        ),
        "inputs": {
            "baseline_archive": archive_meta,
            "decoded_masks_source": decoded_meta,
            "candidate_manifest": candidate_manifest_meta,
        },
        "decoded_mask_identity": {
            "shape": decoded_meta["shape"],
            "dtype": decoded_meta["dtype"],
            "num_pixels": int(masks.size),
            "class_id_u8_sha256": decoded_meta["class_id_u8_sha256"],
            "shape_dtype_raw_sha256": decoded_meta["shape_dtype_raw_sha256"],
            "source_identity_validated_against_candidate_manifest": candidate_manifest is not None,
        },
        "shape_accounting": {
            "frames": int(decoded_meta["shape"][0]),
            "height": int(decoded_meta["shape"][1]),
            "width": int(decoded_meta["shape"][2]),
            "num_classes": int(num_classes),
            "total_pixels": int(masks.size),
        },
        "byte_accounting": byte_accounting,
        "prototype": prototype,
        "readiness_gates": gates,
        "readiness_summary": {
            "ready_for_local_tiny_nerv_training_prototype": not local_training_blockers,
            "ready_for_candidate_archive": False,
            "ready_for_archive_promotion": False,
            "local_training_blockers": local_training_blockers,
            "promotion_blockers": promotion_blockers,
        },
        "next_steps": [
            "Train only against this decoded-baseline mask source after preserving the manifest SHA and shape gates.",
            "Bundle every score-affecting INR payload bit inside archive.zip or fixed contest runtime code.",
            "Build a deterministic candidate archive before any eval.",
            "Run experiments/contest_auth_eval.py --device cuda on exact archive bytes and preserve JSON custody.",
        ],
        "provenance": {
            "tool": "experiments/alpha_inr_readiness.py",
            "command": command,
        },
    }

    if output_json is not None:
        output_json = Path(output_json)
        if output_json.exists() and not force:
            raise FileExistsError(f"{output_json} exists; use --force to overwrite")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-archive", type=Path, default=DEFAULT_BASELINE_ARCHIVE)
    parser.add_argument("--decoded-masks-source", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=None)
    parser.add_argument("--candidate-archive", type=Path, default=None)
    parser.add_argument("--contest-auth-eval-json", type=Path, default=None)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--expected-height", type=int, default=None)
    parser.add_argument("--expected-width", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-freqs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_alpha_inr_readiness(
        baseline_archive=args.baseline_archive,
        decoded_masks_source=args.decoded_masks_source,
        candidate_manifest=args.candidate_manifest,
        candidate_archive=args.candidate_archive,
        contest_auth_eval_json=args.contest_auth_eval_json,
        expected_frames=args.expected_frames,
        expected_height=args.expected_height,
        expected_width=args.expected_width,
        num_classes=args.num_classes,
        num_freqs=args.num_freqs,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        seed=args.seed,
        output_json=args.output_json,
        command=["experiments/alpha_inr_readiness.py", *(argv if argv is not None else sys.argv[1:])],
        force=args.force,
    )
    summary = report["readiness_summary"]
    print(
        "[alpha-inr-readiness] "
        f"wrote {args.output_json} "
        f"local_training_ready={summary['ready_for_local_tiny_nerv_training_prototype']} "
        "promotion_eligible=false "
        "score_claim=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
