#!/usr/bin/env python3
"""Build deterministic archive perturbation variants for official response curves.

This tool prepares inputs for ``experiments/profile_component_sensitivity_official.py``.
It does not run scorers, patch upstream evaluator code, or make score claims.
Each nonzero epsilon point is a new deterministic ``archive.zip`` variant
constructed from a custody-checked baseline archive plus a bounded additive
byte perturbation basis.

The output plan uses ``official_component_response_plan_v1`` and is intended
to be consumed by:

    experiments/profile_component_sensitivity_official.py --perturbation-plan <plan>

Promotion-grade component response still requires exact CUDA evaluation through
``archive.zip -> inflate.sh -> upstream/evaluate.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PLAN_FORMAT = "official_component_response_plan_v1"
PERTURBATION_BASIS_FORMAT = "perturbation_basis_v1"
PREDICTION_DELTAS_FORMAT = "official_component_response_prediction_deltas_v1"
VARIANT_MANIFEST_FORMAT = "official_component_response_archive_variants_v1"
PRODUCER = "experiments/build_component_response_perturbation_plan.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FIXED_ZIP_PERMISSIONS = 0o644
SCORE_EPS = 1e-12
COMPONENTS = ("posenet", "segnet", "combined")
DEFAULT_EPSILONS = (-1.0, 0.0, 1.0)
DEFAULT_ALLOWED_MUTATION_MEMBERS = ("renderer.bin",)
PREDICTION_SOURCE_KIND = "component_sensitivity_map_projection"
PREDICTION_LEAKAGE_KEYS = {
    "actual_delta",
    "contest_auth_eval",
    "contest_auth_eval_json",
    "holdout_error",
    "measured_delta",
    "observed_delta",
    "official_response",
    "official_response_curve",
    "response_curve",
}
SCORER_FREE_RENDERER_MAGICS = frozenset(
    {
        b"ASYM",
        b"DPSM",
        b"FP4A",
        b"FP8H",
        b"I4LZ",
        b"CCh1",
        b"C3R1",
        b"SCv1",
        b"SZv1",
        b"NWC1",
        b"NWCS",
        b"OWV2",
        b"OWV3",
        b"IMPS",
        b"OMG1",
        b"BHv1",
        b"JCSP",
    }
)


class ComponentResponsePerturbationError(ValueError):
    """Raised when perturbation archive inputs or outputs are unsafe."""


@dataclass(frozen=True)
class PatchAtom:
    atom_id: str
    member: str
    offset: int
    delta_per_epsilon: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ArchiveMember:
    name: str
    data: bytes
    source_info: Mapping[str, Any]


@dataclass(frozen=True)
class PointVariant:
    epsilon: float
    archive: Path
    archive_meta: Mapping[str, Any]
    changed_member_bytes: int
    raw_l1_byte_delta: int
    max_abs_byte_delta: int
    member_mutations: tuple[Mapping[str, Any], ...]
    predicted_delta: Mapping[str, float] | None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": _path_for_json(path, root=root),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_hash_any(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_for_json(path: Path, *, root: Path | None) -> str:
    resolved = path.resolve()
    if root is not None:
        root_resolved = root.resolve()
        try:
            return resolved.relative_to(root_resolved).as_posix()
        except ValueError:
            try:
                resolved.relative_to(REPO_ROOT.resolve())
            except ValueError:
                pass
            else:
                return Path(os.path.relpath(resolved, root_resolved)).as_posix()
    return str(resolved)


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComponentResponsePerturbationError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ComponentResponsePerturbationError(f"{field} must be finite")
    return out


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComponentResponsePerturbationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ComponentResponsePerturbationError(f"{field} must be >= {minimum}")
    return int(value)


def _validate_member_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ComponentResponsePerturbationError("archive member name must be non-empty")
    if "\x00" in name or "\\" in name:
        raise ComponentResponsePerturbationError(f"unsafe archive member path: {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ComponentResponsePerturbationError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute():
        raise ComponentResponsePerturbationError(f"unsafe archive member path: {name!r}")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ComponentResponsePerturbationError(f"unsafe archive member path: {name!r}")
    lowered = [part.lower() for part in parts]
    if any(part.startswith(".") for part in parts):
        raise ComponentResponsePerturbationError(f"hidden archive sidecar rejected: {name!r}")
    if any(part == "__macosx" or part.startswith("._") for part in lowered):
        raise ComponentResponsePerturbationError(f"resource fork sidecar rejected: {name!r}")
    if any(part in {"thumbs.db", "desktop.ini"} for part in lowered):
        raise ComponentResponsePerturbationError(f"housekeeping sidecar rejected: {name!r}")
    return path.as_posix()


def _zipinfo_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _read_custody_checked_archive(path: Path) -> dict[str, ArchiveMember]:
    if not path.is_file():
        raise ComponentResponsePerturbationError(f"baseline archive not found: {path}")
    members: dict[str, ArchiveMember] = {}
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    raise ComponentResponsePerturbationError(
                        f"directory entries are not allowed in archive: {info.filename!r}"
                    )
                name = _validate_member_name(info.filename)
                if name in members:
                    raise ComponentResponsePerturbationError(
                        f"duplicate archive member rejected: {name!r}"
                    )
                if _zipinfo_is_symlink(info):
                    raise ComponentResponsePerturbationError(
                        f"symlink archive member rejected: {name!r}"
                    )
                if info.flag_bits & 0x1:
                    raise ComponentResponsePerturbationError(
                        f"encrypted archive member rejected: {name!r}"
                    )
                data = zf.read(info)
                members[name] = ArchiveMember(
                    name=name,
                    data=data,
                    source_info={
                        "name": name,
                        "raw_bytes": int(info.file_size),
                        "compressed_bytes": int(info.compress_size),
                        "crc32": f"{info.CRC:08x}",
                        "date_time": list(info.date_time),
                        "compress_type": int(info.compress_type),
                        "permissions_octal": oct((info.external_attr >> 16) & 0o777),
                        "sha256": _sha256_bytes(data),
                    },
                )
    except zipfile.BadZipFile as exc:
        raise ComponentResponsePerturbationError(
            f"baseline archive is not a valid zip: {path}"
        ) from exc
    if not members:
        raise ComponentResponsePerturbationError("baseline archive has no members")
    if "renderer.bin" not in members:
        raise ComponentResponsePerturbationError("baseline archive missing renderer.bin")
    _validate_renderer_magic(members["renderer.bin"].data, context="baseline renderer.bin")
    return members


def _validate_renderer_magic(data: bytes, *, context: str) -> None:
    magic = bytes(data[:4])
    if magic not in SCORER_FREE_RENDERER_MAGICS:
        raise ComponentResponsePerturbationError(
            f"{context} magic {magic!r} is not in the scorer-free renderer allowlist"
        )


def _parse_patch_spec(spec: str) -> PatchAtom:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ComponentResponsePerturbationError(
            "--patch must have form member:offset:delta_per_epsilon"
        )
    member = _validate_member_name(parts[0])
    try:
        offset = int(parts[1])
        delta = int(parts[2])
    except ValueError as exc:
        raise ComponentResponsePerturbationError(
            f"invalid patch integer in {spec!r}"
        ) from exc
    if offset < 0:
        raise ComponentResponsePerturbationError("patch offset must be >= 0")
    if delta == 0:
        raise ComponentResponsePerturbationError("patch delta_per_epsilon must be nonzero")
    atom_id = f"{member}@{offset}:d{delta:+d}"
    return PatchAtom(
        atom_id=atom_id,
        member=member,
        offset=offset,
        delta_per_epsilon=delta,
        metadata={},
    )


def _load_basis_atoms(path: Path) -> list[PatchAtom]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ComponentResponsePerturbationError(f"{path}: invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ComponentResponsePerturbationError(f"{path}: basis JSON must be an object")
    raw_atoms = payload.get("atoms")
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise ComponentResponsePerturbationError(f"{path}: atoms must be a non-empty list")
    atoms: list[PatchAtom] = []
    for index, raw in enumerate(raw_atoms):
        if not isinstance(raw, Mapping):
            raise ComponentResponsePerturbationError(f"{path}: atoms[{index}] must be an object")
        member_value = raw.get("member")
        if not isinstance(member_value, str):
            raise ComponentResponsePerturbationError(
                f"{path}: atoms[{index}].member must be a string"
            )
        member = _validate_member_name(member_value)
        offset = _require_int(raw.get("offset"), field=f"atoms[{index}].offset", minimum=0)
        if "delta_per_epsilon" in raw:
            delta = _require_int(
                raw.get("delta_per_epsilon"),
                field=f"atoms[{index}].delta_per_epsilon",
            )
        elif "direction" in raw:
            delta = _require_int(raw.get("direction"), field=f"atoms[{index}].direction")
        else:
            raise ComponentResponsePerturbationError(
                f"{path}: atoms[{index}] requires delta_per_epsilon or direction"
            )
        if delta == 0:
            raise ComponentResponsePerturbationError(
                f"{path}: atoms[{index}] delta_per_epsilon must be nonzero"
            )
        atom_id = raw.get("atom_id")
        if atom_id is None:
            atom_id = f"{member}@{offset}:d{delta:+d}"
        if not isinstance(atom_id, str) or not atom_id:
            raise ComponentResponsePerturbationError(f"{path}: atoms[{index}].atom_id invalid")
        metadata = {
            str(key): raw[key]
            for key in sorted(raw)
            if key not in {"atom_id", "member", "offset", "delta_per_epsilon", "direction"}
        }
        atoms.append(
            PatchAtom(
                atom_id=atom_id,
                member=member,
                offset=offset,
                delta_per_epsilon=delta,
                metadata=metadata,
            )
        )
    return atoms


def _load_prediction_map(path: Path | None) -> dict[float, dict[str, float]]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ComponentResponsePerturbationError(f"{path}: invalid JSON") from exc
    raw_points: list[Any]
    if isinstance(payload, Mapping) and isinstance(payload.get("points"), list):
        raw_points = payload["points"]
    elif isinstance(payload, list):
        raw_points = payload
    elif isinstance(payload, Mapping):
        raw_points = []
        for key, value in payload.items():
            try:
                epsilon = float(key)
            except (TypeError, ValueError) as exc:
                raise ComponentResponsePerturbationError(
                    f"{path}: prediction key {key!r} is not a numeric epsilon"
                ) from exc
            raw_points.append({"epsilon": epsilon, "predicted_delta": value})
    else:
        raise ComponentResponsePerturbationError(
            f"{path}: prediction JSON must be an object or list"
        )
    out: dict[float, dict[str, float]] = {}
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping):
            raise ComponentResponsePerturbationError(
                f"{path}: prediction point {index} must be an object"
            )
        epsilon = _finite_float(raw.get("epsilon"), field=f"prediction[{index}].epsilon")
        prediction = raw.get("predicted_delta", raw.get("predicted_deltas"))
        if not isinstance(prediction, Mapping):
            raise ComponentResponsePerturbationError(
                f"{path}: prediction[{index}] requires predicted_delta object"
            )
        values: dict[str, float] = {}
        for component in COMPONENTS:
            if component in prediction:
                values[component] = _finite_float(
                    prediction[component],
                    field=f"prediction[{index}].predicted_delta.{component}",
                )
        if not values:
            raise ComponentResponsePerturbationError(
                f"{path}: prediction[{index}] has no component deltas"
            )
        key = _epsilon_key(epsilon)
        if key in out:
            raise ComponentResponsePerturbationError(
                f"{path}: duplicate prediction epsilon {epsilon!r}"
            )
        out[key] = values
    return out


def _atom_set_sha256(atoms: list[PatchAtom]) -> str:
    payload = [
        {
            "atom_id": atom.atom_id,
            "delta_per_epsilon": int(atom.delta_per_epsilon),
            "member": atom.member,
            "offset": int(atom.offset),
        }
        for atom in sorted(
            atoms,
            key=lambda item: (
                item.member,
                int(item.offset),
                item.atom_id,
                int(item.delta_per_epsilon),
            ),
        )
    ]
    return _canonical_hash_any(payload)


def _reject_prediction_observed_response_leakage(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_s = str(key)
            if key_s.lower() in PREDICTION_LEAKAGE_KEYS:
                raise ComponentResponsePerturbationError(
                    f"{path}.{key_s} is an observed-response field; "
                    "predicted-delta artifacts must be authored before official response eval"
                )
            _reject_prediction_observed_response_leakage(child, path=f"{path}.{key_s}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_prediction_observed_response_leakage(child, path=f"{path}[{index}]")


def _load_prediction_artifact(
    path: Path,
    *,
    baseline_meta: Mapping[str, Any],
    atoms: list[PatchAtom],
    epsilons: list[float],
) -> dict[float, dict[str, float]]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ComponentResponsePerturbationError(f"{path}: invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ComponentResponsePerturbationError(f"{path}: prediction artifact must be an object")
    if payload.get("format") != PREDICTION_DELTAS_FORMAT:
        raise ComponentResponsePerturbationError(
            f"{path}: --require-predicted-deltas requires format "
            f"{PREDICTION_DELTAS_FORMAT!r}, got {payload.get('format')!r}"
        )
    _reject_prediction_observed_response_leakage(payload)
    source = payload.get("prediction_source")
    if not isinstance(source, Mapping):
        raise ComponentResponsePerturbationError(f"{path}: prediction_source object is required")
    if source.get("source_kind") != PREDICTION_SOURCE_KIND:
        raise ComponentResponsePerturbationError(
            f"{path}: prediction_source.source_kind must be {PREDICTION_SOURCE_KIND!r}"
        )
    model = payload.get("prediction_model")
    if not isinstance(model, Mapping) or model.get("uses_official_response_observations") is not False:
        raise ComponentResponsePerturbationError(
            f"{path}: prediction_model.uses_official_response_observations must be false"
        )
    baseline = source.get("baseline_archive")
    if not isinstance(baseline, Mapping):
        raise ComponentResponsePerturbationError(f"{path}: prediction_source.baseline_archive is required")
    if baseline.get("sha256") != baseline_meta["sha256"] or baseline.get("bytes") != baseline_meta["bytes"]:
        raise ComponentResponsePerturbationError(
            f"{path}: prediction baseline archive custody does not match baseline archive"
        )
    basis = source.get("perturbation_basis")
    if not isinstance(basis, Mapping):
        raise ComponentResponsePerturbationError(f"{path}: prediction_source.perturbation_basis is required")
    expected_atom_set = _atom_set_sha256(atoms)
    if basis.get("atom_set_sha256") != expected_atom_set:
        raise ComponentResponsePerturbationError(
            f"{path}: prediction perturbation atom_set_sha256 does not match plan atoms"
        )
    raw_ladder = payload.get("epsilon_ladder")
    if not isinstance(raw_ladder, list):
        raise ComponentResponsePerturbationError(f"{path}: epsilon_ladder list is required")
    ladder = {_epsilon_key(_finite_float(item, field="epsilon_ladder")) for item in raw_ladder}
    missing_eps = [
        eps
        for eps in epsilons
        if abs(eps) > SCORE_EPS and _epsilon_key(eps) not in ladder
    ]
    if missing_eps:
        raise ComponentResponsePerturbationError(
            f"{path}: prediction artifact missing epsilon(s): {missing_eps}"
        )
    return _load_prediction_map(path)


def _load_prediction_artifact_model(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ComponentResponsePerturbationError(f"{path}: invalid JSON") from exc
    if not isinstance(payload, Mapping):
        return None
    model = payload.get("prediction_model")
    if not isinstance(model, Mapping):
        return None
    allowed = {
        "model",
        "prediction_delta_semantics",
        "prediction_error_mode",
        "equation",
        "sign_policy",
        "uses_official_response_observations",
    }
    return {str(key): model[key] for key in sorted(model) if key in allowed}


def _epsilon_key(epsilon: float) -> float:
    return round(float(epsilon), 15)


def _safe_point_label(index: int, epsilon: float) -> str:
    eps = f"{epsilon:+.12g}".replace("+", "p").replace("-", "m")
    eps = re.sub(r"[^A-Za-z0-9_.]+", "_", eps)
    return f"point_{index:03d}_eps_{eps}"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (FIXED_ZIP_PERMISSIONS & 0xFFFF) << 16
    return info


def _archive_bytes(members: Mapping[str, bytes], *, member_order: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in member_order:
            zf.writestr(_zip_info(name), members[name], compresslevel=9)
    return buf.getvalue()


def _archive_manifest_from_bytes(data: bytes) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        for info in zf.infolist():
            member_data = zf.read(info)
            manifest.append(
                {
                    "name": info.filename,
                    "raw_bytes": int(info.file_size),
                    "compressed_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "date_time": list(info.date_time),
                    "compress_type": int(info.compress_type),
                    "permissions_octal": oct((info.external_attr >> 16) & 0o777),
                    "sha256": _sha256_bytes(member_data),
                }
            )
    return manifest


def _member_manifest(members: Mapping[str, ArchiveMember]) -> list[dict[str, Any]]:
    return [dict(members[name].source_info) for name in sorted(members)]


def _validate_atoms(
    atoms: list[PatchAtom],
    *,
    members: Mapping[str, ArchiveMember],
    allowed_members: set[str],
    allow_renderer_magic_mutation: bool,
) -> None:
    if not atoms:
        raise ComponentResponsePerturbationError("at least one perturbation atom is required")
    seen_ids: set[str] = set()
    for atom in atoms:
        if atom.atom_id in seen_ids:
            raise ComponentResponsePerturbationError(f"duplicate atom_id: {atom.atom_id!r}")
        seen_ids.add(atom.atom_id)
        if atom.member not in members:
            raise ComponentResponsePerturbationError(
                f"atom {atom.atom_id!r} references missing member {atom.member!r}"
            )
        if atom.member not in allowed_members:
            raise ComponentResponsePerturbationError(
                f"atom {atom.atom_id!r} mutates member {atom.member!r}; "
                f"allowed members are {sorted(allowed_members)}"
            )
        if atom.offset >= len(members[atom.member].data):
            raise ComponentResponsePerturbationError(
                f"atom {atom.atom_id!r} offset {atom.offset} is outside "
                f"{atom.member} ({len(members[atom.member].data)} bytes)"
            )
        if (
            atom.member == "renderer.bin"
            and atom.offset < 4
            and not allow_renderer_magic_mutation
        ):
            raise ComponentResponsePerturbationError(
                f"atom {atom.atom_id!r} would mutate renderer magic bytes"
            )


def _integer_scaled_delta(epsilon: float, delta_per_epsilon: int, *, atom_id: str) -> int:
    scaled = float(epsilon) * int(delta_per_epsilon)
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-9:
        raise ComponentResponsePerturbationError(
            f"epsilon {epsilon!r} times atom {atom_id!r} delta_per_epsilon "
            f"{delta_per_epsilon} is not an integer byte delta"
        )
    return int(rounded)


def _apply_epsilon(
    *,
    members: Mapping[str, ArchiveMember],
    atoms: list[PatchAtom],
    epsilon: float,
    allow_renderer_magic_mutation: bool,
    max_mutated_bytes: int,
    max_abs_byte_delta: int,
    max_raw_l1_delta: int,
) -> tuple[dict[str, bytes], tuple[Mapping[str, Any], ...], int, int, int]:
    aggregated: dict[tuple[str, int], int] = {}
    for atom in atoms:
        delta = _integer_scaled_delta(
            epsilon,
            atom.delta_per_epsilon,
            atom_id=atom.atom_id,
        )
        if delta == 0:
            continue
        key = (atom.member, atom.offset)
        aggregated[key] = aggregated.get(key, 0) + delta

    out = {name: bytearray(member.data) for name, member in members.items()}
    mutation_records: list[Mapping[str, Any]] = []
    for (member_name, offset), delta in sorted(aggregated.items()):
        if delta == 0:
            continue
        original = out[member_name][offset]
        mutated = original + delta
        if mutated < 0 or mutated > 255:
            raise ComponentResponsePerturbationError(
                f"epsilon {epsilon!r} moves {member_name}@{offset} from "
                f"{original} by {delta:+d}, outside byte range [0, 255]"
            )
        out[member_name][offset] = mutated
        mutation_records.append(
            {
                "member": member_name,
                "offset": int(offset),
                "original_byte": int(original),
                "mutated_byte": int(mutated),
                "delta": int(delta),
            }
        )

    changed_member_bytes = len(mutation_records)
    raw_l1_byte_delta = sum(abs(int(record["delta"])) for record in mutation_records)
    max_observed_abs_delta = max((abs(int(record["delta"])) for record in mutation_records), default=0)
    if changed_member_bytes > max_mutated_bytes:
        raise ComponentResponsePerturbationError(
            f"epsilon {epsilon!r} mutates {changed_member_bytes} bytes, "
            f"exceeding max_mutated_bytes={max_mutated_bytes}"
        )
    if max_observed_abs_delta > max_abs_byte_delta:
        raise ComponentResponsePerturbationError(
            f"epsilon {epsilon!r} has byte delta {max_observed_abs_delta}, "
            f"exceeding max_abs_byte_delta={max_abs_byte_delta}"
        )
    if raw_l1_byte_delta > max_raw_l1_delta:
        raise ComponentResponsePerturbationError(
            f"epsilon {epsilon!r} raw L1 byte delta {raw_l1_byte_delta} exceeds "
            f"max_raw_l1_delta={max_raw_l1_delta}"
        )
    if abs(epsilon) > SCORE_EPS and changed_member_bytes == 0:
        raise ComponentResponsePerturbationError(
            f"nonzero epsilon {epsilon!r} produced no byte mutation"
        )
    final = {name: bytes(data) for name, data in out.items()}
    if "renderer.bin" in final:
        if not allow_renderer_magic_mutation:
            before = bytes(members["renderer.bin"].data[:4])
            after = bytes(final["renderer.bin"][:4])
            if after != before:
                raise ComponentResponsePerturbationError(
                    f"epsilon {epsilon!r} changed renderer magic from {before!r} to {after!r}"
                )
        _validate_renderer_magic(final["renderer.bin"], context=f"epsilon {epsilon!r} renderer.bin")
    return (
        final,
        tuple(mutation_records),
        changed_member_bytes,
        raw_l1_byte_delta,
        max_observed_abs_delta,
    )


def _prediction_for_epsilon(
    epsilon: float,
    predictions: Mapping[float, Mapping[str, float]],
    *,
    require_predicted_deltas: bool,
) -> Mapping[str, float] | None:
    prediction = predictions.get(_epsilon_key(epsilon))
    if abs(epsilon) <= SCORE_EPS:
        return {component: 0.0 for component in COMPONENTS}
    if prediction is None:
        if require_predicted_deltas:
            raise ComponentResponsePerturbationError(
                f"missing predicted deltas for epsilon {epsilon!r}"
            )
        return None
    if require_predicted_deltas and not {"posenet", "segnet"} <= set(prediction):
        raise ComponentResponsePerturbationError(
            f"epsilon {epsilon!r} predictions must include posenet and segnet"
        )
    return dict(sorted(prediction.items()))


def _build_basis_payload(
    *,
    baseline_archive: Path,
    baseline_meta: Mapping[str, Any],
    members: Mapping[str, ArchiveMember],
    atoms: list[PatchAtom],
    epsilons: list[float],
    allowed_members: set[str],
    allow_renderer_magic_mutation: bool,
    max_mutated_bytes: int,
    max_abs_byte_delta: int,
    max_raw_l1_delta: int,
) -> dict[str, Any]:
    atom_payload = [
        {
            "atom_id": atom.atom_id,
            "member": atom.member,
            "offset": int(atom.offset),
            "delta_per_epsilon": int(atom.delta_per_epsilon),
            **({"metadata": dict(atom.metadata)} if atom.metadata else {}),
        }
        for atom in sorted(atoms, key=lambda item: item.atom_id)
    ]
    basis_for_hash = {
        "format": PERTURBATION_BASIS_FORMAT,
        "basis_kind": "archive_byte_additive",
        "epsilon_units": "signed_integer_step",
        "source_archive": {
            "bytes": int(baseline_meta["bytes"]),
            "sha256": str(baseline_meta["sha256"]),
        },
        "source_members": _member_manifest(members),
        "atoms": atom_payload,
        "epsilon_ladder": epsilons,
        "mutation_policy": {
            "byte_delta_formula": "new_byte = old_byte + round(epsilon * delta_per_epsilon)",
            "byte_range": [0, 255],
            "out_of_range": "fail_closed",
            "allowed_mutation_members": sorted(allowed_members),
            "allow_renderer_magic_mutation": bool(allow_renderer_magic_mutation),
            "max_mutated_bytes": int(max_mutated_bytes),
            "max_abs_byte_delta": int(max_abs_byte_delta),
            "max_raw_l1_delta": int(max_raw_l1_delta),
        },
    }
    basis_id = _canonical_hash(basis_for_hash)
    return {
        "schema_version": 1,
        "format": PERTURBATION_BASIS_FORMAT,
        "basis_id": basis_id,
        "producer": PRODUCER,
        "basis_kind": "archive_byte_additive",
        "epsilon_units": "signed_integer_step",
        "source_archive": {
            "path": str(baseline_archive.resolve()),
            "bytes": int(baseline_meta["bytes"]),
            "sha256": str(baseline_meta["sha256"]),
        },
        "source_members": _member_manifest(members),
        "atoms": atom_payload,
        "epsilon_ladder": epsilons,
        "mutation_policy": basis_for_hash["mutation_policy"],
        "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        "auth_eval_required": "cuda",
    }


def build_component_response_perturbation_plan(
    *,
    baseline_archive: Path,
    output_dir: Path,
    atoms: list[PatchAtom],
    epsilons: list[float],
    plan_output: Path | None = None,
    basis_output: Path | None = None,
    variants_manifest_output: Path | None = None,
    baseline_contest_auth_eval_json: Path | None = None,
    predicted_deltas_json: Path | None = None,
    require_predicted_deltas: bool = False,
    allowed_mutation_members: set[str] | None = None,
    allow_renderer_magic_mutation: bool = False,
    max_mutated_bytes: int = 1024,
    max_abs_byte_delta: int = 32,
    max_raw_l1_delta: int = 4096,
    max_archive_bytes: int | None = None,
    max_archive_byte_delta: int | None = None,
) -> dict[str, Any]:
    baseline_archive = baseline_archive.resolve()
    output_dir = output_dir.resolve()
    plan_output = (plan_output or output_dir / "official_component_response_plan.json").resolve()
    basis_output = (basis_output or output_dir / "perturbation_basis_v1.json").resolve()
    variants_manifest_output = (
        variants_manifest_output or output_dir / "archive_variants_manifest.json"
    ).resolve()
    if predicted_deltas_json is not None:
        predicted_deltas_json = predicted_deltas_json.resolve()
    if baseline_contest_auth_eval_json is not None:
        baseline_contest_auth_eval_json = baseline_contest_auth_eval_json.resolve()
    allowed_members = set(allowed_mutation_members or DEFAULT_ALLOWED_MUTATION_MEMBERS)
    allowed_members = {_validate_member_name(name) for name in allowed_members}

    if not epsilons:
        raise ComponentResponsePerturbationError("epsilon ladder must not be empty")
    epsilon_values = sorted({_epsilon_key(_finite_float(eps, field="epsilon")) for eps in epsilons})
    if not any(abs(eps) <= SCORE_EPS for eps in epsilon_values):
        epsilon_values.insert(0, 0.0)
        epsilon_values = sorted(set(epsilon_values))
    if not any(abs(eps) > SCORE_EPS for eps in epsilon_values):
        raise ComponentResponsePerturbationError("epsilon ladder needs at least one nonzero point")

    for label, value in (
        ("max_mutated_bytes", max_mutated_bytes),
        ("max_abs_byte_delta", max_abs_byte_delta),
        ("max_raw_l1_delta", max_raw_l1_delta),
    ):
        if int(value) < 0:
            raise ComponentResponsePerturbationError(f"{label} must be nonnegative")

    members = _read_custody_checked_archive(baseline_archive)
    _validate_atoms(
        atoms,
        members=members,
        allowed_members=allowed_members,
        allow_renderer_magic_mutation=allow_renderer_magic_mutation,
    )
    baseline_meta = _file_meta(baseline_archive)
    if baseline_contest_auth_eval_json is not None and not baseline_contest_auth_eval_json.is_file():
        raise ComponentResponsePerturbationError(
            f"baseline contest_auth_eval_json not found: {baseline_contest_auth_eval_json}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    member_order = sorted(members)
    predictions = (
        _load_prediction_artifact(
            predicted_deltas_json,
            baseline_meta=baseline_meta,
            atoms=atoms,
            epsilons=epsilon_values,
        )
        if predicted_deltas_json is not None and require_predicted_deltas
        else _load_prediction_map(predicted_deltas_json)
    )
    prediction_model = (
        _load_prediction_artifact_model(predicted_deltas_json)
        if predicted_deltas_json is not None and require_predicted_deltas
        else None
    )
    basis_payload = _build_basis_payload(
        baseline_archive=baseline_archive,
        baseline_meta=baseline_meta,
        members=members,
        atoms=atoms,
        epsilons=epsilon_values,
        allowed_members=allowed_members,
        allow_renderer_magic_mutation=allow_renderer_magic_mutation,
        max_mutated_bytes=max_mutated_bytes,
        max_abs_byte_delta=max_abs_byte_delta,
        max_raw_l1_delta=max_raw_l1_delta,
    )
    _write_json(basis_output, basis_payload)

    variants: list[PointVariant] = []
    archive_manifest_records: list[dict[str, Any]] = []
    for index, epsilon in enumerate(epsilon_values):
        prediction = _prediction_for_epsilon(
            epsilon,
            predictions,
            require_predicted_deltas=require_predicted_deltas,
        )
        if abs(epsilon) <= SCORE_EPS:
            variants.append(
                PointVariant(
                    epsilon=0.0,
                    archive=baseline_archive,
                    archive_meta=baseline_meta,
                    changed_member_bytes=0,
                    raw_l1_byte_delta=0,
                    max_abs_byte_delta=0,
                    member_mutations=(),
                    predicted_delta=prediction,
                )
            )
            continue

        (
            point_members,
            mutation_records,
            changed_member_bytes,
            raw_l1_byte_delta,
            max_observed_abs_delta,
        ) = _apply_epsilon(
            members=members,
            atoms=atoms,
            epsilon=epsilon,
            allow_renderer_magic_mutation=allow_renderer_magic_mutation,
            max_mutated_bytes=max_mutated_bytes,
            max_abs_byte_delta=max_abs_byte_delta,
            max_raw_l1_delta=max_raw_l1_delta,
        )
        archive_data = _archive_bytes(point_members, member_order=member_order)
        archive_rebuild = _archive_bytes(point_members, member_order=member_order)
        if archive_data != archive_rebuild:
            raise ComponentResponsePerturbationError(
                f"deterministic archive rebuild mismatch for epsilon {epsilon!r}"
            )
        archive_path = output_dir / "archives" / f"{_safe_point_label(index, epsilon)}.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(archive_data)
        archive_meta = _file_meta(archive_path)
        if max_archive_bytes is not None and archive_meta["bytes"] > max_archive_bytes:
            raise ComponentResponsePerturbationError(
                f"{archive_path} has {archive_meta['bytes']} bytes, "
                f"exceeding max_archive_bytes={max_archive_bytes}"
            )
        archive_byte_delta = int(archive_meta["bytes"]) - int(baseline_meta["bytes"])
        if (
            max_archive_byte_delta is not None
            and abs(archive_byte_delta) > max_archive_byte_delta
        ):
            raise ComponentResponsePerturbationError(
                f"{archive_path} archive byte delta {archive_byte_delta:+d} exceeds "
                f"max_archive_byte_delta={max_archive_byte_delta}"
            )
        manifest = _archive_manifest_from_bytes(archive_data)
        archive_manifest_records.append(
            {
                "epsilon": float(epsilon),
                "archive": _file_meta(archive_path, root=variants_manifest_output.parent),
                "archive_byte_delta_vs_baseline": archive_byte_delta,
                "deterministic_rebuild": True,
                "changed_member_bytes": int(changed_member_bytes),
                "raw_l1_byte_delta": int(raw_l1_byte_delta),
                "max_abs_byte_delta": int(max_observed_abs_delta),
                "member_mutations": list(mutation_records),
                "members": manifest,
            }
        )
        variants.append(
            PointVariant(
                epsilon=epsilon,
                archive=archive_path,
                archive_meta=archive_meta,
                changed_member_bytes=changed_member_bytes,
                raw_l1_byte_delta=raw_l1_byte_delta,
                max_abs_byte_delta=max_observed_abs_delta,
                member_mutations=mutation_records,
                predicted_delta=prediction,
            )
        )

    variants_manifest = {
        "schema_version": 1,
        "format": VARIANT_MANIFEST_FORMAT,
        "producer": PRODUCER,
        "baseline_archive": _file_meta(baseline_archive, root=variants_manifest_output.parent),
        "basis": _file_meta(basis_output, root=variants_manifest_output.parent),
        "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        "auth_eval_required": "cuda",
        "member_order": member_order,
        "points": archive_manifest_records,
    }
    _write_json(variants_manifest_output, variants_manifest)

    plan_points: list[dict[str, Any]] = []
    for variant in sorted(variants, key=lambda item: item.epsilon):
        point: dict[str, Any] = {
            "epsilon": float(variant.epsilon),
            "archive": _path_for_json(variant.archive, root=plan_output.parent),
            "archive_bytes": int(variant.archive_meta["bytes"]),
            "archive_sha256": str(variant.archive_meta["sha256"]),
            "changed_member_bytes": int(variant.changed_member_bytes),
            "raw_l1_byte_delta": int(variant.raw_l1_byte_delta),
            "max_abs_byte_delta": int(variant.max_abs_byte_delta),
        }
        if variant.member_mutations:
            point["member_mutations"] = list(variant.member_mutations)
        if variant.predicted_delta is not None:
            point["predicted_delta"] = dict(variant.predicted_delta)
        plan_points.append(point)

    perturbation_meta = {
        "format": PERTURBATION_BASIS_FORMAT,
        "basis_kind": "archive_byte_additive",
        "basis_id": str(basis_payload["basis_id"]),
        "basis_path": _path_for_json(basis_output, root=plan_output.parent),
        "basis_sha256": _sha256_file(basis_output),
        "epsilon_units": "signed_integer_step",
        "archive_variants_manifest": _path_for_json(
            variants_manifest_output,
            root=plan_output.parent,
        ),
        "archive_variants_manifest_sha256": _sha256_file(variants_manifest_output),
        "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        "auth_eval_required": "cuda",
        "mutation_policy": basis_payload["mutation_policy"],
    }
    if predicted_deltas_json is not None:
        perturbation_meta["predicted_deltas_source"] = _file_meta(
            predicted_deltas_json,
            root=plan_output.parent,
        )
    if prediction_model is not None:
        perturbation_meta["prediction_model"] = prediction_model
    plan: dict[str, Any] = {
        "schema_version": 1,
        "format": PLAN_FORMAT,
        "producer": PRODUCER,
        "baseline_archive": _file_meta(baseline_archive, root=plan_output.parent),
        "perturbation": perturbation_meta,
        "points": plan_points,
    }
    if baseline_contest_auth_eval_json is not None:
        plan["baseline_contest_auth_eval_json"] = _path_for_json(
            baseline_contest_auth_eval_json,
            root=plan_output.parent,
        )
        plan["baseline_contest_auth_eval"] = _file_meta(
            baseline_contest_auth_eval_json,
            root=plan_output.parent,
        )
    _write_json(plan_output, plan)

    summary = {
        "schema_version": 1,
        "format": "official_component_response_perturbation_plan_summary_v1",
        "producer": PRODUCER,
        "plan": _file_meta(plan_output),
        "basis": _file_meta(basis_output),
        "archive_variants_manifest": _file_meta(variants_manifest_output),
        "baseline_archive": baseline_meta,
        "basis_id": str(basis_payload["basis_id"]),
        "point_count": len(plan_points),
        "nonzero_point_count": sum(1 for point in plan_points if abs(float(point["epsilon"])) > SCORE_EPS),
        "epsilon_ladder": [float(point["epsilon"]) for point in plan_points],
        "canonical_response_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        "auth_eval_required": "cuda",
    }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--basis-json",
        type=Path,
        default=None,
        help="Optional perturbation_basis_v1-style JSON with atoms.",
    )
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        help="Byte atom in form member:offset:delta_per_epsilon. Repeatable.",
    )
    parser.add_argument(
        "--epsilon",
        action="append",
        type=float,
        default=None,
        help="Response epsilon. Repeatable. If omitted, uses -1,0,+1.",
    )
    parser.add_argument("--plan-output", type=Path, default=None)
    parser.add_argument("--basis-output", type=Path, default=None)
    parser.add_argument("--variants-manifest-output", type=Path, default=None)
    parser.add_argument("--baseline-contest-auth-eval-json", type=Path, default=None)
    parser.add_argument(
        "--predicted-deltas-json",
        type=Path,
        default=None,
        help="Optional explicit predicted component deltas keyed by epsilon.",
    )
    parser.add_argument(
        "--require-predicted-deltas",
        action="store_true",
        help="Fail if every nonzero epsilon lacks explicit posenet/segnet predictions.",
    )
    parser.add_argument(
        "--allow-mutation-member",
        action="append",
        default=[],
        help="Additional archive member allowed to mutate. renderer.bin is allowed by default.",
    )
    parser.add_argument(
        "--allow-renderer-magic-mutation",
        action="store_true",
        help="Allow atoms in renderer.bin offsets 0..3. Unsafe for normal response plans.",
    )
    parser.add_argument("--max-mutated-bytes", type=int, default=1024)
    parser.add_argument("--max-abs-byte-delta", type=int, default=32)
    parser.add_argument("--max-raw-l1-delta", type=int, default=4096)
    parser.add_argument("--max-archive-bytes", type=int, default=None)
    parser.add_argument("--max-archive-byte-delta", type=int, default=None)
    args = parser.parse_args(argv)

    if args.basis_json is None and not args.patch:
        parser.error("provide --basis-json or at least one --patch")
    for field in (
        "max_mutated_bytes",
        "max_abs_byte_delta",
        "max_raw_l1_delta",
    ):
        if getattr(args, field) < 0:
            parser.error(f"--{field.replace('_', '-')} must be nonnegative")
    for field in ("max_archive_bytes", "max_archive_byte_delta"):
        value = getattr(args, field)
        if value is not None and value < 0:
            parser.error(f"--{field.replace('_', '-')} must be nonnegative")
    epsilons = args.epsilon if args.epsilon is not None else list(DEFAULT_EPSILONS)
    for index, epsilon in enumerate(epsilons):
        if not math.isfinite(float(epsilon)):
            parser.error(f"--epsilon value {index} must be finite")
    args.epsilons = [float(eps) for eps in epsilons]
    return args


def _atoms_from_args(args: argparse.Namespace) -> list[PatchAtom]:
    atoms: list[PatchAtom] = []
    if args.basis_json is not None:
        atoms.extend(_load_basis_atoms(args.basis_json))
    atoms.extend(_parse_patch_spec(spec) for spec in args.patch)
    return atoms


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        allowed_members = set(DEFAULT_ALLOWED_MUTATION_MEMBERS)
        allowed_members.update(args.allow_mutation_member)
        summary = build_component_response_perturbation_plan(
            baseline_archive=args.baseline_archive,
            output_dir=args.output_dir,
            atoms=_atoms_from_args(args),
            epsilons=args.epsilons,
            plan_output=args.plan_output,
            basis_output=args.basis_output,
            variants_manifest_output=args.variants_manifest_output,
            baseline_contest_auth_eval_json=args.baseline_contest_auth_eval_json,
            predicted_deltas_json=args.predicted_deltas_json,
            require_predicted_deltas=args.require_predicted_deltas,
            allowed_mutation_members=allowed_members,
            allow_renderer_magic_mutation=args.allow_renderer_magic_mutation,
            max_mutated_bytes=args.max_mutated_bytes,
            max_abs_byte_delta=args.max_abs_byte_delta,
            max_raw_l1_delta=args.max_raw_l1_delta,
            max_archive_bytes=args.max_archive_bytes,
            max_archive_byte_delta=args.max_archive_byte_delta,
        )
    except ComponentResponsePerturbationError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
