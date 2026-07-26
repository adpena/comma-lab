"""Fresh n600 scorer-plane operands with immutable, resumable custody.

This module compiles only source-derived scorer coordinates:

    Yk[p] = round_u8(DisjointResizeOperator(gt_fk[p]))

It intentionally does not provide a V15 semantic predictor/base stream.  The
stored source-cache poses are exposed as advisory evidence, never as fresh pose
authority.  Candidate and score authority remain false.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator

CONFIG_SCHEMA: Final = "tac.taskspace_fresh_scorer_plane_materializer_config.v1"
PREFLIGHT_SCHEMA: Final = "tac.taskspace_fresh_scorer_plane_preflight.v1"
STAGE_SCHEMA: Final = "tac.taskspace_fresh_scorer_plane_stage.v1"
AGGREGATE_SCHEMA: Final = "tac.taskspace_fresh_scorer_plane_aggregate.v1"
MODE_DIRECT_TASK_LAYERED: Final = "DIRECT_TASK_LAYERED"
SEMANTIC_STATUS_OWED: Final = "PROGRAM_RESIDUAL_LAYERED_OWED_FRESH_V15_BASE_ABSENT"
SOURCE_MEMBER_NAMES: Final = ("n_pairs", "gt_f0", "gt_f1", "gt_poses")
PRODUCTION_ARCHIVE_MEMBERS: Final = (
    "n_pairs",
    "gt_f0",
    "gt_f1",
    "lstars",
    "margins",
    "gt_poses",
)
PRODUCTION_PAIR_COUNT: Final = 600
PRODUCTION_STAGE_PAIRS: Final = 120
PRODUCTION_CAMERA_HW: Final = (874, 1164)
PRODUCTION_SCORER_HW: Final = (384, 512)
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]


class FreshScorerPlaneMaterializationError(RuntimeError):
    """A config, source, materialization, resume, or receipt check failed."""


@dataclass(frozen=True, slots=True)
class FreshScorerPlaneStageV1:
    """One bounded, read-only chronological operand slice."""

    pair_range: tuple[int, int]
    pair_ids: I64
    y0_u8: U8
    y1_u8: U8
    target_labels_u8: U8
    gt_poses_f32: F32
    pose_authority: str = "SEALED_SOURCE_CACHE_ADVISORY_ONLY"


@dataclass(frozen=True, slots=True)
class MaterializerConfigV1:
    raw: Mapping[str, Any]
    path: Path
    run_id: str
    output_root: Path
    source_npz: Mapping[str, Any]
    fresh_teacher_receipt: Mapping[str, Any]
    target_labels: Mapping[str, Any]
    producer_sources: tuple[Mapping[str, Any], ...]
    pair_count: int
    stage_pairs: int
    camera_hw: tuple[int, int]
    scorer_hw: tuple[int, int]
    resume: bool
    test_only_small_fixture: bool
    required_free_bytes: int


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FreshScorerPlaneMaterializationError(f"bound file is absent: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise FreshScorerPlaneMaterializationError(f"payload already has {field}")
    sealed = dict(payload)
    sealed[field] = payload_sha256(payload)
    return sealed


def _verify_seal(payload: Mapping[str, Any], field: str) -> None:
    expected = payload.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise FreshScorerPlaneMaterializationError(f"missing or malformed {field}")
    body = {key: value for key, value in payload.items() if key != field}
    if payload_sha256(body) != expected:
        raise FreshScorerPlaneMaterializationError(f"{field} canonical hash differs")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshScorerPlaneMaterializationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FreshScorerPlaneMaterializationError(f"{label} must be a JSON object")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FreshScorerPlaneMaterializationError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise FreshScorerPlaneMaterializationError(f"{label} must be lowercase SHA-256")
    return value


def _require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FreshScorerPlaneMaterializationError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _require_hw(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise FreshScorerPlaneMaterializationError(f"{label} must be [H,W]")
    return (
        _require_int(value[0], f"{label}[0]", 1),
        _require_int(value[1], f"{label}[1]", 1),
    )


def _verify_expected_file(binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    _require_keys(binding, {"path", "bytes", "sha256"}, label)
    expected = {
        "path": str(Path(str(binding["path"])).expanduser().resolve()),
        "bytes": _require_int(binding["bytes"], f"{label}.bytes", 1),
        "sha256": _require_sha(binding["sha256"], f"{label}.sha256"),
    }
    observed = file_identity(expected["path"])
    if observed != expected:
        raise FreshScorerPlaneMaterializationError(
            f"{label} identity differs: expected={expected}, observed={observed}"
        )
    return observed


def open_stored_npy_memmap(npz_path: str | Path, key: str) -> np.memmap:
    """Map exactly one unencrypted ZIP_STORED NPY member without extraction."""

    path = Path(npz_path)
    member = f"{key}.npy"
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo(member)
            if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
                raise FreshScorerPlaneMaterializationError(
                    f"{path}:{member} must be unencrypted ZIP_STORED"
                )
            local_header = int(info.header_offset)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise FreshScorerPlaneMaterializationError(
            f"sealed source lacks valid member {member}"
        ) from exc
    with path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        if len(header) != 30:
            raise FreshScorerPlaneMaterializationError(f"truncated header for {member}")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise FreshScorerPlaneMaterializationError(f"bad ZIP header for {member}")
        handle.seek(local_header + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise FreshScorerPlaneMaterializationError(
                f"unsupported NPY version {version} for {member}"
            )
        offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        shape=shape,
        offset=offset,
        order="F" if fortran else "C",
    )


def _array_sha256(value: np.ndarray) -> str:
    view = memoryview(np.asarray(value)).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 64 * 1024 * 1024):
        digest.update(view[start : start + 64 * 1024 * 1024])
    return digest.hexdigest()


def _validate_archive_member_names(path: Path, test_only: bool) -> None:
    with zipfile.ZipFile(path) as archive:
        names = tuple(
            name[:-4] for name in archive.namelist() if name.endswith(".npy")
        )
    if not test_only and names != PRODUCTION_ARCHIVE_MEMBERS:
        raise FreshScorerPlaneMaterializationError(
            "production source archive member order/set differs from sealed cache"
        )
    missing = set(SOURCE_MEMBER_NAMES) - set(names)
    if missing:
        raise FreshScorerPlaneMaterializationError(
            f"source lacks required members: {sorted(missing)}"
        )


def load_config(config_path: str | os.PathLike[str]) -> MaterializerConfigV1:
    path = Path(config_path).expanduser().resolve()
    raw = _load_json(path, "materializer config")
    _require_keys(
        raw,
        {
            "schema",
            "run_id",
            "mode",
            "semantic_status",
            "source_npz",
            "fresh_teacher_receipt",
            "target_labels",
            "producer_sources",
            "output_root",
            "pair_count",
            "stage_pairs",
            "camera_hw",
            "scorer_hw",
            "resume",
            "test_only_small_fixture",
            "required_free_bytes",
            "truth",
        },
        "config",
    )
    if raw["schema"] != CONFIG_SCHEMA:
        raise FreshScorerPlaneMaterializationError("config schema differs")
    if raw["mode"] != MODE_DIRECT_TASK_LAYERED:
        raise FreshScorerPlaneMaterializationError("only DIRECT_TASK_LAYERED is supported")
    if raw["semantic_status"] != SEMANTIC_STATUS_OWED:
        raise FreshScorerPlaneMaterializationError(
            "fresh V15 semantic-base absence must remain explicit"
        )
    truth = raw["truth"]
    if not isinstance(truth, dict):
        raise FreshScorerPlaneMaterializationError("config.truth must be an object")
    _require_keys(
        truth,
        {
            "research_only",
            "score_claim",
            "candidate_claim",
            "promotion_eligible",
            "source_cache_pose_advisory_only",
            "fresh_pose_target_custody",
            "program_residual_layered_available",
        },
        "config.truth",
    )
    expected_truth = {
        "research_only": True,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "source_cache_pose_advisory_only": True,
        "fresh_pose_target_custody": False,
        "program_residual_layered_available": False,
    }
    if truth != expected_truth:
        raise FreshScorerPlaneMaterializationError("config truth boundary was weakened")
    run_id = raw["run_id"]
    if not isinstance(run_id, str) or not run_id or "/" in run_id:
        raise FreshScorerPlaneMaterializationError("run_id must be a nonempty path-free string")
    pair_count = _require_int(raw["pair_count"], "pair_count", 1)
    stage_pairs = _require_int(raw["stage_pairs"], "stage_pairs", 1)
    if pair_count % stage_pairs:
        raise FreshScorerPlaneMaterializationError(
            "pair_count must divide into complete immutable stages"
        )
    camera_hw = _require_hw(raw["camera_hw"], "camera_hw")
    scorer_hw = _require_hw(raw["scorer_hw"], "scorer_hw")
    if not isinstance(raw["resume"], bool) or not isinstance(
        raw["test_only_small_fixture"], bool
    ):
        raise FreshScorerPlaneMaterializationError(
            "resume and test_only_small_fixture must be booleans"
        )
    output_root = Path(str(raw["output_root"])).expanduser().resolve()
    if not output_root.is_absolute():
        raise FreshScorerPlaneMaterializationError("output_root must be absolute")
    if not raw["test_only_small_fixture"]:
        expected = (
            PRODUCTION_PAIR_COUNT,
            PRODUCTION_STAGE_PAIRS,
            PRODUCTION_CAMERA_HW,
            PRODUCTION_SCORER_HW,
        )
        if (pair_count, stage_pairs, camera_hw, scorer_hw) != expected:
            raise FreshScorerPlaneMaterializationError(
                f"production population/geometry must be {expected}"
            )
        if not any(
            output_root != root.resolve()
            and output_root.is_relative_to(root.resolve())
            for root in SSD_ROOTS
        ):
            raise FreshScorerPlaneMaterializationError(
                "production output must be a child of the SSD storage waterfall"
            )
    for label in ("source_npz", "fresh_teacher_receipt", "target_labels"):
        if not isinstance(raw[label], dict):
            raise FreshScorerPlaneMaterializationError(f"{label} must be an object")
    producers = raw["producer_sources"]
    if not isinstance(producers, list) or not producers:
        raise FreshScorerPlaneMaterializationError(
            "producer_sources must be a nonempty list"
        )
    for index, producer in enumerate(producers):
        if not isinstance(producer, dict):
            raise FreshScorerPlaneMaterializationError(
                f"producer_sources[{index}] must be an object"
            )
        _require_keys(producer, {"role", "path", "bytes", "sha256"}, f"producer[{index}]")
        if not isinstance(producer["role"], str) or not producer["role"]:
            raise FreshScorerPlaneMaterializationError(
                f"producer_sources[{index}].role is invalid"
            )
    return MaterializerConfigV1(
        raw=raw,
        path=path,
        run_id=run_id,
        output_root=output_root,
        source_npz=raw["source_npz"],
        fresh_teacher_receipt=raw["fresh_teacher_receipt"],
        target_labels=raw["target_labels"],
        producer_sources=tuple(producers),
        pair_count=pair_count,
        stage_pairs=stage_pairs,
        camera_hw=camera_hw,
        scorer_hw=scorer_hw,
        resume=raw["resume"],
        test_only_small_fixture=raw["test_only_small_fixture"],
        required_free_bytes=_require_int(
            raw["required_free_bytes"], "required_free_bytes", 1
        ),
    )


def _validate_source(config: MaterializerConfigV1) -> tuple[dict[str, Any], dict[str, np.memmap]]:
    source_binding = config.source_npz
    _require_keys(source_binding, {"path", "bytes", "sha256", "members"}, "source_npz")
    source_file = _verify_expected_file(
        {key: source_binding[key] for key in ("path", "bytes", "sha256")},
        "source_npz",
    )
    source_path = Path(source_file["path"])
    _validate_archive_member_names(source_path, config.test_only_small_fixture)
    members = source_binding["members"]
    if not isinstance(members, dict) or set(members) != set(SOURCE_MEMBER_NAMES):
        raise FreshScorerPlaneMaterializationError(
            f"source members must be exactly {SOURCE_MEMBER_NAMES}"
        )
    arrays: dict[str, np.memmap] = {}
    observed_members: dict[str, Any] = {}
    for name in SOURCE_MEMBER_NAMES:
        binding = members[name]
        if not isinstance(binding, dict):
            raise FreshScorerPlaneMaterializationError(f"source member {name} is invalid")
        _require_keys(binding, {"shape", "dtype", "bytes", "sha256"}, f"member {name}")
        array = open_stored_npy_memmap(source_path, name)
        observed = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": int(array.nbytes),
            "sha256": _array_sha256(array),
        }
        if observed != binding:
            raise FreshScorerPlaneMaterializationError(
                f"source member {name} differs: expected={binding}, observed={observed}"
            )
        arrays[name] = array
        observed_members[name] = observed
    if int(np.asarray(arrays["n_pairs"]).reshape(())) != config.pair_count:
        raise FreshScorerPlaneMaterializationError("source n_pairs differs")
    expected_frames = (*config.camera_hw, 3)
    for name in ("gt_f0", "gt_f1"):
        if arrays[name].shape != (config.pair_count, *expected_frames):
            raise FreshScorerPlaneMaterializationError(f"{name} geometry differs")
        if arrays[name].dtype != np.uint8:
            raise FreshScorerPlaneMaterializationError(f"{name} must be uint8")
    if arrays["gt_poses"].shape != (config.pair_count, 6):
        raise FreshScorerPlaneMaterializationError("gt_poses geometry differs")
    if arrays["gt_poses"].dtype.kind != "f":
        raise FreshScorerPlaneMaterializationError("gt_poses must be floating")
    return (
        {"file": source_file, "members": observed_members},
        arrays,
    )


def _validate_teacher_and_labels(
    config: MaterializerConfigV1,
) -> tuple[dict[str, Any], dict[str, Any], np.memmap]:
    teacher_binding = config.fresh_teacher_receipt
    _require_keys(
        teacher_binding,
        {"path", "bytes", "sha256", "sealed_receipt_sha256", "scorer_pair_batch_size"},
        "fresh_teacher_receipt",
    )
    teacher_file = _verify_expected_file(
        {key: teacher_binding[key] for key in ("path", "bytes", "sha256")},
        "fresh_teacher_receipt",
    )
    teacher = _load_json(Path(teacher_file["path"]), "fresh teacher receipt")
    _verify_seal(teacher, "receipt_sha256")
    if (
        not config.test_only_small_fixture
        and teacher.get("schema") != "tac.taskspace_fresh_teacher_materialization.v1"
    ):
        raise FreshScorerPlaneMaterializationError(
            "production fresh teacher schema differs"
        )
    if teacher["receipt_sha256"] != _require_sha(
        teacher_binding["sealed_receipt_sha256"],
        "fresh_teacher_receipt.sealed_receipt_sha256",
    ):
        raise FreshScorerPlaneMaterializationError("fresh teacher self-hash differs")
    configured_batch = _require_int(
        teacher_binding["scorer_pair_batch_size"],
        "fresh_teacher_receipt.scorer_pair_batch_size",
        1,
    )
    actual_batch = teacher.get("scorer_pair_batch_size", teacher.get("batch_size"))
    if configured_batch != 16 or actual_batch != 16:
        raise FreshScorerPlaneMaterializationError(
            "fresh teacher must use authoritative batch-16 geometry"
        )
    if (
        teacher.get("pair_count") != config.pair_count
        or teacher.get("encoder_only") is not True
        or teacher.get("candidate_payload_allowed") is not False
        or teacher.get("score_claim") is not False
    ):
        raise FreshScorerPlaneMaterializationError(
            "fresh teacher receipt truth/population differs"
        )
    label_binding = config.target_labels
    _require_keys(
        label_binding,
        {"path", "bytes", "sha256", "shape", "dtype"},
        "target_labels",
    )
    label_file = _verify_expected_file(
        {key: label_binding[key] for key in ("path", "bytes", "sha256")},
        "target_labels",
    )
    expected_shape = [config.pair_count, *config.scorer_hw]
    if label_binding["shape"] != expected_shape or label_binding["dtype"] != "uint8":
        raise FreshScorerPlaneMaterializationError("target-label geometry/dtype differs")
    labels = np.memmap(
        label_file["path"],
        mode="r",
        dtype=np.uint8,
        shape=tuple(expected_shape),
    )
    aggregate = teacher.get("target_labels")
    if not isinstance(aggregate, dict):
        raise FreshScorerPlaneMaterializationError("teacher lacks target_labels binding")
    for key in ("path", "bytes", "sha256", "shape", "dtype"):
        if aggregate.get(key) != label_binding[key]:
            raise FreshScorerPlaneMaterializationError(
                f"teacher target_labels.{key} differs from config"
            )
    return (
        {
            "file": teacher_file,
            "sealed_receipt_sha256": teacher["receipt_sha256"],
            "scorer_pair_batch_size": 16,
        },
        {**label_file, "shape": expected_shape, "dtype": "uint8"},
        labels,
    )


def _validate_producers(config: MaterializerConfigV1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roles: set[str] = set()
    for index, binding in enumerate(config.producer_sources):
        role = str(binding["role"])
        if role in roles:
            raise FreshScorerPlaneMaterializationError("producer roles must be unique")
        roles.add(role)
        observed = _verify_expected_file(
            {key: binding[key] for key in ("path", "bytes", "sha256")},
            f"producer_sources[{index}]",
        )
        rows.append({"role": role, **observed})
    required = {"materializer_module", "materializer_cli", "resize_operator_source"}
    if not config.test_only_small_fixture and not required.issubset(roles):
        raise FreshScorerPlaneMaterializationError(
            f"production producer roles lack {sorted(required - roles)}"
        )
    return rows


def _storage_preflight(config: MaterializerConfigV1) -> dict[str, Any]:
    nearest = config.output_root
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent
    usage = shutil.disk_usage(nearest)
    if usage.free < config.required_free_bytes:
        raise FreshScorerPlaneMaterializationError(
            f"storage preflight needs {config.required_free_bytes}, has {usage.free}"
        )
    return {
        "output_root": str(config.output_root),
        "nearest_existing_parent": str(nearest),
        "free_bytes": int(usage.free),
        "required_free_bytes": config.required_free_bytes,
        "admitted": True,
        "scratch_policy": "success_only_atomic_temp_removed_on_failure",
        "completed_stage_policy": "immutable_never_overwritten",
    }


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = canonical_json_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FreshScorerPlaneMaterializationError(
                f"immutable JSON differs: {path}"
            )
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_preflight(config_path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    """Verify complete custody/storage and write only the immutable preflight receipt."""

    config = load_config(config_path)
    source, _arrays = _validate_source(config)
    teacher, labels, _label_array = _validate_teacher_and_labels(config)
    producers = _validate_producers(config)
    storage = _storage_preflight(config)
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": config.run_id,
        "mode": MODE_DIRECT_TASK_LAYERED,
        "semantic_status": SEMANTIC_STATUS_OWED,
        "config": {
            **file_identity(config.path),
            "rfc8785_sha256": payload_sha256(config.raw),
        },
        "source_npz": source,
        "fresh_teacher_receipt": teacher,
        "target_labels": labels,
        "producer_sources": producers,
        "pair_count": config.pair_count,
        "stage_pairs": config.stage_pairs,
        "stage_ranges": [
            [start, start + config.stage_pairs]
            for start in range(0, config.pair_count, config.stage_pairs)
        ],
        "camera_hw": list(config.camera_hw),
        "scorer_hw": list(config.scorer_hw),
        "storage": storage,
        "resume": config.resume,
        "truth": dict(config.raw["truth"]),
        "materialization_started": False,
    }
    path = config.output_root / "00_preflight_receipt.json"
    if path.exists():
        existing = _load_json(path, "existing preflight receipt")
        _verify_seal(existing, "preflight_sha256")
        existing_storage = existing.get("storage")
        if not isinstance(existing_storage, dict):
            raise FreshScorerPlaneMaterializationError(
                "existing preflight lacks storage evidence"
            )
        for key, value in storage.items():
            if key not in {"free_bytes", "nearest_existing_parent"} and (
                existing_storage.get(key) != value
            ):
                raise FreshScorerPlaneMaterializationError(
                    f"existing preflight storage.{key} differs"
                )
        if _require_int(
            existing_storage.get("free_bytes"),
            "existing preflight storage.free_bytes",
            config.required_free_bytes,
        ) < config.required_free_bytes:
            raise FreshScorerPlaneMaterializationError(
                "existing preflight recorded insufficient storage"
            )
        body["storage"] = existing_storage
        receipt = _seal(body, "preflight_sha256")
        if receipt != existing:
            raise FreshScorerPlaneMaterializationError(
                "existing preflight custody differs"
            )
        return path, receipt
    receipt = _seal(body, "preflight_sha256")
    _write_immutable_json(path, receipt)
    return path, receipt


def exact_resize_round_u8(
    operator: DisjointResizeOperator, frame: np.ndarray
) -> U8:
    numerators, denominator = operator.apply_numerators(frame)
    if denominator <= 0 or np.any(numerators < 0):
        raise FreshScorerPlaneMaterializationError(
            "resize escaped nonnegative exact integer domain"
        )
    rounded = (numerators.astype(np.int64) + denominator // 2) // denominator
    if np.any(rounded > 255):
        raise FreshScorerPlaneMaterializationError("rounded scorer plane exceeds uint8")
    return np.ascontiguousarray(rounded.astype(np.uint8))


def _stage_paths(
    root: Path, index: int, start: int, stop: int
) -> tuple[Path, dict[str, Path]]:
    stage_root = root / f"stage_{index:02d}_{start:04d}_{stop:04d}"
    return stage_root / "manifest.json", {
        "y0_u8": stage_root / "Y0.u8",
        "y1_u8": stage_root / "Y1.u8",
        "gt_poses_f32": stage_root / "gt_poses.f32",
    }


def _open_stage_files(
    files: Mapping[str, Path],
    *,
    count: int,
    scorer_hw: tuple[int, int],
) -> tuple[np.memmap, np.memmap, np.memmap]:
    shape = (count, *scorer_hw, 3)
    return (
        np.memmap(files["y0_u8"], mode="r", dtype=np.uint8, shape=shape),
        np.memmap(files["y1_u8"], mode="r", dtype=np.uint8, shape=shape),
        np.memmap(files["gt_poses_f32"], mode="r", dtype=np.float32, shape=(count, 6)),
    )


def _stage_file_rows(files: Mapping[str, Path]) -> dict[str, Any]:
    return {key: file_identity(path) for key, path in sorted(files.items())}


def _derive_stage(
    config: MaterializerConfigV1,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    index: int,
    start: int,
    stop: int,
    preflight: Mapping[str, Any],
    rederive_only: bool,
) -> dict[str, Any]:
    stage_started = time.monotonic()
    manifest_path, files = _stage_paths(config.output_root, index, start, stop)
    count = stop - start
    operator = DisjointResizeOperator.build(
        camera_h=config.camera_hw[0],
        camera_w=config.camera_hw[1],
        scorer_h=config.scorer_hw[0],
        scorer_w=config.scorer_hw[1],
    )
    existing: tuple[np.memmap, np.memmap, np.memmap] | None = None
    temp_paths: dict[str, Path] = {}
    created_final_paths: list[Path] = []
    completed = False
    outputs: tuple[np.memmap, np.memmap, np.memmap]
    if manifest_path.exists():
        if not config.resume:
            raise FreshScorerPlaneMaterializationError(
                f"completed stage exists but resume=false: {manifest_path}"
            )
        existing = _open_stage_files(files, count=count, scorer_hw=config.scorer_hw)
        outputs = existing
    else:
        if rederive_only:
            raise FreshScorerPlaneMaterializationError(
                f"resume rederive lacks manifest: {manifest_path}"
            )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        for key, final_path in files.items():
            if final_path.exists():
                raise FreshScorerPlaneMaterializationError(
                    f"unmanifested completed file exists: {final_path}"
                )
            fd, name = tempfile.mkstemp(
                prefix=f".{final_path.name}.", dir=manifest_path.parent
            )
            os.close(fd)
            temp_paths[key] = Path(name)
        shape = (count, *config.scorer_hw, 3)
        outputs = (
            np.memmap(temp_paths["y0_u8"], mode="w+", dtype=np.uint8, shape=shape),
            np.memmap(temp_paths["y1_u8"], mode="w+", dtype=np.uint8, shape=shape),
            np.memmap(
                temp_paths["gt_poses_f32"],
                mode="w+",
                dtype=np.float32,
                shape=(count, 6),
            ),
        )
    y0, y1, poses = outputs
    try:
        for local, pair_id in enumerate(range(start, stop)):
            expected_y0 = exact_resize_round_u8(
                operator, np.asarray(arrays["gt_f0"][pair_id])
            )
            expected_y1 = exact_resize_round_u8(
                operator, np.asarray(arrays["gt_f1"][pair_id])
            )
            expected_pose = np.asarray(arrays["gt_poses"][pair_id], dtype=np.float32)
            if existing is None:
                y0[local] = expected_y0
                y1[local] = expected_y1
                poses[local] = expected_pose
            elif (
                not np.array_equal(y0[local], expected_y0)
                or not np.array_equal(y1[local], expected_y1)
                or not np.array_equal(poses[local], expected_pose)
            ):
                raise FreshScorerPlaneMaterializationError(
                    f"resume rederive differs at pair {pair_id}"
                )
        if existing is None:
            for value in outputs:
                value.flush()
            del y0, y1, poses, outputs
            for key, temporary in temp_paths.items():
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.link(temporary, files[key])
                created_final_paths.append(files[key])
                temporary.unlink()
        file_rows = _stage_file_rows(files)
        label_slice = np.asarray(labels[start:stop])
        source_pose_slice = np.asarray(arrays["gt_poses"][start:stop])
        existing_manifest = (
            _load_json(manifest_path, "existing stage manifest")
            if manifest_path.exists()
            else None
        )
        if existing_manifest is not None:
            _verify_seal(existing_manifest, "stage_receipt_sha256")
        elapsed_seconds = (
            existing_manifest.get("elapsed_seconds")
            if existing_manifest is not None
            else round(time.monotonic() - stage_started, 6)
        )
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or elapsed_seconds < 0
        ):
            raise FreshScorerPlaneMaterializationError(
                "stage elapsed_seconds is invalid"
            )
        body = {
            "schema": STAGE_SCHEMA,
            "run_id": config.run_id,
            "stage_index": index,
            "pair_range": [start, stop],
            "pair_count": count,
            "preflight_sha256": preflight["preflight_sha256"],
            "config_rfc8785_sha256": preflight["config"]["rfc8785_sha256"],
            "source_npz": preflight["source_npz"],
            "fresh_teacher_receipt": preflight["fresh_teacher_receipt"],
            "target_labels": preflight["target_labels"],
            "producer_sources": preflight["producer_sources"],
            "files": file_rows,
            "source_pose_slice_sha256": _array_sha256(source_pose_slice),
            "fresh_label_slice_sha256": _array_sha256(label_slice),
            "exact_rederive_equal": True,
            "elapsed_seconds": elapsed_seconds,
            "derivation": (
                "gt_f0_gt_f1_to_DisjointResizeOperator_apply_numerators_"
                "nonnegative_half_up_uint8"
            ),
            "pose_authority": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
            "mode": MODE_DIRECT_TASK_LAYERED,
            "semantic_status": SEMANTIC_STATUS_OWED,
            "truth": dict(config.raw["truth"]),
        }
        manifest = _seal(body, "stage_receipt_sha256")
        if existing_manifest is not None:
            if existing_manifest != manifest:
                raise FreshScorerPlaneMaterializationError(
                    f"resumed stage manifest differs: {manifest_path}"
                )
        else:
            _write_immutable_json(manifest_path, manifest)
        completed = True
        return {
            "path": str(manifest_path.resolve()),
            "bytes": int(manifest_path.stat().st_size),
            "sha256": sha256_file(manifest_path),
            "stage_receipt_sha256": manifest["stage_receipt_sha256"],
            "pair_range": [start, stop],
            "files": file_rows,
        }
    finally:
        for value in existing or ():
            del value
        for temporary in temp_paths.values():
            temporary.unlink(missing_ok=True)
        if not completed and existing is None:
            for created in created_final_paths:
                created.unlink(missing_ok=True)


def materialize(
    config_path: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any]]:
    """Materialize or resume every stage, then seal the aggregate receipt."""

    config = load_config(config_path)
    preflight_path, preflight = run_preflight(config.path)
    source, arrays = _validate_source(config)
    teacher, target_labels, labels = _validate_teacher_and_labels(config)
    producer_sources = _validate_producers(config)
    stage_rows: list[dict[str, Any]] = []
    chain = hashlib.sha256()
    for index, start in enumerate(range(0, config.pair_count, config.stage_pairs)):
        stop = start + config.stage_pairs
        row = _derive_stage(
            config,
            arrays,
            labels,
            index=index,
            start=start,
            stop=stop,
            preflight=preflight,
            rederive_only=False,
        )
        chain.update(bytes.fromhex(row["stage_receipt_sha256"]))
        row["digest_chain_sha256"] = chain.hexdigest()
        stage_rows.append(row)
    body = {
        "schema": AGGREGATE_SCHEMA,
        "run_id": config.run_id,
        "mode": MODE_DIRECT_TASK_LAYERED,
        "semantic_status": SEMANTIC_STATUS_OWED,
        "preflight": file_identity(preflight_path),
        "preflight_sha256": preflight["preflight_sha256"],
        "config": preflight["config"],
        "source_npz": source,
        "fresh_teacher_receipt": teacher,
        "target_labels": target_labels,
        "producer_sources": producer_sources,
        "pair_count": config.pair_count,
        "stage_pairs": config.stage_pairs,
        "camera_hw": list(config.camera_hw),
        "scorer_hw": list(config.scorer_hw),
        "stages": stage_rows,
        "stage_digest_chain_sha256": chain.hexdigest(),
        "coverage": {
            "pair_start": 0,
            "pair_stop": config.pair_count,
            "pair_count": config.pair_count,
            "chronological_contiguous": True,
            "all_stage_rederive_equal": True,
        },
        "pose_authority": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
        "truth": dict(config.raw["truth"]),
    }
    aggregate = _seal(body, "aggregate_receipt_sha256")
    aggregate_path = config.output_root / "aggregate_receipt.json"
    _write_immutable_json(aggregate_path, aggregate)
    return aggregate_path, aggregate


def _resolve_bound_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise FreshScorerPlaneMaterializationError(f"{label} path must be a string")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FreshScorerPlaneMaterializationError(f"{label} file is absent: {path}")
    return path


class FreshScorerPlaneOperandLoaderV1:
    """Strict receipt-closed mmap loader for G52 and later codec consumers."""

    def __init__(self, receipt_path: Path, receipt: Mapping[str, Any]) -> None:
        self.receipt_path = receipt_path
        self.receipt = dict(receipt)
        self._validated: list[dict[str, Any]] = []
        self._labels: np.memmap
        self._validate()

    @classmethod
    def open(
        cls,
        aggregate_receipt_path: str | os.PathLike[str],
        *,
        expected_sha256: str | None = None,
    ) -> FreshScorerPlaneOperandLoaderV1:
        path = Path(aggregate_receipt_path).expanduser().resolve()
        if expected_sha256 is not None and sha256_file(path) != _require_sha(
            expected_sha256, "expected_sha256"
        ):
            raise FreshScorerPlaneMaterializationError("aggregate file SHA-256 differs")
        receipt = _load_json(path, "aggregate receipt")
        return cls(path, receipt)

    def _validate(self) -> None:
        if self.receipt.get("schema") != AGGREGATE_SCHEMA:
            raise FreshScorerPlaneMaterializationError("aggregate schema differs")
        _verify_seal(self.receipt, "aggregate_receipt_sha256")
        truth = self.receipt.get("truth")
        if (
            not isinstance(truth, dict)
            or truth.get("research_only") is not True
            or truth.get("score_claim") is not False
            or truth.get("candidate_claim") is not False
            or truth.get("promotion_eligible") is not False
            or truth.get("source_cache_pose_advisory_only") is not True
            or truth.get("fresh_pose_target_custody") is not False
            or truth.get("program_residual_layered_available") is not False
        ):
            raise FreshScorerPlaneMaterializationError("aggregate truth boundary differs")
        if (
            self.receipt.get("mode") != MODE_DIRECT_TASK_LAYERED
            or self.receipt.get("semantic_status") != SEMANTIC_STATUS_OWED
        ):
            raise FreshScorerPlaneMaterializationError("aggregate representation differs")
        preflight_binding = self.receipt.get("preflight")
        if not isinstance(preflight_binding, dict):
            raise FreshScorerPlaneMaterializationError("aggregate preflight is absent")
        preflight_file = _verify_expected_file(preflight_binding, "aggregate.preflight")
        preflight = _load_json(Path(preflight_file["path"]), "aggregate preflight")
        _verify_seal(preflight, "preflight_sha256")
        if preflight["preflight_sha256"] != self.receipt.get("preflight_sha256"):
            raise FreshScorerPlaneMaterializationError(
                "aggregate preflight self-hash differs"
            )
        config_binding = self.receipt.get("config")
        if not isinstance(config_binding, dict):
            raise FreshScorerPlaneMaterializationError("aggregate config is absent")
        _require_keys(
            config_binding,
            {"path", "bytes", "sha256", "rfc8785_sha256"},
            "aggregate.config",
        )
        config_file = _verify_expected_file(
            {key: config_binding[key] for key in ("path", "bytes", "sha256")},
            "aggregate.config",
        )
        config_payload = _load_json(Path(config_file["path"]), "aggregate config")
        if payload_sha256(config_payload) != config_binding["rfc8785_sha256"]:
            raise FreshScorerPlaneMaterializationError(
                "aggregate config canonical hash differs"
            )
        source_binding = self.receipt.get("source_npz")
        if not isinstance(source_binding, dict) or not isinstance(
            source_binding.get("file"), dict
        ):
            raise FreshScorerPlaneMaterializationError(
                "aggregate source binding is absent"
            )
        _verify_expected_file(source_binding["file"], "aggregate.source_npz.file")
        teacher_binding = self.receipt.get("fresh_teacher_receipt")
        if not isinstance(teacher_binding, dict) or not isinstance(
            teacher_binding.get("file"), dict
        ):
            raise FreshScorerPlaneMaterializationError(
                "aggregate teacher binding is absent"
            )
        teacher_file = _verify_expected_file(
            teacher_binding["file"], "aggregate.fresh_teacher_receipt.file"
        )
        teacher = _load_json(Path(teacher_file["path"]), "aggregate teacher receipt")
        _verify_seal(teacher, "receipt_sha256")
        if teacher["receipt_sha256"] != teacher_binding.get("sealed_receipt_sha256"):
            raise FreshScorerPlaneMaterializationError(
                "aggregate teacher self-hash differs"
            )
        producer_rows = self.receipt.get("producer_sources")
        if not isinstance(producer_rows, list) or not producer_rows:
            raise FreshScorerPlaneMaterializationError(
                "aggregate producer sources are absent"
            )
        for index, producer in enumerate(producer_rows):
            if not isinstance(producer, dict):
                raise FreshScorerPlaneMaterializationError(
                    f"aggregate producer {index} is invalid"
                )
            _require_keys(
                producer,
                {"role", "path", "bytes", "sha256"},
                f"aggregate.producer_sources[{index}]",
            )
            _verify_expected_file(
                {key: producer[key] for key in ("path", "bytes", "sha256")},
                f"aggregate.producer_sources[{index}]",
            )
        pair_count = _require_int(self.receipt.get("pair_count"), "pair_count", 1)
        stage_pairs = _require_int(self.receipt.get("stage_pairs"), "stage_pairs", 1)
        scorer_hw = _require_hw(self.receipt.get("scorer_hw"), "scorer_hw")
        label_binding = self.receipt.get("target_labels")
        if not isinstance(label_binding, dict):
            raise FreshScorerPlaneMaterializationError("aggregate labels are absent")
        label_file = _verify_expected_file(
            {key: label_binding[key] for key in ("path", "bytes", "sha256")},
            "aggregate.target_labels",
        )
        if (
            label_binding.get("shape") != [pair_count, *scorer_hw]
            or label_binding.get("dtype") != "uint8"
        ):
            raise FreshScorerPlaneMaterializationError("aggregate label geometry differs")
        self._labels = np.memmap(
            label_file["path"],
            mode="r",
            dtype=np.uint8,
            shape=(pair_count, *scorer_hw),
        )
        rows = self.receipt.get("stages")
        if not isinstance(rows, list) or len(rows) != pair_count // stage_pairs:
            raise FreshScorerPlaneMaterializationError("aggregate stage count differs")
        chain = hashlib.sha256()
        expected_start = 0
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise FreshScorerPlaneMaterializationError("aggregate stage row is invalid")
            manifest_path = _resolve_bound_path(row.get("path"), f"stage {index}")
            observed_manifest_identity = file_identity(manifest_path)
            for key in ("path", "bytes", "sha256"):
                if observed_manifest_identity[key] != row.get(key):
                    raise FreshScorerPlaneMaterializationError(
                        f"stage {index} manifest identity differs"
                    )
            manifest = _load_json(manifest_path, f"stage {index} manifest")
            _verify_seal(manifest, "stage_receipt_sha256")
            if (
                manifest["stage_receipt_sha256"] != row.get("stage_receipt_sha256")
                or manifest.get("stage_index") != index
                or manifest.get("pair_range")
                != [expected_start, expected_start + stage_pairs]
                or manifest.get("exact_rederive_equal") is not True
                or manifest.get("pose_authority")
                != "SEALED_SOURCE_CACHE_ADVISORY_ONLY"
            ):
                raise FreshScorerPlaneMaterializationError(
                    f"stage {index} semantic/range receipt differs"
                )
            files = manifest.get("files")
            if not isinstance(files, dict):
                raise FreshScorerPlaneMaterializationError(f"stage {index} files absent")
            for name in ("y0_u8", "y1_u8", "gt_poses_f32"):
                binding = files.get(name)
                if not isinstance(binding, dict):
                    raise FreshScorerPlaneMaterializationError(
                        f"stage {index} {name} binding absent"
                    )
                _verify_expected_file(binding, f"stage {index} {name}")
            chain.update(bytes.fromhex(manifest["stage_receipt_sha256"]))
            if chain.hexdigest() != row.get("digest_chain_sha256"):
                raise FreshScorerPlaneMaterializationError(
                    f"stage {index} digest chain differs"
                )
            self._validated.append(
                {
                    "range": (expected_start, expected_start + stage_pairs),
                    "files": files,
                }
            )
            expected_start += stage_pairs
        if (
            expected_start != pair_count
            or chain.hexdigest() != self.receipt.get("stage_digest_chain_sha256")
        ):
            raise FreshScorerPlaneMaterializationError("aggregate coverage/chain differs")
        self.pair_count = pair_count
        self.stage_pairs = stage_pairs
        self.scorer_hw = scorer_hw

    def _open_stage(self, row: Mapping[str, Any]) -> FreshScorerPlaneStageV1:
        start, stop = row["range"]
        count = stop - start
        files = row["files"]
        shape = (count, *self.scorer_hw, 3)
        y0 = np.memmap(files["y0_u8"]["path"], mode="r", dtype=np.uint8, shape=shape)
        y1 = np.memmap(files["y1_u8"]["path"], mode="r", dtype=np.uint8, shape=shape)
        poses = np.memmap(
            files["gt_poses_f32"]["path"],
            mode="r",
            dtype=np.float32,
            shape=(count, 6),
        )
        pair_ids = np.arange(start, stop, dtype=np.int64)
        pair_ids.flags.writeable = False
        return FreshScorerPlaneStageV1(
            pair_range=(start, stop),
            pair_ids=pair_ids,
            y0_u8=y0,
            y1_u8=y1,
            target_labels_u8=self._labels[start:stop],
            gt_poses_f32=poses,
        )

    def iter_stages(
        self, max_pairs: int = PRODUCTION_STAGE_PAIRS
    ) -> Iterator[FreshScorerPlaneStageV1]:
        maximum = _require_int(max_pairs, "max_pairs", 1)
        for row in self._validated:
            start, stop = row["range"]
            if stop - start > maximum:
                raise FreshScorerPlaneMaterializationError(
                    f"immutable stage has {stop - start} pairs, exceeds max_pairs={maximum}"
                )
            yield self._open_stage(row)

    def iter_chunks(self, max_pairs: int) -> Iterator[FreshScorerPlaneStageV1]:
        maximum = _require_int(max_pairs, "max_pairs", 1)
        for row in self._validated:
            stage = self._open_stage(row)
            for offset in range(0, len(stage.pair_ids), maximum):
                stop_offset = min(offset + maximum, len(stage.pair_ids))
                start = int(stage.pair_ids[offset])
                stop = int(stage.pair_ids[stop_offset - 1]) + 1
                ids = stage.pair_ids[offset:stop_offset]
                ids.flags.writeable = False
                yield FreshScorerPlaneStageV1(
                    pair_range=(start, stop),
                    pair_ids=ids,
                    y0_u8=stage.y0_u8[offset:stop_offset],
                    y1_u8=stage.y1_u8[offset:stop_offset],
                    target_labels_u8=stage.target_labels_u8[offset:stop_offset],
                    gt_poses_f32=stage.gt_poses_f32[offset:stop_offset],
                )
