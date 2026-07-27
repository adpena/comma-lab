# SPDX-License-Identifier: MIT
"""Physical deploy/full-state custody for ``tac.fresh_producer_lineage.v1``.

The compact deploy checkpoint alone cannot prove even its current training
state.  This module reopens the deploy checkpoint and its mandatory full-state
resume companion by exact SHA-256, recomputes the cold-root *tuple*, recomputes
the current semantic-state hash and checkpoint-node identity, and proves that
the deploy tensors are exactly the companion EMA shadow.

Important scope: one pair proves only same-checkpoint node integrity.  Its
``parent_checkpoint_id_sha256`` is still an opaque predecessor assertion until
a higher-level recursive chain opener physically reopens every parent back to
the zero-parent root.  This module never calls one pair a complete trajectory.

This is deliberately import-light and independent of the MLX trainer.  The
trainer can later import the pure hash functions from this module without
changing their byte-level definitions.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

FRESH_PRODUCER_LINEAGE_SCHEMA: Final = "tac.fresh_producer_lineage.v1"
FRESH_PHYSICAL_CHECKPOINT_NODE_SCHEMA: Final = (
    "tac.fresh_producer_physical_checkpoint_node.v1"
)
RESUME_SEMANTIC_SCHEMA: Final = "levelset_full_state.v3"
LEGACY_RESUME_SEMANTIC_SCHEMA: Final = "levelset_full_state.v2"
RESUME_EVENT_LEDGER_SCHEMA: Final = "levelset_resume_event_ledger.v1"
ROOT_PARENT_CHECKPOINT_ID: Final = "0" * 64
G109_TARGET_PROJECTION_SHA_KEY: Final = "__cfg_g109_target_projection_sha256"
G111_COMPLETE_TRAJECTORY_SCHEMA: Final = (
    "g111_complete_trajectory_state.v1"
)
G111_COMPLETE_TRAJECTORY_KEY: Final = (
    "__resume_complete_trajectory_manifest_json"
)
G111_TRAJECTORY_COMPONENTS: Final = (
    "primary_model_ema_optimizer_family",
    "protected_seed_optimizer_support",
    "fresh_root_physical_lineage",
    "rng_streams",
    "event_gates_and_duplicate_booleans",
    "stage_transition_rewarmup",
    "spike_rollback_and_last_good_snapshot",
    "ladder_state",
    "tail_controller_and_verdict_inputs",
    "verdict_journal_and_sensor_histories",
    "pending_verdict_reducer_boundary",
    "jacobian_basin_state",
    "polyak_atomic_state",
    "best_and_stage_checkpoint_bookkeeping",
)

RESUME_LIVE_PREFIX: Final = "liveP__"
RESUME_EMA_PREFIX: Final = "emaP__"
RESUME_OPT_PREFIX: Final = "optP__"
RESUME_SEED_LIVE_PREFIX: Final = "seedP__"
RESUME_SEED_OPT_PREFIX: Final = "seedOptP__"
RESUME_POLYAK_PREFIX: Final = "polyakM__"
RESUME_RNG_KEYS: Final = (
    "__rng_np_algo",
    "__rng_np_keys",
    "__rng_np_pos",
    "__rng_np_has_gauss",
    "__rng_np_cached_gauss",
)
LEGACY_EVENT_PREFIXES: Final = (
    "__mg_",
    "__lbg_",
    "__scg_",
    "__tsg_",
    "__pag_",
    "__evt_",
    "__posegate_",
    "__dtp_event_mark_",
)
FRESH_ROOT_KEYS: Final = (
    "__cfg_fresh_lineage_schema",
    "__cfg_fresh_seed",
    "__cfg_fresh_lineage_root_sha256",
    "__cfg_fresh_initial_state_sha256",
    "__cfg_fresh_dsl_compile_hash",
    "__cfg_fresh_target_projection_sha256",
)
FRESH_DEPLOY_KEYS: Final = (
    "__cfg_fresh_producer",
    *FRESH_ROOT_KEYS,
    "__cfg_fresh_current_launch_dsl_compile_hash",
)
FRESH_LINEAGE_DERIVED_CFG_KEYS: Final = frozenset(
    {
        "__cfg_fresh_producer",
        *FRESH_ROOT_KEYS,
        "__cfg_fresh_current_launch_dsl_compile_hash",
        "__cfg_fresh_lineage_parent_checkpoint_id_sha256",
        "__cfg_fresh_lineage_state_sha256",
        "__cfg_fresh_lineage_checkpoint_id_sha256",
        "__cfg_fresh_lineage_epoch",
        "__cfg_fresh_lineage_stage",
    }
)
FRESH_RESUME_KEYS: Final = (
    *FRESH_DEPLOY_KEYS,
    *tuple(sorted(FRESH_LINEAGE_DERIVED_CFG_KEYS)),
)


class FreshProducerLineageV1Error(ValueError):
    """A physical pair or a lineage/full-state invariant failed closed."""


@dataclass(frozen=True, slots=True)
class PhysicalNpzV1:
    """One exact physical NPZ reopen and its immutable identity."""

    path: Path
    sha256: str
    bytes: int
    device: int
    inode: int
    mtime_ns: int
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class FreshProducerCheckpointPairV1:
    """Independently validated same-checkpoint deploy/full-state node."""

    deploy: PhysicalNpzV1
    resume: PhysicalNpzV1
    seed: int
    root_sha256: str
    initial_state_sha256: str
    root_dsl_compile_hash: str
    current_launch_dsl_compile_hash: str
    target_projection_sha256: str
    parent_checkpoint_id_sha256: str
    state_sha256: str
    checkpoint_id_sha256: str
    epoch: int
    stage: str
    live_tensor_count: int
    ema_tensor_count: int
    optimizer_tensor_count: int
    seed_tensor_count: int
    seed_optimizer_tensor_count: int
    polyak_tensor_count: int
    config_array_count: int
    rng_complete: bool = True
    event_ledger_complete: bool = True
    deploy_equals_ema: bool = True
    film_stiefel: bool = False
    complete_trajectory_proven: bool = False
    complete_state_manifest_proven: bool = False


@dataclass(frozen=True, slots=True)
class FreshProducerPhysicalCheckpointNodeV1:
    """One immutable content-addressed node and its same-checkpoint proof."""

    receipt_path: Path
    receipt_sha256: str
    receipt_bytes: int
    sequence_index: int
    pair: FreshProducerCheckpointPairV1
    parent_receipt_path: Path | None
    parent_receipt_sha256: str | None
    complete_trajectory_proven: bool = False


@dataclass(frozen=True, slots=True)
class FreshProducerPhysicalCheckpointChainV1:
    """A physically reopened zero-parent-to-current lineage chain."""

    nodes: tuple[FreshProducerPhysicalCheckpointNodeV1, ...]
    current: FreshProducerPhysicalCheckpointNodeV1
    root_sha256: str
    current_launch_dsl_compile_hash: str
    complete_trajectory_proven: bool


def require_sha256(value: object, *, label: str) -> str:
    """Return a canonical lowercase SHA-256 or fail closed."""

    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FreshProducerLineageV1Error(
            f"{label} must be a canonical lowercase SHA-256"
        )
    return value


def canonical_lineage_sha256(value: Mapping[str, Any]) -> str:
    """Hash the trainer's canonical ASCII JSON identity encoding."""

    try:
        payload = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FreshProducerLineageV1Error(
            "fresh lineage identity is not canonical ASCII JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def sha256_array_mapping(arrays: Mapping[str, Any]) -> str:
    """Hash a tensor mapping exactly as the producer does."""

    digest = hashlib.sha256()
    for key in sorted(str(name) for name in arrays):
        if key not in arrays:
            raise FreshProducerLineageV1Error(
                "array mapping contains a non-string key alias"
            )
        value = np.ascontiguousarray(np.asarray(arrays[key]))
        if value.dtype.hasobject:
            raise FreshProducerLineageV1Error(
                f"lineage array {key} has forbidden object dtype"
            )
        key_bytes = key.encode("utf-8")
        dtype_bytes = value.dtype.str.encode("ascii")
        shape_bytes = json.dumps(
            [int(dimension) for dimension in value.shape],
            separators=(",", ":"),
        ).encode("ascii")
        for field in (key_bytes, dtype_bytes, shape_bytes):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
        raw = memoryview(value).cast("B")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def fresh_producer_root_sha256(
    *,
    seed: int,
    dsl_compile_hash: str,
    target_projection_sha256: str,
    initial_state_sha256: str,
) -> str:
    """Recompute the deterministic cold root from its physical fields."""

    if type(seed) is not int or seed < 0:
        raise FreshProducerLineageV1Error(
            "fresh lineage seed must be a nonnegative integer"
        )
    return canonical_lineage_sha256(
        {
            "schema": FRESH_PRODUCER_LINEAGE_SCHEMA,
            "seed": seed,
            "dsl_compile_hash": require_sha256(
                dsl_compile_hash,
                label="root DSL compile hash",
            ),
            "target_projection_sha256": require_sha256(
                target_projection_sha256,
                label="target projection SHA-256",
            ),
            "initial_state_sha256": require_sha256(
                initial_state_sha256,
                label="initial state SHA-256",
            ),
        }
    )


def _lineage_config_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise FreshProducerLineageV1Error(
            "fresh lineage configuration cannot contain object dtype"
        )
    if array.size == 1:
        item = array.item()
        return item.item() if isinstance(item, np.generic) else item
    return array.tolist()


def fresh_resume_semantic_state_sha256(
    *,
    live_state: Mapping[str, Any],
    ema_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    seed_state: Mapping[str, Any],
    seed_optimizer_state: Mapping[str, Any],
    polyak_state: Mapping[str, Any],
    config_state: Mapping[str, Any],
) -> str:
    """Hash every producer-defined trajectory and control state leg."""

    cfg = {
        str(key): _lineage_config_value(value)
        for key, value in config_state.items()
        if str(key) not in FRESH_LINEAGE_DERIVED_CFG_KEYS
    }
    return canonical_lineage_sha256(
        {
            "live_sha256": sha256_array_mapping(live_state),
            "ema_sha256": sha256_array_mapping(ema_state),
            "optimizer_sha256": sha256_array_mapping(optimizer_state),
            "seed_sha256": sha256_array_mapping(seed_state),
            "seed_optimizer_sha256": sha256_array_mapping(seed_optimizer_state),
            "polyak_sha256": sha256_array_mapping(polyak_state),
            "config_sha256": canonical_lineage_sha256(cfg),
        }
    )


def split_resume_state(
    arrays: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Split a full-state sidecar exactly as the trainer's loader does."""

    live: dict[str, Any] = {}
    ema: dict[str, Any] = {}
    optimizer: dict[str, Any] = {}
    seed: dict[str, Any] = {}
    seed_optimizer: dict[str, Any] = {}
    polyak: dict[str, Any] = {}
    cfg: dict[str, Any] = {}
    for key, value in arrays.items():
        name = str(key)
        if name.startswith(RESUME_LIVE_PREFIX):
            live[name[len(RESUME_LIVE_PREFIX) :]] = value
        elif name.startswith(RESUME_EMA_PREFIX):
            ema[name[len(RESUME_EMA_PREFIX) :]] = value
        elif name.startswith(RESUME_OPT_PREFIX):
            optimizer[name[len(RESUME_OPT_PREFIX) :]] = value
        elif name.startswith(RESUME_SEED_LIVE_PREFIX):
            seed[name[len(RESUME_SEED_LIVE_PREFIX) :]] = value
        elif name.startswith(RESUME_SEED_OPT_PREFIX):
            seed_optimizer[name[len(RESUME_SEED_OPT_PREFIX) :]] = value
        elif name.startswith(RESUME_POLYAK_PREFIX):
            polyak[name[len(RESUME_POLYAK_PREFIX) :]] = value
        elif name.startswith("__"):
            cfg[name] = value
        else:
            raise FreshProducerLineageV1Error(
                f"fresh resume sidecar has an unclassified member: {name}"
            )
    return live, ema, optimizer, seed, seed_optimizer, polyak, cfg


def fresh_resume_semantic_state_sha256_from_flat(
    arrays: Mapping[str, Any],
) -> str:
    """Recompute the full-state semantic hash from a flat resume sidecar."""

    live, ema, optimizer, seed, seed_optimizer, polyak, cfg = split_resume_state(
        arrays
    )
    return fresh_resume_semantic_state_sha256(
        live_state=live,
        ema_state=ema,
        optimizer_state=optimizer,
        seed_state=seed,
        seed_optimizer_state=seed_optimizer,
        polyak_state=polyak,
        config_state=cfg,
    )


def fresh_checkpoint_id_sha256(
    *,
    root_sha256: str,
    parent_checkpoint_id_sha256: str,
    state_sha256: str,
    epoch: int,
    stage: str,
) -> str:
    """Recompute one checkpoint-chain node identity."""

    if type(epoch) is not int or epoch < 0:
        raise FreshProducerLineageV1Error(
            "fresh lineage epoch must be a nonnegative integer"
        )
    if type(stage) is not str or not stage or not stage.isascii():
        raise FreshProducerLineageV1Error(
            "fresh lineage stage must be nonempty ASCII text"
        )
    return canonical_lineage_sha256(
        {
            "schema": FRESH_PRODUCER_LINEAGE_SCHEMA,
            "root_sha256": require_sha256(
                root_sha256,
                label="fresh lineage root SHA-256",
            ),
            "parent_checkpoint_id_sha256": require_sha256(
                parent_checkpoint_id_sha256,
                label="fresh lineage parent checkpoint id",
            ),
            "state_sha256": require_sha256(
                state_sha256,
                label="fresh lineage state SHA-256",
            ),
            "epoch": epoch,
            "stage": stage,
        }
    )


def _strict_npz_members(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if not members or len(names) != len(set(names)):
                raise FreshProducerLineageV1Error(
                    "checkpoint NPZ has no members or duplicate ZIP members"
                )
            for member in members:
                name = member.filename
                if (
                    member.is_dir()
                    or "/" in name
                    or "\\" in name
                    or not name.endswith(".npy")
                    or name in {".npy", "..npy"}
                ):
                    raise FreshProducerLineageV1Error(
                        "checkpoint NPZ has a noncanonical member name"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise FreshProducerLineageV1Error(
            "checkpoint is not a strict NPZ ZIP"
        ) from exc


def open_physical_npz(
    checkpoint: Path,
    *,
    expected_sha256: str,
    label: str,
) -> PhysicalNpzV1:
    """Reopen one physical regular file twice around strict NPZ parsing."""

    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        raise FreshProducerLineageV1Error(
            f"{label} path must be absolute physical custody"
        )
    try:
        if path.resolve(strict=True) != path:
            raise FreshProducerLineageV1Error(
                f"{label} path must be canonical physical custody"
            )
    except OSError as exc:
        raise FreshProducerLineageV1Error(
            f"{label} physical path cannot be resolved"
        ) from exc
    expected = require_sha256(expected_sha256, label=f"{label} SHA-256")
    try:
        before = path.lstat()
    except OSError as exc:
        raise FreshProducerLineageV1Error(f"{label} is not readable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FreshProducerLineageV1Error(
            f"{label} must be a regular non-symlink file"
        )
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise FreshProducerLineageV1Error(
            f"{label} physical SHA-256 differs"
        )
    _strict_npz_members(payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {
                key: np.asarray(archive[key]).copy(order="C")
                for key in archive.files
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise FreshProducerLineageV1Error(
            f"{label} contains an invalid NPY member"
        ) from exc
    after = path.lstat()
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise FreshProducerLineageV1Error(
            f"{label} changed during recursive reopen"
        )
    for key, value in arrays.items():
        if value.dtype.hasobject:
            raise FreshProducerLineageV1Error(
                f"{label} array {key} has forbidden object dtype"
            )
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise FreshProducerLineageV1Error(
                f"{label} array {key} contains a non-finite value"
            )
        value.setflags(write=False)
    return PhysicalNpzV1(
        path=path,
        sha256=expected,
        bytes=len(payload),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mtime_ns=int(after.st_mtime_ns),
        arrays=arrays,
    )


def _scalar(arrays: Mapping[str, Any], key: str) -> object:
    try:
        value = np.asarray(arrays[key])
    except KeyError as exc:
        raise FreshProducerLineageV1Error(
            f"checkpoint is missing required scalar {key}"
        ) from exc
    if value.size != 1:
        raise FreshProducerLineageV1Error(
            f"checkpoint scalar {key} must contain exactly one value"
        )
    return value.item()


def _require_int64_scalar(
    arrays: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
) -> int:
    array = np.asarray(arrays.get(key))
    if array.dtype != np.dtype(np.int64) or array.shape != ():
        raise FreshProducerLineageV1Error(
            f"{key} must be an exact int64 scalar"
        )
    result = int(array.item())
    if result < minimum:
        raise FreshProducerLineageV1Error(
            f"{key} must be at least {minimum}"
        )
    return result


def _validate_complete_trajectory_manifest_skeleton(
    *,
    arrays: Mapping[str, Any],
    seed_active: bool,
    polyak_active: bool,
) -> None:
    """Validate the G111 inventory skeleton without claiming state closure."""

    try:
        raw = _scalar(arrays, G111_COMPLETE_TRAJECTORY_KEY)
    except FreshProducerLineageV1Error as exc:
        raise FreshProducerLineageV1Error(
            "full-state v3 resume companion lacks the complete trajectory-state "
            "manifest"
        ) from exc
    if type(raw) is not str:
        raise FreshProducerLineageV1Error(
            "complete trajectory-state manifest must be scalar JSON text"
        )
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FreshProducerLineageV1Error(
            "complete trajectory-state manifest is malformed"
        ) from exc
    if not isinstance(parsed, dict):
        raise FreshProducerLineageV1Error(
            "complete trajectory-state manifest must be a JSON object"
        )
    if parsed.get("schema") != G111_COMPLETE_TRAJECTORY_SCHEMA:
        raise FreshProducerLineageV1Error(
            "complete trajectory-state manifest schema differs"
        )
    components = parsed.get("components")
    if not isinstance(components, list):
        raise FreshProducerLineageV1Error(
            "complete trajectory-state components must be a list"
        )
    if raw != json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ):
        raise FreshProducerLineageV1Error(
            "complete trajectory-state manifest is not canonical JSON"
        )
    expected_names = set(G111_TRAJECTORY_COMPONENTS)
    names: list[str] = []
    for component in components:
        if not isinstance(component, dict) or type(component.get("name")) is not str:
            raise FreshProducerLineageV1Error(
                "complete trajectory-state manifest has a malformed component"
            )
        names.append(component["name"])
    if (
        len(names) != len(set(names))
        or set(names) != expected_names
        or len(names) != len(G111_TRAJECTORY_COMPONENTS)
    ):
        raise FreshProducerLineageV1Error(
            "complete trajectory-state manifest must inventory each of the "
            "14 domains exactly once"
        )

    known_activity = {
        "primary_model_ema_optimizer_family": True,
        "protected_seed_optimizer_support": bool(seed_active),
        "fresh_root_physical_lineage": True,
        "rng_streams": True,
        "polyak_atomic_state": bool(polyak_active),
        "best_and_stage_checkpoint_bookkeeping": True,
    }
    claimed_keys: set[str] = set()
    for component in components:
        name = component["name"]
        active = component.get("active")
        keys = component.get("keys")
        if type(active) is not bool or not isinstance(keys, list):
            raise FreshProducerLineageV1Error(
                f"complete trajectory-state component {name!r} is malformed"
            )
        if name in known_activity and active != known_activity[name]:
            raise FreshProducerLineageV1Error(
                f"complete trajectory-state activity differs for {name!r}"
            )
        if (active and not keys) or (not active and keys):
            raise FreshProducerLineageV1Error(
                f"complete trajectory-state component {name!r} does not "
                "explicitly match its activity"
            )
        for key in keys:
            if (
                type(key) is not str
                or not key
                or key == G111_COMPLETE_TRAJECTORY_KEY
                or key in claimed_keys
            ):
                raise FreshProducerLineageV1Error(
                    "complete trajectory-state manifest has duplicate or "
                    "colliding state keys"
                )
            if key not in arrays:
                raise FreshProducerLineageV1Error(
                    f"complete trajectory-state manifest references missing "
                    f"state {key!r}"
                )
            claimed_keys.add(key)
    raise FreshProducerLineageV1Error(
        "full-state v3 complete trajectory-state manifest is skeleton-only: "
        "component-specific required keysets and reverse state coverage are not "
        "implemented, so complete trajectory proof is refused"
    )


def _validate_full_state_semantics(
    *,
    arrays: Mapping[str, Any],
    live: Mapping[str, Any],
    ema: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    seed: Mapping[str, Any],
    seed_optimizer: Mapping[str, Any],
    polyak: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[int, str, bool]:
    if not live or not ema:
        raise FreshProducerLineageV1Error(
            "resume companion must contain complete live and EMA model state"
        )
    if set(live) != set(ema):
        raise FreshProducerLineageV1Error(
            "resume companion live/EMA tensor key sets differ"
        )
    for domain_name, state in (("live", live), ("EMA", ema)):
        for key, value in state.items():
            array = np.asarray(value)
            if array.dtype != np.dtype(np.float32):
                raise FreshProducerLineageV1Error(
                    f"resume companion {domain_name} tensor {key} is not float32"
                )
    semantic_schema = _scalar(cfg, "__resume_semantic_schema")
    if semantic_schema not in {
        RESUME_SEMANTIC_SCHEMA,
        LEGACY_RESUME_SEMANTIC_SCHEMA,
    }:
        raise FreshProducerLineageV1Error(
            "resume companion semantic schema is not supported full-state v2/v3"
        )
    if int(_scalar(cfg, "__resume_has_opt")) != 1 or not optimizer:
        raise FreshProducerLineageV1Error(
            "resume companion lacks complete optimizer state"
        )
    seed_active = bool(
        int(
            cfg.get("__cfg_seed_islands", 0)
            if semantic_schema == LEGACY_RESUME_SEMANTIC_SCHEMA
            else _scalar(cfg, "__cfg_seed_islands")
        )
    )
    if semantic_schema == LEGACY_RESUME_SEMANTIC_SCHEMA:
        if seed_active or seed or seed_optimizer:
            raise FreshProducerLineageV1Error(
                "legacy full-state v2 fresh chains cannot carry an active "
                "protected-island seed"
            )
        has_seed = int(cfg.get("__resume_has_seed", 0))
        if has_seed != 0:
            raise FreshProducerLineageV1Error(
                "legacy full-state v2 fresh chains cannot declare protected-seed state"
            )
        complete_state_manifest_proven = False
    else:
        if _scalar(cfg, "__resume_primary_optimizer_family") not in {
            "adamw",
            "muon_multioptimizer",
        }:
            raise FreshProducerLineageV1Error(
                "resume companion primary optimizer family is missing or unknown"
            )
        try:
            active_components = json.loads(
                str(_scalar(cfg, "__resume_active_trainable_components_json"))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FreshProducerLineageV1Error(
                "resume companion active-trainable-component manifest is malformed"
            ) from exc
        expected_components = [
            "primary_model", *(["island_seed"] if seed_active else [])
        ]
        if active_components != expected_components:
            raise FreshProducerLineageV1Error(
                "resume companion active-trainable-component manifest differs"
            )
        complete_state_manifest_proven = False
    has_seed = (
        0
        if semantic_schema == LEGACY_RESUME_SEMANTIC_SCHEMA
        else int(_scalar(cfg, "__resume_has_seed"))
    )
    if seed_active:
        if has_seed != 1 or not seed or not seed_optimizer:
            raise FreshProducerLineageV1Error(
                "resume companion lacks complete protected-island seed state"
            )
        try:
            seed_manifest = json.loads(
                str(_scalar(cfg, "__resume_seed_state_manifest_json"))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FreshProducerLineageV1Error(
                "resume companion protected-seed manifest is malformed"
            ) from exc
        if (
            seed_manifest.get("schema") != "tac.sparse_auxiliary_resume.v1"
            or seed_manifest.get("component") != "island_seed"
            or seed_manifest.get("optimizer_family") != "adamw"
        ):
            raise FreshProducerLineageV1Error(
                "resume companion protected-seed manifest schema is wrong"
            )
        seed_dense_shape = tuple(
            int(v) for v in seed_manifest.get("dense_shape", [])
        )
        if len(seed_dense_shape) < 2:
            raise FreshProducerLineageV1Error(
                "resume companion protected-seed dense shape is malformed"
            )
        for domain_name, domain, entries_key in (
            ("seed live", seed, "live"),
            ("seed optimizer", seed_optimizer, "optimizer"),
        ):
            entries = seed_manifest.get(entries_key)
            if not isinstance(entries, list):
                raise FreshProducerLineageV1Error(
                    f"resume companion {domain_name} manifest inventory is malformed"
                )
            entry_by_key = {
                str(entry.get("key", "")): entry
                for entry in entries
                if isinstance(entry, dict)
            }
            if not all(entry_by_key) or set(entry_by_key) != set(domain):
                raise FreshProducerLineageV1Error(
                    f"resume companion {domain_name} keyset differs from its manifest"
                )
            for key, value in domain.items():
                array = np.asarray(value)
                entry = entry_by_key[key]
                try:
                    logical_shape = tuple(
                        int(v) for v in entry.get("logical_shape", [])
                    )
                    expected_dtype = np.dtype(str(entry.get("dtype", "")))
                except (TypeError, ValueError) as exc:
                    raise FreshProducerLineageV1Error(
                        f"resume companion {domain_name} manifest entry is malformed"
                    ) from exc
                if array.dtype != expected_dtype:
                    raise FreshProducerLineageV1Error(
                        f"resume companion {domain_name} dtype differs at {key}"
                    )
                encoding = str(entry.get("encoding", ""))
                if encoding == "full" and tuple(array.shape) != logical_shape:
                    raise FreshProducerLineageV1Error(
                        f"resume companion {domain_name} shape differs at {key}"
                    )
                if encoding == "support_rows":
                    support_count = int(seed_manifest.get("support_count", -1))
                    channel_shape = logical_shape[len(seed_dense_shape) - 1 :]
                    expected_packed_shape = (support_count, *channel_shape)
                    if (
                        logical_shape != seed_dense_shape
                        or tuple(array.shape) != expected_packed_shape
                    ):
                        raise FreshProducerLineageV1Error(
                            f"resume companion {domain_name} packed support shape differs at {key}"
                        )
                elif encoding != "full":
                    raise FreshProducerLineageV1Error(
                        f"resume companion {domain_name} encoding is unknown at {key}"
                    )
        require_sha256(
            str(seed_manifest.get("support_geometry_sha256", "")),
            label="protected-seed support geometry SHA-256",
        )
    elif has_seed != 0 or seed or seed_optimizer:
        raise FreshProducerLineageV1Error(
            "resume companion carries protected-seed state while seed config is inactive"
        )
    for key in RESUME_RNG_KEYS:
        if key not in cfg:
            raise FreshProducerLineageV1Error(
                f"resume companion lacks RNG state {key}"
            )
    if (
        _scalar(cfg, "__rng_np_algo") != "MT19937"
        or np.asarray(cfg["__rng_np_keys"]).dtype != np.dtype(np.uint32)
        or np.asarray(cfg["__rng_np_keys"]).shape != (624,)
    ):
        raise FreshProducerLineageV1Error(
            "resume companion global NumPy RNG state is not exact MT19937"
        )
    rng_position = int(_scalar(cfg, "__rng_np_pos"))
    rng_has_gauss = int(_scalar(cfg, "__rng_np_has_gauss"))
    rng_cached_gauss = float(_scalar(cfg, "__rng_np_cached_gauss"))
    if (
        not 0 <= rng_position <= 624
        or rng_has_gauss not in (0, 1)
        or not np.isfinite(rng_cached_gauss)
    ):
        raise FreshProducerLineageV1Error(
            "resume companion global NumPy RNG scalars are invalid"
        )
    if "__rng_hardness_json" in cfg:
        hardness_json = _scalar(cfg, "__rng_hardness_json")
        if type(hardness_json) is not str:
            raise FreshProducerLineageV1Error(
                "resume companion hardness RNG state must be scalar JSON text"
            )
        try:
            hardness = json.loads(hardness_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FreshProducerLineageV1Error(
                "resume companion hardness RNG state is malformed"
            ) from exc
        if not isinstance(hardness, dict):
            raise FreshProducerLineageV1Error(
                "resume companion hardness RNG state must be a JSON object"
            )
    recent_losses = np.asarray(arrays.get("__recent_losses"))
    if (
        "__recent_losses" not in arrays
        or recent_losses.dtype != np.dtype(np.float64)
        or recent_losses.ndim != 1
        or recent_losses.size > 50
        or not np.isfinite(recent_losses).all()
    ):
        raise FreshProducerLineageV1Error(
            "resume companion recent-loss controller state is not finite float64[<=50]"
        )
    stage = _scalar(cfg, "__resume_stage")
    if type(stage) is not str or not stage or not stage.isascii():
        raise FreshProducerLineageV1Error(
            "resume companion stage must be nonempty ASCII text"
        )
    ledger_text = _scalar(cfg, "__resume_event_ledger_json")
    if type(ledger_text) is not str:
        raise FreshProducerLineageV1Error(
            "resume companion event ledger must be scalar JSON text"
        )
    try:
        ledger = json.loads(ledger_text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FreshProducerLineageV1Error(
            "resume companion event ledger is malformed"
        ) from exc
    if not isinstance(ledger, dict) or ledger.get("schema") != RESUME_EVENT_LEDGER_SCHEMA:
        raise FreshProducerLineageV1Error(
            "resume companion event ledger schema is wrong"
        )
    if ledger.get("stage") != stage:
        raise FreshProducerLineageV1Error(
            "resume companion event ledger stage differs"
        )
    persisted = ledger.get("persisted_keys")
    active = ledger.get("active_event_flags")
    inactive_explicit = ledger.get("inactive_explicit")
    if (
        not isinstance(persisted, list)
        or not isinstance(active, list)
        or type(inactive_explicit) is not bool
        or any(type(item) is not str for item in persisted + active)
        or persisted != sorted(set(persisted))
        or active != sorted(set(active))
    ):
        raise FreshProducerLineageV1Error(
            "resume companion event ledger lists are not canonical"
        )
    actual_event_keys = sorted(
        key
        for key in cfg
        if key == "__resume_registry_manifest"
        or any(key.startswith(prefix) for prefix in LEGACY_EVENT_PREFIXES)
    )
    if persisted != actual_event_keys:
        raise FreshProducerLineageV1Error(
            "resume companion event ledger persisted-key set differs"
        )
    if (active and not persisted) or (not active and not inactive_explicit):
        raise FreshProducerLineageV1Error(
            "resume companion event-controller state is incomplete"
        )
    canonical_ledger = json.dumps(
        ledger,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if ledger_text != canonical_ledger:
        raise FreshProducerLineageV1Error(
            "resume companion event ledger is not canonical JSON"
        )
    polyak_scalar_keys = {"__pta_arm", "__pta_count", "__pta_start"}
    polyak_declared = any(key in cfg for key in polyak_scalar_keys)
    if polyak_declared:
        if not polyak_scalar_keys.issubset(cfg):
            raise FreshProducerLineageV1Error(
                "resume companion Polyak mean lacks its scalar controller state"
            )
        polyak_count = int(_scalar(cfg, "__pta_count"))
        if int(_scalar(cfg, "__pta_arm")) != 1 or polyak_count < 0:
            raise FreshProducerLineageV1Error(
                "resume companion Polyak scalar controller state is invalid"
            )
        if polyak_count == 0 and polyak:
            raise FreshProducerLineageV1Error(
                "resume companion carries Polyak heavy state at count zero"
            )
        if polyak_count > 0 and not polyak:
            raise FreshProducerLineageV1Error(
                "resume companion declares active Polyak state without its "
                "heavy tensor state"
            )
        if polyak and set(polyak) != set(live):
            raise FreshProducerLineageV1Error(
                "resume companion Polyak tensor key set differs from model state"
            )
        for key, value in polyak.items():
            if np.asarray(value).dtype != np.dtype(np.float64):
                raise FreshProducerLineageV1Error(
                    f"resume companion Polyak tensor {key} is not float64"
                )
    elif polyak:
        raise FreshProducerLineageV1Error(
            "resume companion carries Polyak heavy state without an active "
            "scalar controller declaration"
        )
    if semantic_schema == RESUME_SEMANTIC_SCHEMA:
        _validate_complete_trajectory_manifest_skeleton(
            arrays=arrays,
            seed_active=seed_active,
            polyak_active=polyak_declared,
        )
    return (
        int(_scalar(cfg, "__resume_epoch")),
        stage,
        complete_state_manifest_proven,
    )


def _matching_namespace(
    deploy: Mapping[str, Any],
    resume_cfg: Mapping[str, Any],
    *,
    prefixes: tuple[str, ...],
    explicit: frozenset[str] = frozenset(),
    label: str,
) -> None:
    deploy_keys = {
        key
        for key in deploy
        if any(key.startswith(prefix) for prefix in prefixes) or key in explicit
    }
    resume_keys = {
        key
        for key in resume_cfg
        if any(key.startswith(prefix) for prefix in prefixes) or key in explicit
    }
    if deploy_keys != resume_keys:
        raise FreshProducerLineageV1Error(
            f"deploy/resume {label} key sets differ"
        )
    for key in deploy_keys:
        left = np.asarray(deploy[key])
        right = np.asarray(resume_cfg[key])
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise FreshProducerLineageV1Error(
                f"deploy/resume {label} differs at {key}"
            )


def open_fresh_producer_checkpoint_pair_v1(
    *,
    deploy_checkpoint: Path,
    expected_deploy_sha256: str,
    resume_checkpoint: Path,
    expected_resume_sha256: str,
    expected_current_launch_dsl_compile_hash: str,
) -> FreshProducerCheckpointPairV1:
    """Open and validate one mandatory deploy/full-state companion pair."""

    expected_launch = require_sha256(
        expected_current_launch_dsl_compile_hash,
        label="expected current launch DSL compile hash",
    )
    deploy = open_physical_npz(
        deploy_checkpoint,
        expected_sha256=expected_deploy_sha256,
        label="deploy checkpoint",
    )
    resume = open_physical_npz(
        resume_checkpoint,
        expected_sha256=expected_resume_sha256,
        label="full-state resume companion",
    )
    live, ema, optimizer, seed_state, seed_optimizer, polyak, cfg = (
        split_resume_state(resume.arrays)
    )
    (
        resume_epoch,
        resume_stage,
        complete_state_manifest_proven,
    ) = _validate_full_state_semantics(
        arrays=resume.arrays,
        live=live,
        ema=ema,
        optimizer=optimizer,
        seed=seed_state,
        seed_optimizer=seed_optimizer,
        polyak=polyak,
        cfg=cfg,
    )
    for key in FRESH_DEPLOY_KEYS:
        if key not in deploy.arrays:
            raise FreshProducerLineageV1Error(
                f"deploy checkpoint lacks fresh lineage field {key}"
            )
    for key in FRESH_RESUME_KEYS:
        if key not in cfg:
            raise FreshProducerLineageV1Error(
                f"resume companion lacks fresh lineage field {key}"
            )
    if (
        np.asarray(deploy.arrays["__cfg_fresh_producer"]).dtype
        != np.dtype(np.int8)
        or int(_scalar(deploy.arrays, "__cfg_fresh_producer")) != 1
        or np.asarray(cfg["__cfg_fresh_producer"]).dtype != np.dtype(np.int8)
        or int(_scalar(cfg, "__cfg_fresh_producer")) != 1
    ):
        raise FreshProducerLineageV1Error(
            "deploy/resume fresh-producer marker is not exact int8 one"
        )
    for key in FRESH_DEPLOY_KEYS:
        left = np.asarray(deploy.arrays[key])
        right = np.asarray(cfg[key])
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise FreshProducerLineageV1Error(
                f"deploy/resume fresh root contract differs at {key}"
            )
    if _scalar(cfg, "__cfg_fresh_lineage_schema") != FRESH_PRODUCER_LINEAGE_SCHEMA:
        raise FreshProducerLineageV1Error(
            "fresh producer lineage schema is wrong"
        )
    seed = _require_int64_scalar(
        cfg,
        "__cfg_fresh_seed",
        minimum=0,
    )
    root_sha = require_sha256(
        _scalar(cfg, "__cfg_fresh_lineage_root_sha256"),
        label="fresh lineage root SHA-256",
    )
    initial_state_sha = require_sha256(
        _scalar(cfg, "__cfg_fresh_initial_state_sha256"),
        label="fresh initial-state SHA-256",
    )
    root_dsl = require_sha256(
        _scalar(cfg, "__cfg_fresh_dsl_compile_hash"),
        label="fresh root DSL compile hash",
    )
    target_projection_sha = require_sha256(
        _scalar(cfg, "__cfg_fresh_target_projection_sha256"),
        label="fresh target projection SHA-256",
    )
    launch_dsl = require_sha256(
        _scalar(cfg, "__cfg_fresh_current_launch_dsl_compile_hash"),
        label="fresh current-launch DSL compile hash",
    )
    if launch_dsl != expected_launch:
        raise FreshProducerLineageV1Error(
            "checkpoint current-launch DSL compile hash differs from caller custody"
        )
    if target_projection_sha != require_sha256(
        _scalar(cfg, G109_TARGET_PROJECTION_SHA_KEY),
        label="physical G109 target projection SHA-256",
    ):
        raise FreshProducerLineageV1Error(
            "fresh lineage root is not bound to the physical G109 projection"
        )
    recomputed_root = fresh_producer_root_sha256(
        seed=seed,
        dsl_compile_hash=root_dsl,
        target_projection_sha256=target_projection_sha,
        initial_state_sha256=initial_state_sha,
    )
    if root_sha != recomputed_root:
        raise FreshProducerLineageV1Error(
            "fresh lineage root does not recompute from physical root fields"
        )
    state_sha = require_sha256(
        _scalar(cfg, "__cfg_fresh_lineage_state_sha256"),
        label="fresh lineage state SHA-256",
    )
    recomputed_state = fresh_resume_semantic_state_sha256(
        live_state=live,
        ema_state=ema,
        optimizer_state=optimizer,
        seed_state=seed_state,
        seed_optimizer_state=seed_optimizer,
        polyak_state=polyak,
        config_state=cfg,
    )
    if state_sha != recomputed_state:
        raise FreshProducerLineageV1Error(
            "fresh lineage state SHA-256 does not recompute from full state"
        )
    parent_id = require_sha256(
        _scalar(cfg, "__cfg_fresh_lineage_parent_checkpoint_id_sha256"),
        label="fresh lineage parent checkpoint id",
    )
    checkpoint_id = require_sha256(
        _scalar(cfg, "__cfg_fresh_lineage_checkpoint_id_sha256"),
        label="fresh lineage checkpoint id",
    )
    lineage_epoch = _require_int64_scalar(
        cfg,
        "__cfg_fresh_lineage_epoch",
        minimum=0,
    )
    lineage_stage = _scalar(cfg, "__cfg_fresh_lineage_stage")
    if type(lineage_stage) is not str:
        raise FreshProducerLineageV1Error(
            "fresh lineage stage must be scalar text"
        )
    if (
        lineage_epoch != resume_epoch
        or lineage_stage != resume_stage
        or int(_scalar(deploy.arrays, "__epoch")) != lineage_epoch
    ):
        raise FreshProducerLineageV1Error(
            "deploy/resume/lineage epoch or stage contract differs"
        )
    if parent_id == ROOT_PARENT_CHECKPOINT_ID and (
        lineage_epoch != 0
        or lineage_stage != "stageColdRoot"
        or initial_state_sha != recomputed_state
    ):
        raise FreshProducerLineageV1Error(
            "zero-parent fresh node must be the exact epoch-0 stageColdRoot "
            "semantic state bound by fresh initial_state_sha256"
        )
    recomputed_checkpoint_id = fresh_checkpoint_id_sha256(
        root_sha256=root_sha,
        parent_checkpoint_id_sha256=parent_id,
        state_sha256=state_sha,
        epoch=lineage_epoch,
        stage=lineage_stage,
    )
    if checkpoint_id != recomputed_checkpoint_id:
        raise FreshProducerLineageV1Error(
            "fresh lineage checkpoint id does not recompute"
        )
    film_stiefel = int(_scalar(cfg, "__cfg_film_stiefel"))
    if film_stiefel != 0:
        raise FreshProducerLineageV1Error(
            "G112 requires film-stiefel OFF so deploy equals companion EMA"
        )
    deploy_params = {
        key: value
        for key, value in deploy.arrays.items()
        if not key.startswith("__")
    }
    if set(deploy_params) != set(ema):
        raise FreshProducerLineageV1Error(
            "deploy learned tensor key set differs from companion EMA"
        )
    for key, deploy_value in deploy_params.items():
        ema_value = np.asarray(ema[key])
        if (
            np.asarray(deploy_value).dtype != ema_value.dtype
            or not np.array_equal(deploy_value, ema_value)
        ):
            raise FreshProducerLineageV1Error(
                f"deploy learned tensor {key} differs from companion EMA"
            )
    _matching_namespace(
        deploy.arrays,
        cfg,
        prefixes=("__cfg_pose_carrier",),
        label="pose contract",
    )
    _matching_namespace(
        deploy.arrays,
        cfg,
        prefixes=("__cfg_g109_", "__cfg_g46_"),
        explicit=frozenset(
            {
                "__cfg_target_authority_sha256",
                "__cfg_verdict_batch",
            }
        ),
        label="G109 target contract",
    )
    return FreshProducerCheckpointPairV1(
        deploy=deploy,
        resume=resume,
        seed=seed,
        root_sha256=root_sha,
        initial_state_sha256=initial_state_sha,
        root_dsl_compile_hash=root_dsl,
        current_launch_dsl_compile_hash=launch_dsl,
        target_projection_sha256=target_projection_sha,
        parent_checkpoint_id_sha256=parent_id,
        state_sha256=state_sha,
        checkpoint_id_sha256=checkpoint_id,
        epoch=lineage_epoch,
        stage=lineage_stage,
        live_tensor_count=len(live),
        ema_tensor_count=len(ema),
        optimizer_tensor_count=len(optimizer),
        seed_tensor_count=len(seed_state),
        seed_optimizer_tensor_count=len(seed_optimizer),
        polyak_tensor_count=len(polyak),
        config_array_count=len(cfg),
        complete_state_manifest_proven=complete_state_manifest_proven,
    )


_NODE_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "sequence_index",
        "root_sha256",
        "checkpoint_id_sha256",
        "parent_checkpoint_id_sha256",
        "epoch",
        "stage",
        "root_fields",
        "deploy",
        "resume",
        "parent_receipt",
        "receipt_sha256",
    }
)
_NODE_ROOT_FIELD_KEYS: Final = frozenset(
    {
        "lineage_schema",
        "seed",
        "root_sha256",
        "root_dsl_compile_hash",
        "target_projection_sha256",
        "initial_state_sha256",
        "current_launch_dsl_compile_hash",
    }
)
_PHYSICAL_IDENTITY_KEYS: Final = frozenset({"path", "bytes", "sha256"})


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FreshProducerLineageV1Error(
            "physical-node receipt is not canonical ASCII JSON"
        ) from exc


def _seal_node_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in body:
        raise FreshProducerLineageV1Error(
            "unsealed physical-node body contains receipt_sha256"
        )
    sealed = dict(body)
    sealed["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(body)
    ).hexdigest()
    return sealed


def _open_physical_receipt(
    receipt_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    path = Path(receipt_path).expanduser()
    if not path.is_absolute():
        raise FreshProducerLineageV1Error(
            "physical-node receipt path must be absolute"
        )
    try:
        if path.resolve(strict=True) != path:
            raise FreshProducerLineageV1Error(
                "physical-node receipt path must be canonical"
            )
    except OSError as exc:
        raise FreshProducerLineageV1Error(
            "physical-node receipt path cannot be resolved"
        ) from exc
    expected = require_sha256(
        expected_sha256,
        label="physical-node receipt SHA-256",
    )
    try:
        before = path.lstat()
    except OSError as exc:
        raise FreshProducerLineageV1Error(
            "physical-node receipt is not readable"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FreshProducerLineageV1Error(
            "physical-node receipt must be a regular non-symlink file"
        )
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise FreshProducerLineageV1Error(
            "physical-node receipt SHA-256 differs"
        )
    try:
        parsed = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FreshProducerLineageV1Error(
            "physical-node receipt is not ASCII JSON"
        ) from exc
    if (
        not isinstance(parsed, dict)
        or set(parsed) != _NODE_RECEIPT_KEYS
        or _canonical_json_bytes(parsed) + b"\n" != payload
    ):
        raise FreshProducerLineageV1Error(
            "physical-node receipt key set or canonical encoding differs"
        )
    body = dict(parsed)
    sealed_sha = body.pop("receipt_sha256")
    if sealed_sha != hashlib.sha256(_canonical_json_bytes(body)).hexdigest():
        raise FreshProducerLineageV1Error(
            "physical-node receipt self-hash differs"
        )
    after = path.lstat()
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
    ):
        raise FreshProducerLineageV1Error(
            "physical-node receipt changed during recursive reopen"
        )
    return parsed, payload


def _require_identity_dict(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _PHYSICAL_IDENTITY_KEYS:
        raise FreshProducerLineageV1Error(
            f"{label} physical identity key set differs"
        )
    path = value.get("path")
    size = value.get("bytes")
    sha = value.get("sha256")
    if type(path) is not str or not Path(path).is_absolute():
        raise FreshProducerLineageV1Error(
            f"{label} physical path must be absolute"
        )
    if type(size) is not int or size <= 0:
        raise FreshProducerLineageV1Error(
            f"{label} physical byte count must be positive"
        )
    require_sha256(sha, label=f"{label} physical SHA-256")
    return dict(value)


def _open_physical_node_once(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
) -> tuple[
    FreshProducerPhysicalCheckpointNodeV1,
    dict[str, object] | None,
]:
    parsed, receipt_payload = _open_physical_receipt(
        receipt_path,
        expected_sha256=expected_receipt_sha256,
    )
    if parsed["schema"] != FRESH_PHYSICAL_CHECKPOINT_NODE_SCHEMA:
        raise FreshProducerLineageV1Error(
            "physical-node receipt schema is wrong"
        )
    root_fields = parsed["root_fields"]
    if not isinstance(root_fields, dict) or set(root_fields) != _NODE_ROOT_FIELD_KEYS:
        raise FreshProducerLineageV1Error(
            "physical-node root-field key set differs"
        )
    deploy_identity = _require_identity_dict(
        parsed["deploy"],
        label="physical-node deploy",
    )
    resume_identity = _require_identity_dict(
        parsed["resume"],
        label="physical-node resume",
    )
    pair = open_fresh_producer_checkpoint_pair_v1(
        deploy_checkpoint=Path(str(deploy_identity["path"])),
        expected_deploy_sha256=str(deploy_identity["sha256"]),
        resume_checkpoint=Path(str(resume_identity["path"])),
        expected_resume_sha256=str(resume_identity["sha256"]),
        expected_current_launch_dsl_compile_hash=str(
            root_fields.get("current_launch_dsl_compile_hash", "")
        ),
    )
    if (
        int(deploy_identity["bytes"]) != pair.deploy.bytes
        or int(resume_identity["bytes"]) != pair.resume.bytes
    ):
        raise FreshProducerLineageV1Error(
            "physical-node deploy/resume byte identity differs"
        )
    expected_root_fields = {
        "lineage_schema": FRESH_PRODUCER_LINEAGE_SCHEMA,
        "seed": pair.seed,
        "root_sha256": pair.root_sha256,
        "root_dsl_compile_hash": pair.root_dsl_compile_hash,
        "target_projection_sha256": pair.target_projection_sha256,
        "initial_state_sha256": pair.initial_state_sha256,
        "current_launch_dsl_compile_hash": (
            pair.current_launch_dsl_compile_hash
        ),
    }
    if root_fields != expected_root_fields:
        raise FreshProducerLineageV1Error(
            "physical-node root fields differ from reopened checkpoint pair"
        )
    metadata = {
        "root_sha256": pair.root_sha256,
        "checkpoint_id_sha256": pair.checkpoint_id_sha256,
        "parent_checkpoint_id_sha256": (
            pair.parent_checkpoint_id_sha256
        ),
        "epoch": pair.epoch,
        "stage": pair.stage,
    }
    for key, expected in metadata.items():
        if parsed[key] != expected:
            raise FreshProducerLineageV1Error(
                f"physical-node receipt differs at {key}"
            )
    sequence_index = parsed["sequence_index"]
    if type(sequence_index) is not int or sequence_index < 0:
        raise FreshProducerLineageV1Error(
            "physical-node sequence_index must be a nonnegative integer"
        )
    receipt = Path(receipt_path).expanduser()
    expected_filename = f"{pair.checkpoint_id_sha256}.receipt.json"
    if receipt.name != expected_filename:
        raise FreshProducerLineageV1Error(
            "physical-node receipt filename is not content addressed"
        )
    if pair.deploy.path.name != f"{pair.checkpoint_id_sha256}.deploy.npz":
        raise FreshProducerLineageV1Error(
            "physical-node deploy filename is not content addressed"
        )
    if pair.resume.path.name != f"{pair.checkpoint_id_sha256}.resume.npz":
        raise FreshProducerLineageV1Error(
            "physical-node resume filename is not content addressed"
        )
    if (
        receipt.parent != pair.deploy.path.parent
        or receipt.parent != pair.resume.path.parent
        or receipt.parent.name != "fresh_lineage"
    ):
        raise FreshProducerLineageV1Error(
            "physical-node files do not share one fresh_lineage directory"
        )
    parent_value = parsed["parent_receipt"]
    if parent_value is None:
        parent_identity = None
    else:
        parent_identity = _require_identity_dict(
            parent_value,
            label="physical-node parent receipt",
        )
    return (
        FreshProducerPhysicalCheckpointNodeV1(
            receipt_path=receipt,
            receipt_sha256=expected_receipt_sha256,
            receipt_bytes=len(receipt_payload),
            sequence_index=sequence_index,
            pair=pair,
            parent_receipt_path=(
                None
                if parent_identity is None
                else Path(str(parent_identity["path"]))
            ),
            parent_receipt_sha256=(
                None
                if parent_identity is None
                else str(parent_identity["sha256"])
            ),
        ),
        parent_identity,
    )


def open_fresh_physical_checkpoint_chain_v1(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    expected_current_launch_dsl_compile_hash: str,
) -> FreshProducerPhysicalCheckpointChainV1:
    """Recursively reopen an immutable current-to-root physical ancestry."""

    expected_launch = require_sha256(
        expected_current_launch_dsl_compile_hash,
        label="expected current launch DSL compile hash",
    )
    cursor_path = Path(receipt_path).expanduser()
    cursor_sha = require_sha256(
        expected_receipt_sha256,
        label="current physical-node receipt SHA-256",
    )
    newest_first: list[FreshProducerPhysicalCheckpointNodeV1] = []
    visited_receipts: set[tuple[str, str]] = set()
    expected_child_parent: dict[str, object] | None = None
    for _depth in range(10000):
        visit_key = (str(cursor_path), cursor_sha)
        if visit_key in visited_receipts:
            raise FreshProducerLineageV1Error(
                "physical checkpoint ancestry contains a receipt cycle"
            )
        visited_receipts.add(visit_key)
        node, parent_identity = _open_physical_node_once(
            cursor_path,
            expected_receipt_sha256=cursor_sha,
        )
        if expected_child_parent is not None:
            observed_identity = {
                "path": str(node.receipt_path),
                "bytes": node.receipt_bytes,
                "sha256": node.receipt_sha256,
            }
            if observed_identity != expected_child_parent:
                raise FreshProducerLineageV1Error(
                    "child parent-receipt identity differs from physical parent"
                )
        newest_first.append(node)
        if parent_identity is None:
            break
        cursor_path = Path(str(parent_identity["path"]))
        cursor_sha = str(parent_identity["sha256"])
        expected_child_parent = parent_identity
    else:
        raise FreshProducerLineageV1Error(
            "physical checkpoint ancestry exceeds 10000 nodes"
        )
    nodes = tuple(reversed(newest_first))
    if not nodes:
        raise AssertionError("physical checkpoint ancestry cannot be empty")
    if nodes[-1].pair.current_launch_dsl_compile_hash != expected_launch:
        raise FreshProducerLineageV1Error(
            "current physical node launch DSL differs from caller custody"
        )
    root_sha = nodes[0].pair.root_sha256
    previous: FreshProducerPhysicalCheckpointNodeV1 | None = None
    for index, node in enumerate(nodes):
        if node.sequence_index != index:
            raise FreshProducerLineageV1Error(
                "physical checkpoint sequence index is not contiguous"
            )
        if node.pair.root_sha256 != root_sha:
            raise FreshProducerLineageV1Error(
                "physical checkpoint chain changes cold root"
            )
        if previous is None:
            if (
                node.pair.parent_checkpoint_id_sha256
                != ROOT_PARENT_CHECKPOINT_ID
                or node.parent_receipt_path is not None
                or node.parent_receipt_sha256 is not None
            ):
                raise FreshProducerLineageV1Error(
                    "physical checkpoint root is not the zero-parent node"
                )
        else:
            if (
                node.pair.parent_checkpoint_id_sha256
                != previous.pair.checkpoint_id_sha256
                or node.parent_receipt_path != previous.receipt_path
                or node.parent_receipt_sha256 != previous.receipt_sha256
            ):
                raise FreshProducerLineageV1Error(
                    "physical checkpoint parent link differs"
                )
            if node.pair.epoch < previous.pair.epoch:
                raise FreshProducerLineageV1Error(
                    "physical checkpoint epochs decrease across ancestry"
                )
        previous = node
    return FreshProducerPhysicalCheckpointChainV1(
        nodes=nodes,
        current=nodes[-1],
        root_sha256=root_sha,
        current_launch_dsl_compile_hash=expected_launch,
        complete_trajectory_proven=all(
            node.pair.complete_state_manifest_proven for node in nodes
        ),
    )


def _write_immutable_or_verify(
    path: Path,
    payload: bytes,
    *,
    scratch: Path,
) -> None:
    expected = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        try:
            info = path.lstat()
        except OSError as exc:
            raise FreshProducerLineageV1Error(
                f"immutable physical-node output {path.name} is unreadable"
            ) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise FreshProducerLineageV1Error(
                f"immutable physical-node output {path.name} differs"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".partial",
        dir=scratch,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise FreshProducerLineageV1Error(
                    f"concurrent immutable output {path.name} differs"
                ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_fresh_physical_checkpoint_node_v1(
    *,
    out_dir: Path,
    deploy_checkpoint: Path,
    expected_deploy_sha256: str,
    resume_checkpoint: Path,
    expected_resume_sha256: str,
    expected_current_launch_dsl_compile_hash: str,
    parent_receipt_path: Path | None = None,
    expected_parent_receipt_sha256: str | None = None,
) -> FreshProducerPhysicalCheckpointNodeV1:
    """Publish one immutable node after physically proving its complete parent chain."""

    output = Path(out_dir).expanduser()
    if not output.is_absolute():
        raise FreshProducerLineageV1Error(
            "physical checkpoint node output directory must be absolute"
        )
    try:
        if output.resolve(strict=True) != output:
            raise FreshProducerLineageV1Error(
                "physical checkpoint node output directory must be canonical"
            )
    except OSError as exc:
        raise FreshProducerLineageV1Error(
            "physical checkpoint node output directory cannot be resolved"
        ) from exc
    try:
        output_info = output.lstat()
    except OSError as exc:
        raise FreshProducerLineageV1Error(
            "physical checkpoint node output directory must already exist"
        ) from exc
    if stat.S_ISLNK(output_info.st_mode) or not stat.S_ISDIR(output_info.st_mode):
        raise FreshProducerLineageV1Error(
            "physical checkpoint node output directory must be a real directory"
        )
    pair = open_fresh_producer_checkpoint_pair_v1(
        deploy_checkpoint=deploy_checkpoint,
        expected_deploy_sha256=expected_deploy_sha256,
        resume_checkpoint=resume_checkpoint,
        expected_resume_sha256=expected_resume_sha256,
        expected_current_launch_dsl_compile_hash=(
            expected_current_launch_dsl_compile_hash
        ),
    )
    parent_node: FreshProducerPhysicalCheckpointNodeV1 | None = None
    if pair.parent_checkpoint_id_sha256 == ROOT_PARENT_CHECKPOINT_ID:
        if (
            parent_receipt_path is not None
            or expected_parent_receipt_sha256 is not None
        ):
            raise FreshProducerLineageV1Error(
                "zero-parent node must not supply a parent receipt"
            )
        sequence_index = 0
    else:
        if (
            parent_receipt_path is None
            or expected_parent_receipt_sha256 is None
        ):
            raise FreshProducerLineageV1Error(
                "nonroot node requires a physical parent receipt and SHA-256"
            )
        parent_receipt, _parent_payload = _open_physical_receipt(
            parent_receipt_path,
            expected_sha256=expected_parent_receipt_sha256,
        )
        parent_root_fields = parent_receipt.get("root_fields")
        if not isinstance(parent_root_fields, dict):
            raise FreshProducerLineageV1Error(
                "physical parent receipt root fields have the wrong type"
            )
        parent_launch = require_sha256(
            parent_root_fields.get("current_launch_dsl_compile_hash"),
            label="physical parent current-launch DSL compile hash",
        )
        parent_chain = open_fresh_physical_checkpoint_chain_v1(
            parent_receipt_path,
            expected_receipt_sha256=expected_parent_receipt_sha256,
            # The parent receipt SHA is chained by this node; its own launch
            # hash is an authenticated field, not the current launch's
            # external identity.
            expected_current_launch_dsl_compile_hash=parent_launch,
        )
        parent_node = parent_chain.current
        if (
            pair.parent_checkpoint_id_sha256
            != parent_node.pair.checkpoint_id_sha256
            or pair.root_sha256 != parent_chain.root_sha256
            or pair.epoch < parent_node.pair.epoch
        ):
            raise FreshProducerLineageV1Error(
                "new physical node does not continue its reopened parent"
            )
        sequence_index = parent_node.sequence_index + 1
    lineage_dir = output / "fresh_lineage"
    if lineage_dir.exists() or lineage_dir.is_symlink():
        lineage_info = lineage_dir.lstat()
        if (
            stat.S_ISLNK(lineage_info.st_mode)
            or not stat.S_ISDIR(lineage_info.st_mode)
        ):
            raise FreshProducerLineageV1Error(
                "fresh_lineage must be a real directory"
            )
    else:
        lineage_dir.mkdir(mode=0o700)
        output_fd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
    checkpoint_id = pair.checkpoint_id_sha256
    deploy_target = lineage_dir / f"{checkpoint_id}.deploy.npz"
    resume_target = lineage_dir / f"{checkpoint_id}.resume.npz"
    receipt_target = lineage_dir / f"{checkpoint_id}.receipt.json"
    missing_payload_bytes = sum(
        size
        for path, size in (
            (deploy_target, pair.deploy.bytes),
            (resume_target, pair.resume.bytes),
        )
        if not path.exists()
    )
    required_free_bytes = missing_payload_bytes + (1 << 20)
    observed_free_bytes = int(shutil.disk_usage(lineage_dir).free)
    if observed_free_bytes < required_free_bytes:
        raise FreshProducerLineageV1Error(
            "physical checkpoint node storage preflight failed "
            f"(required={required_free_bytes}, free={observed_free_bytes})"
        )
    scratch = Path(
        tempfile.mkdtemp(
            prefix=".node-scratch.",
            dir=lineage_dir,
        )
    )
    try:
        deploy_payload = pair.deploy.path.read_bytes()
        resume_payload = pair.resume.path.read_bytes()
        if (
            hashlib.sha256(deploy_payload).hexdigest() != pair.deploy.sha256
            or hashlib.sha256(resume_payload).hexdigest() != pair.resume.sha256
        ):
            raise FreshProducerLineageV1Error(
                "source checkpoint pair changed before immutable publication"
            )
        _write_immutable_or_verify(
            deploy_target,
            deploy_payload,
            scratch=scratch,
        )
        _write_immutable_or_verify(
            resume_target,
            resume_payload,
            scratch=scratch,
        )
        deploy_identity = {
            "path": str(deploy_target),
            "bytes": len(deploy_payload),
            "sha256": pair.deploy.sha256,
        }
        resume_identity = {
            "path": str(resume_target),
            "bytes": len(resume_payload),
            "sha256": pair.resume.sha256,
        }
        parent_identity = (
            None
            if parent_node is None
            else {
                "path": str(parent_node.receipt_path),
                "bytes": parent_node.receipt_bytes,
                "sha256": parent_node.receipt_sha256,
            }
        )
        body = {
            "schema": FRESH_PHYSICAL_CHECKPOINT_NODE_SCHEMA,
            "sequence_index": sequence_index,
            "root_sha256": pair.root_sha256,
            "checkpoint_id_sha256": checkpoint_id,
            "parent_checkpoint_id_sha256": (
                pair.parent_checkpoint_id_sha256
            ),
            "epoch": pair.epoch,
            "stage": pair.stage,
            "root_fields": {
                "lineage_schema": FRESH_PRODUCER_LINEAGE_SCHEMA,
                "seed": pair.seed,
                "root_sha256": pair.root_sha256,
                "root_dsl_compile_hash": pair.root_dsl_compile_hash,
                "target_projection_sha256": (
                    pair.target_projection_sha256
                ),
                "initial_state_sha256": pair.initial_state_sha256,
                "current_launch_dsl_compile_hash": (
                    pair.current_launch_dsl_compile_hash
                ),
            },
            "deploy": deploy_identity,
            "resume": resume_identity,
            "parent_receipt": parent_identity,
        }
        receipt_payload = (
            _canonical_json_bytes(_seal_node_receipt(body)) + b"\n"
        )
        _write_immutable_or_verify(
            receipt_target,
            receipt_payload,
            scratch=scratch,
        )
    finally:
        try:
            scratch.rmdir()
        except OSError as exc:
            raise FreshProducerLineageV1Error(
                "physical-node scratch was not empty after publication"
            ) from exc
    chain = open_fresh_physical_checkpoint_chain_v1(
        receipt_target,
        expected_receipt_sha256=hashlib.sha256(
            receipt_payload
        ).hexdigest(),
        expected_current_launch_dsl_compile_hash=(
            expected_current_launch_dsl_compile_hash
        ),
    )
    if (
        chain.current.pair.deploy.sha256 != pair.deploy.sha256
        or chain.current.pair.resume.sha256 != pair.resume.sha256
        or chain.current.pair.checkpoint_id_sha256 != checkpoint_id
    ):
        raise FreshProducerLineageV1Error(
            "published physical checkpoint node differs after recursive reopen"
        )
    return replace(
        chain.current,
        complete_trajectory_proven=chain.complete_trajectory_proven,
    )


__all__ = [
    "FRESH_DEPLOY_KEYS",
    "FRESH_LINEAGE_DERIVED_CFG_KEYS",
    "FRESH_PHYSICAL_CHECKPOINT_NODE_SCHEMA",
    "FRESH_PRODUCER_LINEAGE_SCHEMA",
    "FRESH_RESUME_KEYS",
    "FRESH_ROOT_KEYS",
    "G111_COMPLETE_TRAJECTORY_KEY",
    "G111_COMPLETE_TRAJECTORY_SCHEMA",
    "G111_TRAJECTORY_COMPONENTS",
    "G109_TARGET_PROJECTION_SHA_KEY",
    "LEGACY_RESUME_SEMANTIC_SCHEMA",
    "LEGACY_EVENT_PREFIXES",
    "RESUME_EMA_PREFIX",
    "RESUME_EVENT_LEDGER_SCHEMA",
    "RESUME_LIVE_PREFIX",
    "RESUME_OPT_PREFIX",
    "RESUME_POLYAK_PREFIX",
    "RESUME_RNG_KEYS",
    "RESUME_SEMANTIC_SCHEMA",
    "ROOT_PARENT_CHECKPOINT_ID",
    "FreshProducerCheckpointPairV1",
    "FreshProducerLineageV1Error",
    "FreshProducerPhysicalCheckpointChainV1",
    "FreshProducerPhysicalCheckpointNodeV1",
    "PhysicalNpzV1",
    "canonical_lineage_sha256",
    "fresh_checkpoint_id_sha256",
    "fresh_producer_root_sha256",
    "fresh_resume_semantic_state_sha256",
    "fresh_resume_semantic_state_sha256_from_flat",
    "open_fresh_physical_checkpoint_chain_v1",
    "open_fresh_producer_checkpoint_pair_v1",
    "open_physical_npz",
    "require_sha256",
    "sha256_array_mapping",
    "split_resume_state",
    "write_fresh_physical_checkpoint_node_v1",
]
