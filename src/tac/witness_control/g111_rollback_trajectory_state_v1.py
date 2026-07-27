# SPDX-License-Identifier: MIT
"""Fixed-capacity O2 rollback trajectory state for Fresh-G111.

The live trainer's rollback actuator is trajectory state, not telemetry.  A
crash-faithful continuation needs both the guard/controller scalars and the
optional last-good savepoint that the actuator would restore.  This module
provides a pure NumPy boundary for that state.  It does not import MLX, mutate a
trainer, checkpoint a file, or authorize a launch.

The expected savepoint topology is constructed independently from the freshly
created O1 model/EMA/optimizer/auxiliary trees.  Cold state and savepoint-live
state serialize to exactly the same keys, dtypes, and shapes.  When no
savepoint is present, every savepoint leaf is canonical zero padding and the
savepoint coordinates use their single absent representation.

``exhausted_warned`` is intentionally absent.  It only throttles a log row and
cannot affect a decision, optimizer update, or rollback.  Persisting it as O2
would turn observation state into false trajectory authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

SCHEMA: Final = "tac.g111_rollback_trajectory_state.v1"
SERIALIZED_SCHEMA: Final = "tac.g111_rollback_trajectory_state_arrays.v1"
DEFAULT_PREFIX: Final = "__g111_rollback__"
NO_SEED_SUPPORT_SHA256: Final = hashlib.sha256(b"tac.g111.rollback-trajectory.no-seed-support.v1").hexdigest()
OBSERVATION_ONLY_EXCLUDED_FIELDS: Final[tuple[str, ...]] = ("exhausted_warned",)

TREE_PREFIXES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "live": "rbLiveP__",
        "ema": "rbEmaP__",
        "opt": "rbOptP__",
        "film_polar": "rbFilmPolarP__",
        "seed": "rbSeedP__",
        "seed_opt": "rbSeedOptP__",
    }
)
TREE_NAMES: Final[tuple[str, ...]] = tuple(TREE_PREFIXES)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODE_CODES: Final = MappingProxyType({"legacy": 0, "rollback": 1})
_CODE_MODES: Final = MappingProxyType({value: key for key, value in _MODE_CODES.items()})


class G111RollbackTrajectoryStateError(RuntimeError):
    """The O2 rollback trajectory state is incomplete or malformed."""


def _fail(message: str) -> None:
    raise G111RollbackTrajectoryStateError(message)


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{name} must be an exact non-empty canonical string")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _canonical_string(value, name=name)
    if not _SHA256_RE.fullmatch(result):
        _fail(f"{name} must be a lowercase SHA-256 hex string")
    return result


def _exact_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _fail(f"{name} must be an exact integer")
    result = int(value)
    if minimum is not None and result < minimum:
        _fail(f"{name} must be >= {minimum}")
    return result


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    if minimum is not None and result < minimum:
        _fail(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        _fail(f"{name} must be <= {maximum}")
    return result


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _fail(f"{name} must be an exact bool")
    return value


def _canonical_dtype(value: object, *, name: str) -> np.dtype[Any]:
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise G111RollbackTrajectoryStateError(f"{name} has an invalid dtype") from exc
    if dtype.hasobject or dtype.fields is not None:
        _fail(f"{name} uses an object or structured dtype; pickle-bearing state is forbidden")
    if dtype.kind not in "biufc":
        _fail(f"{name} has unsupported dtype {dtype}")
    return dtype


def _canonical_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    _canonical_dtype(array.dtype, name=name)
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        _fail(f"{name} contains non-finite values")
    original_shape = array.shape
    contiguous = np.ascontiguousarray(array)
    immutable = np.frombuffer(bytes(contiguous.tobytes(order="C")), dtype=contiguous.dtype).reshape(original_shape)
    return immutable


def _utf8(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_utf8(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        _fail(f"{name} must be a one-dimensional uint8 array")
    try:
        return array.tobytes(order="C").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise G111RollbackTrajectoryStateError(f"{name} is not valid UTF-8") from exc


def _scalar(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any] | type[np.generic],
) -> object:
    array = np.asarray(value)
    expected = np.dtype(dtype)
    if array.dtype != expected or array.shape != ():
        _fail(f"{name} must be a {expected} scalar")
    return array.item()


def _tree_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    for key in value:
        _canonical_string(key, name=f"{name} key")
    return value


@dataclass(frozen=True, slots=True, order=True)
class RollbackTreeLeafSpecV1:
    """One independently expected savepoint leaf."""

    key: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        key = _canonical_string(self.key, name="savepoint leaf key")
        dtype = _canonical_dtype(self.dtype, name=f"savepoint leaf {key!r}").str
        shape = tuple(_exact_int(value, name=f"{key!r} shape dimension", minimum=0) for value in self.shape)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)

    def as_dict(self) -> dict[str, object]:
        return {"key": self.key, "dtype": self.dtype, "shape": list(self.shape)}


def _tree_specs(value: Mapping[str, Any], *, name: str) -> tuple[RollbackTreeLeafSpecV1, ...]:
    source = _tree_mapping(value, name=name)
    specs: list[RollbackTreeLeafSpecV1] = []
    for key in sorted(source):
        array = np.asarray(source[key])
        specs.append(
            RollbackTreeLeafSpecV1(
                key=key,
                dtype=_canonical_dtype(array.dtype, name=f"{name}.{key}").str,
                shape=tuple(int(dimension) for dimension in array.shape),
            )
        )
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class RollbackSavepointTopologyV1:
    """Fresh O1 topology and deterministic seed-support binding."""

    live: tuple[RollbackTreeLeafSpecV1, ...]
    ema: tuple[RollbackTreeLeafSpecV1, ...]
    opt: tuple[RollbackTreeLeafSpecV1, ...]
    film_polar: tuple[RollbackTreeLeafSpecV1, ...]
    seed: tuple[RollbackTreeLeafSpecV1, ...]
    seed_opt: tuple[RollbackTreeLeafSpecV1, ...]
    seed_support_geometry_sha256: str

    def __post_init__(self) -> None:
        support = _sha256(
            self.seed_support_geometry_sha256,
            name="seed_support_geometry_sha256",
        )
        for tree_name in TREE_NAMES:
            specs = tuple(getattr(self, tree_name))
            if tuple(sorted(specs)) != specs or len({row.key for row in specs}) != len(specs):
                _fail(f"{tree_name} topology must be uniquely and canonically sorted")
            object.__setattr__(self, tree_name, specs)
        if not self.live or not self.ema or not self.opt:
            _fail("live, EMA, and optimizer savepoint topologies must all be non-empty")
        live_shape = tuple((row.key, row.dtype, row.shape) for row in self.live)
        ema_shape = tuple((row.key, row.dtype, row.shape) for row in self.ema)
        if live_shape != ema_shape:
            _fail("live and EMA topologies must match exactly")
        seed_present = bool(self.seed or self.seed_opt)
        if seed_present and (not self.seed or not self.seed_opt):
            _fail("seed live and optimizer topologies must be present together")
        if seed_present and support == NO_SEED_SUPPORT_SHA256:
            _fail("present seed topology cannot use the no-seed support identity")
        if not seed_present and support != NO_SEED_SUPPORT_SHA256:
            _fail("absent seed topology must use the canonical no-seed support identity")
        object.__setattr__(self, "seed_support_geometry_sha256", support)

    def tree(self, name: str) -> tuple[RollbackTreeLeafSpecV1, ...]:
        if name not in TREE_NAMES:
            _fail(f"unknown savepoint tree {name!r}")
        return tuple(getattr(self, name))

    def canonical_dict(self) -> dict[str, object]:
        return {
            "schema": "tac.g111_rollback_savepoint_topology.v1",
            "trees": {name: [row.as_dict() for row in self.tree(name)] for name in TREE_NAMES},
            "seed_support_geometry_sha256": self.seed_support_geometry_sha256,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def build_rollback_savepoint_topology(
    *,
    live: Mapping[str, Any],
    ema: Mapping[str, Any],
    opt: Mapping[str, Any],
    film_polar: Mapping[str, Any] | None = None,
    seed: Mapping[str, Any] | None = None,
    seed_opt: Mapping[str, Any] | None = None,
    seed_support_geometry_sha256: str | None = None,
) -> RollbackSavepointTopologyV1:
    """Build expected O2 savepoint topology from a fresh O1 runtime."""

    seed_map = {} if seed is None else seed
    seed_opt_map = {} if seed_opt is None else seed_opt
    seed_present = bool(seed_map or seed_opt_map)
    if seed_present and seed_support_geometry_sha256 is None:
        _fail("active seed topology requires its exact O1 support-geometry SHA-256")
    if not seed_present and seed_support_geometry_sha256 is not None:
        _fail("inactive seed topology must not declare a support-geometry SHA-256")
    return RollbackSavepointTopologyV1(
        live=_tree_specs(live, name="live"),
        ema=_tree_specs(ema, name="ema"),
        opt=_tree_specs(opt, name="opt"),
        film_polar=_tree_specs({} if film_polar is None else film_polar, name="film_polar"),
        seed=_tree_specs(seed_map, name="seed"),
        seed_opt=_tree_specs(seed_opt_map, name="seed_opt"),
        seed_support_geometry_sha256=(
            _sha256(
                seed_support_geometry_sha256,
                name="seed_support_geometry_sha256",
            )
            if seed_present
            else NO_SEED_SUPPORT_SHA256
        ),
    )


@dataclass(frozen=True, slots=True)
class G111RollbackTrajectoryConfigV1:
    """Typed guard constants plus exact fresh-O1 identity."""

    typed_config_sha256: str
    o1_topology_sha256: str
    seed_support_geometry_sha256: str
    mode: str
    window: int
    frac: float
    lr_cut: float
    max_rollbacks: int
    recent_losses_capacity: int = 50

    def validate(self) -> None:
        _sha256(self.typed_config_sha256, name="typed_config_sha256")
        _sha256(self.o1_topology_sha256, name="o1_topology_sha256")
        _sha256(
            self.seed_support_geometry_sha256,
            name="seed_support_geometry_sha256",
        )
        if self.mode not in _MODE_CODES:
            _fail(f"mode must be one of {sorted(_MODE_CODES)}")
        _exact_int(self.window, name="window", minimum=1)
        _finite_float(self.frac, name="frac", minimum=0.0, maximum=1.0)
        if float(self.frac) <= 0.0:
            _fail("frac must be > 0")
        _finite_float(self.lr_cut, name="lr_cut", minimum=0.0, maximum=1.0)
        if not 0.0 < float(self.lr_cut) < 1.0:
            _fail("lr_cut must lie strictly inside (0, 1)")
        _exact_int(self.max_rollbacks, name="max_rollbacks", minimum=1)
        _exact_int(
            self.recent_losses_capacity,
            name="recent_losses_capacity",
            minimum=1,
        )

    def validate_topology(self, topology: RollbackSavepointTopologyV1) -> None:
        self.validate()
        if not isinstance(topology, RollbackSavepointTopologyV1):
            raise TypeError("topology must be RollbackSavepointTopologyV1")
        if self.o1_topology_sha256 != topology.sha256:
            _fail("typed rollback config differs from fresh O1 topology")
        if self.seed_support_geometry_sha256 != topology.seed_support_geometry_sha256:
            _fail("typed rollback config differs from fresh O1 seed support")


def config_for_topology(
    *,
    typed_config_sha256: str,
    topology: RollbackSavepointTopologyV1,
    mode: str,
    window: int,
    frac: float,
    lr_cut: float,
    max_rollbacks: int,
    recent_losses_capacity: int = 50,
) -> G111RollbackTrajectoryConfigV1:
    """Bind exact DSL config and fresh O1 topology in one immutable contract."""

    config = G111RollbackTrajectoryConfigV1(
        typed_config_sha256=typed_config_sha256,
        o1_topology_sha256=topology.sha256,
        seed_support_geometry_sha256=topology.seed_support_geometry_sha256,
        mode=mode,
        window=window,
        frac=frac,
        lr_cut=lr_cut,
        max_rollbacks=max_rollbacks,
        recent_losses_capacity=recent_losses_capacity,
    )
    config.validate_topology(topology)
    return config


@dataclass(frozen=True, slots=True)
class RollbackSavepointV1:
    """Optional complete last-good state and its trajectory coordinates."""

    present: bool
    live: Mapping[str, np.ndarray]
    ema: Mapping[str, np.ndarray]
    opt: Mapping[str, np.ndarray]
    film_polar: Mapping[str, np.ndarray]
    seed: Mapping[str, np.ndarray]
    seed_opt: Mapping[str, np.ndarray]
    snap_epoch: int
    completed_optimizer_steps: int

    def tree(self, name: str) -> Mapping[str, np.ndarray]:
        if name not in TREE_NAMES:
            _fail(f"unknown savepoint tree {name!r}")
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class G111RollbackTrajectoryStateV1:
    """Decision-complete O2 state; observation-only fields are excluded."""

    config: G111RollbackTrajectoryConfigV1
    rollbacks: int
    events: tuple[bool, ...]
    lr_scale: float
    ep_spikes: int
    ep_batches: int
    recent_losses: tuple[float, ...]
    savepoint: RollbackSavepointV1


def _capture_tree(
    value: Mapping[str, Any],
    *,
    specs: Sequence[RollbackTreeLeafSpecV1],
    name: str,
) -> Mapping[str, np.ndarray]:
    source = _tree_mapping(value, name=name)
    expected = {row.key: row for row in specs}
    if set(source) != set(expected):
        _fail(
            f"{name} savepoint topology differs; "
            f"missing={sorted(set(expected) - set(source))}, "
            f"unknown={sorted(set(source) - set(expected))}"
        )
    captured: dict[str, np.ndarray] = {}
    for key in sorted(expected):
        array = _canonical_array(source[key], name=f"{name}.{key}")
        spec = expected[key]
        if array.dtype.str != spec.dtype or tuple(array.shape) != spec.shape:
            _fail(
                f"{name}.{key} dtype/shape differs from fresh O1 "
                f"({array.dtype.str},{array.shape}) != ({spec.dtype},{spec.shape})"
            )
        captured[key] = array
    return MappingProxyType(captured)


def _absent_savepoint() -> RollbackSavepointV1:
    empty = MappingProxyType({})
    return RollbackSavepointV1(
        present=False,
        live=empty,
        ema=empty,
        opt=empty,
        film_polar=empty,
        seed=empty,
        seed_opt=empty,
        snap_epoch=-1,
        completed_optimizer_steps=0,
    )


def capture_rollback_trajectory_state(
    *,
    config: G111RollbackTrajectoryConfigV1,
    topology: RollbackSavepointTopologyV1,
    rollbacks: int,
    events: Sequence[bool],
    lr_scale: float,
    ep_spikes: int,
    ep_batches: int,
    recent_losses: Sequence[float],
    savepoint: Mapping[str, Mapping[str, Any]] | None = None,
    snap_epoch: int | None = None,
    completed_optimizer_steps: int | None = None,
) -> G111RollbackTrajectoryStateV1:
    """Capture the real guard and optional complete last-good savepoint."""

    config.validate_topology(topology)
    if isinstance(events, (str, bytes, bytearray)):
        _fail("events must be a boolean sequence")
    normalized_events = tuple(_exact_bool(value, name=f"events[{index}]") for index, value in enumerate(events))
    normalized_losses = tuple(
        _finite_float(value, name=f"recent_losses[{index}]") for index, value in enumerate(recent_losses)
    )
    if len(normalized_events) > config.window:
        _fail("rollback event window exceeds typed fixed capacity")
    if len(normalized_losses) > config.recent_losses_capacity:
        _fail("recent-loss window exceeds typed fixed capacity")
    normalized_rollbacks = _exact_int(rollbacks, name="rollbacks", minimum=0)
    if normalized_rollbacks > config.max_rollbacks:
        _fail("rollbacks exceeds typed rollback budget")
    normalized_ep_spikes = _exact_int(ep_spikes, name="ep_spikes", minimum=0)
    normalized_ep_batches = _exact_int(ep_batches, name="ep_batches", minimum=0)
    if normalized_ep_spikes > normalized_ep_batches:
        _fail("ep_spikes cannot exceed ep_batches")
    normalized_lr_scale = _finite_float(lr_scale, name="lr_scale", minimum=0.0, maximum=1.0)
    if normalized_lr_scale <= 0.0:
        _fail("lr_scale must be > 0")

    if savepoint is None:
        if snap_epoch is not None or completed_optimizer_steps is not None:
            _fail("absent savepoint cannot carry hidden trajectory coordinates")
        captured_savepoint = _absent_savepoint()
    else:
        trees = _tree_mapping(savepoint, name="savepoint")
        if set(trees) != set(TREE_NAMES):
            _fail("savepoint must provide all six topology trees, including explicit empty optional trees")
        if snap_epoch is None or completed_optimizer_steps is None:
            _fail("present savepoint requires snap_epoch and completed_optimizer_steps")
        captured = {
            name: _capture_tree(
                _tree_mapping(trees[name], name=f"savepoint.{name}"),
                specs=topology.tree(name),
                name=name,
            )
            for name in TREE_NAMES
        }
        captured_savepoint = RollbackSavepointV1(
            present=True,
            live=captured["live"],
            ema=captured["ema"],
            opt=captured["opt"],
            film_polar=captured["film_polar"],
            seed=captured["seed"],
            seed_opt=captured["seed_opt"],
            snap_epoch=_exact_int(snap_epoch, name="snap_epoch", minimum=0),
            completed_optimizer_steps=_exact_int(
                completed_optimizer_steps,
                name="completed_optimizer_steps",
                minimum=0,
            ),
        )

    state = G111RollbackTrajectoryStateV1(
        config=config,
        rollbacks=normalized_rollbacks,
        events=normalized_events,
        lr_scale=normalized_lr_scale,
        ep_spikes=normalized_ep_spikes,
        ep_batches=normalized_ep_batches,
        recent_losses=normalized_losses,
        savepoint=captured_savepoint,
    )
    validate_rollback_trajectory_state(state, topology=topology)
    return state


def validate_rollback_trajectory_state(
    state: G111RollbackTrajectoryStateV1,
    *,
    topology: RollbackSavepointTopologyV1,
) -> None:
    """Validate all decision state and exact savepoint coverage."""

    if not isinstance(state, G111RollbackTrajectoryStateV1):
        raise TypeError("state must be G111RollbackTrajectoryStateV1")
    state.config.validate_topology(topology)
    if len(state.events) > state.config.window:
        _fail("rollback event window exceeds typed fixed capacity")
    for index, event in enumerate(state.events):
        _exact_bool(event, name=f"events[{index}]")
    if len(state.recent_losses) > state.config.recent_losses_capacity:
        _fail("recent-loss window exceeds typed fixed capacity")
    for index, value in enumerate(state.recent_losses):
        _finite_float(value, name=f"recent_losses[{index}]")
    rollbacks = _exact_int(state.rollbacks, name="rollbacks", minimum=0)
    if rollbacks > state.config.max_rollbacks:
        _fail("rollbacks exceeds typed rollback budget")
    ep_spikes = _exact_int(state.ep_spikes, name="ep_spikes", minimum=0)
    ep_batches = _exact_int(state.ep_batches, name="ep_batches", minimum=0)
    if ep_spikes > ep_batches:
        _fail("ep_spikes cannot exceed ep_batches")
    lr_scale = _finite_float(state.lr_scale, name="lr_scale", minimum=0.0, maximum=1.0)
    if lr_scale <= 0.0:
        _fail("lr_scale must be > 0")
    savepoint = state.savepoint
    if not isinstance(savepoint, RollbackSavepointV1):
        _fail("savepoint must be RollbackSavepointV1")
    if savepoint.present:
        _exact_int(savepoint.snap_epoch, name="snap_epoch", minimum=0)
        _exact_int(
            savepoint.completed_optimizer_steps,
            name="completed_optimizer_steps",
            minimum=0,
        )
        for name in TREE_NAMES:
            _capture_tree(
                savepoint.tree(name),
                specs=topology.tree(name),
                name=name,
            )
    else:
        if savepoint.snap_epoch != -1 or savepoint.completed_optimizer_steps != 0:
            _fail("absent savepoint has hidden trajectory coordinates")
        if any(savepoint.tree(name) for name in TREE_NAMES):
            _fail("absent savepoint has hidden semantic leaves")
    if state.config.mode == "legacy" and (
        rollbacks != 0 or state.events or lr_scale != 1.0 or ep_spikes != 0 or ep_batches != 0 or savepoint.present
    ):
        _fail("legacy mode cannot carry rollback-only trajectory state")


def _tree_array_key(tree_name: str, leaf_key: str) -> str:
    return TREE_PREFIXES[tree_name] + leaf_key


def _semantic_digest(arrays: Mapping[str, np.ndarray], *, digest_key: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"tac.g111.rollback-trajectory-state-arrays.v1\0")
    for key in sorted(set(arrays) - {digest_key}):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def state_arrays(
    state: G111RollbackTrajectoryStateV1,
    *,
    topology: RollbackSavepointTopologyV1,
    prefix: str = DEFAULT_PREFIX,
) -> Mapping[str, np.ndarray]:
    """Serialize O2 as strict, fixed-shape, pickle-free arrays."""

    prefix = _canonical_string(prefix, name="prefix")
    validate_rollback_trajectory_state(state, topology=topology)
    config = state.config
    savepoint = state.savepoint
    events = np.zeros(config.window, dtype=np.int8)
    events[: len(state.events)] = np.asarray(state.events, dtype=np.int8)
    losses = np.zeros(config.recent_losses_capacity, dtype=np.float64)
    losses[: len(state.recent_losses)] = np.asarray(state.recent_losses, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        f"{prefix}schema": _utf8(SERIALIZED_SCHEMA),
        f"{prefix}typed_config_sha256": _utf8(config.typed_config_sha256),
        f"{prefix}o1_topology_sha256": _utf8(config.o1_topology_sha256),
        f"{prefix}seed_support_geometry_sha256": _utf8(config.seed_support_geometry_sha256),
        f"{prefix}mode_code": np.asarray(_MODE_CODES[config.mode], np.int8),
        f"{prefix}window": np.asarray(config.window, np.int64),
        f"{prefix}frac": np.asarray(config.frac, np.float64),
        f"{prefix}lr_cut": np.asarray(config.lr_cut, np.float64),
        f"{prefix}max_rollbacks": np.asarray(config.max_rollbacks, np.int64),
        f"{prefix}recent_losses_capacity": np.asarray(config.recent_losses_capacity, np.int64),
        f"{prefix}rollbacks": np.asarray(state.rollbacks, np.int64),
        f"{prefix}event_count": np.asarray(len(state.events), np.int64),
        f"{prefix}events": events,
        f"{prefix}lr_scale": np.asarray(state.lr_scale, np.float64),
        f"{prefix}ep_spikes": np.asarray(state.ep_spikes, np.int64),
        f"{prefix}ep_batches": np.asarray(state.ep_batches, np.int64),
        f"{prefix}recent_losses_count": np.asarray(len(state.recent_losses), np.int64),
        f"{prefix}recent_losses": losses,
        f"{prefix}savepoint_present": np.asarray(savepoint.present, np.int8),
        f"{prefix}snap_epoch": np.asarray(savepoint.snap_epoch, np.int64),
        f"{prefix}completed_optimizer_steps": np.asarray(savepoint.completed_optimizer_steps, np.int64),
    }
    for tree_name in TREE_NAMES:
        tree = savepoint.tree(tree_name)
        for spec in topology.tree(tree_name):
            value = (
                np.asarray(tree[spec.key]) if savepoint.present else np.zeros(spec.shape, dtype=np.dtype(spec.dtype))
            )
            arrays[_tree_array_key(tree_name, spec.key)] = value
    digest_key = f"{prefix}state_sha256"
    arrays[digest_key] = _utf8(_semantic_digest(arrays, digest_key=digest_key))
    return MappingProxyType(
        {key: _canonical_array(value, name=f"serialized state {key}") for key, value in arrays.items()}
    )


def _owned_keys(
    arrays: Mapping[str, Any],
    *,
    prefix: str,
) -> set[str]:
    return {
        key
        for key in arrays
        if key.startswith(prefix) or any(key.startswith(tree_prefix) for tree_prefix in TREE_PREFIXES.values())
    }


def state_from_arrays(
    arrays: Mapping[str, Any],
    *,
    expected_config: G111RollbackTrajectoryConfigV1,
    topology: RollbackSavepointTopologyV1,
    prefix: str = DEFAULT_PREFIX,
) -> G111RollbackTrajectoryStateV1:
    """Restore O2 and bind it to independently constructed config/topology."""

    prefix = _canonical_string(prefix, name="prefix")
    expected_config.validate_topology(topology)
    source = _tree_mapping(arrays, name="serialized rollback state")
    scalar_names = (
        "schema",
        "typed_config_sha256",
        "o1_topology_sha256",
        "seed_support_geometry_sha256",
        "mode_code",
        "window",
        "frac",
        "lr_cut",
        "max_rollbacks",
        "recent_losses_capacity",
        "rollbacks",
        "event_count",
        "events",
        "lr_scale",
        "ep_spikes",
        "ep_batches",
        "recent_losses_count",
        "recent_losses",
        "savepoint_present",
        "snap_epoch",
        "completed_optimizer_steps",
        "state_sha256",
    )
    expected_keys = {f"{prefix}{name}" for name in scalar_names}
    for tree_name in TREE_NAMES:
        expected_keys.update(_tree_array_key(tree_name, spec.key) for spec in topology.tree(tree_name))
    actual_keys = _owned_keys(source, prefix=prefix)
    if actual_keys != expected_keys:
        _fail(
            "serialized rollback state census differs; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unknown={sorted(actual_keys - expected_keys)}"
        )
    normalized = {key: _canonical_array(source[key], name=f"serialized state {key}") for key in expected_keys}
    digest_key = f"{prefix}state_sha256"
    stored_digest = _decode_utf8(normalized[digest_key], name="state_sha256")
    _sha256(stored_digest, name="state_sha256")
    if _semantic_digest(normalized, digest_key=digest_key) != stored_digest:
        _fail("serialized rollback state SHA-256 differs")
    if _decode_utf8(normalized[f"{prefix}schema"], name="schema") != SERIALIZED_SCHEMA:
        _fail("serialized rollback schema differs")
    if (
        _decode_utf8(
            normalized[f"{prefix}typed_config_sha256"],
            name="typed_config_sha256",
        )
        != expected_config.typed_config_sha256
    ):
        _fail("serialized rollback state typed config differs")
    if (
        _decode_utf8(
            normalized[f"{prefix}o1_topology_sha256"],
            name="o1_topology_sha256",
        )
        != topology.sha256
    ):
        _fail("serialized rollback state O1 topology differs")
    if (
        _decode_utf8(
            normalized[f"{prefix}seed_support_geometry_sha256"],
            name="seed_support_geometry_sha256",
        )
        != topology.seed_support_geometry_sha256
    ):
        _fail("serialized rollback state seed support differs")
    mode_code = _exact_int(
        _scalar(normalized[f"{prefix}mode_code"], name="mode_code", dtype=np.int8),
        name="mode_code",
        minimum=0,
    )
    if mode_code not in _CODE_MODES or _CODE_MODES[mode_code] != expected_config.mode:
        _fail("serialized rollback mode differs from typed config")
    exact_config_scalars = {
        "window": (np.int64, expected_config.window),
        "max_rollbacks": (np.int64, expected_config.max_rollbacks),
        "recent_losses_capacity": (
            np.int64,
            expected_config.recent_losses_capacity,
        ),
    }
    for name, (dtype, expected) in exact_config_scalars.items():
        value = _exact_int(
            _scalar(normalized[f"{prefix}{name}"], name=name, dtype=dtype),
            name=name,
        )
        if value != expected:
            _fail(f"serialized rollback {name} differs from typed config")
    for name, expected in (
        ("frac", expected_config.frac),
        ("lr_cut", expected_config.lr_cut),
    ):
        value = _finite_float(
            _scalar(
                normalized[f"{prefix}{name}"],
                name=name,
                dtype=np.float64,
            ),
            name=name,
        )
        if value != float(expected):
            _fail(f"serialized rollback {name} differs from typed config")

    event_count = _exact_int(
        _scalar(
            normalized[f"{prefix}event_count"],
            name="event_count",
            dtype=np.int64,
        ),
        name="event_count",
        minimum=0,
    )
    event_array = normalized[f"{prefix}events"]
    if event_array.dtype != np.dtype(np.int8) or event_array.shape != (expected_config.window,):
        _fail("serialized rollback events have wrong fixed dtype/shape")
    if event_count > expected_config.window:
        _fail("serialized rollback event count exceeds fixed capacity")
    if np.any((event_array[:event_count] != 0) & (event_array[:event_count] != 1)):
        _fail("serialized rollback events are not booleans")
    if np.any(event_array[event_count:]):
        _fail("serialized rollback events have nonzero padding")
    loss_count = _exact_int(
        _scalar(
            normalized[f"{prefix}recent_losses_count"],
            name="recent_losses_count",
            dtype=np.int64,
        ),
        name="recent_losses_count",
        minimum=0,
    )
    loss_array = normalized[f"{prefix}recent_losses"]
    if loss_array.dtype != np.dtype(np.float64) or loss_array.shape != (expected_config.recent_losses_capacity,):
        _fail("serialized recent losses have wrong fixed dtype/shape")
    if loss_count > expected_config.recent_losses_capacity:
        _fail("serialized recent-loss count exceeds fixed capacity")
    if not np.isfinite(loss_array[:loss_count]).all():
        _fail("serialized recent losses contain non-finite values")
    if np.any(loss_array[loss_count:]):
        _fail("serialized recent losses have nonzero padding")
    present_raw = _exact_int(
        _scalar(
            normalized[f"{prefix}savepoint_present"],
            name="savepoint_present",
            dtype=np.int8,
        ),
        name="savepoint_present",
        minimum=0,
    )
    if present_raw not in (0, 1):
        _fail("savepoint_present must be 0 or 1")
    present = bool(present_raw)
    snap_epoch = _exact_int(
        _scalar(
            normalized[f"{prefix}snap_epoch"],
            name="snap_epoch",
            dtype=np.int64,
        ),
        name="snap_epoch",
    )
    completed_steps = _exact_int(
        _scalar(
            normalized[f"{prefix}completed_optimizer_steps"],
            name="completed_optimizer_steps",
            dtype=np.int64,
        ),
        name="completed_optimizer_steps",
        minimum=0,
    )
    restored_trees: dict[str, Mapping[str, np.ndarray]] = {}
    hidden_nonzero: list[str] = []
    for tree_name in TREE_NAMES:
        values: dict[str, np.ndarray] = {}
        for spec in topology.tree(tree_name):
            key = _tree_array_key(tree_name, spec.key)
            value = normalized[key]
            if value.dtype.str != spec.dtype or tuple(value.shape) != spec.shape:
                _fail(f"serialized savepoint leaf {key!r} differs from fresh O1 topology")
            if not present and np.any(value != 0):
                hidden_nonzero.append(key)
            if present:
                values[spec.key] = value
        restored_trees[tree_name] = MappingProxyType(values)
    if not present:
        if snap_epoch != -1 or completed_steps != 0:
            _fail("absent savepoint has hidden trajectory coordinates")
        if hidden_nonzero:
            _fail(f"absent savepoint has nonzero hidden leaves: {hidden_nonzero}")
        savepoint = _absent_savepoint()
    else:
        if snap_epoch < 0:
            _fail("present savepoint requires nonnegative snap_epoch")
        savepoint = RollbackSavepointV1(
            present=True,
            live=restored_trees["live"],
            ema=restored_trees["ema"],
            opt=restored_trees["opt"],
            film_polar=restored_trees["film_polar"],
            seed=restored_trees["seed"],
            seed_opt=restored_trees["seed_opt"],
            snap_epoch=snap_epoch,
            completed_optimizer_steps=completed_steps,
        )
    state = G111RollbackTrajectoryStateV1(
        config=expected_config,
        rollbacks=_exact_int(
            _scalar(
                normalized[f"{prefix}rollbacks"],
                name="rollbacks",
                dtype=np.int64,
            ),
            name="rollbacks",
            minimum=0,
        ),
        events=tuple(bool(value) for value in event_array[:event_count]),
        lr_scale=_finite_float(
            _scalar(
                normalized[f"{prefix}lr_scale"],
                name="lr_scale",
                dtype=np.float64,
            ),
            name="lr_scale",
        ),
        ep_spikes=_exact_int(
            _scalar(
                normalized[f"{prefix}ep_spikes"],
                name="ep_spikes",
                dtype=np.int64,
            ),
            name="ep_spikes",
            minimum=0,
        ),
        ep_batches=_exact_int(
            _scalar(
                normalized[f"{prefix}ep_batches"],
                name="ep_batches",
                dtype=np.int64,
            ),
            name="ep_batches",
            minimum=0,
        ),
        recent_losses=tuple(float(value) for value in loss_array[:loss_count]),
        savepoint=savepoint,
    )
    validate_rollback_trajectory_state(state, topology=topology)
    return state


def cross_validate_rollback_against_o1(
    state: G111RollbackTrajectoryStateV1,
    *,
    expected_config: G111RollbackTrajectoryConfigV1,
    fresh_o1_topology: RollbackSavepointTopologyV1,
) -> None:
    """Fail closed unless state belongs to the exact fresh O1/config runtime."""

    expected_config.validate_topology(fresh_o1_topology)
    if state.config != expected_config:
        _fail("rollback state config differs from the expected typed config")
    validate_rollback_trajectory_state(state, topology=fresh_o1_topology)


serialize_rollback_trajectory_state = state_arrays
restore_rollback_trajectory_state = state_from_arrays


__all__ = [
    "DEFAULT_PREFIX",
    "NO_SEED_SUPPORT_SHA256",
    "OBSERVATION_ONLY_EXCLUDED_FIELDS",
    "SCHEMA",
    "SERIALIZED_SCHEMA",
    "TREE_NAMES",
    "TREE_PREFIXES",
    "G111RollbackTrajectoryConfigV1",
    "G111RollbackTrajectoryStateError",
    "G111RollbackTrajectoryStateV1",
    "RollbackSavepointTopologyV1",
    "RollbackSavepointV1",
    "RollbackTreeLeafSpecV1",
    "build_rollback_savepoint_topology",
    "capture_rollback_trajectory_state",
    "config_for_topology",
    "cross_validate_rollback_against_o1",
    "restore_rollback_trajectory_state",
    "serialize_rollback_trajectory_state",
    "state_arrays",
    "state_from_arrays",
    "validate_rollback_trajectory_state",
]
