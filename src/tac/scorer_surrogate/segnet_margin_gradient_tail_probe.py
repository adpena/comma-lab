# SPDX-License-Identifier: MIT
"""Resume-safe contract for the D24a frozen-SegNet gradient-tail probe.

This module deliberately does not import or launch the scorer.  It defines the
content-closed plan, raw observation rows, append/replay behavior, and terminal
receipt that a later scorer-bearing runner must use.  Historical tail numbers
are not encoded as expected values: the measurement must be reproduced from
the bound source, scorer, and cache.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "segnet_margin_gradient_tail_probe_plan.v1"
ROW_SCHEMA = "segnet_margin_gradient_tail_observation.v1"
RECEIPT_SCHEMA = "segnet_margin_gradient_tail_receipt.v1"
PAIR_COUNT = 600
SEGNET_BATCH_SIZE = 32
RADII_PX: tuple[int, ...] = (64, 128, 192)
QUERY_KINDS: tuple[str, ...] = ("high_margin_control", "minimum_margin")
BLOCK_RELATIONS: tuple[str, ...] = ("same_edge", "adjacent_edge", "remote_edge")
NON_PROMOTABLE_AXIS = "macOS-CPU advisory / frozen-SegNet mechanism measurement"
VERDICT_SCOPE = "INSTANCE x N600 x FROZEN_SEGNET x LOCAL_JACOBIAN_FORMULATION"
_SHA256_HEX_LEN = 64


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != _SHA256_HEX_LEN or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field} must be lowercase SHA-256 hex")
    return digest


def _finite_nonnegative(value: float, *, field: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return out


def _path_digest(path: Path) -> tuple[str, int, str]:
    """Hash one file or a directory tree without copying its bytes."""

    resolved = path.resolve(strict=True)
    if resolved.is_file():
        hasher = hashlib.sha256()
        size = 0
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
                size += len(chunk)
        return hasher.hexdigest(), size, "file"
    if not resolved.is_dir():
        raise ValueError(f"artifact is neither a regular file nor directory: {resolved}")
    hasher = hashlib.sha256()
    size = 0
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = child.relative_to(resolved).as_posix().encode("utf-8")
        child_digest, child_size, _ = _path_digest(child)
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(bytes.fromhex(child_digest))
        hasher.update(child_size.to_bytes(8, "big"))
        size += child_size
    return hasher.hexdigest(), size, "tree"


@dataclass(frozen=True)
class ArtifactBinding:
    role: str
    path: str
    sha256: str
    bytes: int
    kind: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("artifact role must be non-empty")
        if not self.path.strip():
            raise ValueError("artifact path must be non-empty")
        _require_sha256(self.sha256, field=f"{self.role}.sha256")
        if int(self.bytes) < 0:
            raise ValueError(f"{self.role}.bytes must be non-negative")
        if self.kind not in {"file", "tree"}:
            raise ValueError(f"{self.role}.kind must be file or tree")

    @classmethod
    def from_path(cls, *, role: str, path: str | Path) -> ArtifactBinding:
        resolved = Path(path).resolve(strict=True)
        digest, size, kind = _path_digest(resolved)
        return cls(role=role, path=str(resolved), sha256=digest, bytes=size, kind=kind)

    def verify_live(self) -> None:
        digest, size, kind = _path_digest(Path(self.path))
        if (digest, size, kind) != (self.sha256, self.bytes, self.kind):
            raise ValueError(
                f"{self.role} custody mismatch: expected "
                f"{(self.sha256, self.bytes, self.kind)}, observed {(digest, size, kind)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes": int(self.bytes),
            "kind": self.kind,
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ArtifactBinding:
        return cls(
            role=str(raw["role"]),
            path=str(raw["path"]),
            sha256=str(raw["sha256"]),
            bytes=int(raw["bytes"]),
            kind=str(raw["kind"]),
        )


@dataclass(frozen=True)
class MarginGradientTailProbePlan:
    scorer: ArtifactBinding
    source: ArtifactBinding
    cache: ArtifactBinding
    pair_count: int = PAIR_COUNT
    scorer_batch_size: int = SEGNET_BATCH_SIZE
    radii_px: tuple[int, ...] = RADII_PX
    query_kinds: tuple[str, ...] = QUERY_KINDS
    block_relations: tuple[str, ...] = BLOCK_RELATIONS
    seed: int = 20260715
    schema: str = PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PLAN_SCHEMA:
            raise ValueError(f"plan schema must be {PLAN_SCHEMA}")
        if self.scorer.role != "frozen_segnet_scorer":
            raise ValueError("scorer binding role must be frozen_segnet_scorer")
        if self.source.role != "n600_source":
            raise ValueError("source binding role must be n600_source")
        if self.cache.role != "scorer_cache":
            raise ValueError("cache binding role must be scorer_cache")
        if int(self.pair_count) != PAIR_COUNT:
            raise ValueError("D24a is exactly an n600 probe")
        if int(self.scorer_batch_size) != SEGNET_BATCH_SIZE:
            raise ValueError("D24a requires frozen-SegNet batch_size=32 custody")
        if tuple(self.radii_px) != RADII_PX:
            raise ValueError(f"D24a radii must be exactly {RADII_PX}")
        if tuple(self.query_kinds) != QUERY_KINDS:
            raise ValueError(f"D24a query kinds must be exactly {QUERY_KINDS}")
        if tuple(self.block_relations) != BLOCK_RELATIONS:
            raise ValueError(f"D24a block relations must be exactly {BLOCK_RELATIONS}")

    @property
    def plan_id(self) -> str:
        return _canonical_sha256(self.to_dict(include_plan_id=False))

    @property
    def expected_row_count(self) -> int:
        return PAIR_COUNT * len(QUERY_KINDS)

    def verify_live_custody(self) -> None:
        self.scorer.verify_live()
        self.source.verify_live()
        self.cache.verify_live()

    def to_dict(self, *, include_plan_id: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "authority": {
                "axis": NON_PROMOTABLE_AXIS,
                "promotion_eligible": False,
                "score_claim": False,
                "verdict_scope": VERDICT_SCOPE,
            },
            "block_relations": list(self.block_relations),
            "cache": self.cache.to_dict(),
            "pair_count": int(self.pair_count),
            "query_kinds": list(self.query_kinds),
            "radii_px": list(self.radii_px),
            "schema": self.schema,
            "scorer": self.scorer.to_dict(),
            "scorer_batch_size": int(self.scorer_batch_size),
            "seed": int(self.seed),
            "source": self.source.to_dict(),
        }
        if include_plan_id:
            out["plan_id"] = self.plan_id
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MarginGradientTailProbePlan:
        plan = cls(
            scorer=ArtifactBinding.from_dict(raw["scorer"]),
            source=ArtifactBinding.from_dict(raw["source"]),
            cache=ArtifactBinding.from_dict(raw["cache"]),
            pair_count=int(raw["pair_count"]),
            scorer_batch_size=int(raw["scorer_batch_size"]),
            radii_px=tuple(int(value) for value in raw["radii_px"]),
            query_kinds=tuple(str(value) for value in raw["query_kinds"]),
            block_relations=tuple(str(value) for value in raw["block_relations"]),
            seed=int(raw["seed"]),
            schema=str(raw["schema"]),
        )
        if raw.get("plan_id") != plan.plan_id:
            raise ValueError("plan_id does not match the canonical plan payload")
        authority = raw.get("authority")
        if authority != plan.to_dict()["authority"]:
            raise ValueError("plan authority fields differ from the D24a contract")
        return plan


@dataclass(frozen=True)
class MarginGradientTailObservation:
    plan_id: str
    pair_index: int
    scored_frame_index: int
    query_kind: str
    query_y: int
    query_x: int
    top_class: int
    rival_class: int
    margin: float
    total_gradient_energy: float
    nonzero_input_fraction: float
    tail_energy_fraction: tuple[tuple[int, float], ...]
    block_jacobian_energy: tuple[tuple[str, float], ...]
    source_frame_sha256: str
    cache_record_sha256: str
    schema: str = ROW_SCHEMA

    def __post_init__(self) -> None:
        _require_sha256(self.plan_id, field="plan_id")
        if not 0 <= int(self.pair_index) < PAIR_COUNT:
            raise ValueError("pair_index must be in [0, 600)")
        if int(self.scored_frame_index) != 2 * int(self.pair_index) + 1:
            raise ValueError("scored_frame_index must be the pair's SegNet frame1")
        if self.query_kind not in QUERY_KINDS:
            raise ValueError(f"query_kind must be one of {QUERY_KINDS}")
        if int(self.query_y) < 0 or int(self.query_x) < 0:
            raise ValueError("query coordinates must be non-negative")
        if int(self.top_class) < 0 or int(self.rival_class) < 0:
            raise ValueError("class indices must be non-negative")
        if int(self.top_class) == int(self.rival_class):
            raise ValueError("top_class and rival_class must differ")
        _finite_nonnegative(self.margin, field="margin")
        if _finite_nonnegative(self.total_gradient_energy, field="total_gradient_energy") <= 0:
            raise ValueError("total_gradient_energy must be positive")
        nonzero = _finite_nonnegative(self.nonzero_input_fraction, field="nonzero_input_fraction")
        if nonzero > 1.0:
            raise ValueError("nonzero_input_fraction must not exceed one")
        tails = tuple((int(radius), float(value)) for radius, value in self.tail_energy_fraction)
        if tuple(radius for radius, _ in tails) != RADII_PX:
            raise ValueError(f"tail radii must be exactly {RADII_PX}")
        tail_values = tuple(
            _finite_nonnegative(value, field=f"tail_energy_fraction[{radius}]")
            for radius, value in tails
        )
        if any(value > 1.0 for value in tail_values):
            raise ValueError("tail energy fractions must not exceed one")
        if any(left < right for left, right in pairwise(tail_values)):
            raise ValueError("tail energy fractions must be non-increasing with radius")
        blocks = tuple((str(relation), float(value)) for relation, value in self.block_jacobian_energy)
        if tuple(relation for relation, _ in blocks) != BLOCK_RELATIONS:
            raise ValueError(f"block relations must be exactly {BLOCK_RELATIONS}")
        for relation, value in blocks:
            _finite_nonnegative(value, field=f"block_jacobian_energy[{relation}]")
        _require_sha256(self.source_frame_sha256, field="source_frame_sha256")
        _require_sha256(self.cache_record_sha256, field="cache_record_sha256")
        if self.schema != ROW_SCHEMA:
            raise ValueError(f"row schema must be {ROW_SCHEMA}")

    @property
    def row_key(self) -> str:
        return f"{int(self.pair_index):03d}:{self.query_kind}"

    @property
    def row_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_row_sha256=False))

    def to_dict(self, *, include_row_sha256: bool = True) -> dict[str, Any]:
        out = {
            "block_jacobian_energy": [list(item) for item in self.block_jacobian_energy],
            "cache_record_sha256": self.cache_record_sha256,
            "margin": float(self.margin),
            "nonzero_input_fraction": float(self.nonzero_input_fraction),
            "pair_index": int(self.pair_index),
            "plan_id": self.plan_id,
            "query_kind": self.query_kind,
            "query_x": int(self.query_x),
            "query_y": int(self.query_y),
            "rival_class": int(self.rival_class),
            "row_key": self.row_key,
            "schema": self.schema,
            "scored_frame_index": int(self.scored_frame_index),
            "source_frame_sha256": self.source_frame_sha256,
            "tail_energy_fraction": [list(item) for item in self.tail_energy_fraction],
            "top_class": int(self.top_class),
            "total_gradient_energy": float(self.total_gradient_energy),
        }
        if include_row_sha256:
            out["row_sha256"] = self.row_sha256
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MarginGradientTailObservation:
        row = cls(
            plan_id=str(raw["plan_id"]),
            pair_index=int(raw["pair_index"]),
            scored_frame_index=int(raw["scored_frame_index"]),
            query_kind=str(raw["query_kind"]),
            query_y=int(raw["query_y"]),
            query_x=int(raw["query_x"]),
            top_class=int(raw["top_class"]),
            rival_class=int(raw["rival_class"]),
            margin=float(raw["margin"]),
            total_gradient_energy=float(raw["total_gradient_energy"]),
            nonzero_input_fraction=float(raw["nonzero_input_fraction"]),
            tail_energy_fraction=tuple(
                (int(radius), float(value)) for radius, value in raw["tail_energy_fraction"]
            ),
            block_jacobian_energy=tuple(
                (str(relation), float(value))
                for relation, value in raw["block_jacobian_energy"]
            ),
            source_frame_sha256=str(raw["source_frame_sha256"]),
            cache_record_sha256=str(raw["cache_record_sha256"]),
            schema=str(raw["schema"]),
        )
        if raw.get("row_key") != row.row_key or raw.get("row_sha256") != row.row_sha256:
            raise ValueError("row identity does not match the canonical observation payload")
        return row


def write_plan(path: str | Path, plan: MarginGradientTailProbePlan) -> None:
    """Atomically persist a plan only after rechecking all live artifact bytes."""

    plan.verify_live_custody()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    payload = _canonical_json(plan.to_dict()) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_plan(path: str | Path) -> MarginGradientTailProbePlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("plan JSON must contain one object")
    return MarginGradientTailProbePlan.from_dict(raw)


def load_observations(path: str | Path) -> tuple[MarginGradientTailObservation, ...]:
    source = Path(path)
    if not source.exists():
        return ()
    rows: list[MarginGradientTailObservation] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"observation line {line_number} is not an object")
        row = MarginGradientTailObservation.from_dict(raw)
        if row.row_key in seen:
            raise ValueError(f"duplicate observation row_key {row.row_key!r}")
        seen.add(row.row_key)
        rows.append(row)
    return tuple(rows)


def append_observation(
    path: str | Path,
    *,
    plan: MarginGradientTailProbePlan,
    row: MarginGradientTailObservation,
) -> bool:
    """Append one row durably; replaying identical work is an idempotent no-op."""

    plan.verify_live_custody()
    if row.plan_id != plan.plan_id:
        raise ValueError("observation plan_id differs from the live plan")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = {item.row_key: item for item in load_observations(destination)}
        prior = existing.get(row.row_key)
        if prior is not None:
            if prior != row:
                raise ValueError(f"immutable observation conflict for {row.row_key}")
            return False
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(row.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return True


def next_probe_tasks(
    plan: MarginGradientTailProbePlan,
    observations: Iterable[MarginGradientTailObservation],
) -> tuple[tuple[int, str], ...]:
    present: set[tuple[int, str]] = set()
    for row in observations:
        if row.plan_id != plan.plan_id:
            raise ValueError("observation plan_id differs from the live plan")
        key = (row.pair_index, row.query_kind)
        if key in present:
            raise ValueError(f"duplicate observation task {key}")
        present.add(key)
    return tuple(
        (pair_index, query_kind)
        for pair_index in range(PAIR_COUNT)
        for query_kind in QUERY_KINDS
        if (pair_index, query_kind) not in present
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty measurement sequence")
    return math.fsum(float(value) for value in values) / len(values)


def build_terminal_receipt(
    plan: MarginGradientTailProbePlan,
    observations: Sequence[MarginGradientTailObservation],
) -> dict[str, Any]:
    """Close a receipt only for the exact complete n600 x two-query matrix."""

    plan.verify_live_custody()
    missing = next_probe_tasks(plan, observations)
    if missing:
        raise ValueError(
            f"D24a terminal receipt requires {plan.expected_row_count} rows; "
            f"{len(missing)} tasks remain"
        )
    ordered = tuple(sorted(observations, key=lambda row: row.row_key))
    raw_rows_sha256 = hashlib.sha256(
        "".join(_canonical_json(row.to_dict()) + "\n" for row in ordered).encode("utf-8")
    ).hexdigest()
    tail_means = {
        str(radius): _mean(
            [dict(row.tail_energy_fraction)[radius] for row in ordered]
        )
        for radius in RADII_PX
    }
    block_means = {
        relation: _mean(
            [dict(row.block_jacobian_energy)[relation] for row in ordered]
        )
        for relation in BLOCK_RELATIONS
    }
    return {
        "authority": {
            "axis": NON_PROMOTABLE_AXIS,
            "promotion_eligible": False,
            "score_claim": False,
            "verdict_scope": VERDICT_SCOPE,
        },
        "block_jacobian_energy_mean": block_means,
        "completion": {
            "pair_count": PAIR_COUNT,
            "query_count_per_pair": len(QUERY_KINDS),
            "row_count": len(ordered),
        },
        "factorization_verdict": "NO_VERDICT_THRESHOLD_NOT_PREREGISTERED",
        "measurement_labels": {
            "block_jacobian_energy_mean": "MEASURED",
            "tail_energy_fraction_mean": "MEASURED",
        },
        "plan": plan.to_dict(),
        "raw_rows_sha256": raw_rows_sha256,
        "schema": RECEIPT_SCHEMA,
        "tail_energy_fraction_mean": tail_means,
    }


__all__ = [
    "BLOCK_RELATIONS",
    "NON_PROMOTABLE_AXIS",
    "PAIR_COUNT",
    "PLAN_SCHEMA",
    "QUERY_KINDS",
    "RADII_PX",
    "RECEIPT_SCHEMA",
    "ROW_SCHEMA",
    "SEGNET_BATCH_SIZE",
    "VERDICT_SCOPE",
    "ArtifactBinding",
    "MarginGradientTailObservation",
    "MarginGradientTailProbePlan",
    "append_observation",
    "build_terminal_receipt",
    "load_observations",
    "load_plan",
    "next_probe_tasks",
    "write_plan",
]
