# SPDX-License-Identifier: MIT
"""Contest-granularity atom primitives.

These dataclasses normalize the four grains that matter for score-oracle
optimization in this contest:

* archive/member bytes, including master-gradient byte offsets;
* decoded pixel regions;
* decoded frames;
* scorer frame-pairs.

The objects are intentionally planning/evidence rows.  They do not claim a
score, authorize dispatch, or mutate archives.  Their job is to let search,
waterfill, xray, master-gradient, and cathedral-autopilot code discuss the same
unit without collapsing pair, frame, pixel, and byte signals into one lossy
table too early.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

DEFAULT_FRAME_HEIGHT = 874
DEFAULT_FRAME_WIDTH = 1164
DEFAULT_FRAME_COUNT = 1200
CONTEST_RATE_DENOMINATOR_BYTES = 37_545_489


class ContestAtomError(ValueError):
    """Raised when a contest-grain atom or input row is malformed."""


class ContestScopeKind(StrEnum):
    """The four durable contest optimization grains."""

    BYTE = "byte"
    PIXEL_REGION = "pixel_region"
    FRAME = "frame"
    PAIR = "pair"


class ContestSignal(StrEnum):
    """Common evidence sources that may overlap on one optimization atom."""

    PAIR_COMPONENT = "pair_component"
    XRAY_PAIR = "xray_pair"
    XRAY_PIXEL = "xray_pixel"
    MASTER_GRADIENT = "master_gradient"
    SCORER_ORACLE = "scorer_oracle"
    SIDECAR_SELECTED = "sidecar_selected"


@dataclass(frozen=True, slots=True)
class ByteScope:
    """One contiguous byte interval in an archive/member domain."""

    archive_sha256: str
    byte_domain: str
    start: int
    length: int = 1
    member_name: str | None = None
    member_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ContestAtomError("ByteScope.start must be non-negative")
        if self.length <= 0:
            raise ContestAtomError("ByteScope.length must be positive")

    @property
    def stop(self) -> int:
        return self.start + self.length


@dataclass(frozen=True, slots=True)
class PixelRegionScope:
    """A decoded pixel rectangle for one frame."""

    frame_index: int
    x0: int = 0
    y0: int = 0
    width: int = DEFAULT_FRAME_WIDTH
    height: int = DEFAULT_FRAME_HEIGHT
    channel: int | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ContestAtomError("PixelRegionScope.frame_index must be non-negative")
        if self.x0 < 0 or self.y0 < 0:
            raise ContestAtomError("PixelRegionScope origin must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ContestAtomError("PixelRegionScope dimensions must be positive")
        if self.channel is not None and self.channel not in (0, 1, 2):
            raise ContestAtomError("PixelRegionScope.channel must be 0, 1, 2, or None")


@dataclass(frozen=True, slots=True)
class FrameScope:
    """A decoded frame, with optional parent pair."""

    frame_index: int
    pair_index: int | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ContestAtomError("FrameScope.frame_index must be non-negative")
        if self.pair_index is not None and self.pair_index < 0:
            raise ContestAtomError("FrameScope.pair_index must be non-negative")


@dataclass(frozen=True, slots=True)
class PairScope:
    """A scorer pair.  The contest scorer consumes two decoded frames per pair."""

    pair_index: int

    def __post_init__(self) -> None:
        if self.pair_index < 0:
            raise ContestAtomError("PairScope.pair_index must be non-negative")

    @property
    def frame_indices(self) -> tuple[int, int]:
        return (2 * self.pair_index, 2 * self.pair_index + 1)


@dataclass(frozen=True, slots=True)
class ScoreVector:
    """Score-term evidence attached to an atom.

    Values may be measured component masses, local deltas, or gradient masses;
    ``evidence_kind`` tells consumers how to interpret them.  The canonical
    score formula helper is still useful for pair/xray rows where absolute
    PoseNet and SegNet distortions are known.
    """

    evidence_kind: str
    pose_dist: float | None = None
    seg_dist: float | None = None
    rate_delta: float = 0.0
    score_delta: float | None = None
    score_mass: float | None = None
    pose_score_mass: float | None = None
    seg_score_mass: float | None = None
    rate_score_mass: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ContestAtomError(f"ScoreVector.{name} must be finite")

    @staticmethod
    def from_pair_distortions(
        *,
        pose_dist: float,
        seg_dist: float,
        evidence_kind: str = "pair_component_distortion",
    ) -> "ScoreVector":
        pose_term = math.sqrt(max(0.0, 10.0 * float(pose_dist)))
        seg_term = 100.0 * float(seg_dist)
        return ScoreVector(
            evidence_kind=evidence_kind,
            pose_dist=float(pose_dist),
            seg_dist=float(seg_dist),
            score_mass=pose_term + seg_term,
            pose_score_mass=pose_term,
            seg_score_mass=seg_term,
        )


@dataclass(frozen=True, slots=True)
class BudgetVector:
    """Budget/cost fields used by waterfill and scorer-oracle loops."""

    archive_delta_bytes: int = 0
    runtime_delta_ms: float = 0.0
    cpu_eval_seconds: float = 0.0
    gpu_cost_usd: float = 0.0

    @property
    def rate_score_delta(self) -> float:
        return 25.0 * float(self.archive_delta_bytes) / float(CONTEST_RATE_DENOMINATOR_BYTES)


@dataclass(frozen=True, slots=True)
class ContestAtom:
    """One normalized byte/pixel/frame/pair optimization atom."""

    atom_id: str
    scope_kind: ContestScopeKind
    scope: ByteScope | PixelRegionScope | FrameScope | PairScope
    score: ScoreVector
    budget: BudgetVector = field(default_factory=BudgetVector)
    source_signals: tuple[ContestSignal, ...] = ()
    evidence_axis: str = "[diagnostic]"
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def __post_init__(self) -> None:
        if not self.atom_id:
            raise ContestAtomError("ContestAtom.atom_id must be non-empty")
        expected = {
            ContestScopeKind.BYTE: ByteScope,
            ContestScopeKind.PIXEL_REGION: PixelRegionScope,
            ContestScopeKind.FRAME: FrameScope,
            ContestScopeKind.PAIR: PairScope,
        }[self.scope_kind]
        if not isinstance(self.scope, expected):
            raise ContestAtomError(
                f"{self.scope_kind.value} atom requires {expected.__name__} scope"
            )
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise ContestAtomError("ContestAtom is planning/evidence-only and must not claim score")

    @property
    def venn_signature(self) -> str:
        if not self.source_signals:
            return "no_signal"
        return "&".join(sorted(signal.value for signal in self.source_signals))

    @property
    def waterfill_priority(self) -> float:
        mass = self.score.score_mass
        if mass is None:
            mass = abs(float(self.score.score_delta or 0.0))
        charged = max(1, abs(int(self.budget.archive_delta_bytes)))
        return float(mass) / float(charged)

    def to_json(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "scope_kind": self.scope_kind.value,
            "scope": asdict(self.scope),
            "score": asdict(self.score),
            "budget": {
                **asdict(self.budget),
                "rate_score_delta": self.budget.rate_score_delta,
            },
            "source_signals": [signal.value for signal in self.source_signals],
            "venn_signature": self.venn_signature,
            "waterfill_priority": self.waterfill_priority,
            "evidence_axis": self.evidence_axis,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    def to_meta_lagrangian_row(self) -> dict[str, Any]:
        pair_support: list[int] = []
        if isinstance(self.scope, PairScope):
            pair_support = [self.scope.pair_index]
        elif isinstance(self.scope, FrameScope) and self.scope.pair_index is not None:
            pair_support = [self.scope.pair_index]
        elif isinstance(self.scope, PixelRegionScope):
            pair_support = [self.scope.frame_index // 2]

        return {
            "atom_id": self.atom_id,
            "family": f"contest_{self.scope_kind.value}",
            "family_group": "contest_multigrain_score_oracle",
            "pareto_scope": self.scope_kind.value,
            "byte_delta": int(self.budget.archive_delta_bytes),
            "expected_seg_dist_delta": 0.0,
            "expected_pose_dist_delta": 0.0,
            "confidence": 0.0,
            "evidence_grade": self.evidence_axis,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": [
                "contest_granularity_atom_is_planning_only",
                "requires_candidate_builder_and_cpu_oracle_measurement",
                "requires_exact_cuda_auth_eval_before_promotion",
            ],
            "pair_support": pair_support,
            "hard_pair_support": pair_support,
            "research_basis_ids": [
                "contest_auth_eval_score_terms",
                "master_gradient" if ContestSignal.MASTER_GRADIENT in self.source_signals else "",
                "xray_pair_component" if ContestSignal.XRAY_PAIR in self.source_signals else "",
            ],
            "contest_scope": self.to_json(),
        }

    def to_cathedral_candidate_row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.atom_id,
            "family": f"contest_{self.scope_kind.value}",
            "predicted_score_delta": 0.0,
            "expected_information_gain": float(self.score.score_mass or 0.0),
            "estimated_dispatch_cost_usd": float(self.budget.gpu_cost_usd),
            "blockers": [
                "planning_only_multigrain_atom",
                "needs_materialized_candidate_before_dispatch",
            ],
            "notes": (
                f"scope={self.scope_kind.value}; venn={self.venn_signature}; "
                f"waterfill_priority={self.waterfill_priority:.9g}"
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "atom_metadata": self.to_json(),
        }


def pair_atom_from_component_row(
    row: Mapping[str, Any],
    *,
    evidence_axis: str,
    evidence_ref: str,
    source_signal: ContestSignal = ContestSignal.PAIR_COMPONENT,
    selected: bool = False,
) -> ContestAtom:
    pair = _int_field(row, "pair", fallback="pair_idx")
    pose = _float_field(row, "posenet_dist", fallback="pose_dist")
    seg = _float_field(row, "segnet_dist", fallback="seg_dist")
    signals = [source_signal]
    if selected:
        signals.append(ContestSignal.SIDECAR_SELECTED)
    scope = PairScope(pair)
    mode_id = row.get("mode_id")
    mode_suffix = f":mode:{mode_id}" if isinstance(mode_id, str) and mode_id else ""
    return ContestAtom(
        atom_id=f"pair:{pair}{mode_suffix}",
        scope_kind=ContestScopeKind.PAIR,
        scope=scope,
        score=ScoreVector.from_pair_distortions(pose_dist=pose, seg_dist=seg),
        source_signals=tuple(signals),
        evidence_axis=evidence_axis,
        evidence_refs=(evidence_ref,),
        metadata={
            "frame_indices": list(scope.frame_indices),
            "mode_id": mode_id,
            "family": row.get("family"),
            "component_score_no_rate": float(row.get("component_score_no_rate", 0.0)),
            "source_row": dict(row),
        },
    )


def frame_and_pixel_atoms_from_xray_row(
    row: Mapping[str, Any],
    *,
    evidence_axis: str,
    evidence_ref: str,
) -> list[ContestAtom]:
    pair = _int_field(row, "pair_idx", fallback="pair")
    atoms: list[ContestAtom] = []
    for local_frame in (0, 1):
        frame_index = 2 * pair + local_frame
        l1_key = f"frame{local_frame}_l1"
        changed_key = f"frame{local_frame}_changed_fraction"
        l1 = float(row.get(l1_key, 0.0))
        changed_fraction = float(row.get(changed_key, 0.0))
        frame_scope = FrameScope(frame_index=frame_index, pair_index=pair)
        frame_score = ScoreVector(
            evidence_kind="xray_frame_pixel_error",
            score_mass=l1 * changed_fraction,
        )
        atoms.append(
            ContestAtom(
                atom_id=f"frame:{frame_index}",
                scope_kind=ContestScopeKind.FRAME,
                scope=frame_scope,
                score=frame_score,
                source_signals=(ContestSignal.XRAY_PAIR,),
                evidence_axis=evidence_axis,
                evidence_refs=(evidence_ref,),
                metadata={
                    "pair_index": pair,
                    "frame_l1": l1,
                    "changed_fraction": changed_fraction,
                    "source_row": dict(row),
                },
            )
        )
        atoms.append(
            ContestAtom(
                atom_id=f"pixel_region:frame{frame_index}:full",
                scope_kind=ContestScopeKind.PIXEL_REGION,
                scope=PixelRegionScope(frame_index=frame_index),
                score=frame_score,
                source_signals=(ContestSignal.XRAY_PIXEL,),
                evidence_axis=evidence_axis,
                evidence_refs=(evidence_ref,),
                metadata={
                    "pair_index": pair,
                    "region_kind": "full_frame_proxy",
                    "frame_l1": l1,
                    "changed_fraction": changed_fraction,
                },
            )
        )
    return atoms


def byte_atoms_from_master_gradient(
    gradient: Any,
    *,
    anchor: Mapping[str, Any],
    top_k: int,
    evidence_ref: str,
) -> list[ContestAtom]:
    """Build top-k byte atoms from a NumPy gradient array.

    The function accepts the already-loaded array so callers keep control over
    file IO.  2D arrays are treated as per-axis rows; 1D arrays are aggregate
    byte gradients.
    """

    if top_k <= 0:
        return []
    import numpy as np

    arr = np.asarray(gradient)
    if arr.ndim == 2:
        mass = np.sum(np.abs(arr), axis=1)
    elif arr.ndim == 1:
        mass = np.abs(arr)
    else:
        raise ContestAtomError(f"master-gradient array must be 1D or 2D, got {arr.shape}")
    if mass.size == 0:
        return []
    order = np.argsort(-mass)[:top_k]
    score_axis_dominance = anchor.get("score_axis_dominance")
    atoms: list[ContestAtom] = []
    for rank, offset in enumerate(order.tolist()):
        axis_values: list[float]
        if arr.ndim == 2:
            axis_values = [float(value) for value in arr[int(offset)].tolist()]
        else:
            axis_values = [float(arr[int(offset)])]
        scope = ByteScope(
            archive_sha256=str(anchor.get("archive_sha256") or ""),
            byte_domain=str(anchor.get("gradient_byte_domain") or "unknown_byte_domain"),
            start=int(offset),
            length=1,
            member_name="x" if anchor.get("gradient_byte_domain") == "zip_inner_member_payload" else None,
            member_sha256=str(anchor.get("gradient_subject_sha256") or "") or None,
        )
        atoms.append(
            ContestAtom(
                atom_id=f"byte:{scope.byte_domain}:{offset}",
                scope_kind=ContestScopeKind.BYTE,
                scope=scope,
                score=ScoreVector(
                    evidence_kind="master_gradient_abs_mass",
                    score_mass=float(mass[int(offset)]),
                ),
                source_signals=(ContestSignal.MASTER_GRADIENT,),
                evidence_axis=str(anchor.get("measurement_axis") or "[diagnostic]"),
                evidence_refs=(evidence_ref,),
                metadata={
                    "rank": rank,
                    "gradient_axis_values": axis_values,
                    "gradient_array_path": str(anchor.get("gradient_array_path") or ""),
                    "gradient_tensor_kind": str(anchor.get("gradient_tensor_kind") or ""),
                    "score_axis_dominance": score_axis_dominance,
                },
            )
        )
    return atoms


def build_lattice_report(
    atoms: Sequence[ContestAtom],
    *,
    source: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    scope_counts = Counter(atom.scope_kind.value for atom in atoms)
    venn_counts = Counter(atom.venn_signature for atom in atoms)
    top_atoms = sorted(atoms, key=lambda atom: atom.waterfill_priority, reverse=True)
    return {
        "schema": "contest_atom_lattice_v1",
        "generated_at_utc": generated_at_utc,
        "source": source,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "atom_count": len(atoms),
        "scope_counts": dict(sorted(scope_counts.items())),
        "venn_counts": dict(sorted(venn_counts.items())),
        "pair_signal_overlap": pair_signal_overlap(atoms),
        "atoms": [atom.to_json() for atom in atoms],
        "meta_lagrangian_rows": [atom.to_meta_lagrangian_row() for atom in atoms],
        "cathedral_candidate_rows": [atom.to_cathedral_candidate_row() for atom in atoms],
        "top_waterfill_atoms": [atom.to_json() for atom in top_atoms[:32]],
        "blockers": [
            "contest_atom_lattice_is_planning_only",
            "requires_materialized_candidate_archive_for_score_oracle",
            "requires_exact_cuda_auth_eval_before_promotion",
        ],
    }


def pair_signal_overlap(atoms: Sequence[ContestAtom]) -> dict[str, Any]:
    """Return a Venn-style overlap report keyed by scorer pair.

    Exact atom IDs preserve mode/action variants, byte offsets, frames, and
    pixel regions.  For search orchestration we also need to know which signals
    intersect on the same scorer pair.  This report builds that pair-level view
    without discarding the lower-level atoms.
    """

    by_pair: dict[int, dict[str, Any]] = {}
    for atom in atoms:
        pair = _pair_index_for_atom(atom)
        if pair is None:
            continue
        row = by_pair.setdefault(
            pair,
            {
                "pair_index": pair,
                "signals": set(),
                "scope_kinds": set(),
                "atom_ids": [],
                "max_waterfill_priority": 0.0,
                "score_mass_sum": 0.0,
                "pose_score_mass_sum": 0.0,
                "seg_score_mass_sum": 0.0,
                "rate_score_mass_sum": 0.0,
                "pixel_proxy_mass_sum": 0.0,
                "gradient_proxy_mass_sum": 0.0,
            },
        )
        row["signals"].update(signal.value for signal in atom.source_signals)
        row["scope_kinds"].add(atom.scope_kind.value)
        row["atom_ids"].append(atom.atom_id)
        row["max_waterfill_priority"] = max(
            float(row["max_waterfill_priority"]),
            float(atom.waterfill_priority),
        )
        score_mass = float(atom.score.score_mass or 0.0)
        row["score_mass_sum"] += score_mass
        row["pose_score_mass_sum"] += float(atom.score.pose_score_mass or 0.0)
        row["seg_score_mass_sum"] += float(atom.score.seg_score_mass or 0.0)
        row["rate_score_mass_sum"] += float(atom.score.rate_score_mass or 0.0)
        if atom.score.evidence_kind == "xray_frame_pixel_error":
            row["pixel_proxy_mass_sum"] += score_mass
        elif atom.score.evidence_kind == "master_gradient_abs_mass":
            row["gradient_proxy_mass_sum"] += score_mass
    materialized: list[dict[str, Any]] = []
    for pair, row in by_pair.items():
        signals = sorted(row["signals"])
        scopes = sorted(row["scope_kinds"])
        materialized.append(
            {
                "pair_index": pair,
                "signals": signals,
                "scope_kinds": scopes,
                "venn_signature": "&".join(signals) if signals else "no_signal",
                "atom_ids": list(row["atom_ids"]),
                "atom_count": len(row["atom_ids"]),
                "max_waterfill_priority": row["max_waterfill_priority"],
                "score_mass_sum": row["score_mass_sum"],
                "typed_score_masses": {
                    "pose_score_mass_sum": row["pose_score_mass_sum"],
                    "seg_score_mass_sum": row["seg_score_mass_sum"],
                    "rate_score_mass_sum": row["rate_score_mass_sum"],
                    "pixel_proxy_mass_sum": row["pixel_proxy_mass_sum"],
                    "gradient_proxy_mass_sum": row["gradient_proxy_mass_sum"],
                    "component_score_mass_sum": (
                        row["pose_score_mass_sum"] + row["seg_score_mass_sum"]
                    ),
                    "mixed_unit_score_mass_sum": row["score_mass_sum"],
                    "mixed_unit_score_mass_sum_deprecated_for_ranking": True,
                },
	            }
	        )
    materialized.sort(
        key=lambda row: (
            len(row["signals"]),
            len(row["scope_kinds"]),
            float(row["max_waterfill_priority"]),
        ),
        reverse=True,
    )
    counts = Counter(row["venn_signature"] for row in materialized)
    return {
        "pair_count": len(materialized),
        "venn_counts": dict(sorted(counts.items())),
        "top_pairs": materialized[:64],
    }


def _pair_index_for_atom(atom: ContestAtom) -> int | None:
    if isinstance(atom.scope, PairScope):
        return atom.scope.pair_index
    if isinstance(atom.scope, FrameScope):
        return atom.scope.pair_index if atom.scope.pair_index is not None else atom.scope.frame_index // 2
    if isinstance(atom.scope, PixelRegionScope):
        return atom.scope.frame_index // 2
    value = atom.metadata.get("pair_index")
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    values = atom.metadata.get("pair_indices")
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], int):
        return int(values[0])
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ContestAtomError(f"{path}:{line_no}: expected JSON object row")
            rows.append(row)
    return rows


def select_latest_master_gradient_anchor(
    rows: Iterable[Mapping[str, Any]],
    *,
    archive_sha256: str | None = None,
) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    for row in rows:
        if archive_sha256 and row.get("archive_sha256") != archive_sha256:
            continue
        if not row.get("gradient_array_path"):
            continue
        candidates.append(row)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: str(row.get("written_at_utc") or row.get("measurement_utc") or ""),
    )[-1]


def merge_atoms_by_id(atoms: Iterable[ContestAtom]) -> list[ContestAtom]:
    """Merge atoms with identical IDs by unioning evidence signals/refs.

    The first atom supplies scope and score.  Metadata from later duplicates is
    nested under ``merged_metadata`` to avoid silently overwriting values.
    """

    by_id: dict[str, ContestAtom] = {}
    merged_metadata: dict[str, list[Mapping[str, Any]]] = {}
    for atom in atoms:
        existing = by_id.get(atom.atom_id)
        if existing is None:
            by_id[atom.atom_id] = atom
            continue
        signals = tuple(sorted(set(existing.source_signals + atom.source_signals), key=lambda s: s.value))
        refs = tuple(dict.fromkeys(existing.evidence_refs + atom.evidence_refs))
        merged_metadata.setdefault(atom.atom_id, [existing.metadata]).append(atom.metadata)
        by_id[atom.atom_id] = ContestAtom(
            atom_id=existing.atom_id,
            scope_kind=existing.scope_kind,
            scope=existing.scope,
            score=existing.score,
            budget=existing.budget,
            source_signals=signals,
            evidence_axis=existing.evidence_axis,
            evidence_refs=refs,
            metadata={**dict(existing.metadata), "merged_metadata": list(merged_metadata[atom.atom_id])},
        )
    return list(by_id.values())


def _int_field(row: Mapping[str, Any], name: str, *, fallback: str | None = None) -> int:
    value = row.get(name)
    if value is None and fallback is not None:
        value = row.get(fallback)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContestAtomError(f"row missing integer field {name!r}")
    return int(value)


def _float_field(row: Mapping[str, Any], name: str, *, fallback: str | None = None) -> float:
    value = row.get(name)
    if value is None and fallback is not None:
        value = row.get(fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContestAtomError(f"row missing numeric field {name!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ContestAtomError(f"row field {name!r} must be finite")
    return out


__all__ = [
    "BudgetVector",
    "ByteScope",
    "CONTEST_RATE_DENOMINATOR_BYTES",
    "ContestAtom",
    "ContestAtomError",
    "ContestScopeKind",
    "ContestSignal",
    "FrameScope",
    "PairScope",
    "PixelRegionScope",
    "ScoreVector",
    "build_lattice_report",
    "byte_atoms_from_master_gradient",
    "frame_and_pixel_atoms_from_xray_row",
    "load_jsonl",
    "merge_atoms_by_id",
    "pair_atom_from_component_row",
    "pair_signal_overlap",
    "select_latest_master_gradient_anchor",
]
