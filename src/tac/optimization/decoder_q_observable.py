# SPDX-License-Identifier: MIT
"""Observable decoder-q mutation lattice for FEC6/PR101 planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # torch is already a runtime dependency for this contest substrate.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DecoderQAtom:
    """One q-domain candidate atom with planning and measurement metadata."""

    candidate_id: str
    tensor_name: str
    q_offset: int
    delta: int
    q_before: int | None
    q_after: int | None
    target_mass: float
    axis_mass: dict[str, float]
    length_delta: int
    fixed_length_runtime_compatible: bool
    advisory_delta_score: float | None = None
    advisory_score: float | None = None
    advisory_seg: float | None = None
    advisory_pose: float | None = None

    @property
    def signed_direction(self) -> int:
        return 1 if self.delta > 0 else -1

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "tensor_name": self.tensor_name,
            "q_offset": self.q_offset,
            "delta": self.delta,
            "q_before": self.q_before,
            "q_after": self.q_after,
            "target_mass": self.target_mass,
            "axis_mass": self.axis_mass,
            "length_delta": self.length_delta,
            "fixed_length_runtime_compatible": self.fixed_length_runtime_compatible,
            "advisory_delta_score": self.advisory_delta_score,
            "advisory_score": self.advisory_score,
            "advisory_seg": self.advisory_seg,
            "advisory_pose": self.advisory_pose,
        }


class DecoderQObservableLattice:
    """A small tensorized observability surface for decoder-q atoms.

    This is intentionally not an authority surface.  It lets search code express
    score priors, measured advisory response, and fixed-length constraints in a
    model-like form, while byte rebuilding and scorer evaluation remain the
    source of truth.
    """

    def __init__(self, atoms: list[DecoderQAtom]):
        self.atoms = atoms

    @classmethod
    def from_feasibility_and_advisory(
        cls,
        feasibility: dict[str, Any],
        advisory_summary: dict[str, Any] | None = None,
        *,
        baseline_score: float | None = None,
    ) -> "DecoderQObservableLattice":
        advisory_by_id = _advisory_by_candidate_id(advisory_summary, baseline_score)
        atoms: list[DecoderQAtom] = []
        for row in feasibility.get("fixed_length_runtime_compatible_rows", []):
            if not isinstance(row, dict):
                continue
            mutation = row.get("mutation")
            if not isinstance(mutation, dict):
                continue
            advisory = advisory_by_id.get(str(row.get("mutation_id")))
            axis_mass = _axis_mass(row)
            atoms.append(
                DecoderQAtom(
                    candidate_id=str(row.get("mutation_id")),
                    tensor_name=str(mutation["tensor_name"]),
                    q_offset=int(mutation["q_offset"]),
                    delta=int(mutation["delta"]),
                    q_before=_maybe_int(row.get("q_before")),
                    q_after=_maybe_int(row.get("q_after")),
                    target_mass=_target_mass(row),
                    axis_mass=axis_mass,
                    length_delta=int(row.get("length_delta", 0)),
                    fixed_length_runtime_compatible=bool(row.get("fixed_length_runtime_compatible")),
                    advisory_delta_score=advisory.get("delta_vs_baseline_score") if advisory else None,
                    advisory_score=advisory.get("canonical_score") if advisory else None,
                    advisory_seg=advisory.get("avg_segnet_dist") if advisory else None,
                    advisory_pose=advisory.get("avg_posenet_dist") if advisory else None,
                )
            )
        return cls(atoms)

    def tensorize(self, *, device: str = "cpu") -> dict[str, Any]:
        """Return tensors for ranking/plotting/search."""

        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for DecoderQObservableLattice.tensorize")
        n = len(self.atoms)
        target_mass = torch.tensor([atom.target_mass for atom in self.atoms], dtype=torch.float64, device=device)
        axis_mass = torch.tensor(
            [
                [
                    atom.axis_mass.get("seg", 0.0),
                    atom.axis_mass.get("pose", 0.0),
                    atom.axis_mass.get("rate", 0.0),
                ]
                for atom in self.atoms
            ],
            dtype=torch.float64,
            device=device,
        )
        delta = torch.tensor([atom.delta for atom in self.atoms], dtype=torch.float64, device=device)
        q_offset = torch.tensor([atom.q_offset for atom in self.atoms], dtype=torch.int64, device=device)
        fixed = torch.tensor([atom.fixed_length_runtime_compatible for atom in self.atoms], dtype=torch.bool, device=device)
        measured_mask = torch.tensor(
            [atom.advisory_delta_score is not None for atom in self.atoms],
            dtype=torch.bool,
            device=device,
        )
        measured_delta = torch.full((n,), float("nan"), dtype=torch.float64, device=device)
        for index, atom in enumerate(self.atoms):
            if atom.advisory_delta_score is not None:
                measured_delta[index] = float(atom.advisory_delta_score)
        return {
            "target_mass": target_mass,
            "axis_mass": axis_mass,
            "delta": delta,
            "q_offset": q_offset,
            "fixed_length_mask": fixed,
            "measured_mask": measured_mask,
            "measured_delta_score": measured_delta,
            "atom_index": list(range(n)),
        }

    def signed_slope_rows(self) -> list[dict[str, Any]]:
        """Estimate local signed response where +/- pairs are measured."""

        by_offset: dict[tuple[str, int], dict[int, DecoderQAtom]] = {}
        for atom in self.atoms:
            if atom.advisory_delta_score is None:
                continue
            by_offset.setdefault((atom.tensor_name, atom.q_offset), {})[atom.delta] = atom
        rows = []
        for (tensor_name, q_offset), by_delta in sorted(by_offset.items()):
            slope = None
            source = None
            if -1 in by_delta and 1 in by_delta:
                slope = (
                    float(by_delta[1].advisory_delta_score)
                    - float(by_delta[-1].advisory_delta_score)
                ) / 2.0
                source = "pm1"
            elif -2 in by_delta and 2 in by_delta:
                slope = (
                    float(by_delta[2].advisory_delta_score)
                    - float(by_delta[-2].advisory_delta_score)
                ) / 4.0
                source = "pm2"
            if slope is None:
                continue
            rows.append(
                {
                    "tensor_name": tensor_name,
                    "q_offset": q_offset,
                    "signed_score_slope_per_q_step": slope,
                    "preferred_sign": -1 if slope > 0 else 1,
                    "source": source,
                }
            )
        return rows

    def top_atoms(self, *, limit: int = 16, require_unmeasured: bool = False) -> list[dict[str, Any]]:
        atoms = [
            atom
            for atom in self.atoms
            if atom.fixed_length_runtime_compatible
            and (not require_unmeasured or atom.advisory_delta_score is None)
        ]
        atoms.sort(
            key=lambda atom: (
                atom.advisory_delta_score if atom.advisory_delta_score is not None else float("inf"),
                -atom.target_mass,
                abs(atom.delta),
                atom.tensor_name,
                atom.q_offset,
                atom.delta,
            )
        )
        return [atom.as_dict() for atom in atoms[: max(0, int(limit))]]

    def summary(self) -> dict[str, Any]:
        measured = [atom for atom in self.atoms if atom.advisory_delta_score is not None]
        improved = [
            atom
            for atom in measured
            if atom.advisory_delta_score is not None and atom.advisory_delta_score < 0.0
        ]
        return {
            "atom_count": len(self.atoms),
            "fixed_length_count": sum(1 for atom in self.atoms if atom.fixed_length_runtime_compatible),
            "measured_count": len(measured),
            "improved_count": len(improved),
            "best_measured": min(
                (atom.as_dict() for atom in measured),
                key=lambda row: float(row["advisory_delta_score"]),
                default=None,
            ),
            "signed_slope_count": len(self.signed_slope_rows()),
        }


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _target_mass(row: dict[str, Any]) -> float:
    evidence = row.get("op3v3_target_evidence")
    if isinstance(evidence, dict):
        return float(evidence.get("score_impact_abs_sum", 0.0))
    return 0.0


def _axis_mass(row: dict[str, Any]) -> dict[str, float]:
    evidence = row.get("op3v3_target_evidence")
    axis = evidence.get("axis_score_impact_abs_sum") if isinstance(evidence, dict) else None
    if not isinstance(axis, dict):
        axis = {}
    return {
        "seg": float(axis.get("seg", 0.0)),
        "pose": float(axis.get("pose", 0.0)),
        "rate": float(axis.get("rate", 0.0)),
    }


def _advisory_by_candidate_id(
    advisory_summary: dict[str, Any] | None,
    baseline_score: float | None,
) -> dict[str, dict[str, float]]:
    if advisory_summary is None:
        return {}
    out = {}
    for row in advisory_summary.get("candidates", []):
        if not isinstance(row, dict) or not row.get("candidate_id"):
            continue
        advisory = row.get("advisory_eval")
        if not isinstance(advisory, dict):
            continue
        score = advisory.get("canonical_score")
        delta = row.get("delta_vs_baseline_score")
        if delta is None and score is not None and baseline_score is not None:
            delta = float(score) - float(baseline_score)
        out[str(row["candidate_id"])] = {
            "canonical_score": float(score) if score is not None else None,
            "delta_vs_baseline_score": float(delta) if delta is not None else None,
            "avg_segnet_dist": float(advisory["avg_segnet_dist"]) if advisory.get("avg_segnet_dist") is not None else None,
            "avg_posenet_dist": float(advisory["avg_posenet_dist"]) if advisory.get("avg_posenet_dist") is not None else None,
        }
    return out


__all__ = ["DecoderQAtom", "DecoderQObservableLattice"]
