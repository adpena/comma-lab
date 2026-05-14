# SPDX-License-Identifier: MIT
"""MDL/Bayesian per-tensor codec selector.

Closes Task #308 sub-component (per-tensor optimal codec via Bayesian model
comparison + minimum description length). Pure CPU; no GPU dependency.

Where this fits
---------------
Sibling module :mod:`tac.mdl_bayesian_codec` ranks codec FAMILIES at the
archive-level (one Bayes factor per (codec, archive) pair). This module
operates one level deeper: for each weight TENSOR (per-layer / per-channel
group / per-FiLM-layer), pick the codec with the smallest
``L_total = L(M) + L(D|M)`` from a candidate set.

Per-tensor selection is essential for the four-way stack:
    - PR101-style brotli works well on near-iid bulk weights
    - Arithmetic coding works on small structured residuals
    - Self-compressed Conv2d benefits from per-channel bit-depth
    - FiLM affines stay FP32 (separately handled by zeta-paradigm protect-list)

The selector consumes empirical evidence rows (cathedral TechniqueEvidence-style
dicts OR direct ``TensorObservation`` dataclasses) and produces a ranking
keyed by tensor id.

Math foundation (MacKay 2003 ITILA Ch. 28)
-----------------------------------------

For codec ``M`` applied to tensor ``y``:

    L_total(M | y) = L(M)  +  L(y | M)
                   = bits-to-ship-codec  +  -log2 p(y | M)

Bayes factor between codecs ``M_1`` and ``M_2`` on the same tensor:

    log2 BF_{12} = L_total(M_2) - L_total(M_1)

A positive log2 BF means codec 1 wins. The Occam's-razor sanity check
(refuses a codec whose ``L(M) > L(y_baseline_codec) - L(y_M)``) prevents
"10KB MLP for 5KB savings" anti-patterns.

Per-tensor selection rule:

    selected[tensor_i] = argmin_{M in candidates} L_total(M | y_i)

with the Occam check applied as a second-stage filter.

CLAUDE.md compliance
--------------------
- Pure CPU; no GPU dependency.
- No silent defaults — every public function arg required-keyword.
- No scorer load (this is a meta-comparison framework).
- Empirical evidence rows must carry an ``evidence_grade`` consistent with
  CLAUDE.md non-negotiables (``[contest-CUDA]`` / ``[empirical:<artifact>]``
  / ``[predicted]``). The selector tags every produced ranking with the
  worst-grade among its inputs.

Out of scope (intentional)
--------------------------
- Producing archive bytes (sibling codecs do that).
- Running the candidate codecs (selector consumes pre-computed L_total).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "MDLBayesianSelectorError",
    "TensorObservation",
    "PerTensorRanking",
    "PerTensorSelectionReport",
    "compute_per_tensor_l_total",
    "select_codec_per_tensor",
    "occam_check_codec",
    "load_observations_from_jsonl",
    "rank_observations",
]


class MDLBayesianSelectorError(ValueError):
    """Raised when MDL selector inputs are malformed."""


# Worst-grade ordering for evidence-grade aggregation. Lower index = stronger.
_GRADE_ORDER = (
    "contest-CUDA",
    "empirical",
    "MPS-research-signal",
    "predicted",
    "advisory only",
)


def _normalize_evidence_grade(grade: str) -> str:
    """Map a free-form grade tag to the canonical _GRADE_ORDER set."""
    if not isinstance(grade, str):
        raise MDLBayesianSelectorError(
            f"evidence_grade must be str; got {type(grade).__name__}"
        )
    g = grade.strip().lower()
    if "contest-cuda" in g or "contest_cuda" in g:
        return "contest-CUDA"
    if "empirical" in g:
        return "empirical"
    if "mps-research" in g or "mps_research" in g:
        return "MPS-research-signal"
    if "predicted" in g or "prediction" in g:
        return "predicted"
    if "advisory" in g:
        return "advisory only"
    # Unknown grade — fail-loud so we never silently treat junk as truth.
    raise MDLBayesianSelectorError(
        f"unrecognized evidence_grade {grade!r}; expected one of "
        f"{_GRADE_ORDER}"
    )


def _worst_grade(grades: Iterable[str]) -> str:
    seen = [_normalize_evidence_grade(g) for g in grades]
    if not seen:
        raise MDLBayesianSelectorError(
            "cannot aggregate worst grade over empty set"
        )
    return max(seen, key=lambda g: _GRADE_ORDER.index(g))


# -- Dataclasses --------------------------------------------------------


@dataclass(frozen=True)
class TensorObservation:
    """One empirical or predicted observation of a codec applied to a tensor.

    Args:
        tensor_id: stable id (e.g. ``"renderer.body.0.weight"``).
        codec: codec name (e.g. ``"pr101_split_brotli"``,
            ``"arithmetic_qint"``, ``"selfcomp_conv2d"``).
        n_data_symbols: tensor element count.
        model_bits: ``L(M)`` — bits to ship the codec parameters
            attributable to this tensor (e.g. shared header bytes / n_tensors).
        residual_bits: ``L(D|M)`` — bits to ship the tensor under the codec.
        evidence_grade: per CLAUDE.md non-negotiables.
        notes: free-form provenance string (artifact path, dispatch label).
    """

    tensor_id: str
    codec: str
    n_data_symbols: int
    model_bits: int
    residual_bits: int
    evidence_grade: str = "predicted"
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.tensor_id, str) or not self.tensor_id:
            raise MDLBayesianSelectorError(
                f"tensor_id must be a non-empty str; got {self.tensor_id!r}"
            )
        if not isinstance(self.codec, str) or not self.codec:
            raise MDLBayesianSelectorError(
                f"codec must be a non-empty str; got {self.codec!r}"
            )
        if not isinstance(self.n_data_symbols, int) or self.n_data_symbols < 1:
            raise MDLBayesianSelectorError(
                f"n_data_symbols must be a positive int; got "
                f"{self.n_data_symbols!r}"
            )
        if not isinstance(self.model_bits, int) or self.model_bits < 0:
            raise MDLBayesianSelectorError(
                f"model_bits must be a non-negative int; got "
                f"{self.model_bits!r}"
            )
        if not isinstance(self.residual_bits, int) or self.residual_bits < 0:
            raise MDLBayesianSelectorError(
                f"residual_bits must be a non-negative int; got "
                f"{self.residual_bits!r}"
            )
        # Will raise MDLBayesianSelectorError on unknown grade.
        _normalize_evidence_grade(self.evidence_grade)

    @property
    def total_bits(self) -> int:
        return self.model_bits + self.residual_bits

    @classmethod
    def from_mapping(cls, m: Mapping[str, object]) -> "TensorObservation":
        """Construct from a TechniqueEvidence-style dict.

        Accepts both this module's canonical field names and the
        cathedral TechniqueEvidence aliases (``technique`` -> ``codec``,
        ``empirical_archive_bytes``-> ``residual_bits``, etc.).
        """
        # Field aliases (cathedral TechniqueEvidence -> our schema).
        tid = m.get("tensor_id") or m.get("name") or m.get("layer_id")
        codec = m.get("codec") or m.get("technique") or m.get("name")
        n = m.get("n_data_symbols") or m.get("n_symbols") or m.get("numel")
        mb = m.get("model_bits") or m.get("predicted_model_bytes") or 0
        if isinstance(mb, (int, float)) and "bytes" in (
            "model_bytes" if "model_bytes" in m else ""
        ):
            mb = int(mb) * 8
        rb = (
            m.get("residual_bits")
            or m.get("empirical_archive_bytes")
            or m.get("predicted_archive_bytes")
        )
        if rb is None:
            raise MDLBayesianSelectorError(
                f"observation must have residual_bits / "
                f"empirical_archive_bytes / predicted_archive_bytes; got "
                f"keys={sorted(m.keys())}"
            )
        # cathedral fields are bytes; we convert to bits for L_total accounting.
        if "empirical_archive_bytes" in m or "predicted_archive_bytes" in m:
            rb = int(rb) * 8
        eg = m.get("evidence_grade", "predicted")
        notes = m.get("notes", "")
        return cls(
            tensor_id=str(tid) if tid is not None else "",
            codec=str(codec) if codec is not None else "",
            n_data_symbols=int(n) if n is not None else 1,
            model_bits=int(mb),
            residual_bits=int(rb),
            evidence_grade=str(eg),
            notes=str(notes),
        )


@dataclass
class PerTensorRanking:
    """Codec ranking for one tensor."""

    tensor_id: str
    candidates: list[TensorObservation] = field(default_factory=list)
    """Candidates sorted ascending by ``total_bits`` (best first)."""
    occam_passed: list[bool] = field(default_factory=list)
    """``occam_passed[i]`` is True iff candidates[i] survives the Occam check."""

    @property
    def best(self) -> TensorObservation | None:
        """The codec with smallest ``total_bits`` that passed the Occam check."""
        for cand, ok in zip(self.candidates, self.occam_passed):
            if ok:
                return cand
        return None


@dataclass
class PerTensorSelectionReport:
    """Aggregated per-tensor selection report."""

    rankings: dict[str, PerTensorRanking]
    aggregated_evidence_grade: str
    total_selected_bits: int
    total_selected_bytes: int
    summary_by_codec: dict[str, int]
    """Number of tensors won by each codec."""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-friendly dict."""
        return {
            "aggregated_evidence_grade": self.aggregated_evidence_grade,
            "total_selected_bits": int(self.total_selected_bits),
            "total_selected_bytes": int(self.total_selected_bytes),
            "summary_by_codec": dict(self.summary_by_codec),
            "rankings": {
                tid: {
                    "best": (
                        asdict(r.best) if r.best is not None else None
                    ),
                    "candidates": [asdict(c) for c in r.candidates],
                    "occam_passed": list(r.occam_passed),
                }
                for tid, r in self.rankings.items()
            },
            "notes": list(self.notes),
        }


# -- Pure-math primitives -----------------------------------------------


def compute_per_tensor_l_total(observation: TensorObservation) -> int:
    """``L_total = L(M) + L(D|M)``."""
    return int(observation.model_bits + observation.residual_bits)


def occam_check_codec(
    *,
    candidate: TensorObservation,
    baseline: TensorObservation,
) -> bool:
    """MacKay's Occam-razor sanity check.

    A codec passes iff its ``L(M)`` does not exceed the savings it provides
    over the baseline:

        L(M_candidate) <= max(0, residual_bits_baseline - residual_bits_candidate)

    This prevents the "10KB MLP for 5KB savings" anti-pattern where a
    fancy codec ships more model bytes than it saves on residual bytes.
    """
    if candidate.tensor_id != baseline.tensor_id:
        raise MDLBayesianSelectorError(
            f"Occam check requires same tensor_id; got "
            f"{candidate.tensor_id!r} vs {baseline.tensor_id!r}"
        )
    savings = max(0, int(baseline.residual_bits) - int(candidate.residual_bits))
    return int(candidate.model_bits) <= savings


def _select_baseline_observation(
    observations: Sequence[TensorObservation],
    *,
    baseline_codec: str | None,
) -> TensorObservation:
    """Pick the baseline observation against which Occam runs.

    If ``baseline_codec`` is provided and present, returns that observation;
    otherwise picks the candidate with maximum ``residual_bits`` (the
    cheapest-to-ship codec — usually the static-prior fallback).
    """
    if baseline_codec is not None:
        matches = [o for o in observations if o.codec == baseline_codec]
        if matches:
            return matches[0]
    # Fall back: pick the largest residual_bits row (typically the trivial
    # / static-prior baseline).
    return max(observations, key=lambda o: o.residual_bits)


def rank_observations(
    observations: Sequence[TensorObservation],
    *,
    baseline_codec: str | None = None,
) -> PerTensorRanking:
    """Rank candidates for a single tensor by ``total_bits``.

    Args (required-keyword for ``baseline_codec``):
        observations: candidate observations for the SAME tensor_id.
        baseline_codec: codec name to use as Occam baseline. None -> pick
            the candidate with maximum residual_bits.
    """
    if not observations:
        raise MDLBayesianSelectorError("cannot rank empty observations list")
    tids = {o.tensor_id for o in observations}
    if len(tids) != 1:
        raise MDLBayesianSelectorError(
            f"rank_observations requires same tensor_id for all; got {tids}"
        )
    sorted_cands = sorted(observations, key=compute_per_tensor_l_total)
    baseline = _select_baseline_observation(
        observations, baseline_codec=baseline_codec
    )
    occam_passed = [
        occam_check_codec(candidate=c, baseline=baseline) for c in sorted_cands
    ]
    return PerTensorRanking(
        tensor_id=next(iter(tids)),
        candidates=sorted_cands,
        occam_passed=occam_passed,
    )


def select_codec_per_tensor(
    observations: Iterable[TensorObservation],
    *,
    baseline_codec: str | None = None,
) -> PerTensorSelectionReport:
    """Group observations by tensor_id and produce a ranking per tensor.

    Args (required-keyword):
        observations: any iterable of :class:`TensorObservation`. Multiple
            codecs per tensor are required for ranking to be meaningful;
            tensors with only one observation will still produce a 1-row
            ranking (best = that single row).
        baseline_codec: the codec name used as Occam baseline. None ->
            per-tensor automatic baseline (max residual_bits).

    Returns:
        :class:`PerTensorSelectionReport`.
    """
    by_tensor: dict[str, list[TensorObservation]] = {}
    for o in observations:
        if not isinstance(o, TensorObservation):
            raise MDLBayesianSelectorError(
                f"observation must be TensorObservation; got "
                f"{type(o).__name__}"
            )
        by_tensor.setdefault(o.tensor_id, []).append(o)
    if not by_tensor:
        raise MDLBayesianSelectorError("no observations supplied")

    rankings: dict[str, PerTensorRanking] = {}
    summary: dict[str, int] = {}
    total_selected_bits = 0
    grades_seen: list[str] = []
    notes: list[str] = []
    for tid, obs in by_tensor.items():
        ranking = rank_observations(obs, baseline_codec=baseline_codec)
        rankings[tid] = ranking
        best = ranking.best
        if best is None:
            notes.append(
                f"tensor {tid!r}: NO candidate passed Occam check; falling "
                f"back to cheapest-residual baseline"
            )
            best = _select_baseline_observation(
                obs, baseline_codec=baseline_codec
            )
        summary[best.codec] = summary.get(best.codec, 0) + 1
        total_selected_bits += best.total_bits
        grades_seen.append(best.evidence_grade)
    aggregated = _worst_grade(grades_seen)
    return PerTensorSelectionReport(
        rankings=rankings,
        aggregated_evidence_grade=aggregated,
        total_selected_bits=total_selected_bits,
        total_selected_bytes=int(math.ceil(total_selected_bits / 8)),
        summary_by_codec=summary,
        notes=notes,
    )


def load_observations_from_jsonl(path: str | Path) -> list[TensorObservation]:
    """Load TensorObservation rows from a JSONL file.

    Each line is one JSON object; cathedral TechniqueEvidence aliases are
    auto-mapped via :meth:`TensorObservation.from_mapping`.
    """
    p = Path(path)
    if not p.exists():
        raise MDLBayesianSelectorError(f"observations file not found: {p}")
    out: list[TensorObservation] = []
    for line_ix, raw in enumerate(p.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MDLBayesianSelectorError(
                f"line {line_ix}: invalid JSON ({exc})"
            ) from exc
        out.append(TensorObservation.from_mapping(row))
    if not out:
        raise MDLBayesianSelectorError(
            f"no observations parsed from {p} (empty or all-comments)"
        )
    return out
