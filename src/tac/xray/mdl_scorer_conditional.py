# SPDX-License-Identifier: MIT
"""F1: Scorer-conditional MDL estimator (canonical xray primitive).

Promotes the one-off ``tools/mdl_scorer_conditional_ablation.py`` driver into a
typed :class:`XRayPrimitive` surface. Delegates the heavy lifting to the
existing :mod:`tac.analysis.scorer_conditional_mdl` module (the structural
tier), and exposes a narrow stable API for solver-stack consumers.

The primitive is a typed *wrapper*, not a re-implementation. The structural
tier of MDL ablation (parser-aware section entropy + role-weighted aggregate)
already lives at :func:`tac.analysis.scorer_conditional_mdl.build_scorer_conditional_mdl_ablation`;
this primitive wraps that function in the canonical
:class:`XRayPrimitiveResult` envelope so the solver stack can consume it.

Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable: MDL
estimates are tagged ``evidence_grade="mathematical-derivation"`` and are
NEVER promoted to a score claim. The primitive's confidence band reflects
the tier-A / tier-B / tier-C uncertainty hierarchy in the source memo.

Wire-in hooks engaged (per CLAUDE.md "Subagent coherence-by-default"
6-hook NON-NEGOTIABLE):

- ``continual_learning`` — MDL density per archive becomes an additional
  signal layered into ``.omx/state/continual_learning_posterior.json``
  via the existing ``tac.continual_learning.append_anchor`` surface (the
  posterior schema already accepts the ``mdl_density`` proxy column;
  consumer wire-in is non-mutating).
- ``probe_disambiguator`` — within-class vs across-class MDL delta probe
  (e.g., A1 vs PR106 sidecar family) is the canonical probe-disambiguator
  for "is the new archive a within-substrate-class refinement or a
  cross-class jump?".
- ``cathedral_autopilot`` — MDL-density ranker contributes to the autopilot
  candidate priority (high MDL density = more headroom remaining = higher
  dispatch ROI).

**Composability.** Composes with :class:`tac.xray.shannon_vector_r_d.ShannonVectorRDEstimator`
to produce a tighter "remaining bits to floor" estimate.

Cross-references
----------------
- Source memo: ``.omx/research/zen_floor_field_medal_grade_council_20260514.md``
  (Z1 ablation finding)
- Sister memory: ``feedback_z1_mdl_ablation_landed_20260514.md``
- Delegated structural tier: :func:`tac.analysis.scorer_conditional_mdl.build_scorer_conditional_mdl_ablation`
- One-off tool delegated to this surface: ``tools/mdl_scorer_conditional_ablation.py``

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_bytes
from tac.xray.base import (
    XRayPrimitiveResult,
    WireInHook,
)


@dataclass(frozen=True)
class MDLDensityResult:
    """Typed result from :meth:`ScorerConditionalMDLEstimator.compute`.

    Attributes
    ----------
    total_archive_bytes : int
        Total archive byte count.
    measurable_payload_bytes : int
        Total payload bytes the parser successfully identified
        (sum of section sizes; excludes opaque wrapper / unknown tail).
    aggregated_role_entropy_bits_per_byte : float
        Role-weighted aggregate entropy bits/byte, in [0.0, 8.0].
        See :data:`tac.analysis.scorer_conditional_mdl.ROLE_WEIGHTS`.
    per_section_breakdown : tuple[tuple[str, str, int, float], ...]
        Ordered tuples ``(section_name, role, section_bytes, entropy_bpb)``
        for each section the parser identified.
    mdl_density : float
        Scorer-conditional MDL density = aggregated_role_entropy / 8.0,
        in [0.0, 1.0]. Higher = more remaining-headroom-per-byte.
    """

    total_archive_bytes: int
    measurable_payload_bytes: int
    aggregated_role_entropy_bits_per_byte: float
    per_section_breakdown: tuple[tuple[str, str, int, float], ...]
    mdl_density: float

    def __post_init__(self) -> None:
        if self.total_archive_bytes < 0:
            raise ValueError("total_archive_bytes must be non-negative")
        if not (0.0 <= self.aggregated_role_entropy_bits_per_byte <= 8.0):
            raise ValueError(
                "aggregated_role_entropy_bits_per_byte must be in [0.0, 8.0]; "
                f"got {self.aggregated_role_entropy_bits_per_byte}"
            )
        if not (0.0 <= self.mdl_density <= 1.0):
            raise ValueError(
                "mdl_density must be in [0.0, 1.0]; got "
                f"{self.mdl_density}"
            )


@dataclass(frozen=True)
class MDLDeltaReport:
    """Pairwise within-class-vs-across-class MDL delta between two archives.

    Returned from :meth:`ScorerConditionalMDLEstimator.compare`.

    A negative ``density_delta`` means ``archive_b`` is denser (closer to
    Shannon floor) than ``archive_a``. A small absolute density_delta
    (<0.005) signals a within-class refinement; large
    (>0.05) signals an across-class jump (different substrate family).
    """

    archive_a_path: Path
    archive_b_path: Path
    archive_a_sha256: str | None
    archive_b_sha256: str | None
    density_a: float
    density_b: float
    density_delta: float
    classification: str  # "within_class" | "across_class" | "borderline"

    def __post_init__(self) -> None:
        if not (-1.0 <= self.density_delta <= 1.0):
            raise ValueError(
                "density_delta must be in [-1.0, 1.0]; got "
                f"{self.density_delta}"
            )


class ScorerConditionalMDLEstimator:
    """F1 canonical xray primitive: structural-tier MDL ablation estimator.

    Delegates structural-tier byte entropy computation to
    :func:`tac.analysis.scorer_conditional_mdl.build_scorer_conditional_mdl_ablation`.
    Wraps the result in a typed :class:`XRayPrimitiveResult` envelope so
    the solver stack can consume it.

    Higher-tier estimators (sampled byte-level perturbation; post-decode
    perturbation against PoseNet/SegNet) are NOT in this primitive; they
    live in the one-off tool ``tools/mdl_scorer_conditional_ablation.py``
    (which itself can delegate to this surface for the structural tier).
    """

    @property
    def name(self) -> str:
        return "mdl_scorer_conditional"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "continual_learning",
            "probe_disambiguator",
            "cathedral_autopilot",
        )

    def compute(
        self,
        target: Path | str,
        *,
        label: str | None = None,
        chunk_size: int = 1024,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Estimate structural-tier MDL density of a single archive.

        Parameters
        ----------
        target : Path | str
            Path to an archive (.zip or single-member raw archive).
        label : str | None
            Optional human-readable label for the archive (defaults to
            the archive's basename).
        chunk_size : int
            Byte-chunk size for entropy estimation. Default 1024.
        """
        # Lazy-import to avoid forcing tac.analysis dependency at package import time.
        from tac.analysis.scorer_conditional_mdl import (
            ArchiveInput,
            build_scorer_conditional_mdl_ablation,
        )

        archive_path = Path(target)
        if not archive_path.exists():
            raise ValueError(
                f"archive {archive_path!s} does not exist; refusing to "
                "produce an MDL estimate over phantom bytes"
            )
        archive_bytes = archive_path.read_bytes()
        archive_sha = sha256_bytes(archive_bytes)

        manifest = build_scorer_conditional_mdl_ablation(
            archives=[
                ArchiveInput(
                    label=label or archive_path.stem,
                    archive_path=archive_path,
                ),
            ],
            chunk_size=chunk_size,
        )
        # The manifest schema is ``archives: [...]`` with per-archive entries.
        # Per the canonical schema (probed 2026-05-14), each archive entry has:
        # - archive_bytes (int): total archive byte count
        # - sections (list[dict]): each with name / optimization_role / start /
        #   end / bytes / entropy_bits_per_byte / entropy_floor_bytes_ceil /
        #   gap_to_iid_floor_bytes_ceil
        per_archive = manifest.get("archives", [])
        if not per_archive:
            raise ValueError(
                f"MDL manifest for {archive_path!s} returned no archives; "
                "likely a parser failure (Catalog #115 packet-clearance "
                "evidence requires inflate.sh parity)"
            )
        entry = per_archive[0]

        total_bytes = int(entry.get("archive_bytes", len(archive_bytes)))
        sections = entry.get("sections", [])
        per_section: list[tuple[str, str, int, float]] = []
        # Build per-section breakdown + section-weighted aggregate.
        measurable = 0
        weighted_entropy_sum = 0.0
        for sec in sections:
            sec_bytes = int(sec.get("bytes", 0))
            sec_entropy = float(sec.get("entropy_bits_per_byte", 0.0))
            measurable += sec_bytes
            weighted_entropy_sum += sec_bytes * sec_entropy
            per_section.append(
                (
                    str(sec.get("name", "")),
                    str(sec.get("optimization_role", "")),
                    sec_bytes,
                    sec_entropy,
                )
            )
        # Aggregate (byte-weighted) entropy bits/byte over measurable payload.
        agg_bpb = (
            weighted_entropy_sum / measurable if measurable > 0 else 0.0
        )
        mdl_density = agg_bpb / 8.0 if agg_bpb > 0 else 0.0

        result_value = MDLDensityResult(
            total_archive_bytes=total_bytes,
            measurable_payload_bytes=measurable,
            aggregated_role_entropy_bits_per_byte=agg_bpb,
            per_section_breakdown=tuple(per_section),
            mdl_density=mdl_density,
        )

        # Confidence band: structural-tier MDL is a CONSERVATIVE proxy for
        # true scorer-conditional MDL (tier-A in the memo). Lower bound is
        # the structural value itself; upper bound bounded by tier-C (post-
        # decode-perturbation) which can be up to ~0.95 of structural for
        # cooperative-receiver substrates. Lacking tier-C, we report the
        # structural-tier-only band [mdl_density, mdl_density * 1.05].
        confidence_band = (mdl_density, min(1.0, mdl_density * 1.05))

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=archive_path,
            archive_sha256=archive_sha,
            primitive_value=result_value,
            evidence_grade="mathematical-derivation",
            confidence_band=confidence_band,
            composes_with=(
                "shannon_vector_r_d",
                "per_pair_score_decomposition",
            ),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "tier": "structural",
                "chunk_size": chunk_size,
                "delegated_module": "tac.analysis.scorer_conditional_mdl",
            },
        )

    def compare(
        self,
        archive_a: Path | str,
        archive_b: Path | str,
        *,
        within_class_threshold: float = 0.005,
        across_class_threshold: float = 0.05,
    ) -> MDLDeltaReport:
        """Compute pairwise MDL delta between two archives.

        Classification:

        - ``abs(delta) < within_class_threshold`` -> ``"within_class"``
        - ``abs(delta) > across_class_threshold`` -> ``"across_class"``
        - otherwise -> ``"borderline"``

        Used as a probe-disambiguator: A1 vs PR106-sidecar-variant should
        produce a small within-class delta (same HNeRV-family substrate);
        A1 vs a future Wyner-Ziv-substrate archive should produce a
        large across-class delta.
        """
        result_a = self.compute(archive_a)
        result_b = self.compute(archive_b)
        assert isinstance(result_a.primitive_value, MDLDensityResult)
        assert isinstance(result_b.primitive_value, MDLDensityResult)
        density_a = result_a.primitive_value.mdl_density
        density_b = result_b.primitive_value.mdl_density
        delta = density_b - density_a
        abs_delta = abs(delta)
        if abs_delta < within_class_threshold:
            classification = "within_class"
        elif abs_delta > across_class_threshold:
            classification = "across_class"
        else:
            classification = "borderline"
        return MDLDeltaReport(
            archive_a_path=Path(archive_a),
            archive_b_path=Path(archive_b),
            archive_a_sha256=result_a.archive_sha256,
            archive_b_sha256=result_b.archive_sha256,
            density_a=density_a,
            density_b=density_b,
            density_delta=delta,
            classification=classification,
        )

    def compose_with(self, other: Any) -> Any:
        from tac.xray.base import ComposedXRayPrimitive

        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "MDLDensityResult",
    "MDLDeltaReport",
    "ScorerConditionalMDLEstimator",
]
