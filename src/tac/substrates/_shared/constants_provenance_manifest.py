# SPDX-License-Identifier: MIT
"""Constants-provenance manifest (v1) — the executable cure for the ARBITRARINESS class.

Source: operator hardening packet 2026-06-09 (the arbitrariness-class crux). The
lab discovered the ARBITRARINESS CLASS: score-relevant magic constants set by
CONVENTION, not derived/measured/learned. The worked symptom is the carrier's
SIREN ``w=30`` (one scalar standing in for a whole spectral geometry); the class
also includes ``distill_weight=0.0`` (the Mistake-B objective-starvation bug),
grad-clip norm, EMA decay 0.997, stage epoch counts, latent dims, the channel
taper, mid/fine injection block indices, codebook size, brotli quality=11, the
trust-region slacks, and the recon-vs-scorer loss weights. Each scalar stands in
for a decision that SHOULD have a provenance.

This module is the typed, machine-checkable manifest that states, per constant,
its PROVENANCE — one of:

* ``DERIVED`` — from a formula / theorem (e.g. a per-scale Nyquist frequency cap).
* ``MEASURED`` — from the data / scorer (e.g. an ``omega`` from the scorer spectral
  transfer function ``tac.analysis.scorer_spectral_sensitivity_v2``). A MEASURED
  constant is ONLY valid within its ``measurement_scope`` — measured-on-4-pairs-at-
  one-amplitude used globally is still fragile (the adversarial guard
  "measured can be cargo-cult too").
* ``LEARNED`` — a trainable parameter (gradient descent finds it).
* ``ARBITRARY`` — convention / cargo-cult inheritance. Must be justified
  (a ``replacement_path`` naming how to derive/measure/learn it) or it BLOCKS the
  vehicle from claiming intrinsic optimization (L2).

This is the HARD-EARNED vs CARGO-CULTED classification (CLAUDE.md cargo-cult audit
/ Catalog #303) applied at the CONSTANT level. It slots into the Vehicle OS
maturity ladder: per ``docs/vehicle_operating_system.md`` L2 = "intrinsically
optimized", and a vehicle cannot honestly be L2 while a SCORE-RELEVANT constant is
ARBITRARY (a constant that directly moves ``100*d_seg + sqrt(10*d_pose) + bytes``
but was guessed means the vehicle is NOT intrinsically optimized — it is
intrinsically un-tuned).

GUARDRAIL (operator-explicit): only ``score_relevant`` OR ``stability_critical``
constants block. Harmless engineering constants (log cadence, buffer sizes,
display precision) are tagged ``score_relevant=False, stability_critical=False``
and are EXEMPT — the gate must not bureaucratize them. This is the difference
between the arbitrariness cure and arbitrariness theater.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "Results must
become system intelligence": this manifest is a reusable typed surface, emitted as
durable JSON under ``.omx/state/constants_provenance/`` (NEVER ``/tmp``), so the
solver / autopilot / next subagent inherits the constant provenance map rather
than re-deriving it from prose. Sister of
``tac.substrates._shared.vehicle_fidelity_manifest`` (which extincts the
orthogonal NAME-LAUNDERING bug class at the docstring surface); this module
extincts the ARBITRARINESS bug class at the constant-provenance surface.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Catalog #287 placeholder discipline: a
``replacement_path`` that is a real tool / module / theorem reference counts;
placeholder literals (``TBD`` / ``<value>`` / ``pending``) are REJECTED so a
constant cannot be un-blocked by a fake replacement plan.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CONSTANT_PROVENANCE_VALUES",
    "MATURITY_LEVELS",
    "PLACEHOLDER_LITERALS",
    "ConstantProvenance",
    "ConstantsProvenanceManifest",
    "ConstantsProvenanceVerifyError",
    "MeasurementScope",
    "constants_provenance_state_dir",
    "emit_constants_provenance_manifest",
    "manifest_path_for_constants",
]

# The canonical 4-value provenance taxonomy (operator 2026-06-09).
CONSTANT_PROVENANCE_VALUES: tuple[str, ...] = ("DERIVED", "MEASURED", "LEARNED", "ARBITRARY")

# Vehicle OS maturity ladder labels (docs/vehicle_operating_system.md L0-L7). The
# gate's blocking surface is "declared maturity >= L2" (intrinsically optimized).
MATURITY_LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")

# Catalog #287 placeholder literals rejected in a replacement_path (a fake plan
# must not un-block an ARBITRARY score-relevant constant).
PLACEHOLDER_LITERALS: frozenset[str] = frozenset(
    {"tbd", "<value>", "<rationale>", "<reason>", "placeholder", "pending", "pending_ratification", "todo"}
)


def _maturity_index(level: str) -> int:
    try:
        return MATURITY_LEVELS.index(level)
    except ValueError as exc:  # pragma: no cover - guarded at construction
        raise ValueError(
            f"maturity level {level!r} not in {MATURITY_LEVELS}"
        ) from exc


@dataclass(frozen=True)
class MeasurementScope:
    """The validity envelope of a MEASURED constant (the adversarial guard).

    "Measured can be cargo-cult too": a constant measured on 4 pairs at one
    amplitude on one scorer surface is fragile when used globally. This records
    EXACTLY where the measurement holds so a reviewer (and the gate) can see when
    a MEASURED value is being over-extended.
    """

    pairs: int = 0
    """Number of source frame pairs the measurement averaged over."""

    frames: int = 0
    """Number of frames (or 0 if expressed in pairs)."""

    amplitude_range: tuple[float, ...] = ()
    """Perturbation amplitudes (LSB) the measurement swept (empty = not swept)."""

    scorer_surfaces: tuple[str, ...] = ()
    """Which scorer terms the measurement covered, e.g. ``("d_seg", "d_pose")``."""

    authority_tier: str = ""
    """The evidence axis, e.g. ``macos_cpu_advisory`` / ``contest_cpu`` /
    ``exact_pair_scorer``. A MEASURED constant from a ``macos_cpu_advisory`` scope
    is mechanism-grade, not a contest-axis fact."""

    confidence_interval: str = ""
    """A human/machine note on the CI of the measurement (e.g. ``+-0.02 (n=6x2)``).
    Empty when the measurement reported no CI (itself a fragility signal)."""

    artifact_path: str = ""
    """Path to the measurement artifact (e.g. a v2 atlas JSON). Durable, NOT /tmp."""

    def __post_init__(self) -> None:
        if self.pairs < 0 or self.frames < 0:
            raise ValueError("MeasurementScope pairs/frames must be non-negative")
        if self.artifact_path and self.artifact_path.startswith("/tmp"):
            raise ValueError(
                "MeasurementScope.artifact_path must be durable (not /tmp) per "
                "CLAUDE.md disk hygiene"
            )

    def is_empty(self) -> bool:
        """True when the scope records no real measurement evidence at all.

        An empty scope on a MEASURED constant is itself a finding: the constant
        claims to be measured but names no pairs/frames/amplitudes/surface/artifact.
        """
        return (
            self.pairs == 0
            and self.frames == 0
            and not self.amplitude_range
            and not self.scorer_surfaces
            and not self.artifact_path
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "pairs": self.pairs,
            "frames": self.frames,
            "amplitude_range": list(self.amplitude_range),
            "scorer_surfaces": list(self.scorer_surfaces),
            "authority_tier": self.authority_tier,
            "confidence_interval": self.confidence_interval,
            "artifact_path": self.artifact_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MeasurementScope:
        return cls(
            pairs=int(payload.get("pairs", 0) or 0),
            frames=int(payload.get("frames", 0) or 0),
            amplitude_range=tuple(
                float(x) for x in (payload.get("amplitude_range", ()) or ())
            ),
            scorer_surfaces=_as_str_tuple(payload.get("scorer_surfaces", ())),
            authority_tier=str(payload.get("authority_tier", "")),
            confidence_interval=str(payload.get("confidence_interval", "")),
            artifact_path=str(payload.get("artifact_path", "")),
        )


@dataclass(frozen=True)
class ConstantProvenance:
    """Provenance record for a single score-relevant (or stability-critical) constant.

    The blocking rule (see :meth:`ConstantsProvenanceManifest.blocking_findings`):
    a constant is BLOCKING at maturity >= ``blocking_maturity_level`` when it is
    (``score_relevant`` OR ``stability_critical``) AND ``provenance == ARBITRARY``
    AND it has NO real ``replacement_path``. A MEASURED constant with an empty (or
    over-narrow) ``measurement_scope`` raises a softer advisory (it is not yet
    blocking, but it is fragile).
    """

    constant_name: str
    value: object
    provenance: str
    """One of :data:`CONSTANT_PROVENANCE_VALUES`."""

    score_relevant: bool
    """Does this constant directly move ``100*d_seg + sqrt(10*d_pose) + bytes``?"""

    stability_critical: bool = False
    """Does a wrong value cause training divergence / NaN / collapse (even if not
    directly score-moving)? grad-clip + EMA decay are the canonical examples."""

    owner: str = ""
    """The module / function that owns the constant (where to change it)."""

    replacement_path: str = ""
    """How to make this constant non-arbitrary: a tool / module / theorem reference
    (e.g. ``tools/measure_scorer_spectral_sensitivity.py v2`` for ``w``, or a
    Nyquist-cap formula). Required to un-block an ARBITRARY score-relevant constant.
    Placeholder literals (Catalog #287) are rejected."""

    blocking_maturity_level: str = "L2"
    """The maturity level at/above which an unresolved ARBITRARY value blocks
    (default L2 = intrinsically optimized per the Vehicle OS ladder)."""

    measurement_scope: MeasurementScope = field(default_factory=MeasurementScope)
    """For MEASURED constants: the validity envelope (the adversarial guard)."""

    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.constant_name, str) or not self.constant_name.strip():
            raise ValueError("constant_name must be a non-empty string")
        if self.provenance not in CONSTANT_PROVENANCE_VALUES:
            raise ValueError(
                f"provenance {self.provenance!r} not in {CONSTANT_PROVENANCE_VALUES}"
            )
        if self.blocking_maturity_level not in MATURITY_LEVELS:
            raise ValueError(
                f"blocking_maturity_level {self.blocking_maturity_level!r} not in "
                f"{MATURITY_LEVELS}"
            )
        # Catalog #287: a replacement_path that is a placeholder literal is rejected
        # so a fake plan cannot un-block an ARBITRARY score-relevant constant.
        rp = self.replacement_path.strip().lower()
        if rp and rp in PLACEHOLDER_LITERALS:
            raise ValueError(
                f"ConstantProvenance({self.constant_name!r}) replacement_path "
                f"{self.replacement_path!r} is a forbidden placeholder literal "
                f"(Catalog #287). Name a real tool / module / theorem."
            )

    @property
    def is_gated(self) -> bool:
        """A constant participates in the gate iff it is score-relevant OR
        stability-critical. Everything else is harmless engineering (the guardrail)."""
        return bool(self.score_relevant or self.stability_critical)

    def has_real_replacement(self) -> bool:
        """True when a non-placeholder replacement_path is present."""
        rp = self.replacement_path.strip()
        return bool(rp) and rp.lower() not in PLACEHOLDER_LITERALS

    def blocks_at(self, declared_maturity_level: str) -> bool:
        """True when this constant BLOCKS the vehicle at ``declared_maturity_level``.

        Blocks iff: (score_relevant OR stability_critical) AND provenance==ARBITRARY
        AND no real replacement_path AND the declared maturity is at/above this
        constant's ``blocking_maturity_level``.
        """
        if not self.is_gated:
            return False
        if self.provenance != "ARBITRARY":
            return False
        if self.has_real_replacement():
            return False
        return _maturity_index(declared_maturity_level) >= _maturity_index(
            self.blocking_maturity_level
        )

    def fragility_advisory(self) -> str | None:
        """Soft advisory for a MEASURED gated constant whose scope is empty/narrow.

        Returns ``None`` when clean. A MEASURED score-relevant constant with an
        empty measurement_scope (no pairs/frames/amplitudes/surface/artifact) is
        fragile — "measured can be cargo-cult too." This does NOT block (it is a
        true MEASURED, not ARBITRARY) but it is surfaced so a reviewer sees a
        measured value being asserted without its validity envelope.
        """
        if not self.is_gated:
            return None
        if self.provenance != "MEASURED":
            return None
        if self.measurement_scope.is_empty():
            return (
                f"FRAGILE-MEASURED (advisory): constant {self.constant_name!r} is "
                f"tagged MEASURED + gated, but its measurement_scope is EMPTY (no "
                f"pairs/frames/amplitude_range/scorer_surfaces/artifact). A measured "
                f"value with no validity envelope is cargo-cult-shaped — record the "
                f"scope it was measured under (the adversarial 'measured can be "
                f"cargo-cult too' guard)."
            )
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "constant_name": self.constant_name,
            "value": self.value,
            "provenance": self.provenance,
            "score_relevant": self.score_relevant,
            "stability_critical": self.stability_critical,
            "owner": self.owner,
            "replacement_path": self.replacement_path,
            "blocking_maturity_level": self.blocking_maturity_level,
            "measurement_scope": self.measurement_scope.as_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConstantProvenance:
        scope_raw = payload.get("measurement_scope", {}) or {}
        return cls(
            constant_name=str(payload["constant_name"]),
            value=payload.get("value"),
            provenance=str(payload["provenance"]),
            score_relevant=bool(payload.get("score_relevant", False)),
            stability_critical=bool(payload.get("stability_critical", False)),
            owner=str(payload.get("owner", "")),
            replacement_path=str(payload.get("replacement_path", "")),
            blocking_maturity_level=str(payload.get("blocking_maturity_level", "L2")),
            measurement_scope=MeasurementScope.from_dict(scope_raw),  # type: ignore[arg-type]
            notes=str(payload.get("notes", "")),
        )


class ConstantsProvenanceVerifyError(ValueError):
    """Raised by :meth:`ConstantsProvenanceManifest.verify` on a blocking finding."""


@dataclass(frozen=True)
class ConstantsProvenanceManifest:
    """Typed constants-provenance manifest for one vehicle.

    Schema (operator 2026-06-09):

    * ``vehicle_id`` — canonical substrate dir name (e.g. ``hi_nerv``).
    * ``declared_maturity_level`` — the maturity the vehicle CLAIMS (L0..L7). The
      gate blocks when this is >= a constant's ``blocking_maturity_level`` and the
      constant is an unresolved ARBITRARY score-relevant/stability-critical value.
    * ``constants`` — list of :class:`ConstantProvenance`.
    * ``source_memos`` — the design/audit memos this manifest was populated from.
    """

    vehicle_id: str
    declared_maturity_level: str
    constants: tuple[ConstantProvenance, ...] = ()
    summary: str = ""
    source_memos: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, str) or not self.vehicle_id.strip():
            raise ValueError("vehicle_id must be a non-empty string")
        if self.declared_maturity_level not in MATURITY_LEVELS:
            raise ValueError(
                f"declared_maturity_level {self.declared_maturity_level!r} not in "
                f"{MATURITY_LEVELS}"
            )
        names = [c.constant_name for c in self.constants]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(
                f"{self.vehicle_id}: duplicate constant_name(s) {sorted(dupes)}"
            )

    # -- query helpers -----------------------------------------------------

    def gated_constants(self) -> tuple[ConstantProvenance, ...]:
        """The constants that participate in the gate (score-relevant OR stability)."""
        return tuple(c for c in self.constants if c.is_gated)

    def arbitrary_score_relevant(self) -> tuple[ConstantProvenance, ...]:
        return tuple(
            c
            for c in self.constants
            if c.score_relevant and c.provenance == "ARBITRARY"
        )

    def provenance_histogram(self) -> dict[str, int]:
        hist = dict.fromkeys(CONSTANT_PROVENANCE_VALUES, 0)
        for c in self.constants:
            hist[c.provenance] += 1
        return hist

    # -- the fail-closed blocking check ------------------------------------

    def blocking_findings(self) -> tuple[str, ...]:
        """Return the blocking findings (empty tuple = clean for the gate).

        A finding is a gated (score-relevant OR stability-critical) ARBITRARY
        constant with NO real replacement_path, when the vehicle's
        ``declared_maturity_level`` is at/above the constant's
        ``blocking_maturity_level``. This is the executable
        "a vehicle cannot reach L2 while score-relevant constants are ARBITRARY"
        rule.
        """
        findings: list[str] = []
        for c in self.constants:
            if c.blocks_at(self.declared_maturity_level):
                kind = []
                if c.score_relevant:
                    kind.append("score_relevant")
                if c.stability_critical:
                    kind.append("stability_critical")
                findings.append(
                    f"ARBITRARY-CONSTANT-AT-{self.declared_maturity_level}: "
                    f"{c.constant_name!r} (value={c.value!r}, {'+'.join(kind)}) is "
                    f"tagged ARBITRARY with no replacement_path, but the vehicle "
                    f"declares maturity {self.declared_maturity_level} >= the "
                    f"constant's blocking level {c.blocking_maturity_level}. A "
                    f"vehicle cannot be intrinsically optimized (L2+) while a "
                    f"score-relevant/stability-critical constant is a guess. Derive "
                    f"it (formula), MEASURE it (scorer), LEARN it (trainable), OR "
                    f"name a replacement_path."
                )
        return tuple(findings)

    def fragility_advisories(self) -> tuple[str, ...]:
        """Soft advisories for MEASURED gated constants with empty measurement_scope."""
        out: list[str] = []
        for c in self.constants:
            adv = c.fragility_advisory()
            if adv is not None:
                out.append(adv)
        return tuple(out)

    def verify(self) -> None:
        """Fail closed on the ARBITRARINESS bug class.

        Raises :class:`ConstantsProvenanceVerifyError` when
        :meth:`blocking_findings` is non-empty — i.e. when the vehicle declares
        L2+ maturity while a score-relevant/stability-critical constant is an
        unresolved ARBITRARY guess. A vehicle whose ARBITRARY constants all carry
        a replacement_path (or whose gated constants are all DERIVED/MEASURED/
        LEARNED) verifies — fragility advisories are a SEPARATE soft signal.
        """
        findings = self.blocking_findings()
        if findings:
            raise ConstantsProvenanceVerifyError(
                f"constants_provenance_manifest.verify() failed for "
                f"{self.vehicle_id!r} (declared_maturity={self.declared_maturity_level}):\n  "
                + "\n  ".join(findings)
            )

    # -- serialization -----------------------------------------------------

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "constants_provenance_manifest.v1",
            "vehicle_id": self.vehicle_id,
            "declared_maturity_level": self.declared_maturity_level,
            "constants": [c.as_dict() for c in self.constants],
            "provenance_histogram": self.provenance_histogram(),
            "summary": self.summary,
            "source_memos": list(self.source_memos),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ConstantsProvenanceManifest:
        schema = str(payload.get("schema", ""))
        if schema and schema != "constants_provenance_manifest.v1":
            raise ValueError(
                f"unexpected schema {schema!r}; expected "
                f"'constants_provenance_manifest.v1'"
            )
        raw = payload.get("constants", []) or []
        if not isinstance(raw, Iterable):
            raise ValueError("constants must be a list")
        constants = tuple(
            ConstantProvenance.from_dict(c)  # type: ignore[arg-type]
            for c in raw
        )
        return cls(
            vehicle_id=str(payload["vehicle_id"]),
            declared_maturity_level=str(payload["declared_maturity_level"]),
            constants=constants,
            summary=str(payload.get("summary", "")),
            source_memos=_as_str_tuple(payload.get("source_memos", ())),
        )


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    raise ValueError(f"expected a sequence of strings; got {type(value)!r}")


def constants_provenance_state_dir(repo_root: str | Path | None = None) -> Path:
    """Return the durable manifest directory ``.omx/state/constants_provenance``.

    NEVER ``/tmp`` per CLAUDE.md "Forbidden /tmp paths" — manifests are durable
    operator-facing evidence consumed by the Catalog #385 gate.
    """
    if repo_root is None:
        # This file lives at src/tac/substrates/_shared/; repo root is 4 up.
        repo_root = Path(__file__).resolve().parents[4]
    return Path(repo_root) / ".omx" / "state" / "constants_provenance"


def manifest_path_for_constants(
    vehicle_id: str, repo_root: str | Path | None = None
) -> Path:
    return constants_provenance_state_dir(repo_root) / f"{vehicle_id}.json"


def emit_constants_provenance_manifest(
    manifest: ConstantsProvenanceManifest,
    repo_root: str | Path | None = None,
    *,
    verify: bool = False,
) -> Path:
    """Write ``manifest`` as durable JSON; return the path.

    When ``verify=True`` the manifest is checked for the blocking condition BEFORE
    writing (so a vehicle with an unresolved ARBITRARY score-relevant constant at
    L2+ can never be emitted as if clean). A manifest with a known blocker is
    written un-verified by default so the Catalog #385 gate can SURFACE it.
    """
    if verify:
        manifest.verify()
    out_dir = constants_provenance_state_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest.vehicle_id}.json"
    out_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path
