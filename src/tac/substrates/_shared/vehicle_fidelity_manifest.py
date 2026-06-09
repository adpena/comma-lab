# SPDX-License-Identifier: MIT
"""Canonical anti-name-laundering manifest for NeRV-family carrier vehicles.

Source: operator NON-NEGOTIABLE directive 2026-06-09 (after the fidelity audits
`.omx/research/deep_hinerv_snerv_fidelity_review_vs_evaluate_py_20260609.md` +
`.omx/research/snerv_all_vehicles_fidelity_review_vs_evaluate_py_20260609.md`
discovered that our "HiNeRV" is a skip-free vanilla NeRV, our `sane_hnerv`
docstring advertises a "bilinear-skip" the code never implements, and the shared
MLX harness defaults SegNet/PoseNet distillation weights to 0.0 so "score-aware"
runs were secretly recon-MSE-only).

The bug class this module extincts is **name laundering** (Catalog #307
DOCUMENTATION-LEVEL fakery): a carrier calling itself ``HNeRV-like`` /
``HiNeRV-like`` / advertising a defining mechanism in its docstring while the
mechanism is absent from the actual forward pass. A reader trusting the label
or docstring believes the carrier has a high-frequency residual path it does
not have, and mis-attributes the d_seg≈0.5 mean-field collapse to training or
bytes instead of to the missing architecture.

This module is the typed, machine-checkable manifest that states, per vehicle,
the **CLAIMED family** vs the **ACTUAL mechanisms present**. ``verify()`` fails
closed when a docstring claims a mechanism the evidence list does not
substantiate — so no carrier can be called "HiNeRV-like" unless the mechanism
actually exists with file:line evidence (and, where the docstring claims it, a
test id proving the behavior).

Mechanism vocabulary (the canonical NeRV-family high-frequency + objective
ingredients per the two audit memos):

* ``bilinear_skip`` — per-block ``sin(PixelShuffle(conv(x)) + skip(bilinear_up(x)))``
  residual (PR95 reference ``model.py:46-51``). The single largest structural
  difference vs the decoder that won.
* ``terminal_refine`` — terminal ``x + 0.1*sin(refine(x))`` dilated-conv HF
  residual before the RGB heads (PR95 ``model.py:51``).
* ``grid_pe`` / ``positional_encoding`` — coordinate-indexed positional /
  multi-resolution feature-grid encoding (HiNeRV's defining feature; the
  literal mechanism that beats spectral bias).
* ``mfu_hfr_tub`` — SNeRV's official store-LF / generate-HF blocks
  (multi-resolution fusion unit + high-frequency restorer + temporal
  up-sampling block).
* ``codebook_vq`` — VQ-VAE nearest-codebook lookup + straight-through estimator
  + EMA codebook update + commitment loss (van den Oord 1711.00937).
* ``scorer_objective_weights_nonzero`` — the trained path optimizes a NONZERO
  frozen-SegNet/PoseNet objective (NOT recon-MSE-only with distillation weights
  defaulting to 0.0).
* ``residual_hf_path`` — ANY genuine residual / high-frequency synthesis path
  (the umbrella term; ``bilinear_skip`` OR ``terminal_refine`` OR ``mfu_hfr_tub``
  OR a nonlinear HF generator satisfy it).

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "Results must
become system intelligence": this manifest is a reusable typed surface, emitted
as durable JSON under ``.omx/state/vehicle_fidelity/`` (NEVER ``/tmp``), so the
solver / autopilot / next subagent inherits the claimed-vs-actual map rather
than re-deriving it from prose.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Catalog #287 placeholder discipline:
fields whose audit is genuinely incomplete carry the honest status string
``AUDIT_PENDING`` (a real status, not a fabricated value). ``verify()`` permits
``AUDIT_PENDING`` for present-mechanism evidence/tests but still fails closed on
the laundering condition (docstring-claims-X-but-X-absent), because an absent
mechanism is a known fact even when an unrelated present mechanism's test is
still being written.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "AUDIT_PENDING",
    "CANONICAL_MECHANISM_VOCABULARY",
    "MECHANISM_DOCSTRING_TRIGGERS",
    "MechanismEvidence",
    "VehicleFidelityManifest",
    "VehicleFidelityVerifyError",
    "default_state_dir",
    "emit_manifest",
    "manifest_path_for_vehicle",
]

# Honest "not-yet-audited" sentinel per Catalog #287 (placeholder rationale
# rejection): AUDIT_PENDING is a real, machine-checkable status — NOT a
# fabricated value like "TBD" / "<value>" / "pending_ratification".
AUDIT_PENDING = "audit_pending"

# The canonical NeRV-family mechanism vocabulary (booleans/evidence).
CANONICAL_MECHANISM_VOCABULARY: tuple[str, ...] = (
    "bilinear_skip",
    "terminal_refine",
    "grid_pe",
    "positional_encoding",
    "mfu_hfr_tub",
    "codebook_vq",
    "scorer_objective_weights_nonzero",
    "residual_hf_path",
)

# Docstring substrings that, if present in a carrier's prose, CLAIM the keyed
# mechanism. ``verify()`` uses this to detect the laundering condition: a
# docstring claim that the mechanism evidence list does not substantiate.
# Keys are mechanism names; values are case-insensitive substrings that count
# as an advertisement of that mechanism.
MECHANISM_DOCSTRING_TRIGGERS: Mapping[str, tuple[str, ...]] = {
    "bilinear_skip": ("bilinear-skip", "bilinear skip"),
    "terminal_refine": ("refine residual", "hf refine", "refine hf"),
    "grid_pe": ("grid positional", "feature grid", "feature-grid"),
    "positional_encoding": ("positional encoding", "positional-encoding"),
    "mfu_hfr_tub": ("mfu", "hfr", "tub"),
    "codebook_vq": ("codebook", "vector-quantiz", "vector quantiz", "vq-vae"),
    "scorer_objective_weights_nonzero": (
        "score-aware",
        "scorer-aware",
        "score aware",
    ),
    "residual_hf_path": ("residual high-frequency", "hf residual path"),
}


@dataclass(frozen=True)
class MechanismEvidence:
    """Evidence that a single mechanism is present in a carrier.

    A present mechanism MUST carry ``file:line`` evidence. Where the carrier's
    docstring CLAIMS the mechanism, ``test_id`` must name a test proving the
    behavior (NOT a constants test, per the NO FAKE Class-2 rule). When the
    audit of a present mechanism's test is genuinely incomplete, ``test_id`` may
    be :data:`AUDIT_PENDING` — an honest status, not a fabricated value.
    """

    mechanism: str
    """One of :data:`CANONICAL_MECHANISM_VOCABULARY`."""

    evidence: str
    """``relative/path.py:LINE`` (or ``LINE-LINE``) locating the implementation."""

    test_id: str = AUDIT_PENDING
    """Test id / path proving the mechanism behaves, or :data:`AUDIT_PENDING`."""

    notes: str = ""
    """Optional one-line mechanism note (e.g. ``OFF by default``)."""

    def __post_init__(self) -> None:
        if self.mechanism not in CANONICAL_MECHANISM_VOCABULARY:
            raise ValueError(
                f"MechanismEvidence.mechanism {self.mechanism!r} not in canonical "
                f"vocabulary {CANONICAL_MECHANISM_VOCABULARY}"
            )
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError(
                f"MechanismEvidence({self.mechanism!r}) requires non-empty "
                f"file:line evidence; got {self.evidence!r}"
            )
        # Catalog #287: reject placeholder evidence strings masquerading as real.
        lowered = self.evidence.strip().lower()
        if lowered in {"tbd", "<value>", "placeholder", "pending_ratification"}:
            raise ValueError(
                f"MechanismEvidence({self.mechanism!r}) evidence {self.evidence!r} "
                f"is a forbidden placeholder literal (Catalog #287). Use a real "
                f"file:line or the honest {AUDIT_PENDING!r} status on test_id."
            )

    def as_dict(self) -> dict[str, str]:
        return {
            "mechanism": self.mechanism,
            "evidence": self.evidence,
            "test_id": self.test_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MechanismEvidence:
        return cls(
            mechanism=str(payload["mechanism"]),
            evidence=str(payload["evidence"]),
            test_id=str(payload.get("test_id", AUDIT_PENDING)),
            notes=str(payload.get("notes", "")),
        )


class VehicleFidelityVerifyError(ValueError):
    """Raised by :meth:`VehicleFidelityManifest.verify` on a laundering finding."""


@dataclass(frozen=True)
class VehicleFidelityManifest:
    """Typed claimed-vs-actual mechanism manifest for one NeRV-family vehicle.

    Schema (per operator 2026-06-09):

    * ``vehicle_id`` — canonical substrate dir name (e.g. ``hi_nerv``).
    * ``claimed_family`` — the family the carrier's NAME/docstring advertises
      (e.g. ``HiNeRV``, ``HNeRV-LC-v2``, ``SNeRV``, ``VQ-VAE NeRV``).
    * ``actual_mechanisms_present`` — list of :class:`MechanismEvidence`, each
      with file:line (and a test id where the docstring claims it).
    * ``mechanisms_absent`` — EXPLICIT list of canonical-vocabulary mechanisms
      the carrier does NOT implement (the honest negative; silence is the
      laundering failure mode).
    * ``docstring_claims`` — verbatim docstring phrases that advertise a family
      or a mechanism. Used by :meth:`verify` to detect the laundering condition.
    * ``tests_proving_mechanism`` — test ids/paths backing present mechanisms.
    """

    vehicle_id: str
    claimed_family: str
    actual_mechanisms_present: tuple[MechanismEvidence, ...] = ()
    mechanisms_absent: tuple[str, ...] = ()
    docstring_claims: tuple[str, ...] = ()
    tests_proving_mechanism: tuple[str, ...] = ()
    summary: str = ""
    """One-line claimed-vs-actual headline for operator dashboards."""

    source_memos: tuple[str, ...] = field(default_factory=tuple)
    """The audit memo paths this manifest was populated from."""

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, str) or not self.vehicle_id.strip():
            raise ValueError("vehicle_id must be a non-empty string")
        if not isinstance(self.claimed_family, str) or not self.claimed_family.strip():
            raise ValueError("claimed_family must be a non-empty string")
        # Every absent mechanism must be in the canonical vocabulary.
        for absent in self.mechanisms_absent:
            if absent not in CANONICAL_MECHANISM_VOCABULARY:
                raise ValueError(
                    f"mechanisms_absent entry {absent!r} not in canonical "
                    f"vocabulary {CANONICAL_MECHANISM_VOCABULARY}"
                )
        # A mechanism cannot be both present and absent.
        present_names = {m.mechanism for m in self.actual_mechanisms_present}
        overlap = present_names & set(self.mechanisms_absent)
        if overlap:
            raise ValueError(
                f"{self.vehicle_id}: mechanism(s) {sorted(overlap)} listed as BOTH "
                f"present and absent — manifest is internally inconsistent."
            )

    # -- query helpers -----------------------------------------------------

    def present_mechanism_names(self) -> frozenset[str]:
        return frozenset(m.mechanism for m in self.actual_mechanisms_present)

    def evidence_for(self, mechanism: str) -> MechanismEvidence | None:
        for m in self.actual_mechanisms_present:
            if m.mechanism == mechanism:
                return m
        return None

    def claims_mechanism(self, mechanism: str) -> tuple[str, ...]:
        """Return the docstring_claims phrases that advertise ``mechanism``.

        A mechanism is CLAIMED when any ``docstring_claims`` phrase contains one
        of the mechanism's :data:`MECHANISM_DOCSTRING_TRIGGERS` substrings
        (case-insensitive).
        """
        triggers = MECHANISM_DOCSTRING_TRIGGERS.get(mechanism, ())
        if not triggers:
            return ()
        hits: list[str] = []
        for claim in self.docstring_claims:
            low = claim.lower()
            if any(trig in low for trig in triggers):
                hits.append(claim)
        return tuple(hits)

    # -- the fail-closed laundering check ----------------------------------

    def laundering_findings(self) -> tuple[str, ...]:
        """Return the name-laundering findings (empty tuple = clean).

        A finding is the carrier's ``docstring_claims`` advertising a mechanism
        that is NOT substantiated by ``actual_mechanisms_present``. Two cases:

        * **claimed-but-absent** — the mechanism is in ``mechanisms_absent`` yet
          the docstring advertises it (the ``sane_hnerv`` "bilinear-skip"
          documentation-fake; the canonical laundering case).
        * **claimed-but-silent** — the docstring advertises a mechanism that is
          neither present nor explicitly declared absent. Silence on a claimed
          mechanism is itself a laundering failure mode (the reader cannot tell
          whether the claim is true).

        This is the pure laundering signal: a FAITHFUL carrier that genuinely
        implements a claimed mechanism (e.g. SNeRV's real MFU/HFR/TUB) produces
        ZERO findings even when its behavior test is still ``AUDIT_PENDING``.
        The test-coverage gap is a SEPARATE, softer advisory (see
        :meth:`unproven_claims`) so it does not dilute the laundering signal.
        """
        present = self.present_mechanism_names()
        absent = set(self.mechanisms_absent)
        findings: list[str] = []

        for mechanism in CANONICAL_MECHANISM_VOCABULARY:
            claim_phrases = self.claims_mechanism(mechanism)
            if not claim_phrases:
                continue
            if mechanism in present:
                continue  # claimed AND substantiated -> not laundering
            if mechanism in absent:
                findings.append(
                    f"NAME-LAUNDERING: docstring claims {mechanism!r} "
                    f"(phrase(s): {list(claim_phrases)}) but it is in "
                    f"mechanisms_absent — the carrier advertises a mechanism it "
                    f"does NOT implement (Catalog #307 documentation-fake)."
                )
            else:
                findings.append(
                    f"NAME-LAUNDERING: docstring claims {mechanism!r} "
                    f"(phrase(s): {list(claim_phrases)}) but it is neither in "
                    f"actual_mechanisms_present nor mechanisms_absent — a claimed "
                    f"mechanism must be explicitly substantiated or explicitly "
                    f"declared absent (silence is the laundering failure mode)."
                )
        return tuple(findings)

    def unproven_claims(self) -> tuple[str, ...]:
        """Return advisories for claimed-AND-present mechanisms lacking a test.

        A docstring-asserted mechanism that IS present (so NOT laundering) but
        whose ``test_id`` is :data:`AUDIT_PENDING` carries a softer NO-FAKE
        Class-2 advisory: the claim is true but its behavior is not yet
        regression-guarded. This is intentionally SEPARATE from
        :meth:`laundering_findings` / :meth:`verify` so an in-flight audit
        (honest ``AUDIT_PENDING``) does not get flagged like a documentation
        fake. Empty tuple = every present-and-claimed mechanism has a test.
        """
        advisories: list[str] = []
        for mechanism in CANONICAL_MECHANISM_VOCABULARY:
            if not self.claims_mechanism(mechanism):
                continue
            ev = self.evidence_for(mechanism)
            if ev is None:
                continue
            if ev.test_id.strip().lower() == AUDIT_PENDING:
                advisories.append(
                    f"UNPROVEN-CLAIM (advisory): docstring claims {mechanism!r} "
                    f"and it IS present at {ev.evidence}, but test_id is "
                    f"{AUDIT_PENDING!r}. The claim is true but unguarded — write "
                    f"the behavior test to close the NO-FAKE Class-2 gap."
                )
        return tuple(advisories)

    def verify(self) -> None:
        """Fail closed on the name-laundering bug class.

        Raises :class:`VehicleFidelityVerifyError` when
        :meth:`laundering_findings` is non-empty — i.e. when the carrier's
        ``docstring_claims`` advertise a mechanism that the evidence list does
        not substantiate (the ``sane_hnerv`` "bilinear-skip" documentation-fake;
        the "named HiNeRV but isn't" pattern). A faithful carrier verifies even
        when its behavior tests are still ``AUDIT_PENDING`` (that is a separate
        :meth:`unproven_claims` advisory, not a fail-closed condition).
        """
        findings = self.laundering_findings()
        if findings:
            raise VehicleFidelityVerifyError(
                f"vehicle_fidelity_manifest.verify() failed for "
                f"{self.vehicle_id!r} (claimed_family={self.claimed_family!r}):\n  "
                + "\n  ".join(findings)
            )

    # -- serialization -----------------------------------------------------

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "vehicle_fidelity_manifest.v1",
            "vehicle_id": self.vehicle_id,
            "claimed_family": self.claimed_family,
            "actual_mechanisms_present": [
                m.as_dict() for m in self.actual_mechanisms_present
            ],
            "mechanisms_absent": list(self.mechanisms_absent),
            "docstring_claims": list(self.docstring_claims),
            "tests_proving_mechanism": list(self.tests_proving_mechanism),
            "summary": self.summary,
            "source_memos": list(self.source_memos),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> VehicleFidelityManifest:
        schema = str(payload.get("schema", ""))
        if schema and schema != "vehicle_fidelity_manifest.v1":
            raise ValueError(
                f"unexpected schema {schema!r}; expected "
                f"'vehicle_fidelity_manifest.v1'"
            )
        present_raw = payload.get("actual_mechanisms_present", []) or []
        if not isinstance(present_raw, Iterable):
            raise ValueError("actual_mechanisms_present must be a list")
        present = tuple(
            MechanismEvidence.from_dict(p)  # type: ignore[arg-type]
            for p in present_raw
        )
        return cls(
            vehicle_id=str(payload["vehicle_id"]),
            claimed_family=str(payload["claimed_family"]),
            actual_mechanisms_present=present,
            mechanisms_absent=_as_str_tuple(payload.get("mechanisms_absent", ())),
            docstring_claims=_as_str_tuple(payload.get("docstring_claims", ())),
            tests_proving_mechanism=_as_str_tuple(
                payload.get("tests_proving_mechanism", ())
            ),
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


def default_state_dir(repo_root: str | Path | None = None) -> Path:
    """Return the durable manifest directory ``.omx/state/vehicle_fidelity``.

    NEVER ``/tmp`` per CLAUDE.md "Forbidden /tmp paths" — manifests are durable
    operator-facing evidence consumed by the Catalog #384 gate.
    """
    if repo_root is None:
        # This file lives at src/tac/substrates/_shared/; repo root is 5 up.
        repo_root = Path(__file__).resolve().parents[4]
    return Path(repo_root) / ".omx" / "state" / "vehicle_fidelity"


def manifest_path_for_vehicle(
    vehicle_id: str, repo_root: str | Path | None = None
) -> Path:
    return default_state_dir(repo_root) / f"{vehicle_id}.json"


def emit_manifest(
    manifest: VehicleFidelityManifest,
    repo_root: str | Path | None = None,
    *,
    verify: bool = False,
) -> Path:
    """Write ``manifest`` as durable JSON; return the path.

    When ``verify=True`` the manifest is checked for the laundering condition
    BEFORE writing (so a launderer can never be emitted as if it were clean).
    A faithful manifest with a documentation-fake is written un-verified by
    default so the Catalog #384 gate can SURFACE it; pass ``verify=True`` only
    for manifests asserted to be clean.
    """
    if verify:
        manifest.verify()
    out_dir = default_state_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest.vehicle_id}.json"
    out_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path
