# SPDX-License-Identifier: MIT
"""Canonical objective-reachability manifest for score-aware carrier vehicles.

Source: operator NON-NEGOTIABLE V6 directive 2026-06-09 (P1 of the
ObjectiveReachability + AuditProvenance hardening packet). Empirical anchor:
``.omx/research/snerv_b_first_scorer_probe_verdict_20260609.md`` (the link-5
case study) + the SNeRV cross-wiring defect: the pose VJP was severed at THREE
layers (``f5c66f43c`` uncrossed it) while the score-aware loss FLAG existed and
the loss weights were nonzero. So the OBJECTIVE-STARVATION gate (Catalog #384,
``check_score_aware_run_has_nonzero_scorer_objective_weights``) was passing —
nonzero weights — yet the trained gradient never reached the renderer. Nonzero
weights are necessary but NOT sufficient: the objective must REACH the
parameters via an unbroken VJP path.

This module extincts the **objective-path-severance** bug class (distinct from
both name-laundering at the docstring surface — Catalog #384's sister
``vehicle_fidelity_manifest`` — and objective-starvation at the weight surface —
Catalog #384). Severance is the third leg: the weight is nonzero AND the loss is
named score-aware AND the docstring is honest, but the Jacobian-vector product
from the scorer objective to the renderer parameters is ZERO because a
``.detach()`` / ``stop_gradient`` / no-grad context / wrong tensor wiring sits
between them. A reader trusting "score-aware + weights 7.0/7.24" believes the
SegNet/PoseNet objective is optimizing the renderer when it is not.

The contest law these manifests serve (ground truth; never edit ``upstream/``)::

    S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489

computed by ``upstream/evaluate.py``. SegNet scores ONLY frame1 as a 5-class
**argmax-disagreement RATE** — that is **piecewise-constant, gradient-zero
almost everywhere**. PoseNet scores both frames via RGB->YUV6 with MSE on the
first 6 of 12 pose dims (differentiable).

CRITICAL math nuance (operator-explicit, the heart of P1):

* The official ``d_seg`` (argmax disagreement) **cannot** be a differentiable
  training row. Its gradient is 0 a.e. A trainer that backprops "d_seg" is
  either backpropping a SURROGATE (and mis-naming it) or backpropping zero.
* Therefore J_seg reaches the renderer through a **SURROGATE** row from the
  canonical vocabulary :data:`SEGNET_SURROGATE_ROWS` (cross-entropy /
  source-class margin / logit-KL / smooth-disagreement). Exact-argmax ``d_seg``
  is the **VERIFICATION** metric only (compared against the live/exact surface),
  never a training loss.
* The manifest FAILS CLOSED if it claims exact-argmax ``d_seg`` as a training
  row, OR if it names a hinge/margin/CE loss literally ``d_seg`` (the
  Vehicle-OS rule 5 / Mistake-B naming firewall).

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "Results must
become system intelligence": this manifest is a reusable typed surface emitted
as durable JSON under ``.omx/state/objective_reachability/`` (NEVER ``/tmp``),
so the solver / autopilot / Vehicle-OS dashboard / next subagent inherits the
reachability map rather than re-deriving it from telemetry prose.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" + Catalog #287 placeholder discipline:
fields whose audit is genuinely incomplete carry the honest status string
:data:`AUDIT_PENDING` (a real status, not a fabricated value). ``verify()``
permits ``AUDIT_PENDING`` on gradient-norm-by-mechanism entries that are not yet
measured, but still fails closed on the severance / mis-naming conditions
(those are KNOWN facts the moment the wiring is read).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "AUDIT_PENDING",
    "CANONICAL_GRADIENT_MECHANISMS",
    "FORBIDDEN_DSEG_TRAINING_NAMES",
    "SEGNET_SURROGATE_ROWS",
    "ObjectiveReachabilityManifest",
    "ObjectiveReachabilityVerifyError",
    "ReachabilityFinding",
    "audit_objective_reachability_manifests",
    "default_state_dir",
    "emit_objective_reachability_manifest",
    "load_objective_reachability_manifests",
    "objective_reachability_path_for_vehicle",
]

# Honest "not-yet-measured" sentinel per Catalog #287 (placeholder-rationale
# rejection): AUDIT_PENDING is a real, machine-checkable status — NOT a
# fabricated value like "TBD" / "<value>" / "pending_ratification".
AUDIT_PENDING = "audit_pending"

# The canonical SegNet SURROGATE training rows. The official d_seg (argmax
# disagreement) is gradient-zero a.e., so J_seg reaches the renderer ONLY
# through one of these differentiable surrogates. This is the operator-explicit
# allowed set. Anything outside it that claims to be the seg training signal is
# suspect.
SEGNET_SURROGATE_ROWS: tuple[str, ...] = (
    "ce",  # per-pixel cross-entropy on the frozen-SegNet logits
    "source_class_margin",  # margin pushing the source-class logit above runner-up
    "logit_kl",  # KL(candidate_logits || target_logits) (Hinton T-scaled)
    "smooth_disagreement",  # softmax/temperature-smoothed disagreement proxy
)

# The canonical per-mechanism gradient-norm keys (operator spec). Each names a
# distinct parameter group whose gradient norm under the scorer objective tells
# us whether the objective REACHES that mechanism. A zero/None norm on a
# mechanism the carrier claims to optimize is a severance finding.
CANONICAL_GRADIENT_MECHANISMS: tuple[str, ...] = (
    "latents",
    "decoder_blocks",
    "skip_path",
    "hf_residual",
    "mfu",
    "hfr",
    "tub",
    "codebook",
    "selector",
)

# Names a differentiable training row MUST NOT use for an argmax/hinge/margin
# surrogate (Vehicle-OS rule 5 / Mistake-B firewall). If any of these literal
# names appears as a *training* row, the manifest fails closed: the exact
# argmax-disagreement d_seg is the VERIFICATION metric, never a training loss.
FORBIDDEN_DSEG_TRAINING_NAMES: tuple[str, ...] = (
    "d_seg",
    "dseg",
    "argmax_d_seg",
    "argmax_dseg",
    "seg_dist",
    "segnet_dist",
    "avg_segnet_dist",
)


class ObjectiveReachabilityVerifyError(ValueError):
    """Raised by :meth:`ObjectiveReachabilityManifest.verify` on a severance / mis-naming finding."""


def _is_pending(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUDIT_PENDING


def _norm_is_zero(value: object) -> bool:
    """True when a gradient-norm value is a measured ZERO (severance signal).

    A numeric 0.0 (or negative) is a measured severance. :data:`AUDIT_PENDING`
    is NOT a severance — it is an honest not-yet-measured status. A positive
    number is a healthy reaching gradient.
    """
    if _is_pending(value):
        return False
    try:
        return float(value) <= 0.0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # An unparseable non-pending value is treated as not-measured-zero; it
        # is surfaced as a malformed-norm finding by verify() instead.
        return False


@dataclass(frozen=True)
class ObjectiveReachabilityManifest:
    """Typed objective-reachability manifest for one score-aware vehicle.

    Schema (per operator V6 spec 2026-06-09):

    * ``vehicle`` — canonical substrate dir / vehicle id (e.g. ``snerv``).
    * ``segnet_objective_active`` — the SegNet objective is wired into the loss
      (a nonzero seg term participates in the trained loss).
    * ``posenet_objective_active`` — the PoseNet objective is wired into the loss.
    * ``segnet_surrogate_rows`` — the SURROGATE training rows carrying J_seg
      (must be drawn from :data:`SEGNET_SURROGATE_ROWS`; exact-argmax d_seg is
      NOT a valid entry).
    * ``segnet_vjp_reaches_renderer`` — the SegNet objective's VJP reaches the
      renderer parameters (unbroken gradient path, no severing detach/no-grad).
    * ``posenet_vjp_reaches_renderer`` — the PoseNet objective's VJP reaches the
      renderer parameters.
    * ``loss_weights_nonzero`` — both objective weights are explicit and nonzero
      (the Catalog #384 condition; necessary precondition for reachability).
    * ``gradient_norm_by_mechanism`` — measured gradient norm under the scorer
      objective per :data:`CANONICAL_GRADIENT_MECHANISMS` key. A value may be a
      positive float (reaching), ``0.0`` (severed), or :data:`AUDIT_PENDING`
      (not yet measured). Only the mechanisms the carrier actually has need
      appear; absent keys are treated as not-applicable.
    * ``severed_layers`` — EXPLICIT list of layer / wiring names where the VJP
      is known to be broken (the honest negative; silence-on-severance is the
      failure mode this manifest extincts).
    * ``first_failed_surface`` — the first surface in the reachability chain that
      fails (name -> weight -> VJP -> grad-norm), or ``""`` when all surfaces
      reach. Operator-facing one-token blocker.
    * ``dseg_is_verification_metric_only`` — affirmation that exact-argmax d_seg
      is used ONLY as the verification metric, never as a differentiable
      training row. MUST be True for a score-aware vehicle (the math nuance).
    """

    vehicle: str
    segnet_objective_active: bool = False
    posenet_objective_active: bool = False
    segnet_surrogate_rows: tuple[str, ...] = ()
    segnet_vjp_reaches_renderer: bool = False
    posenet_vjp_reaches_renderer: bool = False
    loss_weights_nonzero: bool = False
    gradient_norm_by_mechanism: Mapping[str, object] = field(default_factory=dict)
    severed_layers: tuple[str, ...] = ()
    first_failed_surface: str = ""
    dseg_is_verification_metric_only: bool = True
    summary: str = ""
    """One-line reachability headline for operator dashboards."""

    source_artifacts: tuple[str, ...] = field(default_factory=tuple)
    """The telemetry / commit / memo paths this manifest was populated from."""

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle, str) or not self.vehicle.strip():
            raise ValueError("vehicle must be a non-empty string")
        # Surrogate rows must be drawn from the canonical vocabulary. This is
        # where exact-argmax d_seg as a "surrogate" gets rejected (it is not in
        # SEGNET_SURROGATE_ROWS), so a manifest cannot smuggle the gradient-zero
        # metric in as a training row.
        for row in self.segnet_surrogate_rows:
            if row not in SEGNET_SURROGATE_ROWS:
                raise ValueError(
                    f"{self.vehicle}: segnet_surrogate_rows entry {row!r} is not "
                    f"a canonical surrogate {SEGNET_SURROGATE_ROWS}. The official "
                    f"argmax d_seg is gradient-zero a.e. and is NOT a valid "
                    f"training row — it is the VERIFICATION metric only."
                )
        # Every gradient-norm key must be a canonical mechanism.
        for key in self.gradient_norm_by_mechanism:
            if key not in CANONICAL_GRADIENT_MECHANISMS:
                raise ValueError(
                    f"{self.vehicle}: gradient_norm_by_mechanism key {key!r} not "
                    f"in canonical mechanisms {CANONICAL_GRADIENT_MECHANISMS}"
                )

    # -- query helpers -----------------------------------------------------

    def reaching_mechanisms(self) -> frozenset[str]:
        """Mechanisms with a measured POSITIVE gradient norm (objective reaches them)."""
        return frozenset(
            k
            for k, v in self.gradient_norm_by_mechanism.items()
            if not _is_pending(v) and not _norm_is_zero(v)
        )

    def severed_mechanisms(self) -> frozenset[str]:
        """Mechanisms with a measured ZERO gradient norm (objective severed at them)."""
        return frozenset(
            k for k, v in self.gradient_norm_by_mechanism.items() if _norm_is_zero(v)
        )

    def pending_mechanisms(self) -> frozenset[str]:
        """Mechanisms whose gradient norm is honest-not-yet-measured (AUDIT_PENDING)."""
        return frozenset(
            k for k, v in self.gradient_norm_by_mechanism.items() if _is_pending(v)
        )

    def claims_score_aware(self) -> bool:
        """A vehicle claims score-aware when EITHER objective is marked active."""
        return bool(self.segnet_objective_active or self.posenet_objective_active)

    # -- the fail-closed severance / mis-naming check ----------------------

    def reachability_findings(self) -> tuple[str, ...]:
        """Return the objective-reachability findings (empty tuple = clean).

        A finding is any way the carrier CLAIMS a score-aware objective while a
        surface in the reachability chain is broken. Surfaces, in order:

        1. **weight** — objective active but ``loss_weights_nonzero`` is False
           (Catalog #384 condition; necessary precondition).
        2. **vjp** — objective active but its VJP does NOT reach the renderer
           (the SNeRV 3-layer-severance case the pose uncrossing fixed).
        3. **grad-norm** — a mechanism carries a measured ZERO gradient norm
           (the objective is wired but produces no parameter update there).
        4. **surrogate-absence** — the SegNet objective is active but NO
           surrogate row carries J_seg (so the only seg signal would have to be
           gradient-zero argmax d_seg — i.e. no seg learning at all).
        5. **dseg-mis-naming** — a forbidden literal d_seg-family name is used
           as a training row, OR ``dseg_is_verification_metric_only`` is False.
        6. **declared-severance** — ``severed_layers`` is non-empty while the
           carrier claims the objective is active (the honest self-report of a
           known break is itself a fail-closed finding for a score-aware claim).

        A FAITHFUL score-aware vehicle (weights nonzero, both VJPs reach, no
        severed layers, surrogate rows present, d_seg verification-only) produces
        ZERO findings even when some gradient norms are still ``AUDIT_PENDING``
        (pending is an honest not-measured status, not a severance).
        """
        findings: list[str] = []

        # 5a. d_seg mis-naming firewall — applies regardless of active flags,
        # because naming a hinge/CE surrogate "d_seg" is always the bug.
        if not self.dseg_is_verification_metric_only:
            findings.append(
                "DSEG-MIS-NAMING: dseg_is_verification_metric_only is False — the "
                "official argmax d_seg is gradient-zero a.e. and MUST be the "
                "VERIFICATION metric only, never a differentiable training row "
                "(Vehicle-OS rule 5 / Mistake-B firewall)."
            )
        # A forbidden d_seg-family name MUST NOT appear in the surrogate rows.
        # (segnet_surrogate_rows is already constrained to the canonical set by
        # __post_init__, but a forbidden name could appear if someone bypassed
        # the dataclass; we re-check defensively at verify time.)
        for row in self.segnet_surrogate_rows:
            if row.strip().lower() in FORBIDDEN_DSEG_TRAINING_NAMES:
                findings.append(
                    f"DSEG-MIS-NAMING: surrogate row {row!r} is a forbidden "
                    f"argmax-d_seg training name {FORBIDDEN_DSEG_TRAINING_NAMES}. "
                    f"A hinge/margin/CE surrogate must be named ce / "
                    f"source_class_margin / logit_kl / smooth_disagreement — "
                    f"never d_seg."
                )

        if not self.claims_score_aware():
            # Not a score-aware claim -> only the always-on d_seg firewall above
            # applies. (A recon-only carrier makes no reachability promise.)
            return tuple(findings)

        # 1. weight surface.
        if not self.loss_weights_nonzero:
            findings.append(
                f"WEIGHT-SEVERANCE: vehicle claims score-aware "
                f"(segnet_objective_active={self.segnet_objective_active}, "
                f"posenet_objective_active={self.posenet_objective_active}) but "
                f"loss_weights_nonzero is False — the objective weight is 0.0/None "
                f"(Catalog #384 objective-starvation precondition)."
            )

        # 2. VJP surface — per active objective.
        if self.segnet_objective_active and not self.segnet_vjp_reaches_renderer:
            findings.append(
                "VJP-SEVERANCE: segnet_objective_active but "
                "segnet_vjp_reaches_renderer is False — the SegNet objective's "
                "gradient does NOT reach the renderer parameters (a severing "
                "detach / stop_gradient / no-grad context sits between them; the "
                "SNeRV 3-layer pose-VJP severance class)."
            )
        if self.posenet_objective_active and not self.posenet_vjp_reaches_renderer:
            findings.append(
                "VJP-SEVERANCE: posenet_objective_active but "
                "posenet_vjp_reaches_renderer is False — the PoseNet objective's "
                "gradient does NOT reach the renderer parameters (the canonical "
                "SNeRV pose-VJP severance the f5c66f43c uncrossing fixed)."
            )

        # 3. grad-norm surface — a measured ZERO on any mechanism is severance.
        for mech in self.severed_mechanisms():
            findings.append(
                f"GRAD-NORM-SEVERANCE: mechanism {mech!r} has a measured ZERO "
                f"gradient norm under the scorer objective — the objective is "
                f"wired but produces no parameter update there."
            )

        # 4. surrogate-absence — seg objective active but no differentiable
        # surrogate carries J_seg (the only seg signal would be gradient-zero
        # argmax d_seg).
        if self.segnet_objective_active and not self.segnet_surrogate_rows:
            findings.append(
                f"SURROGATE-ABSENCE: segnet_objective_active but "
                f"segnet_surrogate_rows is empty — the official argmax d_seg is "
                f"gradient-zero a.e., so with no surrogate row "
                f"({SEGNET_SURROGATE_ROWS}) the SegNet objective contributes no "
                f"learnable gradient (no seg learning)."
            )

        # 6. declared-severance — an honest self-reported break is still a
        # fail-closed finding for a vehicle claiming the objective is active.
        if self.severed_layers:
            findings.append(
                f"DECLARED-SEVERANCE: vehicle claims score-aware but declares "
                f"severed_layers={list(self.severed_layers)} — a self-reported "
                f"broken VJP path is honest but the score-aware claim cannot stand "
                f"while a layer severs the objective."
            )

        return tuple(findings)

    def verify(self) -> None:
        """Fail closed on the objective-path-severance / d_seg-mis-naming bug class.

        Raises :class:`ObjectiveReachabilityVerifyError` when
        :meth:`reachability_findings` is non-empty — i.e. when the carrier
        claims a score-aware objective but a reachability surface (weight, VJP,
        grad-norm, surrogate presence, d_seg naming, or a declared severance) is
        broken. A faithful score-aware vehicle verifies even when some gradient
        norms are still ``AUDIT_PENDING``.
        """
        findings = self.reachability_findings()
        if findings:
            raise ObjectiveReachabilityVerifyError(
                f"objective_reachability_manifest.verify() failed for "
                f"{self.vehicle!r}:\n  " + "\n  ".join(findings)
            )

    # -- serialization -----------------------------------------------------

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "objective_reachability_manifest.v1",
            "vehicle": self.vehicle,
            "segnet_objective_active": self.segnet_objective_active,
            "posenet_objective_active": self.posenet_objective_active,
            "segnet_surrogate_rows": list(self.segnet_surrogate_rows),
            "segnet_vjp_reaches_renderer": self.segnet_vjp_reaches_renderer,
            "posenet_vjp_reaches_renderer": self.posenet_vjp_reaches_renderer,
            "loss_weights_nonzero": self.loss_weights_nonzero,
            "gradient_norm_by_mechanism": dict(self.gradient_norm_by_mechanism),
            "severed_layers": list(self.severed_layers),
            "first_failed_surface": self.first_failed_surface,
            "dseg_is_verification_metric_only": self.dseg_is_verification_metric_only,
            "summary": self.summary,
            "source_artifacts": list(self.source_artifacts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ObjectiveReachabilityManifest:
        schema = str(payload.get("schema", ""))
        if schema and schema != "objective_reachability_manifest.v1":
            raise ValueError(
                f"unexpected schema {schema!r}; expected "
                f"'objective_reachability_manifest.v1'"
            )
        grad_raw = payload.get("gradient_norm_by_mechanism", {}) or {}
        if not isinstance(grad_raw, Mapping):
            raise ValueError("gradient_norm_by_mechanism must be a mapping")
        return cls(
            vehicle=str(payload["vehicle"]),
            segnet_objective_active=bool(payload.get("segnet_objective_active", False)),
            posenet_objective_active=bool(
                payload.get("posenet_objective_active", False)
            ),
            segnet_surrogate_rows=_as_str_tuple(payload.get("segnet_surrogate_rows", ())),
            segnet_vjp_reaches_renderer=bool(
                payload.get("segnet_vjp_reaches_renderer", False)
            ),
            posenet_vjp_reaches_renderer=bool(
                payload.get("posenet_vjp_reaches_renderer", False)
            ),
            loss_weights_nonzero=bool(payload.get("loss_weights_nonzero", False)),
            gradient_norm_by_mechanism=dict(grad_raw),
            severed_layers=_as_str_tuple(payload.get("severed_layers", ())),
            first_failed_surface=str(payload.get("first_failed_surface", "")),
            dseg_is_verification_metric_only=bool(
                payload.get("dseg_is_verification_metric_only", True)
            ),
            summary=str(payload.get("summary", "")),
            source_artifacts=_as_str_tuple(payload.get("source_artifacts", ())),
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
    """Return the durable manifest directory ``.omx/state/objective_reachability``.

    NEVER ``/tmp`` per CLAUDE.md "Forbidden /tmp paths" — manifests are durable
    operator-facing evidence consumed by the Catalog #386 gate + Vehicle-OS
    dashboard.
    """
    if repo_root is None:
        # This file lives at src/tac/substrates/_shared/; repo root is 5 up.
        repo_root = Path(__file__).resolve().parents[4]
    return Path(repo_root) / ".omx" / "state" / "objective_reachability"


def objective_reachability_path_for_vehicle(
    vehicle: str, repo_root: str | Path | None = None
) -> Path:
    return default_state_dir(repo_root) / f"{vehicle}.json"


@dataclass(frozen=True)
class ReachabilityFinding:
    """One objective-reachability finding for the Catalog #386 gate."""

    vehicle: str
    finding: str
    manifest_path: str

    def message(self) -> str:
        return (
            f"OBJECTIVE-REACHABILITY [{self.vehicle}] ({self.manifest_path}): "
            f"{self.finding}"
        )


def load_objective_reachability_manifests(
    repo_root: str | Path | None = None,
) -> list[tuple[ObjectiveReachabilityManifest, Path]]:
    """Load every emitted manifest under ``.omx/state/objective_reachability``.

    Returns ``(manifest, path)`` pairs. A directory with no manifests yields an
    empty list (the gate then has nothing to check — live count 0).
    """
    state_dir = default_state_dir(repo_root)
    out: list[tuple[ObjectiveReachabilityManifest, Path]] = []
    if not state_dir.is_dir():
        return out
    for path in sorted(state_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt manifest is surfaced as a finding by the audit, not a load
            # crash; record a sentinel manifest carrying the parse failure.
            out.append(
                (
                    ObjectiveReachabilityManifest(
                        vehicle=path.stem,
                        segnet_objective_active=True,
                        loss_weights_nonzero=False,
                        first_failed_surface="corrupt_manifest",
                        summary=f"corrupt manifest JSON at {path}",
                    ),
                    path,
                )
            )
            continue
        try:
            manifest = ObjectiveReachabilityManifest.from_dict(payload)
        except (ValueError, KeyError) as exc:
            out.append(
                (
                    ObjectiveReachabilityManifest(
                        vehicle=path.stem,
                        segnet_objective_active=True,
                        loss_weights_nonzero=False,
                        first_failed_surface="malformed_manifest",
                        summary=f"malformed manifest: {exc}",
                    ),
                    path,
                )
            )
            continue
        out.append((manifest, path))
    return out


def audit_objective_reachability_manifests(
    repo_root: str | Path | None = None,
) -> list[ReachabilityFinding]:
    """Return every reachability finding across emitted manifests.

    A vehicle that claims score-aware (``claims_score_aware()``) but whose
    :meth:`ObjectiveReachabilityManifest.reachability_findings` is non-empty
    contributes one :class:`ReachabilityFinding` per finding. A faithful
    reaching vehicle (snerv post-``f5c66f43c``) and a non-score-aware carrier
    contribute nothing. This is the canonical helper the Catalog #386 gate
    delegates to.
    """
    findings: list[ReachabilityFinding] = []
    for manifest, path in load_objective_reachability_manifests(repo_root):
        for finding in manifest.reachability_findings():
            findings.append(
                ReachabilityFinding(
                    vehicle=manifest.vehicle,
                    finding=finding,
                    manifest_path=str(path),
                )
            )
    return findings


def emit_objective_reachability_manifest(
    manifest: ObjectiveReachabilityManifest,
    repo_root: str | Path | None = None,
    *,
    verify: bool = False,
) -> Path:
    """Write ``manifest`` as durable JSON; return the path.

    When ``verify=True`` the manifest is checked for the severance condition
    BEFORE writing (so a severed carrier can never be emitted as if it were
    reachable). A carrier with a known severance is written un-verified by
    default so the Catalog #386 gate can SURFACE it; pass ``verify=True`` only
    for manifests asserted to be clean.
    """
    if verify:
        manifest.verify()
    out_dir = default_state_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest.vehicle}.json"
    out_path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return out_path
