"""tac.witness_dsl.typed_config — the TYPED authoring surface for a launch config.

Operator binding 2026-07-08 (verbatim): "The config must be defined in the DSL —
no ad hoc or hand crafting. We may need to use pydantic and possibly verification
and validation tools as well and type checking and formalization and integrate all
with apparatus to prevent more dumbass bullshit."

WHY this module exists (the confound it extincts): ``witness_autoconfig`` variants
(``derive_crucible_v6_config`` + siblings) hand-assemble a config "matching the
single DSL emitter" instead of BEING a ``curriculum_dsl.WitnessProgram``. Every
parallel, untyped dict/argv-assembly path is where the PR95 skeleton, hardcoded
epochs, and bare constants re-entered silently — untyped assembly cannot be
validated, so violations shipped. This module is the TYPED, VALIDATED, single
authoring surface: a pydantic v2 model whose every field carries a type + range +
unit + a VALUE-PROVENANCE class (the ladder), refuses unknown keys
(``extra="forbid"`` — the anti-invented-flag law at the schema layer), and
CONSTRUCTS a ``WitnessProgram`` via the adapter :meth:`TypedWitnessConfig.to_program`.

DESIGN CONTRACT (do NOT rewrite ``WitnessProgram`` — huge blast radius):
  * The typed layer VALIDATES + CONSTRUCTS ``WitnessProgram`` instances; the DSL's
    ``compile_trainer_argv`` / ``validate`` internals are UNTOUCHED and keep working.
  * Every value knob is a :class:`Provenanced` wrapper: value + ladder class + unit
    + source + (waiver required iff HARDCODED). A bare literal with no ladder class
    is refused BY CONSTRUCTION.
  * The ``schedule_governance`` block (the schedule-provenance gate's declaration
    surface) is a FIRST-CLASS typed field (:class:`ScheduleGovernance`) so a
    malformed governance declaration is a validation error, not a launch-time
    surprise.
  * :meth:`TypedWitnessConfig.program_manifest` emits the DSL-provenance manifest
    (typed-config hash + argv sha256 + DSL module qualname + timestamp) the launcher
    gate checks — a launch config MUST arrive with DSL provenance or be REFUSED.

means != ends: a typed config is a MEANS. Only a byte-closed n600 exact row < 0.19110
from ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tac.witness_dsl.curriculum_dsl import (
    Anneal,
    Authority,
    Contain,
    Lever,
    Preserve,
    Regularizer,
    Stage,
    WitnessProgram,
)

__all__ = [
    "ProvenanceClass",
    "GovernanceClass",
    "GovernanceRole",
    "Provenanced",
    "ScheduleGovernance",
    "TypedAnneal",
    "TypedStage",
    "TypedRegularizer",
    "TypedLever",
    "TypedWitnessConfig",
    "TypedConfigError",
    "PROGRAM_MANIFEST_SCHEMA",
    "build_launch_manifest",
    "verify_launch_manifest",
]

# A launch-flag value is exactly one of these scalar types (never a container — a
# trainer flag takes a single token or is a bare boolean). Modelled explicitly so an
# accidental list/dict value is a validation error, not a silent str(...) render.
Scalar = Union[bool, int, float, str]

PROGRAM_MANIFEST_SCHEMA = "dsl_program_manifest.v1"


class TypedConfigError(ValueError):
    """Raised when a typed config cannot construct a valid WitnessProgram."""


# ---------------------------------------------------------------------------
# The value-provenance ladder (memory value_provenance_ladder_no_bare_constants).
# ---------------------------------------------------------------------------
class ProvenanceClass(str, Enum):
    """Where a value sits on the provenance ladder — highest legal rung first.

    DERIVED_LIVE       evaluated from MEASURED state at runtime (a live law).
    DERIVED_AT_CONFIG  computed from a stated law + provenanced inputs at config time,
                       carries a re-derivation trigger (NOT a bare hardcode).
    MEASURED_ANCHOR    an artifact-cited anchor, declared CONFIG-CONDITIONAL
                       (a constant measured at one config is stale at another).
    HARDCODED_WITH_WAIVER  a bare literal — LEGAL ONLY with a waiver naming the
                       reason + the condition that forces re-derivation.
    """

    DERIVED_LIVE = "DERIVED-LIVE"
    DERIVED_AT_CONFIG = "DERIVED-AT-CONFIG"
    MEASURED_ANCHOR = "MEASURED-ANCHOR"
    HARDCODED_WITH_WAIVER = "HARDCODED-WITH-WAIVER"


class GovernanceClass(str, Enum):
    """A schedule trigger's governance class (the schedule-provenance gate surface).

    EVENT  a NAMED co-emitted event-condition sensor governs the transition.
    CAP    an explicitly TAGGED requirement-B secondary fail-safe cap that backs up
           a governing event sensor (a fixed epoch may exist ONLY as such a cap).
    """

    EVENT = "event"
    CAP = "cap"


class GovernanceRole(str, Enum):
    """A schedule trigger's ROLE in the event/backstop pair (S4 R1, 2026-07-08).

    FIRES      this trigger is the RUNTIME sensor-fired transition (pairs with class=event); its
               ``sensor`` names the wired event it fires ON.
    BACKSTOPS  this trigger is a fail-safe backstop CAP (pairs with class=cap); its ``sensor`` names
               the co-emitted EVENT it backs up (the event fires FIRST; the cap only if the event
               did not by the cap epoch). Makes a CAP's ``sensor`` un-misreadable as a firing claim.
    """

    FIRES = "fires"
    BACKSTOPS = "backstops"


class Provenanced(BaseModel):
    """A value knob tagged with its ladder class, unit, and source (fail-closed).

    A HARDCODED_WITH_WAIVER value REQUIRES a non-empty waiver rationale — a bare
    literal with no declared provenance is the exact bug class this schema extincts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Scalar = Field(description="the knob value (a single trainer-flag token)")
    provenance: ProvenanceClass = Field(description="ladder class of this value")
    unit: str = Field(default="", description="physical/logical unit, e.g. 'epochs', 'tau', 'dimensionless'")
    source: str = Field(default="", description="artifact path / LawRef equation_id / measurement anchor")
    waiver: str | None = Field(
        default=None,
        description="required iff provenance==HARDCODED-WITH-WAIVER: reason + re-derivation trigger",
    )

    @model_validator(mode="after")
    def _hardcoded_requires_waiver(self) -> "Provenanced":
        if self.provenance is ProvenanceClass.HARDCODED_WITH_WAIVER:
            if not (self.waiver and self.waiver.strip()):
                raise ValueError(
                    "HARDCODED-WITH-WAIVER value requires a non-empty `waiver` naming the "
                    "reason + the condition that forces re-derivation (value-provenance ladder)"
                )
        elif self.waiver:
            raise ValueError(
                f"waiver is only legal on HARDCODED-WITH-WAIVER (got provenance={self.provenance.value}); "
                "a derived/measured value's provenance IS its justification"
            )
        return self


class ScheduleGovernance(BaseModel):
    """A per-flag schedule-governance declaration — the gate's declaration surface.

    Modelled first-class so a malformed governance block (a CAP with no backing
    sensor, an EVENT with no sensor, a blank rationale) is a validation error at
    config-authoring time, not a launch-time REFUSE.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    # 'class' is a Python keyword — accept it as the wire key via alias, expose as `cls`.
    cls: GovernanceClass = Field(alias="class", description="event | cap")
    sensor: str | None = Field(
        default=None,
        description="the governing event-condition sensor flag (e.g. --curriculum-event-triggered)",
    )
    # S4 R1 (2026-07-08): the event/backstop ROLE discriminator. Optional for backward compat — a
    # missing role DEFAULTS from the class (event->fires, cap->backstops) so pre-R1 declarations
    # still validate; when present it must AGREE with the class so a CAP's sensor cannot be misread
    # as a firing claim (the exact event-vs-cap ambiguity the gate exists to kill).
    role: GovernanceRole | None = Field(
        default=None,
        description="fires (class=event: this trigger IS sensor-fired) | backstops (class=cap: this "
                    "trigger is a fail-safe cap backing up the co-emitted event named in `sensor`)",
    )
    rationale: str = Field(description="why this trigger is governed (non-empty)")

    @property
    def effective_role(self) -> GovernanceRole:
        """The role, defaulted from the class when not explicitly declared (backward compat)."""
        if self.role is not None:
            return self.role
        return (GovernanceRole.FIRES if self.cls is GovernanceClass.EVENT
                else GovernanceRole.BACKSTOPS)

    @model_validator(mode="after")
    def _governance_shape(self) -> "ScheduleGovernance":
        if not self.rationale.strip():
            raise ValueError("schedule_governance rationale must be non-empty")
        if self.sensor is not None and not self.sensor.startswith("--"):
            raise ValueError(f"schedule_governance sensor must be a --flag (got {self.sensor!r})")
        # A CAP is only legal when it names the event sensor it backs up (requirement B).
        if self.cls is GovernanceClass.CAP and not self.sensor:
            raise ValueError(
                "schedule_governance class=cap requires a `sensor` naming the governing event "
                "it backs up (a fixed epoch may exist ONLY as a tagged fail-safe cap)"
            )
        if self.cls is GovernanceClass.EVENT and not self.sensor:
            raise ValueError(
                "schedule_governance class=event requires a `sensor` naming the co-emitted "
                "event-condition sensor flag that governs the transition"
            )
        # S4 R1: an explicit role must AGREE with the class (event<->fires, cap<->backstops).
        if self.role is GovernanceRole.FIRES and self.cls is not GovernanceClass.EVENT:
            raise ValueError(
                "schedule_governance role=fires requires class=event (a fires trigger IS the "
                f"runtime sensor-fired transition; got class={self.cls.value})"
            )
        if self.role is GovernanceRole.BACKSTOPS and self.cls is not GovernanceClass.CAP:
            raise ValueError(
                "schedule_governance role=backstops requires class=cap (a backstops trigger is a "
                f"fail-safe cap backing up an event; got class={self.cls.value})"
            )
        return self


# ---------------------------------------------------------------------------
# Typed schedule / curriculum / lever elements (adapters to the DSL dataclasses).
# ---------------------------------------------------------------------------
class TypedAnneal(BaseModel):
    """A start->end annealed schedule (e.g. softmax temperature tau)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: Provenanced = Field(description="anneal start value")
    end: Provenanced = Field(description="anneal end value")

    def to_dsl(self) -> Anneal:
        return Anneal(float(self.start.value), float(self.end.value))  # type: ignore[arg-type]


class TypedStage(BaseModel):
    """A curriculum relaxation stage; ``start_epoch`` maps to a trainer stage gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="stage name, e.g. 'tau_softplus'")
    start_epoch_flag: str | None = Field(
        default=None, description="trainer --*-start-epoch flag, or None for the CE base"
    )
    start_epoch: int | None = Field(
        default=None, ge=0, description="stage start epoch (>=0); None for the CE base"
    )

    @model_validator(mode="after")
    def _flag_epoch_paired(self) -> "TypedStage":
        if (self.start_epoch_flag is None) != (self.start_epoch is None):
            raise ValueError(
                f"stage {self.name!r}: start_epoch_flag and start_epoch must be BOTH set "
                "or BOTH None (a flag with no epoch, or an epoch with no flag, is ambiguous)"
            )
        if self.start_epoch_flag is not None and not self.start_epoch_flag.startswith("--"):
            raise ValueError(f"stage {self.name!r}: start_epoch_flag must be a --flag")
        return self

    def to_dsl(self) -> Stage:
        return Stage(self.name, self.start_epoch_flag, self.start_epoch)


class TypedRegularizer(BaseModel):
    """A live PDE regularizer (eikonal |grad phi|=1, length INT ds)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    flag: str = Field(description="trainer regularizer weight flag, e.g. --eikonal-weight")
    weight: Provenanced = Field(description="the regularizer weight")

    @model_validator(mode="after")
    def _flag_shape(self) -> "TypedRegularizer":
        if not self.flag.startswith("--"):
            raise ValueError(f"regularizer flag must be a --flag (got {self.flag!r})")
        return self

    def to_dsl(self) -> Regularizer:
        return Regularizer(self.flag, float(self.weight.value))  # type: ignore[arg-type]


class TypedLever(BaseModel):
    """A named A/B lever: typed flag overrides + optional extra epochs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="lever name")
    overrides: dict[str, Scalar] = Field(
        default_factory=dict, description="flag->value overrides applied LAST (A/B toggle wins)"
    )
    epochs_delta: int = Field(default=0, description="epochs this lever adds to the run window")
    notes: str = Field(default="", description="human note")

    @model_validator(mode="after")
    def _override_flags(self) -> "TypedLever":
        for k in self.overrides:
            if not k.startswith("--"):
                raise ValueError(f"lever {self.name!r}: override key {k!r} must be a --flag")
        return self

    def to_dsl(self) -> Lever:
        return Lever(self.name, dict(self.overrides), self.epochs_delta, self.notes)


# ---------------------------------------------------------------------------
# The top-level typed config — the ONLY legal authoring surface for a launch.
# ---------------------------------------------------------------------------
class TypedWitnessConfig(BaseModel):
    """Typed, validated authoring surface for a witness launch config.

    ``extra="forbid"`` means an unknown key (a typo, an invented flag routed as a
    config key) is a hard error — the anti-invented-flag law at the schema layer.
    :meth:`to_program` CONSTRUCTS a ``curriculum_dsl.WitnessProgram`` (the DSL SoT),
    whose ``validate`` then applies the structural never-invent-flags + behaviour
    clauses against the REAL trainer argparse.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ── program identity / meta ──────────────────────────────────────────────
    name: str = Field(description="program name (stamped into the manifest)")
    out_dir: str = Field(description="run output directory")
    gt_cache: str = Field(description="path to the clip's GT cache (.npz)")
    num_pairs: int = Field(gt=0, description="pairs (unit: frame-pairs); n600 = 600")
    epochs: int = Field(gt=0, description="training epochs (unit: epochs)")
    mlx_device: str = Field(default="gpu", description="mlx device: 'gpu' | 'cpu'")
    seed: int = Field(default=0, ge=0, description="RNG seed (deterministic-repro)")
    purpose: str | None = Field(
        default=None, description="one-line declared run intent (clean baseline / frontier / A-B)"
    )

    # ── schedule (the temperature anneal + curriculum stages + regularizers) ──
    temp: TypedAnneal = Field(description="softmax-temp anneal start->end")
    stages: tuple[TypedStage, ...] = Field(default=(), description="curriculum relaxation stages")
    regularizers: tuple[TypedRegularizer, ...] = Field(
        default=(), description="live PDE regularizers (eikonal / length)"
    )
    levers: tuple[TypedLever, ...] = Field(default=(), description="composed A/B levers")

    # ── the schedule-governance declaration surface (the gate reads this) ─────
    schedule_governance: dict[str, ScheduleGovernance] = Field(
        default_factory=dict,
        description="per-flag {class,sensor,rationale} governance for --*-start-epoch triggers",
    )

    # ── substrate base flags (arch/basis/chroma/...) — the DSL `base` dict ────
    base: dict[str, Scalar] = Field(
        default_factory=dict, description="substrate flag->value (arch, basis, chroma, ...)"
    )

    # ── verdict device (operator 2026-07-08 GPU-verdict HYBRID) ──────────────
    # DEVICE for the ADVISORY d_seg/d_pose scalars + the CPU-torch positive-control ANCHOR
    # cadence. "cpu" (default) => byte-identical to the sealed #205 verdict. "gpu" => the MLX
    # scorer ports (fast trajectory monitor); pair with verdict_anchor_every>0 to keep the
    # CPU-torch sentinel. NON-PROMOTABLE either way (CLAUDE.md: MLX/MPS is NEVER a score).
    # Compiled into --verdict-device / --verdict-anchor-every via ``base`` in :meth:`to_program`.
    verdict_device: str = Field(default="cpu", description="verdict device: 'cpu' | 'gpu'")
    verdict_anchor_every: int = Field(
        default=0, ge=0, description="gpu HYBRID: run the CPU-torch anchor every Nth gpu verdict (0=off)")

    @model_validator(mode="after")
    def _base_and_governance_flags(self) -> "TypedWitnessConfig":
        for k in self.base:
            if not k.startswith("--"):
                raise ValueError(f"base flag key {k!r} must be a --flag")
        for k in self.schedule_governance:
            if not k.startswith("--"):
                raise ValueError(f"schedule_governance key {k!r} must be a --flag")
        if self.mlx_device not in ("gpu", "cpu"):
            raise ValueError(f"mlx_device must be 'gpu' or 'cpu' (got {self.mlx_device!r})")
        if self.verdict_device not in ("gpu", "cpu"):
            raise ValueError(f"verdict_device must be 'gpu' or 'cpu' (got {self.verdict_device!r})")
        # the two verdict flags are compiled from the typed fields; forbid double-setting via base
        for k in ("--verdict-device", "--verdict-anchor-every"):
            if k in self.base:
                raise ValueError(
                    f"{k} is set via the typed verdict_device/verdict_anchor_every fields; "
                    "do not also put it in base")
        return self

    # ── the adapter: typed config -> DSL WitnessProgram ──────────────────────
    def to_program(self) -> WitnessProgram:
        """Construct the DSL ``WitnessProgram`` this typed config authors.

        The DSL dataclass is the compile SoT; ``WitnessProgram.validate`` then applies
        the never-invent-flags + behaviour clauses against the REAL trainer argparse.
        Preserve/Contain/Authority use the DSL defaults (their binding invariants are
        checked by ``WitnessProgram.validate``)."""
        try:
            # compile the typed verdict-device fields into the base flag dict (SoT stays the DSL).
            _base = dict(self.base)
            _base["--verdict-device"] = self.verdict_device
            _base["--verdict-anchor-every"] = int(self.verdict_anchor_every)
            return WitnessProgram(
                out_dir=self.out_dir,
                gt_cache=self.gt_cache,
                epochs=self.epochs,
                num_pairs=self.num_pairs,
                temp=self.temp.to_dsl(),
                stages=tuple(s.to_dsl() for s in self.stages),
                regularizers=tuple(r.to_dsl() for r in self.regularizers),
                preserve=Preserve(),
                contain=Contain(),
                authority=Authority(),
                base=_base,
                levers=tuple(lv.to_dsl() for lv in self.levers),
                mlx_device=self.mlx_device,
                purpose=self.purpose,
            )
        except Exception as exc:  # surface a typed error, never a raw dataclass crash
            raise TypedConfigError(
                f"typed config {self.name!r} could not construct a WitnessProgram: {exc}"
            ) from exc

    def validate_program(self, trainer_path=None) -> list[str]:
        """Return the DSL ``WitnessProgram.validate`` violations (empty == valid)."""
        return self.to_program().validate(trainer_path=trainer_path)

    # ── the DSL-provenance manifest (the launcher gate reads this) ───────────
    def typed_config_hash(self) -> str:
        """Deterministic sha256 of the typed config's canonical JSON (order-stable)."""
        payload = self.model_dump(mode="json", by_alias=True)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def emitted_flag_names(self) -> tuple[str, ...]:
        """The SORTED set of flag names the authored program emits (order-independent).

        This is the DSL-provenance fingerprint the launcher matches against the
        emitted launch.sh config flags — proving the flags came from the typed DSL
        layer, not hand assembly."""
        fd = self.to_program().flag_dict()
        return tuple(sorted(fd.keys()))

    def program_manifest(self) -> dict:
        """Emit the DSL-provenance manifest the launcher writes beside launch.sh.

        Records: schema tag, program name, typed-config hash, the compiled argv sha256,
        the sorted emitted flag-name fingerprint, the DSL module+qualname, and the
        compile timestamp. A launch config MUST arrive with this manifest (DSL
        provenance) or the launcher REFUSES it (hand-crafted argv is not authored)."""
        prog = self.to_program()
        argv = prog.compile_trainer_argv()
        argv_sha = hashlib.sha256("\x00".join(argv).encode("utf-8")).hexdigest()
        return {
            "schema": PROGRAM_MANIFEST_SCHEMA,
            "program_name": self.name,
            "typed_config_hash": self.typed_config_hash(),
            "argv_sha256": argv_sha,
            "flag_names": list(self.emitted_flag_names()),
            "dsl_module": "tac.witness_dsl.typed_config",
            "dsl_qualname": "TypedWitnessConfig.to_program",
            "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Launch-manifest helpers — the DSL-provenance attestation the LAUNCHER gate reads.
#
# For a real launch, the config that reaches the launcher is a
# ``witness_autoconfig.WitnessConfig`` whose EMITTED argv is the SoT (not a pure
# WitnessProgram). ``build_launch_manifest`` records (a) the emitted flag-name
# fingerprint + (b) a typed-validation attestation (the ``typed_config_hash`` of the
# TypedWitnessConfig that VALIDATED the config's DSL-authorable knobs). The launcher
# then ``verify_launch_manifest``s that a manifest is present AND its fingerprint
# matches the emitted config flags — proving the config was authored+validated
# through the typed DSL layer, not hand-assembled. Absent/mismatched => the launcher
# REFUSES (unless the operator escape hatch stamps a non-DSL provenance).
# ---------------------------------------------------------------------------
def build_launch_manifest(
    *,
    program_name: str,
    emitted_flag_names: "list[str] | tuple[str, ...]",
    typed_config_hash: str,
    typed_validated: bool = True,
) -> dict:
    """Build the DSL-provenance manifest attached to a launch-ready config."""
    names = sorted(set(emitted_flag_names))
    fp = hashlib.sha256("\x00".join(names).encode("utf-8")).hexdigest()
    return {
        "schema": PROGRAM_MANIFEST_SCHEMA,
        "program_name": program_name,
        "typed_validated": bool(typed_validated),
        "typed_config_hash": typed_config_hash,
        "flag_names": names,
        "flag_fingerprint": fp,
        "dsl_module": "tac.witness_dsl.typed_config",
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def verify_launch_manifest(
    manifest: dict | None,
    emitted_flag_names: "list[str] | tuple[str, ...]",
) -> tuple[bool, str]:
    """Return (ok, detail): is ``manifest`` a valid DSL-provenance attestation whose
    flag fingerprint matches the launcher's ``emitted_flag_names``?

    ok=False when the manifest is absent (hand-crafted argv — the confound this gate
    extincts), malformed (wrong/missing schema), not typed-validated, or its recorded
    flag set disagrees with what is actually being emitted (a config mutated after
    authoring). The launcher REFUSES on ok=False unless the operator escape hatch is set.
    """
    if not manifest:
        return False, "no DSL program manifest attached (config was not authored through the typed DSL layer)"
    if manifest.get("schema") != PROGRAM_MANIFEST_SCHEMA:
        return False, f"manifest schema {manifest.get('schema')!r} != {PROGRAM_MANIFEST_SCHEMA!r}"
    if not manifest.get("typed_validated"):
        return False, "manifest is not typed_validated (the typed-config gate did not certify this config)"
    emitted = sorted(set(emitted_flag_names))
    recorded = sorted(set(manifest.get("flag_names") or []))
    if recorded != emitted:
        missing = sorted(set(emitted) - set(recorded))
        extra = sorted(set(recorded) - set(emitted))
        return False, (
            "manifest flag set disagrees with the emitted argv "
            f"(emitting-but-unattested={missing[:6]}; attested-but-not-emitted={extra[:6]}) "
            "— the config was mutated after typed authoring"
        )
    return True, f"DSL-authored ({manifest.get('program_name')!r}, {len(emitted)} flags, typed-validated)"
