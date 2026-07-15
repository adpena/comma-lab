# SPDX-License-Identifier: MIT
"""LawRef — the constant-COMPILER: canonical equations resolved into values.

Operator design 2026-07-08 (task #351): *constants programmed as canonical
equations capturing system dynamics, resolved at DSL-compile time (the
mx.compile analogy) into actual values, with a provenance manifest; single
source of truth.* This module is the executable link that makes the triality's
**equations leg** compile into the **DSL leg**.

It is the by-construction enforcement of T5-crucible ORCHESTRATION_LEDGER
requirement **T** (the VALUE-PROVENANCE LADDER): every constant sits on a
preference ladder and carries its provenance so it *cannot rot silently*. A
``LawRef`` is a class-(2/3) value made executable — an ``equation_id`` + typed
``InputRef``s that resolve, deterministically, into a value + a manifest row.

The ladder (highest → lowest, verbatim from req T):
  1. ``derived_live``       — a law evaluated at runtime from measured state.
  2. ``derived_at_config``  — a law + pinned inputs, executable + cited.
  3. ``measured_anchor``    — an artifact-cited measurement, staleness-scoped,
                              declared CONFIG-CONDITIONAL.
  4. ``hardcoded_waiver``   — last resort: reason + owner + re-derivation trigger.

Fail-closed semantics (the P-CT1 protection mechanized — ν=0.012653 valid at
mod-32/this schedule, re-fit on config change):
  * CONFIG-CONDITIONALITY conflict (an anchor/literal ``config_tag`` whose key
    is also declared in the target config with a DIFFERENT value) →
    :class:`ConfigConditionalityViolation`, ALWAYS (never swallowed by a
    fallback — silently using a wrong-config constant is exactly the bug).
  * artifact missing / sha256 mismatch / staleness exceeded / extract-key
    missing → the underlying :class:`LawResolveError`, UNLESS the LawRef
    declares a ``fallback`` (+ waiver reason), in which case the fallback is
    used and the manifest records ``fallback_used=True`` + the reason.

Determinism: the resolved VALUE depends only on the resolved inputs (artifact
bytes + literals) and the pure-math evaluator. ``resolved_at_utc`` is metadata
only — it never enters the value. Two resolves of the same LawRef against the
same artifacts + target tags yield an identical value (bit-for-bit).

Cross-references:
  * ``tac.canonical_equations.evaluators`` — the executable eval surface called
    here (``resolve_equation_value``).
  * ``tac.witness_dsl.curriculum_dsl.WitnessProgram`` — the DSL consumer; a
    program may carry LawRef-valued flags and compile them via
    :meth:`WitnessProgram.compile_trainer_argv_with_constants`.
  * CLAUDE.md "The Triality — DAG ↔ DSL ↔ equations" + "Canonical equations +
    models registry" non-negotiables; req T/P/Q of the T5-crucible ledger.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ladder classes (req T verbatim tokens).
# ---------------------------------------------------------------------------
LADDER_DERIVED_LIVE = "derived_live"
LADDER_DERIVED_AT_CONFIG = "derived_at_config"
LADDER_MEASURED_ANCHOR = "measured_anchor"
LADDER_HARDCODED_WAIVER = "hardcoded_waiver"

VALID_LADDER_CLASSES = frozenset(
    {
        LADDER_DERIVED_LIVE,
        LADDER_DERIVED_AT_CONFIG,
        LADDER_MEASURED_ANCHOR,
        LADDER_HARDCODED_WAIVER,
    }
)

_KIND_LITERAL = "literal"
_KIND_ANCHOR = "anchor"
_VALID_KINDS = frozenset({_KIND_LITERAL, _KIND_ANCHOR})

# Canonical equation_id pattern (mirrors equation.py / evaluators.py).
_EQUATION_ID_RE = re.compile(r"^[a-z][a-z0-9_]*_v\d+$")


class LawRefError(ValueError):
    """Raised when a LawRef / InputRef violates a construction invariant."""


class LawResolveError(RuntimeError):
    """Raised at resolve time when an input cannot be resolved (fail-closed).

    Fallback-recoverable class (used only when the LawRef declares a fallback):
    missing artifact, sha256 mismatch, staleness exceeded, missing extract key.
    """


class ConfigConditionalityViolation(LawResolveError):
    """A config_tag conflicts with the target config (the P-CT1 protection).

    NEVER swallowed by a fallback: substituting a numeric fallback for a
    wrong-config constant would hide the exact mis-application req T extincts.
    """


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# InputRef — a discriminated union {literal | anchor}.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InputRef:
    """One resolvable input to a law's evaluator.

    ``kind == "literal"``: the value is pinned inline (``value``); it MAY still
    carry ``config_tags`` (a hardcoded MEASURED value must declare its
    conditionality — req T's exact lesson).

    ``kind == "anchor"``: the value is read from a JSON artifact at
    ``artifact_path`` via the ``extract`` key-path ("/"-separated; integer
    segments index lists), with optional ``expected_sha256`` (byte integrity),
    ``config_tags`` (config-conditionality), and ``max_staleness_days``.

    ``provenance`` (required, non-empty) states WHY this input holds its value
    / where it came from — the ladder metadata a bare literal lacks.
    """

    kind: str
    provenance: str
    value: float | int | None = None
    artifact_path: str | None = None
    extract: str | None = None
    expected_sha256: str | None = None
    config_tags: Mapping[str, str] = field(default_factory=dict)
    max_staleness_days: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise LawRefError(f"kind={self.kind!r} must be one of {sorted(_VALID_KINDS)!r}")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise LawRefError("provenance must be a non-empty string (the ladder metadata)")
        if not isinstance(self.config_tags, Mapping):
            raise LawRefError("config_tags must be a mapping")
        for k, v in self.config_tags.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise LawRefError("config_tags must map str->str")
        if self.kind == _KIND_LITERAL:
            if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                raise LawRefError("literal InputRef.value must be int|float (not bool)")
            if self.value != self.value:  # NaN
                raise LawRefError("literal InputRef.value must not be NaN")
            for bad in ("artifact_path", "extract", "expected_sha256", "max_staleness_days"):
                if getattr(self, bad) is not None:
                    raise LawRefError(f"literal InputRef must not set {bad}")
        else:  # anchor
            if not isinstance(self.artifact_path, str) or not self.artifact_path.strip():
                raise LawRefError("anchor InputRef requires a non-empty artifact_path")
            if not isinstance(self.extract, str) or not self.extract.strip():
                raise LawRefError("anchor InputRef requires a non-empty extract key-path")
            if self.value is not None:
                raise LawRefError("anchor InputRef must not set value (it is read from the artifact)")
            if self.max_staleness_days is not None and (
                not isinstance(self.max_staleness_days, int) or self.max_staleness_days <= 0
            ):
                raise LawRefError("max_staleness_days must be a positive int or None")
            if self.expected_sha256 is not None and not re.fullmatch(
                r"[0-9a-f]{64}", self.expected_sha256
            ):
                raise LawRefError("expected_sha256 must be a 64-char lowercase hex digest or None")

    # -- factory helpers (readable construction) ----------------------------
    @classmethod
    def literal(
        cls,
        value: float | int,
        provenance: str,
        *,
        config_tags: Mapping[str, str] | None = None,
    ) -> InputRef:
        return cls(
            kind=_KIND_LITERAL,
            provenance=provenance,
            value=value,
            config_tags=dict(config_tags or {}),
        )

    @classmethod
    def anchor(
        cls,
        artifact_path: str,
        extract: str,
        provenance: str,
        *,
        expected_sha256: str | None = None,
        config_tags: Mapping[str, str] | None = None,
        max_staleness_days: int | None = None,
    ) -> InputRef:
        return cls(
            kind=_KIND_ANCHOR,
            provenance=provenance,
            artifact_path=artifact_path,
            extract=extract,
            expected_sha256=expected_sha256,
            config_tags=dict(config_tags or {}),
            max_staleness_days=max_staleness_days,
        )


# ---------------------------------------------------------------------------
# LawRef — the compilable constant.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LawRef:
    """A constant expressed as an equation_id + typed inputs, resolvable at compile.

    ``fallback`` (+ ``fallback_waiver_reason``): the class-(4) last resort used
    ONLY when an anchor input cannot be resolved (missing/sha/staleness/extract);
    a config-conditionality conflict is NEVER swallowed by it.
    """

    equation_id: str
    inputs: Mapping[str, InputRef]
    ladder_class: str
    fallback: float | int | None = None
    fallback_waiver_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.equation_id, str) or not _EQUATION_ID_RE.match(self.equation_id):
            raise LawRefError(
                f"equation_id={self.equation_id!r} must match snake_case_vN"
            )
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise LawRefError("inputs must be a non-empty mapping of name->InputRef")
        for name, ref in self.inputs.items():
            if not isinstance(name, str) or not name:
                raise LawRefError("input names must be non-empty strings")
            if not isinstance(ref, InputRef):
                raise LawRefError(f"inputs[{name!r}] must be an InputRef, got {type(ref).__name__}")
        if self.ladder_class not in VALID_LADDER_CLASSES:
            raise LawRefError(
                f"ladder_class={self.ladder_class!r} must be one of {sorted(VALID_LADDER_CLASSES)!r}"
            )
        if self.fallback is not None:
            if isinstance(self.fallback, bool) or not isinstance(self.fallback, (int, float)):
                raise LawRefError("fallback must be int|float (not bool) or None")
            if not self.fallback_waiver_reason.strip():
                raise LawRefError(
                    "a fallback REQUIRES a non-empty fallback_waiver_reason "
                    "(the req-T class-4 recorded reason)"
                )
        # A hardcoded_waiver-class LawRef MUST carry the fallback it waives to.
        if self.ladder_class == LADDER_HARDCODED_WAIVER and self.fallback is None:
            raise LawRefError(
                "ladder_class=hardcoded_waiver requires a fallback (+ waiver reason) — "
                "the value it hardcodes to"
            )


# ---------------------------------------------------------------------------
# Resolution.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedInputRecord:
    """The audited record of one resolved input (for the manifest)."""

    name: str
    kind: str
    value: float | int
    source: str  # "literal" | artifact_path
    sha256: str | None
    config_tags: Mapping[str, str]
    provenance: str
    staleness_days: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "source": self.source,
            "sha256": self.sha256,
            "config_tags": dict(self.config_tags),
            "provenance": self.provenance,
            "staleness_days": self.staleness_days,
        }


@dataclass(frozen=True)
class ResolvedConstant:
    """The compiled value + full provenance of one LawRef resolution."""

    value: float | int
    equation_id: str
    resolved_inputs: tuple[ResolvedInputRecord, ...]
    resolved_at_utc: str
    ladder_class: str
    fallback_used: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "equation_id": self.equation_id,
            "ladder_class": self.ladder_class,
            "fallback_used": self.fallback_used,
            "resolved_at": self.resolved_at_utc,
            "inputs": [r.to_dict() for r in self.resolved_inputs],
            "warnings": list(self.warnings),
        }


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _navigate(doc: Any, key_path: str) -> Any:
    """Walk a nested JSON doc by a "/"-separated key-path (int segs index lists)."""
    cur = doc
    for seg in key_path.split("/"):
        if seg == "":
            continue
        if isinstance(cur, Mapping):
            if seg not in cur:
                raise LawResolveError(f"extract key {seg!r} not found in path {key_path!r}")
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError as exc:
                raise LawResolveError(
                    f"extract segment {seg!r} is not a list index in path {key_path!r}"
                ) from exc
            if not (0 <= idx < len(cur)):
                raise LawResolveError(f"extract index {idx} out of range in path {key_path!r}")
            cur = cur[idx]
        else:
            raise LawResolveError(
                f"cannot descend into {type(cur).__name__} at segment {seg!r} of {key_path!r}"
            )
    return cur


def _check_config_conditionality(
    ref_tags: Mapping[str, str],
    target_config_tags: Mapping[str, str],
    where: str,
) -> list[str]:
    """Enforce the P-CT1 protection; raise on conflict, warn on unverifiable."""
    warnings: list[str] = []
    for k, anchor_v in ref_tags.items():
        if k in target_config_tags:
            if target_config_tags[k] != anchor_v:
                raise ConfigConditionalityViolation(
                    f"CONFIG-CONDITIONALITY conflict in {where}: config_tag "
                    f"{k!r} = {anchor_v!r} (anchor) != {target_config_tags[k]!r} (target). "
                    "This value is not valid for the target config — re-derive/re-fit "
                    "(req T P-CT1 protection; NOT fallback-recoverable)."
                )
        else:
            warnings.append(
                f"UNVERIFIABLE config_tag {k!r}={anchor_v!r} in {where}: target config "
                f"did not declare {k!r} (cannot confirm applicability)"
            )
    return warnings


def _resolve_one_input(
    name: str,
    ref: InputRef,
    target_config_tags: Mapping[str, str],
    repo_root: Path,
) -> tuple[ResolvedInputRecord, list[str]]:
    """Resolve a single InputRef to (record, warnings). May raise.

    Config-conditionality is checked FIRST so a conflict fails closed before any
    artifact I/O; unverifiable-tag warnings are returned (not folded into
    provenance) so the manifest keeps a clean provenance string + a warnings list.
    """
    where = f"input {name!r}"
    warnings = _check_config_conditionality(ref.config_tags, target_config_tags, where)
    if ref.kind == _KIND_LITERAL:
        rec = ResolvedInputRecord(
            name=name,
            kind=_KIND_LITERAL,
            value=ref.value,  # type: ignore[arg-type]
            source="literal",
            sha256=None,
            config_tags=dict(ref.config_tags),
            provenance=ref.provenance,
            staleness_days=None,
        )
        return rec, warnings
    # anchor
    apath = Path(ref.artifact_path)  # type: ignore[arg-type]
    if not apath.is_absolute():
        apath = repo_root / apath
    if not apath.exists():
        raise LawResolveError(f"artifact for {where} not found: {apath}")
    sha = _sha256_of_file(apath)
    if ref.expected_sha256 is not None and sha != ref.expected_sha256:
        raise LawResolveError(
            f"sha256 mismatch for {where}: expected {ref.expected_sha256}, got {sha} "
            f"(artifact {apath} changed — input rotted)"
        )
    staleness_days: float | None = None
    if ref.max_staleness_days is not None:
        age_s = _dt.datetime.now(_dt.UTC).timestamp() - os.path.getmtime(apath)
        staleness_days = age_s / 86400.0
        if staleness_days > ref.max_staleness_days:
            raise LawResolveError(
                f"artifact for {where} is stale: {staleness_days:.2f}d > "
                f"{ref.max_staleness_days}d cap ({apath})"
            )
    with open(apath, encoding="utf-8") as fh:
        doc = json.load(fh)
    raw = _navigate(doc, ref.extract)  # type: ignore[arg-type]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise LawResolveError(
            f"extracted value for {where} is not numeric: {raw!r} "
            f"(path {ref.extract!r} in {apath})"
        )
    rec = ResolvedInputRecord(
        name=name,
        kind=_KIND_ANCHOR,
        value=raw,
        source=str(apath),
        sha256=sha,
        config_tags=dict(ref.config_tags),
        provenance=ref.provenance,
        staleness_days=staleness_days,
    )
    return rec, warnings


def resolve(
    lawref: LawRef,
    target_config_tags: Mapping[str, str] | None = None,
    *,
    repo_root: Path | str | None = None,
) -> ResolvedConstant:
    """Resolve a LawRef into a value + provenance manifest (fail-closed).

    ``target_config_tags`` are the config tags of the run this value is being
    compiled FOR (e.g. ``{"schedule": "mod32cap", "stage": "tau_softplus"}``).
    A conflict with any input's ``config_tags`` fails closed (the P-CT1
    protection). ``repo_root`` roots relative artifact paths (defaults to the
    repo root inferred from this file).
    """
    if not isinstance(lawref, LawRef):
        raise LawRefError(f"resolve expected a LawRef, got {type(lawref).__name__}")
    target = dict(target_config_tags or {})
    root = Path(repo_root) if repo_root is not None else _infer_repo_root()

    # Lazy-register the built-in evaluators (idempotent; import-order-proof).
    from tac.canonical_equations.evaluators import (
        populate_lawref_evaluators,
        resolve_equation_value,
    )

    populate_lawref_evaluators()

    records: list[ResolvedInputRecord] = []
    warnings: list[str] = []
    fallback_used = False

    try:
        for name, ref in lawref.inputs.items():
            rec, rec_warnings = _resolve_one_input(name, ref, target, root)
            records.append(rec)
            warnings.extend(rec_warnings)
        value_inputs = {r.name: r.value for r in records}
        value = resolve_equation_value(lawref.equation_id, value_inputs)
    except ConfigConditionalityViolation:
        # NEVER fallback-recoverable — re-raise so the mis-application surfaces.
        raise
    except LawResolveError as exc:
        if lawref.fallback is None:
            raise
        # class-4 hardcoded-with-waiver: use the declared fallback, record it.
        fallback_used = True
        warnings.append(
            f"FALLBACK USED ({lawref.fallback_waiver_reason}): input resolution failed — {exc}"
        )
        value = lawref.fallback
        # Preserve whatever records resolved before the failure for audit.

    return ResolvedConstant(
        value=value,
        equation_id=lawref.equation_id,
        resolved_inputs=tuple(records),
        resolved_at_utc=_utc_now_iso(),
        ladder_class=lawref.ladder_class,
        fallback_used=fallback_used,
        warnings=tuple(warnings),
    )


def _infer_repo_root() -> Path:
    """Repo root = 3 parents up from this file (src/tac/witness_dsl/lawref.py)."""
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Compile integration — resolve LawRef-valued flags in a program's flag_dict.
# ---------------------------------------------------------------------------
def resolve_flag_dict_constants(
    flag_dict: Mapping[str, Any],
    target_config_tags: Mapping[str, str] | None = None,
    *,
    repo_root: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve every LawRef-valued flag; return (resolved_flag_dict, manifest).

    Non-LawRef values pass through unchanged (byte-identical to the input for
    programs with no LawRef flags). The manifest maps each LawRef flag to its
    :meth:`ResolvedConstant.to_dict` record — the ``constants_manifest.json``
    content a launcher writes beside ``launch.sh`` (this function does NOT
    write files; it returns the content).
    """
    resolved: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    for flag, val in flag_dict.items():
        if isinstance(val, LawRef):
            rc = resolve(val, target_config_tags, repo_root=repo_root)
            resolved[flag] = rc.value
            manifest[flag] = rc.to_dict()
        else:
            resolved[flag] = val
    return resolved, manifest


# ---------------------------------------------------------------------------
# Declaration codec — the LOSSLESS, symmetric JSON round-trip for a LawRef.
#
# ``TypedLever.lawrefs`` carries live LawRef objects but is ``exclude=True``
# (resolved metadata must not make identical compiles hash-differently), so a
# serialized WitnessProgram spec keeps only ``lawref_declarations``. Without a
# decoder, ``model_validate(spec).to_program()`` rebuilt levers with EMPTY
# LawRef custody and the #506 dsl_compile_hash self-recompile refused its own
# document (lawref_equation_ids vanished on round-trip; 2026-07-15 incident).
# This pair is the single canonical codec: encode when authoring a TypedLever,
# decode when rehydrating a Lever from a deserialized spec. Fallback fields are
# included so hardcoded_waiver-class refs survive the round-trip too.
# ---------------------------------------------------------------------------
def lawref_to_declaration(ref: LawRef) -> dict[str, Any]:
    """Serialize a :class:`LawRef` to its timestamp-free JSON declaration."""
    return {
        "equation_id": ref.equation_id,
        "ladder_class": ref.ladder_class,
        "fallback": ref.fallback,
        "fallback_waiver_reason": ref.fallback_waiver_reason,
        "inputs": {
            name: {
                "kind": inp.kind,
                "value": inp.value,
                "artifact_path": inp.artifact_path,
                "extract": inp.extract,
                "expected_sha256": inp.expected_sha256,
                "config_tags": dict(inp.config_tags),
                "max_staleness_days": inp.max_staleness_days,
                "provenance": inp.provenance,
            }
            for name, inp in ref.inputs.items()
        },
    }


def lawref_from_declaration(declaration: Mapping[str, Any]) -> LawRef:
    """Rehydrate a :class:`LawRef` from :func:`lawref_to_declaration` output.

    Tolerates legacy declarations that predate the ``fallback`` fields (they
    take the LawRef defaults). Raises :class:`LawRefError` via the
    LawRef/InputRef validators on any malformed declaration — never a silent
    partial object.
    """
    inputs = {
        str(name): InputRef(
            kind=str(inp["kind"]),
            provenance=str(inp["provenance"]),
            value=inp.get("value"),
            artifact_path=inp.get("artifact_path"),
            extract=inp.get("extract"),
            expected_sha256=inp.get("expected_sha256"),
            config_tags=dict(inp.get("config_tags") or {}),
            max_staleness_days=inp.get("max_staleness_days"),
        )
        for name, inp in dict(declaration.get("inputs") or {}).items()
    }
    return LawRef(
        equation_id=str(declaration["equation_id"]),
        inputs=inputs,
        ladder_class=str(declaration["ladder_class"]),
        fallback=declaration.get("fallback"),
        fallback_waiver_reason=str(declaration.get("fallback_waiver_reason") or ""),
    )


__all__ = [
    "LADDER_DERIVED_AT_CONFIG",
    "LADDER_DERIVED_LIVE",
    "LADDER_HARDCODED_WAIVER",
    "LADDER_MEASURED_ANCHOR",
    "VALID_LADDER_CLASSES",
    "ConfigConditionalityViolation",
    "InputRef",
    "LawRef",
    "LawRefError",
    "LawResolveError",
    "ResolvedConstant",
    "ResolvedInputRecord",
    "lawref_from_declaration",
    "lawref_to_declaration",
    "resolve",
    "resolve_flag_dict_constants",
]
