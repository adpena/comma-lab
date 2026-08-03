# SPDX-License-Identifier: MIT
"""GuardedConstant — a constant that resolves at runtime and REFUSES misuse.

Operator directive 2026-08-03, verbatim:

    "We could build a new class that resolves to a constant at runtime that
     includes protection against being used where inappropriate."
    "To help protect against confounding and also not fail silently."

This module EXTENDS :mod:`tac.witness_dsl.lawref` (#351, the constant-COMPILER).
It does not replace it and does not re-implement resolution: a
:class:`GuardedConstant` *composes* a :class:`~tac.witness_dsl.lawref.LawRef`
and delegates value + provenance to it (REUSE-NOT-REBUILD / anti-duplicate-SoT, task #533). What it adds
are the three guard axes that #351 does not carry, each one the cure for a
MEASURED misuse instance from 2026-08-03:

=====================  ====================================  =========================
axis                   what LawRef already does              the instance it misses
=====================  ====================================  =========================
STALENESS              artifact ``sha256`` + mtime           a **derivation that exists
                       ``max_staleness_days``                in-repo and is never
                                                             invoked** has no artifact
                                                             to age.  ``derive_margin_floor``
                                                             (``optimization/lane_guard.py:547``)
                                                             is documented "data-derived
                                                             per run, never a bare
                                                             constant" and the shipped
                                                             value is the frozen literal
                                                             ``0.1``
                                                             (``direct_description_joint_descent.py:2281``).
DOMAIN                 ``config_tags`` str->str equality     the **input DISTRIBUTION** a
                                                             value was derived over.  The
                                                             floor is a percentile of
                                                             GT-Lane-RESTRICTED margins
                                                             (p5 0.0517 / p10 0.1047); read
                                                             off the GLOBAL field the same
                                                             percentile is p5 2.0582 — a
                                                             39.8x different population.
                                                             The value was not wrong; the
                                                             DOMAIN was.
UNITS / ROLE           *(no field at all)*                   ``W = 1.273108215332031`` is an
                                                             EXCHANGE RATE (what a flip is
                                                             worth: 4*DEN/PX, exact) and was
                                                             repeatedly handed to consumers
                                                             as a mechanism COST — measured
                                                             mechanism costs span 65x
                                                             (0.4981 .. 32.53 B/flip).  And
                                                             ``5.91 bits/flip`` was published
                                                             where the correct unit is
                                                             ``0.60 bits/band_pixel``: the
                                                             UNIT was wrong, not the value,
                                                             and it changed the design
                                                             conclusion.
=====================  ====================================  =========================

``CanonicalEquation`` (registry side) already declares ``domain_of_validity``,
``units_in`` and ``units_out`` — but they are free-form prose validated only as
``Mapping``.  This module is the executable half: it makes those three fields
**machine-checkable and REFUSING**.

Semantics
---------
* **Fail-CLOSED on write, LOUD on read.**  :func:`resolve` (the path that hands
  a value to a consumer) raises.  :func:`audit` (the path that only inspects)
  returns a record with the violations listed and never raises on them.
* **An empty scope is VACUOUS, never PASS.**  A resolution that ran zero checks
  reports :data:`STATUS_VACUOUS` and its ``checks_run`` denominator, so a
  guard that could not fire can never be mistaken for a guard that passed.
  (The standing "vacuity == pass" genus.)
* **Refusals are actionable**: every message names the rule that fired, the
  concrete value, and the fix (the preflight-message non-negotiable).
* **Corroboration is represented, not resolved away.**  Two independent
  derivations that AGREE are a feature (the ``margin_floor`` case: an fp32
  cross-hardware drift bound ~0.096 and a Lane-margin p10 0.104748, ratio
  1.091x).  Disagreement beyond a declared, provenance-carrying tolerance is a
  refusal.
* **Determinism**: :attr:`ResolvedGuardedConstant.resolution_hash` is a sha256
  over the canonical JSON of everything that determines the value.  Timestamps
  are excluded.  Same inputs -> same hash, bit-for-bit.
* **Migration honesty**: pass ``incumbent_literal`` and the record states
  whether the resolved value is byte-identical to the literal it replaces.
  Converting a hardcoded literal is a BEHAVIOUR CHANGE unless that is True.

Axis: apparatus.  ``score_claim=false``, ``promotion_eligible=false``.  No
scorer, no archive bytes, no pointer movement.

Cross-references: CLAUDE.md "Canonical equations + models registry",
"Confound self-protection" (the 3-layer immune system; this is an L1 runtime
alarm plus an L3 verdict-clearance precondition), the value-provenance ladder
(req T), Catalog #351 (LawRef/value custody — the catalog row this work belongs under),
task #533 (REUSE-NOT-REBUILD), task #847.  Sibling arm ``ddm_ca1`` owns the repo-wide AUDIT
(which constants are affected); this module owns the CLASS.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.witness_dsl.lawref import (
    LawRef,
    ResolvedConstant,
)
from tac.witness_dsl.lawref import (
    resolve as _resolve_lawref,
)


# ---------------------------------------------------------------------------
# Errors — every one names the rule, the value, and the fix.
# ---------------------------------------------------------------------------
class GuardedConstantError(ValueError):
    """Raised when a GuardedConstant declaration violates a construction invariant."""


class GuardViolation(RuntimeError):
    """Base of every refusal raised at :func:`resolve` time (fail-closed)."""


class UnitMismatch(GuardViolation):
    """Two quantities with different dimensions were compared or combined."""


class RoleMisuse(GuardViolation):
    """A constant was consumed in a semantic role it was not derived for."""


class DomainViolation(GuardViolation):
    """The supplied sample is not the input distribution the constant was derived over."""


class StaleDerivation(GuardViolation):
    """A live derivation is declared for this constant and was not invoked, or its
    output has drifted away from the frozen value beyond the declared tolerance."""


class CorroborationConflict(GuardViolation):
    """Two independent derivations of the same constant disagree beyond tolerance."""


# ---------------------------------------------------------------------------
# Units — a normalized dimensional token with exact equality.
# ---------------------------------------------------------------------------
#: Spelling aliases ONLY.  Deliberately NOT a conversion table: ``bits`` and
#: ``bytes`` are different dimensions here, because converting between them is a
#: deliberate act that must be visible at the call site, not an implicit coercion.
_UNIT_ALIASES: Mapping[str, str] = {
    "b": "bytes",
    "byte": "bytes",
    "bytes": "bytes",
    "bit": "bits",
    "bits": "bits",
    "flip": "flip",
    "flips": "flip",
    "px": "pixel",
    "pixel": "pixel",
    "pixels": "pixel",
    "band_pixel": "band_pixel",
    "band_pixels": "band_pixel",
    "component": "component",
    "components": "component",
    "pair": "pair",
    "pairs": "pair",
    "epoch": "epoch",
    "epochs": "epoch",
    "update": "update",
    "updates": "update",
    "s_unit": "S_unit",
    "s_units": "S_unit",
}

_DIMENSIONLESS_TOKENS = frozenset({"1", "", "dimensionless", "none", "ratio", "fraction"})
_UNIT_TERM_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*|1)(?:\^(-?\d+))?$")


def _canonical_unit_token(tok: str) -> str:
    t = tok.strip()
    return _UNIT_ALIASES.get(t.lower(), t)


@dataclass(frozen=True)
class Units:
    """A dimension expressed as base-token -> integer exponent.

    Equality is exponent-map equality, so ``bits/flip`` != ``bits/band_pixel``
    (the measured (c) instance) and ``bytes/flip`` != ``bytes/component``.
    Parsing accepts ``"bytes/flip"``, ``"bits*pixel^-1"``, ``"1"``,
    ``"dimensionless"``.
    """

    exponents: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.exponents, Mapping):
            raise GuardedConstantError("Units.exponents must be a mapping")
        cleaned: dict[str, int] = {}
        for k, v in self.exponents.items():
            if not isinstance(k, str) or not k:
                raise GuardedConstantError("Units exponent keys must be non-empty strings")
            if isinstance(v, bool) or not isinstance(v, int):
                raise GuardedConstantError(f"Units exponent for {k!r} must be an int, got {v!r}")
            if v != 0:
                cleaned[_canonical_unit_token(k)] = cleaned.get(_canonical_unit_token(k), 0) + v
        object.__setattr__(self, "exponents", {k: v for k, v in sorted(cleaned.items()) if v != 0})

    # -- construction -------------------------------------------------------
    @classmethod
    def parse(cls, spec: str) -> Units:
        """Parse ``"bytes/flip"`` / ``"bits*pixel^-1"`` / ``"1"`` into canonical form."""
        if not isinstance(spec, str):
            raise GuardedConstantError(f"Units.parse expects a str, got {type(spec).__name__}")
        text = spec.strip()
        if text.lower() in _DIMENSIONLESS_TOKENS:
            return cls({})
        exps: dict[str, int] = {}
        # split on '/' first: everything after the first '/' is inverted.
        parts = text.split("/")
        for depth, part in enumerate(parts):
            sign = 1 if depth == 0 else -1
            for term in part.split("*"):
                term = term.strip()
                if not term:
                    continue
                m = _UNIT_TERM_RE.match(term)
                if not m:
                    raise GuardedConstantError(
                        f"Units.parse: cannot parse term {term!r} in {spec!r}; "
                        "expected TOKEN or TOKEN^N (letters/underscore, optional ^int)"
                    )
                base, exp_txt = m.group(1), m.group(2)
                if base == "1":
                    continue
                exp = int(exp_txt) if exp_txt else 1
                key = _canonical_unit_token(base)
                exps[key] = exps.get(key, 0) + sign * exp
        return cls(exps)

    # -- algebra ------------------------------------------------------------
    def is_dimensionless(self) -> bool:
        return not self.exponents

    def __str__(self) -> str:
        if not self.exponents:
            return "dimensionless"
        num = [k if e == 1 else f"{k}^{e}" for k, e in self.exponents.items() if e > 0]
        den = [k if e == -1 else f"{k}^{-e}" for k, e in self.exponents.items() if e < 0]
        head = "*".join(num) if num else "1"
        return head if not den else f"{head}/{'*'.join(den)}"

    def assert_same(self, other: Units, *, where: str) -> None:
        """Refuse if dimensions differ.  The (c) bits/flip-vs-bits/band_pixel cure."""
        if not isinstance(other, Units):
            raise UnitMismatch(
                f"UNIT rule fired in {where}: expected a Units, got {type(other).__name__}. "
                "FIX: wrap the other quantity with Units.parse('<its unit>')."
            )
        if self.exponents != other.exponents:
            raise UnitMismatch(
                f"UNIT rule fired in {where}: {self} != {other}. These are different "
                "dimensions; comparing or combining them is meaningless. "
                f"FIX: convert explicitly at the call site (e.g. bits/flip -> bits/band_pixel "
                "requires the measured band-pixels-per-flip factor), or correct the "
                "declared units on whichever quantity is mislabelled."
            )


DIMENSIONLESS = Units({})


# ---------------------------------------------------------------------------
# Semantic role — what KIND of quantity this is, independent of its units.
# ---------------------------------------------------------------------------
ROLE_EXCHANGE_RATE = "exchange_rate"
ROLE_MECHANISM_COST = "mechanism_cost"
ROLE_THRESHOLD = "threshold"
ROLE_SCALE = "scale"
ROLE_BUDGET = "budget"
ROLE_COUNT = "count"
ROLE_FRACTION = "fraction"

VALID_ROLES = frozenset(
    {
        ROLE_EXCHANGE_RATE,
        ROLE_MECHANISM_COST,
        ROLE_THRESHOLD,
        ROLE_SCALE,
        ROLE_BUDGET,
        ROLE_COUNT,
        ROLE_FRACTION,
    }
)

#: Why exchange_rate and mechanism_cost are NOT interchangeable even though they
#: share the unit ``bytes/flip``: the exchange rate says what a flip is WORTH
#: (an identity of the score function, exact); a mechanism cost says what a
#: given mechanism CHARGES to buy one (an empirical property of that mechanism).
#: MEASURED 2026-08-03: mechanism costs span 65x around the exchange rate —
#: sx2 0.4981 B/flip, mf1 1854 B / 1483 components, qd1 32.53 B/flip.  Reading
#: the rate as a cost silently asserts every mechanism is exactly break-even.
_ROLE_INCOMPATIBILITY_NOTES: Mapping[tuple[str, str], str] = {
    (ROLE_EXCHANGE_RATE, ROLE_MECHANISM_COST): (
        "an EXCHANGE RATE states what one unit is WORTH (an exact identity of the "
        "score function); a MECHANISM COST states what a specific mechanism CHARGES "
        "to buy one (an empirical property of that mechanism). MEASURED 2026-08-03: "
        "real mechanism costs span 65x around the exchange rate (0.4981 .. 32.53 "
        "B/flip). Substituting the rate for a cost silently asserts break-even"
    ),
    (ROLE_MECHANISM_COST, ROLE_EXCHANGE_RATE): (
        "a MECHANISM COST is one mechanism's measured price; it is not the "
        "score-function identity and must not be used as the universal conversion"
    ),
    (ROLE_THRESHOLD, ROLE_SCALE): (
        "a THRESHOLD is a decision boundary chosen for a purpose; a SCALE is a "
        "property of the distribution. Using the distribution's scale as a "
        "threshold is the exact (b) instance"
    ),
}


def _role_note(declared: str, consumer: str) -> str:
    return _ROLE_INCOMPATIBILITY_NOTES.get(
        (declared, consumer),
        "the declared role and the consuming role are different kinds of quantity",
    )


# ---------------------------------------------------------------------------
# Domain — the INPUT DISTRIBUTION a value was derived over.
# ---------------------------------------------------------------------------
_CHECK_QUANTILE = "quantile"
_CHECK_N_SAMPLES = "n_samples"
_CHECK_MEAN = "mean"
_CHECK_MAX_ABS = "max_abs"
_VALID_CHECK_KINDS = frozenset({_CHECK_QUANTILE, _CHECK_N_SAMPLES, _CHECK_MEAN, _CHECK_MAX_ABS})


@dataclass(frozen=True)
class DomainCheck:
    """One declarative, machine-checkable statistic assertion about a sample.

    ``provenance`` is REQUIRED and non-empty: a band with no stated derivation
    is exactly the bare constant this module exists to extinct.  Prefer
    :meth:`max_margin_separator`, which DERIVES the band instead of asserting it.
    """

    kind: str
    lo: float
    hi: float
    provenance: str
    param: float | None = None  # quantile percentile, when kind == "quantile"

    def __post_init__(self) -> None:
        if self.kind not in _VALID_CHECK_KINDS:
            raise GuardedConstantError(
                f"DomainCheck.kind={self.kind!r} must be one of {sorted(_VALID_CHECK_KINDS)!r}"
            )
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise GuardedConstantError(
                "DomainCheck.provenance must be a non-empty string — a band with no stated "
                "derivation is the bare constant this module exists to extinct"
            )
        for name in ("lo", "hi"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise GuardedConstantError(f"DomainCheck.{name} must be a finite number, got {v!r}")
        if self.lo > self.hi:
            raise GuardedConstantError(f"DomainCheck lo={self.lo} > hi={self.hi}")
        if self.kind == _CHECK_QUANTILE and (
            self.param is None or not (0.0 <= float(self.param) <= 100.0)
        ):
            raise GuardedConstantError(
                "DomainCheck(kind='quantile') requires param = percentile in [0, 100]"
            )

    @classmethod
    def max_margin_separator(
        cls,
        *,
        kind: str,
        in_domain_value: float,
        out_of_domain_value: float,
        provenance: str,
        param: float | None = None,
    ) -> DomainCheck:
        """Band DERIVED as the log-space maximum-margin separator between two
        MEASURED statistics: the value this statistic takes on the correct
        distribution, and the value it takes on the distribution we must refuse.

        The multiplicative half-width is ``G = sqrt(out/in)`` so the band edge
        sits exactly at the geometric midpoint of the two measured populations —
        the widest band that still refuses the wrong one, and the tightest that
        still admits the right one.  Nothing here is chosen; both endpoints are
        measurements and G falls out.

        Worked (the (b) instance): the QA80 margin p5 is **0.0517** restricted to
        GT-Lane and **2.0582** on the GLOBAL field (ddm_rt2, n600, 117,964,800
        sites).  ``G = sqrt(2.0582/0.0517) = 6.3096`` -> band
        ``[0.008194, 0.326214]``; the global p5 sits 6.31x above the upper edge
        and is REFUSED; the lane p5 sits at the band centre and PASSES.
        """
        a, b = float(in_domain_value), float(out_of_domain_value)
        if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
            raise GuardedConstantError(
                "max_margin_separator requires two positive finite MEASURED statistics "
                f"(got in={in_domain_value!r}, out={out_of_domain_value!r}); for "
                "non-positive statistics declare an explicit band with provenance instead"
            )
        if a == b:
            raise GuardedConstantError(
                "max_margin_separator: the in-domain and out-of-domain statistics are equal "
                f"({a}); this statistic does NOT separate the two populations — pick a "
                "statistic that does, or the check would be vacuous"
            )
        g = math.sqrt(max(a, b) / min(a, b))
        return cls(
            kind=kind,
            lo=a / g,
            hi=a * g,
            param=param,
            provenance=(
                f"{provenance} | band DERIVED as log-space max-margin separator: "
                f"in_domain={a!r}, out_of_domain={b!r}, G=sqrt(out/in)={g:.6g}, "
                f"band=[{a / g:.6g}, {a * g:.6g}] (edge at the geometric midpoint)"
            ),
        )

    def evaluate(self, sample: Sequence[float] | Any) -> tuple[bool, float, str]:
        """Return ``(ok, observed, label)``.  Raises only on an unusable sample."""
        vals = _as_finite_1d(sample)
        if vals.size == 0:
            raise DomainViolation(
                f"DOMAIN rule fired: the sample supplied for check {self.label()} has ZERO "
                "finite values. An empty scope is VACUOUS, never PASS. "
                "FIX: pass the real field, or resolve without a sample and accept the "
                "VACUOUS status (which is itself reported, not silently green)."
            )
        if self.kind == _CHECK_QUANTILE:
            import numpy as _np

            observed = float(_np.percentile(vals, float(self.param)))
        elif self.kind == _CHECK_N_SAMPLES:
            observed = float(vals.size)
        elif self.kind == _CHECK_MEAN:
            import numpy as _np

            observed = float(_np.mean(vals))
        else:  # max_abs
            import numpy as _np

            observed = float(_np.max(_np.abs(vals)))
        return (self.lo <= observed <= self.hi), observed, self.label()

    def label(self) -> str:
        if self.kind == _CHECK_QUANTILE:
            return f"quantile(p{self.param:g})"
        return self.kind


def _as_finite_1d(sample: Any):
    import numpy as _np

    arr = _np.asarray(sample, dtype=_np.float64).ravel()
    return arr[_np.isfinite(arr)]


@dataclass(frozen=True)
class DomainSpec:
    """The named input distribution a constant was derived over.

    Two lines of defence, both fail-closed:

    1. **Declared-id equality** (cheap, always available): the consumer must NAME
       the distribution it is handing over.  ``domain_id`` mismatch refuses
       immediately, with no statistics needed.  This alone would have caught the
       (b) instance at the typing moment.
    2. **Statistical witness** (when a sample is supplied): the declared checks
       must hold, catching a consumer who labels the sample correctly by accident
       or incorrectly on purpose.
    """

    domain_id: str
    description: str
    checks: tuple[DomainCheck, ...] = ()
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.domain_id, str) or not self.domain_id.strip():
            raise GuardedConstantError("DomainSpec.domain_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise GuardedConstantError(
                "DomainSpec.description must be a non-empty string (what distribution is this?)"
            )
        if not isinstance(self.checks, tuple):
            raise GuardedConstantError("DomainSpec.checks must be a tuple (frozen)")
        for i, c in enumerate(self.checks):
            if not isinstance(c, DomainCheck):
                raise GuardedConstantError(
                    f"DomainSpec.checks[{i}] must be a DomainCheck, got {type(c).__name__}"
                )
        if not isinstance(self.aliases, tuple):
            raise GuardedConstantError("DomainSpec.aliases must be a tuple (frozen)")

    def accepts_id(self, domain_id: str) -> bool:
        return domain_id == self.domain_id or domain_id in self.aliases


# ---------------------------------------------------------------------------
# Derivation — the LIVE resolver.  Instance (a)'s cure.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DerivationRef:
    """A pointer to the in-repo derivation that PRODUCES this constant.

    ``callable_path`` is ``"pkg.mod:func"``.  ``invocation_required=True`` means
    resolving without invoking it is a :class:`StaleDerivation` refusal — which
    is exactly instance (a): a derivation that exists, is documented, and whose
    output was frozen into a literal, so nothing ever fails and the run cannot
    adapt.

    ``drift_tolerance_ratio`` bounds how far the live derivation may move from
    the declared/frozen value before the frozen value is declared ROTTED.  It
    is REQUIRED to carry ``tolerance_provenance``.
    """

    callable_path: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    invocation_required: bool = True
    drift_tolerance_ratio: float = 1.0
    tolerance_provenance: str = ""
    returns_tuple_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.callable_path, str) or ":" not in self.callable_path:
            raise GuardedConstantError(
                f"DerivationRef.callable_path={self.callable_path!r} must be 'pkg.mod:func'"
            )
        if not isinstance(self.kwargs, Mapping):
            raise GuardedConstantError("DerivationRef.kwargs must be a mapping")
        t = self.drift_tolerance_ratio
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not (t >= 1.0):
            raise GuardedConstantError(
                f"drift_tolerance_ratio={t!r} must be a number >= 1.0 (a multiplicative bound)"
            )
        if t > 1.0 and not self.tolerance_provenance.strip():
            raise GuardedConstantError(
                "a drift_tolerance_ratio > 1.0 REQUIRES tolerance_provenance — an unexplained "
                "tolerance is a bare constant guarding a bare constant"
            )
        if self.returns_tuple_index is not None and (
            isinstance(self.returns_tuple_index, bool)
            or not isinstance(self.returns_tuple_index, int)
            or self.returns_tuple_index < 0
        ):
            raise GuardedConstantError("returns_tuple_index must be a non-negative int or None")

    def load(self) -> Callable[..., Any]:
        mod_path, _, fn_name = self.callable_path.partition(":")
        try:
            mod = importlib.import_module(mod_path)
        except Exception as exc:  # pragma: no cover - import environment
            raise StaleDerivation(
                f"STALE-DERIVATION rule fired: cannot import {mod_path!r} for derivation "
                f"{self.callable_path!r} ({exc}). FIX: correct the callable_path, or drop the "
                "DerivationRef if the derivation genuinely no longer exists."
            ) from exc
        fn = getattr(mod, fn_name, None)
        if fn is None or not callable(fn):
            raise StaleDerivation(
                f"STALE-DERIVATION rule fired: {mod_path!r} has no callable {fn_name!r} "
                f"(declared as {self.callable_path!r}). FIX: correct the callable_path."
            )
        return fn

    def invoke(self, sample: Any) -> float:
        fn = self.load()
        out = fn(sample, **dict(self.kwargs))
        if self.returns_tuple_index is not None:
            out = out[self.returns_tuple_index]
        if isinstance(out, bool) or not isinstance(out, (int, float)):
            raise StaleDerivation(
                f"STALE-DERIVATION rule fired: {self.callable_path!r} returned "
                f"{type(out).__name__}, not a number. FIX: set returns_tuple_index if the "
                "derivation returns (value, provenance)."
            )
        return float(out)


@dataclass(frozen=True)
class Corroboration:
    """An INDEPENDENT route to the same constant.  Agreement is a feature.

    The ``margin_floor`` case: an fp32 cross-hardware portability bound (~0.096)
    and a GT-Lane-restricted margin p10 (0.104748) land on the same number
    (ratio 1.091x).  Representing that is the point — collapsing it to one
    "winner" would discard the L7 portability guard, which is real and
    load-bearing.
    """

    route_id: str
    value: float
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str) or not self.route_id.strip():
            raise GuardedConstantError("Corroboration.route_id must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise GuardedConstantError("Corroboration.value must be numeric")
        if not math.isfinite(float(self.value)):
            raise GuardedConstantError("Corroboration.value must be finite")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise GuardedConstantError("Corroboration.provenance must be a non-empty string")


# ---------------------------------------------------------------------------
# Result records.
# ---------------------------------------------------------------------------
STATUS_PASS = "PASS"
STATUS_VACUOUS = "VACUOUS"
STATUS_REFUSED = "REFUSED"

MODE_STRICT = "strict"
MODE_LOUD = "loud"
VALID_MODES = frozenset({MODE_STRICT, MODE_LOUD})


@dataclass(frozen=True)
class GuardCheckRecord:
    """One executed check.  ``ok=None`` means the check could not run (VACUOUS)."""

    axis: str
    name: str
    ok: bool | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class ResolvedGuardedConstant:
    """The guarded value + every check that ran, with its denominator."""

    constant_id: str
    value: float | int | str
    units: str
    role: str
    domain_id: str
    status: str
    checks: tuple[GuardCheckRecord, ...]
    violations: tuple[str, ...]
    lawref_manifest: Mapping[str, Any]
    derivation_invoked: bool
    derived_value: float | None
    corroboration_max_ratio: float | None
    incumbent_literal: float | int | str | None
    byte_identical_to_incumbent: bool | None
    resolution_hash: str

    # -- denominators (the vacuity cure) ------------------------------------
    @property
    def checks_run(self) -> int:
        return sum(1 for c in self.checks if c.ok is not None)

    @property
    def checks_declared(self) -> int:
        return len(self.checks)

    @property
    def behaviour_change(self) -> bool | None:
        """True when adopting this constant CHANGES the shipped value."""
        if self.byte_identical_to_incumbent is None:
            return None
        return not self.byte_identical_to_incumbent

    def to_provenanced(self) -> Any:
        """Bridge to the existing :class:`tac.witness_dsl.typed_config.Provenanced`.

        ``Provenanced`` (value + ladder class + unit + source + waiver) is the
        repo's existing config-side ladder model.  This bridge exists so the two
        are never parallel sources of truth: the guarded resolution EXPORTS into
        it rather than competing with it, and the ``unit`` string it receives is
        the canonical :class:`Units` rendering rather than free text.
        """
        from tac.witness_dsl.typed_config import ProvenanceClass, Provenanced

        ladder = str(self.lawref_manifest.get("ladder_class", ""))
        cls_map = {
            "derived_live": ProvenanceClass.DERIVED_LIVE,
            "derived_at_config": ProvenanceClass.DERIVED_AT_CONFIG,
            "measured_anchor": ProvenanceClass.MEASURED_ANCHOR,
            "hardcoded_waiver": ProvenanceClass.HARDCODED_WITH_WAIVER,
        }
        # A derivation that ACTUALLY RAN on the caller's data is class-1 by
        # definition, whatever the declaration's static rung said.
        pc = (
            ProvenanceClass.DERIVED_LIVE
            if self.derivation_invoked
            else cls_map.get(ladder, ProvenanceClass.HARDCODED_WITH_WAIVER)
        )
        waiver = None
        if pc is ProvenanceClass.HARDCODED_WITH_WAIVER:
            waiver = (
                f"guarded_constant {self.constant_id}: status={self.status} "
                f"checks_run={self.checks_run}/{self.checks_declared}; "
                + ("; ".join(self.violations) if self.violations else "no violations")
            )
        return Provenanced(
            value=self.value,
            provenance=pc,
            unit=self.units,
            source=f"guarded_constant:{self.constant_id}#{self.resolution_hash[:12]}",
            waiver=waiver,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "constant_id": self.constant_id,
            "value": self.value,
            "units": self.units,
            "role": self.role,
            "domain_id": self.domain_id,
            "status": self.status,
            "checks_run": self.checks_run,
            "checks_declared": self.checks_declared,
            "checks": [c.to_dict() for c in self.checks],
            "violations": list(self.violations),
            "derivation_invoked": self.derivation_invoked,
            "derived_value": self.derived_value,
            "corroboration_max_ratio": self.corroboration_max_ratio,
            "incumbent_literal": self.incumbent_literal,
            "byte_identical_to_incumbent": self.byte_identical_to_incumbent,
            "behaviour_change": self.behaviour_change,
            "resolution_hash": self.resolution_hash,
            "lawref": dict(self.lawref_manifest),
        }


# ---------------------------------------------------------------------------
# GuardedConstant.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GuardedConstant:
    """A LawRef-backed constant that refuses inappropriate use.

    ``lawref`` remains the single source of truth for the VALUE and its
    provenance ladder (#351).  This wrapper adds only the guards.
    """

    constant_id: str
    lawref: LawRef
    units: Units
    role: str
    domain: DomainSpec
    derivation: DerivationRef | None = None
    corroborations: tuple[Corroboration, ...] = ()
    corroboration_tolerance_ratio: float = 1.0
    corroboration_tolerance_provenance: str = ""
    incumbent_literal: float | int | str | None = None
    #: Identifier names under which ``incumbent_literal`` appears as a frozen
    #: literal in the codebase.  This is what makes the sister GATE
    #: DECLARATION-DRIVEN and therefore narrow: it looks for ``<name> = <value>``,
    #: never for the bare value.  ddm_gd5 (task #864) BUILT an auto-derived
    #: "is this reachable/called anywhere" detector and REFUTED it (the
    #: import-reachability predicate fires on 1229 of 3251 modules), so this type
    #: deliberately does NOT infer its own scan targets.
    literal_site_names: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.literal_site_names, tuple):
            raise GuardedConstantError("literal_site_names must be a tuple (frozen)")
        for nm in self.literal_site_names:
            if not isinstance(nm, str) or not nm.isidentifier():
                raise GuardedConstantError(
                    f"literal_site_names entry {nm!r} must be a Python identifier"
                )
        if not isinstance(self.constant_id, str) or not self.constant_id.strip():
            raise GuardedConstantError("constant_id must be a non-empty string")
        if not isinstance(self.lawref, LawRef):
            raise GuardedConstantError(
                f"lawref must be a LawRef (the #351 SoT for value+provenance), got "
                f"{type(self.lawref).__name__}"
            )
        if not isinstance(self.units, Units):
            raise GuardedConstantError("units must be a Units (use Units.parse('bytes/flip'))")
        if self.role not in VALID_ROLES:
            raise GuardedConstantError(
                f"role={self.role!r} must be one of {sorted(VALID_ROLES)!r}"
            )
        if not isinstance(self.domain, DomainSpec):
            raise GuardedConstantError("domain must be a DomainSpec")
        if self.derivation is not None and not isinstance(self.derivation, DerivationRef):
            raise GuardedConstantError("derivation must be a DerivationRef or None")
        if not isinstance(self.corroborations, tuple):
            raise GuardedConstantError("corroborations must be a tuple (frozen)")
        for i, c in enumerate(self.corroborations):
            if not isinstance(c, Corroboration):
                raise GuardedConstantError(f"corroborations[{i}] must be a Corroboration")
        t = self.corroboration_tolerance_ratio
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not (t >= 1.0):
            raise GuardedConstantError("corroboration_tolerance_ratio must be a number >= 1.0")
        if len(self.corroborations) >= 2 and t > 1.0 and not (
            self.corroboration_tolerance_provenance.strip()
        ):
            raise GuardedConstantError(
                "corroboration_tolerance_ratio > 1.0 with 2+ routes REQUIRES "
                "corroboration_tolerance_provenance"
            )

    # -- the guarded API ----------------------------------------------------
    def resolve(
        self,
        *,
        consumer_role: str,
        sample: Any | None = None,
        sample_domain_id: str | None = None,
        target_config_tags: Mapping[str, str] | None = None,
        mode: str = MODE_STRICT,
        repo_root: Path | str | None = None,
    ) -> ResolvedGuardedConstant:
        """Resolve for a NAMED consuming role.  Fail-closed (the WRITE path).

        ``consumer_role`` is mandatory and has no default on purpose: a consumer
        that will not say what it is using the number FOR is exactly the (c)
        instance, and a default would let it through silently.
        """
        rec = self._evaluate(
            consumer_role=consumer_role,
            sample=sample,
            sample_domain_id=sample_domain_id,
            target_config_tags=target_config_tags,
            repo_root=repo_root,
            raising=(mode == MODE_STRICT),
        )
        if mode not in VALID_MODES:
            raise GuardedConstantError(f"mode={mode!r} must be one of {sorted(VALID_MODES)!r}")
        return rec

    def audit(
        self,
        *,
        consumer_role: str | None = None,
        sample: Any | None = None,
        sample_domain_id: str | None = None,
        target_config_tags: Mapping[str, str] | None = None,
        repo_root: Path | str | None = None,
    ) -> ResolvedGuardedConstant:
        """Inspect without raising on guard violations (the READ path — LOUD).

        Every violation is listed in ``violations`` and ``status`` is
        :data:`STATUS_REFUSED`; a caller that ignores those is doing so visibly.
        """
        return self._evaluate(
            consumer_role=consumer_role,
            sample=sample,
            sample_domain_id=sample_domain_id,
            target_config_tags=target_config_tags,
            repo_root=repo_root,
            raising=False,
        )

    # -- internals ----------------------------------------------------------
    def _evaluate(
        self,
        *,
        consumer_role: str | None,
        sample: Any | None,
        sample_domain_id: str | None,
        target_config_tags: Mapping[str, str] | None,
        repo_root: Path | str | None,
        raising: bool,
    ) -> ResolvedGuardedConstant:
        checks: list[GuardCheckRecord] = []
        violations: list[str] = []

        def _fail(exc: GuardViolation) -> None:
            violations.append(str(exc))
            if raising:
                raise exc

        # ---- 1. ROLE -------------------------------------------------------
        if consumer_role is None:
            checks.append(
                GuardCheckRecord(
                    "role",
                    "consumer_role_declared",
                    None,
                    "no consumer_role supplied (audit path) — role check VACUOUS, not passed",
                )
            )
        elif consumer_role not in VALID_ROLES:
            _fail(
                RoleMisuse(
                    f"ROLE rule fired for {self.constant_id!r}: consumer_role="
                    f"{consumer_role!r} is not a known role {sorted(VALID_ROLES)!r}. "
                    "FIX: name the role this value plays at the call site."
                )
            )
            checks.append(
                GuardCheckRecord("role", "consumer_role_known", False, f"{consumer_role!r}")
            )
        elif consumer_role != self.role:
            _fail(
                RoleMisuse(
                    f"ROLE rule fired for {self.constant_id!r}: declared role "
                    f"{self.role!r}, consumed as {consumer_role!r}. "
                    f"WHY: {_role_note(self.role, consumer_role)}. "
                    f"FIX: obtain the {consumer_role!r} quantity from its own MEASURED "
                    f"source, or — if the substitution is genuinely intended — declare a "
                    f"separate GuardedConstant with role={consumer_role!r} and its own "
                    "provenance."
                )
            )
            checks.append(
                GuardCheckRecord(
                    "role", "role_match", False, f"declared={self.role} consumed={consumer_role}"
                )
            )
        else:
            checks.append(
                GuardCheckRecord("role", "role_match", True, f"role={self.role}")
            )

        # ---- 2. DOMAIN: declared id ----------------------------------------
        if sample_domain_id is None:
            checks.append(
                GuardCheckRecord(
                    "domain",
                    "declared_domain_id",
                    None,
                    "no sample_domain_id supplied — domain identity UNVERIFIED (VACUOUS); "
                    f"this constant was derived over {self.domain.domain_id!r}",
                )
            )
        elif not self.domain.accepts_id(sample_domain_id):
            _fail(
                DomainViolation(
                    f"DOMAIN rule fired for {self.constant_id!r}: derived over "
                    f"{self.domain.domain_id!r} ({self.domain.description}), but the "
                    f"caller supplied {sample_domain_id!r}. The VALUE may be fine; the "
                    "DOMAIN is not. "
                    f"FIX: re-derive this constant over {sample_domain_id!r}, or pass the "
                    f"{self.domain.domain_id!r} distribution."
                )
            )
            checks.append(
                GuardCheckRecord(
                    "domain",
                    "declared_domain_id",
                    False,
                    f"declared={self.domain.domain_id} supplied={sample_domain_id}",
                )
            )
        else:
            checks.append(
                GuardCheckRecord(
                    "domain", "declared_domain_id", True, f"{sample_domain_id}"
                )
            )

        # ---- 3. DOMAIN: statistical witness --------------------------------
        if sample is None:
            for c in self.domain.checks:
                checks.append(
                    GuardCheckRecord(
                        "domain",
                        f"stat:{c.label()}",
                        None,
                        "no sample supplied — statistical witness VACUOUS, not passed",
                    )
                )
        else:
            for c in self.domain.checks:
                try:
                    ok, observed, label = c.evaluate(sample)
                except DomainViolation as exc:
                    _fail(exc)
                    checks.append(
                        GuardCheckRecord("domain", f"stat:{c.label()}", False, str(exc))
                    )
                    continue
                detail = (
                    f"observed={observed:.6g} band=[{c.lo:.6g},{c.hi:.6g}] "
                    f"provenance={c.provenance}"
                )
                if not ok:
                    _fail(
                        DomainViolation(
                            f"DOMAIN rule fired for {self.constant_id!r}: the sample's "
                            f"{label} = {observed:.6g} is outside the band "
                            f"[{c.lo:.6g}, {c.hi:.6g}] measured for "
                            f"{self.domain.domain_id!r}. This is a DIFFERENT POPULATION. "
                            f"BAND PROVENANCE: {c.provenance}. "
                            "FIX: pass the distribution this constant was derived over, or "
                            "re-derive the constant on the population you actually have."
                        )
                    )
                checks.append(
                    GuardCheckRecord("domain", f"stat:{c.label()}", bool(ok), detail)
                )

        # ---- 4. LawRef value (delegated — #351 stays the SoT) --------------
        lawref_resolved: ResolvedConstant = _resolve_lawref(
            self.lawref, target_config_tags, repo_root=repo_root
        )
        value: float | int | str = lawref_resolved.value

        # ---- 5. STALENESS: the live derivation -----------------------------
        derivation_invoked = False
        derived_value: float | None = None
        if self.derivation is None:
            checks.append(
                GuardCheckRecord(
                    "staleness", "derivation_declared", None,
                    "no in-repo derivation declared for this constant — staleness VACUOUS",
                )
            )
        elif sample is None:
            if self.derivation.invocation_required:
                _fail(
                    StaleDerivation(
                        f"STALE-DERIVATION rule fired for {self.constant_id!r}: this "
                        f"constant declares a LIVE derivation "
                        f"({self.derivation.callable_path}) with invocation_required=True, "
                        "and it was resolved WITHOUT a sample, so the derivation did not "
                        "run and the frozen value was used. That is exactly the failure "
                        "class this type exists to catch: the derivation exists, is "
                        "documented, and never executes, so the value cannot adapt and "
                        "nothing ever goes red. "
                        "FIX: pass sample= (and sample_domain_id=) so the derivation runs, "
                        "or set invocation_required=False and record WHY the frozen value "
                        "is acceptable for this call site."
                    )
                )
                checks.append(
                    GuardCheckRecord(
                        "staleness", "derivation_invoked", False,
                        f"{self.derivation.callable_path} declared but NOT invoked",
                    )
                )
            else:
                checks.append(
                    GuardCheckRecord(
                        "staleness", "derivation_invoked", None,
                        f"{self.derivation.callable_path} declared with "
                        "invocation_required=False and no sample — VACUOUS",
                    )
                )
        else:
            try:
                derived_value = self.derivation.invoke(sample)
                derivation_invoked = True
            except StaleDerivation as exc:
                _fail(exc)
                checks.append(
                    GuardCheckRecord("staleness", "derivation_invoked", False, str(exc))
                )
            if derivation_invoked and isinstance(value, (int, float)) and not isinstance(value, bool):
                ratio = _ratio(float(value), float(derived_value))
                tol = float(self.derivation.drift_tolerance_ratio)
                ok = ratio is not None and ratio <= tol
                if not ok:
                    _fail(
                        StaleDerivation(
                            f"STALE-DERIVATION rule fired for {self.constant_id!r}: the "
                            f"live derivation {self.derivation.callable_path} returns "
                            f"{derived_value!r}, the declared/frozen value is {value!r} "
                            f"(ratio {ratio if ratio is not None else float('nan'):.6g}x > "
                            f"tolerance {tol:.6g}x). The frozen value has ROTTED. "
                            f"TOLERANCE PROVENANCE: "
                            f"{self.derivation.tolerance_provenance or '(none declared)'}. "
                            "FIX: adopt the derived value (a BEHAVIOUR CHANGE — record it), "
                            "or re-derive/justify the frozen one."
                        )
                    )
                checks.append(
                    GuardCheckRecord(
                        "staleness", "derivation_agrees_with_frozen", bool(ok),
                        f"frozen={value!r} derived={derived_value!r} "
                        f"ratio={ratio if ratio is not None else float('nan'):.6g} tol={tol:.6g}",
                    )
                )
                # The live derivation, once it has actually RUN on the caller's own
                # data, is the higher ladder rung (derived_live) — use it.
                value = float(derived_value)

        # ---- 6. CORROBORATION ----------------------------------------------
        corr_max_ratio: float | None = None
        if len(self.corroborations) < 2:
            checks.append(
                GuardCheckRecord(
                    "corroboration", "independent_routes", None,
                    f"{len(self.corroborations)} route(s) declared — agreement check "
                    "VACUOUS (needs 2+)",
                )
            )
        else:
            vals = [float(c.value) for c in self.corroborations]
            ratios = [
                r
                for i in range(len(vals))
                for j in range(i + 1, len(vals))
                if (r := _ratio(vals[i], vals[j])) is not None
            ]
            corr_max_ratio = max(ratios) if ratios else None
            tol = float(self.corroboration_tolerance_ratio)
            ok = corr_max_ratio is not None and corr_max_ratio <= tol
            if not ok:
                _fail(
                    CorroborationConflict(
                        f"CORROBORATION rule fired for {self.constant_id!r}: "
                        f"{len(vals)} independent routes disagree by "
                        f"{corr_max_ratio if corr_max_ratio is not None else float('nan'):.6g}x "
                        f"> tolerance {tol:.6g}x "
                        f"({', '.join(f'{c.route_id}={c.value!r}' for c in self.corroborations)}). "
                        "Two routes that disagree are a real conflict, not noise. "
                        "FIX: find which route's premise is wrong; do not average them."
                    )
                )
            checks.append(
                GuardCheckRecord(
                    "corroboration", "routes_agree", bool(ok),
                    f"routes={len(vals)} max_ratio="
                    f"{corr_max_ratio if corr_max_ratio is not None else float('nan'):.6g} "
                    f"tol={tol:.6g}",
                )
            )

        # ---- 7. migration honesty ------------------------------------------
        byte_identical: bool | None = None
        if self.incumbent_literal is not None:
            byte_identical = _exactly_equal(value, self.incumbent_literal)
            checks.append(
                GuardCheckRecord(
                    "migration", "byte_identical_to_incumbent", byte_identical,
                    f"resolved={value!r} incumbent={self.incumbent_literal!r} — "
                    + (
                        "adoption is byte-identical (safe migration)"
                        if byte_identical
                        else "adoption CHANGES the shipped value (BEHAVIOUR CHANGE — must be "
                        "measured, not assumed)"
                    ),
                )
            )

        ran = sum(1 for c in checks if c.ok is not None)
        if violations:
            status = STATUS_REFUSED
        elif ran == 0:
            status = STATUS_VACUOUS
        else:
            status = STATUS_PASS

        return ResolvedGuardedConstant(
            constant_id=self.constant_id,
            value=value,
            units=str(self.units),
            role=self.role,
            domain_id=self.domain.domain_id,
            status=status,
            checks=tuple(checks),
            violations=tuple(violations),
            lawref_manifest=lawref_resolved.to_dict(),
            derivation_invoked=derivation_invoked,
            derived_value=derived_value,
            corroboration_max_ratio=corr_max_ratio,
            incumbent_literal=self.incumbent_literal,
            byte_identical_to_incumbent=byte_identical,
            resolution_hash=self._resolution_hash(value, lawref_resolved, derived_value),
        )

    def _resolution_hash(
        self,
        value: float | int | str,
        lawref_resolved: ResolvedConstant,
        derived_value: float | None,
    ) -> str:
        """Deterministic digest over everything that DETERMINES the value.

        ``resolved_at`` is excluded (it is metadata), matching #351's own
        determinism contract, so two resolutions of the same declaration against
        the same artifacts hash identically.
        """
        payload = {
            "constant_id": self.constant_id,
            "equation_id": self.lawref.equation_id,
            "ladder_class": self.lawref.ladder_class,
            "units": str(self.units),
            "role": self.role,
            "domain_id": self.domain.domain_id,
            "value": value,
            "derived_value": derived_value,
            "inputs": [
                {"name": r.name, "kind": r.kind, "value": r.value, "sha256": r.sha256}
                for r in lawref_resolved.resolved_inputs
            ],
            "fallback_used": lawref_resolved.fallback_used,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Unit-safe arithmetic helpers (the (c) instance's cure at the call site).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Quantity:
    """A number that carries its dimension, so mixing dimensions REFUSES.

    ``5.91 bits/flip`` and ``0.60 bits/band_pixel`` are both real measurements
    of the same mechanism; comparing them directly changed a design conclusion.
    :meth:`compare_to` and the arithmetic operators refuse that comparison.
    """

    value: float
    units: Units

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise GuardedConstantError("Quantity.value must be numeric")
        if not isinstance(self.units, Units):
            raise GuardedConstantError("Quantity.units must be a Units")

    @classmethod
    def of(cls, value: float, units: str) -> Quantity:
        return cls(float(value), Units.parse(units))

    def compare_to(self, other: Quantity) -> float:
        """Return ``self.value / other.value`` — only if the dimensions match."""
        self.units.assert_same(other.units, where="Quantity.compare_to")
        if other.value == 0:
            raise GuardViolation(
                "COMPARISON rule fired: division by a zero-valued Quantity. "
                "FIX: guard the denominator at the call site."
            )
        return float(self.value) / float(other.value)

    def __add__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        self.units.assert_same(other.units, where="Quantity.__add__")
        return Quantity(float(self.value) + float(other.value), self.units)

    def __sub__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        self.units.assert_same(other.units, where="Quantity.__sub__")
        return Quantity(float(self.value) - float(other.value), self.units)

    def __str__(self) -> str:
        return f"{self.value!r} {self.units}"


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _ratio(a: float, b: float) -> float | None:
    """Symmetric multiplicative ratio max(a/b, b/a); None when undefined."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    if a == 0.0 and b == 0.0:
        return 1.0
    if a == 0.0 or b == 0.0:
        return None
    if (a > 0) != (b > 0):
        return None
    aa, bb = abs(a), abs(b)
    return max(aa / bb, bb / aa)


def _exactly_equal(a: Any, b: Any) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


__all__ = [
    "DIMENSIONLESS",
    "MODE_LOUD",
    "MODE_STRICT",
    "ROLE_BUDGET",
    "ROLE_COUNT",
    "ROLE_EXCHANGE_RATE",
    "ROLE_FRACTION",
    "ROLE_MECHANISM_COST",
    "ROLE_SCALE",
    "ROLE_THRESHOLD",
    "STATUS_PASS",
    "STATUS_REFUSED",
    "STATUS_VACUOUS",
    "VALID_MODES",
    "VALID_ROLES",
    "Corroboration",
    "CorroborationConflict",
    "DerivationRef",
    "DomainCheck",
    "DomainSpec",
    "DomainViolation",
    "GuardCheckRecord",
    "GuardViolation",
    "GuardedConstant",
    "GuardedConstantError",
    "Quantity",
    "ResolvedGuardedConstant",
    "RoleMisuse",
    "StaleDerivation",
    "UnitMismatch",
    "Units",
]
