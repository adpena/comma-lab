# SPDX-License-Identifier: MIT
"""Equation EVALUATORS — the executable eval surface the LawRef compiler calls.

Per operator design 2026-07-08 (task #351, the LawRef constant-compiler) + the
T5-crucible ORCHESTRATION_LEDGER requirement **T** (VALUE-PROVENANCE LADDER):
constants should be *programmed as canonical equations capturing system
dynamics, resolved at DSL-compile time into actual values*. This module is the
executable link that makes the triality's **equations leg** callable into the
**DSL leg**.

Why a NEW surface rather than reuse ``CanonicalEquation.python_callable_module_path``:
that field points at the law's math helper, but every such helper has its OWN
bespoke signature (``store_nothing_rate_term(bytes, ...)``, ``costate_vector(...)``,
...). The LawRef resolver needs ONE uniform calling convention —
``evaluator(resolved_inputs: Mapping[str, value]) -> value`` — so it can resolve
a law's typed InputRefs and hand them off without knowing the law's arity. An
equation OPTS IN by registering such an evaluator here (a thin adapter over its
own math helper), keyed by its canonical ``equation_id``.

The registry is process-global + idempotent-safe (re-registering the SAME
callable object is a no-op; re-registering a DIFFERENT object for an existing
id raises unless ``overwrite=True``). It is intentionally decoupled from the
JSONL ``registry`` persistence: an evaluator is code, not durable state.

Cross-references:
  * ``tac.witness_dsl.lawref`` — the consumer (the LawRef resolver).
  * ``tac.canonical_equations.equation`` — the ``CanonicalEquation`` schema
    whose ``equation_id`` keys this registry.
  * CLAUDE.md "Canonical equations + models registry" + "The Triality —
    DAG ↔ DSL ↔ equations" non-negotiables.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any

# Canonical equation_id pattern (mirrors equation.py::_EQUATION_ID_RE so an
# evaluator can never be registered under a non-canonical id).
_EQUATION_ID_RE = re.compile(r"^[a-z][a-z0-9_]*_v\d+$")

# ln(5) — the Maslov two-class dequantization scale. Bit-stable across hosts
# (IEEE-754 double math.log is deterministic); equals the ``ln5`` field stored
# in the tau-confirm artifact (verified 2026-07-08).
_LN5 = math.log(5.0)


class EvaluatorError(ValueError):
    """Raised on evaluator registration / lookup contract violations."""


class EvaluatorNotRegisteredError(KeyError):
    """Raised when a LawRef references an equation_id with no evaluator."""


# The process-global evaluator registry. equation_id -> callable(inputs)->value.
_EVALUATORS: dict[str, Callable[[Mapping[str, Any]], Any]] = {}


def register_evaluator(
    equation_id: str,
    fn: Callable[[Mapping[str, Any]], Any],
    *,
    overwrite: bool = False,
) -> None:
    """Register ``fn`` as the evaluator for ``equation_id``.

    ``fn`` MUST accept a single ``Mapping[str, value]`` (the resolved inputs)
    and return the constant's value. Idempotent for the SAME callable object;
    a DIFFERENT callable for an existing id raises unless ``overwrite=True``.
    """
    if not isinstance(equation_id, str) or not _EQUATION_ID_RE.match(equation_id):
        raise EvaluatorError(
            f"equation_id={equation_id!r} must match snake_case_vN "
            "(e.g. 'forfeit_matched_exit_v1')"
        )
    if not callable(fn):
        raise EvaluatorError(f"evaluator for {equation_id!r} must be callable, got {type(fn).__name__}")
    existing = _EVALUATORS.get(equation_id)
    if existing is not None and existing is not fn and not overwrite:
        raise EvaluatorError(
            f"evaluator for {equation_id!r} already registered with a different "
            "callable; pass overwrite=True to replace (rare — evaluators are code)"
        )
    _EVALUATORS[equation_id] = fn


def has_evaluator(equation_id: str) -> bool:
    """True iff an evaluator is registered for ``equation_id``."""
    return equation_id in _EVALUATORS


def get_evaluator(equation_id: str) -> Callable[[Mapping[str, Any]], Any]:
    """Return the registered evaluator, or raise EvaluatorNotRegisteredError."""
    try:
        return _EVALUATORS[equation_id]
    except KeyError as exc:
        raise EvaluatorNotRegisteredError(
            f"no evaluator registered for equation_id={equation_id!r}; "
            "register one via tac.canonical_equations.evaluators.register_evaluator "
            "(or call populate_lawref_evaluators() for the built-in laws)"
        ) from exc


def resolve_equation_value(equation_id: str, inputs: Mapping[str, Any]) -> Any:
    """Look up ``equation_id``'s evaluator and call it with ``inputs``.

    This is the single entry point the LawRef resolver uses. Deterministic:
    the value depends only on ``inputs`` (+ the pure-math evaluator).
    """
    if not isinstance(inputs, Mapping):
        raise EvaluatorError("inputs must be a Mapping[str, value]")
    fn = get_evaluator(equation_id)
    return fn(inputs)


def registered_equation_ids() -> tuple[str, ...]:
    """Sorted tuple of all equation_ids with a registered evaluator."""
    return tuple(sorted(_EVALUATORS))


# ---------------------------------------------------------------------------
# Built-in evaluators for the 3 first laws (all have T5-crucible artifacts).
# Each is a PURE function of its resolved inputs so the resolved value is
# deterministic + bit-reproducible from the cited inputs.
# ---------------------------------------------------------------------------
def eval_forfeit_matched_exit_s_star(inputs: Mapping[str, Any]) -> float:
    """s* = nu * forfeit  (forfeit-matched TAU->FIN exit trigger, P-CT3).

    inputs: {"nu": S/ep decay-rate (P-CT1 exponential-meat fit),
             "forfeit": S recovery forfeited by one-cadence-late fire (P-CT3)}.
    """
    nu = float(inputs["nu"])
    forfeit = float(inputs["forfeit"])
    return nu * forfeit


def eval_tau_star_maslov_quantile(inputs: Mapping[str, Any]) -> float:
    """tau* = m_q / ln5  (Maslov two-scale dequantization interface half-width).

    inputs: {"m_q": flip-margin quantile (GT-margin below which q of flip mass
             sits; the quantile CONVENTION is a config-conditional choice),
             "ln5": ln(5) (optional; defaults to math.log(5))}.
    """
    m_q = float(inputs["m_q"])
    ln5 = float(inputs.get("ln5", _LN5))
    return m_q / ln5


def eval_critical_nucleus_release_r_star(inputs: Mapping[str, Any]) -> float:
    """r* = coeff * sigma_eff  (B18 island-release radius; coeff = 0.674*sqrt(2) ~= 0.95).

    inputs: {"coeff": release coefficient (default 0.95 per the operator law
             statement), "sigma_eff": effective interface width (px)}.
    """
    coeff = float(inputs.get("coeff", 0.95))
    sigma_eff = float(inputs["sigma_eff"])
    return coeff * sigma_eff


# ---------------------------------------------------------------------------
# crucible_v6 migration evaluators (#351 follow-up). The CONSUMED trio (τ_end /
# β-pin / LR-pin) + the ν-family / persistence-bar / adaptive-ε LIBRARY laws.
#
# TYPE DISCIPLINE (value-identity is the LAW): the emitted launch.sh renders each
# flag value with ``str(value)`` — ``str(1000) != str(1000.0)`` — so a passthrough
# evaluator MUST return the input UNCOERCED (an int stays int, a float stays
# float) or it would silently break the byte-identity gate. Only the arithmetic
# laws (settle / tail-cycle) return a computed float, matching their float anchors.
# ---------------------------------------------------------------------------
def eval_tau_end_knee_launch(inputs: Mapping[str, Any]) -> Any:
    """τ_end = the P-TAU2 knee probe's chosen ``launch_tau`` (measured anchor).

    inputs: {"launch_tau": the artifact's launch_tau field (0.31; inside the knee
             band [0.19072, 0.54294] and ≈ the mod32cap ep650-best τ=0.3098)}.
    A measured-anchor passthrough: the value IS the recorded launch τ. Uncoerced
    so a float stays float (str-render byte-identity).
    """
    return inputs["launch_tau"]


def eval_hosc_beta_fireband_pin(inputs: Mapping[str, Any]) -> Any:
    """β_end = the shipped hosc-β anneal endpoint PIN (derived-at-config).

    inputs: {"beta_end": the pinned β endpoint (10.0)}. The linear-replica law
    (β LINEAR over the shared --anneal-epochs den) MOTIVATES the pin — matching
    the control's β(ep) slope on [1, muon-freeze] to ≤0.1% needs an endpoint
    ≈10 at den 3000 — but the SHIPPED value is the pinned 10.0 (the law's
    endpoint is an approximation; the pin is the value). Passthrough, uncoerced.
    """
    return inputs["beta_end"]


def eval_lr_control_denominator(inputs: Mapping[str, Any]) -> Any:
    """LR-anneal denominator = the CONTROL vehicle's own --epochs den (derived-at-config).

    inputs: {"control_den": the mod32cap control's LR-cosine denominator (1000)}.
    The LR cosine is the third shared-den sibling; unlike β (LINEAR, endpoint-
    rephasable) a shallow den-3000 cosine cannot reproduce the control's deep
    den-1000 descent by endpoint (curvature differs), so LR gets its OWN
    denominator = the control's den (1000) → reproduces control LR(ep) on
    [1,726] bit-identically. Passthrough, uncoerced (INT stays int → str "1000").
    """
    return inputs["control_den"]


def eval_lr_hold_frac_no_hold(inputs: Mapping[str, Any]) -> Any:
    """LR-hold-frac = 1.0 = NO hold (derived-at-config).

    inputs: {"hold_frac": 1.0}. The control's Muon freeze (726) < the LR den
    (1000), so the control never held LR pre-freeze → hold-frac 1.0 (no hold) is
    the bit-identical-cosine choice. Passthrough, uncoerced.
    """
    return inputs["hold_frac"]


def eval_settle_window(inputs: Mapping[str, Any]) -> float:
    """settle window = coeff / ν  (P-CT1 exponential-meat settle law).

    inputs: {"coeff": settle multiple (3 e-folds), "nu": the stage's S/ep decay
             rate}. Bit-reproduces the artifact's stored ``settle_3_over_nu_ep``
             (3.0/ν). LIBRARY law (the schedule-derivation machinery; not an
             emitted crucible_v6 flag).
    """
    return inputs["coeff"] / inputs["nu"]


def eval_tail_cycle_floor(inputs: Mapping[str, Any]) -> float:
    """tail-cycle floor = coeff/ν + tail_extra  (P-CT1 settle + one dwell floor).

    inputs: {"coeff": settle multiple (3), "nu": stage decay rate,
             "tail_extra": the dwell floor added past settle (150 ep)}.
    Bit-reproduces the artifact's stored ``tail_cycle_floor_ep``. LIBRARY law.
    """
    return inputs["coeff"] / inputs["nu"] + inputs["tail_extra"]


def eval_conley_absolute_bar(inputs: Mapping[str, Any]) -> Any:
    """Conley absolute persistence bar = the P-CON fitted logit threshold (B17′).

    inputs: {"s_fit_logit": the fitted absolute survival bar (1.7504924172 @
             Tau-stage / 1.3017706202 @ MuonBest — near τ-INDEPENDENT while τ
             varied 4.3×)}. Passthrough of the fitted-bar constant. LIBRARY law
             (a certificate-lever param; not an emitted crucible_v6 flag).
    """
    return inputs["s_fit_logit"]


def eval_adaptive_eps_saturation_alarm(inputs: Mapping[str, Any]) -> Any:
    """adaptive-ε saturation ALARM threshold = ε_raw sustained-above clamp.

    inputs: {"alarm_threshold": the sustained-ε_raw saturation alarm (0.7; the
             |c_a(τ)| growth that drove it RELAXES at τ_end 0.31 but the alarm is
             KEPT — v6 §row-1)}. Passthrough. LIBRARY law (a control-loop alarm
             constant; not an emitted crucible_v6 flag).
    """
    return inputs["alarm_threshold"]


# Canonical equation_id -> evaluator for the built-in laws.
LAWREF_BUILTIN_EVALUATORS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "forfeit_matched_exit_v1": eval_forfeit_matched_exit_s_star,
    "tau_star_maslov_quantile_v1": eval_tau_star_maslov_quantile,
    "critical_nucleus_release_v1": eval_critical_nucleus_release_r_star,
    # crucible_v6 migration (#351 follow-up) — CONSUMED trio + LR-hold:
    "tau_end_knee_launch_v1": eval_tau_end_knee_launch,
    "hosc_beta_fireband_pin_v1": eval_hosc_beta_fireband_pin,
    "lr_control_denominator_v1": eval_lr_control_denominator,
    "lr_hold_frac_no_hold_v1": eval_lr_hold_frac_no_hold,
    # crucible_v6 migration — LIBRARY laws (bit-match tested; not emitted flags):
    "settle_window_v1": eval_settle_window,
    "tail_cycle_floor_v1": eval_tail_cycle_floor,
    "conley_absolute_bar_v1": eval_conley_absolute_bar,
    "adaptive_eps_saturation_alarm_v1": eval_adaptive_eps_saturation_alarm,
}


def populate_lawref_evaluators() -> tuple[str, ...]:
    """Register the 3 built-in law evaluators; return their equation_ids.

    Idempotent (register_evaluator is a no-op for the same callable). The
    LawRef resolver calls this lazily so import order never matters.
    """
    for eqid, fn in LAWREF_BUILTIN_EVALUATORS.items():
        register_evaluator(eqid, fn)
    return tuple(sorted(LAWREF_BUILTIN_EVALUATORS))


__all__ = [
    "LAWREF_BUILTIN_EVALUATORS",
    "EvaluatorError",
    "EvaluatorNotRegisteredError",
    "eval_adaptive_eps_saturation_alarm",
    "eval_conley_absolute_bar",
    "eval_critical_nucleus_release_r_star",
    "eval_forfeit_matched_exit_s_star",
    "eval_hosc_beta_fireband_pin",
    "eval_lr_control_denominator",
    "eval_lr_hold_frac_no_hold",
    "eval_settle_window",
    "eval_tail_cycle_floor",
    "eval_tau_end_knee_launch",
    "eval_tau_star_maslov_quantile",
    "get_evaluator",
    "has_evaluator",
    "populate_lawref_evaluators",
    "register_evaluator",
    "registered_equation_ids",
    "resolve_equation_value",
]
