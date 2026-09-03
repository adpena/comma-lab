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


def eval_v9_hosc_beta_endpoint(inputs: Mapping[str, Any]) -> Any:
    """β_end = beta_start * 2**dyadic_refinements (V9 hosc endpoint, derived-at-config).

    inputs: {"beta_start": 1.0, "dyadic_refinements": 3} -> 8.0. Delegates to the
    defining function so the LawRef and the equation cannot drift apart; the
    import is lazy because that module imports ``tac.witness_dsl.lawref``.
    Registered 2026-09-04: ``v9_hosc_beta_endpoint_v1`` had NO evaluator anywhere,
    so ``resolve_v9_hosc_beta_endpoint()`` could never succeed (red test since
    fba5361ed; an equation_id naming an evaluator nobody registers is a split bank).
    """
    from tac.canonical_equations.v9_hosc_beta_endpoint_20260715 import (
        v9_hosc_beta_endpoint,
    )

    return v9_hosc_beta_endpoint(inputs["beta_start"], inputs["dyadic_refinements"])


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


def eval_cgauge_whitney_moddim(inputs: Mapping[str, Any]) -> int:
    """mod-dim* = 2d+1 (+gauge_margin) on the rank-d separatrix manifold (#223 Law 1).

    inputs: {"intrinsic_dim": d (measured; 8 doubly-measured n600),
             "gauge_margin": zero-mode slack (default-usage 2)}.
    LawRef-executable form of ``cgauge_whitney_moddim_v1`` (2026-07-11).
    """
    from tac.canonical_equations.cgauge_parametrization_optima_20260711 import (
        whitney_mod_dim,
    )

    return whitney_mod_dim(
        int(inputs["intrinsic_dim"]), gauge_margin=int(inputs.get("gauge_margin", 2))
    )


def eval_cgauge_parabolic_along_tangent(inputs: Mapping[str, Any]) -> float:
    """nu_along* = sqrt(nu_across) — parabolic-scaling wedge law (#223 Law 3).

    inputs: {"nu_across": the bank's across-edge max frequency (live 64)}.
    LawRef-executable form of ``cgauge_curvelet_parabolic_bank_v1`` (2026-07-11).
    """
    from tac.canonical_equations.cgauge_parametrization_optima_20260711 import (
        parabolic_along_tangent_allocation,
    )

    return parabolic_along_tangent_allocation(float(inputs["nu_across"]))


def eval_cgauge_beta2_window(inputs: Mapping[str, Any]) -> tuple[float, float]:
    """Adam beta2 admissible window [1-1/S, 1-3/(T_c S)] (#223 Law 4).

    inputs: {"steps_per_epoch": S (n600/accum-8 => 75),
             "curvature_timescale_epochs": T_c (assumed 100 from the anneal scale)}.
    LawRef-executable form of ``cgauge_beta2_window_v1`` (2026-07-11). Returns the
    WINDOW; the point value inside it stays a measured anchor (#222 A/B arbiter).
    """
    from tac.canonical_equations.cgauge_parametrization_optima_20260711 import (
        beta2_window,
    )

    return beta2_window(
        int(inputs["steps_per_epoch"]),
        curvature_timescale_epochs=float(inputs.get("curvature_timescale_epochs", 100.0)),
    )


def eval_margin_band_satisficing_threshold(inputs: Mapping[str, Any]) -> float:
    """m_safe = headroom * delta_R for the MarginBandSatisficing DSL lever.

    ``delta_R`` is a MEASURED artifact input and ``headroom`` is DERIVED at
    config time.  Keeping this adapter in the LawRef evaluator registry makes
    the canonical equation executable instead of leaving it as memo prose.
    """
    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        margin_safe_threshold,
    )

    return margin_safe_threshold(
        float(inputs["delta_r"]),
        float(inputs["headroom"]),
    )


def eval_isoperimetric_birth_weight(inputs: Mapping[str, Any]) -> float:
    """Resolve the island-birth support radius from sealed per-class ``P/A`` geometry.

    ``absolute_scale`` is the positive UNMEASURED DSL authoring scale; ``class_p_over_a`` and
    ``reference_p_over_a`` are content-addressed receipt anchors.  The underlying helper validates
    every input as finite and strictly positive.
    """
    from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
        isoperimetric_birth_weight,
    )

    return isoperimetric_birth_weight(
        float(inputs["absolute_scale"]),
        float(inputs["class_p_over_a"]),
        float(inputs["reference_p_over_a"]),
    )


def eval_v9_scientific_declaration(inputs: Mapping[str, Any]) -> Any:
    """Resolve one typed V9 treatment declaration without erasing its law.

    The three isolation arms use already-registered canonical mechanism equations
    (taper, horizon-margin, and step-native activation) for several scalar
    declarations.  Their *mechanism* equations are richer than a scalar evaluator;
    this adapter returns the explicitly named scalar ``value`` while the LawRef's
    typed inputs retain the treatment/config conditionality.  It is deliberately
    not a generic equation id: only the three canonical mechanism ids below opt in.

    The horizon weight declaration is the requested loss-share cap.  The trainer
    consumes it as a DERIVED-LIVE request and freezes the actual measured weight in
    ``hwm_v9_stage_share_boundary.v1`` at the configured boundary.
    """
    if "value" not in inputs:
        raise EvaluatorError("V9 scientific declaration requires a 'value' input")
    return inputs["value"]


def eval_dsl_custodied_scalar_identity(inputs: Mapping[str, Any]) -> Any:
    """dsl_custodied_scalar_identity_v1 — NON-DERIVATIONAL value custody (#351/#332).

    Preserves the declared value's bytes (bool-as-int / int / float / string) for
    MEASURED or WAIVED constants; it CANNOT manufacture scientific authority — the
    LawRef's ``ladder_class`` records which rung the value actually holds (class-4
    ``hardcoded_waiver`` for the #332 flag-custody backfill), and the equation is
    explicitly an identity: value out == value in, no derivation claimed.  Refuses
    a missing value and non-finite floats (per #351: "non-finite literals ...
    refuse").
    """
    if "value" not in inputs:
        raise EvaluatorError("dsl_custodied_scalar_identity requires a 'value' input")
    value = inputs["value"]
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluatorError("dsl_custodied_scalar_identity refuses non-finite values")
    return value


def eval_warm_start_schedule_reconstruction(inputs: Mapping[str, Any]) -> int:
    """warm_start_schedule_reconstruction_v1 — recompute a warm-start schedule boundary.

    The law (2026-07-16, c2_surgical_warm): a weights-only warm start MUST reproduce the
    checkpoint's schedule plant, so every emitted ``--*-start-epoch`` boundary is a total
    function of named inputs — ``mode`` selects the derivation:

      * ``config_of_record``     -> int(config_of_record_value)   (the checkpoint's launch.sh value)
      * ``run_length_exclusion`` -> int(run_epochs) + 1            (l7 never-runs: start = epochs+1 —
                                                                    the trainer's epoch loop is
                                                                    range(start, epochs+1) INCLUSIVE,
                                                                    so start == epochs would RUN l7 on
                                                                    the final epoch [the trainer's own
                                                                    documented off-by-one]; the mod32cap
                                                                    config of record parks l7 at 1001
                                                                    with epochs=1000 — TRUE never.
                                                                    Amended 2026-07-16 by the c2
                                                                    adversarial review; the original
                                                                    registration returned run_epochs
                                                                    and reproduced the off-by-one.)
      * ``resume_plus_window``   -> resume_epoch + re_anchor_window (the surgical engage boundary)
      * ``original_plant_end``   -> int(original_schedule_epochs)  (the backstop cap at plant end)
    """
    mode = str(inputs.get("mode", "")).strip()
    if mode == "config_of_record":
        return int(inputs["config_of_record_value"])
    if mode == "run_length_exclusion":
        return int(inputs["run_epochs"]) + 1
    if mode == "resume_plus_window":
        return int(inputs["resume_epoch"]) + int(inputs["re_anchor_window"])
    if mode == "original_plant_end":
        return int(inputs["original_schedule_epochs"])
    raise EvaluatorError(
        f"warm_start_schedule_reconstruction_v1: unknown mode {mode!r} (must be one of "
        "config_of_record | run_length_exclusion | resume_plus_window | original_plant_end)")


def eval_adam_v_variance_warmup_length(inputs: Mapping[str, Any]) -> int:
    """adam_v_variance_warmup_length_v1 — beta2-derived LR-rewarmup window (epochs).

    warmup_epochs = ceil(c/(1-beta2) / steps_per_epoch); c defaults to 2 (RAdam
    variance-rectification rationale); c=1 reproduces the sister memory bound
    ``rewarmup_beta2_memory_window_v1`` exactly. Registered 2026-07-17
    (p0_resume_warmup_geometry item 2); consumed by the ResumeLRWarmup DSL lever.
    """
    from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
        DEFAULT_C,
        adam_v_variance_warmup_epochs,
    )

    return adam_v_variance_warmup_epochs(
        float(inputs["beta2"]),
        int(inputs["steps_per_epoch"]),
        c=float(inputs.get("c", DEFAULT_C)),
    )
def eval_ema_decay_run_geometry(inputs: Mapping[str, Any]) -> float:
    """ema_decay_run_geometry_v1 — the EMA decay LAW from run geometry (ARM-C, SPEC_v10 §13.3).

    The incumbent 0.997/update is a Quantizr per-step-minibatch provenance that does NOT
    transfer to the deterministic full-batch regime (1 update/epoch: the noise-averaging
    rationale vanishes; MEASURED on the live c2 run: warmup 2/(1-d)=667 updates ~ ep1318 of a
    1400-ep run — the shadow spends the whole run inside warmup; ~64% warm-start seed @ep800).
    The LAW: the exact geometric identities of a constant-decay EMA over ``U`` updates,

        seed_fraction   eps = d**U           (weight the initial shadow seed retains)
        warmup_updates  W   = 2/(1-d)        (the registered two-time-constant warmup)
        warmup_fraction phi = W/U = 2/((1-d)*U)

    inverted for the quantity the config wants (``mode`` selects):

      * ``decay_from_seed_fraction``   -> d = eps**(1/U)      (pin the terminal seed weight)
      * ``decay_from_warmup_fraction`` -> d = 1 - 2/(phi*U)   (pin where warmup completes)
      * ``warmup_fraction_from_decay`` -> phi = 2/((1-d)*U)   (audit an incumbent decay)
      * ``seed_fraction_from_decay``   -> eps = d**U          (audit an incumbent decay)

    Exact closed forms (no approximation); fail-closed on out-of-domain inputs.
    ``mode`` accepts the string name OR the numeric code 1..4 (in the listed order) because
    LawRef literal inputs are numeric-only — the DSL lever passes the code.
    """
    _MODE_CODES = {1: "decay_from_seed_fraction", 2: "decay_from_warmup_fraction",
                   3: "warmup_fraction_from_decay", 4: "seed_fraction_from_decay"}
    mode_raw = inputs.get("mode", "")
    if isinstance(mode_raw, (int, float)) and not isinstance(mode_raw, bool):
        mode = _MODE_CODES.get(int(mode_raw), f"<invalid code {mode_raw}>")
    else:
        mode = str(mode_raw).strip()
    u = int(inputs["updates_per_run"])
    if u <= 0:
        raise EvaluatorError(f"ema_decay_run_geometry_v1: updates_per_run must be > 0, got {u}")
    if mode == "decay_from_seed_fraction":
        eps = float(inputs["target_seed_fraction"])
        if not 0.0 < eps < 1.0:
            raise EvaluatorError(
                f"ema_decay_run_geometry_v1: target_seed_fraction must be in (0,1), got {eps}")
        return float(eps ** (1.0 / u))
    if mode == "decay_from_warmup_fraction":
        phi = float(inputs["warmup_fraction"])
        if phi <= 0.0:
            raise EvaluatorError(
                f"ema_decay_run_geometry_v1: warmup_fraction must be > 0, got {phi}")
        if phi * u <= 2.0:
            raise EvaluatorError(
                f"ema_decay_run_geometry_v1: warmup_fraction*updates_per_run must exceed 2 "
                f"(got {phi * u:.3f}) — otherwise d <= 0 (no valid EMA decay)")
        return float(1.0 - 2.0 / (phi * u))
    if mode == "warmup_fraction_from_decay":
        d = float(inputs["ema_decay"])
        if not 0.0 <= d < 1.0:
            raise EvaluatorError(f"ema_decay_run_geometry_v1: ema_decay must be in [0,1), got {d}")
        return float(2.0 / ((1.0 - d) * u))
    if mode == "seed_fraction_from_decay":
        d = float(inputs["ema_decay"])
        if not 0.0 <= d < 1.0:
            raise EvaluatorError(f"ema_decay_run_geometry_v1: ema_decay must be in [0,1), got {d}")
        return float(d ** u)
    raise EvaluatorError(
        f"ema_decay_run_geometry_v1: unknown mode {mode!r} (must be one of "
        "decay_from_seed_fraction | decay_from_warmup_fraction | warmup_fraction_from_decay | "
        "seed_fraction_from_decay)")


def eval_cpu_cuda_score_gap(inputs: Mapping[str, Any]) -> float:
    """Return the registered ``CUDA - CPU`` score delta for one archive.

    The sign is deliberately not predicted from lineage.  The LC2 anchor added
    by ddm_cn4 has the opposite sign from the older HNeRV cluster, so callers
    must supply both measured per-axis scores for the exact same archive bytes.
    """

    cpu_score = float(inputs["score_cpu"])
    cuda_score = float(inputs["score_cuda"])
    if (
        not math.isfinite(cpu_score)
        or not math.isfinite(cuda_score)
        or cpu_score < 0.0
        or cuda_score < 0.0
    ):
        raise EvaluatorError(
            "cpu_cuda_score_gap_v1: scores must be finite and non-negative"
        )
    return cuda_score - cpu_score


def eval_realization_breakeven_bytes(inputs: Mapping[str, Any]) -> float:
    """Return the exact contest-rate byte budget for realized score recovery."""

    realized_recovery_s = float(inputs["realized_recovery_s"])
    if not math.isfinite(realized_recovery_s) or realized_recovery_s < 0.0:
        raise EvaluatorError(
            "realization_breakeven_bytes_v1: realized_recovery_s must be finite and non-negative"
        )
    return realized_recovery_s / (25.0 / 37_545_489.0)


def eval_radius2_multistart_singleton_escape(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize whether broadened radius-2 starts escaped a singleton optimum.

    This is an apparatus law, not a score predictor.  Its output remains
    advisory unless the supplied score axis is an exact contest authority.
    """

    pair_count = int(inputs["pair_count"])
    accepted_rows = int(inputs["accepted_rows"])
    score_before = float(inputs["score_before"])
    score_after = float(inputs["score_after"])
    d_pose_before = float(inputs["d_pose_before"])
    d_pose_after = float(inputs["d_pose_after"])
    if pair_count <= 0:
        raise EvaluatorError(
            "radius2_multistart_singleton_escape_v1: pair_count must be positive"
        )
    if accepted_rows < 0 or accepted_rows > pair_count:
        raise EvaluatorError(
            "radius2_multistart_singleton_escape_v1: accepted_rows must be in [0, pair_count]"
        )
    for name, value in (
        ("score_before", score_before),
        ("score_after", score_after),
        ("d_pose_before", d_pose_before),
        ("d_pose_after", d_pose_after),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise EvaluatorError(
                f"radius2_multistart_singleton_escape_v1: {name} must be finite and non-negative"
            )
    return {
        "escaped": accepted_rows > 0 and score_after < score_before,
        "accepted_fraction": accepted_rows / pair_count,
        "score_reduction": score_before - score_after,
        "d_pose_reduction": d_pose_before - d_pose_after,
    }


def eval_local_model_step_resolvability_ratio(inputs: Mapping[str, Any]) -> float:
    """Return modeled step magnitude divided by measured forward-mismatch floor.

    A local-model admission decision is instrument-resolved only above one.  The
    ratio deliberately uses the most charitable absolute step magnitude; a
    signed improvement smaller than the model's own forward mismatch cannot be
    distinguished from instrument error.
    """

    step = abs(float(inputs["predicted_step_magnitude"]))
    floor = float(inputs["forward_mismatch_floor"])
    if not math.isfinite(step):
        raise EvaluatorError("local_model_step_resolvability_ratio_v1: predicted_step_magnitude must be finite")
    if not math.isfinite(floor) or floor <= 0.0:
        raise EvaluatorError(
            "local_model_step_resolvability_ratio_v1: forward_mismatch_floor must be finite and positive"
        )
    return step / floor


def eval_receiver_pose_semantic_preservation_ratio(inputs: Mapping[str, Any]) -> float:
    """Return candidate/base PoseNet distortion for a matched receiver row."""

    base = float(inputs["base_d_pose"])
    candidate = float(inputs["candidate_d_pose"])
    if not math.isfinite(base) or base <= 0.0:
        raise EvaluatorError("receiver_pose_semantic_preservation_ratio_v1: base_d_pose must be finite and positive")
    if not math.isfinite(candidate) or candidate < 0.0:
        raise EvaluatorError(
            "receiver_pose_semantic_preservation_ratio_v1: candidate_d_pose must be finite and non-negative"
        )
    return candidate / base


def eval_pose_stack_exact_budget(inputs: Mapping[str, Any]) -> float:
    """Invert the exact contest Pose term after Seg credit and byte cost.

    ``seg_credit_s`` is a non-negative improvement allowance in score units.
    ``archive_delta_bytes`` is signed: positive means added bytes and therefore
    consumes allowance; negative means a rate saving.  A negative net allowance
    is refused because no non-negative Pose degradation can satisfy it.
    """

    base = float(inputs["base_d_pose"])
    seg_credit = float(inputs["seg_credit_s"])
    archive_delta_bytes = float(inputs["archive_delta_bytes"])
    for name, value in (
        ("base_d_pose", base),
        ("seg_credit_s", seg_credit),
        ("archive_delta_bytes", archive_delta_bytes),
    ):
        if not math.isfinite(value):
            raise EvaluatorError(f"pose_stack_exact_budget_v1: {name} must be finite")
    if base < 0.0:
        raise EvaluatorError("pose_stack_exact_budget_v1: base_d_pose must be non-negative")
    if seg_credit < 0.0:
        raise EvaluatorError("pose_stack_exact_budget_v1: seg_credit_s must be non-negative")
    net_allowance = seg_credit - 25.0 * archive_delta_bytes / 37_545_489.0
    if net_allowance < 0.0:
        raise EvaluatorError(
            "pose_stack_exact_budget_v1: byte cost exceeds Seg credit; no non-negative Pose degradation budget exists"
        )
    return ((math.sqrt(10.0 * base) + net_allowance) ** 2) / 10.0 - base


def eval_receiver_lattice_leakage_exponent(inputs: Mapping[str, Any]) -> float:
    """Fit the OLS log-log exponent of leakage against positive amplitudes."""

    amplitudes_raw = inputs["amplitudes"]
    leakages_raw = inputs["leakages"]
    if not isinstance(amplitudes_raw, (list, tuple)) or not isinstance(leakages_raw, (list, tuple)):
        raise EvaluatorError("receiver_lattice_leakage_exponent_v1: amplitudes and leakages must be lists or tuples")
    if len(amplitudes_raw) != len(leakages_raw) or len(amplitudes_raw) < 2:
        raise EvaluatorError(
            "receiver_lattice_leakage_exponent_v1: equal-length series with at least two observations are required"
        )
    log_amplitudes: list[float] = []
    log_leakages: list[float] = []
    for amplitude_raw, leakage_raw in zip(amplitudes_raw, leakages_raw, strict=True):
        amplitude = float(amplitude_raw)
        leakage = float(leakage_raw)
        if not math.isfinite(amplitude) or not math.isfinite(leakage) or amplitude <= 0.0 or leakage <= 0.0:
            raise EvaluatorError(
                "receiver_lattice_leakage_exponent_v1: every amplitude and leakage must be finite and positive"
            )
        log_amplitudes.append(math.log(amplitude))
        log_leakages.append(math.log(leakage))
    mean_x = sum(log_amplitudes) / len(log_amplitudes)
    mean_y = sum(log_leakages) / len(log_leakages)
    denominator = sum((value - mean_x) ** 2 for value in log_amplitudes)
    if denominator == 0.0:
        raise EvaluatorError("receiver_lattice_leakage_exponent_v1: amplitudes must not all be equal")
    numerator = sum(
        (x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(log_amplitudes, log_leakages, strict=True)
    )
    return numerator / denominator


def eval_same_basin_sharp_optimum(inputs: Mapping[str, Any]) -> float:
    """Return the least measured objective displacement inside one declared basin.

    A non-negative result means none of the supplied same-basin directions improved
    the objective.  Basin identity is deliberately not inferred by this evaluator;
    the producer must establish it in the empirical anchor.
    """

    values = [float(value) for value in inputs["objective_deltas"]]
    if not values or any(not math.isfinite(value) for value in values):
        raise EvaluatorError(
            "same_basin_sharp_optimum_v1: objective_deltas must be a non-empty finite sequence"
        )
    return min(values)


def eval_byte_distortion_cross_intersection_count(inputs: Mapping[str, Any]) -> int:
    """Count bodies satisfying both declared byte and distortion predicates."""

    byte_feasible = list(inputs["byte_feasible"])
    distortion_feasible = list(inputs["distortion_feasible"])
    if len(byte_feasible) != len(distortion_feasible):
        raise EvaluatorError(
            "byte_distortion_cross_intersection_count_v1: predicate sequences must have equal length"
        )
    return sum(bool(byte_ok) and bool(distortion_ok) for byte_ok, distortion_ok in zip(byte_feasible, distortion_feasible, strict=True))


def eval_roundtrip_token_to_argmax_affine(inputs: Mapping[str, Any]) -> float:
    """Evaluate the matched-PYAV affine token-error to argmax-error transfer."""

    intercept = float(inputs["intercept_argmax_errors"])
    slope = float(inputs["marginal_argmax_errors_per_token_error"])
    token_errors = float(inputs["token_errors"])
    if any(not math.isfinite(value) for value in (intercept, slope, token_errors)):
        raise EvaluatorError("roundtrip_token_to_argmax_affine_v1: inputs must be finite")
    if intercept < 0.0 or slope < 0.0 or token_errors < 0.0:
        raise EvaluatorError("roundtrip_token_to_argmax_affine_v1: inputs must be non-negative")
    return intercept + slope * token_errors


def eval_field_change_bhw_decomposition(inputs: Mapping[str, Any]) -> dict[str, int]:
    """Partition changed labels into benefit, harm, and wash against ground truth.

    ``before_labels`` are the transmitted token labels and ``after_labels`` are
    the proposed coding-field labels.  Only changed positions are admitted.
    """

    before = list(inputs["before_labels"])
    after = list(inputs["after_labels"])
    truth = list(inputs["ground_truth_labels"])
    if not (len(before) == len(after) == len(truth)):
        raise EvaluatorError("field_change_bhw_decomposition_v1: label sequences must have equal length")
    benefit = harm = wash = 0
    for old, new, gt in zip(before, after, truth, strict=True):
        if old == new:
            raise EvaluatorError("field_change_bhw_decomposition_v1: unchanged positions are outside the domain")
        old_ok, new_ok = old == gt, new == gt
        if not old_ok and new_ok:
            benefit += 1
        elif old_ok and not new_ok:
            harm += 1
        else:
            wash += 1
    return {"benefit": benefit, "harm": harm, "wash": wash}


def eval_context_model_reorder_savings(inputs: Mapping[str, Any]) -> int:
    """Return measured reorder savings, refusing transfer across coder class."""

    has_context_model = bool(inputs["has_context_model"])
    generic_savings = int(inputs.get("generic_coder_savings_bytes", 0))
    context_savings = int(inputs.get("context_model_savings_bytes", 0))
    if generic_savings < 0 or context_savings < 0:
        raise EvaluatorError("context_model_reorder_savings_v1: savings must be non-negative")
    return context_savings if has_context_model else generic_savings


def eval_generator_form_fit_error_entanglement(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Report generator byte ratio together with its inseparable fit error."""

    reference_bytes = int(inputs["reference_bytes"])
    generator_bytes = int(inputs["generator_bytes"])
    fit_error_fraction = float(inputs["fit_error_fraction"])
    if reference_bytes <= 0 or generator_bytes <= 0:
        raise EvaluatorError("generator_form_fit_error_entanglement_v1: byte counts must be positive")
    if not math.isfinite(fit_error_fraction) or not 0.0 <= fit_error_fraction <= 1.0:
        raise EvaluatorError("generator_form_fit_error_entanglement_v1: fit_error_fraction must be in [0,1]")
    return {
        "byte_ratio": reference_bytes / generator_bytes,
        "fit_error_fraction": fit_error_fraction,
        "transferable_as_lossless_credit": fit_error_fraction == 0.0,
    }


def eval_decoder_derivable_ideal_savings_ceiling(inputs: Mapping[str, Any]) -> float:
    """Scale a sampled conditional-codelength gain into an optimistic byte ceiling."""

    sampled_gain_bits = float(inputs["sampled_gain_bits"])
    sampled_fraction = float(inputs["sampled_fraction"])
    if not math.isfinite(sampled_gain_bits) or sampled_gain_bits < 0.0:
        raise EvaluatorError("decoder_derivable_ideal_savings_ceiling_v1: gain must be finite and non-negative")
    if not math.isfinite(sampled_fraction) or not 0.0 < sampled_fraction <= 1.0:
        raise EvaluatorError("decoder_derivable_ideal_savings_ceiling_v1: sampled_fraction must be in (0,1]")
    return sampled_gain_bits / sampled_fraction / 8.0


# Canonical equation_id -> evaluator for the built-in laws.
LAWREF_BUILTIN_EVALUATORS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "forfeit_matched_exit_v1": eval_forfeit_matched_exit_s_star,
    "tau_star_maslov_quantile_v1": eval_tau_star_maslov_quantile,
    "critical_nucleus_release_v1": eval_critical_nucleus_release_r_star,
    # crucible_v6 migration (#351 follow-up) — CONSUMED trio + LR-hold:
    "tau_end_knee_launch_v1": eval_tau_end_knee_launch,
    "hosc_beta_fireband_pin_v1": eval_hosc_beta_fireband_pin,
    "v9_hosc_beta_endpoint_v1": eval_v9_hosc_beta_endpoint,
    "lr_control_denominator_v1": eval_lr_control_denominator,
    "lr_hold_frac_no_hold_v1": eval_lr_hold_frac_no_hold,
    # crucible_v6 migration — LIBRARY laws (bit-match tested; not emitted flags):
    "settle_window_v1": eval_settle_window,
    "tail_cycle_floor_v1": eval_tail_cycle_floor,
    "conley_absolute_bar_v1": eval_conley_absolute_bar,
    "adaptive_eps_saturation_alarm_v1": eval_adaptive_eps_saturation_alarm,
    # V9·CGauge #223 parametrization optima (2026-07-11) — LawRef-executable sizing laws:
    "cgauge_whitney_moddim_v1": eval_cgauge_whitney_moddim,
    "cgauge_curvelet_parabolic_bank_v1": eval_cgauge_parabolic_along_tangent,
    "cgauge_beta2_window_v1": eval_cgauge_beta2_window,
    # MarginBandSatisficing provenance repair (2026-07-12): DERIVED-LIVE threshold law.
    "margin_band_satisficing_threshold_v1": eval_margin_band_satisficing_threshold,
    # Island-birth geometry wire (2026-07-15): ratio DERIVED from sealed n96 P/A receipt;
    # absolute scale remains tunable and efficacy remains owed.
    "isoperimetric_birth_weight_scaling_v1": eval_isoperimetric_birth_weight,
    # V9 top-three one-delta scientific declarations.  These equation ids are
    # canonical registry entries; the scalar adapter makes their config constants
    # LawRef-executable while the trainer/runtime receipt owns measured outcomes.
    "dseg_aware_fourier_taper_reweight_v1": eval_v9_scientific_declaration,
    "horizon_weighted_margin_hinge_v1": eval_v9_scientific_declaration,
    "step_native_activation_edge_optimality_v1": eval_v9_scientific_declaration,
    # AutoClip percentile grad-clip law (2026-07-15, #B-4 clip cure): the scalar
    # adapter makes the Lever's config constants (percentile/window/warmup)
    # LawRef-executable; the mechanism equation lives in
    # autoclip_percentile_grad_clip_20260715 (autoclip_threshold).
    "autoclip_percentile_threshold_v1": eval_v9_scientific_declaration,
    # Warm-start schedule reconstruction (2026-07-16, c2_surgical_warm): lineage schedule
    # boundaries are FUNCTIONS of the config of record / run length / resume window / plant
    # end — the DERIVED form for a warm path where no recognised event sensor is co-emittable
    # (label_floor DEAD below resume d_seg; adverse finding A2).
    "warm_start_schedule_reconstruction_v1": eval_warm_start_schedule_reconstruction,
    # Adam second-moment variance warmup length (2026-07-17, p0_resume_warmup_geometry item 2):
    # the beta2-DERIVED LR-rewarmup window the ResumeLRWarmup DSL lever resolves through
    # (c=1 == the sister rewarmup_beta2_memory_window_v1 bound; c~=2 default, RAdam rationale).
    "adam_v_variance_warmup_length_v1": eval_adam_v_variance_warmup_length,
    # EMA decay from run geometry (2026-07-17, ARM-C / p0_ema_calibration; SPEC_v10 §13.3):
    # decay DERIVED from (updates_per_run, target seed fraction / warmup fraction) — the
    # Quantizr 0.997 per-step provenance does not transfer to the full-batch regime.
    "ema_decay_run_geometry_v1": eval_ema_decay_run_geometry,
    # #332/#351 flag-custody backfill (2026-07-17): the registered NON-DERIVATIONAL
    # identity law.  Preserves bool(0/1)/int/float/str value bytes for measured or
    # waived constants; grants NO derivation authority (ladder_class on the LawRef
    # records the honest rung — class-4 hardcoded_waiver for generic config knobs).
    "dsl_custodied_scalar_identity_v1": eval_dsl_custodied_scalar_identity,
    # ddm_cn4 arc consolidation (2026-08-11): two existing surfaces gain
    # executable LawRef adapters and one new advisory multistart law.
    "cpu_cuda_score_gap_v1": eval_cpu_cuda_score_gap,
    "realization_breakeven_bytes_v1": eval_realization_breakeven_bytes,
    "radius2_multistart_singleton_escape_v1": eval_radius2_multistart_singleton_escape,
    # ddm_cn5 arc consolidation (2026-08-13): bounded instrument, receiver,
    # exact-score-budget, and quantized-lattice laws from the named-arm corpus.
    "local_model_step_resolvability_ratio_v1": eval_local_model_step_resolvability_ratio,
    "receiver_pose_semantic_preservation_ratio_v1": eval_receiver_pose_semantic_preservation_ratio,
    "pose_stack_exact_budget_v1": eval_pose_stack_exact_budget,
    "receiver_lattice_leakage_exponent_v1": eval_receiver_lattice_leakage_exponent,
    # ddm_lv3 current-arc law wave (2026-09-01). These are advisory/source-law
    # evaluators; none dispatches a scorer or promotes a score row.
    "same_basin_sharp_optimum_v1": eval_same_basin_sharp_optimum,
    "byte_distortion_cross_intersection_count_v1": eval_byte_distortion_cross_intersection_count,
    "roundtrip_token_to_argmax_affine_v1": eval_roundtrip_token_to_argmax_affine,
    "field_change_bhw_decomposition_v1": eval_field_change_bhw_decomposition,
    "context_model_reorder_savings_v1": eval_context_model_reorder_savings,
    "generator_form_fit_error_entanglement_v1": eval_generator_form_fit_error_entanglement,
    "decoder_derivable_ideal_savings_ceiling_v1": eval_decoder_derivable_ideal_savings_ceiling,
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
    "eval_adam_v_variance_warmup_length",
    "eval_adaptive_eps_saturation_alarm",
    "eval_cgauge_beta2_window",
    "eval_cgauge_parabolic_along_tangent",
    "eval_cgauge_whitney_moddim",
    "eval_conley_absolute_bar",
    "eval_cpu_cuda_score_gap",
    "eval_critical_nucleus_release_r_star",
    "eval_dsl_custodied_scalar_identity",
    "eval_forfeit_matched_exit_s_star",
    "eval_hosc_beta_fireband_pin",
    "eval_isoperimetric_birth_weight",
    "eval_local_model_step_resolvability_ratio",
    "eval_lr_control_denominator",
    "eval_lr_hold_frac_no_hold",
    "eval_margin_band_satisficing_threshold",
    "eval_pose_stack_exact_budget",
    "eval_radius2_multistart_singleton_escape",
    "eval_realization_breakeven_bytes",
    "eval_receiver_lattice_leakage_exponent",
    "eval_receiver_pose_semantic_preservation_ratio",
    "eval_settle_window",
    "eval_tail_cycle_floor",
    "eval_tau_end_knee_launch",
    "eval_tau_star_maslov_quantile",
    "eval_v9_hosc_beta_endpoint",
    "eval_v9_scientific_declaration",
    "eval_warm_start_schedule_reconstruction",
    "get_evaluator",
    "has_evaluator",
    "populate_lawref_evaluators",
    "register_evaluator",
    "registered_equation_ids",
    "resolve_equation_value",
]
