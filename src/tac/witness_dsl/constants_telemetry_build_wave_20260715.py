# SPDX-License-Identifier: MIT
"""#515 CONSTANTS + TELEMETRY BUILD WAVE (2026-07-15) — DERIVED levers + B0 instruments.

Operator directive (verbatim): "All the constants stuff and unbuilt telemetry and more
must be built and measured as well." This module is the BUILD half: every inherited
constant that a registered law can DERIVE becomes a DSL ``Lever`` factory emitting the
derived value with rung DERIVED (never the inherited literal); every constant that
genuinely cannot be derived carries a typed :class:`HardcodedWaiverCustody` with a
duty-to-measure battery arm; every unbuilt telemetry item becomes a producer + lever
(binding-vs-inert provable). The A/B battery (operator-GO) MEASURES; nothing here
launches, edits the live trainer (pid 31576 dry-start), or mutates a run.

Value-provenance ladder: DERIVED > CONFIG > ANCHOR > WAIVER. Sources:
- ``tac.witness_dsl.adaptivization_tickets_20260715`` (the ticket queue this drains)
- ``.omx/research/c1_config_differential_audit_20260715.md`` §5/§6/§8 (the battery)
- ``.omx/research/v9_missing_signal_constants_audit_20260715.md`` (FEED-510)
- laws: ``cgauge_beta2_window_v1`` (#223 Law 4 stationarity sandwich) ·
  ``costate_lambda_marginal_ds_v1`` (λ_pose = 5/sqrt(10·d_pose)) ·
  ``verdict_parallel_workers_speedup_v1`` (measured pool ladder) ·
  ``inr_weight_norm_radial_ode_v1`` (the ||W|| stream's consumer laws)

CONTAINMENT: builds + validates + compiles only ($0, pure). No dispatch, no run
mutation, no trainer edit. Trainer wire-ins named here are QUEUED behind the live
dry-start exit. Everything MEANS: pointer 0.19108 UNMOVED.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# The live dry-start this wave must not disturb (HARD CONSTRAINT, 2026-07-15).
LIVE_DRY_START_PID = 31576

# Banked pose operating point (MEASURED n600, R1 dxi #238 — memory L68):
# d_pose 0.001610 through byte-close; the pose-finish engage inherits it.
BANKED_R1_DPOSE = 0.001610

# MEASURED C0 verdict economics (run levelset_n600_witness_20260715T095030Z,
# witness_component_wallclock ep25 row; c1 audit §5 --eval-every row): one n600
# CPU verdict wall 2555.7 s; async overlap leaves a measured epoch inflation of
# ~+36 s/ep amortized at cadence 25 => ~900 s effective inflation per verdict on
# a ~325 s/ep base (post-lane-band composite 361 = 325 + 36).
MEASURED_VERDICT_INFLATION_S = 900.0
MEASURED_SEC_PER_EP_BASE = 325.0


class TrainerWireInQueued(ValueError):
    """A lever whose trainer consumer flag has not landed yet (fail-closed).

    Raised with the exact queued insertion point so composing the lever today can
    never emit an invented flag (never-invent-flags). Auto-unlocks: the factory
    re-checks the live trainer argparse, so the same call succeeds the moment the
    queued trainer wire-in lands (behind the pid-31576 dry-start exit).
    """


@dataclass(frozen=True)
class HardcodedWaiverCustody:
    """Typed class-4 custody for a constant that genuinely cannot be derived yet.

    Never silently keeps an inherited literal: every waiver names its reason, its
    owner (the module/ticket accountable for retiring it), the rederivation
    trigger that retires it, and the pre-registered battery arm that MEASURES it
    (duty-to-measure per the off-is-orphan rule). A waiver may only preserve
    honest nonselection; it grants no mechanism, score, or authority.
    """

    constant: str
    value: str
    reason: str
    owner: str
    rederivation_trigger: str
    battery_arm: str

    def __post_init__(self) -> None:
        for f in ("constant", "value", "reason", "owner", "rederivation_trigger",
                  "battery_arm"):
            v = getattr(self, f)
            if not isinstance(v, str) or not v.strip() or v.strip() in {
                "<reason>", "<rationale>", "TBD", "placeholder",
            }:
                raise ValueError(
                    f"HardcodedWaiverCustody.{f} requires a substantive non-placeholder "
                    f"string, got {v!r}"
                )


# ---------------------------------------------------------------------------
# Derivation functions (pure, deterministic fp64 — the laws behind the levers)
# ---------------------------------------------------------------------------
def stationarity_window_steps(
    steps_per_epoch: int = 75, curvature_timescale_epochs: float = 100.0,
) -> tuple[float, float]:
    """Admissible EMA-window sandwich in STEPS: [S, T_c*S/3].

    The #223 Law-4 stationarity sandwich (``cgauge_beta2_window_v1``) transferred
    to ANY exponential moving average over the training process: the window must
    (FLOOR) cover >= one full data cycle S (else it phase-locks to the within-
    epoch pair ordering) and (CEILING) stay <= ~1/3 of the curvature-drift
    timescale T_c*S (else it tracks stale curvature through anneals/stage flows).
    The transfer is a WINDOW argument only — the sandwich constrains any EMA over
    a process with data-cycle period S and drift timescale T_c*S; it does not by
    itself pick the point inside the window.
    """
    from tac.canonical_equations.cgauge_parametrization_optima_20260711 import (
        beta2_window,
    )

    lo_decay, hi_decay = beta2_window(
        int(steps_per_epoch), curvature_timescale_epochs=float(curvature_timescale_epochs)
    )
    return (1.0 / (1.0 - lo_decay), 1.0 / (1.0 - hi_decay))


def log_midpoint_decay(
    steps_per_epoch: int = 75, curvature_timescale_epochs: float = 100.0,
) -> float:
    """DERIVED point inside the stationarity window: decay = 1 - 1/sqrt(N_lo*N_hi).

    Point criterion (DERIVED-AT-CONFIG): the log-midpoint maximizes log-distance
    from BOTH failure modes (phase-lock floor / stale-curvature ceiling), i.e. it
    is the maximally robust choice under multiplicative misestimation of S and
    T_c. n600/accum-8 => S=75, T_c=100 => N* = sqrt(75*2500) ~= 433 steps =>
    decay ~= 0.99769. The battery arm (not this derivation) is the empirical
    arbiter versus the incumbent.
    """
    n_lo, n_hi = stationarity_window_steps(steps_per_epoch, curvature_timescale_epochs)
    return 1.0 - 1.0 / math.sqrt(n_lo * n_hi)


def derived_w_pose_at_engage(
    d_pose: float = BANKED_R1_DPOSE, seg_marginal: float = 100.0,
) -> float:
    """DERIVED seg-relative pose weight at the pose-finish engage.

    ``costate_lambda_marginal_ds_v1``: lambda_pose(d_pose) = 5/sqrt(10*d_pose),
    lambda_seg = 100. The score-marginal-correct RELATIVE weight is their ratio
    (w_pose in units where the seg term carries weight 1). At the banked R1
    operating point d_pose=0.001610: 39.405/100 = 0.394.

    CAVEAT (recorded, not hidden): the trainer's seg/pose LOSSES are surrogates,
    not d_seg/d_pose themselves, so the ratio is exact only insofar as both
    surrogates are calibrated to their score terms — the B3e battery arm
    (1.0 incumbent vs this derived value) is the empirical arbiter. A LIVE
    engage-time consumer (reading the run's own verdict d_pose instead of the
    banked anchor) is a queued trainer wire-in.
    """
    from tac.canonical_equations.costate_lambda_marginal_ds_20260705 import (
        costate_vector,
    )

    lam_seg, lam_pose, _ = costate_vector(float(d_pose))
    del lam_seg  # the law's own seg weight is fixed at 100; use the caller's normalization
    return lam_pose / float(seg_marginal)


def derived_eval_every(
    verdict_inflation_s: float = MEASURED_VERDICT_INFLATION_S,
    sec_per_ep_base: float = MEASURED_SEC_PER_EP_BASE,
    *,
    overhead_frac: float = 0.10,
    information_floor: int = 25,
    verdict_parallel_workers: int = 8,
) -> int:
    """DERIVED verdict cadence from the amortization budget + the VPW ladder.

    Law: eval_every* = max(information_floor,
    ceil(effective_inflation / (overhead_frac * sec_per_ep_base))), where
    effective_inflation subtracts the measured scorer-forward saving of
    ``verdict_parallel_workers_speedup_v1`` when the VPW instrument is composed
    (wall 370.64 s -> 65.18 s at w=8: saving ~305.5 s/verdict).

    n600 measured C0 economics: WITHOUT VPW the 10% budget derives cadence 28
    (the incumbent 25 runs ~11% — mildly over); WITH VPW(8) the derived cadence
    is 19 < 25, so the incumbent 25 is CONFIRMED-DERIVED *conditional on the
    VerdictParallelWorkers composition* (the floor binds: F10 dwell-window
    re-derivation + read-latency keep 25 as the information floor). The NCDE
    information-gain cadence (adaptivization ticket) remains the deeper law;
    its trainer consumer is ticket-only.
    """
    if verdict_inflation_s <= 0 or sec_per_ep_base <= 0 or overhead_frac <= 0:
        raise ValueError("derived_eval_every requires positive economics inputs")
    saving = 0.0
    if int(verdict_parallel_workers) >= 2:
        from tac.canonical_equations.verdict_parallel_workers_speedup_20260715 import (
            WALL_SEQUENTIAL_S,
            verdict_wall_projection,
        )

        saving = WALL_SEQUENTIAL_S - verdict_wall_projection(
            WALL_SEQUENTIAL_S, int(verdict_parallel_workers)
        )
    effective = max(0.0, float(verdict_inflation_s) - saving)
    cadence = math.ceil(effective / (float(overhead_frac) * float(sec_per_ep_base)))
    return max(int(information_floor), int(cadence))


# ---------------------------------------------------------------------------
# LawRef evaluators (idempotent registration; resolve() can execute the laws)
# ---------------------------------------------------------------------------
def _eval_stationarity_log_midpoint(inputs) -> float:
    return log_midpoint_decay(
        int(inputs["steps_per_epoch"]),
        float(inputs.get("curvature_timescale_epochs", 100.0)),
    )


def _eval_w_pose_engage_ratio(inputs) -> float:
    return derived_w_pose_at_engage(
        float(inputs["d_pose"]), float(inputs.get("seg_marginal", 100.0))
    )


def _eval_verdict_cadence_amortization(inputs) -> int:
    return derived_eval_every(
        float(inputs["verdict_inflation_s"]),
        float(inputs["sec_per_ep_base"]),
        overhead_frac=float(inputs.get("overhead_frac", 0.10)),
        information_floor=int(inputs.get("information_floor", 25)),
        verdict_parallel_workers=int(inputs.get("verdict_parallel_workers", 8)),
    )


def _eval_build_wave_custody_identity(inputs) -> float | int:
    """Non-derivational identity custody (measured-anchor / waiver value carrier).

    Mirrors the documented ``dsl_custodied_scalar_identity`` pattern: preserves
    the declared value's bytes for MEASURED or WAIVED constants; it cannot
    manufacture scientific authority (the ladder_class on the LawRef records
    which rung the value actually holds).
    """
    if "value" not in inputs:
        raise KeyError("build-wave custody identity requires a 'value' input")
    return inputs["value"]


_BUILD_WAVE_EVALUATORS = {
    "stationarity_window_log_midpoint_v1": _eval_stationarity_log_midpoint,
    "costate_w_pose_engage_ratio_v1": _eval_w_pose_engage_ratio,
    "verdict_cadence_amortization_v1": _eval_verdict_cadence_amortization,
    "build_wave_custody_identity_v1": _eval_build_wave_custody_identity,
}


def populate_build_wave_evaluators() -> tuple[str, ...]:
    """Register this wave's LawRef evaluators (process-global, idempotent)."""
    from tac.canonical_equations.evaluators import register_evaluator

    for eqid, fn in _BUILD_WAVE_EVALUATORS.items():
        register_evaluator(eqid, fn)
    return tuple(sorted(_BUILD_WAVE_EVALUATORS))


def _custody(equation_id: str, flag: str, inputs: dict, *, ladder_class: str,
             fallback: float | int | None = None,
             fallback_waiver_reason: str = "") -> tuple[dict, dict, float | int]:
    """Resolve one derived/custodied flag -> (lawrefs, manifests, value)."""
    from tac.witness_dsl.lawref import InputRef, LawRef, resolve

    populate_build_wave_evaluators()
    ref = LawRef(
        equation_id=equation_id,
        inputs={
            name: InputRef.literal(
                val, prov, config_tags={"vehicle": "v9_cgauge_ideal_mod19"},
            )
            for name, (val, prov) in inputs.items()
        },
        ladder_class=ladder_class,
        fallback=fallback,
        fallback_waiver_reason=fallback_waiver_reason,
    )
    resolved = resolve(
        ref,
        target_config_tags={"vehicle": "v9_cgauge_ideal_mod19"},
        repo_root=_REPO_ROOT,
    )
    return {flag: ref}, {flag: resolved.to_dict()}, resolved.value


def _trainer_has_flag(flag: str) -> bool:
    from tac.witness_dsl.curriculum_dsl import real_trainer_flags

    return flag in real_trainer_flags(None)


# ---------------------------------------------------------------------------
# PART A — constants-adaptivization Lever factories (DERIVED rung)
# ---------------------------------------------------------------------------
def DerivedAdamBeta2(steps_per_epoch: int = 75,  # noqa: N802 — DSL factory convention
                     curvature_timescale_epochs: float = 100.0):
    """--adam-beta2 from the #223 Law-4 window, point = log-midpoint (DERIVED).

    Window [1-1/S, 1-3/(T_c*S)] = [0.98667, 0.9996] at S=75/T_c=100; the derived
    log-midpoint is ~0.99769 (N*~=433 steps). The incumbent 0.999 (N=1000 steps)
    sits INSIDE the window — this lever is the B3b battery ARM, never a silent
    flip; the boundary hazard is separately cured by R7_beta2_window_rewarmup.
    """
    from tac.witness_dsl.lawref import LADDER_DERIVED_AT_CONFIG
    from tac.witness_dsl.curriculum_dsl import Lever

    refs, manifests, value = _custody(
        "stationarity_window_log_midpoint_v1", "--adam-beta2",
        {
            "steps_per_epoch": (int(steps_per_epoch),
                                "n600/accum-8 => 75 optimizer steps/epoch (config-derived)"),
            "curvature_timescale_epochs": (float(curvature_timescale_epochs),
                                           "T_c=100 ep anneal timescale (#223 Law 4, "
                                           "cgauge_beta2_window_v1 assumption)"),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    return Lever(
        "derived_adam_beta2",
        overrides={"--adam-beta2": round(float(value), 6)},
        notes=("#515 PART-A: beta2 DERIVED from the stationarity-sandwich log-midpoint "
               "(cgauge_beta2_window_v1 transfer); incumbent 0.999 is in-window — B3b arm, "
               "falsify if |d_seg@ep150 delta| < noise band (then 0.999 stays, re-classed "
               "DONT-CARE-in-window)"),
        lawrefs=refs, constant_manifest=manifests,
    )


def DerivedEmaDecay(steps_per_epoch: int = 75,  # noqa: N802
                    curvature_timescale_epochs: float = 100.0):
    """--ema-decay from the SAME stationarity sandwich (DERIVED window + midpoint).

    The Quantizr 0.997 is an ANCESTOR-VEHICLE anchor (L18: numbers do not
    transfer). The sandwich transfers as a WINDOW argument to the weight-EMA
    (data-cycle floor / curvature-drift ceiling); derived log-midpoint ~0.99769
    (N*~=433 steps vs the incumbent's 333). The derivation broadly CONFIRMS the
    incumbent's order and moves the point ~30% longer; B3a (extended: 0.997 vs
    0.9977-derived vs 0.99 vs 0.999) is the arbiter. Stage-dependent averaging
    at the turnpike is already live (Polyak finisher arm).
    """
    from tac.witness_dsl.lawref import LADDER_DERIVED_AT_CONFIG
    from tac.witness_dsl.curriculum_dsl import Lever

    refs, manifests, value = _custody(
        "stationarity_window_log_midpoint_v1", "--ema-decay",
        {
            "steps_per_epoch": (int(steps_per_epoch),
                                "n600/accum-8 => 75 optimizer steps/epoch (config-derived)"),
            "curvature_timescale_epochs": (float(curvature_timescale_epochs),
                                           "T_c=100 ep anneal timescale (#223 Law 4 transfer; "
                                           "window argument only — point = log-midpoint criterion)"),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    return Lever(
        "derived_ema_decay",
        overrides={"--ema-decay": round(float(value), 6)},
        notes=("#515 PART-A: EMA decay DERIVED (stationarity-window log-midpoint; retires the "
               "Quantizr/L18 ancestor literal as a DERIVATION, pending the B3a empirical "
               "arbiter). B3a extended arms {0.997, 0.99769-derived, 0.99, 0.999}; metric "
               "EMA-verdict d_seg@ep150 n24; falsify if best-vs-worst < noise band"),
        lawrefs=refs, constant_manifest=manifests,
    )


def DerivedWPoseAtEngage(d_pose: float = BANKED_R1_DPOSE):  # noqa: N802
    """--w-pose from the costate law at the pose-finish engage (DERIVED).

    lambda_pose/lambda_seg = (5/sqrt(10*d_pose))/100 = 0.394 at the banked R1
    d_pose 0.001610. B3e arm vs the incumbent 1.0 (fork-from-checkpoint tail at
    the c2 sigma_min/ep726 checkpoint). LIVE engage-time consumer (reading the
    run's own verdict d_pose) = queued trainer wire-in (see TRAINER_WIREIN_QUEUE).
    """
    from tac.witness_dsl.lawref import LADDER_DERIVED_AT_CONFIG
    from tac.witness_dsl.curriculum_dsl import Lever

    refs, manifests, value = _custody(
        "costate_w_pose_engage_ratio_v1", "--w-pose",
        {
            "d_pose": (float(d_pose),
                       "MEASURED banked R1 dxi #238 n600 through-byte-close d_pose 0.001610 "
                       "(memory L68) — the pose-finish engage operating point"),
            "seg_marginal": (100.0, "score law: lambda_seg = 100 (S = 100*d_seg + ...)"),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    return Lever(
        "derived_w_pose_at_engage",
        overrides={"--w-pose": round(float(value), 4)},
        notes=("#515 PART-A: w_pose DERIVED from costate_lambda_marginal_ds_v1 at the banked "
               "R1 operating point (score-marginal ratio; surrogate-calibration caveat "
               "recorded). B3e arm vs incumbent 1.0; metric d_pose-at-engage + d_seg "
               "non-regression; falsify on d_seg regression > noise band"),
        lawrefs=refs, constant_manifest=manifests,
    )


def DerivedEvalEvery(verdict_parallel_workers: int = 8):  # noqa: N802
    """--eval-every from the verdict-amortization budget law (DERIVED).

    With VPW(8) composed the derived cadence is max(25-floor, 19) = 25 — the
    incumbent is CONFIRMED-DERIVED conditional on the VerdictParallelWorkers
    instrument; without VPW the same law derives 28 (the incumbent runs ~11%
    verdict overhead, over the 10% budget). Composing this lever therefore
    REQUIRES composing VerdictParallelWorkers(>=2) in the same config.
    """
    from tac.witness_dsl.lawref import LADDER_DERIVED_AT_CONFIG
    from tac.witness_dsl.curriculum_dsl import Lever

    refs, manifests, value = _custody(
        "verdict_cadence_amortization_v1", "--eval-every",
        {
            "verdict_inflation_s": (MEASURED_VERDICT_INFLATION_S,
                                    "MEASURED C0 ep25: verdict inflates hosting epoch "
                                    "250->1153 s => ~+36 s/ep amortized at cadence 25 "
                                    "(~900 s effective inflation per verdict)"),
            "sec_per_ep_base": (MEASURED_SEC_PER_EP_BASE,
                                "MEASURED C0 composite 361 s/ep minus the 36 s/ep verdict "
                                "amortization"),
            "overhead_frac": (0.10, "budget choice: verdict overhead <= 10% of epoch wall"),
            "information_floor": (25, "F10 dwell-window derivation + read-latency floor "
                                      "(the incumbent's information role)"),
            "verdict_parallel_workers": (int(verdict_parallel_workers),
                                         "verdict_parallel_workers_speedup_v1 measured ladder "
                                         "(receipt verdict_parallel_bench_20260715T184252Z)"),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )
    return Lever(
        "derived_eval_every",
        overrides={"--eval-every": int(value)},
        notes=("#515 PART-A: verdict cadence DERIVED from the amortization budget + the VPW "
               "ladder; 25 CONFIRMED-DERIVED conditional on VerdictParallelWorkers(8). B3h "
               "arm 25 vs 50 still owed for the read-latency/would-fire calibration; the "
               "NCDE information-gain cadence stays the deeper (ticket-only) law"),
        lawrefs=refs, constant_manifest=manifests,
    )


# ---------------------------------------------------------------------------
# PART A — class-4 waivers (typed custody + duty-to-measure; never silent)
# ---------------------------------------------------------------------------
CLASS4_WAIVERS: tuple[HardcodedWaiverCustody, ...] = (
    HardcodedWaiverCustody(
        constant="--muon-lr / --muon-momentum / --muon-ns-steps / --muon-lr-final-frac",
        value="0.002 / 0.95 / 5 / 0.1",
        reason=("PR95-lineage internals; optdyn MEASURED an UNCHOSEN per-layer relative-LR "
                "increase (x1.40 film) under flat muon lr, but the eta_rel-pin cure is "
                "ticket-only-unbuilt — no honest derived value exists today"),
        owner="adaptivization_tickets_20260715.--muon-lr (eta_rel-pin ticket)",
        rederivation_trigger=("the B0 per-tensor ||W|| telemetry rows on the c2 run quantify "
                              "the drift (inr_weight_norm_radial_ode_v1); then the B3d "
                              "fork-from-ep726 pin-vs-flat A/B"),
        battery_arm="B3d",
    ),
    HardcodedWaiverCustody(
        constant="--hosc-beta-end",
        value="3.177",
        reason=("control-preserving rephase of the mod32cap beta(ep) (v6.3 MAJOR-2(i)) — "
                "custodied hosc_beta_fireband_pin_v1; the ENDPOINT is contested by the built "
                "step_iso arm (beta_end=8.0, 34.2% duty, never fired); no law picks the "
                "endpoint today"),
        owner="spec_v9_cgauge (hosc_beta_fireband_pin_v1 custody)",
        rederivation_trigger="B3f arm verdict (3.177 vs 8.0, d_seg@ep150 n24 + saturation "
                             "telemetry — fixed-beta divergence is the known failure mode)",
        battery_arm="B3f",
    ),
    HardcodedWaiverCustody(
        constant="--accum-pairs",
        value="8",
        reason=("fixes 75 optimizer steps/epoch at n600 across ALL stages; never swept "
                "jointly with clip/LR; memory permits 4 or 16 (41.86 GiB measured pass-1 "
                "peak at 128 GiB) — the joint objective (epochs x sec/ep) has no closed "
                "form here, only the sweep"),
        owner="adaptivization_tickets_20260715.--accum-pairs",
        rederivation_trigger="B3c joint sweep {8,4,16} with the B1 magnitude-law winner "
                             "(d_seg@ep150 + sec/ep — accum moves BOTH joint axes)",
        battery_arm="B3c",
    ),
    HardcodedWaiverCustody(
        constant="--grad-clip",
        value="0.5",
        reason=("INERT on the c1 config (--grad-normalize per-param unit-norms after the "
                "clip, dividing out any uniform norm scale — the C0 confound); the incumbent "
                "magnitude law (unit-norm x LR) is UNVALIDATED-but-not-measured-worse and "
                "the one measured alternative (naive AutoClip p10/w1000) REVERSED post-ep25"),
        owner="witness_stability.AMBER.grad_clip + adaptivization_tickets_20260715.--grad-clip",
        rederivation_trigger="B1 >=150-ep magnitude-law A/B + S4 causal rebase (the descent "
                             "clock gate — everything downstream is confounded until it lands)",
        battery_arm="B1",
    ),
    HardcodedWaiverCustody(
        constant="--ladder-{movable,lane}-{r0,birth,hold,anneal}",
        value="movable r0=0.2252 birth=60 anneal=200; lane r0=2.0 birth=80 anneal=260",
        reason=("hand-placed island-birth schedule; the continuation reframe says the birth "
                "lever should sit AT the computed saddle-node critical dilation — the "
                "reduced-order class-occupancy ODE derivation is unbuilt"),
        owner="adaptivization_tickets_20260715.--ladder-* (saddle-node derivation ticket)",
        rederivation_trigger="the #318/#344/#180 reduced-order model derivation places birth "
                             "at the computed fold point (curriculum_is_continuation memo)",
        battery_arm="none-yet (derivation-gated, not sweep-gated)",
    ),
)


# ---------------------------------------------------------------------------
# PART B — telemetry instruments (B0; binding-vs-inert provable)
# ---------------------------------------------------------------------------
# Trainer wire-ins queued behind the live dry-start (pid 31576). Each row names
# the EXACT insertion point so the post-exit patch is mechanical, never invented.
TRAINER_WIREIN_QUEUE: tuple[dict[str, str], ...] = (
    {
        "item": "per-tensor ||W|| telemetry row (B0 instrument, the optdyn unlock)",
        "producer": "tac.witness_control.weight_norm_telemetry.weight_norm_row (BUILT this wave)",
        "trainer_flag": "--weight-norm-telemetry (BooleanOptionalAction, default TRUE — "
                        "read-only observability defaults ON per the off-is-orphan rule)",
        "insertion_point": ("at the EMA-verdict emission site where live_np/ema_np are "
                            "already materialized (the _build_resume_state_arrays inputs / "
                            "the verdict row emit ~L9332 region of "
                            "experiments/train_levelset_witness_realized_through_R_mlx.py): "
                            "emit weight_norm_row(ep, live_np, ema_np, baseline=...) at "
                            "verdict cadence; baseline captured at first emission, restored "
                            "via baseline_from_row on resume"),
        "status": "queued-behind-dry-start (pid 31576)",
    },
    {
        "item": "live engage-time w_pose consumer (lambda_pose from the run's OWN verdict d_pose)",
        "producer": "tac.canonical_equations.costate_lambda_marginal_ds_20260705.costate_vector",
        "trainer_flag": "--w-pose-costate-engage (new flag; until it lands, "
                        "DerivedWPoseAtEngage emits the config-time banked-anchor value)",
        "insertion_point": ("the pose-finish engage site (pose_finish_conditioning_gate / "
                            "sigma_min_plateau engage, trainer ~L11698 region): replace the "
                            "static --w-pose with costate_vector(last_verdict_d_pose)[1]/100 "
                            "at the engage event"),
        "status": "queued-behind-dry-start (pid 31576)",
    },
    {
        "item": "NCDE information-gain verdict cadence consumer",
        "producer": "src/tac/witness_control/ncde_trajectory.py (built, advisory)",
        "trainer_flag": "--verdict-event (new flag; EventBackstopGate pattern, cadence-25 cap)",
        "insertion_point": "the eval-every cadence check in the epoch loop (gate the verdict "
                           "on predicted |d_seg delta| >= detection floor OR stage events)",
        "status": "ticket-only (adaptivization ticket --eval-every; NOT this wave's build)",
    },
)


def WeightNormTelemetryRow():  # noqa: N802
    """B0 instrument lever: per-tensor ||W|| rows at verdict cadence (defaults ON).

    FAIL-CLOSED until the trainer flag lands: the producer is BUILT
    (tac.witness_control.weight_norm_telemetry) but the emission site is queued
    behind the live dry-start (pid 31576) — composing this lever before the
    trainer wire-in would emit an invented flag, so it raises
    :class:`TrainerWireInQueued` with the exact insertion point instead. The same
    call auto-unlocks (returns the Lever) the moment the flag lands.

    # NO_EQUATION_NEEDED: read-only weight-norm observability; adds no loss term,
    # controller law, or score value (its CONSUMER laws are registered:
    # inr_weight_norm_radial_ode_v1).
    """
    from tac.witness_dsl.curriculum_dsl import Lever

    flag = "--weight-norm-telemetry"
    if not _trainer_has_flag(flag):
        queue_row = TRAINER_WIREIN_QUEUE[0]
        raise TrainerWireInQueued(
            f"WeightNormTelemetryRow: trainer flag {flag} has not landed — the emission-site "
            f"wire-in is queued behind the live dry-start (pid {LIVE_DRY_START_PID}). "
            f"Producer is BUILT ({queue_row['producer']}); insertion point: "
            f"{queue_row['insertion_point']}"
        )
    return Lever(
        "weight_norm_telemetry_row",
        overrides={flag: True},
        notes=("#515 B0 instrument: per-tensor ||W|| + EMA-norm + rel_from_t0 + eta_rel rows "
               "at verdict cadence (score-neutral read of already-materialized live_np/ema_np; "
               "defaults ON per off-is-orphan). Feeds inr_weight_norm_radial_ode_v1 consumers "
               "(eta_rel pin / row-norm projection / restoring decay — B3d)"),
    )


def VerdictBatch64():  # noqa: N802
    """B0 instrument lever: --verdict-batch 64 (MEASURED never-slower vs 32).

    Value bump only — chunk size of the ADVISORY CPU-torch verdict; bit-identical
    verdict values by construction (eval-mode BatchNorm uses running stats;
    argmax per-pixel; MSE per-pair — the #240 chunking law). Measured anchor:
    the FEED-510 parent audit's Tier-0 row (vb=64 never-slower; c1 audit §2
    'SUBOPTIMAL — vb=64 measured never-slower … flip in c2'). Memory-safe under
    the launcher preflight (verdict(verdict_batch) term scales the projection).
    """
    from tac.witness_dsl.lawref import LADDER_MEASURED_ANCHOR
    from tac.witness_dsl.curriculum_dsl import Lever

    refs, manifests, value = _custody(
        "build_wave_custody_identity_v1", "--verdict-batch",
        {
            "value": (64, "MEASURED never-slower vs 32 (v9_missing_signal_constants_audit_"
                          "20260715 §C.2/D.3-7 Tier-0; c1_config_differential_audit §3 row 2); "
                          "identity custody — the anchor is the audit's measured row, not a "
                          "derivation"),
        },
        ladder_class=LADDER_MEASURED_ANCHOR,
    )
    return Lever(
        "verdict_batch_64",
        overrides={"--verdict-batch": int(value)},
        notes=("#515 B0: vb=64 measured never-slower (free flip; ORPHANED-BY-TIMING in c1 — "
               "recovery-table row 2). Verdict values bit-identical by the #240 chunking law; "
               "first-verdict value-identity check is the B0 acceptance gate"),
        lawrefs=refs, constant_manifest=manifests,
    )


def ModDimDynamicsOn():  # noqa: N802
    """B0 instrument lever: explicit --mod-dim-dynamics emission (custody + registry).

    The trainer default is already True; c1's argv never EMITTED it, leaving the
    D18 k90 sensor registry-unmapped (c1 audit recovery-table row 4 'ORPHANED +
    registry-unmapped'). Explicit emission gives the flag a DSL owner + custody
    so binding-vs-inert is provable and default drift cannot silently disable
    the ~7 KB k90-truncate rate lead.

    # NO_EQUATION_NEEDED: read-only spectral-occupancy sensor (byte-identical
    # per the trainer's own help: no update-path / RNG touch).
    """
    from tac.witness_dsl.curriculum_dsl import Lever

    return Lever(
        "mod_dim_dynamics_explicit",
        overrides={"--mod-dim-dynamics": True},
        notes=("#515 B0: D18 k90 spectral-occupancy sensor explicitly owned (was default-True "
               "but argv-silent => registry-unmapped orphan). Free rate signal for the D18 "
               "byte-close (~7 KB k90-truncate lead); score-neutral, defaults ON"),
    )


# ---------------------------------------------------------------------------
# The updated A/B battery (folds this wave's levers into the c1-audit §8 plan)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatteryArm:
    """One pre-registered battery row (dimension -> arms -> metric -> falsification)."""

    arm_id: str
    dimension: str
    arms: tuple[str, ...]
    metric: str
    falsification: str
    scale_cost: str
    order_gate: str

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError(f"BatteryArm {self.arm_id}: arms must be non-empty")
        for f in ("arm_id", "dimension", "metric", "falsification", "scale_cost",
                  "order_gate"):
            if not str(getattr(self, f)).strip():
                raise ValueError(f"BatteryArm {self.arm_id}: {f} must be non-empty")


BUILD_WAVE_BATTERY: tuple[BatteryArm, ...] = (
    BatteryArm(
        arm_id="B0",
        dimension="instruments (identity/score-neutral by construction)",
        arms=("VerdictParallelWorkers(8) [existing lever]",
              "VerdictBatch64 [this wave]",
              "VerdictLiveGap [existing lever, curriculum_dsl]",
              "ModDimDynamicsOn [this wave]",
              "WeightNormTelemetryRow [producer BUILT this wave; trainer flag "
              "queued-behind-dry-start]"),
        metric="first-verdict value-identity check passes; telemetry rows appear "
               "(weight_norm rows at verdict cadence once the trainer wire-in lands)",
        falsification="any verdict value delta vs the sequential/vb32 incumbent = "
                      "instrument bug, pull the instrument (never ship a drifted verdict)",
        scale_cost="$0; ~2-3 h build+dry-start (the ||W|| trainer patch rides the "
                   "post-dry-start window)",
        order_gate="FIRST — every later arm inherits the instruments",
    ),
    BatteryArm(
        arm_id="B1",
        dimension="magnitude law (the descent-clock dimension; c1-audit row unchanged)",
        arms=("A incumbent (per-param normalize + inert clip 0.5 [waived, custody row])",
              "B normalize-none + AutoClip(p10,w1000)", "C normalize-none + fixed 0.5",
              "S4 causal rebase (fork armB@ep75 onto fixed-0.5)"),
        metric="d_seg (cadence-25 verdict) at ep150 + monotone-tail + log-slope ep1-150",
        falsification="any arm that reverses or trips flicker/gradient-quality alarms; "
                      "winner = lowest ep150 d_seg with monotone tail",
        scale_cost="n24 screen 4 arms x ~3.2 h ~= 13 h; winner CONFIRMED n600 inside "
                   "the c2 run's first 150 ep",
        order_gate="GATES EVERYTHING downstream (sets the descent clock)",
    ),
    BatteryArm(
        arm_id="B2",
        dimension="basis (Fourier vs curvelet; c1-audit row unchanged)",
        arms=("legacy_fourier_ab_control", "windowed_curvelet (#502 frames, staged arm)"),
        metric="curvelet_through_R_dseg_ab: n600 through-R d_seg at matched epoch/bytes",
        falsification="curvelet >= parity at ep500 => curvelet into c2 (doctrine); worse "
                      "=> Fourier stays, curvelet re-queued terminal-band",
        scale_cost="n24 paired ~21 h; n600 bounded ep500 paired ~4 GPU-days (dominant item)",
        order_gate="after B1 (basis-before-capacity, L24)",
    ),
    BatteryArm(
        arm_id="B3a",
        dimension="--ema-decay (EXTENDED this wave with the derived arm)",
        arms=("0.997 incumbent (ancestor anchor)", "0.99769 DERIVED (DerivedEmaDecay)",
              "0.99", "0.999"),
        metric="EMA-verdict d_seg@ep150 n24 SCREEN (never a verdict); winner CONFIRMED at "
               "n600 through-R inside the c2 run's first 150 ep (+ Polyak-on finisher "
               "control already in-config)",
        falsification="best-vs-worst < noise band => 0.997 stays, re-classed "
                      "DONT-CARE-in-window (the derivation then closes the ticket as "
                      "window-confirmation)",
        scale_cost="3 extra arms ~= 9.7 h (was 2 arms/6.5 h — +3.2 h for the derived arm)",
        order_gate="after B1",
    ),
    BatteryArm(
        arm_id="B3b",
        dimension="--adam-beta2 (derived arm now a BUILT lever)",
        arms=("0.999 incumbent (in-window)", "0.99769 DERIVED (DerivedAdamBeta2)"),
        metric="d_seg@ep150 n24 SCREEN; winner CONFIRMED at n600 through-R (c2 first-150-ep "
               "pre-registered checkpoint read)",
        falsification="delta < noise band => 0.999 stays (in-window DONT-CARE)",
        scale_cost="1 arm ~= 3.2 h",
        order_gate="after B1",
    ),
    BatteryArm(
        arm_id="B3c",
        dimension="--accum-pairs (class-4 waiver, duty-to-measure)",
        arms=("8 incumbent [waived, custody row]", "4", "16"),
        metric="d_seg@ep150 + sec/ep n24 SCREEN (JOINT objective — accum moves both axes); "
               "winner CONFIRMED at n600 through-R before entering c2",
        falsification="joint (epochs-to-target x sec/ep) worse than incumbent on both "
                      "alternatives => 8 re-classed measured-optimal-at-n600",
        scale_cost="2 arms ~= 6.5 h",
        order_gate="after B1 (joint with the magnitude-law winner)",
    ),
    BatteryArm(
        arm_id="B3d",
        dimension="Muon internals (class-4 waiver; measure-first via B0 ||W|| rows)",
        arms=("incumbent flat finisher [waived, custody row]",
              "eta_rel-pinned finisher (BUILD-OWED ~1 d; degrade to measure-only if unbuilt)"),
        metric="per-tensor ||W|| drift (B0 rows) + d_seg tail; fork-from-ep726 checkpoint",
        falsification="pin arm regresses d_seg tail vs flat => flat stays, drift re-classed "
                      "benign-gradient-driven (the optdyn T1 shrink finding)",
        scale_cost="fork-from-checkpoint tails only (deferred; not launch-gating)",
        order_gate="after c2 reaches ep726",
    ),
    BatteryArm(
        arm_id="B3e",
        dimension="--w-pose at engage (derived arm now a BUILT lever)",
        arms=("1.0 incumbent", "0.394 DERIVED (DerivedWPoseAtEngage @ banked R1 d_pose)"),
        metric="d_pose at engage + d_seg non-regression (fork both arms from the c2 "
               "ep726/sigma_min checkpoint)",
        falsification="derived arm regresses d_seg > noise band OR fails to reach the "
                      "banked-R1 d_pose parity => incumbent stays; surrogate-calibration "
                      "caveat then becomes the named blocker",
        scale_cost="fork-from-checkpoint tail (deferred; with B3d)",
        order_gate="terminal-band pair with B3d",
    ),
    BatteryArm(
        arm_id="B3f",
        dimension="--hosc-beta-end (class-4 waiver, duty-to-measure)",
        arms=("3.177 incumbent [waived, custody row]", "8.0 (built step_iso arm, 34.2% duty)"),
        metric="d_seg@ep150 n24 SCREEN + activation-saturation telemetry; winner CONFIRMED "
               "at n600 through-R before entering c2",
        falsification="8.0 arm saturates (fixed-beta divergence class) or regresses => "
                      "3.177 stays; wins => endpoint re-derived from the fire-band pin",
        scale_cost="1 arm ~= 3.2 h",
        order_gate="after B1",
    ),
    BatteryArm(
        arm_id="B3h",
        dimension="--eval-every (derivation CONFIRMS 25 conditional on VPW(8))",
        arms=("25 DERIVED-CONFIRMED (DerivedEvalEvery + VerdictParallelWorkers(8))", "50"),
        metric="wall-clock/ep + event-sensor satisfiability (F10 dwell re-derivation at "
               "cadence 50) + d_seg read-latency cost",
        falsification="cadence-50 breaks would-fire calibration or read latency => 25 stays "
                      "(now DERIVED, no longer ancestor-suspect)",
        scale_cost="piggybacks B3a-c arms ~= +3.2 h",
        order_gate="after B0 (needs live-gap + submit-row instruments)",
    ),
)


def battery_cost_summary() -> dict[str, str]:
    """The updated battery cost envelope (delta vs the c1-audit §8 totals)."""
    return {
        "n24_battery": "B0+B1+B3a-c,f-h+B4b ~= 50-56 h ~= 2.1-2.3 GPU-days "
                       "(+3.2 h vs the c1-audit 47-53 h: the B3a derived-EMA arm)",
        "b2_n600_bounded_basis_pair": "+4.0 GPU-days (unchanged; the dominant item)",
        "full_battery": "~6.3 GPU-days (was 6.2)",
        "compressed": "~2.3 GPU-days (was 2.2; defers the n600 curvelet read)",
        "cloud_spend": "$0 (all local M5 Max)",
        "builds_inside_battery": "||W|| trainer wire-in (queued-behind-dry-start; producer "
                                 "DONE this wave) · eta_rel pin (~1 d, optional B3d) · "
                                 "adaptive-width taper extension (~1-2 d, optional B3g) · "
                                 "verdict-submit fix (data-gated B4a)",
    }


def c2_composition_recipe() -> dict[str, str]:
    """Winner-sets-flag recipe for c2_optimal_form (updated with this wave's levers)."""
    return {
        "B0_unconditional": "VerdictParallelWorkers(8) + VerdictBatch64() + "
                            "VerdictLiveGap() + ModDimDynamicsOn() + "
                            "WeightNormTelemetryRow() [after the trainer wire-in lands]",
        "B1_winner": "{--grad-normalize, --grad-clip/--grad-clip-mode} via "
                     "AdaptiveGradClip/GradNormalizeNone levers or incumbent-stay "
                     "(GRAD_CLIP waiver custody retires either way)",
        "B2_winner": "--basis {legacy_fourier_ab_control | windowed_curvelet + bank params} "
                     "per the no-regression rule",
        "B3_winners": "DerivedEmaDecay/DerivedAdamBeta2/accum/DerivedEvalEvery/beta-end "
                      "levers enter IFF their arms win; losers close their adaptivization "
                      "tickets as measured-confirmations of the incumbent",
        "B3d_B3e": "fork-from-checkpoint tails on the c2 run itself (DerivedWPoseAtEngage "
                   "is the B3e arm carrier; the pin/lambda_pose LIVE consumers enter c3)",
        "B4b": "ComputeDtype('bf16') on QC ADMIT, else fp32",
        "everything_else": "byte-identical to c1 (the settled X-ray dimensions); budget "
                           "re-anchored to the measured composite; dry-start GREEN required",
    }


# ---------------------------------------------------------------------------
# Containment (mirrors the adaptivization-ticket queue pattern)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuildWaveManifest:
    """Frozen research-only carrier for the whole wave (validated composition)."""

    waivers: tuple[HardcodedWaiverCustody, ...] = field(default=CLASS4_WAIVERS)
    battery: tuple[BatteryArm, ...] = field(default=BUILD_WAVE_BATTERY)
    trainer_wirein_queue: tuple[dict[str, str], ...] = field(default=TRAINER_WIREIN_QUEUE)
    live_training_enabled: bool = False
    paid_or_remote_dispatch_enabled: bool = False
    live_run_mutation_enabled: bool = False
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if any((self.live_training_enabled, self.paid_or_remote_dispatch_enabled,
                self.live_run_mutation_enabled)):
            raise ValueError("build wave is compile-only; no actuation authority")
        if not self.research_only or self.score_claim or self.promotion_eligible:
            raise ValueError("build wave is MEANS-only non-promotion evidence")
        arm_ids = [a.arm_id for a in self.battery]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("duplicate battery arm ids")
        consts = [w.constant for w in self.waivers]
        if len(consts) != len(set(consts)):
            raise ValueError("duplicate waiver constants")

    def compile_contract(self) -> dict:
        return {
            "schema": "constants_telemetry_build_wave.v1",
            "memo": ".omx/research/constants_telemetry_build_wave_20260715.md",
            "waivers": [asdict(w) for w in self.waivers],
            "battery": [asdict(a) for a in self.battery],
            "trainer_wirein_queue": list(self.trainer_wirein_queue),
            "battery_cost": battery_cost_summary(),
            "c2_recipe": c2_composition_recipe(),
        }


__all__ = [
    "BANKED_R1_DPOSE",
    "BUILD_WAVE_BATTERY",
    "BatteryArm",
    "BuildWaveManifest",
    "CLASS4_WAIVERS",
    "DerivedAdamBeta2",
    "DerivedEmaDecay",
    "DerivedEvalEvery",
    "DerivedWPoseAtEngage",
    "HardcodedWaiverCustody",
    "LIVE_DRY_START_PID",
    "ModDimDynamicsOn",
    "TRAINER_WIREIN_QUEUE",
    "TrainerWireInQueued",
    "VerdictBatch64",
    "WeightNormTelemetryRow",
    "battery_cost_summary",
    "c2_composition_recipe",
    "derived_eval_every",
    "derived_w_pose_at_engage",
    "log_midpoint_decay",
    "populate_build_wave_evaluators",
    "stationarity_window_steps",
]
