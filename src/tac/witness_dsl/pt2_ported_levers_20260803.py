# SPDX-License-Identifier: MIT
"""ddm_pt2 — retired-vehicle levers PORTED to the TR1 frontier, as real DSL ``Lever`` factories.

Operator directive 2026-08-03: *"We can port old flags and levers to our latest frontier."*

**What makes these four a PORT rather than a rename.**  MEASURED: the TR1 trainer
(``experiments/train_tr1_partition_renderer_mlx.py``) imports the SAME ``make_loss_fn`` the
RETIRED levelset trainer uses (from ``experiments/train_witness_realized_through_R_mlx.py``) and
passed **5 of its 18 parameters**.  Four fully-implemented seg forces were therefore already
inside TR1's own loss graph — the mechanism, not a lookalike — and unreachable for exactly one
reason: TR1's argparse never declared their flags.  ``ddm_pt2`` declared them and threaded them.
Nothing about the force was re-implemented, which is why these are class (a) PORTS CLEANLY.

**Why they fire on the LIVE vehicle, which is the part that is easy to get wrong.**  Focal and
the Fisher density fold into ``seg_pixel_w``, and the natural-gradient transform rewrites
``seg_logits``; both surfaces are read by EVERY seg-loss branch before the mean.  EN1 separately
wires ``--margin-weighted-loss`` into ``tau_softplus`` and leaves it owned by the existing
``tr1_seg_margin_weight`` DSL lever; these four ports remain the distinct retired-trainer forces
that were unreachable only because TR1 never declared their own flags.

**Registry contract.**  Every factory here is default-OFF and score-affecting, so each is a
duty-to-measure row the moment this module exists: ``package_known_levers()`` AST-scans the whole
package, so these enter ``never_fired()`` / ``duty_to_measure()`` automatically — no parallel
registry, per the DSL-is-the-single-source-of-truth law.  ``TRAINER_RELPATH`` below is what keeps
them filed under the vehicle we ship (ddm_lr2 §1: an undeclared module silently defaults to the
RETIRED trainer, and a queue cannot drain a lever filed under the wrong vehicle).

Pointer honesty: ``0.1910828242`` [contest-CPU] UNMOVED.  This module is apparatus + reachability;
no arm here has been raced, so no row claims a d_seg effect.  ``score_claim=False``.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

#: ddm_lr2 §1 — declare the vehicle these levers were ported TO, or the registry files them
#: under the retired trainer by silent default and no TR1-scoped query can surface them.
TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"


def lever_seg_focal_gamma(gamma: float = 2.0) -> Lever:
    """PORT of the retired-trainer ``SegFocalGamma`` lever (``--seg-focal-gamma``).

    Focal per-pixel reweight ``(1 - p_y)^gamma`` computed from the REALIZED softmax GT-class
    probability, stop-grad and mean-1 renormalized, folded MULTIPLICATIVELY into ``seg_pixel_w``.
    The loss FORM is unchanged; only the per-pixel gradient BUDGET is reallocated, and the mean-1
    renorm conserves the total budget.  Under stop-grad the gradient-weight ratio between two
    pixels is exactly ``((1-p1)/(1-p2))^gamma``.

    Motivation ON THIS VEHICLE, and stated as a hypothesis rather than a result: seg is the
    largest remaining axis and ``ddm_pc2`` MEASURED that interior pixels contribute ~0.058% of
    flips — i.e. the uniform loss spends most of its budget where nothing can flip.  Focal is the
    concentration knob for exactly that, and it composes on the surface every seg form reads.

    Falsifier (pre-registered): matched-epoch, matched-bytes A/B shows realized d_seg no better
    than the gamma=0 control => concentration by realized-confidence does not help THIS renderer
    and the row closes at INSTANCE (the Fisher-density arm below is the sister allocator, and the
    two DISAGREE on confidently-wrong pixels, so a focal negative does not close that one).
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0 to arm the lever (0.0 is the OFF control, which is "
                         "the trainer default -- do not build a Lever for it)")
    # NAME: deliberately ``tr1_focal_...`` and NOT ``tr1_seg_focal_...``. The first draft used the
    # latter and this arm's own round-1 test caught it as AMBIGUOUS: ``lever_seg_physics`` emits
    # ``f"tr1_seg_{form_start}"``, whose family regex swallows any ``tr1_seg_*`` name. A new lever
    # must not land inside another factory's name family — the join reports it, it does not repair
    # it (test_ported_lever_names_join_back_to_their_factories is the standing guard).
    return Lever(
        name=f"tr1_focal_gamma{gamma:g}",
        overrides={"--seg-focal-gamma": str(gamma)},
        notes="ddm_pt2 PORT of SegFocalGamma; folds into seg_pixel_w => honored by EVERY seg form "
              "including tau_softplus; default 0.0 => "
              "byte-identical; falsifier in the factory docstring",
        constant_manifest={
            "--seg-focal-gamma": {
                "value": float(gamma), "rung": "RACED-NOT-ASSERTED (no measured optimum on TR1)",
                "provenance": "gamma=2.0 is the Lin et al. focal-loss canonical value and the "
                              "retired-trainer lever's own reference point; it is a RACE START on "
                              "this vehicle, not an optimum. The own-optimum law requires sweeping "
                              "gamma before any adopt/kill verdict is admissible."},
        })


def lever_fisher_density(blend: float = 1.0, source: str = "model") -> Lever:
    """PORT of the retired-trainer ``FisherDensityWeight`` lever (``--fisher-density-weight``).

    Per-pixel seg weight from the EXACT registered law ``tr g = (1/2) sech^2(m/2)``
    (``fisher_curvature_equals_categorical_fisher_trace_caustic_v1``, rho=0.978 measured
    calibration): the local categorical-Fisher trace IS the decision-geometry density, so the
    descent force becomes metric-aware — capacity flows where decisions actually bend.  Blend
    ``lambda in (0,1]``: ``w = (1-lambda)*1 + lambda*(tr g / mean tr g)``, mean-1 + stop-grad.

    ``source='model'`` evaluates the density at the LIVE margin (Fisher-natural for a training
    force: the metric at the current point of the flow); ``'gt'`` uses the cached GT margin field
    as a stationary importance prior.  Both are shipped so the A/B is a real race.

    DISTINCT from focal, and the distinction is the whole point: focal UP-weights confidently-wrong
    pixels; the Fisher density DOWN-weights them (a confidently-wrong pixel sits in a metrically
    flat region where logit motion buys almost no decision-geometry distance).  Composing both
    multiplies the maps — race them one at a time first.

    Falsifier (pre-registered): matched-epoch A/B shows realized d_seg no better than the
    lambda=0 control on BOTH sources => the Fisher metric does not bind the per-pixel budget on
    this vehicle; the row closes at INSTANCE for the weighting APPLICATION (never for the
    calibration law itself, which is separately measured).
    """
    if not (0.0 < float(blend) <= 1.0):
        raise ValueError("blend lambda must be in (0, 1]; 0.0 is the OFF control (trainer default)")
    if source not in ("model", "gt"):
        raise ValueError("fisher_density_source is model|gt")
    return Lever(
        name=f"tr1_fisher_density_{source}_l{blend:g}",
        overrides={"--fisher-density-weight": str(blend), "--fisher-density-source": source},
        notes="ddm_pt2 PORT of FisherDensityWeight; exact sech^2 law folded into seg_pixel_w => "
              "honored by every seg form; sister-but-OPPOSITE allocator to focal on "
              "confidently-wrong pixels; default 0.0 => byte-identical",
        constant_manifest={
            "--fisher-density-weight": {
                "value": float(blend), "rung": "RACED-NOT-ASSERTED (endpoint of the blend ladder)",
                "provenance": "lambda=1.0 is the PURE Fisher density (the law with no blending), "
                              "chosen as the race START because a partial blend cannot falsify the "
                              "law -- it only attenuates it. The intermediate ladder is the sweep."},
            "--fisher-density-source": {
                "value": source, "rung": "DERIVED (Fisher-natural default) + RACED against 'gt'",
                "provenance": "'model' is DERIVED: the Fisher metric is the pullback metric AT the "
                              "current point of the flow, so a training force must evaluate the "
                              "density at the live logits. 'gt' is the stationary-prior arm, raced "
                              "not assumed. Registered law "
                              "fisher_curvature_equals_categorical_fisher_trace_caustic_v1"},
        })


def lever_head_natural_grad(eps: float = 1e-3) -> Lever:
    """PORT of the retired-trainer ``HeadNaturalGradient`` lever (``--head-natural-grad``).

    Forward-IDENTITY / backward-``g+`` transform on the seg logits: the frozen SegNet head is
    exact rank-4 linear, so the categorical Fisher ``g = diag(p) - p p^T`` has a closed-form
    pseudo-inverse on its range, and the seg descent direction becomes the Fisher natural gradient
    at O(K) per pixel with no solve.  The loss VALUE and every activation are unchanged — this is
    a preconditioner, not a new term, which is why it composes with any seg form and with either
    per-pixel allocator above.

    Falsifier (pre-registered): matched-epoch A/B shows no faster realized-d_seg descent and no
    better endpoint => the Euclidean logit metric was not the binding limitation on this vehicle;
    closes at INSTANCE.  NOTE the honest confound: a preconditioner changes the effective step
    size, so the arm must be run at ITS OWN lr optimum before a negative is admissible (the
    own-optimum law), otherwise a null result is measuring lr, not the metric.
    """
    if float(eps) <= 0.0:
        raise ValueError("head_natural_grad_eps must be > 0 (damping of 1/p at simplex corners)")
    ov = {"--head-natural-grad": "on"}
    if float(eps) != 1e-3:                    # only emit the value flag when it is non-default;
        ov["--head-natural-grad-eps"] = repr(float(eps))   # the trainer refuses a set-but-ungated
    return Lever(                                          # value, and an inert flag is the genus
        name=f"tr1_head_natural_grad_eps{eps:g}",          # this port exists to avoid.
        overrides=ov,
        notes="ddm_pt2 PORT of HeadNaturalGradient; forward-identity => loss value unchanged, only "
              "the backward pass is preconditioned; 'off' => byte-identical; own-optimum caveat "
              "(re-tune lr per arm) in the factory docstring",
        constant_manifest={
            "--head-natural-grad-eps": {
                "value": float(eps), "rung": "DERIVED-CONSERVATIVE-DAMPING (make_loss_fn default)",
                "provenance": "damping added to p before the 1/p division so near-degenerate "
                              "simplex corners (p->0) do not blow up the preconditioner; 1e-3 is "
                              "the value the implementation was written and validated with on the "
                              "retired vehicle. Not raced on TR1 -- a duty-to-measure slot."},
        })


def lever_tau_softplus_tau(tau: float) -> Lever:
    """PORT of the ``tau_softplus`` form's OWN scalar (``--tau-softplus-tau``).

    ``mean(tau * softplus(-m / tau))``.  Until ddm_pt2 the TR1 trainer took ``make_loss_fn``'s
    default (0.3) with no way to set it, while the live burn lineage ran ``tau_softplus`` for
    ~100% of its epochs (MEASURED, ddm_tp2 row 2: b4s window_01..03 + r1c window_01).  So the ONE
    scalar shaping the live vehicle's actual loss was unreachable from the DSL.  This lever makes
    it a first-class, raceable, provenance-carrying knob rather than an inherited default.

    ``tau`` is the softness of the margin surrogate: as ``tau -> 0`` the form approaches the hard
    top-2 margin (the quantity argmax actually decides on) and its gradient concentrates on the
    separatrix; larger ``tau`` spreads gradient into the interior.  Given ``ddm_pc2``'s measured
    ~0.058% interior flip share this is a temperature with a real geometric meaning here, and it
    has never been swept on this vehicle.

    Falsifier (pre-registered): a matched-epoch sweep over tau finds no arm better than 0.3 =>
    the inherited default was already at its own optimum on TR1 and the row closes MEASURED
    (which is a genuinely useful outcome -- it converts an inherited constant into a derived one).
    """
    if not (float(tau) > 0.0):
        raise ValueError("tau_softplus_tau must be > 0")
    return Lever(
        name=f"tr1_tau_softplus_tau{tau:g}",
        overrides={"--tau-softplus-tau": repr(float(tau))},
        notes="ddm_pt2 PORT: the live seg form's own temperature, previously an unreachable "
              "inherited default; 0.3 == that default => byte-identical; falsifier in the docstring",
        constant_manifest={
            "--tau-softplus-tau": {
                "value": float(tau),
                "rung": "INHERITED-DEFAULT-BEING-PROMOTED-TO-RACED (constants-are-poison)",
                "provenance": "0.3 is make_loss_fn's own default, inherited by every TR1 run to "
                              "date WITHOUT a derivation for this vehicle -- exactly the borrowed "
                              "-constant class. This lever exists so the value becomes MEASURED "
                              "rather than inherited; any tau shipped from here carries its sweep."},
        })


#: The ported set, for a caller that wants to enumerate rather than import one at a time.
PT2_PORTED_LEVERS = (
    lever_seg_focal_gamma,
    lever_fisher_density,
    lever_head_natural_grad,
    lever_tau_softplus_tau,
)
