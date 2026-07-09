# SPDX-License-Identifier: MIT
"""#310 STEP-NATIVE ACTIVATION — the topology-matched chart for the piecewise-constant argmax.

WHAT IT IS (mechanically). The level-set / coord-INR witness applies a periodic first-layer
activation to its coordinate features. The default is the ``hosc`` activation
``a(u) = tanh(beta * sin(omega * u))`` (the trainer's ``_act`` hosc branch). As ``beta -> inf`` this
approaches a SQUARE WAVE ``sign(sin(omega * u))`` — a partition-indicator / STEP basis. The SegNet
argmax target is PIECEWISE-CONSTANT (each pixel is one of 5 class labels), so a step-native basis is
the topology-matched chart: it has NO Gibbs ringing at the class boundary (the smooth sine basis DOES,
which we MEASURED), it is O(1) params per edge, and it is L-infinity-optimal AT the edge. This is the
deep-math "step-native / partition-indicator basis" lever (§OPERATOR PRIORITY item 5 of the measured
d_seg levers).

THE MEASURED SATURATION-DEATH CAVEAT (DAG FEED 2026-06-25a — do NOT strip). FIXED ``beta = 4`` hosc
DIVERGES: ``tanh(4 * sin)`` saturates to +-1 almost everywhere → the gradient through ``tanh`` vanishes
→ AdamW random-walks → d_seg RISES. So a step-native lever MUST NOT emit a fixed high ``beta``. The two
MEASURED-safe realizations are: (1) **annealed-hosc** — anneal ``beta`` from a small value (1.0, where
``tanh(sin)`` is near-linear and gradients flow) toward the target over training, so the basis
STEP-SHARPENS as the SDF partition pins (sister of the softmax-temp anneal); (2) **step_basis** — the
same anneal driven toward a SHARPER target (a steeper step) with the FINER++ variable-periodic
first-layer bias init MANDATORY (arXiv 2407.19434 — each neuron selects its own phase of the period so
the ensemble does not saturate coherently; the published stability fix). Both realizations ride the
SAME ``tanh(beta * sin)`` mechanism; they differ only in the anneal target and whether FINER
stabilization is forced. A constant ``beta`` (``beta_end == beta_start``) is the FORBIDDEN
saturation-death config and :func:`validate_step_native_config` fails closed on it.

WHY it is byte-neutral AND rule-118 FREE. The activation adds ZERO trainable params (``beta`` /
``omega`` are hyperparameters; the anneal schedule is deterministic; the FINER bias is drawn once at
init from a dedicated RNG stream and then TRAINED). The archive ships only the trained weights — the
step-native chart is a train-time PRIOR, recomputable at decode, not learned/video-derived bytes.

DEFAULT-OFF / byte-identity. This module is the numpy REFERENCE ORACLE + the SAFETY predicate the DSL
``StepNativeActivation`` factory consumes. The trainer already computes ``hosc`` in MLX and already has
the ``--hosc-beta-end`` anneal + ``--finer-bias-init`` machinery; this landing does NOT change that
forward path. The DSL lever, when NOT composed, emits none of these flags → the trainer runs its
default → byte-identical. When composed, it ARMS a MEASURED-safe annealed step-native config (never a
fixed high beta). means != ends: this BUILDS the fireable lever; it makes NO score claim; pointer
UNMOVED.

Pure numpy (no MLX / torch / GPU) so the mechanism + schedule are unit-testable at $0. DSL leg:
``tac.witness_dsl.curriculum_dsl.StepNativeActivation`` (which CONSUMES :func:`validate_step_native_config`
as its fail-closed guard). Equations leg: ``tac.canonical_equations.step_native_activation_edge_optimality_20260707``
(``step_native_activation_edge_optimality_v1`` — this module is its numpy REFERENCE ORACLE + producer).
Trainer flags (EXISTING, grep-verified): ``--activation hosc / --hosc-beta / --hosc-beta-end /
--hosc-beta-anneal / --hosc-omega / --finer-bias-init / --finer-bias-k``.
"""
from __future__ import annotations

import numpy as np

# ── canonical basis kinds (the two MEASURED-safe step-native realizations) ──────────────────────────
STEP_BASIS = "step_basis"
ANNEALED_HOSC = "annealed_hosc"
VALID_BASES = (ANNEALED_HOSC, STEP_BASIS)

VALID_ANNEAL_SHAPES = ("linear", "cosine")

# per-basis default anneal TARGET beta (step_basis drives toward a sharper step than annealed_hosc).
# Both are anneal TARGETS reached from beta_start=1.0 over the schedule — NEVER a fixed constant beta.
_DEFAULT_BETA_END = {ANNEALED_HOSC: 4.0, STEP_BASIS: 8.0}


def hosc_activation(u: np.ndarray, beta: float, omega: float = 1.0) -> np.ndarray:
    """The witness's periodic activation ``a(u) = tanh(beta * sin(omega * u))`` — the numpy twin of the
    trainer MLX ``_act`` hosc branch (``mx.tanh(self.hosc_beta * mx.sin(self.hosc_omega * u))``).

    As ``beta -> inf`` this converges pointwise (a.e.) to :func:`step_native_limit` (a square wave); at
    small ``beta`` it is near-linear in ``sin`` (gradients flow — why the anneal STARTS small). Real
    computation on the actual coordinate features, not a marker (NO-FAKE)."""
    u = np.asarray(u, np.float64)
    if beta <= 0.0:
        raise ValueError(f"hosc_activation: beta must be > 0, got {beta!r}")
    if omega <= 0.0:
        raise ValueError(f"hosc_activation: omega must be > 0, got {omega!r}")
    return np.tanh(float(beta) * np.sin(float(omega) * u))


def sine_basis(u: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """The SMOOTH sine reference ``sin(omega * u)`` — the basis whose Gibbs ringing at the
    piecewise-constant argmax edge the step-native chart AVOIDS. Provided so tests can prove the
    step-native basis genuinely DIFFERS from sine (not a relabeled sine)."""
    u = np.asarray(u, np.float64)
    if omega <= 0.0:
        raise ValueError(f"sine_basis: omega must be > 0, got {omega!r}")
    return np.sin(float(omega) * u)


def step_native_limit(u: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """The ``beta -> inf`` STEP limit ``sign(sin(omega * u))`` — the partition-indicator / square-wave
    chart topology-matched to the piecewise-constant argmax target (no Gibbs, O(1)/edge, L-inf-optimal
    at the edge). This is the target :func:`hosc_activation` step-sharpens toward as ``beta`` anneals
    up. ``sign(0) == 0`` (numpy convention) at the exact zero-crossings (a measure-zero set)."""
    u = np.asarray(u, np.float64)
    if omega <= 0.0:
        raise ValueError(f"step_native_limit: omega must be > 0, got {omega!r}")
    return np.sign(np.sin(float(omega) * u))


def resolve_beta_end(basis: str, beta_end: float | None) -> float:
    """Resolve the anneal TARGET beta: explicit ``beta_end`` if given, else the per-basis default
    (annealed_hosc → 4.0, step_basis → 8.0 = a sharper step). Fail-closed on an unknown basis so a
    typo can never silently pick a default."""
    if basis not in VALID_BASES:
        raise ValueError(f"resolve_beta_end: basis must be one of {VALID_BASES}, got {basis!r}")
    if beta_end is not None:
        return float(beta_end)
    return _DEFAULT_BETA_END[basis]


def beta_anneal_schedule(
    ep: int,
    epochs: int,
    beta_start: float,
    beta_end: float | None,
    shape: str = "linear",
    *,
    anneal_epochs: int | None = None,
) -> float:
    """Annealed ``beta`` at 1-based epoch ``ep`` — the numpy twin of the trainer's
    ``_hosc_beta_for_epoch`` (SAME progress denominator + linear/cosine forms).

    ``beta_end is None`` OR ``beta_end == beta_start`` ⇒ CONSTANT ``beta_start`` (the no-anneal /
    bit-identical path). Otherwise anneal ``beta_start`` (at ``ep == 1``) → ``beta_end`` (at
    ``ep == anneal_epochs or epochs``). ``linear`` (default) or ``cosine`` schedule. Pure; the trainer
    NEVER emits a constant high beta through the DSL lever (:func:`validate_step_native_config` forbids
    ``beta_end == beta_start``), but this reference honours the trainer's constant-beta semantics for
    the resume / legacy paths."""
    if shape not in VALID_ANNEAL_SHAPES:
        raise ValueError(f"beta_anneal_schedule: shape must be one of {VALID_ANNEAL_SHAPES}, got {shape!r}")
    if beta_start <= 0.0:
        raise ValueError(f"beta_anneal_schedule: beta_start must be > 0, got {beta_start!r}")
    if beta_end is None or float(beta_end) == float(beta_start):
        return float(beta_start)                                    # constant-beta (no anneal)
    _ae = int(anneal_epochs) if anneal_epochs else int(epochs)
    prog = (int(ep) - 1) / max(_ae - 1, 1)
    prog = min(max(prog, 0.0), 1.0)                                 # clamp (ep may exceed schedule len)
    b0, b1 = float(beta_start), float(beta_end)
    if shape == "cosine":
        return float(b1 + 0.5 * (b0 - b1) * (1.0 + np.cos(np.pi * prog)))
    return float(b0 + (b1 - b0) * prog)


def validate_step_native_config(
    basis: str,
    beta_start: float,
    beta_end: float | None,
    beta_anneal: str,
    omega: float,
    finer_bias_init: bool,
) -> list[str]:
    """The SAFETY predicate the DSL ``StepNativeActivation`` factory consumes — return a list of
    problems (empty ⇒ OK). Fails on the MEASURED saturation-death config so the lever can NEVER emit a
    fixed high ``beta``.

    Enforced (each maps to a measured/deep-math constraint):
      * ``basis`` ∈ {annealed_hosc, step_basis}                      (no invented kind)
      * ``beta_anneal`` ∈ {linear, cosine}                          (no invented schedule)
      * ``beta_start > 0`` and resolved ``beta_end > 0``            (positivity — hosc undefined at 0)
      * resolved ``beta_end != beta_start``                          (**NEVER fixed beta** — the
        saturation-death; a constant beta is the FORBIDDEN config, DAG FEED 2026-06-25a)
      * ``step_basis`` requires ``finer_bias_init``                  (the FINER++ stability fix is what
        makes the sharper step the "stable trainable-slope survivor")
    """
    problems: list[str] = []
    if basis not in VALID_BASES:
        problems.append(f"basis must be one of {VALID_BASES}, got {basis!r}")
        return problems                                            # can't resolve beta_end on a bad basis
    if beta_anneal not in VALID_ANNEAL_SHAPES:
        problems.append(f"beta_anneal must be one of {VALID_ANNEAL_SHAPES}, got {beta_anneal!r}")
    if beta_start <= 0.0:
        problems.append(f"beta_start must be > 0 (hosc needs a positive slope), got {beta_start!r}")
    resolved_end = resolve_beta_end(basis, beta_end)
    if resolved_end <= 0.0:
        problems.append(f"beta_end must be > 0, got {resolved_end!r}")
    if resolved_end == float(beta_start):
        problems.append(
            f"beta_end ({resolved_end}) == beta_start ({beta_start}) is a FIXED-beta config — the "
            "MEASURED saturation-death (tanh(beta*sin) saturates, gradient vanishes, d_seg RISES; "
            "DAG FEED 2026-06-25a). A step-native lever MUST anneal beta (beta_end != beta_start).")
    if basis == STEP_BASIS and not bool(finer_bias_init):
        problems.append(
            "basis='step_basis' requires finer_bias_init=True — the FINER++ variable-periodic "
            "first-layer bias (arXiv 2407.19434) is the stability fix that lets the sharper step "
            "target survive without coherent saturation. Use basis='annealed_hosc' for a milder "
            "target without mandatory FINER, or enable finer_bias_init.")
    return problems


__all__ = [
    "ANNEALED_HOSC",
    "STEP_BASIS",
    "VALID_ANNEAL_SHAPES",
    "VALID_BASES",
    "beta_anneal_schedule",
    "hosc_activation",
    "resolve_beta_end",
    "sine_basis",
    "step_native_limit",
    "validate_step_native_config",
]
