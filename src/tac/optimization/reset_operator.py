# SPDX-License-Identifier: MIT
"""The BOUNDARY RESET OPERATOR — the optimizer-state hook the reset race requires.

ddm_sb2 (task #819), built against ``.omx/research/ddm_gc15_fresh_vs_warm_20260731.md`` §7.

**What was missing (the NOT-EVEN-DESIGNED gap this closes).** ``experiments/
train_tr1_partition_renderer_mlx.py:1543`` builds ``optim.Adam(learning_rate=cfg.lr)``
— one line, no betas, no ``bias_correction``, no state persistence, no injection point —
and ALL SIX ``save_checkpoint`` call sites pass ``opt_state_flat={}``. So the optimizer
moments are never written and never restored. Of gc15 §7's five pre-registered reset
arms, exactly ONE (arm B, zero-reset) was reachable; the other four had no Lever, no
flag, and no mechanism anywhere. A stub sweep cannot see that class: there is nothing
to detect. This module is the mechanism, and it closes A / B / B' / C outright and
supplies D±'s injection point.

**The operator has three independent knobs** (gc15 §7), plus the bias-correction switch
that turns out to dominate all of them:

======  ==========================  ==========================  ==============
knob    field                       values                      arm it selects
======  ==========================  ==========================  ==============
1       ``what`` (re-anchor)        none / both / momentum      A / B / C
2       ``to`` (v resets TO)        previous / zero / prior     C / B / D±
3       ``structure``               uniform / per_channel       D± only
--      ``bias_correction``         False / True                B'
======  ==========================  ==========================  ==============

**Why bias_correction is the load-bearing one.** MLX's ``Adam`` defaults
``bias_correction=False`` (verified from ``mlx.optimizers.Adam`` source, re-verified
here by :func:`effective_lr_multiplier`'s behaviour test against the real optimizer).
With zeroed moments and a locally-constant gradient the uncorrected update is

    lr * m_t / (sqrt(v_t) + eps)  =  lr * (1-b1^t)/sqrt(1-b2^t) * sign(g)

while the bias-corrected update is exactly ``lr * sign(g)``. Their ratio is

    eta(t) = (1 - b1^t) / sqrt(1 - b2^t)

which is **3.162 at t=1, peaks at 6.569 at t=12**, and decays to 1 with time constant
1/(1-b2). Summing ``eta(t) - 1`` gives ~1203 EXTRA sign-steps of free parameter
displacement per reset (~16.0 epochs at the burn's 75 steps/epoch), **82% of it inside
the first 13 epochs**. That is a real, priced, previously-unnamed impulse — not a
neutral "restart". Every number here is DERIVED in closed form and independently
reproduced by this module's tests against the real MLX optimizer; none is recalled.

**The norm-matching move (gc15 §7, the methodological fix).** A naive D± arm would damp
the spike and lose for an uninformative reason. So each non-uniform arm's prior is scaled
by a single scalar ``s`` chosen (:func:`solve_norm_match_scalar`, monotone bisection on
real gradient magnitudes) so its first-N-step cumulative displacement matches the
zero-reset arm within tolerance. The race then tests **where the kick goes**, never
**how big it is**.

**Fail-closed / NO-FAKE discipline.** ``prior`` mode REFUSES unless a real prior file
supplies a finite, positive, correctly-shaped entry for every trainable parameter — an
absent or partial prior raises rather than silently degrading to zero-reset (which would
be arm B wearing arm D's name: marker-without-mechanism). Norm-matching REFUSES to
report success it did not achieve. Nothing here reads a scorer or claims a score; the
module is pure optimizer-state algebra over arrays the caller supplies.

**Byte-identity.** The default config is ``ARM_B_ZERO_RESET`` — ``what=both, to=zero,
structure=uniform, bias_correction=False`` — which is EXACTLY the incumbent trainer
behaviour (fresh Adam, zeroed moments). :func:`apply_reset` on that config returns an
empty state (nothing to restore => MLX lazily re-inits zeros), and
:func:`ResetOperatorConfig.requires_persistence` is False, so checkpoints keep
``opt_state_flat={}`` and stay byte-identical. Verified by test, not asserted.

Pointer honesty: ``0.1910828242`` [contest-CPU] UNMOVED. This is apparatus.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

__all__ = [
    "ARMS",
    "ARM_A_NO_RESET",
    "ARM_BPRIME_BIAS_CORRECTED",
    "ARM_B_ZERO_RESET",
    "ARM_C_MOMENTUM_ONLY",
    "ARM_DMINUS_SCORER_METRIC",
    "ARM_DPLUS_SCORER_METRIC",
    "DEFAULT_BETAS",
    "VALID_STRUCTURE",
    "VALID_TO",
    "VALID_WHAT",
    "NormMatchResult",
    "ResetOperatorConfig",
    "ResetOperatorError",
    "apply_reset",
    "arm_config",
    "boundary_impulse_epochs",
    "cumulative_displacement_ratio",
    "cumulative_excess_sign_steps",
    "effective_lr_multiplier",
    "load_diagonal_prior",
    "resolve_arm_name",
    "solve_norm_match_scalar",
]


class ResetOperatorError(ValueError):
    """A reset-operator contract violation (fail-closed; never a silent degrade)."""


DEFAULT_BETAS: tuple[float, float] = (0.9, 0.999)

# knob 1 — WHICH moments are re-anchored at the boundary
WHAT_NONE = "none"          # persist m and v (arm A control)
WHAT_BOTH = "both"          # zero m and v (arm B incumbent)
WHAT_MOMENTUM = "momentum"  # zero m, persist v (arm C)
VALID_WHAT = (WHAT_NONE, WHAT_BOTH, WHAT_MOMENTUM)

# knob 2 — WHAT v is reset TO
TO_PREVIOUS = "previous"
TO_ZERO = "zero"
TO_PRIOR = "prior"          # externally-derived diagonal prior (arm D+/D-)
VALID_TO = (TO_PREVIOUS, TO_ZERO, TO_PRIOR)

# knob 3 — the prior's structure
STRUCTURE_UNIFORM = "uniform"
STRUCTURE_PER_CHANNEL = "per_channel"
VALID_STRUCTURE = (STRUCTURE_UNIFORM, STRUCTURE_PER_CHANNEL)


@dataclass(frozen=True, slots=True)
class ResetOperatorConfig:
    """One fully-specified boundary reset operator (gc15 §7's three knobs + bias correction)."""

    what: str = WHAT_BOTH
    to: str = TO_ZERO
    structure: str = STRUCTURE_UNIFORM
    bias_correction: bool = False
    prior_path: str | None = None
    prior_exponent: float = -2.0
    norm_match: bool = True
    norm_match_steps: int = 100
    norm_match_tol: float = 0.02

    def __post_init__(self) -> None:
        if self.what not in VALID_WHAT:
            raise ResetOperatorError(f"what must be one of {VALID_WHAT}, got {self.what!r}")
        if self.to not in VALID_TO:
            raise ResetOperatorError(f"to must be one of {VALID_TO}, got {self.to!r}")
        if self.structure not in VALID_STRUCTURE:
            raise ResetOperatorError(
                f"structure must be one of {VALID_STRUCTURE}, got {self.structure!r}")
        if self.to == TO_PRIOR and not self.prior_path:
            # NO-FAKE: a "scorer-metric" arm with no prior file would silently BE arm B.
            raise ResetOperatorError(
                "to='prior' requires prior_path (an externally-derived diagonal prior); "
                "without it the arm would silently degrade to the zero-reset incumbent")
        if self.to != TO_PRIOR and self.prior_path:
            raise ResetOperatorError(
                f"prior_path is only meaningful with to='prior', got to={self.to!r}")
        if self.what == WHAT_MOMENTUM and self.to not in (TO_PREVIOUS, TO_PRIOR):
            raise ResetOperatorError(
                "what='momentum' means 'zero m, keep v' — to must be 'previous' (arm C) "
                f"or 'prior', got {self.to!r}")
        if self.norm_match_steps < 1:
            raise ResetOperatorError(f"norm_match_steps must be >= 1, got {self.norm_match_steps}")
        if not (0.0 < self.norm_match_tol < 1.0):
            raise ResetOperatorError(
                f"norm_match_tol must be in (0, 1), got {self.norm_match_tol}")
        if not math.isfinite(self.prior_exponent):
            raise ResetOperatorError(f"prior_exponent must be finite, got {self.prior_exponent}")

    @property
    def requires_persistence(self) -> bool:
        """True when the arm needs the PREVIOUS optimizer state, so the trainer must persist it.

        Arms A (``what='none'``) and C (``what='momentum'``, ``to='previous'``) read the prior
        moments; arms B / B' / D± do not. This is what keeps the incumbent's checkpoints
        byte-identical: the default config returns False, so ``opt_state_flat={}`` stands.
        """
        return self.what == WHAT_NONE or self.to == TO_PREVIOUS

    @property
    def is_incumbent(self) -> bool:
        """True for the exact behaviour the trainer had before this module existed (arm B)."""
        return (
            self.what == WHAT_BOTH
            and self.to == TO_ZERO
            and self.structure == STRUCTURE_UNIFORM
            and not self.bias_correction
        )

    def describe(self) -> dict[str, object]:
        """Deterministic provenance dict for telemetry / the run receipt."""
        return {
            "what": self.what,
            "to": self.to,
            "structure": self.structure,
            "bias_correction": self.bias_correction,
            "prior_path": self.prior_path,
            "prior_exponent": self.prior_exponent,
            "norm_match": self.norm_match,
            "norm_match_steps": self.norm_match_steps,
            "requires_persistence": self.requires_persistence,
            "is_incumbent": self.is_incumbent,
            "arm": resolve_arm_name(self),
        }


# ── the six pre-registered arms (gc15 §7) ────────────────────────────────────────────
ARM_A_NO_RESET = ResetOperatorConfig(what=WHAT_NONE, to=TO_PREVIOUS, norm_match=False)
ARM_B_ZERO_RESET = ResetOperatorConfig(what=WHAT_BOTH, to=TO_ZERO, norm_match=False)
ARM_BPRIME_BIAS_CORRECTED = ResetOperatorConfig(
    what=WHAT_BOTH, to=TO_ZERO, bias_correction=True, norm_match=False)
ARM_C_MOMENTUM_ONLY = ResetOperatorConfig(
    what=WHAT_MOMENTUM, to=TO_PREVIOUS, norm_match=False)


def _d_arm(exponent: float) -> ResetOperatorConfig:
    return ResetOperatorConfig(
        what=WHAT_BOTH, to=TO_PRIOR, structure=STRUCTURE_PER_CHANNEL,
        prior_path="<REQUIRED>", prior_exponent=exponent, norm_match=True)


ARM_DPLUS_SCORER_METRIC = _d_arm(+2.0)
ARM_DMINUS_SCORER_METRIC = _d_arm(-2.0)

ARMS: dict[str, ResetOperatorConfig] = {
    "A": ARM_A_NO_RESET,
    "B": ARM_B_ZERO_RESET,
    "Bprime": ARM_BPRIME_BIAS_CORRECTED,
    "C": ARM_C_MOMENTUM_ONLY,
    "Dplus": ARM_DPLUS_SCORER_METRIC,
    "Dminus": ARM_DMINUS_SCORER_METRIC,
}


def arm_config(name: str, *, prior_path: str | None = None) -> ResetOperatorConfig:
    """Resolve a pre-registered arm by name, binding ``prior_path`` for the D± arms.

    Fails closed on an unknown name (listing the real ones) and on a D-arm without a prior.
    """
    if name not in ARMS:
        raise ResetOperatorError(
            f"unknown reset arm {name!r}; pre-registered arms: {', '.join(sorted(ARMS))}")
    cfg = ARMS[name]
    if cfg.to == TO_PRIOR:
        if not prior_path:
            raise ResetOperatorError(
                f"arm {name!r} is a scorer-metric arm and requires prior_path")
        return replace(cfg, prior_path=str(prior_path))
    if prior_path:
        raise ResetOperatorError(f"arm {name!r} takes no prior_path (got {prior_path!r})")
    return cfg


def resolve_arm_name(cfg: ResetOperatorConfig) -> str | None:
    """The pre-registered arm name matching ``cfg`` (ignoring the bound prior path), else None."""
    for name, arm in ARMS.items():
        if arm.to == TO_PRIOR:
            if (cfg.to == TO_PRIOR and cfg.what == arm.what
                    and cfg.structure == arm.structure
                    and cfg.prior_exponent == arm.prior_exponent
                    and cfg.bias_correction == arm.bias_correction):
                return name
        elif (cfg.what, cfg.to, cfg.structure, cfg.bias_correction) == (
                arm.what, arm.to, arm.structure, arm.bias_correction):
            return name
    return None


# ── the closed-form impulse price (gc15 §5.1, re-derived here) ───────────────────────
def effective_lr_multiplier(step: int, betas: Sequence[float] = DEFAULT_BETAS) -> float:
    """``eta(t) = (1-b1^t)/sqrt(1-b2^t)`` — uncorrected-vs-bias-corrected Adam step ratio.

    At a locally-constant gradient this is EXACTLY the factor by which an un-bias-corrected
    Adam over-steps a bias-corrected one after a moment reset. ``eta(1)=3.162``,
    ``max eta = eta(12) = 6.569``, ``eta -> 1``. ``step`` is 1-based (MLX's first update
    is step 1). Verified against the real ``mlx.optimizers.Adam`` by behaviour test.
    """
    if step < 1:
        raise ResetOperatorError(f"step is 1-based; got {step}")
    b1, b2 = _validated_betas(betas)
    return (1.0 - b1 ** step) / math.sqrt(1.0 - b2 ** step)


def _validated_betas(betas: Sequence[float]) -> tuple[float, float]:
    if len(tuple(betas)) != 2:
        raise ResetOperatorError(f"betas must be a 2-sequence, got {betas!r}")
    b1, b2 = (float(betas[0]), float(betas[1]))
    for nm, b in (("b1", b1), ("b2", b2)):
        if not (0.0 <= b < 1.0):
            raise ResetOperatorError(f"{nm} must be in [0, 1), got {b}")
    return b1, b2


def cumulative_excess_sign_steps(
    n_steps: int, betas: Sequence[float] = DEFAULT_BETAS
) -> float:
    """``sum_{t=1..n} (eta(t) - 1)`` — the EXTRA sign-steps of displacement one reset buys.

    This is the price of a reset in the only unit that composes: parameter displacement
    measured in ``lr``-sized sign-steps. It converges (~1203 at the default betas), so the
    impulse is set by the RESET COUNT, not by the window length — which is why gc14's
    per-restart impulse is constant and its ``r`` measures the landscape's diminishing
    return to a fixed displacement, not a decaying kick.
    """
    if n_steps < 1:
        raise ResetOperatorError(f"n_steps must be >= 1, got {n_steps}")
    b1, b2 = _validated_betas(betas)
    return float(sum(effective_lr_multiplier(t, (b1, b2)) - 1.0 for t in range(1, n_steps + 1)))


def boundary_impulse_epochs(
    n_steps: int, steps_per_epoch: int, betas: Sequence[float] = DEFAULT_BETAS
) -> float:
    """The reset's price expressed in EPOCHS of free movement (excess sign-steps / steps-per-epoch)."""
    if steps_per_epoch < 1:
        raise ResetOperatorError(f"steps_per_epoch must be >= 1, got {steps_per_epoch}")
    return cumulative_excess_sign_steps(n_steps, betas) / float(steps_per_epoch)


# ── norm matching (gc15 §7's load-bearing methodological move) ───────────────────────
def cumulative_displacement_ratio(
    v0: np.ndarray,
    grad_magnitude: np.ndarray,
    n_steps: int,
    *,
    betas: Sequence[float] = DEFAULT_BETAS,
    eps: float = 1e-8,
    bias_correction: bool = False,
) -> float:
    """Cumulative ``sum_t |m_t / (sqrt(v_t) + eps)|`` over the first ``n_steps``, in ``lr`` units.

    Simulates Adam's own recursion under a locally-constant gradient of magnitude
    ``grad_magnitude`` starting from ``v0`` (and ``m=0``) — the honest first-order model of
    a boundary kick. Returns the L2 norm over parameters of the per-parameter cumulative
    displacement, so it is directly comparable across arms. Pure numpy; no optimizer needed.
    """
    b1, b2 = _validated_betas(betas)
    g = np.abs(np.asarray(grad_magnitude, dtype=np.float64))
    v = np.asarray(v0, dtype=np.float64)
    if v.shape != g.shape:
        raise ResetOperatorError(
            f"v0 shape {v.shape} must match grad_magnitude shape {g.shape}")
    if np.any(v < 0.0) or not np.all(np.isfinite(v)):
        raise ResetOperatorError("v0 must be finite and non-negative")
    if not np.all(np.isfinite(g)):
        raise ResetOperatorError("grad_magnitude must be finite")
    if n_steps < 1:
        raise ResetOperatorError(f"n_steps must be >= 1, got {n_steps}")
    m = np.zeros_like(g)
    total = np.zeros_like(g)
    for t in range(1, n_steps + 1):
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        if bias_correction:
            num = m / (1.0 - b1 ** t)
            den = np.sqrt(v) / math.sqrt(1.0 - b2 ** t) + eps
        else:
            num, den = m, np.sqrt(v) + eps
        total += num / den
    return float(np.linalg.norm(total))


@dataclass(frozen=True, slots=True)
class NormMatchResult:
    """Outcome of the §7 norm match: the scalar, the achieved ratio, and whether it converged."""

    scalar: float
    achieved_ratio: float
    target_displacement: float
    matched_displacement: float
    converged: bool
    iterations: int


def solve_norm_match_scalar(
    prior: np.ndarray,
    grad_magnitude: np.ndarray,
    *,
    n_steps: int = 100,
    tol: float = 0.02,
    betas: Sequence[float] = DEFAULT_BETAS,
    eps: float = 1e-8,
    max_iter: int = 200,
) -> NormMatchResult:
    """Find ``s`` such that ``v0 = s * prior`` matches the ZERO-reset arm's first-``n_steps``
    cumulative displacement to within ``tol`` (gc15 §7).

    The displacement is strictly DECREASING in ``s`` (a larger ``v`` shrinks every step), so a
    bracket-then-bisect on real gradient magnitudes converges deterministically. This is what
    makes the D± race a test of DIRECTION rather than MAGNITUDE, defusing the operator's
    mandatory falsifier by construction instead of by argument.

    Fails closed: a non-positive / non-finite prior raises, and a bracket that cannot be
    established returns ``converged=False`` (the caller must refuse the arm, never proceed
    with an unmatched kick and call it matched).
    """
    p = np.asarray(prior, dtype=np.float64)
    g = np.abs(np.asarray(grad_magnitude, dtype=np.float64))
    if p.shape != g.shape:
        raise ResetOperatorError(f"prior shape {p.shape} must match grad shape {g.shape}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise ResetOperatorError("prior must be finite and strictly positive everywhere")
    if not (0.0 < tol < 1.0):
        raise ResetOperatorError(f"tol must be in (0, 1), got {tol}")

    target = cumulative_displacement_ratio(
        np.zeros_like(g), g, n_steps, betas=betas, eps=eps)

    def disp(s: float) -> float:
        return cumulative_displacement_ratio(s * p, g, n_steps, betas=betas, eps=eps)

    # s=0 reproduces the target exactly; displacement decreases as s grows. Bracket upward.
    lo, hi = 0.0, 1.0
    it = 0
    while disp(hi) > target * (1.0 - tol):
        hi *= 4.0
        it += 1
        if it > 60 or not math.isfinite(hi):
            return NormMatchResult(0.0, 1.0, target, target, False, it)
    for _ in range(max_iter):
        it += 1
        mid = 0.5 * (lo + hi)
        d = disp(mid)
        ratio = d / target if target > 0 else float("inf")
        if abs(ratio - 1.0) <= tol:
            return NormMatchResult(mid, ratio, target, d, True, it)
        if d > target:
            lo = mid
        else:
            hi = mid
    d = disp(hi)
    return NormMatchResult(hi, d / target if target > 0 else float("inf"), target, d, False, it)


# ── the diagonal prior (arm D±'s externally-derived input) ───────────────────────────
def load_diagonal_prior(
    path: str | Path, param_shapes: Mapping[str, tuple[int, ...]], *, exponent: float = -2.0
) -> dict[str, np.ndarray]:
    """Load an externally-derived diagonal prior and raise it to ``exponent``, fail-closed.

    The npz must carry ONE finite, strictly-positive entry per trainable parameter key, each
    broadcastable to that parameter's shape (a per-channel vector broadcasts along the last
    axis — knob 3's ``per_channel`` structure). A missing / extra / non-finite / non-positive /
    non-broadcastable entry RAISES: an arm named "scorer-metric" that silently fell back to
    zeros would be arm B wearing arm D's name (NO-FAKE marker-without-mechanism).

    The PROVIDER (which maps a scorer-side capacity such as #725's per-channel BN capacities
    onto this trainer's parameter tree) is deliberately NOT here — it is a separate, named,
    chartered build. This loader is the injection contract it must satisfy.
    """
    p = Path(path)
    if not p.is_file():
        raise ResetOperatorError(f"diagonal prior file not found: {p}")
    if not math.isfinite(exponent):
        raise ResetOperatorError(f"exponent must be finite, got {exponent}")
    with np.load(p, allow_pickle=False) as z:
        keys = set(z.files)
        want = set(param_shapes)
        missing = sorted(want - keys)
        if missing:
            raise ResetOperatorError(
                f"diagonal prior {p} is missing {len(missing)} trainable parameter(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        extra = sorted(keys - want)
        if extra:
            raise ResetOperatorError(
                f"diagonal prior {p} carries {len(extra)} key(s) that are not trainable "
                f"parameters: {extra[:5]}{'...' if len(extra) > 5 else ''}")
        out: dict[str, np.ndarray] = {}
        for k in sorted(want):
            arr = np.asarray(z[k], dtype=np.float64)
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ResetOperatorError(
                    f"diagonal prior entry {k!r} must be finite and strictly positive")
            try:
                arr = np.broadcast_to(arr, param_shapes[k]).astype(np.float32)
            except ValueError as exc:
                raise ResetOperatorError(
                    f"diagonal prior entry {k!r} shape {arr.shape} is not broadcastable to "
                    f"parameter shape {param_shapes[k]}") from exc
            out[k] = np.power(arr, exponent, dtype=np.float32)
    return out


# ── the operator itself ──────────────────────────────────────────────────────────────
def apply_reset(
    previous: Mapping[str, np.ndarray],
    param_shapes: Mapping[str, tuple[int, ...]],
    cfg: ResetOperatorConfig,
    *,
    prior: Mapping[str, np.ndarray] | None = None,
    scalar: float = 1.0,
) -> dict[str, np.ndarray]:
    """Produce the boundary optimizer state for ``cfg`` — the flat ``{"<param>::m"/"::v"}`` map.

    ``previous`` is the flat state carried across the boundary (empty when none was persisted).
    Returns a flat state to RESTORE into the optimizer; an EMPTY return means "let the optimizer
    lazily initialize" — which is exactly the incumbent zero-reset, so arm B costs nothing and
    changes no bytes.

    Fail-closed: an arm that needs the previous state and did not get it RAISES (rather than
    silently becoming arm B — the precise way a reset race would quietly measure the same arm
    five times).
    """
    if cfg.to == TO_PRIOR and prior is None:
        raise ResetOperatorError("to='prior' requires a loaded prior mapping")
    if cfg.requires_persistence and not previous:
        raise ResetOperatorError(
            f"arm {resolve_arm_name(cfg) or cfg} needs the previous optimizer state but none "
            "was persisted; enable opt-state persistence before firing this arm (refusing "
            "rather than silently degrading to the zero-reset incumbent)")
    if scalar <= 0.0 or not math.isfinite(scalar):
        raise ResetOperatorError(f"scalar must be finite and positive, got {scalar}")

    if cfg.what == WHAT_BOTH and cfg.to == TO_ZERO:
        return {}  # the incumbent: lazily re-initialized zeros

    out: dict[str, np.ndarray] = {}
    for key, shape in sorted(param_shapes.items()):
        mk, vk = f"{key}::m", f"{key}::v"
        # knob 1: m
        if cfg.what == WHAT_NONE:
            m = _require(previous, mk, shape, "previous m")
        else:  # both / momentum both zero m
            m = np.zeros(shape, dtype=np.float32)
        # knob 2: v
        if cfg.to == TO_PREVIOUS:
            v = _require(previous, vk, shape, "previous v")
        elif cfg.to == TO_ZERO:
            v = np.zeros(shape, dtype=np.float32)
        else:
            if key not in prior:  # type: ignore[operator]
                raise ResetOperatorError(f"prior is missing trainable parameter {key!r}")
            v = (float(scalar) * np.asarray(prior[key], dtype=np.float32)).astype(np.float32)
        out[mk] = m
        out[vk] = v
    return out


def _require(
    state: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...], what: str
) -> np.ndarray:
    if key not in state:
        raise ResetOperatorError(f"{what} missing for {key!r} in the persisted optimizer state")
    arr = np.asarray(state[key], dtype=np.float32)
    if tuple(arr.shape) != tuple(shape):
        raise ResetOperatorError(
            f"{what} for {key!r} has shape {arr.shape}, expected {tuple(shape)}")
    if not np.all(np.isfinite(arr)):
        raise ResetOperatorError(f"{what} for {key!r} is not finite")
    return arr
