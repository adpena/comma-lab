"""tac.witness_autoconfig — the reusable ``clip -> witness_config`` actuator.

This module wraps the value-GENERATORS measured/recalled this session into ONE
clip-agnostic pipeline that turns a GT-cache (any clip, or the whole corpus)
into a launch-ready :class:`WitnessConfig` for
``experiments/train_levelset_witness_realized_through_R_mlx.py``.

Design contract (operator 2026-06-30 "build reusable AUTOMATED value-generators,
not ad hoc constants; overfit the contest BUT generalize to any clip / corpus"):

  * Every value is DERIVED from a generator (a measurement OR a recalled proven
    config), never invented. Each generator is a small, tested function.
  * Pure numpy / CPU. The module IMPORTS and ``derive_config`` RUNS with ZERO
    GPU and no heavy I/O: when a generator's heavy input (a trained code matrix,
    a byte-close sweep) is absent, it returns the measured-CONSTANT fallback and
    flags ``source="fallback_constant"``. NO-FAKE: a generator that cannot
    actually measure NEVER silently fabricates a measurement.
  * Each value carries provenance: which generator produced it, the source class,
    its corpus-PORTABILITY classification, and an advisory tag.

means != ends: ``derive_config`` produces a config (a MEANS). The only END is a
byte-closed n600 exact row < 0.19110 from ``upstream/evaluate.py`` (contest-CPU
and/or CUDA, NEVER MPS). This module derives + flag-validates a launch command;
it does NOT launch or score anything. Every value here is
``[macOS-MLX advisory / design]`` and NON-PROMOTABLE.

Provenance anchors (this session, all read-only):
  * proven n200 Muon arm command (run_muon.log) — best realized d_seg 0.003698
    @ ep1000: ``--mod-dim 32 --hidden-dim 96 --verdict-pairs 96 --muon-lr 0.002``,
    curriculum tau@300/l7@600/muon@726, ALL surgical levers + DM1 OFF.
  * n600 v2 design + recursive review (4 binding revisions):
    mod-dim 26 / hidden-dim 96 (NOT 120) / muon-lr 0.002 / verdict-pairs 96;
    surgical levers + DM1 OFF for the attribution-clean FIRST launch.
  * held-out n400 generalization probe: the portable/clip-specific split.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

__all__ = [
    "ProvenancedValue",
    "WitnessConfig",
    "Portability",
    "intrinsic_dim",
    "whitney_mod_dim",
    "mod_dim_generator",
    "hidden_dim_generator",
    "curriculum_schedule",
    "muon_lr_generator",
    "verdict_pairs_generator",
    "lever_priors",
    "warp_priors",
    "portability_split",
    "derive_config",
    "derive_sealed_205_config",
    "derive_store_nothing_205_config",
]

# --------------------------------------------------------------------------
# source-class + portability vocab (the NO-FAKE provenance taxonomy)
# --------------------------------------------------------------------------
SRC_MEASURED = "measured"               # measured this session from real data
SRC_RECALLED = "recalled_proven"        # recalled from the proven 0.003698 arm
SRC_HELDOUT = "held_out_corrected"      # corrected by the held-out n400 probe
SRC_FALLBACK = "fallback_constant"      # heavy input absent -> measured constant
SRC_DESIGN = "design"                   # design-level (v2, not yet a trainer flag)

ADVISORY_TAG = "[macOS-MLX advisory / design]"


class Portability:
    """Corpus-generalization classification of a knob (the production-path core).

    SCORER_FIXED   — invariant across EVERY contest video (frozen SegNet/PoseNet,
                     the R operator, the 37.5M rate normalizer). Ship as a corpus
                     constant; never re-measure.
    DOMAIN         — fundamental to any dashcam clip (codim-1 boundaries ->
                     directional basis; se(3) screw ego-motion -> warp; ground
                     homography + deg-3 lanes -> lane-prior). Ship as corpus
                     default; the FORM transfers, only fine coefficients re-fit.
    INSTANCE       — this-clip-conditioned NUMBERS (mod = code SVD/intrinsic
                     floor; hidden = RD-waterfill vs the normalizer; schedule =
                     root-tracking on this clip's critical-tau). RE-MEASURE per
                     clip; the generator emits that clip's value.
    """

    SCORER_FIXED = "scorer_fixed"
    DOMAIN = "domain_fundamental"
    INSTANCE = "instance_conditioned"


@dataclass(frozen=True)
class ProvenancedValue:
    """A single derived config value plus its full provenance.

    NO-FAKE: ``source == SRC_FALLBACK`` means the generator's heavy input was
    absent and this is the recalled measured CONSTANT, not a fresh measurement.
    """

    value: object
    source: str
    provenance: str
    portability: str
    tag: str = ADVISORY_TAG

    @property
    def is_fallback(self) -> bool:
        return self.source == SRC_FALLBACK


# --------------------------------------------------------------------------
# Generator 1 — mod_dim via intrinsic dimension + Whitney embedding
# --------------------------------------------------------------------------
def _twonn_intrinsic_dim(X: np.ndarray, discard_frac: float = 0.1) -> float:
    """TwoNN estimator (Facco et al. 2017): m = slope of log(1-F(mu)) vs log(mu),
    mu = r2/r1 (2nd/1st NN distance ratio). Parameter-free nonlinear ID. Pure
    numpy; deterministic (no RNG). Mirrors the in-tree probe estimator."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 12:
        return float("nan")
    sq = (X**2).sum(axis=1)
    with np.errstate(all="ignore"):
        D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(D2, np.inf)
    D = np.sqrt(np.clip(D2, 0.0, None))
    D.sort(axis=1)
    r1, r2 = D[:, 0], D[:, 1]
    good = (r1 > 1e-12) & (r2 > 1e-12)
    mu = r2[good] / r1[good]
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    if mu.size < 10:
        return float("nan")
    mu_sorted = np.sort(mu)[: int(mu.size * (1 - discard_frac))]
    N = mu_sorted.size
    F = np.arange(1, N + 1) / (N + 1)
    x = np.log(mu_sorted)
    y = -np.log(1.0 - F)
    return float((x * y).sum() / (x * x).sum())


def _mle_intrinsic_dim(X: np.ndarray, k1: int = 5, k2: int = 15) -> float:
    """Levina-Bickel 2004 MLE intrinsic dim with MacKay-Ghahramani averaging
    (average of inverse-dim over k). Pure numpy; deterministic."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if X.ndim != 2 or n < 12:
        return float("nan")
    sq = (X**2).sum(axis=1)
    with np.errstate(all="ignore"):
        D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(D2, np.inf)
    D = np.sqrt(np.clip(D2, 0.0, None))
    D.sort(axis=1)
    k2 = min(k2, n - 1)
    k1 = min(k1, k2 - 1) if k2 > 1 else 1
    inv_m = []
    for k in range(k1, k2 + 1):
        Tk = D[:, k - 1 : k]
        logr = np.log(np.maximum(D[:, :k], 1e-12))
        logTk = np.log(np.maximum(Tk, 1e-12))
        s = (logTk - logr[:, : k - 1]).sum(axis=1)
        good = s > 1e-9
        mk = (k - 1) / s[good]
        mk = mk[np.isfinite(mk) & (mk > 0)]
        if mk.size:
            inv_m.append(np.mean(1.0 / mk))
    if not inv_m:
        return float("nan")
    return float(1.0 / np.mean(inv_m))


# the recalled measured intrinsic dim of the contest clip's per-pair code +
# GT partition manifold (recursive review TwoNN ~10.93/9.79, MLE ~8.30/9.27 ->
# they AGREE at ~9). The fallback when no live code matrix is provided.
_MEASURED_INTRINSIC_DIM = 9.0


def intrinsic_dim(code_matrix: np.ndarray | None = None) -> ProvenancedValue:
    """Generator: nonlinear intrinsic dimension ``m`` of the per-pair code
    manifold (TwoNN+MLE averaged). If a small code-sample matrix is given,
    MEASURE; else fall back to the recalled measured constant m=9 (NO-FAKE flag).

    The capacity-relevant quantity is the NONLINEAR intrinsic dim (~9), NOT the
    linear participation-ratio (~26, an overcount of a curved manifold) — the
    recursive review measured this two independent ways.
    """
    if code_matrix is not None:
        est = [e for e in (_twonn_intrinsic_dim(code_matrix), _mle_intrinsic_dim(code_matrix))
               if np.isfinite(e)]
        if est:
            m = float(np.mean(est))
            return ProvenancedValue(
                value=m, source=SRC_MEASURED,
                provenance=f"TwoNN+MLE nonlinear ID on a {code_matrix.shape} code matrix "
                           f"= {m:.2f} (mean of {len(est)} finite estimators)",
                portability=Portability.INSTANCE,
            )
    return ProvenancedValue(
        value=_MEASURED_INTRINSIC_DIM, source=SRC_FALLBACK,
        provenance="recalled measured nonlinear ID m~9 (review TwoNN/MLE on proven "
                   "code + GT manifold agree at ~9); no live code matrix supplied",
        portability=Portability.INSTANCE,
    )


def whitney_mod_dim(m: float) -> int:
    """Whitney embedding floor for a curved m-manifold, clamped to the launch band.

    A curved m-manifold needs up to ``2m+1`` LINEAR ambient dims to embed without
    self-intersection. We clamp to [19, 26]: 19 is the aggressive theta*-floor
    (Whitney for m~9), 26 is the headroom ceiling (Whitney for the composite
    m~13 = lane-orbit ~8 (+) screw ~6, since 2*13+1=27 -> clamp 26). Going below
    ~19 risks under-embedding; above 26 wastes rate.
    """
    return int(np.clip(round(2.0 * float(m) + 1.0), 19, 26))


def mod_dim_generator(code_matrix: np.ndarray | None = None, *, overfit: bool = True
                      ) -> ProvenancedValue:
    """Generator: the shipped ``--mod-dim``.

    overfit=True  -> 26 (headroom ceiling = Whitney upper clamp; covers the
                     composite m~13; the n600 v2 review revision #3 value).
    overfit=False -> the Whitney floor for the MEASURED m (19 for m~9): the
                     aggressive theta* rate-saving floor.
    """
    m_pv = intrinsic_dim(code_matrix)
    floor = whitney_mod_dim(m_pv.value)
    value = 26 if overfit else floor
    src = SRC_HELDOUT if overfit else (SRC_MEASURED if not m_pv.is_fallback else SRC_FALLBACK)
    return ProvenancedValue(
        value=value, source=src,
        provenance=(f"intrinsic m={m_pv.value:.2f} ({m_pv.source}); Whitney floor "
                    f"clamp(2m+1,19,26)={floor}; "
                    + ("ship 26 (overfit headroom; composite m~13->27->26; "
                       "aggressive theta* floor 19-21)" if overfit
                       else f"aggressive floor {floor}")),
        portability=Portability.INSTANCE,
    )


# --------------------------------------------------------------------------
# Generator 2 — hidden_dim via rate / capacity (byte-close)
# --------------------------------------------------------------------------
_PROVEN_HIDDEN_DIM = 96  # proven arm + review RD-optimum (~122KB); 26/96 ~ -0.004 S


def hidden_dim_generator(byte_close_result: dict | None = None, *, overfit: bool = True
                         ) -> ProvenancedValue:
    """Generator: the shipped ``--hidden-dim``.

    ``byte_close_result`` (optional): a mapping {hidden_dim -> total_archive_bytes}
    from a real ``quantize_levelset_blob`` sweep -> pick the RATE-minimal arm
    (RATE is the binding sub-0.15 lever). Absent -> proven/review constant 96.

    Trunk params scale ~ hidden^2 (+56% for 96->120, robust to compressibility),
    so the hidden bump is the rate-costly half: review byte-close showed 26/96
    ~90.6KB (rate -0.004 S vs proven) while 26/120 ~111.9KB (+0.010 S), and the
    nonlinear-ID test showed hidden-96 capacity is already adequate.
    """
    if byte_close_result:
        best = min(byte_close_result, key=byte_close_result.get)
        return ProvenancedValue(
            value=int(best), source=SRC_MEASURED,
            provenance=f"RD-min of byte-close sweep {dict(sorted(byte_close_result.items()))} "
                       f"-> hidden={best} ({byte_close_result[best]} bytes)",
            portability=Portability.INSTANCE,
        )
    return ProvenancedValue(
        value=_PROVEN_HIDDEN_DIM, source=SRC_FALLBACK,
        provenance="proven arm hidden=96 = review RD-optimum (~122KB); 26/96 is a "
                   "rate WIN (-0.004 S) vs 26/120 (+0.010 S); no byte-close sweep supplied",
        portability=Portability.INSTANCE,
    )


# --------------------------------------------------------------------------
# Generator 3 — curriculum stage starts via the annealing schedule
# --------------------------------------------------------------------------
# proven fractions of the total epoch budget (CE -> tau@0.300 -> l7@0.600 ->
# muon@0.726), recalled from the 0.003698 arm (epochs=1000).
_TAU_FRAC, _L7_FRAC, _MUON_FRAC = 0.300, 0.600, 0.726


def curriculum_schedule(epochs: int) -> dict[str, ProvenancedValue]:
    """Generator: curriculum stage-start epochs (tau_softplus / l7 / muon).

    Exact proven values at epochs==1000 (tau@300, l7@600, muon@726); scaled
    proportionally otherwise. The curriculum is a homotopy of relaxations
    (CE -> tau -> l7 -> Muon) = deterministic annealing; the fractions are the
    proven schedule's critical-tau placements.
    """
    if epochs == 1000:
        tau, l7, muon = 300, 600, 726
        src, note = SRC_RECALLED, "proven epochs=1000 schedule (CE300/Tau300/L7126/Muon274)"
    else:
        tau = int(round(_TAU_FRAC * epochs))
        l7 = int(round(_L7_FRAC * epochs))
        muon = int(round(_MUON_FRAC * epochs))
        src = SRC_DESIGN
        note = (f"proportional scale of proven fractions "
                f"({_TAU_FRAC}/{_L7_FRAC}/{_MUON_FRAC}) to epochs={epochs}")
    return {
        "tau_softplus_start_epoch": ProvenancedValue(tau, src, note, Portability.INSTANCE),
        "l7_start_epoch": ProvenancedValue(l7, src, note, Portability.INSTANCE),
        "muon_start_epoch": ProvenancedValue(muon, src, note, Portability.INSTANCE),
    }


# --------------------------------------------------------------------------
# Generator 4 — muon_lr (the proven finisher)
# --------------------------------------------------------------------------
def muon_lr_generator() -> ProvenancedValue:
    """Generator: ``--muon-lr``. The proven 0.003698 arm used 0.002 (run_muon.log
    + muon_finisher_switch JSON). The trainer default None -> 0.1*lr = 1e-4 is
    20x too low and would NOT reproduce the measured descent (review CRITICAL #1).
    """
    return ProvenancedValue(
        value=0.002, source=SRC_RECALLED,
        provenance="proven arm muon_lr=0.002 (run_muon.log + muon_finisher_switch JSON); "
                   "inside the help's optimal-form band 1e-3..5e-3",
        portability=Portability.SCORER_FIXED,
    )


def verdict_pairs_generator(num_pairs: int) -> ProvenancedValue:
    """Generator: ``--verdict-pairs``. The proven arm verdicted on 96 pairs; the
    trainer default 24 is a degraded, non-apples-to-apples realized-d_seg verdict
    at n600 (telemetry-accuracy discipline). Use 96 (review revision #4)."""
    return ProvenancedValue(
        value=96, source=SRC_RECALLED,
        provenance=f"proven arm --verdict-pairs 96 (default 24 is degraded telemetry at "
                   f"num_pairs={num_pairs})",
        portability=Portability.SCORER_FIXED,
    )


# --------------------------------------------------------------------------
# Generator 5 — lever priors via attribution (Fisher / margin-saliency /
#               birth-death / screw-fit)
# --------------------------------------------------------------------------
# the measured per-stage attribution verdict (witness_per_stage_attribution +
# north-star causatives + birth-death persistence + screw-warp through-R).
_DEFAULT_ATTRIBUTION = {
    "Road": {"verdict": "STUCK", "mechanism": "causal-warp",
             "note": "error mass GROWING 21.5%->38.8%, margins flat/neg; "
                     "52.7% ego-pose-explained -> se(3) screw warp (~0 byte), not per-pixel FiLM"},
    "Lane": {"verdict": "PRIMED", "mechanism": "trained-residual",
             "note": "47.2%->24.3% mislabeled, margins moving to GT; ~39% flip "
                     "warp-UNEXPLAINABLE + R-recoverable 85% -> learned/stored residual"},
}


def lever_priors(attribution: dict | None = None, *, overfit: bool = True) -> dict:
    """Generator: which surgical levers to keep OFF for the attribution-clean
    FIRST run vs queue ON later, derived from the per-stage attribution verdict.

    The proven 0.003698 arm had ALL 5 surgical levers + DM1 OFF. Stacking 5
    unproven theta*-pending levers + arch change + scale change = a 3-way
    confound with no clean attribution -> launch attribution-clean FIRST; the
    levers (loss/projection-only, add no params) then land as a shape-compatible
    WARM-START re-treatment once #183 fills theta*.
    """
    attr = attribution if attribution is not None else _DEFAULT_ATTRIBUTION
    return {
        # attribution-clean first launch -> all surgical levers + DM1 OFF.
        "surgical_levers_enabled": False,
        "dm1_enabled": False,
        "attribution": attr,
        # per-lever routing (OFF now; ON in the warm-start re-treatment at theta*).
        "deferred_levers": {
            "margin_saliency": "all-class flip-band defense (class-agnostic); "
                               "theta*-pending weight; warm-start re-treat",
            "lane_thin": "birth-death: thin dashes <3px flip ~92%; radius~2px tube; "
                         "Lane PRIMED -> warm-start re-treat",
            "hardness": "waterfill per-pair code budget to hard pairs; "
                        "+~50% wall-clock tax; warm-start re-treat",
            "uniward": "Fridrich texture-aware margin-cost on STUCK texture; warm-start",
            "film_stiefel_dm1": "raises LINEAR PR(M) but nonlinear capacity already "
                                "adequate (~9 << 26) -> likely a non-problem; OFF",
        },
        # the DOMAIN-fundamental geometric guards that ARE on (free, generic).
        "active_geometric_priors": {
            "lane_prior_phi1": "openpilot deg-3 centerline = Road<->Lane separatrix; "
                               "FREE generic geometry; ON",
            "directional_basis": "self-orient curvelet basis (-48% exponent); ON",
            "structured_init": "FEED-ef static-core partition init from cached L*; "
                               "rule-118 FREE; ON",
            "palette_anchor": "init palette to per-class mean GT RGB; ON",
        },
        "rationale": "attribution-clean isolates scale+arch; surgical levers re-treat "
                     "as warm-start (no new params, shape-compatible)"
        if overfit else "aggressive: still attribution-clean first (proven discipline)",
    }


# --------------------------------------------------------------------------
# Generator 6 — warp priors via screw-fit (design-level v2 vehicle)
# --------------------------------------------------------------------------
def warp_priors() -> dict:
    """Generator: per-class se(3) screw-warp / depth init priors (Chasles screw =
    one twist read out at depth strata). DESIGN-LEVEL: the v2 causal vehicle is
    NOT yet a trainer flag, so these are design fields the v2 build consumes; they
    are NOT emitted into the current INR launch command.
    """
    return {
        "status": "design_v2_not_a_trainer_flag",
        "per_class": {
            "Road": {"warp": "ground_homography",
                     "note": "near/ground -> full translational parallax; HELPS Road "
                             "-8% through R; free dual-use d_pose"},
            "sky": {"warp": "rotation_only", "note": "far -> translation/inf -> 0"},
            "hood": {"warp": "identity", "note": "ego-static, moves with camera"},
            "Lane": {"warp": "none_learned_residual",
                     "note": "~39% flip warp-UNEXPLAINABLE -> witness/store residual"},
        },
        "factor": "one SE(3) screw (twist xi, ~6 DOF/frame) x per-class depth; "
                  "temporal worldline = exp(t*xi) geodesic (the rate-half v2 factor)",
        "portability": Portability.DOMAIN,
    }


# --------------------------------------------------------------------------
# Generator 7 — portable vs clip-specific split (the corpus-generalization core)
# --------------------------------------------------------------------------
def portability_split() -> dict:
    """Generator: per-knob corpus-portability classification (held-out n400 probe).

    THIS is the generalizable comma.ai production-path core: for a NEW clip (or
    the whole corpus), the SCORER_FIXED knobs ship as constants, the DOMAIN knobs
    ship as defaults whose FORM transfers (only fine coefficients re-fit), and the
    INSTANCE knobs are RE-MEASURED by re-running the $0 generators on that clip.
    """
    return {
        # scorer-fixed: invariant across every contest video.
        "muon_lr": Portability.SCORER_FIXED,
        "verdict_pairs": Portability.SCORER_FIXED,
        "w_seg": Portability.SCORER_FIXED,
        "w_pose": Portability.SCORER_FIXED,
        "render_h": Portability.SCORER_FIXED,
        "render_w": Portability.SCORER_FIXED,
        # domain-fundamental: any dashcam (form transfers, coefficients re-fit).
        "directional_basis": Portability.DOMAIN,
        "lane_prior_phi1": Portability.DOMAIN,
        "warp_priors": Portability.DOMAIN,
        "activation": Portability.DOMAIN,
        "chroma": Portability.DOMAIN,
        "palette_anchor": Portability.DOMAIN,
        # instance-conditioned: this clip's numbers (re-measure per clip).
        "mod_dim": Portability.INSTANCE,
        "hidden_dim": Portability.INSTANCE,
        "tau_softplus_start_epoch": Portability.INSTANCE,
        "l7_start_epoch": Portability.INSTANCE,
        "muon_start_epoch": Portability.INSTANCE,
        "epochs": Portability.INSTANCE,
        "structured_init": Portability.INSTANCE,
    }


# --------------------------------------------------------------------------
# The WitnessConfig + the actuator
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WitnessConfig:
    """A launch-ready witness config derived from a clip's GT cache.

    Load-bearing DERIVED knobs are explicit named fields; the fixed proven-base
    knobs (render size, activation, basis, stage-transition, etc.) live in
    ``proven_base`` (an ordered flag->value mapping). ``provenance`` maps each
    derived field to its :class:`ProvenancedValue`. ``portability`` is the
    per-knob corpus-generalization classification.
    """

    # clip identity
    gt_cache: str
    num_pairs: int
    overfit: bool

    # derived knobs (the dogfood revisions)
    mod_dim: int
    hidden_dim: int
    muon_lr: float
    verdict_pairs: int
    epochs: int
    tau_softplus_start_epoch: int
    l7_start_epoch: int
    muon_start_epoch: int

    # lever / capacity decisions
    surgical_levers_enabled: bool
    dm1_enabled: bool

    # (F1) all-levers opt-in: when True, ``to_trainer_flags`` renders the deep-math-OPTIMAL all-levers
    # from-scratch argv (--render-aa none + analytic coverage-integrated lane-render-band [Wave D AA
    # correction] + persistence/topology loss + island-birth amplification + annealed hosc 1->4 +
    # l7 DEMOTED + verdict-pairs 0 + adam-beta2),
    # per .omx/research/capstone_witness_launch_config_deepmath_optimal_20260702.md (#205). Default
    # False => the attribution-clean proven_base baseline stays available byte-identically.
    all_levers: bool = False
    # (F5) #205 P3 SEALED capstone config: when True, ``to_trainer_flags`` renders the exact
    # canonical §7 argv (n205_phase3_recursive_adversarial_review_verdict_20260702.md §7) = the
    # deep-math-OPTIMAL all-levers base + the 4 SEALED deltas (mod-dim 32 / adam-beta2 0.999 /
    # w-pose 1.0 + pose-carrier table), emitted in the hand-authored §7 token ORDER so the launcher
    # reproduces the SEALED launch.sh BYTE-IDENTICALLY (modulo --out-dir). Built by
    # ``derive_sealed_205_config``. Default False => the all-levers / proven_base paths are unchanged.
    sealed_205: bool = False
    # (F6) #205 pose-carrier frame0 SOURCE (Track B store-nothing-but-xi, 18927a1ae). "real_keyframe"
    # (default) = warp a STORED real keyframe (warp_real_luma table; COUNTS the keyframe luma).
    # "generated" = STORE-NOTHING: warp the witness's OWN frame0 INR render by the twist (stores ONLY
    # xi/H, ~0 marginal bytes; the render is FREE, rule-118). Emitted as ``--pose-carrier-source`` ONLY
    # when != "real_keyframe" so ``sealed_205`` stays BYTE-IDENTICAL; ``derive_store_nothing_205_config``
    # sets it to "generated". A/B-able against sealed_205 at #205 (measures the store-nothing d_pose
    # through the real byte-closed decode vs the table carrier). Default "real_keyframe" => unchanged.
    pose_carrier_source: str = "real_keyframe"
    all_levers_base: dict = field(default_factory=dict)
    adam_beta2: float = 0.999  # #222; 0.999 == MLX default (bit-identical); all-levers sets 0.9999999.

    # rich design fields
    lever_priors: dict = field(default_factory=dict)
    warp_priors: dict = field(default_factory=dict)
    portability: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    proven_base: dict = field(default_factory=dict)

    tag: str = ADVISORY_TAG

    def to_trainer_flags(self, out_dir: str) -> list[tuple[str, object]]:
        """Ordered list of (flag, value) — value ``None`` means a bare boolean flag.

        Default renders the ATTRIBUTION-CLEAN command (surgical levers + DM1 OFF, byte-identical to the
        pre-F1 output). When ``self.all_levers`` is True it renders the deep-math-OPTIMAL all-levers
        from-scratch argv (per the #205 config artifact): the derived fields (``mod_dim`` 19, ``l7``
        DEMOTED to epochs, ``verdict_pairs`` 0) are already set by ``derive_config``; the extra render/
        loss levers + the annealed-hosc / tau-anneal-shape / lr / adam-beta2 flags are added here.

        Every flag here is validated against the trainer's real argparse by :func:`validate_flags`
        (the CLI dogfood); none is invented.
        """
        if self.sealed_205:
            return self._sealed_205_flags(out_dir)
        al = bool(self.all_levers)
        alb = self.all_levers_base if al else {}
        pb = self.proven_base
        flags: list[tuple[str, object]] = [
            ("--out-dir", out_dir),
            ("--gt-cache", self.gt_cache),
            ("--num-pairs", self.num_pairs),
            ("--mlx-device", pb["mlx_device"]),
            ("--seed", pb["seed"]),
            ("--async-verdict", None),
            ("--epochs", self.epochs),
            ("--eval-every", pb["eval_every"]),
            ("--verdict-pairs", self.verdict_pairs),
            ("--curriculum", None),
            ("--tau-softplus-start-epoch", self.tau_softplus_start_epoch),
            ("--tau-softplus-tau", pb["tau_softplus_tau"]),
            ("--l7-start-epoch", self.l7_start_epoch),
            ("--muon-start-epoch", self.muon_start_epoch),
            ("--muon-lr", self.muon_lr),
            ("--muon-momentum", pb["muon_momentum"]),
            ("--muon-ns-steps", pb["muon_ns_steps"]),
            ("--stage-transition-rewarmup-epochs", pb["st_rewarmup_epochs"]),
            ("--stage-transition-rewarmup-floor", pb["st_rewarmup_floor"]),
            ("--stage-transition-rewarmup-shape", pb["st_rewarmup_shape"]),
            ("--stage-transition-reset-moments", None),
            ("--w-seg", pb["w_seg"]),
            ("--w-pose", pb["w_pose"]),
            ("--score-domain-loss", None),
            ("--mod-dim", self.mod_dim),
            ("--hidden-dim", self.hidden_dim),
            ("--n-hidden", pb["n_hidden"]),
            ("--activation", pb["activation"]),
        ]
        # hosc activation: baseline = fixed --hosc-beta (proven_base); all-levers = annealed 1.0->4.0
        # (the CLAUDE.md fixed-beta=4 divergence fix: start 1.0, anneal to 4.0 as the SDF pins).
        if al:
            flags += [
                ("--hosc-beta", alb["hosc_beta"]),
                ("--hosc-beta-end", alb["hosc_beta_end"]),
                ("--hosc-beta-anneal", alb["hosc_beta_anneal"]),
            ]
        else:
            flags.append(("--hosc-beta", pb["hosc_beta"]))
        flags += [
            ("--hosc-omega", pb["hosc_omega"]),
            ("--siren-init", None),
            ("--softmax-temp-start", pb["softmax_temp_start"]),
            ("--softmax-temp-end", pb["softmax_temp_end"]),
        ]
        if al:
            flags.append(("--tau-anneal-shape", alb["tau_anneal_shape"]))
        flags += [
            ("--self-orient", None),
            ("--n-dir-freqs", pb["n_dir_freqs"]),
            ("--freq-across", pb["freq_across"]),
            ("--freq-along", pb["freq_along"]),
            ("--reorient-every", pb["reorient_every"]),
            ("--max-bank-freq", pb["max_bank_freq"]),
            ("--chroma", None),
            ("--palette-anchor", None),
            ("--eikonal-weight", pb["eikonal_weight"]),
            ("--length-weight", pb["length_weight"]),
            ("--render-h", pb["render_h"]),
            ("--render-w", pb["render_w"]),
        ]
        # all-levers render/loss levers: AA-SDF supersample render + analytic lane-render-band + persistence/
        # topology loss + island-birth amplification (all default-OFF in the trainer => byte-identical
        # when NOT emitted; here they are the ENGAGE values from the deep-math config artifact).
        if al:
            flags += [
                # #224 Wave D AA CORRECTION (aa_feasibility_reconciliation_20260702.md): the
                # contest-feasible OPTIMAL AA is --render-aa NONE + the analytic coverage-integrated
                # --lane-render-band (O(1)/pixel, in the 30-min decode budget, MEASURED to HELP).
                # Brute --render-aa supersample is DISQUALIFIED on TWO independent grounds: (1) it
                # HURTS the witness -49% (0.00086 is a REAL-FRAME ceiling, not witness-realized), and
                # (2) fp64 decode 41min > 30min budget AND neither shipped inflate even applies ss
                # (train/decode observation MISMATCH). The supersample code path stays BUILT +
                # fail-closeable in the trainer but is OUT of the launch config (--aa-supersample /
                # --aa-self-orient-fine-mode are NOT emitted). --render-aa ipe (O(1), decode-safe)
                # is the documented alt if a full-partition AA is ever wanted; never supersample.
                ("--render-aa", alb["render_aa"]),
                ("--lane-render-band", None),
                ("--lane-band-start-epoch", alb["lane_band_start_epoch"]),
                ("--lane-band-uncertainty-source", alb["lane_band_uncertainty_source"]),
                ("--lane-band-tau", alb["lane_band_tau"]),
                ("--lane-band-eps", alb["lane_band_eps"]),
                ("--lane-band-softness", alb["lane_band_softness"]),
                ("--lane-band-dash-forward-max-m", alb["lane_band_dash_forward_max_m"]),
                ("--lane-band-weight", alb["lane_band_weight"]),
                ("--persistence-loss-weight", alb["persistence_loss_weight"]),
                ("--persistence-recall-weight", alb["persistence_recall_weight"]),
                ("--cldice-iters", alb["cldice_iters"]),
                ("--persistence-warmup-epochs", alb["persistence_warmup_epochs"]),
                ("--persistence-classes", alb["persistence_classes"]),
                ("--amplify-weight", alb["amplify_weight"]),
                ("--amplify-form", alb["amplify_form"]),
                ("--amplify-margin-target", alb["amplify_margin_target"]),
                ("--amplify-persist", alb["amplify_persist"]),
                ("--island-dilate-px", alb["island_dilate_px"]),
            ]
        flags += [
            ("--accum-pairs", pb["accum_pairs"]),
            ("--grad-clip", pb["grad_clip"]),
            ("--ema-decay", pb["ema_decay"]),
        ]
        # all-levers explicit lr schedule + adam-beta2 (#222). The lr trio == trainer defaults (the
        # artifact emits them explicitly); adam-beta2 is the small-n optimum (launcher-only F4 superset).
        if al:
            flags += [
                ("--lr", alb["lr"]),
                ("--lr-end", alb["lr_end"]),
                ("--weight-decay", alb["weight_decay"]),
                ("--adam-beta2", self.adam_beta2),
            ]
        flags += [
            ("--structured-init", None),
            ("--structured-init-include-lane", None),
            ("--lane-prior-phi1", None),
            ("--lane-prior-phi1-mode", pb["lane_prior_phi1_mode"]),
            ("--lane-prior-phi1-dash-gate", None),
            ("--ckpt-every", pb["ckpt_every"]),
            ("--stage-checkpoints", None),
        ]
        return flags

    def _sealed_205_flags(self, out_dir: str) -> list[tuple[str, object]]:
        """The #205 Phase-3 SEALED launch argv, in the EXACT canonical §7 token order
        (``.omx/research/n205_phase3_recursive_adversarial_review_verdict_20260702.md`` §7).

        Composition = the deep-math-OPTIMAL all-levers base (every value generator-derived via
        ``derive_config(all_levers=True)``) + the 4 SEALED deltas adjudicated in §2 of the P3
        verdict:

          * Q4  ``--mod-dim 32``  (proven arm reached d_seg 0.003698; covers composite m~13 with
                headroom; d_seg is the BINDING term and 19's neutrality is UNMEASURED) — SEALED over
                the all-levers Whitney-floor 19.
          * Q5  ``--adam-beta2 0.999``  (== the MLX default => byte-identical, no bias-correction
                confound on the first attribution row) — SEALED over the all-levers 0.9999999 anchor.
          * Q2  ``--w-pose 1.0 --pose-carrier --pose-carrier-residual-mode table``  (pose SLOT
                shippable-first; a w_pose=0 row does NOT move the pointer — means/ends firewall;
                carrier wired + durability-proven) — SEALED over the proven_base ``w_pose=0``.

        Every non-delta value is read from the SAME source the all-levers config derived it from
        (``self`` field / ``proven_base`` / ``all_levers_base``), so this method canonicalizes the
        sealed config WITHOUT re-typing the generators' values. Emitted in the hand-authored §7 order
        (the two local re-orderings vs all-levers — ``--async-verdict`` after ``--verdict-pairs``, and
        the structured-init/lane-prior block BEFORE the accum/lr/adam block — are reproduced here) so
        the launcher's ``launch.sh`` reproduces the SEALED argv BYTE-IDENTICALLY (modulo ``--out-dir``).
        ``--lr / --lr-end / --weight-decay`` are the §7 string literals (``1e-3 / 1e-4 / 1e-4`` ==
        the trainer float defaults) so the emitted tokens match the sealed argv byte-for-byte.

        means != ends: this renders a MEANS (a launch config). Only a byte-closed n600 exact row
        < 0.19110 from ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer.
        """
        pb = self.proven_base
        alb = self.all_levers_base
        d = _sealed_205_deltas()
        # (F6) STORE-NOTHING variant: emit --pose-carrier-source generated ONLY when != real_keyframe
        # (default real_keyframe => not emitted => sealed_205 byte-identical). Placed immediately after
        # --pose-carrier-residual-mode (the pose SLOT block) so the argv reads as one coherent group.
        _pc_source_flags: list[tuple[str, object]] = (
            [("--pose-carrier-source", self.pose_carrier_source)]
            if str(self.pose_carrier_source) != "real_keyframe" else [])
        return [
            ("--out-dir", out_dir),
            ("--gt-cache", self.gt_cache),
            ("--num-pairs", self.num_pairs),
            ("--mlx-device", pb["mlx_device"]),
            ("--seed", pb["seed"]),
            ("--epochs", self.epochs),
            ("--eval-every", pb["eval_every"]),
            ("--verdict-pairs", self.verdict_pairs),
            ("--async-verdict", None),
            ("--curriculum", None),
            ("--tau-softplus-start-epoch", self.tau_softplus_start_epoch),
            ("--tau-softplus-tau", pb["tau_softplus_tau"]),
            ("--l7-start-epoch", self.l7_start_epoch),
            ("--muon-start-epoch", self.muon_start_epoch),
            ("--muon-lr", self.muon_lr),
            ("--muon-momentum", pb["muon_momentum"]),
            ("--muon-ns-steps", pb["muon_ns_steps"]),
            ("--stage-transition-rewarmup-epochs", pb["st_rewarmup_epochs"]),
            ("--stage-transition-rewarmup-floor", pb["st_rewarmup_floor"]),
            ("--stage-transition-rewarmup-shape", pb["st_rewarmup_shape"]),
            ("--stage-transition-reset-moments", None),
            ("--w-seg", pb["w_seg"]),
            ("--w-pose", d["w_pose"]),                       # SEALED delta (Q2) over proven_base 0
            ("--score-domain-loss", None),
            ("--pose-carrier", None),                        # SEALED add (Q2)
            ("--pose-carrier-residual-mode", d["pose_carrier_residual_mode"]),  # SEALED add (Q2)
            *_pc_source_flags,                               # (F6) store-nothing variant only (default OFF)
            ("--mod-dim", self.mod_dim),                     # SEALED delta (Q4): 32
            ("--hidden-dim", self.hidden_dim),
            ("--n-hidden", pb["n_hidden"]),
            ("--activation", pb["activation"]),
            ("--hosc-beta", alb["hosc_beta"]),
            ("--hosc-beta-end", alb["hosc_beta_end"]),
            ("--hosc-beta-anneal", alb["hosc_beta_anneal"]),
            ("--hosc-omega", pb["hosc_omega"]),
            ("--siren-init", None),
            ("--softmax-temp-start", pb["softmax_temp_start"]),
            ("--softmax-temp-end", pb["softmax_temp_end"]),
            ("--tau-anneal-shape", alb["tau_anneal_shape"]),
            ("--self-orient", None),
            ("--n-dir-freqs", pb["n_dir_freqs"]),
            ("--freq-across", pb["freq_across"]),
            ("--freq-along", pb["freq_along"]),
            ("--reorient-every", pb["reorient_every"]),
            ("--max-bank-freq", pb["max_bank_freq"]),
            ("--chroma", None),
            ("--palette-anchor", None),
            ("--eikonal-weight", pb["eikonal_weight"]),
            ("--length-weight", pb["length_weight"]),
            ("--render-h", pb["render_h"]),
            ("--render-w", pb["render_w"]),
            ("--render-aa", alb["render_aa"]),
            ("--lane-render-band", None),
            ("--lane-band-start-epoch", alb["lane_band_start_epoch"]),
            ("--lane-band-uncertainty-source", alb["lane_band_uncertainty_source"]),
            ("--lane-band-tau", alb["lane_band_tau"]),
            ("--lane-band-eps", alb["lane_band_eps"]),
            ("--lane-band-softness", alb["lane_band_softness"]),
            ("--lane-band-dash-forward-max-m", alb["lane_band_dash_forward_max_m"]),
            ("--lane-band-weight", alb["lane_band_weight"]),
            ("--persistence-loss-weight", alb["persistence_loss_weight"]),
            ("--persistence-recall-weight", alb["persistence_recall_weight"]),
            ("--cldice-iters", alb["cldice_iters"]),
            ("--persistence-warmup-epochs", alb["persistence_warmup_epochs"]),
            ("--persistence-classes", alb["persistence_classes"]),
            ("--amplify-weight", alb["amplify_weight"]),
            ("--amplify-form", alb["amplify_form"]),
            ("--amplify-margin-target", alb["amplify_margin_target"]),
            ("--amplify-persist", alb["amplify_persist"]),
            ("--island-dilate-px", alb["island_dilate_px"]),
            # §7 tail ORDER: the structured-init / lane-prior-phi1 block precedes the accum / lr /
            # adam block (the reverse of the all-levers renderer) — reproduced here for byte-identity.
            ("--structured-init", None),
            ("--structured-init-include-lane", None),
            ("--lane-prior-phi1", None),
            ("--lane-prior-phi1-mode", pb["lane_prior_phi1_mode"]),
            ("--lane-prior-phi1-dash-gate", None),
            ("--accum-pairs", pb["accum_pairs"]),
            ("--grad-clip", pb["grad_clip"]),
            ("--ema-decay", pb["ema_decay"]),
            ("--lr", _SEALED_205_LR),          # "1e-3" literal == trainer default (§7 byte-identity)
            ("--lr-end", _SEALED_205_LR_END),  # "1e-4"
            ("--weight-decay", _SEALED_205_WD),  # "1e-4"
            ("--adam-beta2", self.adam_beta2),  # SEALED delta (Q5): 0.999
            ("--ckpt-every", pb["ckpt_every"]),
            ("--stage-checkpoints", None),
        ]

    def to_command(self, out_dir: str, *, perf_env: bool = True) -> str:
        """Render the GO-ready launch command string (attribution-clean)."""
        parts = []
        for flag, val in self.to_trainer_flags(out_dir):
            parts.append(flag if val is None else f"{flag} {val}")
        prefix = "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \\\n  " if perf_env else "  "
        body = " \\\n  ".join(
            [".venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py"]
            + parts
        )
        return prefix + body


# the fixed proven-base knobs (recalled verbatim from the 0.003698 run_muon.log).
def _proven_base() -> dict:
    return {
        "mlx_device": "gpu",
        "seed": 0,
        "eval_every": 25,
        "tau_softplus_tau": 0.3,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "st_rewarmup_epochs": 8,
        "st_rewarmup_floor": 0.1,
        "st_rewarmup_shape": "linear",
        "w_seg": 100,
        "w_pose": 0,
        "n_hidden": 4,
        "activation": "hosc",
        "hosc_beta": 4.0,
        "hosc_omega": 1.0,
        "softmax_temp_start": 1.0,
        "softmax_temp_end": 0.05,
        "n_dir_freqs": 2,
        "freq_across": 32,
        "freq_along": 4,
        "reorient_every": 50,
        "max_bank_freq": 64,
        "eikonal_weight": 0.01,
        "length_weight": 0.001,
        "render_h": 384,
        "render_w": 512,
        "accum_pairs": 8,
        "grad_clip": 1.0,
        "ema_decay": 0.997,
        "lane_prior_phi1_mode": "replace",
        "ckpt_every": 25,
    }


# the ENGAGE values for the deep-math-OPTIMAL all-levers from-scratch config (#205 artifact
# .omx/research/capstone_witness_launch_config_deepmath_optimal_20260702.md). Every value verified to
# exist + be wired in the levelset trainer argparse. lane_band_start / persistence_warmup are tied to
# the curriculum ``tau_start`` (the artifact = 300 = tau @ epochs=1000; both generalize with the schedule).
def _all_levers_base(tau_start: int) -> dict:
    return {
        # annealed hosc (fixed-beta=4 divergence fix: start 1.0 -> anneal to 4.0 as the SDF pins).
        "hosc_beta": 1.0,
        "hosc_beta_end": 4.0,
        "hosc_beta_anneal": "linear",
        "tau_anneal_shape": "cosine",
        # #224 Wave D AA CORRECTION (aa_feasibility_reconciliation_20260702.md, supersedes Wave C
        # FIX-2): --render-aa NONE. The contest-feasible OPTIMAL AA is the analytic coverage-integrated
        # --lane-render-band below (O(1)/pixel, decode-in-budget, MEASURED to HELP). Brute supersample
        # is DISQUALIFIED: it HURTS the witness -49% (the 0.00086 floor is a REAL-FRAME ceiling, not
        # the witness-realized number) AND its fp64 decode is 41min > the 30min budget AND neither
        # shipped inflate applies ss (train/decode observation mismatch). ss code path stays BUILT +
        # fail-closeable in the trainer but OUT of the launch config; --render-aa ipe is the O(1)
        # decode-safe alt if a full-partition AA is ever wanted (never supersample).
        "render_aa": "none",
        # analytic lane-render-band (class-1 render-time authority; witness-uncertainty FP gate).
        "lane_band_start_epoch": int(tau_start),
        "lane_band_uncertainty_source": "witness",
        "lane_band_tau": 0.85,
        "lane_band_eps": 0.35,
        "lane_band_softness": 1.0,
        "lane_band_dash_forward_max_m": 55.0,
        "lane_band_weight": 1.0,
        # persistence / topology loss (soft-clDice + persistence-recall on the shared seg forward).
        "persistence_loss_weight": 1.0,
        "persistence_recall_weight": 1.0,
        "cldice_iters": 5,
        "persistence_warmup_epochs": int(tau_start),
        "persistence_classes": "auto",
        # island-birth amplification (rides the shared LEVER-4 _signed margin; AMPLIFY_ONLY WIRED path).
        "amplify_weight": 1.0,
        "amplify_form": "hinge",
        "amplify_margin_target": 1.0,
        "amplify_persist": "inverse_thickness",
        "island_dilate_px": 1,
        # explicit lr schedule (== trainer defaults; the artifact emits them explicitly).
        "lr": 1e-3,
        "lr_end": 1e-4,
        "weight_decay": 1e-4,
    }


# #222 small-n (n = P/accum_pairs ~ 75) AdamW beta2 optimum: 1-beta2 <~ (1-beta1^5)/n^3.5 ~ 1.12e-7
# (beta1=0.9) => beta2* ~ 0.99999988. 0.9999999 (1-beta2=1e-7) clears the threshold. Default path
# stays 0.999 (== MLX default => bit-identical) unless all_levers.
_ALL_LEVERS_ADAM_BETA2 = 0.9999999


# --------------------------------------------------------------------------
# #205 Phase-3 SEALED capstone config (the triality DEFINITION of the launch-ready argv)
# --------------------------------------------------------------------------
# The lr trio is rendered as the §7 STRING literals so the emitted tokens equal the sealed argv
# byte-for-byte (``1e-3``/``1e-4`` == the trainer's float defaults 1e-3/1e-4; float would render
# "0.001"/"0.0001" and break the byte-identity gate).
_SEALED_205_LR = "1e-3"
_SEALED_205_LR_END = "1e-4"
_SEALED_205_WD = "1e-4"


def _sealed_205_deltas() -> dict:
    """The 4 SEALED deltas over the deep-math-OPTIMAL all-levers base, per the #205 Phase-3
    recursive-adversarial-review VERDICT §2 (SEAL). Single source for the non-generator sealed
    values consumed by :meth:`WitnessConfig._sealed_205_flags` + :func:`derive_sealed_205_config`.

      * ``mod_dim`` 32   — Q4 (proven arm; d_seg BINDING; 19's neutrality UNMEASURED).
      * ``adam_beta2`` 0.999 — Q5 (== MLX default; byte-identical; no bias-correction confound).
      * ``w_pose`` 1.0 + ``pose_carrier`` + ``pose_carrier_residual_mode`` "table" — Q2 (pose SLOT
        shippable-first; w_pose=0 does NOT move the pointer).
    """
    return {
        "mod_dim": 32,
        "adam_beta2": 0.999,
        "w_pose": 1.0,
        "pose_carrier": True,
        "pose_carrier_residual_mode": "table",
    }


def derive_sealed_205_config(
    gt_cache_path: str | Path,
    *,
    num_pairs: int,
    epochs: int = 1000,
    code_matrix: np.ndarray | None = None,
    byte_close_result: dict | None = None,
) -> WitnessConfig:
    """Canonical **#205 Phase-3 SEALED capstone config** — the triality DEFINITION of the
    operator's launch-ready argv (the config the SEAL verdict certified).

    Built from the deep-math-OPTIMAL all-levers base (``derive_config(all_levers=True)`` — every
    value generator-derived) + the 4 SEALED deltas (:func:`_sealed_205_deltas`, adjudicated in the
    P3 verdict §2). The returned config's ``to_trainer_flags`` renders the EXACT §7 argv order, so
    launching it via ``tools/launch_witness_run.py --config sealed_205`` writes a ``launch.sh`` whose
    command is BYTE-IDENTICAL (modulo ``--out-dir``) to the hand-authored §7 oracle.

    Pure CPU; does NOT load the (5 GB) GT cache. ``overfit`` is not a parameter — the sealed config
    fixes mod-dim to the proven 32 (NOT the aggressive Whitney floor) per Q4.

    means != ends: returns a MEANS (a launch config). Only a byte-closed n600 exact row < 0.19110
    moves the pointer.
    """
    base = derive_config(
        gt_cache_path, num_pairs=num_pairs, overfit=True, epochs=epochs,
        code_matrix=code_matrix, byte_close_result=byte_close_result, all_levers=True)
    d = _sealed_205_deltas()
    prov = dict(base.provenance)
    prov["mod_dim"] = ProvenancedValue(
        int(d["mod_dim"]), SRC_RECALLED,
        "#205 P3 §2 Q4 SEALED: launch at mod-dim 32 (proven arm reached measured d_seg 0.003698; "
        "covers composite m~13 with headroom; d_seg is the BINDING term and 19's d_seg-neutrality is "
        "UNMEASURED; rate slack 0.055<0.081). SEALED delta over the all-levers Whitney-floor 19.",
        Portability.INSTANCE)
    prov["adam_beta2"] = ProvenancedValue(
        float(d["adam_beta2"]), SRC_RECALLED,
        "#205 P3 §2 Q5 SEALED: launch at adam-beta2 0.999 (== MLX default => byte-identical, no "
        "bias-correction confound on the first attribution row). SEALED delta over the all-levers "
        "0.9999999 mis-anchor.", Portability.SCORER_FIXED)
    prov["w_pose"] = ProvenancedValue(
        float(d["w_pose"]), SRC_RECALLED,
        "#205 P3 §2 Q2 SEALED: w-pose 1.0 + --pose-carrier (residual-mode table) shippable-first — a "
        "w_pose=0 row does NOT move the pointer (means/ends firewall); carrier wired + "
        "durability-proven. SEALED delta over the proven_base w_pose=0.", Portability.SCORER_FIXED)
    prov["pose_carrier_residual_mode"] = ProvenancedValue(
        d["pose_carrier_residual_mode"], SRC_RECALLED,
        "#205 P3 §2 Q2 SEALED: --pose-carrier-residual-mode table (per-pair (P,6) byte-minimal; "
        "isolates the code manifold, cos 5.9e-5 => seg⊥pose additive-S attribution safeguard).",
        Portability.SCORER_FIXED)
    return replace(
        base,
        sealed_205=True,
        mod_dim=int(d["mod_dim"]),
        adam_beta2=float(d["adam_beta2"]),
        provenance=prov,
    )


def derive_store_nothing_205_config(
    gt_cache_path: str | Path,
    *,
    num_pairs: int,
    epochs: int = 1000,
    code_matrix: np.ndarray | None = None,
    byte_close_result: dict | None = None,
) -> WitnessConfig:
    """The **#205 STORE-NOTHING pose-carrier variant** of the SEALED capstone config (Track B
    store-nothing-but-xi, ``18927a1ae`` / ``keyframe_rate_minimization_builds_20260702``).

    IDENTICAL to :func:`derive_sealed_205_config` in every knob EXCEPT the pose-carrier frame0 SOURCE:
    the sealed table carrier warps a STORED real keyframe (COUNTS the keyframe luma in archive.zip);
    this variant sets ``pose_carrier_source="generated"`` -> STORE-NOTHING: frame0 = warp(the witness's
    OWN frame0 INR render, xi). It stores ONLY xi/H (~0 marginal bytes; the render is FREE per rule-118)
    -> the keyframe payload the byte-close measured (697941 B ds4 / much more native) collapses to ~1 KB.

    This is the A/B arm the #205 run runs against ``sealed_205``: same seg/curriculum/optimizer, only
    the pose STORE gauge differs, so the measured d_pose + rate delta is a clean attribution. Its
    ``to_trainer_flags`` renders the sealed §7 argv + a single trailing ``--pose-carrier-source
    generated`` (sealed_205 stays byte-identical because it never emits that flag). The trained rank-6
    dxi residual (residual-mode table, w_pose>0) closes the store-nothing offset toward the target;
    whether it reaches the table carrier's d_pose is the OPEN #205 question this arm MEASURES.

    means != ends: returns a MEANS (a launch config). Only a byte-closed n600 exact row < 0.19110 from
    ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer.
    """
    base = derive_sealed_205_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs,
        code_matrix=code_matrix, byte_close_result=byte_close_result)
    prov = dict(base.provenance)
    prov["pose_carrier_source"] = ProvenancedValue(
        "generated", SRC_RECALLED,
        "#205 STORE-NOTHING variant (Track B 18927a1ae): --pose-carrier-source generated -> frame0 = "
        "warp(the witness's OWN INR render, xi); stores ONLY xi/H (~0 marginal bytes) vs the sealed "
        "table carrier's stored keyframe luma. MEASURED n6/t1 byte-close BIT-EXACT: section 697941B(ds4 "
        "table)->1049B(store_nothing). A/B arm over sealed_205 (same seg/curriculum; only the pose STORE "
        "gauge differs).", Portability.SCORER_FIXED)
    return replace(base, pose_carrier_source="generated", provenance=prov)


def derive_config(
    gt_cache_path: str | Path,
    *,
    num_pairs: int,
    overfit: bool = True,
    code_matrix: np.ndarray | None = None,
    epochs: int = 1000,
    byte_close_result: dict | None = None,
    all_levers: bool = False,
) -> WitnessConfig:
    """Turn a clip's GT cache into a launch-ready :class:`WitnessConfig`.

    Runs the value-generators (intrinsic-dim->Whitney mod-dim, RD->hidden-dim,
    annealing->curriculum, proven muon-lr/verdict-pairs, attribution->levers,
    screw-fit->warp, held-out->portability). Pure CPU; does NOT load the (5GB)
    GT cache — the path identifies the clip; intrinsic dim is MEASURED only if a
    small ``code_matrix`` is supplied, else the recalled constant is used (flagged).

    means != ends: the returned config is a MEANS. Only a byte-closed exact n600
    row < 0.19110 moves the pointer.
    """
    # (F1) all-levers => the deep-math-OPTIMAL config: mod-dim is the aggressive Whitney FLOOR (19 for
    # m~9; rate-saving) regardless of the `overfit` flag; verdict-pairs 0 (ALL pairs, async); l7 DEMOTED
    # to epochs (the measured-defect l7 collapses to <=1 trailing epoch). Baseline (all_levers=False) is
    # unchanged (byte-identical). Per the #205 artifact.
    mod = mod_dim_generator(code_matrix, overfit=(False if all_levers else overfit))
    hid = hidden_dim_generator(byte_close_result, overfit=overfit)
    sched = curriculum_schedule(epochs)
    mlr = muon_lr_generator()
    vp = verdict_pairs_generator(num_pairs)
    levers = lever_priors(overfit=overfit)
    warps = warp_priors()
    port = portability_split()

    tau_start = int(sched["tau_softplus_start_epoch"].value)
    l7_start = int(epochs) if all_levers else int(sched["l7_start_epoch"].value)
    verdict_pairs = 0 if all_levers else int(vp.value)
    adam_beta2 = _ALL_LEVERS_ADAM_BETA2 if all_levers else 0.999
    all_levers_base = _all_levers_base(tau_start) if all_levers else {}

    provenance = {
        "mod_dim": mod,
        "hidden_dim": hid,
        "muon_lr": mlr,
        "verdict_pairs": (
            ProvenancedValue(0, SRC_RECALLED,
                             "all-levers: --verdict-pairs 0 (ALL pairs, --async-verdict; n600-allergy: "
                             "override the proven-arm 96-subset for decision-informing telemetry)",
                             Portability.SCORER_FIXED) if all_levers else vp),
        "epochs": ProvenancedValue(epochs, SRC_RECALLED, "proven total epoch budget",
                                   Portability.INSTANCE),
        **sched,
    }
    if all_levers:
        provenance["l7_start_epoch"] = ProvenancedValue(
            l7_start, SRC_DESIGN,
            "all-levers: l7 DEMOTED to epochs (measured DEFECT eq l7_linf_sharpening_defect: L-inf "
            "sharpening inside a viscosity flow; the curriculum guard forbids l7>epochs so it fires "
            "<=1 trailing epoch under Muon+EMA => negligible)", Portability.INSTANCE)
        provenance["adam_beta2"] = ProvenancedValue(
            adam_beta2, SRC_DESIGN,
            "#222 small-n (n~75) AdamW beta2 optimum ~0.9999999 (1-beta2=1e-7 < (1-0.9^5)/75^3.5"
            "~1.12e-7); DERIVED (arXiv 2603.02092)", Portability.SCORER_FIXED)

    return WitnessConfig(
        gt_cache=str(gt_cache_path),
        num_pairs=int(num_pairs),
        overfit=bool(overfit),
        mod_dim=int(mod.value),
        hidden_dim=int(hid.value),
        muon_lr=float(mlr.value),
        verdict_pairs=int(verdict_pairs),
        epochs=int(epochs),
        tau_softplus_start_epoch=tau_start,
        l7_start_epoch=l7_start,
        muon_start_epoch=int(sched["muon_start_epoch"].value),
        surgical_levers_enabled=bool(levers["surgical_levers_enabled"]),
        dm1_enabled=bool(levers["dm1_enabled"]),
        all_levers=bool(all_levers),
        all_levers_base=all_levers_base,
        adam_beta2=float(adam_beta2),
        lever_priors=levers,
        warp_priors=warps,
        portability=port,
        provenance=provenance,
        proven_base=_proven_base(),
    )
