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

# SoT for the ~17x custom-grouped-backward perf-env prefix (single source, shared by BOTH the
# WitnessConfig.to_command below and TypedWitnessConfig.to_command) — consuming the ONE constant
# makes the two launch paths drift-IMPOSSIBLE, not merely drift-tested (operator 2026-07-08
# "shouldn't have had to be caught manually"). No import cycle: typed_config imports curriculum_dsl
# (which imports witness_autoconfig only LAZILY, inside a function).
from tac.witness_dsl.typed_config import PERF_ENV_PREFIX

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
    "derive_fresh_seeded_config",
    "derive_crucible_v6_config",
]

# --------------------------------------------------------------------------
# source-class + portability vocab (the NO-FAKE provenance taxonomy)
# --------------------------------------------------------------------------
SRC_MEASURED = "measured"               # measured this session from real data
SRC_RECALLED = "recalled_proven"        # recalled from the proven 0.003698 arm
SRC_HELDOUT = "held_out_corrected"      # corrected by the held-out n400 probe
SRC_FALLBACK = "fallback_constant"      # heavy input absent -> measured constant
SRC_DESIGN = "design"                   # design-level (v2, not yet a trainer flag)
SRC_DERIVED = "derived_at_config"       # (req T ladder) DERIVED-AT-CONFIG: computed from a stated
                                        # law + provenanced inputs at config time, carries a
                                        # re-derivation trigger (NOT a bare hardcode)

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
    # (C5, SEAL review 2026-07-04) the explicit --verdict-batch value the sealed-family argv emits
    # (the R2 OOM-fix flag). Default 32 == the trainer default == the sealed §7+R2 argv => the
    # sealed_205 byte-identity gate is UNCHANGED. ``fresh_seeded`` sets 64 (review-verified: bounded
    # verdict spike max(6, 0.11*64)=7.04 GiB, still SAFE-by-projection).
    verdict_batch: int = 32
    # (C5) FRESH SEEDED run-1 config (fresh_run_config_adversarial_review_20260704.md "revised launch
    # shape"): when True, ``_sealed_205_flags`` additionally emits the seed/control flags that have no
    # sealed-argv slot (--seed-islands / --eikonal-weight-end / --film-stiefel /
    # --muon-warm-start-momentum / --muon-lr-final-frac / --closed-loop-control). Built by
    # ``derive_fresh_seeded_config``. Default False => sealed_205 / store_nothing_205 byte-identical.
    fresh_seeded: bool = False
    # (#332 DSL de-orphaning) named DSL Lever factories to COMPOSE onto whatever base this config
    # renders — the launcher selects them by name (``--dsl-lever SeedIslandEased ...``) and
    # ``to_trainer_flags`` DELEGATES to ``tac.witness_dsl.curriculum_dsl`` (the config-generating
    # SoT) to resolve each name → its overrides, merging them over the base. Default () ⇒
    # byte-identical to the pre-#332 output (every existing config/byte-identity gate unchanged).
    dsl_levers: tuple[str, ...] = ()
    all_levers_base: dict = field(default_factory=dict)
    adam_beta2: float = 0.999  # #222; 0.999 == MLX default (bit-identical); all-levers sets 0.9999999.
    # (T5 CRUCIBLE v6.2, seal-round-2 BLOCKER-1) crucible-v6 launch-candidate config: when True,
    # ``_sealed_205_flags`` additionally emits the v6 ABSOLUTE-schedule + determinism + chroma pins
    # (see ``_CRUCIBLE_V6_DELTAS``) as a trailing block, and ``derive_crucible_v6_config`` pins the
    # absolute stage anchors (tau@300 / muon@726 — NEVER family-scaled 0.726*epochs) + composes the
    # v6 DSL levers. Default False => sealed_205 / store_nothing_205 / fresh_seeded byte-identical.
    crucible_v6: bool = False
    # (#351 LawRef migration) the RESOLVED crucible_v6 delta dict — populated by
    # ``derive_crucible_v6_config`` from the CONSUMED LawRefs (τ_end / β-pin / LR-pin), whose
    # resolved values are BIT-IDENTICAL to ``_CRUCIBLE_V6_DELTAS`` (asserted at derive time). The
    # ``crucible_v6`` trailing block of ``_sealed_205_flags`` reads THIS (falling back to the module
    # ``_CRUCIBLE_V6_DELTAS`` when empty) so the config genuinely CONSUMES the compiled constants.
    # Empty for every non-crucible path => byte-identical. Provenance-only, never emitted as a flag.
    crucible_v6_deltas: dict = field(default_factory=dict)
    # (#351) constants_manifest for the CONSUMED LawRefs {delta_key -> ResolvedConstant.to_dict()};
    # the launcher writes it to constants_manifest.json beside launch.sh. Provenance-only, never a flag.
    constants_manifest: dict = field(default_factory=dict)
    # (#353 DSL-authored-config gate) the DSL-provenance manifest attesting THIS config was authored +
    # validated through the typed DSL layer (tac.witness_dsl.typed_config): schema tag + emitted flag
    # fingerprint + typed_config_hash + typed_validated attestation. The launcher's DSL-authored-config
    # gate REFUSES a launch whose config carries no manifest (hand-crafted argv) or whose fingerprint
    # disagrees with the emitted argv (a config mutated after authoring), unless --allow-non-dsl-config
    # stamps a non-DSL provenance. Populated by ``derive_crucible_v6_config`` (the migrated seam).
    # Provenance-only, never emitted as a flag => argv byte-identical for every config.
    dsl_program_manifest: dict = field(default_factory=dict)
    # (run-identity, operator 2026-07-07 "add a label ... run name and possibly a description of
    # its intended purpose; clean baseline or frontier score lowering? a/b probe?") DECLARED
    # one-line intent of THIS run — the first-class, per-run machine-answerable purpose. Set by
    # the launcher (--purpose) and stamped into the run dir's config record (the launch.sh
    # ``# tac-run-purpose:`` header) so dashboards render it with provenance "declared" instead
    # of a derived heuristic. PROVENANCE METADATA ONLY: never emitted as a trainer flag, zero
    # effect on training numerics or the emitted argv (the sealed_205 argv byte-identity gate is
    # unchanged). Default None => consumers fall back to their LABELLED derived-heuristic
    # classification (a guess is never rendered as a declaration).
    purpose: str | None = None

    # rich design fields
    lever_priors: dict = field(default_factory=dict)
    warp_priors: dict = field(default_factory=dict)
    portability: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    proven_base: dict = field(default_factory=dict)

    tag: str = ADVISORY_TAG

    def to_trainer_flags(self, out_dir: str) -> list[tuple[str, object]]:
        """Ordered (flag, value) argv. Renders the base config, then COMPOSES any selected DSL
        levers over it (#332). Byte-identical to the base when ``dsl_levers`` is empty (default),
        so every existing config/byte-identity gate is unchanged."""
        base = self._render_base_flags(out_dir)
        return self._merge_dsl_levers(base) if self.dsl_levers else base

    def _merge_dsl_levers(self, base: list[tuple[str, object]]) -> list[tuple[str, object]]:
        """Resolve each named DSL Lever factory (via the DSL SoT) and merge its overrides over
        ``base``: override an existing flag's value in place, else append. A DSL override value of
        ``True`` renders as a bare boolean ``(flag, None)`` per the argv convention. A ``False``
        override renders as the BooleanOptionalAction negation ``(--no-<flag>, None)``, in-place —
        matching the single DSL emitter ``curriculum_dsl.WitnessProgram.compile_trainer_argv`` (which
        emits ``flag.replace("--", "--no-", 1)`` for ``False``). The two "one emitter" surfaces must
        agree on ``False`` (regression-guarded in test_witness_autoconfig).

        Name resolution goes through the DSL's composability predicate
        (``lever_registry.resolve_composable_lever`` — CLASS-fix, review 2026-07-06): a name that
        requires explicit args (``Muon(start_epoch)``) or returns a composite (``DM1Minimal`` →
        ``tuple[Lever, Lever]``) raises a CLEAR typed :class:`LeverCompositionError` listing the
        composable set, never a raw ``TypeError``/``AttributeError``. A ``False`` override on a
        plain ``store_true`` flag (no ``--no-`` form) is likewise refused — mirroring
        ``curriculum_dsl.WitnessProgram.validate``'s C2 — instead of emitting an argv the trainer
        argparse would crash on."""
        from tac.witness_dsl.curriculum_dsl import real_store_true_flags
        from tac.witness_dsl.lever_registry import (
            LeverCompositionError,
            resolve_composable_lever,
        )
        merged: list[tuple[str, object]] = list(base)
        index = {f: i for i, (f, _) in enumerate(merged)}
        store_true: frozenset[str] | None = None  # lazy — only read the trainer on a False override
        for name in self.dsl_levers:
            lever = resolve_composable_lever(name)
            for flag, val in lever.overrides.items():
                if val is False:
                    # C2 mirror: a plain store_true flag has NO --no- form; negating it would
                    # crash the trainer argparse at launch. Refuse with the typed error.
                    if store_true is None:
                        store_true = real_store_true_flags(None)
                    if flag in store_true:
                        raise LeverCompositionError(
                            f"--dsl-lever {name!r} sets {flag}=False, but {flag} is store_true "
                            f"(no --no-{flag[2:]} form) — the negation would crash the trainer "
                            "argparse at launch (curriculum_dsl validate() C2)")
                    # BooleanOptionalAction negation, aligned to the DSL emitter: emit --no-<flag>
                    # in-place (matches compile_trainer_argv), NOT a silent skip (the prior latent
                    # divergence where the base flag was left untouched).
                    negated = (flag.replace("--", "--no-", 1), None)
                    if flag in index:
                        merged[index[flag]] = negated
                    else:
                        index[flag] = len(merged)
                        merged.append(negated)
                    continue
                rendered = None if val is True else val
                if flag in index:
                    merged[index[flag]] = (flag, rendered)
                else:
                    index[flag] = len(merged)
                    merged.append((flag, rendered))
        return merged

    def _render_base_flags(self, out_dir: str) -> list[tuple[str, object]]:
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
        flags: list[tuple[str, object]] = [
            ("--out-dir", out_dir),
            ("--gt-cache", self.gt_cache),
            ("--num-pairs", self.num_pairs),
            ("--mlx-device", pb["mlx_device"]),
            ("--seed", pb["seed"]),
            ("--epochs", self.epochs),
            ("--eval-every", pb["eval_every"]),
            ("--verdict-pairs", self.verdict_pairs),
            ("--async-verdict", None),
            # (R2, review 2026-07-02) EXPLICIT --verdict-batch 32 (== trainer argparse default +
            # witness_memory_preflight DEFAULT_VERDICT_BATCH): the #205 OOM fix. With --verdict-pairs 0
            # (ALL 600) the CPU-scorer verdict is a P-wide batch = a +66 GiB transient spike ON TOP of
            # the ~41 GiB resident self-orient cf_mx_cache -> the n600 run OOM-died before the first
            # ckpt. Chunking (32) bounds the spike to a ~6 GiB floor (bit-identical d_seg; ~1e-6 BLAS
            # d_pose; the verdict is PURELY OBSERVATIONAL so score-neutral by construction). Emitting it
            # EXPLICITLY (vs relying on the coupled implicit trainer+preflight default) self-documents
            # the fix in launch.sh + makes witness_memory_preflight parse the REAL value (C1 defense-in-
            # depth). Breaks the §7 byte-identity by design: the §7 argv is the config that OOM'd, and
            # this safety flag now belongs IN the launched config (P3 verdict C4). Value is the
            # ``verdict_batch`` field (default 32 == this historical literal => sealed unchanged;
            # fresh_seeded sets 64 per the 2026-07-04 SEAL review).
            ("--verdict-batch", self.verdict_batch),
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
        if self.fresh_seeded:
            # (C5) the FRESH SEEDED flags with no sealed-argv slot, appended as one trailing block
            # (argparse is order-insensitive; the sealed prefix above stays byte-stable). Values per
            # the 2026-07-04 SEAL review "revised launch shape" — every flag grep-verified against the
            # trainer argparse (never-invent-flags).
            flags += [
                ("--seed-islands", None),                 # #208 nucleation: EARLY-SEED lane+movable
                ("--eikonal-weight-end", 0.10),           # 0.05 -> 0.10 step @ tau (survival knee)
                ("--film-stiefel", None),                 # Stiefel column-orthonormal FiLM (mod-19)
                ("--muon-warm-start-momentum", None),     # Muon v <- AdamW m at the switch
                ("--muon-lr-final-frac", 0.1),            # cosine-decay Muon LR to 10% by stage end
                ("--closed-loop-control", None),          # the seed-survival safety net (review AXIS 8)
            ]
        if self.crucible_v6:
            # T5 CRUCIBLE v6.2 trailing pins (no sealed-argv slot; argparse is order-insensitive).
            # See _CRUCIBLE_V6_DELTAS for the derivations (each value cites its draft section).
            # (#351) read the RESOLVED delta dict (LawRef-compiled CONSUMED constants merged over the
            # sealed literals; values bit-identical) when present; else the module literals. So the
            # LR-pin tokens below are genuinely CONSUMED from the resolver, not the module global.
            d6 = self.crucible_v6_deltas or _CRUCIBLE_V6_DELTAS
            flags += [
                # τ leg: EXPLICIT anneal denominator + hold => descent completes at ABSOLUTE ep
                # anneal_epochs*tau_hold_frac = 600 and HOLDS at temp-end 0.31 through the fire band
                # [670,700] and the Muon freeze (726). NOTE plain cosine with --anneal-epochs 600
                # REBOUNDS after ep600 (prog_t unclamped: tau(675)=0.3363, tau(726-freeze)=0.3826
                # != the 0.31 anchor) — cosine_hold is the DERIVED materialization of the draft's
                # "anneal-epochs 600" law (tau(675) = tau(726) = 0.31 EXACTLY).
                ("--anneal-epochs", d6["anneal_epochs"]),
                ("--tau-hold-frac", d6["tau_hold_frac"]),
                # AdamW LR leg (v6.4 MAJOR-2(ii) BUILD): the LR cosine is the THIRD shared-den
                # sibling. Unlike β (LINEAR → endpoint-rephasable), a shallow den-3000 cosine cannot
                # reproduce the control's deep den-1000 descent by endpoint (curvature differs), so
                # LR gets its OWN denominator: --lr-anneal-epochs 1000 (the mod32cap control's den)
                # reproduces the control LR(ep) on [1,726] BIT-IDENTICALLY (the ν/settle/s*/fire-band
                # laws were measured at that annealed LR). --lr-hold-frac 1.0 = no hold (control's
                # Muon freeze 726 < den 1000 => LR never held pre-freeze) = bit-identical cosine.
                ("--lr-anneal-epochs", d6["lr_anneal_epochs"]),
                ("--lr-hold-frac", d6["lr_hold_frac"]),
                # F-DET (v6 fold 1): fused-R determinism; requires --mlx-device gpu (proven_base).
                ("--fused-r-kernel", None),
                # (v6.3 MAJOR-3) re-anchor leg of the event-triggered design: shift the TAU-RELATIVE
                # wall-clock levers (persistence-warmup completion / seed-anneal withdrawal / analytic-
                # band engage) to the FIRED tau boundary. Makes §1.1's AnalyticLaneRenderBand
                # boundary_relative=True ACTUAL (was silently false); requires --curriculum-event-
                # triggered (DSL). hosc-β + chroma are NOT re-anchored (trainer L2049-2072/L7730).
                ("--curriculum-reanchor-levers", None),
                # (v6.3 MINOR-4) dwell-law min-stage pinned to the draft's shipped 250 (default 150).
                ("--curriculum-min-stage-epochs", d6["curriculum_min_stage_epochs"]),
                # ChromaBoundarySharpen (v6 §1.1: weight=0.1, margin_band=1.0, start=300 absolute —
                # chroma re-anchor is a run-2 trainer build item, see _CRUCIBLE_V6_DELTAS).
                ("--seg-chroma-boundary-weight", d6["seg_chroma_boundary_weight"]),
                ("--seg-chroma-boundary-margin-band", d6["seg_chroma_boundary_margin_band"]),
                ("--seg-chroma-boundary-start-epoch", d6["seg_chroma_boundary_start_epoch"]),
            ]
        return flags

    def to_command(self, out_dir: str, *, perf_env: bool = True) -> str:
        """Render the GO-ready launch command string (attribution-clean)."""
        parts = []
        for flag, val in self.to_trainer_flags(out_dir):
            parts.append(flag if val is None else f"{flag} {val}")
        prefix = PERF_ENV_PREFIX if perf_env else "  "
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


# The FRESH SEEDED run-1 deltas over the #205 sealed argv, per the pre-launch SEAL review
# ``.omx/research/fresh_run_config_adversarial_review_20260704.md`` ("The revised launch shape").
# Single source consumed by :func:`derive_fresh_seeded_config` (reuse-not-retype: everything NOT
# listed here is read from the sealed config exactly as ``derive_sealed_205_config`` built it).
_FRESH_SEEDED_DELTAS: dict = {
    # proven_base overrides (flags the sealed argv already emits from proven_base):
    "lane_prior_phi1_mode": "paint",     # #291 paint-then-SDF nucleation fix (replace = MEASURED NO-OP)
    "eikonal_weight": 0.05,              # base of the 0.05 -> 0.10 survival step
    "softmax_temp_end": 1.0,             # constant render-tau 1.0 (with geometric shape: inert-exact, L2)
    "st_rewarmup_epochs": 20,            # rewarmup 20-cosine at stage transitions
    "st_rewarmup_shape": "cosine",
    # all_levers_base overrides (flags the sealed argv emits from all_levers_base):
    "tau_anneal_shape": "geometric",     # with start==end==1.0 => constant tau EXACTLY (review L2)
    "lane_band_start_epoch": 350,        # 50-ep deconflict gap after tau@300
    "hosc_beta_end": 5.134,              # M4: beta(ep726 muon-freeze) = 4.00 exactly
    # WitnessConfig field overrides:
    "mod_dim": 19,                       # Whitney floor (mod-19 + film-stiefel well-posed, review L4)
    "l7_start_epoch": 1001,              # L1: TRUE "never" (1000 would run l7_softplus on the final ep)
    "verdict_batch": 64,
    # (#314 / DAG DRIFT-D2 fix, 2026-07-06) the fresh_seeded lineage's INTENDED pose-carrier frame0
    # source is the STORE-NOTHING generated render (Track B 18927a1ae; the #205 argv ledger KEEP row
    # HAD --pose-carrier-source generated). Because derive_fresh_seeded_config inherits from
    # derive_sealed_205_config (whose field default is "real_keyframe" -> flag NOT emitted), v1->v5
    # silently reverted to the warp-real-luma table path — a RATE-ACCOUNTING drift: any byte-close /
    # S-projection from those runs must charge the COUNTED uint8-keyframe rate (697,941 B ds4), NOT
    # the ~1 KB store-nothing rate. Carrying the delta EXPLICITLY here makes the intent structural:
    # fresh_seeded now always emits --pose-carrier-source generated (regression-pinned in
    # test_witness_autoconfig.py::test_fresh_seeded_carries_store_nothing_pose_carrier_source).
    "pose_carrier_source": "generated",
}


def derive_fresh_seeded_config(
    gt_cache_path: str | Path,
    *,
    num_pairs: int,
    epochs: int = 1000,
    code_matrix: np.ndarray | None = None,
    byte_close_result: dict | None = None,
) -> WitnessConfig:
    """The **FRESH SEEDED run-1 config** — the REVISED launch shape from the pre-launch SEAL
    adversarial review (``.omx/research/fresh_run_config_adversarial_review_20260704.md``, commit
    4bf533cab): the #205 sealed config (:func:`derive_sealed_205_config` — reused, not retyped) +
    the lane-NUCLEATION seed fix + the survival/control levers, EXACTLY the review's deltas
    (:data:`_FRESH_SEEDED_DELTAS` + the six no-sealed-slot flags ``fresh_seeded=True`` appends):

      paint / --seed-islands / eikonal 0.05->0.10 / geometric+constant tau=1.0 / mod-dim 19 /
      --film-stiefel / --muon-warm-start-momentum + --muon-lr-final-frac 0.1 / band 350 /
      rewarmup 20-cosine / --closed-loop-control / --l7-start-epoch 1001 / --hosc-beta-end 5.134 /
      --verdict-batch 64 / --pose-carrier-source generated (#314 drift fix — see below).

    **#314 pose-carrier-source drift fix (2026-07-06, DAG DRIFT-D2):** the fresh_seeded lineage
    INTENDED the store-nothing pose carrier (``--pose-carrier-source generated``, the #205 ledger
    KEEP row), but v1->v5 inherited the sealed default ``real_keyframe`` because the flag is emitted
    only when != default. The delta is now EXPLICIT in :data:`_FRESH_SEEDED_DELTAS`. Rate-accounting
    implication for the affected v1->v5 runs is APPENDED (not rewritten) in the DAG: their byte-close
    rows must charge the counted uint8-keyframe rate, not the ~1 KB store-nothing rate.

    Deliberately NOT included (per the review's CRITICAL findings — do not "fix" these back in):

      * NO ``--curriculum-event-triggered`` — C1 (plateau eps 1e-3 fires CE->tau ~ep150 mid-descent,
        15% CE-floor loss) + C2 (could converge-fire the l7 DEFECT stage) + M1 (de-synchronizes 3
        epoch-anchored levers). Run-2 lever, gated behind recalibration + boundary re-anchoring.
      * NO ``--bank-n-scales`` change — C3: bank-6 is memory-UNSAFE at n600+self-orient (in_feat 176
        -> cf_mx_cache 86.4 GiB -> projected peak 110.81 GiB REFUSE). bank-4 (trainer default) KEEP;
        re-open only after an fp16 / on-the-fly per-pair feats path lands.
      * NO ``--island-dilate-px`` change — KEEP 1 (the sealed value; the seed acceptance gate is the
        measured ep0 ``part_frac[lane] > 0``).

    NOTE (honest dependency): ``--l7-start-epoch 1001`` satisfies the review's L1 hardening but at
    trainer HEAD the ``--curriculum`` ordering guard requires ``l7_start_epoch <= epochs``
    (train_levelset_witness_realized_through_R_mlx.py:5413) — the operator's parallel trainer edit
    (the C2/L1 wave) is expected to admit the ">= epochs == never" form; until it lands, the trainer
    would fail-loud at startup (a REFUSE, never a silent wrong run).

    means != ends: returns a MEANS (a launch config). Only a byte-closed n600 exact row < 0.19110
    from ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer.
    """
    base = derive_sealed_205_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs,
        code_matrix=code_matrix, byte_close_result=byte_close_result)
    d = _FRESH_SEEDED_DELTAS
    pb = dict(base.proven_base)
    pb.update({
        "lane_prior_phi1_mode": d["lane_prior_phi1_mode"],
        "eikonal_weight": d["eikonal_weight"],
        "softmax_temp_end": d["softmax_temp_end"],
        "st_rewarmup_epochs": d["st_rewarmup_epochs"],
        "st_rewarmup_shape": d["st_rewarmup_shape"],
    })
    alb = dict(base.all_levers_base)
    alb.update({
        "tau_anneal_shape": d["tau_anneal_shape"],
        "lane_band_start_epoch": d["lane_band_start_epoch"],
        "hosc_beta_end": d["hosc_beta_end"],
    })
    prov = dict(base.provenance)
    prov["mod_dim"] = ProvenancedValue(
        int(d["mod_dim"]), SRC_RECALLED,
        "FRESH SEEDED (SEAL review 2026-07-04): mod-dim 19 = the Whitney floor for measured m~9 "
        "(2m+1), paired with --film-stiefel (film weight (768,19) tall => column-orthonormalization "
        "well-posed, review L4). Delta over the sealed Q4 value 32.", Portability.INSTANCE)
    prov["l7_start_epoch"] = ProvenancedValue(
        int(d["l7_start_epoch"]), SRC_RECALLED,
        "FRESH SEEDED (SEAL review L1): l7 'never' hardening — at 1000 (== epochs) the final epoch "
        "still runs l7_softplus (ep < l7_start off-by-one); 1001 makes never mean never.",
        Portability.INSTANCE)
    prov["fresh_seeded_deltas"] = ProvenancedValue(
        dict(d), SRC_RECALLED,
        "fresh_run_config_adversarial_review_20260704.md 'revised launch shape': paint+seed-islands "
        "nucleation fix, eikonal 0.05->0.10 survival step, constant tau=1.0 (geometric, inert-exact), "
        "rewarmup 20-cosine, band 350, muon warm-start+final-frac 0.1, hosc-beta-end 5.134 (M4), "
        "verdict-batch 64, closed-loop control ON; event-triggered curriculum + bank-6 + dilate "
        "changes deliberately EXCLUDED (C1/C2/C3).", Portability.INSTANCE)
    prov["pose_carrier_source"] = ProvenancedValue(
        str(d["pose_carrier_source"]), SRC_RECALLED,
        "#314 / DAG DRIFT-D2 fix (2026-07-06): fresh_seeded's INTENDED frame0 source is the "
        "STORE-NOTHING generated render (Track B 18927a1ae; #205 ledger KEEP row). v1->v5 silently "
        "inherited sealed real_keyframe (flag emitted only when != default) — a rate-accounting "
        "drift (counted uint8-keyframe rate vs ~1 KB store-nothing). Now an explicit delta.",
        Portability.SCORER_FIXED)
    return replace(
        base,
        fresh_seeded=True,
        mod_dim=int(d["mod_dim"]),
        l7_start_epoch=int(d["l7_start_epoch"]),
        verdict_batch=int(d["verdict_batch"]),
        pose_carrier_source=str(d["pose_carrier_source"]),
        proven_base=pb,
        all_levers_base=alb,
        provenance=prov,
    )


# ── T5 CRUCIBLE v6.2 (seal-round-2 BLOCKER-1 fix, 2026-07-08) ────────────────────────────────────
# The crucible-v6 launch-candidate deltas over the STORE-NOTHING #205 base (the config the round-2
# dry-run measured), per DRAFT_OPTIMAL_STACK_v6_20260708.md + seal_round2_v6_verdict_20260708.md.
# Single source consumed by :func:`derive_crucible_v6_config` + the ``crucible_v6`` trailing block
# in :meth:`WitnessConfig._sealed_205_flags` (reuse-not-retype, the fresh_seeded pattern).
_CRUCIBLE_V6_DELTAS: dict = {
    # ── τ leg (v6 §1.4a): endpoint re-anchored to the MEASURED optimum (mod32cap ep650-best
    #    τ = 0.3098; the 0.062 anchor was a proven tautology). P-TAU2 (2026-07-08) corroborates:
    #    knee-derived f_target 0.861663/0.862512 ≈ the q̂ = 0.85 convention; 0.31 STANDS.
    "softmax_temp_end": 0.31,            # proven_base override (family pins 0.05)
    # ── τ SCHEDULE materialization (the BLOCKER-1 wrong-schedule fix). The draft's law is
    #    "descent length 600 ABSOLUTE, endpoint 0.31 HELD through fire band [670,700] + Muon
    #    freeze". The trainer's cosine is UNCLAMPED past the denominator (prog_t=(ep-1)/(ae-1),
    #    L2371/L2386), so --anneal-epochs 600 + cosine REBOUNDS (tau(675)=0.3363,
    #    tau(726-freeze)=0.3826). DERIVED tokens: explicit denominator 3000 + cosine_hold at
    #    hold-frac 0.2 => descent completes at ABSOLUTE ep 600 (0.2*3000) and HOLDS at 0.31:
    #    tau(675) = tau(726) = 0.31 EXACTLY (simulated full-precision against the trainer law;
    #    cross-checked: the same law reproduces mod32cap's measured tau(650)=0.3098 at den 1000).
    "tau_anneal_shape": "cosine_hold",   # all_levers_base override (family: cosine)
    "anneal_epochs": 3000,               # trailing flag: EXPLICIT denominator (family emitted NONE)
    "tau_hold_frac": 0.2,                # trailing flag: 0.2*3000 = descent 600 ABSOLUTE
    # ── hosc-β leg (v6.3 seal-round-1 MAJOR-2(i) fix): the β anneal SHARES --anneal-epochs (trainer
    #    L2334, "review C2" same-den by design), so at den 3000 with the inherited end=4.0/linear the
    #    fire-band β(726-freeze) = 1.7252 (NOT the 1.41 cosine-shape misprint) vs the control's 3.177
    #    (den 1000). The trainer couples all three shared-den schedules (no per-schedule denominators),
    #    so the anchors (ν laws, ep650-best, fire band) — all measured at the control's JOINT (τ,β,LR)
    #    state — require the β TRAJECTORY pinned to the trace. β is LINEAR (--hosc-beta-anneal linear,
    #    inherited), so a linear anneal reproduces ANY target trajectory by choosing the endpoint:
    #    DERIVED-AT-CONFIG law — start 1.0 (inherited), end E over den 3000 matches the control's
    #    slope 3/999 when (E-1)/2999 = 3/999 => E = 9.997 ≈ 10.0 => β(726) = 3.1757 ≈ control 3.177
    #    (0.06% slope match, ≤0.1% on [1,726]). RE-DERIVE TRIGGER: any change to muon-start-epoch (the
    #    freeze point), --anneal-epochs (the denominator), or --hosc-beta-anneal (the shape) re-derives
    #    E. LATENT HAZARD: if the Muon freeze ever becomes event-movable and fires past 726, β keeps
    #    climbing toward 10.0 (past the control's 4.0) — pin is safe ONLY while 726 is the fixed cap.
    # ── KEEP-WITH-PROVENANCE (SEAL v7 R1 structure R-6): the S6-R4 blind derivation independently
    #    proposed β 1→4, NOT 1→10. That 1→4 is SUPERSEDED, auditably, for THIS config: (a) it is the
    #    PRE-ANNEAL-FIX era value (a fixed β-end target, before the joint-schedule anneal fix that
    #    couples β to the shared den-3000 trace); (b) the v6-SEALED MEASURED ANCHOR is the mod32cap
    #    CONTROL's β TRAJECTORY (β(726)≈3.18, reached along a LINEAR anneal), and the DERIVED-AT-CONFIG
    #    law above shows end=10.0 is exactly what reproduces that measured slope on [1,726] at the
    #    shared denominator — the anchors (ν, ep650-best, fire band) were all measured at that β state,
    #    so matching the trajectory (not the endpoint label) is what preserves them; (c) the
    #    annealed-β divergence evidence (fixed-β hosc DIVERGES: tanh(β·sin) saturation → vanishing grad;
    #    the launch config must ANNEAL β, never pin a fixed high β) is why the value is an anneal
    #    ENDPOINT, not a constant — so the raw "1→4 vs 1→10" endpoint comparison is not apples-to-apples.
    #    Net: KEEP 10.0 (measured-trajectory-matched); the blind's 1→4 is a superseded pre-fix anchor.
    "hosc_beta_end": 10.0,               # alb override: β-pin (family: 4.0) — reproduces control β(ep)
    # ── AdamW LR leg (v6.4 seal-round-1 MAJOR-2(ii) BUILD — the RISK ROW resolved by the trainer
    #    --lr-anneal-epochs / --lr-hold-frac build). The LR cosine is the THIRD shared-den sibling
    #    (trainer L6628 read anneal_epochs), but unlike β (LINEAR → endpoint-rephasable) a shallow
    #    den-3000 cosine CANNOT reproduce the control's deep den-1000 descent by endpoint alone (the
    #    CURVATURE differs) — so the pin is a per-schedule DENOMINATOR SPLIT, not an endpoint. DERIVED-
    #    AT-CONFIG: the mod32cap CONTROL ran LR at its own --epochs=1000 denominator (LR/lr_end/warmup =
    #    the shared 1e-3/1e-4/1 defaults), so --lr-anneal-epochs 1000 reproduces the control LR(ep) on
    #    [1,726] BIT-IDENTICALLY (max |Δ|/control = 0.0 over [1,726]; the ep650-best/ν/settle-237/s*/
    #    fire-band laws were ALL measured at that annealed LR). --lr-hold-frac 1.0 = NO hold (the
    #    control's Muon freeze at 726 < den 1000, so LR never held pre-freeze) = bit-identical cosine.
    #    RE-DERIVE TRIGGER: any change to muon-start-epoch (the freeze point), --lr / --lr-end / --warmup-
    #    epochs (the LR trio), or the control vehicle's denominator (1000).
    "lr_anneal_epochs": 1000,            # trailing flag: LR-SPECIFIC cosine denominator (control den)
    "lr_hold_frac": 1.0,                 # trailing flag: NO hold (control had none pre-Muon-freeze)
    # ── ABSOLUTE stage anchors (the seal round-2 assumption-challenge fix: the family SCALES
    #    fractions of --epochs (muon 0.726*3000 = 2178 WRONG); the mod32cap trace anchors are
    #    ABSOLUTE epochs — tau@300 / best~650 / fire~675 / Muon cap 726).
    "tau_softplus_start_epoch": 300,     # WitnessConfig field override (family: 900 at 3000 ep)
    "muon_start_epoch": 726,             # WitnessConfig field override (family: 2178 at 3000 ep)
    # ── v6 §1.1 program pins with 1:1 trainer flags:
    "render_aa": "ipe",                  # alb override: AACoverageRender(mode="ipe") (family: none)
    "lane_band_start_epoch": 350,        # alb override: AnalyticLaneRenderBand(start=350)
    "persistence_warmup_epochs": 275,    # alb override: PersistenceTopology(warmup=275)
    # ── v6.3 seal-round-1 MINOR-4: pin the dwell-law min-stage to the draft's shipped 250 (trainer
    #    default 150; k_max net unchanged at floor((2350-250)/387.1)=5, so no schedule-law shift).
    "curriculum_min_stage_epochs": 250,  # trailing flag (§2.2g dwell law; default 150)
    # NOTE (v6.3 MAJOR-1): --curriculum-plateau-windows is NOT a crucible delta. V=5 binds ONLY the
    # B1 co-predicate spec (no trainer flag exists for it); the sister EXISTING flag
    # --curriculum-plateau-windows is the EP_LOSS-plateau window (a DIFFERENT surface, v5 §0.1 row
    # 3(a)) — silently recalibrating it is the per-epoch-normalization bug class. Left at default.
    "seg_chroma_boundary_weight": 0.1,   # trailing: ChromaBoundarySharpen(weight=0.1)
    "seg_chroma_boundary_margin_band": 1.0,   # trailing: margin_band=1.0
    # start=300 ABSOLUTE (v6.3 MAJOR-3 sister-gap): chroma is NOT in the trainer's re-anchor set
    # (L2049-2072 = persistence-warmup + seed-anneal + analytic-band ONLY), so start="tau_fire"
    # boundary-relative is UNREALIZABLE for chroma as-written; emitted = absolute 300 (= tau@300 cap).
    # A chroma re-anchor path is a named run-2 trainer build item.
    "seg_chroma_boundary_start_epoch": 300,   # trailing: start=300 absolute (chroma re-anchor = run-2)
}

# The v6 §1.1 program levers that are zero-arg composable DSL factories (tac.witness_dsl SoT) —
# composed via the existing dsl_levers merge path so the DSL, not this module, owns each lever's
# flag rendering (triality: "the DSL HOLDS every designed lever"). NOT composed here, with reasons:
# FusedRKernel (not zero-arg composable; emitted directly in the crucible trailing block) ·
# AACoverageRender (zero-arg default is the DISQUALIFIED supersample; v6 wants mode="ipe" ->
# alb render_aa pin) · SignedBoundaryWeight/B16 (DEFAULT-OFF, Q1 BETWEEN) · ConleyCertificate-
# fitted/B17' + GNSpectrumProbe (telemetry/build items, not launch flags). EventTriggeredCurriculum's
# co-predicate V=5 is NOT a trainer flag (v6.3 MAJOR-1): it binds the B1 spec only; the re-anchor leg
# of the event-triggered design (--curriculum-reanchor-levers) IS emitted in the trailing block.
_CRUCIBLE_V6_DSL_LEVERS: tuple[str, ...] = (
    "SeedIslandBirth",            # --seed-islands + --witness-alone-island-loss
    "SeedIslandEased",            # --seed-island-eased (r_star release)
    "EventTriggeredCurriculum",   # --curriculum-event-triggered + --curriculum-nucleus-guard
    "LogitAdjust",                # --logit-adjust-loss-tau 1.0
    "AmplifyIsland",              # --amplify-weight 1.0 (in-place, == sealed value)
    "PersistenceTopology",        # --persistence-loss-weight 1.0 (in-place, == sealed value)
    "CacheGtSkeleton",            # --cache-gt-skeleton
    "LengthSigma",                # --length-sigma-matrix fitted-20260707
    "MuonWarmStart",              # --muon-warm-start-momentum + --muon-lr-final-frac 0.1
    "WeightEntropyPenaltyMLX",    # --weight-entropy-penalty-lambda 15.0
)


def derive_crucible_v6_config(
    gt_cache_path: str | Path,
    *,
    num_pairs: int,
    epochs: int = 3000,
    code_matrix: np.ndarray | None = None,
    byte_close_result: dict | None = None,
) -> WitnessConfig:
    """The **T5 CRUCIBLE v6.2 launch-candidate config** (DRAFT_OPTIMAL_STACK_v6_20260708.md;
    seal-round-2 BLOCKER-1 fix): the STORE-NOTHING #205 base (:func:`derive_store_nothing_205_config`
    — reused, not retyped; the pose block w-pose 1.0 + --pose-carrier + residual-mode table +
    --pose-carrier-source generated is INHERITED STRUCTURALLY = the MAJOR-A2 pin and the #314
    pose-carrier-source inheritance-drift guard) + the v6 deltas (:data:`_CRUCIBLE_V6_DELTAS`) +
    the v6 §1.1 composable DSL levers (:data:`_CRUCIBLE_V6_DSL_LEVERS`).

    What this variant fixes (seal_round2_v6_verdict_20260708.md BLOCKER-1, MEASURED on the real
    launcher dry-run): (a) the extras route was REFUSED (C13 duplicate --softmax-temp-end — the
    family pins 0.05; this variant pins 0.31 at the SOURCE, no extras collision); (b) the named-
    config route silently emitted --muon-start-epoch 2178 (family-scaled 0.726*epochs) and NO
    --anneal-epochs token (denominator fell back to --epochs=3000: tau(675)=0.886 — a run the v6
    schedule laws do not describe). Here the stage anchors are ABSOLUTE (tau@300, Muon cap 726)
    and the τ schedule is pinned: --anneal-epochs 3000 (EXPLICIT) + cosine_hold @ --tau-hold-frac
    0.2 => descent completes at ABSOLUTE ep600 and HOLDS tau = 0.31 through the fire band and the
    Muon freeze (tau(675) = tau(726) = 0.31 exactly; a plain cosine at den 600 would REBOUND to
    0.3363/0.3826 — the derived-not-guessed materialization of the draft's "anneal-epochs 600").

    l7 stays DEMOTED to epochs (the all-levers demote; <=1 trailing epoch under Muon+EMA, absolute-
    safe at any --epochs; Muon@726 < l7 triggers the trainer's WARN-not-refuse placement note,
    which IS the v6 intent: Muon is the effective final stage). The forfeit-arm's LIVE firing
    wiring is the B-INJ pre-GO build; until it lands this config carries the arm ARMED-WITH-
    FALLBACK exactly per v6 §2.2f: cap --muon-start-epoch 726 + the event-triggered co-predicate
    (V=5) + the launcher's auto-started score-neutral #247 shadow observer (advisory).

    Value-provenance ladder tags per pinned constant (req T — the tags guard MEANING; the
    materialization test guards VALUES):
      * softmax_temp_end 0.31: MEASURED-ANCHOR (mod32cap ep650-best τ=0.3098), config-conditional;
        P-TAU2 knee band [0.190724, 0.542937] corroborates. RE-DERIVE on live-m_q law promotion (run-2).
      * tau schedule (anneal 3000 / hold 0.2 / cosine_hold): DERIVED-AT-CONFIG from the trainer τ law
        (descent completes ABSOLUTE ep600, HOLDS 0.31 through fire band + Muon freeze). RE-DERIVE on
        any change to --anneal-epochs / --tau-hold-frac / the descent-length intent.
      * hosc_beta_end 10.0: DERIVED-AT-CONFIG — β is LINEAR, so end=10.0 reproduces the control's
        β(ep) slope on [1,726] to 0.1% at the SHARED den 3000 (the anchors were measured at the
        control's joint β state). RE-DERIVE on any change to muon-start-epoch / --anneal-epochs /
        --hosc-beta-anneal. LATENT HAZARD: β climbs toward 10.0 if the 726 freeze becomes event-movable.
      * muon_start_epoch 726 / tau_softplus_start_epoch 300 / min-stage 250 / band 350 / warmup 275:
        MEASURED-ANCHOR (mod32cap trace, ABSOLUTE epochs), config-conditional to this vehicle.
      * lr schedule (lr_anneal_epochs 1000 / lr_hold_frac 1.0): DERIVED-AT-CONFIG — the LR cosine is
        the THIRD shared-den sibling; a shallow den-3000 cosine cannot reproduce the control's deep
        den-1000 descent by endpoint (curvature differs, unlike LINEAR β), so LR gets its OWN
        denominator. --lr-anneal-epochs 1000 (the control's den) reproduces the control LR(ep) on
        [1,726] BIT-IDENTICALLY; --lr-hold-frac 1.0 = no hold (control never held LR pre-freeze).
        RE-DERIVE on any change to muon-start-epoch / --lr / --lr-end / --warmup-epochs / the control den.

    RESOLVED-BY-BUILD — AdamW LR (v6.4, was the v6.3 MAJOR-2(ii) RISK ROW): the LR cosine used to share
    --anneal-epochs (trainer L6628) with NO shape/hold/denominator flag, so at den 3000 the AdamW phase
    [1,726] ran at 2.83× (fire ep675) → 3.41× (freeze ep726) the control's LR (the control genuinely
    annealed 1e-3 → 2.57e-4 over its den 1000; the crucible stayed near-peak ~8.9e-4) — a 3× deviation
    that STALED every control-derived window law (ν, settle 237, s*, fire band ep675). The v6.4 trainer
    build (--lr-anneal-epochs + --lr-hold-frac, L6620) adds the LR-specific denominator split the pin
    needs; --lr-anneal-epochs 1000 + --lr-hold-frac 1.0 now reproduces the control LR(ep) on [1,726]
    BIT-IDENTICALLY (max |Δ|/control = 0.0), so those anchors are evaluated on the plant they were
    measured on. Default-unset (no --lr-anneal-epochs, --lr-hold-frac 1.0) is byte-identical to the
    pre-build trainer. See prov["lr_schedule"] for the derivation + re-derive triggers.

    means != ends: returns a MEANS (a launch config). Only a byte-closed n600 exact row < 0.19110
    from ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer.
    """
    base = derive_store_nothing_205_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs,
        code_matrix=code_matrix, byte_close_result=byte_close_result)
    # (#351) COMPILE the rot-prone constants from their canonical laws (the constant-compiler): resolve
    # the CONSUMED LawRefs (τ_end / β-pin / LR-anneal-den / LR-hold) into a manifest + values. Each MUST
    # bit-match (value AND type) the sealed literal it replaces — VALUE-IDENTITY IS THE LAW (task #351):
    # the migration changes ZERO emitted values. A drift = fail CLOSED (never silently emit a different
    # config). A missing probe artifact -> the LawRef's declared fallback (the sealed literal) => the
    # launch is NEVER blocked by an absent artifact (manifest records fallback_used).
    from tac.witness_dsl.lawref import resolve_flag_dict_constants
    from tac.witness_dsl.lawref_builtins import (
        CRUCIBLE_V6_CONSUMED_LAWREFS,
        CRUCIBLE_V6_CONSUMED_TARGET_TAGS,
    )
    _resolved, _manifest = resolve_flag_dict_constants(
        CRUCIBLE_V6_CONSUMED_LAWREFS, CRUCIBLE_V6_CONSUMED_TARGET_TAGS)
    for _key, _val in _resolved.items():
        _lit = _CRUCIBLE_V6_DELTAS[_key]
        if not (_val == _lit and type(_val) is type(_lit)):
            raise ValueError(
                f"LawRef value-identity violation (#351): crucible_v6 constant {_key!r} resolved to "
                f"{_val!r} ({type(_val).__name__}) != sealed literal {_lit!r} ({type(_lit).__name__}). "
                "The migration must change ZERO emitted values; re-derive the LawRef or update the "
                "sealed literal DELIBERATELY (both must agree).")
    # d6 = the sealed deltas with the CONSUMED constants overlaid by their LawRef-resolved values
    # (bit-identical). Everything else stays the sealed literal; the config CONSUMES d6 end-to-end.
    d6 = {**_CRUCIBLE_V6_DELTAS, **_resolved}
    pb = dict(base.proven_base)
    pb.update({"softmax_temp_end": d6["softmax_temp_end"]})
    alb = dict(base.all_levers_base)
    alb.update({
        "tau_anneal_shape": d6["tau_anneal_shape"],
        "hosc_beta_end": d6["hosc_beta_end"],
        "render_aa": d6["render_aa"],
        "lane_band_start_epoch": d6["lane_band_start_epoch"],
        "persistence_warmup_epochs": d6["persistence_warmup_epochs"],
    })
    prov = dict(base.provenance)
    prov["softmax_temp_end"] = ProvenancedValue(
        float(d6["softmax_temp_end"]), SRC_MEASURED,
        "v6 §1.4a: τ_end re-anchored to the only MEASURED optimum (mod32cap ep650-best τ=0.3098, "
        "inside its own live-field [τ*(q80), τ*(q90)] = [0.27740, 0.40764]); the 0.062 anchor was "
        "a proven tautology (maps-npz gt_margin key). P-TAU2 corroborates (knee f_target "
        "0.861663/0.862512 ≈ q̂ 0.85; 0.31 STANDS). Live-law promotion waits on run-1 f_target.",
        Portability.INSTANCE)
    prov["tau_schedule"] = ProvenancedValue(
        {"anneal_epochs": d6["anneal_epochs"], "tau_hold_frac": d6["tau_hold_frac"],
         "tau_anneal_shape": d6["tau_anneal_shape"]}, SRC_DESIGN,
        "BLOCKER-1 fix: explicit denominator 3000 + cosine_hold@0.2 => descent completes at "
        "ABSOLUTE ep600 and HOLDS 0.31 (tau(675)=tau(726)=0.31 exactly; plain cosine at den 600 "
        "REBOUNDS unclamped to 0.3363/0.3826 — derived from trainer L2371/L2386, cross-checked "
        "against mod32cap's measured tau(650)=0.3098).", Portability.INSTANCE)
    prov["hosc_beta_end"] = ProvenancedValue(
        float(d6["hosc_beta_end"]), SRC_DERIVED,
        "v6.3 MAJOR-2(i) DERIVED-AT-CONFIG: the hosc-β anneal shares --anneal-epochs (trainer L2334), "
        "so at den 3000 the inherited end=4.0/linear gives β(726-freeze)=1.7252 vs the control's 3.177 "
        "(den 1000) — the anchors (ν, ep650-best, fire band) were measured at the control's JOINT β "
        "state. β is LINEAR, so it reproduces any target trajectory by endpoint choice: matching the "
        "control slope 3/999 needs (E-1)/2999 = 3/999 => E=9.997≈10.0 => β(726)=3.1757≈3.177 (0.06% "
        "slope match, ≤0.1% on [1,726]). RE-DERIVE on any change to muon-start-epoch / --anneal-epochs "
        "/ --hosc-beta-anneal. LATENT HAZARD: if the Muon freeze becomes event-movable past 726, β "
        "climbs toward 10.0 (past 4.0) — safe only while 726 is the fixed cap.", Portability.INSTANCE)
    prov["lr_schedule"] = ProvenancedValue(
        {"lr_anneal_epochs": d6["lr_anneal_epochs"], "lr_hold_frac": d6["lr_hold_frac"]}, SRC_DERIVED,
        "v6.4 MAJOR-2(ii) DERIVED-AT-CONFIG (the RISK ROW resolved by the trainer --lr-anneal-epochs / "
        "--lr-hold-frac build): the LR cosine is the THIRD shared-den sibling (trainer L6628). Unlike β "
        "(LINEAR → endpoint-rephasable), a shallow den-3000 cosine cannot reproduce the control's deep "
        "den-1000 descent by endpoint alone (the curvature differs) — so LR gets its OWN denominator. "
        "The mod32cap CONTROL ran LR at its --epochs=1000 den with the shared 1e-3/1e-4/1 LR trio, so "
        "--lr-anneal-epochs 1000 reproduces the control LR(ep) on [1,726] BIT-IDENTICALLY (max |Δ|/"
        "control = 0.0 over [1,726]; the ep650-best/ν/settle-237/s*/fire-band laws were measured at "
        "that annealed LR — this pin STALE-PROOFS them). --lr-hold-frac 1.0 = NO hold (the Muon freeze "
        "726 < den 1000, so the control never held LR pre-freeze) = bit-identical cosine. RE-DERIVE on "
        "any change to muon-start-epoch / --lr / --lr-end / --warmup-epochs / the control den 1000.",
        Portability.INSTANCE)
    prov["muon_start_epoch"] = ProvenancedValue(
        int(d6["muon_start_epoch"]), SRC_RECALLED,
        "v6 §2.2f: ABSOLUTE Muon fail-safe cap 726 (mod32cap trace anchor; fire band [670,700] "
        "precedes it). NOT the family-scaled 0.726*epochs (=2178 at 3000 ep — the measured "
        "wrong-schedule emission the seal round-2 BLOCKER caught).", Portability.INSTANCE)
    prov["tau_softplus_start_epoch"] = ProvenancedValue(
        int(d6["tau_softplus_start_epoch"]), SRC_RECALLED,
        "v6 schedule: ABSOLUTE tau stage start 300 (mod32cap trace anchor; the ~425-ep TAU to the "
        "726 cap). NOT the family-scaled 0.300*epochs (=900 at 3000 ep). Under the event-triggered "
        "curriculum this is the CE->TAU CAP; the plateau+nucleus event may fire earlier (settle "
        "3/ν_CE = 150.3).", Portability.INSTANCE)
    prov["crucible_v6_deltas"] = ProvenancedValue(
        dict(d6), SRC_DESIGN,
        "DRAFT_OPTIMAL_STACK_v6 §1.1/§1.4a/§2.2f-g pins: F-DET fused-R + β-pin (hosc-beta-end 10.0) + "
        "re-anchor levers + min-stage 250 + ChromaBoundarySharpen(0.1, band 1.0, start 300 absolute) "
        "+ AA ipe + band 350 + persistence warmup 275 + LR-pin (lr-anneal-epochs 1000, lr-hold-frac "
        "1.0); DSL levers composed: "
        + ", ".join(_CRUCIBLE_V6_DSL_LEVERS) + ". Pose block INHERITED from store_nothing_205 "
        "(MAJOR-A2 pin; #314 drift guard). v6.3 seal-round-1: dropped --curriculum-plateau-windows "
        "(wrong surface, MAJOR-1); pinned β (MAJOR-2(i)). v6.4: AdamW LR RISK ROW RESOLVED-BY-BUILD "
        "(MAJOR-2(ii) — --lr-anneal-epochs 1000 + --lr-hold-frac 1.0 reproduce the control LR(ep) on "
        "[1,726] bit-identically; the trainer build added the LR-specific denominator split).",
        Portability.INSTANCE)
    cfg = replace(
        base,
        crucible_v6=True,
        tau_softplus_start_epoch=int(d6["tau_softplus_start_epoch"]),
        muon_start_epoch=int(d6["muon_start_epoch"]),
        dsl_levers=_CRUCIBLE_V6_DSL_LEVERS,
        proven_base=pb,
        all_levers_base=alb,
        provenance=prov,
        # (#351) the config CONSUMES the LawRef-resolved deltas (the trailing block reads this) +
        # carries the constants_manifest for the launcher to write beside launch.sh. Both are
        # provenance-only (never emitted as flags); the argv is byte-identical to the pre-migration form.
        crucible_v6_deltas=d6,
        constants_manifest=_manifest,
    )
    # (#353 DSL-authored-config gate) VALIDATE the crucible config's DSL-authorable schedule/curriculum
    # knobs through the typed DSL layer (fail-CLOSED), then ATTACH the DSL-provenance manifest the
    # launcher gate checks. This is the "config must be DSL-defined + typed + validated" requirement V:
    # the typed layer refuses a malformed provenance/range/schedule shape BEFORE any launch, and the
    # attached manifest proves the config was authored through the typed layer (a hand-crafted argv
    # carries no manifest => the launcher REFUSES it). ADDITIVE + argv-inert: dsl_program_manifest is
    # never emitted as a flag, so the emitted argv is BYTE-IDENTICAL to the pre-#353 form.
    cfg = _attach_dsl_program_manifest(cfg, program_name="crucible_v6", d6=d6)
    return cfg


def _attach_dsl_program_manifest(cfg: "WitnessConfig", *, program_name: str, d6: dict) -> "WitnessConfig":
    """Typed-validate the config's DSL-authorable schedule knobs + attach the DSL-provenance manifest.

    Fail-CLOSED: a typed-schema violation (malformed provenance/range/governance) or a
    WitnessProgram.validate violation on the authored schedule raises — the config is NEVER
    launched un-validated. The manifest's flag fingerprint is taken from the config's ACTUAL
    emitted argv (a canonical placeholder out-dir), so the launcher can match it against what it
    emits. Provenance-only (the manifest is never a flag) => the emitted argv is unchanged.
    """
    from tac.local_acceleration.scorer_throughput_gate import derive_wall_clock_budget_days
    from tac.witness_dsl.typed_config import (
        Provenanced,
        ProvenanceClass,
        TypedAnneal,
        TypedRegularizer,
        TypedStage,
        TypedWitnessConfig,
        build_launch_manifest,
    )

    _PC = ProvenanceClass
    # The DSL-authorable schedule surface of the crucible config, each knob given its ladder class
    # (mirroring the ProvenancedValue provenance recorded above). This is the TYPED re-expression the
    # gate validates; it is NOT the emitter (the WitnessConfig argv remains the launch SoT).
    typed = TypedWitnessConfig(
        name=program_name,
        out_dir="experiments/results/__dsl_typed_gate__",
        gt_cache=str(cfg.gt_cache),
        num_pairs=int(cfg.num_pairs),
        epochs=int(cfg.epochs),
        wall_clock_budget_days=Provenanced(
            value=round(derive_wall_clock_budget_days(int(cfg.epochs)), 3),
            provenance=_PC.DERIVED_AT_CONFIG, unit="days",
            source="scorer_throughput_gate.derive_wall_clock_budget_days"
                   "(anchor RUN1_MEASURED_MIN_PER_EP x epochs x WALL_CLOCK_SLACK_FACTOR)"),
        mlx_device="gpu",
        temp=TypedAnneal(
            start=Provenanced(value=1.0, provenance=_PC.MEASURED_ANCHOR, unit="tau",
                              source="render-anneal start (proven base)"),
            end=Provenanced(value=float(d6["softmax_temp_end"]), provenance=_PC.MEASURED_ANCHOR,
                            unit="tau", source="mod32cap ep650-best tau=0.3098 (config-conditional)"),
        ),
        stages=(
            TypedStage(name="tau_softplus", start_epoch_flag="--tau-softplus-start-epoch",
                       start_epoch=int(d6["tau_softplus_start_epoch"])),
            TypedStage(name="muon", start_epoch_flag="--muon-start-epoch",
                       start_epoch=int(d6["muon_start_epoch"])),
        ),
        regularizers=(
            TypedRegularizer(flag="--eikonal-weight", weight=Provenanced(
                value=0.01, provenance=_PC.DERIVED_AT_CONFIG, unit="dimensionless",
                source="theta* lever stack")),
            TypedRegularizer(flag="--length-weight", weight=Provenanced(
                value=0.001, provenance=_PC.DERIVED_AT_CONFIG, unit="dimensionless",
                source="theta* lever stack")),
        ),
        # base/schedule_governance intentionally EMPTY here: this typed gate validates the
        # schedule/provenance SHAPE; the schedule-provenance sibling gate (rc=6) owns the governance
        # CONTENT of the emitted config. Leaving governance empty does not mask that gate.
    )
    prog_viol = typed.validate_program()
    if prog_viol:
        raise ValueError(
            f"DSL-authored-config gate (#353): typed schedule for {program_name!r} produced "
            f"{len(prog_viol)} WitnessProgram.validate violation(s): {prog_viol[:4]}"
        )
    # Flag NAMES are out-dir-independent (out_dir is the VALUE of --out-dir), so any placeholder
    # yields the same fingerprint the launcher matches against cfg.to_trainer_flags(real_out_dir).
    emitted = [flag for flag, _ in cfg.to_trainer_flags("OUT")]
    manifest = build_launch_manifest(
        program_name=program_name,
        emitted_flag_names=emitted,
        typed_config_hash=typed.typed_config_hash(),
        typed_validated=True,
    )
    return replace(cfg, dsl_program_manifest=manifest)


# ═══════════════════════════════════════════════════════════════════════════════
# T5 CRUCIBLE v7 — the FIRST requirement-V-native launch config.
#
# Operator binding 2026-07-08 (requirement V): "The config must be defined in the DSL — no
# ad hoc or hand crafting." Unlike crucible_v6 (a WitnessConfig whose argv is emitted by the
# autoconfig dataclass, with a typed manifest ATTACHED for the gate), crucible_v7 IS a
# ``TypedWitnessConfig`` — its argv is compiled by the DSL ``WitnessProgram.compile_trainer_argv``.
# The authoring surface is the typed schema; the emitter is the DSL. There is NO hand argv and NO
# parallel dict assembly of a NEW config: the substrate ``base`` is the SEALED v6 emitted flag set,
# REUSED not retyped (the same reuse law by which v6 reuses store_nothing_205), transformed by the
# five naked-violation resolutions of ``DRAFT_v7_restart_config_synthesis_20260708.md`` §1.
#
# The five naked-violation resolutions (the schedule-provenance gate's to-fix spec):
#   1. --tau-softplus-start-epoch  -> DELETED (dissolved into the continuous L_tau via
#      --seg-form-unify-tau; the trainer's validate_seg_form_unify_tau_config REFUSES both).
#   2. --l7-start-epoch            -> DELETED (l7 = measured DEFECT; inert under unify).
#   3. --muon-start-epoch 726      -> TAGGED FAIL-SAFE CAP (schedule_governance).
#   4. --lane-band-start-epoch     -> 350->500, TAGGED FAIL-SAFE CAP.
#   5. --seg-chroma-boundary-...   -> 300->450, TAGGED FAIL-SAFE CAP.
# Gate outcome: 0 NAKED (2 deleted, 3 governance-tagged). Schedule spine: cosine_hold->geometric
# tau anneal (floor tau*=0.31 unchanged) + TAIL_k + LADDER; pose block VERBATIM from v6.
#
# means != ends: returns a MEANS. Only a byte-closed n600 exact row < 0.19110 from
# ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer 0.19110.
# ═══════════════════════════════════════════════════════════════════════════════

# The v7 restart's tagged fail-safe caps (req-B) — event-triggered in intent, fixed-epoch as the
# backstop the wired event controller fires before (DRAFT §1). Named constants so the tests +
# council review + the diff-vs-v6 table read the exact values in one place.
_CRUCIBLE_V7_MUON_CAP = 726          # inherited from v6 (mod32cap fire band [670,700] precedes it)
_CRUCIBLE_V7_LANE_BAND_CAP = 500     # DRAFT §1 [council_pending]: past the eased-seed window
_CRUCIBLE_V7_CHROMA_CAP = 450        # DRAFT §1 [council_pending]: past a formed margin boundary
_CRUCIBLE_V7_TAIL_CYCLES_MAX = 2     # DRAFT §6.2 [council_pending]: propose k_max = 2

# ── hosc-β EVENT-mode endpoint (SEAL v7.3 round-2 BLOCKER fix, 2026-07-08) ──────────────────────
# THE BLOCKER (seal_v73_r2_deepmath): v6 DERIVED hosc_beta_end = 10.0 as a CLOCK-frame endpoint — a
# LINEAR anneal on the shared den-3000 clock passes through β(ep726)=3.177 there, then FREEZES at the
# Muon switch (β never physically reaches 10.0 in clock mode; frozen at 726/3000 = 24% of the anneal).
# v7.3 flips --tau-advance-mode to EVENT, where β is driven by the OCTAVE FRACTION, not the ep-clock:
#   hosc_beta_for_epoch = β_start + (β_end − β_start)·octave_fraction()   (tau_advance.py:282,285)
# and the ladder (hence β) FREEZES at the Muon switch. The τ*=0.31 TURNPIKE premise REQUIRES Muon to
# fire when τ reaches its floor rung ⟺ octave_fraction ≈ 1.0 — but β rides the SAME octave_fraction, so
# at the freeze β = β_start + (β_end − β_start)·1.0 = β_end. With the inherited β_end=10.0 that FREEZES
# β ≈ 10.0 for the entire ~2274-ep post-Muon turnpike+TAIL — squarely the FORBIDDEN fixed-high-β
# saturation regime (tanh(β·sin) → vanishing grad → AdamW random-walk → d_seg RISES; CLAUDE.md capstone
# non-negotiable "NEVER fixed β=4"; 10.0 is 2.5× past it) AND invalidating every anchor (ν, ep650-best,
# fire-band) that was MEASURED at the control's frozen β≈3.18. The v6 comment stamped the RE-DERIVE
# TRIGGER for exactly "any change to --hosc-beta-anneal (the shape)" and the mode flip fires it.
# DERIVATION (event-mode): the frozen β = β_end (octave_fraction→1.0 at the ladder floor, ≥ any earlier
# rung), so to PRESERVE the control's healthy frozen β we set β_end to that intended FROZEN value, NOT
# the clock endpoint. The control's frozen β is the mod32cap linear β(726) on ITS own den-1000:
#   β(726) = β_start + (β_control_end − β_start)·(726−1)/(1000−1) = 1 + (4−1)·725/999 = 3.17718 ≈ 3.177.
# So the event-mode endpoint = 3.177 (β anneals 1.0→3.177 over the octaves, FREEZES at ≤3.177 for the
# tail — β_end is the CEILING since frozen β = β_end·[octave_fraction≤1]). ≤ 4.0 honors the anneal-β
# divergence bound; the GPU bit-cert's [1,10] sweep already covers 3.177 (the cert range is a
# superset — bit-identity is β-value-invariant within the domain). PROVENANCE FLIP: v6's
# DERIVED-AT-CONFIG (clock-endpoint) → v7 DERIVED-AT-CONFIG (event-frozen-value); the value is the
# control's MEASURED frozen β, re-derived for the EVENT driver. RE-DERIVE TRIGGER (unchanged form): any
# change to --hosc-beta / the control vehicle (mod32cap β trajectory) / --tau-advance-mode re-derives it.
_CRUCIBLE_V7_HOSC_BETA_END_EVENT = 3.177  # event-mode frozen-β endpoint = control β(726) (BLOCKER fix)

# ── directional-basis REGIME (SEAL v7.3 round-2 M1 fix, 2026-07-08) ─────────────────────────────
# The FEED-07a two-regime allocation the whole v7 seg composition must be COHERENT with. Named ONCE
# so the DirectionalBasisRebalance lever AND the persistence-recall class-targeting coupling
# (persistence_classes_for_basis_regime) read the SAME regime (no drift). lane_offloaded = the FREE
# rule-118 analytic band carries lane (freq_along≈6 cartoon basis); lane_carried (freq_along≈26) is
# the registered counter-arm. v7.3 COMMITS to lane_offloaded per the operator-APPROVED basis rec.
_CRUCIBLE_V7_BASIS_REGIME = "lane_offloaded"

# Flags the DSL ``WitnessProgram`` emits ITSELF (flag_dict header + Preserve) OR that the typed
# temp / stages / regularizers OWN — they must NOT ALSO live in ``base`` (else a double-emit).
_CRUCIBLE_V7_PROGRAM_OWNED: frozenset[str] = frozenset({
    "--out-dir", "--gt-cache", "--num-pairs", "--epochs", "--mlx-device",  # flag_dict header
    "--ckpt-every", "--stage-checkpoints",                                  # Preserve.flags
    "--softmax-temp-start", "--softmax-temp-end",                           # temp TypedAnneal
    "--muon-start-epoch",                                                   # muon TypedStage
    "--eikonal-weight", "--length-weight",                                  # TypedRegularizers
})

# Flags v7 DELETES outright (DRAFT §1 resolutions 1+2 + the cosine_hold-only knob geometric drops).
_CRUCIBLE_V7_DELETED: frozenset[str] = frozenset({
    "--tau-softplus-start-epoch",   # dissolved by --seg-form-unify-tau (trainer refuses both)
    "--l7-start-epoch",             # l7 measured DEFECT; inert under unify (trainer default is 'never')
    "--tau-hold-frac",              # cosine_hold-only; the geometric anneal has no hold segment
})

# The FIVE composable v7 DSL levers (Lever factories in curriculum_dsl — triality: "the DSL HOLDS
# every designed lever"). Applied AS TypedLevers so the emitter stays the DSL, never a hand dict.
# NAMES are the Lever.name (snake_case), matching the CrucibleV7LaunchConfig.dsl_levers property
# (``lv.name for lv in typed.levers``) — the activation-ledger surface the real launcher records.
_CRUCIBLE_V7_DSL_LEVERS: tuple[str, ...] = (
    "seg_form_unify_tau",           # --seg-form-unify-tau (continuous L_tau; removes last PR95 bone)
    "tail_k_warm_restart",          # --tail-* (post-Muon warm-restart cycles; k_max fail-safe cap)
    "n323_ladder_island_homotopy",  # --ladder-* (per-class-lambda-gated island-birth homotopy)
    "FEED_07a_directional_basis_rebalance",  # --freq-across/--freq-along/--n-dir-freqs/--self-orient
                                             # (seal v7 r1 R-1; operator APPROVED; Arm-A first fire)
    "R7_polyak_finisher",           # --polyak-finisher-* (R-7 finisher 2; v7.3 delta 2, synthesis item 8
                                    # IN-v7; start_epoch SIZED to the TAIL turnpike, DERIVED-AT-CONFIG)
)

# Flags whose delta vs v6 is a run-dir artifact (NOT a config semantic) — excluded from the
# operator-facing diff-vs-v6 table (the council reviews SEMANTIC deltas, not the placeholder out-dir).
_CRUCIBLE_V7_DIFF_IGNORE: frozenset[str] = frozenset({"--out-dir", "--gt-cache"})

_CRUCIBLE_V7_PLACEHOLDER_OUT = "experiments/results/__crucible_v7__"


@dataclass(frozen=True)
class CrucibleV7Compiled:
    """The compiled v7 artifact: the typed config + its argv + both manifests + the v6 baseline.

    Everything the launcher gate chain (DSL-manifest gate rc=7 + schedule-provenance gate rc=6)
    and the council review surface (the diff-vs-v6 table) need, in one immutable record.
    """

    typed: object                       # TypedWitnessConfig (the requirement-V-native artifact)
    argv: tuple[str, ...]               # the DSL-compiled trainer argv (WitnessProgram emitter)
    emitted_pairs: tuple[tuple[str, object], ...]  # (flag, value) parse of argv (gate input)
    constants_manifest: dict            # the LawRef constants (inherited from v6; same sealed values)
    dsl_program_manifest: dict          # build_launch_manifest attestation (the launcher rc=7 gate)
    schedule_governance: dict           # {flag: {class, sensor, rationale}} (the rc=6 gate input)
    v6_flags: tuple[tuple[str, object], ...]  # the v6 sealed emitted flags (diff baseline)
    tail_constant_provenance: dict = field(default_factory=dict)  # (S4-R2/S1-R1) req-T rows for the
    #   TAIL constants (stop_marginal_s / tau_halving HARDCODED-WITH-WAIVER; cycle_floor / dwell LawRef)
    #   + the LADDER λ-gate provenance — every sealed TAIL/λ literal is auditable (no silent literals).

    def to_launch_config(self) -> "CrucibleV7LaunchConfig":
        """The launcher-facing cfg (the FULL duck-typed cfg protocol tools/launch_witness_run.py
        consumes) — the emit adapters delegate to :attr:`typed`; the three provenance manifests are
        this compiled artifact's. See :class:`CrucibleV7LaunchConfig`. (seal v7 r1 BLOCKER #1 + MAJOR
        #2: the ONE object that satisfies BOTH the emit protocol AND the gate-chain manifests.)"""
        return CrucibleV7LaunchConfig(
            typed=self.typed,
            constants_manifest=dict(self.constants_manifest),
            dsl_program_manifest=dict(self.dsl_program_manifest),
            schedule_governance=dict(self.schedule_governance),
        )


def _crucible_v7_argv_pairs(argv: "list[str] | tuple[str, ...]") -> list[tuple[str, object]]:
    """Parse a compiled trainer argv (``[python, trainer, --flag, val, --bare, ...]``) into
    ``(flag, value)`` pairs — a bare boolean flag -> value ``None`` — the exact shape the
    schedule-provenance gate + the diff table consume. Pure (unit-testable at $0)."""
    toks = list(argv)
    # skip the leading interpreter + trainer path (the first two non-flag tokens).
    i = 0
    while i < len(toks) and not str(toks[i]).startswith("--"):
        i += 1
    pairs: list[tuple[str, object]] = []
    while i < len(toks):
        t = str(toks[i])
        if t.startswith("--"):
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if nxt is not None and not str(nxt).startswith("--"):
                pairs.append((t, nxt)); i += 2
            else:
                pairs.append((t, None)); i += 1
        else:
            i += 1
    return pairs


def diff_crucible_v6_to_v7(
    v6_flags: "list[tuple[str, object]] | tuple[tuple[str, object], ...]",
    v7_pairs: "list[tuple[str, object]] | tuple[tuple[str, object], ...]",
) -> dict:
    """The operator-facing diff-vs-v6 table (the council's review surface). Returns
    ``{added, removed, changed}`` where a bare flag is normalised to ``True`` on both sides and
    values are compared by their emitted-token string (so ``350`` vs ``'500'`` is a real change,
    not a type artifact). The run-dir placeholder (``--out-dir`` / ``--gt-cache``) is excluded —
    it is not a config semantic. Pure (unit-testable at $0)."""
    def _norm(pairs):
        return {f: (True if v is None else v) for f, v in pairs
                if f not in _CRUCIBLE_V7_DIFF_IGNORE}
    a, b = _norm(v6_flags), _norm(v7_pairs)
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed = sorted(f for f in set(a) & set(b) if str(a[f]) != str(b[f]))
    return {
        "added": [(f, b[f]) for f in added],
        "removed": [(f, a[f]) for f in removed],
        "changed": [(f, a[f], b[f]) for f in changed],
        "v6_flag_count": len(a),
        "v7_flag_count": len(b),
    }


def _crucible_v7_schedule_governance() -> dict:
    """The three transitions as EVENT+BACKSTOP pairs (operator override 2026-07-08, S4 R1).

    The operator OVERRODE the T3 council's launch-with-caps consensus (verbatim: *"Build the
    wirings dumbass that's the whole point"*): each transition now FIRES ON ITS WIRED SENSOR
    (a co-emitted ``--<x>-start-event`` flag: powerlaw_meat / lane_nucleus / annulus_plateau)
    with the numeric ``--<x>-start-epoch`` demoted to a fail-safe BACKSTOP CAP. So the governance
    surface declares SIX entries — three ``role=fires`` EVENTS (the sensor-fired transitions) and
    three ``role=backstops`` CAPS (each referencing the co-emitted event it backs up). The S4 R1
    ``role`` discriminator makes a CAP's ``sensor`` un-misreadable as a firing claim: an EVENT
    names the wired sensor it fires ON; a CAP names the EVENT it backs up. The schedule-provenance
    gate classifies the events EVENT_TRIGGERED and the epoch backstops FAIL_SAFE_CAP."""
    from tac.witness_dsl.typed_config import ScheduleGovernance
    return {
        # ── EVENTS (role=fires): the runtime sensor-fired transitions the wirings implement.
        "--muon-start-event": ScheduleGovernance(**{
            "class": "event", "role": "fires", "sensor": "--muon-start-event",
            "rationale": (
                "Muon metric-finisher entry FIRES on the powerlaw_meat exit of the tau-descent "
                "(tac.witness_control.event_wirings.muon_meat_event), gated on the S2 REV-B "
                "nucleation-complete positive control (all LADDER arms past birth+hold+anneal) so "
                "an island-birth transient cannot be misread as first-order tau exhaustion."),
        }),
        "--lane-band-start-event": ScheduleGovernance(**{
            "class": "event", "role": "fires", "sensor": "--lane-band-start-event",
            "rationale": (
                "Analytic lane band FIRES on the lane-class critical nucleus (born part_frac>0 AND "
                "formed within_flip<=thresh — the #315/#302 per-class hand-off predicate applied to "
                "the lane class; pi1=w/sigma~=5). S3 would_fire telemetry accrues calibration data."),
        }),
        "--seg-chroma-boundary-start-event": ScheduleGovernance(**{
            "class": "event", "role": "fires", "sensor": "--seg-chroma-boundary-start-event",
            "rationale": (
                "Chroma boundary sharpener FIRES on the #333 annulus_frac PLATEAU (a FORMED margin "
                "boundary is the thing chroma sharpens); the plateau detector params carry req-T "
                "tagged provenance (sisters of the curriculum-plateau params)."),
        }),
        # ── BACKSTOP CAPS (role=backstops): the fixed epochs, each referencing its event.
        "--muon-start-epoch": ScheduleGovernance(**{
            "class": "cap", "role": "backstops", "sensor": "--muon-start-event",
            "rationale": (
                "req-B fail-safe BACKSTOP for the powerlaw_meat event (--muon-start-event): 726 = "
                "nu-law settle + floor derivation (the mod32cap fire band [670,700] precedes it); "
                "fires ONLY if the sensor did not by 726 (LOUD cap_fired_before_event, S5)."),
        }),
        "--lane-band-start-epoch": ScheduleGovernance(**{
            "class": "cap", "role": "backstops", "sensor": "--lane-band-start-event",
            "rationale": (
                "req-B fail-safe BACKSTOP for the lane-nucleus event (--lane-band-start-event): 500 "
                "= past the eased-seed window (v6 hand-guessed 350); fires ONLY if the lane nucleus "
                "did not form by 500 (LOUD cap_fired_before_event, S5)."),
        }),
        "--seg-chroma-boundary-start-epoch": ScheduleGovernance(**{
            "class": "cap", "role": "backstops", "sensor": "--seg-chroma-boundary-start-event",
            "rationale": (
                "req-B fail-safe BACKSTOP for the annulus_plateau event "
                "(--seg-chroma-boundary-start-event): 450 = past a formed margin boundary; fires "
                "ONLY if the annulus did not plateau by 450 (LOUD cap_fired_before_event, S5)."),
        }),
        # ── S6-R4 self-paced τ-advance (operator 2026-07-08): the LAST clock-hardcoding (the anneal-
        #    epochs denominator that clocks τ(t)) converted to an EVENT-driven geometric octave ladder.
        "--tau-advance-mode": ScheduleGovernance(**{
            "class": "event", "role": "fires", "sensor": "--tau-advance-mode",
            "rationale": (
                "τ-anneal advances on the per-band RELAXATION of the through-R seg loss WITHIN the "
                "current octave (powerlaw_meat), NOT a fixed --anneal-epochs clock (S6-R4 element 5: "
                "τ advances on per-scale relaxation, self-triggered, one param at a time). The octave "
                "ladder reuses the geometric clock VALUES (τ_k=start·(end/start)^(k/N)); only the "
                "per-rung DWELL is event-driven. The per-octave MAX-DWELL fail-safe backstop is "
                "--tau-octave-max-dwell, which is TRAINER-DERIVED-INTERNAL (derive_octave_max_dwell = "
                "ceil(anneal/N)*slack, from --anneal-epochs + --curriculum-min-stage-epochs) and is "
                "NOT an emitted launch token — so it is intentionally absent from this governance "
                "surface (seal v7 r1 MINOR #3: every schedule_governance KEY is an emitted launch "
                "flag, so the audit table describes launch.sh reality; the derived dwell's provenance "
                "lives in the trainer's derive_octave_max_dwell, not a phantom CAP row here)."),
        }),
    }


def crucible_v7_wiring_gaps() -> list[str]:
    """The sensor->start WIRING STATUS (operator override 2026-07-08 BUILT the three that were OWED
    gaps in the pre-registered v7 memo). Each entry names the transition, its now-WIRED sensor, and
    the fail-safe backstop cap it demotes the fixed epoch to. Retained as the council's audit surface
    (the wirings ARE the deliverable; a firing backstop cap is falsification-relevant per S5)."""
    return [
        "muon: WIRED — --muon-start-event powerlaw_meat fires the Muon switch on the tau-descent "
        "weak-KAM exhaustion (tac.witness_control.event_wirings.muon_meat_event) gated on the S2 "
        "REV-B nucleation-complete positive control; --muon-start-epoch 726 is the fail-safe backstop.",
        "lane-band: WIRED — --lane-band-start-event lane_nucleus fires analytic-band engagement on "
        "the lane-class critical nucleus (born + formed, the #315/#302 per-class predicate); S3 "
        "would_fire telemetry accrues; --lane-band-start-epoch 500 is the fail-safe backstop.",
        "chroma: WIRED — --seg-chroma-boundary-start-event annulus_plateau fires chroma engagement "
        "on the #333 annulus_frac plateau detector (promoted from observability to a trigger); "
        "--seg-chroma-boundary-start-epoch 450 is the fail-safe backstop.",
    ]


def crucible_v7_basis_allocation_provenance(*, num_pairs: int = 600,
                                            total_ram_gib: float = 128.0) -> dict:
    """The MEMORY-WATERFILL derivation behind the v7 Arm-A basis lever (seal v7 r1 R-1; operator
    APPROVED 2026-07-08). NO bare numbers: every value is re-derived here from the REAL preflight
    projection (``tools/witness_memory_preflight.project_peak_rss_gib``) at the candidate in_feats,
    so the council/manifest reads the same math the memo's waterfill table records.

    The candidates (freq allocation -> cf-feature bank -> in_feat -> peak RSS):
      * run-1 / v6 baseline (n_dir_freqs 2, freq_along 4)            -> in_feat 88.
      * minimal along-only rebalance (n_dir_freqs 2, freq_along 6/8) -> in_feat 88 (freq_along VALUE
        does not enter in_feat: trainer ``dir_w = 4 * n_dir_freqs``, MEMORY-NEUTRAL).
      * DSL lever lane_offloaded (n_dir_freqs 4, freq_along 6)       -> in_feat 96 (+8 = 4*(4-2)).

    Verdict: the lane_offloaded allocation's projected peak fits the envelope with the standard
    margin under BOTH the 0.85 sole-workload and the conservative 0.70 concurrent fractions, so the
    DERIVED DSL lever is preferred as-designed (no fall-through to a minimal rebalance). Pure +
    unit-testable ($0). means != ends: advisory memory sizing; the pointer 0.19110 is UNMOVED."""
    import sys as _sys
    from pathlib import Path as _Path
    _tools = _Path(__file__).resolve().parents[2] / "tools"
    if str(_tools) not in _sys.path:
        _sys.path.insert(0, str(_tools))
    import witness_memory_preflight as _wmp  # the REAL projection fn (imported, never forked)

    def _peak(in_feat: int, safe_frac: float) -> object:
        return _wmp.project_peak_rss_gib(
            num_pairs=int(num_pairs), render_h=384, render_w=512, in_feat=int(in_feat),
            self_orient=True, verdict_batch=32, render_aa="ipe",
            total_ram_gib=float(total_ram_gib), safe_frac=safe_frac)

    base_in_feat, lever_in_feat = 88, 96          # derived above (dir_w = 4*n_dir_freqs; +8 for 2->4)
    b70 = _peak(base_in_feat, 0.70)               # baseline @ conservative frac (delta reference)
    l70, l85 = _peak(lever_in_feat, 0.70), _peak(lever_in_feat, 0.85)
    return {
        "regime": "lane_offloaded",
        "chosen_lever": "FEED_07a_directional_basis_rebalance",
        "freq_across": 32, "freq_along": 6, "n_dir_freqs": 4,
        "candidates": {
            "v6_baseline": {"n_dir_freqs": 2, "freq_along": 4, "in_feat": base_in_feat,
                            "peak_gib": b70.projected_peak_gib, "cf_cache_gib": b70.cf_cache_gib},
            "minimal_along_only": {"n_dir_freqs": 2, "freq_along": "6/8", "in_feat": base_in_feat,
                                   "note": "MEMORY-NEUTRAL (freq_along value does not enter in_feat)"},
            "dsl_lever_lane_offloaded": {"n_dir_freqs": 4, "freq_along": 6, "in_feat": lever_in_feat,
                                         "peak_gib": l70.projected_peak_gib,
                                         "cf_cache_gib": l70.cf_cache_gib},
        },
        "in_feat_delta": lever_in_feat - base_in_feat,
        "cf_cache_delta_gib": round(l70.cf_cache_gib - b70.cf_cache_gib, 2),
        "peak_delta_gib": round(l70.projected_peak_gib - b70.projected_peak_gib, 2),
        "envelope_0p70_gib": round(0.70 * total_ram_gib, 1),
        "envelope_0p85_gib": round(0.85 * total_ram_gib, 1),
        "margin_0p70_gib": round(0.70 * total_ram_gib - l70.projected_peak_gib, 1),
        "margin_0p85_gib": round(0.85 * total_ram_gib - l85.projected_peak_gib, 1),
        "admitted_0p70": bool(l70.safe), "admitted_0p85": bool(l85.safe),
        "projected_step_cost_ratio": round(lever_in_feat / base_in_feat, 4),  # ~1.091 (in_proj scale)
        "wall_clock_slack_absorbs": True,  # ~9% in_feat cost < 15% wall-clock slack; rc=8 verifies
        "source": ("tools/witness_memory_preflight.project_peak_rss_gib + "
                   "tac.canonical_equations.anisotropic_basis_two_regime_allocation_v1 "
                   "(freq_along_for_regime); memo basis_integration_v7_20260708.md"),
        "means_not_ends": "advisory memory sizing; pointer 0.19110 UNMOVED",
    }


def crucible_v7_polyak_start_provenance(epochs: int,
                                        muon_cap: int = _CRUCIBLE_V7_MUON_CAP,
                                        frac: float = 0.2) -> dict:
    """DERIVED-AT-CONFIG start_epoch for the v7 Polyak tail averager (v7.3 delta 2; NO bare literal).

    The R-7 ``PolyakFinisher`` default ``start_epoch=0`` is a documented footgun — it averages the
    WHOLE run rather than the finishing tail (r7_finishers_20260708.md residual #2). v7 SIZES it from
    the schedule via the R-7 helper ``polyak_finisher_window_provenance`` (the finisher law
    ``muon_finisher_schedule_warmstart_and_lr_anneal_v1``: tail window ~0.1-0.3× the finishing stage):

      finishing-stage window = ``epochs - muon_cap``  (the post-Muon constant-τ* turnpike phase);
      relative tail start     = ``window - round(frac·window)``  (helper output);
      ABSOLUTE start_epoch    = ``muon_cap + relative_start``.

    Sizing off the muon CAP (the LATEST possible Muon entry, a fail-safe backstop) makes the start the
    most CONSERVATIVE choice — always post-Muon, always inside the turnpike dwell even when the Muon
    event fires earlier (a shorter actual tail fraction, never a pre-Muon average). Pure / $0. Returns
    the R-7 helper's provenance dict augmented with the absolute epoch + the finishing-stage anchors.

    means != ends: advisory schedule sizing; the pointer 0.19110 is UNMOVED."""
    from tac.witness_control.polyak_finisher import polyak_finisher_window_provenance

    stage_window = int(epochs) - int(muon_cap)
    if stage_window <= 0:
        # DEGENERATE (a calibration/smoke run whose epochs <= the Muon backstop — Muon never fires, so
        # there is no finishing phase to average). Do NOT raise (RSS calibration reuses the REAL config
        # name with a tiny --calibrate-epochs to exercise the flag set). GENUINELY INERT (v7.3 round-2
        # MINOR-1 fix): the trainer loop is ``for ep in range(start, epochs+1)`` and PolyakTailAverager.
        # observe fires on ``ep >= start_epoch`` — so start_epoch=epochs is NOT inert (it observes ONCE at
        # the final epoch, count=1). Set start_epoch = epochs+1 (STRICTLY beyond the last loop epoch) so
        # observe can never fire => count stays 0 => byte-identical to an unarmed run. The gate still sees
        # a DERIVED manifest entry; the value is irrelevant for a run that never reaches the tail.
        return {
            "ladder_class": "derived_at_config",
            "equation_id": "muon_finisher_schedule_warmstart_and_lr_anneal_v1",
            "finishing_stage_window_epochs": stage_window,
            "finishing_stage_start_epoch": int(muon_cap),
            "tail_frac": float(frac),
            "polyak_window_epochs": 0,
            "polyak_start_epoch": int(epochs) + 1,  # > last loop epoch => never observes (genuinely inert)
            "polyak_relative_start_epoch": 0,
            "degenerate": True,
            "note": (f"DEGENERATE crucible_v7 polyak sizing: epochs ({epochs}) <= muon_cap ({muon_cap}) "
                     f"=> no finishing phase (calibration/smoke); start_epoch=epochs+1 (strictly beyond "
                     f"the final loop epoch) => Polyak observe never fires => genuinely inert (count 0)."),
        }
    prov = polyak_finisher_window_provenance(stage_window, frac=frac)
    window = int(prov["polyak_window_epochs"])
    # ABSOLUTE start over the INCLUSIVE trainer loop [start, epochs]: to average EXACTLY ``window`` epochs
    # (v7.3 round-2 MINOR-2 off-by-one fix) the half-open fencepost is ``start = epochs - window + 1``.
    # The prior ``muon_cap + (stage_window - window)`` = epochs - window observed window+1 epochs
    # (inclusive-final fencepost). Still post-Muon (inside the constant-τ* turnpike dwell).
    abs_start = int(epochs) - window + 1
    return {
        **prov,
        "finishing_stage_window_epochs": stage_window,
        "finishing_stage_start_epoch": int(muon_cap),  # muon backstop cap = the turnpike-dwell start
        "polyak_start_epoch": abs_start,               # ABSOLUTE (override the helper's relative)
        "polyak_relative_start_epoch": abs_start - int(muon_cap),  # = rel_start + 1 (fencepost fix)
        "note": ("v7 Polyak tail start = epochs - window + 1 (inclusive-final fencepost, averages EXACTLY "
                 "window = round(frac·finishing_window) epochs over the trainer's [start, epochs] loop); "
                 "finishing_window = epochs - muon_cap; always post-Muon inside the constant-τ* turnpike "
                 "dwell. " + str(prov.get("note", ""))),
    }


def crucible_v7_registered_off_levers() -> dict:
    """The DEFAULT-OFF v7 items kept as a TRACKED, REASONED, duty-to-measure QUEUE (v7.3 delta 5;
    the "off is a tracked queue, never a forgotten default" non-negotiable). These stay OFF at launch
    (they would perturb the sealed config / lack their inclusion evidence) but each carries a NAMED
    trigger the controller works down — never a silent grave. The council review + the #247 costate
    SENSE layer read this surface; a real launch records the corresponding activation-ledger state.

    Per the synthesis SYNTHESIS_INCL_symposium_20260708 §FINAL CLASSES + CRUX-ENGINEERING ADDENDUM.
    Pure / $0 (a documentation + duty-to-measure surface, not a launch knob). means != ends."""
    return {
        "micro_batch": {  # synthesis item 3 — v7.1-ARM; crux elevation REVOKED by measurement
            "default": "off", "state": "registered_duty_to_measure",
            "trigger": ("bounded n600 d_seg A/B ~day-1: v7's own first ~300 ep = arm A, a twin config "
                        "(micro-batch ON, same seed, ep0-300) = arm B — admission-gated on v7's measured "
                        "uncontended n600 RSS curve + the governor's 2-job envelope. NOT bit-identity "
                        "engineering: frozen_scorer_forward_batch_dependence_v1 MEASURED the batched "
                        "scorer forward is NOT bit-identical (GPU 0.006% argmax flips) so the crux "
                        "elevation is REVOKED — the A/B measures whether those flips are d_seg-neutral."),
        },
        "verdict_reclaim_330": {  # synthesis item 6 — v7.1-ARM (default-OFF at launch)
            "default": "off", "state": "registered_duty_to_measure",
            "trigger": ("verdict-correlated RSS ratchet in v7 telemetry (the urgency premise was stale "
                        "in run-1: RSS stable over 3 verdicts); crash-composition seam (S5-A3) owed "
                        "before load-bearing — #358 crash-resume tests + disjoint-tmpfile/killpg cover it."),
        },
        "adaptive_eps": {  # synthesis item 7 — REGISTERED
            "default": "off", "state": "registered_duty_to_measure",
            "trigger": ("eikonal re-entry signature in v7 telemetry (pre-built insurance #320; equation "
                        "adaptive_eps_cfl_edge_tracking_v1 ASSUMED_AWAITING_VERIFICATION). Structurally "
                        "absent at the sealed λ_eik 0.01 fixed (eikonal fell 49% in run-1, no re-entry)."),
        },
        "gpu_verdict": {  # synthesis item 10 — REGISTERED (stop-window probe armed)
            "default": "off", "state": "registered_duty_to_measure",
            "trigger": ("D1 GPU-vs-CPU verdict AGREEMENT probe at the run-1 governed stop (harness armed, "
                        "chunked-resumable, pre-registered thresholds). v7.1 scope: --async-verdict "
                        "CONFLICTS with --verdict-device gpu — the hybrid GPU-sensor/CPU-anchor cadence "
                        "needs that conflict resolved (a designed hybrid mode, not just a flag pair)."),
        },
        "fp16_cf_feats": {  # synthesis item 11 — REGISTERED
            "default": "off", "state": "registered_duty_to_measure",
            "trigger": ("re-waterfill gate if ever armed — NOT needed (37.26 GiB headroom post-basis, S1); "
                        "an on-the-fly per-pair fp16 cf-feature path would re-open only under memory pressure."),
        },
        "lane_carried_basis_regime": {  # SEAL v7.3 round-2 M1 counter-arm (structure)
            "default": "off", "state": "registered_duty_to_measure",
            "trigger": ("Road↔Lane per-class d_seg JITTER in v7 telemetry (M1 counter-arm; the two basis "
                        "regimes are MUTUALLY EXCLUSIVE — v7.3 commits to lane_offloaded). If the "
                        "lane_offloaded basis + analytic band UNDER-serve lane OR jitter the binding "
                        "Road↔Lane separatrix, switch to lane_carried: DirectionalBasisRebalance(regime="
                        "'lane_carried') → freq_along≈26 (the witness carries the ~25-cyc dash comb) AND "
                        "restore lane to --persistence-classes ('auto'/'1,3', keep the learned lane recall). "
                        "The freq_along≈26 √-optimum is ASSUMED_AWAITING_VERIFICATION (anisotropic_basis_"
                        "two_regime_allocation_v1); this A/B settles which regime lowers Road-boundary d_seg."),
        },
        "road_boundary_fallback": {  # SEAL v7.3 round-2 M2 Road-first fallback (structure)
            "default": "off", "state": "registered_duty_to_measure",
            "trigger": ("Road per-class d_seg is the PRIMARY run signal (LAUNCH_PACKAGE watch-list): if "
                        "Road flip-rate stays > 0.30 at ep200 (v6 was stuck 0.44→0.40 over 100 ep; ~68% of "
                        "flip mass), the single-basis Road bet UNDER-performed → fire a Road-FIRST fallback "
                        "rather than a cold restart. Mechanism candidates (structure D6/M2): (a) a Road↔"
                        "Undrivable (horizon) directional margin/boundary term orienting the basis to the "
                        "Road↔Undrivable + Road↔Lane tangent field; (b) a Menon logit-adjust Road-offset "
                        "AUDIT — the −1.37 Road offset DE-weights the 68%-binding class in the loss vs the "
                        "rare-class boosts; (c) seg-chroma-boundary engaged EARLIER on the Road boundary "
                        "(binding from ep~50). Pre-registered threshold gates the per-class decomposition "
                        "BEFORE any 'basis helps' claim (verdict-scope: the basis lever's Road effect)."),
        },
    }


def _build_crucible_v7(
    gt_cache_path,
    *,
    num_pairs: int,
    epochs: int,
    out_dir: str,
    code_matrix=None,
    byte_close_result=None,
):
    """Internal builder: construct the v7 ``TypedWitnessConfig`` + return the v6 baseline it
    reuses. Returns ``(typed, v6_cfg, v6_flags)``. Shared by :func:`derive_crucible_v7_config`
    and :func:`compile_crucible_v7_config` so v6 is derived ONCE."""
    from tac.witness_dsl.curriculum_dsl import (
        DirectionalBasisRebalance,
        LadderIslandHomotopy,
        PolyakFinisher,
        SegFormUnifyTau,
        TailCycles,
        persistence_classes_for_basis_regime,
    )
    from tac.local_acceleration.scorer_throughput_gate import derive_wall_clock_budget_days
    from tac.witness_dsl.typed_config import (
        Provenanced,
        ProvenanceClass,
        TypedAnneal,
        TypedLever,
        TypedRegularizer,
        TypedStage,
        TypedWitnessConfig,
    )

    _PC = ProvenanceClass
    # (1) the SEALED v6 substrate base (reuse-not-retype). derive_crucible_v6_config is pure CPU;
    # it does NOT load the GT cache (the path only identifies the clip).
    v6_cfg = derive_crucible_v6_config(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs,
        code_matrix=code_matrix, byte_close_result=byte_close_result)
    v6_flags = v6_cfg.to_trainer_flags("OUT")

    # (2) transform v6's sealed flags into the v7 typed ``base``: drop program-owned + deleted flags,
    # map a bare-boolean (flag, None) -> True (the DSL emitter's bare convention), keep every other
    # sealed value UNCHANGED. Then apply the v7 schedule-spine + event-cap deltas.
    base: dict = {}
    for flag, val in v6_flags:
        if flag in _CRUCIBLE_V7_PROGRAM_OWNED or flag in _CRUCIBLE_V7_DELETED:
            continue
        base[flag] = True if val is None else val
    base["--tau-anneal-shape"] = "geometric"                       # spine: cosine_hold -> geometric
    base["--lane-band-start-epoch"] = _CRUCIBLE_V7_LANE_BAND_CAP   # 350 -> 500 (fail-safe BACKSTOP cap)
    base["--seg-chroma-boundary-start-epoch"] = _CRUCIBLE_V7_CHROMA_CAP  # 300 -> 450 (fail-safe BACKSTOP cap)
    # (operator override 2026-07-08) co-emit the three SENSOR->START WIRING flags so each transition
    # FIRES ON ITS SENSOR (the start-epochs above are demoted to fail-safe backstop caps). These make
    # the governance events (role=fires) actually wired; the schedule-provenance gate classifies them
    # EVENT_TRIGGERED with the epoch backstops FAIL_SAFE_CAP.
    base["--muon-start-event"] = "powerlaw_meat"                   # muon <- tau-descent exhaustion (+REV-B)
    base["--lane-band-start-event"] = "lane_nucleus"              # lane-band <- lane critical nucleus
    base["--seg-chroma-boundary-start-event"] = "annulus_plateau"  # seg-chroma <- annulus_frac plateau
    # (S6-R4 self-paced τ-advance, operator 2026-07-08) convert the LAST clock-hardcoding (the anneal-
    # epochs denominator that clocks τ(t)) to the EVENT-driven geometric octave ladder. The octave count
    # / dwell caps DERIVE in the trainer from --anneal-epochs + --curriculum-min-stage-epochs (no bare
    # literals). NOTE — the self_paced_tau_advance_20260708 memo RECOMMENDS the council run the FIRST
    # unified-L_τ run in CLOCK mode (isolate the unify-τ variable; one continuation parameter at a time),
    # then flip to EVENT for run-2; flipping to 'clock' is a one-token change (byte-identical to the
    # incumbent anneal). Emitting 'event' here honors the operator's explicit conversion directive; the
    # council/seal makes the final launch-mode decision.
    base["--tau-advance-mode"] = "event"                          # τ octave-ladder <- per-band relaxation
    # (v7.3 round-2 BLOCKER fix, seal_v73_r2_deepmath) the CLOCK-frame hosc_beta_end=10.0 inherited from
    # v6 FREEZES β≈10.0 under the EVENT octave-fraction driver (β rides the same fraction that floors τ at
    # the turnpike) = the forbidden fixed-high-β saturation regime. Re-derive the EVENT-mode endpoint to
    # the intended FROZEN value = the control's β(726)≈3.177 (see _CRUCIBLE_V7_HOSC_BETA_END_EVENT). This
    # OVERRIDES the v6 base's --hosc-beta-end 10.0 (a v6/v7 CHANGED delta).
    base["--hosc-beta-end"] = _CRUCIBLE_V7_HOSC_BETA_END_EVENT
    # (v7.3 round-2 M1 fix, seal_v73_r2_structure) LANE-REGIME COHERENCE: the FEED-07a basis is set to
    # regime='lane_offloaded' (freq_along≈6 cartoon scale, CANNOT represent the ~25-cyc dash comb — lane
    # is carried by the FREE rule-118 analytic band, MEASURED lane d_seg 0.00087). Demanding lane-skeleton
    # RECALL from the frequency-starved learned render is unsatisfiable → wasted gradient + Road↔Lane
    # separatrix jitter (part of the binding Road residual). DERIVE the persistence-RECALL class targeting
    # from the active basis regime so the two agree: lane_offloaded → persistence targets movable ONLY
    # (lane offloaded to the band); lane_carried → 'auto' (keep lane). The LADDER island-amplify is ALREADY
    # per-class-λ self-gated (support flows to lane only while its measured cost is high — auto-de-
    # emphasizes when the band handles lane) so it needs no config gate here; the fixed-weight persistence
    # recall was the one regime-BLIND term. Counter-arm (lane_carried) registered duty-to-measure.
    base["--persistence-classes"] = persistence_classes_for_basis_regime(_CRUCIBLE_V7_BASIS_REGIME)
    # (F-3 structural coupling) dropping lane from the learned persistence recall (lane_offloaded)
    # is SOUND ONLY IF the FREE analytic lane band actually carries lane at byte-close. That band is a
    # SEPARATE flag (--lane-render-band, inherited bare-bool from the proven v6 base) with no structural
    # link to the regime — a future base that dropped it would leave lane with NEITHER learned recall
    # NOR analytic band => lane d_seg regression, discovered only at byte-close. Make the coupling
    # STRUCTURAL: when the active regime offloads lane, FAIL-LOUD AT COMPILE if the band flag is absent
    # from the emitted base. lane_carried keeps lane in the recall ("auto") so it needs no band. The
    # levers below never touch --lane-render-band, so asserting on `base` is equivalent to the argv.
    if _CRUCIBLE_V7_BASIS_REGIME == "lane_offloaded" and not base.get("--lane-render-band"):
        raise ValueError(
            "crucible_v7 lane-regime coherence gate (F-3): basis regime 'lane_offloaded' drops lane "
            "from the learned persistence recall (--persistence-classes="
            f"{base['--persistence-classes']!r}, lane excluded), so lane MUST be carried by the analytic "
            "band — but --lane-render-band is ABSENT from the emitted base. Emitting this config would "
            "starve lane of BOTH learned recall and the analytic band (lane d_seg regression at "
            "byte-close). Ensure the proven base carries --lane-render-band, or set the regime to "
            "'lane_carried'.")
    # (v7.3 round-2 R3 fix, seal_v73_r2_structure) turn ON per-group grad-clip: run-1 telemetry fired
    # gnorm_hijack 3× at ep1 (island_amplify ~20% of ep1 total loss, one gradient group scaling the whole
    # step down → seg-starvation risk during the very window the coarse partition + Road boundary forms).
    # --per-group-grad-clip re-clips each param group at --grad-clip (1.0) so a large early island/eikonal
    # term cannot starve the seg gradient. Requires --grad-clip>0 (base carries 1.0). Confound-fix (C4).
    base["--per-group-grad-clip"] = True
    # (v7.3 delta 3, synthesis item 4 ELEVATED by GPU cert) ARM the safe-compile hosc region. The
    # law safe_compile_hosc_device_bitidentity_v1 (crux-engineering GPU per-chip fingerprint bit-cert,
    # .omx/research/safe_compile_gpu_bitcert_20260708.md) MEASURED max|Δ|=0 for the compiled hosc
    # ``tanh(β·sin(x))`` region on M5-Max-GPU at REAL 384x512 coverage, β uniform over [1,10], N=5
    # cross-process — so the flip ADMITS at launch on this fingerprint (score-neutral, bit-identical).
    # It is DEVICE-CONDITIONAL: the SAME chip's CPU device fp-CONTRACTS by 1 ULP (5.96e-8) => REFUSE
    # there; mlx_device is 'gpu' below, and the trainer's resolve_enabled_regions keys off --mlx-device.
    # The launcher b2 gate is the RUNTIME AUTHORITY: it fail-closes off-fingerprint (any chip/os/mlx
    # change) so a stale manifest can never silently activate a compiled region. --safe-compile-manifest
    # is NOT emitted => the default .omx/state/mlx_safe_compile_manifest.json path resolves.
    base["--safe-compile-regions"] = "hosc_activation"

    # (3) the composable v7 levers (DSL Lever factories -> TypedLever; the DSL stays the emitter).
    def _typed_lever(lev) -> "TypedLever":
        return TypedLever(name=lev.name, overrides=dict(lev.overrides),
                          epochs_delta=lev.epochs_delta, notes=lev.notes)

    # (operator APPROVED 2026-07-08, seal v7 r1 R-1) enable the crucible's own Arm-A basis lever:
    # DirectionalBasisRebalance(lane_offloaded) — the DERIVED two-regime along-tangent rebalance
    # (equations leg anisotropic_basis_two_regime_allocation_v1). Triple-convergence justification:
    # measured 3.2x along-tangent deficit (L65) + the blind derivation's independent sqrt(32)~=6
    # minimum + the basis-before-capacity law (-48% directional MEASURED). It OVERRIDES v6's starved
    # basis (n_dir_freqs 2->4, freq_along 4->6; freq_across 32 re-emitted as float; --self-orient
    # already True in base => no-op override) via the DSL's base+lever merge (later wins, deduped).
    # MEMORY-WATERFILLED FIRST (:func:`crucible_v7_basis_allocation_provenance`,
    # basis_integration_v7_20260708.md): n_dir_freqs 2->4 grows in_feat 88->96 => cf_mx_cache
    # +3.93 GiB => projected peak 71.54 GiB, ADMITTED by both the 0.70 concurrent envelope (18.1 GiB
    # margin) and the 0.85 sole-workload envelope (37.3 GiB margin) — the lane_offloaded allocation
    # fits, so the DSL lever is preferred as-designed (the derived form) over a minimal along-only
    # rebalance. window=0 => no epoch delta; the ~9% in_feat step cost is within the 15% wall-clock
    # slack (rc=8 gate verifies at admission with the real bench). This is the lever's FIRST activation
    # (activation-ledger fires 'FEED_07a_directional_basis_rebalance' at launch; never-fired before).
    # (v7.3 delta 2, synthesis item 8 IN-v7 "IFF start_epoch SIZED") the R-7 Polyak/Ruppert tail
    # averager, armed at the TAIL turnpike window start — DERIVED-AT-CONFIG, NEVER the default 0 (the
    # R-7 residual footgun where start_epoch=0 averages the WHOLE run instead of the finishing tail).
    # The finishing (Muon+TAIL) phase spans [muon_cap, epochs]; sizing the tail from the muon CAP (the
    # LATEST possible Muon entry) yields the most CONSERVATIVE start — guaranteed post-Muon inside the
    # constant-τ* turnpike dwell even if the Muon event fires earlier. See crucible_v7_polyak_start_provenance.
    _polyak_prov = crucible_v7_polyak_start_provenance(int(epochs))
    _polyak_start = int(_polyak_prov["polyak_start_epoch"])
    levers = (
        _typed_lever(SegFormUnifyTau()),
        _typed_lever(TailCycles(cycles_max=_CRUCIBLE_V7_TAIL_CYCLES_MAX)),
        _typed_lever(LadderIslandHomotopy()),
        _typed_lever(DirectionalBasisRebalance(regime=_CRUCIBLE_V7_BASIS_REGIME)),
        _typed_lever(PolyakFinisher(start_epoch=_polyak_start)),
    )

    typed = TypedWitnessConfig(
        name="crucible_v7",
        out_dir=out_dir,
        gt_cache=str(v6_cfg.gt_cache),
        num_pairs=int(num_pairs),
        epochs=int(epochs),
        # DERIVED wall-clock budget (operator 2026-07-08 default-on): anchor min/ep x epochs x slack.
        # At epochs=3000 => ~8.31 days (7.23-day anchor projection x 1.15 slack) after the v7.4 round-3
        # SEAL DM-MINOR-1 re-anchor to the STARTUP-AMORTIZED 3000-ep cadence 3.47 min/ep, re-fit on the
        # WIDER ep25->125 window (was 3.39 on the narrow ep75->100 window that fell in a slow-adjacent-fast
        # trough; the full-window MEASURED steady slope is 3.4537, so amortized(3000) = 3.4537 + 51.25/3000
        # = 3.47 — strictly more conservative). NOT hand-picked; re-derives if epochs change or a lever (tau-advance/
        # micro-batch) changes the effective per-ep count => the budget tracks the anchor, the launcher
        # REFUSES a run slower than the honest amortized cadence by more than the slack (a TRUE ~15% gate).
        wall_clock_budget_days=Provenanced(
            value=round(derive_wall_clock_budget_days(int(epochs)), 3),
            provenance=_PC.DERIVED_AT_CONFIG, unit="days",
            source="scorer_throughput_gate.derive_wall_clock_budget_days"
                   "(anchor RUN1_MEASURED_MIN_PER_EP x epochs x WALL_CLOCK_SLACK_FACTOR)"),
        mlx_device="gpu",
        seed=0,
        purpose=(
            "T5 crucible v7 restart (requirement-V-native TypedWitnessConfig): witness-native "
            "continuous L_tau (seg-form-unify-tau) + geometric tau anneal (floor tau*=0.31) + "
            "TAIL_k + LADDER; three fixed start-epochs are TAGGED fail-safe caps (0 naked); pose "
            "block verbatim from v6. MEANS until a byte-closed n600 row < 0.19110."),
        temp=TypedAnneal(
            start=Provenanced(value=1.0, provenance=_PC.MEASURED_ANCHOR, unit="tau",
                              source="render-anneal start (proven base)"),
            end=Provenanced(value=0.31, provenance=_PC.MEASURED_ANCHOR, unit="tau",
                            source="mod32cap ep650-best tau=0.3098 (config-conditional); knee band "
                                   "corroborates (tau_end_knee_launch_v1)"),
        ),
        stages=(
            TypedStage(name="muon", start_epoch_flag="--muon-start-epoch",
                       start_epoch=_CRUCIBLE_V7_MUON_CAP),
        ),
        regularizers=(
            TypedRegularizer(flag="--eikonal-weight", weight=Provenanced(
                value=0.01, provenance=_PC.DERIVED_AT_CONFIG, unit="dimensionless",
                source="theta* lever stack (eikonal |grad phi|=1)")),
            TypedRegularizer(flag="--length-weight", weight=Provenanced(
                value=0.001, provenance=_PC.DERIVED_AT_CONFIG, unit="dimensionless",
                source="theta* lever stack (length INT ds)")),
        ),
        levers=levers,
        schedule_governance=_crucible_v7_schedule_governance(),
        base=base,
    )
    return typed, v6_cfg, v6_flags


def derive_crucible_v7_config(
    gt_cache_path,
    *,
    num_pairs: int,
    epochs: int = 3000,
    out_dir: str = _CRUCIBLE_V7_PLACEHOLDER_OUT,
    code_matrix=None,
    byte_close_result=None,
):
    """The T5 CRUCIBLE v7 restart config — the FIRST requirement-V-native launch config, authored
    AS a :class:`tac.witness_dsl.typed_config.TypedWitnessConfig` (NO hand argv, NO parallel dict
    assembly). See the module banner above for the five naked-violation resolutions.

    Returns the ``TypedWitnessConfig``; ``.to_program().compile_trainer_argv()`` is the DSL-emitted
    launch argv. Use :func:`compile_crucible_v7_config` for the full compiled artifact (argv +
    constants_manifest + dsl_program_manifest + the diff-vs-v6 baseline).

    means != ends: a MEANS. Only a byte-closed n600 exact row < 0.19110 moves the pointer.
    """
    typed, _v6_cfg, _v6_flags = _build_crucible_v7(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir,
        code_matrix=code_matrix, byte_close_result=byte_close_result)
    # fail-CLOSED at authoring time: refuse a config the DSL cannot validate (invented flag /
    # type-incompatible override / broken curriculum ordering) BEFORE it can reach a launcher.
    viol = typed.validate_program()
    if viol:
        raise ValueError(
            f"crucible_v7 DSL-authored-config gate: TypedWitnessConfig produced "
            f"{len(viol)} WitnessProgram.validate violation(s): {viol[:4]}")
    return typed


def compile_crucible_v7_config(
    gt_cache_path,
    *,
    num_pairs: int,
    epochs: int = 3000,
    out_dir: str = _CRUCIBLE_V7_PLACEHOLDER_OUT,
    code_matrix=None,
    byte_close_result=None,
) -> CrucibleV7Compiled:
    """Compile crucible_v7: the typed config -> argv + constants_manifest + dsl_program_manifest +
    the v6 diff baseline. The DSL ``WitnessProgram`` is the argv emitter (requirement V); the
    ``constants_manifest`` is inherited from v6 (v7's base reuses the SAME LawRef-resolved constants
    tau*=0.31 / beta-end 10.0 / lr-anneal 1000 / lr-hold 1.0, so v6's manifest describes them); the
    ``dsl_program_manifest`` is the :func:`build_launch_manifest` attestation the launcher rc=7 gate
    verifies. Fail-CLOSED via :func:`derive_crucible_v7_config`'s validate."""
    from tac.witness_dsl.typed_config import build_launch_manifest

    typed, v6_cfg, v6_flags = _build_crucible_v7(
        gt_cache_path, num_pairs=num_pairs, epochs=epochs, out_dir=out_dir,
        code_matrix=code_matrix, byte_close_result=byte_close_result)
    viol = typed.validate_program()
    if viol:
        raise ValueError(
            f"crucible_v7 DSL-authored-config gate: TypedWitnessConfig produced "
            f"{len(viol)} WitnessProgram.validate violation(s): {viol[:4]}")

    argv = typed.to_program().compile_trainer_argv()
    pairs = _crucible_v7_argv_pairs(argv)
    emitted_names = sorted({f for f, _ in pairs})
    dsl_manifest = build_launch_manifest(
        program_name="crucible_v7", emitted_flag_names=emitted_names,
        typed_config_hash=typed.typed_config_hash(), typed_validated=True)
    governance = {
        k: v.model_dump(mode="json", by_alias=True)
        for k, v in typed.schedule_governance.items()
    }
    # (S4-R2/S1-R1) req-T value-provenance rows for the sealed TAIL + LADDER-λ-gate constants so every
    # literal is auditable in the compiled manifest (no silent literals). Additive; does not touch argv.
    from tac.witness_control.tail_cycles import tail_constant_provenance
    from tac.witness_curriculum.ladder_homotopy import LADDER_LAMBDA_GATE_PROVENANCE
    tail_prov = {"tail_constants": tail_constant_provenance(),
                 "ladder_lambda_gates": {k: dict(v) for k, v in LADDER_LAMBDA_GATE_PROVENANCE.items()}}
    # (v7.3 delta 2) the Polyak tail start-epoch is a DERIVED-AT-CONFIG schedule-WHEN token, so it
    # MUST land in the constants_manifest as a LawRef entry — else the schedule-provenance gate flags
    # ``--polyak-finisher-start-epoch`` NAKED_PRIMARY_EPOCH (a bare positive epoch). It is NOT a
    # curriculum-stage cap/event — it is a value COMPILED from muon_finisher_schedule_warmstart_and_
    # lr_anneal_v1 (the finisher tail-window law), so DERIVED (constants_manifest) is the correct class.
    _pk = crucible_v7_polyak_start_provenance(int(epochs))
    constants_manifest = dict(getattr(v6_cfg, "constants_manifest", {}) or {})
    constants_manifest["polyak_finisher_start_epoch"] = {
        "value": int(_pk["polyak_start_epoch"]),
        "equation_id": _pk["equation_id"],  # muon_finisher_schedule_warmstart_and_lr_anneal_v1
        "ladder_class": "derived_at_config",
        "fallback_used": False,
        "inputs": {
            "epochs": int(epochs),
            "muon_cap": int(_CRUCIBLE_V7_MUON_CAP),
            "finishing_stage_window_epochs": int(_pk["finishing_stage_window_epochs"]),
            "tail_frac": float(_pk["tail_frac"]),
            "polyak_window_epochs": int(_pk["polyak_window_epochs"]),
        },
        "note": _pk["note"],
    }
    return CrucibleV7Compiled(
        typed=typed,
        argv=tuple(argv),
        emitted_pairs=tuple(pairs),
        constants_manifest=constants_manifest,
        dsl_program_manifest=dsl_manifest,
        schedule_governance=governance,
        v6_flags=tuple(v6_flags),
        tail_constant_provenance=tail_prov,
    )


@dataclass(frozen=True)
class CrucibleV7LaunchConfig:
    """The launcher-facing crucible_v7 cfg — the SINGLE object that satisfies the WHOLE duck-typed
    cfg protocol ``tools/launch_witness_run.py`` consumes (seal v7 r1 BLOCKER #1 + MAJOR #2).

    THE WIRING GAP THIS CLOSES: whichever object the launcher previously received carried only HALF
    the protocol. A bare :class:`~tac.witness_dsl.typed_config.TypedWitnessConfig` has the emit
    adapters (``to_command`` / ``to_trainer_flags`` / ``name``) but NO ``dsl_program_manifest`` /
    ``constants_manifest`` (b0.6 degrades to WARN — the v7 DSL-provenance gate is inert on the object
    that would launch) and its raw ``schedule_governance`` holds pydantic objects (b0.5
    ``classify_launch`` needs the DICT form). The :class:`CrucibleV7Compiled` has the manifests +
    governance-dict but NO emit adapters. This adapter composes BOTH: the emit surface delegates to
    the requirement-V-native ``typed`` config (the argv SoT, ~17x perf-env prefix intact); the three
    provenance manifests are the compiled artifact's (b0.5 ``manifest_keys`` +
    ``write_constants_manifest``; b0.6 ``verify_launch_manifest``; b0.5 ``classify_launch``).

    NOT exposed (by design): ``crucible_v6`` / ``fresh_seeded`` / ``sealed_205`` / ``all_levers``
    selector bools — ``config_family`` keys off ``name == 'crucible_v7'`` FIRST, and getattr-with-
    default handles the rest. ``--dsl-lever`` / ``--purpose`` CLI overrides are NOT wired for the
    typed path (v7 AUTHORS its lever set + purpose); passing them refuses at the dataclasses.replace
    seam rather than silently emitting an un-composed lever.
    """

    typed: object             # TypedWitnessConfig — the emit SoT (argv + perf-env prefix)
    constants_manifest: dict   # b0.5 manifest_keys + write_constants_manifest (v6-inherited LawRefs)
    dsl_program_manifest: dict  # b0.6 DSL-provenance attestation (verify_launch_manifest)
    schedule_governance: dict   # b0.5 classify_launch governance (DICT form, not pydantic objects)

    @property
    def name(self) -> str:
        return self.typed.name

    @property
    def purpose(self):
        return self.typed.purpose

    @property
    def wall_clock_budget_days(self):
        """The DERIVED wall-clock budget (Provenanced) the launcher's L45 wall-clock gate reads."""
        return self.typed.wall_clock_budget_days

    @property
    def epochs(self) -> int:
        """The config's epoch count (NEW-1: the launcher resolves omitted ``--epochs`` from HERE,
        so the sealed default cannot be silently trampled by a launcher-level hardcode; the
        wall-clock gate's projection also reads this)."""
        return int(self.typed.epochs)

    @property
    def dsl_levers(self) -> tuple[str, ...]:
        """The lever NAMES this config fires — the activation-ledger surface on a real launch
        (launcher step c.1). v7 pre-composes its lever set; the CLI cannot append to it."""
        return tuple(lv.name for lv in self.typed.levers)

    def to_command(self, out_dir=None, *, perf_env: bool = True) -> str:
        """The GO-ready launch command (delegates to the typed config; carries the ~17x perf-env
        prefix when ``perf_env`` — the exact silent-slow footgun the launcher b-perf gate catches)."""
        return self.typed.to_command(out_dir, perf_env=perf_env)

    def to_trainer_flags(self, out_dir=None):
        """The ``(flag, value)`` pairs the DSL emits (delegates to the typed config)."""
        return self.typed.to_trainer_flags(out_dir)


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
