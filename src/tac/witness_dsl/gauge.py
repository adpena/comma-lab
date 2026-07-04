"""GAUGE meta-layer — the bridge from gauge-INVARIANT math to the concrete program.

DAG FEED-ji (2026-06-29), operator insight "perhaps the gauge fits into the DSL and
new meta layer". This is the layer the DAG↔DSL↔equations triality was missing; it is
the OPERATIONALIZATION of the level-set/fiber QUOTIENT codec (task #155).

The 4-LAYER STACK (triality → quadrality):

    equations (E0-E12, gauge-INVARIANT math, the action S_tau)
        ↓
    GAUGE   (chart choice + cost + selection rule)   ← THIS MODULE (the bridge)
        ↓
    DSL     (tac.witness_dsl program, given a FIXED gauge: levers + θ* campaign)
        ↓
    DAG     (the trajectory / work-graph)

The insight: "how to choose between different-but-equivalent mathematical expressions"
= GAUGE-FIXING. The witness is a gauge-INVARIANT object (the scorer-equivalence class:
all witnesses with the same SegNet argmax + PoseNet output). Equivalent expressions are
GAUGES (charts / fiber representatives) with gauge-DEPENDENT cost. Coding the witness =
picking the cheapest legal fiber representative = the quotient codec.

What this module does (a clean $0 extension of ``tac.witness_dsl``, parallel to the
``Lever`` / ``with_lever`` campaign layer):
  (a) ENUMERATE the equivalent charts per witness component (the gauge Enums below).
  (b) Hold a probe-fed ``GaugeCostTable`` — each $0 gauge-probe MEASURES one cell
      (counted_bytes(MDL) + d_seg_through_R + conditioning), under HARD gates
      (rule-118 compliance + deterministic reproducibility). NO-FAKE: an un-measured
      cell is explicitly PENDING (measured=False, None numbers, provenance names the
      running probe) — never a fabricated number.
  (c) ``fix_gauge`` — hard-gates-first → drop PENDING (can't select what's unmeasured,
      return it so the caller knows a probe is needed) → minimize the component's
      S-contribution → deterministic synergy/composition tiebreak. Returns a GOSDT-style
      verdict whose ``.explain()`` reads back the rule chain (per CLAUDE.md "Preflight
      failure messages must cite the rule chain").
  (d) REJECT non-compliant / non-deterministic charts BY CONSTRUCTION (``GaugeChoice``
      raises a typed ``GaugeViolation``) — like the DSL's preserve/contain/authority.

means≠ends: this is observability + decision INFRASTRUCTURE. It is NOT a score claim.
The exact contest-CPU pointer (0.19110) is UNMOVED; only a byte-closed exact eval moves
it. The gauge layer makes the next byte-closed candidate cheaper to FIND and CERTIFY.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# contest rate denominator (bytes); S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/N
_S_RATE_DENOM = 37_545_489

# Context constant (NOT a gauge cell): the deterministic centerline/polynomial-base
# d_seg floor — what the FREE openpilot/SegNet-fit lane base achieves before any LEARNED
# residual. The trained-through-R witness STARTS here; the residual is the binding move.
# Provenance: a99f41f0 389f84f6f (deg 1→4 SATURATES; ~64% of lane d_seg recovered,
# 0.005885 drop-all → 0.00214).
POLY_BASE_DSEG_FLOOR = 0.00214


# ---------------------------------------------------------------------------
# Component labels + the per-component equivalent charts (the gauges)
# ---------------------------------------------------------------------------
class GaugeComponent(Enum):
    """The witness components, each a fiber with multiple equivalent charts."""

    WARP = "warp"
    CARRIER = "carrier"
    RESIDUAL = "residual"
    POSE = "pose"
    MOVABLES = "movables"
    GENERATION = "generation"
    TOPOLOGY_LOSS = "topology_loss"
    ISLAND_PROTECTION = "island_protection"
    # #224 consolidated-wire-in components (APPEND-ONLY).
    RENDER_AA = "render_aa"            # AA-SDF observation-map render (aa_sdf_observation_render)
    LANE_BAND = "lane_band"            # analytic-lane render-band (analytic_lane_render_band)
    HEAD_GEOMETRY = "head_geometry"    # softmax/ETF/additive-margin/Menon head (laguerre_logit_offset)
    # Wave-F #205 pose-plan (APPEND-ONLY; DESIGN-STAGE, loss-side, pending #224 trainer wire-in).
    POSE_TRAINING = "pose_training"    # HOW d_pose is trained in-loop (H1/D1/E1; NOT the STORE gauge)
    # Triality-sync levers (FEED-03n..03q, APPEND-ONLY; train-time / optimizer-side, 0 archive bytes;
    # NOT GaugeChoice STORE fields -- a loss/optimizer facet like TOPOLOGY_LOSS / POSE_TRAINING).
    MARGIN_SALIENCY = "margin_saliency"  # LEVER-4 multiplier chart (S_R reachability vs texture proxy)
    MUON_MOMENTUM = "muon_momentum"      # Muon finisher momentum-buffer init (warm-start vs cold)
    MUON_LR = "muon_lr"                  # Muon finisher LR schedule (cosine anneal vs flat)
    # Lane-dash root-cause levers (FEED-03t, APPEND-ONLY; train-time basis/loss/optimizer facets, 0
    # archive bytes; NOT GaugeChoice STORE fields — like MARGIN_SALIENCY / MUON_*).
    ALONG_TANGENT_FREQ = "along_tangent_freq"          # anisotropic-basis along-tangent Fourier bandwidth
    VECTOR_MARGIN_SALIENCY = "vector_margin_saliency"  # margin-saliency scalar->vector (asymmetry-t)
    CHROMA_BOUNDARY = "chroma_boundary"                # chroma as an argmax-boundary d_seg actuator
    FLICKER_TREATMENT = "flicker_treatment"            # predictable-replicate vs irreducible-downweight
    TRIPLE_JUNCTION_MARGIN = "triple_junction_margin"  # scalar top1-top2 vs multi-class simplex (WEAK)


class WarpGauge(Enum):
    """How the per-pair geometry is expressed (all derive from the stored pose)."""

    SCREW_TWIST = "screw_twist"                 # SE(3) screw / twist warp parameterization
    PER_CLASS_HOMOGRAPHY = "per_class_homography"  # depth-stratified per-class warp
    LEARNED = "learned"                         # a learned (counted) warp field


class CarrierGauge(Enum):
    """How the lane/edge structure is carried into the rendered frame."""

    SINGLE_SDF = "single_sdf"      # one signed-distance field (interpolation-exact through R)
    MSDF = "msdf"                  # multi-channel SDF (sharp-corner survival)
    HARD_BITMAP = "hard_bitmap"    # a hard 0/1 mask (Gibbs/aliasing through R)


class ResidualGauge(Enum):
    """How the irreducible (learned) lane residual is coded on top of the free base."""

    ALARD_LUPTON = "alard_lupton"          # classical difference-imaging deterministic coder
    DIRECT_LEARNED = "direct_learned"      # trained-through-R lane residual (the binding move)
    PERSISTENCE_EVENTS = "persistence_events"  # temporal-sparsity event coding
    CONDITIONAL_ON_LANE_PRIOR = "conditional_on_lane_prior"  # Wyner-Ziv X-E[X|Y], Y=free centerline


class PoseGauge(Enum):
    """How the 6-dim per-pair PoseNet target is stored (all hit the same target →
    the sqrt(10*d_pose) term is equal across charts; only bytes differ)."""

    SCALAR_STORE = "scalar_store"  # raw fp16 scalars (~5KB sidecar)
    RANGE_DELTA = "range_delta"    # range-coded temporal deltas (compressed)
    LOW_RANK = "low_rank"          # low-rank pose codec (task #140)
    # #224 APPEND-ONLY: render frame0 THROUGH the SE(3) ground-homography warp of the REAL
    # keyframe luma (seg-free f0 -> real-luma pose carrier; warp_real_luma_frame0). Composes with
    # the stored twist + a learnable per-pair residual; hits the same d_pose target, different bytes.
    # MEASURED CORRECTION (Wave-F unified-xi BUILD 2026-07-02,
    # ``lane_band_source_reparam_measured_resolution_v1``): xi is a PURE-POSE codec, NOT a lane
    # dual-use object -- the lane rate axis DECLINED xi (ego-predictive lane coding REFUTED; source
    # smoothing wins with ZERO xi), so xi is optimally calibrated for d_pose at ZERO lane cost
    # (n600 null d_pose 163.12 -> warp 1.367 = -99%). The prior "one xi, both axes" dual-use framing
    # is superseded FOR LANES: xi serves ONLY the pose axis.
    WARP_REAL_LUMA = "warp_real_luma"
    # #205 STORE-NOTHING-but-xi (Track B, 18927a1ae / keyframe_rate_minimization_builds_20260702).
    # SAME warp physics as WARP_REAL_LUMA but the frame0 SOURCE is the witness's OWN frame0 INR render
    # (generated FREE, rule-118) instead of a STORED real keyframe -> stores ONLY xi (+H) = ~0 marginal
    # bytes (the keyframe payload the byte-close measured at 697941 B ds4 / MBs native collapses to a
    # ~1 KB xi/H section). BYTE-CLOSE BIT-EXACT PROVEN (n6/t1, max_abs=0). d_pose is #205-gated (Track B
    # classmean proxy 4.97 pre-residual; the witness render is richer -> <= 4.97; the trained dxi
    # residual closes the offset). The A/B pose arm vs WARP_REAL_LUMA (--pose-carrier-source generated).
    STORE_NOTHING_XI = "store_nothing_xi"


class MovablesGauge(Enum):
    """How movable objects (cars) are represented."""

    STORE = "store"                # store the mask (zero d_seg error, costs bytes)
    WARP_PREDICT = "warp_predict"  # predict from the warp (cheaper bytes, small d_seg)


class GenerationGauge(Enum):
    """rule-118 axis: generic deterministic generation is FREE; learned content COUNTS."""

    DETERMINISTIC_FREE = "deterministic_free"  # generic generator in inflate.py (0 bytes)
    LEARNED_COUNTED = "learned_counted"        # learned weights in archive.zip (counted)


class TopologyLossGauge(Enum):
    """How the finest-scale (low-persistence) island topology is preserved in the seg loss.

    The #218 HEAD/margin-field facet. 0-byte (train-time only): the erasure-tail birth of the
    lane dashes (class-1) + small movables (class-3) the topology-blind per-pixel CE drops.
    Chart ↔ trainer flag: NONE=--persistence-loss-weight 0 (byte-identical baseline);
    CLDICE=clDice-only (w_recall 0); CLDICE_BETTI=clDice + persistence-weighted island recall.
    Impl: tac.boundary_math.persistence_topology_loss. $0 n600 signal (advisory): topology is
    ~111x more erasure-sensitive than CE; toy-descent island-recall gain +0.44."""

    NONE = "none"                    # no topology term (byte-identical to the pre-wire trainer)
    CLDICE = "cldice"                # soft-clDice connectivity (β0/β1) on self-detected thin classes
    CLDICE_BETTI = "cldice_betti"    # clDice + persistence-weighted island recall (the birth force)


class IslandProtectionGauge(Enum):
    """How the finest-scale islands (lane dashes cls-1 + small movables cls-3) are PROTECTED
    from spectral-bias-slow LATE discovery + the bulk-CE wash (FEED-lz; task #208).

    The 3-lever islands-protection stack (0-byte train-time; the seed is an ACCELERANT, NOT a
    shipped dense residual — the HPRC dense-sidecar rate-negative is respected). Chart ↔ trainer:
      NONE          = --no-seed-islands + --amplify-weight 0 (byte-identical baseline);
      AMPLIFY_ONLY  = the island-birth term (rides the SHARED realized margin _signed, the SAME
                      field the levelset trainer's LEVER-4 margin-saliency uses — NOT a 2nd saliency);
      SEED_CONTAIN  = EARLY-SEED (structured-init lane+movable) + CONTAINMENT (protected-pathway
                      grad projection) only;
      FULL          = EARLY-SEED + CONTAINMENT + AMPLIFICATION (all three).
    Impl: tac.boundary_math.island_protection. $0 n600 frozen-CPU-torch signal (advisory): LANE
    erased-recall 0.56 -> seeded 0.93 (+0.37 birth) -> wash 0.80 -> contained 0.95 (+0.16)."""

    NONE = "none"                    # no island protection (byte-identical baseline)
    AMPLIFY_ONLY = "amplify_only"    # the island-birth term only (rides LEVER-4 _signed)
    SEED_CONTAIN = "seed_contain"    # early-seed + containment only
    FULL = "full"                    # early-seed + containment + amplification


class RenderAAGauge(Enum):
    """How the witness render is sampled through R (aa_sdf_observation_render; #224/#220).

    POINT-sampling erases finest-scale lane structure through the contest R; FOOTPRINT-integrated
    (anti-aliased) rendering recovers it (MEASURED #1 rep lever, FEED-ly/-ma; recall-lift +0.374,
    AA floor 0.00086 < oracle-R need 0.00120). ~0-rate (decode-time deterministic; archive bytes
    UNCHANGED — the IPE attenuation + supersample grid are functions of the checkpoint cfg).
    Chart ↔ trainer flag: NONE=--render-aa none (byte-identical baseline); SUPERSAMPLE_2X/3X=
    --render-aa supersample --aa-supersample 2/3 (ground-truth footprint integration); IPE=
    --render-aa ipe (mip-NeRF cone attenuation of the curvelet basis, analytical ~0-compute)."""

    NONE = "none"                    # point-sample (byte-identical to the pre-#224 trainer)
    SUPERSAMPLE_2X = "supersample_2x"  # render at 2*grid + box-downsample (footprint integration)
    SUPERSAMPLE_3X = "supersample_3x"  # render at 3*grid + box-downsample
    IPE = "ipe"                      # mip-NeRF IPE cone attenuation of the curvelet columns


class LaneGauge(Enum):
    """How class-1 (lane) is authored (analytic_lane_render_band; #224/FEED-dv #203/#213/#215).

    The witness owns the smooth classes; LANE (the finest-scale erasure tail) is supplied by the
    analytic render-band = AA-SDF range-dependent coverage × dash gate × witness-uncertainty FP
    killer, composited PRE-R via the base render compose_fn. Chart ↔ trainer flag: NONE=witness
    renders lane itself (byte-identical baseline); BAND_RENDER_AUTHORITY=--lane-render-band (the
    class-1 render-time authority; net-negative d_seg realized by TRAINING WITH the band active).

    BYTE-CLOSE REALIZATION (Wave-F Stage-1; equation ``lane_band_camera_frame_rd_rate_v1``): the
    BAND_RENDER_AUTHORITY per-pair lane coeffs are serialized by the OPTIMAL LBND2 RD codec
    (L4 slots + L3 geometric-tolerance quantize + L2 temporal-delta + zigzag + brotli) = the DEFAULT
    in tools/levelset_byte_close_and_eval.py (opt-out ``--lane-band-naive`` on the BYTE-CLOSE TOOL,
    NOT the trainer). MEASURED n600: naive 156340 B (0.1041) -> RD 41526 B (0.02765) = 3.76x,
    decode-consistent. Shannon floor 26179 B => the residual is INFORMATION-bound.

    STAGE-2 SOURCE RE-PARAMETERIZATION (Wave-F unified-xi BUILD, MEASURED 2026-07-02): the SE(3)
    ego-factorization VIA EGO-WARP is REFUTED (``lane_band_source_reparam_measured_resolution_v1``:
    ego-motion-compensated predictive coding LBND3 = 1.04-1.34x WORSE) -- the camera-frame residual
    is per-frame fit JITTER + SLOT-SWAPs, NOT a coherent ego sweep. The source-reparam THESIS holds
    via a DIFFERENT mechanism: temporal SMOOTHING of the coeff trajectory = -42% (24149 B / 0.01608,
    BELOW the Shannon floor). The DERIVED OPTIMAL next lever is the CORRESPONDENCE-FIRST pipeline
    (``correspondence_first_lane_coding_optimal_pipeline_v1``): global min-cost-flow track assignment
    (LOSSLESS on geometry; kills the 44% slot-swap mass) -> per-track Kalman-RTS batch smoother [or
    RPCA] -> ll1-trend/TV/Potts edge-preserving denoise (lambda_i = margin-saliency d(d_seg)/d(coeff_i))
    -> the UNCHANGED LBND2 backend. It is a COMPRESS-TIME SOURCE PRE-TRANSFORM (ships as LBND2 bytes,
    ZERO new inflate code); DESIGN-STAGE pending the #234 ``tac.boundary_math.lane_track_and_smooth``
    build (DERIVED ~0.007-0.012, UNMEASURED byte-closed -- the #205 d_seg-through-R leg is the gate).
    OPTIONAL design-stage PRIOR: the openpilot coherent recurrent lane model (same comma rig) as the
    tracker's association affinity + RTS measurement-noise covariance
    (``openpilot_unified_physical_prior_both_scored_axes_v1``) -- a PRIOR/INIT/REGULARIZER, NEVER the
    fit target (the SegNet class-1 argmax mask through R is the sole authority)."""

    NONE = "none"                              # witness authors lane (byte-identical baseline)
    BAND_RENDER_AUTHORITY = "band_render_authority"  # --lane-render-band analytic class-1 authority


class HeadGeometryGauge(Enum):
    """How the 5-class SDF head geometry / per-class margin is shaped (laguerre_logit_offset; #218).

    Byte-free HEAD/margin-field facets. Chart ↔ trainer flag: SOFTMAX=--head softmax (default,
    byte-identical); ETF=--head etf (frozen simplex-ETF head, removes minority NORM COLLAPSE,
    regenerable => rate win); ADDITIVE_MARGIN=--head additive-margin --additive-margin <m>;
    MENON_LOGIT_ADJUST=--logit-adjust-per-class --logit-adjust-tau <t> (rare-class target boost).
    Composes with the realized margin-field hinge --margin-field-head-weight."""

    SOFTMAX = "softmax"                  # standard softmax head (byte-identical baseline)
    ETF = "etf"                          # frozen simplex-ETF head (neural-collapse optimal)
    ADDITIVE_MARGIN = "additive_margin"  # additive-margin softmax
    MENON_LOGIT_ADJUST = "menon_logit_adjust"  # Menon per-class logit adjustment (rare-class boost)


class PoseTrainingGauge(Enum):
    """How d_pose is lowered DURING the witness training loop (Wave-F #205 pose plan).

    DISTINCT from PoseGauge (which is how the 6-dim TARGET is STORED = bytes). This is the
    in-training STRATEGY. Survey verdict (equation ``pose_in_training_lever_survey_verdict_v1``):
    13 levers surveyed, NONE beats warp-real-luma alone -- all are COMPLEMENTS. The load-bearing
    quantitative claim: the seg-perp-pose gradient cosine is ~6e-5 => disjoint-frame freeze-and-add
    is EXACT and PCGrad (gradient surgery) is a FALSE FRIEND (a no-op when already orthogonal).

    DESIGN-STAGE: these charts describe the DSL design space; the H1/D1/E1 trainer flags are NOT
    yet wired (pending the #224/#205 trainer wire-in). Per never-invent-flags, the accessor
    ``pose_training_trainer_flags`` emits NOTHING for NONE and RAISES (fail-closed) for the
    design-stage charts, documenting the INTENDED arg -- it does NOT fabricate a flag. Ranked
    plan H1 > D1 > E1:
      NONE                    = pose rides warp-real-luma alone (the measured baseline);
      H1_OPENPILOT_XI_WARMSTART = seed the ego xi from the openpilot polynomial. MEASURED CORRECTION
                                (Wave-F unified-xi BUILD 2026-07-02): xi is PURE-POSE -- the lane rate
                                axis DECLINED xi (ego-predictive lane coding REFUTED; source smoothing
                                wins with ZERO xi), so H1 is a POSE-ONLY warm-start (co-#1 pose lever),
                                NOT the "dual-axis / lane-advection xi" originally claimed. The openpilot
                                LANE prior is a SEPARATE design-stage lever -- the coherent recurrent
                                lane model -> coherent SOURCE for the correspondence-first tracker
                                (LaneGauge; ``openpilot_unified_physical_prior_both_scored_axes_v1``),
                                a PRIOR/INIT/REGULARIZER, never the fit target;
      D1_DISJOINT_FREEZE_ADD  = pose on EVEN frames (f0->real-luma warp) + d_seg on ODD frames
                                (f1->witness) + trunk stop-grad => disjoint params, freeze-and-add
                                EXACT at cos~6e-5 (realizes the measured seg-perp-pose orthogonality);
      E1_KKT_POSE_TUBE        = a trust-region constraint keeping d_pose inside its tube while d_seg
                                descends (most GOAL-aligned; a KKT active-set, not a fixed weight)."""

    NONE = "none"                                  # warp-real-luma alone (measured baseline)
    H1_OPENPILOT_XI_WARMSTART = "h1_openpilot_xi_warmstart"  # pure-pose xi warm-start (co-#1; lane axis declined xi)
    D1_DISJOINT_FREEZE_ADD = "d1_disjoint_freeze_add"        # disjoint-frame freeze-and-add (+trunk stopgrad)
    E1_KKT_POSE_TUBE = "e1_kkt_pose_tube"                    # KKT pose-tube trust region


class MarginSaliencyGauge(Enum):
    """How LEVER-4 margin-saliency weights the fragility hinge (the through-R reachability chart;
    f99a3863a, DAG FEED-03n/03p, memory
    [[msal-uni-texture-proxy-inert-build-exact-sR-reachability-weight]]).

    The base LEVER-4 saliency is ``w = exp(-margin/tau)`` (the fragility factor); THIS component
    fixes the MULTIPLIER on top of it. Both charts presuppose an active ``--margin-saliency-weight
    > 0`` and are LEVELSET-trainer ``store_true`` flags. Chart <-> trainer flag:
      TEXTURE_PROXY          = ``--margin-saliency-uniward`` -- the UNIWARD texture down-weight
                               ``sal /= (1 + beta*tex)``. MEASURED INERT: texture is orthogonal to
                               through-R detector reachability (Pearson -0.033 vs S_R, top-5%
                               Jaccard 0.024 ~= chance 0.026; the full lever's 0.21 S_R-alignment is
                               ENTIRELY the w factor, texture adds +0.009 P) and mildly MISDIRECTS
                               (texprox vs |grad margin| -0.215 -> up-weights smooth interiors AWAY
                               from the boundary band).
      THROUGH_R_REACHABILITY = ``--margin-saliency-reachability`` -- REPLACES the texture path with
                               the exact through-R fragility-weighted margin-Jacobian
                               ``S_R = |d(sum_p w_p*margin_p)/dx|``, ``w = exp(-margin/tau)`` ->
                               ``sal = exp(-margin/tau) * S_R_norm``. MEASURED 3.0x concentrated on
                               the fragile margin band (S_R vs |grad margin| +0.272, vs margin
                               -0.323); theta-INDEPENDENT -> cached alongside ``margins`` in the
                               gt-cache (tools/precompute_sR_reachability.py, strictly cheaper than
                               the per-step tex recompute).
    HONEST scope: upgrades a SECONDARY multiplier (the primary ``w`` already carries the fragility
    alignment) -> expect a MODEST d_seg refinement. The NET n600 d_seg is #205-gated (the A/B is a
    #205 arm; advisory, MEANS)."""

    TEXTURE_PROXY = "texture_proxy"                    # --margin-saliency-uniward (MEASURED INERT)
    THROUGH_R_REACHABILITY = "through_r_reachability"  # --margin-saliency-reachability (S_R replaces it)


class MuonMomentumGauge(Enum):
    """How the Muon finisher's momentum buffer is initialized at the AdamW->Muon boundary
    (cba2e4375, DAG FEED-03o/03q, memory
    [[muon-deep-dive-keep-and-tune-finishing-stage-schedule-not-switch]]).

    Chart <-> trainer flag. These are BASE-trainer flags (``experiments/
    train_witness_realized_through_R_mlx.py`` -- the Muon-stage carrier per CLAUDE.md "Capstone
    theta* witness trainer"; the levelset entry point imports the base's Muon primitives). They are
    NOT on the levelset launch argparse yet (the launch-path wire-in of the two GAP flags is owed):
      COLD_MOMENTUM = fresh ``optim.Muon`` -> zero buffer. A cold first orthogonalized step is a
                      wild unit-norm direction from ONE noisy gradient -> boundary-pixel thrash ->
                      a MEASURED +0.000357 d_seg SPIKE at the transition (sister thetastar run,
                      ep750 saddle-gate). The byte-identical DEFAULT (emits NOTHING).
      WARM_START    = ``--muon-warm-start-momentum`` -- seed the Muon momentum ``v`` from the
                      OUTGOING AdamW first-moment ``m`` (both are gradient EMAs; Newton-Schulz
                      re-normalizes the update, so the transferred DIRECTION removes the cold-start
                      thrash / spike). Net d_seg improvement is #205-gated (advisory, MEANS)."""

    COLD_MOMENTUM = "cold_momentum"    # fresh zero buffer (byte-identical default; +0.000357 spike)
    WARM_START = "warm_start"          # --muon-warm-start-momentum (seed v from AdamW m)


class MuonLRGauge(Enum):
    """How the Muon finisher's learning rate is scheduled across the Muon span (cba2e4375, DAG
    FEED-03o/03q). BASE-trainer flags (the Muon-stage carrier; NOT the levelset launch argparse):
      FLAT_LR   = flat Muon LR (the byte-identical DEFAULT; ``--muon-lr-final-frac`` 1.0 -> no
                  decay). Muon's Newton-Schulz fixes update MAGNITUDE, so a flat LR cannot
                  self-reduce the step near a minimum -> plateaus / oversteps (river-valley Muon
                  2606.21514, Keller Jordan).
      ANNEAL_LR = ``--muon-lr-final-frac <f>`` (default 0.1) -- cosine-DECAY the Muon-group LR from
                  ``--muon-lr`` down to ``--muon-lr * f`` across the Muon span (composed AFTER the
                  optional re-warmup; only the Muon group -- the Adam tail self-adapts via its 2nd
                  moment). Net d_seg improvement is #205-gated (advisory, MEANS)."""

    FLAT_LR = "flat_lr"      # flat Muon LR (byte-identical default; frac>=1.0 = no decay)
    ANNEAL_LR = "anneal_lr"  # --muon-lr-final-frac <f> (cosine decay to floor)


class AlongTangentFrequencyGauge(Enum):
    """How much ALONG-TANGENT Fourier bandwidth the anisotropic self-orient/curvelet basis carries
    (FEED-03t, equation ``anisotropic_basis_along_tangent_frequency_deficit_v1`` + memory
    [[lane-dash-residual-root-is-along-tangent-freq-deficit-R-allpass]]).

    Lens-2 ROOT CAUSE of the lane-dash d_seg residual: the basis is SHARP ACROSS edges
    (freq_across -> 32,64 = Nyquist) but SMOOTH ALONG them (freq_along <= 8 cyc/unit); the lane
    dashes modulate at ~25 cyc/unit ALONG the tangent (10px @scorer-512) => a 3.2x DEFICIT => the
    dashes erase finest-first (error ~ 1/persistence). Both charts emit the REAL levelset-trainer
    value flag ``--n-dir-freqs`` (never-invent-flags; default 6, #205 runs 2 = the deficit). The #1
    ranked ep300+ lever (~0 archive bytes, R-safe <= Nyquist; net d_seg is a #205-class A/B).
      N_DIR_FREQS_2_DEFICIT = ``--n-dir-freqs 2`` (the #205-LIVE deficit config = the ROOT CAUSE)
      N_DIR_FREQS_4         = ``--n-dir-freqs 4`` (the lever; raises along-tangent bandwidth; pairs
                              optionally with ``--bank-n-scales 4->5`` as a 2nd frequency axis)."""

    N_DIR_FREQS_2_DEFICIT = "n_dir_freqs_2_deficit"
    N_DIR_FREQS_4 = "n_dir_freqs_4"


class VectorFieldMarginSaliencyGauge(Enum):
    """Whether margin-saliency #141 weights with a SCALAR magnitude or a VECTOR (the asymmetry-t
    sub-pixel skew; FEED-03t, equation ``separatrix_asymmetry_t_subpixel_boundary_localizer_v1``,
    probe a8afad40 GREEN).

    SCALAR_MAGNITUDE  = the existing scalar margin-saliency path (byte-identical baseline; emits
                        nothing beyond the campaign's ``--margin-saliency-weight``).
    VECTOR_T_SUBPIXEL = BUILT upgrade to the vector ``t = M_p/(M_p+M_q)`` (magnitude + boundary-normal
                        + sub-pixel flip-side). The t diagnostic is MEASURED GREEN (self-consistency
                        +0.560 disjoint); the TRAINING lever LANDED as LEVER-4b
                        ``--seg-subpix-boundary-weight`` (a sub-pixel boundary-placement loss on the
                        SHARED realized margin; default-off byte-identical, PROVEN A==B vs HEAD). The
                        accessor now EMITS the real flag; the net d_seg is a #205-class A/B (owed)."""

    SCALAR_MAGNITUDE = "scalar_magnitude"    # existing scalar saliency (byte-identical baseline)
    VECTOR_T_SUBPIXEL = "vector_t_subpixel"  # DESIGN-STAGE asymmetry-t vector saliency


class ChromaBoundaryGauge(Enum):
    """Whether the witness represents CHROMA (a PROVEN independent argmax-boundary d_seg actuator;
    SegNet reads RGB; FEED-03t, equation ``chroma_decides_lane_and_movable_at_annulus_v1``, probe
    a3e9f0bd GREEN; CLAUDE.md "Chroma is a d_seg lever" / operator "Chroma too" 2026-06-25).

    MEASURED GREEN (a3e9f0bd, n96 advisory, 100% L*-match to the frozen SegNet): removing chroma
    (constant-luma) flips 7.54% Lane->Road + 4.38% Movable->Undrivable, 93.4% of chroma-flips in the
    margin<1 annulus, proven independent of luma (margin-gradient energy 78.8% luma / 21.2% chroma).
    Chart <-> the REAL levelset-trainer BooleanOptional flag ``--chroma`` (default True; #205 runs ON):
      CHROMA_ACTIVE = ``--chroma`` default ON (byte-identical baseline; #205-live; the GREEN DOF)
      LUMA_ONLY     = ``--no-chroma`` (the ablation that MEASURED chroma's d_seg contribution GREEN).
    The deeper "route chroma CAPACITY INTO the boundary annulus" refinement (beyond the on/off flag)
    is now BUILT as LEVER-4c:
      ANNULUS_CHROMA_SHARPEN = ``--seg-chroma-boundary-weight <w>`` (BUILT, --help-verified, default-off
                        byte-identical PROVEN A==B vs HEAD). At the fragile margin annulus it supervises
                        the witness's OWN rendered chroma (rgb - BT.601-luma, LUMA-INVARIANT) toward the
                        GT chroma so the per-pixel RGB head paints the boundary chroma the near-per-class
                        CONSTANT palette can't. Reuses the SHARED realized-through-R render _f1 (no 2nd
                        render/SegNet); ORTHOGONAL to the geometry levers (a BOUNDARY sharpener). The
                        net d_seg is a #205-class A/B (owed)."""

    CHROMA_ACTIVE = "chroma_active"    # --chroma default ON (#205 baseline; the GREEN d_seg DOF)
    LUMA_ONLY = "luma_only"            # --no-chroma (ablation; MEASURED chroma removal HURTS d_seg)
    ANNULUS_CHROMA_SHARPEN = "annulus_chroma_sharpen"  # LEVER-4c --seg-chroma-boundary-weight (BUILT)


class FlickerTreatmentGauge(Enum):
    """How temporal FLICKER is treated (FEED-03t, equation
    ``independent_flicker_jitter_dseg_floor_smooth_optimal_v1``, probe a949ff63 -- still measuring).

    d_seg = q(1-r)+r(1-q) => for a GT minority-flicker q<0.5, an INDEPENDENT witness jitter r only
    RAISES d_seg (smooth r=0 optimal); lowering d_seg by replicating flicker REQUIRES temporal
    CORRELATION (predictability). The two treatment charts are DESIGN-STAGE loss-side levers (no
    trainer flag yet); NONE is the byte-identical baseline:
      NONE                   = no flicker treatment (byte-identical baseline; the #205-live behavior)
      DOWNWEIGHT_IRREDUCIBLE = down-weight the provably-irreducible sensor-noise flicker (smooth-optimal)
      REPLICATE_PREDICTABLE  = replicate+reward the PREDICTABLE (ego-xi/quant-phase) flicker (correlated)
    The design-stage charts fail-closed (never-invent-flags)."""

    NONE = "none"                                      # no flicker treatment (byte-identical baseline)
    DOWNWEIGHT_IRREDUCIBLE = "downweight_irreducible"  # DESIGN-STAGE: smooth-optimal down-weight
    REPLICATE_PREDICTABLE = "replicate_predictable"    # DESIGN-STAGE: correlated replicate+reward


class TripleJunctionMarginGauge(Enum):
    """Whether margin fragility uses the SCALAR top1-top2 or a MULTI-CLASS simplex (FEED-03t,
    equation ``scalar_top1_top2_margin_is_exact_distance_to_flip_v1``, probe a4c66f2f CLOSED WEAK).

    MEASURED n600 exact (a4c66f2f): gap13 >= gap12 at ALL 118M pixels (min 0.0) => the scalar
    top1-top2 margin IS the exact distance-to-flip; the multi-class simplex adds NO flip-ONSET DOF.
    Triple junctions are a flip-STRUCTURE DOF (0.027% of pixels, ~1-2% flip mass, 53.9%
    Road|Undriv|Movable car-corners) = NOT the lane tail (the lane residual is a codim-1 Road|Lane
    FACET = 41.4% of all facets the scalar margin already describes exactly).
      TOP1_TOP2_SCALAR   = the exact scalar distance-to-flip (byte-identical baseline; #205-live)
      MULTICLASS_SIMPLEX = DESIGN-STAGE + BANKED/WEAK (low-EV): the bench lever
                           w(p)=fragility(p)*(1+lambda*1[gap13<eps]) composes orthogonally but
                           targets car-corners -> BEHIND the facet levers. Accessor fail-closes."""

    TOP1_TOP2_SCALAR = "top1_top2_scalar"      # exact scalar distance-to-flip (baseline)
    MULTICLASS_SIMPLEX = "multiclass_simplex"  # DESIGN-STAGE + BANKED WEAK (car-corners, not lane)


# component → its chart Enum class (for fix_gauge iteration + full-stack sweeps)
COMPONENT_GAUGES: dict[GaugeComponent, type[Enum]] = {
    GaugeComponent.WARP: WarpGauge,
    GaugeComponent.CARRIER: CarrierGauge,
    GaugeComponent.RESIDUAL: ResidualGauge,
    GaugeComponent.POSE: PoseGauge,
    GaugeComponent.MOVABLES: MovablesGauge,
    GaugeComponent.GENERATION: GenerationGauge,
    GaugeComponent.TOPOLOGY_LOSS: TopologyLossGauge,
    GaugeComponent.ISLAND_PROTECTION: IslandProtectionGauge,
    # #224 consolidated-wire-in components (APPEND-ONLY).
    GaugeComponent.RENDER_AA: RenderAAGauge,
    GaugeComponent.LANE_BAND: LaneGauge,
    GaugeComponent.HEAD_GEOMETRY: HeadGeometryGauge,
    # Wave-F #205 pose-plan (APPEND-ONLY; DESIGN-STAGE, NOT in GaugeChoice — a loss-side
    # component like TOPOLOGY_LOSS / ISLAND_PROTECTION, not a fixable STORE chart).
    GaugeComponent.POSE_TRAINING: PoseTrainingGauge,
    # Triality-sync levers (FEED-03n..03q, APPEND-ONLY; train-time/optimizer-side, NOT GaugeChoice
    # STORE fields — real flags, unlike the DESIGN-STAGE POSE_TRAINING charts).
    GaugeComponent.MARGIN_SALIENCY: MarginSaliencyGauge,
    GaugeComponent.MUON_MOMENTUM: MuonMomentumGauge,
    GaugeComponent.MUON_LR: MuonLRGauge,
    # Lane-dash root-cause levers (FEED-03t, APPEND-ONLY; train-time basis/loss/optimizer facets,
    # NOT GaugeChoice STORE fields — like MARGIN_SALIENCY / MUON_* / POSE_TRAINING).
    GaugeComponent.ALONG_TANGENT_FREQ: AlongTangentFrequencyGauge,
    GaugeComponent.VECTOR_MARGIN_SALIENCY: VectorFieldMarginSaliencyGauge,
    GaugeComponent.CHROMA_BOUNDARY: ChromaBoundaryGauge,
    GaugeComponent.FLICKER_TREATMENT: FlickerTreatmentGauge,
    GaugeComponent.TRIPLE_JUNCTION_MARGIN: TripleJunctionMarginGauge,
}


# #224 chart -> trainer argv flags (never-invent-flags; the exact levelset-trainer flag names).
# The DSL renders bools as ``--flag`` and valued flags as ``[flag, str(val)]``; NONE/SOFTMAX charts
# emit NOTHING (they ARE the byte-identical default). Mirrors the sister-module *_flags() pattern.
RENDER_AA_TRAINER_FLAGS: dict[RenderAAGauge, tuple[str, ...]] = {
    RenderAAGauge.NONE: (),
    RenderAAGauge.SUPERSAMPLE_2X: ("--render-aa", "supersample", "--aa-supersample", "2"),
    RenderAAGauge.SUPERSAMPLE_3X: ("--render-aa", "supersample", "--aa-supersample", "3"),
    RenderAAGauge.IPE: ("--render-aa", "ipe"),
}
LANE_BAND_TRAINER_FLAGS: dict[LaneGauge, tuple[str, ...]] = {
    LaneGauge.NONE: (),
    LaneGauge.BAND_RENDER_AUTHORITY: ("--lane-render-band",),
}
# AM-softmax margin default (CosFace-family m~0.35-0.5, adapted to the realized-margin-hinge target).
# The fixed ADDITIVE_MARGIN chart cannot carry a per-instance value, so this is the default the static
# map bakes in; head_geometry_trainer_flags(chart, additive_margin=...) threads a campaign override.
ADDITIVE_MARGIN_DEFAULT = 0.5

HEAD_GEOMETRY_TRAINER_FLAGS: dict[HeadGeometryGauge, tuple[str, ...]] = {
    HeadGeometryGauge.SOFTMAX: (),
    HeadGeometryGauge.ETF: ("--head", "etf"),
    # (fix) --head additive-margin ALONE silently no-ops: the trainer's --additive-margin defaults 0.0,
    # so the AM realized-margin-hinge target is 0 = a no-op head. Emit the margin value with it.
    HeadGeometryGauge.ADDITIVE_MARGIN: ("--head", "additive-margin",
                                        "--additive-margin", str(ADDITIVE_MARGIN_DEFAULT)),
    HeadGeometryGauge.MENON_LOGIT_ADJUST: ("--logit-adjust-per-class",),
}


def render_aa_trainer_flags(chart: RenderAAGauge) -> tuple[str, ...]:
    """The levelset-trainer argv flags for a RenderAAGauge chart (NONE => () byte-identical)."""
    return RENDER_AA_TRAINER_FLAGS[chart]


def lane_band_trainer_flags(chart: LaneGauge) -> tuple[str, ...]:
    """The levelset-trainer argv flags for a LaneGauge chart (NONE => () byte-identical)."""
    return LANE_BAND_TRAINER_FLAGS[chart]


def head_geometry_trainer_flags(chart: HeadGeometryGauge,
                                additive_margin: float | None = None) -> tuple[str, ...]:
    """The levelset-trainer argv flags for a HeadGeometryGauge chart (SOFTMAX => () byte-identical).

    ADDITIVE_MARGIN emits ``--head additive-margin --additive-margin <m>`` (never a silent no-op: the
    trainer's --additive-margin defaults 0.0). ``additive_margin`` (optional) overrides the baked
    ADDITIVE_MARGIN_DEFAULT for a campaign-specific margin; ignored for non-AM charts."""
    if chart is HeadGeometryGauge.ADDITIVE_MARGIN and additive_margin is not None:
        return ("--head", "additive-margin", "--additive-margin", str(float(additive_margin)))
    return HEAD_GEOMETRY_TRAINER_FLAGS[chart]


# ---------------------------------------------------------------------------
# Triality-sync levers (FEED-03n..03q): S_R reachability + Muon finishing-stage schedule.
# The margin-saliency charts are LEVELSET-trainer flags (self-orient LEVER-4 lives on the levelset
# entry point per CLAUDE.md); the Muon charts are BASE-trainer flags (the Muon-stage carrier). The
# two GAP flags were built in cba2e4375 on the BASE trainer, so the never-invent-flags validation of
# the muon charts is against ``BASE_TRAINER_REL`` (real), NOT the levelset argparse (owed wire-in).
# ---------------------------------------------------------------------------
LEVELSET_TRAINER_REL = "experiments/train_levelset_witness_realized_through_R_mlx.py"
BASE_TRAINER_REL = "experiments/train_witness_realized_through_R_mlx.py"

MARGIN_SALIENCY_TRAINER_FLAGS: dict[MarginSaliencyGauge, tuple[str, ...]] = {
    MarginSaliencyGauge.TEXTURE_PROXY: ("--margin-saliency-uniward",),
    MarginSaliencyGauge.THROUGH_R_REACHABILITY: ("--margin-saliency-reachability",),
}
MUON_MOMENTUM_TRAINER_FLAGS: dict[MuonMomentumGauge, tuple[str, ...]] = {
    MuonMomentumGauge.COLD_MOMENTUM: (),                          # byte-identical default
    MuonMomentumGauge.WARM_START: ("--muon-warm-start-momentum",),
}
# The cosine-decay floor fraction the ANNEAL_LR chart bakes in (the DAG-cited #205-arm config
# ``--muon-lr-final-frac 0.1`` -> muon_lr 0.002 -> 2e-4 across the Muon span). muon_lr_trainer_flags
# threads a campaign override (mirrors head_geometry_trainer_flags / ADDITIVE_MARGIN_DEFAULT).
MUON_LR_FINAL_FRAC_DEFAULT = 0.1


def margin_saliency_trainer_flags(chart: MarginSaliencyGauge) -> tuple[str, ...]:
    """The LEVELSET-trainer argv flags for a MarginSaliencyGauge chart (both are real store_true
    flags on ``train_levelset_witness_realized_through_R_mlx.py`` — presuppose ``--margin-saliency-
    weight > 0``). THROUGH_R_REACHABILITY REPLACES the (measured-inert) texture path when set."""
    return MARGIN_SALIENCY_TRAINER_FLAGS[chart]


def muon_momentum_trainer_flags(chart: MuonMomentumGauge) -> tuple[str, ...]:
    """The BASE-trainer argv for a MuonMomentumGauge chart. COLD_MOMENTUM => () (byte-identical
    default); WARM_START => ``--muon-warm-start-momentum`` (a real BASE-trainer BooleanOptionalAction
    flag — the Muon-stage carrier is the base trainer, NOT the levelset launch argparse)."""
    return MUON_MOMENTUM_TRAINER_FLAGS[chart]


def muon_lr_trainer_flags(chart: MuonLRGauge, final_frac: float | None = None) -> tuple[str, ...]:
    """The BASE-trainer argv for a MuonLRGauge chart. FLAT_LR => () (byte-identical default);
    ANNEAL_LR => ``--muon-lr-final-frac <f>`` (a real BASE-trainer float flag). ``final_frac``
    (optional) overrides MUON_LR_FINAL_FRAC_DEFAULT for a campaign-specific floor; a value >= 1.0 is
    REFUSED (that IS the no-decay default = the FLAT_LR chart, not an anneal)."""
    if chart is MuonLRGauge.FLAT_LR:
        return ()
    frac = MUON_LR_FINAL_FRAC_DEFAULT if final_frac is None else float(final_frac)
    if frac >= 1.0:
        raise ValueError(
            f"MuonLRGauge.ANNEAL_LR final_frac={frac} >= 1.0 is the no-decay default (== FLAT_LR); "
            "pass a fraction in (0, 1) OR select MuonLRGauge.FLAT_LR")
    return ("--muon-lr-final-frac", str(frac))


# Wave-F #205 pose-plan INTENDED (not-yet-wired) trainer args — documented, NEVER emitted, per
# never-invent-flags. When the #224/#205 wire-in lands with the real argparse flags, replace this
# doc map with a real POSE_TRAINING_TRAINER_FLAGS dict + drop the NotImplementedError branch.
POSE_TRAINING_INTENDED_ARGS: dict[PoseTrainingGauge, str] = {
    PoseTrainingGauge.NONE: "",  # warp-real-luma alone; emits nothing (baseline)
    PoseTrainingGauge.H1_OPENPILOT_XI_WARMSTART: "--pose-xi-warmstart openpilot (INTENDED; not wired)",
    PoseTrainingGauge.D1_DISJOINT_FREEZE_ADD: "--pose-disjoint-frame --trunk-stopgrad (INTENDED; not wired)",
    PoseTrainingGauge.E1_KKT_POSE_TUBE: "--pose-kkt-tube <d_pose_cap> (INTENDED; not wired)",
}


def pose_training_trainer_flags(chart: PoseTrainingGauge) -> tuple[str, ...]:
    """The levelset-trainer argv for a PoseTrainingGauge chart. NONE => () (warp-real-luma baseline).

    DESIGN-STAGE fail-closed (never-invent-flags): the H1/D1/E1 trainer flags are NOT yet wired, so
    this RAISES NotImplementedError naming the INTENDED arg rather than fabricating a flag (mirrors
    the levelset trainer's ``--pose-carrier`` NotImplementedError). Cross-ref equation
    ``pose_in_training_lever_survey_verdict_v1`` + DAG FEED pose-survey."""
    if chart is PoseTrainingGauge.NONE:
        return ()
    raise NotImplementedError(
        f"PoseTrainingGauge.{chart.name} is DESIGN-STAGE (#205 pose-plan, pending #224 trainer "
        f"wire-in); intended arg: {POSE_TRAINING_INTENDED_ARGS[chart]}. never-invent-flags: emit "
        "nothing until the real argparse flag lands."
    )


# ---------------------------------------------------------------------------
# Lane-dash root-cause levers (FEED-03t): along-tangent freq + chroma are REAL levelset-trainer
# flags; vector-t / flicker / multiclass-simplex are DESIGN-STAGE (fail-closed, never-invent-flags,
# mirroring pose_training_trainer_flags). All are 0-archive-byte train-time facets.
# ---------------------------------------------------------------------------
ALONG_TANGENT_FREQ_TRAINER_FLAGS: dict[AlongTangentFrequencyGauge, tuple[str, ...]] = {
    AlongTangentFrequencyGauge.N_DIR_FREQS_2_DEFICIT: ("--n-dir-freqs", "2"),
    AlongTangentFrequencyGauge.N_DIR_FREQS_4: ("--n-dir-freqs", "4"),
}
CHROMA_BOUNDARY_TRAINER_FLAGS: dict[ChromaBoundaryGauge, tuple[str, ...]] = {
    ChromaBoundaryGauge.CHROMA_ACTIVE: (),            # --chroma default ON = byte-identical baseline
    ChromaBoundaryGauge.LUMA_ONLY: ("--no-chroma",),
    # LEVER-4c annulus chroma-sharpening BUILT (probe a3e9f0bd GREEN; landed in the levelset trainer,
    # --help-verified, default-off byte-identical PROVEN A==B vs HEAD). The baked weight is a
    # REPRESENTATIVE scale-appropriate A/B starter: the chroma-match term is in [0,255]^2 units (measured
    # ~10^3 at init vs CE ~243) so a SMALL weight balances it; the A/B tunes weight +
    # --seg-chroma-boundary-margin-band. The gauge selects the STRUCTURE (annulus chroma-sharpen ON).
    ChromaBoundaryGauge.ANNULUS_CHROMA_SHARPEN: ("--seg-chroma-boundary-weight", "0.05"),
}
# LEVER-4b sub-pixel boundary-placement `t` BUILT (probe a8afad40 GREEN; landed in the levelset
# trainer, --help-verified, default-off byte-identical PROVEN A==B vs HEAD). SCALAR_MAGNITUDE => ()
# (the existing scalar #141 path = baseline). VECTOR_T_SUBPIXEL => the REAL weighted flag at a
# REPRESENTATIVE starter weight (the A/B tunes the weight + --seg-subpix-boundary-v-band; the gauge
# selects the STRUCTURE = vector-t on, mirroring ALONG_TANGENT_FREQ baking --n-dir-freqs 4).
VECTOR_MARGIN_SALIENCY_TRAINER_FLAGS: dict[VectorFieldMarginSaliencyGauge, tuple[str, ...]] = {
    VectorFieldMarginSaliencyGauge.SCALAR_MAGNITUDE: (),   # existing scalar saliency = baseline
    VectorFieldMarginSaliencyGauge.VECTOR_T_SUBPIXEL: ("--seg-subpix-boundary-weight", "5.0"),
}

# DESIGN-STAGE INTENDED (not-yet-wired) trainer args — documented, NEVER emitted, per
# never-invent-flags (mirrors POSE_TRAINING_INTENDED_ARGS). Replace with a real *_TRAINER_FLAGS map
# + drop the NotImplementedError branch when the loss/head lever actually lands.
VECTOR_MARGIN_SALIENCY_INTENDED_ARGS: dict[VectorFieldMarginSaliencyGauge, str] = {
    VectorFieldMarginSaliencyGauge.SCALAR_MAGNITUDE: "",  # existing scalar path; emits nothing
    VectorFieldMarginSaliencyGauge.VECTOR_T_SUBPIXEL:
        "--seg-subpix-boundary-weight (WIRED as LEVER-4b; see VECTOR_MARGIN_SALIENCY_TRAINER_FLAGS)",
}
FLICKER_TREATMENT_INTENDED_ARGS: dict[FlickerTreatmentGauge, str] = {
    FlickerTreatmentGauge.NONE: "",  # no flicker treatment; emits nothing (baseline)
    FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE:
        "--flicker-downweight-irreducible (INTENDED; smooth-optimal; not wired)",
    FlickerTreatmentGauge.REPLICATE_PREDICTABLE:
        "--flicker-replicate-predictable (INTENDED; correlated reward; not wired)",
}
TRIPLE_JUNCTION_MARGIN_INTENDED_ARGS: dict[TripleJunctionMarginGauge, str] = {
    TripleJunctionMarginGauge.TOP1_TOP2_SCALAR: "",  # exact scalar margin; emits nothing (baseline)
    TripleJunctionMarginGauge.MULTICLASS_SIMPLEX:
        "--margin-simplex-triple-junction (INTENDED + BANKED/WEAK low-EV; not wired)",
}


def along_tangent_freq_trainer_flags(chart: AlongTangentFrequencyGauge) -> tuple[str, ...]:
    """The levelset-trainer argv for an AlongTangentFrequencyGauge chart. Both charts emit the REAL
    value flag ``--n-dir-freqs <n>`` (2 = the #205-live deficit, 4 = the lever). Optionally pair the
    lever with ``--bank-n-scales 4->5`` as a 2nd along-tangent frequency axis (not baked here)."""
    return ALONG_TANGENT_FREQ_TRAINER_FLAGS[chart]


def chroma_boundary_trainer_flags(chart: ChromaBoundaryGauge) -> tuple[str, ...]:
    """The levelset-trainer argv for a ChromaBoundaryGauge chart. CHROMA_ACTIVE => () (``--chroma``
    default ON = byte-identical #205 baseline); LUMA_ONLY => ``--no-chroma`` (the GREEN-measured
    ablation); ANNULUS_CHROMA_SHARPEN => the REAL BUILT LEVER-4c flag ``--seg-chroma-boundary-weight <w>``
    (probe a3e9f0bd GREEN; landed + --help-verified; default-off byte-identical PROVEN A==B vs HEAD). The
    baked weight is a REPRESENTATIVE scale-appropriate A/B starter (the A/B tunes weight +
    --seg-chroma-boundary-margin-band); the net d_seg is a #205-class A/B (owed)."""
    return CHROMA_BOUNDARY_TRAINER_FLAGS[chart]


def vector_margin_saliency_trainer_flags(chart: VectorFieldMarginSaliencyGauge) -> tuple[str, ...]:
    """The levelset-trainer argv for a VectorFieldMarginSaliencyGauge chart. SCALAR_MAGNITUDE => ()
    (the existing scalar saliency path = baseline). VECTOR_T_SUBPIXEL => the REAL BUILT LEVER-4b flag
    ``--seg-subpix-boundary-weight <w>`` (probe a8afad40 GREEN; landed + --help-verified; default-off
    byte-identical PROVEN A==B vs HEAD). The baked weight is a REPRESENTATIVE A/B starter (the A/B
    tunes weight + --seg-subpix-boundary-v-band); the net d_seg is a #205-class A/B (owed)."""
    return VECTOR_MARGIN_SALIENCY_TRAINER_FLAGS[chart]


def flicker_treatment_trainer_flags(chart: FlickerTreatmentGauge) -> tuple[str, ...]:
    """The levelset-trainer argv for a FlickerTreatmentGauge chart. NONE => () (no treatment =
    baseline). DOWNWEIGHT_IRREDUCIBLE / REPLICATE_PREDICTABLE are DESIGN-STAGE -> RAISE
    NotImplementedError naming the intended arg (never-invent-flags; probe a949ff63 measuring)."""
    if chart is FlickerTreatmentGauge.NONE:
        return ()
    raise NotImplementedError(
        f"FlickerTreatmentGauge.{chart.name} is DESIGN-STAGE (flicker decomposition probe a949ff63); "
        f"intended arg: {FLICKER_TREATMENT_INTENDED_ARGS[chart]}. never-invent-flags: emit nothing."
    )


def triple_junction_margin_trainer_flags(chart: TripleJunctionMarginGauge) -> tuple[str, ...]:
    """The levelset-trainer argv for a TripleJunctionMarginGauge chart. TOP1_TOP2_SCALAR => () (the
    exact scalar distance-to-flip = baseline). MULTICLASS_SIMPLEX is DESIGN-STAGE + BANKED/WEAK ->
    RAISES NotImplementedError naming the intended arg (never-invent-flags; a4c66f2f closed WEAK ->
    low-EV bench item, behind the facet levers)."""
    if chart is TripleJunctionMarginGauge.TOP1_TOP2_SCALAR:
        return ()
    raise NotImplementedError(
        f"TripleJunctionMarginGauge.{chart.name} is DESIGN-STAGE + BANKED/WEAK (simplex probe "
        f"a4c66f2f closed WEAK: scalar margin is already the exact distance-to-flip); intended arg: "
        f"{TRIPLE_JUNCTION_MARGIN_INTENDED_ARGS[chart]}. never-invent-flags: emit nothing until wired."
    )


def component_of(chart: Enum) -> GaugeComponent:
    """Return the GaugeComponent a chart enum member belongs to."""
    for comp, cls in COMPONENT_GAUGES.items():
        if isinstance(chart, cls):
            return comp
    raise TypeError(f"{chart!r} is not a known gauge chart")


# ---------------------------------------------------------------------------
# GaugeCost — one probe-fed cell of the cost ledger
# ---------------------------------------------------------------------------
class GaugeCostError(ValueError):
    """A GaugeCost cell violates the NO-FAKE / provenance contract."""


@dataclass(frozen=True)
class GaugeCost:
    """The gauge-DEPENDENT cost of one chart.

    ``counted_bytes`` (MDL rate), ``d_seg_through_R`` (the realized argmax cost), and
    ``conditioning`` are ``None`` when not-yet-applicable / not-yet-measured. ``compliant``
    (rule-118) and ``deterministic`` (bit-identical decode) are HARD gates. ``measured`` is
    True only for a LANDED probe; False == PENDING (a probe is running / the GPU move is
    pending). ``provenance`` is REQUIRED (non-empty) — it cites the FEED / probe / commit.

    NO-FAKE invariant (enforced in ``__post_init__``): a PENDING cell (measured=False) MUST
    carry None numeric fields — it may never smuggle a fabricated number.
    """

    counted_bytes: int | None
    d_seg_through_R: float | None
    conditioning: float | None
    compliant: bool
    deterministic: bool
    measured: bool
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise GaugeCostError(
                "GaugeCost.provenance must be a non-empty string (NO-FAKE provenance contract)")
        if not self.measured:
            fabricated = [
                name for name, val in (
                    ("counted_bytes", self.counted_bytes),
                    ("d_seg_through_R", self.d_seg_through_R),
                    ("conditioning", self.conditioning),
                ) if val is not None
            ]
            if fabricated:
                raise GaugeCostError(
                    f"NO-FAKE: a PENDING (measured=False) GaugeCost cell must have None numeric "
                    f"fields; got non-None {fabricated} with provenance {self.provenance!r}")

    def passes_hard_gates(self) -> bool:
        """rule-118 compliance AND deterministic reproducibility (both HARD)."""
        return bool(self.compliant and self.deterministic)

    def s_contribution(self) -> float | None:
        """The chart's contribution to S = 100*d_seg + 25*bytes/N (the byte/d_seg terms
        present). Returns None when there is no cost data to rank (the chart is measured
        but its rankable cost is not yet known).

        Pose charts are byte-only (d_seg_through_R is None) — the sqrt(10*d_pose) term is
        equal across pose charts (same stored target) so it cancels in the comparison; the
        byte term decides. This method handles that gracefully (only the byte term sums).
        """
        if self.d_seg_through_R is None and self.counted_bytes is None:
            return None
        s = 0.0
        if self.d_seg_through_R is not None:
            s += 100.0 * self.d_seg_through_R
        if self.counted_bytes is not None:
            s += 25.0 * self.counted_bytes / _S_RATE_DENOM
        return s


# ---------------------------------------------------------------------------
# GaugeCostTable — the probe-fed cost ledger
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaugeCostTable:
    """A queryable (chart → GaugeCost) ledger. The running $0 gauge-probes fill cells;
    ``fix_gauge`` consumes it. Immutable-update via ``with_cell`` (for tests / what-ifs)."""

    cells: dict  # Enum -> GaugeCost

    def lookup(self, chart: Enum) -> GaugeCost | None:
        return self.cells.get(chart)

    def with_cell(self, chart: Enum, cost: GaugeCost) -> "GaugeCostTable":
        new = dict(self.cells)
        new[chart] = cost
        return GaugeCostTable(new)

    def charts_for(self, component: GaugeComponent) -> list[Enum]:
        return list(COMPONENT_GAUGES[component])


def default_cost_table() -> GaugeCostTable:
    """The canonical cost ledger SEEDED from our MEASURED gauge-probes (each cell cites its
    FEED/commit in provenance). Un-measured charts are PENDING (measured=False, None numbers,
    provenance names the running probe / pending GPU move) — NO fabricated numbers."""
    cells: dict[Enum, GaugeCost] = {
        # --- WARP --------------------------------------------------------------
        # per-class depth-stratified homography: grok-test CONFIRMED pose↔d_seg coupling
        # (Road 0.0231→0.0196) but PRE-R/advisory → the through-R d_seg is None (F1 ae868999
        # investigating through R). The warp itself is ~0-byte (derived from the stored pose
        # + a cheap static per-class warp-type mask).
        # screw/twist MEASURED-WIN (a513372a FEED-jj): ~0 marginal bytes (reuses the stored
        # 6-DOF pose sidecar + O(10) static descriptor params). The d_seg numbers are PRE-R
        # ADVISORY lower bounds → d_seg_through_R stays None (through-R probe spawned); the
        # ~0-byte cost is what fix_gauge ranks on (it dominates the 6,600-param homography).
        WarpGauge.SCREW_TWIST: GaugeCost(
            counted_bytes=0, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-jk (screw-warp-through-R) tools/measure_screw_warp_through_R.py: "
                       "REAL through-R d_seg MEASURED (frozen CPU-torch SegNet on warped GT RGB "
                       "through the contest R; NO-FAKE self-check SegNet(gt_f1)==lstars exact). "
                       "BULK-warp term (Road+sky+hood, what this WARP gauge owns) d_seg≈0.00477(n96)/"
                       "0.00512(n200); full screw through-R TOTAL≈0.0076(n96)/0.0080(n200) of which "
                       "LANE≈0.0023 (flip≈0.39, binding wall → CARRIER/RESIDUAL) + MOVABLE≈0.00055 "
                       "(→ MOVABLES). Warp HELPS Road ~8% over naive-persist but is LANE-unexplainable; "
                       "bulk does NOT fit the 1.23e-3 budget (warp-of-PREVIOUS-FRAME upper bound, "
                       "Road-inter-frame-jitter-dominated; clean stored-canonical-warp follow-up "
                       "untested). The numeric d_seg_through_R RANKING field stays None on PURPOSE: "
                       "fix_gauge ranks SCREW vs PER_CLASS_HOMOGRAPHY on S=100*d_seg+25*bytes/N, but "
                       "the two warps share the same GENERALIZABLE bulk d_seg (a513372a: the per-class "
                       "oracle's in-clip edge is 85% non-physical clip-overfit that does NOT "
                       "generalize), so the BYTE axis must decide (SCREW 0 vs 6600) — feeding only "
                       "SCREW's measured d_seg would mis-rank it worse purely for being measured. "
                       "Both warp charts stay d_seg=None until measured on a generalization-adjusted "
                       "axis; the measured numbers live here + in the FEED-jk memo + results JSON "
                       "(no signal loss). ~0 marginal bytes (reuses stored 6-DOF pose + O(10) static "
                       "descriptor); LHP/plane-homography rasterizer FREE in inflate.py (rule-118). "
                       "prior PRE-R label lower bound 0.01074(n96) was a513372a FEED-jj 0bbc147b8 "
                       "(across-pair; the cleaner WITHIN-PAIR pre-R label total is 0.00744(n96))"),
        WarpGauge.PER_CLASS_HOMOGRAPHY: GaugeCost(
            counted_bytes=6600, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="grok-test 2f83e0b9e/FEED-ja + a513372a FEED-jj: depth-stratified per-class "
                       "warp, pose↔d_seg coupling CONFIRMED (Road 0.0231→0.0196). PRE-R ADVISORY "
                       "d_seg≈0.00950(n96)/0.01372(n200) BUT ≈6,600 per-pair params AND ~85% of the "
                       "apparent d_seg edge is NON-PHYSICAL clip-overfit (does not generalize) → "
                       "LOSES to SCREW_TWIST on S-contribution (byte cost dwarfs the tiny edge); "
                       "through-R d_seg pending"),
        WarpGauge.LEARNED: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="learned (counted) warp field; UNMEASURED — dominated by the free "
                       "per-class-homography warp unless R-survival forces it"),

        # --- CARRIER (lane @ render-192) ---------------------------------------
        CarrierGauge.SINGLE_SDF: GaugeCost(
            counted_bytes=None, d_seg_through_R=0.0319, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="F1 FEED-iw 0e44b4e8d: single SDF carrier, R is interpolation-exact "
                       "→ SDF survives (lane @ render-192 d_seg_through_R≈0.0319)"),
        CarrierGauge.HARD_BITMAP: GaugeCost(
            counted_bytes=None, d_seg_through_R=0.166, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="F1 (0e44b4e8d): hard 0/1 bitmap carrier — Gibbs/aliasing through R "
                       "(lane @ render-192 d_seg_through_R≈0.166, ~5.2x worse than SDF)"),
        CarrierGauge.MSDF: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="probe a1d5682964 running (multi-channel SDF; sharp-corner survival through R)"),

        # --- RESIDUAL (the binding axis) ---------------------------------------
        # CONDITIONAL_ON_LANE_PRIOR = the Wyner-Ziv head-start: code X-E[X|Y], Y=free
        # openpilot/SegNet-fit centerline base. The deterministic BASE is MEASURED (0.00214,
        # ~64% of lane d_seg recovered, coeffs 0.5-5KB counted). The LEARNED residual ON TOP
        # (target ≤1.23e-3) is PENDING-GPU → represented as base-measured + residual-pending
        # (the d_seg here is the BASE recovery, an UPPER bound the trained residual lowers).
        ResidualGauge.CONDITIONAL_ON_LANE_PRIOR: GaugeCost(
            counted_bytes=5000, d_seg_through_R=POLY_BASE_DSEG_FLOOR, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="a99f41f0 389f84f6f (centerline BASE measured: 0.005885→0.00214, "
                       "~64% lane d_seg recovered, coeffs 0.5-5KB counted, conservative 5KB) "
                       "+ a5b83c730 head-start pipeline; the LEARNED residual-on-top "
                       "(target ≤1.23e-3 through R) is PENDING-GPU (the binding move) — the "
                       "0.00214 here is the deterministic BASE (an upper bound the residual lowers)"),
        ResidualGauge.ALARD_LUPTON: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="Alard-Lupton difference-imaging deterministic coder a1da84c b052ab09d; "
                       "through-R d_seg for OUR lane residual UNMEASURED (poly-base floor 0.00214 "
                       "a99f41f0 389f84f6f is the deterministic-base context constant)"),
        ResidualGauge.DIRECT_LEARNED: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="GPU run pending (the binding move): trained-through-R lane residual "
                       "(target d_seg ≤1.23e-3); decode is bit-identical (deterministic), "
                       "weights COUNTED (rule-118)"),
        ResidualGauge.PERSISTENCE_EVENTS: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="persistence / event-coded residual (temporal sparsity); UNMEASURED"),

        # --- POSE (byte-only; same stored target → equal sqrt(10*d_pose)) ------
        PoseGauge.RANGE_DELTA: GaugeCost(
            counted_bytes=875, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="F4 095ed3e1a + a99f41f0 389f84f6f: range-coded temporal-delta pose "
                       "sidecar (≈474-875B; conservative 875B)"),
        PoseGauge.SCALAR_STORE: GaugeCost(
            counted_bytes=4800, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="F4: raw fp16 scalar pose sidecar (~5KB, 600×6×fp16 ≈ 4800B zlib)"),
        PoseGauge.LOW_RANK: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="low-rank pose codec (task #140; ~2.7x over scalar-store predicted, "
                       "NOT byte-closed) — UNMEASURED"),
        # #205 STORE-NOTHING carrier: a byte-close-MEASURED (BIT-EXACT) frame0-warp source. Unlike the
        # SIDECAR charts above (dead bytes the render never reads on a code/texture-pose witness), this
        # CARRIER actually warps frame0 -> it MOVES d_pose (Track B classmean proxy 4.97 pre-residual;
        # #205-gated after the trained dxi residual). counted_bytes = byte-optimal xi-only (600*12 B fp16;
        # H FREE via exp_se3); the store-nothing collapses the WARP_REAL_LUMA keyframe payload (697941 B
        # ds4 MEASURED) to this ~7 KB. d_seg_through_R None (frame0 is seg-free) so it does not rank on
        # d_seg; the byte term is the rankable cost.
        PoseGauge.STORE_NOTHING_XI: GaugeCost(
            counted_bytes=7200, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="Track B store-nothing-but-xi (18927a1ae; keyframe_rate_minimization_builds_"
                       "20260702) + byte-close BIT-EXACT n6/t1 (levelset_byte_close_and_eval "
                       "--pose-carrier-mode store_nothing; max_abs=0; section 697941B ds4 table -> 1049B "
                       "store_nothing). byte-optimal xi-only 600*12B fp16 = 7200B (H derived FREE). "
                       "CARRIER (moves d_pose), not a dead sidecar; d_pose #205-gated (<=4.97 pre-residual)"),

        # --- MOVABLES ----------------------------------------------------------
        MovablesGauge.STORE: GaugeCost(
            counted_bytes=2700, d_seg_through_R=0.0, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="F3 FEED-je 930b6d348: STORE the movable mask → d_seg≈0.0 "
                       "(≈900-2700B; conservative 2700B). Verdict: STORE not predict"),
        MovablesGauge.WARP_PREDICT: GaugeCost(
            counted_bytes=None, d_seg_through_R=0.00082, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="F3 (930b6d348): per-object rigid warp-predict floor d_seg≈0.00082 "
                       "(cheaper bytes but dominated by STORE on S-contribution)"),

        # --- GENERATION (rule-118 axis) ----------------------------------------
        GenerationGauge.DETERMINISTIC_FREE: GaugeCost(
            counted_bytes=0, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="rule-118 (README.md:118): generic deterministic generator runs in "
                       "inflate.py for FREE (untimed except the 30-min budget) → 0 counted bytes"),
        GenerationGauge.LEARNED_COUNTED: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=False, measured=False,
            provenance="rule-118: LEARNED/video-derived content is COUNTED in archive.zip; bytes "
                       "= the learned residual (UNMEASURED, sister of ResidualGauge.DIRECT_LEARNED). "
                       "deterministic flag depends on the trained-decode being certified bit-identical"),

        # --- RENDER_AA (#224/#220; ~0-rate decode-time observation model) --------------------
        # NONE = the byte-identical point-sample baseline. Its d_seg IS the witness's OWN through-R
        # floor (NOT an independent chart delta) -> UNRANKED here (bytes/d_seg None => s_contribution
        # None => fix_gauge lists it "unrankable", so the auto-selector cannot mis-pick it merely for
        # being byte-free). Still measured=True + compliant+deterministic so GaugeChoice.validate()
        # (the default field) passes.
        RenderAAGauge.NONE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="#224/#220 --render-aa none: byte-identical point-sample baseline (the "
                       "witness's own through-R floor; not a chart delta -> unranked in fix_gauge)"),
        RenderAAGauge.SUPERSAMPLE_2X: GaugeCost(
            counted_bytes=0, d_seg_through_R=0.00086, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-ly/-ma tools/levelset_gate_discriminators_n600.py [macOS-CPU advisory "
                       "NON-PROMOTABLE]: ss=2 supersample->box footprint integration recovers the "
                       "finest-scale lane structure through the contest R (AA floor d_seg~0.00086, "
                       "lane recall +0.374 vs point-sample); ~0-rate (decode-time deterministic, "
                       "archive bytes UNCHANGED). NOTE: composing with --self-orient is FAIL-CLOSED "
                       "at n600 (fine per-pair dir-feat cache OOM / on-demand EDT wall-clock; see the "
                       "levelset trainer supersample guard); use with --render-aa without self-orient "
                       "OR pair self-orient with --render-aa ipe / --lane-render-band"),
        RenderAAGauge.SUPERSAMPLE_3X: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="ss=3 supersample->box (finer footprint integration; 9x forward; 0-rate "
                       "decode-time); through-R d_seg UNMEASURED (pending) -- dominated-until-"
                       "measured by SUPERSAMPLE_2X"),
        RenderAAGauge.IPE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="mip-NeRF IPE cone attenuation of the curvelet columns (analytical ~0-compute "
                       "AA proxy, 0-rate; self-orient-COMPATIBLE, already wired in the trainer); "
                       "through-R d_seg floor UNMEASURED (pending) -- supersample->box is the authority"),

        # --- LANE_BAND (#224/FEED-dv #203/#213/#215; class-1 render-time authority, 0-byte) ----
        LaneGauge.NONE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="--lane-render-band OFF: the witness authors class-1 itself (byte-identical "
                       "baseline; the lane d_seg is the witness's own floor -> unranked in fix_gauge)"),
        LaneGauge.BAND_RENDER_AUTHORITY: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="FEED-dv #203/#213/#215 --lane-render-band: analytic class-1 render authority "
                       "composited PRE-R (AA-SDF range-dependent coverage x dash gate x witness-margin "
                       "FP killer); 0-byte decode-deterministic. Advisory openpilot-poly band base "
                       "d_seg~0.00087 (FEED-dj); the NET-NEGATIVE through-R d_seg is realized by "
                       "TRAINING WITH the band active (GPU-pending) -> measured=False PENDING. NOW "
                       "self-orient-composable (Option-B lane-band wire-in)"),

        # --- HEAD_GEOMETRY (#218; byte-free head/margin-field facets) --------------------------
        HeadGeometryGauge.SOFTMAX: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="--head softmax: standard softmax head (byte-identical baseline; the head "
                       "d_seg is the witness's own floor -> unranked in fix_gauge)"),
        HeadGeometryGauge.ETF: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="#218 --head etf: frozen simplex-ETF head (Yang 2022, neural-collapse optimal) "
                       "removes minority-class NORM COLLAPSE that erases Lane/Movable; regenerable "
                       "from a fixed seed at inflate => the KxD head weight is FREE (rate win, 0 bytes "
                       "counted); through-R d_seg delta UNMEASURED (pending)"),
        HeadGeometryGauge.ADDITIVE_MARGIN: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="#218 --head additive-margin: additive-margin softmax (boundary sharpening, "
                       "0-byte); through-R d_seg delta UNMEASURED (pending)"),
        HeadGeometryGauge.MENON_LOGIT_ADJUST: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="#218 --logit-adjust-per-class: Menon per-class logit adjustment (rare-class "
                       "Lane/Movable target boost, 0-byte); through-R d_seg delta UNMEASURED (pending)"),

        # --- MARGIN_SALIENCY (FEED-03n/03p; LEVER-4 multiplier chart, 0 archive bytes train-time) ----
        # Both charts are 0-archive-byte train-time saliency weights; the DISCRIMINATOR is a d_seg-
        # through-R effect that is #205-gated, NOT a byte cost -> BOTH numeric fields stay None so
        # fix_gauge lists them as PENDING (measured-but-unrankable) = the NO-FAKE honest state (like the
        # baseline RENDER_AA.NONE cell). The MEASURED correlation diagnostics live in provenance (the
        # same discipline as the WARP charts' pre-R numbers: feeding a byte=0 tie would MIS-RANK them).
        MarginSaliencyGauge.TEXTURE_PROXY: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="f99a3863a + memory [[msal-uni-texture-proxy-inert-build-exact-sR-reachability-"
                       "weight]] (scratchpad/measure_SR_vs_texproxy.py, n6/n16/n96 advisory): the "
                       "--margin-saliency-uniward texture multiplier 1/(1+beta*tex) is MEASURED INERT -- "
                       "texture is orthogonal to through-R detector reachability (Pearson -0.033 vs S_R, "
                       "top-5% Jaccard 0.024 ~= chance 0.026; the full lever's 0.21 S_R-alignment is "
                       "ENTIRELY the fragility factor w=exp(-margin/tau), texture adds +0.009 P) AND "
                       "mildly MISDIRECTS (texprox vs |grad margin| -0.215). 0 archive bytes (train-time); "
                       "numeric fields None (the net d_seg is a #205-gated A/B, not a byte cost) -> "
                       "PENDING/unrankable in fix_gauge (NO-FAKE)"),
        MarginSaliencyGauge.THROUGH_R_REACHABILITY: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="f99a3863a + tools/precompute_sR_reachability.py: --margin-saliency-reachability "
                       "REPLACES the texture path with the exact through-R fragility-weighted margin-"
                       "Jacobian S_R=|d(sum_p w_p*margin_p)/dx|, w=exp(-margin/tau) -> sal=exp(-margin/"
                       "tau)*S_R_norm. MEASURED 3.0x concentrated on the fragile margin band (S_R vs "
                       "|grad margin| +0.272, vs margin -0.323); theta-INDEPENDENT -> cached alongside "
                       "'margins' in the gt-cache (strictly cheaper than the per-step tex recompute). "
                       "0 archive bytes; net d_seg #205-gated -> numeric fields None (unrankable, MEANS)"),

        # --- MUON_MOMENTUM (FEED-03o/03q; base-trainer finisher schedule, 0 archive bytes train-time) -
        MuonMomentumGauge.COLD_MOMENTUM: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="cba2e4375 (base-trainer default) + memory [[muon-deep-dive-keep-and-tune-"
                       "finishing-stage-schedule-not-switch]]: fresh optim.Muon -> zero buffer; the cold "
                       "first orthogonalized step is a wild unit-norm direction from ONE noisy gradient "
                       "-> boundary thrash -> MEASURED +0.000357 d_seg SPIKE at the sister thetastar Muon "
                       "transition (ep750 saddle-gate). byte-identical default; the spike is a transition "
                       "effect not a byte cost -> numeric fields None (PENDING)"),
        MuonMomentumGauge.WARM_START: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="cba2e4375 --muon-warm-start-momentum (base trainer; BooleanOptionalAction): seed "
                       "Muon momentum v from the OUTGOING AdamW first-moment m (both gradient EMAs; "
                       "Newton-Schulz re-normalizes) -> removes the cold-start +0.000357 spike. PROVEN "
                       "byte-identical when OFF (0 nonzero cold v). Net d_seg #205-gated (arm #270, "
                       "GO-gated) -> numeric fields None (unrankable, advisory MEANS)"),

        # --- MUON_LR (FEED-03o/03q; base-trainer finisher schedule, 0 archive bytes train-time) --------
        MuonLRGauge.FLAT_LR: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="cba2e4375 (base-trainer default; --muon-lr-final-frac 1.0 = no decay): flat Muon "
                       "LR. Muon's Newton-Schulz fixes update MAGNITUDE so a flat LR cannot self-reduce "
                       "the step near a minimum -> plateaus/oversteps (river-valley Muon 2606.21514). "
                       "byte-identical default -> numeric fields None (PENDING)"),
        MuonLRGauge.ANNEAL_LR: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="cba2e4375 --muon-lr-final-frac <f> (base trainer; default 0.1): cosine-DECAY the "
                       "Muon-group LR to --muon-lr*f across the Muon span (the #205-arm config 0.002-> "
                       "2e-4). PROVEN byte-identical at frac>=1.0 (max|dLR|=0). Net d_seg #205-gated -> "
                       "numeric fields None (unrankable, advisory MEANS)"),

        # --- ALONG_TANGENT_FREQ (FEED-03t; the ROOT-CAUSE lever, 0 archive bytes, R-safe) -----------
        # Both charts emit the real --n-dir-freqs flag; the deficit is MEASURED, the net d_seg of
        # raising the frequency is a #205-class A/B -> numeric fields None (unrankable, advisory MEANS).
        AlongTangentFrequencyGauge.N_DIR_FREQS_2_DEFICIT: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t + eq anisotropic_basis_along_tangent_frequency_deficit_v1 + memory "
                       "[[lane-dash-residual-root-is-along-tangent-freq-deficit-R-allpass]]: --n-dir-"
                       "freqs 2 = the #205-LIVE deficit (basis freq_along<=8 vs dash ~25 cyc/unit = 3.2x "
                       "deficit -> lane dashes erase finest-first). The ROOT CAUSE. R is all-pass to 2px "
                       "(eq contest_r_operator_mtf_allpass_to_2px_v1) -> the deficit is representational, "
                       "not R-washout. numeric None (baseline; net is the A/B arm)"),
        AlongTangentFrequencyGauge.N_DIR_FREQS_4: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t --n-dir-freqs 4: the #1 ranked ep300+ lever -- raises along-tangent "
                       "bandwidth to attack the dash residual at its representational root (~0 archive "
                       "bytes, R-safe <=Nyquist; optionally pair --bank-n-scales 4->5). Deficit MEASURED; "
                       "net n600 d_seg is a #205-class A/B (operator GO, next-run/improved-Muon-restart) "
                       "-> numeric None (unrankable, advisory MEANS)"),

        # --- VECTOR_MARGIN_SALIENCY (FEED-03t; margin-saliency scalar->vector, 0 archive bytes) ------
        VectorFieldMarginSaliencyGauge.SCALAR_MAGNITUDE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t: the existing SCALAR margin-saliency #141 path (byte-identical "
                       "baseline; #205-live). numeric None (baseline; the witness's own floor)"),
        VectorFieldMarginSaliencyGauge.VECTOR_T_SUBPIXEL: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t + eq separatrix_asymmetry_t_subpixel_boundary_localizer_v1 (probe "
                       "a8afad40 GREEN): t=M_p/(M_p+M_q) is a self-consistent (+0.560 disjoint) sub-pixel "
                       "boundary position + flip-side localizer -> upgrades margin-saliency scalar->vector. "
                       "TRAINING lever BUILT (LEVER-4b --seg-subpix-boundary-weight; default-off "
                       "byte-identical PROVEN A==B vs HEAD); the net d_seg is a #205-class A/B (owed) -> "
                       "numeric None (unrankable, advisory MEANS)"),

        # --- CHROMA_BOUNDARY (FEED-03t; PROVEN independent d_seg DOF, 0 archive bytes; --chroma) -----
        ChromaBoundaryGauge.CHROMA_ACTIVE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t + eq chroma_decides_lane_and_movable_at_annulus_v1 (probe a3e9f0bd "
                       "GREEN, n96 advisory, 100% L*-match to frozen SegNet): --chroma default ON = the "
                       "#205-live baseline; chroma is a PROVEN independent boundary d_seg DOF (margin-"
                       "gradient energy 78.8% luma / 21.2% chroma). numeric None (baseline)"),
        ChromaBoundaryGauge.LUMA_ONLY: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t (probe a3e9f0bd GREEN): --no-chroma ablation MEASURED chroma removal "
                       "(constant-luma) flips 7.54% Lane->Road + 4.38% Movable->Undrivable, 93.4% of "
                       "chroma-flips in the margin<1 annulus (33.7% at margin<0.25) -> chroma is an "
                       "independent d_seg DOF concentrated on the lane crux + Movable over-dilation, "
                       "orthogonal to the geometry levers. The 'route chroma INTO the annulus' refinement "
                       "is the A/B-owed lever -> numeric None (advisory MEANS)"),

        # --- FLICKER_TREATMENT (FEED-03t; design-stage loss levers, probe a949ff63 still measuring) --
        FlickerTreatmentGauge.NONE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t: no flicker treatment (byte-identical baseline; #205-live). For an "
                       "INDEPENDENT jitter, smooth (r=0) is provably optimal (eq independent_flicker_"
                       "jitter_dseg_floor_smooth_optimal_v1). numeric None (baseline)"),
        FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="FEED-03t DESIGN-STAGE (probe a949ff63 still measuring the predictable/irreducible "
                       "split): down-weight the provably-irreducible sensor-noise flicker (smooth-optimal). "
                       "loss lever UNBUILT (accessor fail-closes) -> PENDING (measured=False)"),
        FlickerTreatmentGauge.REPLICATE_PREDICTABLE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="FEED-03t DESIGN-STAGE (probe a949ff63 still measuring): replicate+reward the "
                       "PREDICTABLE (ego-xi/quant-phase) flicker (requires temporal correlation to lower "
                       "d_seg). loss lever UNBUILT (accessor fail-closes) -> PENDING (measured=False)"),

        # --- TRIPLE_JUNCTION_MARGIN (FEED-03t; a4c66f2f CLOSED WEAK, 0 archive bytes) ----------------
        TripleJunctionMarginGauge.TOP1_TOP2_SCALAR: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t + eq scalar_top1_top2_margin_is_exact_distance_to_flip_v1 (probe "
                       "a4c66f2f, n600 exact): gap13>=gap12 at ALL 118M pixels -> the scalar top1-top2 "
                       "margin IS the exact distance-to-flip (PROVEN sufficient baseline; #205-live). "
                       "numeric None (baseline)"),
        TripleJunctionMarginGauge.MULTICLASS_SIMPLEX: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t (probe a4c66f2f CLOSED WEAK -- BANKED low-EV): the multi-class simplex "
                       "adds NO flip-onset DOF (gap13>=gap12 everywhere); triple junctions = 0.027% of "
                       "pixels, ~1-2% flip mass, 53.9% Road|Undriv|Movable car-corners = NOT the lane "
                       "tail. Bench lever w(p)=fragility*(1+lambda*1[gap13<eps]) composes orthogonally but "
                       "targets car-corners -> BEHIND the facet levers. WEAK NEGATIVE that SHARPENS the "
                       "crux (aim at the codim-1 Road|Lane facet). numeric None (banked, MEANS)"),
    }
    return GaugeCostTable(cells)


# ---------------------------------------------------------------------------
# GaugeChoice — one chart per component, with BY-CONSTRUCTION compliance gating
# ---------------------------------------------------------------------------
class GaugeViolation(ValueError):
    """A GaugeChoice selected a chart that fails a HARD gate (compliance/determinism)
    or that has no certifiable cost cell — REJECTED BY CONSTRUCTION."""


@dataclass(frozen=True)
class GaugeChoice:
    """A fixed gauge: one chart per witness component. ``validate`` REJECTS (raises
    GaugeViolation) any selected chart that is non-compliant or non-deterministic per the
    cost table — like the DSL's preserve/contain/authority enforced clauses."""

    warp: WarpGauge
    carrier: CarrierGauge
    residual: ResidualGauge
    pose: PoseGauge
    movables: MovablesGauge
    generation: GenerationGauge
    # #224 Option-B APPEND-ONLY render/head components. OPTIONAL (defaults = the fail-closed OFF /
    # byte-identical member) so EVERY existing 6-field GaugeChoice(...) caller is UNBROKEN: an old
    # call constructs render_aa=NONE / lane_band=NONE / head_geometry=SOFTMAX, which validate() as
    # compliant+deterministic byte-identical baselines. Selecting an ACTIVE chart is opt-in.
    render_aa: RenderAAGauge = RenderAAGauge.NONE
    lane_band: LaneGauge = LaneGauge.NONE
    head_geometry: HeadGeometryGauge = HeadGeometryGauge.SOFTMAX

    def items(self) -> tuple[tuple[GaugeComponent, Enum], ...]:
        return (
            (GaugeComponent.WARP, self.warp),
            (GaugeComponent.CARRIER, self.carrier),
            (GaugeComponent.RESIDUAL, self.residual),
            (GaugeComponent.POSE, self.pose),
            (GaugeComponent.MOVABLES, self.movables),
            (GaugeComponent.GENERATION, self.generation),
            (GaugeComponent.RENDER_AA, self.render_aa),
            (GaugeComponent.LANE_BAND, self.lane_band),
            (GaugeComponent.HEAD_GEOMETRY, self.head_geometry),
        )

    def validate(self, table: GaugeCostTable | None = None) -> "GaugeChoice":
        """Raise GaugeViolation on the first non-compliant / non-deterministic / uncertifiable
        selected chart; otherwise return self (fluent). Mirrors WitnessProgram's behavior
        clauses, but RAISES (a gauge choice is rejected BY CONSTRUCTION, FEED-ji (d))."""
        t = table if table is not None else default_cost_table()
        for comp, chart in self.items():
            cost = t.lookup(chart)
            if cost is None:
                raise GaugeViolation(
                    f"{comp.value}={chart.name}: no GaugeCost cell — cannot certify "
                    "compliance/determinism (add a probe cell first)")
            if not cost.compliant:
                raise GaugeViolation(
                    f"{comp.value}={chart.name}: NON-COMPLIANT chart (rule-118 HARD gate) "
                    f"— {cost.provenance}")
            if not cost.deterministic:
                raise GaugeViolation(
                    f"{comp.value}={chart.name}: NON-DETERMINISTIC chart (HARD gate) "
                    f"— {cost.provenance}")
        return self


# The current canonical (best-known) gauge — what ``fix_gauge`` selects where a measured
# winner exists; warp = SCREW_TWIST (a513372a FEED-jj MEASURED-WIN: ~0 marginal bytes beats
# the 6,600-param overfit homography); residual = the Wyner-Ziv head-start base
# CONDITIONAL_ON_LANE_PRIOR (measured 0.00214 base; the DIRECT_LEARNED residual on top is the
# pending binding move); generation = DETERMINISTIC_FREE (rule-118: maximize the free generic
# generator). All six charts are compliant + deterministic, so ``CANONICAL_GAUGE.validate()``
# passes. This is the default ``with_gauge`` falls back to for unspecified components.
CANONICAL_GAUGE = GaugeChoice(
    warp=WarpGauge.SCREW_TWIST,
    carrier=CarrierGauge.SINGLE_SDF,
    residual=ResidualGauge.CONDITIONAL_ON_LANE_PRIOR,
    pose=PoseGauge.RANGE_DELTA,
    movables=MovablesGauge.STORE,
    generation=GenerationGauge.DETERMINISTIC_FREE,
)


# ---------------------------------------------------------------------------
# fix_gauge — the selection rule (hard-gates → drop pending → min-S → tiebreak)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaugeVerdict:
    """The fix_gauge readback. ``.explain()`` reads back the rule chain (GOSDT-style)."""

    component: GaugeComponent
    chosen: Enum | None
    runner_up: Enum | None
    pending: tuple[Enum, ...]
    hard_gate_dropped: tuple[Enum, ...]
    rationale: str

    def explain(self) -> str:
        return self.rationale


def fix_gauge(component: GaugeComponent, table: GaugeCostTable | None = None) -> GaugeVerdict:
    """Fix the gauge for one component: (i) drop charts failing a HARD gate
    (non-compliant OR non-deterministic); (ii) drop PENDING/unmeasured charts (and those
    measured-but-unrankable) — returned in ``pending`` so the caller knows a probe is
    needed; (iii) among the rest minimize the component's S-contribution
    (S = 100*d_seg_through_R + 25*counted_bytes/N; pose is byte-only, handled gracefully);
    (iv) deterministic synergy/composition tiebreak by enum-declaration order.

    Returns a GaugeVerdict whose ``.explain()`` cites the rule chain.
    """
    t = table if table is not None else default_cost_table()
    gauge_cls = COMPONENT_GAUGES[component]

    hard_dropped: list[Enum] = []
    pending: list[Enum] = []
    ranked: list[tuple[float, int, Enum]] = []  # (S-contribution, enum-order, chart)

    for idx, chart in enumerate(gauge_cls):
        cost = t.lookup(chart)
        if cost is None:
            pending.append(chart)  # no cell → needs a probe
            continue
        if not cost.passes_hard_gates():  # rule (i)
            hard_dropped.append(chart)
            continue
        if not cost.measured:  # rule (ii): can't select what's unmeasured
            pending.append(chart)
            continue
        s = cost.s_contribution()
        if s is None:  # measured but no rankable cost yet
            pending.append(chart)
            continue
        ranked.append((s, idx, chart))

    ranked.sort(key=lambda r: (r[0], r[1]))  # rule (iii) min-S, then (iv) enum-order tiebreak
    chosen = ranked[0][2] if ranked else None
    runner_up = ranked[1][2] if len(ranked) > 1 else None

    rationale = _build_rationale(component, ranked, pending, hard_dropped, t)
    return GaugeVerdict(
        component=component, chosen=chosen, runner_up=runner_up,
        pending=tuple(pending), hard_gate_dropped=tuple(hard_dropped), rationale=rationale)


def _build_rationale(
    component: GaugeComponent,
    ranked: list[tuple[float, int, Enum]],
    pending: list[Enum],
    hard_dropped: list[Enum],
    table: GaugeCostTable,
) -> str:
    """GOSDT-style decision-path readback (per CLAUDE.md preflight-rule-chain discipline)."""
    parts = [f"fix_gauge({component.value}):"]
    parts.append(
        f"hard_gate_drop(non-compliant|non-deterministic)=[{', '.join(c.name for c in hard_dropped)}]")
    parts.append(
        f"pending(needs probe / unrankable)=[{', '.join(c.name for c in pending)}]")
    if ranked:
        ladder = " < ".join(f"{c.name}={s:.6f}" for s, _i, c in ranked)
        parts.append(f"rank by S-contribution(=100*d_seg_R+25*bytes/N): {ladder}")
        chosen = ranked[0][2]
        runner = ranked[1][2].name if len(ranked) > 1 else "none"
        parts.append(f"→ CHOOSE {chosen.name} (runner-up {runner}); tiebreak=enum-order")
        cost = table.lookup(chosen)
        if cost is not None:
            parts.append(f"[{chosen.name} provenance: {cost.provenance}]")
    else:
        parts.append("→ NO selectable chart (all charts pending/dropped); a probe is the next step")
    return " ; ".join(parts)


# ---------------------------------------------------------------------------
# Composition with the DSL program (standalone, mirrors campaign.compose_theta_star)
# ---------------------------------------------------------------------------
def compose_gauged_program(program, gauge_choice: GaugeChoice, *,
                           table: GaugeCostTable | None = None):
    """Validate ``gauge_choice`` (raises GaugeViolation on a non-compliant/non-deterministic
    chart) and attach it to ``program`` — returns a NEW WitnessProgram (pure; the input is
    unmutated). Equivalent to ``program.with_gauge(gauge_choice)``; provided for symmetry with
    ``campaign.compose_theta_star``."""
    gauge_choice.validate(table)
    return program.with_gauge(gauge_choice, table=table)
