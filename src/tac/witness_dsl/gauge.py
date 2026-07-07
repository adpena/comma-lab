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
    # Deep-math pass #284 "Amortizing the Argmax" levers (FEED-03y/03z, APPEND-ONLY; train-time
    # schedule/loss facets, 0 archive bytes; NOT GaugeChoice STORE fields — like ALONG_TANGENT_FREQ /
    # MUON_*). Both are the §7 cross-chapter-converged next-run A/Bs (config B / #285), UNMEASURED.
    GAMMA_TAU_EIKONAL = "gamma_tau_eikonal"              # Ch.4 phase-field: geometric tau-floor + eikonal
    STAGE_TRANSITION_EASING = "stage_transition_easing"  # Ch.6 dynamics: deconflict ep300 + LR re-warmup
    CONTROL_SYSTEM = "control_system"                    # #292: seed-paint + eik-ramp + event-trigger + closed-loop
    # #302 curriculum-derivation symposium (FEED-05i, APPEND-ONLY; train-time schedule facet, 0
    # archive bytes; NOT a GaugeChoice STORE field — like CONTROL_SYSTEM / STAGE_TRANSITION_EASING).
    CURRICULUM = "curriculum"                            # #302: clock-vs-trigger schedule provenance
    # Mallat/Ballé review row 7 BUILD 1 (FEED-08h, APPEND-ONLY; BYTE-CLOSE-side section-codec
    # facet, ships in the counted 5th block; NOT a GaugeChoice STORE field and NOT a trainer flag —
    # the flag lives on tools/levelset_byte_close_and_eval.py, like LaneGauge's --lane-band-naive).
    LANE_BAND_CODER = "lane_band_coder"                  # LBND2 raw-zigzag vs LBND4 ξ-residual entropy stage


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


class LaneBandCoderGauge(Enum):
    """How the COUNTED lane-coeff payload (the LaneGauge BAND_RENDER_AUTHORITY statistic) is
    entropy-coded in the byte-close 5th block (Mallat/Ballé review row 7 BUILD 1, FEED-08h).

    A BYTE-CLOSE-side section-codec chart, NOT a trainer flag (never-invent-flags: these flags
    live on ``tools/levelset_byte_close_and_eval.py``, exactly like LaneGauge's documented
    ``--lane-band-naive`` opt-out — the accessor is ``lane_band_coder_byte_close_flags``, kept
    OUT of the trainer-argv surface so the lever_registry stale-drift check stays clean).
    Every chart decodes to the BIT-IDENTICAL dequantized LaneLine statistic (same L3/L4 grid);
    only the entropy stage differs, so the choice is pure rate (d_seg/d_pose invariant).

    MEASURED n600 (real gt_n600 fit, the byte-close tool's own build path; brotli-counted,
    ``experiments/results/lane_band_res_coder_20260707/lane_band_res_coder_n600_measured.json``,
    [macOS-CPU advisory]): LBND2 41,526 B (0.02765) -> LBND4 varint 30,892 B (0.02057) =
    **−10,634 B / −25.6%**, decode-reencode byte-identical for all three residual schemes
    (varint 30,892 < rice 33,229 < zlib9 34,701 post-brotli). Sister of the ξ delta_res coder
    (−486 B, commit a44a06fb8) — ONE shared residual entropy stage
    (``tac.boundary_math.xi_spline_residual_coder``), never a duplicated copy.

    DEFAULT = RD (LBND2, the sealed-config shipping default; RES is default-OFF, REGISTERED with
    duty-to-measure in the activation ledger — the "off is a tracked queue" discipline). SHIPPING
    RES requires inlining the LBND4 decode half into _INFLATE_PY first (parity gate fails closed
    on the unknown magic until then)."""

    RD = "rd"        # LBND2: raw uint32 zigzag words + outer brotli (shipping default)
    RES = "res"      # LBND4: ξ delta/context residual stage (best-of-three) + outer brotli
    NAIVE = "naive"  # LBND1: per-pair float64 (kept for the naive-vs-RD comparison gate)


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

    NO-FAKE cross-ref (#284 FEED-03y/03z): "Muon = natural gradient" is a FALSE FRIEND. Muon's
    Newton-Schulz orthogonalization is a WEIGHT-SPACE SPECTRAL normalizer (unit singular values of the
    update matrix), NOT the OUTPUT-SPACE Fisher-Rao natural gradient the categorical head actually
    defines (equations ``ce_softmax_mirror_descent_natural_gradient_v1`` — CE-softmax descent IS
    mirror-descent / natural-gradient in the logit simplex — and
    ``fisher_curvature_equals_categorical_fisher_trace_caustic_v1`` — the Fisher metric lives in
    output/class space). The MEASURED -32%
    d_seg of Muon vs AdamW is REAL; the ATTRIBUTION "because Muon approximates the natural gradient" is
    a CONJECTURE (registered as a conjecture, NOT a canonical law) — the schedule levers below are tuned
    on the measured win, never on the conjectured mechanism.

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
    FEED-03o/03q).

    NO-FAKE cross-ref (#284 FEED-03y/03z; sister MuonMomentumGauge): "Muon = natural gradient" is a
    FALSE FRIEND — Newton-Schulz is a weight-space SPECTRAL normalizer, not the output-space Fisher-Rao
    natural gradient of the categorical head (equations ``ce_softmax_mirror_descent_natural_gradient_v1``
    + ``fisher_curvature_equals_categorical_fisher_trace_caustic_v1``). The -32% d_seg win is REAL; the
    NG attribution is a CONJECTURE (not registered as a law). The LR-anneal lever is tuned on the
    measured win (a flat Newton-Schulz update magnitude cannot self-reduce near a minimum), not the
    conjectured mechanism.

    BASE-trainer flags (the Muon-stage carrier; NOT the levelset launch argparse):
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


class EikonalViscoStabGauge(Enum):
    """How the ViscoReg eikonal-viscosity eps is scheduled (V6 #320; DAG FEED-06c). The DERIVED
    mechanism cure for the v5 ep110 eikonal RE-ENTRY (memo adaptive_eps_mechanism_cure_20260705;
    equation ``adaptive_eps_cfl_edge_tracking_v1``).

    NO-FAKE / mechanism (DE #318 §3.1): the v5 death is a TWO-SIDED CFL squeeze — eps ANNEALS DOWN
    toward the lower edge ``eps_lower = |c_a|*sqrt(eta*lambda_eik/8)`` while progressive sharpening
    GROWS ``|c_a(t)|`` (raising that edge) => ``pi_eik = eta*lambda_eik*|c_a|^2/(8*eps^2)`` crosses 1.
    ADAPTIVE_EPS floors/tracks eps ABOVE the rising edge instead of annealing into it.

    LEVELSET-trainer flags (real store_true/float flags on the levelset entry point; requires
    ``--eikonal-viscosity > 0`` so the visco term is active):
      LINEAR_ANNEAL = () (the byte-identical DEFAULT: ``--eikonal-viscosity-anneal`` linear decay to 0).
      ADAPTIVE_EPS  = ``--eikonal-viscosity-adaptive --eikonal-visco-eps-floor <lo>
                       --eikonal-visco-eps-upper <hi> --eikonal-visco-margin-factor <m>`` -- eps(t) =
                       clamp(|c_a(t)|*sqrt(eta*lambda_eik/8)*(1+m), lo, hi), recomputed per-epoch with
                       |c_a(t)| measured no-grad on the witness decision margin (witness-only, zero
                       SegNet cost). NOTE (constant-dependent, honest): at eta~1e-3/lambda~0.05 the edge
                       is far below the 0.3 floor unless |c_a| explodes => ADAPTIVE_EPS degrades to a
                       CONSTANT-0.3-FLOOR (which alone removes the eps->0 HALF of the DE §3.1 re-entry)
                       + explosion insurance. Net d_seg is #205/n600-gated (advisory, MEANS)."""

    LINEAR_ANNEAL = "linear_anneal"  # --eikonal-viscosity-anneal (byte-identical default)
    ADAPTIVE_EPS = "adaptive_eps"    # --eikonal-viscosity-adaptive + floor/upper/margin (V6 #320)


class AlongTangentFrequencyGauge(Enum):
    """How much ALONG-TANGENT Fourier bandwidth the anisotropic self-orient/curvelet basis carries
    (FEED-03t, equation ``anisotropic_basis_along_tangent_frequency_deficit_v1`` + memory
    [[lane-dash-residual-root-is-along-tangent-freq-deficit-R-allpass]]).

    Deep-math cross-ref (#284 FEED-03y/03z, equation ``shearlet_nterm_upper_bounds_task_rate_v1``):
    the self-orient directional basis IS a discrete SHEARLET frame — the provably-optimal sparse basis
    for a curved codim-1 (cartoon) singularity — so the MEASURED -48% D1 d_seg of the all-class
    directional basis is the cartoon-optimal outcome, and the N-term shearlet coefficient count
    UPPER-BOUNDS the task rate of the argmax-edge manifold (the rate half of the sub-0.15 path). HONEST
    scope: the shearlet advantage over wavelets is ASYMPTOTIC (N->infinity), not guaranteed at the
    finite curvelet-column budget here; the cross-ref frames the basis-match, the net is still a #205 A/B.

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
    """How temporal FLICKER (the #205 residual = the popout floor) is treated. FEED-03t/03u/03v/03w,
    equations ``independent_flicker_jitter_dseg_floor_smooth_optimal_v1`` (probe a949ff63) +
    ``leverd_flicker_residual_reactivation_economics_v1`` (the FEED-03w synthesis of the Lever-D net-S
    band + the 6.897 coder floor).

    d_seg = q(1-r)+r(1-q) => for a GT minority-flicker q<0.5, an INDEPENDENT witness jitter r only
    RAISES d_seg (smooth r=0 optimal); lowering d_seg by replicating flicker REQUIRES temporal
    CORRELATION (predictability). The THREE treatment buckets (NONE = the byte-identical baseline):
      NONE                   = no flicker treatment (byte-identical baseline; the #205-live behavior)
      DOWNWEIGHT_IRREDUCIBLE = BUILT (#274, 6e355170d): down-weight the provably-irreducible sensor-noise
                               flicker (smooth-optimal). REAL flags ``--seg-spike-reweight
                               --seg-spike-downweight <w<1.0>`` (the value NO-OPs without the gate flag,
                               the additive-margin trap -> BOTH emitted); default-off byte-identical. The
                               STANDING seg play if Lever-D NO-GOes.
      REPLICATE_PREDICTABLE  = NOT-WARRANTED (design-stage, fail-closed): replicate+reward the PREDICTABLE
                               flicker would need strong temporal correlation, but only ~11.4% of the
                               spike floor is predictable and the ego-coupling is weak (r=0.16) -> the
                               correlated-replicate lever is not warranted at the measured predictability.
      STORE_REGIONAL_LEVERD  = Lever-D reactivation (#279, DESIGN-STAGE, fail-closed): STORE the
                               regionally-coherent temporal flip-residual as a COUNTED 7th archive block
                               (inverse-steg detector-informed INDUCE at decode; no ``--seg-flip-residual``
                               flag BUILT yet). GATED on the ONE Stage-0 byte-measurement min(b)<0.65
                               B/flip AND subset net ΔS<0 (eq ``leverd_flicker_residual_reactivation_
                               economics_v1``); net-S band -0.35 optimistic / -0.048 expected / +0.117
                               pessimistic-WORSE. Even the optimistic corner leaves witness S ~0.40 (~2x
                               above the 0.19110 pointer) -- a d_seg-competitiveness increment, NOT a
                               pointer move.
    DOWNWEIGHT_IRREDUCIBLE emits its REAL BUILT flags; REPLICATE_PREDICTABLE + STORE_REGIONAL_LEVERD
    fail-closed (never-invent-flags)."""

    NONE = "none"                                      # no flicker treatment (byte-identical baseline)
    DOWNWEIGHT_IRREDUCIBLE = "downweight_irreducible"  # BUILT #274: --seg-spike-reweight + -downweight
    REPLICATE_PREDICTABLE = "replicate_predictable"    # NOT-WARRANTED (11.4% predictable, weak ego r=0.16)
    STORE_REGIONAL_LEVERD = "store_regional_leverd"    # #279 DESIGN-STAGE: COUNTED 7th block, b<0.65 gate


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


class GammaTauEikonalGauge(Enum):
    """Ch.4 phase-field lever from the #284 deep-math pass: the Γ-optimal τ-anneal SHAPE + a
    resolution-scale τ-FLOOR + a raised eikonal, treated as ONE COUPLED arm (FEED-03y/03z, config B
    / #285 `.omx/research/deepmath_converged_next_run_config_20260704.md`; equations
    ``tau_eps_hbar_one_dequantization_two_scales_v1``
    + ``multiphase_modica_mortola_perimeter_gamma_limit_v1``
    + ``mcf_minority_erasure_inevitability_v1``).

    Deep-math: the softmax temperature τ IS the Modica-Mortola interface width (τ=ε=ħ). Three coupled
    consequences: (a) the anneal SHAPE should be ``geometric`` = equal epochs per octave of interface
    width (scale-space / GNC-correct), not cosine; (b) τ_end should FLOOR at the resolution scale —
    annealing to the razor default 0.05 = a 0.025px interface = ~40x sub-grid aliasing (wasted, and it
    fuels the late-τ d_seg volatility of the minority-erasure MCF); (c) the eikonal ``|grad phi|->1``
    must be RAISED so a non-vanishing τ still yields a well-conditioned SDF partition — the eikonal is
    what ENABLES the τ-floor, hence the three flags are ONE arm. All three charts emit the REAL
    levelset-trainer flags ``--tau-anneal-shape`` / ``--softmax-temp-end`` / ``--eikonal-weight``
    (never-invent-flags; READ defaults: cosine / 0.05 / 0.01). 0 archive bytes (train-time schedule).
      BASELINE = the trainer's CURRENT defaults (cosine anneal / temp-end 0.05 / eikonal-weight 0.01)
                 -> emits () = BYTE-IDENTICAL to the #205-live config (the pinned baseline arm).
      GEOMETRIC_TAU_FLOOR_EIKONAL = ``--tau-anneal-shape geometric --softmax-temp-end <resolution-scale
                 ~1.0> --eikonal-weight 0.05`` (the config-B tuple). NOTE the trainer's default
                 ``--softmax-temp-start`` is ALSO 1.0, so end=1.0 HOLDS τ at the resolution/dequantization
                 scale (the eikonal-conditioned SDF, not a sub-grid τ, supplies the crisp partition); a
                 campaign that wants a genuine geometric DECAY to the floor raises --softmax-temp-start
                 above it (threaded via gamma_tau_eikonal_trainer_flags(temp_end=..., eikonal_weight=...)).
    UNMEASURED -> the cost cell is PENDING; the net d_seg is a #205-class A/B (net-S #205-gated, operator-
    GO-gated for any dispatch; MEANS, pointer 0.19110 UNMOVED)."""

    BASELINE = "baseline"                                          # cosine/0.05/0.01 defaults = () byte-identical
    GEOMETRIC_TAU_FLOOR_EIKONAL = "geometric_tau_floor_eikonal"    # geometric + tau-floor + raised eikonal


class StageTransitionEasingGauge(Enum):
    """Ch.6 dynamics lever from the #284 deep-math pass: EASE the ep300 curriculum stage transition so
    the two homotopy/continuation params do not collide at full LR with stale momentum (FEED-03y/03z +
    the MEASURED FEED-ft bump; config B / #285; equations
    ``ce_softmax_mirror_descent_natural_gradient_v1``
    + ``muon_finisher_schedule_warmstart_and_lr_anneal_v1``).

    Deep-math: the ep300 d_seg bump (MEASURED FEED-ft: 0.0056 -> 0.020, 3.4x, PERSISTENT 75+ep) is a
    NUMERICAL-CONTINUATION failure, NOT a loss failure — the CE->tau switch
    (``--tau-softplus-start-epoch 300``) and the lane-band engage (``--lane-band-start-epoch 300``)
    change the objective SIMULTANEOUSLY at one epoch at full LR with stale AdamW momentum. Fix = move
    ONE homotopy param at a time (deconflict the band to 350) + a reduced-step corrector (a short LR
    re-warmup from a floor eases the optimizer through the objective change). All flags are REAL
    levelset-trainer flags (never-invent-flags; READ defaults: band-start 300, rewarmup-epochs 0 =
    OFF, rewarmup-floor 0.1, rewarmup-shape linear). 0 archive bytes (train-time schedule). The
    DECONFLICT_REWARMUP arm presupposes an active ``--lr-schedule`` (the trainer REQUIRES it when
    rewarmup-epochs > 0), the sister of MarginSaliency presupposing ``--margin-saliency-weight > 0``.
      NONE = the CURRENT #205 config (band engages @300 colliding with the tau switch; rewarmup OFF)
             -> emits () = BYTE-IDENTICAL (the pinned baseline arm).
      DECONFLICT_REWARMUP = ``--lane-band-start-epoch 350 --stage-transition-rewarmup-epochs 20
             --stage-transition-rewarmup-floor 0.1 --stage-transition-rewarmup-shape cosine`` (config-B
             tuple; band moved off 300 + a 20-epoch cosine LR re-warmup from 0.1 of the scheduled LR).
    UNMEASURED -> the cost cell is PENDING; the net d_seg is a #205-class A/B (net-S #205-gated,
    operator-GO-gated; MEANS, pointer 0.19110 UNMOVED)."""

    NONE = "none"                                    # band@300 collide + rewarmup off = () byte-identical
    DECONFLICT_REWARMUP = "deconflict_rewarmup"      # band@350 + 20ep cosine LR re-warmup


# component → its chart Enum class (for fix_gauge iteration + full-stack sweeps)
class ControlSystemGauge(Enum):
    """#292 control-system composed arm (FEED-04i..04l): the fresh seeded run's training-as-a-
    controlled-dynamical-system layer, deployed as ONE composed arm (operator-approved Tier-1+Tier-3,
    2026-07-04). Four coupled facets, each built + tested + default-OFF/byte-identical in the LIVE
    levelset trainer (commits #291 4f1580d0c / build-1 2a125ab62 / build-2 2bf4ac94f / build-3
    9da07aa34 / SEAL-fix 3e3b9c697):
      (a) SEED nucleation: ``--lane-prior-phi1-mode paint`` + ``--seed-islands`` (paint-then-SDF;
          ``replace`` is the MEASURED no-op; lane_FN 0.00713->0.00211 on real GT; the ep0 admission
          gate is MEASURED part_frac[lane]>0, never flag presence);
      (b) SURVIVAL eikonal STEP-ramp: ``--eikonal-weight 0.05 --eikonal-weight-end 0.10`` stepping at
          the ACTUAL tau/MCF onset (event-fired boundary when (c) is on — the 3e3b9c697 coupling fix;
          measured knee sigma0.8/93%% vs sigma1.5/49%% lane survival);
      (c) EVENT-TRIGGERED curriculum: ``--curriculum-event-triggered`` — stages advance on the
          DETERMINISTIC synchronous ep_loss plateau (hardcoded stage epochs become CAPS; fired epochs
          are logged OUTPUTS; same seed+config => same fired epochs);
      (d) CLOSED LOOP: ``--closed-loop-control`` — sustained DIVERGING_ERASING (monitor-parity
          classification, K-window transient/erosion split) => bounded eikonal bump (<=2, cap 0.20)
          => early-stop+preserve-best if erosion persists post-budget. Decision-only; never launches.
    BASELINE = all four OFF => () = BYTE-IDENTICAL to the #205-live config (the pinned baseline arm).
    CONTROLLED = the FULL composed arm (all four ON) — after the 2026-07-04 adversarial REVISE
    round this is the RUN-2 CANDIDATE form, NOT run-1: the review MEASURED the event trigger
    firing CE->tau ~ep150 on #205's own trace at the default eps (C1, 15%% CE-floor loss) and
    found the l7-fire hole (C2, since guarded 7226d2651) -> run-2 re-arms it at recalibrated
    eps 1e-4 / windows 25 / min-stage 250.
    CONTROLLED_NO_EVENT = the RUN-1 SEALED form (the ``fresh_seeded`` launcher config, FEED-04n):
    (a)+(b)+(d) ON, event trigger OFF -> the eikonal steps at the hardcoded tau onset ep300 =
    the true boundary, fully coherent. Config-as-code SoT: ``tac.witness_autoconfig.
    derive_fresh_seeded_config()`` (this gauge's tuple is its control-system SUBSET; the
    launcher's ``--config fresh_seeded`` is the ONE launch path).
    UNMEASURED as a composed arm -> net-S verdict is the fresh run's byte-closed exact row (MEANS,
    pointer 0.19110 UNMOVED; launch operator-GO + governor gated)."""

    BASELINE = "baseline"        # all control-system flags off = byte-identical #205 path
    CONTROLLED = "controlled"    # paint-seed + eik-ramp + event-trigger + closed-loop (RUN-2 cand.)
    CONTROLLED_NO_EVENT = "controlled_no_event"  # run-1 SEALED: paint-seed + eik-ramp + closed-loop


class CurriculumGauge(Enum):
    """#302 curriculum-derivation symposium lever (FEED-05i; memo
    ``.omx/research/council_grand_symposium_curriculum_derivation_20260705.md``; equations
    ``curriculum_handoff_critical_nucleus_v1`` + ``ema_window_pi_group_v1``
    + ``muon_switch_conditioning_criterion_v1`` + ``rewarmup_beta2_memory_window_v1``).

    The audit's headline: the schedule's CONTINUOUS laws (tau path, eikonal ramp, length, Muon
    tuning, band deconflict) are DERIVED; what remains PR95/Quantizr-inherited is the CLOCK — the
    discrete event structure fires on wall-clock epochs transferred from OTHER trajectories
    (tau@300 = the cert arm's CE knee; Muon@726 = "PR95 stage-8 placement" verbatim, cert A7), and
    the live run-2 still carries a 3-way ep300 collision (tau onset + persistence-warmup completion
    + seed-anneal completion — Ch.6's one-homotopy-param rule applied to only 1 of 4 schedules).
    Charts (0 archive bytes; train-time schedule provenance):
      PR95_ECHO = the as-launched run-2 fixed-epoch clock (300/726/1000, ema 0.997 flat, linear
                  hosc-beta, 3-way ep300 collision) -> emits () = BYTE-IDENTICAL (the pinned live
                  arm; the honest name for what is running).
      DERIVED_NATIVE = the run-3 derived-native schedule DELTA that is expressible with REAL flags
                  today: stagger the ep300 collision (``--seed-anneal-epochs 275
                  --persistence-warmup-epochs 275`` — crutch + protection complete >=25 ep BEFORE
                  MCF onset) + stage-dependent finisher EMA (``--ema-decay-finisher 0.9995``,
                  pi_ema ~ 0.1-0.3 of the finisher per ema_window_pi_group_v1). COMPOSES with
                  ControlSystemGauge.CONTROLLED (the recalibrated CE->tau event trigger, eps 1e-4 /
                  windows 25 / min-stage 250; l7 converge-fire guard LANDED 7226d2651). Remaining
                  BUILDs named in the memo run-3 spec: per-class nucleus guard (handoff law's
                  forall-class clause), Muon engage-on-trigger (Muon is verified NOT event-fireable
                  today), geometric ``--hosc-beta-anneal`` (choices are linear|cosine only).
      HANDOFF_NUCLEUS = (#302 build, LANDED) the COMPLETED CE->tau hand-off: the recalibrated
                  event trigger (``--curriculum-event-triggered``; eps default now 1e-4) GATED on
                  the per-class critical-nucleus guard (``--curriculum-nucleus-guard`` — forall
                  scored class BORN part_frac>0 AND FORMED within-flip<=thresh, MEASURED at verdict
                  cadence), with the TAU-RELATIVE wall-clock levers re-anchored to the fired boundary
                  (``--curriculum-reanchor-levers``: persistence-warmup + seed-anneal + analytic
                  band) and the per-class handoff_readiness telemetry on. This is DERIVED_NATIVE's
                  stagger SUPERSEDED by event-triggering (the levers now track the FIRED tau, so the
                  fixed 275 stagger is unnecessary) COMPOSED with the nucleus guard. All flags
                  DEFAULT-OFF => byte-identical until this chart emits them; net d_seg is a run-3 A/B.
      UNIFIED_ENERGY = the theta*/capstone design (ONE continuously-annealed energy, costate
                  controller, stages dissolve; #218/#78/#247 lineage) — DESIGN-STAGE, fail-closed.
    UNMEASURED as arms -> net d_seg is a run-3-class A/B (net-S gated, operator-GO gated; MEANS,
    pointer 0.19110 UNMOVED)."""

    PR95_ECHO = "pr95_echo"              # run-2 as launched: fixed-epoch clock = () byte-identical
    DERIVED_NATIVE = "derived_native"    # run-3 delta: collision stagger + finisher EMA (real flags)
    HANDOFF_NUCLEUS = "handoff_nucleus"  # #302 build: event trigger + per-class nucleus guard + reanchor
    UNIFIED_ENERGY = "unified_energy"    # theta* design-stage (fail-closed NotImplementedError)


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
    # Deep-math pass #284 levers (FEED-03y/03z, APPEND-ONLY; train-time schedule/loss facets, NOT
    # GaugeChoice STORE fields — like ALONG_TANGENT_FREQ / MUON_*).
    GaugeComponent.GAMMA_TAU_EIKONAL: GammaTauEikonalGauge,
    GaugeComponent.STAGE_TRANSITION_EASING: StageTransitionEasingGauge,
    GaugeComponent.CONTROL_SYSTEM: ControlSystemGauge,
    # #302 curriculum-derivation symposium (FEED-05i, APPEND-ONLY; train-time schedule facet, NOT
    # a GaugeChoice STORE field — like CONTROL_SYSTEM / STAGE_TRANSITION_EASING).
    GaugeComponent.CURRICULUM: CurriculumGauge,
    # Mallat/Ballé review row 7 BUILD 1 (FEED-08h, APPEND-ONLY; byte-close 5th-block section
    # codec — a RATE-only chart over the identical dequantized statistic; NOT a GaugeChoice
    # STORE field and NOT a trainer flag; accessor = lane_band_coder_byte_close_flags).
    GaugeComponent.LANE_BAND_CODER: LaneBandCoderGauge,
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
# BYTE-CLOSE TOOL argv (tools/levelset_byte_close_and_eval.py), NOT trainer argv — deliberately a
# separate map + accessor so these flags never enter the DSL's trainer-emitted surface (the
# lever_registry ``stale == []`` invariant + never-invent-flags stay intact). RD = () because
# LBND2 IS the tool's default; the active charts emit the tool's real opt-in flags.
LANE_BAND_CODER_BYTE_CLOSE_FLAGS: dict[LaneBandCoderGauge, tuple[str, ...]] = {
    LaneBandCoderGauge.RD: (),
    LaneBandCoderGauge.RES: ("--lane-band-res",),
    LaneBandCoderGauge.NAIVE: ("--lane-band-naive",),
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


def lane_band_coder_byte_close_flags(chart: LaneBandCoderGauge) -> tuple[str, ...]:
    """The BYTE-CLOSE-TOOL argv flags for a LaneBandCoderGauge chart (RD => () — LBND2 is the
    tool default). These are ``tools/levelset_byte_close_and_eval.py`` flags, NEVER trainer
    argv (kept out of the DSL trainer-emitted surface by design; see the chart docstring)."""
    return LANE_BAND_CODER_BYTE_CLOSE_FLAGS[chart]


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


# V6 #320 adaptive-eps CFL-edge tracker (DAG FEED-06c; the EIK-STAB flag family, minimally
# triality-synced per #317 — a full EIK-STAB gauge pass (steik/steik-normalized/visco base) is a
# noted follow-up). LEVELSET-trainer flags (real; the eikonal stabilizer lives on the levelset entry
# point). ADAPTIVE_EPS presupposes --eikonal-viscosity > 0 (the visco term must be active).
EIKONAL_VISCO_EPS_FLOOR_DEFAULT = 0.3    # FEED-05v measured stable floor (never anneal below)
EIKONAL_VISCO_EPS_UPPER_DEFAULT = 0.7    # below the eps=1.0 biharmonic explosion (FEED-05v)
EIKONAL_VISCO_MARGIN_FACTOR_DEFAULT = 0.5  # DE #318 §7.4 safety margin above the CFL lower edge
EIKONAL_VISCO_STAB_TRAINER_FLAGS: dict[EikonalViscoStabGauge, tuple[str, ...]] = {
    EikonalViscoStabGauge.LINEAR_ANNEAL: (),                       # byte-identical default
    EikonalViscoStabGauge.ADAPTIVE_EPS: (
        "--eikonal-viscosity-adaptive",
        "--eikonal-visco-eps-floor", str(EIKONAL_VISCO_EPS_FLOOR_DEFAULT),
        "--eikonal-visco-eps-upper", str(EIKONAL_VISCO_EPS_UPPER_DEFAULT),
        "--eikonal-visco-margin-factor", str(EIKONAL_VISCO_MARGIN_FACTOR_DEFAULT)),
}


def eikonal_visco_stab_trainer_flags(chart: EikonalViscoStabGauge,
                                     eps_floor: float | None = None,
                                     eps_upper: float | None = None,
                                     margin_factor: float | None = None) -> tuple[str, ...]:
    """The LEVELSET-trainer argv for an EikonalViscoStabGauge chart. LINEAR_ANNEAL => () (the
    byte-identical --eikonal-viscosity-anneal default). ADAPTIVE_EPS => the real
    ``--eikonal-viscosity-adaptive --eikonal-visco-eps-floor/-upper/-margin-factor`` tuple (V6 #320);
    optional overrides thread a campaign-specific clamp window (defaults = the FEED-05v-measured
    stable floor 0.3 / biharmonic-safe upper 0.7 / DE §7.4 margin 0.5). Presupposes
    ``--eikonal-viscosity > 0`` so the visco term is active (never a silent no-op)."""
    if chart is EikonalViscoStabGauge.LINEAR_ANNEAL:
        return ()
    lo = EIKONAL_VISCO_EPS_FLOOR_DEFAULT if eps_floor is None else float(eps_floor)
    hi = EIKONAL_VISCO_EPS_UPPER_DEFAULT if eps_upper is None else float(eps_upper)
    mf = EIKONAL_VISCO_MARGIN_FACTOR_DEFAULT if margin_factor is None else float(margin_factor)
    if hi < lo:
        raise ValueError(
            f"EikonalViscoStabGauge.ADAPTIVE_EPS eps_upper={hi} < eps_floor={lo} is degenerate; "
            "pass eps_upper >= eps_floor (the clamp window)")
    return ("--eikonal-viscosity-adaptive",
            "--eikonal-visco-eps-floor", str(lo),
            "--eikonal-visco-eps-upper", str(hi),
            "--eikonal-visco-margin-factor", str(mf))


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
# DOWNWEIGHT_IRREDUCIBLE is BUILT (#274, 6e355170d): the spike-aware seg-CE reweight. The value flag
# --seg-spike-downweight only takes effect WITH the --seg-spike-reweight gate (else silent no-op = the
# additive-margin trap) -> emit BOTH. The baked value is a REPRESENTATIVE scale-appropriate A/B starter
# (the A/B tunes it + the sister --seg-coherent-upweight); mirrors ADDITIVE_MARGIN_DEFAULT / the other
# BUILT levers baking a starter value.
SEG_SPIKE_DOWNWEIGHT_DEFAULT = 0.25

FLICKER_TREATMENT_TRAINER_FLAGS: dict[FlickerTreatmentGauge, tuple[str, ...]] = {
    FlickerTreatmentGauge.NONE: (),  # byte-identical baseline
    FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE: (
        "--seg-spike-reweight", "--seg-spike-downweight", str(SEG_SPIKE_DOWNWEIGHT_DEFAULT)),
}

# REPLICATE_PREDICTABLE (NOT-WARRANTED) + STORE_REGIONAL_LEVERD (#279 Lever-D DESIGN-STAGE) are UNBUILT
# -> documented, NEVER emitted (never-invent-flags; mirrors POSE_TRAINING_INTENDED_ARGS). Replace with a
# real map + drop the NotImplementedError branch when/if the lever lands (STORE_REGIONAL is gated on the
# Stage-0 min(b)<0.65 B/flip byte-measurement).
FLICKER_TREATMENT_INTENDED_ARGS: dict[FlickerTreatmentGauge, str] = {
    FlickerTreatmentGauge.NONE: "",  # no flicker treatment; emits nothing (baseline)
    FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE:
        "--seg-spike-reweight --seg-spike-downweight (WIRED as #274; see FLICKER_TREATMENT_TRAINER_FLAGS)",
    FlickerTreatmentGauge.REPLICATE_PREDICTABLE:
        "--flicker-replicate-predictable (NOT-WARRANTED: ~11.4% predictable, weak ego r=0.16; not wired)",
    FlickerTreatmentGauge.STORE_REGIONAL_LEVERD:
        "--seg-flip-residual (#279 Lever-D DESIGN-STAGE; gated on Stage-0 min(b)<0.65 B/flip; not wired)",
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


def flicker_treatment_trainer_flags(chart: FlickerTreatmentGauge,
                                    downweight: float | None = None) -> tuple[str, ...]:
    """The levelset-trainer argv for a FlickerTreatmentGauge chart. NONE => () (no treatment =
    baseline). DOWNWEIGHT_IRREDUCIBLE => the REAL BUILT #274 flags ``--seg-spike-reweight
    --seg-spike-downweight <w>`` (the value NO-OPs without the gate flag -> BOTH emitted; ``downweight``
    (optional, in [0, 1)) overrides SEG_SPIKE_DOWNWEIGHT_DEFAULT). REPLICATE_PREDICTABLE (NOT-WARRANTED)
    + STORE_REGIONAL_LEVERD (#279 Lever-D DESIGN-STAGE, gated on Stage-0 min(b)<0.65 B/flip) are UNBUILT
    -> RAISE NotImplementedError naming the intended arg (never-invent-flags)."""
    if chart is FlickerTreatmentGauge.NONE:
        return ()
    if chart is FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE:
        if downweight is None:
            return FLICKER_TREATMENT_TRAINER_FLAGS[chart]
        w = float(downweight)
        if not 0.0 <= w < 1.0:
            raise ValueError(
                f"FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE downweight={w} must be in [0, 1) "
                "(>= 1.0 is the byte-identical no-op == FlickerTreatmentGauge.NONE)")
        return ("--seg-spike-reweight", "--seg-spike-downweight", str(w))
    raise NotImplementedError(
        f"FlickerTreatmentGauge.{chart.name} is UNBUILT (never-invent-flags); intended arg: "
        f"{FLICKER_TREATMENT_INTENDED_ARGS[chart]}. REPLICATE_PREDICTABLE is NOT-WARRANTED (~11.4% "
        "predictable, weak ego r=0.16); STORE_REGIONAL_LEVERD is #279 DESIGN-STAGE gated on the Stage-0 "
        "min(b)<0.65 B/flip byte-measurement (eq leverd_flicker_residual_reactivation_economics_v1)."
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


# ---------------------------------------------------------------------------
# Deep-math pass #284 levers (FEED-03y/03z, config B / #285): Ch.4 phase-field (geometric tau-floor +
# eikonal, COUPLED) + Ch.6 dynamics (deconflict ep300 + LR re-warmup). ALL flags are REAL levelset-
# trainer flags (never-invent-flags; the UNBUILT Ch.5-M2 NTK-whitening + Ch.1 dash-comb are EXCLUDED —
# no flag exists, emitting them would be an invented flag). 0-archive-byte train-time schedule facets.
# The BASELINE/NONE chart is the trainer's CURRENT defaults -> emits () BYTE-IDENTICAL (READ from the
# argparse: --tau-anneal-shape cosine / --softmax-temp-end 0.05 / --eikonal-weight 0.01 ; band@300 /
# rewarmup 0=off / floor 0.1 / shape linear).
# ---------------------------------------------------------------------------
# Ch.4 config-B values (READ from the trainer + the §7 converged config): the resolution-scale tau
# FLOOR (from the razor default 0.05) + the raised eikonal (from 0.01) that ENABLES the floor. These
# are REPRESENTATIVE A/B starters the arm tunes; gamma_tau_eikonal_trainer_flags threads overrides
# (mirrors head_geometry_trainer_flags / muon_lr_trainer_flags baking a starter value).
TAU_FLOOR_SOFTMAX_TEMP_END_DEFAULT = 1.0    # resolution/dequantization-scale tau floor (default 0.05)
EIKONAL_TAU_FLOOR_WEIGHT_DEFAULT = 0.05     # raised |grad phi|->1 eikonal (default 0.01) enables the floor

GAMMA_TAU_EIKONAL_TRAINER_FLAGS: dict[GammaTauEikonalGauge, tuple[str, ...]] = {
    GammaTauEikonalGauge.BASELINE: (),  # cosine / temp-end 0.05 / eikonal 0.01 defaults = byte-identical
    GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL: (
        "--tau-anneal-shape", "geometric",
        "--softmax-temp-end", str(TAU_FLOOR_SOFTMAX_TEMP_END_DEFAULT),
        "--eikonal-weight", str(EIKONAL_TAU_FLOOR_WEIGHT_DEFAULT)),
}

# Ch.6 config-B values (READ from the trainer + §7): band deconflicted off the tau@300 collision + a
# short cosine LR re-warmup from a floor. Representative A/B starters.
DECONFLICT_LANE_BAND_START_EPOCH_DEFAULT = 350   # move band off the tau@300 collision (default 300)
STAGE_TRANSITION_REWARMUP_EPOCHS_DEFAULT = 20    # length of the reduced-step corrector (default 0=off)
STAGE_TRANSITION_REWARMUP_FLOOR_DEFAULT = 0.1    # LR fraction at the boundary epoch (== trainer default)

STAGE_TRANSITION_EASING_TRAINER_FLAGS: dict[StageTransitionEasingGauge, tuple[str, ...]] = {
    StageTransitionEasingGauge.NONE: (),  # band@300 + rewarmup off = byte-identical current #205 config
    StageTransitionEasingGauge.DECONFLICT_REWARMUP: (
        "--lane-band-start-epoch", str(DECONFLICT_LANE_BAND_START_EPOCH_DEFAULT),
        "--stage-transition-rewarmup-epochs", str(STAGE_TRANSITION_REWARMUP_EPOCHS_DEFAULT),
        "--stage-transition-rewarmup-floor", str(STAGE_TRANSITION_REWARMUP_FLOOR_DEFAULT),
        "--stage-transition-rewarmup-shape", "cosine"),
}

# #292 control-system composed arm (real levelset-trainer flags, grep-verified; BASELINE=() byte-identical).
# CONTROLLED_NO_EVENT = the run-1 SEALED subset (FEED-04n/04u): event trigger DROPPED per the
# REVISE round (C1 measured-premature at default eps + C2 l7-hole); eikonal steps at the hardcoded
# tau onset. CONTROLLED (full arm) = run-2 candidate at recalibrated eps 1e-4/windows 25/min-stage
# 250 (the C1 disposition) — its min-stage-epochs 150 default below predates that recalibration.
CONTROL_SYSTEM_TRAINER_FLAGS: dict[ControlSystemGauge, tuple[str, ...]] = {
    ControlSystemGauge.BASELINE: (),  # all off = byte-identical #205 path
    ControlSystemGauge.CONTROLLED: (
        "--lane-prior-phi1-mode", "paint",
        "--seed-islands",
        "--eikonal-weight", "0.05",
        "--eikonal-weight-end", "0.10",
        "--curriculum-event-triggered",
        "--curriculum-min-stage-epochs", "150",
        "--closed-loop-control"),
    ControlSystemGauge.CONTROLLED_NO_EVENT: (
        "--lane-prior-phi1-mode", "paint",
        "--seed-islands",
        "--eikonal-weight", "0.05",
        "--eikonal-weight-end", "0.10",
        "--closed-loop-control"),
}


def gamma_tau_eikonal_trainer_flags(chart: GammaTauEikonalGauge, *,
                                    temp_end: float | None = None,
                                    eikonal_weight: float | None = None) -> tuple[str, ...]:
    """The levelset-trainer argv for a GammaTauEikonalGauge chart. BASELINE => () (the trainer's
    current cosine / temp-end 0.05 / eikonal-weight 0.01 defaults = byte-identical). GEOMETRIC_TAU_
    FLOOR_EIKONAL => the REAL COUPLED tuple ``--tau-anneal-shape geometric --softmax-temp-end <te>
    --eikonal-weight <ew>`` (config-B defaults te=1.0, ew=0.05). ``temp_end`` / ``eikonal_weight``
    (optional) thread a campaign override; ``temp_end`` must be > 0 (``--tau-anneal-shape geometric``
    requires --softmax-temp-end > 0) and ``eikonal_weight`` must be >= 0."""
    if chart is GammaTauEikonalGauge.BASELINE:
        return ()
    if temp_end is None and eikonal_weight is None:
        return GAMMA_TAU_EIKONAL_TRAINER_FLAGS[chart]
    te = TAU_FLOOR_SOFTMAX_TEMP_END_DEFAULT if temp_end is None else float(temp_end)
    ew = EIKONAL_TAU_FLOOR_WEIGHT_DEFAULT if eikonal_weight is None else float(eikonal_weight)
    if not te > 0.0:
        raise ValueError(
            f"GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL temp_end={te} must be > 0 "
            "(--tau-anneal-shape geometric requires --softmax-temp-end > 0)")
    if not ew >= 0.0:
        raise ValueError(
            f"GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL eikonal_weight={ew} must be >= 0")
    return ("--tau-anneal-shape", "geometric",
            "--softmax-temp-end", str(te), "--eikonal-weight", str(ew))


def stage_transition_easing_trainer_flags(chart: StageTransitionEasingGauge) -> tuple[str, ...]:
    """The levelset-trainer argv for a StageTransitionEasingGauge chart. NONE => () (band@300 +
    rewarmup off = the byte-identical current #205 config). DECONFLICT_REWARMUP => the REAL config-B
    tuple ``--lane-band-start-epoch 350 --stage-transition-rewarmup-epochs 20 --stage-transition-
    rewarmup-floor 0.1 --stage-transition-rewarmup-shape cosine`` (deconflict the band off the tau@300
    collision + a 20-epoch cosine LR re-warmup). NOTE: the trainer REQUIRES an active ``--lr-schedule``
    when rewarmup-epochs > 0 (the arm presupposes it, like MarginSaliency presupposes
    --margin-saliency-weight > 0)."""
    return STAGE_TRANSITION_EASING_TRAINER_FLAGS[chart]


# #302 chart -> trainer argv (never-invent-flags; all three flags grep-verified in the levelset
# trainer argparse: --seed-anneal-epochs / --persistence-warmup-epochs / --ema-decay-finisher).
CURRICULUM_TRAINER_FLAGS: dict[CurriculumGauge, tuple[str, ...]] = {
    CurriculumGauge.PR95_ECHO: (),
    CurriculumGauge.DERIVED_NATIVE: (
        "--seed-anneal-epochs", "275",
        "--persistence-warmup-epochs", "275",
        "--ema-decay-finisher", "0.9995",
    ),
    # (#302 build, LANDED) the completed CE->tau hand-off: recalibrated event trigger (eps default
    # now 1e-4) + per-class nucleus guard + tau-relative lever re-anchor + readiness telemetry.
    # Event-triggering makes the levers track the FIRED tau, so the fixed 275 collision stagger is
    # unnecessary here (re-anchor replaces it). Compose with the finisher-EMA arm separately.
    CurriculumGauge.HANDOFF_NUCLEUS: (
        "--curriculum-event-triggered",
        "--curriculum-plateau-rel-eps", "1e-4",
        "--curriculum-plateau-windows", "25",
        "--curriculum-min-stage-epochs", "250",
        "--curriculum-nucleus-guard",
        "--curriculum-reanchor-levers",
        "--handoff-readiness-telemetry",
    ),
}


def curriculum_trainer_flags(chart: CurriculumGauge) -> tuple[str, ...]:
    """The levelset-trainer argv for a CurriculumGauge chart. PR95_ECHO => () (the as-launched
    run-2 fixed-epoch clock = byte-identical). DERIVED_NATIVE => the run-3 delta expressible with
    REAL flags today (ep300 collision stagger + SWA finisher EMA); compose with
    ``CONTROL_SYSTEM_TRAINER_FLAGS[ControlSystemGauge.CONTROLLED]``-class arms for the
    recalibrated CE->tau event trigger. UNIFIED_ENERGY is DESIGN-STAGE fail-closed
    (never-invent-flags): RAISES rather than fabricating a flag (mirrors
    ``pose_training_trainer_flags``)."""
    if chart is CurriculumGauge.UNIFIED_ENERGY:
        raise NotImplementedError(
            "CurriculumGauge.UNIFIED_ENERGY is DESIGN-STAGE (theta* unified-energy costate "
            "curriculum, #218/#78/#247 lineage; memo "
            ".omx/research/council_grand_symposium_curriculum_derivation_20260705.md sec C.iii). "
            "never-invent-flags: emit nothing until the real trainer surface lands."
        )
    return CURRICULUM_TRAINER_FLAGS[chart]


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
            compliant=True, deterministic=True, measured=True,
            provenance="FEED-03t/03u: down-weight the provably-irreducible sensor-noise flicker (smooth-"
                       "optimal for an INDEPENDENT jitter). TRAINING lever BUILT (#274, 6e355170d; REAL "
                       "flags --seg-spike-reweight --seg-spike-downweight <w<1.0>; default-off byte-"
                       "identical). The STANDING seg play if Lever-D NO-GOes; net d_seg is a #205-class "
                       "A/B (owed) -> numeric None (unrankable, advisory MEANS)"),
        FlickerTreatmentGauge.REPLICATE_PREDICTABLE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="FEED-03t/03u NOT-WARRANTED (probe a949ff63): replicate+reward the PREDICTABLE "
                       "(ego-xi/quant-phase) flicker REQUIRES strong temporal correlation, but only ~11.4% "
                       "of the spike floor is predictable and the ego-coupling is weak (r=0.16) -> the "
                       "correlated-replicate lever is not warranted. loss lever UNBUILT (accessor fail-"
                       "closes) -> PENDING (measured=False)"),
        FlickerTreatmentGauge.STORE_REGIONAL_LEVERD: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="FEED-03u/03v/03w (#279 Lever-D reactivation, eq leverd_flicker_residual_"
                       "reactivation_economics_v1): STORE the regionally-coherent temporal flip-residual "
                       "as a COUNTED 7th archive block (rule-118 compliant: residual COUNTED, nudge/contour "
                       "FREE; CPU-locked bit-exact decode). DESIGN-STAGE, no --seg-flip-residual flag BUILT "
                       "(accessor fail-closes). GATED on the ONE Stage-0 byte-measurement min(b)<0.65 "
                       "B/flip AND subset net ΔS<0; net-S band -0.35 optimistic / -0.048 expected / +0.117 "
                       "pessimistic-WORSE (even optimistic ~2x above the 0.19110 pointer). PENDING "
                       "(measured=False)"),

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

        # --- GAMMA_TAU_EIKONAL (#284 Ch.4; COUPLED geometric tau-floor + eikonal, 0 archive bytes) ----
        # UNMEASURED deep-math-pass lever (config B / #285): both charts are PENDING (measured=False,
        # None numbers per the NO-FAKE __post_init__ invariant). The net d_seg is a #205-class A/B; no
        # isolated-measured probe has run for this facet yet (unlike the FEED-03t levers whose deficit
        # was measured). fix_gauge lists both as pending -> chosen None (honest state).
        GammaTauEikonalGauge.BASELINE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="#284 FEED-03y/03z + config B (deepmath_converged_next_run_config_20260704.md): "
                       "the trainer's CURRENT defaults (--tau-anneal-shape cosine / --softmax-temp-end "
                       "0.05 / --eikonal-weight 0.01) = the byte-identical baseline arm. PENDING (no "
                       "isolated through-R d_seg probe for this facet; net is the #205-class A/B, MEANS)"),
        GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="#284 FEED-03y/03z + config B + eqs tau_eps_hbar_one_dequantization_two_scales_v1 "
                       "+ multiphase_modica_mortola_perimeter_gamma_limit_v1 + mcf_minority_erasure_"
                       "inevitability_v1: geometric anneal (equal epochs/octave of interface width) + "
                       "tau_end floored at the resolution scale (--softmax-temp-end 1.0, from 0.05) + "
                       "raised eikonal (--eikonal-weight 0.05, from 0.01) that ENABLES the floor = ONE "
                       "COUPLED arm, 0 archive bytes (train-time schedule). UNMEASURED -> PENDING "
                       "(measured=False, numeric None); net d_seg is a #205-class A/B (net-S #205-gated, "
                       "operator-GO-gated, MEANS, pointer 0.19110 UNMOVED)"),

        # --- STAGE_TRANSITION_EASING (#284 Ch.6; deconflict ep300 + LR re-warmup, 0 archive bytes) -----
        StageTransitionEasingGauge.NONE: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="#284 FEED-03y/03z + config B: the CURRENT #205 config (band engages @300 "
                       "colliding with the CE->tau switch; --stage-transition-rewarmup-epochs 0 = off) "
                       "= the byte-identical baseline arm. PENDING (net is the #205-class A/B, MEANS)"),
        StageTransitionEasingGauge.DECONFLICT_REWARMUP: GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False,
            provenance="#284 FEED-03y/03z + config B + the MEASURED FEED-ft ep300 bump (d_seg 0.0056-> "
                       "0.020, 3.4x persistent 75+ep = a numerical-continuation failure, NOT a loss "
                       "failure) + eqs ce_softmax_mirror_descent_natural_gradient_v1 + muon_finisher_"
                       "schedule_warmstart_and_lr_anneal_v1: deconflict the band to 350 (one homotopy "
                       "param at a time) + a 20-epoch cosine LR re-warmup from 0.1 (reduced-step "
                       "corrector). 0 archive bytes (train-time schedule); requires --lr-schedule. "
                       "UNMEASURED -> PENDING (measured=False, numeric None); net d_seg is a #205-class "
                       "A/B (net-S #205-gated, operator-GO-gated, MEANS, pointer 0.19110 UNMOVED)"),
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
