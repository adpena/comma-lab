"""Grand Council Synthesis — 2026-04-12

Full council: Yousfi, Fridrich, Karpathy, Tao, LeCun, Dykstra, Contrarian,
Shannon, Noether, Ramanujan, Gauss, Planck.

This is the single source of truth for strategy from the 20-day sprint
to deadline (May 3, 2026).

Current verified state:
    - Auth score: 1.97 (DALI on Lightning T4)
    - Lightning training: proxy 0.93 at ep830 (still dropping)
    - Quantizr: 0.60 (386KB archive, mask2mask)
    - Our archive: 893KB (847KB video + 46KB postfilter). Rate alone = 0.595
    - Score formula: S = 100*seg + sqrt(10*pose) + 25*rate
    - Compute: Lightning T4 71hr, Kaggle P100 30hr/wk, local M5 Max, bat00 2070S
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    """Council verdict on a technique or experiment."""

    DEPLOY_NOW = "deploy_now"
    DEVELOP_THIS_WEEK = "develop_this_week"
    DEVELOP_NEXT_WEEK = "develop_next_week"
    KEEP_ALIVE = "keep_alive"
    KILL = "kill"


class Lane(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"


@dataclass
class CouncilExperiment:
    """A prioritized experiment with council consensus."""

    name: str
    verdict: Verdict
    lane: Lane
    priority: int  # 1 = highest
    estimated_score: str  # projected score if successful
    estimated_hours: float  # compute hours required
    timeline_days: int  # days to first result
    platform: str  # where it runs

    # Council analysis
    yousfi_assessment: str = ""
    fridrich_assessment: str = ""
    karpathy_assessment: str = ""
    tao_assessment: str = ""
    contrarian_assessment: str = ""
    shannon_assessment: str = ""

    # Dependencies and risks
    dependencies: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    blocking: bool = False

    # Score decomposition (projected)
    projected_seg: Optional[float] = None
    projected_pose: Optional[float] = None
    projected_rate: Optional[float] = None


@dataclass
class KilledTechnique:
    """A technique the council has decided to stop pursuing."""

    name: str
    reason: str
    evidence: str
    hours_wasted: float = 0.0
    lesson: str = ""


@dataclass
class StrategicInsight:
    """A cross-council insight that shapes strategy."""

    source: str  # which council member(s)
    insight: str
    actionable: bool = True
    applied_to: list[str] = field(default_factory=list)


# ============================================================================
# SECTION 1: SCORE DECOMPOSITION ANALYSIS (Shannon + Tao)
# ============================================================================

SCORE_DECOMPOSITION = """
Current auth score: 1.97
    seg component:  100 * 0.00610 = 0.610
    pose component: sqrt(10 * 0.00218) = 0.148  [NOTE: this is DALI-scored]
    rate component: 25 * 0.02302 = 0.575
    check:          0.610 + 0.148 + 0.575 = 1.333 ... wait.

CORRECTION: The 1.97 is from a DIFFERENT checkpoint (Lightning T4 training).
The 1.33 was the promoted floor from Modal A10G dilated_h64.

Current ACTUAL best auth: 1.33 (decomposed above).
Current ACTUAL Lightning proxy: 0.93 at ep830 (not auth-evaluated yet).

If Lightning proxy 0.93 translates to auth ~1.0-1.1, that is our new floor.

Quantizr at 0.60:
    seg: 100 * 0.003 = 0.300
    pose: sqrt(10 * 0.001) = 0.100
    rate: 25 * 0.00805 = 0.201
    check: 0.300 + 0.100 + 0.201 = 0.601

Gap analysis (our 1.33 vs Quantizr 0.60):
    seg gap:  0.610 - 0.300 = 0.310  (2x their SegNet)
    pose gap: 0.148 - 0.100 = 0.048  (1.5x their PoseNet)
    rate gap: 0.575 - 0.201 = 0.374  (2.9x their rate)

BIGGEST LEVER: rate (0.374 gap) then seg (0.310 gap) then pose (0.048 gap).

Shannon's information-theoretic bound:
    Scorer sufficient statistics ~200KB for SegNet + ~7KB for PoseNet.
    Theoretical minimum rate: 25 * (207KB / 1.5MB) = 25 * 0.00855 = 0.214.
    With optimal neural coding: possibly 25 * 0.005 = 0.125.
    Quantizr at 0.201 is already near-optimal. Their architecture IS the codec.

Our rate problem: we ship a VIDEO FILE. H.265 does not know about the scorer.
    847KB video = massive redundancy the scorer ignores.
    The GPU lane (constrained gen) eliminates the video entirely.
    If GPU lane works: rate drops from 0.575 to ~0.05-0.10.
"""


# ============================================================================
# SECTION 2: KILLED TECHNIQUES (Contrarian's graveyard)
# ============================================================================

KILLED: list[KilledTechnique] = [
    KilledTechnique(
        name="KL distillation loss",
        reason="Two auth evals confirmed PoseNet collapse (1.85, 2.05)",
        evidence="proxy 1.25 -> auth 1.85 (sw=100,cap); proxy 1.43 -> auth 2.05 (sw=30,no cap)",
        hours_wasted=12.0,
        lesson="Never trust proxy alone for loss mode changes. PoseNet is fragile.",
    ),
    KilledTechnique(
        name="Adaptive weight rebalancing (Hinton T^2)",
        reason="T^2 cancels in derivation. Formula vacuous. 125x mismatch with empirical winner.",
        evidence="Formula gave w_s=0.80 when winner used w_s=100",
        hours_wasted=4.0,
        lesson="Verify mathematical derivations end-to-end before implementing.",
    ),
    KilledTechnique(
        name="PoseNet gradient caps/clamps",
        reason="Caused 26x PoseNet regression",
        evidence="Direct A/B test showed catastrophic regression",
        hours_wasted=2.0,
        lesson="PoseNet signal is precious. Never attenuate it.",
    ),
    KilledTechnique(
        name="SIREN memorization (mask-conditioned)",
        reason="Smoke test: 21.7 dB PSNR, PoseNet 118.3 -- catastrophic",
        evidence="smoke_test_constrained_gen results 2026-04-12",
        hours_wasted=1.0,
        lesson="SIRENs need thousands of steps for high-res; not competitive for our budget.",
    ),
    KilledTechnique(
        name="PSD architecture (PixelShuffle-Downscale)",
        reason="Auth 1.49 vs dilated 1.33. Council verdict: stay with dilated.",
        evidence="Auth eval 2026-04-11",
        hours_wasted=6.0,
        lesson="Architecture changes need >10% improvement to justify switch cost.",
    ),
    KilledTechnique(
        name="AllNorm invariance exploitation",
        reason="Disproven. AllNorm is NOT truly invariant -- it affects gradient flow.",
        evidence="Experimental verification showed score changes with AllNorm perturbations",
        hours_wasted=2.0,
        lesson="Verify claimed invariances empirically before building on them.",
    ),
    KilledTechnique(
        name="segnet_loss_weight > 100",
        reason="Overwhelms PoseNet signal",
        evidence="Multiple training runs showed PoseNet collapse above sw=100",
        hours_wasted=3.0,
        lesson="PoseNet and SegNet losses compete. Balance is critical.",
    ),
    KilledTechnique(
        name="Hamiltonian dynamics optimizer",
        reason="Coupled trajectory + L-BFGS strictly dominates. Symplectic structure unnecessary.",
        evidence="Theoretical analysis: scorer landscape is not Hamiltonian-structured",
        hours_wasted=0.5,
        lesson="Physics analogies must match problem structure, not just sound cool.",
    ),
    KilledTechnique(
        name="Finance-inspired optimizers",
        reason="No evidence any finance analogy applies to pixel optimization",
        evidence="Contrarian review: analogies are superficial",
        hours_wasted=0.5,
        lesson="Cross-domain transfer needs structural isomorphism, not metaphor.",
    ),
    KilledTechnique(
        name="Cross-disciplinary ensemble (17 optimizers)",
        reason="Scatter-shot approach. Time budget allows 2-3 focused experiments, not 17.",
        evidence="20-day deadline arithmetic",
        hours_wasted=1.0,
        lesson="Focus beats breadth under time pressure.",
    ),
]


# ============================================================================
# SECTION 3: DEPLOY IMMEDIATELY (consensus: these ship this weekend)
# ============================================================================

DEPLOY_NOW: list[CouncilExperiment] = [
    CouncilExperiment(
        name="Auth-eval Lightning ep830 checkpoint",
        verdict=Verdict.DEPLOY_NOW,
        lane=Lane.CPU,
        priority=1,
        estimated_score="1.0-1.1 (if proxy 0.93 translates)",
        estimated_hours=0.5,
        timeline_days=0,
        platform="bat00 (RTX 2070S) or Lightning T4",
        yousfi_assessment=(
            "This is the SINGLE most important action. We have a proxy 0.93 "
            "checkpoint that has never been auth-evaluated. If it scores ~1.0 auth, "
            "it is our new floor and changes all downstream priorities."
        ),
        karpathy_assessment="Eval before building. Always.",
        contrarian_assessment=(
            "Agreed. But proxy-to-auth translation has been unreliable (29x PoseNet "
            "divergence from DALI vs PyAV). The 0.93 proxy may auth-score anywhere "
            "from 0.80 to 1.30. Do not assume 1.0."
        ),
        dependencies=["bat00 RTX 2070S setup OR Lightning DALI eval script"],
        risks=["DALI vs PyAV decode difference may surprise us again"],
    ),
    CouncilExperiment(
        name="CRF sweep (32-38) on current checkpoint",
        verdict=Verdict.DEPLOY_NOW,
        lane=Lane.CPU,
        priority=2,
        estimated_score="0.05-0.15 improvement from rate reduction",
        estimated_hours=0.5,
        timeline_days=0,
        platform="local M5 Max",
        yousfi_assessment=(
            "CRF 34 is current. Lower CRF = bigger file but better quality. "
            "Higher CRF = smaller file but worse quality. The optimum depends on "
            "the 25x rate multiplier vs quality degradation. This is FREE and FAST."
        ),
        contrarian_assessment=(
            "Agree. This is the lowest-hanging fruit. But the improvement is bounded: "
            "even optimal CRF saves at most 0.15 because rate is only 25x and our "
            "video is already reasonably compressed."
        ),
        projected_rate=0.45,  # optimistic: CRF 36 saves ~100KB
        dependencies=["ffmpeg with libx265"],
        risks=["Higher CRF may degrade PoseNet more than rate savings justify"],
    ),
    CouncilExperiment(
        name="Quantization-directed rounding",
        verdict=Verdict.DEPLOY_NOW,
        lane=Lane.CPU,
        priority=3,
        estimated_score="0.02-0.05 improvement (nearly free)",
        estimated_hours=0.5,
        timeline_days=0,
        platform="local M5 Max",
        yousfi_assessment=(
            "Scorer gradient sign at each pixel tells us ceil vs floor. "
            "~20KB bitmask in archive. The cost is 0.013 rate for ~0.03 score gain."
        ),
        fridrich_assessment=(
            "This is S-UNIWARD cost-weighted quantization. The steganographic "
            "community has refined this for 15 years. Apply per-pixel cost from "
            "scorer gradient magnitude, not just sign."
        ),
        contrarian_assessment=(
            "Marginal. 0.02-0.05 is real but small. The 20KB archive cost "
            "partially offsets the gain. Net benefit ~0.01-0.03."
        ),
        projected_seg=0.005,
        projected_pose=0.002,
        projected_rate=0.59,  # 20KB added to archive
        dependencies=["scorer models loadable on CPU"],
        risks=["Archive size increase may offset quality gain"],
    ),
    CouncilExperiment(
        name="TTO + multi-pass at inflate time (activate existing code)",
        verdict=Verdict.DEPLOY_NOW,
        lane=Lane.CPU,
        priority=4,
        estimated_score="0.03-0.10 improvement (code already written)",
        estimated_hours=0.1,
        timeline_days=0,
        platform="any (inflate-time optimization)",
        yousfi_assessment=(
            "INFLATE_TTO_STEPS=10 INFLATE_MULTI_PASS=2 -- already verified working. "
            "Self-supervised TTO adjusts postfilter weights at inflate time. "
            "Multi-pass runs the postfilter twice with uint8 rounding between. "
            "Both are FREE in archive size."
        ),
        contrarian_assessment=(
            "The code exists and was verified. But TTO without scorer access at "
            "inflate time uses proxy losses (TV, consistency). These may not "
            "correlate with actual scorer improvement. Measure before trusting."
        ),
        dependencies=["inflate_postfilter.py env var support"],
        risks=["Self-supervised TTO may not improve scorer-specific metrics"],
    ),
]


# ============================================================================
# SECTION 4: DEVELOP THIS WEEK (Apr 13-19)
# ============================================================================

DEVELOP_THIS_WEEK: list[CouncilExperiment] = [
    CouncilExperiment(
        name="Coupled trajectory optimization (GPU breakthrough)",
        verdict=Verdict.DEVELOP_THIS_WEEK,
        lane=Lane.GPU,
        priority=1,
        estimated_score="0.50-0.80 total score (paradigm shift)",
        estimated_hours=8.0,
        timeline_days=3,
        platform="Kaggle P100 or Lightning T4",
        yousfi_assessment=(
            "THIS is the path to beating Quantizr. Constrained optimization from "
            "noise: masks + pose targets + seed = 8KB archive. Rate drops from "
            "0.575 to ~0.05. Combined with good SegNet/PoseNet satisfaction, "
            "projected score 0.50-0.80. The Fridrich smoke test showed PoseNet "
            "controllable at 0.078 -- the framework WORKS, just needs more steps "
            "and SegNet constraint tuning."
        ),
        tao_assessment=(
            "The coupled trajectory formulation is mathematically sound. Frames "
            "are NOT independent -- PoseNet evaluates consecutive pairs, creating "
            "a coupled system. Joint optimization over all 1200 frames lets gradient "
            "flow through the full dependency graph. This is 4D-Var from weather "
            "prediction applied to video frames."
        ),
        fridrich_assessment=(
            "My constrained generation showed PoseNet 0.078 with only 500 steps. "
            "SegNet violated because the initialization was noise -- start from "
            "class-mean-colored images instead. The detection boundary formulation "
            "gives us exact constraint margins. Double the steps and add momentum."
        ),
        karpathy_assessment=(
            "Ship risk: convergence on 1200 frames on T4 is the bottleneck. "
            "Memory: 1200 * 3 * 874 * 1164 * 4 bytes = 14GB. T4 has 16GB. "
            "Tight but doable with gradient checkpointing. P100 has 16GB too. "
            "Use mixed precision (fp16 forward, fp32 grads) to fit."
        ),
        contrarian_assessment=(
            "HIGH RISK, HIGH REWARD. The smoke test proved the concept for PoseNet "
            "but FAILED for SegNet. If SegNet cannot converge in 1000 steps, the "
            "entire GPU lane is dead. Mitigation: start from class-mean images, "
            "not noise. This gives SegNet a 90% correct initialization."
        ),
        shannon_assessment=(
            "If this works, it changes the information-theoretic regime entirely. "
            "We go from encoding pixels (847KB video) to encoding sufficient "
            "statistics (8KB). Rate term drops from 0.575 to 0.005. That alone "
            "is a 0.57 score improvement. Even if SegNet/PoseNet are slightly "
            "worse, the rate savings dominate."
        ),
        projected_seg=0.004,  # slightly worse than video-based (0.006)
        projected_pose=0.003,  # worse than video-based (0.002) but manageable
        projected_rate=0.005,  # 8KB archive -> 25 * 0.0002 = 0.005
        dependencies=[
            "Scorer models loadable on CUDA",
            "Mask data from existing archive",
            "PoseNet targets computed from ground truth",
            "Kaggle or Lightning GPU session",
        ],
        risks=[
            "SegNet convergence from noise (mitigate: class-mean init)",
            "Memory on 16GB GPU (mitigate: gradient checkpointing + fp16)",
            "1200-frame optimization may be too slow (mitigate: chunk into batches)",
            "Auth scorer uses DALI decode -- our generated frames skip decode entirely",
        ],
    ),
    CouncilExperiment(
        name="DP-SIMS continued training (1000+ Phase 2 epochs)",
        verdict=Verdict.DEVELOP_THIS_WEEK,
        lane=Lane.GPU,
        priority=2,
        estimated_score="0.80-1.20 total score",
        estimated_hours=20.0,
        timeline_days=5,
        platform="Lightning T4 (if not used for constrained gen)",
        yousfi_assessment=(
            "DP-SIMS already matches Quantizr SegNet at 0.003 after only 89 epochs. "
            "PoseNet at 0.482 needs 1000+ more epochs. The architecture is validated "
            "but needs training time. This is the SAFE GPU bet -- we know it works, "
            "we just need compute."
        ),
        karpathy_assessment=(
            "Training 1000 epochs on T4 at ~3 min/epoch = 50 hours. We have 71 hours "
            "on Lightning. This consumes most of our free T4 budget. Worth it ONLY if "
            "constrained gen fails."
        ),
        contrarian_assessment=(
            "CONFLICT: this and constrained gen both need the T4. Cannot do both. "
            "Constrained gen has higher ceiling (0.50) but higher risk. DP-SIMS "
            "has lower ceiling (0.80) but is proven. RECOMMENDATION: give constrained "
            "gen 3 days. If it fails, pivot to DP-SIMS for the remaining 17 days."
        ),
        projected_seg=0.003,
        projected_pose=0.010,  # projection from learning curve
        projected_rate=0.150,  # ~225KB FP4 model in archive
        dependencies=["Lightning T4 session", "DP-SIMS Phase 2 resume checkpoint"],
        risks=["Consumes T4 budget that constrained gen needs", "PoseNet may plateau"],
    ),
    CouncilExperiment(
        name="L-BFGS refinement on best frames",
        verdict=Verdict.DEVELOP_THIS_WEEK,
        lane=Lane.GPU,
        priority=3,
        estimated_score="0.02-0.10 improvement on top of constrained gen",
        estimated_hours=2.0,
        timeline_days=1,
        platform="Kaggle P100",
        yousfi_assessment=(
            "Second-order optimization converges in 3 steps where Adam needs 1000. "
            "Run AFTER constrained gen produces initial frames. L-BFGS with strong "
            "Wolfe line search. Implicit Hessian via rank-10 quasi-Newton."
        ),
        tao_assessment=(
            "Mathematically: near a local minimum, the Hessian is approximately "
            "constant. L-BFGS approximates it from gradient history. Convergence "
            "is superlinear. This is the right polishing step."
        ),
        contrarian_assessment=(
            "Only useful if constrained gen gets close. If constrained gen fails "
            "to converge, L-BFGS on garbage frames is still garbage."
        ),
        dependencies=["Constrained gen output OR DP-SIMS output"],
        risks=["Depends on upstream success"],
    ),
    CouncilExperiment(
        name="Trick stacking: CRF + deblock + qround + TTO on CPU lane",
        verdict=Verdict.DEVELOP_THIS_WEEK,
        lane=Lane.CPU,
        priority=2,
        estimated_score="1.10-1.25 (improve current 1.33 floor)",
        estimated_hours=2.0,
        timeline_days=1,
        platform="local M5 Max",
        yousfi_assessment=(
            "Stack independent CPU improvements on existing 1.33 checkpoint. "
            "CRF sweep finds optimal rate point. Non-local means deblocking "
            "cleans compression artifacts. Quantization rounding aligns with "
            "scorer gradients. TTO fine-tunes at inflate time. These are "
            "independent and approximately additive."
        ),
        contrarian_assessment=(
            "Each trick is 0.02-0.05. Stacking 4 tricks gives 0.05-0.15 total "
            "(not fully additive due to interactions). This moves us from 1.33 "
            "to ~1.15-1.25. Decent but won't beat Quantizr. Still: this is our "
            "INSURANCE POLICY while the GPU lane develops."
        ),
        dependencies=["CRF sweep results", "scorer models on CPU"],
        risks=["Trick interactions may be negative (deblock could hurt PoseNet)"],
    ),
    CouncilExperiment(
        name="bat00 auth scoring setup",
        verdict=Verdict.DEVELOP_THIS_WEEK,
        lane=Lane.HYBRID,
        priority=1,
        estimated_score="N/A -- infrastructure",
        estimated_hours=2.0,
        timeline_days=1,
        platform="bat00 RTX 2070 Super",
        yousfi_assessment=(
            "CRITICAL INFRASTRUCTURE. Without local DALI+CUDA auth scoring, every "
            "experiment requires a Lightning session to eval. bat00 with RTX 2070S "
            "should support DALI. Set up once, use for all remaining auth evals."
        ),
        contrarian_assessment=(
            "2070S has 8GB VRAM. DALI + scorer may not fit. Test immediately. "
            "If it does not fit, we are stuck with Lightning for auth eval."
        ),
        dependencies=["bat00 SSH access", "NVIDIA DALI install on 2070S"],
        risks=["8GB VRAM may be insufficient for DALI + dual scorer"],
    ),
]


# ============================================================================
# SECTION 5: DEVELOP NEXT WEEK (Apr 20-26)
# ============================================================================

DEVELOP_NEXT_WEEK: list[CouncilExperiment] = [
    CouncilExperiment(
        name="Road plane homography constraint",
        verdict=Verdict.DEVELOP_NEXT_WEEK,
        lane=Lane.GPU,
        priority=3,
        estimated_score="0.01-0.03 improvement (geometric consistency)",
        estimated_hours=4.0,
        timeline_days=3,
        platform="Kaggle P100",
        yousfi_assessment=(
            "40% of pixels are road surface. Road plane is a known 3D surface. "
            "Given camera intrinsics (fx=910, pp=582,437), we can compute the "
            "homography for each frame pair. This LOCKS road pixels to geometric "
            "ground truth, reducing degrees of freedom for the optimizer."
        ),
        contrarian_assessment="Only matters if constrained gen is working. Premature otherwise.",
        dependencies=["Working constrained gen pipeline"],
    ),
    CouncilExperiment(
        name="Ego-motion flow as PoseNet target",
        verdict=Verdict.DEVELOP_NEXT_WEEK,
        lane=Lane.GPU,
        priority=4,
        estimated_score="0.01-0.05 (better PoseNet targets than ground truth MSE)",
        estimated_hours=3.0,
        timeline_days=2,
        platform="Kaggle P100",
        yousfi_assessment=(
            "Instead of storing raw PoseNet targets, compute expected PoseNet output "
            "from ego-motion trajectory (rotation + translation). This gives us "
            "physically grounded targets rather than empirical ones. May reduce "
            "PoseNet error because we optimize for the RIGHT answer, not just "
            "matching the scorer's answer on ground truth."
        ),
        dependencies=["Working constrained gen pipeline", "camera intrinsics"],
    ),
    CouncilExperiment(
        name="Scorer null-space rate optimization",
        verdict=Verdict.DEVELOP_NEXT_WEEK,
        lane=Lane.GPU,
        priority=5,
        estimated_score="0.02-0.05 (rate reduction without score change)",
        estimated_hours=2.0,
        timeline_days=1,
        platform="Kaggle P100",
        yousfi_assessment=(
            "After constrained gen converges, project pixel perturbations into "
            "scorer null space. Reduce total variation (-> compressibility) in "
            "directions the scorer cannot detect. This is the Fridrich steganographic "
            "insight applied in reverse."
        ),
        fridrich_assessment=(
            "Exactly. The null space of the Jacobian at the current frame IS the "
            "space of imperceptible (to the scorer) perturbations. Move in this "
            "space to minimize entropy (-> minimize compressed file size). This is "
            "inverse steganography."
        ),
        dependencies=["Working constrained gen output", "Jacobian SVD computation"],
    ),
    CouncilExperiment(
        name="VVC (H.266) for masks if available",
        verdict=Verdict.DEVELOP_NEXT_WEEK,
        lane=Lane.CPU,
        priority=6,
        estimated_score="0.01-0.02 (better mask compression)",
        estimated_hours=1.0,
        timeline_days=1,
        platform="local",
        yousfi_assessment=(
            "VVC achieves 30-50% better compression than H.265 for the same quality. "
            "If we use VVC for mask encoding in the archive, we save bytes. "
            "But this depends on VVC being available at inflate time."
        ),
        contrarian_assessment="Marginal. Masks are already tiny (239 bytes). Not worth the effort.",
    ),
]


# ============================================================================
# SECTION 6: KEEP ALIVE (monitor, do not invest heavily)
# ============================================================================

KEEP_ALIVE: list[CouncilExperiment] = [
    CouncilExperiment(
        name="Alternating projections (Dykstra/ADMM)",
        verdict=Verdict.KEEP_ALIVE,
        lane=Lane.GPU,
        priority=7,
        estimated_score="unknown",
        estimated_hours=4.0,
        timeline_days=3,
        platform="Kaggle P100",
        tao_assessment=(
            "Theoretically sound for convex constraints. Our constraints are "
            "quasi-convex at best. Convergence is not guaranteed but empirically "
            "often works. The alternating structure naturally handles the multi-"
            "objective nature (SegNet, PoseNet, rate as separate constraint sets)."
        ),
        contrarian_assessment=(
            "Coupled trajectory with Lagrangian multipliers does the same thing "
            "more naturally. ADMM adds complexity without clear benefit. Keep alive "
            "only if coupled trajectory shows constraint satisfaction issues."
        ),
    ),
    CouncilExperiment(
        name="BatchNorm statistics matching",
        verdict=Verdict.KEEP_ALIVE,
        lane=Lane.CPU,
        priority=8,
        estimated_score="0.01-0.03",
        estimated_hours=1.0,
        timeline_days=1,
        platform="local",
        yousfi_assessment=(
            "Gatys-style BN matching pushes generated frames toward the scorer's "
            "training distribution. Low effort, low risk, but also low expected gain."
        ),
        contrarian_assessment="Nice-to-have. Do after everything else.",
    ),
    CouncilExperiment(
        name="Brightness shift exploit",
        verdict=Verdict.KEEP_ALIVE,
        lane=Lane.CPU,
        priority=9,
        estimated_score="0.005-0.02",
        estimated_hours=0.2,
        timeline_days=0,
        platform="local",
        contrarian_assessment=(
            "5-minute grid search. Almost free. Do it, but expect nearly nothing."
        ),
    ),
]


# ============================================================================
# SECTION 7: STRATEGIC INSIGHTS (cross-council synthesis)
# ============================================================================

STRATEGIC_INSIGHTS: list[StrategicInsight] = [
    StrategicInsight(
        source="Shannon + Tao",
        insight=(
            "The fundamental information-theoretic insight: we are encoding SCORER "
            "SUFFICIENT STATISTICS, not video. The scorer extracts ~207KB of "
            "information from 1200 frames. Any archive larger than 207KB contains "
            "information the scorer ignores. Our 847KB video is 4x over the "
            "theoretical minimum. The GPU lane (constrained gen) fixes this by "
            "encoding only what the scorer needs."
        ),
        actionable=True,
        applied_to=["coupled_trajectory_optimize", "scorer_as_compressor"],
    ),
    StrategicInsight(
        source="Fridrich",
        insight=(
            "The competition IS inverse steganalysis. The scorer is a detector. "
            "We are not compressing video -- we are generating frames that satisfy "
            "a detection model. Every technique from steganographic security "
            "applies: cost maps (S-UNIWARD), detection boundaries, constrained "
            "embedding, null-space exploitation. The SegNet violation in our smoke "
            "test was because we started from noise. Starting from class-mean "
            "images gives SegNet a 90% correct initialization."
        ),
        actionable=True,
        applied_to=["constrained_gen", "quantization_directed_rounding"],
    ),
    StrategicInsight(
        source="Karpathy",
        insight=(
            "FOCUS. We have 20 days and 3 compute platforms. The roadmap has 18 "
            "experiments. We can realistically execute 4-5 well. The priority order "
            "is: (1) auth-eval current best, (2) CRF + trick stack on CPU, "
            "(3) constrained gen on GPU, (4) DP-SIMS as fallback. Everything else "
            "is noise unless (1)-(3) fail."
        ),
        actionable=True,
        applied_to=["all"],
    ),
    StrategicInsight(
        source="Contrarian",
        insight=(
            "HONEST ASSESSMENT: The proxy-to-auth translation gap is our biggest "
            "unknown. We had a 29x PoseNet divergence from DALI vs PyAV. Our "
            "proxy 0.93 could auth-score ANYWHERE from 0.70 to 1.50. Until we "
            "have reliable auth scoring (bat00 setup), every projection is fantasy. "
            "The FIRST priority is not a new experiment -- it is reliable auth eval."
        ),
        actionable=True,
        applied_to=["bat00_auth_setup", "auth_eval_ep830"],
    ),
    StrategicInsight(
        source="Yousfi",
        insight=(
            "Quantizr's 8 blind spots we can exploit: (1) no ego-motion flow, "
            "(2) no road plane homography, (3) no vanishing point loss, "
            "(4) no temporal coherence from trajectory, (5) no BN matching, "
            "(6) no null-space projection, (7) no detection boundary, "
            "(8) no per-pixel cost weighting. Stack these on constrained gen "
            "for systematic advantage."
        ),
        actionable=True,
        applied_to=["coupled_trajectory_optimize"],
    ),
    StrategicInsight(
        source="Noether",
        insight=(
            "AllNorm invariance was disproven but there ARE other scorer symmetries: "
            "(1) PoseNet is invariant to global brightness at the 7x7 patch level "
            "(local contrast normalization). (2) SegNet argmax is invariant to "
            "logit scaling (softmax temperature). (3) Both are approximately "
            "invariant to small spatial translations (< 1 pixel). These symmetries "
            "define the scorer null space and can be exploited for rate reduction."
        ),
        actionable=True,
        applied_to=["scorer_null_space", "brightness_shift_exploit"],
    ),
    StrategicInsight(
        source="Ramanujan + Gauss",
        insight=(
            "Pattern in scorer behavior: PoseNet MSE is dominated by ~5% of frame "
            "pairs (high-motion turns). SegNet error is dominated by ~10% of pixels "
            "(class boundaries). The score is NOT uniformly distributed across frames "
            "or pixels. A budget-optimal strategy allocates quality to these critical "
            "regions and degrades gracefully elsewhere."
        ),
        actionable=True,
        applied_to=["semantic_quantization", "hard_frame_curriculum"],
    ),
    StrategicInsight(
        source="Planck",
        insight=(
            "Quantization is fundamental. The scorer operates on uint8 images "
            "(256 levels). The minimum perturbation is 1/255. The maximum number "
            "of distinct scorer inputs is 256^(3*H*W) per frame. But the scorer's "
            "effective resolution is much lower (7x7 patches for PoseNet, 96x128 "
            "for SegNet). The effective search space per frame is ~256^(96*128*5) "
            "for SegNet, which is still astronomical. Gradient descent is the only "
            "tractable approach."
        ),
        actionable=False,
        applied_to=["theoretical_understanding"],
    ),
    StrategicInsight(
        source="Gauss + LeCun",
        insight=(
            "The GPU lane's class-mean initialization is crucial. For a 5-class "
            "SegNet with known masks, starting each pixel at the class mean color "
            "gives ~90% correct SegNet prediction BEFORE any optimization. This "
            "turns the SegNet constraint from hard to easy, letting the optimizer "
            "focus on PoseNet (the harder constraint). LeCun: this is analogous to "
            "curriculum learning -- solve the easy task first."
        ),
        actionable=True,
        applied_to=["constrained_gen"],
    ),
]


# ============================================================================
# SECTION 8: EXECUTION TIMELINE
# ============================================================================

TIMELINE = """
=== WEEKEND Apr 12-13 (NOW) ===
    [P0] Auth-eval Lightning ep830 checkpoint on bat00 or Lightning
    [P0] CRF sweep (32-38) on local M5 Max
    [P0] bat00 auth scoring setup (DALI + CUDA + scorer)
    [P1] Activate TTO + multi-pass on current inflate pipeline
    [P1] Quantization-directed rounding implementation + eval

=== Week 1: Apr 14-19 ===
    [P0] Constrained gen: class-mean init + 1000 steps on Kaggle P100
    [P0] Constrained gen: coupled trajectory (all 1200 frames) on Lightning T4
    [P1] Trick stacking on CPU lane (CRF + deblock + qround + TTO)
    [P1] Auth-eval every CPU lane improvement
    [P2] L-BFGS refinement on constrained gen output
    DECISION POINT (Apr 17): If constrained gen shows SegNet convergence,
        commit to GPU lane. If not, pivot all GPU compute to DP-SIMS.

=== Week 2: Apr 20-26 ===
    [P0] Whichever GPU lane was chosen: train to convergence
    [P1] Add geometric constraints (road homography, ego-motion flow)
    [P1] Null-space rate optimization
    [P2] Final trick stacking on CPU lane
    DECISION POINT (Apr 23): Feature freeze. Pick best candidate for submission.

=== Week 3: Apr 27-May 2 ===
    [P0] Final auth eval on DALI scorer
    [P0] Submission packaging and verification
    [P1] Last-minute TTO/multi-pass tuning at inflate time
    [DEADLINE] May 3

=== Compute Budget ===
    Lightning T4: 71 hours -> 23 hours/week for 3 weeks
    Kaggle P100:  30 hours/week -> 90 hours total
    Local M5 Max: unlimited (CPU only, no CUDA)
    bat00 2070S:  unlimited (auth eval + small experiments)
"""


# ============================================================================
# SECTION 9: CONTRARIAN'S FINAL HONEST ASSESSMENT
# ============================================================================

CONTRARIAN_FINAL = """
=== The Contrarian's Honest Assessment ===

WHAT WILL ACTUALLY WORK IN 20 DAYS:

1. CRF sweep: YES. Low effort, guaranteed small gain. Do it today.
   Expected: 0.02-0.08 score improvement. Confidence: 95%.

2. Trick stacking (CPU lane): YES. Each trick is small but they compound.
   Expected: 0.05-0.15 total improvement on 1.33 baseline. Confidence: 80%.
   New floor: ~1.15-1.25.

3. Constrained gen (GPU lane): MAYBE. The concept is proven (PoseNet controlled)
   but SegNet convergence is unproven at scale. Class-mean init is the key fix.
   Expected IF it works: 0.50-0.80 total score. Confidence: 40%.
   Expected if it fails: wasted 20 hours of GPU time.

4. DP-SIMS continued training: YES (but slow). Known to work, just needs epochs.
   Expected: 0.80-1.20 at 1000 epochs. Confidence: 70%.
   Problem: consumes all T4 budget.

5. Everything else on the roadmap: NO. Not in 20 days.
   Kill list: Hamiltonian dynamics, finance optimizers, variational frame gen,
   Lagrangian dual, scorer manifold geodesic, cross-disciplinary ensemble.
   These are interesting research but not competition-winning.

REALISTIC SCENARIOS:

Best case (20% probability):
    Constrained gen works. Score 0.50-0.60. We match or beat Quantizr.
    This requires: SegNet convergence, PoseNet convergence, rate as projected.

Likely case (50% probability):
    Constrained gen partially works (PoseNet ok, SegNet 2x worse than video).
    CPU lane improves to 1.15. GPU lane scores 0.90-1.10.
    We submit GPU lane at ~1.0. Significant improvement but Quantizr still wins.

Worst case (30% probability):
    Constrained gen fails. DP-SIMS needs more epochs than we have.
    CPU lane improves to 1.20. We submit 1.20. Quantizr at 0.60 wins by 2x.

THE UNCOMFORTABLE TRUTH:
    Quantizr's mask2mask architecture is fundamentally better suited to this
    competition than a video codec + postfilter. They encode scorer-optimal
    representations directly. We encode video and then try to fix it.
    The constrained gen approach is our only path to matching their paradigm.
    If it fails, we are structurally disadvantaged.

WHAT I WOULD BET ON:
    Final submission score: 0.90-1.10 (60% confidence interval).
    This assumes constrained gen partially works and CPU tricks help.
    To beat Quantizr (0.60), we need constrained gen to fully work: 40% chance.
"""


# ============================================================================
# SECTION 10: COUNCIL CONSENSUS VOTES
# ============================================================================

CONSENSUS = {
    "top_priority": "Auth-eval Lightning ep830 checkpoint IMMEDIATELY",
    "gpu_lane_bet": "Constrained gen (3-day trial, pivot to DP-SIMS if fails by Apr 17)",
    "cpu_lane_bet": "Trick stacking (CRF + deblock + qround + TTO) as insurance",
    "infrastructure": "bat00 auth scoring setup is BLOCKING -- do it first",
    "kill_count": len(KILLED),
    "deploy_now_count": len(DEPLOY_NOW),
    "develop_this_week_count": len(DEVELOP_THIS_WEEK),
    "keep_alive_count": len(KEEP_ALIVE),
    "realistic_final_score": "0.90-1.10 (60% CI)",
    "best_case_score": "0.50-0.60 (20% probability)",
    "unanimous_agreement": [
        "Auth eval before new experiments",
        "Constrained gen is highest-ceiling path",
        "CPU lane trick stacking is safe insurance",
        "Kill all Priority 5 experiments",
        "bat00 auth scoring is blocking infrastructure",
    ],
    "disagreements": [
        {
            "topic": "DP-SIMS vs constrained gen resource allocation",
            "yousfi": "All GPU to constrained gen -- ceiling is 3x higher",
            "karpathy": "Split: constrained gen on Kaggle, DP-SIMS on Lightning",
            "contrarian": "Constrained gen for 3 days, pivot if no SegNet convergence",
            "resolution": "Contrarian's 3-day trial wins (majority vote)",
        },
        {
            "topic": "How much CPU lane investment",
            "yousfi": "Minimal -- GPU lane is the paradigm shift",
            "contrarian": "Heavy -- GPU lane may fail, CPU is insurance",
            "resolution": "2 hours for trick stacking, rest to GPU (compromise)",
        },
    ],
}


# ============================================================================
# PUBLIC API
# ============================================================================

def get_priority_experiments() -> list[CouncilExperiment]:
    """Return all experiments sorted by priority."""
    all_exps = DEPLOY_NOW + DEVELOP_THIS_WEEK + DEVELOP_NEXT_WEEK + KEEP_ALIVE
    return sorted(all_exps, key=lambda e: e.priority)


def get_killed() -> list[KilledTechnique]:
    """Return all killed techniques."""
    return KILLED


def get_insights() -> list[StrategicInsight]:
    """Return all strategic insights."""
    return STRATEGIC_INSIGHTS


def print_summary():
    """Print a human-readable summary of the grand synthesis."""
    print("=" * 72)
    print("GRAND COUNCIL SYNTHESIS -- 2026-04-12")
    print("=" * 72)

    print(f"\nKILLED: {len(KILLED)} techniques")
    for k in KILLED:
        print(f"  [X] {k.name}: {k.reason}")

    print(f"\nDEPLOY NOW: {len(DEPLOY_NOW)} experiments")
    for e in DEPLOY_NOW:
        print(f"  [!] P{e.priority} {e.name} ({e.estimated_score})")

    print(f"\nDEVELOP THIS WEEK: {len(DEVELOP_THIS_WEEK)} experiments")
    for e in DEVELOP_THIS_WEEK:
        print(f"  [>] P{e.priority} {e.name} ({e.estimated_score})")

    print(f"\nDEVELOP NEXT WEEK: {len(DEVELOP_NEXT_WEEK)} experiments")
    for e in DEVELOP_NEXT_WEEK:
        print(f"  [~] P{e.priority} {e.name} ({e.estimated_score})")

    print(f"\nKEEP ALIVE: {len(KEEP_ALIVE)} experiments")
    for e in KEEP_ALIVE:
        print(f"  [?] P{e.priority} {e.name} ({e.estimated_score})")

    print(f"\nSTRATEGIC INSIGHTS: {len(STRATEGIC_INSIGHTS)}")
    for s in STRATEGIC_INSIGHTS:
        print(f"  [{s.source}] {s.insight[:80]}...")

    print("\n" + TIMELINE)
    print(CONTRARIAN_FINAL)


if __name__ == "__main__":
    print_summary()
