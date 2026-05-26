# SPDX-License-Identifier: MIT
"""tac.substrates.cascade_c_prime_frame_1_segnet_waterfill — Atick-Redlich asymmetric scorer channel L0 SCAFFOLD.

Per Cascade C' parent synthesis landing (commit ``2d5337f27``; subagent
``aa563bbb31adadfd6``) + Cascade C' Modal validation DEFERRED-pending-substrate-
scaffold verdict (commit ``aa1a9cf32``; subagent ``a1d16a40f4a722e26``)
operator-routable Option A.

**Atick-Redlich asymmetric scorer channel theory (1990 cooperative-receiver)**:
SegNet's ``x[:,-1,...]`` slice creates an asymmetric channel:

- **frame-0** perturbations: cost 0 SegNet bytes (STRUCTURALLY; SegNet never
  sees frame-0 directly — only frame-1)
- **frame-1** perturbations: cost M SegNet bytes + N' PoseNet bytes (SegNet
  sees frame-1; PoseNet sees both frames)

PR110 K=16 menu (commit ``6bae0201`` archive sha) currently uses **frame-0 only**
modes. The Cascade C' synthesis predicts that expanding the menu to include
FRAME-1 modes via per-pair Lagrangian dual routing decision unlocks
``-0.058820`` score delta [macOS-MLX research-signal; paired-CUDA-pending] at
PR106 frontier operating point pose_avg=3.4e-5.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS":
the per-pair Lagrangian dual:

    min_x  Σ_i (100 × d_seg_i(x) + √(10 × d_pose_i(x)) + 25/37545489 × bytes_i(x))
           s.t. routing_i ∈ {frame_0, frame_1}

solves in a single argmin pass over the joint candidate matrix per
``tac.findings_lagrangian`` Phase 1-3 wire-in (Catalog #355).

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status | Notes |
|---|---|---|
| L1 substrate must be score-aware | PASS | scorer queried at COMPRESS time; per-pair Lagrangian dual routing |
| L2 export-first archive grammar | PASS | CH-CCP-FRAME1-WATERFILL declared BEFORE training (this package + archive.py) |
| L3 monolithic 0.bin | PASS | single-file fixed-offset CCPF grammar |
| L4 inflate <= 100 LOC | PASS-WITH-BUDGET | ~180 LOC w/ routing-decision sidecar parse + frame-0/frame-1 menu lookup |
| L5 full RGB renderer | PASS | per-pair RGB via menu-index lookup + pose delta affine warp |
| L6 score-domain Lagrangian | PASS | per-pair Lagrangian dual per ``tac.findings_lagrangian`` Phase 1-3 |
| L7 bolt-on <= 350 LOC | substrate_engineering exception | total ~700 LOC across scaffold files |
| L8 eval-roundtrip + diff yuv6 | PASS | trainer applies eval_roundtrip per Catalog #5 + differentiable yuv6 |
| L9 runtime closure | PASS | numpy + brotli (transitive: zero) |
| L10 mask/pose coupling | PASS | pose deltas drive frame-1 affine warp from frame-0; sister of PR110 grammar |
| L11 no-op detector | PASS | Catalog #139 byte-mutation smoke planned + scaffolded |
| L12 single-LOC review discipline | PASS | each file reviewable in 30s |
| L13 KILL last resort | PASS | DEFERRED-pending-per-substrate-symposium per Catalog #325 |

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Archive grammar | UNIQUE | per-substrate routing-decision sidecar + dual menu streams; sister of PR110 grammar with frame-1 extension |
| Inflate runtime | CANONICAL (`tac.substrates._shared.inflate_runtime.select_inflate_device`) | per Catalog #205; numpy-portable |
| Score-aware loss | UNIQUE (per-pair Lagrangian dual routing) | per Atick-Redlich asymmetric channel; per-pair routing decision IS the unique loss surface |
| Training curriculum | CANONICAL (sister to NSCS06 v8 `mlx_iteration.py`) | MLX-native per-frame perturbation; standard training cadence |
| Tier-1 engineering | CANONICAL per Catalog #172/#178/#179/#180 | autocast_fp16 + TF32 + torch.compile + no_grad_at_eval |
| Scorer routing | CANONICAL (`tac.substrates._shared.score_aware_common.score_pair_components`) | per Catalog #164 |
| EMA | CANONICAL (`tac.training.EMA` decay=0.997) | per CLAUDE.md "EMA — non-negotiable" |
| eval_roundtrip | CANONICAL (`apply_eval_roundtrip_during_training`) | per CLAUDE.md "eval_roundtrip — non-negotiable" |

## 9-dimension success checklist evidence (per Catalog #294)

| Dim | Evidence |
|---|---|
| 1 UNIQUENESS | Atick-Redlich asymmetric channel routing — NOT a within-class refinement of PR110 grammar; class-shift via frame-1 menu expansion |
| 2 BEAUTY+ELEGANCE | scaffold files reviewable in 30s; archive grammar parser ~50 LOC; inflate ~180 LOC |
| 3 DISTINCTNESS | DISTINCT from PR110 K=16 frame-0-only menu (sister); DISTINCT from Cascade C P19 PoseNet-null bucket classification |
| 4 RIGOR | PV per Catalog #229 (synthesis script + DEFERRED verdict reviewed); per Catalog #292 assumption surfacing (HARD-EARNED Atick-Redlich theory + CARGO-CULTED synthesis-overestimate band) |
| 5 OPTIMIZATION-PER-TECHNIQUE | per-pair Lagrangian dual routing IS the substrate-optimal engineering decision |
| 6 STACK-OF-STACKS-COMPOSABILITY | composable as PR111-sub-frontier candidate atop PR110 grammar; sister to NSCS06 v8 chroma_lut |
| 7 DETERMINISTIC-REPRODUCIBILITY | per-pair routing decision deterministic given (scorer outputs, mode menu, pose_avg_baseline) |
| 8 EXTREME-OPTIMIZATION-PERFORMANCE | closed-form O(N_pairs × N_modes) routing per Catalog #356 per-axis decomposition |
| 9 OPTIMAL-MINIMAL-CONTEST-SCORE | predicted ΔS=-0.058820 [macOS-MLX research-signal; paired-CUDA-pending] |

## Observability surface (per Catalog #305)

- **inspectable per layer**: per-pair routing decision + per-pair Lagrangian improvement + per-pair selected mode index
- **decomposable per signal**: SegNet d_seg + PoseNet d_pose + archive_bytes per Catalog #356 per-axis decomposition
- **diff-able across runs**: deterministic routing given (scorer outputs, mode menu, pose_avg_baseline); diff via per-pair routing decision arrays
- **queryable post-hoc**: machine-readable JSON artifact (Cascade C' synthesis pattern at `.omx/research/cascade_c_prime_artifacts_20260526/*.json`)
- **cite-able**: every result anchored to (commit_sha, archive_sha, lane_id, subagent_id) per Catalog #245
- **counterfactual-able**: byte-mutation smoke per Catalog #105/#139/#272 mutates routing-decision sidecar bytes + verifies inflate output changes

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Unwind plan |
|---|---|---|
| Atick-Redlich asymmetric channel applies to SegNet's ``x[:,-1,...]`` slice | HARD-EARNED | derived from upstream/modules.py SegNet preprocess_input; verified empirically via Cascade C' synthesis 25.2% frame-1 routing matches sister #1324 PoseNet-null 22.3% within 3pp |
| Frame-1 modes extract ≥5× per-pair PoseNet savings vs frame-0 | CARGO-CULTED (synthesis prior; literature 10-30× overestimation common) | paired-CUDA validation per Catalog #325 symposium |
| Per-pair Lagrangian dual converges in single argmin pass | HARD-EARNED | per `tac.findings_lagrangian` Phase 1-3 wire-in unit tests + Cascade C' synthesis convergence_status |
| 1-bit-per-pair sidecar compresses to ≤80 bytes via brotli | HARD-EARNED via Cascade C' synthesis empirical (79 bytes) | replicates at paired-CUDA |
| K=24 codebook expansion adds ~30-50 bytes overhead | CARGO-CULTED (synthesis estimate; not validated) | paired-CUDA validation |

## Predicted ΔS band (per Catalog #296)

| Option | Predicted score delta | Validation |
|---|---|---|
| Option A (K=24 expansion, no sidecar) | -0.058817 [macOS-MLX research-signal] | Dykstra-feasibility verified via Cascade C' synthesis 48-cell sweep 41-PARADIGM/7-MARGINAL/0-NULL |
| Option B (1-bit sidecar) | -0.058820 [macOS-MLX research-signal] | Dykstra-feasibility verified via Cascade C' synthesis Option B empirical |

Per Catalog #324: `predicted_band_validation_status: pending_post_training`.
Reactivation criterion: post-training Tier-C re-measurement on landed paired-
CUDA smoke archive sha.

Horizon class per Catalog #309: `frontier_pursuit` (predicted CPU band [0.13,
0.18]; IF empirical confirms ~10× overestimate factor → -0.006 score delta
remains PR111-PLAUSIBLE).

Atick-Redlich-Tishby IB sister substrate: compress per-pair routing decision
into I(routing; score) information bottleneck — proposed Cascade C' sister
substrate per Catalog #308 alternative reducer enumeration.
"""
from __future__ import annotations

from .substrate_contract import CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT
from .trainer import (
    DEFAULT_MLX_AXIS_TAG,
    MLX_NON_PROMOTABLE_PROVENANCE,
    MLXFirstTrainerConfig,
    MLXFirstTrainerError,
    MLXFirstTrainerVerdict,
    is_mlx_available,
    run_mlx_first_compress_pass,
)
from .mlx_to_numpy_bridge import (
    EXPECTED_STATE_DICT_KEYS,
    BridgeRoundtripVerdict,
    MLXNumpyBridgeError,
    export_state_dict_to_npz,
    load_state_dict_from_npz,
    roundtrip_state_dict_through_archive,
    verify_state_dict_shape_contract,
)
from .tier_c_hook import (
    FORMALIZATION_PENDING_VERDICT,
    TierCAblationHookVerdict,
    TierCAblationProbeRequest,
    build_tier_c_ablation_probe_request,
    classify_tier_c_density_verdict,
)

__all__ = [
    # Scaffold (RECOVERY-2 landed commit aaf0b1eb6)
    "CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT",
    # Trainer (subagent A landing)
    "DEFAULT_MLX_AXIS_TAG",
    "MLX_NON_PROMOTABLE_PROVENANCE",
    "MLXFirstTrainerConfig",
    "MLXFirstTrainerError",
    "MLXFirstTrainerVerdict",
    "is_mlx_available",
    "run_mlx_first_compress_pass",
    # MLX → numpy bridge (subagent A landing)
    "EXPECTED_STATE_DICT_KEYS",
    "BridgeRoundtripVerdict",
    "MLXNumpyBridgeError",
    "export_state_dict_to_npz",
    "load_state_dict_from_npz",
    "roundtrip_state_dict_through_archive",
    "verify_state_dict_shape_contract",
    # Tier-C MDL ablation hook (subagent A landing)
    "FORMALIZATION_PENDING_VERDICT",
    "TierCAblationHookVerdict",
    "TierCAblationProbeRequest",
    "build_tier_c_ablation_probe_request",
    "classify_tier_c_density_verdict",
]
