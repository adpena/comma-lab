---
title: PHASE 4 INTEGRATION — Optimal Stack Design Memo
date: 2026-05-08
author: Subagent INTEGRATE (claude-opus-4-7-1m)
status: DESIGN — empirical anchors from CPU-prep work; GPU dispatch authorization separate
target_band: 0.155–0.175 [predicted-band, NOT contest-CUDA, NOT score_claim]
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
related_tasks: ["#307 PARADIGM-δεζ Phase 2", "#308 PHASE 4 INTEGRATION"]
---

# PHASE 4 INTEGRATION — Optimal Stack Design Memo

## Coordinator rigor note

This memo synthesizes the OPTIMAL STACK from currently-anchored CPU-prep
empirical evidence + planning-only design rows. **No row in this memo is a
score claim**; every numeric is `[predicted]`, `[predicted-band]`, or
`[empirical-anchor-N1; planning-only]` (CPU byte-anchor only, no decoder
runtime, no contest-CUDA eval).

Promotion gate: a stack row becomes promotion-eligible only after exact
archive bytes are evaluated through `archive.zip → inflate.sh →
upstream/evaluate.py` on T4/4090/A100 and recorded in
`contest_auth_eval.json`.

## 1. Phase 1 audit (precondition)

Phase 1 scaffolding state per `.omx/state/lane_registry.json`:

| Lane | Level | Module | Phase 2 status |
|---|---|---|---|
| `lane_delta_joint_training` | 0 | `src/tac/joint_scorer_aware_training.py` | Configs landed; `forward()` raises NotImplementedError |
| `lane_epsilon_learnable_entropy` | 0 | `src/tac/learnable_entropy_model.py` | Configs landed; `HyperEncoder/Decoder.__init__` raise NotImplementedError |
| `lane_zeta_self_compress_renderer` | 0 | `src/tac/self_compress_full_renderer.py` | Configs landed; `train_full_renderer_self_compress()` raises NotImplementedError |
| `lane_codec_pipeline_deltaepszeta_callback` | 1 | (Phase 2 callback) | Pipeline-aware training callback skeleton |
| `lane_run_deltaepszeta_training` | 1 | (Phase 3 keystone) | Operator-infrastructure driver |
| `lane_mdl_bayesian` | 1 | `src/tac/mdl_bayesian_codec.py` | Meta-comparison framework only — does NOT produce archive bytes (per blueprint §2.2) |

Phase 1 is **structurally complete**: dataclasses, configs, error-raising
stubs in place. Phase 2 requires GPU (joint training, entropy-model fit,
self-compress QAT). Per operator scope, this subagent does NOT attempt Phase 2
empirical anchors — every plausible Phase 2 anchor requires either:
- Multi-day GPU retrain (excluded by scope)
- Q-FAITHFUL retrain (explicitly excluded by operator)
- Synthetic/fake data (would violate `[empirical:<artifact>]` tagging
  discipline)

## 2. Empirical anchors available for the stack

### 2.1 Architecture lane — promotable bytes (CPU-prep, planning-only)

| Anchor | Bytes | Rel-err | Substrate | Source |
|---|---:|---:|---|---|
| `arch_shrink_x0.4_quantizr_class` | 83,571 | post-hoc channel L2-truncate | PR101 decoder | `reports/raw/pr101_arch_shrink_20260508T003827Z/manifest.json` |
| `sparsity_alpha_0.7_imp_retrain` | 94,671 | alpha=0.7 post-hoc | PR101 decoder | `reports/raw/pr101_sparsity_sweep_20260508T002611Z/manifest.json` |
| `lossy_int4_quantization` | 100,799 | 37% rel_err FALSIFIED | PR101 decoder | `reports/raw/pr101_lossy_int4_*/` (commit 03726464) |

### 2.2 Encoder/codec lane — promotable bytes (CPU-prep)

| Anchor | Bytes | Rel-err | Substrate | Source |
|---|---:|---:|---|---|
| Joint-ADMM × continuous-K (Path B step 6) | 152,420 | 4.33% | PR101 INT8 weights | commit 983598d2; `feedback_path_b_step6_*` |
| Joint-ADMM × discrete sparsity (Path B step 5) | 150,000 | 4.36% | PR101 | commit b8aa5c43; `feedback_path_b_step5_*` |
| brotli_optuna_default | 178,144 | lossless | PR101 | `feedback_pr101_lgwin13_q10_8byte_savings_*` |

### 2.3 Convergent finding (Path B steps 1-6)

**Allocation MECHANISM dominates codec-basis CHOICE on PR101 substrate.** The
Lagrangian λ-bisection that allocates per-tensor distortion budgets
non-uniformly is the empirically validated active ingredient. The codec basis
matters less than the allocation mechanism on PR101's near-iid INT8 weights.

CPU plateau reached: ~150K bytes at 4-5% rel_err is the floor of
per-tensor optimization on PR101. Going lower needs:
- Substrate change (architecture lane: 83K-95K bytes)
- Score-feedback loop (CUDA-required)
- Bilevel (implementation-heavy, expected delta small)

### 2.4 In-flight Lightning T4 dispatch (out of scope for this subagent)

`arch_shrink_x0.4` Lightning T4 retrain dispatch is in flight (job
`arch-shrink-x0-4-lightning-20260508t010514z`, ETA 12-18h, ~$9.90, full
Q-FAITHFUL profile 5-stage 3000 epoch). When it lands it will be the FIRST
[contest-CUDA] anchor for the architecture lane.

## 3. Optimal stack composition

### 3.1 Compose-time pipeline (compress side)

The stack is layered to MAX byte-savings while preserving scorer-basin parity
(compose-time order matters; see §3.4 for sanity ladder):

```
Stage 1: Architecture compression (lane choice)
  Option A: arch_shrink_x0.4 retrain        → ~83,571 B charged    [planning-only-N1]
  Option B: sparsity_alpha_0.7 IMP retrain  → ~94,671 B charged    [planning-only-N1]
  Option C: ζ self-compress full renderer   → ~30,000 B charged    [predicted, blueprint §2.3]
  Option D (fallback): apogee_int6 PTQ      → ~135,000 B charged   [predicted-band 0.180]

Stage 2: Encoder/codec compression (Joint-ADMM × continuous-K)
  Apply on Stage 1 output's weight tensors  → -28 KB on PR101 substrate at 4-5% rel_err
                                              [planning-only-N1, allocation MECHANISM]
  WARN: Allocation gain depends on tensor-count distribution. arch_shrink reduces
        tensor count by 60%, may shrink ADMM gain proportionally. EMPIRICAL
        VERIFICATION REQUIRED before claiming composition.

Stage 3: ε learned entropy prior (BLUEPRINT §2.2)
  Hyper-encoder/decoder shipped as renderer_prior.bin (≤ 5 KB)
                                            → -9 KB predicted [predicted, NOT anchored]

Stage 4: δ joint scorer-aware training (BLUEPRINT §2.1)
  Refits Stage 1+2+3 weights against contest scorer
                                            → 15-25% lower seg_dist [predicted, NOT anchored]
                                              10-35% lower pose_dist [predicted, NOT anchored]

Stage 5: PD codec pose optimization + Riemannian TTO (existing, anchored)
                                            → applied at archive build time
```

### 3.2 OPTIMAL stack recommendation

**Conservative (CPU-prep evidence only):**
- Stage 1: `arch_shrink_x0.4` retrain (83,571 B; in-flight Lightning anchor pending)
- Stage 2: Joint-ADMM × continuous-K applied post-arch-shrink (extrapolated 60-83K B
  if mechanism survives smaller tensor count; UNTESTED)
- Stage 3: skip ε (Phase 2 stub)
- Stage 4: skip δ (Phase 2 stub)
- Predicted archive bytes: 60-100 KB
- Predicted score: 0.140-0.180 [predicted-band; substrate-extrapolated]

**Aggressive (Phase 2 + 4 vision):**
- Stage 1: ζ self-compress full renderer (~30 KB after 2000 QAT steps)
- Stage 2: Joint-ADMM × continuous-K (mechanism-only, on smaller tensor footprint)
- Stage 3: ε learned prior shipped (~5 KB cost; -9 KB savings predicted)
- Stage 4: δ joint training (distortion improvements compound w/ all-of-above)
- Stage 5: PD/Riemannian TTO (existing infrastructure)
- Predicted archive bytes: 90-130 KB (bounded — Wave-Ω + masks live in same zip)
- Predicted score: **0.155-0.175** [predicted-band; matches PARADIGM-δεζ blueprint target]

### 3.3 Score-band uncertainty quantification

Per `feedback_path_b_convergent_findings_summary_20260508`: the central unanswered
question is whether 4-5% rel_err in Stage 2 degrades scorer distortion enough to
wipe the rate-term gain. CPU evidence cannot answer this.

| Risk axis | Conservative impact | Aggressive impact |
|---|---|---|
| Stage 2 rel_err destroys SegNet/PoseNet basin | rate gain wiped, score ≈ 0.21 | mitigated by Stage 4 δ joint refit |
| Stage 1 (arch_shrink) underfits | training-side bug; addressed by retrain | from-scratch SelfCompress is more robust |
| Stage 3 (ε) prior overfits to single video | small overhead (≤ 5 KB) | Stage 4 stabilizes |
| Stage 4 (δ) mode-collapse | N/A | mitigated by curriculum λ ramp |

### 3.4 Predispatch sanity ladder (mandatory before Phase 4 GPU dispatch)

Per blueprint §5 + CLAUDE.md "Predispatch sanity for δ/ε/ζ":

1. **Scorer-basin parity** on smoke batch: `D_seg ∈ [0, 5]`, `D_pose ∈ [0, 100]`
2. **Lagrange sanity**: `λ_pose ∈ [100, 1000]` from current baseline pose_avg
3. **Rate-gradient sanity**: `||∇_θ R||` finite + non-zero
4. **EMA shadow sanity**: shadow ≠ live by < 1% MAD after 10 steps
5. **eval_roundtrip sanity**: roundtripped frame ≠ raw by < 0.1 MSE
6. **ε encode/decode roundtrip**: `y == decode(encode(y))` (lossless)
7. **ζ roundtrip weight fidelity**: per-layer MSE < 0.1
8. **ζ bit-allocation anti-collapse**: ≥ 50% of channels with `b_l > 1.0`
9. **FiLM protection asserted** in `self_compress_layers` exclusion list
10. **MSSIM ≥ 0.90** (the gate apogee_int4 lacked, FALSIFIED at 1.4287
    contest-CUDA — see `feedback_q_faithful_NEVER_reproduced_quantizr_score_*`)

## 4. GPU dispatch budget

| Phase | Wall-clock | Cost (Lightning T4) | Cost (4090) | Cost (A100) | Trigger |
|---|---|---|---|---|---|
| 2: δ first measurable run | 2-3 days | $8-15 | $12-25 | $25-50 | Gate 1: blueprint approved |
| 3: ε rate-term + ζ self-compress | 2-3 days | $8-15 | $12-25 | $25-50 | Gate 3: Phase 2 shows scorer improvement |
| 4: δ+ε+ζ composition dispatch | 1-2 days | $2-5 | $5-10 | $10-20 | Gate 4: Phase 3 roundtrip tests pass |

**Total Phase 2-4 budget**: $18-35 (Lightning T4) / $29-60 (4090) / $60-120 (A100).

**Mandatory precondition**: `apogee_int6` [contest-CUDA] eval lands ($0.30-0.60).
If int6 ≤ 0.180, reassess whether Phase 2 is highest-ROI move. (Per blueprint
Gate 2.)

## 5. Risk assessment

| # | Risk | Probability | Mitigation |
|---|---|---|---|
| 1 | Joint-ADMM allocation gain doesn't survive smaller tensor count | Medium | Re-run Path B step 6 on arch_shrink output before Stage 2 commits |
| 2 | δ joint-training mode collapse | Medium (30-40%) | Curriculum λ ramp; pixel-MSE floor ≥ 0.001 |
| 3 | ζ collapses FiLM conditioning | Medium (40%) | `protect_film_layers=True` hard default |
| 4 | int4-catastrophe class inheritance | Medium | MSSIM ≥ 0.90 gate (apogee_int4 lacked it) |
| 5 | ε prior overfits to single video | Low | ≤ 5 KB hard cap |
| 6 | Archive-compliance failure at inflate | Medium | New `b"LEPR"` member OPTIONAL; fallback to static Laplace |
| 7 | Stack-composition byte budget collision | Medium | All components inside one zip; total bound by 186 KB current frontier |
| 8 | Subagent commit-serializer race | Low | `tools/subagent_commit_serializer.py` mandatory |
| 9 | MPS-CUDA divergence in any anchored row | Low (CPU-prep only) | All anchors are byte-counts, not score numbers |

## 6. Reactivation criteria (per CLAUDE.md "KILL is LAST RESORT")

This stack is NOT killed; it is DEFERRED-pending-Phase-2-empirical anchors.
Reactivation criteria for advancing past this design memo:

1. apogee_int6 [contest-CUDA] eval lands; score recorded.
2. arch_shrink_x0.4 Lightning dispatch lands; first [contest-CUDA] architecture
   anchor recorded.
3. Operator authorizes Phase 2 GPU spend ($8-15 Lightning T4) for δ joint-training.
4. Subagent E (recursive adversarial review of design) found the design clean
   in 3 consecutive passes — already complete (commit referenced in
   `feedback_recursive_adversarial_review_omega_opt_designs_20260508`).

## 7. Cross-references

- `paradigm_delta_epsilon_zeta_phase1_blueprint_20260507_claude.md` — Phase 1 design
- `feedback_path_b_convergent_findings_summary_20260508.md` — empirical CPU plateau
- `feedback_path_b_step6_admm_x_lossy_coarsening_allocation_mechanism_dominant_20260508`
- `feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508`
- `project_arch_shrink_x0_4_lightning_DISPATCHED_20260508` — in-flight anchor
- `project_cathedral_autopilot_recommender_landed_20260507` — top-3 candidates
- `feedback_recursive_adversarial_review_omega_opt_designs_20260508` — design review clean
- `feedback_goal_is_lowest_score_not_quantizr_paradigm_match_20260506` — strategic position
- `feedback_may_4_hnerv_race_postmortem_20260505` — race-mode rigor inversion rule

## 8. Operator decision points (recap)

- **Gate 0 (NOW)**: review this Phase 4 design memo. Cost $0. Risk: none.
- **Gate 1**: approve Phase 2 dispatch plan. Cost $0; Phase 2 dispatch $8-15.
- **Gate 2**: apogee_int6 [contest-CUDA] precondition. Cost $0.30-0.60. MANDATORY.
- **Gate 3**: approve ε/ζ Phase 3 dispatch. Precondition: Phase 2 scorer improvement.
- **Gate 4**: approve composition + Phase 4 dispatch. Cost $2-5. Precondition:
  Phase 3 roundtrip tests pass.
- **Gate 5**: approve public submission. IRREVERSIBLE disclosure. Precondition:
  [contest-CUDA] < 0.193; 5-turn clean-pass council review.

---

*Phase 4 INTEGRATION design complete. Empirical anchors land via existing
infrastructure (Path B + autopilot CPU sweeps + in-flight Lightning T4).
Phase 2 stub→implementation requires operator GPU authorization per blueprint
Gate 1.*
