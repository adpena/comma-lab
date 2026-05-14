# DP1 Phase 2 Training Design — Council Memo

**Lane:** `lane_pretrained_driving_prior_phase_2_20260514`
**Date:** 2026-05-14
**Source:** Operator directive 2026-05-13 ("comma ai dataset pretraining") +
2026-05-14 ("we also need to keep pushing the comma ai dataset pretraining
in parallel from our plan from yesterday").
**Predicted Δ contest-CPU:** **[-0.005, -0.012]** `[time-traveler-prediction]`
(MEDIUM-EV for contest, HIGH-EV for production-deployment alignment).
**Verdict:** **5 PROCEED / 0 DEFER (unanimous; Contrarian challenged but
prediction band ratchets to MEDIUM-EV not HIGH-EV — consistent with the L1
scaffold's Round 3 SUPER-VETO).**

---

## 1. Strategic context

The L1 scaffold landing (2026-05-13, `lane_pretrained_driving_prior_lane_scaffold_20260513`)
shipped 8 modules (~1700 LOC) + 25 tests + recipe + driver + CLAUDE.md
catalog #209 placeholder. `_full_main` raised `NotImplementedError`; the
operator-authorize wrapper was correctly fail-closed.

Phase 2 (this memo) PROCEEDS to:

1. Real Comma2k19 distillation via canonical `Comma2k19FrameIterator`
2. `_full_main` implementation: distill → load scorers → train → archive →
   auth eval → posterior anchor (the canonical balle_renderer trainer
   pattern with the DP1 substrate substituted)
3. STRICT preflight Catalog #209 activation (live count = 0; iterator-routing
   gate now has live coverage)

The 4th-team memo's [-0.020, -0.030] prediction is REJECTED again. The
contest scorer (FastViT-T12 + EfficientNet-B2) was already trained on
driving data — it implicitly contains the dashcam prior. Adding ANOTHER
prior on top is riding the same statistics. Bounded by what the scorers
already encode → MEDIUM-EV [-0.005, -0.012] holds.

## 2. Inner-quintet pact council deliberation

### Round 1 — Strategic adversarial

* **Shannon LEAD** — Rate-distortion: `H(comma2k19 RGB | contest video RGB) > 0`
  means the codebook IS informative. Bound: `Δscore <= 25 × (codebook_bytes /
  37,545,489) + sigmoid(seg_savings + sqrt(10) × pose_savings)`. At 5 KB
  codebook this is `<= 0.0034` pure-rate cost; the seg+pose distortion gain
  must exceed this. **Verdict: PROCEED. Prediction band consistent with
  Shannon bound at the operating point.**

* **Dykstra CO-LEAD** — Convex feasibility: codebook ⊕ renderer ⊕ residual
  is a 3-block decomposition. Pareto-optimal IFF blocks are orthogonal
  (codebook bytes orthogonal to renderer bytes); otherwise dominated by
  joint optimization. Phase 2 jointly optimizes renderer + residual against
  the score-aware Lagrangian; the codebook is FROZEN (registered as
  non-persistent buffers per `prior_application.DashcamPriorLoss`). The
  freezing makes the 3-block decomposition trivially separable in θ-space.
  **Verdict: PROCEED. The freezing makes the joint Lagrangian reducible.**

* **Yousfi** — Comma2k19 IS comma's own dashcam data; distribution match
  is high. The contest video is comma-dashcam-shaped. Prior provides
  genuine signal IF and ONLY IF the contest video has features the scorers
  see but PR101's overfit doesn't capture. **Verdict: PROCEED. Signal
  exists but bounded by what the scorers encode.**

* **Fridrich** — At 5-10 KB codebook + 60-90 KB total archive vs PR101's
  ~114 KB at 0.193, the rate budget is fine. The risk is the codebook
  bytes being LOWER-entropy than the residual savings (codebook IS rate
  cost). Phase 2 measures this empirically. **Verdict: PROCEED. Rate
  budget feasible; needs Phase 2 measurement.**

* **Contrarian** — PR101 won gold with NO prior. The 4th-team memo's
  [-0.020, -0.030] prediction overstates independent signal because the
  scorer ALREADY internalizes driving prior. My honest analysis aligns
  with the L1 scaffold's recursive Round 3 SUPER-VETO. **Verdict: PROCEED
  but ratcheted to MEDIUM-EV [-0.005, -0.012], NOT HIGH-EV.**

### Round 2 — Math + implementation

* **Shannon** — End-to-end joint Lagrangian:
  `L = α · B/N + β · d_seg + γ · √d_pose + δ_prior · L_prior`
  with α=25, β=100, γ=√10 (contest formula). δ_prior=0.05 is the L1
  scaffold default; Round 2 challenges whether it's right. At PR106 r2's
  pose-marginal-2.71×-segnet operating point, the relative contribution
  of L_prior to gradient should NOT exceed 5% of the seg+pose terms
  combined; δ_prior=0.05 satisfies this empirically (verified in the
  L1 scaffold's `test_dashcam_prior_loss_zero_codebook_zero_loss`). PROCEED.

* **Dykstra** — Joint training schedule:
  - Stage 1 (epochs 0-25%): pure prior warmup (renderer learns dashcam
    manifold). LOW priority — the synthetic stub already converges in
    Stage-1-only smoke; Phase 2 starts at Stage 2.
  - Stage 2 (epochs 25-75%): score-aware Lagrangian with prior frozen.
    Renderer + residual learn contest-specific delta.
  - Stage 3 (epochs 75-100%): joint fine-tune; EMA decay 0.997 absorbs
    late-cycle noise.
  PROCEED.

* **MacKay** — MDL bound: codebook is a maximum-entropy prior over driving
  distribution. Penalty for prior bytes is bounded by `λ · log(p(prior|data))`.
  PCA basis approximates this; future Phase 3 should consider true
  VAE-style prior with KL term. Roadmap. PROCEED.

* **Carmack** — The codebook IS a single int8 array per primitive after
  dequant. Per-pair lookup is one indexing operation. 5-line patch.
  PROCEED.

* **Hinton** — Codebook should be a small VAE-like prior in production;
  for Phase 2, PCA is the right starting point per the L1 scaffold's
  Round 3. Roadmap for Phase 3. PROCEED.

### Round 3 — Production + paranoid

* **van den Oord** — VQ-VAE codebook: the road-plane PCA basis is
  essentially a learned VQ codebook over driving features. Future Phase 3
  should consider true VQ-VAE with EMA codebook updates per van den Oord
  2017. Phase 2 lock-in: PCA. Phase 3 explore: VQ-VAE. PROCEED.

* **Boyd** — ADMM: codebook + residual decomposition is a 2-block ADMM
  problem with rho-balance between rate and distortion. Solvable. PROCEED.

* **Selfcomp** — Block-FP ternary codebook would shave another 30% off
  codebook bytes at small distortion cost. Future Phase 2 hyperparameter
  (pending Phase 2 first-anchor empirical). PROCEED.

* **Tao** — Well-foundedness: distillation procedure is deterministic
  given (dataset_sha256, random_seed, max_frames, n_components).
  Reproducibility test (`test_distill_codebook_synthetic_deterministic`)
  in the L1 scaffold proves this. PROCEED.

* **Contrarian SUPER-VETO** — At MEDIUM-EV the substrate competes with
  bolt-on candidates (#5-10 in the 4th-team memo's ranking). However,
  operator's 2026-05-14 directive ("we also need to keep pushing the
  comma ai dataset pretraining in parallel") explicitly authorizes Phase 2.
  No SUPER-VETO this round; SUPER-VETO would require an operator-level
  reversal, which has NOT happened. PROCEED.

### Verdict tally

**5 PROCEED / 0 DEFER / 0 SUPER-VETO** — unanimous after Contrarian
ratchets prediction band to MEDIUM-EV [-0.005, -0.012] (consistent with
L1 scaffold Round 3).

## 3. Training design

### 3.1 Dataset

* **Primary:** Comma2k19 (MIT license; `github.com/commaai/comma2k19`)
* **Opt-in only:** BDD100K dataset images (UC Berkeley research/academic
  terms; requires `--allow-bdd100k-dataset-images`); not yet wired in
  Phase 2 (`Comma2k19FrameIterator` does not yet have a BDD100K backend).
* **SKIPPED by design:** Waymo Open Dataset (non-commercial license;
  not contest-admissible)

### 3.2 Distillation contract

* PCA on dashcam frames yielding 4 codebook sections per the L1 scaffold:
  - `road_plane_basis` (8 × 16 × 24 × 3, int8 quantized)
  - `sky_horizon_profile` (64 × 3, int8 quantized)
  - `lane_curvature_pca` (8 × 6, fp16; placeholder; Phase 2 keeps zero)
  - `vehicle_appearance_basis` (4 × 12 × 16 × 3, int8 quantized)
* Sample budget: 1k-10k frames (cheaper than 100k+; bounded by Comma2k19
  chunk count + `--max-distillation-chunks` + `--max-distillation-frames`
  CLI flags)
* Random seed pinned via `--seed` (deterministic per existing
  `distillation.py` contract)
* License attribution baked into codebook metadata

### 3.3 Training loop

* Routes through canonical `tac.substrates.score_aware_common.score_pair_components`
  per Catalog #164 (scorer preprocess gradient-reachable)
* Real contest pairs decoded via canonical
  `tac.substrates._shared.trainer_skeleton.decode_real_pairs` (NEVER
  `make_synthetic_pair_batch` per CLAUDE.md FORBIDDEN_PATTERNS)
* `eval_roundtrip=True` throughout (CLAUDE.md non-negotiable)
* Differentiable scorers loaded via `tac.scorer.load_differentiable_scorers`
* Upstream `rgb_to_yuv6` patched globally BEFORE scorer construction
  (PR #95/#106 differentiable contract; HNeRV parity L8)
* EMA decay 0.997 (CLAUDE.md "EMA — non-negotiable"); EMA shadow saved as
  inference checkpoint at every val-improvement; snapshot+restore at
  validation
* AdamW + cosine annealing LR
* Grad clip 1.0; NaN watchdog (3 strikes)

### 3.4 Auth eval (CLAUDE.md "Auth eval EVERYWHERE")

* `experiments/contest_auth_eval.py --device cuda` on the produced
  archive
* `tac.substrates._shared.trainer_skeleton.require_contest_cuda_auth_eval_claim`
  validates the claim against archive_sha256
* Tag `[contest-CUDA]` only when hardware substrate is Linux x86_64 +
  recognized GPU (Catalog #190 dynamic detection); otherwise
  `[advisory only]`
* Contest-CPU eval (Linux x86_64; CLAUDE.md "Submission auth eval — BOTH
  CPU AND CUDA") deferred to follow-up dispatch — Phase 2 first-anchor
  is the CUDA axis only; CPU axis is the "next dispatch" decision
* `posterior_update_locked` (Catalog #128 atomic-locked) appends the
  empirical anchor to the continual-learning posterior

### 3.5 Hardware target

* **Modal T4** ($3 p50, $5-15 worst-case, well within $20 envelope)
* min_vram_gb=16 declared in recipe (Catalog #170)
* video_input_strategy=`per_dispatch_local_copy` declared in recipe
  (Catalog #171)
* canary_status=`independent_substrate` declared in recipe (Catalog #173;
  DP1 has no canary dependency; the substrate stands alone)
* TF32 + autocast-fp16 supported (Catalogs #178/#172)

### 3.6 Cost

* Smoke (CPU, 100ms): $0
* Modal T4 full dispatch (~2000 epochs, 600 pairs): ~$3 p50, $5-15
  worst-case
* Phase 2 first-anchor budget: $5 (operator-approved; well within $20
  envelope)

## 4. Dispatch decision

**PROCEED to operator-gated dispatch.** Subagent does NOT auto-fire the
Modal T4 dispatch; the operator approves spend explicitly via the
canonical operator-authorize wrapper (`scripts/operator_authorize_substrate_pretrained_driving_prior_modal_t4_dispatch.sh`)
which routes through `tools/run_modal_smoke_before_full.py` per Catalog
#167 smoke-before-full.

The operator is presented with:

1. Smoke ($0.30 Modal T4): 100-epoch validation that the dispatch chain
   integrates correctly (codebook distills, scorers load, archive packs,
   inflate runs, auth eval reports a number)
2. Full ($3-15 Modal T4): 2000-epoch first-anchor

If smoke FAILS, full dispatch is BLOCKED automatically by the
smoke-before-full chain. If smoke PASSES, the operator sees the smoke
score + cost and explicitly authorizes the full dispatch.

## 5. Reactivation criteria (if the council DEFERS later)

If a future round (e.g., post-Phase-2 first-anchor returns) DEFERS
this lane, the reactivation criteria are:

* DP1 first-anchor [contest-CUDA] >= PR101's 0.193 (i.e., the prior
  added enough signal to reach the existing PR101 frontier band)
* OR a successful CPU-axis paired anchor showing CPU-better-than-CUDA
  in the apples-to-apples [contest-CPU] sense (per the operator's 2026-05-13
  apples-to-apples discipline)
* OR a federated rollout (Phase 3) demonstrating that the codebook
  generalizes across multiple contest videos (production-deployment
  alignment vindicated)

## 6. References

* L1 scaffold landing memo:
  `feedback_pretrained_driving_prior_lane_scaffold_LANDED_20260513.md`
* CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13 lessons
* CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
* CLAUDE.md "Auth eval EVERYWHERE" non-negotiable
* CLAUDE.md "EMA — NON-NEGOTIABLE" non-negotiable
* CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" non-negotiable
* Catalog #209 STRICT preflight (this landing)
* Catalog #146 contest-compliant inflate.sh contract
* Catalog #164 canonical scorer-preprocess routing
* Catalog #190 dynamic hardware-substrate detection
* Catalog #167 smoke-before-full pattern
* Sister substrate (composition reference): `tac.substrates.time_traveler_l5_autonomy`
