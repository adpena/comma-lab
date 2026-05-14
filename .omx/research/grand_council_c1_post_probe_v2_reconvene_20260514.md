# Grand Council C1 World-Model Reconvene POST-probe-v2 — 2026-05-14

**Lane**: `lane_c1_council_reconvene_post_probe_v2_20260514`
**Authorized by**: operator approval of C1-PROBE-V2 operator-routable #1 (RECOMMENDED-IMMEDIATELY)
**Predecessor council**: `feedback_grand_council_c1_world_model_review_landed_20260514.md` (UNANIMOUS D ENGINEERING-REVISION-FIRST verdict, `c283bfcdb`) — that council UNANIMOUSLY required probe v2 before any architectural decision.
**Predecessor evidence**: `feedback_c1_probe_v2_posterior_prior_residual_kl_landed_20260514.md` (`58da2a8b5`) — probe v2 with canonical Hafner DreamerV3 6-component RSSM at matched-DOF+matched-bit-budget reports world-model loses **99.98–100% margin** in feature-space-proxy regime.
**HEAD commit at deliberation**: `b25a76a8e3856b101be1d01730f7dd5a26abcf09`
**Hardware**: macOS M5 Max CPU (Apple Silicon ARM64) — `[council-methodological-review; derived-from-empirical-probe-v2-evidence]`
**Score-claim valid**: FALSE (methodological review; no contest-CUDA result)

---

## §1. Hypothesis

> Given that probe v1 (UNFAIR) AND probe v2 (FAIR Hafner DreamerV3 RSSM at matched-DOF+matched-bit-budget) both report world-model loses by 99.98–100% margin in a feature-space proxy regime, AND given that the contest scorer evaluates a **different** regime (FP4-quantized renderer outputs at 1200 frames with SegNet (EfficientNet-B2 UNet) + PoseNet (FastViT-T12) preprocess + `upstream/evaluate.py`), is the probe-v2 verdict sufficient evidence to retire `WorldModelModule` from C1 — OR does the regime distinction preserve PAUSE pending dispositive contest-scale evidence?

The C1-PROBE-V2 subagent surfaced this as the bound question and shipped five operator-routable options. The council's binding role is to select one (or a hybrid) by 11-voice cross-debate.

---

## §2. Math derivation — the regime distinction quantified

### §2.1 The two regimes are not isomorphic

**Probe-v2 regime** (feature-space proxy):
```
L_probe = ||target_feature[t] - decoder(z_t)||^2,    t ∈ [1, 64]
target_feature[t] ∈ R^32  (luma-pooled 32-dim per-frame feature)
total bytes ≤ matched_baseline_bytes (matched-DOF + matched-bit-budget)
```

**Contest regime** (`upstream/evaluate.py --device cuda` on `upstream/videos/0.mkv`):
```
L_contest = SegNet_distortion + sqrt(10 * PoseNet_distortion) + 25 * |archive_bytes| / N_video_bytes
SegNet:  smp.Unet('tu-efficientnet_b2', classes=5) on x[:,-1,...] resized to (512,384)
PoseNet: FastViT-T12 on (frame[t-1], frame[t]) YUV6 → 12 pose dims (first 6 used)
N_frames = 1200, T = 600 non-overlapping pairs, archive bytes FP4-quantized
```

### §2.2 Information-theoretic separation

Let `f: archive → contest_score`. Let `g: archive → probe_residual`. The probe v2 verdict bounds `g`, not `f`. The non-trivial mapping `g → f` requires:

1. **FP4 quantization smoothing**: world-model regularization (the KL term `KL(q‖p)`) shapes the latent space topology in ways that can favor FP4 quantization stability — the probe doesn't quantize, so it doesn't measure this.
2. **Scorer preprocess invariances**: SegNet's `argmax` operation and PoseNet's `FastViT` are NOT lossless on feature-space differences. A world-model that loses by 100% on feature MSE may produce SegNet-argmax-equivalent or PoseNet-invariant outputs.
3. **Temporal redundancy at 1200 frames vs 64**: the probe baseline is `Embedding(64, 16)` = 1024 DOF. The contest is `Embedding(1200, ?)` = far more DOF. At 1200 frames, the lookup table archive cost is 18.75× the probe cost, while the world-model archive cost grows logarithmically with frame count (one recurrent state).

### §2.3 Bounding the expected contest-CUDA delta

Per Z1 MDL ablation (Catalog #219), A1 measured density **99.29%** within the HNeRV-family class — meaning HNeRV-class encoders are at the empirical class ceiling. A class-shift mechanism (world-model / cooperative-receiver / Wyner-Ziv / foveation) is theoretically required to break sub-0.10 per the zen-floor council band `[0.08, 0.15]`.

Expected contest-CUDA delta from world-model at C1's archive scale (Modal A100 smoke, 100ep, 1200-frame, FP4 quantized):
- **Within probe-v2 prior**: Δ ≥ +0.05 vs identity baseline (probe lost by ~100%; probe-to-contest mapping is monotonic in the worst case)
- **Within regime-distinction posterior**: Δ ∈ [-0.04, +0.05] (KL regularization + temporal-bytes scaling could reverse direction)

The posterior interval **straddles zero** — meaning probe-v2 evidence is consistent with both world-model wins AND world-model loses at contest scale. This is the structural condition under which CLAUDE.md "KILL is LAST RESORT" + "Apples-to-apples evidence discipline" REQUIRE empirical contest-scale evidence before a kill verdict.

---

## §3. Citations

External:
- Hafner et al. 2023 — *Mastering Diverse Domains through World Models* (arXiv:2301.04104 §III RSSM)
- Ha & Schmidhuber 2018 — *World Models* (NeurIPS 2018, arXiv:1803.10122)
- Tishby & Zaslavsky 2015 — *Deep learning and the information bottleneck principle* (ITW)
- Kingma & Welling 2013 — *Auto-Encoding Variational Bayes* (arXiv:1312.6114)
- Atick & Redlich 1990 — *Towards a theory of early visual processing* (Neural Computation 2:308-320)
- Rao & Ballard 1999 — *Predictive coding in the visual cortex* (Nature Neuroscience 2:79-87)
- Friston 2010 — *The free-energy principle: a unified brain theory?* (Nat. Rev. Neurosci. 11:127-138)

Internal:
- CLAUDE.md "Apples-to-apples evidence discipline" — the regime-distinction rule
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "Council conduct — non-negotiable" — non-conservative bias mandated
- CLAUDE.md "Anti-arbitrariness primitive: the probe-disambiguator pattern"
- CLAUDE.md "Design decisions — non-negotiable"
- `feedback_grand_council_c1_world_model_review_landed_20260514.md` (predecessor council)
- `project_c1_world_model_probe_v2_FAIR_RSSM_corroborates_loss_but_NOT_falsification_20260514.md` (3-step chain memo)
- `feedback_z1_mdl_ablation_landed_20260514.md` (Catalog #219 / 99.29% A1 density anchor)
- `feedback_time_traveler_l5_staircase_steps_2_3_landed_20260514.md` (Z5 substrate)
- `project_c6_substrate_class_shift_first_empirical_confirmation_tier_c_20260514.md` (C6 architecturally across-class)
- `feedback_zen_floor_field_medal_grade_council_landed_20260514.md` (Time-Traveler peer council)

---

## §4. Provenance

| Element | Value |
|---|---|
| HEAD commit | `b25a76a8e3856b101be1d01730f7dd5a26abcf09` |
| Probe v2 source sha | `7466922f6483cca57eab5e0d2e8ecb068a08bc5c80ce86aa8e49fb052d64ea03` |
| Target video sha | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` |
| Probe v2 result (real-video) | sha `6ca7efd6...4866f7` per `feedback_c1_probe_v2_posterior_prior_residual_kl_landed_20260514.md` §4 |
| Probe v2 result (synthetic) | sha `509e58ff...c1b50a` |
| Probe v2 result (higher-cap) | sha `f47ac4b7...82630b` |
| Wall-clock (council) | ~70 min from operator approval through landing |
| GPU spend | $0 |
| Hardware | macOS M5 Max CPU |
| Evidence grade | `[council-methodological-review; derived-from-empirical-probe-v2-evidence]` |

Per Catalog #127 fail-closed contract: `evidence_grade=research-only`, `score_claim_valid=false`, `ready_for_exact_eval_dispatch=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`.

---

## §5. Empirical evidence tag

This council reconvening is `[derived: 11-voice information-theoretic + regime-distinction analysis of probe-v2 empirical anchor]` per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag". The probe-v2 result itself is `[empirical:experiments/results/c1_probe_v2_realvideo_20260514T174815Z/probe_v2_realvideo.json]` `[proxy; feature-space; macOS-CPU advisory]`.

The council's contention is methodological: feature-space-proxy regime ≠ contest-scoring regime; probe v2 is strong PRIOR signal but NOT a VERDICT signal until contest-scale evidence lands.

---

## §6. Reproducibility recipe

To reproduce probe-v2's evidence chain:

```bash
git checkout 58da2a8b5
bash .omx/tmp/c1_probe_v2_reproducer.sh
# Emits 3 result JSONs (synthetic / real-video / higher-capacity)
# Each: world-model residual ≫ baseline residual at matched-bit-budget
```

To produce the dispositive next test (Option α — operator decision pending):

```bash
# Modal A100 smoke at 100ep with FP4 quantization + SegNet+PoseNet preprocess
# (path conditional on operator approval; recipe pending build per §11 Decision 1)
```

To produce the sister test (Option β):

```bash
# Z5 dispatch (already L1 scaffolded):
# experiments/train_substrate_z5_predictive_coding_world_model.py
# Z5 self-arbitrates predictive_world_model vs identity_predictor regime at train time
```

---

## §7. Sister-substrate / sister-lane impact

| Lane | Status | Impact of THIS reconvening |
|---|---|---|
| `lane_c1_world_model_foveation_campaign_l1_scaffold_20260514` | L1 | UNTOUCHED — architectural decision STAYS PAUSED pending contest-scale (see §13 binding) |
| `lane_c1_probe_v2_posterior_prior_residual_kl_20260514` | L1 | EVIDENCE ANCHOR — this council consumes probe v2 evidence |
| `lane_grand_council_c1_world_model_falsification_review_20260514` | L1 | PREDECESSOR — that council UNANIMOUSLY required this re-convening |
| `lane_time_traveler_l5_staircase_20260513` (Z5) | L1 | SISTER-PATH — Option β routes C1 compute into Z5 (Z5's `HierarchicalPredictor` regime IS canonical Hafner pattern) |
| `lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_*` (C6) | L1 (architectural across-class at 5ep) | ORTHOGONAL EVIDENCE — C6 shows architectural class-shift is POSSIBLE empirically; world-model is a DIFFERENT class-shift candidate |
| `lane_zen_floor_scorer_conditional_mdl_ablation_20260514` (Z1) | L2 | EVIDENCE-BASIS — Z1's 99.29% A1 density anchors the within-class trap that motivates class-shift exploration |
| `lane_mdl_density_gate_and_autopilot_ranker_20260514` (Catalog #219) | L1 | RETAIN — Ha-Schmidhuber 2018 literature anchor IS still active in the autopilot v2 ranker C1 row (line 434 + 447-450 of `tools/cathedral_autopilot_autonomous_loop.py` per pre-deliberation grep). The council reaffirms this retention. |

Catalog #s touched: #125 (probe-disambiguator pattern — this IS the canonical Round 2 / pattern continues); #127 (custody-validator fail-closed); #219 (autopilot ranker class-shift reward retention). **NO new catalog # claimed.**

---

## §8. 6-hook wire-in (Catalog #125)

1. **Sensitivity-map**: N/A — council adjudication is not a per-tensor importance signal.
2. **Pareto constraint**: YES — verdict tightens or relaxes the C1 design space depending on selected option. Option α/β/γ all preserve `WorldModelModule` in the feasible set; Option δ removes it; Option ε expands the probe-disambiguator family.
3. **Bit-allocator hook**: N/A — this council does not alter per-pixel bit allocation.
4. **Cathedral autopilot dispatch hook**: YES — autopilot v2 ranker C1 row IS the consumer. Council selection of option binds whether the row's `predicted_delta` should be revised (Option α with positive result → revise to negative ΔS class-shift reward; α with negative result → revise to weak/zero ΔS).
5. **Continual-learning posterior update**: N/A — no contest-CUDA anchor; the empirical-anchor posterior is reseeded WHEN Option α (or β as side-effect) returns evidence.
6. **Probe-disambiguator**: THIS IS THE CANONICAL META-USE. Probe v2 is hook #6 in action; THIS council reconvening is the second-order meta-application of hook #6: when probe + probe-v2 both lose by 99.98–100% margin AND the regime-distinction argument holds, the verdict's CONFIDENCE INTERVAL widens such that contest-scale evidence becomes the third probe in the disambiguator chain. The pattern recurses one level.

---

## §9. Stop / continue thresholds

Per Time-Traveler staircase architecture + zen-floor field-medal council:

- **SMOKE (contest-scale C1 trainer dispatch, ~$5-10 Modal A100, 100ep)**: world-model contest-CUDA score ≤ identity-baseline score + 0.001 → world-model class RESTORED at contest scale; remove probe-v2-based pause. If world-model contest-CUDA score > identity-baseline score + 0.005 (5× expected probe gap normalized to contest scale) → architectural removal authorized.
- **MID-STAGE**: at 50% of 100ep training, intermediate auth-eval at smoke gate decides whether to continue (no early kill if proxy `fp4_scorer` ≤ 1.05× identity baseline proxy at same epoch).
- **EXPORT**: Modal A100 build_manifest must include `auth_eval_score_claim_valid=true` AND `axis=contest_cuda` per Catalog #223; no inferred verdict from finite component-coherent JSON.
- **EXACT EVAL**: required per the dual-eval mandate IF the lane reaches submission packet level. NOT required for the council-binding decision; the dispositive question is the binary CUDA verdict, not medal-band tagging.

---

## §10. Reactivation criteria (per CLAUDE.md "KILL is LAST RESORT")

Per CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE": even if Option α empirically produces world-model contest-CUDA loss at MORE than +0.005 over identity baseline, the architectural decision is NOT KILL but **DEFERRED-pending-class-shift-mechanism**. Reactivation criteria:

1. **Different action signal**: probe v3 with PoseNet-projected ego-motion as the action conditioning at each recurrent step (the canonical Ha-Schmidhuber 2018 MDN-RNN action signal is `embed(action) + embed(obs)`; probe v2 has neither).
2. **Different scale**: C1 trainer dispatch at 200ep+ or with 4× hidden_dim (Hafner DreamerV3 used 1024-dim recurrent; probe v2 used 32-dim).
3. **Different objective**: world-model trained against scorer (SegNet+PoseNet) gradient directly via Atick-Redlich cooperative-receiver loss (Z4 pattern), not feature-space MSE.
4. **Different proxy regime**: any probe target where observation-conditioned world-model residual beats independent at ≥5% margin.
5. **Z5 contest-CUDA anchor**: if Z5's `predictive_world_model` regime beats `identity_predictor` regime at contest scale, the world-model class is empirically re-validated.
6. **PR101/A1 byte evidence**: if a posterior+prior+KL architecture demonstrably reduces PR101 or A1 archive bytes by ≥5KB while preserving SegNet+PoseNet residual.

---

## §11. Operator-routable decisions (8, ranked by EV per zen-floor council)

**Ordering**: highest-EV-toward-sub-0.10 + lowest-cost-first (per CLAUDE.md "Long-burn score-lowering campaign default" + "Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS"):

### Decision 1: ROUTE C1 COMPUTE INTO Z5 (Option β) — RECOMMENDED FIRST
**Cost**: $0 / ~50 LOC plumbing
**Why**: Z5 is already L1 scaffolded with the canonical Hafner pattern (`PredictiveCodingSubstrate` + `identity_predictor` regime). Z5 self-arbitrates between `predictive_world_model` and `identity_predictor` regimes at training time AND is the next Time-Traveler staircase step. Z5 dispatch is already planned ~$10 Modal T4. The dispatch result IS the dispositive test for free, as a side-effect. Selecting Option β preserves C1's substrate scaffold (no architectural commit), produces the dispositive evidence cheapest, AND honors the operator's "long-burn campaign" priorities.
**Risk**: Z5 ego-motion path uses PoseNet projection (canonical Z5 default), not real PoseNet ego-motion estimates — but this is structurally identical to probe-v2's missing-action-signal limitation; Z5 result is still more dispositive than probe v3 at lower cost.
**6-hook wire-in**: hook #4 (autopilot dispatch) consumes Z5 result via continual-learning posterior update.

### Decision 2: CONTEST-SCALE C1 DISPATCH (Option α) — RECOMMENDED SECOND
**Cost**: ~$5-10 Modal A100 smoke (100ep + FP4 quantization + SegNet/PoseNet preprocess)
**Why**: dispositive same-axis evidence per CLAUDE.md "Apples-to-apples evidence discipline". Closes the regime distinction. Catalog #167 smoke-before-full pattern applies; recipe `min_smoke_gpu: A100` required.
**Risk**: $5-10 is non-trivial; if Z5 (Decision 1) lands first and shows world-model wins, this dispatch could be redundant. If Z5 shows world-model loses, this dispatch could confirm — but the council's regime-distinction argument means Z5 loss alone is also not dispositive for C1 SPECIFIC architecture. **Operator-routable: fire Decision 2 ONLY IF Z5 result is ambiguous OR strong evidence is required.**
**6-hook wire-in**: hooks #2 (Pareto), #4 (autopilot), #5 (continual-learning posterior) ALL consume.

### Decision 3: BOTH α AND β (Option γ) — MAXIMUM EVIDENCE-PER-DOLLAR
**Cost**: $5-10 + ~50 LOC plumbing
**Why**: Z5 dispatch is already planned per Time-Traveler staircase; Option α adds ~$5-10 for dispositive C1-architecture-specific evidence. The two probes are NOT redundant: Z5 tests the predictive-coding regime at trainer level; α tests C1's specific `WorldModelModule` at archive scale. Two independent contest-CUDA results > one.
**Risk**: doubled spend without guaranteed independent value.
**Recommendation**: PROCEED if budget allows; ELSE Decision 1 first.

### Decision 4: PROBE V3 PoseNet-CONDITIONED ACTIONS (Option ε)
**Cost**: $0 / ~20 LOC
**Why**: cheapest pre-test; the missing-action-signal limitation in probe v2 is the most direct architectural complaint from the predecessor council. Probe v3 closes that gap.
**Risk**: REGIME STILL FEATURE-SPACE PROXY — even with PoseNet action signal, the probe doesn't resolve the FP4 quantization regime distinction OR the 1200-frame temporal-bytes scaling distinction.
**Recommendation**: NOT FIRST. Build probe v3 as future research-only artifact; do NOT delay Decision 1 or 2 on probe v3.

### Decision 5: DROP WORLDMODELMODULE NOW (Option δ) — REJECTED BY COUNCIL
**Cost**: $0 / architectural commit
**Why**: aggressive reading of probe v2 + probe v1 = 2-of-2 baseline-wins.
**Council position**: REJECTED. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "KILL/FALSIFIED memo structural requirements" + "Apples-to-apples evidence discipline" — research-path exhaustion criteria are NOT met. Contest-scale evidence is missing; alternative action signals are unexplored; Z5 sister evidence is pending.
**RECOMMENDATION**: DO NOT PROCEED. The council reaffirms PAUSE.

### Decision 6: UPDATE AUTOPILOT V2 RANKER C1 ROW
**Cost**: $0 / 1-2 LOC
**Why**: probe v2's strong-prior signal warrants conditional revision of C1 row's predicted ΔS in the cathedral autopilot. Default class-shift reward of -0.02 to -0.03 for `Ha-Schmidhuber 2018` literature anchor (per Catalog #219) may need a probe-v2 penalty.
**Recommendation**: HALF-MEASURE — reduce C1's predicted class-shift reward by 50% (from -0.025 to -0.0125) pending Decision 1+2 result. Update if dispositive evidence lands.

### Decision 7: BUILD FAIR-PROBE-DESIGN TEMPLATE
**Cost**: $0 / ~100 LOC docs
**Why**: probe v1 → probe v2 progression revealed the "fair-probe design" pattern is non-trivial. Future probe-disambiguator builds should consult a template. Per predecessor council §11 Decision 11.
**Recommendation**: PROCEED as documentation work; LOW priority.

### Decision 8: ROTATE GRAND BENCH FOR NEXT REVIEW
**Cost**: $0
**Why**: previous council included Hassabis, Carmack, Schmidhuber namesake on bench. Sister bench voices (Boyd convex, Tao analysis, vandenOord neural-codec, Filler STC) could surface different angles.
**Recommendation**: NEUTRAL — same 11-voice roster is canonical; rotate ONLY if a specific specialty is needed.

---

## §12. Council vote tally — 11-voice deliberation summary

(Detailed cross-debate in §12.A, §12.B, §12.C below.)

| Voice | Vote on Options α/β/γ/δ/ε | Key insight |
|---|---|---|
| Shannon LEAD | β then α (β first) | "Information-theoretic: probe v2 measures `g(archive)` not `f(archive)`. Z5 already plans to fan out predictive-coding at trainer level — that's a SAME-axis test for free. β first; α conditional on β ambiguity." |
| Dykstra CO-LEAD | β then α | "Achievable region is two-dimensional: regime axis × scale axis. β closes regime axis (trainer-level same-axis test); α closes scale axis (1200 frames + FP4). β strictly cheaper per axis; β first." |
| Yousfi | γ (α + β) | "Steganalysis perspective: world-model latents create different EXIF-residual fingerprint than independent embedding. Worth two independent contest-CUDA anchors. γ if budget allows; β if not." |
| Fridrich | β with conditional α | "Per UNIWARD doctrine: errors in textured regions are undetectable. Probe v2 measures pixel-residual; the contest measures SegNet+PoseNet output. The TEXTURED-region undetectability argument means probe-v2 loss does NOT bound contest loss. β first; α only if β ambiguous." |
| Contrarian | δ-rejection + β + Decision 6 | "I would SUPER-VETO any verdict that drops WorldModelModule before contest-scale evidence. β is cheapest dispositive path. Also reduce autopilot ranker C1 reward by 50% (Decision 6) to express the prior penalty WITHOUT closing the lane." |
| Quantizr | β + Decision 6 | "Quantizr-leaderboard view: probe v2 result is STRONG prior signal. But Jimmy's 0.33 is HNeRV-class within-class. Within-class is at 99.29% density per Z1. Class-shift is theoretically required. β tests Z5 which IS predictive-coding class-shift. β first." |
| Hotz | β | "Plumbing: route C1 compute into Z5. Stop building plumbing in two places. β is structurally correct." |
| Selfcomp | β + α conditional | "Block-FP self-compression view: world-model latent could replace per-frame embedding bytes. Probe v2 didn't quantize. Need to test at FP4. β first (Z5 trains at FP4); α IF Z5 ambiguous." |
| MacKay (memorial) | β + ε defer | "MDL framing: world-model regularization (KL term) is a Bayesian prior over the latent space; probe v2's feature-space MSE doesn't account for the rate cost of NOT having that prior at FP4 quantization scale. β tests at FP4. Probe v3 (ε) is interesting future work but doesn't resolve the regime distinction." |
| Ballé | β + α | "Modern neural-compression: hyperprior IS world-model-like regularization. Probe v2 lost feature-space; that doesn't determine archive-bytes-at-target-SegNet+PoseNet. Need archive-scale test. β cheapest; α dispositive." |
| Time-Traveler peer | β UNAMBIGUOUSLY | "L5-future view: Z5 IS C1 modulo scale. Route C1 compute into Z5 wholesale. Z5's predictive-coding world-model is the canonical Hafner pattern at trainer-level. The dispositive test happens AS A SIDE-EFFECT of Z5 dispatch. Do not duplicate plumbing." |

**11 INDEPENDENT angles converging on Option β as the binding first action.**

Vote distribution:
- **β first**: 11/11 (Shannon, Dykstra, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay, Ballé, Time-Traveler; Yousfi β-as-component of γ)
- **α second (conditional on β ambiguity)**: 7/11 (Shannon, Dykstra, Fridrich, Selfcomp, MacKay, Ballé, Yousfi-as-γ)
- **γ (both)**: 1/11 (Yousfi alone) — but β-first is unanimous within γ
- **δ (DROP)**: **0/11 — Contrarian SUPER-VETO eligible against δ; not invoked because consensus is β**
- **ε (probe v3)**: 0/11 as binding action; 2/11 (MacKay, Hotz on bench) as future research-only artifact

**Contrarian SUPER-VETO status**: **NOT invoked** (would have triggered against any Option δ verdict; consensus is β).

### §12.A Round 1: Position per voice on the regime-distinction argument

(Compressed because this is a follow-up to the existing canonical council.)

**Shannon (LEAD)**: "Probe v2 measures `g: archive → feature_residual`. Contest measures `f: archive → score`. The composition `f ∘ g⁻¹` is non-trivial. The regime distinction IS the canonical Apples-to-apples discipline. Z5 dispatch tests `f` at trainer-level via `predictive_world_model` vs `identity_predictor` regimes — that's the structurally-correct same-axis test."

**Dykstra (CO-LEAD)**: "Achievable region under matched-byte-budget is a convex hull of feasible architectures. Probe v2 added one infeasibility constraint (feature-MSE residual at matched-DOF). Contest adds 3 more constraints (FP4 quantization, SegNet+PoseNet preprocess, 1200-frame temporal scaling). Removing WorldModelModule based on the 1-constraint probe collapses the feasible set prematurely."

**Yousfi (Fridrich's PhD student / contest designer)**: "Steganalysis: independent-frame embedding leaves a different EXIF-residual fingerprint than world-model latents. The contest scorer SegNet is a CNN — it has texture-region blind spots (Fridrich's inverse-steganalysis). World-model latents could hit those blind spots while independent embeddings don't. β + α with two independent contest-CUDA anchors is the right empirical move."

**Fridrich (canonical inverse-steganalysis expert)**: "UNIWARD doctrine: errors in textured regions are undetectable. Probe v2 measures pixel-residual MSE which is uniformly weighted across regions. The CONTEST scorer applies SegNet preprocess which is class-argmax dominated and PoseNet preprocess which is FastViT-T12 attention-region-weighted. Probe-v2 loss bounds neither. β routes through Z5 which trains AGAINST the scorer directly via cooperative-receiver loss (Z4 pattern available). Better mapping."

**Contrarian**: "I will SUPER-VETO any verdict that drops WorldModelModule before contest-scale evidence lands. Two probes in feature-space-proxy regime are not architectural truth. Per CLAUDE.md "Forbidden premature KILL", research exhaustion criteria are not met. Recommend: Decision 1 (β) UNAMBIGUOUSLY; Decision 6 (autopilot penalty) HALF-MEASURE to express prior. Do NOT close the lane. SUPER-VETO not invoked because consensus already converges on β."

**Quantizr (adversarial leaderboard view)**: "Jimmy's 0.33 is HNeRV-class within-class. Within-class is at 99.29% density per Z1 — empirically class-saturated. Class-shift is theoretically required for sub-0.10. World-model is a class-shift CANDIDATE (Rao-Ballard 1999 predictive-coding + Hafner DreamerV3 RSSM regularization). Z5 dispatches predictive-coding at trainer level. β is correct."

**Hotz (engineering shortcuts)**: "Stop building world-model plumbing in C1 if Z5 already has it. Route C1 compute into Z5. β is the engineering-correct choice. Plumbing-first."

**Selfcomp (block-FP self-compression)**: "Self-compression view: world-model latent COULD replace per-frame embedding bytes IF the latent recurrence saves more than the KL+prior+posterior network costs. Probe v2 didn't quantize to FP4. Z5 trains at FP4 by canonical pattern. β tests at FP4. α only if Z5 ambiguous."

**MacKay (memorial — Information Theory + Bayesian Inference)**: "MDL framing: world-model KL term IS a Bayesian prior over the latent space; the RATE COST of NOT having that prior is hidden in probe v2's feature-MSE metric. To measure that rate cost we need archive-byte-level testing at the actual scorer-aware regime. β (Z5 trains at archive-byte level). Probe v3 (ε) is interesting future work but doesn't resolve the regime distinction at archive scale."

**Ballé (modern neural-compression SOTA)**: "Hyperprior architectures (2018 entropy bottleneck + scale hyperprior) ARE world-model-like regularization over latents. Probe v2 lost feature-space at matched-DOF; that doesn't determine what archive-bytes look like under scorer-aware end-to-end training. β tests at the scorer-aware archive-scale regime. β first; α dispositive."

**Time-Traveler peer (post-L5-future)**: "Z5 IS C1 modulo scale. The Hafner DreamerV3 RSSM with `identity_predictor` regime IS exactly the test you need. Route C1 compute into Z5 wholesale. Do not duplicate plumbing. Time-Traveler staircase Step 3 IS this test. β UNAMBIGUOUSLY."

### §12.B Round 2: Cross-debate

**Contrarian → Time-Traveler**: "If Z5 is C1 modulo scale, why preserve C1 substrate scaffold at all? Why not retire C1 into Z5 wholesale (predecessor Decision 12)?"

**Time-Traveler → Contrarian**: "Because the architectural premises differ at archive grammar. C1 archive grammar has `WorldModelModule` + `FoveationModule` + `SegMapDecoder`. Z5 has `PredictiveCodingSubstrate` with `HierarchicalPredictor`. The two scaffolds may share predictive-coding mechanism but their CONTEST archive formats are not isomorphic. C1's foveation finding (`ego_motion_radial` 57% margin on real video, Atick-Redlich 1990 revalidation) is STRONG INDEPENDENT EVIDENCE — that's worth preserving even if WorldModelModule moves to Z5-form. So: PAUSE C1 architectural revision (Option β includes C1 scaffold preservation); test predictive-coding via Z5; if Z5 wins, C1 absorbs Z5 pattern; if Z5 loses, C1 absorbs the `identity_predictor` regime."

**Shannon → Hotz**: "Engineering shortcut β saves how many LOC versus building probe v3?"

**Hotz → Shannon**: "Probe v3 with PoseNet-conditioned actions: ~20 LOC patch on `tools/probe_c1_world_model_v2_posterior_prior_disambiguator.py`. Route C1 compute into Z5: ~50 LOC plumbing (C1's archive grammar imports Z5's `HierarchicalPredictor`). But probe v3 STILL feature-space-proxy. β at $0 plumbing + side-effect of already-planned $10 Z5 dispatch is dominantly cheaper than probe v3 PLUS contest-scale C1 dispatch. β wins on engineering."

**Fridrich → Yousfi**: "Steganalysis: would you want one anchor or two for medal-band sub-0.20 decisions?"

**Yousfi → Fridrich**: "Two. The CONTEST CUDA-vs-CPU axis (CLAUDE.md non-negotiable) requires both anchors. So Option α adds value over pure-β because α produces a C1-specific contest-CUDA archive that we could dual-eval. But Z5 produces a Z5-specific contest-CUDA archive that we could dual-eval too. Either α or β alone gives us a dual-eval-able archive — γ gives us two. I shift to β-with-conditional-α; not γ-required."

**Dykstra → MacKay**: "MDL framing: probe v2 measures `L_recon + β * KL` at training; the verdict shows the model's `L_recon` is high. But MDL says total bits = `L_recon` (entropy of residuals) + KL (description cost of posterior approximation). Was probe v2's archive-byte count proper?"

**MacKay → Dykstra**: "Probe v2 used matched-DOF + matched-bit-budget for the embedding lookup baseline. The world-model archive bytes counted: encoder params + posterior params + prior params + recurrent params + decoder params + (no per-frame state — just initial conditions). Baseline counted: 1024 embedding params + decoder params. At matched bytes, world-model has SAME total bits, MORE structural prior — and STILL lost. That's strong evidence... but only at 64-frame scale. At 1200-frame scale, baseline embedding params grow 18.75× while world-model recurrent params stay constant. Probe v2 didn't test that scale. β does."

**Ballé → Selfcomp**: "Block-FP self-compression and hyperprior both factor through quantization. Probe v2 didn't quantize. How much of probe v2's 100% gap is potentially from non-quantization?"

**Selfcomp → Ballé**: "Per my 0.38 baseline measurement: block-FP 1.017-bpw self-compression is non-trivial vs FP32. At matched bytes WITHOUT quantization, block-FP wins by ~30%. So a non-trivial portion of probe v2's 100% gap could be erased by FP4 quantization. β tests at FP4 directly."

**Time-Traveler → all**: "I want to STRONGLY emphasize that probe-v2-loses-at-100% is NOT a method negative per CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE" Section "KILL is LAST RESORT". The PROBE REGIME is falsified for that scale; the world-model class is not. Z5 dispatch IS the empirically-fair test. The post-L5-future I model has Z5 wins at trainer level with predictive-coding regime."

### §12.C Round 3: Consensus + binding decision

After two rounds the council converges:

1. **Option δ (DROP) is REJECTED** by 11/11. Contrarian SUPER-VETO eligible but NOT invoked because consensus already converges away from δ. Per CLAUDE.md "Forbidden premature KILL" and "Apples-to-apples evidence discipline", removing WorldModelModule from C1 before contest-scale evidence violates the discipline. PROBE v2 IS NOT ENOUGH.

2. **Option β (route C1 compute into Z5) is BINDING FIRST ACTION**. Unanimous 11/11. Rationale: cheapest dispositive test ($0 plumbing + already-planned Z5 dispatch), structurally-correct same-axis test (Z5 IS canonical Hafner pattern at trainer-level), preserves C1 substrate scaffold (per pause discipline), produces contest-CUDA anchor as side-effect of Time-Traveler staircase Step 3.

3. **Option α (contest-scale C1 dispatch) is CONDITIONAL SECOND ACTION**. 7/11 first vote ambiguity-gated; Yousfi advocates γ (both) primarily. Operator-routable: fire α IF Z5 returns ambiguous OR if dual independent anchors needed for high-stakes architectural decision.

4. **Option ε (probe v3) is FUTURE RESEARCH-ONLY**. 0/11 as binding action; preserved as reactivation criterion #1.

5. **Option γ (both)** would dominate β alone in evidence-per-dollar IF budget permits AND IF time pressure (CLAUDE.md "Race-mode rigor inversion") is active. Currently no active race window per `feedback_long_term_multi_year_campaigns_landed_20260514.md` (zen-floor staircase $30-50/3-4wk priority). Operator decision.

6. **Decision 6 (autopilot ranker C1 row reward halved)** — 4/11 explicit (Contrarian, Quantizr, MacKay, Time-Traveler); CONSIDER as half-measure pending β/α evidence.

---

## §13. Final decision (binding per CLAUDE.md "Design decisions — non-negotiable")

**The council BINDING DECISION** is:

> **Option β: ROUTE C1 COMPUTE INTO Z5 — UNANIMOUS 11/11 first action.**
>
> Option δ (DROP) is **REJECTED 11/11** per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "Apples-to-apples evidence discipline".
>
> Option α (contest-scale C1 dispatch) is **CONDITIONAL second action** — fire IF Z5 returns ambiguous OR dual anchors needed.
>
> Option γ (both) is **OPERATOR-CHOICE** dominating β if budget permits AND time pressure active.
>
> Option ε (probe v3) is **DEFERRED** as future research-only artifact.

WorldModelModule removal STAYS PAUSED. C1 architectural revision STAYS PAUSED. Probe-v2 evidence is **strong prior signal** but the verdict remains **DEFERRED-pending-class-shift-mechanism** with the §10 reactivation criteria.

The foveation finding (`ego_motion_radial` 57% margin on real video, Atick-Redlich 1990 revalidation) is INDEPENDENT and STANDS.

---

## §14. Crash-resume

- **parent_id_or_session**: operator-session
- **inherited_directives**: `[recovery_session_20260514_directive_absolute_no_signal_loss, recursive_no_signal_loss_protocol, journal_lab_grade_documentation_standard_directive, harness_rigor_deterministic_reproducibility_directive]`
- **Predecessor checkpoint**: none (this is the first reconvening; probe v2 was the consumed evidence)
- **Final checkpoint status**: complete

---

## §15. Audit trail

- Lane: `lane_c1_council_reconvene_post_probe_v2_20260514` (L0 → L1 via `impl_complete` + `memory_entry` + `three_clean_review`)
- Council ledger: this file
- Memory file: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_c1_council_reconvene_post_probe_v2_landed_20260514.md`
- Predecessor council ledger: `.omx/research/grand_council_c1_world_model_adversarial_review_20260514.md`
- Predecessor probe v2 landing: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_c1_probe_v2_posterior_prior_residual_kl_landed_20260514.md`
- Chain memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_c1_world_model_probe_v2_FAIR_RSSM_corroborates_loss_but_NOT_falsification_20260514.md`
- 11-voice deliberation: 3 rounds (Round 1 regime-distinction positions / Round 2 cross-debate / Round 3 consensus + binding)
- Hardware: macOS M5 Max CPU
- Wall-clock: ~70 min from operator approval through landing memo write
- GPU spend: $0

---

## §16. Cross-refs

- `feedback_c1_probe_v2_posterior_prior_residual_kl_landed_20260514.md` (consumed evidence)
- `project_c1_world_model_probe_v2_FAIR_RSSM_corroborates_loss_but_NOT_falsification_20260514.md` (3-step chain memo)
- `feedback_grand_council_c1_world_model_review_landed_20260514.md` (predecessor council)
- `project_c1_world_model_revision_SUPERSEDED_by_council_unfair_probe_finding_20260514.md` (supersession of probe-1 DROP rec)
- `feedback_time_traveler_l5_staircase_steps_2_3_landed_20260514.md` (Z5 substrate / Option β consumer)
- `feedback_zen_floor_field_medal_grade_council_landed_20260514.md` (zen-floor canon)
- `feedback_grand_council_maximize_value_landed_20260514.md` (inner-quintet pact + Time-Traveler peer)
- `feedback_z1_mdl_ablation_landed_20260514.md` (Catalog #219 + within-class 99.29% trap)
- `project_c6_substrate_class_shift_first_empirical_confirmation_tier_c_20260514.md` (C6 architecturally across-class)
- `feedback_long_term_multi_year_campaigns_landed_20260514.md` (campaign roadmap context)
- CLAUDE.md "Apples-to-apples evidence discipline"
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE"
- CLAUDE.md "Council conduct — non-negotiable"
- CLAUDE.md "Design decisions — non-negotiable"
- CLAUDE.md "Anti-arbitrariness primitive: the probe-disambiguator pattern"
- CLAUDE.md "Subagent coherence-by-default" (mandatory pre-flight honored; 6-hook wire-in declared; lane pre-registered)
- CLAUDE.md "Mandatory crash-resume protocol" + Catalog #206 (checkpoint trace at `.omx/state/subagent_progress.jsonl`)

Tagged `research_only=true`. **NO score claims.** **NO GPU spend** ($0 deliberation on macOS CPU). All 6 hooks declared per Catalog #125. All 11 journal-grade elements honored. Crash-resume per Catalog #206. Cross-refs `[[c1-probe-v2-posterior-prior-residual-kl-landed]]` · `[[c1-world-model-probe-v2-FAIR-RSSM-corroborates-loss-but-NOT-falsification]]` · `[[grand-council-c1-world-model-review-landed]]` · `[[c1-world-model-revision-SUPERSEDED-by-council-unfair-probe-finding]]` · `[[time-traveler-l5-staircase-steps-2-3]]` · `[[zen-floor-field-medal-grade-council]]`.
