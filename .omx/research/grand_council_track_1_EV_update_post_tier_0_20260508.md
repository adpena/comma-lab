# Grand Council EV update on Track 1 — post-Tier-0 diagnostic

**Date:** 2026-05-08
**Operator authorization:** "approved, proceed with all; no time/GPU limits; sure about EV before GPU spend; consult grand council again"
**Trigger:** Tier 0 diagnostic on `feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508.md` — discovered 2,228 B encoder-class headroom on PR101 weights kills bolt-on Ballé hyperprior

## The Tier 0 finding (decisive)

PR101 weights are near-iid. Brotli baseline 178,144 B; Shannon iid empirical floor 175,916 B. Total encoder-class headroom = **2,228 B**, smaller than the smallest learnable neural codec's compressed-weights overhead. Bolt-on Ballé / FactorizedPrior / NWC / hyperprior of any kind on PR101 weights structurally PAYS more bytes than it saves.

This is not a config bug. It's a structural ceiling on bolt-on entropy coding for already-optimized iid weights.

## What this changes in the Track 1 plan

**Old plan (PRE-Tier-0):** 5 stacked decisions on a new substrate, predicted 0.158-0.165 [contest-CPU].

**New plan (POST-Tier-0):** Decision 1 (Ballé hyperprior) is FALSIFIED as bolt-on. It requires co-trained weights — the model must see hyperprior rate as a loss term FROM EPOCH 0 — for the weight distribution to become hyperprior-friendly (heavy-tailed where the hyperprior wins).

This converts Decision 1 from "5% probability of working" (bolt-on with prior anchor 0.985 score) to "70% probability of working" (co-design where the substrate is shaped by the codec's loss).

## Updated EV table

| Decision | Bolt-on EV | Co-design EV | Standalone test cost |
|---|---:|---:|---:|
| 1 — Ballé scale hyperprior | 5% (FALSIFIED) | 70% | ~$15 (small co-design ablation) |
| 2 — Score-gradient inner-loop | 80% | 90% | ~$8 (200-epoch fine-tune) |
| 3 — Sensitivity-aware quantization | 85% | 90% | ~$1 (CPU, no retrain) |
| 4 — Pose-deriver + residual coding | 70% | 80% | ~$5 |
| 5 — Frame-conditional bit budget | 60% | 75% | ~$3 |

**Council verdict (full bench):** Run all 5 standalone ablations in parallel BEFORE committing to the full stack. Total Phase A cost: ~$30-50. Insurance against the Contrarian's compound-probability argument (5 components each at 70% give 17% all-work; landing each individually first gives ~70% on the validated subset).

## Council positions (full inner-ten + grand bench refinement)

### Inner ten

**Shannon (LEAD).** Joint-entropy floor on PR101 weights is 148-162 KB; achievable only via co-trained weights or different architecture. Bolt-on is mathematically capped at brotli + epsilon. **Verdict updated:** Track 1 EV is HIGH conditional on co-design; KILLED conditional on bolt-on. Phase A4 (co-trained Ballé small ablation) is the diagnostic.

**Dykstra (CO-LEAD).** The achievable region on the (rate, seg, pose) axes is non-convex (sqrt on pose). Lagrangian dual update with 5 active constraints is not guaranteed to converge. Recommend Boyd-style ADMM with explicit slacks. **Verdict:** HIGH with caveat; verify feasibility numerically before full commit.

**Yousfi.** SegNet stride-2 stem is the structural blind spot. Decision 3 (sensitivity-aware quant) and Decision 2 (score-gradient) both attack this. EV unchanged — HIGH.

**Fridrich.** UNIWARD discipline at weight level still applies. Decision 3's per-tensor importance weighting is canonical. EV HIGH.

**Contrarian.** "5 components × 70% each = 17% all-work probability" — STILL the strongest critique. The Tier 0 finding actually strengthens this: it shows that any single bolt-on can fail despite being theoretically sound. Validates Phase A staging requirement. **Verdict:** Phase A is mandatory; full stack dispatch only after each component anchored.

**Quantizr.** Reverse-engineered Quantizr 0.33 archive used architecture choice (FiLM + DSConv) more than codec cleverness. The HNeRV jump was architectural too. **Verdict:** Decision 2 (score-gradient on a smaller, simpler architecture) may be more important than Decisions 1+3+4+5 combined. Recommend Phase A1 (score-gradient ablation) gets highest priority.

**Hotz.** Track 2 (custom CUDA decoder) is parallel-dispatchable, $0 GPU, ~3-5 days dev. Council recommendation: **launch Track 2 scaffold in parallel with Phase A**.

**Selfcomp.** Block-FP at 1.017 bpw is a separate codec primitive. Compose with Track 1: use block-FP on the high-sensitivity tensors, Decision 1 hyperprior on the rest. **Verdict:** add Phase A6 — block-FP × hyperprior composability test (~$5).

**MacKay (memorial).** Bayesian-MDL closed-form lower bound — request: compute it explicitly for the proposed substrate before dispatch. ~30 min CPU work. **Verdict:** Phase A0 — write the closed-form MDL calculator (free, deterministic).

**Ballé.** "The 2,228 B math kills bolt-on. Co-trained Ballé is a different code path. Recommend implementing the ChARM 2020 variant with channel-conditional autoregression for INT8 residuals — that's the right tool." **Verdict:** Phase A4 spec must specify ChARM 2020, not 2018 ScaleHyperprior canonical.

### Grand bench

**Boyd (operational ADMM).** 5-axis ADMM converges slowly. Suggest incremental: 2-axis (rate, seg) → 3-axis (add pose) → full 5-axis. Each step validates dual feasibility. **Wired into Phase C staging.**

**Tao (pure math).** Score function `25·B/N + 100·seg + sqrt(10·pose)` is concave in pose, linear in seg/rate. Achievable set is non-convex. ADMM not guaranteed to find global optimum. Recommend solving dual problem with explicit slack variables and projecting back. **Wired into Phase C.**

**Filler (STC).** Pose tensor is perfect for STC encoding (small alphabet, parity-friendly). Replace Decision 4 (pose-deriver) alternative: STC-encode pose tensor with side-info from a noise model. Lower architecture risk. **Phase A4-alt: STC pose encoding.**

**Mallat (wavelets).** Per-tensor importance can be replaced by wavelet-coefficient importance for higher resolution. Decision 3 refinement: wavelet basis × per-tensor importance. **Phase A3-extended.**

**van den Oord (VQ-VAE).** Hybrid: VQ-VAE on activations + scalar quant on weights. Could replace Decision 3. **Phase A3-alt.**

**Carmack (engineering).** "5 decisions × 250K params = 6 weeks debugging." Recommend Decision 2 alone first. If proxy-auth gap closes from 7×rel_err coefficient down to 1×, the rest may not be needed. **Phase A1 highest-priority.**

**Hassabis (strategic).** A/B Track 1 vs Track 2 explicitly. Track 2 has lower component-count risk. **Run both in parallel.**

**Hinton (KL distill, memorial).** Decision 2 IS the KL-distill pattern from his 2014 paper. Already in `losses.py`. Component-tested. **EV HIGH for Phase A1.**

**Karpathy (engineering).** "Let compute speak — operator has unlimited GPU." Run Phase A as 5-6 parallel ablations. Tier 1 cost $30-50, Tier 2 anchor cost $50-100, Tier 3 full stack $50-100. Total ~$130-250. Operator authorized "no GPU limit"; this is well within. **Strongly endorse parallel dispatch.**

**Schmidhuber (compression-as-intelligence).** Joint MDL framework is correct. Implementation risk is in component composition. **Endorse Phase A staging.**

**Jack-from-skunkworks.** Decision 3 was his lane. Sensitivity-aware quant lands at Phase A2 with no retraining required ($1 CPU work). **Highest EV per dollar in Phase A.**

## Council vote on dispatch order (updated)

Phase A0 (MDL calculator, free) → ALL 10 endorse.
Phase A1-A6 parallel (each Decision standalone) → 18 of 22 endorse parallel; Carmack + Hassabis + Contrarian endorse but emphasize **A1 (score-gradient) is the gating priority** — if it fails to close the proxy-auth gap, the entire Track 1 EV collapses regardless of other decisions.
Phase B (Track 2 scaffold) parallel with Phase A → unanimous (Hotz strong, Hassabis A/B argument).
Phase C (Track 1 stack composition) gated on Phase A success → unanimous.

## Phase A unit-test budget allocation

| Phase | Decision | $/unit | Allocation | Notes |
|---|---|---:|---:|---|
| A0 | MDL closed-form calc | $0 | 1× | gate; deterministic; ~30 min CPU |
| A1 | Score-gradient (Carmack/Quantizr/Hinton priority) | $8 | 2× ($16) | Two configs: seg-grad-only and full seg+pose-grad |
| A2 | Sensitivity-aware quant | $1 | 1× | CPU only |
| A3 | Pose-deriver | $5 | 1× | |
| A3-alt | Mallat wavelet importance | $4 | 1× | |
| A4 | Co-trained Ballé (ChARM 2020) | $15 | 1× | Critical Tier 0 follow-up |
| A4-alt | Filler STC pose | $5 | 1× | |
| A5 | Frame-conditional bit budget | $3 | 1× | |
| A6 | Selfcomp block-FP × hyperprior compose | $5 | 1× | |
| **Total Phase A** | | | **$55** | All authorized — operator no-GPU-limit |

## Phase C trigger criteria

Phase C (full Track 1 stack dispatch, ~$50-100 Lightning T4) launches IFF:
- Phase A1 (score-gradient) shows ≥10% reduction in seg or pose term on PR101 substrate
- Phase A4 (co-trained Ballé) shows ≥5 KB savings vs brotli on its small co-design ablation
- ≥3 of 5 Phase A decisions anchor positively

If <3 anchor: Phase C deferred. Track 2 (Hotz) becomes primary.

## Long-term hardening (parallel with Phase A-C)

Operator directive: "do the long term research and work and implementation necessary and also keep fixing and hardening and examining and analyzing and exploring and hardenning and making our tools more sophisticated and smarter."

Continuous workstreams (no specific dispatch; ongoing):
1. **Preflight B-classes 5-8 backfill** (rel_err / wire-format / archive-custody / dispatch-blocker — ~75 violations remaining)
2. **Per-architecture-class profile registry extension** — ingest more PRs as data
3. **meta-Lagrangian + sensitivity_map + score_geometry refinement** — wire all three into a unified field-equation planner
4. **Cathedral_autopilot evidence-row backfill** — operator's "make tools smarter" directive lands here
5. **Council deliberation persistence** — every council session lands as a memo, indexed
6. **Lane Maturity registry extension** — backfill any in-flight lane discipline
7. **Tool sophistication**: cross-paradigm composer, predictor accuracy harness, dual-axis CUDA-CPU calibration on every architecture class

## Updated /loop standing prompt (proposed)

OLD (stale, race-mode framing):
> "push toward Shannon theoretical floor in absolute minimum wall clock — on wake harvest 5 in-flight subagents (...) + start next Path-B Ω-OPT level (multi-pass IMP composition); harvest Lightning jobs T020205 + T020152 too"

NEW (post-contest, long-term substrate co-design):
> "push toward sub-0.17 substrate co-design — on wake check Phase A ablation status + Track 2 scaffold progress + harden one preflight gate or one tool sophistication item (whichever has highest EV-per-tick); never bolt-on without a co-design rationale; council-deliberate before any new dispatch tier"

This reframes the loop from race-velocity to research-discipline, which matches the post-contest strategic phase.

## Cross-references

- `.omx/research/track_1_co_designed_substrate_design_20260508_claude.md` — original design memo
- `feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508.md` — Tier 0 source (2,228 B headroom math)
- `feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md` — Shannon's 148-162 KB joint floor
- Task #307 PARADIGM-δεζ — this work's parent
- Task #308 PHASE 4 INTEGRATION — Phase D umbrella

## Verdict

Phase A0 + A2 (free / CPU-only) launch immediately on operator green-light. Phase A1+A3-A6 launch in parallel within ~2h after Phase A0 lands the MDL calculator. Phase B (Track 2 scaffold) starts immediately as parallel workstream. Phase C deferred ~24h pending Phase A results. Phase D continuous always-on.

Total Phase A spend: $55. Authorized.
