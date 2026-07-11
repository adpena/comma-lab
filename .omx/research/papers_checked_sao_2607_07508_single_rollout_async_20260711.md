# Papers-checked: SAO — Single-Rollout Asynchronous Optimization for Agentic RL (arXiv 2607.07508, Hou/Li/Tang/Dong, Tsinghua/Z.ai lineage) — TIER-1 TRAINING-DESIGN INPUT for the #426 costate organ

Date: 2026-07-11 · operator-supplied · read live (arXiv abstract fetched this pass) · routed
during the #426 build. Sister of `papers_checked_nvfp4rl_qerl_solrl_explore_decouple_20260711`
+ `paper_harvest_v9cgauge_20260711` (ANR 2606.16303).

**MEASURED-by-paper (abstract-level, fetched):** (1) **asynchronous optimization** — rollout
decoupled from optimization; stability of async/stale off-policy updates via **strict double-side
token-level clipping** ("stably for one thousand steps"); (2) **single-rollout sampling** — ONE
rollout per prompt replacing GRPO-style group-wise sampling; reduces off-policy divergence
(tighter coupling between updates and data freshness) and **improves generalization**;
(3) results: outperforms GRPO variants on SWE-Bench Verified / BeyondAIME / IMOAnswerBench;
effective in "simulated online learning" (evolving environments); used in training GLM-5.2.

**TRANSFER to the #426 costate organ (TIER-1 — this paper is written for our regime):**
- **Async λ-training off the control-loop critical path.** The organ's SENSE→DECIDE→ACT loop
  never blocks on λ-network training: the controller keeps sensing/acting on the CURRENT fitted
  field while retraining happens asynchronously on the accruing trajectory (composes with the
  Sol-RL explore-cheap/commit-exact decoupling). Design note recorded in the cluster memo §5c;
  the stability ingredient to adopt when the learned lenses train online = SAO's double-side
  clipping analogue: **clip the per-component field update against the previous fit** (a trust
  region on Λ between refits) so a stale/async refit cannot swing the DECIDE layer. BUILD-OWED
  (enters when λ-training goes online; today's refit-per-invocation is trivially synchronous).
- **Single-rollout is OUR regime, named.** We have exactly ONE campaign rollout (the #205 run);
  the backtest flagged "limited trajectory → overfit" as the top risk, and the measured tournament
  confirmed it (MLP/GRU/DeepONet lose to the SOLVE). SAO's evidence that single-rollout training
  with tight freshness-coupling GENERALIZES BETTER than group-batching is the literature-side
  de-risk for training on our single trajectory — provided updates stay conservative (their
  clipping ↔ our ridge/L1 + evidence-shrunk pooling; same role, different substrate).
- **System-2/agent-spawn:** their agentic-RL task class (long-horizon tool use) is the regime our
  SpawnTicket agents operate in; no mechanism transfer needed there beyond the async principle.

**Honest limits:** abstract-level read (full-text mechanisms like the exact clipping bounds not
extracted); LLM-RL substrate ≠ our tiny-field regression — what transfers is the TRAINING-REGIME
pattern (async + single-rollout + conservative clipping), not hyperparameters. verdict_scope:
read-level; no lane opened/killed; the async-λ trust-region is a BUILD-OWED design note on the
duty queue's growth path.

Pointer 0.19108282 UNMOVED (training-design = MEANS).
