# RL-for-LLM transfer → costate-organ / surrogate / verdict-clearance (2026-07-12)

**Source** `[FROM-LITERATURE]`: Arjun Kocher, "RL Algorithm Questions", https://www.k-a.in/rl-algo.html
(answers to Xiuyu Li's question set). Fetched via the `.md` source (page renders markdown+KaTeX
client-side). Reference/design-input only — **pointer 0.18804 [contest-CPU] UNMOVED**, `score_claim=false`.

**Unifying read** `[DERIVED]`: the whole page is ONE object — *variance-reduction of a policy gradient*
`∇J = E[∇log π · (G − b)]` on a huge action space — told through the GRPO family. The load-bearing
identity: subtracting ANY baseline `b` is unbiased (`E[∇log π · b] = 0`) but kills variance; optimal
`b = E[G_t]`. The whole GRPO lineage is "the value function over sequences is too hard to learn, so
replace the critic with a group-relative baseline." That is *exactly* our costate-organ situation.

## The three folded design inputs (graded; each mapped to a live task)

1. **RLOO / GRPO leave-one-out group baseline → #434 (SOL-ultra organ / n=1 data-starvation).**
   `[DERIVED, NEW-FRAMING]` LLM-RL's answer to "the critic/value-fn is data-hungry and hard to learn"
   was NOT "get more data" — it was drop the critic, use a **group-relative (leave-one-out) baseline**
   over sampled rollouts: `A_i = (r_i − mean_{j≠i} r_j)` (unbiased, no critic, no dataset). Transfer:
   before/instead of training a data-hungry costate SURROGATE to cure the n=1 starvation, evaluate the
   organ's advantage via a **leave-one-out baseline over sampled witness perturbations** through the
   exact R objective. Cheaper than a surrogate and sidesteps the starvation. Composes with SFESS
   control-variates (#396) — both are the same variance-reduction move. Backtest-gated per #434.

2. **Dr.GRPO / DAPO zero-variance-group SKIP → #319 (band-spans-0) + verdict-clearance.**
   `[DERIVED, SAME-MATH-WE-HALF-HAD]` GRPO's std-normalization degenerates when a group is all-correct
   or all-wrong (zero variance → no learning signal); Dr.GRPO removes std-norm + skips those groups,
   DAPO does dynamic-sampling (filter all-right/all-wrong prompts). This IS our "through-R evaluator
   band spans 0 → no gradient signal" (#319 SimpleTES). Adopt the rule verbatim as the K>1-emit / SKIP
   trigger in the shadow_controller recommendation shape: when the sampled advantage group has
   ~zero variance, skip the update (or widen to K>1), don't divide by a ~0 std.

3. **OPD (On-Policy Distillation) → #449 distilled-surrogate arm + #428/#431.**
   `[DERIVED, INDEPENDENT CORROBORATION]` OPD (DeepSeek-V4 collapse step; arXiv 2602.12125): the student
   generates its OWN trajectories, the teacher supervises on THOSE (dense per-token
   `r_t = log π*(y_t|·)/π_ref(y_t|·)`; the reference cancels in the objective, so it can be any size).
   This independently confirms the goldmine-hunt codistillation finding for #449: a frozen-SegNet
   surrogate must be trained **ON-POLICY** on the renders as the witness PRODUCES them, never on a fixed
   offline dataset (naive forward/logit distillation is already formulation-falsified — forward
   agreement did NOT preserve exact-teacher descent, base memo §1.2). Fold the OPD on-policy discipline
   into the master #451 surrogate reconciliation.

## Humility calibration (for #434)
`[FROM-LITERATURE]` Kocher's frontier answer: RL mostly expands *reliable use* of existing capability +
test-time search; whether it pushes past the pretraining frontier is open (arXiv 2504.13837). Prior to
hold: expect the organ + synthetic data to SHARPEN reliable use of the witness's existing basin, not to
CREATE new capability. Don't over-claim.

## Bonus transfers (lower-priority, logged not folded)
- **SimKO** asymmetric top-K boost (correct) / top-1 penalty (incorrect) at high-entropy tokens ≈ our
  per-class-λ island-birth (boost unborn classes) — probe-worthy analogy, not yet folded.
- **CISPO** clip-the-ratio-in-gradient-not-loss (preserve signal for clipped samples) — informs how the
  costate controller handles out-of-trust-region proposals without zeroing their gradient.

## Triality
`[no new equation / DSL lever / archive]` — pure reference/routing. DAG FEED row appended to
`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`. Consumed by: master #451 prompt (folded),
task #434 + #319 (pinned). MEMORY.md intentionally NOT touched (already ≥17KB, over the <17KB budget;
this memo + the DAG FEED are the durable landing).
