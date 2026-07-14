# Papers checked 2026-07-14 — GenCeption + Looped-TTT (bookmarks, not witness levers)

Two operator-dropped items assessed 2026-07-14. Both DOMINATED for the witness line; recorded here so
the papers-checked ledger (L55) prevents re-research. Sisters: `[[proactive-recall-consult-own-research-before-concluding]]`,
`[[relative-not-absolute-significance-near-goal-dont-orphan-small-deltaS]]`.

**Recall-discipline note (this session):** the operator ALSO dropped arXiv **2607.09061** (locality/length-gen,
Madan/Memisevic) and **2607.07508** (SAO single-rollout async RL, Zhipu/GLM). BOTH were already in the
corpus from 2026-07-11 — SAO has a full `papers_checked_sao_2607_07508_single_rollout_async_20260711.md`
(filed TIER-1 into the RL/post-training regime addendum, #433); locality is cited in
`whole_teacher_distilled_student_20260713.md` + `amortized_operator_pontryagin_loop_cluster_20260711.md`.
I re-assessed both before recalling — my verdicts were consistent with the priors, but the effort was
redundant. LESSON (sharpens L55): grep an operator-dropped paper id against `papers_checked_*` +
`arxiv_scout_seen.jsonl` BEFORE assessing. `arxiv_scout._known_ids()` automates exactly this dedup.

## GenCeption — arXiv 2607.09024 (multimodal foundation-model self-consistency)
- **What:** iterative image→description→image self-consistency degradation as a training-free MLLM
  hallucination/semantic-drift benchmark (GENCEPTION score over N regeneration cycles).
- **Regime divergence:** foundation-model evaluation over many samples; our regime is n=1 single-clip
  witness fit of a frozen-scorer argmax. No transfer to the witness or the (supervised, n=1) costate organ.
- **Verdict:** DOMINATED. Bookmark ONLY to the organ-data-efficiency cluster (#434 synthetic-data / #499
  n=1 learning-theory / #211 amortized pre-seeding) as a *self-consistency-as-signal* reference — not a
  witness lever. No OSS harvested (benchmark repo is eval-harness, not a reusable witness primitive).

## Looped-TTT — alvinzh04.github.io/blog/looped-ttt.html (Loop-TTT; Ouro-1.4B recurrent-depth LM)
- **What:** inference-time adaptation on recurrent-depth transformers — entropy-minimize (label-free) over
  prompt tokens, ONE Adam step on the 97 RMSNorm scale vectors (0.014% of params), decode, reset per
  batch. Fixes "overthinking" (accuracy declines past training recurrence depth r=4); GSM8K 0.766→0.848.
  OSS: github.com/AlvinZH04/Loop-TTT.
- **Regime divergence:** recurrent-transformer CoT reasoning with a label-free entropy loss; our witness
  HAS the target (frozen argmax) and is a from-scratch coord-INR, no recurrence/CoT.
- **The transferable signal (their OWN ablation):** *"most of the gain is a reusable calibration vector
  capturable with NO test-time gradient."* External evidence that per-instance TTT gradient is largely
  replaceable by a cheap FIXED calibration — cross-checks our solve-don't-train (#342) + costate
  amortization (#426) + the yopo per-step-validation economics killer (#454).
- **Verdict:** DOMINATED as a witness lever; WARM-START into the **costate-economics / solve-don't-train**
  cluster (#454/#426/#342) with that ablation framing — "test the boundary where a no-gradient calibration
  replaces per-instance costate reuse." Not a current-path item; no $0 witness probe.

**Pointer:** UNMOVED 0.18804 / 0.19108 — all four are MEANS, none a score-mover.
