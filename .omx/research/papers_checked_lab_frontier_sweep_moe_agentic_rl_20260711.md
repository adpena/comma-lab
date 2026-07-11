# Papers-checked: 2026 lab-frontier sweep (DeepSeek · ByteDance/Seed · Alibaba/Qwen · Z.ai/Zhipu · Tencent · Google/DeepMind · Meta) — MoE routing + agentic-RL + self-improving agents, mapped to the #426 costate organ

Date: 2026-07-11 · live web survey (subagent, $0; labels MEASURED-by-paper / PRESS-SECONDARY /
PRE-CUTOFF-KNOWN carried per item in the synthesis) · routed during the #426 build. Full synthesis
recorded in `amortized_operator_pontryagin_loop_cluster_20260711.md` §3b. Anti-re-research rows:

- **DeepSeek**: aux-loss-free load balancing 2408.15664 (per-expert bias INTEGRAL CONTROLLER on
  utilization — balance-by-controller not balance-by-loss; the 2026 SOTA answer) · V3 2412.19437 ·
  V4 2606.19348 (>10 specialist teachers → On-Policy Distillation reverse-KL = evidence-weighted
  stacking at expert scale; OPD detail PRESS) · Engram 2601.07372 (hashed lookup memory = second
  sparsity axis) · GRPO 2402.03300.
- **ByteDance/Seed**: DAPO 2503.14476 (clip-higher · dynamic zero-signal-group dropping ·
  token-level loss) · VAPO 2504.05118 (learned critic BEATS value-free — but at 5k steps × large
  batches = the crossover regime marker, 3+ orders beyond one campaign trajectory) · DORA/verl
  async infra (PRESS) · UltraMem ~2411.12364.
- **Alibaba/Qwen**: GSPO 2507.18071 (SEQUENCE-level importance ratio/clipping; stabilizes MoE RL
  without router-freezing) · global-batch balance ~2501.11873 · AgenticQwen 2604.21590.
- **Z.ai/Zhipu**: SAO 2607.07508 (own TIER-1 row `papers_checked_sao_*`) · AgentRL 2510.04206
  (fully-async pipeline; cross-policy sampling; task advantage normalization) · GLM-5 2602.15763.
- **Tencent**: Hy3 (PRESS; systems+openness, no routing novelty published) · TurboS 2505.15431.
- **Google/DeepMind**: expert-choice 2202.09368 (perfect balance BY CONSTRUCTION, no aux loss) ·
  Mixture-of-Depths 2404.02258 · AlphaEvolve (production propose-then-hard-evaluate System-2 —
  the reference shape for our SpawnTicket + act-gate loop).
- **Meta/FAIR**: Self-Rewarding 2401.10020 → Meta-Rewarding 2407.19594 (shrinkage of the judge) ·
  Memory Layers ~2412.09764 · Collaborative Reasoner (site).
- Adjacent: Routing-Free MoE 2604.00801 (self-activating experts still REQUIRE an explicit
  load-balancing framework — nobody ships ungated self-activation) · ROLL Flash 2510.11345 ·
  memory-agents survey 2603.07670.

**VERDICT (the ask: does the frontier change our measured conclusion?): NO — it CONFIRMS all
four measured conclusions and only informs the growth path.** (1) closed-form solve at n≈5
CONFIRMED (every frontier stability move REMOVES learned machinery when signal is scarce; VAPO
sets the crossover expectation, not a refutation); (2) raw self-activation failure CONFIRMED
(DeepSeek's external bias controller exists precisely because loss-side self-balancing fails;
routing-free MoE needs bolted-on balancing); (3) evidence-shrunk stacking = the INDUSTRIAL
DEFAULT (V4 specialist panel + reverse-KL); (4) Rashomon-gated System-2 CONFIRMED in production
(AlphaEvolve: System-2 proposes, hard evaluator gates, invoked on hard instances).

**GROWTH-PATH ADOPTIONS (ordered, folded into the duty queue):** (i) NOW/$0: DeepSeek-style
per-lens bias integral controller as a starvation/degeneracy GUARD under the MDL-prior stacking
(a load mechanism, never an evidence mechanism); (ii) n_traj≳tens: SAO async single-rollout λ
updates + strict double-side clipping + GSPO trajectory-level importance + DAPO zero-signal
dropping; (iii) VAPO-regime data: A/B a small learned λ-critic vs the ridge solve; (iv) Engram/
Memory-Layers-style hashed telemetry→prior lookup memory.

verdict_scope: read-level triage; no internal lane opened/killed; growth items are duty-queue
notes. Pointer 0.19108282 UNMOVED (design = MEANS).
