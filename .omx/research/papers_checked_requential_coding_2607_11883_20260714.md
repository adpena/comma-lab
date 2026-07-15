# PAPER CHECKED — arXiv 2607.11883 "Requential Coding" (on-policy distillation as universal coding)

**Assessed 2026-07-14** (organ-n1-RL arm; formalized as a papers_checked memo 2026-07-14 PM so recall is O(1) — it was previously only in `codex_findings_warmstart_organ_n1_rl_20260714_codex.md §10`, which forced a dig). MEANS, pointer UNMOVED 0.19108/0.18804. HTML: arxiv.org/html/2607.11883.

## Method (engaged)
On-policy distillation framed as UNIVERSAL CODING: draw public proposals from the STUDENT's own generative
distribution; the teacher accepts an index whose marginal equals the teacher distribution; send a prefix-free
universal code for that index. Expected message length = cumulative teacher↔student KL + log REC overhead
(Appendix B: martingale variance bound on realized-message fluctuations). EMA-teacher smoothing + iso-loss
projection shorten the trajectory. The valid prefix-free code enters a PAC-Bayes bound; increasing code-debt
under data repetition predicts overfitting.

## Two applications, two verdicts
1. **ORGAN n=1 curriculum (BUILT + MEASURED — small INSTANCE win, NOT a capacity floor).** Divergence: our
   costate organ is a continuous point/posterior predictor, not a normalized generative model with a
   shared-randomness proposal decoder. Warm-started: re-derived `KL_bits = δS²/(2·var_S·ln2)` under a
   shared-variance Gaussian posterior; built `costate_requential_curriculum.py` (NumPy-fp32 + MLX parity) —
   protect half each real row's replay mass, allocate the other half ∝ past-prefix disagreement, cap 2×,
   fit measured targets only. MEASURED: disagreement-replay WF-MAE **0.002463** vs uniform 0.002496 (per-class
   +29.19% vs uniform, still **3.66× persistence**); the 0.6076-bit Gaussian-KL proxy is NOT a REC code /
   capacity floor / PAC-Bayes cert / validated overfit predictor (4/7 variances at fp32 floor, last interval
   68.02%, late-debt slope positive = warning only). Verdict: INSTANCE improvement. A real capacity measure
   needs normalized organ proposal/teacher distributions + a decodable prefix-free stream + non-floor variance
   custody + independent validation.
2. **RATE-side: requential coding = the MDL PARENT of the margin-conditional flip coder (#226/#307) — NOT
   BUILT, the genuinely-open + more score-relevant thread.** Coding the d_seg FLIP RESIDUAL via a requential/
   universal-MDL code (student = the decoder-regenerated margin prior, teacher = the true flip set) is the
   principled parent of the margin-conditional residual coder (#72/#226) + the contour-string/digital-
   straightness flip coder (#307, vs the 0.65 B/flip GO bar). ROUTE to #226/#307: does the requential/MDL
   framing beat the current ~8 bits/flip toward the published ~1-1.5 bits/contour-px floor? This touches
   d_seg RATE (score-relevant), unlike the organ curriculum (apparatus).

## Verdict
Fully assessed + organ-curriculum BUILT/MEASURED (small INSTANCE win). No re-work on the organ side. The LIVE
follow-on is the RATE handoff (#226/#307): requential/MDL coding of the d_seg flip residual — the one thread
that touches the pointer. No standalone code repo (method is in the paper; our implementation is
`costate_requential_curriculum.py`). Sisters: `[[n1_organ_capacity_ceiling_shrinkage_physics_residual_measured_20260714]]`,
`[[paper_warm_start_from_assumption_divergence_not_route_or_dismiss_20260714]]`. MEANS.
