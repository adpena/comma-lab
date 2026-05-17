# L5 v2 Rudin post-L1 probe surface hardening

## Change

The Rudin floor design-time disambiguator result is now preserved as a committed
`.omx/research` JSON artifact and consumed by the L5-v2 asymptotic candidate
surface.

## Evidence

- Tool: `tools/probe_rudin_floor_substrate_disambiguator.py`
- Artifact: `.omx/research/rudin_floor_proxy_disambiguator_20260516_codex.json`
- Verdict: `MEANINGFUL_INTERPRETABILITY`
- Interpretability-tax proxy: `0.01784276143750971`
- Pixels sampled: `40150`
- Frames sampled: `10`
- Authority: design-time proxy only; `score_claim=false`,
  `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`

## Routing Effect

The L5-v2 surface now adds:

`rudin_proxy_positive_requires_t3_ratification_and_scorer_probe`

This preserves the positive Rudin signal while still blocking paid dispatch
until T3 ratification plus a scorer-bearing probe exist.

## Authority

This is planning-surface hardening and signal preservation. It is not a score
claim, not a rank change, not a promotion result, and not an exact-eval dispatch
authorization.
