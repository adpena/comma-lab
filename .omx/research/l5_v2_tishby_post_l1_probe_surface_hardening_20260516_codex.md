# L5 v2 Tishby post-L1 probe surface hardening

## Change

The Tishby IB-pure post-L1 decision basis is now preserved in committed
`.omx/research` artifacts and consumed by the L5-v2 asymptotic candidate
surface.

## Why

The original Tishby row treated `.omx/state/h_latent_given_scorer_class_tishby_ib_pure.json`
as an expected first artifact. `.omx/state/**/*.json` is ignored, so that
decision basis was local-only and easy to lose in another checkout. This is a
no-signal-loss problem for L5-v2 routing.

## Evidence Preserved

- `.omx/research/tishby_ib_pure_d4_probe_20260516_codex.json`
  - Verdict: `INDEPENDENT`
  - Mutual information: `0.006385502752311645` bits
  - Wyner-Ziv gain ceiling fraction: `0.0009071575012169224`
- `.omx/research/tishby_ib_pure_variational_ib_tractability_20260516_codex.json`
  - Verdict: `TRACTABLE`
  - Worst-case gradient SNR: `4.9758341440833425`

## Routing Effect

The L5-v2 candidate surface now records a Tishby post-L1 blocker:

`tishby_path_vib_paid_dispatch_blocked_d4_independent_scorer_class_probe`

That blocks spend on the measured Path-VIB side-info interpretation until a
meaningful scorer-conditioning signal, Path-MINE redesign, or beta-sweep
justification exists. It does not kill Tishby IB-pure as a family.

## Authority

This is a planning-surface and state-preservation hardening. It is not a score
claim, not a promotion result, and not an exact-eval dispatch authorization.
