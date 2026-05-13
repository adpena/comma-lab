# Adversarial review follow-ups captured from completed subagents (2026-05-13)

This ledger preserves the completed review-agent signal so it can be routed into
future bug-fix waves. No finding below is a score claim.

## Highest priority correctness follow-ups

1. Score-aware substrate losses may underweight PoseNet relative to the contest
   formula. Review `src/tac/substrates/*/score_aware_loss.py` against
   `tac.losses.core` and add equality tests for the canonical `100*seg +
   sqrt(10*pose)` semantics.
2. The new substrate score-aware helper must not drift from canonical scorer
   preprocessing. Make it a thin wrapper over `tac.losses` or prove equivalence.
3. Provider claim lifecycle can fail open in `tools/operator_authorize.py` when a
   provider command returns nonzero after an active claim. Pre-validate provider
   dependencies before claim and always write terminal failure/unknown claim rows.
4. PacketIR optimize mode must not accept a bare boolean
   `runtime_consumption_proof=True`; require a typed proof tied to archive SHA,
   runtime SHA, changed sections, and consumed bytes.
5. PR106 PacketIR identity/decode proof is strong but still distinct from full
   `inflate.sh archive_dir output_dir file_list` parity. Add an explicit
   inflate-parity harness or label prefix/non-inflate proofs accordingly.
6. Composition ranking currently carries blocker-bearing rows. Dispatch queues
   should expose a clean blocker-free top-k and keep blocked rows in an operator
   review bucket.
7. SIREN residual sparse Fourier atoms must treat non-DC real FFT modes as
   conjugate pairs or a real Fourier basis; single complex-bin atoms can
   under-reconstruct by construction.
8. Dense SIREN residual tests should decode through the runtime, not only inspect
   bytes.
9. GHA CPU wrapper release asset URLs must use public `browser_download_url`, not
   API `.url`, before any future CPU-axis dispatch relies on it.
10. Kaggle artifacts must remain proxy/diagnostic unless a future target contract
    explicitly supplies exact-eval lifecycle, runtime closure, claim, harvest, and
    adjudication. This turn fixed the PR106 score-table harvest path accordingly.

## Current routing

- Immediate fixed this turn:
  - Kaggle PR106 y-shift harvester added.
  - Kaggle score-table ingest recognizes latent and y-shift manifests.
  - Kaggle ingest now uses the current download manifest as a stale-file
    allowlist.
  - Operator briefing now surfaces y-shift Kaggle bundle/harvest tools.
- Deferred but high value:
  - Provider claim terminal-row hardening.
  - Typed PacketIR runtime-consumption proof.
  - Full PR106 `inflate.sh` parity harness.
  - Canonical substrate loss wrapper/equality tests.
  - SIREN Fourier atom correction.
