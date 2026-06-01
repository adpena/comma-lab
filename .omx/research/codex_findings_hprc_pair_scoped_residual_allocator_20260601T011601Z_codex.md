# Codex Findings - HPRC Pair-Scoped Residual Allocator

UTC: 2026-06-01T01:16:01Z
Author: Codex
Authority: MLX-local advisory plus receiver proof; no contest CPU/CUDA score claim

## Finding

The global HPRC residual shrink grid was leaving score on the table by applying
one token transform to every pair.  The full-video MLX component arrays already
exposed pair-local deltas, so the allocator now emits executable pair-scoped
residual transforms.

The first full-video pair-scoped candidate applies:

`threshold_abs_le_pairs=3@...`

to 284 measured low-value pairs while protecting 316 locally harmful pairs.

## Measured Artifact

Committed evidence directory:

`.omx/research/hprc_pair_scoped_threshold_abs_le3_receiver_proof_20260601T011601Z_codex/`

SSD scorer-cache directory:

`/Volumes/VertigoDataTier/pact/hprc_pair_scoped_residual_full600_20260601T011601Z`

Receiver-proof SSD source:

`/Volumes/VertigoDataTier/pact/hprc_pair_scoped_threshold_abs_le3_receiver_proof_20260601T011601Z`

Archive:

- bytes: `859923`
- sha256: `234ef386b71d4bb439517ecf7dab7623e737b0c54da66456fc4d189a8d6c0ec3`
- receiver proof: `receiver_proof/hprc_receiver_proof.json`
- expected decoded raw bytes: `3662409600`
- false authority preserved: `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`

## MLX Advisory Delta

Compared with the HPRC baseline in the same full-video run:

- baseline archive bytes: `1163860`
- pair-scoped archive bytes: `859923`
- bytes removed: `303937`
- delta non-rate score: `-1.949817179715751`
- delta rate score: `-0.20237917263509342`
- delta total MLX advisory score: `-2.152196352350849`
- marginal status: `cut_candidate_distortion_nonworse`

This is better than the earlier global `threshold_abs_le=3` advisory result,
despite removing fewer bytes, because it preserves pairs where the global
threshold was locally harmful.  That is the desired allocator behavior:
score-value protection first, then rate savings.

## Engineering Changes

- Added `threshold_abs_le_pairs=<threshold>@<ranges>` to the compact HPRC
  receiver residual transform grammar.
- Added deterministic bounded variant slugs so large pair plans cannot exceed
  filesystem path limits.
- Extended the shrink backlog with `pair_scoped_residual_candidate_rows`, using
  the measured pair-local MLX non-rate delta versus the per-pair archive-rate
  price.

## Blockers

- MLX-local response remains advisory and cannot claim contest score.
- Exact contest CPU/CUDA replay has not been executed.
- Class-region and boundary scopes still need scorer-logit or boundary-cache
  extension before the allocator can optimize below pair granularity.

## Next Action

Make pair-scoped HPRC residual candidates first-class bounded-runner inputs,
then add cache reuse/vectorized scorer-response execution.  The full-video
measurement took `1221.861698627472` seconds, with direct cache no longer the
dominant cost; the scorer-response pass is now the performance target.
