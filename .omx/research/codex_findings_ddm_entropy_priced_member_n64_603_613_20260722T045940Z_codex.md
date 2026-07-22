---
title: Codex adversarial round-1 review of DDM entropy-priced member solve v3
date_utc: 2026-07-22T04:59:40Z
task: 603
feeds_task: 613
lane_id: lane_ddm_mdl_member_solve_v3_entropy_603_613_20260722
review_round: 1
review_disposition: ACCEPT_SCOPED_N64_ENTROPY_APPARATUS_AND_FORMULATION_NEGATIVE_WITH_MAIN_API_STABILITY_REVIEW
research_only: true
score_claim: false
main_landing_review_required: true
---

# Adversarial disposition

The exact entropy grammar and n64 rate-gradient result are supported. The negative is only for the
exhaustive power set of the three maximal safe-zero residual collapses under absolute per-stratum
membership. It does not close coefficient-level rate-distortion search, nonzero residual
quantization, anchor/gradient optimization, pre-uint8 member states, n600, or the wider DDM family.

# Falsification attempts

1. **Was v2's fixed-width wall merely relabeled? — FALSIFIED.** Exact candidate bytes have eight
   unique values from 45,369 down to 6,553. `len(final entropy ZIP)` is measured after every semantic
   change and is the selection rate.
2. **Was the 83.482% baseline reduction lossy or receiver-external? — FALSIFIED.** All six semantic
   payload SHA-256 values reconstruct byte-identically before the existing integer uint8 chart
   receiver. Parse/re-encode identity and compiler/decode determinism x2 are green.
3. **Was a favored coder assumed without measurement? — FALSIFIED.** Each eligible stream/transform
   measured Brotli, LZMA, AQc1, rank-Huffman, and split Rice/Golomb/zlib candidates. Winners were
   Brotli 20, LZMA 12, and split Rice 16 across the eight archives. AQc1 and Huffman non-selection is
   not promoted to a family verdict.
4. **Did aggregate membership create a false 6,553-byte knee? — CONFIRMED HAZARD, CORRECTLY REFUSED.**
   Mask 7 has aggregate membership 0.493612130483, but Road, Lane, MyCar, and Movable membership are
   each zero. Every rung has zero feasible candidates; published bytes are explicitly diagnostic.
5. **Was the candidate search claimed exhaustive beyond its real scope? — NO.** It is exhaustive only
   over eight subsets of three maximal safe-zero residual collapses. The receipt and verdict_scope
   preserve that boundary.
6. **Could a malformed/trailing entropy section be silently ignored? — FALSIFIED.** Every frame binds
   semantic length/hash, canonical-transform length, coded length, and exact exhaustion. Thirty-nine
   sampled archive-home mutations all refused; all homes cover exactly one final archive byte.
7. **Was resumability performative? — FALSIFIED.** The run stopped after subset 3 and rung 1, resumed
   using config/DSL/argv/candidate-table-bound envelopes, and post-run validation loaded all 13
   immutable checkpoints.
8. **Does the cached target crosscheck establish exact batch identity? — NO.** Target-versus-cache
   agreement remains 0.999873638153. The paired batch16 target-versus-description measurement is
   deterministic, but the cache caveat remains visible.
9. **Did the run establish score or promotion evidence? — FALSIFIED.** This is local macOS-CPU frozen-
   SegNet advisory membership plus exact archive accounting. No PoseNet score, contest evaluator,
   contest CPU/CUDA replay, candidate archive, or pointer movement exists.

The 2026-07-22T04:53:58Z MAIN coder-survey coordination arrived after measurement and is consumed,
not silently substituted. Its PPCS/event/exception payload results use different stream grammars;
this arm measured the actual six Task #603 semantic payloads. Both agree that adaptive arithmetic
must not inherit a global win claim. MAIN should compare the survey landing against this tournament
at merge time without transferring byte counts across payload families.

# MAIN review requirement

Before landing, MAIN must re-run the focused suite, validate the immutable receipt/checkpoints,
confirm the aggregate-versus-strata refusal, compare the separate optimal-coder survey landing, and
review the entropy module's imports of existing private arithmetic/Huffman helpers for API-stability
risk. Those helpers are reused rather than forked, but the coupling deserves explicit owner review
at merge time.

0.1910828242 [contest-CPU] — unchanged.
