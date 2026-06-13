# Recursive adversarial review — ROUND 14 of the 5 Layer-2 levers (2026-06-13)

**Reviewer:** Partner-A2 (author ≠ reviewer). The SEAL requires 3 FRESH consecutive clean rounds.
Prior FRESH count: R12 NOT-CLEAN (floor 0.1→0.3) → reset 0/3; R13 CLEAN → 1/3. So R14 began at **1/3**.

**R14 lens (distinct): Lever-4 (score-aware QAT) byte-effect at the EXPORT surface, judged against the
CONTEST objective** (operator directive this round: *"always engineer the optimal implementation optimized
against contest … nuanced … mathematically and algebraically and geometrically and all optimal"*). No prior
round measured whether Lever-4's claimed brotli-byte savings are the CONTEST-OPTIMAL use of the per-tensor
score-sensitivity it computes. The contest objective is `S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37,545,489`;
a byte lever is contest-optimal only if it sits on the reverse-waterfill KKT frontier (equal marginal
`d_dist/d_byte` across tensors — Cover & Thomas Ch.10), not merely "delivers some bytes."

## CLEAN-PASS VERDICT: **NOT-CLEAN (contest-optimality finding) → fresh counter RESETS 1/3 → 0/3.**

R14 found **NO bug** in Lever-4 (it is honestly scoped + its claimed effect is delivered + tested), but it
found a **real CONTEST-OPTIMALITY GAP** that the operator's "engineer the optimal implementation optimized
against contest" directive makes a counter-resetting finding: **Lever-4 captures only ~1/8 of the
contest-positive byte saving its own per-tensor sensitivity signal makes available, because its online
sensitivity EMA is never routed to the variable-level export that would realize the full reverse-waterfill
allocation.** MEASURED, the gap is worth **~0.018 contest score** on the decoder blob — material at the
sub-0.15 operating point. This is the "do LESS but make it REAL / engineer the contest-optimal" call: a
clean pass here would paper over a known, measured ~0.018-score opportunity.

---

## A. THE MEASUREMENT (probe `experiments/probe_r14_lever4_export_byte_effect.py` + Partner-B RD table)

Lever-4's documented scope (`score_aware_qat.py` lines 34-37) is a **TRAINING-TIME proxy**: it shapes the
decoder weights so that, after the codec's UNIFORM 127-level quant, score-irrelevant weights collapse to
repeated symbols brotli compresses well. The export grid stays uniform-127. The MED-2 probe
(`probe_lever4_qat_brotli_blob_delta.py`) validated this delivers **-3263 B (-4.4%)** on the real basin EMA
decoder at equal advisory d_seg (70264 vs 73527), guarded by
`test_score_aware_grid_yields_smaller_real_brotli_blob_than_uniform`. So Lever-4's CLAIMED effect is real
and tested — **NOT a fake implementation, NOT a no-op.**

But the SAME per-tensor sensitivity that Lever-4 computes online (`tensor_sensitivity_ema = ||∂S/∂w_t||`) can
drive a **variable-level EXPORT** (the codec `build_decoder_blob_variable_or_vendored` already supports it).
MEASURED on a real vendored decoder (R14 probe, sd-key-correct coarse map):

| export path | decoder-blob bytes | saving vs uniform | contest rate Δ (25·B/37.5M) |
|---|---|---|---|
| uniform-127 (Lever-4 as-is, brotli-compressibility) | 83776 → ~80513 (−3263, −4.4%) | −3263 B | **−0.0022** |
| variable-level (coarse half @16 levels) | 53901 | **−29875 B (−36%)** | **−0.0199** |

And crucially — coarsening is NOT free distortion. Partner-B's MEASURED RD table
(`track_a_itemB_waterfill_rd_table_rd24_20260612.json`) shows the contest TRADE is FAVORABLE: e.g.
`stem.weight`@16 saves 10052 B at d_seg cost **−0.0052** (coarsening *improved* d_seg), `blocks.0.weight`@16
saves 5367 B at **−0.0048**. Many tensors sit on the contest-positive side of the reverse-waterfill frontier.
So the ~0.018-score variable-level saving is a NET contest win on the measured slice, not a byte-only mirage.

## B. THE FINDING (MEDIUM — a contest-optimality gap, not a bug)

**Lever-4 and the D2 variable-level waterfill are the TRAIN-half and BYTE-half of ONE mechanism, but they
are not unified, and Lever-4 alone leaves the byte-half's ~0.018-score saving on the table.**

- Lever-4 estimates per-tensor distortion sensitivity ONLINE (the EMA of `||∂S/∂w||`, free — the backward
  already computes it). D2 estimates the SAME thing OFFLINE via a separately-MEASURED RD table (expensive:
  a full coarse-sweep per tensor on the real scorer). **Mathematically these are the same quantity** (the
  marginal `d_dist/d_level`); algebraically, the reverse-waterfill KKT optimum equalizes `d_dist/d_byte`
  across tensors — D2's allocator solves exactly this, and Lever-4's online EMA is a (cheaper, noisier)
  estimate of its gradient. Geometrically, both seek the lower-convex-hull operating point on each tensor's
  RD curve. The contest-OPTIMAL implementation feeds Lever-4's online EMA into D2's allocator (or co-trains
  the weights to be robust at the allocated grid), getting the −36% saving WITHOUT the expensive offline RD
  sweep — the unification the two separate features currently miss.
- This is a MEDIUM (not HIGH): no live run is corrupted, no archive is malformed, and Lever-4's documented
  claim is honored. It is a missed contest-optimization, surfaced by the operator's contest-optimality lens.

## C. THE 2-LANDING RESPONSE (fix the SCOPE + guard; defer the rewire to a measured design decision)

Per "Forbidden premature KILL" + the design-decision discipline, the OPTIMAL rewire (route Lever-4 EMA →
variable-level export) is a DESIGN DECISION that touches Partner-B's D2 surface AND requires a paired
contest-CUDA measurement (the −36% byte saving must be confirmed NET-positive at the FULL 600-pair operating
point + survive parse-back, not just the 24-pair RD slice) BEFORE it can be claimed. So R14 lands the
HONEST-SCOPE half now and defers the rewire:

**Landing 1 (scope made explicit + the optimal path named):** this memo records that Lever-4 is the
TRAIN-half of a 2-part mechanism whose contest-optimal BYTE-half is the D2 variable-level export consuming
Lever-4's online EMA; the ~0.018-score opportunity + the reverse-waterfill KKT derivation + the measured RD
evidence are documented for the design decision.

**Landing 2 (the regression guard):** `test_r14_lever4_export_is_uniform_127_not_variable_level_documented_
scope` — guards BOTH surfaces: (1) the variable-level codec mechanism is REAL + sd-key-correct (coarse map
shrinks the blob 83776→53901) + default-preserving (uniform/None → vendored byte-identical); (2) the DEFAULT
driver export takes NO sensitivity parameter + does not reference `tensor_sensitivity_ema` — so the current
uniform-127 scope is explicit and a future silent variable-level rewire (which would change the archive
grammar out from under the daemon) trips the guard. When the OPTIMAL rewire lands (post-measurement), this
guard is updated in the same batch to assert the NEW contract.

## D. FRESH-EYES "QUESTION EVERYTHING" (+ the contest-optimality lens)

1. **Is Lever-4's byte-savings a fake/no-op (Catalog #220)?** NO — it delivers −4.4% on the real uniform
   codec (MED-2 probe + regression test). Honestly scoped, not fake.
2. **Is Lever-4 CONTEST-OPTIMAL?** NO — it captures ~1/8 of the contest-positive saving its own sensitivity
   signal makes available (−0.0022 vs −0.0199 score). The variable-level export is the optimal byte-half.
3. **Is the −36% saving a real NET contest win or a byte-only mirage?** Per the MEASURED RD table, FAVORABLE
   on the 24-pair slice (many tensors save bytes at ≤0 d_seg cost) — but it MUST be re-measured at 600 pairs
   + parse-back + dual CPU/CUDA before a score claim (the deferred design decision).
4. **My probe's first run said the codec "delivers no savings" — was that a finding?** NO — instrument bug
   (I keyed the level map by MODULE names; the codec keys by STATE-DICT keys). Fixed; the codec delivers
   −36% with sd-key-correct keys (the R10-lesson discipline: separate instrument from substrate).
5. **Should I rewire Lever-4→D2 now?** NO — design decision touching Partner-B's surface + needs a paired
   contest-CUDA measurement first. R14 names the optimal path + guards the scope; the rewire is the next unit.

## Findings by severity

- **HIGH:** NONE.
- **MEDIUM-R14-1 (the counter-resetting finding — contest-optimality):** Lever-4 leaves ~0.018 contest score
  on the table — its online per-tensor sensitivity EMA is never routed to the variable-level export that
  would realize the full reverse-waterfill allocation (−36% decoder blob at ≤0 measured d_seg cost vs
  Lever-4's −4.4% via brotli compressibility). The contest-optimal implementation unifies Lever-4's online
  EMA with D2's allocator. NOT a bug (honestly scoped); a missed contest-optimization surfaced by the
  operator's "engineer the contest-optimal implementation" directive. Scope made explicit + guarded +
  optimal path named; the rewire deferred to a measured design decision.
- **LOW:** NONE.

## Test-run count

- `test_r14_lever4_export_is_uniform_127_not_variable_level_documented_scope` + sister
  `test_score_aware_grid_yields_smaller_real_brotli_blob_than_uniform`: **2 passed in 1.14s.**
- R14 probe (`probe_r14_lever4_export_byte_effect.py`): codec_mechanism_real=true (−29875 B on coarse map),
  driver_export_consumes_sensitivity_ema=false (documented uniform-127 scope), R14 byte measurements emitted.
- Full suite confirmed green earlier this session (batch A 77 + B1 3 + B2 11 + real-scorer R10/R12/R13 + the
  recovered waterfill default-byte-identity + waterfill-driver 4); R14 adds 1 test (+the probe).

## Wire-in / provenance

6-hook (Catalog #125): #3 bit-allocator ACTIVE (R14 quantifies the reverse-waterfill byte-allocation gap +
names the Lever-4-EMA→D2-allocator unification — the contest-optimal bit-allocator); #6 probe-disambiguator
ACTIVE (`probe_r14_lever4_export_byte_effect.py`); #1/#2/#4/#5 N/A (review-round + 1 guard + memo).
Mission contribution: `frontier_breaking` (names a measured ~0.018-score contest optimization + guards the
scope so the optimal rewire lands cleanly). Authority: byte measurements [contest-CPU advisory]
NON-PROMOTABLE (byte-effect + RD-slice evidence; the −36% NET-score win is a PREDICTION pending the 600-pair
paired CPU/CUDA measurement). No GPU launched, no daemon touched, no archive-build region rewired (the D2
surface is untouched — the rewire is deferred to a design decision). Frontier UNMOVED
`0.19109982419209975` contest-CPU.

**VERDICT: NOT-CLEAN (1 MEDIUM contest-optimality finding — Lever-4 leaves ~0.018 score on the table by not
routing its online sensitivity EMA to the variable-level export) → fresh counter RESETS 1/3 → 0/3.** Per the
operator's contest-optimality directive, this is a real finding, not a clean pass. The scope is made
explicit + guarded; the optimal Lever-4↔D2 unification is named + mathematically grounded (reverse-waterfill
KKT) and deferred to a measured design decision. R15 (and beyond) must begin a FRESH clean-pass count
(0/3 → …); the SEAL now requires THREE more consecutive clean rounds.
