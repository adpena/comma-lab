# et4 pair-17 C2 failure — diagnosis + repair receipt (2026-08-06)

Owner: MAIN. Scorer-free diagnosis on the live et4 n600 chain (task #974 leg C / #613).
score_claim=false. Verdict scope: INSTRUMENT (the C2 custody gate's reference cache), not
the solve, not the decode, not the family.

## Symptom

Shard A of the et4 v2 driver died at `ddm_et4_solve_within_cvp_n600.py:756`:
`RuntimeError: C2 failed for pair 17: decoded parent argmax != cached parent`.
Deterministic (solo rerun of pairs 17-18 reproduced). Shards B/C kept passing C2/C3.

## Measured isolation chain

1. Live batch-1 forward vs cache, threads=4: pair 16 → 0/196,608 diff · **pair 17 →
   1/196,608** at (y=289, x=43), live=4 (MyCar) vs cache=0 (Road) — a hood-boundary
   near-tie pixel (canonical MyCar zone starts ~row 290) · pair 18 → 0.
2. Same at threads=6: pair 17 still 1 diff → **NOT thread count.**
3. Batch-16 forward (pairs 16-31 in one batch, threads=6, matching the et2 builder whose
   receipts are `parent_score/batch_0000_0016.json`…): **0 diffs on all 16 pairs incl. 17.**

Root cause: the et2 cache was built with batch-16 forwards; the et4 runner (and its solve
loop) forwards batch-1. oneDNN selects different conv kernels/blocking per batch shape →
different FP summation order → argmax flips at exact-tie logits. Deterministic-algorithms
mode + fixed threads do NOT remove this: **batch shape is part of the forward instrument.**
Observed landmine rate: 1 pair in 219 (~0.5%) ⇒ ~2-3 more expected in the untested 381
pairs if unrepaired.

## Repair (never weakens C2)

The runner uses its OWN batch-1 `lstar` as the solve reference (:749-751); the cache is
only the equality assertion. So the honest repair is to make the reference cache share the
solver's instrument: `experiments/ddm_et4_rebuild_parent_argmax_cache.py` (6319114cd3)
re-scores all 600 pairs batch-1 @ threads=4 via the runner's own `load_models`/`forward`
imports, preserves the original as `parent_tq1c_argmax_n600.batch16.npy`, atomically
installs the new cache (mmap-safe os.replace), and writes
`parent_score/rebuild_batch1_receipt.json` with per-pair diff sites + old/new sha256.
Already-checkpointed pairs passed C2 under BOTH instruments (equality held), so banked
checkpoints remain valid and resume is untouched. Driver final stage pinned to threads=4
(same instrument end-to-end). Sequencing: rebuild done-receipt → verify receipt →
pairs 17-18 solo green probe → kill driver+shards → relaunch v2 driver (resume skips all
banked pairs; shard A continues from 17 inside the driver).

## Blast radius note (durable law)

Counterexample to the "eval-mode BN ⇒ batch-size-invariant ⇒ chunked==monolithic
bit-identity" shorthand: BN is batch-invariant, conv reductions are not. Any byte-equality
gate between forwards must pin (code, weights, threads, batch shape). Tolerance-based
checks are unaffected (1 px / 42M scale). Memory:
`batch_shape_is_part_of_the_forward_instrument_20260806.md`.

## Live chain state at writing

Rebuild 150/600 (pid 46940). Shards B @ pair 318 / C @ pair 514, 242 checkpoints banked.
Aggregate η B=0.35383 · C=0.35933 vs bar 0.1710048742 (~2.1×), per-pair pose ratios ~1.00
(tube-respecting) — consistent with et3's n=32 η 0.3562364 and, if it holds through the
final composed byte-close + n600 evaluate, the projected net is ≈ −0.03 S vs the tq1c
baseline 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]. Realized verdict comes ONLY
from the final stage's evaluate row.
