# DDM RX2 — current-label MC36 HPAC

**Status:** Stage 0 is complete. The real reference-form IHS1 training job is
FIRED and running under the governed CPU launcher; it has not reached the
terminal epoch-60 checkpoint, so the Stage-2 real-stream verdict is
**UNMEASURED**. No scorer was loaded, no candidate archive was promoted, and
the frontier did not move.

## Stage-0 headline — NON-KILL-AUTHORITY

All rows use the exact full 117,964,800-token MC36 spatial field, SHA-256
`9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`.
The model column holds MC36's 70,835-B model charge fixed and the table column
uses a retained Brotli-q11 count table. These are empirical entropy
projections, not coded streams and not neural-family verdicts.

| weak context class | bits/token | ideal token bytes | table bytes | projected model+token+table | delta vs 186,073 B | accelerated GO |
|---|---:|---:|---:|---:|---:|---|
| order 0 | 1.6155178189 | 23,821,780 | 44 | 23,892,659 | +23,706,586 | no |
| causal left + upper | 0.0182700559 | 269,403 | 297 | 340,535 | +154,462 | no |
| causal left + upper + previous-frame same position | 0.0158269138 | 233,378 | 822 | 305,035 | +118,962 | no |

Verdict scope: `FORMULATION/WEAK-COUNT-CONTEXT`. None of the count models
projects a win. Per m94 and the charter, this result cannot kill the richer
trained IHS1 family, so Stage 1 ran rather than stopping on the table.

Retained receipt:
`/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/stage0/stage0_entropy_telemetry.json`.
Every raw count table, compressed table, and shape receipt is retained beside
it. The governed Stage-0 status is `status=ok`, `exit=0`, 8.786 s.

## Stage 1 — real object, active

The exact source was copied to the bulk SSD as a uint8 Torch cache:

- source: 117,964,800 B, SHA-256 `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`;
- cache: 117,967,085 B, SHA-256 `f53db4e8e65789d7d0442e97f8531bfb9765f41a2c37c8509c6ccdaeb8a6c888`;
- cache tensor raw SHA-256: the exact source SHA above;
- receipt: `/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/inputs/mc36_training_cache_receipt.json`.

The active job is the full HB1/HB2 reference form: 60 epochs, batch 8,
evaluation every 2 epochs, LR 0.003, exponent LR 0.0002, bit LR 0.01,
rate lambda 1, QAT fraction 0.5, channels 64, patch 64, delta 2, frame-dim 8,
weight and activation bounds 127, weight scales, exponent minimum -6, SPM,
raw targets, EMA deployment shadow, seed 20260716, CPU Torch. No budget or
architecture scope reduction was taken.

The launch passed the 16-GiB governed admission and writes bulk only to
`/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/` because the
mandated Vertigo tier had only 1.5 GiB free while APDataStore had 528 GiB.
Small receipts remain under the charter's Vertigo work root. Status receipt:
`/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/training/safe_run_status.json`.

The epoch-0 causal checkpoint is preserved at
`/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/training/checkpoints/mc36_hpac_best_ema.checkpoints/initial_stage_start.pt`,
537,707 B, SHA-256
`db29a20c629511b3ac9a429027d04340eb838bbd96046d82ef1ca65502eb84c4`.
Its stored causal-state hash recomputes exactly. The job preserves immutable
continuous-stage, QAT-stage, and periodic checkpoints plus `latest.pt`, live
weights, EMA shadow, optimizer, scheduler, RNG, config, and resume lineage.

The first immutable training checkpoint is now preserved at
`/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/training/checkpoints/mc36_hpac_best_ema.checkpoints/periodic/epoch_0001.pt`,
1,050,895 B, SHA-256
`c6a7fd6630f7aa7a783745b744cc034b10e8689bb2bfffd59d16c6059e01204b`.
Its stored causal-state hash
`651057591cf2622c87750b50032f265c02e47d7769d2fb5d45082a83d95c1952`
recomputes exactly. MEASURED `[macOS-CPU advisory, training surrogate]` at
epoch 1: 0.0079942244 bpp, top-1 error 0.0019700538, estimated token bytes
117,880, theoretical model bytes 27,026, estimated joint bytes 144,906, all
517 variable-depth elements still at 8 bits. The joint surrogate is 41,167 B
below MC36's 186,073-B model+token accounting, so the live trajectory is
encouraging. It is explicitly `ADVISORY_ESTIMATE_NOT_SERIALIZED`: it neither
admits a candidate nor substitutes for the terminal real IHS1+RC64 race.

At this seal the terminal epoch-60 checkpoint, trainer report, real IHS1
payload, fitted correction table, probability lattices, RC64 stream, and RX2
archives do not yet exist. Their bytes are `UNMEASURED`, not zero. The prior
HB1 CPU reference took 70,311 s; that is a recalled scheduling anchor, not a
projection of the eventual RX2 bytes.

## Stage 2 — sealed identity race, not yet fired

The committed runner `experiments/ddm_rx2_mc36_identity_race.py` is ready to
consume only a completed epoch-60 EMA checkpoint whose causal hash, trainer
report, and certified artifact manifest all verify. It then performs:

1. canonical IHS1 packing with exact deployed state-dict equality and logit
   equality, while retaining the raw payload before measuring it;
2. XZ plus Brotli q0-q11 lossless representations with parse-back;
3. full-n600 base float-logit and exact 25-state feature exports;
4. full-field checkpoint-specific boundary/predicted-class table fitting,
   followed by a seeded, non-prefix, equal-strata n120 selection screen;
5. neutral plus the four selected fitted tables through retained full-n600
   int16 probability lattices and checkpointable native RC64;
6. exact event-order and spatial-token equality, all complete archives and
   deterministic repeats, shipped-receiver component parse-back, and a full
   3,662,409,600-B lifted CPU raw-identity replay for the byte winner;
7. an SHA-256/byte inventory of every retained payload.

The runner deliberately reuses the proven native RC64 implementation rather
than re-racing coders on an unchanged stream under task #996. Here the stream
is allowed to change only because the trained probability object and fitted
table change its probabilities. The development n120 screen chooses which
tables receive real n600 coding; it is never a verdict surface.

## Candidate accounting

| object | model/container bytes | token bytes | archive bytes | token identity | full raw identity | disposition |
|---|---:|---:|---:|---|---|---|
| MC36 Variant C incumbent | 70,835 model; 196 non-model container overhead | 115,238 | 186,269 | exact | exact | current contest-CUDA frontier |
| frozen tq1c transfer, RX1 best | +15 vs MC36 | +5,462 vs MC36 | 191,746 | exact | exact | FOLDED at `INSTANCE/FROZEN-TRANSFER` |
| RX2 current-label terminal IHS1 | UNMEASURED | UNMEASURED | UNMEASURED | pending | pending | FIRED training; Stage 2 gated on epoch 60 |

No score is inferred from the training curve. A candidate is admitted only if
its complete retained archive is below 186,269 B and both token and full raw
identity pass. The score delta would then be a clearly labelled arithmetic
projection until MAIN runs `upstream/evaluate.py` on the exact bytes.

## RECALL EVIDENCE

Before design, the bounded recall searched `.omx/research/` memos and arm
receipts, arm-final messages, `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG
FEED blocks, `main_hot_state.md`, lane/task/probe stores, the canonical
equations registry, and the actual receiver/trainer sources. Content queries
included `MC36 HPAC IHS1 label fitted prior self-fit`, `tq1c deploy-bound
weight-bound`, `probability object exact token identity`, `RC64 native`,
`#996`, `xi screw conditioned context`, and `115238 f0ba4bb4`.

Beyond the charter seeds, the search found that XI1/XI2 own the separate
context-extension axis and that the live hot state already routes ξ-context
promotion there. That changed RX2 by keeping its treatment at the matched
HB1/HB2 spatial reference form rather than absorbing an unowned ξ mechanism.
The canonical registry and HB2 receipt reinforced that target-payload
training plus exact decode equality is mandatory before adopting HPAC bytes.
Task #996 and LP135 closed unchanged-stream coder reraces; that changed Stage
2 into reuse of native RC64 on only the newly changed probability streams.
No completed current-label MC36 IHS1 checkpoint or prior n600 current-label
real-stream race was found in the searched scopes.

## Borrowed substrate and verification

PR130 supplies IntegerHPAC, IHS1, and its model architecture. HB2 supplies the
deploy-bound `-128 -> -127` consistency repair and pack gate. RX1 supplies the
whole-container/native-RC64/receiver protocol. RX2's owned contribution is the
exact MC36-target training treatment, terminal checkpoint custody,
checkpoint-specific table fit, and whole-container comparison.

Implementation commits are `87d0709b96`, `a6014a67a8`, `7a0b791d64`, and
follow-up custody hardening through `3e35f65061`. Focused verification: Ruff
clean, `py_compile` clean, 23 focused tests across the Stage-0/trainer/race
surfaces, payload-retention detector clean, two post-edit review-tracker
passes per Python edit, and serializer-only commits with `[no-triality]
[p0-ledger-ok]` and no co-author trailer.

## Follow-on dispositions

- `ddm_rx2_reference_training`: **FIRED**. Owner: current RX2 process. Consumer
  store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/training/`.
  Fire trigger already met: exact cache identity plus governed storage and
  admission preflight.
- `ddm_rx2_terminal_identity_race`: **QUEUED-WITH-A-FIRE-ORDER**. Owner: RX2
  harvester. Consumer store:
  `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/FINAL_RESULT.json`.
  Fire trigger: safe-run `status=ok`, complete trainer result/manifest, and a
  causal-hash-valid epoch-60 QAT checkpoint.
- `ddm_rx2_exact_t4`: **QUEUED-WITH-A-FIRE-ORDER**. Owner: MAIN. Consumer
  store: `.omx/state/main_hot_state.md` plus the canonical frontier pointer.
  Fire trigger: the exact retained RX2 archive is strictly below 186,269 B and
  has shipped-receiver, n600 token, and full raw identity receipts.

## Frontier

RX2 has not produced a complete archive, so it has not moved the pointer.
Current exact own frontier remains MC36 Variant C
`S = 0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`, archive SHA-256
`f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`.

## NEXT_IF_RESUMED

- **FIRED** — owner: RX2 harvester; consumer store:
  `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/training/`; fire
  trigger: inspect the governed status and newest immutable checkpoint, then
  continue from `latest.pt` only if the original process ended before epoch
  60, preserving the same run identity and config.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: RX2 harvester; consumer store:
  `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/FINAL_RESULT.json`;
  fire trigger: completed epoch-60 trainer report/manifest; execute
  `prepare -> export-base -> fit -> materialize/encode/build for every selected
  variant -> cpu-decode -> finalize` through the governed launcher.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: canonical
  frontier pointer plus `.omx/state/main_hot_state.md`; fire trigger: a
  retained RX2 archive below 186,269 B with exact shipped-receiver, token, and
  full raw identity; run one exact contest T4 evaluation on those bytes.

## LIVE-HYPOTHESES

- Direct MC36-label IHS1 training can reverse the frozen transfer's 5,462-B
  token penalty. This is plausible because the same architecture reached a
  97,928-B real stream on its own tq1c labels; RX1 isolated cross-label
  mismatch rather than a receiver failure.
- A checkpoint-specific 96-B correction table can pay for itself after
  retraining. This is plausible because even the transplanted MC36 table saved
  337 B versus neutral on the frozen tq1c transfer, while the new table is fit
  to the exact deployed checkpoint.
- The neural prior can beat every Stage-0 count rung despite their large
  projections. This remains plausible because IHS1 has learned spatial and
  previous-frame features far richer than the three weak empirical context
  classes; the charter intentionally denies those rungs kill authority. The
  epoch-1 144,906-B joint surrogate is the first treatment-specific positive
  signal, while remaining non-serialized and non-verdict.

## DEAD-ENDS

- Order-0 as the RX2 representation is closed at
  `FORMULATION/WEAK-COUNT-CONTEXT`: its full-field projection is 23,892,659 B.
- Left+upper and left+upper+previous-frame count tables are closed as direct
  RX2 representations at the same scope: their best full-field projections
  are 340,535 B and 305,035 B, both above the 186,073-B model+token bar. This
  does not close the neural IHS1 family.
- Frozen tq1c IHS1 transfer is not being retried: RX1 measured 191,746 B,
  +5,477 B versus MC36, with exact token and raw identity. Its verdict scope is
  `INSTANCE/FROZEN-TRANSFER`, not current-label retraining.
- Unchanged-stream coder reraces are closed by task #996/LP135; RX2 reuses the
  settled native RC64 implementation and changes only its probability stream.

---

## REBASE NOTE (appended 2026-08-16 by `ddm_fb1`) — APPEND-ONLY, nothing above is changed

**The body above was CORRECT WHEN WRITTEN. This note exists so the bar is not consumed stale.**
Per Catalog #110/#113 HISTORICAL_PROVENANCE no line above is rewritten; this is a superseding row.

At the time of writing, the frontier was `S = 0.1619344578804448 @ 186,269 B` (MC36 Variant C).
**It has since moved twice:** `MC36 -> e480b v2 (183,502 B) -> hv1 ep0634`.

**LIVE BASE as of 2026-08-16:**
`S = 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]`,
sha `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`
(`.omx/state/canonical_frontier_pointer.json`, `effective_frontier`).

**WHY THIS MATTERS — the staleness runs in the dangerous direction.** The `186,269 B` bar sits
**3,510 B ABOVE what we already ship**. A candidate can PASS the bar written above while scoring
**+0.002337165 WORSE** than the incumbent — 233.7x the 1e-5 naming bar.

**USE THIS INSTEAD — a bar that does not go stale.** `seg + pose` is decode-identical across the
whole `cp135 -> MC36 -> e480b v2 -> hv1` lineage (measured to 1e-15), so only rate moves:

```
sub-0.15  <=>  archive <= 168,345.5977 B      (from the live 182,759 B: cut 14,413.4 B)
beat the incumbent  <=>  archive <  182,759 B  (at equal-or-better distortion)
```

Caveat that travels with the invariant: it is a PURE-RATE target, valid only while distortion is
held. Any candidate that CHANGES `d_seg` or `d_pose` must re-measure against the live pointer.

Full derivation, the repo-wide sweep with its denominator, and the bank-union verdict:
`.omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md`.
