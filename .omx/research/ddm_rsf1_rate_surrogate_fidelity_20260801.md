# ddm_rsf1 — is the in-loop RATE surrogate aimed at the coder we actually ship?

**Date:** 2026-08-01 · **Arm:** rsf1 (rate-surrogate fidelity) · **Cost:** $0 (scorer-free,
training-free, no dispatch) · **Axis:** `[macOS-CPU advisory]` for every surrogate column;
**byte columns are EXACT** (the r7 encoder is deterministic and lossless — not an advisory proxy).
**score_claim=false · promotable=false · pointer UNMOVED.**

**This EXTENDS `.omx/research/ddm_fh1_forces_harvest_20260731.md` §1 row A3** (which catalogues the
`--rate-model entropy|smevr_surrogate` option set and the derived-vs-live w_rate) **and closes the
measurement gap left open by `.omx/research/ddm_gd1_generic_default_census_20260731.md` row T4**
(which classifies the live `entropy` default **GENERIC-CHOSEN-UNRACED, LIVE NOW** and *asserts*
"the burn's rate gradient is steered by a coder-mismatched objective" without measuring it).
It is NOT an independent discovery. Neither fh1, gd1, vh1 nor ax1 contains a rank-correlation or
surrogate-fidelity number — grepped for `spearman|rank correlation|surrogate fidelity`, zero hits.

---

## §0 VERDICT (answer first)

**No — at the operating point the burn actually runs in, the live `entropy` surrogate is not merely
weaker than the alternative, it is ANTI-correlated with the shipped coder's bytes.**

| population | ρ(entropy, SMEVR bytes) | ρ(smevr_surrogate) | ρ(mode-base) |
|---|---|---|---|
| **B — r1c ep504→640 (THE burn-4 parent lineage)** | **−0.7235** [−0.943, −0.227] | **+0.7412** [+0.231, +0.973] | +0.7412 [+0.231, +0.973] |
| C — bc1 burn_out ep9→399 | −0.5382 [−0.988, +0.110] | **+1.0000** [+1.000, +1.000] | +1.0000 [+1.000, +1.000] |
| D — lv1 unmasked ep9→399 | +1.0000 | +1.0000 | +1.0000 |
| A — cross-config finals (n=22) | +0.3424 [−0.210, +0.741] | +0.3107 [−0.201, +0.767] | +0.3096 [−0.217, +0.771] |

On the **B (live-lineage)** trajectory the entropy CI **excludes zero on the negative side**: over
136 epochs the surrogate fell monotonically −1.80% while real shipped bytes rose +1.36%. On **C**
the same inversion is larger: ep109→ep399 entropy **−6.79%** while SMEVR bytes **+28.9%**
(194,236 → 250,358 B). Adjacent-step **sign agreement** with real bytes: entropy **33–40%**
(worse than a coin flip) vs **67–100%** for both delta-referenced surrogates.

**Scope on the ladder: FORMULATION-level, regime-scoped.** This falsifies `rate_model=entropy`
**in the saturated/rearrangement regime** (r1c ep504+, bc1 ep109+), which is exactly where burn-4
resumes. It does **NOT** kill the marginal-entropy FAMILY: in the *filling* regime (D, and bc1
ep9→109) entropy tracks bytes at ρ=+1.00. The mechanism below says why, and the regime boundary is
the finding, not a bare "no".

**Recommendation (NOT landed — MAIN owns the seal):** a `--rate-model` flip is a real, cheap,
now-evidence-backed lever, but it is a **config change to a sealed lineage** and it is *already*
pre-registered as gd1's "Burn-2 config corrections bundle" item 3 and fh1 A3's QA86a A/B. Fire it
as that pre-registered A/B, not as a silent swap. Details in §6.

---

## §1 What each thing actually is (re-derived from the primary artifacts, not from names)

| | factorization | what it can see |
|---|---|---|
| `--rate-model entropy` (LIVE) | marginal soft-histogram of kept-cell token VALUES (`token_rate_term`, `_soft_hist_entropy_bits`) | the symbol distribution ONLY — **provably invariant under any permutation of frames and of cells** |
| `--rate-model smevr_surrogate` (BUILT, never fired) | soft-histogram of **CONSECUTIVE-FRAME** deltas | first-order temporal difference structure |
| **SMEVR, the shipped coder** (`_encode_smevr`) | `factor_mode_delta`: per-cell **temporal MODE** base, `delta=(v−mode) mod L`; then an **occupancy** arithmetic stream (context = base value, previous-frame occupancy, left/upper spatial occupancy, age bucket) + a **value** stream for non-zero residuals | mode-referenced event sparsity, its temporal persistence, its spatial clustering, and residual magnitude |

Three different factorizations. The live surrogate is the only one that cannot see time at all.
`rate_model` has only ever been `entropy` in **29/29** landed TR1 configs on disk; `smevr_surrogate`
has **never fired** (confirmed: zero configs, zero telemetry) — the default-off-is-orphaned-signal case.

## §2 Method + provenance

Population: **70 token fields** from **22 distinct n600 TR1 run dirs**, all matched geometry
`(600,24,32,4)`, `levels=16`, `shared_base` → **raw byte totals are directly comparable** (no
size confound). Diversity spans variant (lotto/plain), init (solve_project/zero/none), cell mask
(QA24 keep-50 vs unmasked), margin-coupled quant on/off, delta-sparsity on/off, and 4 training
trajectories. Basis = `ema::` (the basis the trainer's own gate/ledger uses).

Fields are reconstructed **exactly as the trainer's byte-close does** (`_full_token_field_np` →
`quantize_tokens_np`) and both surrogates are computed with **the trainer's own**
`_soft_hist_entropy_bits` and the **same gather algebra as `token_rate_term`** — no reimplementation.

**Pipeline verified against the run's own logged authority:** reconstructing
`ddm_r1c_20260731/window_01/checkpoints/stage_seg_trunk_tau_final.npz` gives SMEVR = **269,504 B**,
which **exactly equals** that run's logged `tokens_bytes` / `tokens_bytes_smevr` at its ep640 gate
(`telemetry.jsonl`, `gate_params:"ema_shadow"`). Independently, pass-2 re-encoded every field and
asserted equality against pass-1's stored bytes — **70/70 agreed**.

```bash
.venv/bin/python experiments/ddm_rsf1_rate_surrogate_fidelity.py \
    --manifest .omx/research/ddm_rsf1_manifest_20260801.json \
    --out .omx/research/ddm_rsf1_rows_20260801.jsonl
.venv/bin/python experiments/ddm_rsf1_rate_surrogate_analyze.py \
    .omx/research/ddm_rsf1_rows_20260801.jsonl
```

Rows: `.omx/research/ddm_rsf1_rows_20260801.jsonl` (70) · full statistics printout:
`.omx/research/ddm_rsf1_analysis_20260801.txt`.

## §3 The live lineage, measured (r1c window_01 = the burn-4 parent)

| epoch | SMEVR B | occupancy B | value B | `entropy` | `smevr_surrogate` | mode-base |
|---:|---:|---:|---:|---:|---:|---:|
| 504 | 265,876 | 82,456 | 182,685 | 3.0786 | 2.2079 | 2.2441 |
| 549 | 268,876 | 81,636 | 186,514 | 3.0753 | 2.2622 | 2.2908 |
| 599 | **270,229** | 81,147 | 188,359 | 3.0528 | 2.3181 | 2.3475 |
| 640 | 269,504 | 81,554 | 187,240 | **3.0231** | 2.3602 | 2.3856 |

`entropy` falls **monotonically** across all 16 checkpoints while bytes rise then plateau. Byte
drift over the window is 4,353 B = **0.0029 S** (at 25/37,545,489 = 6.6586e-7 S/byte) — small in
absolute S, but the *direction* is what an in-loop gradient consumes, and it is inverted.

## §4 Mechanism (why — not just that)

**SMEVR's bytes live in the VALUE stream, and `entropy` cannot see it.**

| population | base | occupancy (event) | value (residual magnitude) |
|---|---:|---:|---:|
| A cross-config | 749 B (0.3%) | 116,198 B (41.6%) | **162,538 B (58.2%)** |
| B r1c (live) | 664 B (0.2%) | 81,568 B (30.3%) | **186,737 B (69.4%)** |
| C bc1 | 680 B (0.3%) | 92,317 B (43.7%) | **118,073 B (55.9%)** |

On the live lineage the **value stream is 69% of shipped bytes** and it is the part that *grows*
(+2.49% over the window) while occupancy *shrinks* (−1.09%). The value stream is priced by
mode-referenced residual magnitudes — a purely temporal quantity. A marginal histogram is invariant
under temporal permutation, so once the field stops *filling* (symbol distribution saturates) and
starts *rearranging* (residuals migrate in time), `entropy` goes blind and can even fall while the
coder's cost rises. That is precisely the observed regime split: entropy tracks in the filling
regime (D, bc1-early) and inverts in the rearrangement regime (r1c, bc1-late).

Note the hard mode-occupancy *oracle* is +0.97 on bc1 but **−0.69 on r1c** — occupancy alone is
also insufficient at the live operating point. Any replacement surrogate must price residual
MAGNITUDE, not just event count.

## §5 This bears directly on the derived w_rate (fh1 A3 / gd1 T19)

`spec_tr1_burn2.derive_w_rate_exchange_rate` derives w_rate from an explicit premise: *"reducing the
mean [surrogate] by 1 bit/token saves `n_counted/8` bytes."* I re-derived it and **the corpus value
is correct**: `n_counted = 601·384·4 = 923,136` (base coded once + 600 delta frames) →
w_rate = **0.0768348**. (My own first attempt, 0.0767, was wrong — I omitted the once-coded base
frame. Corrected.)

**The premise assumes a bits→bytes slope of n/8 = 115,392 B per bit/token. Measured:**

| population | `entropy` | `smevr_surrogate` | mode-base |
|---|---|---|---|
| C bc1 | +135,830 (R²=**0.12**) | **+120,239 (R²=0.84)** | +125,329 (R²=0.80) |
| B r1c (live) | **−45,228** (R²=0.42) | +23,947 (R²=0.73) | +25,189 (R²=0.72) |
| A cross-config | **−45,945** (R²=0.20) | +24,443 (R²=0.59) | +28,950 (R²=0.62) |

The assumed 115,392 B/bit is **well supported for a delta-referenced surrogate in the filling regime**
(bc1 consec-delta: 120,239, within 4% of assumed, R²=0.84) and **fails for the live `entropy`
surrogate — wrong sign, low R²**.

**Consequence for the 0.05-vs-0.0768 question:** the derived 0.0768 is derived against an exchange
rate that the *live* surrogate does not satisfy. **Fixing the surrogate is logically PRIOR to
re-deriving the weight** — raising w_rate 0.05→0.0768 while `rate_model=entropy` would scale a
gradient whose sign, at this operating point, points the wrong way. The two levers should not be
raced independently in that order. (Both remain UNRACED; this arm changes neither.)

## §6 Recommendation for the burn-4/burn-5 charter (evidence only — NOT landed)

1. **Do not silently flip `--rate-model` on the sealed burn-4 lineage.** The charter's argv-diff vs
   the r1c parent is asserted to be EXACTLY {class-weight-lane, telemetry-v9-port, epochs,
   max-wall-minutes, out-dir, resume-from}; `--rate-model` is not in that set and the ticket builder
   refuses any other flag. Verified: sealed `window_01_resmoke_ticket.json` carries the charter's
   cited `ticket_hash 098b5aea32feb048` / `sealed_sha256 a75ccb37e6381b8c`.
2. **Fire it as the already-pre-registered race**, gd1 "Burn-2 config corrections bundle" item 3 /
   fh1 A3 QA86a: `entropy` (control) vs `smevr_surrogate` at matched budget and matched w_rate, from
   the same resume point. Falsifier is already written there: no byte/d_seg win at matched budget →
   the item closes at INSTANCE.
3. **Consider a third arm.** gd1 T4's named derived candidate is "mode-base factored, matching
   SMEVR's event/value split" — which the BUILT `smevr_surrogate` is **not** (§7). A mode-base
   residual surrogate is buildable in-loop (stop-gradient on the mode base keeps it differentiable in
   the values; implemented and measured here as `surr_modebase_bits`). It ranks identically to
   consec-delta on B/C/D and slightly better cross-config (R² 0.62 vs 0.59), so on this evidence it
   is **not yet distinguishable** from the cheaper built option — do not build it on rank evidence
   alone; the built flag is the cheaper first race.
4. **Sequence w_rate AFTER `rate_model`** (§5).
5. **Byte-ledger note (gd1 T5 adjacent):** `token_stream_bytes_smevr` hardcodes `codec="smevr"`.
   On the live masked geometry SMEVR wins on **65/70** fields, but on **5/70** — all UNMASKED —
   brotli11 is smaller, once by 2.4× (225,388 → 94,368 B). The strictly-correct price is
   `encode_token_codes(codec="auto")` (min over the registered codecs). Not a live-lineage defect;
   recorded so a future unmasked-geometry arm does not inherit a wrong ledger.

## §7 Naming adjudication (`smevr_surrogate`)

**The flag NAME is defensible; one DSL provenance claim about it is FALSE; and its RUNG is FALSE.**

- **The name** — two readings. (a) "a surrogate FOR the smevr byte count" = a statement of intent;
  (b) "a surrogate that mimics smevr's mechanism" = a statement of mechanism. Under (b) it is a
  mismatch: SMEVR factors against the temporal **MODE**, the surrogate uses **consecutive-frame**
  deltas. Under (a) it is fine, and the measurement **vindicates (a) by outcome** — it ranks SMEVR
  bytes at ρ=+0.74/+1.00. **Verdict: keep the name; it earns it empirically.** The trainer's own
  docstring is already honest ("the zlib-on-delta coder surrogate `token_stream_bytes` runs") and
  should be kept, since it is the accurate mechanism statement. Recommend one clarifying clause:
  *"consecutive-frame delta (SMEVR itself factors against the per-cell temporal mode); named for the
  coder it targets, not the factorization it uses."*
- **FALSE mechanism claim (material).**
  `src/tac/witness_dsl/spec_tr1_renderer_20260728.py:298-299` states
  `"smevr_surrogate=temporal-delta (matches the shipped SMEVR event/value split)"`. It does **not**
  match that split: SMEVR's event/value split is defined by `factor_mode_delta`'s mode base, which
  this surrogate does not compute. The DSL is the SoT a future agent consults; this sentence would
  drive a config decision on a false premise. It also directly contradicts the trainer's own
  docstring — a triality drift between the DSL leg and the code.
- **FALSE provenance rung (material).** The same manifest declares
  `"--rate-model": {"rung": "RACED (QA86a)"}` while its own provenance string immediately says the
  race was *skipped* and is *planned* ("sg1 §3.4 skipped 'race-if-cheap' ... Burn-2 A/B"), and gd1 T4
  independently classifies it **GENERIC-CHOSEN-UNRACED, LIVE NOW**. A rung asserting a measurement
  that does not exist is a value-provenance-ladder violation. Correct rung:
  `GENERIC-CHOSEN-UNRACED (race QUEUED: QA86a)`.
- **Adjacent, same class:** `--w-rate`'s rung reads `DERIVED-ESTIMATE` but is attached to the live
  value **0.05**, which is not what the derivation produces (0.0768348). The manifest prose is honest
  about the gap; the rung label is not. Correct rung:
  `INHERITED-GENERIC (derived alternative 0.0768348 available, UNRACED)`.

**Seal-safety of the fix — MEASURED, not assumed.** `constant_manifest` is **not** part of
`sealed_ticket()`'s hashed payload (which is `{schema, trainer, argv, levers[name,overrides,notes],
score_claim}`), and the sealed burn-4 ticket JSON contains no `constant_manifest` at all. I
recomputed `window_01_ticket.json`'s `ticket_hash` from that payload alone and reproduced
`6206cf56ede3a14daa762430f1317361103da99aaa3ccbce7f30bcb533fbf8a5` exactly. **Correcting the two
rungs and the false mechanism sentence therefore cannot move any ticket or seal hash.** Not landed
here — `spec_tr1_renderer_20260728.py` is inside MAIN's live burn window and MAIN owns it; the
`notes` field (which IS hashed) must not be touched.

## §8 What is NOT clean (read before citing this)

1. **Cross-config, nothing discriminates.** In population A (n=22, the most nearly independent
   sample) every surrogate's 95% CI **straddles zero** and they are statistically indistinguishable.
   The verdict is carried entirely by the within-trajectory populations. Anyone wanting a
   *global* ranking claim does not have one here.
2. **Trajectory rows are serially correlated.** B/C/D are consecutive checkpoints of single runs, so
   the effective n is much smaller than the nominal 16 and the bootstrap CIs on those rows are
   **optimistic**. Treat B's [−0.943, −0.227] as indicative, not as a clean 95% interval. The
   sign-agreement and the monotone §3 table are the more robust statements.
3. **D discriminates nothing** (all surrogates ρ=+1.00) — it is a filling-regime trajectory. Reported
   because excluding it would have flattered the verdict.
4. **The r1c byte range is 1.6% (0.0029 S).** The inversion there is clean and monotone but small in
   absolute S. The large-amplitude evidence is bc1 (+190% bytes, 33% sign agreement).
5. **In-loop noise floor.** The gradient sees a batch-8 estimate, not the population value I measured.
   On the live r1c trajectory the between-checkpoint surrogate spread is only **3.6×** the batch-8
   sampling std (0.057 vs 0.016 bits) — the in-loop rate signal there is weak regardless of which
   surrogate is used. This bounds how much any `rate_model` flip can be expected to do at that
   operating point, and I have **not** measured what it would do.
6. **No ΔS prediction.** I measured ordering, not outcome. Nothing here says a `rate_model` flip
   lowers the exact score; only the pre-registered A/B can say that. Any ΔS number attached to this
   arm would be fabricated.
7. **Concurrency hazard, checked.** `experiments/ddm_r7_token_coder.py` was being rewritten by a
   sister agent DURING this measurement (`_encode_smevr` → `_encode_smevr_reference` + a new
   `_encode_smevr`, +528 lines, uncommitted). I re-encoded 8 randomly sampled fields under **both**
   the working-tree and the **HEAD (9ff1622962)** coder: **stored == worktree == HEAD on all 8** —
   the rewrite is byte-preserving and the byte column is version-invariant. Had it not been, every
   byte number here would have been void.
8. **Stale line citations in the corpus.** gd1 T4 cites `TR:452-469` / `TR:1038` and T5 cites
   `TR:481-493`; in the current trainer these are `_soft_hist_entropy_bits` at **652**,
   `token_rate_term` at **1892**, `token_stream_bytes` at **681**. Content matches; line numbers have
   drifted ~200–860 lines. Re-verify by symbol name, not line number.
9. **Not covered:** row-band (QA84) geometries are excluded by fail-closed (the offline
   reconstruction would not reproduce the tied field); no `independent` temporal-mode field exists on
   disk to test; `levels != 16` and `D != 16` untested. Verdict is scoped to
   `shared_base × D16 × L16 × c4 × n600`.

## §9 Owed legs (triality/quadrality debt, declared not silently skipped)

- **equations leg — OWED.** This is a measured finding and touches no
  `src/tac/canonical_equations/` row. The candidate law is the surrogate↔coder rank-fidelity
  regime split (filling vs rearrangement). Deliberately not registered from a single arm's
  advisory measurement; register when the QA86a A/B supplies the matched-budget confirmation.
- **DSL leg — OWED to MAIN.** The two provenance-rung corrections + the false mechanism sentence
  in `spec_tr1_renderer_20260728.py` (§7). Seal-safe (verified) but MAIN owns the file.
- **DAG leg — this receipt.** Consumer: burn-4/burn-5 charter §4 lever list, via fh1 A3 and gd1 T4.
