# R6CAL — byte-close the solved seg object → real `upstream/evaluate.py` → the honest distortion calibration row

**Date:** 2026-07-27 · **Arm:** `r6cal_solved_object_byteclose_eval_20260727T230213Z`
**Evidence axis:** `[macOS-CPU advisory — real evaluator, real archive bytes]`
**`score_claim=false` · `promotion_eligible=false` · `rank_or_kill_eligible=false`.**
The canonical frontier pointer is UNMOVED by this arm and nothing here is a frontier claim.

---

## Verdict first

1. **Two premises in the arm brief are FALSIFIED by the artifacts.** The archive at
   `04_candidate/` is the **box-tolerance solve** (`d_seg = 1.16e-3`, 136,839 errors), **not** the
   1.52e-4 object. The 1.52e-4 object is the **q1 exact control** — a *measured scorer control*
   with **zero materialized records on disk** (only `q4`/`q8` chunks exist). It is not byte-closed,
   so it cannot be shipped or scored. Separately, the V10 production receiver has **no pose
   section at all** (`SECTION_ORDER = ("y_description","frame0_policy","quotient_residual")`, zero
   `pose` occurrences); composing the R1 `dxi` stream into it would be **counted-but-inert** — the
   #417 fake class — so it was **not** done. This base is *not* pose-less: it realizes
   `d_pose = 0.01663` implicitly, because PoseNet reads the frames the description writes.

2. **THE ROW, measured on the real object that actually exists.** Full 600-sample
   `upstream/evaluate.py --device cpu` on the exact archive bytes, after a real receiver-closed
   inflate (600/600 pairs, 707,788,800 plane values verified):
   **S = 194.42556** = `0.115997` seg + `0.407838` pose + `193.901723` rate.
   **99.731% of S is rate; 0.269% is all distortion combined.** The evaluator independently
   reproduces the solve's own `d_seg`/`d_pose` to 2.9e-8 / 6.1e-9 — the measurement chain is
   **validated end-to-end** (§3.1). The predicted shape was written down before the run and
   matched to 5 dp.

3. **The real deliverable — the crux is localized, sized, and the #603 blocker is answered.**
   ms2r_r3 removes **exactly 3,103,689 seg errors = 100.00%** of the debt #603 declared unowned
   ("*no measured receiver-closed component in the composed inventory owns that residual all-role
   debt*"). It owns all of it, at a **measured price of 93.8 bytes per removed error** against a
   **score-optimal price of 1.2731 B/error** — **74× too expensive**. The #603 register entry
   changes from *unowned* to *owned-and-priced*.

4. **Description compression cannot close the gap, and this is now measured, not argued.** The
   records are **already Brotli-Q11 internally** (`CONTENT_CODEC_ID = "brotli-q11.v1"`). Measured
   H₀ = 7.999 / H₁ = 7.986 bits per byte; every general-purpose coder **expands** the payload. The
   in-tree 4-coder race over all 50 streams agrees: RAW wins 50/50. **Coding is dead as a lever.**
   The gap is a **description** problem: `descriptor_len = 0` on all 1,200 records, mode uniformly
   `SPATIAL_SMOOTH_121` — the predictor is a fixed, parameter-free 1‑2‑1 blur carrying **zero
   video-derived parameters**, and 72.1% of the bytes are the resulting dense residual.

---

## 1. Custody (independently re-derived, not copied from the source records)

| artifact | bytes | sha256 |
|---|---:|---|
| `.../04_candidate/archive.zip` | 291,205,400 | `e3d0581ff4a3f475057e77e530374dad444b640a049b058cd66b37563534773e` |
| sole ZIP member `0.bin` (STORED, `compress_type=0`) | 291,205,292 | `daf1e1db6314e8cdbf63347afa35899e9891e3068428d42dc5a2fca235bb5295` |
| container overhead (archive − member) | 108 | — |
| inflated `0.raw` (1200 frames @ 874×1164×3) | 3,662,409,600 | `32a773a23a79c036ca39352b9ca9a048e20c089dc45beaa4c847689083641558` |
| inflate output tree | — | `bc7e37b021dbaa228ca30d8a3dbb5496eb2e98a8a054fb9018527314c5ced14b` |
| `upstream/videos` denominator (measured) | 37,545,489 | — |

Source root: `/Volumes/VertigoDataTier/pact/ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/stage_checkpoints/`.
Receiver contract `tac.witness_dsl.v10_production_receiver.v1`; schema `ddm_ms2r_r2_candidate.v1`.
Frozen scorer custody (from `02_scorers`): `modules.py` `065961ba…`, SegNet `68956e32…`, PoseNet `0f3a0874…`,
target cache `gt_n600.npz` `cf8d8360…`.

**Parse-back / byte-close proof.** The receiver re-parsed the packet from the exact archive bytes
and completed all 600 pairs, verifying **707,788,800** numerator values
(= 600 pairs × 2 frames × 512 × 384 × 3) against the declared per-record SHAs, and promoting a
full-length `0.raw`. The source records independently assert
`strict_production_parseback_exact = true` and `canonical_archive_determinism_x2 = true`; this arm
re-derived the archive/member SHAs and the record-walk byte accounting from scratch and both agree
(payload accounting closes to **0 unaccounted bytes**; parsed selected payload 291,203,920 B vs
declared `predictor_payload_bytes` 291,203,948 B — the +28 delta is exactly one `CHUNK_PREFIX`).

---

## 2. Premise correction, in numbers

`total scorer sites = 600 × 512 × 384 = 117,964,800` (verified against `stratum_sites` sum).

| object | seg errors | d_seg | d_pose | byte-closed? |
|---|---:|---:|---:|---|
| **q1 exact control** (the "1.52e-4 object") | 17,931 | 1.5200e-4 | 1.0184e-4 | **NO — zero records on disk** |
| **box-tolerance solve** (what `archive.zip` IS) | 136,839 | 1.1600e-3 | 1.6633e-2 | YES |

`136,839 / 17,931 = 7.63` — the "7.6× below box allowance" in the brief refers to the **q1 control
sitting 7.6× below the box**, not to the shipped archive. The shipped archive spends the **entire**
box (`realized_errors == allowed_errors == 136,839`, exactly).

`01_rate/` contains 100 chunk files → 1,200 records = `q4` and `q8` variants of 600 pairs.
**No `q1` file exists.** Materializing q1 is unbudgeted new work, not a repackaging.

**Receiver has no pose section — structurally, not incidentally.** `grep -ic` over
`v10_production_receiver.py` and `v10_predictor_residual.py` returns **0** for `pose`, and 0 for
`dxi|xi_|twist|screw`. `SECTION_ORDER` is a closed 3-tuple and the parser computes
`expected_ids = SECTION_ORDER[:section_count]`, so a **fourth section is rejected at parse time**.
Composing R1's `dxi` would therefore either fail parse-back or, if smuggled inside an existing
section, add counted bytes the receiver never reads. Refused under NO-FAKE #417
(counted-but-inert). Note also that the as-built `d_pose = 0.01663`
is **10.3× above** the #366 pose gate (`d_pose ≤ 0.00161`), while the unmaterialized q1 control's
`d_pose = 1.018e-4` would clear that gate by 15.8×.

---

## 3. THE ROW — real evaluator, real bytes

`[macOS-CPU advisory — real evaluator, real bytes]` · **not** `contest-CPU` (not Linux x86_64) ·
**not** a score claim.

Verbatim from `upstream/evaluate.py` (`report.txt`, 600 samples, `--device cpu`, batch 16, threads 4,
seed 1234):

```
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.01663316
  Average SegNet Distortion: 0.00115997
  Submission file size: 291,205,400 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 7.75606891
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 194.43
```

Recomputed from components (never from the printed rounded `Final score`):

| term | value | share of S |
|---|---:|---:|
| `100 · d_seg` | 0.1159970 | 0.060% |
| `√(10 · d_pose)` | 0.4078377 | 0.210% |
| `25 · rate` (rate = 291,205,400 / 37,545,489 = 7.75606891) | **193.9017228** | **99.731%** |
| **S** | **194.4255575** | 100% |

Inflate 382.3 s; evaluate 498.8 s.

**Prediction check.** The predicted shape `S ≈ 194.4256` was written into this memo *before* the
evaluator returned; measured **194.42556** — match to 5 dp. The arm brief's expected shape
(seg ≈ 0.015, pose ≈ 0.127, rate ≈ 194) assumed the q1 object and an R1 pose stream; neither is in
these bytes, so seg lands 7.6× and pose 3.2× above that expectation. **Rate matches the brief exactly.**

### 3.1 End-to-end validation of the measurement chain (the load-bearing result)

The real evaluator independently reproduces the solve's own frozen-scorer record on the exact bytes:

| term | solve record (`03_solve`) | real evaluator | \|Δ\| |
|---|---:|---:|---:|
| `d_seg` | 0.00115999858 (= 136,839 / 117,964,800) | 0.00115997 | 2.86e-8 |
| `d_pose` | 0.01663315391 | 0.01663316 | 6.09e-9 |

Both deltas are at or below the evaluator's 8-decimal print precision. **The ms2r_r3 scorer-measurement
chain — exact constructor → box solve → receiver-closed archive → inflate → frozen SegNet/PoseNet — is
validated end-to-end on real bytes.** Everything downstream in this memo that quotes solve-record
distortion values is therefore quoting values the real evaluator confirms.

### 3.2 Distance to bar

Bar = `min(0.15, official leaderboard best ≈ 0.172)` = **0.172**. `S − bar = +194.2536`, of which
**99.731% is rate** and **0.269% is all distortion combined**. This is a *distortion-solved,
rate-dead* calibration row.

This row is a **distortion-solved calibration row**, not a frontier candidate. Its distance to the
bar (`min(0.15, official leaderboard best ≈ 0.172)`) is ≈ **+194.25**, essentially all of it rate.

---

## 4. Description-compression floor — MEASURED

Artifact: `description_compression_floor.json` (schema `r6cal_description_compression_floor.v1`).

### 4.1 Where the bytes go (600 selected records, exact walk, 0 unaccounted)

| component | bytes | share | per pair | bits / plane value |
|---|---:|---:|---:|---:|
| bootstrap (frame-0, brotli-q11) | 81,085,747 | 27.85% | 135,143 | 1.8330 |
| **residual (int16 LE, brotli-q11)** | **210,029,373** | **72.12%** | **350,049** | **4.7478** |
| descriptor (video-derived predictor params) | **0** | **0.000%** | **0** | **0.0000** |
| record headers | 88,800 | 0.030% | 148 | — |
| **selected payload** | **291,203,920** | 100% | **485,340** | **6.5828** |

Per selected step: `q4` (208 pairs) 536,072 B/pair · `q8` (392 pairs) 458,421 B/pair.
Plane = 384×512×3 = 589,824 values. Predictor mode histogram: **`SPATIAL_SMOOTH_121` × 1,200** (only mode used).

### 4.2 Coder race — coding is exhausted

In-tree race (`05_coder_race`, 50 streams, all `parseback_exact=true`):

| coder | total bytes | Δ vs RAW | ratio |
|---|---:|---:|---:|
| RAW_COMPACT | 291,205,320 | +0 | 1.00000 |
| E4_BROTLI_Q11 | 291,208,619 | +3,299 | 1.00001 |
| ZSTD19_TRAINED_DICTIONARY | 291,523,801 | +318,481 | 1.00109 |
| CONSTRICTION_ORDER1_CONTEXT_ANS | 293,924,800 | +2,719,480 | 1.00934 |

**Winners: RAW_COMPACT 50/50.** Re-derived independently on real chunks in this arm:

| chunk | raw | brotli-q11 | lzma-9e | zlib-9 | bz2-9 |
|---|---:|---:|---:|---:|---:|
| `chunk-0000.q8` | 5,200,322 | +17 | +186 | +647 | +24,623 |
| `chunk-0002.q4` | 6,257,464 | +20 | +376 | +1,513 | +30,289 |

Measured entropy: **H₀ = 7.99912 / 7.99877** and **H₁ = 7.98586 / 7.98395 bits per byte**, 256/256
symbols present. **Mechanism:** `v10_predictor_residual` sets `CONTENT_CODEC_ID = "brotli-q11.v1"` —
bootstrap and residual are *already* brotli-q11 streams. Re-compressing them is compressing a
compressed file. **The 291,205,400 B IS the coded floor of this description.**

### 4.3 Residual structure — why it is expensive

Decoded pair 0 / pair 1: residual **89.08% / 86.24% nonzero**, `|r|` mean 5.75 / 5.43,
σ ≈ 10.0 / 9.4, max 229 / 219; `H(|r|) = 4.013 / 3.945` bits/value against a **coded 5.315 / 5.218**
bits/value. Bootstrap codes at 2.07 / 1.91 bits/px.

Two readings, both load-bearing:
- **Residual-internal coding slack ≈ 25%** (5.32 coded vs 4.01 entropy). Worth ≈ 73 MB — real, and
  **irrelevant**: it leaves ~218 MB against a 154,524 B box.
- **The predictor does no prediction.** `descriptor_len = 0` everywhere and the mode is a fixed
  1‑2‑1 blur. The residual is dense because the description is literally *"blur the previous plane,
  store the whole difference."* No motion compensation, no learned model, no video-derived
  predictor parameters exist in this family.

### 4.4 The box, priced

| box | bytes | factor required | rate term now → at box | B/pair at box | bits/plane-value at box |
|---|---:|---:|---|---:|---:|
| strict stress (#603) | 154,524 | **1,884.5×** | 193.9017 → 0.1029 | 257.5 | 0.00349 |
| hard planning (#603) | 200,000 | **1,456.0×** | 193.9017 → 0.1332 | 333.3 | 0.00452 |

**The description must go from 6.5828 bits/plane-value to ≈ 0.0035** — i.e. from a dense per-pixel
residual to a few hundred bytes per pair. That is not a coder target; it is the **~8-dim sufficient
statistic** regime the DAG already names as the indirect-RD crux.

---

## 5. Exchange rates — the crux in one number

`λ_B = 25 / 37,545,489 = 6.658590e-7` score/byte ·
`λ_E = 100 / 117,964,800 = 8.477105e-7` score/seg-error
→ **score-optimal exchange = 1.2731 bytes per corrected seg error.**

| exchange | B / corrected error | vs score-optimal |
|---|---:|---:|
| **score break-even** | **1.2731** | 1× |
| #603 waterfill requirement (28.1677 errors/B) | 0.03550 | 36× *cheaper* than break-even |
| ms2r_r3 measured `q8→q4` local edge | **405.50** | **319× worse** |
| ms2r_r3 vs #603 v14 control (whole-family) | **93.8** | **74× worse** |

**Consequence — the box solve's `q4` selections are score-negative by 30.32 points.** Reverting all
208 `q4` pairs to `q8` gives −45,681,677 B (rate −30.4176) for +112,656 errors (seg +0.0955), a net
**ΔS = −30.3221**. The solve did exactly what it declared ("*minimum real coded predictor-record
bytes subject to the Seg BOX*"), but at this rate operating point **the error box is 319× tighter
than the score-optimal trade** — so the box is the wrong constraint for S here. This is the
`objective_is_min_S_over_solution_set_not_box_or_point` law firing on live artifacts.
*Scope: exact over the finite `q4`/`q8` per-pair family only; the source record explicitly scopes
its dual as `MEASURED_ERRORS_NONADDITIVE_SHARED_RATE_EDGE`, `actionable_for_allocator = false`.*

---

## 6. #603 G-register — what this clears and what it blocks

The #603 box is **200,000 B hard planning / 154,524 B strict stress**, with a **MEASURED** v15 exact
control at 133,941 B. The two description families are **exactly complementary**:

| object | bytes | seg errors | rate in box? | seg in box? | pose in #366 gate? | S |
|---|---:|---:|---|---|---|---:|
| #603 v14 exact selected control | 133,247 | 3,240,528 | **YES** | NO (23.7× over) | NO (101,280× over) | 43.22 |
| #603 v15 archive | 133,941 | 3,240,528 | **YES** | NO (23.7× over) | NO (101,280× over) | 43.22 |
| **ms2r_r3 as-built** | 291,205,400 | 136,839 | NO (1,885× over) | **YES** | NO (10.3× over) | **194.43** |
| ms2r_r3 q1 control | *unmaterialized* | 17,931 | UNKNOWN | **YES** | **YES** | n/a |

**CLEARED — the "unowned debt" entry.** #603's verdict was: remove 3,103,689 errors from the v14
control, and *"no measured receiver-closed component in the composed inventory owns that residual
all-role debt."* ms2r_r3 removes **exactly 3,103,689 — 100.00%** of it, and is receiver-closed and
measured. It also beats v14 on pose by **9,803×** (0.01663 vs 163.061). **The distortion target is
demonstrably reachable.** That converts #603's ambiguous "does not arithmetically reach the box"
into a precise statement: **distortion is reached; the entire deficit is rate.**

**BLOCKED — every byte-home row.** At 93.8 B/removed-error the ms2r_r3 description cannot be routed
into any #603 reserve. The largest uncommitted reserve is the 25,789 B contextual/bounded-collateral
line, which the spec prices at **28.1677 removed errors/B**; ms2r_r3 delivers **0.01066** errors/B —
**2,642× short**. The 16,384 B v18b and J3 reserves and the 7,232 B contingency are likewise
unreachable by orders of magnitude. **No admission rule in the #603 waterfill can accept this
description at this price.** G4's separately-measured 89,161 B future-stream saving is irrelevant at
this scale.

**Also blocked: the `PREDICT` stage is empty in this family.** #603's `PREDICT` row (worldsheet
events, Lane seed, counted Pose6/xi chart) has **no counterpart** here — `descriptor_len = 0` means
ms2r_r3 carries *zero* prediction parameters. The 291 MB is what "no PREDICT stage" costs.

---

## 7. First rungs (each names its next measurement)

1. **Price a real PREDICT stage against the residual (highest value).** `descriptor_len = 0` is a
   measured, unexploited hole: the entire family runs on a parameter-free blur. *Next measurement:*
   materialize the two already-implemented modes (`AFFINE6_Q12`, 706 counted params in J2's lift;
   `PREVIOUS_PLANE_COPY`) on the same 600 pairs and record Δ residual bytes at fixed seg errors.
   This is the only lever in-family that touches the 72.1% cost centre. **Not run here — it needs a
   re-encode pass, not a repackage.**
2. **Re-solve the same family under the S objective instead of the error box.** The 30.32-point
   `q4→q8` result is exact and available today. *Next measurement:* re-run the DP with the
   1.2731 B/error dual as the stopping rule and record the S-optimal point of this family. Expect it
   to sit at all-`q8` or coarser — which means **the family's own S-optimum is a coarser, cheaper
   description than the one that was shipped.**
3. **Decide q1's status explicitly.** The 1.52e-4 / `d_pose` 1.018e-4 object clears both the seg box
   and the #366 pose gate but **does not exist as bytes**. *Next measurement:* materialize one q1
   record and read its `bootstrap`/`residual` split — a single pair, ~minutes — to price the q1
   family before anyone plans on 1.52e-4 again. Until then **1.52e-4 must not be cited as a
   byte-closed result.**
4. **Retire "compress the description" as a rate path for this family.** Measured dead: coders
   expand it, H₁ = 7.986 b/B, 50/50 RAW wins. Any future rate proposal here must change the
   *description*, and should be required to state its bits/plane-value target against the
   **0.0035** box figure.

---

## 8. What this arm did NOT do, and why

- **Did not compose R1's `dxi` into the archive.** The V10 receiver has no pose section; the bytes
  would be counted-but-inert (NO-FAKE #417). R1's `dxi` belongs to a different receiver lineage.
- **Did not present this as a frontier row.** It is 194.25 above the bar, essentially all rate. It
  is a *calibration* row: it fixes where distortion actually is once, on real bytes.
- **Did not fabricate coder gains.** Every compression number is from an executed coder on the real
  artifact; where a coder was unavailable (`zstandard` not installed locally) it is recorded as
  unavailable rather than estimated — the in-tree race supplies the zstd-19 row.
- **Did not touch the main working tree, PR110-lineage surfaces, or the RG4 pricing base.**

### Known limitations of this arm's own tooling (stated, not hidden)

- **No sibling unit tests** exist for `tools/r6cal_byteclose_and_eval.py` or
  `tools/r6cal_description_compression_floor.py`. Both were instead *executed against the real
  artifacts*, and the floor tool's accounting is self-checking (it raises on any unaccounted
  trailing byte and on non-uniform plane geometry). The eval driver's output was cross-checked
  against a hand-computed prediction made before the run.
- **`r6cal_byteclose_and_eval.py` leaves a trailing empty `PYTHONPATH` entry** (from
  `env.get("PYTHONPATH","")`), which Python reads as "add CWD to `sys.path`". Here CWD is
  `upstream/`, which is already on the path, so it is inert and did not affect the row. **Not
  patched**, deliberately: the file was executing and had already produced the inflate receipt, so
  editing it would have made the committed source differ from the source that produced the
  artifact. Fix belongs in a follow-up.
- **The floor tool's step classifier was fixed mid-review** (`else`-catches-everything → fail-closed
  `qN` parse) and the artifact **regenerated from the fixed code**; outputs were byte-for-byte
  identical, confirming the latent bug never fired on this data.
- Distortion components are read back from the evaluator's report at **8-decimal print precision**;
  the rate term is exact (integer bytes). At these magnitudes the readback error is ≤ 3e-8, i.e.
  ~1e-8 on S.
- `zstandard` is not installed in the local venv, so this arm's independent coder race has no
  zstd row; the in-tree `05_coder_race` supplies the zstd-19 measurement (it also loses to RAW).

## STORES CONSULTED

`CLAUDE.md` (NO-FAKE supreme rule; measured-scored-quantity axis; THE GOAL bar re-anchor;
SSD-first disk hygiene; `/tmp`-free evidence; pointer-only frontier scores) ·
`MEMORY.md` current-state block, incl. `dont_compose_on_weak_pricing_base_byteclose_the_solved_objects_20260727`,
`goal_is_sub015_or_below_official_leaderboard_best_pointer_fixation_abandoned_20260727`,
`objective_is_min_S_over_solution_set_not_box_or_point_20260724`,
`past_solves_rate_naive_free_null_counted_partition_20260721`,
`receiver_consumption_bijection_counted_but_inert_weight_groups_20260710`,
`borrowed_incumbent_rate_polish_permanently_dead_20260725`,
`decompose_every_headline_number_disaggregation_is_the_signal_20260720`,
`v10_description_pivot_budget_box_and_realization_crux_20260719` ·
`.omx/research/ddm_c1_composed_candidate_spec_603_613_20260723.md` (box, waterfill, G-rows, #366 gate) ·
`ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/stage_checkpoints/{01_rate,02_scorers,03_solve,04_candidate,05_coder_race}` ·
`src/tac/witness_dsl/v10_production_receiver.py` · `src/tac/codec/v10_predictor_residual.py` ·
`upstream/evaluate.py`.
