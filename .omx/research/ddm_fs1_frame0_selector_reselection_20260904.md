# ddm_fs1 — the frame-0 selector re-selection is byte-closed, measured and SEALED; it buys 16 more bytes than pr1 priced, and the pointer is UNMOVED until MAIN fires

Arm: `ddm_fs1_frame0_selector_reselection` (2026-09-04). Tokens: `[no-triality] [p0-ledger-ok]`.
Craft contract: `docs/operating_manual_craft_handoff.md`.
Naming note: an unrelated 2026-08-14 arm also used the id `ddm_fs1`
(`.omx/research/ddm_fs1_fire_seal_adapters_20260814.md`). Nothing here refers to it.
Axis of every pose row below: **`[macOS-CPU advisory]`, frozen CPU-torch PoseNet, DALI-lineage GT**.
Every byte row is EXACT and device-free. `score_claim=false`, `promotable=false`.

## ANSWER FIRST

1. **The candidate exists, it is byte-closed, and it is sealed.** `archive.zip` sha
   `50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf`, **180,022 B** (base 180,002 +
   **20 B**), sealed for `[contest-CUDA T4 n600]` at
   `/Volumes/VertigoDataTier/pact/ddm_fs1_frame0_selector/SEAL_fs1_frame0_selector_byte_optimal_contest_cuda.json`
   (`SEAL_VALID`, seal sha `21b2e351e401f03a…`). **MAIN fires; I did not.**

2. **The byte-optimal set beats pr1's pre-registered ratio gate by 7.0% for 16 FEWER bytes, and it
   lies INSIDE pr1's own robustness gate.** The blob length is a function of the ACTIVE COUNT alone and
   the per-pair gains are additive, so the exact net-`dS` frontier is a one-dimensional scan and not a
   heuristic. **MEASURED at n600 batch 8:**

   | adoption | changed | active `k` | blob | ΔB | Δ`S`_pose | Δ`S`_rate | **net Δ`S`** | projected `S` |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | pr1's ratio gate > 1.01 | 39 | 42 | 50 B | +36 | −1.271505e-04 | +2.397092e-05 | **−1.031796e-04** | 0.14787299166240067 |
   | **byte-optimal (SEALED)** | **21** | **24** | **34 B** | **+20** | −1.236782e-04 | +1.331718e-05 | **−1.103610e-04** | **0.1478658102574271** |

   B's edge is **7.181e-06 in net `dS`** and it is bought on the EXACT leg: a 1.065e-05 byte credit
   against a 3.472e-06 measured pose deficit, so the exact half is **3.07×** the measured half.

3. **Both candidates reproduce their advisory projections to better than 0.04%,** far inside the ±6%
   exchange-noise shorthand — and, more to the point, both are **ADMISSIBLE under the registered law's
   own rule** (`exchange_ratio_noise_floor_v1`: `Q_0.975(dS_b) < 0` over a seeded n600 pair bootstrap).
   B: point −1.1036e-04, 95% interval **[−1.8888e-04, −5.6632e-05]**.

4. **The batch-1 → batch-8 cross-shape step, which pr1 flagged as the open risk, cost essentially
   nothing.** Screened n600 mean gain 1.95873e-07 → measured 1.95824e-07 (**0.025%** apart). **Zero of
   the 21 adopted pairs got worse at batch 8**, and the per-pair ratios agree to three decimals
   (§4 table).

5. **`d_seg` cannot move, and this is now proved at the byte level, not argued.** The selector writes
   only `output[2 * frame_ids]` (`submissions/semantic_joint_ctxmix/runtime/f26_inflate.py:133`) — frame
   `2p` — while SegNet scores `x[:, -1, ...]` (`upstream/modules.py:100`) — frame `2p+1`. The build's
   no-op detector additionally proves the semantic section, token stream, HPAC model, residual table
   and CAP1 carrier are **byte-identical** through the shipped receiver's own parse, so the odd frames
   are bit-identical. Independently: **579 of 600 pairs measure `max |Δd_pose| = 0.0` — bit-identical
   at the score level too.**

6. **Two things that would have quietly poisoned this row, both caught and both structural.**
   (a) The live pointer row (`ddm_g8v1_gen8_tree_cuda_reproof_20260903`) was measured on the **public
   PR-tree** runtime, NOT on afr1's native tree; I had staged against the native tree first, which
   would have mixed a runtime change into the delta. (b) The public `inflate.py` **PINS the archive it
   will accept** (`:18-19`, enforced `:28-31`), so "only `archive.zip` changed" was never achievable —
   the receiver must be re-pinned, and the staged tree now proves the diff is **exactly those two
   constant lines** and nothing else.

7. **The encoder is the durable artifact.** The shipped runtime is decode-only; nothing in the repo
   could write a selector blob. `tac.semantic_pipeline.frame0_selector_codec` is the inverse, it
   verifies every output through the SHIPPED decoder before returning it, and its identity control
   rebuilds the archive's own 14-byte blob **exactly**.

**Pointer: UNMOVED.** No exact row was bought by this arm.

## 1. WHAT THIS ARM WAS HANDED

`ddm_pr1` §12.1 (`.omx/research/ddm_pr1_pose_resolve_on_renderer_change_20260904.md`, commit
`c7b537053`) swept all 8 frame-0 selector modes over all 600 pairs of the LIVE afr1 body: 39 pairs beat
their shipped mode by more than 1%, **pair 85's shipped op is actively harmful**, and the whole adoption
priced at **+36 B for −1.032e-04 S**. It stopped there, because the shipped runtime is DECODE-ONLY.

The charter's chain: encoder → byte-closed splice → batch-8 re-measure → seal → MAIN fires the T4 →
promote iff exact `S < 0.14797617125559104` on `[contest-CUDA T4 n600]`.

## 2. THE ENCODER, AND THE CONTROL THAT MAKES IT AN ENCODER

`submissions/semantic_joint_ctxmix/runtime/frame0_selector.py` carries `decode_selector` and
`apply_pixel_mode` and no inverse. The format, read off the decoder rather than recalled:

```
struct "<4sBH"    magic b"F0E1", version 1, active count k        7 B   (:13, :95)
rank              big-endian, ceil(bit_length(C(600,k) - 1)/8) B        (:98-99, :105)
labels            3 bits per active position, MSB-first, zero-padded    (:70-88, :100)
```

Three facts the encoder had to honour, each with its source line:

* the rank is the **colex** combinatorial rank of the sorted active positions —
  `_combination_unrank` (`:48-67`) subtracts `comb(positions[w-1], w)` for `w = k..1` and refuses a
  non-zero remainder, so the forward map is `rank = Σ_i C(p_i, i+1)`;
* each stored label is `mode_index − 1` and `_unpack_labels` refuses `label >= 7` (`:85-86`), so
  **IDENTITY is not representable as a label** — it is expressed by ABSENCE. An all-identity selector
  therefore has NO encoding (the header refuses `count == 0`, `:96`), and `encode_selector` says so
  and refuses rather than inventing one;
* the label padding bits must be zero (`:74-75`).

**`encode_selector` verifies itself through the SHIPPED decoder** and refuses to return bytes whose
parse-back differs. A round trip against a locally re-implemented decoder would only prove the module
agrees with itself.

**MEASURED controls** (`src/tac/tests/test_frame0_selector_codec.py`, 27 tests):

| control | result |
|---|---|
| **encoder identity**: `encode(decode(shipped_blob))` vs the 14 bytes the archive carries | **byte-identical** |
| closed length formula at `k = 5` | **14 B** — exactly what the archive carries |
| encoded length == formula at `k ∈ {1,2,3,5,17,24,42,52,128,300,599,600}` | 12/12 |
| 300-selector fuzz round trip through the shipped decoder | 300/300 |
| all-identity / out-of-range mode / wrong length / bad label / bad rank | all refuse |
| corrupted label packer (monkeypatched) | refused by the shipped-decoder check |

A format curiosity worth recording: `k = 600` encodes in **232 B**, one byte LESS than `k = 599`'s 234,
because `C(600,600) = 1` needs zero rank bytes.

## 3. THE SPLICE, AND WHY IT IS SMALLER THAN `ddm_up3`'s

`ddm_up3` had to run the whole CAP1 stack backwards because it changed the carrier CODES. This arm
changes only the selector, and **the selector tail is INVARIANT** under every layer between the brotli
stream and `decode_selector`:

* `rr5_arith_basis.split_carrier_body` / `assemble_carrier_body` (`:374-441`) carry
  `body_tail = body[packed_portion:]` through verbatim, and `restore_carrier_body` (`:499`) rebuilds it
  unchanged;
* `dx2_cabac_coefficients.restore_carrier_body` (`:319-330`) uses the same split and likewise never
  touches `body_tail`;
* `residual_archive._decode_rx1_models` (`:210-247`) derives `_packed_portion` from the body's own u24
  bit counts and lifts the selector as `SPARSE_SELECTOR_PREFIX + carrier_body[cap1_bytes:]`; the fixed
  part of `_cap1_body_bytes` is `6+36+96+32+12 = 182 = 102 + 80`, exactly the restored packed portion.

That is not assumed. `ShippedBody` **asserts it on the shipped body at construction** and refuses if the
stored tail and the fully-restored tail differ. **MEASURED on afr1**: stored tail 9 B
`05000dab3567db69e6`, receiver tail identical, decoded selector = 5 active pairs (60, 85, 116, 241, 373)
in modes (4, 3, 4, 7, 4).

The afr1 RX1 header is `codec=2 (brotli), reserved=0x1a`, i.e. CK2-semantic-plane2 + RR5 + DX2 active,
CK2-carrier-plane2 **not** set. The brotli parameters were **DISCOVERED by identity search** over the
shipped carrier stream (q=9, lgwin=16 reproduces its 22,010 bytes exactly) and then held.

### 3.1 The container identity control — the one the whole byte-close rests on

Rebuilding the shipped body through this module's own writer with the tail **UNCHANGED** must reproduce
`archive.zip` bit-for-bit. **MEASURED: it does** — 180,002 B, sha
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`. `run_build` REFUSES if it does
not, because without it the +20 B would be a mixture of the selector and the container and no delta
would be attributable.

### 3.2 The archive delta EQUALS the blob delta — measured, not assumed

The combinatorial rank is near-uniform, so brotli passes the tail through 1:1 and the receiver's closed
blob formula becomes an **exact archive price** rather than a lower bound. **MEASURED at k ∈ {12, 24,
42, 80}**: `archive_bytes − 180,002 == selector_blob_length(k) − 14` in every case. This is what let the
selection in §4 be priced before a single archive was written.

### 3.3 The container is held fixed on purpose, and it costs nothing here

`--container-search` measures the alternatives. **MEASURED**: q=9/lgwin=16 and q=9/lgwin=24 both give
180,022 B; q=10 and q=11 give 180,024 B — **2 bytes WORSE**. The shipped shape is already the minimum,
so there is no orthogonal container credit to separate out, and the one-variable comparison holds.

### 3.4 The no-op detector, through the shipped receiver's own parse

`read_residual_archive` on the candidate vs the base:

| section | identical? |
|---|---|
| semantic blob (sha + bytes) | **yes** |
| CAP1 carrier (sha + bytes) | **yes** |
| HPAC model (sha + bytes) | **yes** |
| RC64 token stream (sha + bytes) | **yes** |
| residual payload + table codes + scale | **yes** |
| compensation blob (`None` on both) | **yes** |
| **selector blob** | **DIFFERS** — 14 B `67d43d90…` → 34 B `c6864efa…` |
| per-pair selector choices | **21 differ, 579 identical** |

## 4. THE SELECTION, AND THE FRONTIER pr1 DID NOT SWEEP

pr1 compared exactly two rules: a >1% ratio gate and ungated. Neither is the byte optimum. Because the
blob length depends on the ACTIVE COUNT alone and the gains are additive, the best set of a given `k` is
the top-`k` admissible rows by gain, and sweeping `k` gives the **exact** net-`dS` frontier. Rows that
keep `k` fixed or LOWER it — pair 85, `3 → 0` — are always taken: gain at zero or negative byte cost.

**MEASURED** (39-point frontier, advisory pricing against pr1's batch-8 base `d_pose`
6.3656845167356244e-06):

| rule | changed | `k` | blob | ΔB | min adopted ratio | net Δ`S` |
|---|---:|---:|---:|---:|---:|---:|
| ratio gate > 1.01 (pr1's, reproduced exactly) | 39 | 42 | 50 B | +36 | 1.0128 | −1.032126e-04 |
| ratio gate > 1.00 (ungated, pr1's, reproduced) | 49 | 52 | 59 B | +45 | 1.0002 | −9.789950e-05 |
| **byte-optimal, gate > 1.01** | **21** | **24** | **34 B** | **+20** | **1.0169** | **−1.103919e-04** |
| byte-optimal, gate > 1.00 | 21 | 24 | 34 B | +20 | 1.0169 | −1.103919e-04 |

**The byte optimum lies INSIDE pr1's own robustness gate** — the two byte-optimal rows are the SAME set,
and its weakest adopted margin is 1.0169. That is not a tuning choice dressed as a result: the gate
exists to survive the batch-1 → batch-8 step, and the optimum never needed to reach below it.

### 4.1 Per-pair receipts, screen vs measurement

**MEASURED**, n600 batch 8, one instrument. Mode catalog: `0` identity, `1` luma+1, `2` luma−1,
`3` channel(1,0,−1), `4` roll(1,0), `5` roll(0,1), `6` tile(0,1), `7` tile(3,1).

| pair | mode | screen ratio (batch 1) | base `d_pose` | candidate `d_pose` | measured gain | measured ratio |
|---:|---|---:|---:|---:|---:|---:|
| 85 | **3 → 0** | 1.934× | 3.108235e-05 | 1.608705e-05 | 1.4995e-05 | **1.932×** |
| 555 | 0 → 1 | 4.254× | 1.635818e-05 | 3.837826e-06 | 1.2520e-05 | 4.262× |
| 488 | 0 → 2 | 1.472× | 3.862196e-05 | 2.623540e-05 | 1.2387e-05 | 1.472× |
| 259 | 0 → 4 | 1.843× | 2.217084e-05 | 1.203837e-05 | 1.0132e-05 | 1.842× |
| 514 | 0 → 7 | 2.096× | 1.787836e-05 | 8.527740e-06 | 9.3506e-06 | 2.096× |
| 372 | 0 → 3 | 4.307× | 1.045969e-05 | 2.432158e-06 | 8.0275e-06 | 4.301× |
| 585 | 0 → 6 | 6.126× | 9.367026e-06 | 1.528420e-06 | 7.8386e-06 | 6.129× |
| 479 | 0 → 1 | 1.982× | 1.388289e-05 | 7.010745e-06 | 6.8721e-06 | 1.980× |
| 77 | 0 → 5 | 4.300× | 7.822788e-06 | 1.819457e-06 | 6.0033e-06 | 4.300× |
| 504 | 0 → 3 | 2.609× | 8.384713e-06 | 3.211123e-06 | 5.1736e-06 | 2.611× |
| 436 | 0 → 2 | 3.467× | 6.983384e-06 | 2.020226e-06 | 4.9632e-06 | 3.457× |
| 5 | 0 → 6 | 3.704× | 6.088108e-06 | 1.640698e-06 | 4.4474e-06 | 3.711× |
| 71 | 0 → 4 | 1.167× | 2.975241e-05 | 2.549592e-05 | 4.2565e-06 | 1.167× |
| 70 | 0 → 3 | 1.017× | 1.652404e-04 | 1.624972e-04 | 2.7432e-06 | 1.017× |
| 547 | 0 → 1 | 2.077× | 4.498957e-06 | 2.160957e-06 | 2.3380e-06 | 2.082× |
| 161 | 0 → 5 | 1.123× | 1.405403e-05 | 1.252060e-05 | 1.5334e-06 | 1.122× |
| 95 | 0 → 3 | 1.104× | 8.941860e-06 | 8.089830e-06 | 8.5203e-07 | 1.105× |
| 173 | 0 → 3 | 1.510× | 2.492028e-06 | 1.652183e-06 | 8.3985e-07 | 1.508× |
| 221 | 0 → 6 | 1.077× | 1.099671e-05 | 1.020972e-05 | 7.8700e-07 | 1.077× |
| 518 | 0 → 2 | 1.493× | 2.240691e-06 | 1.500895e-06 | 7.3980e-07 | 1.493× |
| 586 | 0 → 6 | 3.054× | 1.030056e-06 | 3.364287e-07 | 6.9363e-07 | 3.062× |

**Zero pairs got worse. The screen and the measurement agree to three decimals on every row.** The
n600 mean gain: screened **1.95873e-07**, measured **1.95824e-07** — 0.025% apart. pr1's flagged
cross-shape risk was real to name and turned out to be small to pay.

**Pair 85 is the largest single gain in the set**, and it is the one pair where the archive was
*already spending bytes to make the score worse*: a selector op chosen for an earlier body and never
re-derived. Turning it OFF both gains 1.4995e-05 of `d_pose` and REMOVES a byte from the active count —
the same expiry class as `[[binding-instruction-numbers-expire-and-nobody-rederives-them]]`, in shipped
bytes.

## 5. THE MEASUREMENT, AND ITS CONTROLS

Three n600 batch-8 measures on ONE instrument (frozen CPU-torch PoseNet, DALI-lineage GT
`gt_cache_dali.pt`, renderer section `17e0fd0b…` from `ddm_g8s`), each reading its selector out of **its
own archive** rather than from memory — so what is measured is the archive → selector → `d_pose` chain
end to end.

| object | archive | archive sha | `d_pose` | pose leg |
|---|---:|---|---:|---:|
| base (shipped selector) | 180,002 B | `cbb8d928…` | **6.3656845167356244e-06** | 0.007978523996790148 |
| A, ratio gate | 180,038 B | `d507fb35…` | 6.164406554266268e-06 | 0.007851373481287378 |
| **B, byte-optimal** | **180,022 B** | `50fcaf1a…` | **6.169860284911831e-06** | 0.007854845819563762 |

**Control 1 — determinism across arms and sessions.** The base row reproduces `ddm_pr1`'s independently
measured `6.3656845167356244e-06` to **all 17 digits**. Same instrument, different arm, different
session, different day.

**Control 2 — the unchanged pairs are bit-identical.** `max |Δd_pose|` over the 579 untouched pairs is
**0.0**, exactly. The no-op detector holds at the score level, not just at the byte level.

**Control 3 — the float instrument applies the SHIPPED integer operator.** The pose instrument applies
the selector in float (`ddm_up2.apply_selector_float`) while the archive applies it in integer
(`frame0_selector.apply_pixel_mode`); up2's docstring asserts they agree "because every input is already
an exact integer". That is a docstring, so it is now MEASURED: for **all eight** catalog modes, on uint8
frames deliberately saturated at both 0 and 255 so the clamp branch is exercised, the two paths are
**exactly equal**. Had they differed, every gain above would be measuring a different operator than the
T4 will run.

**Control 4 — the base row is the body the build spliced, and the candidate row is the built archive.**
`run_compose` refuses on either mismatch, and refuses a base and candidate carrying the same selector.

### 5.1 `d_seg` is unchanged by construction, and it is proved twice

**STRUCTURAL, verified at source**: `_apply_frame0_selector` writes only `output[2 * frame_ids]`
(`f26_inflate.py:133`) — frame `2p`; `SegNet.preprocess_input` slices `x[:, -1, ...]`
(`upstream/modules.py:100`) — frame `2p+1`. **BYTE-LEVEL**: every section that feeds the odd frames is
byte-identical through the receiver's own parse (§3.4), so the odd frames are bit-identical. Δ`d_seg` is
therefore **exactly 0**, not "measured small". The seal's first falsifier makes any T4 `d_seg` movement
a withdrawal.

### 5.2 The closing arithmetic

Recomputed from components (`upstream/evaluate.py:90`), never from the 2-dp `Final score` display
(#877). The control: `100·0.00020139 + √(10·6.37e-06) + 25·180002/37,545,489` = **0.14797617125559104**,
the receipt exactly.

| | base | candidate B |
|---|---:|---:|
| archive bytes | 180,002 | 180,022 |
| rate leg (exact) | 0.11985594327989708 | 0.11986926045895953 |
| pose leg (advisory) | 0.007978523996790148 | 0.007854845819563762 |
| Δ`S`_rate | — | **+1.331718e-05** |
| Δ`S`_pose | — | **−1.236782e-04** |
| Δ`S`_seg | — | **0** (structural) |
| **net Δ`S`** | — | **−1.103610e-04** |
| projected `S` `[macOS-CPU advisory projection]` | 0.14797617125559104 | **0.1478658102574271** |

The LEVEL is the contest-CUDA T4 receipt; the DELTA is this arm's advisory same-instrument difference.
That composition is labelled and it is **not a score**.

## 6. TWO THINGS THAT WOULD HAVE POISONED THE ROW

### 6.1 The base runtime was nearly the wrong tree

I staged the first fire runtime from `ddm_afr1_tile48_receiver_identity/runtime_candidate_native` —
the tree that fired the afr1 row. But the LIVE pointer row is
`ddm_g8v1_gen8_tree_cuda_reproof_20260903`, and that arm exists *precisely because* the public
entrypoint had never executed on contest-CUDA: it re-proved the same archive bytes under the **PR-tree**
`inflate.py`/`inflate.sh` (`7c5a5e9a…` / `1300e6ee…`), which differ from afr1's native pair
(`a499942a…` / `971eaa12…`).

Firing the native tree would have made the T4 delta a mixture of a selector change and a runtime change
— exactly what "Source runtime must match the comparison" forbids. **MEASURED**: the g8v1 tree is the PR
tree with three files differing (`compress.py`, `MANIFEST.sha256`, `README.md`), none on the decode
path; the whole `runtime/` and `cpr1/` package is byte-identical. The staged tree is now g8v1's, and
`run_stage` proves it file by file.

The general shape: **the runtime that produced the pointer row is not always the runtime that produced
the archive.** Read the pointer's own receipt for the tree, never the archive's lineage.

### 6.2 "Only archive.zip changed" was never achievable

`inflate.py:18-19` declares `ARCHIVE_SHA256` and `ARCHIVE_BYTES` and refuses a mismatch at `:28-31`.
A candidate archive under the shipped receiver would have been refused **by its own tree**. The seal
tool caught it before creating anything:

> `SEAL PIN MISMATCH: ARCHIVE_SHA256 pins cbb8d928… but the staged archive is 50fcaf1a…`

The cure is the canonical `tac.candidate_seal.repin_receiver`, which rewrites those two constant lines
and restores the original bytes if the result is not `CONSISTENT`. Trusting it to have touched nothing
else would be a comment-only contract, so `run_stage` now **proves** it: same line count, exactly two
differing lines, both assigning a pin constant. **MEASURED diff**:

```
line 18  ARCHIVE_SHA256 = "cbb8d928…"  ->  ARCHIVE_SHA256 = "50fcaf1a…"
line 19  ARCHIVE_BYTES  = 180_002      ->  ARCHIVE_BYTES  = 180_022
```

So the honest statement is **not** "only archive.zip changed". It is: *the staged tree is the pointer
tree with `archive.zip` replaced and the receiver's two identity constants re-pinned to it, proved to be
those two lines and nothing else.* 41 files, 878,468 B.

## 7. THE EQUATIONS LEG

**Consumed — `tac.canonical_equations` `exchange_ratio_noise_floor_v1`** (ddm_xr1). Its acceptance rule
is the one this arm applied, in place of the charter's ±6% shorthand: resample the 600 PAIRS with
replacement (site-level resampling is forbidden — `ddm_fs3` measured AVERAGE ≠ MARGINAL by 2.24×) and
ADMIT iff `quantile_0.975(dS_b) < 0`. The same draw matrix drives both pose vectors, or the pairing
between base and candidate is destroyed. Seed 20260903, 200 resamples, via the law's own callables.

| candidate | point net Δ`S` | 95% interval | half-width | ADMISSIBLE |
|---|---:|---|---:|:--:|
| A, ratio gate | −1.031796e-04 | [−1.808755e-04, −4.974848e-05] | 6.556353e-05 | **yes** |
| **B, byte-optimal** | **−1.103610e-04** | **[−1.888824e-04, −5.663172e-05]** | 6.612536e-05 | **yes** |

**Appended — a 4th anchor to that law**: `fs1_frame0_selector_pure_pose_near_win_pair_bootstrap_20260904`
(append-only via `update_equation_with_empirical_anchor`, no builder file touched). It is the law's
first **pure-pose** case: ΔB is a whole-archive constant (a selector blob is not a per-pair byte stream,
so it enters the fixed calibration term `c` and carries no resample dispersion — consistent with the
law's own measured `σ_B = 0`), and Δ`d_seg` is exactly zero. The finding it records:

> even with ZERO byte dispersion and ZERO seg leg, the half-width is **0.599× the point estimate**. The
> pose mean is owned by a handful of pairs (`ddm_pr1` §5: ten pairs own 69% of it), and resampling 600
> pairs with replacement moves that tail. **A pose near win needs the bootstrap even when nothing about
> its bytes is uncertain.**

**Cited — `renderer_seg_pose_coupling_shipped_object_v1`.** This arm adds NO anchor to it and must not:
the coupling law governs a RENDERER change, where a seg move drags pose with it at `k ∈ [166.81,
217.30]`. The selector is a different actuator on a different frame with Δ`d_seg ≡ 0`, so its
denominator is identically zero and `k` is undefined. Reporting a coupling here would be exactly the
cross-regime constant transfer that law's own domain block forbids. `pr1`'s §12 framing stands: the
selector is the per-pair actuator `ddm_ft1` said a renderer change did not have — true of the render,
false of the frame the render is scored against.

## 8. HONEST LIMITS AND `verdict_scope`

* **verdict_scope: INSTANCE.** Everything here is about ONE archive (`cbb8d928…`) and ONE adopted
  selector set. No family claim, no transfer.
* **No score was measured.** Every pose number is `[macOS-CPU advisory]`. The projected
  `S = 0.1478658102574271` is a LEVEL from the T4 receipt plus a DELTA from a CPU instrument. It is a
  projection and it is labelled as one. Only `upstream/evaluate.py` on contest hardware, on these exact
  bytes, is a score.
* **The advisory→CUDA transfer is the live risk and it is unmeasured for THIS edit.** The instrument's
  calibration is strong (base `d_pose` 0.068% from the T4 receipt, base `d_seg` 0.033% per pr1), but a
  0.068% calibration on a LEVEL does not bound the error on a DELTA of 3.2% of that level. The seal
  pre-registers this: a T4 net Δ`S ≥ 0` refutes the transfer.
* **The 8-dp report quantisation is real and pre-registered.** The projected two-row delta bound is
  **7.316359e-06** (base row 3.632965e-06 + candidate row 3.683394e-06, DERIVED via
  `tac.report_8dp_bounds`, never typed). The advisory net Δ`S` is **15.08×** it, so the win should be
  resolvable — but a landed net Δ`S` inside `(−7.32e-06, 0)` is **UNRESOLVED, not a win**, and the seal
  says so. MAIN should recompose the exact sentence with `tools/report_8dp_delta_bound.py` once the
  candidate row lands.
* **A vs B is a near call at that resolution.** B's edge is 7.181e-06, roughly one two-row bound. B is
  sealed because its edge is bought on the EXACT leg (a 1.065e-05 byte credit) against a 3.472e-06
  measured pose deficit — the exact half is 3.07× the measured half — not because 7.181e-06 is
  comfortably outside the noise. A is retained, built, staged and unsealed.
* **The selector was optimised against a FIXED carrier.** pr1 §12.1 already noted this is a LOWER bound
  on the axis: re-solving each changed pair's carrier against its new frame 0 can only help, and neither
  arm did it.
* **I did not run a full inflate.** The archive is proved through the receiver's own strict
  `read_residual_archive` parse and the receiver's own `decode_selector`, not through a 30-minute CPU
  decode. The T4 fire is the first end-to-end execution.

## 9. NEXT_IF_RESUMED

0. **State.** All runs complete. Store `/Volumes/APDataStore/pact/ddm_fs1_frame0_selector/`, fire trees
   and seal on `/Volumes/VertigoDataTier/pact/ddm_fs1_frame0_selector/`.

1. **FIRE FIRST — MAIN fires the sealed candidate.** One governed T4 call:

   ```
   .venv/bin/python tools/fire_modal_auth_eval.py \
       --seal /Volumes/VertigoDataTier/pact/ddm_fs1_frame0_selector/SEAL_fs1_frame0_selector_byte_optimal_contest_cuda.json \
       --output-dir <dir> --lane-id <lane> --instance-job-id <job>
   ```

   **PROMOTE IFF exact `S < 0.14797617125559104` on `[contest-CUDA T4 n600]`.** Expect
   `d_seg = 0.00020139` unchanged, 180,022 B, `d_pose ≈ 6.17e-06`.

2. **Then re-solve the carrier on the 21 changed pairs.** Their frame 0 moved; their 12-dim carrier
   codes were fitted against the OLD frame 0. `ddm_jg5.refine_pair` through pr1's `solve` mode, then
   price the changed coordinates through `up2.price_full_resolve_bytes` (whose control reproduces the
   shipped 78,628 bits). This can only lower `d_pose`; it costs bytes, so it needs the same frontier
   scan §4 used.

3. **Sweep the selector on the pairs the carrier cannot reach, jointly.** pr1 §12.2 measured the
   selector on the RE-SOLVED candidate at 137 improving pairs. The general order is to alternate
   `selector` and `solve` until neither moves. The selector sweep costs about 2.3 s per pair.

4. **The frontier scan is reusable and cheap.** `adoption_from_sweep(..., strategy="byte_optimal")`
   applies to any sparse per-pair actuator whose blob length is a function of the active count. If
   another such actuator exists in the receiver, it can be priced before it is built.

5. **Do NOT adopt the ungated set.** MEASURED: 52 active pairs, +45 B, net −9.79e-05 — strictly worse
   than both gated sets. The last 10 pairs cost more rate than they buy pose.

## RECEIPTS

| artifact | what it is |
|---|---|
| `retained/fs1_select.json` | the 4 adoption variants + the 39-point byte frontier |
| `retained/candidate_{A,B}*/archive.zip` | the two candidate archives (+ blob, choices, shipped blob) |
| `retained/fs1_build_{A,B}*.json` | identity control, no-op detector, parse-back, container search |
| `retained/measure_{base,candA,candB}*_n600.json` (+ `_payload/`) | the three batch-8 n600 pose rows, per-pair `.npy` with sha256 |
| `retained/fs1_compose_{A,B}*.json` | the closing arithmetic + the pair-bootstrap admissibility |
| `retained/fs1_stage_{A,B}*.json` | the staged fire trees, the re-pin, the proved receiver diff |
| `…/VertigoDataTier/…/fire_runtime_B_byte_optimal_101/` | the tree MAIN fires (41 files, 878,468 B) |
| `…/VertigoDataTier/…/SEAL_fs1_frame0_selector_byte_optimal_contest_cuda.json` | `SEAL_VALID`, sha `21b2e351…` |
| `logs/measures_n600/` | launch manifest, run log, `safe_run` status (peak RSS 2.81 GiB) |

Code: `src/tac/semantic_pipeline/frame0_selector_codec.py` (the encoder),
`experiments/ddm_fs1_frame0_selector_reselection.py` (select / build / stage / measure / compose),
`src/tac/tests/test_frame0_selector_codec.py` + `src/tac/tests/test_ddm_fs1_frame0_selector_reselection.py`
(51 tests).

## Own-vehicle frontier

**afr1 S 0.14797617125559104 @ 180,002 B `[contest-CUDA T4 n600]` — UNMOVED by this arm.**
Sealed candidate, projected `S` **0.1478658102574271 @ 180,022 B**
`[macOS-CPU advisory projection]` — **not a score** until the T4 row lands.
