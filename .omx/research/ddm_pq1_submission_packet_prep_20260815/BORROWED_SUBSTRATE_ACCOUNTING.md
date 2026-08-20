# Borrowed-substrate accounting — packet generation 5 (jg5 joint-waterfill candidate, archive `f3bce5d259a08183…` / 180,625 B)

> **⚠ READ §9 FIRST.** This document is APPEND-ONLY and its section numbers count ITS OWN
> revisions, one ahead of the packet's `GENERATION_LOG.md`, which counts CANDIDATES. So §9 is
> the live section and it describes packet generation 5. Every figure in §§1–8 stands for the
> candidate it was written against; none of them describes these bytes except where §9 says it
> does. The prior title and banner are preserved immediately below, unedited.
>
> **What §9 changes.** Generation 5 keeps generation 4's re-classification of rows 1 and 2
> (the semantic renderer state and the pose carrier state are no longer byte-identical to
> PR #135 after decode) and adds a new `ours-original` mechanism row for the joint admission
> waterfill. It does NOT restore any byte-identity claim.

---

## Preserved generation-4 title and banner (verbatim, append-only)

# Borrowed-substrate accounting — packet generation 4 (ck1 composed row-prune candidate, archive `35c318d541d70370…` / 177,182 B)

> **⚠ READ §8 FIRST.** Generation 4 breaks the single most load-bearing claim in
> §§1–7: rows 1 and 2 of the §2 table classified the semantic renderer state and the pose
> carrier state as `inherited-substrate` **proven byte-identical to PR #135 after decode**.
> **That byte-identity is GONE at this generation.** The semantic values are now a lossy
> re-representation of theirs, and 6,713 of 7,200 carrier coordinates were re-solved. §8
> carries the re-classification and the attribution question it opens. Every figure in
> §§1–7 stands for the candidate it was written against; none of them describes these bytes
> except where §8 says it does.
>
> **Numbering, stated in the title's own neighbourhood (round-11 F5, 2026-08-18).** This
> document's section headings count ITS OWN revisions; the packet's `GENERATION_LOG.md` counts
> CANDIDATES. The two counters are one apart, so §7 is headed "Generation 4 amendment" while the
> packet candidate it describes is generation 3. The title previously read "generation 3 (rr4
> re-encode candidate)" — naming rr4, the generation-2 candidate at 181,161 B — long after §7
> promoted the document to the sz1 composed candidate at 179,930 B. A maintainer who read only
> the title concluded the wrong candidate was being submitted. Corrections must land in
> HEADLINES, not only in bodies; the body was already correct and is unchanged.
>
> **§§1–6 were written against the rr4 candidate and are preserved verbatim, append-only. §7
> carries every delta.** Read §7 before citing any figure from §§1–6.

This accounting is mechanism-level and deliberately unflattering to us. The archive is a new
exact-byte composition, but it is **not** a wholly original learned vehicle, and this table
exists so that nobody has to infer which part is which.

Generation 3 supersedes generation 2. It changes four things and none of them in our favour on
net: it separates "byte-identical to the base candidate we inherited" from "byte-identical to
PR130/PR135" (generation 2 conflated them), it withdraws an unverified originality label on the
residual payload, it names the two mechanisms of ours that generation 2 omitted, and it discloses
a concurrently-published pull request that describes the same class of mechanism as our one
claimed contribution.

## Categories (closed set)

| Class | Meaning |
|---|---|
| `contest-frame` | Supplied by the challenge itself. Not anyone's contribution. |
| `inherited-substrate` | Someone else's work, taken and used. Attribution is mandatory; no originality is claimed. |
| `mechanism-adopt-with-attribution` | Their idea or their source, our implementation or our re-fit. The idea is credited to them; only the implementation is ours. |
| `ours-original` | Designed and built in this repository. Claimable — but only with a receipt in this table. |

A row with no receipt does not get `ours-original`. That rule is why the residual-payload row
moved this generation.

## The one-line honest summary

The learned *vehicle* — the semantic renderer and the pose carrier — is PR130/PR135's, carried
into our archive byte-identically after decode. We do not claim it. What is ours inside these
bytes is narrower and it is three things: the HPAC probability object (PR130's architecture,
retrained in this repository on our own label field), a small set of admitted compensation edits
carried in the base, and the zero-byte decode-time probability corrector that produced this
generation's entire 1,598-byte saving.

---

## 1. Candidate-agnostic core

### 1.1 Ancestry chain

Every link is an archive we hold or measured, in order. Byte counts are the archive.

| Step | What | Archive bytes | Receipt |
|---|---|---|---|
| PR #130 | `semantic-pose-HPAC_CPR1`, Fesal Fayed (`fesalfayed`) — the base vehicle | 191,052 | `.omx/research/pr86_pr130_fullstack_intake_20260728.md:45-52`; acquisition `.omx/research/pr130_eureka_intake_acquisition_20260806.md:7-12` |
| PR #135 | `semantic-pose-HPAC_CPR1_polished`, Shreyan Mohanty (`codexblack`) — PR130's vehicle plus PR #133's constrained-basis and re-solved int12 carrier | 186,724 | `.omx/research/ddm_pi135_pr135_intake_20260810.md:11, 23, 29, 41` |
| cp135 | our lossless recompose of PR135's sections | 186,252 | `.omx/research/ddm_cp135_rate_compose_20260810.md` |
| mc36 Variant C | our admitted micro-edit union (qs2 ∪ re1), promoted on T4 | 186,269 | `.omx/research/ddm_mc36_promotion_complete_s_verdict_20260814.md:3-6` |
| e480b → hv1 | our HPAC retrain endpoint; checkpoint ep0634 selected from 81 retained candidates | 182,759 | `.omx/research/ddm_hv1_harvest_compose_ep508_20260815.md:9-16, 92-97, 217` |
| rr2 → **rr4** | **this submission**: the free decode-time corrector re-encodes the token stream | **181,161** | `.omx/research/ddm_rr4_cuda_prob_reencode_20260817.md` |

**Note on PR #133.** We never took PR133's code. It is in our ancestry anyway, transitively:
PR135 — the archive we actually built on — already incorporates PR133's constrained basis and
re-solved int12 carrier (`ddm_pi135_pr135_intake_20260810.md:11`). PR #133 is
`cpr1_cbq_matched8`, by `JasonMo123`. It is cited here because a reader tracing our substrate
reaches it whether or not we mention it, and we would rather mention it.

### 1.2 Contest frame — `contest-frame`

The upstream challenge repository, `evaluate.py` (SHA-256 `7da71a84ce24286b…`), the frozen SegNet
and PoseNet scorer weights, the 600-sample public test list, the 37,545,489-byte denominator, and
the 1,800-second inflate budget. Ours to satisfy, never ours to claim.

### 1.3 Runtime dependencies

| Dependency | Class | Note |
|---|---|---|
| PyTorch | `inherited-substrate` | third-party; provided by the evaluation image |
| Brotli 1.2.0 | `inherited-substrate` | pinned wheel, self-installed fail-closed (`exit 69`) |
| NumPy | `inherited-substrate` | third-party; provided by the evaluation image |
| a working C compiler (`cc`) | `contest-frame` assumption | `inflate.sh:32` compiles the range-coder backend on every run. Satisfied on the evaluated T4 image. Declared here because it is a hard runtime requirement and, unlike Brotli, it is currently unguarded — see the README dependency note. |

---

## 2. The rr4 candidate, section by section

Eight parsed sections. The `Class` column is the honest classification; the last column is what
the pull-request body says about the same section, so the two documents can be checked against
each other rather than trusted separately.

**Read "byte-identical to base" precisely.** It means identical to the archive we inherited at the
previous step (hv1 ep0634, `80d9c8c6…`). It does **not** by itself mean identical to PR130's or
PR135's bytes, because the base already contains our retrained HPAC object and our compensation
edits. Generation 2 used one label for both and that was wrong.

| # | Section or mechanism | Class | Receipt | What the PR body says |
|---|---|---|---|---|
| 1 | Semantic renderer state | `inherited-substrate` (PR135, proven byte-identical) | decoded 36,051 B, `b489c73567046e64…`; cp135 restores "the exact PR135 semantic blob" at this size and sha (`ddm_cp135_rate_compose_20260810.md:41`) | `PR130/135-byte-identical` — agrees |
| 2 | Pose carrier state | `inherited-substrate` (PR135, proven byte-identical) | decoded 22,242 B, `196f0e5136f4d6bf…`; same cp135 receipt | `PR130/135-byte-identical` — agrees |
| 3 | Compressed model container | `inherited-substrate` (unchanged from base; PR-level sha equality **not** independently verified) | 70,453 B, `e35d12371fa79747…`; base-identity from `ARCHIVE_MANIFEST.json:21` | `PR130/135-byte-identical` — **overstates**: we hold no receipt equating this container to PR130's or PR135's own |
| 4 | **HPAC probability object** | **`mechanism-adopt-with-attribution`** — PR130's architecture, **retrained in this repository on our own label field** | 17,952 B, `e8c0cfd73d3275ad…`. Checkpoint ep0634 from the e960 long burn, selected from 81 retained candidates by `tools/select_hpac_checkpoint.py` (`ddm_hv1_harvest_compose_ep508_20260815.md:9-16, 217`); trained on the MC36 label field, sha `9ba2e52b3096…` (`ddm_rx2_mc36_label_hpac_20260814.md:12-14`). hv1 records it as "the only change" in that generation (`:92-97`) | `PR130-lineage, inherited unchanged here` — **understates**, and drops "retrained on our labels" entirely. Body corrected this generation. |
| 5 | Compensation blob | `mechanism-adopt-with-attribution` — container inherited, **contents include our admitted edits** | 36 B, `38792b4953318117…`, byte-identical to base. The base carries mc36 Variant C, whose runtime parse-back recovers compensation pairs `[7, 96, 105, 176, 178, 517, 523]` = (qs2's six minus the measured-harmful 532) ∪ (re1's 96 and 7) (`.omx/research/falsified_premise_registry.jsonl`, premise `qs2_re1_bank_union_is_held_and_unfired_20260817`) | `PR130/135-byte-identical` — **overstates the borrow**; these compensation events are ours |
| 6 | Residual payload + table codes | `inherited-substrate` — **provenance UNRESOLVED; no originality claimed** | residual 100 B `74775aab04c7615c…` (RCF1 framing, `ddm_rr2_encoder_byteclose_20260817.md:124`); table codes `76afdc3ceda1212a…`. cp135 records PR135's own 100-byte residual as `bd27a2dd…` (`:58`) and hv1 carries `bd27a2dd…` unchanged (`:95`). Whether `74775aab…` is a re-framing of the same PR135 content or a genuinely re-fitted payload is **not settled by any receipt we hold**. | `ours-original, inherited unchanged here` — **WITHDRAWN this generation.** An originality label with no receipt is not admissible; see §6 item 1. |
| 7 | **RC64 token stream — the only section this generation changes** | **`ours-original` mechanism over inherited probabilities** | 110,512 B, `6c3757bd52a18d3c…` (base 112,110 B, `73a878891a31c366…`). The free decode-time corrector `ddm_rr4_free_corrector_v2`, sha `96fd35aaf82c737a…`, adjusts the probability model from already-decoded symbols, stores zero archive bytes, and re-encodes losslessly. **The probabilities it corrects are the HPAC object's.** Our estimator over that model. | agrees |
| 8a | RC64 range-coder backend, encoder side | `inherited-substrate` (PR135, verbatim) | compiles PR135's `rc64_backend.c`, `5c75e2c70b89f148…`, unmodified (`.omx/research/ddm_rc64p_native_cpu_decode_20260810.md:80-81`) | `PR135-byte-identical` — agrees |
| 8b | RC64 range-coder backend, shipped receiver | `mechanism-adopt-with-attribution` (PR135-derived, modified) | shipped `runtime/entropy/rc64_backend.c`, `05839d1416e68a49…`, which **differs** from the PR135 source above | `PR135-lineage-modified` — agrees. Stated explicitly so the difference is mistaken for neither full originality nor a clean copy. |
| 9 | Receiver binding, archive assembly, custody chain | `ours-original` | runtime tree `7acedb07e670e76c…`; archive `35ac2b9beb7e6fa8…`, 181,161 B; deterministic repeat recorded | `ours-original` — agrees |
| 10 | End-to-end compression entry point | `ours-original` | `experiments/ddm_pq2_compress_e2e.py`; rebuild verified 2026-08-17; stage scripts carry no local filesystem defaults | `ours-original` — agrees |

---

## 3. Concurrent public disclosure of the same mechanism class — PR #138

**This section exists because a maintainer will notice, and it is better that we say it.**

PR #138 `opal_v1` (Cristian, `ccastillo1043`, opened 2026-08-17T08:31:32Z) describes a mechanism
in the **same class** as our one claimed contribution: a probability correction learned online
from the already-decoded prefix, reproduced identically by encoder and decoder, adding **no**
learned tables or weights to the archive, with the entire gain arriving as rate because
reconstruction is unchanged. Their implementation uses 55 causal context families and a per-slot
online mixer (`.omx/research/ddm_hx1_pr_wave_harvest_20260817.md:57-62, 73-84`); ours is a
per-group statistical corrector. Different construction, same idea.

**We do not claim priority over PR #138, and we did not derive our corrector from it.** The
timeline, from commit timestamps in this repository:

| When (UTC) | What |
|---|---|
| 2026-07-22 | our "free decoder-derived context model" recorded — `ddm_g4_spatial_stationarity_603_DAG_FEED_20260722T212138Z.md`, commit `915e87dce3` |
| 2026-07-23 | priced at 89,161 B for a future stream — `ddm_c1_composed_candidate_spec_603_613_20260723.md:62` |
| 2026-07-29 | "decoder-derived contexts are free under rule 118" — `ddm_r7_token_coder_race_20260729.md:124` |
| **2026-08-17 08:31Z** | **PR #138 opened publicly** |
| 2026-08-17 14:41Z | our first measured corrector result, 1,549 B — commit `fdf3298801` |
| 2026-08-17 14:49Z | 1,598 B — commit `c8e6ee416c` |
| 2026-08-17 16:04Z | byte-closed at 181,161 B — commit `f7e29a124c` |
| 2026-08-17 19:32Z | we first read PR #138 — commit `f1de91eb46` |

So our design work predates PR #138's publication by about twenty-six days, and our first
measured result postdates it by about six hours on the same day. Neither the rr1, rr2, nor rr4
memo cites PR #138, because the intake came after the byte-close. The honest description is
**concurrent independent development**, and the correct citation is that PR #138 published this
mechanism class first.

PR #136 `hnerv_rc` (Jacky Li, `JPL11`) is adjacent and also earlier: an adaptive range coder with
per-tensor context reset, on a different vehicle (`ddm_hx1_pr_wave_harvest_20260817.md:161, 327`).

---

## 4. CONDITIONAL rows — only if a hot swap lands

These are not in the submitted archive. They are written now so that a swap cannot ship an
unattributed mechanism under deadline.

| Candidate | Mechanism | Class if it ships | Attribution owed |
|---|---|---|---|
| `fx1` | fixed-point integer log-odds mixer | `mechanism-adopt-with-attribution` | The **device-exact fixed-point formulation is ours** (radicals rather than lookup tables, because IEEE requires correctly-rounded `sqrt` but not `log`/`exp`). The **log-odds / context-mixing idea is not**: it is the PAQ lineage (Matt Mahoney), and in this contest it appears in PR #138 (§3) and, in weaker form, PR #136. Claim the formulation, cite the lineage. |
| `t1h` | zero-added-byte pose-coefficient re-solve | `mechanism-adopt-with-attribution` | **Traces directly to PR #133** (`cpr1_cbq_matched8`, `JasonMo123`), whose author published an effort-matched control showing 89.5% of his −0.0057 came from re-solving already-transmitted integer coefficients against the exact forward PoseNet at zero added bytes (`ddm_hx1_pr_wave_harvest_20260817.md:18-24`). If t1h ships, PR #133 must be cited in the body, not only here. |

---

## 5. What we claim, and what we do not

**We claim**, narrowly and with receipts above:

1. A **lossless entropy re-encode** of an inherited PR130/PR135-lineage archive using an original
   zero-byte decode-time probability corrector — 1,598 fewer archive bytes, decoded field provably
   unchanged. This is the whole of this generation's improvement. The mechanism *class* was
   published first by PR #138 (§3); the implementation and the measured result are ours.
2. The **HPAC probability object retrained in this repository** on our own label field, on PR130's
   architecture, with distortion-aware checkpoint selection over 81 retained candidates.
3. A small set of **admitted compensation edits** carried in the base.
4. The **receiver binding, custody chain, and reproducible build path**, including a compression
   entry point that carries no local filesystem layout.

**We do not claim** the learned vehicle, the semantic renderer, the pose carrier, the HPAC
architecture, the range-coder design, the compressed model container, or the residual payload. We
do not claim a distortion result: `d_seg` and `d_pose` are unchanged from the base candidate by
construction, and we say so rather than presenting an unchanged number as an achievement. We do
not claim priority for the decode-time-corrector idea.

## 6. Open provenance items

1. **Residual payload provenance is unresolved** (row 6). The shipped 100-byte residual is
   `74775aab…`; PR135's is `bd27a2dd…`; hv1 carries `bd27a2dd…` unchanged. Either `74775aab…` is
   an RCF1 re-framing of PR135's content, or it was re-fitted here. Generation 2 called it
   `ours-original` with no receipt. It is now classified `inherited-substrate` with the claim
   withdrawn — the conservative direction — and it stays that way until a receipt settles it.
2. **Row 3 (compressed model container)** is classified against the base only. If a PR-level sha
   comparison exists, it should replace the current wording.
3. **The C-compiler dependency** (§1.3) is currently unguarded in `inflate.sh`. Hash-safe cure for
   this candidate is the README declaration; any new receiver should carry a `command -v cc` guard
   from birth.

---

## 7. Generation 4 amendment — the hot swap landed (sz1 composed candidate; packet generation 3)

Numbering note: this document's internal generations count its own revisions; the packet's
`GENERATION_LOG.md` counts candidates. This amendment covers the packet's generation 3
(archive `debb025f45bb42e3…`, 179,930 bytes, measured `[contest-CUDA]` 0.15771357797660338).
Everything above stands for the rr4 candidate it was written against; the deltas are here,
append-only.

### 7.1 The fx1/fx2 conditional row is PROMOTED to the live table

The §4 conditional row for the fixed-point integer log-odds mixer shipped, in its deepened
`fx2` form: 13 contexts (causal spatial template + a decoded-class homogeneity context),
integer-only arithmetic, zero archive bytes. Classification exactly as pre-written:
**`mechanism-adopt-with-attribution`** — the device-exact fixed-point formulation and the
context design are ours; the log-odds / context-mixing idea is the PAQ lineage (Matt Mahoney),
published in this contest first by PR #138 and, in weaker form, PR #136. Token stream
110,512 → 109,801 bytes; decoded field unchanged (`9ba2e52b3096…`). The §3 concurrency
disclosure applies to this mechanism unchanged.

### 7.2 NEW live row — semantic serialization split

8,284 bytes of raw interleaved fp16 metadata in the semantic section are byte-planed (high-byte
plane, then low-byte plane) before the container's Brotli pass; the receiver applies the exact
inverse permutation before parsing; decoded values are unchanged; −520 bytes measured with a
delta-zero control. Classification: **`mechanism-adopt-with-attribution`** — byte-plane
(shuffle-filter) layouts are standard compression practice (the HDF5/Blosc shuffle lineage);
what is ours is the section-scoped measurement showing only the semantic section pays, the
zero-transmitted-byte versioning in an existing reserved header bit, and the fail-closed
receiver integration. We deliberately do not label a standard shuffle filter `ours-original`.
Offset selection: argmax over offsets 0–400; the ~22 B improvement over the derived offset is
Brotli alignment noise fitted to this frozen payload, not mechanism (adjacent offsets swing
±20 B) — the same qualification the fx2 token-model row carries (commit `31c64e4ce0`).

### 7.3 The t1h conditional row did NOT ship

The pose-coefficient re-solve remains out of the archive; its §4 conditional row stands
unchanged.

### 7.4 Updated claim arithmetic (§5 claim 1)

Cumulative lossless saving vs the inherited hv1 base: 2,829 bytes
(1,598 rr4 corrector + 711 fx2 mixer on the token stream, 520 serialization split), decoded
output provably unchanged at every step — the final step verified at the byte level (first
inflated output hash-identical between the fx2-only and composed rows, `9a6b75e5…`). The
per-generation claims in §5 otherwise stand; nothing in this amendment upgrades any
classification in our favour.

---

## 8. Generation 5 amendment — packet generation 4, the ck1 composed row-prune candidate

Archive `35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3`, 177,182 bytes,
measured `[contest-CUDA]` 0.15710198138050818 on a Tesla T4 over 600 samples. The document's
own section numbering runs one ahead of the packet's candidate numbering; see the banner.

**This amendment is the least flattering one this document has carried, and the reason is
structural rather than rhetorical.** Every prior generation's improvement was LOSSLESS: the
decoded state was held constant and provably identical to the inherited vehicle's, which is
what made "we borrowed their trained model and only re-coded the bytes" both true and easy to
say. This candidate abandons that property. It re-quantizes their semantic tensors and
re-solves their pose carrier. The result is smaller and scores better, but it is no longer a
faithful reproduction of anyone's trained state — including theirs.

### 8.1 Rows 1 and 2 lose their byte-identity — RE-CLASSIFIED

| §2 row | was | now | receipt |
|---|---|---|---|
| 1 Semantic renderer state | `inherited-substrate` (PR135, **proven byte-identical after decode**) | `mechanism-adopt-with-attribution` — **our format over their values, and the values are lossily changed** | semantic body 36,130 B; stream 31,469 B; `SA3_REBASE.json:results.candidate_nosplit` |
| 2 Pose carrier state | `inherited-substrate` (PR135, **proven byte-identical after decode**) | `mechanism-adopt-with-attribution` — **their solver form, our binding, their lattice re-solved** | 6,713 of 7,200 signed-int12 coordinates changed; `SA3_REBASE.json:results.candidate_nosplit.changed_coordinates` |

Neither re-classification is an upgrade in our favour, and the direction matters. The
underlying learned content is still PR #130 / PR #135's — we did not train a renderer or a
pose model. What we did is take their trained tensors and represent them worse on purpose,
then repair the damage to one scorer using their own solver form. Calling the result
`ours-original` would be a fabrication; calling it `inherited-substrate` would now be a false
claim of fidelity to their bytes. `mechanism-adopt-with-attribution` is the only honest cell.

### 8.2 The two live mechanisms

**SM3R mode 6 — row-pruned, mixed-depth semantic quantization.**
`mechanism-adopt-with-attribution`. Three FiLM weight tensors keep only their two
highest-L2-norm rows, sent as a row bitmask plus a compact kept-rows block; a per-tensor
4-bit depth nibble table then drops `frame_embed.weight` and `blocks.0.film.weight` to 3-bit
codes while the remaining quantized tensors stay at 4. **What is ours:** the SM3R/SD1M wire
formats (defined in this repository, explicitly not understood by the public receiver), the
measurement that identified those two tensors as the surviving marginal after a 99% row
prune, and the fail-closed receiver integration (the receiver recomputes the tensor selection
mask and refuses on mismatch). **What is not:** magnitude-based structured pruning and
mixed-precision weight quantization are standard practice with a long public literature, and
the tensors being pruned are PR #135's. We do not label a row-prune `ours-original`.

**In-compile frame-0 pose compensation.** `mechanism-adopt-with-attribution`, and the
attribution is heavier than the previous rows'. The solve adapts PR #135's own banked
Gauss-Newton form and bounded integer-cube solver from its published experiment book; PR #135's
competitive mechanism was itself "joint renderer edit followed by frame-0 carrier re-solve",
so **the edit-then-recompensate pattern is theirs, not ours.** What is defensibly ours: the
in-compile content-fingerprint binding that fails closed rather than carrying a compensation
onto a changed lattice (this repository shipped that bug once and the guard is the cure), the
frame-0/frame-1 disjointness argument that makes the compensation `d_seg`-invariant by
construction, the step-matched Jacobian, and the rate route — folding the compensation into
the existing Rice-coded lattice instead of a sidecar overlay, which is what turns roughly
7,000 bytes into 41. Encoder-side disclosure: the build-time decode probe puts PR #135's
experiment-book source on `sys.path`. It is encoder-side only and nothing from it enters the
archive; §2 row 8a is the precedent for that kind of row.

### 8.3 One row is DROPPED and one is UNCHANGED

- **§7.2 semantic serialization split: DROPPED.** `reserved = 0`, `semantic_split = false`.
  The row-prune changes the semantic body length, so the split's pinned region no longer
  covers the metadata it was fitted to; re-measured on the edited body the split is
  **negative**. Two credits over the same redundancy do not add. Its receiver support ships
  and is inert on these bytes. The §7.4 cumulative-saving arithmetic therefore no longer
  applies to the shipped archive: 520 of those 2,829 bytes are not in it.
- **§7.1 fx2 token model: UNCHANGED and still shipping.** The tail section (token stream plus
  residual payload and table codes, 109,897 B) and the HPAC stream (13,515 B) are spliced
  byte-identically from the previous candidate. Its classification and its PR #138 / PR #136
  concurrency disclosure stand verbatim.

### 8.4 NEW row — carrier framing runtime patch

`ours-original`, **zero counted bytes.** `runtime/residual_archive.py` gains
`DDM_SA2_VARIABLE_PACKED_CAP1_V1`: the packed-CAP1 section length is derived from the
section's own u24 bit counts instead of a pinned byte constant, because a compensated lattice
does not have the base lattice's Rice-residual length. It is a generic framing algorithm and
carries no video-derived content (rule 118); the compile receipt records
`counted_bytes: 0` against it.

### 8.5 Claim arithmetic

Against the immediately prior row in this lineage (177,576 B, S 0.1571619225142182) the
measured legs are rate −2.6235e-04, seg +1.7400e-04, pose +2.8407e-05, net −5.994113e-05 —
about 23% of the rate credit retained. **The distortion legs are real costs, not rounding.**
This is the first packet candidate whose improvement is not purely rate, and the two scorers
are paid out of the byte saving rather than held constant.

Two honest qualifications belong here rather than in a footnote:

1. **The seg leg landed at roughly three times the modelled cost.** The realized net is 35%
   of the pre-fire projection, and the entire miss is SegNet: this repository has a measured
   CPU→CUDA transfer law for pose and none for seg, so the CPU-advisory seg delta that fed
   the projection was an upper bound on the win, not an estimate of it. Recorded as a
   standing gap, not smoothed over.
2. **Reproduction is NOT re-verified for these bytes.** Generation 3 could claim an
   end-to-end rebuild from pinned retained inputs through one entry point. That entry point
   has not been re-run for this candidate. The compile receipt proves how these bytes were
   assembled and the receiver parse-back passes over the shipped runtime, but the
   generation-3 VERIFIED label does not transfer and is not inherited.

Nothing in this amendment upgrades any classification in our favour, and two rows moved
against us.


---

## 9. Generation 6 amendment — packet generation 5, the jg5 joint-waterfill candidate

Archive `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`, 180,625 B,
`[contest-CUDA]` T4 n600 S = 0.14839100138338618.

### 9.1 Ancestry — one step appended to §1.1

The §1.1 chain is unchanged through generation 4 and gains one step:

| Step | What | Archive bytes |
|---|---|---:|
| … | (§1.1 chain through the generation-4 candidate) | 177,182 |
| **this** | **jg5 joint admission waterfill + carrier re-solve on the candidate's own renders** | **180,625** |

The ancestry root is unchanged and is restated here so it is never one link away
from a reader: **PR #130 `semantic-pose-HPAC_CPR1` by Fesal Fayed (`fesalfayed`)**
and **PR #135 `semantic-pose-HPAC_CPR1_polished` by Shreyan Mohanty (`codexblack`)**,
with **PR #133 `cpr1_cbq_matched8` by `JasonMo123`** transitively in the ancestry via
PR #135. The trained semantic renderer state and the pose carrier state that this
submission re-represents are theirs. We claim no part of them.

### 9.2 Rows carried forward from §8 UNCHANGED

| §2 row | Class at this generation | Note |
|---|---|---|
| 1 Semantic renderer state | `mechanism-adopt-with-attribution` | Our format over PR #135's values, values lossily changed. §8.1 unchanged. |
| 2 Pose carrier state | `mechanism-adopt-with-attribution` | Their solver form, our binding, their lattice re-solved. §8.1 unchanged. |
| 3 Compressed model container | `inherited-substrate` | PR #130/#135. |
| 4 HPAC probability object | `inherited-substrate` | Architecture PR #130/#135. |
| 5 Compensation blob | `mechanism-adopt-with-attribution` | Edit-then-recompensate is PR #135's pattern. |
| 6 Residual payload + table codes | `inherited-substrate` | PR #130/#135. |
| 7 RC64 token stream | `mechanism-adopt-with-attribution` | fx1/fx2 model-axis work is ours; the coder is theirs. |
| 8a/8b RC64 backend | `inherited-substrate` | Encoder-side and shipped receiver both. |
| 9 Receiver binding / assembly / custody | `ours-original` | Unchanged from §8. |
| 10 End-to-end compression entry point | `ours-original` | Unchanged from §8; **not re-run for these bytes.** |

### 9.3 NEW `ours-original` row — joint admission waterfill

| Field | Value |
|---|---|
| Mechanism | Joint admission of seg token edits and the pose carrier as ONE waterfill, swept over a Lagrange multiplier on pose damage, with the carrier re-solved against the candidate's OWN edited renders |
| Class | `ours-original` |
| Counted archive bytes attributable to the mechanism | **0** — it selects and re-solves values inside sections that already exist; it adds no new section |
| Receipt | `.omx/research/ddm_jg5_pose_resolve_on_edited_renders_20260819.md`; seal `96e9860aad9021e6…`; admitted 455 of 573 edits; 600/600 pairs stopped on `no_improving_step` with zero budget hits |

**What is ours.** The decision rule. The predecessor kept all 573 edits and paid about 13
times more pose than the edits bought in seg; the waterfill admits an edit only if it pays
for the pose it costs, and the stop rule for the carrier descent is derived from materiality
rather than set as a fixed iteration budget.

**What is not ours.** Everything the rule operates ON. The renderer state, the carrier
lattice, the container, the probability model architecture and the range coder are the
inherited PR #130 / PR #135 substrate. A better decision rule over someone else's
representation is a contribution to the decision, not to the representation.

**Prior art we do not claim.** The decode-time-corrector mechanism class was published first
by **PR #138 `opal_v1`**; the edit-then-recompensate pattern is **PR #135's**. We make no
priority claim on either. Solving admission and compensation jointly rather than in sequence
is the part we claim, and only that part.

### 9.4 Claim arithmetic

Score 0.14839100138338618 = seg 0.020139 + pose 0.007981227975693965 + rate 0.1202707734076922.

Against packet generation 4 the legs are rate +2.2926e-03 (+3,443 bytes), seg −1.0170e-02,
pose −8.3353e-04, net −8.7110e-03. Both distortion legs are bought and the rate leg is paid —
the reverse of every earlier generation in this packet.

**Two honest qualifications, restated rather than inherited.**

1. **The improvement is a re-decision over borrowed content.** No new learned artifact was
   trained for this candidate. The gain comes from choosing which of someone else's
   representations to perturb and by how much.
2. **The end-to-end rebuild has not been re-run for these bytes.** §2 row 10 remains
   `ours-original` as a mechanism, but its verification status for THIS candidate is open,
   and it is listed as owed rather than quietly carried over.
