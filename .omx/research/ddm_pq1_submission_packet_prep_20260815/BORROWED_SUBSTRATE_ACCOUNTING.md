# Borrowed-substrate accounting — generation 3 (rr4 re-encode candidate)

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
| `constriction` | `inherited-substrate` | third-party entropy-coding library |
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
