# PR #86 + PR #130 — FULL-STACK ENGINEERING INTAKE (both-ways)

**Date (actual UTC):** 2026-07-27T22:44Z. *(Filename carries the operator-specified `20260728` stamp;
the real fetch/analysis UTC is recorded here to avoid a date-drift confound.)*
**Charter:** operator 2026-07-27 — *"We can do much better than pr86 and pr130 but we could learn from
their engineering and full stack."*
**Scope (BINDING):** HARVEST-SIGNAL-ONLY. NO-FAKE #7 — intake is **lessons only**, never vehicles,
carriers, or calibration. No mechanism below is proposed for adoption without a derive-or-race path.
All their numbers are **[external]**; all ours carry their measurement axis and receipt.
**Pointer delta: ZERO.** This is MEANS. It does not move the exact score.

---

## 0. STORES CONSULTED

`CLAUDE.md` (§NO FAKE IMPLEMENTATIONS #7/#8 · §Public frontier watch and intake · §Bit-level
deconstruction and entropy discipline · §Apples-to-apples evidence discipline · §Canonical leaderboard
binding-depth discipline L1–L32 + its 2026-07-24 DEMOTION BANNER · §inflate.py is a FREE interpreter /
rule-118 · §Frontier scores are pointer-only) ·
`.omx/state/canonical_frontier_pointer.json` (read 2026-07-27; `upstream_leaderboard_snapshot`
fetched 2026-07-27T18:03:21Z) ·
**prior intakes consumed, not duplicated:** `.omx/research/public_pr129_132_intake_20260725.md`
(Fable, incl. its §2b PR86 deep-intake) · `.omx/research/original_taskspace_inverse_witness_codec_20260725/
codex_findings_pr86_pr130_program_archive_crux_20260726_codex.md` ·
`.omx/research/public_pr86_pr87_intake_20260504_codex.md` (815 lines, 2026-05-04) ·
`.omx/research/codex_findings_ddm_gc4_full_pantheon_pr130_adjudication_20260725_codex.md` ·
memories `goal_is_sub015_or_below_official_leaderboard_best_pointer_fixation_abandoned_20260727` ·
`no_old_lineage_ban_hnerv_pr_substrates_20260723` · `borrowed_incumbent_rate_polish_permanently_dead_20260725` ·
`archive_gravity_pulls_off_realization_crux_20260720` · `seg_and_pose_solved_exact_lattice_realization_one_rd_axis_20260719` ·
`v10_description_pivot_budget_box_and_realization_crux_20260719` · `frozen_scorer_exact_factorization_20260715` ·
`past_solves_rate_naive_free_null_counted_partition_20260721` · `staleness_is_a_named_confound_class_freshness_at_consumption_20260723`.

**Recall outcome:** acquisition was already complete. Detached clones for both PRs exist
(`experiments/results/public_pr{86,130}_intake_20260725_fable/`, plus PR86's actual `archive.zip` in
`public_pr86_intake_20260504_codex/`). **Nothing was re-cloned; nothing was checked out into the main
working tree.** This intake's marginal contribution is (a) a **code-verified** byte ledger that
corrects two of our own memos, (b) full pose-carrier mechanics for the A3 line, (c) the both-ways gap
table, (d) box arithmetic at the real operating point.

---

## 1. CUSTODY

| | PR #86 | PR #130 |
|---|---|---|
| title | `jas0xf_adversarial_neural_representation` | `semantic-pose-HPAC_CPR1` |
| author | jas0xf (Jason Yuan, UCSD) | fesalfayed |
| head SHA | `0eabe354f09b7490fd1cbb2b05a9102ab528d4d4` | `9a77b6ad660d6310ab54436757ab07a9bbc9f3e1` |
| merge | **MERGED** `14bcede815306415a0005c3cd98804151bce4049` (2026-05-04) | closed, unmerged |
| leaderboard | — | **RANK 1**, official display **0.172** |
| archive bytes | 207,579 | 191,052 |
| archive sha256 | `e67b7c22…ceb03cef` *(measured locally)* | `0491d5df…1e18fdc7cd` *(from `verification.json`; archive not held locally)* |
| submission LOC | 610 | 3,201 |
| external deps | torch, numpy, constriction, **pyppmd** | torch, numpy, constriction (+ stdlib `lzma`) |

Both are the **same lineage**: PR130's own `LINEAGE_AND_CITATIONS.md` names PR86 as its *"direct public
predecessor"* and asserts a mechanically verifiable `git merge-base`. PR86 is the root; PR130 is the
refinement two and a half months later.

---

## 2. BIT-LEVEL ANATOMY (measured / code-verified)

### 2.1 PR #130 — and a correction to our own record

`verification.json` (schema v4) gives a complete ledger. I **cross-verified it against
`inflate.py:main()`'s actual parse order** rather than trusting the JSON:

```
payload = (data_dir / "p").read_bytes()                    # 190,952 B, single STORED member
models_bytes = struct.unpack_from("<I", payload)[0]        #      4 B u32 prefix
models_raw   = lzma.decompress(payload[4:4+models_bytes])  # 73,968 B xz  → 83,493 B raw
tokens       = decode_tokens(hpac, payload[4+models_bytes:])  # 116,980 B range-coded
```

| level | object | bytes | note |
|---|---|---:|---|
| archive.zip | | **191,052** | |
| | ZIP overhead | 100 | **one** member, `stored` |
| member `p` | | 190,952 | |
| | u32 length prefix | 4 | |
| | xz models blob | 73,968 | raw 83,493 → xz saves 9,525 |
| | **HPAC token stream** | **116,980** | range-coded, **0.007933 bpp** |
| models raw | 2× u32 length fields | 8 | |
| | semantic renderer (int4) | 40,252 | |
| | **pose carrier (CPR1)** | **23,054** | |
| | HPAC integer model | 20,179 | |

Arithmetic closes **exactly**: `4 + 73,968 + 116,980 = 190,952` and
`8 + 40,252 + 23,054 + 20,179 = 83,493`.

> ⚠ **CORRECTION TO OUR OWN RECORD.** `.omx/research/public_pr129_132_intake_20260725.md` §2 reported
> *"semantic stream 40,252B + models … 73,944B"*. **40,252 is the int4 RENDERER, not the token
> stream**, and the memo **omitted the 116,980 B token stream entirely** (61.3% of the member). Its
> derived lesson-#3 benchmark — *"~114KB-for-exact-partition"* — is therefore **wrong by ~50%**.
> The 2026-07-26 codex crux memo's mapping was correct. Corrected leg attribution (DERIVED; xz
> compresses the three model objects jointly, and CPR1 is already entropy-coded so assumed
> ~incompressible):
>
> | leg | bytes | % of member |
> |---|---:|---:|
> | **seg / partition** (tokens + renderer + prior) | **~167,894** | **87.9%** |
> | **pose** (CPR1 carrier) | **23,054** | 12.1% |
>
> **The Vehicle-B rate benchmark is ~168 KB for the partition, not ~114 KB.**

### 2.2 PR #86 — measured from the archive, and a second correction

Enumerated directly from the local `archive.zip`:

| member | bytes | % | compression |
|---|---:|---:|---|
| `tokens.bin` | 113,900 | 54.87 | stored |
| `slave.pt.gz` | 32,287 | 15.55 | stored |
| `master.pt.gz` | 31,144 | 15.00 | stored |
| `hpac.pt.ppmd` | 28,243 | 13.61 | stored |
| `meta.pt` | 1,499 | 0.72 | stored |
| **Σ members** | 207,073 | 99.76 | |
| **ZIP overhead** | **506** | 0.24 | **5 members** |
| **archive.zip** | **207,579** | | |

> ⚠ **SECOND CORRECTION.** The writeup's section table (master 31,154 / slave 32,210 / meta 1,563 /
> hpac 27,579 / tokens 113,860), which our 2026-07-25 memo propagated, **does not match the shipped
> members** — up to 664 B divergence on `hpac`. The totals only agree because the writeup implicitly
> absorbs the 506 B ZIP overhead. **The measured member sizes are authoritative.**

Every member is `stored`: each is already entropy-coded, so ZIP adds nothing. PR86 pays **506 B** of
member overhead against PR130's **100 B** — **406 B lost purely to member count.** An independent
measured confirmation of our own L20 monolithic-archive lesson.

---

## 3. SCORES — recomputed from components, axis-labeled

All recomputations reproduce the reported values to 6 decimals (`S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489`).

| row | d_seg | d_pose | bytes | seg | pose | rate | **S** | axis |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PR86 bot | 6.7815e-4 | 4.5701e-4 | 207,579 | 0.06782 | 0.06760 | 0.13822 | **0.273636** | **[contest-CUDA]** |
| PR130 bot (CPR1) | 2.9660e-4 | 2.331e-5 | 191,052 | 0.02966 | 0.01527 | 0.12721 | **0.172141** | **[contest-CUDA]** |
| PR130 author, RTX 2000 Ada, **SOURCE** archive | 2.8609e-4 | 1.967e-5 | 194,380 | 0.02861 | 0.01402 | 0.12943 | 0.172064 | [external, non-1:1] |
| PR130 author, RTX A4500, **SOURCE** archive | 2.9607e-4 | 1.981e-5 | 194,380 | 0.02961 | 0.01407 | 0.12943 | 0.173111 | [external, non-1:1] |
| "CPR1 projected 0.169848" | 2.8609e-4 | 1.967e-5 | 191,052 | 0.02861 | 0.01402 | 0.12721 | 0.169848 | ⚠ **PROJECTION** |

> ⚠ **HONESTY FLAG on 0.1698.** That number substitutes distortion metrics measured on the **194,380 B
> SOURCE** archive into the **191,052 B CPR1** byte count. **No evaluator ever produced 0.1698 on the
> CPR1 bytes.** Our 2026-07-26 codex crux memo cited it as *"approximately 0.1698476624 from displayed
> components"* — that reads as a measured row and should not. **The only bot-measured PR130 row is
> 0.172141 [contest-CUDA]**, matching the leaderboard display 0.172.
>
> Also: **PR130 has no published [contest-CPU] row.** Per apples-to-apples discipline, its rank-1
> standing rests on a single axis.

### 3.1 The dominant structural fact

**PR130's score is 73.9% rate.**

| term | value | share |
|---|---:|---:|
| rate | 0.12721 | **73.9%** |
| seg | 0.02966 | 17.2% |
| pose | 0.01527 | 8.9% |

Driving **both** distortions to exactly zero leaves them at **S = 0.1272**. Their remaining headroom is
overwhelmingly rate. This governs every "can we do better" judgement below.

---

## 4. FULL-STACK ENGINEERING READ

### 4.1 PR130's pose carrier — full mechanics (the direct A3 consumer)

`frame_0` is a **pure PoseNet actuator with zero semantic content** (SegNet reads the last frame only).
Stored objects and their exact costs:

| object | shape | precision | coding | bits | **bytes** |
|---|---|---|---|---:|---:|
| basis | `[12, 3, 24, 32]` (27,648 sym) | int5 | canonical Huffman of signed-zigzag | 104,135 | **13,017** |
| coefficients | `[600, 12]` (7,200 sym) | int12 | per-dim Rice of **delta-zigzag** codes | 79,076 | **9,885** |
| header + fp32 scales + Huffman lengths + Rice k | | | | | 152 |
| | | | | | **23,054** |

Decode recipe (all of it a **free** deterministic generator under rule-118 — only the codes are counted):

```
basis  = int5_codes * per_dim_fp32_scale
       → bicubic ↑ (384,512)
       → zero-mean, unit-RMS normalize per basis element
coeff  = cumsum(unzigzag(delta)) & 0xFFF, sign-corrected, * per_dim_fp32_scale
carrier = einsum("bk,kchw->bchw", coeff, basis) / sqrt(12)
slave  = round(clamp(127.5 + 64.0 * carrier, 0, 255))
       → bicubic ↑ (874,1164) → clamp → round → uint8
```

Three engineering facts that matter to us:

1. **The actuator is stored on a 24×32 grid — 768 pixels — and bicubic-upsampled twice.** d_pose
   2.33e-5 is reachable with a 12-dimensional, *very-low-spatial-frequency* field. Strong empirical
   evidence about the smoothness of the pose preimage.
2. **Decode-side normalization is a precision reducer.** Zero-mean + unit-RMS at decode strips scale,
   so the stored integers only carry *shape* — which is why **5 bits suffice** for the basis
   (3.77 bits/symbol achieved). This is rule-118 used to cut precision, not just to dodge bytes. A
   sharper use of the doctrine than we have applied.
3. **Cost structure:** 38.4 B/pair total = 16.5 B/pair marginal (coefficients) + 21.7 B/pair amortized
   (basis). The Rice `k` values are all 8–9, i.e. each per-frame coefficient delta carries ~9–10 bits
   of genuine entropy — the temporal delta+cumsum is exploiting real trajectory smoothness, and there
   is **not** much left in that direction.

### 4.2 PR86's distinctive approach, and why it scored 0.2736

PR86's contribution is the **reframe**, stated in its writeup §1: *"the true entropy isn't the video
itself, but the expected outputs of the evaluation models… The answer itself is the true entropy."*
An independent public discovery of the evaluator-equivalent-witness paradigm.

Its **adversarial existence proof** [external claim, their measurement]: direct gradient descent on
input pixels against the frozen evaluators reached **d_seg 0.00 and PoseNet MSE < 1e-9 in 2h on one
T4**. They abandoned it because storing per-pixel adversarial frames busts rate, and generating them at
inflate time would need scorer weights in-archive — the same rule-118 wall we codified.

Architecture: **master** `TokenRendererV62` renders frame_1 from the 5-class token grid (RF-7 claim
below); **slave** `ShrinkSingleNeRV` renders frame_0 from a 6-float latent for pose duty; **HPAC**
prices the token stream at 0.00773 bpp via a two-level patch-group causal factorization — 192
independent P=32 patches × group order `s = c + 2r`, giving **94 sequential steps/frame instead of
196,608** (~2000× fewer), which is what makes an AR prior fit the 30-min budget at all.

**Why 0.2736:** its pose leg was weak (slave NeRV, 32,287 B → d_pose 4.57e-4 → **0.0676 S**, as large
as its entire seg term) and its rate was 207,579 B.

### 4.3 PR86 → PR130: where the 0.1015 actually came from

This is the most instructive single comparison in the intake, and it **inverts the intuitive story**:

| leg | PR86 | PR130 | Δ |
|---|---:|---:|---:|
| token stream | 113,900 B @ 0.00773 bpp | 116,980 B @ 0.00793 bpp | **+3,080 B, bpp slightly WORSE** |
| models (renderer + prior) | 60,886 B | ~50,914 B (xz'd) | −9,972 B |
| pose carrier | 32,287 B | 23,054 B | −9,233 B |
| ZIP/member overhead | 506 B | 100 B | −406 B |
| **d_seg** | 6.78e-4 | 2.97e-4 | **2.29× better** |
| **d_pose** | 4.57e-4 | 2.33e-5 | **19.6× better** |
| **S** | 0.273636 | 0.172141 | **−0.1015** |

**The entropy coding of the partition did not improve — it got marginally worse.** The gain came from
(a) a better renderer at fewer bytes, (b) the **pose leg swap** (NeRV slave → low-rank neutral-gray
carrier: 19.6× better d_pose at 9,233 *fewer* bytes), and (c) determinism engineering that let them
ship the integer path.

The pose swap alone is worth **0.0523 S** (0.0676 → 0.0153). **Roughly half the total improvement is
one carrier redesign.**

### 4.4 Determinism engineering — the two generations

- **PR86:** pin the non-portable ops to CPU FP32. `bake_film_table()` computes the FiLM table on CPU
  because cuBLAS tiny-matmul kernels differ Ada vs Turing; the **entire HPAC decode is forced to CPU**
  with the comment that different GPUs produce slightly different FP32 softmax → *the arithmetic
  decoder asserts*. Correct, but pays a large speed penalty.
- **PR130:** replace float inference with an **integer lattice** — bounded integer intermediates,
  dyadic requantization, canonical entropy logits on a 1/8-logit lattice → symbol/token-exact decode
  across architectures, and the decode can stay on GPU. Strictly stronger *and* faster.
- **Honest boundary they draw:** PR130 claims **symbol/token exactness only**. RGB bit-identity is
  explicitly **not** claimed; the portability gate is "bitstream decodes + official evaluator
  succeeds." Evidence: 2 full official evaluations on independent architectures (Ada, Ampere),
  cross-GPU entropy-logit hash match, and a 2,197.6 s CPU full-token decode reproducing the expected
  token sha256.
- **PR130 has no CPU fallback at all** — `inflate.py` raises if `torch.cuda.is_available()` is false.
  A deliberate single-rail bet on the official GPU runner.

### 4.5 Reproduction and provenance gating

`compress.sh` is **fail-closed on bytes at both ends**: it verifies the SOURCE archive's sha256 and
byte count *before* repacking, and the OUTPUT's sha256 and byte count *after*, refusing on mismatch.
Scope is honestly narrowed: `"full_training_from_original_video_included": false` — the reproduction
covers the lossless frozen-source repack only. Plus `MANIFEST.sha256` over all 12 source files, 14
pytest cases, 200 randomized roundtrip cases, and malformed-stream tests.

`LINEAGE_AND_CITATIONS.md` is the **best borrowed-substrate accounting observed in this contest**: a
mermaid lineage map, a dated chronology with immutable anchors, a mechanically verifiable
`git merge-base` assertion, an explicit inherited-vs-original table, and a narrowing clause —
*"original within this audited lineage"* ≠ worldwide priority. It even cites qpose #67/#79 as
**novelty-limiting prior art against its own claim**. That is the NO-FAKE #7 discipline, independently
invented, and it is worth citing as external precedent for our own disclosure format.

---

## 5. BOTH-WAYS GAP TABLE

The operator's *"we can do much better"* made checkable. Our receipts are labeled with their axis and
age; per the staleness discipline, the v10-line receipts below are **6–8 days old** and should be
re-verified at consumption if they become load-bearing for a launch.

### 5.1 Where OUR stack already exceeds theirs

| # | dimension | OURS (receipt) | THEIRS | verdict |
|---|---|---|---|---|
| **E1** | **seg distortion physics** | **d_seg = 0.0 exactly**, 3,538,944/3,538,944 blocks `FEASIBLE_EXACT` (n6, receipt `v10_uint8_lattice_feasibility_receipt_20260718.json`); n600 replay d_seg 9.66e-7 all-ULP | trained approximation, d_seg 2.97e-4 [contest-CUDA] | **OURS decisively** — we have an *exact solution*, they have a fit. Their seg costs 0.0297 S; ours ~0. |
| **E2** | **pose distortion physics** | **d_pose 9.3e-10** through the full upstream `DistortionNet` on solved frames, **at zero extra pose bytes** (pose falls out of plane proximity; law `pose_plane_proximity_corollary_v1`, prediction confirmed 96/96) | 2.33e-5 from a dedicated 23,054 B carrier | **OURS decisively** in the exact regime — 25,000× better *and* the carrier bytes vanish. |
| **E3** | **scorer factorization depth** | **exact algebraic factorization**: `PoseNet.preprocess_input` resizes FIRST with the same bilinear `A` (`modules.py:71-75`) ⇒ pose input is a pure function of `A(frame)` ⇒ **one realization buys BOTH scorers** | exploit only the weaker empirical fact that frame_0 is seg-free; they train two representations and compose, and their own lineage doc concedes "PoseNet still evaluates their composition" | **OURS** — strictly deeper. They found the corollary; we have the theorem. |
| **E4** | **authority / axis discipline** | dual-axis CPU+CUDA mandate; numpy-fp32 bit-identical authority; MPS-never; v10 spine measured **twice bit-identical** across macOS-CPU and [contest-CPU] Modal Linux x86_64 (`fc-01KXXRAR7341QCJ6XWKV4S3QCW`) | single-axis: **no [contest-CPU] row exists** for the rank-1 archive; RGB bit-identity explicitly not claimed | **OURS** |
| **E5** | **apparatus** | ~400 catalog gates, verdict-scope ladder, 3-layer confound immune system, triality/quadrality consistency | one excellent `verification.json` per artifact | **OURS** in coverage — though see T5 on what that coverage has *produced*. |

### 5.2 Where THEIR execution beats ours

| # | dimension | THEIRS | OURS (receipt) | verdict |
|---|---|---|---|---|
| **T1** | **rate — the whole game** | **191,052 B receiver-closed at d_seg 2.97e-4 / d_pose 2.33e-5**, on the official rail, bot-verified | our small receiver-closed archive is v3 **91,062 B at d_seg 3.556e-3** (12× worse seg; 99% of its S loss is pose). Our good-distortion spine is **409,526,925 B** — 2,144× over box | **THEIRS decisively.** We have **no** receiver-closed archive that is simultaneously small *and* good-distortion. They do. **This single row is the entire gap.** |
| **T2** | **a working learned prior for the partition** | 116,980 B for 600×384×512 5-class tokens = **0.00793 bpp**, decoding inside the 30-min budget | exact-plane lossless storage is **RATE-DEAD at FAMILY scope** — 5 codec families within 1.9× of a ~334 KB/pair floor (#541); best implied S 168.71 | **THEIRS** — they have a shipped answer to the question our best five families failed. |
| **T3** | **a shipped pose leg in a compact archive** | 23,054 B → 0.0153 S contribution | banked R1 `dxi` **7.2 KB at d_pose 1.610e-3 → 0.127 S contribution** | **THEIRS on shipped S** (8.3× better contribution at 3.2× the bytes). Note this does *not* contradict E2: our 9.3e-10 lives in the rate-dead exact regime; our *shipped compact* pose leg is worse than theirs. |
| **T4** | **end-to-end closure** | 3,201 LOC, one member, self-verifying manifest, byte-exact reproduction gate, 2 independent architecture validations — **shipped** | far more apparatus; **no shipped competitive archive** | **THEIRS** |
| **T5** | **apparatus→artifact conversion** | PR86→PR130 in ~2.5 months: one carrier redesign (≈half the gain) + determinism rework, and they took rank 1 | same period: pointer **0.19108 UNMOVED** | **THEIRS.** Our E5 coverage advantage has not converted. Stating this plainly is required by the means/ends firewall. |
| **T6** | **member economy** | 1 member, 100 B overhead | — | **THEIRS vs PR86** (406 B); we already hold the lesson as L20, but it is a measured confirmation. |

### 5.3 Neutral / convergent (independent confirmations, nothing new to build)

| dimension | note |
|---|---|
| answer-as-entropy ≡ evaluator-equivalent witness | same reframe, found independently (PR86 writeup §1). FROZEN-SPACE. |
| frame_0 seg-free ⇒ master/slave split | they shipped it May 2026; we measured it independently (`frozen_scorer_exact_factorization_20260715`). Doubly confirmed. |
| decode-time generated bases at zero rate | qpose `make_dct_basis()`; PR130's `normalized_basis` — maintainer-merged precedent for our rule-118 doctrine. |
| GT-decode custody | their **20× gap** (train d_seg 3.5e-5 vs evaluator 7.12e-4) from PyAV-vs-NVDEC chroma conversion — an independent instance of our GT-decode law. |
| pose-legible-by-construction beats reconstruction | their PR86→PR130 carrier swap is external confirmation of our photometric-wall law. |

### 5.4 The honest one-line summary

**Our distortion physics is strictly and provably ahead; their rate engineering is strictly and
measurably ahead; rate is 74% of the score.** The operator's "we can do much better" is *supportable* —
but only through the rate axis, and only if our exact-distortion advantage survives compression. It is
not supportable from anything we have shipped to date.

---

## 6. BOX ARITHMETIC AT THE REAL OPERATING POINT (→ #613)

`bytes = (S − distortion) · 37,545,489 / 25`. Break-even reference: **1e-3 S = 1,501.8 B**.

| distortion assumption | Σ distortion | max bytes @ S=0.172 | max bytes @ **S=0.150** |
|---|---:|---:|---:|
| PR130-class (seg 2.97e-4, pose 2.33e-5) | 0.04493 | 190,840 | **157,800** |
| **our v10 spine seg (1.52e-4) + PR130-class pose** | 0.03047 | 212,556 | **179,516** |
| zero distortion | 0.00000 | 258,313 | 225,273 |

Two consequences:

1. **PR130 is 33,252 B over the sub-0.15 line** at its own distortion. Their path to 0.15 is a **17.4%
   byte cut** — and their token stream is already at 0.0079 bpp with a *learned* prior, i.e. the cheap
   entropy wins are spent.
2. **Our measured seg advantage is worth ~21,700 B of rate budget** at any target (157,800 → 179,516 B
   for S=0.15). That is the quantitative form of "we can do much better": *not* a better score today,
   but a **14% larger byte budget** to spend, **if** the distortion holds through compression.

Cross-check against our own caps (memory `v10_description_pivot…20260719`, 8 days old): pointer-tie
216,223 B; strict sub-0.15 **154,524 B**. My 157,800 B figure is at PR130's *measured* distortion; the
154,524 B cap assumes a slightly higher distortion. Same neighbourhood — **the two derivations agree**,
which is a useful independent check on #613.

---

## 7. RANKED LESSONS TABLE

**Adoption classes.** `LESSON-ONLY` = a fact/discipline we absorb, no code adoption.
`RACE-AS-CANDIDATE` = vehicle-agnostic mechanism eligible to enter a **real race** against our own
alternatives; never adopted by citation. `N-A` = we already exceed it or it is inapplicable.
**Constants are poison:** no number below enters a config without derive-or-race.

| # | lesson | evidence | our consumer | class |
|---|---|---|---|---|
| **1** | **A 12-dim, 24×32-grid, bicubic-upsampled low-frequency field steers PoseNet to d_pose 2.33e-5 for 23,054 B (38.4 B/pair).** The pose preimage is *smooth and very low-rank*. | `inflate.py:601-651`, `verification.json.carrier_codec`; bot d_pose 2.331e-5 [contest-CUDA] | **A3 pose line** — an external existence proof that a compact, seg-free, pose-legible frame_0 works at scale on the official rail. De-risks the fork; sets 38.4 B/pair as the price to beat. | **LESSON-ONLY** (existence proof). Their basis/dims/amplitude = VEHICLE-SCOPED. |
| **2** | **Decode-side zero-mean + unit-RMS normalization is a precision reducer** — it strips scale at decode so stored codes carry only shape, letting int5 suffice (3.77 bits/sym achieved). | `inflate.py:600-606` `normalized_basis`; `BASIS_BITS = 5` | **coder stack** + rule-118 doctrine. Generalizes: *any* decode-time invariance we can synthesize for free reduces the bit-depth of what we must store. We have used rule-118 to dodge bytes, not to cut precision. | **RACE-AS-CANDIDATE** — vehicle-agnostic; race against our current quantization ladder. |
| **3** | **Rate is 73.9% of the rank-1 score; zero distortion still leaves S=0.1272.** | §3.1, recomputed from bot components | **#613 box arithmetic** + campaign routing. Confirms the describe/rate axis is the critical path and bounds how much any distortion win can be worth. | **LESSON-ONLY** (frozen-space arithmetic). |
| **4** | **PR86→PR130's 0.1015 came ~half from ONE pose-carrier redesign, and NOT from better partition entropy coding** (bpp got 2.6% worse). | §4.3, both ledgers measured | **c1 composition** ranking: at this operating point a carrier-shaped redesign outranks another entropy-coder pass. Directly relevant to how we sequence c1. | **LESSON-ONLY** (directional). |
| **5** | **Integer-lattice entropy inference gives symbol-exact cross-device decode; float RGB stays outside the exactness claim.** Two-generation evidence: PR86 pinned ops to CPU FP32 (slow); PR130 replaced them with an integer lattice (exact *and* fast). | PR86 `inflate.py:412-417` comment; PR130 `hpac_integer.py`, `portability` block | **E4 export** + deterministic-reproducibility spine. If we ship arithmetic-coded streams, the proven portability contract is: integer-exact entropy path + symbol hashes, float renderer explicitly outside the claim. | **RACE-AS-CANDIDATE** (engineering pattern). |
| **6** | **~168 KB is the real public price of the exact partition** (tokens 116,980 + renderer + prior), **not ~114 KB**. | §2.1, code-verified | **c1 / describe-line** rate benchmark. **Supersedes** the 2026-07-25 memo's lesson-#3 figure. | **LESSON-ONLY** (benchmark). |
| **7** | **Single-member archive grammar: 100 B overhead vs 506 B for 5 members**, all `stored`. | measured on both archives | **E4 export**. Confirms L20 with a number. | **N-A** — already our doctrine. |
| **8** | **Fail-closed byte-exact reproduction gate at both ends** (source sha *and* output sha), with honestly narrowed scope (`full_training_from_original_video_included: false`). | `compress.sh:152-198`, `verification.json.compression_reproduction` | **E4 export** receiver-closure hygiene. | **RACE-AS-CANDIDATE** (pattern; ours is comparable — adopt the *narrowed-scope declaration* idea). |
| **9** | **Temporal diff-factorization of a partition stream is 3.5× WORSE than direct conditional coding** — `p(class-diff\|prev)` 0.02786 bpp vs `p(class\|ctx,prev)` 0.00797. Abandoned. | PR86 writeup Table 7 [external, VEHICLE-SCOPED] | **coder stack** / grammar productions: temporal structure should enter as **conditioning** or structural events, never as coded label-diffs. | **LESSON-ONLY** (negative, directional). |
| **10** | **Patch-group causal factorization makes an AR prior fit the budget**: 94 sequential steps/frame vs 196,608 (~2000×). | PR86 §HPAC; PR130 `HPAC_PATCH=64, HPAC_DELTA=2` | **coder stack**: if we ever race an AR prior, this is the feasibility trick that makes it admissible under the 30-min budget. | **RACE-AS-CANDIDATE**. |
| **11** | **RF-7 boundary-placement claim**: SegNet's boundaries are decided by a 7-input-pixel window (stem + Stage-0); 3 conv layers suffice to place them. | PR86 writeup §7.2 derivation + working ship | **c1 / describe-line**: if placement capacity is this cheap, *description* bytes dominate — strengthens the describe-vocabulary priority. Complementary to our rank-4-head + ERF r50~85px measurements. | **LESSON-ONLY, UNVERIFIED** — FROZEN-SPACE *candidate*. Verify on our apparatus before it becomes load-bearing. |
| **12** | **`LINEAGE_AND_CITATIONS.md` as a disclosure format** — lineage map, dated anchors, verifiable merge-base, inherited-vs-original table, explicit narrowing clause, self-limiting prior-art citation. | the file itself | **our NO-FAKE #7 disclosure format** for OSS release. | **RACE-AS-CANDIDATE** (format). |
| **13** | **Near-zero distortion is cheap to FIND at unconstrained rate** (d_seg 0.00, pose MSE <1e-9, 2h/T4). | PR86 writeup Table 1 [external claim] | context for A3/c1: the binding constraint is **realization at rate**, exactly where our crux sits. | **N-A** — our exact lattice solve already supersedes this (d_seg 0.0 *constructively*, not by descent). |

---

## 8. HONESTY BLOCK

- Every PR86/PR130 score above is **[external]**. PR86 0.273636 and PR130 0.172141 are
  github-actions-bot [contest-CUDA] rows recomputed from components. **0.169848 is a PROJECTION, not a
  measured row.** No [contest-CPU] row exists for PR130.
- **Two of our own memos are corrected here** (§2.1 PR130 ledger; §2.2 PR86 member sizes). Per
  APPEND-ONLY discipline the originals are not mutated; this memo supersedes those specific figures.
- The RF-7 claim (lesson 11) is **their derivation**, tagged FROZEN-SPACE *candidate* — not consumed
  as fact.
- Our v10-line receipts are **6–8 days old** (memories dated 2026-07-18/19) and carry staleness
  warnings. They are cited here as of-date measurements; re-verify at consumption before any launch
  depends on them.
- The leg attribution in §2.1 (~168 KB seg / 23 KB pose) is **DERIVED**, not measured: xz compresses
  the three model objects jointly and the split assumes the already-entropy-coded CPR1 carrier is
  ~incompressible.
- **No vehicle, carrier, weight, basis, coefficient, or archive byte from either PR is proposed for
  adoption.** Lessons classed `RACE-AS-CANDIDATE` must enter through a real race against our own
  alternatives, never by citation.
- **Pointer delta: ZERO.** `effective_frontier` remains 0.172 (PR130, official display); our
  submittable local frontier is unmoved. This intake is apparatus. It did not lower the exact score.

---

## 9. TRIALITY

- **DAG** — feed: `FEED-pr86-pr130-fullstack-intake-20260728` — the corrected ~168 KB partition
  benchmark and the 73.9%-rate decomposition both belong on the describe-line rate axis.
- **DSL** — no lever landed. Lessons 2, 5, 8, 10, 12 are `RACE-AS-CANDIDATE`; each must arrive as a
  `Lever` factory *if and when* raced, never as a hand-added flag.
- **Equations** — no new equation. §6 is arithmetic on the existing score law and cross-checks the
  #613 caps (157,800 B here vs 154,524 B banked — agreeing derivations at slightly different
  distortion assumptions).
- **Tasks/ledgers** — consumers named: **A3** (lesson 1), **E4 export** (5, 7, 8), **c1 composition**
  (4, 6, 11), **coder stack** (2, 9, 10), **#613 box arithmetic** (3, 6, §6).
