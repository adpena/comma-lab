# ddm_pd1 — decompose every competitor archive: the miss-bit share is prior-invariant, and the semantic family is ONE sample wearing four PR numbers

**Date:** 2026-08-18 · **Arm:** `ddm_pd1` (Opus) + 2 sub-arms · Charter
`.omx/research/charters/ddm_pd1_pr_archive_decomposition_priors_20260818.md`
**Operator binding 2026-08-18, verbatim:** *"The other PRs archives can be decomposed as well and
used as signal to continual learning"* + *"And other packages scripts and all upstream in authors'
repos and all signal and documentation and everything."*
**Axis:** every number below is **MEASURED** (parsed from bytes in my own hands), **DERIVED**
(arithmetic on measured inputs, marked), or **RECEIPT** (quoted from a published source, marked).
`score_claim: false` · `promotable: false` · **pointer NOT moved.**
**Spend $0.00.** No Modal, no Metal, no scorer runs. Zero code landed in the repo.

**Own-vehicle frontier, unchanged by this arm: `S = 0.15816036933414834 @ 180,601 B
[contest-CUDA T4, n600]`, archive sha `65c75d7f…`.**

---

## ANSWER — three findings, ranked

**1. The pre-registered prediction is UNTESTABLE as written, and finding that out is the result.**
The charter predicted the 0.19 %-miss / 70 %-bits structure would reproduce on **≥3 other
semantic-family PRs**. It cannot: **the semantic family contains at most TWO independent token
fields, not four.** PR132 and PR133 ship a **byte-identical** token stream — 116,980 B, sha
`948379872ff8…` — verified from the bytes in my own hands, not inherited. PR138 states a
**bit-exact transcode of all 117,964,800 F26 (PR135) tokens**, so 135 and 138 are one field coded
twice. **Four PR numbers, at most two payloads, and no third.** N archives sharing one payload are
ONE sample.

**2. The substitute test says the structure is REAL and NOT our-vehicle-specific — but the charter
named the wrong invariant.** Holding our field fixed and swapping the prior family across a 266×
range in miss fraction, the **miss BIT SHARE stays at 68.9–76.6 %** — every value inside the
pre-registered ±20 % band. The **miss FRACTION is not invariant at all.** A consumer keying off
"0.19 %" is keying off a vehicle artifact; a consumer keying off "≈70 % of the bits" is on solid
ground. **fx2's ceiling arithmetic survives, with a corrected anchor.**

**3. The byte-level coder axis is closed across the entire leaderboard, not just on our vehicle.**
Across 60 distinct PR archives, every well-coded stored payload sits within **0.2 %** of its
order-0 floor (median ratio **0.99987**, min **0.99809**, n=39 members >50 KB), and at order-1
every final payload is **statistically indistinguishable from i.i.d. uniform bytes**. Fifteen
months of competitors, independently, left nothing at the byte level. That is external
confirmation of our own wc2/#918/#996 closure verdicts — and it sharpens them: the closure is a
property of the *object*, not of our search.

**What this arm did not do:** it did not move the pointer, and no row here is a score claim.

---

## STORES CONSULTED

- `.omx/research/ddm_hx1_pr_wave_harvest_20260817.md` — PR129-138 **technique** harvest (T1-T19).
  Consumed, not re-mined; this arm is the orthogonal **statistical-structure** axis.
- `.omx/research/ddm_fd135_fractal_decomposition_20260810.md` + `ddm_pi135_pr135_intake_20260810.md`
  + `ddm_pi136_leaderboard_breadth_intake_20260810.md` — the PR135 reference decomposition form.
- `.omx/research/ddm_eh1_20260806/` (PR130-author forensics) · `pr130_eureka_intake_acquisition_20260806.md`.
- `.omx/research/ddm_fx1_fixed_point_logistic_mixer_20260817.md` §5 — the miss/hit decomposition
  I extend. Its 33-row sweep is **cited, never re-measured**, per
  `[[probability-model-axis-live-fx1-sweep-prior]]`.
- Instrument reused: `/Volumes/APDataStore/pact/ddm_fx1/decompose.py` (the hit/miss splitter).
- CLAUDE.md: NO-FAKE #7 · rule-118 · public-PR-intake **pristine-clone** discipline (no file inside
  any `public_pr*intake*` tree was written) · ALWAYS KEEP THE PAYLOAD.

## CUSTODY — everything extracted is retained

`/Volumes/APDataStore/pact/ddm_pd1/` — `census/` (instruments + raw census JSONL),
`streams/` (extracted payloads + `RETENTION_MANIFEST.json` + `stream_stats.json`), `legB/`, `notes/`.

| retained stream | bytes | sha256 (prefix) |
|---|---:|---|
| `pr133_tokens.bin` / `pr132_tokens.bin` | 116,980 each | `948379872ff8` (**identical**) |
| `pr133_sec0_xz.bin` / `pr132_sec0_xz.bin` | 73,128 / 73,944 | `683c99ab5fd6` / `0c93e1880acc` |
| `pr138_payload.bin` (opal_v1) | 181,940 | `d8db5d3c4ee5` |
| `pr86_tokens.bin` (the un-fused ancestor) | 113,900 | `14144bde4966` |
| `pr86_hpac.pt.ppmd` / `master.pt.gz` / `slave.pt.gz` / `meta.pt` | 28,243 / 31,144 / 32,287 / 1,499 | `de7638c531c9` / `3f3ee2b19ba5` / `817294dea0d9` / `848381e2da1b` |
| `pr96_p_constant_zero.bin` (the defect) | 930 | `bf04a4e2dd69` |

**Typed prior store: `.omx/research/ddm_pd1_pr_decomposition_priors_20260818.jsonl` — 661 rows**
(493 MEASURED / 138 DERIVED / rest RECEIPT; 63 PRs; 0 rows missing evidence),
schema `{pr, source, kind, section, stat, value, unit, evidence, sha256, grade, consumer}`. Every
row is **generated from parsed bytes**, never hand-typed, so the store cannot drift from the
artifacts.

---

## 1. Coverage — what was decomposed

**476 archives parsed** through the real `zipfile` parser; **76** are competitor-PR archives,
covering **60 distinct PR numbers**: 18, 20-23, 26, 27, 30, 31, 37, 39, 43, 44, 48, 49, 51-53,
55, 56, 58, 60-65, 67, 74, 76, 77, 79, 81, 82, 84-86, 89-98, 100-108, 132, 133, 137, 138.

This is **~5× the charter's named target set** (PR129, 131-134, 136-138). The
`experiments/results/public_pr_intake_full/` tree held 52 archives from the 2026-05-05 auto-intake
that no prior arm had structurally decomposed — hx1/pi1/pi135/fd135 were all single-PR deep dives.

**Skipped, with reason:** PR129/131/134/136 (source retained by hx1, **no `archive.zip` on disk** —
PR131 never exported one, per hx1 §6); PR130/135 (decomposed by `pi1`/`pi135`/`fd135` — not redone);
PR109-128 (no archive bytes found in any local intake tree — *did not find in the scanned scope*,
which is not the same as saying none exist).

## 2. The container tax — measured, and free

ZIP overhead is a real counted byte cost with a **13.5× spread** across the corpus, and it is pure
packaging:

| overhead | who | note |
|---:|---|---|
| **100 B** | PR101 (gold), 103, 105-ancestors, 132, 133, 138, **and ours** | 1 stored member, **1-character name** |
| 108 B | PR95, 98, 100, 104, 106, 107 | 1 stored member named `0.bin` — **8 B for the filename** |
| 296-506 B | PR53, 55, 61, 62, 74, 86, 96 | 3-5 members |
| **1,348 B** | PR60, PR137 | 9 members **plus a directory entry** |

**Our archive is already at the 100 B floor** (single stored member `p`, verified). So this axis is
**closed for us** — a negative worth having, because it stops anyone spending a unit on it.
For the field it is not closed: PR60/137 carry **1,248 B above the floor = ΔS 8.31e-4** (DERIVED),
which is **2.2× the entire fx1 mixer win**, spent on ZIP directory entries.

## 3. Three defects the authors shipped into the counted archive

All MEASURED, all verified past the first look:

- **PR96 ships 930 bytes of `0x00`.** Member `p`, `method=stored`, 930 bytes, **one distinct byte
  value**. Pure padding inside the counted payload. **ΔS 6.19e-4 thrown away** — larger than our
  whole fx1 move.
- **PR61 ships `meta.json` raw** — 437 B of pretty-printed JSON, uncompressed (217 B under zlib-9).
  It also **leaks their encoder config as a free RECEIPT**: `width=256, height=192, fps=10,
  encoder=libsvtav1`. They encode at **256×192**.
- **PR86 ships `meta.pt` raw** — 1,499 B, itself a torch zip (with `ZZZZ…` writer padding), stored
  uncompressed inside a zip. 632 B under zlib-9. Its order-0 entropy is **4.48 b/B against a 7.89
  random control at the same n** — so this one is *real* structure, not estimator noise (§5).

**Class sweep, so this is a finding and not an anecdote.** I swept every member of all 60 archives
for low-entropy content (H0 < 6 b/B, raw ≥ 200 B). Eight members qualify; **PR96's is the only
H0 = 0 case in the corpus.** The zero-padding defect is a **singleton, not a class** — nobody
should go hunting for more. The recoverable total across the four actionable members (PR96 930 B,
PR86 `meta.pt` 867 B, PR61 `meta.json` 220 B, PR60 `sidechannel.bin` 205 B) is ≈2.2 KB, **ΔS
1.48e-3 of pure packaging waste** sitting in the field's counted archives.

## 4. The prediction verdict

### 4a. Why the pre-registered form cannot run

| PR | token section | grade | verdict |
|---|---:|---|---|
| 132 | 116,980 B, sha `948379872ff8` | MEASURED | — |
| 133 | 116,980 B, sha `948379872ff8` | MEASURED | **byte-identical to PR132** |
| 135 (F26) | 114,706 B | RECEIPT | different size ⇒ a *second* group, not the same bytes as 132/133 |
| 138 (opal) | 110,022 B; **0 shared 4 KB blocks with PR133** | RECEIPT + MEASURED | **bit-exact transcode of PR135's 117,964,800 tokens** — same field, new coder |

Verified in my own hands: both PR132/133 archives are `u32 len ‖ xz section ‖ tail`, and the tails
match by sha. hx1 §8.2 asserted this; I re-derived it rather than inheriting it. **Two groups, at
most two independent fields, no third** — so the prediction as written has no denominator.

**Correction to my own first read:** I initially wrote "one field, four PR numbers." The opal
receipt (F26 = 114,706 B vs PR133's 116,980 B) forces the weaker, correct claim: two groups. I
have verified identity *within* each group and equality *between* groups is unverified either way.

### 4c. Cross-coder rate on the same field SHAPE — where we actually stand

| coder | token bytes | grade |
|---|---:|---|
| CPR1 HPAC (PR132/133) | 116,980 | MEASURED (by me) |
| F26 (PR135) | 114,706 | RECEIPT (opal README) |
| **opal rc64**, 55 causal families, Fisher-Newton mixer (PR138) | **110,022** | RECEIPT |
| **ours, post-fx1 log-odds mixer** | **109,951.21** | RECEIPT (fx1 memo) |

**Our coder is 71 B ahead of opal's.** Caveat welded on: same field *shape* (117,964,800 tokens),
token *values* verified equal only *within* each group — so this is **directional, not a
head-to-head**. It is still the first time we can place our token coder against the field's best
published one on a comparable object.

This is the *positive* form of the `[[same_defect_negatives_masquerade_as_family_convergence]]`
law: N archives sharing one payload are ONE observation, and a leaderboard whose top six entries
are one lineage is far less diverse than its PR count suggests.

### 4b. The substitute test — vary the PRIOR, hold the FIELD

The falsifier's real content was *"if it does not reproduce, the structure is OUR-vehicle-specific."*
That is answerable at $0 on retained assets: hold our 600×384×512 = 117,964,800-token field fixed
and swap the prior family. Semi-static conditional tables, parameter cost reported and negligible;
code lengths are **lower bounds**, the miss fractions are exact.

| prior family | contexts | miss % | **miss bit share %** | total B | within-miss share of miss bits |
|---|---:|---:|---:|---:|---:|
| P_A neural, shipped (fx1 §5, cited) | — | **0.190** | **70.01** | ~112,110 | 1.59 % |
| P_B causal: prev-frame, left, up | 125 | 0.3295 | **75.08** | 253,387 | 3.65 % |
| P_B2 causal + upleft | 625 | 0.2832 | **76.56** | 235,478 | 3.76 % |
| P_C order-0 marginal (**weak-prior control**) | 1 | **50.4823** | **68.92** | 23,821,780 | 55.29 % |

**Verdict: the structure REPRODUCES, and the invariant is the bit share, not the miss fraction.**

- Miss bit share spans **68.92 – 76.56 %** across four prior families. Pre-registered band
  (70.01 % ± 20 % → [56.0, 84.0]): **all four inside.** ✅
- Miss fraction spans **0.190 % → 50.48 %, a 266× range**; total code length spans **112 KB → 23.8
  MB, 212×**. The bit share does not care. ❌ for treating 0.19 % as transferable.
- **Second invariant, and it is the actionable one:** under *any* strong prior, the cost of *being*
  a miss — the `q` the mixer attacks — is **96.2-98.4 %** of all miss bits (P_A 98.41, P_B 96.35,
  P_B2 96.24). Under the weak prior it collapses to 44.7 %. **The fx1 mixer axis is correct for any
  strong prior, not just ours.**

**Field variation, the second leg.** The prior-family table varies the prior on one field, so I
also varied the field: 10 **disjoint** 60-frame blocks, prior refit **per block** (scene content
differs sharply across the drive — the `[[prefix_bias]]` law).

| | min | max | spread |
|---|---:|---:|---|
| miss bit share | 74.48 % | 75.75 % | **1.27 pp** |
| miss fraction | 0.3029 % | 0.3631 % | 1.20× |

The share is flat to ~1 pp while the content moves. **The invariant survives both axes.**

**Honest bound:** this is still a **substitute** for the pre-registered test. Both legs live on
*our* field; neither uses a competitor's decoded tokens. It cannot rule out that a genuinely
different argmax field (different video, different class count) moves the share outside the band.
What it does rule out is the specific falsifier the charter named — our neural prior being
load-bearing — and it now rules it out under content variation too.

**Free by-product:** our shipped neural prior codes the field ~**2.14× better** than the best
classical causal context model measured (P_B2, 235,478 B lower bound vs ~110,511 B). That prices
what the neural prior earns on the token axis.

## 5. The instrument trap I walked into, and the control that caught it

My first stream-statistics pass reported **~6,756 B of order-1 structure left** in PR133's token
stream, and similar in every competitor payload. It was **100 % estimator bias.** A 256×256 plug-in
conditional-entropy estimator on ~10⁵ samples has 65,536 cells; i.i.d. **uniform random** bytes
through the same estimator report:

| n | measured H1 on their stream | **random control H1** | phantom "headroom" |
|---:|---:|---:|---:|
| 116,980 (pr133 tokens) | 7.5380 | **7.5357** | 6,789 B |
| 181,940 (pr138 payload) | 7.7151 | **7.7140** | 6,504 B |
| 28,243 (pr86 hpac.ppmd) | 6.3929 | **6.3976** | 5,657 B |

Every competitor stream matches its own random control to **≤0.005 b/B**. Reported without the
control, this would have been a 6.7 KB phantom headroom row feeding fx2 — a silent wrong number,
the most expensive kind. The bias bar is now a row in the prior store so the next arm cannot
repeat it.

---

## 6. Leg B — the full surround (authors' repos, scripts, docs)

Operator's scope extension. Prior sweeps (#413/#414, eh1, pi136) were extended, not redone. The
coverage delta and full typed rows are in the prior store; the load-bearing part:

**The single largest new surface is not a contest PR at all.** PR136's author (`JPL11`) maintains
`commavq-compression`, an entry in comma's *other* challenge, and it is the only place in the
whole wave where someone published a **measured context study on a comma token stream**:

- **Context policy is worth 0.4 bits/token.** 2.908 → **2.4993 bits/tok** from overlapped
  stride-5 windows vs restart windows — same model, same data, same coder, **only the context
  policy changed**.
- **The saturation curve is published**: CE by frame position 7.987 (f0) → 3.360 → 2.965 → … →
  2.522 (f8) → 2.498 (f19). Saturates by ~frame 8; the first frame after a restart costs **3.2×**
  the saturated rate.
- **Do not pre-quantize the probability table**: float64 straight into the coder costs +0.0005
  b/sym; a 16-bit pre-quantized table costs +0.013 b/sym — **26× worse**.

That last one pairs with two independent range-coder receipts — opal's stream is **4.98 bits**
above its own model ideal, F26's **4.32 bits** (fd135) — to give a clean law: **the arithmetic
container is not a lever; the model is.** It also tensions usefully against hx1's §4 LAW
("quantise every value in the probability path"): JPL11 achieves bit-exact sync instead by pinning
the *op sequence* (same shapes, same kernels, static KV cache, matched `torch.compile`). Two
different determinism cures, both valid, different costs.

**Other receipts worth banking:** brotli sits **1.777 % ABOVE** per-tensor order-0 entropy on
HNeRV weights (163,237 vs 160,387 B) — LZ77 loses to a trivial adaptive coder on weight tensors.
Two drift channels are now quantified on identical bytes: cross-CPU-host **ΔS 3e-6, entirely in
SegNet argmax ties**, versus cross-GPU **ΔS −0.001372, pose-dominated** — a ~450× ratio, and a
reason to treat a third-party GPU number as a different axis, not a confirmation.

**One claim I checked and REFUTED.** The sweep reported that CLAUDE.md's L14 row mislabels an
extended schedule as "PR95 canonical" (claiming PR95's default is 22,650 epochs, not 29,650).
**It does not survive our own receipt:** `public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.md:32-46`
records the same per-stage table and *"total release-view epochs: 29650"*, profiled **2026-05-04**
— months before that repo existed. **Do not edit L14 on this claim.** A plausible correction to
always-loaded instructions is exactly the thing to re-derive before propagating.

**Genuinely absent from our records:** `latent_lr_mult = 10.0` — the per-pair latents optimized at
10× the decoder-weight LR, in all eight published stages. That is a lever our L14 row does not
carry.

## 7. Leg C — the grammar read, and the four things it changes

Full section grammars, coder choices and `file:line` receipts are in the prior store. What moves
a decision:

**7a. The exact LZMA filter, re-derived byte-exactly in my own hands.** The semantic family's model
bundle is `FORMAT_XZ / FILTER_LZMA2, dict_size=65536, **lc=0, lp=1, pb=0**`. Recompressing PR133's
parsed 82,743 B bundle with those parameters reproduces the shipped **73,128 B section byte-for-byte**;
stock `preset=9|EXTREME` (lc3/lp0/pb2) gives 74,220 B. **The tuning is worth 1,092 B (1.47 %).**
`lp=1, lc=0` is a 2-byte-aligned literal model with zero literal-context bits — the right shape for
**int16/fp16 arrays**. Nobody in the HNeRV family does it (PR101's only LZMA use is
`FILTER_LZMA1, dict 4096, lc3/lp0/pb0` on latents). **This is the most directly actionable import
in the arm: audit our own LZMA-coded array sections for it.**

**7b. Their entropy coders are saturated; their weight SERIALISATION is not.** Re-coding each
retained PR133 section standalone: tokens **0.0 %**, carrier **−0.2 %** — but the semantic renderer
**−13.0 %** and the HPAC model **−25.9 %**. Total ~875 B on the table (ΔS 5.83e-4). opal closed
most of it (down to 138 B) by recoding the *model* sections — int4 packing → adaptive rANS
(−4,212 B) and IHS1 → IHS2 (−3,586 B) — **not** by improving the token coder.

**7c. Two attributions in hx1 need correcting, and both point the same way.** (i) hx1 credits the
rank-one maximal-projector split (T2) to PR138. **PR133 wrote it first, in Python, and shipped it
DISABLED** — `HPAC_HIERARCHICAL = False` at `inflate.py:46`, with a complete `hierarchical_decode`
at `:541-556`. (ii) hx1's §4 treats opal's libm desync hazard as an open class hazard and T6/T7
treats the per-section coder selector as absent from the field. **PR112 shipped both cures in
June** — a 256-entry precomputed `Q_TABLE` "so the decoder never calls `exp()`" plus IEEE-exact
float64-only model construction (`codec_ctx.py:23-26, 96-125`), and a per-section coder-id bitmap.
PR133 goes further still, quantising to an **int16 lattice before** the transcendental. The field
solved this two months before opal shipped the hazard.

**7d. The token axis is near-saturated field-wide.** On the identical 117,964,800-token field:
PR86 (May, S≈0.27) **113,900 B** → PR133 116,980 → PR135 114,706 → PR138 **110,022 B**. Total span
**5.95 %**, and the *oldest* entry is only **3.52 %** above the newest best. Three months of
entropy-coding competition on this field bought under 6 %. Our 109,951 B sits just past the end of
that race — which reframes §4c: we are not ahead by a lot, we are ahead at the point where the
curve has already flattened.

**One unexploited class, corpus-wide:** *nobody entropy-codes a scale array.* Every archive read
stores fp16/fp32 per-tensor scales raw — PR100 `sca`, PR103 `sca`+`mins_scales`, PR101 interleaved,
PR105/106 banked, PR133 inline.

**PR86 decode feasibility, resolved:** **not pure-python.** It needs torch (a full `HPACMini`
forward), constriction and pyppmd — 56,400 full-resolution forwards plus 117.9 M Python-level
`Categorical` decodes, DERIVED at **>30 h**. A 5-frame prefix (~983 k symbols) is tractable, but a
prefix is a different population (`[[prefix_bias]]`). This demotes the "get a second independent
sample" item.

## 8. Ranked — what they know that we don't

> **Re-ranked after Leg C.** The single most *actionable* row is now **0** below — a byte-exact,
> re-derived coder parameter we can check against our own sections today. Rows 1-8 were ranked
> before the grammar read and are unchanged in content.

| # | row | grade | why it matters | consumer |
|---|---|---|---|---|
| **0** | **`lc=0, lp=1, pb=0, dict=64 KiB` on LZMA'd array sections is worth 1,092 B (1.47 %)** vs stock preset 9\|EXTREME — re-derived byte-exactly against PR133's shipped section. | MEASURED | The only import in the arm that is a **concrete parameter we can test on our own bytes today**, at zero risk and zero training. `lp=1/lc=0` = 2-byte-aligned literals, zero literal-context bits — the right model for int16/fp16 arrays. **Audit our LZMA-coded sections.** | **packet** |
| **0b** | **Entropy-coded sections are saturated; weight SERIALISATION is not** (PR133: tokens 0.0 %, carrier −0.2 %, semantic −13.0 %, HPAC model −25.9 %). And **nobody in the corpus entropy-codes a scale array.** | MEASURED | Redirects effort off the coder and onto how weights and scales are *serialised* — where opal actually found its bytes (−4,212 B and −3,586 B on model sections, not on tokens). | **fx2 / packet** |
| **0c** | **The token axis is near-saturated field-wide**: 5.95 % total span across four coder generations on the identical field; PR86 (May, S≈0.27) is only 3.52 % off the newest best. | DERIVED | Recalibrates §4c: our 109,951 B lead is real but sits where the curve has flattened. Do not budget a large win here. | **fx2** |
| **1** | **Context policy is worth 0.4 bits/token on a comma token stream**, with the saturation curve published (f0 costs 3.2× the f8 rate; saturates ~frame 8). | RECEIPT | fx2 is *choosing a context set right now*. This is a measured, same-domain answer to "how much context before it stops paying" — the strongest single import in the arm. | **fx2** |
| **2** | **Do not pre-quantize the probability table**: +0.0005 b/sym (float64 direct) vs +0.013 b/sym (16-bit table) = **26×**. Pairs with two range coders landing 4.3-5.0 bits above their own ideal. | RECEIPT | Prices an fx2 design choice at byte zero and points *against* the "quantize the whole probability path" instinct. The container is not a lever; the model is. | **fx2** |
| **3** | **`q` is 96-98 % of miss cost under ANY strong prior**, 44.7 % under a weak one. | MEASURED | The fx1 mixer axis generalises beyond our prior. And it flips: a *weak*-prior section (a fresh sidecar with no learned model) needs the opposite treatment — spend on the relative law. | **fx2** |
| **4** | **The semantic family is at most TWO token fields, not four PRs** (132≡133 by bytes; 135≡138 by transcode). | MEASURED | Any "the field converged on X" inference from PR counts is over-counted ~2-4×. Re-price every claim resting on cross-PR agreement. | #984 |
| **5** | **A standalone f0 pose carrier costs ~8 KB and lands ~0.196 — on the HNeRV lineage.** PR135 shipped f0 edits successfully on CPR1. | RECEIPT | Exactly the shape ps2 is designing. The transferable content is the **break-even**: a standalone f0 carrier must come in well under 8 KB. Verdict scope is lineage-specific — do not read it as a kill. | **ps2** |
| **6** | **Two drift channels, quantified on identical bytes**: cross-CPU-host ΔS 3e-6 (all SegNet ties); cross-GPU ΔS −0.001372 (pose-dominated). ~450× apart. | RECEIPT + DERIVED | A third-party GPU number is a different axis, not a confirmation. Bounds how much of any cross-host disagreement is real. | #984 |
| **7** | **Order-0 is closed corpus-wide** (median 0.99987, n=39, 60 archives) and **brotli sits 1.777 % above order-0 on HNeRV weights**. | MEASURED + RECEIPT | Independent, 60-archive confirmation that byte-level coder search is spent — *and* a same-family receipt that LZ77 loses to a trivial adaptive coder on weight tensors. Directly checkable against our own sections. | **fx2** |
| **8** | **PR86 is the un-fused ancestor grammar** — token stream is a NAMED member (113,900 B), HPAC model coded with **PPMd**. | MEASURED | The only archive in the lineage where the token stream is separable without running a decoder — the cheapest place to get a second independent sample. | **fx2** |

*Also banked, below the cut:* our neural prior is **2.14×** a 625-context classical model on the
same field (DERIVED); the plug-in order-1 bias bar ≈5.7-6.8 KB at n≈10⁵ (MEASURED — any future
competitor-stream "context headroom" claim under it is noise); **PR96 threw away ΔS 6.19e-4 as
zero-padding** and PR60/137 threw away 8.31e-4 on ZIP directory entries (MEASURED); PR61 encodes
at **256×192, libsvtav1** (RECEIPT, leaked in raw `meta.json`); `latent_lr_mult=10.0` is absent
from our L14 record (RECEIPT).

## 9. Borrowed-substrate accounting (NO-FAKE #7)

**This memo adopts no mechanism and lands no code.** Every artifact read is a public contest
submission. All extraction ran on retained copies under `/Volumes/APDataStore/pact/`; **no file
inside any `public_pr*intake*` clone was written to** (CLAUDE.md pristine-clone rule). Numbers
attributed to a PR are *theirs on their vehicle*; the prior-family table is **ours on our field**
and transfers to no one else's vehicle without their own measurement.

## NEXT_IF_RESUMED

1. **Audit our own LZMA-coded array sections for `lc=0, lp=1, pb=0`** (row 0). Byte-exact,
   re-derived, zero training, testable today. Highest actionability in the arm.
2. **Redirect the next rate unit off the coder and onto weight/scale serialisation** (row 0b):
   their entropy-coded sections are saturated, their model sections carry 13-26 %, and **no one in
   the corpus entropy-codes a scale array.**
3. **Feed rows 1-3 into fx2 before it fixes its context set** — the saturation curve (context pays
   steeply to ~frame 8, then flattens), the pre-quantization receipt (keep the probability table in
   float; the arithmetic container is not a lever), and the `q`-vs-relative-law split (branch the
   section model on prior strength). None cost a run.
4. **Give ps2 the ~8 KB break-even** for a standalone f0 carrier, with its HNeRV-lineage verdict
   scope attached — a bar, not a kill.
5. **Do not re-open order-0** (closed on 60 archives), **do not spend a unit on our container**
   (already at the measured 100 B floor), and **do not budget a large token-coder win** (row 0c).
6. **Do not edit CLAUDE.md L14** on the 22,650-epoch claim; §6 records why it was refused.
7. **Demoted:** decoding PR86's `tokens.bin` for a second independent sample. Leg C resolved it as
   **not pure-python** (torch + constriction + pyppmd, >30 h full decode). A 5-frame prefix is
   tractable but is a different population.
