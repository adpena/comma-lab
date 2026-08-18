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
semantic-family PRs**. It cannot: **the semantic family does not contain three independent token
streams.** PR132 and PR133 ship a **byte-identical** token stream — 116,980 B, sha `948379872ff8…`
— which I verified from the bytes rather than inheriting the claim. PR135 is their parent and
PR138 re-codes the same field. **Four PR numbers, one payload.** N archives sharing one payload
are ONE sample.

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

**Typed prior store: `.omx/research/ddm_pd1_pr_decomposition_priors_20260818.jsonl` — 631 rows**,
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

## 4. The prediction verdict

### 4a. Why the pre-registered form cannot run

| PR | token stream | verdict |
|---|---|---|
| 132 | 116,980 B, sha `948379872ff8` | — |
| 133 | 116,980 B, sha `948379872ff8` | **byte-identical to PR132** |
| 135 | parent of both | same lineage |
| 138 | re-coded; **0 shared aligned 4 KB blocks with PR133** | same *field*, new coder |

Verified in my own hands: both archives are `u32 len ‖ xz section ‖ tail`, and the tails match by
sha. hx1 §8.2 asserted this; I did not inherit it, I re-derived it — and the consequence is
stronger than hx1 drew. **The semantic family is one token field wearing four PR numbers.** There
are not three independent samples, so the prediction as written has no denominator.

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

**Honest bound:** this is a **substitute** for the pre-registered test, not the test itself. It
varies the prior on one field; it does not vary the field. It cannot rule out that a *different*
argmax field (a different video, a different class count) moves the share outside the band. What it
does rule out is the specific falsifier the charter named — our neural prior being load-bearing.

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

## 6. Ranked — what they know that we don't

| # | row | grade | why it matters | consumer |
|---|---|---|---|---|
| **1** | **The top-6 leaderboard is ONE token field.** PR132≡PR133 by bytes; 135 parent; 138 recodes. | MEASURED | Every "the field converged on X" inference from PR counts is over-counted ~4×. Re-price any claim resting on cross-PR agreement. | #984 |
| **2** | **`q` is 96-98 % of miss cost under ANY strong prior**, 44.7 % under a weak one. | MEASURED | The mixer axis generalises; a *weak*-prior section (a fresh sidecar with no learned model) needs the opposite treatment — spend on the relative law. | fx2 |
| **3** | **PR86 is the un-fused ancestor grammar** — token stream is a NAMED member (113,900 B) and the HPAC model is coded with **PPMd**. | MEASURED | The only archive in the lineage where the token stream is separable without running a decoder. It is the cheapest place to test any token-model idea against a competitor object. | fx2 |
| **4** | **Order-0 is closed corpus-wide** (median 0.99987, n=39). | MEASURED | Independent, 60-archive confirmation that byte-level coder search is spent. Stops re-opening it. | fx2 |
| **5** | **PR96 threw away ΔS 6.19e-4 as zero-padding**; PR60/137 threw away 8.31e-4 as ZIP directory entries. | MEASURED | Container/packaging hygiene is worth more than a coder round to parts of the field. Ours is already at the floor — a closed axis, confirmed. | packet |
| **6** | **PR61 encodes at 256×192, libsvtav1** (leaked in raw `meta.json`). | RECEIPT | A concrete classical-family operating point, free. | #984 |
| **7** | **The plug-in order-1 bias bar** (≈5.7-6.8 KB at n≈10⁵). | MEASURED | Any future competitor-stream "context headroom" claim under this bar is noise. | fx2 |
| **8** | **Our neural prior is 2.14× a 625-context classical model** on the same field. | DERIVED | Prices the neural prior's contribution to token rate; bounds what a classical fallback would cost. | fx2 |

## 7. Borrowed-substrate accounting (NO-FAKE #7)

**This memo adopts no mechanism and lands no code.** Every artifact read is a public contest
submission. All extraction ran on retained copies under `/Volumes/APDataStore/pact/`; **no file
inside any `public_pr*intake*` clone was written to** (CLAUDE.md pristine-clone rule). Numbers
attributed to a PR are *theirs on their vehicle*; the prior-family table is **ours on our field**
and transfers to no one else's vehicle without their own measurement.

## NEXT_IF_RESUMED

1. **Decode PR86's `tokens.bin` (113,900 B, retained, sha `14144bde4966`)** — the only competitor
   token stream separable without a full renderer forward. It would convert §4a's blocking fact
   into a real second independent sample and let the pre-registered test actually run.
2. **Feed row 2 into fx2's context discovery**: the `q`-vs-relative-law split flips with prior
   strength, so section-level model choice should branch on it rather than defaulting to the
   token-stream shape.
3. **Do not re-open order-0 anywhere in the packet** — row 4 closes it on 60 archives.
