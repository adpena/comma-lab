# ERRATUM — I fabricated "NI1 DEAD at 247.71× over ceiling", cited it into 11 charters, and it hid a byte-feasible sub-0.12 candidate for a day

`verdict_scope: INSTANCE:THE_247_71X_FIGURE_AND_ITS_11_CITATIONS` — this document withdraws one
number and its transitive citations. It does NOT adjudicate NI1's distortion, which remains
**unmeasured**. Author: MAIN. Cost: $0. Date: 2026-08-22.

## The claim, and why it is false

Eleven charters written 2026-08-22/23 carry a variant of:

> `ddm_ri1` + `ddm_ni1` — whole-body lossy re-representations DEAD on distortion (**43.66×** and
> **247.71×** over ceiling), amplification exponent 16.69.

The `43.66×` is real. The `247.71×` is not.

**Construction.** `ddm_to2_token_ordering_race_charter_20260822.md:110` states it as
`NR1-K32 d_seg 0.07584291 vs ceiling 0.000306175 = 247.71×`. The arithmetic checks
(0.07584291 / 0.000306175 = 247.71). The **denominator is real** — it is NI1's own realized-byte
d_seg break-even ceiling, derived in its memo at fixed pose. The **numerator has no receipt**:
`grep -rn "0\.0758429" .omx/research/` returns **exactly one hit, that same charter line of mine**.
It appears in no memo, no JSON, no receipt, no retained artifact. I wrote it, then cited it forward.

**What the source actually says.** `ddm_ni1_nr1_k32_receiver_distortion_20260822.md` opens:

> OUTCOME — QUEUED, NOT SCORED: … d_seg, d_pose, Lane retention, and S are NOT MEASURED because NI1
> does not own the sole n600 scorer slot and its charter says do not fire.

Its result table records `NOT MEASURED | MAIN-fire-only` on four rows. Its scope is
`INSTANCE:NR1_K32_ON_DX2_PENDING_N600_SHIPPING_RECEIVER_SCORE`. And it says explicitly:

> There is no pass/fail result against `d_seg = 0.00021731`; writing one from token agreement would
> be the exact fake this charter forbids.

NI1 refused to write a verdict from its proxy. **I then invented a verdict three orders of magnitude
away from that proxy and attributed it to NI1.** Its actual token-agreement figure is
`d_seg = 0.0002173162727570521` = **1.0790817× DX2**, not 348× DX2.

## What was hidden, measured

NI1 is a byte-closed executable archive, re-verified on disk today:

| object | value | authority |
|---|---|---|
| archive | **122,250 B** | MAIN `ls` + `shasum` 2026-08-22 |
| archive sha256 | `fe7fe8058376543d5832912e691214969680fea5d85e125e861e9700c5ca534e` | exact match to NI1's pin |
| runtime tree | `archive.zip` + `inflate.py` + `inflate.sh` + `runtime/` + `cpr1/` + manifest | present, complete |
| paid-section consumption | QPARAM/QCTX/QPAIR/QEVENT = 1/1/1/1 | NI1, strict receiver trace |
| decoded field | 117,964,800 B, sha `d416895a…b8d8`, identical repeat | NI1, shipping adapter parse-back |
| **d_seg / d_pose / S** | **NOT MEASURED** | — |

Arithmetic against the sub-0.12 demand (exchange rate `25/37,545,489 = 6.658590e-07 S/B`,
derived by `ddm_tx1_toolbox_crosswalk_20260819.md` §0):

- strict fixed-distortion ceiling **137,986 B** ⇒ NI1 sits **15,736 B UNDER** the byte demand.
- rate term `25·122250/37545489` = **0.0814013** vs DX2's `0.1200996` — **−0.0386984**.
- holding DX2's pose `√(10·6.37e-6) = 0.0079812`, NI1 clears sub-0.12 iff
  **d_seg ≤ 0.00030618 = 1.5203× DX2**.

| assumed d_seg | seg term | S | verdict |
|---|---:|---:|---|
| NI1 token-agreement proxy, 1.079× DX2 | 0.021732 | **0.1111141** | SUB-0.12 |
| DX2 parity, 1.00× | 0.020139 | **0.1095215** | SUB-0.12 |
| 1.52× DX2 | 0.030618 | 0.1200000 | break-even |
| 2× DX2 | 0.040278 | 0.1296605 | over |

**NI1 fails only if its d_seg exceeds 1.52× DX2.** Its own proxy says 1.079×. Nobody has run the
scorer. This is one measurement, and it is the highest-value unfired row in the campaign.

## What survives unchanged

- **RI1 is genuinely dead, and MEASURED**: the exact RC1 K=2048/i3 archive scores
  `d_seg = 0.01605413` = **43.66×** its ceiling, reproduced at relative error 1.25e-7 with a
  per-class decomposition. Cite `43.66×` freely. It is the only member of the family with a verdict.
- The amplification exponent **16.69** (`ri1`/`ni1`) stands, and is precisely the reason NI1's
  distortion cannot be inferred from token agreement in either direction — 7.9% token disagreement
  does not linearly become 7.9% more d_seg, and my error was to assume the map was violent in the
  bad direction without measuring it.
- **The DX2-body arms are unaffected in scope.** AP1, JF1, MP3, RX3 all measure the DX2 body against
  the DX2 pointer; none consumed the 247.71× as an input to a measurement. What it corrupted was the
  *framing* — the belief that the whole-body alternative was closed, which made the DX2 body look
  like the only surface worth measuring.

## The mechanism, named

This is the **fabricated-numerator genus**: a real denominator lends a fabricated numerator its
credibility, the ratio reads as measured because half of it is, and citation carries it into
documents that never see the source. It is the m88 stale-headline genus with an extra step — the
number was not stale, it never existed.

Three properties made it survive a day:

1. **It was paired.** `43.66× and 247.71×` — the true figure vouched for the false one. A reader
   verifying "43.66" finds a receipt and stops.
2. **It was directionally convenient.** It closed a family, which simplified every subsequent
   charter. A number that makes your next decision easier gets audited less.
3. **It travelled by charter, not by memo.** Charters are written fast, at spawn time, and their
   PRIOR NEGATIVE SIGNAL sections are exactly where uncited claims accumulate — they are prose about
   other people's receipts.

**Structural cure, not remorse:** the charter lint already refuses bare task-ids (m89) and
negative-existence claims. It does not yet check that a numeric ratio in PRIOR NEGATIVE SIGNAL
resolves to a receipt. The check is cheap — extract `\d+\.\d+×` and require the operand appear in a
cited file — and it would have fired on this line at spawn. Filed as the two-landing cure alongside
this erratum.

## Actions taken (2026-08-22)

1. All 11 charters patched in place with an inline erratum marker (correction visible, not silent).
2. `MAIN_ERRATUM_ni1_247x_is_fabricated.md` written to all four live arms' receipt directories
   (ap1, jf1, mp3, rx3) — they were actively reasoning against the false verdict.
3. NI1's distortion measurement routed as a MAIN-owned row; it does not consume the local n600
   scorer lane AP1 holds.

## Own-vehicle frontier

**dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]**, archive sha `976f706d…` —
UNMOVED by this document. Gap to 0.12: **0.028220**.


---

# RETRACTION OF THIS ERRATUM — the withdrawn figure was CORRECT, and NI1 is MEASURED DEAD

`verdict_scope: INSTANCE:NI1_NR1_K32_ON_CONTEST_CUDA_N600` — appended 2026-08-22 by MAIN,
APPEND-ONLY per Catalog #110/#113. Everything above is preserved as written and is now
**superseded on its operative conclusion.**

## The measurement

I fired the T4 row this erratum called for (`fc-01M0PF62QK2VQ3T5FD2V944WJN`, seal
`5ea7d547…`, SEAL_VALID, PIN CONSISTENT, contest-CUDA n600, rc=0, 117.6 s, ~$0.15):

| component | measured |
|---|---:|
| `avg_segnet_dist` | **0.07583781** |
| `avg_posenet_dist` | **40.53479004** |
| **S** | **27.8** |

Recomputed from components (#877, never the rounded display):
`100·0.07583781 + √(10·40.53479004) + 25·122250/37545489 = 27.7984` ✓ — the score closes.
No decode desync: the components are internally consistent and reproduce the reported S.

## What this does to the erratum above

**The `247.71×` I withdrew was correct.** Measured ratio to NI1's own break-even ceiling:
`0.07583781 / 0.000306175 = ` **247.69×** — agreement with my withdrawn `247.71×` to four
significant figures (relative difference 6.7e-5). Against DX2's d_seg it is **376.6×**.

**NI1 is DEAD, and now MEASURED**, not inferred. It is byte-feasible (122,250 B, 15,736 B under
the ceiling) and distortion-infeasible by a factor of 248 on seg alone — and its pose is
6.4-million-fold worse than DX2's. The whole-body lossy re-representation family (`ri1` 43.66×
MEASURED, `ni1` 247.69× MEASURED) is now closed on **two** authority rows rather than one.

**The token-agreement proxy is refuted as an evaluator**, exactly as NI1's own memo insisted:
98.6786% token agreement projected `d_seg = 0.0002173` (1.079× DX2); reality is 376.6× DX2 —
the proxy understated by **349×**. NI1 was right to refuse to write a verdict from it, and the
amplification exponent 16.69 is vindicated. My erratum quoted that proxy as though it were
evidence NI1 might clear sub-0.12. It is not evidence of anything about d_seg.

## Where the erratum went wrong — the genus, corrected

The erratum's *factual* claim survives: I still cannot locate a receipt for the numerator.
`grep -rnE "0\.075[0-9]{0,6}"` over the NI1 memo, the RI1 memos, and the arm receipt trees
returns **zero** hits outside my own charter and this document. Token agreement was 98.6786%,
not 7.58%, so it did not come from there either.

But **"I cannot find the receipt" and "the number is invented" are different claims, and I
collapsed them.** That is a **negative-existence claim** — which my own memory index names as
the **#1 false-claim class** (`m53`: *exhaustive search or say "did not find in \<scope\>"*),
and which the charter lint refuses in charters. I searched **one directory** with **one
digit-exact string** (`0.0758429` — my transcription, which the true value `0.07583781` does
not contain as a substring) and concluded fabrication.

Two live facts make absence-of-receipt weak evidence in this corpus specifically:
`#1190` measured **42 research memos existing only on local disk**, invisible to git/graph/corpus;
and both SSD tiers are at 100%, having already killed arms mid-write with zero artifacts. A
receipt that never reached the corpus is the *expected* state here, not an anomaly.

**Named genus: `absence_of_receipt_is_not_proof_of_fabrication`.** Searching for a
*transcription* rather than a *quantity* returns zero hits even when the source is real, and
zero hits reads as invention. The correct move was the one I eventually made by accident —
**measure the quantity** — but I should have made it *before* writing an erratum, not after.

## What the fire bought anyway

The dispatch was the right action under either belief, and it improved the record in a way the
erratum could not: NI1's distortion went from *a citation with no locatable receipt* to a
**contest-CUDA n600 authority row**. Folklore became a receipt. That is a real gain, and it is
why the correct response to "I cannot find the receipt" is a measurement, never a retraction.

## Corrective actions (2026-08-22)

1. This retraction appended; the erratum body above preserved unaltered.
2. All 11 charter markers rewritten from WITHDRAWN to MEASURED-CONFIRMED with the authority row.
3. `MAIN_CORRECTION` written to all four live arms' receipt directories, superseding the erratum
   note they received: **treat `ri1`/`ni1` as CLOSED on measured evidence.**
4. The two-landing cure this erratum proposed (charter-lint ratio-provenance check) **still
   stands and is still owed** — it would have flagged the uncited ratio at spawn. A second cure
   joins it: the lint must refuse a bare negative-existence claim about a *number* unless the
   search scope is stated.

## Own-vehicle frontier

**dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]**, sha `976f706d…` — UNMOVED.
Gap to 0.12: **0.028220**.
