# DDM LQ1 lane-quotient representability — 2026-08-22

**MEASURED:** RC1's Lane-recall oracle over the existing K=2,048 codebook reaches
**417,267 / 691,095 = 60.377662%**, while NR1-K32 independently retains only
**101,672 / 691,095 = 14.711726%** Lane agreement. The registered representational prediction
required **<60%** and its falsifier required **>90%**. The result is therefore
`INCONCLUSIVE_AT_PREREGISTERED_THRESHOLDS`, with `verdict_scope=FORMULATION`; the quotient family is
**not closed**.

**Disposition:** `MIXED_MECHANISM_RESOLVED_FORMULATION_NOT_FAMILY`

**Authority:** `[macOS-CPU scorer-free retained-token n600]`. Every agreement and IoU value in this
memo is **PROXY-NOT-SCORE**. LQ1 ran no scorer, RGB receiver, `upstream/evaluate.py`, Metal, MPS,
CUDA, or Modal job.

## Result first

The failure has three measured components rather than one clean cause:

| Retained surface | Lane correct / 691,095 | Lane agreement, PROXY-NOT-SCORE | All token mismatches / 117,964,800 | Delta from retained RC1 |
|---|---:|---:|---:|---:|
| RC1 retained assignment | 101,792 | 14.729089% | 1,420,331 | baseline |
| RC1 full-Hamming oracle assignment | 159,903 | 23.137629% | 1,185,227 | **−235,104** mismatches |
| RC1 Lane-recall oracle assignment | 417,267 | **60.377662%** | 4,175,654 | **+2,755,323** mismatches |
| NR1-K32 retained receiver | 101,672 | **14.711726%** | 1,558,833 | separate representation |

The full-Hamming oracle proves **assignment loss**: RC1's source records
`global_reassignment_to_added_programs=false`, and globally reusing its already-counted 1,792 added
exact programs improves both the total objective and Lane. It reassigns **5,455 / 30,428** unique
programs, representing **6,431 / 196,608** spatial sites, recovers **58,111** Lane tokens, and removes
**235,104** total mismatches.

That cure is not enough. The stronger Lane oracle recovers **315,475** Lane tokens but creates
**3,070,798** additional non-Lane mismatches, or **9.733887 collateral mismatches per Lane token
recovered**. Even after accepting that cost, **273,828 / 691,095 = 39.622338%** of Lane positions
remain unrepresentable by any one existing codeword at their spatial program site. Thus assignment
loss is real, population-Hamming objective geometry strongly suppresses Lane, and the current
codebook is also incomplete for Lane. No single component alone explains the collapse.

## Inherited-state reproduction

All charter pins passed before measurement: CB2 memo and result, RC1 payload/codebook/assignment/
shadow hashes, RC1 and NR1 modules, NR1 K32 packet and decoded field, and the shared DX2 token source
SHA-256 `cc10a7b0…3eefb`. RC1 and NR1 both compare against that exact
`600×384×512 = 117,964,800`-token object.

LQ1 independently reproduced CB2's five rows exactly:

| Class | Source positions / 117,964,800 | Area | K=2,048 slots / 1,228,800 | Capacity | Capacity / area | Retained agreement given true class, PROXY-NOT-SCORE | Share of 1,420,331 mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Road | 27,406,888 | 23.233107% | 548,935 | 44.672445% | 1.9228× | 98.909066% | 21.050797% |
| **Lane** | **691,095** | **0.585848%** | **64,539** | **5.252197%** | **8.9651×** | **14.729089%** | **41.490540%** |
| Undrivable | 58,413,222 | 49.517502% | 329,201 | 26.790446% | 0.5410× | 99.681074% | 13.116309% |
| Movable | 1,460,458 | 1.238046% | 244,863 | 19.927002% | 16.0955× | 80.155540% | 20.405103% |
| MyCar | 29,993,137 | 25.425497% | 41,262 | 3.357910% | 0.1321× | 99.813551% | 3.937251% |

The independently reconstructed RC1 confusion is byte-identical to CB2's retained matrix:

```text
true rows, predicted columns
[[27107897,     6275,   163704,    72077,    56935],
 [  583728,   101792,     2034,     2985,      556],
 [  122729,        0, 58226927,    63563,        3],
 [  153010,        1,   136668,  1170638,      141],
 [   55866,       22,       11,       23, 29937215]]
```

Any disagreement would have been the finding. There was none.

## Hypothesis diagnosis

### 1. Intrinsic diversity — measured, materially binding

Lane is thin in area but broad in temporal-program support:

- **19,361 / 30,428** unique full temporal programs contain Lane, across
  **34,357 / 196,608** spatial sites.
- Those programs contain **15,348** distinct 600-step binary Lane masks.
- The codebook has **1,527 / 2,048** words containing Lane but only **1,412** distinct Lane masks.
- Only **1,523 / 19,361 = 7.866329%** of Lane-bearing source programs occur exactly in the codebook.
- Exact full-program members cover **78,115 / 691,095 = 11.303077%** of Lane positions. Allowing
  any codeword with the exact same Lane mask raises that only to
  **83,414 / 691,095 = 12.069831%**.
- Lane-token-weighted program entropy is **13.498340 bits**, effective perplexity
  **11,571.917**. The most common **3,298** programs cover 50% of Lane mass; **9,968** cover 90%,
  **12,001** cover 95%, and **15,328** cover 99%.

The `64,539` codebook Lane slots therefore do not mean `64,539` independently addressable source
positions. They are reused time slots inside only 2,048 shared programs. On the actual program-space
denominator, Lane is diverse enough that the retained codebook does not contain most Lane masks.

### 2. Assignment loss — measured, real, but not sufficient

RC1 did not globally reassign all programs after appending 1,792 exact residual-debt codewords. The
full-Hamming oracle performs that missing operation without changing the codebook:

```text
current true-Lane row:     [583728, 101792, 2034, 2985, 556]
full-Hamming oracle row:   [528457, 159903,  221, 2063, 451]
```

Lane agreement rises from **14.729089%** to **23.137629%**, and total mismatches fall from
**1,420,331** to **1,185,227**. This is unambiguous assignment loss: the existing codebook already
contains better full-program choices that the retained assignment policy never considered.

The honest immediate cure is **global full-Hamming reassignment against all existing K=2,048
codewords before any refit or K increase**. It needs zero new codewords, zero new payload sections,
and zero additional raw assignment bytes because it replaces the same fixed
`196,608×uint16 = 393,216`-byte field. Its exact compressed-byte delta is **UNKNOWN**: the retained
oracle field was not recut into RC1 because this charter forbids live-payload recutting. The current
10,900-byte compressed assignment stream cannot be assumed unchanged, so LQ1 does not call the cure
archive-byte-free.

Global reassignment is a Pareto improvement in token space, not a Lane solution. It still misclassifies
**531,192 / 691,095 = 76.862371%** of Lane positions and is not evaluator evidence.

### 3. Objective geometry — measured, strongly binding

The Lane oracle chooses, independently for each spatial program, the existing codeword with the most
correct class-1 time slots. It then minimizes full-program Hamming among Lane-maximizing ties and uses
the smallest codeword index as the final deterministic tie break. Programs without Lane use the
full-Hamming oracle. This preserves the one-codeword-per-spatial-site representation while removing the
encoder and population objective from the upper bound.

Its confusion is:

```text
true rows, predicted columns
[[24317914, 1210738,  779967, 910636, 187633],
 [  203432,  417267,   30228,  23503,  16665],
 [  142761,   14483, 58159701, 92747,   3530],
 [  290546,   44057,  128951, 995040,   1864],
 [   83639,    6214,    3107,    953, 29899224]]
```

Maximizing Lane within this codebook raises Lane agreement to **60.377662%**, but total mismatch rises
to **4,175,654**. The **3,070,798** new non-Lane errors for **315,475** recovered Lane errors explain
why an unweighted population-Hamming assignment rejects these codewords: the scarce class is cheap in
the objective, while the collateral majority-class cost is large.

The oracle still cannot exceed 60.38%, so objective geometry is not the only wall. The codebook lacks
Lane support at many source time patterns even when all collateral is ignored.

## NR1-K32 cross-application

LQ1 independently re-derived NR1's confusion from its retained `received_tokens.u8` against the same
DX2 source. The matrix matches NR1's retained decode manifest exactly. Every value is
**PROXY-NOT-SCORE**:

| Class | Correct / true positions | Agreement given true class | IoU, proxy | Share of 1,558,833 mismatches |
|---|---:|---:|---:|---:|
| Road | 26,991,462 / 27,406,888 | 98.484228% | 0.950817 | 26.649808% |
| **Lane** | **101,672 / 691,095** | **14.711726%** | **0.135998** | **37.811812%** |
| Undrivable | 58,142,705 / 58,413,222 | 99.536891% | 0.989936 | 17.353815% |
| Movable | 1,289,050 / 1,460,458 | 88.263408% | 0.819497 | 10.995918% |
| MyCar | 29,881,078 / 29,993,137 | 99.626385% | 0.993335 | 7.188647% |

NR1-K32 therefore reproduces the visible Lane collapse almost exactly: **14.711726%** versus RC1's
**14.729089%**. This is strong cross-instance evidence that population-dominant quotient objectives on
this token field suppress Lane. It is not a family closure: NR1's own codebook-containment oracle was not
chartered or measured, and RC1's 60.38% result misses the registered <60% family-support threshold.

NI1's scorer-side result remains separately owned. LQ1 did not wait for it and does not substitute this
agreement for d_seg.

## Prior-law verdict and scope

The prior-law prediction was:

> Lane is representationally absent, so the RC1 Lane oracle remains below 60%; NR1-K32 also collapses.

The NR1 leg is confirmed, but RC1 lands at **60.377662%**, **0.377662 percentage points above** the
registered support threshold and **29.622338 points below** the >90% falsifier. Rounding it below 60 or
promoting it to “high” would both violate the charter.

**Verdict:** `INCONCLUSIVE_AT_PREREGISTERED_THRESHOLDS`,
`verdict_scope=FORMULATION` — the retained RC1 K=2,048 codebook, its two assignment rules, and the
retained NR1-K32 agreement cross-check. The evidence supports the qualitative representational-pressure
direction, directly proves assignment and objective-geometry losses, and forbids `verdict_scope=FAMILY`.

No third representation is adopted from this proxy result. A representation that addresses the measured
wall would have to **factor Lane boundary/mask topology from the shared five-class temporal program** so
that Lane support is not purchased by selecting a whole codeword that creates 9.73 collateral errors per
recovered Lane token. That is a design requirement, not an implemented or scored candidate.

## Score arithmetic and boundaries

The charter arithmetic remains unchanged:

- DX2: `S=0.14821987563243377 @ 180,368 B`, `d_seg=0.00020139`,
  `d_pose=0.00000637` `[contest-CUDA T4, n600]`.
- Strict sub-0.12 archive ceiling at current distortion: **137,986 B**; required cut **42,382 B**.
- RC1 shadow: **113,006 B**, leaving **24,980 B** headroom and a fixed-pose d_seg ceiling
  **0.0003677271516778194**.
- Perfect-Lane mismatch-share arithmetic remains only an upper bound; mismatch shares do not add into
  d_seg, and LQ1 ran no scorer.

LQ1 measured token fields only. It did not edit RC1/NR1 payloads, receiver trees, RI1/NI1/AD2 memos,
their retained stores, the sacred jo1 r9 directory, `upstream/`, the staged index, or unrelated dirty
worktree files. The exact frontier delta from LQ1 is **0**; this did not achieve sub-0.12.

## RECALL EVIDENCE

Sources searched:

- governing surfaces: `AGENTS.md`/`CLAUDE.md`, `PROGRAM.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the LQ1 charter, and the
  common contract;
- full research corpus queries for `Lane`, `quotient`, `dictionary`, `codebook`, `temporal program`,
  `oracle assignment`, `representability`, `global_reassignment_to_added_programs`, `K32`, and
  `class_confusion` under `.omx/research/` and arm receipts;
- the canonical-equation registry via
  `.venv/bin/python tools/list_canonical_equations.py --json`, plus
  `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, design/spec/source surfaces, the canonical task ledger,
  and harness bridge;
- primary retained bytes: RC1 source index/codebook/assignment/decoded field, CB2 result, NR1 packet/
  decode manifest/decoded field, and both source modules.

What was found beyond the charter seeds, and what changed:

- RC1 source and result explicitly record `global_reassignment_to_added_programs=false`. This split the
  requested oracle into a full-Hamming assignment oracle and a Lane-containment oracle; without that
  split, assignment loss would have been mislabeled representational.
- The kernel-setoid quotient FEED requires fiber completeness before an ideal quotient-rate or family
  claim. No current token-field Lane containment oracle was found beyond LQ1 in the searched index/DAG/
  equation scopes, so the verdict remains formulation-scoped.
- NR1 already retained a confusion matrix, but LQ1 did not trust the summary: it re-derived the matrix
  from the shared source and receiver bytes and matched it exactly.
- Older Lane/level-set work concerns scorer-visible RGB witnesses, not this terminal categorical-token
  quotient. It supplies a possible successor mechanism class but no transferable number here.

Within those bounded scopes, no prior retained oracle answered whether RC1's current codebook contains
Lane programs under a one-codeword-per-site assignment.

## Receipts and independent verification

- primary result:
  `/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/measurement_v2/RESULT.json`,
  **23,221 B**, SHA-256 `dfd4c95df7f0becb7c82aebec468c900be6bc8d83a7c8647ccb80c4bccf90e9c`;
- independent different-chunk repeat:
  `/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/verification_repeat_v2/RESULT.json`,
  **23,604 B**, SHA-256 `345ed7003e9b80565b06a73d9ce9afbf3c94fb0fca1ba90f0b538ca3a02e59e7`;
- normalized scientific block, both runs: SHA-256
  `2b05bf0885c0c2c3873adcd35bf0f35c6b0ad5255ffa96ae268ec4f09bf46ff4`;
- retained producer: **39,824 B**, SHA-256
  `d0c05702b8bb72665a9fe732da261bf2e703b9fe6ddcb15363a0911244a34ee5`;
- retained diversity/program field: **4,612,004 B**, SHA-256
  `526b59c318d8a18ca42a43fbb5841a8fd5ca349a59a12e06fe5e984124ee9dc1`;
- full-Hamming site assignment: **393,216 B**, SHA-256
  `e01e9eecee0a7fdc271b86cbadac14bfab92480c6c6768358af72e03ef54ed16`;
- Lane-recall site assignment: **393,216 B**, SHA-256
  `3a40cba94cfddf7d0beb88233cbf206f10f0e4fbfe0fc8b191b4800681526925`;
- per-class decomposition: **11,824 B**, SHA-256
  `9f248cbc7501eefa7e7c7b0c8ccfc248d83ae48b6ea92372179ca925c9abd89a`.

The repeat used program/time chunks `97/17` instead of `128/20`. Both site fields, all five unique-
program oracle arrays, and all four confusion matrices are byte-identical. Ten independently recomputed
program/codebook oracle checks passed **10/10**. The receipt set stayed on APDataStore after the
256-MiB requirement plus 8-GiB reserve preflight passed.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-FIRE-ORDER`; **owner:** `MAIN-assigned RC1 successor after AD2/RI1 ownership release`; **consumer store:** `/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/successor_global_reassignment/`; **fire trigger:** this memo is committed, AD2 and RI1 are terminal or explicitly non-overlapping, no duplicate RC1 recut is active, and the retained full-Hamming field SHA is `e01e9eec…ed16`; **action:** real-code that replacement assignment with all losers/repeats retained, parse it through an otherwise unchanged K=2,048 RC1 receiver, require a complete archive at or below 113,006 B, then queue any scorer row to MAIN rather than firing locally.
- **Disposition:** `QUEUED-WITH-FIRE-ORDER`; **owner:** `MAIN-assigned NR1 successor after NI1 ownership release`; **consumer store:** `/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/nr1_k32_oracle/`; **fire trigger:** NI1 is terminal, the K32 packet/source hashes remain pinned, and no duplicate NR1 analysis owns the tree; **action:** measure NR1's own class-1 containment and assignment oracles on the same 691,095-position denominator so a later arm can admit or refuse FAMILY scope.

## LIVE-HYPOTHESES

- A Lane-mask or Lane-transition specialist conditioned by the shared quotient may beat whole-program
  assignment. It is plausible because 15,348 source Lane masks are being forced through 1,412 codebook
  masks, and choosing a whole Lane-richer program costs 9.73 collateral errors per recovered Lane token.
- Global full-Hamming reassignment plus a small topology-specialist residual may dominate either alone.
  The first leg already removes 235,104 total mismatches without a new codeword; the remaining Lane debt
  is localized to support absent from the codebook, not merely a stale assignment.
- NR1 may share RC1's objective-geometry failure rather than a universal quotient representability wall.
  Its 14.71% Lane agreement is nearly identical, but only an NR1-specific containment oracle can separate
  absent task-cell patterns from its retained assignment rule.

## DEAD-ENDS

- Simple class-balanced or flip-weighted capacity reallocation is closed: CB2 measured Lane at 8.97× its
  area share and stopped under its own falsifier. LQ1 does not resurrect that refit.
- Global reassignment as a complete Lane cure is closed: it is Pareto-better overall but reaches only
  23.14% Lane agreement.
- Calling the current codebook highly Lane-representable is closed: even the Lane-only oracle reaches
  only 60.38%, not the registered >90% falsifier.
- Calling the whole quotient/dictionary family dead is also closed on this evidence: 60.38% misses the
  registered <60% support threshold, and NR1 has no containment oracle yet.
- Raising K is closed for this route: RC1's retained K=4,096 shadow is 158,933 B, above the 137,986-byte
  ceiling.
- Promoting token agreement, mismatch shares, or perfect-Lane arithmetic into d_seg is closed: no scorer
  ran, and per-class mismatch effects do not compose additively through the evaluator.

**LQ1 own-vehicle frontier line:** **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**; LQ1 delta **0**, pointer unmoved.
