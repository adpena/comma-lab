# ddm_fcd1 — the pose-free field re-selection has a large real rate credit; distortion adjudication is queued

**Date:** 2026-08-29  
**Task:** #1295  
**Status:** `QUEUED-WITH-A-FIRE-ORDER` — byte and public-receiver legs complete; scorer/Schur leg not run because qbt2b r10 owns the single scorer slot  
**Verdict scope:** none yet. No candidate can be admitted or refused, and the family cannot be closed, without n600 realized SegNet and compensated PoseNet rows.  
**Measurement axis:** `[macOS-CPU advisory / scorer-free exact B/H labels and real joint re-encode bytes]`; `score_claim=false`, `promotable=false`.

The useful result is a concrete receiver-closed bank, not a score claim: on the retained jt21 object, 5,268
label-beneficial field edits jointly shrink the real archive from **180,192 B to 176,436 B**
(**−3,756 B**, rate term **−0.0025009662279268756 S**). Three disjoint stratified batches each
shrink it by 1,240–1,292 B. Realized distortion and the fresh Schur pose cost are deliberately
blank, so this does not move the frontier and does not satisfy either exit in the charter yet.

## 1. Break-even arithmetic re-derived before measurement

The dg2 receipt measured the k060000 diagonal at −1,576 B. Its exact rate credit was

`C = 25*1576/37,545,489 = 0.001049393710120542 S`.

The matched advisory row measured `+0.048464 S` Seg damage and `+0.672745 S` Pose damage, for
`+0.720160 S` net after rate. Pose was 93.3% of the gross distortion damage. Even granting perfect
pose compensation on those same moves leaves

`(+0.048464 - C) / C = 45.1828573`

rate credits of residual net harm; equivalently the Seg damage alone was 46.1828573 times its
credit. Therefore same-move compensation is not the live hypothesis. The only plausible reopening
is a different move set chosen under the pose-free cost. That is the object measured below.

## 2. Re-selected field and exact denominators

Inputs were content-pinned before use:

- jt21 bank: 180,192 B, sha256 `ec0dd68ff241070f1c76d5d0da4d8a89b33039bcf56528729a791ec9fd66aef3`.
- decoded token field: 117,964,800 positions, sha256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.
- DX2 coding-argmax field: 117,964,800 positions, sha256 `db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e`.
- DALI-lineage GT argmax: 117,964,800 values, sha256 `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`.

Across all 117,964,800 positions, 227,671 transmitted tokens differ from the coding argmax. Exact
GT attribution on that disagreement population is **B=5,268 (2.3139%)**, **H=221,862
(97.4485%)**, and **W=541 (0.2376%)**. The candidate pool is the full B set:
`token != GT AND coding_argmax == GT`. Therefore its token-label classification is exactly
5,268 BENEFIT / 0 HARM / 0 WASH. This is not a realized SegNet claim: the real receiver and R may
move many scorer cells per token.

Seed 1295 assigns the pool round-robin within `(60-frame block, old->new class)` strata. The three
batches are disjoint, their union is exact, all ten 60-frame blocks are represented, and no prefix
population was used. Retained pool:
`/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/retained/coordinates/benefit_pool.frame_y_x_old_new_assignment.npz`
(27,105 B, sha256 `cc09fd9d4cb9a7253df30dbe38d5f60e33ee9e62c8217d9d0b1276ea5c2b5042`).

## 3. Real joint-coder rows

The n600 inverse-coder control emitted **113,601 B**, byte-identical to the shipped jt21 stream,
sha256 `4c9dc10c0746e1f3bbaed1b754544fbc8ab4b981bbdb37136dc3076cdb976ba7`; prefix match was
113,601/113,601 B. Every row below is a fresh full-field joint re-encode from the jt21 bank, not an
entropy estimate and not an addition of separately banked credits.

| row | support | exact B/H/W labels | real archive | real delta vs jt21 | rate-only delta S | realized d_seg | fresh compensated d_pose, repeat | net delta S | disposition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| batch0 | 1,761 edits / 490 pairs | 1,761 / 0 / 0 | 178,900 B, `1ed41531…` | **−1,292 B** | −0.0008602897674338454 | NOT MEASURED | NOT MEASURED | NOT AVAILABLE | QUEUED |
| batch1 | 1,754 / 486 | 1,754 / 0 / 0 | 178,952 B, `d73dff66…` | **−1,240 B** | −0.0008256651018714925 | NOT MEASURED | NOT MEASURED | NOT AVAILABLE | QUEUED |
| batch2 | 1,753 / 482 | 1,753 / 0 / 0 | 178,951 B, `85826a8d…` | **−1,241 B** | −0.0008263309608246147 | NOT MEASURED | NOT MEASURED | NOT AVAILABLE | QUEUED |
| union | 5,268 / 555 | 5,268 / 0 / 0 | **176,436 B**, `c45ab4e6…` | **−3,756 B** | **−0.0025009662279268756** | NOT MEASURED | NOT MEASURED | NOT AVAILABLE | QUEUED |

The union stream is 109,845 B. Its measured price is **−5.703872437 bits per changed token**;
negative means these beneficial substitutions save coded bits. The batch prices are −5.8694,
−5.6556, and −5.6634 bits/edit. Their agreement is evidence that the rate opening is distributed,
but it is not evidence about distortion. Exact label-benefit counts per saved byte are batch0
**1.363**, batch1 **1.415**, batch2 **1.413**, and union **1.403**, all above the qs5 portable
**0.785 realized-flip/B** break-even. This is only a screening reason to run the scorers: token-label
benefits are not realized scorer flips. The exact byte fire order is **union → batch0 → batch2 →
batch1**.

Primary receipts and all candidate fields, edit planes, streams, archives, checkpoints, and staged
runtimes are retained under
`/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`. The machine-readable summary is
`BYTE_ONLY_RESULT.json`; the candidate archive hashes in that receipt were re-read from disk.

## 4. Receiver and in-compile Schur gate

The first independent receiver attempt exposed the debt already declared by jt21: its archive was
encoded with the 21-family Python `FreeCorrector`, while the copied `f26_corrector_native.c` still
contains an older family generation and correctly refuses `families` drift. The failed attempts
were preserved; neither produced or discarded a raw payload.

Fresh fcd1 runtime copies now default explicitly to the semantically matching shipped Python
fallback and record the `inflate.sh` hash in `PREPARE.json` / `BYTE_ONLY_RESULT.json`. The bank and
source runtime were not edited. Both n600 public-receiver decodes passed:

| object | decoded token identity | retained uint8 raw | elapsed | result |
|---|---|---|---:|---|
| jt21 control | `cc10a7b0…`, exact pinned source token field | 3,662,409,600 B, sha `7246a4ff…` | 1,658.7 s | PASS |
| fcd1 union | `7988b148…`, exact retained union field | 3,662,409,600 B, sha `042fad94…` | 1,740.7 s | PASS |

Both token checkpoints name `FreeCorrector`; both archive bindings and raw hashes were re-read by
the summarizer. Both completed under the 1,800-second public budget even while running together.
Receipts are `decode/{base_jt21,union}/DECODE.json`.

Boundary correction: those first receipts used the phrase “receiver + R + uint8.” Only the public
receiver-to-uint8 output leg ran. Scorer preprocessing R and both frozen scorers did not. The
receipts remain retained; the correction is machine-readable in `RECEIVER_BOUNDARY_CORRECTION.json`,
and the emitter now uses the narrower correct label.

`experiments/ddm_fcd1_incompile_schur.py` makes the later pose compensation part of compilation,
not a carried sidecar. It binds jg5 to the candidate's archive and retained raw, requires n600 on
the same-instrument jt21 control, requires two byte-identical close builds, requires frame-1
sections to remain identical, and asserts inside publish:

`d_pose_after <= d_pose_base + pose_band`.

Publication also raises explicitly if that predicate is false under `python -O`. No Schur result
has run yet, so this apparatus is a queued gate rather than evidence that compensation succeeds.

## 5. Why the scorer leg is queued

The common contract permits one full-n600 scorer job fleet-wide. Active claims
`ddm_qbt2b_r10_metal_20260829` and `ddm_qbt2b_r10_scorer_20260829` (counter 698) own that surface;
this arm has no scorer ownership and did not touch their run directory or claims. Consequently:

- no n600 realized `d_seg` was measured;
- no fresh same-object Schur solve or repeat `d_pose` was measured;
- no net S was computed;
- no admit/refuse bar was applied;
- no candidate seal, family closure, pointer move, or Modal dispatch is claimed.

This is the only honest boundary. Exact BENEFIT labels do not substitute for SegNet, and a rate-only
projection does not substitute for a score.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus and receipts by content for
`field-for-coder|pose-free|schur|hpac|token field|marginal.*re-encode|real re-encode`, then searched
the canonical research indexes, `sub015_DAG_*` FEED blocks, design/spec surfaces, and task-ledger
rows. I also listed the canonical equations registry and consumed the score-marginal, pairset
decomposition, token-direction, greedy-average-vs-marginal, joint-descent, and section-coding laws.

Beyond the charter seeds:

- td1 had already isolated 807 historical “model corrects GT” positions, but explicitly had no real
  re-encode and used an older token object. That changed the plan from reusing 807 coordinates to a
  full reclassification on the pinned live field; the live B population is 5,268.
- fs3/cf2 showed that a selected set's marginal price can differ by 2.24× from its average and that
  projected prices do not survive the admission cut. That changed every candidate price to a real
  full joint re-encode.
- fs2 showed a threshold substitution that looked favorable before the live encoder but failed
  after recapture; this prevented treating exact B labels as an inferred byte or score win.
- jt21 measured 30.4% overlap between standalone coder credits. This forced the bank itself to be
  the base and the union to be actually re-encoded, rather than adding −23 B afterward.
- qs2 is already consumed by the qs5 lineage. It supplied the optimal form but no additive credit;
  this arm does not double-count it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: the qbt2b r10 scorer claim reaches a terminal state. Run the receiver-closed union fresh-object baseline/Gauss-Newton/close/repeat chain, and publish only if its in-compile pose assertion passes.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: the union publishes with repeat-identical compensation. Run n600 frozen SegNet and PoseNet on base and union, recompute S from components, and apply the ±3.5e-6 admit band.
- **FOLDED** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: the union is not admitted but its realized refusal is ≤5×. Process batch0, batch2, then batch1 through the same receiver/Schur/scorer gate; do not infer them from the union.
- **FOLDED** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: any row is admitted beyond the band. Write the canonical candidate seal and dual-axis fire-order into this store, then let MAIN register and dispatch it under single-flight.

## LIVE-HYPOTHESES

- The 5,268 exact label-benefit substitutions may lower realized Seg error after R, because every
  edit changes a wrong transmitted label to the coding model's GT-correct argmax and the rate credit
  is distributed across 555/600 pairs. Only the frozen SegNet row tests transfer from label space.
- Fresh Schur resolution may remove the dominant Pose tax without erasing the −3,756 B field credit,
  because qs5 demonstrated below-base pose with repeat identity on a fresh compiled object. That
  precedent establishes reach, not transfer to this union.
- A strict subset may beat the union on net S even if the union refuses, because the batches have
  independently large rate credits and disjoint support; the union's realized scorer cost need not
  be additive.

## DEAD-ENDS

- Same dg2 moves plus perfect pose compensation: closed for that instance because measured Seg harm
  alone remains 45.18 rate credits net worse.
- Entropy, ideal-code, average-price, or separately-added jt21 estimates: closed as adjudication
  methods because the retained real joint re-encodes now give exact marginal archive bytes.
- Carrying qs4/qs5 compensation onto this body: forbidden and closed; only a fresh candidate-bound
  solve with the in-compile pose assertion can publish.
- Calling exact B/H labels a realized SegNet result: closed as a wrong-object inference; R, uint8,
  receiver behavior, and frozen scorer cells are not represented by token-label counts.
- Reusing td1's 807-coordinate historical instance: closed because it names an older field and had no
  real re-encode; the pinned live object has 5,268 exact B coordinates.
- Running the old generation native corrector on jt21: closed because its embedded family list is
  stale and correctly refuses; the public Python `FreeCorrector` path decoded both exact fields under
  budget, while a future native speed port must be versioned as generation 21.

Own-vehicle frontier: **S = 0.14811799921260607 @ 180,215 B `[contest-CUDA T4 n600]` — UNMOVED.**

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `field_change_bhw_decomposition_v1` — `tac.canonical_equations.ddm_lv3_current_arc_laws_20260901` (`tac.canonical_equations`). **Relation:** IN-DOMAIN ANCHOR (this memo IS the law's anchor).

The B/H/W decomposition and the B-only union's −3,756 B (180,192 → 176,436 B) are the law's registered anchor. The law's own `excluded` clause carries this memo's caveat verbatim: token-label benefit does NOT infer survival through R or PoseNet.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
