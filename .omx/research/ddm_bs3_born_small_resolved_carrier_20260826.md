# DDM BS3 — residual-lifted born-small body and exact-carrier handoff

**Date:** 2026-08-26

**Disposition:** `BODY_BYTE_CLOSED / RESOLVED-CARRIER-MEASUREMENT_BLOCKED_NO_SCORER_OWNERSHIP`

**Measured axis:** `[macOS-CPU advisory / scorer-free exact byte measurement]`

**Verdict scope:** `INSTANCE: HG1 four-generator vocabulary with zero corrections and the inherited DX2 carrier`

**Score claim:** false
**Promotion eligible:** false

## Verdict first

BS3 removed the content of HG1's self-imposed semantic residual and built the resulting real
container. The primary and repeat are both **101,150 B**, SHA-256
`5743f0ac7e8881e970ef8ba53c4bee3fd2a7a6157d2a50d381fd609ae624fea6`. The packet decodes
with **zero corrections** to the retained born-small field byte-for-byte: 117,964,800 B, SHA-256
`2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b`. The body is 79,065 B
smaller than GB1 and has 36,836 B of room under the 137,986 B fixed-GB1-distortion cap.

That is a rate result, not the charter's resolved-carrier distortion result. This arm's queue record
states `owns_scorer=false`. The common contract therefore forbids this arm from taking the scorer
slot and requires byte-only work plus a queued scorer step. No SegNet, PoseNet, full evaluator,
Modal, or MPS run was launched. The body still carries DX2's inherited 22,010 B carrier, so
`d_seg`, `d_pose`, the re-solve improvement versus BO2, the prior-law prediction, and net `S` are
all **UNMEASURED ON THIS OBJECT**.

The requested exact continuation is sealed in
`/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/FIRE_ORDER.json`. It binds the selected
population, the actual DX2 carrier object, the QS5 central-difference/integer-search solve, RJ2's
identity-controlled production carrier encoder, both baselines on one instrument, and the learned
nonlinear holdout screen. It explicitly refuses the tempting but invalid CP135 substitution.

## Measured body

The complete byte accounting closes exactly:

| member | bytes | status |
|---|---:|---|
| semantic renderer | 30,856 | inherited, pinned |
| DX2 pose carrier | 22,010 | inherited, pinned; **not re-solved** |
| compact residual | 96 | inherited, pinned |
| five real-coded HG1 packet streams | 47,689 | measured below |
| HG1 packet/container/ZIP framing | 499 | measured by subtraction |
| **complete archive** | **101,150** | primary/repeat identical |

This is the HG1 `HG1C/HG1P` measurement container used to price all counted sections. It is not a
standalone contest submission: BS3 did not build or run a new `inflate.sh`, and the current parse-back
proves the generator packet and section roster, not a full public-runtime decode of the changed
carrier. Receiver-runnable integration remains part of the queued fresh-carrier stage.

The packet retains an actual 20-byte zero-correction residual header whose winning zlib payload is
22 B. This explains the 22 B difference from BS2's 101,128 B arithmetic construction, which priced
complete removal of the residual stream. In the existing HG1 grammar the roster has five mandatory
streams, so silently pricing that header at zero would be a fake implementation. Removing the
22-byte empty member would require a new receiver version; it is not needed to establish the route's
36,836 B fixed-distortion headroom.

### Real coder rows

Every raw stream was raced through Brotli q11, zlib 9, and LZMA2 extreme. Every coder output and
deterministic repeat is retained, and every winning payload parses back to the exact raw bytes.

| member | raw B | winner | coded B | coded SHA-256 |
|---|---:|---|---:|---|
| Road/Undrivable | 20,447 | LZMA2 extreme | 4,572 | `71a3904a2e5485650eaf34556b324caa3ac71c53ef723886d23b8f0d713d52b6` |
| Lane | 159,395 | LZMA2 extreme | 36,308 | `2bea560ebb1b5ed1b4e32e85a97c5fa12180205d23150d1f61f95cda430ff63f` |
| Movable | 24,599 | LZMA2 extreme | 6,692 | `f28e0485c2e4e9ab491aa634a68563265ed33dc052aca14343d24cac2a4a32b3` |
| MyCar | 24,589 | Brotli q11 | 95 | `afd4cc75649c11d2af8c0a020840564180c1d6dca2d3df7467e4f1f243879709` |
| zero-correction residual | 20 | zlib 9 | 22 | `3100923d7cdc6e9f39c811a007b053e50a1ff8edab13a7bfd04c18e28050f7d9` |
| **total** | **229,050** | — | **47,689** | — |

## Both sub-0.12 demand readings

The live GB1 object is 180,215 B at `S=0.14811799921260607`. Its measured distortion is not
transferred to BS3; the following is arithmetic only.

| reading | strict cap | GB1 relationship | BS3 body relationship | conclusion |
|---|---:|---:|---:|---|
| hold GB1 distortion fixed | 137,986 B | 42,229 B too large | **36,836 B below cap** | the body has enough rate room only if the fresh carrier keeps the total distortion inside the resulting budget |
| set both distortions to zero | 180,218 B | 3 B below cap | **79,068 B below cap** | byte-feasible in the impossible zero-distortion limit; this is not a score claim |

The measured BS3 body rate term is `25*101150/37545489 = 0.06735163310830763`. If, and only if,
it inherited GB1's distortion, the projected score would be `0.0954718610840016`; that projection
is not a row because the born-small render has different pixels. The resolved carrier may also
change its own real-coded byte count, so even 101,150 B is a pre-solve body price, not the final
member size.

## Required baselines and member rows

Baseline numbers below are source-labeled context. They are not copied into the new member.

| row | axis and population | bytes | d_seg | d_pose | role |
|---|---|---:|---:|---:|---|
| GB1 | `[contest-CUDA T4 n600]` | 180,215 | 0.00020139 | 6.37e-6 | live own-vehicle baseline; recalled authority row |
| BO2 HG1 field + stale carrier | `[macOS-CPU advisory PyAV n600]` | 180,368 | 0.01294921 | 1.52821589 | recalled same generated field with inherited carrier; baseline the re-solve must beat |
| BS3 zero-residual body + inherited carrier | `[macOS-CPU advisory scorer-free]` | **101,150** | **UNMEASURED** | **UNMEASURED** | exact byte object built here; not a verdict row |
| BS3 zero-residual body + fresh exact carrier | queued random n32 | **UNMEASURED AFTER RECODE** | **UNMEASURED** | **UNMEASURED** | charter member; not executed without scorer ownership |
| learned implicit evaluator-cell carrier | queued train/holdout screen | **UNMEASURED** | **UNMEASURED** | **UNMEASURED** | screen only; cannot close a family |

The same-instrument three-way measurement is mandatory. Comparing BO2's aggregate n600 PyAV row
to a future sampled result would conflate object, sample, and instrument, so BS3 does not report a
re-solve factor yet.

## Seeded population receipt

The continuation uses seed `20260826`, uniform sampling without replacement from all 600 pairs,
then sorts the draw. This is not a prefix. The selected pairs are:

`26, 39, 41, 62, 69, 71, 73, 77, 90, 100, 104, 114, 223, 227, 235, 253, 296, 308,
310, 322, 330, 362, 364, 388, 415, 481, 482, 493, 509, 522, 557, 588`.

The retained int32 array is 256 B, SHA-256
`1d088e908e74de605128083bff80949ae7574f50f7f495be8a625e0cfc2a9a1f`. Random n32 is a bounded
advisory screen. It is large enough to avoid the forbidden n8-prefix pattern, but it is not an n600
authority row or a learned-carrier family closure.

## Exact carrier binding

The source inspection changed the implementation plan. The QS5 reference archive uses a CP135
carrier blob of 22,242 B, SHA-256 `196f0e51...`; HG1/GB1 uses the DX2 physical carrier of 22,010 B,
SHA-256 `932b979f...`. They are different objects. Constants, codes, basis state, or physical bytes
cannot be transferred between them.

The fire order therefore composes two proven mechanisms without merging their objects:

1. Decode the exact DX2 600x12 signed-int12 lattice with PO1/RJ2.
2. For each retained born-small master, execute QS5's central-difference 6x12 Jacobian, damped
   least-squares centre, integer neighbourhood, and exact coordinate descent until one full
   non-improving pass.
3. Fingerprint tokens, rendered master, pair, codes, scorer inputs, and outputs at the compile
   moment; transferred constants are forbidden.
4. Re-encode the complete lattice through RJ2's identity-controlled
   `CPR1 -> CAP1 -> DX2 -> RR5 -> Brotli q9/lgwin16` chain and receiver-parse the exact codes.
5. Measure GB1/DX2, BO2 stale-carrier born-small, and freshly solved born-small on the same selected
   pairs and scorer process before computing any factor or `S`.

An autograd-only single step, carried overlay, fitted linear model, or CP135 code substitution does
not satisfy this order.

## Learned implicit carrier screen

Status: **QUEUED / UNMEASURED**. The screen fires only after exact solved code deltas exist. Its
minimum honest form is a deterministic one-hidden-layer nonlinear model trained on a fixed subset
of the n32 exact solves and evaluated on held-out pairs. The model weights, real-coded weight
payloads, predictions, frames, scorer inputs, and outputs must all be retained. A result can route a
successor only if held-out realized distortion beats the stale carrier at a materially smaller
payload than storing exact deltas. It cannot close the learned-carrier family in either direction.

PK3/PK4's heldout-dead fitted-linear overlay ceiling forbids substituting a linear regression and
calling it the learned member. The screen must be nonlinear and must survive the real receiver and
scorer cells.

## Amendment-2 form-grade table

| requirement | reference exemplar | BS3 status | grade consequence |
|---|---|---|---|
| residual-lifted born-small semantic body | BS2/HG1 | **PASS** — zero corrections, exact field decode | reference-form body on the existing grammar |
| real coders and complete deterministic measurement container | HG1/GB1 | **PASS** — all three coders raced, repeat equal | reference-form rate evidence for the pre-solve body; public runtime still owed |
| exact carrier object | RJ2 DX2 identity control | **PASS IN FIRE ORDER** — CP135 substitution explicitly refused | correct production surface selected |
| in-compile fresh exact solve | QS5 | **NOT REACHED** — scorer not owned | resolved member not built or graded |
| real render -> R -> uint8 -> scorer | BO2 matched instrument | **NOT REACHED** | no distortion verdict |
| seeded random population, never prefix | common contract | **PASS FOR SELECTION**, scorer pending | n32 screen defined, not measured |
| two baselines in one instrument | BO2 rigor | **NOT REACHED** | no recovery factor |
| nonlinear learned holdout carrier | charter screen | **NOT REACHED** | learned member remains live |
| n600 authority / CPU-CUDA parity | GB1 | **OUT OF SCOPE / NOT REACHED** | no promotion |

Family closure is forbidden. The exact member is not at reference form until the fresh solve,
production recode, real-path measurement, and both baselines close. The learned member is a screen
and cannot close a family even after execution.

## Prior-law prediction

The charter predicts at least a 10x cut in born-small pose damage versus BO2 and treats less than 3x
on the seeded sample as evidence that the damage is body-intrinsic. Both clauses are
**UNADJUDICATED**. No factor is reported from mismatched baseline populations. If random n32 measures
less than 3x recovery on the exact DX2 solve, the analytic re-solve formulation closes with this
form-grade table; the nonlinear learned screen remains separately open.

## Distinction from NI1

NI1 is a whole-body lossy K32 receiver reduction. BS3 instead replaces the semantic token field
with HG1's four born-small generator programs, carries no semantic correction sites, and preserves
the shipped semantic renderer/compact section while making the pose carrier an explicit fresh-solve
obligation. The two representations have different sufficient statistics and different receiver
mechanisms. NI1's 247x result is negative evidence against generic whole-body lossy reduction, not a
distortion number for BS3.

## RECALL EVIDENCE

**Surfaces and queries.** The pass searched the complete `.omx/research/` corpus, charter and arm
receipts, canonical research indexes, the `sub015_DAG_*` graph, the canonical equation registry,
task ledger, main hot state, queue/lane surfaces, and retained APDataStore/Vertigo stores. Content
queries included `born-small`, `resolved carrier`, `in-compile compensation`, `Schur`, `QS5`,
`carrier mismatch`, `DX2`, `GB1`, `HG1`, `BO2`, `BS2`, `JF2`, `RJ2`, `SA3`, `PK3`, `PK4`,
`learned implicit`, `Amendment-2`, `137986`, and `zero-distortion`.

**Findings beyond the charter seeds and effects on the plan.** RJ2 supplied the current DX2 carrier
surface and an identity-proven production encoder, preventing the invalid reuse of CP135 physical
bytes. SA3 and the canonical `compensated_semantic_edit_exchange_v1` law reinforced that the solve
must occur after the exact compile object exists. FB2 supplied the live GB1 update: fixed-distortion
demand is 42,229 B, while zero-distortion demand is now zero with 3 B of slack. RJ2's n1 result also
warns that a fresh solve can help without removing a majority of the damage, so the charter's 10x
prediction cannot be assumed. The canonical indexes did not contain a measured current-object
born-small fresh-carrier row or learned-carrier row in the searched scope. The task ledger had no
BS3 row before this arm registered its three typed entries.

## Custody and payload identities

All generated artifacts are under
`/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/`. Nothing was deleted, moved, reduced to a
scalar-only receipt, or written under `/tmp`.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `BODY_RESULT.json` | retained | `ea3ce5b18ec88d1451c5cd90cd49afc97ee1e52b67cebfe1524aa7abf49f84f3` |
| `FIRE_ORDER.json` | retained | `d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5` |
| primary archive | 101,150 | `5743f0ac7e8881e970ef8ba53c4bee3fd2a7a6157d2a50d381fd609ae624fea6` |
| repeat archive | 101,150 | `5743f0ac7e8881e970ef8ba53c4bee3fd2a7a6157d2a50d381fd609ae624fea6` |
| born-small packet | 48,067 | `963a733cbfbd015c577de7f76548d8ff128510f5968717a2a0c3e9e239bd0e90` |
| direct decoded field | 117,964,800 | `2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b` |
| archive-parseback field | 117,964,800 | `2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b` |
| random-n32 selection | 256 | `1d088e908e74de605128083bff80949ae7574f50f7f495be8a625e0cfc2a9a1f` |
| runner at measurement | 23,113 | `7205b3365eada632ba416a4d7bf31472dd273a3ea087fbf8af946fec56f59557` |
| serializer fallback blocker | 2,471 | `ded7f4f53495004e044c9b6d4bc9acc53b2a89fa1228e4a44e511ddab16bbed3` |

The retained queue transcript records zero live arms, an idle scorer slot, and BS3 marked `DIED`
without scorer ownership. An idle slot is not a transfer of ownership. The lane summary records no
active lane and a stale-only census. Source `upstream/` was read-only.

## Ledger receipts

The canonical append-only store now contains:

- line 687: `ddm_bs3_born_small_resolved_carrier`, registered to actor/owner `ddm_bs3`;
- line 688: `ddm_bs3_exact_resolved_carrier_random32`, `pending`, owner `MAIN sole scorer-lane router`;
- line 689: `ddm_bs3_learned_implicit_carrier_screen`, `pending`, owner `MAIN scorer-lane successor`;
- line 690: main BS3 row `blocked`, with the no-scorer-ownership blocker and exact store paths.
- line 692: FB2 routing note folding its rank-2 scorer fire order into BS3 rather than duplicating it;
- line 694: BS3 serializer-failure note with the retained blocker receipt and uncommitted-artifact status.

The first post-append `tools/canonical_task_status.py --validate` returned exit 0 at 688 rows. A final
validation after concurrent append-only writers returned `{"rows": 692, "status": "valid"}`. Both
runs repeated two pre-existing unrelated unreadable-history warnings for tasks `1079...` and
`1082...`; BS3 did not alter or claim them.

## Verification

```text
.venv/bin/python -m py_compile experiments/ddm_bs3_born_small_resolved_carrier.py
.venv/bin/ruff check experiments/ddm_bs3_born_small_resolved_carrier.py
All checks passed!

.venv/bin/python experiments/ddm_bs3_born_small_resolved_carrier.py \
  --output /Volumes/APDataStore/pact/ddm_bs3_born_small_resolved \
  --resume-from /Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/BODY_RESULT.json
exit 0; completed payload identities revalidated; no coder or payload was rematerialized

tools/review_tracker.py mark-file pass 1: 13 entities reviewed
tools/review_tracker.py mark-file pass 2: 13 entities reviewed

.venv/bin/python tools/canonical_task_status.py --validate
{"rows": 692, "status": "valid"}
```

`git diff --check` is clean for the BS3 source, memo, and ledger append. The first source launch
failed closed before copying or coding because the charter's abbreviated `2884c570...` field hash
had been expanded incorrectly; the full retained source digest was measured, pinned, and the
resumable run then completed. A second early resume refused to overwrite the immutable storage
preflight; the runner now treats that passed preflight as a stage checkpoint and rechecks the live
reserve. Neither failure materialized or discarded a candidate payload.

### Serializer outcome

The required serializer was invoked with the post-edit source and memo hashes, `base=new`,
`--no-coauthor`, and the required governance tokens. The main checkout could not create a Git object
(`Operation not permitted`). Its #1293 fallback then began a full shared-repository checkout on
APDataStore; that rebuildable scratch reached 8.4 GiB and drove free space below the contract's 4 GiB
reserve. The fallback was stopped and exited 130. It produced no commit, bundle, or format patch, so
the BS3 source, memo, and shared-ledger appends remain uncommitted.

The incomplete fallback checkout was retained because it lacks the certification required for
deletion. The machine-readable blocker receipt is
`/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/receipts/commit_serializer_fallbacks/20260826T202505.595571Z-39636/SERIALIZER_FALLBACK_BLOCKER.json`
(2,471 B; SHA-256
`ded7f4f53495004e044c9b6d4bc9acc53b2a89fa1228e4a44e511ddab16bbed3`). The fallback's automatic
temporary-directory cleanup removed only part of its rebuildable clone scratch before termination;
no scientific source, candidate, retained coder output, or BS3 measurement payload was targeted.

## GESTALT-DELTA

Before BS3, born-small's rate case was an arithmetic subtraction and its catastrophic BO2 row made
the family look rate-feasible but practically remote. After BS3, the residual-free object is a real,
repeat-identical 101,150 B container with 36,836 B of fixed-distortion headroom. The uncertainty is
now sharply localized: not the semantic residual and not the byte representation, but whether an
exact fresh DX2 carrier can keep the born-small render inside the large distortion budget that its
rate cut buys. The useful next experiment is one three-way random-pair solve, not another residual
curve or inherited-carrier score.

## What is and is not concluded

- **Concluded:** the residual-lifted HG1 born-small body is a real 101,150 B byte object with exact
  semantic packet parse-back and no correction sites.
- **Concluded:** the object is rate-feasible under both sub-0.12 limiting readings.
- **Concluded:** CP135 is the wrong carrier object; the continuation must operate on DX2 and use the
  RJ2 production encoder.
- **Not concluded:** the born-small body has acceptable Seg or Pose distortion.
- **Not concluded:** exact carrier re-solving yields 10x, 3x, or any recovery on random n32.
- **Not concluded:** the analytic or learned member improves GB1, moves the pointer, or is dead.
- **Not concluded:** the 101,150 B measurement container is a standalone contest-runnable archive.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN sole scorer-lane router`; consumer store: `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/stage_40_three_way_measurement.json`; fire trigger: MAIN explicitly grants scorer ownership, no other full-n600 scorer job is active, and every source/payload SHA in `BODY_RESULT.json` revalidates; execute `FIRE_ORDER.json` stages 0-4 exactly, with chunks no larger than 32.
- `QUEUED-BEHIND-EXACT-SOLVE` — owner: `MAIN scorer-lane successor`; consumer store: `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/stage_50_learned_implicit_screen.json`; fire trigger: stage 40 retains exact random-n32 solved carrier deltas and the analytic member remains arithmetically live; train and hold out the nonlinear screen, retain and real-code every weight/prediction/scorer payload, and route only a held-out GO.
- `BLOCKED-WITH-A-LANDING-ORDER` — owner: `MAIN repository custodian`; consumer store: Pact main plus `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/receipts/commit_serializer_fallbacks/`; fire trigger: Git-object writes become available or a reviewed sparse/no-checkout fallback with certified free space is ready; revalidate final content hashes, land only the BS3 source/memo and exact BS3 ledger hunks, and certify or externalize the incomplete fallback scratch before removing it.

## LIVE-HYPOTHESES

- Exact re-solving may remove most of BO2's pose damage because BO2 changed the master frame while
  retaining a carrier fitted to the old object, and QS5 has already shown near-complete cancellation
  on exact compile objects. The rate body has enough headroom that the member does not need GB1-level
  distortion to remain arithmetically live.
- A nonlinear implicit carrier may compress the 600 solved code deltas because the born-small field
  is generated from temporally coherent low-dimensional programs. This is plausible only if held-out
  realized Pose beats the stale carrier; fitted linear overlays are already heldout-dead.
- The zero-residual body's large byte credit may buy a materially looser absolute Pose budget than
  the incumbent. This follows from the contest equation, but the exact joint Seg/Pose allocation is
  still unknown until the fresh carrier is scored.

## DEAD-ENDS

- Do not score the 101,150 B inherited-carrier body again as the requested member: BO2 already
  measured the stale-carrier failure, and dropping semantic residual bytes does not change pixels.
- Do not transfer CP135 carrier bytes, codes, basis state, or fitted constants into HG1/GB1: the
  physical carrier objects differ.
- Do not replace the QS5 integer search with a fitted linear overlay or an autograd-only one-step
  adjustment: those are different mechanisms, and the fitted-linear family is heldout-dead.
- Do not interpret the n32 selection as an n600 or family verdict, and never replace it with a
  contiguous prefix.
- Do not rerun HG1 residual curves: the semantic residual is now exactly zero and the remaining
  uncertainty is carrier/scorer behavior.
- Do not rerun the serializer's current full-checkout fallback on APDataStore: it allocated 8.4 GiB,
  breached the storage reserve, and still produced no Git landing artifact.

`[contest-CUDA T4 n600] own-vehicle frontier: GB1 — S=0.14811799921260607, archive=180,215 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4; BS3 did not move the pointer.`
