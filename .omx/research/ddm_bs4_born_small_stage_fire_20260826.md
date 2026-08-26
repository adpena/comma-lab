# DDM BS4 — born-small exact stage fire

**Verdict:** `REFUSED_STORAGE_PREFLIGHT`, scoped to the **INSTANCE**
`BS4 random-n32 QS5 exact-retention launch`. Stage 0 revalidated every pinned input,
the sealed DX2 runtime, and the free scorer slot, then failed closed because the
charter-mandated APDataStore root was 15,276,958,976 B short of the conservative
retention launch floor. Stages 1–4 did not fire. There were zero SegNet forwards,
zero PoseNet forwards, zero other scorer forwards, and no score claim.

Axis for every prospective BS4 measurement:
`[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE`.

## Stage receipts

| stage | named checkpoint | disposition | bytes | SHA-256 |
|---|---|---|---:|---|
| 0 | `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/stage_00_source_preflight.json` | **RETAINED — REFUSED_STORAGE_PREFLIGHT** | 121,250 | `bfd33e8dc9e6407c218aef14b9095ec40887566705ea1d0ca1e11b8c6ed4e2a7` |
| 1 | `checkpoints/stage_10_exact_born_small_masters.json` | **NOT FIRED; absent** | — | — |
| 2 | `checkpoints/stage_20_qs5_exact_pair_solves.json` | **NOT FIRED; absent** | — | — |
| 3 | `checkpoints/stage_30_resolved_carrier_container.json` | **NOT FIRED; absent** | — | — |
| 4 | `checkpoints/stage_40_three_way_measurement.json` | **NOT FIRED; absent** | — | — |
| 5 | `checkpoints/stage_50_learned_implicit_screen.json` | **CONDITIONAL GATE NOT REACHED; absent** | — | — |

The Stage-0 checkpoint is append-only. It records `stage_1_through_4_fired=false`,
`all_materialized_payloads_retained=true`, `upstream_mutated=false`, and
`modal_invocations=0`.

## Stage-0 controls

| control | result |
|---|---|
| pinned file and custody identities | **151 / 151 PASS**, including before/after file-stability checks |
| `BODY_RESULT.json` | 55,525 B, `ea3ce5b18ec88d1451c5cd90cd49afc97ee1e52b67cebfe1524aa7abf49f84f3` |
| `FIRE_ORDER.json` | 4,949 B, `d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5` |
| BO2 exact born-small raw | 3,662,409,600 B, `43c359eadd7c6e263adf7a1e2732a2b34948b1db8681bcc1be8f7c493b2ac841` |
| PoseNet weights | 55,835,560 B, `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576` |
| SegNet weights | 38,502,892 B, `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` |
| DX2 archive pin | **CONSISTENT**, 180,368 B, `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| sealed shippable DX2 runtime | 39 files, 685,975 B, tree SHA-256 `7799b291a99027c705b42f094cf0533459399f3ea711ec34d754f81c1fde5f1d` |
| shared scorer slot | **FREE** before storage refusal |
| fire trigger before storage | **PASS** |

The exact seeded sample remains the retained uniform-without-replacement draw from all
600 pairs, seed `20260826`, sorted after draw:
`26, 39, 41, 62, 69, 71, 73, 77, 90, 100, 104, 114, 223, 227, 235, 253,
296, 308, 310, 322, 330, 362, 364, 388, 415, 481, 482, 493, 509, 522, 557, 588`.
Its retained NPY is 256 B, SHA-256
`1d088e908e74de605128083bff80949ae7574f50f7f495be8a625e0cfc2a9a1f`.

## Storage refusal

QS5/QS1's reference retention surface writes an uncompressed camera batch plus its
two-frame pose input for every candidate. That is 9,156,024 B per candidate. The
strict lower-bound census is:

| required work per pair | candidate evaluations |
|---|---:|
| current baseline | 1 |
| exact stale-object event | 1 |
| 12-dimensional central difference: base plus `+1/-1` | 25 |
| minimum radius-2 active three-dimensional integer neighbourhood | 27 |
| one complete non-improving coordinate pass: current plus one legal step per dimension | 13 |
| **minimum per pair** | **67** |

`67 × 32 × 9,156,024 = 19,630,515,456 B` of mandatory retained payloads.
Adding the 8,589,934,592 B reserve gives a launch floor of **28,220,450,048 B**.
This lower bound excludes codes, vectors, masters, JSON framing, every improving
descent pass, the stage-3 container, and all stage-4 three-way scorer payloads.

| tier checked at Stage 0 | free bytes | eligible for this charter | result |
|---|---:|---|---|
| `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved` | 12,943,491,072 | yes, mandated additive root | **FAIL by 15,276,958,976 B** |
| `/Volumes/VertigoDataTier/pact` | 8,970,735,616 | no; charter mandates the APDataStore consumer root | not a fallback |
| `/Users/adpena/Projects/pact` | 381,214,527,488 | no; local bulk needs explicit operator opt-in | not a fallback |

No partial scorer launch was legal. Existing BS3 retained trees are read-only and the
certify-or-block rule forbids deleting or moving them without custody proof. The
retained incomplete BS3 serializer scratch is also insufficient by itself to erase the
measured deficit.

## Three-way measurement

The required same-instrument table does **not** exist. These are typed absences, not
zeroes and not transfers from other populations.

| prospective leg | selected pairs with measurements | real final bytes | `d_seg` | `d_pose` | recomputed `S` |
|---|---:|---:|---:|---:|---:|
| GB1 / exact DX2 base | 0 / 32 | **UNMEASURED in BS4** | **UNMEASURED** | **UNMEASURED** | **UNMEASURED** |
| BO2 born-small / stale carrier | 0 / 32 | **UNMEASURED in BS4** | **UNMEASURED** | **UNMEASURED** | **UNMEASURED** |
| born-small / fresh exact carrier solve | 0 / 32 | **NOT BUILT** | **UNMEASURED** | **UNMEASURED** | **UNMEASURED** |

There are therefore no per-pair rows, recovery factor, real-byte delta, or
ADMIT/refusal comparison at the scientific scope. The recalled GB1 contest row and BO2
advisory n600 row use different authority surfaces and are context only; they are not
inserted into this n32 table.

## Real-byte arithmetic

BS3's repeat-identical 101,150 B object is a **pre-solve body price**, not the final
resolved-carrier container. Its inherited-carrier `d_seg`, `d_pose`, and `S` remain
unmeasured. Stage 3 did not re-encode the carrier and Stage 4 did not price a final
container, so no negative real-byte arithmetic can be claimed.

## Amendment-2 form grade

| required reference-form property | BS4 grade | consequence |
|---|---|---|
| residual-lifted body and lossless parse-back | **PASS, inherited from BS3 and revalidated** | body identity is usable |
| real coder payloads and repeat-identical pre-solve container | **PASS, inherited from BS3 and revalidated** | rate-half custody remains real |
| exact BO2 born-small receiver raw and semantic sources | **PASS at Stage 0** | correct changed object was pinned |
| exact DX2 carrier object and sealed runtime | **PASS at Stage 0** | no CP135 or half-updated-pin substitution |
| seeded uniform random n32 selection | **PASS as a retained definition** | no prefix population |
| QS5 central-difference, damped, integer-neighbourhood exact solve | **NOT REACHED — STORAGE REFUSAL** | no compensation verdict |
| exact render -> R -> uint8 -> scorer | **NOT REACHED — STORAGE REFUSAL** | no distortion verdict |
| GB1, stale BO2, and fresh solve in one instrument | **NOT REACHED — STORAGE REFUSAL** | no joint delta or recovery factor |
| RJ2 production-chain re-encode and repeat parse-back | **NOT REACHED — STORAGE REFUSAL** | final bytes unknown |
| learned implicit carrier holdout screen | **CONDITIONAL GATE NOT REACHED** | learned family remains open |
| n600 contest authority / CPU-CUDA parity | **OUT OF SCOPE** | no promotion |

This table supports only the typed **instance launch refusal**. It does not support a
FORMULATION- or FAMILY-scoped scientific refusal.

## RECALL EVIDENCE

The original recall searched the canonical research store, task ledger, indexes/DAG,
and source tree for `born-small`, `resolved carrier`, `exact DX2`, `fresh carrier`,
`in-compile compensation`, `carrier mismatch`, `Amendment-2`, and the governing body
and carrier identifiers. The sources that changed the execution were:

- QS5's pinned source proves the required central-difference, damped, integer-neighbourhood,
  full-non-improving-pass form; it was read directly rather than recalled from memory.
- HD1's landed pin-consistency and runtime-tree instruments were reused for the DX2
  object instead of duplicating a weaker checker.
- RJ2's n1 evidence recovered 45.073% of a carrier pose gap, so the earlier “10x”
  recovery expectation remains a hypothesis, not a fact.
- FB2 states that a pure frame-0 carrier re-solve cannot rescue BO2's last-frame
  segmentation damage; even perfect pose is not enough on the recalled BO2 instance.
  The mandated same-instrument n32 measurement is still required before closing the
  requested formulation because BO2 aggregate distortion cannot be transferred.
- The canonical equations add no special law for the present result: this is a
  measured capacity refusal under the retention contract, not a new distortion or
  rate model. No new formal equation was warranted.

## Custody and payload identities

| artifact | bytes | SHA-256 |
|---|---:|---|
| Stage-0 checkpoint | 121,250 | `bfd33e8dc9e6407c218aef14b9095ec40887566705ea1d0ca1e11b8c6ed4e2a7` |
| Stage-0 runner | 18,049 | `daee880fd22125eff19aed25ba5d4592e620c93c9ef42d501accfddd56242e99` |
| BS3 `BODY_RESULT.json` | 55,525 | `ea3ce5b18ec88d1451c5cd90cd49afc97ee1e52b67cebfe1524aa7abf49f84f3` |
| BS3 `FIRE_ORDER.json` | 4,949 | `d684c9bc859f825e5d5341c822dcd8c989f91d3a8e7aef1a44316ced3b333db5` |

No scorer payload was materialized and then discarded. The only new execution payload
is the retained Stage-0 checkpoint.

## Ledger receipts

Append-only rows 697–700 of `.omx/state/canonical_task_status.jsonl` record:

- `ddm_bs3_exact_resolved_carrier_random32`: **BLOCKED**, actor `ddm_bs4`, Stage-0
  controls green, storage deficit and checkpoint identity attached.
- `ddm_bs3_learned_implicit_carrier_screen`: **BLOCKED**, actor `ddm_bs4`, because its
  conditional Stage-2/Stage-4 inputs do not exist.
- `ddm_bs3_born_small_resolved_carrier`: advanced from its superseded scorer-ownership
  blocker to **IN_PROGRESS**, then **BLOCKED** on the measured storage floor.

The validator continues to report two pre-existing unreadable task histories
(`1079_pv1...` and `1082_ddm_hm1...`); BS4 did not alter or claim to repair them.

## Verification

- Runner compilation: **PASS**.
- Ruff on the runner: **PASS**.
- Focused storage arithmetic, tamper negative, append-only, alternate-root refusal,
  runtime-seal, file-stability, and wrong-hash controls: **PASS**.
- Two genuine post-edit review-tracker passes over the Python runner: **PASS**.
- Stage-0 execution: expected exit 2 with retained typed refusal; **PASS**.
- `git diff --check`: **PASS**.
- `upstream/`: unchanged by BS4.
- Mandatory serializer landing: **PASS** at commit `d82500f19c`; the post-commit
  content checks passed and no fallback clone or bundle was needed.

## GESTALT-DELTA

Before BS4, the exact fresh-carrier question was queued behind scorer ownership. After
BS4, ownership and all pinned identities are cleared, and the remaining launch blocker
is quantified as a conservative 15,276,958,976 B APDataStore deficit. The scientific
uncertainty is unchanged because no scorer computation occurred. This is apparatus
progress, not score progress, and the exact pointer did not move.

## What is and is not concluded

- **Concluded:** all 151 Stage-0 identities passed, the exact DX2 runtime was consistent,
  and the scorer slot was free for this launch.
- **Concluded:** the mandatory root could not retain the conservative minimum QS5
  payload surface plus reserve; the correct action was to stop before Stage 1.
- **Not concluded:** fresh in-compile compensation improves BO2 pose, beats either
  baseline, yields negative score arithmetic, or is dead at formulation/family scope.
- **Not concluded:** the 101,150 B body is a final carrier-resolved contest archive.

## NEXT_IF_RESUMED

- **BLOCKED-WITH-A-FIRE-ORDER** — owner: `MAIN storage/scorer router`; consumer store:
  `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/`; fire trigger: the mandated
  APDataStore root has at least **28,220,450,048 B free** under certify-or-block custody,
  the scorer slot is free, and every Stage-0 pin revalidates; then resume the immutable
  order at Stages 1–4 with new additive checkpoints.
- **QUEUED-BEHIND-THE-EXACT-SOLVE** — owner: `MAIN scorer-lane successor`; consumer
  store: `checkpoints/stage_50_learned_implicit_screen.json`; fire trigger: Stage 4 has
  retained same-instrument exact deltas and real-byte arithmetic that remain live; then
  run the deterministic one-hidden-layer holdout screen and label it SCREEN only.

## LIVE-HYPOTHESES

- A fresh exact-object QS5 solve can still materially improve the stale-carrier pose
  leg because the known defect is compensation transferred across master objects and
  QS5 has shown that in-compile compensation can cross below its base on another
  object. RJ2's 45.073% n1 recovery makes improvement plausible but does not support a
  10x prediction.
- The joint candidate is unlikely to beat GB1 if its last-frame segmentation resembles
  recalled BO2, because carrier-only frame-0 work cannot change that last-frame debt.
  The retained random-n32 three-way instrument is still useful because it can close
  that exact requested instance without illegally transferring BO2's aggregate row.
- A learned implicit carrier remains worth a labeled screen only if the exact solve
  leaves favorable joint arithmetic; nonlinear compression of solved code deltas is
  plausible, but it cannot manufacture missing last-frame Seg action.

## DEAD-ENDS

- Do not rerun Stages 1–4 at the measured APDataStore capacity: even the conservative
  retained-payload lower bound fails before later descent passes and score payloads.
- Do not redirect this charter to local disk, Vertigo, or split roots without an
  explicit amended authority; the shared APDataStore root is binding.
- Do not delete or move BS3 custody to make room. It is read-only under this charter,
  certify-or-block applies, and the retained serializer scratch alone is smaller than
  the measured deficit.
- Do not report n32 `d_seg`, `d_pose`, `S`, a recovery factor, an ADMIT, or a scientific
  FORMULATION refusal: none was measured.
- Do not substitute CP135, stale compensation, autograd-only/fitted-linear overlays,
  prefixes, or a promoted learned screen for the exact FIRE_ORDER.

`[contest-CUDA T4 n600] own-vehicle frontier: GB1 — S=0.14811799921260607, archive=180,215 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4; BS4 did not move the pointer.`
