# DDM BS4X — Stage-0 cure and selected-object fire refusal

**Verdict:** the Stage-0 re-run deadlock is **CURED**. Both required post-cure
Stage-0 executions returned `READY_FOR_STAGE_1` with all 151 identity controls
green and without changing the original retained refusal. Before heavy Stage 2,
the exact selected DX2 code geometry raised the real QS5 retained-payload launch
floor from the endpoint-safe projection of 28,220,450,048 B to **60,449,654,528
B**. The mandated APDataStore root had **48,102,899,712 B** free, a
**12,346,754,816 B shortfall**, so the charter's typed storage stop fired.
Stages 1–4 did not start, Stage 5 stayed gated, no scorer was loaded, and no score
claim was made.

Axis for every BS4X measurement:
`[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE`.

## Stage-0 cure and executed controls

Commit `72eb83538b5eea84a68e6086192622d97a8fe778` added
`additive_checkpoint_path`: it probes the original checkpoint name, then `_r2`,
`_r3`, and later names, returning the first absent name or a byte-identical
existing name. The mandated-root guard and `atomic_json_once` were not weakened.
The runner cure itself was 23 insertions and 1 deletion; its focused test added
31 lines.

| control | result |
|---|---|
| positive, first post-cure run | `_r2.json` retained; `READY_FOR_STAGE_1`; 151/151 identity controls pass; DX2 pins consistent; scorer slot free; zero scorer forwards |
| positive, second post-cure run | `_r3.json` retained; `READY_FOR_STAGE_1`; 151/151 identity controls pass; DX2 pins consistent; scorer slot free; zero scorer forwards |
| original checkpoint preservation | 121,250 B and SHA-256 `bfd33e8dc9e6407c218aef14b9095ec40887566705ea1d0ca1e11b8c6ed4e2a7` before and after both runs |
| negative, occupied same revision with different bytes | focused test raises `BS4PreflightError`; append-only writer still refuses replacement |
| first Stage-0 free-space observation | 48,104,734,720 B free versus the generic 28,220,450,048 B floor |
| second Stage-0 free-space observation | 48,104,472,576 B free versus the generic 28,220,450,048 B floor |

The source cure passed compilation, Ruff, its focused test, the original
Stage-0 test, two genuine post-edit review passes, serializer post-content
verification, and `git diff --check`.

## Why the Stage-0 projection did not authorize Stage 2

Stage 0 intentionally used endpoint-safe universal minima: a 27-point active
cube and a 13-point first descent pass. The charter separately requires the
storage waterfall to be rechecked before the heavy Stage-2 materialization.
BS4X therefore decoded the pinned DX2 carrier without loading a scorer and
projected the sealed random-n32 selection onto the actual signed-int12 code
lattice.

All 32 selected rows have minimum endpoint margin **200**; the selected code
range is `[-1690, 1847]`. QS5 clips its coordinate update at 32, adds a radius-2
cube, and the first strict-descent pass adds one more step. Only margin 35 is
needed, so every selected row must admit the full 125-point cube and all 25
first-pass coordinate candidates.

| mandatory work per selected pair | candidate evaluations |
|---|---:|
| retained baseline | 1 |
| exact edited event | 1 |
| 12-dimensional central difference | 25 |
| radius-2 cube on 3 active dimensions | 125 |
| first strict-descent pass | 25 |
| **minimum per pair** | **177** |

QS5/QS1 retains the uint8 slave frame and its two-frame uint8 PoseNet input for
each candidate: 9,156,024 raw payload bytes per candidate. Therefore:

`177 × 32 × 9,156,024 = 51,859,719,936 B`

Adding the required 8,589,934,592 B reserve gives the selected-object launch
floor **60,449,654,528 B**. The observed 48,102,899,712 B left a
12,346,754,816 B deficit. This is still a lower bound: it excludes codes,
vectors, exact masters, JSON framing, all improving descent passes after the
first pass, the resolved container, and Stage-4 three-way measurement payloads.

The durable selected-object gate is commit
`971c6d52ccd0570e4991b3a0e7b2210a01d8c9fd`. It passed compilation, Ruff,
focused arithmetic and refusal tests, the Stage-0 regression test, two genuine
post-edit review passes, and serializer post-content verification. Its typed
execution returned rc=2 as required:
`REFUSED_SELECTED_OBJECT_STORAGE_PREFLIGHT`.

The shared assumption challenged during review was that the universal Stage-0
candidate minimum remained representative after the actual object was known.
Violating that assumption did not invent a new scientific mechanism; it exposed
that the sealed code geometry makes the full reference surface mandatory. The
QS5 reference surface and ALWAYS-KEEP contract remain binding, so silently
compressing away, discarding, or recomputing candidate scorer payloads was not
treated as an authorized escape.

## Stage receipts

| stage | named checkpoint | disposition | bytes | SHA-256 |
|---|---|---|---:|---|
| original 0 | `checkpoints/stage_00_source_preflight.json` | **RETAINED — original refusal, unchanged** | 121,250 | `bfd33e8dc9e6407c218aef14b9095ec40887566705ea1d0ca1e11b8c6ed4e2a7` |
| post-cure 0a | `checkpoints/stage_00_source_preflight_r2.json` | **READY_FOR_STAGE_1** | 121,051 | `bce336dbc2aef698e9eb1b6b64fbb44b3a874a1f19d13f3e19b0498a539964dc` |
| post-cure 0b | `checkpoints/stage_00_source_preflight_r3.json` | **READY_FOR_STAGE_1** | 121,051 | `25ad8bc978de6fc1ee49be3cc699ba83469f703954015d2ed35234cd241ba2c7` |
| selected gate 1.5 | `checkpoints/stage_15_selected_storage_preflight.json` | **REFUSED_SELECTED_OBJECT_STORAGE_PREFLIGHT** | 5,943 | `6a53dda5d2bd8cb40c2bc1c85309955577962f24f816644310656731f5765749` |
| 1 | `checkpoints/stage_10_exact_born_small_masters.json` | **NOT FIRED; absent** | — | — |
| 2 | `checkpoints/stage_20_qs5_exact_pair_solves.json` | **NOT FIRED; absent** | — | — |
| 3 | `checkpoints/stage_30_resolved_carrier_container.json` | **NOT FIRED; absent** | — | — |
| 4 | `checkpoints/stage_40_three_way_measurement.json` | **NOT FIRED; absent** | — | — |
| 5 | `checkpoints/stage_50_learned_implicit_screen.json` | **CONDITIONAL GATE NOT REACHED; absent** | — | — |

The selected-object gate retained every newly decoded semantic array:

| retained payload | bytes | SHA-256 |
|---|---:|---|
| `all_dx2_codes.int16.npy` | 14,528 | `2038550e09824f6757fe0096b8fb1014ab6e21408b999de14f7099453ab4ad96` |
| `all_selector_choices.uint8.npy` | 728 | `d5affe43d74400164fc5de746459161e354781faa03a4bddb7f97696b5cb9d59` |
| `selected_dx2_codes.int16.npy` | 896 | `6067acade3204ff219338a3041a8d7161b2b6d75c77539c1499d409c728bbc93` |
| `selected_selector_choices.uint8.npy` | 160 | `dbbf8e64912797f9e49af9f24de2faae24745b200418bcb0561562af9b7f6ad8` |

No retained custody was deleted or moved, no alternate output root was used,
`upstream/` remained read-only, Modal was not invoked, and SegNet/PoseNet/scorer
forward counts are all zero.

## Three-way measurement and the 209.07x prior

The required same-instrument table does not exist. These are typed absences,
not zero measurements:

| leg | measured selected pairs | final real bytes | `d_seg` | `d_pose` | recomputed `S` |
|---|---:|---:|---:|---:|---:|
| GB1 / exact DX2 base | 0 / 32 | **UNMEASURED in BS4X** | **UNMEASURED** | **UNMEASURED** | **UNMEASURED** |
| BO2 born-small / stale carrier | 0 / 32 | **UNMEASURED in BS4X** | **UNMEASURED** | **UNMEASURED** | **UNMEASURED** |
| born-small / fresh exact carrier | 0 / 32 | **NOT BUILT** | **UNMEASURED** | **UNMEASURED** | **UNMEASURED** |

Consequently BS4X neither confirms nor revises BO2's recalled 209.07x refusal.
That row remains measured for the HG1 n600 stale-carrier instance
(`d_seg=0.01294921`, `d_pose=1.52821589`, distortion increase `5.131079`), but
it is not transferred into the selected random-n32 fresh-carrier instrument.
The requested scientific handoff is therefore a **BLOCKER**, not a formulation
verdict: selected-object retained capacity must clear before the same-instrument
question can be measured.

The 101,150 B BS3 object remains a real, retained pre-solve body. It is not a
carrier-resolved final archive and cannot support final-byte or score arithmetic.

## RECALL EVIDENCE

The bounded recall searched the canonical research store, canonical task
ledger, indexes/DAG, source executor lineage, and canonical equations for
`born-small`, `resolved carrier`, `exact DX2`, `fresh carrier`, `in-compile
compensation`, `binary collapse`, `Amendment-2`, `BO2`, `QS5`, and the governing
body/fire-order identities.

Evidence beyond the charter seeds changed the plan in four ways:

- `.omx/research/ddm_bo2_born_small_distortion_row_20260824.md` kept the
  209.07x row instance-scoped; it could not substitute for the requested n32
  three-way measurement.
- `.omx/research/ddm_qs5_resolve_compensation_20260813.md` and its exact source
  fixed the required central-difference, radius-2 cube, and strict-descent
  retention form; no stale compensation was transferred.
- `.omx/research/ddm_rj2_joint_renderer_object_change_20260823.md` reports only
  45.073% pose-gap recovery at n1, so the earlier majority/10x expectation
  remains unproven.
- Prior binary-collapse work reinforced denominator-first, typed-scope
  reporting. The selected denominator is 32/32 code rows for the storage proof;
  the scientific scorer denominator is 0/32 and stays explicitly unmeasured.

The canonical equations supplied no new distortion or rate law. The result is
a measured capacity refusal under the exact retention contract.

## Ledger receipts

Append-only rows 715–719 of `.omx/state/canonical_task_status.jsonl`, actor
`ddm_bs4x`, record:

- `ddm_bs3_exact_resolved_carrier_random32`: resumed only through the no-scorer
  gate, then **BLOCKED** on the 60,449,654,528 B floor.
- `ddm_bs3_born_small_resolved_carrier`: prior tool/scorer blocker cleared for
  the launch, then **BLOCKED** on selected-object storage.
- `ddm_bs3_learned_implicit_carrier_screen`: **DEFERRED / QUEUED-BEHIND-THE-EXACT-SOLVE**;
  Stage 5 did not fire.

The canonical reader still emits warnings for older unrelated malformed or
uncustodied histories. BS4X did not alter those histories.

## GESTALT-DELTA

Before BS4X, Stage 0 could not represent a second run and the fire projection
treated 67 candidate evaluations per pair as the safe minimum. After BS4X, the
versioned checkpoint ladder is proven in both directions, two independent
Stage-0 runs are READY, and the exact selected object is proven to require at
least 177 candidate evaluations per pair. The blocker is now 12,346,754,816 B
of APDataStore capacity under the immutable retention form. This is apparatus
truth, not score progress; the exact pointer did not move.

## What is and is not concluded

- **Concluded:** the versioned Stage-0 cure works without weakening either
  guard; the original checkpoint remained byte-identical.
- **Concluded:** all 151 identities, DX2 pin consistency, and the scorer-slot
  check passed in both post-cure Stage-0 executions.
- **Concluded:** all 32 selected DX2 rows force the full candidate surface and
  the mandated AP root was 12,346,754,816 B short of its lower-bound fire floor.
- **Not concluded:** fresh in-compile compensation improves BO2, recovers a
  majority of its pose debt, beats GB1, or admits the route.
- **Not concluded:** any n32 or n600 `d_seg`, `d_pose`, final bytes, or `S` for
  the fresh-carrier object.

## NEXT_IF_RESUMED

- **BLOCKED-WITH-A-FIRE-ORDER** — owner: `MAIN storage/scorer router`; consumer
  store: `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/`; fire trigger:
  the mandated root has at least **60,449,654,528 B free** under certify-or-block
  custody, the scorer slot is free, and every Stage-0/selected-object pin
  revalidates; action: start the immutable order at Stage 1 and retain each
  additive Stage 1–4 checkpoint.
- **QUEUED-BEHIND-THE-EXACT-SOLVE** — owner: `MAIN scorer-lane successor`;
  consumer store: `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/checkpoints/stage_50_learned_implicit_screen.json`;
  fire trigger: retained Stage 4 same-instrument deltas and real-byte arithmetic
  exist and leave the learned screen live; action: run only the labeled
  deterministic holdout screen, never promote it as an exact row.

## LIVE-HYPOTHESES

- Fresh exact-object QS5 compensation can still reduce the stale-carrier pose
  debt because QS5 has crossed below its base on another object and the defect
  here is cross-object compensation transfer. RJ2's 45.073% n1 recovery makes a
  reduction plausible, but not a majority or 10x recovery.
- The joint object is unlikely to beat GB1 if its last-frame Seg damage resembles
  BO2, because frame-0 carrier compensation cannot repair a frame-1 Seg miss.
  The random-n32 same-instrument table remains necessary to test that exact
  instance without population transfer.
- A lossless representation of retained candidate payloads might reduce physical
  storage, but it is not an authorized substitute for the sealed QS5 reference
  surface. It is worth pursuing only under an amended charter that proves exact
  payload recovery, hashes every candidate, and preserves resumability.

## DEAD-ENDS

- Do not rerun Stage 2 at the present AP capacity: the exact selected-object
  lower bound fails before later descent passes, Stage 3, or Stage 4.
- Do not use the old 28,220,450,048 B trigger: it is only the endpoint-safe
  universal projection and is falsified for all 32 selected rows.
- Do not redirect payloads to local disk or Vertigo, split the roots, delete or
  move BS3 custody, discard candidates, or recompute them instead of retaining
  them; none is authorized by this charter.
- Do not substitute stale compensation, CP135, a linear/autograd proxy, a
  prefix, or Stage 5 for the exact FIRE_ORDER.
- Do not report a recovery factor, ADMIT, family refusal, `d_seg`, `d_pose`,
  final bytes, or `S`: the scorer denominator is 0/32.

`[contest-CUDA T4 n600] own-vehicle frontier: GB1 — S=0.14811799921260607, archive=180,215 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4; BS4X did not move the pointer.`
