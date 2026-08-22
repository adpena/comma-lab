# DDM CB2 class-balanced dictionary — 2026-08-22

**Disposition:** `PRIOR_LAW_REFUTED_AT_STEP_2_NO_REFIT`

**Verdict scope:** `FORMULATION` — the class-valued time-slot allocation in RC1's retained
K=2,048 codebook versus class area in the same retained DX2 token tensor. This does not kill
fixed-K dictionary reweighting as a family and does not measure scorer response.

**Authority:** `[macOS-CPU scorer-free retained-token n600]`; CB2 ran no scorer, RGB evaluation,
Metal, MPS, CUDA, Modal, or `upstream/evaluate.py`. It consumed RI1's terminal read-only advisory
receipt after that receipt landed, but RI1 retained no per-class breakout.

## Result first

The charter's named mechanism is refuted. Lane occupies **691,095 / 117,964,800 = 0.585848%**
of the source token positions, but it occupies **64,539 / 1,228,800 = 5.252197%** of the
existing K=2,048 codebook's class-valued time slots. The final codebook therefore gives Lane
**8.9651× its area share**, not an area-tracking share. The K=256→2,048 incremental capacity is
more Lane-heavy still: **64,243 / 1,075,200 = 5.974981%**.

The dictionary nevertheless preserves Lane badly: **101,792 / 691,095 = 14.729089%** true-Lane
agreement and IoU **0.145961**. Lane contributes **589,303 / 1,420,331 = 41.490540%** of all RC1
token mismatches. This is a real failure, but the measured cause is not the charter's proposed
“capacity follows area” mechanism. Per the mandatory step-2 stop rule, CB2 did not design a
weight, refit the codebook, or materialize any new payload.

## Existing-fit decomposition

All agreement values below compare RC1's independently receiver-decoded categorical tensor with
the retained DX2 token tensor. They are **PROXY-NOT-SCORE** and are not d_seg.

“Capacity” has one explicit denominator: one class-valued time slot in the raw counted
`2,048 × 600` codebook. The 48,920-byte codebook stream is jointly compressed, so this memo does
not invent a per-class byte partition.

| Class | Source positions / 117,964,800 | Area | K=2,048 slots / 1,228,800 | Capacity | Capacity / area | K=256 capacity | Added 1,792 capacity | Agreement given true class | IoU, proxy | Share of 1,420,331 mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Road | 27,406,888 | 23.233107% | 548,935 | 44.672445% | 1.9228× | 74.896484% | 40.354725% | 98.909066% | 0.957125 | 21.050797% |
| **Lane** | **691,095** | **0.585848%** | **64,539** | **5.252197%** | **8.9651×** | **0.192708%** | **5.974981%** | **14.729089%** | **0.145961** | **41.490540%** |
| Undrivable | 58,413,222 | 49.517502% | 329,201 | 26.790446% | 0.5410× | 19.477865% | 27.835100% | 99.681074% | 0.991677 | 13.116309% |
| Movable | 1,460,458 | 1.238046% | 244,863 | 19.927002% | 16.0955× | 1.420573% | 22.570778% | 80.155540% | 0.732058 | 20.405103% |
| MyCar | 29,993,137 | 25.425497% | 41,262 | 3.357910% | 0.1321× | 4.012370% | 3.264416% | 99.813551% | 0.996221 | 3.937251% |

The full area and capacity distributions have total-variation distance **0.447946** and Pearson
correlation **0.347867**. The K=2,048 codebook contains all **256 / 256** retained base codewords
plus **1,792 / 1,792** incremental codewords. The split is load-bearing: the base K=256 Lane
share is below area, but the residual-debt expansion adds Lane-rich programs and lifts total Lane
capacity to 8.97× area. The final fit cannot honestly be described as allocating Lane capacity
in proportion to area.

The independent K=256 comparison reaches the same conclusion on effect rather than representation
occupancy. K=256 has **1,791,175** mismatches; K=2,048 has **1,420,331**, so the added tranche
removes **370,844**. Lane receives **64,221 / 370,844 = 17.317524%** of all removed mismatches,
despite 0.585848% area. The expansion therefore spent material effective capacity on Lane already.
It still removed only **64,221 / 653,524 = 9.826877%** of Lane's K=256 mismatches, versus 24.47%
Road, 28.82% Undrivable, and 30.28% Movable. The remaining Lane collapse is better described as a
hard-program or assignment-geometry problem than as failure to allocate any rare-class capacity.

For completeness, codeword-presence counts are not used as the verdict metric because a temporal
program can contain several classes. Lane appears in **1,527 / 2,048** codewords but is the
plurality class of **0 / 2,048** under a smallest-class-id tie break; that contrast is exactly why
raw codeword counts or plurality labels alone would be misleading.

The independent n600 confusion re-derivation matched RC1's retained confusion matrix exactly:

```text
true rows, predicted columns
[[27107897,     6275,   163704,    72077,    56935],
 [  583728,   101792,     2034,     2985,      556],
 [  122729,        0, 58226927,    63563,        3],
 [  153010,        1,   136668,  1170638,      141],
 [   55866,       22,       11,       23, 29937215]]
```

The complete exact 5×5 matrix is also retained in the stage-2 checkpoint and `RESULT.json`; all
class totals, agreements, IoUs, and mismatch shares above were emitted from that retained matrix.

## Step-2 verdict on the prior law

The prior-law prediction required the existing K=2,048 capacity to track class area, leaving Lane
near its 0.59% area share and therefore roughly 32× below the charter's cited scorer-flip share.
The necessary premise fails on the existing object:

- total Lane capacity is **5.252%**, not approximately **0.586%**;
- the added 1,792-codeword tranche is **5.975% Lane**, showing that RC1's residual-debt expansion
  already redirected capacity toward Lane-rich programs; and
- that tranche delivers **17.318%** of its measured mismatch reduction to Lane, another
  denominator on which the allocation is not area-like; and
- Movable is also strongly overrepresented at **19.927% capacity / 1.238% area = 16.10×**, while
  MyCar is strongly underrepresented, so the full distribution is not area-like.

**Verdict:** `REFUTED_AT_STEP_2_NO_REFIT`, at `FORMULATION` scope. The narrow claim refuted is
that RC1's current capacity allocation tracks class area. The data do not show that K=2,048 is
scorer-optimal, nor that flip-debt weighting cannot help. They show only that the campaign must
not build or fire a refit under this causal story.

## Weighting-field provenance

A retained spatial-debt field does join exactly, but it was not consumed after the stop gate:

- field: `/Volumes/VertigoDataTier/pact/ddm_g4_spatial_stationarity_n600_20260722T212138Z/stage_checkpoints/01_recurrence_arrays.npz`;
- SHA-256: `dbc85e7a4f593ab9b7a7f4ed017dbb63a064cb681df806d0bb93277ae8f42451`;
- axis: `[macOS-CPU frozen-scorer advisory, v12 vehicle]`;
- join: row-major `(y,x)` on the shared `384×512` scorer grid, **196,608 / 196,608** RC1 program
  sites joined; **30,420 / 30,428** unique RC1 programs receive nonzero G4 flip mass;
- retained G4 flip denominator: **4,011,236** flip events; target-class Lane receives
  **301,748 / 4,011,236 = 7.522569%** on that v12 field.

This is a total coordinate join, not current-DX2 calibration. G4 measures v12 predicted-versus-
target cells, while CB2 would alter the DX2/RC1 token field. RI1's terminal n600 receipt later
landed and was consumed exactly (SHA-256
`9d08795f9101a38c03f5b90e4081ced5fd112b15796af345a76789c168ed6425`), but it contains only
aggregate `d_seg=0.01605413` and `d_pose=24.41603851` on an env-mismatch advisory CPU axis. Its
per-pair retention failed with `DistortionNet.__init__()` argument mismatch, and the terminal JSON
contains no per-class or confusion key. Thus no charter-required per-class calibration exists to
consume. The G4 field remains an ancestor-vehicle advisory prior, not a license to call a refit
“sensitivity-weighted.” No uniform class prior, synthetic 19% constant, or invented multiplier was
substituted.

The other charter-named candidates do not repair that authority gap:

- JG3 retains EDIT and KEEP information but not the DROP arm, Pose deltas, or byte deltas;
- task 869/HV2 prepared exact-key orders but did not execute the stated 768-cell × 4-rung scorer
  experiment; and
- old pre-FX5 grouped prices cannot be assigned to individual current-DX2 tokens because the
  arithmetic coder couples later contexts.

## Byte custody and arithmetic

All inherited pins passed before measurement:

| Object | Bytes | SHA-256 |
|---|---:|---|
| RC1 selected payload | 59,884 | `eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164` |
| Raw K=2,048 codebook | 1,228,800 | `d4e5f28b27bef4fca622108db92403942fe1d72470af40ebf11f8ceb308921bf` |
| Raw assignment map | 393,216 | `34c4eaf615d8030a0afd877cc3c2f5896e1e4ed56e0460a8a7932d247f5f2053` |
| Complete RC1 shadow archive | 113,006 | `6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a` |

The inherited selected-section anatomy is unchanged: codebook **48,920 B**, assignment map
**10,900 B**, header **64 B**, RC1 payload **59,884 B**, complete shadow archive **113,006 B**.
No reweighted section exists; `materialized_payloads=[]` and `losing_variants=[]` in the receipt
are deliberate consequences of the step-2 stop, not missing retention.

Arithmetic independently closes:

- DX2: `S = 0.14821987563243377 @ 180,368 B`, `d_seg=0.00020139`,
  `d_pose=0.00000637` `[contest-CUDA T4, n600]`;
- distortion term: `100*d_seg + sqrt(10*d_pose) = 0.028120227975693968`;
- strict continuous byte boundary: `137,986.83879444358 B`, hence integer ceiling **137,986 B**;
- required cut: **42,382 B**;
- RC1 headroom: `137,986 - 113,006 = 24,980 B`;
- RC1 d_seg ceiling at fixed DX2 pose: **0.0003677271516778194**;
- rate exchange: **6.658589531221714e-7 S/B**, or **1,501.81956 B per 0.001 S**.

## Sealed scorer fire-order

The sealed order is
`/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5/SEALED_FIRE_ORDER.json`, SHA-256
`ad75f3dcd22123415a8e2357abd3291f864e82ec583d64003ad9c093a45ae2d8`.

- disposition: `QUEUED_WITH_FIRE_ORDER_BLOCKED_BY_STEP2_FALSIFIER`;
- owner: `MAIN scorer-lane owner`;
- consumer store: `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5/main_fire/`;
- current dispatch argv: `null` — CB2 has no refitted candidate and must not fire;
- fire trigger: a new arm, explicitly not justified by the refuted area-tracking premise, has a
  retained current-DX2 per-class scorer breakout, retains a fixed-K=2,048 candidate whose complete
  receiver archive is at most 113,006 B, and MAIN owns the non-duplicated n600 scorer lane;
- deciding run: baseline RC1 and that candidate through the same shipping RI1 full-RGB receiver on
  all 600 pairs, with per-class Seg confusion/d_seg, d_pose, exact bytes, parse-back, repeat noise,
  and recomputed S; only an advisory pass may proceed to an exact T4 row.

Agreement alone cannot admit the future candidate. The current order is blocked rather than fired,
because CB2 produced no legal candidate and RI1's terminal receipt has no per-class breakout.

## RECALL EVIDENCE

Sources searched:

- binding surfaces: `CLAUDE.md`/`AGENTS.md`, `PROGRAM.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the CB2 charter, and
  `.omx/tmp/codex_runs/_common_contract.md`;
- full-corpus queries in `.omx/research/`: `class-balanced`, `boundary-debt`, `flip-propensity`,
  `per-pixel flip-frequency`, `spatial-stationarity`, `edit drop keep`, `768 cells`, `4 rungs`,
  `token-by-token waterfill`, `joint remeasure`, `dictionary`, `codebook`, and `RC1`;
- canonical equations via `.venv/bin/python tools/list_canonical_equations.py --json`, plus
  `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, the canonical task ledger, and the harness bridge;
- retained source objects: RC1 source index, K=256/K=2,048 codebooks, decoded/source tensors,
  candidate result, G4 recurrence arrays, VF1's source audit, and RI1's live store/status.

What was found beyond the charter seeds, and what changed:

- The K=256 base can be identified exactly inside K=2,048 (**256 / 256**), exposing that the
  1,792-codeword residual expansion—not the base fit—raises Lane capacity from **0.193%** to a
  **5.975%** incremental share. This made the step-2 refutation stronger and stopped the refit.
- G4's field joins every RC1 program site, so the problem is not missing coordinate geometry; its
  vehicle is v12, however, so it cannot silently become current-DX2 sensitivity.
- VF1's source audit establishes that JG3 and task 869 are incomplete for the required scorer/byte
  tuple. This removed both as alternate current-DX2 weighting authority.
- RI1 had built and retained a 113,006-byte shipping bridge. Its advisory scorer run became terminal
  during CB2's final custody pass, so CB2 consumed the exact terminal receipt; the promised per-class
  breakout is absent and per-pair retention failed, so no calibration values were silently inferred.

Within those scopes, no current-DX2 per-program scorer-sensitivity field was found. The spatial G4
join exists; the current-vehicle calibration does not.

## Receipts and boundaries

- result: `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5/RESULT.json`, SHA-256
  `5e27148484152aa9553eafb6a9cc96412064c07b613b69c930f1cb4d3c007682`;
- producer: `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5/retained/producer_source/ddm_cb2_class_balanced_dictionary_gate.py`, SHA-256
  `62d3c3236dfe7923fdaf787419061909c9b7d7718dcd6cdb525a781e48d3a2aa`;
- independent scientific repeat: `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/verification_repeat_v3/RESULT.json`, SHA-256
  `6153f8cac3b7def1c5588725052cc096bc7b0b71b0997256989978ddf8cdaf71`; all sixteen scientific
  result blocks matched byte-for-byte as JSON values, while path/storage metadata remained
  correctly run-specific;
- stage checkpoints: inherited custody, existing-fit decomposition, weighting-field join, and
  terminal stop are distinct atomic receipts;
- `upstream/` was read only; RC1/RI1/NI1 retained trees and memos were not edited;
- no staged-index content or unrelated dirty worktree files were touched;
- exact frontier delta from CB2: **0**. This scorer-free stop did not achieve the sub-0.12 goal.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-FIRE-ORDER, BLOCKED`; **owner:** `MAIN scorer-lane owner`; **consumer store:** `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5/main_fire/`; **fire trigger:** a retained current-DX2 per-class scorer breakout exists and a differently motivated fixed-K=2,048 candidate has a retained complete receiver archive at or below 113,006 B while MAIN owns the unique n600 scorer lane; **action:** run the sealed same-receiver n600 baseline/candidate comparison and admit nothing from agreement alone.

## LIVE-HYPOTHESES

- Lane failure may be caused by allocation topology rather than total class capacity. Lane is present
  in 1,527 codewords and gets 8.97× its area share, yet no codeword is Lane-plurality and true-Lane
  agreement is 14.7%; short intermittent Lane segments may be embedded in majority-class trajectories
  where Hamming assignment erases them.
- A current-DX2 boundary-debt objective may still outperform population Hamming at fixed K. This is
  plausible because Lane supplies 41.5% of RC1 token mismatches despite 0.59% area, but it needs RI1's
  current per-class scorer calibration rather than ancestor G4 transfer.
- Joint class-transition costs may matter more than inverse class frequency. G4's v12 field assigns
  only 7.52% of target-class flips to Lane but 36.02% to Movable, while RC1's mismatch distribution is
  different; ordered transition debt could explain what a scalar class weight cannot.

## DEAD-ENDS

- Re-fitting under the claim that K=2,048 capacity tracks area is closed: measured Lane capacity is
  5.252%, 8.97× its 0.586% area share, and the incremental tranche is 5.975% Lane.
- Raising K is closed for this arm: RC1 already measured K=4,096 at 158,933 B, above the 137,986-byte
  ceiling; it changes the byte object rather than testing a byte-free fixed-K cure.
- Treating overall 98.796% token agreement as evaluator evidence is closed: Lane agreement is 14.729%,
  and no scorer ran here.
- Calling G4 a current-DX2 sensitivity field is closed: its coordinate join is exact but its scorer
  field is ancestor-v12 advisory.
- Treating JG3 or task 869 as a complete retained sensitivity corpus is closed: JG3 lacks DROP/Pose/
  bytes, and task 869 stopped at exact-key preparation without the scorer A/B.
- Synthesizing a uniform class prior, a 19% Lane constant, or per-class compressed byte shares is
  closed: none is a retained current-vehicle measurement, and the codebook coder is joint.

**CB2 own-vehicle frontier line:** **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**; CB2 delta **0**, pointer unmoved.
