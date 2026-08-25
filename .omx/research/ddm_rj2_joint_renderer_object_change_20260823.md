# DDM RJ2 — exact-object joint renderer smoke

**Date:** 2026-08-25  
**Disposition:** `MECHANISM_COMPLETE / POINTER_MOVE_REFUSED_AT_N1_SCOPE`  
**verdict_scope:** `INSTANCE: film_amortized_flat_w96, pair 0, one float32 CPU step, one bounded int12 carrier solve`  
**Authority:** `[macOS-CPU advisory n1 exact local scorers; engineering smoke]`  
**Score claim:** false  
**Promotion eligible:** false

## Verdict first

RJ2 built and exercised the complete requested mechanism on the exact DX2 object. The run jointly
optimized the moved W96 renderer against both frozen scorers, consumed MF1's boundary/margin surface
inside the objective, packed the EMA shadow, solved carrier compensation against that final packet,
re-encoded the full 600x12 carrier through the production chain, receiver-parsed the resulting
semantic packet and carrier codes, and retained byte-identical primary/repeat archives.

The mechanism passes. The bounded candidate does not. It is **179,274 B**, 1,094 B below DX2, but
the pair-0 scope arithmetic moves from **0.17024397523303644** to
**0.3530185565980673**, a measured **+0.18277458136503086 S** on the stated n1 CPU axis. This is not
an n600 score and cannot move or kill a family. It does falsify the bounded smoke's negative-joint-
delta admission trigger.

The final primary and repeat are both 179,274 B with SHA-256
`82bda77dd2eb582a21fa607c0473c636857f040d603b493b6c88a15c947af12a`. The first machine receipt is
`/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change/SMOKE_RESULT.json`; the final
reviewed replay receipt supersedes it and is
`/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change/reviewed_replay_r1/SMOKE_RESULT.json`,
SHA-256 `db7bed77238190627dc7e34bde06a18575958d304cccde5d3cf1f290a668e0f8`.
No Metal, Modal, full-n600 scorer, full inflate, or evaluator invocation ran.

## Exact source and retention binding

| object | bytes | SHA-256 | use |
|---|---:|---|---|
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | exact base container |
| DX2 local raw | 3,662,409,600 | `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` | retained pair source |
| DX2 categorical field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | exact unchanged field |
| RJ1 W96 initializer | 253,955 | `e74ba046af251808ef105cf0a2295f6133efa194360148f3110762765b9db434` | governed birth |
| MF1 boundary r9 | 14,745,600 | `8adbd0f04f66f7527c7245448ff10351b41e2c640b7170cce0d04a794366e501` | training input |
| MF1 manufactured mask | 14,745,600 | `cd7b0176e0d6a41d73c9ae539acf9a24304f3ff0a87a96faaa83709673beffb6` | training input |
| GT cache | 5,078,017,610 | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` | one-row Seg/Pose targets and margins |

The output was re-homed to
`/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change/`; the final replay is under its
`reviewed_replay_r1/` child, and RJ1's Vertigo tree remained read-only. Final preparation observed
117,223,456,768 free bytes and retained its storage, source subset, birth checkpoint, and initial
packet receipts. Every later materialized encoder stage is retained, including failed identity
controls. No scientific or candidate payload was deleted or moved; the narrow cache cleanup is
reported under Verification.

The reviewed replay binds repo HEAD `35b5d50fd89bbe53c62c166d421613da0ea437de`, implementation
SHA-256 `552399b45c758fef0f790903532c520401e9bcb560844c8bbffd342192008163`, seed
20260823, deterministic CPU settings, upstream Git HEAD
`11ad728f563d8970929e8947a1cf6124ee6303e4`, and exact local upstream-tree snapshot SHA-256
`5e66356d77dac47d2102c827c9b7c26e547913112dd4a4d6c3a68d0e0e7bb36b`. That snapshot contains
19,628 regular files, five symlinks, and pre-existing executable bytecode, so it is explicitly a
local-advisory fingerprint, not a source-only authority snapshot or promotion surface. `upstream/`
was not altered.

## Mechanism-gate table

| gate | status | exact receipt or boundary |
|---|---|---|
| joint optimization against both frozen scorers | **PASS** | `checkpoints/stage_20_joint_smoke_end_step_0001.pt`, 1,043,261 B, SHA `53a17a99...f7f4` |
| MF1 boundary/margin consumed during training | **PASS** | `retained/source_pair_0000/selected_cells.npy`, 39,363 / 196,608 cells, SHA `67f47861...6400`; no mask ships |
| compensation solved in-compile against final packet/render | **PASS** | `retained/smoke_pair_0000/carrier_jacobian.float64.npy`, 6x12 Jacobian, SHA `8e4c4daf...9ac5`; 12 / 12 pair-0 coordinates changed |
| carrier re-solved and re-encoded after move | **PASS** | `retained/final_object/carrier_encoder/`; full 600x12 lattice, production `CAP1 -> DX2 -> RR5 -> Brotli q9 lgwin16` |
| real coders on real payloads | **PASS** | retained CPR1, CAP1, packed body, DX2 body, RR5 body, Brotli stream, semantic stream, member, and archives |
| semantic receiver parse-back | **PASS** | WD2S strict state load and repack byte identity, packet SHA `b6bf44f...87f42` |
| carrier receiver parse-back | **PASS** | `retained/final_object/PARSEBACK_TRANSCRIPT.txt`, `codes_exact=true`, SHA `c72dfe20...96bdb` |
| primary + repeat archive | **PASS** | both 179,274 B and SHA `82bda77d...f12a` |
| full decode / n600 score | **NOT-REACHED** | chartered CPU smoke stops before a heavy/full scorer or evaluator row |

The exact production identity control is stronger than semantic equality: freezing DX2's 36-byte
predictor metadata and using its recorded Brotli `quality=9, lgwin=16` reproduced the shipped RR5
body at 22,008 B, SHA `b73eab2c...56e`, and shipped stream at 22,010 B, SHA
`932b979f...9b12`, byte-for-byte.

## Resumability receipt

| stage | path | bytes | SHA-256 |
|---|---|---:|---|
| birth | `checkpoints/stage_00_birth.pt` | 527,328 | `61d2e3b24bb16946cb6eef62f951117dd5357bc2b4d8cba1b73f471968746f3e` |
| periodic step 1 | `checkpoints/stage_10_joint_step_0001.pt` | 1,043,261 | `53a17a99bde8ed600b804a33a449fd599c3098d6c6df3d1c2229831d06b4f7f4` |
| stage end | `checkpoints/stage_20_joint_smoke_end_step_0001.pt` | 1,043,261 | `53a17a99bde8ed600b804a33a449fd599c3098d6c6df3d1c2229831d06b4f7f4` |

The periodic and stage-end payloads are equal because this sealed stage contains exactly one step;
they are preserved under distinct stage names. Each carries live weights, EMA shadow, optimizer,
Python/NumPy/Torch RNG states, source hashes, retained-subset hashes, history, and resume source. The
deployed packet uses the EMA shadow. Writes use temporary file plus rename and never overwrite prior
stages.

## Bounded smoke measurement

SCOPE reduction: pair 0 only; one optimizer step; CPU float32; single-FiLM flattened W96 only; one
bounded int12 Gauss-Newton carrier solve on pair 0; the unchanged 599 carrier rows are nevertheless
passed through the real full-lattice coder. This is a legal scope reduction, not a mechanism
reduction.

| row | d_seg | d_pose | archive B used in arithmetic | S from components | status |
|---|---:|---:|---:|---:|---|
| exact DX2 source pair | 0.000396728515625 | 0.000010965180990751833 | 180,368 | 0.17024397523303644 | **MEASURED n1 CPU** |
| untrained RJ1 W96 packet | 0.0004933674936182797 | 0.006175666581839323 | 179,274 | 0.41721683211148547 | **MEASURED n1 CPU** |
| one-step EMA packet, before compensation | 0.0004933674936182797 | 0.006161967292428017 | 179,274 | 0.416941049315267 | **MEASURED n1 CPU** |
| final packet + admitted carrier solve | 0.0004933674936182797 | 0.003397040069103241 | 179,274 | **0.3530185565980673** | **MEASURED n1 CPU** |
| n600 d_seg / d_pose | **UNMEASURED** | **UNMEASURED** | 179,274 | **UNMEASURED** | no scorer claim |
| contest CPU / contest CUDA | **UNMEASURED** | **UNMEASURED** | 179,274 | **UNMEASURED** | no authority row |

Every S above is recomputed as
`25*B/37,545,489 + 100*d_seg + sqrt(10*d_pose)`. The final components are rate
`0.11937119796202415`, Seg `0.04933674936182797`, and Pose `0.18431060927421516`.

The optimizer alone recovered **0.22222145924895363%** of the untrained d_pose gap to source.
Optimizer plus fresh exact-object compensation recovered **45.07317276313828%** of that d_pose gap,
or **26.969826231982735%** in nonlinear Pose score currency. Compensation therefore moved the right
way and admitted all 12 coordinates, but did not recover a majority.

## Prior-law prediction adjudication

| prediction clause | measured result | adjudication |
|---|---|---|
| bounded W96 row will not have negative joint delta S | `delta_S = +0.18277458136503086` | **CONFIRMED at n1 scope** |
| in-compile compensation will recover a majority of the refusal | d_pose-gap recovery 45.073%; pose-score-gap recovery 26.970% | **REFUTED at n1 scope** |
| mechanism recovery occurs in the correct direction | precomp d_pose 0.0061619673 -> 0.0033970401 | **CONFIRMED at n1 scope** |
| a retained negative-delta archive fires the campaign route | no negative delta | **NOT FIRED** |

This is narrower and more current than RJ1's superseded 3.51x prose. RF1 later measured the retained
untrained film-W96 object at 2.7749x on a matched n600 macOS-CPU advisory row, with 97.59% of its
damage in Pose. RJ2 does not transfer RF1's n600 value into this n1 instrument; it uses RF1 only to
correct the historical premise and to keep the negative verdict instance-scoped.

## Both byte currencies and AR1B decomposition

The fixed exchange is **6.658590e-07 S/B**, cited from
`ddm_tx1_toolbox_crosswalk_20260819.md` section 0. It is not re-derived here.

| currency | source DX2 | RJ2 candidate | result |
|---|---:|---:|---|
| current-distortion strict cap | 137,986 B | 179,274 B | candidate remains **41,288 B over**; it removed 1,094 / 42,382 B |
| zero-distortion byte condition | shed 150 B | shed 1,094 B | byte condition passes, but distortion is explicitly nonzero and worse; no score admission |
| rate credit | 0 | 1,094 x 6.658590e-07 | **0.000728449746 S** |

AR1B's content-pinned, uncommitted-at-charter decomposition closes exactly:

| residue | DX2 B | RJ2 B | delta B |
|---|---:|---:|---:|
| renderer stream | 30,856 | 29,762 | -1,094 |
| carrier stream | 22,010 | 22,010 | 0; contents changed |
| HPAC model | 13,515 | 13,515 | 0 |
| compact residual | 96 | 96 | 0 |
| framing | 114 | 114 | 0 |
| token stream | 113,777 | 113,777 | 0 |
| **archive total** | **180,368** | **179,274** | **-1,094** |

## Object-change re-pricing table

`RE-PRICED` means the old number cannot transfer to the moved model/carrier. It does not mean the leg
is automatically reopened or credited.

| leg | object status | what changed | run disposition | boundary |
|---|---|---|---|---|
| QS2 | **RE-PRICED** | renderer model and pair-0 carrier coordinates changed; categorical field unchanged | **FIRED n1** | fresh exact-object solve reduced d_pose but left positive joint delta; no n600 transfer |
| RE1 | **RE-PRICED** | renderer model and carrier changed; base categorical field did not | **FOLDED INTO S1 STAGE B** | old zero-byte scorer effect is not banked on this object |
| EC1 | **RE-PRICED** | renderer model changed; no event-coordinate field was emitted | **FOLDED INTO S1 STAGE B** | B/H/W must be measured on a complete moved field before any reopening |
| LD1 | **RE-PRICED** | renderer model changed; token coder and current field stayed unchanged | **FOLDED INTO S1 STAGE B** | old RC64 rate closure stands; only a new moved field can create a new distortion object |
| AE1 | **UNCHANGED** | HPAC probability object and categorical field are byte-identical | **CLOSED** | anti-predicted excess result transfers |
| OE1 | **UNCHANGED** | HPAC probability object and categorical field are byte-identical | **CLOSED** | escape-member result transfers |
| HPAC sharp-optimum rows | **UNCHANGED** | HPAC model and categorical field are byte-identical | **CLOSED** | renderer-only movement does not reopen model-rate perturbations |

The renderer and pair-0 carrier have changed enough to invalidate scorer-effect prices tied to the
old render, but not the unchanged-field HPAC/coder prices. That is the exact boundary. RJ2 does not
assert a surviving route from a positive n1 row.

## RECALL EVIDENCE

**Queries and surfaces.** The full-corpus pass searched repo research memos, charters, arm receipts,
the canonical indexes/DAG, task and harness surfaces, and the 449-entry canonical-equation registry
for `RJ2`, `joint renderer`, `W96`, `film`, `moved object`, `token re-encode`, `JG2`, `QS2`, `QS4`,
`QS5`, `RE1`, `EC1`, `LD1`, `AE1`, `OE1`, `HPAC`, `carrier predictor`, `CAP1`, `RR5`, `DX2`,
`Brotli`, `resume`, and `#1270`.

**Findings beyond the charter's seeds and their effect.** RF1 supplied the later measured 2.7749x,
97.59%-Pose W96 instance and prevented reuse of RJ1's withdrawn 3.51x premise. S1, the live #1270
successor, proved that the current interfaces lacked an RJ1-initialized trainer, a moved-field
producer, and a final-object compensation adapter; RJ2 now supplies a reviewed candidate for the
first and third mechanisms on DX2, but not the GB1 port or moved n600 token field. AP1 supplied the
exact fixed-predictor carrier encoder law and prevented a generic predictor refit from masquerading
as the shipped object. The retained DX2 result supplied the actual outer coder tuple `q9, lgwin16`;
using assumed q11 or lgwin24 produced valid but non-identical streams and was refused. The canonical
equations added no current-object transfer license. These findings changed the build from a generic
CAP1 recode into an identity-controlled production recode and changed the follow-on from a standalone
RJ2 scorer fire into an S1 interface handoff.

**STORES CONSULTED:** repo `.omx/research/`, charters, indexes, DAG, task/harness and equation
surfaces; `/Volumes/APDataStore/pact/ddm_dx2/r7/`;
`/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1/` read-only;
`/Volumes/APDataStore/pact/ddm_mf1_manufactured_seg_repair/measurement_v3/`;
`experiments/results/mlx_fleet_gt_cache/gt_n600.npz`; local read-only `upstream/` scorer weights;
RJ2 output `/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change/`; Metal store: none;
Modal store: none.

## Verification

```text
.venv/bin/python -m pytest -q src/tac/tests/test_ddm_rj2_joint_renderer_object_change.py
10 passed

.venv/bin/python -m pytest -q src/tac/tests/test_ddm_rj2_joint_renderer_object_change.py \
  src/tac/tests/test_ddm_dx2_cabac_receiver_fold.py \
  src/tac/tests/test_ddm_rr5_rider_apply.py \
  experiments/tests/test_ddm_sa2_compensated_semantic_edit.py
84 passed

.venv/bin/ruff check experiments/ddm_rj2_joint_renderer_object_change.py \
  src/tac/tests/test_ddm_rj2_joint_renderer_object_change.py
All checks passed

tools/review_tracker.py: rj2_final_pass1 = 41 entities reviewed;
rj2_final_pass2 = 41 entities reviewed
```

The bounded smoke ran from `--resume-from checkpoints/stage_00_birth.pt --max-steps 1` and completed
with `MECHANISM_COMPLETE_ENGINEERING_SMOKE`. The candidate runtime was copied atomically without the
source archive, then sealed against the RJ2 archive. The stale first runtime copy and every failed
identity-control payload remain retained for diagnosis; they are not candidates.
Repeating the completed resume command returned a byte-identical `SMOKE_RESULT.json`, SHA
`db7bed77238190627dc7e34bde06a18575958d304cccde5d3cf1f290a668e0f8`, without relaunching compute.
The custody suite then exposed host bytecode/AppleDouble residue created by dynamic imports. Only
those enumerated trivial cache/metadata files were removed from the DX2 source and RJ2 copied
runtimes; source modules, archives, and scientific payloads were untouched. Bytecode suppression,
copy-without-xattrs, and automatic copied-runtime cache cleanup now prevent recurrence.

The repository-wide development preflight remains red on eight pre-existing gate families outside
RJ2's owned files: one state-writer finding, one custody finding, one bare-write finding, 25
codebase-drift launchers, one AGENTS terminal-claim finding, 124 historical landing memos, two
unregistered `lane_contingency` findings under `ddm_ma2`, and five substrate score-aware findings.
Individual non-strict listings did not identify either RJ2 Python file or this memo. This bounded
negative is only for the enumerated preflight output; it does not certify the rest of the worktree.

## What is and is not concluded

- **Concluded:** the exact requested n1 mechanism is executable, resumable, receiver-closed, and
  payload-retaining on the DX2 object.
- **Concluded:** the bounded candidate is not a pointer-move row; +0.18277458 S overwhelms its
  0.00072845 S rate credit.
- **Concluded:** fresh final-object carrier compensation helps materially but recovered less than a
  majority on pair 0.
- **Not concluded:** trained film-W96 is dead as an instance at longer budgets, on stratified pairs,
  on GB1, or as a family.
- **Not concluded:** moved-field token re-encoding, RE1, EC1, or LD1 fails on the final S1 object.
- **Not concluded:** any n600 or contest score for the 179,274 B archive.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER / S1-STAGE-A-ADAPTER`; **owner:** MAIN-designated WD3/S1 implementer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_a/`; **fire trigger:** RJ1 custody is reissued coherently and reviewed adapters port RJ2's initializer load, both-scorer objective, EMA/resume schema, and packet build from DX2 to the exact GB1 body with both registered seeds.
- **Disposition:** `QUEUED-BEHIND-STAGE-A / MOVED-FIELD-AND-JG2`; **owner:** MAIN-designated moved-field producer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_b/`; **fire trigger:** stage A retains an n600 moved renderer/runtime and an explicit receiver-consumed moved token field; then run JG2 control plus real re-encode and re-price RE1, EC1, and LD1 only on that object.
- **Disposition:** `QUEUED-BEHIND-STAGE-B / EXACT-OBJECT-COMPENSATION`; **owner:** MAIN-designated QS5/RJ2 implementer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_c/`; **fire trigger:** stage B fingerprints the final archive, runtime, realized frame-1 field, and Pose6 targets; then extend RJ2's retained per-pair Jacobian/solve/recode/parseback mechanism across all required pairs.
- **Disposition:** `QUEUED-BEHIND-RECEIVER-AND-BYTE-GATES`; **owner:** MAIN sole n600 scorer-lane router; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/admission/`; **fire trigger:** stages A-C are receiver-closed and repeat-identical, every payload SHA verifies, scorer chunks are at most 120 pairs, and exact byte arithmetic gives a plausible negative composed delta before one authority fire.

## LIVE-HYPOTHESES

- A longer, multi-pair film-W96 solve can recover more pose than this one-step pair-0 smoke; it is plausible because fresh compensation cut d_pose from 0.006162 to 0.003397 and the optimizer was allowed only one update.
- Re-encoding a genuinely moved token field may expose conditional structure absent from the unchanged DX2 field; it is plausible because the renderer changes the render-to-token object while this smoke held all 117,964,800 categorical symbols fixed.
- Porting the exact adapter to GB1 can close S1's missing initializer and compensation interfaces without inventing new mechanisms; it is plausible because GB1 is a lossless corrector-family extension of the same DX2 decoded field, though byte/container bindings must be re-proved.
- RE1, EC1, and LD1 may have different scorer-effect prices on a fully moved field; it is plausible because their old distortion prices were measured on the old renderer, but no credit exists until the moved field is real and receiver-consumed.

## DEAD-ENDS

- Generic CAP1 predictor refitting is closed for exact-object identity: it decoded correctly but changed the production carrier body and stream.
- Applying RR5 before DX2 is closed: receiver restoration is RR5 then DX2, so the encoder must apply DX2 then RR5.
- Brotli q11 is closed for this DX2 carrier, and q9 with lgwin24 is also closed: both produced valid round trips but not the shipped stream; the source-pinned tuple is q9 with lgwin16.
- Reusing a runtime copy that still contains DX2's source archive is closed: the immutable RJ1 helper correctly refuses to overwrite it; the RJ2 copy excludes the archive atomically before sealing.
- Treating the 1,094-byte renderer cut as an admission is closed by measured n1 arithmetic: its 0.00072845 S credit is dominated by +0.18277458 S joint damage.
- Re-firing AE1, OE1, or unchanged-field HPAC perturbations is closed on this renderer-only move because neither their categorical field nor probability model changed.

Own-vehicle frontier unchanged: **S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4, n600]`**,
GB1 archive SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.
