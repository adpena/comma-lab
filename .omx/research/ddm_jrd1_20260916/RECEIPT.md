# ddm_jrd1 — partial byte preflight; joint exchange is unmeasured

`[no-triality] [p0-ledger-ok]` · `research_only=true` · no score claim.
Lane: `ddm_jrd1_joint_rate_aware_descent_step0_exchange_probe_20260916`.

The exact pointer did not move. The requested three-row scientific probe is
**PARTIAL / QUEUED-WITH-A-FIRE-ORDER**, not complete and not a negative result.
The common contract says: “If you do NOT [own the scorer slot], do byte-only /
scorer-free work and QUEUE the scorer step.” This charter does not assign a
scorer slot, and pd5 has an active host claim. No scorer job was launched.
The user was asked about the missing assignment; no answer has been received.

The work completed here is real byte preparation: move-52 source custody,
parser/null reconstruction, the fixed random sample, and current renderer-codec
controls. The full-stream coder control completed successfully in **1,742.654096 s**:
both real encoders reproduced the **119,097 B** stream and **179,332 B** archive
byte-for-byte over all 600 planes. `TAIL_CONTROL.json` is the harvested result,
SHA-256 `7666c1059373b493cfc05c2dc9c943f4d4e93cf383b6efad045b96a898480b6d`.
This is the canonical pricer's known-symbol loop, not a new cold video decode or
a scorer result. Its recorded RNG seed is 20260912; the sample/grid seed is
20260916. No unmeasured result is substituted with its expected value.

## Requested exchange rows

| Charter row | measured pairs / required | median S/B | IQR S/B | median/IQR coded bits | status |
|---|---:|---|---|---|---|
| Field-only single/2-token control | 0 / 24 | unmeasured | unmeasured | unmeasured | scorer queued |
| Renderer-only, held field | 0 / 24 | unmeasured | unmeasured | unmeasured | gradients and scorer queued |
| Joint field + renderer, resolved pose | 0 / 24 | unmeasured | unmeasured | unmeasured | full mechanism not implemented |
| Global collateral for each direction | 0 / 600 | unmeasured | unmeasured | not applicable | scorer queued |

Joint/best-single ratio: **undefined, no paired measurements**. Gate: **UNMEASURED**;
neither the >=2x success gate nor the <1.5x falsifier was tested. No claim that
this operating point or the joint formulation is finished is admissible.

## MEASURED byte instrument

Axis for all new byte measurements:
`[macOS-CPU advisory / scorer-free exact byte measurement]`.
The bytes are physical; this axis conveys no Seg/Pose or contest score authority.

The archive, seal, and all four memo SHA prefixes in the charter match their
full measured hashes in `PROVENANCE.json`. HEAD at start is
`1c1aba7cecfb77af068d9203666e2dd8c84d4a52`, ahead of charter HEAD `59bcc5a9f`;
the pinned archive remains unchanged. The source runtime was copied into the
owned APDataStore directory, and all copied runtime files were hash-checked.

| Parsed section | physical B |
|---|---:|
| HPAC | 11,629 |
| renderer, compressed SM1S/CK2 member | 29,862 |
| carrier | 18,470 |
| token tail, including 96 B prefix and 64 B rider | 119,257 |
| RX1 header | 14 |
| ZIP overhead | 100 |
| **archive** | **179,332** |

The actual token stream is **119,097 B**. Its share of the archive is
66.41146%; the nonstream remainder is 60,235 B. The charter's approximate
60,135 B remainder excludes the 100 B ZIP wrapper. `PARSER_INPUTS.json` carries
the complete section hashes and exact field pin.

The current semantic coder is **SM1S**, unlike rw1's older RC1S surface. Using
the shipped receiver's SM3R walker, frozen fp16 scales, SM1S encode/restore,
CK2 transform and the byte-identical Brotli setting **q10/lgwin16**, the null
and six seeded single-code actions produced:

| Instrument action (NOT gradient-selected) | archive delta B | coded delta bits | archive SHA-256 |
|---|---:|---:|---|
| null | 0 | 0 | `ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e` |
| head code −1 | +10 | +80 | `5607e9d12d6bf6211ea72d9134e86b3b66a07569686d1adf7f33bb7febfe57a1` |
| head code +1 | +25 | +200 | `5d0d5084859fed76653283e7c42351ca685e2c4e749e190ba8118779245ebce3` |
| blocks.3.dw code −1 | −3 | −24 | `383add0211e5033477863be70d41f8cd5f22760d97dcda660b5b2eb9d24b3884` |
| blocks.3.dw code +1 | +3 | +24 | `263b6f31fbc3a655b7edb43a5adf93f5297127bd0f93154cf2f489668d3c96fb` |
| blocks.3.pw code −1 | +29 | +232 | `9f4964ce77d771f90faa005c59b0052410d9ee25c38d7c7955a15d4db2715c85` |
| blocks.3.pw code +1 | −4 | −32 | `08fdff1a9ecae65c2986e57c9a21bd006528cfb7a97e0b31f5e6c9793babfd1d` |

All 14 archives were independently re-hashed after the producer completed;
all seven twins agree. Every body restores exactly and the unrelated sections
stay byte-identical. **These six steps are only positive instrumentation
controls.** Their small byte savings are not score improvements; gradients,
rendering, resolved pose, global damage and all S/B quotients are unmeasured.
No renderer rate law is fitted to six probes.

## DERIVED arithmetic, from move 52's recorded components

`100*0.00010304 + sqrt(10*4.21e-6) + 25*179332/37545489`
reproduces **0.13620226906030858**. The rate exchange is
**6.658589531221714e-7 S/B**. At held recorded distortion, the continuous
sub-0.12 boundary is **154,999.11540884574 B**; the strict integer ceiling is
**154,999 B**, requiring a **24,333 B** cut. This corrects the hot-state
154,806 B figure; no source state or pointer was overwritten. These are
derivations from a prior exact row, not fresh scorer measurements.

## Fixed sample and scope boundaries

PCG64 seed **20260916**, NumPy 1.26.4. Pool definition: 156 unique pair IDs in
pd4's `sheets/priced_rows.jsonl`, independently corroborated by pd5's charter.
The source pool is copied and pinned at
`67c779eb5d1db645e27693b935cf975606c22695f5c9fca4a2469f221502f944`.

* Pool draw: **250, 398, 296, 236, 80, 385, 214, 382, 208, 583, 53, 406**.
* Complement draw: **547, 342, 127, 319, 502, 137, 376, 32, 183, 33, 134, 536**.

`SAMPLE.json` preserves the entire 156/444 partition and the selected token
planes' SHA. “Untouched remainder” is interpreted as outside pd4's priced
pool, not never touched in the campaign. The draw precedes any jrd1 scorer
outcome. The balanced K24 median describes this design, not the 600-pair
population median without weighting. Global collateral must still use all 600.

Unresolved protocol details are explicit in `FIRE_ORDER.md`: pd4's subset
12.0 bits/token control does not identify the charter's approximate per-pair
“1/12 bar per bit-equivalent” statistic/tolerance; a literal zero-net-byte
step has no finite S/B quotient; K24 alone falls below the common contract's
n>=32 requirement for banking a sampled negative. None was silently resolved
by invented numbers, denominator epsilons, or an unauthorized larger run.

## RECALL EVIDENCE

`RECALL_SEARCHES.json` retains exact argv, result paths, byte counts, hashes and
scope counts. Search surfaces were `.omx/research/` memo contents and arm
receipts; all **490** records returned by
`.venv/bin/python tools/list_canonical_equations.py --json`; the canonical
research index and primary `sub015_DAG_*` FEED document; design/SPEC filenames
with content matching; and `.omx/state/canonical_task_status.jsonl`.
Queries included `joint.{0,50}(renderer|field)`,
`renderer.{0,50}collateral`, `selection.on.{0,15}price`, `UNION.{0,8}SUM`,
`rate.aware`, and the four seed arm names. The memo search found 210 matching
files; design search 35; DAG/index search was explicitly bounded to eight hits
per file; a further search over all `sub015_DAG_*` and
`CANONICAL_RESEARCH_INDEX*` file globs returned five hits with a four-hit-per-file
cap; the task query returned three S1 joint-renderer lifecycle rows.
This is a full-corpus search with bounded reading, not a claim to have read
every matching document. A memory-registry query found no relevant hit.

Findings **beyond the charter seeds**, and their consequences:

* **RJ2**, `ddm_rj2_joint_renderer_object_change_20260823.md`: real n1 joint
  renderer/pose smoke on DX2, fixed field, different object. **JF1**,
  `ddm_jf1_joint_field_model_refit_20260823.md`: field/prior refit with a
  +7,554 B null-stream deficit at epoch 2. These are relevant joint precedents,
  but neither measures this move-52 three-row direction. They preclude a
  blanket claim that no joint work ever existed and make null reproduction
  mandatory before interpreting any candidate.
* **XR1**, `ddm_xr1_exchange_ratio_noise_floor_20260903.md`, and canonical
  `exchange_ratio_noise_floor_v1`: identical input re-encodes had zero physical
  byte spread. Exchange uncertainty is pair/object-dependent. Do not present
  pd4's 34.8 B across-edit container spread as random noise on a fixed archive.
  Current twin byte equality is the relevant physical control.
* **JC1**, `ddm_jc1_afr_rc64_joint_redesign_20260901.md`: field/context/prior
  co-design remains distinct from renamed exogenous token thinning; actual
  rendered Seg/Pose and full causal re-encoding are required. Jrd1 keeps its
  narrower charter variables, and claims no prior-training result.
* **PC2**, `ddm_pc2_pose_carrier_live_remainder_20260826.md`, plus ledger
  `ddm_pc2_s1_joint_renderer`: S1E closed the Film-W96 width-distillation family.
  It is not a closure of this current mixed-int4 renderer/field step. No replay
  of that family is queued.
* Canonical `model_section_edit_container_break_fee_v1`,
  `renderer_edge_layer_foldback_reach_v1`,
  `compensated_semantic_edit_exchange_v1`,
  `score_atomic_flip_byte_exchange_v1`, and
  `token_edit_composition_subadditive_on_pair_overlap_v1` were inspected from
  the registry. Their object-specific prices and bounds are not transferred;
  they require exact current byte pricing, re-solved pose and union scoring.
* The design/SPEC search surfaced `ddm_jo1_joint_objective_design_20260821.md`;
  its joint objective is a design precedent, not empirical transfer. The DAG
  likewise records measured collateral and nonadditivity on older objects.

## Boundaries and custody

Owned bulk store: **`/Volumes/APDataStore/pact/ddm_jrd1/`**, maximum 3 GiB.
Final measured custody: **728 files, 164,677,657 logical B,
254,410,752 allocated B**; APDataStore free **23,324,917,760 B** at harvest.
`RETENTION_MANIFEST.json` lists every retained path, byte count and SHA-256,
including logs, checkpoints, native builds and macOS sidecars; its SHA is
`3aa4b68667ed24997163e58266648d53c696eb554aa77007f62f19b145211a3c`.
`RENDERER_BYTES.json` SHA is
`fc0b821bd3d6805684e9a3b9fa65029a0c2a55e0408b4f4fbf2127867d9ffd07`.
Payloads and their determinism repeats are kept. No deletion, move, cold-store
shortcut, symlink workaround or storage-reserve reduction was performed.
The byte driver refuses capacity overflow and writes outside its owned root.
The native token coder preserves frame-25 encoder and receiver checkpoints.
The renderer producer has immutable per-action stage receipts and accepts
`--resume-from /Volumes/APDataStore/pact/ddm_jrd1`.

All heavy steps use the charter launcher and named done receipts. The first
launch returned rc 8 because sandboxed `setpriority` was denied **after the
SSD manifest write succeeded**. This was not a filesystem-write refusal.
The documented `--nice-best-effort` launch then required the child itself to
measure `getpriority(PRIO_PROCESS, 0)==0` before doing work; the child logs
prove 0. No in-session heavy fallback was used.

No training, burn, Modal, authorization, fire, packet, scorer weights, scorer
execution, MPS, GT decode, full video materialization, pose solve, or renderer
forward ran. No changes to upstream, contract code, receiver code, shipped
weights, prior, basis, PR/staging trees, sealed trees, pd1–pd5 stores, or Vertigo.
Only the owned archive copies contain the explicit test code mutations.
The byte-only archives carry **unknown distortion** and are not candidates.

The common contract's embedded QO1/macOS frontier is historical and is
superseded by this charter and the verified live move-52 pointer.
Serializer and final retention receipts are separate harvest artifacts, so a
commit failure cannot turn these bytes into a fabricated landing.

No new equation anchor was registered: the charter's exchange observation does
not exist yet. All six production hooks are N/A under `research_only=true`:
there is no sensitivity/allocator result, Pareto constraint, dispatchable
candidate, posterior exchange observation, or scorer interpretation to wire in.
The real remaining obligation is registered in the canonical task ledger.

## NEXT_IF_RESUMED

* **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; executing owner ddm_jrd1 successor;
  consumer store `/Volumes/APDataStore/pact/ddm_jrd1/`; fire trigger explicit
  scorer assignment after pd5/fleet release and pin validation. Implement and
  measure the complete three rows and all-600 collateral under `FIRE_ORDER.md`.

## LIVE-HYPOTHESES

* A simultaneous field/renderer step may compensate boundary jitter more
  efficiently than either coordinate alone. It remains plausible because the
  charter's single-axis closures do not test that compensated direction;
  **jrd1 has not measured it**.
* A nearest grid action can release enough bytes to finance a small token
  change: −3 B and −4 B instrument examples exist. Whether either is useful
  after frozen-scorer damage and resolved pose is entirely untested.

## DEAD-ENDS

* Treating nearest int4 code steps as zero-byte changes is invalid here:
  measured deltas range from −4 B to +29 B under the actual current coder.
* Pricing move 52 by rw1's older RC1S loader is the wrong receiver surface;
  current null byte identity requires SM1S. The old experiment itself was not
  re-run or reclassified.
* The byte controls cannot decide the joint gate. No scientific direction or
  family was closed by this partial run.

composition S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600] (move 52)

<!-- # FORMALIZATION_PENDING: partial apparatus receipt; no exchange result to
     register as an EmpiricalAnchor until the scorer fire order completes -->
