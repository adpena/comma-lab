# DDM VD1 — one-job n600-authority EC1 event validator

**Status: `READY_TO_FIRE`, not fired.** The validator is implemented, locally custody-tested, and
pinned for one MAIN-owned Modal T4 dispatch. The sole n600 scorer lane is still owned by the live
ps135 gen3 pose re-solve, so VD1 did not claim or dispatch a GPU job. No scorer ran locally, no event
has an n600 result yet, no candidate was composed, and the frontier did not move.

## What was built

`experiments/ddm_vd1_modal_batch_event_validator.py` packages the exact CP135 archive, a filtered
adapted runtime, and the sealed 200-event EC1 store into deterministic SHA-bound inputs. It opens the
governed Modal claim, runs the fleet-wide single-flight guard immediately before `.spawn()`, records
the call ID fail-closed, and provides a dedicated recovery path that does not mislabel an
affected-pair delta job as a contest score row.

`experiments/ddm_vd1_batch_event_validator_worker.py` runs only under
`/opt/upstream-locked-venv` on a T4. In one job it:

1. parses the CP135 archive and decodes the 600x384x512 C1 semantic-token plane once;
2. reconstructs the exact shipped semantic renderer, pose carrier, and frame-0 selector;
3. uses the canonical upstream n600 video list, CUDA DALI decode, weights, preprocessing, SegNet
   argmax, and PoseNet first-six outputs;
4. retains all n600 ground-truth batches, while scoring only the unique pairs touched by the K events;
5. caches each affected base pair once, then copies its C1 token frame in memory, applies one EC1
   event, rerenders only the affected semantic frame, and scores that pair;
6. emits exact singleton `delta_flips_candidate_minus_base`, `net_flip_gain_base_minus_candidate`,
   `delta_d_pose_pair`, and `delta_d_pose_global_n600=delta_d_pose_pair/600` rows; and
7. writes immutable stage checkpoints and commits the Modal volume every 20 seconds while the locked
   worker runs.

The persisted volume keeps the uploaded archive/runtime/events, extracted runtime and RC64 library,
decoded token plane, semantic weights, carrier tensors, selector, every materialized n600 GT batch,
affected GT/base scorer inputs and outputs, and every event's raw payload, indices, candidate token
frame, rendered camera frame, pair, scorer inputs, logits/outputs, argmax, and deltas. Runtime bundling
rejects `__pycache__`, `.pyc`, `._*`, `.DS_Store`, `.git`, and the separately transported archive.

## K arithmetic

These are derived from measured CP135/JS7 T4 components, not from a VD1 remote run:

| term | seconds | status |
|---|---:|---|
| conservative fixed charge | 393.566 | measured prior full decode plus full scorer wall time; deliberately overcharges the affected-pair path |
| measured full 600-master render | 33.300 | prior T4 receipt |
| measured full 600-pair scorer | 39.405 | prior exact T4 receipt |
| raw per-pair render+score mean | 0.121175 | derived `(33.300+39.405)/600` |
| charged per event | 1.211750 | derived with 10x safety factor |
| fixed + K=200 | 635.916 | derived |
| storage/runtime reserve | 300.000 | preregistered safety reserve |
| projected K=200 total | **935.916** | derived, 52.0% of the 1,800 s limit |

With the reserve, `floor((1800-393.566-300)/1.21175) = 913`, so the full **K=200** EC1 alphabet fits.
The implemented fallback remains fail-closed: if the measured constants change and requested K
exceeds the bound, the wrapper refuses; smaller K is selected by JO1's +3 B rate ordering.

The local scorer-free request build measured these retained transport objects on this worktree:

| object | bytes | axis |
|---|---:|---|
| exact CP135 `archive.zip` | 186,252 | existing `[contest-CUDA T4, n600]` custody object, SHA `6eb1a3b7...` |
| filtered 24-file runtime bundle | 50,617 | `[macOS-CPU scorer-free bundle/custody]` |
| full 200-event bundle | 47,570 | `[macOS-CPU scorer-free bundle/custody]` |

Transport sizes can change if the source files change before MAIN dispatch; the wrapper hashes and
records the bytes it actually sends.

## Downstream selection rule — documented, not run

An isolated event is eligible only when:

- its realized n600 net flip gain is positive; and
- `delta_d_pose_pair/600 <= 1.3e-7/44`, equivalently
  `delta_d_pose_pair <= 1.7727272727272727e-6` and
  `delta_d_pose_global_n600 <= 2.9545454545454543e-9`.

The validator reports a deterministic additive-singleton prefilter, explicitly marked
`selection_only_not_composed` and `singleton_interactions_unmeasured`. MAIN must resolve overlaps and
interactions, materialize one JO1 direct-token object, prove the resulting full archive is no more
than +3 B versus CP135, and buy the second/final exact T4 row. Neither singleton sums nor the prefilter
are composition authority.

The preregistered falsifier is unchanged. The six-event JO1 exact row recomputes to
`S=0.1621711682636563`, or `+0.00021602998541453422` versus CP135. If even the optimistic additive
n600 singleton projection is below 0.000216 in Seg score gain, EC1 generation 1 is exhausted at this
formulation scope. Route the next proposal generator to grammar-v2 targets informed by LC1's measured
5,557 introduced Lane-over-Road pixels. If the projection clears the bar, that only authorizes the
retained composition and final exact row; it does not prove the composition will clear it.

## Exact MAIN command

Fire only after the live ps135 scorer claim is terminal, claim reconciliation shows no other Modal
work, and the release preflight is adjudicated:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/ddm_vd1_modal_batch_event_validator.py::main \
  --archive /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip \
  --runtime /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime \
  --event-store /Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200 \
  --jo1-analysis /Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json \
  --output-dir .omx/state/ddm_vd1_modal_batch_event_validator/ddm_vd1_20260812 \
  --k 200 --run-id ddm_vd1_20260812 --resume-from ddm_vd1_20260812 \
  --lane-id ddm_vd1_modal_batch_event_validator \
  --instance-job-id modal:ddm_vd1_20260812 --claim-agent codex:ddm_vd1 \
  --detach --provider-detach-ack
```

Harvest with:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/ddm_vd1_modal_batch_event_validator.py recover \
  --output-dir .omx/state/ddm_vd1_modal_batch_event_validator/ddm_vd1_20260812
```

Then download complete retained custody with the `volume_download_command` in the harvested result.
The volume is `comma-ddm-vd1-event-validator-retained`, path `ddm_vd1_20260812/`.

## RECALL EVIDENCE

Before implementation, `tools/corpus_query.py` searched all seven durable stores: research 8,461,
equations 886, memory 2,114, DAG 915, council 297, tasks 531, and docs 96. Exact queries were:

- `EC1 event coordinate`
- `JS7 exact event overlay`
- `JO1 six event composition`
- `Lane over Road 5557`

Direct bounded content searches also covered `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG/FEED,
task/state ledgers, `main_hot_state.md`, the current dispatch claims, canonical equations, the exact
receiver, upstream evaluator/scorer modules, and Modal auth-eval/single-flight/call-ledger precedents.

Beyond the charter seeds, recall found that the JO1 six-event row had now completed at exact
contest-CUDA authority (`S=0.1621711682636563`, 186,253 B), confirming the charter's rounded +0.000216
damage scale. It also found that all 200 direct events cost only +3 B as one probability object, so
the validator keeps the entire alphabet instead of repricing sparse packet bytes. JS7's exact row
fixed the pose-budget interpretation: the `1.3e-7` stack budget is a global n600 `d_pose` budget, so
the worker reports both pair and divided-by-600 deltas. LC1's 5,557-pixel Lane-over-Road failure
remains a generation-2 target, not a generation-1 selection prior.

No existing one-job exact affected-pair Modal validator was found in those bounded scopes. That scoped
absence changed the plan from adapting the n32 transported-input harness to rebuilding exact base and
GT scorer inputs through the frozen upstream path on the T4.

Sources are SHA-bound by the local request; primary recall anchors at landing are:

- `.omx/research/ddm_ec1_event_coordinate_producer_20260812.md`, SHA `341236f4...`;
- `.omx/research/ddm_js7_exact_row_verdict_20260812.md`, SHA `e887b217...`;
- `.omx/research/ddm_jo1_joint_probability_object_20260812.md`, SHA `2a682057...`; and
- `.omx/research/ddm_lc1_20260805/LC1_RECEIPT.md`, SHA `d02c7cfa...`.

## Validation and boundaries

- Focused tests: 9 passed.
- Ruff: all three Python files passed.
- AST parse and `git diff --check`: passed.
- P0 retention audit: 3/3 Python files examined, zero measure-and-discard findings.
- Review tracker: two post-fix `mark-file --status reviewed` passes for all three Python files.
- Deterministic local request rehearsal: 200/200 unique event payloads retained; K=200 census mode;
  filtered runtime contains 24 files and no generated residue.
- Modal/scorer: **not run**. T4 runtime, actual VD1 wall time, all singleton deltas, selection, archive
  composition, and final exact score remain unmeasured.

The developer preflight was 17/25 green. Its eight red gates are existing shared-repository findings:
one strict-load state writer, one authoritative-tag custody site, 25 legacy codebase-drift launch
surfaces, an AGENTS terminal-claim documentation gap, 124 old landing memos, eight old/unrelated lane
references, 56 substrate scorer-contract files, and 21 substrate pose defaults. Targeted adjudication
found no VD1 path in any red denominator. No gate was weakened or waived. This is therefore not a
codebase-wide green or release-preflight claim; MAIN must rerun/adjudicate the release preflight at the
fire boundary.

Unified-Lagrangian wire-in (Catalog #125): sensitivity-map N/A — validates sealed events; Pareto N/A
— emits rows but does not promote them; bit-allocator N/A — no allocation decision here;
cathedral-autopilot ACTIVE — the explicit scorer-free-to-MAIN fire gate is the dispatch hook;
continual-learning N/A — no score row or posterior update; probe-disambiguator ACTIVE — the retained
singleton table resolves generation-1 admission versus the predeclared generation-2 falsifier.

## Disposition

- **QUEUED-WITH-A-FIRE-ORDER:** one K=200 validator dispatch. Owner: MAIN scorer-lane router.
  Consumer store: Modal volume `comma-ddm-vd1-event-validator-retained/ddm_vd1_20260812/` plus local
  harvest `.omx/state/ddm_vd1_modal_batch_event_validator/ddm_vd1_20260812/`. Fire trigger: ps135 and
  every other Modal/scorer claim are terminal, single-flight passes, and release-preflight findings are
  adjudicated.
- **QUEUED-BEHIND-VALIDATOR:** one composed +<=3 B exact row. Owner: MAIN JO1 composer and exact-row
  router. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/final_t4/`. Fire trigger:
  harvested singleton rows produce an interaction-safe selected object whose optimistic gain clears
  0.000216 and whose retained archive/runtime pass exact custody checks.
- **QUEUED-WITH-A-FIRE-ORDER behind the falsifier:** EC1 generation-2 grammar. Owner: future
  MAIN-routed EC1 producer.
  Consumer store: `/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/gen2_lane_over_road/`. Fire trigger:
  the harvested generation-1 optimistic n600 projection is below 0.000216; seed targets from LC1's
  Lane-over-Road 5,557-pixel error stratum.

Effective frontier remains **CP135 S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**.
Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.
