---
schema: ddm_lt1.pr130_levelset_longtail_force_port.v1
date_utc: 2026-08-10
arm: ddm_lt1
base: PR130_CPR1
base_archive_bytes: 191052
base_archive_sha256: 0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd
base_pin_commit: 113b52fdb15318549c063b7bb67aa7f02f253c60
axis: "[scorer-free source and retained-receipt inspection]"
score_claim: false
promotion_eligible: false
pointer_moved: false
---

# DDM LT1 — PR130 long-tail force port

## Conclusion

No level-set force is ready to fire on PR130 today. Five force families have a real PR130 training
consumer and reusable mathematics, but their **RIGHT-STUFF gate is unresolved until `ddm_sd2`
retains the PR130 target→prediction error tensor**, and their **PORTED gate still needs a typed
PR130 loss/telemetry adapter**. Two exact mechanisms are `DEAD-ON-THIS-BASE`: anisotropic
surface tension has no explicit PR130 interface/SDF on which to act, and pair-internal temporal
screw consistency has no second semantic field because PR130 renders frame 1 with the semantic
renderer and frame 0 with the independent pose carrier.

This is not the older finding that PR130 has no training window. FX3 landed a launch-admissible,
EMA-backed, crash-resumable QAT wrapper with complete periodic and stage-boundary checkpoints.
The window exists. What does not yet exist is the PR130-specific force hook, directed-edge provider,
and retained full-population residual decomposition needed to use it honestly.

Routing count: **0 `PORTS-NOW` · 5 `PORTS-AFTER-<prerequisite>` · 2
`DEAD-ON-THIS-BASE`**. No scorer, trainer, archive builder, or Modal job ran in this arm. No payload
was materialized or discarded.

## Authority and source pins

| Object | Source pin | Use here |
|---|---|---|
| PR130 CPR1 base | commit `113b52fdb15318549c063b7bb67aa7f02f253c60`; archive SHA above | Base and unchanged-instance closures |
| Prior force harvest | `.omx/research/ddm_fh1_forces_harvest_20260731.md` @ `72ac061bd32bdf92136ad75a50b4ca78d68de4ca` | Inventory and adaptation method only; TR1 values are not transferred |
| SN1 sided tolerance | `.omx/research/codex_findings_ddm_sn1_segnet_telemetry_asymmetry_20260723_codex.md` @ `89d8ee9e4b119a2840c9ef43df4436d10abf50d9` plus its n600 JSONL | Directed-head-space design constraint |
| Persistence source | `src/tac/boundary_math/persistence_topology_loss.py` @ `83e0df8a4209ea58c3c9750db484517278e9af69` | NumPy/MLX reference and canonical equation |
| Torch long-tail twins | `src/tac/cuda_levelset_training.py` @ `50ce946838c6624feee246848bdcb83bf666e5d4` | Reusable Torch signed-margin, island-birth, area, and persistence math |
| Lane guard | `src/tac/optimization/lane_guard.py` @ `d4a4b3c5412173e2e908f9b3d90748329f8322e9` | Primal-dual pattern and derived-floor method, not its Lane constants |
| Island ladder | `src/tac/witness_curriculum/ladder_homotopy.py` @ `3563b9c9b12f4b006fa00d79d4a6e732d75ec140` | Event-order/state-machine pattern only |
| Pair tension | `src/tac/boundary_math/length_sigma.py` @ `caf747203b4e9933c72189f177e931b998d76b85` | Explicit-interface mechanism and non-transfer proof |
| PR130 training window | `src/tac/pr130_lift/train_semantic_quantized_resumable.py` @ `f43180a76171fccc03fd92fd590466db7c20c56d` | Resumable CE→softplus-margin→expected-flip window |
| PR130 receiver | `src/tac/pr130_runtime/fx1_runtime_tree/inflate.py` @ `6a292a271fc7cbf8f84c72fcdb783708c8a49fb7` | Separate semantic-master and pose-slave paths |

Every source file above was clean in the current working tree when inspected. File-content SHA-256s
were also recorded during the arm; the report relies on the commits because they are the reviewable
source identities.

## RECALL EVIDENCE

The recall pass searched the full `.omx/research/` corpus, source, canonical-equation registry,
research index/DAG surfaces, live task ledgers, and current hot state. Queries included `long-tail`,
`persistence`, `critical nucleus`, `island birth`, `margin`, `UNIWARD`, `head hyperplane`,
`sigma_cc`, `surface tension`, `temporal screw`, `separatrix asymmetry`, `PR130 semantic QAT`,
`resume`, `ddm_sd2`, `ddm_pk2`, `#996`, and `gauge`. The canonical registry was enumerated with
`tools/list_canonical_equations.py --json` and filtered by those mechanisms.

The charter seeds were consumed rather than rebuilt: `ddm_fh1` supplied the force inventory and
cross-vehicle adaptation discipline; SN1 supplied the sided-tolerance receipt. Findings beyond those
seeds materially changed the route:

- FX3 proves that PR130 now has a resumable QAT window, so the five live families are not blocked on
  training existence. They are blocked on a force adapter and PR130 residual aim.
- `cuda_levelset_training.py` already contains differentiable Torch twins for signed margin,
  per-class island birth, area constraint, and persistence topology. The port should reuse these
  equations rather than translate the old MLX trainer wholesale.
- The deployed PR130 receiver proves the semantic renderer writes only `output[2*i+1]` and the
  independent pose carrier writes `output[2*i]`. That closes the exact pair-internal temporal-screw
  mechanism on the unchanged base.
- The original QAT and FX3 wrapper contain only CE, softplus-margin, and expected-flip scientific
  losses. None of the candidate force hooks is secretly already active.
- The current closures remove false escape hatches: #996 closes coder replacement on unchanged
  sections; `ddm_pk2` leaves the incumbent pose carrier selected on its declared n120 surface; and
  gauge commit `113b52fdb1` found only 64 B against a 2,000 B fire bar on its declared 432-row bank.
- `ddm_sg2` found a 20,671/117,964,800 AV-vs-DALI source-target difference with Road participating
  in 89.65% `[macOS-CPU advisory]`. That is a source-control object, not PR130 candidate error.
  It cannot aim any force. This is why `ddm_sd2` retention is a hard prerequisite.

Canonical equations used as constraints, not as transferred efficacy claims, include
`persistence_topology_cldice_betti_island_recall_v1`,
`frozen_scorer_fisher_curvature_margin_colocation_v1`,
`scalar_top1_top2_margin_is_exact_distance_to_flip_v1`,
`separatrix_asymmetry_t_subpixel_boundary_localizer_v1`,
`margin_band_satisficing_threshold_v1`, `junction_young_angle_sigma_fit_v1`, and
`island_topological_charge_conservation_v1`. None has a measured PR130 force-efficacy anchor.

## Directed asymmetry recalled at source

SN1's retained n600 sided-tolerance rows are `[macOS-CPU frozen-SegNet advisory]`,
`score_claim=false`, and scoped to frozen-head boundary samples; pixel-space realization and Pose
collateral are explicitly unproven.

| Ordered side | n600 q10 head-space tolerance d2 | Reverse | Ratio |
|---|---:|---:|---:|
| Road→Lane | 0.02488675814049674 | Lane→Road 0.019704731403216966 | 1.2629838809390603 |
| Undrivable→Lane | 0.1283532461772616 | Lane→Undrivable 0.09603813702714589 | 1.3364820492194847 |

The first pair had 629,474 Road→Lane and 514,023 reverse boundary samples; the second had 1,521
and 1,497. MyCar↔Undrivable had no measured boundary support, so no zero or symmetry assumption is
invented.

These are **not PR130 target→prediction error counts** and not weights to copy. They establish only
the design constraint: `(target=c, prediction=c')` and `(target=c', prediction=c)` must be separate
cells, with separately derived floors, budgets, and telemetry. `ddm_pc2`'s 87.8% Road participation
and 49.2% Road↔Lane share came from another vehicle; `ddm_sg2`'s 89.65% Road participation is the
wrong source-control object. They justify edge decomposition, not a PR130 Road/Lane conclusion.

## Four-gate force table

Class order is the canonical `Road, Lane, Undrivable, Movable, MyCar`. In the table, `c→c'` always
means **target class c, PR130 prediction c'**. `SD2` means the retained PR130 base outputs planned by
`.omx/research/charters/ddm_sd2_pr130_seg_decomposition_runner.md`: full n600 target/prediction
matrix, per-frame mass, per-edge mass, and boundary/interior split, computed from retained argmax.

| Force | RIGHT-STUFF — directed PR130 mass | RIGHT-WAY — PR130 derivation | RIGHT-TIME | PORTED — reuse versus rebuild | Honest route / SD2 application |
|---|---|---|---|---|---|
| **Persistence-preserving loss** | **BLOCKED on SD2.** Eligible only for `c→c'` cells where target connected components or thin support vanish, especially Lane→Road, Movable→Road, or Lane→Undrivable if those cells actually carry mass. Reverse over-paint cells are not persistence erasure. | Reuse soft-clDice plus density-weighted recall on exact-R SegNet logits, but choose target classes and `recall_class_scale` from PR130 component-survival debt. Re-derive skeleton depth, recall scale, warmup, loss-share cap, and gradient cap in the PR130 window. No old 8.9 ratio or v9 weights. | Formation through early QAT, before terminal expected-flip-only polishing; begin from the stage-07 parent or a fresh matched trajectory. A terminal post-hoc loss cannot birth a feature the renderer never learned. | **Reuse:** `persistence_topology_loss_torch` and NumPy authority. **Rebuild:** PR130 adapter, component-survival provider, typed config/checkpoint fields, per-term loss/gnorm and packed-archive validation. | **PORTS-AFTER-SD2-ERASURE-LOCALIZATION-AND-PR130-FORCE-HOOK.** SD2 selects the directed cells and frames; if it finds displacement without component loss, this row folds at instance scope. |
| **Margin / UNIWARD weighting** | **BLOCKED on SD2.** Eligible on boundary-local `c→c'` mass whose realized GT-vs-runner-up margins lie in the PR130 low-margin tail. Every orientation is separate; a Road→Lane budget cannot stand in for Lane→Road. | Primary cost is frozen-head hyperplane distance `m/||w_c-w_c'||`, evaluated through exact R. A UNIWARD/local-variance term may only be a secondary within-edge allocation arm, normalized mean-one and raced against head-distance-only; image texture is not a substitute for reachability. Re-measure PR130 R jitter and derive each oriented satisficing floor from it. | Post-formation QAT: softplus-margin or expected-flip stage, with weights changed only at stage boundaries. It is a budget allocator/cap, not a new finisher that keeps deepening already-safe margins. | **Reuse:** `realized_signed_margin`, frozen-head norms, registry margin laws. **Rebuild:** oriented rival provider, PR130 R-noise receipt, capped allocator, telemetry and typed resume fields. | **PORTS-AFTER-SD2-DIRECTED-BOUNDARY-LOCALIZATION-AND-PR130-MARGIN-PROVIDER.** SD2 supplies edge/frame/boundary aim; retained frames let the governed trainer derive margins without treating the SD2 argmax as logits. |
| **Per-class λ plus critical-nucleus guard** | **BLOCKED on SD2.** A class scalar alone is inadmissible. The guard must target an incoming erasure cell such as Lane→Road separately from an outgoing expansion cell such as Road→Lane, and only when component/area debt—not placement—dominates. | Generalize λ to oriented constraints `g_cc' = E_cc' - B_cc'`, where `E` is the retained directed error level and `B` its derived budget, with dual ascent and complementarity telemetry. Derive budgets from the PR130 base matrix and matched-control noise. Define a critical nucleus from retained component area/thickness and realized signed margin; never copy the level-set P/A threshold. | Formation stage, before or at the first stable birth census. Dual and loss weights update only at declared gates/stage boundaries; the guard must persist in the complete FX3 checkpoint. | **Reuse:** lane-guard primal-dual state pattern, `island_birth_perclass_from_signed_torch`, `area_constraint_torch`. **Rebuild:** 5×5 oriented guard state, PR130 component tracker, non-inertness alarm, typed checkpoint and loss injection. | **PORTS-AFTER-SD2-AREA-VERSUS-ERASURE-SPLIT-AND-PR130-EDGE-GUARD-ADAPTER.** SD2 decides which edge constraints exist; if only placement debt remains, the birth/area branch folds. |
| **Island-birth ladder** | **BLOCKED beyond SD2.** Eligible only if a live PR130 training trajectory shows the same oriented component repeatedly below and then across a birth threshold. A static endpoint component miss is necessary but not sufficient evidence of saddle-node dynamics. | Re-derive the ladder from PR130 checkpoint-to-checkpoint component survival: continuation variable, entry event, dwell, completion gate, hysteresis, and stop are all measured on that trajectory. Compose with the oriented λ/nucleus guard; do not copy `ladder_homotopy.py` constants or the old Lane/Movable schedule. | Earliest formation stage. It must precede margin sharpening, quantization tail polish, and terminal selection; stage-boundary checkpoints are the measurement surface. | **Reuse:** ladder state-machine/event-order pattern and Torch birth primitive. **Rebuild:** PR130 trajectory logger, oriented event detector, typed ladder state, resume tests, packed-stage candidates. | **PORTS-AFTER-PR130-ORIENTED-NUCLEUS-TRAJECTORY.** SD2 first identifies candidate edges/frames; the guard A/B must then produce a retained stage trajectory showing a real threshold crossing. Otherwise this ladder is `FOLDED`, not fired speculatively. |
| **Anisotropic per-class-pair surface tension `sigma_cc'`** | Pair direction matters diagnostically, but the exact force needs an explicit evolving interface and curvature/length pressure. PR130 exposes RGB from a convolutional semantic renderer, not `phi_c`, an SDF, interface length, or MCF. | Copying Young-angle or fragility `sigma` into a generic class weight changes the mechanism and sign. Directed boundary weighting belongs to the margin row above; it is not a surface-tension port. Re-entry would require a new explicit-interface PR130 representation and a fresh PR130 junction/tension derivation. | No action point exists in the unchanged PR130 QAT energy. Decode time is irrelevant to this negative. | **Reusable only after a new base:** `LengthSigma`, Eikonal/length laws, and fitting method. **Missing now:** the state variable and energy term they act on. Coder #996 and the 64 B gauge result do not create one. | **DEAD-ON-THIS-BASE — FORMULATION.** Killing closure: pinned PR130 renderer/trainer has no explicit interface/MCF channel. SD2 can show pair mass but cannot make this force well-typed. |
| **Lane-guard λ primal-dual plus head-hyperplane margin floor** | **BLOCKED on SD2.** Highest-priority live port if one or more directed Lane cells carry material boundary mass. Lane→Road erasure, Road→Lane expansion, Lane→Undrivable, and Undrivable→Lane require distinct budgets/floors; no aggregate “Lane error” guard. | Lift the guard from one Lane scalar to oriented edge constraints. Derive each floor from PR130 exact-R `m/||w_c-w_c'||` samples and each budget from the retained base matrix plus matched-control uncertainty. Keep complementarity, planned-horizon derivation, false-positive/inertness alarms, stage-boundary caps, and a no-op default. | Gate cadence during formation and QAT; engage only after enough retained gates estimate a stable base level. Update λ at gates, not every optimizer step. | **Reuse:** `LaneGuardState`/dual-ascent/horizon and head-distance methods. **Rebuild:** oriented 5×5 state, exact-R evaluator adapter, per-edge born masks, Torch pixel weights, FX3 schema/resume and deterministic tests. All legacy numeric constants are discarded. | **PORTS-AFTER-SD2-DIRECTED-LANE-MASS-AND-PR130-EDGE-GUARD-ADAPTER.** Apply SD2 by ranking Lane-involving edge cells by total error, boundary share, and frame concentration; do not fire if Lane cells are not material. |
| **Temporal screw-consistency** | The original force compares semantic fields for pair frames 0 and 1 on ground-class annuli. On PR130, frame 1 is the semantic master and frame 0 is an independently generated pose carrier; no frame-0 semantic field exists. SD2's per-frame Seg error concerns only the semantic/master frame and cannot aim this missing comparison. | A loss between the semantic master and the pose carrier would force unlike representations together and is fake. Adjacent-master regularization would be a new temporal formulation, not this pair screw, and would need chronology/correspondence proof plus a new joint renderer. No v9 `0.44` or GT-screw weight transfers. | There is no legal point in the unchanged semantic-only QAT graph. It can re-enter only during joint two-frame semantic training, before the two frames split into independent receiver paths. | **Reusable after representation change:** homography/SE(3) transport utilities and temporal DSL. **Missing now:** two co-trained semantic fields and a joint checkpoint/receiver. `ddm_pk2` selected the incumbent independent pose representation on its declared search surface, so a post-hoc pose-side detour is closed. | **DEAD-ON-THIS-BASE — FORMULATION.** Killing closure: PR130 receiver separation plus `ddm_pk2`'s unchanged pose-representation selection. Decode time is not part of the verdict. |

## Routing order once SD2 lands

Apply the SD2 tensor as a decision tree, not as a global ranking copied from another vehicle:

1. Split every non-diagonal cell by boundary/interior and frame contribution. Keep orientations
   separate.
2. Join target components to predicted components. If a directed cell is dominated by lost thin or
   disconnected support, route first to the **oriented lane/edge guard**, then persistence; admit
   the ladder only after checkpoint telemetry proves threshold-crossing dynamics.
3. If components survive but boundaries move, route to **head-hyperplane margin satisficing**. Race
   head-distance-only against head-distance plus within-edge UNIWARD; do not stack them unmeasured.
4. If interior/area expansion dominates, use the oriented λ constraint. Do not apply a persistence
   cure to an over-paint cell.
5. If no directed cell is materially concentrated, fold specialized edge forces and keep the
   matched PR130 control. A different-vehicle Road prior is not enough to fire.

Within each admitted family, the first scientific row is a matched fresh PR130 A/B from the same
stage-07 parent, seed, active-pair order, schedule, exact-R path, EMA policy, and packed receiver.
Retain every periodic/stage checkpoint and every candidate payload. Selection requires full n600
packed semantic evaluation and then a byte-closed archive; contest promotion still requires the
named CPU/CUDA authority owner.

## Consumer binding: ddm_sd2

`ddm_sd2` is the sole first consumer/producer join for this report. Its output is applied through
the table using these required joins:

- `target_class, prediction_class, frame_id` → force orientation and frame queue;
- boundary/interior split → margin/placement versus area/interior routing;
- target-component survival and thickness, derived from retained argmax plus target → persistence
  and critical-nucleus eligibility;
- retained decoded camera frames → exact-R PR130 logit/margin provider inside the later governed
  training job;
- retained PoseNet output → collateral guard for any packed candidate, not a Seg-force weight.

In the searched current corpus, `ddm_sd2` had a charter and a live queue row but no runner or result
artifact yet. Its charter promises retained argmax, decoded frames, Pose outputs, and the directed
matrix, but not a retained logit tensor. That is sufficient: SD2 aims the force; the
PR130 force adapter must recompute and retain logits/margins under its own governed training run.
No scalar displayed `d_seg` may be inverted into counts, and no source-target control may substitute
for candidate argmax.

## Named closures and boundaries

- **Coder axis #996:** unchanged semantic, pose, and HPAC sections are closed to another outer
  memoryless/order-1 coder race; the token ANS win is receiver-blocked and separate. No training
  force here is relabeled as a coder win.
- **Gauge `113b52fdb1`:** 432 declared gauge candidates produced at most 64 full-archive bytes of
  saving versus a 2,000 B fire bar `[macOS-CPU advisory]`; the existing gauge-QAT action is folded
  on that instance/search surface. Gauge rotation is not a substitute for directed residual force.
- **Pose representation `ddm_pk2`:** the unchanged CPR1 carrier remained the best measured n120
  row on its declared surface `[macOS-CPU advisory]`; no losing row advanced to n600. This closes
  a post-hoc pose-carrier detour, not all future joint retraining.
- **Training window:** FX3 is a CPU mechanism proof, not a production force-efficacy result. Its
  full 6,000-step QAT trajectory, full n600 deploy parity, and contest-device candidate evaluation
  remain owed for any new force.
- **No PR130 efficacy measurement:** this arm did not measure whether any long-tail force lowers
  PR130 `d_seg`, bytes, Pose, or exact score. All five live rows are derived routes with explicit
  falsifiers, not adoption claims.
- **Disk and retention:** no experiment bulk or payload was written; research-bulk footprint is
  **0 B**, and nothing was moved or deleted. A 6.6 MiB sparse administrative clone under the named
  `ddm_lt1` scratch path retains the serializer commit after the managed sandbox refused writes to
  the main checkout's Git object store. The main index stayed untouched; this report remains
  untracked there until MAIN lands the retained commit range.

Base unchanged: **PR130 CPR1 S = 0.172141297491896447 @ 191,052 B
`[contest-CUDA, DALI GT, n600]`**. This value is the pinned base, not a measurement by LT1.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / MAIN-COMMIT-LANDING** — owner: MAIN/operator with a writable Git object store; consumer store: main branch; fire trigger: Git object writes are permitted in the main checkout. Fetch `/Users/adpena/Projects/pact/.omx/tmp/codex_worktrees/ddm_lt1_commit_fallback_20260810` branch `main`, then cherry-pick `0269bcab7b1704fafc26c16160928eb2791630ca..FETCH_HEAD`; verify the resulting report blob has the SHA-256 recorded in the final handoff.
- **QUEUED-WITH-A-FIRE-ORDER / SD2-HARVEST-AND-AIM** — owner: MAIN `ddm_sd2` harvest owner; consumer store: `.omx/research/ddm_lt1_levelset_longtail_forces_port_20260810/RESULTS.md`; fire trigger: `ddm_sd2` lands a complete retained n600 PR130 base argmax payload with target→prediction 5×5 matrix, per-frame mass, boundary/interior split, bytes, and SHA-256. Apply the routing tree and select or fold each live force by directed cell.
- **QUEUED-WITH-A-FIRE-ORDER / PR130-DIRECTED-GUARD-PORT** — owner: future PR130 semantic-force builder; consumer store: `.omx/state/codex_arm_queue.next_if_resumed.jsonl`; fire trigger: SD2 shows a material directed boundary or erasure cell and the scorer slot is free. Build the smallest typed FX3-compatible adapter for the selected force, with complete resume state, per-stage retained checkpoints, exact-R margin/component telemetry, and a no-op identity test; do not fire training in the build arm.
- **QUEUED-WITH-A-FIRE-ORDER / MATCHED-PACKED-A-B** — owner: MAIN PR130 semantic optimization and scorer owner; consumer store: `.omx/state/main_hot_state.md` PR130 base row; fire trigger: the selected adapter passes deterministic tests, P0 retention preflight, n600 pack/parse identity for its control, and storage admission. Run a matched control/treatment from the same parent and retain every payload; advance to contest authority only if the full n600 packed candidate has negative total delta S.

## LIVE-HYPOTHESES

- A directed Lane-involving guard is the best first force **if** SD2 finds material Lane→Road or
  Road→Lane boundary mass. It is plausible because SN1 proves the two sides have different frozen-head
  tolerances and the guard can enforce separate budgets instead of averaging them away.
- Persistence loss is likely to pay only on a small subset of oriented cells where components vanish,
  not on the whole residual. It is plausible because the Torch primitive directly pressures thin
  target support, but PR130 erasure versus displacement is still unmeasured.
- Margin satisficing can free gradient budget in the terminal QAT tail. It is plausible because
  PR130 already optimizes margins/expected flips and the frozen-head distance is the correct local
  flip coordinate; its cutoff must nevertheless be re-measured through PR130's exact R path.
- The island ladder may become useful after an oriented guard creates genuine births. It is plausible
  as a continuation method, but a static missing island does not establish the saddle-node dynamics
  it needs.
- A new joint two-frame semantic renderer could reopen screw consistency. It is plausible because the
  transport law is scene-side, but it is a representation change and cannot be claimed as a port to
  unchanged CPR1.

## DEAD-ENDS

- Copying any v7–v10/TR1 force constant, class ratio, margin floor, sigma matrix, temporal weight, or
  efficacy number into PR130: the target vehicle and operating point changed.
- Treating a per-class Road or Lane weight as the answer: SN1 and the requested consumer contract make
  the target→prediction relation directed.
- Using `ddm_sg2`'s AV-vs-DALI source-target difference or `ddm_pc2`'s other-vehicle flips as PR130
  residual localization: both are wrong objects.
- Exact anisotropic surface tension on unchanged PR130: there is no explicit interface/SDF/MCF energy
  for `sigma_cc'` to multiply; pair weighting is a different mechanism.
- Exact pair-internal temporal screw consistency on unchanged PR130: frame 0 is the independent pose
  carrier and frame 1 is the semantic master, so the required two semantic fields do not exist.
- Another unchanged-section coder race, a gauge rotation in the closed 432-row bank, or a post-hoc
  pose-carrier transform as a substitute for training the semantic residual: #996, `113b52fdb1`, and
  `ddm_pk2` respectively close those declared surfaces.
- Inverting PR130's rounded displayed `d_seg` into a mismatch count or fabricating an argmax tensor:
  the product is non-integer and the original payload was discarded; only SD2's retained output cures
  that evidence gap.
