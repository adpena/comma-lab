# ADVISORY — receiver discrete calculus, receiver-quantized topology, and hybrid-adjoint build contract — 2026-07-10

**Status:** `research_only=true` · advisory architecture and falsifiers only · no training, dispatch,
pointer move, process signal, run mutation, or actuation authority.

**Mission:** break the frontier through the original task-space SDF / level-set witness paradigm.
HiNeRV/PR128 remains a complete-artifact control and mechanism donor, not the mission. The purpose of
this unit is to turn the evaluator-quotient, topological-derivative, Hodge, and costate ideas in
`ADVISORY_sdf_evaluator_quotient_geometry_information_costate_curriculum_20260710.md` into typed,
testable build and probe contracts while auditing the telemetry/curriculum work that landed afterward.

**Pointer honesty:** the canonical pointer observed in this unit remains
`0.19108282419209976 [contest-CPU]`, archive `177,169 B`, SHA-256
`ad02b0124cbb3405c23d3480ac16f12b4e48cbf6f75878dd77a5e621bebd079c`. Nothing in this advisory
moves it. The current archive's CUDA axis remains unmeasured here.

## 0. Outcome first

The next correct build is not another loss knob. It is a three-layer measurement instrument:

1. **RDEC** — Receiver Discrete Exterior Calculus over legal archive mutations, with exact
   componentwise score cochains, inverse-loop receipts, order-commutator receipts, and interaction
   hyperedges.
2. **RQTD** — Receiver-Quantized Topological Derivative, which compiles and prices the smallest legal
   class birth/bridge/hole edit that survives archive → inflate → receiver → fresh scorer.
3. **HAC** — a lineage-aware, read-only Hybrid Adjoint Curriculum observer identified from matched
   checkpoint branches, with reset/saltation receipts at tau, birth, Muon, pose, rollback, and
   quantization events.

The post-advisory landings improve readback, but they do not yet supply any of those three layers.
The audit also found five apparatus defects that must precede curriculum optimization:

- the completed v7.5.2 dry-start/resume is **false-red at the parser**, despite positive primary
  resume evidence;
- the curriculum-pool store/module/digest integration does not exist; the recovered memo now carries
  an honest design-only banner, repairing the earlier prose/source contradiction but not the wiring;
- real `CostateEstimate` objects cannot enter the cross-run posterior because the writer reads
  `.tier` while the object exposes `.status`;
- the existing `action_commutator` measures non-additive interaction, not order noncommutativity or
  loop circulation;
- the canonical click-polish law drops the nonlinear Pose term at fixed bytes, although fixed bytes
  remove only the rate term. The implementation still exact-gates full score; the equation statement
  is the overclaim.

These are source-level findings. They are not authorization to patch shared source in this unit.

## 1. Authority snapshot and corrected launch disposition

### 1.1 Immutable evidence read

| artifact | SHA-256 / authority | finding |
|---|---|---|
| prior evaluator-quotient advisory | `47647e44656fbb2eb737a35ba4706ea76cd3132a12602092f1d126f78a23ea64` | design authority for EQM/RDEC/RQTD/HAC |
| telemetry audit memo | `b5c7d0f25520d6e7bcdc2f96ab149bbef19b07e2de9b3487b3391b5e40e71d76` | #404 claims and queued Q1–Q7 |
| default-off memo | `668595ea56ef75ba980a532a79b4f7a190dbe3d8a06725c4e9fc261fab4ff406` | #405 185-row snapshot |
| default-off JSONL | `8e863aed500ea6299a4cf6935175c89bf30b4b6720013c4b344eea51996ddc8b` | one meta + 185 data rows at audit time |
| curriculum-pool memo | `474a3c5d8df169bd2329a43b222099f3bb5aa906d15ce70962a5df12398cd003` | committed inventory (`22418c342`) with honest design-only banner; code/store surfaces absent |
| dry-start report | `0b9aedcdec01fc36d5d39a229cbf50037bbca45c11ef063497fef56f2ed3ee57` | official `green=false`; parser verdict invalid |
| dry-start pass-1 log | `54a84b3310dd0a915bf7361366784acd6247f38ef9a21d03a71b46cbc20f6834` | epoch-1 checkpoint written |
| dry-start pass-2 log | `28d6d11cac7079c93e03f1ce55b14680650b4157f48b1e83b40d247d2a8cda03` | resumed epoch 1 → start epoch 2; stepped and checkpointed epoch 2 |
| pass-1 resume NPZ | `fb216ca80634928c701ab6fc62cddb63b8d5972b4fc4728b894b128fdae4df7c` | preserved checkpoint |
| pass-2 resume NPZ | `1240b8cf2a0f2569c377b936c78b7a6fa3ecce41ffa04713dda5f5da4d10d1fe` | preserved resumed checkpoint |
| owed16v2 safe-run log | `8dba2eb10ba0306d948dfbfc805c33e7fff88d80eea94d92a457fea9b1a1ff2c` | clean single run; `status=ok`, exit 0, epoch 700 final verdict and checkpoints |
| owed16v2 final resume NPZ | `f1ac16f4a01a39283c6fdd72ffa57d9447457745df18a6966e984ea1718470c2` | preserved stage-Tau epoch-700 resume state |
| owed16v2 BEST EMA | `dea40f0b6b5be136a9ea86ad0d9250179aa42081d7f4342d0c969492de9907cf` | preserved best checkpoint at epoch 700, advisory axis |
| owed16v2 final stage checkpoint | `5bf56e93bae08b9ba14ae3ecbdc1fb4cdd74815e6d34b82829a26da59561bb29` | preserved stage-Tau epoch-700 EMA checkpoint |
| owed16v2 verdict | `355e9b93e24bc5ab80537f7a58b2cb0400535bf7df226d3263200489c4f52336` | formulation-scoped measured NO-GO; pointer unmoved |

### 1.2 What happened to the two v7.5.2 run surfaces

The earlier pilot `levelset_v752_pilot_20260710T154100Z` used `--grad-clip 1.0` plus
`--per-group-grad-clip`, had no stability-preset/pose-coefficient change, and crashed before its first
training epoch with `SigmaMinPlateauDetector` missing `should_ship_banked_r1`. It is not a live run and
not a launch-ready receipt.

The later governed `__v752_drystart_final__` used `--grad-clip 0.5`, per-group clipping, and
`--pose-grad-coeff-max 25.0`. Its bounded safe-run passes exited naturally under their designed timeout.
Pass 1 wrote epoch-1 EMA/resume checkpoints. Pass 2 emitted:

```text
stage=resume, resumed_epoch=1, start_epoch=2, restored_opt=true
stage=loss_terms, ep=2, weights_stepped=true, accepted_frac=1.0
stage=checkpoint, epoch=2, has_opt=true
```

The official report is red only because `parse_dry_start_run_metrics()` searches for a top-level
`resume_start_epoch`, while the trainer's real schema is `stage=resume, start_epoch`. Its unit test
uses the invented field and therefore protects the mismatch.

The separately watched owed16v2 rebalanced-ON arm completed naturally while this advisory was being
validated. Its immutable terminal log records `status=ok`, exit 0, `13,202.09 s` elapsed under the
`14,400 s` limit, peak RSS `74,248 MiB` under the `96,000 MiB` limit, an epoch-700 exact advisory
verdict, and final EMA/resume checkpoints. It was one clean seed-0 run with no resume. Epoch 700
measured `d_seg=0.004213` versus the matched self-orient-OFF `0.004181` — a RELATIVE significance of
Δ = +3.2e-5 worse = +3.2e-3 S-term, which is **7.8% of the 0.0411 remaining gap** (0.19108→0.15 at the
current operating point) i.e. NOT negligible-by-magnitude but MEASURED-worse-in-a-matched-A/B — so the
freq-along-heavy warm-start formulation receives a formulation-scoped NO-GO (verdict_scope: formulation
— freq-along-heavy warm-start init specifically). This does not kill the wider SDF family or the distinct
from-scratch formulation. # MAGNITUDE_DISMISSAL_OK: not a magnitude dismissal — both numbers stated + relative-to-gap (7.8%); it is a MATCHED-A/B measured-worse verdict at formulation scope, the family stays open. A read-only process refresh found
no matching trainer or `tail -F` watcher. This unit sent no signal and did not mutate the run.

The six SUM-over-RAM governor refusals belong to the earlier **owed16** ON-resume attempt for its
missing epoch-700 cell, not to owed16v2. They remain valid historical safety evidence, but are neither
the current owed16v2 disposition nor a blocker to harvesting its completed artifacts.

Literal disposition:

| surface | disposition | reason |
|---|---|---|
| pilot run | **FAILED / HISTORICAL** | real AttributeError before training; superseded by repaired dry-start path |
| dry-start boot mechanics | **POSITIVE PRIMARY EVIDENCE** | epoch-1 step and complete checkpoint exist |
| resume mechanics | **POSITIVE PRIMARY EVIDENCE** | parent epoch restored, optimizer restored, epoch-2 step and checkpoint exist |
| official dry-start report | **FALSIFIED AT INSTRUMENTATION** | parser/schema mismatch yields false red |
| dry-start gate | **RED UNTIL SUPERSEDED** | preserve the report; repair parser and issue a new correction receipt from immutable evidence |
| real v7.5.2 launch | **HOLD** | this unit has no launch authority and the canonical gate has no corrected green receipt |
| running process | **NO MATCHING TRAINER OR WATCHER OBSERVED** | read-only process refresh after natural completion; this unit sent no signal |
| owed16v2 run | **COMPLETED NATURALLY / HARVESTED** | clean epoch-700 terminal receipt and preserved final checkpoints exist |
| owed16v2 warm-start formulation | **MEASURED NO-GO / FORMULATION-SCOPED** | rebalanced-ON was marginally worse than OFF at every trained comparison cell; wider family remains open |

### 1.3 Minimal dry-start instrument repair contract

The superseding parser must require all of the following, not merely a large epoch number:

1. a `stage=resume` row with `resumed_epoch`, `start_epoch=resumed_epoch+1`, `restored_opt=true`, and
   the exact parent checkpoint path/hash;
2. a post-resume `loss_terms` row at `ep >= start_epoch` with `weights_stepped=true` and a positive
   accepted fraction;
3. a post-resume checkpoint row at `epoch >= start_epoch`, plus NPZ metadata whose resume epoch agrees;
4. a real-schema regression fixture copied from the trainer's emitted field names;
5. a correction receipt that cites the old report/log/NPZ hashes and never overwrites the false-red
   historical report;
6. peak RSS reported as the maximum over both passes, not only pass 1.

## 2. Delta audit: what closed, what did not

| surface | disposition | exact audit result |
|---|---|---|
| #404 telemetry query module + CLI + digest section | **LANDED / USEFUL SENSE LAYER** | committed in `60dcada34`; read-only analyzers and digest wiring exist |
| event table | **PARTIAL** | normalizes emitted events; held decisions and a uniform engage schema remain Q5/Q7 |
| amber/global clipping | **PARTIAL** | pre-clip global norm is visible when resolved config is emitted; it is not actual per-group activation under per-group clipping; final dry-start has only two rows and honestly returns `UNKNOWN` |
| pilot amber readback | **OPEN** | no unconditional resolved-config row or launch-manifest fallback; pilot reads `UNKNOWN` |
| chroma `BINDING` | **LOSS-ACTIVE ONLY** | nonzero loss share proves pressure in the optimized loss, not causal receiver-exact `d_seg`/Pose/byte value |
| pose detector liveness | **PARTIAL** | useful cadence alarm, but log-lineage mixing can create false stall/health calls |
| telemetry version awareness | **OPEN** | an old log that predates a sensor can be labeled stalled; expected-emitter/schema identity is not carried, so legacy `DETECTOR_STALLED` is inadmissible without source-version proof |
| EMA-lag | **PARTIAL** | trend signature, not a live-versus-EMA exact evaluator delta; Q3 remains |
| D27b `d27b_ready` | **HEURISTIC ONLY** | any Muon row + eight-row endpoint `d_seg` plateau; ignores Pose, bytes, full score, frozen steps, mode/reset identity, uncertainty, and conditioning; digest surfaces it, no actuator consumes it |
| TAIL endpoints | **PARTIAL** | post-hoc join exists; source-path/segment/checkpoint lineage is absent |
| Q1–Q7 | **OPEN** | no per-group activation, inert-term alarm, live verdict gap, explicit endpoint, held event, ladder-complete, or uniform engage rows landed |
| v8 carrier attribution | **DESIGN ONLY** | no carrier rows, carrier-off anchors, bytes, receiver, or exact attribution implementation |
| #405 table snapshot | **INTERNALLY CONSISTENT AT AUDIT** | one meta + 185 unique rows; current 70 duty and 107 unmapped surfaces matched |
| #405 consume gate | **PARTIALLY CLOSED** | syntax/anti-truncation warning, not semantic consumption; minimal one-row table can pass, substring token passes, same-day/undated/differently named finalizers bypass |
| curriculum candidate pool | **MEMO HONEST / IMPLEMENTATION ABSENT** | current recovered memo says design-only; JSONL/module/digest remain absent, so no controller-held pool exists |
| rate visibility correction | **VISIBLE BUT NOT DSL-REGISTERED** | D18/mod32 are included by the significance-key union and visible in ranked findings; prior statement that the controller could not see them was too strong |
| LADDER costate gates | **INERT / OPEN** | both thresholds remain `0.0`; proxy is not an action derivative |
| cross-run costate posterior | **BROKEN** | writer reads `c.tier`, real dataclass exposes `c.status`; fake-tier test masks it; canonical JSONL is absent |
| within-stage costates | **PARTIAL** | regular verdicts carry `seg_form`, but baseline-v0 uses only `phase`; the pilot/dry-start never reached the first regular stage verdict, so no stage slope is identifiable there |
| per-class costate feed | **STRUCTURALLY UNIDENTIFIABLE** | `ShadowController` passes verdict rows, while `per_class_within_flip_costates()` expects `handoff_readiness`-shape `within_flip`/`part_frac` fields that live outside that list |
| recommendation confidence gate | **OPEN** | current advice can admit on a favorable central sign even when its 95% interval crosses regression; HAC must gate the upper confidence bound |
| true dynamic adjoint/saltation | **ABSENT** | no state/control Jacobians, reset derivatives, adjoint trace, or matched branch backtest |
| event treatment reset | **PROVENANCE-SPLIT** | early sensor-fired lane/chroma/temporal engagement clears the spike median, but fixed-cap `_stage_boundary_now` drives LR rewarm/moment reset; tau advances likewise sit outside that boundary flag |
| event ordering/collisions | **UNMODELED** | Muon executes before pose and tau advancement; simultaneous/colliding events have no ordered reset receipt for adjoint composition |
| tau ladder reachability | **CONFIG-STRUCTURAL BLOCKER** | 12 octaves with 250-epoch minimum dwell and Muon cap/freeze at 726 can complete at most about two advances before freeze; the advertised ladder is not temporally reachable as a whole |
| pose/Muon conjunction | **OPEN** | the sigma pose gate is not explicitly conjuncted with `muon_fired`; a true early plateau can engage Pose before the intended finisher boundary |
| v7.5.3 readiness flags | **SCHEMA-MISMATCHED** | readiness reader asks for `launch_flag_tokens`; #405 rows store `flags`, so real Horizon/StepNative flags fall back to fictitious name-derived prefixes |
| v7.5.3 standard builder | **INCOMPLETE ACTUATION SURFACE** | it documents 11 registered-off rungs but exposes no general rung-selection parameters; the standard launcher cannot compose the table's fire-now set |

### 2.1 Telemetry needs lineage before it can drive a controller

`load_run_rows()` concatenates root and one-level child logs, then analyzers sort primarily by epoch.
It carries no source path, run/segment ID, launch hash, checkpoint hash, parent lineage, global step, or
deduplication rule. In the real dry-start tree, pass 1 and pass 2 therefore appear as one anonymous
trajectory with duplicate startup/verdict rows. This can contaminate EMA trends, detector cadence,
TAIL endpoints, and D27b readiness.

Every emitted/read row used for decisions must be wrapped by:

| field | invariant |
|---|---|
| `run_id` | one governed launch identity |
| `segment_id` | one process lifetime; changes on resume |
| `lineage_id` | stable across a resume-compatible chain |
| `parent_checkpoint_sha256` | required on resumed segments |
| `launch_sha256`, `config_sha256` | exact compiled program/config identity |
| `global_step`, `epoch`, `row_seq` | monotone within the selected lineage |
| `source_log_sha256`, `source_path` | custody and conflict diagnosis |
| `telemetry_schema`, `expected_emitters_hash` | distinguish absent-by-version from a live detector that stopped |
| `weights_stepped`, `accepted_frac` | liveness filter |

Default policy: analyze one explicitly selected lineage. If two segments claim the same child step
with conflicting values or ancestry, refuse rather than merge.

### 2.2 Costate terminology and memory must be repaired before HAC

Keep four types distinct:

- `TerminalScoreCovector = grad_y S`;
- `DynamicAdjoint = p_k` propagated through checkpoint dynamics;
- `ConstraintDual = mu, nu` for topology/area/rate constraints;
- `ContinuationCoordinate = tau, epsilon, beta, r`.

The current source additionally retains the prior advisory's open defects: Seg-only transition jump,
minimum-`d_seg` rollback selection before full-score comparison, worst-unweighted per-class slope despite
area-weighting prose, independent-channel uncertainty, and coarse action cost. Repairing the posterior's
`.tier`/`.status` mismatch is necessary but not sufficient; it would only persist these existing
estimates, not turn them into dynamic adjoints.

The per-class path has a second, independent wiring error: its function asks for per-class
`handoff_readiness` fields but receives only verdict rows. Moving or joining those rows must preserve
epoch, stage, and lineage; copying the aggregate into each class is forbidden. Baseline-v0 rows also
need an explicit chart label or exclusion so `phase` and `seg_form` do not become a silent stage split.

## 3. Score law and units: the componentwise closure rule

For archive bytes `B` and denominator `D=37,545,489`, the exact objective is

\[
S=100d_{seg}+\sqrt{10d_{pose}}+\frac{25B}{D}.
\]

Therefore every legal mutation must close

\[
\Delta S=
100\Delta d_{seg}+
\left(\sqrt{10d'_{pose}}-\sqrt{10d_{pose}}\right)+
\frac{25\Delta B}{D}.
\]

If rate is represented in bits, the coefficient is `25/(8D)` score units per bit. If represented in
bytes, it is `25/D` per byte. The type must say which.

This exposes a current canonical-equation defect. `clickpolish_exact_gated_discrete_latent_ratchet_v1`
states that `delta B=0` implies `delta S=100 delta d_seg`. Fixed bytes remove only the rate term; the
Pose term remains unless exact Pose invariance is separately proved. `tac.click_polish` does recompute
Pose and exact-gates full `S`, so the accepted pointer row is not invalidated by this prose/LaTeX bug.
Disposition: **repair the equation's domain and formula; preserve the empirical anchor**.

Likewise, a factory that replaces missing `d_pose_per_pair` with a constant aggregate vector may be
adequate for an aggregate report but is false authority for pair-local proposal selection, covariance,
or locality. RDEC/HAC consumers must fail closed when the required per-pair Pose surface is absent.

## 4. Receiver Discrete Exterior Calculus (RDEC)

### 4.1 The correct complex

Let `K0` be fully materialized legal archive states. Let `K1` be typed legal mutations whose head and
tail both parse, inflate, and evaluate. Let `K2` contain commuting mutation squares for which both
orders are legal and endpoint identity is explicitly tested. Define the exact score 0-cochain
`s(A)=S(A)` and its coboundary

\[
(\delta s)(A\xrightarrow{m}A')=s(A')-s(A).
\]

Then `delta^2 s = 0`: the oriented sum around a genuinely closed state loop telescopes to zero.
Nonzero circulation is not a new physical effect; it means the state identity omitted something,
decode/scoring was nondeterministic, caches/axes changed, custody drifted, or arithmetic tolerances were
misdeclared.

Three quantities must not share one name:

1. **interaction / mixed difference**

   \[
   I_{ab}=S(A_{ab})-S(A_a)-S(A_b)+S(A_0);
   \]

   this may be nonzero and measures synergy/conflict;
2. **order commutator**

   \[
   C^{order}_{ab}=S(A_{ab})-S(A_{ba});
   \]

   this tests order dependence and requires both orders;
3. **loop circulation** `\oint \delta s`, which must vanish for a closed, same-authority loop.

Existing `tac.action_commutator.v1` computes item 1 from `A`, `B`, and one `AB` composite. It is a
useful interaction ledger but cannot prove items 2 or 3. Extend it additively; do not reinterpret its
historical rows.

### 4.2 Typed records

| record | required payload |
|---|---|
| `ReceiverIdentity.v1` | receiver/manifest versions; inflate, dependency, evaluator and scorer hashes; axis, hardware/microarchitecture, batch/thread law; expected raw cardinality |
| `ReceiverState.v1` | `state_id=H(receiver_identity,archive_sha,payload_sha,raw_sha)`; archive bytes/member hashes; exact `d_seg`, `d_pose`, `B`, `S`; scorer-trace parity; per-pair/per-class availability; topology/chart hashes. Score is excluded from `state_id` so repeated-score nondeterminism is detectable |
| `LegalMutationSpec.v1` | mutation ID/family; domain and exact preconditions; deterministic callable hash; declared touched members/fields; inverse mutation if any; legal alphabet; expected locality; no-smuggling declaration |
| `RDECEdgeReceipt.v1` | tail/head state hashes; mutation hash; componentwise deltas; exact closure residual; changed archive members/raw frames/scorer cells; wall time; fresh-process/cache status |
| `RDECInverseLoop.v1` | `A0 -> A1 -> A2`; inverse pair; canonical archive/raw endpoint equality; component circulation; refusal reason |
| `RDECMutationSquare.v1` | four states `A0,Aa,Ab,Aab,Aba`; both order traces; endpoint archive/raw/evaluator equivalence; interaction; order commutator; loop circulation |
| `ReceiverEquivalenceClass.v1` | class ID; member archives; equality level: archive/raw/evaluator; invariants; witness separating any non-equivalent members |
| `RDECInteractionHyperedge.v1` | ordered action set; complete-state measurements; Möbius/ANOVA interaction; uncertainty; authority and scope |
| `ReceiverTopology.v1` | declared foreground/background connectivity (default proposal 4/8); per-class components/holes/`beta_0,beta_1`; RAG; triple/higher junctions; explicitly named cubical barcodes if computed |
| `ReceiverTransversality.v1` | fresh-scorer margin surface; stencil/interpolation law; interface/junction hashes; sampled `tau_1,tau_2`; receiver perturbation envelope; sampled-clearance versus certified-lower-bound label |

Every score receipt carries all three objective components even when a component is predicted invariant.
An invariant is a measured field, never an omitted field.

### 4.3 Build algorithm

1. Materialize a baseline archive in a fresh process; validate member table, exact size, inflate
   cardinality, raw SHA, evaluator hash, and component score.
2. Apply one `LegalMutationSpec` to archive/payload source, rebuild the complete archive, parse back,
   inflate fresh, and score fresh. Never mutate inflated output outside a legal archive program.
3. Emit an edge receipt whose edge delta is independently measured, then compare it with the endpoint
   difference. Merely defining every edge as endpoint subtraction makes `delta^2=0` tautological and
   cannot catch a broken instrument.
4. If an inverse exists, apply it and test archive-canonical, raw, and evaluator endpoint identity
   separately.
5. For action pairs, build `A`, `Aa`, `Ab`, `Aab`, and `Aba`. Report interaction and order commutator
   separately. A missing order is `UNMEASURED`, never zero.
6. Repeat the same archive at least three fresh-process decodes. Raw SHA and score vector must be
   deterministic before any loop result is admissible.
7. Recurse over archive member, payload block, carrier, class edge, pair, spatial footprint, symbol,
   and bitplane, while preserving a complete receiver at every vertex.

For incidence matrices `B1` (vertex-edge) and `B2` (edge-face), validate `B1 B2 = 0`, then compare the
independently measured 1-cochain with `B1^T s` and require `B2^T B1^T s = 0`. A square whose `AB` and
`BA` destination hashes differ is not a 2-cell; report `NONCOMMUTING_NO_2_CELL` rather than forcing a
circulation number.

### 4.4 Tolerances

- archive/member/raw hashes: exact equality;
- Seg flip counts: exact integers before normalization;
- bytes: exact integers;
- Pose/evaluator floats: same evaluator's declared deterministic tolerance, with endpoint recomputation
  and axis held fixed;
- score closure: derived from the above, not an independent magic epsilon;
- CPU and CUDA: separate complexes until the same exact archive is evaluated on both.

For pair-local edits, remember that additive per-pair `d_pose` changes can still create a nonzero
full-score interaction because the video-level Pose term applies one square root after aggregation.
That is legitimate nonlinearity, not order curl.

## 5. Receiver-Quantized Topological Derivative (RQTD)

Classical smooth shape derivatives move an existing boundary and cannot create a new connected
component or handle. RQTD is the finite, legal receiver counterpart: the value of the smallest
topology-changing archive edit that survives all discontinuities.

For a typed insertion `q=(class, pair, site, grammar, shape, quantized_amplitude)`, define

\[
a^*(q)=\min\{a\in\mathcal A_q:\ T(R(I(A,q,a)))\text{ satisfies the registered topology event}\},
\]

where `I` is archive/inflate, `R` is the exact resize/uint8 receiver, and `T` is the topology/scorer
receipt. The exact value per bit is

\[
V_{bit}(q)=-\frac{\Delta S(q)}{8\Delta B(q)}
\]

when `delta B>0`; zero/negative-byte edits report the full component vector rather than divide by zero.

### 5.1 First insertion compiler

The smallest useful compiler supports one class and one site, but its contract is complete:

1. enumerate a deterministic finite legal alphabet of disk/ellipse seed, thin curve/bridge, or
   hole-fill grammars, ordered by charged bits, footprint, and canonical ID;
2. compile each candidate into the counted payload/archive;
3. parse back in a fresh receiver and measure raw cardinality/hash;
4. compute class cubical `beta_0/beta_1`, persistence clearance, RAG, junctions, and intended scorer
   corrected cells before and after;
5. score full Seg, nonlinear Pose, and exact bytes;
6. exhaust the declared family until every cheaper member has a receipt; resize phase, uint8,
   rasterization, entropy coding, and the scorer can make survival non-monotone, so binary search alone
   is not a minimality proof;
7. emit rejected candidates too, including `generator-only`, `decode-only`, `receiver-erased`,
   `topology-wrong`, `Pose-harm`, `rate-dominated`, or `exact-score-negative`;
8. rerun the winning candidate three fresh times and embed it in an RDEC inverse/interaction test.

### 5.2 Topology trust region

An insertion is admissible only if its receipt declares:

- allowed births and target class/site;
- forbidden deaths, mergers, splits, and handles in every affected class;
- minimum persistence/clearance above the measured receiver perturbation envelope;
- evaluator-cell benefit, not only generator-mask agreement;
- Pose and remote-class spill bounds;
- exact archive and metadata cost.

Differentiable topology losses may propose candidates, but only this receiver-exact discrete receipt is
authority.

Minimality is always family-relative. For `delta B=0`, report exact `delta S` and do not manufacture an
infinite value/byte. For added bytes, report both exact total `delta S` and nonrate score saved per
added byte.

### 5.3 Receiver-discrete transversality

For class interface `Gamma_ij` and a declared finite-difference stencil `D`, measure

\[
\tau_1^{disc}=\min_{\Gamma_{ij}}\|D(\phi_i-\phi_j)\|_2,
\]

and at a three-class junction

\[
\tau_2^{disc}=\min_{J_{ijk}}\sigma_{min}
\begin{bmatrix}
D(\phi_i-\phi_j)\\
D(\phi_i-\phi_k)
\end{bmatrix}.
\]

A positive sampled value is only `SAMPLED_CLEARANCE`. It becomes a continuum certificate only with a
declared interpolation law plus a Lipschitz/interval lower bound over the cells between samples. Record
the same quantities on generator, decoded/raw, and fresh-scorer margin charts; disagreement is a
receiver event, not numerical noise to smooth away.

### 5.4 Complexity and storage

- RDEC costs `O(V_new * C_eval)` with sparse complex storage `O(V+E+F)`;
- class Hodge is negligible at `K=5`, `E<=10` (`O(K^3)` dense solve);
- receiver topology/transversality over all pairs is approximately `O(P*K*H*W)`, plus sorting for
  barcode-style persistence;
- RQTD over `A` atoms costs `O(A*(C_compile+C_inflate+C_eval+C_topology))`.

The current click-polish n600 exact evaluation receipt records `176.3 s` on its measured Linux CPU
substrate. A five-state square is therefore about `14.7 min` before reusable-cache savings on that
specific substrate; remeasure elsewhere. Do not retain a 3.66-GB raw video per vertex. Preserve
archives, manifests, raw hashes, compact per-pair/component/topology receipts, and certified SSD
scratch only where deterministic reproduction requires it.

## 6. v8 integrability, Hodge structure, and global labels

For a connected five-class region-adjacency graph with nine active edges, the cycle-space dimension is
`E-V+1 = 5`. Five independently encoded cycle degrees are therefore extra global-consistency debt.

Let `B1` be the vertex-edge incidence matrix and `e(x)` an oriented edge-margin 1-cochain. A
potential-derived field has

\[
e(x)=B_1^T\phi(x),\qquad C^Te(x)=0
\]

for every cycle basis `C`. The additive common mode `phi -> phi + c 1` is a true gauge. Semantic class
permutations are not gauge because evaluator labels are fixed; radial scaling is not assumed gauge
because quantization/clamp/receiver operations can change its result.

Two first receivers are admissible:

1. **centered-potential codec:** store/derive `K-1` centered global potentials and compute every edge
   difference from them;
2. **Hodge-safe tree codec:** store a root plus `V-1` tree differences, reconstruct potentials by path
   integration, derive non-tree edges, and charge tree metadata/path-stretch error.

An independent-edge codec remains HOLD unless it provides an explicit global labeler and wins a
matched complete-archive comparison. Its receipt must decompose

\[
e=B_1^T\phi+B_2\psi+h
\]

where applicable, charge/dispose of curl/harmonic residuals, and show the resulting label partition is
stable after quantization. Silent least-squares projection of inconsistent edges is not free; its lost
residual and score effect are part of the receipt.

The class-graph Hodge complex is not the archive-mutation complex. Keep their incidence matrices,
cochains, and closure claims separate. For measured edge field `g`, solve the centered weighted
projection

\[
\phi^*=\arg\min_{\mathbf 1^T\phi=0}\|W^{1/2}(B_1^T\phi-g)\|^2,
\qquad r=g-B_1^T\phi^*.
\]

Report a fundamental-cycle holonomy `C^T g`, weighted residual norms, and the post-quantization result.
If a per-frame RAG is disconnected, there is one free offset per connected component; the receiver
must store/derive cross-component anchors or refuse global labeling. This is another reason global K
potentials are the first v8 arm.

Minimum v8 `GraphReceiverSpec.v1` fields:

- fixed semantic vertex order, oriented edge order, incidence and cycle-basis hashes;
- active RAG per pair/frame and shared-interface ownership;
- gauge rule and quantization order;
- pointwise and aggregate holonomy before/after receiver;
- global-label algorithm and tie rule;
- junction closure/Young-force receipt;
- exact headers, metadata, and ZIP bytes;
- topology/RQTD and RDEC receipts for each carrier increment.

## 7. Hybrid Adjoint Curriculum (HAC)

### 7.1 Full state versus an honest reduced observer

The full checkpoint state is the serialized model, optimizer, EMA, RNG, event controllers, stage,
topology/area duals, and rate state:

\[
x_{k+1}=F_{q_k}(x_k,u_k,\omega_k).
\]

An observer will usually fit a reduced chart `z=Psi(x)`. It must call its result an
**identified reduced adjoint**, not a true full optimizer costate, until closure error is bounded.

Use matched branches from one preserved checkpoint to identify

\[
\delta z_{k+1}=A_k\delta z_k+B_k\delta u_k+\epsilon_k,
\qquad
p_N=\nabla_z\Phi,
\qquad
p_k=\nabla_z\ell_k+A_k^Tp_{k+1}.
\]

Categorical controls use matched one-sided branches. Continuous continuation coordinates use legal
central differences when both sides preserve the same chart/topology. No derivative is inferred by
comparing unrelated runs.

### 7.2 Reduced state chart

At minimum `z_k` contains typed, unit-bearing fields:

- exact/advisory `d_seg`, `d_pose`, bytes, full `S`, and per-class/per-pair availability;
- receiver-realized topology, persistence clearance, RAG, junction and holonomy;
- stage/mode, tau/beta/epsilon/radius, optimizer and LR state;
- EMA/live gap, accepted-step fraction, frozen/spike/confound state;
- conditioning spectrum and quotient-normal/tangent energy;
- archive-section bytes and rate-home state;
- wall time, reversibility, checkpoint size, and lineage identity.

Missing fields do not become zeros. They make the corresponding derivative or decision
`UNIDENTIFIABLE`.

### 7.3 Events and saltation

For guard `h_e(x,t)=0`, possibly time-dependent reset `R_e`, pre/post vector fields `f-/f+`, and
`n=grad_x h_e`, the continuous-event linearization is

\[
\Xi_e=D_x R_e+
\frac{(f^+-D_x R_e f^- - \partial_t R_e)n^T}{n^Tf^-+\partial_t h_e},
\qquad
p^-=\Xi_e^Tp^++\nabla c_e.
\]

This applies only when the denominator is safely separated from zero and a continuous interpolation
is justified. A grazing event or purely discrete checkpoint guard must use a discrete reset/event-time
sensitivity or be marked unidentifiable.

Events needing receipts: tau octave and simultaneous LR change, birth completion/ramp, Muon optimizer
and moment reset, pose engagement, rollback, texture/paint engagement, quantization/rate activation,
head solve, and terminal stop.

Every event receipt also carries `collision_id`, `execution_index`, and event-vs-cap provenance. For a
collision, compose saltation/reset matrices in actual forward execution order; the adjoint traverses
their transposes in reverse order. If the order or one reset policy is absent, the jump is
`UNIDENTIFIABLE`.

### 7.4 Typed HAC records

| record | minimum payload |
|---|---|
| `CheckpointLineage.v1` | run/segment/parent IDs; checkpoint/config/code/RNG hashes; stage and complete-state availability |
| `StageDynamicsLinearization.v1` | chart/hash/units; source checkpoint; mode/control; `A,B`; branch deltas; fit method; residual/cross-validation; condition/rank; covariance |
| `HybridEventReceipt.v1` | guard/reset hashes; pre/post state; event-vs-cap provenance; sensor epoch/lag; collision ID/execution index; event time; `n,f-,f+,D_xR,partial_tR`; denominator; saltation/discrete-jump matrix; uncertainty; matched evidence |
| `AdjointCostateTrace.v1` | terminal objective/covector; backward steps; event jumps; covariance; reduced/full label; closure residual |
| `ActionCost.v1` | exact wall time, complete ZIP delta, uncertainty, reversibility, opportunity cost, constraints; no guessed scalarization |
| `SwitchingSurfaceSpec.v1` | current/proposed modes; component Hamiltonian advantage; switch/risk costs; confidence bound; topology/receiver guards; disposition |
| `HACBacktest.v1` | predicted versus observed terminal component deltas for held-out matched branches; calibration, sign accuracy, resume identity |

### 7.5 Robust switching rule

Contest value and operational cost are distinct. Start with the vector

\[
(\Delta S_{contest},\Delta t,\Delta\text{risk},\Delta\text{reversibility},\Delta\text{information})
\]

and scalarize only with registered, unit-bearing duals. A switch is advice-eligible only when the
upper confidence bound on the registered augmented advantage is below zero and all receiver/topology/
resume guards pass. More explicitly, require

\[
UCB_{95}[\Delta S_{exact}+C_{switch}+C_{risk}]<-\delta_S,
\]

where `delta_S` is a measured replay/evaluation floor, not a guessed margin. The observer remains
read-only until held-out backtests pass.

### 7.6 D27b readiness v2

Replace the current `Muon && abs(endpoint d_seg change)<0.5%` heuristic with a typed terminal-solve
probe gate requiring:

1. one conflict-free lineage and one stable mode/window with no unresolved reset inside it;
2. enough accepted, weight-stepping rows; no frozen or telemetry-unknown interval;
3. robust full-score and component slopes with covariance/confidence bands;
4. no materially worsening class, Pose term, rate, topology, or conditioning signal;
5. Muon/tail/Polyak state consistent with the selected checkpoint;
6. exact checkpoint availability and receiver/cost readiness;
7. output `PROBE_READY`, never automatic actuation.

## 8. Curriculum-source-of-truth contract

### 8.1 Curriculum pool

The current `curriculum_candidate_pool_p0_20260710.md` is a valuable inventory but not a landed pool.
Before any costate-driven curriculum selection, the implementation unit must atomically supply:

- the append-only JSONL store;
- its importable module and schema validation;
- costate-digest consumer;
- real source/store/digest integration tests;
- a producer for every memo row or an explicit proposed-only import receipt;
- a source hash and latest-row-wins rule;
- no half-wired DSL stubs.

Until then, every candidate in that memo remains `MEMO-INVENTORIED`, not controller-held.

### 8.2 Default-off consumption

The #405 table is a sound dated snapshot. Its current gate is only a warn-only syntactic reminder.
The final contract should be a machine-generated `ConfigDecisionReceipt.v1` keyed by:

- decision-table schema/version/SHA;
- recomputed current surface census and hashes;
- exact set equality, unique names, required owner/gate/evidence fields, and dispositions;
- every compiled lever/tool/curriculum candidate's selected state and reason;
- compiled launch/config SHA;
- explicit carry-forward of blocked rows.

Gate the compiled receipt, not filenames or date substrings. Same-day, undated, renamed, or edited
finalizers must not bypass it. Strict promotion can occur only after warn-only calibration is clean.
The gate must validate a staged arm plan, not require every `fire-now` row simultaneously in one launch:
exactly one isolated rung (or one pre-registered interaction bundle) is armed, while all other rows
carry explicit next-rung/defer receipts. Presence-only all-at-once checking destroys attribution.

## 9. v7.5.3 event graph: build/probe contract

This is sequencing, not launch authority.

### V0 — repair authority and lineage

- correct the dry-start parser and issue a superseding receipt;
- land the real curriculum pool and costate-posterior type fix;
- add unconditional resolved-config and telemetry-lineage rows;
- consume #405 through a hashed config-decision receipt;
- align the readiness reader with the table's actual `flags` field (or migrate both atomically), and
  test real HorizonWeightedMargin/StepNativeActivation tokens rather than name-derived fictions;
- expose typed, validated rung selection in the v7.5.3 builder; registered-off prose is not an
  executable staged controller;
- preserve the current pointer and every checkpoint.

Exit: the apparatus can distinguish pilot, dry-start pass 1, and resumed pass 2; a real
`CostateEstimate` round-trips into a test posterior; no memo claims absent source.

### V1 — receiver census and RDEC baseline

- compile one complete receiver early;
- perturb/remove every counted section;
- require intended raw/evaluator effect or classify it inert/dead;
- close determinism, inverse loops, interactions, and order tests;
- correct the click-polish equation's Pose clause.

Exit: componentwise exact score closure and receiver custody are green.

### V2 — coarse topology under one-sided protection

- form the partition with structured initialization and existing birth/area machinery;
- compute receiver-realized class presence, persistence, RAG, junction, and transversality;
- use one-sided protection only for missing/subcritical features;
- run the first RQTD insertion compiler on one missing class/site.

Exit: every required class has quantified receiver clearance and no forbidden topology event.

### V3 — quotient-normal geometry

- optimize active pair margins/shared interfaces rather than five unrelated SDFs;
- measure common-mode gauge, tangent/normal energy, root-eikonal triangle closure, and single-count
  interface geometry;
- change one geometry force per isolated branch;
- update topology/area duals only at preserved checkpoint boundaries.

### V4 — self-paced continuation with event receipts

- retain geometric tau octaves;
- isolate tau from LR where the discriminant is near, or account for the coupled reset explicitly;
- fit dwell from measured relaxation modes only after data exists;
- reconcile 12 octaves at 250-epoch minimum dwell with the epoch-726 Muon freeze: make the reachable
  subsequence explicit, derive a later event boundary, or refuse the unreachable schedule;
- require an ordered reset receipt for event-versus-cap engagement, and explicitly conjunct Pose with
  the intended Muon/geometry state or test the alternative ordering as its own formulation;
- use full-score/class/topology/conditioning advantage, not one scalar `d_seg` EMA;
- backtest event/reset predictions before using them in advice.

### V5 — geometry-to-texture chart reset

- engage texture only after separatrix/topology clearance;
- measure geometry-texture interaction from matched branches;
- route exact-D/chroma candidates only after receiver-exact Pose invariance;
- treat paint engagement as a chart reset with a new linearization/adjoint boundary;
- keep default-off arms isolated until exact evidence composes them.

### V6 — rate/home duals

- require complete archives at two or more matched task/bit points;
- allocate over geometry, topology tokens, texture, pose, headers, and exceptions using exact ZIP deltas;
- preserve interaction hyperedges; do not force a scalar greedy order when bundles are superadditive;
- distinguish entropy proxies from exact rate dual authority.

### V7 — conditioning, Pose, and terminal compile

- select optimizer switch from quotient-conditioned advantage with Muon as backstop;
- preserve EMA, Polyak, rollback, and every stage checkpoint;
- engage Pose only after geometry/topology stability and a healthy conditioning gate;
- require per-pair Pose rather than aggregate substitution for pair-local decisions;
- run head solve, exact discrete finisher, Pareto prune, byte-close, then CPU/CUDA separately.

## 10. v8 event graph: build/probe contract

### E0 — paint-free matched partition screen

Measure decoupled versus matched-compute control, seed-spread floor, scorer-grid partition, ties, and
topology. Make no through-R/rate claim.

### E1 — gauge-fixed global potentials

Use centered potentials; derive every edge. Record incidence/cycle basis, RAG, interface ownership,
gauge tests, and topology.

### E2 — edge-owned geometry and RQTD

Single-count each interface; impose root-eikonal/junction balance; coordinate both incident classes at
shared edges; insert missing islands through RQTD; keep paint absent.

### E3 — receiver/Hodge closure

Quantize and parse back; require global labels, zero or explicitly charged holonomy, topology clearance,
and complete metadata. Compare potential and tree codecs at matched ZIP bytes.

### E4 — merge, diff, correct, then texture

Merge carriers into one partition, freeze and score the residual, correct with task-adapted integer
atoms, then engage texture. The merge/paint boundary is a hybrid chart reset.

### E5 — graph/home rate control

Allocate exact bytes over tree/potentials, topology, texture, pose, hard-pair innovations, and headers.
Charge tree stretch, cycle metadata, exceptions, and interactions.

### E6 — Pose connection and terminal compile

Measure the horizontal Pose section conditioned on decoded geometry and temporal history; treat
nonzero holonomy as a candidate hard-pair innovation, not silently as edge freedom; exact-finish and
evaluate axes separately.

## 11. Information and experiment design

The next probe should maximize frontier-relevant information, not estimated novelty. For probe `a`,
rank a typed value-of-information objective such as

\[
\operatorname{VoI}(a)=
\mathbb E_{Y_a}\left[\min_u\mathbb E[J\mid Y_a]\right]
-\min_u\mathbb E[J]
\]

with sign convention stated, then divide by measured wall/custody cost only if a registered policy
allows scalarization. Candidate interactions use complete matched archives and Möbius/functional-ANOVA
terms; they are not inferred from independent marginal EVs.

Priority information questions:

1. which archive/payload homes actually control receiver cells and Pose;
2. smallest receiver-surviving topology change and its exact value/bit;
3. whether geometry, texture, Pose, and rate branches are separable or interaction-dominated;
4. whether a saltation-aware boundary model predicts held-out branch outcomes better than a matched
   no-jump baseline;
5. whether potential/tree v8 receivers dominate independent edges after all metadata and global-label
   costs are charged.

## 12. Minimal proof matrix

| proof | positive | negative/control | admission |
|---|---|---|---|
| dry-start parser | real `stage=resume,start_epoch` fixture | missing/mismatched parent or no stepped checkpoint | superseding receipt, old red preserved |
| telemetry lineage | one explicit resume chain | conflicting child ancestry | isolate or refuse; never merge |
| costate posterior | real `CostateEstimate(status=...)` writes | unidentifiable/nonfinite rejected | real object, durable row, readback |
| per-class feed | lineage/stage join of real handoff rows | aggregate copied across classes | area-weighted fields reach estimator without fabrication |
| default-off receipt | exact current surface set/hash | new/missing/duplicate surface | fail config finalization |
| launch readiness | actual `flags` tokens + one-rung arm plan | fictitious kebab fallback / all-rungs composition | recognizes real flags and preserves attribution |
| receiver determinism | 3 fresh identical raw/score results | changed runtime/scorer hash | exact custody or refuse |
| RDEC inverse loop | canonical endpoint closes | non-inverse mutation | component circulation zero within derived tolerances |
| mutation square | both AB and BA | only AB available | interaction and order reported separately |
| interaction semantics | synthetic commuting square `S00=0,S10=1,S01=2,S11=4` | unequal-destination add/multiply pair | interaction `1`, order `0`, circulation `0`; no 2-cell for unequal endpoints |
| click law | fixed bytes + measured Pose invariant | fixed bytes + Pose change | pure-dseg clause only in first case |
| RQTD birth | survives full receiver and fresh scorer | generator-only or receiver-erased birth | topology + exact net score/bit |
| v8 Hodge | potential/tree zero holonomy | injected cycle inconsistency | stable global labels and charged bytes |
| reduced dynamics | held-out matched branch | shuffled/unrelated trajectory | calibrated delta prediction and bounded closure error |
| saltation | event-aware model | no-jump/reset-only baseline | held-out sign/magnitude gain; no grazing denominator |
| event collision | ordered Muon/Pose/tau receipt | shuffled or missing order | forward product/reverse-adjoint product matches branch |
| tau reachability | compiled event graph reaches intended rungs | 12x250 dwell frozen at 726 | refuse unreachable schedule or declare reachable subsequence |
| D27b v2 | full-score/class/topology stable | flat Seg but worsening Pose/rate/frozen steps | `PROBE_READY` only |
| per-pair Pose | actual per-pair vector | aggregate-filled constant | locality consumers refuse fallback |

## 13. Roadmap and outstanding advisory work

### P0 — apparatus truth before optimization

1. Repair the dry-start parser against the trainer's real resume schema and issue a correction receipt.
2. Add the pass-2 checkpoint/NPZ and max-RSS assertions described in section 1.3.
3. Land unconditional resolved-config telemetry and segment/lineage envelopes.
4. Make telemetry analyzers select/refuse lineages instead of concatenating anonymous logs.
5. Land Q1 per-group activation and rename global pressure so it cannot masquerade as actual clipping.
6. Distinguish loss-active chroma from receiver-causal score value.
7. Replace D27b with the v2 probe-readiness contract.
8. Repair `CostateEstimate.status` → posterior recording, test the real dataclass, and join real
   per-class handoff rows into the estimator by lineage/stage.
9. Land the actual curriculum-pool store/module/digest or retract the memo's landed language.
10. Upgrade #405 to a hashed current-surface `ConfigDecisionReceipt` after warn-only calibration;
    repair `flags`/`launch_flag_tokens` and expose typed one-rung v7.5.3 composition.
11. Repair the canonical click-polish formula/domain while preserving its exact empirical anchor.
12. Split action interaction, order commutator, and loop circulation schemas.
13. Fail pair-local Pose consumers closed when only aggregate Pose exists; unify event-versus-cap
    treatment receipts and refuse the unreachable 12-octave/epoch-726 schedule as currently stated.

### P1 — receiver calculus

14. Build `ReceiverState.v1` and a fresh-process deterministic baseline runner.
15. Build the counted-section receiver census.
16. Build one reversible legal mutation plus `RDECEdgeReceipt` and inverse loop.
17. Build AB/BA mutation squares and migrate the existing interaction ledger as an input, not an order proof.
18. Recurse over member/block/carrier/class/pair/footprint/symbol scales.
19. Add exact component closure and separate CPU/CUDA complexes.
20. Build the first class/site RQTD insertion compiler and rejection taxonomy.
21. Add cubical persistence/RAG/junction receipts at generator, raw receiver, and scorer-cell surfaces.

### P2 — hybrid observer

22. Define the reduced state chart with units and missing-data semantics.
23. Add matched checkpoint microbranch specifications for one continuous and one categorical control.
24. Identify `A,B` with covariance and held-out validation.
25. Add one event reset receipt, beginning with tau+LR or Muon.
26. Fit saltation/discrete jump only when transversality and denominator checks pass.
27. Backtest terminal component deltas against no-jump and reset-only baselines.
28. Measure real action wall time, reversibility, information value, and complete-archive rate.
29. Keep all HAC output advisory until a separately governed actuation gate exists.

### P3 — v7.5.3 and v8 full stack

30. Recompile the v7.5.3 candidate graph only after P0 receipts are green.
31. Measure geometry/texture/Pose/rate interaction and route nonadditivity into hyperedges.
32. Require topology/transversality clearance at every curriculum boundary.
33. Establish exact task/bit slopes before enabling rate dual advice.
34. Build and compare v8 centered-potential and Hodge-safe tree receivers.
35. Charge global-label, cycle, tree, header, exception, and path-stretch bytes.
36. Couple shared-edge continuation decisions across both incident classes and junction guards.
37. Build task-adapted integer correction atoms only after the global partition is receiver-closed.
38. Measure conditional Pose innovation rate given geometry, ego motion, and temporal history.
39. Preserve PR128/HNeRV as a matched control on complete bytes/runtime/receiver/axis, not a byte-nibbling objective.
40. Promote nothing without a complete archive, exact parse-back, provenance, and separate exact axis receipts.

## 14. Literal dispositions and blockers

| item | disposition | blocker / reason |
|---|---|---|
| RDEC record/schema build | **ADVISORY DESIGN COMPLETE / BUILD HOLD** | no execution or source-edit authority in this unit |
| RQTD one-site compiler | **ADVISORY DESIGN COMPLETE / BUILD HOLD** | legal mutation/compiler and receiver topology receipts absent |
| HAC observer | **ADVISORY DESIGN COMPLETE / BUILD AND ACTUATION HOLD** | lineage, matched branches, linearizations, event receipts, backtest absent |
| v7.5.2 real launch | **HOLD** | no authority; canonical dry-start report lacks a corrected green receipt |
| v7.5.3 training/ladder | **HOLD** | P0 apparatus and measured isolated winners incomplete |
| v8 training | **HOLD** | no receiver-closed potential/tree implementation, topology/Hodge proof, or matched archive |
| independent v8 edge payload | **HOLD** | five cycle degrees plus global-label debt remain unpriced |
| curriculum pool as landed controller state | **REFUSE CLAIM** | current memo correctly says design-only; store/module/digest do not exist |
| #405 as final optimality proof | **REFUSE CLAIM** | snapshot consistent, consume gate syntactic and bypassable |
| current costate posterior | **REFUSE CLAIM** | real object field mismatch prevents recording |
| current per-class costate | **UNIDENTIFIABLE** | estimator receives verdicts, not required handoff per-class rows |
| current D27b as terminal authority | **REFUSE CLAIM** | d_seg-only anonymous-lineage heuristic |
| current tau/event graph as executable optimum | **REFUSE CLAIM** | most octaves unreachable before Muon freeze; event/cap resets and collision order differ or are unstamped |
| current v7.5.3 readiness/build path | **HOLD** | `flags` schema not consumed and standard builder cannot arm the registered rungs |
| click-polish pointer row | **PRESERVE** | exact full-score implementation/eval anchor stands; equation's pure-dseg clause needs correction |
| PR128/HNeRV | **CONTROL / DONOR** | not the task-space witness mission |
| pointer/CUDA claim | **UNCHANGED / CUDA UNMEASURED** | advisory-only; no new exact eval |

Exact remaining blockers are therefore: no corrected dry-start receipt; no governed launch authority; no
telemetry lineage; Q1–Q7 absent; no actual curriculum pool; broken cross-run posterior writer; existing
observer/per-class math defects; both LADDER thresholds zero; unreachable tau ladder and unstamped event
reset ordering; malformed v7.5.3 readiness/builder wiring; no RDEC/RQTD artifacts; no matched HAC branches
or event Jacobians; no v8 receiver/global-label/Hodge proof; no current-archive CUDA row; and no
execution authority in this advisory campaign.

## 15. Primary research grounding

- discrete cochains, exterior derivative, duals, and Hodge structure: Desbrun, Hirani, Leok, and
  Marsden, [*Discrete Exterior Calculus*](https://arxiv.org/abs/math/0508341);
- DEC in variational computer vision: Desbrun, Hirani, and Marsden,
  [*Discrete Exterior Calculus for Variational Problems in Computer Vision and Graphics*](https://www.geometry.caltech.edu/pubs/DHM03.pdf);
- topology-changing differentiable geometry: Mehta, Chandraker, and Ramamoorthi,
  [*A Theory of Topological Derivatives for Inverse Rendering of Geometry*](https://openaccess.thecvf.com/content/ICCV2023/papers/Mehta_A_Theory_of_Topological_Derivatives_for_Inverse_Rendering_of_Geometry_ICCV_2023_paper.pdf);
- differentiable segmentation topology: Hu, Fuxin, Samaras, and Chen,
  [*Topology-Preserving Deep Image Segmentation*](https://arxiv.org/abs/1906.05404);
- hybrid event linearization: Kong et al.,
  [*Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems*](https://arxiv.org/abs/2306.06862);
- hybrid adjoint jumps: Corner, Sandu, and Sandu,
  [*Adjoint Sensitivity Analysis of Hybrid Multibody Dynamical Systems*](https://arxiv.org/abs/1802.07188).

These sources ground the mathematical operators. Every Pact-specific admission rule above is a
derived proposal and remains receiver-exactly falsifiable.

## 16. STORES CONSULTED and cathedral triality

**STORES CONSULTED:** `CLAUDE.md` · `AGENTS.md` · top Pact Claude memory · canonical pointer/frontier
surfaces · lane/subagent/directive state · v7.5/v8 SPECs · restart handoff and completed vehicle
advisories · prior evaluator-quotient advisory · #404 telemetry source/tests/memo · #403 curriculum
memo and claimed paths · #405 memo/JSONL/gate/tests · costate estimator/posterior/shadow source/tests ·
action-commutator source/tests · click-polish source/canonical equation · v7.5.2 pilot and final
dry-start launch/log/checkpoint/report artifacts · owed16v2 terminal log/checkpoints/verdict · primary
sources above.

**Triality future wire-in:**

- **DSL:** typed legal mutations, curriculum candidates, event controls, and config decision receipts;
- **DAG:** authority repair → receiver census/RDEC → RQTD → matched dynamics → event adjoint backtest
  → isolated v7.5.3 graph → v8 Hodge receiver;
- **equations:** exact component score cochain, interaction/order/circulation separation, RQTD value,
  Hodge constraints, reduced adjoint, and saltation with explicit domains/falsifiers;
- **costate/autopilot:** read-only consumers only until empirical closure and a separately governed
  actuation gate.

**Pointer delta:** exactly zero. This advisory is a durable build contract, not score evidence.
