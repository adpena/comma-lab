# ADVISORY — SDF witness exit criteria and hundred-year program — 2026-07-10

**Status:** `research_only=true` · standalone exit contract · advisory specification only · no
training, dispatch, evaluation, pointer move, process signal, live-run mutation, source edit, or
engineering-actuation authority.

**Lane:** `lane_advisory_codex_v752_v753_v8_fresh_eyes_20260710`. The registered lane permits only
new `ADVISORY_*.md` outputs.

**Mission:** define the exact evidence required to exit every stage of Pact's original task-space
SDF / level-set witness program, from receiver correctness through frontier promotion and the
long-horizon research program. Exit means a preregistered machine-checkable predicate has a durable
receipt. It never means “looks promising.”

**Relationship to prior work:** this contract specializes the roadmap and blocker set in
`ADVISORY_sdf_receiver_neighborhood_atlas_first_probe_20260710.md`. PR128 remains an HNeRV-family
payload-polish child; its useful receiver, parser, exact-consumption, entropy, custody, and discrete
search techniques transfer into this SDF program without transferring its representation objective.

## 0. Authority snapshot and instruction boundary

At the refresh used to write this contract:

| field | exact value / disposition |
|---|---|
| checkout | `e8d080dd807175ba197cc398ee4fb23c8bfebba8` on local `main`; one commit ahead of `origin/main` at observation |
| `CLAUDE.md` | SHA-256 `52405bac18c6227df1d99b597a2f55987614f035e501eb74a9d08be48e1dbdd7` |
| `AGENTS.md` | SHA-256 `d2bdceb42d394d78bac4f9ddcaa9e0b3758d0be206fea2920784ccdc6f2ec495` |
| contest-CPU pointer | `0.19108282419209976 [contest-CPU]` |
| pointer archive | `177,169 B`; SHA-256 `ad02b0124cbb3405c23d3480ac16f12b4e48cbf6f75878dd77a5e621bebd079c` |
| pointer CUDA | **UNMEASURED for those exact bytes** |
| FEED-417 | refusal gate plus optional-family receiver consumption/parity landed at `e8d080dd8`; macOS advisory parity only; full R1 authority receipt remains open |
| live v7.5.2 | preserve and observe only |
| live click-polish control | preserve and do not duplicate or harvest |
| this unit | new advisory document only |

The user's standing objective is to continue until the exit criteria are met while producing solely
advisory documents. This creates an important authority distinction:

- this lane may specify a gate, audit evidence already produced elsewhere, and declare an exact
  blocker;
- this lane may not create the missing code, run, archive, score, dispatch, pointer move, or process
  transition;
- therefore `ADVISORY_SPEC_COMPLETE` is reachable here, while `ENGINEERING_GATE_PASS` is reachable
  only when independently authorized execution produces the required evidence;
- absence of authority is `ACTUATION_NOT_AUTHORIZED`, not evidence that a method failed.

## 1. Universal exit contract

Every gate must be declared before the evidence used to judge it and must contain:

```text
gate_id
schema_version
verdict_scope
authority_axis
entry_state_hashes
receiver/runtime/scorer/source hashes
seed + complete typed config
required predicates
threshold derivations
falsifiers
required artifacts
PASS destination
COMPLETE_NONPROMOTABLE destination
BLOCKED destination
reopen condition
STORES CONSULTED
triality legs
```

Let `P_i` be mandatory predicates. Gate passage is conjunctive:

\[
\operatorname{PASS}(G)=\bigwedge_i P_i.
\]

No weighted average, composite score, or favorable headline may compensate for a failed mandatory
predicate.

### 1.1 Allowed terminal statuses

| status | exact meaning |
|---|---|
| `RECEIPT_COMPLETE` | one state/edge/run receipt has every required field and custody identity |
| `EXPERIMENT_EXECUTED` | every preregistered arm/state was measured; no claim about the next gate follows |
| `EXPERIMENT_CLOSED_FOR_NEXT_GATE` | the experiment's registered falsifiers and transition predicates permit entry to the named next gate |
| `ACTUATOR_ELIGIBLE` | held-out, uncertainty, rollback and exact-score predicates permit later actuation under separate authority |
| `AXIS_CHALLENGER` | one exact archive strictly beats the refreshed control on one declared contest axis |
| `FRONTIER_PROMOTABLE` | the same exact archive passes every promotion predicate on both required contest axes |
| `PASS` | every mandatory predicate of the specifically named gate holds on its declared authority surface |
| `EVIDENCE_COMPLETE_NONPROMOTABLE` | the experiment answered its registered question, but next-gate, actuation or promotion predicates failed |
| `BLOCKED` | required authority, custody, identifiability, or input evidence was unavailable; no scientific negative follows |
| `FORMULATION_NEGATIVE` | the declared formulation was fairly tested and falsified within its registered scope |
| `ACTUATION_NOT_AUTHORIZED` | a design/evidence contract exists but this lane cannot execute it |

`PARTIAL_PASS` is forbidden. A missing required field is not zero; it is `UNIDENTIFIABLE` or
`MISSING_AUTHORITY`. Completion states form a strict ladder: no receipt-, experiment-, next-gate-,
actuator-, or single-axis status may move the dual-axis frontier.

Gate-specific tokens are typed aliases of the base ladder, not additional unranked statuses:

| gate-specific token | base status / meaning |
|---|---|
| `LVLS1_CENTER_CPU_ATLAS_ELIGIBLE` | `EXPERIMENT_CLOSED_FOR_NEXT_GATE` for R3 → R4 |
| `CELL_EXECUTED` | `EXPERIMENT_EXECUTED` |
| `CELL_2CELL_CLOSED` | `EXPERIMENT_CLOSED_FOR_NEXT_GATE` for face/Hodge construction |
| `CELL_LOCAL_MODEL_ADMISSIBLE` | `EXPERIMENT_CLOSED_FOR_NEXT_GATE`; not yet actuator eligibility |
| `RQTD_EXECUTED` | `EXPERIMENT_EXECUTED` |
| `PREDICTOR_HODGE_AUDITED` | `EXPERIMENT_CLOSED_FOR_NEXT_GATE`; actuator predicates remain separate |
| `TOPOLOGY_ACTUATOR_ELIGIBLE` | `ACTUATOR_ELIGIBLE` for one exact topology atom |
| `INSTANCE_SIGNAL` | `EVIDENCE_COMPLETE_NONPROMOTABLE` |
| `ADVISORY_SPEC_COMPLETE` | advisory-layer `RECEIPT_COMPLETE` |
| `ADVISORY_EVIDENCE_CLOSED` | advisory-layer `EXPERIMENT_CLOSED_FOR_NEXT_GATE` |

`MUTATION_NEGATIVE` through `PARADIGM_NEGATIVE` are verdict-scope values carried by the base
`FORMULATION_NEGATIVE`/`NEGATIVE(scope=...)` outcome; they do not form a promotion ladder.

### 1.2 Evidence hierarchy

From strongest to weakest:

1. exact contest-axis receipt on the complete archive bytes;
2. exact fresh inflate plus frozen component scorer on the complete bytes;
3. deterministic receiver/component measurement on a complete local center;
4. advisory CPU/MLX/Torch parity evidence;
5. proxy, screen, surrogate, subset, or design derivation.

A weaker row cannot close a stronger gate. CPU and CUDA are separate evidence axes.

### 1.3 Threshold derivation

For three or more fresh repetitions of the same exact archive and receiver, define

\[
\epsilon_{repeat}
=\max_{i,j}|S_i-S_j|,
\]

and bind those repetitions to one immutable receiver fingerprint `rho`. Define

\[
\epsilon_{det}(\rho)=\epsilon_{repeat}(\rho).
\]

and let `epsilon_formula` be the measured closure tolerance between parsed components and the canonical
score helper. Let `delta_preregistered` be a materiality floor declared before candidate outcomes.
Then

\[
\delta_{admit}
=\max(\epsilon_{repeat},\epsilon_{formula},\delta_{preregistered}).
\]

A deterministic candidate is beneficial only if

\[
\Delta S<-\delta_{admit}.
\]

For stochastic or across-seed decisions, the preregistered confidence procedure must satisfy

\[
UCB_{95}(\Delta S)<-\delta_{admit}.
\]

Adaptive selection requires a separate confirmation set or a preregistered multiplicity correction.
The candidate-selection screen cannot supply its own confirmation evidence.

Hash equality has zero tolerance. Archive, member, raw, checkpoint, config, runtime, and source
identities either match exactly or do not.

## 2. Gate R0 — custody, legality, and deterministic identity

Every later gate inherits R0. It exits only when:

- archive SHA-256 and exact byte count are recorded;
- every member path, byte range, size, and SHA-256 is recorded;
- the grammar consumes the entire archive/member with no ignored trailing video-derived data;
- inflate/runtime tree, dependency versions, evaluator and scorer hashes are recorded;
- git/source hash, dirty-source manifest, seed, typed config, hardware and axis are recorded;
- output cardinality and dtype match the contest contract;
- three fresh decodes are byte-identical;
- runtime is within the declared contest bound;
- no scorer weights, GT argmax table, hidden video-derived code payload, mutable remote dependency, or
  authority-changing cache is present;
- large scratch has a certified storage/cleanup record before release;
- every result names the exact authority surface and sample count.

Any custody mismatch makes all downstream score, topology, Hodge, and costate claims
`NONAUTHORITATIVE` until remeasured.

## 3. Gate R1 — receiver-consumption bijection

Let

- `C_counted` be learned/video-derived parameter groups charged to the archive;
- `C_consumed` be groups used by the canonical receiver;
- `C_live` be learned/video-derived state that can affect decoded output.

The required bijection laws are

\[
C_{counted}\subseteq C_{consumed},
\qquad
C_{live}\subseteq C_{counted}.
\]

For archive `A` and immutable receiver fingerprint `rho`, the full predicate is

```text
BIJECTION_CLOSED(A, rho) iff
  manifest_counted_keys == serialized_learned_keys
  and dynamic_access == false
  and allow_unconsumed == empty
  and every counted key is consumed by the shipped inflate source
  and every optional family has an actual-archive signed mutation
      that changes the expected raw support through R_rho
  and MLX/NumPy/inflate parity uses the same manifest and shapes
  and no live video-derived parameter is omitted from counted bytes.
```

R1 exits only when:

- no counted group is absent from the receiver;
- no live learned/video-derived state is excluded from byte accounting;
- `dynamic_access=false` for the static consumption proof;
- `TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS` is absent;
- unknown manifest groups fail closed;
- MLX, NumPy, byte-close and inflate implement the same manifest-conditioned forward;
- every optional group has a default-off negative control yielding byte-identical legacy output;
- every optional group has a nonzero mutation positive control that changes the expected through-R
  output support;
- shapes, dtypes, op order and quantization law agree across receivers;
- legacy witnesses still report zero receiver-orphan groups.

An AST reference proves only syntactic consumption. It does not prove branch activation or output
effect. Record it only as `STATIC_BIJECTION`. A synthetic mirror test is `MIRROR_PARITY`. Neither
permits a scored row. A waiver or dynamic access makes the receipt advisory-only.

At this document's pinned source snapshot, optional-family consumption and mirror/composition parity
are committed at `e8d080dd8`. The shared-head path is byte-identical, the static gate reports zero
orphans, and advisory MLX/NumPy composition differences are measured. This closes the source repair,
not the stronger R1 authority predicate: an actual-archive signed mutation, clean receiver manifest,
and exact through-R support receipt are still required for `BIJECTION_CLOSED`.

Family-specific exits:

| family | exit predicate |
|---|---|
| A1 legacy `out_tex` | normal parity plus through-R effect receipt |
| A2 `out_tex_h.*` | manifest-aware hidden-head consumption, output-shape parity, positive control |
| A3 `tex_trunk.*` | fixed-bank reconstruction, learned-coefficient consumption, exact forward parity, positive control |
| v8 `decoupled_head.*` | canonical partition actually consumes the head; class-isolation proof also passes |

The landed refusal gate changes the default failure mode from a silently fake score row to a refusal.
It does not itself close receiver consumption.

## 4. Gate R2 — full-score-safe SDF mutation/finisher

The diagonal finisher exits source and evidence review only when:

- PoseNet is evaluated from both reconstructed frames;
- real `per_pair_dpose` exists;
- real per-pair/per-class Seg components exist;
- pair IDs and array cardinality are exactly aligned to all 600 pairs;
- vector-to-aggregate closure is checked rather than assumed;
- no aggregate scalar is expanded into a fabricated pair vector;
- every candidate endpoint is serialized, parsed back, inflated and evaluated by the two-frame scorer;
- base and candidate are complete legal archives under the same immutable receiver `rho`;
- both are inflated with the same fresh receiver identity;
- actual complete-archive byte deltas enter the objective;
- all accepts use

\[
\Delta S=
100\Delta d_{seg}
+\sqrt{10d'_{pose}}-\sqrt{10d_{pose}}
+\frac{25\Delta B}{37\,545\,489};
\]

- the accepted candidate satisfies `delta S < -epsilon_det(rho)`, with any stronger materiality floor
  included through `delta_admit`;
- support hashes cover both raw frames, affected Seg cells, the Pose pair and topology;
- every acceptance banks base/candidate archives, component receipts, support fingerprints and exact
  rollback state;
- rejected and accepted rollback paths are re-inflated and reproduce their declared archive/raw state
  exactly;
- a fresh exact confirmation follows any in-process screen;
- “Pose-invariant” or “Seg-invariant” is emitted only after structural or exact raw/scorer equality;
- diagonal batching occurs only after support, interaction and order tests prove it admissible.

Missing pair IDs, vector cardinality, manifest support or component authority forces
`UNIDENTIFIABLE`; the selector must not fall back to a partial objective.

A failed coordinate is `MUTATION_NEGATIVE`. It cannot kill the finisher family or the SDF paradigm.

## 5. Gate R3 — authoritative n600 SDF atlas center

An `LVLS1` center exits specimen status only when:

- a complete n600 archive exists;
- contest-Linux CPU axis is explicit;
- grammar parse-back and complete-stream consumption pass;
- inflate emits exactly 1,200 frames;
- raw cardinality is exactly

\[
1200\cdot874\cdot1164\cdot3
=3\,662\,409\,600\ \text{bytes};
\]

- three fresh-process decodes produce identical raw SHA-256, component vectors, topology receipt and
  scalar score;
- full score closes from unrounded components and exact archive bytes;
- receiver/runtime/scorer/source/config/hardware/dependency custody is complete;
- the receiver fingerprint records batch size, thread law, tie rule, quantization/rounding law,
  manifest grammar and raw-layout law;
- R0 and R1 pass;
- inflate remains within the contest runtime limit;
- CPU evidence is not promoted as CUDA evidence.

Until R3 passes, an `LVLS1` packet is a wire-format specimen and cannot center the Receiver
Neighborhood Atlas. Passing R3 yields `LVLS1_CENTER_CPU_ATLAS_ELIGIBLE`; it is neither a frontier
claim nor CUDA evidence.

## 6. Gate R4 — five-state `C1 x X` safety cell

The required state set is

\[
\{A_0,A_C,A_X,A_{CX},A_{XC}\}.
\]

### 6.1 Experiment-complete exit

R4 is `CELL_EXECUTED` when:

- all five archives independently parse, inflate and score;
- baseline repeatability passes;
- every state has complete hashes, actual bytes, full components, per-pair/class effects and topology;
- `C1` frame/raw/Seg/Pose support is measured;
- `X` frame/raw/Seg/Pose support is measured;
- both action orders are independently constructed;
- component interaction, objective curvature and true Pose interaction are separated;
- topology/RAG/junction chart identity is present;
- missing fields remain `UNIDENTIFIABLE` rather than imputed.

The experiment may exit complete even if it falsifies locality or commutativity.

`CELL_2CELL_CLOSED` additionally requires `A_CX` and `A_XC` to share one complete destination
identity. Otherwise the durable status is `NONCOMMUTING_NO_2_CELL`; the experiment remains executed,
but no face or face-Hodge claim is admissible.

### 6.2 Actuator-admission exit

The cell authorizes later diagonal use only when:

- `C1` and `X` support claims pass on the selected manifest;
- `A_CX` and `A_XC` have identical complete destination hashes if a commuting square is claimed;
- independently observed closed-loop score circulation lies within the apparatus floor;
- no unstamped topology or receiver chart transition occurs;
- interaction magnitude is below the preregistered batching tolerance relative to admitted gain;
- every accepted direction remains full-score beneficial after exact confirmation.

`CELL_LOCAL_MODEL_ADMISSIBLE` additionally requires baseline repeat residual within
`epsilon_det(rho)`, no unexplained apparatus error `zeta`, observed rather than asserted support, and
every topology crossing stamped `CRITICAL_FACE` and removed from that chart's Jacobian.

Different destination hashes require `NONCOMMUTING_NO_2_CELL`, not a nonzero-curl claim.

## 7. Gate R5 — 24-state Receiver-Quotient Tangent Dictionary

The exact core contains:

- three independent center replays;
- ten signed singleton endpoints across `{C0,C1,G,T,P}`;
- four interaction pairs in both orders;
- three inverse returns.

`T` means the legacy receiver-consumed linear `out_tex`; receiver-blind optional families are excluded.

### 7.1 Dataset-complete exit

- status is `RQTD_EXECUTED`, not automatically `PREDICTOR_HODGE_AUDITED`;
- all 24 state receipts are complete;
- the atom alphabet and deterministic coordinate-selection rule were preregistered;
- singleton, interaction, order and inverse blocks are all present;
- each row contains raw/scorer/topology/archive support and wall-time/resource measurements;
- every inverse reproduces the exact center archive and raw hashes;
- storage cleanup preserves deterministic rebuild manifests;
- no candidate-selection proxy is mislabeled authority.

### 7.2 Local-model-identifiable exit

- the five-direction design has declared rank;
- full five-dimensional claims require rank five;
- the smallest singular value exceeds the floor derived from measurement noise and desired response
  resolution;
- condition number is below a preregistered numerical-identifiability ceiling;
- no central difference crosses an unstamped chart;
- signed contrast and lattice curvature are stored:

\[
D_i=\frac{Y_{i+}-Y_{i-}}2,
\qquad
Q_i=Y_{i+}+Y_{i-}-2Y_0;
\]

- interaction, order and apparatus circulation are distinct fields;
- held-out chords remain untouched during fit.

Rank deficiency is an informative atlas result but blocks claims about missing coordinates.

## 8. Gate R6 — predictor/Hodge authority

Exact score edges first must satisfy the apparatus identity

\[
\zeta=\omega_{observed}-B_1^Ts.
\]

R6 refuses predictor analysis if `zeta` exceeds the exact repeatability/formula floor on a genuinely
closed legal loop.

For predictor error

\[
e=B_1^Ts-\widehat\omega
=d_0u+\delta_2\psi+h,
\]

predictor-model exit requires:

- versioned oriented `B1`, `B2` and immutable vertex identities;
- after gauge fixing, `rank(B1)=|V|-1` on each claimed connected component;
- `B1 B2=0` on the declared complex;
- weights, adjoints, gauges and nullspaces are explicit;
- gradient, coexact and harmonic terms reconstruct measured error within numerical tolerance;
- held-out chords were not used for fitting or selection;
- no held-out edge with `abs(delta S) > delta_admit` has the wrong predicted sign;
- held-out error is below the controller's decision margin;
- confidence intervals satisfy preregistered coverage and multiplicity handling;
- curl/harmonic mass capable of reversing a decision forces predictor refusal;
- archive-mutation and v8 class-incidence complexes remain separate.

No universal residual or information threshold is inferred from the 24-state count. The stopping rule
must be preregistered from repeated exact receipts, desired decision margin and measured evaluation
cost. A Hodge-audited predictor may rank candidates; every candidate remains independently full-score
gated.

Low training residual alone never makes a predictor actuator-admissible.

## 9. Gate R7 — legal topology-changing atoms

`Q_top` is build-complete only when every declared disk/ellipse, island, bridge, hole or junction form
satisfies:

- a typed bounded integer parameterization;
- deterministic apply/inverse, or an explicit `NONREVERSIBLE` classification that prohibits inverse
  and closed-loop claims;
- serializer/parser round-trip exactly;
- inverse mutation reproduces original bytes and raw output;
- intended generator-field event occurs;
- the event survives rasterization, resize and uint8;
- SegNet logits/argmax and PoseNet effects are measured;
- foreground 4-connectivity and background 8-connectivity conventions are fixed;
- per-class `beta0`, `beta1`, component identities, RAG and junction ordering are recorded;
- the outcome is stamped `SAME_CHART`, `CRITICAL_FACE`, or `UNIDENTIFIABLE`;
- protected classes have no unregistered collateral birth, death or merge;
- actual complete-archive byte delta is measured;
- exact full-score value clears `delta_admit`;
- persistence/topology losses remain predictors rather than evaluator authority.

A generator-only event that disappears through the receiver is `FORMULATION_NEGATIVE` for that atom.
`TOPOLOGY_ACTUATOR_ELIGIBLE` additionally requires an exact individual full-score/byte receipt;
topology priority alone carries no authority.

## 10. Gate R8 — six-leaf Muon-boundary observer

### 10.1 Experiment-complete exit

- one complete full-state checkpoint exists at or before `e_star-Delta` and contains model,
  optimizer, EMA, RNG, controller, stage, schedule and receiver state;
- every leaf has identical ancestry and differs only in the declared reset/event variables;
- three matched AdamW-boundary and three actual-Muon leaves exist at
  `e_star-Delta`, `e_star`, `e_star+Delta`;
- `Delta` is derived from the run's measurement/checkpoint cadence and the derivation is emitted in
  the branch manifest; it is never a durable hardcoded constant;
- all leaves terminate at the same absolute horizon;
- Pose and other events are held common or stamped as a compound transition;
- exact event collision order is recorded;
- every stage checkpoint remains complete and preserved;
- full state, component, receiver, topology and custody receipts exist.

### 10.2 Directional-observer exit

- four corner leaves fit both event-time slopes and the categorical Muon interaction;
- two center leaves remain held out;
- center residuals lie inside their componentwise measured floors;
- predicted categorical jump agrees with the observed center jump in sign and magnitude;
- immediate matched decoder/EMA outputs are identical before optimizer evolution;
- design rank and conditioning pass;
- the topology chart remains common across central differences;
- real per-pair Pose exists.

### 10.3 Controller-actuation exit

- at least one additional seed or a registered across-seed variance floor exists;
- selection/confirmation multiplicity is controlled;
- the recommended switch satisfies `UCB95(delta S) < -delta_admit`;
- no claim of a full state matrix, full reset Jacobian, continuous saltation matrix or full optimizer
  costate is made from the six leaves.

With one seed the maximum status is `INSTANCE_SIGNAL`.

## 11. v7.5.3 stage exits

Every stage transition occurs at a preserved complete EMA checkpoint. Loss weights are fixed within a
stage.

Every stage record carries input checkpoint/hash, fixed in-stage loss weights, liveness and positive-
control clearance, receiver/chart ID, state predicate, complete output EMA checkpoint, and exactly one
transition: `advance`, `hold`, or `stop(scope,reactivation)`.

| stage | mandatory exit predicate |
|---|---|
| custody/identity | R0-R3 pass; NumPy/MLX/Torch receiver identities close; exact center exists |
| coarse topology | required classes exist; forbidden islands/holes bounded; area dual and topology signatures stable for a full measurement cadence |
| geometry/pose | exact component debt falls beyond floor; no protected Seg regression; pose support and byte price measured |
| appearance | A1/A2/A3 are receiver-valid matched arms; entry checkpoint preserved; a new chart atlas is measured |
| quantized receiver | serialize/inflate agrees with the declared candidate within parity; all decisions use realized-through-R evidence |
| terminal discrete | every accepted mutation clears exact full-score threshold, has rollback and complete custody |
| vehicle promotion | current-pointer comparison passes on a declared contest axis with all promotion predicates |

Old numeric Jacobians, costates and mutation values expire at every chart transition until remeasured.
This curriculum contract does not override v7.5's sealed owed-before-launch chain or its measured Pose
gate. No v7.5 threshold transfers to v7.5.3 without a new manifest-matched receipt.

## 12. v8 stage exits

v8 inherits every v7.5.3 gate and adds:

- v7.5 fires first and a manifest-matched receipt shows it missed the registered target trajectory;
- the v8 P-C geometry phase completes before paint design begins;
- design review explicitly closes or mitigates all six named risks in
  `SPEC_v8_perclass_decomposition_20260708.md`;
- the full fix-all seal reaches three clean passes and the n600 gate closes;
- a one-class optimizer step leaves non-target class parameters bit-identical;
- non-target class field values remain identical; legitimate final argmax changes caused by competition
  are recorded separately;
- shared pair-code coupling is eliminated or explicitly modeled and priced;
- one stable class-label ordering and kill-predicate semantic is used everywhere;
- each RAG edge names both incident classes and orientation;
- `merge`, `diff`, and `correct` have separate before/after receipts;
- a spanning-tree carrier basis is identifiable before adding cycle chords;
- coexact terms exist only for explicitly declared 2-cells;
- harmonic cycle payload earns positive exact score value per byte;
- increment 1a remains mask-level and non-promotional;
- increment 1b exits only with actual decoupled receiver consumption and exact through-R evidence;
- increment 1b has independent carrier byte-close and resume proof;
- receipts preserve the staged order `fields -> paint -> merge -> diff -> correct`;
- only schema/method transfers from v7.5.3; numerical effects never transfer.

The v8 formulation remains intact when a current receiver implementation fails. The correct negative
scope is implementation/formulation until the architecture itself is fairly falsified.

## 13. Frontier-promotion exit

A candidate moves the local pointer only when:

- exact archive bytes are frozen and hashed;
- exact n600 evaluator completes;
- hardware axis and substrate are contest-compliant and explicit;
- score is recomputed from unrounded components and exact bytes;
- candidate clears the current same-axis pointer by more than `delta_promote`;
- three fresh decodes are deterministic;
- legality, output cardinality and runtime limit pass;
- no forbidden scorer/GT/video-derived hidden payload exists;
- complete source/runtime/archive/scorer/config/hardware provenance exists;
- the canonical helper refreshes the pointer from evidence rather than manual prose;
- the pointer record names the exact archive SHA and measured time;
- public or author-claimed scores remain external/unratified until exact replay.

Those predicates yield `AXIS_CHALLENGER` on the measured axis. Dual-axis promotion is stricter:

```text
FRONTIER_PROMOTABLE iff
  one exact archive SHA is deterministic and contest-compliant
  and clean-runtime inflate finishes within budget
  and exact n600 upstream/evaluate.py receipts exist on 1:1 contest CPU and CUDA
  and every receipt binds source/runtime/scorer/hardware/archive provenance
  and the candidate beats each refreshed same-axis control beyond derived uncertainty
  and all custody, receiver, parse-back and no-hidden-data gates pass.
```

A contest-CPU-only SDF result is a CPU challenger/atlas receipt, not dual-axis frontier promotion.
Neither axis is inferred from the other.

## 14. Negative, kill, stop, and reopen criteria

### 14.1 Verdict scope ladder

Every negative names one of:

```text
MUTATION_NEGATIVE
COORDINATE_NEGATIVE
INSTANCE_NEGATIVE
FORMULATION_NEGATIVE
VEHICLE_NEGATIVE
FAMILY_NEGATIVE
PARADIGM_NEGATIVE
```

Escalation requires an explicit proof. One failed mutation, seed, proxy, subset or receiver
implementation cannot kill a family.

An exact negative receipt names archive, receiver, axis, typed config, full component vector,
apparatus-validity status, positive-control status, narrowest verdict scope and reactivation criterion.
A gate failure without a fair authoritative test is `BLOCKED`, not a method negative.

### 14.2 Rigorous vehicle/family kill evidence

A broad kill is permitted only if at least one registered proof closes, such as:

- a certified minimum byte count makes the rate term alone unable to beat the target even at zero
  distortion;
- a certified relaxation gives a best-achievable component lower bound above the target;
- every legal member of a preregistered finite atom family has been exhausted;
- required receiver rank is structurally unattainable;
- a certified runtime lower bound exceeds the contest limit;
- legality prevents carrying the necessary sufficient statistic.

These are not kill evidence: governor refusal, missing implementation, current budget, one seed, one
coordinate, one topology crossing, one proxy regression, or absence of an authorized launch.

### 14.3 Campaign stopping exit

The campaign may stop only when:

- no unmeasured legal candidate has expected improvement above the admission floor at acceptable
  measured cost; or
- all remaining uncertainty is below decision materiality; or
- a certified lower bound proves the registered target unreachable under the declared representation.

Otherwise status is `DEFERRED`, `BUILD_BLOCKED`, or `ACTUATION_NOT_AUTHORIZED`.

Every stopped item records a concrete reopen condition: new receiver support, a new exact center,
lower measured noise, new legal atom, better certified bound, or new authority.

Before any governed stop or teardown, preserve every complete stage checkpoint and write the terminal
reason, last good state, process disposition and rebuild/custody manifest.

## 15. Advisory-only handoff and campaign exit

Because this campaign's deliverables are solely advisory documents, advisory closure has its own
criteria.

### 15.1 `ADVISORY_SPEC_COMPLETE`

- every gate has entry state, mandatory predicates, thresholds, falsifiers and terminal states;
- every number is measured, derived, or explicitly unresolved;
- current evidence and missing evidence are separated;
- every negative has verdict scope and reopen condition;
- every proposed execution has a later-authority requirement;
- every live/shared ownership boundary is named;
- all source and evidence paths exist at validation time;
- STORES CONSULTED and triality legs are recorded;
- the advisory validates mechanically and is committed as the sole owned file.

Every handoff also names owner and lane ID, immutable state/custody hashes, live-process disposition,
next safe action, explicit launch/dispatch/signal history, and the exact remaining blocker. Missing any
one of these prevents no-signal-loss handoff closure.

### 15.2 `ADVISORY_EVIDENCE_CLOSED`

This stronger status requires every engineering gate to link to either:

- a passing exact receipt;
- a complete nonpromotable result with scoped falsifier; or
- an exact blocker with owner, missing artifact and next disambiguating measurement.

Chat-only findings, vague “TBD,” unowned blockers and inferred axes prevent evidence closure.

### 15.3 `ENGINEERING_GATE_PASS`

An advisory may report this only by citing independent exact evidence. It cannot manufacture passage
through prose. If execution remains unauthorized, the honest exit is a complete advisory plus exact
open blockers.

## 16. Hundred-year phase exits

A century plan is a research constitution, not a credible fixed calendar. Precise far-future numeric
thresholds would be fabricated; each phase must re-preregister its metrics on a rolling horizon while
preserving the invariant of minimal legal task-sufficient representations.

The authoritative phase contract is state-based:

```text
Foundation  -> all P0 custody/correctness gates close
Measurement -> safety cell and RQTD execute with no false Hodge authority
Control     -> held-out predictor and multi-seed event evidence close;
               every candidate remains independently exact-gated
Frontier    -> at least one dual-axis FRONTIER_PROMOTABLE archive exists
Stewardship -> triality, resume, custody, canonical consumers, handoff and
               reactivation records remain independently reproducible
```

The following time bands are planning cadences, never substitutes for the state exits:

| horizon | phase exit |
|---|---|
| 0-3 months | authoritative SDF center, receiver-safe full-score finisher, five-state cell and 24-state atlas complete |
| 3-12 months | byte-closed SDF witness beats the then-current exact CPU frontier; same bytes obtain separately measured CUDA evidence |
| 1-3 years | given a sealed evaluator, receiver and legal grammar, the compiler proposes, measures, rejects and promotes witnesses end-to-end with deterministic receipts and no manual mutation choice |
| 3-10 years | quotient-space lower bounds predict achievable rate/distortion on held-out tasks tightly enough to prune impossible representations before full training, with calibrated coverage |
| 10-25 years | one typed witness IR and proof system works across multiple perception/control evaluators without task-specific custody exceptions |
| 25-50 years | task-sufficient transmissions preserve closed-loop behavior inside a preregistered regret bound while using materially less information than observation reconstruction |
| 50-100 years | no honest fixed terminal metric exists today; success is redefined in rolling ten-year contracts while independently reproducing minimal legal sufficient representations |

The durable century invariant is the loop

\[
\text{declare authority}
\to\text{compile legal actions}
\to\text{measure receiver geometry}
\to\text{infer costates/topology}
\to\text{allocate bits and experiments}
\to\text{exactly verify}
\to\text{learn from the receipt}.
\]

The deepest universal exit criterion is:

> State exactly which information is necessary; prove why the rest is unnecessary; compile the
> necessary statistic into a legal deterministic witness; and reproduce its task behavior
> independently from the same counted bytes.

Anything weaker is an experiment or a bound, not completion.

## 17. Literal current dispositions and blockers

| surface | current disposition |
|---|---|
| universal exit contract | **SPECIFIED IN THIS ADVISORY** |
| receiver refusal plus optional-family consumption | **SOURCE/PARITY LANDED `e8d080dd8`; FULL R1 AUTHORITY RECEIPT OPEN** |
| `out_tex_h.*`, `tex_trunk.*`, `decoupled_head.*` authority | **SOURCE-ADMISSIBLE; ACTUAL-ARCHIVE EFFECT + EXACT R1 STILL OWED** |
| joint SDF code/Pose finisher | **ACTUATION REFUSE UNTIL R2 PASSES** |
| exact n600 `LVLS1` center | **ABSENT; R3 BLOCKED** |
| five-state cell | **DESIGN COMPLETE; EXECUTION NOT AUTHORIZED** |
| 24-state RQTD | **DESIGN COMPLETE; DEPENDS ON R3/R4** |
| Hodge predictor | **DESIGN COMPLETE; NO COMPLETE ATLAS TO FIT** |
| topological atom compiler | **MISSING** |
| six-leaf Muon observer | **BLOCKED BY PRE-EVENT CHECKPOINT, HARNESS, COLLISION AND SEED EVIDENCE** |
| v7.5.3 promotion | **BLOCKED BY R1-R5 AND FULL VEHICLE RECEIPT** |
| v8 promotion | **BLOCKED BY R1, CLASS ISOLATION, INCREMENT-1B RECEIVER AND EXACT RECEIPT** |
| pointer | **UNCHANGED BY THIS ADVISORY** |
| training/dispatch/eval/signal | **NONE AUTHORIZED OR PERFORMED** |

## 18. Triality, stores consulted, and pointer-delta honesty

### DSL leg

The advisory defines the typed fields and terminal statuses for an eventual `exit_gate.v1` record.
It invents no trainer flags and authorizes no config mutation.

### DAG leg

```text
R0 custody
  -> R1 receiver bijection
  -> R2 full-score mutation safety
  -> R3 exact SDF center
  -> R4 five-state safety cell
  -> R5 24-state atlas
  -> R6 predictor/Hodge authority
  -> R7 topology atoms
  -> v7.5.3/v8 vehicle gates
  -> exact frontier promotion

separate dynamic branch:
pre-event checkpoint
  -> R8 six-leaf observer
  -> multi-seed confirmation
  -> controller advice
  -> actuation only under later authority
```

### Equation leg

The controlling equations are conjunctive gate passage, exact nonlinear score difference, derived
admission floor, receiver counted/consumed/live inclusions, RDEC closure, predictor-error Hodge
decomposition, signed lattice contrast/curvature, and confidence-bounded actuation.

### STORES CONSULTED

- full `CLAUDE.md` and `AGENTS.md` with unchanged session hashes;
- current top-10 Pact Claude memory entries;
- canonical pointer, lane registry, subagent ownership/progress, dirty-tree and directive surfaces;
- `ADVISORY_sdf_receiver_neighborhood_atlas_first_probe_20260710.md`;
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`, the canonical v7.5.3
  advisory/design surfaces, `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`, and their
  receiver/curriculum source anchors; and
- current FEED-417 receiver-consumption ownership state.

### HISTORICAL_PROVENANCE / pointer delta

Derived against checkout `e8d080dd807175ba197cc398ee4fb23c8bfebba8`. Concurrent shared work may
advance `main`; this hash is a derivation anchor, not a claim that HEAD remains fixed.

The canonical pointer delta caused by this unit is exactly **zero**. No archive, source, state ledger,
run, checkpoint, dispatch, evaluator, pointer, or process was changed. The sole owned output is this
new advisory document.
