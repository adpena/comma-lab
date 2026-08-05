# AM1 Crosswalk Receipt - Acceleration Matching

## Answer First

AM1 does not produce a measured Pact score move today. The paper's useful transfer is not its trajectory-inference problem: Pact already measures the relevant ego-motion and solver streams at encode time. The transferable object is representational: a smooth phase-space or acceleration-field generator can be a compact counted packet and a smoothness prior for known streams.

The strongest verdict is:

| Surface | Grade | Honesty | Consumer | Falsifier |
|---|---|---|---|---|
| `xi(t)` / pose-stream rate | ALREADY-EMBODIED plus narrow ADOPT | DERIVED, with the specific acceleration-residual packet still CONJECTURE | PC1-style pose-stream packet; post-OD3 OD8 persistence if a dense stream is emitted | A byte-closed acceleration packet fails to beat the existing spline/control-curve or direct `dxi` packet at identical receiver output, or deterministic integration drifts outside the pose/receiver tube |
| OD8 cross-pair smoothness prior | ADOPT as a $0 test design; not measurable from current OD8 docs alone | DERIVED test design, OPEN-QUESTION result | Post-OD3 re-derive and `experiments/ddm_od8_js1_persist.py solve-persist` / `price-persisted` | Persisted values show acceleration-coded residual bytes are not below flat or first-order delta coding, with exact decode equality required |
| AM unpaired marginal inference | N-A | DERIVED | None | Becomes applicable only if Pact intentionally discards known per-pair trajectories and only keeps marginal snapshots, which is not the live vehicle |
| AM neural acceleration training as a shipped model | REFUTED-SEED for current archive use | DERIVED from contest byte rules | None | Becomes viable only if learned weights or video-derived tables are counted and still beat the existing packet rate |

No new canonical equation is registered. The relevant law shape is already covered by the PC1 pose-stream laws and the trajectory-derived stopping law; adding an AM-branded duplicate would be registry noise.

Scorer forwards: 0. `upstream/evaluate.py`: not run. n600: not run. Paid launches: none. Score claim: none.

## Paper Custody

Paper read: Gabriele Dazzini, Giovanni Conforti, Alain Durmus, and Aram-Alexandre Pooladian, "Trajectory inference via Acceleration Matching," arXiv:2608.03916, submitted 2026-08-04.

Sources read:

| Source | What was used |
|---|---|
| `https://arxiv.org/abs/2608.03916` | Abstract, submission date, authors, version |
| `https://arxiv.org/html/2608.03916` | Full paper text, definitions, algorithm, theorem statements, conclusion |

Direct shell PDF fetch was blocked by sandbox DNS resolution, so this receipt does not claim a local PDF hash. The official arXiv full-text HTML was sufficient for the deep read.

Core paper facts used:

- AM solves trajectory inference from unpaired positional snapshots by lifting to phase space `(X, V)`.
- The method regresses an explicit conditional acceleration field.
- The acceleration field is derived through kinetic Brownian bridge Markovianization.
- The training objective is simulation-free and uses positional data, with velocities integrated out or sampled from bridge conditionals.
- Generated trajectories are obtained by sampling an initial velocity and integrating the learned phase-space dynamics.

## Recall Evidence

| Query | Source | Plan impact |
|---|---|---|
| `pose stream xi se3 bspline scorer_targets pose_from_embedding` | `.omx/research/pose_in_training_levers_survey_20260702.md` | Reframed AM from a new pose solution into an already-built smooth `xi(t)` / SE(3) B-spline representation family. The live transfer is a packet-pricing A/B, not a new mechanism claim. |
| `ddm_pc1 pose stream canonical equations` | `.omx/research/ddm_pc1_pose_stream_admission_canonical_equations_20260724.md`; `.omx/research/codex_findings_ddm_pc1_pose_stream_admission_20260724_codex.md` | Found a counted 32-knot smooth twist curve packet already admitted as a typed component, with zero-home results negative but the descent-trainable family not killed. This makes AM's smooth control curve mostly ALREADY-EMBODIED. |
| `xi temporal delta coder 574` | `.omx/research/xi_temporal_delta_coder_574_20260721T222234Z.md` | Prevented overclaiming: planar-3 xi prediction was a measured formulation negative on an already canonicalized Lane chart, not a family-wide kill. AM acceleration coding must be scoped as a new formulation only. |
| `wrong levels describe sweep 610` | `.omx/research/wrong_levels_describe_sweep_610_DAG_FEED_20260721.md`; `reports/lane_maturity.md` | Prevented immediate routing: #610 surfaces were blocked on receiver/rate/custody, so AM cannot be promoted without a receiver-consumed packet and byte proof. |
| `OD8 native DOF delta entropy` | `.omx/research/ddm_od8_20260805/OD8_NATIVE_DOF_RECEIPT.md` | Seed 2 cannot be measured from the current OD8 receipt because OD2/OD7 did not persist solved paint/DCT values. The right producer is post-OD3 re-derive with OD8 persistence. |
| `trajectory stopping canonical equation` | `.omx/state/canonical_equations_registry.jsonl`; `.omx/research/ddm_tj1_20260805/tj1_summary.md` | AM's costate/curriculum trajectory angle is already covered by the trajectory-derived stopping law. No new registry row. |
| `workflow velocity rigor worldsheet xi` | `docs/operating_workflow_v2_velocity_rigor_autonomy_20260720.md` | Reinforced that Pact already treats time as one worldsheet with natural coordinates and SE(3) `xi` curves, not 600 unrelated slices. |

## Per-Seed Verdicts

### Seed 1 - `xi`-Curve / Pose-Stream Rate Lever

Verdict: ALREADY-EMBODIED for the broad smooth trajectory representation; ADOPT only for a narrow acceleration-residual packet A/B.

Honesty: DERIVED for the crosswalk, CONJECTURE for any byte gain until priced on a persisted stream.

The paper represents trajectories through phase-space dynamics and an acceleration field. Pact's corresponding object is not unknown object motion inferred from marginals. It is known ego-motion or solver state: `xi`, `dxi`, per-pair pose targets, warp controls, DCT paint coefficients, and event trajectories. Existing Pact work already has:

- `dxi` sidecar shape at 7.2 KB.
- PC1's 32-knot smooth counted twist curve and nested packet admission.
- SE(3) B-spline / control-pose tooling.
- PFS1's 194 B warp-base member as a small but not score-moving precedent.
- E2 as receiver/export apparatus, not a promoted counted pose member.
- OD2 cheapdct4 at 48 int16 coefficients per pair, 57,600 B projected n600, but without persisted solved values in the current OD8 receipt.

AM's concrete candidate is not "learn trajectories from snapshots." It is a deterministic receiver packet:

1. Store a small initial state and a quantized acceleration/control curve.
2. Integrate a generic phase-space update in `inflate.py`.
3. Emit the same per-pair `xi` or solver stream that a direct packet would have emitted.
4. Price exact counted bytes against existing spline, delta, and flat coders.

Falsifier: if the acceleration/control packet plus residuals is not smaller than the incumbent stream packet at byte-exact decoded output, or if deterministic integration produces receiver drift that changes the scored witness behavior, the AM transfer is rejected for this surface.

Named consumer: PC1 descendant pose-stream packet; post-OD3 OD8 persistence if OD3 emits a dense cross-pair stream needing temporal compression.

### Seed 2 - Smoothness Solve Prior For Cross-Pair Solution Trajectories

Verdict: ADOPT as a $0 test design; not executable against current OD8 artifacts because the needed values are not persisted.

Honesty: DERIVED test design, OPEN-QUESTION result.

OD8 names the relevant discriminator: delta-entropy-small versus decoded-base-large for native DOF streams. The receipt also says the actual support/values and cheapdct4 solved values were not stored. That makes a real AM-style entropy comparison impossible without re-producing or post-OD3 persisting the stream.

Test design once values exist:

1. Load a persisted sequence over pair index: block16 paint values, cheapdct4 coefficients, `dxi`, or any OD3/OD8 native stream.
2. Encode four scorer-free predictors with bit-exact reconstruction: flat per-pair values, first-order delta, second-order delta, and acceleration/control-curve residual.
3. Use the same coder family and same denominator for all arms; record raw bytes, compressed bytes, residual entropy, and decode equality.
4. Admit AM only if the acceleration arm wins on counted bytes without changing receiver output.

Producer if resumed: run the OD8 post-OD3 persistence path named in the OD8 receipt, then execute the pricing-only comparison. If OD3 terminal artifacts already include native payload fields, use them; otherwise re-run the same pair set with `solve-persist` before pricing.

Falsifier: acceleration-coded residual bytes are greater than or equal to first-order delta or flat coding under exact decode equality, or the residual model needs video-derived learned weights large enough to erase the byte gain.

Named consumer: OD8 post-OD3 native-DOF persistence and the fork discriminator that decides whether a cross-pair smoothness prior is worth carrying into the next packet.

### Seed 3 - Boundaries And Non-Applications

Verdict: N-A for unpaired marginal inference; REFUTED-SEED for shipping learned AM weights as a free artifact.

Honesty: DERIVED.

The paper's primary inference setting is unpaired snapshots. Pact's live setting has known source video, known pair index, known scorer calls only when authorized, and measured encode-time trajectories. We do not need to infer which marginal point became which later point. We need the shortest compliant counted packet that reconstructs the chosen witness.

Non-applications:

- AM does not alter exact score authority. Only archive bytes through the contest evaluator can move the pointer.
- AM does not make learned/video-derived acceleration weights free. Generic integration code is free; learned or video-derived payload is counted.
- AM does not justify scorer runs, n600 eval, or a launch under this charter.
- AM does not supersede existing receiver-closed pose or OD8 packet gates.

Falsifier for N-A status: a future Pact branch intentionally discards pairwise trajectories and only keeps unpaired per-time marginals while still needing a legal receiver packet. That is not the current vehicle.

Named consumer: none for unpaired inference. The only retained consumer is the compact-packet formulation above.

## Beyond-Seed Sweep

| AM idea | Grade | Honesty | Pact transfer | Falsifier | Consumer |
|---|---|---|---|---|---|
| Phase-space lift `(state, velocity)` | ALREADY-EMBODIED | DERIVED | Matches existing `xi`, `dxi`, control-pose, and temporal worldsheet treatment | No byte gain versus current state/control packets | PC1, OD8 post-OD3 |
| Conditional acceleration fields | ADOPT narrowly | CONJECTURE | Use tiny deterministic/quantized acceleration controls for residual streams, not a learned uncounted neural model | Counted controls plus residuals lose to spline/delta coding | OD8 persistence A/B |
| Simulation-free training | N-A / ALREADY-BETTER | DERIVED | Pact can fit directly to persisted values; no need to simulate trajectories for this charter | A direct least-squares or entropy coder is not available for a future stream | Post-OD3 scorer-free pricing |
| Worldsheet event tracks | ADOPT as design lens | CONJECTURE | Encode topology/event changes as sparse acceleration or jerk events along pair index | Event grammar does not reconstruct exact stream values or costs more than direct residuals | Future native DOF packet if eventful |
| Costate/curriculum trajectories | ALREADY-EMBODIED | DERIVED | Existing trajectory-derived stopping law already handles recorded objective trajectories and continuation cutoff | AM gives a strictly better, measured stopping criterion from existing logs | TJ1 descendants only after proof |
| Temporal amortization / keyframe+warp | ALREADY-EMBODIED plus OPEN-QUESTION | DERIVED | The natural comparison is keyframe/control plus warp residual versus acceleration residual | Acceleration arm loses byte-closed to keyframe+warp | #148-family temporal packet, OD8 |

## Next If Resumed

1. Wait for OD3 or a post-OD3 OD8 re-derive that persists native payload values.
2. Price a scorer-free residual-coder A/B on the persisted stream: flat, delta, second-order, acceleration/control-curve.
3. Require byte-exact decode equality before any score-facing claim.
4. If acceleration wins, register the packet as a counted component with receiver parse-back and only then consider a score-bearing composition.
5. Do not reopen AM unpaired marginal inference unless a future Pact branch actually has only unpaired marginals.

## Completion Boundary

Measured in this AM1 unit: no scorer measurements. The measured facts cited are prior Pact artifacts and the paper's stated algorithmic claims. No new score, no new archive bytes, no n600 evaluation, no exact evaluator run, and no frontier movement occurred.

Not measured: acceleration-coded residual byte gain, pose effect, segmentation effect, OD8 value entropy, or exact archive score.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
