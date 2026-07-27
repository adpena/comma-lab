# G90 — Current-base projected population costates

Status: production implementation landed; governed stage-0 materialization is
resumable and in progress. This is encoder-only research signal, not a
candidate, score, rate claim, or pointer move.

## Purpose

G90 prices real task-space actuator directions at the exact G85 operating
point without persisting dense pixel gradients. Its only consumer is the
whole-state allocator (G83/G92 lineage). It cannot locally admit a factor.

The base is the exact G85 full-n600 row:

- archive: 129,392 bytes,
  `b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd`;
- decoded raw: 3,662,409,600 bytes,
  `436ce2b6965c859556a217df9b1cc17784d988f2af900c35201d3e3c7f372782`;
- `d_pose=163.06130981`, `d_seg=0.02747120`, `N=600`; and
- pose per-pair raw-MSE VJP scale
  `5/(600*sqrt(10*163.06130981)) =
  0.00020636844449905425`.

## Coordinates

Pose and Seg are never mixed.

1. Pose: differentiate the upstream raw six-value per-pair PoseNet MSE, then
   multiply by the exact population score chain-rule scale above. The
   established differentiable BT.601/YUV6 twin removes only upstream's
   `@torch.no_grad` barrier and matches its numeric preprocessing.
2. Seg screening: sum `target_logit-current_logit` only on current argmax
   mismatches and differentiate it. This is proposal screening, not an exact
   Seg objective.
3. Exact replay: after two-axis Pareto screening, rerun frozen SegNet/PoseNet on
   the realized uint8 camera result. Persist exact mismatch-count/score and
   population pose-mean/score deltas.
4. Rate: remain `null` until a whole composed ZIP is actually built. G74 member
   bytes are explicitly not used as ZIP rate.

Dense costates exist only inside one batch of at most 16 pairs. They are
immediately paired with finite actuator displacements and destroyed. Durable
state contains compact actuator-coordinate rows only.

## Current-base composition law

G78/G87 were compiled at semantic P, but G85 is `P + incumbent A`. The
incumbent G74 operand addresses pair 0 and selects `BOTH`; proposed G72
directions select `Y1`. G90 therefore:

1. strictly parses the exact G85 outer archive and PVSA member;
2. derives current Seg cells from exact G85 raw;
3. regenerates only incumbent-addressed G72 proposal geometry and proves all
   unaffected proposals equal fresh G87;
4. decodes the incumbent `BOTH` operand as the current base;
5. jointly realizes incumbent-on-Y1 plus proposed atoms through G74;
6. copies current-base Y0 unchanged and uses the jointly realized Y1; and
7. refuses any incumbent/proposed donor-address collision because a replacement
   coordinate law is not yet defined.

This is an exact encoder-side tangent realization. PVSA V1 cannot serialize two
different frame selectors in one G74 actuator, so G90 does not claim a
receiver-closed mixed-selector operand or rate. G83/G92 must compose the
selected state into a wire that actually supports it.

## Durable ABI

Schemas:

- `tac.taskspace_projected_population_costate_preflight.v1`
- `tac.taskspace_projected_population_costate_batch.v1`
- `tac.taskspace_projected_population_costate_stage.v1`
- `tac.taskspace_projected_population_costate_aggregate.v1`
- `tac.taskspace_projected_population_costate_blocker.v1`

Each immutable batch checkpoint carries:

- exact pair range and G85/source/target/current-cell identities;
- base per-pair float32 pose MSE values and Seg mismatch count;
- exact population pose scale;
- separate projected Pose and Seg coordinates;
- exact replay deltas only for the unweighted two-axis Pareto set;
- exact proposed atom dictionaries, proposal fingerprints, and proposed-operand
  SHA;
- a separate incumbent-operand SHA;
- `exact_zip_delta_bytes=null`, no local admission, no dense costate.

Stages are immutable 120-pair checkpoints. Only five complete stages may emit
the aggregate, and full n600 base components must reproduce the exact G85 row
to its reported precision.

## Resumability and custody

The typed config is
`.omx/research/configs/taskspace_projected_population_costates_n600_20260727.json`.
Output is on
`/Volumes/VertigoDataTier/pact/taskspace_projected_population_costates_n600_20260727_r1`.
Every scorer batch and every 120-pair stage is written atomically and never
overwritten. The 50 GiB SSD reserve is rechecked on every resume.

The immutable preflight first closed strict G78/G87/V15 custody. A later
concurrent edit to a mutable receiver dependency invalidated G78's runtime
source closure, so resumed data access uses a prior-preflight-gated data-only
loader that rehashes the exact aggregate, stage receipts, and mmap fields. It
does not reinterpret them as current-base truth.

## Known blockers

- Full five-stage n600 materialization is not complete yet.
- G88 has no fresh counted n600 XIP2 trajectory/quantization-scale custody.
- G89/G92 class-complete operands require their own exact tangent provider.
- Mixed-selector current state is not serializable by PVSA V1.
- No actual ZIP delta exists until the whole-state allocator composes a
  receiver-closed archive.

Pointer truth: effective frontier remains `0.172`; G90 has not moved it.

## First real current-base batch result

The governed materializer sealed batch `[0,16)` against the exact G85
`P + incumbent A` raw base. This is bounded encoder evidence, not an n600
candidate or a local admission:

- exact population Pose scale: `0.00020636844449905425`;
- eight compact grouped actuator coordinates, with no dense derivatives stored;
- three unweighted Pose/Seg screening-Pareto coordinates replayed through exact
  realized `uint8` R and the frozen scorers;
- `actual_zip_delta_measured=false`, because no receiver-closed composed ZIP
  exists yet; and
- current G85 cells differ from the historical G78 P-only description at 134
  cells in this batch, confirming that current-base custody is load-bearing.

The exact replay exposes a structural nonlinearity that the whole-state
controller must retain. `UndrivableBoundary:d0:a1` had a negative first-order
Seg gap coordinate (`-3070.2491`) yet improved exact Seg mismatch by `2835`
cells and exact Seg score by `0.0024032593`; it also improved the exact global
Pose score contribution by `0.0000079101`. Conversely,
`Road:d1:a0.5` had a positive first-order Seg gap coordinate
(`+16445.9794`) but worsened exact Seg mismatch by `1943` cells. Therefore the
compact costates are useful proposal coordinates, not an admission oracle.
Finite step, collision, saturation, argmax topology, and R jointly change the
sign. G92/G83 must select using the exact realized rows and re-evaluate the
composed population state; no scalarization of these local derivatives is
authorized.

## Adversarial classification: V1 is a coarse family atlas

V1 exposes only eight `role × direction × amplitude` coordinates per 16-pair
batch. A Road coordinate can contain 2,533 atoms (32,955 uncompressed operand
member bytes). This grouping preserves a useful family direction, but it
suppresses the within-group structure needed for factor/program induction.
Therefore:

- V1 is a **coarse family costate atlas**, not an atom selector;
- neither one coordinate's linearized costate nor its exact joint replay may be
  assigned to its constituent atoms;
- `operand_member_bytes` is neither additive atom rate nor archive rate;
- G92/G83 may use V1 to choose a family for refinement, but not to order atoms
  or claim a factorized program from these coordinates alone; and
- the current stage remains useful because it maps the population family field
  and its nonlinear exact replay, but it cannot close the allocator by itself.

The next typed seam is
`tac.taskspace_projected_population_costate_refinement_request.v1`. It is a
request/response hierarchy, not inferred evidence:

```text
coarse group
  -> temporal span
  -> geometric chart (role, orientation, scale, shear, spatial tile)
  -> collision-free atom block
```

Every requested leaf must carry the exact parent group id, pair span,
constituent proposal fingerprints, proposed-atoms SHA, incumbent-atoms SHA, and
a deterministic partition hash. A response may carry additive VJP/JVP
coordinates only after an explicit additivity/interaction check; otherwise it
must report a joint block coordinate and exact realized replay. Per-atom exact
gain and per-atom rate remain `null` unless independently measured. A
hierarchical response must reconstruct the exact parent's fingerprint
multiset—no missing or duplicated atoms—and must preserve collision closure at
every level.

This refinement ABI is a blocker/next contract, not part of the active V1 run:
mutating its schema while immutable V1 checkpoints are being produced would
break resumability and source custody.

### Pareto screening is not an admissible false-negative gate

Batch `[32,48)` sharpened the nonlinearity finding. The first-order two-axis
Pareto set retained only two of eight coarse groups. Both retained Road groups
improved exact Pose but catastrophically worsened exact Seg
(`+0.0238554213` and `+0.0113533868` score units for their 16-pair finite
interventions). Because earlier batches already proved that finite exact
response can reverse either directional sign, the six linearly dominated groups
cannot honestly be called exact-dominated. V1 therefore has incomplete exact
coverage and its Pareto set is a replay-budget heuristic only.

V2 must exact-replay every deterministic physical group produced by the
semantic family axes plus collision partitioning (at least eight, and observed
as twelve for `[288,304)`), or establish a measured trust region with an
amplitude continuation whose sign/rank fidelity is verified before Pareto
pruning. Only after that coarse exact gate may the
hierarchical refinement ABI allocate additional replay. Stage 0 may finish as a
useful population atlas, but stages 1–4 must not be launched under the V1
Pareto-as-filter interpretation.

## Triangular composition with G91

Fresh G91 custody supplies a useful exact separation. Its selected n600
trajectory treatment changes Y0 and proves Y1 byte-identical for every pair.
G90's G72 tangent changes Y1 while preserving current Y0. Therefore the
semantic measurement has a narrow exact transfer law:

- SegNet reads Y1 only;
- G91 preserves Y1 exactly; and
- a G90 exact Seg mismatch delta remains valid only when the exact measured Y1
  mutation is reproduced byte-for-byte.

Pose has no corresponding transfer. PoseNet reads both frames, so G90's Pose
costate and exact Pose delta at the G85 pair are invalid after any Y0 change or
any different Y1 state. G91 itself was optimized against the pre-G94 Y1, so its
current Y0 is initializer/factorability evidence rather than the terminal pose
solution. Treating either marginal as state-independent or additive would be
another local linearization fake.

The correct joint controller is triangular:

1. choose and exactly realize the final semantic Y1 state
   (`incumbent + G89/G90-derived program`);
2. reuse a G90 exact Seg delta only if that exact Y1 mutation is reproduced;
3. use G91 as an initializer/factorization prior and solve Y0 conditionally as
   `Y0 | final Y1`; and
4. use G94 to serialize and price the sequential disjoint-frame actuator
   product.

V2 responses must consequently bind both
`pose_conditioning_y0_sha256` and `pose_conditioning_y1_sha256`, plus an exact
`seg_mutated_y1_sha256`. A response whose pair hash differs must invalidate its
Pose fields. A response whose Y1 mutation hash differs must invalidate its Seg
fields. This is the minimal state/costate custody rule needed for the contingent
two-objective differential system.

## V1 terminal state

V1 sealed six immutable batches, `[0,96)`, then failed twice before `[96,112)`.
The hardened terminal blocker binds:

- one differing cell at pair 100, scorer coordinate `(200,429)`;
- inference-authority class 0 versus autograd-forward class 2;
- logits `1.7809603214` versus `1.7809610367`, a margin of only
  `7.1525574e-7`;
- actual differentiable cell SHA
  `4f19ba7e09f19df484ef06c1e82f5fb78e8cb9f02ea81fe2671fb6b732053a34`;
  and
- expected inference-authority cell SHA
  `0c4ac563e6706d1efbfd942263c825a4614bb5e8269339371a928341b229bb82`.

An isolated inference-mode recheck reproduced the expected cells exactly. The
failure is therefore not raw-byte or G78 custody drift: it is a real numerical
boundary tie between the scorer-authority inference surface and the
autograd-enabled surrogate surface. V1 remains stage-0 incomplete and cannot
emit a stage or aggregate receipt. Its six batches are preserved as bounded
screening evidence only.

V2 must separate authority inference cells from differentiable cells, annotate
every tie drift, use authority cells for exact realized replay, exact-replay
every deterministic collision-partitioned physical group, bind each batch's
ordered group IDs/count, aggregate actual checkpoint counts, and preserve new
immutable batch/stage schemas under a new SSD output root. No V1 Pareto discard
or linear direction is admissible.
