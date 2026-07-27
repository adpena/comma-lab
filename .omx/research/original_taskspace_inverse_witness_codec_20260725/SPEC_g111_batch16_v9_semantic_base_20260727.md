# G111 — physical batch-16 V9 semantic-base producer

Date: 2026-07-27  
Axis: `[macOS-MLX producer / encoder-only target custody]`  
Lane: `lane_g111_fresh_batch16_v9_training_binding_20260727`  
Authority: BUILD + governed local verification; no score or pointer claim

Status: typed producer implemented but **HELD** before heavy launch. The legacy
trainer selects checkpoints after an arbitrary-scale int8 realization, while
G105 ships a different power-of-two int8/int16 public wire. G111 may fire only
after the external G120/G121 compiler path proves that every immutable stage is
serialized, parsed, scored through the shipped public wire, and retained
without consulting the legacy BEST. Post-G105 pose refit cannot repair
semantic quantization spill.

Every cold, periodic, stage, and final checkpoint is also published as an
immutable deploy/resume/receipt node under `fresh_lineage/`. Each nonroot node
recursively reopens its exact parent receipt and SHA back to the unique
zero-parent cold root. A fresh-producer resume requires the external parent
receipt path and file SHA; the resume NPZ is never allowed to self-attest its
ancestry. `fresh_lineage_tip.json` is a rolling locator, not an authority
substitute.

## Purpose

G111 is the first launchable full-n600 semantic-base producer in the new
selected-preimage stack. It composes the settled V9 mod-32 task-space geometry
with the physical G109 capsule containing Seg labels, winner-minus-runner-up
margins, and source Pose6 from the same chronological upstream batch-16
forward. G109 remains encoder-only evidence. Dense targets and scorer weights
are forbidden from the candidate payload.

The target is the effective frontier pointer, dynamically
`min(local_best, upstream_best)`, currently displayed as `0.172`. No historical
`0.18`, `0.188`, `0.191`, or `0.19110` literal authorizes a launch verdict.

## Settled composition

1. G109 materialized all 600 source pairs in 38 chronological batches (37x16
   plus 1x8), using the frozen upstream scorer callback for all three target
   tensors. The physical aggregate receipt file is
   `/Volumes/VertigoDataTier/pact/taskspace_v9_training_target_capsule_n600_20260727/21_v9_training_target_capsule_receipt.json`,
   external SHA-256
   `1fd651c9a668360c161f6aa5acc721086843cd12c2c362141d7dc21a1e9bdb01`,
   sealed aggregate SHA-256
   `c5dac3863d06d812e81e445ed2c023c802e628d59c88f3b557855dfe479cb305`.
2. The G111 typed DSL compiler recursively reopens that receipt, all G109 batch
   checkpoints, G46 source/label custody, frozen scorer files, raw arrays, and
   deterministic NPZ. It admits only real n600, batch 16, non-test custody.
3. The trainer independently hashes its active `uint8[B,2,H,W,3]` cache against
   each G109 chronological batch before substituting labels, margins, or Pose6.
   This prevents a valid target capsule from being used on a different source
   coordinate.
4. The target projection, target-authority hash, G46 source chain, target-array
   hashes, G109 aggregate hashes, and live verdict batch are carried into both
   deploy and resume checkpoints. Resume reopens the same physical capsule and
   revalidates the complete projection before model restoration.
5. G105 now admits the G109 aggregate schema and requires a fresh producer.
   Cold lineage is state-bound, not a copyable Boolean: the root hashes the
   governed DSL identity, seed, physical G109 projection, and deterministic
   initial tensors. The seed is persisted, so consumers independently recompute
   the cold root rather than trusting its claimed digest. Every full-state
   checkpoint hashes live, EMA, optimizer, Polyak, RNG, recent-loss, controller,
   and configuration state into a parent-linked checkpoint ID. A governed
   resume may have a new launch DSL hash, but it must reconstruct the original
   cold root and validate the complete resumed state before restoration. G112
   must bind the deploy child to its preserved full-state companion before any
   product calls the lineage own-custodied. G111 emits a unique zero-parent
   full-state cold checkpoint after deterministic initialization and before the
   first optimizer step, so ancestry has a physical terminal rather than a
   logical root label.
6. FreSh spectral initialization is not falsely claimed. The current FreSh
   path requires self-orientation, while the exact public G105 decoder has no
   reviewed decoder-owned fixed-point self-orientation ABI. G111 therefore uses
   the exact polar, self-orient-off public gauge and distinguishes cold lineage
   from the optional FreSh algorithm.
7. G111 also trains a generated pose carrier. Its checkpoint therefore contains
   shared trunk tensors, interleaved even/odd codes, `pose_carrier.xi_stored`,
   and `pose_carrier.dxi`. The active `generated_y1` forward derives frame0 from
   the final odd-code Y1 at the scorer-grid uint8 boundary, then applies the
   exact sparse V10 factor-2 selected-preimage map
   `tac.v10_factor2_selected_preimage.v1`; it never consumes an even code.
   The MLX path uses the identical gather/valid-mask forward with a scorer-round
   STE, so the shared-trunk pose gradient is written through the camera operand
   that public inflate actually emits. G105 owns the shared trunk plus odd Y1
   rows. The conditional compiler folds `xi_stored + residual_scale*dxi` into
   an initializer and explicitly proves the even rows dead for this candidate
   forward rather than silently dropping them. This closes the pose-gradient
   camera operand only. The semantic training loss still acts on the pre-G105
   floating state, so it is not falsely labeled identical to the parsed public
   wire. Exact post-hoc stage selection must close that deployment boundary; a
   separate terminal wire-QAT stage is admitted only if measured wire regret
   makes it worthwhile.
8. G111 forces `render_aa=none`, because the exact G105 public receiver currently
   implements the unattenuated polar Fourier basis and not the trainer's IPE
   feature transform. Train-time seed and lane-band assists may remain, but the
   selected deploy/verdict surface is the bare public-compatible generator.
9. G105 exposes two canonical-within-family counted Y1 transforms: raw
   chronological int16 and temporal delta-Rice with deterministic best-`k`.
   Neither inner length is treated as contest rate. G110 must build the complete
   deterministic archive for the Cartesian product
   `{raw, delta-Rice} x {ZIP STORE, ZIP DEFLATE}` and select by exact ZIP bytes;
   the parser preserves and strictly re-emits the selected wire and outer-method
   tags.
10. G105 power-of-two packet quantization still changes the semantic selected
    preimage relative to the trainer's scored deploy quantizer. Therefore the
    trained effective twist is encoder evidence and an initializer only. The
    authoritative conditional operand is re-solved after parsing the exact G105
    packet, against that public uint8 Y1 and the physical G109 Pose6 targets.

## Triality

DSL:

`V9 ideal mod32 + G111PhysicalBatch16TargetCustody`

DAG:

`G46 source/labels -> G109 same-forward batch16 targets -> G111 cold V9 train
-> EMA/stage checkpoints -> exact G105 semantic packet -> G110 Y1 decoder
-> conditional Y0|Y1 -> archive -> public inflate.sh -> upstream/evaluate.py`

Equations:

`theta_1* = argmin_theta L_V9(theta; Y1_G109, M_G109, Pose6_G109)`

subject to:

`source_cache_batch_sha256(b) = G109.source_pair_batch_sha256(b)` for every
chronological batch `b`, and

`checkpoint.target_projection = current_physical_G109_projection`.

The final system objective is not an arbitrary independent Seg/Pose/rate gate:

`min_Y1 [100 D_seg(Y1) + min_Y0|Y1(sqrt(10 D_pose(Y0,Y1)) + lambda R(Y0|Y1))
         + lambda R(Y1)]`.

G111 closes the physical Y1 producer coordinate. It does not yet implement the
outer conditional value function; that costate becomes the later joint-descent
feedback from G110/G112 into Y1.

## Pre-launch coupled feasibility envelope

The typed G111 compile now derives the exact value-independent G105 raw semantic
packet load before training:

- polar input dimension: `80`;
- counted model values: `71,159`;
- quantized model data: `72,430` bytes;
- raw chronological Y1 code data: `38,400` bytes;
- complete G105 raw semantic packet, including its own headers and metadata:
  `111,840` bytes;
- packet-only rate contribution (not a complete archive):
  `0.0744696653172`.

This is an exact structural byte reference for the forced-raw packet shape,
not an entropy prediction, archive measurement, score, or candidate.
G110 still owes the exact four-way complete-archive race and may choose a
smaller Rice/DEFLATE combination.

For scale only, the separately custodied G54 batch-16 low-distortion existence
coordinate (`d_seg=0.00015196058485243054`,
`d_pose=0.00010184347386600314`) contributes `0.0471089828053` distortion
score. It is not transferred as a G111 result. If G111 independently reaches
that coordinate, the raw semantic packet leaves about `75,724` additional bytes
under `0.172`, or `42,684` under `0.15`, for public runtime, conditional pose,
and ZIP/container overhead. Thus the new product is not merely structurally
interesting: its coupled envelope is large enough to be frontier-capable, while
the authoritative decision remains the final G110 archive/eval row.

## Superseded governed dry-run evidence

The earlier full launcher dry-run at
`/Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dryrun_20260727`
passed its then-current contract:

- 229/229 emitted flags exist in the real trainer parser;
- 20/20 expected typed levers, including the custody rollup;
- DSL compile hash
  `b958382acb39312dd76cd00d4840694627b0374da4fe1f2849f87676f42f1694`;
- LawRef self-recompile and 228-constant manifest;
- all event/derived schedule gates;
- projected peak 24.48 GiB;
- governed system admission with 43.3 GiB reported headroom;
- no process spawned in dry-run mode.

That receipt is now explicitly superseded: it predates `generated_y1`, the
scorer-grid uint8 conditional boundary, `render_aa=none`, and complete
pose-carrier checkpoint custody. A new clean-directory dry run was owed before
any launch; the current evidence below closes that compile-only debt. The old
DSL hash is not launch authority.

G109 materialization itself completed under `safe_run` in 262.889 seconds with
peak RSS 6053 MiB. The durable target directory is 1.5 GiB on
`VertigoDataTier`; it is not candidate payload.

## Superseded governed dry-run v2 evidence

The prior clean-directory compile exists at
`/Volumes/VertigoDataTier/pact/g111_batch16_v9_semantic_base_dryrun_v2_20260727`.
It spawned no trainer process and proved its then-current full-n600 launch surface:

- 229/229 emitted flags exist in the real trainer parser;
- 20/20 expected typed levers are active;
- DSL compile hash
  `38f81d03045ca021fed8b73ba5f8626bc80262566bdd4bbd7bc940901eeb10ac`;
- 228-entry constants manifest;
- projected peak 24.48 GiB, admitted with 42.3 GiB headroom at preflight;
- `generated_y1` is present at launch line 33, `render_aa none` at line 49,
  n600 at line 149, and the physical G109 path/SHA plus `fresh_producer` at
  lines 240-242.

Durable dry-run identities:

- `launch.sh`: 242 lines, SHA-256
  `31e31c065af957169e9d8bc1298cae6c8ed2a1300f5a9235c3564436edc40f61`;
- `dsl_provenance.json`: SHA-256
  `223f2dcdbd15b2538efa556b35e6b276b827eae0c69138f21c2890edcbfc34dc`;
- `launch_manifest.json`: SHA-256
  `8ccf1353a3b7462ddcba0789b86fe4d35308275fa5783ef9e783e580371b52f5`;
- `constants_manifest.json`: SHA-256
  `5918b2ac85001e6fb5b92067df9a3f06e279c99f7b0fddd61a7b9462e79c3911`.

This v2 dry-run is now superseded because its DSL contract predates the exact
train/public V10 selected-preimage identity and complete-archive raw/Rice
arbitration. It is launch-configuration evidence only, not current launch
authority, a checkpoint, archive, score, candidate, or pointer-mutation receipt.

## Structural bugs extinguished

- The trainer formerly targeted a stale local/historical score literal instead
  of the effective frontier pointer.
- V9 targets previously mixed batch-32 cache fibers with batch-16 upstream
  authority.
- G105 demanded the old margin-only schema rather than the joint G109 target.
- “fresh” was conflated with FreSh, creating an impossible self-orient/public
  decoder contract.
- The first cold-lineage marker incorrectly refused crash resume, violating the
  P0 resumability contract; the next marker-only form was copyable onto foreign
  state. It is replaced by a cold-root plus parent/state checkpoint chain.
- G111 formerly trained the shared semantic trunk's pose gradient through a
  bicubic camera upsample while public G110 emitted the sparse V10 factor-2
  preimage. A structured operand differed at 69.1% of camera samples with MAE
  30.09. The trainer and NumPy verdict now use the exact public map.
- G105 formerly chose and enforced raw versus Rice by inner payload bytes. A
  deterministic regression has a smaller Rice packet (4240 vs 4915 bytes) but
  a larger single-member ZIP (1109 vs 1065 bytes); both wire forms are now
  legal and exact whole-archive arbitration owns the choice.
- Re-running V9’s inherited constant reconciliation overwrote historical
  `beta_end=10` custody with the emitted 3.177 and broke LawRef self-recompile;
  G111 preserves the parent reconciliation and retires only rows its child
  actually changes.

## Unified-stack wire-in

- Sensitivity map: G109 margins establish scorer-cell proximity; per-cell
  response remains supplied by the existing V9 through-R telemetry.
- Pareto constraint: G111 is admitted only as a semantic-base producer; no
  archive promotion occurs without exact `D_seg`, conditional `D_pose`, and
  bytes from the same archive.
- Bit allocator: the G105 packet owns shared trunk plus temporally coded odd Y1
  codes. The generated pose operand owns only the post-G105-refit effective
  twist. Even code rows are encoder-only dead state under `generated_y1` and
  have zero candidate ownership.
- Cathedral/autopilot: named config
  `g111_batch16_v9_semantic_base` is routed through the governed launcher, not
  raw Python.
- Continual learning: the full producer and each stage checkpoint must emit
  exact post-R telemetry and become an empirical row; this BUILD receipt is not
  such a row.
- Probe disambiguator: mod-32 is selected for first feasible distortion. Rate
  factorization and mod-19 are evaluated from exact G105 packet/archive bytes
  after feasibility, rather than assumed from parameter count.

## Next gate

G110 complete-archive arbitration/public conditional closure is implemented.
Finish the strict one-stage G120 production wrapper, the external resumable
G121 exhaustive stage harvester, and the G119 post-G105 conditional pose
runner. Then regenerate a clean governed dry-run under the exact
selected-preimage, state-bound lineage, and external-stage-compiler contracts.
Run the governed two-pass dry-start to prove cold boot, checkpoint creation,
physical G109 binding, same-lineage crash resume, total tensor accounting, G105
temporal parse-back, conditional parse-back, and public reconstruction.

If green, launch the resumable per-stage n600 producer on SSD. G121 must compile
every immutable stage and retain every row satisfying only the strict
distortion obstruction `100*d_seg < live_target`. G119 then refits conditional
pose for each retained row. Whole-archive `d_seg`, `d_pose`, and exact bytes,
not a semantic-only BEST or a fixed component threshold, select the archive for
clean-extract double decode and `upstream/evaluate.py`.
