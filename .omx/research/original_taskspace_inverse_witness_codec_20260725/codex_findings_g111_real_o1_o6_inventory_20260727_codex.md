# Codex findings: G111 real O1-O6 inventory and construction dominance

Date: 2026-07-27  
Actor: Codex/root with independent Sol xhigh read-only audit  
Source HEAD: `6a3125480e0c7afd64febb0a053f8d8c4feef8d9`  
Verdict scope: fresh-G111 trainer state topology and restore order only  
Candidate claim: false  
Score claim: false  
Pointer moved: false  
Research only: false

## Decisive finding

The generic native-v3 transaction and verdict barrier are present, but the real
trainer-owned O1-O6 inventory and its construction order are not. The principal
blocker is dominance, not serialization syntax: legacy resume mutates live model
state before several active controller, rollback, verdict-history, and BEST
topologies have been constructed. A total restore cannot be staged against
objects which do not yet exist.

Current real-G111 typed compile SHA-256:
`511c61b81647c47311a48576bbc1e029cda589a31f1ad69e8cc98a22f4175df1`.
It is n600/batch-16 with async verdict, protected sparse seed and an independent
AdamW. Self-orient is off. Polyak, ladder, birth, closed-loop, Tail, Jacobian and
sigma-min conditioning are active. The current Tail Pareto floor is `0.0`; the
historical v6 value `0.0001` is not current topology authority.

## Exact atomic-owner inventory

### O1 current train

Required groups:

- `liveP__*`, `emaP__*`, `optP__*`;
- `seedP__*`, `seedOptP__*`, sparse-support manifest/count/geometry SHA;
- `polyakM__*`;
- epoch, stage, completed optimizer-step coordinate, optimizer family;
- Polyak arm/start/count.

Existing sources include `_build_resume_state_arrays()`,
`_snapshot_numpy_state()`, the live model, EMA, primary optimizer, protected
seed module/optimizer, and Polyak registration. Missing: a completed-step
coordinate, independent fresh optimizer topology in native-v3 admission, and a
fatal rather than fail-open Polyak heavy-state symmetry check.

### O2 rollback savepoint

Required groups:

- supervisor counters, window/max, LR scale, recent losses, rearm/event window,
  current epoch counters and exhausted latch;
- optional all-or-none `rbLiveP__*`, `rbEmaP__*`, `rbOptP__*`,
  `rbSeedP__*`, `rbSeedOptP__*`;
- savepoint content hash and coordinate.

The in-memory `_sg_state` and `_sg_take_snapshot()` currently have no complete
serializer/restorer. This is an entire missing owner, not a missing field.

### O3 schedule and control

Required groups:

- RNG;
- Muon/lane/chroma/temporal/phase gates;
- event curriculum, tau, birth and closed-loop controller;
- boundary, previous segmentation form, Muon switch, LR/group-LR and rewarmup;
- liveness/EMA updates, ladder, Tail, Jacobian/sigma-min pose state;
- active pose-law/coupling latches.

Several active controllers lack typed state adapters: boundary/LR, ladder, Tail,
Jacobian, liveness and pose-law holders. Late registry construction makes early
restore asymmetric. Native-v3 must reject every `__cl_pend_` key rather than
migrating the legacy pending-verdict sidecar.

### O4 verdict transaction and history

Required groups:

- the eleven native-v3 barrier arrays;
- bounded canonical verdict columns and reducer position;
- skipped-verdict count;
- annulus, lane and pose sensor histories consumed by controllers.

The generic barrier capture exists, but the trainer does not serialize the
complete history. The legacy worker mutates history, telemetry, BEST and output
directly. The history must be bounded and must not duplicate retained full
scorer snapshots.

### O5 causal selection and BEST

Required groups:

- BEST present/metric/epoch/tie sequence/result ID/content hash;
- deploy filename/SHA/bytes;
- fork-EMA clearance and latches;
- stage-checkpoint inventory with physical hashes;
- causal ordinal, current boundary and tip.

The whole owner is absent from legacy restore. Every referenced BEST/stage file
must be independently reopened and hashed. A worse first verdict after resume
must not replace the true pre-resume BEST.

### O6 lineage envelope

Required groups:

- semantic schema, typed DSL/config hash, seed, target/G109 bindings and static
  config;
- cold-root ID, git/upstream source, exact coordinate and parent
  checkpoint/receipt;
- zero-pending declaration;
- derived complete-state hash and checkpoint ID.

The complete-state hash must be recomputed across O1-O5 plus non-derived O6.
Current publication can occur before full barrier-held reopen and validation.

## Independent topology sources

Expected topology must be constructed without trusting checkpoint declarations:

1. Instantiate the current model from the compiled G111 DSL; its parameters
   define primary live/EMA keys, dtypes and logical shapes.
2. Freshly initialize the optimizer family selected by validated O3 state and
   checkpoint coordinate; flatten its virgin state.
3. Freshly instantiate the protected seed module, its independent optimizer and
   deterministic sparse-support geometry.
4. Require O2 savepoint topology to equal O1 exactly, including sparse seed
   selection and geometry.
5. Derive Polyak scalar topology from the fresh averager. Count zero permits no
   heavy leaves; positive count requires the complete primary topology in
   fp64.
6. Derive controller activity from the fresh typed G111 compile. Every active
   controller supplies a typed schema; inactive state owns zero keys.
7. Fix O4 barrier topology in code and fix bounded history columns in the
   reducer schema.
8. Fix O5 pointer topology in code, then reopen every referenced physical
   object by exact bytes and SHA.
9. Derive non-derived O6 custody from the DSL/capsule/source surfaces and
   recompute all complete-state identities.

## Construction-order defect

Current source order is structurally impossible for native-v3 restore:

1. `_do_checkpoint()` is defined before legacy resume.
2. Legacy resume begins and mutates model state.
3. `recent_losses`, stage inventory, event/tau controllers, late registry
   entries, liveness, boundary state, spike guard and Tail are constructed only
   later.
4. Cold-root capture happens after those constructions, while restore happens
   before them.

Required order:

`construct topology -> immutable load -> total O1-O6 validation -> pure stage
O1-O5 -> physical O5 reopen -> canonical reserialization equality -> stage O6
-> one live publication -> executor creation`

No worker, optimizer update, BEST publication or checkpoint may precede that
sequence.

## Minimum cross-invariants

1. Live/EMA topology equals the fresh model; optimizer topology equals the
   freshly selected family.
2. Seed activity is DSL-derived and its sparse geometry matches fresh support.
3. Rollback savepoint is all-or-none, equals O1 topology, verifies its content
   hash and does not point beyond current O1.
4. Polyak count and heavy leaves obey the exact zero/positive rule.
5. Muon gate, optimizer family, Muon-switch latch and coordinate agree.
6. Boundary, rewarmup and effective LR/group-LR state agree.
7. Ladder parallel-arm topology is exact and derived island state rebuilds from
   stored arms plus fresh geometry.
8. Tail config equals the fresh DSL and every cycle/history reference resolves
   into canonical O4 rows.
9. O4 is quiescent, contiguous, bounded, uniquely identified and contains no
   `__cl_pend_` key in any namespace.
10. Sigma-min/Jacobian histories are aligned, monotone and agree with pose
    engagement.
11. BEST names an applied O4 result; every BEST/stage object reopens with exact
    bytes and SHA.
12. O6 coordinate agrees with O1/O3/O4/O5 and its complete-state hash is
    recomputed.

## Implementation sequence

1. Land the pure scorer worker, deterministic reducer, bounded effect intent and
   replay-idempotent main-thread publisher while launch admission remains an
   unconditional refusal.
2. Make every O1-O5 topology source exist before resume/cold-root capture.
3. Add pure capture, independent expected-schema, stage and one-publish adapters
   for O2, missing O3, bounded O4 history and O5.
4. Hold `QuiescentVerdictTransaction.checkpoint()` from before snapshot through
   transaction reopen, every physical write, lineage receipt and pointer
   publication.
5. Restore through the required construction order above and only then create
   the executor.
6. Prove active-worker drain, rollback savepoint, BEST-after-resume,
   crash-between-file-and-pointer replay, restore-before-mutation, and
   continuous-versus-interrupted exact next-step equality.

The fresh-v8 and real-n600 gates remain closed until this proof is green.
