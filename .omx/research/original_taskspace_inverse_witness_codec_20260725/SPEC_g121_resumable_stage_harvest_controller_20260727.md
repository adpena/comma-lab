# G121 — resumable immutable-stage harvest controller

Date: 2026-07-27  
Axis: `[encoder-side macOS CPU orchestration; no score authority]`  
Lane: `lane_g121_resumable_stage_harvest_controller_20260727`  
Parent: G111/G112/G120-v2/G119  
Status: implementation contract; G120 v1 is explicitly inadmissible; no
launch, score, candidate, or pointer claim

## Structural decision

Do not put full-n600 public-wire scoring inside the MLX optimization loop and
do not let the trainer's legacy `levelset_best.json` choose the deployable
state. Training and exact codec compilation are separate resumable transducers:

`optimizer -> immutable physical stage nodes -> public-wire compiler ->`
`conditional-pose compiler -> whole-archive selector`.

The optimizer owns search and checkpoint production. G121 owns exhaustive
harvest of every immutable stage state. G120 owns one-stage receiver/scorer
authority. G119 owns the conditional pose value function. Only the final exact
archive/evaluator row owns score ordering.

This separation is required to prevent three signal-loss classes:

1. a floating-state or arbitrary-quantizer BEST pointer suppressing a better
   parsed public-wire state;
2. a semantic-only ordering suppressing a stage with a cheaper conditional
   pose solution;
3. a long CPU scorer pass blocking, perturbing, or coupling itself to the MLX
   optimizer process.

## Production API

```python
harvest_g111_stages_v1(
    *,
    producer_run_dir: Path,
    expected_launch_manifest_sha256: str,
    output_dir: Path,
    progress_dir: Path,
) -> G121StageHarvestResultV1
```

No target float, labels, scorer callback, tensor arrays, checkpoint identity
strings, or BEST path are accepted. The controller derives all inputs from
physical receipts.

## Input enumeration and custody

1. Open the governed launch manifest and bind the G111 program, full-n600
   geometry, batch 16, physical G109 receipt, and fresh-producer mode.
2. Enumerate immutable `fresh_lineage/*.receipt.json` nodes. Never enumerate
   `levelset_best.json`, `levelset_witness_ema_BEST.npz`, rolling
   `levelset_witness_ema_mlx.npz`, or an in-memory model snapshot.
3. Recursively reopen every lineage node and admit only preserved stage/final
   nodes with distinct physical deploy and full-state resume checkpoints.
4. Materialize or reopen one G112 partition for each admitted node.
5. Invoke `run_g120_parsed_stage_production_authority_v2` by G112 receipt path
   plus external SHA only. Reject v1 receipts and schemas.

The controller must be restartable after any stage. Each stage has an atomic
state row keyed by physical G112 receipt SHA and G120 measurement identity.
Completed measurements are reused even when the leaderboard pointer changes;
the live pointer is rebound only to the conditional observation.

## Retention rule

Before pose is measured, retain every stage satisfying the exact integer
inequality:

`100*k*T_den < T_num*N`,

where `k` is the exact public-wire disagreement count,
`N=600*384*512=117,964,800`, and `T_num/T_den` is the reduced rational parsed
from the pointer's exact lexical decimal. Binary floating-point values are
telemetry only.

Rejecting equality is intentional because rate and pose contributions are
nonnegative. This is the only pre-pose pruning rule. Do not prune by semantic
bytes, semantic action, source-float regret, semantic dominance, top-k, or a
fixed number of stages.

For every retained stage, preserve:

- G111 stage name and epoch;
- physical lineage receipt path/SHA and checkpoint ID/root/sequence;
- deploy checkpoint path/bytes/SHA;
- resume checkpoint path/bytes/SHA;
- G112 receipt path/bytes/SHA;
- parsed G105 semantic child identity;
- G112 pose initializer identity;
- exact public plugin-tree SHA;
- exact public-wire `d_seg`;
- exact public-wire disagreement numerator `k` and denominator `N`;
- exact target lexical decimal, reduced rational, and integer comparison
  operands;
- exact four-way semantic-floor archive winner and all alternatives;
- source-float-to-public-wire regret or explicit unmeasured state;
- G120 measurement receipt and pointer-independent cache identity;
- live-pointer observation and post-verification identity.

`g105_public_wire_best.json` may be emitted only as a deterministic scheduling
hint into this retained set. It has no pruning or score authority.

Every attempt receives one typed disposition:

- `RETAIN_POST_G105_POSE`;
- `DEFER_G115_WIRE_QAT`;
- `PRUNE_EXACT_DISTORTION_OBSTRUCTION`;
- `BLOCKED_SCOPED`.

Wire-blocked plus float-state unmeasured/open must defer. Exact pruning requires
the float state also to be exactly blocked or a physical terminal G115 QAT
child to have been measured blocked. Deferred and blocked rows remain in the
append-only attempt ledger and completion proof; only retained rows enter the
G119 manifest.

## Resumability and storage

- Stage discovery, G112 materialization, G120 measurement, retention rebuild,
  and completion each have distinct atomic checkpoints.
- Re-running is idempotent by content identity.
- A partial or corrupt stage never poisons other rows; it produces a scoped
  fail-closed blocker row.
- Large artifacts use the storage waterfall and certified cleanup rules. The
  ledger and all evidence paths are durable and never live under `/tmp`.
- The controller may run after training or opportunistically as a separate
  governed process. It must not share mutable optimizer state or GPU memory.

## Output IR

The durable ledger is an append-only typed row stream. Deterministic reductions
produce:

- `g121_stage_measurements.jsonl`: all completed and blocked stage attempts;
- `g121_retained_prepose.json`: every non-obstructed stage;
- `g105_public_wire_best.json`: scheduling hint only;
- `g121_completion_receipt.json`: exhaustive-enumeration and custody proof.

G119 consumes `g121_retained_prepose.json`, not the trainer BEST. After each
conditional refit, the downstream whole-archive controller computes the exact
three-axis Pareto set over `d_seg`, `d_pose`, and archive bytes and then invokes
the shipped archive/evaluator path.

## Acceptance

- A fixture with a semantically worse stage but a cheaper synthetic downstream
  pose value proves that G121 retains it.
- A pointer-only refresh reuses measurement work and emits a new observation.
- Missing resume custody, broken recursive lineage, nonstage rolling nodes,
  duplicate identities, legacy BEST references, and an injected scorer/target
  are rejected.
- Interruption after any stage resumes without repeating completed scorer
  batches.
- The completion receipt proves that every eligible immutable stage node is
  either measured or carries a scoped blocker.

## Triality

DSL:

`G111PhysicalStage* -> G112Partition -> G120PublicWireObservation`.

DAG:

`physical optimizer stages -> exhaustive harvest -> retained pre-pose set ->`
`G119 per-stage conditional refit -> exact whole-archive Pareto -> evaluate.py`.

Equation:

`K_prepose = {s : 100 k_s T_den < T_num N}`.

No smaller set is justified before observing
`V_pose(s) = min_Y0|Y1 [sqrt(10 D_pose) + lambda R(Y0|Y1)]`.
