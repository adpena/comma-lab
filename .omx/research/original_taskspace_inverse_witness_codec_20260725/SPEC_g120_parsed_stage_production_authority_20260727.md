# G120 — parsed semantic-stage production authority and cross-stage Pareto

Date: 2026-07-27  
Axis: `[macOS-CPU exact parsed-wire producer screen; non-promotable until final archive eval]`  
Parent: G111/G112/G115/G117/G110/V10  
Status: implementation contract; no score or pointer claim

## Objective

Turn G117's intentionally injection-based, engine-only selector into a
production authority wrapper that cannot self-attest its target, labels,
scorer, checkpoint, pose initializer, or frontier value. Wire that wrapper at
every immutable G111 stage checkpoint and retain the cross-stage Pareto set
needed by the later conditional-pose value function.

## Required production custody

The wrapper must open, not merely accept claimed hashes for:

1. a recursively verified G112 partition receipt and its physical G111
   deploy/resume/fresh-lineage chain;
2. the physical G109 aggregate receipt named by that checkpoint projection,
   including exact n600 labels, batch-16 geometry, source video, SegNet weights,
   and upstream source closure;
3. the frozen CPU SegNet loaded from the physically rehashed weight file and
   reviewed upstream source;
4. a fresh `DynamicFrontierTargetSnapshot`, recomputed from the canonical
   pointer before the screen and reverified after all 600 pairs finish;
5. the exact shipped G110 public plugin tree, including its tree SHA and a
   repository/public population-equality proof.

G117 callback/array/float injection remains test-only and its receipt remains
engine-only. A G120 production receipt may exist only after all four custody
domains agree.

## Exact execution object

For each preserved G111 stage:

1. materialize/reopen its G112 semantic child and generated-Y1 pose initializer;
2. compile both G105 Y1 wire families from the exact semantic child;
3. wrap each in receiver-valid rank-zero G110 (`Y0 == Y1`);
4. race ZIP STORE and DEFLATE on exact counted archive bytes;
5. parse back, double-decode G105/G110 through the shipped public plugin tree,
   realize V10 camera `uint8` Y1, and prove equality with the repository twin;
6. run frozen CPU SegNet in chronological batches `37x16 + 1x8`;
7. atomically checkpoint every batch under an identity binding semantic packet,
   archive variants, source labels, scorer, public plugin tree, G112
   initializer, and physical G111 stage;
8. post-verify the same pointer object before publishing production authority.

The expensive measurement-cache identity is pointer-independent: a leaderboard
refresh cannot invalidate unchanged pixels, labels, scorer, packet, or public
runtime. The live pointer is bound only to the conditional observation formed
after measurement, and it must still be reverified before that observation is
published.

## Coupled retention, not arbitrary thresholds

The per-stage semantic lower bound is:

`A_sem = 100*d_seg_wire + 25*archive_bytes/37_545_489`.

`100*d_seg_wire >= target` is a strict obstruction only for that exact parsed
Y1/stage/pointer. If only `A_sem >= target`, disposition is the scoped
`DEFER_POST_G105_POSE_REFIT`, not a representation-family impossibility.

Do not retain only `argmin A_sem`, a semantic Pareto set, or a hand-selected
"semantically second-best" row. Until conditional pose has been measured, keep
every physical stage satisfying:

`100*d_seg_wire < target`.

This is the only sound pre-pose pruning rule: a stage may have worse semantic
distortion and more semantic bytes yet carry a substantially cheaper
conditional pose solution. Preserve every such row in
`g105_public_wire_pareto.json` with at least:

- exact `d_seg_wire`;
- exact receiver-valid archive bytes;
- measured source-float-to-wire regret when available, otherwise explicit
  unmeasured status;
- G112 pose-initializer identity;
- paired deploy checkpoint, full-state resume checkpoint, and recursive fresh
  lineage receipt.

After G119 measures exact conditional `d_pose` and final receiver-valid archive
bytes, compute the true three-axis Pareto set and score ordering. A semantically
inferior stage may be the full-score winner. `g105_public_wire_best.json` is
only an execution pointer into the preserved pre-pose set. It must never name
`levelset_best.json` or the legacy arbitrary-scale int8 BEST.

## Trainer/DSL integration

Add one explicit typed G111-only flag. Default-off behavior elsewhere must
remain byte-identical. When enabled, fail closed unless all of these hold:

- n600, verdict batch 16;
- physical G109 target binding;
- fresh producer with physical recursive lineage;
- HOSC, exact polar basis, self-orient off;
- render grid 384x512, `render_aa=none`;
- generated-Y1 pose source using V10;
- every selected row names paired deploy/resume/lineage custody.

The semantic training loss remains pre-G105 floating-state optimization.
Post-hoc parsed-wire stage selection closes deployment custody. Terminal
wire-QAT is a separate resumable stage admitted only if measured quantization
regret justifies it.

## Acceptance

- Focused unit tests reject stale/hard-coded frontier values, fake labels,
  wrong G109/SegNet/upstream hashes, changed pointer during a screen, arbitrary
  callback production use, missing paired resume or lineage custody, and a
  legacy BEST pointer.
- A strict fixture proves per-batch resume, exact four-way archive arbitration,
  public/repository population equality, pointer-independent measurement reuse,
  retain-all-below-distortion-obstruction behavior, and deterministic
  tie-breaking.
- G111 real physical compile emits the explicit selector flag and is no longer
  held on the parsed-wire-selection blocker.
- No full-n600 launch occurs until a governed clean dry-run proves the new
  production wrapper, storage preflight, physical cold root, and crash resume.

## Do not touch

Do not alter the pinned upstream snapshot, scorer weights, candidate payload
lineage, or public runtime semantics. Do not claim a candidate, score, frontier
move, or pose result from a semantic-only screen.
