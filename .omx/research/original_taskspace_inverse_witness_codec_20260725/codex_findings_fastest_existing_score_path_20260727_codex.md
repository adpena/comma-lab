# Codex findings — fastest existing exact-score path

Date: 2026-07-27  
Lane: `lane_fastest_existing_score_path_20260727`  
Verdict: no existing fresh-original archive can lower the `0.172` effective
frontier; do not divert from G111 into G25/G29 or G55

## Outcome

The attractive short path is illusory. G25's `80,238`-byte LVPG2 archive is a
real, exact, same-state rate improvement, but G28 already measured that state
at full n600:

```text
d_seg = 0.0035127170849591494
d_pose = 127.35955810546875
distortion-only score = 36.03874263394379
score at 80,238 bytes = 36.092169824624406
```

No public decoder can change a lossless same-state recode's distortion. Closing
or dispatching G29 would therefore buy a score near `36.09`, not a frontier
attempt. The correct shortest action is to finish the real fresh G111 producer,
then score its first immutable parsed-G105 stage through the already-landed
G121/G120/G119/G110 chain.

The machine-readable receipt is
`fastest_existing_score_path_blocker_receipt_20260727.json`.

## Audited alternatives

| path | exact state today | hard conclusion |
|---|---|---|
| G20/G25/G29 ep725 | 80,238 B, same state as G28 | zero-rate score `36.0387`; decoder closure is a dominated diversion |
| G41/G42/G45 label-local G | real pairwise receiver seam | no fresh n600 G-page population, changed-state score, counted outer archive, or coupled pose operand |
| C1/MS1 full lattice | batch-16 distortion term `0.047109` | historical payload is forbidden and 409,526,925 B must become at most 187,563 B while preserving distortion; this is a new factorized producer, not a decoder patch |
| MS2R tolerance-capped solve | 291,205,400 B | zero-rate score `0.523838`; same-state recoding can never reach `0.172` |
| G51/G55/G57 direct layers | exact public n600 row exists | already scored `39.305935`; redispatch would repeat a settled failure |
| G49/G108/G110 product | public ABI/runtime exists | no real fresh G112 partition, parsed-G105 producer, conditional pose refit, or archive bytes exist |

This audit therefore found no honest ready archive whose missing work is merely
packaging, public decode, or evaluator dispatch.

## Why the full lattice still matters

The full-lattice result is not discarded. It proves that the scorer fiber has a
very low-distortion representative. Its failure is representation:

```text
selected raster state -> hundreds of MB
needed counted program -> at most 187,563 B for sub-0.172 at the same distortion
```

The required reduction is about `2,183x`, so the missing object is a semantic
program/factorization/learned irreducible quotient. G111 is precisely the
fresh-own-lineage route that tries to compile that object. Re-running the old
raster solve or wrapping it in another lossless codec cannot bridge the gap.

## Shortest frontier-relevant exact path

1. Finish G111 native-v3 total-trajectory write/restore and its governed
   cold-root/resume proof.
2. Launch the fresh n600 batch-16 producer with immutable stages and the
   receipt-bound G121 monitor.
3. Score the first complete preserved stage immediately through G112 and
   G120-v2 exact parsed-G105 public-wire measurement. This is the earliest new
   full-n600 component row; do not wait for terminal training to learn whether
   the vehicle has entered the target sublevel.
4. Apply G119 conditional pose inverse/refit to retained non-obstructed stages.
5. Let G110 race complete archive variants, public double-decode the selected
   exact bytes, and invoke `upstream/evaluate.py`.

This sequence shares every step with the final candidate path. It does not
create a throwaway proxy score.

## Ownership and no-clobber boundary

The shared checkout is dirty. Root owns the trainer, fresh-lineage producer,
native-v3 opener, and current checkpoint integration. This audit changed none
of those files and launched no scorer or training process. Its only source
artifacts are this findings memo and the adjacent JSON receipt. The lane
registry entry was added through the canonical lane tool but is intentionally
left for the shared owner to serialize with the other concurrent registry
updates.

## Triality and system intelligence

- DAG: old recodes and public wrappers terminate at measured infeasible states;
  the live branch is G111 -> G121/G120 -> G119 -> G110.
- Equation: a decoder-only action has `Delta d_seg = Delta d_pose = 0`; if the
  current state's zero-rate score exceeds the target, decoder work cannot admit
  it.
- DSL consequence: candidate routing must require a fresh state-changing
  producer before public-closure work when the same-state zero-rate lower bound
  is already infeasible.
- Bit allocator/Pareto: G25's `-789` bytes remain valid same-state signal but
  have zero acquisition priority until a compatible state is inside the
  distortion sublevel.
- Autopilot: suppress G25/G29 and G55 redispatch; wake on a real G111 immutable
  stage.
- Continual learning: retain the distinction between low byte count and
  competitive state; neither alone is candidate readiness.

## Pointer honesty

No score was produced, no candidate was built, and the pointer did not move.
The existing direct public row and ep725 row are far outside the frontier. The
next exact frontier attempt starts only when G111 emits the first real immutable
stage.
