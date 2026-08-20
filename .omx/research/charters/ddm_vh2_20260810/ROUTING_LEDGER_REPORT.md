# ddm_vh2 vehicle harvest routing ledger report

Axis: `[macOS-CPU advisory, scorer-free]`; `score_claim=false`; PR130 CPR1 frontier unchanged.

## Result

- Current `.omx/research/*.md` count at materialization: **7093**. The charter's
  7,092 snapshot is one lower and is therefore recorded as a drifted seed, not current authority.
- The charter's `ddm_* artifacts = 3,001` did **not** reproduce under a stated filesystem scope.
- Reproducible vehicle denominator before writes: **1238** top-level `ddm_*` entries;
  files and directories each count once, nested run payloads remain content of one artifact.
- Appended **48** rows; **0** were already present and byte-semantically identical.
- Loaded prior findings: wl1=15, fh1=11, vp1=15. Drained entropy=7/7 artifacts.
- Coverage after append: artifacts=1238, harvested=10,
  routed=10, un-harvested=1228.
- Canonical ledger SHA-256: `8ba96c1e5f65fec4e42ded0e9ec3fedc138b7e3ebe2963067675376092c79dde` before, `4608d8f06b0e8545105e3e26fcf9e859db54e3c8c9a09b1e6489bd8ec9a1bd12` after.

## Typed row dispositions

- `DEAD-ON-THIS-BASE`: 2
- `DEFERRED-WITH-BLOCKER`: 5
- `NEEDS-REMEASURE`: 9
- `ROUTED-FIRED`: 17
- `ROUTED-QUEUED`: 15

No row is UNOWNED. Every queued/fired/remeasure row has owner, consumer, and fire order; every
deferred/dead row has owner, consumer, named blocker, and fire condition.

## Partition-1 choice and finding

Entropy was selected because current PR130 campaign state is rate-dominant while pose is closed by
pk2 and the seg decomposition is blocked. The seven entropy artifacts form a complete, bounded
lineage. Their strongest old-object fact is real: exact entropy reduced a six-stream n64 payload
from 274,664 B to 45,369 B. It does not port as a PR130 win. All eight safe-zero subsets were
infeasible at all five tolerances, the evidence is n64, and PR130 has no matching direct-description
object. The retained payloads remain useful fixtures only after `ddm_rc2` closes the #996 coder-axis
scope review and a new representation creates a distinct probability object.

## Closure boundaries

- `pk2` closes the tested frozen-PR130 pose recode family, not all pose retraining.
- #996 is cited only as **coder axis, scope-under-review by `ddm_rc2`**.
- `113b52fdb1` closes the declared receiver-v7 gauge bank at its 2,000 B trigger.
- #917 closes retired-vehicle instruments as current production routes.
- Decode time was not used as a disqualifier.

## RECALL EVIDENCE

Read the common contract, PROGRAM, operating manual, current hot state, canonical equation index,
sub-0.15 DAG, task-status surfaces, and the complete wl1/fh1/vp1 and entropy source artifacts before
routing. The prior harvest wording and routing were loaded rather than re-derived. Source SHA-256
values are carried on every ledger row; the retained input and manifest carry the complete list.

## Validation boundary

No scorer, trainer, Modal job, upstream file, protected state file, archive evaluator, or exact-score
pointer was touched. This unit builds a means, not goal progress: no exact row moved and the exact
frontier remains unchanged.
