# FEED — DDM M5 multicoefficient solve coverage

**research_only=true · score_claim=false · MAIN review required**

```text
M3 control counts + C1 200,000 B box
                  |
                  v
canonical CLASS_ORDER import (self-detected IDs)
                  |
                  v
v15 133,941 B control receiver -----+
                                     +-> uint8 camera -> exact R -> frozen scorers
v19b 137,825 B integer stack -------+                 (38 x batch-16, n600)
                                                           |
                                                           v
                                   per-stratum H helpful / C collateral / residual
                                                           |
                           +-------------------------------+------------------+
                           |                                                  |
                           v                                                  v
              measured partial Road/Lane reach                  certificate eligibility
              every class has C_k > 0                           finite set: NO
              zero solved strata                                exhaustive proof: NO
                                                                isolated solves: NO
                                                                          |
                                                                          v
                                           NUMERIC_TRUE_SCOPE_NOT_CERTIFIABLE
                                           #366 interval remains [0, 2,377,273]
```

## Receiver-closed transition table

| frame_1 stratum | control errors | helpful closed | harmful collateral | net closed | candidate residual | shared candidate bytes | zero collateral | certified infeasible |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Road | 2,210,770 | 114,814 | 31,990 | 82,824 | 2,127,946 | 137,825 (+3,884) | no | not certifiable |
| Lane | 300,563 | 4,636 | 2,633 | 2,003 | 298,560 | 137,825 (+3,884) | no | not certifiable |
| Undrivable | 236,896 | 11,426 | 36,617 | -25,191 | 262,087 | 137,825 (+3,884) | no | not certifiable |
| Movable | 425,853 | 84,409 | 57,035 | 27,374 | 398,479 | 137,825 (+3,884) | no | not certifiable |
| MyCar | 66,446 | 17,255 | 943 | 16,312 | 50,134 | 137,825 (+3,884) | no | not certifiable |

The byte delta is a shared joint-stack cost, not a per-class allocation. It is
repeated in the table only to make the non-attributability explicit.

## Findings routed

- **Road decisive bit:** partial receiver-closed inverse reach is measured, but
  full Road closure and zero-collateral feasibility remain unknown. The result
  neither collapses nor certifies the trunk-territory scope.
- **Lane full-solve bit:** not admitted. G2G2's 0/6 result was
  under-parameterized and failed semantic/pose predicates; this n600 replay
  proves nonzero Lane reach while also measuring collateral.
- **#366:** no config change and no launch. A numeric `X` remains withheld
  because the universal-negative proof legs are absent. The current interval is
  `[0, 2,377,273]`; M3's v19b counterfactual point remains `2,303,328`, not a
  certificate.
- **#536:** consume helpful/collateral separately in the three-axis waterfill;
  do not price net flips as if they were isolated reach.
- **#541:** the constructive solve now has a full-n600 receiver transition
  endpoint and a strict certificate eligibility predicate.

## Re-derivation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/adpena/Projects/pact/.venv/bin/python \
  tools/audit_ddm_m5_multicoefficient_solve_coverage.py \
  --config .omx/research/configs/ddm_m5_multicoefficient_solve_coverage_20260723.json \
  --output-directory .omx/research/ddm_m5_multicoefficient_solve_coverage_20260723T103457Z
```

The run is `$0`, local foreground CPU only, resumable from 38 immutable scorer
batch checkpoints, and writes no large camera tensor or candidate archive.
R1 is signal-only (`d_pose=0.001610`, `7,195 B`); zero R1 bytes are consumed.

Pointer `0.1910828242 [contest-CPU]` unchanged.

