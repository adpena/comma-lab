# Codex Findings: DDM m5r Road Frame-1 Fresh Solve

Date: 2026-07-23  
Run: `ddm_m5r_road_frame1_fresh_solve_20260723T115443Z`  
Authority: `[macOS-CPU frozen-scorer advisory]`  
`research_only=true`, `score_claim=false`, MAIN landing review required.

## Verdict

`SUBSET_PROPOSAL_NOT_ADMITTED_AT_FULL_N600`.

This was a fresh receiver-closed integer-lattice solve rather than another
measurement of the v19b greedy stack. The top-24 g3 proposal solve selected
`lane_program_340_-1`, but its full-n600 replay worsened the joint objective
from `43.217068235930284` to `43.32467839132237` (`+0.107610155392086`).
Therefore it earns no reach or Catalog #366 scope credit.

Verdict scope:
`INSTANCE:V15_368_RECEIVER_EFFECTIVE_INTEGER_DOF_X_TOP24_PROXY_SCREEN_X_EXACT_RESTRICTED_MASTER_ORDER4_X_C1_200000_BYTE_BOX`.
This is not a negative claim about the representation family.

## Parameterization Truth

The historical Catalog #627 “706 parameter” label is a superseded named-field
overcount, not an executable receiver surface. Re-derivation from the current
compiler gives 368 receiver-effective integer coordinates:

| Receiver-consumed group | DOFs |
|---|---:|
| G1 island worldsheet translations | 326 |
| Lane program wire coordinates | 24 |
| Shared template RGB coordinates | 18 |
| **Total** | **368** |

Aspect/rotation lift metadata and BEV/range seed fields have no current decoder
wire coordinates. The solve screened 188 active top-24 directions from all
three receiver-consumed groups, exactly replayed 23 singleton states and 24
non-greedy sets through R and the frozen scorers, and recorded two instance
compiler refusals for `coherent_worldsheet_x_±1` escaping scorer geometry.

Catalog #631 v18b solve-generated columns could not be silently composed:
their common-master archive SHA
`50332acf742717f463111cc0ead2878c33a9e5d4fa7cc15dee9329bdafca8714`
differs from the V15 lift master SHA
`759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
Verdict scope:
`INSTANCE:V18B_POSTSOLVE_COMMON_MASTER_COLUMNS_X_V15_G1_LIFT`.
The column family remains open pending a reviewed hybrid compiler.

## Exact Top-24 Reach Curve

The g3 top-24 subset has measured subset-to-full Pearson
`r=0.5953065905385343` over 338 proposals, so it is proposal-only and cannot
rank or kill a family. Its exact receiver/scorer byte-Pareto envelope is:

| Exact subset state | Archive bytes | Road errors closed | Road candidate errors | Joint objective |
|---|---:|---:|---:|---:|
| `lane_-1 + lane_+1 + island58_y+1 + island122_x+1` | 133,941 | 689 | 72,014 | 46.392953923437204 |
| `coherent_worldsheet_y_-1` | 133,944 | 936 | 71,767 | 46.471360459265900 |
| `lane_program_340_-1` | 134,211 | 2,017 | 70,686 | 46.293485007608815 |

The normalized distance-to-chord knee is
`coherent_worldsheet_y_-1`: 133,944 bytes and 936 Road errors closed.
The joint-objective winner, not the Road-only knee, was
`lane_program_340_-1`.

## Full-n600 Endpoint

The selected archive SHA is
`db68ecd1b9b47d2039be5248a59f48b6e5f313bb9fe4a0493b35af46b849bd81`.
All 38 batch-16 checkpoints bind that SHA.

| Stratum | V15 control errors | Fresh-solve errors | Net errors closed |
|---|---:|---:|---:|
| Road | 2,210,770 | 2,239,369 | -28,599 |
| Lane | 300,563 | 452,173 | -151,610 |
| Undrivable | 236,896 | 209,839 | +27,057 |
| Movable | 425,853 | 411,303 | +14,550 |
| MyCar | 66,446 | 55,982 | +10,464 |
| **All** | **3,240,528** | **3,368,666** | **-128,138** |

The selected state is inside the 200,000-byte box at 134,211 bytes, but is not
admitted. Its objective component deltas versus V15 are:

- Seg: `+0.108623928493924`
- Pose: `-0.001193555019178`
- Rate: `+0.000179781917343`
- Total: `+0.107610155392089`

Road solvable fraction at the c1 box is therefore credited as zero for this
instance. The non-exhaustive Road infeasible-residual interval remains
`[0, 2,210,770]`; no point residual is certified.

## Catalog #366 And Greedy Confound

Catalog #366 true-scope remains `[0, 2,377,273]` errors. It is not narrowed,
because the selected subset state failed the full-n600 joint-objective gate.
`numeric_certified_residual=null`.

The v19b greedy instrument closed 82,824 Road errors (`3.7463870054%`).
This fresh restricted solve closed `-28,599` (`-1.2936216793%`) at full n600,
a gap of `-111,423` errors versus greedy. Verdict:
`GREEDY-INSTRUMENT-CONFOUND-NOT-CONFIRMED_BY_THIS_INSTANCE`.
This does not validate a v19c family asymptote; it only fails to confirm the
confound under this exact restricted solve.

## Custody And Re-Derivation

Receipt:
`.omx/research/ddm_m5r_road_frame1_fresh_solve_20260723T115443Z/receipt.json`
(SHA-256
`5e0886321e0a5f8d7d24b45ce6db9137f417d4119b8c2100bb18f7f1a5a44745`).
Typed config SHA:
`b11a6315a94cc644be25f957f90cc0a4792476bc7cd1f1ae096b15914c5917d1`.
Batch-chain SHA:
`ded64c5ef8e76608f9ff43ea9c7a94bbeb3b1bcafaa4c04fb9309145fc90e343`.

Re-derive, resuming all immutable checkpoints:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /Users/adpena/Projects/pact/.venv/bin/python \
  tools/measure_ddm_m5r_road_frame1_fresh_solve.py \
  --config .omx/research/configs/ddm_m5r_road_frame1_fresh_solve_20260723.json \
  --output-directory \
  .omx/research/ddm_m5r_road_frame1_fresh_solve_20260723T115443Z
```

STORES CONSULTED: `CLAUDE.md`, `AGENTS.md`,
`PROGRAM.md`, the m3 and m5 receipts, g2g2 Catalog #608 findings, Catalogs
#547/#549/#559/#391/#580/#532 findings and equations, V15/V16/V17/V18b/V19/V19b
receipts and configs, the SHA-pinned g3 hard-pair registry, lane registry,
subagent progress, latest Codex findings/session summaries, latest T3/design
memos, cost-band/continual-learning state, and probe-outcomes ledger.

Pointer: `0.1910828242 [contest-CPU]`, unchanged.

