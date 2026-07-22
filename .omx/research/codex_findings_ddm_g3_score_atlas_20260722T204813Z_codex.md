# Codex Findings: DDM G3 Full-n600 Score Atlas

Date: 2026-07-22  
Verdict: `MEASURED_ADVISORY_N600_SCORE_ATLAS_COMPLETE_V13_POINTER_BOUND`

## Outcome

The n600 atlas is complete: 600 typed pair rows cover exact Seg flips, class ×
margin × topology mass, rank-four flip distances, Pose debt/sensitivity, exact
byte-ledger attribution, response geometry, and scene covariates. The SSD JSONL
is 9,121,001 bytes with SHA-256
`faaff7299d86aa49c97e25e9cce2eeb0201f64e919f110015d31708788bcec09`.

Global reconstruction closes to 4,011,236 flips (`d_seg=0.0340036688910590`),
`d_pose=163.034719422881`, and 106,106 exact archive bytes. These are reused
frozen-scorer advisory quantities, not a score claim.

Implementation custody is commit
`7c019626f94d49f2bac5dfe61d5a8ae91e0802d8`; the receipt also re-hashes the
builder, typed library, and config independently.
Compact receipt SHA-256 is
`6c4157092a7bdf7ba44b458cd470725cc470d84a8fc77ed7d3dedb59160734f5`;
hard-pair registry SHA-256 is
`0c9ce6d0ce2b2c0830400f096438355242527d40f682fc1b201f67d8d951a4e4`.

## Concentration

| Currency | top 10 | top 50 | top 100 |
|---|---:|---:|---:|
| Joint Seg + Pose | 1.9785% | 9.5476% | 18.7039% |
| Seg only | 3.5379% | 15.4902% | 26.8544% |
| Pose only | 2.0239% | 9.7521% | 18.9224% |

The honest finding is broad joint debt, not a strong heavy tail. Seg is more
concentrated than the Pose-dominated joint currency.

## Top ten pair debts

| Rank | Pair | Joint mass | Flips | Pair d_pose |
|---:|---:|---:|---:|---:|
| 1 | 523 | 0.0906710 | 7,279 | 204.7154 |
| 2 | 54 | 0.0881681 | 7,140 | 198.9373 |
| 3 | 1 | 0.0875481 | 5,688 | 200.4172 |
| 4 | 90 | 0.0864979 | 5,609 | 198.0352 |
| 5 | 21 | 0.0864549 | 5,077 | 199.0236 |
| 6 | 446 | 0.0858762 | 13,912 | 179.4771 |
| 7 | 0 | 0.0853882 | 5,654 | 195.2544 |
| 8 | 14 | 0.0852144 | 4,377 | 197.4558 |
| 9 | 18 | 0.0851838 | 5,239 | 195.6114 |
| 10 | 327 | 0.0851547 | 13,396 | 178.7888 |

Timeline proxy anchors are pair 279 `intersection_proxy` (rank 133), pair 286
`lane_change_proxy` (rank 204), and pair 452 `lead_car_pass_proxy` (rank 33).
They are deterministic covariate spikes. Visual review supports the visible pass
at pair 452 but does not promote the other names to semantic ground truth.

## Hard-pair registry

- top24/full correlation: `r=0.5953065905` (47 distortion-touched proposals).
- top64/full correlation: `r=0.5750628696` (73 touched).
- stratified-control24/full correlation: `r=0.2341751674` (66 touched).
- Replay source: 338 exact measured v12 proposals; maximum full-objective replay
  residual `1.1e-14`.

The subset contract is measure top24, then top64/control, then full n600. The
moderate r values prohibit subset-only promotion or family closure.

## #36 and v13 truth

#36 is intact and consumable for geometry: all 600 embedded cone-map SHA-256
values were re-derived over 2,540,732,982 current bytes. Its cone/free-budget
fields are nevertheless rotted as score-rank currency and remain diagnostic.
Ranking uses only exact flips plus proportional nonlinear Pose mass. The
canonical operator P0 ledger already marks
`ddm_v13_worldsheet_event_predictor` spawned/in progress; the pointer artifact
binds the ledger SHA and exact row SHA instead of claiming registration is owed.

## Blocker delta versus #603

Resolved: n600 pair/frame score debt, stratum-margin-topology mass, Pose
sensitivity, byte attribution, v10-v12 admission efficiencies, charts, hard
subsets, replay r, costate schema, v13 pointer, stage checkpoints, and cleanup
custody. Remaining beyond this research-only lane: live v13 must consume the
atlas, and each new candidate must refresh subset/full correlation with
contemporaneous frozen-scorer measurements.

Bounded re-derivation:

```sh
/Users/adpena/Projects/pact/.venv/bin/python tools/build_ddm_g3_score_atlas.py \
  --config .omx/research/configs/ddm_g3_score_atlas_n600_20260722.json --resume
```

SSD HTML/PNG and certified cleanup manifest live under
`/Volumes/VertigoDataTier/pact/ddm_g3_score_atlas_n600_20260722T204000Z/`.
Pointer honesty: `0.1910828242 [contest-CPU]` unchanged. MAIN landing review is
required.

STORES CONSULTED: `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, operating manual,
#36 atlas, v10/v11/v12 receipts, v12 scorer caches and byte-close ledger, g2
aggregate ledger, `reports/latest.md`, lane/task/P0 ledgers.
