# Codex findings — DDM CO4 Road-local ranker and precision propagation

UTC: 2026-07-24  
Lane: `lane_ddm_co4_road_local_and_precision_20260724`  
Receipt content SHA-256:
`f9e8b0e4e4f7d025a05f26470942d03b8418fb98871581baa42f3d3297b74693`.

## Verdict

`PARTIAL_DEFECT_CURE_WITH_FORMULATION_SCOPED_ROAD_FAILURE`.

The full-field precision defect is closed by validated propagation:
`15 DIRECT`, `585 PROPAGATED`, `0 UNRANKED`. The Road-local ranking defect is
not cured by the two preregistered ridge formulations. The better frozen
candidate, `g3_stratum_experts`, measured:

- Road held-out NDCG@4: `0.1796465097835245` over 288 pairs;
- Road held-out Spearman rho: `0.6927251495713694`;
- global held-out NDCG@4: `0.8133546756293046`;
- global held-out Spearman rho: `0.8065274678827845`.

The Road `0.60` admission bar failed. The verdict is restricted to these two
frozen ridge forms; the Road-local family remains open. The organ retains the
exact sealed CO3 OOF predictions and global duty authority.

## Precision and safety

Every one of the 1,200 MS4D rows is joined to the identically keyed PF2
assignment row. The builder fails closed on duplicate/missing buckets,
event-count disagreement, non-conserved pair-support mass, pair-membership
disagreement, non-finite/non-PSD Grams, or an incomplete N600 support vector.

The 585 propagated rows state their assumptions and use
`1 + CV(contribution trace)^2` as a design effect. Measured design effects are
strictly greater than one, so every emitted propagated interval is wider than
its nominal interval. Direct blocks override propagation. Adjacent overlapping
intervals are labeled `TIED`.

## Other required dispositions

- #611: reactivated construction route, typed blocked pending a typed counted
  scorer-recursive application operator with receiver parse-back and realized
  evaluator-cell custody.
- Unsound-method audit: clean across the 14 consumed CO2/CO3 files. No `77x`,
  `2.71x`, `params^-0.71`, or label-noise-ceiling prior was found; zero rows
  required purging.
- M34 replacement: aggregate RD1 prices remain provenance only.
  State-indexed dual consistency is
  `AWAITING_J8F_M34_PER_STATE_DUALS`.
- Bellman residual: `AWAITING_J8F`; OOF innovations are not promoted to an
  ordered trajectory.
- Innovations were recomputed on the retained sealed CO3 vector. Lag-one
  correlation is `0.06679580189709491`, a scoped
  `NO_LARGE_LAG_ONE_COLOR_DETECTED` diagnostic.
- MS2R immutable-stage mismatch: the stored oracle admission stage had
  `WRAPPED=14`, `TYPED-GAP=7`; fresh GC2-bound coverage has
  `WRAPPED=21`, `TYPED-GAP=0`. The 72 structural diffs are input
  supersession, not checkpoint corruption or CO4 drift. Preserve the immutable
  stage; any rerun requires a new stage/config identity.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, and `docs/operating_manual_craft_handoff.md`;
- canonical lane registry, subagent progress, frontier report, G3/EV1 source
  discovery, and campaign source registry;
- sealed CO3 ranker receipt and RD1 typed dual receipt;
- scorer-value oracle rows for MS4D margin-Fisher, PF2 assignments, Pose6
  tubes, and G4 stationarity;
- G3 hard-pair atlas, PF2 support table, and MS2R immutable stage 01;
- per-arm and broadcast inboxes.

## Verification

- Two clean review events per entity in every changed Python file.
- `47 passed` across ranker, campaign evidence, campaign state, and costate
  organ tests.
- Ruff, format check, Python compilation, diff check, receipt determinism,
  campaign determinism, authority-firewall assertions, and four-consumer
  state-digest equality are clean.
- Local SSD venv only; no provider, GPU, archive, run, external task, score, or
  pointer mutation.

MAIN must review the branch diff and the formulation-scoped Road rejection
before merging.

