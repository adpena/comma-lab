# Codex findings: DDM AT1x scorer atlas materialization

Date: 2026-07-23  
Authority: `codex_delegate:ddm_at1x_atlas_materialize:20260723T203027Z`  
Lane: `lane_ddm_at1_scorer_analytic_atlas_20260723`  
Scope: `FIRST-RUNG=true`, `research_only=true`, `score_claim=false`  
Axis: `[macOS-CPU locked-env upstream frozen-harness advisory]`  
Landing: isolated Codex worktree only; MAIN review is required.

## Disposition

`PASS_ATLAS_MATERIALIZED__FULL_EVALUATE_SH_BLOCKED_MISSING_LOCKED_BROTLI`

The requested closed-form and n600 contraction atlas exists with complete
custody. The scorer-only official replay is measured and exactly reproduces
the prior reported axes. The full official `evaluate.sh` path does not pass:
the exact locked environment reaches E2 `inflate.py`, then fails before scorer
execution because `brotli` is absent from `upstream/pyproject.toml` and
`upstream/uv.lock`. No dependency was injected and no upstream byte changed.

## Evidence

### Exact lock and source inventory

- The literal `uv sync --locked --group cpu --python 3.11` validation attempt
  rejected stale lock metadata. Receipt
  `35436a27086cfdf92e603ac60ef60d7b5362555b75b2541728fbceec162f59a8`
  preserves the exact failure.
- `uv sync --frozen --group cpu --python 3.11` then consumed the supplied
  `uv.lock` verbatim, without attempting to update it. The preserved SSD
  environment contains 19,505 files / 539,126,642 bytes and has tree SHA-256
  `fcf16c85bf098ff886d711a1306773b8e1ac1ac5390f9fae6dfb6fcab602287f`.
- Rebuilt inventory: `PASS_LOCKED_LIBRARY_SOURCES_MATERIALIZED`, zero version
  drift, 616 PoseNet modules and 540 SegNet modules.

### Frozen closed forms

`DERIVED`: 438 source-line-bound factors totaling 667,132,546 shard bytes:

- 166 eval BatchNorm affine tables;
- 24 actual SE gates: 23 SiLU and one ReLU;
- 203 native two-dimensional convolution-kernel DFT tables;
- 45 observed BN-to-SiLU compositions.

Every factor carries checkpoint, locked source line/hash, version-set,
freshness, consumer, and content-hash custody. The 45 overlapping layer
opportunities are non-additive pools. No global DFT dead band is claimed:
the exact set is empty and the decision is `REFUSE_ZERO_BYTE_TRUNCATION`.

### Settled n600 VJP materialization

`MEASURED_FROM_SETTLED_VJPS`: 600/600 pair checkpoints, 4,200 tensor-index
rows, and 600 contraction rows at the only two measured relay depths:
`scorer_plane_y` and `camera_input_x`. Every tensor index row carries archive
and decompressed-tensor hashes, shape, dtype, producer version stamp, and
version-set SHA. No internal-layer measurement is inferred.

Mean n600 contractions:

| Surface | scorer plane | camera input |
|---|---:|---:|
| Pose six-row Gram trace | 0.009099998989996678 | 0.00404366902722937 |
| Seg contracted singular energy | 309098.2538985123 | 137470.44368699292 |

The Seg rows bind explicitly to
`segnet_head_rank4_linear_flipdist_v1`. Exact V19 joins are pair IDs
`53, 278, 296, 346, 416, 447, 501, 547`; the other 592 rows are explicitly
`GAZE_MEASURED_V19_JOIN_OWED_COUNTED_INERT`.

### Calibration and realization gap

`MEASURED`, scorer-only official `evaluate.py` over the prior SHA-certified E2
inflation, locked Python, CPU, seed 1234, 600 samples, 1,663 seconds:

- archive bytes: 343,466;
- `d_seg=0.02861482`;
- `d_pose=162.58094788`;
- recomputed total: `43.41150975143166`.

Locked-minus-observed drift is zero for archive bytes, d_seg, d_pose, and all
three score terms; the total differs only by binary64 formula roundoff
`-3.410605131648481e-13`. The frozen-scorer realization remains
`d_seg=0.027470296224`; its observed-minus-frozen gap is
`+0.0011445237759999984` (`+0.001144523776` at the requested precision).

This does not convert into a full-harness pass. The separate full
`evaluate.sh` receipt is
`5bf8959d8c5ae9f5a5b095e4a1c6c5ec93b43163e5be8df00c5bab96d1c5488f`
and remains `BLOCKED_LOCKED_RUNTIME_DEPENDENCY: brotli`.

### Explicit zeros and authority limits

- Amplitude factors: zero, because no factor has a through-R/uint8 survival
  row.
- Score claim: false. No contest-CPU/CUDA result or frontier promotion is
  inferred.
- Pointer: unchanged.
- Storage: factor shards and the certified environment remain on the SSD;
  certify-or-block prevented deletion.

## Durable receipts

- Tracked summary:
  `.omx/research/ddm_at1x_atlas_materialize_20260723/atlas_receipt.json`.
- SSD atlas manifest:
  `/Volumes/VertigoDataTier/pact/evidence/ddm_at1x_atlas_materialize_20260723T203027Z/atlas_manifest.json`,
  16,926 bytes,
  SHA-256 `251cc1e4268fb909a9f9a3ac2af845614c98aab15948f90d33d43a8c1542a1d9`.
- SSD n600 contraction atlas:
  `/Volumes/VertigoDataTier/pact/evidence/ddm_at1x_atlas_materialize_20260723T203027Z/gaze_contraction_atlas.json`,
  3,178,175 bytes,
  SHA-256 `a9e444cd7652061368d6daa8cf85e4c793c1b57da385890799ce7de19fd311e4`.

## STORES CONSULTED

- `CLAUDE.md` and `AGENTS.md`;
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`;
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`;
- `.omx/research/OPERATING_MANUAL.md`;
- `.omx/research/ddm_at1_scorer_analytic_atlas_20260723T194312Z/`;
- `.omx/research/ddm_v19_pure_priced_objective_20260723T041500Z/`;
- the settled n600 VJP campaign receipt under the SSD evidence tier;
- the pinned E2 archive and prior SHA-certified inflation;
- broadcast directives through `2026-07-19T19:48:01Z`;
- the per-arm inbox, which was absent/empty at each checkpoint.

## MAIN landing review

MAIN must review the isolated commit before merge. In particular, verify:

1. real ReLU versus SiLU SE activation binding;
2. no reinterpretation of the 592 V19 join-owed rows;
3. scorer-only calibration is not presented as full `evaluate.sh` success;
4. no new amplitude or zero-byte DFT claim;
5. only the owned files in the isolated commit are landed.
