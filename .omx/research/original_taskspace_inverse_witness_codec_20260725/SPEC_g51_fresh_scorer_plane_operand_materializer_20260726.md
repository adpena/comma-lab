# G51 Fresh Scorer-Plane Operand Materializer

Date: 2026-07-26  
Owner: G51  
Status: COMPLETE — DIRECT_TASK_LAYERED operand custody only  
Axis: `[encoder-only exact source-derived operand custody]`  
Authority: research-only; never a score or candidate payload

## Goal

Produce the missing own-lineage n600 scorer-plane operands for the codec race:
chronological `Y0`, `Y1`, current batch-16 SegNet target labels, and bound
`gt_poses`, without reading any historical C1/V15 archive, plane, receipt, or
payload file.

This materializer is not an inverse-solved witness and not a candidate. It is a
fresh compiler of source-derived scorer coordinates. The generic operator is:

```text
Y0[p] = round_u8(DisjointResizeOperator(gt_f0[p]))
Y1[p] = round_u8(DisjointResizeOperator(gt_f1[p]))
```

The sealed source cache may lawfully contain other members, but the
implementation opens only `n_pairs`, `gt_f0`, `gt_f1`, and `gt_poses`.
Current semantic custody comes from the fresh batch-16 teacher receipt and its
target-label bank. Equal output hashes with older derivations are allowed and
are not copied provenance.

## Closed production inputs

- sealed `gt_n600.npz` file identity;
- exact `gt_f0`, `gt_f1`, and `gt_poses` member shapes, dtypes, byte counts,
  and raw-content SHA-256 values;
- fresh batch-16 teacher receipt file SHA and sealed self-hash;
- fresh target-label bank path, shape `[600,384,512]`, `uint8`, bytes, and SHA;
- exact `DisjointResizeOperator` implementation-source identity;
- current materializer module and CLI producer source identities;
- typed config file SHA and canonical config identity;
- fixed n600 geometry and five chronological 120-pair stages;
- SSD-first output root, resumability, and storage admission.

The typed config has an exact-key allowlist. It contains no C1/V15 archive,
plane, prepare-root, receipt, payload, or SHA field. Production validation
rejects forbidden-lineage key/path vocabulary and local output roots.

## Durable stages

For ranges `[0,120)`, `[120,240)`, `[240,360)`, `[360,480)`, and
`[480,600)`, write atomically:

- `Y0` uint8 bytes, shape `[n,384,512,3]`;
- `Y1` uint8 bytes, same shape;
- `gt_poses` float32 bytes, shape `[n,6]`;
- one immutable stage manifest binding the range, source/config/producer
  identities, file bytes/hashes, source pose-slice hash, fresh label-slice
  hash, exact rederive equality, and false-authority fences.

Every resume reopens and rehashes the stage files, then independently
rederives each pair from the sealed source operands and proves byte equality.
The final aggregate binds all five ordered stage hashes, their digest chain,
whole-population output hashes, the fresh label/pose custody, source and
producer identities, storage/cleanup records, and `600/600` coverage.

Temporary stage files are success-only scratch and are removed automatically.
Completed stage files are sacred. No completed output is overwritten.

## Public typed loader

`FreshScorerPlaneOperandLoaderV1.open(receipt_path, expected_sha256=...)`
strictly validates the aggregate and its five stages without consulting
historical C1/V15 state.

`iter_stages(max_pairs=120)` yields read-only mmap-backed:

- `pair_range` and chronological `pair_ids`;
- `y0_u8`: `[n,384,512,3]`;
- `y1_u8`: `[n,384,512,3]`;
- `target_labels_u8`: `[n,384,512]` from the fresh batch-16 bank;
- `gt_poses_f32`: `[n,6]`.

The loader also exposes bounded substage iteration for consumers that require
smaller chunks. It refuses range gaps, extra/missing stages, hash drift,
mutable files, non-SSD production custody, or any candidate/score claim.

## Triality

- DSL: exact config/receipt/stage schemas, fixed source allowlist, false
  authority, no historical candidate lineage.
- DAG: sealed source/cache plus fresh labels/pose custody -> exact integer
  source-to-scorer compiler -> five immutable stages -> aggregate/digest chain
  -> typed G52 loader -> codec search -> receiver closure -> exact evaluator.
- Equations: exact rational resize numerator/denominator followed by
  nonnegative half-up uint8 rounding; no learned transform and no historical
  payload reconstruction.

## Acceptance before heavy launch

```bash
.venv/bin/ruff check \
  src/tac/witness_control/taskspace_fresh_scorer_plane_materializer_v1.py \
  src/tac/witness_control/tests/test_taskspace_fresh_scorer_plane_materializer_v1.py \
  tools/materialize_taskspace_fresh_scorer_planes_n600.py \
  tools/tests/test_materialize_taskspace_fresh_scorer_planes_n600.py

.venv/bin/pytest -q \
  src/tac/witness_control/tests/test_taskspace_fresh_scorer_plane_materializer_v1.py \
  tools/tests/test_materialize_taskspace_fresh_scorer_planes_n600.py

.venv/bin/python tools/materialize_taskspace_fresh_scorer_planes_n600.py \
  .omx/research/configs/taskspace_fresh_scorer_planes_n600_20260726.json \
  --preflight-only
```

Tests must mutate real source arrays and prove exact operator output, atomic
stage persistence, resume rederive equality, stage/config/source/label/pose
tamper refusal, forbidden C1/V15 config refusal, loader range/hash closure, and
production n600/120-pair/SSD gating. Tests are implementation evidence only.

Only after all gates pass may the full materialization launch through the
governed RSS/storage runner.

## Closure

Focused Ruff and pytest passed (`7 passed`). Production preflight sealed at
`/Volumes/VertigoDataTier/pact/taskspace_fresh_scorer_planes_n600_20260726/00_preflight_receipt.json`
(file SHA-256 `fff8973c89cdde75249f1dfe216432af333eb718c6fe5a94b96e373148f1399b`).

The first governed launch failed closed before any stage at its deliberately
tight 2 GiB cap. The unchanged config then completed through the governed
runner at an 8 GiB cap: peak RSS 3,862 MiB, elapsed 50.256 seconds, five
immutable 120-pair stages. The aggregate is:

`/Volumes/VertigoDataTier/pact/taskspace_fresh_scorer_planes_n600_20260726/aggregate_receipt.json`

File SHA-256:
`ae9048dfc24947a6268315590b65da02b56549379e347cbaced25e2e6f67d915`.
Sealed self-hash:
`4363827c2aeb613916029d8bacde8aeb4ded961c4d1ca297310a1e53e204619c`.

Strict loader reopen passed over all 600 pairs with read-only mmap-backed
Y0/Y1, fresh batch-16 labels, and source-cache poses. Poses remain advisory,
not fresh pose authority. A fresh V15 semantic predictor/base is absent;
`PROGRAM_RESIDUAL_LAYERED` remains owed. This artifact is research-only and
moves no score or pointer.
