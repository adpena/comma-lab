# 2026-05-07 Worker CATEGORICAL Typed Label Atoms

Scope: categorical/labelling/self-compression stack only. No PR103/PR106
runtime, dispatch state, LA-pose, HNeRV exact-eval runtime, entropy candidate,
or meta-Lagrangian files were touched.

## Patch

Added a deterministic typed label-atom table for the contest zero-based
comma10k class contract. The table is now embedded in:

- `build_categorical_class_codebook()`
- `build_categorical_label_prior_payload_manifest()`
- `build_categorical_charged_label_plan()`
- `build_categorical_compression_contract()`

The readiness audit now rejects a charged label-prior payload manifest whose
typed atom table drifts from the canonical helper. The archived runtime
consumer also verifies the typed atom table in both `class_codebook.json` and
`label_prior_payload_manifest.json`, then reports the consumed atom count in
the runtime execution proof summary.

## Research Basis

- comma10k README is the primary class/color source: road, lane markings,
  undrivable, movable, my car, with the sixth interior-only class excluded
  from contest SegNet channels.
- comma.ai's crowdsourced Segnet post confirms the five-label grouping and
  the pose-groundtruth importance of filtering movable and my-car classes.
- SPADE motivates spatially adaptive semantic conditioning from segmentation
  layouts.
- CLADE motivates class-adaptive conditioning as a lower-overhead semantic
  alternative when class identity carries most of the modulation value.

## Evidence

Verification command:

```text
.venv/bin/python -m pytest src/tac/tests/test_categorical_label_atoms.py src/tac/tests/test_categorical_compression_contract.py src/tac/tests/test_categorical_candidate_readiness.py src/tac/tests/test_build_categorical_candidate_fixture.py src/tac/tests/test_build_categorical_candidate_payload.py src/tac/tests/test_learnable_class_targets.py src/tac/tests/test_mask_grayscale_lut.py src/tac/tests/test_train_segmap_lct.py -q
```

Result: 83 passed, 1 expected duplicate-ZIP warning in
`test_audit_categorical_candidate_manifest_rejects_duplicate_archive_members`.

Evidence grade: `empirical` / `planning_manifest_audit`. This is not a score
claim and is not exact-eval dispatch readiness.
