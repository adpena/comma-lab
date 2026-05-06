# Analysis And Optimization Package Map

The public `tac` code should read like a small archive compiler, not a pile of
contest scripts. Use these package boundaries when adding or recovering code.

## `src/tac/analysis/`

Measured or derived telemetry about the contest video, scorers, public
archives, and candidate outputs.

Allowed:

- pair difficulty maps;
- hard-pair and component telemetry;
- LA-POSE-lite or openpilot/camera-derived feature records;
- byte anatomy and archive member profiles;
- foveation, motion, class, and residual opportunity manifests.

Not allowed:

- remote dispatch;
- score claims;
- hidden sidecars;
- archive mutation without a builder contract.

Current examples:

- `tac.analysis.lapose_lite_inputs`
- `tac.analysis.lapose_motion_evidence`
- `tac.analysis.lapose_motion_atoms`

Lane W pair weights belong here conceptually. They are general CUDA scorer
telemetry over all 600 pairs, not Lane-W-private state.

## `src/tac/optimization/`

Planning and allocation over charged atoms.

Allowed:

- meta-Lagrangian score deltas;
- water-fill ledgers;
- atom ranking;
- stack-interaction planning;
- policy JSON generation.

Not allowed:

- score promotion without exact CUDA;
- archive-byte mutation without a deterministic builder;
- runtime side effects.

Current example:

- `tac.optimization.meta_lagrangian_allocator`

## `tools/`

Thin CLIs over canonical `tac` APIs. Tools may read/write JSON artifacts, but
their business logic should live in `src/tac`.

Compatibility wrappers at old import paths are acceptable while scripts are
migrated, but new code should import the canonical package.
