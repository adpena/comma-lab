# Reverse Engineering

This directory is the clean, canonical surface for contest-specific public
submission deconstruction.

It is for curated, small, reproducible material:

- public PR intake indexes and runbooks;
- archive/member byte-anatomy notes;
- payload grammar notes and adapter boundaries;
- reproducible commands for profiling or exact replay;
- small manifests that point to raw custody artifacts elsewhere.

It is not a raw dump directory. Do not put public PR clones, downloaded
archives, provider logs, Lightning/Modal/Vast transcripts, checkpoints, cache
trees, or large generated artifacts here. Keep those in ignored custody
locations such as `experiments/results/public_pr*_intake_*`, `.omx/state/`, or
external artifact storage, then link to them from a dated `.omx/research/`
ledger or a small manifest.

Reusable implementation belongs in `src/tac`, not here. This directory should
document how public submissions were inspected and how the resulting reusable
parsers, profilers, planners, or packers are invoked.

## Current Surfaces

- `pr95_hnerv/` documents the PR95-family HNeRV single-member archive grammar,
  exact-replay boundary, and residual-atom planning surface.
- `public_pr102_pr108_intake_20260508/` records the corrected PR102 HNeRV
  custody target, the stale wrong-asset gap, PR102 public CPU/CUDA drift, and
  PR108's non-frontier AV1/ROI/sharpening archive intake.
- `public_frontier/` contains curated source-sized public runtime references
  from public submission deconstruction and orphan-pyc recovery. It is forensic
  reference material, not score evidence or an active experiment output tree.
- `orphan_pyc_recovery_20260505_codex/` is a temporary recovery intake, not a
  canonical source tree. Promote useful code into `src/tac`, thin CLIs into
  `experiments/`, and findings into `.omx/research/` before deleting anything.

Audit the boundary with:

```bash
python tools/audit_reverse_engineering_tree.py \
  --repo-root . \
  --json-out .omx/research/artifacts/reverse_engineering/latest.json \
  --md-out .omx/research/artifacts/reverse_engineering/latest.md
```

For public release, also run the stricter release-manifest gate:

```bash
python tools/audit_reverse_engineering_tree.py \
  --repo-root . \
  --release-strict \
  --release-manifest .omx/research/reverse_engineering_release_manifest_20260505_codex.json \
  --summary
```

The release manifest is the explicit bridge from private recovery custody to a
public source tree: it states which classes are promoted as curated source,
which are summarized in ledgers, and which raw artifacts remain excluded.

## Evidence Boundary

Public submission body scores, README snippets, CPU/MPS outputs, and replay
through a public PR's own runtime are `external` evidence. They can guide
deconstruction and candidate design, but our score claims require exact CUDA
evaluation of our exact archive bytes through the canonical contest path.
