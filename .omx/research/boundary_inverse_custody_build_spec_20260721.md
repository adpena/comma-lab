# Boundary inverse custody: implementation specification

Date: 2026-07-21
Lane: `boundary_inverse_custody_20260721`
Authority: delegated local build and `[macOS-CPU advisory]` mask-fidelity measure only

## Objective

Close `BLOCKED_TARGET_BOUNDARY_INVERSE_CUSTODY` by mapping the true Lane-mask
residual of the decoded 41,303-byte coherent-slot polynomial chart into a
finite set of genuine literal-polar-curvelet and compact-shearlet atoms.  The
landing must produce two independently decoded treatments:

1. a generic image-coordinate 2-D sparse control; and
2. an arc-length/dash-phase-conditioned sparse arm whose coefficient rows and
   coordinates are explicitly indexed by decoded chart phase.

The measured output is mask precision/recall/F1 only.  It is not through-R,
not `d_seg`, not a score, and cannot move the `0.19108 [contest-CPU]` pointer.

## Owned surfaces

- New `src/tac/optimization/boundary_inverse_custody.py`.
- New `src/tac/tests/test_boundary_inverse_custody.py`.
- Extend `tools/measure_s2_lane_mask_curve.py`; do not fork the harness.
- New dated receipt, findings memo, and DAG feed under `.omx/research/`.
- Lane-registration/audit rows for `boundary_inverse_custody_20260721`.

Do not edit the S2 partition seed, scorer, frontier pointer, reports, sibling
worktrees, source cache, `CLAUDE.md`, or `AGENTS.md`.

## Finite inverse and treatment contract

- The dictionary is the immutable 80-column `literal_polar_curvelet` program
  plus a fixed, seed-free `CompactShearletConfig`.  Atom construction calls the
  #502 implementations directly; no Fourier substitute, relabelled spatial
  window, continuum claim, or learned frame parameter is permitted.
- The solver is deterministic correlation screening followed by ridge-stable
  least squares on the screened finite columns.  Each prefix is refit and then
  quantized to signed int8.  The receipt records dictionary width, family atom
  counts, selected atom IDs, nonzero counts, quantization step, and thresholds.
- Generic control evaluates atoms at normalized image `(x,y)` coordinates
  inside a decoded-chart corridor.
- Phase treatment computes polynomial-centerline arc length with fixed
  Gauss-Legendre quadrature, anchors it at the decoded dash phase, assigns a
  finite phase bin, and evaluates atoms at `(normal coordinate, local phase)`.
  Only decoded dashed lines participate.  Coefficients have shape
  `[phase_bin, dictionary_atom]`, making phase conditioning structural rather
  than a label attached to a generic 2-D control.
- The correction sum has a counted positive/negative threshold.  Positive
  excursions add Lane pixels, negative excursions remove Lane pixels, and the
  dead zone leaves the decoded polynomial chart unchanged.

## Counted state, parseback, and rule 118

- COUNTED: coordinate mode, phase-bin count, quantization step, correction
  threshold, and the dense signed-int8 coefficient tensor.  Zero/nonzero
  positions encode atom selection.
- The qint tensor is encoded by the repository-owned #557 left/up
  sign-magnitude context arithmetic codec.  Container headers and model bytes
  are counted.  Decode must recover exact shape, dtype, values, and a
  byte-identical re-encode; malformed, truncated, and trailing inputs refuse.
- FREE INTERPRETER: literal curvelet construction, compact-shearlet
  construction, chart-corridor/arc-length coordinate construction, sparse
  solve replay, and correction rendering.  These algorithms are generic and
  must decode a different video's selected atoms without access to truth.
- The n600 cache is read-only and memory-mapped.  Stage and receipt writes are
  atomic.  A stage directory plus `--resume` preserves chart, sample/solve,
  and evaluation milestones; no source custody is deleted.

## Measurement and rate gate

- Reproduce the decoded dash and continuous chart rows before composing any
  sidecar.  Sweep bounded atom prefixes for both treatment arms and record
  exact sidecar bytes/SHA-256/parseback, atom count, and aggregate/stratum F1.
- Compose every row as `41,303 chart bytes + sidecar bytes`; do not conceal
  container overhead or substitute compressed estimates for exact sidecar
  bytes.
- For every changed pixel and stratum record beneficial flips, harmful flips,
  remaining false negatives, and remaining false positives (the honest
  eat-the-flip remainder).
- Consume `realization_breakeven_bytes_v1` only in its valid domain.  Since this
  task has no through-R realized score recovery, report the sidecar's required
  recovery `sidecar_bytes * 25 / 37,545,489` and
  `waterfill_status=FORMALIZATION_PENDING`; never convert mask-F1 to score.

## Verification and landing

- Tests cover genuine family composition, generic/phase coordinate separation,
  deterministic sparse selection, coefficient quantization, exact context
  arithmetic parseback, malformed/trailing refusal, another-video decode, flip
  accounting, and no-score authority labels.
- Run targeted pytest with warnings as errors, Ruff on changed Python, compile,
  `git diff --check`, and deterministic repeat checks.
- Run two clean `review_tracker` passes for each new/changed Python file, then
  commit with `tools/subagent_commit_serializer.py --patch-file` and exact
  expected-content SHA-256.  Final handoff is branch-only and requires fresh
  MAIN landing review before merge or promotion.
