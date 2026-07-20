# Full resize-kernel compiler build spec (2026-07-20)

Authority: delegated arm `codex_delegate:null_compiler_fix:20260720T160856Z`.
Lane: `lane_null_compiler_full_kernel_20260720` (L0, research-only until MAIN
adoption). This spec implements the already-requested design; it does not grant
launch, score, pointer, or promotion authority.

## Objective

Extend `resize_null_preimage_compiler.v1` with a reusable implicit compiler for
the complete real-linear kernel of the shared scorer resize
`A(X) = A_h X A_w^T`, camera `(874,1164)` to scorer `(384,512)`, and measure the
bounded uint8/coder subset on the SHA-pinned #49 source fixture. Preserve the
old zero-weight fill as the baseline and compose with its real Brotli/LZMA
admission helpers.

## Exact structure (implementation contract)

The canonical half-pixel resize has disjoint two-tap supports on both axes.
Each axis matrix has full row rank. Let

- `Q_h = A_h^T (A_h A_h^T)^-1 A_h`, `P_h = I-Q_h`;
- `Q_w = A_w^T (A_w A_w^T)^-1 A_w`, `P_w = I-Q_w`.

Because `A = A_w tensor A_h` under column-vectorization,

`P_ker(X) = X - Q_h X Q_w = P_h X + Q_h X P_w`.

The two terms are an orthogonal direct sum. An implicit, nonredundant
parameterization is

`K(U,V) = N_h U + B_h V N_w^T`,

where `N_h`/`N_w` span the per-axis kernels and `B_h=A_h^T` spans the height
row-space. Shapes are `U:(H-h,W)` and `V:(h,W-w)`, so the parameter count is

`(874-384)*1164 + 384*(1164-512) = 570,360 + 250,368 = 820,728`.

For a two-tap rational row with exact integer numerators `(a,b)`, the local
axis-kernel atom is the primitive integer vector `(b/g,-a/g)`,
`g=gcd(a,b)`. Unowned input indices contribute coordinate atoms. The projector
uses normalized fp64/fp32 weights, while exact integer verification uses the
existing `DisjointResizeOperator` numerator path; never infer integer exactness
by rounding a float projection.

Exact contest counts to re-derive in code/tests, not hardcode as pass criteria:

- domain `1,017,336`, rank `196,608`, full nullity `820,728`
  (`80.6742315223%`);
- old axis-aligned zero-weight mask `230,904` (`22.6969260893%`);
- closure gap `589,824` dimensions (`57.9773054330` percentage points);
- per-axis nullities `490/874` and `652/1164`.

## Owned files and boundaries

Create:

- `src/tac/optimization/resize_full_kernel.py`: exact rational support metadata,
  implicit per-axis basis, `Q`/`P` application, full projector,
  parameterization/synthesis, coverage receipt, bounded-uint8 basis/cell
  diagnostics, and an exact-coder-admitted full-kernel fill that composes with
  `resize_null_preimage`.
- `src/tac/tests/test_resize_full_kernel.py`: behavioral projector, synthesis,
  rational, fp32, dimension, uint8 and coder-admission tests on small geometry
  plus contest-count tests.
- `tools/measure_resize_full_kernel.py`: deterministic local measurement tool
  with explicit input path + expected SHA-256; emit JSON receipt atomically.
- `src/tac/canonical_equations/resize_full_kernel_structure_20260720.py` and a
  focused equation test; export through canonical-equations `__init__`.

Create after measurement:

- `.omx/research/null_compiler_full_kernel_<UTC>.json` receipt;
- `.omx/research/null_compiler_full_kernel_<UTC>.md` memo;
- `.omx/research/null_compiler_full_kernel_DAG_FEED_20260720.md`.

Update only the lane/equation registries through their canonical helpers. Do
not edit scorer, trainer, upstream, pointer, dispatch, run, or submission state.

## Bounded uint8 and min-description measurement

Fixture: reuse #49 source video `0.mkv` from the SSD custody copy, require its
actual SHA-256 before decode, and record decoded-frame SHA-256. Use a bounded
number of frames (default one) so this remains a $0 local measurement.

For each exact disjoint 2x2 cell and channel:

1. derive the three tensor-kernel integer atoms from the row/column primitive
   null vectors;
2. report whether either sign of each primitive atom fits around the fixture
   byte vector; report ranks/fractions by family and total. This is explicitly
   the *canonical primitive-basis bounded-reachability* measurement. If it is
   only a lower bound on the whole bounded lattice intersection, label it as
   such rather than promoting it to equality;
3. construct full-kernel candidates by solving the exact numerator equation
   against deterministic compressibility preferences (constant, horizontal,
   vertical, neighbor/local mean as practical), using the existing bounded
   integer cell solver. Every candidate must be uint8 and exact under
   `apply_numerators`; failed/budget cells fail closed to the source/old-mask
   value;
4. compare original, old #49 measured-best mask fill, and every full-kernel
   candidate with the real existing Brotli-q11 and LZMA coders. Admit/report the
   full-kernel winner only if it is no larger than the old-mask baseline.

The receipt must decompose: per-axis and total dimensions/fractions; old/full
coverage; primitive uint8 feasibility/rank by left-height vs right-width tensor
family and zero-weight class; exact vs fallback cell counts; original/old/full
bytes for each coder; deltas and percentages; projection/numerator residuals;
fixture/tool/source hashes; hardware/axis; `score_claim=false`,
`promotion_eligible=false`, pointer `0.19108` unchanged. Do not call the bounded
heuristic a global MDL optimum.

## Consumers

The memo/DAG must name exact callable routing:

- r2b sparse target-selection stream: project proposed camera residuals through
  `P_ker` and derive/drop the free component before charging sparse bytes;
- R1 `d_B` preimage cells: synthesize cell-local equal-numerator alternatives
  from the implicit tensor coordinates, then let hard decoded evidence choose;
- #401 blind fill: replace the mask-only fill with coder admission over the
  full exact affine cell while retaining the old mask as fallback/control.

## Acceptance checks

- `PYTHONPATH=src:upstream python3 -m pytest src/tac/tests/test_resize_full_kernel.py src/tac/canonical_equations/tests/test_resize_full_kernel_structure_20260720.py -q`
- existing `src/tac/tests/test_resize_null_preimage.py` remains green;
- ruff/py_compile on new/edited Python files;
- real measurement completes from SHA-pinned SSD fixture and receipt verifies
  exact numerator equality for every admitted cell;
- two clean `review_tracker` passes for every changed `.py` after the last fix;
- serializer commit with base/post content SHA-256; clean branch afterward.

One bounded self-review attacks: tensor-index orientation, rank/dimension
double-counting, fp32-vs-rational authority, hidden float-round exactness,
bounded-lattice overclaim, coder baseline fairness, and consumer callability.
MAIN landing review remains mandatory even if branch-local checks are clean.
