# Integer-plane vehicle composition receipt

**Date:** 2026-07-19
**Lane:** `lane_integer_plane_vehicle_spec_20260719`
**Verdict:** **SPEC COMPOSED; NOT BUILD-READY; POINTER UNMOVED.**

## Outcome

The additive successor artifact is
`.omx/research/SPEC_v10_integer_plane_vehicle_20260719.md`. It composes only landed 2026-07-19
measurements and source-derived scorer laws. It produced no training run, new measurement,
candidate archive, score, paid dispatch, promotion, or pointer mutation. The preserved pointer is
`0.1910828242 [contest-CPU Linux x86_64]`.

The selected vehicle is a compact learned description of two independently addressable
`uint8 [384,512,3]` scorer planes, followed by two separate factor-2 camera-preimage solves. The
measured `3.775 min` projection remains explicitly scoped to one solved plane plus
`repeat-frame1`; the distinct-plane receiver is structural but has no full-n600 timing receipt.

## What the SPEC fixes in place

- Every learned, projected, solved, receiver, payload, and allocation stage names the exact
  `upstream/modules.py` / `upstream/frame_utils.py` frozen line it serves.
- Payload grammar is reduced by the intrinsic-complexity rule to a minimal header, generator
  tensor stream, independent pair/frame codes, and an optional PDW2 section only if consumed.
- Pose is represented on the exact 2×2 preprocessor lattice: four lossless luma phases and two
  chroma box averages. Within-block zero-mean chroma is a d_seg-only carrier; luma has no analogous
  slack. Pose-state bytes are priced at block granularity.
- Seg content is the frozen head’s coupled halfspace system: ten pairwise hyperplanes in the
  measured four-dimensional quotient, with candidate-local native-f32 margins and nonlinear
  spatial realization still owed. Fixed-capacity `U4`/pair-margin basis testing precedes any
  capacity increase; superseded directional-gain numbers are not consumed.
- The 24 immutable Seg/Pose VJP sidecars are encode-side proposal/trust-region custody, never
  payload or verdict. Class/pair, luma/chroma, skip/deep, range/ker, and Pose-block effects receive
  conditional parent-hash byte/distortion rows under one physical archive-rate authority.
- Training uses exact forward `uint8` rounding with saturation-aware STE, a differentiable frozen
  Seg margin/logit loss, a hard per-pair Pose gate at the derived `2.5e-4` crossover, explicit
  2×2 Pose-visible proximity, and exact native-f32 admission.
- Per-tensor int8+Brotli-q11 is the donor-measured baseline. Block-FP remains default OFF pending
  through-R admission. Joint size-in-loss remains specification-only and default OFF.
- PDW2 remains target-only unless a scorer-free deterministic pullback consumes it and passes a
  packet mutation/no-op test.
- The strict candidate byte ceiling and real-valued marginal indifference allowance are derived
  from the exact score law; one shared KKT governs non-additive pool×channel consumers.

## Owed-before-build

The SPEC records falsifiable charters C0–C11. The first unit is canonical-law reconciliation and a
full-n600, two-distinct-plane receiver timing receipt. Later gates cover the learned emitter,
PDW2 spatial causality, 2×2 Pose proximity, native-f32 preimage policy, minimal grammar bytes,
block-FP, default-OFF size-in-loss, receiver-closed R-D/KKT closure, vehicle-specific resumability,
and final separate contest CPU/CUDA custody. No charter grants launch authority by itself.

## Authority drift surfaced for MAIN

1. Pose registry row 745 has four anchors but the Python builder emits three; the registry row also
   retains a contradictory “intermediate unmeasured” domain field.
2. The f32 registry has three anchors but its Python builder emits two.
3. The reconciled predecessor predates the landed positive-band secant/KKT evidence.
4. Final #537 resume proof supersedes older tracked blocked-state prose, but the final receipts
   still require tracked custody.

This delegated arm records the drift; it does not silently alter shared canonical-law sources.

## Directive, review, and verification

The per-arm inbox directives at `2026-07-19T13:03:26Z` and `2026-07-19T13:05:03Z` were consumed.
Their frozen-source, intrinsic-complexity, exact 2×2 Pose, and weight-level
hyperplane/basis/channel/VJP bindings are in the SPEC body rather than a footnote.

Self-review stayed within the cap:

1. combined receiver/foundations, certificates/coding, and economics/laws review found scoped
   timing, coder-donor, KKT-label, Pose-count, preimage, and source-line issues; all were corrected;
2. fresh requirements/equation/source audit was clean on substantive content; and
3. the late weight-level directive triggered a frozen-head/PDW2 arithmetic, VJP-custody,
   channel-scope, and completeness audit; it corrected the `77%` and class-share scopes, made
   basis-before-capacity falsifiable, and rejected reuse of superseded directional-gain values.

Verification: `git diff --check` clean; exact byte-ceiling arithmetic independently re-derived;
required source/memo paths reopened; no code tests run because the deliverables are documentation
only.

## MAIN landing review required

MAIN must review the complete base-to-head branch diff, both canonical-law drifts, the
`repeat-frame1` versus distinct-plane timing boundary, the four-section minimal grammar,
2×2 Pose-null/chroma and lossless-luma conclusions, frozen-head halfspaces, fixed-capacity basis
A/B, immutable-VJP custody, channel-resolved KKT rows, conditional byte ceilings, and every
C0–C11 falsifier before merging. Merge does not authorize training, dispatch, evaluation, or a
pointer change.
