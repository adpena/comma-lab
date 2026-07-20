# DAG FEED — full shared-resize null compiler (2026-07-20)

`research_only=true`; authority
`codex_delegate:null_compiler_fix:20260720T160856Z`; lane
`lane_null_compiler_full_kernel_20260720`; no score/pointer/promotion authority.

## Node and law

- Node: `resize_null_preimage_full_kernel.v1`
- Equation: `separable_resize_full_kernel_direct_sum_v1`
- Producer:
  `tac.optimization.resize_full_kernel.FullResizeKernel`
- Structural output: complete implicit real-linear resize kernel,
  `820,728 / 1,017,336 = 80.6742315223%` dimensions per channel.
- Exact discrete guard: source/candidate integer resize numerators must match.
- Bounded search guard: report canonical primitive-basis reachability only as a
  lower bound; never equate it to the complete uint8 lattice intersection.
- Admission guard: compare real Brotli/LZMA bytes and retain the legacy #49 mask
  on ties or regressions.

## Governed edges

1. `r2b sparse target proposal`
   -> `FullResizeKernel.project_kernel`
   -> free/charged decomposition
   -> exact receiver/coder admission.
2. `R1 d_B cell target`
   -> `FullResizeKernel.synthesize` or bounded exact cell solve
   -> integer-numerator equality
   -> hard decoded evidence.
3. `#401 blind fill`
   -> `FullResizeKernel.compile_min_description_preimage`
   -> old-mask/full-kernel candidates
   -> coder argmin with old-mask tie preference.

## Current empirical anchor

On one SHA-pinned #49 frame, the canonical three-channel primitive-basis
reachability lower bound is `34.1931390993%`. The tested constant-preference
candidate was `+512,550 B` versus the old mask under Brotli and `+546,524 B`
under LZMA, so the old control was selected. This is a formulation-scoped
negative, not a family verdict. Receipt:
`.omx/research/null_compiler_full_kernel_20260720T163500Z.json`.

## Activation gate

Remain research-only until MAIN independently reviews the exactness/counting
proof and a consumer demonstrates receiver-closed counted-byte improvement on
exact archive bytes. The current contest-CPU pointer remains unmoved.
