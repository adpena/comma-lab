# Pythagorean exact arithmetic → MLX cross-process bit identity (2026-07-13)

**Lead verdict:** **UNMEASURED-BLOCKED, not a claimed L70 break.** The real-op probe is built for the
384→874 bicubic render-R transpose accumulation, but this execution surface completed **MEASURED
0/10** requested MLX-GPU children because Metal device creation was refused. The integer formulation
therefore has no cross-process verdict. The pure NumPy authority side is **MEASURED green**: Q15/int32
is overflow-safe by **DERIVED 192.428×** headroom and differs from NumPy-fp32 by **MEASURED
0.007968902587890625 max-abs**, inside the **DERIVED 0.011561438380080984** quantization-plus-fp32
bound. This is not evidence that MLX integer atomics schedule deterministically.

**One-line requested taxonomy:** integer/Gaussian-integer lowering **BREAKS L70 for a real op =
UNMEASURED**, because the decisive Metal cell ran **MEASURED 0/10 processes**, not measured 0/10
divergences; NumPy parity is bounded-green; overall verdict is **ENVIRONMENT-BLOCKED / NO ADMISSIBLE
SELECTION YET** among `REAL-L70-LEVER`, `L70-DEEPER-THAN-FP-REORDER`, and `INERT-CURIO`.

This is **MEANS**, never the pointer. No training, evaluator, paid dispatch, live-run mutation,
archive mutation, score claim, or pointer movement occurred. Axis:
`[macOS-MLX research-signal; NumPy-fp32/int32 authority; non-promotable MEANS]`.

`verdict_scope`: `INSTANCE × current managed execution surface × MLX 0.31.2 Metal acquisition`.
It is not a negative verdict on integer lowering, MLX integer atomics, Gaussian-integer geometry,
render-R, or the witness paradigm.

## Stores consulted / recall before decide

- Canonical equation `decode_determinism_integer_arithmetic_v1`: same-archive cross-host decode
  discipline requires integer/fixed-point arithmetic; it has **MEASURED zero empirical anchors** in
  the registry, so it is a theoretical discipline, not prior proof of this MLX result.
- Memory `mlx_gpu_not_bit_identical_crossprocess_bitexact_proof_cpu_locked_20260702`: the original
  **MEASURED 28/28** cross-process witness-tensor divergence.
- Memory `mlx_gpu_determinism_localized_fused_r_cure_20260707`, canonical equation
  `mlx_gpu_crossprocess_nondeterminism_v1`, and
  `.omx/research/deterministic_gpu_accum_348_20260707.md`: L70 was later localized to duplicate-index
  atomic scatter-add. The real poison was the gather-based bicubic-UP backward. Fixed-order fused-R
  already made the full smoke witness **MEASURED 0/28 diverged tensors at N=10** and about **MEASURED
  8% faster**.
- `tac.optimization.pythagorean_crossterm` / FEED-pythag-crossterm: the Amari generalized-
  Pythagorean additivity diagnostic for reverse waterfill. It is mathematically distinct from
  elementary integer triples and is not evidence for the thesis tested here.
- Actual render-R source:
  `src/tac/local_acceleration/pr95_hnerv_mlx_training.py::{apply_contest_faithful_roundtrip_nhwc,
  resize_nhwc_align_corners_false,_resize_axis_nhwc,_resize_indices_weights}`.
- Vehicle authorities: CLAUDE.md, AGENTS.md, `docs/operating_manual_craft_handoff.md`, PROGRAM.md,
  SPEC v7.5 §8, and SPEC v8.

## Exact arithmetic thesis — what is derived and what is not

For coprime integers `m>n` of opposite parity,

```text
a = m²-n²,  b = 2mn,  c = m²+n²,
(m+ni)² = a+bi,
a²+b² = c².
```

Therefore

```text
Q(a,b,c) = (1/c) [[a,-b],[b,a]],
QᵀQ = ((a²+b²)/c²) I = I.
```

This is **DERIVED** exact rational orthogonality. Composition is exact before any chosen numeric
realization:

```text
(a₁+b₁i)(a₂+b₂i)
  = (a₁a₂-b₁b₂) + (a₁b₂+b₁a₂)i,
denominator = c₁c₂.
```

The composite numerator again has squared norm `(c₁c₂)²`. The result need not remain primitive,
and numerator/denominator growth requires gcd reduction and an overflow proof. Integer addition and
multiplication are reduction-order independent only when the chosen representation has no overflow,
saturation, undefined shift, or implementation-dependent division. “Integer” alone is not a proof.

The transfer to Pact is narrower than the analogy:

- **DERIVED:** a geometry operation deliberately represented by bounded integer numerators and a
  declared common scale can have exact, order-independent accumulation.
- **NOT DERIVED:** the real measured ego twist `ξ` has a Pythagorean angle. It is not a free angle;
  snapping it to one changes the geometry and requires a joint Seg/Pose receiver-through-R A/B.
- **NOT DERIVED:** arbitrary `sin`, `cos`, bicubic weights, MLP activations, or scorer arithmetic
  become exact merely because rational rotations exist.

## The smallest real L70 operation

The actual contest-faithful roundtrip is:

```text
render grid → bicubic up to 874×1164 → uint8 at camera resolution
            → bilinear down to 384×512 scorer resolution.
```

The prior #348 bisect is load-bearing:

- **MEASURED:** bicubic-UP backward is cross-process nondeterministic.
- **MEASURED:** bilinear-DOWN backward was deterministic in that probe (low fan-in).
- **MEASURED:** duplicate-index `.at[idx].add` is the failing atomic op class.
- **MEASURED:** fixed-order fused-R already cures the whole smoke graph.

The new probe therefore uses one actual vertical bicubic axis, `384→874`, with the exact
align-corners-false coordinate map and cubic coefficient `a=-0.75`. It carries **DERIVED 384
independent lanes** (one 128-wide RGB slice), **DERIVED four taps**, and **MEASURED/fixture maximum
fan-in 10**. Both cells use exactly the same destination indices and seeded integer cotangent:

1. current float32 duplicate-index MLX atomic scatter-add;
2. cubic weights rounded to Q15, products accumulated through duplicate-index MLX int32 atomic add.

The Q15 cell is a fixed-point lowering of the real resize adjoint, not a Pythagorean-angle toy. It
does not claim byte identity to the unmodified float operator; it has an explicit NumPy-fp32 error
contract.

## Decisive N=10 table

| cell | source | requested N | completed N | unique hashes | divergence | NumPy parity | verdict |
|---|---|---:|---:|---:|---:|---|---|
| prior reference-R bicubic-UP backward | #348 prior measurement | **MEASURED 10** | **MEASURED 10** | **MEASURED unique/process** | **MEASURED yes** | prior fused-R tests cover the authority comparison | float atomic op class is nondeterministic |
| fresh 384→874 float atomic adjoint | this probe | **DERIVED/contract 10** | **MEASURED 0** | not measured | not measured | NumPy reference materialized | `BLOCKED_CURRENT_ENVIRONMENT` |
| fresh 384→874 Q15/int32 atomic adjoint | this probe | **DERIVED/contract 10** | **MEASURED 0** | not measured | not measured | static NumPy contract green; MLX parity unmeasured | `BLOCKED_CURRENT_ENVIRONMENT` |
| NumPy Q15/int32 fixed-order reference | local CPU | N/A | **MEASURED 1 fixture + 8 tests** | deterministic-by-construction | N/A | exact int32 self-parity; bounded fp32 parity | implementation contract only |

The fresh receipt is
`.omx/research/pythagorean_exact_arithmetic_bitident_probe_20260713.json`. It records the exact
failure at float trial 0: `[metal::load_device] No Metal device available`. Direct `open`, native
Terminal AppleScript, and the computer-use Terminal surface were unavailable/refused before the
probe ran; no sandbox or app-policy bypass was attempted.

### NumPy-fp32 authority derivation and measurements

Let `q=2¹⁵`, `ŵ=round(qw)`, and let the seeded cotangent `g` be integer-valued in `[-127,127]`.
The integer output is

```text
I_i = Σ_(j,k : idx(j,k)=i) ŵ_jk g_j,
R̂_i = I_i/q.
```

The static fixture measured:

| quantity | result | label |
|---|---:|---|
| maximum sum of absolute integer contributions to one destination | `11,159,918` | **MEASURED** |
| int32 positive limit | `2,147,483,647` | **DEFINED** |
| overflow headroom | `192.42826398903648×` | **DERIVED** |
| maximum contributions to one destination | `10` | **MEASURED** |
| maximum bicubic row-sum error | `1.6689300537109375e-6` | **MEASURED** |
| two-order float32 accumulation bound | `0.0008129412114279191` | **DERIVED** |
| max `|R̂ - NumPy-fp32 reference|` | `0.007968902587890625` | **MEASURED** |
| RMSE versus NumPy-fp32 | `0.0019357680482321848` | **MEASURED** |
| quantization + two-order fp32 bound | `0.011561438380080984` | **DERIVED** |

The bound sums `|g|·|ŵ/q-w|` at every destination and adds a standard two-reduction-order
`γ_n` fp32 envelope using measured maximum fan-in. The measured error is inside the bound. This
proves that the chosen fixed-point representation is numerically well-formed for the fixture. It
does not prove the missing Metal scheduling result.

## Concrete integer-lowerable operations in render-R and the witness

| actual operation | integer/rational lowering | admission state |
|---|---|---|
| bicubic-UP transpose accumulation | rational coordinate map; fixed-point weights; bounded integer products and sums | **REAL and built in probe; Metal N=10 UNMEASURED** |
| bilinear camera→scorer resize | rational coordinate map; two-tap fixed-point weights | **DERIVED lowerable**, but prior float backward already deterministic, so not an L70 lever by itself |
| camera uint8 clamp/round and palette/flat paint | native integer values, clipping, table lookup | **INTEGER-NATIVE**; receiver bytes still require exact parse-back |
| mask/argmax labels, region counts, topology occupancy | integer labels/counts and exact comparisons | **INTEGER-NATIVE** except logits feeding argmax remain float today |
| coordinate grids | store integer pixel numerators and axis denominators; defer division | **DERIVED lowerable**; rational denominators must be custodied |
| repeated planar rotations | Gaussian-integer numerator composition plus denominator product | **DERIVED exact for admitted rational rotations only**; real `ξ` is not selectable |
| #348 duplicate-index accumulation | quantize operands, prove accumulator bounds, use integer atomic or fixed-order kernel | **TARGETED by this probe**; fixed-order fused-R is already the measured cure |
| Fourier-feature phase | integer phase index + fixed LUT or exact roots at special phases | **PARTLY lowerable**; arbitrary sine/cosine is not exact and LUT bytes/error must be charged |
| SDF polynomial/raster stencils | fixed-point coordinate/stencil algebra with interval bounds | **PLAUSIBLE/DERIVED per primitive**, not proven graph-wide |
| homography / ego-twist warp | rational matrix only if calibrated coefficients are rationalized under a receiver trust region | **NOT FREE**; current measured `ξ` must not be snapped silently |
| MLP GEMM + `sin/tanh/softmax` + scorer forwards | integer GEMM is possible, nonlinearities need explicit approximants; frozen scorer arithmetic is outside decode | **NOT integer-exact as the current graph** |

The important separation is receiver geometry versus training arithmetic. An integer-lowered
decoder can make identical bytes on every host. A fixed-point training surrogate may instead change
the optimization vector field and must pass NumPy-fp32 parity plus receiver-through-R debt gates.

## Divisibility by 60 — mathematically true, operationally inert here

For a primitive triple, exactly one leg is even and the even leg is divisible by four; exactly one
leg is divisible by three; modulo five at least one of `a,b,c` is divisible by five, and primitivity
prevents two from being divisible by five. Hence `2²·3·5=60 | abc`. This is **DERIVED elementary
number theory**.

It does not clear the actual resize denominators:

```text
874 = 2·19·23,  1164 = 2²·3·97,
384 = 2⁷·3,     512 = 2⁹.
```

The factors `19`, `23`, and `97` remain, and no legal choice of Pythagorean triple changes the
measured camera/scorer grid or the real ego twist. Divisibility by 60 can cheaply filter a synthetic
rational-angle catalogue or help choose an artificial common grid, but it buys **no concrete exact
alignment for the actual render-R ratios**. Verdict: **div-60 bookkeeping useful = NO for this
stack; beautiful but inert bookkeeping**.

## L70 and #356 consequence

Even a future positive integer N=10 result has narrow authority:

- It would show that bounded int32 atomic addition breaks the duplicate-index atomic-scatter L70
  instance for this real resize-adjoint operation.
- It would supply an alternative to fixed-order fused-R for exact accumulation, not a new cure for
  an unsolved render-R wall; #348 already measured fused-R green and faster.
- It would **not by itself unblock #356 whole-step megakernel fusion**. #356's negative is scoped to
  MLX fp32 whole-step fusion/reordering across the remaining graph. A megakernel becomes an
  integer-exact-proof candidate only after every reordered reduction/nonlinearity in its claimed
  domain has a bounded integer/fixed-point lowering and parity receipt.

Current consequence: **no #356 unblock note is registered**. Calling this an unblock with 0 Metal
children would be fake.

## Triality, research-only containment, and reactivation

- **DSL leg:** N/A. This is a compute/determinism probe, not a witness config lever. The existing
  `--fused-r-kernel` remains the live measured DSL cure.
- **Equation leg:** consumed `decode_determinism_integer_arithmetic_v1` and
  `mlx_gpu_crossprocess_nondeterminism_v1`. No new equation registered because the required empirical
  anchor did not run.
- **DAG leg:** this memo is the research-only FEED with the exact edge:
  `real bicubic adjoint → N=10 float/int Metal hash cells → parity gate → equation/#356 candidate`.
  The edge is blocked at the Metal hash cells.
- **Lane:** `pythagorean_exact_arithmetic_bitident`, L0, `research_only=true`.
- **Pointer delta:** none. No score claim.

Reactivation is one local, resumable command on a real Metal-capable shell:

```bash
bash tools/run_pythagorean_exact_arithmetic_bitident_host.command
```

The receipt is atomically updated after every child, so interruption loses at most one child. Admit
`REAL-L70-LEVER` only if the float cell diverges, the integer cell has one hash across all ten
processes (0/10 divergence), every integer child is bit-identical to NumPy-int32, and every
dequantized child stays within the derived NumPy-fp32 bound. If integer hashes diverge, select
`L70-DEEPER-THAN-FP-REORDER` with the narrow instance scope above. If the float cell is deterministic,
the fixture did not reproduce L70 and must be reformulated before any family conclusion.

## Reproduction / verification

```bash
.venv/bin/python tools/probe_pythagorean_exact_arithmetic_bitident.py --numpy-only
.venv/bin/python -m pytest -q src/tac/tests/test_probe_pythagorean_exact_arithmetic_bitident.py
.venv/bin/ruff check tools/probe_pythagorean_exact_arithmetic_bitident.py \
  src/tac/tests/test_probe_pythagorean_exact_arithmetic_bitident.py
bash tools/run_pythagorean_exact_arithmetic_bitident_host.command  # real Metal shell only
```
