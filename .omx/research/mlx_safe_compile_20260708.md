# SAFE-COMPILE: determinism-first mx.compile layer (#252) — 2026-07-08  [no-triality]

Operator 2026-07-08: *"Can we engineer our own mx.compile using fixed point or other techniques
for fidelity and engineering determinism"*.

**MEANS, not ends.** Nothing here moves the pointer (0.19110). A certified region is bit-identical
to the uncompiled path — SPEED only, score-neutral by construction ([macOS-MLX research-signal];
a bit-identity certificate is a valid on-device FACT, NEVER a score). Only a byte-closed n600 exact
row from `upstream/evaluate.py` moves the pointer.

## STORES CONSULTED
`.omx/research/v7_compute_exploitation_audit_20260708.md` (lever #3: mx.compile MEASURED-EXCLUDED —
fp-contraction/FMA flips the uint8-STE d_seg argmax knife-edge, fwd Δ~4.8e-3 MEASURED 2026-07-03) ·
`tools/mlx_gpu_determinism_probe.py` (#348 cross-process instrument — the localization table: all
core forwards + reductions cross-process bit-identical, only dup-index scatter-add + its VJP diverge;
fused-R = fixed-order VJP is the product) · `src/tac/local_acceleration/deterministic_primitives.py`
(#348 fixed-order `kahan_compensated_sum` reference) · `src/tac/local_acceleration/mlx_compile_step.py`
(prior whole-loss compile with `atol=1e-6` — this module is STRICTER: per-region max|Δ|==0) ·
`experiments/train_levelset_witness_realized_through_R_mlx.py` (hot-loop elementwise candidates:
hosc activation, FiLM affine, compose_rgb tail, CE loss) · DSL `Lever`/`lever_registry` ·
deferral ledger D17.

## THE IDEA
mx.compile fuses MLX's many small launches into fewer kernels (kills lazy-eval overhead) BUT its
fp-contraction can change a single output bit and thus d_seg. The answer is not "never compile" — it
is "compile ONLY subgraphs where fusion provably cannot change one bit, and PROVE it per region."

- **Partitioner** (`classify_op_kinds`): falling-rule `contraction > reduction > unknown-op >
  elementwise`. Only SAFE_ELEMENTWISE is compile-eligible; a `mul+add` chain is still SAFE but flagged
  `fma_contraction_possible` (the highest-risk-of-empirical-failure class). PRACTICAL allowlist API
  (we nominate regions); the automatic graph-walker is a **v2 scope note** (see below).
- **Certification harness** (`certify_region`): (a) bit-equality vs uncompiled on n≥32 seeded real
  inputs, `max|Δ|` must be EXACTLY 0; (b) cross-process determinism (N separate processes rebuild +
  compile + hash — the #348 pattern, via `python -m tac.mlx_safe_compile --certify-child`); (c)
  measured wall-clock. Verdict CERTIFIED iff (a)∧(b). A FAILED region records the first diverging
  output (the fp-contraction fingerprint) — op-level attribution is v2.
- **Fixed-point option**: a reduction region is NOT compile-eligible; `certify_fixed_point_reduction`
  routes it to `fixed_order_reduce_mlx` (explicit left-fold — the MLX analogue of #348 kahan) with a
  determinism-only certificate. (Note: native `mx.sum`/`logsumexp` are ALREADY cross-process
  bit-identical per the #348 probe, so this is a fallback for regions that must pin the order.)
- **Manifest** (`.omx/state/mlx_safe_compile_manifest.json`, schema `mlx_safe_compile.manifest.v1`):
  the machine-readable evidence — `{region_id · verdict · bit_equal · max_abs_delta · determinism
  n/n · speedup · input_coverage · fma_flag · first_divergence}`. A region activates ONLY via a
  CERTIFIED row (no vibes).
- **Activation** (`safe_compile` / `resolve_enabled_regions` / `maybe_safe_compile`): default OFF =>
  fn returned unchanged (byte-identical). Fail-closed: an uncertified/absent id NEVER compiles.

## MEASURED REGION TABLE (this M5 Max GPU, n_inputs=32, N=5 cross-process)
| region | class | verdict | bit_equal (max\|Δ\|) | determinism | speedup | fma |
|---|---|---|---|---|---|---|
| `hosc_activation` tanh(β·sin(ω·u)) | safe_elementwise | **CERTIFIED** | True (0.0) | 5/5 | **1.41×** | no |
| `sigmoid_scale` sigmoid(base+tex)·255 | safe_elementwise | **CERTIFIED** | True (0.0) | 5/5 | ~1.0× | yes |
| `film_modulate` h·scale+shift | safe_elementwise | **CERTIFIED** | True (0.0) | 5/5 | ~1.03× | yes |
| `ce_reduction` logsumexp+sum+mean | unsafe_reduction | CERTIFIED (fixed-point) | True (0.0, fixed-order) | 5/5 | 1.0× | — |

**Honest reading of the measurement.** All 4 regions certified bit-identical + cross-process
deterministic on THIS GPU — INCLUDING the FMA-eligible `film_modulate`/`sigmoid_scale`. That means
for these SIMPLE elementwise patterns mx.compile did NOT contract the FMA in a bit-changing way on
this chip. This does NOT contradict the R-operator exclusion: the R exclusion is specific to the
resize/uint8-STE knife-edge (v7 audit lever #3, MEASURED separately). The harness is the arbiter, and
it reports what it MEASURED per region — the certificate is per-chip and re-run per host (fail-closed
if a future host contracts an FMA, exactly like the `--fused-r-kernel` per-chip parity gate). The
speedups here are per-region MICRO-benchmarks on synthetic-shaped tensors (256×96) — the WHOLE-STEP
delta needs the real trainer and is a v7.1 measurement (below). `hosc_activation`'s clean 1.41× is the
strongest single-region signal (unary+mul chain, zero FMA risk).

## WHAT LANDED
- `src/tac/mlx_safe_compile.py` — partitioner + certification harness + fixed-point option + manifest
  + activation API + 4 canonical registered regions + `--certify` / `--certify-child` CLI.
- `src/tac/tests/test_mlx_safe_compile.py` — 22 tests (classification falling-rule · verdict/cert
  value objects · manifest save/load round-trip · default-OFF byte-identity · fail-closed resolution ·
  known-unsafe reduction FAILS `certify_region` · MLX-gated empirical bit-equality + fixed-order sum).
- `tools/mlx_gpu_determinism_probe.py` — `--safe-compile [--safe-compile-regions …] [--safe-compile-out …]`
  cell (delegates to the harness; the probe stays the single certification instrument).
- `experiments/train_levelset_witness_realized_through_R_mlx.py` — `--safe-compile-regions`
  (default `none` => byte-identical) + `--safe-compile-manifest`; startup resolution logs the enabled
  CERTIFIED regions. **NO hot-loop call site rewired** — the per-region `maybe_safe_compile` wraps are
  the v7.1 activation (needs the v7 baseline as the A/B comparator; sister of D15 micro-batch).
- `src/tac/witness_dsl/curriculum_dsl.py` — `SafeCompileRegions(...)` `Lever` factory (the DSL now
  HOLDS `--safe-compile-regions`; registry coverage confirmed).

## WHAT v7.1 GETS
The certified-region manifest is the evidence gate (D17). v7.1 arms `--safe-compile-regions
all-certified` ALONGSIDE the v7 baseline as the A/B comparator, wires the `maybe_safe_compile` call
sites at the nominated hot-loop regions (activation first — the 1.41× region), and measures the
WHOLE-STEP wall-clock delta at the real n600 config. Because every activated region is CERTIFIED
bit-identical, the A/B is expected d_seg-NEUTRAL by construction (the check is a formality, not a
score risk) — unlike D15 micro-batch which is trajectory-affecting.

## v2 AUTO-PARTITIONER SCOPE NOTE
This layer is a PRACTICAL allowlist: WE nominate regions and the harness certifies them. A v2 could
walk an MLX graph automatically (trace the lazy graph, cut it at every reduction/matmul/conv/scatter
boundary, and hand each maximal elementwise island to the certifier). That needs MLX graph
introspection (op-kind per node) which the current public API does not cleanly expose; the allowlist
+ per-region certificate is the honest, shippable v1. The op-level "first-diverging-op" attribution
(vs the current first-diverging-OUTPUT) is the same v2 dependency.
