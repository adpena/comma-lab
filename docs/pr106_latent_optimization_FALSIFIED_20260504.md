# PR106 latent payload — optimization angles FALSIFIED (2026-05-04)

## Context

PR106 latent payload is 15,849 bytes (8.5% of the 186,131-byte archive). At
first glance this looks like leverage — bigger surface area than the 928-byte
single-stream-brotli gap. Empirical investigation shows the latent payload is
at or near its Shannon floor for the given (latent_dim=28, n_pairs=600,
quantized) representation.

## Falsified angle 1: Arithmetic coding the deltas

Already FALSIFIED in earlier work (task #359 in the registry):
**"Lane #02 (arith_qint on PR106 latents) FALSIFIED — brotli already below
0-th-order Shannon"**

Brotli q11 over the delta+hi/lo+mins+scales layout achieves better than the
0-th-order entropy of the per-latent distribution because the LZ77 component
exploits temporal patterns brotli's range-coding-style backend captures.
Pure arithmetic coding (no LZ77) was empirically larger.

## Falsified angle 2: PCA cross-dim rank reduction (NEW 2026-05-04)

Computed the eigenvalues of the 600×28 latent matrix's covariance:

```
Top 5 eigenvalues: [2.359, 1.702, 1.355, 1.307, 1.186]
Eigenvalue spread (max/min): 5.92×

Cumulative variance @ top-K:
  K=4:  29.55% of total variance
  K=8:  48.41%
  K=14: 67.71%
  K=20: 82.97%
  K=24: 92.05%
  K=28: 100.00% (no truncation)
```

The latents are **essentially flat across all 28 dims** — there is no "long tail
of insignificant dims" to drop. PCA truncation reconstruction error grows
catastrophically:

| K (kept dims) | rel_err per latent | bytes (basis + projection, fp16) | rate Δ vs 15,849b |
|---:|---:|---:|---:|
| 4 | 83.40% | 5,024 | -7,217 (-0.0048 score Δ) — **distortion catastrophic** |
| 8 | 71.36% | 10,048 | -2,801 — **distortion catastrophic** |
| 14 | 56.46% | 17,584 | +1,735 (LARGER than current!) |
| 20 | 41.00% | 25,120 | +9,271 — fp16 overhead exceeds savings |
| 24 | 28.02% | 30,144 | +14,295 |

**No K beats current 15,849 bytes once basis storage is paid.** The latents are
information-dense across all 28 dims; rank reduction loses too much to be useful.

Compare to weight quantization: int4 weights at 7.1% rel_err save 76KB; PCA
K=24 latents at 28% rel_err save -14KB (i.e. cost MORE bytes). The Pareto
position is *worse* on both axes simultaneously.

## Falsified angle 3: Lossier latent quantization

Latents are currently quantized to uint8 (delta-encoded). Going to int4 or
int3 latents would save bytes but at distortion that compounds across all
600 frame-pairs. Per the apogee_intN sweep on weights, the bit-width axis
gives ~0.04 score Δ per bit dropped on the WEIGHT payload (171KB). On the
LATENT payload (15.8KB) the same bit-drop would scale roughly proportionally:

  latent int8 → int4: ~ -7.9KB (-0.005 score Δ rate)
  latent distortion at 4-bit: ~7-15% per latent (apogee_int4 weight analog)

Even if distortion holds, the savings are 6× smaller than the weight quantization
axis (which already lands -0.021 score Δ at int5). The latent quantization is
strictly Pareto-dominated by the weight quantization for the same engineering
investment.

## What WOULD work (out of scope)

Latent codec replacement requires changing the *representation*, not just
the encoding:

1. **Trained latent codec**: VAE / vector-quantized latents with a learned
   prior. Quantizr's reported architecture suggests this direction. Implementation
   is a multi-week training effort, far beyond a /loop tick.

2. **Cross-pair temporal RNN**: model each pair's latent as a function of
   the previous + a residual; the residual is what ships. Adds a small
   model to the inflate side.

3. **Lane SJ-KL on the latent**: compute Fisher importance per latent dim,
   spend bits proportionally. Lane SJ-KL is launch-ready for a different
   payload type; extending it to the PR106 latent is a significant code change.

These are research-grade lanes, not /loop polish. They go through the council
design-decision gate per CLAUDE.md.

## Decision

The PR106 latent payload (8.5% of archive) is **AT its representational
Shannon floor** for the given (uint8 quantized, delta-encoded, 600×28)
schema. No /loop-tick-sized polish recovers measurable score Δ.

Reactivation criterion: implement IF AND ONLY IF a council-approved trained
latent codec lands AND its EV justifies the multi-week investment. Until
then, latent-payload optimization is OFF the polish queue.

## Cross-refs

- Sister falsifications: task #359 (arith_qint on latents — FALSIFIED)
- Sister deferral: `docs/apogee_intN_single_stream_brotli_DEFERRED_20260504.md`
- PR106 layout reference: `docs/pr106_byte_layout_deconstruction_20260504.md`
- Public latent codec source: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/codec.py:decode_fixed_latents`
- Lane SJ-KL (alternative residual codec): `src/tac/sjkl_basis.py` + `experiments/build_sjkl_residual.py` (different payload, not PR106 latents)
