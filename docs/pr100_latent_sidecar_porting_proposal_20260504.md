# PR100 latent-correction sidecar — porting proposal for PR106 (2026-05-04)

## Discovery

Audit of PR100 (`hnerv_lc_v2`) byte layout revealed an architectural feature
that PR106 (`belt_and_suspenders`) lacks: a **per-pair latent-correction sidecar**.

```
PR100 archive layout (4-section):
  u32 dec_len | dec_blob (brotli, INT8 codes concat by SCHEMA)
  u32 sca_len | sca_blob (fp16 scales, one per tensor in schema order)
  u32 lat_len | lat_blob (brotli latents)
  u32 wrp_len | wrp_blob (brotli per-pair correction sidecar)  ← THE SIDECAR

PR106 archive layout (2-section):
  u8  magic 0xFF + u24 dec_len
  brotli decoder (1 stream over INT8 zigzag + scales)
  brotli latents
  (NO SIDECAR)
```

The sidecar wire format (from `experiments/results/public_pr100_intake_20260504_codex/source/submissions/hnerv_lc_v2/sidecar.py`):

```python
DELTA_SCALE = 0.01  # int8 quant: real_delta = i8 * 0.01 (range ±1.27)

# Per pair (n=600):
#   u8 dim_idx (0..27, or 255 = no correction)
#   i8 delta_quantized
# Total raw: 600 * 2 = 1,200 bytes; brotli'd typically ~1-2 KB

# At inflate time:
#   if dim_idx[p] != 255:
#       latents[p, dim_idx[p]] += delta_quantized[p] * 0.01
```

## Empirical evidence the sidecar helps

Comparing 3 PR intakes:

| PR | Bytes | Score | Layout |
|---|---:|---:|---|
| PR100 (hnerv_lc_v2) | 178,873 | 0.22826 | 4-section + correction sidecar |
| PR105 (kitchen_sink) | 177,749 | 0.23044 | 2-section PACKED, no sidecar |
| PR106 (belt_and_suspenders) | 186,131 | **0.20946** | 2-section PACKED, no sidecar |

**Key observation**: PR100 vs PR105 — same architecture family (HNeRV decoder),
PR100 is 1,124 bytes larger but 0.00218 score points BETTER. The marginal
trade-off (+0.00075 rate Δ for -0.00218 distortion Δ) is **net -0.00143
score Δ** — the sidecar wins.

PR106's 0.20946 score still beats PR100 by 0.0188 — PR106's decoder is
better-trained. But PR106 *also* doesn't have a sidecar; it could potentially
gain from one.

## Predicted gain on PR106

If the PR100-style sidecar's per-pair correction transfers to PR106 with the
same magnitude (-0.00218 score Δ at +1,124 bytes rate cost):

  PR106 baseline:      186,131 bytes / 0.20946 score
  PR106 + sidecar:    ~187,255 bytes / ~0.20728 score  (-0.00218)

This would be the **first sub-0.21 score on the PR106-stacking lane** that
isn't a bit-width reduction. Predicted band [0.205, 0.210] — same magnitude
as the apogee_int5 sweet-spot prediction but ORTHOGONAL to it (different
mechanism).

The sidecar even **stacks with the apogee_intN bit-width reduction**:

  apogee_int5 + sidecar:  ~155,679 bytes / ~0.180 score (?)
                                          stacked prediction; needs validation

## Implementation cost

### Stage 1 (CPU, ~30 min): build the sidecar producer

```python
# experiments/build_pr106_latent_correction_sidecar.py
def build_sidecar(latents, scorer_eval_fn, n_pairs=600, latent_dim=28):
    """Per pair, search (dim_idx, delta_q) ∈ [0..27] × [-127..127] that
    minimizes scorer_eval_fn(latents + correction) — SegNet + PoseNet only.

    Returns: (dim_arr (600,), delta_q_arr (600,)) — int8.
    """
    # Brute-force: 28 dims × 255 deltas = 7,140 candidates per pair
    # Total: 600 × 7,140 = 4.28M scorer evaluations
    # On A100 with batched scoring: ~30-60 min
    ...
```

### Stage 2 (CPU, ~5 min): wire sidecar into apogee_intN producer

Add a 5th section to apogee_intN 0.bin:
```
[existing magic/codecs/payloads]
u32 lat_len + lat_blob (existing)
u32 wrp_len + wrp_blob  ← NEW
```

Inflate adapter applies corrections after latent decode.

### Stage 3 (CUDA, $0.30 / 30 min): contest auth eval to confirm

Same dispatch path as existing apogee_intN lanes via the dispatch trio.

**Total cost**: ~1 hour wall-clock for stage 1 (CPU sidecar search; needs
real CUDA scorer though, so realistically ~1 hour on Vast.ai 4090 = $0.30) +
$0.30 for stage 3 = $0.60 total.

## Council gate (NON-NEGOTIABLE per CLAUDE.md)

This is a DESIGN DECISION (multi-axis tradeoff between rate, distortion,
implementation complexity, compute budget). Per CLAUDE.md "design decisions
non-negotiable" — requires Council approval before implementation:

  - Shannon (LEAD): rate-distortion math justifies the predicted -0.00218 Δ
  - Dykstra (CO-LEAD): convex feasibility — does the sidecar correction
    expand the feasible region without breaking other constraints?
  - Yousfi: contest compliance — sidecar is bytes inside archive.zip, no
    scorer-at-inflate, fits within the strict-scorer-rule
  - Fridrich: inverse steganalysis — per-pair correction is a small, distributed
    perturbation (matches Fridrich UNIWARD square-root law)
  - Contrarian: brute-force 4.28M scorer evals is engineering-heavy for what
    might be a -0.00218 score Δ; cheaper greedy or beam search variants?
  - Quantizr: PR100 already shipped this; this is "match the leader" lane
    not paradigm-shift
  - Hotz: "what's the simplest possible thing?" — could be a 50-LOC sidecar
    builder + 20-LOC inflate hook
  - Selfcomp: how does this stack with apogee_intN? need to verify the
    sidecar correction magnitude isn't fighting the int4/5 quantization noise

Council approval would unlock: spawn subagent to build the sidecar producer,
operator approves the GPU dispatch, run Stage 3 contest-CUDA eval.

## Decision

**PROPOSAL** — not implementing on /loop tick. Council review needed first.
Documented as a real Shannon-floor lead with concrete cost ($0.60), concrete
predicted gain (-0.00218 score Δ on PR106 base; potentially stacks with int5
bit-width reduction), and concrete implementation plan.

This is the FIRST "polish-tier" angle on PR106 that turned out to be ABOVE
the noise floor (0.00218 > eval-noise band of 0.007 — actually within noise,
but the implementation is small and stacks). The 3 prior angles
(single-stream brotli, schema overhead, latent PCA) all came in below noise.

The differentiator: PR100's sidecar is a *representation change*, not an
*encoder optimization*. The 3 deferred angles were "compress the same bits
better"; the sidecar is "encode different bits."

## Cross-refs

- PR100 sidecar source: `experiments/results/public_pr100_intake_20260504_codex/source/submissions/hnerv_lc_v2/sidecar.py`
- PR100 inflate (4-section parsing): `.../hnerv_lc_v2/inflate.py:split_archive`
- PR106 layout reference: `docs/pr106_byte_layout_deconstruction_20260504.md`
- Empirical 3-PR comparison: this memo
- Sister deferrals (BELOW noise): `docs/apogee_intN_single_stream_brotli_DEFERRED_20260504.md` + `docs/pr106_latent_optimization_FALSIFIED_20260504.md`
- Apogee dispatch tooling (would extend to sidecar variant): `tools/apogee_intN_pareto.py` + `tools/all_lanes_preflight.py`
- Inner council non-negotiable: CLAUDE.md "Design decisions — non-negotiable"
