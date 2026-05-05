# Score-aware sidechannel paradigm — cross-PR pattern audit (2026-05-04)

## Pattern recognition

Auditing the 32 inflate.py files in `experiments/results/public_pr100_intake_20260504_codex/source/submissions/` revealed multiple submissions using the same underlying concept under different names:

**Score-aware sidechannel: a small parameter-sparse payload, trained against the scorer at compress time, that improves distortion without much rate cost.**

## Three independent implementations of the paradigm

### 1. PR100 hnerv_lc_v2 — per-pair latent correction

```python
# sidecar.py
DELTA_SCALE = 0.01
# Per pair (n=600):
#   u8 dim_idx (0..27, or 255 = no correction)
#   i8 delta_quantized
# Total: 1.2 KB raw, ~1-2 KB brotli'd

# At inflate:
#   if dim_idx[p] != 255:
#       latents[p, dim_idx[p]] += delta_q[p] * 0.01
```

Mechanism: chooses ONE latent dim to perturb per frame-pair (out of 28); perturbation magnitude is i8 quantized × 0.01. Search at compress time: brute-force (dim, delta) ∈ [0..27] × [-127..127] minimizing scorer distortion.

### 2. codex_metric_yshift_av1 — per-frame YUV correction

```python
# inflate.py
SIDECHANNEL_MAGIC = b"SC01"
SIDECHANNEL_HEADER = struct.Struct("<4sBBIf")  # magic + mode + channels + n_frames + step
SIDECHANNEL_MODE_Y_SAT = 6
SIDECHANNEL_MODE_Y_SHIFT = 7

# Per frame: int8 raw value × step → applies Y-channel shift or saturation
# Total: 4-byte magic + 11-byte header + n_frames*channels bytes
```

Plus a SECOND independent sidechannel:

```python
LATENT_LUMA_MAGIC = b"LRL1"
LATENT_LUMA_HEADER = struct.Struct("<4sBBHHHff")  # latent luma reconstruction
```

Mechanism: per-frame Y-channel adjustments on top of AV1-decoded video. Scorer-trained at compress time.

### 3. Lane SJ-KL — Fisher-info residual (in-house)

```python
# src/tac/sjkl_basis.py
class SJKLBasis:
    eigenvectors: torch.Tensor  # (K, D) top-K Fisher eigenvectors
    coefficients: torch.Tensor  # (M,) per-frame projection coefficients
    rank: int
```

Mechanism: low-rank Fisher-info basis + per-frame projection coefficients. K-rank correction in pixel space. Already implemented in this repo (Lane SJ-KL launch-ready per the dispatch trio).

## The unifying insight

All three implementations:
1. Compute the scorer at compress time (legal — strict-scorer-rule allows this)
2. Search a small space of corrections (dim/delta, Y-shift, basis-projection)
3. Pick the per-pair/per-frame correction minimizing distortion
4. Quantize to ≤ 1 byte/parameter (i8 deltas, int8 Y-shifts, intN coefficients)
5. Compress with brotli + a tiny header for layout discovery

Contrast with the **3 PR106 micro-optimization angles deferred earlier this session** (single-stream brotli, schema overhead, latent PCA): those tried to "compress the same bits better" and all came in below noise floor. The sidechannel paradigm "encodes different bits" and consistently wins ~0.001-0.005 score Δ.

## Implications for next dispatch

The PR100 sidecar porting proposal (`docs/pr100_latent_sidecar_porting_proposal_20260504.md`) is **one instance** of this paradigm. Other instances worth investigating:

| Variant | Mechanism | Sidechannel size | Expected gain |
|---|---|---:|---:|
| Per-pair latent | PR100 pattern, ported to PR106 | ~1.2 KB | -0.00218 (predicted) |
| Per-frame Y-shift | codex_metric_yshift pattern | ~600-1200 B | unknown — not measured on PR106 |
| Per-frame Y-sat | codex_metric_yshift pattern | ~600-1200 B | unknown |
| Lane SJ-KL pixel residual | Fisher basis, low-rank | ~1-3 KB | scaffold exists; CUDA dispatch needed |

All four are **stackable** — different correction targets (latents, Y-shift, Y-sat, pixel residual) shouldn't fight each other if their gradients are orthogonal.

Council-gate threshold: each variant needs its own design-decision review per CLAUDE.md. The sidecar proposal is already in the gate queue (registered as `lane_pr106_latent_sidecar` at L1).

## What's NOT this paradigm

These are **codec optimizations** (compress same bits) not **representation extensions** (encode different bits):

- single-stream brotli: -0.000618 — DEFERRED, below noise
- PACKED schema vs per-tensor headers: -0.000281 — DEFERRED, below noise
- latent PCA truncation: NEGATIVE — FALSIFIED
- arith-code the latent deltas: -0 — FALSIFIED (brotli already wins)

The clear distinction: codec-optimization is bounded by Shannon entropy of the GIVEN representation; representation-extension defines a NEW representation with its own (lower) Shannon floor.

## Cross-refs

- PR100 sidecar source: `experiments/results/public_pr100_intake_20260504_codex/source/submissions/hnerv_lc_v2/sidecar.py`
- codex_metric_yshift source: `.../source/submissions/codex_metric_yshift_av1/inflate.py:50-65`
- Lane SJ-KL: `src/tac/sjkl_basis.py` + `experiments/build_sjkl_residual.py`
- Sidecar porting proposal: `docs/pr100_latent_sidecar_porting_proposal_20260504.md`
- Sister deferrals (codec-optimization, below noise): `docs/apogee_intN_single_stream_brotli_DEFERRED_20260504.md` + `docs/pr106_latent_optimization_FALSIFIED_20260504.md`
