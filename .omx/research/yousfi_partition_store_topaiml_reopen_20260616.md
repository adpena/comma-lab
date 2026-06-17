# YOUSFI partition store — TOP-AIML re-open ($0 CPU probe) — 2026-06-16

**Authority:** `[macOS-CPU advisory]` NON-PROMOTABLE. Exact CPU-torch SegNet, GT decode via
`upstream/frame_utils.yuv420_to_rgb` only. No GPU, no MPS, no PR. `n_pairs=24`.

## Why re-opened

CLAUDE.md "ANTI-SIGNAL-LOSS / janky-prototype -> top-AIML RE-OPEN" non-negotiable. The prior
verdict (`experiments/yousfi_tolerance_partition_remeasure.py`) rested on a PROTOTYPE coder:
per-frame **LZMA over RAW uint8 labels** (`tac.boundary_math.contour_codec.partition_description_bytes`),
measured at 669-873 B/frame -> rate term 0.27-0.35 alone -> DEAD on rate. A prototype-grade
verdict is implementation-level falsified, paradigm intact (Catalog #307).

## What was built (top-AIML, NO FAKE)

`src/tac/boundary_math/context_partition_codec.py` — a REAL context-adaptive arithmetic codec
for the 5-class partition stack, the SOTA coder for a smooth label source (JBIG/LOCO-I/CABAC family):

- **Lever A — context coding:** `constriction` RangeEncoder over a causal `(left, up)` template
  (`spatial`). Achievable length = `sum_ctx N_ctx*H(p_ctx)`, the contour/RLE-subsuming floor.
- **Lever B — temporal:** adds the co-located previous-frame pixel to the context `(left, up, prev)`
  (`temporal`) -> exploits dashcam temporal redundancy.
- Shared per-context model built over the WHOLE stack, transmitted ONCE (brotli'd uint16 counts),
  so model bytes are amortized and **counted** (no free model). Vectorized rank-2 encode; causal
  per-pixel decode. **`decode(encode(x))` is bit-exact** (18 NO-FAKE tests; verified inline on the
  real partitions for both templates).
- **Lever C — margin-weighted simplification:** drop sub-margin boundary wiggle up to a d_seg budget
  (ported from the prototype), rate-vs-d_seg Pareto sweep.

Probe: `experiments/yousfi_partition_topaiml_probe.py`. Report: `reports/yousfi_partition_topaiml.json`.

## Measured verdict (n_pairs=24)

| coder | B/frame | rate term |
|---|---:|---:|
| PROTOTYPE (per-frame LZMA-over-raw-labels) | 902.3 | 0.360 |
| TOP-AIML spatial (left,up) | 565.5 | 0.226 |
| **TOP-AIML temporal (left,up,prev)** | **456.5** | **0.182** |

- **49.4% smaller than the prototype.** Temporal lever ~19% over spatial (as predicted).
- Roundtrip bit-exact on real partitions: spatial ✓ temporal ✓.

### Apples-to-apples vs frontier (0.1911 = rate 0.118 + seg 0.056 + pose 0.017)

The lossless store achieves **d_seg = 0** (BETTER than the frontier's 5.6e-4) but pays a HIGHER rate:

- lossless-store total at d_seg=0 = rate 0.182 + seg 0 + pose 0.017 = **0.1994**
- store vs frontier gap = **+0.0083** (store pays +0.064 rate to save 0.056 seg -> net +0.008 worse)
- to BEAT the frontier at d_seg=0: need ≤ 435.8 B/frame -> **1.05× smaller** (have 456.5)
- to BEAT sub-0.15 at d_seg=0: need ≤ 332.9 B/frame -> **1.37× smaller**

### Lever C (margin simplify) is DOMINATED

Every drop level makes `100·d_seg` grow faster than rate falls; best store score is at drop_frac=0
(0.1994). Simplification does not open a rate-competitive operating point.

## VERDICT: `NO_GO_PARTITION_STORE_DEAD_ON_RATE` — but BORDERLINE, not closed

Even the SOTA context+temporal+margin codec cannot get the non-neural partition store below the
frontier. This CONFIRMS **d_seg belongs in training** (a learned renderer carries the partition
cheaper than storing it directly). HOWEVER the gap collapsed from the prototype's hopeless
+0.04…+0.16 to only **+0.0083** — the partition is **borderline**, needing only ~1.05× tighter
coding. A tighter coder (full per-pixel CABAC-style adaptive context with a richer template, or a
store/training hybrid that lets the renderer carry the easy regions and the store carry only the
hard boundary residual) is a LIVE lever, not a closed door.

## Wire-in / reusable surface

- `tac.boundary_math.context_partition_codec` is a reusable codec (encode/decode/bpf) any
  partition-coding lane can import — strictly dominates `contour_codec.partition_description_bytes`
  on the partition rate axis (49% smaller). Sister of `tac.optimization.partition_contour_entropy`
  (which measures the floors this codec realizes).
- 6-hook: #3 bit-allocator (partition rate term) ACTIVE; #6 probe-disambiguator (this probe) ACTIVE;
  #1/#2/#4/#5 N/A for an advisory non-promotable $0 probe.
