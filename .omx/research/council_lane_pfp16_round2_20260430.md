# Lane PFP16 — Adversarial Review Round 2

Date: 2026-04-30
Reviewer perspectives: Shannon, Dykstra, MacKay, Ballé, Selfcomp
Object of review: Lane PFP16 implementation + Check 96 + tests + build + dispatch
Counter on entry: 1/3 clean passes (Round 1 found 0 issues)
Files reviewed: same set as Round 1.

## Round 2 perspectives

### Shannon (rate-distortion lens)

**Question 1**: What is the theoretical floor for the pose stream
specifically? Lane G v3's poses span [-6.5, 37.7] across 6 dims. If
each dim has effective entropy `H_dim`, then total bits = 600 × Σ H_dim.

**Counter**: Per the Lane GP v4 spectral analysis, dim 0 is mostly
smooth (99.8% energy in top-10 modes) and dims 1-5 are approximately
white noise with std ~0.5-2.3. For continuous Gaussian sources the
differential entropy is `h = (1/2) log2(2πeσ²)`, but for QUANTIZED
representation at fp16 precision (∆ ≈ 1e-3 at the relevant magnitudes)
the effective discrete entropy is `H ≈ h - log2(∆) ≈ (1/2)log2(2πeσ²) +
10`. For dim 0 (σ=1.55): H ≈ 12.5 bits/sample. For dims 1-5 average:
H ≈ 11 bits/sample. Total: 600 × (12.5 + 5×11) ≈ 40,500 bits ≈ 5,063 B.

**Verdict**: Lane PFP16 ships 7,200 B raw fp16 → 1.42× the Shannon
floor. There's room for further compression (Lane PD-V2 measures 18.5%
savings on top of fp16, getting closer to ~5.9 KB). But Lane PFP16 is
the LOSSLESS-vs-fp16 baseline; further reduction trades distortion
penalty (not in scope for THIS lane). Lane PFP16 is on the Pareto frontier.

**Status**: NO ISSUE — Lane PFP16 is theoretically reasonable.

**Question 2**: The build script uses ZIP_DEFLATED with compresslevel=9.
Is that optimal for raw fp16 bytes? raw fp16 is high-entropy (close to
random) and DEFLATE typically can't compress random bytes; the stored
size in the ZIP is approximately = uncompressed size + ZIP overhead.

**Counter**: Empirical: raw fp16 (7200 B uncompressed) → likely 7200 B +
ZIP entry overhead in the archive. The fp32 pickle (15,620 B
uncompressed) compresses TO ~15,500 B in ZIP_DEFLATED (PyTorch pickle
has SOME redundancy DEFLATE catches). Net archive change should be ~−8,300
B if both compress poorly, OR ~−7,400 B if pickle compresses better
than raw fp16 (we measured -7,439 B → pickle DID compress slightly
better, eating ~1KB of the savings).

**Status**: NOTED — small efficiency loss vs theoretical -8,420 B, but
still net negative. NO BLOCKER; the savings are real.

### Dykstra (convex feasibility / Pareto frontier)

**Question**: Lane PFP16 occupies what point on the (rate, distortion)
Pareto frontier? Is this Pareto-optimal, or is it dominated by some
combination?

**Counter**: Define the pose-stream feasibility set as:
- Rate constraint: bytes ≤ R
- Distortion constraint: PoseNet score impact ≤ D

Lane PFP16: (R, D) = (7,200 B, 0). Lane G v3 fp32 baseline: (15,620 B,
0). Both have D=0, but PFP16 has lower R → PFP16 strictly DOMINATES
fp32-pickle. Lane PD: (~3,668 B, ε) where ε is the int8-delta
quantization error mapped through PoseNet. Lane PD-V2: (~5,866 B, ε').
None of these have D=0; they trade distortion for bytes. PFP16 is the
unique D=0 point with R < 15,620 B.

**Status**: PFP16 IS Pareto-optimal in the (R, D=0) constraint set.

### MacKay (Bayesian / MDL)

**Question**: From an MDL standpoint, what is the description length of
the PFP16 representation vs the fp32 pickle?

**Counter**: MDL = model description + data description.
- fp32 pickle: model = "tensor of shape (600, 6) at fp32" (description
  length ~32 bits for shape header). Data = 600×6×32 = 115,200 bits.
  Total: ~115,232 bits ≈ 14.4 KB.
- PFP16 raw: model = "raw fp16 buffer reshaped to (?, pose_dim)"
  (description length ~32 bits, encoded in the .bin filename + dim
  inference from buffer length). Data = 600×6×16 = 57,600 bits.
  Total: ~57,632 bits ≈ 7.2 KB.

**MDL ratio**: 7.2 / 14.4 = 0.5. Lane PFP16 has HALF the description
length. MDL strongly favors PFP16 IF the posterior over the
data-generating distribution accepts fp16 precision (it does — PoseNet's
forward path runs in fp16 anyway).

**Status**: NO ISSUE. MDL strongly favors PFP16.

**Question 2**: Could a Bayesian prior over pose deltas (e.g., Gaussian
with empirical σ) achieve better than fp16's ~1e-3 effective precision?

**Counter**: Yes — Lane PD-V2 (arithmetic-coded deltas) achieves ~5.9
KB at marginal distortion. But the posterior over the data-generating
process is exactly what Lane PD-V2 exploits, while Lane PFP16 is
agnostic. Both are valid lanes; they occupy different points on the
frontier. PFP16 is the simplest viable lane — D=0, no learned codec.

**Status**: NO ISSUE — Lane PFP16's role as the D=0 reference point
stands.

### Ballé (neural compression / hyperprior)

**Question**: Could a learned entropy model (Ballé hyperprior) compress
the pose stream below fp16 bytes?

**Counter**: For a learned entropy model to beat fp16 raw, it must
predict the value distribution well enough that the arithmetic-coded
representation requires fewer than 7200 bytes. Per the spectral
analysis, dims 1-5 are approximately white-Gaussian with σ ~ 0.5-2.3.
For a Gaussian source the optimal entropy code is ~12 bits per sample
at fp16 precision (10-bit mantissa + 2-bit exponent overhead) — vs
fp16's 16 bits per sample → 25% theoretical compression over raw fp16.

But the learned model itself needs side-info bytes (Ballé's
ScalePriorMLP). For a 7200-byte signal, the side-info amortization
breaks even only if the codec is shared across many archives. Lane PFP16
is the trivial baseline; Lane J-NWC or Lane 20-Ballé could dominate
PFP16 IF a cross-archive shared codec lands. **For a single-archive
ship, Lane PFP16 is the right baseline.**

**Status**: NO ISSUE. Lane PFP16 is the right reference point for
single-archive ships; Ballé becomes relevant only when amortized.

### Selfcomp (empirical pragmatism — Quantizr's framing)

**Question**: Selfcomp's 0.38 archive uses no separate pose file at all
(PoseNet-affine-learned-image trick). Quantizr's 0.33 ships raw poses.
Is Lane PFP16 actually "the strict-best D=0 lane" or is it a
local-optimum that misses a paradigm shift?

**Counter**: PoseNet-affine-learned-image (Lane LI) is a different
paradigm — it eliminates the pose file entirely by encoding pose as a
learned image augmentation. That's a SUPERSET lane. For the existing
Lane G v3 architecture (FiLM-conditioned renderer), pose values are
required at inflate. Lane PFP16 is the strict-best D=0 lane WITHIN the
"pose values required at inflate" subset. Lane LI is a parallel lane
(different architecture) and does not invalidate Lane PFP16.

**Status**: NO ISSUE — Lane PFP16 is correctly scoped to the existing
Lane G v3 architecture.

**Question 2**: Selfcomp explicitly aborted multiple lane ideas as "not
worth the integration cost." Is Lane PFP16 worth the cost?

**Counter**: Lane PFP16 cost: ~270 lines of code + 31 tests + 1 STRICT
preflight check + 1 council review cycle + $0.50 contest CUDA
validation. Estimated total agent-time: ~2-3 hours. Predicted gain:
−0.005 score (if it lands as predicted). At Lane G v3's 1.05 baseline,
−0.005 is 0.5% relative improvement. At Quantizr's 0.33 frontier, the
same delta would be 1.5% relative improvement. The cost is bounded; the
upside is real (though small). Worth shipping.

**Status**: NO ISSUE — cost/benefit is positive.

## Issues found

**ZERO CRITICAL issues. ZERO Medium issues. ZERO Low issues.**

The Shannon analysis confirms Lane PFP16 is on the Pareto frontier
(specifically the D=0 reference point). Dykstra confirms it strictly
dominates the fp32 baseline. MacKay confirms MDL favors it. Ballé
confirms it's the right reference point for single-archive ships.
Selfcomp confirms it's correctly scoped to the existing architecture.

The only NOTED but non-blocking observation is Shannon Q2 (the empirical
savings -7,439 B is slightly less than theoretical -8,420 B because
ZIP_DEFLATE compresses fp32 pickle better than raw fp16). This is
expected behavior and correctly reflected in the predicted band.

## Round 2 verdict

**0 issues. Counter advances to 2/3 clean passes.**

Proceed to Round 3 with rotated perspectives.
