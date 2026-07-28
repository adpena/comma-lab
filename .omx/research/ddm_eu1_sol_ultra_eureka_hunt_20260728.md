# DDM EU1 — evaluator-first codec theory and eureka hunt

**Pointer honesty — MEASURED historical local-custody anchor:** `0.1910828242`
`[contest-CPU]` is **UNMOVED**. Nothing in this research-only memo creates an
archive, runs an exact evaluation, or moves any frontier pointer. The current
competitive/effective pointer is a separate pointer-file concern; `0.1910828242`
is used here only in the exact local-custody sense required by this delegation.

**Authority boundary:** research only; no launch, no paid work, no training, no
n600 scorer job, no candidate promotion. MAIN landing review is required.

**Phase-order receipt:** the scientific body of P1 below was written before
reading any `ddm_*` campaign artifact, TR1 specification, FD2 receipt, EE1
memo, or campaign memory. A subsequent serializer-preflight check exposed
recent commit subjects before this first commit; no P1 scientific text was
changed in response. The only scientific inputs to P1 were the pinned upstream
evaluator, the pinned scorer definitions, the pinned upstream README, and basic
non-scorer video statistics.

**STORES CONSULTED:** P1: `upstream/evaluate.py`,
`upstream/modules.py`, `upstream/README.md`, `upstream/videos/0.mkv` basic
container/downsample statistics only. P2–P4: PENDING.

## P1 — uncontaminated evaluator-first derivation

### P1.0 Primary-source custody

- **VERIFIED_VIA_SOURCE_INSPECTION:** `evaluate.py` SHA-256
  `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`;
  `modules.py` SHA-256
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`;
  README SHA-256
  `68ea239d7333696e79716e47a9c4288d2918efbcd8912f78932b0befe0af872b`.
- **MEASURED:** the sole video is `37,545,489` bytes, SHA-256
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`,
  HEVC Main, `1164×874`, `yuv420p`, 20 fps, 1,200 frames, 60 seconds.
- **MEASURED, BASIC-STATISTIC ONLY:** an area-downsampled `64×48` RGB read of
  all 1,200 frames has adjacent-frame pixel correlation `0.983360`,
  adjacent RGB MAD `1.7326`, two-frame RGB MAD `2.1875`, and adjacent absolute
  difference quantiles `(q50,q90,q99)=(1,4,13)`. This establishes strong
  temporal redundancy at a coarse scale; it is not scorer evidence and is not
  a distortion or score result.

### P1.1 The actual mathematical object

**DERIVED:** Let pair \(i\) contain reconstructed uint8 frames
\(y_i=(y_{i,0},y_{i,1})\). From `modules.py`, SegNet observes only
\(y_{i,1}\), resizes it bilinearly to `512×384`, and is scored only through
five-way argmax disagreement. PoseNet observes both frames through a
12-channel YUV6 representation, also at `512×384`, and only its first six
outputs enter an MSE. Therefore the exact distortion is

\[
D_{\rm seg}(y)=\frac{1}{600HW}\sum_{i,u}
  1[\arg\max G(y_{i,1})_u \ne m_{i,u}],
\qquad
D_{\rm pose}(y)=\frac{1}{600}\sum_i
  \|H(y_{i,0},y_{i,1})_{:6}-p_i\|_2^2/6,
\]

where \(m_i\) and \(p_i\) are encoder-side targets obtained from the original.
The scored program minimizes

\[
100D_{\rm seg}+\sqrt{10D_{\rm pose}}
+25\,|\mathrm{archive.zip}|/37{,}545{,}489.
\]

**DERIVED:** This is not fundamentally an RGB-reconstruction problem. It is a
minimum-description representative problem over evaluator equivalence cells:
find the shortest legal message whose decoder emits any uint8 frame pair in
the intersection

\[
\mathcal C_i(m_i,p_i)=
\{(a,b):\arg\max G(b)=m_i,\ H(a,b)_{:6}=p_i\},
\]

with controlled relaxations when exact intersection membership costs more
bytes than its score value. Pixel fidelity has no independent term.

**DERIVED:** Frame 0 is asymmetric: it has no direct SegNet obligation.
Consequently the optimum should first place frame 1 in a large, robust SegNet
argmax cell, then use the remaining degrees of freedom in frame 0—and
SegNet-null directions of frame 1—to close PoseNet. A symmetric two-frame RGB
codec spends rate on constraints the evaluator never imposes.

### P1.2 Byte economics

**DERIVED:** one archive byte costs
`25/37,545,489 = 6.6585895312e-7` score units. A binary kilobyte costs about
`0.0006818`; 64 KiB costs `0.04363773`.

**DERIVED:** at 64 KiB, a sub-0.15 row has only `0.10636227` total distortion
budget. If \(D_{\rm seg}=5\times10^{-4}\), PoseNet must satisfy approximately
\(D_{\rm pose}<3.1767\times10^{-4}\). If
\(D_{\rm seg}=3\times10^{-4}\), the corresponding bound relaxes to about
\(5.8312\times10^{-4}\). These are feasibility arithmetic, not predictions.

**DERIVED:** admitting a payload increment \(\Delta B>0\) is rational only if

\[
-100\Delta D_{\rm seg}
-\left(\sqrt{10D_{\rm pose,new}}-\sqrt{10D_{\rm pose,old}}\right)
> 6.6585895312\times10^{-7}\Delta B.
\]

The square-root term makes pose byte value state-dependent; a fixed
Seg/Pose loss-weight ratio cannot be globally optimal.

### P1.3 Derived codec form

**CONJECTURE — minimum-description witness codec:** the optimal family is a
task-space witness compiler with five jointly designed parts:

1. **Encoder-side solved witnesses.** Use the frozen scorers only offline to
   solve uint8 frame pairs directly inside or near the target evaluator cells.
   The solve must include the exact resize, color conversion, clipping, and
   integer lattice; a continuous image later rounded is solving the wrong
   feasible set.
2. **A shared conditional renderer.** Amortize repeated road-scene structure
   across all 600 pairs with one compact, quantized decoder. It should emit
   piecewise-smooth regions and sharp learned boundaries rather than optimize
   generic perceptual texture.
3. **Predictive pair tokens.** Entropy-code a temporally predictive stream of
   low-dimensional pair state. The measured `0.983360` coarse adjacent
   correlation supports prediction, but the latent dimension and entropy are
   OPEN-QUESTIONs until measured on the learned tokens.
4. **Sparse evaluator certificates.** Spend exceptions only on sites/pairs
   whose evaluator-cell violation has positive score value per byte. Each
   exception needs a receiver-consumption proof; a stored target that the
   renderer does not consume has zero mechanism.
5. **Maximal generic decoder, minimal video-derived message.** The decoder
   algorithm, deterministic bases, integer rasterizer, entropy decoder, and
   generic optimization machinery belong in the free runtime. Every
   video-derived weight, token, target, exception, or learned table belongs in
   the counted archive.

**CONJECTURE — teacher-to-packet path:** first solve a large unconstrained
teacher object on the exact discrete evaluator surface; next distill its
evaluator outputs into the small renderer and predictive tokens; finally
optimize archive bytes and task distortion jointly. The teacher is useful as
an existence witness and initialization oracle, not as a shippable payload.

**CONJECTURE — cell-interior training:** because SegNet scores argmax, the
renderer should maximize the minimum winner–runner-up robustness under the
actual integer/resize neighborhood at already-correct sites, while spending
capacity on incorrect sites. Cross-entropy everywhere can waste bits increasing
already-safe margins that do not change \(D_{\rm seg}\).

**CONJECTURE — integer-native terminal optimizer:** once continuous descent
reaches a sub-quantum basin, switch to discrete coordinate/block proposals on
the uint8 lattice, ranked by exact evaluator-cell changes per coded bit.
Continuous straight-through gradients are proposal generators, not terminal
authority.

### P1.4 Falsifiers derived before campaign recall

1. **G1 renderer falsifier — OPEN-QUESTION:** on all 600 pairs and the exact
   realized uint8/resize path, can a counted renderer of at most 64 KiB reach
   native \(D_{\rm seg}\le5\times10^{-4}\), with
   \(3\times10^{-4}\) as the stretch target? A smaller sample is not a verdict.
2. **Amortization falsifier — OPEN-QUESTION:** after entropy coding, do shared
   weights plus predictive pair tokens beat direct compressed witness frames
   at equal exact task distortion?
3. **Asymmetry falsifier — OPEN-QUESTION:** does reserving frame 0 primarily
   for pose reduce total score relative to a symmetric renderer at identical
   archive bytes?
4. **Cell-interior falsifier — OPEN-QUESTION:** at equal bytes, does
   margin/cell-aware allocation reduce realized argmax error more than dense
   logit or RGB regression?
5. **Discrete-finish falsifier — OPEN-QUESTION:** after continuous convergence,
   does an integer-native local optimizer find positive net
   \(\Delta S/\mathrm{byte}\) moves that round-to-nearest/STE misses?

## P2 — verified transformed state

PENDING until the P1 commit is sealed.

## P3 — eureka hunt

PENDING until transformed-state recall and independent source research.

## P4 — ranked eureka table

PENDING.
