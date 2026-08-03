# DDM DE1 — description-efficiency derivation

**Status:** IN PROGRESS · independent upstream-only derivation written before campaign-receipt intake  
**Authority:** `ddm_de1` / `codex_delegate:ddm_de1:20260803T112347Z`  
**Evidence axis:** scorer-definition and frozen-object derivation only; no scorer run, archive, exact row, launch, or pointer movement  
**Landing:** research memo only; **MAIN landing review is required**

## 0. Scope, custody, and contamination cut

This section records the deliberately independent answer before reading the
campaign research corpus. The delegated worktree at HEAD `4e2a609` did not
contain the promised `upstream/modules.py` or `upstream/evaluate.py` in its
filesystem, Git tree, or Git object history. I therefore recovered the exact
public upstream sources and frozen objects from comma.ai's immutable commit
`5387a097398ec6581c7e4e428231e1821fc62670`:

- `models/segnet.safetensors`: 38,502,892 B, SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
- `models/posenet.safetensors`: 55,835,560 B, SHA-256
  `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`;
- `videos/0.mkv`: 37,545,489 B, SHA-256
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`.

The files were streamed to `/Volumes/VertigoDataTier/pact/ddm_de1_20260803/`
after an SSD free-space check. No scorer was run because this arm owns no
scorer slot. The required live-inbox check exposed historical broadcast text;
none of it was used below. `PROGRAM.md` was read only after the derivation had
been worked out, to satisfy the repository's pre-edit rule. The campaign
receipts, `CLAUDE.md`, and craft handoff remained unopened at this cut.

## 1. Independent recursive derivation from the evaluator

### 1.1 The exact task map

For each of 600 two-frame samples, let (x_i=(x_{i,0},x_{i,1})). The evaluator
reduces both frames from camera size (1164\times874) to (512\times384).
Its task statistic is

\[
T(x)=\left(\{L_i\}_{i=1}^{600},\{p_i\}_{i=1}^{600}\right),
\]

where

- (L_i(u)=\arg\max_{c\in\{0,\ldots,4\}} S_c(x_{i,1})(u)) at each of
  (384\cdot512) sites; frame 0 is absent from Seg distortion;
- (p_i\in\mathbb R^6) is the first half of the PoseNet 12-vector computed
  from both resized frames after the exact RGB-to-YUV6 transform.

Thus the Seg target contains

\[
N_s=600\cdot384\cdot512=117{,}964{,}800
\]

discrete decisions, and the Pose target contains (600\cdot6=3{,}600)
continuous scalar decisions. RGB fidelity is absent from the objective except
through this map. A description optimal for the evaluator should therefore
encode a legal preimage of (T(x)), not the source RGB field itself.

### 1.2 DE1-1: what exactly sets `W`

This is the sharpest independent correction. `W = 1.273108215332031` is **not
a measured coder rate, representation rate, or information rate**. It is an
exact unit conversion imposed by the evaluator's scalarization.

One additional realized Seg agreement saves

\[
\Delta S_{\rm flip}=\frac{100}{N_s},
\]

while one archive byte costs

\[
\Delta S_{\rm byte}=\frac{25}{N_0},\qquad N_0=37{,}545{,}489.
\]

Consequently the break-even byte allowance per realized flip is exactly

\[
W=\frac{\Delta S_{\rm flip}}{\Delta S_{\rm byte}}
 =\frac{4N_0}{N_s}
 =\frac{4{,}171{,}721}{3{,}276{,}800}
 =1.27310821533203125\ \mathrm{B/flip}.
\]

Equivalently, one byte must prevent more than (1/W=0.7854791823) realized
flips at the operating margin. A coder, layout, or representation changes the
achieved marginal cost (C=\Delta B/(-\Delta F)); it does not change `W`.
`W` can halve only if the contest weights, original-size denominator, sample
count, or Seg grid changes. The quantity called “residual worth 603,009 B” is
therefore the maximum break-even spend for that residual, not an estimate of
the bytes required to describe it.

This also decomposes the apparent byte/flip question correctly:

\[
\frac{\Delta B}{-\Delta F}
= C_{\rm model}+C_{\rm topology}+C_{\rm geometry}
 +C_{\rm assignment}+C_{\rm exceptions}+C_{\rm container/coder}.
\]

Only the right-hand side is a description-efficiency object. An entropy coder
already at its stream entropy merely says
(C_{\rm container/coder}\approx0) for that representation; it says nothing
about the other terms or about an information-theoretic floor.

### 1.3 What the frozen affine Seg head gives for free

The exact Seg head is a frozen affine (3\times3) convolution from 16 decoder
feature planes to five logits. At an interior site, flattening its local patch
gives

\[
\ell_c(z)=w_c^\top z+b_c,\qquad z\in\mathbb R^{144}.
\]

Argmax is invariant to a common logit offset. Direct inspection of the frozen
object gives centered-head singular values

\[
(3.1283763,\ 2.1542714,\ 2.0247079,\ 1.7962638,\ 3.7\cdot10^{-16}),
\]

so the decision quotient has **exact rank four** and a 140-dimensional local
feature nullspace. The ten winner/rival normal norms range from 2.6018062 to
4.0073639. Hence every decision is a half-space test in a known four-dimensional
quotient; the five cells are simultaneously affine argmax cells and Laguerre
power cells. For current winner (c), the exact feature-space distance to rival
(j)'s hyperplane is

\[
d_{cj}(z)=
\frac{(w_c-w_j)^\top z+(b_c-b_j)}{\lVert w_c-w_j\rVert_2}.
\]

There is a stronger, previously non-obvious head-only consequence. Center the
five output channels and take the DFT symbol of the (3\times3) convolution.
Across all (384\times512=196{,}608) discrete frequencies, the resulting
(4\times16) complex matrix had rank four. Its minimum singular value was
0.5371345; the worst condition number was 4.90583. On the periodic interior
model, **every arbitrary four-channel logit-difference field therefore has a
well-conditioned exact penultimate-feature preimage**, computable by a fixed
frequency-wise pseudoinverse. This is a generic scorer-derived operator.

Scope is critical: this proves surjectivity only from penultimate features to
centered logits in the periodic convolution model. It does **not** prove that
the preceding frozen U-Net trunk has an RGB preimage for that feature field,
nor that such a preimage is describable from contours. The remaining inverse
problem is the trunk plus resize/uint8 receiver, not the affine head.

### 1.4 DE1-2: does separatrix-dimensional coding exist?

**In the target-label domain, yes, constructively. On the public RGB wire,
not yet proved.**

For one (A=384\cdot512) label image, let (E) be the number of dual-grid
edges across which the class changes and (K) the number of boundary-chain
components/junction records. A chain code can losslessly transmit the labeled
partition using, up to topology bookkeeping,

\[
L_{\partial}
\le E\log_2 3
 +K\bigl(\log_2(2A)+\log_2 5+O(\log E)\bigr)
 +L_{\rm faces}.
\]

The first edge is located in the grid; each non-backtracking continuation has
at most three directions; faces are then colored. Conversely, the count of
bounded-degree digital curves grows exponentially in (E), so worst-case
contour codes require \(\Omega(E)\) bits. The exact discrete assignment class
therefore has description length

\[
L(L)=\Theta(E+K\log A),
\]

which scales with the 1-D separatrix rather than 2-D region area whenever
(E=o(A)). For 600 frames the independent-frame upper bound is the sum of
these terms; a temporally coherent codec should instead encode 2-D
world-sheets in ((x,y,t)), their births/deaths/junctions, and motion
innovations rather than 600 unrelated contours.

For a (C^2) boundary of pixel-length (P), bounded curvature \(\kappa\), and
polygon chord length (h), the symmetric-difference area obeys the usual
local chord bound

\[
F\lesssim \kappa P h^2/12.
\]

With (M=P/h) control segments, a predictive vertex code of (b_v) bits per
control therefore gives the explicit lossy upper-bound shape

\[
B_{\partial}(F)
\lesssim \frac{b_v}{8}
\sqrt{\frac{\kappa P^3}{12F}}
 +B_{\rm topology},
\]

until the pixel-grid/exact-chain regime takes over. A single (q)-bit contour
control that displaces arc length \(\ell\) by \(|\delta|\) pixels affects
approximately \(f=\ell|\delta|\) decisions and costs

\[
C_{\rm control}\approx \frac{q}{8\ell|\delta|}\ \mathrm{B/flip}.
\]

It is score-positive precisely when this realized, parse-back cost is below
`W`. This is the mechanism by which dimensional reduction can supply an
order-of-magnitude gain: one geometric parameter moves many decisions.

But a contour bitstream is not a contest witness. It becomes one only if there
is a fixed, generic, deterministic receiver (G) such that

\[
\arg\max S(G(L,p))=L
\]

to the claimed error after camera-resolution output, uint8, evaluator resize,
and the actual frozen trunk, while PoseNet also reaches (p). Source-video
existence proves an RGB preimage, not a *low-description contour-conditioned*
preimage. The affine-head quotient and its right inverse remove the head as an
obstruction but do not solve trunk inversion. This is the decisive falsifier:
if the optimal generic contour-to-RGB right inverse needs video-specific
texture over region area (rather than a fixed-width boundary tube plus class
materials), the public-wire rate returns to 2-D scaling.

### 1.5 DE1-3: the description-space rate–distortion object

For a fixed admissible receiver family (G), the honest object is

\[
R_G(D_s,D_p)=
\min_{b}\left\{|b|:\ d_s(T_s(G(b)),T_s(x))\le D_s,
\ d_p(T_p(G(b)),T_p(x))\le D_p,
\ G(b)\text{ is deterministic and runtime-legal}\right\}.
\]

The scalar contest envelope is

\[
\min_b\left[
\frac{25|b|}{N_0}+100D_s(b)+\sqrt{10D_p(b)}
\right].
\]

This is an algorithmic/MDL rate–distortion problem for one fixed object, not a
Shannon rate–distortion problem unless an ensemble and admissible model class
are specified. There is no non-trivial universal “information limit” for a
single video when decoder code is uncharged: changing the universal machine
could put the target in free code. Rule 118 is exactly the semantic constraint
preventing that degeneracy. Therefore description-class comparisons must fix
or pre-register the generic receiver and count every video-derived fitted
parameter, regardless of whether it is spelled as data or code. In two-part
MDL language the contest waives genuinely generic (L(G)), but it does not
waive video-derived (L(G_x)).

For sparse corrections to a fixed base, choosing (k) sites from a declared
candidate set of size (M), with one of at most four replacement labels,
costs at least/approximately

\[
L_{\rm exc}\simeq \log_2 {M\choose k}+2k
\]

before container overhead. Its marginal location-plus-label price is roughly

\[
\log_2\frac{M-k}{k}+2\quad\text{bits/correction}.
\]

Thus restricting (M) to a scorer-derived separatrix tube is not cosmetic:
it changes the combinatorial rate. At break-even, the marginal price must be
below (8W=10.1848657) bits per *realized* correction. Corrections attempted
but lost through the receiver still consume bits and must remain in the
denominator.

Pose is a separate low-output-dimensional inverse problem. The evaluator sees
only 3,600 real scalars; the frozen Pose final affine map from 32 hidden values
to its scored first six outputs has rank six (singular values 8.7686 down to
0.3533). Directly describing quantized target scalars costs
(3{,}600q/8) bytes before prediction/entropy coding, but those target values
alone are not a solution: the receiver must render two RGB frames whose frozen
Pose trunk produces them. As with Seg, output dimensionality supplies the
right description target but not the public-wire inverse.

## 2. Pre-corpus answers to the four questions

### DE1-1 — `W`

`W` is the evaluator's exact Lagrange exchange rate. It is not set by the
coder, representation, or source information. The live optimization variable
is achieved Δbytes per realized Δflip relative to `W`.

### DE1-2 — dimensional reduction

A separatrix-scaling code exists for the discrete label assignment and has
Θ(total boundary length + topology) rate. Its transfer to a contest-valid RGB
witness is conditional on a generic contour/world-sheet-to-RGB right inverse
through the full trunk and receiver. The affine head is not the obstruction;
the trunk inverse is.

### DE1-3 — description-space `R(D)`

Use fixed-receiver algorithmic `R_G(D_seg,D_pose)` and its contest scalar
envelope. A one-video information floor is undefined without a fixed generic
receiver/model class or ensemble; entropy saturation of one token layout is
not such a floor.

### DE1-4 — frozen-head leverage not already exhausted

Beyond “do not ship the weights,” the head provides:

1. an exact four-dimensional decision quotient and 140 local null directions;
2. exact winner/rival hyperplanes and distances, so precision and exception
   bits can be allocated by realized margin rather than RGB error;
3. a well-conditioned frequency-wise right inverse from arbitrary centered
   logit fields to penultimate features in the periodic interior model;
4. a canonical target field: piecewise-constant quotient prototypes plus
   narrow transition tubes, rather than dense RGB or dense 144-D features.

Items 3–4 are only useful if the RGB trunk inverse preserves their dimensional
advantage; that is the first measurement owed.

## 3. Corpus and receipt diff — STUB

Pending mandatory corpus intake after this independent cut. Every agreement
will be labeled corroboration; every disagreement will name which premise or
measurement decides it. Corpus-query coverage denominators will be reported.

## 4. Literature and OSS race — STUB

Pending task-aware/VCM coding, contour/region MDL, level-set/quotient codecs,
grammar/description vocabulary, and min-description lattice literature/OSS
intake. Primary sources only; no old contest lineage will be used as a vehicle,
carrier, or calibration source.

## 5. Ranked description classes — PRE-CORPUS STUB

| Rank | Class | Derived rate mechanism | Public-wire crux | Pre-registered falsifier | Named consumer |
|---:|---|---|---|---|---|
| 1 | Temporal separatrix/world-sheet grammar + generic trunk right inverse | boundary topology + controls; each control costs `q/(8 * realized_flips)` B/flip | existence and stability of contour-to-RGB inverse | n600 receiver-closed controls do not beat `W`, or fitted content expands over region area | DE1-2 inverse-race consumer; refine after corpus routing |
| 2 | Scene/ego factorization + labeled surfaces + exceptions | persistent scene surfaces and camera trajectory amortize assignments across time | disocclusion and non-rigid exceptions | counted scene + exception stream scales no better than independent contours | DE1 temporal-description consumer; refine after corpus routing |
| 3 | Frozen-head quotient field + transform code | four decision coordinates instead of 144 local features; head inverse is generic | trunk range and dense spatial field | quotient payload remains area-scaling or has no RGB preimage | DE1-4 head-inverse consumer; refine after corpus routing |
| 4 | Separatrix-restricted sparse exception code | `log2 C(M,k)+2k` bits; reduce `M` using scorer-derived geometry | base witness and parse-back survival | marginal bits/realized correction ≥ `8W` at optimal form | DE1-3 exception-waterfill consumer; refine after corpus routing |
| 5 | Region/pixel assignment code | entropy of 2-D labels/latents | pays area rather than boundary | retained only as a measured upper bound | baseline comparator only |

No class is adopted or killed by this table. Bytes are formula-derived, not
asserted measurements; the n600 receiver-closed falsifiers are still owed.

## 6. Round-1 adversarial self-review — STUB

Pending after corpus and literature diff. Required attacks include: hidden
video-derived content in the supposedly generic inverse; confusing
penultimate-feature surjectivity with RGB-trunk surjectivity; counting
attempted rather than realized flips; zero-padding versus periodic-head
boundary conditions; topology overhead; Pose interference; and receiver
runtime/quantization survival.

## 7. Pointer-delta honesty

Pointer delta: **none**. This memo is a means, not frontier progress. It
contains no archive and no exact evaluation row.

## NEXT-IF-RESUMED

1. Read `CLAUDE.md`, then `docs/operating_manual_craft_handoff.md`, exactly as
   delegated.
2. Fetch/rebase check for the `ddm_cu1` unclonable-tree repair before corpus
   work; never rebuild absent modules.
3. Query the indexed corpus for DE1-1..4, #151/#611, #620/#650/#651,
   #662/#663, contour/region MDL, quotient/level-set, and frozen-head/trunk
   inversion; record hit and index denominators.
4. Diff the campaign evidence against Sections 1–2 and replace placeholder
   consumers with canonical names.
5. Race primary literature and OSS, then complete the ranked table and conduct
   the adversarial self-review.
6. Commit only through `tools/subagent_commit_serializer.py`; MAIN must review
   the landing.
