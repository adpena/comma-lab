# DDM DE1 — description-efficiency derivation

**Status:** COMPLETE RESEARCH DERIVATION · no scorer run, candidate, or frontier progress
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

The immutable source files used for the cut were also hash-checked:

- `modules.py`: `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`;
- `evaluate.py`: `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`;
- `frame_utils.py`: `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90`.

After this independent cut was committed, I read `PROGRAM.md`, all of
`CLAUDE.md`, and `docs/operating_manual_craft_handoff.md`, verified that the
`cu1` repair was already in the delegated base, and only then queried campaign
stores. Sections 3 onward are the resulting diff; Sections 1--2 preserve the
independent answer except where Round 1 found and labels a correction.

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

For the `ddm_sg2` integer counts, the gap is

\[
F=508{,}639-34{,}988=473{,}651,
\qquad D_{\rm gap}=F/N_s=0.004015189276801215,
\]

and its exact fixed-pose ceiling is

\[
FW=\frac{1{,}975{,}939{,}823{,}371}{3{,}276{,}800}
 =603{,}008.9793002319\ \mathrm{B},
\]

whose rounded headline is 603,009 B. Because a strict score improvement needs
an integer spend below this real-valued break-even point, the largest admissible
fixed-pose spend is 603,008 B; 603,009 B is the score-equivalent rounded worth,
not a strictly winning integer budget. This
`D_gap` is not the campaign's separately defined population coarea density
`rho`; no `rho` value is imported here. The delegated authority says the live
archive was 353,808 B while the queried `ddm_sg2` snapshot says 360,406 B.
Those are different snapshots and are not combined. Neither number enters
`W` or the 603,009 B derivation.

`W` is a fixed-pose exchange only. If a treatment adds `c` archive bytes,
removes `r` realized Seg mismatches, and changes population Pose MSE by
`delta`, its exact joint change is

\[
\Delta S=-\frac{100r}{N_s}
+\sqrt{10(D_p+\delta)}-\sqrt{10D_p}+\frac{25c}{N_0}.
\]

It wins only if

\[
\frac cr < W-
\frac{N_0}{25r}\left[\sqrt{10(D_p+\delta)}-\sqrt{10D_p}\right].
\]

This is why every empirical price below is a byte-closed finite secant of the
whole object, not a sum of independently measured lever credits.

The useful fixed-object decomposition is an ordering of minima, not a claim
that interacting costs add independently. Every term below refers to the same
source object `x`, distortion pair, byte units, and nested description
languages:

\[
R_{{\rm output},x}(D)\le R_{G,x}(D)\le
R_{{\mathcal V},x}(D)\le L_{C,x}(D).
\]

Here `R_output,x` is the shortest fixed-language description of an approximate
scorer-output object when video realizability is relaxed; `R_G,x` adds a legal
fixed right inverse; `R_V,x` restricts that inverse to a chosen representation
class; and `L_C,x` is the achieved archive length under a concrete probability
model, coder, and ZIP container. Telescoping finite differences along one
declared path can assign the gaps to realizability, representation, and coder
redundancy. An entropy coder already at its stream entropy merely makes the
last gap small for that serialization; it says nothing about the other gaps or
the fixed-object output-description floor. The ensemble RDF introduced in
Section 1.5 is a separate expected-bit object and is not numerically compared
with this hierarchy without normalizing all four terms to that same ensemble.

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

More exactly, the power representation is the identity

\[
\arg\max_c(w_c^\top z+b_c)
=\arg\min_c\left(\lVert z-w_c\rVert_2^2
-\lVert w_c\rVert_2^2-2b_c\right).
\]

Affine-argmax, tropical max-affine, and Laguerre/power-cell descriptions are
therefore exact reparameterizations **in the 144-D terminal-feature patch
space**. They do not make the cells spatial power diagrams in `(x,y)` or in
RGB space. The stronger campaign premise that a frozen affine head is
automatically a Morse--Smale complex is false: a Morse--Smale complex also
needs a smooth scalar potential, its gradient flow, nondegenerate critical
points, and transverse stable/unstable manifolds. Five affine logits do not
supply these. “Bregman Voronoi” is likewise a selected convex-generator
coordinate description, not additional free spatial structure beyond the
power identity.

There is a stronger, previously non-obvious head-only consequence. Center the
five output channels and take the DFT symbol of the (3\times3) convolution.
Across all (384\times512=196{,}608) discrete frequencies, the resulting
(4\times16) complex matrix had rank four. Its minimum singular value was
0.5371345; the worst condition number was 4.90583. On the periodic interior
model, **every arbitrary four-channel logit-difference field therefore has a
well-conditioned exact penultimate-feature preimage**, computable by a fixed
frequency-wise pseudoinverse. This is a generic scorer-derived operator.

The numerical singular values depend on quotient coordinates. Using four
reference-class differences gives local singular values
`(4.703452, 2.831397, 2.038933, 2.018443)`, periodic minimum singular value
`0.642995`, and worst condition `4.80761`; the orthonormal centered quotient
above gives `0.5371345` and `4.90583`. Rank, cells, and reachability are the
same. This basis dependence is not evidence of two different heads.

Scope is critical: this proves surjectivity only from penultimate features to
centered logits in the periodic convolution model. It does **not** prove that
the preceding frozen U-Net trunk has an RGB preimage for that feature field,
nor that such a preimage is describable from contours. The remaining inverse
problem is the trunk plus resize/uint8 receiver, not the affine head.
Moreover, this is primarily an **encoder-side coordinate system**: the contest
receiver does not expose SegNet, and scorer weights may not be embedded in the
archive or disguised as decoder code. A legal receiver must compile the
head-derived design into generic non-scorer code plus counted video-specific
statistics; it cannot call this pseudoinverse by shipping the scorer.

### 1.4 DE1-2: does separatrix-dimensional coding exist?

**In the target-label domain, yes, constructively. On the public RGB wire,
not yet proved.**

For one `H=384`, `W=512` label image, let `A=HW`, let `L` be the
number of internal grid edges across which the class changes, and let `R` be
the number of connected labeled regions. The grid has

\[
E_0=H(W-1)+(H-1)W=392{,}320
\]

possible internal edges. An enumerative exact code can transmit the boundary
edge subset and one class per recovered region using at most

\[
B_{\partial}\le
\left\lceil
\frac{\left\lceil\log_2{E_0\choose L}+R\log_2 5\right\rceil}{8}
\right\rceil+B_{\rm header}.
\]

For `K` coherent boundary chains, a traversal code gives the sharper
conditional bound

\[
B_{\rm chain}\lesssim
\frac{L\log_2 3+K\log_2(2A)+R\log_2 5+L_{\rm topology}}{8}
+B_{\rm header}.
\]

The first formula is general but pays the sparse-position logarithm;
the second needs coherent traversable components. Thus the corrected claim is

\[
B_{\partial}=O\!\left(L\log(eE_0/L)+R+L_{\rm topology}\right)
\]

in general, and `O(L+K log A+R)` for the coherent-chain subclass. My initial
`Theta(L+K log A)` statement was too strong for arbitrary disconnected maps
and is withdrawn. Checkerboards, pixel-scale islands, or fractal boundaries
can have `L` or `R=Theta(A)` and erase the dimensional gain. For finite-
perimeter regions under grid refinement `h`, however, `L=Theta(h^-1)` while
`A=Theta(h^-2)`: the assignment really can be separatrix-scaled, up to logs
and topology. Over time, the corresponding object is a 2-D worldsheet in
`(x,y,t)`, not literally a 1-D curve; motion prediction can code its temporal
innovations instead of 600 independent contours.

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

Before measurement this is only a geometric potential price. Authority
replaces `ell*|delta|` by the net n600 receiver-surviving `r`, after subtracting
new harms, and uses the exact payload delta. At fixed Pose the resulting
`Delta B/r` is score-positive precisely when it is below `W`; if Pose changes,
the threshold is the adjusted joint bound in Section 1.2. This is the mechanism
by which dimensional reduction can supply an order-of-magnitude gain: one
geometric parameter moves many decisions.

A useful robust cross-check is a **conditional** tube bound. Let
`K_partial` be the number of boundary components. Assume rectifiable target
boundaries of total length `P` and reach greater than `delta`, a fixed
class-preserving region correspondence and orientation, no uncharged topology
change, and that every boundary-induced disagreement lies in the target's
`delta`-tube. The continuous symmetric-difference area then obeys

\[
A_{\triangle}\le 2P\delta+\pi K_{\partial}\delta^2.
\]

On the sampled grid the accountable statement is

\[
F\le A_{\triangle}+F_{\rm grid}+F_{\rm topology}+F_{\rm section},
\]

where `F_grid` is an explicit lattice-cover slack, not silently zero. Identical
curves with permuted class labels violate the correspondence assumption and
belong in `F_section`, not in the boundary-displacement term.

For a normal displacement field of range `[-U_d,U_d]`, sampled every `ell`
pixels, quantized at step `q`, and satisfying `||u''||_infty<=A_2`,

\[
B_{\rm disp}\approx\frac{P}{8\ell}
\left\lceil\log_2(1+2U_d/q)\right\rceil+B_{\rm topology},
\]

\[
F\lesssim P(A_2\ell^2/8+q/2)+F_{\rm topology}+F_{\rm section}.
\]

These equations predict bytes per realized flip before a run; the run must
replace `F` by exact n600 receiver-surviving corrections.

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

For a fixed source-independent, admissible receiver `G`, the single-object
operational curve is

\[
R_{G,x}(D_s,D_p)=
\min_{b}\left\{|b|_{\rm bytes}:\ d_s(T_s(G(b)),T_s(x))\le D_s,
\ d_p(T_p(G(b)),T_p(x))\le D_p,
\ G(b)\text{ is deterministic and runtime-legal}\right\}.
\]

The scalar contest envelope is

\[
\min_b\left[
\frac{25|b|_{\rm bytes}}{N_0}+100D_s(b)+\sqrt{10D_p(b)}
\right].
\]

For a finite archive language this is an integer, nonincreasing staircase; it
need not be convex. If `G` itself is selected for this video, its video-specific
description must be charged:

\[
R_x(D_s,D_p)=\min_{G,b}\bigl[L_{\rm video}(G)+|b|\bigr].
\]

Declaring `L_video(G)=0` for a per-video decoder makes the problem degenerate:
the target can simply be hard-coded. Rule 118 is the semantic constraint that
prevents this. Description-class comparisons must therefore pre-register the
generic receiver and count every video-derived fitted parameter, regardless
of whether it is spelled as data or code. In two-part MDL language the contest
waives genuinely generic `L(G)` but not video-derived `L(G_x)`.

Shannon rate--distortion appears only after declaring an ensemble `Q_X` and
allowing asymptotic coding. Its indirect/task version is

\[
R_Q(D_s,D_p)=\inf I(X;B)
\]

over codes whose legal reconstruction `G(B)` satisfies both distortions. By
data processing,

\[
R_Q(D_s,D_p)\ge R_{(Y,T)}(D_s,D_p)
\ge\max\{R_Y(D_s),R_T(D_p)\}.
\]

This output-space RDF is a lower bound, not an achievable contest code: a
fixed legal right inverse from approximate `(Y,T)` to uint8 videos is still
required. Joint Seg/Pose realizability does not follow from separately
encodable outputs.

For a declared random five-class output process `Y^N`, Fano gives

\[
R_Y(D_s)\ge H(Y^N)-N_s\left[h_2(D_s)+D_s\log_2 4\right].
\]

For iid uniform labels this expression is exact, and its marginal ideal rate
is

\[
-\frac{dR}{dF}=\log_2\frac{4(1-D_s)}{D_s}.
\]

It is tangent to the contest budget `8W=10.18486572265625` bits/flip at
`D_s=0.0034246802925893974`. This is an illustrative model, **not** a bound
for the actual structured 600 scorer maps.

For sparse corrections to a fixed base, choosing (k) sites from a declared
candidate set of size (M), with one of four replacement labels, has the exact
fixed-length enumerative cost

\[
L_{\rm exc}=\left\lceil\log_2 {M\choose k}\right\rceil+2k
\]

bits before byte padding and container overhead. Its ideal marginal
location-plus-label price is roughly

\[
\log_2\frac{M-k}{k}+2\quad\text{bits/correction}.
\]

Thus restricting (M) to a scorer-derived separatrix tube is not cosmetic:
it changes the combinatorial rate. At break-even, the marginal price must be
below (8W=10.1848657) bits per *realized* correction. Corrections attempted
but lost through the receiver still consume bits and must remain in the
denominator.

The campaign gap permits a sharper model check. For `N_s=117,964,800` and
`k=473,651`, uniform arbitrary support alone has ideal log-cardinality

\[
\log_2{N_s\choose k}=4{,}452{,}361.27\ \text{bits}
=556{,}545.16\ \text{B}=9.40009\ \text{bits/fix}.
\]

The corresponding fixed-length integer stream occupies 556,546 B. Only
`0.78478` ideal bits/fix remain before renderer and container overhead. Adding
a uniform one-of-four destination class costs another 2 bits/fix, for
674,958 B (`1.42501 B/fix`) after bit/byte rounding, above `W`. This does
**not** prove that every pixel-exception code is dead: location-only already
fits, and real destination classes may be predictable. With a uniform 2-bit
destination and zero other overhead, the strict 603,008-B integer budget
permits a decoder-known candidate set up to `M=50,942,609` sites
(`43.1845847%` of the grid). What is proved is narrower: unconditioned
full-grid support **plus** uniform destination labels cannot pay; separatrix
conditioning, grouped controls, predictable classes, temporal sharing, or
another source of conditional entropy reduction is required.

Pose is a separate low-output-dimensional inverse problem. The evaluator sees
only 3,600 real scalars; the frozen Pose final affine map from 32 hidden values
to its scored first six outputs has rank six (singular values 8.7686 down to
0.3533). Directly describing quantized target scalars costs
(3{,}600q/8) bytes before prediction/entropy coding, but those target values
alone are not a solution: the receiver must render two RGB frames whose frozen
Pose trunk produces them. As with Seg, output dimensionality supplies the
right description target but not the public-wire inverse.

One frozen decomposition is nevertheless exact: frame 0 is Seg-null while
Pose consumes both frames. Conditionally, if a stable frame-0 pose right
inverse exists,

\[
R_{\rm joint}\le R_{\rm seg,frame1}+R_{\rm pose,frame0\mid frame1}+R_{\rm mux}.
\]

This is an upper-bound architecture, not a measured witness result. It is
falsified by a rank-deficient or ill-conditioned `d Pose / d frame0` map, or
by failure after uint8 and exact receiver replay.

## 2. Answers after Round 1

### DE1-1 — what sets `W`

`W` is the evaluator's exact fixed-pose willingness-to-pay. It is made only of
the contest weights and denominators: `4*N0/Ns`. Calling it “measured” is
acceptable only in the narrow sense that `N0` is the measured frozen source
size; `W` itself is derived. A codec cannot halve `W`. It can halve its achieved
byte cost per realized correction. The 603,009 B value is the rounded
score-equivalent worth `473,651*W`; the strict fixed-pose integer spend cap is
603,008 B. Neither number predicts the residual's description length.

### DE1-2 — the dimensional question

**Yes for the assignment; conditional for a legal witness.** A general exact
boundary representation costs
`O(L log(E0/L)+R+topology)` bits and a coherent-chain subclass costs
`O(L+K log A+R)` rather than `O(A)`. Smooth lossy contours give the explicit
`B(F) proportional to F^-1/2` upper-bound shape above. The dimensional claim
fails for area-scale topology and, on the public wire, whenever the fixed
contour-to-RGB section needs area-scale video-specific texture.

### DE1-3 — description-space `R(D)`

The right object is the fixed-receiver, single-object staircase
`R_{G,x}(D_seg,D_pose)` plus the exact contest scalar envelope. Shannon/indirect
RDF is a conditional ensemble lower bound. Task `#611` should own this
formalism only as a proposal-and-bound layer; `J8F` must turn a proposal into a
counted receiver application, and only a whole-object n600 exact finite secant
can admit it.

### DE1-4 — leverage from the frozen affine head

Beyond omitting weights, the head gives:

1. an exact four-coordinate decision quotient and 140 local feature-patch null
   directions;
2. exact winner/rival normals, margins, and tie distances for encoder-side bit
   allocation;
3. a stable periodic feature-space right inverse for arbitrary quotient fields;
4. an exact universal five-cell fan and a canonical piecewise-constant quotient
   target with narrow transition tubes.

It gives **no free class-adjacency sparsity**: full quotient rank makes all five
cells full-dimensional and every class pair generically adjacent. It gives no
spatial contours, topology, zero-padded global surjectivity, RGB-trunk preimage,
or Morse--Smale flow. The missing object remains a legal fixed RGB section.

## 3. Corpus and receipt diff

### 3.1 Coverage and scope

The delegated snapshot reported an index covering about 7,398 of 9,706
research units (`76.2209%`). The live tool at this branch reported 7,387
research records. I conservatively scope every corpus negative to the smaller
`7,387 / 9,706 = 76.1076%` denominator. The scoped queries additionally
consulted `equations(869)`, `dag(915)`, and `tasks(417)`. An all-store census
reported `memory(2060)`, `docs(96)`, and `council(0)`, but no absent hit is
treated as corpus-wide exhaustion. Queries covered DE1-1..4, `#151/#611`,
`#573`, `#620/#650/#651`, `#662/#663`, contour/region MDL, quotient/level-set,
rank-4 inverse geometry, PR130/pi1, and J8F.

### 3.2 Agreements

- `ddm_sg2` independently recomputes the same `W` from the same components.
  Its integer counts reproduce the 603,009 B rounded worth exactly. This
  corroborates the arithmetic, not an information limit.
- The `#155` quotient/setoid work and `#151` indirect-RD line agree that one
  should pay for a shortest receiver-reachable representative, not RGB
  reconstruction. Their missing receiver section is exactly the gap in
  `R_output,x <= R_G,x`.
- `generator_description_crux_synthesis_20260719.md` already measured a
  133-B head packet and stopped because the spatial/RGB pullback was absent.
  That is direct corpus corroboration of DE1-4: packet polishing cannot solve
  trunk inversion.
- `codex_findings_ddm_dv1_description_vocabulary_20260723_codex.md` is the
  strongest n600 assignment-space evidence. Its selected event-plus-worldsheet
  composition is 134,216 counted bytes (132,606 existing bytes plus a real
  1,610-B joint section) and describes 70.5352453% of its own Road-error
  denominator in semantic cell space. Its verdict explicitly says receiver
  realization is owed. Those bytes and errors are not combined with the DE1
  gap because the bases and denominators differ.
- The min-description lattice line (`#662/#663`, G21) correctly treats dense
  lattice objects as encoder teachers and requires a newly compiled shortest
  selected-solution description. Exact feasibility does not imply short
  description.

### 3.3 Material disagreements and supersessions

- The DE1 question asks whether coder, representation, or information “sets”
  `W`. None does. `ddm_ix2`'s supplied `346,478 raw -> 346,483 Brotli` result
  only closes coder redundancy for that serialization. Its 5,184-to-5 layout
  comparison instead points to representation/modeling leverage.
- `src/tac/boundary_math/contour_codec.py` is not the explicit boundary-edge
  codec its module prose claims. The implementation serializes every uint8
  label in raster order and LZMA-compresses the dense array. It is reversible
  and may empirically track boundary complexity on piecewise-constant maps,
  but it does not establish a 1-D wire representation or the stated
  edge-plus-region-label mechanism. MAIN should review this name/mechanism
  mismatch under NO-FAKE; this research arm does not modify the code.
- The n600 `#307` receipt really codes a post-hoc mismatch map at 361,953 B /
  441,329 flips = 0.8201 B/flip, with 142,270 components. That is below today's
  fixed-pose `W` but above its own preregistered 0.65 B/flip whole-path bar. More
  importantly, it transmits **where the witness is wrong**; it does not apply
  physical RGB corrections. It is target-map coder evidence, not a candidate.
- The RA1 memo said J8F was owed on 2026-07-24. The later
  `codex_findings_ddm_j8f_counted_application_20260725T023547Z_codex.md` and
  `tac.optimization.ddm_dm4_j5_counted_application` supersede that state: a
  typed one-quantum counted application and exact n600 joint-delta function now
  exist. The measured 12-step row is bounded advisory application evidence,
  not a description-class or frontier result. Task-store rows for `#611` and
  `#573` remain incomplete and are the correct consumers of this memo.
- “Affine head = Morse--Smale complex” is not an identity. Power/Laguerre and
  tropical forms are exact at the terminal-feature head; Morse--Smale requires
  extra potential/flow hypotheses. No such hypotheses were found in the
  queried 7,387/9,706 corpus slice.

## 4. Primary literature and OSS race

| Source | Legitimate import | What it does not prove here |
|---|---|---|
| [MPEG Video Coding for Machines](https://www.mpeg.org/standards/Explorations/34/) | Machine-task bitstreams may jointly code video and extracted features instead of optimizing only human fidelity. | No fixed-scorer, single-video, legal RGB right inverse or contest byte bound. |
| [Schuster and Katsaggelos, optimal polygonal boundary coding](https://pubmed.ncbi.nlm.nih.gov/18267376/) | Dynamic programming over an admissible polygon band is the right operational “minimum bits at distortion” pattern. Replace geometric distortion with exact realized scorer disagreement. | Published geometric shape distortion is not SegNet argmax disagreement and does not include the nonlinear trunk. |
| [Zheng, Cheung, and Florencio, context-tree contour coding](https://arxiv.org/abs/1604.08001) | A real contour-symbol probability model and DP can jointly optimize contour rate and distortion. | Its chain distortion and contexts do not create an RGB witness or establish our bytes. |
| [Yang et al., decoder-guided contour refinement](https://arxiv.org/abs/2607.26426) | The most directly aligned new mechanism: derive a coarse contour at the decoder and transmit sparse ordered anchor refinements. This validates Rank 1's conditional side-information form. | External reported savings are not PACT calibration; our decoder must derive its coarse contour without a scorer and survive the frozen trunk. |
| [Vereshchagin and Vitanyi, individual-data algorithmic R-D](https://arxiv.org/abs/cs/0411014) | A single object's algorithmic R-D curve can have non-Shannon shapes; fixed-description balls are the proper conceptual object. | Kolmogorov complexity is uncomputable and gives no numeric contest floor. |
| [Liu et al., indirect semantic R-D](https://arxiv.org/abs/2201.12477) | Formalizes joint semantic/observation distortion and conditional reverse-waterfilling for suitable Gaussian ensembles. | Convex ensemble RDF does not make the finite one-video staircase convex or supply a right inverse. |
| [Grohs et al., alpha-curvelets](https://arxiv.org/abs/1404.1043) | Curve-adapted systems achieve near-optimal approximation of piecewise-smooth cartoon images, supporting boundary-aligned residual bases under their model. | L2 cartoon approximation does not imply low argmax error, counted bytes, or RGB-trunk reachability. |
| [CGAL regular triangulations](https://doc.cgal.org/latest/Triangulation_2/index.html) | Mature exact-predicate OSS can implement generic weighted-site/power geometry offline. | The Seg head's power diagram lives in feature-patch space; CGAL does not turn it into an `(x,y)` partition or a legal renderer. |

The literature changes no authority label. It strengthens the conditional
mechanisms and supplies algorithms, not a measured byte-closed row. No
HNeRV/PR95/110/128/130 mechanism, carrier, or calibration constant is adopted.
PR130 remains only the supplied external existence proof that a better point
exists.

## 5. Ranked, falsifiable description classes

Let `r=-Delta F` denote **realized n600 receiver-surviving** corrections, not
attempted edits. Every empirical class is admitted only by the pose-adjusted
joint inequality in Section 1.2.

In the table, `B_section` is only the irreducible **video-specific** payload
needed by an RGB/Pose right inverse; the source-independent section algorithm
belongs in `inflate.py` and costs zero archive bytes. `B_key/B_top/B_exc/B_pose`
are counted keyframe, topology, exception, and pose payloads. For the power
grammar, `K` is the transmitted site count; `q_xy`, `q_w`, and `q_c` are bits
per coordinate, weight, and class/facet record; `E_event*q_e` prices
temporal topology events. All such fields are properties of the `(field,
archive)` pair and remain counted when video-fitted, however they are spelled.

| Rank | Description class | Derived byte/accuracy law | Present evidence and public-wire crux | Pre-registered falsifier | Named consumer |
|---:|---|---|---|---|---|
| 1 | Receiver-known temporal separatrix/worldsheet plus ordered anchor/control innovations and a fixed joint RGB section | `B = B_key+B_top+sum_t P_t/(8 ell_t)*ceil(log2(1+2U_d/q))+B_exc+B_section+B_pose`; `F <= sum_t P_t(A_2 ell_t^2/8+q/2)+F_grid+F_top+F_section`; price is `Delta B/r` | DV1 gives n600 semantic-cell reach at real counted bytes; decoder-guided contour literature supplies the side-information construction. The fixed non-scorer contour-to-RGB/Pose section is absent. | On a same-base n600 ladder, `Delta B/r` never beats adjusted `W`; or nested boundary-tube ablations show material receiver-consumed video payload or correction debt remains over region interiors. | `ddm_ra1::task_573_ddm_generator_description` -> `ddm_ra1::task_611_scorer_recursive_proposals` -> J8F `ddm_dm4_j5_counted_application`; exact-byte audit in `ddm_min_description_contract`. |
| 2 | Temporal spatial power/curve grammar: sites, weights, class facets, ego warp, topology events, then a counted inverse residual | `B approx [K(2q_xy+q_w+q_c)+E_event*q_e]/8+B_res+B_section`; scorer price is the whole-object finite secant, with boundary error controlled by the tube law | Frozen head makes the **feature-space** fan generic; CGAL can implement generic geometry. Spatial sites and their count are still a chosen video representation, and the corpus's 133-B head packet is spatially non-identifying. | Minimum site/event count or inverse residual grows with area; exact rendered cells do not match the grammar; or topology churn destroys temporal amortization. | `tac.boundary_math.power_diagram_witness`, `ddm_description_vocabulary`, `#573`, then J8F. |
| 3 | Low-dimensional joint scorer-latent generator with frame-1 Seg chart and conditional frame-0 Pose chart | For stable intrinsic dimension `k`, a local smooth/Gaussian model predicts `R(D) approx (k/2)log2(C/D)+B_innov+B_section`; an explicit q-bit latent costs `kq/8` before entropy coding | Rank-4/6 heads and frame-0 Seg nullity identify target coordinates, not intrinsic `k` or a legal section. No current own-lineage n600 curve proves the needed low-dimensional joint inverse. | Required `k`, innovation entropy, or exception bytes scale with boundary length/area; frame-0 pose Jacobian is ill-conditioned; exact ZIP secants reject the logarithmic slope. | `DDMWitnessProgramV1`, `#573/#611`, `ddm_min_description_contract`, J8F. |
| 4 | Conditional sparse exceptions around a receiver-known base/separatrix | `[log2 C(M,k)+k H(C|Gamma)+L_Gamma+L_inverse]/(8r)` B/realized fix. Full-grid uniform support+class model is 1.42501 B/fix; support alone is 1.17501 B/fix. | Exact combinatorics show why context and class predictability matter. `#307` measures target-map coding only; physical receiver application remains separate. | Exact counted application, including lost attempts and inverse bytes, is `>=` adjusted `W`, or the candidate set/class entropy does not shrink on n600. | `#611` proposal ranker -> J8F exact secants -> `taskspace_whole_archive_allocator`. |
| 5 | Iid dense five-class label assignment | Ideal five-ary packing is `N_s log2(5)/8 = 34,238,222.92 B`, or 72.2858 B per DE1 gap flip; raw uint8 is 249.0543 B/gap flip | A negative upper-bound baseline, not a lower bound for structured maps. It makes no claim about dense logits or region codes. | Reject when exact packed payload divided by realized gap fixes is not below the pose-adjusted threshold; the iid ideal already fails fixed-pose. If context compression wins, the iid premise is falsified and the stream moves to a structured class. | `tac.optimization.ddm_min_description_contract` negative-control byte audit, then `#573` class selector. |

**Verdict:** Rank 1 is the best-derived class because it matches the measured
codimension-one debt and already has n600 semantic-description evidence. It is
not yet a legal candidate. The decisive unknown is `B_section+F_section`, the
cost and error of the fixed RGB-trunk section. No other term deserves another
coder-only campaign until that section is either built or falsified.

## 6. Round-1 adversarial self-review

Round 1 materially changed the headline in ten places:

1. I initially entertained “decomposing `W`.” That is a category error. The
   decomposable object is achieved code length or its finite-secanted slope.
2. My first `Theta(L+K log A)` statement overclaimed arbitrary maps. The
   corrected general bound pays `L log(E0/L)`; linear chain cost needs coherent
   components and explicit topology.
3. “Separatrix conditioning is necessary” was too broad. Full-grid support
   locations alone fit the fixed-pose budget; full-grid support plus uniform
   two-bit destination classes does not. Conditioning can act through support,
   labels, grouping, time, or the inverse.
4. The periodic head inverse used a basis-dependent quotient and periodic
   boundary conditions. I now report both quotient spectra and refuse transfer
   to the actual zero-padded head or nonlinear RGB trunk.
5. Power/Laguerre/tropical equivalence is exact only at the terminal-feature
   head. Morse--Smale equivalence is not automatic. The public receiver also
   cannot ship or smuggle scorer weights to exploit the pseudoinverse.
6. The supposedly generic RGB section is the easiest place to hide
   video-derived content. Every prototype, texture, lookup, inverse correction,
   and chosen instruction must be counted as a `(field, archive)` property.
7. Corpus names are not mechanism proof. `contour_codec.py` currently encodes a
   dense raster through LZMA; DV1 and `#307` stop at semantic/mismatch maps;
   neither is a receiver-closed RGB contour codec.
8. A pair-level regression `B=aP+bR+cA` cannot identify area scaling here:
   `A=384*512` is constant and collinear with the intercept. On this fixed grid,
   the admissible diagnostic is an all-n600 nested boundary-tube ablation with
   exact payload/write-support accounting. An asymptotic area exponent would
   additionally require a matched multiresolution/ROI design that preserves
   receiver semantics; this memo does not claim one.
9. The contour-price rule is fixed-pose only, and a Hausdorff tube by itself
   does not bound labeled disagreement. The corrected law uses the joint Pose
   threshold plus class-preserving correspondence, reach/rectifiability,
   topology, lattice, and receiver-section terms.
10. My first rate hierarchy mixed a fixed-object operational length with an
    ensemble information floor. The revised hierarchy puts all four minima in
    one fixed-object byte regime; ensemble indirect RDF remains a separate
    conditional lower-bound construction. The dense iid row is likewise only
    a named negative-control class, not a claim about logits or region codes.

Further retained cautions:

- Codimension-one support need not have codimension-one entropy when islands,
  junctions, or topology are area-scale.
- A source RGB preimage proves existence only, not a short fixed-section
  preimage. Separate Seg and Pose preimages need not intersect.
- Shannon/Fano and Gaussian formulas require declared ensembles. They are not
  lower bounds for this one structured video without those assumptions.
- ZIP byte length is contextual and integer-valued; small marginal edits can
  compress non-monotonically. Use matched whole-archive finite secants.
- The upstream evaluator itself does not unpack `archive.zip`, compare sample
  indices, or require exactly 600 samples. These are closure gaps, not exploit
  licenses; external archive-to-inflated n600 closure remains mandatory.

### First decisive measurement, once separately authorized

Use one fixed source-independent Rank-1 receiver on the same base and run a
full n600, byte-closed quantization/control ladder. Each row must report exact
archive bytes, realized `F`, `d_pose`, boundary length/components, attempted
controls, parse-back survival, and errors inside versus outside the predicted
boundary tube. Across the complete n600 population, apply nested tube-width
ablations while keeping the receiver fixed, and record exact finite secants
when receiver-consumed video-specific payload or writes outside each tube are
removed. Do not add stratum credits: admit only matched whole-archive secants.
The class fails if material payload or correction debt remains distributed over
region interiors, `F_section` is not tube-controlled, or every pose-adjusted
secant misses `W`. This arm does not own a scorer slot and did not launch it.

## 7. Triality and handoff

- **Equations:** `W=4N0/Ns`; the pose-adjusted joint admission inequality;
  fixed-receiver `R_{G,x}`; enumerative/chain contour bounds; tube and
  displacement-rate laws; conditional-exception combinatorics.
- **DSL/apparatus:** `ddm_description_vocabulary` supplies proposal strings;
  `ddm_min_description_contract` must bind every counted receiver-consumed
  stream; `ddm_dm4_j5_counted_application.exact_joint_delta` supplies the n600
  joint finite-secant authority.
- **DAG:** `#573 description class -> #611 scorer-recursive proposal -> J8F
  counted application -> whole-archive n600 exact replay -> pointer`, with no
  skipped receiver-section edge.

## 8. Pointer-delta honesty

Pointer delta: **none**. This memo is a means, not frontier progress. It
contains no archive and no exact evaluation row.

## NEXT-IF-RESUMED

1. MAIN reviews the category correction (`W` versus achieved slope), the
   Morse--Smale correction, the `contour_codec.py` name/mechanism mismatch,
   ensemble assumptions, and all receiver-section caveats before landing.
2. If MAIN accepts the route and separately assigns a scorer lane, instantiate
   the Rank-1 fixed receiver and the exact n600 falsifier in Section 6. Do not
   spend another unit optimizing target-map coder bytes without RGB
   application.
3. Feed the formulas to `ddm_ra1::task_611_scorer_recursive_proposals` and the
   ranked class to `ddm_ra1::task_573_ddm_generator_description`; preserve J8F
   as the counted-application boundary.
4. Any future continuation rechecks both inbox files, current pointer, task
   status, and corpus denominator before claiming the route is still live.
