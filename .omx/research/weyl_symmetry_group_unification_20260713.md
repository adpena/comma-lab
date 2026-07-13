# Weyl symmetry-group unification of the evaluator witness

**Date:** 2026-07-13

**Role:** SOL xhigh deep-math reader/designer

**Scope:** DESIGN / grounding / rate accounting only (`research_only=true`)

**Execution:** no launch, no scorer run, no live-run mutation, no shared registry mutation

**Pointer:** UNMOVED. Only a receiver-closed exact archive row can move it.

## Verdict first

**WORTH-AN-ARM + FEED-capstone-171-CGauge.** The exact full invariance object of the
fixed-reference distortion pair is the score-fiber groupoid

\[
  \mathfrak G_S := X\times_S X \rightrightarrows X,
  \qquad
  G_S:=\operatorname{Bis}(\mathfrak G_S)
      =\operatorname{Aut}_X(S)
      \cong \prod_{s\in\operatorname{im}S}\operatorname{Sym}\bigl(S^{-1}(s)\bigr).
\]

The named evaluator symmetries are not a direct product. They form an overlapping,
state-dependent constructive atlas inside the stricter statistic-fiber groupoid
`X x_T X`, with locally additive null translations acted on semidirectly by admissible
covariance reparameterizations and glued across SegNet tie, clamp, and topology strata.

**Quantified gap versus current V9 CGauge accounting:**

- **MEASURED, globally certified subset:** 22.696926% of camera pixels per frame are
  blind through the frozen resize chain for both scorers; this is 1,385,424 RGB
  coordinates per pair.
- **MEASURED + DERIVED, real-linear R chart:** `ker R` contains 4,924,368 of
  6,104,016 pair-RGB directions, or **80.674232%**.
- **DERIVED-EXACT on an unclipped fixed-active-set chart:** adding the frame-0 YUV6
  kernel gives 5,219,280 null directions, or **85.505674%**. This is local, not a
  global finite uint8 action.
- **DERIVED from the current payload grammar:** direct blind-coordinate saving in the
  pure-generator V9 payload is **0 B**. The compressed saving from quotienting the
  remaining evaluator fibers is **UNKNOWN**, not 80.67% or 85.51% of the archive.
- The mod-32 to mod-19 arm removes **15,600 raw int8 code-table symbols**
  (`1200*(32-19)`), but its Brotli/archive delta and exact statistic equality remain
  unmeasured.

The honest negative is **NO-GO at FORMULATION scope** for one global finite-dimensional
Lie group or a direct product of the requested named factors. The groupoid formulation
is clean and is the reformulation that survives.

## 0. Authority, dictionary, and stores consulted

Hermann Weyl's program is used in its precise form: characterize an object through the
transformations preserving its structure, then identify the invariant content. Weyl
explicitly connects objectivity to invariance under automorphisms and, following the
Klein program, studies a geometry through the invariants of its transformation group.
Source read: [Hermann Weyl, *Symmetry* (1952)](https://www.nku.edu/~longa/classes/mat115/days/resources/docs/Symmetry.pdf),
especially the discussions of automorphisms, invariants, and group composition.

No compression theorem is imported from Weyl. The rate law below is separately derived
from the orbit equivalence relation plus Shannon/MDL coding and the legal-receiver
constraint.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`;
`docs/operating_manual_craft_handoff.md`; the current canonical frontier pointer and lane /
subagent ownership state; `upstream/modules.py`; `upstream/frame_utils.py`;
`upstream/evaluate.py`; `src/tac/canonical_equations/blind_coordinate_rate_lever_20260711.py`;
the evaluator-invisibility, Pose-null, chroma-asymmetry, SegNet-margin, covariance-totality,
d_pose-mirror, Einstein-pass, flicker, defect-network, V9 naming/config, and V9 ideal-A/B
artifacts; `src/tac/boundary_math/lever_b_levelset_generator.py`; and the current V9 typed
DSL. No provider, GPU, live process, evaluator, or mutable sibling surface was actuated.

## 1. Exact scorer maps, full group, and conservative groupoid

Let

\[
X=\bigl(\{0,\ldots,255\}^{874\times1164\times3}\bigr)^{2P}
\]

be the legal decoded witness space for \(P\) pairs. For fixed source frames \(x^*\), define

\[
\begin{aligned}
A(x)&=\operatorname*{argmax}\operatorname{SegNet}(R x_1),\\
P(x)&=\operatorname{PoseNet}(Y_6(Rx_0,Rx_1)),\\
T(x)&=(A(x),P(x)),\\
S_{x^*}(x)&=\left(
\operatorname{mean}\mathbf 1[A(x)\ne A(x^*)],
\operatorname{mean}\lVert P(x)-P(x^*)\rVert_2^2
\right).
\end{aligned}
\]

Here \(R\) is the fixed bilinear camera→scorer resize and \(Y_6\) is the frozen
two-frame RGB→YUV6 preprocessor. The first exact object is the **score-fiber groupoid**

\[
\boxed{\mathcal G_S=X\times_S X
=\{(x,y):S(x)=S(y)\}\rightrightarrows X.}
\]

Composition is \((x,y)\circ(y,z)=(x,z)\), inversion is \((x,y)^{-1}=(y,x)\), and
identity is \((x,x)\). Because \(X\) is finite, its full bisection group is

\[
\boxed{
G_S=\operatorname{Aut}_X(S)
=\{g\in\operatorname{Sym}(X):S\circ g=S\}
\cong\prod_{s\in\operatorname{im}S}\operatorname{Sym}(S^{-1}(s)).
}
\tag{1}
\]

Equation (1) is the requested **full invariance group as one object**. It is exact and
global. It is also nonconstructive: a generic element is an arbitrary permutation inside a
score fiber.

For engineering, use the stricter sufficient-statistic groupoid

\[
\mathcal G_T=X\times_T X,\qquad
G_T\cong\prod_{t\in\operatorname{im}T}\operatorname{Sym}(T^{-1}(t))
\subseteq G_S.
\tag{2}
\]

Preserving \(T\) preserves every source-relative \((d_{seg},d_{pose})\); preserving only
\(S\) additionally permits error redistribution. For example, the scalar score admits
Seg error exchanges with the same total Hamming count and orthogonal rotations of the full
Pose residual vector on an equal-MSE sphere, when such output moves lift to actual frames.
Those score-only symmetries are not generated by the named null factors. Therefore the list
in the mission cannot literally be the full \(G_S\); it is an atlas of useful arrows in
\(\mathcal G_T\).

## 2. The constructive symmetry atlas

### 2.1 One formula, with overlaps explicit

On a fixed regular stratum \(\sigma\)—fixed Seg labels, fixed ReLU/clamp pattern, fixed
boundary topology—let

- \(\mathcal K_B\) be blind-coordinate replacements;
- \(\mathcal K_R\) be translations in the real-linear resize kernel;
- \(\mathcal K_{Y,0}\) be the frame-0 YUV6/chroma kernel after resize;
- \(\mathcal F_P=X\times_PX\) be the exact nonlinear Pose fiber groupoid;
- \(\mathcal C_A=X\times_AX\) be the Seg argmax-cell groupoid;
- \(\mathcal F_{phot}\) be the intersection with a specified photometric family;
- \(H_{cov}\) be the admissible decoder/coordinate-gauge action induced by ego holonomy
  \(h=\exp\xi\), not physical camera motion at fixed ground truth.

The overlap-safe constructive object is

\[
\boxed{
\mathcal G_{constructive,\sigma}
=\Big[
\mathcal K_B\vee\mathcal K_R\vee\mathcal K_{Y,0}
\vee\mathcal F_P\vee\mathcal C_A\vee\mathcal F_{phot}
\Big]_{\sigma}\rtimes H_{cov,\sigma}
\;\subseteq\;\mathcal G_T.
}
\tag{3}
\]

The symbol \(\vee\) is the **join of subgroupoids**, not a direct product. It prevents the
false sum
`blind + R-null + chroma-null + Pose-null + argmax-interior + photometric slack`, because
several factors are nested or intersect heavily.

At tangent level on an unclipped chart, a useful splitting is

\[
N_x^T=\ker dT_x
=\Big(K_R+L(K_{Y,0})+N_{pose,x}+N_{arg,x}+N_{phot,x}\Big),
\tag{4}
\]

with every dimension computed by

\[
\dim(U+V)=\dim U+\dim V-\dim(U\cap V),
\tag{5}
\]

never by naive addition. \(L\) is a chosen lift through \(R\); two choices differ by an
element of \(K_R\).

### 2.2 Factor-by-factor authority

| factor | exact mathematical object | evidence | status and boundary |
|---|---|---|---|
| blind-coordinate translations | On uint8, at least \((\mathbb Z/256\mathbb Z)^{d_B}\); in fact arbitrary per-coordinate replacements on the certified blind set | 106/874 rows and 140/1164 columns; 230,904 pixels/frame; n600 arbitrary-fill bit identity through both preprocessors | **MEASURED-EXACT** on this frozen apparatus. Its real support/tangent subspace lies in \(K_R\); it is not an independent dimension addend. |
| resize-null | additive \((K_R,+)\) in the real lift; bounded translations become partial arrows on uint8 | 820,728 null dimensions/channel/frame | **MEASURED/DERIVED-EXACT real-linear**. On bounded uint8 it is a groupoid, because clipping and lattice membership restrict which translations compose. |
| chroma-null | \((K_{Y,0},+)\) on an unsaturated YUV6 chart | 6 null atoms per 2×2 block, 49,152 blocks, 294,912 frame-0 directions | **DERIVED-EXACT LOCAL** where Y/U/V clamps do not change active set. Frame 0 is Seg-free. On frame 1 it must additionally remain inside the Seg cell. |
| Pose-null | nonlinear fiber groupoid \(X\times_PX\); tangent \(N_{pose,x}=\ker J_P(x)\) | rank 6 in a 6×589,824 frame-0 working-resolution Jacobian on 4 pairs; effective dimension 1.077 | **MEASURED TANGENT ONLY**. The 589,818-dimensional tangent complement is not a finite-width additive symmetry: measured curvature breaks it. |
| argmax-cell interior | \(\mathcal C_A=\coprod_a C_a\times C_a\), with cells split further into connected activation strata | n600 margin atlas and frozen argmax | **EXACT AS A FIBER GROUPOID**; **MEASURED occupancy proxy**, not measured cell volume. Ties are orbit/stratum boundaries. |
| photometric slack | \(\mathcal F_{phot}=\mathcal F_T\cap\mathcal P_{phot}\) for a declared brightness/chroma family | chroma sensitivity/slack observations | **INFERRED/STATE-DEPENDENT** unless a candidate is rechecked through exact \(T\). It is an intersection, not an independent factor. |
| \((\xi,R)\) covariance quotient | decoder/coordinate gauge \(H_{cov}\) with \(h=\exp\xi\); \(R\) is a fixed intertwining/measurement operator | covariance explained-fraction bracket [0.42, 0.93], event/gauge decomposition, dPose mirror | **MEASURED SUPPORT + ASSUMED/INFERRED TOTALITY**, not an exact fixed-reference scorer symmetry. Generic physical \(SE(3)\) motion changes the score. |
| phase zero-mode | \(A_b\cong U(1)\) (periodic phase) or \((\mathbb R,+)\) (unwrapped phase) per connected boundary worldtube | T1 transport law and flicker/phase artifacts | **DERIVED symmetry of the transport subaction only**. Fixed \(R\), uint8, argmax, and the source pin absolute phase, so it is broken in full \(S\). |

The measured post-covariance residual attribution—approximately 85% Movable events and
approximately 11% lane/hood gauge phase—is not another free group factor. Genuine Movable
events are **symmetry-breaking coordinates** and remain payload unless another exact
equivalence is proved. The absolute phase zero-mode is likewise payload once the frozen
measurement lattice breaks its transport symmetry.

### 2.3 Which factors commute

**Commuting on a fixed real-linear chart:**

- additions in \(K_R\) commute;
- blind translations commute with resize-kernel additions locally when no bound/wrap chart
  changes; their real support lies inside \(K_R\), so this is nesting, not a direct-product
  byte windfall;
- after choosing a right-inverse lift, \(K_R\) and \(L(K_{Y,0})\) commute locally;
- phase shifts on distinct fixed boundary components commute, giving
  \(A_{phase,\sigma}=\prod_{b=1}^{B_\sigma}A_b\).

**Not globally commuting:**

- an argmax-cell deformation followed by a Pose-fiber deformation need not equal the reverse
  composition, because either intermediate point may cross a tie, ReLU, clamp, or topology
  boundary;
- \(H_{cov}\) pushes null fields forward and permutes boundary components, so
  \(H_{cov}\) acts by conjugation. The correct structure is semidirect:
  \(A_{phase}\rtimes H_{cov}\), not a central direct product;
- photometric slack is defined by intersection with the current score fiber and changes with
  the state; it is not closed under arbitrary composition;
- topology-changing events change \(B_\sigma\), the isotropy rank, and the set of admissible
  arrows. They connect strata rather than acting inside one fixed group.

This is the precise closure finding: there is a global group \(G_S\), but the named physical
generators do **not** close as one global finite-dimensional Lie group. They close as a
stratified groupoid/pseudogroup with a semidirect covariance action on each regular chart.

### 2.4 The covariance correction: \(R\) is not a group factor

The L87 law says pair dependence must factor through ego coordinate \(\xi\) and the
measurement operator \(R\), with residuals assigned to events or gauge. That is a covariance
and sufficient-statistic claim. It does **not** make \(R\) a member of the automorphism group.
Likewise, the measured decomposition \(d_{seg}=d_{cov}+d_{gauge}\) is a decomposition of
error mechanisms, not a direct-product decomposition of \(G_S\).

- \(R\) is a fixed map chosen by the evaluator.
- \(\ker R\) supplies exact/local invariance directions.
- A coordinate change \(h\in SE(3)\) can be a decoder gauge if latent fields and \(\xi\) are
  transformed together while the decoded frames remain unchanged.
- Physical motion of the decoded frames with fixed \(x^*\) generally changes \(A\), \(P\),
  and \(S\); it is not in \(G_S\).
- A joint coordinate transformation of witness, source, and apparatus is a covariance of the
  *law*, but the contest holds the source and apparatus fixed. Only its fixed-reference
  isotropy subgroup survives as a score symmetry.

Thus current CGauge's holonomy quotient is scientifically valuable but logically different
from the evaluator-null quotient derived here.

## 3. Quantified quotient dimensions, without double counting

All dimensions below are **per two-frame pair** unless stated otherwise. Percentages are
relative to \(D=6{,}104{,}016\) camera RGB coordinates.

| surface | non-double-counted dimension/count | fraction | authority |
|---|---:|---:|---|
| certified blind RGB coordinates | 1,385,424 | 22.696926% | **MEASURED-EXACT**, n600 preprocessing bit identity |
| full two-frame resize kernel | 4,924,368 | 80.674232% | **MEASURED/DERIVED-EXACT real-linear** |
| resize kernel + frame-0 YUV6 kernel | 5,219,280 | 85.505674% | **DERIVED-EXACT on an unclipped fixed-active-set chart** |
| frame-0 Pose tangent null after resize | 589,818 of 589,824 | 99.998983% of that frame-0 working space | **MEASURED**, 4 pairs; tangent only |
| pair tangent using measured frame-0 Pose rank plus both resize kernels | 5,514,186 | 90.337017% | **DERIVED LOCAL TANGENT**, not finite invariance |
| regular argmax/Pose stratum upper tangent fiber | \(D-6=6{,}104{,}010\) | 99.999902% | **INFERRED LOCAL** from an interior Seg cell and rank-6 Pose read; no global/byte authority |

The strongest apparatus-wide finite certificate is the **22.6969% blind subset**. The
80.6742% figure is exact in the real-linear resize lift and becomes a partial uint8 action
after bounds/lattice restrictions. The 85.5057% figure adds the frame-0 YUV6 kernel only on
an unclipped chart. The 90.3370% and \(D-6\) rows are tangent facts; the measured Pose escape
test explicitly forbids promoting them to finite-width or byte claims.

### Argmax-cell “volume”

The n600 margin artifact contains 117,964,800 scored pixel instances. With the preregistered
2-logit threshold:

- 5,701,513 are fragile (4.833232%);
- 112,263,287 are in the >2-logit interior band (95.166768%);
- the margin/Fisher alignment is Pearson 0.978.

This is a **MEASURED sample occupancy**, not the Lebesgue volume of a cell in frame or payload
space. A logit gap is also not a metric distance to an input-space orbit boundary without a
declared input norm and Jacobian/Lipschitz calibration. It is a highly predictive transversal
coordinate for the tie stratum. The requested cell-interior *volume* therefore remains
**UNMEASURED**; reporting 95.1668% as volume or byte saving would be fake.

### Existing raw-pixel compression anchor

On 32 real GT frames, deleting the blind coordinates before Brotli reduced the mean stored
camera-frame section by 296,075.25 bytes/frame, from 1,438,886.5625 to 1,142,811.3125 bytes
(20.5473%). This is **MEASURED on that raw-camera section**. It is not transferable to V9's
weight/code payload, where that section does not exist.

## 4. The Weyl rate law

### 4.1 Abstract quotient law

For a random legal witness \(X\), the exact zero-distortion Shannon law for the sufficient
statistic is

\[
\boxed{
R_T(0)=H(X/\mathcal G_T)=H(T(X)),
}
\tag{6}
\]

and for the scalar score alone,

\[
R_S(0)=H(X/\mathcal G_S)=H(S(X)).
\tag{7}
\]

These are quotient identities, because \(X/\mathcal G_T\cong\operatorname{im}T\) and
\(X/\mathcal G_S\cong\operatorname{im}S\). They do **not** prove a legal codec exists.

For a deterministic contest instance and a legal decoder \(D_{recv}\), the operational law is

\[
\boxed{
L_{\mathcal G}^{D}(x)
=\min_{z:\;D_{recv}(z)\in[x]_{\mathcal G}} |C(z)|,
}
\tag{8}
\]

where \(|C(z)|\) is the exact counted archive length after the real packer. For fixed
distortion cells, contest optimization is precisely the search for the shortest receiver-
closed orbit representative.

The wording “orbit representative plus symmetry-breaking coordinates” needs one refinement:
the payload should encode the **orbit label/transverse symmetry-breaking coordinates**. A
generic decoder supplies a canonical representative. Coordinates *along* the orbit are free
and must not be stored. If the canonical section

\[
\sigma:X/\mathcal G\longrightarrow X,
\qquad \pi\circ\sigma=\operatorname{id},
\tag{9}
\]

requires the source frames, SegNet/PoseNet weights, GT argmax, or an out-of-band video-derived
table, it is illegal or unclosed and equation (8) blocks the byte claim.

### 4.2 Why dimension is not bytes

Dimension controls the number of local real coordinates before quantization. Archive bytes
also depend on:

- which orbit directions are actually represented by serialized V9 parameters;
- finite argmax/Pose-cell reach, not merely tangent nullity;
- quantizer step and integer lattice;
- entropy/correlation after gauge fixing;
- Brotli/container interactions;
- a receiver-computable section and exact parse-back.

Therefore no legitimate formula multiplies `archive_bytes × 0.855057`. The quotient bound
must be pulled back through the decoder and packer first.

## 5. Gap versus current V9·CGauge accounting

### 5.1 What CGauge currently quotients

The current master action represents

\[
z_p=\operatorname{Hol}_{\xi_p}(\bar z)\oplus\phi_p\oplus e_p,
\]

with a single covariant trunk, gauge/phase and event residuals, plus a dedicated low-dimensional
Pose \(d\xi\) channel. Its archive estimator int8-quantizes learned weights and the
`(2P, mod_dim)` code table, then Brotli-compresses base weights and codes. The deterministic
bank is regenerated at decode.

So the prompt's “only by \((\xi,R)\)” is directionally right about the missing evaluator
quotient, but literally incomplete: CGauge also names phase and event residuals. More
importantly, \(R\) is apparatus, not a group coordinate. Current accounting does not attach an
exact score-fiber basis, overlap rank, or receiver section to each serialized payload section.

### 5.2 Honest gap table

| quantity | result | interpretation |
|---|---:|---|
| free raw camera directions | 22.6969% certified; 80.6742% real-linear \(R\)-kernel; 85.5057% unclipped joint-preprocessor chart | **dimension gap**, not V9 bytes |
| current V9 bytes mapped to camera blind coordinates | **0 B** | **DERIVED from payload grammar**: it stores weights/codes, not camera pixels |
| current exact compressed scorer-fiber quotient saving | **UNKNOWN** | needs decoder pullback + packer + receiver-closed exact A/B |
| mod32→mod19 code table | \(2\cdot600\cdot(32-19)=15{,}600\) raw int8 bytes/symbols | **DERIVED raw-section delta**; not yet an exact Brotli/archive delta and not yet proved a score symmetry |
| prior D18 estimate | approximately 40%, “~7 KB class” | **ESTIMATED/DESIGN**, not an exact row |
| task-452 phase section | 6,382 B smaller standalone, conditionally −0.0042495 rate units | **MEASURED but NOT a symmetry win**: all saving is header deduplication; component stream is 2,349 B worse; receiver does not consume it |

The compressed Weyl GAP versus current V9 accounting is therefore **UNKNOWN, not zero**.
The 0-byte statement applies only to direct deletion of blind camera coordinates from the
current weight/code payload; other scorer-fiber savings remain open until decoder pullback.

Reporting `0.855 * archive_bytes` as savings would be a category error.

### 5.3 Concrete WORTH-AN-ARM accounting change

Add a byte-close `weyl_orbit_accounting` row for every serialized V9 section. It must
contain:

```text
section_name
raw_symbols_before / compressed_bytes_before
groupoid_chart_id / stratum_id
exact_statistic_preserved = none | A | P | T | S
orbit_rank / transversal_rank / overlap_rank
evidence = MEASURED | DERIVED | INFERRED | ASSUMED
receiver_section_id / receiver_section_hash
raw_symbols_after / compressed_bytes_after
exact_decode_equal / exact_T_equal / exact_S_equal
archive_sha256 / axis / promotion_eligible
```

The overlap rank is mandatory: blind, `R`, chroma, Pose, and photometric factors may not
each claim the same byte. Brotli bytes are assigned only by an actual section A/B, never
pro rata from dimension.

The first no-launch accounting receipt for the present config is already determined:

```text
blind_addressable_compressed_bytes = 0
mod32_code_raw_symbols = 38_400
mod19_code_raw_symbols = 22_800
mod19_raw_symbol_delta = -15_600
mod19_brotli_delta = UNKNOWN
mod19_exact_T_equal = UNKNOWN
promotion_eligible = false
```

The actual arm is: choose a canonical V9 parameter representative within a finite,
receiver-realized `T` fiber; serialize both representatives; demand exact parse-back and
`T` equality; then report compressed bytes. An internal code/FiLM basis change that
leaves decoded frames identical is the safest first chart because it proves a decoder
gauge before attempting scorer-only changes. Its byte delta is **UNMEASURED** and is not
prejudged here.

## 6. Noether leg: phase zero-mode

### 6.1 The one-parameter subgroup

On a connected boundary world-tube `b`, let `phi_b(t,s)` be unwrapped subpixel phase. If
the transport subaction depends on `phi` only through the `xi`-covariant derivative,

\[
\mathcal A_{T1}[\phi]
=\frac12\int\!\left\|D_t^\xi\phi_b-a_b\right\|_{W_b}^2\,ds\,dt,
\tag{10}
\]

then it is invariant under

\[
\phi_b(t,s)\mapsto\phi_b(t,s)+\alpha_b,
\qquad \alpha_b\in(\mathbb R,+),
\tag{11}
\]

or `U(1)` when phase is taken modulo one lattice period. At fixed topology the group is
`prod_b U(1)_b` (or its unwrapped real cover). This is the one-parameter subgroup whose
generator is the constant field `delta phi_b = 1`.

### 6.2 Charge

The canonical momentum and Noether charge are

\[
\pi_b=\frac{\partial\mathcal L}{\partial(D_t^\xi\phi_b)}
     =W_b(D_t^\xi\phi_b-a_b),
\qquad
Q_b=\int_{\Gamma_b}\pi_b\,ds,
\qquad
D_t^\xi Q_b=0
\tag{12}
\]

between events, provided transport preserves the component measure and no source term is
present. In discrete pair time, the Euler-Lagrange equation says the transported
component sum of `W*r` is constant from one pair to the next.

### 6.3 Correction to the L87 language

The per-component integration constant `c_b` and the Noether charge `Q_b` are related but
are not the same object:

- `c_b` is the coordinate along the symmetry orbit left undetermined by a
  difference-only transport equation;
- `Q_b` is momentum conjugate to that shift;
- at a zero-residual minimizer, `Q_b=0` while `c_b` can remain arbitrary.

Thus "conserved charge = phase zero-mode" is an informative shorthand, not a literal
identity. The precise statement is: the phase zero-mode is generated by the symmetry;
the corresponding conserved quantity is its conjugate momentum.

### 6.4 Why this is not a symmetry of the full score

The fixed resampling lattice, uint8 rounding, Seg argmax boundaries, and absolute GT
phase make the full scorer depend on `c_b`. They explicitly break (11). The stored
zero-mode is therefore symmetry-breaking information selecting the receiver's orbit
representative. Topology events create or destroy boundary components and appear as
charge flux/source terms.

This unifies the findings cleanly but narrowly:

\[
\text{transport symmetry}\quad\xrightarrow{\;R/\text{argmax/GT breaking}\;}
\quad\text{flicker floor + counted absolute phase zero-mode}.
\tag{13}
\]

**VERDICT_SCOPE:** the Noether law is **DERIVED for the T1 transport subaction between
events**. Claiming it as a global symmetry or globally conserved charge of `S` is
**NO-GO at FORMULATION scope**.

## 7. Verdict ladder and reformulation queue

### 7.1 WORTH-AN-ARM

**YES:** land `weyl_orbit_accounting` beside the V9 byte-close compiler, beginning with
exact decoder gauges and section-local before/after byte receipts. The present derived
receipt is 15,600 fewer raw code symbols for mod19 than mod32, with compressed delta and
score equality explicitly UNKNOWN. This is an accounting/build arm, not launch approval.

### 7.2 FEED-capstone-171-CGauge

**YES:** the capstone should consume equations (1) and (6)-(9) as the formal gap statement:
CGauge currently factors physical covariance but has no exact scorer-fiber quotient
ledger. The feed must require a legal section (9), overlap accounting, and an exact
byte-close A/B before any pointer claim.

### 7.3 Honest NO-GOs

1. **FORMULATION NO-GO:** one global Lie group equal to
   `blind x argmax x pose-null x chroma x R-null x photometric x (xi,R)`.
   Reason: overlaps, partial actions, strata, topology changes, and covariance/invariance
   conflation.
2. **FORMULATION NO-GO:** turn frame-space null dimension into proportional archive
   bytes. Reason: V9 serializes generator parameters, not frame coordinates; entropy
   coding is nonlinear.
3. **FORMULATION NO-GO:** treat the measured Pose tangent null as a finite-width group.
   Reason: the existing finite-walk probe measured curvature leakage.
4. **FORMULATION NO-GO:** call 95.166768% robust pixel occupancy a cell volume.
   Reason: no deformation measure or camera-space metric conversion was measured.
5. **FORMULATION NO-GO:** attribute task #452's 6,382 B to zero-mode symmetry. Reason:
   its component stream is 2,349 B worse; header dedup is the measured mechanism.

No family or paradigm is killed.

### 7.4 Reformulation queue

1. Use `G_S` for the exact theorem and `G_T` for the conservative compiler target.
2. Build regular-stratum charts with explicit overlap rank; cross a tie/clamp/event only
   through a new chart.
3. Start with exact internal decoder gauges that leave frames bit-identical.
4. Then admit scorer-only fiber moves under exact `T` parse-back equality.
5. Serialize a legal receiver section and measure actual Brotli/archive bytes.
6. Only then run the declared CPU/CUDA exact evaluator axis and consider pointer movement.

## 8. Triality, durability, and pointer delta

**Equation leg:** `.omx/research/weyl_symmetry_group_unification_equation_feed_20260713.md`
contains the canonical equation feed for (1), (6), and (8). It is not registered in the
shared canonical-equation registry in this design-only unit.

**DAG leg:** `.omx/research/weyl_symmetry_group_unification_DAG_FEED_20260713.md` provides
the FEED-capstone-171-CGauge dependency/gate sequence.

**DSL leg:** proposed `weyl_orbit_accounting` byte-close row schema in section 5.3. No
trainer or typed DSL file was edited because the user requested analysis/design and live
sibling files are protected.

**Durable artifact:** this memo plus the two isolated feeds. They are intentionally
uncommitted for main review.

**Pointer delta honesty:** zero. No exact archive, scorer run, hardware-axis row, or
receiver-consumed quotient was produced.

## 9. Self-adversarial review

- The full group (1) is exact but nonconstructive; the memo does not pretend otherwise.
- The named factors generate only a subgroupoid atlas, not all of `G_S`.
- `R` is not mislabeled as a group element.
- covariance is not promoted to fixed-reference invariance.
- blind, resize, chroma, Pose, and argmax figures are not summed across overlaps.
- 85.505674% is explicitly local/unclipped; 22.696926% is the globally measured arbitrary-fill subset.
- Pose rank is tangent-only and finite-width failure is retained.
- margin occupancy is not called volume.
- the 6,382-byte phase-section result is attributed to its measured header mechanism.
- current direct blind-byte saving is 0 B; total exact archive-byte gap remains UNKNOWN.
- no result here moves the pointer.
