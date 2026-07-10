# ADVISORY — SDF Receiver Neighborhood Atlas, Hodge predictor audit, and first exact probe — 2026-07-10

**Status:** `research_only=true` · executable design, source falsification, and proof contract only ·
no training, dispatch, pointer move, process signal, live-run mutation, source patch, or actuation
authority in this unit.

**Mission:** break the frontier with Pact's original task-space SDF / level-set witness. The current
PR110/HNeRV-lineage pointer is a frozen score/control reference, not the atlas substrate and not the
mission. This unit asks a narrower, more valuable question: *what is the exact local geometry of a
byte-closed SDF witness after quantization, inflate, receiver processing, and both frozen scorers?*

**Lane:** continuation of `lane_advisory_codex_v752_v753_v8_fresh_eyes_20260710`. The registered lane
is research-only and limits writes to new `ADVISORY_*.md` artifacts.

## 0. Outcome first — what I want to explore

The highest-information next object is a **Receiver Neighborhood Atlas** around an exact, complete
SDF archive. It should measure five structures that the current loss/controller stack can only infer:

1. the **receiver injectivity radius** — the smallest quantized code/geometry step that changes raw
   pixels, SegNet cells, PoseNet output, topology, or archive bytes;
2. the **beneficial tangent cone** — legal integer directions that lower the full contest score,
   separated into Seg-only, Pose-only, joint, and rate-coupled strata;
3. the **Morse critical set** — births, bridges, holes, mergers, and junction changes that survive the
   complete archive → inflate → resize/uint8 → scorer chain;
4. the **integrability error of our predictors** — whether gradients, costates, and cheap screens form
   a coherent score potential or contain local curl/global harmonic error;
5. the **v8 Hodge debt** — which edge-carrier degrees are true potential differences and which are
   globally inconsistent cycle payload that must be removed or explicitly priced.

The companion PR128 reverse-engineering advisory classifies PR128 as an **HNeRV-family payload-polish
child**, not a new task-space/SDF representation family. Its transferable value here is receiver,
parser, exact-consumption, entropy, and custody technique—not its byte-nibbling search objective.

The smallest decisive SDF experiment is a same-pair five-state `C1 × X` cell, where `X` denotes one
quantized `Xi`/`XIP2` pose-twist atom:

\[
\{A_0, A_C, A_X, A_{CX}, A_{XC}\},
\]

where `C1` is one legal quantized step in a frame-1 FiLM code. It measures both action orders and the
complete vector

\[
Y(A)=(d_{seg},d_{pose},B,S,\text{per-pair/class effects},\text{topology},\text{hashes}).
\]

This one cell tests Seg/Pose separability, nonlinear Pose curvature, entropy-code interaction, action
order, topology survival, and a newly found terminal-finisher correctness defect. It is **not runnable
yet**: no exact contest-Linux SDF center with the current receiver identity has been identified.

After that safety cell closes, the smallest useful multi-stratum atlas is a 24-evaluation core over
five signed atom strata `{frame0-code, frame1-code, SDF geometry, texture, palette}` plus four selected
interaction squares and three inverse returns. Selection is spectral/information-optimal, not an
arbitrary top-K.

## 1. Authority snapshot and protected work

### 1.1 Canonical score control

At the refresh used by this advisory:

| field | exact value / disposition |
|---|---|
| initial derivation checkout | `540f0ff4fa291eb66829df690b40d8e47e0efd99` on `main`; matched `origin/main` at observation |
| source-evidence refresh | `c7056f5e955c78541f0fdbf14cfc32f4c45f02e2` on `main`; added verified FEED-417 receiver-bijection evidence |
| receiver-gate refresh | `805e4f8f8aa2d7be060b34e2a5dd2ad0b3bb5ec1` on local `main`; FEED-417 fail-closed gate landed |
| final pre-commit refresh | `48fc6053397be3327eb376a910db1f151a5576c4` on `main`; matched `origin/main`; pointer unchanged |
| contest-CPU pointer | `0.19108282419209976 [contest-CPU]` |
| pointer archive | `177,169 B`, SHA-256 `ad02b0124cbb3405c23d3480ac16f12b4e48cbf6f75878dd77a5e621bebd079c` |
| pointer components | `d_seg=0.00055961`, `d_pose=0.00002942`, rate `0.00471878` |
| exact archive path | `experiments/results/clickpolish_pr110_20260710/n8_validation/candidate_archive.zip` |
| sole member | `x`, `177,069 B`, SHA-256 `d1bf188ab5e98ebf843c7716e7930ba552a7347d7ef30c64aaac52cf7fd96fde` |
| pointer CUDA | **UNMEASURED for these exact bytes** |

From the printed component precision, the derived score split is

\[
S \simeq 0.05596100 + 0.0171522593264 + 0.1179695648657
  = 0.1910828241921.
\]

This archive is a PR110/HNeRV-lineage `FP11/CTXR` packet with member `x`. It is **not** an `LVLS1`
level-set packet. An SDF mutation therefore cannot be represented as a local reversible edit around
this pointer. The honest geometry has two centers:

\[
A_{ptr}\quad\text{(frozen complete control)},\qquad
A_0^{SDF}\quad\text{(future exact LVLS1 atlas center)},
\]

and the representation-level hurdle

\[
H(A)=S(A)-S(A_{ptr}).
\]

The atlas lives only inside the SDF chart around `A0_SDF`; `H` compares complete representations.

### 1.2 SDF custody currently available

The newest local SDF wire-format specimen found in this pass is
`experiments/results/levelset_packet_20260710T165204Z/archive.zip`:

| field | measured value |
|---|---|
| archive | `69,984 B`, SHA-256 `aed7b669af81df708f6adbedacc1b9c9fd5f701dbb9f7bdbc30b7d62caa1bb27` |
| member | `0.bin`, `70,689 B`, SHA-256 `e50e37051fa66e2270498cbdbac639cf218336ea108cdc60f7ca445cce2adce7` |
| member grammar | begins `LVLS1` |
| retained raw | `6,104,016 B`, which is one pair at camera resolution, not a complete 600-pair raw |
| exact full-video score | **ABSENT** |

It is a useful parser/repacker specimen, not an authority center. No atlas score claim may use it.

### 1.3 Protected active work

- `experiments/results/levelset_v752_baseline_20260710T185913Z` is a separately owned, live governed
  v7.5.2 baseline. It resumed epoch 2 → 3, reported grouped backward active, and was inside its initial
  full n600 verdict at the read-only snapshot. Its `launch.sh` SHA-256 is
  `d714ddd72ad4979f7b53a784f236b923b4fd0cf2c04a703b5f1a47ebf88b78ed`.
- The run's log declares launch git SHA `6a34b66d6966546c4a3d677dc2f70879cd54a342` and `git_dirty=true`.
  That coarse dirty bit does not prove source contamination, but it means an authoritative descendant
  must bind the exact trainer/runtime file manifest rather than treating the git SHA alone as custody.
- `clickpolish-build` has a live Modal n600 control search on the PR110/HNeRV packet. Its source,
  archive, job, harvest surfaces, and active dispatch claim are out of scope.
- The dirty `.omx/state/*` and `paper/__marimo__/` surfaces are shared partner state. This advisory
  does not edit or stage them.
- During validation, the separately owned FEED-417 refusal gate landed as `c3697bbf1` with DAG receipt
  `805e4f8f8`: the byte-close builder now fails closed when it would count a parameter group the
  receiver does not consume. This closes the self-protection half only. Teaching every receiver to
  consume `tex_trunk.*` / `decoupled_head.*` remains owned by #395/#398; this advisory does not stage or
  absorb either landing.

Literal current disposition: **observe and preserve both live lanes; atlas execution HOLD**.

## 2. Immediate source falsification — the terminal diagonal finisher is not full-score safe

### 2.1 The false invariant

Commit `a2cf1b46f` added the SDF pair-local diagonal finisher. Its current source states that a frame-1
FiLM-code click leaves `d_pose` untouched and therefore accepts a code candidate using only

\[
100d_{seg}+25B/D.
\]

The actual evaluator proves the omitted dependency:

- `upstream/modules.py:70-74` resizes **both** frames, converts each to YUV6, and concatenates them as
  PoseNet's 12-channel input;
- `upstream/modules.py:107-113` makes only SegNet frame-1-only;
- `src/tac/through_r/mc_finisher.py:756-766` asserts frozen `Xi` makes Pose constant under a code click;
- `make_through_r_code_measure()` at `mc_finisher.py:1415-1445` renders and scores only frame 1 through
  SegNet, then constructs the acceptance objective without Pose.

Frozen `Xi` does not make PoseNet invariant to frame 1. Therefore the code-axis accept path can admit a
Seg improvement whose full contest score regresses through Pose.

The exact required edge law is

\[
\Delta S = 100\Delta d_{seg}
+\left[\sqrt{10d'_{pose}}-\sqrt{10d_{pose}}\right]
+\frac{25\Delta B}{37\,545\,489}.
\]

This is a structural proof from the frozen evaluator graph. No empirical assumption is needed to
reject the pure-`d_seg` acceptance contract.

### 2.2 Manifest-conditioned locality is stricter than pair locality

Pair locality and scorer-axis locality are different claims.

- With `self_orient=false`, a frame-1 code row is pair-local, but it changes that pair's frame 1 and
  therefore may change **both** `d_seg` and `d_pose`.
- With `self_orient=true`, `tools/levelset_byte_close_and_eval.py:1609-1629` derives the direction
  features and shared hidden state `h0` from the frame-1 code; `:1638-1657` then renders both frames
  from that shared state. The same code click may change both frames before PoseNet.
- A `Xi` click changes the pair's frame 0 under the declared pose carrier. SegNet invariance is valid
  only when exact frame-1 raw equality is proved for the selected manifest.
- A frame-0 code click may be dead when a stored frame-0 carrier replaces its render, or active under
  `store_nothing`; its semantics must come from the packet manifest, not the atom name.

Thus every mutation needs a measured support declaration over

\[
\{\text{archive sections},\text{raw frames},\text{Seg cells},\text{Pose pairs},\text{classes},
\text{topology}\}.
\]

### 2.3 Per-pair Pose fabrication blocks diagonal selection

`make_byte_close_xi_pose_measure()` at `mc_finisher.py:1397-1408` currently falls back to

```text
full(eval_pairs, aggregate_d_pose)
```

when `d_pose_per_pair` is absent. That vector is not measured per-pair evidence. It cannot identify
which diagonal candidate helped which pair, estimate covariance, prove locality, or support D-optimal
selection. The factory must return `UNIDENTIFIABLE` unless a real per-pair Pose vector exists.

### 2.4 Canonical equation and recall apparatus repeat the same omission

`clickpolish_exact_gated_discrete_latent_ratchet_v1` currently says fixed archive bytes imply
`delta S = 100 delta d_seg`. Fixed bytes remove only the rate term. The pointer receipt remains valid
because the PR110 click-polish implementation recomputed both distortions and exact-gated full score;
the registered explanatory law is wrong.

A current `graph_memory_recall` query for `clickpolish fixed bytes pose term defect pure dseg` seeded
the defective equation and did not reconstruct the already committed correction in
`ADVISORY_receiver_discrete_calculus_hybrid_adjoint_build_contract_20260710.md`. This is an observed
query-specific recall miss, not a claim that graph memory is generally broken. It shows that the graph
needs contradiction/supersession edges, not only topical proximity.

### 2.5 Literal safety dispositions

| surface | disposition | reason |
|---|---|---|
| current PR110 pointer receipt | **PRESERVE** | exact full-score eval on exact bytes remains valid |
| canonical pure-`d_seg` click-polish law | **REPAIR REQUIRED** | Pose term omitted from fixed-byte delta |
| SDF `code` diagonal selection | **ACTUATION REFUSE** | Seg-only measure can regress full score through Pose |
| SDF `Xi` diagonal selection | **UNIDENTIFIABLE AS PAIR-LOCAL** | aggregate-filled pseudo-vector must be removed |
| D27b terminal diagonal stage | **HOLD BEFORE ENTRY** | both defects precede any terminal accept/rollback claim |
| graph-memory equation recall | **PARTIAL / CONTRADICTION EDGE OWED** | exact defect query returned the stale law without correction |
| optional-weight receiver bijection | **DEFAULT REFUSE FOR v7.5.3/v8** | before `c3697bbf1`, or under its explicit waiver, `out_tex_h.*` / `tex_trunk.*` / `decoupled_head.*` can be counted yet ignored through R |

This advisory does not patch those shared source surfaces.

## 3. The receiver quotient as a stratified cubical complex

Let `A` be a complete legal archive and `rho` the complete receiver identity: inflate/runtime/evaluator
hashes, dependency versions, scorer hashes, axis, hardware, batch/thread law, raw cardinality, packet
manifest, quantization law, and tie rules. Define receiver equivalence

\[
A\sim_{\rho} A'
\iff
R_{\rho}(A)=R_{\rho}(A')
\]

at a declared equality level: archive, decoded payload, raw frames, scorer outputs, or final score.
These levels are nested but not interchangeable.

The SDF archive's quantized symbols form an integer lattice. The map from that lattice through
inflate, uint8/resize, argmax, topology, and PoseNet partitions the lattice into strata:

- **raw plateau:** archive mutations decode to identical raw bytes;
- **Seg plateau:** raw changes remain inside the same SegNet argmax cell;
- **Pose plateau:** changes remain below the measured deterministic Pose envelope;
- **topology stratum:** per-class Betti/RAG/junction signature is unchanged;
- **critical face:** a quantized step crosses a receiver, argmax, or topology wall.

The receiver injectivity radii are distinct:

\[
\rho_{raw}(A;v)=\min\{|k|:R_{raw}(A+kv)\ne R_{raw}(A)\},
\]

\[
\rho_{seg},\ \rho_{pose},\ \rho_{topo}
\]

defined analogously on their authority surfaces. A single scalar radius would hide the task quotient
we are trying to exploit.

The beneficial cone is finite and receiver-defined:

\[
\mathcal C^{-}(A)=\{v\in\mathcal L_{legal}:S(A+v)<S(A)\}.
\]

Its most valuable sub-cones are:

\[
\mathcal C_{pose\mid seg0}^{-}
=\{v:\Delta d_{seg}=0,\ \Delta S<0\},
\]

\[
\mathcal C_{seg\mid pose0}^{-}
=\{v:\Delta d_{pose}=0,\ \Delta S<0\},
\]

and joint descent. These are exact finite sets within a declared atom alphabet, not smooth nullspaces
asserted from a proxy Jacobian.

## 4. Exact RDEC and interaction algebra

### 4.1 State complex

Fix one exact SDF center `A0_SDF` and receiver `rho`.

- `K0`: complete legal SDF archives with exact parse-back, inflate, component score, and custody.
- `K1`: deterministic typed mutations with complete legal endpoints.
- `K2`: commuting squares only when both orders are legal and endpoint identity is measured.

For incidence matrices `B1` (vertices × edges) and `B2` (edges × faces), require

\[
B_1B_2=0.
\]

The exact score 0-cochain `s` induces

\[
\alpha=d s=B_1^Ts,\qquad B_2^T\alpha=0.
\]

Exact score circulation on a genuinely closed loop is therefore zero. Nonzero circulation is an
apparatus/state-identity failure, not a new physical score effect.

### 4.2 Interaction, order, and loop are different quantities

For actions `a,b`:

\[
I_{ab}=Y_{ab}-Y_a-Y_b+Y_0
\]

is component interaction;

\[
C^{order}_{ab}=Y_{ab}-Y_{ba}
\]

is order dependence; and a closed independently measured edge sum is loop circulation. A differing
`AB`/`BA` destination hash means `NONCOMMUTING_NO_2_CELL`, not a face with nonzero curl.

The existing `tac.action_commutator.v1` computes the first quantity from one composite. It cannot
stand in for the other two.

### 4.3 Separate scorer curvature from mechanism interaction

Let the base components be `(D,P,B)` and

\[
\delta_a=P_a-P,\quad \delta_b=P_b-P,\quad
I_P=P_{ab}-P_a-P_b+P.
\]

Define `f(P)=sqrt(10P)`, `I_D=D_ab-D_a-D_b+D`, and similarly `I_B`. Then exactly

\[
I_S=100I_D+\frac{25I_B}{37\,545\,489}+K^{curve}_{ab}+J^{pose}_{ab},
\]

with

\[
K^{curve}_{ab}
=f(P+\delta_a+\delta_b)-f(P+\delta_a)-f(P+\delta_b)+f(P),
\]

\[
J^{pose}_{ab}
=f(P+\delta_a+\delta_b+I_P)-f(P+\delta_a+\delta_b).
\]

`K_curve` is objective curvature even when Pose effects add perfectly. `J_pose` is the scalar score
effect of genuine Pose-component interaction. Since

\[
f''(P)=-\frac{\sqrt{10}}{4P^{3/2}},
\]

two same-sign Pose improvements can be superadditive in score without any receiver coupling. The atlas
must not call that curl or neural synergy.

At the current pointer only, the derived local Pose costate is

\[
\lambda_P=f'(P)=\frac{5}{\sqrt{10P}}\simeq291.5068,
\]

while one byte costs `6.6585895e-7` score and one bit costs `8.3232369e-8`. These values are control
economics; the SDF atlas must derive its own costate at `A0_SDF`.

One corrected Seg argmax cell at `600 x 384 x 512` has control value

\[
\frac{100}{600\cdot384\cdot512}=8.4771050\times10^{-7}\ \text{score},
\]

equal to `1.273108` pointer-priced bytes or `10.184866` bits before any Pose or entropy-context effect.
This is an exact exchange-rate derivation from the score law, not evidence that a one-cell repair is
constructible.

## 5. Hodge audit of predictors, not of the exact score

The exact score edge field is conservative by construction. The useful audit is therefore the error
between exact measured edges and a predictor such as a proxy gradient, a diagonal screen, a learned
costate, or a local surrogate. For measured edge set `E`, let

\[
\widehat\omega_e=\text{predicted }\Delta S_e,
\qquad
e=B_1^Ts-\widehat\omega.
\]

In the Euclidean cochain inner product, solve

\[
e=B_1^Tu+B_2\psi+h,
\]

where `B2` maps declared faces to their oriented edge boundaries. With nontrivial vertex/edge/face
weights, replace the second term by the weighted codifferential

\[
\delta_2\psi=W_E^{-1}B_2W_F\psi
\]

and solve orthogonally in the declared `W_E` inner product. Omitting those weights while calling the
result weighted Hodge would be a metric error. The gauge, weights, and nullspace convention belong in
the receipt. The three terms have different repair meanings:

- `B1^T u`, the **gradient error**, is an integrable missed potential or calibration bias;
- `B2 psi`, the **coexact/curl error**, is local non-integrability, usually a missing interaction,
  receiver state, or illegal claim that two actions commute;
- `h`, the **harmonic error**, is a global cycle inconsistency, hidden state, or an incompletely
  measured complex.

Do not Hodge-decompose `dS` itself and celebrate a zero curl; that is a tautology once every vertex
has an exact score. Also keep two complexes separate:

1. the **archive-mutation complex**, whose edges are legal packet mutations; and
2. v8's **class-incidence complex**, whose edges are shared class boundaries/carriers.

They may exchange measurements, but their boundary operators, cohomology, and physical meanings are
not interchangeable.

An apparatus error must be recorded independently:

\[
\zeta=\omega_{observed}-B_1^Ts.
\]

Nonzero `zeta` means a destination, receiver, scorer, or custody mismatch. It must not be absorbed into
the learned predictor's Hodge terms.

### 5.1 Information-optimal edge selection

Let a candidate edge have incidence vector `b_e`, measured precision `w_e`, and cost `c_e` including
exact evaluation time and byte traffic. For a selected graph,

\[
L_E=B_EW_EB_E^T.
\]

After fixing one vertex gauge, the D-optimal information criterion is

\[
\Phi(E)=\log\det L_E^{(root)}.
\]

The determinant lemma gives the marginal information of an added edge:

\[
\Delta_e\Phi
=\log\left(1+w_e b_e^TL_E^+b_e\right)
=\log\left(1+w_eR_{eff}(e)\right).
\]

The atlas should therefore begin with a spanning tree for identifiability, then add the legal chord
with the largest measured information-per-cost

\[
\frac{\log(1+w_eR_{eff}(e))}{c_e}
\]

subject to stratum coverage and safety. High effective resistance identifies poorly constrained
contrasts; chords expose predictor curl. There is no universal `K`. Stop when held-out edge residuals
reach the measured deterministic floor or the next chord's information-per-cost falls below the
registered threshold.

The precision `w_e` must come from repeated exact decode/eval behavior or a conservative registered
floor. It must never be inferred from 600 video pairs as though they were 600 independent optimizer
seeds.

## 6. Receiver information geometry and legal natural steps

Inside one fixed raw/argmax/topology stratum, collect a receiver-response vector

\[
r(q)=(d_{seg,p,c},d_{pose,p},\text{margins},\text{flip mass},\text{topology},B,\ldots)
\]

for legal quantized coordinates `q`. A finite-difference response Jacobian `J=dr/dq` defines the
positive semidefinite receiver pullback metric

\[
G_R=J^TW_RJ,
\]

where `W_R` states the scale/noise model of every response coordinate. This is the appropriate local
information geometry. The concave Pose term makes the Hessian of the scalar contest score indefinite
or negative in places; that Hessian is not a Riemannian metric.

For terminal score coordinates `(d_seg,d_pose,B)`, the exact local covector away from `d_pose=0` is

\[
\lambda=
\left(100,\frac{5}{\sqrt{10d_{pose}}},
\frac{25}{37\,545\,489}\right).
\]

If `J_score` maps legal coordinates to those components, the pulled-back score covector is

\[
g=J_{score}^T\lambda.
\]

The continuous chart suggests the natural direction

\[
v_*=-G_R^+g.
\]

But `v_*` is not an archive mutation. The executable decision is a constrained lattice problem:

\[
\min_{v\in\mathcal L_{legal}}
\|v-v_*\|_{G_R}^2
\quad\text{subject to parse-back, support, byte, topology, and trust-region gates},
\]

followed by exact inflate and full-score evaluation of every admitted endpoint. At an argmax,
quantization, topology, or manifest wall, start a new chart and use finite differences; do not extend a
smooth Jacobian through a discontinuity.

### 6.1 Symmetry is a proved quotient, not a convenience

Only a transformation group `G` for which

\[
R_\rho(g\cdot A)=R_\rho(A)
\quad\text{and}\quad
B(g\cdot A)=B(A)
\]

has been measured may be quotiented out. Neuron permutations, sign/scale reparameterizations, palette
permutations, pair permutations, and class relabelings are not automatically legal packet
symmetries. The frozen semantic labels, temporal order, entropy coder, and manifest usually break
them. Quotienting a merely architectural symmetry can erase a real byte or scorer direction.

The resulting receiver space is best treated as a **stratified cubical/orbi-complex**: smooth
approximations are local charts; verified finite symmetries create quotient cells; quantization,
argmax, and topology events create singular faces. This is why exact mutation squares and chart
transition receipts matter more than a globally smooth manifold story.

## 7. Typed legal atom grammar

Every atlas edge must compile from a typed mutation, parse back into the same declared grammar, and
emit a support fingerprint. The current useful atom families are:

| id | legal primitive | expected support | mandatory qualification |
|---|---|---|---|
| `C0(p,j,s)` | `code[2p,j] += s`, `s in {-1,+1}` quantized unit | pair `p`, nominal frame 0 | active only under the selected carrier/manifest; may be masked by stored frame 0 |
| `C1(p,j,s)` | `code[2p+1,j] += s` | pair `p`, frame 1; possibly both frames under self-orient | Seg **and** Pose coupled until exact invariance proves otherwise |
| `G(k,s)` | one quantized `out_sdf` weight/bias symbol | potentially every pair/class/topology cell | global geometry atom, not pair-local |
| `T(k,s)` | one quantized `out_tex` symbol | potentially every pair and both scorers | new chart once texture is active |
| `P(k,s)` | one palette symbol | every pixel selecting that palette entry | class/label and entropy effects must be measured |
| `H(k,s)` | `in_proj`, FiLM, or hidden-network symbol | global, highly coupled | positive control; not an initial descent actuator |
| `L(p,k,s)` | one decoded `LBND2` absolute quantized symbol | manifest-declared pair/frame boundary carrier | exact re-encode and raw support required |
| `X(p,k,s)` | one decoded `XIP2` twist symbol | frame 0 of pair `p` by intent | real per-pair Pose mandatory; frame-1 equality must be hashed |
| `TH(k,s)` | one v7.5.3 `out_tex_h.*` symbol | widened A2 texture head; currently no canonical receiver support | refuse until manifest-aware consumption, shape parity, and through-R positive control |
| `TT(k,s)` | one `tex_trunk.*` symbol | **currently no canonical receiver support** | candidate chart atom only after FEED-417 counted⇒consumed bijection repair |
| `D8(k,s)` | one v8 `decoupled_head.*` symbol | **currently no canonical receiver support** | candidate chart atom only after FEED-417 repair and class-isolation proof |
| `Q_top(p,c,kind,theta)` | finite disk/ellipse/bridge/hole potential insertion | local SDF region, possibly topology-changing | design atom only; typed compiler is currently absent |

Manifest scalars such as global scale, architecture dimensions, carrier presence, or packet version are
**chart transitions**, not ordinary local atoms. A mutation that changes tensor length, decoder
semantics, entropy partitions, or receiver identity opens a new complex and cannot share local
derivatives with the old chart.

`T`, `TH`, and `TT` are deliberately distinct. `T` mutates the legacy receiver-consumed linear
`out_tex` symbols. `TH` names A2's widened hidden texture head, and `TT` names A3's trainer-side texture
trunk. Current canonical NumPy/byte-close/inflate code does not consume `out_tex_h.*`, `tex_trunk.*`,
or v8 `decoupled_head.*`.

Before `c3697bbf1`, or only under the explicit `TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS` waiver, those
groups can be serialized and charged while remaining inert. Current default behavior refuses them
before archive construction. A dynamic `P[name]` access makes static analysis advisory rather than
refusing. Atlas/scored authority must reject both a waiver and `dynamic_access=true`: an AST reference
proves only syntactic consumption, not branch activation or output effect. Every optional family still
needs a through-R mutation positive control. The live v7.5.2 shared-head baseline uses none of these
groups and is unaffected by this formulation-level blocker.

Each measured atom fingerprint must include at least:

\[
\phi_e=(\Delta B,\Delta raw,\Delta d_{seg,p,c},\Delta d_{pose,p},
\Delta\beta,\Delta RAG,\Delta junctions,\Delta margins,\text{support hashes}).
\]

This fingerprint is the input to pivoted-QR/D-opt selection, locality tests, and held-out predictor
audits. A scalar `delta S` alone is insufficient system intelligence.

## 8. First exact experiment — five-state `C1 × X` safety cell

### 8.1 Admission prerequisites

The cell is admitted only after all of the following exist:

1. one complete n600 `LVLS1` archive with exact contest-Linux CPU component receipt;
2. archive, member, raw, packet-manifest, inflate/runtime, scorer, dependency, source, and hardware
   hashes, plus three fresh deterministic baseline decodes;
3. a typed reversible `C1` mutation and typed reversible `X` mutation with exact parse-back;
4. real per-pair `d_pose`, real per-pair/per-class `d_seg`, Seg argmax/margins, and topology/RAG output;
5. enough certified SSD scratch for sequential decode/eval, with success-only cleanup and durable
   receipts; and
6. zero unwaived counted-but-unconsumed groups, `dynamic_access=false`, no
   `TAC_ALLOW_UNCONSUMED_ARCHIVE_GROUPS`, and a through-R mutation positive control for every admitted
   optional family; and
7. no ownership collision with the live v7.5.2 or click-polish lanes.

Let `c=C1(p,j,s_c)` and `x=X(p,k,s_x)` be selected from the same pair by a deterministic fingerprint
rule declared before exact scoring. Evaluate:

| state | construction | purpose |
|---|---|---|
| `A0` | exact center | custody/noise reference |
| `AC` | `c(A0)` | frame-1 Seg/Pose support |
| `AX` | `x(A0)` | twist Pose support and conditional Seg invariance |
| `ACX` | `x(c(A0))` | component interaction, entropy interaction, first order |
| `AXC` | `c(x(A0))` | endpoint commutativity/order audit |

For every state record the complete `Y(A)` vector, archive/member/raw hashes, exact bytes, full raw
cardinality, receiver identity, component and full score, per-pair/class effects, Seg margins/argmax,
Betti/RAG/junction signature, and wall-clock/resource measurements.

### 8.2 Decisive falsifiers

- If `AC` changes `d_pose`, the old pure-Seg code law is empirically falsified as expected; the edge
  remains usable only under the joint objective.
- If `AX` changes frame 1, its claimed support is false for that manifest; do not call it Pose-only.
- If `ACX` and `AXC` do not have identical complete destination hashes, record
  `NONCOMMUTING_NO_2_CELL`; do not compute a face curl.
- If destination hashes match but independently measured edge circulation is nonzero, classify an
  apparatus/custody defect.
- If a topology signature changes, the finite contrast remains valid but the local Jacobian and
  natural metric do not cross that face.
- If either component vector is missing, the cell is `INCOMPLETE`, never filled from an aggregate.

Passing this cell does not authorize a diagonal finisher. It supplies the minimal evidence needed to
repair and test a full-score selector.

## 9. The 24-evaluation Receiver-Quotient Tangent Dictionary core

After the safety cell and source repair pass, build a five-stratum dictionary over

\[
\{C0,C1,G,T,P\}.
\]

Here `T` means the legacy receiver-consumed linear `out_tex` family. The core explicitly excludes
`TH`, `TT`, and `D8` until each has manifest-aware consumption and a through-R positive control.

Use a deterministic max-volume/pivoted-QR selection on non-authoritative response fingerprints to
choose one rank-revealing coordinate per stratum. Exact measurement then comprises:

| block | exact states | count |
|---|---|---:|
| fresh center decodes/evals | three independently inflated `A0` receipts | 3 |
| signed singletons | `+/-` endpoint for each of five selected coordinates | 10 |
| interaction/order cells | four pre-registered coordinate pairs, both action orders | 8 |
| inverse returns | three selected `A -> A+v -> A` reconstruction receipts | 3 |
| **total** |  | **24** |

The four interaction pairs are selected after singleton fingerprints by effective-resistance/D-optimal
gain subject to coverage of: code/geometry, geometry/texture, texture/palette, and one highest-residual
cross-stratum contrast. This is a design rule, not a promise that a particular coordinate is useful.

The resulting Receiver-Quotient Tangent Dictionary (`RQTD`) stores:

- signed exact component differences and full nonlinear `delta S`;
- response support and injectivity radii;
- local central differences and asymmetry;
- interaction, order, apparatus circulation, and Hodge-residual terms separately;
- topology/persistence/RAG/junction changes and chart identity;
- archive byte and entropy-section changes;
- predictor provenance, residual, evidence grade, and held-out status; and
- receiver/runtime/scorer/source/custody hashes.

For each signed atom, report the central lattice contrast and curvature

\[
D_i=\frac{Y_{i+}-Y_{i-}}{2},
\qquad
Q_i=Y_{i+}+Y_{i-}-2Y_0,
\]

without calling either a continuous derivative across a critical face. When `delta B > 0`, also report

\[
V_{bit}=-\frac{\Delta S}{8\Delta B}.
\]

For zero- or negative-byte edits, retain the full Pareto vector rather than dividing by a nonpositive
byte delta.

### 9.1 Stratified Morse/topology receipt

For every per-class SDF field and its receiver realization, store at least:

- connected-component and hole counts before and after the exact receiver;
- persistence pairs with declared filtration and pixel/field coordinate map;
- RAG edges with both incident semantic labels;
- junction degree, bridge candidates, island birth/death, and boundary displacement;
- transversality or margin-to-criticality measure where the discrete data supports it; and
- a `SAME_CHART`, `CRITICAL_FACE`, or `UNIDENTIFIABLE` verdict.

Persistence is a prioritizer, not evaluator authority. The frozen Seg argmax and Pose output remain the
score surfaces. A topological side loss earns admission only when its finite receiver effect predicts
exact score value per byte better than the registered alternatives.

### 9.2 Resource bound

One full raw video is `3,662,409,600 B`. Retaining all 24 decoded raws would consume about `81.86 GiB`
before scorer caches and duplication. The executable design must therefore stream states through a
certified SSD workspace, hash before release, retain compact per-state topology/scorer receipts, and
cold-store only explicitly selected raws. It must fail closed when the storage waterfall cannot prove
capacity.

The observed PR110 exact-eval time (`176.3 s`) is a different packet/receiver path and cannot price an
`LVLS1` atlas. A separate resumable timing smoke is required before any future 24-state launch.

## 10. Dynamic atlas overlay — the minimal Muon-boundary identification probe

The static atlas describes archive mutations. Curriculum and costate control also need a local model
of a discrete training event. The smallest honest experiment is a **six-leaf matched checkpoint
replay**, not a two-arm Muon/AdamW comparison.

From one complete, hashed full-state checkpoint `C` preceding the earliest tested boundary, derive the
nominal Muon event epoch `e_*` from the baseline's actual fire receipt. Let

\[
\Delta=25\ \text{epochs},
\]

derived from the current eval/checkpoint cadence, not chosen as an aesthetic constant. Branch:

| leaf | boundary | reset mode |
|---|---:|---|
| `A-` | `e_* - Delta` | matched AdamW stage boundary |
| `A0` | `e_*` | matched AdamW stage boundary |
| `A+` | `e_* + Delta` | matched AdamW stage boundary |
| `M-` | `e_* - Delta` | actual Muon transition bundle |
| `M0` | `e_*` | actual Muon transition bundle |
| `M+` | `e_* + Delta` | actual Muon transition bundle |

The AdamW control applies every common boundary action—tau freeze, spike-window reset, matched
moment/rewarm policy, checkpoint, unchanged stage-boundary loss weights—but retains AdamW. The Muon
treatment adds only the actual `MultiOptimizer` replacement and its warm-start policy. Pose engagement
and all other events must be common and collision-free. Every leaf runs to the same absolute endpoint

\[
T=e_*+2\Delta,
\]

so the late branch has one complete cadence after its boundary.

Fit the four corner leaves:

\[
b_q=\frac{y_{q,+}-y_{q,-}}{2\Delta},
\]

\[
\widehat c=
\frac{(y_{M,+}+y_{M,-})-(y_{A,+}+y_{A,-})}{2},
\qquad
i=b_M-b_A.
\]

Hold out the center leaves and test

\[
r_q=y_{q,0}-\frac{y_{q,+}+y_{q,-}}{2},
\qquad
c_{obs}=y_{M,0}-y_{A,0}.
\]

`c_hat` must predict `c_obs` componentwise within the measured floor. Six leaves are minimal here:
four estimate event-time slopes and categorical interaction, while two independently test midpoint
linearity. Fewer leaves confound generic boundary-time drift with the optimizer reset.

### 10.1 What this identifies—and what it cannot

Muon is an epoch/checkpoint-discrete guard. A continuous saltation matrix is unjustified because the
guard has no measured differentiable denominator. For reduced state `z`, estimate only the directional
reset response

\[
v^-=\frac{z^-_+-z^-_-}{2\Delta},
\qquad
\widehat{DR_qv^-}=\frac{z^+_{q,+}-z^+_{q,-}}{2\Delta},
\qquad
j_q=z^+_{q,0}-z^-_0.
\]

This identifies `DR_q` only along the event-time/flow direction `v-`. It does **not** identify a full
state transition matrix, a full optimizer adjoint, or a full optimizer costate.

The state receipt must include model, optimizer, EMA, RNG, event-controller, stage, topology/area
duals, rate state, exact/advisory score components, real per-pair components, tau/beta/LR groups,
accepted-step fraction, conditioning, receiver topology, EMA/live gap, collision order, and every
checkpoint/config/code/receiver hash. Missing coordinates or deficient design rank are
`UNIDENTIFIABLE`, never zero.

For `N=600` real pair components and video mean `Pbar=d_pose`, terminal component costates are

\[
\frac{\partial S}{\partial d_{seg,p}}=\frac{100}{N},
\qquad
\frac{\partial S}{\partial d_{pose,p}}
=\frac{5}{N\sqrt{10\bar d_{pose}}}.
\]

Every verdict still uses the exact nonlinear score difference. At `d_pose=0`, the local derivative is
singular and only finite differences are admissible.

### 10.2 Confidence gate and present blockers

The result remains `INSTANCE_SIGNAL / PROBE_READY`, never an actuator, unless checkpoint ancestry,
three fresh decode hashes, center residuals, categorical prediction, immediate decoder/EMA equality,
real per-pair Pose, common topology chart, event order, rank, and conditioning all pass. One seed cannot
provide an actuation-grade switching UCB; pair/block bootstrap describes video heterogeneity, not
optimizer-seed uncertainty.

Current literal blockers are:

- no compatible preserved pre-Muon full-state checkpoint was identified in the live v7.5.2 lineage;
- `stageMuonStart` is saved after optimizer replacement and tau freeze, so it cannot identify the
  reset;
- the live config aligns Muon and Pose backstops at epoch 726, requiring a collision-free control or
  an explicitly stamped compound-event order;
- no typed resume-registered branch harness expresses the matched AdamW boundary and event offsets;
- existing telemetry lacks complete lineage/collision receipts;
- `record_run_costates()` reads nonexistent `.tier` rather than `CostateEstimate.status`;
- current transition-costate and cross-run-lever estimates remain `d_seg`-only; and
- the receiver-neighborhood overlay schema does not exist.

No branch of this experiment is authorized in this advisory.

## 11. v7.5.3 and v8 curriculum/costate integration

Everything in this section is a proposal contract for a later separately authorized build/launch.
Imperative mathematical phrasing describes required semantics, not present actuation authority.

### 11.1 The curriculum should follow receiver charts, not only epochs

An optimal curriculum is a sequence of measured chart transitions. An epoch number may trigger a
checkpoint, but it does not prove readiness. Each stage should expose a state predicate, terminal
receipt, and fail-closed transition:

1. **custody/identity chart:** deterministic decode, exact source/runtime manifest, and NumPy/MLX/
   Torch parity surfaces close;
2. **coarse SDF topology chart:** class support, islands, holes, RAG, area dual, and boundary margins
   become measurable without texture confounding;
3. **geometry/pose chart:** Xi and geometry carriers jointly reduce exact per-pair Pose/Seg debt under
   a full-score trust region;
4. **appearance chart:** receiver-supported chroma/texture/palette engage at a checkpoint boundary and
   open a new atlas chart; optional A2/A3 groups remain refused until their receiver contracts close;
   old numeric Jacobians are not silently reused;
5. **quantized receiver chart:** training coordinates are projected onto the exact packet lattice and
   all decisions are realized through serialize → inflate → fresh scorer;
6. **terminal discrete chart:** only full-score-safe finite mutations with exact rollback/custody may
   be considered.

Loss weights remain fixed within a stage. The costate controller may recommend the next stage,
experiment, or trust-region size; it must not mutate loss weights per step. Every stage-ending
checkpoint is complete, EMA-bearing, atomically written, distinctly named, and preserved.

### 11.2 v7.5.3: single-trunk optimum

For v7.5.3, keep the three appearance arms distinct:

- A1 legacy linear `out_tex` is receiver-supported;
- A2 `out_tex_h.*` is default-refused pending manifest-aware consumption, output-shape parity, and a
  through-R mutation positive control; and
- A3 `tex_trunk.*` is default-refused pending the same receiver/effect proof.

Then use the static atlas and six-leaf overlay in this order:

- refuse the first scored texture-trunk row until FEED-417 proves `tex_trunk.*` is both counted and
  consumed by the canonical NumPy/byte-close/inflate receiver, with a mutation positive control;
- close the exact SDF center and joint Seg/Pose measurement defect first;
- construct the geometry/code atlas with chroma and texture held fixed;
- estimate receiver metric, costate residual, injectivity radii, and topology critical faces;
- engage chroma/texture only at a stage boundary, create a new chart, and remeasure rank-revealing
  directions rather than composing old numeric effects;
- use exact component shadow prices to allocate the next measured byte among SDF, pose, texture, and
  palette carriers; and
- admit a terminal mutation only when its exact finite `delta S`, not a proxy directional derivative,
  clears the deterministic floor and rollback gate.

The six-leaf Muon result may refine event timing or invalidate a reset story. It cannot select Muon,
set a controller gain, or transplant a full costate until another seed and the confidence contract
close.

### 11.3 v8: edge-centric carriers and two distinct Hodge systems

For v8, index each receiver response by edge carrier, both incident classes, pair/frame, RAG state,
global semantic label, and packet chart. Build a spanning-tree carrier basis first; add cycle chords
only when their effective-resistance information or exact score value per byte justifies them.

Before the first scored v8 row, FEED-417 must also prove `decoupled_head.*` counted⇒consumed through the
canonical receiver. A one-class optimizer-step isolation test must show that updating one class does not
move another class's decoded field; a row perturbation alone cannot prove optimizer isolation. The
current shared pair-code means full per-class independence is not yet established. The kill predicate
also needs one stable, tested semantic definition rather than reversed config/evaluator inequalities.
Increment 1a remains a mask-level, non-promotional screen. Increment 1b is the first through-R/scored
stage and remains blocked until the decoupled head has a real canonical consumer.

On the class-incidence graph, decompose carrier state into

\[
q_{edge}=B_{class}^T\varphi+B_{2,class}\psi+h_{class}.
\]

- the potential part `B_class^T phi` is globally reconcilable class state;
- the coexact part marks local cycle disagreement or a missing merge/diff/correct interaction;
- the harmonic part is global cycle payload and must earn its bytes empirically rather than being
  mistaken for unavoidable detail.

`B2,class` exists only for explicitly declared class-interface/junction 2-cells. If the representation
supplies only a graph, the coexact term is absent and unresolved cycle flow lives in the harmonic
subspace. Inventing filled faces merely to make local curl computable would change the topology of the
model.

This Hodge decomposition does not replace the archive-mutation Hodge audit in section 5. The former
regularizes v8 representation state; the latter diagnoses predictor error.

The curriculum should preserve the specified `merge -> diff -> correct` order, open a new atlas chart
at merge, paint/texture, quantization, and reconciliation boundaries, and reset numeric response
models at each. Transfer the **schema and matched-probe method** from v7.5.3; never transfer its numeric
Jacobians, optimizer effects, topology thresholds, or costates into v8.

### 11.4 Recursive/fractal full-stack optimization, made operational

“Fractal optimization” is useful only when it means the same exact calculus recurs at nested scales:

\[
\text{archive section}\to\text{pair}\to\text{class}\to\text{RAG edge}
\to\text{boundary arc}\to\text{receiver cell}.
\]

At every level, expose:

1. a typed legal action alphabet;
2. a response Jacobian or exact finite response table;
3. the pulled-back terminal costate;
4. a positive receiver metric/noise model;
5. byte, compute, topology, and risk shadow prices;
6. a trust region and chart-transition test; and
7. an exact parent-level reconciliation receipt.

Child proposals are aggregated by pullback, not by adding unrelated proxy losses. Parent dual prices
allocate measurement and bytes downward; exact receiver effects aggregate upward. This gives a
mathematically explicit multiscale control stack without pretending a discontinuous packet/scorer
space is one smooth neural manifold.

## 12. Reusable apparatus and missing build surfaces

| need | reusable local surface | exact missing piece |
|---|---|---|
| PR110 packet mutation control | `src/tac/click_polish.py` `FrozenPacket.parse`, repack, renderer, component helpers | not an LVLS1 mutator; useful only as custody/repack pattern |
| exact PR110 remote evaluation | `experiments/modal_click_polish_cpu.py::_run_exact_eval` | packet-specific timing cannot price SDF; retained raw custody is incomplete |
| hard-region/topology seeds | `src/tac/analysis/receiver_replay_scorer_hard_regions.py`, `experiments/probe_frozen_partition_topology.py` | unified exact per-pair/class Pose+Seg+RAG+persistence adapter |
| provenance | `src/tac/provenance/builders.py` | atlas state/edge/face schema and receiver-neighborhood overlay |
| exact custody validation | `src/tac/exact_eval_custody.py` | raw hash/manifest must survive wrappers that currently delete raw after size check |
| current static atlas | `src/tac/optimization/evaluator_response_atlas.py` | versioned dynamic overlay keyed by checkpoint/reset/event/horizon/receiver/pair |
| SDF diagonal finisher | `src/tac/through_r/mc_finisher.py` | joint two-frame objective, real per-pair Pose, full-score accept, fail-closed support |
| training event implementation | `experiments/train_levelset_witness_realized_through_R_mlx.py` | pre-event full-state checkpoint and typed matched-branch replay harness |
| topological atoms | current SDF rasterizer/level-set machinery | reversible typed disk/ellipse/bridge/hole mutation compiler |
| optional receiver groups | trainer emits `out_tex_h.*` / `tex_trunk.*` / `decoupled_head.*`; FEED-417 default refusal is landed | manifest-aware consumption in every receiver, shape/effect positive controls, authority refusal on waiver/dynamic access, plus live-derived⇒counted gate |

The existing `EvaluatorResponseAtlas` expects prebuilt joint-cone/margin/Jacobian fields. It should not
be relabeled as the new Receiver Neighborhood Atlas. Add a versioned overlay keyed by

```text
checkpoint_sha x reset_mode x event_epoch x horizon x receiver_sha x pair_index
```

and preserve existing static rows unchanged.

## 13. Ranked advisory roadmap

Every item below is a proposal for later separately authorized implementation or launch. None is an
instruction to mutate the current tree, run, pointer, or dispatch state.

### P0 — correctness and custody before any atlas launch

1. Finish FEED-417's still-open receiver-consumption half with one manifest-aware forward law across
   MLX, NumPy, byte-close, and inflate. Preserve the now-landed refusal gate for any counted-but-
   unconsumed group, add the reciprocal live-video-derived⇒counted gate, reject waiver/dynamic-access
   receipts for authority, and require a through-R effect positive control for every optional group.
2. Repair the pure-`d_seg` canonical law and add contradiction/supersession edges to graph memory.
3. Replace SDF Seg-only code acceptance with joint two-frame component measurement and exact full-score
   gating; remove aggregate-filled per-pair Pose.
4. Produce one complete exact contest-CPU `LVLS1` center with fresh raw, scorer, runtime, source,
   manifest, and archive custody.
5. Add the typed receiver fingerprint/atlas schema and fail-closed scorer adapter.
6. Add strict tests proving a frame-1 code mutation is never declared Pose-neutral without exact
   evidence.

### P1 — first geometry and information acquisition

7. Execute the five-state `C1 x X` safety cell only after all P0 gates close and a later launch is
   authorized.
8. Build max-volume/pivoted-QR coordinate selection and D-opt/effective-resistance edge acquisition.
9. Execute the 24-state RQTD core only under later launch authority, with sequential certified SSD
   scratch and exact inverse returns.
10. Fit predictor-error Hodge terms and hold out chords; wire the resulting residuals into sensitivity,
   Pareto, bit-allocation, autopilot, and continual-learning consumers.
11. Build the reversible topological-atom compiler and test only mutations whose receiver survival and
    byte cost are measurable.

### P2 — curriculum and hybrid control

12. Add the versioned receiver-neighborhood overlay and exact chart-transition receipts.
13. Fix the `.tier`/`status` costate-posterior bug and remove `d_seg`-only transition-costate claims.
14. Build the resume-registered six-leaf matched-boundary harness with full pre-event checkpoints and
    collision stamping; run only after a separate launch authorization.
15. Add a second seed or conservative registered across-seed floor before any switching UCB.
16. Let the controller allocate experiments by expected score information per measured dollar/second,
    not just predicted gradient magnitude.

### P3 — v8 and frontier-scale extensions

17. Instantiate the edge-carrier class complex, spanning-tree basis, cycle-chord probes, and
    potential/coexact/harmonic receipts.
18. Measure merge/diff/correct as separate chart transitions with both incident-class and global-label
    support.
19. Build a hierarchical atlas cache whose local pair/class/RAG-edge responses reconcile to the exact
    full-video score and archive bytes.
20. Add active-set branch-and-bound over legal integer mutations using receiver-metric lower bounds,
    with every survivor exact-gated.
21. Explore representation crossover only as complete byte-closed SDF children—multiresolution SDF,
    sparse critical-point grammars, topology skeleton plus local signed-distance patches, and
    evaluator-inverse boundary carriers—not as a return to HNeRV byte nibbling.
22. Repair the public-frontier watcher to query the actual challenge repository and regenerate stale
    reports from the canonical pointer; keep public claims external/unratified until exact replay.

The most promising frontier-breaking hypothesis is that the archive should encode a sparse **critical
set plus receiver-aware reconstruction law**, not a visually faithful field: class topology skeleton,
high-value boundary arcs, pair-local pose carrier, and only the texture/chroma necessary to land in the
frozen evaluator cells. The atlas is the measurement apparatus that can distinguish this from an
attractive but false geometric story.

## 14. Literal dispositions and exact remaining blockers

| item | literal disposition |
|---|---|
| live v7.5.2 baseline | **PRESERVE / OBSERVE ONLY** |
| live PR110 click-polish control | **PRESERVE / OUT OF SCOPE** |
| current `0.19108282419209976 [contest-CPU]` pointer | **UNCHANGED CONTROL** |
| SDF diagonal code/Xi finisher | **ACTUATION REFUSE UNTIL JOINT OBJECTIVE + REAL PER-PAIR POSE** |
| exact SDF atlas center | **BLOCKED: COMPLETE N600 CONTEST-CPU LVLS1 RECEIPT ABSENT** |
| five-state safety cell | **DESIGN COMPLETE / BUILD AND LAUNCH HOLD** |
| 24-state RQTD core | **DESIGN COMPLETE / DEPENDS ON SAFETY CELL + TIMING/STORAGE PREFLIGHT** |
| six-leaf Muon replay | **DESIGN COMPLETE / BLOCKED BY CHECKPOINT, HARNESS, COLLISION, AND SEED EVIDENCE** |
| v7.5.3 numeric atlas transfer | **REFUSE ACROSS CHART TRANSITIONS WITHOUT REMEASUREMENT** |
| v8 transfer | **SCHEMA/METHOD ONLY; NUMERIC TRANSFER REFUSE** |
| v7.5.3 `out_tex_h.*` / `tex_trunk.*` and v8 `decoupled_head.*` scored rows | **DEFAULT REFUSE UNTIL RECEIVER CONSUMPTION + EFFECT PROOF CLOSE** |
| FEED-417 refusal/self-protect gate | **LANDED `c3697bbf1`; RECEIVER-CONSUMPTION HALF STILL OPEN** |
| training, dispatch, pointer move, process signal | **NONE AUTHORIZED; NONE PERFORMED** |

Exact remaining blockers are therefore:

1. FEED-417 self-protection is landed, but canonical receivers still do not consume
   `out_tex_h.*` / `tex_trunk.*` / `decoupled_head.*`; waiver/dynamic-access receipts are
   non-authoritative, and v8 class isolation/kill-predicate semantics remain unproved;
2. no complete exact contest-Linux CPU SDF center with current receiver identity;
3. no joint two-frame SDF mutation objective or real per-pair Pose contract;
4. no typed reversible manifest-aware mutation grammar with support hashes for every proposed atom;
5. no combined scorer/topology adapter retaining Seg/Pose/RAG/persistence outputs;
6. no receiver-neighborhood state/edge/face and dynamic-overlay schema;
7. no SDF-specific exact timing smoke or certified 24-state storage plan;
8. no compatible pre-Muon full-state checkpoint or collision-free matched replay harness;
9. one seed is insufficient for an actuation-grade hybrid confidence bound;
10. full `A`, full `DR`, and a full optimizer costate are mathematically unidentifiable from the
   proposed six leaves; and
11. SDF exact CUDA evidence remains separate and absent even after a CPU center exists.

## 15. Primary research anchors

The construction above uses the following primary research ideas, with authority kept below the
frozen contest evaluator:

- Jiang, Lim, Yao, and Ye, [Statistical Ranking and Combinatorial Hodge Theory](https://arxiv.org/abs/0811.1067) — gradient/curl/harmonic decomposition of edge inconsistency.
- Desbrun, Hirani, Leok, and Marsden, [Discrete Exterior Calculus](https://arxiv.org/abs/math/0508341) — incidence/cochain calculus on discrete complexes.
- Spielman and Srivastava, [Graph Sparsification by Effective Resistances](https://arxiv.org/abs/0803.0929) — effective resistance as edge information leverage.
- Röttger, Kahle, and Schwabe, [Optimal designs for discrete choice models via graph Laplacians](https://arxiv.org/abs/2208.08926) — determinant-based experimental design on graph contrasts.
- Nanda and Tombari, [Stratified Morse Theory for Cell Complexes](https://arxiv.org/abs/2601.18343) — critical behavior on stratified discrete spaces; used here as design inspiration, not evaluator authority.
- Amari, [Natural Gradient Works Efficiently in Learning](https://doi.org/10.1162/089976698300017746) — pullback metric/natural direction, adapted here to receiver responses.
- Park et al., [DeepSDF](https://arxiv.org/abs/1901.05103) — continuous signed-distance representation background; Pact's receiver-quantized task quotient is stricter than reconstruction fidelity.
- Brüel-Gabrielsson et al., [A Topology Layer for Machine Learning](https://arxiv.org/abs/1905.12200) and Byrne et al., [A persistent homology-based topological loss for CNN-based multi-class segmentation of CMR](https://arxiv.org/abs/2107.12689) — differentiable/topological signals used only as probes and curriculum state.
Local source anchors include `upstream/modules.py`, `src/tac/through_r/mc_finisher.py`,
`tools/levelset_byte_close_and_eval.py`, `src/tac/click_polish.py`,
`experiments/modal_click_polish_cpu.py`, `src/tac/optimization/evaluator_response_atlas.py`,
`tools/levelset_receiver_bijection_gate.py`, `src/tac/witness_control/costate_posterior.py`, the
v7.5/v8 SPECs, and
`ADVISORY_receiver_discrete_calculus_hybrid_adjoint_build_contract_20260710.md`.

## 16. Triality, stores consulted, and pointer-delta honesty

### DSL leg

The executable DSL consists of typed manifest-bound atoms `C0/C1/G/T/P/H/L/X/Q_top`, typed state and
edge receipts, exact support fingerprints, chart-transition records, and the six-leaf
`reset_mode x event_epoch` branch specification. No invented trainer flag is authorized.

### DAG leg

```text
canonical pointer + protected-lane refresh
  -> exact LVLS1 center custody
  -> joint Seg/Pose/topology receiver adapter
  -> five-state C1 x X safety cell
  -> typed RQTD schema + rank-revealing coordinate selection
  -> 24-state exact atlas + predictor Hodge audit
  -> receiver metric / finite score costate / active experiment policy
  -> chart-aware v7.5.3 curriculum
  -> schema-only transfer to v8 edge-carrier complex

separately:
pre-Muon full-state checkpoint
  -> six matched branches
  -> held-out midpoint/categorical validation
  -> instance-signal dynamic overlay
  -> multi-seed confidence gate
  -> controller advice only after explicit actuation authority
```

### Equation leg

The controlling equations are the exact nonlinear score edge law, `B1 B2=0`, the interaction/
order/apparatus split, the Pose-curvature decomposition, predictor-error Hodge decomposition,
D-opt/effective-resistance acquisition, receiver pullback metric, legal lattice projection, and the
six-leaf directional reset estimator. No proxy equation supersedes exact full-score finite
differences.

### STORES CONSULTED

- full `CLAUDE.md` and `AGENTS.md`, plus the current top-10 Pact Claude memory entries;
- `reports/latest.md`, frontier scan payload, lane registry, subagent-progress ownership map,
  dispatch claims, modal ledger, gradient anchors, cost-band/continual-learning posteriors, probe
  outcomes, canonical task status, current directives, and recent sister/council/design/advisory memos;
- live v7.5.2 launch/log/process surfaces read-only and the active click-polish custody/claim surfaces;
- current pointer archive/member and the newest local `LVLS1` specimen;
- exact evaluator, SDF finisher/renderer, click-polish, atlas, costate, provenance, and custody source;
  and
- the primary online research anchors in section 15.

### HISTORICAL_PROVENANCE / pointer delta

This advisory began against checkout `540f0ff4fa291eb66829df690b40d8e47e0efd99`, incorporated the
source finding at `c7056f5e955c78541f0fdbf14cfc32f4c45f02e2`, refreshed after the refusal-gate
landing at `805e4f8f8aa2d7be060b34e2a5dd2ad0b3bb5ec1`, and was finally rechecked at
`48fc6053397be3327eb376a910db1f151a5576c4`. Concurrent work may advance `main`; these hashes are
derivation anchors, not claims that HEAD remains fixed.

The canonical score pointer delta caused by this unit is exactly **zero**. No archive, run, dispatch,
trainer state, source, canonical pointer, or shared state was changed. The only owned output is this
new advisory document.
