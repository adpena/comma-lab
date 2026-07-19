# Grid LSTM crosswalk against the v10 integer-plane vehicle

- Date: 2026-07-19 UTC
- Task: `#562`
- Lane: `grid_lstm_crosswalk_20260719`
- Status: `research_only=true`; paper-to-Pact crosswalk; no build or execution
- Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

Authority: no launch, paid dispatch, contest score, promotion, submission, or
pointer authority. This isolated-branch artifact requires **MAIN landing
review**.

## Verdict

**PARTIAL SURVIVOR.** Grid LSTM supplies a legitimate learned spatial-context
mechanism, but the paper does not demonstrate our object: a shortest exact
archive whose receiver emits two frozen-evaluator-equivalent integer planes at
`[384,512]` across 600 pairs. The only recommended `$0` probe is a lossless,
plane-content context-codec A/B after C6 exposes an exact `PLANE`/margin symbol
section. This is distinct from the #557/#558 weight-context formulations.

A dense learned 2D/3D Grid-LSTM receiver is **REFUTED_FOR_THIS_SLOT**, not as a
family theorem and not by runtime. At the current `83,838 B` generator rung, an
optimistic int8 Grid-LSTM core alone consumes essentially the entire archive,
before frame codes, output projection, grammar, or headers. It also has no
measured exact-partition, realized-through-R, or archive result. The corrected
runtime contract and full T4 parallelism make its arithmetic plausibly
receiver-viable; therefore runtime remains **UNMEASURED**, not a negative
verdict.

A tied/priority Grid LSTM may still be used **ENCODE_SIDE_ONLY** as a proposal
network whose result is projected into the existing explicit C2/quotient-
residual payload and whose learned recurrence is discarded. It is not
authorized to become another shipped trunk until the analytic
solve/seed/project chain and the already-landed quotient residual `T` have been
measured and shown insufficient.

## Ranked paper-to-Pact crosswalk

The paper citation is Kalchbrenner, Danihelka, and Graves,
[“Grid Long Short-Term Memory,” arXiv:1507.01526](https://arxiv.org/abs/1507.01526),
with equations and experiment sections taken from the
[paper PDF](https://arxiv.org/pdf/1507.01526).

| rank | paper section / mechanism | our measured artifact | verdict | explicit `verdict_scope` | falsifiable gate | consumer |
|---:|---|---|---|---|---|---|
| 1 | §3.1 grid recurrence plus §4.3 learned character context, repurposed as a lossless probability model over serialized plane/margin symbols | #557 measured hand left/up weight contexts worse than IID/Brotli (`1.41719x` base, `1.04010x` code; constriction `1.50960x`, `1.05585x`); #558 measured DeepCABAC base replacement `+1,436 B`; neither tested plane content | **ADOPT_AS_PROBE** | One conditional, lossless C6 plane/margin section only. Does not reopen the named weight coders, authorize fitted decoder state for free, or claim score movement. | On one exact rederived C6 section, count all learned context weights, tables, headers, and termination; fresh decode must reproduce identical symbols and re-encode identically; complete archive bytes must be strictly below the exact Brotli baseline; exploited receiver timing must preserve `T_inflate + T_scoring < 1800 s`. | `C6`, then `C9` |
| 2 | §3.2 priority dimension and §3.5 tied depth, reinterpreted as a training-time trunk over the already-defined C2 stages rather than a new semantic factor | measured `~3.2x` along-tangent resolution deficit and Lane long tail; later owed16 `freq_along=26` was marginally worse than OFF at every trained cell, so deficit does not establish efficacy | **ENCODE_SIDE_ONLY** | Proposal/optimizer only. The recurrence may order or propose existing `P/T` or basis coefficients, but its learned state is absent from the receiver and it may not replace the explicit rank-4/head and integer-plane contracts. | After the fixed-capacity C2 `U4` A/B and quotient `T` baseline exist, project proposals into the same exact payload budget. Admit only if fresh parse-back lowers exact `d_seg`, preserves the Pose term, and beats the non-recurrent solve/seed/project control at identical archive bytes. | `C2`, `#497` |
| 3 | §3.1/§5.2 dense 2D spatial Grid LSTM or 3D `x/y/depth` Grid LSTM as the actual plane generator | current complete generator archive `83,838 B`; target has `196,608` pixels per plane and `235,929,600` pixel-sites over 1,200 frames; no Grid-LSTM plane/archive receipt exists | **REFUTED_FOR_THIS_SLOT** | Direct replacement of the current approximately 84 KB generator before C1/C2 closure. This is a byte-and-evidence verdict, not a family or exploited-T4 timing verdict. | Reopen only with a complete two-plane archive below the comparison rung, exact factor-2 numerator verification, fresh double decode, hard Seg/Pose improvement under the marginal-admission law, and measured contest-hardware `T_inflate + T_scoring < 1800 s`. | `C2` |
| 4 | §3.1 learned context applied to shipped witness weights | #557 and #558 exact current-donor rows above; #496 also places the current witness past its tested sub-int8 rate knee | **REFUTED_FOR_THIS_SLOT** | Named current-donor learned/hand weight-context formulations only. Other weight coders and a future wider vehicle remain open. Plane-content context is row 1 and is not inferred from weight results. | A new formulation must reduce complete exact archive bytes after every fitted probability-model byte while preserving a fresh receiver and realized-through-R score. Proxy likelihood or parameter entropy is insufficient. | `#496` |
| 5 | §4.1 addition, §4.2 memorization, §4.3 character modeling, §4.4 translation, Appendix §5.1 parity, and Appendix §5.2 MNIST as transfer evidence | our target is a frozen rank-4, ten-hyperplane argmax partition; current exact factors and receiver contracts are local artifacts, not any paper benchmark | **REFUTED_FOR_THIS_SLOT** | Treating the paper’s task results as evidence that Grid LSTM improves our score, byte rate, or exact receiver. The mechanisms remain eligible under rows 1–3. | Only exact v10-plane bytes, exact receiver parse-back, realized-through-R hard metrics, and the governed total-runtime receipt can replace this task-mismatch verdict. | `C2`, `C9`, `#497` |

## 1. What the paper actually establishes

For `N` dimensions with equal hidden width `d`, §3.1 concatenates the `N`
incoming hidden vectors into `H ∈ R^(Nd)`. Each dimension owns a distinct LSTM
transform and memory vector. For each axis, the four gate/candidate matrices
have shape `d × Nd`. Therefore the core weight count is

\[
P_N = N(4d\,Nd) = 4N^2d^2,
\]

or `4N²d² + 4Nd` with conventional per-gate biases. This gives:

| grid | core parameters with biases | notes |
|---|---:|---|
| 2D | `16d² + 8d` | two distinct axis transforms; tying across positions does not tie the two axes to each other |
| 3D | `36d² + 12d` | three distinct axis transforms |

The priority-dimension construction in §3.2 delays one axis until it sees the
other axes’ new outputs. It still executes `N` transforms with the same
parameters and matrix-MAC count as a normal block, but it lengthens the block’s
dependency path. Section 3.5 permits weight sharing along selected grid
dimensions. The counts above are the maximally tied core: leaving a grid
dimension of length `L` untied creates `L` position-specific core copies and
multiplies this parameter term by `L`. The paper’s “depth as a dimension” is
consequently a useful organizational analogy for our trunk-over-layers, but
not evidence that another paid semantic trunk is needed.

The six experiments establish long dependency transport and multidimensional
credit assignment in their own domains:

- §4.1: two 15-digit integer addition;
- §4.2: memorization of random 20-symbol sequences;
- §4.3: character-level language modeling;
- §4.4: source/target/hierarchy translation grids;
- Appendix §5.1: long parity strings;
- Appendix §5.2: MNIST with `x/y/depth` recurrence.

None measures entropy-coded witness bytes, exact integer-plane generation,
`uint8`/resize survival, frozen SegNet/PoseNet cells, 384×512 decode, or the
30-minute total official-evaluation path. The image experiment is discriminative
and uses directional grid passes; it is not a demonstrated autoregressive plane
decoder. Those distinctions are why the paper supports a probe, not adoption.

## 2. Parameter budget versus the measured generator

Use the measured complete generator archive `B_gen = 83,838 B` only as an
optimistic envelope. It already contains base and pair-specific content, so a
Grid-LSTM comparison must count its recurrence, output projection, every
video-fitted state, per-frame/pair codes, grammar, and headers inside the same
archive.

If every core parameter cost exactly one byte and all non-core costs were
pretended zero, the largest equal-width cores are:

| core | largest integer `d` | core parameters | bytes left from `83,838 B` |
|---|---:|---:|---:|
| 2D int8 | 72 | 83,520 | 318 |
| 3D int8 | 48 | 83,520 | 318 |
| 2D int16 | 50 | 40,400 parameters / 80,800 B | 3,038 |
| 3D int16 | 33 | 39,600 parameters / 79,200 B | 4,638 |

These are **DERIVED optimistic upper bounds**, not designs. They omit the RGB
or plane output head, exact arithmetic metadata, entropy-model state,
frame/pair variation, and receiver grammar. At int8, either core alone consumes
`99.62%` of the measured archive rung. A smaller compressed core could fit, but
the paper provides no rate point and the remaining payload must still be
measured rather than presumed away.

The local score law is

\[
S=100d_{seg}+\sqrt{10d_{pose}}+
25B_{archive}/37{,}545{,}489.
\]

At fixed Pose, one unit of `d_seg` is worth `150,181,956 B`, or
`150.181956 B` per `10^-6 d_seg`. For an added learned payload `M`, strict
pose-neutral admission requires

\[
\Delta d_{seg} > M/150{,}181{,}956.
\]

Thus `M=83,838 B` requires more than `0.000558243` `d_seg` improvement, and
`M=61,598 B` requires more than `0.000410156`, before any additional overhead.
For row 1’s lossless context coder, distortion is unchanged; its simpler gate
is strict net archive-byte reduction after the learned context model and all
tables are counted.

## 3. Fit to the frozen argmax geometry and the Lane deficit

Our frozen Seg head is exactly rank 4 over the 144-dimensional patch, with ten
pair hyperplanes. The v10 vehicle therefore asks a generator to encode a
piecewise-constant cell partition and then satisfy exact integer preimage
constraints. Coordinate INR, power-diagram, and explicit plane grammars expose
those boundaries directly. A dense Grid LSTM instead hides partition state in
recurrent activations and emits a site at a time. The paper supplies no exact
cell, topology, or integer-numerator invariant.

The measured Lane evidence is real but does not select Grid LSTM:

- baseline along-tangent content is approximately 25 cycles/unit while the
  incumbent basis resolves at most 8, a `~3.2x` deficit;
- Lane exhibits approximately six persistent components plus a long tail of
  roughly 166 dash births per frame; small dashes below three pixels flip far
  more often than large ones, and naive matching retains only `12.4%` of Lane
  tracks for at least three steps;
- the later owed16v2 along-heavy `freq_along=26` formulation was marginally
  worse than OFF at every trained cell.

Therefore the deficit and long tail justify a long-context hypothesis, but the
negative owed16 result prevents converting that hypothesis into an efficacy
claim. A Grid LSTM also has no intrinsic tangent alignment. Row 2’s matched-byte
projection gate is the first honest way to distinguish “long context was
missing” from “the recurrence merely spent capacity differently.”

## 4. Plane/margin context is distinct from weight context

The #557/#558 results settle named **weight** probability models on the current
donor. Weight tensors are not the same source as a piecewise-constant plane,
margin map, class-pair stream, or quotient residual. Plane-side context remains
open because adjacent spatial symbols may share cell identity and boundary
geometry even when adjacent learned weights do not.

Rule 118 remains binding. Generic deterministic Grid-LSTM execution code may
live outside the archive, but every video-trained context weight, fitted table,
initial state, chosen seed that carries video information, and coded symbol is
counted. Calling the decoder generic does not make its fitted state free.

### Sole recommended `$0` probe

After C6 exposes one canonical exact `PLANE`/margin section, perform one local
lossless A/B:

1. Baseline: exact current Brotli section with complete framing.
2. Treatment: one tied 2D Grid-LSTM probability model whose width is derived
   so the complete model envelope is at most 1% of that same baseline section;
   count quantized context weights, output/CDF tables, headers, and termination.
3. Require source symbols = fresh decoded symbols = re-encoded symbols and
   byte-identical deterministic decode.
4. Pass only on strictly smaller complete archive bytes. Report decode timing
   separately and route a survivor to C9’s receiver-closed joint frontier.

This is one probe, not a width sweep and not a new witness generator. It needs
no scorer run because the decoded plane symbols are identical. If C6 cannot
provide exact source bytes, the probe is blocked rather than reconstructed from
aggregate byte totals.

## 5. Train-nothing falling rule and quotient residual `T`

The measured flat/palette versus textured realization gap is approximately
`d_seg 0.0416` versus `0.0048` (`8.7x`). That is a real unresolved target, but
it does not identify recurrence as the missing mechanism.

Task #531 has already landed a structurally unique, terminal,
class/cell-conditioned `QuotientResidualT` at factor 5 over quotient base
classes `{1,3a,3b,4,6,7,8,9}`, with frozen predecessors and a unique parameter
group. This proves custody and non-duplication, not numerical efficacy,
uint8/resize survival, scorer improvement, or rate. The exact next comparison
is therefore:

1. measure C2 with the analytic solve/seed/project chain and explicit `T`;
2. identify the residual cell/class/pair gap that survives at matched bytes;
3. only then allow a Grid LSTM to propose `T` or other existing payload values;
4. project those proposals into the same explicit receiver grammar and discard
   the recurrence.

Until steps 1–2 produce a measured remainder, a shipped trained Grid-LSTM
trunk would relearn settled factors and violate the falling rule. Row 2 stays
encode-side only.

## 6. Corrected receiver confrontation: full legal hardware

The MAIN correction dated 2026-07-19 is binding:

\[
T_{inflate}+T_{scoring}<1800\;\mathrm{s}
\]

on the chosen contest instance. `T_scoring` is part of the same total and is
measured, never guessed. The legal choices are a T4 CUDA instance with 26 GB
host RAM and 16 GB VRAM or a 4-core/16 GB CPU instance; multiprocessing,
threading, asynchronous I/O, CUDA, and deterministic per-pair parallelism are
allowed. The 600 independent pairs must be exploited before issuing a timing
verdict.

The operation count is large but not itself a T4 blocker. Ignoring biases,
heads, nonlinearities, memory movement, and depth repetitions, a maximum int8
core from §2 performs `82,944` matrix MACs per pixel-site for either 2D `d=72`
or 3D `d=48`. Over `196,608 × 1,200 = 235,929,600` sites this is
`19.5689 TMAC`, or `39.1379 TOP` when a multiply-add is counted as two
operations. NVIDIA specifies up to `130 INT8 TOPS` for T4 in its
[official T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf),
giving a purely theoretical `0.301 s` arithmetic floor. That floor is
**DERIVED, not a forecast**.

A 384×512 2D diagonal schedule has `384+512-1 = 895` sequential wavefronts per
layer, with cells inside each diagonal parallel. Batching the 600 independent
pairs supplies abundant cells per wavefront; independent plane roles may be an
additional batch axis only if C1 proves their exact distinct outputs. A tied
depth axis multiplies work by its executed depth even when it shares weights.
Exact depth recurrence prevents full layer parallelism, but diagonal pipelining
between layers is possible. Fully parallel layer streams require a non-LSTM
depth axis (§3.3) or another decoupling and are therefore a different model
whose parity must be proved. Chunking pairs bounds state, while row/frontier
buffers avoid retaining every hidden state. For example, the largest 2D
frontier at `d=72`, two axes, hidden plus memory, 1,200 planes, and fp16 storage
is approximately 253 MiB, before workspace and outputs—well below 16 GB as a
capacity check, but not a timing receipt.

The gaps between the theoretical floor and an admissible runtime are kernel
utilization across 895 wavefronts, sigmoid/tanh and cell updates, precision and
determinism, output projection, archive parse, assembly/I/O, double-decode
verification, and the measured scorer pass. No contest-hardware implementation
or timing artifact exists. The honest label is therefore:

`EXPLOITED_T4_RECEIVER_TIMING_UNMEASURED; ARITHMETIC_AND_FRONTIER_CAPACITY_DO_NOT_REFUTE`.

The dense generator remains rejected for the current slot on bytes and missing
exact receiver/score evidence, not on a naive sequential-CPU estimate.

## Evidence, operating contract, and pointer delta

**STORES CONSULTED:** delegated authority and both inboxes; full `CLAUDE.md` and
`AGENTS.md`; current frontier/lane/subagent canonical surfaces; latest sister
findings/design/council surfaces; v10 integer-plane vehicle specs; C1/C2
charters; frozen-scorer exact factorization; #531 quotient-residual receipts;
#557 arithmetic/selfcomp and #558 neural-selfcomp receipts; Lane/along-tangent
artifacts; and `docs/operating_manual_craft_handoff.md` (SHA-256
`40d157a039d4dd242bfb189d53e6b82abcc5d037adceb0a52c9bb2956903f212`).
The deep-read arXiv PDF was `847,373 B`, SHA-256
`7993018d83f17f52e89910536f14060de587b5baa8b30c886013b10fb3daa63a`.
The corrected MAIN v10 spec read for runtime accounting has SHA-256
`5e926fcdf88c9745d95f8340692e5df4bbfc24a6358cfb8acfd5a9cc53de5f46`.

The craft handoff’s relevant controls are preserved: read authority before
acting, distinguish measured from derived evidence, keep a negative verdict
formulation-scoped, avoid re-opening settled weight-coder results, land a
durable artifact, and route isolated work through MAIN review.

Triality: equations are the Grid-LSTM parameter/operation derivation, exact
score marginal law, and total-runtime inequality; DAG routing is the ranked
`C2/C6/C9/#496/#497` table and sole conditional probe; no DSL/config changed
because this is `research_only=true` and execution is unauthorized.

Pointer delta: **none**. No score, launch, paid dispatch, archive, run pointer,
or sacred run byte changed. In particular,
`experiments/results/levelset_n600_witness_20260717T113932Z/` remained
read-only.

`verdict_scope`: one lossless plane-content context probe survives; a tied Grid
LSTM may serve as a discarded encode-side proposal mechanism; a dense shipped
receiver is rejected only for the current pre-C1/C2 approximately 84 KB slot
on rate and missing exact evidence. Grid LSTM as a family, plane-side context
coding as a family, and exploited T4 receiver timing remain open.

Self-review: round 1 reconciled the paper equations and six experiments against
the exact local factors. Round 2 incorporated the MAIN total-runtime correction
and removed the naive sequential-CPU rejection. Round 3 corrected priority-axis
scheduling, tied/untied parameter scaling, and the limit on exact layer
parallelism. Round 4 normalized the memo metadata so the final branch diff is
whitespace-clean. The five-round ceiling is respected.
