# TC2: the minimal causal lane fit captures little of the GT-edge reference gain

Owner `ddm_tc2` · `[no-triality] [p0-ledger-ok]` · 2026-09-10.

**MEASURED: the real n600 twin encode saves78 B including its five new weights.**
The GT reference clears3000 ideal B, so the charter's better-predictor handoff
fires. Full conditional causal parse-back reproduces every token byte-identically.
All numbers below refer to the immutable **move37** field, not the newer live pointer.
`score_claim=false`; no scorer ran and no archive was fired.

## Full-field result

Axis: `[macOS-CPU advisory / real full-n600 counts, scorer-free]`.
Population: **117,964,800 symbols = 600 × 384 × 512**. The Lane/Road selection is
**28,098,264 symbols**; all of these happen to lie in rows 128–319 on this field.
Both totals include correct predictions, not only mispredicted tokens.

| Model | All-symbol codelength, bits | Lane/Road codelength, bits | All-symbol ideal reduction, bits / 8 |
|---|---:|---:|---:|
| Shipped TC1 full integer probabilities | 958,224.184891 | 692,286.182664 | 0 |
| Plus GT distance, optimized continuous weights | 914,381.007165 | 648,669.310713 | 5,480.397216 |
| Plus causal distance, optimized continuous weights | 957,562.013238 | 691,642.668240 | 82.771457 |
| Plus GT distance, rounded int8 weights | 914,382.014386 | 648,674.057171 | 5,480.271313 |
| Plus causal distance, rounded int8 weights | 957,561.734488 | 691,646.189674 | 82.806300 |

These are sums of realized per-token codelengths under the fitted mixer, **not
serialized payload charges**. The two continuous fits differ by 5,397.625759
ideal bytes; the minimal causal fit captures about 1.51% of the GT reference gain.
Continuous weights were optimized on the 18,294,067 retained positions, then
priced on all symbols. Consequently, rounded causal weights can slightly beat
the continuous solution in the all-symbol total; it is not the optimum of that
larger population's loss.
The oracle clears the charter's 3,000-byte ideal gate. It does not reach the
15,000-byte prior for this concrete extension.

The original 35 TC1 weights remain fixed. TC2 adds one feature per coding winner,
so the actual implementation has **five new int8 weights (5 B)**, rather than the
charter's estimated seven weights / 14 B. The continuously optimized oracle
weights round to `[44,40,28,31,28]`; causal weights to `[30,23,6,10,5]` at scale32.
No per-frame polynomial coefficients or context maps are transmitted.

## Conditional entropy, separate from fitted codelength and charges

These descriptive counts condition on the coding winner and quantized
top-probability confidence: `winner*64 + min(floor(-2*log2(1-p_max)),63)`,
optionally crossed with a distance bin. The other four probabilities are not
retained in this bucket. Plugin entropies use
the observed full-n600 counts; Miller–Madow corrections are shown separately.
The Lane/Road table additionally conditions on membership of the true decoded
symbol in that class pair. That selection is not a free decoder decision.

| Context, Lane/Road symbols in rows128–319 | Plugin conditional entropy, bits | Miller–Madow, bits | Occupied contexts / context-symbol cells |
|---|---:|---:|---:|
| Shipped probability bucket | 574,532.178685 | 574,634.610033 | 197 / 339 |
| Bucket + GT distance | 467,323.530408 | 467,633.709842 | 1,037 / 1,467 |
| Bucket + causal distance | 567,071.116634 | 567,508.253231 | 1,083 / 1,689 |

Denominator for every row: 28,098,264 actual symbols. Descriptive plugin reductions
are 13,401.081035 and 932.632756 bits/8 respectively. They are neither payload
prices nor the realizable gains of the five-weight extension.

For all 117,964,800 symbols, the corresponding plugin totals are
1,162,429.213861 / 981,892.672125 / 1,130,792.078388 bits; corrected totals are
1,162,781.231451 / 983,173.063973 / 1,131,969.317542 bits. Coarse bucketing loses
information already carried by the full TC1 probabilities: the bucket-only
entropy even exceeds the actual shipped codelength. It cannot replace that
baseline or serve as a general context ceiling.

## Bound and verdict scope

`lane_boundary_context_map_bound_v1` is registered with an actual executable
callable, empirical anchor, input hashes, and explicit exclusions. It is a
**numerical convex supporting-tangent bound on a fixed five-weight extension**:

`gain <= [base_bits - (F(w) - sum_j g_j*(w_j-e_j))/ln(2)] / 8`,

where `e_j=-4` for nonnegative gradient and `127/32` otherwise. The feature map
and all shipped probabilities are fixed. This is a floating-point numerical
certificate, not an interval-arithmetic proof or integer coder bound.

The fit kept 18,294,067 non-near-certain positions. Final reported codelengths
use **every** symbol. For the optimistic certificate only, omitted positions
receive zero candidate loss, granting the entire omitted baseline
663.146466 bits (82.893308 bits/8). Certificate gaps are 0.000107940 bits for GT
and 0.000032867 bits for causal. Optimistic gains are **5,513.942164** and
**161.425251** ideal bytes. Those bounds exclude a joint refit of the old35
weights, other features, other geometry, and integer rounding effects.

The GT-edge map is a **noncausal reference, not the charter's proposed universal
upper bound on every lane predictor**. Its quantization discards information,
GT boundaries differ from the shipped token boundaries, and alternative context
maps are not nested refinements of this map. The NO-FAKE rule requires this
correction. The 15 KB prior fails at **INSTANCE** scope for this particular
reference extension. A small causal result cannot close the geometry FAMILY or
all boundary-distance FORMULATIONS. The build handoff will retain that boundary.

## Real encode and parse-back

Axis: `[exact tail bytes, macOS-CPU scorer-free, n600]`.

| Physical bytes | Source | TC2 causal candidate | Saving |
|---|---:|---:|---:|
| Raw RC64 stream | 119,779 | 119,696 | 83 |
| Counted rider (header plus all weights plus raw stream) | 119,819 | 119,741 | **78** |
| Native padded envelope, without weight rider | 119,784 | 119,700 | 84 |

The candidate rider is `TC2M` + version1 (5B), original35 int8 coefficients,
new5 int8 coefficients, and raw RC64 bytes. The source rider is40B plus its raw
stream. Native envelope padding accounts for the different envelope delta;
**78B is the comparable net tail gain**, and no ZIP-size delta was measured.
Both independent encoders consumed all117,964,800 symbols. Payload SHA for each
119,700B native envelope is
`bb0a01a15b11ccee8ea481475dc97b0625bfe86bbd3b3cb00fcf41c0b17f96d0`.
Both119,741B riders have SHA
`f8eec37c914631ea83a9e871c05e634709b21754f1c536dba28a66f734b3425e`.
The exact integer-frequency codelength is957,562.134581bits; final RC64 rounding
and headers are accounted by the physical byte table, not inferred from this sum.
Encode elapsed258.54s.

The 3–8KB encoded prior is rejected for this **INSTANCE**, by38.5–102.6× against
78B. The causal gain is below1000B, so this arm supplies the better-predictor
build charter instead of implementing or encoding a richer predictor here.
The tiny positive payload is retained for MAIN's seal-chain intake; successor
field rebasing, public receiver integration and actual archive pricing remain
separate gates. Held-distortion arithmetic would value78 archive bytes at
0.0000519369984 score units, but **78 tail bytes are not yet78 archive bytes**,
and there is no measured TC2 score delta.

Conditional decode passed real frames0–1 with exact group-map, feature and token
identity, then resumed the complete saved decoder/KT/previous-plane state at
frame2. **All600 frames / 117,964,800 tokens passed**, with decoded SHA
`361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94` and all
114,000 group contexts/features identical. Resume decode elapsed196.97s, after
the3.02s two-frame control. The decoder regenerates the new context
from its own decoded prefix; the base HPAC/TC1 probability rows come from the
byte-verified source trace. It does not rerun the base network or public renderer.
Standalone public-receiver proof, full inflate-output parity, cross-host bin
parity, and exact scorer evaluation are explicitly **not measured** here.

## Mechanism and legal receiver inputs

The predictor is a generic slotwise degree1 lane fit, not an imported openpilot
forward model. Eight 64-column slots use the previous 16 strictly above rows.
Current-plane tokens are legal only when their HPAC group is earlier:
`g=(x%64)+2*(y%64)`, with 190 groups. Raster-row order alone would leak future
tokens. Previous-plane Lane pixels contribute weight0.25. Fixed moment equations
fit x(y); residual width estimates two boundaries. Fits reject underdetermined
moments, steep slope, high residual, wide spans, known non-Lane gaps between
known Lane pixels, and multiple prior-plane runs. Distances are rounded ties-even
and binned at `[0,1,2,4,8,16]`, with unavailable and outside-band bins.

The oracle uses unsigned Manhattan distance to the cached DALI GT Lane/Road
4-neighbour interface, including both endpoints. The causal predictor never
reads GT. Online KT observed/expected ratios supply one Q1024 log-ratio feature
per candidate symbol and coding winner, clipped to [1/16,16]. Counts update only
after a complete plane. The frozen 35-weight TC1 mixer supplies the base rows.

Mutation controls at real frames0,317,599 and groups0,31,63,94,126,158,189 changed
unavailable current-plane symbols without changing legal contexts. An online
geometry implementation matched every group of real frame0. These are causality
and implementation controls, not subset-based scientific verdicts.

## Custody, reproducibility, and originality

Owned store: `/Volumes/VertigoDataTier/pact/ddm_tc2_lane_context_map`.

- Source archive: 180,388 B,
  `670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc`.
- Move37/pass4 field: 117,964,800 B,
  `361cc6c9749fdec1381936836c9b45f4e04702f02eed9f8ea5343b1afa957b94`.
  The original TC1 pass3 field was rejected as stale before counting.
- Verified source trace: bnd2 generation2, full600 control and twin envelope
  identity; `trace/RESULT.json` SHA
  `a715df024b8e6d07186717358db496c58f07948881ad6b10c7e82d2c27f219fd`.
- Cached DALI GT labels:
  `44d3c7437401cc665a7f8166cf23f1d061f317e4db20f3c0d122b875a103bfa5`.
  Existing cache lineage verified; no GT video decode or scorer run occurred.
- Read-only evaluator SHA:
  `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`.
- Seed20260910, NumPy CPU arithmetic, threads1. Pointer snapshots are retained
  before every stage. Live pointer advances do not transfer these field-specific
  measurements to successor archives.

Prepared maps/features/frequencies, all fitted weights and optimizer iterations,
every stage and per-frame checkpoint, all trial payloads, native build sources,
twins, and parsed fields remain on the SSD with SHA/byte receipts. `--resume-from`
requires exact source/input bindings and receipt-complete atomic checkpoints.
The automatic hygiene path checks a 40GiB reserve before writes and fails closed;
no retained artifact was deleted or moved. There is no disposable bulk scratch.
Detached work uses the governed launcher, `--nice-best-effort`, and at most one
measurement process at a time (below the charter's two-process cap).

Fit elapsed58.1884s, own-process peak RSS10,164,830,208 B via `getrusage`.
The safe-run monitor's RSS0 reflects unavailable sandbox process visibility,
not zero memory or verified RSS enforcement. Preparation elapsed310.94s and
totals141.65s. Exact launch argv and statuses live in the SSD `launch_*` folders.

Borrowed substrate is explicit: shipped TC1 mixer/weights, HPAC probabilities,
bnd2's byte-verified trace, JG2 IO/native-build helpers, and the native RC64 coder.
TC2 contributes the causal geometry, online extra feature, fixed-family fit and
certificate, and retained experiment/conditional decode. It does not claim an
original full vehicle, solver, public receiver, or exact score.

All three Python files received two visible source reviews and Ruff checks.
The independent reviewer checked the full governing contract and the concrete
causal/retention/resume implementations. See the final review receipt for hashes.

## RECALL EVIDENCE

Independent content searches used
`lane.boundary|boundary.distance|lane.geometry|lane.line.fit|token.tail|shared.mixer`
over `.omx/research/` (including arm receipts), the canonical research index and
`sub015_DAG_*` FEED surfaces, docs/SPECs, canonical task state and lane registry.
The canonical equation registry was exported using
`.venv/bin/python tools/list_canonical_equations.py --json` to the owned SSD
`recall_equations.json`. Named equations were inspected with their domains;
charter references were not treated as a complete corpus.

Beyond the charter's seeds:

- `ddm_dds1_decoder_derivable_verdict_20260901.md` and its
  `ddm_dds1_ceiling_readjudication_20260901.md`: decoder-derived neighbour/L1 and
  previous-frame geometry had weak gains on AFR1 n120; the early M-only partial
  status was subsequently closed by the all-live ceiling adjudication. This
  changed the predictor from naive previous-plane reuse to a legal current-prefix
  fit, and prevents treating those different-field results as TC2 evidence.
- `ddm_dcc1_decoder_causal_conditioning_verdict_20260901.md` and
  `decoder_causal_condition_transport_v1`: legal conditioning needs encoder and
  decoder to derive identical CDFs from the same prefix. This caused the explicit
  group-order legality mask and the online decoder-side context assertions.
- `ddm_d3a_analytic_lane_carrier_20260826.md`: source lane charts cost27–43KB,
  with jitter/dash residuals. This prevents transferring the charter's rough
  eight-numbers-per-frame story as an established compact representation. Fits
  here are recomputed from legal tokens; stored coefficients would be charged.
- `ddm_mi1_indicator_model_axis_20260824.md`: existing correction already has
  temporal/boundary context families. TC2 measures an incremental feature, and
  does not claim lane geometry was globally absent from the receiver.
- TC1's complete memo and `token_tail_context_mixing_bound_v1`, together with
  `wyner_ziv_decoder_side_information_conditional_entropy_savings_v1`, distinguish
  descriptive side-information entropy from a realizable small mixer. This
  changed the pricing to full shipped probability rows, explicit fixed-family
  fitting, and separate entropy / ideal codelength / serialized charges.

The bnd2 follow-up distinguishes component-count and cell-count denominators and
concentrates residual near the far-field band. It is a motivation for a future
predictor, not a substituted denominator or a retry of greedy address grammars.

The stage reread of hot state also surfaced the newly completed
`ddm_bnd3_address_term_decomposition_20260910.md`. Its **83,259 B** oracle saving
supplies all235,044 miss locations **and true values** free while preserving the
TC1 remainder trajectory. That is a different, stronger side-information object
than TC2's GT-distance bin and is not a universal ceiling on other probability
models. Its48 measured joint/gap variants lose; it explicitly declines FAMILY
closure. This changes the handoff: do not relabel83,259 B as TC2's measured
geometry ceiling, and do not substitute its free values for a causal context.

## Landing and follow-on disposition

The initial reviewed measurement source could not write
Git objects in the shared repository; the serializer preserved bundle-only
commit `75fb5caabe564a1c46119616904082de71c7ffba`. This is not a main landing.
The final patch/manifest and owned equation append will preserve unrelated dirty
files and the shared staged index.

Three actions are **QUEUED-WITH-A-FIRE-ORDER**, owner **MAIN**, in the canonical
task ledger under `ddm_tc2_lane_boundary_context_map_20260910::{BETTER_PREDICTOR,
SEAL_INTAKE,LANDING}`. The durable consumer index is
`/Volumes/VertigoDataTier/pact/ddm_tc2_lane_context_map/handoff/FIRE_ORDERS.json`.

- **BETTER_PREDICTOR**: MAIN harvests the completed handoff, verifies the already
  met oracle/encode gate, deduplicates ownership, and assigns one arm on a fresh
  pinned field. Consumer `handoff/better_predictor/FIRE_ORDER.json`; build charter
  `.omx/research/charters/ddm_tc2_better_causal_lane_predictor_build_20260910.md`.
  Both live hypotheses below are **FOLDED** into this one build order.
- **SEAL_INTAKE**: harvest of successful full600 conditional parse-back triggers
  compatibility/rebase and standalone public-prefix proof. Cross-host bin parity
  and positive actual archive pricing precede any seal/evaluation. Consumer
  `handoff/seal_intake/FIRE_ORDER.json`; payload `codec/encode/twin0.rider`.
- **LANDING**: harvest in a Git-writable context triggers the serializer and
  required checks, replaying only absent owned events and preserving the staged
  index. Consumer `handoff/landing/FIRE_ORDER.json`; final patch/manifest are
  under `.omx/research/ddm_tc2_20260910/`.

## LIVE-HYPOTHESES

- A coherent decoder-side lane/run tracker may capture more of the GT-reference
  gain: this instance's eight independent slots and conservative ambiguity
  rejection discard continuity and dash information. Untested, not promised.
- Jointly refitting the original35 and new5 coefficients may improve calibration;
  the frozen-base certificate does not cover those interactions. Untested and
  folded into the better-predictor build rather than a second encode here.

## DEAD-ENDS

- This exact minimal extension as a multi-kilobyte move: the full-n600 real
  twin encode saves only78B including the new weights.
- GT-distance as a universal geometric ceiling: nonnested maps and quantization
  invalidate that inference. No geometry-family closure follows.
- Raster-order causality, stale pass3 inputs, and unbound native-wrapper resumes:
  each violates actual receiver order or custody and was rejected before launch.
- Promoting conditional token identity into full public-render/score proof: that
  would claim work the tested path does not perform.

Live own-vehicle frontier at the last stage read: **S0.13766931482209038 @180,186 B
[contest-CUDA T4 n600]**, rp1 move39. TC2 did not move it; sub0.12 remains unmet.
