# D3A analytic Lane carrier — complete counted-chart n600 verdict

**Date:** 2026-08-26

**Axis:** `[macOS-CPU advisory / DALI-GT pinned n600]`

**Denominator:** 4 planned fidelity rungs; 4/4 independently parsed, packed, rendered, and scored at n600

**Verdict:** **REFUSED at FORMULATION scope.** The complete source-local polynomial LBND2 chart is
27,440--43,032 B, not the predicted 5--10 KB, and every measured composition loses badly on realized
Seg/Pose despite saving 20,888--36,480 archive bytes. No candidate earns a seal, public runtime, Modal
dispatch, or pointer motion.

The best D3A row is q1: **S = 5.62821305922116 @ 159,327 B**, `d_seg =
0.004249233669704861`, `d_pose = 2.598145174268742` `[macOS-CPU advisory / DALI-GT pinned
n600]`. Against the GB1 reference `S = 0.14811799921260607 @ 180,215 B` `[contest-CUDA T4
n600]`, the mixed-axis triage delta is **+5.480095060008554**. The axis mismatch makes that delta
non-promotable, but its sign and multi-unit margin are enough to refuse a remote fire.

## 1. Object measured and scope

The source is the exact GB1 token field `cc10a7b0...`. Its class-1 Lane pixels are fitted into the
existing degree-3 openpilot-IPM `LaneLine` chart. The complete video-derived chart is quantized by the
existing LBND2 codec, correspondence-packed into six coherent slots, Brotli-q11 coded, and wrapped in
a counted D3A packet. The receiver independently reopens that exact packet, decompresses LBND2,
dequantizes every coefficient, regenerates AA-SDF coverage, thresholds it at 0.95, and paints Lane
only where the receiver-closed D3 field says Road. That token field drives the retained GB1 renderer;
the resulting uint8 raw drives the frozen CPU SegNet/PoseNet against pinned DALI GT.

The four rungs scale every LBND2 geometric tolerance together: q8, q4, q2, q1. All use coherent-slot
packing, no temporal smoothing, degree-3 centerlines, the existing range-dependent dash gate, and the
same precision-first 0.95 coverage threshold. This is a complete verdict on that formulation, not a
family verdict on every analytic Lane program, task-aware knot fit, joint-trained renderer, or
photometric realization.

The D3 four-symbol store remained read-only. All new bulk lives under
`/Volumes/APDataStore/pact/ddm_d3a_analytic_lane_carrier/`.

## 2. RECALL EVIDENCE

### Surfaces and queries searched

- Full `.omx/research/` memo/receipt corpus, canonical research indexes, `sub015_DAG_*` FEED blocks,
  design specs, task ledger, and code with content queries: `analytic lane`, `lane band`, `LBND1`,
  `LBND2`, `LBND3`, `LBND4`, `lane program`, `lane carrier`, `Road gate`, `Lane->Road`, `AA-SDF`,
  `coherent_slot`, `lane coeff`, `correspondence`, `ego predictor`, and `lane coverage`.
- Canonical equations registry via
  `.venv/bin/python tools/list_canonical_equations.py --json`; no equation-specific override of this
  receiver/rate object was found.
- Charter seeds at source: D3 instrument and memo; v8 per-class spec; v10 J2 lane-program receipt;
  RG1 receiver grammar; CB1 byte-close receipts; Wave-F build/RD/tracking memos; Rust runtime; LD1,
  MSR1, and LM1 negatives.

### Beyond the charter seeds

- `analytic_lane_render_band.py` already contains LBND4 residual/context coding and the
  correspondence-first `serialize_lane_band_rd_tracked` path. The measured Wave-F evidence says
  coherent-slot correspondence is only a small lossless rate improvement, persistent tracks worsen
  bytes, and independent fit jitter dominates. This changed the build from a new AR coder to the
  existing coherent-slot LBND2 wire.
- The existing LBND3 ego predictor had already measured worse than LBND2 on the prior n600 Lane
  source. That removed an otherwise tempting predictor detour.
- The Rust `lane_coverage.rs` AA-SDF implementation already has Python-canonical parity/golden
  coverage evidence. That changed the plan from rebuilding a rasterizer to using the canonical Python
  authority while keeping the Rust path as an existing implementation option.
- Prior Road↔Lane interface pricing showed that even the cheapest real-coded crop representation was
  only an n32 projection, not an n600 carrier verdict. That prevented importing its projected byte
  rate into this row.

## 3. Rule-118 accounting

| Object | Counted in `archive.zip` | Free generic code | Proof |
|---|---:|---:|---|
| Quantized LaneLine centerline, half-width, dash, range, presence, and temporal-delta data | yes | no | exact LBND2 body is inside each D3A packet |
| Scalar tolerance schedule, 0.95 coverage threshold, shape, and Road-gate declaration | yes | no | canonical JSON D3A header is inside each packet |
| Brotli-q11 LBND2 payload | yes | no | real coder bytes retained and embedded |
| D3 four-symbol RC64 stream | yes | no | 49,696 B stream `84fa2f49...` embedded |
| GB1 video-derived model/carrier/pose/corrector sections | yes | no | inherited bytes remain in the actual packed archive |
| LBND2 parse/dequantize algorithm | no | yes | generic algorithm, no source-local table |
| AA-SDF range-dependent-dash rasterizer | no | yes | generic algorithm; source-local coefficients are counted |
| Road gate and Lane token paint | no | yes | generic Boolean/token operation |
| GT masks, scorer weights, donor charts, per-pixel lookup tables | forbidden and absent | forbidden | carrier header records the absence; independent parser needs none |

The retained source LBND1 fit is 355,642 B raw (`a5f295a7...`) and 156,408 B under Brotli q11. It is
not shipped in addition to LBND2: LBND2 is the complete decoded chart state needed by the receiver,
not a marginal patch over an absent donor chart.

## 4. Counted chart and source-Lane frontier

The source mask denominator is 691,095 Lane pixels. Its retained exact packbits payload is 14,745,600
B, SHA-256 `6ca82a7883411d0eb27addac7dcf662e84d2f9cc66404c299da2e15761c0e0cf`.

| Rung | lateral tolerance | counted chart | actual archive | source precision | source recall | source IoU | parse-back |
|---|---:|---:|---:|---:|---:|---:|---|
| q8 | 0.16 m | 27,440 B | 143,735 B | 0.574912 | 0.376384 | 0.294442 | 600/600, LBND2 bytes identical |
| q4 | 0.08 m | 32,293 B | 148,588 B | 0.665876 | 0.444155 | 0.363207 | 600/600, LBND2 bytes identical |
| q2 | 0.04 m | 38,566 B | 154,861 B | 0.694674 | 0.468753 | 0.388675 | 600/600, LBND2 bytes identical |
| q1 | 0.02 m | 43,032 B | 159,327 B | 0.702218 | 0.475509 | 0.395716 | 600/600, LBND2 bytes identical |

The actual archive is the pre-pack projection plus eight bytes at every rung, exactly the added D3
carrier-length framing. The prior-law prediction `complete chart <= 10,000 B` is falsified 4/4.

## 5. Realized n600 carriage frontier

Scores below are recomputed as `100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489`; no two-decimal
evaluator print is used. Candidate components are `[macOS-CPU advisory / DALI-GT pinned n600]`.
`delta S` compares them to the GB1 `[contest-CUDA T4 n600]` reference and is therefore mixed-axis
route triage, not a promotable matched-axis delta.

| Rung | archive | `d_seg` | `d_pose` | recomputed S | bytes vs GB1 | exact rate contribution to delta | exact Seg contribution | exact Pose contribution | delta S vs GB1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q8 | 143,735 B | 0.005327368842 | 3.139843449776 | 6.231874721630 | -36,480 | -0.024290535 | +0.512597884 | +5.595449373 | +6.083756722 |
| q4 | 148,588 B | 0.004565836589 | 2.698337941963 | 5.750075169394 | -31,627 | -0.021059121 | +0.436444659 | +5.186571632 | +5.601957170 |
| q2 | 154,861 B | 0.004317609999 | 2.671105301217 | 5.703150280722 | -25,354 | -0.016882188 | +0.411622000 | +5.160292470 | +5.555032282 |
| q1 | 159,327 B | 0.004249233670 | 2.598145174269 | 5.628213059221 | -20,888 | -0.013908462 | +0.404784367 | +5.089219155 | +5.480095060 |

At the requested GB1 derivatives, `dS/dB = 6.658...e-7`, `dS/dd_seg = 100`, and
`dS/dd_pose = 626.5...`. The receipts preserve that first-order decomposition, but the pose move is
so large that linearizing it at GB1 is not decision-safe; the table therefore reports the exact
nonlinear square-root contribution.

Finer charts improve all observed distortion metrics, but not enough to approach viability. Even
granting q1 **perfect pose for free**, its zero-pose lower bound is
`100*0.004249233669704861 + 25*159327/37545489 = 0.5310126763945823`. QS5 compensation alone cannot
rescue this formulation; its Seg+rate floor already exceeds GB1 by more than 0.38 score units.

## 6. Payload custody

Root machine-readable joins:

- `SUMMARY.json`: 4,148 B, SHA-256
  `76542fb2c2f699c338ebcda38083acb5b83e7504855a754f711233cecb7fe5de`
- `BUILD_RESULT.json`: 33,488 B, SHA-256
  `78f7bb283969c3810cf2bb1ff1da307576960fc95e6bb1ba7d8ca7ba2f42d744`
- root: `/Volumes/APDataStore/pact/ddm_d3a_analytic_lane_carrier/`

| Rung | counted carrier `(path, sha256)` | archive `(path, sha256)` | raw `(path, sha256)` |
|---|---|---|---|
| q8 | `retained/rungs/q8/counted_lane_carrier.d3a`, `a94ed42cab3b415bfe338621fa48b8a99ab057212f94cc0b747a9f7893926559` | `retained/candidates/q8/submission/archive.zip`, `b8bd90c2be3bf8afb44f4d3d7dd6a00742369db0c459cdebc78bfad7ce7e26af` | `retained/candidates/q8/submission/inflated/0.raw`, `e4c6999fe1088b954b286758f3ccd4ef13ee7c83cec9156e7a107c8be3a79e98` |
| q4 | `retained/rungs/q4/counted_lane_carrier.d3a`, `6224aa8f186af45b4d331c624517092e1111793386f2bd982f0bf2e2e91efe4b` | `retained/candidates/q4/submission/archive.zip`, `d396bbe4cccd5840c0755f6ac9422c9ea8c277292174d074e1a832b8e0268f59` | `retained/candidates/q4/submission/inflated/0.raw`, `4deb014c3db371d828c0954f9b225a0434152d44a3208993c0291bf330c3327c` |
| q2 | `retained/rungs/q2/counted_lane_carrier.d3a`, `6e352054a0eb7eb7bf7513c88099f861ed14230f663a33eb4505b3a26c836a53` | `retained/candidates/q2/submission/archive.zip`, `e3bb3df6ff6038b17cafae9fc74b5f05857ea5a51d9991bad46eeada0d0cd134` | `retained/candidates/q2/submission/inflated/0.raw`, `1965fc6a529b9e58b40048012d281c42134b2e284a8d0622e6d20f3ce41ce483` |
| q1 | `retained/rungs/q1/counted_lane_carrier.d3a`, `f5e745d4886541ba83e503a6658bc23044686ef4361da654691bf93045b64750` | `retained/candidates/q1/submission/archive.zip`, `1194bc9d559cccc97b5baaa4acde946f5aa132ccb0f876abfc75f654c319af30` | `retained/candidates/q1/submission/inflated/0.raw`, `c19ecf530e5d076f5e461322f36d0b48a1f8a48c0d880b60b9cebc11a941898e` |

Each rung also retains its 117,964,800 B token field, 472 MB float32 coverage NPY, raw/road-gated
packbits masks, eight render chunk receipts, 38 scorer chunk receipts, 117,964,928 B n600 argmax NPY,
and 14,528 B pose6 NPY. The q1 scorer hashes are `c60a6b23619e2595be97f81719a05ab7ec2c7ded04d3d027205a4aef13673b69`
(argmax) and `37e5da82f73335b3312a978288d8a999ed6bec931db423381c11a6230f5f6143`
(pose6); corresponding complete facts for all rungs are in `SUMMARY.json` and each retained `RESULT.json`.

## 7. Ledger, seal, and authority disposition

Canonical task `ddm_no1_row3_alphabet_merge::d3a_analytic_lane_carrier` has an append-only lifecycle
at `.omx/state/canonical_task_status.jsonl:602-604`, actor `ddm_d3a`: registered -> in_progress ->
completed, `test_status=green`, with the FORMULATION-scoped negative and best measured row.

No candidate has net advisory improvement, so `make_candidate_seal` was correctly not invoked. No
Modal job was launched. No public contest inflate runtime was emitted: these are closed research
containers exercised by the retained renderer path, not submissions or exact-eval candidates. The
canonical frontier pointer remains GB1.

## 8. GESTALT-DELTA

**GESTALT-DELTA:** D3 proved that removing Lane from the coded alphabet exposes enough rate to matter;
D3A now proves that a small-looking geometric chart is not automatically a scorer-equivalent Lane
carrier. The scarce object is neither the four-symbol stream nor generic lane geometry. It is a
high-precision, pose-safe realization that reproduces the evaluator's Lane cells without painting a
photometrically destructive token band. Representation and realization must be optimized together.

## 9. Routing disposition

- **FOLDED:** QS5 pose compensation on these four fixed raws. Perfect pose still leaves q1 at S
  0.5310126763945823, so compensation cannot reverse the verdict.
- **FOLDED:** finer uniform LBND2 quantization as the next action. q1 is already the finest canonical
  tolerance and its Seg+rate lower bound is losing by a wide margin; more bytes attack neither fit
  topology nor photometric realization.
- **QUEUED-WITH-A-FIRE-ORDER:** a different D3 carrier may proceed only as a joint evaluator-native
  representation/realization, not another fixed polynomial-chart token paint. Owner: MAIN. Consumer
  store: `/Volumes/APDataStore/pact/ddm_d3_next_lane_carrier/`. Fire trigger: a retained real-coded
  complete packet, independently parsed, demonstrates on a stratified real-render screen that its
  **zero-pose** n600 extrapolation can beat GB1 and that its actual pose response is bounded; only then
  claim the single n600 scorer lane.

## 10. LIVE-HYPOTHESES

- A jointly trained photometric/evaluator-native Lane realization can preserve D3's 63,928 B stream
  credit without the multi-unit pose catastrophe because the present damage is caused by discrete
  token repainting, not by the four-symbol receiver itself. It must also reduce q1's Seg error by an
  order of magnitude; pose compensation alone is insufficient.
- A task-aware knot/topology program can outperform the fixed degree-3 source fit because the finest
  chart still recovers only 47.55% of source Lane pixels, while v10/RG1 already prove receiver-effective
  knot/program DOFs. The hypothesis is about a different formulation, not a q0.5 rerun.
- LBND4/context residual coding may reduce chart bytes after a task-valid geometry is found because
  prior Wave-F work locates independent fit jitter as the rate source. It is secondary: the present
  binding wall is realized Seg/Pose, not the remaining 27--43 KB chart rate.

## 11. DEAD-ENDS

- The prior-law prediction that the complete source-local analytic chart would cost <=10,000 B is
  closed for this formulation: all four real-coded packets cost 27,440--43,032 B.
- The q8/q4/q2/q1 coherent-slot, unsmoothed LBND2 chart with fixed 0.95 precision-first AA-SDF token
  paint is closed at FORMULATION scope: 4/4 n600 rows score 5.628--6.232 advisory.
- A pose-only repair is closed for these rows: q1 remains S 0.531013 with perfect pose.
- Repeating D3's block-mask, lossless-mask, Lane-inside-five-symbol-stream, zero-byte boundary-repair,
  CB1-marginal-with-hidden-chart, or HPAC-model-axis paths remains closed by D3/LD1/MSR1/CB1/LM1.
- LBND3 ego prediction and persistent-track column expansion should not be retried as byte cures; the
  prior n600 source measurements already made both worse than compact LBND2.

**OWN-VEHICLE FRONTIER:** GB1 remains **S = 0.14811799921260607 @ 180,215 B**
`[contest-CUDA T4 n600]`; D3A's best measured row is q1 at **S = 5.62821305922116 @ 159,327 B**
`[macOS-CPU advisory / DALI-GT pinned n600]` and does not move it.
