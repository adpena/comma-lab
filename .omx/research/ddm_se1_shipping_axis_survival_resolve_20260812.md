# ddm_se1 — C1 shipping-axis survival resolution

Tags: [no-triality] [p0-ledger-ok]  
Date: 2026-08-12  
Axis: `[macOS-CPU advisory, stratified-random n32, instrument floor 0.0131 S]`  
Score claim: false  
Promotion eligible: false

## RESULT

F1 **FIRED** at FORMULATION scope. The hidden-4 C1-event-teacher conditioner, adapted through the
retained JS4 PoseNet-null projector and shipped as a parse-backed module plus five decoded-class
magnitudes, produced no state that simultaneously satisfied all three admission conditions:

1. realized section-additive joint `Delta S < 0` through the camera/uint8/SegNet/PoseNet chain;
2. realized pose delta `< 2e-6`; and
3. at least one beneficial flip beyond `delta = 0.08036041259765625`.

The numerically best row was undrivable-only at magnitude `0.125`: **section-additive
`Delta S = -0.003710011964`**, pose delta `-3.2379040e-05`, and 32 beneficial flips. However,
**0/32** beneficial flips cleared delta, so its delta-margin mass is **0%** and the apparent gain is
inside the measured CPU/CUDA disagreement band. It is rejected, not a candidate.

The best composed row used scales
`[0.0078125, 0.0078125, 0.015625, 0.015625, 0.015625]` and measured
**section-additive `Delta S = -0.000142214135`**, with pose delta `-5.4507986e-06` and 7 beneficial
flips. Again **0/7** cleared delta, so it is rejected. No n600 extension or exact recipe was earned.

F2 **DID NOT FIRE**. The all-class magnitude-1 row reached 67 robust beneficial flips and 21.61%
delta-margin mass, above F2's 20% threshold, but it destroyed pose (`+0.01316035`) and worsened joint
`Delta S` by `+0.32576577`. This is F1's joint-survival failure, not a thin-survival F2 case.

## SURVIVAL CURVE

Each cell is `section-additive Delta S / delta-margin mass`. All 30 rows are stratified-random n32
relative-gauge measurements. Every packet is 768 B and was parsed back before execution. None passed
admission.

| Decoded class | 1.0 | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---:|---:|---:|---:|---:|
| Road | +0.241290 / 13.02% | +0.117931 / 2.04% | +0.051374 / 0% | +0.018766 / 0% | +0.007997 / 0% |
| Lane | +0.152456 / 7.03% | +0.072360 / 0% | +0.031476 / 0% | +0.012625 / 0% | +0.004924 / 0% |
| Undrivable | +0.096987 / 7.27% | +0.028124 / 0% | +0.001818 / 0% | **-0.003710 / 0%** | -0.000936 / 0% |
| Movable | +0.001048 / 6.67% | +0.000017 / 0% | -0.000231 / 0% | -0.000714 / 0% | -0.000558 / 0% |
| MyCar | +0.040979 / 0% | +0.015134 / 0% | +0.006237 / 0% | +0.000914 / 0% | +0.002693 / 0% |
| All classes | +0.325766 / 21.61% | +0.167762 / 5.46% | +0.081848 / 0% | +0.032640 / 0% | +0.018413 / 0% |

The five composed shrink rows were:

| Shrink | Five scales | Joint `Delta S` | Pose delta | Robust / beneficial | Admitted |
|---:|---|---:|---:|---:|---|
| 1 | `.0625,.0625,.125,.125,.125` | +0.012106954 | +1.1224024e-04 | 0 / 43 | no |
| 0.5 | `.03125,.03125,.0625,.0625,.0625` | +0.008902620 | +7.7749185e-05 | 0 / 22 | no |
| 0.25 | `.015625,.015625,.03125,.03125,.03125` | +0.005853829 | +4.8108570e-05 | 0 / 17 | no |
| 0.125 | `.0078125,.0078125,.015625,.015625,.015625` | **-0.000142214** | -5.4507986e-06 | 0 / 7 | no |
| 0.0625 | `.00390625,.00390625,.0078125,.0078125,.0078125` | +0.000773867 | +1.4003019e-06 | 0 / 4 | no |

## WHAT WAS ACTUALLY BUILT AND MEASURED

- The literal HC1 archive was custody-checked at 187,046 B, SHA-256
  `12a5b181fef4e15ad8a752161c744347beca0b5a1224c5d3d542ab148f6ece80`. It remains the refuted
  exact baseline (`S=0.4044688071472634`, `rho` about `-1.223`); it was not repackaged or re-scored.
- C1 was used only as an encoder-side event teacher. The sealed 117,964,800-B C1 plane had SHA-256
  `2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5`. The n32 sample contained
  1,368 C1 events, a stratified projection of 25,660/600. Neither this plane nor the event mask ships.
- The receiver remained CP135. Four deterministic adaptation steps used the real camera/uint8 chain,
  the delta-margin hinge, and the retained 32-pair JS4 PoseNet-null projector. Four distinct checkpoints
  retain model, EMA, optimizer, Torch/NumPy RNG, history, and full config. Live and EMA byte-close
  exports exist at every stage.
- The selected live module has 563 parameters and is 744 B after int8+Brotli q11, SHA-256
  `4b017f97064c29cb8dcb194a12a181301f4f72c79f01ca880d604efc16a22f82`. The five float16 class
  scales bring every complete packet to 768 B. The projector contributes 0 shipped bytes, no scorer
  runs at decode, and the packet parser exactly reconstructs both module and scales.
- Thirty single/all-class curve rows plus five composed rows were executed. All 35 packets have a
  byte-identical deterministic repeat. Every row retains its packet, correction, camera uint8 bytes,
  logits, argmax, pose errors, and 32 pair receipts. The run directory is 8.9 GiB at
  `/Volumes/APDataStore/pact/ddm_se1_20260812/`.
- The governed run completed successfully in 995.921 s. The durable final receipt is 302,046 B,
  SHA-256 `a71fed878eeab67f82144996a92cd1e14b31dbc9fcfa1ee8dc5a95e97c3bdc5f`.

## FIVE-MECHANISM CLOSURE

1. **Chain in loop:** yes; adaptation and every candidate used CP135 receiver state, bilinear camera
   lift, uint8, scorer resize, custody transport, and frozen scorers.
2. **Delta margin:** yes; `delta=0.08036041259765625` was in the loss and in strict admission. Code,
   initial weights, seed 20260812, 8 threads, and batch 16 were pinned.
3. **Pose null:** yes during adaptation via retained JS4 pair projectors; 0 projector bytes shipped.
   Realized pose was checked independently for every parse-backed packet.
4. **Realized acceptance:** yes; joint Seg, nonlinear pose, and 768-B rate terms were recomputed after
   parse-back. The magnitude and composed ladders performed reject-and-shrink. No row was accepted.
5. **Exact rung:** not earned. No archive was built, no n600 scorer was run, no Modal job was dispatched,
   and no exact recipe was sealed.

## RECALL EVIDENCE

Recall searched all seven corpus stores with these queries:

- `shipping axis survival C1 cp135 delta margin pose null realized acceptance`
- `camera uint8 receiver chain C1 per class survival curve`
- `projector distilled event coordinates realized acceptance C1`

Each query covered 8,433 research rows, 886 equation rows, 2,112 memory rows, 915 DAG rows, 297 council
rows, 531 task rows, and 96 docs rows. The canonical equation registry was also listed, and the research
indexes and `sub015_DAG_*` surfaces were searched for C1, receiver survival, camera uint8, delta margin,
pose-null, and realized acceptance.

Beyond the charter's seeds, recall found that JS5's all-class projector-distilled alpha ladder already
had 0/15 accepted proposals, while the completed EC1 arm had already extracted and priced the distinct
event-coordinate representation and preserved the existing 200-proposal bank. This changed the plan
from another all-class JS5 replay to C1-event-teacher adaptation plus decoded-class magnitude routing,
and it changed the F1 route from “build EC1” to “consume EC1's completed proposal bank in realized
acceptance.” Recall also established that the retained measured JS4 source, not the evolving worktree
copy, was the projector authority.

## BORROWED SUBSTRATE ACCOUNTING

**Borrowed:** CP135 archive, renderer, decoded semantic plane, scorer custody tensors, JS3 hidden-4
conditioner and real camera/uint8 chain, JS4's retained projector and selected module, the frozen CPU
SegNet/PoseNet, HC1's literal-container evidence, and the contest score equation.

**Ours-original in this arm:** C1-event-teacher adaptation without shipping C1; the five-scale packet
grammar and exact parser; per-decoded-class and composed reject/shrink survival curve; strict joint,
pose-endpoint, and delta-margin admission; per-pair retained receipts; and the F1 route into the already
completed EC1 event representation. This is not an original renderer, scorer, projector, or exact row.

## VERIFICATION AND PROVENANCE

- Measured runner source: 39,146 B, SHA-256
  `09cf47cac660ccf353ce6379f611c6d857ba16a933364a511c853d9df3c3080b`, retained under the run's
  content-addressed source custody.
- Landing source: `experiments/ddm_se1_shipping_axis_survival_resolve.py`, SHA-256
  `b1541d72c40ed170823d3ef31490f71896089b8a20cc29f098861a5ab62c00ef`. Post-measurement changes are
  formatting/lint cleanup and a complete-state custody fast path only; measured candidate logic was not
  rerun or relabelled.
- Test source: `experiments/tests/test_ddm_se1_shipping_axis_survival_resolve.py`, SHA-256
  `df8a75997a59f9d820c2c357b8183f2be85226dbe76f193e6408d8558c2ba3a3`.
- Focused plus inherited suites: 36 passed. Ruff check and format-check pass. Review tracker records
  37 entities in each of `ddm_se1_pass1` and `ddm_se1_pass2`.
- The full developer preflight was not globally green: 8/25 gates were red on the shared dirty
  worktree. Individual non-strict enumeration found 1 state-writer, 1 authoritative-tag, 25 drift,
  1 dispatch-helper, 124 historical wire-in, 9 lane-registration, 56 substrate scorer-contract, and
  21 substrate pose-default violations; **0 in every set referenced either ddm_se1 file**. The staged
  fast hook passed. This arm did not edit unrelated violations to manufacture a green result.
- Shared staged index remained empty; protected files and `upstream/` were untouched.

## BOUNDARIES

- `Delta S` values are macOS-CPU relative-gauge, section-additive packet prices. They are not complete
  archive stat sizes, exact scores, n600 measurements, or contest-CUDA evidence.
- The weighted `projected_n600_*` flip counts are stratified-n32 projections, not n600 scorer runs.
- F1 closes only the named hidden-4, C1-event-teacher, five-class-scale formulation on this n32
  population. It does not close representation-changing event coordinates, a larger/distinct receiver,
  or a joint train-time pose mechanism.
- The best raw negative `Delta S` row is rejected because its entire Seg movement is below delta. It is
  not a “win,” a sealed candidate, or grounds for a pointer update.
- No external message, scorer lane, GPU, Modal job, full archive, or exact evaluation was created.

The effective frontier remains CP135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4 n600]`.
The own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4 adjudicated n600]`.
This arm moved neither pointer.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN training-leg router; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_ec1_20260812/realized_acceptance`; fire trigger: MAIN confirms the
  completed EC1 200-proposal bank is content-distinct, owns the sole scorer lane, and runs parse-backed
  realized acceptance through the same camera/uint8, delta-margin, joint-score, and pose-endpoint gates.

## LIVE-HYPOTHESES

- EC1's event coordinates remain plausible because the class-scale packet changed every pixel of a
  decoded class together, while EC1 can address the sparse, content-distinct C1 events that the teacher
  identified; that is the representation change F1 explicitly leaves open.
- A receiver that compiles event-local corrections into its deterministic generic rasterizer may retain
  robust Seg margin without the global photometric pose spill, because robust movement appeared only at
  large class-wide magnitudes and pose damage grew with that global support.
- A joint train-time pose mechanism remains plausible because the fixed first-order null removed tangent
  pose motion during adaptation but bare nonlinear packets still leaked after uint8; shaping the shipped
  bytes directly against pose may close the distillation gap.

## DEAD-ENDS

- Literal C1-plane substitution is not retried: HC1 already proved exact token carriage and measured
  `rho` about `-1.223` with `S=0.4044688071472634` on contest CUDA.
- Another all-class JS5 amplitude replay is closed: JS5 measured 0/15 accepted proposals, and this arm's
  all-class curve reproduced the same robust-Seg/pose-spill tradeoff.
- Treating a negative local `Delta S` below the delta margin as survival is closed: the best row had
  0/32 robust beneficial flips despite `Delta S=-0.003710011964`.
- Five decoded-class scalar magnitudes on this hidden-4 conditioner are closed at FORMULATION scope:
  no single-class, all-class, or composed shrink state passed joint, pose, and delta-margin admission.
- Shipping the JS4 projector is closed for this path: it is hundreds of megabytes of training-only
  scorer-derived state, while the legal packet must remain the bare module plus scales.
