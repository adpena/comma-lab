# DDM BG1 — bilinear gate postmortem and pricing

Date: 2026-08-14  
Charter: `charter_ddm_bg1_bilinear_gate_pricing_20260814.md`  
Disposition: **REFUSED-TO-SEAL / BLOCKED-ON-PRIMARY-CUSTODY**  
Verdict scope: **no mechanism verdict**; the EC2 instance remains refused, while the BG1 frame-gate formulation remains untested.

## Result first

The four-channel gate is genuinely small, but the charter's mandatory admission
test could not be completed from local custody. The exact EC2 `FINAL_RESULT.json`
was recovered byte-for-byte from the local Modal return receipt. It confirms the
`[contest-CUDA T4 frozen-SegNet, n600] COMPONENT-ONLY` endpoint: the selected
plain oriented adapter fixed 12,075 of 34,970 base errors, introduced 52,854 new
errors, and therefore worsened the field by 40,779 flips. Its retained module is
1,369 B and its exact archive is 187,723 B, +1,471 B over CP135. The endpoint
argmax field and `BATCH_RECEIPTS.jsonl`, however, exist only on the remote Modal
volume and were not embedded in the return receipt. Modal DNS is unavailable in
this managed sandbox, so the per-edge, per-class, spatial, and frame-correlation
postmortem is **UNKNOWN**, not negative.

The scorer-free price is complete on the retained oriented EC1 design surrogate.
The proposed interaction has exactly 36 logical parameters: a 4x8 int8 matrix
and four float16 biases, or 40 tensor bytes before metadata and Brotli. Inline in
the existing oriented module, a seeded nonzero capacity reference added **38 B**
to both the Brotli module and the deterministic whole archive; the exact-zero
identity control added **76 B**. The non-monotone result is a real whole-stream
Brotli interaction. Both are `[macOS-CPU scorer-free exact byte/container price
on retained EC1 oriented design surrogate]`; neither is a receiver-valid or
scorer-valid candidate. The conservative observed gate allowance is therefore
+76 B, below rfo1's +128 B design cap, but the actual trained-EC2 inline price is
still unknown because the 1,369-B trained module was not locally retrievable.

No training fire order is sealed. No Modal or Metal action was fired. The
postmortem evidence gate comes before the mechanism branch, so neither a BG1
train nor the alternative #978 route may be promoted from the endpoint totals
alone.

## Authority ledger

| item | result | authority and boundary |
|---|---:|---|
| EC2 base errors | 34,970 | inherited and reverified from exact recovered `FINAL_RESULT.json`; `[contest-CUDA T4 frozen-SegNet, n600] COMPONENT-ONLY` |
| EC2 fixed base errors | 12,075 | same receipt; 34.5296% of base errors |
| EC2 introduced errors | 52,854 | same receipt; 4.3771 introduced per fixed error |
| EC2 endpoint errors | 75,749 | same receipt |
| EC2 net flip reduction | -40,779 | same receipt; instance refused |
| EC2 selected adapter | 1,369 B, SHA-256 `9559c2ab5128f193c8b0c754c5d61851b7784070fa049e04cf48cfd157eead82` | exact remote payload identity from recovered receipt; payload bytes not local |
| EC2 selected archive | 187,723 B, SHA-256 `3fcef97c9857123f4c8fde4ec0f74d20cee3244b131bc7e12f19e1a3b7b2e97b` | exact remote payload identity; +1,471 B versus CP135; archive bytes not local |
| EC2 elapsed time | 543.597070261 s / 1,800 steps | measured T4 component-worker receipt; includes the endpoint/package work |
| BG1 logical core | 36 parameters | verified against EC1 hidden=4 and CP135 frame latent=8 |
| BG1 raw tensor addition | 40 B | 32 int8 weights + 4 float16 biases |
| BG1 nonzero price | +38 B | exact module and whole-container delta on the retained oriented EC1 seeded nonzero design surrogate; no receiver/scorer claim |
| BG1 identity price | +76 B | exact module and whole-container delta on the same surrogate; no receiver/scorer claim |
| EC2 introduced-error decomposition | **UNKNOWN** | endpoint field and batch receipts unavailable locally |
| EC2 frame-latent association | **UNKNOWN** | cannot be inferred from aggregate totals |
| BG1 score | **not measured** | no receiver implementation, training, Seg/Pose replay, or exact evaluation |

The recovered primary receipt is
`/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/ec2_oriented/FINAL_RESULT.json`,
4,169 B, SHA-256
`b9df6af2175b2003e8a236e67e3315b60a065c7913792fcd93464c203c321a1b`.
The extraction receipt is
`/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/ec2_oriented/EMBEDDED_RECEIPT_RECOVERY.json`,
1,422 B, SHA-256
`668b39d8bc68c32300dee2d7e998f742c665a518d1dd8b4f2db6f4026979da4c`.

## EC2 postmortem admission bar

### What the available receipt proves

The targeting signal is real at this instance: 12,075 baseline mistakes became
correct. The frame-blind correction is still inadmissible because it introduced
52,854 mistakes at baseline-correct pixels. The aggregate ratio is diagnostic
of collateral domination, but it contains no location information. It cannot
tell whether the new errors concentrate in particular frames, classes, directed
class edges, or spatial neighborhoods, nor whether the already-counted 8-D
frame state predicts that concentration.

The local return receipt names the missing primary objects exactly:

- endpoint argmax field: 117,964,928 B, SHA-256
  `803a1d8755cafcf31b03d8ad1494d49f89f6e4fb2115341423308e0db20b3a1a`;
- endpoint batch receipts: 67,055 B, SHA-256
  `ffa88ae4478727edb9df89a35f023407ddc1f6cdc8c029530448fdb601087b55`;
- trained adapter: 1,369 B, SHA-256
  `9559c2ab5128f193c8b0c754c5d61851b7784070fa049e04cf48cfd157eead82`.

The custody blocker is retained at
`/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/EC2_PRIMARY_CUSTODY_BLOCKER.json`,
2,554 B, SHA-256
`5d09e3c5737ec48eee99d71fad77ed7e95bdd949c587e902902e44638f821568`.
Read-only `modal volume get` and `modal volume ls` attempts returned no file;
direct diagnostics reported that `api.modal.com` could not be resolved. The
in-app browser fallback had no available browser session. No local file with
the pinned endpoint-field or batch-receipt hash was found in the bounded Pact
workspace/SSD custody search. The similarly named local `ddm_ec2` directory is
a different, older scorer-free run and is not substituted.

### Exact analysis to run when the two payloads arrive

The postmortem is pre-registered so recovery does not turn into an
after-the-fact mechanism story:

1. Verify the two SHA-256 values above and the batch ordering before reading
   any result. Use the already-retained GT and CP135 fields with SHA-256
   `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`
   and `7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727`.
2. Reconstruct exact masks:
   `fixed=(base!=gt)&(ec2==gt)` and
   `introduced=(base==gt)&(ec2!=gt)`. Require totals 12,075 and 52,854.
3. Count introduced cells by GT class, EC2 predicted class, directed
   `GT->EC2` edge, unordered edge, frame, exact pixel coordinate, and the same
   fixed 12x16 spatial grid (32x32 scorer pixels per cell). Join the result to
   the JS1C confusion-cell and QS4 local-collateral receipts only after this
   EC2-native census is complete.
   Report every denominator: baseline-correct opportunity cells by the same
   class/edge/frame/spatial slice.
4. For each frame, form introduced rates overall and per directed edge. Report
   dispersion and concentration, but do not call mere overdispersion a gate
   mechanism: different frames expose different edge mass.
5. Parse the exact already-counted CP135 600x8 frame embedding. On seeded,
   non-prefix five-fold splits stratified by baseline-error/exposure quartile,
   test whether that 8-D state predicts held-out per-frame introduced rates
   after class/edge/spatial exposure is included. The primary mechanism gate is
   out-of-fold `R^2>0` and a seeded 10,000-permutation frame-label
   `p<=0.01`; per-edge models are diagnostic. The
   mechanism-supporting fact is positive out-of-fold predictivity of the
   available state, not simply a high top-decile share. A seeded permutation
   test over frame labels supplies the null.
6. Cross-check the per-batch sums against `BATCH_RECEIPTS.jsonl`; retain the
   full decomposition, masks, and every fitted/predicted array, not scalar-only
   summaries.

Decision rule: if the existing 8-D frame state has no held-out association with
the residual introduced-error rate after exposure controls, close this
frame-gate formulation on CP135 and route the Seg axis to the coupled
multi-token #978 formulation. If it does, the postmortem supports — but does
not itself prove — the low-rank gate mechanism, and only then may the training
order below be sealed.

Because the required arrays are missing, the decision rule is not evaluated.
This is a blocker classification, not evidence that the collateral is or is
not frame-structured.

## Four-channel bilinear gate

### Real module anatomy

At EC1 commit `fa29eb9ea17d3bfd5138478470600f322050634d`, the oriented
conditioner consumes a 25-channel decoded-token edge context. It applies a
3x3 `25->4` context convolution, GELU, a four-channel 3x3 depthwise
convolution, GELU, and a `4->96` 1x1 head. Its 96-channel delta is added after
CP135 `coord_mix` and before the four TokenBlocks. Only after that addition does
the receiver load `semantic.frame_embed(pair_indices)` and apply the existing
frame FiLM inside each TokenBlock. The EC1 correction is therefore
frame-invariant where it is formed even though later nonlinear blocks can
interact with frame state.

Let `h in R^4` be the EC1 hidden state after the depthwise GELU and let
`z_i in R^8` be CP135's existing frame embedding for pair `i`. The minimal
identity-safe interaction is

`g_i = tanh(W_g z_i + b_g)`

`h'_i(x,y,c) = h_i(x,y,c) * (1 + g_i(c))`

followed by the existing `4->96` head. This contains the bilinear term
`h_c * sum_d W_g[c,d] z_d`. Initialize `W_g=b_g=0`, so the receiver is an
exact identity relative to its chosen EC checkpoint before training. The gate
adds `4*8 + 4 = 36` shared logical parameters and no new per-pair table. All 36
parameters are counted. If optimal-form training changes the existing 600x8
state, those changed bytes remain counted in the rebuilt CP135 model section;
“already paid” means no new table shape, not free learned values.

Injection anywhere after the 96-channel head would cost more gates and would
not be the rfo1 low-rank object. Injection before the second GELU would alter
the identity and interaction geometry. The sealed design point is therefore
after the four-channel depthwise GELU and immediately before the existing
1x1 head.

### Retained int8+Brotli price

The price materializer and every generated payload are retained under
`/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/pricing/`. Its final
manifest is `PRICE_MANIFEST.json`, 8,153 B, SHA-256
`bc2bfef827d03e3e49930d075d68ff2407e5e6b826527790bb57b82f56a565a9`.
It used the pinned CP135 archive, 186,252 B, SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`,
and the retained EC1 oriented seeded capacity-reference module, 1,605 B,
SHA-256
`616ee8d94bf14e5399e56e5009d4e98b5e10f965c3888073d57fe97d4e03b83e`.
The matching baseline design archive is 187,959 B, SHA-256
`aac8552c0718d96945f824c82769c52a692d015b1ef65e0c2ff27e59ab6cd008`.

| inline v2 control | raw module | Brotli-q11 module | deterministic archive | delta vs oriented design surrogate | fixed flips to repay gate bytes at zero pose | fixed flips for `Delta S <= -1e-5` |
|---|---:|---:|---:|---:|---:|---:|
| zero identity gate | 2,391 B | 1,681 B | 188,035 B | +76 B | 60 | 72 |
| seeded nonzero capacity reference | 2,391 B | 1,643 B | 187,997 B | +38 B | 30 | 42 |

Each row retains raw bytes, Brotli bytes, a repeat, archive bytes, and an
archive repeat. The repeats are byte-identical and the raw v2 tensor stream
parses back exactly. The current v1 receiver intentionally rejects the v2
schema, so this is serialization/container authority only. Receiver changes
and exact identity must be tested before any candidate label.

The high-entropy nonzero reference compressing 38 B better than the all-zero
control is not an error: Brotli codes the whole module, including reordered
metadata and the prior tensor stream. It forbids estimating the final price as
“40 raw bytes” or selecting the favorable 38-B row as a guaranteed trained
price. For pre-fire budgeting, +76 B is the conservative observed control and
+128 B remains the hard design cap.

The trained EC2 archive is +1,471 B over CP135. If the 38–76 B surrogate delta
transferred additively, a BG1 successor would be 187,761–187,799 B, or
+1,509–1,547 B over CP135. That is only a **DERIVED projection**, not a
byte-closed row. At zero pose damage, those total prices require 1,186–1,216
fixed flips for score parity and 1,198–1,227 for `Delta S <= -1e-5`.
The actual trained adapter must be in local custody and reserialized inline to
replace this bracket.

For arithmetic hygiene, the remote selected receipt's stored
`break_even_required_flip_reduction=1155` is a rounded/floored control. Exact
ceiling arithmetic for +1,471 B is 1,156 flips at zero pose damage. The
charter's 1,341/1,353 gates correspond to the older +1,707-B oriented design
price: 1,341 for parity and 1,353 for nameability. None affects the EC2 refusal,
which misses by more than forty thousand flips.

## Contingent training specification — not sealed and must not fire

This specification exists so the postmortem has a bounded consumer. It is not
a fire order.

- Start from the sealed CP135 runtime and a retained EC2 checkpoint chosen only
  after the endpoint decomposition is available. Do not mutate or resume the
  completed EC2 run in place.
- Extend the same worker family with the v2 gate. Optimize the EC1 conditioner,
  gate, and existing 600x8 frame state jointly; keep all other CP135 and SegNet
  weights frozen. Rebuild the complete counted model and adapter archive at
  every stage boundary.
- Use the EC2 stage objectives and learning rates, shortened to 1,500 steps:
  600 target-birth steps at `1e-3`, 600 balanced-descent steps at `3e-4`, and
  300 seeded stratified collateral-finish steps at `1e-4`. The first two stages
  retain full-population coverage; the finish subset is seeded and non-prefix.
- Preserve the EC2 realized-through-R path: bilinear camera lift, uint8 STE,
  bilinear downsample, and frozen CUDA SegNet. Retain every pre-R, camera,
  scorer-input, logit, argmax, target, receipt, module, archive, and repeat.
- `--resume-from` is mandatory. Save live weights, EMA shadow, optimizer, all
  RNG states, typed configuration, and stage/step cursor atomically at every
  step. Preserve distinct packages and live+EMA checkpoints at all three stage
  boundaries. Never overwrite prior stage packages.
- Before the full endpoint, run a seeded stratified-random n=32 receiver screen
  on the same retained trained object with the true frame state, a seeded
  frame-state permutation, and zeroed frame-gate output. Exact-zero identity is
  a byte/pixel equality preflight, not another scorer pass. The shuffled screen
  tests the claimed frame mechanism without a second training run; fire the one
  n600 endpoint only if true state beats both controls at matched pose.
- Parse back the exact shipped v2 module and complete rebuilt model before the
  n600 endpoint. Admit only a receiver-identical identity control and a
  candidate whose exact `d_seg`, `d_pose`, and whole-container bytes clear its
  candidate-specific score and nameability gates.

EC2 measured 543.597070261 s total. Its last full endpoint portion was
94.207510737 s, leaving 449.389559524 s across 1,800 training steps, or a
derived 0.2496608664 s/step. Reusing the same endpoint allowance, 1,500 steps
project to 468.698810 s. Three n=32 control passes add a linear 15.073202 s,
for 483.772012 s total; proportional scaling of the lane's approximate $0.15
receipt is about $0.1335. The future fire order must cap the worker at 543.597070261 s
and $0.15, so it cannot exceed EC2's measured cost. These are
`[DERIVED from contest-CUDA T4 component timing]`, not a new runtime
measurement.

If the primary postmortem fails the frame-state association test, this entire
training specification is **FOLDED without firing**, and the Seg consumer is
the coupled multi-token #978 representation route.

## RECALL EVIDENCE

Recall was not limited to the charter seeds. The following surfaces were read
or searched before pricing:

- rfo1 H2 and fire-chain memo; EC1 design/runtime at commit `fa29eb9ea1`;
  EC2 dispatcher, worker, request, lane claim, call ledger, local return receipt,
  and recovered exact final receipt;
- JS8 singleton receipts, JS1C CUDA collateral decomposition, QS3
  `GT_ATTRIBUTED_DECOMPOSITION`, and QS4 spatial collateral memo;
- HP4 frame-embedding prediction, SR1 implicit conditioning, JS2b relative
  gauge, and JS3 learned implicit conditioning;
- `.omx/research/` content queries for `bilinear`, `frame_embed`, `600x8`,
  `oriented`, `introduced`, `collateral`, `B/H`, `edge conditioning`,
  `multi-token`, `same-cell`, and `neighbor-cell`;
- canonical equations via `tools/list_canonical_equations.py --json`, filtered
  for flip/byte exchange, realization, collateral, receiver survival, and
  composition; no registered equation replaces the retained endpoint field;
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED surfaces, design/SPEC
  documents, task-status/bridge surfaces, active lane claims, and
  `main_hot_state.md` for the overloaded `#978/#982/#984` campaign labels.

Findings beyond the charter seeds changed the plan as follows:

1. HP4 closes post-hoc lossless prediction of the current 600x8 frame table,
   but not interaction with that already-counted state. The gate therefore
   adds no new per-frame table and makes no compression claim.
2. SR1 closes small additive scalar/current-frame entropy calibration, while
   leaving joint nonlinear distortion conditioning open. The BG1 object must
   act inside the distortion renderer with the full 8-D state, not bolt on a
   post-hoc sign table.
3. QS4 found 60/76 harmful cells in neighboring rather than edited cells for a
   different six-pair explicit-edit instance. This makes spatial decomposition
   important, but does not transfer a frame-structure verdict to EC2.
4. JS3 showed that a learned hidden=4 implicit correction can cross a robust
   Seg sign while harming pose. The future endpoint therefore keeps an exact
   pose admission gate even though EC2 itself was component-only.
5. The current CP135/F26 receiver already contains semantic tokens, causal
   context, and the 600x8 frame FiLM state. “Add latent context” is not new;
   only the explicit low-rank interaction is the live BG1 hypothesis.
6. The task-number join is overloaded. This receipt preserves rfo1's campaign
   labels `#982` and `#984` and the alternative `#978`; it does not call any
   route ownerless.

## Measurement boundaries and frontier

Measured or exactly reverified this arm: the recovered EC2 final receipt and
its totals/remote hashes; the real EC1/CP135 tensor shapes; the two retained
int8+Brotli module streams; deterministic whole-container byte prices and
repeats; and exact score arithmetic for those byte deltas.

Not measured: the 52,854-error decomposition, any frame correlation, receiver
validity of schema v2, trained BG1 bytes, Seg/Pose response, full n600 candidate,
contest score, CPU/CUDA parity, or leaderboard movement. No scorer slot was
used. No training or paid dispatch occurred. Every payload materialized by the
price/recovery work is retained on the SSD tier with SHA-256 and byte count.

EFFECTIVE FRONTIER: `cp135 S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`; unchanged.  
OWN-VEHICLE FRONTIER: `lc2 S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`; unchanged.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER / BLOCKED-PRIMARY-HARVEST** — owner: operator or runner with authenticated Modal network access; consumer store: `/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/ec2_oriented/`; fire trigger: `api.modal.com` is reachable or the operator manually supplies the files; retrieve `endpoint/retained/argmax_n600.npy`, `endpoint/retained/BATCH_RECEIPTS.jsonl`, and `stages/selected/retained/ec1_latent.int8.br` from volume `comma-ddm-js1b-argmax-retained/ddm_ec2_oriented_20260814`, verify the three pinned hashes, and run the pre-registered postmortem before any training decision.
- **QUEUED-CONTINGENT** — owner: #982 BG1 successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_bg1_20260814/retained/` then #984 composition store; fire trigger: the recovered endpoint field shows significant held-out predictivity from the existing 8-D frame state after class/edge/spatial exposure control; implement receiver-valid v2, reprice inline on the trained adapter, claim the lane, and seal the <=543.597070261-s/<= $0.15 resumable retained train; if the trigger fails, fold BG1 and route the Seg axis to owner #978's coupled multi-token consumer store instead.

## LIVE-HYPOTHESES

- The already-counted 8-D frame state predicts which oriented EC1 corrections create collateral, so a four-channel multiplicative gate can preserve part of the 12,075 fixed cells while suppressing introduced cells. This is plausible because EC1 forms a useful edge-ranked correction before frame FiLM, but it remains untested until the endpoint field is recovered.
- The trained inline gate will remain below the +128-B cap. This is plausible because two exact surrogate controls cost only +38 and +76 whole-container bytes, though Brotli interaction with the missing 1,369-B trained module and any jointly changed frame table can move the final price.
- A seeded frame-state permutation will remove the benefit of a true BG1 gate without requiring a second train. This is plausible because the proposed mechanism is explicitly pair-state dependent; failure of that control would expose generic extra-capacity gain instead.
- If the collateral is not predictable from frame state, coupled multi-token semantic support can still attack it. This is plausible because that route changes spatial/semantic allocation jointly rather than asking a frame-global four-channel gate to solve unstructured local spill.

## DEAD-ENDS

- The exact EC2 plain oriented instance is closed: it fixed 12,075 cells but introduced 52,854, for -40,779 net flips, and cannot repay any positive byte price.
- Repeating the same frame-blind EC2 adapter with more capacity is closed for this base/schedule formulation: it duplicates the failed mechanism rather than adding the missing interaction.
- Aggregate fixed/introduced totals cannot establish frame structure. Treating 52,854 as evidence either for or against BG1 is closed because the required location field is absent.
- A bare “add semantic or latent context” proposal is closed as fake novelty: CP135/F26 already has semantic tokens, causal context, and a learned 600x8 frame FiLM state.
- Post-hoc prediction/repacking of the current frame table is closed for the HP4 tested formulations; all predictive variants enlarge the complete container, and BG1 may only reuse the state inside joint distortion conditioning.
- The current v1 EC1 receiver cannot be used as evidence for BG1 parse-back: it rejects the v2 tensor schema by construction, so the scorer-free priced streams are controls, not candidates.
- The older local `/Volumes/VertigoDataTier/pact/ddm_ec2` artifacts are closed as substitutes for the postmortem because their hashes and scorer-free schema identify a different run.
