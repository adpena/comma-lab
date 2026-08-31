# DDM BLP1 — the retained r10 born-Lane predictor closes at its weight floor

`axis: [macOS-CPU scorer-free exact rate and receiver-dependency measurement]` · `score_claim: false` · `$0` · no scorer · no Modal · no checkpoint load · `upstream/` untouched

## Verdict

**CLOSED-AT-FLOOR.** The smallest real-coded serialization of the retained r10 tensors that the QBFLOW receiver consumes to produce class logits is **60,191 B**. That weight payload alone is **24,147 B over** the 36,044 B D3B reopening trigger and **38,492 B over** the 21,699 B campaign door, before one exact Lane residual bit is charged.

The charter therefore stops at Stage 0. It does not authorize loading the r10 checkpoint, materializing 600 logit fields, running the 32/568 conditional-coder race, composing an archive, or calling a scorer. The D3B reopening trigger did not fire.

`verdict_scope: INSTANCE` — this closes the born-Lane exact-conditioner proposal at the retained r10 endpoint. It does not by itself close every learned Lane predictor. A `FAMILY` disposition requires the ltg1 sibling row joined with this result, as preregistered.

## Stage 0 arithmetic

| charged object | bytes | authority |
|---|---:|---|
| receiver-consumed r10 predictor tensors, Brotli q11 | **60,191** | real retained coder payload; exact decode |
| optimistic n32 residual entropy | 5,581.5605 | estimate only; false-positive-free, in-sample, independent-error upper entropy at the observed error mass |
| optimistic Stage 0 sum | **65,772.5605** | mixed: measured weights + estimated residual |
| GF1/D3B reopening trigger | 36,044 | retained incumbent bar |
| complete Lane carriage door | 21,699 | campaign bar |

The closure does **not** depend on the entropy estimate: `60,191 > 36,044` before the residual. Even a hypothetical zero-byte residual leaves the instance 24,147 B underwater.

The 64,276 B D3B exact incumbent remains the exact packet to beat. BLP1's 60,191 B is not a 4,085 B exact win over it because BLP1 has not coded the exact residual; comparing weights alone to D3B's complete packet would be a wrong-denominator claim.

## Real weight serialization and coder race

The retained `reencode_payloads.tar::packet.qbf` was decoded through the in-tree QBFLOW packet parser. The measurement preserved the original quantized codes and retained precisions; it did not requantize float weights. The minimal class-logit dependency set is all `coarse_*`, `flow_*`, and `step_*` tensors, with `flow_head_w` sliced from `104×26` to the ten interface outputs `104×10` and `flow_head_b` sliced from 26 to ten. Interior, render, pose, and the 16 boundary-feature outputs are excluded because they do not feed `class_logits`.

| coder | coded bytes | SHA-256 | exact decode | repeat |
|---|---:|---|:---:|:---:|
| **Brotli q11** | **60,191** | `9f17847ae16228ab110560dec1e14a973732ed05eddc06a91c352f887981b88d` | yes | byte-identical |
| LZMA9 extreme | 60,932 | `03d4c5b8930606bb8e5238b2e8af9375db035c50ad1fd133a64b145badb52e01` | yes | byte-identical |
| zlib 9 | 62,103 | `bf51170aeab64ba0d18fb37f5e118060ee14949671101d67d719a2106e568eea` | yes | byte-identical |

The minimal raw QBT1 table is 66,488 B, 60,087 scalar codes, and 65,622 packed-code bytes. It is retained with a byte-identical repeat at SHA-256 `8538e844c84c13d973eed2421c96179c926ead5bcedfae41766931323174e69b`.

Two receiver controls, pair IDs 4 and 573 at `24×32`, produced bit-identical `class_logits` after all excluded tensors were zeroed and the two flow-head tensors were restricted to their first ten outputs. Their output SHAs are `f1bcd069…3edd` and `a7b42007…ef40`. These are dependency/serialization controls, not a population-accuracy claim.

All raw bytes, all three coded candidates, and every deterministic repeat are retained under `/Volumes/APDataStore/pact/ddm_blp1/stage0/retained/`. The machine-readable receipt is `/Volumes/APDataStore/pact/ddm_blp1/stage0/STAGE0_RESULT.json`, schema `ddm_blp1_stage0_weight_floor.v2`, SHA-256 `1a72679976f6277b5717dc89e76e312c1bcc9ea282566d2c4f65b315184fb2fa`.

## Optimistic in-sample residual estimate

The retained r10 endpoint telemetry reports Lane within-class error `werr = 0.11357018054746651` on the n32 training set. The exact shipped-field Lane mask has 37,550 positive pixels in the selected n32 pairs inside the global support band rows `[158,294)`. To make the estimate favorable to the predictor, Stage 0 assumes every error is a false negative and that there are zero false positives.

```text
support positions N = 32 pairs × 136 rows × 512 columns = 2,228,224
optimistic errors   = 37,550 Lane ones × 0.11357018054746651
                    = 4,264.560279557368
p over support band = 4,264.560279557368 / 2,228,224
                    = 0.0019138831102965266
H2(p)               = 0.020039495209792306 bits/position
N × H2(p)           = 44,652.48417434425 bits
                    = 5,581.5605217930315 B
```

This is an **OPTIMISTIC-IN-SAMPLE estimate**, not a real coded residual and not a Stage 1 result. It can only help close Stage 0; it cannot clear a gate. False positives, bin/context headers, held-out degradation, and exact-coder overhead are all omitted. The weight-only row makes those omissions immaterial to this instance verdict.

## Denominators

- **Training population:** 32 seeded-stratified pair IDs: `4,31,49,52,62,90,100,113,128,148,173,179,186,187,214,236,256,260,268,278,326,328,341,352,368,382,444,456,483,508,563,573`.
- **Held-out population:** 568/600 pairs, not run because Stage 0 failed before checkpoint load. No pooled 600-pair estimate is reported.
- **Support-band denominator:** 2,228,224 binary positions = 32 pairs × 136 rows × 512 columns.
- **Lane-positive denominator:** 37,550 exact positives in those n32 support bands. The 4,264.5603 error count is a telemetry-derived expectation, not an observed integer residual count.
- **Weight denominator:** 28 receiver-consumed tensor records, 60,087 scalar quantized codes at their retained role precisions; the real coder races the entire 66,488 B QBT1 table including names, shapes, scales, and framing.
- **Trigger comparison:** complete candidate cost must be predictor weights + exact residual + envelope. The canonical residual-hybrid equation is `K_predictor + R_residual + H_envelope`; residual bytes are never free.

## pk3/pk4 prior-law outcome

The preregistered pk3/pk4 prediction was held-out/in-sample conditional entropy `≥2×`, with `<1.3×` as a law update. **Neither ratio was measured.** Stage 0 failed before any held-out logit or conditional stream was authorized, so the prior law is **NOT ADJUDICATED** by BLP1. Reporting the predicted collapse as observed would be fake.

The prior remains plausible because pk3 had 23/23 in-sample-improving exact-valid overlays but 0/23 leave-one-pair-out improvements, and pk4's 48-train/16-held-out optimal-form rungs were all held-out negative or zero. BLP1 closes earlier on weight economics and contributes no new generalization evidence.

## Reading semantics and provenance

- The target is the exact Lane restriction of the **shipped source token field**, not GT. Source field SHA-256: `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`; exact Lane packbits SHA-256: `6ca82a7883411d0eb27addac7dcf662e84d2f9cc66404c299da2e15761c0e0cf`. The gti1 GT-lineage fork does not apply.
- Source retained packet container: 2,723,840 B, SHA-256 `18d69e4da2024d39ef13e73ef92623ca9857e67cc4f7b551f83d557f9880709d`; `packet.qbf` SHA-256 `607abebd…19f7`; decoded model section 87,854 B, SHA-256 `48bf43db…4d38`.
- r10 authorization config: SHA-256 `87eff6e8cc0339c8b669de9f714e8c666d13a9a8f406a396245540e774c200e9`; prior memo lineage pins canonical config `36a40bdf…`, identity `17a0769a…`, initialization state `8d35cbcfc00fa49f4145b573eb1b4d5d787d1605d823615dd7bb1b9313096e0f`.
- Measurement source module: `experiments/ddm_qbflow_packet.py`, SHA-256 `cdf90d1a4d7d13001118f50a76692c04605f8e5ae9a7816c80f6e346160c7b9c` at measurement time.
- Repo HEAD before the memo edit: `692ec6e7da4e306c6b227b2113d1edf12ab52680`.

## RECALL EVIDENCE

The recall pass searched `.omx/research/` memos and arm receipts, retained APDataStore receipts, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/spec surfaces, `main_hot_state.md`, task-ledger text, the canonical-equations registry/source, and the falsified-premise corpus. Content queries included `born Lane`, `Lane head`, `exact Lane`, `conditional entropy`, `paid model`, `47.4`, `D3A`, `D3B`, `GF1`, `36,044`, `21,699`, `64,276`, `generator form`, `2.178`, `pk3`, `pk4`, `heldout`, `token-stream-is-one-binary-question`, and the exact source/mask SHAs.

Beyond the charter seeds, recall found:

1. The QBFLOW receiver's class-logit path consumes the whole coarse/flow/step trunk, not merely the final Lane output matrix. This changed Stage 0 from a misleading final-layer parameter estimate into a real serialization of every transitive receiver dependency.
2. The final flow head has 16 outputs used only by the later renderer. This changed the first conservative 62,626 B serialization into the exact minimal 60,191 B class-logit row while preserving bit-identical controls.
3. The canonical equation `procedural_predictor_plus_residual_correction_savings_v1` explicitly charges `K_predictor + R_residual + H_envelope`. This prevented the born field from being priced as free generation and prevented its weights from being compared as a complete packet against D3B.
4. LC3 already measured the exact D3B packet at 64,276 B and named 36,044 B as the materially-different-predictor reopening trigger. No prior under-36,044 B receiver-consumed born-Lane predictor was found in the searched corpus.
5. The older image-space/openpilot Lane-head references were roughly 65 KB/600 and did not carry the same object or receiver contract, so they were excluded from the verdict rather than treated as corroborating measurements.

The requested `tools/list_canonical_equations.py --json` registry command stalled while importing SciPy in this sandbox and was interrupted without a result. Static inspection of the registered equation module and `__init__` recovered the applicable law above. That is a bounded apparatus failure, not evidence that the registry lacks other equations.

## What was and was not measured

**MEASURED:** exact minimal receiver-dependency serialization; three real coder sizes; exact decoder round trips; deterministic repeats; two class-logit dependency controls; retained artifact bytes and SHAs; exact source-mask SHA verification.

**NOT MEASURED:** r10 checkpoint load; 600-pair logits; exact conditional residual; any logit-bin scheme; the 32/568 entropy ratio; any population generalization result; archive composition; d_seg; d_pose; complete S; contest-CPU; contest-CUDA. No frontier or score claim follows from this rate-only Stage 0 close.

## Follow-on dispositions

- `FIRED → CLOSED-AT-FLOOR` — r10 born-Lane exact-conditioner instance; owner `ddm_blp1`; consumer store `/Volumes/APDataStore/pact/ddm_blp1/`; trigger was the chartered Stage 0 real weight floor; result 60,191 B before residual.
- `FOLDED` — BLP1 Stage 1/2 checkpoint, logits, 32/568 coder, and composition rungs; owner `ddm_blp1`; consumer store `/Volumes/APDataStore/pact/ddm_blp1/`; fire trigger would require a new retained serialization of the receiver-consumed predictor below 36,044 B before residual. The current endpoint does not satisfy it.
- `FOLDED` — D3B reopening and sub-0.12 compose fire-order; owner `MAIN`; consumer store `/Volumes/APDataStore/pact/ddm_blp1/`; trigger required total predictor-plus-exact-residual below 36,044 B, or below 21,699 B for the campaign door. Neither fired.
- `QUEUED-WITH-A-FIRE-ORDER` — joint family-scope disposition; owner `MAIN`; consumer store `/Volumes/APDataStore/pact/ddm_blp1/`; fire when the ltg1 sibling publishes its terminal real-coder floor, then join that row with this 60,191 B instance and label only the supported shared scope.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN`; consumer store: `/Volumes/APDataStore/pact/ddm_blp1/`; fire trigger: the ltg1 sibling's terminal real-coder receipt is available. Join it with this instance row to decide the preregistered family scope; do not load the r10 checkpoint or run BLP1 Stage 1 unless a materially smaller receiver-consumed weight serialization first clears 36,044 B.

## LIVE-HYPOTHESES

- A genuinely shared or generically generated Lane predictor could still reopen D3B because BLP1's optimistic residual estimate is only about 5.6 KB; the failure is dominated by 60.2 KB of counted weights, so collapsing or sharing the predictor—not refining its residual coder—is the plausible lever.
- The ltg1 topology/event sibling may still separate the family verdict because it generates Lane structure with a different counted object. Its plausibility rests on eliminating most learned-weight tax while charging exact event/shape residuals honestly.
- The pk3/pk4 held-out-collapse law may also hold for born fields, but BLP1 did not test it. It remains worth testing only after a future predictor clears the weight floor; before then, the 32/568 race cannot change the decision.

## DEAD-ENDS

- The retained r10 born-Lane head plus its transitive coarse/flow/step trunk as an exact-coding conditioner: 60,191 B before residual, already 24,147 B over the reopening trigger.
- Charging only the final Lane-output matrix: the real receiver uses the full transitive trunk, so that price would omit paid learned content.
- Charging renderer-only tensors or 16 unused flow-head outputs: dependency inspection and bit-identical controls show they are outside the class-logit object; the tightened row already removes them.
- Running the 600-pair logit/binning/held-out race on this checkpoint: Stage 0 closes even at zero residual bytes, so Stage 1 cannot reverse the verdict and is forbidden by the charter.
- Treating the 5,581.56 B entropy estimate as a coded residual or claiming a pk3-law ratio: neither stream nor held-out population was measured.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; BLP1 ran no scorer and did not move the pointer.`
