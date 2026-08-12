# ddm_js3 learned implicit conditioning — robust motion exists, but not with the pose guard yet

## Result first

The exact pointer did not move. This arm built the representation-changing learned conditioning path, retained every stage object, and completed two bounded stratified-random n32 screens through the current CP135 receiver delta, camera uint8 lattice, frozen CPU SegNet, and custody PoseNet planes.

The useful signal is narrow:

- The 20-step hidden-4 screen crossed the robustness bar. Its parse-backed 751 B int8+Brotli-q11 module produced 44 robust beneficial and 24 robust harmful flips on n32: net `-20` robust flips, stratified projection `-377` at n600.
- That same object is disqualified. It worsened total flips by `+79` on n32 (projection `+1,467`) and regressed pose by `+0.0107992463`, far beyond the `2e-6` guard.
- A clean 8-step high-pose-weight control selected a parse-backed 819 B EMA module that passed pose (`-1.0120974e-6`) and improved total flips by `-3` on n32 (projection `-57`), but produced zero robust movement.

Therefore the gradient is not dead through the receiver/R/uint8 chain, but this bounded screen did not produce a module that combines robust Seg movement with the pose guard. The result is **BOUNDED EXISTENCE / NOT ADMITTED**, not a FORMULATION or FAMILY negative. No T4 row was justified or dispatched.

## Measured rows

Axis for every row below: `[macOS-CPU advisory, instrument floor 0.0131 S]`. The sample is the sealed js2b seeded stratified-random n32 draw; stratum weights alternate 18/19 and sum to 600. Baseline sample errors are 2,686. All module sizes are real Brotli quality-11 payload bytes after parse-back.

| row | steps | selected object | coded bytes | n32 total delta | projected n600 total delta | n32 robust delta | projected n600 robust delta | pose delta | gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| ordinary joint smoke | 20 | live int8 parse-back | 751 | +79 | +1,467 | **-20** | **-377** | +0.0107992463 | fail total + pose + robust threshold |
| pose-heavy control | 8 | EMA int8 parse-back | 819 | -3 | -57 | 0 | 0 | **-1.0120974e-6** | fail robust threshold |

The ordinary row had 276 beneficial / 355 harmful flips; 44 beneficial / 24 harmful were robust at `delta = 0.08036041259765625`. Its 232 other beneficial flips were tie-fragile. The control had 8 beneficial / 5 harmful flips, all tie-fragile.

Measured training throughput was 12.8494 s/step for the ordinary schedule and 13.5937 s/step with the PoseNet guard every step. The sealed 300-step pose-guarded schedule projects to 67.97 minutes before stage-evaluation overhead. The bounded runs themselves stayed inside the charter's 30-minute wall.

## Objective and module derivation

Facts fixed the form:

1. js2b measured the conservative transfer bar `delta = 0.08036041259765625`; its nine two-code FiLM continuation states produced only tie-fragile repairs, so direct continuation is forbidden.
2. sr1 measured that standalone additive edge probability context saves only 2 B and scalar pose context costs 43 B. The remaining open route is representation-level distortion conditioning.
3. CP135's master renderer works on decoded semantic tokens at 384x512, then lifts bilinearly to 874x1164 and rounds on the camera lattice. The learned action must precede that lattice.
4. The local instrument is a relative gauge. Candidate action is transported onto the retained T4 custody input as a same-object delta; no absolute local score is promoted.

For pair `i`, the receiver-free context is

`phi_i = [one_hot(tokens_i), four neighbour edge indicators, x, y, receiver_RGB_i]`.

The smallest implemented learned family is

`c_theta(phi) = A * tanh(W_head GELU(W_dw GELU(W_context phi)))`,

with `A = 6` RGB units, a 3x3 context convolution, 3x3 depthwise convolution, and 1x1 RGB head. Weights are fake-int8-quantized in the forward pass; live and EMA states are exported independently. The receiver-chain action is

`R_theta = bilinear_down(Q8(bilinear_camera(receiver_RGB + c_theta)))`,

and the relative custody input is

`x_theta = x_custody + (R_theta - R_0)`.

For frozen SegNet logits `z`, GT label `y`, and signed GT margin

`m_y(z) = z_y - max_{k != y} z_k`,

the implemented Seg loss is

`L_seg = mean_{base-wrong} [delta - m_y(z)]_+ + mean_{base-correct} [delta - m_y(z)]_+`.

The first term rewards repairs beyond the transfer bar. The second prices collateral at the same bar. The PoseNet guard is the actual custody-plane term

`L_pose = mean_i [MSE(PoseNet(T_custody + Delta T_theta)_i, p_i*) - e_i(base)]_+`,

not an RGB proxy. Training minimizes `L_seg + lambda_pose L_pose` through camera uint8 STE and both frozen scorers.

## Capacity and byte ladder

The module has no per-frame learned payload and transmits no edge mask. Only learned weights/biases are counted. Initial deterministic modules were raced as fp16 and symmetric-int8 using real Brotli q11; both modes parse back to the exact receiver parameter arrays.

| hidden width | parameters | selected mode | initial real coded bytes | <=1,500 B gate |
|---:|---:|---|---:|---|
| 4 | 563 | int8 | 798 | yes |
| 8 | 1,123 | int8 | 1,282 | yes |
| 12 | 1,683 | int8 | 1,825 | no |

The trained hidden-4 payloads compressed to 751 B (ordinary live) and 819 B (pose-control EMA). The 20-step row's linearized price is 1.992 B per projected robust flip, or about 3,984 B for 2,000 flips if its weak scaling transferred. That is an INSTANCE warning only. F2 requires every capacity rung and did not become eligible.

## Falsifiers and disposition

- **F1 NOT ELIGIBLE / NOT FIRED.** The maximum trained horizon was 20 steps, not the charter's 300. Robust movement was nonzero on the ordinary row, so the bounded evidence also does not have the zero-movement shape.
- **F2 NOT ELIGIBLE / NOT FIRED.** All rungs were priced, but only hidden=4 was trained. The hidden-4 instance is weakly priced, but no every-rung conclusion is allowed.
- **F3 NOT ELIGIBLE / NOT FIRED.** Only hidden=4 was trained. The high-pose-weight EMA control passed the pose guard, so even this rung does not show an unavoidable pose wall.

The T4 gate required projected robust delta at most `-2,000`, coded bytes at most 1,500, and pose delta below `2e-6`. The ordinary row missed robust by 1,623 flips and failed pose; the pose-control row missed robust by 2,000. No paid dispatch occurred.

## Resumability, retention, and receiver boundary

The runner is `experiments/ddm_js3_learned_implicit_conditioning.py`; tests are `experiments/tests/test_ddm_js3_learned_implicit_conditioning.py`. Checkpoints are atomic and contain live weights, EMA shadow, optimizer, torch/numpy RNG state, complete mechanism config, step history, and stage identity. Resume fails closed when mechanism config differs. Periodic and per-stage checkpoints are distinct files.

Every stage retains:

- complete live and EMA checkpoints;
- fp16 and int8 raw module payloads plus every Brotli-q11 output;
- parse-backed coded candidate camera frames, corrections, logits, argmax arrays, and Pose errors;
- float-QAT counterparts for exact export-drift comparison.

The retained root is `/Volumes/VertigoDataTier/pact/ddm_js3_20260812/`, about 3.1 GiB across 128 files at completion. No materialized payload was discarded. Key custody:

- main checkpoint: 29,638 B, SHA `b551fbd8d22e30fbdffab1ab5684921d66125830610afa8309c371f3891a4829`;
- main selected module: 751 B, SHA `e8c29f39f5fa34aefa492bfcfd2d89b0aee917ff00f8b9f798ade0c08af328ca`;
- pose-control checkpoint: 28,550 B, SHA `13cffc8800073666cdf388caeca72d48c7cba078b3f9838513fb9fa55afcaf85`;
- pose-control selected module: 819 B, SHA `6ee17cd997fb0df345da20815d6e9b61dd15b70727c4a4453b18499a08aa21ea`;
- compact final handoff: `/Volumes/VertigoDataTier/pact/ddm_js3_20260812/FINAL_HANDOFF.json`, 3,135 B, SHA `14d36f89b34802e2e2e71fc88c6fa5f6caed9b48728f0d9d1da072596a9936ee`;
- sealed MAIN recipe: `/Volumes/VertigoDataTier/pact/ddm_js3_20260812/SEALED_MAIN_RECIPE.json`, 2,029 B, SHA `22fbe3d41ed5a64f1af0641bb8ddf45514c5e16a18905055e2f6b96d0ad9cd95`.

This build-to-admission arm intentionally did not patch the production receiver or build a candidate archive. The module computation is implemented and trained through the real current-vehicle delta/R/uint8 chain, and its learned bytes are real-coded, but a future T4 candidate still owes complete receiver consumption plus an n600 local projection. Calling either retained module an archive or contest score would be fake.

## RECALL EVIDENCE

The governing charter, common contract, PROGRAM, full local AGENTS/CLAUDE rules, operating manual, live board, evaluator, lane surfaces, current CP135 runtime, and retained js2/js2b custody were read before implementation.

The full retrieval surface was then queried for `learned implicit conditioning edge context`, `CP135 pose guard robust margin`, `quantize compensate semantic FiLM`, and `Road incident boundary current vehicle`. Each query consulted research (8,415 rows), equations (886), memory (2,110), DAG (915), council (297), tasks (531), and docs (96), without truncation.

Direct decision records included sr1, fd135, js1, js2, js2b, hr1, rvs1, the CP135/F26 receiver, ExperimentBook surfaces, the canonical research index/DAG/task-ledger surfaces, and canonical equations including `seg_rate_breakeven_v1`, `realization_breakeven_bytes_v1`, and the scorer-conditional joint RD laws.

Findings beyond the charter seeds that changed the implementation:

1. sr1 already measured F26's standalone edge and scalar-pose probability contexts; rebuilding a post-hoc table would duplicate a closed formulation. The module therefore acts on distortion before the camera lattice.
2. CP135's retained master receiver uses a bilinear camera lift, not the broader HR1 bicubic template. The runner follows the actual current vehicle and types this difference in the receipt.
3. js2b's working relative gauge transports receiver-realized deltas onto exact custody scorer inputs. Reusing that gauge avoids pretending the local raw stream is the promoted raw stream.
4. prior quantize-then-compensate results show export state must be scored after hard quantization. The screen therefore evaluates the exact parse-backed selected Brotli payload, not only float training weights.
5. pose-sensitive high-frequency structure makes an RGB-norm proxy unsafe. The control consequently keeps PoseNet in the actual differentiable loop and selects pose-passing EMA before Seg ranking.

No canonical equation displaced the charter's measured delta or T4 gates. No foreign-paper percentage or ancestor-vehicle score was transferred.

## Borrowed-substrate accounting and claim boundary

Borrowed in-repo/granted substrate: CP135 archive and receiver, PR135/F26 semantic renderer, retained tokens and raw, js2b sample/calibration and relative-gauge transport, upstream frozen scorers, GT cache, and custody scorer planes.

Original ddm_js3 work: the edge/context learned module, delta-hinge/collateral objective, actual Pose guard coupling, QAT export grammar and parse-back, resumable stage runner, capacity price ladder, two bounded screens, and sealed recipe.

Measured here: n32 scorer deltas, robust flip counts at delta, pose deltas, training throughput, real coded module bytes, deterministic parse-back, hashes, and retained payload inventory.

Not measured: a complete receiver-integrated archive, n600 trained action, full-n600 local scorer row, contest CPU/CUDA score, exact archive delta, or frontier movement.

The effective frontier remains **cp135 `S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`**. The own-vehicle frontier remains **lc2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`**. This unit did not reach sub-0.15.

## Queued fire order (verbatim)

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js3_20260812/main_burn`. **Fire trigger:** MAIN owns the training leg, no full-n600 scorer job is active, the target-host memory preflight passes, and js2b `ROUTE.json` is consumed as a prohibition on direct two-W4 continuation.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js3_20260812/main_burn`. **Fire trigger:** MAIN owns the training leg, no full-n600 scorer job is active, the target-host memory preflight passes, and js2b `ROUTE.json` is consumed as a prohibition on direct two-W4 continuation. Run the sealed pose-guarded 25/100/300-step recipe from a clean checkpoint; admit T4 only after complete receiver consumption and an n600 local projection passes all three gates.

## LIVE-HYPOTHESES

- A longer pose-guarded schedule can move the control EMA from tie-fragile repairs into delta-robust repairs. This is plausible because the ordinary schedule proved the same module can cross delta, while the high-pose-weight control proved EMA can preserve pose and improve total flips.
- Hidden width 8 may improve robust repairs per byte before the 1,500 B wall. Its real initial coded size is 1,282 B, leaving 218 B of gate headroom, and it has twice the shared context capacity without transmitting a mask.
- A stage-boundary schedule that first locks the pose-safe EMA basin and only then raises the delta-hinge weight may dominate simultaneous descent. The two bounded rows expose a tradeoff between robust motion and pose, not zero reachability on either axis.

## DEAD-ENDS

- Direct two-W4 FiLM continuation is not retried: js2b closed its fixed nine-seed catalog because every beneficial flip was tie-fragile and the selected exact-coded instance cost bytes for zero robust movement.
- Standalone additive edge or scalar-pose probability tables are not retried: sr1 measured only -2 B and +43 B respectively on the full F26 token stream.
- Selecting float-QAT weights without scoring their coded parse-back is closed: the receiver consumes the coded weights, and the 20-step live object showed that the parse-backed state must be the verdict object.
- Treating ordinary flip improvement as robust progress is closed: the pose-control row improved total flips but had zero delta-robust movement.
- Dispatching either bounded object to T4 is closed: neither passes the combined robust, byte, and pose gate, and neither has complete receiver/archive integration.
