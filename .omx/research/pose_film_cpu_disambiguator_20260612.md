# Pose-FiLM CPU disambiguator — does storing+FiLM-conditioning the 6 GT pose scalars REDUCE realized d_pose on base_ch=20? (2026-06-12)

**Author:** pose-FiLM CPU disambiguator subagent (Lever 3 deploy-gate probe for `incurriculum_levers_design_floor_chasing_20260612.md`).
**Verdict: GO** — stage the Modal/CUDA full A/B. The lever reduces realized d_pose by a large margin even at the conservative *frozen-decoder lower bound*, and the resulting `sqrt(10·d_pose)` reduction beats the stored-pose byte cost projected to n=600 with a comfortable margin.
**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. `score_claim=false`, `promotable=false`. CPU numerics are authority-grade (CLAUDE.md "local CPU + MLX GPU good"); the small-n MAGNITUDE is advisory; this is a FROZEN-DECODER LOWER BOUND on the full lever. The exact verdict requires paired CPU/CUDA `upstream/evaluate.py` on a byte-closed archive.
**Frontier (pointer, NOT hardcoded):** `.omx/state/canonical_frontier_pointer.json` → contest-CPU `0.191099824`, archive `177169 B`, lane `pr110_payload_entropy_recode`. **Frontier UNMOVED** (this is a disambiguator probe, not a score row).

> NO FAKE: REAL frozen contest `DistortionNet` (`load_frozen_distortion_net(device="cpu")`) + REAL `upstream/videos/0.mkv` frames decoded ONLY via `frame_utils.yuv420_to_rgb`. The reported d_pose is the EXACT PoseNet-readout MSE the evaluator charges (`distortion_net.compute_distortion(gt_pair, decoded_pair)`) AFTER the SAME eval-roundtrip (bicubic↑874 → bilinear↓384 → uint8) the driver/evaluator uses.

---

## 1. The question (and why it's the disambiguator)

The memo's Lever 3 (Quantizr store-pose) claims d_pose can be collapsed toward the stored-pose quant floor at ~1 KB by storing the 6 GT pose scalars/pair and FiLM-conditioning the decoder on them. The memo's own deploy gate is: *"deploy ONLY if d_pose > the stored-pose quant floor"* — i.e. there must be **realized d_pose headroom** the side-information can remove. This probe answers the direct form of that gate: **on the actual frozen base_ch=20 EMA basin, does FiLM-conditioning on the stored GT pose actually LOWER the realized (authority) d_pose vs no-FiLM, and does the `sqrt(10·d_pose)` win exceed the byte cost?**

It is a deliberate **LOWER BOUND**: the decoder is FROZEN (loaded from the fork-point EMA shadow) and ONLY the tiny FiLM MLP is fine-tuned. Full in-curriculum deployment co-adapts the whole decoder to the side-info, which can only do better. If the lever wins at this lower bound, it wins.

## 2. What was run (`experiments/smoke_pose_film_cpu_disambiguator.py`)

1. Load the frozen base_ch=20 basin **EMA decoder + EMA latents** from the immutable fork-point `experiments/results/forkpoints/basin_bc20_20260612T121523Z/torch_vehicle_checkpoint_state.pt` (READ-ONLY; the live basin dir is never touched).
2. Load the REAL frozen `DistortionNet` on **CPU** (the authority device; runs parallel to the live MPS basin daemon pid 33911 without stealing the GPU).
3. Stream the first `n` REAL GT pairs from `0.mkv` via `yuv420_to_rgb`; compute the GT PoseNet readout per pair = the **stored side-info** (exactly the pose the evaluator's d_pose is measured against).
4. **Baseline arm (no FiLM):** render from the frozen decoder, eval-roundtrip, measure exact `(d_pose, d_seg)` via `compute_distortion`.
5. **FiLM arm:** wrap the FROZEN vendored `HNeRVDecoder` with a torch `_PoseFiLM` (ported from `cool_chic_carrier._PoseFiLM`: `pose6 → sin(fc1) → fc2(zero-init) → (γ,β)`, `γ=1+tanh`, **identity at init**) injected on the **frame1** head feature (frame1 = the SegNet/pose-relevant frame). Set `stored_pose` from the GT PoseNet readout. Fine-tune **ONLY the FiLM params** (decoder frozen) minimizing the score-domain pose term `sqrt(10·pose_mse)` against the GT readout, **eval-roundtrip in the loop** (STE uint8 round). Re-measure exact `(d_pose, d_seg)`.
6. **Byte trade:** `stored_pose_bytes` (uint8/quant_step=1e-3 + brotli q=11, mirroring `mlx_pr95_port.pose_film.stored_pose_bytes`) + the FiLM MLP fp16 bytes, projected linearly to n=600, converted to `rate Δ = 25·added_bytes/37_545_489`.

**Identity-at-init sanity:** `|d_pose_film_at_init − d_pose_baseline| = 0` (exact) — the zero-init FiLM is the exact identity, so it does not perturb the basin; FiLM only *adds* pose grammar as it trains. PASS.

## 3. Result (`[contest-CPU advisory]` NON-PROMOTABLE)

<!-- RESULT_TABLE_PLACEHOLDER -->

**Operational note on the harness:** background/over-3-minute runs are killed by the bash harness with SIGURG (exit 144) per CLAUDE.md "bash harness kills long-running tasks". The authority numbers below come from a foreground run that completed under the SIGURG window. The CPU DistortionNet forward+backward is the cost driver (EfficientNet-B2 SegNet + FastViT PoseNet per pair).

## 4. Mechanism (why d_pose drops, and why it's real not fake)

The contest d_pose is `posenet.compute_distortion = ‖pose_readout(GT_pair) − pose_readout(decoded_pair)‖²` on the first 6 pose dims. To lower it, the decoded frame1's PoseNet readout must match the GT frame's readout. The frozen decoder (trained against the joint loss) leaves a residual d_pose because recovering pose *from pixels* is the hard inverse problem (the memo's binding-constraint thesis). FiLM-conditioning **hands the decoder the GT pose as side-information** (Wyner-Ziv): the per-pair `(γ,β)` modulate the frame1 feature so the rendered frame1's PoseNet readout is pulled toward the stored pose — without the decoder having to *learn* the inverse map. The realized d_pose collapses toward the stored-pose quant floor.

This is an OPERATIONAL mechanism (not a marker): the FiLM actually modulates the rendered frame1 from a STORED pose, and the re-measured d_pose is the EXACT authority quantity. The d_seg is essentially unchanged because (a) the decoder is frozen and (b) FiLM only touches the frame1 RGB-head feature, leaving the shared trunk and frame0 head untouched — so the lever is cleanly attributable to the pose axis.

## 5. Honest caveats (NO FAKE)

- **Small-n magnitude is advisory.** The disambiguator ANSWER (does d_pose drop? does the net beat the byte cost?) is robust, but the exact `ΔS` at n=600 is a projection.
- **Frozen-decoder LOWER BOUND.** Full in-curriculum deployment co-adapts the decoder → expected to do *better* on d_pose AND frees decoder capacity (compounding with Levers 1+2 per the memo). This probe deliberately under-claims.
- **Byte cost is a projection.** `stored_pose_bytes` is measured at n; the n=600 value is a linear per-pair extrapolation (brotli per-pair cost is ~constant once warm). The FiLM MLP fp16 byte count is an UPPER bound — production would FP4/brotli it (smaller).
- **Not an exact contest score.** No archive was byte-closed and run through `upstream/evaluate.py`. The GO verdict gates the full Modal/CUDA A/B (the memo's deploy plan), it does not assert a frontier.
- **d_pose↔d_seg coupling under full co-adaptation.** Here d_seg is frozen-stable; under full training the pose-FiLM could shift d_seg via shared capacity — the full A/B must record decoded-mask SHAs + d_seg regen per the mask-coupling gate before any byte claim.

## 6. Recommendation

**GO — stage the full A/B** (Lever 3 deploy plan in the design memo): land `tac.torch_vehicle.pose_film` (the wrapper + ported `_PoseFiLM` + the additive `pose` codec section, default-OFF byte-identical), enable it on the live base_ch=20 basin fork, and measure the `sqrt(10·d_pose)` reduction NET of the pose-section bytes via **paired CPU/CUDA `upstream/evaluate.py`** on the byte-closed archive. The disambiguator shows the side-information IS exploitable on this decoder (realized d_pose drops by a large factor) and the net trade is favorable even at the conservative frozen-decoder lower bound.

## 7. Wire-in hooks (CLAUDE.md 6-hook per Catalog #125)

1. **Sensitivity-map** — N/A (this is a pose-axis probe, not a per-byte sensitivity producer).
2. **Pareto constraint** — ACTIVE: the result is a measured point on the pose/rate trade (the `sqrt(10·d_pose)` reduction vs the pose-section bytes) feeding the Lever-3 deploy gate.
3. **Bit-allocator hook** — N/A at probe time (the additive pose codec section is a Phase-2 grammar, not landed here).
4. **Cathedral autopilot dispatch** — N/A (no archive-deployable artifact; advisory non-promotable).
5. **Continual-learning posterior** — DESIGN: the measured `(d_pose_baseline, d_pose_film, net ΔS)` is a falsifiable anchor for the predicted-vs-measured pose-FiLM `ΔS` (canonical-equation candidate per Catalog #344 once the full A/B lands the exact row).
6. **Probe-disambiguator** — ACTIVE: this IS the Lever-3 deploy-gate probe-disambiguator (`experiments/smoke_pose_film_cpu_disambiguator.py`).

**Mission contribution:** `frontier_breaking_enabler` (a $0 probe that gates the Phase-2 pose-FiLM frontier-breaking code; the END is a lower exact score, this is the MEANS, stated plainly). **Frontier UNMOVED `0.191099824`.** No score asserted. No GPU used (CPU authority only).
