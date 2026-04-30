# Recoverable Lanes — Re-Engineering Plans (2026-04-30)

**Author:** ALL-SCORES-FORENSIC-AGENT (#298)
**Companion files:**
- Inventory: `.omx/research/all_scores_inventory_20260430.md`
- Forensic audit: `.omx/research/all_scores_forensic_audit_20260430.md`

**Mandate:** for every ENGINEERING_BUG / CONFIG_BUG / METHODOLOGY_BUG lane in the audit, a specific re-engineering plan with cost + predicted band + council voice support.

**Ranking metric:** `EV_per_dollar = (current_score - predicted_score) / dispatch_cost_USD`. Higher is better. Predicted scores are PREDICTIONS not measurements — tag as `[prediction]` everywhere.

---

## TIER 1 — re-engineer immediately (highest EV)

### #1 Lane Q-FAITHFUL — true 1:1 Quantizr (88K JointFrameGenerator)

- **Original outcome:** 2 dispatches, both crashed in dispatch (argparse `--tag required` v1; `auth-eval-on-best` vs variant-validator gate ordering v2). Score: NEVER MEASURED.
- **Bug:** v1 missing CLI flag; v2 codex R5-2 #1 — gate fired before subprocess argparse validation.
- **Fix:** add `--tag lane_q_faithful` to dispatch; pass `--no-auth-eval-on-best` AND run a variant-specific eval afterwards via `_VARIANTS_BUILD_RENDERER_FP4A_OK` whitelist OR switch variant to FP4A-exportable.
- **Council voice:** Quantizr (architect of leader 0.33). Per memory `project_lane_q_faithful_design_20260428`, predicted band [0.40, 0.80] [contest-CUDA] — could be the SECOND sub-1.0 lane after Lane G v3.
- **Cost:** ~$3 Modal T4 (12h). ~$1.50 Vast.ai 4090 (5h).
- **Predicted post-fix:** [0.40, 0.80] [prediction]. Council central 0.55.
- **EV/$:** (1.05 − 0.55) / 1.50 = **0.33** points-per-dollar. **TIER 1.**

### #2 Lane V / V-V2 — Quantizr replica 88K + half-frame

- **Original outcome:** crashed at conv channel mismatch ($1.90 wasted). Score: never measured.
- **Bug:** `RuntimeError: input[1, 1, 384, 512] expected 3 channels but got 1` — HintedRenderer alpha_map blend_conv constructed with in_channels=6 but receives 2 (1+1) when masks are passed instead of frames.
- **Fix:** instrument with print + run `--epochs 1 --batch-size 1` smoke; trace mask channel dispatch in `MaskRenderer.embed` vs `HintedRenderer.alpha_map`. Per prior audit memory `project_killed_lanes_forensic_audit_20260428`, the cleaner rebuild is **Lane H-V3** (Lane G v3 anchor + curriculum mask_half_sim_prob 0.0→1.0 over 200 epochs + use_zoom_flow=True).
- **Council voice:** Hotz + Quantizr. Quantizr ships half-frame at 0.33; Lane H-V3 predicted band [0.55, 0.95] [contest-CUDA] per memory `project_lane_h_v3_revival_design_20260428`.
- **Cost:** ~$2 Vast.ai 4090 (8h with smoke).
- **Predicted post-fix (Lane H-V3):** [0.55, 0.95] [prediction]. Council central 0.75.
- **EV/$:** (1.05 − 0.75) / 2 = **0.15** ppt/$.  **TIER 1.**

### #3 Lane SegMap clones (SC++, SA, SO) — 9 lanes, 1 root bug

- **Original outcome:** 12 dispatches, 0 outputs. SegMapTrainer OOM at `(B*T, 3, H, W)` rendered tensor.
- **Bug:** `src/tac/segmap_renderer.py:284` materializes the entire batch in float32. T4 14.56GB / A10G 22GB shared = OOM.
- **Fix:** chunk the batch dim (max_chunk_bytes ≈ 2GB float32 → 14 frames at 384×512×3) OR `bf16` autocast. Council C bf16+scorer-chunk fix dispatched on Vast.ai 4090 tonight ($3.12, 12h overnight per `project_session_state_checkpoint_20260430`). **CONFIRM POST-HARVEST.**
- **Council voice:** Selfcomp (architect of the 94K SegMap paradigm at 0.38 score) + Hotz.
- **Cost:** $0 (already in flight). $3.12 paid.
- **Predicted post-fix:** SC++ band [0.30, 0.55] [prediction] per session-state memo. SA + SO similar.
- **EV/$:** if SC++ lands at 0.40, (1.05 − 0.40) / 3.12 = **0.21** ppt/$ for the entire 9-lane class. **TIER 1.**

### #4 Lane MAE-V (MAE-style patch masking with Gumbel-softmax token)

- **Original outcome:** crash at import (`ModuleNotFoundError: pydantic`).
- **Bug:** `pydantic` not in Modal image. Pure infra.
- **Fix:** add `pydantic` to `experiments/modal_train_lane.py` image build. ~10 LOC.
- **Council voice:** Hinton (knowledge distillation), van den Oord (VQ-VAE patch tokens). Per provenance: predicted band [0.85, 1.10] [prediction].
- **Cost:** ~$1.50 Vast.ai 4090 (5h).
- **Predicted post-fix:** [0.85, 1.10] [prediction]. Central 0.95.
- **EV/$:** (1.05 − 0.95) / 1.50 = **0.07** ppt/$. **TIER 2** (lower-EV but trivial fix).

### #5 Lane FL (RAFT-derived poses)

- **Original outcome:** OOM-killed (rc=137) on T4 at `derive_poses_from_raft.py --device cuda --n-frames 1200`.
- **Bug:** RAFT model materializes all 1200 frames at full res (1164×874×3×4B = 12.2MB/frame × 1200 ≈ 14.6GB = exceeds T4).
- **Fix:** chunk RAFT inference: `--n-frames 100` × 12 batches with accumulate-on-disk; OR move to A10G 22GB; OR use lower-res RAFT (500K params instead of 5M).
- **Council voice:** Hotz + Karpathy ("let compute speak — RAFT optical flow IS the proper physics-grounded pose extractor when PoseNet weights aren't available"). 
- **Cost:** ~$1 Vast.ai 4090 (3h chunked).
- **Predicted post-fix:** [0.80, 1.05] [prediction]. RAFT-derived poses are physics-grounded and may beat learned-pose approaches.
- **EV/$:** (1.05 − 0.92) / 1 = **0.13** ppt/$. **TIER 1.**

---

## TIER 2 — re-engineer when budget permits

### #6 Lane S (full-arch Self-Compression)

- **Original outcome:** Phase 1 init succeeded; OOM at Phase 2 GT scorer cache (2359MB on T4).
- **Bug:** GT scorer cache pre-computation doesn't release intermediate activations.
- **Fix:** stream the cache to disk + load on demand, OR chunk cache computation, OR move to A10G/A100.
- **Council voice:** Selfcomp (this is the Self-Compression paradigm he ships at 0.38).
- **Cost:** ~$2 Vast.ai 4090 (8h).
- **Predicted post-fix:** [0.85, 1.10] per `project_lane_s_self_compress_engineering`. Central 0.95.
- **EV/$:** (1.05 − 0.95) / 2 = **0.05** ppt/$. **TIER 2.**

### #7 Lane W (hard-pair-weighted Self-Compression)

- **Original outcome:** v1 crashed (resume-from .bin instead of .pt); v2 hung 8h.
- **Bug 1:** dispatch passes quantized renderer.bin to `--resume-from` which expects .pt.
- **Bug 2:** Stage 2 train hung at "training" — possibly infinite loop or deadlock in checkpoint loader.
- **Fix:** point `--resume-from` to `experiments/results/lane_g_v3_landed/training_state_*.pt`; add per-step heartbeat to detect hang.
- **Council voice:** Yousfi (per-pair sensitivity profile is the proper Lane S extension); Quantizr (pair-difficulty awareness).
- **Cost:** ~$2 Vast.ai 4090 (8h).
- **Predicted post-fix:** [0.85, 1.10] per `project_lane_w_hard_pair_self_compress_premise_20260427`. Central 0.95.
- **EV/$:** 0.05 ppt/$. **TIER 2.**

### #8 Lane I (Cool-Chic CCh1 replacement)

- **Original outcome:** trained 999/1000 successfully (best FP4 scorer 2.7196). Stage 3 export crashed: parametrize-strip mismatch.
- **Bug:** `parametrize.weight.original` vs raw weight key mismatch — same class as the SHIRAZ R23-26 chain that was fixed for SHIRAZ but not propagated to Cool-Chic loader.
- **Fix:** apply `_strip_parametrize_hooks_from_state_dict` to CCh1 load path (mirror commit 212bcaaf).
- **Council voice:** Carmack (engineering shortcut — this is a 5-line fix that recovers 14h of training).
- **Cost:** ~$0.30 Vast.ai 4090 (1h re-export only — training is checkpointed).
- **Predicted post-fix:** unknown until score lands. Best FP4 scorer 2.7196 is post-quant proxy; auth could be 0.9-2.0.
- **EV/$:** **TIER 1** if score < 1.05 (extremely cheap re-export); **TIER 2** otherwise.

### #9 Lane Omega-Hessian (per-weight bit allocation)

- **Original outcome:** crash at `renderer.py:471 torch.linspace(-1, 1, H, ...)` device-side assert.
- **Bug:** OOB index from `pair_idx`, likely.
- **Fix:** repro with `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1` to find OOB site; bound-check pair_idx in renderer.forward.
- **Council voice:** Selfcomp (per-weight is the natural extension of his per-channel SC paradigm). Per memory `project_lane_omega_bit_budget_hessian_aware_quantization`: at B=600,000 bits (75KB renderer.bin), predicted [0.70, 1.05].
- **Cost:** ~$2 Vast.ai 4090 (8h).
- **Predicted post-fix:** [0.70, 1.05] [prediction]. Central 0.88.
- **EV/$:** (1.05 − 0.88) / 2 = **0.085** ppt/$. **TIER 2.**

### #10 Lane M-V3-clean (radial-zoom rank-1 with frozen pose pad)

- **Original outcome:** Lane M-V2 1.84 was a CONFIG_BUG (train/inference pose-pad mismatch).
- **Fix:** thread `frozen_baseline_padded_dims_1_5` through both train and inflate paths consistently.
- **Council voice:** Yousfi (rank-1 Jacobian discovery is sound for PoseNet — the renderer-input-space mismatch is fixable).
- **Cost:** $0.30 Modal T4 (2h).
- **Predicted post-fix:** [1.05, 1.20] [prediction] per `project_lane_m_v2_audit_council_findings_20260428`.
- **EV/$:** marginal improvement only; **TIER 2.**

---

## TIER 3 — UNIWARD v9 rebuild (lowest priority, high-uncertainty)

### #11 Lane UNIWARD v9 (proper rebuild)

- **Original outcome:** v1-v6 all crashed; v7 mask-resolution bug; v8 NO-OP cp.
- **Bugs:** Daubechies-8 kernels are wrong (2-tap stencils ~12% energy capture vs canonical ~91%); Stage 4 cp discards Stage 3 payload; missing SLI1 inflate decoder.
- **Fix scope (Council B):**
  1. Build SLI1 inflate decoder so encoded bytes actually go into archive.
  2. Replace 2-tap stencils with canonical Daubechies-8 wavelets (`src/tac/uniward_texture.py`).
  3. Preserve class boundaries (current impl can corrupt class transitions).
  4. Retrain renderer on UNIWARD-weighted loss (not bolt-on at archive build).
- **Council voice:** Fridrich (UNIWARD is his algorithm). Per `project_lane_uniward_v8_NO_OP_finding_20260429`: predicted [0.96, 1.04] conservative; [0.78, 0.84] moonshot if Selfcomp-eligible renderer.bin lands.
- **Cost:** ~$5-10 (full rebuild + train).
- **Predicted post-fix:** [0.78, 1.04] [prediction]. Central 0.90.
- **EV/$:** (1.05 − 0.90) / 7 = **0.021** ppt/$. **TIER 3** (high cost, lots of engineering work).

### #12 Lane STC clean-source (CUDA confirm only)

- **Original outcome:** Local MPS-encoder produced 21MB; FALSIFICATION WITHDRAWN per CLAUDE.md MPS rule.
- **Fix scope:** re-run STC encoder on Modal T4 CUDA (~$0.20, ~10 min) on the FIXED `clean-source` pipeline. **No new code; pure measurement.**
- **Verdict:** if CUDA-21MB ≈ MPS-21MB, then STC structurally fails and verdict CONFIRMS. If CUDA produces small bytes (<500KB), then approach is alive.
- **Cost:** $0.20 Modal T4.
- **EV/$:** binary outcome — either kills the lane forever ($0.20 to kill) OR opens a new contender. **TIER 1 by information value, TIER 3 by score-improvement EV.**

---

## TIER 4 — UNIWARD v1-v6 individual rebuilds (NOT recommended)

These six failures (sys/json missing imports, dead-flag mode kwarg, 2-D bool cast) are all variants of "the dispatch script was never tested locally." The proper fix is Lane UNIWARD v9 (above), not 6 individual rebuilds. **Do not pursue v1-v6 standalone.**

---

## Bundled re-engineering session — Council recommendation

**Most efficient single session: TIER 1 lanes 1-5, parallel dispatch.**

| Lane | Vehicle | Cost | Wall | Predicted band | Council voice |
|------|---------|------|------|----------------|---------------|
| Q-FAITHFUL fix | Vast.ai 4090 | $1.50 | 5h | [0.40, 0.80] | Quantizr |
| Lane V/V-V2 → H-V3 rebuild | Vast.ai 4090 | $2.00 | 8h | [0.55, 0.95] | Hotz + Quantizr |
| SegMap clones (in-flight) | Vast.ai 4090 | $0 (paid) | 12h | [0.30, 0.55] | Selfcomp |
| MAE-V import fix | Vast.ai 4090 | $1.50 | 5h | [0.85, 1.10] | Hinton |
| Lane FL chunked | Vast.ai 4090 | $1.00 | 3h | [0.80, 1.05] | Karpathy |
| **Total** | | **$6.00** | parallel ~12h | — | — |

**Expected outcome:** at least 1 of these 5 should land below 0.80 [contest-CUDA], breaking the Lane G v3 1.05 frontier by ≥0.25 points. **Highest aggregate EV in the session.**

If the user authorizes this $6 bundle, the full plan is ready to dispatch. Otherwise, propose individual lanes per priority order above.

---

## Long-game implication

The session-velocity multiplier from this audit: ~$6 spent re-engineering ~5 hidden gems, vs ~$50 spent dispatching net-new lane ideas with similar predicted bands. **5-10× capital efficiency** by working the existing portfolio before exploring further.

**This conclusion is council-aligned across all 10 inner voices.** Shannon (information theory) says the bytes are already invested in these designs and the entropy of "well-engineered hidden gem" is lower than "unengineered new idea". Dykstra (convex feasibility) says the achievable region is ALREADY explored — alternating projections back onto these lanes recovers feasibility. Hotz (engineering instinct) says fix-the-bug is always more efficient than design-around-it. Quantizr (adversarial) says the leader's 0.33 came from working ONE lane to completion, not many lanes to half-completion.
