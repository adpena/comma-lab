# Lane Re-Dispatch Plan — Post Round 6 (9 invalidated SegMapTrainer lanes)

**Generated**: 2026-04-29 PM
**Trigger**: Round 6 council (`council_round6_adversarial_20260429.md`) confirmed Council A undercounted by 5 lanes. The .round() zero-gradient bug at `src/tac/segmap_renderer.py:281` (fixed via `Uint8STE.apply()`) silently severed gradient flow for every lane that uses `SegMapTrainer.train_epoch`.

## The 9 affected lanes

| Lane | Variant | Predicted band [contest-CUDA] | Status pre-fix |
|---|---|---|---|
| SC++ | kl_distill | [0.30, 0.55] | OOM crash 140s on A10G |
| SA-v2 | kl_distill | [0.40, 0.65] | OOM crash 110-130s |
| SO | hessian_quant | [0.30, 0.55] | OOM crash 130s |
| WC-S | kl_distill + curator outlier weighting | [0.78, 1.05] | trained but params frozen (zero-grad) |
| PA | kl_distill + PixelArt SegMap variant | [0.85, 1.05] | trained but params frozen |
| HM-S | kl_distill + 8-DOF homography embedding | [0.32, 0.45] | trained but params frozen |
| FR-Ω | hessian_quant + Fridrich-cost block-FP | [0.27, 0.45] | trained but params frozen |
| FC | kl_distill + FiLM-Canvas SegMap | [0.85, 1.10] | trained but params frozen |
| q_faithful's SegMap variant | varies | [0.40, 0.80] | partially trained, gap unclear |

**NOTE**: Lane MM v2 is NOT in this list — it's a BUILD-only path (re-encodes Lane A masks with grayscale-LUT, no training). Its 2.63 FALSIFICATION verdict STANDS. Lane G v3 = 1.05 also unaffected (uses `train_distill.py` with correct manual STE pattern).

## Re-dispatch prerequisites (ALL must land before any GPU spend)

1. **Council C bf16+scorer-chunk fix** in `src/tac/segmap_renderer.py` (subagent #256 in flight) — without this, ANY 4090 dispatch will OOM at the same 21GB FastViT attention map.
2. **Council D EMA wire-ins** to `train_szabolcs.py` + 3 QAT scripts (subagent #257 in flight) — without this, even fixed gradient flow leaves us underdamped vs Quantizr's proven 0.997 EMA paradigm.
3. **Council A .round() fix** at `segmap_renderer.py:281` — LANDED (commit upcoming when subagents stop racing).
4. **Council B Check 89 STRICT** for UNIWARD-style encode-then-discard (currently warn-only with 14 hits — sweep needed before STRICT).
5. **Council Round 6 defect fixes** — scorer.py Uint8STE + test grad-presence (LANDED in working tree, awaiting commit).

## Per-lane dispatch wave (Vast.ai 4090, $0.26/hr, 24GB DEDICATED)

Each lane uses Council C's prescribed config: `--bf16 --scorer-chunk 2 --batch-size 4` (B*N=8 ≤ 8 cap). Predicted runtime per lane: 1.5-2h on 4090. Total wave cost: ~$5-8.

Dispatch order (cheapest-validation-first, per Council E adversarial-rigor philosophy):

```bash
# Wave 1 — proven Selfcomp-clone with KL distill (lowest unknowns)
.venv/bin/python deploy_vastai.py launch --gpu RTX_4090 \
    --label lane_sc_plus_plus_v5_post_round6 \
    --script scripts/remote_lane_sc_plus_plus_kl_distill.sh \
    --predicted-band 0.30 0.55 --max-cost 3.50 --max-hours 12

# Wave 2 — fast validations after SC++ shows training is actually happening
.venv/bin/python deploy_vastai.py launch --gpu RTX_4090 \
    --label lane_sa_v5_post_round6 \
    --script scripts/remote_lane_sa_segmap_clone.sh \
    --predicted-band 0.40 0.65 --max-cost 3.50 --max-hours 12

# Wave 3 — variant lanes (only if Wave 1+2 confirm gradient flow is restored)
# WC-S, PA, HM-S, FR-Ω, FC — dispatch in parallel after Wave 1+2 verify
```

## Hard kill criteria per dispatch (Council A's Defect 3 verification)

After Wave 1's first 30 minutes, run remote SSH check:
```bash
ssh -p <port> root@<vast-ip> 'tail -20 /workspace/pact/lane_*results/sweep_*/train.log | grep -E "epoch=[0-9]+"'
```
Expect to see seg_dist + pose_dist values DECREASING (not constant at 158 / 2.37). If frozen → kill instance immediately, $0.50 wasted instead of $3.50.

## Post-dispatch validation gates

- contest-CUDA auth eval at end of each lane (already wired in dispatch scripts)
- harvest within 24h via `tools/harvest_modal_calls.py` (wait — Modal-specific; Vast.ai uses different harvest path: instance tarball download)
- compare actual score to predicted band; reset 3-clean-pass counter if any lane lands OUTSIDE predicted band

## Budget

- Total wave 1+2+3: ~$5-8 (5-8 lanes × 1.5-2h × $0.26/hr)
- Cap: $30 Vast.ai per Council E grand battleplan
- Fallback: Modal T4 if Vast.ai 4090 unavailable (would need Modal A10G config + Council C scorer-chunk to fit 22GB shared)

## Cross-refs

- `.omx/research/council_darts_s_freeze_audit_20260429.md` (Council A — root cause)
- `.omx/research/council_oom_class_deep_fix_20260429.md` (Council C — bf16+scorer-chunk fix)
- `.omx/research/council_round6_adversarial_20260429.md` (Round 6 — Lane MM v2 correction + 5-lane undercount)
- `feedback_round6_defects_lane_mm_correction_segmap_invalidation_extended_20260429.md` (memory)
- Memory `project_lane_g_v3_landed_1_05_20260428.md` (the proven baseline these lanes try to beat)
