---
name: Lane UNIWARD v8 harvested 1.14 [contest-CPU advisory] — needs CUDA confirm
description: 2026-04-29 PM. Modal harvest recovered Lane UNIWARD v8 with auth score 1.14 — competitive with Lane A's 1.15 baseline. Device was CPU (not CUDA), so per CLAUDE.md non-negotiable this is ADVISORY only. Needs Vast.ai 4090 CUDA re-eval to promote to [contest-CUDA] and become a strategic lane.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What happened

Lane UNIWARD-Texture was dispatched to Modal as `lane_uniward_v8` (Apr 29, 09:37:35Z, 779s runtime). The training script `scripts/remote_lane_uniward.sh` produced a `texture_probability.pt` (4MB) + assembled an archive (694KB) + ran contest_auth_eval.py with `--device cpu`.

The result sat in the FunctionCall return-value cache for hours, unharvested. Today I ran `tools/harvest_modal_calls.py` and recovered the artifacts.

## Recovered numbers

```json
{
  "final_score": 1.14,
  "avg_posenet_dist": 0.00449546,
  "avg_segnet_dist": 0.00460933,
  "rate_unscaled": 0.01851,
  "archive_size_bytes": 694045,
  "device": "cpu"
}
```

For reference:
- Lane A (Yousfi+Fridrich pose TTO baseline): score 1.15 [contest-CUDA] (memory: project_lane_g_v3_landed_1_05_20260428.md)
- Lane G v3 (KL distill weight=0.002 + pose TTO retry on Lane A anchor): score 1.05 [contest-CUDA]
- Lane UNIWARD predicted band: [1.05, 1.18] (per scripts/remote_lane_uniward.sh dispatch metadata)

**1.14 lands inside the predicted band.** UNIWARD texture probability paradigm (Fridrich inverse-steganalysis: "errors in textured regions are undetectable, weight loss by inverse local variance") appears to be working AS DESIGNED.

## CRITICAL CAVEAT — needs CUDA re-eval

Per CLAUDE.md "MPS auth eval is NOISE" + Check 83 STRICT (no MPS-derived strategic decision):
- Device was CPU, not CUDA. CPU-vs-CUDA drift on these scorers is documented as smaller than MPS-vs-CUDA (which is 23× on PoseNet) but is NON-ZERO.
- Until Vast.ai 4090 CUDA re-eval lands, this score is `[contest-CPU advisory]` only.
- DO NOT call this a Phase 1 GREEN until CUDA-confirmed.

## Re-eval plan

1. Harvested archive at: `experiments/results/lane_uniward_v8_modal/harvested_artifacts/lane_uniward_results/eval_work/archive.zip`
2. Verify SHA matches: the `submissions/robust_current/contest_eval_archive.json` should record the SHA the CPU eval ran on.
3. Dispatch a Vast.ai 4090 instance running `experiments/contest_auth_eval.py --archive <path> --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda`. Cost: ~$0.50, ~10 min.
4. If CUDA result within 0.05 of 1.14, promote to `[contest-CUDA]` and add Lane UNIWARD v8 to the kept lanes table.
5. If CUDA result drifts (>0.10 spread), investigate — UNIWARD may have a CUDA-vs-CPU codec divergence.

## Stack potential (per memory project_codec_stacking_composition_canonical_orders_20260429.md)

UNIWARD is a DISTORTION SHAPING technique (Fridrich inverse-steganalysis), not a codec. It changes WHERE the renderer puts errors (textured regions vs smooth regions), not HOW MANY bytes the archive is. So it can stack with:
- Lane PD-V2 (arithmetic-coded pose deltas) — orthogonal, no overlap
- Lane Ω-W-V2 (water-fill + arithmetic on block-FP weights) — orthogonal, applies to renderer.bin not poses
- Lane LCT (10-byte payload bolt-on) — orthogonal
- Lane STC (mask-class boundary coding) — partial overlap (UNIWARD weights LOSS, STC encodes BYTES; both interact with masks)

Expected stacked floor with UNIWARD anchor: ~1.10 → 0.85-0.95 [prediction] depending on stacking efficiency.

## Cross-refs

- Harvested archive: `experiments/results/lane_uniward_v8_modal/harvested_artifacts/lane_uniward_results/eval_work/archive.zip`
- Provenance: `experiments/results/lane_uniward_v8_modal/harvested_artifacts/lane_uniward_results/eval_work/provenance.json`
- Modal call_id: `fc-01KQCTS1BKB5GYNA2AKREFRYXY` (dispatched 2026-04-29T09:37:35Z, 779s runtime)
- Lane script: `scripts/remote_lane_uniward.sh`
- UNIWARD module: `src/tac/uniward_texture.py`
- Memory: feedback_modal_spawn_result_cache_pattern_20260429.md (the harvest-or-lose discovery)
