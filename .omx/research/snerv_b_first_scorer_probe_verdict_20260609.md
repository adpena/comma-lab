# SNeRV-B first scorer probe — REFRAME: live surface already evaluator-close; the chasm is EXPORT (G1b) + RATE

UTC 2026-06-09 · claude · `[macOS-MLX research-signal]` / telemetry_proxy + per-step live frozen-scorer
terms → `mechanism_update_eligible` ONLY. NOT a score claim. Run: `snerv_mistake_b_g1a_20260609T201221Z`
(PID 42886, commit `f5c66f43c`+, path-B official conv MFU/HFR/TUB, skip=full,
`segnet_direct_live=7.24` / `pose_direct_live=7.0` / CE=3.78 / recon annealed 0.2 — the uncrossed
objective CONFIRMED ACTIVE in `loss_components.active_loss_weight__*`).

## The probe data (telemetry.jsonl, 400 epochs; every value is the live frozen-scorer batch surface)
| ep | live argmax d_seg (cand-vs-target) | cand/target argmax class-mean | pose score term | pose raw MSE |
|---|---|---|---|---|
| 0 | **0.0023** | 2.037 / 2.037 | 0.113 | 0.0013 |
| 100 | 0.0022 | 2.043 / 2.042 | 0.233 | 0.0054 |
| 200 | 0.0021 | 2.089 / 2.088 | 0.100 | 0.0010 |
| 300 | 0.0030 | 2.030 / 2.030 | 0.154 | 0.0024 |
| 399 | 0.0026 | 2.045 / 2.044 | 0.420 | 0.0176 |

Receiver-surface per-step movement ≈ 0 (0–3 argmax pixels flip per step; step-guard backtracking engaged
53/400 epochs to cap pose worsening).

## The three findings
1. **The live surface starts evaluator-CLOSE (d_seg≈0.0023 at ep0) and stays there.** Not a training win —
   a DESIGN property: SNeRV stores the LF wavelet payload from the SOURCE (store-LF/generate-HF), so the
   render begins as a real LF reconstruction. The class distribution is real (mean≈2.04, not collapsed).
   **The mean-field d_seg≈0.5 disease is specific to latent-SYNTHESIZING carriers; it never applied to
   SNeRV's live surface.**
2. **Training is not the lever here.** d_seg flat at ~0.0023 over 400ep; pose raw-MSE NOISILY DEGRADING
   (0.0013→0.0176; sqrt(10·d_pose) 0.113→0.42) with the guard repeatedly backtracking. The uncrossed
   scorer objective is live but has almost nothing to improve on this surface — and may be slightly
   hurting pose. (Mechanism note: with the LF stored, the residual learnable surface is small.)
3. **The 0.71 baseline was APPLES-TO-ORANGES** (surface-conflation bug class, the same class as
   PSNR≠d_seg): ep22399's `avg_segnet_dist=0.7115` was a DIFFERENT candidate (Haar score renderer) on the
   EXPORT/receiver-side surface; today's 0.0023 is the live in-memory render. The "0.71 → <0.2" SNeRV-B
   prediction was therefore against the wrong baseline — recorded honestly per Catalog #307. The
   audit-provenance rule (cite surface + candidate_id) would have prevented this; third such lapse today.

## The reframe (where SNeRV's score actually lives)
`S = 100·d_seg + sqrt(10·d_pose) + 25·rate`, and on the LIVE surface SNeRV already sits at
~0.26 + ~0.4 + 25·rate [telemetry_proxy units]. So the score-determining questions are DOWNSTREAM of
training:
- **G1b (export/receiver binding):** does the archive→inflate→evaluate render PRESERVE the live fidelity?
  The audits say the path-B MLX→official-payload export is BLOCKED (`carrier.py` export blockers). Until
  G1b lands, the exact-eval surface may bear NO relation to the live 0.0023 (the prior 0.71-class numbers
  suggest the receiver surface was far worse — but per finding 3 that needs an apples-to-apples re-measure
  on THIS candidate).
- **RATE:** the stored LF payload's bytes. Sister Z8 evidence: wavelet/LF storage was ~99.5% of archive and
  the rate killer. The competitive question is whether the LF payload compresses to ~100-200 KB while
  holding the live d_seg/d_pose — the LF/HF byte-pressure problem (task #29's original framing), NOT the
  training objective.

## Route (authority-disciplined)
- Let the 600ep run finish (cheap; guard bounds pose damage), but do NOT extend it — more epochs are not
  the lever (finding 2).
- **The decisive next milestone for SNeRV is G1b**: bind the trained path-B weights into the official
  decoder payload → byte-closed archive → measure (a) archive BYTES (the rate term) + (b) exact
  archive-surface d_seg/d_pose, apples-to-apples vs the live surface. That single measurement answers
  whether SNeRV is a real Vehicle-2 (live fidelity survives the receiver at acceptable bytes) or whether
  the receiver/rate chasm consumes it.
- Update the J_scorer·J_renderer doctrine with this case: SNeRV's product was NONZERO all along on the
  live surface; the failure class here is the FIFTH link of the operator's chain (effect ≠ AUTHORITY —
  the live effect exists; the archive-surface authority does not yet). Different link, different fix.
- The pose drift (finding 2) is a real mechanism finding for the objective: with LF stored, the seg axis
  is near-floor and the pose axis is the live optimization battleground — consistent with the
  operating-point marginal (pose dominates at low distortion per CLAUDE.md).

## Cross-refs
`snerv_fullstack_extreme_scrutiny_vs_evaluate_py_20260609.md` (G1a/G1b split; export blockers) ·
`feedback_*snerv_crosswire*` / commit `f5c66f43c` (the pose VJP uncrossing — confirmed live in-gradient) ·
`b1_f1_recon_ablation_verdict_*` (the sister surface-conflation lesson) · the operator's
name→mechanism→gradient→effect→authority chain (this is a link-5 failure, not link-3/4).
