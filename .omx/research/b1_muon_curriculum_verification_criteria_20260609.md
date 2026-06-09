# B1 launch-manifest VERIFICATION CRITERIA — PR95 8-stage curriculum + Muon stage-8-only

UTC: 2026-06-09 · claude · the criteria the orchestrator verifies `b1_launch_manifest.json`
against BEFORE the 64h run burns hours (operator-endorsed human gate). Source: operator
relay 2026-06-09 ("pr95 is not straight up muon" x2, with the full 8-stage recipe + the
broader Muon param-group exclusion). PR95 = baseline-to-beat + rigor/synergy reference, NOT
a mold; Muon is the FINAL conditioning stage, not the global optimizer.

## CRITERION 1 — the 8-stage curriculum (Muon ONLY in stage 8)
PR95 public recipe (verify each stage config against `profile_pr95_hnerv_muon_intake.py`
ground-truth read from the real PR95 source, NOT hallucinated):
| stage | name | optimizer | muon_active |
|---|---|---|---|
| 1 | ce_seg (cross-entropy Seg) | adamw/adam | false |
| 2 | tau_softplus_margin | adamw/adam | false |
| 3 | smooth_disagreement | adamw/adam | false |
| 4 | qat | adamw/adam | false |
| 5 | hard_pixel_L7 + C1a | adamw/adam | false |
| 6 | lambda_sweep | adamw/adam | false |
| 7 | sigma_sweep | adamw/adam | false |
| 8 | muon_continuation | muon (matrix decoder weights only) | true |

`BAD: optimizer=Muon globally from epoch 1.` `GOOD: stages 1-7 staged scorer/QAT/rate
curriculum (non-Muon); stage 8 = Muon continuation on matrix-like weights only.`

## CRITERION 2 — Muon param-group partition (the broader exclusion; the real correctness fix)
Muon applies ONLY to matrix-like DECODER weight tensors:
  is_muon = (ndim >= 2) AND ("stem" not in name.lower()) AND (not name.lower().startswith("rgb"))
            AND (".rgb_" not in name.lower())  [profiler rule, applied to DECODER weights]
EXCLUDED from Muon (→ AdamW), even if ndim>=2:
  - the per-pair LATENTS (28-d codes; 2D ⇒ the naive rule would wrongly Muon them — the trap)
  - biases, norms (LayerNorm/GroupNorm/BN), scalar schedules
  - entropy / rate parameters (C1a, prior params)
  - QAT quantization parameters (scales/zero-points/step sizes)
  - sidecar / action / waterfill parameters
  - the stem + the rgb output head
Expected split (decoder only): ~177,156 Muon / ~51,802 AdamW (plus latents+entropy+QAT all AdamW).

## CRITERION 3 — manifest must encode optimizer_schedule explicitly
```json
{"optimizer_schedule": [
  {"stage": 1, "name": "ce_seg", "optimizer": "adamw_or_adam", "muon_active": false},
  ... stages 2-7 muon_active=false ...
  {"stage": 8, "name": "muon_continuation", "optimizer": "muon_for_matrix_weights_only", "muon_active": true}
]}
```
+ muon_param_groups (allowed) + adamw_param_groups (latents/biases/norms/scalars/entropy/QAT/sidecar).

## CRITERION 4 — telemetry must prove the geometry per epoch
stage_id, stage_name, optimizer_kind_by_param_group, muon_active, muon_param_groups,
adamw_param_groups, latents_optimizer, bias_norm_scalar_optimizer, weight_decay,
grad_norm_by_group, proxy_d_seg/d_pose/rate/score, sidecar_exported(=false), pay_rent_gate_active.

## KILL/RESTART conditions (the orchestrator acts on the first telemetry rows)
- `muon_active == true` in ANY of stages 1-7 → KILL/RESTART (PR95-violating black-box Muon train).
- `sidecar_exported == true` without `pays_rent == true` → KILL/RESTART.
- latents/entropy/QAT params in the Muon group → PATCH the stage-8 partition (safe: stage 8 is
  last; stages 1-7 give ~40h to patch before Muon activates; run is resumable).
- no telemetry / no heartbeat → KILL/PATCH.

## Why verify-and-patch is SAFE here (no 3rd relaunch needed)
Muon activates ONLY in stage 8 (the LAST stage, ~5000 ep after ~24650 ep of stages 1-7).
The run is resumable. So even if the agent's first manifest has the narrow exclusion, the
orchestrator verifies the manifest + the first telemetry rows, and patches the stage-8 Muon
param-group partition any time during stages 1-7 — zero hours lost. The URGENT check is
muon_active in stages 1-7 (must be false); that is verified immediately on the first rows.

## Burning question this gate answers
If PR95's gain came from an 8-stage scorer/rate curriculum ENDING in Muon (not Muon alone),
does B1's manifest preserve that optimizer geometry — non-Muon formation stages 1-7, QAT/rate
shaping, final matrix-weight-only Muon continuation — and do the first telemetry rows prove
muon_active=false in stages 1-7 rather than a black-box Muon train?
