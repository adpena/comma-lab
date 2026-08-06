# ddm_mx1 Row-1 launch ticket

This is a queued fire order for MAIN's Metal host at the et4 boundary. It is not a local launch receipt and not a contest score.

## Stage

Run PR130 semantic renderer stage 08 tail from the retained width-96, four-block, 4-bit QAT12k checkpoint:

- init: `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt`
- input labels: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt`
- target labels: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt`
- steps: `6000`
- lr: `2e-7`
- ce_fraction: `0.0`
- softplus_fraction: `-999.0` (fixed near tau `0.05`, matching PR130 tail shape)
- bits: `4`
- checkpoint/eval cadence: `250`

Dedicated machine-readable ticket:

| artifact | bytes | sha256 |
| --- | ---: | --- |
| `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_ticket.json` | 6375 | `b8eb99b8022de8b4e691d0eba07575397b64976a4c78149c967f35a1d2c1cf0a` |

## Fire Order: n32

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-train --device gpu --pairs 32 --steps 6000 --lr 2e-07 --ce-fraction 0.0 --softplus-fraction -999.0 --bits 4 --seed 20260806 --checkpoint-every 250 --eval-every 250 --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal --out /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal/result.json
```

Stratified n32 indices, seed `20260806`:

`[8, 28, 40, 65, 91, 97, 128, 149, 152, 188, 201, 209, 246, 252, 284, 285, 307, 334, 346, 379, 398, 406, 432, 437, 464, 491, 497, 514, 538, 561, 577, 591]`

## Fire Order: n120

Fire only after n32 produces a real Metal step-time and non-degenerate training telemetry.

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-train --device gpu --pairs 120 --steps 6000 --lr 2e-07 --ce-fraction 0.0 --softplus-fraction -999.0 --bits 4 --seed 20260807 --checkpoint-every 250 --eval-every 250 --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n120_metal --out /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n120_metal/result.json
```

Stratified n120 indices, seed `20260807`, are stored in `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_ticket.json` under `launch_ticket.n120_stratified_indices`.

## Resume

n32 resume command shape:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-train --device gpu --pairs 32 --steps 6000 --lr 2e-07 --ce-fraction 0.0 --softplus-fraction -999.0 --bits 4 --seed 20260806 --checkpoint-every 250 --eval-every 250 --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal --resume-from /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal/mlx.latest.npz --out /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal/result.json
```

The driver writes stage checkpoints as `mlx_stage_stepNNNNNN.npz` plus `mlx.latest.npz`.

## Wall Clock And Memory

Local torch CPU smoke measured `4.949445009231567` s/step for n=2, but this is not a Metal estimate. Local MLX step time is blocked by the no-device runtime, so MAIN must replace the estimate after the first n32 Metal measurements.

Known bytes:

| input | bytes |
| --- | ---: |
| retained QAT12k checkpoint | 283432 |
| tq1c label cache | 943720090 |
| GT label cache | 943720076 |

## Verdict Protocol

- Training telemetry axis: `[macOS-MLX research-signal]`.
- d_seg verdict for n32/n120: frozen CPU-torch SegNet through exact R/uint8 against OUR `gt_seg_cache.pt`.
- Compare against fp1 flat-paint floor `0.008305` and PR130 external d_seg `0.00029660`.
- No prefix banking. No n600 scorer work here; et4 owns the full scorer slot. No contest promotion without a byte-closed archive and `upstream/evaluate.py`.

## ROUND-1 ADVERSARIAL AMENDMENT (MAIN, 2026-08-06 — BINDING at fire)

**F1 (assumption-challenge axis finding): the token-source choice conflates two questions.**
The arm's torch smoke ran input=tq1c labels, target=GT — that measures ERROR-CORRECTION
capacity (can the renderer fix tq1c's own d_seg 0.0043 while rendering), NOT the receiver-
capacity question PR130's 2.9660e-4 [external] answers (their tokens ≈ their shipped
partition; renderer reproduces what tokens SAY). The Row-1 fire therefore runs TWO arms at
n32 (same budget each, stratified pairs shared):
  ARM-CAP  tokens=GT lstars,  target=GT   — mirrors their form; the receiver-capacity
           number comparable against fp1 0.008305 and their 2.9660e-4 [external].
  ARM-VEH  tokens=tq1c labels, target=GT  — the composed-vehicle question (tokens we can
           actually afford to ship + correction reach).
ARM-CAP is the EH1 Row-1 discriminator; ARM-VEH prices the real vehicle. n120 continues
whichever arm the n32 read makes decisive (both if divergent). Verify at fire time that
--input-cache/--target-cache route as declared (never assume the smoke default).

**Runnability note (axis 9):** first Metal s/step measured at the n32 rung start; abort
bar = projected wall-clock > 4× the CPU-derived estimate; memory projection re-checked at
the real batch before n120. Provenance: ticket args verified at source = their stage-08
TAIL (e2e.py:419-424: bits 4, steps 6000, lr 2e-7, eval-every 250, resumed from the 12k
QAT checkpoint) — calibration lineage CLOSED this round.
