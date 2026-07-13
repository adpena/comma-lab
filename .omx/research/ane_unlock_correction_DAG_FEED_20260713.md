# DAG FEED — ANE frozen-SegNet correction ladder — 2026-07-13

`research_only=true` · `score_claim=false` · `promotion_eligible=false` · `pointer_moved=false`  
Lane: `lane_ane_unlock_correction_20260713` · checkpoint: `ane_unlock`

## Terminal state

`NO_JOINT_BAR_CANDIDATE`. Full CoreML FLOAT32 on `CPU_AND_GPU` is the only measured held-out
fidelity pass (`28/2,359,296 = 1.1867947e-5`), but its matched forward speedup is `3.609338x`, below
the preregistered `10x` bar. No training, trainer edit, paid dispatch, archive mutation, score claim,
or pointer update is authorized by this FEED.

## Executable dependency graph

```text
real weights + fixed real n24 frames + canonical 1-thread CPU reference
  -> R0 FLOAT32 CPU_ONLY / CPU_AND_GPU vs Torch-fp32
      -> op-substitution floor + precision delta + margin/class/boundary localization
      -> R1 selective-fp32 {head, decoder-4, decoder-3:4}
      -> R2 calibration-only {channel affine, one-layer 3x3 logit residual}
      -> R3 calibration-only margin thresholds
           -> full CPU_AND_NE donor emits logits + 23 SE gates
           -> approximate 64x64 core / 32px halo / stride-32 CPU tiles
           -> atomic checkpoint after each of 24 frames
      -> R4 {W8A8 PTQ, FLOAT32 CPU_AND_GPU}
  -> R5 measured-only selection
      -> flip <= 3.3e-5 AND matched forward speedup >= 10x ?
           NO -> label-grade REFUSE; pointer remains unmoved
           YES -> still require n600, gradient/VJP, receiver, and contest-axis gates
```

R3 is outside the exact tile-halo kill because it injects full-frame donated SE gates and accepts a
finite-halo approximation. Its negative is scoped only to the measured `core=64, halo=32` full-network
tile formulation. It does not reopen or weaken the exact `halo=685 + 23 global reductions` result.

## Canonical decision gates

1. **Label gate:** held-out `argmax_flip_rate <= 3.3e-5`.
2. **Economics gate:** corrected end-to-end forward `>=10x` versus a matched one-thread CPU-Torch
   reference. Cross-receipt timing ratios are sensitivity only.
3. **Training-gradient gate:** real n600 global and minimum-pair input-cotangent cosine `>=0.99`, plus
   measured teacher wall improvement. Forward logits cannot satisfy this gate.
4. **Authority gate:** local CoreML is advisory only. Exact evaluator cells and archive bytes remain
   CPU/CUDA authority; no local row can promote a score.
5. **Placement gate:** a future ANE claim requires a clean E5RT placement receipt. This run observed an
   E5RT cache permission failure, so its dense-ladder timing cannot establish true ANE placement.

## Triality and six-hook disposition

- **Equation:** reuse `amdahl_measured_disjoint_wall_split_with_async_cpu_verdict_v2`; R5 supplies the
  measured forward component as one disjoint lever. Registering a duplicate Amdahl law is forbidden.
- **DSL:** N/A-with-reason. These are backend measurements; no legal trainer flag exists and the user
  explicitly assigned trainer wire-in to a later ticket.
- **DAG:** this isolated FEED is the execution/promotion boundary; no shared hot DAG file was edited.
- **Sensitivity map:** R0 emits class-pair, Torch-margin, and reference-boundary localization. Because
  this landing is `research_only=true`, it does not mutate the live sensitivity map; a future backend
  consumer must ingest `r0_decomposition.json` rather than remeasure it.
- **Pareto constraint:** flip and matched-wall bars are conjunctive. A fidelity-only or speed-only pass
  is inadmissible.
- **Bit allocator:** N/A; the training-time corrector ships nowhere and changes zero archive bytes.
- **Cathedral/autopilot:** no executable backend is admitted. Three reviewed-next-step rows are staged
  through `record_candidate` in `ane_unlock_correction_candidate_rows_20260713.jsonl`, not silently
  inserted into the shared pool.
- **Continual learning:** the memo, R0–R5 receipts, artifact manifest, and staged rows are the durable
  posterior signal. The settled R1/R2/R3/R4 formulations must not be rediscovered.
- **Probe disambiguator:** the ladder directly arbitrates distributed precision, output calibration,
  and approximate donated-global-state correction. All three interpretations were kept until measured.

## Staged successors

| Candidate | Status | Required evidence |
|---|---|---|
| `ane_full_float32_true_placement_remeasure_v1` | `reformulation-queue` | clean placement; n24 joint bar; n600 confirmation |
| `ane_training_input_cotangent_parity_gate_v1` | `needs-build` | n600 global/min-pair cotangent cosine, direction agreement, wall win |
| `ane_dense_encoder_sparse_decoder_donated_se_v2` | `reformulation-queue` | cut after global encoder/SE; held-out joint bar |

## Artifact edges

- Primary decision: `experiments/results/ane_unlock_correction_20260713/r5_composition.json`
- R0 decomposition/WHERE: `r0_decomposition.json`
- R1/R2/R3/R4: corresponding rung receipts in the same directory
- R3 resume surface: `r3_progress.json` plus 24 JSON/NPZ frame checkpoint pairs
- Custody: `artifact_manifest.json`
- Full narrative and verdict scopes: `.omx/research/ane_unlock_correction_20260713.md`
