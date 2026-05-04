# Renderer Block-FP / PR75 QP1 Sub-0.314 Readiness - 2026-05-03 Worker

Scope: local readiness only. No remote GPU, Lightning, Modal, Vast.ai, or
other dispatch was performed. Outputs are under
`experiments/results/renderer_blockfp_pr75_qp1_sub314_20260503_worker/`.

## Target

Current A++ frontier used here:

- candidate: `c088_c067_pr75_actions_top40_p3_t4`
- score: `0.3155226919767294`
- bytes: `276386`
- SHA-256:
  `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`

At `25 / 37545489 = 6.658589531221714e-7` score per archive byte, a
component-neutral sub-`0.314` route needs at least `2287` bytes saved versus
C088, or archive bytes at or below `274099`.

## Readiness Matrix

Primary artifact:

`experiments/results/renderer_blockfp_pr75_qp1_sub314_20260503_worker/readiness_matrix.json`

It records existing candidate archives, bytes, SHA-256, pose-safety status,
exact blockers, byte deltas, sub-0.314 feasibility, and next command templates.

## Local Preflight Run

Ran the cheap deterministic local pose-safety gate on the best byte-only
Q-FAITHFUL QZS3/QP1 archive:

```text
.venv/bin/python experiments/preflight_renderer_transplant_pose_safety.py --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip --candidate-archive experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/postprocess_fixed_snapshot_20260501T2146Z_fix1/qzs3_rp2_qp1/archive.zip --output-json experiments/results/renderer_blockfp_pr75_qp1_sub314_20260503_worker/qfaithful_2146_qzs3_rp2_qp1_pose_safety_preflight.json --max-pairs 5
```

Result: exit `2`, fail closed.

- candidate bytes/SHA: `272986`,
  `d90a937da2127086f28b66f7df58a027c8c565488eb8e765e468808361602128`
- formula-only score if C088 components were unchanged: `0.313258771536114`
- `safe_for_exact_eval_dispatch=false`
- fail-closed reasons: `mask_payload_changed`, `pose_payload_changed`,
  `render_output_parity_unsafe`
- aggregate source-vs-candidate mean absolute output delta: `72.32969665527344`
- aggregate RMS output delta: `87.7930527843035`
- aggregate max absolute output delta: `254.98912048339844`

Interpretation: the archive has enough bytes for a formula-only sub-0.314
path, but it is not a contest-faithful renderer transplant. It changes masks
and lossy QP1 poses relative to the C067 source, then renders far outside the
source output basin.

## Route Ranking

1. **Active fixed-renderer burn output gates.**
   This is the top route, but only after the already-active
   `c067_fixed_renderer_burn_qfaithful_fix2_h100p5` or
   `c067_fixed_renderer_burn_qfaithful_fix2_a100p4d` claims produce visible
   artifacts. No new dispatch should be launched from this worker. Required
   gates: harvest logs and snapshot archive, reject source-identical or
   unsafe exports, run trained-renderer transplant preflight, run pose-safety,
   compress with QBF1/Block-FP only after raw export parity passes, and rerun
   pose-safety on the compressed archive.

2. **Existing Q-FAITHFUL QZS3/QP1 byte artifacts.**
   These have the bytes: `qzs3_rp2_qp1` is `-3400` bytes versus C088 and
   `qzs3_pr64_qp1` is `-3283` bytes versus C088. They are blocked by local
   safety and transplant-contract mismatch. The best one failed this pass.

3. **Existing trained QBF1 / Block-FP transplant artifacts.**
   The best archive, `trained_qbf1_b0512`, is `283432` bytes, `+7046` bytes
   versus C088, and exact-negative with score `17.72267562501643`,
   avg SegNet `0.0026408`, avg PoseNet `29.82484055`, `n_samples=600`.
   It also fails local pose-safety as `render_output_parity_unsafe`.

## Main-Orchestrator Next Action

Do not dispatch more remote GPU work for this lane now. Harvest the existing
H100 p5 fixed-renderer burn first when artifacts become visible. The first
local command after harvest should be:

```text
.venv/bin/python experiments/preflight_trained_renderer_transplant.py --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip --renderer-export <harvested_renderer_export.bin> --output-dir experiments/results/renderer_blockfp_pr75_qp1_sub314_20260503_worker/<burn_candidate>_raw_preflight --block-sizes 256,512,1024
```

Then run:

```text
.venv/bin/python experiments/preflight_renderer_transplant_pose_safety.py --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip --candidate-archive <harvested_or_preflight_candidate_archive.zip> --output-json experiments/results/renderer_blockfp_pr75_qp1_sub314_20260503_worker/<burn_candidate>_pose_safety_preflight.json --max-pairs 5
```

Only a burn-derived archive that passes both raw-export parity and compressed
archive parity should be handed back for exact CUDA auth eval consideration.
