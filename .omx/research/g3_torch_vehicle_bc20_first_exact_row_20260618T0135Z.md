# G3 — First byte-closed dual CPU+CUDA EXACT-eval row for the torch_vehicle bc20 small-basis vehicle

**Lane:** `lane_g3_torch_vehicle_bc20_exact_eval_20260617`
**Task:** #127 (G3)
**Date:** 2026-06-18 UTC
**Operator authorization:** Modal <$5 dual CPU+CUDA dispatch for exactly this row.
**Authority:** `[contest-CUDA]` (Modal T4, 600 samples) + `[contest-CPU]` (Modal Linux x86_64, 600 samples).
**Pointer impact:** NONE. This is a CALIBRATION row (exact S ≫ the 0.19110 frontier pointer). The
`canonical_frontier_pointer.json` is NOT touched. The value is the measured advisory→exact GAP that
de-risks the small-basis campaign.

## The vehicle
* Byte-closed basin: `experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best/`
  (base_channels=20, latent_dim=28, 600 pairs; HNeRV-Muon decoder).
* `0.bin` payload = 89,136 B; contest `archive.zip` (ZIP_STORED member `0.bin`) = 89,244 B.
* archive.zip sha256 = `856e3bf076a5acf7260213468a650f51110218da4bb02674b7bff40b922ee931`
* runtime tree (CUDA upload) sha256 = `d9c8451acf2512ac0a5293bdd8832d59d721427cbe95694b57b52105305b3e9a`
* Packet built by `tools/build_torch_vehicle_g3_contest_packet.py` (commit 36773b024), reusing the
  canonical `tools/verify_e2e_byte_close_eval.py` loader + `_byte_close_and_verify_parity` (G2 fixed-point
  parity contract). NO codec reimplemented.

## $0 MVP-first gate (passed BEFORE any spend)
1. **Byte-close parity:** parse-back is a fixed point (weights + latents bit-exact on re-encode/re-parse);
   `parity_ok=true` (`build_deterministic`, `weights_fixed_point`, `latents_fixed_point`, `keys_match` all true).
2. **Runtime closure (the likely failure mode — caught free):** the assembled contest packet was run through
   the EXACT contest chain locally on CPU — `inflate.sh -> python inflate.py archive/0.bin inflated/0.raw` ->
   `saved 1200 frames` -> `0.raw` = exactly **3,662,409,600 B** = (1200, 874, 1164, 3) uint8, the shape
   `upstream/evaluate.py` requires. Proof: `.omx/research/g3_torch_vehicle_bc20_runtime_closure_proof.json`
   (raw sha256 `ace6e7b98e65a7b35b6fcbd7ef29e839cd3c5f0c664101120dcdb497b3821f89`). The 3.66 GB scratch was
   certified-rebuildable and deleted (disk hygiene).
   * Runtime is dependency-closed + portable: `inflate.py` = 66 LOC; deps = torch + torch.nn.functional +
     numpy + brotli only; CUDA-or-CPU agnostic (`torch.device('cuda' if available else 'cpu')`). The
     vendored PR95 `-m submissions.X.inflate` form was REPLACED by a clean direct-call `inflate.sh` so the
     packet runs from any extracted location (the public-PR intake clone is NOT edited in place).
3. **5 NO-FAKE tests** (`src/tac/torch_vehicle/tests/test_g3_contest_packet_builder.py`) incl. a subprocess
   inflate.sh runtime-closure test at unit scale — all green.

## The dual exact row (the deliverable)

| axis | call_id | d_seg (avg_segnet_dist) | d_pose (avg_posenet_dist) | rate (×25, 89244 B) | S (recomputed from components) | samples |
|---|---|---|---|---|---|---|
| **[contest-CPU]** (Linux x86_64) | `fc-01KVC5TXBKVGY4MBMH46WCJVQY` | 0.00260094 | 0.00034168 | 0.059424 | **0.37797132** | 600 |
| **[contest-CUDA]** (T4) | `fc-01KVC5T6G4ZWCRWF9T7F8MADB9` | 0.00262703 | 0.00048168 | 0.059424 | **0.39153009** | 600 |

S recompute (verified by hand — the rounded `final_score` field is NOT used):
* CPU : `100*0.00260094 + sqrt(10*0.00034168) + 25*89244/37545489 = 0.260094 + 0.058453 + 0.059424 = 0.37797132`
* CUDA: `100*0.00262703 + sqrt(10*0.00048168) + 25*89244/37545489 = 0.262703 + 0.069403 + 0.059424 = 0.39153009`

Both passed (rc=0), 600 samples, identical archive bytes, identical runtime tree (content-hash equal across
axes). Rate term is axis-invariant; the axes differ only in d_seg / d_pose.

## The two gaps (the de-risking signal — the real value of this row)

### Gap 1 — advisory (macOS in-Python) vs exact contest-CPU: ESSENTIALLY ZERO (the headline)

| component | advisory (macOS in-Python, basin best_meta) | exact [contest-CPU] | rel gap |
|---|---|---|---|
| d_seg  | 0.00260092 | 0.00260094 | **+0.0008%** |
| d_pose | 0.00034169 | 0.00034168 | **−0.0019%** |
| S      | 0.37797    | 0.37797    | ~0 |

**The macOS in-Python authority path IS the contest-CPU axis to within ~0.001% on the real 600-pair video.**
The local advisory we steer the small-basis campaign by is bit-equivalent (to 4 sig figs) to the exact
`upstream/evaluate.py --device cpu` on 1:1 contest Linux x86_64 hardware. This is the major de-risk: every
advisory d_seg / d_pose / S the campaign reports locally is a faithful contest-CPU number. No surprise on
the CPU axis when a converged basin byte-closes.

### Gap 2 — contest-CPU vs contest-CUDA: the per-archive axis drift to CARRY

| component | [contest-CPU] | [contest-CUDA] | rel gap |
|---|---|---|---|
| d_seg  | 0.00260094 | 0.00262703 | **+1.00%** |
| d_pose | 0.00034168 | 0.00048168 | **+40.97%** |
| S      | 0.37797    | 0.39153    | +0.01356 |

CUDA reads HIGHER on both terms (same direction as the PR102 family's CUDA−CPU > 0). d_seg drift is tight
(+1%); d_pose drift is large (+41%) — the canonical PoseNet CPU↔CUDA numeric divergence. At this operating
point the d_pose TERM is √(10·d_pose), so +41% on d_pose → +18.7% on the term (+0.011 S). The contest
leaderboard ranks by the CPU eval (S=0.378 here); the CUDA axis (S=0.392) is the promotion-truth axis.

**Verdict (what to expect when the converged margin-hinge / FiLM-v2 basin byte-closes):**
* Local advisory d_seg / d_pose / S ≈ exact contest-CPU (gap ~0.001%) — steer with full confidence.
* For the CUDA axis, BUDGET d_seg ×1.01 and d_pose ×1.41 over the local/CPU numbers; the d_pose×1.41
  matters because ∂S/∂d_pose is 86% of ∂S/∂d_seg at the small-basis floor point.
* The small-basis vehicle's exact eval is CONSISTENT with the advisory — no SegNet collapse, no shape
  mismatch, no rate blowup. The whole pipeline (byte-close → contest inflate.sh → frozen SegNet/PoseNet →
  score) is end-to-end verified at contest grade on BOTH axes.

## Mission honesty (the means/ends firewall)
This row does NOT lower the frontier (exact S 0.378 [CPU] / 0.392 [CUDA] ≫ pointer 0.19110). It is a
CALIBRATION milestone that de-risks the small-basis campaign: the exact dual-axis baseline + the measured
transfer functions (advisory↔contest-CPU ≈ 0; contest-CPU↔CUDA d_seg +1% / d_pose +41%). The next
score-moving unit is the converged margin-hinge / FiLM-v2 basin byte-closed + a fresh dual exact row;
sub-0.15 (CPU) requires d_seg ≈ 0.000322 (per the floor-point note), which is ~8× below this vehicle's
exact d_seg 0.00260 — the d_seg-axis attack remains the campaign's job, and we now KNOW the local advisory
d_seg it is measured by is the contest-CPU d_seg.

## Artifacts / custody
* Packet: `experiments/results/g3_torch_vehicle_bc20_packet_20260618T012713Z/submission_dir/`
* CUDA result: `experiments/results/modal_auth_eval/g3_torch_vehicle_bc20_paired_modal_auth_20260618T013243Z_cuda/`
  (`modal_cuda_auth_eval_result.json`, `report.txt`, `provenance.json`)
* CPU result: `experiments/results/modal_auth_eval_cpu/g3_torch_vehicle_bc20_paired_modal_auth_20260618T013243Z_cpu/`
* Pair group: `g3_torch_vehicle_bc20_20260618T013243Z`
* Runtime closure proof: `.omx/research/g3_torch_vehicle_bc20_runtime_closure_proof.json`
