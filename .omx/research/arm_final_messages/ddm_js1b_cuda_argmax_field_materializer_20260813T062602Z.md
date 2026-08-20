`READY_TO_FIRE` is landed in commit `592f965e4f`. No Modal job or scorer ran, so the frontier is unchanged.

The implementation retains approximately 22.8 GB of receiver/scorer payloads, is stage- and batch-resumable, and admits Stage 0 only when downloaded bytes independently reproduce CP135 `34,964` and C1 `17,926` flips. Full commands and custody details are in the [JS1B receipt](/Users/adpena/Projects/pact/.omx/research/ddm_js1b_cuda_argmax_field_materializer_20260813.md:59).

Validation: Ruff clean, 11 focused tests passed, Python compilation passed, payload-discard audit clean, and two review passes completed. Developer preflight had 17/25 gates green; the eight red gates were existing unrelated codebase findings, with no JS1B matches in scoped inspection.

Pinned dispatch:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/ddm_js1b_modal_cuda_argmax_field_materializer.py::main \
  --run-id ddm_js1b_20260813 \
  --resume-from ddm_js1b_20260813 \
  --lane-id ddm_js1b_cuda_argmax_field_materializer \
  --instance-job-id modal:ddm_js1b_20260813 \
  --claim-agent main:ddm_js1b \
  --output-dir /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813_dispatch \
  --detach --provider-detach-ack
```

Pinned post-step after harvesting and downloading the receipt plus four fields:

```bash
.venv/bin/python experiments/ddm_js1_stage0_per_edge.py summarize \
  --output /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/stage0_from_js1b \
  --from-argmax-fields /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813
```

Projected time is 1,350.215 seconds including a 300-second reserve, leaving 449.785 seconds below the 1,800-second limit. This remains derived, not JS1B-measured.

Effective frontier: CP135 `S=0.16195513827824176 @ 186,252 B`. Own-vehicle frontier: LC2 `S=0.16959899569230852 @ 187,226 B`. Neither moved.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN/js1 scorer-lane owner. **Consumer store:** `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/`. **Fire trigger:** all other Modal/scorer claims are terminal, release preflight findings are adjudicated, and single-flight is clear; dispatch JS1B, harvest/download its fields, then run the pinned post-step. Fire V0–V5 only if the receipt is `ADMITTED`.

## LIVE-HYPOTHESES

- The T4 receiver should restore the CP135 control because the prior local and promoted raw-video hashes already differ before scoring.
- Road-incident interfaces may remain the dominant CUDA allocation hub because independent full-population evidence repeatedly places most flips there.
- T1R1 may recover substantially more CUDA debt than its withheld 7.53% local diagnostic; renderer-device drift is large enough that the local ranking is not transferable.

## DEAD-ENDS

- The macOS-CPU fields are closed as Stage-0 authority: CP135 measured `50,395` rather than `34,964` flips.
- Changing only the local scorer to batch 16 is closed: the complete local and promoted raw streams differ by hash.
- Substituting T1R1 output for the separate C1 control is closed; they are distinct objects.
- Hand-rolling the F26 parser is closed by the VD1b failure chain; JS1B executes each shipped `inflate.sh` unchanged.