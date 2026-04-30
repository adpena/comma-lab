# CUDA Sensitivity/Fisher Authority - 2026-04-30 - Worker 1

This is a readiness note, not a score ledger. No CUDA eval or promotable
component-sensitivity artifact was produced locally.

## Local Finding

`experiments/profile_component_sensitivity.py` had a direct-script import
defect: `python experiments/profile_component_sensitivity.py --help` could not
import `experiments.convert_fisher_to_owv3_sensitivity_map` unless the caller
pre-set `PYTHONPATH` to include the repo root. The profiler now inserts the repo
root before local imports, and a regression test covers direct CLI help from the
repo root.

Local host is non-authoritative for CUDA claims:

```text
torch=2.11.0
cuda_available=False
cuda_device_count=0
```

## Artifact Inputs Verified Locally

- PFP16 A++ frontier archive:
  `experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip`
  - bytes: `686635`
  - sha256: `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
  - exact T4 eval:
    `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`
  - score recomputed from components: `1.043987524793892`
  - SegNet/PoseNet: `0.00400656` / `0.00346442`
  - `eval_provenance.json`: `device=cuda`, `gpu_model=Tesla T4`,
    `gpu_t4_match=True`
- Lane G v3 OWV3 anchor archive:
  `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`
  - bytes: `694074`
  - sha256: `9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b`
  - local CUDA score-grade eval:
    `experiments/results/lane_g_v3_landed/contest_auth_eval.json`
  - score recomputed from components: `1.048866504774971`
  - SegNet/PoseNet: `0.00400846` / `0.00345458`
- Lane G v3 renderer/masks/poses:
  - `iter_0/renderer.bin`: bytes `296776`, sha256
    `08f12d722dd33f9061deee72f49d782035597f78cd65ed1463a241ab430a7529`
  - `iter_0/masks.mkv`: bytes `421483`, sha256
    `d3eeb82ce28b988476a920265751cca3d9fa2ca1364de4f33a1c7e970b7895e9`
  - `iter_0/optimized_poses.pt`: bytes `15620`, sha256
    `cb8517f7a7e3c9382e952ff278dc3f8de44ba066db07746f16354c1dbe2cbca4`
- Ground-truth/scorer/runtime inputs:
  - `upstream/videos/0.mkv`: bytes `37545489`, sha256
    `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
  - `upstream/models/posenet.safetensors`: sha256
    `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`
  - `upstream/models/segnet.safetensors`: sha256
    `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`
  - `submissions/robust_current/inflate.sh`: sha256
    `6c1cb5f8d9a2c9ecfb972aa9937713ba0e5f58e2b9edd834d1ea64853af33d36`
  - `upstream/evaluate.py`: sha256
    `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`

## Exact CUDA Producer Command

Current `profile_component_sensitivity.py` is a CUDA Fisher-proxy producer with
official component response-curve diagnostics. It intentionally emits
`promotion_eligible=false` and blocks `--manifest-output`; do not label its
outputs as `component_sensitivity_v1`.

Run on a CUDA-visible, supply-chain-clean T4/equivalent host:

```bash
RUN_ID=component_sensitivity_worker1_20260430_r1
EVID=experiments/results/component_sensitivity_worker1_20260430_r1

.venv/bin/python scripts/scan_lightning_supply_chain.py --strict \
  --json-out ".omx/state/lightning_supply_chain_scan_${RUN_ID}.json"

nvidia-smi
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.__version__)
print(torch.cuda.get_device_name(0))
PY

.venv/bin/python experiments/profile_component_sensitivity.py \
  --checkpoint experiments/results/lane_g_v3_landed/iter_0/renderer.bin \
  --video upstream/videos/0.mkv \
  --masks-mkv experiments/results/lane_g_v3_landed/iter_0/masks.mkv \
  --poses experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt \
  --upstream upstream \
  --output-dir "$EVID/profile" \
  --all-pairs \
  --pair-batch 2 \
  --response-top-k 32 \
  --response-epsilons=-0.002,-0.001,-0.0005,0,0.0005,0.001,0.002 \
  --split-seed 20260430 \
  --holdout-fraction 0.2 \
  --aggregate sum \
  --device cuda
```

Expected diagnostic outputs:

- `posenet_sensitivity_map.pt`
- `segnet_sensitivity_map.pt`
- `combined_sensitivity_map.pt`
- `posenet_response_curve.json`
- `segnet_response_curve.json`
- `combined_response_curve.json`
- `perturbation_basis_v1.json`
- `sample_plan.json`
- `stability.json`
- `component_sensitivity_profile_summary.json`

## Promotion Gates Next

Before OWV3 exact eval can be promotion-grade:

1. CUDA host preflight must pass supply-chain scan, CUDA availability, and
   NVDEC/DALI exact-eval readiness.
2. Sensitivity producer must use all 600 pairs for promotion-intended maps.
   Pair-weight/top-k modes remain diagnostic unless separately reviewed.
3. Current profiler outputs remain Fisher-proxy artifacts. A promotable
   `component_sensitivity_v1` requires either official finite-difference
   component maps or a reviewed proxy-to-official response proof strong enough
   to remove `FISHER_PROXY_PROMOTION_BLOCKER`.
4. Manifest assembly must remain through
   `experiments/build_component_sensitivity_manifest.py`, with exact CUDA
   `contest_auth_eval.json`, archive custody, stability JSON, response curves,
   and sample plan from the same run.
5. OWV3 build must use `--fallback-action keep_asym`, conversion/build
   `missing-policy error`, and default byte comparator:
   `PFP16_A++`, bytes `686635`, sha256
   `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
6. Do not run OWV3 exact eval for promotion if the deterministic archive is
   larger than the PFP16 A++ frontier unless an exact distortion-reduction
   justification and review tag are present.

## Local Verification

Passed:

```bash
.venv/bin/python experiments/profile_component_sensitivity.py --help
.venv/bin/python -m py_compile experiments/profile_component_sensitivity.py \
  src/tac/tests/test_profile_component_sensitivity.py
.venv/bin/python -m pytest src/tac/tests/test_profile_component_sensitivity.py -q
.venv/bin/python -m pytest \
  src/tac/tests/test_profile_component_sensitivity.py \
  src/tac/tests/test_build_component_sensitivity_manifest.py \
  src/tac/tests/test_component_sensitivity_artifact.py \
  src/tac/tests/test_owv3_sensitivity_conversion.py \
  src/tac/tests/test_remote_lane_g_v3_owv3_fisher_stack_script.py \
  -q
bash -n scripts/remote_lane_g_v3_owv3_fisher_stack.sh
git diff --check
```

Dry fail-closed checks passed:

- CPU without `--allow-diagnostic-cpu` exits diagnostic-only.
- CUDA exits cleanly on the local non-CUDA host.
- `--manifest-output` is rejected before profiling because current outputs are
  diagnostic Fisher-proxy artifacts.
