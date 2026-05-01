# OWV3 Fisher Dispatch Runbook

Updated: 2026-04-30

## Current Blocker State

- Local machine is not a CUDA host: `torch.cuda.is_available() == False`, `torch.cuda.device_count() == 0`, and `nvidia-smi` is absent.
- Local MPS is available, but MPS/CPU Fisher maps are smoke-only and must not be used for promotion.
- Hardened Vast dispatch `lane_g_v3_owv3_fisher_20260430_codex_a1` was attempted on 2026-04-30 with two retry attempts. Both hosts reached `phase2-launch` and failed `NVDEC_BAD`; the launcher auto-destroyed both instances. No OWV3 Fisher artifact or promotion evidence was produced.
- Lightning SSH is working through
  `s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai`; the current Studio host
  reported Tesla T4 CUDA access. Use `scripts/lightning_repro_workspace.py`
  for reproducible source/artifact staging before exact eval or Batch Jobs.
- The main Lightning Studio tree is now source-manifested, but `--no-install`
  verification used system Python and recorded no torch package. Run locked
  runtime install or point `PYBIN` at a CUDA venv before exact eval.
- Modal CLI is installed in `.venv/bin/modal` and profile `adpena` is active, but the existing Modal lane wrapper forces auth eval to CPU. Use Modal only for `RUN_CONTEST_EVAL=0` Fisher/build artifacts, then run exact CUDA eval on a CUDA/NVDEC host.

## Latest CUDA Fisher Artifact

- Lightning run:
  `experiments/results/lane_g_v3_owv3_fisher_lightning_20260430_codex_r2/`.
- Fisher source: CUDA/T4, `FISHER_TOP_K=30`, `PAIR_BATCH=4`,
  `include_protected_conv2d=true`.
- Sensitivity map: `owv3_sensitivity_map.pt`, SHA-256
  `ed69bec3c9c530e4d574d82d3b6764399a6feca0289f2114899fa09689fabeba`,
  `19` layers, `717` channels, no missing Conv2d keys.
- Built archive: `archive_lane_g_v3_owv3.zip`, `689342` bytes, SHA-256
  `29a02b2af2c37371eec80ca3e278c4ce368703ba0a0a2121e2b32f570106a84c`.
- Byte verdict: archive is `+2707` bytes versus the PFP16 A++ frontier and is
  blocked from exact promotion eval until the byte plan is improved or an exact
  distortion-reduction justification is reviewed.
- Lightning Batch Jobs exact-eval queue has a dry-run record only for this
  archive; no exact score exists for the r2 OWV3 candidate.

## Promotion Byte Gate

- OWV3 promotion builds default to `--fallback-action keep_asym`. Protected
  channels and all-protected layers must preserve the compact ASYM-style
  representation instead of silently falling back to FP16.
- `--fallback-action diagnostic_fp16` is smoke/debug only. Artifacts built with
  it are tagged non-promotable in the OWV3 byte plan.
- The builder compares candidate archives against the active PFP16 A++
  frontier by default: `archive_bytes=686635`, SHA-256
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- A candidate larger than Lane G v3 or the PFP16 frontier exits nonzero unless
  an explicit smoke/debug override is passed. Do not run exact promotion eval
  on a byte-regressing archive without an exact distortion justification and
  review tag.
- Build provenance must include the OWV3 byte plan, deterministic ZIP rebuild
  status, member manifest, archive SHA-256, and frontier comparator result.

## Hardened Vast Launcher Dispatch

Use the retry launcher so failed NVDEC hosts are detected by `phase2-launch` and destroyed before they can produce advisory artifacts:

```bash
.venv/bin/python scripts/launch_lane_with_retry.py \
  --lane-script scripts/remote_lane_g_v3_owv3_fisher_stack.sh \
  --label lane_g_v3_owv3_fisher_YYYYMMDD_unique \
  --max-dph 0.40 \
  --predicted-band 0.95 1.25 \
  --estimated-cost 2.00 \
  --max-retries 3 \
  --retry-delay 20
```

After dispatch, verify state with:

```bash
.venv/bin/python scripts/verify_vast_instances.py
```

Only `contest_auth_eval.json` from the exact archive on a CUDA/NVDEC host can be used for promotion. `NVDEC_BAD`, CPU, MPS, Modal CPU eval, and synthetic sensitivity-map builds are not promotion evidence.

## Full CUDA/NVDEC Dispatch

Run this from a CUDA host with the current repo, upstream scorer assets, Lane G v3 artifacts, and NVDEC/DALI available:

```bash
bash scripts/remote_lane_g_v3_owv3_fisher_stack.sh
```

For Lightning Studio, stage a reproducible tree first:

```bash
.venv/bin/python scripts/lightning_repro_workspace.py \
  --remote s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai \
  --run-id owv3_repro_contract_YYYYMMDD \
  --artifact experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --artifact experiments/results/lane_g_v3_landed/iter_0/renderer.bin \
  --artifact experiments/results/lane_g_v3_landed/iter_0/masks.mkv \
  --artifact experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt
```

Expected outputs:

```text
lane_g_v3_owv3_fisher_stack_results/hessian_per_weight.pt
lane_g_v3_owv3_fisher_stack_results/owv3_sensitivity_map.pt
lane_g_v3_owv3_fisher_stack_results/archive_lane_g_v3_owv3.zip
lane_g_v3_owv3_fisher_stack_results/contest_auth_eval.json
lane_g_v3_owv3_fisher_stack_results/provenance.json
```

Optional knobs:

```bash
FISHER_TOP_K=30 \
PAIR_BATCH=4 \
BIT_BUDGET_RATIO=0.7 \
PROTECT_THRESHOLD=1e-3 \
AGGRESSIVE_THRESHOLD=1e-5 \
bash scripts/remote_lane_g_v3_owv3_fisher_stack.sh
```

## Modal Fisher/Build-Only Dispatch

Use this only to generate CUDA Fisher/sensitivity/archive artifacts when no exact CUDA eval host is reachable:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_train_lane.py \
  --lane-script scripts/remote_lane_g_v3_owv3_fisher_stack.sh \
  --label lane_g_v3_owv3_fisher \
  --gpu A10G \
  --timeout-hours 10 \
  --env-overrides RUN_CONTEST_EVAL=0
```

Poll and recover:

```bash
.venv/bin/modal call get "$(cat experiments/results/lane_lane_g_v3_owv3_fisher_modal/modal_call_id.txt)"
.venv/bin/python experiments/modal_recover_lane.py --label lane_g_v3_owv3_fisher
```

After recovery, exact CUDA eval still needs a CUDA/NVDEC host:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/lane_lane_g_v3_owv3_fisher_modal/lane_g_v3_owv3_fisher_stack_results/archive_lane_g_v3_owv3.zip \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir experiments/results/lane_lane_g_v3_owv3_fisher_modal/lane_g_v3_owv3_fisher_stack_results/eval_work
```

## Notes

- The script auto-prefers Lane W v2 pair weights if present at `experiments/results/lane_lane_w_v2_modal/harvested_artifacts/lane_w_results/pair_weights.pt`; otherwise it falls back to all pairs.
- `AUTH_EVAL_DEVICE=cpu` is rejected by default because it is advisory only. Set `RUN_CONTEST_EVAL=0` for Fisher/build-only runs.
- The archive is not promotion-grade until `contest_auth_eval.json` records `device: cuda` on the exact `archive_lane_g_v3_owv3.zip` bytes.
- Use `--fallback-action diagnostic_fp16`, `--allow-size-regression`, or
  `--allow-frontier-regression` only for smoke/debug reproductions. Those
  outputs must stay out of promotion and leaderboard claims.
