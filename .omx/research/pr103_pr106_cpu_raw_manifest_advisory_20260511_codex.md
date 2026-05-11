# PR103-on-PR106 CPU raw-output manifest advisory (2026-05-11)

## Scope

Operator concern: prior CPU/CUDA HNeRV conclusions may have conflated score
axis, inflate device, runtime contract, and raw rendered outputs. This pass
ran a local CPU diagnostic with retained raw-output custody so mechanism
analysis can move from "raw outputs missing" to "partial raw-output custody".

No GPU dispatch was launched. This is a local macOS CPU advisory run and is
not a public `[contest-CPU]` reproduction claim.

## Command

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip \
  --inflate-sh submissions/pr103_pr106_final_runtime/inflate.sh \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --device cpu \
  --work-dir experiments/results/dual_device_auth_eval/pr103_pr106_cpu_raw_manifest_20260511T0335Z/work \
  --json-out experiments/results/dual_device_auth_eval/pr103_pr106_cpu_raw_manifest_20260511T0335Z/contest_auth_eval.json \
  --keep-work-dir \
  --inflate-timeout 1800 \
  --evaluate-timeout 1800
```

## Result

- evidence grade: `macOS-CPU advisory`
- score claim: `false`
- promotion eligible: `false`
- rank/kill eligible: `false`
- hardware blocker: `contest_cpu_requires_linux_x86_64`
- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- runtime content tree SHA-256:
  `f2ebe56a408a55b39070f9f86ba77fb11a9b43d83c0e02692f0acc0bf1ff28bb`
- canonical score: `0.22965642878181575`
- pose: `0.00016399`
- seg: `0.00065592`
- archive bytes: `185578`
- inflate elapsed: `36.253875916823745`
- evaluate elapsed: `413.02323295897804`

Retained raw output:

- raw file: `0.raw`
- raw bytes: `3662409600`
- raw SHA-256:
  `8a3b73df0f1a576125319f8587ac65a99b2008bbbab599842c4226695cebe17e`
- aggregate SHA-256:
  `e7a4b402b0ec381616f625985984dc72cfe386fa060d80e358347701bf6351b1`
- inflated output manifest SHA-256:
  `51a205d00929aa828331cca01f00af30e08bb99c97bcd40d5e43dec8a729a657`

## Paired analyzer status

Command:

```bash
.venv/bin/python tools/analyze_cpu_cuda_eval_drift.py \
  --exact-pair \
  experiments/results/dual_device_auth_eval/pr103_pr106_cpu_raw_manifest_20260511T0335Z/contest_auth_eval.json \
  experiments/results/modal_auth_eval/pr103_pr106_dual_runtime_cuda_v2_20260511T022553Z/contest_auth_eval.json \
  --json-out .omx/research/artifacts/pr103_pr106_cpu_raw_vs_cuda_no_raw_20260511_codex/analysis.json \
  --markdown-out /tmp/pr103_pr106_cpu_raw_vs_cuda_no_raw.md
```

Classification:

- valid for pair score analysis: `false`
  - blocker: local CPU is `cpu_advisory`, not Linux x86_64 `[contest-CPU]`
- valid for mechanism analysis: `false`
  - blocker: CUDA side still lacks an inflated-output manifest
- raw output pairing status: `partial_raw_output_manifest`
- same archive SHA: `true`
- same archive bytes: `true`
- same runtime tree: `true`

CUDA minus local CPU advisory:

- score gap: `-0.020673376001992372`
- pose term gap: `-0.022165376001992365`
- seg term gap: `0.0014919999999999933`
- rate gap: `0.0`

## Adversarial interpretation

This run confirms the PR103-on-PR106 local CPU path reproduces the earlier
CPU-class score neighborhood and now has raw-output custody. It does **not**
prove the CPU/CUDA mechanism because the CUDA artifact still lacks raw-output
hashes. It also does not replace Linux x86_64 `[contest-CPU]` evidence.

The next mechanism step is a CUDA rerun with retained
`inflated_output_manifest` on the same archive/runtime. Until that lands,
claims about scorer-kernel drift versus inflate-device drift remain
indeterminate.

## Next action

After the active T1 Modal claim is no longer pending, run a bounded Modal/T4
CUDA rerun of the same PR103-on-PR106 archive/runtime with raw-output manifest
retention. Then re-run `tools/analyze_cpu_cuda_eval_drift.py`; only a pair with
both CPU and CUDA raw-output aggregates can classify the mechanism.
