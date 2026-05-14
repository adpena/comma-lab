# C6 decoder_blob:34858 Exact CUDA Dispatch - 2026-05-14

## Candidate

- lane_id: `lane_c6_ibps1_decoder34858_byte_patch_exact_cuda_20260514`
- instance_job_id: `c6_ibps1_decoder34858_patch_t4_20260514T182251Z`
- Modal call_id: `fc-01KRKVR2VK30AVTW35J5SCAW4Z`
- Modal app run: `https://modal.com/apps/adpena/main/ap-M6NQIQfdAnL5mQTr3LGktU`
- archive: `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_34858/submission_dir/archive.zip`
- archive_sha256: `2d6416874c6563d6f2ebf9b502c98a45b5b4dfee88f42124dbbaa1910579bb3a`
- archive_size_bytes: `224481`
- source_repo_commit: `e1015379cb36c2bf94724a89bf9551145aff1756`

## Runtime

- submission_dir: `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_34858/submission_dir`
- inflate_sh: `inflate.sh`
- local compliance runtime_tree_sha256: `82d270c997dbbb91ad3c06a7737cd3c3980a126e03991ee8a86a2bed8df3635e`
- Modal uploaded runtime_tree_sha256: `03c0788a1fb2cc4842803135820b350cf5db62f598403954996faef972c56f98`
- Modal uploaded runtime_content_tree_sha256: `439b8b0998cd0f0c23988086270c3ad7fddca4f1e6dcb0198cd2d84a93af207c`
- submission_dir_zip_sha256: `a97a686e34cce186f2baec5b24a0b3f3c7e2a5e76d03061d2d7937218a93fb87`

The first local Modal entrypoint attempt intentionally failed before dispatch
because it used the local compliance runtime hash as
`--expected-runtime-tree-sha256`. The second attempt used the projected Modal
uploaded runtime hash and passed the pre-dispatch runtime-tree check.

## Command

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval.py \
  --archive experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_34858/submission_dir/archive.zip \
  --output-dir experiments/results/modal_auth_eval/c6_ibps1_decoder34858_patch_t4_20260514T182251Z \
  --inflate-sh inflate.sh \
  --submission-dir experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_34858/submission_dir \
  --gpu T4 \
  --scorer-device cuda \
  --inflate-device auto \
  --expected-runtime-tree-sha256 03c0788a1fb2cc4842803135820b350cf5db62f598403954996faef972c56f98 \
  --detach \
  --provider-detach-ack \
  --lane-id lane_c6_ibps1_decoder34858_byte_patch_exact_cuda_20260514 \
  --instance-job-id c6_ibps1_decoder34858_patch_t4_20260514T182251Z \
  --claim-agent codex:gpt-5.5 \
  --claim-notes "Exact CUDA auth eval for CPU600-confirmed C6 IBPS1 decoder_blob:34858 byte-patch candidate; score_claim=false until recovery; modal_projected_runtime_tree=03c0788a1fb2cc4842803135820b350cf5db62f598403954996faef972c56f98"
```

## Status

- pre-auth-eval compliance: pass, with expected warning
  `auth_eval_optional_missing`
- first recovery attempt:
  `tools/recover_modal_auth_eval.py --output-dir experiments/results/modal_auth_eval/c6_ibps1_decoder34858_patch_t4_20260514T182251Z`
- recovery status at `2026-05-14T18:25:13Z`: `pending`
- score_claim: `false`
- promotion_eligible: `false`

Next action: rerun the recover command until `modal_cuda_auth_eval_result.json`
and `contest_auth_eval.json` materialize or the call fails, then adjudicate
from the recovered archive/runtime custody.
