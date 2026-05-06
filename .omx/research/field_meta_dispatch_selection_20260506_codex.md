# Field/Meta Candidate-Packet Dispatch Selection

**Artifact:** `.omx/research/field_meta_dispatch_selection_20260506_codex.json`

**Scope:** local, deterministic field/meta selection only. No score claim, no
remote/GPU dispatch, and no lane claim was created.

## What Changed

- Added `tools/build_field_meta_dispatch_selection.py` as a generic consumer of
  local candidate-packet manifests.
- The selector requires all of:
  - local archive file proof with matching bytes and SHA-256;
  - runtime proof with a valid `runtime_tree_sha256`;
  - `experiments/preflight_candidate_manifest_dispatch_readiness.py` returning
    `ready_for_exact_eval_dispatch=true` for the same manifest.
- `tools/build_frontier_roadmap_status.py` can now ingest packet manifests via
  `--packet-manifest` / `--packet-manifest-glob` and embeds the selection report
  under `next_comprehensive_tranche.field_meta_candidate_packet_selection`.

## Local Evidence

Command:

```bash
.venv/bin/python tools/build_field_meta_dispatch_selection.py \
  --manifest experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/public_replay_preflight.json \
  --manifest experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json \
  --manifest experiments/results/joint_stack_noop_manifest_20260506_codex/manifest.json \
  --manifest experiments/results/pr91_hpm1_readiness_20260506_codex/readiness.json \
  --json-out .omx/research/field_meta_dispatch_selection_20260506_codex.json
```

Summary:

- `candidate_count=4`
- `ready_candidate_count=1`
- selected local static-ready packet:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/public_replay_preflight.json`
- blocked WR01 apply manifest next proof:
  `passing_experiments/preflight_candidate_manifest_dispatch_readiness.py`
- blocked JCSP noop next proofs:
  `local_archive_file_with_matching_sha256_and_bytes`,
  `runtime_tree_sha256_from_public_replay_preflight_or_exact_runtime_contract`,
  `passing_experiments/preflight_candidate_manifest_dispatch_readiness.py`
- blocked PR91/HPM1 next proofs:
  `local_archive_file_with_matching_sha256_and_bytes`,
  `runtime_tree_sha256_from_public_replay_preflight_or_exact_runtime_contract`,
  `passing_experiments/preflight_candidate_manifest_dispatch_readiness.py`

## Guardrail

The selector may report a local candidate as `ready_for_exact_eval_dispatch`
only when the strict manifest preflight reports the same readiness. It still
emits report-level blockers requiring a lane dispatch claim and exact CUDA auth
eval before any remote/GPU work or score claim.
