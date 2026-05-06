# Field/Meta Candidate-Packet Dispatch Selection

**Artifact:** `.omx/research/field_meta_dispatch_selection_20260506_codex.json`

**Scope:** local, deterministic field/meta selection only. No score claim, no
remote/GPU dispatch, and no lane claim was created.

## What Changed

- Added `tools/build_field_meta_dispatch_selection.py` as a generic consumer of
  local candidate-packet manifests.
- 2026-05-06 adversarial review fix: split static packet readiness from
  dispatch authorization. A row can now set `candidate_static_preflight_ready`
  when local archive, runtime, and strict static preflight pass, but
  `ready_for_exact_eval_dispatch` remains false until the selector verifies a
  matching active Level-2 lane claim for the manifest lane/job.
- Static candidate readiness requires all of:
  - local archive file proof with matching bytes and SHA-256;
  - ZIP custody proof: readable ZIP, no duplicate members, safe member names,
    and matching local/central header names;
  - runtime proof with a valid `runtime_tree_sha256`;
  - static candidate preflight passing through
    `experiments/preflight_candidate_manifest_dispatch_readiness.py`.
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
  --claims-path .omx/state/active_lane_dispatch_claims.md \
  --now-utc 2026-05-06T22:53:55Z \
  --json-out .omx/research/field_meta_dispatch_selection_20260506_codex.json
```

Summary:

- `candidate_count=4`
- `candidate_static_preflight_ready_count=1`
- `ready_candidate_count=0`
- report-level `ready_for_exact_eval_dispatch=false`
- selected local static-ready packet:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/public_replay_preflight.json`
- selected packet blockers:
  `missing_active_lane_dispatch_claim`,
  `claim:dispatch_lane_id_missing`,
  `claim:dispatch_instance_job_id_missing`
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

The selector may report a local candidate as `candidate_static_preflight_ready`
when byte custody, runtime custody, and strict static manifest preflight pass.
It may report `ready_for_exact_eval_dispatch=true` only when that static packet
also has a matching active Level-2 lane claim in
`.omx/state/active_lane_dispatch_claims.md`. This run created no claim, made no
score claim, and dispatched no remote/GPU work. Non-ZIP archives, hidden ZIP
members, duplicate members, zip-slip names, and central/local header name
mismatches block static readiness.
