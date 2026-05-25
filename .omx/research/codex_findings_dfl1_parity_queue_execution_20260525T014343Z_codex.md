# Codex Findings: DFL1 Parity Queue Execution Bridge

UTC: 2026-05-25T01:43:43Z
Lane: codex_dfl1_parity_queue_execution_20260525

## Summary

The renderer payload DFL1 full-frame parity bridge is now queue-owned instead of
requiring a manually pre-attached proof in the initial materializer context.
When a DFL1 work row includes a source runtime plus a full-frame file list, the
experiment queue inserts a shell parity proof step between materialization and
harvest, then passes the compact proof artifact into harvest before the
exact-readiness bridge runs.

## Landed Engineering

- `tools/prove_shell_inflate_parity.py` now fails closed unless the compact proof
  carries `full_frame_inflate_output_parity_claim=true`; byte equality without a
  full-frame file-list claim exits nonzero.
- The shell parity proof tool refuses non-empty/symlinked output directories and
  cleans transient raw scratch on failure by default.
- `byte_shaving_campaign_queue` emits
  `prove_renderer_payload_dfl1_shell_parity` as a queue step for DFL1 contexts
  with `source_runtime_dir` plus `full_frame_file_list` or file-list entries.
- `harvest_materializer_chain_candidates.py` accepts
  `--renderer-payload-dfl1-inflate-parity-proof` and re-verifies sidecar proof
  artifacts against source/candidate archive SHAs before annotating harvested
  rows.
- Exact-readiness remains the authority boundary: sidecar proof rows still
  carry no score/promotion authority and only become dispatch-ready through the
  existing exact-readiness promoter.

## Verification

- `py_compile` passed for the modified tools, scheduler modules, optimizer
  adapter, and tests.
- `ruff check` passed for the modified files.
- `test_materializer_chain_harvest_scheduler.py`: 39 passed.
- `test_byte_shaving_campaign_queue.py`: 66 passed.
- `test_optimizer_exact_readiness.py`: 61 passed.
- Broader materializer/readiness pack:
  `test_family_agnostic_materializers.py`,
  `test_materializer_chain_harvest_scheduler.py`,
  `test_byte_shaving_campaign_queue.py`,
  `test_optimizer_exact_readiness.py`: 195 passed.
- `test_byte_shaving_materializer_campaign_runner.py`: 49 passed.

## Remaining Work

1. Run the new DFL1 queue-owned parity path on the real robust-current full
   contest file list and preserve only compact proof/manifests by default.
2. Extend sidecar proof ingestion to grouped materializer chains so DFL1 can
   compose with section entropy recode, packet member recompress, tensor
   factorization, and header elision without a manifest rerun.
3. Promote the same sidecar-proof contract into the learned grouped acquisition
   surface so operation sets can require proof-producing follow-up nodes.
4. Continue the PR95/HNeRV MLX reproduction lane in parallel: HNeRVDecoderMLX,
   native resize/pixel-shuffle, Muon/AdamW, stage timing smokes, PyTorch export
   parity, and byte-closed archive smoke.

