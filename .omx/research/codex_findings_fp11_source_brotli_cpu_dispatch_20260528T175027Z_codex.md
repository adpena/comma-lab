# FP11 Source Brotli Recode CPU Dispatch

## Summary

- Candidate: `fp11_source_brotli_recode_b7106c9bdbb8`
- Archive SHA-256: `b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c`
- Archive bytes: `178493`
- Source CPU frontier archive SHA-256: `18e3155fbbbe9ab23e1c21bc0d99ba8d18657a71c3129fc5ff9e0405b67d1669`
- Source bytes: `178530`
- Rate delta: `37` bytes saved
- Full-frame shell inflate parity: proven on contest full sample, output raw SHA-256 `00b479229c97ede3e776846297269f7785285702b8dbf3e5dccc733557da605a`
- Exact-readiness: ready row emitted with zero blockers

## Landed Code Boundary

- Commit: `70df24fe3 Make FP11 recode exact-ready`
- Pushed: `origin/main`
- Key proof files:
  - `.omx/research/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/results/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/per_archive/current_contest_cpu_frontier/fp11_source_brotli_recode_v1/fp11_source_brotli_recode/fp11_source_brotli_recode_full_frame_inflate_parity_proof.json`
  - `.omx/research/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/results/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/per_archive/current_contest_cpu_frontier/fp11_source_brotli_recode_v1/fp11_source_brotli_recode/exact_eval_handoff/exact_readiness/fp11_source_brotli_recode_b7106c9bdbb8.exact_ready_queue.json`

## Modal CPU Dispatch

- Lane id: `fp11_source_brotli_recode_b7106c9bdbb8_cpu_exact`
- Instance/job id: `fp11_source_brotli_recode_b7106c9bdbb8_cpu_modal_20260528T174824Z`
- Modal call id: `fc-01KSQVA02JN5VGJ89MD75NFQNC`
- Output dir: `experiments/results/modal_auth_eval_cpu/fp11_source_brotli_recode_b7106c9bdbb8_cpu_20260528T174824Z`
- Initial recovery: `pending` at `2026-05-28T17:50:27Z`
- Recover command: `.venv/bin/python tools/recover_modal_auth_eval.py --output-dir experiments/results/modal_auth_eval_cpu/fp11_source_brotli_recode_b7106c9bdbb8_cpu_20260528T174824Z`

## Notes

The active claim and modal ledger files already contained unrelated partner rows in the working tree, so this memo records the dispatch without absorbing those adjacent state changes into the FP11 code/artifact commit.
