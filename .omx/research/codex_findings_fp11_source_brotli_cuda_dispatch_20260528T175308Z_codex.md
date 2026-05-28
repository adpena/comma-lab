# FP11 Source Brotli Recode CUDA Dispatch

## Summary

- Candidate: `fp11_source_brotli_recode_b7106c9bdbb8`
- Archive SHA-256: `b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c`
- Archive bytes: `178493`
- Exact-readiness source: `.omx/research/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/results/frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/per_archive/current_contest_cpu_frontier/fp11_source_brotli_recode_v1/fp11_source_brotli_recode/exact_eval_handoff/exact_readiness/fp11_source_brotli_recode_b7106c9bdbb8.exact_ready_queue.json`
- CPU sibling call: `fc-01KSQVA02JN5VGJ89MD75NFQNC`

## Modal CUDA Dispatch

- Lane id: `fp11_source_brotli_recode_b7106c9bdbb8_cuda_exact`
- Instance/job id: `fp11_source_brotli_recode_b7106c9bdbb8_cuda_modal_20260528T175216Z`
- Modal call id: `fc-01KSQVFBVHVEWTCR3QS04R8HX1`
- GPU: `T4`
- Output dir: `experiments/results/modal_auth_eval/fp11_source_brotli_recode_b7106c9bdbb8_cuda_20260528T175216Z`
- Initial recovery: `pending` at `2026-05-28T17:53:08Z`
- Recover command: `.venv/bin/python tools/recover_modal_auth_eval.py --output-dir experiments/results/modal_auth_eval/fp11_source_brotli_recode_b7106c9bdbb8_cuda_20260528T175216Z`

## Notes

Launched in parallel with the CPU anchor to measure the CPU/CUDA behavior of the same byte-closed, full-frame-parity FP11 packet without waiting for another turn. Ledger files had unrelated unstaged partner rows, so this dispatch is memo-preserved without absorbing adjacent WIP.
