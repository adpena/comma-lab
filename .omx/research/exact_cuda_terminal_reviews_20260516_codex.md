# Exact CUDA terminal reviews

- date: `2026-05-16`
- agent: `codex`
- review_directory: `.omx/research/exact_cuda_terminal_reviews_20260516_codex/`
- terminal_review_rows: `31`
- unique_review_packets: `30`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Scope

Backfilled exact-CUDA terminal review packets for recovered Modal auth-eval
outputs. These packets preserve real contest-CUDA evidence and forensic
classification, but none are paired promotion results and none should be used
as rank/kill authority without matching baseline and CPU/CUDA pairing.

The count distinction is intentional: 31 terminal claim rows were covered, and
30 unique review-packet files were written because the
`modal_hdm8_cuda_selector_sparse_top001_t4_20260515T030736Z` terminal claim
appears twice and maps to the same packet.

The review rows were appended to:

```text
reports/cathedral_autopilot_evidence.jsonl
```

## Best Reviewed CUDA Scores In This Batch

Lower is better, but these are still non-promotional single-axis reviews:

| score_axis | score | review_packet | note |
| --- | ---: | --- | --- |
| `contest_cuda` | `0.205330029020` | `.omx/research/exact_cuda_terminal_reviews_20260516_codex/pr106_format0d_latent_score_table_paired_modal_auth_20260516T071622Z_cuda_review.json` | best reviewed score in this batch; no CPU pair |
| `contest_cuda` | `0.206316386616` | `.omx/research/exact_cuda_terminal_reviews_20260516_codex/pr106_format0c_exact_radix_paired_20260515T0918Z_cuda_review.json` | duplicate score family with HDM12/format0c |
| `contest_cuda` | `0.206316386616` | `.omx/research/exact_cuda_terminal_reviews_20260516_codex/pr106_hdm12_hlm3_fmt0c_t4_20260515T090445Z_review.json` | duplicate score family with format0c |
| `contest_cuda` | `0.206325708641` | `.omx/research/exact_cuda_terminal_reviews_20260516_codex/pr106_hdm11_hlm3_fmt0b_t4_20260515T073414Z_review.json` | no CPU pair |
| `contest_cuda` | `0.206331035513` | `.omx/research/exact_cuda_terminal_reviews_20260516_codex/pr106_hdm10_hlm3_fmt0a_t4_20260515T055017Z_review.json` | no CPU pair |

## Decision

Preserve as signal, not authority:

- exact CUDA evidence exists for these measured configurations;
- `score_claim=false`;
- `promotion_eligible=false`;
- `ready_for_exact_eval_dispatch=false`;
- matching `contest_cpu` evidence and paired runtime identity are still required
  before any L5 staircase promotion or architecture-lock decision.

The D1 modal-only row is preserved as reviewed contest-CUDA evidence, but its
packet remains explicitly non-promotional because the forensic audit reports
`runtime_manifest_missing` and `runtime_payload_closure_missing`.

This batch reinforces the L5 v2 pair-identity hardening: single-axis CUDA
scores are useful for forensic analysis and prioritization, but they cannot be
silently spliced into paired CPU/CUDA evidence.

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/backfill_terminal_claim_evidence.py --dry-run --max-rows 25
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/all_lanes_preflight.py
```

Observed:

- `missing_terminal_claims: 0`
- Gate #29 `terminal dispatch evidence coverage`: PASS
- Gate #10 still fails on untracked current-turn research artifacts before
  commit and the pre-existing `experiments/results/` runtime-source baseline
  drift; this review batch does not silently normalize that drift.
