# zpaq local-only exact result

## status

- exact round-trip confirmed over `5000` items
- classification: `local_only`

## result

- profile: `zpaq_baseline`
- method: `zpaq`
- archive: `reports/raw/2026-04-11-commavq-zpaq-baseline-rerun/zpaq_baseline_submission.zip`
- archive bytes: `538876316`
- original bytes: `960000000`
- compression ratio: `1.7814848630311673`

## interpretation

- This beats the promoted `lzma_baseline` floor locally.
- It is not promoted as the canonical lossless floor because the current `zpaq` runtime story is still local-only, not challenge-valid.
- Exact evidence lives in `zpaq_baseline_result.json`.
