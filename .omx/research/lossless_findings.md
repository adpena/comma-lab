# Lossless Findings

## current promoted result

- Current promoted rate: `1.5947`.
- Profile: `lzma_baseline`.
- Method: `lzma`.
- The lossless promotion flow is separate from the lossy promotion flow.
- Lossless report surfaces are derived from the promoted lossless result record.

## 2026-04-11 follow-on work

- `zpaq` is now installed locally and an exact local baseline run has completed archive construction; exact contract replay is underway.
- `zpaq` remains `local_only` unless a self-contained runtime is bundled into the submission payload.
- `gpt_arithmetic_small` and `gpt_arithmetic_large` now have real workload estimates over splits `0,1`, each at `774005000` flattened tokens.
- `gpt_arithmetic_small` split `0` token-stream materialization produced `train_split0.bin` at `774005000` bytes.
- Council memo conclusion: lossless optimization is purely archive-byte minimization under exact reconstruction; no soft distortion term exists to trade against bytes.
