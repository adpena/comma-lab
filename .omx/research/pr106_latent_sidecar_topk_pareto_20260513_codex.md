# PR106 latent sidecar top-k Pareto profile - 2026-05-13

## Scope

Planning-only, no score claim. This ledger records a deterministic top-k
Pareto profile over the existing PR106 latent score table. The selector signal
is the compress-time pair objective in the score table, not exact contest eval.

## Command

```bash
.venv/bin/python tools/profile_pr106_latent_sidecar_topk_pareto.py \
  --score-table-npy reports/raw/kaggle_ingested/kaggle_pr106_latent_score_table_20260513_codex_clean/pr106_latent_score_table/latent_run/score_table/score_table.npy \
  --score-table-manifest reports/raw/kaggle_ingested/kaggle_pr106_latent_score_table_20260513_codex_clean/pr106_latent_score_table/latent_run/score_table/score_table_manifest.json \
  --source-archive experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip \
  --n-pairs 600 \
  --latent-dim 28 \
  --delta-radius 2 \
  --top-k-values 0,1,2,4,8,16,32,64,96,128,192,256,384,512,600 \
  --json-out experiments/results/pr106_latent_sidecar_topk_pareto_20260513_codex/profile.json \
  --md-out experiments/results/pr106_latent_sidecar_topk_pareto_20260513_codex/profile.md
```

## Custody

- Tool: `tools/profile_pr106_latent_sidecar_topk_pareto.py`
- JSON report: `experiments/results/pr106_latent_sidecar_topk_pareto_20260513_codex/profile.json`
- JSON bytes: `98714`
- JSON sha256: `aa375afbaffcd96a526619606b0b7adcdbc3636f6e545ec6dcca55ceb017034f`
- Markdown report: `experiments/results/pr106_latent_sidecar_topk_pareto_20260513_codex/profile.md`
- Markdown bytes: `2906`
- Markdown sha256: `2be48c0f4b706f5697eeab5b7bfc83f38f07c6c7d221601ede282203b0ed1ecf`
- Score table sha256: `326038a5e2dbb9daf8498e7e8c07fdfc15d832b443b7ad1758cab4f0517001fe`
- Candidate grid sha256: `35903b191807c4fb11c1484c0984cbb4c2c09fc3fa647e4d22e569b54528fb43`
- Manifest validation: source archive SHA mismatched the Kaggle private path,
  but single-member payload SHA matched via the canonical PacketIR reader
  (`0.bin`/`x` both accepted).

## Findings

All 600 pairs have a strict compress-time improvement at radius 2. The full
600-pair selection remains on the Pareto frontier: it retains 100% of the
selector improvement and costs 533 runtime-consumed PR101-grammar sidecar bytes
inside the sidecar wrapper.

Pareto frontier, using the already-runtime-consumed PR101 ranked/no-op sidecar
grammar:

| top_k | corrections | selector improvement sum | runtime sidecar bytes | rate delta vs top_k=0 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.000000000000 | 9 | 0.000000000000 |
| 64 | 64 | 0.368315413594 | 100 | 0.000060593165 |
| 128 | 128 | 0.660874638706 | 175 | 0.000110532586 |
| 256 | 256 | 1.147431150079 | 305 | 0.000197094250 |
| 384 | 384 | 1.534571226686 | 414 | 0.000269672876 |
| 512 | 512 | 1.839074354619 | 500 | 0.000326936746 |
| 600 | 600 | 1.967402987182 | 533 | 0.000348910091 |

The curve is smooth rather than cliff-like: top_k=512 keeps 93.48% of selector
gain at 500 bytes, while top_k=600 adds 33 bytes for the final 6.52% of
selector gain. This suggests the next exact-eval candidate should not be an
arbitrary smaller sidecar. Either keep the known 600-pair semantic selection or
create a new candidate only if the exact-eval cost of the long tail is suspected
to hurt PoseNet/SegNet more than the selector table predicts.

## Claim discipline

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

This profile is not a contest score and does not make a candidate archive
submission-ready. It only establishes the semantic/rate frontier for the
existing measured score table. Any promoted candidate still needs a
byte-closed archive, runtime-consumption proof, no-op proof, lane claim, exact
contest CUDA eval, paired CPU if used for public-axis reasoning, and
pre-submission compliance.

## Next action

The full 600-pair PR101-grammar sidecar remains the mathematically clean next
semantic sidecar baseline. The higher-EV score-lowering work is now:

1. Materialize one exact packet from this canonical top-k selector only if the
   runtime path is already the PR101-grammar sidecar path and the archive SHA is
   byte-closed.
2. Use the same profiler to evaluate radius-3/radius-4 score tables only after
   a real CUDA table exists.
3. Spend engineering effort on the larger HDM4 decoder payload, HNeRV parity
   training loop, or new substrate trainers before inventing another sidecar
   byte grammar below the 533-byte PR101 grammar frontier.
