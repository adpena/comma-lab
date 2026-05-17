# Draft PR Body - FEC6 CPU-axis packet

This is the short public-facing cut. It intentionally excludes internal process,
funding, employment, council, and speculative causal-budget material from the
long research dossier in `docs/pr_writeups/cpu_frontier_fec6_20260517.md`.

## Claim

Lower score is better.

| axis | evidence status | score | seg dist | pose dist | archive bytes |
|---|---|---:|---:|---:|---:|
| Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending | primary | `0.192051316881` | `0.00056029` | `0.00002943` | `178517` |
| Modal T4 CUDA replay | paired context, not primary | `0.226210021693` | `0.00066299` | `0.00016846` | `178517` |

Archive SHA-256:

```text
6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
```

Evidence artifacts:

```text
experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/modal_cpu_auth_eval_result.json
experiments/results/modal_auth_eval/archive_6bae0201fb08/modal_cuda_auth_eval_result.json
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip
experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.sh
```

The CPU result is currently a Modal Linux x86_64 reproduction, not a
host-bot/GitHub Actions validation. The public body should keep that distinction
until the host/GHA artifact exists.

## What Changed

This packet starts from the public PR101 HNeRV-family grammar and adds FEC6:
a K=16 frame-conditional per-pair selector encoded with a compact fixed-Huffman
selector stream. The selector is chosen offline for the fixed 600-pair contest
video and consumed deterministically by `inflate.py`; no scorer or search runs
at inflate time.

Novel mechanism:

- PR101 baseline: HNeRV decoder, qpose14/qzs3 pose encoding, FP4 weights.
- FEC6 addition: per-pair mode index from a 16-mode frame-0 chroma/tile palette.
- Rate control: fixed-Huffman encoding of selector indices.
- Runtime contract: deterministic archive bytes and deterministic inflate path
  for a fixed hardware axis.

## Reproduce

Current checked artifact path:

```bash
sha256sum experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip
```

Expected:

```text
6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
```

Modal CPU replay command from the evidence JSON:

```bash
/usr/local/bin/python -u /workspace/pact/experiments/contest_auth_eval.py \
  --archive /tmp/modal_auth_eval_cpu/archive.zip \
  --inflate-sh /tmp/modal_auth_eval_cpu/submission_dir/inflate.sh \
  --upstream-dir /workspace/pact/upstream \
  --video-names-file /workspace/pact/upstream/public_test_video_names.txt \
  --device cpu \
  --keep-work-dir \
  --work-dir /root/modal_auth_eval_cpu_work/eval_work \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --expected-runtime-tree-sha256 f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166
```

For a local checkout, the equivalent paths are:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --inflate-sh experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --video-names-file upstream/public_test_video_names.txt \
  --device cpu \
  --expected-runtime-tree-sha256 f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166
```

## Limitations

- Host/GHA validation for this exact archive is pending.
- The Modal T4 CUDA replay is paired context, not the promoted axis.
- The CPU/CUDA split is observed and documented by paired artifacts; detailed
  causal attribution should remain hypothesis-labeled until controlled toggles
  isolate each factor.
- This packet is contest-specific: it uses offline access to the fixed contest
  video and should not be framed as directly production-deployable.

## Appendix Links

Long research dossier:

```text
docs/pr_writeups/cpu_frontier_fec6_20260517.md
```

Senior-engineer/taste review:

```text
.omx/research/pr_body_senior_engineering_taste_review_20260517_codex.md
```
