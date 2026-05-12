# Sparse L2 wavelet real-prefix probe (2026-05-11)

## Verdict

**No score claim.** This is a local CPU prefix-smoke and byte-closure probe,
not `[contest-CPU]`, not `[contest-CUDA]`, and not `[macOS-CPU advisory]`.

The useful finding is a wire-format / byte-budget result:

- full sparse Wavelet-L2 residual on a real 2-frame PR106-r2 prefix is too
  large for the intended score-lowering envelope;
- top-k sparse truncation can produce a byte-closed prefix candidate under the
  same sparse runtime grammar, but its L2 proxy rate term is post-hoc and
  cannot be promoted without full 1200-frame exact eval.

## Custody

- summary:
  `experiments/results/sparse_l2_wavelet_realprefix_20260511_codex/summary.json`
- decoded PR106-r2 prefix raw SHA-256:
  `ed86276e2d6c957e9d9b87052586f364d37b251099107f8d1235492c51f18d2b`
- GT prefix raw SHA-256:
  `b4ff7cd4db07d4e8b5224dd65386af9eb9b069f889989b4b80d1b22c3acbcf5a`
- sample scope: first 2 frames only

## Findings

### Full sparse residual negative

- archive bytes: `22,394,727`
- residual bytes: `22,208,003`
- archive SHA-256:
  `990399a8e1d6ad8efed11de72de745cfebe70a111d26722ce2a83e36694bb842`
- refusal against small budget:
  `sparse residual bytes 22208003 exceed --byte-budget 100000`
- classification:
  `exact_prefix_negative_sparse_wavelet_l2_too_large_for_score_lowering`

This falsifies the naive assumption that simply wrapping the dense L2 wavelet
residual in sparse PacketIR makes it byte-competitive.

### Top-512 prefix candidate

- archive bytes: `191,927`
- residual bytes: `5,203`
- archive SHA-256:
  `b53c5936303f5d6598ed224db22dacf497431c5356d8ef78972f3a386733ac40`
- classification:
  `budgeted_prefix_candidate_no_score_claim_needs_full_exact_eval_before_status_change`

This is a useful compiler primitive proof, not a score result. The materializer
records `sparse_rate_term_is_posthoc_not_encoder_loss=1.0`, so the dense
Wavelet-L2 proxy objective is not mathematically aligned with the emitted
truncated sparse bytes.

## Code changes justified by this probe

- sparse PacketIR temporal wrapping now supports non-uniform frame payloads via
  length-prefixed/padded signal frames;
- sparse PR106 family inflates accept both the new length-prefixed signal-frame
  payload and the legacy raw signal-frame payload;
- Wavelet-L2 materialization exposes `--sparse-top-k-per-frame` so sparse byte
  budgets can be explored without pretending dense-proxy rate is a score term.

## Next work

1. Promote top-k selection from magnitude-only to score-aware saliency over
   full 1200-frame residuals.
2. Run exact runtime smoke and no-op mutation proof for every full-sequence
   archive.
3. Run `[contest-CPU]` only on Linux x86_64 with the exact contest
   `archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu` path.
4. Promote to CUDA only if exact CPU and custody packet are positive.

Until then, this lane remains `score_claim=false`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.
