The charter is complete. The best learner saved only 368 B, so the 3,000 B receiver-integration trigger did not fire. No receiver patch, candidate archive, Modal run, or scorer run was produced.

Key results:

- Strict T4 receiver slack: 27.581 s, or 233.809 ns/symbol—not the charter’s assumed ~540 s.
- Lane residual stack: 119,381 B, saving 368 B; projected T4 overhead 36.721–40.247 s.
- Leaky recurrence: 119,784 B, adding 35 B; projected 25.355–31.392 s.
- Previous-frame hash: 119,635 B, saving 114 B; projected 15.768–19.795 s.
- Best result realizes 1.42% of the 25,899 B demand and 4.08% of tc1’s 9,011 B oracle.
- Move 44 already contains a 23-family PAQ-style online adaptive corrector, falsifying the premise that the vehicle lacked online context learning.
- Frontier remains `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_mxo1_free_decode_time_online_context_mixing_20260911.md)
- [Reproducible experiment](/Users/adpena/Projects/pact/experiments/ddm_mxo1_online_context_mixing.py)
- [Retained payloads and checkpoints](/Volumes/VertigoDataTier/pact/ddm_mxo1_free_decode_time_online_context_mixing/)
- Primary commit: `b5dd73f1e`
- Terminal-format amendment: `f28812aa6`

The exact shipped control and all four streams matched an independent cold repeat byte-for-byte. Checkpoints were preserved every 25 frames. Cross-host determinism remains unproven for the prototype’s floating-point surfaces; receiver adoption would require integerization and Python/C full-stream twins. Ruff, compilation, targeted tests (`7 passed`), two review passes, and serializer preflight all passed. The `ddm_mxo1` checkpoint is recorded COMPLETE.

## NEXT_IF_RESUMED

- `ddm_mxo2_full_family_residual_stacker`; disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN`; consumer store=`/Volumes/VertigoDataTier/pact/ddm_mxo1_free_decode_time_online_context_mixing/`; fire trigger=`an exact n600 integer-row screen over the shipping 23-family pre-mix outputs predicts at least 3,000 B savings and a native dry-run costs at most 200 ns/symbol, or a new exact T4 receipt establishes at least 60 s strict receiver slack`.

## LIVE-HYPOTHESES

- A low-rank nonlinear stacker over the 23 pre-mix family predictions may recover information lost when those predictions collapse into the final shipped probability.
- Motion-aligned previous-frame contexts may outperform the tested collision-heavy, fixed-location 3×3 hash, provided a generic native warp fits the compute budget.

## DEAD-ENDS

- Rebuilding a generic PAQ/cmix-style learner on the premise that move 44 lacks online adaptation: the shipped corrector already performs it.
- Three-expert Lane residual stack in the tested formulation: only 368 B saved and projected beyond strict slack.
- Tested 25-state leaky recurrence: added 35 B.
- Static previous-frame 3×3 hash: only 114 B saved, far below the temporal gate.
- Raster-order replay as an instrument: reproduced length but not the shipped RC64 bytes; only 190-group order is valid.