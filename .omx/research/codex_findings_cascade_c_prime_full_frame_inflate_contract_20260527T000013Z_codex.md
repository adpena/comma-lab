<!-- HISTORICAL_PROVENANCE: APPEND-ONLY per Catalog #110/#113. Codex WAVE-4 receiver-contract fix for Cascade C' full-frame inflate scaffold. -->

# Codex findings: Cascade C' full-frame inflate contract

**UTC:** 2026-05-27T00:00:13Z  
**Agent:** Codex  
**Lane:** `lane_cascade_c_prime_option_a_build_scaffold_20260526`  
**Substrate:** `cascade_c_prime_frame_1_segnet_waterfill`

## Verdict

Wave-3 was an implementation-level receiver scaffold bug, not a paradigm-level
negative. The Cascade C' archive/runtime reached inflate, but the scaffold
receiver wrote local low-resolution `arc.n_pairs * 2` frames instead of the
contest raw-output contract.

## Bug class extincted

Old scaffold behavior:

- `height=384`, `width=512`
- `n_frames=arc.n_pairs * 2`
- full-run `n_pairs=600` therefore wrote `707,788,800` bytes

Contest contract:

- `height=874`, `width=1164`
- `n_frames=1200`
- expected raw bytes `3,662,409,600`

The fixed receiver now keeps the archive parser and routing/menu/pose streams
deterministically consumed, then emits the contest-sized raw stream through a
sparse all-zero writer. Local smokes with fewer than 600 pair decisions pad to
the full contest shape; archives that encode more than 1200 frames fail closed.

## Authority posture

This is receiver/runtime contract hardening only.

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- MLX local smoke remains `[macOS-MLX research-signal]`
- no contest score, rank, promotion, or kill authority is inferred

The receiver remains decode-only: it does not inspect scorer state, optimize,
fetch sidecars, or adapt at eval time.

## Verification

- `.venv/bin/ruff check src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/tests/test_scaffold_smoke.py`
- `.venv/bin/pytest src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/tests/test_scaffold_smoke.py -q`
  - `15 passed`
- MLX-local smoke:
  - command: `.venv/bin/python experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py --output-dir .omx/research/cascade_c_prime_full_frame_runtime_mlx_smoke_20260526T_after_wave4_codex --device cpu --smoke --n-pairs 8`
  - `stage_3_mlx_compress_pass_begin n_pairs=8`
  - `archive_bytes=184`
  - `axis=[macOS-MLX research-signal]`
- Inflate smoke:
  - `raw_bytes=3,662,409,600`
  - `raw_size_matches_contest_contract=true`
  - `head_16_zero=true`
  - `tail_16_zero=true`

## Next concrete gate

Re-fire the paired-axis Cascade C' WAVE-4 dispatch with the same Modal T4 lane
only after this commit is pushed. Expected outcome: the prior `WRONG-SIZE .raw`
auth-eval blocker should be extinct; the next artifact should either produce a
valid contest-axis auth-eval payload or expose the next implementation-level
runtime blocker with no score authority leakage.
