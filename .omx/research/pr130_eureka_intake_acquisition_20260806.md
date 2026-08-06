# PR130 eureka-harvest ACQUISITION receipt (MAIN, 2026-08-06)

Operator charter: *"harvest all signal possible from PR130... look at the archive, but also the
author's repo and any other signal available anywhere."* MAIN did the network-bound acquisition;
arm ddm_eh1 does the offline deep forensics. HARVEST-SIGNAL-ONLY per NO-FAKE #7. Pointer delta: 0.

## Custody (all under /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/)

| asset | bytes | sha256 (verified against PR manifests) |
|---|---:|---|
| releases/cpr1/archive.zip — THE RANK-1 archive | 191,052 | 0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd |
| releases/landslide/archive.zip — predecessor | 194,380 | f4457de09a6e69c8cd29e886a84705462a8c77dc6978020b11dff52e661a1451 |
| repro_repo/ — fesalfayed/comma-ai-semantic-pose-hpac-cpr1 (FULL 49-stage training pipeline) | 6.3M | clone (HEAD recorded in eh1 receipt) |
| fork_clone/ — fesalfayed fork, master + agent branch (blob-filtered) | — | unique commit 9a77b6a |
| pr130_comments.txt · pr86_comments.txt · jas0xf_repos.json | — | gh exports |

Sources: the PR's own compress.sh named the fork release URL + shas; both downloads sha-match.
The reproduction repo was found via the author's profile (repo `comma-ai-semantic-pose-hpac-cpr1`,
"Frozen byte-exact CPR1 rebuild and strict raw-video-to-score reproduction", pushed 07-19).

## What the recipe already shows (DERIVED-from-recipe; eh1 verifies at file:line)

1. 49-stage hashed pipeline raw-video → archive → official eval (`scripts/e2e.py`), trainers
   included: train_semantic_full/quantized, train_pose_carrier_full, train_hpac_self_compress,
   refine_pose_coeff_codes.
2. Renderer: width-96, four-block, 4-bit semantic renderer (stages 02–08) → 40,252 B.
3. Pose: random-init 12-DIRECTION POSE BASIS + per-pair coefficients, exact int12 searches,
   six-bit stabilization + coefficient tail, Huffman/Rice repack (stages 09–32) → 23,054 B.
4. Coder: random-init INTEGER HPAC, patch-64 migration, JOINT model-rate/token-rate
   self-compression (stages 33–41) → model 20,179 B + tokens 116,980 B.
5. Token-economics derivation (MAIN, arithmetic): 116,980 B @ their 0.007933 bpp over
   117,964,800 px = exactly 600 × 384×512 — the token stream covers ONLY the seg-scored frame's
   plane; frame_0 carries no seg bytes.
6. Their early-stop pattern: deployed checkpoints SELECTED before scheduler horizon (pose
   hard-mining 750/4000, coefficient rescue 1000/2000, basis adaptation best@250/2000).

Gaps pi1 could not close, now closable: pi1 (2026-07-28) had the anatomy from verification.json
but never the archive BYTES nor any training-side code. Consumer: ddm_eh1 (charter
.omx/tmp/codex_runs/eh1_prompt.md) → EUREKA_TABLE + CURE_TABLE vs our named negatives
(fp1 0.008305 floor · tk2 C1/C2 · TR1 arch · pose carriers · hp1 scoping · IX2TOK01 coder).
