# G86 — G78 reproducible full-n600 scorer-state materialization

**UTC:** 2026-07-27T05:26:10Z  
**Lane:** `lane_g78_batch16_margin_base_scorer_cache_materializer_20260726`  
**Job:** `g78_batch16_margin_base_n600_r4_20260727`  
**Verdict:** `COMPLETE_REPRODUCIBLE_ENCODER_ONLY_CACHE`; no candidate, score, promotion, or pointer claim.

## Outcome

The governed G78 materializer completed the exact chronological n600 batch-16
SegNet target/described state cache and then reproduced the same sealed aggregate
on a second resume run.

| Evidence | Exact value |
|---|---:|
| aggregate path | `/Volumes/VertigoDataTier/pact/taskspace_batch16_margin_base_scorer_cache_n600_20260726_r4/aggregate_receipt.json` |
| aggregate bytes | 42,681 |
| aggregate file SHA-256 | `f2422488fb8a3158d191b9a5fbc1150ce6e24a9c6bd7cace80b57845f86f7fb4` |
| aggregate sealed self SHA-256 | `fc6a2de90de0c8f8037c88bc4ae9853ab3bbffb9cb7f5a42b0e849098a15f3b7` |
| pair count | 600 |
| class count | 5 |
| scorer geometry | chronological max-16 batches at 384×512 |
| batch checkpoints | 38 |
| stage checkpoints | 5 × 120 pairs |
| stage digest chain SHA-256 | `beb7cbf4abb58db030d941ac072f9dbf08002adcffd4fcb890ba946937944fc1` |
| first materialization | 492.37 s, 6,900 MiB peak RSS, exit 0 |
| exact resume replay | 113.96 s, 2,580 MiB peak RSS, exit 0 |
| evidence axis | `[macOS-CPU encoder-only batch16 frozen-scorer evidence]` |

An independent `--status` reopen rehashed and validated the aggregate after the
first run. The resume run regenerated the identical aggregate file SHA and sealed
self SHA; it did not create a divergent lineage.

## Custody closed

The aggregate binds all of the following rather than trusting path names:

- fresh own-lineage V15 semantic archive: 133,941 bytes,
  SHA-256 `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`;
- owned G46 target labels, SHA-256
  `6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85`;
- fresh G51 Y0/Y1 scorer-plane aggregate self SHA-256
  `4363827c2aeb613916029d8bacde8aeb4ded961c4d1ca297310a1e53e204619c`;
- exact V15 camera identity over all 38 batches, digest-chain SHA-256
  `5d502c1eafe0bd6b3a3e8ea323b02a66573f51939e0a21ebde6e592e04141d7c`;
- 632 sealed inputs and the 558-file transitive V15/scorer runtime closure.

`target_argmax_equal_owned_g46_all_batches` is true, chronological coverage is
exactly `[0,600)`, and global batch geometry remains intact across every
120-pair stage boundary.

## Blocker disposition

Closed by this aggregate:

1. `G72_FRESH_BATCH16_TARGET_MARGIN_CUSTODY_OWED`
2. `G72_FRESH_V15_CAMERA_R_BATCH16_BASE_SCORER_STAGE_CACHE_OWED`

Still open, unchanged:

1. `G72_G49_ROLE_PRESERVING_ANALYTIC_WIRE_ABI_OWED`
2. `G72_V15_ROLE_AWARE_PREPAINT_ANALYTIC_DECODER_PROOF_OWED`
3. `G72_FRESH_POSE_TARGET_AUTHORITY_OR_EXACT_UPSTREAM_FINAL_REPLAY_OWED`
4. `G72_FIVE_STAGE_EXACT_WHOLE_OBJECT_JOINT_ADMISSION_OWED`

The dense target/described fields and scorer weights are encoder-only and
explicitly forbidden from candidate payload. This result supplies exact compiler
state; it is not itself a compact representation.

## Triality and next consumer

- **DSL:** the typed G78 config fixes batch size, stage size, scorer surface,
  source custody, storage tier, and strict reopen contract.
- **DAG:** G46 labels + G51 Y0/Y1 + fresh V15 realization → exact live-R
  batch-16 scorer fields → immutable batch checkpoints → five stage
  checkpoints → sealed aggregate → G72/G87 compiler adapter.
- **Equations:** each cached margin is
  `winner_logit - max(nonwinner_logits)` under the owned target winner; archive
  admission remains the exact nonlinear contest score on a complete n600 row,
  never a component threshold or sum of local proxy deltas.

The next legal use is an encoder-side G72/G87 selected-preimage compiler that
consumes these sealed fields and emits only counted compact operands. The next
authority surface remains public `inflate.sh` parse-back plus exact
`upstream/evaluate.py`; the effective frontier pointer remains 0.172.

## STORES CONSULTED

- `CLAUDE.md`
- `AGENTS.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g78_batch16_margin_base_scorer_cache_materializer_20260726.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/codex_findings_g81_g78_batch16_margin_base_scorer_cache_adversarial_launch_review_20260727_codex.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/codex_findings_g84_g78_launch_reentry_adversarial_review_20260727_codex.md`
- `/Volumes/VertigoDataTier/pact/taskspace_batch16_margin_base_scorer_cache_n600_20260726_r4/00_preflight_receipt.json`
- `/Volumes/VertigoDataTier/pact/taskspace_batch16_margin_base_scorer_cache_n600_20260726_r4/aggregate_receipt.json`
- `.omx/state/active_lane_dispatch_claims.md`

