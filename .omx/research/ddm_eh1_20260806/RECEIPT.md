# ddm_eh1 PR130 / PR86 eureka harvest receipt

Date: 2026-08-06
Axis: `[offline archive/source forensics; no scorer slot]`
Score claim: false
Network: not used
Scorer: not used
Pointer moved: false

## Boundary

This arm harvested the frozen PR130/PR86 local assets under
`/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/` and wrote only the
four requested report files in this directory. It did not fetch network, did not
edit `upstream/`, did not touch the protected files named by the common
contract, did not run n600 scorer work, and did not alter the staged index.

The live own-vehicle frontier remains the current hot-state row:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. The borrowed contest
pointer remains unchanged.

## RECALL EVIDENCE

- Governing files read: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`,
  the EH1 charter, and the common contract it references.
- Prior public-intake artifacts consumed instead of duplicated:
  `.omx/research/pr86_pr130_fullstack_intake_20260728.md`,
  `.omx/research/public_pr129_132_intake_20260725.md`, and
  `.omx/research/pr130_eureka_intake_acquisition_20260806.md`.
- Primary PR130 source read from
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/`.
  Important citations: `README.md:32-48`, `recipe/TRAINING.md:21-38`,
  `recipe/TRAINING.md:57-86`, `recipe/PROVENANCE.md:5-27`,
  `LINEAGE_AND_CITATIONS.md:13-24`, `LINEAGE_AND_CITATIONS.md:39-54`,
  `scripts/e2e.py:300-1416`, and `code/inflate.py:119-690`.
- Archive forensics read real local archive bytes:
  `releases/cpr1/archive.zip` and `releases/landslide/archive.zip`.
  Section parsing used `zipfile`, `lzma`, `struct`, and the PR130 repo's own
  `carrier_codec` / `inflate` helpers.
- Local negative/cure surfaces read: `ddm_fp1_class_field_projection_20260731.md`,
  `ddm_tk2_20260806/RECEIPT.md`, `ddm_tb1_renderer_build_20260728.md`,
  `ddm_pfs1_posefield_and_recompose_20260729.md`,
  `ddm_sc1_seeded_scene_carrier_20260728.md`,
  `ddm_wd1_pose_wiring_falsified_and_correction_minimum_scale_20260802.md`,
  `ddm_hp1_20260806/RECEIPT.md`,
  `ddm_ix2_renderer_split_and_decoder_20260802.md`,
  `ddm_tw1_token_waterfill_state_dependence_20260801.md`,
  `ddm_tz1_token_sweep_rate_attack_20260804.md`, and
  `scorer_batch_20260804.md`.
- Ecosystem residue read: `pr130_comments.txt`, `pr86_comments.txt`,
  `jas0xf_repos.json`, and local `fork_clone` git logs. Blobless `fork_clone`
  stat inspection attempted to fetch promisor objects and failed due no network,
  so only local log facts are used.

## Custody

| object | bytes | sha256 / id | note |
|---|---:|---|---|
| PR130 CPR1 archive | 191,052 | `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd` | one stored member `p`, 190,952 B |
| PR130 landslide archive | 194,380 | `f4457de09a6e69c8cd29e886a84705462a8c77dc6978020b11dff52e661a1451` | one stored member `p`, 194,280 B |
| reproduction repo | - | `2f94596bb0136d342254022a5c9584756eae0468` | clean local checkout |
| fork clone local head | - | `c88661d3e22926b14fac01bb8d201129d33786ba` | local log only |
| frozen PR130 PR branch/tag | - | `9a77b6ad660d6310ab54436757ab07a9bbc9f3e1` | from local fork log and PR comment |

The official PR130 comment row reports `[contest-CUDA]` d_seg `0.00029660`,
d_pose `0.00002331`, 191,052 B, recomputed `S = 0.172141297492`. A PR comment
also reports an independent RTX 5070 precision diagnostic on the same CPR1
bytes with d_seg `0.0002959781268145889`, d_pose `1.94819294847548e-05`,
191,052 B, recomputed `S = 0.170769265655`. Both are external/public rows, not
our own-vehicle rows.

## Pipeline Extraction

| subsystem | PR130 mechanism | source citations |
|---|---|---|
| End-to-end graph | raw video to targets, semantic renderer, pose basis/coefs, integer HPAC, CPR1, official eval | `README.md:32-48`; `scripts/e2e.py:300-1416` |
| Stage groups | 49 stages: targets, semantic training, pose pilot/full carrier, exact searches, HPAC, self-compress, pack/eval | `recipe/TRAINING.md:21-38`; `scripts/e2e.py:300-1416` |
| Selected boundaries | early stop records for pose hard-mining, coefficient rescue, basis adaptation, CPU polish, HPAC | `recipe/TRAINING.md:40-55`; `scripts/e2e.py:490-905` |
| Semantic architecture | width-96, frame embedding, token embedding, coord mix, 4 residual blocks, RGB head | `code/inflate.py:119-168`; `code/semantic_renderer_oracle.py:79-161` |
| Semantic loss/selection | CE, softplus margin, expected-flip loss, exact uint8/R path, best exact checkpoint | `code/semantic_renderer_oracle.py:164-204`; `code/train_semantic_full.py:26-180` |
| QAT | fake quantization, exact-path render, packed-size tracking, best quantized checkpoint | `code/train_semantic_quantized.py:30-46`; `code/train_semantic_quantized.py:90-142`; `code/train_semantic_quantized.py:245-340` |
| Pose carrier | neutral-gray low-rank 12-direction 24x32 basis, 600x12 int12 coefficients, QAT and hard mining | `code/train_pose_carrier_full.py:26-455`; `code/learned_pose_carrier_oracle.py:64-264` |
| Exact int12 search | worst-row greedy coordinate search and code refinement through PoseNet | `code/search_pose_coeff_cpu.py:1-190`; `code/refine_pose_coeff_codes.py:161-295` |
| HPAC | integer-lattice masked convs, patch-group masks, SPM, previous-frame context, logits / 8 | `code/hpac_integer.py:78-398`; `code/hpac_integer_sparse.py:66-180`; `code/inflate.py:277-598` |
| HPAC self-compress | counted per-channel bit depths and joint model/token rate objective | `code/hpac_self_compress.py:45-132`; `code/train_hpac_self_compress.py:139-211` |
| Archive pack/repack | deterministic one-member zip, XZ model bundle, CPR1 compact Huffman/Rice carrier | `code/build_submission_archive.py:33-139`; `code/repack_carrier.py:96-280`; `code/carrier_codec.py:13-400` |

The key structural difference versus our flat/template-paint negatives is that PR130 trains a
source-forward RGB receiver through the exact SegNet path. It does not ask SegNet to read a flat
5-class cartoon plane.

## Archive Forensics

| section | CPR1 bytes | landslide bytes | equality / delta |
|---|---:|---:|---|
| archive.zip | 191,052 | 194,380 | CPR1 saves 3,328 B |
| stored member `p` | 190,952 | 194,280 | one member in both |
| XZ model bundle | 73,968 | 77,296 | CPR1 saves 3,328 B |
| raw model bundle | 83,493 | 88,615 | CPR1 saves 5,122 raw B |
| semantic renderer | 40,252 | 40,252 | equal |
| carrier section | 23,054 | 28,176 | compact CPR1 carrier saves 5,122 raw B |
| HPAC raw model | 20,179 | 20,179 | equal |
| token stream | 116,980 | 116,980 | equal |

Semantic, HPAC raw model, and token stream are identical between landslide and
CPR1. The CPR1 gain is a lossless carrier/model-bundle repack, not a new
distortion row.

Distribution facts from the CPR1 archive:

| field | value |
|---|---|
| token bpp | `0.007933213975694445` over 600 x 384 x 512 semantic labels |
| semantic int4 q_count | 63,936 values, entropy `3.5345` bits/symbol, zero fraction `0.14535` |
| carrier magic | `CPR1`, prefix 152 B, basis payload 13,017 B, coefficient payload 9,885 B |
| basis codes | 27,648 symbols, range -15..15, entropy `3.7451` bits/symbol |
| coefficient codes | 7,200 symbols, range 0..4095, entropy `10.5847` bits/symbol |
| HPAC bit-depth hist | `{0:6, 1:2, 2:7, 3:11, 4:38, 5:59, 6:110, 7:275, 8:9}` |
| HPAC codec receipt | token bytes 116,980, ideal bpp `0.007788886871760884`, exact pack verified |
| HPAC compressed model | 15,164 B compressed from 20,179 B raw, `max_logit_diff=0.0` |

## Ecosystem Residue

- PR130 comments contain the official `semantic-pose-HPAC_CPR1` eval row and an
  independent CPR1 reproduction. The PR130 author also froze a reproducibility
  repo at `2f94596bb0136d342254022a5c9584756eae0468`.
- PR86 comments contain the older official row: d_seg `0.00067815`, d_pose
  `0.00045701`, 207,579 B, rounded final score `0.27`; comments also record
  moving heavy artifacts to `jas0xf/comma-anr-supplementary`.
- `jas0xf_repos.json` is a local snapshot with 30 repos. Relevant residues:
  `comma-anr-supplementary` pushed 2026-05-04, forked
  `comma_video_compression_challenge` pushed 2026-05-04, and later unrelated
  repos `claude-unstuck`, `ggm-tree`, `PA_Agent`, plus FPGA coursework repos.
- `fork_clone` local log shows today's visible maintenance commits on master:
  `c88661d3` remove workflow triggers, `77a61a5` remove one-shot CPR1 publisher,
  `b28af11` temporarily publish locked CPR1 release, and branch/tag
  `semantic-pose-HPAC_CPR1` at `9a77b6a`. Blob stats are unavailable offline.

## What Was Not Measured Here

No new `d_seg`, `d_pose`, exact score, CPU/CUDA contest row, or own-vehicle
frontier movement was measured. All `|Delta S|` values in the companion tables
are projections or source-row comparisons unless explicitly labeled as measured
archive-byte deltas.
