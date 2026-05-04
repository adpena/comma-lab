# PR95/PR96 HNeRV Delta Intake - 2026-05-04

Scope: local read-mostly inspection only. No remote jobs, no GPU dispatch, no shared ledger edits.

## Evidence Inputs

- PR95 API/source/archive intake:
  - `experiments/results/leaderboard_intel_20260504_codex/pr95_api.json`
  - `experiments/results/leaderboard_intel_20260504_codex/pr95_files.json`
  - `experiments/results/public_pr95_intake_20260504_codex/archive.zip`
  - `experiments/results/public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.json`
  - `experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/profile_pr95_hnerv_muon_packing.json`
- PR96 API/source/archive intake:
  - `experiments/results/leaderboard_intel_20260504_codex/pr96_api.json`
  - `experiments/results/leaderboard_intel_20260504_codex/pr96_files.json`
  - `experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip`
- Current internal exact frontier:
  - `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.json`
  - `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/pre_submission_compliance_check.json`

## Score Comparison

| artifact | evidence status | archive bytes | seg | pose | rate component | recomputed score | delta vs PR85+STBM1BR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR85+STBM1BR | exact CUDA A++ local artifact | 229756 | 0.00057185 | 0.00018940 | 0.15298500 | 0.253690110294 | 0 |
| PR95 hnerv_muon | external/static PR body + local archive intake | 178417 | 0.00061212 | 0.00003494 | 0.11880056 | 0.198704801220 | -0.054985309074 |
| PR96 rem2_HNeRV | external/static PR body + local archive intake | 186631 | 0.00062231 | 0.00003675 | 0.12426992 | 0.205671211793 | -0.048018898501 |

PR95/PR96 reported formulas are materially better than our PR85+STBM1BR exact frontier, but they are not current internal A++ evidence until replayed through `archive.zip -> inflate.sh -> upstream/evaluate.py` with CUDA custody.

## HNeRV Runtime/Codec Anatomy

Both PR95 and PR96 use the same HNeRV-class decoder shape:

- 600 frame pairs, one 28-dimensional latent per pair.
- 229K-ish parameter decoder, counted locally as 228958 params.
- `latent -> Linear stem -> six PixelShuffle upsample blocks -> dilated refine -> two RGB heads`.
- Eval-space output is 384x512 RGB pairs, then bicubic upsample to camera raw size 874x1164.
- Runtime emits flat uint8 RGB raw frames, two frames per latent row.

PR95 `hnerv_muon`:

- PR title: `hnerv_muon submission (0.20)`, head `9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9`.
- Source adds a full reproducible training tree: `README.md`, `compress.sh`, `inflate.py`, `inflate.sh`, `src/model.py`, `src/codec.py`, optimizer/loss/stage code.
- Training curriculum in source: CE, tau-Softplus, smooth disagreement, QAT, L7+C1a, lambda sweep, sigma sweep, Muon finetune.
- Archive is a single stored ZIP member `0.bin`.
- `0.bin` layout:
  - `meta_json_brotli`: 80 bytes compressed, meta `{base_channels:36, eval_size:[384,512], latent_dim:28, n_pairs:600}`.
  - `decoder_state_int8_brotli`: 162349 bytes compressed, 230048 raw state-table bytes.
  - `latents_delta_uint8_brotli`: 15868 bytes compressed, 33720 raw latent payload bytes.
- Decoder codec: per-tensor symmetric INT8, zigzag, metadata table, brotli quality 11.
- Latent codec: per-dim uint8 min/max, first-order temporal delta, zigzag uint16, lo/hi byte streams, brotli. The latent hi-byte nonzero fraction is only 0.004345.
- Local lossless repack profile already found a byte-exact candidate at 178321 bytes, a 96-byte improvement, by compacting metadata and reordering decoder records.

PR96 `rem2_HNeRV`:

- PR title: `rem2_HNeRV submission (0.21)`, head `c692688419e8960d62201903c908f56b6c16c13d`.
- Source adds only `inflate.py` and `inflate.sh`; no compression/retraining script.
- Archive has three members:
  - `decoder.bin`: ZIP deflated, 169242 raw member bytes, 169272 compressed bytes.
  - `latents.bin`: ZIP deflated, 16920 raw member bytes, 16133 compressed bytes.
  - `p`: stored, 930 bytes, all zero bytes by local inspection, not referenced by `inflate.py`.
- Decoder member layout is a hybrid:
  - header `<IIIIB>` = brotli-table length, histogram length, AC metadata length, AC lengths length, histogram codec id.
  - small-tensor brotli table: 45658 bytes compressed, 53088 raw, 24 tensors, 52270 params.
  - arithmetic-coded large tensors: 4 tensors, 176688 params (`stem.weight`, `blocks.0.weight`, `blocks.1.weight`, `blocks.2.weight`).
  - histogram codec id is `2`, so the submitted archive uses brotli-compressed histograms.
- Latents are direct per-dim uint8 min/scale plus 600x28 bytes, without PR95's temporal delta split.

## PR95 vs PR96 Deltas

- PR96 is 8214 bytes larger than PR95: decoder area +6923 bytes, latents +265 bytes, unused `p` +930 bytes, ZIP/member overhead roughly +188 bytes.
- PR96 reported score is worse than PR95 by 0.006966410573:
  - seg contribution: +0.001019
  - pose contribution: +0.000478045
  - rate contribution: +0.005469365
- PR96's hybrid AC is not an improvement on its own artifact, despite PR95 source noting an earlier hybrid AC was about 217 bytes smaller on PR95's trained weights.
- PR96 removes PR95's source-level reproducibility story. It is useful as a decoder-packing sketch, not as a stronger archive family.

## Compliance Caveats

- PR95 and PR96 scores are PR-body/static intake evidence only in this pass. I did not run CUDA replay or dispatch anything.
- PR95's archive member timestamp is not normalized to 1980; our strict deterministic custody would rebuild before promotion.
- PR95 runtime/source is outside `archive.zip`, so exact claims must record runtime tree hash alongside archive SHA, as with our PR85+STBM1BR A++ artifact.
- PR96's unused all-zero `p` member is charged bytes and an avoidable payload-closure smell. It is not consumed by its `inflate.py`.
- PR96 imports `constriction` unconditionally. That works in this local Pact environment, but any exact replay must pin and record dependency provenance.
- Neither PR95 nor PR96 should be promoted, ranked as internal truth, or used for paper claims until exact CUDA replay verifies sample count, component distances, archive SHA, runtime tree hash, hardware, and recomputed formula.

## Top Moves To Beat PR95

1. Treat PR95 as the base family, not PR96. PR96 is a useful static sketch of per-tensor AC, but the submitted artifact is larger and worse on reported components.
2. First local implementation move: reproduce PR95 inflate byte-for-byte and rebuild a strict stored ZIP with deterministic metadata. The existing 96-byte PR95 repack is a low-risk first candidate, but it still needs exact replay before score language.
3. Add a deterministic per-tensor coder decision table for PR95 weights. Include PR95 pure brotli, PR96-style AC, and raw/brotli variants per tensor, selected by measured charged bytes with round-trip tests. Expected immediate upside is small, but it is contest-compliant and isolated.
4. Keep PR95's latent delta path, then replace part of the 600x28 free latent table with a charged low-rank temporal/ego-motion/foveation basis plus residual coefficients. This targets both bytes and generalization while preserving the HNeRV decoder contract.
5. Add a tiny charged residual atom stream over HNeRV output, not PR85 bundle syntax. Candidate atoms should be selected from exact component traces/hard-pair maps and must prove break-even score benefit per charged byte before dispatch.
6. Make the training objective coder-aware across decoder tensors and latents. PR95 already uses C1a for weight entropy; extend this into measured section-byte proxies for decoder, latent delta streams, and optional residual atoms.
7. Use PR85+STBM1BR only as a custody/guardrail template, not as the representation to incrementally polish toward PR95. The PR95 reported win is driven by a different representation: much lower pose distortion and 51,339 fewer bytes than PR85+STBM1BR, despite slightly worse seg.

