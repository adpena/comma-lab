# C6 IBPS1 Aggressive MDL Byte-Patch Sweep - Codex - 2026-05-14

## Scope

Lane: `lane_c6_ibps1_mdl_byte_patch_20260514_codex`

Objective: convert Z1 scorer-conditional MDL signal into byte-closed C6
IBPS1 candidate archives with minimum wall-clock while preserving axis labels
and exact-eval custody.

Source archive:

`experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260514T144949Z__smoke__5ep_modal/harvested_artifacts/lane_substrate_c6_e4_mdl_ibps_results/output/archive.zip`

Source size: `224481` bytes.

## Tooling Landed

- `tools/mdl_scorer_conditional_ablation.py`
  - Added `--include-section` / `--exclude-section` sharding.
  - IBPS1 `encoder_blob` is excluded by default because it is
    `training_provenance_only`, not inflate-time frame-affecting payload.
  - Added `TierBResult.length_bytes` so `--skip-tier-a` shards no longer
    undercount section density as `n_samples`.
  - Added `--byte-offset` for targeted CPU confirmation of MPS-discovered
    offsets.
  - Added grammar alias normalization and fail-closed unknown grammar handling
    unless `--allow-generic-whole-blob` is explicit.
  - Added `pair_indices` persistence and `decision_grade`; MPS outputs are
    `decision_grade=false`.
  - Added `--scorer-batch-size` for memory-for-wall-clock tradeoff.

- `tools/build_ibps1_byte_patch_archive.py`
  - Deterministically patches `0.bin` bytes by `SECTION:OFFSET`.
  - Writes same-size candidate `archive.zip` plus manifest.
  - Makes no score claim; exact auth eval remains required.

Verification:

```bash
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
  src/tac/tests/test_build_ibps1_byte_patch_archive.py \
  src/tac/tests/test_mdl_scorer_conditional_ablation.py -q

.venv/bin/python -m py_compile \
  tools/build_ibps1_byte_patch_archive.py \
  tools/mdl_scorer_conditional_ablation.py
```

Result: `42 passed`.

## Wide MPS Classifier

Command shape:

```bash
PYTHONPATH=src:upstream .venv/bin/python tools/mdl_scorer_conditional_ablation.py \
  --archive "$C6_ARCH" \
  --archive-name c6_dec_mps_s<seed> \
  --grammar ibps1 \
  --upstream-dir upstream \
  --output-dir experiments/results/mdl_ablation_c6_decoder_mps_s<seed>_20260514_codex \
  --device mps \
  --pair-samples 16 \
  --byte-samples 128 \
  --significance-threshold 1e-3 \
  --seed <3001..3008> \
  --skip-tier-a --skip-tier-c \
  --include-section decoder_blob
```

Artifacts:

- `experiments/results/mdl_ablation_c6_decoder_mps_s3001_20260514_codex/`
- `experiments/results/mdl_ablation_c6_decoder_mps_s3002_20260514_codex/`
- `experiments/results/mdl_ablation_c6_decoder_mps_s3003_20260514_codex/`
- `experiments/results/mdl_ablation_c6_decoder_mps_s3004_20260514_codex/`
- `experiments/results/mdl_ablation_c6_decoder_mps_s3005_20260514_codex/`
- `experiments/results/mdl_ablation_c6_decoder_mps_s3006_20260514_codex/`
- `experiments/results/mdl_ablation_c6_decoder_mps_s3007_20260514_codex/`
- `experiments/results/mdl_ablation_c6_decoder_mps_s3008_20260514_codex/`

Summary:

- `1024` intended decoder-byte probes, `583` successful decode/scorer samples
  in first harvest artifact after parser failures were excluded.
- `137` unique decoder offsets with MPS delta `<= -0.02`.
- `163` unique decoder offsets with MPS delta `<= -0.005`.
- Decoder density consistently landed near `0.57-0.60` on MPS screening.
- Latent screening stayed near-dead:
  - `c6_lat_mps_s4001`: `3/128`, density `337.5` bytes.
  - `c6_lat_mps_s4002`: `1/128`, density `112.5` bytes.

Evidence label: `[MPS screening]`, `decision_grade=false`.

## CPU24 Confirmation

Top `36` MPS-negative decoder offsets were split into three 12-offset CPU
confirmation chunks:

- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/cpu_confirm_aa/`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/cpu_confirm_ab/`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/cpu_confirm_ac/`

Result:

- `36/36` CPU24 confirmed significant.
- `34/36` CPU24 offsets had delta `<= -0.05`.
- Best CPU24 offsets:
  - `34858`: `-0.5599296005`
  - `86325`: `-0.5221631101`
  - `65756`: `-0.4855240046`
  - `1395`: `-0.4760075473`
  - `40510`: `-0.4633876969`

Evidence label: `[macOS-CPU advisory]`, `decision_grade=true` for local CPU
MDL deltas; not `[contest-CPU]` and not `[contest-CUDA]`.

## MPS600 Full-Pair Screening

MPS600 replay of MPS-top-5:

Artifact: `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/mps600_top5/`

Offsets and deltas:

- `21093`: `-0.3983110498`
- `34576`: `-0.3162075442`
- `28936`: `-0.3128413830`
- `43437`: `-0.3070578217`
- `15987`: `-0.1596692550`

MPS600 replay of CPU24-top-5:

Artifact: `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/mps600_cpu24_top5/`

Offsets and deltas:

- `86325`: `-0.4785659327`
- `34858`: `-0.4263233771`
- `65756`: `-0.4035141445`
- `1395`: `-0.3247090363`
- `40510`: `-0.3071851146`

Evidence label: `[MPS screening]`, `decision_grade=false`.

## Byte-Closed Candidate Archives

Single-offset and top-5 stacked candidates were built. All listed candidates
are `224481` bytes, matching source archive size.

MPS-top-5 candidates:

- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates/decoder_15987/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates/decoder_21093/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates/decoder_28936/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates/decoder_34576/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates/decoder_43437/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates/decoder_top5_stack/archive.zip`

CPU24-top-5 candidates:

- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_1395/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_34858/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_40510/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_65756/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_86325/archive.zip`
- `experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_cpu24_top5/decoder_top5_stack/archive.zip`

Decode smoke:

`experiments/results/c6_ibps1_mdl_aggressive_20260514_codex/patch_candidates_decode_smoke_mps/`

All MPS-top-5 candidates, including `decoder_top5_stack`, decoded and scored
on an 8-pair MPS smoke. This is not a score claim.

## Memory / Wall-Clock Control

- Initial monolithic Z1 jobs were killed because they produced no partial
  result files after long CPU runtime.
- MPS classifier fan-out used approximately `1.1-1.2 GB RSS` per 16-pair
  process.
- CPU24 confirmation used approximately `3.4-4.1 GB RSS` per process.
- CPU600 triple exceeded safe free-page margin; lower-priority CPU600 offsets
  were terminated and deferred. Kept `34858` CPU600 plus MPS600 top-5.

## Current Decision State

Confirmed:

- C6 5ep archive is decoder-dominated.
- Latent byte stream is nearly inert under byte flips at this stage.
- Many decoder byte flips are score-lowering on MPS screening and CPU24 local
  advisory confirmation.
- Byte-closed same-size patch candidates now exist.

Still required before dispatch/rank change:

- CPU600 result for `decoder_blob:34858`.
- Exact auth eval on one or more byte-closed candidate packets after lane claim.
