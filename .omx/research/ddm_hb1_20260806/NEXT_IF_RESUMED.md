# ddm_hb1 next if resumed

Status: QUEUED-WITH-FIRE-ORDER. HB1 did not produce an HPAC row on OUR payloads
because this local host has `torch.cuda.is_available() == false`,
`cuda_device_count == 0`, and no MPS backend. Do not cite PR130's retained HPAC
bytes as an OUR-payload result until stages below are run on the tq1c and/or GT
label caches and decoded exactly.

## Fire Order: PR130 HPAC On OUR Labels

Use the retained PR130 code under:

`/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code`

Required payloads:

| payload | source |
|---|---|
| tq1c parent argmax labels | `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy` |
| GT `lstars` labels | `/Volumes/VertigoDataTier/pact/ddm_ph1_lstars_u8.npy` or `experiments/results/mlx_fleet_gt_cache/gt_n600.npz:lstars` |

Stage 0: build a torch cache for each payload with key `seg`, shape
`[600,384,512]`, integer labels 0..4, and a SHA-256 receipt for raw bytes and
file bytes. Keep these under the SSD tier, not local scratch:

`/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/`

Stage 1: extract the PR130 patch-64 HPAC initialization if the target is the
PR130 continuation recipe:

```bash
PYTHONPATH=/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code \
  .venv/bin/python /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/extract_integer_hpac_archive.py \
  --archive /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/base/int5_delta_archive.zip \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --out /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/<payload>/hpac_p64_exact_from_archive.pt
```

Stage 2: run the PR130 self-compress recipe, replacing only `--cache`,
`--init`, `--save`, and `--out` with the HB1 payload paths:

```bash
PYTHONPATH=/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code \
  .venv/bin/python /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_hpac_self_compress.py \
  --cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/<payload>_seg_cache.pt \
  --init /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/<payload>/hpac_p64_exact_from_archive.pt \
  --epochs 60 --batch-size 8 --eval-batch-size 4 --eval-every 2 \
  --lr 0.003 --lr-exponent 0.0002 --lr-bits 0.01 --bit-eps 1e-6 \
  --rate-lambda 1.0 --qat-fraction 0.5 --init-bits 8.0 \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --target-mode raw \
  --seed 20260716 --device cuda \
  --save /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/<payload>/hpac_selfcompress_e60.pt \
  --out /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/reports/<payload>_hpac_selfcompress_e60.json
```

Stage 3: pack and exact-round-trip the self-compressed HPAC model:

```bash
PYTHONPATH=/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code \
  .venv/bin/python /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/pack_hpac_self_compress.py \
  --checkpoint /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/<payload>/hpac_selfcompress_e60.pt \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --weight-bound 127 --activation-bound 127 --weight-exponent-min -6 \
  --device cpu \
  --blob /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/<payload>/hpac.bin.xz \
  --report /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/reports/<payload>_hpac_pack.json
```

Stage 4: encode tokens, then decode with `--require-exact`. These two reports
are the HPAC row's decode-equality gate:

```bash
PYTHONPATH=/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code \
  .venv/bin/python /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/codec_hpac_integer.py \
  --checkpoint /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/<payload>/hpac_selfcompress_e60.pt \
  --cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/<payload>_seg_cache.pt \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --sparse --self-compress \
  --target-mode raw --frames 600 --device cuda \
  --tokens-out /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/<payload>/tokens.bin \
  --report /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/reports/<payload>_tokens_encode.json

PYTHONPATH=/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code \
  .venv/bin/python /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/codec_hpac_integer.py \
  --checkpoint /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/<payload>/hpac_selfcompress_e60.pt \
  --cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/<payload>_seg_cache.pt \
  --channels 64 --patch 64 --delta 2 --frame-dim 8 \
  --norm-mode none --activation relu --frame-scale \
  --weight-bound 127 --activation-bound 127 --weight-scales \
  --weight-exponent-min -6 --spm --sparse --self-compress \
  --target-mode raw --frames 600 --device cuda \
  --decode-from /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/<payload>/tokens.bin \
  --require-exact \
  --report /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/reports/<payload>_tokens_decode.json
```

Promote to `BYTE_RACE_TABLE.md` only after recording: token bytes, packed model
bytes, total bytes, raw and file SHA-256, encode wall-clock, decode wall-clock,
and exact decode equality.

## Stretch Rows

Rows 4 and 6 from the EH1 table were not run because they depend on a completed
OUR-payload HPAC model:

| stretch | status | resume gate |
|---|---|---|
| CPR1-style Huffman/Rice repack on low-rank carrier | queued | run after the base HPAC row has exact token decode and a packed model blob |
| bit-depth self-compression of HPAC model | queued | run after a baseline packed HPAC model exists; require max logit diff `0.0` before counting bytes |

