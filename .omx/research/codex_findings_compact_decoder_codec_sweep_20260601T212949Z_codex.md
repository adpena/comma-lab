# Compact decoder codec sweep landed

Codex landed a reusable compact decoder codec sweep API/CLI for PACT-NeRV-VQ and PACT-NeRV-Selector-V4 archives.

Purpose: re-pack an existing byte-closed compact archive through charged decoder-state codecs without retraining, then optionally receiver-prove each variant. This directly supports post-training rate-axis selection for compact carriers.

Entry points:
- API: `tac.substrates._shared.compact_decoder_codec_sweep.sweep_compact_decoder_codecs`
- CLI: `tools/sweep_compact_decoder_codecs.py`

Default codec portfolio:
- `portfolio_auto`
- `int8_mixed`
- `int8_scale_bundled`
- `int4_mixed`
- `int4_scale_bundled`
- `int2_mixed`
- `int2_scale_bundled`
- `fp16_enveloped`

Evidence from smoke:
- Source tiny PACT-VQ archive: `/Volumes/VertigoDataTier/pact/portfolio_auto_pact_vq_smoke_20260601T215000Z/pact_nerv_vq_mlx_training/archive.zip`
- Proof-backed sweep: `/Volumes/VertigoDataTier/pact/compact_decoder_codec_sweep_smoke_20260601T214300Z/compact_decoder_codec_sweep_report.json`
- Best tiny variant: `portfolio_auto`, selected parsed decoder codec `int8_mixed`/`brotli_q11`, receiver proof passed, still exact-blocked and scorer-replay-blocked.

Adversarial correction during landing: the first sweep row inherited `charged_bits_changed=false` from the false-authority template. Fixed and tested so variant rows now carry `charged_bits_changed=true`, `score_affecting_payload_changed=true`, and `exact_axis_score_affecting_adjudication_required=true` while still refusing score/exact authority.

Next use: when the active 600-pair PACT/VQ run emits `archive.zip`, run:

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/sweep_compact_decoder_codecs.py \
  --source-archive-zip /Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_score_bound_full600_2000ep_codex_20260601T194633Z/pact_nerv_vq_mlx_training/archive.zip \
  --output-dir /Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_score_bound_full600_codec_sweep_<UTC> \
  --repo-root /Volumes/VertigoDataTier/pact/worktrees/pact-main-codec-20260601 \
  --overwrite
```

No score authority is granted by this sweep. Promotion still requires full-video MLX scorer value replay and exact CPU/CUDA gates.
