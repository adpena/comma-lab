# Code Review Tracker Report — 2026-04-11 17:35 UTC

## Summary

- **Total entities**: 1598
- **Reviewed**: 74 (5%)
- **Unreviewed**: 1195
- **Stale**: 0
- **Needs fix**: 329

## Priority Review Queue (by complexity)

| Entity | Type | Lines | Complexity | Status | File |
|--------|------|-------|------------|--------|------|
| `train` | function | 410 | 58 | needs_fix | train_renderer.py |
| `main` | function | 236 | 32 | unreviewed | train_postfilter_dilated_h64.py |
| `main` | function | 240 | 29 | unreviewed | train_postfilter_qat_ema.py |
| `TacLosslessArithmeticTests` | class | 478 | 27 | unreviewed | test_tac_lossless_arithmetic.py |
| `main` | function | 223 | 27 | unreviewed | train_postfilter_saliency.py |
| `main` | function | 267 | 26 | unreviewed | train_postfilter_canonical.py |
| `_run_lossless` | function | 222 | 24 | unreviewed | cli.py |
| `main` | function | 267 | 24 | unreviewed | cloud_h96_trainer.py |
| `main` | function | 214 | 23 | unreviewed | cloud_segnet_attack_h32_trainer.py |
| `main` | function | 196 | 23 | unreviewed | train_postfilter_v2.py |
| `parse_kaggle_status_text` | function | 68 | 22 | unreviewed | kaggle_status_sync.py |
| `decode_uint16_prev_symbol_stream` | function | 68 | 20 | unreviewed | frequency_coder.py |
| `main` | function | 194 | 19 | unreviewed | train_postfilter_segnet_attack.py |
| `_parse_header` | function | 52 | 18 | unreviewed | frequency_coder.py |
| `sweep` | function | 123 | 17 | unreviewed | crf_search.py |
| `main` | function | 168 | 17 | unreviewed | train_postfilter_cvar.py |
| `main` | function | 98 | 16 | unreviewed | train_postfilter_film_conditioned.py |
| `main` | function | 155 | 16 | unreviewed | train_postfilter_kalman.py |
| `main` | function | 152 | 16 | unreviewed | train_postfilter_uint8ste.py |
| `main` | function | 140 | 16 | unreviewed | trust_region_sweep.py |
| `main` | function | 189 | 15 | unreviewed | train_postfilter.py |
| `main` | function | 166 | 15 | unreviewed | train_postfilter_segaware.py |
| `DiffusionRenderer` | class | 316 | 15 | needs_fix | diffusion_renderer.py |
| `main` | function | 89 | 14 | unreviewed | monte_carlo_layer_scale_search.py |
| `train_variant` | function | 129 | 14 | unreviewed | run_saliency_sweep.py |
| `TacLosslessBaselineTests` | class | 396 | 14 | unreviewed | test_tac_lossless_baseline.py |
| `main` | function | 137 | 14 | unreviewed | train_postfilter_featmatch.py |
| `main` | function | 134 | 14 | unreviewed | train_postfilter_h32.py |
| `main` | function | 157 | 14 | unreviewed | train_postfilter_segnet_boundary.py |
| `train` | function | 213 | 14 | unreviewed | train_renderer_mlx.py |

## Recent Review Activity

- `tac.fp4_quantize::_quantize_block` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::_dequantize_block` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::_pack_indices_signs` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::_unpack_indices_signs` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::quantize_fp4` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::dequantize_fp4` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::FakeQuantFP4` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::FakeQuantFP4.forward` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::FakeQuantFP4.backward` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::fake_quant_fp4` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::FP4Parametrize` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::FP4Parametrize.__init__` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::FP4Parametrize.forward` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::QATRendererFP4` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::QATRendererFP4.__init__` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::QATRendererFP4._register_parametrizations` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::QATRendererFP4.remove_hooks` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::QATRendererFP4.forward` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::save_fp4` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
- `tac.fp4_quantize::load_fp4` — marked_needs_fix by council_5369_review (comprehensive_review_20260410)
