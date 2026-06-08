# HiNeRV parse-back collapse — falsification chain + build-path localization (v7/v8/v9 smokes)

UTC: 2026-06-08T220000Z · claude (solo) · planning, no score claim. Authority: every
number below is `[macOS-MLX research-signal]` / counterfactual; NONE is a contest score.
The only authority terms are exact `100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489`
on the submitted archive bytes (not measured here).

## The phenomenon
A target-region SegNet birth wins ~12k region pixels live/fakequant; the runner's
authoritative selected-archive parse-back wins only **2**. This tranche diagnosed WHY,
letting the artifacts decide. Five hypotheses were tested; four were falsified.

## Hypothesis ledger (artifact-decided)

| # | hypothesis | status | deciding artifact |
|---|---|---|---|
| H1 | a single/pairwise DECODER SECTION collapses L | FALSIFIED (under int8) | v8 guilt sweep: every section ret_vs_fakequant=1.0, no collapse — but used wrong codec (int16_raw), so SUPERSEDED by H-codec |
| H2 | the LATENT codec int8_brotli_q11 collapses L | FALSIFIED | v9 grid: latent axis drop = 1376 (~11%, mild); int8_brotli_q11 keeps 10781/12157 |
| H3 | the DECODER codec int4_mixed collapses L | TRUE in grid, but NOT the selected cause | v9 grid: int4_mixed → 17 wins (catastrophic). BUT selected backend payload ~9197 ≈ int8 (9115), NOT int4 (5771) → selected is int8, so int4 is not what ships |
| H4 | EMA-vs-live selection collapses L | FALSIFIED | runner ema AND live candidate parsebacks BOTH = 2 |
| H5 | the runner's ARCHIVE BUILD PATH (sidecar wrap / coder-aware-qat export) collapses L | **LEADING** | same live weights + same int8 codec: my counterfactual pack (no sidecar, 9115B) = 10781; runner archive (8127B sidecar, 16326B zip) = 2 |

## v9 controlled codec grid (same compact live export, ONE builder — apples-to-apples)
```
decoder/latent                wins    retention_vs_live   payload_bytes
int8_mixed / int16_raw        12157   0.998               9115   (faithful)
int8_mixed / int8_brotli_q11  10781   0.885               9123   (latent mild: -1376)
int4_mixed / int16_raw           17   0.0014              5771   (decoder catastrophic: -12140)
int4_mixed / int8_brotli_q11     17   0.0014              5779
```
collapse_axis = decoder_codec (decoder drop 12140 >> latent drop 1376).

## Selected-archive custody (v9, sha a30859e1, 16325B zip, candidate ema)
- effective_decoder_codec = int8_mixed (telemetry); backend payload ~9197 ≈ int8, NOT int4.
- latent blob is tiny (1 pair); int16_raw vs int8_brotli differ by 8 bytes — latent codec is not the byte lever here.
- payload 17324 = int8 backend (~9197) + base64 target-region action sidecar (~8127).
- ema parseback = 2; live parseback = 2 (EMA non-causal).
- target_margin over region: p10 -1.78, p50 -1.15 (birth is well below the SegNet wall at parse-back).

## The apples-to-apples bug this tranche caught (twice)
1. My decoder-section sweep (v8) re-packed with int8_mixed+int16_raw → 12157 wins, contradicting the
   runner's authoritative 2. Cross-check against the runner's parseback row exposed it.
2. The PRIOR `archive_roundtrip_shadow` "latents_fine not the cause" finding used int16_raw, not the
   real int8_brotli_q11 — but v9 shows even int8_brotli_q11 is mild, so latents remain not-the-cause.

The structural lesson: a section/codec shadow is only evidence about the SELECTED archive if it packs the
SAME build path the runner ships. Counterfactual packs are alternative-codec candidates, not selected-archive verdicts.

## Decisive next probe (cheap, precise — NOT done this tranche)
Render the runner's live/archive.zip **backend (unwrapped, no sidecar)** vs **with-sidecar**, counting region
target wins with `hi_nerv_hard_region_miner_inputs.npz::target_labels_bhw` + SegNet:
- if backend = 2 → the export build path (coder-aware-qat / pack_archive at selection) collapses it →
  compare against my counterfactual int8 pack (10781) byte-for-byte to localize the transform.
- if backend ~10781 but with-sidecar = 2 → the SIDECAR WRAP is inverting the birth (critical Rule #8 bug).
File/function: `tac.substrates.hi_nerv.inflate.build_model_from_archive` (wraps sidecar via
`wrap_model_with_target_region_actions`); `tac.substrates.hi_nerv.archive_candidate.pack_archive_from_exported_state_dict`
(my counterfactual pack) vs the runner's export/selection pack.

## DO NOT
- name a guilty decoder section (H1 superseded; the selected codec is int8, faithful in the grid).
- implement section QAT or decoder-codec QAT (int8 is faithful; int4 isn't what ships; the gap is build-path).
- treat the v9 grid cells as selected-archive evidence — they are counterfactual same-export packs.
- call int16_raw "the fix" — its win-preservation is on a DIFFERENT (sidecar-less) build path than ships.
