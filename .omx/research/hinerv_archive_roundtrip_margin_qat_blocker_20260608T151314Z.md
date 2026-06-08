# HiNeRV ArchiveRoundtripMarginQAT — root-cause + precise handoff (NOT blocked; not-yet-wired)

UTC: 2026-06-08T15:13:14Z · Author: claude (takeover after codex weekly rate-limit, resets Jun 9 12:00 America/Chicago) · Authority: planning/control, NO promotion claim, NO score claim. Recovered current `main` HEAD `3dd771a91` (clean, synced); no active jobs; dispatch claims stale (May).

## Verified root cause (file:line evidence)

The HiNeRV hard-birth target margin survives live (13,488 wrong→target) and fakequant (12,183; retention 0.903) but COLLAPSES at HIV1 archive parse-back (2; retention 0.000148). The audit localized this correctly:
- `first_failed_surface = quantization_mismatch` (live→parseback audit)
- `first_failed_surface = latents_fine_quantizer_delta`, `next_operator = "bind ArchiveRoundtripQAT to the named tensor group and require archive-roundtrip scorer-margin survival before export"`, `quantizer_contract = "HIV1QuantizerMirror.latent_sections.v1"` (quantizer parity receipt)
- `archive_decoded_vs_parseback_max_abs_delta = 0.0` ⇒ **parse-back is byte-faithful to the archive; this is NOT a parse-back bug.**

The mechanism, confirmed by source inspection:
1. The HIV1 latent quantizer mirror **already exists and is MLX-ready**: `hiv1_quantize_dequantize_for_training(values, section_config)` at `src/tac/substrates/hi_nerv/archive.py:391-430`. For MLX arrays it returns `values + mx.stop_gradient(decoded - values)` (straight-through; archive-decoded forward, gradient through raw). Codec for latents is `int16_raw` (`LATENT_CODEC_RAW_INT16`).
2. The MLX birth actuator `fit_target_region_birth_from_segnet` (`src/tac/substrates/hi_nerv/mlx_renderer.py`) **never calls it** — grep for `hiv1_*`/`archive_roundtrip` in that file is EMPTY. The birth applies only `_receiver_uint8_roundtrip_ste_nhwc01` (RGB-uint8 STE, mlx_renderer.py:204). So the birth optimizes a margin robust to RGB-uint8 rounding but NOT to the int16 latent quantizer.
3. The diagnostic receipt's per-tensor `_training_roundtrip_delta` (archive_candidate.py:1455-1470) DOES run the HIV1 mirror on exported latents and found latents_fine moves ~0.005675 (2/4 elements). That tiny latent delta erases 13,486 of 13,488 birthed pixels.

## Interpretation: knife-edge margin

A ~0.0057 latent perturbation flipping ~99.98% of birthed pixels means the accepted birth sits exactly on SegNet argmax decision boundaries — a knife-edge solution. RGB-uint8 STE (the birth's only quantizer) has coarse enough quanta that 90% survived it, masking the fragility. The int16 latent quantizer is the real archive surface and it has no mercy for boundary-adjacent margins. The fix is not a bug-patch; it is **training the birth ON the quantized-latent manifold so the created margin carries slack against int16 rounding** (QAT, exactly as `next_operator` states).

## The two precise gaps

G1. **Birth forward does not route latents through HIV1.** Fix: in `fit_target_region_birth_from_segnet`, wrap the charged latent sections (start: `latents_fine`; codec `int16_raw`) through `hiv1_quantize_dequantize_for_training(..., section_config={"section_id":"latents_fine","latent_codec":"int16_raw"})` during the birth forward, behind an opt-in (`archive_roundtrip_qat=True`) so the no-pose / legacy paths stay byte-identical. The helper is MLX-native already, so this is a wrap at the latent read, not a rewrite.
G2. **The certificate `fakequant_vs_archive_decoded_max_abs_delta` is hardcoded `None`** at `archive_candidate.py:1210`. IMPORTANT: HEAD commit `3dd771a91 "Bind HiNeRV archive quantizer parity"` is the same author who just authored this function and left this field `None` **deliberately** — likely pending the MLX-side shadow that supplies the birth's actual fakequant latent forward (unavailable at the torch receipt-builder layer). **Do not overwrite this null speculatively.** Coordinate with codex (rate-limited until Jun 9) OR supply it from the MLX shadow surface where the birth's fakequant forward is in scope.

## Recommended next patch (one coherent landing, post-coordination)

1. Add `archive_roundtrip_qat: bool = False` + section list to `fit_target_region_birth_from_segnet`; when set, route `latents_fine` (then optionally `head_rgb_1`/`fine_injector`/feature grids) through the HIV1 mirror in the birth forward.
2. Emit a 4-surface target-margin survival row for the SAME `action_id`: `live` / `fakequant` / `archive_roundtrip_shadow` / `parseback`, each with `wrong_to_target`, `target_to_wrong`, `wrong_to_wrong`, `target_margin_{min,p10,p50,mean}`, pose-output delta, exact ΔS_nonrate, support/action/archive hashes.
3. Supply `fakequant_vs_archive_decoded_max_abs_delta` from the shadow (the birth's quantized-latent forward vs HIV1 archive decode) — the certificate the contract `fakequant_forward_must_equal_hiv1_archive_parseback_decoded_forward` names.
4. Gate (already partially in place via `429c50f36`/`cf4098fae`): parse-back birth survival requires a margin certificate AND positive target-margin slack; archive_roundtrip_shadow collapse must block launch; fakequant alone is insufficient.

## Tests to add
- birth-with-`archive_roundtrip_qat` produces a margin that survives the HIV1 latent mirror where the symmetric-fakequant birth did not (behavioral; a marker stub fails it).
- `fakequant_vs_archive_decoded_max_abs_delta` is non-null and equals the shadow-measured delta; quantization mismatch still emits `first_failed_surface=quantization_mismatch`.
- payload/program survival cannot clear scorer-effect survival; margin certificate required for parse-back birth survival.

## Why no code landed this turn
Faithful G1+G2 wiring touches the 700-line MLX birth method + a torch receipt function the HEAD-author just edited, with no GPT review available (rate-limited) and ~60 gates live. Per CLAUDE.md MVP-first + sister-coherence non-negotiables, a speculative solo edit to a hot file whose author deliberately left the target field null is higher-risk than the EV of one turn. This memo is the takeover's explicit secondary path: precise root cause + exact insertion points + certificate semantics, ready for a single clean landing next turn or by codex post-reset. No promotion/score claim.

## SNeRV lane (untouched, separate): per takeover, `decoder_payload.tub` is noncausal in current fixture ⇒ DROP_OR_REIFY; not mixed with this HiNeRV quantizer work.
