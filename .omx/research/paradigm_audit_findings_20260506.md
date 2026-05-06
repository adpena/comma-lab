# Paradigm audit findings — 2026-05-06

Six parallel adversarial audits launched on the full paradigm surface
(α / β / γ / δεζ / la-pose / cross-paradigm-integration). Findings
captured here so none are lost as fixes are batched.

## PARADIGM-β (sensitivity-aware everything) — 22 findings

### CRITICAL (4)

1. **`imp_sensitivity_weighted.py:278,309`** — `std_model.get_parameter()` on a sub-module attr returns the module, not the parameter. Will `AttributeError` on every real renderer with dotted layer names. Fix: `getattr(std_model, _qname_to_subset_attr(qname)).weight`.

2. **`owv3_sensitivity_weighted.py:86-95`** — Fixed `+32` byte constant in `_v1_raw_byte_estimate` inflates budget for partial-channel layers; over-allocates bits for low-sensitivity channels. Fix: gate `+32` on `len(quant_idx) == c_out` or remove.

3. **`neural_weight_codec_sensitivity.py:1002`** — `decode_with_per_block_codebook` tightly coupled to exact codec instance; no archive-embedded bucket-size fallback. Mismatch silently corrupts reconstruction. Fix: assert bucket_sizes match codec config before decode.

4. **`neural_weight_codec_sensitivity.py:630-633`** — `compute_per_block_sensitivity` formula is `|g|³` (gmag×ghess), not `g²` (Fisher). Inconsistent with docstring + theory. Fix: use `per_elem = ghess` alone.

### IMPORTANT (12) + Hygiene (6) — see audit transcript

## PARADIGM-δεζ (joint training + Self-Compress + MDL + TTO) — 14 findings

### CRITICAL (3)

1. **`self_compress.py:411-559`** — `train_self_compressing` has NO EMA. CLAUDE.md NON-NEGOTIABLE. Fix: instantiate EMA, update after optimizer.step, save EMA shadow.

2. **`self_compress.py:480-520`** — `train_self_compressing` has NO eval_roundtrip. CLAUDE.md NON-NEGOTIABLE. Fix: add STE uint8 round-trip before scorer.

3. **`self_compress.py:427`** — `device: str = "cpu"` default. CLAUDE.md FORBIDDEN PATTERN. Fix: require explicit device or raise on CPU without `allow_cpu=True`.

### Important (5) + Hygiene (6) — see audit transcript

## PARADIGM-α (mask payload overhaul) — 25 findings

### CRITICAL (5)

1. **`wavelet_mask_codec.py:96-128`** — Haar DWT round-trip not validated. NO test for `decode_wavelet_codec(encode_wavelet_codec(masks)) ≈ masks`. Codec is lossy with no error-rate documentation.

2. **`wavelet_mask_codec.py:330-338`** — `_decode_static_arithmetic` silently returns `sorted_keys[0]` if `scaled` falls outside every interval (corrupted/truncated payload). Fix: `else: raise ValueError`.

3. **`hnerv_wavelet_apply_transform.py:113-114`** — `REPACKABLE_SECTIONS[1]/[0]` index access fragile to tuple reorder. Fix: explicit string constants + assert membership.

4. **`hnerv_wavelet_sidechannel.py:104`** — Fixed output filename `hnerv_wavelet_sidechannel_candidate.zip` causes silent overwrite under parallel dispatch (CLAUDE.md race-mode rule violated). Fix: include `_slug(source_label)` in filename.

5. **`hnerv_wavelet_apply_transform.py:64`** — `strength_numerator=0` accepted, full brotli compress wasted before raise. Fix: validate `strength_numerator > 0` at gate.

### IMPORTANT (15) + Hygiene (5) — see audit transcript at output_file: a601ae1e4672e1c81

Top important: pipeline.py integration gap (no profile for any of 4 mask encoders) · preflight_all wiring gap · /tmp path guard missing · runtime_consumption_proof tautology · zero byte-delta wavelet not blocked · bare `except Exception` swallows KeyboardInterrupt.

## PARADIGM-γ (joint score-aware codec stack) — IN FLIGHT
## PARADIGM-γ (joint score-aware codec stack) — IN FLIGHT
## la-pose + raft + pose_gp — IN FLIGHT
## Cross-paradigm integration — IN FLIGHT
