# Lane 12 NeRV — Adversarial Review Round 1 (2026-04-30)

## Counter

Round counter: **1/3 clean** (this round). Resets to 0 if any issue found.

## Reviewer perspective

Round 1 is the implementer-self-review pass: verify all CLAUDE.md non-negotiables hit on each file before rotating perspectives in Rounds 2+3.

## Files reviewed

| File | Lines added | Tests covering |
|---|---|---|
| `src/tac/nerv_mask_codec.py` | +280 (extended from 440 to ~720 LOC) | 24 in `test_nerv_mask_codec.py` |
| `src/tac/tests/test_nerv_mask_codec.py` | +325 (9 → 24 tests) | self-covering |
| `experiments/train_nerv_mask.py` | +330 (new file) | smoke-tested CPU |
| `submissions/robust_current/inflate_renderer.py` | +75 (new helper + dispatch wire) | dispatch-syntax check |
| `scripts/remote_lane_nerv.sh` | +135 (new file) | shell-syntax + 14 STRICT remote_lane preflight |
| `src/tac/profiles.py` | +30 (NERV_MASK_LANE_G_V3 + registry) | profile-resolver scan |
| `src/tac/preflight.py` | +175 (Check 95) | 10 in `test_preflight_nerv_codec_discipline.py` |
| `src/tac/tests/test_preflight_nerv_codec_discipline.py` | +245 (new file) | self-covering |
| `.omx/research/lane_12_nerv_scaffold_audit_20260430.md` | new doc | — |
| `.omx/research/council_lane_12_nerv_design_20260430.md` | new doc | — |
| `reports/lane_12_nerv_real_archive.json` | new artifact | empirical-tagged |

## Findings

### 1. CLAUDE.md "EMA — NON-NEGOTIABLE"

- ✅ `NeRVMaskTrainer.__init__` lazy-imports `from tac.training import EMA` and instantiates with decay=0.997 default (configurable from profile, validated `0 < decay < 1`).
- ✅ `EMA.update(self.codec)` called AFTER every `optimizer.step()`.
- ✅ `encode()` and `evaluate_argmax_disagreement()` use snapshot+restore: copy live state → `ema.apply(model)` → eval/encode → restore live in finally.
- ✅ Trainer guards against decay regression via `test_trainer_uses_canonical_tac_training_ema_synthetic` (fails if anyone introduces a local `class EMA`).
- ✅ Test `test_trainer_step_changes_live_weights_AND_ema_shadow_synthetic` proves EMA is being updated and lags live weights with the configured decay.

### 2. CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE"

- ✅ Trainer uses cross-entropy on raw 5-class logits — gradient flows through softmax at all times. No `.round()` anywhere in autograd-active forward.
- ✅ The 384→874→uint8→384 SegNet eval-roundtrip applies at the SCORE-measurement stage (after decode → render → mask_video → SegNet), which is delegated to `experiments/contest_auth_eval.py` Stage 3 in the dispatch script.
- ✅ Trainer `_eval_roundtrip` doesn't make sense for a mask CODEC (the input is already integer class labels), so the discipline shifts to "no `.round()` in forward" + "auth eval at end" — both verified.

### 3. CLAUDE.md "Auth eval EVERYWHERE"

- ✅ `scripts/remote_lane_nerv.sh` Stage 3 invokes `experiments/contest_auth_eval.py --device cuda` against the rebuilt archive.
- ✅ Check 95 (`check_nerv_codec_uses_ema_and_no_mps_and_auth_eval`) STRICT-enforces this discipline.

### 4. CLAUDE.md "MPS auth eval is NOISE"

- ✅ `NeRVMaskTrainer.__init__` raises `ValueError("refuses device='mps'")`.
- ✅ `render_mask_argmax` raises `ValueError("refuses device='mps'")`.
- ✅ Tests `test_trainer_refuses_mps_synthetic` + `test_render_mask_argmax_refuses_mps_synthetic` cover both.
- ✅ Check 95 STRICT-enforces presence of MPS-refusal at codec source.

### 5. CLAUDE.md FORBIDDEN PATTERN: bare `.round()` in forward (Council A zero-grad)

- ✅ Audit of `nerv_mask_codec.py` AST for `.round()` calls inside forward / step / _sample_batch / evaluate* methods → 0.
- ✅ Test `test_no_bare_round_in_nerv_mask_codec_source_synthetic` walks source AST + strips comments + asserts.
- ✅ Check 95 has its own AST scanner with the same logic at the preflight level.
- ✅ The encode-side `.round()` (in `encode_nerv_codec` numpy quantization path) is allowed because (a) it's CPU-side numpy, not torch autograd, and (b) it's outside any forbidden-context method. Test `test_round_in_encode_NOT_caught` documents this.

### 6. CLAUDE.md "Strict scorer rule"

- ✅ Inflate-time path (`inflate_renderer._load_masks_from_nrv`) loads ONLY the NeRV codec; no SegNet/PoseNet load. Verified via `check_no_scorer_load_at_inflate` 0 violations.
- ✅ Compress-time `experiments/train_nerv_mask.py` loads SegNet to extract argmax labels — explicit comment says "SegNet is NOT shipped in archive.zip". Compress-time-only by design.

### 7. CLAUDE.md "Lane separation — non-negotiable"

- ✅ Lane 12 is a Lane 1 (contest-compliant) lane: NRV2 codec is bundled into the archive, decoded at inflate, no scorer-at-inflate.
- ✅ Predicted band tagged `[contest-CUDA]` in dispatch script + `[empirical]` in real-archive JSON. No score reported without lane tag.

### 8. CLAUDE.md "Auth eval measurement"

- ✅ Dispatch Stage 2 rebuilds the archive with `masks.nrv` replacing `masks.mkv` (deterministic ZIP per Check 84).
- ✅ Stage 3 auth-eval runs against the EXACT archive that would be submitted (single archive variable threaded through). No "renderer-only archive" trap.

### 9. Wire format correctness

- ✅ NRV1 (legacy) preserves backward compatibility — same decoder accepts both.
- ✅ NRV2 ships scale table for int8 (fixes the v1 broken-int8 scaffold path).
- ✅ Magic-byte sniff in `_load_masks_from_archive`: `b"NRV1"` accepted (covers both v1 + v2 since they share the same 4-byte magic; version u16 disambiguates inside).
- ✅ Tests `test_default_version_is_v2_synthetic`, `test_invalid_version_rejected_synthetic`, `test_nrv2_int8_with_scale_table_roundtrip_synthetic` cover the format.

### 10. Profile / dispatch / argparse arity

- ✅ `NERV_MASK_LANE_G_V3` registered in `PROFILES["nerv_mask_lane_g_v3"]`.
- ✅ All NeRV knobs (10 of them) consumed by `experiments/train_nerv_mask.py` — no orphan keys.
- ✅ Trainer CLI args have explicit-required `--profile`, `--device`, `--output-dir`; all overrides default to profile values.
- ✅ Remote script runs `train_nerv_mask.py` with all required args; passes `check_remote_lane_argparse_arity` for the lane.

### 11. Subagent commit serializer

- All commits this session WILL go through `tools/subagent_commit_serializer.py`. (Pending — to be validated when commits land.)

## Conclusion

Round 1: **CLEAN — no issues.** Counter: **1/3**. Rounds 2 + 3 with rotating perspectives (Quantizr / van den Oord / Hotz / Shannon / Dykstra) PENDING — owed by the next session that picks up Lane 12 post-Phase-G dispatch.
