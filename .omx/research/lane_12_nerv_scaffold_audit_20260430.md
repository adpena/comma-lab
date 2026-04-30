# Lane 12 — NeRV mask codec — Phase A SCAFFOLD AUDIT (2026-04-30)

## Scope

Audit the existing Lane 12 (NeRV / Cool-Chic mask codec) scaffold ahead of Level 1 → Level 3 promotion.

## Files present at audit time

| File | Size | Purpose |
|---|---|---|
| `src/tac/nerv_mask_codec.py` | 16.2 KB / 440 LOC | codec module: `NeRVMaskCodec` (4-layer coordinate-MLP), `positional_encode`, `encode_nerv_codec` / `decode_nerv_codec`, `render_mask_logits`, byte-count helpers. NRV1 magic = `b"NRV1"`. |
| `src/tac/tests/test_nerv_mask_codec.py` | ~5 KB / 9 tests | encode/decode round-trip, shape/finite, byte-count vs raw fp16 baseline, no-silent-defaults, bad-arch rejection. All synthetic. |

## What's there (Level 1 SCAFFOLD)

- Coordinate MLP class with sin/cos positional encoding at log-spaced frequencies (NeRV / NeRF-style)
- Deterministic Xavier init via `torch.Generator().manual_seed(seed)`
- Self-describing NRV1 binary format (magic + 6×uint16 + uint64 payload-size + flat weights)
- fp16 round-trip path (preserves weights to fp16 precision, ~1e-3 max error)
- int8 path (scaffold: per-tensor symmetric quantization but DOES NOT preserve scale → decoder cannot reconstruct)
- Render-grid helper that runs the MLP over (T,H,W) coords in batched eval mode
- Byte-count accounting helpers (`raw_fp16_baseline_bytes`, `nerv_codec_bytes`)
- All public functions enforce explicit kwargs (no silent defaults — Check 81 STRICT compliant)

## What's MISSING (gap to Level 3 per `feedback_production_hardened_standard_definition_20260430.md`)

### Level 1 → Level 2 gap

1. **NO training loop.** Module is pure codec primitives — no `fit()` / `train()` method. Cannot learn from a real mask sequence today.
2. **NO EMA wired** — CLAUDE.md non-negotiable for every training path.
3. **NO eval_roundtrip simulation** — must apply `_eval_roundtrip_chain` at SegNet measurement to close proxy/auth gap.
4. **NO auth eval at end** — every training script must end with CUDA auth eval.
5. **int8 path is broken at decode** — encode quantizes per-tensor with implicit scale, decoder reads bytes as int8 floats with NO scale table. Round-trip cannot reproduce the float behavior; tests pass only because they check shape/parameter-count, not numerical equivalence after decode.
6. **NO archive integration.**
   - `submissions/robust_current/inflate_renderer.py` has dispatch for `AMRC` / `STCB` mask codecs but NOT `NRV1`.
   - `compress_archive.py` does not produce or pack NRV1 mask payloads.
   - No magic-byte routing into the inflate path.
7. **NO `experiments/train_nerv_mask.py`** standalone trainer.
8. **NO `scripts/remote_lane_nerv.sh`** dispatch script.
9. **NO `NERV_MASK_LANE_G_V3` profile** in `src/tac/profiles.py`.
10. **NO STRICT preflight check** for "NeRV training path uses EMA + eval_roundtrip".

### Level 2 → Level 3 gap

11. **NO real-archive empirical measurement** — never trained on the actual 1200-frame Lane G v3 mask sequence; AV1 421 KB benchmark untouched.
12. **NO contest-CUDA validation** — never run on Vast.ai 4090 / Modal A100 with `[contest-CUDA]` tag.
13. **NO 3-clean-pass adversarial review** — zero rounds run.
14. **NO memory entry** documenting empirical result.
15. **NO heartbeat / watchdog / harvest-path** in remote dispatch (because no remote dispatch).

## Risk classes flagged from CLAUDE.md FORBIDDEN PATTERNS / Council audits

- **`.round()` zero-gradient bug** (Council A § DARTS-S freeze): NOT triggered yet because there is no training loop, BUT when we add the training loop, the eval_roundtrip chain MUST use `Uint8STE.apply()` not bare `.round()`. The eval-time `argmax` for byte-saving is fine (no gradient needed), but if we ever back-prop through the simulated decoded mask we must use a STE.
- **MPS-derived strategic decision**: any auth-eval / kill / promote MUST be CUDA. Training scaffold currently device-agnostic (CPU); we will require CUDA at the trainer level.
- **eval_roundtrip default False trap**: trainer MUST default `eval_roundtrip=True` and refuse `False` (matches `SegMapTrainer.__init__` pattern).
- **EMA call shadowing live weights**: the canonical pattern is snapshot-restore at eval; trainer will follow `experiments/train_distill.py` pattern (and the new wire-ins from Council D).
- **Modal `.spawn()` artifact loss**: if dispatched on Modal, MUST register call_id + 24h harvest. Dispatch script will be canonical 4-stage Vast.ai-first per `scripts/remote_lane_ec_v2.sh` template.
- **Kaggle long slug fail**: not relevant (Vast.ai dispatch).
- **NVDEC bad-host roulette**: NeRV training does NOT need NVDEC (it operates on mask labels, not video frames), so we can skip the NVDEC probe — but we still need the probe for the auth-eval stage which decodes `videos/0.mkv`. Keep Stage 0 probe.

## Cross-refs

- Phase 2 Lane 12 design: `memory/project_phases_2_3_4_design_implementation_math_provenance_20260429.md` § "Lane 12: NeRV / Cool-Chic mask codec"
- Production-hardened standard: `memory/feedback_production_hardened_standard_definition_20260430.md`
- Canonical EMA: `tac.training.EMA` at `src/tac/training.py:391`
- Canonical eval-roundtrip chain: `src/tac/segmap_renderer.py:259` `_eval_roundtrip_chain`
- Magic-byte dispatch reference (AMRC/STCB pattern): `submissions/robust_current/inflate_renderer.py:1042-1051`
- Canonical remote-lane template: `scripts/remote_lane_ec_v2.sh`
- Subagent commit serializer: `tools/subagent_commit_serializer.py`

## Conclusion

Scaffold is **clean Level 1**: codec primitives exist, all 9 unit tests pass, no FORBIDDEN PATTERNS triggered. Path to Level 3 is well-defined and incremental. Estimated GPU: ~$0.60-$0.85 for training + auth eval on Vast.ai 4090 — under the $10 unattended cap.
