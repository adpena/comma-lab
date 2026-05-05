# Lane J-NWC Landed (Neural Weight Compression end-to-end)

**Date**: 2026-04-29
**Commit**: 12b43507 ("Lane J-NWC: full neural weight codec end-to-end (producer + consumer + dispatch + tests)")
**Status**: IMPLEMENTATION COMPLETE — dispatch script ready, no GPU launched yet

## What landed

End-to-end Lane J-NWC pipeline closing the gap surfaced by the bidirectional magic-byte test:
1. **Inflate dispatch** — `submissions/robust_current/inflate_renderer.py` `_load_renderer` now routes `b"NWC1"` magic to `tac.renderer_export.load_neural_compressed_checkpoint` (mirrors SCv1/OMG1/CCh1/SZv1 lane policy: tac wheel required, no inline fallback).
2. **Pipeline producer** — `experiments/pipeline.py` `step_compress_weights` adds `mode == "nwc"` branch + new `weight_codec_path` config field. Arch-header gate analogue: when codec checkpoint missing, falls back to FP4 with WARN (instead of silently shipping a stub).
3. **Remote dispatch script** — `scripts/remote_lane_nwc.sh`: 4-stage canonical pipeline (NVDEC probe / corpus verify / codec train / archive build + CUDA auth eval). Predicted band `[0.95, 1.30]` `[prediction]`. Heartbeat to `/tmp/heartbeat_lane_j_nwc.log`, provenance.json, dead-flag scanner, deterministic-zip archive build, `[contest-CUDA]` tag at completion. Passes all 14 STRICT remote_lane preflight checks.
4. **Tests** — `src/tac/tests/test_lane_j_nwc.py`: 8 new tests (magic-byte, round-trip, inflate dispatch, determinism, arch-header gate, umbrella bidirectional test).

## Codec architecture (refresh)

`src/tac/neural_weight_codec.py` (already existed — Codex landed it earlier):
- VQ-VAE-style codec, ~16K codec params
- `block_size=16` (16 weight elements per code → 1 byte uint8 index + 2 byte fp16 scale = 3 bytes per 16 elements ≈ 1.5 bits/weight nominal)
- `codebook_size=64`, `latent_dim=16`, `hidden=64`
- Encoder/decoder are 3-layer MLPs with GELU
- Codec weights are bundled INSIDE the NWC1 binary, so the inflate-side loader is fully self-contained — no external codec asset required at inflate time.

NWC1 binary layout (from `renderer_export.py:4284-4298`):
```
magic            (4B = b"NWC1")
header_len       (4B uint32 LE)
header_json      (UTF-8, JSON: arch config + codec_config + per-tensor metadata)
codec_state_blob_len     (4B uint32)
codec_state_blob         (torch.save bytes of codec state_dict)
per_tensor_blobs         — sequence of (4B blob_len, blob_bytes)
```

## Predicted byte savings (vs FP4) — `[prediction]`

On a synthetic ~3.3K-param TinyRenderer test (CI tests):
- NWC1 binary size: 39,641 bytes = 95.66 bits/param including codec
  (codec dominates at this small scale — 16K codec params @ fp32 ≈ 64KB)
- For Lane A's 88K-param renderer:
  - Raw weights: 88K × 1.5 bits = ~16.5 KB
  - Plus amortized codec: +64 KB
  - Total predicted: ~80 KB
  - FP4A baseline on same weights: 88K × 4.4 bits = ~48 KB
  - **Net: NWC1 only beats FP4A once the renderer payload is large enough that the 64KB codec amortizes.** For an 88K renderer that crossover is unfavorable; needs ≥150K params before NWC1 wins on raw bytes.
- This makes Lane J-NWC most valuable for the LARGER renderers (Lane V family at ~88K, Lane SZ at ~94K SegMap, Lane I Cool-Chic at ~120K with latents) and a CO-LANE for small renderers (need a smaller codec or a shared corpus codec).

## What's NOT yet done (intentional — implementation-only round)

1. **No GPU run yet.** Per parent agent's directive, this round is implementation only. `scripts/remote_lane_nwc.sh` is ready to dispatch but has not been launched.
2. **No CUDA `[contest-CUDA]` score.** The CLAUDE.md FORBIDDEN-empirical-claim-without-evidence rule explicitly prohibits any unqualified savings claim until a contest-CUDA `[contest-CUDA]` artifact exists. Predictions above are clearly tagged `[prediction]`.
3. **Codec is currently per-checkpoint.** A single shared codec across the whole experiments/ tree (trained once on the corpus, then reused for many renderer compressions) would unlock the amortization win. The dispatch script trains a fresh codec per dispatch.

## Dispatch readiness

- All 14 STRICT preflight checks targeting `scripts/remote_lane_*.sh` PASS for `remote_lane_nwc.sh`:
  - check_remote_scripts_record_predicted_band ✅
  - check_remote_scripts_tag_contest_cuda_at_completion ✅
  - check_remote_scripts_have_nvdec_probe ✅
  - check_remote_scripts_probe_nvdec_early ✅
  - check_remote_scripts_executable_bit ✅
  - check_remote_scripts_write_provenance ✅
  - check_lane_scripts_strip_macos_resource_forks ✅
  - check_no_tmux_kill_server_in_lane_scripts ✅
  - check_no_unconditional_ensurepip ✅
  - check_shell_set_e_present ✅
  - check_no_shell_zip_binary ✅
  - check_no_pipefail_grep_q_trap ✅
  - check_archive_builders_use_deterministic_zip ✅
  - preflight_shell_lane_arity ✅
- Full `preflight_all(check_codebase=True)` PASS.
- 68 tests across test_lane_j_nwc.py (8) + test_neural_weight_codec.py (10) + test_preflight_arity.py (50) PASS.

## Files

- `submissions/robust_current/inflate_renderer.py` — NWC1 dispatch case added
- `experiments/pipeline.py` — `weight_compression="nwc"` branch + `weight_codec_path` field
- `scripts/remote_lane_nwc.sh` — NEW canonical 4-stage dispatch
- `src/tac/tests/test_lane_j_nwc.py` — NEW 8 tests

## Council notes (anti-conservative-bias compliant)

Per CLAUDE.md "Council conduct — non-negotiable": this lane is a **compositional** orthogonal axis (different bit budget mechanism from FP4/I4LZ/Hessian-Ω/Self-Compress). It is NOT a replacement for SCv1/OMG1 — Selfcomp's "no premature convergence" rule says we keep multiple weight-codec lanes alive. The amortization caveat (NWC1 wins only on larger renderers) suggests an immediate follow-up:

1. Train ONE shared codec on the entire `experiments/results/` corpus (~hundreds of `.pt` files).
2. Persist that codec to `submissions/shared_codec.pt` (council-blessed asset).
3. Then NWC1's per-renderer cost drops to just `~1.5 bits/weight + small per-tensor framing` — at which point NWC1 wins on ALL renderer sizes. This is the Ballé-2018 hyperprior pattern: rate prediction networks (the codec) replace fixed factorized priors when archive size matters.

Quantizr's question to ask: does NWC1 stack with Lane A's PoseNet-affine-learned-image trick? Yes — the affine image is a small renderer-state tensor; if it's bundled into renderer.bin, the codec will compress it.
