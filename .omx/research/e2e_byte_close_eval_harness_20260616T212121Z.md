# E2E byte-close -> eval verification harness (#107 readiness actuator) — LANDED

**Subagent:** Y1 (parallel to the bind-all build-out). **Date:** 2026-06-16T21:21:21Z.
**Authority:** ALL numbers `[contest-CPU advisory] NON-PROMOTABLE`. CPU only — NO MPS, NO CUDA, NO remote,
NO paid eval. **Mission tag:** frontier_breaking_enabler (turns a trained small-basis checkpoint into an
advisory S today, and is the EXACT packet-builder the G3 dual CPU/CUDA exact row consumes on contest
hardware). **NO score / frontier / promotion claim** — the frontier is 0.191 and UNMOVED.

## HEADLINE: the harness REALLY byte-closes the real basin and reproduces its advisory bit-for-bit

`tools/verify_e2e_byte_close_eval.py` is a one-command actuator:
`--ckpt-dir <best/> [--max-pairs N] [--taper-channels c0,..,c6] [--rate-denom B] [--out J] [--keep-packet]`.
It (1) loads the checkpoint, (2) byte-closes it into the REAL contest `0.bin` via the vendored codec,
(3) assembles the REAL contest `archive.zip` (containing ONLY `0.bin`, per `compress.sh`) + the runtime
tree, (4) runs the authority-faithful eval, (5) reports S recomputed FROM components.

**REAL n600 basin e2e result** (`reports/e2e_byte_close_eval_n600_headline.json`, full 600-pair eval):

| field | value |
|---|---|
| ckpt | `experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best` (base_ch=20, 83,356 params, 600×28 latents) |
| **byte-close `0.bin`** | **89,136 B** (== the basin's recorded `archive_bytes`) |
| **contest `archive.zip`** | **89,274 B** (0.bin + 138 B zip container overhead) — the REAL contest rate numerator |
| section breakdown | meta 81 + decoder_blob 73,469 + latents 15,574 + 12 B prefixes = 89,136 (consumed exact, trailing 0) |
| parse-back parity | **parity_ok=True** (build_deterministic + weights_fixed_point + latents_fixed_point + keys_match) |
| **d_seg** | **0.0026009199** (== recorded advisory, bit-for-bit) |
| **d_pose** | **0.0003416866** (== recorded advisory, bit-for-bit) |
| rate (zip bytes / 37,545,489) | 0.0023777557 |
| seg / pose / rate terms | 0.260092 / 0.058454 / 0.059444 |
| **S (archive.zip bytes)** | **0.3779899** |
| S (raw .bin bytes) | 0.3778980 (== the basin's recorded `score` 0.377898) |
| archive.zip sha256 | `0320297a89fb90d67d95f858d84a4444c19c2d451ea6e1800c17d8d2911d7e76` |

The harness reproduces the basin's recorded advisory d_seg/d_pose EXACTLY (NO-FAKE: it runs the same
frozen SegNet/PoseNet over the same streamed GT, not a proxy). The ONLY new number is the honest
+0.0001 S correction from using the REAL `archive.zip` st_size (89,274) instead of the raw `0.bin` size
(89,136) the in-loop advisory used as a rate proxy — i.e. the in-loop advisory slightly UNDER-counted the
contest rate term, and this harness surfaces it.

## The pipeline (5 steps) and what each REUSES (search-first, named — nothing reimplemented)

1. **Load** `best_ema_decoder.pt` + `best_ema_latents.pt` + `best_meta.json` (mirrors
   `experiments/probe_dseg_sensitivity_map_basin_n600.py`); infers (latent_dim, base_channels, n_pairs)
   from tensors when meta is silent (base_ch from `stem.weight` rows / 48). Missing files RAISE (NO-FAKE).
2. **Byte-close** via `tac.torch_vehicle.driver.import_vendored_bundle().build_archive` — the
   schedule-agnostic vendored PR95 codec (`.../hnerv_muon/src/codec.py:145`), the SAME call
   `driver._build_archive_and_eval_decoder` (driver.py:2105) wires. G2 (`8f...` verdict
   `g2_c8_bilinear_skip_byte_close_verdict_20260616T160046Z.md`) proved the bilinear-skip round-trips;
   the harness asserts that contract: build is deterministic + parse-back of a parse-back is bit-exact on
   weights AND latents + key parity (`tac.torch_vehicle.tests.test_bilinear_skip_byte_close_g2`). If the
   fixed-point fails the harness RAISES — refuses to report a score.
3. **Packet** assembled per the vendored `compress.sh` + `evaluate.sh` contract: `archive.zip` holds
   EXACTLY `0.bin` (deterministic zip: fixed 1980 timestamp -> byte-stable st_size), and the runtime tree
   (`inflate.py`/`inflate.sh`/`src/` copied from `VENDORED_SRC.parent`) lives BESIDE the zip — NOT in it.
   This matters: `upstream/evaluate.py:63` scores `(submission_dir/'archive.zip').stat().st_size`, so the
   runtime is NOT counted in the rate (only `0.bin` is). The packet_dir is a runnable contest submission_dir.
4. **Eval** the parse-back decoder (rebuilt via the vendored `model.HNeRVDecoder`, or
   `ConfigurableTaperHNeRVDecoder` when `--taper-channels` is given — mirrors `driver._new_vendored_decoder`)
   through `score.evaluate_decoder` + `score.compute_score` — the SAME vendored primitives
   `RealScorerContext.exact_eval` (scorer_context.py:184) uses. GT decodes ONLY via
   `frame_utils.yuv420_to_rgb`; the decoder forward -> bicubic 874×1164 -> uint8 is the SAME roundtrip
   `inflate.py` + `evaluate.py` do. Authority device = CPU (NEVER MPS). The harness calls these primitives
   DIRECTLY (not via `RealScorerContext`) to skip the per-step-loss target precompute, which for n=600
   loads a ~900 MB cache into RAM (the eval path streams GT itself and never reads those targets) — this
   is a faithful optimization, not a divergence.
5. **Report** S = 100·d_seg + sqrt(10·d_pose) + 25·archive_zip_bytes/denom, recomputed FROM components
   (never a rounded field). Denominator defaults to the source video st_size (= `evaluate.py:64`
   `uncompressed_size` = 37,545,489 = `0.mkv`). JSON tagged `[contest-CPU advisory] NON-PROMOTABLE` with
   BOTH byte accountings (.bin vs archive.zip), the per-section breakdown, parse-back parity, the zip
   sha256, and the checkpoint's recorded-advisory line for an apples-to-apples sanity check.

## Byte-accounting finding (a real, durable signal for the production run + G3)

The contest rate term is the **`archive.zip` st_size**, not the raw `0.bin` size. For this basin the zip
container adds **+138 B** (89,136 -> 89,274), i.e. +0.0000037 rate, +0.0000919 to 25·rate, +0.0001 to S.
Small here, but it is the honest contest number and the production run / G3 should quote the zip-bytes S.
The driver's in-loop `best_meta.json` records the `.bin` size as `archive_bytes`; this harness is the
surface that converts it to the contest-accurate S.

## How the production run + G3 consume this

- **Production small-basis run (P3 / Y-main):** after the long train converges, point this harness at the
  `best/` dir to get the contest-accurate advisory S in one command — no manual byte-close, no manual
  packet, no rounded-field trust. `--taper-channels 22,16,15,14,15,14,10` (the solved taper) is supported
  for the taper vehicle (the parse-back decoder rebuilds via `ConfigurableTaperHNeRVDecoder`).
- **G3 dual CPU/CUDA exact row:** run with `--keep-packet` to retain the assembled submission_dir
  (`archive.zip` + `inflate.sh`/`inflate.py`/`src/`). That dir is the EXACT input to `upstream/evaluate.sh
  --submission-dir <pkt> --device {cpu,cuda}` on contest-compliant Linux x86_64 / CUDA — the advisory S
  this harness reports becomes the exact row once that runs (advisory -> authority on 1:1 hardware).

## Tests (NO-FAKE) — `src/tac/tests/test_verify_e2e_byte_close_eval.py`, 7 passed

Real-basin: checkpoint load + dim inference; missing-ckpt RAISES (not fabricates); byte-close == 89,136 B
+ parity_ok; section breakdown sums to the archive + decoder>latents>meta. Packet: archive.zip holds
EXACTLY `0.bin` + runtime BESIDE it + byte-deterministic. FULL pipeline (real scorer, `--max-pairs=2`
CI-fast): byte-close all-600-latents to 89,136 B, parity exact, rate = zip/denom, real d_seg/d_pose same
ORDER as recorded advisory, S recomputed-from-components, packet runnable. CLI `main()` writes valid JSON.
Tests SKIP (not fail) when the gitignored vendored clone / basin / frozen scorer are absent (portable).
The G2 byte-close contract I reuse re-verified green (7 passed).

## Wire-in (6-hook) per Catalog #125

- #1 sensitivity-map: N/A (a readiness actuator, not a new score signal).
- #2 Pareto: N/A. #3 bit-allocator: N/A (it MEASURES the rate term; the section breakdown is an input a
  bit-allocator could consume, but this harness emits it as a report, not a planner mutation).
- #4 cathedral autopilot: N/A (no new dispatchable candidate; it is the byte-close+eval pipeline existing
  candidates flow THROUGH).
- #5 continual-learning: this memo + `reports/e2e_byte_close_eval_n600_headline.json` (durable, tracked)
  are the artifacts; the basin's contest-accurate advisory S = **0.37799** (zip bytes) is the durable row.
- #6 probe-disambiguator: the harness IS the disambiguator between "in-loop advisory S (0.377898, .bin
  proxy)" and "contest-accurate advisory S (0.377990, archive.zip st_size)".

## Guardrails honored

CPU/code only; NO `driver.py` / `experiments/launch_*.py` edits (sister-owned); did NOT disturb any
running job (own background runs only, polled by pid, cleaned on exit); GT via `yuv420_to_rgb`; MPS never
an authority; disk hygiene (the kept 600-pair packet was 280 KB rebuildable — deleted after recording its
sha256 + all bytes in the JSON report); review-gated (2 passes, no `.py` override); commit via serializer.
