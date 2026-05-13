# Ballé / CompressAI Byte-Closure Audit - 2026-05-13

Scope: local hardening only for `lane_substrate_balle_renderer_20260512`.
No Modal, Kaggle, Lightning, Vast.ai, GPU dispatch, or score claim was made.

## Findings

1. **BRV1 hyper-latent side stream was parse-visible but render-silent.**
   The `scales` section changed archive bytes and parsed tensors, but inflate
   did not bind it to the packaged decoder path. This is a byte-closure risk:
   mutated side-info bytes could survive as dead rate.

2. **Auth-eval artifact naming was brittle for the Ballé remote chain.**
   The trainer wrote `contest_auth_eval_cuda.json`, while the existing Ballé
   remote script checks for `auth_eval.json`. The trainer now emits the stable
   alias after validated CUDA auth-eval JSON exists. This is local packet-path
   hardening only; no auth eval was launched in this pass.

3. **Archive-vs-payload custody needed stronger provenance shape.**
   The trainer already passes `archive.zip` SHA/bytes to auth-eval validation
   and keeps `0.bin` as payload metadata. Provenance now records an
   `exact_eval_packet` block with archive path/SHA/bytes, inflate path, payload
   path/SHA/bytes, score axis tag, and score-claim flag.

4. **Readiness docs overstated coding closure.**
   The substrate docs said ANS-coded int16 latents, but current BRV1 is raw
   int16 main/hyper latents plus brotli-compressed state dicts. Arithmetic or
   range coding remains a blocker before exact replacement dispatch.

## Changes Landed

- `src/tac/substrates/balle_renderer/archive.py`
  preserves quantization metadata in `arc.meta` so inflate can apply
  byte-closure tolerances.
- `src/tac/substrates/balle_renderer/inflate.py`
  validates the archived `scales` stream against the packaged
  `hyper_analysis(latents)` path and fails closed on mismatch.
- `experiments/train_substrate_balle_renderer.py`
  writes the `auth_eval.json` alias and records an explicit
  `exact_eval_packet` provenance block.
- Ballé roundtrip/trainer tests now cover scale-stream closure, valid inflate,
  archive-vs-payload custody, no-scorer inflate, and deterministic zip output.

## Six-Hook Wire-In Declaration

1. Sensitivity-map contribution: N/A - local byte-closure/runtime-custody guard,
   no new empirical scorer sensitivity signal.
2. Pareto constraint: N/A - no score, byte count, or component movement was
   measured.
3. Bit-allocator hook: N/A - no per-tensor bit allocation policy changed.
4. Cathedral autopilot dispatch hook: unchanged - existing Ballé
   smoke-before-full recipe remains the dispatch surface; this pass does not
   authorize or fire it.
5. Continual-learning posterior update: N/A - no empirical anchor.
6. Probe-disambiguator: N/A - no two defensible runtime modes were introduced;
   the side-info stream now has one fail-closed interpretation.

## Remaining Blockers To Exact Non-HNeRV Replacement Dispatch

- Ballé BRV1 still ships raw int16 latent streams; arithmetic/range coding is
  not yet wired into the renderer packet.
- No clean local generated packet has been run through the exact auth-eval path
  in this pass; dispatch should still start with smoke-before-full.
- `lane_compressai_integration_20260512` has packet-compiler primitives but is
  still not a full replacement packet with a renderer, runtime closure, and
  exact-eval packet recipe.
- CPU/CUDA axis closure remains absent for Ballé; any future result must keep
  `[contest-CUDA]`, `[contest-CPU]`, and proxy/advisory axes separate.
- The existing Ballé remote script still logs `0.bin` as `ARCHIVE_PATH`; trainer
  provenance now records the scored `archive.zip`, but the script log line
  should be corrected in a runbook-owned pass before paid dispatch.
