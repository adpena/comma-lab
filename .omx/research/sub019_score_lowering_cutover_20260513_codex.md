# Sub-0.19 score-lowering cutover

**Date:** 2026-05-13

**Decision:** default score-lowering worklists now target candidates whose
declared predicted band can plausibly beat `0.19` on a labelled exact contest
axis. Rows above that cutoff are preserved as historical/reference signal, but
are hidden from default active routing.

## Evidence motivating the cutoff

- HLM1 exact closure is real custody, but not enough for the current target:
  `0.20638030907530963 [contest-CUDA]` and
  `0.22782680632923968 [contest-CPU]` on archive
  `8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0`.
- PR106 latent-sidecar/HLM1 composition was proven full-frame same-runtime
  parity, but classified negative against HLM1 because it is `+330` bytes versus
  the HLM1 archive and only moves the rate term at `~4.6e-5` score scale.
- SIREN and Ballé/CompressAI hardening both landed with local-only proxy
  authority guards. Their worker verdicts agree: standalone first anchors are
  not credible sub-0.19 dispatch targets without a PR95/PR101 parity stack,
  residual selector, consumed-byte packet path, and local manifest showing a
  plausible sub-0.19 case before GPU spend.

## Active-only lanes

Keep implementation pressure on lanes that can change distortion, not tiny
rate-only deltas:

1. PR95/HNeRV parity training discipline: RGB renderer, inner-loop
   eval-roundtrip, differentiable `rgb_to_yuv6`, scorer-preprocess correctness,
   EMA/export parity, Muon/WD/C1a/L7 stage discipline, exact archive build in
   loop.
2. SIREN only as a scorer-aligned PR106 residual selector or composed
   representation that has local consumed-byte/runtime proof and a plausible
   `<0.19` manifest before dispatch.
3. Ballé/CompressAI only as a co-designed HNeRV/PR95-or-PR101 parity stack or
   structural entropy model with real archive/inflate byte-consumption proof.
4. PacketIR/compiler work only when it attacks high-byte payload structure or
   learned distortion/rate tradeoffs; generic Brotli/format deltas below
   `0.001` score are reference work unless they unlock a larger pass.
5. CPU/CUDA xray only when it changes a dispatch/device decision or exposes a
   contest-compliant mechanism that can improve an exact axis.

## Deferred by cutoff

- PR106 latent sidecar, Y-shift, and LRL1 single-sidechannel rows whose
  predicted bands stay near `0.205-0.208`.
- HLM1 repeat dispatches and HLM1+sidecar recodes that are rate-only and cannot
  close the `>0.016` score gap to `0.19`.
- Standalone SIREN/Ballé/CompressAI canaries without consumed-byte exact packet
  proof and a plausible sub-0.19 local manifest.

## Tooling

`tac.score_target_filter` now centralizes the cutoff logic. The operator
briefing annotates predicted rows with `score_target_routing` and hides
above-target predicted rows by default while preserving them in JSON for audit.
Use `tools/operator_briefing.py --show-above-target` only for review or
historical context, not as active score-lowering guidance.
