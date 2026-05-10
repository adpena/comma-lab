# Current score-lowering roadmap (2026-05-10)

## Evidence anchors

- A1 `[contest-CPU GHA Linux x86_64]`: `0.19284757743677347`,
  archive bytes `178262`, archive SHA-256
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`.
- A1 paired `[contest-CUDA T4]`: `0.2263520234784395`.
- Do not describe the A1 CPU anchor as CUDA-frontier, gold-equivalent, or
  submission-ready until paired CUDA policy is satisfied.
- Current theoretical-floor planning anchor remains approximate:
  `0.140 ± 0.012`; lower movement requires byte-closed substrate/training work,
  not MPS or macOS advisory promotion.

## P0: local score-lowering and custody

1. Retire the measured A1 per-pair latent sidecar `proxy_mse` packet.
   - Archive candidate: `178316` bytes,
     `c7f3d88e1ad23bf8cda987583e702ac57e293b64bc7bfea77902e835d19cea10`.
   - Local packet proof completed: 600/600 scalar-equivalent records, exact
     `inflate.sh` smoke, strict pre-submission compliance, and live claim
     binding.
   - Exact `[contest-CPU GHA Linux x86_64]` dispatch result:
     `0.20962552129271272`, worse than A1 baseline
     `0.19284757743677347`; measured configuration retired.
   - Reactivate only with score-domain or joint SegNet/PoseNet sidecar search,
     not `proxy_mse` pair selection.
2. Keep A1 bias-coordinate work bounded.
   - Existing broad variants regressed or failed to beat V1.
   - Reopen only with a small reviewed candidate set and CPU-positive evidence.
3. Keep AVVideoDataset CUDA-path discriminator closed as CPU-only unresolved
   unless a fresh CUDA-capable claim is filed.

## P1: substrate recovery and HNeRV parity

1. Reproduce PR95/PR101/PR103 mechanics with exact custody:
   eval-roundtrip-in-training, differentiable YUV path, runtime constants,
   EMA/export discipline, and archive build in the loop.
2. Reactivate Track4 only through score-gradient/cliff-aware saliency or a
   stronger criterion. The old UNIWARD/STC/Hessian measured configs remain
   negative and are not exact-eval candidates.
3. Rebuild A5 only around score-domain channel allocation or q-bit noise during
   training. Scalar/global splits are exhausted.

## P2: high-upside architecture lanes

1. Phase1/T1 stays local until it emits a runtime-consumed byte-different packet
   with no-op proof, exact smoke, and custody. No blind GPU dispatch.
2. Lane12-v2 stays local until it has hermetic runtime, scorer-preprocess
   gradcheck, PR95/PR100 parity or deviation record, packet builder, and dual
   exact-eval readiness.
3. Phase2/T15/T17/T18/T9 stay deferred until Phase1 or a single-axis substrate
   produces a validated exact anchor.

## Status corrections

- MPS is advisory only for sweeps, curves, configuration discovery, and training
  starts; never for auth eval promotion.
- `[contest-CPU]`, `[contest-CUDA]`, local macOS CPU, and MPS proxy evidence
  must remain separate.
- A1 sidecar `proxy_mse` packet passed local custody but regressed on exact
  `[contest-CPU]`; do not redispatch this measured packet.
- No active dispatch claims remain after closing the AV discriminator CPU-only
  run as unresolved for CUDA and closing the A1 sidecar regression claim.
