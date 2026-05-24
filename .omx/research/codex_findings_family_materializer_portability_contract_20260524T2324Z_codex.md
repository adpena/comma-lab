# Family Materializer Portability Contract - Codex - 2026-05-24T23:24Z

## Scope

This follow-up preserves a sidecar portability signal that landed after the
all-member ZIP header elision push. Family-agnostic materializer manifests now
carry a `family_agnostic_materializer_portability_contract.v1` record so queue
feedback can distinguish portable CPU/Python reference paths from future native
Rust/Swift/Metal/MLX lowerings.

## Contract

The contract records:

- implementation language;
- required Python modules;
- deterministic implementation surface;
- unsupported archive/features;
- explicit `requires_gpu=false`, `requires_mlx=false`, `requires_metal=false`,
  and `requires_cuda=false`;
- false authority for score, promotion, and rank/kill use.

The packet ZIP header elision path is still the portable Python raw-ZIP32
reference. Native lowering should target measured hotspots only and must keep
this contract as the oracle/fallback boundary.

## Verification

- The focused all-member/materializer/action-functional regression suite passed
  after the contract was preserved in materializer results, sweep observations,
  and normalized inverse-steganalysis observations.
