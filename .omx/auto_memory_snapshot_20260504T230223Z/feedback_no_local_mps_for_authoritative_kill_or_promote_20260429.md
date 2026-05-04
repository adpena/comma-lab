---
name: NEVER use MPS or local CPU for authoritative kill/promote decisions — except where bit-identical
description: 2026-04-29 PM. Reinforced after the clean-source STC FALSIFICATION error: I declared the council's #1 hope dead based on a local MPS-encoder run. The 50× regression measurement was MPS-derived. CLAUDE.md non-negotiable already forbids MPS for strategic decisions; this memory captures the broader rule and the narrow exception.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The rule

ANY kill/promote decision based on a measurement that depends on a neural-network forward pass (SegNet, PoseNet, the renderer itself, distilled scorers, learned codecs, etc.) MUST come from contest-CUDA hardware. Not MPS. Not local CPU. Not Modal CPU. Tag every score `[contest-CUDA]` (valid for strategy) or `[advisory only]` / `[MPS-PROXY]` / `[Modal-T4-CPU]` (NOT valid for strategy).

Verified MPS-vs-CUDA drift (CLAUDE.md, 2026-04-25):
- PoseNet distortion: 0.245 (MPS) vs 0.0107 (CUDA) — **23× worse on MPS**
- SegNet distortion: 0.0024 vs 0.00116 — **2× worse on MPS**
- Final score: 2.26 vs 0.90 — **2.5× worse on MPS**

Modal-T4-CPU vs Modal-T4-CUDA drift is smaller (CPU PyTorch is deterministic IEEE 754 vs CUDA's nondeterministic-by-default kernels) but still NOT bit-identical for any model containing softmax / attention / batch norm. Treat Modal-T4-CPU as advisory only for kill decisions unless the gap to contest-CUDA has been verified < 0.005 score.

## The narrow exception (where MPS / local is OK)

Use MPS or local CPU ONLY when the measurement is provably bit-identical across hardware. Specifically:

1. **Pure file IO / hashing**: ZIP byte count, SHA256 of an archive, file existence checks — these are bit-identical.
2. **ffmpeg decode of an existing video**: ffmpeg is deterministic CPU; the decoded uint8 frames are bit-identical regardless of the calling host.
3. **STC encoder operating on FIXED class IDs**: once the argmax class IDs are fixed (whether by ffmpeg-decode-of-AV1 or some other reproducible source), the STC encode is pure Python/CPU integer arithmetic. The OUTPUT BYTE COUNT is deterministic.
4. **Pure-Python preprocessing**: numpy vector operations on uint8 arrays, scipy convex solvers on pre-computed Hessians, etc.
5. **Code-correctness tests**: pytest checks that a function returns the right shape / raises the right exception — orthogonal to numeric output.

The dividing line: **does the measurement depend on a neural-net forward pass?** If yes → contest-CUDA only. If no → local OK, but document the dependency chain.

## Why this rule exists (incidents that prove it)

- 2026-04-25: 5 catastrophic measurement bugs incident. All historical scores above 0.90 were MPS artifacts inflated 2-3× over the true CUDA value.
- 2026-04-29 PM (this incident): I called clean-source STC dead based on a 21MB measurement. The measurement WAS partially deterministic (STC encode is CPU-deterministic) but the INPUT class IDs depended on MPS-SegNet output. Different pipeline stage, same class of error. Cost: incorrect kill of council's #1 hope, ~2 hours of confused strategy until user catch. The codex max-rigor verdict revealed a REAL implementation bug (one-majority-plus-exceptions encoding) that would have stayed hidden longer if I hadn't been forced to look back.

## How to apply

1. Before declaring any lane DEAD or PROMOTE, name the upstream dependencies of the measurement. Which neural models ran? Which device?
2. If any neural pass was non-CUDA, downgrade the measurement to advisory and require a CUDA replication before the kill/promote takes effect.
3. The narrow exception applies only to measurements with NO neural pass in the dependency chain.
4. When in doubt, treat as MPS-PROXY and require CUDA validation. Cost is small (~$0.20 for Modal T4 / ~$0.05 for Vast.ai 4090); cost of wrong kill is hours-to-days.
5. Tag scores in commit messages, run_log, and memory files with `[contest-CUDA]`, `[Modal-T4-CUDA]`, `[Vast-4090-CUDA]`, `[Modal-T4-CPU]`, `[MPS-PROXY]`, or `[advisory only]`. No untagged scores.

## Cross-refs

- CLAUDE.md "MPS auth eval is NOISE" non-negotiable
- feedback_mps_cuda_drift_critical.md (the original 23× drift evidence)
- project_lane_stc_clean_source_FALSIFIED_20260429.md (the incident this rule responds to)
- project_cuda_gate_result_20260425.md (the 0.90 vs 2.26 measurement)
