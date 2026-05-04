# Public PR95 HNeRV/Muon Reproduction Intake - 2026-05-04

## Scope

Owned additive scaffold only. No remote jobs were dispatched. The intake target
is `experiments/results/public_pr95_intake_20260504_codex/archive.zip` plus the
public PR95 `submissions/hnerv_muon` source mirror already present in the repo.

## Artifact Contract

New profiler:

- `experiments/profile_pr95_hnerv_muon_intake.py`
- default JSON output:
  `experiments/results/public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.json`
- default Markdown output:
  `experiments/results/public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.md`

The profiler performs static archive and source accounting only. It parses ZIP
anatomy, the `0.bin` PR95 HNeRV/Muon codec sections, INT8 decoder tensor tables,
latent stream metadata, source curriculum stages, optimizer partitioning, and
score-term math when the existing static intake JSON provides public inputs.

## Dispatch Readiness

Fail closed. Static PR95 intake is external/static evidence, not a score claim.
Replay exact eval through `archive.zip -> inflate.sh -> upstream/evaluate.py` on
CUDA is required before any PR95 score claim. Owned retraining requires explicit
manifest and checkpoint custody for source SHA, stage config, seed, checkpoint
paths, checkpoint SHA-256 values, optimizer-state policy, final archive bytes,
archive SHA-256, and exact eval provenance.

## HNeRV Improvement Hooks

1. RAFT/ego-motion/foveation latent bases: constrain part of the per-frame-pair
   latent table to camera-aware charged bases rather than replaying free latent
   memorization.
2. Cool-Chic/C3/wavelet residual bases: add a tiny charged residual program for
   systematic local errors that global HNeRV latent movement cannot cheaply
   correct.
3. Fridrich/Lagrangian hard-pair weighting: use exact component traces to weight
   frame pairs and allocate latent/residual/entropy budget by byte-normalized
   score benefit.
4. Engineered corrections and pixel-diff sparse atoms: encode deterministic
   sparse corrections only where changed pixels clear break-even component math.
5. Self-compression entropy objectives: extend C1a into measured coder-aware
   pressure over decoder tensors, latent deltas, and optional residual atoms.

## Evidence Status

Evidence grade: `external_static_intake_only`.

Promotion blockers:

- no local exact CUDA replay artifact yet;
- no owned checkpoint custody yet;
- no retrain manifest yet;
- no archive mutation or remote dispatch in this scoped scaffold.
