# Orthogonal Frozen-Stream Optimization - 2026-05-01

## Scope

This note records an execution strategy for the current Shannon-floor push:
optimize one archive stream at a time while freezing the others as scorer
anchors, then validate every composition as a new exact archive. It is motivated
by live CUDA evidence that byte wins can be real while scorer geometry collapses.

This is a design/control-plane document, not a score ledger.

## Model

Let a contest archive be decomposed into coupled streams:

- `M`: mask/video geometry stream.
- `R`: renderer/model stream.
- `P`: pose stream.
- `C`: charged side information, repair atoms, entropy models, and decoders.

The exact objective is still the contest score:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

A proposed update is valid only after deterministic archive construction and
exact CUDA auth eval through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Alternating Discipline

Use an alternating trust-region workflow:

1. Freeze `R` and `P`; optimize `M` under PoseNet/SegNet response constraints.
2. Freeze `M` and `P`; optimize `R` under component response constraints.
3. Freeze `M` and `R`; optimize `P` only when pose bytes or pose regeneration
   can be validated as charged archive payloads.
4. Build a stacked archive for any cross-stream composition. Do not add deltas
   algebraically.

Each update must carry:

- Archive bytes and SHA-256.
- Per-component CUDA response or exact eval evidence.
- Explicit rate accounting for every repair bit, decoder, model, table, and
  side-information stream.
- A rollback/repair rule if PoseNet or SegNet exits the trust region.

## Current Operational Implications

- Plain Alpha grayscale/CRF mask reduction is not enough. It can cut bytes while
  destroying PoseNet geometry. Alpha successors must include sparse repair,
  PoseNet-aware atom selection, NeRV/INR geometry preservation, pose
  regeneration, or another charged protection mechanism.
- Direct-FD and OWV3 renderer moves should keep the mask stream frozen until a
  stacked exact archive proves interaction.
- The component-response optimizer can rank candidate deltas only within one
  baseline/custody context. Mixing response bundles from different baselines is
  invalid unless explicitly modeled as a new stacked archive.
- L40S/RTX/A100/H100 runs are useful for fast CUDA signal. T4 or contest-
  equivalent exact eval remains the promotion-grade confirmation path when the
  evidence grade requires it.
- Hard pairs, hard zones, semantic classes, pose-sensitive intervals,
  engineered corrections, adversarial/learned repair, and inverse-steg style
  payload allocation are valid search degrees of freedom. They become
  actionable only when expressed as charged archive bytes plus a deterministic
  decoder and exact CUDA component evidence.
- Leaderboard reverse engineering is an input stream, not evidence. Top
  external submissions appear to concentrate on scorer-aligned learned
  representations and tiny charged decoders; use that to prioritize mask-stream
  and learned-geometry work, but do not import claims without exact local
  archive custody.

## CRF63 Response Lesson

The first Alpha primitive response bundle measured CRF63 lossy mask variants
against the PFP16 baseline. Every nonzero point collapsed PoseNet/SegNet, so
the bundle is evidence about the lossy base and decoder geometry, not a valid
marginal sparse-atom ranking. The corrected loop is:

1. Isolate the lossy-base threshold with exact CRF sweeps.
2. If a base stays near component gates, add sparse repair from official
   response data.
3. If the base collapses, switch to geometry-preserving Alpha families:
   NeRV/INR, learned LUT/SegMap, or pose-conditioned decoder designs.

## Implementation Queue

1. Harvest and audit Alpha CRF60/CRF62 threshold exact evals.
2. If either threshold archive is component-safe, convert response curves into
   protected sparse-repair atom sets.
3. Instantiate only the smallest Alpha lossy-repair specs that satisfy response
   gates and deterministic custody.
4. Run fast CUDA exact eval on a high-throughput GPU for screening.
5. Promote finalists to T4/equivalent exact eval.
6. Stack the best Alpha geometry-preserving archive with renderer OWV3/direct-FD
   candidates only as complete archives, then exact eval again.
7. In parallel after the threshold read, harden and relaunch Q-FAITHFUL or a
   Selfcomp/SegMap-class learned representation only if its archive/inflate
   contract can be made deterministic and fully charged.

## Non-Claims

No leaderboard, floor, or stack-composition claim is made here. This document
defines the scientific control loop required before those claims can exist.
