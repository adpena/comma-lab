# Postmortem: Bridging The Gap

This is a candid engineering postmortem for the comma video compression
challenge. It is not a score ledger; exact scores remain governed by the
adjudicated CUDA artifacts named in the results section.

## Core Diagnosis

We were not primarily missing ideas. We independently explored or requested
many of the mechanisms that later appeared in strong public submissions:
implicit neural video representations, HNeRV/NeRV-style mask codecs, learned
motion latents, scorer-targeted corrections, arithmetic and range coding,
bit-level archive anatomy, water-filling, hard-pair allocation, foveation,
openpilot/camera priors, and self-compression. The gap was that too many of
those ideas stayed as research notes, partial lanes, or proxy experiments
instead of becoming byte-closed archives with deterministic inflate paths and
exact CUDA scores early enough.

The contest reward was not for having the right theory. It was for converting
theory into charged bytes that survived:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

Under that standard, the highest-leverage failure modes were operational:

- **Research-to-archive latency.** HNeRV, arithmetic coding, low-level packing,
  learned pose/motion priors, and residual correction ideas needed canonical
  builders, decoders, manifests, and no-op controls sooner.
- **Local-minimum pressure.** Some effort continued polishing already-known
  basins after marginal returns were small, while orthogonal high-EV lanes
  needed more parallel engineering pressure.
- **Proxy-to-CUDA gaps.** MPS/CPU/proxy/local-stub results consumed attention
  even when they could not rank, promote, or retire a method.
- **Fragile experiment plumbing.** Dead flags, missing dispatch claims,
  wrapper mismatches, runtime-custody drift, no-op recodes, and sidecar risk
  repeatedly converted valid scientific ideas into invalid or low-signal runs.
- **Leaderboard-intake lag.** The late public meta required continuous
  archive/PR/body/comment ingestion, immediate exact replay, and fast
  adversarial deconstruction. That loop existed, but it became decisive too
  late.
- **Over-retirement of families.** Bad implementations of grayscale-LUT,
  NeRV, foveation, or mask grammars were sometimes treated too broadly before
  the exact confound was isolated.

## What Would Have Been Different

The stronger operating model would have been a two-track archive compiler from
the first week.

Track one: a strict promotion lane. Every candidate must have typed inputs,
typed payloads, deterministic archive emission, runtime tree hashing, no-op
controls, exact CUDA eval, and evidence-grade bookkeeping.

Track two: a reckless but bounded frontier lane. Every promising family gets a
minimal byte-closed prototype immediately, even if the first version is ugly,
expensive, or likely to fail. The rule is not "wait until elegant"; the rule is
"make the archive consume the idea, then let CUDA decide." Elegance follows
after the first byte-closed result.

Concrete changes:

1. Promote public-submission deconstruction to a standing process: poll PRs,
   fetch blobs, profile members, replay exactly, and generate a diffable byte
   scorecard on a cadence measured in minutes during deadline windows.
2. Put every idea behind a shared archive contract:
   `representation -> prediction -> quantization -> hyperprior -> arithmetic -> pack`.
   Hidden lane-local scripts should not be the only implementation of a concept.
3. Require every research note to name its smallest byte-closed experiment:
   archive member, charged bytes, decoder, expected score term, no-op control,
   and exact-eval command.
4. Keep one unsafe GPU lane running for high-EV architecture changes:
   HNeRV/NeRV/SIREN/RAFT/CLADE/SPADE/Cool-Chic-style learned representations,
   even when the strict lane is still hardening.
5. Treat bug classes as compiler errors: dead flags, wrong runtime, MPS/CPU
   leakage, missing side-channel application, duplicate payload containers,
   and parser drift should fail before dispatch.
6. Use hard-pair, foveation, motion, and LA-POSE-style latent priors as routing
   fields for charged atoms, not as free-standing narratives.
7. Preserve negative results, but narrow their scope. A bad grayscale-CRF
   result retires that implementation, not analog grayscale-LUT as a family.

## Scientific Lesson

The Yousfi-Fridrich view remains right: this is a charged sufficient-statistic
program for a fixed video and fixed neural evaluator. The useful floor is below
generic human-video intuition because the archive only needs to satisfy
SegNet/PoseNet, not human perception. But the path to that floor is an
engineering compiler problem as much as a mathematical one. The math proposes
atoms; the compiler must lower them to bytes; exact CUDA decides whether the
atom exists.

The post-deadline `tac` direction is therefore clear: make the research system
less dependent on ad hoc working memory and more like a production archive compiler.
Every lane should be discoverable, typed, reproducible, benchmarkable,
stackable, and fail-closed. That is how the month of research becomes useful
for future comma.ai work even where the final contest timing was missed.
