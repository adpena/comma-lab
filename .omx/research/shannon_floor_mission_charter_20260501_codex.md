# Shannon-Floor Mission Charter - 2026-05-01

Deadline: `2026-05-03T12:00:00-05:00` America/Chicago.

Evidence type: `mission_control_policy`. This document is not score evidence.
It defines the operating objective and decision rule for all contest work until
the deadline.

## Mission

Drive the comma video compression challenge system as close as possible to the
Shannon-theoretical floor by `12:00 PM America/Chicago` on Sunday, May 3, 2026.

## Primary Objective

Minimize the official contest score:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

The target is not incremental improvement. The target is a contest-valid
archive that approaches or breaks through the current public frontier by
combining mathematical rate-distortion optimization, exact scorer-aligned
engineering, reverse engineering of public top submissions, and aggressive but
reproducible search.

## Operating Principle

Treat every possible representation, codec, decoder, latent, pose stream, mask
stream, renderer, packer, repair atom, learned model, entropy code, and archive
layout as an optimization variable. Keep the search space broad, but require
exact evidence before making score claims.

## Evidence Standard

The only score truth is exact CUDA evaluation of the exact archive bytes
through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

Every candidate must record archive bytes, SHA-256, component distances, sample
count, recomputed score, hardware, eval command, logs, manifest, provenance,
and failure classification.

CPU, MPS, proxy, local renderer checks, smoke tests, byte-only results, stale
logs, and intuition are development signals only. They cannot promote, rank,
kill, or anchor stack math.

## Deadline Behavior

Until the deadline, prioritize wall-clock score reduction over elegance. Use
fast available hardware for iteration, reserve T4/equivalent for
promotion-grade confirmation, and run independent hypotheses in parallel when
they do not corrupt custody or compete for the same artifact tree.

## Search Agenda

1. Reverse engineer the top public submissions at the archive, byte, tensor,
   pose, mask, model, and runtime levels.
2. Reproduce their inflate/runtime behavior locally and through exact CUDA
   traces.
3. Compare our best archive against public-floor traces pair-by-pair and
   component-by-component.
4. Formalize hard pairs as marginal rate-distortion opportunities, not static
   labels.
5. Use Lagrangian/water-filling allocation across nested atoms: global
   topology, mask stream, renderer weights, pose channels, frame pairs, classes,
   connected components, tiles, boundaries, residuals, latents, and
   entropy-coded payloads.
6. Search for synergy and antagonism between atoms; do not assume additive
   deltas compose until a stacked archive is exact-evaluated.
7. Treat learned representations, inverse steganalysis, adversarial scorer
   alignment, ego-motion priors, camera geometry, foveation, and
   openpilot/video-specific structure as valid degrees of freedom.
8. Charge every score-affecting bit inside the archive. No sidecars, scorer
   patches, hidden runtime dependencies, or unrecorded external state.

## Decision Rule

At every branch, choose the action with the highest expected score reduction
per wall-clock minute under contest-compliance constraints. Prefer exact
evidence over theory, but use theory to choose the next expensive experiment.

Mathematically, rank actions by expected utility:

```text
EU(action) =
  E[score_drop | action, evidence]
  / expected_wall_clock_minutes
  - custody_risk
  - compliance_risk
  - opportunity_cost
```

Dispatch the highest-EU action whose custody and compliance risks are bounded.

## Negative Result Policy

No broad method kill from one bad result. Preserve artifacts, classify the
failure, identify the confound, and convert the result into a sharper next
experiment. Retire only measured implementations unless exact independent
evidence or a mathematical impossibility argument supports a broader
conclusion.

## Desired Agent Output

Continuously produce:

- candidate archives;
- exact CUDA evals;
- component traces;
- reverse-engineering artifacts;
- mathematical allocation tables;
- reproducible ledgers;
- next-action dispatch decisions.

## End State

By the deadline, produce the lowest contest-valid archive achievable, plus a
complete scientific custody trail explaining what was tried, what worked, what
failed, what remains promising, and what exact next tranche should run after
the deadline.
