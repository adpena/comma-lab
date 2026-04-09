# Lab Methodology

Last refreshed: `2026-04-08 16:43:45 -0500`

## Purpose

This repo is a lab, not just a pile of experiments. The goal is to improve the official challenge score while preserving enough evidence, timestamps, and methodology that another engineer can reconstruct what happened, why we believed it, and where we were wrong.

## Source Of Truth Hierarchy

1. The pinned upstream snapshot is the source of truth for scorer behavior, submission format, and evaluation mechanics.
2. The local authoritative CPU scorer path is the promotion authority for the current non-GPU shipped lane.
3. The official upstream evaluator used through the faithful proxy path is the ranking authority for cheap candidate screening.
4. Training metrics, local loss curves, and side-lane results are hypothesis-generating signals only.
5. Remote side lanes are useful for throughput and ranking, not for promotion claims unless they reproduce the official path and leave full evidence on disk.

## Scientific Method

1. State the hypothesis.
2. Limit the experimental change set.
3. Produce or reuse a concrete artifact.
4. Run the cheapest faithful check that can falsify the hypothesis.
5. Promote only if the next stronger check still supports it.
6. Record the result, including failures and surprises.

Every serious branch should answer:
- What changed?
- Why should that help?
- What is the cheapest meaningful falsifier?
- What evidence would justify promotion?

## Experiment Lifecycle

1. Create or identify a candidate artifact.
2. Verify packaging and inflation behavior.
3. Run smoke on exact frame count, geometry, and sampled semantic sanity.
4. Run official-path proxy evaluation when scorer time is still too expensive.
5. Run authoritative scorer evaluation only for candidates that clear the proxy bar.
6. Update durable state and writeup surfaces immediately after the result lands.

## Promotion Rules

A candidate is promoted only after:
- packaging succeeds
- inflation succeeds
- shape and frame-count checks pass
- the official-path proxy is promising, if used
- the authoritative scorer confirms the gain

If any of those fail, the branch is not a promotion candidate.

## Null Hypothesis Discipline

The default assumption is that a new branch does **not** beat the floor.

That means:
- close misses are recorded as close misses, not “almost wins”
- smaller artifacts that do not improve score are not promotions
- better local training curves do not matter if the official-path proxy or scorer disagrees

## Timekeeping Rules

Use actual timestamps whenever possible.

Preferred sources:
1. evidence file mtimes
2. report contents
3. remote/job manifests
4. git history

Avoid vague wording like “earlier” or “later” when a concrete timestamp exists.

## Distributed Execution Rules

One authoritative CPU scorer owner at a time.

Other lanes may run in parallel only if they do not interfere with that scorer lane:
- local MPS/GPU training
- remote proxy/eval on other machines
- remote GPU training on other machines
- writeup/site rebuilding

Each remote lane should have:
- a declared host
- a declared purpose
- a declared artifact
- a log path
- a timestamped manifest

## Upstream Compliance

Never edit, patch, monkeypatch, or “temporarily” modify the pinned upstream snapshot without explicit human approval.

If upstream behavior is inconvenient or blocking:
- work around it from the allowed mutation frontier
- record the issue in repo state
- do not silently change the source of truth

## Writeup Discipline

The writeup should evolve with the evidence, not after it.

Whenever a meaningful result lands:
- update the current floor and frontier tables
- update the timeline
- update the queue
- update the public-facing summary if the story changed

## What Counts As Success

Success is not only a lower score.

It is also:
- fewer false positives from proxy drift
- clearer evidence trails
- timestamped milestones
- stable distributed execution
- smaller, more faithful shipped artifacts
