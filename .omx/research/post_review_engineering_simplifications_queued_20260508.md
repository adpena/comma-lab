---
title: Engineering simplifications queued from REVIEW-ENG (S4 + S6)
date: 2026-05-08
author: Subagent FIX-ALL-FINDINGS (claude-opus-4-7-1m)
status: research-lane queue items; deferred from FIX-ALL-FINDINGS scope
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
---

# Engineering simplifications queued from REVIEW-ENG (S4 + S6)

REVIEW-ENG flagged five engineering simplifications (S1-S6, plus S8). The
critical and free-win items (S1=C2, S3=C4, S8=C1) landed in commits
266fa12c, 0b24e5d1, e61d385c, f2ba8205, f4f6270c, abc991c5, 0c736176 across
the FIX-ALL-FINDINGS subagent session.

The remaining two (S4 + S6) are research-lane queue items:

## S4 — Analytic λ-bracket replacing geometric ramp (≥1 hour effort)

### Current state

`bisect_admm_for_global_rms` in
`tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py` initializes
`lo, hi = 0.0, 1e15` and uses `mid = (lo + hi) / 2 if hi < 1e15 else
lo * 10 + 1`. The first dozen iterations are a geometric ramp: 1, 11, 111,
1111, ... until `hi` first becomes finite, then standard bisection. This
wastes ~10 iterations exploring the unbounded upper region.

### Recommended improvement

Compute an analytic upper bound for `λ_max` by solving the optimality
condition at the lossless K=1 floor:

```
λ_max ≈ max_t  (byte_proxy(t, K=highest) - byte_proxy(t, K=1)) /
                 max(rel_err(t, K=highest)^2, eps)
```

This bounds the largest λ at which any tensor would still pick a non-K=1
choice. Use `λ_max` directly as the initial `hi`; the geometric ramp
becomes unnecessary, and the bisection always operates in the well-defined
[0, λ_max] interval.

Predicted savings: ~10-15 wasted iterations per bisect call → ~12-18%
wall-clock reduction. Stacks with S2 (commit 0c736176) which already cuts
~3x via memoization.

### Effort estimate

- ~1-2 hours: implement helper, write 4-6 unit tests for edge cases
  (constant curve = no need to bisect; equal byte-proxies = degenerate λ_max).
- Risk: low (pure refactor of a deterministic loop).
- Trigger to advance: any time the bisect wall-clock becomes a noticeable
  fraction of total tool runtime (currently <5%; not worth doing eagerly).

## S6 — Extract `tac.deploy.lightning_archive_eval` from 1537-LOC dispatcher

### Current state

`experiments/lossy_coarsening_lightning_cuda_test.py` (and its
arch_shrink/admm_step6 siblings) contain ~1537 LOC of mixed dispatcher +
build + helper code. The "build the byte-closed archive + smoke roundtrip
+ optionally launch Lightning + harvest" responsibilities are intertwined
with no shared library boundary.

The PR101 build helpers (`_build_inner_blob`, `_split_pr101_inner_blob`,
`_write_pr101_archive`, `_stage_forked_submission_dir`, etc.) are
imported into 3+ build tools today via `importlib.util` + module probing,
which is brittle.

### Recommended improvement

Extract a `tac.deploy.lightning_archive_eval` module containing:

- `class LightningArchiveBuilder`:
  - `build_inner_blob(decoder_bytes, latent_blob, sidecar_blob) -> bytes`
  - `write_pr101_archive(inner_blob, archive_path) -> None`
  - `stage_forked_submission_dir(submission_dir, pr101_source_dir) -> None`
  - `read_pr101_inner_blob(archive_path) -> bytes`
  - `split_pr101_inner_blob(inner_blob) -> Tuple[bytes, bytes, bytes]`
- `class LightningDispatchActuator`:
  - bootstrap-call wrapper (no inline dep install per CLAUDE.md
    `forbidden_remote_bootstrap_inline`)
  - per-dispatch / total-cost gating
  - heartbeat/watchdog wiring
- `class LightningResultHarvester`:
  - JSONL harvest from instance log streams
  - contest_auth_eval.json validation

Each tool currently doing `importlib.util.spec_from_file_location(...)`
becomes `from tac.deploy.lightning_archive_eval import LightningArchiveBuilder`.

### Effort estimate

- ~150 LOC of new tac module + ~100 LOC of unit tests.
- Risk: medium — changes import boundaries; need careful audit of every
  caller's expectations (e.g., `_read_pr101_inner_blob` returns bytes vs
  Path; `_stage_forked_submission_dir` writes vs returns; etc.).
- 3-4 hours of work + adversarial review for the abstraction surface.
- Trigger to advance: when adding a 4th build tool that needs the same
  helpers, OR when a CRITICAL bug surfaces in one of the importlib-spec
  copies and would require synchronized fixes across all 3+ tools.

## Why these are queued (not landed)

- S4 is low-priority — bisect wall-clock is already small and S2 already
  buys back most of the time.
- S6 is medium-effort + medium-risk — refactor across 3+ files;
  appropriate when there's a fourth caller to amortize the cost.

Both items respect the FIX-ALL-FINDINGS mandate of "don't gold-plate";
they are explicitly out-of-scope for this commit cycle and queued for a
future research-lane subagent.

## Cross-references

- `feedback_review_eng_findings_*.md` (TBD by REVIEW-ENG council)
- Commits closing C1-C4: 266fa12c, 0b24e5d1, e61d385c, f2ba8205
- Commit closing Dykstra naming: f4f6270c
- Commit closing Phase 4 memo (Ballé + MacKay): abc991c5
- Commit closing S2: 0c736176
- This memo (queued S4 + S6): commit pending.
