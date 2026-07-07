---
name: witness-status
description: One-shot honest status of the live level-set witness training run — alive/dead, best d_seg + epoch, file freshness, RSS and free memory. Use when asked how the witness run is doing, to check in on training, or before any decision that depends on run health.
---

# Witness run status check-in

Run the canonical read-only status tool:

```bash
.venv/bin/python tools/witness_checkin.py
```

For machine-readable output (or when you need field-level detail):

```bash
.venv/bin/python tools/witness_checkin.py --json
```

## Reporting protocol (grounded — no over-promising)

1. Lead with the tool's single output line, verbatim. It is the evidence;
   your report must not claim more than it shows.
2. If the line contains `DEAD` or `⚠ STALE`, that IS the headline — say it
   first, plainly. Never soften a dead run into "may have finished" without
   checking for a completion artifact.
3. `best d_seg` is the best-so-far EMA verdict, an advisory `[macOS-MLX
   research-signal]` — NOT a score. The exact pointer moves only through a
   byte-closed `upstream/evaluate.py` row; say "pointer unmoved" if asked
   about the score.
4. Freshness caveat travels with the number: `resume Xm ago` is the
   liveness signal; telemetry (`costate_shadow.jsonl`) updates on a slower
   cadence, so an old telemetry age alone is NOT evidence of a stall.
5. One line of interpretation maximum (e.g. "descending, ~Nx above the
   ~0.001 goal") — derived from the reported numbers only.

Do not gate, restart, or touch the run from this skill. It is observability
only.
