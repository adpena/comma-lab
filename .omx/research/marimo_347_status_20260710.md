# Task #347 (marimo/molab Notebook Competition #2) — status disposition, 2026-07-10

**Disposition: RESOLVED-VERIFIED. No action remains.** This is a pointer/summary note, not a
rebuild; the full investigation + verification already landed in
`.omx/research/marimo_linkfix_20260710.md` (see its `# RESOLUTION` section) and
`.omx/state/deferral_ledger.md` row D10. This note exists only to give #347/#407 a single,
easy-to-find disposition line per the task's request.

## Facts (MEASURED / DERIVED from the artifacts above, not re-verified in this pass)

- **Deadline:** 2026-07-09 11:59 PM PST — has PASSED.
- **Entry:** "The Witness Machine" (`notebooks/witness_machine_v12.py`), a standalone public
  notebook in `github.com/adpena/witness-machine`, served via molab's GitHub-proxy URL:
  `https://molab.marimo.io/github/adpena/witness-machine/blob/main/notebooks/witness_machine_v12.py`
  (unchanged throughout — link never re-minted, only content at that path was fixed).
- **Publication timing (pre-deadline):** commits `f58568d` (2026-07-09 21:51 PST) and `be424ad`
  (2026-07-09 23:08 PST) landed the notebook content before the deadline.
- **Runtime break (pre-deadline, discovered after publish):** the published revisions assumed a
  repo checkout for internal-package imports; molab's GitHub-mirror execution has no checkout, so
  the notebook failed to bootstrap on molab's server at run time.
- **Repair-in-place (post-deadline):** commit `f111248` (2026-07-10 05:22 PST) made the notebook
  self-bootstrapping — downloads + sha256-verifies a sealed release bundle (`v1.2.0-rc2`,
  3,704,001 bytes) when no checkout exists, drops the static internal-package imports, adds
  cold/warm/self-heal regression tests, and commits a clean cached-output snapshot. Same URL.
- **E2E verification (this session, `marimo_linkfix_20260710.md`):** anonymous visitor view loads
  clean (no console errors), static preview renders 42/42 cells with cached outputs and zero
  tracebacks, the release-bundle asset chain is byte/sha exact, and a molab-faithful local cold
  bootstrap (fresh cache, no checkout) exports all 42 cells with exit 0 and zero runtime
  exceptions. The ONLY step that did not complete: clicking "Run it now" on molab's server
  requires sign-in (anonymous server execution is not permitted by molab) — not attempted, per
  the no-credentials rule for this agent.
- **Whether judges see the pre-repair (broken) or post-repair (working) runtime state** at
  evaluation time is out of our control — we only ever changed content at the fixed URL, never
  the link itself, per the operator's hard constraint.

## Remaining action

**None identified.** The link is fixed in place, durably documented in `paper/README.md` (§
"Published molab entry"), verified end-to-end short of an authenticated server run (which
requires the operator's own signed-in session — not something this ledger can or should acquire).
Deferral ledger row D10 already carries the terminal `✅ RESOLVED-VERIFIED 2026-07-10` status.

## Sources consulted (proactive recall, per CLAUDE.md)

- `.omx/research/marimo_linkfix_20260710.md` (full investigation + `# RESOLUTION` + E2E evidence)
- `.omx/state/deferral_ledger.md` row D10
- `paper/README.md` § "Published molab entry (marimo Notebook Competition, task #347)"
- `.omx/state/lane_registry.json` lane `lane_witness_molab_release_rc_20260710` (level 0 — the
  lane-registry level reflects that this is a verification/release-checkpoint lane, not a
  production-hardened substrate lane; it does not indicate the molab task itself is unresolved)

Pointer 0.19110... UNMOVED — this is a disposition-only note; no code, no launches, no
competition-form action taken.
