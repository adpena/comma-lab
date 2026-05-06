# Research State Tracking Policy

This repo has two layers:

- `tac`: clean reusable task-aware codec library, contest/runtime code,
  canonical contest/runtime preflight, and reusable Python implementation.
- `comma_lab`: research operations, custody, hosted supplements, dashboards,
  provider state, and experiment/replay orchestration.

Claude, OMX, provider logs, public-PR forensics, and recovery ledgers belong to
the comma-lab layer. They are important, but they should not leak into `tac`
APIs unless the code is truly reusable codec/runtime logic.

Reusable Python code belongs in `tac` when it implements a codec primitive,
archive grammar, scorer/eval contract, byte profiler, payload parser,
preflight guard, planning primitive, visualization primitive, or other
contest-relevant algorithm. Thin CLIs may live in `experiments/`, `scripts/`,
or `tools/`, but they should delegate to `tac` modules rather than burying
implementation in ad hoc entry points.

Keep outside `tac`: one-off recovery scripts, provider/job-state harvesters,
Cloudflare/Lightning/Hugging Face publishing glue, raw public-PR intake
workspaces, generated site bundles, Claude/OMX memory processing, and anything
that is not reusable after the contest.

Use `reverse_engineering/` for clean contest-specific public-submission
deconstruction: curated runbooks, byte-anatomy notes, small intake indexes, and
small manifests. Raw PR clones, archives, provider transcripts, and large
artifacts stay in ignored custody locations and are linked from ledgers.
Reusable parsers, profilers, planners, and archive builders still belong in
`tac`.

Preflight is the main exception to the "lab state outside tac" rule: checks
that protect archive validity, inflate/runtime compliance, CUDA score custody,
and package safety are canonical in `tac.preflight`. `comma_lab.preflight`
exists only as an adapter/catalog surface for ARA, reports, hosted supplements,
and dashboards.

## Git Boundary

Track in git:

- small `.omx/research/**/*.md` ledgers;
- small `.omx/research/**/*.json` structured summaries;
- selected small `.omx/state/*.md` control-plane files;
- source docs under `docs/` and paper/site source manifests.

Do not track raw:

- `.omx/state/*.json` provider/job state;
- `.omx/logs/` and `.omx/tmp/`;
- `.omx/auto_memory_snapshot_*/` raw backups;
- `reports/raw/` and `reports/private/`;
- generated `reports/graphs/public_site/` bundles;
- large media, archives, checkpoints, and rebuildable binary artifacts.

For ignored-but-interesting material, write a small dated ledger in
`.omx/research/` and, when useful, a committed external-artifact manifest
pointing to Hugging Face, Lightning, Cloudflare, or another public host.

## Audit Command

Run:

```bash
python tools/audit_research_state_tracking.py \
  --repo-root . \
  --json-out .omx/research/artifacts/research_state_tracking/latest.json \
  --md-out .omx/research/artifacts/research_state_tracking/latest.md
```

The implementation lives in `src/comma_lab/research_state.py`. That is
intentional: research custody is a comma-lab concern, not a `tac` library API.
