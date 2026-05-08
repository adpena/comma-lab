# Generated Custody Source Disposition - 2026-05-08

generated_at_local: 2026-05-08
operator: Codex
branch: main
scope: no-signal-loss source hygiene for untracked source-like files inside raw custody roots

## Decision

Do not track raw generated custody trees. Source-like files below these roots are
explicitly dispositioned in
`.omx/research/untracked_source_dispositions_20260505_codex.json` by prefix:

- `.omx/research/artifacts/`: `ignore_rebuildable`
- `.omx/state/`: `ignore_private`
- `experiments/results/`: `ignore_rebuildable`
- `outputs/`: `ignore_rebuildable`
- `reports/raw/`: `ignore_private`
- `reports/private/`: `ignore_private`

This closes the audit blind spot where runtime packets, `inflate.py`,
`inflate.sh`, generated `src/*.py`, manifests, provenance JSON, GHA reports, and
other source-like files under ignored custody roots could be neither tracked nor
dispositioned.

## Snapshot

Manual pre-change scan of untracked generated-custody source suffixes found 178
source-like files, all under `experiments/results/` in the current status
surface. Representative clusters:

- Lightning/auth-eval custody JSON under `experiments/results/lightning_batch/`
- ADMM/lossy coarsening `submission_dir/{inflate.py,inflate.sh,src/*.py}`
- PR101 CodeCop `full_runtime_packet/{README.md,inflate.py,inflate.sh,src/*.py}`
- Monolithic stack candidate manifests and runtime-consumption proofs
- PR107 GHA report text and local CPU-eval shell/report files
- Sub017 factorized HNeRV runtime packet copies
- Unified winner stack runtime packet copies

No raw `experiments/results/`, `reports/raw/`, provider state, runtime packet, or
payload directory is promoted by this ledger. The durable action is the manifest
policy plus the tool change that makes these files visible in the audit summary.

## Required Promotion Rule

If any generated-custody source-like file becomes authoritative, promote the
smallest sanitized source-of-truth surface instead of the raw directory:

1. move or recreate reusable source in `src/`, `tools/`, `experiments/`, or
   `docs/runbooks/`;
2. summarize evidence in `.omx/research/*.md` or `.omx/research/*.json`;
3. leave the raw custody file under its ignored root with this prefix
   disposition, unless an explicit reviewed manifest supersedes it.

## Verification

Run:

```bash
.venv/bin/python tools/audit_untracked_source_artifacts.py \
  --repo-root . \
  --disposition-manifest .omx/research/untracked_source_dispositions_20260505_codex.json \
  --format json \
  --strict
```

Expected policy state: strict audit remains ready while reporting generated
custody source-like files as dispositioned, not invisible.

Verified post-change:

- `ready_for_no_signal_loss_canonicalization`: `true`
- `untracked_source_like_count`: `182`
- `generated_custody_source_like_count`: `178`
- `dispositioned_count`: `182`
- `undispositioned_count`: `0`
- `invalid_disposition_count`: `0`
- `by_disposition`: `ignore_rebuildable=178`, `track=4`

The four current `track` dispositions are the two small research ledgers plus
the candidate evidence contract helper and its focused regression test.

## Runtime Source Baseline Hardening Addendum

Adversarial review found the prefix disposition was too broad for executable
runtime packets: a new `experiments/results/.../submission_dir/inflate.py`
could have been hidden by the `experiments/results/` prefix alone. The audit
now treats generated runtime source as a stricter sub-class. Runtime source
under a broad prefix requires either:

1. an exact per-artifact disposition entry, or
2. a pinned `runtime_source_baseline` on the prefix entry.

Current `experiments/results/` runtime-source baseline:

- `algorithm`: `pact_runtime_source_set_v1`
- `count`: `81`
- `sha256`: `b31fd76d44c800135e2a5b28308a49d15a3a75e746e6b829ae81eb0ceab54223`

This preserves raw custody signal without promoting raw runtime packets to
canonical source. Any newly generated submission/runtime packet now changes
the baseline count or hash and fails strict audit until it is either
summarized/promoted, exactly dispositioned, or intentionally rebaselined.
