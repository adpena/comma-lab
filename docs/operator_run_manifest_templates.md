# operator run manifest templates

This repo already treats manifests as the durable handoff between launch,
polling, and reporting. The files in `configs/run_manifests/` are
placeholder templates only. They are meant to be copied into real per-run
records when a job is actually launched.

## Goal

Keep Kaggle, Modal, and Coiled runs compatible with the existing scheduler
and reporting surfaces:

- `remote_job.py`-style manifest fields stay flat and readable
- `status` remains explicit instead of inferred from chat or shell history
- checkpoint summaries stay separate from the launch command
- no active runs are invented in the template layer

## Canonical fields

Use the same core fields everywhere:

- `slug`
- `host`
- `platform`
- `status`
- `remote_root`
- `remote_log`
- `remote_command`
- `best_checkpoint`
- `latest_printed_checkpoint`
- `manifest_path`
- `status_path`
- `written_at`

The scheduler/reporting layer can then attach platform-specific metadata
without changing the basic shape of the record.

## Status conventions

Use stable, human-readable states:

- `draft`
- `queued`
- `launching`
- `running`
- `completed_no_promotion`
- `completed_close_miss`
- `failed`
- `paused`

Avoid inventing states that are really just notes. If a run needs extra
detail, put it under `notes`.

## Template usage

1. Copy the relevant template from `configs/run_manifests/`.
2. Fill in the concrete command, log path, and platform-specific block.
3. Write the real manifest beside the run or into the existing
   `.omx/logs/remote_jobs/` ledger.
4. Keep the matching status file separate so polling can update it without
   rewriting the launch intent.

## Platform notes

### Kaggle

Treat Kaggle as a notebook-style launcher with a persistent working
directory. The manifest should point at the notebook workspace and the log
path inside that workspace, not at a fabricated remote shell path.

### Modal

Treat Modal as an app-driven remote runtime. Keep the app name, function
name, and image reference in the platform-specific block. The common launch
fields should still point at the job log and state record.

### Coiled

Treat Coiled as a cluster-backed Python runtime. Keep the cluster and
environment details in the platform block, but still expose the same common
launch fields for reporting.

## Reporting fit

The existing reporting surfaces read best when the manifest tells the whole
story:

- what was launched
- where the logs live
- whether the run is active
- which checkpoint was best so far
- which checkpoint was last printed

That makes the templates useful for both scheduler status pages and manual
operator polling.

## Template index

- `configs/run_manifests/kaggle_run_manifest.template.json`
- `configs/run_manifests/modal_run_manifest.template.json`
- `configs/run_manifests/coiled_run_manifest.template.json`
- `configs/run_manifests/run_status.template.json`
