Implemented and committed the full watched-launch hardening charter as commit `6504608bd5a973ecc761d7edfec6f49baafecc76`.

Key outcomes:

- Hardened launcher with fresh-root protection, verified niceness, receipt identity/reservation/tombstones, derived resource envelopes, and atomic watcher arming.
- Promoted generic liveness and quality watchers with checked-in rx2-equivalent configs.
- Added real-monitor suppression for launch-adjudicated failures.
- Added advisory capability registry and MAIN-handoff charter lint.
- Added the full defect→cure→control [receipt memo](/Users/adpena/Projects/pact/.omx/research/ddm_lh1_watched_launch_hardening_20260814.md).
- Preserved all unrelated dirty work; the committed target files are clean and the staged index is empty.
- Did not touch live rx2 PIDs, watcher alerts, receipts, or run directories.

Verification: 95 scoped tests passed, both watcher configs validated, capability lint produced the expected warning, compilation and Ruff passed, and two review-tracker passes completed. One unrelated pre-existing dirty-worktree MX1 test failure observed during broader exploration was left untouched.

This was scorer-free apparatus work. No production-duration soak, Metal execution, candidate, or scorer measurement occurred. Own-vehicle frontier remains `S 0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: next new detached-run owner; consumer store: the new durable run directory plus `.omx/tmp/codex_runs/`; fire trigger: the next eligible local long-running launch after commit `6504608bd5`; action: launch through the hardened surface with fresh-root, unique receipt, measured resource inputs, and both watchers. Do not retrofit the live rx2 run.

## LIVE-HYPOTHESES

- A real long-duration watched launch will remain single-alert under early failure and monitor restart because launch verification gates watcher release and receipt acknowledgements require a delivering monitor channel; this is controlled in subprocess tests but not yet production-soaked.
- Non-rx2 quality policies should be expressible without another bespoke script because paths, fields, thresholds, knee behavior, and conditions are configuration-driven; only exact rx2 semantics were exercised.
- The derived 116 GiB resource envelope should use the host effectively without the old artificial 16 GiB cap because it delegates enforcement and aggregate admission to the existing governor; no heavy job was launched in this scorer-free unit.

## DEAD-ENDS

- Reusing one unconsumed receipt name across launches is closed: active reservations refuse even explicit supersede, while stale artifacts require preserved supersession.
- Manually arming watcher scripts after launch is closed for new runs: `--arm-watchers` creates both before command release.
- Re-alerting a verify-alive failure is closed: the receipt is launch-adjudicated, waiting watchers are stopped, and the fleet reader suppresses it.
- Copying WC1’s 16,384 MiB cap is closed: resource mode derives the enforced ceiling from measured demand and canonical governor policy.
- Spending another arm merely rediscovering denied Metal or priority controls is closed within the recorded sandbox scope: charter lint now warns and names the MAIN handoff.