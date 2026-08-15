# DDM LH1 watched-launch hardening

**Disposition:** `LANDED-SCORER-FREE / NEW-LAUNCHES-ONLY`.

The detached-launch path now refuses stale roots and ambiguous receipt reuse,
can apply and verify niceness, can derive and enforce a real resource envelope,
and can arm generic liveness and quality watchers before the job is released.
No live rx2 PID, alert, watcher, receipt, run directory, or payload was changed.
The current rx2 processes and `rx2_*` / `wc1_profile_done` receipts remain on
their old shape until they finish.

Axis: `[apparatus / scorer-free]`. This work created no candidate, ran no
scorer, and did not move the exact frontier.

## Defect to cure to control

| measured defect | permanent cure | executed control |
|---|---|---|
| D1: the WC1 attempt entered a nonempty root and the trainer correctly refused `fresh run output already exists` | repeatable `--fresh-root` checks every declared root before launch; `--fresh-root-suffix` mints a UTC-suffixed sibling and rewrites the real argv/output paths; neither path deletes prior bytes | a nonempty root returns rc 7 without mutation; suffix mode writes only below the minted root and preserves the prior sentinel |
| D2: one done-receipt name described two launches and verify-alive produced a second asynchronous alert | every JSON done receipt carries manifest path, PID, and a monotonic launch counter; an unconsumed name returns rc 6; explicit supersede preserves the old bytes and a tombstone; launch-adjudicated receipts are suppressed by the real `codex_arm_watch.py` reader | collision refusal, dry-run nonmutation, explicit tombstone, structured identity, and real-reader suppression all pass; the combined early-death/auto-watcher control proves watchers remain behind a launch-verified gate and cannot create a duplicate alert |
| D3: arms rediscovered denied Metal and priority controls | `.omx/state/codex_arm_capabilities.json` records the two measured families and receipt provenance; queue add, spawn, and explicit lint emit advisory-only MAIN-handoff warnings | fixture charter demanding verified niceness emits `charter-lint WARN`, names `process_priority_control`, and names the MAIN handoff; the actual LH1 charter emits the same family warning |
| D4: rx2 watcher organs lived in session scratch and were armed manually | generic config-driven `tools/run_liveness_watcher.py` and `tools/run_quality_poller.py`; `--arm-watchers` validates and spawns both before job release; checked-in rx2 configs preserve the old thresholds and field semantics | fake dead PID produces an atomic alert and a fleet-visible structured event receipt; rx2 bar 186,073 at epoch 33 alerts at 186,074; auto-arm returns two live watcher identities |
| Addendum 2: WC1 copied a 16,384 MiB cap despite a 116 GiB machine policy | `--derive-resource-budgets` requires a measured peak, measured thread need, and wall cap; it reads the canonical governor ceiling, caps threads by available CPUs, and wraps the real command with `safe_run.py`; missing measurements remain explicitly child-owned instead of receiving invented limits | dry-run receipt derives 118,784 MiB from the current 116 GiB policy, carries 12.5 GiB measured demand and two measured threads, and contains no 16,384 literal |

The launcher uses two gates. The command gate opens after the manifest,
niceness, and watcher processes exist. The watcher gate opens only after the
command survives the verify-alive window. A synchronous launch failure stops
the waiting watchers and marks the terminal receipt
`adjudicated_at_launch=true`; this closes the initially found double-alert race.

## Canonical watcher migration

New runs should declare unique output/receipt paths, a fresh root, measured
resource inputs for heavy work, and both configs in the initial launch command:

```text
.venv/bin/python tools/launch_detached_process.py \
  --output-dir <durable-run>/detached \
  --fresh-root <durable-run> --fresh-root-suffix \
  --done-receipt <unique-name> \
  --nice 10 \
  --derive-resource-budgets \
  --measured-peak-rss-gib <receipt-backed-peak> \
  --measured-thread-need <profile-backed-threads> \
  --walltime-cap-s <bounded-wall> \
  --arm-watchers \
  --liveness-config <liveness.json> \
  --quality-config <quality.json> \
  -- <real-command-and-argv>
```

`configs/watchers/rx2_liveness_watcher.json` reproduces the old 120-second
initial/poll cadence, 600-second receipt ceiling, and 45-minute checkpoint
ceiling after a 45-minute artifact grace. `configs/watchers/rx2_quality_poller.json`
reproduces the 186,073 joint-byte bar beginning at epoch 33, QAT knee at epoch
31 with a three-epoch/1.25x shock window, nonfinite `bpp` detection, three
2,880-second stale periods, and the discrete-QAT best-not-latest informational
row. These files are migration references only; LH1 did not execute them
against the live rx2 process.

The resource mode does not infer a workload peak from host capacity. It
requires receipt-backed demand, then uses the governor's current 116 GiB
operator ceiling as the real process-group cap and passes the measured
projection into system admission. Thread variables are set to the smaller of
the measured need and available logical CPUs. Niceness plus the admission
governor handle contention; the tool does not manufacture a low fixed cap or
idle the machine.

## RECALL EVIDENCE

Searched the full research corpus, canonical equation registry, research
index/DAG surfaces, specs, and queue/hot-state rows with the content queries
`detached launch`, `done receipt`, `outer rc`, `silent failure`, `verify-alive`,
`fresh root`, `nice setpriority taskpolicy launchd`, `Metal denied`, `watcher`,
`m79`, `116 GiB`, and `thread budget`.

Beyond the charter seeds:

- `.omx/research/ddm_rr8_20260806/ROUND8_FINDINGS.md` and canonical equation
  `ddm_rr8_stage_rc_success_contract_v1` establish that an outer detached rc is
  not stage-chain success when a wrapper swallows a failing inner rc. This
  changed the receipt work from a text-format tweak into a structured direct-
  child identity contract. It does not claim to repair a shell wrapper that
  itself returns the wrong rc.
- `.omx/research/ddm_ac1_automatic_endpoint_closure_20260814.md` supplied the
  current one-shot endpoint/receipt closure idiom. This changed watcher alerts
  into fleet-consumable event receipts rather than standalone alert files that
  MAIN would still need to poll.
- canonical equation `ddm_rr9_mem_probe_fire_protocol_v1` separates system
  admission arithmetic from a required measured Metal load-stage receipt.
  Therefore LH1's derived resource envelope does not claim that a projected
  peak proves Metal readiness.
- The bounded capability search found a durable Metal denial receipt and the
  WC1 priority-control denial receipts. It did not find a durable git-push
  denial receipt in the searched scope, so the registry does not declare that
  family denied.

Primary defect anchors:

- WC1 attempt log:
  `/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/detached_main/run.log`,
  SHA-256 `4b55b54a9bec1b7474c6f5621fd9189054c0b91cc9113d50099693a8953f8a9d`.
- WC1 failed-attempt status:
  `/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/safe_run_status_main.json`,
  SHA-256 `1ac2deda0940df7bd123ac35ba87f9e49a7facafa6317c45f793866983a34490`.
- WC1 denied-priority attempt status:
  `/Volumes/APDataStore/pact/ddm_wc1_hpac_throughput_port_20260814/profile_cpu_reference/safe_run_status.json`,
  SHA-256 `2df49c05e71122127eb435aa9f33a5aef7e693542fc1f2c9ac8db45fb32ffcc4`.
- Metal denial receipt:
  `.omx/research/ddm_mp2_20260809T105302Z/RECEIPT.json`, SHA-256
  `69791a8cdca7c4321dbcd13b22312528d0d6b5b9d56d60dba434b4d676d6c41d`.
- Promoted source semantics:
  `.omx/tmp/rx2_quality_poller.py` SHA-256
  `51bd777f1114555c1e7dad93ea4d0931e6b9c5aaf7978773f4d59ccacc1f0f13`;
  `.omx/tmp/rx2_liveness_watcher.py` SHA-256
  `e92e0fa651fb05e8b5f0adc61915939f6448064df70dad93673e8c2decda1e0f`.

## Verification and boundaries

The final focused suite covers the canonical launcher, both watcher organs,
the real fleet read path, arm queue/relay behavior, endpoint closure, SIGURG
hardening, and the no-silent-failure static gate. Config validation, Python
compilation, Ruff (with the file's pre-existing C408 excluded), and
`git diff --check` also pass. Exact command/results and post-edit content hashes
are in the serializer landing receipt.

The tests use real subprocesses, real process niceness APIs through a capability
fixture, real filesystem publication, and the production monitor parser. They
do not constitute a long-duration production soak, a scorer result, a Metal
capability result, or proof that an arbitrary shell wrapper propagates its
inner stage rc.

Own-vehicle frontier: unchanged at `S 0.1619344578804448 @ 186,269 B`
`[contest-CUDA T4, n600]` (MC36 Variant C); LH1 is scorer-free apparatus work.
