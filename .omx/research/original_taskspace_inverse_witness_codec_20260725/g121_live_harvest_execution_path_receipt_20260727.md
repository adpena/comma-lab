# G111 -> G121 live harvest execution-path receipt

Date: 2026-07-27  
Axis: `[encoder-side macOS CPU orchestration; no score authority]`  
Task: `pact-g121-live-stage-harvest-handoff-20260727`  
Status: implementation landed and focused tests green; full-n600 G120 launch
not fired; pointer unchanged at 0.172

## Audit verdict

The pre-fix G111 -> G121 production path was open, not closed:

1. `harvest_g111_stages_v1` had no production CLI or caller.
2. It took one receipt snapshot and emitted
   `EXHAUSTIVE_STAGE_HARVEST_COMPLETE` even when a live producer could append a
   later physical stage.
3. It supplied the newest governed launch DSL hash while reopening every old
   receipt. After a governed same-directory resume, valid pre-resume nodes carry
   the earlier launch hash and were rejected instead of being admitted through
   the newest tip's recursive ancestry.
4. It admitted every immutable physical node, including retained periodic
   checkpoints, although the G121 contract admits preserved stage/final nodes
   only.

## Landed execution path

`tools/run_taskspace_g121_live_stage_harvest.py` is the production monitor.
Once it is started, it polls the real run directory and automatically invokes
the non-exhaustive G121 entrypoint at the earliest complete preserved stage:

`levelset_ckpt_stage*_ep*.npz + levelset_resume_stage*_ep*.npz`

The pair is eligible only when both files are stable, reopen as a complete
fresh-producer checkpoint, and recompute the exact checkpoint ID of one node in
the recursively reopened current-tip physical ancestry. The alias NPZ container
SHA need not equal the physical copy because the trainer performs two separate
atomic `np.savez` calls; the semantic checkpoint identity must be exact.
`levelset_periodic_*`, rolling aliases, and trainer BEST remain inadmissible.

While G111 continues, the monitor calls:

```python
harvest_g111_available_stages_v1(
    producer_run_dir=RUN,
    expected_launch_manifest_sha256=EXTERNAL_SHA,
    output_dir=RUN / "g121_harvest",
    progress_dir=RUN / "g121_progress",
)
```

This may append exact checkpoint-keyed G120 work to
`g121_stage_measurements.jsonl`, but its return type hard-codes
`exhaustive_enumeration_proven=False` and it cannot publish
`g121_completion_receipt.json` or `g121_retained_prepose.json`.

Only after `levelset_train_result.json` names the current physical final-tip
receipt/checkpoint and the preserved-stage census remains unchanged across the
harvest does the monitor call terminal `harvest_g111_stages_v1` and allow the
exhaustive reductions.

## Exact production invocation

For a governed real G111 run on the SSD tier:

```bash
.venv/bin/python tools/run_taskspace_g121_live_stage_harvest.py \
  --producer-run-dir /Volumes/VertigoDataTier/pact/G111_REAL_RUN \
  --expected-launch-manifest-sha256 MANIFEST_SHA256 \
  --g120-dry-run-receipt /Volumes/VertigoDataTier/pact/G111_REAL_RUN/g120_dry_run_gate/g120_governed_clean_dry_run_receipt.json \
  --expected-g120-dry-run-receipt-sha256 G120_DRY_RUN_RECEIPT_SHA256 \
  --output-dir /Volumes/VertigoDataTier/pact/G111_REAL_RUN/g121_harvest \
  --progress-dir /Volumes/VertigoDataTier/pact/G111_REAL_RUN/g121_progress \
  --poll-seconds 30
```

`MANIFEST_SHA256` is an external value computed from the exact current
`launch_manifest.json`; the tool reopens those bytes before registering the
launch epoch. The dry-run receipt SHA is also external; the tool reopens that
receipt against the exact producer manifest, output/progress/cache paths, and
current G120/G121 source bytes before monitor binding or launch-epoch
registration. On governed resume, rerun the two-process dry-run in a fresh gate
directory, then invoke the monitor with the new externally computed manifest
and receipt SHAs. If the original monitor is still live, the second invocation
appends the new launch epoch and exits; the fcntl-singleton monitor adopts it.
The fixed monitor binding refuses a different producer path, so an old G111
payload population cannot be silently substituted. A resume after terminal
G121 reductions already exist must use a fresh output/progress pair, preventing
stale “complete” files from remaining visible during a new treatment.

Reuse is limited to same-producer, exact physical checkpoint identities:
G112 partitions are checkpoint-ID-addressed, G120 reuse requires the prior
measurement receipt path plus external SHA, and a different producer cannot
reuse the output ledger.

## Launch authority boundary

The monitor itself was not auto-spawned beside the active G111 dry-start and no
full-n600 G120 job was launched. The formerly open gate is now closed for the
exact v6 producer launch and current source bytes by
`g120_governed_clean_dry_run_gate_receipt_20260727.md`, receipt SHA-256
`178907cd677d2d6a2e2b6a3a394e110778a2d3c524365e5db38db786882ffc65`.
That receipt proves the production wrapper's physical SSD storage preflight and
two-process batch resume with zero scorer calls. The next authorized action,
when the G111 producer is ready for scorer-carrying monitoring, is the literal
receipt-bound G121 monitor invocation recorded there.

## Triality

DSL:

`G111CurrentTipAncestry -> PreservedStageSnapshot* -> G121IncrementalLedger`

and only

`G111TerminalTipProof -> ExhaustiveG121Reduction`.

DAG:

`governed G111 launch/resume epoch -> immutable stage node -> external G121
monitor -> G112 -> G120-v2 -> terminal census seal -> G119`.

Equation:

`eligible(t) = preserved_alias(t) intersect ancestry(current_tip(t))`;

`exhaustive = terminal_tip_bound and census_before == census_after`.

## Verification

- `ruff check` passed on the G121 module, monitor, and focused tests.
- `21 passed` across the G121 contract and live-monitor test files.
- No MLX training, n600 scorer replay, exact evaluation, candidate promotion,
  or pointer move was claimed.
