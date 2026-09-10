# Root review receipts

## Pass 1 — producer CLI and measurement runner

Read the new producer argument path through build_seal and its consuming validator. The CLI
requires exactly one measured/inherited leg and passes the same object to the seal builder.
JSON/type failures are refused before writing a seal. The self-validator requires the leg.
Read the actual public receiver shell, f26 decode stages, native checkpoint writer and final
raw report: the timing runner invokes the unchanged copied shell, disables shared cache and
alternate HPAC, retains token checkpoints and raw output, and records the advisory CPU switch.
No scorer is imported or invoked by the child. Ruff passed and review_tracker marked both files.

Found during launch: ps raises PermissionError rather than returning a nonzero subprocess
status. Fixed before token decode, retaining the failed launch and prelaunch binding. Current
runner records unavailable concurrency, never a fabricated zero. First meaningful timing run
is fresh and has its own binding and launch receipt.

## Pass 2 — resume, retention, authority and identity

Reviewed producer CLI again after source API alignment: measured and inherited paths cannot
bypass the same validator. Reviewed runner fresh-start scope and actual stage report. Any
resumed run is explicitly labeled and is excluded by the timing contract's cold-start guard.
A producer/HEAD change causes a safe resume refusal; it does not turn suffix time into full time.
Two clean full outputs plus token fields consume 7,560,748,800 bytes, leaving 1,029,185,792 bytes
before the 8 GiB ceiling. Observed checkpoint files are small compressed states; no retry with
an interrupted large raw is authorized without additional reservation. Existing generic shell
build scratch is automatically removed by its own trap. All materialized candidate payloads
are retained on SSD. Ruff passed and both files received a second review_tracker mark.

## Independent contract review findings

- Receiver digest initially included archive bytes, defeating inheritance; fixed to separate
  payload identity and normalize only explicit archive-pin literal spans.
- Calibration initially bound only the remote archive; fixed to remeasure the remote-runtime
  digest using the exact existing upload projection helper.
- Inherited source lookup could mask a measured T4 failure with missing concurrency; adjusted
  to load its paths without margin acceptance, then strictly validate the source.
- Resumed suffix time needed explicit rejection; cold-start and resume guards added.
- Ratio transfer across different local hosts needed refusal; requested same-host validation.
- tc4's observed remote timeout must veto a deceptively optimistic CPU projection; a hashed
  candidate T4 receipt now binds archive/runtime/hardware, timeout command and configured limit.

Shared-assumption challenge: the charter assumed move41 would comfortably pass. The actual
T4 decode receipt disproves that premise. The implementation preserves the failed result and
sends MAIN an explicit decision order; it does not change the limit to save the hypothesis.

## Timing retention preflight correction — two passes

Before tc4 launch, reviewed prepare() reserving the exact source proof's raw+token bytes
plus64MiB checkpoint allowance before another attempt. The reserve catches an interrupted
raw plus fresh retry before spawning. A separate reviewer verified both source proof sizes
and the original tc3 producer snapshot hash. The64MiB amount is an observed lineage allowance,
not a universal bound; remaining budget is also reported and post-run bytes are checked.
Both passes were marked and ruff passed. Source1 producer bytes remain retained independently.

## Fire regression review — two passes, execution pending strict flip

First read both dry-run and normal-path controls: they preserve a valid legacy seal but remove
only the new leg, then prohibit subprocess execution and require the typed timing refusal before
any fire manifest. Independent review found that the smoke-waiver test left smoke present,
so its waiver was inactive. Fixed by removing both blocks and checking the exact timing refusal.
Second pass traced allow_missing_public_smoke=True through the generic validator and the new
required timing flag; the smoke waiver can now actually engage without granting a timing waiver.
Ruff and both review marks are complete. Tests remain unclaimed until the guard is flipped last.

## Final source2 retention and physical-byte accounting

Two reviewed storage corrections now use `(st_dev,st_ino)` identity, preventing certified
hardlinks from being counted twice while keeping logical pathname totals visible in timing
receipts. Fresh-output reservations are still charged in full before launch. An independent
review rechecked both actual hardlink pairs and source2's9317B/hash against tc4 BINDING.json.
Source2 is preserved before the accounting edit; no active timing producer was modified until
tc4 had completed. Components first ran only after its accounting change had two reviews.

The separate move40 runner received two reviews of its unmodified copied-receiver binding,
input-verifier signature, public-shell native compiler flags, four-thread backend call,
2400-second child-group timeout, retained per-stage state, and absence of scorer execution.
A build-only retry is labeled resumed so a cached native build cannot masquerade as a cold run.
Its explicit proxy scope excludes public-shell startup; no copied receiver source is patched.
