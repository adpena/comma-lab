# ddm_pr10 — second-family review of the quiesced decode-timing rule

Date: 2026-09-10  
Axis: `[review; receipt re-derivation; macOS-CPU timing; scorer-free]`  
`score_claim=false`  
Tokens: `[no-triality] [p0-ledger-ok]`

## Verdict

**RERUN-REQUIRED.** The final measured-impact rule is not an admissible calibration rule for these
receipts. It was fixed after both real runs were available, replaces the rule recorded by the sampling
producer, and uses the run's own median to erase any slowdown common to most instrumented stages. It
then calls the first run `quiesced` despite recorded external work near one core for minutes and despite
only 62.00% of its wall being checkpoint-bounded.

The receipt-visible stage excess in attempt 1 is 5.658483744 s, or 0.709842851% of its 797.145979125 s
wall. That quantity is well inside the 540 s reserve between the 1,260 s policy limit and the 1,800 s
job wall, but it is not a bound on total contamination: a uniform pace shift disappears into the
self-median and 37.996% of the wall has no stage-rate measurement. Attempt 2 is refused only because
ChatGPT used 124.2% CPU in `during_0001`, outside the checkpoint span. Its two 100.0% Python processes,
98.3% `tar`, and 98.2% `git` overlap the +9.887% stage 75 and are admitted because aggregate stage
excess is below 1% of wall. The consumer memo's statement that those Python/git processes were in the
native-build interval is not supported by the retained sample/checkpoint epochs.

The move-40 score and its directly measured T4 decode time remain real. This review withdraws only the
claims that attempt 1 proves a quiesced local base and that its `cpu_to_t4_ratio=1.2419981475836523` may
be used for successor projection/inheritance. Move 40's retained exact T4 receipt independently says
990.053829427 s, below 1,260 s; no local calibration is needed to preserve that instance-level fact.

`verdict_scope: FORMULATION` — the present hybrid rule as a timing-admission and calibration rule on
this 18-logical-core/6-P-core host, with these two retained runs. This is not a negative on process
sampling in general, stage checkpoints as diagnostics, move 40's exact score, or its actual T4 time.

## Reviewed objects and provenance pins

Aliases used below:

- `A1` = `/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/receipts/move40_quiesced/CONCURRENCY.json`
- `A2` = `/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/receipts/move40_quiesced2/CONCURRENCY.json`
- `L1` / `L2` = the adjacent `move40_quiesced_local.json` / `move40_quiesced2_local.json`
- `CAL` / `LEG` = the adjacent `move40_quiesced_calibration.json` / `move40_quiesced_decode_wall_clock.json`

| Object | SHA-256 / identity |
|---|---|
| `A1` | `f19efa1a1233e477643d92dcb91d37ec09bbeca5878d936b2fdae1ce126a86bb` |
| `A2` | `040fe2c19399dfe7bd559a7683828273b7d362cba3cbd8e278e3a6af023fd611` |
| `L1` | `9cefbac9f1a003a380d817308f070b6122b99e94303bb78eeac7304fc3bc43fd` |
| `L2` | `edc60626329e7175c86332a3258f60b8cb4aad111dda7671f9e3a0dcf455f02a` |
| `CAL` | `5bea9356aeacd2c01461567a0d4c05df65165aba4a237dbc20806d8fdf2323ac` |
| `LEG` and the admitted sidecar | `e58adf1df4523da645bf1651258ce798daeaac201a71d4ef4d90be9cc0cdb97b` |
| `MOVE40_QUIESCED_LEG.json` | `35326830712435d42ed6c035ab2943277e53c5d835d4ac7a6eebc8e879d2e847` |
| dwc1 memo | `291ff648088fd55206684b417173a08894575e0b40098928485362141f07698b` |
| instrument source | `bf467026f97e938e12a19d11bcff829ce63ff901` |
| hardware-fingerprint follow-up | `fc42f8a52f71df31f78f5cf5d827af42bd85bc7f` |

The chronology is itself evidence of post-selection. `A1.samples[-1].time_utc` is
`2026-09-10T09:28:51Z`; `A2.samples[-1].time_utc` is `2026-09-10T09:48:38Z`. Commit `bf467026f` was
authored at `2026-09-10T09:56:18Z` and explicitly says the rule was fixed after two measured runs
(`src/tac/decode_timing_concurrency.py:14-23`). Commit `fc42f8a52` followed at `10:19:08Z`.

The raw receipt rule in both `A1.rule` and `A2.rule` is
`threshold_pcpu=25.0`, `ancestor_cap_pcpu=50.0`, `aggregate_cap_pcpu=700.0`, and
`visible_pcpu=5.0`; it yields stored counts 3 and 4 and `quiesced=false` for both. The assembled
`L1.concurrency.verdict_rule` and `L2.concurrency.verdict_rule` instead use
`threshold_pcpu=100.0`, `ancestor_cap_pcpu=100.0`, `impact_tolerance=0.01`, and
`stage_tolerance=0.05`. The assembler accepts those fresh CLI values at
`tools/quiesced_decode_timing.py:71-78,136-145` and re-derives a different verdict from the retained
`visible` list at `src/tac/decode_timing_concurrency.py:227-289`.

## Per-claim evidence table

| Charter question | Receipt-derived answer | Verdict scope |
|---|---|---|
| 1. Is own-median impact sound? | **No.** A1/A2 cover 62.004%/62.770% of wall. Their robust stage jitter is 0.481%/0.827% and IQR/median is 0.600%/1.069%; the 5% listing cutoff is above ordinary jitter and finds the two spikes. The 1% wall cutoff is also numerically above the observed summed excess, but it is not a contention bound because the median moves with a sustained slowdown. | `FORMULATION`: own-run-median impact used as authority; not checkpoint timing as a diagnostic. |
| 2. Does the split survive contamination swap? | **No.** Original rule: A1 REFUSE (3), A2 REFUSE (4). Final rule: A1 ADMIT (0), A2 REFUSE (1). A1's highest outside-stage `dasd` sample is 95.9%; A2 has 124.2% ChatGPT. Any outside threshold in `(95.9,124.2]`, including 100, makes this exact split. The stage branch admits much larger named activity whenever self-median excess stays under 1%. | `INSTANCE`: these two receipts under the two recorded rules. `FORMULATION`: phase-dependent swap instability. |
| 3. Does uninstrumented contention escape? | **Yes.** Three independent 99% CPU jobs can consume about three cores while each is below 100 and their 297% sum is below the 700% cap derived from 18 logical CPUs, even though only two P cores remain beyond the four decode threads. Sub-20-second CPU bursts, I/O/memory-bandwidth pressure, and thermal pressure can also escape 20-second `%CPU` samples. | `FORMULATION`: 100/700/20 s rule on this 6-P-core host; not proof that one named escaping class occurred. |
| 4. Gate error direction and size | A contaminated slow local denominator makes successor projections optimistic. A1's receipt-visible envelope is 5.658483744/797.145979125 = 0.709842851% of the recorded wall, or a 0.714917% upward correction relative to a hypothetical 791.487495381 s clean base. At a 1,260 s projected limit that is about 9.008 s, 1.67% of the 540 s policy reserve. Total optimism is **not bounded** by the receipts. A2 being 12.007698916 s faster gives a 1.529% cross-run sensitivity, not a causal or clean-baseline bound. | `INSTANCE`: A1 arithmetic. `FORMULATION`: absence of a total bound. |
| 5. Can the instrument manufacture PASS? | **Yes.** One quiet settle sample is enough; ancestor work up to 100% is excluded; `--pause-pid` records only a PID; and assembly may change the admission rule after the run. A1 waited through 36 busy settle samples, accepted `settle_0037`, then recorded `dasd` around 95.5–96.8% in `during_0016..0029`. | `FORMULATION`: pass surfaces in current source. `INSTANCE`: the named samples and PID 2028 custody in A1/A2. No claim that SIGSTOP itself malfunctioned. |
| 6. Other gate-with-no-door candidates | A bounded static census found **no additional confirmed required-receipt-field case**. Three public validator symbols had only direct test references; inspection removed them from this class, listed below. | `BOUNDED ABSENCE`: 330 tracked top-level `validate_`/`check_`/`require_` definitions outside `preflight.py` and test files, with name-use searched across tracked Python under `src/tac` and `tools`; not a global repository theorem and not a method-level call-graph proof. |

## Independent checkpoint-mtime re-derivation

I read every `stage_*.npz` mtime in both required checkpoint directories and formed consecutive
differences exactly as `stage_rate_deviations` does (`decode_timing_concurrency.py:199-218`). No timing
producer, decoder, scorer, or receipt writer was run.

| End stage | A1 seconds | A1 vs median | A2 seconds | A2 vs median |
|---:|---:|---:|---:|---:|
| 50 | 21.071334362 | -1.037696% | 22.044956923 | +3.167089% |
| 75 | 21.090006590 | -0.950001% | 23.480802774 | **+9.886632% listed** |
| 100 | 21.119748592 | -0.810317% | 21.454753637 | +0.405026% |
| 125 | 21.321965218 | +0.139401% | 21.087273836 | -1.314724% |
| 150 | 21.153526306 | -0.651678% | 21.079798698 | -1.349706% |
| 175 | 21.196909428 | -0.447928% | 21.197307587 | -0.799782% |
| 200 | 21.297195911 | +0.023071% | 21.167489052 | -0.939329% |
| 225 | 21.247272253 | -0.211397% | 21.185463190 | -0.855212% |
| 250 | 21.192252636 | -0.469799% | 21.248942614 | -0.558138% |
| 275 | 21.184626579 | -0.505615% | 21.231879711 | -0.637990% |
| 300 | 21.322813034 | +0.143383% | 21.166393280 | -0.944457% |
| 325 | 26.030774593 | **+22.254499% listed** | 21.190872908 | -0.829896% |
| 350 | 21.826760292 | +2.510190% | 21.275751591 | -0.432676% |
| 375 | 21.305647373 | +0.062764% | 21.406502485 | +0.179218% |
| 400 | 21.265398979 | -0.126264% | 21.369016886 | +0.003791% |
| 425 | 21.291826963 | -0.002144% | 21.386067629 | +0.083586% |
| 450 | 21.292283535 | +0.000000% | 21.368206739 | +0.000000% |
| 475 | 21.361316919 | +0.324218% | 21.393735647 | +0.119471% |
| 500 | 21.287071705 | -0.024478% | 21.438385010 | +0.328424% |
| 525 | 21.300213099 | +0.037241% | 21.371555805 | +0.015673% |
| 550 | 21.311920643 | +0.092226% | 21.333004236 | -0.164742% |
| 575 | 21.372742891 | +0.377880% | 21.493697643 | +0.587279% |
| 600 | 21.422252655 | +0.610405% | 21.461419582 | +0.436222% |

| Statistic | A1 | A2 |
|---|---:|---:|
| wall seconds | 797.145979125 | 785.138280209 |
| checkpoint-covered seconds | 494.265860558 | 492.833277464 |
| covered fraction | 62.004435% | 62.770252% |
| uninstrumented seconds | 302.880118567 | 292.305002745 |
| median stage seconds | 21.292283535 | 21.368206739 |
| median absolute deviation | 0.069033384 s | 0.119264126 s |
| robust sigma (`1.4826*MAD`)/median | 0.480685% | 0.827496% |
| IQR/median | 0.600255% | 1.068660% |
| largest unlisted absolute deviation | 2.510190% | 3.167089% |
| total positive excess over median | 5.658483744 s | 3.250619888 s |
| excess/wall | 0.709842851% | 0.414018775% |

Thus 5% is a defensible **listing** cutoff for these two traces: it is above both robust noise estimates
and every unlisted deviation. The 1% cutoff is above the observed positive-excess sums, but that says
only that the chosen rule admits them. There are two traces, one per condition, no independently clean
replicates, and the threshold was selected after both existed. It has no calibration status.

The cross-run decomposition reinforces the blind spot without assigning cause. A1 is 12.007698916 s
slower overall, but only 1.432583094 s of that difference is inside the checkpoint span; 10.575115822 s
is in the uninstrumented remainder. The within-run excess statistic therefore cannot bound whole-wall
variation.

## Rule comparison and required replacement

| Rule applied to retained samples | A1 | A2 |
|---|---|---|
| Original raw rule: 25% individual, 50% ancestor, 700% aggregate, all non-settle samples | **REFUSE**, stored count 3 | **REFUSE**, stored count 4 |
| Final hybrid: 100% individual outside stages, 100% ancestor, 700% aggregate, stage excess >1% | **ADMIT**, count 0; full-window 100% count also 0 | **REFUSE**, count 1; full-window 100% maximum is 2 |
| Final hybrid with outside threshold 96% | **ADMIT**; A1 outside maximum is 95.9% | **REFUSE**; 124.2% ChatGPT remains |
| Final hybrid with outside threshold at or below 95.9% | **REFUSE** | **REFUSE** |

The exact required rerun rule is conservative admissibility, not retrospective causal estimation:

1. Freeze and serialize the rule **before producer launch**. Add raw-receipt fields
   `admission_rule`, `admission_rule_sha256`, `admission_rule_frozen_at_utc`, and
   `producer_started_at_utc`. Assembly must require the same SHA and copy the rule; an assembler CLI
   may not override it. A mismatch is a typed refusal.
2. Use `threshold_pcpu=25.0` for every non-excluded process across the entire non-settle wall. This is
   the original, pre-data quarter-core threshold; it is deliberately a false-negative-prone
   admission rule rather than a claim that every 25% process causes a measurable slowdown.
3. Use `ancestor_cap_pcpu=25.0` with the same `>=` comparison. Control-plane ancestry is not permission
   to consume a quarter core during a calibration.
4. Use `aggregate_cap_pcpu=200.0`, derived from `(6 P cores - 4 decode threads)*100`, rather than 700
   derived from 18 logical CPUs. This is secondary to the individual threshold but closes the
   many-subthreshold-process class.
5. Apply these process rules to all non-settle samples, including checkpointed stages. Keep
   `stage_tolerance=0.05` and the per-stage table as diagnostics only. Remove `impact_tolerance` from
   the `quiesced` decision; it must never turn known competitors into count zero.
6. Require three consecutive quiet settle samples at the normal 20 s cadence before launch. The
   current one-sample return at `decode_timing_concurrency.py:329-341` is insufficient after A1 and A2
   each began immediately after one isolated quiet sample.
7. Replace `paused_pids: [pid]` with identity and transition custody for every paused process:
   `{pid, ppid, comm, start_time, stop_requested_at_utc, stop_confirmed, resume_requested_at_utc,
   resume_confirmed}`. The current source sends SIGSTOP/SIGCONT (`:344-360`) but the receipt proves only
   that PID 2028 was named, not what process instance it was or that both transitions succeeded.

A new matched calibration requires a clean move-40 local base under that frozen rule and a clean
candidate local run on the same hardware/rule, or it may be replaced by a direct exact contest-T4 cold
public-entrypoint time for the candidate. Neither existing local receipt is salvageable by another
assembly: reclassification after observing it is the defect.

`verdict_scope: INSTANCE` — A1 and A2 cannot establish a clean local denominator.  
`verdict_scope: FORMULATION` — the exact replacement above governs the next calibration attempt; it
does not retroactively say `dasd` caused the two retained walls.

## PASS-manufacturing surfaces

- **Settle exclusion:** excluding pre-run time from the timed wall is correct. Accepting the first
  quiet sample is not. A1 has 36 busy settle samples followed by one empty `settle_0037`; A2 has 22
  followed by empty `settle_0023`. Both later show `dasd` near 96% for fourteen during samples.
- **Ancestor exclusion:** `classify` lists ancestors separately and counts them only above a distinct
  cap using `>` (`decode_timing_concurrency.py:145-172`). At the final 100% cap, an unrelated control
  plane can consume a full core without refusing. Identity-by-current-PID-tree also lacks start-time
  custody.
- **Dashboard pause:** intentional isolation can be valid, but `paused_pids:[2028]` is not proof of
  process identity or successful stop/resume. No receipt evidence shows that pausing manufactured the
  observed timing, so this is a custody defect, not a causal accusation.
- **Assembly-time re-derivation:** retaining every process at >=5% makes 25% and 100% sensitivity
  analysis possible. It does not authorize changing the authority rule. The source explicitly
  reconstructs competitors from `visible` and the assembler accepts new threshold/tolerance arguments.
- **Phase asymmetry:** process competitors inside checkpoint intervals are discarded whenever summed
  self-median excess is <=1%, while identical work immediately outside the interval refuses. A2 is the
  concrete positive control: the large Python/tar/git set is admitted in stage 75; one 124.2% ChatGPT
  sample before the first checkpoint refuses the run.
- **Aggregate topology:** `_default_aggregate_cap` uses `os.cpu_count()` and produces 700 on this host
  (`decode_timing_concurrency.py:76-88`), despite the receipt's six-P-core fact and four decode threads.

`verdict_scope: FORMULATION` — these are reachable pass surfaces in the reviewed code.  
`verdict_scope: INSTANCE` — only the stated A1/A2 samples and `paused_pids` values are claimed to have
occurred; no unsampled burst, thermal event, or PID-reuse event is asserted as fact.

## Gate-with-no-door candidate census

I searched 330 tracked top-level `validate_`, `check_`, and `require_` definitions under `src/tac`
outside tests and `preflight.py`, then searched their symbol uses across tracked Python under `src/tac`
and `tools`. Three symbols had only direct test references:

| Static candidate | Inspection result |
|---|---|
| `src/tac/submission_archive.py:1016` `validate_archive_seg_tile_actions_payloads` | **Not this class.** It is a public wrapper; the production archive validator calls the private payload validation at `:1003-1013`. There is no required receipt field whose only writer is a fixture. |
| `src/tac/witness_control/trajectory_transaction_v2.py:1062` `require_matching_topology` | **Not this class.** It validates caller-supplied in-memory key pairs and complete inventories, not a producer receipt or a persisted required field. |
| `src/tac/pr85_bundle.py:534` `validate_pr85_member_name` | **Not this class.** It validates a literal archive member name and has no producer-only required field. |

I also inspected likely false positives with explicit required proof fields, including
`src/tac/optimizer/exact_readiness.py:1887` and `src/tac/hdm8_selector_cuda_gate.py:147,508`; committed
materializers/builders for those proof schemas exist, so they are not candidates on this test.

**Bounded result:** I did not find another confirmed no-door validator in this symbol-and-required-field
scope. This is not a claim that none exists elsewhere: method-only validators, dynamic call sites,
`preflight.py` source gates, non-Python producers, and semantic fields written under different names
were outside this bounded census. No candidate was fixed.

`verdict_scope: BOUNDED CENSUS` — tracked Python surfaces and the exact static method above, not global
nonexistence.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus by content for `decode_wall_clock`, `quiesced`,
`competing_process_count`, `margin_time_basis`, `impact_tolerance`, `stage_tolerance`, `timing
calibration`, `process concurrency`, `whole job wall`, and `gate with no door`; searched
`CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED surfaces, design/SPEC documents, and task-ledger rows;
and generated/searched the canonical equation registry with
`.venv/bin/python tools/list_canonical_equations.py --json`.

Beyond the charter's named seeds:

1. `.omx/research/ddm_wc2_wall_clock_pass_20260820.md` establishes that the public 1,800 s bound is the
   whole CI job, not an inflate-only allowance, and warns that cache state/instrument mutations require
   explicit denominators and a new calibration. This kept the present 1,260 s internal policy separate
   from the reviewed contention rule and prevented treating the 540 s reserve as proof of measurement
   validity.
2. `.omx/research/ddm_wc1_decode_wallclock_verdict_20260816.md` contains cache-dependent macOS advisory
   timings on a different vehicle/instrument. It changed nothing numerically here and was not imported
   as a clean-baseline precedent.
3. The canonical-equation registry contains decode/runtime relationships but no equation that derives
   a process `%CPU`, stage-excess, settle, or aggregate threshold for this host. No canonical equation
   was used to promote the post-hoc rule.
4. The live hot state now queues an RLC1 `g5` timing after pr10 and records `g3/g4` as refused. Those
   receipts were created outside this charter's named evidence set and were not reviewed or used to
   tune the replacement rule. They changed only the follow-on owner/fire order below.
5. Task-ledger rows preserve `.omx/research/ddm_rlc1_20260910/TIMING_FIRE_ORDER.json` as the timing
   consumer. The rerun therefore has an existing owner/store rather than becoming an orphan note.

No beyond-seed source supplied independent repeated clean runs that could calibrate 1% or prove that
100% individual/700% aggregate thresholds are safe on a six-P-core host.

## Boundaries

- Review only: no source code, test, receipt, checkpoint, sidecar, pointer, ledger, upstream file, or
  SSD payload was edited, moved, or deleted. The only research deliverable edit is this memo; the
  charter-mandated checkpoint writer also appended `.omx/state/subagent_progress.jsonl`.
- No decoder, producer, scorer, exact evaluator, dispatch, or timing rerun was launched. The mtime and
  JSON calculations are read-only re-derivations of retained bytes.
- The checkpoint timestamp mapping reconstructs sample epochs from the receipt's UTC/monotonic pair;
  the UTC string is second-granular. The instrument's +/-20 s overlap padding makes the cited large
  stage/process overlaps insensitive to sub-second truncation, but this is not nanosecond custody.
- `%CPU` is the retained `ps` decaying average, not instantaneous core residency, P-core assignment,
  frequency, memory bandwidth, I/O wait, or thermal state.
- The two runs share one host and one day, with no independent clean replicate. Jitter figures are
  descriptive for these traces, not a population confidence interval.
- The 5% threshold is cleared only as a diagnostic listing threshold. The 1% threshold is not cleared
  as an admission threshold.
- No claim is made that A2 is clean, that its shorter wall is the true clean base, or that named
  processes caused exactly the observed seconds.
- Move 40's exact contest score and direct 990.053829427 s T4 time are not withdrawn. Local
  calibration/inheritance from `L1` is the refused surface.
- The current RLC1 `g3/g4` facts in hot state were not part of this review. They cannot cure a rule
  selected after the evidence or alter this receipt-specific verdict.
- The validator census is explicitly bounded as stated; “did not find” is not global absence.

<!-- # FORMALIZATION_PENDING: review-only timing-admission finding; the required frozen-rule rerun may
measure a calibration relation, but this memo creates no measured score row or canonical equation. -->

## Follow-on dispositions

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / ddm_dwc1`; consumer store:
  `.omx/research/ddm_rlc1_20260910/TIMING_FIRE_ORDER.json`; fire trigger: harvest of ddm_pr10 before
  RLC1 `g5`. Implement the frozen-rule/hash fields and whole-window thresholds specified above; do not
  assemble another authority receipt with post-run CLI thresholds.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / ddm_dwc1`; consumer store:
  `.omx/research/ddm_rlc1_20260910/TIMING_FIRE_ORDER.json` plus the retained timing receipt directory;
  fire trigger: cured instrument, three consecutive quiet settle samples, and a no-commit/no-spawn/
  no-pytest host window. Rerun the move-40 local calibration base and unchanged RLC1 local public path
  under the same frozen rule, or obtain a direct exact contest-T4 RLC1 timing; retain all outputs and
  refuse either local run on any competitor.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`; consumer store: RLC1 candidate seal and canonical
  evaluation ledger; fire trigger: exact runtime/archive identity plus timing <=1,260 s established
  without the refused L1 ratio. Rebind the public proof/seal and authorize an exact evaluation only
  after the timing condition clears.
- **FOLDED** — owner: `MAIN / ddm_dwc1`; consumer store:
  `.omx/research/ddm_dwc1_20260910/MOVE40_QUIESCED_LEG.json`; fire trigger: any future attempt to inherit
  timing from move 40. Preserve the direct T4 990.053829427 s instance fact, but refuse the present
  `quiesced`/ratio lineage as calibration authority until replaced.

Frontier unchanged: `composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)`.

## LIVE-HYPOTHESES

- A precommitted strict whole-window rule can produce a usable local calibration on this host if the
  macOS activity jobs are allowed to finish and the control planes remain idle. This is plausible
  because both retained attempts eventually produced at least one quiet settle sample, but three
  consecutive quiet samples and a fully clean wall have not been observed.
- RLC1 may still clear the 1,260 s T4 timing policy. The live board's unreviewed g3/g4 local walls are
  near 830 s, and a correctly remeasured move-40 local base may yield adequate headroom; this remains
  unproven until both denominator and candidate run pass the same frozen rule or RLC1 is timed directly
  on contest T4.
- Direct contest-T4 timing is likely the shortest sound route if local host quiescence remains scarce.
  It removes the contaminated local denominator entirely while testing the exact runtime that the gate
  is meant to protect.

## DEAD-ENDS

- Reassembling A1 with another threshold is closed: the receipt has already been exposed, and 100 lies
  inside the exact `(95.9,124.2]` interval that admits A1 while refusing A2.
- Treating a run's own median as a clean counterfactual is closed: sustained slowdown shifts the median
  and becomes invisible.
- Treating 5% diagnostic separation as validation of the 1% wall-admission rule is closed: they answer
  different questions, and both authority thresholds were fixed after the traces existed.
- Calling A1 `quiesced` because `competing_process_count=0` in the assembled block is closed: the raw
  receipt says count 3 and `quiesced=false`, and the zero was created by post-run reclassification.
- Calling A2's Python/git activity “native build” is closed by checkpoint/sample epoch overlap: it sits
  beside stage 75; only the 124.2% ChatGPT sample is outside and dispositive under the final rule.
- Using the 0.7 safety factor to cure invalid calibration evidence is closed: margin can absorb a small
  measured bias but cannot prove the denominator or bound unmeasured uniform slowdown.
- Withdrawing move 40's score or direct T4 timing is closed: this review did not run or falsify the
  exact evaluator receipt; it rejects only the local quiescence/calibration authority.
