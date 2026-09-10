# ddm_hb1 — exact v3 host-baseline burst sampler

Implementation complete in the working tree; main-branch landing and genuine host
freeze depend on the recorded environment blockers. This is apparatus, not score
progress. Axis: `[macOS-CPU advisory; retained-trace counterfactual tests; scorer-free]`.
`score_claim=false`. Scope delta: none. Mechanism delta: none from pr11's amendment.

The sampler freezes the complete v3 object, matches exact comm/integer ppid, retains
subvisible matching rows, adjudicates the whole non-settle window globally, and
subtracts matching CPU only when every allowance bound passes. Settle ignores a
member provisionally only while its observed prefix passes. Later invalid window
bounds remove all waivers. The assembler re-derives and compares the complete burst
adjudication and count, preserves rule/hash/freeze/start custody, and rejects absent
captured-host evidence. Calibration accepts the validated bounded basis through the
already-landed consumer. Stage diagnostics remain non-authority; paused-process
custody is unchanged. The v2 frozen SHA remains unchanged.

## Measured verification

All 248 retained samples (78/64/59/47) were loaded read-only with their SHA-256 pins.
The test host is explicitly simulated as Darwin/arm64/18 logical/6 performance cores.
The retained receipts are NOT reassembled or upgraded.

| Trace | Counterfactual v3 result | Maximum-count label | Burst samples / inclusive seconds |
|---|---|---|---|
| A1 | REFUSE, count 3 | during_0014 | 14 / 280.416 |
| A2 | REFUSE, count 5 | during_0003 | 14 / 280.428 |
| A3 | ADMIT, count 0 | none | 10 / 200.292 |
| A4 | ADMIT, count 0 | none | 11 / 220.320 |

`counterfactual_results.json` contains full source paths, source hashes, exact burst
spans and unwaived aggregate maxima. The pr11 memo hash and source commit/file pins
are in `source_pins.json`. The exact reference-host rule hash in the TEST is
`25d0a778f0fd31a0f218bc129be0d1504b32c00cf58b082282430c14c198e50c`.
This is not a prospective host freeze.

Final tests: `python -m pytest src/tac/tests/test_decode_timing_concurrency.py
src/tac/tests/test_decode_wall_clock.py -q --tb=line`: **89 passed / 92 collected;
3 fail with PermissionError executing ps in this sandbox**. Of 43 sampler cases,
40 pass; all 49 consumer cases pass. The original 17 tests remain intact (14 pass,
3 environment-blocked). All 26 new cases pass, including exact pr11 JSON, the four
retained traces, four host mismatches, renamed/reparented daemon cases, all cap
boundaries, second burst, settle, v2 hash preservation, tampering and end-to-end
bounded receipt validation. No test is being reported as live process evidence.
Ruff is clean. Two visible review passes for each changed Python file are recorded
by review_tracker and described in `reviews.md`.

## RECALL EVIDENCE

Queries `quiesced|host.baseline|syspolicyd|timing.admission` searched the full
`.omx/research/` Markdown corpus by content, including CANONICAL_RESEARCH_INDEX and
sub015_DAG surfaces, plus docs and canonical task/dispatch/lane stores. Retained
query output is in `recall_research.txt` and `recall_design_tasks.txt`. The complete
canonical equations command was run (`tools/list_canonical_equations.py --json`);
483 entries were inspected for `wall_clock` and `metal_concurrency_speedup`, with
selected entries, command, census and full-output hash in `recall_equations.json`.

Beyond the charter seeds: pr10's original review and the dwc1 seal-leg history
preserve the withdrawn own-median rule; neither may confer post-hoc authority.
The DAG at lines 2401/2409 records dasd contention on a GPU-feed workload, confirming
that this amendment is not a general daemon exemption. The equations registry's
metal_concurrency_speedup_gv1_v2 records inconsistent repeated concurrency windows;
it supplies no calibrated daemon admission law. A disk-admission memo (ddm_dk2)
also scopes saturation to storage timing, not this CPU classification. These findings
kept the exact narrow envelope, diagnostic boundary, and no-new-timing scope; no
additional mechanism was introduced. No ddm_hb1 entry was found in the searched
canonical task/dispatch/lane rows; the charter and ddm_hb1 checkpoint own this work.

## Boundaries and blockers

- No real timing window, n600 pass, training, scorer, paid dispatch, or sustained
  heavy CPU workload was launched. Unit tests use synthetic timings only; their
  live-process checks fail at ps before any timing producer starts.
- No SSD object, retained receipt, PR tree, upstream file, candidate payload,
  frontier pointer, or common-contract protected file was edited, moved or deleted.
  The staged index is untouched by manual commands; serializer custody is separate.
- `src/tac/decode_wall_clock.py` is unchanged; its existing fields suffice.
- A1–A4 remain refused under their original frozen rules. Counterfactual ADMIT is
  not a score, a timing measurement, or retrospective authority.
- The host P-core query `sysctl -n hw.perflevel0.physicalcpu` returns Operation not
  permitted. Existing p_core_count falls back to 18 logical CPUs here. That is not
  a measured performance-core count, so the requested ADMISSION_RULE_V3_FROZEN.json
  was NOT written. Exact probe evidence and the follow-on are in
  `host_freeze_blocker.json`. The fixed v3 aggregate cap is 200.0 as pr11 requires;
  it cannot silently expand to 1400.0 from this fallback.
- No claim of zero daemon impact, population causality, CPU/CUDA transfer, whole-CI
  completion, or goal completion follows from these tests.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / ddm_hb1; consumer store: the three
  changed Python files and this handoff, through the serializer landing receipt;
  fire trigger: harvest of the reviewed implementation with a writable Git object
  store. Verify and land the exact serializer commit/bundle if the current attempt
  is retained rather than landed.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / ddm_dwc1; consumer store:
  `.omx/research/ddm_hb1_20260910/pytest_full_final.txt`; fire trigger: the reviewed
  commit is landed and ps access is available on the host. Rerun the three retained
  live-process tests and record their actual results.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / ddm_dwc1; consumer store:
  `.omx/research/ddm_dwc1_20260910/ADMISSION_RULE_V3_FROZEN.json`; fire trigger:
  readable real-host P-core query and landed reviewed v3, before any next local
  timing run. Write frozen()+sha256+utc from the real host and serializer-commit it;
  never substitute the sandbox fallback or test fixture.

Frontier unchanged: composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40).

## LIVE-HYPOTHESES

- A prospective frozen-v3 run can admit a bounded maintenance burst: all four
  retained traces satisfy that class and A3/A4 have zero ordinary competitors.
  Prospective repeatability remains untested; further timing is owned by MAIN's
  existing pr11 fire order after rp1 releases the host, not by this arm.

## DEAD-ENDS

- Retroactive admission of A3/A4: their actual v2 freeze still refuses them.
- Broader daemon exemptions or stage-median waivers: ordinary activity in A1/A2
  must refuse and the charter allows only exact bounded members.
- Freezing 18 fallback logical CPUs as measured P-cores: sysctl was denied; this
  would misstate the host evidence. Query the real host before freezing.
