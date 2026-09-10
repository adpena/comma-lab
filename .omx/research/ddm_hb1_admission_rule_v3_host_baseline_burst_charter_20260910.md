# ddm_hb1 — implement `decode_wall_clock.admission_rule.v3` (host-baseline burst class) in the timing sampler, exactly as ddm_pr11 wrote it — charter, MAIN 2026-09-10

## What and why
ddm_pr11 (second family, sol xhigh) adjudicated the frozen timing-admission rule against four refused
cold runs of the move 40 receiver and ruled AMEND: keep everything in v2 (quarter-core threshold over the
whole non-settle window, ancestors at 25 %, aggregate 200 %, three-sample settle, frozen hash, paused-
process custody) and ADD ONLY an exact, bounded host-baseline burst class for the PID-1 Apple maintenance
pair `/usr/libexec/dasd` + `/usr/libexec/syspolicyd`. The exact JSON object and the nine normative
comparison/storage points are in
`.omx/research/ddm_pr11_adjudicate_timing_admission_on_daemon_bursts_20260910.md` §"Exact prospective
amendment — decode_wall_clock.admission_rule.v3". Implement THAT text. Do not redesign it. The validator
side (`tac.decode_wall_clock` accepting `margin_time_basis == "bounded_host_baseline"` only under v3 with
a matching frozen hash, count 0, and every allowance check true) is already landed (0524522f0) — read it
and produce exactly the fields it consumes (`concurrency.admission_rule` with `schema` v3,
`admission_rule_sha256`, `host_baseline_allowance = {valid: bool, checks: {name: bool, ...}, ...}`).

## Files (read first)
- `src/tac/decode_timing_concurrency.py` (v2 rule: `ConcurrencyRule`, `classify`, `sample_competitors`,
  `summarize`, `wait_until_quiet`, `assemble_local_receipt`), `tools/quiesced_decode_timing.py`,
  `src/tac/tests/test_decode_timing_concurrency.py` (17 tests; keep them passing or update them with the
  stated reason), `src/tac/decode_wall_clock.py::_local` (the consumer of the new basis).
- The four refused move 40 receipts for counterfactual tests (read-only; do NOT edit or reassemble them):
  `/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/receipts/move40_quiesced{,2}/CONCURRENCY.json`
  (v1 sampler), `move40_quiesced3/CONCURRENCY_attempt3_refused.json`, `move40_quiesced4/CONCURRENCY.json`
  (v2). pr11's counterfactual table: A1 REFUSE (count 3 at during_0014), A2 REFUSE (count 5 at during_0003),
  A3 ADMIT (count 0; 10 samples / 200.292 s), A4 ADMIT (count 0; 11 samples / 220.320 s). Your
  implementation, applied to those retained sample lists as a TEST (not as an assembly of a new authority
  receipt), must reproduce exactly those four results.

## Deliverable
1. `ConcurrencyRule` v3: fields `host_constraints` (platform_system, machine, logical_cpu_count,
   performance_core_count — filled from the running host at freeze time), `host_baseline_burst`
   (schema `decode_wall_clock.host_baseline_burst.v1`, activation 25.0, members with exact `comm` +
   `ppid` + `pcpu_max_inclusive` [dasd 100.0, syspolicyd 75.0], combined cap 175.0, max 1 burst in the
   window, max 15 active samples per burst, max inclusive span 300.0 s), the `definition` string as pr11
   wrote it. `frozen()`/`sha256()` cover the complete object. `schema` = `decode_wall_clock.admission_rule.v3`.
2. Sampling: `classify` records for each row whether it is an exact host-baseline member match (exact
   `comm` string and exact `ppid`; no basename/substring/vendor matching). Allowance disabled unless all
   four host constraints match exactly (`platform.system()`, `platform.machine()`, `os.cpu_count()`,
   `p_core_count()`).
3. Burst adjudication after the window (pr11 points 3–7): bursts = maximal runs of consecutive non-settle
   samples with ≥ 1 active matching row (pcpu ≥ activation); each member ≤ its cap, combined ≤ 175.0, burst
   count ≤ 1, active samples per burst ≤ 15, inclusive span ≤ 300.0 s; if valid, `unwaived_other_pcpu_sum
   = other_pcpu_sum − Σ matched rows` per sample and the aggregate test uses it; if invalid, subtract zero
   (the rows are ordinary competitors). `competing_process_count` = max unwaived count. Store
   `host_baseline_bursts`, every matched row, every bound with its result, `unwaived_other_pcpu_sum` per
   sample, and `host_baseline_allowance = {valid, checks: {...all bounds...}, bursts: [...]}`.
4. Settle (point 4): may provisionally ignore a matching row only while the observed prefix still
   satisfies every bound; any later violation refuses the run.
5. `margin_time_basis`: `quiesced` only when NO burst-active sample exists; `bounded_host_baseline` when
   the allowance is valid and count is 0; otherwise `measured_concurrency_nonzero`.
6. Assembler (point 8): copies rule, hash, freeze/start times, and the burst adjudication from the
   producer receipt; verifies the hash; re-derives the count under the frozen rule (with the same burst
   adjudication) and refuses on mismatch; no override flags.
7. Stage diagnostics stay non-authority (point 9).
8. Tests: the four counterfactual traces (load the retained CONCURRENCY.json sample lists, apply v3,
   assert pr11's table), host-constraint mismatch disables the allowance, a renamed/reparented daemon is
   ordinary, second burst refuses, span/samples/caps refuse at the boundary (`<=` allowed), the
   `bounded_host_baseline` receipt passes `tac.decode_wall_clock.build_decode_wall_clock` end to end
   (mirror `test_assembled_quiesced_receipt_passes_the_validator`), and the v2 tests still hold.
9. `ruff check` clean; two visible review passes per changed .py; ONE serializer commit
   (`--expected-content-sha256` per file, zsh arrays, no co-author trailer, tags `[no-triality] [p0-ledger-ok]`).
   Then write the frozen v3 rule receipt for THIS host to
   `.omx/research/ddm_dwc1_20260910/ADMISSION_RULE_V3_FROZEN.json` (`frozen()` + `sha256` + utc) and commit it.

## Boundaries
- Do not edit `src/tac/decode_wall_clock.py` beyond what the `bounded_host_baseline` consumer already
  expects; if you find it needs a field, STOP and report the exact field.
- Do not run any timing window, any n600 pass, or anything at ≥ 25 % CPU for more than a minute — rp1's
  frontier chain owns the host; unit tests are fine.
- Do not reassemble A1–A4 as authority receipts (pr11: "A rule fixed after a run cannot confer authority").
- Never edit `upstream/`, the PR tree, or any `/Volumes/*/pact/` artifact.

## OPTIMAL FORM
- Reference form: pr11's amendment text (exact JSON + nine points) — implement, do not design.
- Provenance pins: pr11 memo sha (record it), commits 52c4962cc (v2) and 0524522f0 (validator side),
  the four receipt paths + sha256.
- Scope delta: none; mechanism delta: none.

## Prior negatives accounted (operator 2026-08-15)
- pr10: a rule fixed after the runs is the defect — v3 is frozen and hashed BEFORE any run; A1–A4 stay refused.
- mv2/mv3 stopped on over-literal readings; here the text is exact — ambiguity is to be reported, not
  resolved silently, but "implement the JSON as written" is not ambiguity.
- dwc1's gate with no door — the counterfactual tests on real traces are the door check.

Checkpoint as `ddm_hb1` every ~10 tool uses. Final message: commit sha, test counts, the four
counterfactual results, the frozen v3 sha for this host, every boundary, and the frontier line
`composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)` unchanged.
