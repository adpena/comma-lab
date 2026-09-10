Verdict: **AMEND + T4-DIRECT**. The completed adjudication is in [ddm_pr11_adjudicate_timing_admission_on_daemon_bursts_20260910.md](/Users/adpena/Projects/pact/.omx/research/ddm_pr11_adjudicate_timing_admission_on_daemon_bursts_20260910.md).

All 248 retained samples were examined:

- A1/A2 remain refused; v3 counterfactual counts are 3 and 5.
- A3/A4 remain actually refused under frozen v2, but v3 would admit both with count 0.
- Under the observed 16-minute cycle and ≥3-minute burst envelope, the maximum clean gap is 780 seconds. The shortest decode was 783.057 seconds before settling, giving `p=0` and no finite expected attempt count under that model.
- Move 40 has direct timing authority: 990.053829427 seconds `[contest-CUDA T4 n600]`, below the 1,260-second policy limit.
- rp1 may inherit that direct leg after exact normalized-receiver equality is rechecked.
- RLC1 cannot inherit it because its receiver digest differs; it needs its own T4-direct receipt or new receiver-matched v3 calibration.

Exact prospective rule:

```json
{
  "schema": "decode_wall_clock.admission_rule.v3",
  "threshold_pcpu": 25.0,
  "visible_pcpu": 5.0,
  "ancestor_cap_pcpu": 25.0,
  "aggregate_cap_pcpu": 200.0,
  "interval_seconds": 20.0,
  "settle_quiet_samples": 3,
  "stage_tolerance": 0.05,
  "decode_threads": 4,
  "host_constraints": {
    "platform_system": "Darwin",
    "machine": "arm64",
    "logical_cpu_count": 18,
    "performance_core_count": 6
  },
  "host_baseline_burst": {
    "schema": "decode_wall_clock.host_baseline_burst.v1",
    "activation_pcpu": 25.0,
    "members": [
      {"comm": "/usr/libexec/dasd", "ppid": 1, "pcpu_max_inclusive": 100.0},
      {"comm": "/usr/libexec/syspolicyd", "ppid": 1, "pcpu_max_inclusive": 75.0}
    ],
    "combined_pcpu_max_inclusive": 175.0,
    "max_bursts_in_window": 1,
    "max_active_samples_per_burst": 15,
    "max_inclusive_span_seconds": 300.0
  },
  "definition": "ordinary competitor = pcpu >= threshold_pcpu for every process outside producer tree, monitor tree, and pid 0; monitor ancestors compete at pcpu >= ancestor_cap_pcpu. The host-baseline allowance is disabled unless platform.system(), platform.machine(), os.cpu_count(), and p_core_count() exactly equal host_constraints. A host-baseline row matches only exact comm and ppid; a burst-active sample has at least one matching row at pcpu >= activation_pcpu. The allowance is valid only when matching rows stay pcpu <= their member caps, matching-row sum is <= combined cap in every sample, active samples form at most one consecutive run, active-sample count is <= max_active_samples_per_burst, and last.monotonic - first.monotonic + interval_seconds <= max_inclusive_span_seconds. If valid, matching rows do not compete and are subtracted from the aggregate; if any condition fails, no matching row is waived. Unwaived summed pcpu competes at >= aggregate_cap_pcpu. Apply to every non-settle sample; count = max over samples; admitted iff count == 0. Start after settle_quiet_samples consecutive samples with zero unwaived competitors; a later allowance failure refuses the final receipt. Stage rates are diagnostic only."
}
```

`t4_direct` is fail-closed: it binds the exact receipt, archive SHA, normalized receiver digest, uploaded T4 runtime digest, canonical public entrypoint, CUDA availability, accepted T4 hardware name, n600 completion, and cold-decode proof. Its measured inflate time must be finite, positive, and at most 1,260 seconds. It accepts no local receipt, calibration receipt, or CPU/T4 ratio. Timing authority does not create score authority.

Boundaries:

- The daemon allowance is limited to the exact host, paths, PID ancestry, caps, and burst envelope above. It does not declare these daemons harmless generally.
- No existing receipt is retroactively promoted.
- Stage rates remain diagnostic; own-run medians cannot waive contention.
- Retained `%CPU` is a decaying `ps` average and does not measure residency, frequency, memory pressure, I/O wait, or thermals.
- Four runs on one host/day do not establish population causality.
- No code, tests, receipts, checkpoints, payloads, pointers, upstream files, or SSD artifacts were edited or rerun. No scorer or paid dispatch ran.
- The serializer could not write this checkout’s Git object database (`Operation not permitted`). It retained verified intended commit `31121e0a498e0dc06e5b26fa22226245d02bc5d4`; the memo is therefore **not attached to repository HEAD**. The bundle and machine-readable custody record are in [the serializer fallback directory](/Volumes/VertigoDataTier/pact/ddm_pr11/receipts/commit_serializer_fallbacks/20260910T123146.291004Z-81713/receipts.jsonl).

Frontier unchanged: `composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: repository HEAD; fire trigger: Git object writes are available. Land intended commit `31121e0a…` from the retained bundle or format-patch and verify memo SHA `7fd5ee3b…`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN/ddm_dwc1; consumer store: timing implementation, tests, and frozen v3 receipt; fire trigger: the committed pr11 memo is harvested. Implement v3 and `t4_direct` verbatim and freeze/hash before any local run.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN/rp1; consumer store: rp1 candidate seal; fire trigger: the validator accepts direct-source inheritance and staged normalized receiver equality passes. Inherit move 40’s T4-direct leg.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN/rlc1; consumer store: RLC1 timing fire-order record and candidate seal; fire trigger: exact RLC1 runtime/archive identity is frozen and the T4 lane is available. Prefer its own cold public-entrypoint T4-direct fire.

## LIVE-HYPOTHESES

- Frozen v3 will produce repeatable local receipts because A3/A4 had no unwaived competitors and differed by only 0.2190% despite materially different burst timing.
- RLC1 will clear 1,260 seconds on T4 because its two refused local runs clustered around 830 seconds, but no direct RLC1 T4 receipt exists.
- rp1 will preserve receiver equality after staging because its chartered differences are only archive bytes and two normalized pin literals; the final staged digest check remains unperformed.

## DEAD-ENDS

- Leaving v2 unchanged is closed under the observed-cycle model: the decode cannot fit within the maximum clean interval.
- Retroactively admitting A3/A4 is closed by their frozen v2 SHA.
- Restoring the own-median impact rule is closed because uniform slowdown moves the median and much of the wall is outside checkpoint coverage.
- A broad system-daemon exemption is closed by ordinary daemon competition in A2 and prior workload-dependent `dasd` evidence.
- `caffeinate -u` as prevention is closed by A4.
- Reusing the withdrawn move-40 local CPU/T4 ratio is closed; only the independent exact T4 fact survives.
- Requiring local calibration beside an exact same-object T4 decode is closed because it adds a noisier denominator without strengthening identity.
- Move-40 inheritance for RLC1 is closed by receiver-digest inequality.