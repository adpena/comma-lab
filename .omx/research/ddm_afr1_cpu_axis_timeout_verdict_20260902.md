# AFR1 contest-CPU axis verdict — inflate exceeds the 30-minute contest budget (measured)

Date: 2026-09-02 · Owner: MAIN · Lane: `ddm_afr1_gen7_cpu_axis_n600_20260902` (closed `completed_modal_auth_eval_harvested_not_passed`)
Call: `fc-01M1HGRHPMYH0FEX3T751K77ZH` · Modal elapsed 1,804.8 s · Cost ≈ $0.10–0.15 (#381)

verdict_scope: instance — the AFR1 gen-7 archive (`cbb8d928…`, 180,002 B) × its sealed 38-file
receiver, decoded on the Modal CPU worker class. The budget infeasibility binds THIS object only:
the predecessor receiver on the same worker decoded in 831.5 s, so the CPU axis is NOT closed for
the lineage or family — a lighter-decode receiver (e.g. generation 8 without the context-mixing
stages, or a native-port decode path) could re-open it. GitHub CI-runner transfer is unmeasured.

## Object identity (exact)

- Archive: `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`, 180,002 B — the frozen gen-7 AFR1 bytes, unchanged.
- Runtime: the 38 MANIFEST-enumerated authority files + archive.zip, staged to
  `/Volumes/APDataStore/pact/ddm_pq12/receipts/afr1_cpu_fire_staging/` (38/38 sha-verified vs
  `MANIFEST.sha256`; 42 ExFAT AppleDouble sidecars purged pre-seal — #1122 genus).
- Seal: `CANDIDATE_SEAL_afr1_gen7_cpu_axis.json`, seal_sha256 `9082a07a…`, axis contest_cpu,
  RECORD-WITH-REASON admit framing, single-axis waiver recorded at fire.
- Worker provenance pins the same archive sha; extraction + member whitelist PASSED; the
  rc64_backend `cc` compile and dependency bootstrap SUCCEEDED (inflate was *running*, not failing).

## The measurement

`inflate.sh` was killed by the tool's contest-budget enforcement at **exactly 1,800.0 s**
(`ProcessGroupTimeout` → "[inflate] TIMED OUT after 1800s. Contest budget is 30 min"). rc=1,
no `contest_auth_eval.json`, no score produced. This is a **lower bound**: CPU inflate takes
MORE than the entire 30-minute budget, before evaluation (which itself cost ~1,000+ s CPU on
the predecessor row) even starts.

Same-instrument comparison (all rows on the same Modal CPU worker class):

| Object | CPU inflate | T4 inflate |
|---|---:|---:|
| MC36 lineage (#1054, 2026-08-14) | 831.5 s (completed; S 0.20513) | — |
| **AFR1 gen-7** | **>1,800 s (killed at budget)** | 578.9 s (authority row) |

AFR1's CPU decode is ≥2.17× the predecessor's and >3.11× its own T4 decode. Mechanism
attribution: the five post-rc2 lossless stages (fx5 integer log-odds context mixing · dx2
group-conditioned contexts · gb1/afr1 tile48×groupbin8 conditioning · lb1 joint collect) each
made the archive smaller by making the DECODE-side context model heavier; per-token sequential
context mixing has no GPU to hide behind on the CPU axis.

## Adjudication

1. **The CPU axis disposition upgrades from inherited to MEASURED.** The packet's
   RECORD-WITH-REASON section cited a predecessor delta (+0.0432, pose ~21× degraded). The
   stronger, now-measured fact: **an AFR1 contest-CPU score cannot exist inside the contest
   budget** — any CPU row would exceed 30 min at the inflate step alone and is dead-by-rule
   (sister of the ux1 strict-scorer DEAD-BY-RULE guard). The declared `linux-nvidia-t4` runtime
   is demonstrably necessary, not merely preferred.
2. **No extended-timeout re-fire.** The tool offers `--inflate-timeout 7200`; that would buy a
   diagnostic CPU score that can never be a contest row, at ~$0.4–0.7 of the ~$1.1 remaining
   Modal headroom. Refused under the means/ends firewall; the operator may order it explicitly
   if they want the out-of-budget CPU score for the record.
3. **yr1 F8 consequence sharpened:** the generation-8 receiver fix is the FAIL-CLOSED
   CUDA-required error (yr1's preferred option), now backed by a measured reason — the silent
   CPU fallback in `inflate.py` would run into a budget violation, never a comparable score.
4. **Compliance red `contest_cpu_auth_eval_exists`** resolves as
   MEASURED-INFEASIBLE-IN-BUDGET with this receipt, not as a score row. The PR body's CPU
   sentence (operator-authored) can now say: "CPU decode measured >1,800 s on the same worker
   class that decoded the predecessor in 831 s; GPU is required to meet the 30-minute budget."
5. **Caveat, stated plainly:** the Modal worker class is our CPU instrument, not the GitHub CI
   runner; runner transfer is unmeasured. The margin is decisive anyway — inflate alone
   consumes >100% of the total budget, and evaluation is still owed on top.

## Custody

Full receipts retained: `/Volumes/APDataStore/pact/ddm_pq12/afr1_cpu_axis_materialized/`
(`MODAL_REMOTE_RESULT.json` with stdout/stderr/preflight/provenance artifacts, FIRE_MANIFEST,
launch manifest). Frozen gen-7 packet untouched. No score claim is made by this memo;
`[contest-CUDA T4 n600]` remains the sole authority axis for AFR1.

Consumers: task #1389 (gen-8 F8 leg) · the operator's PR-body CPU sentence · the pq12
compliance ledger at its next regeneration (gen-8, never gen-7).
