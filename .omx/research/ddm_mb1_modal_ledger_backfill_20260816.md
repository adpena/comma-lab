# ddm_mb1 — Modal ledger backfill + the $20 cap answered from the record

**Arm:** `ddm_mb1_modal_ledger_backfill` · **UTC:** 2026-08-16 · **Repo:** `/Users/adpena/Projects/pact`
**Charter:** make the operator's binding Modal budget constraint (task #381, ≤ **$20 hard cap**, set
2026-07-09) answerable from the record instead of from prose.

---

## THE ANSWER — lead with it

**Cap-window Modal spend = `$18.6124` of the `$20.00` cap. Headroom = `$1.3876` (6.9%). The cap is
93.1% consumed.** [MEASURED — Modal workspace billing API, workspace `adpena`, window
2026-07-09 → 2026-08-16 inclusive.]

**At the last-7-day burn of `$1.1808`/day, that is `1.2` days of headroom.** A single day like
2026-08-13 (`$2.8846`) or 2026-07-15 (`$5.4756`) breaches the cap on its own. [MEASURED]

This is **2.7× the prose figure (≈$6.9)** and **10.7× the ledger's own recorded `$1.74`**. The prose
was not merely imprecise — it was wrong in a direction that mattered, and it would have licensed
spending that breaches a binding operator constraint.

**Owed to the operator: a spend decision, now.** Every further Modal dispatch must be priced against
$1.3876, not against a comfortable margin.

---

## Why the record disagreed with itself (the mechanism)

Three independent under-counts compounded. Each is a real defect, not an estimation error.

1. **`cost_actual_usd` is not an actual cost.** Of 21 priced call_ids ledger-wide: 14 are `$0.00`
   (13 `pre_spawn_fatal` — never spawned — plus one `$0.0` T4), 1 is **provably a rate-derived
   estimate** written into a field named "actual", and 6 are unflagged nonzero. The contaminated row
   (`fc-01KS1HQQ0F9GY1VYCQESSH4R4K`, A10G, `$1.004`) carries its own confession in
   `harvest_result.cost_band_anchor`: `cost_estimate_source = modal_elapsed_seconds_x_configured_hourly_rate`,
   `hourly_rate_usd = 1.1`, `cost_estimate: True`. **Deriving a $/hr rate from that row recovers the
   1.1 that was assumed.** It is circular. [MEASURED — source inspection of the row]

   Of the 6 unflagged nonzero, 4 are `$0.001` (the recording granularity floor — implied rate
   meaningless) and one is `$0.15` on a `manually_terminated` call whose elapsed is exactly `720.0` s
   (12.0 min — a round number, i.e. an entry, not a measurement). **Exactly one** row in the entire
   ledger is nonzero, unflagged, and paired with a non-round measured elapsed: `fc-01KXKRRF450JTF4BNAM9XG6B35`,
   H100, `$1.59` / `1452.309` s → `$3.941`/hr. Even that one cannot be proven to come from Modal
   billing rather than operator entry. **There is no defensible cost calibration in the ledger.** [MEASURED]

2. **`modal_elapsed_seconds` is a lower bound on billable time, not billable time.** The producer
   (`experiments/modal_auth_eval.py:795`) starts `time.monotonic()` immediately before
   `subprocess.run(cmd)` — *after* image pull, uv env build, and container start. Those are billed and
   excluded. Any elapsed×rate model therefore under-prices by construction. [MEASURED — source inspection]

3. **CPU and Memory are billed separately and are 41.8% of spend.** By-resource split of the same
   window: T4 `$5.9079` (31.7%), CPU `$5.0260` (27.0%), H100 `$4.9208` (26.4%), Memory `$2.7578`
   (14.8%). A GPU-seconds model sees none of the `$7.78` in CPU+Memory. [MEASURED]

My own pre-billing estimate, built from recovered timings and the single H100 rate, landed at
**`$6.29`** — within 10% of the prose `$6.9`, and **3× under the truth**. Two independent wrong methods
agreeing is not corroboration. Only the billing API settled it.

---

## Search scope (negative-existence claims are scoped)

Searched for `*REMOTE_RESULT*.json`, `modal_metadata.json`, `modal_call_id.txt` across:

| Tier | Hits | Note |
|---|---:|---|
| `/Volumes/VertigoDataTier/pact` | 44 | **tier is 100% full — 953 MiB free** (hygiene flag, not this arm's scope) |
| `/Volumes/APDataStore/pact` | 12 | 241 GiB free |
| repo `experiments/results/` + `.omx/` (maxdepth 6) | 629 | |
| **total, AppleDouble `._*` excluded** | **680** | 468 `modal_call_id.txt`, 180 `modal_metadata.json`, 32 `REMOTE_RESULT*.json` |

Of 680 artifacts, **12** carry `modal_elapsed_seconds`; 0 were listed-but-missing. I did **not** search
outside these three roots. Any claim about artifacts elsewhere is out of scope and unmade.

**Two artifact schemas exist**, and only one carries timing:
- `MODAL_REMOTE_RESULT.json` (exact-eval family) — **has** `modal_elapsed_seconds`, predating today's cure.
- `QS1_T4_REMOTE_RESULT.json` / `EC2_…` (dual-axis family) — **no timing fields at all**, including files
  written today. Today's poller cure (`f03c185acb`) fixes the harvest path; it does not make this family's
  producer emit timing. **That gap is still open.**

---

## Join method

Three joins, in priority order. No join was guessed; ambiguous candidates were left blind and counted.

- **J1 — ledger-authored `harvest_result.result_path`** (strongest: the harvester itself wrote the path
  at harvest time). 17 of 63 cap-window calls carry one. → 5 recoveries.
- **J2 — sibling `modal_call_id.txt` in the artifact's own directory.** Verified per-directory before
  use: exactly one `modal_call_id.txt` and one `MODAL_REMOTE_RESULT.json` per dir, exact call_id string
  match, and mtime ordering consistent with a single dispatch (call_id written at dispatch, result at
  harvest). **Independently cross-checked against filesystem mtimes**: `ddm_f26p` 14:42→15:32 ≈ 3000 s vs
  recovered `2938.831` s; `ddm_f26r` 16:55→17:11 ≈ 960 s vs recovered `951.880` s. → 4 recoveries.
- **J3 — `harvest_result.elapsed_seconds` already nested in this ledger**, never promoted to the
  top-level column. No artifact involved. → 1 recovery.

**Clean negative on J3 as a bulk source:** across the whole ledger, nested vs top-level `elapsed_seconds`
agree exactly on **216** rows, disagree on **0**, and the nested value exists without a top-level value on
exactly **1**. The nested field is redundant, not a hidden reserve. That hypothesis is dead. [MEASURED]

---

## Rows backfilled

**10 rows appended** via the canonical writer `tac.deploy.modal.call_id_ledger.update_call_id_outcome`.
Ledger 812 → 822 rows.

**Append-only verified:** the first 812 lines are byte-identical pre/post
(`sha256 3e5c1212478b5b35617d5f1e5ffca62177ca9f48dd30634b36981860667789f1`, `diff` empty). Zero mutation.

| call_id | status | elapsed (s) | gpu | join |
|---|---|---:|---|---|
| `fc-01M0073TSNJEKW2BA4XTGF950X` | harvested | 543.600 | T4 | J3 |
| `fc-01M00RBQM4RMGEG2GXK4H8MEVX` | harvested | 406.017 | Tesla T4 | J2 |
| `fc-01M00RY87TDDE5N556HXNQ1JZX` | failed | 7.518 | T4 | J2 |
| `fc-01M00WRPPSEE6HVT3Y5FFSTM37` | failed | 2938.831 | T4 | J2 |
| `fc-01M014B5F4DB6FJ5BSXXRGNB83` | harvested | 951.880 | CPU | J2 |
| `fc-01M02Q4XFXSE2NHFPGWV5NMB8A` | harvested | 9.811 | T4 | J1 |
| `fc-01M02QE9PSM7FP46VZ60DCJGQS` | harvested | 10.675 | T4 | J1 |
| `fc-01M02QMN3SQ9SNHXZMRWXYEJEW` | harvested | 421.559 | Tesla T4 | J1 |
| `fc-01M0367KFQ27VM3E8F7K6092DA` | harvested | 8.749 | T4 | J1 |
| `fc-01M036FY225QC9A75CM0Y7X7NP` | harvested | 421.620 | Tesla T4 | J1 |

Each row asserts **timing and GPU only**, and carries its own provenance:
`backfill=true`, `backfill_arm`, `backfill_join_method`, `backfill_evidence_path`,
`backfill_elapsed_semantics` (states the lower-bound caveat inline, so it travels with the number),
`backfill_is_new_outcome=false`.

Two deliberate restraints:
- **`rc` is written `None`.** Three calls have a ledger `rc=0`/`harvested` while their artifact says
  `returncode=1` (`fc-01M02Q4XFXSE2NHFPGWV5NMB8A`, `fc-01M02QE9PSM7FP46VZ60DCJGQS`,
  `fc-01M0367KFQ27VM3E8F7K6092DA`). Re-asserting either value would launder an unresolved
  discrepancy. Both are recorded verbatim in `backfill_artifact_returncode` /
  `backfill_ledger_rc_at_backfill_time`. **This rc/status discrepancy is owed to a future arm.**
- **Each row re-asserts the call's existing terminal status**, so no call's resolved state changes.
  `score=None` is passed, so the frontier auto-refresh provably no-ops
  (`auto_refresh_canonical_frontier_after_dispatch_outcome` returns `None` when `score is None`).

---

## Coverage: what is now priced or timed, what stays blind

Cap window = 63 call_ids (`dispatched_at_utc >= 2026-07-09`).

| | before | after |
|---|---:|---:|
| timed (`elapsed_seconds`) | 10 | **20** |
| priced (`cost_actual_usd`, incl. 6 zeros) | 8 | 8 |
| **blind (neither)** | **47** | **43** |

The 43 blind decompose usefully: **5 are `pre_spawn_fatal`** — never spawned, `$0` by definition — leaving
**38 that actually ran** (35 T4, 3 CPU; **zero H100**). 37 of those carry `max_seconds`; 1 stale T4 does not.

Two corrections to the numbers I was handed: the blind count before backfill is **47**, not 49 (8 priced
and 10 timed overlap on 2 calls). And the ledger-wide timed count is **178 of 275** — it is specifically
the *cap window* that is blind, not the ledger as a whole.

**These 43 no longer block the budget question.** Billing settled it directly. They remain a
*forensic attribution* gap — we know the total, not which call spent it.

---

## Bounded spend statement vs the $20 cap

**MEASURED (authority).** Modal workspace billing API, workspace `adpena`,
`2026-07-09` → `2026-08-16` inclusive: **`$18.61243961`**.

Window construction: two calls, `--start` inclusive / `--end` exclusive —
`[2026-07-09, 2026-08-09)` + `[2026-08-09, 2026-08-17)`. No overlap, no gap.

**Cross-checked two independent ways**, both from the same API output:
by app-day = `$18.61243961` (350/350 rows parsed); by app-day-resource = `$18.61243951` (879/879 rows).
**Residual `1.0e-7` USD** — Modal's own per-resource rounding (each line rounded to 8 dp, so the sum of
rounded parts ≠ the rounded whole). Not a parse error.

| | USD | share |
|---|---:|---:|
| T4 | 5.9079 | 31.7% |
| CPU | 5.0260 | 27.0% |
| H100 | 4.9208 | 26.4% |
| Memory | 2.7578 | 14.8% |
| **total** | **18.6124** | |

**Uncertainty.** Low, and structural rather than statistical. The number is the provider's own billing
record, not an inference. Residual risks, stated: (a) it is *workspace*-scoped — all 15 app descriptions
are project apps (`comma-*`, `clickpolish-*`, `ddm-*`), so I found no foreign workload, but I cannot prove
the workspace is used for nothing else; (b) very recent usage may not have fully settled in Modal's
billing pipeline, which biases the figure **low**, not high; (c) it interprets the cap as spend *from*
2026-07-09, matching when the constraint was set.

**Rejected bounds, recorded so no one rebuilds them.** The `max_seconds` ceiling over the 38 blind
running calls is 48.83 h → `$28.81` at published T4 rates: it exceeds the cap and is a *timeout*
ceiling, not runtime, so it is too loose to constrain anything. And the configured-rate model
over-prices: on the single cross-checkable H100 row, configured `$7.911`/hr vs realized `$3.941`/hr =
**2.01×** over-prediction. (The figure I was handed for this was 2.8×; the measured ratio is 2.01×.)
Both models are now moot — billing is authority and neither is needed.

---

## What this changes

1. **The cap is nearly spent.** `$1.3876` left, ~1.2 days at current burn. Any dispatch decision must
   be made against that number.
2. **`cost_actual_usd` should not be trusted by any consumer.** It provably mixes measured entries,
   rate-derived estimates, and hand entries under one name. Either split it into
   `cost_measured_usd` / `cost_estimated_usd`, or have the harvester populate it from
   `modal billing report` and nothing else.
3. **`modal billing report` exists and is authoritative, cheap, and read-only.** No arm had used it.
   The whole ledger-archaeology exercise — mine included — was reconstructing from proxies a number
   the provider will simply state. **Query billing first; use the ledger for per-call attribution.**
4. **The dual-axis result schema still emits no timing**, so today's poller cure cannot help it.
5. **`/Volumes/VertigoDataTier` is 100% full (953 MiB free)** — flagged, untouched, ALWAYS-KEEP-THE-PAYLOAD
   respected. Nothing was moved or deleted by this arm.

---

## Evidence (durable, no `/tmp`)

- `.omx/state/modal_call_id_ledger.jsonl` — 822 rows; first 812 byte-identical to pre-backfill.
- `.omx/research/ddm_mb1_modal_billing_20260816/modal_billing_reconciliation.json` — totals by day,
  app, and resource; cross-check residual; window construction.
- `.omx/research/ddm_mb1_modal_billing_20260816/modal_billing_{agg,byresource}_*.txt` — raw provider output.

Reproduce the authoritative figure:

```bash
.venv/bin/modal billing report --start 2026-07-09 --end 2026-08-09 --resolution d
.venv/bin/modal billing report --start 2026-08-09 --end 2026-08-17 --resolution d
```

(`--resolution` accepts only `d`/`h`; daily reports span ≤ 31 days; workspace billing requests are
rate-limited — a limit error is transient, retry, do not treat it as empty data. I hit exactly that and
briefly computed a wrong cross-check from a silently-empty file.)

---

## Ending on measurement

`$18.61243961` spent of `$20.00`. `$1.38756039` remains. `93.1%` consumed. Last-7-day burn
`$1.1808`/day → `1.2` days of headroom. 20 of 63 cap-window calls timed, 43 blind, 10 rows appended,
0 rows mutated.
