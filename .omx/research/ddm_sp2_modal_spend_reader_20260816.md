# ddm_sp2 — the canonical Modal spend reader, and the cap answered by a tool

**Arm:** `ddm_sp2_modal_spend_reader` · **UTC:** 2026-08-16 · **Repo:** `/Users/adpena/Projects/pact`
**Charter:** build the canonical Modal spend reader so cap-respect stops being a guess.

STORES CONSULTED: `tools/corpus_query.py 'modal billing spend'` (research 8728 / equations 897 /
memory 2136 / dag 925 / council 297 / tasks 548 / docs 96); `.omx/research/ddm_mb1_modal_ledger_backfill_20260816.md`
and its raw provider outputs; `.omx/state/modal_call_id_ledger.jsonl`; `git log`; `tools/` listing and
a `billing` grep across `tools/ src/ experiments/ scripts/`. Deliberately NOT consulted: the harness
TaskList (arms cannot see it), so task #381 is re-derived from `ddm_mb1`'s memo, not from the ledger row.

---

## THE ANSWER

**Modal spend `2026-07-09` → `2026-08-17` is `$18.61972873` of the `$20.00` cap.
Headroom is `$1.38027127`. The cap is 93.1% consumed.** [MEASURED — `modal billing report`,
workspace `adpena`, 350 rows across 2 chunks.]

Receipt: `.omx/state/modal_spend_receipts.jsonl` (schema `modal_spend_report_v1`), carrying the
per-window totals, the row count, and the two exact commands, so the number is re-derivable.

Reproduce in one command:

```bash
.venv/bin/python tools/modal_spend_report.py
```

**The total is a moving floor, and that is the honest reading.** Four reads today, all consistent:

| reader | UTC | total |
|---|---|---|
| `ddm_mb1`, by hand | 22:52 | `$18.61243961` |
| this tool | 23:29 | `$18.61852110` |
| this tool | 23:31 | `$18.61852110` |
| this tool, post-fix | 23:52 | `$18.61972873` |

Two reads 92 s apart agreed to the cent; the drift across the day is usage still accruing, not
disagreement. The figure is also biased LOW — very recent usage may not have settled in the
provider's billing pipeline.

---

## What was built

`tools/modal_spend_report.py` — one surface, no duplicate. A grep across `tools/`, `src/`,
`experiments/`, and `scripts/` found **no** existing spend-reading tool. `ddm_mb1` measured the number
by hand today and wrote in its own memo that `modal billing report` "exists and is authoritative...
No arm had used it" — it left the tool unbuilt. Nothing was extended because nothing existed.

It chunks any window to the provider's 31-day limit, refuses on any unreadable window, appends a typed
receipt, and cross-checks the ledger.

### The one property that matters

**An unreadable window is reported `UNREADABLE`. It is never reported as `$0.00`.**

That is the shape this tool exists to refuse. `ddm_mb1` hit the workspace rate limit, redirected stdout
to a file without checking the return code, and briefly computed a cross-check from a silently empty
file. A spent cap read as untouched is the most dangerous output a spend reader can produce.

The tool mirrors `src/tac/process_liveness.py`: blindness and emptiness demand opposite responses, so
they must be distinguishable at the source. Concretely — the provider's JSON `[]` is a real day with no
usage and READS as zero; empty stdout REFUSES.

Five guards, in order: non-zero return code · rate-limit marker on stderr (scanned even when the return
code is 0) · empty stdout · stdout that is not a JSON list · a row with a missing or non-numeric cost.
A single bad row refuses the whole window, because a partial sum is worse than a refusal — it looks
like an answer. The same rule holds across chunks: one unreadable chunk aborts the whole reading.

Money is summed as `Decimal`, never float.

Exit codes keep the distinction alive downstream: `0` readable · `2` UNREADABLE · `3` readable but over
cap. On a refusal the banner goes to **stdout**, not only stderr, so a caller who redirects stdout and
ignores the return code gets a loud refusal instead of an empty file — the original defect, closed at
the surface where it happened.

---

## What I re-derived at the source (and one premise I corrected)

| premise handed to me | verdict |
|---|---|
| `modal billing report` has a hard 31-day daily-range max | **CONFIRMED.** rc=1, stdout **empty**, stderr `"Billing report range limit exceeded. Daily reports cannot span more than 31 days."` It is refused, not truncated. |
| It is rate-limited | **CONFIRMED, live.** Chunk 2 of my first real run came back `rc=1` + `"Rate limit exceeded for workspace billing report requests."` |
| "the error comes back as a normal-looking frame" and "an unguarded parse reads it as $0.00" | **CORRECTED.** The frame is `rc=1` with **empty stdout**. The $0.00 does not come from parsing the frame — it comes from redirecting stdout without checking the return code. The mechanism is confirmed; the shape is not what the seed said. The tool guards both anyway, because I cannot prove the rate limit never arrives with `rc=0`. |
| `--start` inclusive / `--end` exclusive | **CONFIRMED against real data, at zero API cost.** Chunk `[07-09, 08-09)` contains **zero** rows dated `2026-08-09`; chunk `[08-09, 08-17)` begins at `2026-08-09`. No gap, no double count. |
| `--json` output | **NEW.** `ddm_mb1` parsed rich text tables, which truncate `object_id` and `description`. `--json` returns them whole, with cost as an 8-decimal string. |

**Not re-derived:** task #381's own ledger row. It lives in the harness TaskList, which arms cannot
read. The `$20` cap and its `2026-07-09` start date are taken from `ddm_mb1`'s memo. Labelled, not assumed.

---

## The defect I found in my own code

My first ledger cross-check reported **22 calls and `$0.15`**. `ddm_mb1` had independently reported
**63 calls and `$1.74`** for the same window. The disagreement was mine.

**Cause:** the ledger is append-only and its rows are PARTIAL. The dispatch row carries
`dispatched_at_utc`; later outcome rows omit it. Folding whole-row latest-wins destroys that timestamp
on **136 of 275** call_ids, and the window filter then silently drops them.

**A wrong fold on an append-only store reads as a smaller world, not as an error.** Nothing raised.
The number just got quieter, in the direction that licenses spending.

Folding field-wise — latest non-null value per field — reproduces `63` calls and `$1.7400` exactly.
The fix is pinned by `test_ledger_fold_is_field_wise_not_whole_row`, and the fold name now travels in
the receipt as `ledger_fold: latest_non_null_value_per_field`.

Second defect, smaller: `Decimal` cannot be an argparse `type`. It raises `InvalidOperation`, which
inherits from `ArithmeticError` and **not** from `ValueError`, so argparse does not catch it and
`--cap-usd abc` printed a raw traceback. Fixed with a wrapper that raises `ArgumentTypeError`.

---

## Controls (P4 — no meter without a canary)

**Executed, live, against the real provider** (`TAC_MODAL_SPEND_LIVE_CONTROL=1`, 2 passed):

- **positive** — the billed day `2026-08-15` reads nonzero with nonzero rows.
- **negative** — a >31-day span, with the local guard lifted so the *provider's* own `rc=1` drives the
  refusal, REFUSES. Lifting the local guard matters: without it the control would be testing our
  arithmetic, not the meter.

**Executed in production, unplanned:** the live rate limit fired mid-read on my first real invocation.
The tool refused and wrote an `UNREADABLE` receipt row rather than a number. That row is in
`.omx/state/modal_spend_receipts.jsonl`, timestamped `2026-08-16T23:27:04Z`, `reason: rate_limited`.
The negative control ran itself.

**Deterministic suite:** 52 passed, 2 skipped (`src/tac/tests/test_modal_spend_report.py`). Every
negative control asserts `pytest.raises`, never a field value — a test asserting `total == 0` would
PASS on the broken code.

**Mutation check — proof the controls are not decoration.** Thirteen defects injected into scratch
copies, **every one killed**: drop the return-code guard · drop the stderr rate-limit guard · treat
empty stdout as `[]` · revert the ledger fold to whole-row · move the refusal banner off stdout ·
make `_default_end()` today · empty the receipt's `commands` · empty its `windows` · make the ledger
window end inclusive · restore the bare `429` marker · disable the contradiction check · allow a
negative cap · remove the receipt-write guard. The mutated CLI emits exactly the bug this tool
extincts: `"total_usd": "0"`.

---

## Fresh-context verification round

A fresh-eyes verifier audited the code against the eight required properties with no knowledge of how
it was built. **All eight PASS.** It independently re-derived the chunk arithmetic by enumeration
(39 days covered exactly once, plus a 3,000-window property sweep with zero violations), independently
confirmed end-exclusivity from the raw provider files, and independently reproduced the field-wise vs
whole-row fold split (`63` / `$1.74` against `22` / `$0.15`).

It also found **eight defects my own review missed.** All are fixed:

1. **The contradiction detector was computed and never consulted.** The tool already had the ledger's
   lower bound in hand next to the billed total, and never compared them. Every window returning `[]`
   would print `$0` beside a ledger sum of `$1.74` and exit `0` — the last surviving path to a false
   `$0.00`. Now `ledger_contradiction()` refuses when billing says `$0` while the ledger prices real
   calls. It deliberately does **not** refuse when billing merely comes in under the ledger: the
   ledger carries rate-derived estimates measured over-predicting 2.01x, so that case is flagged, not
   failed.
2. **A bare `"429"` rate-limit marker over-matched** any line containing those digits — an app id, a
   log count — and because that classification triggers backoff, a false positive discarded a
   *readable* window and could cascade into refusing the whole report. Removed; the measured provider
   phrase is caught by `"rate limit"`.
3. **`_default_end()` had no test.** Changing it to today — silently dropping today's spend from every
   default read — survived the entire suite. Pinned.
4. **The receipt's `commands` and `windows` were unpinned**, and re-derivability rests entirely on
   them. Emptying either survived. Pinned.
5. **The ledger window's exclusive end was untested**; flipping it to inclusive survived. Pinned.
6. **A receipt-write failure would have destroyed an expensive rate-limited read.** Now the write is
   guarded and the total prints regardless, with the failure named out loud.
7. Docstring cited a stale total. Corrected, with the accrual explained.
8. `--cap-usd` accepted a negative cap. Refused.

Two more, found while fixing: `ledger_lower_bound_usd`'s default path binds at definition time and so
was not overridable (added `--ledger`), and one of my earlier edits **reported success but was not in
the file** — the pre-commit `receipt_path` ordering fix had been silently lost. An edit's success
message is not proof the change survived; I now read back after editing a file I am also formatting.

---

## The ledger cross-check, and why it is not authority

The receipt carries the ledger's own sum beside the billed total, permanently labelled:

```
ledger_sum_usd            $1.74      ← LOWER BOUND
ledger_is_authority       false
ledger_calls_in_window    63
ledger_calls_unpriced     55  (87%)
```

**`$1.74` against a billed `$18.62` is a 10.7x under-count**, and it cannot be repaired into a total:
billing is app-by-day, calls are not apps. Per `ddm_mb1`, `cost_actual_usd` also mixes measured
entries with rate-derived estimates under one name, and `modal_elapsed_seconds` starts after image pull
so any elapsed×rate model under-prices by construction. The ledger is per-call attribution. It is not a
spend total, and the receipt now says so in the same row as the number, so it can never again be read
alone.

---

## Owed to the operator

**A spend decision.** `$1.38027127` remains — 6.9% of the cap. Price every further Modal dispatch
against that figure, and re-read before committing to one: the total moved `$0.007` in the hour I was
watching it.

Run `tools/modal_spend_report.py --fail-over-cap` in any gate that must respect the cap; it exits `3`
when spend exceeds it and `2` when it cannot tell.

---

## Deliberately not done

The `$20` cap has one code consumer: `tools/cathedral_autopilot_autonomous_loop.py`
(`DEFAULT_CUMULATIVE_CAP_USD = 20.00`), whose `cumulative_spent_usd` starts at `0.0` each process and
accumulates **estimates** (`:600`, `:677`). Its own docstring says this is deliberate — a per-session
envelope guard, not a lifetime total. Seeding it from measured billing would change a gate's semantics
inside a 9,000-line dual-gated dispatch path, on an arm with no authority to loosen a spend gate. I
recorded the finding and left the code alone.

**Owed:** decide whether the autopilot's per-session envelope should be seeded from
`modal_spend_report.read_spend()`. It is a real gap — the operator's cap is a lifetime `$20`, and the
only automated guard resets to zero every process.

---

## Evidence

- `tools/modal_spend_report.py` — the reader.
- `src/tac/tests/test_modal_spend_report.py` — 52 controls, 2 live opt-in.
- `.omx/state/modal_spend_receipts.jsonl` — 4 rows: one live `unreadable`/`rate_limited`, three
  `readable`. Un-ignored in `.gitignore` alongside `modal_call_id_ledger.jsonl`; a local-only receipt
  would defeat its purpose on a fresh checkout, which is how the cap became prose.

## Ending on measurement

`$18.61972873` measured of `$20.00`. `$1.38027127` left. 93.1% consumed. 350 rows, 2 chunks, 4
agreeing reads across 60 minutes. 52 controls green; 13 injected defects, 13 killed.
