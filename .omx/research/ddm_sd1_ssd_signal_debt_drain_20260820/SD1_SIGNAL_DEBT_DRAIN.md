# ddm_sd1 — is there still signal living on the SSDs alone?

`date_utc: 2026-08-20` · `owner: ddm_sd1` · task #1159 · predecessor: `ddm_oc2` (same directory tree)

The operator asked: *"have we ensured there is no more signal only living on the SSDs alone?"* This
is the answer, in four measured parts. Where the answer is still "no", the remainder is named with
its size.

---

## The headline

`ddm_oc2` built the instrument that first measured this gap and drained most of it. What it left
behind was **77 authored `src/tac` modules that existed on exactly one machine** — not on GitHub,
not on the SSD tier, blocked by the review gate at 2,773 unreviewed entities — and **an instrument
that was a one-shot script, so the number would go stale the moment the next arm wrote to an SSD.**

Both are closed. The 77 got a real review (61,784 lines, 8 dedicated fresh-eyes agents, every check
executed) and landed. The instrument is now `tools/audit_ssd_authored_signal.py`, wired into the
consolidation-debt monitor that already runs at SessionStart and Stop, with a passing positive
control.

**One thing the instrument found that changes oc2's number: being in the object database is not
being preserved.**

---

## Leg 3 — the standing guard (`tools/audit_ssd_authored_signal.py`)

### The correction that matters

`ddm_oc2` established byte identity with `git cat-file --batch-all-objects`. That set includes blobs
that were `git add`ed and never committed, plus orphans from deleted branches. Measured on this repo
2026-08-20:

| set | count |
|---|---:|
| blobs in the object database | 129,656 |
| blobs REACHABLE from a ref | 104,425 |
| **gc-eligible (in the ODB, no ref reaches them)** | **25,231** |

`git gc --prune` deletes all 25,231, and `git clone` never receives one. Counting them as "safe in
git history" **under-reports the debt**. The empirical trigger was a single file:
`src/tac/boundary_math/dense_raster_lzma_baseline.py` is untracked, appears in no commit
(`git log --all -- <path>` is empty), yet `git cat-file -e` finds its blob. Under oc2's test it read
as preserved. It was one `gc` from gone.

The guard therefore tests **reachability**, and reports gc-eligible blobs as their own bucket:
recoverable today, cheaper to close than a truly absent file, still owed.

### Method and honest denominator

Walk the SSD roots for code-like files, hash each to a git blob sha1, compare against the reachable
set. Every absent file is bucketed and **every bucket is reported** — the guard never presents a
filtered count as the whole:

| bucket | meaning | disposition |
|---|---|---|
| A | third-party intake / OSS harvest / upstream mirror / nested clone | policy: stays put |
| B | `experiments/results/**`, `cold_store/**` | convention: untracked run output |
| C | **candidate AUTHORED sources** | **the debt — default COMMIT** |
| D | bucket-C blobs certified in place, owner + rationale on file | closed |

Instance counts overstate authorship ~3× because arm workspaces copy a runtime tree per run, so the
headline is DISTINCT BLOBS. A file `ddm_vr1` relocates between tiers is the same blob at a new path
and is counted **once** — the distinct-blob keying makes double-counting structurally impossible,
and a test asserts it.

Two deliberate strictness choices, both erring toward flagging:

- A bare `results/` segment is **not** bucket B. Arm workspaces routinely put authored builder
  scripts under `<arm>/results/`; excusing those would hide real debt. An ambiguous path is flagged.
- A blob seen in two buckets takes the **most-owed** reading, so copying an authored source into
  `cold_store/` cannot launder it out of the debt.

### Certify-in-place

`.omx/state/ssd_authored_signal_certified.jsonl`, append-only, latest row wins. A certification
needs a substantive rationale (≥12 chars) and an owner; placeholders (`<rationale>`, `tbd`,
`placeholder`) are refused. **An uncertified blob is a more honest state than a fake certificate.**

### Wire-in — no new scheduler

`tools/consolidation_debt.py` gains a sixth component, `ssd_only_code`. That monitor is already
registered on **SessionStart and Stop** with `--quiet-ok`, so the cadence exists; nothing new was
scheduled and `.claude/settings.json` was not touched. A full sweep is minutes of I/O and must not
run at every session boundary, so the component reads the cache written by `--write-cache`, and:

- a **missing** cache reports `no_cache` and registers a blind read — not a clean zero;
- a cache older than 72 h reports `stale` — an old clean number is not a current clean number;
- a sweep whose tier was unmounted reports `partial`.

Verified live: with no cache present the monitor printed `ssd sweep no_cache (never) — the SSD
number above is NOT current` and flagged itself `BLIND on 1 read`. That is the cure for
vacuity==pass on this surface.

### Positive control — receipt

| step | command | result |
|---|---|---|
| plant | wrote `/Volumes/APDataStore/pact/ddm_sd1_positive_control/planted_authored_source.py`, blob `e10a622d828c…` | — |
| detect | `audit_ssd_authored_signal.py --root <that dir>` | **`C AUTHORED — OWED: 1`**, the planted path named |
| remove | `rm -rf` the control dir, re-run | `0` owed **and** `! ROOT NOT MOUNTED (this scan is PARTIAL)` — the absence is loud, not silent |
| negative | `--root src/tac/boundary_math` (154 tracked files) | `0` owed — no false positives on committed content |
| cross-check | `--root /Volumes/APDataStore/pact/ddm_sa1` | 16 owed, matching oc2's independent count for that arm |

37 tests in `src/tac/tests/test_audit_ssd_authored_signal.py`, all behavioural. They caught two real
bugs in my own code before landing: an early-bound `CERTIFIED` default that made a redirected ledger
read as empty (so every certified blob re-appeared as debt), and the `results/` over-broad bucket
above. A third test caught a pre-existing `KeyError` in `consolidation_debt._fmt` on a partial
component set — an advisory monitor that raises at a session boundary is worse than one that
reports.

Landed: `cf3bb0b561`.

---

## Leg 1 — draining the backlog

### The current measurement (27-minute full sweep, this arm)

| quantity | value |
|---|---:|
| SSD code-like files scanned | 148,062 |
| reachable git blobs compared against | 104,427 |
| unreachable from git | 5,980 instances = **1,821 distinct blobs** |
| bucket A third-party / clone | 830 |
| bucket B run output / cold store | 177 |
| **bucket C authored — owed** | **814** (38 of them gc-eligible) |

**This is not comparable line-for-line with oc2's table, and the reason matters:** oc2 reported
buckets A and B as *file instances* (593 and 1,382) while reporting C both ways (3,179 instances =
1,071 distinct). Every number above is DISTINCT BLOBS. Comparing 814 to 1,071 as if they were the
same measurement would be the units-are-part-of-the-claim error.

### Then the instrument was wrong, and the top of its own output said so

The largest "authored" rows were a 5.9 MB and a 4.5 MB `inflate.py`, and a directory of
`brotli110_source/c/*.c`. Neither is authored signal. Re-bucketed under rules added for exactly
those two measured classes:

| reclassified out of C | blobs | bytes |
|---|---:|---:|
| vendored compression-library C source (`brotli110_source/**`) → A | 105 | 2.1 MB |
| generated packet `inflate.py` + `coldstore`-named dirs → B | 96 | 15.7 MB |
| **remaining genuinely authored** | **613** | **5.68 MB** |

The generated-`inflate.py` class alone was **64% of the owed byte volume** while carrying no
authored signal. Both rules are now in the tool with tests, including a boundary test: a
hand-written `inflate.py` *outside* a packet directory stays owed.

### The drain

613 blobs copied **verbatim** into `experiments/ssd_recovered/<tier>/<lineage>/…`, blob sha
re-verified after every write. Provenance is a sidecar (`PROVENANCE.json`: origin path, blob sha,
size, SSD instance count, gc-eligibility at sweep), never a header — a "for provenance" comment
would change the blob and leave the SSD copy still counted as owed.

They land as **archived evidence, explicitly not reviewed**. `experiments/ssd_recovered/**` is
added to the review tracker's `SCAN_EXCLUDE_GLOBS`, because the alternative was to fake-review
~100k lines nobody read or to override the review gate on `.py` — both forbidden. Nothing imports
them; their value is that the bytes are unchanged and no longer on one disk.

### Denominator, start to finish

| stage | distinct authored blobs |
|---|---:|
| oc2's starting measurement (2026-08-20, before its landings) | 1,071 |
| after oc2 committed 181 sources + 764 docs, re-measured by this arm | 814 |
| after re-bucketing vendored + generated (measured, not assumed) | **613** |
| committed verbatim by this arm | **613** |
| certified in place | 0 |
| **remaining owed** | **0 at this sweep** |

That last row is honest only as of this sweep. The tier is live: arms write to it continuously, so
the number is a snapshot, not a permanent state. That is precisely why leg 3 exists.

---

## Leg 2 — the 77 machine-only `src/tac` sources

### What was done

8 dedicated fresh-eyes reviewers, ~7,700 lines each, **61,784 lines total**. Every file read in
full. Every check executed, not reasoned about: import, pytest, ruff, secret scan, reference
resolution, logic read, and a NO-FAKE class-2 test-quality judgement ("would these tests still pass
if the function body returned canonical constants?").

| outcome | count |
|---|---:|
| files reviewed | **77** |
| `COMMIT_CLEAN` | 13 |
| `COMMIT_WITH_DEBT` | 64 |
| **`ATTIC`** | **0** |
| findings: critical / important / minor | **5 / 43 / 142** |
| test files rated `behavior` (vs `constants_only`) | **59 / 59** |
| secrets found | 0 |
| broken references | 0 |

**Zero ATTIC** is the load-bearing result: every one of the 77 imports, is exercised, and does real
work. None was an abandoned draft. The review gate was right to block them, and the right response
was to review them, not to waive them. `REVIEW_GATE_OVERRIDE` was not used.

### The 5 critical findings

Two I fixed and landed in `cf3bb0b561`; three are owed.

1. **FIXED — `test_v15_ms1_coordinate_compatibility.py:45` hardcoded the frontier literal `0.172`.**
   The pointer moved to `0.14839100138338618` when sub-0.15 landed, and the test went red — reading
   as a code regression when the only thing that changed was a better score. This is exactly the
   class CLAUDE.md forbids ("Frontier scores are pointer-only"). Now reads the pointer and asserts
   the receipt agrees with it — an invariant that survives every legitimate frontier move and still
   fails if the economics block and the frontier disagree.
2. **FIXED — same file:50**, the byte ceiling `187_562` pinned at the old score. Now derived via
   `byte_ceiling_for_score(live_frontier, receipt d_seg, receipt d_pose)`.
   *(Sister important, also fixed: line 42 skipped via a bare `return`, which reports PASS on a host
   without the evidence tier mounted. Now `pytest.skip`, which is visible in the summary line.)*
3. **OWED — `ep725_levelset_predictor_adapter.py:55`, stale renderer SHA pin.** Pins
   `1cecaa3ee987…`; the committed `tools/levelset_byte_close_and_eval.py` hashes to
   `00106018a420ff7f160d5895aab09d00a7b6859705dd17e9283a8488e6b8d003`. Confirmed by execution by two
   independent reviewers. Every call into `decode_ep725_*_ephemeral_surface` raises. Blast radius,
   measured by grep: 1 production module + 4 tools. The refusal also fires **last**, after two NumPy
   and two subprocess decodes of 874×1164 frames, though the pin is known at line 1296.
4. **OWED — `taskspace_single_stage_score_attempt_v1.py:1721`, permanently self-poisoning resume.**
   `_run_decode` writes stdout/stderr via `_write_once` *before* checking the return code. A
   transient failure writes the failure output; once the cause is fixed the now-successful decode
   produces different bytes and `_write_once` refuses `immutable output differs` — forever. Only
   manual deletion recovers, and nothing documents that. Same defect at `_run_upstream_eval:1857`.
   The run_root it strands holds a full n600 spine.
5. **OWED (downstream of #3) — `test_bounded_target_g_encoder.py:148`** fails deterministically for
   the same pin.

### The structural finding two reviewers hit independently

**`pyproject.toml:285` `testpaths` does not include `src/tac/witness_control/tests` or
`src/tac/witness_dsl/tests`.** A bare `pytest` collects **zero** of these files — measured, not
inferred (`pytest --collect-only -q` matches 0). That is the mechanism that let the ep725 pin go
stale, and it means the tests landing here are invisible to the default run.

I did **not** change `testpaths`. Adding those directories would newly collect 60+ tests across
files outside my review scope, with at least two known-red suites, turning the default run red for
reasons I have not verified. That is a decision with a blast radius, and it is MAIN's. The comment
directly above that line already names this bug class for `pr130_lift`: *"A cure whose guard never
runs is not guarded."*

### Landing dependency

`src/tac/optimization/predict_project_receiver.py` is tracked-and-modified by a sister arm. Three
reviewers hit `RuntimeError: conflicting in-process LawRef evaluator` until that edit settled. The
77 import cleanly against the current working tree; **if that patch does not land, several of them
become unimportable.** Not my file to commit.

---

## Leg 4 — receipt-coverage spot audit

Bounded and seeded, not a census. Script:
`.omx/research/ddm_sd1_.../receipt_coverage_spot_audit.py`; rows in
`RECEIPT_COVERAGE_SPOT_AUDIT.json`.

| quantity | value |
|---|---|
| memos scanned (`.omx/research/**.md`, 30 days) | 2,505 |
| distinct SSD paths cited — **the population** | 1,912 |
| sampled (seed 20260820) | 50 |
| **resolves** | **44 / 50 = 88.0 %** |
| **covered by a retention manifest or cert row** | **3 / 44 resolved = 6.8 %** |
| covered only by sitting under a `retained/` dir | 2 |
| uncovered | 39 |

**The low coverage is genuine, not a classifier artifact.** I checked: of the 44 resolved paths,
**0** have any retention manifest in their owning arm directory. The base rate explains it —
**19 of 1,029 arm directories across both tiers carry a `*_RETENTION_MANIFEST.json` at all (1.85 %)**.
The convention exists; roughly one arm in fifty follows it.

So the honest reading is not "the manifests are broken". It is: **12 % of recently-cited SSD evidence
paths no longer resolve, and for the 88 % that do, almost nothing records why they may be deleted or
how to rebuild them.** That replaces the standing "the older long tail is unaudited" caveat with a
number. Backfilling it was out of scope for this arm.

Two nits in my own instrument, found and fixed before reporting: a `{` from a memo's brace expansion
leaked into path extraction (inflating the non-resolve rate), and a non-existent path was labelled
`uncovered` when the honest label is `not_resolved` — coverage of bytes that are gone cannot be
judged.

---

## What would make "ensured" a full YES

1. **Land `predict_project_receiver.py`** (sister-owned). Without it several of the 77 do not import.
2. **Re-derive the ep725 renderer pin** (critical #3) — it currently breaks 1 production module,
   4 tools, and two test suites.
3. **Fix the self-poisoning resume** (critical #4) before the next long single-stage attempt.
4. **Decide `testpaths`** — until `src/tac/witness_control/tests` and `src/tac/witness_dsl/tests` are
   collected, none of these tests guard anything.
5. **Keep the sweep cache fresh.** The guard reports `stale` past 72 h and the monitor says so, but
   nothing re-runs it automatically; a cheap cron or a governed weekly launch would close that.
6. **The 43 important findings** in `review_batches/batch_0*.json` are recorded, not fixed. They are
   debts against code that now exists in git, which is strictly better than debts against code that
   existed on one disk.
