# ddm_hv1 — HARVEST→POINTER AUTOPILOT: the exact row's ten consequences now come from the payload (2026-09-04)

Tokens: `[no-triality] [p0-ledger-ok]`

## Verdict first

The apparatus now writes every consequence of an exact contest row. I replayed both of
today's pointer moves through it in an isolated scratch state and diffed the output
against what MAIN wrote by hand. The packet **reproduced every load-bearing number**
and **found five hand-errors** MAIN's own memos carry, two of which are live compliance
REDs. The replay also found **five bugs in my own packet**, all fixed and tested.

This arm did not move the frontier. It removes the hand steps that move it.

## What landed

| file | what it is |
|---|---|
| `src/tac/pointer_move.py` | the pure functions: recompute S, cross-check the evaluator's report, re-derive both sub-0.12 corners, render the memo / POINTER_LINE / event row |
| `tools/pointer_move_packet.py` | the 10-stage CLI, dry by default, `--repo-root`-isolatable |
| `src/tac/modal_source_snapshot.py` | the immutable clone of every Modal-mounted source tree |
| `tools/fire_modal_auth_eval.py` | stage 4b SNAPSHOT; refusal archiving; threads `--instance-job-id` to the poller |
| `tools/modal_harvest_poller.py` | canonical terminal claim row with FULL shas; axis-derived status; stages the packet when a row beats the pointer |
| `src/tac/tests/test_pointer_move_packet.py` · `test_modal_source_snapshot.py` | 25 tests, in `testpaths` so they run by default |

Commits `feac0ce30`, `9e796af7b`.

## Replay verification — the diff table

Method: for each move I rebuilt a sparse scratch repo-root holding only what
`tac.frontier_scan` and the packet read, **removed that move's anchor mirror**, refreshed
the pointer to reconstruct the true PRE-move state from the data (not by hand), restored
the mirror, and ran the packet with `--repo-root <scratch> --no-custody --apply`.

Reconstructed pre-states, MEASURED: pre-fs1 = `0.14797617125559104` @ 180,002 B, sha
`cbb8d928…` (afr1) · pre-fs2 = `0.14786319521362173` @ 180,022 B, sha `50fcaf1a…` (fs1).
Both match the "prior pointer" line of the committed memos exactly.

### fs1 (24th move)

| artifact | committed by hand | packet | verdict |
|---|---|---|---|
| S | 0.14786319521362173 | same | MATCH |
| rate term | 0.11986926045895953 | same | MATCH |
| seg term | 0.020139 | same | MATCH |
| **pose term** | 0.007854934754662**192** | 0.007854934754662**193** | **MAIN hand-error** (1 ulp; no formula variant produces …192) |
| Δ vs afr1 | −1.1297604196930378e-4 | same | MATCH |
| held distortion | 0.027993934754662192 | same | MATCH |
| gap to 0.12 | 0.02786319521362174 | same | MATCH |
| rate corner | ≤ 138,176.5 B → −41,845.5 B | same | MATCH |
| distortion corner | 1.3074e-4 → 214.1× | same | MATCH |
| B_max / margin | 180,218.347 / 196.347 B | same | MATCH |
| projection error | −2.6e-6 | −2.615043805359596e-06 | MATCH |
| pointer after | fs1 anchor | fs1 anchor (score, sha, bytes identical) | MATCH |
| Catalog #316 | **failed twice** on the way | 0 violations, first pass | **packet cures** |
| terminal claim row | 2 rows; the first mistyped an archive sha | 1 row, 5/5 GREEN | **packet cures** |

### fs2 (25th move)

| artifact | committed by hand | packet | verdict |
|---|---|---|---|
| S | 0.14784474152757654 | same | MATCH |
| **rate term** | "0.11986992607…" | 0.11986992631791266 | **MAIN hand-error** (2.5e-10) |
| **pose term** | "0.007836134…" | 0.007835815209663893 | **MAIN hand-error** (3.2e-7) |
| **gap to 0.12** | 0.02784474152757653 | 0.027844741527576544 | **MAIN hand-error** |
| Δ vs fs1 | −1.8453686045194484e-5 | same | MATCH |
| rate corner | ≤ 138,205.2 B → −41,817.8 B | same | MATCH |
| distortion corner | 1.3007e-4 → 215.1× | same | MATCH |
| B_max / margin | 180,218.347 / 195.347 B | same | MATCH |
| projection error | +3.69e-6 | +3.6917959990168114e-06 | MATCH |
| pointer after | fs2 anchor | fs2 anchor (score, sha, bytes identical) | MATCH |
| **runtime-tree sha in the claim row** | `915d25f93ad6` (12 hex) | full 64 hex | **MAIN hand-error → live RED** |

The three fs2 memo intermediates are wrong in prose only: the S they are said to sum to
is correct, so no downstream number inherited them. They are still hand-errors, and they
are exactly the class a template eliminates.

### The claim row, checked against the real compliance gate

I ran `scripts/pre_submission_compliance_check.py`'s own `inspect_dispatch_claims` over a
ledger holding the poller's generated row, and over the LIVE ledger holding MAIN's rows:

| check | packet/poller | MAIN committed (fs1) | MAIN committed (fs2) |
|---|---|---|---|
| `dispatch_claim_terminal_row` | GREEN | GREEN | GREEN |
| `dispatch_claim_successful_exact_eval_terminal_row` | GREEN | **RED** | **RED** |
| `dispatch_claim_terminal_archive_sha_bound` | GREEN | GREEN | GREEN |
| `dispatch_claim_terminal_runtime_tree_sha_bound` | GREEN | GREEN | **RED** |
| `dispatch_claim_prior_active_row` | GREEN | GREEN | GREEN |

**A FOURTH RED, on both lanes, that ps1 did not report:** MAIN's status string
`completed_modal_auth_eval_harvested_S_…` does not start with `completed_contest_cuda`,
which is what `SUCCESSFUL_EXACT_EVAL_TERMINAL_STATUS_PREFIXES` requires. The poller now
emits `completed_contest_cuda_exact_eval_harvested` (or `…_cpu_…`, derived from the
receipt's axis — the old code hardcoded cuda for every axis, which would have mis-closed
every CPU row).

## Bugs the replay found in my own packet (all fixed, all tested)

1. **Every stage silently no-oped.** Sub-tools ran with `cwd=<repo-root>` and RELATIVE
   tool paths; a scratch root has no `tools/`, so all five exited non-zero and the packet
   printed success over nothing. Cure: absolute tool paths, `cwd=REPO`, and a hard refusal
   on a non-zero pointer-refresh rc. This is why a replay is not optional.
2. **A replay wrote real custody.** The first run copied the archive to
   `/Volumes/APDataStore/pact/custody_pointer24` — the tier ROOT, not under the lane.
   Cure: `--no-custody`, and the default subdir is now `<lane>/custody_pointer<N>`.
   (The stray directory was removed.)
3. **`lane_maturity.py` and `main_hot_state.py` are not `--repo-root` aware** and would
   have written LIVE state from an isolated replay. Cure: skipped under a non-default
   root, with the argv still recorded so the replay can diff what WOULD run.
4. **The snapshot digest double-counted nested mounts** (`src` and the resolved
   `src/tac`), so it depended on a redundant path list. Cure: `dedupe_nested`.
5. **A hardcoded "T4 n600" axis label.** Cure: derived from `gpu_model` and `n_samples`.

## The Modal source snapshot — the fs2 build race, closed

The charter asked for `git archive HEAD`. **That is impossible here, MEASURED:**
`upstream/` has **0 tracked files** of 19,677 on disk (it is pinned outside git) and the
two intake clones have 0 of 227. A `git archive` snapshot would ship an empty `upstream/`
and the remote would die at `missing evaluate.py`. So the snapshot clones the *mounted
paths themselves*, and the mount list is parsed from the app module's AST — never
hand-typed, so a new mount is snapshotted with no edit here.

MEASURED on the real CUDA entrypoint: **16,060 files / 780 MB in 6.7 s**, via APFS
`clonefile(2)` (`cp -Rc`), consuming ~0 extra space. Race self-test: a background loop
rewrote a mounted `src/` file 40 times during the build; the manifest digest and an
honest re-digest afterwards are **identical**, the live-tree digest **differs**, and the
mid-race value is frozen in the snapshot. The race is structurally closed.

Fail-closed: every mounted path's file set is compared in both trees and any asymmetry
REFUSES (rc=8) rather than firing a short mount, which would spend the meter and crash.
`add_local_python_source("tac")` resolves by import, not by path, so the dispatch runs
with `PYTHONPATH=<snap>/src` — and rather than trust that, the tool EXECUTES the
resolution with the exact cwd/env the fire will use and refuses any module that resolves
outside the snapshot. Positive and negative controls both measured.

**Honest limit:** I could not fire, so the snapshot is proved locally (completeness,
immunity, import resolution, dry-run manifest) but has **never carried a real Modal image
build**. The first real fire is the remaining test. `--no-source-snapshot` restores the
old behaviour in one flag if it goes wrong.

Refusal archiving (item 4) works: a re-fire into a directory holding `FIRE_REFUSED.json`
moves it to `refusals/<utc>.json`, records the path in the manifest, and proceeds — no
more `_r2` directory split for a bookkeeping reason.

## What the apparatus now writes without MAIN

At harvest: the anchor mirror · the call-id ledger row · the **canonical terminal claim
row with both full shas, the components, and an axis-correct status** · and, when the row
beats the pointer on its axis, a staged `POINTER_MOVE_PLAN.json` with the exact
`--apply` command.

On `--apply`: S recomputed from components and cross-checked against the evaluator's own
report (thousands separators parsed) · archive re-hashed against the row · seal checked ·
pointer refreshed · citation surfaces regenerated **and the #316 gate asserted** · lane
registered and three gates marked · custody duplicated to the other SSD tier and the copy
re-hashed · the memo rendered with the delta table, projection fidelity, both re-derived
corners and the zero-distortion margin · the hot-state POINTER_LINE set · the event row
appended · the serializer command printed.

## What is NOT automated, and why

The memo's **prose** — the mechanism, the non-claims, the equations leg — is a claim about
the world only the arm that produced the row can make, so the packet takes it as input.
**Publishing** is untouched: PR #140 stays operator-gated. The **commit** prints by default
so MAIN reviews the memo before it lands. The **move ordinal** is derived from the events
ledger; the ledger was seeded with moves 24 and 25 so the next is 26 with no argument.

## What this does NOT claim

No score moved. No Modal fire was made and none of this is measured on contest hardware.
The snapshot's effect on a real image build is UNVERIFIED. The packet has never run
against the LIVE state — both replays ran against isolated scratch roots — so its first
production run should be watched. Nothing was published.

## Also found, not mine to fix

`src/tac/tests/test_claim_lane_dispatch.py::test_terminal_prefixes_constants` fails on
`main` before my changes: `tools/claim_lane_dispatch.py` carries the prefix `"stale_"`
where the test expects `"stale_assumed_dead"` and `"stale_superseded"`. Whoever widened
the prefix left the test behind.

## Equations leg (`tac.canonical_equations`)

No new equation and no re-fit: this arm produced no measured ΔS. It does give
`exchange_ratio_noise_floor_v1` a mechanical producer — `tac.pointer_move.target_arithmetic`
re-derives the 6.658589531221714e-7 S/B exchange and both corner demands from the row at
every move, so the "binding numbers expire and nobody re-derives them" law is now executed
rather than remembered.

Own-vehicle frontier (unmoved by this arm): **fs2 — S 0.14784474152757654 @ 180,023 B
[contest-CUDA T4 n600]**, archive sha `a8f3a379…0427bb6`.
