# ddm_rv14f — the rv13 code-fix batch (F4 · F7 · F12 · F9/F3 · F13 · F8)

Date: 2026-08-19 · Arm: `ddm_rv14f` · **Score claim: false** · **Pointer moved: false**
· Axis: no measurement claimed — this is apparatus.

`verdict_scope`: **INSTANCE** for every live-artifact fact below (the five mirror rows, the
pointer's CUDA leg, the two receipts, the 241-file rc64 census — each binds only the object
hashed). **CLASS** for the two schema/threading cures (F4 mirror custody, F7 lane-keyed
harvest close), which change what every future row carries, not just tonight's.

STORES CONSULTED: `.omx/research/ddm_rv13_landing_wave_review_20260819.md` (the source, all
13 findings read) · `.omx/research/ddm_rc1x_rc64_recipe_fix_20260819.md` ·
`.omx/research/ddm_fx2_t4_sealed_fire_order_SUPERSESSION_20260819.json` (the precedent this
batch follows) · live artifacts: `experiments/results/modal_auth_eval_mirror/` (5 rows),
`.omx/state/canonical_frontier_pointer.json`, `.omx/state/modal_call_id_ledger.jsonl`,
`.omx/state/active_lane_dispatch_claims.md`, the ck1/ck2/to1/up3 `MODAL_REMOTE_RESULT.json`
receipts on APDataStore, and the rc64 body census over three trees · code:
`src/tac/frontier_scan.py`, `tools/modal_harvest_poller.py`, `tools/make_candidate_seal.py`,
`src/tac/candidate_seal.py`, `src/tac/deploy/claims.py`,
`src/tac/deploy/modal/call_id_ledger.py`, `experiments/contest_auth_eval.py` · memories:
`m53` (negative-existence scope), `m89` (task-ledger split), the bounds-add law,
`structural_beats_procedural_and_the_detector_that_zeroes_on_the_cure`, `m50` (vacuity==pass),
`the-instruments-own-units-level-and-aggregation-are-part-of-the-claim`.

**Posture on the source review.** rv13 is a review, not an oracle. Every claim I acted on was
re-derived from the primary artifact before I wrote a line of cure. One of its premises was
**wrong** (F9, below) and the fix is shaped by the corrected version, not by the memo.

---

## What landed

| # | Defect | Root cause found | Cure | Commit |
|---|---|---|---|---|
| F12 | Pointer CUDA leg `archive_bytes`/`lane_id`/`measured_at_utc` all null | `frontier_scan.load_experiments_results_anchors` built `extra` with 2 keys while the mirror wrote all three | carry them; accept both size spellings | `319b971d7a` |
| F4 | One archive sha carried two contradictory passing contest-CUDA scores | mirror schema v1 had no runtime-tree field; scanner keyed on archive alone | v2 writer emits `runtime_tree_sha256`; scanner stamps + warns; sha-verified backfill of the 5 live rows | `319b971d7a` |
| F7 | Every completed fire looked like a phantom to the claim reconcile | poller had `--lane-id` and used it only for the mirror — the `harvested` ledger row dropped it | thread `lane_id` + the scored facts; auto-close the terminal claim | `319b971d7a` |
| F9/F3 | Bounds hand-typed into seals: one-row division (2.00× over), addends summing to half, linearized pose bound | no helper existed; `--falsifier` is free text | `tac.report_8dp_bounds` + `tools/report_8dp_delta_bound.py`; seal composes it | `46685755a3` |
| F13 | Four bodies wear `rc64_backend.c`, not two — cost us twice | no registry; name-keyed search has 241 hits | `reverse_engineering/rc64_backend_role_registry.{md,json}` | this batch |
| F8 | Refuted rc64 claims sat uncorrected in the memos that made them | corrections lived only in *other* files | append-only supersessions in ma1 and fx2 | this batch |

---

## F12 — the drop site, and why it survived three refreshes

The pointer's CUDA leg carried `archive_bytes: null` while the CPU leg carried a number. rv13
flagged it against the ck2 refresh; to1 and up3 reproduced it.

Root cause, one site: `load_experiments_results_anchors` (frontier_scan.py) built

```python
extra={"evidence_grade": ..., "promotion_eligible": ...}
```

while its sister `load_continual_learning_anchors` — 100 lines above, over the same `Anchor`
type — carried `archive_bytes`, `lane_id`, `measured_at_utc`. The mirror **writes** all three.
Three populated fields were read off disk and discarded one line before the pointer consumed
them. A field-name difference hid it: the mirror emits `archive_size_bytes` (the receipt's own
name), the pointer reads `archive_bytes`. Both spellings are now accepted, so neither producer
had to be renamed.

Measured consequence, RED on HEAD before the fix:

```
tac.candidate_seal.SealContractError: pointer axis 'contest_cuda' is missing archive_sha256
or extra.archive_bytes (sha='7ce46fd7…', bytes=None) — refusing to guess an admission bar
FAILED test_candidate_seal_pin_consistency.py::test_live_pointer_supplies_a_usable_bar
```

Before / after through the **real** refresh (`tools/refresh_canonical_frontier.py --no-update-upstream`):

| field | before | after |
|---|---|---|
| `score` | 0.15652626435208142 | 0.15652626435208142 (unchanged — no score claim here) |
| `extra.archive_bytes` | `null` | **176420** |
| `lane_id` | `null` | **lane_ddm_up3_thirteenth_move_t4_20260819** |
| `extra.runtime_tree_sha256` | absent | **d829ff29…** |
| test | 1 failed, 15 passed | **18 passed** |

`measured_at_utc` is still `null` and I left it so: the receipts carry no such field, and the
honest answer to "when was this measured?" is *not recorded* rather than a plausible
timestamp derived from a file mtime. The v2 writer emits it when a receipt supplies one.

## F4 — the hazard was live, and it is now keyed correctly

Full census of `experiments/results/modal_auth_eval_mirror/` at start: **5 rows, 0 carrying
runtime-tree custody.** Two of them share archive `35c318d5…`:

| score | archive | runtime tree | lane |
|---:|---|---|---|
| 0.15710198138050818 | `35c318d5…` | `da91e06744b94f77…` | `ddm_ck1_composed_r4_t4` |
| **79.40216174747616** | `35c318d5…` | `71c754686eba2ca3…` | `ddm_ck1_composed_rebased_t4` |

Both `passed: true`, both `contest-CUDA`, both n600. The only differing input is the runtime
tree — and rv13's independently-reported `71c75468…` for the 79.40 row is **exactly** what the
backfill recovered from that row's hash-pinned receipt. Two independent routes, same digits.

The cure is three parts:

1. **Writer** (`build_anchor_mirror`) emits schema **v2** with `runtime_tree_sha256` copied
   from the receipt's `expected_runtime_tree_sha256`, plus `inflate_sh_rel`,
   `inflate_device_policy`, `submission_dir_zip_sha256`.
2. **Scanner** stamps every row `runtime_custody = pinned|missing`, carries the tree sha so
   consumers can key on `(archive, runtime_tree)`, and raises a typed `RuntimeCustodyWarning`
   on a custody-less mirror row. **Additive, not invalidating**: legacy rows keep their score
   and their qualifying status. Refusing them would have invalidated the four honest rows that
   already moved the pointer.
3. **Backfill** (`backfill_runtime_custody`) recovers the field from the `source_receipt` the
   mirror already pins by sha256 — **re-hashing the receipt and checking it against that pin
   before copying anything**. This is recovery, not repair-by-assertion; a row whose custody
   cannot be proved keeps its honest `missing` stamp.

All 5 live rows backfilled cleanly. Custody warnings: **5 → 0**.

**Why today's pointer was correct only by accident.** `min()` selection rejects the 79.40 row
because a broken receiver can only *raise* distortion and therefore *raise* S. The exposure
the min-rule cannot catch is the reverse — a receiver that scores *low* for reasons the
archive does not license. That is what the tree sha refuses.

## F7 — the lane-keyed blindness

`dispatched` rows carry `lane_id`; `harvested` rows carried `null`. Measured on the live
ledger, the last six harvests:

```
fc-01M0BGYVF1RBYZKJPMA9KN7K9B |lane= None |score= None |sha= None
fc-01M0BKKHWT2S2ZTET8BKXNPEXW |lane= None |score= None |sha= None
fc-01M0BQ95EWWYZX8BFERE95YSGN |lane= None |score= None |sha= None
fc-01M0BVWYZWS9VY23G5Z24EYG0Q |lane= None |score= None |sha= None   <- ck2's call
fc-01M0C3P0BGWCWBE2F5NB83DC1X |lane= None |score= None |sha= None   <- to1's call
fc-01M0CG6G4WTDFJAS0GDF8C62JK |lane= None |score= None |sha= None   <- up3's call
```

`modal_harvest_poller.py` accepted `--lane-id` and used it **only** for the mirror; the
`update_call_id_outcome(...)` call omitted it. So a lane-keyed reconcile saw zero live calls
for a lane whose call had already harvested `rc=0`, and classified it as a provable phantom.
That is why ck2's terminal row states `stale_superseded_reconciled_no_live_call` for a call
harvested 1 h 52 m earlier, and why to1's claim sat ACTIVE with the job long finished.

Three changes, one site:

* `lane_id=args.lane_id` threaded into the outcome row.
* The **scored facts** go in too (`score`, `score_axis`, `archive_sha256`, `archive_bytes`,
  `evidence_grade`) — rv13's systemic half: the ledger recorded *that* a harvest happened and
  never *what it scored*. `score` is read **only** from
  `score_recomputed_from_components`; there is deliberately no fallback to `final_score`,
  which is rounded to 2dp (0.16 on the live row). A fallback is how a rounded number becomes
  an anchor.
* `_close_terminal_claim(...)` appends the terminal dispatch-claim row at harvest, through
  `tac.deploy.claims.terminal_dispatch_claim` — the tool, per the no-manual-dispatch binding,
  not a hand-edit. It **fails open**: a bookkeeping error writes `CLAIM_CLOSE_FAILED.json` and
  never fails a real paid harvest, because losing a harvested row to a ledger hiccup is
  strictly worse than one manual close.

## F9/F3 — rv13's premise was wrong, and the corrected version is the cure

rv13 F9 says *"Both receipts publish the field directly"* and lists
`report_8dp_score_worst_case_abs_error_bound` et al. **They do not** — not at top level.
`MODAL_REMOTE_RESULT.json` is a summary wrapper; the bounds sit in
`artifacts["contest_auth_eval.json"]`, stored as the **`repr` of a bytes object**, needing
`ast.literal_eval` → `.decode()` → `json.loads` to open. I found this by enumerating the
receipt's keys rather than trusting the memo.

That reframes F9 usefully. The seal writer did not ignore an easy field; the field was three
decodes deep in a wrapper, and retyping the number was the path of least resistance. So the
cure has to **encapsulate the extraction**, not merely say "consume the published field".
`extract_auth_eval_components()` does exactly that.

The math, verified against the live to1 receipt:

| form | value |
|---|---|
| published `report_8dp_pose_score_worst_case_abs_error_bound` | `2.836608391523776e-06` |
| **exact endpoint form** (max over both rounding endpoints) | `2.836608391523776e-06` ← exact match |
| linearized `5/√(10·d_pose)·ε` — what to1's seal used | `2.836151978254699e-06` |

So the derivation uses the **exact endpoint form**, matching the producer in
`experiments/contest_auth_eval.py` digit for digit. The linearization is a first-order
approximation of a concave function and disagrees in the 4th significant figure — it is
`2.836152e-06`, precisely the value rv13 named.

`DeltaBound` is **structurally two-row**: there is no single-row constructor, and it refuses
to render unless its stated addends sum to its stated total. F2 and F3 become
unrepresentable rather than merely discouraged. Reproduced on the real receipts:

```
net dS -6.991519e-05 is 10.48x the SUMMED two-row report-8dp error bound 6.673217e-06
report-8dp bound on the DELTA = 6.673217e-06 (bounds ADD for deltas: base 3.336608e-06
  + candidate 3.336608e-06; the two rows' bounds are equal here)
  base      ck2 row bound 3.336608e-06 (seg 5.000000e-07 + pose 2.836608e-06; published)
  candidate to1 row bound 3.336608e-06 (seg 5.000000e-07 + pose 2.836608e-06; published)
```

Total `6.673217e-06` is the harness-published two-row value (the seal said `6.672304e-06`);
the addends are shown and they sum; and "**equal** here" replaces the seal's false "unequal
per row" — `d_pose` was identical in both rows. The ck2 case reproduces both numbers:
**65.6×** correct, **131.1×** from the one-row division, ratio exactly **2.00**.

There is no API in the module that accepts a caller-supplied bound, and a test asserts that
no function signature contains a `bound` parameter.

## F13 — four bodies, measured independently

Re-derived from the filesystem, not copied from rv13:

    scope:  /Volumes/VertigoDataTier/pact + /Volumes/APDataStore/pact + /Users/adpena/Projects/pact
    method: find -name rc64_backend.c -not -name '._*' -type f, then sha256 every hit
    result: 241 files, 4 distinct contents

| sha256 (16) | bytes | copies | role |
|---|---:|---:|---|
| `05839d1416e68a49` | 5,638 | 237 | shipped receiver, decoder-only |
| `1941923a94e4e0a1` | 14,825 | 2 | checkpoint-extended **encoder**, under the plain name |
| `b249b77bb06a27c8` | 22,179 | 1 | foreign intake (PR138 `opal_v1`) |
| `5c75e2c70b89f148` | 12,222 | 1 | **encoder — THE PIN** |

rv13 measured 239; I measure 241 over the same three trees. The two extra are copies of the
shipped receiver in tonight's runtime trees, which did not exist when rv13 ran. Same four
contents, same roles. Landed as `reverse_engineering/rc64_backend_role_registry.{md,json}`.

## F8 — the corrections now live where the claims live

Append-only supersession sections added to the **end** of both memos; not one character of
the original claims was mutated (Catalog #110/#113).

* `ddm_ma1_model_axis_miss_cost_20260819.md` — supersedes §7's *"158 copies, 2 distinct
  contents … neither matches"* and NEXT_IF_RESUMED item 1.
* `ddm_fx2_model_axis_all_sections_20260818.md` — supersedes the *"no file on either SSD
  matches it"* bullet and the §9 blocker routing at line 461.

Both state the corrected status as **UNBLOCKED-BUT-DOMINATED**, and both state the
denominator **with its scope**, which is the half of `m53` the campaign kept dropping. fx2's
diagnosis was right about *where* (input verification, and refusing to bypass the fail-closed
check was correct) and wrong about *what* — a scoped miss published as a universal absence.

---

## Controls — every gate was made to fire

A detector that never fired is not a detector. Three source mutations, each reverting one
cure, each failing exactly its own control and nothing else:

| mutation | result |
|---|---|
| revert the F12 `archive_bytes` carry | 3 failed (`…carries_archive_bytes…`, `…other_two_spellings`, `…zero_bytes_is_a_measurement`) |
| silence the F4 custody warning | 1 failed (`test_f4_missing_custody_FIRES_the_warning`) |
| revert the F7 `lane_id` threading | 1 failed (`test_f7_poller_threads_lane_id_into_the_outcome_row`) |
| all restored | 22 passed |

Further executed controls: the backfill **refuses** on a receipt-sha mismatch and leaves the
row untouched; `_harvest_outcome_facts` **refuses** to emit a score when only the rounded
`final_score` exists; the bound module reproduces the *wrong* 131.1× and `2.836152e-06`
values on demand, proving it distinguishes them rather than merely producing something
plausible; a stray `True` is not accepted as a published bound (`bool` subclasses `int`).

Tests: **22** (`test_rv14f_anchor_runtime_custody.py`) + **22**
(`test_report_8dp_bounds.py`) = 44 new, all passing. Regression: 106 passed across
`test_frontier_scan*`, `test_canonical_frontier_pointer`, `test_candidate_seal_pin_consistency`.

## Pre-existing reds observed, NOT caused by this batch (verified by stashing onto HEAD)

Reported rather than absorbed:

1. `test_check_330_modal_harvester_call_id_ledger_outcome::test_check_330_live_repo_has_no_unmirrored_modal_harvesters`
   — 2 files (`experiments/modal_ot_offset_n600_gate.py:33` and one sibling) harvest Modal
   calls without recording a ledger outcome. **Same genus as F7** and a natural follow-on.
2. `test_candidate_seal.py::test_the_fire_path_refuses_an_advisory_seal` and
   `::test_the_producer_cli_seals_and_validates_its_own_output` — both hardcode a pointer
   baseline (`0.15771358`) and go red on **every honest pointer move**. Confirmed red on HEAD
   with the pre-rv14f pointer restored. A test that must be edited each time the campaign
   succeeds is a maintenance trap; it should read the live pointer or pin a fixture.

## What I did NOT do

* No Modal fire, no paid dispatch, no score claim, no pointer move. The pointer's **score** is
  byte-identical before and after (`0.15652626435208142`); only its custody fields changed.
* Did not touch the sealed fire order (F1) — already superseded by MAIN's sidecar — or any
  seal. Seals are immutable custody.
* Did not fix F5 / F6 / F10 / F11 (rv13 fixed F2/F5/F6 itself; F10 and F11 are documentation
  conventions outside this batch's six).
* Did not run the sibling sweep rc1x opened (*other recipe/driver pins named generically and
  located off-tree*). Still open, now recorded in the registry.
* `measured_at_utc` on the CUDA leg is still null. The receipts do not carry it; I declined to
  synthesise one.

## Follow-ons, owned

| item | owner | fire condition |
|---|---|---|
| check_330: 2 unmirrored Modal harvesters (F7 genus) | queued | next apparatus pass |
| `test_candidate_seal.py` hardcoded pointer baseline | queued | next pointer move makes it red again |
| rc1x's sibling sweep: generically-named off-tree pins | queued | before the next recipe-pinned byte-close |
| flip `RuntimeCustodyWarning` to a refusal | queued | once every live mirror row is v2-native (all 5 are backfilled today; the *writer* is v2 from now) |
