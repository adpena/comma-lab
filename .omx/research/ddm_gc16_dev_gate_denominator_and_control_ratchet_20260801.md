# ddm_gc16 — the commit hook's flip is blocked by SIX red gates, not four bypasses (#852) + positive-control ratchet 5→8 (#831)

**Date:** 2026-08-01 · **Agent:** ddm_gc16 · **Cost:** $0, scorer-free, apparatus only
**Pointer:** UNMOVED. No score claim. This is apparatus.

---

## Answer first

1. **#852 is NOT closable this unit, and the reason is 6× larger than the brief believed.**
   The named blocker — 4 Catalog #127 custody bypasses — is **CURED (MEASURED: 0 remaining)**.
   But `--scope dev` is still red: **6 of 25 declared developer gates are RED**, carrying
   **~316 violations**. The hook default was NOT flipped, because the operator's own stated
   precondition ("once green") is not met, and flipping would block every commit in the repo
   including the ones needed to pay the debt.

2. **The brief's "4" was a first-failure, not a debt total** — and that is a real instrument
   defect, now fixed. `preflight_developer` was fail-fast: it raised the first red gate and
   cancelled the rest, so a reader saw `check_authoritative_tag_requires_custody_metadata: 4
   bypass(es)` and reasonably read it as the whole wall. **This is the failure-side sibling of
   "vacuity is indistinguishable from PASS": both answer with a symbol where the caller needed a
   count.** A red `--scope dev` now names every red gate with `N of M declared`.

3. **#831 ratcheted 5 → 8 of 25** refuse-capable gates carrying a live positive control;
   uncovered queue 20 → 17. Both ratchets tightened (floor 4→8, ceiling 20→17).

---

## Task 1 (#852) — what was measured

### The Catalog #127 adjudication (all four, DIRECTION rule applied first)

Per ddm_rt1's rule: when a gate and the code disagree about an authority/custody quantity, ask
**which way the disagreement errs** before deciding who is wrong. All four err in the SAFE
direction (they can only refuse), so none was cured by weakening a check.

| # | site | direction | disposition |
|---|---|---|---|
| 1 | `src/tac/optimization/einstein_kolmogorov_frontier.py:680` | refusal-only: line 669 already refuses any axis outside `AUTHORITY_AXES` (exactly 2 members, MEASURED), so the tag→semantic-axis map at 680 is TOTAL, and 681-696 routes into a *stricter* validator | **CLASS FIX** |
| 2-3 | `verified_continuation_certificate_v1.py:364,366` | refusal-only consistency leg; the advisory branch affirmatively refuses a contest tag in an advisory axis | **waiver + residual condition recorded at the site — NOT COMMITTED BY THIS LANDING, see below** |
| 4 | `tools/score_coupled_witness_raw_debt.py:201` | one leg of an inline joint chain that already checks tag AND axis AND grade AND `device=="cpu"` AND `platform_system=="Linux"` AND `cuda_available is False` AND `mps_available is False` | **waiver naming the joint chain** |

**Site 1 is a CLASS defect in the gate, not in the code.** `_CUSTODY_VALIDATOR_TOKENS` did not
include `tac.exact_eval_custody.validate_exact_eval_evidence` — the OTHER canonical validator of
the same triple. MEASURED by reading `exact_eval_custody.py:531-670`, it validates axis
(`expected_axis` → `axis_mismatch`), substrate (`require_hardware` → `hardware_not_cuda` /
`hardware_not_contest_cpu`), and BOTH devices (`require_devices`), and additionally binds
`archive_sha256`, `runtime_tree_sha256`, `n_samples == 600`, and the auth-eval command shape —
none of which `validate_custody` checks. Both are return-based, so admitting it opens no
"called but ignored" hole the existing tokens did not already have; this gate is a token-adjacency
heuristic, never a dataflow proof. **14 non-test files call that validator**, so the omission was a
latent trip-wire across the exact-eval custody family.

**Blast radius of the token, MEASURED not assumed:** with waivers held fixed, the token cures
**exactly one** live site. It buys no hidden amnesty.

**CUSTODY NOTE — two of the four sites are not mine to commit.**
`src/tac/witness_control/verified_continuation_certificate_v1.py` is **UNTRACKED** — a sister
agent's in-flight file that does not exist in `HEAD` (`git cat-file -e HEAD:<path>` → fatal).
Staging it would absorb another agent's uncommitted work under this commit body, the Catalog
#314/#340 absorption pattern. So it is deliberately **left unstaged**, and the precise claim is:

* **On `main` after this landing: Check 127 live count = 0.** The cert file is not on main, so
  main only ever had sites 1 and 4, both cured in tracked files.
* **In this working tree: also 0**, because the cert file carries the waivers I wrote into it.
* **When the owning agent lands that file**, it must carry those two waivers (or its own
  adjudication) or it will land 2 fresh Check 127 bypasses. The docstring I added there records
  the residual condition at the site so the next reader does not have to rediscover it.

**Controls landed for the token** (both EXECUTED): a positive control (the validator adjacent →
accepted), a negative control (the validator's NAME in a comment → still flagged), and a registry
test pinning every accepted token to a live callable. **Mutation-proved:** typo the token and both
fail (rc=1, 2 failed); clean, 44 passed.

### The wall the flip actually faces (MEASURED 2026-08-01, this repo state)

Running every dev gate independently instead of fail-fast:

```
DENOMINATOR: 27 declared (25 in the dev_checks tuple + 2 not independently invocable)
  GREEN 19   RED 6   not-independently-invocable 2
```

| red gate | violations |
|---|---|
| `check_codebase_drift` | 23 (13 ad-hoc launchers + 10 bash scripts in `experiments/`) |
| `check_dispatch_claim_helper_present` | banner, count not self-reported |
| `check_subagent_landing_has_solver_wire_in` | 124 landing memos |
| `check_lane_pre_registered_before_work_starts` | 92 unregistered lane_ids |
| `check_substrate_score_aware_losses_use_canonical_scorer_contract` | 56 |
| `check_substrate_trainer_pose_defaults_match_contest_formula` | 21 |

None is cheaply curable and several are **policy** decisions (ad-hoc launcher policy touches live
campaign scripts including `launch_v4d_detached.sh` for the current frontier vehicle), not
apparatus hygiene. Curing any of them by widening exemptions would be a weakening, which is the
exact trap the DIRECTION rule exists to catch.

### Wall-clock, re-measured at the would-be flip

| configuration | wall-clock | n |
|---|---|---|
| current hook default (`--no-codebase --acknowledge-empty-scope`) | **0.44 s**, 0 of 27 examined | 1 |
| `--scope dev`, fail-fast (before this landing) | 23.5 / 24.1 / 25.3 s | 3 |
| `--scope dev`, aggregating (after this landing) | **20.2 / 21.5 s** | 2 |
| all 27 gates serially, no parallel pool | 44.7 s | 1 |

Aggregation is **cheaper**, not more expensive: `submit()` dispatches every check before the first
`run()`, so fail-fast was only cancelling work already in flight. Headroom against the 30 s budget
is ~28-33%, better than the ~21% the brief assumed — but it is headroom on a *red* run, and gate
count only grows.

### What was NOT done, and why

The hook default was **not** flipped. The operator's condition was "once green." It is not green.
A pinning scheme (run all 25, block on the 19 green, hold the 6 red in a loud ratcheted queue) is
the obvious next design and would take commits from 0 → 25 examined — but it changes what a commit
is *allowed to carry*, it lands on the single most load-bearing function in `preflight.py`, and it
was not asked for. Inventing it unasked in the same unit is the "built new machinery instead of
paying the identified debt" reflex. **Recommended as the next unit, with operator assent.**

### A landmine on the flip path (DERIVED from reading; unreachable today)

`preflight_developer` has a clean-cache short-circuit: `_preflight_developer_clean_cache_hit(...)`
→ `check_codebase = False` → **the entire dev_checks block is skipped**. That is a fourth instance
of the vacuity genus sitting directly on the flip path. It is unreachable today only because the
dev scope never passes, so a clean cache is never stored. **The moment someone makes `--scope dev`
green, the second commit gets a cache hit and examines zero codebase gates again.** Whoever lands
the flip must check what the scope ledger reports on a cache-hit run before trusting it.

---

## Task 2 (#831) — positive-control ratchet, with denominator

**Live measurement, not the brief's snapshot: coverage was 5 of 25 (20 uncovered), not 4 of 23.**
The queue moved under the brief (vc1 added one control; two gates were added).

**Now: 8 of 25 covered, 17 uncovered.** Three controls landed, chosen as the most load-bearing
entries in the queue — the original 2026-07-05 confound-hunt immune system:

| gate | catalog | why it was the priority |
|---|---|---|
| `check_no_spike_guard_defaults_to_deadlock_mode` | #397 | the ANCHOR confound: `default='legacy'` froze BOTH the v5 and v6 n600 runs at ep114/ep103 while telemetry kept advancing, poisoning a whole session's verdicts |
| `check_reject_filter_updates_reference_from_accepted_only_has_rearm` | #398 | the GENERALIZED structural form of the same bug |
| `check_verdict_pairs_default_is_n600` | #401 | a non-zero default runs best-checkpoint selection and all d_seg telemetry on a subset — a toy that looks like a measurement |

All three were **STRICT-flipped at live-count 0**, which is precisely the state where nothing else
proves the detector still fires: a live count of zero looks identical whether the gate works or has
been gutted. That is the vacuity genus one layer down, on the gates themselves.

**Both legs, per the standard.** Each control FIRES on a planted instance (verified by the meta-gate
executing it) and each has a paired negative control that stays SILENT on the gate's own named cure
(`rollback` default / `--verdict-pairs 0` / a `.clear()` re-arm beside the accepted-only append).
Each negative control carries a **non-vacuity assertion**: the same fixture path with the defect
restored must fire, so a clean `== []` can never come from an unscanned path. Positive proves
sensitivity; negative proves specificity; non-vacuity proves the scan surface. A control missing
any leg is not proof.

**One fixture was wrong on the first attempt** and the meta-gate caught it: the #398 fixture put the
append outside the spike-guarded `if`, so it was not the pattern the detector looks for. That is the
whole argument for executing controls rather than declaring them — a fixture you *believe* is a
violation frequently is not.

**Ratchets tightened:** `MIN_POSITIVE_CONTROL_COVERAGE` 4 → 8, `MAX_UNCOVERED_REFUSE_GATES` 20 → 17,
plus a test asserting the two constants still describe the measured coverage (a ratchet that drifts
from its own measurement is a silent instrument).

**17 still uncovered** — named in the gate's own `uncovered` output every run, so the queue is
surfaced rather than remembered.

---

## Pre-existing red, found while measuring — NOT caused by this landing

Verified by test-reverting `src/tac/preflight.py` to HEAD and re-running (byte-identical failure
signatures at the same assertion lines):

- `test_preflight_cli_timeout.py::test_developer_preflight_budget_timeout_reports_hot_step` —
  hot-step list no longer names `check_codebase_drift` (HEAD:83, mine:83, same message).
- `test_preflight_cli_timeout.py::test_full_preflight_library_budget_timeout_reports_hot_step` —
  no per-check timings recorded (HEAD:111).
- `test_confound_gates.py::...[check_levelset_hosc_requires_beta_end]` — live count grew 9 → 10 via
  `experiments/results/_jbasin_smoke/{off,on}/launch.sh`. The ratchet is doing its job; the debt is
  an artifact-drift, not mine.

---

## Round-2 review of this landing's OWN fix — one CRITICAL caught

The first version of failure aggregation collected `BaseException`. That swallowed
`PreflightTimeoutError`, which is a **wall-clock budget breach, not a gate violation**. Two harms:
the CLI lost the exception type it needs to report the hot step, and — worse — **a run that had
already blown its deadline would keep executing the remaining gates**, inverting the budget's
purpose. `KeyboardInterrupt` / `SystemExit` / `MemoryError` were swallowed the same way.

Cure: `_is_non_collectable_control_exception` — **aggregate what the DEVELOPER must fix; propagate
what the RUN must obey.** Pinned by tests including the subtle case that
`PreflightTimeoutError` *subclasses* `PreflightError`, so a naive base-type check would have
re-introduced it.

Behaviour contract, so aggregation is never amnesty: zero failures → proceed; ONE failure →
re-raise the **original exception object**, type intact (every existing caller and error-message
assertion keeps working); MANY → one `PreflightError` naming all of them with `N of M declared`.
**Nothing that raised before stops raising.**

---

## Standing caution — held, not spent

Neither `PREFLIGHT_SKIP_CI_BLIND_TESTS=1` nor `--acknowledge-empty-scope` was used to make anything
pass. `--acknowledge-empty-scope` remains on exactly one caller (the hook's fast path) for exactly
its designed reason, and this landing did not extend it. An override that silences a loud instrument
reproduces the original silence with extra steps.

## Labels

MEASURED: every count, timing, coverage fraction, the 6-red census, the token blast radius, the
mutation results, the pre-existing-red HEAD comparison. DERIVED: the clean-cache landmine (read from
control flow, not executed — it is unreachable in the current red state). INFERRED: that the brief's
"4 bypasses" came from reading the fail-fast first failure. ASSUMED: nothing load-bearing.

**Verdict scope:** the "flip is blocked" verdict is INSTANCE-level (this repo state, 2026-08-01) —
it says nothing about whether the flip is achievable, only that it is 6 gates and ~316 violations
away rather than 4 waivers away.
