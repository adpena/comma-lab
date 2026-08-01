# Vacuity cure + census — "an instrument that examined an EMPTY SCOPE emits the same symbol as one that examined a full scope cleanly"

**Arm:** ddm_vc1 · **Date:** 2026-08-01 · **Task:** #842 · **Cost:** $0 (scorer-free, no dispatch)
**Law cured:** memory `vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801` (derived by ddm_rt1)
**Pointer:** UNMOVED. This is apparatus, not a score mover. No score/promotion claim rides on it.

---

## 1. The measurement that opens the case

The law's preflight instance (#842) was recorded as "`--no-codebase` ⇒ 502 of 502 gate call
sites not run ⇒ gate reports PASS". Re-derived here rather than confirmed, and it is **worse
than recorded** — the numerator is literally zero:

| invocation | declared gates | **examined** | wall | symbol emitted |
|---|---:|---:|---:|---|
| `--no-codebase` (scope dev — **the commit-hook default**) | 27 | **0** | **0.52 s** | `PREFLIGHT PASSED` |
| `--no-codebase --scope all` | 448 | **0** | — | `PREFLIGHT PASSED` |
| `--scope dev` (codebase ON) | 27 | 25 | 23.6 s | fails on 4 real violations |
| `--scope all` (codebase ON) | 448 | 13 | — | fails on real drift |

*MEASURED* via `tac.preflight._called_preflight_cli_check_names` (denominator) vs the timing
recorder's rows (numerator). **Both instruments already existed. Nobody had ever compared them.**
The 448-vs-502 difference is definitional: 502 counts call *sites*, 448 counts distinct resolvable
gate *names*.

The consequence stated plainly: **every commit for as long as the hook has had its
`--no-codebase` default has run zero preflight gates and printed green.**

---

## 2. The cure (landing 1 of 2)

**`src/tac/scope_ledger.py`** — canonical typed `ScopeLedger`. A verdict is not a symbol; it is a
symbol **plus a denominator**. Three numbers, because two are not enough:

- `population` — the universe *before* the scope filter. This is the number that makes a filter's
  damage legible: the codex-findings instance is `population=1260, declared=0, examined=0`
  ("you filtered out 1260 things"), not the far more innocent `0 of 0` ("nothing to do").
- `declared` — what the instrument intended to examine.
- `examined` — **the numerator**.

Verdict ladder `COMPLETE` / `PARTIAL` / `VACUOUS`, where `examined == 0` is `VACUOUS`
*unconditionally* — including `declared == 0`. `render()` can never spell an empty scope with a
pass word, and that property is asserted in the tests against the gate's own token list.

**Severity is the caller's decision, deliberately.** Some scopes are legitimately empty (a gate
globbing `experiments/results/**/launch.sh` on a fresh clone). Hard-failing those would cry wolf,
and an apparatus that cries wolf gets overridden — which reproduces the silence it was built to
end. So the ledger always reports; `require_non_vacuous()` chooses.

**Wire-in:** `_preflight_cli_main` now emits `PREFLIGHT VACUOUS — examined 0 of 27 declared —
scopes SKIPPED: codebase — NOTHING WAS EXAMINED` and **exits rc=3** instead of printing
`PREFLIGHT PASSED`.

### The override, and why it is shaped this way

`tools/preflight_hook.py` is the **only** in-repo caller of `--no-codebase` (MEASURED). It now
passes `--acknowledge-empty-scope` and echoes the coverage line on every commit.

Per rt1's binding corollary — *an override that silences a loud instrument reproduces the original
silence with extra steps* — acknowledgement suppresses **the refusal only**. The verdict stays
`VACUOUS`, `render()` still says so, and the hook prints the number where a committer reads it.

Why the hook was not simply flipped to run real gates: `--scope dev` with codebase ON is
**timing-feasible (23.6 s, inside the 30 s DX budget)** but currently **RED** on 4 pre-existing
`check_authoritative_tag_requires_custody_metadata` bypasses. Flipping it would block every commit
fleet-wide. **That is the actionable handoff, not a fix I should make silently** — see §5.

## 3. Self-protection (landing 2 of 2)

**`check_verdict_surfaces_report_examined_count`** in `src/tac/confound_gates.py`, wired
**STRICT** in `preflight_all` at MEASURED live count **0**.

The refused signature is deliberately mechanical, because the bug's fingerprint is structural: a
function that **enumerates a scope** (`glob`/`rglob`/`iterdir`/`scandir`/`walk`, or a
`collect_*`/`*_files`/`scan_*` call) and emits a success verdict as a **bare string constant**. A
bare constant physically cannot carry a count — the denominator is absent *because the literal has
no room for one*. So `print(f"ALL {n} CHECKS PASSED")` passes and `print("PREFLIGHT PASSED")`
refuses, with no judgement call in between. Waiver: `# VACUITY_LEDGER_OK:<rationale>`
(placeholder rationales rejected, Catalog #287 sister discipline).

Second leg: named canonical surfaces must reference `ScopeLedger`, so deleting the CLI wire-in
re-fires the gate rather than silently restoring #842.

**Both controls, as required — a guard that cannot tell them apart reproduces the bug it cures:**

- **POSITIVE** — a planted `rglob` + `print('AUDIT PASSED')` fixture, registered in
  `POSITIVE_CONTROLS` so the sibling `check_refusal_gates_have_live_positive_control` **executes**
  it. It is a live assertion, not a claim. Coverage 4 → 5 gates; uncovered stays 20 (ceiling held,
  not raised).
- **NEGATIVE** — the *same* enumeration, the *same* emptiness, differing only in that the verdict
  carries `{len(bad)} of {len(seen)}`. Gate silent.
- Plus: bare verdict with no enumeration → silent (a relayed subprocess rc is not a scope claim);
  real waiver honoured; placeholder waiver refused.

Two bugs in my own work were caught by these tests rather than by review: `ast.get_source_segment`
excludes trailing comments, so waivers on a function's last line were invisible — *the same
narrowed-detector shape this module exists to guard* — and a crude `"PASS" not in render()`
assertion flagged the render's own negation ("this is not a pass"), a phrase that makes the report
better. The code erred toward clarity; the test was wrong. Per the source memo's second finding, I
checked **which way the disagreement erred** before deciding who was wrong.

## 4. Census — scoped honestly

**These are counts within NAMED scopes, never totals.** Scope swept: `tools/*.py` +
`src/tac/*.py` (2404 files, 1595 scope-enumerating functions) + `src/tac/confound_gates.py`
`_finish` call sites + `.omx/research`. **Not reached:** `experiments/**`, `scripts/**`,
`src/tac/**/` subpackages, non-`print` verdict surfaces (return values, JSON rows, exit codes),
shell/CI surfaces, and every dashboard. A vacuous instrument outside those paths is not
contradicted by anything here.

| # | instance | scope | status |
|---|---|---|---|
| 1 | preflight CLI `--no-codebase` — 0 of 27 examined, printed PASSED | `src/tac/preflight.py` | **CURED** |
| 2 | `review_tracker.cmd_selftest` — bare `print("ALL TESTS PASSED")` over an enumerated entity set | `tools/` | **CURED** (verified: 8 headers, 8 PASS prints, **no skip path**, so declared == examined there) |
| 3 | codex-findings `mtime` window (the law's third instance) | `src/tac/preflight.py:89283` | **NO LONGER VACUOUS, still unreported** — window widened 3 d → 30 d; MEASURED **now declares 1884 of 6919** `.md`, so **5035 files (73 %) are filtered out and that number is reported nowhere**. Not mine to change; the denominator is the handoff. |
| 4 | `_finish(ok_detail=...)` prose carrying **no count** | `confound_gates.py` | **12 found, NOT cured.** Two are outright vacuity-shaped: `"witness_control dir absent"` and `"no significance store on disk"` — absent scope, OK symbol. In the module built to cure confounds. |
| 5 | `local_pre_deploy_check.py:1033` — `ALL {len(CHECKS)} CHECKS PASSED` | `tools/` | **1 found, NOT cured.** Reports the **declared** registry size, not what ran; a skipped check is invisible. Weaker form, real. |
| 6 | recency-window filters | `tools/*.py` + `src/tac/*.py` | 39 `mtime`-touching lines matched; on inspection **most are staleness/caching, not scope filters**. Reported as a lead, NOT as 39 instances. |

Rows 4–6 are a **tracked queue, not a grave** ("off is a tracked queue, never a forgotten
default"). None was cured because none is #842 and each belongs to a live sibling surface; curing
row 4 would put ~12 gates red at once, which trains readers to ignore the suite (the #821 lesson).

## 5. Handoffs (owner ≠ ddm_vc1)

1. **The commit hook still provides zero gate coverage.** Unblocking it needs the 4
   `check_authoritative_tag_requires_custody_metadata` bypasses closed
   (`src/tac/optimization/einstein_kolmogorov_frontier.py:680`,
   `src/tac/witness_control/verified_continuation_certificate_v1.py:364,366`). Then drop the
   acknowledgement and commits get 25 real gates for 23.6 s.
2. **`check_levelset_hosc_requires_beta_end` is RED on main**: measures 10, bound 9 — a sibling
   arm added `experiments/results/_jbasin_smoke/{on,off}/launch.sh` with fixed-β hosc. **Not
   touched here**: raising that bound is precisely the anti-pattern it exists to prevent.
3. **Two pre-existing test failures fixed forward** (both were `KeyError`/count drift from
   `check_upstream_pin_no_content_drift` landing without a bounds entry or a name-set row — i.e.
   the same "red test sat unnoticed on main" failure that produced this law).
4. Rows 4–6 above.

## 6. Verdict scope

`VACUOUS` as a distinct symbol, the ledger, and the STRICT gate are **INSTANCE-** and
**FORMULATION-level** cures: they close #842 at the preflight CLI and refuse one mechanical
signature over one named population. They are **not** a family-level claim that the repo's
verdicts now carry denominators — rows 3–6 are standing evidence they do not. The genus stays
open; what changed is that the largest measured instance is closed and re-introducing its exact
shape now fails a strict gate with an executed positive control.

**MEASURED:** every table number, the live counts, the control firing, the 23.6 s timing.
**DERIVED:** that `test_all_gates_registered` was already failing before this arm (the tuple's
tail on main appends `check_upstream_pin_no_content_drift`, which the test's 23-name set omits) —
subsequently **confirmed MEASURED** by that gate's missing bounds entry.
**ASSUMED:** nothing load-bearing.
