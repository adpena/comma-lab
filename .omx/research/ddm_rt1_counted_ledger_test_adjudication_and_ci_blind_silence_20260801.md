# ddm_rt1 (#845) — counted-ledger test adjudication + the CI-blind silence that hid it

**Date:** 2026-08-01 · **Arm:** ddm_rt1 (scorer-free, $0, local) · **Pointer:** UNMOVED
(this is apparatus, not a score mover; `score_claim=false`)
**verdict_scope:** INSTANCE for the test drift; CLASS for the silence (see §4)

---

## 1. The question

`src/tac/tests/test_ddm_tb1_tr1_renderer.py::test_counted_ledger_keys_and_selector_counted_both_variants`
failed at HEAD, line 155. Counted-ledger keys ARE the rule-118 counted-vs-free boundary, so
the adjudication is a compliance question, not a style one: is the TEST stale, or is the CODE
mis-accounting counted bytes?

## 2. Verdict: the TEST was stale. The CODE is correct.

**MEASURED** (`pytest` at HEAD): the ledger emits four keys the frozen assertion did not know about —
`rowband_spec_bytes`, `tokens_bytes_zlib`, `tokens_bytes_smevr`, `token_ledger_coder`.

**MEASURED** (`experiments/train_tr1_partition_renderer_mlx.py:828-881`): the ledger has **four
COUNTED streams** and **three OBSERVABILITY keys**:

| key | class | in `total_counted_bytes`? |
|---|---|---|
| `tokens_bytes` | COUNTED | yes |
| `renderer_bytes` | COUNTED | yes |
| `selector_ledger_bytes` | COUNTED | yes |
| `rowband_spec_bytes` | COUNTED (QA84 §4.2 decoder side-info) | yes |
| `tokens_bytes_zlib` / `tokens_bytes_smevr` / `token_ledger_coder` | OBSERVABILITY | **no** (merged by `ledger.update(obs)` AFTER the total) |

**DERIVED — why "the code is wrong" is the wrong answer, and the dangerous one.** The stale test
asserts `total == tokens + renderer + selector`, i.e. it demands `rowband_spec_bytes` be EXCLUDED
from the price. "Fixing" the code to satisfy the test would have removed a real, decoder-visible,
video-derived side-info stream from the counted total — a rate **UNDER-count**, which is the
compliance defect. The code errs in the conservative direction (it prices MORE), which is correct.

**MEASURED — the code path is already covered elsewhere.** `src/tac/tests/test_ddm_b2b_burn2_composition.py`
(23 passed) contains `test_counted_ledger_total_excludes_observability_keys` and
`test_rowband_ledger_counts_spec_bytes_in_total`, which assert exactly the post-change contract
including the **nonzero** rowband term. The b2b arm wrote new tests in its own file and left the
sibling tb1 test — which also pinned the key set — untouched.

**MEASURED — no downstream rate claim is corrupted.** Every consumer of the ledger on `main`
reads `["total_counted_bytes"]` directly; none reconstructs a sum (which is what could have
double-counted the observability keys) and none drops the rowband term:

- `experiments/train_tr1_partition_renderer_mlx.py:2185` (telemetry via `gate_row.update`), `:2510`
- `experiments/ddm_pa1r_endpoint_verdict.py:89` — additive S; also reports `rowband_spec_bytes` explicitly
- `experiments/ddm_b4r_endpoint_extras.py:150`
- `experiments/ddm_ep2_compile_candidate_archives.py` — uses the total only as the gap reference
  against the REAL `archive.zip` stat

(Exhaustive: repo-wide grep for `counted_bytes_ledger`, excluding `.git`/`.venv`; the only other
hits are inside `.claude/worktrees/**` copies, which are not `main`.)

## 3. Timeline of the drift (MEASURED, `git log -S`)

| when | commit | what |
|---|---|---|
| 2026-07-28 14:49 | `db105b6f51` | tb1 test written; froze the key set at 4 |
| 2026-07-30 11:10 | `f28e427dd9` | QA86/QA83 adds the 3 OBSERVABILITY keys — test not updated |
| 2026-07-30 11:20 | `e8d531e735` | QA84 adds `rowband_spec_bytes` as a 4th COUNTED stream — test not updated |

The test went red on 2026-07-30 ~11:10 and stayed red for ~2 days.

## 4. Why nobody saw it — and the root cause it shares with #842

**MEASURED.** There are exactly **two** automated surfaces, and neither runs this test:

1. **GitHub Actions** (`.github/workflows/ci.yml`) DOES run `pytest src/tac/tests/` on push to
   `main` — but on `ubuntu-24.04`, installing `.[dev,runtime]`. `mlx` is a **separate** optional
   extra (`pyproject.toml:126`, `mlx = ["mlx>=0.5"]`) with **no Linux wheel**. Every module guarded
   by `pytest.importorskip("mlx...")` is therefore SKIPPED there — and **pytest reports a skip as
   green**. **57 test modules** under `src/tac/tests/` are MLX-gated (AST-confirmed, not substring).
2. **The local pre-commit hook** (`tools/preflight_hook.py`) runs ruff F821 + `tac.preflight
   --no-codebase` + the review gate. It ran **no tests at all**.

So the only machine that CAN execute those 57 modules (this Mac) had no automated moment that did.

**DERIVED — the shared root cause with #842.** Task #842 reports 502 of 502 `preflight_all` gate
call sites SKIPPED on a normal commit; the week-coherence audit
(`.omx/research/ddm_cn3_week_coherence_audit_20260731.md:135`) records
`check_codex_findings_memos_consumed` as "LIVE COUNT 0 … scans only `mtime < 3 days` ⇒ **0 of 1,260
files currently in scope**. The gate is structurally vacuous." All three are the same class:

> **An instrument that evaluated an EMPTY SCOPE reports the same symbol as one that evaluated a
> full scope and found nothing wrong. Vacuity is indistinguishable from PASS.**

`skip == green`; `--no-codebase == 502 gates not run`; `mtime<3d == 0 of 1260 files`. The fix
direction is the same in all three: **an instrument must report what it actually evaluated**, and
an empty evaluation must be visibly different from a clean one. This landing does that for one of
the three (§5, item 2 prints the module count it ran, and treats pytest's "no tests collected"
exit code 5 as an explicit *nothing verified* notice rather than a silent green). #842 and the
`mtime` gate are NOT fixed here — reported, not repaired, per scope.

## 5. The two landings

1. **FIX** — `src/tac/tests/test_ddm_tb1_tr1_renderer.py`: the frozen key-set assertion is replaced
   by an explicit rule-118 classification. `COUNTED_LEDGER_KEYS` / `OBSERVABILITY_LEDGER_KEYS` are
   declared; the total must equal the sum of **exactly** the counted keys; and any key belonging to
   neither set FAILS with a message telling the author that leaving a byte-bearing key out of the
   total under-prices the rate term. A future additive observability key no longer goes red for the
   wrong reason; a future *counted* key cannot be added silently. Also corrected the now-false
   docstring at `train_tr1_partition_renderer_mlx.py:835` ("sums ONLY the three real streams" — it
   sums four).

2. **GUARD** — `tools/preflight_hook.py` gains step 3, `run_ci_blind_tests()`: on commit it selects
   the MLX-gated (CI-blind) modules reachable from the staged files (whole-word importable-name
   match, plus any staged blind module itself) and runs them, **blocking** on red and on timeout.
   It deliberately does NOT duplicate CI: modules CI can already run are excluded
   (`test_ddm_b2b_burn2_composition.py` is correctly not selected).
   - **MEASURED, historical proof:** staging `experiments/train_tr1_partition_renderer_mlx.py`
     selects 5 modules including `test_ddm_tb1_tr1_renderer.py`; running HEAD's (red) copy of that
     test through the step returns **rc=1 → commit BLOCKED**. The guard would have caught
     `e8d531e735` at 11:20 on 2026-07-30.
   - **MEASURED cost:** those 5 modules = 85 tests in **2.82 s**. Commits touching nothing
     MLX-adjacent select 0 modules and pay only a 57-file scan.
   - 18 new tests in `src/tac/tests/test_preflight_hook.py` (the existing surface, not a new file),
     including that a timeout BLOCKS rather than soft-passes, and that the step is wired into
     `main()` rather than defined-and-orphaned.

## 6. Honest limitations

- The classification guard **mitigates but cannot prevent** the failure mode where someone silences
  a future failure by filing a byte-bearing key under `OBSERVABILITY_LEDGER_KEYS`. It makes that an
  explicit, reviewable edit with a warning at the point of decision; it is not a proof.
- The reverse-dependency selection is textual (whole-word importable names). It over-selects rather
  than under-selects by design — over-selection costs runtime, under-selection costs silence — but
  a test that reaches a module through a fully dynamic name would be missed.
- The step covers the MLX-gated set only. Other CI-invisible classes (GPU-only, network-only, and
  `-m slow`-marked tests) remain unaddressed and are NOT claimed to be covered here.
- `.claude/worktrees/**` copies of the trainer/test are not on `main` and were not touched.

## 7. What the blindness had accumulated — and the cost of turning it loud

**MEASURED (2026-08-01, after the §5 fix landed as `57d4747e60`).** Each of the 57 CI-blind modules
was run in its own process (`-m "not slow"`, 90 s budget each). **8 are not green:**

| module | rc | reading |
|---|---|---|
| `test_run_pr95_mlx_timing_smoke.py` | 1 | failing |
| `test_train_substrate_hi_nerv_mlx_local.py` | 1 | failing |
| `test_loss_term_telemetry.py` | 1 | failing |
| `test_levelset_micro_batch_loss.py` | 1 | failing |
| `test_mlx_scorer_adapters.py` | 1 | failing |
| `test_instant_projected_adjoint.py` | 1 | failing |
| `test_micro_batch_bit_identity_probe.py` | 124 | exceeded **my 90 s sweep budget** — may be merely slow, NOT established as red |
| `test_compact_renderer_mlx_spine_runner.py` | 124 | same caveat |

Separately, running all 57 in ONE process **SEGFAULTS** at ~55% inside
`src/tac/substrates/_shared/mlx_score_aware/adapter.py:5486` (`_score_aware_loss_part_metrics`,
reached via `run_mlx_score_aware_full_main`). Not attributed to a module here.

I did NOT triage or fix these — out of scope, and two of the six failures are on the BANNED
PR95/HNeRV lineage (lessons-only per the no-old-lineage rule), so their priority is not obvious.
They are reported, not repaired.

**The operational consequence, stated plainly rather than discovered by whoever hits it first:**
step 3 BLOCKS, so a commit whose staged files reach one of those 6 modules will now be refused for
a red the committer did not cause. That is the intended trade — silence became noise — but it is a
real cost. Options for the operator, in preference order: (1) triage the 6 (they are now visible,
which is the whole point); (2) mark genuinely-obsolete banned-lineage modules `-m slow` or delete
them; (3) `PREFLIGHT_SKIP_CI_BLIND_TESTS=1` for a single commit — loud, documented, and NOT a
default. What must not happen is that override becoming habitual: that reproduces the original
silence with extra steps.

---

## 8. Closure of §7 — and the genus at a THIRD surface (MAIN, 2026-08-01, later same day)

§7 said "reported, not repaired." Repaired, and the repair was **already done** when MAIN picked it up.

**MEASURED at HEAD, each module in its own process, `-m "not slow"`:**

| module | result |
|---|---|
| `test_run_pr95_mlx_timing_smoke.py` | 1 xfailed, 4.40s |
| `test_train_substrate_hi_nerv_mlx_local.py` | 69 passed, 5 xfailed, 1.34s |
| `test_loss_term_telemetry.py` | 17 passed, 0.42s |
| `test_levelset_micro_batch_loss.py` | 111 passed, 1.81s |
| `test_mlx_scorer_adapters.py` | 43 passed, 14.31s |
| `test_instant_projected_adjoint.py` | 8 passed, 1.16s |
| `test_ddm_tb1_tr1_renderer.py` (task #845) | 18 passed, 0.45s |

**All green.** The CI-blind hook step will not refuse an innocent commit on these. `b02b99cecb` (11:52,
`ddm_tr6`) did the triage — "2 REAL code defects, 3 stale tests, 1 pinned open finding; both
over-budget modules are SLOW not red" — and `aa0a27a2af` (12:59, `ddm_mk1`) the follow-on. Tasks #851
and #845 stayed open in the ledger regardless.

### The third surface

The genus of §5 (`vacuity_is_indistinguishable_from_pass`) and of the staleness class recurs here in a
form neither names:

> **A verdict TABLE is a consumption surface. §7's RED list was true when measured and false when read
> — and because it errs PESSIMISTIC, nothing flags it.** An over-optimistic stale claim gets caught the
> moment someone depends on it. An over-pessimistic one just sits there generating phantom work, and
> the work looks like diligence.

**MEASURED sweep, 400 commits vs the 54 open (pending + in_progress) task rows: 13 open rows carry a
commit that NAMES them** — #366 #729 #809 #815 #820 #821 #822 #824 #831 #843 #845 #846 #851. A naming
commit is **not** proof of closure (#366 is merely referenced by a consolidation harvest), so the
cross-check is a PROMPT to re-derive, never an auto-close. Both rows closed above were closed on
MAIN's own re-run, not on the commit message's word.

**Cheap apparatus fix, not yet built:** the same `git log | grep "#<id>"` cross-check the sweep above
performed by hand, surfaced next to any open row before work is dispatched against it. Cost is one
`git log` per session. It would have saved this triage entirely.

**Routed:** `#843` has two claiming commits (`743e70fb06` typed fail-closed `token_codec` selector at
`ddm_tr1_runtime.py:319-320,342-343,462`; `fe1175c4e5` "the SMEVR cut is ALREADY SPENT"). The live
`rh1` arm owns #843/#853 and was messaged with both, plus the two things MAIN did **not** verify: that
the EXPORTER emits what the receiver demands, and whether "already spent" means the −0.0499 S rate half
is double-counted against the v4d 360,238 B baseline — which would flip that headline's sign.
