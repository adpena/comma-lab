# tac CI fix by authoring missing tests (Option B)

**Subagent:** `tac_ci_fix_authoring_tests_20260519T192727Z`
**Lane:** `lane_tac_ci_fix_authoring_tests_20260519`
**Landed UTC:** 2026-05-19T19:36:00Z
**Operator quote (verbatim, 2026-05-19):**
> *"iterate on cleaning and OSS of comma-lab and iterating on thePR and adding the tests and fixing all faliures and everything"*

The "adding the tests and fixing all failures" framing OVERRODE Slot L's recommended Option A (remove non-existent test references). This subagent executed Option B — AUTHOR the missing tests + fix all CI failures.

---

## Authority chain

1. **Slot H audit** (commit `66a0a6aad`): tac PASS_WITH_MINOR_GAPS; 2 non-blocking gaps surfaced.
2. **Slot L audit** (commit `054ba63cb`): empirically confirmed tac CI Gap 1 — `.github/workflows/test.yml` references stale test paths; CI run `25406255121` FAILS; 4 of last 4 runs RED for 14 days.
3. **Operator's 2026-05-19 verbatim directive** above — Option B AUTHOR tests.

---

## Sister-scope acknowledgment (Catalog #230)

In-flight sister subagents at dispatch time:
- **Slot M** (`a5313e169802ee5b3`): PR 95 deep research — READ-ONLY scope. Disjoint.
- **Slot N** (NEW, dispatched in parallel): comma-lab sanitization sweep. Scope: `adpena/comma-lab` repo. Disjoint.

This subagent's scope:
- LOCAL clone of `adpena/tac` at `/tmp/tac_ci_fix_20260519T192727Z/` (zero overlap with comma-lab working tree)
- NEW files in `adpena/tac` only
- Draft PR to `adpena/tac` (NOT MERGED — operator approves)
- This audit memo + landing memo in comma-lab (NEW files only; no working-tree mutation of sister territory)

---

## Phase 1: Premise verification (Catalog #229 PV)

Cloned `adpena/tac` (commit head: `2c84c95c0d`) to `/tmp/tac_ci_fix_20260519T192727Z/tac`. Verified:

| Premise | Status |
|---|---|
| `tac/tests/test_meta_lagrangian.py` exists | ✓ EXISTS (workflow refs this; runs clean) |
| `tac/tests/test_predictor_score_band.py` exists | ✗ **MISSING** (workflow refs this — broken) |
| `tac/tests/test_distortion_proxy_local.py` exists | ✗ **MISSING** (workflow refs this — broken) |
| `tac/tests/test_score_band_predictor.py` exists | ✓ EXISTS (canonical-name'd cousin of missing file 1) |
| `tac.predictor.score_band` module exists | ✓ EXISTS at `tac/predictor/score_band.py` |
| `tac.predictor.distortion_proxy_local` module exists | ✗ **MISSING** (referenced as `experiments.distortion_proxy_local` in `tac.optimizer.meta_lagrangian` docstring; not present in OSS tac) |
| Baseline test suite passes | ✓ 24 pass / 2 skip (existing files) |

The premise verification confirmed exactly what Slot L claimed empirically. Slot L's recommendation was Option A (remove workflow references). Operator's 2026-05-19 directive overrode that with Option B (author the tests).

---

## Phase 2: Test + reference module authoring

### File 1: `tac/predictor/distortion_proxy_local.py` (NEW, 268 LOC)

Canonical closed-form distortion proxy that the score-band predictor calls when `rel_err > 1%` and no full empirical proxy is provided. Architecture:

- **Per-axis power-law fitting**: separate `D_pose = floor_pose + a_pose * rel_err^b_pose` and `D_seg = floor_seg + a_seg * rel_err^b_seg` curves fit via closed-form log-linear regression on lossy anchors (mirrors the math in `tac.predictor.score_band.fit_distortion_curve` but splits per-axis rather than applying a global pose/seg ratio).
- **Floor source**: the LOSSLESS anchor with TIGHTEST distortion (deterministic disambiguation when multiple lossless anchors exist).
- **Refusal taxonomy**: `INSUFFICIENT_ANCHORS` (matches predictor refusal #1) / `NO_LOSSLESS_ANCHOR` (cannot establish floor) / `NO_LOSSY_ANCHORS` (cannot fit curve, need ≥2 lossy) / `RUNAWAY_EXTRAPOLATION` (>5× calibrated max).
- **Public API**: `ProxyFit` frozen dataclass + `fit_proxy(anchors)` + `predict_distortion(fit, rel_err)` + `make_distortion_proxy(anchors)` (returns `DistortionProxy` callable) + `make_distortion_proxy_from_file(path)`.

The full empirical proxy lives in `experiments.distortion_proxy_local` in the broader Pact research repo (referenced in `tac.optimizer.meta_lagrangian` docstring); this OSS-tac module is the closed-form shippable fallback.

### File 2: `tac/tests/test_distortion_proxy_local.py` (NEW, 22 tests, 433 LOC)

Categories of coverage:

| Category | Tests |
|---|---|
| Per-axis fit (happy path + 4 refusal modes) | 5 |
| Predict (rel_err=0 / calibrated anchor / runaway extrapolation / negative / refused fit / floor clamp) | 6 |
| Determinism (fit + predict byte-for-byte across calls) | 2 |
| Callable API (3-arg signature / unused-arg invariance / O(1) per-call / refused fit / file roundtrip / missing file) | 6 |
| Integration with `predict_score_band` (proxy unblocks high-rel_err / proxy missing triggers refusal #3) | 2 |
| Type/Protocol (DistortionProxy alias accepts callable / ProxyFit frozen / as_str format) | 3 |

### File 3: `tac/tests/test_predictor_score_band.py` (NEW, 24 tests, 378 LOC)

Module-level public-API contract complementing the existing `test_score_band_predictor.py` (which is comprehensive on apogee_int4 failure scenarios). Categories:

| Category | Tests |
|---|---|
| Module imports + canonical symbols | 1 |
| Contest-defined constants match upstream/evaluate.py | 2 |
| `_score_from_components` matches PR106 + apogee_int4 + edge cases (zero pose / negative pose) | 4 |
| `CalibrationAnchor` (frozen / rate consistency / notes default) | 3 |
| `ScoreBand` (frozen / as_str accepted+refused / default values safe) | 4 |
| Edge cases (zero anchors / fit_distortion_curve zero / load missing file / load non-list JSON) | 4 |
| `DistortionProxy` alias acceptance + consumption | 2 |
| Determinism (predict_score_band + fit_distortion_curve byte-for-byte) | 2 |

### File 4: `tac/predictor/__init__.py` (MODIFIED, +16 lines)

Re-exports the new `distortion_proxy_local` public symbols so `from tac.predictor import make_distortion_proxy` works (mirrors the existing `from tac.predictor import predict_score_band` pattern).

---

## Phase 3: Local pytest verification (Python 3.12, macOS Darwin)

Workflow's exact pytest command:

```
$ pytest tac/tests/test_meta_lagrangian.py \
         tac/tests/test_predictor_score_band.py \
         tac/tests/test_distortion_proxy_local.py -v --timeout=60
======================== 56 passed, 1 skipped in 0.15s =========================
```

Workflow's formula-verification step:

```
$ python -c "from tac.optimizer import contest_score; ..."
All formula constants + reproductions verified.
```

Existing `test_score_band_predictor.py` (comprehensive) unaffected:

```
$ pytest tac/tests/test_score_band_predictor.py
======================== 14 passed, 1 skipped in 0.08s =========================
```

**Total: 70/70 tests pass on the in-scope test surface. 2 skips are pre-existing (require external fixtures not in OSS tac).**

---

## Phase 4: Branch + draft PR

- Branch: `fix/ci-test-paths-and-add-missing-tests` (head: `573d56a7a1eafb33fb00178ae6a474a4e09bc9a4`)
- Pushed to: `git@github.com:adpena/tac.git`
- Draft PR: **https://github.com/adpena/tac/pull/1**
- Title: `ci: fix stale test paths + add missing tests`
- Status: DRAFT (per Catalog operator-care discipline — `gh pr merge` NOT invoked)

---

## Phase 5: CI verification on the PR branch

Run ID: `26120635237`

| Job | Status | Duration |
|---|---|---|
| test (3.11) | ✓ GREEN | 28s |
| test (3.12) | ✓ GREEN | 24s |

Both jobs ran every workflow step clean:
- ✓ Install package + test deps (uv venv + uv pip install)
- ✓ Run meta-Lagrangian + predictor + distortion-proxy tests
- ✓ Verify contest_score formula matches upstream constants

The 14-day RED streak is broken on this branch. CI will go green on main once the PR is merged.

(Annotation noise: Node.js 20 deprecation warning for `actions/checkout@v4` and `actions/setup-python@v5`. NOT a test failure; future cleanup separate from this PR.)

---

## Operator-routable: merge approval pending

Per CLAUDE.md "Executing actions with care" non-negotiable + this subagent's prompt forbidding `gh pr merge`, the draft PR awaits operator approval. Recommended approval command:

```
gh pr ready 1 --repo adpena/tac && gh pr merge 1 --repo adpena/tac --squash
```

Optional pre-merge polish (NOT blocking, can be a follow-up PR):
- Bump `actions/checkout@v4` → `@v5` and `actions/setup-python@v5` → `@v6` to silence Node 20 deprecation annotations.
- Consider tagging a new patch version (`tac` is currently at 1.0.5; the new `tac.predictor.distortion_proxy_local` module is a minor-API addition).

---

## Discipline acknowledgment

- **Catalog #229 PV**: empirically verified the workflow's exact test commands + module imports BEFORE authoring code.
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: NEW files only; no in-place mutation of existing test files or `tac.predictor.score_band` module.
- **Catalog #230 sister-subagent ownership map**: scope confined to `adpena/tac` repo + 2 NEW files in comma-lab (this audit memo + landing memo). Zero overlap with Slot M (read-only PR 95 research) / Slot N (comma-lab sanitization).
- **CLAUDE.md "Executing actions with care"**: Draft PR only; merge requires explicit operator approval.
- **CLAUDE.md "Always use uv"**: All installs via `uv pip install`; Python 3.12 venv (3.14t free-threaded incompatible with `constriction` wheel matrix).
- **CLAUDE.md "Bugs must be permanently fixed AND self-protected against"**: every authored test has clear assertions + descriptive name + canonical fail-closed where appropriate (e.g. refusal-mode tests assert NaN return; frozen-dataclass tests assert mutation raises).

## 6-hook wire-in (Catalog #125)

- Hook #1 sensitivity-map: **N/A** (CI hygiene fix; no score-signal contribution)
- Hook #2 Pareto constraint: **N/A**
- Hook #3 bit-allocator: **N/A**
- Hook #4 cathedral autopilot dispatch: **N/A**
- Hook #5 continual-learning posterior: **N/A**
- Hook #6 probe-disambiguator: **N/A**

This landing is pure CI hygiene + reference-module addition; no solver wire-in is applicable.

---

## Summary

| Surface | Result |
|---|---|
| Premise verification | ✓ Empirically confirmed Slot L's findings |
| Reference module authored | ✓ `tac/predictor/distortion_proxy_local.py` (268 LOC) |
| Missing test file 1 authored | ✓ `tac/tests/test_distortion_proxy_local.py` (22 tests / 433 LOC) |
| Missing test file 2 authored | ✓ `tac/tests/test_predictor_score_band.py` (24 tests / 378 LOC) |
| `tac/predictor/__init__.py` re-exports | ✓ +16 lines, all new symbols |
| Local pytest verdict | ✓ 56/56 pass on workflow's exact command (+ 14 existing pass) |
| Workflow formula step | ✓ Pass |
| Branch pushed | ✓ `fix/ci-test-paths-and-add-missing-tests` → `origin/adpena/tac` |
| Draft PR opened | ✓ https://github.com/adpena/tac/pull/1 |
| **CI verification on PR branch** | ✓ **GREEN on Python 3.11 + 3.12** (14-day RED streak broken) |
| PR merged | ⏸ DEFERRED-to-operator (per "Executing actions with care") |

$0 GPU + ~85 min wall-clock. Operator approves merge with `gh pr ready 1 --repo adpena/tac && gh pr merge 1 --repo adpena/tac --squash` when ready.
