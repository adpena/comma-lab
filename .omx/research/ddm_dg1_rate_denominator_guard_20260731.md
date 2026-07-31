# ddm_dg1 — rate-denominator cleanliness guard (task #812)

**Date:** 2026-07-31 · **Actor:** ddm_dg1 · **Charter:** us1 NEW-SIGNAL rank-2
(`.omx/research/ddm_us1_upstream_reread_20260731.md`, denominator row) ·
**Cost:** $0 · scorer-free (never ran evaluate.py; burn-4 owns that slot) ·
`upstream/` strictly read-only.

**Pointer honesty:** `0.1910828242 [contest-CPU]` UNMOVED. This is APPARATUS
(a correctness self-protect), NOT a score mover — no rate/score claim.

---

## 1. Verification receipt (never trusted the charter — re-derived from source)

`upstream/evaluate.py` (read directly, lines 55-65):

- **line 14** `--uncompressed-dir` default = `Path('./videos/')` (our repo:
  `upstream/videos/`).
- **line 56** `test_video_names = [line.strip() for line in file.readlines()]`
  — the SCORED set comes from `public_test_video_names.txt`.
- **line 64 (the denominator):**
  ```python
  uncompressed_size = sum(file.stat().st_size
                          for file in args.uncompressed_dir.rglob('*')
                          if file.is_file())
  ```
- **line 65** `rate = compressed_size / uncompressed_size`.

**The asymmetry that IS the vulnerability:** line 56 reads the names file to pick
the scored videos, but line 64 rglobs the ENTIRE dir for the denominator. A stray
that is NOT in the names list still inflates the denominator.

**Measured on the live tree (2026-07-31):**
- dynamic sum `= 37,545,489` == constant `37_545_489` → **MATCH (clean)**.
- inventory: exactly `{0.mkv}` (the only file; `public_test_video_names.txt` = `0.mkv`).

**Empirically confirmed the dotfile hazard is real** (tmp fixture, real upstream
never touched): `Path('videos').rglob('*')` with `is_file()` COUNTS `._0.mkv`
(AppleDouble) and `.DS_Store` — a fixture sum went 100 → 180 with two strays.
`pathlib.rglob('*')` matches dotfiles (unlike shell glob), so the premise holds.
Historical precedent for the mitigation: old bootstrap scripts carried
`find upstream -name '._*' -delete`.

## 2. Hardcode-count survey (chokepoints guarded, NOT mass-migrated)

`grep -rEn "37[_]?545[_]?489" src/tac tools experiments --include="*.py"` →
**203 hits**. Per the charter, these are NOT all migrated — only the canonical
consumption chokepoint is guarded. Canonical homes observed:
- `src/tac/contest_score.py::UNCOMPRESSED_SIZE_BYTES` (the canonical score SoT, #168).
- `src/tac/archive_byte_profile.py::CONTEST_ORIGINAL_BYTES` (rate-term sister).
- `src/tac/joint_scorer_aware_training.py::LAMBDA_RATE_CONSTANT` (training-side).
- `src/tac/exact_eval_custody.py::CONTEST_REFERENCE_BYTES` (custody validator).

`compute_contest_score` (the #168 canonical mirror) has **83 import sites**; all
route their rate arithmetic through `rate_term`, so guarding `rate_term` covers
the whole canonical chain in one point.

## 3. What is guarded, where (two landings)

**Landing 1 — fail-closed guard at consumption** (`src/tac/contest_score.py`,
extends the EXISTING canonical SoT per anti-duplicate-SoT — no new module):
- `verify_upstream_videos_clean(dir, expected_sum, expected_names)` → structured
  `RateDenominatorVerdict` (present/clean/strays/missing/sum_matches/report).
  Replicates evaluate.py:64 EXACTLY (rglob + is_file, counts dotfiles). Never
  raises for cleanliness — it REPORTS.
- `assert_upstream_videos_clean(...)` → raises `RateDenominatorMismatchError`
  iff the tree is PRESENT and NOT clean; NAMES the stray/missing files. NEVER
  deletes (upstream immutable — surfaced to operator).
- `rate_term()` calls a cached per-process guard `_assert_default_denominator_clean_cached()`
  **before any rate arithmetic**, gated on `uncompressed_size == UNCOMPRESSED_SIZE_BYTES`
  (i.e. only when the caller relies on the canonical constant; explicit
  non-canonical denominators — deliberate hypotheticals — skip it). One rglob of
  a ~1-file dir, cached → O(1) after first call. This also covers
  `compute_contest_score` + `break_even_d_seg` (both route through `rate_term`).
- Absent/unreadable tree → present=False → NEVER a violation (unverifiable → the
  constant stands; the guard never fabricates a violation from a missing surface).

**Landing 2 — warn-only preflight** (`src/tac/preflight.py`):
- `check_upstream_videos_dir_clean(*, repo_root, strict, verbose)` — detects
  stray (`._*` / `.DS_Store` / unexpected) + missing files under
  `upstream/videos/`, delegating to Landing 1's `verify_upstream_videos_clean`
  (single measurement SoT — no duplicated rglob). Inventory-scoped (the
  byte-sum-vs-constant assertion is Landing 1's job). Repo_root-parametric →
  testable. Wired warn-only into `preflight_all` next to the sibling
  filesystem-anti-rot gates.

## 4. Strict vs warn-only decision + live count

**Live count today: 0** (real `upstream/videos/` is clean — single `0.mkv` =
37,545,489). Per the Strict-flip atomicity rule this UNBLOCKS a same-commit
strict-flip.

**Decision: WARN-ONLY (deliberate, not purgatory).** Reasoning:
1. The HARD fail-closed enforcement already lives at `rate_term` (Landing 1) —
   where a WRONG SCORE would actually be produced. No wrong score can be computed
   internally even while the preflight only warns.
2. This operator's dev box is macOS; Finder generates `.DS_Store` routinely. A
   STRICT gate in the broad `preflight_all` sweep would intermittently
   false-block UNRELATED commits/dispatches on a transient artifact — the exact
   "never block an unrelated pipeline" rationale the two sibling filesystem
   anti-rot gates (`check_operating_manual_pointer_integrity`,
   `check_harness_failure_ledger_v2_hygiene`) carry. Sibling-consistent.
3. **Concrete strict-flip condition (avoids purgatory):** flip to `strict=True`
   once the stray-dotfile creation vector under `upstream/videos/` is
   structurally eliminated on the dev box / CI (so a strict broad gate cannot
   false-block on a transient `.DS_Store`). No waiver token — the fix is to clear
   the stray, never suppress the finding.

Operator may override to strict at will; the code is one-line-flip ready.

## 5. Catalog-row status

Claimed **Catalog #407** via `tools/claim_catalog_number.py claim` (atomic
counter authority; state file `.omx/state/next_catalog_number.txt` 407→408,
tracked, committed). Verified #407/#408 were unreferenced before claiming.
NOTE (apparatus-debt flag, NOT this task's scope): the atomic counter (was 407)
is out of sync with sister-memo/task cross-references in the docs/code that
mention higher IDs like `#823`/`#835`/`#836` — those are memo/incident/task
references, NOT tool-claimed catalog gate numbers (confirmed by inspecting the
claim log, which tops out well below them). #407 is the correct next gate number.
The CLAUDE.md catalog-table row for #407 is OWED (not landed here to avoid a
same-turn cap collision) — flagged for a follow-up docs pass.

## 6. Tests

`src/tac/tests/test_rate_denominator_guard.py` — **26 tests, all green**:
clean tree, evaluate.py:64 dotfile-count replication, AppleDouble stray,
`.DS_Store` stray, unexpected named file, nested-subdir stray, missing payload,
wrong-size sum mismatch, absent-dir-unverifiable, never-deletes-the-stray,
rate_term fail-closed on dirty cache, explicit-non-canonical-denominator bypass,
cached-guard absent-tree no-op, and the 8 preflight cases (clean / AppleDouble /
.DS_Store / missing / strict-raises / absent / names-fallback / real-repo live
count 0). Existing `test_contest_score_*` regression suite: 48 pass. ruff F clean.

## 7. Commit

`e66f225934` (serializer, post-edit shas, `[p0-ledger-ok]`, no co-author trailer;
self-collision bypass because the only overlapping checkpoint was ddm_dg1's own).
