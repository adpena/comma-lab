# ddm_gp1 — make the auth-eval parity reference LOCK-DERIVED, not path-derived

## THE ONE THING
`experiments/contest_auth_eval.py:658` hardcodes
`upstream_ref = upstream_dir / ".venv" / "bin" / "python"`. When that path does not exist the gate
refuses with `reason: "missing"` — **unconditionally, and unsatisfiably**. Replace the *missing*
branch with a reference DERIVED FROM THE PINNED LOCK for a DECLARED dependency group.

This UNBLOCKS the pointer: a measured candidate at **S = 0.170536856816211 @ 188,636 B**
(−0.001604 vs the bar) is sitting refused on this branch alone.

## WHY THE OBVIOUS FIXES ARE ALL WRONG (do not re-propose these)
Four alternatives were considered and REFUTED against the code. Re-proposing one is a regression.

1. **Trust the passed `--upstream-python`.** REFUTED. `contest_auth_eval.py:673-683` carries a
   hardening comment dated **2026-08-10**: the parity block *previously* ran only when the flag was
   ABSENT, "so passing the flag disabled the check entirely — an operator DECLARATION of parity
   stood in for a MEASUREMENT of it." Trusting the flag re-introduces the defect cured that day.
   **Declaration is never proof. This is the invariant the whole arm must preserve.**
2. **Run `contest_auth_eval.py` itself under the locked interpreter.** INSUFFICIENT. The
   `upstream_ref.exists()` test at :659 still fails and still refuses, whatever interpreter runs.
3. **Build the venv at `upstream/.venv`.** FORBIDDEN. `tac.contest_compliance._iter_upstream_files`
   rejects symlinks BEFORE any exclusion rule, deliberately: *"otherwise evaluator code or weights
   could be consumed through an unbound path while the purported full-tree digest remained
   unchanged."* A venv is exactly where that would happen. `.venv/lib64` is a symlink; this killed a
   real dispatch. `_SKIP_DIR_NAMES` = {.git, .pytest_cache, .mypy_cache, node_modules} — adding
   `.venv` would NOT help, because symlink-rejection runs first, by design. Also: writing into the
   pinned snapshot violates the read-only-upstream rule outright.
4. **Parse `uv.lock` directly.** AMBIGUOUS. The lock holds MULTIPLE torch entries
   (2.9.0+cu126 / 2.9.0+cu128 / 2.9.0+cu130 / 2.10.0 / 2.10.0+cpu). Resolution is PER-GROUP.
   Picking one by hand is the same guessing defect in a new place.

## THE SURVIVING DESIGN (verified by MAIN before this charter was written)
Derive the reference by asking **uv** to resolve the **pinned lock** for a **declared group**.
The operator may declare WHICH AXIS (legitimate — `upstream/.github/workflows/eval.yml:32` selects
it: `UV_GROUP: ${{ inputs.runner == 'linux-nvidia-t4' && 'cu128' || 'cpu' }}`). The operator may
NEVER declare THAT PARITY HOLDS. That split is the whole point.

MEASURED by MAIN on this repo, 2026-08-10 (reproduce it before trusting it):
```
uv export --frozen --no-emit-project --no-hashes --format requirements-txt \
    --directory upstream --group cu128
  → numpy==2.3.4 · timm==1.0.22 · torch==2.9.0+cu128
  → torchvision==0.24.0      ; ... platform_machine == 'aarch64' ...
  → torchvision==0.24.0+cu128 ; ... platform_machine != 'aarch64' ...
```
- **READ-ONLY CONFIRMED**: `upstream/uv.lock` sha256 `eca4542ad8d21354…` byte-identical before and
  after. Re-verify this yourself; if the lock mutates, ABORT — upstream is immutable.
- This EQUALS the image's built venv (`3.11.12 2.9.0+cu128 0.24.0+cu128 1.0.22 2.3.4`), so the
  honest case passes by measurement.
- All five flags verified against the real `uv export --help`. Do not invent flags; re-grep.

## THE MARKER TRAP (this is the one that will bite)
torchvision exports as TWO rows split by environment marker. A naive first-match parser picks the
`aarch64` row and manufactures a FALSE MISMATCH on x86_64 — a refusal that looks like a real finding.
Evaluate each row's marker in the **EVALUATION interpreter's** marker environment (`packaging.markers`
with that interpreter's `sys_platform`/`platform_machine`/`python_full_version`), not the wrapper's,
not the host's. If two rows survive marker evaluation, that is genuine ambiguity → REFUSE with a
reason naming both, never pick one.

## FAIL-CLOSED CONTRACT (non-negotiable — this is a REFUSAL gate on score authority)
Every one of these keeps the refusal, with a precise machine-readable reason:
- no group declared · uv absent · export rc≠0 · a parity package missing from the export ·
  marker evaluation ambiguous · the lock file's sha changes across the call.
Weakening any of these manufactures a fake score claim (NO-FAKE #8). When in doubt, REFUSE.
The existing `upstream/.venv` branch is UNCHANGED — this only replaces the unsatisfiable
`missing` branch.

## OPTIMAL FORM
- REFERENCE form: the gate's own existing comparison shape (per-package + python_version), reused
  verbatim; only the SOURCE of `reference` changes. No new comparison semantics.
  Provenance pins:
  - `experiments/contest_auth_eval.py:658` (the derived-path defect), `:673-683` (the 2026-08-10
    declaration-vs-measurement hardening that must survive), `:684-726` (the mismatch/report shape).
  - `src/tac/contest_compliance.py:48-76` (`_iter_upstream_files`, symlink-first rejection).
  - `src/tac/deploy/modal/locked_env_probe.py` — the SISTER SITE where this same derivation defect
    was already cured; its docstring names it: *"an earlier version guessed
    `{upstream_dir}/.venv/bin/python` and that guess was the defect."*
  - Refused row: `fc-01KZNZNYECSJP143ZHZ14452RX`, archive sha
    `0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84` @ 188,636 B;
    decoded provenance persisted at
    `experiments/results/modal_auth_eval/archive_0f5a797fda84/artifacts/provenance.json`.
  - Bar: `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd` @ 191,052 B,
    S = 0.172141297491896447.
- SCOPE reductions (legal): none needed; the change is small.
- MECHANISM reductions: **none permitted**. Do not stub the marker evaluation. Do not hardcode
  "cu128". Do not skip the fail-closed branches because they are tedious.

## VERIFICATION (the deliverable — the byte count is not the point here, the REFUSAL SEMANTICS are)
1. **POSITIVE CONTROL, executed, receipt recorded**: a genuinely mismatched interpreter (e.g. a env
   with torch != the lock's) MUST still produce `env_mismatch` + `score_claim=False`. A cure that
   cannot refuse is not a cure. Show the executed rc.
2. **NEGATIVE CONTROL**: an interpreter matching the lock resolution passes with NO mismatch.
3. **MARKER CONTROL**: the torchvision two-row case resolves to the correct row for a simulated
   x86_64-linux evaluation env, and to the OTHER row for aarch64. Both directions.
4. **IMMUTABILITY**: assert `upstream/uv.lock` sha256 is unchanged across the whole test run.
5. Two review passes per touched `.py` via `tools/review_tracker.py mark-file <f> --status reviewed`.
   `REVIEW_GATE_OVERRIDE=1` is FORBIDDEN with `.py` files.
6. Commit through `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`
   per file, tags `[no-triality] [p0-ledger-ok]`, NO co-author trailer.

## DO NOT
- Do not touch anything under `upstream/` (read-only; read the lock, never write it).
- Do not re-fire the paid Modal row. Emit READY + the exact re-fire command; MAIN owns the fire and
  the single-flight claim.
- Do not claim a score. The row stays `score_claim=False` until it re-fires clean.

## STORES TO CONSULT FIRST
task #1004 (the full root-cause + the open sub-question: is `evaluation_python` the interpreter that
actually ran `upstream/evaluate.py`, or only the wrapper? resolve it at source — it may change which
branch is even reached) · `src/tac/deploy/modal/locked_env_probe.py` + its tests (the cured sister) ·
`experiments/modal_auth_eval.py` (`UPSTREAM_LOCKED_VENV`, `UPSTREAM_UV_GROUP_CUDA/CPU`) ·
task #1005 (3 RED CPU-dispatcher tests — same file family; check whether this change interacts).

## NEXT_IF_RESUMED
Emit the standard block. On success the named successor is MAIN re-firing the CUDA row, then the
CPU axis (`--group cpu`, which the lock resolves to torch 2.10.0+cpu — a DIFFERENT reference; that
difference is a finding the CPU dispatcher needs).
