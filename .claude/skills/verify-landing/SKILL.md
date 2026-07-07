---
name: verify-landing
description: Canonical verification and commit chain for landing .py changes in this repo — ruff F check, targeted tests, review-gate two clean passes, serializer commit with post-edit shas. Use before committing any Python change.
---

# Verify a .py landing (canonical chain)

Run these in order for every set of touched `.py` files. Do not skip steps;
do not reorder.

1. **Lint (undefined-name class):**
   ```bash
   .venv/bin/ruff check --select F <file1> <file2> ...
   ```
   Fix every finding before proceeding.

2. **Tests for the touched modules** (match the sibling test file under
   `src/tac/tests/` or `tests/`):
   ```bash
   .venv/bin/python -m pytest src/tac/tests/test_<module>*.py -q
   ```

3. **Review gate — two clean passes per .py file.** Actually re-read the
   file between passes (pass 2 reviews your pass-1 fixes; fixes are
   unreviewed new code):
   ```bash
   python tools/review_tracker.py mark-file <file> --status reviewed
   # re-read the file, then:
   python tools/review_tracker.py mark-file <file> --status reviewed
   ```
   NEVER use `REVIEW_GATE_OVERRIDE=1` on `.py` files. If the gate blocks,
   the code needs review — that is the gate working.

4. **Commit via the serializer with POST-EDIT working-tree shas**
   (never bare `git commit`):
   ```bash
   SHA=$(shasum -a 256 <file> | awk '{print $1}')
   python tools/subagent_commit_serializer.py \
       --message "<what changed>: <why>" \
       --files <file1> <file2> ... \
       --expected-content-sha256 "<file>=${SHA}"
   ```
   Repeat the `--expected-content-sha256` flag per file. On rc=4 a sister
   agent landed first: re-read the file, re-base your edit, re-hash, retry.

5. **Report honestly:** list what ran and what you did NOT verify.
