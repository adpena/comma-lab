# Contributing to pact / tac

Thank you for considering a contribution. This document follows the
[comma.ai openpilot](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md)
contribution conventions.

By submitting a contribution, you agree that:

1. Your contribution is licensed under the **MIT License** (see `LICENSE`).
2. You have the right to submit the contribution under that license.
3. You consent to having `Copyright (c) 2026 Alejandro (Alex) Peña` retained as
   the project copyright; your individual authorship is recorded in git history.

## Code of conduct

Be excellent to each other. Bad-faith contributions (license violations,
plagiarism, harassment, malicious code, attempts to leak operator infrastructure
state) are not welcome.

## Development workflow

```bash
# Clone
git clone https://github.com/adpena/pact.git
cd pact

# Install (dev extras include linter, type checker, test runner)
uv venv
uv pip install -e ".[dev,runtime]"

# Run tests
.venv/bin/pytest src/tac/tests/ tests/

# Lint and format
.venv/bin/ruff check src/tac/
.venv/bin/ruff format src/tac/

# Type-check
.venv/bin/mypy src/tac/
```

## Pull request expectations

- One logical change per PR.
- Tests for new behavior.
- No regressions in the existing test suite.
- All `.py` files in the diff carry a `# SPDX-License-Identifier: MIT` header
  on the first non-shebang line.
- Commit messages follow `<what changed>: <why>` (see `CLAUDE.md` "Git
  discipline" section).
- For trainer / codec / archive changes that may affect score, include either
  an empirical anchor (CUDA auth eval JSON), a derivation, or a
  `[provenance-only; no score claim]` tag.

## What we do NOT accept

- Edits to the pinned upstream snapshot (`upstream/`,
  `submissions/exact_current/inflate.py`,
  `submissions/exact_current/inflate.sh`, `start.sh`).
- Hard-coded local absolute paths (e.g. `/Users/<name>/...`,
  `/home/<name>/...`, `/tmp/...` as durable evidence). Use placeholders
  (`<repo-root>`, `<remote-home>`) or repo-relative paths.
- Score claims based on MPS, local CPU scorers, or non-Linux-x86_64 CPU eval
  without the `[macOS-CPU advisory]` tag.
- Dependencies that are GPL / AGPL in the default install path. Copyleft
  obligations belong in opt-in extras (see `THIRD_PARTY_NOTICES.md`).

## Release process

`pact` / `tac` follows semver (`MAJOR.MINOR.PATCH-PRERELEASE`):

- `0.x.y-rcN` — pre-1.0 release candidates. v0.2.0-rc1 is the current OSS
  posture-alignment milestone.
- `1.0.x` — was used for the early `tac` library milestones in the research
  sprint; that lineage is preserved in `CHANGELOG.md` for historical
  continuity.

Tags are created locally; pushing to a public remote requires explicit
operator approval per the "Public Disclosure Hygiene" rule in `CLAUDE.md`.

## License

By contributing, you license your work under the MIT License. See `LICENSE`.

`Copyright (c) 2026 Alejandro (Alex) Peña`
