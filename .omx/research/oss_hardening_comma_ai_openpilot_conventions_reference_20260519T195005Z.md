# comma.ai / openpilot canonical OSS conventions reference
Generated: 2026-05-19T19:55:00Z
Lane: lane_comprehensive_oss_hardening_per_round_6_7_20260519
Surveyor: comprehensive_oss_hardening_p_20260519T195005Z

## Source-of-truth repos surveyed

- https://github.com/commaai/openpilot (production driver assistance OS; MIT)
- https://github.com/commaai/comma_video_compression_challenge (the contest we target; MIT)

Both repos pinned via `gh api repos/commaai/<name>/...` as of 2026-05-19.

## Canonical metadata pattern

### LICENSE
- MIT License
- Copyright owner: `Vehicle Researcher` (openpilot) / explicit author (contest)
- Plain SPDX-standard MIT body (`text = "MIT License"` in pyproject)

### Top-level required files (intersection of openpilot + contest)
- `LICENSE` (MIT)
- `README.md` (centered HTML header + badge row + quickstart)
- `pyproject.toml` (uv-managed; `requires-python = "~= 3.X"`)
- `.python-version`
- `.gitignore` / `.gitattributes`
- CI workflow under `.github/workflows/`

### openpilot adds (production scale)
- `RELEASES.md` (changelog with channel notes)
- `SECURITY.md`
- `.editorconfig`
- `Dockerfile.openpilot`
- `conftest.py` (project-wide pytest fixtures)
- `uv.lock` (locked dependency snapshot)

## README.md canonical layout (openpilot pattern)

```
<div align="center" style="text-align: center;">

<h1>project-name</h1>

<p>
  <b>One-line elevator pitch.</b>
  <br>
  Second-line context.
</p>

<h3>
  <a href="https://docs.example.ai">Docs</a>
  <span> · </span>
  <a href="...">Contribute</a>
  <span> · </span>
  <a href="https://discord.example.ai">Community</a>
</h3>

Quick start: `<one-liner>`

[![CI](badge)](url)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/handle)](url)
[![Discord](https://img.shields.io/discord/ID)](url)

</div>
```

Sections (in order):
1. Header + badges + quickstart one-liner
2. Hero video/image table (if applicable)
3. "Using X in Y" practical wiring
4. Branches/versions table
5. Development setup
6. Testing instructions
7. Contributing pointer (link to `docs/CONTRIBUTING.md`)
8. Licensing footer

## pyproject.toml canonical pattern

### `[project]` section
- `name`, `version`, `description`, `authors`, `license`
- `requires-python` with explicit upper bound (openpilot: `">= 3.12.3, < 3.13"`)
- `dependencies` list with inline comments explaining why each is required

### `[build-system]`
- `hatchling.build` is openpilot's choice
- `setuptools.build_meta` is also acceptable (contest, smaller projects)

### `[tool.ruff]` canonical config
```toml
[tool.ruff]
indent-width = 2  # openpilot; 4 is also common
line-length = 160  # openpilot is wider than PEP8

lint.select = [
  "E", "F", "W",      # pycodestyle + pyflakes
  "PIE", "C4", "ISC", # complexity / comprehensions / string-concat
  "A", "B",           # bugbear
  "NPY",              # numpy
  "UP",               # pyupgrade
  "TRY203", "TRY400", "TRY401",  # try/except discipline
  "RUF008", "RUF100",            # ruff-specific
  "TID251",                      # tidy-imports
  "PLE", "PLR1704",              # pylint errors
]

lint.ignore = [
  "E741",  # ambiguous variable name (E/l/I etc., often physics)
  "E402",  # module-level import not at top
  "C408",  # unnecessary dict call (often clearer)
  "ISC003", "B027", "B024",
  "NPY002",  # new numpy random syntax can be worse
]

exclude = [
  "vendored_dirs",
  "*.ipynb",
  "generated",
]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
# Common pattern: ban internal package abbreviations to force `package.X` form
"selfdrive".msg = "Use openpilot.selfdrive"
"time.time".msg = "Use time.monotonic"
"unittest".msg = "Use pytest"

[tool.ruff.format]
quote-style = "preserve"
```

### `[tool.pytest.ini_options]` canonical
```toml
minversion = "6.0"
addopts = "--ignore=vendored -Werror --strict-config --strict-markers --durations=10 -n auto"
markers = [
  "slow: tests that take awhile",
  "skip_ci: tests skipped in CI",
]
```

### Type checker preference
- openpilot uses `ty` (Astral's Rust-native type checker) NOT mypy or pyright
- `[tool.ty.rules]` with `ignore` for known-noisy categories (Cython, capnp, raylib)

### `[tool.uv]`
```toml
[tool.uv]
python-preference = "only-managed"
```

## CI workflow canonical pattern (openpilot tests.yaml)

```yaml
name: tests

on:
  push:
    branches: [master]
  pull_request:
  workflow_dispatch:
  workflow_call:
    inputs:
      run_number: { default: '1', required: true, type: string }

concurrency:
  group: tests-ci-run-${{ inputs.run_number }}-${{ ... }}
  cancel-in-progress: true

env:
  CI: 1
  PYTHONPATH: ${{ github.workspace }}
  PYTEST: pytest --continue-on-collection-errors --durations=0 -n logical

jobs:
  build_release:
    name: build release
    runs-on: ubuntu-24.04  # or namespace profile for paid pool
    steps:
    - uses: actions/checkout@v6  # NOT v4 (Node.js 20 deprecation)
      with:
        submodules: true
    - run: ./tools/op.sh setup
    - name: Build
      timeout-minutes: 30
      run: ...
    - name: Run tests
      timeout-minutes: 1
      run: ...
```

**Key CI conventions (canonical):**
1. `actions/checkout@v6` (current; v4 emits Node.js 20 deprecation warning forced June 2026)
2. `actions/setup-python@v6` (current)
3. Explicit `timeout-minutes` on every step (NOT just the job)
4. `runs-on: ubuntu-24.04` (NOT `ubuntu-latest` for reproducibility)
5. `concurrency` block with `cancel-in-progress: true`
6. `env.CI: 1` set globally
7. Cancel pattern: `nick-fields/retry@<pinned-sha>` for LFS / network-fragile steps

## Contributing/SECURITY pattern

- `docs/CONTRIBUTING.md` (NOT root `CONTRIBUTING.md` for openpilot; root is acceptable)
- Short, direct, code-of-conduct-light (no template-style verbosity)
- Lists what NOT to accept (license violations, plagiarism, malicious code)
- `SECURITY.md` links to security@comma.ai email (private disclosure)

## Tone and style (canonical)

- **Direct + technical** — no marketing voice, no emoji in code/docs
- **No SPDX-License-Identifier required** in source files (openpilot does NOT use them)
- Lowercase project names in titles (`openpilot`, not `OpenPilot`)
- "we" instead of "the project" or "this software"
- Specific quantitative claims ("300+ supported cars") not vague superlatives

## What we DIVERGE from + rationale

### `adpena/comma-lab`
- **SPDX headers required** — extends comma.ai (we already added them; keep)
- **Apostrophe character "Peña"** in Copyright — operator's actual name
- **More verbose CONTRIBUTING.md** — covers research-evidence-grade discipline that openpilot doesn't have
- **`pre-commit-config.yaml` recommended** — comma.ai doesn't use; we benefit from local enforcement

### `adpena/tac`
- **`hatchling.build`** matches openpilot — keep
- **No `[tool.ruff]`** in pyproject yet — gap, should add
- **No `[tool.pytest.ini_options]`** in pyproject yet — gap, should add
- **Two Python versions** in CI matrix (3.11 + 3.12) is appropriate; openpilot pins to 3.12 only

## Recommendations for both repos

1. Upgrade CI actions to `@v6` (mandatory before June 2026)
2. Add `actions/setup-python` Python version matrix as needed (3.11 + 3.12 minimum)
3. Add `concurrency` block with cancel-in-progress
4. Set `timeout-minutes` on every step
5. Add `.editorconfig`
6. Add `SECURITY.md` (even minimal: "report security issues to ...")
7. Add `[tool.ruff]` + `[tool.pytest.ini_options]` to tac pyproject
8. Verify `pyproject.toml` `[project.urls]` all resolve

## Cross-references

- comma-lab Slot H audit: `.omx/research/oss_audit_adpena_tac_for_pr_link_20260519T185843Z.md`
- comma-lab Slot N sanitization: `.omx/research/comma_lab_sanitization_sweep_20260519T194221Z.md`
- comma-lab Slot O tac CI green: `.omx/research/tac_ci_fix_authoring_tests_20260519T193600Z.md`

## Discipline (Catalog gates honored)

- Catalog #229 PV: verified files exist + read full content before drafting
- Catalog #208 (docs/local-paths): all `/tmp/` references are scratch contexts not durable evidence
- Catalog #287 (placeholder-rationale): no placeholder tokens used
- Catalog #110/#113 HISTORICAL_PROVENANCE: this is a NEW reference memo, not a mutation
