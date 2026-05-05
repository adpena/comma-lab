---
name: tac Tooling Standards — ruff, ty, uv, full testing, fuzzing, docs
description: Non-negotiable tooling and quality standards for the tac library
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Tooling Stack
- **uv** — Package management (already in use, non-negotiable per CLAUDE.md)
- **ruff** — Linting + formatting (replaces flake8, black, isort, pylint)
- **ty** — Type checking (Astral's type checker, replaces mypy/pyright)
- **pytest** — Testing framework (already in use)
- **hypothesis** — Property-based testing / fuzzing

## Quality Standards

### Linting (ruff)
- ALL code must pass `ruff check src/tac/`
- ALL code must be formatted with `ruff format src/tac/`
- Config in pyproject.toml [tool.ruff] section
- Target: zero warnings, zero errors

### Type Checking (ty)
- ALL public functions must have type annotations
- ALL configs use pydantic models (already true for TrainConfig)
- `ty check src/tac/` must pass with zero errors
- `py.typed` marker already exists

### Testing
- Every module must have a corresponding test file
- Minimum: shape tests for all forward passes
- Minimum: round-trip tests for all encode/decode/quantize paths
- Integration tests: full training smoke (5 epochs)
- Target: >90% line coverage

### Fuzzing (hypothesis)
- Property-based tests for all quantization functions
- Fuzz: random input shapes, dtypes, values through all architectures
- Fuzz: random masks through encode/decode pipelines

### Documentation
- Every public function: docstring with Args, Returns, Example
- Every module: module-level docstring explaining purpose
- Architecture docs: diagrams for CPU and GPU lane pipelines
- API reference: auto-generated from docstrings
- Tutorial: "Train your first task-aware postfilter in 10 minutes"

## Repo Separation Plan
tac should be extractable into its own repo at any point:
- `src/tac/` is self-contained (no imports from parent `comma_lab`)
- All competition-specific code (inflate paths, compress.sh) stays in pact
- tac imports nothing from experiments/ or submissions/
- The scorer models (PoseNet/SegNet) are loaded by path, not hardcoded
- Configuration is injectable (pydantic models, CLI args)

When ready to separate:
1. Create `github.com/adpena/tac` repo
2. Move `src/tac/` → `tac/src/tac/`
3. Move `tests/test_*.py` → `tac/tests/`
4. Keep `pyproject.toml`, `Formula/tac.rb`, `README.md`
5. pact repo adds `tac` as a dependency (`pip install tac`)

**Why:** Production-quality tooling demonstrates engineering discipline.
**How to apply:** Set up ruff + ty in pyproject.toml. Run on CI. Fix all issues.
