#!/usr/bin/env sh
# Astral ty type-check for the witness DSL authoring surface (operator 2026-07-08
# "We like uv, ruff, and ty"). Scopes `ty check` to src/tac/witness_dsl — the typed
# config layer + its DSL SoT — extending the repo [tool.ty] config (pyproject.toml)
# rather than a parallel config. Pragmatic gradual typing: annotate what we touch,
# surface real bugs (unresolved-reference = error per [tool.ty.rules]); we do NOT
# chase a perfectly-clean baseline across all 113 trainer flags (the ocean-boil the
# ty config itself warns against). POSIX-portable (sh, no bashisms) per the
# cross-platform-scripting discipline.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TY="${ROOT}/.venv/bin/ty"
if [ ! -x "$TY" ]; then TY="ty"; fi  # fall back to PATH if venv binary absent
# ty discovers its [tool.ty] config by walking UP from the CWD, and its src include/
# exclude globs are resolved relative to that config root — NOT to the path argument.
# Invoking with an absolute path from an arbitrary CWD can therefore miss the config
# (observed: "WARN No python files found under the given path(s)"). Anchor CWD at the
# repo root and pass a RELATIVE path so config discovery + include-glob matching agree.
cd "$ROOT"
echo "# ty check: src/tac/witness_dsl (config: pyproject.toml [tool.ty]; cwd=repo root)"
# NOTE (recovery 2026-07-08): a non-zero exit is EXPECTED for now — curriculum_dsl.py
# carries 2 pre-existing `unsupported-operator` FALSE-POSITIVES (ty cannot follow the
# `None not in (a, b)` guard at :1413) unrelated to the typed_config layer. This script
# is ADVISORY (not wired into any hard gate); the typed_config.py surface itself is
# error-clean (only gradual invalid-argument-type warnings on the DSL adapters).
exec "$TY" check src/tac/witness_dsl
