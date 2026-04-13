from __future__ import annotations

from tac.deploy.cloud_bootstrap import BOOTSTRAP_STUB


def render_bootstrap(*, required_symbols: tuple[str, ...], dataset_hint: str) -> str:
    """Render a provider-agnostic tac bootstrap preamble for embedding in kernel scripts.

    The generated preamble uses a two-stage approach:
      Stage 1 — stdlib-only pre-install (BOOTSTRAP_STUB): finds the tac wheel in the
        dataset mount, installs via uv or pip, then calls cloud_bootstrap.bootstrap()
        for post-install verification.
      Stage 2 — entrypoint validation: verifies that tac.entrypoints exposes all
        *required_symbols* expected by the experiment script.

    The preamble is designed to be prepended verbatim to any Python script; it defines
    and immediately calls ``_tac_bootstrap()``.  ``SCRIPT_PATH`` must be in scope when
    the preamble runs (define it as ``Path(__file__).resolve()`` before the preamble).

    Args:
        required_symbols: names that must be present as attributes of tac.entrypoints.
        dataset_hint:     dataset slug subdirectory expected to contain the tac wheel.
    """
    quoted = ", ".join(repr(name) for name in required_symbols)
    return f"""\
import sys
from pathlib import Path
SCRIPT_PATH = Path(__file__).resolve()

{BOOTSTRAP_STUB}

_tac_bootstrap(
    dataset_hint={dataset_hint!r},
    extra_search_dirs=(str(SCRIPT_PATH.parent),),
)

# Verify required entrypoints are present in the installed wheel
from tac import entrypoints as _tac_ep
_missing_ep = [n for n in ({quoted},) if not hasattr(_tac_ep, n)]
if _missing_ep:
    raise ImportError(
        f"tac wheel is missing entrypoints: {{_missing_ep}}.\\n"
        f"  Rebuild and re-upload:\\n"
        f"    uv build && kaggle datasets version -p dist/ -m \\'tac vX.Y.Z\\'"
    )
del _missing_ep, _tac_ep
""".strip()
