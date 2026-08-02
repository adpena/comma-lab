# SPDX-License-Identifier: MIT
"""Catalog #184 scope extension: no module-scope heavy imports on the hook path.

Task #892. `tools/preflight_hook.py` runs `python -m tac.preflight` on EVERY
commit and push. A module-scope `import torch` in `src/tac/preflight.py` was
therefore paid every time -- including in `--no-codebase` mode, which examines
0 of 27 gates and never reaches a `torch.load`.

The cost was BIMODAL, which is why it read as flaky drift rather than a fixed
tax: 0.48s warm, 43.86s real / 0.44s user cold (blocked faulting torch's ~1 GB
of dylibs back into the page cache, which the fleet's memory pressure evicts).
The hook's own 30s timeout sat inside that gap -- green on every warm run,
rc=124 on every cold one.

These tests pin the two halves that matter: the gate FIRES on a re-introduction
(a refusal gate never shown to fire is untrusted, per the positive-control
discipline) and does NOT fire on the cure (a function-local import).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import _check_184_module_scope_heavy_imports as scan


def _mk(tmp_path: Path, body: str) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "tac" / "preflight.py").write_text(body, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "body",
    [
        "import torch\n",
        "from torch import load\n",
        "import torch.nn.functional as F\n",
        "import mlx.core as mx\n",
        "import cv2\n",
    ],
)
def test_module_scope_heavy_import_fires(tmp_path: Path, body: str) -> None:
    """POSITIVE CONTROL -- the gate must actually refuse the thing it names."""
    v = scan(_mk(tmp_path, body))
    assert len(v) == 1, v
    assert "MODULE SCOPE" in v[0]


def test_function_local_import_is_the_cure_and_never_fires(tmp_path: Path) -> None:
    """The fix must not trip the gate, or the gate would forbid its own remedy."""
    body = "def f(p):\n    import torch\n    return torch.load(p)\n"
    assert scan(_mk(tmp_path, body)) == []


@pytest.mark.parametrize(
    "body",
    [
        # The most likely evasion: a "defensive" guard that still pays in full.
        "try:\n    import torch\nexcept ImportError:\n    torch = None\n",
        "try:\n    import nonexistent_xyz\nexcept ImportError:\n    import torch\n",
        "import os\nif os.environ.get('X'):\n    import torch\n",
        "with open(__file__) as _f:\n    import torch\n",
        "class C:\n    import torch\n",
    ],
)
def test_import_time_but_indented_still_fires(tmp_path: Path, body: str) -> None:
    """Scope is COST, not indentation -- all of these run on `import`."""
    v = scan(_mk(tmp_path, body))
    assert len(v) == 1, (body, v)


def test_same_line_waiver_is_respected(tmp_path: Path) -> None:
    body = "import torch  # HOOK_HEAVY_IMPORT_OK:oracle parity needs it at import\n"
    assert scan(_mk(tmp_path, body)) == []


def test_bare_waiver_token_without_rationale_does_not_waive(tmp_path: Path) -> None:
    """Placeholder-rationale rejection (Catalog #287 sister discipline)."""
    body = "import torch  # HOOK_HEAVY_IMPORT_OK:\n"
    assert len(scan(_mk(tmp_path, body))) == 1


@pytest.mark.parametrize("body", ["import json\n", "import numpy as np\n", "import re\n"])
def test_light_imports_never_fire(tmp_path: Path, body: str) -> None:
    """numpy is deliberately NOT heavy: 0.04s and small enough to stay resident."""
    assert scan(_mk(tmp_path, body)) == []


def test_missing_file_is_a_no_op_not_a_crash(tmp_path: Path) -> None:
    assert scan(tmp_path) == []


def test_live_repo_is_clean() -> None:
    """The landed state must satisfy its own gate (live count 0)."""
    repo_root = Path(__file__).resolve().parents[3]
    assert scan(repo_root) == []
