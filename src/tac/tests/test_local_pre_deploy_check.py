# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools/local_pre_deploy_check.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("local_pre_deploy_check", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auth_eval_reachability_ignores_artifact_globs(tmp_path: Path) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "from pathlib import Path\n"
        "def main():\n"
        "    return list(Path('x').glob('contest_auth_eval*.json'))\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is False
    assert "has no auth_eval invocation" in message


def test_auth_eval_reachability_accepts_canonical_helper_call(tmp_path: Path) -> None:
    module = _load_module()
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "def main():\n"
        "    gate_auth_eval_call(archive='archive.zip')\n",
        encoding="utf-8",
    )

    passed, message = module.check_auth_eval_reachability(trainer)

    assert passed is True
    assert "invokes auth_eval" in message
