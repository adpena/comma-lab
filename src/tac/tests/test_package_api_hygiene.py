from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_project_version_matches_public_package_version() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    import tac

    assert pyproject["project"]["version"] == tac.__version__


def test_project_metadata_does_not_claim_stable_maturity() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    classifiers = set(pyproject["project"]["classifiers"])
    maturity_classifiers = {c for c in classifiers if c.startswith("Development Status ::")}

    assert maturity_classifiers == {"Development Status :: 3 - Alpha"}
    assert "Development Status :: 5 - Production/Stable" not in classifiers


def test_top_level_import_keeps_heavy_public_api_lazy() -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    code = (
        "import sys; "
        "import tac; "
        "print(int('torch' in sys.modules), int('pydantic' in sys.modules))"
    )

    output = subprocess.check_output([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, text=True)

    assert output.strip() == "0 0"


def test_lazy_public_api_exports_stay_in_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    import tac

    exported_lazy_symbols = set(tac.__all__) - {"__version__"}
    assert set(tac._LAZY_PUBLIC_API) == exported_lazy_symbols
    assert "Trainer" in dir(tac)

    calls: list[tuple[str, str]] = []

    def fake_import_module(module_path: str, package: str) -> SimpleNamespace:
        calls.append((module_path, package))
        return SimpleNamespace(Trainer="trainer sentinel")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert tac.__getattr__("Trainer") == "trainer sentinel"
    assert calls == [(".training", "tac")]

    with pytest.raises(AttributeError, match="does_not_exist"):
        tac.__getattr__("does_not_exist")
