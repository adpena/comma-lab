# SPDX-License-Identifier: MIT
"""Unit tests for the numpy-portable inflate verifier (ast-only; runs everywhere)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
)
from tac.substrates._shared.mlx_score_aware.portability import (
    FORBIDDEN_INFLATE_IMPORT_ROOTS,
    assert_numpy_portable_inflate,
)


def test_forbidden_roots_are_mlx_and_torch() -> None:
    assert set(FORBIDDEN_INFLATE_IMPORT_ROOTS) == {"mlx", "torch"}


def test_accepts_numpy_pil_inflate(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text(
        "import numpy as np\nfrom PIL import Image\nimport struct\n",
        encoding="utf-8",
    )
    result = assert_numpy_portable_inflate(inflate)
    assert result["numpy_portable"] is True
    assert "numpy" in result["import_roots"]


def test_rejects_mlx_import(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text("import mlx.core as mx\n", encoding="utf-8")
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        assert_numpy_portable_inflate(inflate)


def test_rejects_torch_import(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text("import torch\n", encoding="utf-8")
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        assert_numpy_portable_inflate(inflate)


def test_rejects_from_torch_import(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text("from torch.nn import functional as F\n", encoding="utf-8")
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        assert_numpy_portable_inflate(inflate)


def test_rejects_dotted_mlx_submodule(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text("from mlx.utils import tree_flatten\n", encoding="utf-8")
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        assert_numpy_portable_inflate(inflate)


def test_allows_mlx_mention_in_comment(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text(
        "import numpy as np\n"
        "# decodes MLX-trained weights but imports no mlx / torch\n"
        '"""docstring mentions torch and mlx but imports neither."""\n',
        encoding="utf-8",
    )
    assert assert_numpy_portable_inflate(inflate)["numpy_portable"] is True


def test_allows_relative_import(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text(
        "import numpy as np\nfrom . import archive\n", encoding="utf-8"
    )
    assert assert_numpy_portable_inflate(inflate)["numpy_portable"] is True


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="not found"):
        assert_numpy_portable_inflate(tmp_path / "does_not_exist.py")
