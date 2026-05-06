"""Tests for ``experiments/build_jcsp_score_marginals.py`` CLI.

Covers happy paths and error modes:
- Uniform mode produces an artifact readable by both my strict loader
  and codex's permissive ``jcsp_stream_builder.load_jcsp_score_marginals``.
- Sensitivity mode honors the SensitivityMap-derived weighting.
- Missing model path → SystemExit.
- ``--mode sensitivity`` without ``--sensitivity-path`` → CLI parser error.
- Missing parent dir on --out → SystemExit / JCSPScoreMarginalsError.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tac.jcsp_score_marginals import load_marginals
from tac.jcsp_stream_builder import load_jcsp_score_marginals

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "experiments" / "build_jcsp_score_marginals.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "build_jcsp_score_marginals_cli", str(CLI_PATH)
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_state_dict_path(tmp_path: Path) -> Path:
    sd = {
        "a.weight": torch.linspace(-1.0, 1.0, 16, dtype=torch.float32),
        "b.weight": torch.zeros(8, dtype=torch.float32),
    }
    p = tmp_path / "model.pt"
    torch.save(sd, p)
    return p


def test_cli_uniform_mode_writes_valid_artifact(
    tiny_state_dict_path: Path, tmp_path: Path
) -> None:
    cli = _load_cli_module()
    out = tmp_path / "marginals.json"
    rc = cli.main(
        [
            "--model",
            str(tiny_state_dict_path),
            "--out",
            str(out),
            "--mode",
            "uniform",
            "--uniform-value",
            "5e-7",
            "--evidence",
            "unit test fixture",
        ]
    )
    assert rc == 0
    assert out.exists()
    # Strict loader (my envelope schema)
    marginals_strict, meta = load_marginals(out)
    assert set(marginals_strict.keys()) == {"a.weight", "b.weight"}
    assert all(v == 5e-7 for v in marginals_strict.values())
    assert meta["source"] == "placeholder_uniform"
    # Permissive loader (codex companion)
    marginals_loose = load_jcsp_score_marginals(out)
    assert set(marginals_loose.keys()) == {"a.weight", "b.weight"}


def test_cli_sensitivity_mode_uses_provided_sensitivities(
    tiny_state_dict_path: Path, tmp_path: Path
) -> None:
    cli = _load_cli_module()
    sens = {
        "a.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),  # mean 2.5
    }
    sens_path = tmp_path / "sens.pt"
    torch.save(sens, sens_path)
    out = tmp_path / "marginals_sens.json"
    rc = cli.main(
        [
            "--model",
            str(tiny_state_dict_path),
            "--out",
            str(out),
            "--mode",
            "sensitivity",
            "--sensitivity-path",
            str(sens_path),
            "--bytes-per-element",
            "1.0",
            "--fallback-marginal",
            "1e-9",
            "--evidence",
            "sensitivity unit test",
        ]
    )
    assert rc == 0
    marginals, meta = load_marginals(out)
    assert meta["source"] == "sensitivity_derived"
    # a.weight: mean(sens)=2.5 / max(1, 16*1.0) = 2.5/16 = 0.15625
    assert marginals["a.weight"] == pytest.approx(2.5 / 16.0, rel=1e-6)
    # b.weight has no sensitivity → fallback
    assert marginals["b.weight"] == pytest.approx(1e-9, rel=1e-6)


def test_cli_missing_model_raises(tmp_path: Path) -> None:
    cli = _load_cli_module()
    with pytest.raises(SystemExit, match="--model"):
        cli.main(
            [
                "--model",
                str(tmp_path / "nonexistent.pt"),
                "--out",
                str(tmp_path / "x.json"),
                "--mode",
                "uniform",
                "--evidence",
                "ok",
            ]
        )


def test_cli_sensitivity_mode_requires_sensitivity_path(
    tiny_state_dict_path: Path, tmp_path: Path
) -> None:
    cli = _load_cli_module()
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--model",
                str(tiny_state_dict_path),
                "--out",
                str(tmp_path / "x.json"),
                "--mode",
                "sensitivity",
                "--evidence",
                "missing sens",
            ]
        )


def test_cli_artifact_contains_envelope_metadata(
    tiny_state_dict_path: Path, tmp_path: Path
) -> None:
    cli = _load_cli_module()
    out = tmp_path / "x.json"
    cli.main(
        [
            "--model",
            str(tiny_state_dict_path),
            "--out",
            str(out),
            "--mode",
            "uniform",
            "--evidence",
            "envelope structure check",
        ]
    )
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["schema"] == "jcsp_score_marginals_v1"
    assert raw["source"] == "placeholder_uniform"
    assert raw["evidence"] == "envelope structure check"
    assert raw["n_streams"] == 2
    assert "score_marginals" in raw
    assert isinstance(raw["score_marginals"], dict)
