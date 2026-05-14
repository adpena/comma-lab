"""Catalog #215 preflight gate tests.

The gate refuses substrate operator-authorize recipes whose full-run GPU is
A100/H100/L40S WITHOUT declaring ``min_smoke_gpu``. Same-line
``# MIN_SMOKE_GPU_OK:<reason>`` waiver accepts (placeholder rejected).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_modal_smoke_recipe_min_gpu_class_consistent,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_recipe(tmp_path: Path, name: str, body: str) -> Path:
    recipes_dir = tmp_path / ".omx" / "operator_authorize_recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    path = recipes_dir / f"substrate_{name}_modal_test_dispatch.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_recipe_t4_does_not_require_min_smoke_gpu(tmp_path: Path) -> None:
    _make_recipe(tmp_path, "cheap", 'platform: modal\ngpu: "T4"\n')
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert violations == []


def test_recipe_a100_without_min_smoke_gpu_violation(tmp_path: Path) -> None:
    _make_recipe(tmp_path, "heavy", 'platform: modal\ngpu: "A100"\n')
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "min_smoke_gpu" in violations[0]
    assert "A100" in violations[0]


def test_recipe_a100_with_min_smoke_gpu_clean(tmp_path: Path) -> None:
    _make_recipe(
        tmp_path,
        "good",
        'platform: modal\ngpu: "A100"\nmin_smoke_gpu: "A100"\n',
    )
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert violations == []


def test_recipe_h100_without_min_smoke_gpu_violation(tmp_path: Path) -> None:
    _make_recipe(tmp_path, "heaviest", 'platform: modal\ngpu: "H100"\n')
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_recipe_l40s_without_min_smoke_gpu_violation(tmp_path: Path) -> None:
    _make_recipe(tmp_path, "midweight", 'platform: modal\ngpu: "L40S"\n')
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_recipe_modal_gpu_env_var_form_a100(tmp_path: Path) -> None:
    _make_recipe(
        tmp_path, "envform", 'platform: modal\ngpu: "${MODAL_GPU:-A100}"\n'
    )
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "A100" in violations[0]


def test_recipe_waiver_accepts_violation(tmp_path: Path) -> None:
    _make_recipe(
        tmp_path,
        "waived",
        'platform: modal\ngpu: "A100"  # MIN_SMOKE_GPU_OK:cheap-substrate-forward-fits-T4\n',
    )
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert violations == []


def test_recipe_waiver_placeholder_rejected(tmp_path: Path) -> None:
    _make_recipe(
        tmp_path,
        "placeholder",
        'platform: modal\ngpu: "A100"  # MIN_SMOKE_GPU_OK:<reason>\n',
    )
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_recipe_invalid_min_smoke_gpu_class(tmp_path: Path) -> None:
    _make_recipe(
        tmp_path,
        "weirdclass",
        'platform: modal\ngpu: "A100"\nmin_smoke_gpu: "BANANA_GPU"\n',
    )
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "not in valid set" in violations[0]


def test_recipe_missing_gpu_field_skipped(tmp_path: Path) -> None:
    _make_recipe(tmp_path, "nogpufield", 'platform: modal\nsummary: x\n')
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    # Recipe missing top-level gpu field — Catalog #170 catches it; #215 skips.
    assert violations == []


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _make_recipe(tmp_path, "strictraise", 'platform: modal\ngpu: "A100"\n')
    with pytest.raises(PreflightError) as ei:
        check_modal_smoke_recipe_min_gpu_class_consistent(
            repo_root=tmp_path, strict=True
        )
    assert "Catalog #215" in str(ei.value)


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    _make_recipe(
        tmp_path,
        "stricthappy",
        'platform: modal\ngpu: "A100"\nmin_smoke_gpu: "A100"\n',
    )
    # No raise.
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path, strict=True
    )
    assert violations == []


def test_missing_recipes_dir_returns_empty(tmp_path: Path) -> None:
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert violations == []


def test_siren_live_recipe_has_min_smoke_gpu_a100(tmp_path: Path) -> None:
    """Live regression guard for the SIREN anchor."""
    siren_path = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml"
    )
    assert siren_path.is_file()
    body = siren_path.read_text(encoding="utf-8")
    assert 'min_smoke_gpu: "A100"' in body


def test_multiple_recipes_aggregated(tmp_path: Path) -> None:
    _make_recipe(tmp_path, "good1", 'platform: modal\ngpu: "A100"\nmin_smoke_gpu: "A100"\n')
    _make_recipe(tmp_path, "bad1", 'platform: modal\ngpu: "A100"\n')
    _make_recipe(tmp_path, "bad2", 'platform: modal\ngpu: "H100"\n')
    _make_recipe(tmp_path, "good2", 'platform: modal\ngpu: "T4"\n')
    violations = check_modal_smoke_recipe_min_gpu_class_consistent(
        repo_root=tmp_path
    )
    assert len(violations) == 2
