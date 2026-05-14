# SPDX-License-Identifier: MIT
"""FIX-B Wave 3 unblock: regression tests for tac.losses package re-exports.

The historical single-file ``tac.losses`` was split into the
``tac.losses.{core, cat_entropy_v2}`` package by GGGG's PR95 port
(commit f2a20ed6). Forty-seven legacy callsites still do
``from tac.losses import X`` for various ``X``. Each name must remain reachable
at the top level of ``tac.losses.__init__`` so that:

  1. Runtime imports succeed (``from tac.losses import scorer_loss`` etc.).
  2. AST-based preflight ``Check 13`` (DEAD-RESOLVER / DEAD-IMPORT) sees each
     name as an ``ImportFrom`` alias at the top of ``__init__.py``.

The original ``from .core import *`` + ``globals()`` injection produced (1) but
not (2). The explicit ``from .core import (...)`` form lands both.

These tests guard against three recurrence vectors:

  (A) Someone replaces the explicit import with ``from .core import *`` again.
  (B) Someone adds a public symbol to ``core.py`` without re-exporting it here.
  (C) Someone removes one of the 47-caller-required names without grepping
      for callsites first.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


# Every symbol that the 47 legacy callsites historically imported.
# Sourced from the original `Check 13` failure on commit eb70062c plus
# the cat_entropy_v2 explicit pair.
HISTORICAL_REQUIRED_NAMES = frozenset(
    {
        # Sinkhorn constants
        "DEFAULT_SINKHORN_BLUR",
        "DEFAULT_SINKHORN_ITERS",
        "SEGMENTATION_SURROGATE_FISHER_RAO",
        "SEGMENTATION_SURROGATE_SINKHORN",
        "SEGMENTATION_SURROGATE_SOFT_COSINE",
        # Public scorer helpers
        "scorer_forward_pair",
        "scorer_loss",
        "scorer_loss_cached",
        "scorer_loss_cached_with_aux",
        "scorer_loss_pcgrad",
        "scorer_loss_terms_btchw",
        "scorer_loss_terms_cached_btchw",
        "scorer_loss_with_aux",
        "eval_scorer_loss",
        # Public training losses
        "focal_segnet_ste_loss",
        "frequency_aware_loss",
        "kl_distill_segnet_only",
        "parse_class_weights_csv",
        "posenet_embedding_loss",
        "segnet_surrogate_per_pixel",
        "segnet_uncertainty_weighted_loss",
        "uniward_quant_noise_loss",
        # Private helper that is intentionally part of the cross-trainer API
        "_hwc_to_chw",
        # cat_entropy_v2 explicit exports
        "CatEntropyV2Config",
        "cat_entropy_v2",
    }
)


def _init_path() -> Path:
    import tac.losses

    init_path = Path(tac.losses.__file__)  # type: ignore[arg-type]
    assert init_path.name == "__init__.py", (
        f"tac.losses must be a package, got {init_path}"
    )
    return init_path


def _ast_top_level_imported_names(init_path: Path) -> set[str]:
    """Names that AST-visible scanners (preflight Check 13) see at __init__ top level."""
    text = init_path.read_text()
    tree = ast.parse(text, filename=str(init_path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
    return names


def test_runtime_import_each_historical_symbol_resolves() -> None:
    """Vector (A) + the 47-caller runtime API contract.

    Every historically imported name must be reachable via ``getattr`` on the
    ``tac.losses`` module. This is the runtime symbol-table check.
    """
    import tac.losses as losses_pkg

    missing = sorted(
        name for name in HISTORICAL_REQUIRED_NAMES if not hasattr(losses_pkg, name)
    )
    assert not missing, (
        f"tac.losses is missing {len(missing)} runtime symbols: {missing}. "
        f"Add them to src/tac/losses/__init__.py."
    )


def test_init_py_uses_explicit_importfrom_not_star() -> None:
    """Vector (A): forbid bare ``from .core import *``.

    The AST-based preflight ``Check 13`` skips ``*`` aliases (preflight.py
    L5878). A package that loses its explicit ImportFrom list silently
    re-introduces all 47 dead-import violations.
    """
    text = _init_path().read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    pytest.fail(
                        f"tac/losses/__init__.py uses 'from .{node.module} import *' "
                        f"which is invisible to preflight Check 13. Replace with "
                        f"explicit 'from .{node.module} import (X, Y, Z, ...)'."
                    )


def test_ast_visible_names_cover_historical_requirements() -> None:
    """Vector (C) + preflight Check 13 contract.

    Every historically required name must appear as an ``ImportFrom`` alias at
    the top of ``__init__.py`` so AST-based scanners resolve it. This is what
    flipped Check 13 from 47 -> 0.
    """
    ast_names = _ast_top_level_imported_names(_init_path())
    missing = sorted(HISTORICAL_REQUIRED_NAMES - ast_names)
    assert not missing, (
        f"tac/losses/__init__.py is missing {len(missing)} explicit ImportFrom "
        f"aliases: {missing}. Add each to the 'from .core import (...)' "
        f"or 'from .cat_entropy_v2 import (...)' block."
    )


def test_all_dunder_lists_every_runtime_export() -> None:
    """Vector (B): __all__ must reflect the full re-export surface."""
    import tac.losses as losses_pkg

    declared_all = set(getattr(losses_pkg, "__all__", []))
    missing = sorted(HISTORICAL_REQUIRED_NAMES - declared_all)
    assert not missing, (
        f"tac.losses.__all__ is missing {len(missing)} historically required "
        f"names: {missing}."
    )


def test_each_reexport_is_actually_the_core_attribute() -> None:
    """Semantic-correctness: every re-export points at the right symbol.

    Contrarian's specific challenge: 'are the re-exports semantically correct?
    Will any 47 callers get different behavior because of subtle
    symbol-aliasing?' This test imports both ``tac.losses`` and ``tac.losses.core``
    and asserts each shared name is the same object (``is``, not just ``==``).
    """
    import tac.losses as losses_pkg
    from tac.losses import core as losses_core

    # cat_entropy_v2 lives in its own submodule, so we check it separately.
    cat_entropy_v2_names = {"CatEntropyV2Config", "cat_entropy_v2"}

    for name in HISTORICAL_REQUIRED_NAMES - cat_entropy_v2_names:
        pkg_attr = getattr(losses_pkg, name)
        core_attr = getattr(losses_core, name)
        assert pkg_attr is core_attr, (
            f"tac.losses.{name} ({pkg_attr!r}) is not the same object as "
            f"tac.losses.core.{name} ({core_attr!r}). Symbol-aliasing drift."
        )

    # ``cat_entropy_v2`` is BOTH a function on the package and a submodule
    # name, so use importlib to grab the submodule unambiguously.
    import importlib

    cat_entropy_v2_mod = importlib.import_module("tac.losses.cat_entropy_v2")
    for name in cat_entropy_v2_names:
        pkg_attr = getattr(losses_pkg, name)
        submod_attr = getattr(cat_entropy_v2_mod, name)
        assert pkg_attr is submod_attr, (
            f"tac.losses.{name} is not the same object as "
            f"tac.losses.cat_entropy_v2.{name}. Symbol-aliasing drift."
        )


def test_check_13_preflight_dead_resolvers_clean_for_losses_imports() -> None:
    """End-to-end: run the actual preflight scanner against a tiny tac.losses
    consumer fixture and confirm zero violations.
    """
    from tac.preflight import _scan_python_for_dead_imports

    # Inline fixture: a script that mimics the import shapes used by the 47
    # legacy callers. Written into a tempfile so the scanner sees a real
    # on-disk import-from node.
    import tempfile

    fixture_src = (
        "from tac.losses import scorer_forward_pair, _hwc_to_chw\n"
        "from tac.losses import kl_distill_segnet_only, parse_class_weights_csv\n"
        "from tac.losses import (\n"
        "    SEGMENTATION_SURROGATE_FISHER_RAO,\n"
        "    SEGMENTATION_SURROGATE_SINKHORN,\n"
        "    SEGMENTATION_SURROGATE_SOFT_COSINE,\n"
        "    scorer_loss_terms_btchw,\n"
        "    scorer_loss_terms_cached_btchw,\n"
        ")\n"
        "from tac.losses import CatEntropyV2Config, cat_entropy_v2\n"
        "from tac.losses import (\n"
        "    DEFAULT_SINKHORN_BLUR,\n"
        "    DEFAULT_SINKHORN_ITERS,\n"
        ")\n"
        "from tac.losses import (\n"
        "    eval_scorer_loss,\n"
        "    frequency_aware_loss,\n"
        "    focal_segnet_ste_loss,\n"
        "    posenet_embedding_loss,\n"
        "    scorer_loss,\n"
        "    scorer_loss_cached,\n"
        "    scorer_loss_cached_with_aux,\n"
        "    scorer_loss_pcgrad,\n"
        "    scorer_loss_with_aux,\n"
        "    segnet_surrogate_per_pixel,\n"
        "    segnet_uncertainty_weighted_loss,\n"
        "    uniward_quant_noise_loss,\n"
        ")\n"
    )

    repo_root = Path(__file__).resolve().parents[3]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=str(repo_root / "experiments")
    ) as f:
        f.write(fixture_src)
        fixture_path = Path(f.name)
    try:
        violations = _scan_python_for_dead_imports(fixture_path, repo_root)
    finally:
        fixture_path.unlink(missing_ok=True)

    assert not violations, (
        f"tac.losses fixture has {len(violations)} dead-import violations "
        f"after Wave 3 unblock fix:\n  "
        + "\n  ".join(violations)
    )
