# SPDX-License-Identifier: MIT
"""Z8 Wave 10/11 (2026-05-29) unimix-propagation-from-sister-DreamerV3 tests.

[verified-against: src/tac/substrates/dreamer_v3_rssm/module.py (sister canonical)
+ Hafner et al. 2023 arXiv:2301.04104 §3 + Wave 3 audit memo
.omx/research/wave_3_dreamerv3_rssm_math_fidelity_audit_landed_20260529.md]

Wave 10/11 audit found Z8's local `gumbel_softmax_sample` reimplemented the
categorical sampler WITHOUT threading `unimix_alpha`, despite the docstring
claiming "Sister A=DreamerV3 canonical implementation reused per Catalog #290
ADOPT_CANONICAL_BECAUSE_SERVES". The Wave 3 math-fidelity audit on sister
DreamerV3 added the 1% unimix mixture per Hafner 2023 §3 "Robustness" — but
the fix did NOT propagate to Z8 because Z8's path was a duplicate.

The Wave 10/11 fix replaced Z8's local duplicate with a thin delegation to
sister `tac.substrates.dreamer_v3_rssm` so future sister fixes propagate
structurally. THIS test pins the delegation as a regression guard.

Classification per Catalog #307: IMPLEMENTATION-LEVEL gap (the docstring
claimed canonical reuse but the implementation diverged); the categorical-
posterior PARADIGM stays INTACT (the local Z8 implementation was mathematically
correct except for the unimix omission).

Per Catalog #303 cargo-cult audit: the Z8 local helper carried the
"Sister A=DreamerV3 canonical implementation reused" docstring claim as a
HARD-EARNED inheritance from the design memo, but the operational delegation
was CARGO-CULTED at L0 scaffold time (the duplicate was easier to type than
the delegation). The Wave 10/11 fix unwinds the cargo-cult.
"""

from __future__ import annotations

import inspect

import pytest


def test_z8_gumbel_softmax_sample_threads_unimix_alpha_default() -> None:
    """Z8 gumbel_softmax_sample carries unimix_alpha default 0.01 per Wave 3.

    Regression guard for Wave 10/11 (2026-05-29) fix. Pre-fix, Z8's local
    implementation lacked the `unimix_alpha` parameter entirely; the
    delegation pattern threads it through to sister canonical.
    """
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        gumbel_softmax_sample,
    )

    sig = inspect.signature(gumbel_softmax_sample)
    assert "unimix_alpha" in sig.parameters
    assert sig.parameters["unimix_alpha"].default == 0.01


def test_z8_apply_unimix_to_logits_exists_and_threads_default() -> None:
    """Z8 re-exports apply_unimix_to_logits per Wave 10/11 surface canonicalization."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        apply_unimix_to_logits,
    )

    sig = inspect.signature(apply_unimix_to_logits)
    assert "unimix_alpha" in sig.parameters
    assert sig.parameters["unimix_alpha"].default == 0.01


def test_z8_helpers_delegate_to_sister_dreamerv3_canonical_source() -> None:
    """Z8 surface helpers must invoke sister canonical helpers (no local duplicate).

    The Wave 10/11 fix replaces Z8's local categorical-sampler duplicate with
    a thin pass-through to `tac.substrates.dreamer_v3_rssm`. This test inspects
    the source text to verify the delegation pattern is present.
    """
    import tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer as z8_mod

    src = inspect.getsource(z8_mod.gumbel_softmax_sample)
    assert "from tac.substrates.dreamer_v3_rssm import" in src
    assert "gumbel_softmax_sample as _sister_gumbel_softmax_sample" in src

    src_un = inspect.getsource(z8_mod.apply_unimix_to_logits)
    assert "from tac.substrates.dreamer_v3_rssm import" in src_un
    assert "apply_unimix_to_logits as _sister_apply_unimix_to_logits" in src_un


def test_z8_local_duplicate_implementation_removed() -> None:
    """Z8 mlx_renderer must NOT carry a local Gumbel-perturbation implementation.

    Regression guard: the duplicate path that prompted Wave 10/11 closure
    referenced `mx.random.uniform` + `-mx.log(-mx.log(uniform))` (the Gumbel
    perturbation) directly inside `gumbel_softmax_sample`. Post-fix the body
    delegates and contains NEITHER pattern at the function level (the
    canonical helper at sister contains them inside the delegated call).
    """
    import tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer as z8_mod

    src = inspect.getsource(z8_mod.gumbel_softmax_sample)
    # Post-delegation, the function body is small (~5 lines under docstring)
    # and contains no Gumbel math. The canonical sister has the math.
    assert "mx.random.uniform" not in src, (
        "Z8 gumbel_softmax_sample still carries local Gumbel-perturbation code; "
        "Wave 10/11 fix REQUIRES delegation to sister canonical helper to "
        "structurally propagate Wave 3 unimix fix."
    )
    assert "mx.log(-mx.log(uniform))" not in src


@pytest.mark.skipif(
    pytest.importorskip("mlx", reason="MLX runtime required for end-to-end test")
    is None,
    reason="MLX runtime required",
)
def test_z8_gumbel_softmax_unimix_propagates_to_runtime() -> None:
    """End-to-end: Z8 delegation produces same output as sister direct call.

    MLX-LOCAL macOS-CPU advisory per CLAUDE.md "MLX portable-local-substrate
    authority" Catalog #192 — NEVER promotable, but verifies the canonical
    delegation produces bit-identical output between Z8 surface + sister
    canonical surface.
    """
    import mlx.core as mx

    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        gumbel_softmax_sample as z8_gss,
    )
    from tac.substrates.dreamer_v3_rssm import (
        gumbel_softmax_sample as sister_gss,
    )

    logits = mx.array([[1.0, 0.0, -1.0, 0.5]] * 4)
    key = mx.random.key(42)

    z8_soft, z8_idx = z8_gss(logits, temperature=1.0, unimix_alpha=0.01, key=key)
    sister_soft, sister_idx = sister_gss(
        logits, temperature=1.0, unimix_alpha=0.01, key=key
    )

    assert mx.array_equal(z8_idx, sister_idx).item()
    # Soft samples may have tiny float drift across two separate MLX evaluations;
    # require equality of the canonical-key-seeded outputs.
    diff = mx.abs(z8_soft - sister_soft).max().item()
    assert diff < 1e-6, f"Z8 vs sister soft sample drift {diff} > 1e-6"
