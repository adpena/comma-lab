# SPDX-License-Identifier: MIT
"""Tests for Catalog #383 STRICT preflight gate
:func:`tac.preflight.check_mlx_primitives_route_through_canonical_helper`.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS: every test uses REAL synthetic
file fixtures + REAL AST parsing (not stub fixtures).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_mlx_primitives_route_through_canonical_helper,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class TestCheck383BasicBehavior:
    def test_live_repo_returns_list(self):
        """Live-repo regression guard: gate returns a list."""
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, verbose=False
        )
        assert isinstance(violations, list)

    def test_live_repo_warn_only_bounded(self):
        """Live-repo regression guard: violations bounded at landing.

        Per audit inventory 2026-05-30: gumbel_softmax_sample duplicate-
        impl detection identifies 2 substrate-side impls (DreamerV3 +
        Z8). Strict-flip planned after canonical extraction migration.
        """
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, verbose=False
        )
        assert len(violations) <= 10, (
            f"live count exceeded 10; got {len(violations)}; "
            "operator-routable backfill needed"
        )

    def test_strict_silent_on_clean_synthetic(self, tmp_path):
        """No substrate dir + no train_substrate files = no violations."""
        # Empty tmp_path (no src/tac/substrates) → no violations
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert violations == []

    def test_strict_raises_on_violation(self, tmp_path):
        """Synthetic substrate with primitive re-impl raises in strict."""
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx

            def gumbel_softmax_sample(logits, temperature=1.0):
                # Re-implements canonical primitive without waiver
                return logits
        """))
        with pytest.raises(
            PreflightError,
            match="check_mlx_primitives_route_through_canonical_helper",
        ):
            check_mlx_primitives_route_through_canonical_helper(
                strict=True, repo_root=tmp_path
            )


class TestCheck383CanonicalImportAccepts:
    def test_canonical_import_from_pr95_hnerv_mlx_accepted(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc

            def use_canonical():
                return pixel_shuffle_2x_nhwc
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        # No def of canonical primitive → no violation
        assert violations == []

    def test_canonical_import_with_def_skipped(self, tmp_path):
        """If file imports + defs a primitive, the import accepts the def."""
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            from tac.framework_agnostic import gumbel_softmax_sample as canonical_gumbel

            def gumbel_softmax_sample(logits, temperature=1.0):
                # This is local but imports canonical via alias
                return canonical_gumbel(logits, temperature=temperature)
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        # Import detected → routes through canonical → no violation
        assert violations == []

    def test_canonical_import_from_framework_agnostic_accepted(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            from tac.framework_agnostic.canonical_kernels import (
                gumbel_softmax_sample,
                rgb_to_yuv6,
            )
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert violations == []


class TestCheck383WaiverSemantics:
    def test_substantive_waiver_accepted(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx

            # MLX_PRIMITIVE_UNIQUE_BECAUSE_SUBSTRATE_OPTIMAL_FORK:substantive rationale for fork
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert violations == []

    def test_placeholder_rationale_rejected(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx

            # MLX_PRIMITIVE_UNIQUE_BECAUSE_FORK:<rationale>
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert len(violations) == 1

    def test_short_rationale_rejected(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx

            # MLX_PRIMITIVE_UNIQUE_BECAUSE_FORK:abc
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert len(violations) == 1

    def test_waiver_on_preceding_line_accepted(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx

            # MLX_PRIMITIVE_UNIQUE_BECAUSE_EMPIRICAL_FALSIFIED:canonical drift exceeded 5e-3 on this substrate paired smoke
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert violations == []


class TestCheck383ScopeAndExemptions:
    def test_non_substrate_path_skipped(self, tmp_path):
        # File outside substrates/ is out of scope
        sub = tmp_path / "src" / "tac" / "experimental"
        sub.mkdir(parents=True)
        (sub / "x.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert violations == []

    def test_file_without_mlx_import_skipped(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import numpy as np
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert violations == []

    def test_test_files_exempt(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "test_x.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert violations == []

    def test_canonical_source_dir_exempt(self, tmp_path):
        """Files in canonical source dirs DEFINE primitives, not duplicate them."""
        sub = tmp_path / "src" / "tac" / "local_acceleration"
        sub.mkdir(parents=True)
        (sub / "x.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        # local_acceleration is exempt (it's where canonical primitives live)
        assert violations == []

    def test_experiments_results_exempt(self, tmp_path):
        sub = (
            tmp_path
            / "experiments"
            / "results"
            / "old_run"
            / "substrates"
            / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        # experiments/results/ is exempt per artifact lifecycle Catalog #113
        assert violations == []


class TestCheck383MultiViolationAggregation:
    def test_multiple_violations_aggregated(self, tmp_path):
        sub1 = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_a"
        )
        sub2 = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_b"
        )
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)
        (sub1 / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        (sub2 / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            def pixel_shuffle_2x_nhwc(x, upscale_factor=2):
                return x
        """))
        violations = check_mlx_primitives_route_through_canonical_helper(
            strict=False, repo_root=tmp_path
        )
        assert len(violations) == 2


class TestCheck383StrictMode:
    def test_strict_silent_on_clean_repo(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc
            def use():
                return pixel_shuffle_2x_nhwc
        """))
        # No primitives defined locally → no violations even in strict
        result = check_mlx_primitives_route_through_canonical_helper(
            strict=True, repo_root=tmp_path
        )
        assert result == []

    def test_strict_includes_383_in_error_message(self, tmp_path):
        sub = (
            tmp_path / "src" / "tac" / "substrates" / "test_synth_substrate"
        )
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        with pytest.raises(PreflightError) as excinfo:
            check_mlx_primitives_route_through_canonical_helper(
                strict=True, repo_root=tmp_path
            )
        assert "check_mlx_primitives_route_through_canonical_helper" in str(
            excinfo.value
        )


class TestCheck383CallableViaGlobals:
    """Catalog #185 sister regression guard: STRICT-flipped gates must
    be callable via tac.preflight globals."""

    def test_gate_callable_via_module(self):
        from tac import preflight

        assert callable(
            getattr(
                preflight,
                "check_mlx_primitives_route_through_canonical_helper",
            )
        )

    def test_orchestrator_wires_warn_only(self):
        """Catalog #176 sister regression guard: gate wired into preflight_all."""
        import inspect
        from tac import preflight

        src = inspect.getsource(preflight.preflight_all)
        assert "check_mlx_primitives_route_through_canonical_helper" in src
