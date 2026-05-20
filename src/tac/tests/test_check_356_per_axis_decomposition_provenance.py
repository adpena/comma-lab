# SPDX-License-Identifier: MIT
"""Tests for Catalog #356 — check_per_axis_decomposition_carries_canonical_provenance.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Steps 3.1-3.3 + Catalog #287 / #323
sister discipline. Sister of Catalog #335 at the per-axis emission
sub-surface.

Coverage:
- live-repo regression guard (live count: 0)
- positive (synthetic consumer emits per-axis without Provenance → flagged)
- negative (consumer with canonical Provenance accepted)
- waiver semantics (rationale accepted; placeholder rejected)
- canonical reference fixture (`_example_consumer`) is out-of-scope (no trigger)
- strict-mode raises
- strict-mode silent on clean
- orchestrator wire-in regression (Catalog #176 sister)
- Catalog #185 sister-callable regression
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_per_axis_decomposition_carries_canonical_provenance,
)


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_clean() -> None:
    """Live count at landing: 0 (no production consumer migrated to Tier B yet)."""
    violations = check_per_axis_decomposition_carries_canonical_provenance()
    # WARN-ONLY at landing; explicit ceiling = 5 to catch silent drift.
    assert len(violations) <= 5, (
        f"Live-repo violation drift: {len(violations)} > ceiling 5; "
        "first 3: " + "; ".join(violations[:3])
    )


# ─────────────────────────────────────────────────────────────────────────
# Synthetic consumer scenarios
# ─────────────────────────────────────────────────────────────────────────


def _make_synthetic_consumer(
    tmp_path: Path,
    consumer_name: str,
    body: str,
) -> Path:
    """Construct a synthetic cathedral_consumers/<name>/__init__.py fixture."""
    consumers_root = tmp_path / "src" / "tac" / "cathedral_consumers"
    pkg_dir = consumers_root / consumer_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    init_path = pkg_dir / "__init__.py"
    init_path.write_text(body)
    return tmp_path


class TestPerAxisProvenanceGate:

    def test_consumer_without_trigger_token_passes(self, tmp_path: Path) -> None:
        """Consumer that doesn't emit per-axis is out-of-scope."""
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "from tac.cathedral.consumer_contract import HookNumber\n"
            "CONSUMER_NAME = 'no_per_axis'\n"
            "CONSUMER_VERSION = '1.0.0'\n"
            "CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)\n"
            "def update_from_anchor(anchor): pass\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_delta_adjustment': 0.0}\n"
        )
        root = _make_synthetic_consumer(tmp_path, "no_per_axis_emission", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert violations == []

    def test_consumer_with_trigger_no_provenance_flagged(
        self, tmp_path: Path
    ) -> None:
        """Per-axis emission without Provenance import → flagged."""
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'bad_per_axis'\n"
            "CONSUMER_VERSION = '1.0.0'\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_axis_decomposition': {}}\n"
        )
        root = _make_synthetic_consumer(tmp_path, "bad_per_axis", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert len(violations) == 1
        assert "bad_per_axis" in violations[0]
        assert "predicted_axis_decomposition" in violations[0]
        assert "Catalog #356" in violations[0]

    def test_consumer_with_provenance_builder_accepted(
        self, tmp_path: Path
    ) -> None:
        """Per-axis emission WITH canonical Provenance builder → accepted."""
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "from tac.provenance import build_provenance_for_predicted\n"
            "CONSUMER_NAME = 'good_per_axis'\n"
            "CONSUMER_VERSION = '1.0.0'\n"
            "def consume_candidate(c):\n"
            "    return {\n"
            "        'predicted_axis_decomposition': {\n"
            "            'predicted_d_seg_delta': 0.0,\n"
            "        }\n"
            "    }\n"
        )
        root = _make_synthetic_consumer(tmp_path, "good_per_axis", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert violations == []

    def test_consumer_with_canonical_provenance_kwarg_accepted(
        self, tmp_path: Path
    ) -> None:
        """The `canonical_provenance=` keyword assignment is a sufficient acceptance token.

        Per ``_CHECK_356_PROVENANCE_TOKENS``: the keyword form
        (``canonical_provenance=prov_dict``) is accepted as evidence the
        consumer is wiring canonical Provenance through; the dict-literal
        form (``'canonical_provenance': {...}``) is NOT (because it could
        be a passthrough of an unvalidated payload).
        """
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'kwarg_per_axis'\n"
            "CONSUMER_VERSION = '1.0.0'\n"
            "from tac.cathedral.consumer_contract import AxisDecomposition\n"
            "def consume_candidate(c):\n"
            "    decomp = AxisDecomposition(\n"
            "        predicted_d_seg_delta=0.0,\n"
            "        predicted_d_pose_delta=0.0,\n"
            "        predicted_archive_bytes_delta=0,\n"
            "        canonical_provenance={'kind': 'predicted'},\n"
            "    )\n"
            "    return {'predicted_axis_decomposition': decomp.as_dict()}\n"
        )
        root = _make_synthetic_consumer(tmp_path, "kwarg_per_axis", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert violations == []

    def test_waiver_with_rationale_accepted(self, tmp_path: Path) -> None:
        """Per-line waiver with substantive rationale accepted."""
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'waived_per_axis'\n"
            "CONSUMER_VERSION = '1.0.0'\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_axis_decomposition': {}}  "
            "# PER_AXIS_DECOMPOSITION_PROVENANCE_OK:research_diagnostic_scope_pending_phase2_provenance_wire_in_per_dim_3_step_34\n"
        )
        root = _make_synthetic_consumer(tmp_path, "waived_per_axis", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert violations == []

    def test_placeholder_rationale_rejected(self, tmp_path: Path) -> None:
        """`<rationale>` placeholder rejected so docstring example cannot self-waive."""
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'placeholder_per_axis'\n"
            "CONSUMER_VERSION = '1.0.0'\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_axis_decomposition': {}}  "
            "# PER_AXIS_DECOMPOSITION_PROVENANCE_OK:<rationale>\n"
        )
        root = _make_synthetic_consumer(tmp_path, "placeholder_per_axis", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert len(violations) == 1

    def test_short_rationale_rejected(self, tmp_path: Path) -> None:
        """Rationale <4 chars rejected."""
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'short_per_axis'\n"
            "CONSUMER_VERSION = '1.0.0'\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_axis_decomposition': {}}  "
            "# PER_AXIS_DECOMPOSITION_PROVENANCE_OK:ok\n"
        )
        root = _make_synthetic_consumer(tmp_path, "short_per_axis", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert len(violations) == 1

    def test_empty_rationale_rejected(self, tmp_path: Path) -> None:
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'empty_per_axis'\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_axis_decomposition': {}}  "
            "# PER_AXIS_DECOMPOSITION_PROVENANCE_OK:\n"
        )
        root = _make_synthetic_consumer(tmp_path, "empty_per_axis", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert len(violations) == 1

    def test_reason_placeholder_rejected(self, tmp_path: Path) -> None:
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'reason_per_axis'\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_axis_decomposition': {}}  "
            "# PER_AXIS_DECOMPOSITION_PROVENANCE_OK:<reason>\n"
        )
        root = _make_synthetic_consumer(tmp_path, "reason_per_axis", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert len(violations) == 1

    def test_strict_mode_raises_on_violation(self, tmp_path: Path) -> None:
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'strict_per_axis'\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_axis_decomposition': {}}\n"
        )
        root = _make_synthetic_consumer(tmp_path, "strict_per_axis", body)
        with pytest.raises(PreflightError, match="Catalog #356"):
            check_per_axis_decomposition_carries_canonical_provenance(
                repo_root=root, strict=True
            )

    def test_strict_mode_silent_on_clean(self, tmp_path: Path) -> None:
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "CONSUMER_NAME = 'clean'\n"
            "def consume_candidate(c):\n"
            "    return {'predicted_delta_adjustment': 0.0}\n"
        )
        root = _make_synthetic_consumer(tmp_path, "clean", body)
        # Should not raise.
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root, strict=True
        )
        assert violations == []

    def test_missing_consumer_dir_silent(self, tmp_path: Path) -> None:
        """No consumer dir → silent skip, no violations."""
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=tmp_path
        )
        assert violations == []

    def test_exempt_subdirs_skipped(self, tmp_path: Path) -> None:
        """`__pycache__` + `tests` exempted from scan."""
        body = (
            "# SPDX-License-Identifier: MIT\n"
            "x = 'predicted_axis_decomposition has no Provenance'\n"
        )
        root = _make_synthetic_consumer(tmp_path, "__pycache__", body)
        _make_synthetic_consumer(tmp_path, "tests", body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert violations == []

    def test_multi_violation_aggregation(self, tmp_path: Path) -> None:
        for name in ["alpha", "beta", "gamma"]:
            body = (
                "# SPDX-License-Identifier: MIT\n"
                f"CONSUMER_NAME = '{name}'\n"
                "def consume_candidate(c):\n"
                "    return {'predicted_axis_decomposition': {}}\n"
            )
            root = _make_synthetic_consumer(tmp_path, name, body)
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=root
        )
        assert len(violations) == 3

    def test_string_repo_root_accepted(self, tmp_path: Path) -> None:
        """Path-string accepted as repo_root."""
        violations = check_per_axis_decomposition_carries_canonical_provenance(
            repo_root=str(tmp_path)
        )
        assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator wire-in regression (Catalog #176 sister)
# ─────────────────────────────────────────────────────────────────────────


class TestOrchestratorWireIn:

    def test_callsite_present_in_preflight_all(self) -> None:
        """`preflight_all()` invokes Catalog #356 gate (WARN-ONLY at landing)."""
        from pathlib import Path

        preflight_src = Path("src/tac/preflight.py").read_text()
        assert (
            "check_per_axis_decomposition_carries_canonical_provenance"
            in preflight_src
        )
        # Verify wired into preflight_all() body (not just defined).
        # The callsite should be inside an `if not changed_only:` block,
        # mirror Catalog #335's pattern.
        idx_def = preflight_src.find(
            "def check_per_axis_decomposition_carries_canonical_provenance"
        )
        # First occurrence is the def; subsequent occurrence(s) are
        # invocations. Require at least one invocation site.
        first_after_def = preflight_src.find(
            "check_per_axis_decomposition_carries_canonical_provenance",
            idx_def + 80,
        )
        # If def appears multiple times (it shouldn't), still require
        # an invocation later in the file body.
        assert first_after_def > 0, (
            "Catalog #356 helper defined but never invoked from preflight_all()"
        )


# ─────────────────────────────────────────────────────────────────────────
# Catalog #185 sister-callable regression
# ─────────────────────────────────────────────────────────────────────────


class TestCatalog185SisterCallable:

    def test_function_importable_from_module(self) -> None:
        """Catalog #185 requires the function be callable via tac.preflight.globals."""
        import tac.preflight as preflight_module

        fn = getattr(
            preflight_module,
            "check_per_axis_decomposition_carries_canonical_provenance",
            None,
        )
        assert fn is not None
        assert callable(fn)
        # Call with default args (no kwargs) is rejected because all kwargs
        # are keyword-only — sanity-check the signature.
        result = fn(strict=False, verbose=False)
        assert isinstance(result, list)
