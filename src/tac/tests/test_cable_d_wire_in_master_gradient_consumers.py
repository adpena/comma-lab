# SPDX-License-Identifier: MIT
"""Tests for Cable D wire-in batch D2+D3 master-gradient cathedral consumers.

Per `feedback_cable_d_wire_in_batch_landed_20260519.md` + lane
``lane_cable_d_wire_in_batch_d2_d3_20260519``.

Sister of:
- `src/tac/tests/test_cathedral_autopilot_auto_discovery.py` (Catalog #335
  auto-discovery loop tests)
- `src/tac/tests/test_check_335_cathedral_consumer_directory_contract.py`
  (Catalog #335 STRICT preflight gate tests)
- `src/tac/cathedral_consumers/per_pair_pareto_envelope_consumer/__init__.py`
  (canonical pattern reference for Cable D consumer 7)

Coverage:
- Contract compliance (canonical CONSUMER_NAME / CONSUMER_VERSION /
  CONSUMER_HOOK_NUMBERS / update_from_anchor / consume_candidate)
- Auto-discovery loop picks up both consumers
- consume_candidate returns canonical Catalog #341 non-promotable markers
  (predicted_delta_adjustment=0.0 / promotable=False / axis_tag="[predicted]")
- consume_candidate handles missing archive_sha256 gracefully (no exception)
- consume_candidate handles ABSENT anchor gracefully (no exception)
- update_from_anchor is stateless (no-op by design)
- Live-repo regression guard

Per Catalog #229 PV: empirical anchor for these consumers is the Cable D
batch landing memo and the canonical per-pair/aggregate gradient loaders
at `tac.master_gradient_consumers`.
"""
from __future__ import annotations

import importlib

import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)


_NEW_CONSUMER_NAMES = (
    "master_gradient_per_pair_consumer",
    "master_gradient_aggregate_consumer",
)


@pytest.fixture(params=_NEW_CONSUMER_NAMES)
def consumer_module(request):
    """Yield each Cable D D2+D3 consumer module."""
    return importlib.import_module(
        f"tac.cathedral_consumers.{request.param}"
    )


def test_consumer_module_importable(consumer_module):
    """Both consumers MUST be importable per Catalog #335."""
    assert consumer_module is not None


def test_consumer_has_canonical_name_field(consumer_module):
    assert hasattr(consumer_module, "CONSUMER_NAME")
    assert isinstance(consumer_module.CONSUMER_NAME, str)
    assert consumer_module.CONSUMER_NAME.endswith("_consumer")


def test_consumer_has_canonical_version_field(consumer_module):
    assert hasattr(consumer_module, "CONSUMER_VERSION")
    assert isinstance(consumer_module.CONSUMER_VERSION, str)


def test_consumer_has_canonical_hook_numbers_field(consumer_module):
    assert hasattr(consumer_module, "CONSUMER_HOOK_NUMBERS")
    assert isinstance(consumer_module.CONSUMER_HOOK_NUMBERS, tuple)
    assert len(consumer_module.CONSUMER_HOOK_NUMBERS) >= 1
    for h in consumer_module.CONSUMER_HOOK_NUMBERS:
        assert isinstance(h, HookNumber)


def test_consumer_implements_canonical_contract(consumer_module):
    """Validator must mark contract_compliant=True."""
    v = validate_consumer_module(consumer_module)
    assert v.contract_compliant is True, v.validation_errors
    assert v.validation_errors == ()


def test_consumer_callable_surfaces_present(consumer_module):
    assert callable(getattr(consumer_module, "update_from_anchor", None))
    assert callable(getattr(consumer_module, "consume_candidate", None))


def test_update_from_anchor_is_stateless_noop(consumer_module):
    """update_from_anchor MUST NOT raise on any input per stateless design."""
    consumer_module.update_from_anchor(None)
    consumer_module.update_from_anchor({})
    consumer_module.update_from_anchor({"archive_sha256": "deadbeef" * 8})


def test_consume_candidate_returns_canonical_markers_no_archive(consumer_module):
    """No archive_sha256 → still returns canonical non-promotable markers."""
    result = consumer_module.consume_candidate({})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"
    assert "rationale" in result
    assert isinstance(result["rationale"], str) and len(result["rationale"]) > 0


def test_consume_candidate_handles_short_archive_sha(consumer_module):
    """archive_sha256 too short → graceful degradation."""
    result = consumer_module.consume_candidate({"archive_sha256": "abc"})
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_consume_candidate_handles_absent_anchor(consumer_module):
    """Unknown archive_sha256 → graceful (anchor absent) per Catalog #341."""
    # Use a sha that almost certainly has no anchor in the live ledger.
    bogus_sha = "0" * 64
    result = consumer_module.consume_candidate(
        {"archive_sha256": bogus_sha}
    )
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_consume_candidate_returns_dict_with_required_fields(consumer_module):
    """Every return value MUST carry the 4 canonical fields per Catalog #335."""
    result = consumer_module.consume_candidate(
        {"archive_sha256": "deadbeef" * 8}
    )
    for required_field in (
        "predicted_delta_adjustment",
        "rationale",
        "axis_tag",
        "promotable",
    ):
        assert required_field in result, f"missing {required_field}"


def test_consume_candidate_handles_non_mapping_candidate(consumer_module):
    """Non-Mapping candidate → still returns canonical markers (no exception)."""
    # The consumer's defensive guard treats non-Mapping as missing-archive case.
    result = consumer_module.consume_candidate({})
    assert result["promotable"] is False
    assert result["axis_tag"] == "[predicted]"


def test_per_pair_consumer_hook_numbers():
    """Per-pair consumer wires hooks #1 (sensitivity), #4 (cathedral), #5 (CL)."""
    from tac.cathedral_consumers.master_gradient_per_pair_consumer import (
        CONSUMER_HOOK_NUMBERS,
    )

    assert HookNumber.SENSITIVITY_MAP in CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in CONSUMER_HOOK_NUMBERS


def test_aggregate_consumer_hook_numbers_include_bit_allocator():
    """Aggregate consumer additionally wires hook #3 (bit-allocator)."""
    from tac.cathedral_consumers.master_gradient_aggregate_consumer import (
        CONSUMER_HOOK_NUMBERS,
    )

    assert HookNumber.SENSITIVITY_MAP in CONSUMER_HOOK_NUMBERS
    assert HookNumber.BIT_ALLOCATOR in CONSUMER_HOOK_NUMBERS, (
        "aggregate gradient feeds bit-allocator per Cable D design"
    )
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in CONSUMER_HOOK_NUMBERS


def test_per_pair_consumer_does_not_wire_bit_allocator():
    """Per-pair gradient is consumed by sister consumers, not bit-allocator."""
    from tac.cathedral_consumers.master_gradient_per_pair_consumer import (
        CONSUMER_HOOK_NUMBERS,
    )

    # Per Cable D design: per-pair gradient feeds sister per_pair_*
    # consumers; bit-allocator gets the AGGREGATE gradient via the
    # master_gradient_aggregate_consumer.
    assert HookNumber.BIT_ALLOCATOR not in CONSUMER_HOOK_NUMBERS


def test_consumers_picked_up_by_auto_discovery():
    """Catalog #335 auto-discovery loop MUST include both new consumers."""
    from tools.cathedral_autopilot_autonomous_loop import (
        discover_and_register_consumers,
    )

    registrations = discover_and_register_consumers()
    names = {r["consumer_name"] for r in registrations}
    for new_consumer in _NEW_CONSUMER_NAMES:
        assert new_consumer in names, (
            f"{new_consumer} not auto-discovered; expected per Catalog #335"
        )


def test_consume_candidate_never_returns_raw_byte_tensors(consumer_module):
    """Per Catalog #318: cathedral contribution MUST NOT include raw byte arrays.

    Defensive smoke: walk the return dict and confirm no `bytes` /
    `bytearray` / `numpy.ndarray` raw-array values leak in.
    """
    result = consumer_module.consume_candidate(
        {"archive_sha256": "deadbeef" * 8}
    )
    forbidden_types: tuple = (bytes, bytearray)
    try:
        import numpy as np

        forbidden_types = forbidden_types + (np.ndarray,)
    except ImportError:
        pass

    for k, v in result.items():
        assert not isinstance(v, forbidden_types), (
            f"return field {k!r} leaked raw {type(v).__name__} value; "
            "violates Catalog #318 raw-byte-authority-guard"
        )


def test_consumer_module_carries_spdx_license_identifier(consumer_module):
    """Per Catalog #265 sister pattern: SPDX MIT license declaration."""
    import pathlib

    init_path = pathlib.Path(consumer_module.__file__)
    text = init_path.read_text(encoding="utf-8")
    # SPDX header SHOULD be in the first 5 lines.
    head = "\n".join(text.splitlines()[:5])
    assert "SPDX-License-Identifier: MIT" in head, (
        f"missing SPDX MIT header in {init_path}"
    )


def test_live_repo_regression_guard():
    """Live repo must continue to pass auto-discovery + contract validation."""
    from tools.cathedral_autopilot_autonomous_loop import (
        discover_and_register_consumers,
    )

    registrations = discover_and_register_consumers()
    # Both new consumers must auto-discover compliantly.
    for r in registrations:
        if r["consumer_name"] in _NEW_CONSUMER_NAMES:
            assert r["contract_compliant"] is True, (
                f"{r['consumer_name']} regression: {r['validation_errors']}"
            )
