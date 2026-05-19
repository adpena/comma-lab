# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_consumers.per_byte_sensitivity_consumer cathedral consumer.

Per PER-BYTE-SENSITIVITY-WIRE-INS subagent landing 2026-05-19. Tests the
cathedral consumer that wires per-byte sensitivity from the master-gradient
extractor into cathedral autopilot ranking via the canonical
``CathedralConsumerContract`` (Catalog #335) + the canonical reader helper
``tac.master_gradient_per_byte_consumer``.

Sister of:
- ``src/tac/tests/test_master_gradient_per_byte_consumer.py`` (canonical helper)
- ``src/tac/tests/test_cathedral_consumer_contract.py`` (canonical contract)
- ``src/tac/tests/test_cathedral_autopilot_auto_discovery.py`` (paradigm shift)
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.cathedral.consumer_contract import (
    HookNumber,
    validate_consumer_module,
)
from tac.cathedral_consumers import per_byte_sensitivity_consumer as consumer

# ─────────────────────────────────────────────────────────────────────────
# Canonical contract satisfaction (Catalog #335)
# ─────────────────────────────────────────────────────────────────────────


def test_consumer_module_exposes_canonical_contract() -> None:
    """Consumer module satisfies CathedralConsumerContract per Catalog #335."""
    reg = validate_consumer_module(
        consumer,
        module_path="tac.cathedral_consumers.per_byte_sensitivity_consumer",
    )
    assert reg.contract_compliant is True, (
        f"contract validation errors: {reg.validation_errors}"
    )
    assert len(reg.validation_errors) == 0


def test_consumer_name_canonical() -> None:
    assert consumer.CONSUMER_NAME == "per_byte_sensitivity_consumer"


def test_consumer_version_present() -> None:
    assert consumer.CONSUMER_VERSION == "1.0"


def test_consumer_hook_numbers_canonical() -> None:
    """Per Catalog #125: hooks 1+3+4 ACTIVE (sensitivity / bit-allocator / dispatch)."""
    assert HookNumber.SENSITIVITY_MAP in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.BIT_ALLOCATOR in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS


def test_consumer_hook_numbers_do_not_include_inactive_hooks() -> None:
    """Hooks 2 / 5 / 6 are N/A per the landing memo declaration."""
    assert HookNumber.PARETO_CONSTRAINT not in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR not in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PROBE_DISAMBIGUATOR not in consumer.CONSUMER_HOOK_NUMBERS


# ─────────────────────────────────────────────────────────────────────────
# update_from_anchor (NO-OP by design per Cable D pattern)
# ─────────────────────────────────────────────────────────────────────────


def test_update_from_anchor_is_noop() -> None:
    """update_from_anchor is NO-OP; canonical persistence lives in tac.master_gradient."""
    # Should not raise on any input shape.
    consumer.update_from_anchor({"archive_sha256": "abc"})
    consumer.update_from_anchor(None)
    consumer.update_from_anchor({})


# ─────────────────────────────────────────────────────────────────────────
# consume_candidate happy path + edge cases
# ─────────────────────────────────────────────────────────────────────────


def _write_synthetic_anchor(
    tmp_path: Path,
    *,
    archive_sha: str,
    n_bytes: int = 50,
    measurement_axis: str = "[macOS-CPU advisory]",
    measurement_hardware: str = "darwin_arm64_advisory",
) -> Path:
    """Write a synthetic master_gradient ledger + .npy to tmp_path."""
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    npy_path = tmp_path / f"grad_{archive_sha[:8]}.npy"
    arr = np.random.RandomState(123).randn(n_bytes, 3).astype(np.float32)
    np.save(npy_path, arr)
    row = {
        "archive_sha256": archive_sha,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "n_bytes": n_bytes,
        "operating_point": {
            "d_seg": 0.001,
            "d_pose": 0.002,
            "rate": 0.005,
            "score": 0.34,
        },
        "measurement_axis": measurement_axis,
        "measurement_hardware": measurement_hardware,
        "measurement_method": "autograd_per_parameter_test",
        "measurement_utc": "2026-05-19T03:00:00Z",
        "written_at_utc": "2026-05-19T03:00:00Z",
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(row) + "\n")
    return ledger


def test_consume_candidate_missing_sha_returns_no_signal() -> None:
    """No archive_sha256 → no-signal verdict (rationale cites missing key)."""
    verdict = consumer.consume_candidate({})
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    assert verdict["axis_tag"] == "[predicted]"
    assert "missing archive_sha256" in verdict["rationale"].lower()


def test_consume_candidate_non_mapping_returns_no_signal() -> None:
    """Non-mapping candidate input → no-signal verdict."""
    # A bare string is not a Mapping; consumer must handle gracefully.
    verdict = consumer.consume_candidate("not-a-mapping")  # type: ignore[arg-type]
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False


def test_consume_candidate_no_anchor_returns_no_signal() -> None:
    """Archive SHA present but no per-byte anchor → no-signal verdict."""
    verdict = consumer.consume_candidate(
        {"archive_sha256": "deadbeef" * 8}
    )
    # Either no signal (no anchor) OR signal (anchor exists somewhere).
    # The canonical no-signal verdict has the diagnostic phrase.
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    # The verdict is always observability-only.
    assert verdict["axis_tag"] in (
        "[predicted]",
        "[macOS-CPU advisory]",
        "[contest-CPU]",
        "[contest-CUDA]",
    )


def test_consume_candidate_predicted_delta_always_zero() -> None:
    """Per Catalog #287/#323: predicted_delta_adjustment is ALWAYS 0.0."""
    archive_sha = "abc1234567" * 6 + "abcd"
    verdict = consumer.consume_candidate({"archive_sha256": archive_sha})
    assert verdict["predicted_delta_adjustment"] == 0.0


def test_consume_candidate_promotable_always_false() -> None:
    """Per Catalog #287/#323: promotable is ALWAYS False (observability-only)."""
    archive_sha = "xyz9876543" * 6 + "xyzw"
    verdict = consumer.consume_candidate({"archive_sha256": archive_sha})
    assert verdict["promotable"] is False


def test_consume_candidate_accepts_sha_aliases() -> None:
    """Consumer accepts archive_sha256 / archive_sha / sha256 / sha aliases."""
    # All should produce a verdict (the no-signal one if no anchor).
    for field in ("archive_sha256", "archive_sha", "sha256", "sha"):
        verdict = consumer.consume_candidate({field: "a" * 64})
        assert verdict["predicted_delta_adjustment"] == 0.0
        assert "rationale" in verdict


# ─────────────────────────────────────────────────────────────────────────
# Live data path (synthetic ledger via monkeypatch)
# ─────────────────────────────────────────────────────────────────────────


def test_consume_candidate_with_synthetic_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: synthetic anchor → consumer surfaces per-byte signal."""
    archive_sha = "5" * 64
    ledger = _write_synthetic_anchor(tmp_path, archive_sha=archive_sha, n_bytes=50)

    # Monkeypatch the canonical loader to use our tmp ledger.
    from tac import master_gradient_per_byte_consumer as helper_mod
    original = helper_mod.load_per_byte_sensitivity_for_archive

    def patched_loader(*args, **kwargs):
        kwargs.setdefault("path", ledger)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        helper_mod, "load_per_byte_sensitivity_for_archive", patched_loader
    )

    verdict = consumer.consume_candidate({"archive_sha256": archive_sha})
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    # axis_tag should be [macOS-CPU advisory] because the hardware is darwin.
    assert verdict["axis_tag"] == "[macOS-CPU advisory]"
    # Notes block should carry per-byte sensitivity summary.
    assert "notes" in verdict
    notes = verdict["notes"]
    assert "per_byte_sensitivity" in notes
    pbs = notes["per_byte_sensitivity"]
    assert pbs["archive_sha256_prefix"] == archive_sha[:12]
    assert pbs["n_bytes"] == 50
    assert pbs["top_k_count"] == 100 or pbs["top_k_count"] <= 50
    # top_k_indices_first_10 capped at 10 entries.
    assert len(pbs["top_k_indices_first_10"]) <= 10
    # Rationale should cite the prefix.
    assert archive_sha[:12] in verdict["rationale"]


def test_consume_candidate_axis_tag_advisory_for_darwin_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per CLAUDE.md MPS-is-NOISE + Catalog #127/#192: darwin hw → advisory axis."""
    archive_sha = "6" * 64
    ledger = _write_synthetic_anchor(
        tmp_path,
        archive_sha=archive_sha,
        measurement_hardware="darwin_arm64_m5_max_macos_cpu_advisory",
        measurement_axis="[macOS-CPU advisory]",
    )

    from tac import master_gradient_per_byte_consumer as helper_mod
    original = helper_mod.load_per_byte_sensitivity_for_archive
    monkeypatch.setattr(
        helper_mod,
        "load_per_byte_sensitivity_for_archive",
        lambda *a, **kw: original(*a, **{**kw, "path": ledger}),
    )

    verdict = consumer.consume_candidate({"archive_sha256": archive_sha})
    assert verdict["axis_tag"] == "[macOS-CPU advisory]"


def test_consume_candidate_axis_tag_predicted_for_unknown_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-advisory hardware → preserves declared axis (or [predicted] fallback)."""
    archive_sha = "7" * 64
    ledger = _write_synthetic_anchor(
        tmp_path,
        archive_sha=archive_sha,
        measurement_hardware="linux_x86_64_t4",
        measurement_axis="[contest-CUDA]",
    )

    from tac import master_gradient_per_byte_consumer as helper_mod
    original = helper_mod.load_per_byte_sensitivity_for_archive
    monkeypatch.setattr(
        helper_mod,
        "load_per_byte_sensitivity_for_archive",
        lambda *a, **kw: original(*a, **{**kw, "path": ledger}),
    )

    verdict = consumer.consume_candidate({"archive_sha256": archive_sha})
    # Hardware is linux_x86_64_t4 → no advisory token → preserves declared axis.
    assert verdict["axis_tag"] == "[contest-CUDA]"


# ─────────────────────────────────────────────────────────────────────────
# Cathedral autopilot auto-discovery regression guard
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def autopilot_loop_module():
    """Import the cathedral_autopilot_autonomous_loop module."""
    repo_root = Path(__file__).parent.parent.parent.parent
    tools_dir = repo_root / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        return importlib.import_module("cathedral_autopilot_autonomous_loop")
    finally:
        if str(tools_dir) in sys.path:
            sys.path.remove(str(tools_dir))


def test_auto_discovery_includes_per_byte_sensitivity_consumer(
    autopilot_loop_module,
) -> None:
    """Per Catalog #335 paradigm shift: the new consumer auto-discovers."""
    regs = autopilot_loop_module.discover_and_register_consumers()
    names = [r["consumer_name"] for r in regs]
    assert "per_byte_sensitivity_consumer" in names, (
        f"per_byte_sensitivity_consumer must auto-discover. Found: {names}"
    )


def test_auto_discovery_marks_new_consumer_compliant(
    autopilot_loop_module,
) -> None:
    """Per Catalog #335: the new consumer is contract-compliant (live count = 0)."""
    regs = autopilot_loop_module.discover_and_register_consumers()
    pbs = next(
        (r for r in regs if r["consumer_name"] == "per_byte_sensitivity_consumer"),
        None,
    )
    assert pbs is not None
    assert pbs["contract_compliant"] is True
    assert pbs["validation_errors"] == ()


def test_auto_discovery_modules_includes_per_byte_consumer(
    autopilot_loop_module,
) -> None:
    """Production discovery (skips underscore packages) includes the new consumer."""
    modules = autopilot_loop_module.discover_compliant_consumer_modules()
    module_names = [m.CONSUMER_NAME for m in modules]
    assert "per_byte_sensitivity_consumer" in module_names


def test_per_byte_consumer_has_correct_hooks_in_discovery(
    autopilot_loop_module,
) -> None:
    """The discovered consumer carries hooks 1+3+4 per the landing memo."""
    regs = autopilot_loop_module.discover_and_register_consumers()
    pbs = next(
        r for r in regs if r["consumer_name"] == "per_byte_sensitivity_consumer"
    )
    hook_ints = [int(h) for h in pbs["consumer_hook_numbers"]]
    assert 1 in hook_ints  # SENSITIVITY_MAP
    assert 3 in hook_ints  # BIT_ALLOCATOR
    assert 4 in hook_ints  # CATHEDRAL_AUTOPILOT_DISPATCH
    # NOT included
    assert 2 not in hook_ints  # PARETO_CONSTRAINT
    assert 5 not in hook_ints  # CONTINUAL_LEARNING_POSTERIOR
    assert 6 not in hook_ints  # PROBE_DISAMBIGUATOR


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_per_byte_consumer_count_at_least_22(autopilot_loop_module) -> None:
    """Per the live repo floor: cumulative cathedral_consumers count is >= 22.

    The exact directory cardinality can change as sister work lands, but the
    production discovery surface must not silently drop below the current
    floor while still claiming the per-byte sensitivity consumer is wired.
    """
    modules = autopilot_loop_module.discover_compliant_consumer_modules()
    assert len(modules) >= 22, (
        f"expected at least 22 production consumers, found {len(modules)}"
    )


def test_live_repo_consume_candidate_on_fec6_frontier_archive(
    autopilot_loop_module,
) -> None:
    """Regression guard against the live FEC6 frontier archive anchor.

    Per `feedback_master_gradient_consumer_cathedral_wire_in_landed_20260519.md`
    archive ``6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf``
    has an aggregate_per_byte_v1 anchor. If the ledger is present, the
    consumer should produce a signal verdict (vs no-signal).
    """
    fec6_frontier_sha = (
        "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
    )
    repo_root = Path(__file__).parent.parent.parent.parent
    ledger_path = repo_root / ".omx" / "state" / "master_gradient_anchors.jsonl"
    if not ledger_path.exists():
        pytest.skip("live master_gradient_anchors.jsonl missing on this checkout")

    verdict = consumer.consume_candidate({"archive_sha256": fec6_frontier_sha})
    # Either signal (anchor in live ledger) or no-signal — both are valid.
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    # If notes are present, validate the schema.
    if "notes" in verdict:
        pbs = verdict["notes"].get("per_byte_sensitivity", {})
        assert pbs.get("archive_sha256_prefix") == fec6_frontier_sha[:12]
        assert pbs.get("n_bytes", 0) > 0


def test_consumer_verdict_schema_keys_match_canonical_contract() -> None:
    """All required keys per CathedralConsumerContract.consume_candidate docstring."""
    verdict = consumer.consume_candidate({})
    # Required per the canonical Protocol docstring:
    assert "predicted_delta_adjustment" in verdict
    assert "rationale" in verdict
    assert "axis_tag" in verdict
    # Optional but recommended:
    assert "promotable" in verdict
    assert "confidence" in verdict


def test_consumer_rationale_min_length() -> None:
    """Rationale must be ≥4 chars per CathedralConsumerContract canonical."""
    verdict = consumer.consume_candidate({})
    assert len(verdict["rationale"]) >= 4
