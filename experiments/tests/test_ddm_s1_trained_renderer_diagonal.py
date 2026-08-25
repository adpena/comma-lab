from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments import ddm_s1_trained_renderer_diagonal as s1


def test_break_even_uses_named_archive_denominator() -> None:
    row = s1.break_even_row(1_502)
    assert row["bytes_shed_numerator"] == 1_502
    assert row["gb1_archive_byte_denominator"] == 180_215
    assert math.isclose(row["rate_credit_s"], 1_502 * 25.0 / 37_545_489, rel_tol=0.0, abs_tol=0.0)
    assert row["maximum_combined_seg_plus_pose_damage_s_for_delta_s_lt_zero"] == row["rate_credit_s"]


def test_seed_window_ledger_has_two_seeds_per_window() -> None:
    rows = s1.seed_window_rows()
    assert len(rows) == 6
    for epoch in s1.WINDOW_END_EPOCHS:
        seeds = {row["seed"] for row in rows if row["window_end_epoch"] == epoch}
        assert seeds == set(s1.SEEDS)
    assert all(row["status"] == "BLOCKED_NOT_RUN" for row in rows)
    assert all(row["mechanism"] == "TRAINED-not-SVD W96 renderer" for row in rows)


def test_interface_audit_types_current_incompatibilities() -> None:
    audit = s1.audit_interfaces(
        """if int(config["seed"]) != SEED:\n    pass\n"""
        "model = receiver.StudentSemanticRenderer(ARM_SPECS[str(config['arm'])])\n"
        "hpac, _, carrier = wd2_build._source_streams()\n"
        "member = wd2_build.SOURCE_RESIDUAL.read_bytes() + wd2_build.SOURCE_TOKEN.read_bytes()\n",
        'def apply_edits(tokens, edits_path):\n    pass\nparser.add_argument("--edits")\n',
        'OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_qs5_20260813")\narchive = CP135_ARCHIVE\n',
    )
    assert audit["ready"] is False
    assert len(audit["blockers"]) == 4
    assert audit["jg2"]["real_reencoder_present"] is True
    assert audit["jg2"]["moved_field_producer_present"] is False
    assert audit["qs5"]["generic_exact_object_entrypoint_present"] is False


def test_unknown_interface_names_are_not_inferred_ready() -> None:
    audit = s1.audit_interfaces(
        "def train_s1_multiseed_gb1():\n    pass\n",
        "def moved_object_reencoder():\n    pass\n",
        "def generic_exact_object_qs5():\n    pass\n",
    )
    # The audit refuses an unproved JG2 edits surface rather than inferring readiness
    # from names in a fixture.
    assert audit["ready"] is False
    assert audit["blockers"] == ["JG2 edited-field input surface is absent"]


def test_source_markers_alone_cannot_create_runnable_order() -> None:
    result = s1.compile_seal(
        source_receipt={"passed": True},
        interface_audit={"blockers": [], "ready": True},
    )
    assert result["disposition"] == "BLOCKED_MISSING_COMPOSED_INTERFACES"
    assert result["blockers"] == [
        "S1 stage adapters are absent; readiness cannot be inferred from source markers or command-shaped strings"
    ]
    assert all(stage["status"] == "BLOCKED" for stage in result["stages"])


def test_compile_seal_never_emits_argv_while_blocked() -> None:
    audit = {"blockers": ["typed blocker"], "ready": False}
    result = s1.compile_seal(source_receipt={"passed": True}, interface_audit=audit)
    assert result["disposition"] == "BLOCKED_MISSING_COMPOSED_INTERFACES"
    assert result["training_launched"] is False
    assert result["frontier_moved"] is False
    assert all(stage["exact_command_argv"] is None for stage in result["stages"])
    assert result["verified_two_missing_halves"] == [
        "TRAINED-not-SVD W96 renderer",
        "token re-encode on the moved object",
    ]
    assert result["representation_dispositions"]["pointwise_svd_w96_r32"].startswith("DEAD")


def test_custody_failure_is_a_typed_fire_blocker() -> None:
    result = s1.compile_seal(
        source_receipt={
            "passed": False,
            "rj1": {
                "verified_file_numerator": 191,
                "inventory_file_denominator": 192,
            },
        },
        interface_audit={"blockers": [], "ready": True},
    )
    assert result["disposition"] == "BLOCKED_MISSING_COMPOSED_INTERFACES"
    assert result["blockers"] == ["RJ1 custody inventory is not coherent: 191/192 retained file records verified"]


def test_atomic_retention_is_resumable_and_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "retained" / "payload.json"
    first = s1.atomic_json(path, {"value": 1})
    repeat = s1.atomic_json(path, {"value": 1})
    assert first == repeat
    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(s1.S1Error, match="refusing to overwrite differing"):
        s1.atomic_json(path, {"value": 2})


def test_retention_inventory_excludes_appledouble_metadata(tmp_path: Path) -> None:
    (tmp_path / "payload.bin").write_bytes(b"payload")
    (tmp_path / "._payload.bin").write_bytes(b"filesystem metadata")
    inventory = s1.retention_inventory(tmp_path)
    assert inventory["file_count"] == 1
    assert inventory["files"][0]["path"] == str((tmp_path / "payload.bin").resolve())


def test_storage_preflight_reuses_retained_free_space_snapshot(tmp_path: Path) -> None:
    receipt_root = tmp_path / "seal"
    receipt_root.mkdir()
    first = s1.storage_preflight_payload(receipt_root, tmp_path, 2 << 30)
    s1.atomic_json(receipt_root / "STORAGE_PREFLIGHT.json", first)
    resumed = s1.storage_preflight_payload(receipt_root, tmp_path, 3 << 30)
    assert resumed == first
    assert resumed["observed_free_bytes"] == 2 << 30


def test_actual_sources_remain_fail_closed_until_adapters_land() -> None:
    audit = s1.audit_interfaces(
        s1.WD3_SOURCE.read_text(encoding="utf-8"),
        s1.JG2_SOURCE.read_text(encoding="utf-8"),
        s1.QS5_SOURCE.read_text(encoding="utf-8"),
    )
    assert audit["ready"] is False
    assert audit["wd3"]["fixed_seed_guard_lines"]
    assert audit["wd3"]["wd2_body_binding_lines"]["source_token"]
    assert audit["jg2"]["real_reencoder_present"] is True
    assert audit["qs5"]["cp135_binding_lines"]
