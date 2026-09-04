#!/usr/bin/env python3
"""Focused tests for the sealed-config pin re-root tool.

The tool writes a FORENSIC receipt, so these tests exercise the receipt's integrity claims
rather than field shapes: that ``config_in.sha256`` is the sha of the bytes actually read
(the ddm_bh1 read-after-write defect), that an in-place re-root is refused outright, that
the writes are atomic and leave no scratch behind, and that content drift still fails closed.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddm_reseal_pins_inside_sealed_tree as rs  # noqa: E402


def _pins(root: str) -> dict:
    return {
        "gt_cache": {"path": f"{root}/experiments/results/gt.npz", "bytes": 11, "sha256": "aa" * 32},
        "qbt_no2_gate": {"path": f"{root}/.omx/research/no2.md", "bytes": 22, "sha256": "bb" * 32},
    }


def _config(root: str = "/Users/adpena/Projects/pact") -> dict:
    return {
        "schema": "ddm_qbr1_cell.v1",
        "cell_id": "seed_1_fairform",
        "seed": 1,
        "total_steps": 5000,
        "source_pins": _pins(root),
    }


@pytest.fixture
def sealed(monkeypatch):
    """Stub the sealed-tree subprocess with content-identical pins at snapshot paths."""

    def _fake(sealed_tree: Path) -> dict:
        return _pins(str(sealed_tree))

    monkeypatch.setattr(rs, "verify_inputs_inside", _fake)
    return _fake


def _write_config(tmp_path: Path, config: dict | None = None) -> Path:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(config if config is not None else _config(), indent=2) + "\n")
    return path


# --- the ddm_bh1 defect: the receipt must attest to the bytes actually READ ----------------


def test_receipt_config_in_sha_is_the_original_input_bytes(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    expected = hashlib.sha256(config_in.read_bytes()).hexdigest()
    receipt = rs.reroot(
        config_in, tmp_path / "tree", tmp_path / "out" / "cell.rerooted.json", tmp_path / "out" / "receipt.json"
    )
    assert receipt["config_in"]["sha256"] == expected


def test_receipt_config_in_sha_differs_from_config_out_sha_when_paths_changed(tmp_path, sealed):
    """The re-root MUST change bytes, so the two shas must not collide -- the defect's symptom."""

    config_in = _write_config(tmp_path)
    receipt = rs.reroot(
        config_in, tmp_path / "tree", tmp_path / "out.json", tmp_path / "receipt.json"
    )
    assert receipt["paths_rerooted"] == ["gt_cache", "qbt_no2_gate"]
    assert receipt["config_in"]["sha256"] != receipt["config_out"]["sha256"]


def test_receipt_config_out_sha_matches_the_file_on_disk(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    out = tmp_path / "out.json"
    receipt = rs.reroot(config_in, tmp_path / "tree", out, tmp_path / "receipt.json")
    assert hashlib.sha256(out.read_bytes()).hexdigest() == receipt["config_out"]["sha256"]


def test_input_file_is_left_byte_identical(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    before = config_in.read_bytes()
    rs.reroot(config_in, tmp_path / "tree", tmp_path / "out.json", tmp_path / "receipt.json")
    assert config_in.read_bytes() == before


# --- in-place is refused, never silently tolerated -----------------------------------------


def test_in_place_reroot_is_refused(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    with pytest.raises(rs.ResealError, match="config_out must differ from config_in"):
        rs.reroot(config_in, tmp_path / "tree", config_in, tmp_path / "receipt.json")


def test_receipt_over_config_in_is_refused(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    with pytest.raises(rs.ResealError, match="receipt_out must differ from config_in"):
        rs.reroot(config_in, tmp_path / "tree", tmp_path / "out.json", config_in)


def test_receipt_over_config_out_is_refused(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    out = tmp_path / "out.json"
    with pytest.raises(rs.ResealError, match="receipt_out must differ from config_out"):
        rs.reroot(config_in, tmp_path / "tree", out, out)


def test_refusal_writes_nothing(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    before = config_in.read_bytes()
    with pytest.raises(rs.ResealError):
        rs.reroot(config_in, tmp_path / "tree", config_in, tmp_path / "receipt.json")
    assert config_in.read_bytes() == before
    assert not (tmp_path / "receipt.json").exists()


# --- atomic writes -------------------------------------------------------------------------


def test_writes_leave_no_tmp_scratch(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    rs.reroot(config_in, tmp_path / "tree", tmp_path / "o" / "out.json", tmp_path / "r" / "receipt.json")
    assert list(tmp_path.rglob("*.tmp")) == []


def test_receipt_parent_directory_is_created(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    receipt_out = tmp_path / "deep" / "nested" / "receipt.json"
    rs.reroot(config_in, tmp_path / "tree", tmp_path / "out.json", receipt_out)
    assert json.loads(receipt_out.read_text())["schema"] == "sealed_config_pin_reroot.v1"


# --- the seal itself still fails closed -----------------------------------------------------


def test_content_drift_is_refused(tmp_path, monkeypatch):
    config_in = _write_config(tmp_path)
    drifted = _pins("/snap")
    drifted["gt_cache"]["sha256"] = "cc" * 32
    monkeypatch.setattr(rs, "verify_inputs_inside", lambda _t: drifted)
    with pytest.raises(rs.ResealError, match="CONTENT drift"):
        rs.reroot(config_in, tmp_path / "tree", tmp_path / "out.json", tmp_path / "receipt.json")


def test_byte_count_drift_is_refused(tmp_path, monkeypatch):
    config_in = _write_config(tmp_path)
    drifted = _pins("/snap")
    drifted["gt_cache"]["bytes"] = 12
    monkeypatch.setattr(rs, "verify_inputs_inside", lambda _t: drifted)
    with pytest.raises(rs.ResealError, match="CONTENT drift"):
        rs.reroot(config_in, tmp_path / "tree", tmp_path / "out.json", tmp_path / "receipt.json")


def test_pin_key_set_mismatch_is_refused(tmp_path, monkeypatch):
    config_in = _write_config(tmp_path)
    partial = {"gt_cache": _pins("/snap")["gt_cache"]}
    monkeypatch.setattr(rs, "verify_inputs_inside", lambda _t: partial)
    with pytest.raises(rs.ResealError, match="pin key sets differ"):
        rs.reroot(config_in, tmp_path / "tree", tmp_path / "out.json", tmp_path / "receipt.json")


def test_only_pin_paths_change_nothing_else(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    out = tmp_path / "out.json"
    rs.reroot(config_in, tmp_path / "tree", out, tmp_path / "receipt.json")
    before, after = _config(), json.loads(out.read_text())
    assert {k: v for k, v in after.items() if k != "source_pins"} == {
        k: v for k, v in before.items() if k != "source_pins"
    }
    for name, row in after["source_pins"].items():
        assert row["sha256"] == before["source_pins"][name]["sha256"]
        assert row["bytes"] == before["source_pins"][name]["bytes"]
        assert row["path"] != before["source_pins"][name]["path"]


def test_pins_total_counts_every_pin(tmp_path, sealed):
    config_in = _write_config(tmp_path)
    receipt = rs.reroot(config_in, tmp_path / "tree", tmp_path / "out.json", tmp_path / "receipt.json")
    assert receipt["pins_total"] == 2


def test_aliased_spelling_of_the_same_file_is_refused(tmp_path, sealed):
    """Two spellings of one path must not alias past the distinctness guard."""

    config_in = _write_config(tmp_path)
    aliased = tmp_path / "sub" / ".." / "cell.json"
    (tmp_path / "sub").mkdir()
    with pytest.raises(rs.ResealError, match="config_out must differ from config_in"):
        rs.reroot(config_in, tmp_path / "tree", aliased, tmp_path / "receipt.json")


def test_written_files_are_utf8_regardless_of_locale(tmp_path, sealed):
    """The receipt's config_out sha is taken over UTF-8, so the file must be UTF-8 too."""

    config = _config()
    config["note"] = "é中"  # non-ASCII: locale-encoded bytes would differ
    config_in = _write_config(tmp_path, config)
    out = tmp_path / "out.json"
    receipt = rs.reroot(config_in, tmp_path / "tree", out, tmp_path / "receipt.json")
    assert hashlib.sha256(out.read_bytes()).hexdigest() == receipt["config_out"]["sha256"]
    assert json.loads(out.read_text(encoding="utf-8"))["note"] == "é中"
