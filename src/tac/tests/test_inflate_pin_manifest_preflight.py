"""Controls for Catalog #420's inflate-pin manifest-rebind guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import PreflightError, check_inflate_pin_patch_rebinds_manifest


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_definition_without_manifest_rebind_is_caught(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/bad.py", "def patch_inflate_pins(root):\n    root.write_text('pin')\n")
    found = check_inflate_pin_patch_rebinds_manifest(repo_root=tmp_path)
    assert len(found) == 1 and "bad.py:1" in found[0]


def test_canonical_import_and_rebind_pass(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "experiments/good.py",
        "from tac.receiver_manifest import rebind_receiver_manifest\n"
        "def patch_inflate_pins(root):\n"
        "    root.write_text('pin')\n"
        "    rebind_receiver_manifest(root, ('inflate.py',))\n",
    )
    assert check_inflate_pin_patch_rebinds_manifest(repo_root=tmp_path, strict=True) == []


def test_unknown_patch_call_is_caught(tmp_path: Path) -> None:
    _write(tmp_path, "tools/caller.py", "def stage(root):\n    patch_inflate_pins(root)\n")
    assert "no manifest-rebinding producer" in check_inflate_pin_patch_rebinds_manifest(
        repo_root=tmp_path
    )[0]


def test_call_to_a_canonical_producer_passes(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "experiments/producer.py",
        "from tac.receiver_manifest import rebind_receiver_manifest\n"
        "def patch_inflate_pins(root):\n"
        "    rebind_receiver_manifest(root, ('inflate.py',))\n",
    )
    _write(tmp_path, "tools/caller.py", "def stage(root):\n    patch_inflate_pins(root)\n")
    assert check_inflate_pin_patch_rebinds_manifest(repo_root=tmp_path, strict=True) == []


def test_substantive_same_line_waiver_is_respected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "experiments/waived.py",
        "def patch_inflate_pins(root):  # MANIFEST_REBIND_OK: external writer owns this transaction\n"
        "    root.write_text('pin')\n",
    )
    assert check_inflate_pin_patch_rebinds_manifest(repo_root=tmp_path, strict=True) == []


@pytest.mark.parametrize("rationale", ["<rationale>", "TODO", "placeholder"])
def test_placeholder_waiver_is_rejected(tmp_path: Path, rationale: str) -> None:
    _write(
        tmp_path,
        "experiments/waived.py",
        f"def patch_inflate_pins(root):  # MANIFEST_REBIND_OK:{rationale}\n"
        "    root.write_text('pin')\n",
    )
    assert check_inflate_pin_patch_rebinds_manifest(repo_root=tmp_path)


def test_strict_mode_raises(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/bad.py", "def patch_inflate_pins(root):\n    pass\n")
    with pytest.raises(PreflightError, match="Catalog #420"):
        check_inflate_pin_patch_rebinds_manifest(repo_root=tmp_path, strict=True)


def test_live_repository_count_is_zero() -> None:
    assert check_inflate_pin_patch_rebinds_manifest(strict=True) == []
