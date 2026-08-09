"""Bind DDM-RC1's added Brotli/ANS runtime surface to the FX5 manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / "src" / "tac" / "pr130_runtime" / "fx1_runtime_tree"
MANIFEST = json.loads((TREE / "runtime-dependencies.json").read_text())
ENTRYPOINT = (TREE / "inflate.sh").read_text()


def _declared() -> dict[str, dict]:
    return {dependency["name"]: dependency for dependency in MANIFEST["dependencies"]}


def test_ddm_rc1_entrypoint_pins_both_self_installed_wheels() -> None:
    expected = {
        "EXPECTED_CONSTRICTION_VERSION": "constriction",
        "EXPECTED_BROTLI_VERSION": "brotli",
    }
    for variable, package in expected.items():
        match = re.search(rf"^{variable}=(\S+)$", ENTRYPOINT, re.MULTILINE)
        assert match is not None, f"inflate.sh no longer declares {variable}"
        assert match.group(1) == _declared()[package]["version"]


def test_ddm_rc1_entrypoint_checks_every_declared_self_installed_api() -> None:
    for dependency in MANIFEST["dependencies"]:
        if dependency["provisioning"] != "self_installed_by_entrypoint":
            continue
        for api in dependency["required_apis"]:
            leaf = api.rsplit(".", 1)[-1]
            assert ENTRYPOINT.count(f'"{leaf}"') >= 2, (
                f"{dependency['name']} declares {api}, but inflate.sh does not "
                "check it in both dependency verification paths"
            )


def test_ddm_rc1_entrypoint_selects_brotli_from_the_wire_tag(tmp_path: Path) -> None:
    cases = (
        (0, "legacy_lzma", "0"),
        (1, "split_brotli", "1"),
        (2, "split_lzma2", "0"),
    )
    env = dict(os.environ)
    env.update({
        "PYTHON": sys.executable,
        "PR130_DEPENDENCY_SELECTION_ONLY": "1",
    })
    for selector, model_codec, needs_brotli in cases:
        data_dir = tmp_path / model_codec
        data_dir.mkdir()
        (data_dir / "p").write_bytes(struct.pack("<I", (selector << 29) | 1))
        result = subprocess.run(
            [str(TREE / "inflate.sh"), str(data_dir), "0", str(data_dir / "out.raw")],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.stdout.strip() == (
            f"PR130_DEPENDENCY_SELECTION model_codec={model_codec} "
            f"needs_brotli={needs_brotli}"
        )

    reserved_dir = tmp_path / "reserved"
    reserved_dir.mkdir()
    (reserved_dir / "p").write_bytes(struct.pack("<I", (3 << 29) | 1))
    reserved = subprocess.run(
        [str(TREE / "inflate.sh"), str(reserved_dir), "0", str(reserved_dir / "out.raw")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert reserved.returncode != 0
    assert "reserved model-codec selector 3" in reserved.stderr


def test_ddm_rc1_manifest_hashes_the_current_runtime_tree() -> None:
    for name, expected in MANIFEST["source"]["copied_files"].items():
        observed = hashlib.sha256((TREE / name).read_bytes()).hexdigest()
        assert observed == expected, f"stale runtime custody hash for {name}"
