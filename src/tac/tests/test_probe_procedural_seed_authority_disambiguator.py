# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "probe_procedural_seed_authority_disambiguator.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("procedural_seed_probe", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, payload)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_packet(
    root: Path,
    name: str,
    *,
    members: list[tuple[str, bytes]],
    script: str,
) -> Path:
    packet = root / name
    packet.mkdir()
    _write_zip(packet / "archive.zip", members)
    _write_executable(packet / "inflate.sh", script)
    return packet


def _archive_seed_script(seed_member: str = "seed.bin") -> str:
    return f"""#!/bin/sh
set -eu
archive_dir="$1"
output_dir="$2"
file_list="$3"
mkdir -p "$output_dir"
seed="$(cat "$archive_dir/{seed_member}")"
while IFS= read -r item; do
  [ -n "$item" ] || continue
  safe_item="$(printf "%s" "$item" | tr '/:' '__')"
  printf "decoded:%s seed:%s\\n" "$item" "$seed" > "$output_dir/$safe_item.out"
done < "$file_list"
"""


def _runtime_constant_script(seed: str = "abc") -> str:
    return f"""#!/bin/sh
set -eu
output_dir="$2"
file_list="$3"
mkdir -p "$output_dir"
while IFS= read -r item; do
  [ -n "$item" ] || continue
  safe_item="$(printf "%s" "$item" | tr '/:' '__')"
  printf "decoded:%s seed:{seed}\\n" "$item" > "$output_dir/$safe_item.out"
done < "$file_list"
"""


def _make_valid_pair(tmp_path: Path) -> tuple[Path, Path]:
    archive_seeded = _make_packet(
        tmp_path,
        "archive_seeded",
        members=[("seed.bin", b"abc"), ("payload.bin", b"charged")],
        script=_archive_seed_script(),
    )
    runtime_constant = _make_packet(
        tmp_path,
        "runtime_constant",
        members=[("payload.bin", b"charged")],
        script=_runtime_constant_script("abc"),
    )
    return archive_seeded, runtime_constant


def test_probe_accepts_safe_archive_seed_and_hashes_outputs(tmp_path: Path) -> None:
    tool = _load_tool()
    archive_seeded, runtime_constant = _make_valid_pair(tmp_path)

    payload = tool.build_probe_payload(
        archive_seeded_packet=archive_seeded,
        runtime_constant_packet=runtime_constant,
        seed_member="seed.bin",
        file_list_entries=("0.mkv", "nested/1.mkv"),
        timeout_seconds=5,
    )

    assert payload["schema"] == tool.SCHEMA
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["comparison"]["same_inflated_output_manifest"] is True

    seed = payload["variants"]["archive_seeded"]["seed_member"]
    assert seed["member"] == "seed.bin"
    assert seed["bytes"] == 3
    assert seed["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert seed["provenance_kind"] == "archive_member_seed"

    manifest = payload["variants"]["archive_seeded"]["output_manifest"]
    assert manifest["file_count"] == 2
    assert manifest["provenance_kind"] == "inflated_output_manifest"


@pytest.mark.parametrize(
    ("members", "seed_member", "match"),
    [
        ([("payload.bin", b"x")], "seed.bin", "missing"),
        (
            [("seed.bin", b"a"), ("seed.bin", b"b")],
            "seed.bin",
            "duplicate",
        ),
        ([("../seed.bin", b"a")], "../seed.bin", "zip-slip"),
        ([(".seed.bin", b"a")], ".seed.bin", "hidden"),
    ],
)
def test_probe_rejects_unsafe_archive_seed_members(
    tmp_path: Path,
    members: list[tuple[str, bytes]],
    seed_member: str,
    match: str,
) -> None:
    tool = _load_tool()
    archive_seeded = _make_packet(
        tmp_path,
        "bad_archive_seeded",
        members=members,
        script=_archive_seed_script(seed_member),
    )
    runtime_constant = _make_packet(
        tmp_path,
        "runtime_constant",
        members=[("payload.bin", b"charged")],
        script=_runtime_constant_script("abc"),
    )

    with pytest.raises(ValueError, match=match):
        tool.build_probe_payload(
            archive_seeded_packet=archive_seeded,
            runtime_constant_packet=runtime_constant,
            seed_member=seed_member,
            timeout_seconds=5,
        )


def test_same_output_keeps_archive_and_script_seed_authority_distinct(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    archive_seeded, runtime_constant = _make_valid_pair(tmp_path)

    payload = tool.build_probe_payload(
        archive_seeded_packet=archive_seeded,
        runtime_constant_packet=runtime_constant,
        timeout_seconds=5,
    )

    assert payload["comparison"]["same_inflated_output_manifest"] is True
    dispositions = payload["authority_disposition"]
    assert (
        dispositions["archive_seeded"]["compliance_class"]
        == "canonical_archive_charged_seed"
    )
    assert (
        dispositions["runtime_constant"]["compliance_class"]
        == "organizer_risk_script_side_payload"
    )
    assert dispositions["archive_seeded"]["default_for_promotion"] is True
    assert dispositions["runtime_constant"]["default_for_promotion"] is False
    assert payload["authority_packet"]["ready_for_exact_eval_dispatch"] is False


def test_write_probe_json_does_not_write_outside_output_path_or_temp(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive_seeded, runtime_constant = _make_valid_pair(workspace)
    output = workspace / "probe.json"

    before_entries = {p.relative_to(workspace).as_posix() for p in workspace.rglob("*")}
    before_packet_hashes = {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }

    payload = tool.write_probe_json(
        output=output,
        archive_seeded_packet=archive_seeded,
        runtime_constant_packet=runtime_constant,
        timeout_seconds=5,
    )

    after_entries = {p.relative_to(workspace).as_posix() for p in workspace.rglob("*")}
    assert after_entries - before_entries == {"probe.json"}
    for rel, before_bytes in before_packet_hashes.items():
        assert (workspace / rel).read_bytes() == before_bytes
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == tool.SCHEMA
    assert payload["score_claim"] is False


def test_cli_accepts_script_seed_alias_and_writes_json(tmp_path: Path) -> None:
    tool = _load_tool()
    archive_seeded, runtime_constant = _make_valid_pair(tmp_path)
    output = tmp_path / "cli.json"

    rc = tool.main(
        [
            "--archive-seeded-packet",
            str(archive_seeded),
            "--script-seed-packet",
            str(runtime_constant),
            "--output",
            str(output),
            "--timeout-seconds",
            "5",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == tool.SCHEMA
    assert payload["score_claim"] is False
    assert payload["variants"]["runtime_constant"]["provenance_kind"] == (
        "runtime_constant_script_seed_submission_packet"
    )
    assert os.path.exists(output)
