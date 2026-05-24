# SPDX-License-Identifier: MIT
"""Tests for `tac.packet_compiler.deterministic_compiler`.

Coverage matrix:

* 3 modes (identity / canonicalize / optimize) x 4 profiles
* fail-closed gates: hidden sidecars, scorer-at-inflate, network /
  external-state, unsupported ZIP features, parser divergence, optimize
  mode contract preconditions.
* identity round-trip preserves bytes
* canonicalize re-emits canonical metadata + preserves payload bytes
* manifest + no-op proof contents
* golden vectors are produced for every mode + profile
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import tac.packet_compiler.deterministic_compiler as compiler
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.packet_compiler.deterministic_compiler import (
    COMPILER_MODES,
    DETERMINISTIC_COMPILER_REQUIRED_ORDER,
    DETERMINISTIC_ZIP_DATE_TIME,
    MANIFEST_NAME,
    NO_OP_PROOF_NAME,
    PACKET_IR_OPERATION_SET_BRIDGE_CONTRACT_SCHEMA,
    PACKET_IR_OPERATION_SET_REQUIRED_PROOFS,
    PACKET_IR_OPERATION_SET_SCHEMA,
    SCHEMA_VERSION,
    TARGET_PROFILES,
    DeterministicPacketCompilerError,
    compile_packet,
    inspect_packet_oracle,
    packetir_operation_set_bridge_contract,
)
from tac.repo_io import sha256_file

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_canonical_member(
    zf: zipfile.ZipFile,
    name: str,
    payload: bytes,
    *,
    date_time: tuple[int, int, int, int, int, int] = DETERMINISTIC_ZIP_DATE_TIME,
    mode: int = 0o644,
    create_system: int = 3,
    compress_type: int = zipfile.ZIP_STORED,
) -> None:
    info = zipfile.ZipInfo(name, date_time=date_time)
    info.compress_type = compress_type
    info.external_attr = mode << 16
    info.create_system = create_system
    zf.writestr(info, payload, compress_type=compress_type)


def _write_canonical_packet(root: Path) -> Path:
    """Write a minimal canonical contest-shaped packet under ``root``."""

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        _write_canonical_member(zf, "renderer.bin", b"renderer-payload-bytes")
        _write_canonical_member(zf, "masks.mkv", b"masks-payload-bytes")
    inflate_sh = root / "inflate.sh"
    inflate_sh.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'DATA_DIR="$1"\nOUTPUT_DIR="$2"\nFILE_LIST="$3"\n'
        'python3 "$(dirname "$0")/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)
    inflate_py = root / "inflate.py"
    inflate_py.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "archive_dir = Path(sys.argv[1])\n"
        "output_dir = Path(sys.argv[2])\n"
        "file_list = Path(sys.argv[3])\n"
        "for line in file_list.read_text().splitlines():\n"
        "    name = line.strip()\n"
        "    if not name:\n"
        "        continue\n"
        "    member = archive_dir / name\n"
        "    (output_dir / name).write_bytes(member.read_bytes())\n",
        encoding="utf-8",
    )
    return root


def _write_packet_with_scorer_at_inflate(root: Path) -> Path:
    """Same shape but inflate.py imports a scorer (strict-scorer-rule)."""

    _write_canonical_packet(root)
    (root / "inflate.py").write_text(
        "from upstream.modules import PoseNet, SegNet  # forbidden\n",
        encoding="utf-8",
    )
    return root


def _write_packet_with_network_in_inflate(root: Path) -> Path:
    _write_canonical_packet(root)
    (root / "inflate.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'pip install torch  # forbidden network token\n',
        encoding="utf-8",
    )
    (root / "inflate.sh").chmod(0o755)
    return root


def _write_packet_with_external_state(root: Path) -> Path:
    _write_canonical_packet(root)
    (root / "inflate.py").write_text(
        "from pathlib import Path\n"
        'Path("/Users/evil/leak").write_bytes(b"x")\n',
        encoding="utf-8",
    )
    return root


def _write_packet_with_unsupported_zip_method(root: Path) -> Path:
    """Use ZIP_BZIP2 (8 is DEFLATED OK, 12 is BZIP2 = forbidden)."""

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("renderer.bin", date_time=DETERMINISTIC_ZIP_DATE_TIME)
        info.compress_type = zipfile.ZIP_BZIP2
        info.external_attr = 0o644 << 16
        zf.writestr(info, b"x" * 200, compress_type=zipfile.ZIP_BZIP2)
    inflate_sh = root / "inflate.sh"
    inflate_sh.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\necho "$1" "$2" "$3"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)
    inflate_py = root / "inflate.py"
    inflate_py.write_text(
        "import sys\nfor line in open(sys.argv[3]).read().splitlines():\n"
        "    pass\n",
        encoding="utf-8",
    )
    return root


def _write_packet_with_hidden_sidecar(root: Path) -> Path:
    _write_canonical_packet(root)
    (root / "secret_sidecar.bin").write_bytes(b"hidden")
    return root


def _runtime_consumption_proof_for_packet(packet: Path) -> dict[str, object]:
    """Return a minimal typed proof bound to the fixture packet."""

    return {
        "schema": "deterministic_compiler_test_runtime_consumption_proof_v1",
        "mutated_archive_sha256": sha256_file(packet / "archive.zip"),
        "runtime_inflate_py_sha256": sha256_file(packet / "inflate.py"),
        "runtime_sidecar_decode_consumption_claim": True,
        "runtime_sidecar_apply_consumption_claim": True,
        "score_affecting_section_names": ["renderer.bin"],
        "sections": [{"name": "renderer.bin", "bytes": 22}],
    }


# ---------------------------------------------------------------------------
# Identity-mode tests
# ---------------------------------------------------------------------------


def test_identity_mode_byte_for_byte_parity(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    in_sha = sha256_file(packet / "archive.zip")

    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )

    out_archive = Path(result.archive_path)
    assert sha256_file(out_archive) == in_sha
    assert result.archive_sha256 == in_sha
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.no_op_proof["no_op_detector_passed"] in (True, None)


def test_identity_mode_writes_manifest_and_no_op_proof(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )

    manifest_path = Path(result.output_dir) / MANIFEST_NAME
    no_op_path = Path(result.output_dir) / NO_OP_PROOF_NAME
    assert manifest_path.is_file()
    assert no_op_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["mode"] == "identity"
    assert manifest["target_profile"] == "contest_one_video_replay"
    assert manifest["score_claim"] is False
    assert manifest["golden_vectors"]["archive_sha256"] == result.archive_sha256
    no_op = json.loads(no_op_path.read_text(encoding="utf-8"))
    assert no_op["score_affecting_payload_changed"] is False


def test_identity_mode_baseline_sha_mismatch_blocks(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
        baseline_archive_sha256="0" * 64,
    )
    assert any("parser_divergence" in b for b in result.blockers)


# ---------------------------------------------------------------------------
# Canonicalize-mode tests
# ---------------------------------------------------------------------------


def _write_packet_with_noncanonical_metadata(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        _write_canonical_member(
            zf,
            "renderer.bin",
            b"renderer",
            date_time=(2026, 5, 7, 12, 34, 56),
            mode=0o755,
            create_system=0,
        )
    inflate_sh = root / "inflate.sh"
    inflate_sh.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\necho "$1" "$2" "$3"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)
    (root / "inflate.py").write_text(
        "import sys\nfor line in open(sys.argv[3]).read().splitlines():\n"
        "    pass\n",
        encoding="utf-8",
    )
    return root


def test_canonicalize_mode_rewrites_metadata_only(tmp_path: Path) -> None:
    packet = _write_packet_with_noncanonical_metadata(tmp_path / "packet")
    in_sha = sha256_file(packet / "archive.zip")

    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="canonicalize",
        target_profile="contest_one_video_replay",
    )

    out_archive = Path(result.archive_path)
    out_sha = sha256_file(out_archive)
    # Payload preserved, but archive bytes change because metadata changed.
    assert out_sha != in_sha
    with zipfile.ZipFile(out_archive, "r") as zf:
        info = zf.getinfo("renderer.bin")
        assert tuple(info.date_time) == DETERMINISTIC_ZIP_DATE_TIME
        assert (info.external_attr >> 16) & 0o7777 == 0o644
        assert info.create_system == 3
        assert zf.read("renderer.bin") == b"renderer"


def test_canonicalize_mode_refuses_score_affecting_payload_changed(
    tmp_path: Path,
) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    with pytest.raises(DeterministicPacketCompilerError):
        compile_packet(
            input_packet=packet,
            output_dir=tmp_path / "out",
            mode="canonicalize",
            target_profile="contest_one_video_replay",
            score_affecting_payload_changed=True,
        )


def test_canonicalize_report_lists_changed_members(tmp_path: Path) -> None:
    packet = _write_packet_with_noncanonical_metadata(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="canonicalize",
        target_profile="contest_one_video_replay",
    )
    manifest = json.loads(
        (Path(result.output_dir) / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    report = manifest["canonicalize_report"]
    assert report["changed_member_count"] == 1
    assert report["changed_members"][0]["name"] == "renderer.bin"


# ---------------------------------------------------------------------------
# Optimize-mode tests
# ---------------------------------------------------------------------------


def test_optimize_mode_requires_score_affecting_payload_changed(
    tmp_path: Path,
) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    with pytest.raises(DeterministicPacketCompilerError):
        compile_packet(
            input_packet=packet,
            output_dir=tmp_path / "out",
            mode="optimize",
            target_profile="contest_one_video_replay",
        )


def test_optimize_mode_requires_baseline(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    with pytest.raises(DeterministicPacketCompilerError):
        compile_packet(
            input_packet=packet,
            output_dir=tmp_path / "out",
            mode="optimize",
            target_profile="contest_one_video_replay",
            score_affecting_payload_changed=True,
            runtime_consumption_proof=_runtime_consumption_proof_for_packet(packet),
        )


def test_optimize_mode_requires_runtime_consumption_proof(
    tmp_path: Path,
) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    with pytest.raises(DeterministicPacketCompilerError):
        compile_packet(
            input_packet=packet,
            output_dir=tmp_path / "out",
            mode="optimize",
            target_profile="contest_one_video_replay",
            score_affecting_payload_changed=True,
            baseline_archive_sha256="0" * 64,
            baseline_archive_size_bytes=100,
            runtime_consumption_proof=False,
        )


def test_optimize_mode_refuses_bare_boolean_runtime_proof(
    tmp_path: Path,
) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    with pytest.raises(
        DeterministicPacketCompilerError,
        match="bare booleans are forbidden",
    ):
        compile_packet(
            input_packet=packet,
            output_dir=tmp_path / "out",
            mode="optimize",
            target_profile="contest_one_video_replay",
            score_affecting_payload_changed=True,
            baseline_archive_sha256="0" * 64,
            baseline_archive_size_bytes=100,
            runtime_consumption_proof=True,
        )


def test_optimize_mode_writes_no_op_proof_with_byte_delta(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="optimize",
        target_profile="contest_one_video_replay",
        score_affecting_payload_changed=True,
        baseline_archive_sha256="0" * 64,
        baseline_archive_size_bytes=100,
        runtime_consumption_proof=_runtime_consumption_proof_for_packet(packet),
    )
    assert result.no_op_proof["score_affecting_payload_changed"] is True
    assert result.no_op_proof["sha_changed"] is True
    assert result.no_op_proof["runtime_consumption_proof"] is True
    assert result.no_op_proof["runtime_consumption_proof_source"] == "<mapping>"
    assert result.no_op_proof["no_op_detector_passed"] is True


def test_optimize_mode_blocks_runtime_proof_archive_mismatch(
    tmp_path: Path,
) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    proof = _runtime_consumption_proof_for_packet(packet)
    proof["mutated_archive_sha256"] = "f" * 64

    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="optimize",
        target_profile="contest_one_video_replay",
        score_affecting_payload_changed=True,
        baseline_archive_sha256="0" * 64,
        baseline_archive_size_bytes=100,
        runtime_consumption_proof=proof,
    )

    assert any(
        b.startswith("runtime_consumption_proof_archive_sha256_mismatch")
        for b in result.blockers
    )
    assert result.no_op_proof["runtime_consumption_proof"] is False
    assert result.no_op_proof["no_op_detector_passed"] is False


def test_oracle_inspection_failure_is_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")

    def _raise_oracle(*args, **kwargs):
        raise compiler._OraclePacketCompilerError("fixture oracle failure")

    monkeypatch.setattr(compiler, "_oracle_inspect_packet", _raise_oracle)

    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )

    assert "packet_oracle_inspect_failed:fixture oracle failure" in result.blockers
    manifest = json.loads(
        (Path(result.output_dir) / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert "packet_oracle_inspect_failed:fixture oracle failure" in manifest["blockers"]


# ---------------------------------------------------------------------------
# Profile-specific tests (4 profiles)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", list(TARGET_PROFILES))
def test_all_profiles_identity_mode_works(tmp_path: Path, profile: str) -> None:
    packet = _write_canonical_packet(tmp_path / f"packet_{profile}")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / f"out_{profile}",
        mode="identity",
        target_profile=profile,
    )
    assert result.target_profile == profile
    assert result.target_profile_policy["description"]
    assert result.score_claim is False


def test_production_profile_does_not_require_inflate_sh(tmp_path: Path) -> None:
    # Production profile allows packet without inflate.sh (Python/Rust entry).
    root = tmp_path / "packet"
    root.mkdir()
    archive_path = root / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        _write_canonical_member(zf, "decoder.bin", b"d")
    (root / "src").mkdir()
    (root / "src" / "__init__.py").write_text(
        "def decode():\n    pass\n", encoding="utf-8",
    )
    result = compile_packet(
        input_packet=root,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="production_generalized",
    )
    blockers = list(result.blockers)
    assert not any("inflate_sh_missing" in b for b in blockers)
    assert not any("inflate_py_missing" in b for b in blockers)


def test_contest_profile_requires_inflate_sh(tmp_path: Path) -> None:
    root = tmp_path / "packet"
    root.mkdir()
    archive_path = root / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        _write_canonical_member(zf, "decoder.bin", b"d")
    # No inflate.sh / inflate.py written.
    result = compile_packet(
        input_packet=root,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )
    assert any("inflate_sh_missing" in b for b in result.blockers)
    assert any("inflate_py_missing" in b for b in result.blockers)


def test_unknown_target_profile_rejected(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    with pytest.raises(DeterministicPacketCompilerError):
        compile_packet(
            input_packet=packet,
            output_dir=tmp_path / "out",
            mode="identity",
            target_profile="contest_invalid",  # type: ignore[arg-type]
        )


def test_unknown_mode_rejected(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    with pytest.raises(DeterministicPacketCompilerError):
        compile_packet(
            input_packet=packet,
            output_dir=tmp_path / "out",
            mode="bogus",  # type: ignore[arg-type]
            target_profile="contest_one_video_replay",
        )


# ---------------------------------------------------------------------------
# Fail-closed gate tests
# ---------------------------------------------------------------------------


def test_scorer_at_inflate_blocks(tmp_path: Path) -> None:
    packet = _write_packet_with_scorer_at_inflate(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )
    assert any("PoseNet" in b for b in result.blockers)
    assert any("SegNet" in b for b in result.blockers)
    assert any("upstream.modules" in b for b in result.blockers)


def test_scorer_names_in_python_comments_do_not_block(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    (packet / "inflate.py").write_text(
        "# offline probe referenced PoseNet + SegNet, but runtime imports none\n"
        "from pathlib import Path\n"
        "Path(__file__)\n",
        encoding="utf-8",
    )

    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )

    assert not any("PoseNet" in b or "SegNet" in b for b in result.blockers)


def test_network_in_inflate_blocks(tmp_path: Path) -> None:
    packet = _write_packet_with_network_in_inflate(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )
    assert any("pip install" in b for b in result.blockers)


def test_external_state_in_inflate_blocks(tmp_path: Path) -> None:
    packet = _write_packet_with_external_state(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )
    assert any("/Users/" in b for b in result.blockers)


def test_unsupported_zip_method_blocks(tmp_path: Path) -> None:
    packet = _write_packet_with_unsupported_zip_method(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )
    assert any("unsupported_zip_method" in b for b in result.blockers)


def test_hidden_sidecar_blocks(tmp_path: Path) -> None:
    packet = _write_packet_with_hidden_sidecar(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )
    assert any("hidden_sidecar" in b for b in result.blockers)


def test_output_dir_non_empty_blocks(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.bin").write_bytes(b"x")
    with pytest.raises(DeterministicPacketCompilerError):
        compile_packet(
            input_packet=packet,
            output_dir=out,
            mode="identity",
            target_profile="contest_one_video_replay",
        )


def test_output_dir_non_empty_allowed_with_flag(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.bin").write_bytes(b"x")
    result = compile_packet(
        input_packet=packet,
        output_dir=out,
        mode="identity",
        target_profile="contest_one_video_replay",
        allow_existing_output_dir=True,
    )
    # Stale file was removed; the new archive is the only file with that name.
    assert not (out / "stale.bin").exists()
    assert Path(result.archive_path).is_file()


def test_input_packet_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(DeterministicPacketCompilerError):
        compile_packet(
            input_packet=tmp_path / "missing",
            output_dir=tmp_path / "out",
            mode="identity",
            target_profile="contest_one_video_replay",
        )


def test_bare_archive_zip_input_accepted(tmp_path: Path) -> None:
    # A bare archive.zip is a valid input shape for the oracle.
    packet = _write_canonical_packet(tmp_path / "packet")
    archive = packet / "archive.zip"
    result = compile_packet(
        input_packet=archive,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )
    # The bare archive input has no runtime tree; contest profile will flag.
    assert any(
        "inflate_sh_missing" in b or "inflate_py_missing" in b
        for b in result.blockers
    )


def test_oracle_inspect_wrapper_carries_profile_metadata(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    base = inspect_packet_oracle(packet, target_profile="production_generalized")
    assert base["deterministic_compiler_target_profile"] == "production_generalized"
    assert (
        base["deterministic_compiler_target_profile_policy"][
            "allows_optional_device_learning"
        ]
        is False
    )


def test_oracle_inspect_wrapper_rejects_unknown_profile(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    with pytest.raises(DeterministicPacketCompilerError):
        inspect_packet_oracle(packet, target_profile="bogus")  # type: ignore[arg-type]


def test_golden_vectors_emitted_for_every_mode_profile(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    for i, profile in enumerate(TARGET_PROFILES):
        out = tmp_path / f"out_{i}_{profile}"
        result = compile_packet(
            input_packet=packet,
            output_dir=out,
            mode="identity",
            target_profile=profile,
        )
        manifest = json.loads(
            (Path(result.output_dir) / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        gv = manifest["golden_vectors"]
        assert gv["schema_version"] == "deterministic_golden_vectors.v1"
        assert gv["target_profile"] == profile
        assert gv["mode"] == "identity"
        # At least one member vector for the renderer.bin payload.
        assert any(m["name"] == "renderer.bin" for m in gv["member_vectors"])


def test_parser_section_manifest_includes_all_members(tmp_path: Path) -> None:
    packet = _write_canonical_packet(tmp_path / "packet")
    result = compile_packet(
        input_packet=packet,
        output_dir=tmp_path / "out",
        mode="identity",
        target_profile="contest_one_video_replay",
    )
    psm = result.parser_section_manifest
    assert psm["section_count"] == 2
    assert "renderer.bin" in psm["section_names"]
    assert "masks.mkv" in psm["section_names"]
    assert all(isinstance(s, str) and len(s) == 64 for s in psm["section_sha256s"])


def test_compiler_modes_constant() -> None:
    assert COMPILER_MODES == ("identity", "canonicalize", "optimize")


def test_target_profiles_constant() -> None:
    assert TARGET_PROFILES == (
        "contest_one_video_replay",
        "contest_generalized",
        "production_generalized",
        "production_edge_adaptive",
    )


def test_packetir_operation_set_bridge_contract_is_fail_closed() -> None:
    contract = packetir_operation_set_bridge_contract()

    assert contract["schema"] == PACKET_IR_OPERATION_SET_BRIDGE_CONTRACT_SCHEMA
    assert contract["canonical_packet_compiler_module"] == (
        "tac.packet_compiler.deterministic_compiler"
    )
    assert contract["canonical_packet_compiler_schema"] == SCHEMA_VERSION
    assert contract["recommended_ir_schema"] == PACKET_IR_OPERATION_SET_SCHEMA
    assert contract["required_order"] == list(DETERMINISTIC_COMPILER_REQUIRED_ORDER)
    assert contract["required_proofs"] == list(PACKET_IR_OPERATION_SET_REQUIRED_PROOFS)
    assert "runtime_consumption_proof" in contract["required_proofs"]
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert contract[key] is value
