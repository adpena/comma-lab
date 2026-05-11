"""Tests for the Phase 1 packet compiler.

Coverage:

* Identity mode preserves byte-for-byte input archive (synthetic + A1
  canonical when present).
* Canonicalize mode preserves payload SHAs, reports re-emit metadata.
* Optimize mode requires score_affecting_payload_changed=True + baseline
  SHA + size.
* Fail-closed gates: hidden sidecar, scorer-at-inflate token, external-state
  path, network token in inflate.sh, missing positional args, unsafe ZIP
  member, duplicate ZIP members, unsupported ZIP method, public_pr*_intake_
  output dir, empty archive, /tmp output path, symlink in tree.
* HNeRV-parity manifest declares all 8 required fields.
* No-op proof correctly captures sha-changed / size-delta / runtime-consumes
  flags.
* Empty checkpoint → explicit error (not silent empty packet).
* Smoke: build_arg_parser CLI flag set is consistent with the API.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from tac.phase1_packet_compiler import (
    A1_CANONICAL_ARCHIVE_SHA256,
    A1_CANONICAL_ARCHIVE_SIZE_BYTES,
    ALLOWED_ZIP_METHODS,
    COMPILER_MODES,
    DETERMINISTIC_ZIP_DATE_TIME,
    FORBIDDEN_INFLATE_TOKENS,
    FORBIDDEN_NETWORK_TOKENS,
    FORBIDDEN_PYTHON_RUNTIME_MODULES,
    HNERV_PARITY_FIELDS,
    Phase1PacketCompilerError,
    Phase1PacketResult,
    SCHEMA_VERSION,
    TARGET_MODES,
    compile_phase1_packet,
)
from tac.repo_io import sha256_bytes, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
A1_CANONICAL_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex"
    / "harvested_artifacts/finetuned_archive/archive.zip"
)
A1_CANONICAL_PACKET_DIR = (
    REPO_ROOT
    / "experiments/results/track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex"
    / "harvested_artifacts/finetuned_archive/submission_dir"
)


def _load_contest_auth_eval_module():
    path = REPO_ROOT / "experiments" / "contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location(
        "contest_auth_eval_for_phase1_tests",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic packet fixture builders
# ---------------------------------------------------------------------------


def _write_minimal_inflate_sh(packet_dir: Path) -> None:
    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        'while IFS= read -r line; do\n'
        '  [ -z "$line" ] && continue\n'
        '  BASE="${line%.*}"\n'
        '  SRC="${DATA_DIR}/x"\n'
        '  DST="${OUTPUT_DIR}/${BASE}.raw"\n'
        '  "${PYTHON:-python3}" "$HERE/inflate.py" "$SRC" "$DST"\n'
        'done < "$FILE_LIST"\n'
    )
    (packet_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (packet_dir / "inflate.sh").chmod(0o755)


def _write_minimal_inflate_py(packet_dir: Path) -> None:
    inflate_py = (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "src, dst = sys.argv[1], sys.argv[2]\n"
        "data = Path(src).read_bytes()\n"
        "Path(dst).write_bytes(data)\n"
    )
    (packet_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _write_synthetic_archive(packet_dir: Path, payload: bytes = b"PHASE1_FAKE_PAYLOAD") -> str:
    archive_path = packet_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", date_time=DETERMINISTIC_ZIP_DATE_TIME)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = (0o100000 | 0o644) << 16
        info.create_system = 3
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    return sha256_file(archive_path)


def _write_synthetic_packet(
    tmp_path: Path,
    *,
    payload: bytes = b"PHASE1_FAKE_PAYLOAD",
    name: str = "synthetic_packet",
) -> Path:
    packet_dir = tmp_path / name
    packet_dir.mkdir(parents=True, exist_ok=True)
    src_dir = packet_dir / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "codec.py").write_text("# Phase 1 codec stub\n", encoding="utf-8")
    (src_dir / "model.py").write_text("# Phase 1 model stub\n", encoding="utf-8")
    _write_synthetic_archive(packet_dir, payload=payload)
    _write_minimal_inflate_sh(packet_dir)
    _write_minimal_inflate_py(packet_dir)
    return packet_dir


# ---------------------------------------------------------------------------
# Identity mode tests
# ---------------------------------------------------------------------------


def test_identity_preserves_synthetic_archive_bytes(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    pre_archive_sha = sha256_file(packet_dir / "archive.zip")

    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )

    assert isinstance(result, Phase1PacketResult)
    assert result.mode == "identity"
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.archive_sha256 == pre_archive_sha
    post_archive_sha = sha256_file(out_dir / "archive.zip")
    assert post_archive_sha == pre_archive_sha


def test_runtime_fallback_archive_zip_does_not_satisfy_consumption_proof(
    tmp_path: Path,
) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    fallback_py = (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "archive_dir = Path(sys.argv[1])\n"
        "output_dir = Path(sys.argv[2])\n"
        "file_list = Path(sys.argv[3])\n"
        "data = (Path(__file__).resolve().parent / 'archive.zip').read_bytes()\n"
        "for line in file_list.read_text().splitlines():\n"
        "    if not line.strip():\n"
        "        continue\n"
        "    dst = output_dir / (line.rsplit('.', 1)[0] + '.raw')\n"
        "    dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "    dst.write_bytes(data)\n"
    )
    (packet_dir / "inflate.py").write_text(fallback_py, encoding="utf-8")
    (packet_dir / "inflate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "DATA_DIR=\"$1\"\n"
        "OUTPUT_DIR=\"$2\"\n"
        "FILE_LIST=\"$3\"\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "exec \"${PYTHON:-python3}\" "
        "\"$HERE/inflate.py\" \"$DATA_DIR\" \"$OUTPUT_DIR\" \"$FILE_LIST\"\n",
        encoding="utf-8",
    )
    (packet_dir / "inflate.sh").chmod(0o755)

    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=tmp_path / "out",
        mode="identity",
    )

    assert any(
        blocker.startswith("inflate_does_not_consume_archive_bytes:")
        for blocker in result.blockers
    )


def test_identity_preserves_runtime_tree_files(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )

    relpaths = {row["relpath"] for row in result.runtime_files}
    assert "inflate.sh" in relpaths
    assert "inflate.py" in relpaths
    assert "src/codec.py" in relpaths
    assert "src/model.py" in relpaths


def test_identity_writes_build_manifest_and_no_op_proof(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert (out_dir / "build_manifest.json").is_file()
    assert (out_dir / "no_op_proof.json").is_file()
    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["lane_class"] == "substrate_engineering"
    assert manifest["mode"] == "identity"


def test_build_manifest_does_not_publish_stale_final_runtime_tree_hash(
    tmp_path: Path,
) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )

    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    cae = _load_contest_auth_eval_module()
    auth_runtime = cae._runtime_dependency_manifest(
        out_dir / "inflate.sh",
        REPO_ROOT / "upstream",
        repo_root=REPO_ROOT,
    )
    auth_files = {entry["relative_path"] for entry in auth_runtime["files"]}

    assert {"build_manifest.json", "no_op_proof.json"}.issubset(auth_files)
    assert manifest["runtime_tree_sha256"] == ""
    assert manifest["runtime_tree_sha256_status"] == (
        "withheld_self_referential_final_manifest_hash_use_contest_auth_eval"
    )
    assert manifest["pre_manifest_runtime_tree_sha256"] == result.runtime_tree_sha256
    assert manifest["pre_manifest_runtime_tree_sha256"] != auth_runtime[
        "runtime_tree_sha256"
    ]


@pytest.mark.skipif(
    not A1_CANONICAL_ARCHIVE.is_file(),
    reason="A1 canonical archive not present in this checkout",
)
def test_identity_preserves_a1_canonical_archive_bytes(tmp_path: Path) -> None:
    """Identity mode preserves the A1 canonical archive bit-identically."""
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=A1_CANONICAL_PACKET_DIR,
        output_dir=out_dir,
        mode="identity",
    )
    assert result.archive_sha256 == A1_CANONICAL_ARCHIVE_SHA256
    assert result.archive_size_bytes == A1_CANONICAL_ARCHIVE_SIZE_BYTES


def test_identity_refuses_score_affecting_change_kwarg(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="forbids score_affecting_payload_changed"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
            score_affecting_payload_changed=True,
        )


def test_identity_refuses_existing_nonempty_output_dir(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(Phase1PacketCompilerError, match="not empty"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
        )


def test_identity_replaces_existing_output_when_allowed(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
        allow_existing_output_dir=True,
    )
    assert result.score_claim is False
    assert not (out_dir / "stale.txt").exists()


# ---------------------------------------------------------------------------
# Canonicalize mode tests
# ---------------------------------------------------------------------------


def test_canonicalize_preserves_payload_member_sha(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    pre_member_sha = sha256_bytes(b"PHASE1_FAKE_PAYLOAD")

    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="canonicalize",
    )
    assert result.mode == "canonicalize"
    members_by_name = {row["name"]: row for row in result.archive_members}
    assert members_by_name["x"]["sha256"] == pre_member_sha


def test_canonicalize_records_baseline_sha_and_size(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    pre_archive_sha = sha256_file(packet_dir / "archive.zip")
    pre_archive_size = (packet_dir / "archive.zip").stat().st_size

    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="canonicalize",
    )
    assert result.no_op_proof["baseline_archive_sha256"] == pre_archive_sha
    assert result.no_op_proof["baseline_archive_size_bytes"] == pre_archive_size


def test_canonicalize_refuses_score_affecting_change_kwarg(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="forbids score_affecting_payload_changed"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="canonicalize",
            score_affecting_payload_changed=True,
        )


# ---------------------------------------------------------------------------
# Optimize mode tests
# ---------------------------------------------------------------------------


def test_optimize_requires_score_affecting_change_flag(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="optimize mode requires score_affecting_payload_changed=True"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            baseline_archive_sha256="0" * 64,
            baseline_archive_size_bytes=12345,
            score_affecting_payload_changed=False,
        )


def test_optimize_requires_baseline_archive_sha256(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="baseline_archive_sha256"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            score_affecting_payload_changed=True,
            baseline_archive_size_bytes=12345,
        )


def test_optimize_requires_baseline_archive_size_bytes(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="baseline_archive_size_bytes"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            score_affecting_payload_changed=True,
            baseline_archive_sha256="0" * 64,
        )


def test_optimize_records_old_new_sha_delta(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="optimize",
        score_affecting_payload_changed=True,
        baseline_archive_sha256="0" * 64,
        baseline_archive_size_bytes=99999,
    )
    proof = result.no_op_proof
    assert proof["score_affecting_payload_changed"] is True
    assert proof["baseline_archive_sha256"] == "0" * 64
    assert proof["baseline_archive_size_bytes"] == 99999
    assert proof["new_archive_sha256"] == result.archive_sha256
    assert proof["sha_changed"] is True
    assert proof["byte_delta"] == result.archive_size_bytes - 99999


# ---------------------------------------------------------------------------
# Fail-closed gates
# ---------------------------------------------------------------------------


def test_fail_closed_on_missing_inflate_sh(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "inflate.sh").unlink()
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert "inflate_sh_missing" in result.blockers


def test_fail_closed_on_missing_inflate_py(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "inflate.py").unlink()
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert "inflate_py_missing" in result.blockers


def test_fail_closed_on_scorer_at_inflate_token_in_sh(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    text = (packet_dir / "inflate.sh").read_text()
    text += "# bad: PoseNet at inflate time\n"
    (packet_dir / "inflate.sh").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any("PoseNet" in b for b in result.blockers)


def test_fail_closed_on_network_token_in_sh(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    text = (packet_dir / "inflate.sh").read_text()
    text += "# bad: curl http://...\ncurl http://example.com\n"
    (packet_dir / "inflate.sh").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any("curl " in b for b in result.blockers)
    assert any("http://" in b for b in result.blockers)


def test_fail_closed_on_network_token_in_inflate_py(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    text = (packet_dir / "inflate.py").read_text()
    text += (
        "\n# bad: runtime dependency fetch from inflate.py\n"
        "import subprocess\n"
        "subprocess.run(['uv', 'pip', 'install', 'brotli'], check=False)\n"
        "MODEL_URL = 'https://example.invalid/model.pt'\n"
    )
    (packet_dir / "inflate.py").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any(
        "inflate.py" in b and "'uv', 'pip', 'install'" in b
        for b in result.blockers
    )
    assert any("inflate.py" in b and "https://" in b for b in result.blockers)
    assert any(
        "inflate.py" in b and "Python runtime literal contains external URL" in b
        for b in result.blockers
    )


def test_fail_closed_on_python_network_import_in_inflate_py(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    text = (packet_dir / "inflate.py").read_text()
    text += (
        "\nimport requests as req\n"
        "from urllib.request import urlopen\n"
        "def _unused_fetch():\n"
        "    return req.get('file://local'), urlopen\n"
    )
    (packet_dir / "inflate.py").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
        runtime_dep_closure=["requests", "urllib"],
    )
    assert any(
        "inflate.py" in b and "forbidden Python runtime module 'requests'" in b
        for b in result.blockers
    )
    assert any(
        "inflate.py" in b and "forbidden Python runtime module 'urllib.request'" in b
        for b in result.blockers
    )


def test_fail_closed_on_python_package_install_command_in_inflate_py(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    text = (packet_dir / "inflate.py").read_text()
    text += (
        "\nimport subprocess\n"
        "installer = [\n"
        "    'pip',\n"
        "    'install',\n"
        "    'brotli',\n"
        "]\n"
        "subprocess.run(installer, check=False)\n"
    )
    (packet_dir / "inflate.py").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any(
        "inflate.py" in b and "forbidden sequence 'pip install'" in b
        for b in result.blockers
    )


def test_fail_closed_on_network_import_in_runtime_python_tree(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "src" / "codec.py").write_text(
        "import socket\n"
        "def decode(data):\n"
        "    return data\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
        runtime_dep_closure=["socket"],
    )
    assert any(
        "src/codec.py" in b and "forbidden Python runtime module 'socket'" in b
        for b in result.blockers
    )


def test_fail_closed_on_uv_runtime_dependency_fetch_in_sh(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    text = (packet_dir / "inflate.sh").read_text()
    text += (
        'exec uv run --with brotli==1.1.0 --with torch==2.5.1+cu124 '
        '--extra-index-url https://download.pytorch.org/whl/cu124 '
        '"$HERE/inflate.py" "$@"\n'
    )
    (packet_dir / "inflate.sh").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    for token in ("uv run --with", "--extra-index-url", "https://"):
        assert any(token in b for b in result.blockers)


def test_fail_closed_on_network_token_in_runtime_shell_helper(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    helper = packet_dir / "helper.sh"
    helper.write_text("curl https://example.com/x -o /tmp/x\n", encoding="utf-8")
    text = (packet_dir / "inflate.sh").read_text()
    text += '\n"$HERE/helper.sh"\n'
    (packet_dir / "inflate.sh").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"

    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )

    assert any("helper.sh" in b and "curl " in b for b in result.blockers)
    assert any("helper.sh" in b and "https://" in b for b in result.blockers)


def test_fail_closed_on_executable_runtime_shell_helper(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    helper = packet_dir / "helper.sh"
    helper.write_text("echo helper\n", encoding="utf-8")
    helper.chmod(0o755)
    out_dir = tmp_path / "out"

    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )

    assert any(
        "runtime_file_mode_mismatch:helper.sh" in blocker
        for blocker in result.blockers
    )


def test_fail_closed_on_external_state_path_in_sh(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    text = (packet_dir / "inflate.sh").read_text()
    text += "DATA=/Users/agent/leak\n"
    (packet_dir / "inflate.sh").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any("/Users/" in b for b in result.blockers)


def test_fail_closed_on_missing_positional_args(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    bad_sh = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo 'no positional args here'\n"
    )
    (packet_dir / "inflate.sh").write_text(bad_sh, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any("missing required positional args" in b for b in result.blockers)


def test_fail_closed_on_missing_set_e(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    bad_sh = (
        "#!/usr/bin/env bash\n"
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'echo "$DATA_DIR $OUTPUT_DIR $FILE_LIST"\n'
    )
    (packet_dir / "inflate.sh").write_text(bad_sh, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any("set -e" in b for b in result.blockers)


def test_fail_closed_on_inflate_sh_loc_budget(tmp_path: Path) -> None:
    """Round 1 Fridrich HIGH fix: LOC budget excludes blank + comment lines, so
    the test must add real shell statements (not just comments)."""
    packet_dir = _write_synthetic_packet(tmp_path)
    bloated_sh = (packet_dir / "inflate.sh").read_text()
    bloated_sh += "\n".join([f'echo "padding line {i}"' for i in range(200)]) + "\n"
    (packet_dir / "inflate.sh").write_text(bloated_sh, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any("inflate_sh_loc_" in b and "exceeds_budget_100" in b for b in result.blockers)


def test_inflate_sh_loc_excludes_comment_and_blank_lines(tmp_path: Path) -> None:
    """Round 1 Fridrich HIGH fix verification: 200 comment-only lines must
    NOT trigger the LOC budget."""
    packet_dir = _write_synthetic_packet(tmp_path)
    text = (packet_dir / "inflate.sh").read_text()
    text += "\n".join(["# comment-only line"] * 200) + "\n"
    text += "\n" * 50
    (packet_dir / "inflate.sh").write_text(text, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert not any(
        "inflate_sh_loc_" in b and "exceeds_budget_100" in b for b in result.blockers
    ), f"comment-only padding triggered LOC budget: {result.blockers}"


def test_fail_closed_on_unsafe_zip_member(tmp_path: Path) -> None:
    packet_dir = tmp_path / "bad_packet"
    packet_dir.mkdir()
    src_dir = packet_dir / "src"
    src_dir.mkdir()
    (src_dir / "codec.py").write_text("# stub\n")
    (src_dir / "model.py").write_text("# stub\n")
    archive_path = packet_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("../oops_traversal", date_time=DETERMINISTIC_ZIP_DATE_TIME)
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, b"x")
    _write_minimal_inflate_sh(packet_dir)
    _write_minimal_inflate_py(packet_dir)

    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="unsafe member name"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
        )


def test_fail_closed_on_zero_member_archive(tmp_path: Path) -> None:
    """Round 9 Tao 2nd HIGH fix verification: a ZIP with zero members (just
    central directory + EOCD ≈ 22 bytes) is non-empty as a file, but is a
    semantic no-op packet. Refuse it as well as the file-size-zero case."""
    packet_dir = tmp_path / "zero_member"
    packet_dir.mkdir()
    src = packet_dir / "src"
    src.mkdir()
    (src / "codec.py").write_text("# stub\n")
    (src / "model.py").write_text("# stub\n")
    archive = packet_dir / "archive.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    _write_minimal_inflate_sh(packet_dir)
    _write_minimal_inflate_py(packet_dir)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="zero members"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
        )


def test_fail_closed_on_empty_archive(tmp_path: Path) -> None:
    packet_dir = tmp_path / "empty_packet"
    packet_dir.mkdir()
    (packet_dir / "archive.zip").write_bytes(b"")
    _write_minimal_inflate_sh(packet_dir)
    _write_minimal_inflate_py(packet_dir)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="empty"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
        )


def test_fail_closed_on_missing_input(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError):
        compile_phase1_packet(
            input_packet=tmp_path / "does_not_exist",
            output_dir=out_dir,
            mode="identity",
        )


def test_fail_closed_on_unknown_mode(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="unknown mode"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="not_a_mode",  # type: ignore[arg-type]
        )


def test_fail_closed_on_unknown_target_mode(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="unknown target_mode"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
            target_mode="not_a_target",  # type: ignore[arg-type]
        )


def test_fail_closed_on_public_pr_intake_clone_output(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "experiments_results_public_pr101_hnerv_intake_codex" / "out"
    # The path itself contains the intake marker: rename to match exactly.
    intake_dir = tmp_path / "public_pr101_hnerv_intake_codex"
    intake_dir.mkdir()
    out_dir = intake_dir / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert any(
        "output_dir_inside_public_pr_intake_clone_forbidden" in b
        for b in result.blockers
    )


def test_fail_closed_on_unsupported_zip_method(tmp_path: Path) -> None:
    packet_dir = tmp_path / "bzip2_packet"
    packet_dir.mkdir()
    src_dir = packet_dir / "src"
    src_dir.mkdir()
    (src_dir / "codec.py").write_text("# stub\n")
    (src_dir / "model.py").write_text("# stub\n")
    archive_path = packet_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_BZIP2) as zf:
        info = zipfile.ZipInfo("x", date_time=DETERMINISTIC_ZIP_DATE_TIME)
        info.compress_type = zipfile.ZIP_BZIP2
        zf.writestr(info, b"x" * 256, compress_type=zipfile.ZIP_BZIP2)
    _write_minimal_inflate_sh(packet_dir)
    _write_minimal_inflate_py(packet_dir)

    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="unsupported ZIP compression"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
        )


def test_fail_closed_on_symlink_in_runtime_tree(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    target = packet_dir / "src" / "model.py"
    link = packet_dir / "src" / "model_link.py"
    if link.exists():
        link.unlink()
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="symlink"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
        )


# ---------------------------------------------------------------------------
# HNeRV parity manifest + no_op_proof fields
# ---------------------------------------------------------------------------


def test_hnerv_parity_manifest_declares_all_8_fields(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    parity = result.hnerv_parity_manifest
    for field in HNERV_PARITY_FIELDS:
        assert field in parity, f"HNeRV parity field missing: {field}"


def test_no_op_proof_records_runtime_consumes_payload_bytes(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    proof = result.no_op_proof
    assert proof["runtime_consumption_proof"] is True
    assert proof["score_affecting_payload_changed"] is False
    assert proof["sha_changed"] is False


def test_no_op_proof_when_runtime_tree_missing_inflate_py(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "inflate.py").unlink()
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert result.no_op_proof["runtime_consumption_proof"] is False
    assert "inflate_py_missing" in result.blockers


def test_no_op_proof_runtime_consumes_false_when_inflate_py_does_not_read(tmp_path: Path) -> None:
    """Round 1 Yousfi HIGH fix verification: a malicious empty inflate.py
    with no read patterns must not pass runtime_consumption_proof."""
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "inflate.py").write_text(
        "# This inflate does not actually read the archive\n"
        "import sys\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    assert result.no_op_proof["runtime_consumption_proof"] is False


def test_optimize_refuses_when_new_sha_matches_baseline(tmp_path: Path) -> None:
    """Round 1 Hotz MEDIUM fix verification: optimize mode asserts payload
    bytes changed, but a packet whose new archive SHA equals
    baseline_archive_sha256 contradicts that. Refuse, don't ship the
    contradiction.
    """
    packet_dir = _write_synthetic_packet(tmp_path)
    pre_sha = sha256_file(packet_dir / "archive.zip")
    pre_size = (packet_dir / "archive.zip").stat().st_size
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="matches baseline_archive_sha256"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="optimize",
            score_affecting_payload_changed=True,
            baseline_archive_sha256=pre_sha,
            baseline_archive_size_bytes=pre_size,
        )


def test_undeclared_runtime_dep_in_inflate_py_is_blocker(tmp_path: Path) -> None:
    """Round 1 Selfcomp MEDIUM fix verification: if inflate.py imports
    `compressai` but operator only declares `["torch"]`, blocker fires.
    """
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "inflate.py").write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "import compressai\n"
        "from pathlib import Path\n"
        "src, dst = sys.argv[1], sys.argv[2]\n"
        "data = Path(src).read_bytes()\n"
        "Path(dst).write_bytes(data)\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
        runtime_dep_closure=["torch"],
    )
    assert any("undeclared_runtime_dep" in b and "compressai" in b for b in result.blockers)


def test_declared_runtime_dep_in_inflate_py_passes(tmp_path: Path) -> None:
    """Sister test to ensure declared deps don't false-positive."""
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "inflate.py").write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "import compressai\n"
        "import torch\n"
        "from pathlib import Path\n"
        "src, dst = sys.argv[1], sys.argv[2]\n"
        "data = Path(src).read_bytes()\n"
        "Path(dst).write_bytes(data)\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
        runtime_dep_closure=["torch", "compressai"],
    )
    assert not any("undeclared_runtime_dep" in b for b in result.blockers)


def test_missing_packet_local_tac_import_is_blocker(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "src" / "model.py").write_text(
        "from tac.paradigm_delta_epsilon_zeta.decoder_128k import Decoder128K\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
        runtime_dep_closure=["torch"],
    )

    assert any("undeclared_runtime_dep" in b and "tac" in b for b in result.blockers)


def test_packet_local_tac_import_passes_without_declaring_repo_tac(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    local_pkg = packet_dir / "src" / "tac" / "paradigm_delta_epsilon_zeta"
    local_pkg.mkdir(parents=True)
    (packet_dir / "src" / "tac" / "__init__.py").write_text("", encoding="utf-8")
    (local_pkg / "__init__.py").write_text("", encoding="utf-8")
    (local_pkg / "decoder_128k.py").write_text("class Decoder128K: pass\n", encoding="utf-8")
    (packet_dir / "src" / "model.py").write_text(
        "from tac.paradigm_delta_epsilon_zeta.decoder_128k import Decoder128K\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
        runtime_dep_closure=["torch"],
    )

    assert not any("undeclared_runtime_dep" in b and "tac" in b for b in result.blockers)


def test_repo_local_runtime_search_token_is_blocker(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    (packet_dir / "src" / "model.py").write_text(
        "from pathlib import Path\n"
        "def _find_repo_src():\n"
        "    here = Path(__file__).resolve()\n"
        "    for parent in here.parents:\n"
        "        candidate = parent / 'src' / 'tac'\n"
        "        if candidate.is_dir():\n"
        "            return candidate.parent\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )

    assert any("src/model.py" in b and "_find_repo_src" in b for b in result.blockers)
    assert any("src/model.py" in b and "for parent in here.parents" in b for b in result.blockers)


def test_parser_section_manifest_declares_required_fields(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    manifest = result.parser_section_manifest
    for field in (
        "section_count",
        "section_names",
        "lengths",
        "section_sha256s",
        "offsets",
        "entropy_estimates",
        "old_new_section_boundaries",
    ):
        assert field in manifest, f"parser_section_manifest missing field: {field}"
    assert "three-member ZIP grammar" in manifest["old_new_section_boundaries"]


def test_hnerv_parity_manifest_declares_three_member_phase1_grammar(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )

    assert (
        result.hnerv_parity_manifest["archive_grammar"]
        == "Phase1-three-member-x-decoder-bin-balle-bin"
    )
    assert (
        result.hnerv_parity_manifest["export_format"]
        == "phase1_three_member_x_decoder_bin_balle_bin"
    )


def test_executable_no_op_smoke_uses_inflate_sh_and_nonempty_file_list(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"

    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )

    smoke = result.no_op_proof["executable_smoke"]
    assert smoke["executable_smoke_attempted"] is True
    assert smoke["executable_smoke_passed"] is True
    assert "byte_mutation_changed" in smoke["executable_smoke_reason"]


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_identity_roundtrip_archive_member_sha_preserved(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path, payload=b"X" * 8192)
    pre_member_sha = sha256_bytes(b"X" * 8192)

    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    members_by_name = {row["name"]: row for row in result.archive_members}
    assert members_by_name["x"]["sha256"] == pre_member_sha


def test_identity_then_canonicalize_preserves_member_sha(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir1 = tmp_path / "out_identity"
    r1 = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir1,
        mode="identity",
    )
    out_dir2 = tmp_path / "out_canonicalize"
    r2 = compile_phase1_packet(
        input_packet=out_dir1,
        output_dir=out_dir2,
        mode="canonicalize",
    )
    m1 = {row["name"]: row["sha256"] for row in r1.archive_members}
    m2 = {row["name"]: row["sha256"] for row in r2.archive_members}
    assert m1 == m2


# ---------------------------------------------------------------------------
# Smoke: archive-only input path + CLI parser
# ---------------------------------------------------------------------------


def test_input_can_be_bare_archive_zip(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir / "archive.zip",
        output_dir=out_dir,
        mode="identity",
    )
    assert result.score_claim is False


def test_cli_parser_flag_set_matches_api_kwargs() -> None:
    """Per CLAUDE.md NEVER invent CLI flags rule: every flag in the CLI
    must map to an actual API kwarg in compile_phase1_packet (no dead flags).
    """
    cli_path = REPO_ROOT / "tools" / "build_phase1_packet_compiler.py"
    spec = importlib.util.spec_from_file_location("build_phase1_packet_compiler_cli", cli_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parser = mod.build_arg_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    cli_dests = {action.dest for action in parser._actions if action.dest != "help"}
    # Every CLI dest should map onto either a compile_phase1_packet kwarg or
    # an output-side flag (output_dir / print_result_json) we explicitly add.
    api_kwargs = {
        "mode",
        "target_mode",
        "runtime_dep_closure",
        "export_format",
        "bolt_on_loc_budget",
        "allow_existing_output_dir",
        "score_affecting_payload_changed",
        "baseline_archive_sha256",
        "baseline_archive_size_bytes",
        "fail_on_score_affecting_change",
        "packet_compiler_transforms",
    }
    cli_only_extras = {"input_packet", "output_dir", "print_result_json"}
    expected = api_kwargs | cli_only_extras
    unexpected = cli_dests - expected
    assert not unexpected, f"CLI has dead flags not in API: {unexpected}"


def test_cli_parser_defaults_to_phase1_three_member_export_format() -> None:
    cli_path = REPO_ROOT / "tools" / "build_phase1_packet_compiler.py"
    spec = importlib.util.spec_from_file_location("build_phase1_packet_compiler_cli", cli_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    args = mod.build_arg_parser().parse_args(["--input-packet", "packet_dir"])

    assert args.export_format == "phase1_three_member_x_decoder_bin_balle_bin"


def test_compiler_modes_constant_matches_documented_set() -> None:
    assert COMPILER_MODES == ("identity", "canonicalize", "optimize")


def test_target_modes_constant_matches_documented_set() -> None:
    assert TARGET_MODES == (
        "contest_one_video_replay",
        "contest_generalized",
    )


def test_allowed_zip_methods_is_stored_or_deflated_only() -> None:
    assert ALLOWED_ZIP_METHODS == frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


def test_forbidden_inflate_tokens_includes_scorer_classes() -> None:
    for token in ("PoseNet", "SegNet", "rgb_to_yuv6"):
        assert token in FORBIDDEN_INFLATE_TOKENS


def test_forbidden_network_tokens_includes_curl_wget() -> None:
    assert "curl " in FORBIDDEN_NETWORK_TOKENS
    assert "wget " in FORBIDDEN_NETWORK_TOKENS
    assert "uv run --with" in FORBIDDEN_NETWORK_TOKENS
    assert "--extra-index-url" in FORBIDDEN_NETWORK_TOKENS
    assert "https://" in FORBIDDEN_NETWORK_TOKENS


def test_forbidden_python_runtime_modules_include_network_install_surfaces() -> None:
    for module in ("requests", "urllib", "httpx", "socket", "pip", "ensurepip"):
        assert module in FORBIDDEN_PYTHON_RUNTIME_MODULES


def test_a1_canonical_constants_match_designation() -> None:
    assert A1_CANONICAL_ARCHIVE_SIZE_BYTES == 178262
    assert len(A1_CANONICAL_ARCHIVE_SHA256) == 64
    assert A1_CANONICAL_ARCHIVE_SHA256.startswith("87ec7ca5")


def test_schema_version_is_v1() -> None:
    assert SCHEMA_VERSION == "phase1_packet_compiler.v1"


def test_runtime_tree_sha256_changes_when_inflate_text_changes(tmp_path: Path) -> None:
    packet_a = _write_synthetic_packet(tmp_path, name="a")
    packet_b = _write_synthetic_packet(tmp_path, name="b")
    text = (packet_b / "inflate.sh").read_text()
    text += "# DIFFERENT\n"
    (packet_b / "inflate.sh").write_text(text, encoding="utf-8")

    r_a = compile_phase1_packet(
        input_packet=packet_a,
        output_dir=tmp_path / "out_a",
        mode="identity",
    )
    r_b = compile_phase1_packet(
        input_packet=packet_b,
        output_dir=tmp_path / "out_b",
        mode="identity",
    )
    assert r_a.runtime_tree_sha256 != r_b.runtime_tree_sha256


def test_identity_archive_modes_normalized_to_644(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    # Set the source archive to an unusual mode; identity copy normalises.
    (packet_dir / "archive.zip").chmod(0o600)
    out_dir = tmp_path / "out"
    compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    out_archive_mode = stat.S_IMODE((out_dir / "archive.zip").stat().st_mode)
    assert out_archive_mode == 0o644


def test_inflate_sh_executable_in_output(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    sh_mode = stat.S_IMODE((out_dir / "inflate.sh").stat().st_mode)
    assert sh_mode == 0o755


def test_build_manifest_records_lane_class_substrate_engineering(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    assert manifest["lane_class"] == "substrate_engineering"


def test_build_manifest_records_evidence_grade_byte_custody_only(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    assert manifest["evidence_grade"] == "byte_custody_only"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_optimize_no_op_proof_no_op_detector_passes_when_payload_unchanged(tmp_path: Path) -> None:
    """Sanity: the no_op detector should pass when payload SHA matches the
    declared baseline (i.e. the operator claimed no payload change AND the
    SHA agrees). For optimize mode this is achieved by setting the baseline
    SHA to the new SHA — we DON'T do this, because optimize mode requires
    score_affecting_payload_changed=True. So this check is via canonicalize.
    """
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="canonicalize",
    )
    proof = result.no_op_proof
    assert proof["sha_changed"] is False
    assert proof["no_op_detector_passed"] is True


def test_canonicalize_emits_deterministic_zip_timestamps(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="canonicalize",
    )
    for member in result.archive_members:
        assert member["date_time"] == list(DETERMINISTIC_ZIP_DATE_TIME)


def test_archive_members_are_sorted_deterministically(tmp_path: Path) -> None:
    packet_dir = tmp_path / "multi_member"
    packet_dir.mkdir()
    src_dir = packet_dir / "src"
    src_dir.mkdir()
    (src_dir / "codec.py").write_text("# stub\n")
    (src_dir / "model.py").write_text("# stub\n")
    archive_path = packet_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in ["zzz", "aaa", "mmm"]:
            info = zipfile.ZipInfo(name, date_time=DETERMINISTIC_ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, name.encode("utf-8") * 16)
    _write_minimal_inflate_sh(packet_dir)
    _write_minimal_inflate_py(packet_dir)

    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    names = [row["name"] for row in result.archive_members]
    assert names == sorted(names)


def test_identity_preserves_inflate_sh_byte_size(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    pre_size = (packet_dir / "inflate.sh").stat().st_size
    out_dir = tmp_path / "out"
    compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    post_size = (out_dir / "inflate.sh").stat().st_size
    assert post_size == pre_size


def test_input_packet_dir_with_no_archive_zip_fails_closed(tmp_path: Path) -> None:
    bad_dir = tmp_path / "no_archive"
    bad_dir.mkdir()
    (bad_dir / "inflate.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="missing archive.zip"):
        compile_phase1_packet(
            input_packet=bad_dir,
            output_dir=out_dir,
            mode="identity",
        )


def test_identity_creates_build_manifest_with_8_hnerv_fields_at_top_level(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    parity = manifest["hnerv_parity_manifest"]
    for field in HNERV_PARITY_FIELDS:
        assert field in parity, f"build_manifest.hnerv_parity_manifest missing {field}"


def test_intra_member_section_manifest_folded_when_present(tmp_path: Path) -> None:
    """Round 2 Shannon HIGH fix verification: when the trainer's
    archive_section_manifest.json sidecar is present, the parser_section
    manifest folds it in.
    """
    packet_dir = _write_synthetic_packet(tmp_path)
    sidecar = {
        "schema_version": "phase1_intra_member_sections.v1",
        "section_names": ["balle_strings", "decoder_state_dict", "balle_state_dict"],
        "section_lengths": [1000, 50000, 8000],
        "section_offsets": [4, 1004, 51008],
    }
    (packet_dir / "archive_section_manifest.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    parser_manifest = result.parser_section_manifest
    assert parser_manifest["intra_member_section_manifest_present"] is True
    assert (
        parser_manifest["intra_member_section_manifest"]["section_names"]
        == sidecar["section_names"]
    )


def test_intra_member_section_manifest_absent_when_no_sidecar(tmp_path: Path) -> None:
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    parser_manifest = result.parser_section_manifest
    assert parser_manifest["intra_member_section_manifest_present"] is False


def test_inflate_sh_bash_n_check_recorded_in_manifest(tmp_path: Path) -> None:
    """Round 2 Contrarian MEDIUM fix verification: bash -n result is
    recorded in inflate_sh_info."""
    packet_dir = _write_synthetic_packet(tmp_path)
    out_dir = tmp_path / "out"
    compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    bash_n = manifest["inflate_sh_info"]["bash_n"]
    assert "attempted" in bash_n
    if bash_n["attempted"]:
        assert bash_n["passed"] is True


def test_fail_closed_on_inflate_sh_syntax_error(tmp_path: Path) -> None:
    """Round 2 Contrarian MEDIUM fix verification: bash -n catches syntax
    errors as a blocker (rather than letting the operator burn $80 on a
    script that won't parse)."""
    packet_dir = _write_synthetic_packet(tmp_path)
    bad_sh = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        "if [ -z\n"  # missing closing ] / fi
    )
    (packet_dir / "inflate.sh").write_text(bad_sh, encoding="utf-8")
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    # bash -n may or may not be present; if present, should catch this.
    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    bash_n = manifest["inflate_sh_info"]["bash_n"]
    if bash_n.get("attempted"):
        assert any("bash -n parse-only" in b for b in result.blockers)


def test_fail_closed_on_top_level_symlink_child(tmp_path: Path) -> None:
    """Round 3 Tao MEDIUM fix verification: refuse if any direct child of
    the packet dir is a symlink (a malicious caller could symlink
    ``src/`` to a different directory and exfiltrate state)."""
    packet_dir = _write_synthetic_packet(tmp_path)
    # Replace src/ with a symlink to a sister directory.
    other = tmp_path / "other_dir"
    other.mkdir()
    (other / "leak.py").write_text("# bad\n", encoding="utf-8")
    import shutil
    shutil.rmtree(packet_dir / "src")
    try:
        os.symlink(other, packet_dir / "src")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    out_dir = tmp_path / "out"
    with pytest.raises(Phase1PacketCompilerError, match="symlink"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=out_dir,
            mode="identity",
        )


def test_fail_closed_when_output_dir_equals_input_packet_dir(tmp_path: Path) -> None:
    """R10 Quantizr 3rd HIGH fix verification: refuse to write output into
    the input packet directory (would nuke the input on
    allow_existing_output_dir=True)."""
    packet_dir = _write_synthetic_packet(tmp_path)
    with pytest.raises(Phase1PacketCompilerError, match="same as input"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=packet_dir,
            mode="identity",
            allow_existing_output_dir=True,
        )


def test_fail_closed_when_output_dir_is_parent_of_input(tmp_path: Path) -> None:
    """R10 Quantizr 3rd HIGH fix verification: refuse if output_dir is a
    parent of the input packet dir."""
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    packet_dir = parent_dir / "packet"
    packet_dir.mkdir()
    src_dir = packet_dir / "src"
    src_dir.mkdir()
    (src_dir / "codec.py").write_text("# stub\n")
    (src_dir / "model.py").write_text("# stub\n")
    _write_synthetic_archive(packet_dir)
    _write_minimal_inflate_sh(packet_dir)
    _write_minimal_inflate_py(packet_dir)
    with pytest.raises(Phase1PacketCompilerError, match="parent of input"):
        compile_phase1_packet(
            input_packet=packet_dir,
            output_dir=parent_dir,
            mode="identity",
            allow_existing_output_dir=True,
        )


def test_compiler_rejects_trainer_scaffold_inflate_sh_signature(tmp_path: Path) -> None:
    """Round 4 Hassabis HIGH fix verification: the Phase 1 trainer's CURRENT
    scaffold output uses ``inflate.py <src.bin> <dst.raw>`` (single-file
    legacy contract) NOT the contest's ``inflate.sh <archive_dir>
    <output_dir> <file_list>``. The compiler MUST refuse this with a
    concrete blocker so the operator does not burn $80 on a packet the
    contest auth eval will reject.
    """
    packet_dir = tmp_path / "trainer_scaffold"
    packet_dir.mkdir()
    src_dir = packet_dir / "src"
    src_dir.mkdir()
    (src_dir / "codec.py").write_text("# Phase 1 codec stub\n", encoding="utf-8")
    (src_dir / "model.py").write_text("# Phase 1 model stub\n", encoding="utf-8")
    _write_synthetic_archive(packet_dir)
    # Verbatim copy of the trainer's current scaffold inflate.sh shape:
    bad_sh = (
        "#!/bin/bash\n"
        "# PACT_RUNTIME_DEPENDENCY_ROOT = src/tac\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'exec uv run --with torch==2.5.1+cu124 --extra-index-url '
        'https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match '
        '--with compressai==1.2.8 "$HERE/inflate.py" "$@"\n'
    )
    (packet_dir / "inflate.sh").write_text(bad_sh, encoding="utf-8")
    (packet_dir / "inflate.sh").chmod(0o755)
    _write_minimal_inflate_py(packet_dir)
    out_dir = tmp_path / "out"
    result = compile_phase1_packet(
        input_packet=packet_dir,
        output_dir=out_dir,
        mode="identity",
    )
    # The trainer scaffold's inflate.sh fetches from a network URL AND lacks
    # the contest's positional args. Both should be blockers.
    assert any("missing required positional args" in b for b in result.blockers)
    for token in ("uv run --with", "--extra-index-url", "https://"):
        assert any(token in b for b in result.blockers)


def test_six_hook_wire_in_declared_in_module_docstring() -> None:
    """Per Check #125 'Subagent coherence-by-default': the module docstring
    must declare all 6 wire-in hooks."""
    import tac.phase1_packet_compiler as mod
    doc = mod.__doc__ or ""
    for hook in (
        "Sensitivity-map",
        "Pareto frontier",
        "Bit-allocator",
        "Cathedral autopilot",
        "Continual-learning",
        "Probe-disambiguator",
    ):
        assert hook in doc, f"6-hook wire-in declaration missing: {hook}"
