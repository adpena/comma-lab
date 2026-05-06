# pyc-recovery pass2: rehydrated from git blob 9616858a3100cf3fadca1c72a858663c75591697 via `git fsck --lost-found`
# original path: src/tac/tests/test_preflight_public_replay_intake.py
# OUR source dropped during commit 66c59aae filter-repo cleanup; .pyc was sole orphan left.
# Blob verified intact + parses cleanly with python ast.
# Recovered: 2026-05-05 by Sherlock pass2
from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "preflight_public_replay_intake.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("preflight_public_replay_intake_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _det_bytes(label: str, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out.extend(hashlib.sha256(f"{label}:{counter}".encode("ascii")).digest())
        counter += 1
    return bytes(out[:n])


def _br(data: bytes) -> bytes:
    return brotli.compress(data, quality=5)


def _write_runtime(root: Path, *, embedded_payload: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    inflate_sh = root / "inflate.sh"
    inflate_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'python "$(dirname "$0")/inflate.py" "$@"\n',
        encoding="utf-8",
    )
    if embedded_payload:
        payload = "A" * 70000
        inflate_py = (
            "import base64\n"
            f"PAYLOAD = base64.b85decode({payload!r})\n"
            "print(len(PAYLOAD))\n"
        )
    else:
        inflate_py = "from pathlib import Path\nprint(Path(__file__).name)\n"
    (root / "inflate.py").write_text(inflate_py, encoding="utf-8")
    return inflate_sh


def _write_upstream(root: Path) -> Path:
    upstream = root / "upstream"
    upstream.mkdir(parents=True)
    (upstream / "evaluate.py").write_text("print('fixture evaluate')\n", encoding="utf-8")
    return upstream


def _write_x_archive(path: Path) -> None:
    mask_bitstream = _det_bytes("qma9-bitstream", 64)
    segments = {
        "mask": b"QMA9"
        + (600).to_bytes(4, "little")
        + (512).to_bytes(4, "little")
        + (384).to_bytes(4, "little")
        + len(mask_bitstream).to_bytes(4, "little")
        + mask_bitstream,
        "model": _br(b"QH0" + _det_bytes("model", 96)),
        "pose": _br(b"P1D1" + _det_bytes("pose", 64)),
        "post": _br(_det_bytes("post", 128)),
        "shift": _br(b"SD4" + _det_bytes("shift", 32)),
        "frac": _br(b"FV1" + _det_bytes("frac", 32)),
        "frac2": _br(b"FH2" + _det_bytes("frac2", 32)),
        "frac3": _br(b"FD3" + _det_bytes("frac3", 32)),
        "bias": _br(b"BD1" + _det_bytes("bias", 32)),
        "region": _br(b"RH1" + _det_bytes("region", 32)),
        "randmulti": _br(_det_bytes("randmulti", 32)),
    }
    raw = pack_pr85_bundle(segments, header_mode="explicit_30")
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, raw)


def _blocker_codes(payload: dict) -> set[str]:
    return {row["code"] for row in payload["blockers"]}


def test_public_replay_preflight_accepts_byte_closed_x_archive_and_runtime(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    inflate_sh = _write_runtime(tmp_path / "replay_submission")
    upstream = _write_upstream(tmp_path)
    _write_x_archive(archive)

    payload = module.build_preflight(archive, inflate_sh, upstream_dir=upstream)

    assert payload["ready_for_exact_eval_dispatch"] is True
    assert payload["evidence_grade"] == module.EVIDENCE_GRADE
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["dispatch_performed"] is False
    assert payload["archive"]["charged_member_allowlist"] == "passed"
    smoke = payload["archive"]["members"][0]["decode_smoke"]["format"]
    assert smoke["format"] == "pr85_explicit_30byte_lengths"
    assert [row["name"] for row in smoke["segments"]] == list(SEGMENT_ORDER)
    assert smoke["segments"][0]["codec"] == "QMA9"
    assert smoke["segments"][1]["decoded_magic_ascii"].startswith("QH0")
    assert payload["runtime"]["runtime_tree_sha256"]


def test_public_replay_preflight_blocks_duplicate_and_sidecar_members(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    inflate_sh = _write_runtime(tmp_path / "replay_submission")
    upstream = _write_upstream(tmp_path)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("x", b"not-a-pr85-bundle")
        zf.writestr("x", b"duplicate")
        zf.writestr("notes.debug", b"sidecar")

    payload = module.build_preflight(archive, inflate_sh, upstream_dir=upstream)

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert {
        "duplicate_member_names",
        "charged_member_allowlist",
        "member_decode_smoke",
    } <= _blocker_codes(payload)


def test_public_replay_preflight_blocks_zip_central_local_name_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    inflate_sh = _write_runtime(tmp_path / "replay_submission")
    upstream = _write_upstream(tmp_path)
    _write_x_archive(archive)
    raw = bytearray(archive.read_bytes())
    with zipfile.ZipFile(archive, "r") as zf:
        offset = zf.getinfo("x").header_offset
    assert raw[offset : offset + 4] == b"PK\x03\x04"
    raw[offset + 30] = ord("y")
    archive.write_bytes(raw)

    payload = module.build_preflight(archive, inflate_sh, upstream_dir=upstream)

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "zip_container_integrity" in _blocker_codes(payload)


def test_public_replay_preflight_blocks_source_embedded_payload_runtime(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    inflate_sh = _write_runtime(tmp_path / "replay_submission", embedded_payload=True)
    upstream = _write_upstream(tmp_path)
    _write_x_archive(archive)

    payload = module.build_preflight(archive, inflate_sh, upstream_dir=upstream)

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "runtime_source_or_sidecar_payload" in _blocker_codes(payload)
    assert payload["runtime"]["source_payload_scan"]["status"] == "failed"


def test_public_replay_preflight_expected_runtime_tree_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    inflate_sh = _write_runtime(tmp_path / "replay_submission")
    upstream = _write_upstream(tmp_path)
    _write_x_archive(archive)

    payload = module.build_preflight(
        archive,
        inflate_sh,
        upstream_dir=upstream,
        expected_runtime_tree_sha256="0" * 64,
    )

    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "expected_runtime_tree_sha256_matches" in _blocker_codes(payload)
