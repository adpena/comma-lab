# pyc-recovery pass2: rehydrated from git blob 0b4a67005937534394b213483a65570b9ab98dfb via `git fsck --lost-found`
# original path: src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py
# OUR source dropped during commit 66c59aae filter-repo cleanup; .pyc was sole orphan left.
# Blob verified intact + parses cleanly with python ast.
# Recovered: 2026-05-05 by Sherlock pass2
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


brotli = pytest.importorskip("brotli")

from tac.pr85_bundle import FIXED_V5_LENGTHS, PR85_HEADERLESS_RANDMULTI_SPECS, pack_pr85_bundle, parse_pr85_bundle
from tac.stbm1br_mask_codec import STBM1BR_MAGIC


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_stbm1br_rmb1_randmulti_candidate.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("build_pr85_stbm1br_rmb1_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _zip_info() -> zipfile.ZipInfo:
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _row(counts: list[tuple[int, int]]) -> bytes:
    out = bytearray()
    out.append(len(counts))
    last = -1
    for idx, _value in counts:
        delta = idx - last - 1
        last = idx
        while True:
            byte = delta & 0x7F
            delta >>= 7
            out.append(byte | 0x80 if delta else byte)
            if not delta:
                break
    out.extend(value for _idx, value in counts)
    return bytes(out)


def _headerless_rows() -> bytes:
    rows = [_row([(0, 5), (3, 7)])]
    rows.extend(b"\x00" for _ in range(sum(spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS) - 1))
    return b"".join(rows)


def _rmb1_from_rows(raw: bytes) -> bytes:
    cursor = 0
    mask = bytearray()
    values = bytearray()
    while cursor < len(raw):
        count = raw[cursor]
        cursor += 1
        if count == 255:
            count = int.from_bytes(raw[cursor : cursor + 2], "little")
            cursor += 2
        row_mask = bytearray(75)
        idx = -1
        for _ in range(count):
            delta = 0
            shift = 0
            while True:
                byte = raw[cursor]
                cursor += 1
                delta |= (byte & 0x7F) << shift
                if byte < 128:
                    break
                shift += 7
            idx += delta + 1
            row_mask[idx // 8] |= 1 << (idx % 8)
        values.extend(raw[cursor : cursor + count])
        cursor += count
        mask.extend(row_mask)
    mask_br = brotli.compress(bytes(mask), quality=5)
    vals_br = brotli.compress(bytes(values), quality=5)
    return b"RMB1" + len(mask_br).to_bytes(2, "little") + mask_br + vals_br


def _write_archive(path: Path, *, mask: bytes, randmulti: bytes) -> Path:
    segments = {
        "mask": mask,
        "model": brotli.compress(b"QH0" + b"m" * 64, quality=5),
        "pose": brotli.compress(b"P1D1" + b"p" * 32, quality=5),
        "post": brotli.compress(b"\x00" * 12, quality=5),
        "shift": brotli.compress(b"SD4" + b"\x00" * 12, quality=5),
        "frac": brotli.compress(b"FH1" + b"\x04" * 12, quality=5),
        "frac2": brotli.compress(b"FH2" + b"\x04" * 12, quality=5),
        "frac3": brotli.compress(b"FD3" + b"\x00" * 12, quality=5),
        "bias": b"B" * FIXED_V5_LENGTHS["bias"],
        "region": b"R" * FIXED_V5_LENGTHS["region"],
        "randmulti": randmulti,
    }
    raw = pack_pr85_bundle(segments, header_mode="v5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(), raw)
    return path


def _write_runtime_support(root: Path, *, support_x: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "inflate_renderer.py").write_text(
        "STBM1BR_MAGIC = b'STBM1BR\\0'\n"
        "def _load_masks_from_stbm1br():\n"
        "    pass\n",
        encoding="utf-8",
    )
    (root / "apply_qzs3_postprocess.py").write_text(
        "def _decode_rmb1_randmulti_payload():\n"
        "    pass\n"
        "if blob[:4] == b\"RMB1\":\n"
        "    pass\n",
        encoding="utf-8",
    )
    x_probe = 'if [ -f "$ARCHIVE_DIR/x" ]; then echo x; fi\n' if support_x else ""
    (root / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + x_probe,
        encoding="utf-8",
    )
    (root / "unpack_renderer_payload.py").write_text(
        "PAYLOAD_SHORT_BR = \"x\"\n" if support_x else "PAYLOAD_SHORT_BR = \"p\"\n",
        encoding="utf-8",
    )
    return root


def test_builder_replaces_only_randmulti_and_records_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    rows = _headerless_rows()
    stbm_archive = _write_archive(
        tmp_path / "stbm.zip",
        mask=STBM1BR_MAGIC + b"mask",
        randmulti=brotli.compress(rows, quality=0),
    )
    pr92_archive = _write_archive(
        tmp_path / "pr92.zip",
        mask=b"QMA9" + b"mask",
        randmulti=_rmb1_from_rows(rows),
    )
    monkeypatch.setattr(module, "EXPECTED_STBM_SHA256", module._sha256_file(stbm_archive))
    monkeypatch.setattr(module, "EXPECTED_STBM_BYTES", stbm_archive.stat().st_size)
    monkeypatch.setattr(module, "EXPECTED_PR92_SHA256", module._sha256_file(pr92_archive))
    monkeypatch.setattr(module, "ROBUST_CURRENT_DIR", _write_runtime_support(tmp_path / "runtime"))

    summary = module.build_pr85_stbm1br_rmb1_randmulti_candidate(
        stbm_archive=stbm_archive,
        pr92_archive=pr92_archive,
        out_dir=tmp_path / "out",
    )

    manifest = json.loads((tmp_path / "out" / summary["candidate_id"] / "manifest.json").read_text())
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_after_lane_claim"] is True
    assert manifest["fail_closed_preflight"]["status"] == "passed"
    assert manifest["fail_closed_preflight"]["checks"]["robust_current_runtime_support"] is True
    assert manifest["fail_closed_preflight"]["checks"]["duplicate_rmb1_builder_lane_coordination"] is True
    assert manifest["runtime_support"]["status"] == "passed"
    assert manifest["runtime_support"]["checks"]["stbm1br_magic_present"] is True
    assert manifest["runtime_support"]["checks"]["single_member_x_unpack_stage_present"] is True
    assert manifest["runtime_support"]["checks"]["rmb1_branch_present"] is True
    assert manifest["parity"]["randmulti_decoded_rows_equal"] is True
    assert manifest["segments"]["mask"]["sha256"] == module._sha256_bytes(STBM1BR_MAGIC + b"mask")
    assert manifest["segments"]["candidate_randmulti"]["codec"] == "RMB1_bitmask_value_randmulti"
    gate = manifest["dispatch_gate"]
    claim = gate["claim"]
    command = claim["command_template"]
    assert gate["status"] == "eligible_for_exact_eval_after_level2_lane_claim"
    assert gate["dispatch_performed"] is False
    assert claim["command_not_executed_by_builder"] is True
    assert command[:3] == [".venv/bin/python", "tools/claim_lane_dispatch.py", "claim"]
    assert "--dry-run" not in command
    assert command[command.index("--claims-path") + 1] == ".omx/state/active_lane_dispatch_claims.md"
    assert command[command.index("--lane-id") + 1] == "pr85_stbm1br_pr92_rmb1_randmulti"
    assert command[command.index("--platform") + 1] == "lightning"
    assert command[command.index("--status") + 1] == "exact_eval_ready"
    assert "archive_sha256=" in command[command.index("--notes") + 1]
    duplicate = manifest["duplicate_builder_coordination"]
    assert summary["canonical_builder"] == "experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py"
    assert duplicate["status"] == "passed"
    assert duplicate["legacy_lane_id"] == duplicate["canonical_constants"]["LANE_ID"]
    assert duplicate["checks"]["canonical_tool_matches_path"] is True

    with zipfile.ZipFile(tmp_path / "out" / summary["candidate_id"] / "archive.zip") as zf:
        parsed = parse_pr85_bundle(zf.read("x"))
    assert parsed.segments["mask"] == STBM1BR_MAGIC + b"mask"
    assert parsed.segments["randmulti"].startswith(b"RMB1")

    custom = module.build_pr85_stbm1br_rmb1_randmulti_candidate(
        stbm_archive=stbm_archive,
        pr92_archive=pr92_archive,
        out_dir=tmp_path / "custom",
        candidate_id="custom_candidate",
    )
    assert custom["candidate_id"] == "custom_candidate"


def test_builder_fails_closed_when_rmb1_rows_differ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    rows = _headerless_rows()
    stbm_archive = _write_archive(
        tmp_path / "stbm.zip",
        mask=STBM1BR_MAGIC + b"mask",
        randmulti=brotli.compress(rows, quality=5),
    )
    pr92_archive = _write_archive(
        tmp_path / "pr92.zip",
        mask=b"QMA9" + b"mask",
        randmulti=_rmb1_from_rows(b"\x00" * sum(spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS)),
    )
    monkeypatch.setattr(module, "EXPECTED_STBM_SHA256", module._sha256_file(stbm_archive))
    monkeypatch.setattr(module, "EXPECTED_STBM_BYTES", stbm_archive.stat().st_size)
    monkeypatch.setattr(module, "EXPECTED_PR92_SHA256", module._sha256_file(pr92_archive))
    monkeypatch.setattr(module, "ROBUST_CURRENT_DIR", _write_runtime_support(tmp_path / "runtime"))

    with pytest.raises(module.Rmb1CandidateBuildError, match="decoded randmulti rows differ"):
        module.build_pr85_stbm1br_rmb1_randmulti_candidate(
            stbm_archive=stbm_archive,
            pr92_archive=pr92_archive,
            out_dir=tmp_path / "out",
        )


def test_builder_fails_closed_without_single_member_x_runtime_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    rows = _headerless_rows()
    stbm_archive = _write_archive(
        tmp_path / "stbm.zip",
        mask=STBM1BR_MAGIC + b"mask",
        randmulti=brotli.compress(rows, quality=0),
    )
    pr92_archive = _write_archive(
        tmp_path / "pr92.zip",
        mask=b"QMA9" + b"mask",
        randmulti=_rmb1_from_rows(rows),
    )
    monkeypatch.setattr(module, "EXPECTED_STBM_SHA256", module._sha256_file(stbm_archive))
    monkeypatch.setattr(module, "EXPECTED_STBM_BYTES", stbm_archive.stat().st_size)
    monkeypatch.setattr(module, "EXPECTED_PR92_SHA256", module._sha256_file(pr92_archive))
    monkeypatch.setattr(module, "ROBUST_CURRENT_DIR", _write_runtime_support(tmp_path / "runtime", support_x=False))

    with pytest.raises(module.Rmb1CandidateBuildError, match="robust_current_runtime_support"):
        module.build_pr85_stbm1br_rmb1_randmulti_candidate(
            stbm_archive=stbm_archive,
            pr92_archive=pr92_archive,
            out_dir=tmp_path / "out",
        )
