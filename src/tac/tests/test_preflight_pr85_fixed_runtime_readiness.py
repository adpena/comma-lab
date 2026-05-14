# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "preflight_pr85_fixed_runtime_readiness.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("preflight_pr85_fixed_runtime_test", SCRIPT)
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


def _vlq(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _p1d1_pose_raw() -> bytes:
    values = [0] * 600
    stream = bytearray()
    previous = 0
    for value in values:
        stream.extend(_vlq(_zigzag(value - previous)))
        previous = value
    return b"P1D1" + bytes([1, 0]) + len(stream).to_bytes(2, "little") + bytes(stream)


def _randmulti_zero_payload() -> bytes:
    return b"\x00" * sum(spec[3] for spec in module.PR85_HEADERLESS_RANDMULTI_SPECS)


def _write_pr85_archive(path: Path, *, post_label: str = "post") -> None:
    mask_bitstream = _det_bytes("qma9-bitstream", 1100)
    segments = {
        "mask": b"QMA9"
        + (600).to_bytes(4, "little")
        + (512).to_bytes(4, "little")
        + (384).to_bytes(4, "little")
        + len(mask_bitstream).to_bytes(4, "little")
        + mask_bitstream,
        "model": _br(b"QH0" + _det_bytes("qh0-model", 2600)),
        "pose": _br(_p1d1_pose_raw()),
        "post": _br(_det_bytes(post_label, 2400)),
        "shift": _br(b"SD4" + _det_bytes("shift", 600)),
        "frac": _br(b"FV1" + _det_bytes("frac", 180)),
        "frac2": _br(b"FH2" + _det_bytes("frac2", 600)),
        "frac3": _br(b"FD3" + _det_bytes("frac3", 600)),
        "bias": _br(b"BD1" + _det_bytes("bias", 600)),
        "region": _br(b"RH1" + _det_bytes("region", 600)),
        "randmulti": _br(_randmulti_zero_payload()),
    }
    raw = pack_pr85_bundle(segments, header_mode="explicit_30")
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, raw)


def _write_runtime(runtime: Path, *, wired: bool) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "range_mask_codec.cpp").write_text("// qma9 decode fixture\n", encoding="utf-8")
    if wired:
        (runtime / "inflate.sh").write_text(
            "#!/usr/bin/env bash\n# PR85 x-member adapter emits qpost.bin\n",
            encoding="utf-8",
        )
        (runtime / "unpack_renderer_payload.py").write_text(
            "\n".join(
                [
                    "from tac.pr85_bundle import parse_pr85_bundle",
                    "# PR85 member named \"x\" expands renderer.bin masks.qma9 qpost.bin post shift frac randmulti",
                    "def _parse_payload(payload):",
                    "    return {'renderer.bin': payload, 'masks.qma9': payload, 'qpost.bin': payload}, {}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (runtime / "inflate_renderer.py").write_text(
            "\n".join(
                [
                    "# QH0 P1D1 masks.qma9",
                    "def _load_masks_from_qma9(path):",
                    "    return path",
                    "def _load_renderer(path, device):",
                    "    return ('QH0', path, device)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (runtime / "apply_qzs3_postprocess.py").write_text(
            "\n".join(
                [
                    "import json",
                    "# PR85 qpost.bin post shift frac randmulti QRM1",
                    f"PR82_QRM1_RANDMULTI_SPECS = {module.PR85_HEADERLESS_RANDMULTI_SPECS!r}",
                    "def read_qpost(path, device):",
                    "    return path, device",
                    "def apply_qpost_to_raw(path):",
                    "    return path",
                    "def _decode_qrm1_randmulti(raw, device):",
                    "    return raw, device",
                    "def _decode_randmulti(raw, device):",
                    "    PR85_HEADERLESS_RANDMULTI_SPECS = ((224, 222, 4, 1),)",
                    "    return raw, device, PR85_HEADERLESS_RANDMULTI_SPECS",
                    "def main(argv=None):",
                    "    print(json.dumps({\"qpost\": \"fixture\", \"records\": []}, sort_keys=True))",
                    "    return 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (runtime / "inflate.sh").write_text(
            "#!/usr/bin/env bash\nif [ -f \"$ARCHIVE_DIR/qpost.bin\" ]; then echo qpost.bin; fi\n",
            encoding="utf-8",
        )
        (runtime / "unpack_renderer_payload.py").write_text(
            "# public p payload only\n# renderer.bin masks.qma9\n"
            "def _parse_payload(payload):\n"
            "    return payload\n",
            encoding="utf-8",
        )
        (runtime / "inflate_renderer.py").write_text(
            "# masks.qma9\n"
            "def _load_masks_from_qma9(path):\n"
            "    return path\n"
            "def _load_renderer(path, device):\n"
            "    return path, device\n",
            encoding="utf-8",
        )
        (runtime / "apply_qzs3_postprocess.py").write_text(
            "# QRM1 qpost.bin\n"
            "def read_qpost(path, device):\n"
            "    return path, device\n"
            "def apply_qpost_to_raw(path):\n"
            "    return path\n"
            "def _decode_qrm1_randmulti(raw, device):\n"
            "    return raw, device\n"
            "def _decode_randmulti(raw, device):\n"
            "    return raw, device\n",
            encoding="utf-8",
        )


def test_preflight_reports_blockers_for_unwired_fixed_runtime(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    runtime = tmp_path / "robust_current"
    _write_pr85_archive(archive)
    _write_runtime(runtime, wired=False)

    payload = module.build_preflight(archive, runtime)

    assert payload["planning_only"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert payload["ready_for_fixed_runtime_exact_eval"] is False
    assert {
        row["code"] for row in payload["fixed_runtime_bridge"]["remaining_blockers"]
    } == {"qh0_model_loader_available"}
    assert payload["bundle"]["format"] == "pr85_explicit_30byte_lengths"
    assert payload["bundle"]["roundtrip_matches_input"] is True
    assert payload["bundle"]["cheap_segment_probe_ok"] is True
    caps = {row["id"]: row["status"] for row in payload["robust_current_runtime"]["capabilities"]}
    assert caps["qma9_mask_decode_available"] == "passed"
    assert caps["qpost_runtime_available"] == "passed"
    direct_blocker_codes = {
        row["id"]
        for row in payload["robust_current_runtime"]["capabilities"]
        if row["status"] != "passed"
    }
    assert {
        "pr85_single_member_x_dispatch",
        "pr85_bundle_expands_to_runtime_members",
        "qh0_model_loader_available",
        "p1d1_pose_loader_available",
        "pr85_sidechannels_exposed_to_qpost",
        "pr85_randmulti_v5_consumption",
    } <= direct_blocker_codes
    bridge_caps = {
        row["id"]: row["status"]
        for row in payload["fixed_runtime_bridge"]["capabilities"]
    }
    assert bridge_caps["pr85_single_member_x_dispatch"] == "passed"
    assert bridge_caps["pr85_bundle_expands_to_runtime_members"] == "passed"
    assert bridge_caps["p1d1_pose_loader_available"] == "passed"
    assert bridge_caps["pr85_sidechannels_exposed_to_qpost"] == "passed"
    assert bridge_caps["pr85_randmulti_v5_consumption"] == "passed"
    assert "qh0_model_loader_available" in {row["code"] for row in payload["blockers"]}
    assert "pr85_atom_substrate:qrm1_runtime_schedule_matches_pr85" in {
        row["code"] for row in payload["blockers"]
    }
    assert "pr85_atom_substrate:qpost_apply_summary_observable" in {
        row["code"] for row in payload["blockers"]
    }


def test_preflight_can_pass_after_static_pr85_runtime_markers_are_wired(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    runtime = tmp_path / "robust_current"
    _write_pr85_archive(archive)
    _write_runtime(runtime, wired=True)

    payload = module.build_preflight(archive, runtime)

    assert payload["ready_for_fixed_runtime_exact_eval"] is True
    assert payload["blockers"] == []
    assert all(
        row["status"] == "passed"
        for row in payload["robust_current_runtime"]["capabilities"]
    )
    assert all(
        row["status"] == "passed"
        for row in payload["fixed_runtime_bridge"]["atom_substrate"]["checks"]
    )


def test_preflight_atom_edit_guard_blocks_source_preserving_candidate(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    runtime = tmp_path / "robust_current"
    _write_pr85_archive(archive)
    _write_runtime(runtime, wired=True)

    payload = module.build_preflight(archive, runtime, atom_source_archive=archive)

    assert payload["ready_for_fixed_runtime_exact_eval"] is False
    assert payload["atom_edit_guard"]["required"] is True
    assert payload["atom_edit_guard"]["status"] == "failed"
    assert payload["atom_edit_guard"]["changed_segments"] == []
    assert {
        row["code"] for row in payload["blockers"]
    } == {"pr85_atom_edit_noop_source_preserving"}


def test_preflight_atom_edit_guard_passes_changed_charged_segment(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    runtime = tmp_path / "robust_current"
    _write_pr85_archive(source, post_label="source-post")
    _write_pr85_archive(candidate, post_label="candidate-post")
    _write_runtime(runtime, wired=True)

    payload = module.build_preflight(candidate, runtime, atom_source_archive=source)

    assert payload["ready_for_fixed_runtime_exact_eval"] is True
    assert payload["atom_edit_guard"]["status"] == "passed"
    assert payload["atom_edit_guard"]["payload_sha256_changed"] is True
    assert [row["segment"] for row in payload["atom_edit_guard"]["changed_segments"]] == ["post"]
    assert payload["atom_edit_guard"]["changed_segments"][0]["runtime_surfaces"] == [
        "qpost.bin",
        "qpost:post",
    ]


def test_preflight_expected_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    runtime = tmp_path / "robust_current"
    _write_pr85_archive(archive)
    _write_runtime(runtime, wired=True)

    payload = module.build_preflight(
        archive,
        runtime,
        expected_archive_sha256="0" * 64,
    )

    assert payload["ready_for_fixed_runtime_exact_eval"] is False
    assert {
        row["code"] for row in payload["blockers"]
    } == {"pr85_custody:expected_archive_sha256_matches"}


def test_preflight_cli_writes_deterministic_report_and_can_fail_on_blockers(
    tmp_path: Path,
    capsys,
) -> None:
    archive = tmp_path / "archive.zip"
    runtime = tmp_path / "robust_current"
    out = tmp_path / "preflight.json"
    _write_pr85_archive(archive)
    _write_runtime(runtime, wired=False)

    args = ["--archive", str(archive), "--robust-current-dir", str(runtime), "--json-out", str(out)]
    assert module.main(args) == 0
    first = out.read_text(encoding="utf-8")
    assert capsys.readouterr().out == first

    assert module.main(args) == 0
    second = out.read_text(encoding="utf-8")
    capsys.readouterr()
    assert first == second
    assert json.loads(second)["schema"] == module.SCHEMA
    assert module.main([*args, "--fail-if-not-ready"]) == 2


def test_real_pr85_archive_parse_if_available() -> None:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    if not archive.is_file():
        pytest.skip("public PR85 intake archive is not present")

    payload = module.build_preflight(archive, REPO / "submissions/robust_current")
    lengths = payload["bundle"]["segment_lengths"]

    assert payload["archive"]["known_public_pr85_v5_match"]["matches"] is True
    assert payload["archive"]["member_name"] == "x"
    assert [row["name"] for row in payload["bundle"]["segments"]] == list(SEGMENT_ORDER)
    assert lengths["mask"] == 159011
    assert lengths["model"] == 57074
    assert lengths["pose"] == 1487
    assert lengths["randmulti"] == 16101
    assert payload["fixed_runtime_bridge"]["expansion_available"] is True
    bridge_caps = {
        row["id"]: row["status"]
        for row in payload["fixed_runtime_bridge"]["capabilities"]
    }
    assert bridge_caps["pr85_bundle_expands_to_runtime_members"] == "passed"
    assert bridge_caps["p1d1_pose_loader_available"] == "passed"
    assert bridge_caps["pr85_randmulti_v5_consumption"] == "passed"
    assert payload["ready_for_fixed_runtime_exact_eval"] is True
    assert payload["fixed_runtime_bridge"]["remaining_blockers"] == []
    assert payload["blockers"] == []
