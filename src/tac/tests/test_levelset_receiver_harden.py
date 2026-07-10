"""#402 RECEIVER FAIL-CLOSED HARDENING regression tests (advisory PR128 §8.2 defect list applied to
the level-set witness byte-close / inflate surfaces in ``tools/levelset_byte_close_and_eval.py``).

Six checklist axes, each pinned:
  1. exact stream consumption   -- section readers fail closed on trailing/short bytes;
  2. final raw output assertion -- a short .raw is refused before scoring (structurally impossible);
  3. atomic + resumable writes  -- inflate writes tmp+rename, never a scoreable-looking partial;
  4. storage preflight          -- ~3.66 GB output fails closed when the volume lacks room;
  5. dependency pinning         -- decode-path dep versions recorded in the receiver env manifest;
  6. no cross-microarch sha     -- authority is numpy-fp32; no portable sha256-of-.raw claim.

All fixtures are tiny synthetic blobs / stubbed subprocesses -- NO n600 decode, NO heavy checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "levelset_byte_close_and_eval.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("lbc_harden", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture(scope="module")
def inflate_ns(tool):
    """Exec the shipped inflate.py template into an isolated namespace so its parsers can be tested."""
    ns: dict = {"__name__": "_inflate_under_test"}
    exec(compile(tool._INFLATE_PY, "<inflate.py>", "exec"), ns)  # noqa: S102 — pinning shipped receiver
    return ns


def _minimal_lvls1(tool, base=b"BASE", code=b"CODE", manifest=None) -> bytes:
    man = manifest if manifest is not None else {"lane_render_band": None, "pose_carrier": None}
    mj = json.dumps(man).encode()
    return tool._io_pack(mj, base, code, None)


# --------------------------------------------------------------------------- #
# ITEM 1 -- exact stream consumption
# --------------------------------------------------------------------------- #
def test_read_blob_bytes_happy_path(tool):
    blob = _minimal_lvls1(tool)
    man, base, code, pose, lane, pcar = tool._read_blob_bytes(blob)
    assert base == b"BASE" and code == b"CODE"
    assert lane is None and pcar is None


def test_read_blob_bytes_trailing_byte_fails_closed(tool):
    blob = _minimal_lvls1(tool) + b"\x00"
    with pytest.raises(ValueError, match="unconsumed trailing"):
        tool._read_blob_bytes(blob)


def test_read_blob_bytes_truncated_section_fails_closed(tool):
    bad = tool._MAGIC + struct.pack("<I", 100) + b"short"  # declares 100 B, provides 5
    with pytest.raises(ValueError, match="truncated"):
        tool._read_blob_bytes(bad)


def test_read_blob_bytes_wrong_magic_fails_closed(tool):
    with pytest.raises(AssertionError):
        tool._read_blob_bytes(b"NOPE0\x00" + b"\x00" * 20)


def test_inflate_read_blob_trailing_fails_closed(inflate_ns, tool, tmp_path):
    blob = _minimal_lvls1(tool) + b"\xff\xff"
    p = tmp_path / "0.bin"
    p.write_bytes(blob)
    with pytest.raises(ValueError, match="unconsumed trailing"):
        inflate_ns["_read_blob"](str(p))


def test_inflate_read_blob_happy_path(inflate_ns, tool, tmp_path):
    p = tmp_path / "0.bin"
    p.write_bytes(_minimal_lvls1(tool))
    m, base, code, pose, lane, pcar = inflate_ns["_read_blob"](str(p))
    assert base == b"BASE" and code == b"CODE"


def test_inflate_dequant_happy_and_exact(inflate_ns):
    order = ["a", "b"]
    shapes = {"a": [2, 2], "b": [3]}
    scales = {"a": 1.0, "b": 0.5}
    blob = np.arange(4 + 3, dtype=np.int8).tobytes()
    out = inflate_ns["_dequant"](blob, order, shapes, scales)
    assert out["a"].shape == (2, 2) and out["b"].shape == (3,)


def test_inflate_dequant_trailing_int8_fails_closed(inflate_ns):
    order = ["a"]
    shapes = {"a": [2, 2]}
    scales = {"a": 1.0}
    blob = np.arange(4 + 1, dtype=np.int8).tobytes()  # one trailing int8
    with pytest.raises(ValueError, match="unconsumed int8"):
        inflate_ns["_dequant"](blob, order, shapes, scales)


def test_inflate_dequant_short_fails_closed(inflate_ns):
    order = ["a"]
    shapes = {"a": [2, 3]}  # needs 6 int8
    scales = {"a": 1.0}
    blob = np.arange(4, dtype=np.int8).tobytes()  # only 4
    with pytest.raises(ValueError, match="short"):
        inflate_ns["_dequant"](blob, order, shapes, scales)


def test_lane_parse_trailing_float64_fails_closed(inflate_ns):
    # hand-built minimal LBND1: 1 pair, 1 line, nc=1 nh=1 no-dash => 4 float64 values (+1 trailing).
    hdr = {"pairs": [[{"nc": 1, "nh": 1, "has_dash": False}]]}
    hj = json.dumps(hdr).encode()
    vals = np.array([1.0, 2.0, 3.0, 4.0, 9.9], dtype=np.float64)  # 5th is illegal trailing
    blob = b"LBND1\x00" + struct.pack("<I", len(hj)) + hj + vals.tobytes()
    with pytest.raises(ValueError, match="unconsumed float64"):
        inflate_ns["_lane_parse"](blob)


def test_lane_parse_exact_happy(inflate_ns):
    hdr = {"pairs": [[{"nc": 1, "nh": 1, "has_dash": False}]]}
    hj = json.dumps(hdr).encode()
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    blob = b"LBND1\x00" + struct.pack("<I", len(hj)) + hj + vals.tobytes()
    pairs, out_hdr = inflate_ns["_lane_parse"](blob)
    assert len(pairs) == 1 and len(pairs[0]) == 1


def _store_nothing_pcar(tool, P=3):
    xi = np.zeros((P, 6), dtype=np.float64)
    xi[:, 0] = np.linspace(0.0, 0.1, P)  # small nonzero twist
    blob, _rep = tool.serialize_pose_carrier_store_nothing(xi, {"pitch": 0.02}, coder="none")
    return blob


def test_pose_carrier_v2_roundtrip_no_false_trailing(tool):
    # regression: the v2 serializer writes a trailing u32 n_kf(=0); the hardened parsers must CONSUME
    # it (not raise on a legitimate blob). Both the tool-side and inflate-side parsers must agree.
    blob = _store_nothing_pcar(tool)
    pc = tool.parse_pose_carrier(blob)
    assert pc["H"].shape == (3, 3, 3)


def test_pose_carrier_v2_trailing_fails_closed_tool(tool):
    blob = _store_nothing_pcar(tool) + b"\x00"
    with pytest.raises(ValueError, match="unconsumed trailing"):
        tool.parse_pose_carrier(blob)


def test_pose_carrier_v2_inflate_parser_consumes_n_kf(inflate_ns, tool):
    # the shipped inflate._pcar_parse previously IGNORED the trailing n_kf=0 (§8.2 defect); it must
    # now consume it exactly (happy) and reject a real trailing byte.
    blob = _store_nothing_pcar(tool)
    pc = inflate_ns["_pcar_parse"](blob)
    assert pc["H"].shape == (3, 3, 3)
    with pytest.raises(ValueError, match="unconsumed trailing"):
        inflate_ns["_pcar_parse"](blob + b"\x01")


# --------------------------------------------------------------------------- #
# ITEM 2 -- final raw output assertion (short raw refused before scoring)
# --------------------------------------------------------------------------- #
def _tiny_packet(tool, tmp_path):
    blob = _minimal_lvls1(tool)
    pkt = tmp_path / "pkt"
    tool.assemble_packet(blob, pkt)
    return pkt


def test_run_inflate_raises_on_short_raw(tool, tmp_path, monkeypatch):
    pkt = _tiny_packet(tool, tmp_path)

    def _fake_run(cmd, **kw):
        dst = Path(cmd[-1])
        dst.write_bytes(b"\x00" * 10)  # deliberately short

        class R:  # noqa: D401
            returncode = 0
            stdout = "short"
            stderr = ""
        return R()

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="short raw is evaluator truncation"):
        tool.run_inflate(pkt, n_pairs_total=1, max_pairs=None)


def test_run_inflate_accepts_exact_raw_and_reports_storage(tool, tmp_path, monkeypatch):
    pkt = _tiny_packet(tool, tmp_path)
    expected = 2 * 1 * tool.CAMERA_H * tool.CAMERA_W * 3

    def _fake_run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"\x00" * expected)

        class R:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return R()

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)
    info = tool.run_inflate(pkt, n_pairs_total=1, max_pairs=None)
    assert info["full_output_shape_ok"] is True
    assert info["raw_bytes"] == expected
    assert info["storage_preflight"]["ok"] is True  # item 4 wired into run_inflate


def test_run_inflate_raises_when_inflate_subprocess_fails(tool, tmp_path, monkeypatch):
    pkt = _tiny_packet(tool, tmp_path)

    def _fake_run(cmd, **kw):
        class R:
            returncode = 3
            stdout = ""
            stderr = "boom"
        return R()

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="inflate.py FAILED"):
        tool.run_inflate(pkt, n_pairs_total=1, max_pairs=None)


# --------------------------------------------------------------------------- #
# ITEM 3 -- atomic + resumable output writes (structural pin of the shipped inflate)
# --------------------------------------------------------------------------- #
def test_inflate_template_writes_atomically(tool):
    src = tool._INFLATE_PY
    assert '.partial' in src, "inflate must write to a .partial sibling"
    assert 'os.replace(tmp, dst)' in src, "inflate must atomically rename tmp -> dst"
    assert 'os.path.getsize(tmp)' in src, "inflate must size-check the tmp before rename"
    assert 'inflate output SHORT' in src, "inflate must fail closed on a short output"


# --------------------------------------------------------------------------- #
# ITEM 4 -- storage preflight
# --------------------------------------------------------------------------- #
def test_storage_preflight_ok(tool, tmp_path):
    info = tool._raw_storage_preflight(tmp_path, 1024)
    assert info["ok"] is True and info["expected_raw_bytes"] == 1024


def test_storage_preflight_fails_closed(tool, tmp_path):
    with pytest.raises(RuntimeError, match="storage preflight FAILED"):
        tool._raw_storage_preflight(tmp_path, 10 ** 18)


# --------------------------------------------------------------------------- #
# ITEM 5 -- dependency pinning (receiver env manifest)
# --------------------------------------------------------------------------- #
def test_receiver_env_manifest_pins_decode_deps(tool):
    man = tool.receiver_env_manifest()
    vers = man["decode_path_versions"]
    for dep in ("python", "numpy", "torch", "brotli", "scipy"):
        assert dep in vers and isinstance(vers[dep], str) and vers[dep]


def test_receiver_env_manifest_constriction_is_na(tool):
    man = tool.receiver_env_manifest()
    assert "N/A" in man["decode_path_versions"]["constriction"]


# --------------------------------------------------------------------------- #
# ITEM 6 -- no cross-microarch bit-identity assumption / no portable sha claim
# --------------------------------------------------------------------------- #
def test_receiver_env_manifest_cross_host_note(tool):
    man = tool.receiver_env_manifest()
    note = man["cross_host_bit_identity"].lower()
    assert "bicubic" in note
    assert "no portable sha256" in note or "no portable sha" in note
    assert "numpy-fp32" in man["authority_forward"].lower()


def test_tool_emits_no_portable_raw_sha_authority(tool):
    # the numpy-fp32 forward is the authority and R is a shared per-host op; the tool must NOT ship
    # a portable expected_output.sha256-style claim (advisory PR128 §8.2 flagged that as the defect).
    src = _TOOL.read_text()
    assert "expected_output.sha256" not in src
