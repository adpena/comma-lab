# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np

from tac.pr85_bundle import pack_pr85_bundle, parse_pr85_bundle
from tac.qma9_range_mask_contract import sha256_bytes
from tac.stbm1br_mask_codec import STBM1BR_MAGIC


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_stbm1br_mask_recode_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_stbm1br_mask_recode_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_stored_zip(path: Path, member: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _fake_pr85_archive(path: Path, source_mask: bytes) -> None:
    segments = {
        "mask": source_mask,
        "model": b"m" * 32,
        "pose": b"p" * 8,
        "post": b"o" * 7,
        "shift": b"s" * 6,
        "frac": b"f" * 5,
        "frac2": b"g" * 5,
        "frac3": b"h" * 5,
        "bias": b"b" * 223,
        "region": b"r" * 273,
        "randmulti": b"z" * 9,
    }
    _write_stored_zip(path, "x", pack_pr85_bundle(segments, header_mode="v5"))


def _runtime_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "inflate_renderer.py").write_text(
        "STBM1BR_MAGIC = b'STBM1BR\\\\0'\n"
        "def _load_masks_from_qma9(path):\n"
        "    return b'QMA9'\n"
        "def _load_masks_from_stbm1br(path):\n"
        "    return path\n"
        "def _load_masks_from_archive(path):\n"
        "    return _load_masks_from_stbm1br(path)\n",
        encoding="utf-8",
    )
    return path


def _replay_runtime_dir(path: Path, *, stbm_aware: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "inflate.sh").write_text(
        "#!/usr/bin/env bash\npython \"$PWD/inflate.py\" \"$@\"\n",
        encoding="utf-8",
    )
    if stbm_aware:
        inflate = (
            "from pathlib import Path\n"
            "from tac.stbm1br_mask_codec import decode_stbm1br_mask_segment\n"
            "def load_compact_archive_bundle(data_dir):\n"
            "    payload = (data_dir / \"x\").read_bytes()\n"
            "    return {'mask': payload}\n"
            "def load_stbm1br_mask(mask_payload):\n"
            "    return decode_stbm1br_mask_segment(mask_payload, expected_shape=(600, 384, 512))\n"
            "def main(data_dir):\n"
            "    bundle = load_compact_archive_bundle(Path(data_dir))\n"
            "    if bundle[\"mask\"][:8] == b\"STBM1BR\\0\":\n"
            "        return load_stbm1br_mask(bundle[\"mask\"])\n"
            "    if bundle[\"mask\"][:4] == b\"QMA9\":\n"
            "        return b'QMA9'\n"
            "    import brotli\n"
            "    return brotli.decompress(bundle[\"mask\"])\n"
        )
    else:
        inflate = (
            "from pathlib import Path\n"
            "def load_compact_archive_bundle(data_dir):\n"
            "    payload = (data_dir / \"x\").read_bytes()\n"
            "    return {'mask': payload}\n"
            "def main(data_dir):\n"
            "    bundle = load_compact_archive_bundle(Path(data_dir))\n"
            "    import brotli\n"
            "    return brotli.decompress(bundle[\"mask\"])\n"
        )
    (path / "inflate.py").write_text(inflate, encoding="utf-8")
    (path / "range_mask_codec.cpp").write_text("// qma9 fallback fixture\n", encoding="utf-8")
    return path


class _Meta:
    def __init__(self, segment: bytes, body_bytes: int):
        self.segment_bytes = len(segment)
        self.segment_sha256 = sha256_bytes(segment)
        self.brotli_body_bytes = body_bytes
        self.brotli_body_sha256 = sha256_bytes(segment[len(STBM1BR_MAGIC):])


def test_builder_writes_single_x_stbm_candidate_and_records_parity(monkeypatch, tmp_path: Path) -> None:
    shape = (2, 2, 3)
    render = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape) % 5
    storage = render.transpose(0, 2, 1).copy()
    expected_sha = sha256_bytes(render.tobytes())
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(storage.tobytes())
    source_mask = (
        b"QMA9"
        + shape[0].to_bytes(4, "little")
        + shape[2].to_bytes(4, "little")
        + shape[1].to_bytes(4, "little")
        + (4).to_bytes(4, "little")
        + b"abcd"
    )
    pr85 = tmp_path / "pr85.zip"
    _fake_pr85_archive(pr85, source_mask)
    pr90_body = b"body"
    pr90 = tmp_path / "pr90.zip"
    _write_stored_zip(pr90, "p", pr90_body + b"tail")
    policy = tmp_path / "policy.json"
    policy.write_text(
        '{"score_claim": false, "dispatch_performed": false, '
        '"ranked_candidates": [{"policy_id": "pr90_stbm1br_lossless_pr85_mask_recode", '
        '"status": "builder_ready_after_runtime_port"}]}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "parse_stbm1br_metadata",
        lambda segment: _Meta(segment, len(segment) - len(STBM1BR_MAGIC)),
    )
    monkeypatch.setattr(module, "metadata_as_dict", lambda meta: dict(meta.__dict__))
    monkeypatch.setattr(module, "decode_stbm1br_mask_segment", lambda _segment, expected_shape: render)

    manifest = module.build_pr85_stbm1br_mask_recode_candidate(
        pr85_archive=pr85,
        pr90_archive=pr90,
        policy_json=policy,
        out_dir=tmp_path / "out",
        token_source=token_source,
        robust_current_dir=_runtime_dir(tmp_path / "runtime"),
        pr85_replay_runtime_dir=_replay_runtime_dir(tmp_path / "replay_runtime"),
        require_exact_eval_runtime=True,
        expected_shape=shape,
        expected_pr85_render_sha256=expected_sha,
        expected_pr85_archive_bytes=None,
        expected_pr85_archive_sha256=None,
        expected_pr90_archive_bytes=None,
        expected_pr90_archive_sha256=None,
        expected_pr90_mask_body_bytes=len(pr90_body),
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_performed"] is False
    assert manifest["remote_jobs_dispatched"] is False
    assert manifest["parity"]["decoded_mask_equal"] is True
    assert manifest["fail_closed_preflight"]["ready_for_exact_eval_after_lane_claim"] is True
    assert manifest["fail_closed_preflight"]["remaining_exact_eval_blockers"] == []
    assert manifest["exact_eval_runtime_contract"]["status"] == "passed"
    assert manifest["exact_eval_runtime_contract"]["runtime_tree_sha256"]
    assert {
        row["path"]
        for row in manifest["exact_eval_runtime_contract"]["files"]
    } == {"inflate.py", "inflate.sh", "range_mask_codec.cpp"}
    assert manifest["segments"]["byte_delta_vs_source_mask"] < 0
    archive = REPO / manifest["candidate_archive"]["path"]
    with zipfile.ZipFile(archive, "r") as zf:
        assert zf.namelist() == ["x"]
        candidate_x = zf.read("x")
    parsed = parse_pr85_bundle(candidate_x)
    assert parsed.segments["mask"] == STBM1BR_MAGIC + pr90_body
    assert parsed.segment_lengths["mask"] == len(STBM1BR_MAGIC) + len(pr90_body)


def test_missing_explicit_replay_runtime_marks_candidate_non_dispatchable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    shape = (2, 2, 3)
    render = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape) % 5
    storage = render.transpose(0, 2, 1).copy()
    expected_sha = sha256_bytes(render.tobytes())
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(storage.tobytes())
    source_mask = (
        b"QMA9"
        + shape[0].to_bytes(4, "little")
        + shape[2].to_bytes(4, "little")
        + shape[1].to_bytes(4, "little")
        + (4).to_bytes(4, "little")
        + b"abcd"
    )
    pr85 = tmp_path / "pr85.zip"
    _fake_pr85_archive(pr85, source_mask)
    pr90_body = b"body"
    pr90 = tmp_path / "pr90.zip"
    _write_stored_zip(pr90, "p", pr90_body + b"tail")
    policy = tmp_path / "policy.json"
    policy.write_text(
        '{"score_claim": false, "dispatch_performed": false, '
        '"ranked_candidates": [{"policy_id": "pr90_stbm1br_lossless_pr85_mask_recode"}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "parse_stbm1br_metadata",
        lambda segment: _Meta(segment, len(segment) - len(STBM1BR_MAGIC)),
    )
    monkeypatch.setattr(module, "metadata_as_dict", lambda meta: dict(meta.__dict__))
    monkeypatch.setattr(module, "decode_stbm1br_mask_segment", lambda _segment, expected_shape: render)

    manifest = module.build_pr85_stbm1br_mask_recode_candidate(
        pr85_archive=pr85,
        pr90_archive=pr90,
        policy_json=policy,
        out_dir=tmp_path / "out",
        token_source=token_source,
        robust_current_dir=_runtime_dir(tmp_path / "runtime"),
        expected_shape=shape,
        expected_pr85_render_sha256=expected_sha,
        expected_pr85_archive_bytes=None,
        expected_pr85_archive_sha256=None,
        expected_pr90_archive_bytes=None,
        expected_pr90_archive_sha256=None,
        expected_pr90_mask_body_bytes=len(pr90_body),
    )

    preflight = manifest["fail_closed_preflight"]
    assert preflight["ready_for_exact_eval_after_lane_claim"] is False
    assert preflight["readiness_status"] == "non_dispatchable"
    assert {
        row["code"] for row in preflight["remaining_exact_eval_blockers"]
    } == {"exact_runtime:missing_explicit_pr85_replay_runtime"}
    summary = (tmp_path / "out" / "candidate_summary.json").read_text(encoding="utf-8")
    assert '"ready_for_exact_eval_after_lane_claim": false' in summary


def test_replay_runtime_contract_mismatch_fails_when_required(
    monkeypatch,
    tmp_path: Path,
) -> None:
    shape = (2, 2, 3)
    render = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape) % 5
    storage = render.transpose(0, 2, 1).copy()
    expected_sha = sha256_bytes(render.tobytes())
    token_source = tmp_path / "tokens.bin"
    token_source.write_bytes(storage.tobytes())
    source_mask = (
        b"QMA9"
        + shape[0].to_bytes(4, "little")
        + shape[2].to_bytes(4, "little")
        + shape[1].to_bytes(4, "little")
        + (4).to_bytes(4, "little")
        + b"abcd"
    )
    pr85 = tmp_path / "pr85.zip"
    _fake_pr85_archive(pr85, source_mask)
    pr90_body = b"body"
    pr90 = tmp_path / "pr90.zip"
    _write_stored_zip(pr90, "p", pr90_body + b"tail")
    policy = tmp_path / "policy.json"
    policy.write_text(
        '{"score_claim": false, "dispatch_performed": false, '
        '"ranked_candidates": [{"policy_id": "pr90_stbm1br_lossless_pr85_mask_recode"}]}\n',
        encoding="utf-8",
    )
    stale_contract = tmp_path / "stale_runtime_contract.json"
    stale_contract.write_text(
        '{"expected_candidate": {"candidate_archive_sha256": "' + ("0" * 64) + '"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "parse_stbm1br_metadata",
        lambda segment: _Meta(segment, len(segment) - len(STBM1BR_MAGIC)),
    )
    monkeypatch.setattr(module, "metadata_as_dict", lambda meta: dict(meta.__dict__))
    monkeypatch.setattr(module, "decode_stbm1br_mask_segment", lambda _segment, expected_shape: render)

    try:
        module.build_pr85_stbm1br_mask_recode_candidate(
            pr85_archive=pr85,
            pr90_archive=pr90,
            policy_json=policy,
            out_dir=tmp_path / "out",
            token_source=token_source,
            robust_current_dir=_runtime_dir(tmp_path / "runtime"),
            pr85_replay_runtime_dir=_replay_runtime_dir(tmp_path / "replay_runtime"),
            pr85_replay_runtime_contract_json=stale_contract,
            require_exact_eval_runtime=True,
            expected_shape=shape,
            expected_pr85_render_sha256=expected_sha,
            expected_pr85_archive_bytes=None,
            expected_pr85_archive_sha256=None,
            expected_pr90_archive_bytes=None,
            expected_pr90_archive_sha256=None,
            expected_pr90_mask_body_bytes=len(pr90_body),
        )
    except module.STBMRecodeBuildError as exc:
        assert "candidate_archive_sha256_matches_contract_json" in str(exc)
    else:
        raise AssertionError("stale exact runtime contract should fail closed when required")
