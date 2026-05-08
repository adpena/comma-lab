"""Tests for ``tools.codec_op_param_sweep_manifest``.

The sweep manifest keeps rich CodecOp provenance, while the optional
meta-Lagrangian candidate output must stay in the strict schema accepted by
``MetaLagrangianSearch.evaluate_candidate(**candidate)``.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import pathlib
import sys
import types
import zipfile

import torch


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "codec_op_param_sweep_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "codec_op_param_sweep_manifest", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_to_meta_lagrangian_candidates_emits_strict_evaluate_schema() -> None:
    mod = _load_tool_module()
    candidate = mod.SweepCandidate(
        candidate_id="kl_pose_k2",
        op_module="tac.codec_pipeline_kl_pose",
        op_class="Op_KLPoseStream",
        op_params={"n_components": 2},
        candidate_substream_bytes=128,
        bytes_in=1024,
        bytes_out=128,
        predicted_archive_bytes=185_120,
        predicted_score=0.2085,
        predicted_decomposition={"total": 0.2085},
        predicted_band=[0.2075, 0.2095],
        score_delta_vs_anchor=-0.0004,
        rate_delta_vs_anchor=-0.0004,
    )

    rows = mod.to_meta_lagrangian_candidates(
        [candidate], lane_class="kl_pose_stream"
    )

    assert rows == [
        {
            "candidate_id": "kl_pose_k2",
            "archive_bytes": 185_120,
            "rel_err_pct": 0.0,
            "n_layers": 0,
            "lane_class": "kl_pose_stream",
            "archive_path": None,
        }
    ]

    from tac.optimizer.meta_lagrangian import MetaLagrangianSearch

    accepted = set(inspect.signature(MetaLagrangianSearch.evaluate_candidate).parameters)
    accepted.discard("self")
    assert set(rows[0]).issubset(accepted)


def test_materialized_payload_output_dir_writes_codec_op_blobs(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()
    payload = b"materialized-section-bytes"

    class FakeEncodeResult:
        blob = payload
        bytes_out = len(payload)

    class Op_TestMaterialize:
        def __init__(self, *, q: int = 1) -> None:
            self.q = q

        def encode(self, state_dict, context):
            _ = state_dict, context
            return FakeEncodeResult()

        def decode(self, blob, context):
            _ = blob, context
            return {}

        def validate(self, before, after, context):
            _ = before, after, context
            return types.SimpleNamespace(ok=True, metrics={})

    fake_module = types.ModuleType("fake_materialized_codec_op")
    fake_module.Op_TestMaterialize = Op_TestMaterialize
    sys.modules[fake_module.__name__] = fake_module

    state_dict_path = tmp_path / "state.pt"
    torch.save({"dummy": torch.zeros(1)}, state_dict_path)
    manifest_path = tmp_path / "manifest.json"
    payload_dir = tmp_path / "payloads"

    rc = mod.main([
        "--module", fake_module.__name__,
        "--class", "Op_TestMaterialize",
        "--state-dict-path", str(state_dict_path),
        "--param-grid", '{"q": [2]}',
        "--anchor-d-seg", "0.000671",
        "--anchor-d-pose", "0.0000336",
        "--anchor-archive-bytes", "185578",
        "--baseline-substream-bytes", "30",
        "--baseline-substream-role", "decoder_packed_brotli",
        "--label-prefix", "mat_payload",
        "--output", str(manifest_path),
        "--materialized-payload-output-dir", str(payload_dir),
        "--materialized-payload-contract", "pr106_decoder_packed_brotli",
    ])

    assert rc == 0
    row = json.loads(manifest_path.read_text(encoding="utf-8"))["candidates"][0]
    materialized_path = pathlib.Path(row["materialized_payload_path"])
    assert materialized_path.is_file()
    assert materialized_path.read_bytes() == payload
    assert row["materialized_payload_bytes"] == len(payload)
    assert row["materialized_payload_sha256"]
    assert row["materialized_payload_contract"] == "pr106_decoder_packed_brotli"


def test_pr101_substitution_updates_manifest_and_meta_lagrangian_rows(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()

    decoder_len = 162_164
    latent_len = 15_387
    replacement = b"\x44" * decoder_len

    class FakeEncodeResult:
        blob = replacement
        bytes_out = len(replacement)

    class Op1_PR101SplitBrotli:
        def __init__(self, *, brotli_quality: int = 11) -> None:
            self.brotli_quality = brotli_quality

        def encode(self, state_dict, context):
            _ = state_dict, context
            return FakeEncodeResult()

        def decode(self, blob, context):
            _ = blob, context
            return {}

        def validate(self, before, after, context):
            _ = before, after, context
            return types.SimpleNamespace(ok=True, metrics={})

    fake_module = types.ModuleType("fake_pr101_codec_op")
    fake_module.Op1_PR101SplitBrotli = Op1_PR101SplitBrotli
    sys.modules[fake_module.__name__] = fake_module

    source_archive = tmp_path / "source_pr101.zip"
    inner = (b"\x11" * decoder_len) + (b"\x22" * latent_len) + b"\x33" * 607
    info = zipfile.ZipInfo("x")
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(source_archive, "w") as zf:
        zf.writestr(info, inner)

    state_dict_path = tmp_path / "state.pt"
    torch.save({"dummy": torch.zeros(1)}, state_dict_path)
    manifest_path = tmp_path / "manifest.json"
    meta_path = tmp_path / "meta.json"
    archive_root = tmp_path / "archives"

    rc = mod.main([
        "--module", fake_module.__name__,
        "--class", "Op1_PR101SplitBrotli",
        "--state-dict-path", str(state_dict_path),
        "--param-grid", '{"brotli_quality": [11]}',
        "--anchor-d-seg", "0.000671",
        "--anchor-d-pose", "0.0000336",
        "--anchor-archive-bytes", "185578",
        "--baseline-substream-bytes", str(decoder_len),
        "--baseline-substream-role", "decoder_blob",
        "--label-prefix", "pr101_test",
        "--output", str(manifest_path),
        "--meta-lagrangian-output", str(meta_path),
        "--substrate-archive-pr101", str(source_archive),
        "--substituted-archive-output-dir", str(archive_root),
    ])

    assert rc == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["candidates"][0]
    assert row["archive_path"]
    assert pathlib.Path(row["archive_path"]).is_file()
    assert row["archive_size_bytes"] == pathlib.Path(row["archive_path"]).stat().st_size
    assert row["expected_archive_size_bytes"] == row["archive_size_bytes"]
    assert row["archive_sha256"] == row["expected_archive_sha256"]
    assert row["archive_substitution_report_path"].endswith("substitution_report.json")
    assert row["archive_member_name"] == "x"
    assert row["source_archive_sha256"]
    assert row["source_inner_member_sha256"]
    assert row["replacement_decoder_blob_sha256"]
    assert row["input_latent_blob_sha256"] == row["output_latent_blob_sha256"]
    assert row["input_sidecar_blob_sha256"] == row["output_sidecar_blob_sha256"]
    assert "archive_substitution_surgery_pending" not in row["blockers"]
    assert "exact_runtime_parity_not_supplied" in row["blockers"]
    assert row["ready_for_exact_eval_dispatch"] is False

    meta_rows = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_rows[0]["candidate_id"] == row["candidate_id"]
    assert meta_rows[0]["archive_path"] == row["archive_path"]
    assert meta_rows[0]["archive_bytes"] == row["archive_size_bytes"]


def test_pr101_archive_state_source_can_replace_loose_pt(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    mod = _load_tool_module()

    decoder_len = 162_164
    latent_len = 15_387
    replacement = b"\x55" * decoder_len

    class FakeEncodeResult:
        blob = replacement
        bytes_out = len(replacement)

    class Op1_PR101SplitBrotli:
        def __init__(self, *, brotli_quality: int = 11) -> None:
            self.brotli_quality = brotli_quality

        def encode(self, state_dict, context):
            _ = state_dict, context
            return FakeEncodeResult()

        def decode(self, blob, context):
            _ = blob, context
            return {}

        def validate(self, before, after, context):
            _ = before, after, context
            return types.SimpleNamespace(ok=True, metrics={})

    fake_module = types.ModuleType("fake_pr101_archive_state_codec_op")
    fake_module.Op1_PR101SplitBrotli = Op1_PR101SplitBrotli
    sys.modules[fake_module.__name__] = fake_module

    source_archive = tmp_path / "source_pr101.zip"
    inner = (b"\x11" * decoder_len) + (b"\x22" * latent_len) + b"\x33" * 607
    info = zipfile.ZipInfo("x")
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(source_archive, "w") as zf:
        zf.writestr(info, inner)

    state_source = {
        "kind": "pr101_archive_decoder_blob",
        "archive_path": source_archive.as_posix(),
        "inner_member_name": "x",
        "decoder_blob_bytes": decoder_len,
        "decoder_blob_sha256": "decoder-sha",
    }
    monkeypatch.setattr(
        mod,
        "_load_state_dict_from_pr101_archive",
        lambda archive: ({"dummy": torch.zeros(1)}, state_source),
    )

    manifest_path = tmp_path / "manifest.json"
    rc = mod.main([
        "--module", fake_module.__name__,
        "--class", "Op1_PR101SplitBrotli",
        "--state-dict-from-pr101-archive", str(source_archive),
        "--param-grid", '{"brotli_quality": [11]}',
        "--anchor-d-seg", "0.000671",
        "--anchor-d-pose", "0.0000336",
        "--anchor-archive-bytes", "178258",
        "--baseline-substream-bytes", str(decoder_len),
        "--baseline-substream-role", "decoder_blob",
        "--label-prefix", "pr101_archive_state",
        "--output", str(manifest_path),
        "--substrate-archive-pr101", str(source_archive),
        "--substituted-archive-output-dir", str(tmp_path / "archives"),
    ])

    assert rc == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["state_dict_source"] == state_source
    row = manifest["candidates"][0]
    assert row["archive_path"]
    assert row["archive_member_name"] == "x"
    assert row["ready_for_exact_eval_dispatch"] is False


def test_pr101_archive_backed_sweep_rejects_nonexistent_pose_baseline(
    tmp_path: pathlib.Path,
) -> None:
    mod = _load_tool_module()

    class FakeEncodeResult:
        blob = b"x"
        bytes_out = 1

    class Op_KLPoseStream:
        def __init__(self, *, n_components: int = 2) -> None:
            self.n_components = n_components

        def encode(self, state_dict, context):
            _ = state_dict, context
            return FakeEncodeResult()

        def decode(self, blob, context):
            _ = blob, context
            return {}

        def validate(self, before, after, context):
            _ = before, after, context
            return types.SimpleNamespace(ok=True, metrics={})

    fake_module = types.ModuleType("fake_pr101_pose_codec_op")
    fake_module.Op_KLPoseStream = Op_KLPoseStream
    sys.modules[fake_module.__name__] = fake_module

    source_archive = tmp_path / "source_pr101.zip"
    info = zipfile.ZipInfo("x")
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(source_archive, "w") as zf:
        zf.writestr(info, b"x" * (162_164 + 15_387 + 607))
    state_dict_path = tmp_path / "state.pt"
    torch.save({"poses_se3": torch.zeros(2, 6)}, state_dict_path)

    try:
        mod.main([
            "--module", fake_module.__name__,
            "--class", "Op_KLPoseStream",
            "--state-dict-path", str(state_dict_path),
            "--param-grid", '{"n_components": [2]}',
            "--anchor-d-seg", "0.000671",
            "--anchor-d-pose", "0.0000336",
            "--anchor-archive-bytes", "178258",
            "--baseline-substream-bytes", "3600",
            "--baseline-substream-role", "pose_blob",
            "--label-prefix", "bad_pr101_pose",
            "--output", str(tmp_path / "manifest.json"),
            "--substrate-archive-pr101", str(source_archive),
        ])
    except SystemExit as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected SystemExit")

    assert "PR101 has no separate pose" in message
