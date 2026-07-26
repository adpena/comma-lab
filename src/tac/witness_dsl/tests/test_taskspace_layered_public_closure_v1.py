from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
)
from tac.witness_control import taskspace_fresh_scorer_plane_materializer_v1
from tac.witness_dsl import taskspace_layered_public_closure_v1 as closure
from tac.witness_dsl.taskspace_fresh_selected_plane_codec_v1 import (
    AGGREGATE_SCHEMA,
)
from tac.witness_dsl.taskspace_layered_public_closure_v1 import (
    AUTH_SCHEMA,
    BUILD_SCHEMA,
    ClosureError,
    _verify_fresh_operand_receipt,
    canonical_json,
    promote,
    sha256_file,
    stage_exact_eval,
)


def _load_public_runtime():
    path = Path(__file__).parents[4] / "submissions/robust_current/taskspace_layered_public/inflate.py"
    spec = importlib.util.spec_from_file_location("taskspace_layered_public_inflate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(path: Path, role: str | None = None) -> dict:
    payload = path.read_bytes()
    row = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if role is not None:
        row["role"] = role
    return row


def test_typed_separate_channel_layer_assembles_rgb(monkeypatch, tmp_path: Path) -> None:
    runtime = _load_public_runtime()
    streams = []
    decoded = {}
    stream_dir = tmp_path / "streams"
    stream_dir.mkdir()
    for index, semantic in enumerate(("r", "g", "b")):
        path = stream_dir / f"{semantic}.bin"
        payload = bytes([index + 1])
        path.write_bytes(payload)
        row = {
            "path": f"streams/{path.name}",
            "bytes": 1,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "frame_count": 2,
            "codec_name": "ffv1",
            "coded_pixel_format": "gray",
            "decoded_pixel_format": "gray",
            "rgb_conversion_path": "PyAV-VideoFrame.to_ndarray-rgb24",
            "semantic_channels": semantic,
            "decoded_sha256": hashlib.sha256(bytes([index])).hexdigest(),
        }
        streams.append(row)
        decoded[path.name] = np.full((2, 384, 512), index, dtype=np.uint8)
    monkeypatch.setattr(
        runtime,
        "_decode_stream",
        lambda path, _row: decoded[path.name],
    )
    value, rows = runtime._decode_layer(
        tmp_path,
        {"packing": "separate_gray8_rgb", "streams": streams},
        "layer",
    )
    assert value.shape == (2, 384, 512, 3)
    assert np.array_equal(value[0, 0, 0], np.array([0, 1, 2], dtype=np.uint8))
    assert rows == streams


def _exact_g52_receipt_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    bundle = tmp_path / "counted_stream_bundle.zip"
    bundle.write_bytes(b"fresh-counted-operands")
    provider = tmp_path / "g51-aggregate.json"
    provider.write_bytes(b"sealed-g51-provider")
    for index in range(5):
        path = tmp_path / "stages" / f"stage_{index:02d}" / "receipt.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"stage-{index}".encode())
    final = tmp_path / "final" / "receipt.json"
    final.parent.mkdir()
    final.write_bytes(b"final")
    public_decode = {
        "module": "av",
        "version": "17.0.0",
        "library_versions": {},
        "decode_path": ("av.open->container.decode(video=0)->native-gbrp-extract-or-VideoFrame.to_ndarray(rgb24)"),
        "thread_count": 1,
        "authority": "public-runtime parse-back",
        "required_public_version": "17.0.0",
    }
    receipt = {
        "schema": AGGREGATE_SCHEMA,
        "experiment_schema": "taskspace_fresh_selected_plane_codec.v1",
        "status": "closed_pending_public_receiver_and_exact_eval",
        "research_only": True,
        "candidate_lineage_allowed": True,
        "historical_payload_reused": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_delta": "UNMOVED",
        "config_sha256": "1" * 64,
        "pair_count": 600,
        "stage_count": 5,
        "stage_receipt_sha256": [
            sha256_file(tmp_path / "stages" / f"stage_{index:02d}" / "receipt.json") for index in range(5)
        ],
        "operand_provider": {
            "aggregate_receipt_path": str(provider),
            "aggregate_receipt_sha256": sha256_file(provider),
        },
        "representation_mode": "DIRECT_TASK_LAYERED",
        "program_residual_layered": {
            "available": False,
            "status": "blocked_missing_fresh_semantic_base_bytes",
            "v15_composition_claim": False,
        },
        "pose_custody": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
        "pose_authority": False,
        "codec": {},
        "public_decode": public_decode,
        "upstream_pyav_lock": {
            "path": str(tmp_path / "upstream/uv.lock"),
            "bytes": 0,
            "sha256": "5" * 64,
            "package": "av",
            "version": "17.0.0",
        },
        "endpoint": {},
        "final_recode_receipt_sha256": sha256_file(final),
        "counted_stream_bundle": {
            "path": str(bundle),
            "bytes": bundle.stat().st_size,
            "sha256": sha256_file(bundle),
            "payload_stream_bytes": 0,
            "container_and_manifest_bytes": bundle.stat().st_size,
            "manifest_sha256": "2" * 64,
            "manifest": {},
        },
        "rate_term_if_used_as_exact_archive": 0.0,
        "dynamic_frontier": {},
        "admission_rule": "exact public score must be strictly below dynamic_frontier.target_score",
        "next_authority_gate": "public",
    }
    receipt_path = tmp_path / "aggregate_receipt.json"
    receipt_path.write_bytes(canonical_json(receipt))
    return receipt_path, bundle, receipt


def test_exact_g52_gate_rejects_historical_plane_input(tmp_path: Path) -> None:
    receipt_path, bundle, receipt = _exact_g52_receipt_fixture(tmp_path)
    receipt["historical_payload_reused"] = True
    receipt_path.write_bytes(canonical_json(receipt))
    with pytest.raises(ClosureError, match="fresh DIRECT_TASK_LAYERED"):
        _verify_fresh_operand_receipt(
            receipt_path,
            sha256_file(receipt_path),
            bundle,
            sha256_file(bundle),
        )


def test_exact_g51_g52_gate_accepts_fresh_direct_lineage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    receipt_path, bundle, _ = _exact_g52_receipt_fixture(tmp_path)
    fake_loader = type(
        "FakeLoader",
        (),
        {
            "receipt": {
                "schema": "tac.taskspace_fresh_scorer_plane_aggregate.v1",
                "pair_count": 600,
                "aggregate_receipt_sha256": "3" * 64,
                "stage_digest_chain_sha256": "4" * 64,
                "run_id": "fresh_n600_fixture",
            }
        },
    )()
    monkeypatch.setattr(
        taskspace_fresh_scorer_plane_materializer_v1.FreshScorerPlaneOperandLoaderV1,
        "open",
        lambda *_args, **_kwargs: fake_loader,
    )
    verified = _verify_fresh_operand_receipt(
        receipt_path,
        sha256_file(receipt_path),
        bundle,
        sha256_file(bundle),
    )
    assert verified["g51_run_id"] == "fresh_n600_fixture"
    assert verified["historical_payload_reused"] is False


@pytest.mark.parametrize(
    ("encoder", "encoder_args", "expected_codec", "expected_coded_format"),
    [
        (
            "libx264rgb",
            ["-preset", "ultrafast", "-crf", "0"],
            "h264",
            "gbrp",
        ),
        (
            "libx265",
            [
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv444p",
                "-x265-params",
                "lossless=1:pools=none:frame-threads=1:log-level=error",
            ],
            "hevc",
            "yuv444p",
        ),
    ],
)
def test_public_runtime_decodes_typed_non_ivf_streams(
    encoder: str,
    encoder_args: list[str],
    expected_codec: str,
    expected_coded_format: str,
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable for test-fixture generation")
    runtime = _load_public_runtime()
    yy, xx = np.indices((384, 512), dtype=np.uint16)
    frames = np.empty((2, 384, 512, 3), dtype=np.uint8)
    frames[0, ..., 0] = xx % 256
    frames[0, ..., 1] = yy % 256
    frames[0, ..., 2] = (xx + yy) % 256
    frames[1] = 255 - frames[0]
    source = tmp_path / "source.rgb"
    source.write_bytes(frames.tobytes())
    encoded = tmp_path / f"{encoder}.mkv"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "512x384",
        "-framerate",
        "20",
        "-i",
        str(source),
        "-frames:v",
        "2",
        "-threads",
        "1",
        "-c:v",
        encoder,
        *encoder_args,
        str(encoded),
    ]
    subprocess.run(command, check=True, capture_output=True)
    parseback = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(encoded),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    row = {
        "path": f"streams/{encoded.name}",
        "bytes": encoded.stat().st_size,
        "sha256": sha256_file(encoded),
        "frame_count": 2,
        "codec_name": expected_codec,
        "coded_pixel_format": expected_coded_format,
        "decoded_pixel_format": "rgb24",
        "rgb_conversion_path": (
            "native-gbrp-plane-extraction-and-rgb-reorder.v1"
            if encoder == "libx264rgb"
            else "PyAV-VideoFrame.to_ndarray-rgb24"
        ),
        "semantic_channels": "rgb",
        "decoded_sha256": hashlib.sha256(parseback).hexdigest(),
    }
    decoded = runtime._decode_stream(encoded, row)
    assert decoded.shape == (2, 384, 512, 3)
    assert hashlib.sha256(decoded.tobytes()).hexdigest() == row["decoded_sha256"]


def test_public_factor2_realization_matches_shared_authority() -> None:
    runtime = _load_public_runtime()
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    generator = np.random.default_rng(1700)
    plane = generator.integers(
        0,
        256,
        size=(384, 512, 3),
        dtype=np.uint8,
    )
    expected = realize_factor2_uint8_scorer_plane(operator, plane)
    assert np.array_equal(runtime.realize_factor2(plane), expected)


def test_promotion_requires_exact_bound_archive_and_strict_live_target(monkeypatch, tmp_path: Path) -> None:
    preview = tmp_path / "archive.preview.zip"
    preview.write_bytes(b"exact-preview")
    raw_sha = hashlib.sha256(b"raw").hexdigest()
    build_receipt = {
        "schema": BUILD_SCHEMA,
        "research_only": True,
        "candidate_lineage_allowed": True,
        "promotion_eligible": False,
        "score_claim": False,
        "pointer_delta": "UNMOVED",
        "archive_preview": {
            "path": str(preview),
            "bytes": preview.stat().st_size,
            "sha256": sha256_file(preview),
        },
        "source_custody": {"expected_public_raw_sha256": raw_sha},
        "runtime_dependency": {"pyav_version": "17.0.0"},
    }
    target = 0.165
    pointer_sha = hashlib.sha256(b"pointer").hexdigest()
    selection_rule = "minimum_qualifying_exact_score.v1"
    monkeypatch.setattr(
        closure,
        "_live_frontier",
        lambda _root: (
            object(),
            {
                "target_score": target,
                "pointer_sha256": pointer_sha,
                "selection_rule": selection_rule,
            },
        ),
    )
    monkeypatch.setattr(
        closure,
        "verify_dynamic_frontier_target_snapshot",
        lambda _snapshot: _snapshot,
    )
    build_receipt["competitive_target"] = {
        "pointer_sha256": pointer_sha,
        "selection_rule": selection_rule,
    }
    staged_archive, staging = stage_exact_eval(preview, build_receipt)
    assert staged_archive.name == "archive.zip"
    assert staging["promotion_eligible"] is False
    assert staging["score_claim"] is False
    evidence = {}
    for name in (
        "upstream_snapshot_receipt",
        "evaluate_stdout_log",
        "authority_hardware_receipt",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        evidence[name] = _record(path)
    upstream_evaluate = tmp_path / "upstream/evaluate.py"
    upstream_evaluate.parent.mkdir()
    upstream_evaluate.write_bytes(b"evaluator")
    evidence["upstream_evaluate"] = _record(upstream_evaluate)
    evidence["exact_eval_staging_receipt"] = _record(Path(staging["receipt_path"]))
    pyav = {
        "version": "17.0.0",
        "library_versions": {},
        "thread_count": 1,
        "decode_path": (
            "av.open->container.decode(video=0)->typed native-gbrp-plane-extraction-or-VideoFrame.to_ndarray(rgb24)"
        ),
    }
    stage_receipts = [{"raw_sha256": raw_sha}]
    inflate_payload = canonical_json(
        {
            "schema": "taskspace_layered_public_inflate_receipt.v1",
            "manifest_sha256": "b" * 64,
            "raw_sha256": raw_sha,
            "raw_bytes": 3_662_409_600,
            "pair_count": 600,
            "frame_count": 1200,
            "pyav": pyav,
            "initial_output_root_was_clean": True,
            "stage_count": 1,
            "stage_fresh_decode_count": 1,
            "stage_resume_count": 0,
            "final_assembly_action": "fresh_assembly",
            "invocation_mode": "fresh",
            "output_root_identity_sha256": "c" * 64,
            "stage_receipts": stage_receipts,
        }
    )
    inflate_rows = []
    for index in range(2):
        path = tmp_path / f"inflate-{index}.json"
        value = json.loads(inflate_payload)
        value["output_root_identity_sha256"] = "c" * 64 if index == 0 else "d" * 64
        path.write_bytes(canonical_json(value))
        inflate_rows.append(_record(path))
    evidence["double_inflate_receipts"] = inflate_rows
    rate = 25.0 * preview.stat().st_size / 37_545_489
    d_seg_at_target = (target - rate) / 100.0
    auth = {
        "schema": AUTH_SCHEMA,
        "archive_sha256": sha256_file(preview),
        "raw_sha256": raw_sha,
        "pair_count": 600,
        "evaluation_entrypoint": "upstream/evaluate.py",
        "authority_axis": "contest-CPU",
        "double_inflate_identical": True,
        "candidate_lineage_allowed": True,
        "score_claim": True,
        "score": target,
        "d_seg": d_seg_at_target,
        "d_pose": 0.0,
        "archive_bytes": preview.stat().st_size,
        "raw_bytes": 3_662_409_600,
        "frame_count": 1200,
        "competitive_target_score": target,
        "canonical_frontier_pointer_sha256": pointer_sha,
        "selection_rule": selection_rule,
        "evidence": evidence,
    }
    auth_path = tmp_path / "auth.json"
    auth_path.write_bytes(canonical_json(auth))
    with pytest.raises(ClosureError, match="promotion refused"):
        promote(preview, build_receipt, auth_path, sha256_file(auth_path))
    auth["d_seg"] = d_seg_at_target - 0.00001
    auth["score"] = 100.0 * auth["d_seg"] + rate
    second_inflate_path = Path(inflate_rows[1]["path"])
    same_root_receipt = json.loads(second_inflate_path.read_bytes())
    same_root_receipt["output_root_identity_sha256"] = "c" * 64
    second_inflate_path.write_bytes(canonical_json(same_root_receipt))
    auth["evidence"]["double_inflate_receipts"][1] = _record(second_inflate_path)
    auth_path.write_bytes(canonical_json(auth))
    with pytest.raises(ClosureError, match="distinct clean-root"):
        promote(preview, build_receipt, auth_path, sha256_file(auth_path))
    same_root_receipt["output_root_identity_sha256"] = "d" * 64
    second_inflate_path.write_bytes(canonical_json(same_root_receipt))
    auth["evidence"]["double_inflate_receipts"][1] = _record(second_inflate_path)
    auth_path.write_bytes(canonical_json(auth))
    archive = promote(preview, build_receipt, auth_path, sha256_file(auth_path))
    assert archive.read_bytes() == preview.read_bytes()


def test_public_inflate_is_idempotent_over_preserved_stage_receipts(monkeypatch, tmp_path: Path) -> None:
    runtime = _load_public_runtime()
    archive_root = tmp_path / "archive"
    output_root = tmp_path / "output"
    archive_root.mkdir()
    names = tmp_path / "names.txt"
    names.write_text("0.mp4\n")
    stage = tmp_path / "stage.raw"
    stage.write_bytes(b"abcdef")
    stage_receipt = {"raw_sha256": hashlib.sha256(b"abcdef").hexdigest()}
    manifest = {
        "chunks": [{"pair_start": 0, "pair_stop": 1}],
        "expected_raw_sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }
    usage = namedtuple("usage", "total used free")(10**12, 0, 10**12)
    monkeypatch.setattr(runtime, "EXPECTED_RAW_BYTES", 6)
    monkeypatch.setattr(runtime, "load_manifest", lambda _root: (manifest, "a" * 64))
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _root: usage)
    monkeypatch.setattr(runtime, "_build_stage", lambda *_args, **_kwargs: (stage, stage_receipt, "fresh_decode"))
    first = runtime.inflate(archive_root, output_root, names)
    second = runtime.inflate(archive_root, output_root, names)
    assert first["invocation_mode"] == "fresh"
    assert first["initial_output_root_was_clean"] is True
    assert second["invocation_mode"] == "resume"
    assert second["initial_output_root_was_clean"] is False
    assert first["raw_sha256"] == second["raw_sha256"]
    assert (output_root / "0.raw").read_bytes() == b"abcdef"
