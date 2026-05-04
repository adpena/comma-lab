from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

from tac.qma9_range_mask_contract import encode_qma9_mask, sha256_bytes, write_stored_single_member_zip


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_qma9_range_mask_candidate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_qma9_range_mask_candidate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_candidate_reencodes_raw_mask_and_emits_deterministic_archive(tmp_path: Path) -> None:
    script = _load_script()
    raw = bytes((y + x) % 5 for y in range(3) for x in range(4))
    raw_path = tmp_path / "mask.raw"
    raw_path.write_bytes(raw)
    model = tmp_path / "model.br"
    pose = tmp_path / "pose.br"
    router = tmp_path / "router.bin"
    model.write_bytes(b"model")
    pose.write_bytes(b"pose")
    router.write_bytes(b"router")

    manifest = script.build_candidate(
        output_dir=tmp_path / "out",
        candidate_id="tiny",
        raw_mask_path=raw_path,
        model_payload_path=model,
        pose_payload_path=pose,
        router_payload_path=router,
        template_pr81_archive=None,
        frame_count=1,
        width=3,
        height=4,
    )

    candidate_dir = tmp_path / "out" / "tiny"
    written = json.loads((candidate_dir / "manifest.json").read_text())
    assert written == manifest
    assert manifest["score_claim"] is False
    assert manifest["dispatch_performed"] is False
    assert manifest["qma9_header"]["decoded_mask_bytes"] == 12
    assert manifest["segments"][0]["source"]["encoded_by"] == "pure_python_qma9_codec"

    with zipfile.ZipFile(candidate_dir / "archive.zip") as zf:
        payload = zf.read("p")
    qma9 = encode_qma9_mask(raw, frame_count=1, width=3, height=4)
    assert payload == qma9 + b"model" + b"pose" + b"router"
    assert manifest["archive"]["payload_sha256"] == sha256_bytes(payload)

    second = script.build_candidate(
        output_dir=tmp_path / "out",
        candidate_id="tiny_again",
        raw_mask_path=raw_path,
        model_payload_path=model,
        pose_payload_path=pose,
        router_payload_path=router,
        template_pr81_archive=None,
        frame_count=1,
        width=3,
        height=4,
    )
    assert second["archive"]["sha256"] == manifest["archive"]["sha256"]


def test_build_candidate_can_reuse_template_pr81_style_segments(tmp_path: Path) -> None:
    script = _load_script()
    qma9 = encode_qma9_mask(b"\x00\x01\x02\x03", frame_count=1, width=2, height=2)
    template = tmp_path / "template.zip"
    write_stored_single_member_zip(template, qma9 + b"abcde" + b"pose" + b"router")

    old_sizes = (
        script.PR81_RANGE_MASK_BYTES,
        script.PR81_MODEL_BYTES,
        script.PR81_POSE_BYTES,
        script.PR81_ROUTER_BYTES,
    )
    script.PR81_RANGE_MASK_BYTES = len(qma9)
    script.PR81_MODEL_BYTES = 5
    script.PR81_POSE_BYTES = 4
    script.PR81_ROUTER_BYTES = 6
    try:
        manifest = script.build_candidate(
            output_dir=tmp_path / "out",
            candidate_id="template_reuse",
            template_pr81_archive=template,
            frame_count=1,
            width=2,
            height=2,
        )
    finally:
        (
            script.PR81_RANGE_MASK_BYTES,
            script.PR81_MODEL_BYTES,
            script.PR81_POSE_BYTES,
            script.PR81_ROUTER_BYTES,
        ) = old_sizes

    assert manifest["archive"]["payload_bytes"] == len(qma9) + 15
    assert manifest["segments"][0]["source"]["source_kind"] == "template_qma9_payload_reuse_for_profile_or_baseline"
    assert [row["source"] for row in manifest["segments"][1:]] == [
        "template_pr81_archive",
        "template_pr81_archive",
        "template_pr81_archive",
    ]
