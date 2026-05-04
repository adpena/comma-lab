from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import brotli
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_post_motion_group_policy_candidates.py"


def _load_script():
    sys.path.insert(0, str(REPO / "experiments"))
    spec = importlib.util.spec_from_file_location("build_pr85_post_motion_group_policy_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _fixture_archive(path: Path) -> None:
    post = b"".join(bytes([(stage * 10 + pair) % 251 for pair in range(module.PAIR_COUNT)]) for stage in range(4))
    shift_values = bytes([40 if pair % 3 else 41 for pair in range(module.PAIR_COUNT)])
    frac_values = bytes([4 if pair % 5 else 6 for pair in range(module.PAIR_COUNT)])
    frac2_values = bytes([4 if pair % 7 else 2 for pair in range(module.PAIR_COUNT)])
    frac3_values = bytes([4 if pair % 11 else 7 for pair in range(module.PAIR_COUNT)])
    randmulti_raw = bytes(sum(spec[3] for spec in module.recode.HEADERLESS_RANDMULTI_SPECS))
    segments = {
        "mask": b"QMA9" + b"M" * 1001,
        "model": b"QH0" + b"W" * 1001,
        "pose": b"P1D1" + b"P" * 101,
        "post": brotli.compress(post, quality=5),
        "shift": brotli.compress(module.recode._encode_delta_choice(b"SD4", shift_values, default_choice=40), quality=5),
        "frac": brotli.compress(module.recode._encode_sparse_choice(b"FV1", frac_values, default_choice=4), quality=5),
        "frac2": brotli.compress(b"FH2" + frac2_values, quality=5),
        "frac3": brotli.compress(module.recode._encode_delta_choice(b"FD3", frac3_values, default_choice=4), quality=5),
        "bias": b"B" * module.recode.FIXED_V5_LENGTHS["bias"],
        "region": b"R" * module.recode.FIXED_V5_LENGTHS["region"],
        "randmulti": brotli.compress(randmulti_raw, quality=5),
    }
    header = b"".join(_u24(len(segments[name])) for name in module.recode.SEGMENT_ORDER[:8])
    payload = header + b"".join(segments[name] for name in module.recode.SEGMENT_ORDER)
    _zip(path, payload)


def _segments(archive: Path):
    _source_archive, raw = module.recode._read_pr85_archive(archive)
    _bundle, segments = module.recode._parse_bundle(raw)
    return segments


def _post_stage_choices(segment: bytes) -> list[bytes]:
    decoded = module.recode._decode_segment_raw("post", segment)
    return [choices for _stage_id, choices in module._post_stage_records(module.recode._segment_semantics("post", decoded))]


def test_preserves_selected_post_stages_and_neutralizes_unselected_stage(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    _fixture_archive(source)

    payload = module.build_candidates(
        archive=source,
        out_dir=out_dir,
        policy_ids=["preserve_post123_motion"],
    )

    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert payload["candidate_count"] == 1
    manifest = payload["candidates"][0]
    assert manifest["policy_id"] == "preserve_post123_motion"
    assert manifest["promotion_status"] == "non_promotable_pending_exact_cuda_eval"
    assert manifest["eval_ready"] is True
    assert manifest["candidate"]["member_name"] == "x"
    assert manifest["candidate"]["zip_stored"] is True
    assert manifest["source_member"]["member_sha256"] == payload["source_member"]["member_sha256"]
    assert manifest["selected_groups"] == [
        "post_stage1",
        "post_stage2",
        "post_stage3",
        "motion_shift",
        "motion_frac",
        "motion_frac2",
        "motion_frac3",
    ]
    assert manifest["neutralized_groups"] == ["post_stage4"]
    assert "randmulti" in manifest["preserved_non_target_segments"]
    assert manifest["candidate_bundle_validation"]["status"] == "passed"

    source_segments = _segments(source)
    candidate_segments = _segments(out_dir / "preserve_post123_motion" / "archive.zip")
    source_stages = _post_stage_choices(source_segments["post"])
    candidate_stages = _post_stage_choices(candidate_segments["post"])
    assert candidate_stages[:3] == source_stages[:3]
    assert candidate_stages[3] == bytes(module.PAIR_COUNT)
    for name in ("shift", "frac", "frac2", "frac3", "randmulti"):
        assert candidate_segments[name] == source_segments[name]


def test_preserves_post_and_shift_but_neutralizes_fractional_motion(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    _fixture_archive(source)

    payload = module.build_candidates(
        archive=source,
        out_dir=out_dir,
        policy_ids=["preserve_post_all_shift"],
    )

    manifest = payload["candidates"][0]
    assert manifest["policy_id"] == "preserve_post_all_shift"
    assert manifest["neutralized_groups"] == ["motion_frac", "motion_frac2", "motion_frac3"]
    by_segment = {row["segment"]: row for row in manifest["transforms"]}
    assert by_segment["post"]["noop_segment"] is True
    assert by_segment["shift"]["selected_groups"] == ["motion_shift"]
    assert by_segment["frac"]["neutralized_groups"] == ["motion_frac"]
    assert by_segment["frac2"]["neutralized_groups"] == ["motion_frac2"]
    assert by_segment["frac3"]["neutralized_groups"] == ["motion_frac3"]

    candidate_segments = _segments(out_dir / "preserve_post_all_shift" / "archive.zip")
    frac_values = module.recode._decode_choice_semantics("frac", brotli.decompress(candidate_segments["frac"]))
    frac2_values = module.recode._decode_choice_semantics("frac2", brotli.decompress(candidate_segments["frac2"]))
    frac3_values = module.recode._decode_choice_semantics("frac3", brotli.decompress(candidate_segments["frac3"]))
    assert frac_values == bytes([4]) * module.PAIR_COUNT
    assert frac2_values == bytes([4]) * module.PAIR_COUNT
    assert frac3_values == bytes([4]) * module.PAIR_COUNT


def test_group_policy_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="duplicate selected group"):
        module._validate_selected_groups(["post_stage1", "post_stage1"])
    with pytest.raises(ValueError, match="unknown selected group"):
        module._validate_selected_groups(["randmulti_group0"])
    with pytest.raises(ValueError, match="unknown policy"):
        module.build_candidates(policy_ids=["bogus"])


def test_cli_writes_summary(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    _fixture_archive(source)

    assert module.main(
        [
            "--archive",
            str(source),
            "--out-dir",
            str(out_dir),
            "--policy",
            "preserve_post123_motion",
        ]
    ) == 0

    assert (out_dir / "candidate_summary.json").is_file()
    assert (out_dir / "preserve_post123_motion" / "archive.zip").is_file()
    summary = json.loads((out_dir / "candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_count"] == 1
    assert '"policy_id": "preserve_post123_motion"' in capsys.readouterr().out
