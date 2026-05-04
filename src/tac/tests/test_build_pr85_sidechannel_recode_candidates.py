from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_sidechannel_recode_candidates.py"


def _load_script():
    sys.path.insert(0, str(REPO / "experiments"))
    spec = importlib.util.spec_from_file_location("build_pr85_recode_test", SCRIPT)
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
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _runtime_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py \"$@\"\n", encoding="utf-8")
    (path / "inflate.py").write_text("def load_compact_archive_bundle(data_dir):\n    return None\n", encoding="utf-8")
    (path / "range_mask_codec.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    return path


def _lcg_choices(prefix: bytes, *, alphabet: int, seed: int) -> bytes:
    out = bytearray(prefix)
    value = seed
    for _ in range(module.PAIR_COUNT):
        value = (1664525 * value + 1013904223) & 0xFFFFFFFF
        out.append(value % alphabet)
    return bytes(out)


def _fixed_length_brotli(prefix: bytes, *, target_bytes: int, alphabet: int, seed: int, quality: int, lgwin: int) -> bytes:
    encoded = brotli.compress(
        _lcg_choices(prefix, alphabet=alphabet, seed=seed),
        quality=quality,
        lgwin=lgwin,
    )
    assert len(encoded) == target_bytes
    assert brotli.decompress(encoded).startswith(prefix)
    return encoded


def _randmulti_zero_payload() -> bytes:
    return bytes(sum(spec[3] for spec in module.HEADERLESS_RANDMULTI_SPECS))


def _fixture_archive(path: Path, *, frac2_preoptimized: bool = False) -> None:
    raw_frac2 = b"FH2" + bytes([4]) * module.PAIR_COUNT
    frac2 = (
        module._brotli_best(raw_frac2)[0]
        if frac2_preoptimized
        else brotli.compress(raw_frac2, quality=1, lgwin=10)
    )
    segments = {
        "mask": b"QMA9" + b"M" * 1000,
        "model": b"QH0" + b"W" * 1000,
        "pose": b"P1D1" + b"P" * 100,
        "post": brotli.compress(bytes([1]) * module.PAIR_COUNT * 4, quality=5),
        "shift": brotli.compress(b"SH4" + bytes([40]) * module.PAIR_COUNT, quality=1, lgwin=10),
        "frac": brotli.compress(b"FH1" + bytes([4]) * module.PAIR_COUNT, quality=1, lgwin=10),
        "frac2": frac2,
        "frac3": brotli.compress(b"FH3" + bytes([4]) * module.PAIR_COUNT, quality=1, lgwin=10),
        "bias": _fixed_length_brotli(
            b"BD1",
            target_bytes=module.FIXED_V5_LENGTHS["bias"],
            alphabet=3,
            seed=3,
            quality=1,
            lgwin=10,
        ),
        "region": _fixed_length_brotli(
            b"RH1",
            target_bytes=module.FIXED_V5_LENGTHS["region"],
            alphabet=3,
            seed=6,
            quality=2,
            lgwin=16,
        ),
        "randmulti": brotli.compress(_randmulti_zero_payload(), quality=5),
    }
    header = b"".join(_u24(len(segments[name])) for name in module.SEGMENT_ORDER[:8])
    _zip(path, header + b"".join(segments[name] for name in module.SEGMENT_ORDER))


def test_builds_lossless_recode_manifest_with_decoded_parity(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    runtime_dir = _runtime_dir(tmp_path / "runtime")
    _fixture_archive(source)

    payload = module.build_candidates(
        source,
        out_dir,
        policy_ids=["segment_bias_best"],
        replay_runtime_dir=runtime_dir,
    )

    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    manifest = payload["candidates"][0]
    assert manifest["policy_id"] == "segment_bias_best"
    assert manifest["candidate"]["member_name"] == "x"
    assert manifest["candidate"]["zip_stored"] is True
    assert manifest["runtime_dependency_closure"]["status"] == "passed"
    assert {row["name"] for row in manifest["runtime_dependency_closure"]["required_files"]} == {
        "inflate.sh",
        "inflate.py",
        "range_mask_codec.cpp",
    }
    assert manifest["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
    assert manifest["decoded_parity_metadata"]["status"] == "passed"
    assert manifest["header_mode"] == "explicit_30"
    assert manifest["fixed_length_parser_safety"] == {
        "explicit_30byte_header_used": True,
        "reason": "candidate uses public-runtime explicit bias/region length parser branch",
        "status": "passed",
        "v5_fixed_lengths_preserved": False,
    }
    transform = manifest["transforms"][0]
    assert transform["segment"] == "bias"
    assert transform["decoded_parity_status"] == "passed"
    assert transform["source_semantic_sha256"] == transform["candidate_semantic_sha256"]
    assert transform["candidate_segment_bytes"] < transform["source_segment_bytes"]
    assert (out_dir / "segment_bias_best" / "archive.zip").is_file()
    assert (out_dir / "segment_bias_best" / "manifest.json").is_file()


def test_v5_fixed_length_segments_fail_closed_when_size_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _fixture_archive(source)
    _bundle, segments = module._parse_bundle(module._read_pr85_archive(source)[1])
    segments = dict(segments)
    segments["bias"] = segments["bias"][:-1]

    with pytest.raises(ValueError, match="changed fixed-length segment 'bias'"):
        module._pack_bundle(segments, header_mode="v5")


def test_noop_candidate_is_marked_planning_only(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    runtime_dir = _runtime_dir(tmp_path / "runtime")
    _fixture_archive(source, frac2_preoptimized=True)

    payload = module.build_candidates(
        source,
        out_dir,
        policy_ids=["segment_frac2_best"],
        replay_runtime_dir=runtime_dir,
    )

    manifest = payload["candidates"][0]
    assert manifest["policy_id"] == "segment_frac2_best"
    assert manifest["noop"] is True
    assert manifest["dispatch_gate"] == "planning_only/no_remote_dispatch"
    assert manifest["decoded_parity_metadata"]["status"] == "passed"
    assert manifest["byte_delta_vs_source_archive"] == 0


def test_unknown_policy_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    _fixture_archive(source)

    with pytest.raises(ValueError, match="unknown policy"):
        module.build_candidates(source, tmp_path / "out", policy_ids=["bogus"])


def test_missing_runtime_closure_blocks_dispatch_gate(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    _fixture_archive(source)

    payload = module.build_candidates(
        source,
        out_dir,
        policy_ids=["segment_bias_best"],
        replay_runtime_dir=tmp_path / "missing-runtime",
    )

    manifest = payload["candidates"][0]
    assert manifest["runtime_dependency_closure"]["status"] == "failed"
    assert manifest["runtime_dependency_closure"]["missing_files"] == [
        "inflate.sh",
        "inflate.py",
        "range_mask_codec.cpp",
    ]
    assert manifest["dispatch_gate"] == "planning_only/no_remote_dispatch"
    assert payload["dispatchable_candidate_count"] == 0


def test_cli_writes_summary(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    runtime_dir = _runtime_dir(tmp_path / "runtime")
    _fixture_archive(source)

    assert module.main(
        [
            "--archive",
            str(source),
            "--out-dir",
            str(out_dir),
            "--replay-runtime-dir",
            str(runtime_dir),
            "--policy",
            "segment_bias_best",
        ]
    ) == 0

    assert (out_dir / "candidate_summary.json").is_file()
    assert '"policy_id": "segment_bias_best"' in capsys.readouterr().out
