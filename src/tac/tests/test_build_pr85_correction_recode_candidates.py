from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_correction_recode_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_correction_recode_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _runtime_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "inflate.sh").write_text("#!/usr/bin/env bash\npython inflate.py \"$@\"\n", encoding="utf-8")
    (path / "range_mask_codec.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (path / "inflate.py").write_text("\n".join(module.REQUIRED_RUNTIME_TOKENS), encoding="utf-8")
    return path


def _plan_json(path: Path) -> Path:
    payload = {
        "schema": "pr85_correction_atom_waterfill_plan_v1",
        "candidate_policies": {
            "policies": [
                {
                    "candidate_policy_id": "decoded_parity_recode_all_correction_streams",
                    "eval_gate": "requires_decoded_output_parity_report_for_every_changed_stream_before_eval",
                    "planning_only": True,
                    "score_claim": False,
                    "selected_atom_ids": [],
                }
            ]
        },
    }
    path.write_text(module._json_text(payload), encoding="utf-8")
    return path


def _best(raw: bytes) -> bytes:
    return module._brotli_best(raw)[0]


def _choice(prefix: bytes, value: int) -> bytes:
    return prefix + bytes([value]) * module.PAIR_COUNT


def _sparse_empty(prefix: bytes) -> bytes:
    return prefix + (0).to_bytes(2, "little")


def _randmulti_zero_payload() -> bytes:
    return b"\x00" * sum(spec[3] for spec in module.PR85_HEADERLESS_RANDMULTI_SPECS)


def _fixture_archive(path: Path, *, frac2_bad_compression: bool = True) -> None:
    raw_frac2 = _choice(b"FH2", 4)
    frac2 = (
        brotli.compress(raw_frac2, quality=0, lgwin=10)
        if frac2_bad_compression
        else _best(raw_frac2)
    )
    if frac2_bad_compression:
        assert len(_best(raw_frac2)) < len(frac2)
    segments = {
        "mask": b"QMA9" + b"M" * 1000,
        "model": b"QH0" + b"W" * 1000,
        "pose": b"P1D1" + b"P" * 120,
        "post": _best(bytes([1]) * module.PAIR_COUNT * 4),
        "shift": _best(_choice(b"SD4", 0)),
        "frac": _best(_sparse_empty(b"FV1")),
        "frac2": frac2,
        "frac3": _best(_choice(b"FD3", 0)),
        "bias": _best(_sparse_empty(b"BV1")),
        "region": _best(_sparse_empty(b"RV1")),
        "randmulti": _best(_randmulti_zero_payload()),
    }
    payload = module.pack_pr85_bundle(segments, header_mode="explicit_30")
    _zip(path, payload)


def test_builds_only_byte_winning_decoded_parity_archive(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    _fixture_archive(source, frac2_bad_compression=True)

    summary = module.build_candidates(
        source,
        out_dir,
        plan_json=_plan_json(tmp_path / "plan.json"),
        replay_runtime_dir=_runtime_dir(tmp_path / "runtime"),
        max_archive_candidates=4,
        require_known_source=False,
    )

    assert summary["score_claim"] is False
    assert summary["dispatch_performed"] is False
    assert summary["exact_eval_unlocked"] is True
    assert summary["archive_candidate_count"] >= 1
    best = summary["best_candidate"]
    assert best["byte_delta_vs_source_archive"] < 0
    manifest = summary["candidates"][0]
    assert manifest["policy_id"] == "segment_frac2_best"
    assert manifest["changed_segments"] == ["frac2"]
    assert manifest["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
    assert manifest["decoded_parity_metadata"]["status"] == "passed"
    assert all(row["status"] == "passed" for row in manifest["decoded_parity_metadata"]["rows"])
    assert (out_dir / "rank001_segment_frac2_best" / "archive.zip").is_file()
    assert (out_dir / "rank001_segment_frac2_best" / "manifest.json").is_file()


def test_semantic_mismatch_fails_closed() -> None:
    source = _best(_choice(b"FH2", 4))
    bad_candidate = _best(_choice(b"FH2", 5))

    with pytest.raises(module.CorrectionRecodeError, match="failed decoded-semantics parity"):
        module._assert_segment_semantic_parity("frac2", source, bad_candidate, variant="bad_test")


def test_no_byte_win_emits_negative_json_without_archive(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    out_dir = tmp_path / "out"
    _fixture_archive(source, frac2_bad_compression=False)

    summary = module.build_candidates(
        source,
        out_dir,
        plan_json=_plan_json(tmp_path / "plan.json"),
        replay_runtime_dir=_runtime_dir(tmp_path / "runtime"),
        require_known_source=False,
    )

    assert summary["archive_candidate_count"] == 0
    assert summary["exact_eval_unlocked"] is False
    assert summary["result_class"] == "exact_local_negative_no_byte_winning_recode"
    assert (out_dir / "candidate_summary.json").is_file()
    assert not list(out_dir.rglob("archive.zip"))


def test_real_pr85_archive_smoke_emits_candidate_or_exact_negative(tmp_path: Path) -> None:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    runtime = REPO / "experiments/results/public_pr85_intake_20260503_codex/replay_submission"
    if not archive.is_file() or not runtime.is_dir():
        pytest.skip("public PR85 intake artifact is not present")

    summary = module.build_candidates(
        archive,
        tmp_path / "real-out",
        plan_json=tmp_path / "rebuilt_plan.json",
        replay_runtime_dir=runtime,
        max_archive_candidates=4,
        require_known_source=True,
    )

    assert summary["source_archive"]["sha256"] == module.SOURCE_PR85_SHA256
    assert summary["score_claim"] is False
    assert (tmp_path / "real-out" / "candidate_summary.json").is_file()
    if summary["archive_candidate_count"]:
        assert summary["best_candidate"]["byte_delta_vs_source_archive"] < 0
        assert summary["exact_eval_unlocked"] is True
    else:
        assert summary["result_class"] == "exact_local_negative_no_byte_winning_recode"
        assert summary["exact_eval_unlocked"] is False
        assert not list((tmp_path / "real-out").rglob("archive.zip"))
