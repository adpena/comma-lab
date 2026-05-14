# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_archive_bit_budget.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_pr85_archive_bit_budget_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _write_archive(path: Path) -> dict[str, bytes]:
    segments = {
        "mask": b"QMA9" + b"m" * 996,
        "model": b"QH0" + b"a" * 397,
        "pose": b"P1D1" + b"p" * 46,
        "post": b"post" * 25,
        "shift": b"shift" * 3,
        "frac": b"frac" * 2,
        "frac2": b"frac2" * 2,
        "frac3": b"frac3" * 2,
        "bias": b"B" * module.FIXED_V5_LENGTHS["bias"],
        "region": b"R" * module.FIXED_V5_LENGTHS["region"],
        "randmulti": b"randmulti" * 20,
    }
    header = b"".join(_u24(len(segments[name])) for name in module.SEGMENT_ORDER[:8])
    payload = header + b"".join(segments[name] for name in module.SEGMENT_ORDER)
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)
    return segments


def _write_profile(path: Path, archive: Path, segments: dict[str, bytes]) -> None:
    path.write_text(
        json.dumps(
            {
                "archive": {
                    "archive_size_bytes": archive.stat().st_size,
                    "archive_sha256": module._sha256_file(archive),  # noqa: SLF001
                    "member_name": "x",
                },
                "segments": [
                    {
                        "name": name,
                        "bytes": len(segments[name]),
                        "sha256": module._sha256_bytes(segments[name]),  # noqa: SLF001
                    }
                    for name in module.SEGMENT_ORDER
                ],
                "score_claim": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_profile_accounts_member_segments_bits_rate_and_flags(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    profile_json = tmp_path / "profile.json"
    ablations = tmp_path / "ablations.json"
    post_motion = tmp_path / "post_motion.json"
    recodes = tmp_path / "recodes.json"
    segments = _write_archive(archive)
    _write_profile(profile_json, archive, segments)
    ablations.write_text(
        json.dumps(
            {
                "schema": "pr85_sidechannel_ablation_candidates_v1",
                "score_claim": False,
                "dispatch_performed": False,
                "candidates": [
                    {
                        "policy_id": "minus_post",
                        "neutralized_segments": ["post"],
                        "byte_delta_vs_source_archive": -90,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    post_motion.write_text(
        json.dumps(
            {
                "schema": "pr85_post_motion_group_policy_candidates_v1",
                "score_claim": False,
                "dispatch_performed": False,
                "candidates": [
                    {
                        "policy_id": "preserve_motion_only",
                        "changed_segments": ["post"],
                        "whole_stream_negative_context": {
                            "minus_post": {
                                "role": "exact T4 negative; post stream carries high-value signal",
                                "score": 0.31,
                                "evidence_grade": "A++ exact T4 negative",
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    recodes.write_text(
        json.dumps(
            {
                "schema": "pr85_sidechannel_recode_candidates_v1",
                "score_claim": False,
                "dispatch_performed": False,
                "candidates": [
                    {
                        "schema": "pr85_sidechannel_recode_candidate_v1",
                        "policy_id": "segment_post_best",
                        "changed_segments": ["post"],
                        "byte_delta_vs_source_archive": 7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_profile(
        archive,
        profile_json=profile_json,
        sidechannel_summaries=[ablations, post_motion, recodes],
        top_k=4,
    )

    assert payload["schema"] == module.SCHEMA
    assert payload["planning_only"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert payload["member_accounting"]["matches_member_file_size"] is True
    assert payload["member_accounting"]["profile_archive_mismatches"] == []
    rows = {row["name"]: row for row in payload["segments"]}
    assert rows["mask"]["bytes"] == len(segments["mask"])
    assert rows["mask"]["bits"] == len(segments["mask"]) * 8
    assert rows["mask"]["rank_by_bytes"] == 1
    assert rows["post"]["known_flags"]["deletion_screened"] is True
    assert rows["post"]["known_flags"]["exact_deletion_negative"] is True
    assert rows["post"]["known_flags"]["protected"] is True
    assert rows["post"]["known_flags"]["best_lossless_recode_delta_bytes"] == 7
    assert payload["opportunity_rankings"][0]["segment"] == "mask"
    assert not str(payload["inputs"]["archive"]).startswith("/")


def test_cli_writes_deterministic_json_and_optional_markdown(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "archive.zip"
    profile_json = tmp_path / "profile.json"
    summary = tmp_path / "summary.json"
    json_out = tmp_path / "budget.json"
    markdown_out = tmp_path / "budget.md"
    segments = _write_archive(archive)
    _write_profile(profile_json, archive, segments)
    summary.write_text('{"schema":"empty","score_claim":false,"candidates":[]}\n', encoding="utf-8")

    argv = [
        "--archive",
        str(archive),
        "--profile-json",
        str(profile_json),
        "--sidechannel-summary",
        str(summary),
        "--json-out",
        str(json_out),
        "--markdown-out",
        str(markdown_out),
    ]
    assert module.main(argv) == 0
    first = json_out.read_text(encoding="utf-8")
    stdout = capsys.readouterr().out
    assert stdout == first
    assert module.main(argv) == 0
    second = json_out.read_text(encoding="utf-8")
    capsys.readouterr()

    assert first == second
    assert json.loads(second)["deterministic"] is True
    assert "PR85 Archive Bit-Budget Profile" in markdown_out.read_text(encoding="utf-8")
