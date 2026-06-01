# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from tac.substrates.hprc.representation_spine import (
    HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
)
from tac.substrates.hprc.spine_bounded_runner import (
    HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA,
)
from tac.substrates.hprc.spine_receiver_execution import (
    HPRC_SPINE_RECEIVER_EXECUTION_REPORT_SCHEMA,
    HPRC_SPINE_RECEIVER_PROOF_SCHEMA,
    SpineReceiverRuntimeOverride,
    execute_spine_receiver_rows,
)


def test_execute_spine_receiver_rows_runs_runtime_and_cleans_success(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"payload")
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(
            {
                "schema": HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
                "family": "pr95_hnerv",
                "manifest": {
                    "representation_spine": {
                        "source": {
                            "path": archive.as_posix(),
                            "bytes": archive.stat().st_size,
                            "sha256": _sha256_file(archive),
                            "member_name": "0.bin",
                            "member_bytes": len(b"payload"),
                        },
                        "manifest_extra": {"num_pairs": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "runner.json"
    plan.write_text(
        json.dumps(
            {
                "schema": HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA,
                "selected_runner_rows": [
                    {
                        "runner_row_id": "pr95_hnerv:178000",
                        "family": "pr95_hnerv",
                        "projection_manifest_path": projection.as_posix(),
                    },
                    {
                        "runner_row_id": "pr95_hnerv:216000",
                        "family": "pr95_hnerv",
                        "projection_manifest_path": projection.as_posix(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    inflate = tmp_path / "runtime" / "inflate.sh"
    inflate.parent.mkdir()
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "data_dir=\"$1\"; out_dir=\"$2\"; file_list=\"$3\"\n"
        "mkdir -p \"$out_dir\"\n"
        "while IFS= read -r line; do\n"
        "  base=\"${line%.*}\"\n"
        "  test -f \"$data_dir/${base}.bin\"\n"
        "  head -c 12 /dev/zero > \"$out_dir/${base}.raw\"\n"
        "done < \"$file_list\"\n",
        encoding="utf-8",
    )
    os.chmod(inflate, 0o755)

    report = execute_spine_receiver_rows(
        runner_plan_path=plan,
        output_dir=tmp_path / "proofs",
        repo_root=tmp_path,
        runtime_overrides=[
            SpineReceiverRuntimeOverride(family="pr95_hnerv", inflate_sh=inflate)
        ],
        expected_raw_bytes_overrides={"pr95_hnerv": 12},
        max_output_bytes=12,
    )

    assert report["schema"] == HPRC_SPINE_RECEIVER_EXECUTION_REPORT_SCHEMA
    assert report["deduped_execution_row_count"] == 1
    assert report["receiver_proof_passed_count"] == 1
    row = report["receiver_rows"][0]
    assert row["receiver_contract_satisfied"] is True
    proof = json.loads(Path(row["proof_path"]).read_text(encoding="utf-8"))
    assert proof["schema"] == HPRC_SPINE_RECEIVER_PROOF_SCHEMA
    assert proof["runtime_consumption_proof_ready"] is True
    assert proof["cleanup"]["work_dir_removed_after_hashing"] is True
    assert not (Path(row["proof_path"]).parent / "runtime_consumption_work").exists()
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_execute_spine_receiver_rows_blocks_large_output_without_explicit_allow(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"payload")
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(
            {
                "schema": HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
                "family": "pr95_hnerv",
                "manifest": {
                    "representation_spine": {
                        "source": {
                            "path": archive.as_posix(),
                            "sha256": _sha256_file(archive),
                            "member_name": "0.bin",
                            "member_bytes": len(b"payload"),
                        },
                        "manifest_extra": {"num_pairs": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "runner.json"
    plan.write_text(
        json.dumps(
            {
                "schema": HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA,
                "selected_runner_rows": [
                    {
                        "runner_row_id": "pr95_hnerv:178000",
                        "family": "pr95_hnerv",
                        "projection_manifest_path": projection.as_posix(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = execute_spine_receiver_rows(
        runner_plan_path=plan,
        output_dir=tmp_path / "proofs",
        repo_root=tmp_path,
        expected_raw_bytes_overrides={"pr95_hnerv": 128},
        max_output_bytes=12,
    )

    row = report["receiver_rows"][0]
    assert row["receiver_contract_satisfied"] is False
    assert "predicted_raw_output_exceeds_guardrail" in row["blockers"]


def test_execute_spine_receiver_rows_proves_archive_embedded_png_runtime(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"payload")
        zf.writestr(
            "inflate.sh",
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "data_dir=\"$1\"; out_dir=\"$2\"; file_list=\"$3\"\n"
            "test -f \"$data_dir/0.bin\"\n"
            "while IFS= read -r line; do\n"
            "  base=\"${line%.*}\"\n"
            "  mkdir -p \"$out_dir/$base\"\n"
            "  printf png > \"$out_dir/$base/frame_000000.png\"\n"
            "  printf png > \"$out_dir/$base/frame_000001.png\"\n"
            "done < \"$file_list\"\n",
        )
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(
            {
                "schema": HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
                "family": "pact_nerv",
                "manifest": {
                    "representation_spine": {
                        "source": {
                            "path": archive.as_posix(),
                            "sha256": _sha256_file(archive),
                            "member_name": "0.bin",
                            "member_bytes": len(b"payload"),
                        },
                        "manifest_extra": {"num_pairs": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    second_archive = tmp_path / "archive_two.zip"
    with zipfile.ZipFile(second_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", b"payload-two")
        zf.writestr(
            "inflate.sh",
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "data_dir=\"$1\"; out_dir=\"$2\"; file_list=\"$3\"\n"
            "test -f \"$data_dir/0.bin\"\n"
            "while IFS= read -r line; do\n"
            "  base=\"${line%.*}\"\n"
            "  mkdir -p \"$out_dir/$base\"\n"
            "  printf png2 > \"$out_dir/$base/frame_000000.png\"\n"
            "  printf png2 > \"$out_dir/$base/frame_000001.png\"\n"
            "done < \"$file_list\"\n",
        )
    second_projection = tmp_path / "projection_two.json"
    second_projection.write_text(
        json.dumps(
            {
                "schema": HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
                "family": "pact_nerv",
                "manifest": {
                    "representation_spine": {
                        "source": {
                            "path": second_archive.as_posix(),
                            "sha256": _sha256_file(second_archive),
                            "member_name": "0.bin",
                            "member_bytes": len(b"payload-two"),
                        },
                        "manifest_extra": {"num_pairs": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "runner.json"
    plan.write_text(
        json.dumps(
            {
                "schema": HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA,
                "selected_runner_rows": [
                    {
                        "runner_row_id": "pact_nerv:178000",
                        "family": "pact_nerv",
                        "projection_manifest_path": projection.as_posix(),
                    },
                    {
                        "runner_row_id": "pact_nerv:178000",
                        "family": "pact_nerv",
                        "projection_manifest_path": second_projection.as_posix(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = execute_spine_receiver_rows(
        runner_plan_path=plan,
        output_dir=tmp_path / "proofs",
        repo_root=tmp_path,
        max_output_bytes=12,
        allow_large_output=True,
    )

    row = report["receiver_rows"][0]
    assert report["deduped_execution_row_count"] == 2
    assert len({item["proof_path"] for item in report["receiver_rows"]}) == 2
    assert row["receiver_contract_satisfied"] is True
    assert row["receiver_output_kind"] == "png_tree"
    assert row["receiver_output_frame_count"] == 2


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
