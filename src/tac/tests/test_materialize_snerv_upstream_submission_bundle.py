# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools import materialize_snerv_upstream_submission_bundle as cli


def test_materialize_snerv_upstream_submission_bundle_externalizes_runtime(
    tmp_path: Path,
) -> None:
    source = _write_fake_source_submission(tmp_path / "source_submission")
    source_archive = tmp_path / "source_archive.zip"
    _write_zip(
        source_archive,
        {
            "0.bin": b"payload",
            "inflate.py": (source / "inflate.py").read_bytes(),
        },
    )
    out = tmp_path / "out_submission"

    report = cli.materialize_upstream_submission_bundle(
        source_submission_dir=source,
        output_submission_dir=out,
        source_archive_zip=source_archive,
        run_receiver_proof=False,
        generated_utc="2026-06-04T00:00:00+00:00",
    )

    assert report["schema"] == cli.SCHEMA
    assert report["upstream_contest_contract"][
        "rate_uses_submission_archive_zip_stat_only"
    ] is True
    assert (out / "inflate.sh").is_file()
    assert (out / "inflate.py").is_file()
    assert not (out / "0.bin").exists()
    with zipfile.ZipFile(out / "archive.zip", "r") as zf:
        assert zf.namelist() == ["x"]
        assert zf.read("x") == b"payload"
    assert report["archive_zip"]["member_name"] == "x"
    assert report["archive_zip"]["data_only"] is True
    assert report["archive_zip"]["candidate_count"] >= 2
    assert report["archive_zip"]["bytes"] < report["source_archive_zip"]["bytes"]
    assert report["external_runtime"]["contains_unarchived_payload_packet"] is False
    assert (
        report["external_runtime"]["runtime_source_rate_charged_by_upstream_evaluate_py"]
        is False
    )
    assert report["receiver_proof"]["runtime_consumption_proof_passed"] is False
    assert "snerv_upstream_submission_receiver_proof_not_requested" in report[
        "blockers"
    ]
    assert report["ready_for_exact_eval_dispatch"] is False


def test_materialize_snerv_upstream_submission_bundle_runs_receiver_proof(
    tmp_path: Path,
) -> None:
    source = _write_fake_source_submission(tmp_path / "source_submission")
    out = tmp_path / "out_submission"
    output_json = tmp_path / "report.json"

    assert cli.main(
        [
            "--source-submission-dir",
            str(source),
            "--output-submission-dir",
            str(out),
            "--output-json",
            str(output_json),
            "--run-receiver-proof",
            "--expected-receiver-output-bytes",
            "8",
        ]
    ) == 0

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["receiver_proof"]["runtime_consumption_proof_passed"] is True
    assert report["receiver_proof"]["receiver_contract_satisfied"] is True
    assert report["receiver_proof"]["receiver_output_bytes"] == 8
    assert report["receiver_proof"]["receiver_output_retained"] is False
    assert not (out / "receiver_proof" / "runtime_out").exists()
    assert "paired_contest_cpu_cuda_auth_eval_missing" in report["blockers"]
    assert "pre_submission_compliance_gate_missing" in report["blockers"]


def _write_fake_source_submission(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "0.bin").write_bytes(b"payload")
    inflate_sh = root / "inflate.sh"
    inflate_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'exec "${PYTHON:-python3}" "$HERE/inflate.py" "$1" "$2" "$3"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)
    (root / "inflate.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "archive_dir = Path(sys.argv[1])\n"
        "output_dir = Path(sys.argv[2])\n"
        "file_list = Path(sys.argv[3])\n"
        "packet = archive_dir / 'x'\n"
        "if not packet.is_file():\n"
        "    packet = archive_dir / '0.bin'\n"
        "for line in file_list.read_text().splitlines():\n"
        "    if line.strip():\n"
        "        (output_dir / Path(line).with_suffix('.raw')).write_bytes(packet.read_bytes() + b'!')\n",
        encoding="utf-8",
    )
    (root / "src").mkdir()
    (root / "src" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, payload in members.items():
            info = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload)
