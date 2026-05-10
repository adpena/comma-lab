from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.hnerv_lowlevel_packer import write_stored_single_member_zip
from tac.pr103_lc_ac_runtime_adapter import (
    ADAPTER_SCHEMA,
    PACKET_SCHEMA,
    Pr103RuntimeAdapterError,
    build_pr103_lc_ac_candidate_packet,
    build_pr103_lc_ac_runtime_adapter,
)
from tac.repo_io import sha256_file, write_json


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools/build_pr103_lc_ac_runtime_adapter.py"


def test_pr103_runtime_adapter_patches_constants_and_proves_consumption(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    report = build_pr103_lc_ac_runtime_adapter(
        candidate_manifest=fixture["manifest"],
        source_runtime_dir=fixture["runtime"],
        output_runtime_dir=tmp_path / "adapted",
        repo_root=tmp_path,
    )

    assert report["schema"] == ADAPTER_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    changed = {row["name"]: row for row in report["constant_changes"]}
    assert changed["HIST_LEN"]["old"] == 4
    assert changed["HIST_LEN"]["new"] == 2
    assert report["runtime_consumption_probe"]["passed"] is True
    assert report["runtime_consumption_probe"]["parsed_lengths"]["HIST_LEN"] == 2
    assert report["runtime_consumption_probe"]["state_dict_tensors"] == 1
    assert report["runtime_consumption_probe"]["latents_shape"] == [2, 3]
    assert report["decoder_state_parity_proof"]["passed"] is True
    assert report["decoder_state_parity_proof"]["full_frame_output_parity_inferred"] is True
    assert "strict_pre_submission_compliance_json_missing" in report["readiness_blockers"]
    assert "full_frame_inflate_output_parity_missing" not in report["readiness_blockers"]
    assert len(report["runtime_tree_sha256"]) == 64
    adapted_inflate = (tmp_path / "adapted/inflate.py").read_text(encoding="utf-8")
    adapted_shell = (tmp_path / "adapted/inflate.sh").read_text(encoding="utf-8")
    assert "HIST_LEN = 2" in adapted_inflate
    assert '"${PYTHON:-python}" "$HERE/inflate.py" "$SRC" "$DST"' in adapted_shell
    assert report["shell_patch"]["changed"] is True


def test_pr103_runtime_adapter_rejects_candidate_archive_custody_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(Path(fixture["manifest"]).read_text(encoding="utf-8"))
    manifest["candidate_archive"]["sha256"] = "0" * 64
    bad_manifest = tmp_path / "bad_manifest.json"
    write_json(bad_manifest, manifest)

    with pytest.raises(Pr103RuntimeAdapterError, match="sha256 mismatch"):
        build_pr103_lc_ac_runtime_adapter(
            candidate_manifest=bad_manifest,
            source_runtime_dir=fixture["runtime"],
            output_runtime_dir=tmp_path / "adapted",
            repo_root=tmp_path,
        )


def test_build_pr103_runtime_adapter_cli_writes_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output_runtime = tmp_path / "adapted"
    json_out = tmp_path / "adapter.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate-manifest",
            str(fixture["manifest"]),
            "--source-runtime-dir",
            str(fixture["runtime"]),
            "--output-runtime-dir",
            str(output_runtime),
            "--json-out",
            str(json_out),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["schema"] == ADAPTER_SCHEMA
    assert report["tool_run_manifest"]["ready_for_exact_eval_dispatch"] is False
    assert output_runtime.joinpath("inflate.py").is_file()
    assert report["runtime_consumption_probe"]["passed"] is True


def test_pr103_candidate_packet_copies_archive_runtime_and_custody(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    adapter = build_pr103_lc_ac_runtime_adapter(
        candidate_manifest=fixture["manifest"],
        source_runtime_dir=fixture["runtime"],
        output_runtime_dir=tmp_path / "adapted",
        repo_root=tmp_path,
    )
    adapter_manifest = tmp_path / "adapter_manifest.json"
    write_json(adapter_manifest, adapter)

    packet = build_pr103_lc_ac_candidate_packet(
        runtime_adapter_manifest=adapter_manifest,
        packet_dir=tmp_path / "packet",
        repo_root=tmp_path,
    )

    packet_dir = tmp_path / "packet"
    assert packet["schema"] == PACKET_SCHEMA
    assert packet["score_claim"] is False
    assert packet["ready_for_exact_eval_dispatch"] is False
    assert packet_dir.joinpath("archive.zip").read_bytes() == fixture["archive"].read_bytes()
    assert packet_dir.joinpath("inflate.py").is_file()
    assert packet_dir.joinpath("inflate.sh").is_file()
    assert packet_dir.joinpath("archive_manifest.json").is_file()
    assert packet_dir.joinpath("report.txt").is_file()
    report_text = packet_dir.joinpath("report.txt").read_text(encoding="utf-8")
    assert packet["archive"]["sha256"] in report_text
    assert str(packet["archive"]["bytes"]) in report_text
    assert "lane_dispatch_claim_missing" in packet["readiness_blockers"]


def _fixture(tmp_path: Path) -> dict[str, Path]:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime.joinpath("inflate.py").write_text(
        """
from __future__ import annotations

import numpy as np

SCA_LEN = 2
BR_LEN = 3
HIST_LEN = 4
MERGED_AC_LEN = 5
LATENT_META_LEN = 1
LO_LEN = 6
HI_HIST_LEN = 2


class brotli:
    @staticmethod
    def decompress(data):
        return data


class _Tensor:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


def parse_archive(blob):
    o = 0
    sca = blob[o:o + SCA_LEN]; o += SCA_LEN
    br = blob[o:o + BR_LEN]; o += BR_LEN
    hists_b = blob[o:o + HIST_LEN]; o += HIST_LEN
    merged_ac = blob[o:o + MERGED_AC_LEN]; o += MERGED_AC_LEN
    mins_scales = blob[o:o + LATENT_META_LEN]; o += LATENT_META_LEN
    lo_b = blob[o:o + LO_LEN]; o += LO_LEN
    hi_hist_b = blob[o:o + HI_HIST_LEN]; o += HI_HIST_LEN
    wrp_b = blob[o:]
    return sca, br, hists_b, merged_ac, mins_scales, lo_b, hi_hist_b, wrp_b


def build_state_dict(br_b, hists_b, merged_ac, sca, hi_hist):
    return {"w": np.zeros(7, dtype=np.float32)}, np.asarray([0, 1], dtype=np.uint16)


def decode_latents(mins_scales, lo_b, hi_decoded):
    return np.zeros((2, 3), dtype=np.float32)


def apply_corrections(latents, wrp_b):
    return latents
""".lstrip(),
        encoding="utf-8",
    )
    runtime.joinpath("inflate.sh").write_text(
        '#!/usr/bin/env bash\npython "$HERE/inflate.py" "$SRC" "$DST"\n',
        encoding="utf-8",
    )
    runtime.joinpath("ignored.pyc").write_bytes(b"skip")

    source_payload = b"aa" + b"bbb" + b"hhhh" + b"mmmmm" + b"z" + b"llllll" + b"\x01\x00" + b"t"
    source_archive = tmp_path / "source.zip"
    write_stored_single_member_zip(source_archive, member_name="x", payload=source_payload)
    payload = b"aa" + b"bbb" + b"hh" + b"mmmmm" + b"z" + b"llllll" + b"\x01\x00" + b"t"
    archive = tmp_path / "candidate.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    manifest = tmp_path / "candidate_manifest.json"
    write_json(
        manifest,
        {
            "schema": "pr103_arithmetic_histogram_candidate_v1",
            "planning_only": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "candidate_archive": {
                "path": "candidate.zip",
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            },
            "source_archive": {
                "path": "source.zip",
                "bytes": source_archive.stat().st_size,
                "sha256": sha256_file(source_archive),
            },
            "runtime_adapter_contract": {
                "source_runtime_constants": {
                    "BR_LEN": 3,
                    "HIST_LEN": 4,
                    "MERGED_AC_LEN": 5,
                    "LO_LEN": 6,
                    "HI_HIST_LEN": 2,
                },
                "public_runtime_constants": {
                    "BR_LEN": 3,
                    "HIST_LEN": 2,
                    "MERGED_AC_LEN": 5,
                    "LO_LEN": 6,
                    "HI_HIST_LEN": 2,
                }
            },
            "semantic_stream_parity": {
                "all_stream_symbol_sha_match": True,
                "stream_count": 1,
            },
        },
    )
    return {"runtime": runtime, "manifest": manifest, "archive": archive, "source_archive": source_archive}
