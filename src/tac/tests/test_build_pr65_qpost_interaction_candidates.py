# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_pr65_qpost_interaction_candidates.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_pr65_qpost_interaction_candidates_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def test_active_duplicate_guard_detects_c089_qpost_claim() -> None:
    module = _load()
    qpost = module.QPostSpec("bias_top040", ("bias",), 40, "test")
    claims = [
        {
            "lane_id": "lane_qzs3_postprocess_sidecar",
            "instance_job_id": "exact_eval_pr65_qpost_v2_bias_poseadv_top040_t4_20260503T0805Z",
            "status": "eval",
            "notes": "active exact eval over source frontier C089",
        }
    ]

    status = module._active_duplicate_status(
        source_sha256=module.EXPECTED_C089_SHA256,
        qpost=qpost,
        candidate_archive_sha256=None,
        active_claims=claims,
    )

    assert status["duplicates_active_job"] is True
    assert status["matched_claims"][0]["instance_job_id"].endswith("top040_t4_20260503T0805Z")


def test_build_interactions_writes_non_dispatch_manifest(tmp_path: Path) -> None:
    module = _load()
    source = tmp_path / "source_p6.zip"
    _stored_zip(source, {"p": b"P6" + b"x" * 128})
    source_sha = module._sha256_path(source)
    claims = tmp_path / "active_claims.md"
    claims.write_text(
        "# Active lane dispatch claims\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )

    summary = module.build_interactions(
        output_dir=tmp_path / "out",
        active_claims_path=claims,
        action_bases=(
            module.ActionBase(
                "fixture_p6",
                source,
                0.3154707273953505,
                "fixture exact score",
                source_sha,
            ),
        ),
        qpost_specs=(
            module.QPostSpec("bias_top040", ("bias",), 40, "test_bias_only"),
        ),
    )

    assert summary["score_claim"] is False
    assert summary["remote_dispatch"]["dispatched"] is False
    screen = summary["candidate_screens"][0]
    assert screen["dispatchable_later"] is True
    assert screen["active_job_duplication"]["duplicates_active_job"] is False
    assert screen["selected_active_atoms_total"] == 40
    assert screen["qpost_bytes"] > 0
    archive = Path(screen["archive"])
    assert archive.is_file()
    with zipfile.ZipFile(archive, "r") as zf:
        assert sorted(info.filename for info in zf.infolist() if not info.is_dir()) == ["p", "qpost.bin"]

    manifest = json.loads((archive.parent / "interaction_manifest.json").read_text())
    assert manifest["score_claim"] is False
    assert manifest["remote_dispatch"]["dispatched"] is False
    assert manifest["dispatchable_later"] is True
    assert manifest["selected_pair_rank_records"][0]["pair_index"] == screen["selected_pairs"][0]
    assert "contest_auth_eval.py" in manifest["exact_eval_command_template"]
