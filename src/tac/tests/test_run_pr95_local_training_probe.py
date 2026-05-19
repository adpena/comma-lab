# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = REPO_ROOT / "tools" / "run_pr95_local_training_probe.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "run_pr95_local_training_probe",
        HELPER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_layout_points_at_source_faithful_public_pr95_intake() -> None:
    helper = _load_helper()

    layout = helper.resolve_layout()

    assert layout.source_dir.name == "hnerv_muon"
    assert layout.challenge_root.name == "source"
    assert layout.train_py.exists()
    assert layout.compress_sh.exists()
    assert layout.inflate_sh.exists()
    assert (layout.challenge_root / "videos/0.mkv").exists()
    assert layout.public_archive is not None
    assert layout.public_archive.stat().st_size == 178417


def test_plan_is_false_authority_and_records_source_hash(tmp_path: Path) -> None:
    helper = _load_helper()
    layout = helper.resolve_layout()

    plan = helper.build_plan(
        layout=layout,
        output_dir=tmp_path,
        device="mps",
        full_curriculum=False,
        stage_limit=1,
        stage_epoch_overrides={1: 1},
        eval_every=1,
        allow_mps_fallback=False,
        seed=1234,
    )

    assert plan["source_dir"].endswith(
        "public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon"
    )
    assert len(plan["source_tree_sha256"]) == 64
    assert plan["public_archive"]["bytes"] == 178417
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["rank_or_kill_eligible"] is False
    assert "contest_CPU" in plan["authority_contract"]["score_authority"]
    assert plan["stages"] == [
        {
            "index": 1,
            "module": "stage1_v328_ce",
            "epoch_override": 1,
            "eval_every_override": 1,
        }
    ]


def test_stage_epoch_override_parser_is_fail_closed() -> None:
    helper = _load_helper()

    assert helper.parse_stage_epoch_overrides(["1=2", "8=5"]) == {1: 2, 8: 5}
    with pytest.raises(ValueError, match="N=EPOCHS"):
        helper.parse_stage_epoch_overrides(["bad"])
    with pytest.raises(ValueError, match="outside"):
        helper.parse_stage_epoch_overrides(["9=1"])
    with pytest.raises(ValueError, match="positive"):
        helper.parse_stage_epoch_overrides(["1=0"])


def test_stored_single_member_zip_strips_extra_fields(tmp_path: Path) -> None:
    helper = _load_helper()
    payload = tmp_path / "0.bin"
    payload.write_bytes(b"already dense payload")
    archive = tmp_path / "archive.zip"

    meta = helper.write_stored_single_member_zip(payload, archive)

    assert meta["bytes"] == payload.stat().st_size + 108
    assert meta["compression_method"] == "stored"
    assert meta["extra_field_bytes"] == 0
    assert meta["member_sha256"] == helper._sha256_file(payload)
    with zipfile.ZipFile(archive, mode="r") as zf:
        infos = zf.infolist()
    assert [info.filename for info in infos] == ["0.bin"]
    assert infos[0].compress_type == zipfile.ZIP_STORED
    assert infos[0].extra == b""


def test_stored_single_member_zip_requires_canonical_payload_name(tmp_path: Path) -> None:
    helper = _load_helper()
    payload = tmp_path / "not_zero.bin"
    payload.write_bytes(b"x")

    with pytest.raises(ValueError, match=r"0\.bin"):
        helper.write_stored_single_member_zip(payload, tmp_path / "archive.zip")


def test_mps_fallback_policy_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load_helper()
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable on this runner")

    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    assert helper.select_device("mps", allow_mps_fallback=True).type == "mps"
    with pytest.raises(RuntimeError, match="silent CPU fallback"):
        helper.select_device("mps", allow_mps_fallback=False)


def test_plan_only_writes_plan_without_running_training(tmp_path: Path) -> None:
    helper = _load_helper()

    args = helper.parse_args([
        "--plan-only",
        "--output-dir",
        str(tmp_path),
        "--device",
        "cpu",
        "--stage-epochs",
        "1=1",
    ])
    result = helper.run_probe(args)

    assert Path(result["manifest_path"]).name == "plan.json"
    assert (tmp_path / "plan.json").exists()
    assert not (tmp_path / "stage1_stage1_v328_ce").exists()
