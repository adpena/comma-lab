from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_trainer_module():
    path = Path("experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py")
    spec = importlib.util.spec_from_file_location("phase1_t1_trainer_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase1_trainer_auth_eval_no_longer_refuses_before_training() -> None:
    module = _load_trainer_module()
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "research_only_no_export" not in source
    assert "--auth-eval refused before training" not in source


def test_phase1_trainer_auth_eval_reads_declared_json_out(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_trainer_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "require_active_dispatch_claim", lambda **_: None)

    auth_script = tmp_path / module.CONTEST_AUTH_EVAL_RELATIVE
    auth_script.parent.mkdir(parents=True, exist_ok=True)
    auth_script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--json-out') + 1])\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text(json.dumps({'ok': True}))\n",
        encoding="utf-8",
    )
    (tmp_path / "upstream").mkdir()

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")

    result = module.maybe_run_auth_eval(
        archive_path=archive,
        submission_dir=submission_dir,
        output_dir=output_dir,
        enabled=True,
        dispatch_lane_id="lane_t1_phase1",  # FAKE_LANE_OK: auth-eval fixture
        dispatch_claims_path=tmp_path / "claims.md",
    )

    expected = output_dir / "contest_auth_eval.json"
    assert result == {"returncode": 0, "auth_json_path": str(expected)}
    assert expected.read_text(encoding="utf-8") == '{"ok": true}'
