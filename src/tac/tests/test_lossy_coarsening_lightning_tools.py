from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lossy_coarsening_generated_inflate_uses_model_schema_once() -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_cuda_test.py")

    inflate_source = module.FORKED_INFLATE_PY

    assert "def _fixed_state_schema()" in inflate_source
    assert "probe.state_dict().items()" in inflate_source
    assert '("blocks.0.weight", (36, 28, 1, 1))' not in inflate_source
    assert "reconstructed_q = chunk" in inflate_source
    assert "chunk * Ks" not in inflate_source


def test_lossy_coarsening_remote_command_uses_auth_eval_json_path() -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_cuda_test.py")

    command = module.build_remote_command(
        job_name="job-test",
        remote_pact="/remote/pact",
        archive_relpath="experiments/results/lossy/archive.zip",
        submission_dir_relpath="experiments/results/lossy/submission_dir",
    )

    assert "experiments/contest_auth_eval.py" in command
    assert "--archive \"$ARCHIVE\"" in command
    assert "--inflate-sh \"$INFLATE_SH\"" in command
    assert "--device cuda" in command
    assert "--keep-work-dir" in command
    assert "contest_auth_eval.json" in command


def test_lossy_coarsening_harvest_requires_numeric_score_and_archive_bytes(tmp_path: Path) -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_harvest.py")
    evidence_out = tmp_path / "evidence.jsonl"
    target_row = {
        "archive_bytes": 156404,
        "rel_err_budget": 0.05,
        "rel_err_actual_int8": 0.03856,
        "rel_err_actual_fp32_smoke": 0.034811,
        "predicted_band": [0.18, 0.22],
    }

    row = module._emit_evidence_row(
        evidence_out=evidence_out,
        auth_eval={"score": 0.189, "archive_bytes": 156404, "rate": 0.004},
        target_row=target_row,
        job_name="job-test",
    )

    assert row["evidence_grade"] == "[contest-CUDA]"
    assert row["score_claim"] is False
    assert row["archive_bytes_match_expected"] is True
    assert json.loads(evidence_out.read_text(encoding="utf-8"))["score_contest_cuda"] == 0.189

    with pytest.raises(SystemExit, match="without numeric auth_eval score"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval={"archive_bytes": 156404},
            target_row=target_row,
            job_name="job-test",
        )
    with pytest.raises(SystemExit, match="archive_bytes 1 != expected 156404"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval={"score": 0.189, "archive_bytes": 1},
            target_row=target_row,
            job_name="job-test",
        )


@pytest.mark.parametrize(
    ("script_path", "lane_id", "job_name"),
    [
        (
            "experiments/lossy_coarsening_lightning_harvest.py",
            "lossy_coarsening_analytical_cuda",
            "lossy-test",
        ),
        (
            "experiments/arch_shrink_x0.4_lightning_harvest.py",
            "arch_shrink_x0.4_lightning",
            "arch-test",
        ),
    ],
)
def test_lightning_harvest_requires_provider_env_before_sdk(
    monkeypatch: pytest.MonkeyPatch,
    script_path: str,
    lane_id: str,
    job_name: str,
) -> None:
    module = _load_script(script_path)

    monkeypatch.setattr(
        module,
        "_load_active_jobs",
        lambda: [{"lane_id": lane_id, "job_name": job_name, "terminal_status": None}],
    )

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError("Lightning SDK resolution should not be reached")

    monkeypatch.setattr(module, "_resolve_lightning_job", fail_if_called)

    with pytest.raises(SystemExit, match="missing required Lightning provider values"):
        module.main(["--job-name", job_name, "--once"])


@pytest.mark.parametrize(
    ("script_path", "lane_id", "job_name"),
    [
        (
            "experiments/lossy_coarsening_lightning_harvest.py",
            "lossy_coarsening_analytical_cuda",
            "lossy-test",
        ),
        (
            "experiments/arch_shrink_x0.4_lightning_harvest.py",
            "arch_shrink_x0.4_lightning",
            "arch-test",
        ),
    ],
)
def test_lightning_force_harvest_requires_rsync_env_before_artifact_pull(
    monkeypatch: pytest.MonkeyPatch,
    script_path: str,
    lane_id: str,
    job_name: str,
) -> None:
    module = _load_script(script_path)

    monkeypatch.setattr(
        module,
        "_load_active_jobs",
        lambda: [{"lane_id": lane_id, "job_name": job_name, "terminal_status": None}],
    )

    def fail_if_called(**_kwargs: object) -> Path:
        raise AssertionError("rsync should not be reached")

    monkeypatch.setattr(module, "_rsync_artifacts", fail_if_called)

    with pytest.raises(SystemExit, match="missing required Lightning artifact-rsync values"):
        module.main(["--job-name", job_name, "--force-harvest", "--remote-pact", ""])


@pytest.mark.parametrize(
    ("script_path", "lane_id", "job_name"),
    [
        (
            "experiments/lossy_coarsening_lightning_harvest.py",
            "lossy_coarsening_analytical_cuda",
            "lossy-test",
        ),
        (
            "experiments/arch_shrink_x0.4_lightning_harvest.py",
            "arch_shrink_x0.4_lightning",
            "arch-test",
        ),
    ],
)
def test_lightning_terminal_harvest_closes_claim_on_rsync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_path: str,
    lane_id: str,
    job_name: str,
) -> None:
    module = _load_script(script_path)
    active_rows = [{"lane_id": lane_id, "job_name": job_name, "terminal_status": None}]
    saved_rows: list[list[dict[str, object]]] = []
    terminal_claims: list[dict[str, object]] = []

    def fail_rsync(**_kwargs: object) -> Path:
        raise module.LightningHarvestRsyncError(
            returncode=23,
            remote_path="studio:/missing/artifacts/",
        )

    monkeypatch.setattr(module, "_rsync_artifacts", fail_rsync)
    monkeypatch.setattr(module, "_load_active_jobs", lambda: [dict(active_rows[0])])
    monkeypatch.setattr(module, "_save_active_jobs", lambda rows: saved_rows.append(rows))
    monkeypatch.setattr(
        module,
        "_terminal_claim",
        lambda **kwargs: terminal_claims.append(kwargs),
    )

    args = SimpleNamespace(
        ssh_target="studio",
        remote_pact="/remote/pact",
        evidence_out=tmp_path / "evidence.jsonl",
    )

    with pytest.raises(SystemExit, match="artifact rsync failed rc=23"):
        module._harvest_terminal(target={"job_name": job_name}, args=args)

    assert terminal_claims[0]["status"] == "failed_artifact_rsync_rc_23"
    assert "studio:/missing/artifacts/" in terminal_claims[0]["notes"]
    assert saved_rows[0][0]["terminal_status"] == "failed_artifact_rsync_rc_23"
