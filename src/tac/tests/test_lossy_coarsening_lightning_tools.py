from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTEST_DENOMINATOR = 37_545_489


def _load_script(relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_auth_eval(*, archive_bytes: int, pose: float = 0.0001, seg: float = 0.001):
    score = 100.0 * seg + (10.0 * pose) ** 0.5 + 25.0 * archive_bytes / CONTEST_DENOMINATOR
    return {
        "score_recomputed_from_components": score,
        "canonical_score": score,
        "canonical_score_source": "score_recomputed_from_components",
        "archive_size_bytes": archive_bytes,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "rate_unscaled": archive_bytes / CONTEST_DENOMINATOR,
        "n_samples": 600,
        "provenance": {
            "archive_size_bytes": archive_bytes,
            "archive_sha256": "a" * 64,
            "device": "cuda",
            "cuda_available": True,
            "cuda_device_count": 1,
            "gpu_t4_match": True,
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": "b" * 64,
            },
        },
    }


def test_round3_strict_validator_accepts_auth_eval_contribution_rounding() -> None:
    module = _load_script("src/tac/deploy/lightning/round3_harvest.py")
    auth_eval = {
        "score_recomputed_from_components": 0.351718793322788,
        "canonical_score": 0.351718793322788,
        "canonical_score_source": "score_recomputed_from_components",
        "archive_size_bytes": 156404,
        "avg_posenet_dist": 0.00037762,
        "avg_segnet_dist": 0.00186125,
        "rate_unscaled": 0.00416572,
        "score_pose_contribution": 0.061450793322787946,
        "score_seg_contribution": 0.186125,
        "score_rate_contribution": 0.10414300000000001,
        "n_samples": 600,
        "provenance": {
            "archive_size_bytes": 156404,
            "archive_sha256": "a" * 64,
            "device": "cuda",
            "cuda_available": True,
            "cuda_device_count": 1,
            "gpu_t4_match": True,
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": "b" * 64,
            },
        },
    }

    assert module.contest_cuda_auth_eval_blockers(auth_eval) == []


def test_round3_strict_validator_rejects_score_and_byte_aliases() -> None:
    module = _load_script("src/tac/deploy/lightning/round3_harvest.py")
    auth_eval = _strict_auth_eval(archive_bytes=156404)
    auth_eval["canonical_score_source"] = "canonical_score"
    auth_eval["canonical_score"] = auth_eval.pop("score_recomputed_from_components")
    auth_eval["archive_bytes"] = auth_eval.pop("archive_size_bytes")

    blockers = module.contest_cuda_auth_eval_blockers(auth_eval)

    assert "score_recomputed_from_components_missing_or_nonfinite" in blockers
    assert "canonical_score_source_not_recomputed_from_components" in blockers
    assert "archive_size_bytes_missing_or_invalid" in blockers


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

    assert "scripts/remote_archive_only_eval.sh" in command
    assert 'ARCHIVE_PATH="$ARCHIVE"' in command
    assert 'INFLATE_SH="$INFLATE_SH"' in command
    assert "Stage 0a: GPU presence check" in command
    assert "torch.cuda.is_available()" in command
    assert "KEEP_EVAL_WORK=0" in command
    assert "contest_auth_eval.json" in command
    assert "canonical wrapper" in command
    assert "export INFLATE_BROTLI_SPEC=brotli==1.2.0" in command
    assert "-m', 'pip', 'install'" not in command


def test_lossy_coarsening_generated_inflate_declares_brotli_runtime_dep() -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_cuda_test.py")

    inflate_sh = module.FORKED_INFLATE_SH

    assert '"$UV_BIN" run' in inflate_sh
    assert '"$UV_BIN" run --no-project' in inflate_sh
    assert '--with "$INFLATE_BROTLI_SPEC"' in inflate_sh
    assert '--with "$INFLATE_TORCH_SPEC"' in inflate_sh
    assert "mutating the repo/evaluator environment" in inflate_sh
    assert "import brotli" in module.FORKED_INFLATE_PY


def test_arch_shrink_remote_command_uses_qfaithful_manual_eval_path() -> None:
    module = _load_script("experiments/arch_shrink_x0.4_lightning_full.py")

    command = module.build_remote_command(
        job_name="arch-test",
        remote_pact="/remote/pact",
    )

    assert "--qfaithful-training-poses" in command
    assert "--no-auth-eval-on-best" in command
    assert "save_qfai" in command
    assert "scripts/bootstrap_dali_hash_pinned.py" in command
    assert "contest_auth_eval.json" in command


def test_qfaithful_training_path_passes_pose_through_forward_kwargs() -> None:
    source = (REPO_ROOT / "src/tac/experiments/train_renderer.py").read_text(
        encoding="utf-8"
    )

    assert 'forward_kwargs["ego_flow"] = ego_flow' in source
    assert 'forward_kwargs["pose"] = qfaithful_pose' in source
    assert "rendered_pair = model(mask_t, mask_t1, **forward_kwargs)" in source
    assert "Q-FAITHFUL forward requires an explicit deployed" in source


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
        auth_eval=_strict_auth_eval(archive_bytes=156404),
        target_row=target_row,
        job_name="job-test",
    )

    assert row["evidence_grade"] == "[contest-CUDA]"
    assert row["score_claim"] is False
    assert row["archive_bytes_match_expected"] is True
    assert json.loads(evidence_out.read_text(encoding="utf-8"))["score_contest_cuda"] == row["score_contest_cuda"]

    with pytest.raises(SystemExit, match="without numeric auth_eval score"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval={"archive_size_bytes": 156404},
            target_row=target_row,
            job_name="job-test",
        )
    with pytest.raises(SystemExit, match="archive_bytes 1 != expected 156404"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval=_strict_auth_eval(archive_bytes=1),
            target_row=target_row,
            job_name="job-test",
        )
    loose_auth_eval = {
        "score_recomputed_from_components": 0.189,
        "archive_size_bytes": 156404,
        "rate_unscaled": 0.004,
    }
    with pytest.raises(SystemExit, match="auth_eval custody blockers"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval=loose_auth_eval,
            target_row=target_row,
            job_name="job-test",
        )


def test_arch_shrink_harvest_requires_canonical_score_and_archive_bytes(
    tmp_path: Path,
) -> None:
    module = _load_script("experiments/arch_shrink_x0.4_lightning_harvest.py")
    evidence_out = tmp_path / "evidence.jsonl"
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(b"x" * 333)
    auth_eval = _strict_auth_eval(archive_bytes=333)
    auth_eval["provenance"]["archive_sha256"] = hashlib.sha256(
        archive_path.read_bytes()
    ).hexdigest()

    row = module._emit_evidence_row(
        evidence_out=evidence_out,
        auth_eval=auth_eval,
        job_name="arch-test",
        archive_path=archive_path,
    )

    assert row["score_contest_cuda"] == auth_eval["score_recomputed_from_components"]
    assert row["empirical_archive_bytes"] == 333
    assert row["score_claim"] is False
    with pytest.raises(SystemExit, match="without numeric auth_eval score"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval={"archive_size_bytes": 333},
            job_name="arch-test",
            archive_path=None,
        )
    with pytest.raises(SystemExit, match=r"without local archive\.zip custody"):
        module._emit_evidence_row(
            evidence_out=evidence_out,
            auth_eval=auth_eval,
            job_name="arch-test",
            archive_path=None,
        )


@pytest.mark.parametrize(
    ("script_path", "job_name", "sdk_job_name"),
    [
        (
            "experiments/lossy_coarsening_lightning_harvest.py",
            "lossy-coarsening-cuda-20260508T024250Z",
            "lossy-coarsening-cuda-20260508t024250z",
        ),
        (
            "experiments/arch_shrink_x0.4_lightning_harvest.py",
            "arch-shrink-x0-4-lightning-20260508T024304Z",
            "arch-shrink-x0-4-lightning-20260508t024304z",
        ),
    ],
)
def test_lightning_harvest_rsync_falls_back_to_teamspace_jobs_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_path: str,
    job_name: str,
    sdk_job_name: str,
) -> None:
    module = _load_script(script_path)
    calls: list[list[str]] = []

    class Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def fake_run(cmd: list[str], **_kwargs: object) -> Result:
        calls.append(cmd)
        return Result(23 if len(calls) == 1 else 0)

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(module, "rsync_progress_args", lambda _name: [])
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    local_dir = module._rsync_artifacts(
        ssh_target="studio",
        remote_pact="/teamspace/studios/this_studio/pact",
        job_name=job_name,
    )

    remote_args = [cmd[-2] for cmd in calls]
    assert local_dir == tmp_path / "experiments" / "results" / "lightning_batch" / job_name
    assert (
        f"studio:/teamspace/studios/this_studio/pact/experiments/results/"
        f"lightning_batch/{job_name}/"
    ) in remote_args
    assert (
        f"studio:/teamspace/jobs/{sdk_job_name}/artifacts/pact/experiments/"
        f"results/lightning_batch/{job_name}/"
    ) in remote_args


@pytest.mark.parametrize(
    "script_path",
    [
        "experiments/lossy_coarsening_lightning_harvest.py",
        "experiments/arch_shrink_x0.4_lightning_harvest.py",
    ],
)
def test_lightning_harvest_tracks_actual_auth_eval_json_path(
    tmp_path: Path,
    script_path: str,
) -> None:
    module = _load_script(script_path)
    nested = tmp_path / "artifacts" / "auth_eval_work"
    nested.mkdir(parents=True)
    expected = _strict_auth_eval(archive_bytes=333)
    (nested / "contest_auth_eval.json").write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    parsed = module._parse_auth_eval_json(tmp_path / "artifacts")

    assert parsed is not None
    assert parsed["_auth_eval_json_path"] == str(nested / "contest_auth_eval.json")


def test_lossy_harvest_prefers_nested_auth_eval_work_path(tmp_path: Path) -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_harvest.py")
    local_dir = tmp_path / "artifacts"
    nested = local_dir / "auth_eval_work"
    nested.mkdir(parents=True)
    top_level = _strict_auth_eval(archive_bytes=111)
    nested_eval = _strict_auth_eval(archive_bytes=333)
    (local_dir / "contest_auth_eval.json").write_text(
        json.dumps(top_level),
        encoding="utf-8",
    )
    (nested / "contest_auth_eval.json").write_text(
        json.dumps(nested_eval),
        encoding="utf-8",
    )

    parsed = module._parse_auth_eval_json(local_dir)

    assert parsed is not None
    assert parsed["archive_size_bytes"] == 333
    assert parsed["_auth_eval_json_path"] == str(nested / "contest_auth_eval.json")


def test_lossy_terminal_harvest_reports_nested_auth_eval_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script("experiments/lossy_coarsening_lightning_harvest.py")
    job_name = "lossy-test"
    local_dir = (
        tmp_path
        / "experiments"
        / "results"
        / "lightning_batch"
        / job_name
    )
    nested = local_dir / "auth_eval_work"
    nested.mkdir(parents=True)
    (nested / "contest_auth_eval.json").write_text(
        json.dumps(_strict_auth_eval(archive_bytes=333)),
        encoding="utf-8",
    )
    active_rows = [
        {
            "lane_id": "lossy_coarsening_analytical_cuda",
            "job_name": job_name,
            "terminal_status": None,
        }
    ]
    saved_rows: list[list[dict[str, object]]] = []
    terminal_claims: list[dict[str, object]] = []
    evidence_out = tmp_path / "evidence.jsonl"

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "_rsync_artifacts", lambda **_kwargs: local_dir)
    monkeypatch.setattr(module, "_load_active_jobs", lambda: [dict(active_rows[0])])
    monkeypatch.setattr(module, "_save_active_jobs", lambda rows: saved_rows.append(rows))
    monkeypatch.setattr(
        module,
        "_terminal_claim",
        lambda **kwargs: terminal_claims.append(kwargs),
    )

    args = SimpleNamespace(
        ssh_target="studio",
        remote_pact="/teamspace/studios/this_studio/pact",
        evidence_out=evidence_out,
    )

    rc = module._harvest_terminal(
        target={"job_name": job_name, "archive_bytes": 333},
        args=args,
    )

    expected_path = (
        f"experiments/results/lightning_batch/{job_name}/"
        "auth_eval_work/contest_auth_eval.json"
    )
    evidence = json.loads(evidence_out.read_text(encoding="utf-8"))
    assert rc == 0
    assert evidence["auth_eval_json"] == expected_path
    assert expected_path in evidence["source"]
    assert terminal_claims[0]["status"].startswith("completed_score_")
    assert terminal_claims[0]["notes"].endswith(f"artifact={expected_path}")
    assert saved_rows[0][0]["harvested_auth_eval_json"] == expected_path


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
    "script_path",
    [
        "experiments/lossy_coarsening_lightning_harvest.py",
        "experiments/arch_shrink_x0.4_lightning_harvest.py",
    ],
)
def test_lightning_terminal_claim_failure_is_loud(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_path: str,
) -> None:
    module = _load_script(script_path)

    class Result:
        returncode = 2

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: Result())

    with pytest.raises(RuntimeError, match="terminal claim failed"):
        module._terminal_claim(
            job_name="job-test",
            status="failed_test",
            notes="test",
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


@pytest.mark.parametrize(
    ("script_path", "lane_id", "job_name", "log_name", "log_text", "expected_status"),
    [
        (
            "experiments/lossy_coarsening_lightning_harvest.py",
            "lossy_coarsening_analytical_cuda",
            "lossy-test",
            "auth_eval.log",
            "Traceback\nModuleNotFoundError: No module named 'brotli'\n",
            "failed_dependency_runtime_missing_brotli_before_score",
        ),
        (
            "experiments/arch_shrink_x0.4_lightning_harvest.py",
            "arch_shrink_x0.4_lightning",
            "arch-test",
            "train.log",
            "RuntimeError: Q-FAITHFUL forward requires an explicit deployed pose tensor\n",
            "failed_train_contract_qfaithful_pose_missing_before_score",
        ),
    ],
)
def test_lightning_terminal_harvest_classifies_no_score_failures_and_closes_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_path: str,
    lane_id: str,
    job_name: str,
    log_name: str,
    log_text: str,
    expected_status: str,
) -> None:
    module = _load_script(script_path)
    local_dir = tmp_path / "artifacts"
    local_dir.mkdir()
    (local_dir / log_name).write_text(log_text, encoding="utf-8")
    active_rows = [{"lane_id": lane_id, "job_name": job_name, "terminal_status": None}]
    saved_rows: list[list[dict[str, object]]] = []
    terminal_claims: list[dict[str, object]] = []

    monkeypatch.setattr(module, "_rsync_artifacts", lambda **_kwargs: local_dir)
    monkeypatch.setattr(module, "_load_active_jobs", lambda: [dict(active_rows[0])])
    monkeypatch.setattr(module, "_save_active_jobs", lambda rows: saved_rows.append(rows))
    monkeypatch.setattr(
        module,
        "_terminal_claim",
        lambda **kwargs: terminal_claims.append(kwargs),
    )

    args = SimpleNamespace(
        ssh_target="studio",
        remote_pact="/teamspace/studios/this_studio/pact",
        evidence_out=tmp_path / "evidence.jsonl",
    )

    with pytest.raises(SystemExit, match=r"no contest_auth_eval\.json"):
        module._harvest_terminal(target={"job_name": job_name}, args=args)

    classification_path = local_dir / "lightning_no_score_failure_classification.json"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    assert classification["terminal_status"] == expected_status
    assert classification["score_claim"] is False
    assert terminal_claims[0]["status"] == expected_status
    assert saved_rows[0][0]["terminal_status"] == expected_status


@pytest.mark.parametrize(
    ("script_path", "lane_id", "job_name", "target"),
    [
        (
            "experiments/lossy_coarsening_lightning_harvest.py",
            "lossy_coarsening_analytical_cuda",
            "lossy-test",
            {"job_name": "lossy-test", "archive_bytes": 156404},
        ),
        (
            "experiments/arch_shrink_x0.4_lightning_harvest.py",
            "arch_shrink_x0.4_lightning",
            "arch-test",
            {"job_name": "arch-test"},
        ),
    ],
)
def test_lightning_terminal_harvest_closes_claim_on_invalid_auth_eval_custody(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script_path: str,
    lane_id: str,
    job_name: str,
    target: dict[str, object],
) -> None:
    module = _load_script(script_path)
    local_dir = tmp_path / "artifacts"
    local_dir.mkdir()
    (local_dir / "archive.zip").write_bytes(b"x" * 156404)
    (local_dir / "contest_auth_eval.json").write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.189,
                "archive_size_bytes": 156404,
                "avg_posenet_dist": 0.0001,
                "avg_segnet_dist": 0.001,
                "rate_unscaled": 156404 / CONTEST_DENOMINATOR,
            }
        ),
        encoding="utf-8",
    )
    evidence_out = tmp_path / "evidence.jsonl"
    active_rows = [{"lane_id": lane_id, "job_name": job_name, "terminal_status": None}]
    saved_rows: list[list[dict[str, object]]] = []
    terminal_claims: list[dict[str, object]] = []

    monkeypatch.setattr(module, "_rsync_artifacts", lambda **_kwargs: local_dir)
    monkeypatch.setattr(module, "_load_active_jobs", lambda: [dict(active_rows[0])])
    monkeypatch.setattr(module, "_save_active_jobs", lambda rows: saved_rows.append(rows))
    monkeypatch.setattr(
        module,
        "_terminal_claim",
        lambda **kwargs: terminal_claims.append(kwargs),
    )

    args = SimpleNamespace(
        ssh_target="studio",
        remote_pact="/teamspace/studios/this_studio/pact",
        evidence_out=evidence_out,
    )

    with pytest.raises(SystemExit, match="rejected before evidence emission"):
        module._harvest_terminal(target=target, args=args)

    assert not evidence_out.exists()
    assert terminal_claims[0]["status"] == "failed_invalid_auth_eval_custody"
    assert "auth_eval custody blockers" in terminal_claims[0]["notes"]
    assert saved_rows[0][0]["terminal_status"] == "failed_invalid_auth_eval_custody"
