from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "launch_lightning_alpha_geo0_pose_regen.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("launch_lightning_alpha_geo0_pose_regen_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_claim(
    claims: Path,
    *,
    lane_id: str,
    instance_job_id: str,
    status: str,
) -> None:
    claims.write_text(
        "# Active lane dispatch claims -- test\n\n"
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-05T18:00:00Z | codex:test | {lane_id} | lightning | {instance_job_id} | 2026-05-05T19:00Z | {status} | test |\n",
        encoding="utf-8",
    )


def _write_required_artifacts(repo_root: Path) -> dict[str, str]:
    rels = {
        "candidate": "candidate/archive.zip",
        "baseline": "baseline/archive.zip",
        "warm": "poses/warm.pt",
        "targets": "poses/targets.pt",
    }
    for rel in rels.values():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{rel}\n".encode())
    return rels


def test_alpha_geo0_lightning_identity_defaults_are_provider_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    for name in (
        "LIGHTNING_STUDIO",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_SDK_USER",
        "LIGHTNING_USER",
    ):
        monkeypatch.delenv(name, raising=False)

    args = mod.build_parser().parse_args(["--job-name", "Alpha_Geo0_Test"])

    assert args.studio == ""
    assert args.teamspace == ""
    assert args.sdk_user == ""


def test_non_dry_run_alpha_geo0_submit_requires_active_dispatch_claim(tmp_path: Path) -> None:
    mod = _load_module()
    claims = tmp_path / "active_lane_dispatch_claims.md"
    args = mod.build_parser().parse_args(
        [
            "--job-name",
            "Alpha_Geo0_Test",
            "--dispatch-lane-id",
            "lane_alpha_geo0",
            "--dispatch-claims-path",
            str(claims),
        ]
    )

    with pytest.raises(SystemExit, match="missing active dispatch claim"):
        mod._require_alpha_geo0_dispatch_claim(args)

    _write_claim(
        claims,
        lane_id="lane_alpha_geo0",
        instance_job_id="alpha-geo0-test",
        status="eval",
    )
    mod._require_alpha_geo0_dispatch_claim(args)

    _write_claim(
        claims,
        lane_id="lane_alpha_geo0",
        instance_job_id="alpha-geo0-test",
        status="completed_score=0.42",
    )
    with pytest.raises(SystemExit, match="missing active dispatch claim"):
        mod._require_alpha_geo0_dispatch_claim(args)


def test_alpha_geo0_break_glass_reason_is_recorded_in_queue_metadata(tmp_path: Path) -> None:
    mod = _load_module()
    args = mod.build_parser().parse_args(
        [
            "--job-name",
            "Alpha_Geo0_Test",
            "--dispatch-claims-path",
            str(tmp_path / "claims.md"),
            "--allow-missing-dispatch-claim-reason",
            "operator audited emergency provider-state recovery",
        ]
    )

    mod._require_alpha_geo0_dispatch_claim(args)
    assert mod._dispatch_claim_metadata(args) == {
        "dispatch_claim_skip_reason": "operator audited emergency provider-state recovery",
    }


def test_alpha_geo0_main_checks_claim_before_remote_preflight_or_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    rels = _write_required_artifacts(tmp_path)
    called = {"remote_preflight": False, "client": False}

    def fake_remote_preflight(args) -> None:
        called["remote_preflight"] = True

    class FakeClient:
        def __init__(self, *, state_path: Path) -> None:
            called["client"] = True

        def submit(self, spec, *, dry_run: bool = False):
            raise AssertionError("submit must not run before dispatch claim validation")

    monkeypatch.setattr(mod, "_remote_supply_chain_preflight", fake_remote_preflight)
    monkeypatch.setattr(mod, "LightningBatchJobsClient", FakeClient)

    with pytest.raises(SystemExit, match="missing active dispatch claim"):
        mod.main(
            [
                "--job-name",
                "Alpha_Geo0_Test",
                "--candidate-archive",
                rels["candidate"],
                "--baseline-archive",
                rels["baseline"],
                "--warm-poses",
                rels["warm"],
                "--gt-pose-targets",
                rels["targets"],
                "--dispatch-lane-id",
                "lane_alpha_geo0",
                "--dispatch-claims-path",
                str(tmp_path / "claims.md"),
                "--state-path",
                str(tmp_path / "jobs.json"),
            ]
        )

    assert called == {"remote_preflight": False, "client": False}
