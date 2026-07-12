from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.deploy.witness_cloud_launcher import (
    ASSET_REUSE_DEFER,
    CHILD_TIMEOUT_SECONDS,
    CUDA_ENV,
    DEFAULT_STOP_AFTER_EPOCHS,
    EXACT_H100_GPU,
    H100_GPU_USD_PER_HOUR,
    INVOCATION_TIMEOUT_SECONDS,
    LANE_ID,
    REVIEWED_SENTINELS,
    TASK438_GT_CACHE_BYTES,
    TASK438_GT_CACHE_SHA256,
    build_plan,
)


def _kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "provider": "modal",
        "source_git_head": "a" * 40,
        "gt_cache": "cache.npz",
        "gt_cache_sha256": TASK438_GT_CACHE_SHA256,
        "gt_cache_bytes": TASK438_GT_CACHE_BYTES,
        "segnet_sha256": "b" * 64,
        "posenet_sha256": "c" * 64,
        "label": "unit-cuda",
        "gpu": EXACT_H100_GPU,
        "epochs": 3000,
        "num_pairs": 600,
    }
    values.update(overrides)
    return values


def test_modal_plan_is_deterministic_sha_bound_and_cost_conservative():
    a = build_plan(**_kwargs())
    b = build_plan(**_kwargs())
    assert a.schema == "witness_cloud_plan.v7"
    assert a.plan_sha256 == b.plan_sha256
    assert a.lane_id == LANE_ID
    assert a.source_git_head == "a" * 40
    assert a.stop_after_epochs == DEFAULT_STOP_AFTER_EPOCHS == 3
    assert a.epochs == 3000
    assert a.num_pairs == 600
    assert a.child_timeout_seconds == CHILD_TIMEOUT_SECONDS == 1500
    assert a.invocation_timeout_seconds == INVOCATION_TIMEOUT_SECONDS == 1800
    assert a.gpu_usd_per_hour_ceiling == H100_GPU_USD_PER_HOUR == 5.0
    assert a.gpu == EXACT_H100_GPU == "H100"
    assert a.planned_total_cost_usd == pytest.approx(3.296256)
    assert a.planned_total_cost_usd <= a.max_plan_cost_usd == 5.0
    assert a.budgeted_image_staging_allowance_usd == 0.50
    assert "not provider-enforced or measured" in a.cost_assumptions_as_of
    assert a.dispatch_argv[:4] == (
        ".venv/bin/modal", "run", "--detach", "experiments/modal_train_lane.py"
    )
    assert a.dispatch_argv[a.dispatch_argv.index("--expected-mounted-code-git-head") + 1] == "a" * 40
    assert "--require-clean-head" in a.dispatch_argv
    assert a.dispatch_argv[a.dispatch_argv.index("--timeout-hours") + 1] == "0.416666667"
    assert a.dispatch_argv[a.dispatch_argv.index("--expected-cost-usd") + 1] == "3.296256"
    assert a.dispatch_argv[a.dispatch_argv.index("--gpu") + 1] == EXACT_H100_GPU
    assert "--preflight-first" in a.dispatch_argv
    assert a.reviewed_sentinels == REVIEWED_SENTINELS
    assert "src/tac/cuda_v9_throughput.py" in a.reviewed_sentinels
    assert "upstream/models/segnet.safetensors" in a.reviewed_sentinels
    assert "upstream/models/posenet.safetensors" in a.reviewed_sentinels
    assert a.environment["WITNESS_TRAINER_MODE"] == "full"
    assert a.environment["DALI_DISABLE_NVML"] == CUDA_ENV["DALI_DISABLE_NVML"]
    assert a.gt_cache_bytes == TASK438_GT_CACHE_BYTES
    assert a.remote_gt_cache == (
        f"/modal_results/assets/v9_cgauge/gt_{TASK438_GT_CACHE_SHA256}.npz"
    )
    assert a.environment["WITNESS_SEGNET_SHA256"] == "b" * 64
    assert a.environment["WITNESS_POSENET_SHA256"] == "c" * 64
    assert a.resume_from is None
    assert a.resume_sha256 is None
    assert a.environment["WITNESS_RESUME_FROM"] == ""
    assert a.environment["WITNESS_RESUME_SHA256"] == ""
    assert a.deferred_egress_blockers == (ASSET_REUSE_DEFER,)
    assert "never runs asset_stage_argv" in ASSET_REUSE_DEFER
    assert a.harvest_argv == (
        ".venv/bin/python", "tools/harvest_modal_calls.py",
        "--from-ledger", "--call-id", "<call-id-from-spawn>", "--execute",
    )


@pytest.mark.parametrize("missing", ("gt_cache_sha256", "segnet_sha256", "posenet_sha256"))
def test_missing_sha_blocks_execution_and_never_stages_unaddressed_asset(missing: str):
    plan = build_plan(**_kwargs(**{missing: None}))
    assert (plan.remote_gt_cache is None) is (missing == "gt_cache_sha256")
    assert plan.asset_stage_argv == ()
    assert plan.dispatch_argv == ()
    assert not plan.execution_allowed
    assert any("custody value is not supplied" in blocker for blocker in plan.setup_blockers)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label", "Bad_Label"),
        ("label", "unsafe/path"),
        ("label", "a" * 64),
        ("gpu", "T4"),
        ("gpu", "H100!"),
        ("gpu", "H100-80GB"),
        ("epochs", 3),
        ("num_pairs", 2),
        ("stop_after_epochs", 0),
        ("stop_after_epochs", 4),
        ("child_timeout_seconds", 0),
        ("invocation_timeout_seconds", float("nan")),
        ("max_plan_cost_usd", 5.01),
        ("gt_cache_sha256", "not-a-sha"),
        ("segnet_sha256", "not-a-sha"),
        ("posenet_sha256", "not-a-sha"),
        ("source_git_head", "not-a-git-head"),
        ("resume_from", "/modal_results/foo/../escape.pt"),
        ("resume_from", "/modal_results/has,comma/checkpoint.pt"),
        ("resume_from", "/modal_results/has=equals/checkpoint.pt"),
        ("resume_from", "/modal_results/has whitespace/checkpoint.pt"),
        ("resume_from", "/modal_results/has\tcontrol/checkpoint.pt"),
        ("resume_from", "/modal_results//non-normalized/checkpoint.pt"),
        ("resume_from", "/modal_results/non-normalized/./checkpoint.pt"),
    ],
)
def test_plan_refuses_unsafe_or_nonconservative_inputs(field: str, value: object):
    with pytest.raises(ValueError):
        build_plan(**_kwargs(**{field: value}))


def test_resume_path_is_normalized_modal_custody_and_in_plan_hash():
    plan = build_plan(
        **_kwargs(
            resume_from="/modal_results/unit-cuda/output/checkpoint.pt",
            resume_sha256="D" * 64,
        )
    )
    assert plan.resume_from == "/modal_results/unit-cuda/output/checkpoint.pt"
    assert plan.resume_sha256 == "d" * 64
    assert plan.environment["WITNESS_RESUME_FROM"] == plan.resume_from
    assert plan.environment["WITNESS_RESUME_SHA256"] == plan.resume_sha256
    assert plan.plan_sha256 != build_plan(**_kwargs()).plan_sha256


@pytest.mark.parametrize(
    ("resume_from", "resume_sha256"),
    [
        ("/modal_results/unit-cuda/output/checkpoint.pt", None),
        (None, "d" * 64),
        ("/modal_results/unit-cuda/output/checkpoint.pt", "not-a-sha"),
    ],
)
def test_resume_path_and_sha_are_required_iff(resume_from, resume_sha256):
    with pytest.raises(ValueError):
        build_plan(**_kwargs(resume_from=resume_from, resume_sha256=resume_sha256))


def test_plan_hash_changes_when_reviewed_source_head_changes():
    at_a = build_plan(**_kwargs(source_git_head="a" * 40))
    at_b = build_plan(**_kwargs(source_git_head="b" * 40))
    assert at_a.plan_sha256 != at_b.plan_sha256


@pytest.mark.parametrize("field", ("segnet_sha256", "posenet_sha256"))
def test_plan_hash_changes_when_scorer_custody_changes(field: str):
    baseline = build_plan(**_kwargs())
    changed = build_plan(**_kwargs(**{field: "d" * 64}))
    assert baseline.plan_sha256 != changed.plan_sha256


def test_cli_parser_threads_resume_sha256_into_plan(monkeypatch, capsys):
    tool = _load_launcher_tool()
    captured = {}

    def capture_build_plan(**kwargs):
        captured.update(kwargs)
        return build_plan(**kwargs)

    monkeypatch.setattr(tool, "_main_git_head", lambda: "a" * 40)
    monkeypatch.setattr(
        tool,
        "_required_file_sha256",
        lambda _path, *, custody_name: "b" * 64 if custody_name == "SegNet" else "c" * 64,
    )
    monkeypatch.setattr(
        tool,
        "_verify_local_asset_stage_contract",
        lambda _plan: {"schema": "witness_asset_stage_readiness.v1", "status": "passed"},
    )
    monkeypatch.setattr(tool, "build_plan", capture_build_plan)
    assert tool.main(
        [
            "--gt-cache-sha256",
            TASK438_GT_CACHE_SHA256,
            "--resume-from",
            "/modal_results/unit-cuda/output/checkpoint.pt",
            "--resume-sha256",
            "D" * 64,
        ]
    ) == 0
    capsys.readouterr()
    assert captured["resume_from"] == "/modal_results/unit-cuda/output/checkpoint.pt"
    assert captured["resume_sha256"] == "D" * 64
    assert captured["gpu"] == EXACT_H100_GPU
    assert captured["gt_cache_bytes"] == TASK438_GT_CACHE_BYTES


def test_cli_refuses_noncanonical_task438_gt_digest_before_plan_build(monkeypatch):
    tool = _load_launcher_tool()
    monkeypatch.setattr(
        tool,
        "build_plan",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("plan build reached")),
    )
    with pytest.raises(SystemExit, match="canonical Task438 n600 digest"):
        tool.main(["--gt-cache-sha256", "a" * 64])


def test_local_asset_stage_readiness_binds_bytes_sha_volume_and_mount(
    monkeypatch, tmp_path
):
    tool = _load_launcher_tool()
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"exact-gt-cache")
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    plan = build_plan(
        **_kwargs(
            gt_cache="cache.npz",
            gt_cache_sha256=digest,
            gt_cache_bytes=cache.stat().st_size,
        )
    )
    monkeypatch.setattr(tool, "REPO", tmp_path)
    receipt = tool._verify_local_asset_stage_contract(plan)
    assert receipt == {
        "schema": "witness_asset_stage_readiness.v1",
        "status": "passed",
        "provider_contacted": False,
        "staging_executed": False,
        "volume": "comma-train-lane-results",
        "local_path": "cache.npz",
        "bytes": cache.stat().st_size,
        "sha256": digest,
        "volume_relative_path": f"assets/v9_cgauge/gt_{digest}.npz",
        "mounted_path": f"/modal_results/assets/v9_cgauge/gt_{digest}.npz",
        "asset_stage_argv": [
            ".venv/bin/modal",
            "volume",
            "put",
            "comma-train-lane-results",
            "cache.npz",
            f"assets/v9_cgauge/gt_{digest}.npz",
        ],
    }


def _load_launcher_tool():
    path = Path(__file__).resolve().parents[3] / "tools/launch_witness_cloud.py"
    spec = importlib.util.spec_from_file_location("launch_witness_cloud_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _execution_fixture(tmp_path, *, resume_from=None):
    tool = _load_launcher_tool()
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"exact-gt-cache")
    segnet = tmp_path / "upstream/models/segnet.safetensors"
    posenet = tmp_path / "upstream/models/posenet.safetensors"
    segnet.parent.mkdir(parents=True)
    segnet.write_bytes(b"reviewed-segnet")
    posenet.write_bytes(b"reviewed-posenet")
    plan = replace(
        build_plan(
            **_kwargs(
                gt_cache="cache.npz",
                gt_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                gt_cache_bytes=cache.stat().st_size,
                segnet_sha256=hashlib.sha256(segnet.read_bytes()).hexdigest(),
                posenet_sha256=hashlib.sha256(posenet.read_bytes()).hexdigest(),
                resume_from=resume_from,
                resume_sha256="d" * 64 if resume_from is not None else None,
            )
        ),
        execution_allowed=True,
        plan_sha256="f" * 64,
    )
    return tool, plan, cache


def _valid_local_preflight_receipt(plan) -> dict[str, object]:
    return {
        "schema": "v9_cgauge_torch_preflight.v1",
        "status": "passed",
        "typed_total_epochs": plan.epochs,
        "runtime_stop_after_epochs": plan.stop_after_epochs,
        "runtime_epoch_window": [1, plan.stop_after_epochs],
        "resume_epoch": 0,
        "num_pairs": plan.num_pairs,
        "output_created": False,
        "no_implicit_resume": True,
        "resume_checkpoint": None,
        "cuda_v9_port_coverage": {"status": "COMPLETE_1_TO_1", "blockers": []},
        "scorer_custody": {
            "segnet": {
                "sha256": plan.segnet_sha256,
                "expected_sha256": plan.segnet_sha256,
                "sha_authority": "PLAN_EXPECTED_MATCH",
            },
            "posenet": {
                "sha256": plan.posenet_sha256,
                "expected_sha256": plan.posenet_sha256,
                "sha_authority": "PLAN_EXPECTED_MATCH",
            },
        },
        "scorer_constructor_load": {
            "status": "passed",
            "device": "cpu",
            "networks": {
                "segnet": {"class": "SegNet", "eval": True, "frozen": True},
                "posenet": {"class": "PoseNet", "eval": True, "frozen": True},
            },
        },
    }


def test_head_drift_refuses_before_any_provider_mutation(monkeypatch):
    tool = _load_launcher_tool()
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        command = tuple(command)
        calls.append(command)
        if command[1:] == ("branch", "--show-current"):
            return SimpleNamespace(stdout="main\n")
        if command[1:] == ("status", "--porcelain=v1", "--untracked-files=all"):
            return SimpleNamespace(stdout="")
        if command[1:] == ("rev-parse", "main"):
            return SimpleNamespace(stdout=("b" * 40) + "\n")
        raise AssertionError(f"unexpected subprocess call: {command}")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="HEAD changed after plan review"):
        tool._require_clean_main_worktree("a" * 40)
    assert not any("modal" in command or "volume" in command for command in calls)


def test_scorer_drift_refuses_before_any_provider_mutation(monkeypatch, tmp_path):
    tool = _load_launcher_tool()
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"exact-gt-cache")
    segnet = tmp_path / "upstream/models/segnet.safetensors"
    posenet = tmp_path / "upstream/models/posenet.safetensors"
    segnet.parent.mkdir(parents=True)
    segnet.write_bytes(b"reviewed-segnet")
    posenet.write_bytes(b"reviewed-posenet")
    plan = replace(
        build_plan(
            **_kwargs(
                gt_cache="cache.npz",
                gt_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                gt_cache_bytes=cache.stat().st_size,
                segnet_sha256="0" * 64,
                posenet_sha256=hashlib.sha256(posenet.read_bytes()).hexdigest(),
            )
        ),
        execution_allowed=True,
        plan_sha256="f" * 64,
    )
    provider_calls: list[tuple[str, ...]] = []

    def forbid_provider(command, **_kwargs):
        provider_calls.append(tuple(command))
        raise AssertionError(f"provider mutation reached: {command}")

    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "_main_git_head", lambda: "a" * 40)
    monkeypatch.setattr(tool, "_require_clean_main_worktree", lambda _head: None)
    monkeypatch.setattr(tool, "build_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(tool.subprocess, "run", forbid_provider)
    with pytest.raises(SystemExit, match="SegNet SHA-256 changed after plan review"):
        tool.main(
            [
                "--execute",
                "--operator-go-token",
                tool.OPERATOR_TOKEN,
                "--expected-plan-sha256",
                plan.plan_sha256,
            ]
        )
    assert provider_calls == []


@pytest.mark.parametrize(
    ("cache_bypass", "value"),
    (("MODAL_FORCE_BUILD", "1"), ("MODAL_IGNORE_CACHE", " ")),
)
def test_image_cache_bypass_refuses_before_any_provider_mutation(
    monkeypatch, tmp_path, cache_bypass: str, value: str
):
    tool = _load_launcher_tool()
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"exact-gt-cache")
    segnet = tmp_path / "upstream/models/segnet.safetensors"
    posenet = tmp_path / "upstream/models/posenet.safetensors"
    segnet.parent.mkdir(parents=True)
    segnet.write_bytes(b"reviewed-segnet")
    posenet.write_bytes(b"reviewed-posenet")
    plan = replace(
        build_plan(
            **_kwargs(
                gt_cache="cache.npz",
                gt_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                gt_cache_bytes=cache.stat().st_size,
                segnet_sha256=hashlib.sha256(segnet.read_bytes()).hexdigest(),
                posenet_sha256=hashlib.sha256(posenet.read_bytes()).hexdigest(),
            )
        ),
        execution_allowed=True,
        plan_sha256="f" * 64,
    )
    provider_calls: list[tuple[str, ...]] = []

    def forbid_provider(command, **_kwargs):
        provider_calls.append(tuple(command))
        raise AssertionError(f"provider mutation reached: {command}")

    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "_main_git_head", lambda: "a" * 40)
    monkeypatch.setattr(tool, "build_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(tool.subprocess, "run", forbid_provider)
    monkeypatch.setenv(cache_bypass, value)
    with pytest.raises(SystemExit, match="image-cache bypass"):
        tool.main(
            [
                "--execute",
                "--operator-go-token",
                tool.OPERATOR_TOKEN,
                "--expected-plan-sha256",
                plan.plan_sha256,
            ]
        )
    assert provider_calls == []


def test_lane_global_active_claim_refuses_different_label(monkeypatch):
    tool = _load_launcher_tool()
    plan = SimpleNamespace(lane_id=LANE_ID)
    payload = {
        "active": [
            {
                "lane_id": LANE_ID,
                "instance_job_id": "another-v9-label",
                "status": "active_dispatching",
            }
        ]
    }
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )
    with pytest.raises(SystemExit, match="lane-global duplicate spending"):
        tool._require_no_active_v9_claim(plan)


def test_outer_dispatch_guard_identity_is_lane_global_and_distinct(monkeypatch, tmp_path):
    tool = _load_launcher_tool()
    monkeypatch.setattr(tool, "REPO", tmp_path)
    first = SimpleNamespace(lane_id=LANE_ID, label="first-label")
    second = SimpleNamespace(lane_id=LANE_ID, label="second-label")
    first_path = tool._outer_v9_guard_path(first)
    second_path = tool._outer_v9_guard_path(second)
    assert first_path == second_path
    assert first_path.parent.name == "witness_cloud_outer_dispatch_guards"
    assert first_path.parent.name != "modal_train_lane_dispatch_guards"


def test_outer_dispatch_guard_releases_on_failure(monkeypatch, tmp_path):
    tool = _load_launcher_tool()
    operations = []
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(
        tool.fcntl,
        "flock",
        lambda _fd, operation: operations.append(operation),
    )
    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        tool._outer_v9_lane_dispatch_guard(SimpleNamespace(lane_id=LANE_ID)),
    ):
        raise RuntimeError("synthetic failure")
    assert operations == [tool.fcntl.LOCK_EX, tool.fcntl.LOCK_UN]


def test_local_modal_app_definition_preflight_is_zero_provider(monkeypatch, capsys):
    tool = _load_launcher_tool()
    provider_calls = []

    class FakeFunction:
        def with_options(
            self, *, gpu=None, timeout=None, retries=None, cpu=None, memory=None
        ):
            return None

    class PoisonVolume:
        @classmethod
        def from_name(cls, name, *, create_if_missing=False):
            provider_calls.append((name, create_if_missing))
            raise AssertionError("provider lookup must not run")

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(Function=FakeFunction, Volume=PoisonVolume),
    )
    tool._run_local_modal_app_definition_preflight()
    assert provider_calls == []
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["schema"] == "witness_modal_app_definition_preflight.v1"
    assert receipt["provider_contacted"] is False
    assert receipt["exported_endpoint"] == "run_lane_training_cpu"
    assert receipt["static_gpu"] is False
    assert receipt["static_timeout_seconds"] == 600


def test_local_modal_app_definition_preflight_refuses_incompatible_sdk(monkeypatch):
    tool = _load_launcher_tool()

    class MissingGpuFunction:
        def with_options(self, *, timeout=None, retries=None, cpu=None, memory=None):
            return None

    class FakeVolume:
        @classmethod
        def from_name(cls, name, *, create_if_missing=False):
            return None

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(Function=MissingGpuFunction, Volume=FakeVolume),
    )
    with pytest.raises(SystemExit, match="with_options lacks required controls"):
        tool._run_local_modal_app_definition_preflight()


def test_claim_summary_error_prevents_local_preflight_and_provider_io(monkeypatch, tmp_path):
    tool, plan, _cache = _execution_fixture(tmp_path)
    events: list[str] = []

    def clean(_head):
        events.append("clean")

    def claim_failure(_plan):
        events.append("claim_check")
        raise SystemExit("REFUSED: claim summary unavailable")

    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "_main_git_head", lambda: "a" * 40)
    monkeypatch.setattr(tool, "_require_clean_main_worktree", clean)
    monkeypatch.setattr(tool, "build_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(tool, "_require_no_active_v9_claim", claim_failure)
    monkeypatch.setattr(
        tool, "_run_local_v9_preflight", lambda *_args, **_kwargs: events.append("preflight")
    )
    monkeypatch.setattr(
        tool, "_modal_asset_is_reusable", lambda *_args, **_kwargs: events.append("modal")
    )
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda command, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"provider I/O reached: {command}")
        ),
    )
    with pytest.raises(SystemExit, match="claim summary unavailable"):
        tool.main(
            [
                "--execute",
                "--operator-go-token",
                tool.OPERATOR_TOKEN,
                "--expected-plan-sha256",
                plan.plan_sha256,
            ]
        )
    assert events == ["clean", "clean", "claim_check"]


@pytest.mark.parametrize("resume_from", (None, "/modal_results/unit-cuda/output/checkpoint.pt"))
def test_local_v9_preflight_uses_real_strict_flags_and_no_output(
    monkeypatch, tmp_path, capsys, resume_from
):
    tool, plan, cache = _execution_fixture(tmp_path, resume_from=resume_from)
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **kwargs):
        calls.append(tuple(command))
        assert kwargs["cwd"] == tmp_path
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_valid_local_preflight_receipt(plan)),
            stderr="",
        )

    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    tool._run_local_v9_preflight(plan, local_cache=cache)
    assert len(calls) == 1
    command = calls[0]
    expected_values = {
        "--gt-cache": str(cache),
        "--num-pairs": "600",
        "--epochs": "3000",
        "--stop-after-epochs": "3",
        "--expected-segnet-sha256": plan.segnet_sha256,
        "--expected-posenet-sha256": plan.posenet_sha256,
        "--device": "cpu",
    }
    for flag, value in expected_values.items():
        assert command[command.index(flag) + 1] == value
    assert "--no-implicit-resume" in command
    assert "--preflight-only" in command
    assert "--resume-from" not in command
    out = Path(command[command.index("--out-dir") + 1])
    assert ".omx/preflight/witness_cloud_launcher" in out.as_posix()
    assert not out.exists()
    scope = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert scope["schema"] == "witness_local_v9_preflight_scope.v1"
    assert scope["remote_resume_requested"] is (resume_from is not None)
    assert scope["remote_resume_content_validated"] is False
    assert "provider CPU preflight" in scope["remote_resume_authority"]


def test_local_preflight_failure_prevents_modal_lookup_and_upload(monkeypatch, tmp_path):
    tool, plan, _cache = _execution_fixture(tmp_path)
    events: list[str] = []

    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "_main_git_head", lambda: "a" * 40)
    monkeypatch.setattr(
        tool, "_require_clean_main_worktree", lambda _head: events.append("clean")
    )
    monkeypatch.setattr(tool, "build_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        tool, "_require_no_active_v9_claim", lambda _plan: events.append("claim_check")
    )
    monkeypatch.setattr(
        tool,
        "_run_local_modal_app_definition_preflight",
        lambda: events.append("app_preflight"),
    )

    def fail_preflight(*_args, **_kwargs):
        events.append("local_preflight")
        raise SystemExit("REFUSED: synthetic local preflight failure")

    monkeypatch.setattr(tool, "_run_local_v9_preflight", fail_preflight)
    monkeypatch.setattr(
        tool, "_modal_asset_is_reusable", lambda *_args, **_kwargs: events.append("modal")
    )
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda command, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"provider I/O reached: {command}")
        ),
    )
    with pytest.raises(SystemExit, match="synthetic local preflight failure"):
        tool.main(
            [
                "--execute",
                "--operator-go-token",
                tool.OPERATOR_TOKEN,
                "--expected-plan-sha256",
                plan.plan_sha256,
            ]
        )
    assert events == [
        "clean", "clean", "claim_check", "app_preflight", "local_preflight"
    ]


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("typed_total_epochs",), 2999),
        (("runtime_stop_after_epochs",), 2),
        (("runtime_epoch_window",), [1, 2]),
        (("resume_epoch",), 1),
        (("num_pairs",), 599),
        (("output_created",), True),
        (("no_implicit_resume",), False),
        (("resume_checkpoint",), "implicit.pt"),
        (("cuda_v9_port_coverage", "status"), "INCOMPLETE"),
        (("cuda_v9_port_coverage", "blockers"), ["missing"]),
        (("scorer_constructor_load", "device"), "cuda"),
        (("scorer_constructor_load", "networks", "segnet", "frozen"), False),
        (("scorer_custody", "segnet", "sha256"), "0" * 64),
        (("scorer_custody", "posenet", "sha_authority"), "MEASURED_ONLY"),
    ],
)
def test_local_preflight_refuses_partial_or_mismatched_receipt(
    monkeypatch, tmp_path, path, bad_value
):
    tool, plan, cache = _execution_fixture(tmp_path)
    receipt = copy.deepcopy(_valid_local_preflight_receipt(plan))
    cursor = receipt
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = bad_value
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(receipt), stderr=""
        ),
    )
    with pytest.raises(SystemExit, match="preflight receipt mismatch"):
        tool._run_local_v9_preflight(plan, local_cache=cache)


def test_local_preflight_refuses_schema_status_only_receipt(monkeypatch, tmp_path):
    tool, plan, cache = _execution_fixture(tmp_path)
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"schema": "v9_cgauge_torch_preflight.v1", "status": "passed"}
            ),
            stderr="",
        ),
    )
    with pytest.raises(SystemExit, match="preflight receipt mismatch"):
        tool._run_local_v9_preflight(plan, local_cache=cache)


def test_local_preflight_refuses_any_output_path_mutation(monkeypatch, tmp_path):
    tool, plan, cache = _execution_fixture(tmp_path)
    monkeypatch.setattr(tool, "REPO", tmp_path)

    def fake_run(command, **_kwargs):
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir(parents=True)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_valid_local_preflight_receipt(plan)),
            stderr="",
        )

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="mutated its must-not-exist output path"):
        tool._run_local_v9_preflight(plan, local_cache=cache)


@pytest.mark.parametrize(("remote_size", "expected"), ((17, True), (16, False)))
def test_modal_asset_reuse_requires_exact_sha_name_and_byte_size(
    monkeypatch, remote_size: int, expected: bool
):
    tool = _load_launcher_tool()
    digest = TASK438_GT_CACHE_SHA256
    remote_path = f"assets/v9_cgauge/gt_{digest}.npz"
    plan = SimpleNamespace(
        gt_cache_sha256=digest,
        label="unit-cuda",
        resume_from=None,
        asset_stage_argv=(".venv/bin/modal", "volume", "put", "volume", "local", remote_path),
    )

    class FakeVolume:
        @staticmethod
        def from_name(name, *, create_if_missing):
            assert name == "comma-train-lane-results"
            assert create_if_missing is False

            def listdir(path):
                if path == "unit-cuda":
                    return []
                return [SimpleNamespace(path=path, type=1, size=remote_size)]

            return SimpleNamespace(listdir=listdir)

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=FakeVolume))
    assert tool._modal_asset_is_reusable(plan, local_size=17) is expected


@pytest.mark.parametrize(
    ("resume_entries", "match"),
    [
        ([], "resume checkpoint is missing"),
        (
            [
                SimpleNamespace(
                    path="unit-cuda/output/checkpoint.pt", type=1, size=1024
                ),
                SimpleNamespace(
                    path="unit-cuda/output/checkpoint.pt", type=1, size=1024
                ),
            ],
            "lookup is ambiguous",
        ),
        (
            [SimpleNamespace(path="unit-cuda/output/checkpoint.pt", type=2, size=1024)],
            "not a regular file",
        ),
        (
            [SimpleNamespace(path="unit-cuda/output/checkpoint.pt", type=1, size=0)],
            "must be a nonzero file",
        ),
    ],
)
def test_modal_resume_presence_gate_refuses_invalid_remote_custody(
    monkeypatch, resume_entries, match
):
    tool = _load_launcher_tool()
    digest = TASK438_GT_CACHE_SHA256
    remote_asset = f"assets/v9_cgauge/gt_{digest}.npz"
    plan = SimpleNamespace(
        gt_cache_sha256=digest,
        label="unit-cuda",
        resume_from="/modal_results/unit-cuda/output/checkpoint.pt",
        resume_sha256="d" * 64,
        asset_stage_argv=(
            ".venv/bin/modal", "volume", "put", "volume", "local", remote_asset,
        ),
    )

    class FakeVolume:
        @staticmethod
        def from_name(_name, *, create_if_missing):
            assert create_if_missing is False

            def listdir(path):
                if path == "unit-cuda":
                    return []
                assert path == "unit-cuda/output/checkpoint.pt"
                return resume_entries

            return SimpleNamespace(listdir=listdir)

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=FakeVolume))
    with pytest.raises(SystemExit, match=match):
        tool._modal_asset_is_reusable(plan, local_size=17)


def test_modal_valid_resume_presence_reaches_asset_and_provider_preflight_path(
    monkeypatch, capsys
):
    tool = _load_launcher_tool()
    digest = TASK438_GT_CACHE_SHA256
    resume_sha256 = "d" * 64
    remote_asset = f"assets/v9_cgauge/gt_{digest}.npz"
    resume_path = "unit-cuda/output/checkpoint.pt"
    plan = build_plan(
        **_kwargs(
            resume_from=f"/modal_results/{resume_path}",
            resume_sha256=resume_sha256,
        )
    )

    class FakeVolume:
        @staticmethod
        def from_name(_name, *, create_if_missing):
            assert create_if_missing is False

            def listdir(path):
                if path == "unit-cuda":
                    return [SimpleNamespace(path="unit-cuda/output", type=2, size=0)]
                if path == resume_path:
                    return [SimpleNamespace(path=path, type=1, size=1024)]
                assert path == remote_asset
                return [SimpleNamespace(path=path, type=1, size=17)]

            return SimpleNamespace(listdir=listdir)

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=FakeVolume))
    assert tool._modal_asset_is_reusable(plan, local_size=17) is True
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["expected_sha256"] == resume_sha256
    assert receipt["checkpoint_content_validated"] is False
    assert "--preflight-first" in plan.dispatch_argv
    overrides = plan.dispatch_argv[plan.dispatch_argv.index("--env-overrides") + 1]
    assert f"WITNESS_RESUME_SHA256={resume_sha256}" in overrides


@pytest.mark.parametrize("destination_exists", (False, True))
def test_modal_cross_label_resume_requires_fresh_destination(
    monkeypatch, destination_exists
):
    tool = _load_launcher_tool()
    digest = TASK438_GT_CACHE_SHA256
    remote_asset = f"assets/v9_cgauge/gt_{digest}.npz"
    resume_path = "source-label/output/checkpoint.pt"
    plan = build_plan(
        **_kwargs(
            label="new-label",
            resume_from=f"/modal_results/{resume_path}",
            resume_sha256="d" * 64,
        )
    )

    class FakeVolume:
        @staticmethod
        def from_name(_name, *, create_if_missing):
            assert create_if_missing is False

            def listdir(path):
                if path == "new-label":
                    return (
                        [SimpleNamespace(path="new-label/output", type=2, size=0)]
                        if destination_exists
                        else []
                    )
                if path == resume_path:
                    return [SimpleNamespace(path=path, type=1, size=1024)]
                assert path == remote_asset
                return [SimpleNamespace(path=path, type=1, size=17)]

            return SimpleNamespace(listdir=listdir)

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=FakeVolume))
    if destination_exists:
        with pytest.raises(SystemExit, match="same output lineage"):
            tool._modal_asset_is_reusable(plan, local_size=17)
    else:
        assert tool._modal_asset_is_reusable(plan, local_size=17) is True


def test_modal_state_lookup_rejects_existing_nonresume_output(monkeypatch):
    tool = _load_launcher_tool()
    digest = "a" * 64
    remote_path = f"assets/v9_cgauge/gt_{digest}.npz"
    plan = SimpleNamespace(
        gt_cache_sha256=digest,
        label="unit-cuda",
        resume_from=None,
        asset_stage_argv=(".venv/bin/modal", "volume", "put", "volume", "local", remote_path),
    )

    class FakeVolume:
        @staticmethod
        def from_name(_name, *, create_if_missing):
            assert create_if_missing is False

            def listdir(path):
                assert path == "unit-cuda"
                return [SimpleNamespace(path="unit-cuda/output", type=2, size=0)]

            return SimpleNamespace(listdir=listdir)

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=FakeVolume))
    with pytest.raises(SystemExit, match="already has remote output custody"):
        tool._modal_asset_is_reusable(plan, local_size=17)


def test_modal_asset_reuse_lookup_error_refuses_blind_upload(monkeypatch):
    tool = _load_launcher_tool()
    digest = "a" * 64
    remote_path = f"assets/v9_cgauge/gt_{digest}.npz"
    plan = SimpleNamespace(
        gt_cache_sha256=digest,
        label="unit-cuda",
        resume_from=None,
        asset_stage_argv=(".venv/bin/modal", "volume", "put", "volume", "local", remote_path),
    )

    class FailingVolume:
        @staticmethod
        def from_name(*_args, **_kwargs):
            raise RuntimeError("offline")

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=FailingVolume))
    with pytest.raises(SystemExit, match="read-only Modal Volume asset lookup failed"):
        tool._modal_asset_is_reusable(plan, local_size=17)


def test_modal_asset_reuse_fresh_label_notfound_is_not_refused(monkeypatch):
    # Regression (2026-07-12): a fresh (non-resume) label has no output dir yet,
    # so volume.listdir(label) raises modal NotFoundError. That is the NORMAL
    # no-collision state and must NOT be mislabeled as the "refusing blind
    # upload" money-safety refusal. The staged SHA-asset is present + size-match
    # => reusable True (skip staging), not a SystemExit.
    tool = _load_launcher_tool()
    digest = "a" * 64
    remote_path = f"assets/v9_cgauge/gt_{digest}.npz"
    plan = SimpleNamespace(
        gt_cache_sha256=digest,
        label="fresh-label",
        resume_from=None,
        asset_stage_argv=(".venv/bin/modal", "volume", "put", "volume", "local", remote_path),
    )

    class NotFoundError(Exception):
        pass

    class FreshVolume:
        @staticmethod
        def from_name(_name, *, create_if_missing):
            assert create_if_missing is False

            def listdir(path):
                if path == "fresh-label":
                    raise NotFoundError("No such file or directory")
                return [SimpleNamespace(path=remote_path, type=1, size=17)]

            return SimpleNamespace(listdir=listdir)

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=FreshVolume))
    assert tool._modal_asset_is_reusable(plan, local_size=17) is True


@pytest.mark.parametrize("asset_reusable", (True, False))
def test_execute_requires_reusable_asset_and_never_implicitly_stages(
    monkeypatch, tmp_path, asset_reusable: bool
):
    tool = _load_launcher_tool()
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"exact-gt-cache")
    segnet = tmp_path / "upstream/models/segnet.safetensors"
    posenet = tmp_path / "upstream/models/posenet.safetensors"
    segnet.parent.mkdir(parents=True)
    segnet.write_bytes(b"reviewed-segnet")
    posenet.write_bytes(b"reviewed-posenet")
    plan = replace(
        build_plan(
            **_kwargs(
                gt_cache="cache.npz",
                gt_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                gt_cache_bytes=cache.stat().st_size,
                segnet_sha256=hashlib.sha256(segnet.read_bytes()).hexdigest(),
                posenet_sha256=hashlib.sha256(posenet.read_bytes()).hexdigest(),
            )
        ),
        execution_allowed=True,
        plan_sha256="f" * 64,
    )
    calls: list[tuple[str, ...]] = []
    call_kwargs: list[dict[str, object]] = []
    events: list[str] = []

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        call_kwargs.append(dict(_kwargs))
        events.append("asset_stage" if tuple(command) == plan.asset_stage_argv else "dispatch")
        return SimpleNamespace(returncode=0)

    def clean(_head):
        events.append("clean")

    def no_active(_plan):
        events.append("claim_check")

    def local_preflight(_plan, *, local_cache):
        assert local_cache == cache
        events.append("local_preflight")

    def app_preflight():
        events.append("app_preflight")

    def modal_lookup(_plan, *, local_size):
        assert local_size == cache.stat().st_size
        events.append("modal_lookup")
        return asset_reusable

    @contextmanager
    def outer_guard(_plan):
        events.append("outer_lock_enter")
        try:
            yield tmp_path / "outer.lock"
        finally:
            events.append("outer_lock_exit")

    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "_main_git_head", lambda: "a" * 40)
    monkeypatch.setattr(tool, "_require_clean_main_worktree", clean)
    monkeypatch.setattr(tool, "build_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(tool, "_require_no_active_v9_claim", no_active)
    monkeypatch.setattr(tool, "_run_local_modal_app_definition_preflight", app_preflight)
    monkeypatch.setattr(tool, "_run_local_v9_preflight", local_preflight)
    monkeypatch.setattr(tool, "_modal_asset_is_reusable", modal_lookup)
    monkeypatch.setattr(tool, "_outer_v9_lane_dispatch_guard", outer_guard)
    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    argv = [
        "--execute",
        "--operator-go-token",
        tool.OPERATOR_TOKEN,
        "--expected-plan-sha256",
        plan.plan_sha256,
    ]
    if asset_reusable:
        assert tool.main(argv) == 0
    else:
        with pytest.raises(SystemExit, match="Automatic `modal volume put` is disabled"):
            tool.main(argv)
    expected_calls = [plan.dispatch_argv] if asset_reusable else []
    assert calls == expected_calls
    if asset_reusable:
        assert call_kwargs[-1]["timeout"] == plan.invocation_timeout_seconds
    assert plan.asset_stage_argv not in calls
    expected_events = [
        "clean",
        "clean",
        "outer_lock_enter",
        "claim_check",
        "app_preflight",
        "local_preflight",
        "claim_check",
        "modal_lookup",
    ]
    if asset_reusable:
        expected_events.extend(("claim_check", "clean", "dispatch"))
    expected_events.append("outer_lock_exit")
    assert events == expected_events


def test_remote_driver_only_emits_existing_timing_trainer_flags():
    repo = Path(__file__).resolve().parents[3]
    driver = (repo / "scripts/remote_v9_cgauge_cuda.sh").read_text(encoding="utf-8")
    trainer = (repo / "experiments/train_levelset_witness_realized_through_R_torch.py").read_text(
        encoding="utf-8"
    )
    for flag in (
        "--gt-cache", "--num-pairs", "--epochs", "--out-dir", "--device",
        "--compile-probe", "--stop-after-epochs", "--no-implicit-resume", "--resume-from",
        "--expected-segnet-sha256", "--expected-posenet-sha256",
    ):
        assert flag in driver
        assert flag in trainer
    assert 'RESUME_SHA256="${WITNESS_RESUME_SHA256:-}"' in driver
    assert "WITNESS_RESUME_SHA256 is forbidden without WITNESS_RESUME_FROM" in driver
    assert '[ ! -f "$RESUME_FROM" ] || [ ! -s "$RESUME_FROM" ]' in driver
    assert '[[ "$RESUME_FROM" != "$OUT_DIR/"* ]]' in driver
    assert "requires resume checkpoint under the same output lineage" in driver
    assert 'validate_file_sha256 "$RESUME_FROM" "$RESUME_SHA256" "resume checkpoint"' in driver


def test_remote_preflight_runs_before_output_or_cuda_mutation():
    repo = Path(__file__).resolve().parents[3]
    driver = (repo / "scripts/remote_v9_cgauge_cuda.sh").read_text(encoding="utf-8")
    trainer = (repo / "experiments/train_levelset_witness_realized_through_R_torch.py").read_text(
        encoding="utf-8"
    )
    preflight = driver.index('if [ "$PREFLIGHT_ONLY" = "1" ]; then')
    gpu_branch = driver.index("# The CPU preflight normally caught this")
    mkdir = driver.index('if ! mkdir "$OUT_DIR"')
    cuda_probe = driver.index('import torch')
    timeout_gate = driver.index("if ! command -v timeout", gpu_branch)
    assert "validate_gt_cache" in driver
    assert '"${TRAINER_ARGS[@]}" --preflight-only' in driver
    assert "v9_cgauge_torch_preflight.v1" in trainer
    assert "validate_gt_cache" in driver[preflight:mkdir]
    assert "validate_scorer_custody" in driver[preflight:mkdir]
    assert "validate_resume_custody" in driver[preflight:mkdir]
    assert "storage_preflight" in driver[preflight:mkdir]
    assert "witness_remote_timeout_contract.v1" in driver
    assert "--kill-after=" in driver
    assert "env-override delimiters" in driver
    assert preflight < gpu_branch < timeout_gate < cuda_probe < mkdir
    gpu_storage = driver.index("storage_preflight", gpu_branch)
    gpu_gt_hash = driver.index('GT_CUSTODY_RECEIPT="$(validate_gt_cache)"', gpu_branch)
    assert gpu_storage < gpu_gt_hash
    gpu_scorer = driver.index("validate_scorer_custody", gpu_branch)
    gpu_resume = driver.index("validate_resume_custody", gpu_branch)
    assert gpu_gt_hash < gpu_scorer < gpu_resume < mkdir
    assert "fresh WITNESS_OUT_DIR collided after validation" in driver
    gpu_segment = driver[gpu_branch:]
    assert gpu_segment.count("validate_gt_cache") == 1
    assert "with source.open" not in gpu_segment
    assert '"$GT_CUSTODY_RECEIPT" "$GT_CACHE" "$GT_CACHE_SHA256"' in gpu_segment


def test_scaffold_providers_refuse_execution_without_invented_actuator():
    for provider in ("aws", "gcp"):
        plan = build_plan(**_kwargs(provider=provider))
        assert plan.status == "scaffold"
        assert not plan.execution_allowed
        assert plan.asset_stage_argv == plan.dispatch_argv == plan.harvest_argv == ()
