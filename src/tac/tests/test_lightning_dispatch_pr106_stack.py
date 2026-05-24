# SPDX-License-Identifier: MIT
"""Regression test for the Lightning $1.55 catastrophe (2026-05-05).

Eight Lightning dispatches failed at the first ``cd`` because
``tools/lightning_dispatch_pr106_stack.py`` passed ``str(REPO_ROOT)`` (an
operator-local path) as ``--repo-dir`` to the launcher. The wrapper has been
fixed (commits 5156a9ea + 92a47a06) to pass ``args.remote_pact`` and
repo-relative ``--archive`` / ``--inflate-sh``. This test pins that fix.

The test exercises ``submit_dispatch`` with ``print_only=True`` so no live
Lightning SDK / SSH activity occurs.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DISPATCHER_PATH = REPO_ROOT / "tools" / "lightning_dispatch_pr106_stack.py"


def _load_dispatcher():
    spec = importlib.util.spec_from_file_location(
        "lightning_dispatch_pr106_stack_test", DISPATCHER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture_invocation(*, archive: Path, inflate_sh: Path, remote_pact: str) -> list[str]:
    """Run submit_dispatch in print_only mode, return the resolved cmd list."""
    dispatcher = _load_dispatcher()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = dispatcher.submit_dispatch(
            lane="apogee_int4",
            job_name="test_job",
            archive=archive,
            manifest=REPO_ROOT / "experiments" / "results" / "lightning_batch" / "test_job" / "source_manifest.json",
            inflate_sh=inflate_sh,
            predicted_low=0.155,
            predicted_high=0.180,
            ssh_target=dispatcher.DEFAULT_SSH_TARGET,
            machine="g4dn.2xlarge",
            print_only=True,
            remote_pact=remote_pact,
        )
    assert rc == 0
    output = buf.getvalue()
    # The print_only branch emits backslash-continued lines. Reconstruct the
    # cmd list by splitting on the same separator the dispatcher uses.
    body = output.split("=== resolved invocation (would run) ===\n", 1)[1]
    tokens = [tok.strip() for tok in body.replace("\\\n", " ").split() if tok.strip()]
    return tokens


def _flag_value(tokens: list[str], flag: str) -> str:
    """Return the value following ``flag`` (single-value flags only)."""
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            return tokens[i + 1]
    raise AssertionError(f"flag not found: {flag}")


def _all_flag_values(tokens: list[str], flag: str) -> list[str]:
    return [tokens[i + 1] for i, tok in enumerate(tokens) if tok == flag and i + 1 < len(tokens)]


def test_submit_dispatch_repo_dir_is_remote_path() -> None:
    """``--repo-dir`` MUST be the remote Lightning path, not the operator's mac."""
    archive = REPO_ROOT / "submissions" / "exact_current" / "archive.zip"
    inflate_sh = REPO_ROOT / "submissions" / "exact_current" / "inflate.sh"
    if not archive.is_file() or not inflate_sh.is_file():
        # Use any existing repo file as a stand-in — the test is about the
        # dispatcher's path-handling shape, not the contents.
        archive = REPO_ROOT / "pyproject.toml"
        inflate_sh = REPO_ROOT / "pyproject.toml"

    tokens = _capture_invocation(
        archive=archive,
        inflate_sh=inflate_sh,
        remote_pact="/teamspace/studios/this_studio/pact",
    )
    repo_dir = _flag_value(tokens, "--repo-dir")
    assert repo_dir == "/teamspace/studios/this_studio/pact", (
        f"--repo-dir leaked operator-local path: {repo_dir!r}"
    )
    assert not repo_dir.startswith("/Users/"), (
        f"--repo-dir leaked /Users/...: {repo_dir!r}"
    )


def test_submit_dispatch_upstream_dir_is_under_remote_pact() -> None:
    archive = REPO_ROOT / "pyproject.toml"
    inflate_sh = REPO_ROOT / "pyproject.toml"
    tokens = _capture_invocation(
        archive=archive,
        inflate_sh=inflate_sh,
        remote_pact="/teamspace/studios/this_studio/pact",
    )
    upstream = _flag_value(tokens, "--upstream-dir")
    assert upstream == "/teamspace/studios/this_studio/pact/upstream", (
        f"--upstream-dir leaked operator-local path: {upstream!r}"
    )


def test_submit_dispatch_archive_is_repo_relative() -> None:
    """``--archive`` must be relative; the launcher resolves it against ``--repo-dir``."""
    archive = REPO_ROOT / "pyproject.toml"
    inflate_sh = REPO_ROOT / "pyproject.toml"
    tokens = _capture_invocation(
        archive=archive,
        inflate_sh=inflate_sh,
        remote_pact="/teamspace/studios/this_studio/pact",
    )
    archive_value = _flag_value(tokens, "--archive")
    assert not archive_value.startswith("/"), (
        f"--archive must be repo-relative; got absolute path: {archive_value!r}"
    )
    assert "Users/adpena" not in archive_value
    assert archive_value == "pyproject.toml"


def test_submit_dispatch_inflate_sh_is_repo_relative() -> None:
    archive = REPO_ROOT / "pyproject.toml"
    inflate_sh = REPO_ROOT / "pyproject.toml"
    tokens = _capture_invocation(
        archive=archive,
        inflate_sh=inflate_sh,
        remote_pact="/teamspace/studios/this_studio/pact",
    )
    inflate_value = _flag_value(tokens, "--inflate-sh")
    assert not inflate_value.startswith("/"), (
        f"--inflate-sh must be repo-relative; got absolute path: {inflate_value!r}"
    )
    assert "Users/adpena" not in inflate_value


def test_submit_dispatch_honours_custom_remote_pact() -> None:
    """Passing a non-default --remote-pact propagates to all 3 path flags."""
    archive = REPO_ROOT / "pyproject.toml"
    inflate_sh = REPO_ROOT / "pyproject.toml"
    tokens = _capture_invocation(
        archive=archive,
        inflate_sh=inflate_sh,
        remote_pact="/workspace/pact",
    )
    assert _flag_value(tokens, "--repo-dir") == "/workspace/pact"
    assert _flag_value(tokens, "--upstream-dir") == "/workspace/pact/upstream"


def test_submit_dispatch_carries_inflate_torch_cu124_pin() -> None:
    """The cu13-vs-cu124 driver trap is a NON-NEGOTIABLE in CLAUDE.md."""
    archive = REPO_ROOT / "pyproject.toml"
    inflate_sh = REPO_ROOT / "pyproject.toml"
    tokens = _capture_invocation(
        archive=archive,
        inflate_sh=inflate_sh,
        remote_pact="/teamspace/studios/this_studio/pact",
    )
    env_values = _all_flag_values(tokens, "--env")
    assert any("INFLATE_TORCH_SPEC=torch==2.5.1+cu124" in v for v in env_values), (
        f"INFLATE_TORCH_SPEC pin missing or wrong; --env values: {env_values}"
    )


def test_apogee_dispatch_requires_distortion_gate(tmp_path: Path) -> None:
    dispatcher = _load_dispatcher()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"apogee bytes")

    with pytest.raises(SystemExit, match="apogee_intN dispatch requires"):
        dispatcher.validate_apogee_dispatch_gate(
            lane="apogee_int4",
            archive=archive,
            gate_json=None,
            allow_forensic_print_only=False,
            print_only=False,
        )


def test_apogee_forensic_override_only_allows_print_only(tmp_path: Path) -> None:
    dispatcher = _load_dispatcher()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"apogee bytes")

    dispatcher.validate_apogee_dispatch_gate(
        lane="apogee_int4",
        archive=archive,
        gate_json=None,
        allow_forensic_print_only=True,
        print_only=True,
    )
    with pytest.raises(SystemExit, match="requires --print-only"):
        dispatcher.validate_apogee_dispatch_gate(
            lane="apogee_int4",
            archive=archive,
            gate_json=None,
            allow_forensic_print_only=True,
            print_only=False,
        )


def test_apogee_distortion_gate_must_match_archive_sha(tmp_path: Path) -> None:
    dispatcher = _load_dispatcher()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"apogee bytes")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": "wrong",
                "evidence_semantics": "scorer_basin_parity_gate",
                "distortion_model_status": "passed",
                "scorer_basin_parity_status": "passed",
                "ready_for_exact_eval_dispatch": True,
            }
        )
    )

    with pytest.raises(SystemExit, match="candidate_archive_sha256 mismatch"):
        dispatcher.validate_apogee_dispatch_gate(
            lane="apogee_int4",
            archive=archive,
            gate_json=gate,
            allow_forensic_print_only=False,
            print_only=False,
        )


def test_apogee_distortion_gate_can_unlock_when_positive_and_sha_matched(tmp_path: Path) -> None:
    dispatcher = _load_dispatcher()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"apogee bytes")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": dispatcher._sha256_file(archive),
                "evidence_semantics": "scorer_basin_parity_gate",
                "scorer_basin_parity_status": "passed",
                "ready_for_exact_eval_dispatch": True,
            }
        )
    )

    dispatcher.validate_apogee_dispatch_gate(
        lane="apogee_int4",
        archive=archive,
        gate_json=gate,
        allow_forensic_print_only=False,
        print_only=False,
    )


def test_apogee_distortion_gate_rejects_proxy_only_semantics(tmp_path: Path) -> None:
    dispatcher = _load_dispatcher()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"apogee bytes")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "candidate_archive_sha256": dispatcher._sha256_file(archive),
                "evidence_semantics": "local_distortion_proxy",
                "distortion_model_status": "passed",
                "ready_for_exact_eval_dispatch": True,
            }
        )
    )

    with pytest.raises(SystemExit, match="unsupported evidence_semantics"):
        dispatcher.validate_apogee_dispatch_gate(
            lane="apogee_int4",
            archive=archive,
            gate_json=gate,
            allow_forensic_print_only=False,
            print_only=False,
        )


def test_main_print_only_has_no_stage_or_claim_side_effects(monkeypatch) -> None:
    dispatcher = _load_dispatcher()

    def _forbidden(*args, **kwargs):
        raise AssertionError("print-only must not stage or claim")

    submit_calls = []

    def _fake_submit_dispatch(**kwargs):
        submit_calls.append(kwargs)
        assert kwargs["print_only"] is True
        return 0

    monkeypatch.setattr(dispatcher, "stage_workspace", _forbidden)
    monkeypatch.setattr(dispatcher, "claim_lane", _forbidden)
    monkeypatch.setattr(dispatcher, "submit_dispatch", _fake_submit_dispatch)

    rc = dispatcher.main(
        [
            "--lane",
            "apogee_int4",
            "--archive",
            "pyproject.toml",
            "--inflate-sh",
            "pyproject.toml",
            "--predicted-low",
            "0.1",
            "--predicted-high",
            "0.2",
            "--print-only",
            "--allow-forensic-apogee-intN",
            "--job-name",
            "dry_print_only",
        ]
    )

    assert rc == 0
    assert len(submit_calls) == 1


def test_main_print_only_accepts_repo_relative_inflate_sh(capsys) -> None:
    dispatcher = _load_dispatcher()

    rc = dispatcher.main(
        [
            "--lane",
            "apogee_int4",
            "--archive",
            "pyproject.toml",
            "--inflate-sh",
            "pyproject.toml",
            "--predicted-low",
            "0.1",
            "--predicted-high",
            "0.2",
            "--print-only",
            "--allow-forensic-apogee-intN",
            "--job-name",
            "dry_print_only",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "=== resolved invocation (would run) ===" in output
    assert "--inflate-sh \\\n  pyproject.toml" in output
    assert "--archive \\\n  pyproject.toml" in output


def test_claim_lane_does_not_force_without_explicit_force_claim(monkeypatch) -> None:
    dispatcher = _load_dispatcher()
    calls = []

    def _fake_run(cmd, cwd, check):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dispatcher.subprocess, "run", _fake_run)

    dispatcher.claim_lane(
        lane="demo",
        job_name="job1",
        dispatch_lane_id=None,
        dispatch_claims_path=dispatcher.REPO_ROOT
        / ".omx/state/active_lane_dispatch_claims.md",
        force_claim=False,
        force_claim_reason=None,
    )
    assert "--force" not in calls[-1]

    dispatcher.claim_lane(
        lane="demo",
        job_name="job2",
        dispatch_lane_id=None,
        dispatch_claims_path=dispatcher.REPO_ROOT
        / ".omx/state/active_lane_dispatch_claims.md",
        force_claim=True,
        force_claim_reason="terminal stale claim",
    )
    assert "--force" in calls[-1]


def test_main_existing_claim_uses_exact_dispatch_lane_and_job(monkeypatch) -> None:
    dispatcher = _load_dispatcher()

    def _forbidden_claim(*args, **kwargs):
        raise AssertionError("existing-claim mode must not create a second claim")

    submit_calls = []

    def _fake_submit_dispatch(**kwargs):
        submit_calls.append(kwargs)
        assert kwargs["dispatch_lane_id"] == "fixture_lane"
        assert kwargs["job_name"] == "fixture_job"
        assert kwargs["dispatch_claims_path"].name == "claims.md"
        return 0

    monkeypatch.setattr(dispatcher, "stage_workspace", lambda **kwargs: REPO_ROOT / "pyproject.toml")
    monkeypatch.setattr(dispatcher, "claim_lane", _forbidden_claim)
    monkeypatch.setattr(dispatcher, "submit_dispatch", _fake_submit_dispatch)

    rc = dispatcher.main(
        [
            "--lane",
            "fixture_lane",
            "--archive",
            "pyproject.toml",
            "--inflate-sh",
            "pyproject.toml",
            "--predicted-low",
            "0.1",
            "--predicted-high",
            "0.2",
            "--ssh-target",
            "fixture-ssh",
            "--job-name",
            "fixture_job",
            "--dispatch-lane-id",
            "fixture_lane",
            "--dispatch-claims-path",
            str(REPO_ROOT / "claims.md"),
            "--use-existing-dispatch-claim",
        ]
    )

    assert rc == 0
    assert len(submit_calls) == 1
