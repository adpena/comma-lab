# SPDX-License-Identifier: MIT
"""The local preflight must reproduce the refusals it is meant to move earlier.

Every positive control here is a refusal that ACTUALLY FIRED on 2026-08-10 after
a ~5-minute Modal image build. A guard that cannot reproduce them is theater, so
each is asserted by its typed refusal kind, not merely by non-emptiness.

The negative control is equally load-bearing: the real forwarded command must
pass CLEAN. An early version of this module resolved the entrypoint signature
from Modal's callable wrapper -- which is varargs-only -- and consequently
reported EVERY real flag as unknown. The controls caught it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tac.deploy.modal.paired_dispatch import paired_auth_eval_axis_command
from tac.deploy.modal.paired_dispatch_preflight import (
    parse_axis_command,
    preflight_axis_command,
    preflight_axis_commands,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = "submissions/robust_current"


def _axis_command(axis: str, *, tree_sha: str = "auto", submission_dir: str | None = SUBMISSION_DIR):
    return paired_auth_eval_axis_command(
        axis=axis,
        modal_bin=".venv/bin/modal",
        archive_path=f"{SUBMISSION_DIR}/archive.zip",
        archive_sha256="0" * 64,
        inflate_sh="inflate.sh",
        output_dir="experiments/results/preflight_test_out",
        pair_group_id="pair_test",
        lane_id="lane_test",
        instance_job_id="job_test",
        claim_agent="test",
        claim_notes="preflight regression",
        submission_dir=submission_dir,
        expected_runtime_tree_sha256=tree_sha,
    )


def _kinds(refusals) -> set[str]:
    return {r.kind for r in refusals}


@pytest.mark.parametrize("axis", ["contest_cuda", "contest_cpu"])
def test_real_forwarded_command_preflights_clean(axis: str) -> None:
    """NEGATIVE CONTROL: the command the dispatcher actually emits must pass.

    Without this, a signature-resolution bug turns the guard into a
    false-refusal storm that blocks every legitimate dispatch.
    """

    # No --submission-dir: keeps the check to structure + flags, independent of
    # whichever submission tree happens to exist in the working copy.
    refusals = preflight_axis_command(_axis_command(axis, submission_dir=None))
    assert refusals == [], f"real {axis} command refused: {[str(r) for r in refusals]}"


@pytest.mark.parametrize("axis", ["contest_cuda", "contest_cpu"])
def test_every_forwarded_flag_resolves_against_the_entrypoint(axis: str) -> None:
    """The parse must bind real flags, not silently see zero parameters."""

    parse, refusals = parse_axis_command(_axis_command(axis, submission_dir=None))
    assert parse is not None
    assert refusals == []
    assert parse.unknown_tokens == ()
    # Spot-check that value flags carried their values and bool flags did not
    # consume the next token.
    assert parse.flags["inflate_sh"] == "inflate.sh"
    assert parse.flags["detach"] is True
    assert parse.flags["archive"].endswith("archive.zip")


def test_positive_control_bare_wrapper_target_is_refused() -> None:
    """Refusal #2, 2026-08-10: 'modal run <file>' with two local entrypoints."""

    command = _axis_command("contest_cuda", submission_dir=None)
    command[3] = command[3].split("::")[0]
    assert "ENTRYPOINT_SPEC_BARE" in _kinds(preflight_axis_command(command))


def test_positive_control_unknown_entrypoint_is_refused() -> None:
    command = _axis_command("contest_cpu", submission_dir=None)
    command[3] = command[3].split("::")[0] + "::not_an_entrypoint"
    assert "ENTRYPOINT_MISSING" in _kinds(preflight_axis_command(command))


def test_positive_control_invented_flag_is_refused() -> None:
    """The dead-flag class CLAUDE.md names: grep the signature before emitting."""

    command = _axis_command("contest_cuda", submission_dir=None)
    command.extend(["--auth-eval-masks", "yes"])
    refusals = preflight_axis_command(command)
    assert "UNKNOWN_FLAG" in _kinds(refusals)
    assert any("--auth-eval-masks" in str(r) for r in refusals)


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not (REPO_ROOT / SUBMISSION_DIR / "inflate.sh").is_file(),
    reason="needs a real submission tree to derive the runtime FILES digest",
)
def test_positive_control_concrete_tree_hash_is_refused_on_cpu() -> None:
    """Refusal #3, 2026-08-10: the r9m deadlock, previously paid for with a build.

    The CPU wrapper structurally refuses a concrete tree hash on the uploaded
    --submission-dir axis. This asserts the LOCAL preflight reproduces that
    refusal from the wrapper's own validator.

    MEASURED COST (2026-08-10): this check is NOT sub-second. Deriving the
    runtime FILES digest walks the real submission tree and resolves the
    repo-local tac import scan; it exceeded the 60 s default pytest timeout.
    That is still ~5x cheaper than the image build it replaces, and in the
    WIRED dispatcher the same digest is already computed by build_plan --
    but the honest claim is "cheaper than a build", not "free".
    """

    command = _axis_command("contest_cpu", tree_sha="fc665bb2" + "a" * 56)
    kinds = _kinds(preflight_axis_command(command))
    assert "RUNTIME_TREE_EXPECTATION" in kinds


def test_preflight_of_both_axes_is_fast_enough_to_precede_dispatch() -> None:
    """Structural checks are cheap enough to ALWAYS run before dispatch.

    Scope is deliberate: this covers entrypoint resolution + flag binding, the
    checks that caught refusals #2 and the dead-flag class. The runtime-tree
    check on a real --submission-dir is separately expensive (see the timeout
    note on the r9m control) -- do not read this bound as covering it.
    """

    commands = {
        axis: _axis_command(axis, submission_dir=None)
        for axis in ("contest_cuda", "contest_cpu")
    }
    start = time.monotonic()
    result = preflight_axis_commands(commands)
    elapsed = time.monotonic() - start
    assert set(result) == {"contest_cuda", "contest_cpu"}
    # Generous bound: measured ~0.12 s cold for both wrappers. This asserts the
    # ORDER OF MAGNITUDE claim (seconds, not minutes) without being flaky.
    assert elapsed < 20.0, f"preflight took {elapsed:.2f}s; it must be cheap enough to always run"
