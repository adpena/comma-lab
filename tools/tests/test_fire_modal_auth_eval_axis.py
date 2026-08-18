# SPDX-License-Identifier: MIT
"""Self-protection for the fire tool's CPU axis (task #1105, landing L1).

The canonical fire path was CUDA-only, so pq1's sealed contest-CPU fire-order could
not execute through it and would have had to be hand-assembled — the exact hazard
the hand-assembled-dispatch law names an error factory. These tests hold the cure in
place: the axis selector picks a REAL entrypoint, emits only flags that entrypoint
actually declares, keeps every custody field the T4 path carries, watches a CPU call
longer than the CPU worker can live, and refuses to write an axis-untagged row.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import fire_modal_auth_eval as fire  # noqa: E402

# Flags the dispatch template never emits, because only the CUDA entrypoint declares
# them and --inflate-env would demote the row to a diagnostic axis anyway.
CUDA_ONLY_PARAMS = {"gpu", "scorer_device", "inflate_device", "inflate_env"}


def _entrypoint_params(module_rel: str) -> set[str]:
    """Read the REAL local_entrypoint signature — argparse truth, not memory."""

    tree = ast.parse((REPO / module_rel).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return {a.arg for a in node.args.args + node.args.kwonlyargs}
    raise AssertionError(f"no main() local_entrypoint found in {module_rel}")


def _argv(axis: str, **overrides) -> list[str]:
    kwargs = {
        "spec": fire.axis_spec(axis),
        "archive": Path("/tmp/rt/archive.zip"),
        "runtime_dir": Path("/tmp/rt"),
        "archive_sha": "a" * 64,
        "out_dir": Path("/tmp/out"),
        "lane_id": "lane_x",
        "instance_job_id": "job_x",
        "claim_agent": "MAIN",
    }
    kwargs.update(overrides)
    return fire.build_dispatch_argv(**kwargs)


def _flags(argv: list[str]) -> set[str]:
    return {tok for tok in argv if tok.startswith("--")}


# --------------------------------------------------------------------------- axes


def test_axis_spec_resolves_both_contest_axes():
    assert fire.axis_spec("cuda")["entrypoint"] == "experiments/modal_auth_eval.py::main"
    assert fire.axis_spec("cpu")["entrypoint"] == "experiments/modal_auth_eval_cpu.py::main"
    assert fire.axis_spec("cuda")["evidence_axis_tag"] == "[contest-CUDA]"
    assert fire.axis_spec("cpu")["evidence_axis_tag"] == "[contest-CPU]"
    assert fire.axis_spec("CPU")["axis"] == "cpu"  # case-insensitive


@pytest.mark.parametrize("bad", ["", "  ", "mps", "gpu", "t4", None])
def test_axis_spec_refuses_unknown_axis(bad):
    """Fail-closed: an unrecognised axis is never silently treated as CUDA."""

    with pytest.raises(ValueError, match="unknown auth-eval axis"):
        fire.axis_spec(bad)


def test_both_entrypoints_exist_on_disk():
    for spec in fire.AXES.values():
        module_rel = spec["entrypoint"].split("::")[0]
        assert (REPO / module_rel).is_file(), f"missing worker module {module_rel}"


# ------------------------------------------------------------------- flag custody


@pytest.mark.parametrize("axis", sorted(fire.AXES))
def test_every_emitted_flag_is_declared_by_its_entrypoint(axis):
    """never-invent-flags, EXECUTED against the real signature.

    pq1 verified this by hand ("read the local_entrypoint signature ... and confirmed
    every flag above maps to a declared parameter"). Hand-verification is exactly what
    this landing determinizes.
    """

    argv = _argv(
        axis,
        single_axis_waiver_reason="r",
        pair_group_id="",
        claim_policy="require_active",
    )
    declared = _entrypoint_params(fire.axis_spec(axis)["entrypoint"].split("::")[0])
    emitted = {f.removeprefix("--").replace("-", "_") for f in _flags(argv)}
    # --detach is both a modal CLI flag and a wrapper parameter; both are declared.
    undeclared = emitted - declared
    assert not undeclared, f"{axis} emits flags the entrypoint does not declare: {sorted(undeclared)}"


def test_cpu_and_cuda_carry_identical_custody_fields():
    """The CPU route must not be a weaker custody path than the T4 route."""

    common = {"single_axis_waiver_reason": "w", "claim_policy": "require_active"}
    cuda, cpu = _argv("cuda", **common), _argv("cpu", **common)
    assert _flags(cuda) == _flags(cpu)
    # ... and identical values everywhere except the entrypoint itself.
    assert cuda[3] == "experiments/modal_auth_eval.py::main"
    assert cpu[3] == "experiments/modal_auth_eval_cpu.py::main"
    assert cuda[:3] == cpu[:3] and cuda[4:] == cpu[4:]


def test_cuda_template_is_unchanged_regression():
    """The proven rr4/fx1 T4 invocation must not drift while adding the CPU axis."""

    argv = _argv("cuda")
    assert argv[:4] == [fire.MODAL_BIN, "run", "--detach", "experiments/modal_auth_eval.py::main"]
    for flag, value in (
        ("--archive", "/tmp/rt/archive.zip"),
        ("--submission-dir", "/tmp/rt"),
        ("--inflate-sh", "inflate.sh"),
        ("--expected-archive-sha256", "a" * 64),
        ("--expected-runtime-tree-sha256", "auto"),  # failure F3, pinned always
        ("--output-dir", "/tmp/out"),
        ("--lane-id", "lane_x"),
        ("--instance-job-id", "job_x"),
        ("--claim-agent", "MAIN"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert "--detach" in argv and "--provider-detach-ack" in argv


@pytest.mark.parametrize("axis", sorted(fire.AXES))
def test_inflate_env_is_never_emitted_on_either_axis(axis):
    """--inflate-env demotes a row to a diagnostic axis; the contest path never sets it."""

    assert "--inflate-env" not in _argv(axis)


@pytest.mark.parametrize("axis", sorted(fire.AXES))
def test_runtime_tree_sha_is_pinned_auto(axis):
    """Both workers REFUSE any tree-sha but ''/'auto'/the FILES digest (r9m deadlock).

    pq1's hand-written fire-order pinned the runtime_tree_sha256 instead, which the CPU
    worker refuses. The template pins 'auto' so that class of fire-order cannot recur.
    """

    argv = _argv(axis)
    assert argv[argv.index("--expected-runtime-tree-sha256") + 1] == "auto"


def test_cuda_only_params_are_absent_from_the_cpu_entrypoint():
    """Guards the premise behind flag parity: the CPU worker really lacks these."""

    cpu = _entrypoint_params("experiments/modal_auth_eval_cpu.py")
    cuda = _entrypoint_params("experiments/modal_auth_eval.py")
    assert cuda >= CUDA_ONLY_PARAMS
    assert not (CUDA_ONLY_PARAMS & cpu)


# --------------------------------------------------------- paired-axis gate + watch


@pytest.mark.parametrize("axis", sorted(fire.AXES))
def test_paired_axis_gate_flags_forward_on_both_axes(axis):
    """Single-axis waiver / pair-group semantics are preserved verbatim on CPU."""

    waived = _argv(axis, single_axis_waiver_reason="pair completed by the CUDA row")
    assert waived[waived.index("--single-axis-waiver-reason") + 1] == (
        "pair completed by the CUDA row"
    )
    paired = _argv(axis, pair_group_id="pg-1")
    assert paired[paired.index("--pair-group-id") + 1] == "pg-1"
    # Neither is emitted when unset: the worker then refuses, which is the gate.
    bare = _argv(axis)
    assert "--single-axis-waiver-reason" not in bare and "--pair-group-id" not in bare


def test_cpu_poller_deadline_outlives_the_cpu_worker_timeout():
    """Failure F5 was an unwatched paid call. The CUDA default would abandon a CPU row."""

    worker_timeout_s = 9000  # @app.function(..., timeout=9000) in modal_auth_eval_cpu.py
    src = (REPO / "experiments" / "modal_auth_eval_cpu.py").read_text()
    assert f"timeout={worker_timeout_s}" in src, "CPU worker timeout moved — retune the deadline"
    assert fire.AXES["cpu"]["poller_deadline_s"] > worker_timeout_s
    assert fire.AXES["cpu"]["poller_deadline_s"] > fire.AXES["cuda"]["poller_deadline_s"]


# ------------------------------------------------ axis-tagged manifest (fail-closed)


def _manifest(**overrides) -> dict:
    base = {
        "schema": "fire_modal_auth_eval.v2",
        "axis": "cpu",
        "evidence_axis_tag": "[contest-CPU]",
        "score_axis": "contest_cpu",
        "stage5_entrypoint": "experiments/modal_auth_eval_cpu.py::main",
    }
    base.update(overrides)
    return base


def test_write_fire_manifest_accepts_a_tagged_row(tmp_path):
    path = fire.write_fire_manifest(tmp_path / "out", _manifest())
    assert path.is_file()
    import json

    assert json.loads(path.read_text())["evidence_axis_tag"] == "[contest-CPU]"


@pytest.mark.parametrize("missing", ["axis", "evidence_axis_tag", "stage5_entrypoint"])
def test_write_fire_manifest_refuses_an_untagged_row(tmp_path, missing):
    """The other direction: a row that cannot be read as CPU or CUDA evidence is refused.

    CPU and CUDA are separate evidence spaces and neither may be inferred from the
    other, so an axis-less manifest is not weak evidence — it is none.
    """

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="axis-untagged"):
        fire.write_fire_manifest(out, _manifest(**{missing: ""}))
    assert not (out / "FIRE_MANIFEST.json").exists(), "refused row must leave no artifact"


def test_write_fire_manifest_refuses_an_unknown_tag(tmp_path):
    with pytest.raises(ValueError, match="unknown evidence axis tag"):
        fire.write_fire_manifest(tmp_path / "out", _manifest(evidence_axis_tag="[macOS-CPU advisory]"))


# ------------------------------------------------------- dry run must not mutate


def test_dry_run_reports_litter_without_deleting_it(tmp_path):
    """A dry run rehearses a SEALED tree; deleting a file would change its FILES digest."""

    (tmp_path / "._AppleDouble").write_bytes(b"x")
    (tmp_path / "inflate.sh").write_bytes(b"#!/bin/sh\n")

    reported = fire.sanitize_litter(tmp_path, apply=False)
    assert reported == ["._AppleDouble"]
    assert (tmp_path / "._AppleDouble").is_file(), "dry run must not mutate the tree"

    # ... and the real path still removes it (the other direction).
    removed = fire.sanitize_litter(tmp_path, apply=True)
    assert removed == ["._AppleDouble"]
    assert not (tmp_path / "._AppleDouble").exists()


# --------------------------------------------- refusals must land on disk, not only rc


def test_refuse_writes_a_disk_receipt_a_pipe_cannot_swallow(tmp_path):
    """t1h r1 returned rc=5 and a `| tail` ate the exit code, so a refused fire read clean.

    The exit code alone is not evidence. A refusal must leave a file behind.
    """

    out = tmp_path / "out"
    rc = fire.refuse(out, 5, "dispatch produced no spawn record", {"axis": "cpu"})

    assert rc == 5
    receipt = json.loads((out / "FIRE_REFUSED.json").read_text())
    assert receipt["refused"] is True
    assert receipt["refusal_rc"] == 5
    assert receipt["refusal_reason"] == "dispatch produced no spawn record"
    # A refusal is NOT a fire: it must never masquerade as a manifest.
    assert not (out / "FIRE_MANIFEST.json").exists()


def test_refuse_survives_an_unserializable_manifest(tmp_path, capsys):
    """Bookkeeping must never mask the refusal it is recording."""

    class Unserializable:
        pass

    rc = fire.refuse(tmp_path / "out", 3, "r", {"axis": "cpu", "odd": Unserializable()})

    assert rc == 3  # the refusal still propagates
    assert "REFUSED (rc=3)" in capsys.readouterr().out


def test_a_successful_manifest_clears_a_stale_refusal_receipt(tmp_path):
    """Otherwise a fire that DID take reads as refused, which is the inverse of r1."""

    out = tmp_path / "out"
    fire.refuse(out, 5, "earlier attempt", {"axis": "cpu"})
    assert (out / "FIRE_REFUSED.json").exists()

    fire.write_fire_manifest(out, _manifest())
    assert (out / "FIRE_MANIFEST.json").exists()
    assert not (out / "FIRE_REFUSED.json").exists()


def test_refuse_does_not_mutate_the_caller_manifest():
    manifest = {"axis": "cpu"}
    fire.refuse(Path("/nonexistent-root-xyz/out"), 2, "r", manifest)
    assert manifest == {"axis": "cpu"}


def test_validate_tree_skips_litter_so_both_modes_see_one_tree(tmp_path):
    """Without the skip set a dry run would refuse on litter the real path removes."""

    (tmp_path / "._AppleDouble").write_bytes(b"x")
    (tmp_path / "inflate.sh").write_bytes(b"#!/bin/sh\n")
    litter = fire.sanitize_litter(tmp_path, apply=False)

    assert fire.validate_tree(tmp_path) != [], "premise: the validator does refuse litter"
    assert fire.validate_tree(tmp_path, skip=frozenset(litter)) == []
