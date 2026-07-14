from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from tac.witness_dsl.margin_adaptive_mixed_precision_20260714 import (
    MarginAdaptiveMixedPrecisionPolicy,
    compile_margin_adaptive_probe_argv,
    validate_bound_paths,
)


def test_policy_compiles_exact_resumable_n600_host_argv() -> None:
    policy = MarginAdaptiveMixedPrecisionPolicy()
    argv = compile_margin_adaptive_probe_argv(policy)

    assert argv[:2] == (
        ".venv/bin/python",
        "tools/probe_margin_adaptive_mixed_precision_n600.py",
    )
    assert argv[argv.index("--pair-stop") + 1] == "600"
    assert argv[argv.index("--n-processes") + 1] == "10"
    assert "--resume" in argv
    assert argv[argv.index("--profile-caps") + 1].endswith("29,30,31")
    assert policy.to_dict()["score_claim"] is False
    assert policy.to_dict()["pointer_moved"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("pair_stop", 599),
        ("design_stop", 263),
        ("profile_caps", (7, 31)),
        ("n_processes", 9),
        ("accumulation", "fp32"),
        ("weight_scale_granularity", "global"),
        ("integer_operand_storage_buckets", (32,)),
        ("spatial_waterfill_native_execution", True),
        ("resume", False),
        ("score_claim", True),
    ),
)
def test_policy_refuses_false_authority_or_weakened_custody(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(MarginAdaptiveMixedPrecisionPolicy(), **{field: value})


def test_bound_path_validation_is_fail_closed(tmp_path: Path) -> None:
    policy = MarginAdaptiveMixedPrecisionPolicy()
    with pytest.raises(FileNotFoundError):
        validate_bound_paths(policy, repo=tmp_path)


def test_host_command_is_bound_to_typed_compiler_inputs() -> None:
    repo = Path(__file__).resolve().parents[4]
    host = repo / "tools/run_margin_adaptive_mixed_precision_n600_host.command"
    command = host.read_text(encoding="utf-8")
    argv = compile_margin_adaptive_probe_argv()

    # OUT is a shell variable so its compiled relative value is intentionally
    # not duplicated. Every immutable input and treatment token must agree.
    output_index = argv.index("--output")
    for token in argv[1:output_index]:
        assert token in command
    assert 'exec .venv/bin/python tools/probe_margin_adaptive_mixed_precision_n600.py' in command
    assert ' --output "$OUT"' in command
    assert os.stat(host).st_mode & 0o111
