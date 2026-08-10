# SPDX-License-Identifier: MIT
"""The paired dispatcher must FORWARD ``auto``, not resolve it.

Measured 2026-08-10: resolving ``--expected-runtime-tree-sha256 auto`` into a
concrete per-axis tree hash turned a working flag into a guaranteed CPU-axis
refusal. ``experiments/modal_auth_eval_cpu.py:372`` accepts only
``{"", "auto", <files-digest>}`` on the Modal-uploaded --submission-dir axis
(the r9m deadlock: projected and remote tree hashes cannot agree in general).
The concrete projection is still the anchor-lookup key and the plan's
informational record -- those two uses must not be conflated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "dispatch_modal_paired_auth_eval.py"


@pytest.fixture(scope="module")
def dispatcher():
    spec = importlib.util.spec_from_file_location("_paired_dispatch_tool", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_hashes(dispatcher, monkeypatch, cuda: str, cpu: str) -> None:
    """Stand in for the real local projection (needs a materialized tree)."""

    def _fake(*, submission_dir, inflate_sh_rel, remote_submission_dir):
        is_cuda = remote_submission_dir == dispatcher.CUDA_REMOTE_SUBMISSION_DIR
        return {
            "runtime_tree_sha256": cuda if is_cuda else cpu,
            "runtime_content_tree_sha256": "cc" * 32,
        }

    monkeypatch.setattr(dispatcher, "_modal_uploaded_runtime_hashes_for_axis", _fake)


@pytest.mark.parametrize("requested", ["auto", "", None])
def test_auto_is_forwarded_verbatim_not_resolved(dispatcher, monkeypatch, requested):
    cuda_hash, cpu_hash = "aa" * 32, "bb" * 32
    _stub_hashes(dispatcher, monkeypatch, cuda_hash, cpu_hash)

    out = dispatcher._resolve_axis_runtime_expectations(
        submission_dir="/tmp/does-not-need-to-exist",
        inflate_sh_rel="inflate.sh",
        expected_runtime_tree_sha256=requested,
        expected_cuda_runtime_tree_sha256="",
        expected_cpu_runtime_tree_sha256="",
    )

    # What the WRAPPERS receive: the literal 'auto' on BOTH axes.
    assert out["forwarded_contest_cuda"] == dispatcher.AUTO_RUNTIME_TREE
    assert out["forwarded_contest_cpu"] == dispatcher.AUTO_RUNTIME_TREE
    # What the ANCHOR LOOKUP and the plan record receive: the concrete
    # projection, unchanged. Losing this would silently disable anchor reuse.
    assert out["contest_cuda"] == cuda_hash
    assert out["contest_cpu"] == cpu_hash


def test_explicit_operator_hash_is_forwarded_verbatim(dispatcher, monkeypatch):
    """An EXPLICIT hash is a real expectation; a wrapper refusing it is a real
    disagreement, so it must reach the wrapper unmodified."""

    pinned = "aa" * 32
    _stub_hashes(dispatcher, monkeypatch, pinned, pinned)

    out = dispatcher._resolve_axis_runtime_expectations(
        submission_dir="/tmp/does-not-need-to-exist",
        inflate_sh_rel="inflate.sh",
        expected_runtime_tree_sha256=pinned,
        expected_cuda_runtime_tree_sha256="",
        expected_cpu_runtime_tree_sha256="",
    )

    assert out["forwarded_contest_cuda"] == pinned
    assert out["forwarded_contest_cpu"] == pinned
    assert dispatcher.AUTO_RUNTIME_TREE not in (
        out["forwarded_contest_cuda"],
        out["forwarded_contest_cpu"],
    )
