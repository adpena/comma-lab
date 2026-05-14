# SPDX-License-Identifier: MIT
"""Tests for tools/xray_inflate_op_cost_profiler.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import xray_inflate_op_cost_profiler as xi  # noqa: E402


def test_classify_op_known_op():
    name, klass = xi._classify_op("F.interpolate")
    assert name == "F.interpolate"
    assert klass == "per-frame"


def test_classify_op_torch_unknown_defaults_cheap():
    name, klass = xi._classify_op("torch.something_obscure")
    assert klass == "cheap"


def test_classify_op_F_unknown_defaults_per_frame():
    name, klass = xi._classify_op("F.weird_op")
    assert klass == "per-frame"


def test_classify_op_tensor_method_per_channel_mutation():
    name, klass = xi._classify_op("up.sub_")
    assert name == "Tensor.sub_"
    assert klass == "per-channel-mutation"


def test_classify_op_tensor_method_known():
    name, klass = xi._classify_op("up.permute")
    assert name == "Tensor.permute"
    assert klass == "per-frame"


def test_classify_op_unknown_returns_unknown():
    name, klass = xi._classify_op("some.completely.unrelated.func")
    assert klass == "unknown"


def test_profile_inflate_py_simple_file(tmp_path):
    src = tmp_path / "inflate.py"
    src.write_text(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def inflate(x):\n"
        "    up = F.interpolate(x, size=(10, 10))\n"
        "    up.sub_(1.0)\n"
        "    return up.permute(0, 2, 3, 1)\n"
    )
    rep = xi.profile_inflate_py(src)
    qns = [o["qualified_name"] for o in rep["ops"]]
    assert "F.interpolate" in qns
    assert "up.sub_" in qns
    assert "up.permute" in qns


def test_profile_detects_per_channel_mutation(tmp_path):
    src = tmp_path / "inflate.py"
    src.write_text(
        "def f(up):\n"
        "    up[:, 0, 0].sub_(1.0)\n"
        "    up[:, 0, 2].sub_(1.0)\n"
        "    up[:, 1, 1].sub_(1.0)\n"
    )
    rep = xi.profile_inflate_py(src)
    assert rep["per_channel_mutation_count"] == 3


def test_profile_handles_pr101_canonical_inflate():
    pr101 = (
        REPO_ROOT
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/"
        "source/submissions/hnerv_ft_microcodec/inflate.py"
    )
    if not pr101.exists():
        pytest.skip(f"PR101 inflate.py not present: {pr101}")
    rep = xi.profile_inflate_py(pr101, label="pr101")
    assert rep["op_count"] > 5
    # PR101 has the iconic three lines of per-channel sub_(1.0)
    assert rep["per_channel_mutation_count"] >= 3


def test_profile_records_line_count(tmp_path):
    src = tmp_path / "i.py"
    src.write_text("x = 1\ny = 2\nz = 3\n")
    rep = xi.profile_inflate_py(src)
    assert rep["line_count"] == 3


def test_profile_records_sha256(tmp_path):
    src = tmp_path / "i.py"
    src.write_text("x = 1\n")
    rep = xi.profile_inflate_py(src)
    assert len(rep["inflate_sha256"]) == 64


def test_profile_handles_syntax_error(tmp_path):
    src = tmp_path / "broken.py"
    src.write_text("def broken(:")
    rep = xi.profile_inflate_py(src)
    assert "parse_error" in rep
    assert rep["ops"] == []


def test_profile_groups_by_cost_class(tmp_path):
    src = tmp_path / "i.py"
    src.write_text(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "def f(up, x):\n"
        "    a = F.interpolate(x)\n"
        "    b = F.interpolate(x)\n"
        "    up.sub_(1.0)\n"
    )
    rep = xi.profile_inflate_py(src)
    bcc = rep["by_cost_class"]
    assert bcc["per-frame"] == 2
    assert bcc["per-channel-mutation"] == 1


def test_main_writes_outputs(tmp_path):
    src = tmp_path / "inflate.py"
    src.write_text(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "def f(x):\n"
        "    return F.interpolate(x)\n"
    )
    out = tmp_path / "out"
    rc = xi.main([
        "--inflate-py", str(src),
        "--label", "test",
        "--output-dir", str(out),
    ])
    assert rc == 0
    assert (out / "op_catalog.json").exists()
    assert (out / "op_catalog.md").exists()
    rep = json.loads((out / "op_catalog.json").read_text())
    assert rep["score_claim"] is False
    assert len(rep["files"]) == 1


def test_main_multi_file_compare(tmp_path):
    s1 = tmp_path / "a.py"
    s1.write_text("import torch\nimport torch.nn.functional as F\ndef f(x): return F.interpolate(x)\n")
    s2 = tmp_path / "b.py"
    s2.write_text("import torch\ndef f(up): up.sub_(1.0)\n")
    out = tmp_path / "out"
    rc = xi.main([
        "--inflate-py", str(s1),
        "--inflate-py", str(s2),
        "--label", "first",
        "--label", "second",
        "--output-dir", str(out),
    ])
    assert rc == 0
    rep = json.loads((out / "op_catalog.json").read_text())
    assert len(rep["files"]) == 2
    assert {f["label"] for f in rep["files"]} == {"first", "second"}


def test_main_missing_file_returns_2(tmp_path):
    rc = xi.main([
        "--inflate-py", str(tmp_path / "nope.py"),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_main_label_count_mismatch_returns_2(tmp_path):
    s = tmp_path / "i.py"
    s.write_text("x = 1\n")
    rc = xi.main([
        "--inflate-py", str(s),
        "--label", "a",
        "--label", "b",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_markdown_lists_per_channel_mutations(tmp_path):
    src = tmp_path / "i.py"
    src.write_text(
        "def f(up):\n"
        "    up[:, 0, 0].sub_(1.0)\n"
        "    up[:, 0, 2].sub_(1.0)\n"
    )
    out = tmp_path / "out"
    xi.main(["--inflate-py", str(src), "--output-dir", str(out)])
    md = (out / "op_catalog.md").read_text()
    assert "per-channel mutations" in md
    assert "[diagnostic: inflate op-cost xray]" in md


def test_qualified_name_handles_dotted_call():
    import ast as _ast
    tree = _ast.parse("torch.nn.functional.interpolate(x)")
    call = tree.body[0].value
    assert isinstance(call, _ast.Call)
    qn = xi._qualified_name(call)
    assert qn == "torch.nn.functional.interpolate"


def test_qualified_name_handles_bare_call():
    import ast as _ast
    tree = _ast.parse("foo()")
    call = tree.body[0].value
    qn = xi._qualified_name(call)
    assert qn == "foo"


def test_is_subscript_lhs_call_detects_pattern():
    import ast as _ast
    tree = _ast.parse("up[:, 0].sub_(1.0)")
    call = tree.body[0].value
    assert xi._is_subscript_lhs_call(call) is True


def test_is_subscript_lhs_call_negative():
    import ast as _ast
    tree = _ast.parse("up.sub_(1.0)")
    call = tree.body[0].value
    assert xi._is_subscript_lhs_call(call) is False


def test_score_claim_false_in_main_output(tmp_path):
    src = tmp_path / "i.py"
    src.write_text("x = 1\n")
    out = tmp_path / "out"
    xi.main(["--inflate-py", str(src), "--output-dir", str(out)])
    rep = json.loads((out / "op_catalog.json").read_text())
    assert rep["score_claim"] is False
    assert rep["evidence_grade"] == "diagnostic_only"
