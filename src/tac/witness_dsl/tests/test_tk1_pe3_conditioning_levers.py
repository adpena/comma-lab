# SPDX-License-Identifier: MIT
"""DSL tests for TK1 PE3 conditioning and cheapdct4 TR1 consumers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from experiments.train_tr1_partition_renderer_mlx import build_argparser
from tac.witness_dsl import lever_registry as LR
from tac.witness_dsl import tk1_pe3_conditioning_levers_20260805 as M
from tac.witness_dsl.curriculum_dsl import Lever

_REPO = Path(__file__).resolve().parents[4]
_TR1 = _REPO / "experiments" / "train_tr1_partition_renderer_mlx.py"


def _trainer_declared_flags() -> set[str]:
    tree = ast.parse(_TR1.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            out.add(node.args[0].value)
    return out


def test_pe3_factory_returns_lever_with_real_tr1_flags():
    lv = M.lever_tk1_pe3_conditioning("/x/pe3.bin")
    assert isinstance(lv, Lever)
    assert lv.policy_contracts["score_claim"] is False
    assert lv.policy_contracts["label_replacement"] is False
    assert set(lv.overrides) <= _trainer_declared_flags()


def test_cheapdct4_factory_returns_lever_with_real_tr1_flags():
    lv = M.lever_tk1_cheapdct4_pose_accounting("/x/OD9_RECEIPT.json")
    assert isinstance(lv, Lever)
    assert lv.policy_contracts["score_claim"] is False
    assert lv.policy_contracts["full_in_loop_consumption"] is False
    assert set(lv.overrides) <= _trainer_declared_flags()


def test_factories_refuse_empty_paths():
    with pytest.raises(ValueError):
        M.lever_tk1_pe3_conditioning("")
    with pytest.raises(ValueError):
        M.lever_tk1_cheapdct4_pose_accounting("  ")


def test_pe3_factory_parse_roundtrip_through_trainer_argparse():
    lv = M.lever_tk1_pe3_conditioning("/x/pe3.bin")
    argv = ["--variant", "plain", "--out-dir", "unused"]
    for flag, value in lv.overrides.items():
        argv += [flag, str(value)]
    ns = build_argparser().parse_args(argv)
    assert ns.pe3_conditioning_mode == "conditioning_only"
    assert str(ns.pe3_conditioning_cache) == "/x/pe3.bin"


def test_cheapdct4_factory_parse_roundtrip_through_trainer_argparse():
    lv = M.lever_tk1_cheapdct4_pose_accounting("/x/OD9_RECEIPT.json")
    argv = ["--variant", "plain", "--out-dir", "unused"]
    for flag, value in lv.overrides.items():
        argv += [flag, str(value)]
    ns = build_argparser().parse_args(argv)
    assert ns.cheapdct4_pose_mode == "accounting"
    assert str(ns.cheapdct4_pose_cache) == "/x/OD9_RECEIPT.json"


def test_tk1_registry_files_both_factories_under_tr1():
    builds = [
        f for f in LR.build_completeness().factories
        if f.module == "tk1_pe3_conditioning_levers_20260805.py"
    ]
    names = {f.factory for f in builds}
    assert names == {"lever_tk1_pe3_conditioning", "lever_tk1_cheapdct4_pose_accounting"}
    for f in builds:
        assert f.missing_flags == (), f"TK1 factory emits undeclared flags {f.missing_flags}"
        assert f.stub_marker is False
        assert f.trainer_declared is True
        assert f.trainer == M.TRAINER_RELPATH


def test_tk1_constant_manifests_carry_provenance():
    for lv in (
        M.lever_tk1_pe3_conditioning("/x/pe3.bin"),
        M.lever_tk1_cheapdct4_pose_accounting("/x/OD9_RECEIPT.json"),
    ):
        for flag in lv.overrides:
            row = lv.constant_manifest.get(flag)
            assert row is not None
            assert row["rung"]
            assert len(row["provenance"]) > 40


def test_pe3_manifest_names_expected_sha():
    lv = M.lever_tk1_pe3_conditioning("/x/pe3.bin")
    text = lv.constant_manifest["--pe3-conditioning-cache"]["provenance"]
    assert M.PE3_EXPECTED_SECTION_SHA256 in text


def test_tk1_receipt_schemas_are_named():
    pe3 = M.lever_tk1_pe3_conditioning("/x/pe3.bin")
    cheap = M.lever_tk1_cheapdct4_pose_accounting("/x/OD9_RECEIPT.json")
    assert "pe3_conditioning_init" in pe3.runtime_receipt_schemas
    assert "cheapdct4_pose_accounting_init" in cheap.runtime_receipt_schemas
