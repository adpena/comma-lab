from __future__ import annotations

import importlib.util
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "experiments/ddm_pq2_compress_e2e.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ddm_pq2_compress_e2e_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_afr1_chain_registry_is_ordered_and_pin_consistent() -> None:
    module = load_module()
    chain = module.AFR1_CHAIN
    stages = list(chain["stages"])

    assert [row["name"] for row in stages] == ["fx5", "dx2", "gb1", "lb1", "afr1"]
    assert stages[0]["input_sha256"] == chain["base_sha256"]
    assert stages[-1]["output_sha256"] == chain["archive_sha256"]
    assert stages[-1]["output_bytes"] == chain["archive_bytes"] == 180_002
    for left, right in pairwise(stages):
        assert left["output_sha256"] == right["input_sha256"]
    for row in stages:
        assert (ROOT / row["tool"]).is_file()
        assert (ROOT / row["receipt"]).is_file()
        assert len(row["input_sha256"]) == len(row["output_sha256"]) == 64


def test_gb1_fork_and_rc64_roles_are_explicit() -> None:
    module = load_module()
    gb1 = list(module.AFR1_CHAIN["stages"])[2]
    branches = {row["name"]: row for row in gb1["branch_outputs"]}

    assert branches["gb1_pointer"]["bytes"] == 180_215
    assert branches["jt21_bank_consumed_by_lb1"]["sha256"] == gb1["output_sha256"]
    assert module.AFR1_CHAIN["archive_sha256"] not in module.NOT_EXPRESSIBLE
    assert module.AFR1_CHAIN["archive_sha256"] in module.CHAIN_RECIPES
    assert module.AFR1_CHAIN_INPUTS["rc64_source"]["sha256"].startswith("5c75e2c7")
    assert module.AFR1_CHAIN_INPUTS["rc64_shipped_member"]["sha256"].startswith("05839d14")
    assert module.AFR1_CHAIN_INPUTS["afr1_source_runtime"]["sha256"].startswith("6462ba51")
    assert "afr1_candidate_runtime" not in module.AFR1_CHAIN_INPUTS


def test_chain_resume_reaches_the_checkpointed_encoder(tmp_path: Path) -> None:
    module = load_module()
    argv = module.jg2_argv(
        stage="encode",
        store=tmp_path / "store",
        runtime=tmp_path / "runtime",
        pointer=tmp_path / "archive.zip",
        pointer_sha256="0" * 64,
        tokens=tmp_path / "tokens.u8",
        tag="resume_probe",
        resume=True,
    )

    assert argv[-1] == "--resume"
