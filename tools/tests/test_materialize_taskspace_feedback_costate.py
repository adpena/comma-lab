from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from tac.witness_dsl.pair_population_envelope import PairPopulation

REPO = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tool = _load(REPO / "tools/materialize_taskspace_feedback_costate.py", "g26_materializer_tool")
g18_fixture = _load(
    REPO / "src/tac/witness_dsl/tests/test_taskspace_g8_a3_interaction_feedback.py",
    "g26_materializer_g18_fixture",
)


def _population() -> bytes:
    pair_ids = tuple(range(600))
    return PairPopulation.derive(
        source_pair_ids=pair_ids,
        v9_local_to_source_pair_ids=pair_ids,
        pbr_local_to_source_pair_ids=pair_ids,
        ir_local_to_source_pair_ids=pair_ids,
        v10_local_to_source_pair_ids=pair_ids,
    ).to_bytes()


def test_cli_materializes_write_once_equal_terminal_receipt(tmp_path: Path) -> None:
    final_receipt = tmp_path / "final_receipt.json"
    population = tmp_path / "pair_population.json"
    output = tmp_path / "materialized.json"
    final_receipt.write_bytes(g18_fixture._receipt(tmp_path / "g14"))
    population.write_bytes(_population())
    argv = [
        "--g14-final-receipt",
        str(final_receipt),
        "--pair-population",
        str(population),
        "--output",
        str(output),
    ]
    assert tool.main(argv) == 0
    first = output.read_bytes()
    assert tool.main(argv) == 0
    assert output.read_bytes() == first


def test_cli_refuses_nonterminal_input(tmp_path: Path) -> None:
    partial = tmp_path / "partial.json"
    population = tmp_path / "pair_population.json"
    partial.write_text('{"schema":"tac.taskspace_g8_a3_n2_allocator.v1"}\n', encoding="ascii")
    population.write_bytes(_population())
    assert (
        tool.main(
            [
                "--g14-final-receipt",
                str(partial),
                "--pair-population",
                str(population),
                "--output",
                str(tmp_path / "must_not_exist.json"),
            ]
        )
        == 2
    )
    assert not (tmp_path / "must_not_exist.json").exists()


def test_stable_read_identity_ignores_atime_but_not_content_metadata() -> None:
    fields = {
        "st_dev": 1,
        "st_ino": 2,
        "st_mode": 0o100644,
        "st_nlink": 1,
        "st_uid": 501,
        "st_gid": 20,
        "st_size": 123,
        "st_mtime_ns": 456,
        "st_ctime_ns": 789,
        "st_atime_ns": 10,
    }
    before = SimpleNamespace(**fields)
    after_read = SimpleNamespace(**{**fields, "st_atime_ns": 11})
    changed = SimpleNamespace(**{**fields, "st_mtime_ns": 457})
    assert tool._stable_read_identity(before) == tool._stable_read_identity(after_read)
    assert tool._stable_read_identity(before) != tool._stable_read_identity(changed)
