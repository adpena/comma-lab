from __future__ import annotations

from pathlib import Path

from tac.witness_dsl.pair_population_envelope import PairDomain, PairPopulation
from tools.materialize_canonical_pair_population import PAIR_COUNT, main


def test_materializes_write_once_equal_canonical_n600(tmp_path: Path) -> None:
    output = tmp_path / "population.json"
    assert main(["--output", str(output)]) == 0
    first = output.read_bytes()
    assert main(["--output", str(output)]) == 0
    assert output.read_bytes() == first
    population = PairPopulation.from_bytes(first)
    pair_ids = tuple(range(PAIR_COUNT))
    assert population.source_pair_ids == pair_ids
    assert all(population.domain_index(domain).local_to_source_pair_ids == pair_ids for domain in PairDomain)


def test_refuses_different_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "population.json"
    output.write_bytes(b"not-the-canonical-envelope")
    assert main(["--output", str(output)]) == 2
    assert output.read_bytes() == b"not-the-canonical-envelope"
