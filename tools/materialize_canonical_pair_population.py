#!/usr/bin/env python3
"""Materialize the frozen canonical ordered n600 PairPopulation envelope."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from tac.witness_dsl.pair_population_envelope import PairPopulation  # noqa: E402

PAIR_COUNT = 600


class CanonicalPairPopulationMaterializerError(ValueError):
    """Raised before output when an existing artifact is not exact."""


def canonical_n600_pair_population() -> PairPopulation:
    pair_ids = tuple(range(PAIR_COUNT))
    return PairPopulation.derive(
        source_pair_ids=pair_ids,
        v9_local_to_source_pair_ids=pair_ids,
        pbr_local_to_source_pair_ids=pair_ids,
        ir_local_to_source_pair_ids=pair_ids,
        v10_local_to_source_pair_ids=pair_ids,
    )


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or path.read_bytes() != payload:
            raise CanonicalPairPopulationMaterializerError(f"existing output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        population = canonical_n600_pair_population()
        payload = population.to_bytes()
        _write_once_or_equal(args.output, payload)
    except (CanonicalPairPopulationMaterializerError, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.resolve(strict=True)),
                "bytes": len(payload),
                "pair_count": len(population.source_pair_ids),
                "population_sha256": population.population_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
