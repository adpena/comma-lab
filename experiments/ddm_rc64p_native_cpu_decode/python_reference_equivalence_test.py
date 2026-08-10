#!/usr/bin/env python3
"""Pinned-Python oracle check for a compiled lc2 native ANS library."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import constriction
import numpy as np

from native_ans import NativeAnsDecoder


def check(library: Path, *, seed: int = 6464, count: int = 10_003) -> dict:
    version = importlib.metadata.version("constriction")
    if version != "0.5.0":
        raise RuntimeError(f"expected constriction 0.5.0, resolved {version}")
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(count, 5)).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits).astype(np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    symbols = rng.integers(0, 5, count, dtype=np.int32)
    family = constriction.stream.model.Categorical(perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(symbols, family, probabilities)
    compressed = encoder.get_compressed().copy()
    payload = compressed.astype("<u4", copy=False).tobytes(order="C")
    oracle = constriction.stream.stack.AnsCoder(compressed.copy())
    native = NativeAnsDecoder(library, payload)
    split = count // 3
    oracle_first = oracle.decode(family, probabilities[:split])
    native_first = native.decode(family, probabilities[:split])
    snapshot_identity = np.array_equal(
        oracle.get_compressed().astype("<u4", copy=False),
        native.get_compressed().astype("<u4", copy=False),
    )
    oracle_all = np.concatenate(
        [oracle_first, oracle.decode(family, probabilities[split:])]
    )
    native_all = np.concatenate(
        [native_first, native.decode(family, probabilities[split:])]
    )
    result = {
        "schema": "ddm_rc64p_python_reference_equivalence.v1",
        "seed": seed,
        "symbols": count,
        "constriction_version": version,
        "oracle_matches_source": bool(np.array_equal(oracle_all, symbols)),
        "native_matches_source": bool(np.array_equal(native_all, symbols)),
        "native_matches_oracle": bool(np.array_equal(native_all, oracle_all)),
        "midstream_snapshot_identity": bool(snapshot_identity),
        "oracle_empty": bool(oracle.is_empty()),
        "native_empty": bool(native.is_empty()),
    }
    if not all(
        value for name, value in result.items()
        if name not in {"schema", "seed", "symbols", "constriction_version"}
    ):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.library), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
