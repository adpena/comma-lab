"""Tests for the PR105 ``kitchen_sink`` packed-state-schema size-sorted helper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    PackedStateSchemaEntry,
    pack_state_schema_size_sorted,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Round-trip / behavior ───────────────────────────────────────────────────


def test_pr105_entry_is_frozen_dataclass() -> None:
    entry = PackedStateSchemaEntry(name="a", shape=(2, 2), n_elements=4)
    with pytest.raises((AttributeError, TypeError)):
        entry.n_elements = 999  # type: ignore[misc]


def test_pr105_sorts_by_descending_size() -> None:
    schema = [
        ("a.bias", (3,)),
        ("a.weight", (3, 18, 3, 3)),
        ("b.bias", (72,)),
        ("b.weight", (72, 36, 3, 3)),
    ]
    sorted_schema = pack_state_schema_size_sorted(schema)
    sizes = [e.n_elements for e in sorted_schema]
    assert sizes == sorted(sizes, reverse=True)
    names = [e.name for e in sorted_schema]
    assert names[0] == "b.weight"  # largest: 72*36*3*3 = 23328
    assert names[1] == "a.weight"  # 3*18*3*3 = 486
    assert names[2] == "b.bias"  # 72
    assert names[3] == "a.bias"  # 3


def test_pr105_uses_stable_sort_for_equal_sizes() -> None:
    """Python's sorted() is stable — equal-size entries preserve input order."""
    schema = [
        ("first", (4,)),
        ("second", (4,)),
        ("third", (4,)),
    ]
    sorted_schema = pack_state_schema_size_sorted(schema)
    assert [e.name for e in sorted_schema] == ["first", "second", "third"]


def test_pr105_matches_pr105_source_recipe() -> None:
    """PR105 line 58:
        PACKED_STATE_SCHEMA = sorted(FIXED_STATE_SCHEMA, key=lambda item: -int(np.prod(item[1])))
    Verify equivalence on a representative HNeRV-shaped subset.
    """
    fixed = [
        ("blocks.5.weight", (72, 36, 3, 3)),
        ("blocks.5.bias", (72,)),
        ("skips.2.weight", (27, 36, 1, 1)),
        ("skips.2.bias", (27,)),
    ]
    expected_packed = sorted(fixed, key=lambda item: -int(np.prod(item[1])))
    our_packed = pack_state_schema_size_sorted(fixed)
    expected_names = [item[0] for item in expected_packed]
    our_names = [e.name for e in our_packed]
    assert our_names == expected_names


def test_pr105_returns_tuple_of_frozen_entries() -> None:
    sorted_schema = pack_state_schema_size_sorted(
        [("a", (2,)), ("b", (4,))]
    )
    assert isinstance(sorted_schema, tuple)
    for e in sorted_schema:
        assert isinstance(e, PackedStateSchemaEntry)


def test_pr105_accepts_scalar_shape() -> None:
    """A scalar tensor has shape () and np.prod(()) = 1.0 (numpy convention)."""
    sorted_schema = pack_state_schema_size_sorted([("scalar", ()), ("vec", (2,))])
    # vec (size 2) > scalar (size 1).
    assert sorted_schema[0].name == "vec"
    assert sorted_schema[1].name == "scalar"
    assert sorted_schema[1].n_elements == 1


def test_pr105_empty_schema_returns_empty_tuple() -> None:
    assert pack_state_schema_size_sorted([]) == ()


def test_pr105_preserves_shape_tuple() -> None:
    sorted_schema = pack_state_schema_size_sorted([("a", (3, 4, 5))])
    assert sorted_schema[0].shape == (3, 4, 5)
    assert sorted_schema[0].n_elements == 60


# ── Failure modes ───────────────────────────────────────────────────────────


def test_pr105_rejects_non_tuple_entry() -> None:
    with pytest.raises(ValueError, match="must be a"):
        pack_state_schema_size_sorted([["a", (2,)]])  # type: ignore[list-item]


def test_pr105_rejects_non_string_name() -> None:
    with pytest.raises(TypeError, match="name must be str"):
        pack_state_schema_size_sorted([(42, (2,))])  # type: ignore[list-item]


def test_pr105_rejects_non_tuple_shape() -> None:
    with pytest.raises(ValueError, match="shape must be tuple"):
        pack_state_schema_size_sorted([("a", [2, 3])])  # type: ignore[list-item]


def test_pr105_rejects_non_int_shape_dim() -> None:
    with pytest.raises(ValueError, match="must be int"):
        pack_state_schema_size_sorted([("a", (2.5,))])  # type: ignore[arg-type]


def test_pr105_rejects_bool_shape_dim() -> None:
    """bool is a subtype of int — be explicit about rejecting bool dims."""
    with pytest.raises(ValueError, match="must be int"):
        pack_state_schema_size_sorted([("a", (True,))])  # type: ignore[arg-type]


def test_pr105_rejects_negative_shape_dim() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        pack_state_schema_size_sorted([("a", (-1,))])


def test_pr105_accepts_zero_size_dim() -> None:
    """Empty tensors are mathematically valid; np.prod returns 0."""
    sorted_schema = pack_state_schema_size_sorted([("empty", (0,)), ("nonempty", (1,))])
    # empty has n_elements = 0; nonempty has 1.
    assert sorted_schema[0].name == "nonempty"
    assert sorted_schema[1].name == "empty"
    assert sorted_schema[1].n_elements == 0


# ── Golden vector ───────────────────────────────────────────────────────────


class TestPR105GoldenVector:
    def test_packed_state_schema_golden_vector(self) -> None:
        """The fixed-schema subset from PR105's FIXED_STATE_SCHEMA — verify
        that pack_state_schema_size_sorted produces the exact byte-pinned
        ordering."""
        # Subset of PR105's FIXED_STATE_SCHEMA — representative HNeRV layout.
        schema = [
            ("blocks.0.weight", (36, 28, 3, 3)),
            ("blocks.0.bias", (36,)),
            ("blocks.1.weight", (36, 36, 3, 3)),
            ("blocks.1.bias", (36,)),
            ("blocks.2.weight", (45, 36, 3, 3)),
            ("blocks.2.bias", (45,)),
            ("blocks.3.weight", (54, 45, 3, 3)),
            ("blocks.3.bias", (54,)),
            ("blocks.4.weight", (63, 54, 3, 3)),
            ("blocks.4.bias", (63,)),
            ("blocks.5.weight", (72, 63, 3, 3)),
            ("blocks.5.bias", (72,)),
            ("skips.2.weight", (27, 36, 1, 1)),
            ("skips.2.bias", (27,)),
            ("skips.3.weight", (20, 27, 1, 1)),
            ("skips.3.bias", (20,)),
            ("skips.4.weight", (18, 20, 1, 1)),
            ("skips.4.bias", (18,)),
            ("refine.0.weight", (9, 18, 3, 3)),
            ("refine.0.bias", (9,)),
            ("refine.1.weight", (18, 9, 3, 3)),
            ("refine.1.bias", (18,)),
            ("rgb_0.weight", (3, 18, 3, 3)),
            ("rgb_0.bias", (3,)),
            ("rgb_1.weight", (3, 18, 3, 3)),
            ("rgb_1.bias", (3,)),
        ]
        sorted_schema = pack_state_schema_size_sorted(schema)
        # Canonical sort-order representation: (name, n_elements) pairs.
        ordering = [(e.name, e.n_elements) for e in sorted_schema]
        ordering_blob = json.dumps(ordering, sort_keys=False).encode("utf-8")
        digest = hashlib.sha256(ordering_blob).hexdigest()
        golden = GOLDEN_DIR / "pr105_packed_state_schema_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR105 packed-state-schema ordering changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "input_n_entries": len(schema),
                        "ordering_len": len(ordering_blob),
                        "schema": "pr105_packed_state_schema.v1",
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
