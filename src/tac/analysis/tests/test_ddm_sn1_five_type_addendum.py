# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.analysis.ddm_sn1_five_type_addendum import (
    CANONICAL_REPRESENTATION_TYPES,
    LAYER_HOMES,
    DDMSN1FiveTypeError,
    SN1FiveTypeStreamTag,
    build_five_type_addendum,
    sha256_file,
)


def _tag(path: Path, *, stream_id: str, type_name: str, home: str) -> SN1FiveTypeStreamTag:
    return SN1FiveTypeStreamTag(
        stream_id=stream_id,
        artifact_path=str(path),
        artifact_sha256=sha256_file(path),
        artifact_selector="/fixture",
        representation_type=type_name,
        layer_home=home,
        evaluate_recursion_level="L1_TERM_NATIVE_GEOMETRY",
        derivation="derived from the evaluator term's native geometry",
        metric_geometry="SEG_MARGIN_FISHER_RANK4",
        first_rung="measure the next receiver-closed rung",
        verdict_scope="synthetic contract fixture only",
    )


def test_addendum_closes_five_types_and_l1_l5(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    tags: list[SN1FiveTypeStreamTag] = []
    for index, (type_name, home) in enumerate(zip(CANONICAL_REPRESENTATION_TYPES, LAYER_HOMES, strict=True)):
        artifact = tmp_path / f"artifact-{index}.json"
        artifact.write_text(f'{{"index":{index}}}\n')
        tags.append(
            _tag(
                artifact,
                stream_id=f"stream-{index}",
                type_name=type_name,
                home=home,
            )
        )

    payload = build_five_type_addendum(
        repo_root=tmp_path,
        source_receipts=[{"path": str(receipt), "sha256": sha256_file(receipt)}],
        tags=tags,
        typed_stream_tag_status="ABSENT_AT_FIRE_TIME",
    )
    assert payload["representation_types_covered"] == sorted(CANONICAL_REPRESENTATION_TYPES)
    assert payload["layer_homes_covered"] == list(LAYER_HOMES)
    assert payload["authority"]["parallel_enum_created"] is False
    assert all(row["identity_euclidean_control"] is False for row in payload["rows"])


def test_addendum_refuses_stale_artifact(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n")
    tags = []
    for index, (type_name, home) in enumerate(zip(CANONICAL_REPRESENTATION_TYPES, LAYER_HOMES, strict=True)):
        artifact = tmp_path / f"artifact-{index}.json"
        artifact.write_text("{}\n")
        tags.append(
            _tag(
                artifact,
                stream_id=f"stream-{index}",
                type_name=type_name,
                home=home,
            )
        )
    Path(tags[0].artifact_path).write_text('{"drift":true}\n')

    with pytest.raises(DDMSN1FiveTypeError, match="SHA drift"):
        build_five_type_addendum(
            repo_root=tmp_path,
            source_receipts=[{"path": str(receipt), "sha256": sha256_file(receipt)}],
            tags=tags,
            typed_stream_tag_status="ABSENT_AT_FIRE_TIME",
        )


def test_tag_reuses_canonical_type_vocabulary(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"x")
    with pytest.raises(DDMSN1FiveTypeError, match="REPRESENTATION_TYPES"):
        _tag(
            artifact,
            stream_id="bad",
            type_name="PARALLEL_ENUM_VALUE",
            home="L1_PROGRAM",
        )
