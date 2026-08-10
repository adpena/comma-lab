from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import ddm_ps135_stage_c_mixed_precision as stage_c

GOVERNED_TEST_BASE = (
    stage_c.DEFAULT_OUTPUT_ROOT / "test_retained" / "runtime_custody_p0"
)


@pytest.fixture(scope="module")
def governed_test_store() -> Path:
    root = GOVERNED_TEST_BASE / f"run_{os.getpid()}_{time.time_ns()}"
    storage = stage_c.pose.require_vertigo_free_space(
        root,
        required_free_bytes=100_000_000,
        stage="ddm_ps135_stage_c_focused_tests",
    )
    root.mkdir(parents=True, exist_ok=True)
    stage_c.pose.atomic_json(
        root / "TEST_RUN_STARTED.json",
        {
            "schema": "ddm_ps135_stage_c_test_run.v1",
            "complete": False,
            "score_claim": False,
            "storage_preflight": storage,
            "payloads_retained": True,
        },
    )
    yield root
    files = [
        stage_c.pose.file_record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "TEST_RUN_COMPLETE.json"
    ]
    stage_c.pose.atomic_json(
        root / "TEST_RUN_COMPLETE.json",
        {
            "schema": "ddm_ps135_stage_c_test_run.v1",
            "complete": True,
            "score_claim": False,
            "file_count": len(files),
            "files": files,
            "payloads_retained": True,
        },
    )


@pytest.fixture(scope="module")
def mounted_stage_c_sources() -> tuple[
    list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
]:
    required = (
        stage_c.CHECKPOINT,
        stage_c.pose.LC2_ARCHIVE,
        stage_c.pose.LC2_INPUTS / "semantic.raw",
        stage_c.pose.LC2_INPUTS / "carrier.raw",
        stage_c.SD1_ROOT / "allocations" / f"{stage_c.RUNG_STEMS[0]}.json",
    )
    if any(not path.is_file() for path in required):
        pytest.skip("pinned LC2/SD1 Stage-C sources are not mounted")
    return stage_c.semantic_candidates(), stage_c.pose.load_lc2_source()


@pytest.fixture(scope="module")
def built_all_candidates(
    governed_test_store: Path,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
) -> list[stage_c.ArchiveProduct]:
    candidates, source = mounted_stage_c_sources
    root = governed_test_store / "all_candidate_payloads"
    products: list[stage_c.ArchiveProduct] = []
    for index, candidate in enumerate(candidates):
        product = stage_c.build_stage_c_archive(
            candidate,
            source.carrier,
            source,
            failure_root=root / f"candidate_{index:02d}" / "build_failures",
        )
        records = stage_c.persist_archive_product(
            root / f"candidate_{index:02d}", product, repeat=False
        )
        stage_c.pose.atomic_json(
            root / f"candidate_{index:02d}" / "attempt_receipt.json",
            {
                "schema": "ddm_ps135_stage_c_test_archive_attempt.v1",
                "complete": True,
                "score_claim": False,
                "candidate_id": candidate.candidate_id,
                "records": records,
                "parseback": product.parseback,
                "payloads_retained": True,
            },
        )
        products.append(product)
    return products


def test_q4_semantic_and_archive_are_exact_lc2_identity(
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, _ = mounted_stage_c_sources
    q4_product = built_all_candidates[0]
    assert candidates[0].semantic_blob == (
        stage_c.pose.LC2_INPUTS / "semantic.raw"
    ).read_bytes()
    assert q4_product.archive == stage_c.pose.LC2_ARCHIVE.read_bytes()
    assert q4_product.parseback["semantic_format"] == "legacy_int4"


def test_allocations_are_the_registered_cumulative_q3_prefixes(
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
) -> None:
    candidates, _ = mounted_stage_c_sources
    assert len(candidates) == 5
    expected = dict(candidates[0].allocation)
    assert set(expected.values()) == {4}
    for index, changed_name in enumerate(stage_c.RUNG_TENSORS, 1):
        expected[changed_name] = 3
        assert dict(candidates[index].allocation) == expected


def test_real_lc2_mixed_archive_truthfully_scopes_token_consumption(
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, source = mounted_stage_c_sources
    mixed = built_all_candidates[1]
    parsed = mixed.parseback
    assert parsed["semantic_format"] == "sd1_mixed_v1"
    assert parsed["semantic_allocation"] == dict(candidates[1].allocation)
    assert parsed["carrier_sha256"] == stage_c.pose.sha256_bytes(source.carrier)
    assert parsed["tokens_sha256"] == stage_c.pose.sha256_bytes(source.tokens)
    assert parsed["temporal_sha256"] == stage_c.pose.sha256_bytes(
        source.temporal_packed
    )
    assert parsed["model_sections_consumed"] is True
    assert parsed["token_payload_consumed"] is False
    assert parsed["token_terminal_finish_verified"] is False
    assert parsed["all_sections_consumed"] is False
    assert parsed["validation_scope"]["tokens"] == "BYTE_EQUAL_ONLY_NOT_ANS_FINISHED"
    assert len(mixed.archive) <= stage_c.BYTE_CEILING
    runner_parseback = stage_c.pose.parse_candidate_archive(
        mixed.archive,
        source.carrier,
        source,
        expected_semantic=candidates[1].semantic_blob,
    )
    assert runner_parseback["semantic_allocation"] == dict(candidates[1].allocation)


def _retained_candidate_receipt(
    root: Path,
    semantic: stage_c.SemanticCandidate,
    source: stage_c.pose.LC2Source,
    product: stage_c.ArchiveProduct,
) -> Path:
    allocation_payload = (
        json.dumps(dict(semantic.allocation), indent=2, sort_keys=True) + "\n"
    ).encode()
    records = {
        "semantic": stage_c.pose.persist_exact(root / "semantic.raw", semantic.semantic_blob),
        "carrier": stage_c.pose.persist_exact(root / "carrier.raw", source.carrier),
        "tokens": stage_c.pose.persist_exact(root / "tokens.ans", source.tokens),
        "allocation": stage_c.pose.persist_exact(root / "allocation.json", allocation_payload),
    }
    records.update(stage_c.persist_archive_product(root, product, repeat=False))
    records.update(stage_c.persist_archive_product(root, product, repeat=True))
    receipt = {
        "schema": stage_c.CANDIDATE_RECEIPT_SCHEMA,
        "complete": True,
        "candidate_id": semantic.candidate_id,
        "semantic_allocation": dict(semantic.allocation),
        "source_records": semantic.source_records,
        "byte_ceiling_passes": True,
        "parseback": product.parseback,
        "records": records,
    }
    path = root / "receipt.json"
    stage_c.pose.atomic_json(path, receipt)
    return path


def test_retained_payload_mutation_is_refused(
    governed_test_store: Path,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, source = mounted_stage_c_sources
    q4_product = built_all_candidates[0]
    root = governed_test_store / "retained_payload_mutation"
    receipt_path = _retained_candidate_receipt(
        root, candidates[0], source, q4_product
    )
    receipt = stage_c.verify_retained_candidate(
        receipt_path,
        semantic=candidates[0],
        carrier=source.carrier,
        source=source,
    )
    carrier_path = Path(receipt["records"]["carrier"]["path"])
    carrier_path.write_bytes(carrier_path.read_bytes() + b"mutation")
    stage_c.persist_typed_failure(
        root / "test_failures",
        phase="intentional_carrier_mutation",
        candidate_id=candidates[0].candidate_id,
        reason="focused negative test appended one mutation payload",
        records={"mutated_carrier": stage_c.pose.file_record(carrier_path)},
        details={"bound_receipt": stage_c.pose.file_record(receipt_path)},
    )
    with pytest.raises(stage_c.pose.PoseResolveError, match="bound artifact changed"):
        stage_c.verify_retained_candidate(
            receipt_path,
            semantic=candidates[0],
            carrier=source.carrier,
            source=source,
        )


def test_self_consistent_transformed_record_mutation_is_recomputed_and_refused(
    governed_test_store: Path,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, source = mounted_stage_c_sources
    root = governed_test_store / "self_consistent_transform_mutation"
    receipt_path = _retained_candidate_receipt(
        root, candidates[1], source, built_all_candidates[1]
    )
    receipt = stage_c.pose.load_json(receipt_path)
    transformed_path = Path(receipt["records"]["semantic_cx2"]["path"])
    mutated = bytearray(transformed_path.read_bytes())
    mutated[0] ^= 1
    transformed_path.write_bytes(mutated)
    receipt["records"]["semantic_cx2"] = stage_c.pose.file_record(transformed_path)
    stage_c.pose.atomic_json(receipt_path, receipt)
    stage_c.persist_typed_failure(
        root / "test_failures",
        phase="intentional_transform_mutation",
        candidate_id=candidates[1].candidate_id,
        reason="focused negative test changed transformed bytes and self-hash",
        records={
            "mutated_transform": stage_c.pose.file_record(transformed_path),
            "mutated_receipt": stage_c.pose.file_record(receipt_path),
        },
        details={},
    )
    with pytest.raises(stage_c.StageCError, match="recomputed LC2 wire bytes"):
        stage_c.verify_retained_candidate(
            receipt_path,
            semantic=candidates[1],
            carrier=source.carrier,
            source=source,
        )


def test_candidate_receipt_requires_current_semantic_source_records(
    governed_test_store: Path,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, source = mounted_stage_c_sources
    root = governed_test_store / "source_record_mutation"
    receipt_path = _retained_candidate_receipt(
        root, candidates[1], source, built_all_candidates[1]
    )
    receipt = stage_c.pose.load_json(receipt_path)
    receipt["source_records"] = {}
    stage_c.pose.atomic_json(receipt_path, receipt)
    stage_c.persist_typed_failure(
        root / "test_failures",
        phase="intentional_source_record_mutation",
        candidate_id=candidates[1].candidate_id,
        reason="focused negative test changed source_records",
        records={"mutated_receipt": stage_c.pose.file_record(receipt_path)},
        details={"expected_source_records": candidates[1].source_records},
    )
    with pytest.raises(stage_c.StageCError, match="source records differ"):
        stage_c.verify_retained_candidate(
            receipt_path,
            semantic=candidates[1],
            carrier=source.carrier,
            source=source,
        )


def test_candidate_primary_payloads_exist_before_repeat_comparison(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
) -> None:
    candidates, source = mounted_stage_c_sources
    root = governed_test_store / "candidate_persist_before_compare"
    original = stage_c.build_stage_c_archive
    calls = 0

    def observed_build(
        semantic: stage_c.SemanticCandidate,
        carrier: bytes,
        lc2_source: stage_c.pose.LC2Source,
        *,
        failure_root: Path,
    ) -> stage_c.ArchiveProduct:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert (root / "archive.zip").is_file()
            assert (root / "payload.p").is_file()
            assert (root / "semantic.q10.br").is_file()
            assert (root / "hpac_plus_temporal.q10.br").is_file()
        return original(
            semantic,
            carrier,
            lc2_source,
            failure_root=failure_root,
        )

    monkeypatch.setattr(stage_c, "build_stage_c_archive", observed_build)
    receipt = stage_c.retain_candidate(
        root,
        candidates[1],
        source.carrier,
        source,
    )
    assert calls == 2
    assert set(receipt["records"]) == stage_c.CANDIDATE_RECORD_LABELS
    assert Path(receipt["records"]["archive_repeat"]["path"]).is_file()


def test_parseback_failure_retains_every_materialized_wire_payload(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
) -> None:
    candidates, source = mounted_stage_c_sources
    root = governed_test_store / "forced_parseback_failure"

    def fail_parseback(*_args, **_kwargs):
        raise stage_c.StageCError("forced parseback failure")

    monkeypatch.setattr(stage_c, "parse_stage_c_archive", fail_parseback)
    with pytest.raises(stage_c.StageCError, match="forced parseback failure"):
        stage_c.build_stage_c_archive(
            candidates[1],
            source.carrier,
            source,
            failure_root=root,
        )
    expected = {
        "semantic.raw",
        "carrier.cpr1",
        "hpac_plus_temporal.raw",
        "tokens.ans",
        "semantic.signed_zigzag_lane2.raw",
        "semantic.q10.br",
        "carrier.identity.raw",
        "carrier.q9.br",
        "hpac_plus_temporal.xor80.raw",
        "hpac_plus_temporal.q10.br",
        "models.split_pack.bin",
        "payload.p",
        "archive.zip",
    }
    assert expected.issubset({path.name for path in root.iterdir()})
    failure_paths = list(root.glob("failure_archive_parseback_*.json"))
    assert len(failure_paths) == 1
    failure = stage_c.pose.load_json(failure_paths[0])
    assert failure["complete"] is False
    assert failure["payloads_retained"] is True


@pytest.mark.parametrize(
    ("fail_at", "phase", "expected_labels"),
    (
        (
            "inverse",
            "archive_cx2_inverse",
            {
                "semantic",
                "carrier",
                "hpac_wire",
                "tokens",
                "semantic_cx2",
                "carrier_cx2",
                "hpac_cx2",
            },
        ),
        (
            "compressor",
            "archive_carrier_brotli",
            {
                "semantic",
                "carrier",
                "hpac_wire",
                "tokens",
                "semantic_cx2",
                "carrier_cx2",
                "hpac_cx2",
                "semantic_brotli",
            },
        ),
        (
            "model_pack",
            "archive_model_pack",
            {
                "semantic",
                "carrier",
                "hpac_wire",
                "tokens",
                "semantic_cx2",
                "carrier_cx2",
                "hpac_cx2",
                "semantic_brotli",
                "carrier_brotli",
                "hpac_brotli",
            },
        ),
        (
            "member_pack",
            "archive_member_pack",
            {
                "semantic",
                "carrier",
                "hpac_wire",
                "tokens",
                "semantic_cx2",
                "carrier_cx2",
                "hpac_cx2",
                "semantic_brotli",
                "carrier_brotli",
                "hpac_brotli",
                "model_pack",
            },
        ),
        (
            "zip",
            "archive_zip",
            {
                "semantic",
                "carrier",
                "hpac_wire",
                "tokens",
                "semantic_cx2",
                "carrier_cx2",
                "hpac_cx2",
                "semantic_brotli",
                "carrier_brotli",
                "hpac_brotli",
                "model_pack",
                "member",
            },
        ),
    ),
)
def test_archive_build_failures_retain_each_completed_step(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
    phase: str,
    expected_labels: set[str],
) -> None:
    root = governed_test_store / "synthetic_archive_failures" / fail_at
    semantic = stage_c.SemanticCandidate(
        candidate_id=f"synthetic_{fail_at}",
        allocation=stage_c.OrderedDict(),
        semantic_blob=b"semantic",
        expected_state=stage_c.OrderedDict(),
        source_records={},
    )
    source = SimpleNamespace(hpac_wire=b"hpac", tokens=b"tokens")

    class Receiver:
        @staticmethod
        def encode_cx2_model_sections(*_args):
            return b"semantic-cx2", b"carrier-cx2", b"hpac-cx2"

        @staticmethod
        def decode_cx2_model_sections(*_args):
            if fail_at == "inverse":
                raise RuntimeError("synthetic inverse failure")
            return semantic.semantic_blob, b"carrier", source.hpac_wire

        @staticmethod
        def pack_payload(*_args, **_kwargs):
            if fail_at == "member_pack":
                raise RuntimeError("synthetic member-pack failure")
            return b"member"

    compress_calls = 0

    def compress(payload: bytes, *, quality: int) -> bytes:
        nonlocal compress_calls
        compress_calls += 1
        if fail_at == "compressor" and compress_calls == 2:
            raise RuntimeError("synthetic compressor failure")
        return b"br" + bytes([quality]) + payload

    def pack(_streams) -> bytes:
        if fail_at == "model_pack":
            raise RuntimeError("synthetic model-pack failure")
        return b"model-pack"

    def archive(_member: bytes) -> bytes:
        if fail_at == "zip":
            raise RuntimeError("synthetic zip failure")
        return b"archive"

    monkeypatch.setattr(
        stage_c.pose,
        "import_runtime_modules",
        lambda: (None, Receiver(), None),
    )
    monkeypatch.setattr(stage_c.pose, "brotli_compress", compress)
    monkeypatch.setattr(stage_c.pose, "split_pack", pack)
    monkeypatch.setattr(stage_c.pose, "deterministic_stored_zip", archive)
    with pytest.raises(RuntimeError, match="synthetic"):
        stage_c.build_stage_c_archive(
            semantic,
            b"carrier",
            source,
            failure_root=root,
        )
    failures = list(root.glob(f"failure_{phase}_*.json"))
    assert len(failures) == 1
    receipt = stage_c.pose.load_json(failures[0])
    assert set(receipt["records"]) == expected_labels
    assert receipt["payloads_retained"] is True
    for label, record in receipt["records"].items():
        stage_c.pose.verify_file_record_binding(
            record, label=f"synthetic retained {label}"
        )


def test_candidate_repeat_and_ceiling_failures_are_typed_and_retained(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, source = mounted_stage_c_sources
    product = built_all_candidates[1]
    calls = 0

    def mismatched_build(*_args, **_kwargs) -> stage_c.ArchiveProduct:
        nonlocal calls
        calls += 1
        if calls == 1:
            return product
        return dataclasses.replace(product, parseback={**product.parseback, "repeat": False})

    mismatch_root = governed_test_store / "candidate_repeat_failure"
    monkeypatch.setattr(stage_c, "build_stage_c_archive", mismatched_build)
    with pytest.raises(stage_c.StageCError, match="repeat differs"):
        stage_c.retain_candidate(
            mismatch_root, candidates[1], source.carrier, source
        )
    assert len(
        list(
            (mismatch_root / "failures").glob(
                "failure_candidate_repeat_mismatch_*.json"
            )
        )
    ) == 1

    ceiling_root = governed_test_store / "candidate_ceiling_failure"
    monkeypatch.setattr(stage_c, "build_stage_c_archive", lambda *_a, **_k: product)
    monkeypatch.setattr(stage_c, "BYTE_CEILING", 1)
    with pytest.raises(stage_c.StageCError, match="byte ceiling"):
        stage_c.retain_candidate(
            ceiling_root, candidates[1], source.carrier, source
        )
    assert len(
        list(
            (ceiling_root / "failures").glob(
                "failure_candidate_byte_ceiling_*.json"
            )
        )
    ) == 1


def test_resume_binds_paths_order_receipt_and_candidate_archive(
    governed_test_store: Path,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, source = mounted_stage_c_sources
    q4_product = built_all_candidates[0]
    root = governed_test_store / "resume_prefix"
    receipt_path = _retained_candidate_receipt(
        root / "candidate", candidates[0], source, q4_product
    )
    bindings = {"immutable_verified": {"source": "sha"}, "resume_bound": {"code": "sha"}}
    output_root = root.resolve()
    resume_path = (root / "state.v2.json").resolve()
    state = {
        "schema": "ddm_ps135_stage_c_preflight_state.v2",
        "complete": False,
        "sources": json.loads(json.dumps(bindings)),
        "output_root": str(output_root),
        "resume_path": str(resume_path),
        "manifest_path": str(output_root / "PREFLIGHT.json"),
        "expected_candidate_ids": list(stage_c.EXPECTED_CANDIDATE_IDS),
        "candidates": [
            {
                "candidate_id": candidates[0].candidate_id,
                "receipt": stage_c.pose.file_record(receipt_path),
                "archive_bytes": len(q4_product.archive),
                "archive_sha256": stage_c.pose.sha256_bytes(q4_product.archive),
            }
        ],
    }
    stage_c.verify_preflight_resume_state(
        state,
        bindings,
        output_root=output_root,
        resume_from=resume_path,
    )

    changed_bindings = json.loads(json.dumps(bindings))
    changed_bindings["resume_bound"]["code"] = "changed"
    with pytest.raises(stage_c.StageCError, match="resume bindings differ"):
        stage_c.verify_preflight_resume_state(
            state,
            changed_bindings,
            output_root=output_root,
            resume_from=resume_path,
        )

    state["candidates"][0]["candidate_id"] = stage_c.RUNG_STEMS[0]
    with pytest.raises(stage_c.StageCError, match="exact ordered prefix"):
        stage_c.verify_preflight_resume_state(
            state,
            bindings,
            output_root=output_root,
            resume_from=resume_path,
        )
    state["candidates"][0]["candidate_id"] = candidates[0].candidate_id
    state["complete"] = True
    with pytest.raises(stage_c.StageCError, match="exact five candidates"):
        stage_c.verify_preflight_resume_state(
            state,
            bindings,
            output_root=output_root,
            resume_from=resume_path,
        )


def test_completed_resume_requires_exact_five_q4_identity_ceilings_and_manifest_path(
    governed_test_store: Path,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, source = mounted_stage_c_sources
    root = governed_test_store / "completed_resume"
    bindings = {"immutable_verified": {"source": "sha"}, "resume_bound": {"code": "sha"}}
    output_root = root.resolve()
    resume_path = (root / "state.v2.json").resolve()
    rows: list[dict[str, object]] = []
    for index, (candidate, product) in enumerate(
        zip(candidates, built_all_candidates, strict=True)
    ):
        receipt_path = _retained_candidate_receipt(
            root / f"candidate_{index:02d}", candidate, source, product
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "receipt": stage_c.pose.file_record(receipt_path),
                "archive_bytes": len(product.archive),
                "archive_sha256": stage_c.pose.sha256_bytes(product.archive),
            }
        )
    manifest_path = output_root / "PREFLIGHT.json"
    manifest = {
        "schema": "ddm_ps135_stage_c_preflight.v2",
        "complete": True,
        "q4_identity": True,
        "cumulative_rung_count": 4,
        "candidate_count": 5,
        "expected_candidate_ids": list(stage_c.EXPECTED_CANDIDATE_IDS),
        "output_root": str(output_root),
        "resume_path": str(resume_path),
        "manifest_path": str(manifest_path),
        "candidates": rows,
        "byte_ceiling": stage_c.BYTE_CEILING,
        "all_byte_ceiling_pass": True,
    }
    stage_c.pose.atomic_json(manifest_path, manifest)
    state = {
        "schema": "ddm_ps135_stage_c_preflight_state.v2",
        "complete": True,
        "sources": json.loads(json.dumps(bindings)),
        "output_root": str(output_root),
        "resume_path": str(resume_path),
        "manifest_path": str(manifest_path),
        "expected_candidate_ids": list(stage_c.EXPECTED_CANDIDATE_IDS),
        "candidates": rows,
        "manifest": stage_c.pose.file_record(manifest_path),
    }
    stage_c.verify_preflight_resume_state(
        state,
        bindings,
        output_root=output_root,
        resume_from=resume_path,
    )
    stage_c.verify_completed_preflight_manifest(
        manifest,
        state,
        output_root=output_root,
        resume_from=resume_path,
    )
    copied_manifest = output_root / "copied.PREFLIGHT.json"
    copied_manifest.write_bytes(manifest_path.read_bytes())
    state["manifest"] = stage_c.pose.file_record(copied_manifest)
    with pytest.raises(stage_c.StageCError, match="wrong path"):
        stage_c.verify_preflight_resume_state(
            state,
            bindings,
            output_root=output_root,
            resume_from=resume_path,
        )


def test_all_four_mixed_rung_archives_parse_through_the_real_receiver(
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
    built_all_candidates: list[stage_c.ArchiveProduct],
) -> None:
    candidates, _ = mounted_stage_c_sources
    assert len(built_all_candidates) == len(stage_c.EXPECTED_CANDIDATE_IDS)
    for candidate, product in zip(
        candidates[1:], built_all_candidates[1:], strict=True
    ):
        assert product.parseback["semantic_format"] == "sd1_mixed_v1"
        assert product.parseback["semantic_allocation"] == dict(candidate.allocation)
        assert product.parseback["archive_sha256"] == stage_c.pose.sha256_bytes(
            product.archive
        )
        assert len(product.archive) <= stage_c.BYTE_CEILING


def _tiny_master_chunks(root: Path) -> tuple[list[dict[str, object]], bytes]:
    rows: list[dict[str, object]] = []
    payloads: list[bytes] = []
    for start, end in stage_c.master_chunk_ranges():
        payload = bytes(pair % 251 for pair in range(start, end))
        primary = stage_c.pose.persist_exact(
            root / "chunks" / f"chunk_{start:04d}_{end:04d}.uint8.raw",
            payload,
        )
        repeat = stage_c.pose.persist_exact(
            root
            / "chunks"
            / f"chunk_{start:04d}_{end:04d}.repeat.uint8.raw",
            payload,
        )
        rows.append(
            {
                "pair_start": start,
                "pair_end": end,
                "shape": [end - start, stage_c.pose.CAMERA_H, stage_c.pose.CAMERA_W, 3],
                "dtype": "uint8",
                "layout": "BHWC_pair_order",
                "payload": primary,
                "payload_repeat": repeat,
            }
        )
        payloads.append(payload)
    concatenated = b"".join(payloads)
    concatenated_record = stage_c.pose.persist_exact(
        root / "tiny_ordered_concatenation.raw", concatenated
    )
    stage_c.pose.atomic_json(
        root / "tiny_chunk_receipt.json",
        {
            "schema": "ddm_ps135_tiny_master_chunks.v1",
            "complete": True,
            "score_claim": False,
            "chunks": rows,
            "ordered_concatenation": concatenated_record,
            "payloads_retained": True,
        },
    )
    return rows, concatenated


def _synthetic_registered_semantic() -> stage_c.SemanticCandidate:
    return stage_c.SemanticCandidate(
        candidate_id=stage_c.RUNG_STEMS[0],
        allocation=stage_c.OrderedDict(),
        semantic_blob=b"semantic",
        expected_state=stage_c.OrderedDict(),
        source_records={},
    )


def test_q4_parity_validator_rejects_incomplete_receipt(
    governed_test_store: Path,
) -> None:
    output_root = governed_test_store / "incomplete_parity_gate"
    receipt_path = stage_c.q4_parity_receipt_path(output_root)
    stage_c.pose.atomic_json(
        receipt_path,
        {
            "schema": "ddm_ps135_q4_odd_master_parity.v2",
            "complete": False,
            "score_claim": False,
            "candidate_id": "q4_legacy_control",
            "payloads_retained": False,
        },
    )
    with pytest.raises(stage_c.StageCError, match="incomplete"):
        stage_c.validate_q4_parity_receipt(output_root)


def test_q4_expected_master_accepts_pinned_cold_store_relocation(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "q4_expected_cold_store_relocation"
    frame = b"q4ok"
    frame_bytes = len(frame)
    raw = b"x" * ((2 * stage_c.pose.N - 1) * frame_bytes) + frame
    current_raw = root / "cold_decode" / "0.raw"
    stage_c.pose.persist_exact(current_raw, raw)
    literal_receipt = root / "literal_decode_receipt.json"
    canonical_raw_sha = "a" * 64
    stage_c.pose.atomic_json(
        literal_receipt,
        {
            "raw": {
                "path": str(root / "retired_literal_path" / "0.raw"),
                "bytes": len(raw),
                "sha256": canonical_raw_sha,
            },
            "provenance": {"python": "pinned-test-runtime"},
        },
    )
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", frame_bytes)
    monkeypatch.setattr(
        stage_c, "Q4_PARITY_EXPECTED_SHA256", stage_c.pose.sha256_bytes(frame)
    )
    monkeypatch.setattr(stage_c, "Q4_LITERAL_DECODE_RECEIPT", literal_receipt)
    monkeypatch.setattr(stage_c.pose, "LC2_RAW", current_raw)
    monkeypatch.setattr(stage_c.pose, "LC2_RAW_BYTES", len(raw))
    monkeypatch.setattr(stage_c.pose, "LC2_RAW_SHA256", canonical_raw_sha)

    expected, binding = stage_c._q4_expected_master_frame()

    assert expected == frame
    assert binding["cold_store_relocated"] is True
    assert binding["literal_raw_path"] != binding["current_raw_path"]
    assert binding["raw_receipt_sha256"] == canonical_raw_sha
    assert binding["current_raw_full_sha256_pin"] == canonical_raw_sha


def test_q4_expected_master_refuses_relocated_receipt_with_wrong_full_pin(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "q4_expected_wrong_full_pin"
    frame = b"q4ok"
    frame_bytes = len(frame)
    raw = b"x" * ((2 * stage_c.pose.N - 1) * frame_bytes) + frame
    current_raw = root / "cold_decode" / "0.raw"
    stage_c.pose.persist_exact(current_raw, raw)
    literal_receipt = root / "literal_decode_receipt.json"
    stage_c.pose.atomic_json(
        literal_receipt,
        {
            "raw": {
                "path": str(root / "retired_literal_path" / "0.raw"),
                "bytes": len(raw),
                "sha256": "b" * 64,
            },
            "provenance": {"python": "pinned-test-runtime"},
        },
    )
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", frame_bytes)
    monkeypatch.setattr(
        stage_c, "Q4_PARITY_EXPECTED_SHA256", stage_c.pose.sha256_bytes(frame)
    )
    monkeypatch.setattr(stage_c, "Q4_LITERAL_DECODE_RECEIPT", literal_receipt)
    monkeypatch.setattr(stage_c.pose, "LC2_RAW", current_raw)
    monkeypatch.setattr(stage_c.pose, "LC2_RAW_BYTES", len(raw))
    monkeypatch.setattr(stage_c.pose, "LC2_RAW_SHA256", "a" * 64)

    with pytest.raises(stage_c.StageCError, match="custody pin"):
        stage_c._q4_expected_master_frame()


def test_q4_parity_validator_binds_actual_checkpoint_and_current_sources(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = governed_test_store / "synthetic_complete_parity_gate"
    receipt_path = stage_c.q4_parity_receipt_path(output_root)
    actual = b"q4ok"
    expected_sha = stage_c.pose.sha256_bytes(actual)
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", len(actual))
    monkeypatch.setattr(stage_c, "Q4_PARITY_EXPECTED_SHA256", expected_sha)
    bindings = {"synthetic_current_binding": True}
    monkeypatch.setattr(
        stage_c,
        "q4_parity_bindings",
        lambda *_args, **_kwargs: bindings,
    )
    actual_path = (
        receipt_path.parent
        / f"pair_{stage_c.Q4_PARITY_PAIR_INDEX:04d}_master.uint8.raw"
    )
    actual_record = stage_c.pose.persist_exact(actual_path, actual)
    checkpoint_path = actual_path.with_name(f"{actual_path.name}.state.json")
    stage_c.pose.atomic_json(
        checkpoint_path,
        {
            "schema": "ddm_ps135_master_frame_attempt.v1",
            "complete": True,
            "frames_committed": 1,
            "payload": actual_record,
            "binding": {
                "driver": stage_c.pose.file_record(Path(stage_c.__file__)),
                "candidate_id": "q4_legacy_control",
                "attempt_kind": "parity",
                "pair_start": stage_c.Q4_PARITY_PAIR_INDEX,
                "pair_end": stage_c.Q4_PARITY_PAIR_INDEX + 1,
                "frame_bytes": len(actual),
                "render": bindings,
            },
        },
    )
    receipt = {
        "schema": "ddm_ps135_q4_odd_master_parity.v2",
        "complete": True,
        "parity": True,
        "axis": stage_c.Q4_PARITY_AXIS,
        "score_claim": False,
        "candidate_id": "q4_legacy_control",
        "pair_index": stage_c.Q4_PARITY_PAIR_INDEX,
        "raw_frame_index": stage_c.Q4_PARITY_RAW_FRAME_INDEX,
        "expected_sha256": expected_sha,
        "mismatch_count": 0,
        "payloads_retained": True,
        "output_root": str(output_root.resolve()),
        "receipt_path": str(receipt_path.resolve()),
        "actual": actual_record,
        "render_checkpoint": stage_c.pose.file_record(checkpoint_path),
        "bindings": bindings,
    }
    stage_c.pose.atomic_json(receipt_path, receipt)
    validated_path, validated = stage_c.validate_q4_parity_receipt(
        output_root, source=SimpleNamespace()
    )
    assert validated_path == receipt_path
    assert validated == receipt


def test_q4_parity_zero_frame_failure_does_not_claim_payload_retention(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = governed_test_store / "q4_zero_frame_failure"
    receipt_path = stage_c.q4_parity_receipt_path(output_root)
    attempt_state = stage_c.pose.persist_exact(
        receipt_path.parent / "synthetic_attempt_state.json",
        b'{"frames_committed": 0}\n',
    )
    source = SimpleNamespace(semantic=b"q4-semantic")
    q4 = stage_c.SemanticCandidate(
        candidate_id="q4_legacy_control",
        allocation=stage_c.OrderedDict(),
        semantic_blob=source.semantic,
        expected_state=stage_c.OrderedDict(),
        source_records={},
    )
    monkeypatch.setattr(
        stage_c.pose,
        "require_vertigo_free_space",
        lambda *_args, **_kwargs: {"passes": True},
    )
    monkeypatch.setattr(stage_c.pose, "load_lc2_source", lambda: source)
    monkeypatch.setattr(stage_c, "semantic_candidates", lambda: [q4])
    monkeypatch.setattr(stage_c, "q4_parity_bindings", lambda *_a, **_k: {})
    monkeypatch.setattr(
        stage_c,
        "_q4_expected_master_frame",
        lambda: (b"expected", {}),
    )

    def zero_frame_failure(_source):
        raise stage_c.RetainedMasterRenderError(
            "synthetic q4 frame-zero failure",
            records={"attempt_state": attempt_state},
            completed_frames=0,
        )

    monkeypatch.setattr(stage_c, "load_exact_token_tensor", zero_frame_failure)
    with pytest.raises(
        stage_c.RetainedMasterRenderError, match="frame-zero failure"
    ):
        stage_c.materialize_q4_odd_master_parity(output_root)
    failures = list(
        (receipt_path.parent / "failures").glob(
            "failure_q4_odd_master_parity_render_*.json"
        )
    )
    assert len(failures) == 1
    failure = stage_c.pose.load_json(failures[0])
    assert failure["records"] == {"attempt_state": attempt_state}
    assert failure["payloads_retained"] is False


def test_master_launch_requires_exact_threads_before_any_state_creation(
    governed_test_store: Path,
) -> None:
    output_root = governed_test_store / "wrong_threads_output"
    bulk_root = governed_test_store / "wrong_threads_bulk"
    with pytest.raises(stage_c.StageCError, match="exactly 2"):
        stage_c.materialize_master_bank(
            output_root,
            bulk_root,
            index=1,
            semantic=_synthetic_registered_semantic(),
            threads=3,
        )
    assert not bulk_root.exists()


def test_master_launch_requires_parity_then_frozen_scorer_seam_before_state(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = _synthetic_registered_semantic()
    source = SimpleNamespace()
    output_root = governed_test_store / "master_launch_gates"
    bulk_root = governed_test_store / "master_launch_gates_bulk"
    monkeypatch.setattr(stage_c.pose, "load_lc2_source", lambda: source)
    with pytest.raises(stage_c.StageCError, match="parity receipt is absent"):
        stage_c.materialize_master_bank(
            output_root,
            bulk_root,
            index=1,
            semantic=semantic,
            threads=stage_c.MASTER_RENDER_THREADS,
        )
    assert not bulk_root.exists()

    fake_parity = output_root / "fake_parity_receipt.json"
    stage_c.pose.atomic_json(fake_parity, {"complete": True})
    monkeypatch.setattr(
        stage_c,
        "validate_q4_parity_receipt",
        lambda *_args, **_kwargs: (fake_parity, {"complete": True}),
    )
    with pytest.raises(stage_c.StageCError, match="scorer seam is frozen"):
        stage_c.materialize_master_bank(
            output_root,
            bulk_root,
            index=1,
            semantic=semantic,
            threads=stage_c.MASTER_RENDER_THREADS,
        )
    assert not bulk_root.exists()


def test_master_chunk_primary_is_retained_before_repeat_comparison(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "master_persist_before_compare"
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", 1)
    calls = 0

    def render(start: int, end: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert (
                root / "chunks" / f"chunk_{start:04d}_{end:04d}.uint8.raw"
            ).is_file()
        return bytes(range(start, end))

    row = stage_c.retain_master_chunk(
        root,
        candidate_id="test_master_persist_before_compare",
        start=0,
        end=4,
        render_chunk=render,
    )
    stage_c.pose.atomic_json(
        root / "success_receipt.json",
        {
            "schema": "ddm_ps135_tiny_master_chunk.v1",
            "complete": True,
            "score_claim": False,
            "row": row,
            "payloads_retained": True,
        },
    )
    assert calls == 2
    assert row["payload"]["sha256"] == row["payload_repeat"]["sha256"]


def test_master_chunk_mismatch_retains_typed_failure_receipt(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "master_chunk_mismatch"
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", 1)
    payloads = iter((b"abcd", b"abce"))
    with pytest.raises(stage_c.StageCError, match="repeat differs"):
        stage_c.retain_master_chunk(
            root,
            candidate_id="test_master_chunk_mismatch",
            start=0,
            end=4,
            render_chunk=lambda _start, _end: next(payloads),
        )
    failures = list(
        (root / "failures").glob(
            "failure_master_chunk_geometry_or_repeat_*.json"
        )
    )
    assert len(failures) == 1
    failure = stage_c.pose.load_json(failures[0])
    assert failure["complete"] is False
    assert set(failure["records"]) == {"primary", "repeat"}


def test_streamed_master_failure_retains_frames_and_resumes_without_rerender(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "streamed_master_failure_resume"
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", 2)
    calls: dict[str, list[int]] = {"primary": [], "repeat": []}
    active_kind = "primary"
    fail_primary_once = True

    def frame(_model, _tokens, _torch, *, pair: int) -> bytes:
        nonlocal fail_primary_once
        calls[active_kind].append(pair)
        if active_kind == "primary" and pair == 2 and fail_primary_once:
            fail_primary_once = False
            raise RuntimeError("synthetic mid-frame render failure")
        return bytes((pair, pair))

    def render_to_path(
        start: int,
        end: int,
        payload_path: Path,
        attempt_kind: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        nonlocal active_kind
        active_kind = attempt_kind
        return stage_c.stream_semantic_master_chunk_attempt(
            None,
            None,
            None,
            candidate_id="synthetic_stream",
            attempt_kind=attempt_kind,
            start=start,
            end=end,
            payload_path=payload_path,
            render_binding={"synthetic": True},
        )

    monkeypatch.setattr(stage_c, "render_semantic_master_frame", frame)
    with pytest.raises(stage_c.RetainedMasterRenderError, match="pair 2"):
        stage_c.retain_master_chunk(
            root,
            candidate_id="synthetic_stream",
            start=0,
            end=4,
            render_to_path=render_to_path,
        )
    failures = list(
        (root / "failures").glob("failure_master_primary_render_*.json")
    )
    assert len(failures) == 1
    failure = stage_c.pose.load_json(failures[0])
    assert failure["payloads_retained"] is True
    assert {"partial", "attempt_state"}.issubset(failure["records"])
    retained_partial = stage_c.pose.verify_file_record_binding(
        failure["records"]["partial"], label="retained synthetic partial"
    )
    assert retained_partial.read_bytes() == b"\x00\x00\x01\x01"

    row = stage_c.retain_master_chunk(
        root,
        candidate_id="synthetic_stream",
        start=0,
        end=4,
        render_to_path=render_to_path,
    )
    assert calls["primary"] == [0, 1, 2, 2, 3]
    assert calls["repeat"] == [0, 1, 2, 3]
    assert Path(row["payload"]["path"]).read_bytes() == bytes(
        value for pair in range(4) for value in (pair, pair)
    )
    assert row["payload"]["sha256"] == row["payload_repeat"]["sha256"]


def test_streamed_master_failure_before_frame_zero_does_not_claim_payload(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "streamed_master_zero_frame_failure"
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", 2)

    def fail_frame(*_args, **_kwargs) -> bytes:
        raise RuntimeError("synthetic frame-zero failure")

    def render_to_path(
        start: int,
        end: int,
        payload_path: Path,
        attempt_kind: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        return stage_c.stream_semantic_master_chunk_attempt(
            None,
            None,
            None,
            candidate_id="synthetic_zero",
            attempt_kind=attempt_kind,
            start=start,
            end=end,
            payload_path=payload_path,
            render_binding={"synthetic": True},
        )

    monkeypatch.setattr(stage_c, "render_semantic_master_frame", fail_frame)
    with pytest.raises(stage_c.RetainedMasterRenderError, match="pair 0"):
        stage_c.retain_master_chunk(
            root,
            candidate_id="synthetic_zero",
            start=0,
            end=1,
            render_to_path=render_to_path,
        )
    failure_path = next(
        (root / "failures").glob("failure_master_primary_render_*.json")
    )
    failure = stage_c.pose.load_json(failure_path)
    assert failure["payloads_retained"] is False
    assert set(failure["records"]) == {"attempt_state"}


def test_streamed_master_recreates_empty_payload_after_initialization_crash(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "streamed_master_empty_init_recovery"
    payload_path = root / "chunk.uint8.raw"
    state_path = payload_path.with_name(f"{payload_path.name}.state.json")
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", 2)
    render_binding = {"synthetic": "empty-init-recovery"}
    stage_c.pose.atomic_json(
        state_path,
        {
            "schema": "ddm_ps135_master_frame_attempt.v1",
            "complete": False,
            "binding": {
                "driver": stage_c.pose.file_record(Path(stage_c.__file__)),
                "candidate_id": "synthetic_empty_init",
                "attempt_kind": "primary",
                "pair_start": 0,
                "pair_end": 1,
                "frame_bytes": 2,
                "render": render_binding,
            },
            "frames_committed": 0,
            "next_pair": 0,
            "payload_bytes": 0,
            "prefix_sha256": stage_c.pose.sha256_bytes(b""),
            "recoveries": [],
        },
    )
    assert not payload_path.exists()
    monkeypatch.setattr(
        stage_c,
        "render_semantic_master_frame",
        lambda *_args, **_kwargs: b"ok",
    )
    record, checkpoint = stage_c.stream_semantic_master_chunk_attempt(
        None,
        None,
        None,
        candidate_id="synthetic_empty_init",
        attempt_kind="primary",
        start=0,
        end=1,
        payload_path=payload_path,
        render_binding=render_binding,
    )
    assert Path(record["path"]).read_bytes() == b"ok"
    final_state = stage_c.pose.load_json(Path(checkpoint["path"]))
    assert final_state["complete"] is True
    assert len(final_state["recoveries"]) == 1
    recovery_path = Path(final_state["recoveries"][0]["path"])
    recovery = stage_c.pose.load_json(recovery_path)
    assert recovery["payload_lost"] is False
    assert "initialization crash" in recovery["reason"]
    assert stage_c.pose.sha256_file(recovery_path)[:16] in recovery_path.name


def test_streamed_master_recovery_retains_uncheckpointed_frame_tail(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "streamed_master_tail_recovery"
    payload_path = root / "chunk.uint8.raw"
    state_path = payload_path.with_name(f"{payload_path.name}.state.json")
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", 2)
    render_binding = {"synthetic": "tail-recovery"}
    binding = {
        "driver": stage_c.pose.file_record(Path(stage_c.__file__)),
        "candidate_id": "synthetic_tail",
        "attempt_kind": "primary",
        "pair_start": 0,
        "pair_end": 2,
        "frame_bytes": 2,
        "render": render_binding,
    }
    committed = b"aa"
    uncheckpointed = b"stale-tail"
    initial_crash_record = stage_c.pose.persist_exact(
        root / "initial_crash_image.raw", committed + uncheckpointed
    )
    stage_c.pose.persist_exact(payload_path, committed + uncheckpointed)
    stage_c.pose.atomic_json(
        state_path,
        {
            "schema": "ddm_ps135_master_frame_attempt.v1",
            "complete": False,
            "binding": binding,
            "frames_committed": 1,
            "next_pair": 1,
            "payload_bytes": len(committed),
            "prefix_sha256": stage_c.pose.sha256_bytes(committed),
            "recoveries": [],
        },
    )
    monkeypatch.setattr(
        stage_c,
        "render_semantic_master_frame",
        lambda *_args, pair, **_kwargs: b"bb" if pair == 1 else b"aa",
    )
    record, checkpoint = stage_c.stream_semantic_master_chunk_attempt(
        None,
        None,
        None,
        candidate_id="synthetic_tail",
        attempt_kind="primary",
        start=0,
        end=2,
        payload_path=payload_path,
        render_binding=render_binding,
    )
    assert Path(record["path"]).read_bytes() == b"aabb"
    final_state = stage_c.pose.load_json(Path(checkpoint["path"]))
    assert final_state["complete"] is True
    assert len(final_state["recoveries"]) == 1
    recovery = stage_c.pose.load_json(Path(final_state["recoveries"][0]["path"]))
    assert Path(recovery["tail"]["path"]).read_bytes() == uncheckpointed
    recovery_path = Path(final_state["recoveries"][0]["path"])
    assert stage_c.pose.sha256_file(recovery_path)[:16] in recovery_path.name
    stage_c.pose.verify_file_record_binding(
        initial_crash_record, label="synthetic initial crash image"
    )


def test_q4_final_odd_master_matches_retained_lc2_raw(
    governed_test_store: Path,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
) -> None:
    candidates, source = mounted_stage_c_sources
    assert candidates[0].candidate_id == "q4_legacy_control"
    literal_receipt_path = (
        stage_c.pose.LC2_ROOT / "retained" / "decode" / "decode_receipt.json"
    )
    literal_receipt = stage_c.pose.load_json(literal_receipt_path)
    provenance = literal_receipt["provenance"]
    required_environment = {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONHASHSEED": "0",
    }
    runtime_matches = (
        sys.version.split()[0] == provenance["python"].split()[0]
        and np.__version__ == provenance["numpy"]
        and stage_c.torch.__version__ == provenance["torch"]
        and all(os.environ.get(name) == value for name, value in required_environment.items())
    )
    if not runtime_matches:
        pytest.skip(
            "q4 literal parity requires the retained Python/NumPy/Torch runtime and 2/2/0 environment"
        )
    output_root = governed_test_store / "q4_final_odd_master_parity"
    receipt = stage_c.materialize_q4_odd_master_parity(
        output_root, threads=stage_c.MASTER_RENDER_THREADS
    )
    receipt_path, validated = stage_c.validate_q4_parity_receipt(
        output_root,
        source=source,
        threads=stage_c.MASTER_RENDER_THREADS,
    )
    assert receipt == validated
    assert receipt_path == stage_c.q4_parity_receipt_path(output_root)
    assert receipt["schema"] == "ddm_ps135_q4_odd_master_parity.v2"
    assert receipt["complete"] is True
    assert receipt["parity"] is True
    assert receipt["expected_sha256"] == stage_c.Q4_PARITY_EXPECTED_SHA256
    assert receipt["actual"]["sha256"] == stage_c.Q4_PARITY_EXPECTED_SHA256
    assert receipt["mismatch_count"] == 0


def test_master_chunks_and_final_bank_require_exact_geometry_and_ordered_bytes(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "master_geometry_and_concatenation"
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", 1)
    monkeypatch.setattr(stage_c, "MASTER_BANK_BYTES", stage_c.pose.N)
    chunks, concatenated = _tiny_master_chunks(root)
    primary_paths = stage_c.validate_master_chunks(chunks, bank_root=root)
    bank_path = root / "masters.uint8.raw"
    record = stage_c.pose.persist_exact(bank_path, concatenated)
    ordered = stage_c.verify_master_bank_concatenation(
        record,
        primary_paths,
        bank_root=root,
    )
    assert ordered["chunk_count"] == len(stage_c.master_chunk_ranges())
    assert ordered["bytes"] == stage_c.pose.N

    changed_geometry = json.loads(json.dumps(chunks))
    changed_geometry[0]["pair_end"] = 23
    with pytest.raises(stage_c.StageCError, match="geometry"):
        stage_c.validate_master_chunks(changed_geometry, bank_root=root)

    bank_path.write_bytes(concatenated[::-1])
    stage_c.persist_typed_failure(
        root / "test_failures",
        phase="intentional_master_bank_permutation",
        candidate_id="tiny_master_geometry",
        reason="focused negative test reversed the same-size final bank",
        records={"mutated_bank": stage_c.pose.file_record(bank_path)},
        details={"ordered_concatenation": ordered},
    )
    with pytest.raises(stage_c.StageCError, match="ordered chunk concatenation"):
        stage_c.verify_master_bank_concatenation(
            stage_c.pose.file_record(bank_path),
            primary_paths,
            bank_root=root,
        )


def test_master_assembly_retains_and_replays_uncheckpointed_tail(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = governed_test_store / "master_assembly_tail"
    monkeypatch.setattr(stage_c, "MASTER_FRAME_BYTES", 1)
    monkeypatch.setattr(stage_c, "MASTER_BANK_BYTES", stage_c.pose.N)
    chunks, _ = _tiny_master_chunks(root)
    state_path = root / "state.v2.json"
    state: dict[str, object] = {
        "chunks": chunks,
        "assembly_chunks": 1,
        "assembly_recoveries": [],
    }
    stage_c.pose.atomic_json(state_path, state)
    assembly = root / "masters.assembling.uint8.raw"
    first = Path(chunks[0]["payload"]["path"]).read_bytes()
    tail = b"uncheckpointed-tail"
    assembly.write_bytes(first + tail)
    assert stage_c.reconcile_master_assembly(assembly, state_path, state) == 1
    assert assembly.read_bytes() == first
    recovered = stage_c.pose.load_json(state_path)
    assert len(recovered["assembly_recoveries"]) == 1
    recovery = stage_c.pose.load_json(
        Path(recovered["assembly_recoveries"][0]["path"])
    )
    assert Path(recovery["uncheckpointed_tail"]["path"]).read_bytes() == tail


def test_master_manifest_refuses_stale_current_runtime_binding(
    governed_test_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    mounted_stage_c_sources: tuple[
        list[stage_c.SemanticCandidate], stage_c.pose.LC2Source
    ],
) -> None:
    candidates, source = mounted_stage_c_sources
    semantic = candidates[1]
    root = governed_test_store / "stale_master_manifest"
    manifest_path = root / "manifest.json"
    manifest = {
        "schema": "ddm_ps135_stage_c_master_bank.v3",
        "complete": True,
        "candidate_id": semantic.candidate_id,
        "candidate_index": 1,
        "output_root": str(root / "output"),
        "bank_root": str(root.resolve()),
        "pair_count": stage_c.pose.N,
        "shape": [
            stage_c.pose.N,
            stage_c.pose.CAMERA_H,
            stage_c.pose.CAMERA_W,
            3,
        ],
        "dtype": "uint8",
        "layout": "master_only_pair_order",
        "bindings": {"threads": stage_c.MASTER_RENDER_THREADS, "stale": True},
    }
    stage_c.pose.atomic_json(manifest_path, manifest)
    monkeypatch.setattr(stage_c.pose, "load_lc2_source", lambda: source)
    monkeypatch.setattr(stage_c, "SCORER_SEAM_STATUS", "FROZEN")
    monkeypatch.setattr(
        stage_c,
        "current_master_bank_bindings",
        lambda **_kwargs: {
            "threads": stage_c.MASTER_RENDER_THREADS,
            "current": True,
        },
    )
    with pytest.raises(stage_c.StageCError, match="bindings are stale"):
        stage_c.load_master_bank_manifest(manifest_path, semantic=semantic)


def test_official_batch_geometry_preserves_final_unpadded_eight() -> None:
    assert stage_c.official_batch_geometry() == [16] * 37 + [8]


def test_semantic_rung_admission_is_independent_of_carrier_row_count() -> None:
    admitted = stage_c.semantic_rung_admission(
        previous_compensated_score=0.18,
        candidate_compensated_score=0.17,
        carrier_accepted_rows=0,
    )
    assert admitted["semantic_rung_admitted"] is True
    assert admitted["carrier_accepted_rows"] == 0


def test_jrd_policy_is_dormant_and_scorer_launch_is_typed_blocked() -> None:
    policy = stage_c.JrdReusablePriorPolicy().compile_warm_start()
    blocker = stage_c.scorer_launch_blocker()
    assert policy["state"] == "DORMANT_N1_SCREEN"
    assert policy["active"] is False
    assert policy["precision_actuation"] == "REFUSED_PENDING_N600_CONFIRMATION"
    assert blocker["status"] == "QUEUED_MASTER_BANKS_AND_SCORER_ORCHESTRATION"
    assert blocker["receipt_complete"] is True
    assert blocker["complete"] is False
    assert blocker["stage_c_scorer_ready"] is False
    assert blocker["launch_allowed"] is False
    assert blocker["master_bank_launch_allowed"] is False
    assert blocker["blocker_active"] is True
    assert {row["function"] for row in blocker["closed_runner_seams"]} == {
        "rate_aware_select/save_selected_pass",
        "pose_outputs/GN/JRD/exact_population_refresh",
    }
