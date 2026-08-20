# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import tac.witness_dsl.generative_taskspace_correction as _g
from tac.optimization.direct_description_carrier_compose import (
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
    GenerativeTaskspaceCorrectionError,
)
from tac.witness_dsl.taskspace_ep725_label_local_g_stream import (
    FRAME_BYTES,
    PAIR_BYTES,
    Ep725LabelLocalGStreamError,
    Ep725StreamSourceContractV1,
    GPageRefV1,
    execute_ep725_label_local_g_stream,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    compile_generative_taskspace_correction_v2,
)

_RENDERER_SHA256 = "e" * 64
_FIXTURE_RUNTIME = b"""\
import numpy as np
_FP32 = False
_G = {}

def _labels():
    labels = np.full((384, 512), 2, dtype=np.uint8)
    labels[210:, :] = 0
    labels[275:290, 230:270] = 3
    return labels

def _setup(src):
    if not open(src, "rb").read():
        raise ValueError("empty member")
    n_pairs = 2
    _G.update(
        m={"n_pairs": n_pairs},
        code=np.arange(2 * n_pairs, dtype=np.float32).reshape(2 * n_pairs, 1),
        P={}, rh=384, rw=512, ch=874, cw=1164,
        framebytes=874 * 1164 * 3, dst=None,
    )

def _outputs_from_h0(P, h0, code_row, m, want_rgb, *args, **kwargs):
    labels = _labels().reshape(-1)
    phi = np.full((384 * 512, 5), -8.0, dtype=np.float32)
    phi[np.arange(phi.shape[0]), labels] = 8.0
    rgb = np.zeros((384 * 512, 3), dtype=np.float32) if want_rgb else None
    return phi, rgb

def _render_pair(pi):
    frames = []
    for fk in range(2):
        _outputs_from_h0({}, None, _G["code"][2 * pi + fk], _G["m"], True)
        frames.append(np.full((874, 1164, 3), 17 + 2 * pi + fk, dtype=np.uint8).tobytes())
    with open(_G["dst"], "r+b") as handle:
        handle.seek(pi * 2 * _G["framebytes"])
        handle.write(b"".join(frames))
    return pi
"""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _labels() -> np.ndarray:
    labels = np.full((384, 512), 2, dtype=np.uint8)
    labels[210:, :] = 0
    labels[275:290, 230:270] = 3
    return labels


def _source(tmp_path: Path, *, runtime: bytes = _FIXTURE_RUNTIME) -> tuple[Ep725StreamSourceContractV1, bytes]:
    member = b"fixture-counted-ep725-program"
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("0.bin", member)
    archive = archive_buffer.getvalue()
    archive_path = tmp_path / "archive.zip"
    member_path = tmp_path / "0.bin"
    runtime_path = tmp_path / "inflate.py"
    archive_path.write_bytes(archive)
    member_path.write_bytes(member)
    runtime_path.write_bytes(runtime)
    return (
        Ep725StreamSourceContractV1(
            source_archive_path=archive_path,
            source_member_path=member_path,
            runtime_path=runtime_path,
            source_archive_sha256=_sha256(archive),
            source_archive_bytes=len(archive),
            source_member_sha256=_sha256(member),
            source_member_bytes=len(member),
            runtime_sha256=_sha256(runtime),
            runtime_bytes=len(runtime),
            predictor_renderer_sha256=_RENDERER_SHA256,
            n_pairs=2,
        ),
        member,
    )


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (20, 80, 20),
            (240, 220, 40),
            (30, 30, 30),
            (220, 40, 40),
            (40, 80, 220),
        )
    )


def _state(contract: Ep725StreamSourceContractV1, member: bytes, pair_index: int) -> TaskspacePredictorStateV2:
    return TaskspacePredictorStateV2(
        predictor_program=member,
        predictor_renderer_sha256=contract.predictor_renderer_sha256,
        source_archive_sha256=contract.source_archive_sha256,
        source_runtime_sha256=contract.runtime_sha256,
        source_member_name="0.bin",
        source_pair_ids=(pair_index,),
        labels=_labels()[None],
        transport=NoTransportV2(),
    )


def _teacher(state: TaskspacePredictorStateV2) -> EncoderOnlyTeacherEvidenceV1:
    target = state.labels.copy()
    target[:, 10:12, 12:14] = 1
    return EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=_sha256(memoryview(target).cast("B").tobytes()),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=4,
    )


def _label_local_packet(state: TaskspacePredictorStateV2, pair_index: int) -> bytes:
    program = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(pair_index, "Lane", "birth", "box", 1, 10, 12, 12, 14),),
        realization_profile=_profile(),
    )
    return compile_generative_taskspace_correction_v2(
        state,
        program,
        teacher_evidence=_teacher(state),
    ).packet


def _transport_packet(state: TaskspacePredictorStateV2, pair_index: int) -> bytes:
    program = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(pair_index, "Lane", "birth", "box", 2, 10, 12, 12, 14, 1, 0),),
        realization_profile=_profile(),
    )
    # Encoding is intentionally lower-level: the test needs a well-formed page
    # whose live program is transport-dependent but whose binding names NONE.
    # G45 must reject it before calling the semantic or camera mutators.
    return _g._encode_packet(state, program)


def _page(path: Path, pair_index: int, packet: bytes) -> GPageRefV1:
    path.write_bytes(packet)
    return GPageRefV1(
        pair_index=pair_index,
        path=path,
        sha256=_sha256(packet),
        byte_length=len(packet),
    )


def _valid_pages(
    tmp_path: Path,
    contract: Ep725StreamSourceContractV1,
    member: bytes,
) -> tuple[GPageRefV1, GPageRefV1]:
    return tuple(
        _page(
            tmp_path / f"g_{pair_index}.bin",
            pair_index,
            _label_local_packet(_state(contract, member, pair_index), pair_index),
        )
        for pair_index in range(2)
    )


def test_pairwise_stream_executes_real_v2_g_and_preserves_y0_and_unowned_y1(
    tmp_path: Path,
) -> None:
    contract, member = _source(tmp_path)
    pages = _valid_pages(tmp_path, contract, member)
    run_root = tmp_path / "run"

    receipt = execute_ep725_label_local_g_stream(
        contract,
        pages,
        pair_count=2,
        run_root=run_root,
        storage_safety_reserve_bytes=0,
        allow_non_ssd_for_tests=True,
    )
    repeated = execute_ep725_label_local_g_stream(
        contract,
        pages,
        pair_count=2,
        run_root=run_root,
        storage_safety_reserve_bytes=0,
        allow_non_ssd_for_tests=True,
    )

    output = (run_root / "ep725_label_local_g.raw").read_bytes()
    assert receipt == repeated
    assert receipt.output_raw_size_bytes == 2 * PAIR_BYTES == len(output)
    assert receipt.path_backed_output is True
    assert receipt.full_population_resident_in_python is False
    assert receipt.transport_kind == "NONE"
    assert receipt.score_claim is False
    assert receipt.candidate_claim is False
    assert not (run_root / "ep725_label_local_g.raw.partial").exists()

    checkpoints = []
    for pair_index in range(2):
        checkpoint = json.loads((run_root / "pair_checkpoints" / f"pair_{pair_index:04d}.json").read_bytes())
        checkpoints.append(checkpoint)
        pair = output[pair_index * PAIR_BYTES : (pair_index + 1) * PAIR_BYTES]
        assert pair[:FRAME_BYTES] == bytes([17 + 2 * pair_index]) * FRAME_BYTES
        assert pair[FRAME_BYTES:] != bytes([18 + 2 * pair_index]) * FRAME_BYTES
        assert checkpoint["y0_preserved"] is True
        assert checkpoint["unowned_y1_preserved"] is True
        assert checkpoint["changed_scorer_cells"] == 4
        assert checkpoint["transport_requirement"] == "LABEL_LOCAL"
        assert checkpoint["predictor_labels_sha256"] == _sha256(_labels().tobytes())
    assert receipt.pair_checkpoint_sha256s == tuple(
        _sha256((run_root / "pair_checkpoints" / f"pair_{pair_index:04d}.json").read_bytes()) for pair_index in range(2)
    )


def test_transport_dependent_page_refuses_none_and_resume_reuses_only_verified_prefix(
    tmp_path: Path,
) -> None:
    contract, member = _source(tmp_path)
    state0 = _state(contract, member, 0)
    state1 = _state(contract, member, 1)
    page0 = _page(tmp_path / "g_0.bin", 0, _label_local_packet(state0, 0))
    page1_bad = _page(tmp_path / "g_1_bad.bin", 1, _transport_packet(state1, 1))
    run_root = tmp_path / "run"

    with pytest.raises(
        GenerativeTaskspaceCorrectionError,
        match="primitive lifetime escaped the bound predictor pair population",
    ):
        execute_ep725_label_local_g_stream(
            contract,
            (page0, page1_bad),
            pair_count=2,
            run_root=run_root,
            storage_safety_reserve_bytes=0,
            allow_non_ssd_for_tests=True,
        )
    assert (run_root / "pair_checkpoints" / "pair_0000.json").exists()
    assert not (run_root / "pair_checkpoints" / "pair_0001.json").exists()
    first_checkpoint = (run_root / "pair_checkpoints" / "pair_0000.json").read_bytes()

    page1 = _page(tmp_path / "g_1.bin", 1, _label_local_packet(state1, 1))
    receipt = execute_ep725_label_local_g_stream(
        contract,
        (page0, page1),
        pair_count=2,
        run_root=run_root,
        storage_safety_reserve_bytes=0,
        allow_non_ssd_for_tests=True,
    )
    assert receipt.pair_count == 2
    assert (run_root / "pair_checkpoints" / "pair_0000.json").read_bytes() == first_checkpoint


def test_resume_refuses_committed_output_drift_and_page_drift(tmp_path: Path) -> None:
    contract, member = _source(tmp_path)
    state0 = _state(contract, member, 0)
    state1 = _state(contract, member, 1)
    page0 = _page(tmp_path / "g_0.bin", 0, _label_local_packet(state0, 0))
    page1_bad = _page(tmp_path / "g_1_bad.bin", 1, _transport_packet(state1, 1))
    run_root = tmp_path / "run"
    with pytest.raises(
        GenerativeTaskspaceCorrectionError,
        match="primitive lifetime escaped the bound predictor pair population",
    ):
        execute_ep725_label_local_g_stream(
            contract,
            (page0, page1_bad),
            pair_count=2,
            run_root=run_root,
            storage_safety_reserve_bytes=0,
            allow_non_ssd_for_tests=True,
        )

    partial = run_root / "ep725_label_local_g.raw.partial"
    with partial.open("r+b") as handle:
        handle.seek(7)
        handle.write(b"\x00")
    page1 = _page(tmp_path / "g_1.bin", 1, _label_local_packet(state1, 1))
    with pytest.raises(Ep725LabelLocalGStreamError, match="output range drifted"):
        execute_ep725_label_local_g_stream(
            contract,
            (page0, page1),
            pair_count=2,
            run_root=run_root,
            storage_safety_reserve_bytes=0,
            allow_non_ssd_for_tests=True,
        )

    page0.path.write_bytes(page0.path.read_bytes() + b"x")
    with pytest.raises(Ep725LabelLocalGStreamError, match="bytes/hash drifted"):
        execute_ep725_label_local_g_stream(
            contract,
            (page0, page1),
            pair_count=2,
            run_root=tmp_path / "other-run",
            storage_safety_reserve_bytes=0,
            allow_non_ssd_for_tests=True,
        )


def test_runtime_must_expose_exactly_one_actual_frame1_phi(tmp_path: Path) -> None:
    doubled = _FIXTURE_RUNTIME.replace(
        b'_outputs_from_h0({}, None, _G["code"][2 * pi + fk], _G["m"], True)\n',
        b'_outputs_from_h0({}, None, _G["code"][2 * pi + fk], _G["m"], True)\n'
        b'        _outputs_from_h0({}, None, _G["code"][2 * pi + fk], _G["m"], True)\n',
    )
    contract, member = _source(tmp_path, runtime=doubled)
    state0 = _state(contract, member, 0)
    page0 = _page(tmp_path / "g_0.bin", 0, _label_local_packet(state0, 0))

    with pytest.raises(Ep725LabelLocalGStreamError, match="exactly one actual frame-1 phi"):
        execute_ep725_label_local_g_stream(
            contract,
            (page0,),
            pair_count=1,
            run_root=tmp_path / "run",
            storage_safety_reserve_bytes=0,
            allow_non_ssd_for_tests=True,
        )


def test_final_promotion_recovers_canonical_receipt_from_durable_intent(tmp_path: Path) -> None:
    contract, member = _source(tmp_path)
    pages = _valid_pages(tmp_path, contract, member)
    run_root = tmp_path / "run"
    expected = execute_ep725_label_local_g_stream(
        contract,
        pages,
        pair_count=2,
        run_root=run_root,
        storage_safety_reserve_bytes=0,
        allow_non_ssd_for_tests=True,
    )
    canonical = run_root / "execution_receipt.json"
    canonical.unlink()

    recovered = execute_ep725_label_local_g_stream(
        contract,
        pages,
        pair_count=2,
        run_root=run_root,
        storage_safety_reserve_bytes=0,
        allow_non_ssd_for_tests=True,
    )

    assert recovered == expected
    assert canonical.read_bytes() == expected.to_bytes()
    promotion_receipts = tuple((run_root / "promotion_receipts").glob("*.json"))
    assert len(promotion_receipts) == 1
    assert promotion_receipts[0].read_bytes() == expected.to_bytes()
