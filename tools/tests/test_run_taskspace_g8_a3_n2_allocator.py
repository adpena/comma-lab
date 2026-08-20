from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO / "tools/run_taskspace_g8_a3_n2_allocator.py"
SPEC = importlib.util.spec_from_file_location("run_taskspace_g8_a3_n2_allocator", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha(value: str | bytes) -> str:
    payload = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


BASELINE_BUNDLE = _sha("immutable-base-bundle")


def test_stage_token_accepts_derived_maximum_source_identity() -> None:
    proposal_id = "p" * 192
    program_id = "a" * 192
    program_sha256 = "f" * 64
    source_identity = f"{proposal_id}:{program_id}:{program_sha256}"
    assert len(source_identity) == 450

    token = runner._stage_token("stage_310.g0_a_control", source_identity)
    assert token == (
        f"stage_310.g0_a_control.{proposal_id[:96]}.{hashlib.sha256(source_identity.encode('ascii')).hexdigest()[:12]}"
    )


def test_stage_token_accepts_concrete_post_g8_copy_matched_control_identity() -> None:
    proposal_id = "class_bounded_target_medoids_v1__k16__target_rgb_sse_descending_v1__n16"
    program_id = "g86e80800bf258:post_g8_y1_support_copy_v1:16:4fae70417b5f"
    program_sha256 = "4fae70417b5f6d507bb04ffe382e4100ce9ca9f653f85612fa8e8efedee46ec2"
    source_identity = f"{proposal_id}:{program_id}:{program_sha256}"
    assert len(source_identity) == 194

    token = runner._stage_token("stage_310.g0_a_control", source_identity)
    assert token.startswith(f"stage_310.g0_a_control.{proposal_id}")
    assert token.endswith(hashlib.sha256(source_identity.encode("ascii")).hexdigest()[:12])


def test_stage_token_hashes_long_identity_suffix_beyond_visible_prefix() -> None:
    shared = "a" * 449
    first = runner._stage_token("stage_310.g0_a_control", f"{shared}x")
    second = runner._stage_token("stage_310.g0_a_control", f"{shared}y")
    first_visible, first_digest = first.rsplit(".", 1)
    second_visible, second_digest = second.rsplit(".", 1)
    assert first_visible == second_visible
    assert first_digest != second_digest


@pytest.mark.parametrize(
    "source_identity",
    (
        "a" * 451,
        "",
        "_starts-with-punctuation",
        "contains whitespace",
        "contains/slash",
        "contains@unsafe",
        "nonascii-\N{LATIN SMALL LETTER E WITH ACUTE}",
    ),
)
def test_stage_token_rejects_out_of_contract_source_identity(source_identity: str) -> None:
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="stage source identity"):
        runner._stage_token("stage_310.g0_a_control", source_identity)


def test_stage_token_rejects_unsafe_prefix_through_general_id_contract() -> None:
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="stage prefix"):
        runner._stage_token("stage/310", "lawful-source")


def test_stage_token_short_identifier_is_byte_identical_to_historical_output() -> None:
    assert runner._stage_token("stage_100.g8_pass_a", "alpha:beta.gamma-delta_7") == (
        "stage_100.g8_pass_a.alpha_beta.gamma-delta_7.7c67ec49df46"
    )


def _measurement(
    measurement_id: str,
    *,
    g8: str | None,
    a_program: str,
    a_mode: runner.AModeV1,
    a_rows: int,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    y1: str,
    seg: str,
) -> runner.WholeObjectMeasurementV1:
    d_seg = float(np.float32(d_seg))
    d_pose = float(np.float32(d_pose))
    payload = (measurement_id.encode("ascii") * (archive_bytes // len(measurement_id) + 1))[:archive_bytes]
    selected_sha = _sha(payload)
    deflated_sha = _sha(payload + b":deflated-control")
    return runner.WholeObjectMeasurementV1(
        measurement_id=measurement_id,
        baseline_bundle_sha256=BASELINE_BUNDLE,
        bundle_sha256=_sha(f"bundle:{measurement_id}"),
        g8_program_sha256=g8,
        a_program_sha256=a_program,
        a_packet_sha256=_sha(f"a-packet:{measurement_id}:{a_program}"),
        a_source_binding_sha256=_sha(f"a-source:{y1}"),
        a_mode=a_mode,
        a_row_count=a_rows,
        raw_section_bytes=max(1, archive_bytes - 4),
        member_bytes=max(1, archive_bytes - 2),
        member_sha256=_sha(f"member:{measurement_id}"),
        stored_archive_bytes=archive_bytes,
        stored_archive_sha256=selected_sha,
        deflated_archive_bytes=archive_bytes + 3,
        deflated_archive_sha256=deflated_sha,
        selected_encoding="STORE",
        selected_archive_bytes=archive_bytes,
        selected_archive_sha256=selected_sha,
        selected_archive_payload=payload,
        decoded_output_sha256=_sha(f"decoded:{measurement_id}"),
        receiver_receipt_sha256=_sha(f"receiver:{measurement_id}"),
        camera_y1_sha256=y1,
        candidate_seg_labels_sha256=seg,
        scorer_evidence=runner.BoundedScorerEvidenceV1(
            measurement_receipt_sha256=_sha(f"measurement-receipt:{measurement_id}"),
            candidate_forward_receipt_sha256=_sha(f"candidate-forward:{measurement_id}"),
            candidate_pose6_sha256=_sha(f"candidate-pose:{measurement_id}"),
            per_pair_d_seg=(d_seg, d_seg),
            per_pair_d_pose=(d_pose, d_pose),
            sample_count=2,
            frozen_scorer_sha256=_sha("frozen-scorer"),
            target_forward_receipt_sha256=_sha("target-forward"),
        ),
        d_seg=d_seg,
        d_pose=d_pose,
    )


def _proposal(
    family: str,
    order: str,
    prefix: int,
    *,
    palette: int | None = None,
    alias_sha: str | None = None,
) -> SimpleNamespace:
    proposal_id = f"{family.lower()}:{order.lower()}:{palette or 0}:{prefix}"
    program_sha = alias_sha or _sha(f"program:{proposal_id}")
    receipt = SimpleNamespace(
        family=SimpleNamespace(value=family),
        prefix_order=SimpleNamespace(value=order),
        palette_bound_per_class=palette,
        requested_prefix_cell_count=prefix,
        program_sha256=program_sha,
    )
    return SimpleNamespace(proposal_id=proposal_id, program=object(), receipt=receipt)


def _acquisition(*, alias: bool = False) -> SimpleNamespace:
    families = (
        ("CLASS_SHARED_TARGET_MEDOID_V1", 1),
        ("CLASS_BOUNDED_TARGET_MEDOIDS_V1", 2),
        ("TARGET_PIXEL_RGB_ORACLE_CONTROL_V1", None),
    )
    orders = ("CANONICAL_ADDRESS_V1", "TARGET_RGB_SSE_DESCENDING_V1")
    proposals = []
    alias_sha = _sha("intentional-alias")
    for family, palette in families:
        for order in orders:
            for prefix in (1, 4, 16):
                proposals.append(
                    _proposal(
                        family,
                        order,
                        prefix,
                        palette=palette,
                        alias_sha=(
                            alias_sha if alias and prefix == 16 and order == "TARGET_RGB_SSE_DESCENDING_V1" else None
                        ),
                    )
                )
    return SimpleNamespace(proposals=tuple(proposals))


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.acquisition = _acquisition(alias=True)

    def measure_primary_baseline(self) -> runner.WholeObjectMeasurementV1:
        self.calls.append("baseline")
        return _measurement(
            "g0-pass",
            g8=None,
            a_program=_sha("canonical-pass-program"),
            a_mode=runner.AModeV1.PASS_A_V1,
            a_rows=0,
            d_seg=0.04,
            d_pose=4.0,
            archive_bytes=100,
            y1=_sha("g0-y1"),
            seg=_sha("g0-seg"),
        )

    def measure_exact_semantic_control(self) -> runner.WholeObjectMeasurementV1:
        self.calls.append("exact-control")
        return _measurement(
            "exact-semantic-control",
            g8=None,
            a_program=_sha("exact-control-pass-program"),
            a_mode=runner.AModeV1.PASS_A_V1,
            a_rows=0,
            d_seg=0.03,
            d_pose=3.8,
            archive_bytes=130,
            y1=_sha("exact-y1"),
            seg=_sha("exact-seg"),
        )

    def acquire_g8(self) -> tuple[object, dict[str, Any]]:
        self.calls.append("acquire-g8")
        return self.acquisition, {
            "realization_debt_cell_count": 16,
            "four_way_counts": {"closed": 100, "realization": 16, "topology": 2, "fortunate": 1},
        }

    def freeze_g8_acquisition(
        self,
        acquisition: object,
        branches: tuple[runner.G8BranchV1, ...],
    ) -> dict[str, Any]:
        assert acquisition is self.acquisition
        return {
            "schema": "fake.typed_g8_restore.v1",
            "branches": [branch.as_dict() for branch in branches],
            "payload_sha256": _sha(runner._canonical_json([branch.as_dict() for branch in branches])),
        }

    def restore_g8_acquisition(
        self,
        payload: dict[str, Any],
    ) -> tuple[runner.G8BranchV1, ...]:
        self.calls.append("restore-g8")
        rows = payload["branches"]
        assert payload["payload_sha256"] == _sha(runner._canonical_json(rows))
        return tuple(
            runner.G8BranchV1(
                proposal_id=row["proposal_id"],
                program_sha256=row["program_sha256"],
                family=row["family"],
                prefix_order=row["prefix_order"],
                palette_bound_per_class=row["palette_bound_per_class"],
                prefix_cell_count=row["prefix_cell_count"],
                program=object(),
            )
            for row in rows
        )

    @staticmethod
    def _branch_quality(branch: runner.G8BranchV1) -> tuple[float, float, int]:
        family_offset = {
            "TARGET_PIXEL_RGB_ORACLE_CONTROL_V1": 0.0,
            "CLASS_BOUNDED_TARGET_MEDOIDS_V1": 0.004,
            "CLASS_SHARED_TARGET_MEDOID_V1": 0.008,
        }[branch.family]
        prefix_gain = {1: 0.004, 4: 0.008, 16: 0.012}[branch.prefix_cell_count]
        return 0.04 + family_offset - prefix_gain, 3.5, 105 + branch.prefix_cell_count

    def measure_g8_pass_a(self, branch: runner.G8BranchV1) -> runner.WholeObjectMeasurementV1:
        self.calls.append(f"g8-pass:{branch.proposal_id}")
        d_seg, d_pose, size = self._branch_quality(branch)
        return _measurement(
            f"g8-pass-{branch.program_sha256[:12]}",
            g8=branch.program_sha256,
            a_program=_sha(f"pass-program:{branch.program_sha256}"),
            a_mode=runner.AModeV1.PASS_A_V1,
            a_rows=0,
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=size,
            y1=_sha(f"y1:{branch.program_sha256}"),
            seg=_sha(f"seg:{branch.program_sha256}"),
        )

    def acquire_g0_a(self, row_counts: tuple[int, ...]) -> tuple[runner.AProgramV1, ...]:
        self.calls.append("acquire-g0-a")
        y1 = _sha("g0-y1")
        programs = [
            runner.AProgramV1(
                program_id="g0:pass",
                program_sha256=_sha("canonical-pass-program"),
                mode=runner.AModeV1.PASS_A_V1,
                row_count=0,
                acquisition_y1_sha256=y1,
                program=object(),
                ranking_sha256=_sha("g0-pass-ranking-empty"),
            )
        ]
        for count in row_counts:
            if count == 0:
                continue
            for mode in (
                runner.AModeV1.TARGET_CONSTANT_RGB_V1,
                runner.AModeV1.POST_G8_Y1_SUPPORT_COPY_V1,
            ):
                programs.append(
                    runner.AProgramV1(
                        program_id=f"g0:{mode.value.lower()}:{count}",
                        program_sha256=_sha(f"g0-a:{mode.value}:{count}"),
                        mode=mode,
                        row_count=count,
                        acquisition_y1_sha256=y1,
                        program=object(),
                        ranking_sha256=_sha(f"g0-ranking:{mode.value}"),
                    )
                )
        return tuple(programs)

    def acquire_a(
        self,
        branch: runner.G8BranchV1,
        row_counts: tuple[int, ...],
    ) -> tuple[runner.AProgramV1, ...]:
        self.calls.append(f"acquire-a:{branch.proposal_id}")
        y1 = _sha(f"y1:{branch.program_sha256}")
        pass_program = runner.AProgramV1(
            program_id=f"pass:{branch.program_sha256[:12]}",
            program_sha256=_sha(f"pass-program:{branch.program_sha256}"),
            mode=runner.AModeV1.PASS_A_V1,
            row_count=0,
            acquisition_y1_sha256=y1,
            program=object(),
            ranking_sha256=_sha("pass-ranking-empty"),
        )
        programs = [pass_program]
        for count in row_counts:
            if count == 0:
                continue
            for mode in (
                runner.AModeV1.TARGET_CONSTANT_RGB_V1,
                runner.AModeV1.POST_G8_Y1_SUPPORT_COPY_V1,
            ):
                programs.append(
                    runner.AProgramV1(
                        program_id=f"{branch.program_sha256[:8]}:{mode.value.lower()}:{count}",
                        program_sha256=_sha(f"a:{branch.program_sha256}:{mode.value}:{count}"),
                        mode=mode,
                        row_count=count,
                        acquisition_y1_sha256=y1,
                        program=object(),
                        ranking_sha256=_sha(f"post-g8-ranking:{branch.program_sha256}:{mode.value}"),
                    )
                )
        return tuple(programs)

    def measure_g8_a(
        self,
        branch: runner.G8BranchV1,
        program: runner.AProgramV1,
    ) -> runner.WholeObjectMeasurementV1:
        self.calls.append(f"g8-a:{branch.proposal_id}:{program.program_id}")
        d_seg, d_pose, size = self._branch_quality(branch)
        return _measurement(
            f"g8-a-{_sha(branch.proposal_id + program.program_id)[:12]}",
            g8=branch.program_sha256,
            a_program=program.program_sha256,
            a_mode=program.mode,
            a_rows=program.row_count,
            d_seg=d_seg,
            d_pose=max(0.0, d_pose - 0.02 * max(1, program.row_count)),
            archive_bytes=size + max(1, program.row_count),
            y1=_sha(f"y1:{branch.program_sha256}"),
            seg=_sha(f"seg:{branch.program_sha256}"),
        )

    def measure_g0_a(self, program: runner.AProgramV1) -> runner.WholeObjectMeasurementV1:
        self.calls.append(f"g0-a:{program.program_id}")
        return _measurement(
            f"g0-a-{program.program_sha256[:12]}",
            g8=None,
            a_program=program.program_sha256,
            a_mode=program.mode,
            a_rows=program.row_count,
            d_seg=0.04,
            d_pose=max(0.0, 4.0 - 0.01 * max(1, program.row_count)),
            archive_bytes=100 + max(1, program.row_count),
            y1=_sha("g0-y1"),
            seg=_sha("g0-seg"),
        )


def _store(tmp_path: Path) -> runner.AtomicRunStore:
    return runner.AtomicRunStore.create(
        tmp_path / "durable-run",
        stable_contract={"config": "v1", "implementation_sha256": _sha("impl")},
        pointer_start={"pointer_sha256": _sha("pointer-start"), "target_score": 0.172},
    )


def test_reviewed_prefix_universe_clips_and_keeps_debt_endpoint() -> None:
    assert runner.resolve_g8_prefixes("cheap", debt_count=10) == (1, 4, 10)
    assert runner.resolve_g8_prefixes("full", debt_count=7952) == (
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        7952,
    )
    assert runner.resolve_g8_prefixes("1,16,4096", debt_count=16) == (1, 16)
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="subset"):
        runner.resolve_g8_prefixes("2,8", debt_count=16)


@pytest.mark.parametrize("value", ["", "1,1", "4,1", "-1", "1, 4"])
def test_a_prefix_parser_refuses_noncanonical_or_nonreviewed(value: str) -> None:
    with pytest.raises(runner.argparse.ArgumentTypeError):
        runner.parse_a_prefixes(value)


def test_g12_adapter_requires_all_families_orders_and_dedupes_aliases() -> None:
    branches = runner.g12_branches_from_acquisition(_acquisition(alias=True))
    assert set(branches[0].as_dict()) == {
        "proposal_id",
        "program_sha256",
        "family",
        "prefix_order",
        "palette_bound_per_class",
        "prefix_cell_count",
    }
    unique, aliases = runner.dedupe_g8_branches(branches, allowed_prefixes=(1, 4, 16))
    assert len(branches) == 18
    assert len(unique) == 16
    assert len(aliases[_sha("intentional-alias")]) == 3
    broken = _acquisition()
    broken.proposals = tuple(
        proposal
        for proposal in broken.proposals
        if proposal.receipt.family.value != "TARGET_PIXEL_RGB_ORACLE_CONTROL_V1"
    )
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="omitted"):
        runner.g12_branches_from_acquisition(broken)


def test_pairwise_delta_byte_ceiling_and_interaction_are_exact() -> None:
    before = _measurement(
        "before",
        g8=None,
        a_program=_sha("pass"),
        a_mode=runner.AModeV1.PASS_A_V1,
        a_rows=0,
        d_seg=0.02,
        d_pose=1.0,
        archive_bytes=100,
        y1=_sha("y1-0"),
        seg=_sha("seg-0"),
    )
    after = _measurement(
        "after",
        g8=_sha("g8"),
        a_program=_sha("pass-g8"),
        a_mode=runner.AModeV1.PASS_A_V1,
        a_rows=0,
        d_seg=0.019,
        d_pose=0.81,
        archive_bytes=120,
        y1=_sha("y1-1"),
        seg=_sha("seg-1"),
    )
    audit = runner.PairwiseTransitionV1.between(before, after)
    expected = 100 * (after.d_seg - before.d_seg) + (10 * after.d_pose) ** 0.5 - (10 * before.d_pose) ** 0.5
    expected += 25 * 20 / runner.CONTEST_REFERENCE_BYTES
    assert audit.exact_score_delta == pytest.approx(expected, abs=1e-15)
    assert audit.greatest_strict_integer_byte_delta == (pytest.approx(audit.greatest_strict_integer_byte_delta, abs=0))
    assert audit.greatest_strict_integer_byte_delta < audit.finite_byte_ceiling_real
    assert audit.greatest_strict_integer_byte_delta + 1 >= audit.finite_byte_ceiling_real
    assert runner.interaction_score(g8_a=after, g8_pass=after, g0_a=before, g0_pass=before) == 0.0


def test_y1_seg_and_dseg_must_be_invariant_within_g8_branch() -> None:
    backend = FakeBackend()
    branch = runner.g12_branches_from_acquisition(backend.acquisition)[0]
    anchor = backend.measure_g8_pass_a(branch)
    program = backend.acquire_a(branch, (0, 1))[1]
    variant = backend.measure_g8_a(branch, program)
    runner.assert_a_variant_invariants(anchor, (anchor, variant))
    changed = _measurement(
        "changed-seg",
        g8=branch.program_sha256,
        a_program=program.program_sha256,
        a_mode=program.mode,
        a_rows=program.row_count,
        d_seg=variant.d_seg + 0.001,
        d_pose=variant.d_pose,
        archive_bytes=variant.selected_archive_bytes,
        y1=variant.camera_y1_sha256,
        seg=_sha("changed-seg"),
    )
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="invariant"):
        runner.assert_a_variant_invariants(anchor, (changed,))


def test_measurement_parseback_binds_dense_free_per_pair_scorer_evidence() -> None:
    row = _measurement(
        "scorer-evidence",
        g8=_sha("g8-evidence"),
        a_program=_sha("a-evidence"),
        a_mode=runner.AModeV1.PASS_A_V1,
        a_rows=0,
        d_seg=0.02,
        d_pose=1.5,
        archive_bytes=121,
        y1=_sha("y1-evidence"),
        seg=_sha("seg-evidence"),
    )
    serialized = row.as_dict(archive_path="branches/evidence/selected.not_a_candidate.zip")
    restored = runner.WholeObjectMeasurementV1.from_dict_and_archive(serialized, row.selected_archive_payload)
    assert restored.scorer_evidence.sample_count == 2
    assert restored.scorer_evidence.aggregate_d_seg == restored.d_seg
    assert restored.scorer_evidence.aggregate_d_pose == restored.d_pose
    assert serialized["scorer_evidence"]["dense_frames_logits_rgb_serialized"] is False

    tampered = json.loads(json.dumps(serialized))
    tampered["scorer_evidence"]["per_pair_d_pose"][0] += 0.25
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="aggregate"):
        runner.WholeObjectMeasurementV1.from_dict_and_archive(tampered, row.selected_archive_payload)


def test_all_nondominated_rows_retain_each_series_geometric_neighbors() -> None:
    branches = (
        runner.G8BranchV1("p1", _sha("p1"), "F", "O", 2, 1, object()),
        runner.G8BranchV1("p4", _sha("p4"), "F", "O", 2, 4, object()),
        runner.G8BranchV1("p16", _sha("p16"), "F", "O", 2, 16, object()),
        runner.G8BranchV1("other", _sha("other"), "F2", "O", 2, 4, object()),
    )
    rows = tuple(
        _measurement(
            f"row-{index}",
            g8=branch.program_sha256,
            a_program=_sha(f"pass-{index}"),
            a_mode=runner.AModeV1.PASS_A_V1,
            a_rows=0,
            d_seg=(0.04, 0.02, 0.0195, 0.018)[index],
            d_pose=(4.0, 3.0, 3.2, 4.5)[index],
            archive_bytes=(101, 104, 116, 102)[index],
            y1=_sha(f"y1-{index}"),
            seg=_sha(f"seg-{index}"),
        )
        for index, branch in enumerate(branches)
    )
    retained = runner.retained_g8_branch_ids(branches, rows)
    assert retained == ("p1", "p4", "p16", "other")


def test_nonwinning_nondominated_family_receives_conditional_a_followup(tmp_path: Path) -> None:
    class CrossFamilyBackend(FakeBackend):
        @staticmethod
        def _branch_quality(branch: runner.G8BranchV1) -> tuple[float, float, int]:
            key = (branch.family, branch.prefix_order)
            if key == ("TARGET_PIXEL_RGB_ORACLE_CONTROL_V1", "CANONICAL_ADDRESS_V1"):
                return 0.02, 3.0, 105
            if key == ("CLASS_BOUNDED_TARGET_MEDOIDS_V1", "TARGET_RGB_SSE_DESCENDING_V1"):
                # Nondominated on Seg/rate, but deliberately worse by scalar
                # PASS-A score.  It must still receive A because interaction
                # is unknown before the coupled measurement.
                return 0.015, 4.5, 100
            return 0.04, 5.0, 130

    backend = CrossFamilyBackend()
    result = runner.run_coupled_experiment(
        store=_store(tmp_path),
        backend=backend,
        g8_prefix_mode="1",
        a_prefixes=(0, 1),
    )
    nondominated = set(result["g8_screen"]["nondominated_proposal_ids"])
    scalar_winner = min(
        result["g8_screen"]["measurements"],
        key=lambda row: (row["derived_component_total"], row["measurement_id"]),
    )
    nonwinning = next(
        branch
        for branch in result["g8_screen"]["branches"]
        if branch["proposal_id"] in nondominated and branch["program_sha256"] != scalar_winner["g8_program_sha256"]
    )
    assert nonwinning["proposal_id"] in result["g8_screen"]["retained_for_a_proposal_ids"]
    assert any(
        treatment["g8_branch"]["proposal_id"] == nonwinning["proposal_id"]
        for treatment in result["conditional_a_treatments"]
    )


def test_atomic_store_is_write_once_and_pointer_resume_tolerates_new_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.checkpoint_json("stage", {"value": 1})
    store.checkpoint_json("stage", {"value": 1})
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="overwrite"):
        store.checkpoint_json("stage", {"value": 2})
    resumed = runner.AtomicRunStore.resume(
        store.run_dir,
        expected_stable_contract={"config": "v1", "implementation_sha256": _sha("impl")},
    )
    resumed.append_pointer_snapshot(
        {"pointer_sha256": _sha("new-pointer"), "target_score": 0.16},
        label="resume",
    )
    assert resumed.manifest["pointer_start"]["target_score"] == 0.172
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="custody changed"):
        runner.AtomicRunStore.resume(
            store.run_dir,
            expected_stable_contract={"config": "v2", "implementation_sha256": _sha("impl")},
        )


def test_atomic_partial_recovery_cleans_certified_scratch_and_blocks_foreign_bytes(tmp_path: Path) -> None:
    target = tmp_path / "state" / "row.json"
    target.parent.mkdir(parents=True)
    payload = b'{"complete":true}\n'
    partial = target.parent / f".{target.name}.partial.111"
    partial.write_bytes(payload[:7])
    runner._atomic_write_once_or_equal(target, payload)
    assert target.read_bytes() == payload
    assert not tuple(target.parent.glob(f".{target.name}.partial.*"))

    recovered = target.parent / "recovered.json"
    full_partial = target.parent / f".{recovered.name}.partial.222"
    full_partial.write_bytes(payload)
    runner._atomic_write_once_or_equal(recovered, payload)
    assert recovered.read_bytes() == payload
    assert not full_partial.exists()

    blocked = target.parent / "blocked.json"
    foreign = target.parent / f".{blocked.name}.partial.333"
    foreign.write_bytes(b"foreign-complete-bytes")
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="neither recoverable"):
        runner._atomic_write_once_or_equal(blocked, payload)
    assert foreign.read_bytes() == b"foreign-complete-bytes"


def test_global_copy_extension_is_zero_body_and_not_sparse_emulation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    native = (
        runner.AProgramV1(
            "pass",
            _sha("pass"),
            runner.AModeV1.PASS_A_V1,
            0,
            _sha("y1"),
            object(),
            _sha("rank-pass"),
        ),
    )
    global_copy = runner.AProgramV1(
        "global-copy",
        _sha("global-copy"),
        runner.AModeV1.COUNTED_GLOBAL_COPY_CONDITIONAL_Y1_V1,
        0,
        _sha("y1"),
        object(),
        _sha("rank-global-action"),
    )
    merged = runner.merge_a_programs(native, (global_copy, global_copy))
    assert [program.mode for program in merged] == [
        runner.AModeV1.PASS_A_V1,
        runner.AModeV1.COUNTED_GLOBAL_COPY_CONDITIONAL_Y1_V1,
    ]
    extension = store.register_extension_contract("g13", {"module_sha256": _sha("g13")})
    assert extension.is_file()
    assert "196608" not in json.dumps([program.as_dict() for program in merged])


def test_vertical_experiment_checkpoints_each_branch_and_resumes_without_remeasure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = FakeBackend()
    result = runner.run_coupled_experiment(
        store=store,
        backend=first,
        g8_prefix_mode="cheap",
        a_prefixes=(0, 1),
    )
    assert result["truth"]["research_only"] is True
    assert result["truth"]["candidate_archive_eligible"] is False
    assert result["g8_screen"]["retained_for_a_proposal_ids"]
    assert result["selected_research_row"]["measurement_id"] != "exact-semantic-control"
    assert result["diagnostic_controls"]["production_selection_excluded_measurement_ids"] == ["exact-semantic-control"]
    assert (store.run_dir / "final.selected.not_a_candidate.zip").is_file()
    checkpoints = tuple((store.run_dir / "checkpoints").glob("*.json"))
    archives = tuple((store.run_dir / "branches").glob("*/selected.not_a_candidate.zip"))
    assert len(checkpoints) > len(result["g8_screen"]["measurements"])
    assert len(archives) >= len(result["g8_screen"]["measurements"]) + 2

    resumed_store = runner.AtomicRunStore.resume(
        store.run_dir,
        expected_stable_contract={"config": "v1", "implementation_sha256": _sha("impl")},
    )
    second = FakeBackend()
    resumed = runner.run_coupled_experiment(
        store=resumed_store,
        backend=second,
        g8_prefix_mode="cheap",
        a_prefixes=(0, 1),
    )
    assert resumed["selected_research_row"] == result["selected_research_row"]
    assert not any(
        call.startswith(("baseline", "exact-control", "g8-pass:", "g8-a:", "g0-a:")) for call in second.calls
    )
    assert "acquire-g8" not in second.calls
    assert "restore-g8" in second.calls


def test_resume_blocks_drifted_g0_a_acquisition_before_any_measurement(tmp_path: Path) -> None:
    class CrashAfterG0AcquisitionGate(FakeBackend):
        def measure_g0_a(self, program: runner.AProgramV1) -> runner.WholeObjectMeasurementV1:
            raise RuntimeError(f"simulated crash after stage024 for {program.program_id}")

    store = _store(tmp_path)
    with pytest.raises(RuntimeError, match="simulated crash after stage024"):
        runner.run_coupled_experiment(
            store=store,
            backend=CrashAfterG0AcquisitionGate(),
            g8_prefix_mode="1",
            a_prefixes=(0, 1),
        )
    assert (store.run_dir / "checkpoints/stage_024.g0_a_programs.json").is_file()

    class DriftedG0Acquisition(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.measurement_invoked = False

        def acquire_g0_a(self, row_counts: tuple[int, ...]) -> tuple[runner.AProgramV1, ...]:
            programs = list(super().acquire_g0_a(row_counts))
            programs[1] = replace(programs[1], ranking_sha256=_sha("drifted-g0-ranking"))
            return tuple(programs)

        def measure_g0_a(self, program: runner.AProgramV1) -> runner.WholeObjectMeasurementV1:
            self.measurement_invoked = True
            raise AssertionError(f"drifted G0 descriptor reached measurement: {program.program_id}")

    resumed_store = runner.AtomicRunStore.resume(
        store.run_dir,
        expected_stable_contract={"config": "v1", "implementation_sha256": _sha("impl")},
    )
    drifted = DriftedG0Acquisition()
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match=r"stage_024\.g0_a_programs"):
        runner.run_coupled_experiment(
            store=resumed_store,
            backend=drifted,
            g8_prefix_mode="1",
            a_prefixes=(0, 1),
        )
    assert drifted.measurement_invoked is False


def test_resume_blocks_drifted_branch_a_acquisition_before_any_measurement(tmp_path: Path) -> None:
    class CrashAfterBranchAcquisitionGate(FakeBackend):
        def measure_g8_a(
            self,
            branch: runner.G8BranchV1,
            program: runner.AProgramV1,
        ) -> runner.WholeObjectMeasurementV1:
            raise RuntimeError(f"simulated crash after stage250 for {branch.proposal_id}:{program.program_id}")

    store = _store(tmp_path)
    with pytest.raises(RuntimeError, match="simulated crash after stage250"):
        runner.run_coupled_experiment(
            store=store,
            backend=CrashAfterBranchAcquisitionGate(),
            g8_prefix_mode="1",
            a_prefixes=(0, 1),
        )
    assert tuple((store.run_dir / "checkpoints").glob("stage_250.branch_a_programs.*.json"))

    class DriftedBranchAcquisition(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.measurement_invoked = False

        def acquire_a(
            self,
            branch: runner.G8BranchV1,
            row_counts: tuple[int, ...],
        ) -> tuple[runner.AProgramV1, ...]:
            programs = list(super().acquire_a(branch, row_counts))
            programs[1] = replace(programs[1], ranking_sha256=_sha("drifted-branch-ranking"))
            return tuple(programs)

        def measure_g8_a(
            self,
            branch: runner.G8BranchV1,
            program: runner.AProgramV1,
        ) -> runner.WholeObjectMeasurementV1:
            self.measurement_invoked = True
            raise AssertionError(f"drifted branch descriptor reached measurement: {branch.proposal_id}")

    resumed_store = runner.AtomicRunStore.resume(
        store.run_dir,
        expected_stable_contract={"config": "v1", "implementation_sha256": _sha("impl")},
    )
    drifted = DriftedBranchAcquisition()
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match=r"stage_250\.branch_a_programs"):
        runner.run_coupled_experiment(
            store=resumed_store,
            backend=drifted,
            g8_prefix_mode="1",
            a_prefixes=(0, 1),
        )
    assert drifted.measurement_invoked is False


def test_final_receipt_is_closed_and_latest_pointer_can_differ(tmp_path: Path) -> None:
    store = _store(tmp_path)
    experiment = runner.run_coupled_experiment(
        store=store,
        backend=FakeBackend(),
        g8_prefix_mode="1",
        a_prefixes=(0,),
    )
    latest = {"pointer_sha256": _sha("latest"), "target_score": 0.16}
    payload = runner.finalize_receipt(store=store, experiment=experiment, pointer_latest=latest)
    parsed = runner.parse_final_receipt(payload)
    assert parsed["pointer_start"]["target_score"] == 0.172
    assert parsed["pointer_latest"]["target_score"] == 0.16
    assert parsed["truth"]["pointer_moved"] is False
    serialized = json.dumps(parsed, sort_keys=True)
    assert '"below_target"' not in serialized
    assert '"gap_to_target"' not in serialized
    assert '"target_sublevel_admission"' not in serialized
    evidence_tampered = json.loads(json.dumps(parsed))
    evidence_tampered["baseline"]["scorer_evidence"]["per_pair_d_seg"][0] += 0.125
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="aggregate"):
        runner.parse_final_receipt(runner._canonical_json(evidence_tampered) + b"\n")
    tampered = dict(parsed)
    tampered["truth"] = {**parsed["truth"], "score_claim": True}
    with pytest.raises(runner.TaskspaceG8A3N2AllocatorError, match="authority"):
        runner.parse_final_receipt(runner._canonical_json(tampered) + b"\n")


def test_print_reviewed_command_never_enters_real_path(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def forbidden(_args: Any) -> bytes:
        raise AssertionError("real path must not run")

    monkeypatch.setattr(runner, "_run_reviewed_real", forbidden)
    assert runner.main(["--print-reviewed-command"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["executed"] is False
    assert "--execute-reviewed" in printed["reviewed_command"]


def test_reviewed_resume_command_is_unambiguous_and_preserves_timeout(tmp_path: Path) -> None:
    resume_from = tmp_path / "durable-resume"
    args = runner.build_argument_parser().parse_args(
        [
            "--print-reviewed-command",
            "--resume-from",
            str(resume_from),
            "--timeout-seconds",
            "17.5",
        ]
    )
    command = runner._reviewed_command(args)
    assert "--run-dir" not in command
    assert command[command.index("--resume-from") + 1] == str(resume_from)
    assert command[command.index("--timeout-seconds") + 1] == "17.5"


def test_runtime_environment_and_transitive_real_path_are_stable_contract_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = runner.build_argument_parser().parse_args([])
    monkeypatch.setattr(runner, "_real_input_custody", lambda: {"synthetic_input": _sha("input")})
    torch_was_loaded = "torch" in sys.modules
    baseline = runner._stable_contract(args, git_head_at_start="a" * 40)
    environment = baseline["runtime_environment_custody"]
    assert set(environment) == {
        "schema",
        "python",
        "numpy_module_version",
        "distributions",
        "platform",
        "zlib",
    }
    assert {"numpy", "torch", "timm", "segmentation_models_pytorch"} <= set(environment["distributions"])
    assert ("torch" in sys.modules) is torch_was_loaded

    target = (runner.REPO / "src/tac/score_geometry.py").resolve()
    original_read = runner._read_stable_regular

    def drifted_read(path: Path) -> bytes:
        payload = original_read(path)
        if Path(path).resolve() == target:
            return payload + b"\n# synthetic resume drift"
        return payload

    monkeypatch.setattr(runner, "_read_stable_regular", drifted_read)
    drifted = runner._stable_contract(args, git_head_at_start="a" * 40)
    before_by_path = {row["path"]: row for row in baseline["implementation_custody"]}
    after_by_path = {row["path"]: row for row in drifted["implementation_custody"]}
    assert "src/tac/score_geometry.py" in runner.REAL_PATH_IMPLEMENTATION_CLOSURE
    assert before_by_path["src/tac/score_geometry.py"] != after_by_path["src/tac/score_geometry.py"]
    assert baseline != drifted


def test_exact_r_projection_is_constant_preserving_and_hash_bound() -> None:
    camera = np.full((1, 874, 1164, 3), 137, dtype=np.uint8)
    projected, custody = runner._exact_scorer_plane_rgb(camera)
    assert projected.shape == (1, 384, 512, 3)
    assert projected.dtype == np.uint8
    assert projected.flags.writeable is False
    assert np.all(projected == 137)
    assert custody["camera_frames_sha256"] == runner._array_sha256(camera)
    assert custody["projected_rgb_u8_sha256"] == runner._array_sha256(projected)
    assert custody["rounding"] == "nonnegative_nearest_ties_up.v1"
    assert custody["dense_rgb_serialized"] is False


def test_g10_blocker_is_typed_and_before_real_work(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def blocked() -> dict[str, object]:
        raise runner.G10ProductionCompositionUnavailable("dispatch not frozen")

    monkeypatch.setattr(runner, "require_g10_production_surface", blocked)
    code = runner.main(["--execute-reviewed", "--run-dir", "/var/tmp/durable-g14-test"])
    assert code == 3
    blocker = json.loads(capsys.readouterr().err)
    assert blocker["blocker_code"] == "G10_PRODUCTION_COMPOSITION_UNAVAILABLE"
    assert blocker["before_source_decode"] is True
    assert blocker["before_scorer_load"] is True
    assert blocker["fallback_used"] is False
