"""Tests for ``tac.composition.stack_of_stacks``.

Covers Inner / Middle / Outer composition layers + roundtrip identity +
score-aware mixing + byte-budget enforcement per the operator spec.

Per CLAUDE.md "EVERY training path MUST instantiate EMA, ... eval_roundtrip
... auth eval EVERYWHERE": this test module is BUILD/COMPOSE-time only;
no scorer / training / eval loop runs here. The composer is a deterministic
byte builder; scorer-driven score-aware mixing is exercised through the
``score_aware_mixing_weights`` helper with explicit synthetic component
deltas.
"""

from __future__ import annotations

import pytest

from tac.composition.stack_of_stacks import (
    SCHEMA_VERSION,
    SOS_SIDECAR_MAGIC,
    BoundaryAtomSpec,
    HFSidecarSpec,
    InnerStack,
    InnerStackSpec,
    MiddleStack,
    MiddleStackSpec,
    OuterStack,
    OuterStackSpec,
    ResidualSpec,
    StackOfStacksError,
    compose_stack_of_stacks,
    decompose_stack_of_stacks,
    score_aware_mixing_weights,
    validate_byte_budget,
)
from tac.composition.stack_of_stacks.compose import (
    LAYER_BIT_INNER,
    LAYER_BIT_MIDDLE,
    LAYER_BIT_OUTER,
    MAX_OUTER_ARMS,
    SABOR_BOUNDARY_MAGIC,
    SOS_HEADER_STRUCT,
    S2SBS_HF_MAGIC,
    SCORE_GRAD_RESIDUAL_MAGIC,
)

_HEADER_LEN = SOS_HEADER_STRUCT.size
from tac.composition.stack_of_stacks.inflate import (
    parse_sos_trailer,
    selector_for_pair,
    slice_arm_bytes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_a1_bytes() -> bytes:
    """Synthetic A1 base bytes (178162 B-scale; we use a small stub for tests)."""
    return b"A1_BASE_" + b"\xab" * 1024


@pytest.fixture
def fake_sane_hnerv_bytes() -> bytes:
    """Synthetic sane_hnerv base bytes (different from A1 so we can tell arms apart)."""
    return b"HNERV_BASE_" + b"\xcd" * 1024


# ---------------------------------------------------------------------------
# Inner stack: substrate × SABOR atoms / S2SBS HF / score-grad residual
# ---------------------------------------------------------------------------


class TestInnerStackSpec:
    def test_inner_stack_substrate_alone(self, fake_a1_bytes: bytes) -> None:
        """Substrate-only inner stack adds zero overhead."""
        spec = InnerStackSpec(substrate_id="a1", base_bytes=fake_a1_bytes)
        composed, meta = InnerStack(spec).compose()
        assert composed == fake_a1_bytes
        assert meta["substrate_id"] == "a1"
        assert meta["base_length"] == len(fake_a1_bytes)
        assert meta["added_bytes"] == 0
        assert meta["sub_layers"] == []

    def test_inner_stack_with_sabor_atoms(self, fake_a1_bytes: bytes) -> None:
        """Substrate × SABOR boundary atoms appends an SAB1 sidecar."""
        boundary = BoundaryAtomSpec(
            capacity_bytes=64,
            audit_capacity_bytes=14_600_000,
            atom_payload=b"\xfe" * 64,
        )
        spec = InnerStackSpec(
            substrate_id="a1",
            base_bytes=fake_a1_bytes,
            boundary_atom_spec=boundary,
        )
        composed, meta = InnerStack(spec).compose()
        # Header: 4 magic + 1 ver + 1 reserved + 4 length = 10 B
        assert meta["added_bytes"] == 10 + 64
        assert SABOR_BOUNDARY_MAGIC in composed
        layers = meta["sub_layers"]
        assert len(layers) == 1 and layers[0]["kind"] == "sabor_boundary"

    def test_inner_stack_with_s2sbs_sidecar(self, fake_a1_bytes: bytes) -> None:
        """Substrate × S2SBS HF sidecar appends an SHF1 sidecar."""
        hf = HFSidecarSpec(
            capacity_bytes=128,
            audit_capacity_bytes=38_000_000,
            hf_payload=b"\x7f" * 128,
        )
        spec = InnerStackSpec(
            substrate_id="a1", base_bytes=fake_a1_bytes, hf_sidecar_spec=hf
        )
        composed, meta = InnerStack(spec).compose()
        assert meta["added_bytes"] == 10 + 128
        assert S2SBS_HF_MAGIC in composed
        assert meta["sub_layers"][0]["kind"] == "s2sbs_hf"

    def test_inner_stack_substrate_times_both_sabor_and_s2sbs(self, fake_a1_bytes: bytes) -> None:
        """Substrate × both atoms applies both sidecars deterministically (SABOR before S2SBS)."""
        boundary = BoundaryAtomSpec(
            capacity_bytes=32, audit_capacity_bytes=10000, atom_payload=b"\x01" * 32
        )
        hf = HFSidecarSpec(
            capacity_bytes=48, audit_capacity_bytes=10000, hf_payload=b"\x02" * 48
        )
        spec = InnerStackSpec(
            substrate_id="a1",
            base_bytes=fake_a1_bytes,
            boundary_atom_spec=boundary,
            hf_sidecar_spec=hf,
        )
        composed, meta = InnerStack(spec).compose()
        assert meta["added_bytes"] == (10 + 32) + (10 + 48)
        # Both magics present.
        assert SABOR_BOUNDARY_MAGIC in composed
        assert S2SBS_HF_MAGIC in composed
        # SABOR comes before S2SBS deterministically.
        sabor_idx = composed.index(SABOR_BOUNDARY_MAGIC)
        s2sbs_idx = composed.index(S2SBS_HF_MAGIC)
        assert sabor_idx < s2sbs_idx

    def test_inner_stack_with_residual(self, fake_a1_bytes: bytes) -> None:
        """Substrate × score-grad residual appends an SGR1 sidecar."""
        residual = ResidualSpec(residual_int8_bytes=b"\x10" * 96, scale=4.0)
        spec = InnerStackSpec(
            substrate_id="a1", base_bytes=fake_a1_bytes, residual_spec=residual
        )
        composed, meta = InnerStack(spec).compose()
        # 10 B header + 4 B scale + 96 B residual
        assert meta["added_bytes"] == 10 + 4 + 96
        assert SCORE_GRAD_RESIDUAL_MAGIC in composed

    def test_inner_stack_with_all_three_layers(self, fake_a1_bytes: bytes) -> None:
        """Substrate × SABOR × S2SBS × residual layers ALL three sidecars in order."""
        spec = InnerStackSpec(
            substrate_id="a1",
            base_bytes=fake_a1_bytes,
            boundary_atom_spec=BoundaryAtomSpec(
                capacity_bytes=16, audit_capacity_bytes=10000
            ),
            hf_sidecar_spec=HFSidecarSpec(
                capacity_bytes=24, audit_capacity_bytes=10000
            ),
            residual_spec=ResidualSpec(residual_int8_bytes=b"\x00" * 32, scale=2.0),
        )
        composed, meta = InnerStack(spec).compose()
        assert len(meta["sub_layers"]) == 3
        kinds = [layer["kind"] for layer in meta["sub_layers"]]
        assert kinds == ["sabor_boundary", "s2sbs_hf", "score_grad_residual"]

    def test_inner_stack_refuses_capacity_exceeding_audit(self) -> None:
        """SABOR capacity_bytes > audit_capacity_bytes raises."""
        with pytest.raises(StackOfStacksError, match="exceeds.*audit_capacity_bytes"):
            BoundaryAtomSpec(capacity_bytes=200, audit_capacity_bytes=100)

    def test_inner_stack_refuses_payload_exceeding_capacity(self) -> None:
        """SABOR payload longer than capacity raises."""
        with pytest.raises(StackOfStacksError, match="exceeds capacity_bytes"):
            BoundaryAtomSpec(
                capacity_bytes=10, audit_capacity_bytes=100, atom_payload=b"\x00" * 20
            )

    def test_inner_stack_pads_payload_to_capacity(self, fake_a1_bytes: bytes) -> None:
        """SABOR atom_payload shorter than capacity_bytes pads with zeros."""
        boundary = BoundaryAtomSpec(
            capacity_bytes=64, audit_capacity_bytes=10000, atom_payload=b"\xff" * 16
        )
        spec = InnerStackSpec(
            substrate_id="a1", base_bytes=fake_a1_bytes, boundary_atom_spec=boundary
        )
        composed, meta = InnerStack(spec).compose()
        # Total added = 10 B header + 64 B padded payload
        assert meta["added_bytes"] == 74

    def test_inner_stack_refuses_negative_capacity(self) -> None:
        """Negative capacity rejected at spec build time."""
        with pytest.raises(StackOfStacksError, match="must be >= 0"):
            BoundaryAtomSpec(capacity_bytes=-1, audit_capacity_bytes=100)

    def test_inner_stack_refuses_invalid_residual_scale(self) -> None:
        """ResidualSpec scale must be positive finite."""
        with pytest.raises(StackOfStacksError, match="positive finite"):
            ResidualSpec(residual_int8_bytes=b"\x00", scale=0.0)
        with pytest.raises(StackOfStacksError, match="positive finite"):
            ResidualSpec(residual_int8_bytes=b"\x00", scale=float("inf"))

    def test_inner_stack_refuses_empty_base(self) -> None:
        with pytest.raises(StackOfStacksError, match="non-empty"):
            InnerStackSpec(substrate_id="a1", base_bytes=b"")
        with pytest.raises(StackOfStacksError, match="non-empty"):
            InnerStackSpec(substrate_id="", base_bytes=b"x")


# ---------------------------------------------------------------------------
# Middle stack: A1 + LAPose / A1 + wavelet / A1 + LAPose + wavelet
# ---------------------------------------------------------------------------


class TestMiddleStack:
    def test_middle_stack_a1_plus_lapose(self, fake_a1_bytes: bytes) -> None:
        """Single-arm middle stack with an A1+LAPose-style inner."""
        # We model the LAPose sidecar as an inner residual (the canonical
        # LAPose sidecar is a substrate-specific archive; the middle stack
        # accepts the substrate-formatted base as opaque bytes).
        a1_lapose_base = fake_a1_bytes + b"LPA1" + b"\x00" * 100  # fake A1+LAPose archive
        inner = InnerStackSpec(substrate_id="a1_plus_lapose", base_bytes=a1_lapose_base)
        middle = MiddleStack(MiddleStackSpec(inner_specs=(inner,)))
        arm_bytes_list, arm_meta_list = middle.compose_arms()
        assert len(arm_bytes_list) == 1
        assert arm_bytes_list[0] == a1_lapose_base

    def test_middle_stack_a1_plus_wavelet(self, fake_a1_bytes: bytes) -> None:
        """Single-arm middle stack with an A1+wavelet-style inner."""
        a1_wavelet_base = fake_a1_bytes + b"WAV1" + b"\x00" * 50
        inner = InnerStackSpec(
            substrate_id="a1_plus_wavelet", base_bytes=a1_wavelet_base
        )
        middle = MiddleStack(MiddleStackSpec(inner_specs=(inner,)))
        arm_bytes_list, _ = middle.compose_arms()
        assert arm_bytes_list[0] == a1_wavelet_base

    def test_middle_stack_lapose_and_wavelet_arms(self, fake_a1_bytes: bytes) -> None:
        """Two-arm middle stack — LAPose arm + wavelet arm."""
        lapose_base = fake_a1_bytes + b"LPA1" + b"\x00" * 100
        wavelet_base = fake_a1_bytes + b"WAV1" + b"\x00" * 50
        spec = MiddleStackSpec(
            inner_specs=(
                InnerStackSpec(substrate_id="a1_plus_lapose", base_bytes=lapose_base),
                InnerStackSpec(substrate_id="a1_plus_wavelet", base_bytes=wavelet_base),
            )
        )
        arm_bytes_list, arm_meta_list = MiddleStack(spec).compose_arms()
        assert len(arm_bytes_list) == 2
        assert arm_meta_list[0]["substrate_id"] == "a1_plus_lapose"
        assert arm_meta_list[1]["substrate_id"] == "a1_plus_wavelet"

    def test_middle_stack_three_arms(self, fake_a1_bytes: bytes, fake_sane_hnerv_bytes: bytes) -> None:
        """Three-arm middle stack — MAX_OUTER_ARMS = 3 limit honored."""
        spec = MiddleStackSpec(
            inner_specs=(
                InnerStackSpec(substrate_id="a1", base_bytes=fake_a1_bytes),
                InnerStackSpec(substrate_id="sane_hnerv", base_bytes=fake_sane_hnerv_bytes),
                InnerStackSpec(substrate_id="a1_alt", base_bytes=fake_a1_bytes + b"_alt"),
            )
        )
        arm_bytes_list, _ = MiddleStack(spec).compose_arms()
        assert len(arm_bytes_list) == 3

    def test_middle_stack_refuses_more_than_max_arms(self, fake_a1_bytes: bytes) -> None:
        """4 arms is over the cap."""
        with pytest.raises(StackOfStacksError, match=f"must be ≤ {MAX_OUTER_ARMS}"):
            MiddleStackSpec(
                inner_specs=tuple(
                    InnerStackSpec(substrate_id=f"a{i}", base_bytes=fake_a1_bytes)
                    for i in range(MAX_OUTER_ARMS + 1)
                )
            )

    def test_middle_stack_refuses_empty(self) -> None:
        with pytest.raises(StackOfStacksError, match="non-empty tuple"):
            MiddleStackSpec(inner_specs=())

    def test_middle_stack_rate_budget_partition_enforced(self, fake_a1_bytes: bytes) -> None:
        """rate_budget_partition refuses arms that exceed their budget."""
        boundary = BoundaryAtomSpec(
            capacity_bytes=1024, audit_capacity_bytes=10000, atom_payload=b"\x01" * 1024
        )
        inner = InnerStackSpec(
            substrate_id="a1", base_bytes=fake_a1_bytes, boundary_atom_spec=boundary
        )
        with pytest.raises(StackOfStacksError, match="added .* but budget is"):
            MiddleStackSpec(
                inner_specs=(inner,),
                rate_budget_partition={"a1": 100},
            )


# ---------------------------------------------------------------------------
# Outer stack: K=3 ensemble, per-pair selector, no-op-controllable
# ---------------------------------------------------------------------------


class TestOuterStack:
    def test_outer_stack_k1_no_ensemble(self) -> None:
        """K=1 outer stack adds the SOS1 trailer but selects arm 0 uniformly."""
        spec = OuterStackSpec(k=1)
        stack = OuterStack(spec, n_pairs=10)
        trailer = stack.pack([{"substrate_id": "a1", "arm_offset": 0, "arm_length": 50}], 0)
        # Header + 10 selector bytes + brotli meta (at least 1 B)
        assert len(trailer) >= _HEADER_LEN + 10 + 1
        # Selector should be all zeros for K=1.
        selector = trailer[_HEADER_LEN : _HEADER_LEN + 10]
        assert selector == b"\x00" * 10

    def test_outer_stack_k3_per_pair_selector(self) -> None:
        """K=3 outer stack stores per-pair arm indices."""
        per_pair = tuple([0, 1, 2, 0, 1, 2, 1, 0, 2, 1])
        spec = OuterStackSpec(k=3, per_pair_arm=per_pair, temperatures=(1.0, 0.5, 0.1))
        stack = OuterStack(spec, n_pairs=len(per_pair))
        trailer = stack.pack(
            [
                {"substrate_id": f"arm_{i}", "arm_offset": 0, "arm_length": 100}
                for i in range(3)
            ],
            LAYER_BIT_OUTER,
        )
        selector_bytes = trailer[_HEADER_LEN : _HEADER_LEN + 10]
        assert tuple(selector_bytes) == per_pair

    def test_outer_stack_refuses_invalid_k(self) -> None:
        with pytest.raises(StackOfStacksError, match="must be in"):
            OuterStackSpec(k=0)
        with pytest.raises(StackOfStacksError, match="must be in"):
            OuterStackSpec(k=MAX_OUTER_ARMS + 1)

    def test_outer_stack_refuses_out_of_range_arm(self) -> None:
        with pytest.raises(StackOfStacksError, match="must be in"):
            OuterStackSpec(k=2, per_pair_arm=(0, 1, 2))  # 2 invalid when k=2

    def test_outer_stack_refuses_pair_count_mismatch(self) -> None:
        spec = OuterStackSpec(k=2, per_pair_arm=(0, 1, 0))
        with pytest.raises(StackOfStacksError, match="!= n_pairs"):
            OuterStack(spec, n_pairs=5)

    def test_outer_stack_is_no_op_controllable_when_per_pair_is_empty(self) -> None:
        """OuterStackSpec with empty per_pair_arm defaults to arm 0 (no-op)."""
        spec = OuterStackSpec(k=2)  # K=2 enabled but no selection: all arm 0
        stack = OuterStack(spec, n_pairs=5)
        trailer = stack.pack(
            [
                {"substrate_id": "a1", "arm_offset": 0, "arm_length": 1},
                {"substrate_id": "b1", "arm_offset": 0, "arm_length": 1},
            ],
            LAYER_BIT_OUTER,
        )
        selector = trailer[_HEADER_LEN : _HEADER_LEN + 5]
        assert selector == b"\x00\x00\x00\x00\x00"


# ---------------------------------------------------------------------------
# Composition byte-budget enforcement
# ---------------------------------------------------------------------------


class TestByteBudgetEnforcement:
    def test_validate_byte_budget_passes_when_within_cap(self, fake_a1_bytes: bytes) -> None:
        inner = InnerStackSpec(substrate_id="a1", base_bytes=fake_a1_bytes)
        middle_spec = MiddleStackSpec(inner_specs=(inner,))
        result = validate_byte_budget(
            middle_spec,
            base_substrate_bytes=len(fake_a1_bytes),
            max_total_bytes=10_000,
        )
        assert result["predicted_total_bytes"] < 10_000
        assert result["k"] == 1

    def test_validate_byte_budget_refuses_over_cap(self, fake_a1_bytes: bytes) -> None:
        boundary = BoundaryAtomSpec(
            capacity_bytes=8000, audit_capacity_bytes=10000, atom_payload=b"\x01" * 8000
        )
        inner = InnerStackSpec(
            substrate_id="a1", base_bytes=fake_a1_bytes, boundary_atom_spec=boundary
        )
        middle_spec = MiddleStackSpec(inner_specs=(inner,))
        with pytest.raises(StackOfStacksError, match="exceeds.*max_total_bytes"):
            validate_byte_budget(
                middle_spec,
                base_substrate_bytes=len(fake_a1_bytes),
                max_total_bytes=2_000,
            )

    def test_compose_stack_of_stacks_enforces_max_total_bytes(self, fake_a1_bytes: bytes) -> None:
        """The top-level composer rejects archives over max_total_bytes."""
        inner = InnerStackSpec(substrate_id="a1", base_bytes=fake_a1_bytes)
        with pytest.raises(StackOfStacksError, match="exceeds max_total_bytes"):
            compose_stack_of_stacks(
                middle_stack_spec=MiddleStackSpec(inner_specs=(inner,)),
                max_total_bytes=100,  # absurdly small
            )


# ---------------------------------------------------------------------------
# Round-trip identity (compose → decompose → verify)
# ---------------------------------------------------------------------------


class TestRoundtripIdentity:
    def test_roundtrip_single_arm_single_layer(self, fake_a1_bytes: bytes) -> None:
        """Compose A1-only single arm, decompose, verify base bytes recovered."""
        inner = InnerStackSpec(substrate_id="a1", base_bytes=fake_a1_bytes)
        composed, meta = compose_stack_of_stacks(
            middle_stack_spec=MiddleStackSpec(inner_specs=(inner,)),
            n_pairs=10,
        )
        decoded = decompose_stack_of_stacks(composed)
        assert decoded["k"] == 1
        assert decoded["n_pairs"] == 10
        assert decoded["arm_concat_bytes"] == fake_a1_bytes
        assert decoded["arm_meta"]["arms"][0]["substrate_id"] == "a1"

    def test_roundtrip_two_arm_with_inner_sidecars(self, fake_a1_bytes: bytes) -> None:
        """Two-arm + inner SABOR + S2SBS, recover via inflate parser."""
        boundary = BoundaryAtomSpec(
            capacity_bytes=64, audit_capacity_bytes=14_600_000, atom_payload=b"\x12" * 64
        )
        hf = HFSidecarSpec(
            capacity_bytes=80, audit_capacity_bytes=38_000_000, hf_payload=b"\x34" * 80
        )
        inner_a = InnerStackSpec(
            substrate_id="a1_plus_lapose",
            base_bytes=fake_a1_bytes,
            boundary_atom_spec=boundary,
        )
        inner_b = InnerStackSpec(
            substrate_id="a1_plus_wavelet",
            base_bytes=fake_a1_bytes,
            hf_sidecar_spec=hf,
        )
        composed, meta = compose_stack_of_stacks(
            middle_stack_spec=MiddleStackSpec(inner_specs=(inner_a, inner_b)),
            outer_stack_spec=OuterStackSpec(
                k=2,
                per_pair_arm=tuple([0, 1] * 5),
                temperatures=(1.0, 0.5),
            ),
            n_pairs=10,
        )
        # Inflate-time parser recovers structure
        parsed = parse_sos_trailer(composed)
        assert parsed["k"] == 2
        assert parsed["n_pairs"] == 10
        assert parsed["selector"] == bytes([0, 1] * 5)
        # Both arms recoverable
        arm0 = slice_arm_bytes(parsed, 0)
        arm1 = slice_arm_bytes(parsed, 1)
        assert arm0.startswith(fake_a1_bytes)
        assert arm1.startswith(fake_a1_bytes)
        assert SABOR_BOUNDARY_MAGIC in arm0
        assert S2SBS_HF_MAGIC in arm1

    def test_roundtrip_k3_ensemble(self, fake_a1_bytes: bytes, fake_sane_hnerv_bytes: bytes) -> None:
        """K=3 outer stack with three distinct substrate arms."""
        inners = (
            InnerStackSpec(substrate_id="a1", base_bytes=fake_a1_bytes),
            InnerStackSpec(substrate_id="sane_hnerv", base_bytes=fake_sane_hnerv_bytes),
            InnerStackSpec(substrate_id="a1_alt", base_bytes=fake_a1_bytes + b"_alt"),
        )
        composed, _ = compose_stack_of_stacks(
            middle_stack_spec=MiddleStackSpec(inner_specs=inners),
            outer_stack_spec=OuterStackSpec(
                k=3,
                per_pair_arm=tuple([0, 1, 2, 0, 1, 2]),
                temperatures=(1.0, 0.3, 0.1),
            ),
            n_pairs=6,
        )
        parsed = parse_sos_trailer(composed)
        assert parsed["k"] == 3
        assert tuple(parsed["selector"]) == (0, 1, 2, 0, 1, 2)
        assert selector_for_pair(parsed, 0) == 0
        assert selector_for_pair(parsed, 5) == 2
        # Each arm slice contains its respective base bytes.
        assert slice_arm_bytes(parsed, 0) == fake_a1_bytes
        assert slice_arm_bytes(parsed, 1) == fake_sane_hnerv_bytes
        assert slice_arm_bytes(parsed, 2) == fake_a1_bytes + b"_alt"

    def test_decompose_refuses_bad_magic(self, fake_a1_bytes: bytes) -> None:
        """Decompose raises if SOS1 magic missing."""
        with pytest.raises(StackOfStacksError, match="no SOS1 magic"):
            decompose_stack_of_stacks(fake_a1_bytes)

    def test_decompose_refuses_truncated_trailer(self, fake_a1_bytes: bytes) -> None:
        """Truncated trailer raises a structured error."""
        inner = InnerStackSpec(substrate_id="a1", base_bytes=fake_a1_bytes)
        composed, _ = compose_stack_of_stacks(
            middle_stack_spec=MiddleStackSpec(inner_specs=(inner,)),
            n_pairs=5,
        )
        truncated = composed[:-3]  # chop the last 3 bytes
        with pytest.raises(StackOfStacksError):
            decompose_stack_of_stacks(truncated)


# ---------------------------------------------------------------------------
# Score-aware mixing rules
# ---------------------------------------------------------------------------


class TestScoreAwareMixing:
    def test_score_aware_picks_best_per_pair(self) -> None:
        """Best-of-K selector picks the arm with the lowest contest contribution."""
        # 4 pairs, 2 arms. Arm 0 better on seg, arm 1 better on pose.
        seg = [
            [0.10, 0.20],  # pair 0: arm 0 better on seg
            [0.05, 0.30],  # pair 1: arm 0 dominant
            [0.10, 0.05],  # pair 2: arm 1 better on seg
            [0.50, 0.50],  # pair 3: tied on seg → rate breaks tie
        ]
        pose = [
            [1e-4, 1e-6],  # pair 0: arm 1 much better on pose
            [1e-5, 1e-4],  # pair 1: arm 0 better on pose too
            [1e-4, 1e-4],  # pair 2: tied on pose
            [1e-4, 1e-5],  # pair 3: arm 1 better on pose
        ]
        rates = [100, 200]  # arm 1 uses more bytes

        sel = score_aware_mixing_weights(seg, pose, rates)
        assert len(sel) == 4
        # pair 1: arm 0 dominant on both, so arm 0.
        assert sel[1] == 0
        # pair 2: arm 1 better on seg (0.05 vs 0.10) by 5 contest-CPU sensitivity units,
        # arm 1 burns 100 B more (rate cost 100 * 25 / 37545489 ≈ 6.66e-5);
        # arm 1 selected.
        assert sel[2] == 1

    def test_score_aware_picks_lower_rate_when_tied(self) -> None:
        """When all distortion components match, lower-rate arm wins."""
        seg = [[0.1, 0.1]]
        pose = [[1e-5, 1e-5]]
        rates = [100, 50]  # arm 1 cheaper
        sel = score_aware_mixing_weights(seg, pose, rates)
        assert sel == [1]

    def test_score_aware_refuses_empty(self) -> None:
        with pytest.raises(StackOfStacksError, match="non-empty"):
            score_aware_mixing_weights([], [], [])

    def test_score_aware_refuses_arm_length_mismatch(self) -> None:
        with pytest.raises(StackOfStacksError, match="must equal k"):
            score_aware_mixing_weights([[0.1, 0.2]], [[1e-5]], [10, 20])

    def test_score_aware_handles_three_arms(self) -> None:
        """K=3 with explicit dominant arm pattern."""
        seg = [
            [0.1, 0.05, 0.2],  # arm 1 best
            [0.2, 0.3, 0.05],  # arm 2 best
            [0.05, 0.2, 0.3],  # arm 0 best
        ]
        pose = [[1e-5, 1e-5, 1e-5]] * 3
        rates = [0, 0, 0]
        sel = score_aware_mixing_weights(seg, pose, rates)
        assert sel == [1, 2, 0]


# ---------------------------------------------------------------------------
# Layer-mask discipline
# ---------------------------------------------------------------------------


class TestLayerMask:
    def test_layer_mask_inner_only(self, fake_a1_bytes: bytes) -> None:
        """K=1, single arm with SABOR inner sidecar → INNER bit only."""
        inner = InnerStackSpec(
            substrate_id="a1",
            base_bytes=fake_a1_bytes,
            boundary_atom_spec=BoundaryAtomSpec(
                capacity_bytes=64, audit_capacity_bytes=10000
            ),
        )
        composed, meta = compose_stack_of_stacks(
            middle_stack_spec=MiddleStackSpec(inner_specs=(inner,)),
            n_pairs=5,
        )
        assert meta["layer_mask"] & LAYER_BIT_INNER
        assert not (meta["layer_mask"] & LAYER_BIT_MIDDLE)
        assert not (meta["layer_mask"] & LAYER_BIT_OUTER)

    def test_layer_mask_middle_and_outer(self, fake_a1_bytes: bytes) -> None:
        """K=2 arms with no inner sidecars → MIDDLE + OUTER bits, no INNER."""
        composed, meta = compose_stack_of_stacks(
            middle_stack_spec=MiddleStackSpec(
                inner_specs=(
                    InnerStackSpec(substrate_id="a1", base_bytes=fake_a1_bytes),
                    InnerStackSpec(substrate_id="a1_alt", base_bytes=fake_a1_bytes + b"_b"),
                )
            ),
            outer_stack_spec=OuterStackSpec(
                k=2, per_pair_arm=tuple([0, 1] * 3), temperatures=(1.0, 0.5)
            ),
            n_pairs=6,
        )
        assert not (meta["layer_mask"] & LAYER_BIT_INNER)
        assert meta["layer_mask"] & LAYER_BIT_MIDDLE
        assert meta["layer_mask"] & LAYER_BIT_OUTER


# ---------------------------------------------------------------------------
# Schema / version stability
# ---------------------------------------------------------------------------


def test_schema_version_pinned_to_1() -> None:
    """Schema version must stay at 1 until a versioned migration lands."""
    assert SCHEMA_VERSION == 1


def test_sos_magic_is_correct() -> None:
    """SOS1 magic bytes are exactly the published 4-byte token."""
    assert SOS_SIDECAR_MAGIC == b"SOS1"
    assert len(SOS_SIDECAR_MAGIC) == 4


def test_inflate_module_uses_same_magic_as_compose() -> None:
    """Inflate and compose share the SOS1 magic constant."""
    from tac.composition.stack_of_stacks.inflate import (
        SOS_SIDECAR_MAGIC as INFL_MAGIC,
    )

    assert INFL_MAGIC == SOS_SIDECAR_MAGIC
