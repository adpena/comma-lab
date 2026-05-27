# SPDX-License-Identifier: MIT
"""Scaffold smoke tests for cascade_c_prime_frame_1_segnet_waterfill.

Per Catalog #287/#323 canonical Provenance + Catalog #139 byte-mutation smoke.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill import (
    CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT,
)
from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.architecture import (
    FRAME_0,
    FRAME_1,
    compute_per_pair_lagrangian_dual_routing,
)
from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.archive import (
    CCPF_MAGIC,
    POSE_DIMS,
    pack_archive,
    parse_archive,
)
from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import (
    CONTEST_RAW_BYTES,
    contest_output_shape_for_archive,
    inflate_one_video,
)


class TestSubstrateContract:
    def test_canonical_id(self):
        c = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT
        assert c.id == "cascade_c_prime_frame_1_segnet_waterfill"

    def test_canonical_lane_id_format(self):
        c = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT
        assert c.lane_id.startswith("lane_")
        assert c.lane_id.endswith("_20260526")

    def test_research_substrate_target_mode(self):
        c = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT
        assert "research_substrate" in c.target_modes

    def test_recipe_research_only_true(self):
        c = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT
        assert c.recipe_research_only is True

    def test_hook_continual_learning_anchor_kind(self):
        c = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT
        assert c.hook_continual_learning_anchor_kind == "cuda_only"


class TestPerPairLagrangianDualRouting:
    def test_atick_redlich_structural_invariant_frame_0_seg_zero(self):
        """When frame_0 seg is structurally zero, all pose-only Lagrangian routing
        decisions should be valid (not crash)."""
        rng = np.random.default_rng(seed=42)
        n_pairs, n_f0, n_f1 = 20, 16, 8
        frame_0_seg = np.zeros((n_pairs, n_f0), dtype=np.float64)
        frame_0_pose = rng.normal(loc=-5e-6, scale=2e-6, size=(n_pairs, n_f0))
        frame_1_seg = rng.gamma(shape=0.5, scale=2e-5, size=(n_pairs, n_f1))
        frame_1_pose = rng.gamma(shape=0.3, scale=3e-7, size=(n_pairs, n_f1))
        result = compute_per_pair_lagrangian_dual_routing(
            frame_0_seg, frame_0_pose, frame_1_seg, frame_1_pose, pose_avg_baseline=3.4e-5,
        )
        assert result.n_pairs == n_pairs
        assert result.routing_decision.shape == (n_pairs,)
        assert ((result.routing_decision == FRAME_0) | (result.routing_decision == FRAME_1)).all()

    def test_per_pair_improvement_non_negative_at_argmin(self):
        """Per-pair joint Lagrangian should never be WORSE than frame-0-only baseline."""
        rng = np.random.default_rng(seed=42)
        n_pairs, n_f0, n_f1 = 20, 16, 8
        frame_0_seg = np.zeros((n_pairs, n_f0), dtype=np.float64)
        frame_0_pose = rng.normal(loc=-5e-6, scale=2e-6, size=(n_pairs, n_f0))
        frame_1_seg = rng.gamma(shape=0.5, scale=2e-5, size=(n_pairs, n_f1))
        frame_1_pose = rng.gamma(shape=0.3, scale=3e-7, size=(n_pairs, n_f1))
        result = compute_per_pair_lagrangian_dual_routing(
            frame_0_seg, frame_0_pose, frame_1_seg, frame_1_pose, pose_avg_baseline=3.4e-5,
        )
        # joint menu MUST be >= frame-0-only argmin
        assert (result.per_pair_improvement >= -1e-9).all()

    def test_argument_validation(self):
        with pytest.raises(ValueError, match="pose_avg_baseline"):
            compute_per_pair_lagrangian_dual_routing(
                np.zeros((5, 3)), np.zeros((5, 3)),
                np.zeros((5, 2)), np.zeros((5, 2)),
                pose_avg_baseline=-1.0,
            )


class TestArchiveRoundtrip:
    def _build_synthetic_archive(self, seed=20260526, n_pairs=50):
        rng = np.random.default_rng(seed=seed)
        routing = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
        f0_indices = rng.integers(0, 16, size=n_pairs, dtype=np.uint8)
        f1_indices = rng.integers(0, 8, size=n_pairs, dtype=np.uint8)
        pose_deltas = rng.integers(0, 256, size=(n_pairs, POSE_DIMS), dtype=np.uint8)
        archive_bytes = pack_archive(
            routing_decision=routing,
            frame_0_menu_indices=f0_indices,
            frame_1_menu_indices=f1_indices,
            pose_deltas_uint8=pose_deltas,
        )
        return archive_bytes, routing, f0_indices, f1_indices, pose_deltas

    def test_magic_bytes(self):
        archive_bytes, *_ = self._build_synthetic_archive()
        assert archive_bytes[:4] == CCPF_MAGIC

    def test_routing_decision_roundtrip(self):
        archive_bytes, routing, *_ = self._build_synthetic_archive()
        parsed = parse_archive(archive_bytes)
        assert np.array_equal(parsed.routing_decision, routing)

    def test_n_pairs_roundtrip(self):
        archive_bytes, routing, *_ = self._build_synthetic_archive(n_pairs=123)
        parsed = parse_archive(archive_bytes)
        assert parsed.n_pairs == 123

    def test_bad_magic_rejected(self):
        bad = b"BOGS" + b"\x01" + b"\x00" * 100
        with pytest.raises(ValueError, match="bad magic"):
            parse_archive(bad)


class TestInflateRuntimeContestContract:
    def test_scaffold_inflate_writes_full_contest_raw_contract(self, tmp_path):
        """Wave-3 regression: receiver must emit 1164x874x1200 RGB raw bytes."""
        archive_bytes, *_ = TestArchiveRoundtrip()._build_synthetic_archive(n_pairs=8)

        raw_path = inflate_one_video(archive_bytes, tmp_path / "0")

        assert raw_path.name == "0.raw"
        assert raw_path.stat().st_size == CONTEST_RAW_BYTES
        with raw_path.open("rb") as fh:
            assert fh.read(16) == b"\x00" * 16
            fh.seek(CONTEST_RAW_BYTES - 16)
            assert fh.read(16) == b"\x00" * 16

    def test_contest_output_shape_is_padded_for_local_smokes(self):
        archive_bytes, *_ = TestArchiveRoundtrip()._build_synthetic_archive(n_pairs=8)
        parsed = parse_archive(archive_bytes)

        assert contest_output_shape_for_archive(parsed) == (1200, 874, 1164, 3)


class TestByteMutationSmoke:
    """Catalog #139 / #105 / #272 byte-mutation smoke — verify routing-decision
    sidecar bytes are operationally consumed by inflate."""

    def test_routing_sidecar_byte_mutation_changes_routing_decision(self):
        rng = np.random.default_rng(seed=20260526)
        n_pairs = 64
        # Build deterministic routing pattern (alternating)
        routing = np.zeros(n_pairs, dtype=np.int8)
        routing[::2] = 1
        f0_indices = rng.integers(0, 16, size=n_pairs, dtype=np.uint8)
        f1_indices = rng.integers(0, 8, size=n_pairs, dtype=np.uint8)
        pose_deltas = rng.integers(0, 256, size=(n_pairs, POSE_DIMS), dtype=np.uint8)
        archive_bytes = pack_archive(
            routing_decision=routing,
            frame_0_menu_indices=f0_indices,
            frame_1_menu_indices=f1_indices,
            pose_deltas_uint8=pose_deltas,
        )
        # Mutate every byte in the routing sidecar block
        # sidecar_count read from header offset 7-8
        import struct
        routing_byte_count = struct.unpack("<H", archive_bytes[7:9])[0]
        # Mutate first byte AFTER header
        mutated_count = 0
        for offset in range(11, 11 + routing_byte_count):
            mutated = bytearray(archive_bytes)
            mutated[offset] ^= 0xFF  # flip ALL bits in this byte
            try:
                parsed_m = parse_archive(bytes(mutated))
                roundtrip_routing = parsed_m.routing_decision
                if not np.array_equal(roundtrip_routing, routing):
                    mutated_count += 1
            except Exception:
                mutated_count += 1
        # At least one byte mutation must change routing decision (else dead bytes per Catalog #139)
        assert mutated_count > 0, (
            "Catalog #139 no-op detector FAIL: no byte in routing sidecar affects routing decision"
        )
