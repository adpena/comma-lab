# SPDX-License-Identifier: MIT
"""MLX-first trainer tests for cascade_c_prime_frame_1_segnet_waterfill.

Per Catalog #287/#323 canonical Provenance + Catalog #139 byte-mutation sister
+ Catalog #229 premise verification: tests cover trainer config invariants,
MLX-native primitive verification (no torch imports), MLX→numpy bridge
round-trip, and Tier-C ablation hook smoke.

Per per-substrate symposium PROCEED_WITH_REVISIONS verdict (commit
``aaf0b1eb6``): all score-axis fields verified non-promotable per
MLX_NON_PROMOTABLE_PROVENANCE; promotion to ``[contest-CUDA]`` defers to
sister subagent C paired-CUDA Modal smoke per revision #3.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill import (
    CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT,
)
from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.mlx_to_numpy_bridge import (
    EXPECTED_STATE_DICT_KEYS,
    BridgeRoundtripVerdict,
    MLXNumpyBridgeError,
    export_state_dict_to_npz,
    load_state_dict_from_npz,
    roundtrip_state_dict_through_archive,
    verify_state_dict_shape_contract,
)
from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.tier_c_hook import (
    DEFAULT_TIER_C_TOOL_PATH,
    FORMALIZATION_PENDING_VERDICT,
    TierCAblationHookVerdict,
    build_tier_c_ablation_probe_request,
    classify_tier_c_density_verdict,
)
from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.trainer import (
    DEFAULT_MLX_AXIS_TAG,
    MLX_NON_PROMOTABLE_PROVENANCE,
    MLXFirstTrainerConfig,
    MLXFirstTrainerVerdict,
    is_mlx_available,
    run_mlx_first_compress_pass,
)


def _mlx_available() -> bool:
    """Sister of trainer.is_mlx_available; used for skip decorators."""
    return is_mlx_available()


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (
            parent
            / "experiments"
            / "train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py"
        ).exists():
            return parent
    raise AssertionError("repo root not found")


def _trainer_wrapper_source() -> str:
    return (
        _repo_root()
        / "experiments"
        / "train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py"
    ).read_text(encoding="utf-8")


def _load_trainer_wrapper():
    path = (
        _repo_root()
        / "experiments"
        / "train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cascade_c_prime_frame_1_segnet_waterfill_trainer_wrapper",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# MLX-native primitive verification (no torch imports)
# ---------------------------------------------------------------------------


class TestMLXFirstNoTorchImports:
    """Per CLAUDE.md standing directive 2026-05-26 'MLX-FIRST': verify the
    trainer module has ZERO torch imports."""

    def test_trainer_module_does_not_import_torch(self):
        from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill import (
            trainer as trainer_mod,
        )
        source = Path(trainer_mod.__file__).read_text()
        # Forbidden: top-level torch imports (allows mentioning 'torch' in
        # docstrings + comments per Catalog #287 self-waiver-pattern).
        forbidden_patterns = (
            "\nimport torch\n",
            "\nimport torch.",
            "\nfrom torch ",
            "\nfrom torch.",
        )
        for pat in forbidden_patterns:
            assert pat not in source, (
                f"trainer.py contains forbidden torch import pattern {pat!r} "
                "per MLX-FIRST canonical doctrine"
            )

    def test_bridge_module_does_not_import_torch(self):
        from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill import (
            mlx_to_numpy_bridge as bridge_mod,
        )
        source = Path(bridge_mod.__file__).read_text()
        forbidden_patterns = (
            "\nimport torch\n",
            "\nimport torch.",
            "\nfrom torch ",
            "\nfrom torch.",
        )
        for pat in forbidden_patterns:
            assert pat not in source, (
                f"bridge.py contains forbidden torch import pattern {pat!r} "
                "per HNeRV parity L9 numpy + brotli only"
            )

    def test_tier_c_hook_module_does_not_import_torch(self):
        from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill import (
            tier_c_hook as hook_mod,
        )
        source = Path(hook_mod.__file__).read_text()
        forbidden_patterns = (
            "\nimport torch\n",
            "\nimport torch.",
            "\nfrom torch ",
            "\nfrom torch.",
        )
        for pat in forbidden_patterns:
            assert pat not in source, (
                f"tier_c_hook.py contains forbidden torch import pattern {pat!r}"
            )


# ---------------------------------------------------------------------------
# Trainer config + non-promotable provenance
# ---------------------------------------------------------------------------


class TestMLXFirstTrainerConfig:
    def test_default_config_valid(self):
        cfg = MLXFirstTrainerConfig()
        assert cfg.n_pairs > 0
        assert cfg.frame_0_seg_floor == 0.0  # Atick-Redlich invariant

    def test_frame_0_seg_floor_invariant(self):
        with pytest.raises(ValueError, match="Atick-Redlich"):
            MLXFirstTrainerConfig(frame_0_seg_floor=1e-5)

    def test_n_pairs_invalid_zero(self):
        with pytest.raises(ValueError, match="n_pairs"):
            MLXFirstTrainerConfig(n_pairs=0)

    def test_n_pairs_invalid_too_large(self):
        with pytest.raises(ValueError, match="n_pairs"):
            MLXFirstTrainerConfig(n_pairs=70_000)

    def test_negative_perturbation_scale_rejected(self):
        with pytest.raises(ValueError, match="perturbation"):
            MLXFirstTrainerConfig(perturbation_scale_pose=-1.0)

    def test_pose_avg_baseline_must_be_positive(self):
        with pytest.raises(ValueError, match="pose_avg"):
            MLXFirstTrainerConfig(pose_avg_baseline=-1.0)

    def test_as_dict_round_trip(self):
        cfg = MLXFirstTrainerConfig(n_pairs=42, seed=123)
        d = cfg.as_dict()
        assert d["n_pairs"] == 42
        assert d["seed"] == 123


class TestNonPromotableProvenance:
    """Per Catalog #127/#192/#317/#341: every score-axis MLX field carries the
    canonical non-promotable provenance."""

    def test_score_claim_false(self):
        assert MLX_NON_PROMOTABLE_PROVENANCE["score_claim"] is False

    def test_promotion_eligible_false(self):
        assert MLX_NON_PROMOTABLE_PROVENANCE["promotion_eligible"] is False

    def test_axis_tag_canonical(self):
        assert "macOS-MLX" in MLX_NON_PROMOTABLE_PROVENANCE["axis_tag"]

    def test_evidence_grade_research_signal(self):
        assert "research" in MLX_NON_PROMOTABLE_PROVENANCE["evidence_grade"]

    def test_canonical_equation_proposal_atick_redlich(self):
        assert (
            MLX_NON_PROMOTABLE_PROVENANCE["canonical_equation_proposal"]
            == "atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1"
        )

    def test_canonical_equation_status_pending(self):
        assert (
            MLX_NON_PROMOTABLE_PROVENANCE["canonical_equation_status"]
            == "FORMALIZATION_PENDING"
        )

    def test_blockers_include_paired_cuda(self):
        blockers = MLX_NON_PROMOTABLE_PROVENANCE["blockers"]
        assert any("paired" in b for b in blockers)


class TestTrainerWrapperContestArchiveContract:
    def test_deterministic_archive_zip_wraps_payload_as_zero_bin(self, tmp_path: Path):
        wrapper = _load_trainer_wrapper()
        archive_zip = tmp_path / "archive.zip"

        size, digest = wrapper._write_deterministic_archive_zip(
            archive_zip,
            member_name="0.bin",
            member_bytes=b"payload",
        )

        assert size == archive_zip.stat().st_size
        assert len(digest) == 64
        with zipfile.ZipFile(archive_zip) as zf:
            infos = zf.infolist()
            assert [info.filename for info in infos] == ["0.bin"]
            assert infos[0].compress_type == zipfile.ZIP_STORED
            assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)
            assert zf.read("0.bin") == b"payload"

    def test_wrapper_auth_eval_uses_contest_archive_zip_path(self):
        source = _trainer_wrapper_source()
        assert 'archive_zip_path = submission_dir / "archive.zip"' in source
        assert "archive_zip=archive_zip_path" in source
        assert 'archive_zip=submission_dir / "0.bin"' not in source
        assert '"archive_payload_sha256"' in source
        assert '"archive_payload_bytes"' in source
        assert "auth_eval_cuda_score" in source
        assert "auth_eval_cpu_score" in source


# ---------------------------------------------------------------------------
# MLX-local compress-pass smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _mlx_available(), reason="MLX unavailable (non-Apple Silicon)")
class TestMLXLocalCompressPassSmoke:
    """5ep-equivalent (synthetic) MLX-local smoke; non-promotable per
    Catalog #192 macos_cpu_advisory + Catalog #317 paired-env canonical markers."""

    def test_smoke_returns_non_promotable_verdict(self):
        cfg = MLXFirstTrainerConfig(
            n_pairs=20, n_frame_0_modes=4, n_frame_1_modes=2, seed=42
        )
        v = run_mlx_first_compress_pass(cfg=cfg)
        assert isinstance(v, MLXFirstTrainerVerdict)
        assert v.score_claim is False
        assert v.axis_tag == DEFAULT_MLX_AXIS_TAG
        assert v.hardware_substrate.startswith("macos_") or v.hardware_substrate == "unknown_mlx_substrate"

    def test_smoke_routing_invariants(self):
        cfg = MLXFirstTrainerConfig(
            n_pairs=30, n_frame_0_modes=8, n_frame_1_modes=4, seed=20260526
        )
        v = run_mlx_first_compress_pass(cfg=cfg)
        # Per-pair joint Lagrangian NEVER worse than frame-0-only baseline
        assert (v.routing.per_pair_improvement >= -1e-9).all()
        assert 0.0 <= v.frame_1_routing_pct <= 100.0
        assert v.routing.n_pairs == 30

    def test_smoke_deterministic_given_seed(self):
        cfg1 = MLXFirstTrainerConfig(n_pairs=15, n_frame_0_modes=4, n_frame_1_modes=2, seed=99)
        cfg2 = MLXFirstTrainerConfig(n_pairs=15, n_frame_0_modes=4, n_frame_1_modes=2, seed=99)
        v1 = run_mlx_first_compress_pass(cfg=cfg1)
        v2 = run_mlx_first_compress_pass(cfg=cfg2)
        assert np.array_equal(v1.routing.routing_decision, v2.routing.routing_decision)
        assert np.array_equal(v1.routing.selected_mode_idx, v2.routing.selected_mode_idx)

    def test_smoke_emits_json_sidecar(self):
        cfg = MLXFirstTrainerConfig(n_pairs=12, n_frame_0_modes=4, n_frame_1_modes=2, seed=7)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            run_mlx_first_compress_pass(cfg=cfg, output_dir=out, emit_json_sidecar=True)
            sidecar = out / "mlx_first_compress_pass_verdict.json"
            assert sidecar.exists()
            payload = json.loads(sidecar.read_text())
            assert payload["score_claim"] is False
            assert payload["axis_tag"] == DEFAULT_MLX_AXIS_TAG
            assert payload["n_pairs"] == 12

    def test_smoke_atick_redlich_frame_0_seg_invariant(self):
        """frame-0 SegNet penalty is STRUCTURALLY zero per Atick-Redlich."""
        cfg = MLXFirstTrainerConfig(n_pairs=10, n_frame_0_modes=4, n_frame_1_modes=2, seed=1)
        v = run_mlx_first_compress_pass(cfg=cfg)
        # Selected seg delta MUST be 0.0 for frame_0-routed pairs
        f0_mask = (v.routing.routing_decision == 0)
        assert (v.routing.selected_seg_delta[f0_mask] == 0.0).all()


# ---------------------------------------------------------------------------
# MLX → numpy bridge round-trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _mlx_available(), reason="MLX unavailable (non-Apple Silicon)")
class TestMLXToNumpyBridgeRoundtrip:
    def _build_state_dict(self, n_pairs=20, seed=20260526):
        cfg = MLXFirstTrainerConfig(
            n_pairs=n_pairs, n_frame_0_modes=4, n_frame_1_modes=2, seed=seed
        )
        v = run_mlx_first_compress_pass(cfg=cfg)
        return v.state_dict

    def test_bridge_contract_keys_present(self):
        sd = self._build_state_dict()
        n_pairs = verify_state_dict_shape_contract(sd)
        assert n_pairs == 20

    def test_bridge_missing_key_rejected(self):
        sd = self._build_state_dict()
        bad = {k: v for k, v in sd.items() if k != "routing_decision"}
        with pytest.raises(MLXNumpyBridgeError, match="missing canonical key"):
            verify_state_dict_shape_contract(bad)

    def test_bridge_export_load_roundtrip(self):
        sd = self._build_state_dict()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.npz"
            written = export_state_dict_to_npz(sd, path)
            assert written.exists()
            loaded = load_state_dict_from_npz(path)
            for key in EXPECTED_STATE_DICT_KEYS:
                assert np.array_equal(loaded[key], sd[key])

    def test_bridge_archive_roundtrip_byte_identical(self):
        sd = self._build_state_dict(n_pairs=50)
        rt = roundtrip_state_dict_through_archive(sd)
        assert isinstance(rt, BridgeRoundtripVerdict)
        assert rt.routing_decision_byte_identical is True
        assert rt.n_pairs == 50
        assert rt.archive_byte_count > 0
        assert len(rt.archive_sha256) == 64

    def test_bridge_load_missing_file_rejects(self):
        with pytest.raises(MLXNumpyBridgeError, match="missing"):
            load_state_dict_from_npz(Path("/nonexistent/foo.npz"))


# ---------------------------------------------------------------------------
# Tier-C MDL ablation hook smoke
# ---------------------------------------------------------------------------


class TestTierCAblationHook:
    def test_build_request_returns_pending_verdict(self):
        v = build_tier_c_ablation_probe_request(
            archive_sha256="a" * 64,
            archive_path=Path("/tmp/synth/archive.zip"),
            output_dir=Path("/tmp/tc_out"),
        )
        assert isinstance(v, TierCAblationHookVerdict)
        assert v.formalization_status == FORMALIZATION_PENDING_VERDICT
        assert v.is_pending is True
        assert v.axis_tag == "[macOS-MLX research-signal]"
        assert v.probe_request is not None

    def test_request_canonical_cli_uses_tier_c_flag(self):
        v = build_tier_c_ablation_probe_request(
            archive_sha256="b" * 64,
            archive_path=Path("/tmp/x/archive.zip"),
            output_dir=Path("/tmp/y"),
        )
        cli = v.probe_request.canonical_cli_invocation()
        assert "--tier" in cli
        assert "c" in cli
        assert DEFAULT_TIER_C_TOOL_PATH in cli

    def test_request_carries_canonical_equation_proposal(self):
        v = build_tier_c_ablation_probe_request(
            archive_sha256="c" * 64,
            archive_path=Path("/tmp/x/archive.zip"),
            output_dir=Path("/tmp/y"),
        )
        assert (
            v.probe_request.canonical_equation_proposal
            == "atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1"
        )

    def test_classify_within_class(self):
        assert classify_tier_c_density_verdict(0.85) == "within_class"

    def test_classify_across_class(self):
        assert classify_tier_c_density_verdict(0.20) == "across_class"

    def test_classify_indeterminate(self):
        assert classify_tier_c_density_verdict(0.50) == "indeterminate"

    def test_classify_threshold_boundaries(self):
        assert classify_tier_c_density_verdict(0.70) == "within_class"
        assert classify_tier_c_density_verdict(0.30) == "across_class"

    def test_classify_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="density"):
            classify_tier_c_density_verdict(1.5)
        with pytest.raises(ValueError, match="density"):
            classify_tier_c_density_verdict(-0.1)

    def test_hook_provenance_carries_non_promotable_markers(self):
        v = build_tier_c_ablation_probe_request(
            archive_sha256="d" * 64,
            archive_path=Path("/tmp/x/archive.zip"),
            output_dir=Path("/tmp/y"),
        )
        assert v.provenance["score_claim"] is False
        assert v.provenance["promotion_eligible"] is False
        assert v.provenance["canonical_equation_status"] == "FORMALIZATION_PENDING"


# ---------------------------------------------------------------------------
# Catalog #229 PV: substrate contract still satisfied
# ---------------------------------------------------------------------------


class TestSubstrateContractStillSatisfied:
    """Per Catalog #229 premise verification: the existing scaffold's
    substrate_contract is preserved (no mutation; only NEW modules added)."""

    def test_canonical_id_unchanged(self):
        assert (
            CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT.id
            == "cascade_c_prime_frame_1_segnet_waterfill"
        )

    def test_recipe_research_only_unchanged(self):
        assert (
            CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT.recipe_research_only
            is True
        )

    def test_lane_id_preserved(self):
        assert (
            CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT.lane_id
            == "lane_cascade_c_prime_option_a_build_scaffold_20260526"
        )
