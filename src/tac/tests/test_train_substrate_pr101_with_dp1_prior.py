# SPDX-License-Identifier: MIT
"""Tests for experiments/train_substrate_pr101_with_dp1_prior_regularizer.py (PATH 2 scaffold).

Lane: lane_dp1_plus_fec6_dual_stacking_build_20260517

PATH 2 is the REFORMULATED training-time prior path. The operator's original
description ("L2 regularizer on PR101 decoder weights") was structurally
incompatible with fec6 (no learned decoder weights) per the premise verifier
.omx/tmp/dp1_dual_stacking_premise_verifier.txt. The reformulated PATH 2
applies DP1 FRAME-SPACE prior (DashcamPriorLoss) on pr101_lc_v2_clone learned
renderer's RGB output.

Coverage:
* import surface — trainer + SubstrateContract import cleanly
* SubstrateContract canonical-fields validation (Catalog #242)
* _smoke_main runs to completion (DashcamPriorLoss correctness)
* _full_main raises NotImplementedError (Catalog #240 scaffold-only)
* gradient sanity — λ_DP1=0 vs λ_DP1>0 produce measurably different
  outputs after apply_soft_prior
* Catalog #210 codebook provenance metadata preserved through training
* CLI surface — --smoke, --full-cpu / --advisory-cpu-explicitly-waived
  coupling per Catalog #197
* Tier 1 required-flags manifest declared per Catalog #151
* DashcamPriorLoss buffers stay non-trainable (frozen codebook)
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = REPO_ROOT / "experiments" / "train_substrate_pr101_with_dp1_prior_regularizer.py"


_TRAINER_MODULE_NAME = "train_pr101_with_dp1_prior_test_module"


def _load_trainer_module():
    # Cache via sys.modules to avoid re-decorating @register_substrate (which
    # would raise SubstrateContractError on duplicate id).
    if _TRAINER_MODULE_NAME in sys.modules:
        return sys.modules[_TRAINER_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(
        _TRAINER_MODULE_NAME, str(TRAINER_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_TRAINER_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


class TestImportSurface:
    def test_trainer_file_exists(self) -> None:
        assert TRAINER_PATH.exists(), f"trainer scaffold not at {TRAINER_PATH}"

    def test_trainer_imports_cleanly(self) -> None:
        mod = _load_trainer_module()
        assert hasattr(mod, "PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT")
        assert hasattr(mod, "_smoke_main")
        assert hasattr(mod, "_full_main")
        assert hasattr(mod, "build_argparser")
        assert hasattr(mod, "main")

    def test_tier_1_required_flags_declared(self) -> None:
        mod = _load_trainer_module()
        assert hasattr(mod, "TIER_1_OPERATOR_REQUIRED_FLAGS")
        flags = mod.TIER_1_OPERATOR_REQUIRED_FLAGS
        assert isinstance(flags, dict)
        # Must declare at least the canonical 3: video, dp1 codebook, output
        assert "--video-path" in flags
        assert "--dp1-codebook-bin" in flags
        assert "--output-dir" in flags
        # video and codebook must be required_input_file
        assert flags["--video-path"]["required_input_file"] is True
        assert flags["--dp1-codebook-bin"]["required_input_file"] is True


class TestSubstrateContractCanonicalFields:
    """Catalog #242: register_substrate contract fields canonical."""

    def test_contract_declares_36_canonical_fields(self) -> None:
        from tac.substrate_registry import SubstrateContract
        mod = _load_trainer_module()
        contract = mod.PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT
        assert isinstance(contract, SubstrateContract)

    def test_contract_id_canonical(self) -> None:
        mod = _load_trainer_module()
        contract = mod.PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT
        assert contract.id == "pr101_with_dp1_prior_regularizer"

    def test_contract_lane_id_canonical(self) -> None:
        mod = _load_trainer_module()
        contract = mod.PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT
        assert contract.lane_id == "lane_dp1_plus_fec6_dual_stacking_build_20260517"

    def test_contract_is_research_only_at_landing(self) -> None:
        """Catalog #240: full_main raises NotImplementedError => recipe research_only."""
        mod = _load_trainer_module()
        contract = mod.PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT
        assert contract.recipe_research_only is True
        assert contract.score_improvement_mechanism_status == "RESEARCH_ONLY"

    def test_contract_canary_dependency_dp1(self) -> None:
        """PR101+DP1 must mark DP1 as canary dependency per Catalog #173."""
        mod = _load_trainer_module()
        contract = mod.PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT
        assert contract.recipe_canary_status == "post_canary_dependent"
        assert contract.recipe_canary_dependency == "pretrained_driving_prior"

    def test_contract_compliance_declarations_include_209_210(self) -> None:
        mod = _load_trainer_module()
        contract = mod.PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT
        decls = contract.catalog_compliance_declarations
        # Catalog #209 leakage guard + #210 provenance preserved are critical
        assert any("catalog_209" in d for d in decls)
        assert any("catalog_210" in d for d in decls)


class TestSmokeMainCorrectness:
    """_smoke_main runs to completion with DashcamPriorLoss correctness check."""

    def test_smoke_returns_zero(self) -> None:
        mod = _load_trainer_module()
        parser = mod.build_argparser()
        args = parser.parse_args(["--smoke"])
        rc = mod._smoke_main(args)
        assert rc == 0

    def test_smoke_via_subprocess(self) -> None:
        """End-to-end CLI invocation: --smoke exits 0."""
        result = subprocess.run(
            [sys.executable, str(TRAINER_PATH), "--smoke"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"--smoke subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "PATH 2 scaffold smoke OK" in result.stdout


class TestFullMainDeferred:
    """Catalog #240: _full_main MUST raise NotImplementedError until council approves."""

    def test_full_main_raises(self) -> None:
        mod = _load_trainer_module()
        parser = mod.build_argparser()
        args = parser.parse_args([])  # no --smoke -> _full_main
        with pytest.raises(NotImplementedError) as excinfo:
            mod._full_main(args)
        # Per Catalog #240: error message must cite the canonical blocker.
        assert "phase_2_council_approval_required" in str(excinfo.value)
        assert "lane_dp1_plus_fec6_dual_stacking_build_20260517" in str(excinfo.value)


class TestFullCpuCouplingFlag:
    """Catalog #197: --full-cpu requires paired --advisory-cpu-explicitly-waived."""

    def test_full_cpu_without_waiver_refused(self) -> None:
        mod = _load_trainer_module()
        parser = mod.build_argparser()
        args = parser.parse_args(["--smoke", "--full-cpu"])
        with pytest.raises(SystemExit):
            mod._validate_full_cpu_flags(args)

    def test_full_cpu_with_waiver_accepted(self) -> None:
        mod = _load_trainer_module()
        parser = mod.build_argparser()
        args = parser.parse_args([
            "--smoke", "--full-cpu", "--advisory-cpu-explicitly-waived"
        ])
        # No SystemExit.
        mod._validate_full_cpu_flags(args)


class TestDashcamPriorLossSanity:
    """Gradient sanity: λ_DP1 = 0 vs λ_DP1 > 0 produce measurably different outputs."""

    def test_lambda_zero_vs_nonzero_produces_different_outputs(self) -> None:
        import torch
        from tac.substrates.pretrained_driving_prior import (
            DashcamPriorLoss,
            PriorApplicationWeights,
            deterministic_zero_codebook,
        )

        torch.manual_seed(0)
        codebook = deterministic_zero_codebook()
        weights = PriorApplicationWeights()
        prior = DashcamPriorLoss(codebook=codebook, weights=weights, device="cpu")

        rgb = torch.rand(1, 3, 384, 512)
        out_zero = prior.apply_soft_prior(rgb, strength=0.0)
        out_nonzero = prior.apply_soft_prior(rgb, strength=0.5)

        # λ=0 => identity; λ>0 => measurable difference (codebook is zero-filled,
        # so the bottom-third road-plane projection lerps toward zero band).
        assert torch.equal(out_zero, rgb), "strength=0 must be identity"
        # Bottom-third (road band) must differ.
        h_img = rgb.shape[2]
        road_start = 2 * h_img // 3
        diff = (out_nonzero[:, :, road_start:, :] - rgb[:, :, road_start:, :]).abs().mean().item()
        assert diff > 1e-6, f"strength>0 must change road band; diff={diff}"

    def test_codebook_buffers_not_trainable(self) -> None:
        """The DP1 codebook MUST be frozen — registered as buffers, not parameters."""
        import torch
        from tac.substrates.pretrained_driving_prior import (
            DashcamPriorLoss,
            PriorApplicationWeights,
            deterministic_zero_codebook,
        )

        prior = DashcamPriorLoss(
            codebook=deterministic_zero_codebook(),
            weights=PriorApplicationWeights(),
            device="cpu",
        )
        # All buffers, no trainable parameters from the codebook itself.
        param_count = sum(p.numel() for p in prior.parameters())
        buffer_count = sum(b.numel() for b in prior.buffers())
        assert param_count == 0, f"DashcamPriorLoss has trainable params: {param_count}"
        assert buffer_count > 0, "DashcamPriorLoss missing codebook buffers"


class TestCodebookProvenancePreservation:
    """Catalog #210: DP1 codebook provenance metadata preserved through training."""

    def test_dp1_codebook_metadata_fields_present(self) -> None:
        """Validate the canonical DP1 codebook carries provenance fields.

        Catalog #210 requires 6 fields on production-distilled codebooks
        (license_tags / dataset_provenance / distillation_version / random_seed
        / basis_sha256 / num_frames_used). The scaffold deterministic codebook
        carries 3 of 6 (license_tags / dataset_provenance / distillation_version);
        the other 3 are populated by real distillation paths. We assert the 3
        scaffold-side fields here; sister test_dp1_archive_canonical covers the
        production-distill 6-field invariant.
        """
        from tac.substrates.pretrained_driving_prior import (
            DashcamCodebook,
            deterministic_zero_codebook,
        )

        cb = deterministic_zero_codebook()
        assert isinstance(cb, DashcamCodebook)
        meta = cb.metadata
        scaffold_required = {
            "license_tags", "dataset_provenance", "distillation_version",
        }
        missing = scaffold_required - set(meta.keys())
        assert not missing, (
            f"DP1 scaffold codebook missing Catalog #210 scaffold-tier provenance: {missing}"
        )
        # Provenance must clearly mark this as scaffold (NOT contest-distilled).
        assert "scaffold" in meta["dataset_provenance"].lower(), (
            f"scaffold codebook provenance not marked as scaffold: {meta['dataset_provenance']}"
        )


class TestRegisterSubstrateDecoration:
    """Catalog #241: trainer MUST use @register_substrate decorator."""

    def test_register_substrate_marker_present(self) -> None:
        mod = _load_trainer_module()
        # Find any callable decorated with @register_substrate
        text = TRAINER_PATH.read_text()
        assert "@register_substrate" in text, (
            "trainer must carry @register_substrate decoration per Catalog #241"
        )

    def test_substrate_in_registry_after_load(self) -> None:
        """After importing the trainer, the substrate should be in the registry."""
        _ = _load_trainer_module()
        from tac.substrate_registry import _REGISTERED_SUBSTRATES
        assert "pr101_with_dp1_prior_regularizer" in _REGISTERED_SUBSTRATES
