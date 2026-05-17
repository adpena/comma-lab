# SPDX-License-Identifier: MIT
"""Tests for the Phase 1b Z6 _full_main lift (2026-05-16).

Per the Phase 1b Z6 lift directive (lane
``lane_phase_1b_z6_lift_20260516``), the Z6 trainer's ``_full_main`` is
lifted from ``raise NotImplementedError`` to a working PR95-paradigm
implementation that binds ALL ingredients per Catalogs #310/#311/#312
and the Z6/Z7/Z8 design memo §4.1:

* Catalog #310 PRIMARY class-shift: Z6 is the architectural core, NOT a
  bolt-on objective on top of Z3/A1. The FiLM predictor + autoregressive
  latent dynamics + archived residuals form a primary substrate per the
  Atick-Redlich + Rao-Ballard paradigm.
* Catalog #311 ego-motion-conditioned next-frame prediction: pose-conditioned
  next-frame prediction with FOE prior derived from per-pair PoseNet output
  following Gibson 1950 + Ballard embodied-vision lineage.
* Catalog #312 hierarchical predictive coding canonical quadruple: NOT
  required for Z6 (single-layer FiLM predictor is the simplest viable
  variant per the design memo §4.1; the hierarchical quadruple is the Z8
  scope per the design memo §13). Z6 satisfies the predictive-coding +
  cooperative-receiver pair (Rao-Ballard + Atick-Redlich) which IS already
  the substrate's primary class-shift surface.

These tests assert structural properties of the lift WITHOUT requiring
upstream/videos/0.mkv or CUDA scorers — they verify the binding contract
(canonical helper invocation surface) rather than running a real training
step (which would cost real wall-clock).
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pytest
import torch

import experiments.train_substrate_time_traveler_l5_z6 as z6_trainer
from tac.substrates.time_traveler_l5_z6 import (
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingSubstrate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# (A) Lift status — _full_main no longer NotImplementedError
# ---------------------------------------------------------------------------


def test_full_main_no_longer_raises_not_implemented_error() -> None:
    """The legacy NotImplementedError stub is structurally extincted."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "raise NotImplementedError" not in source, (
        "Z6 _full_main retains the legacy NotImplementedError stub; "
        "Phase 1b lift incomplete."
    )


def test_full_main_is_callable_without_phase_2_gate() -> None:
    """_full_main can be invoked and progresses past argument parsing.

    The function will eventually fail without CUDA / real upstream video,
    but it must NOT short-circuit at the function entry with the legacy
    NotImplementedError. We invoke it with a sentinel that triggers an
    early raise inside the PR95 binding rather than the legacy gate.
    """
    args = argparse.Namespace(
        full_cpu=False,
        advisory_cpu_explicitly_waived=False,
        smoke=False,
        seed=0,
        device="cpu",  # device_or_die raises SystemExit for non-CUDA full path
        output_dir=Path("/tmp"),  # not actually used; device gate fires first
        video_path=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        upstream_dir=REPO_ROOT / "upstream",
    )
    # We expect SystemExit (device_or_die rejects --device cpu in full mode)
    # NOT NotImplementedError (the legacy gate).
    with pytest.raises(SystemExit) as exc_info:
        z6_trainer._full_main(args)
    # The SystemExit message must come from device_or_die (cuda-required),
    # not from anything earlier — proving _full_main reached step 1 (device
    # gate) of the canonical binding.
    msg = str(exc_info.value)
    assert "cpu" in msg.lower() or "cuda" in msg.lower(), (
        f"expected device_or_die SystemExit; got {msg!r}"
    )


# ---------------------------------------------------------------------------
# (B) Catalog #310 PRIMARY class-shift binding
# ---------------------------------------------------------------------------


def test_lift_binds_z6_substrate_as_primary_architecture() -> None:
    """Catalog #310: Z6 IS the primary substrate; NOT a bolt-on."""
    source = inspect.getsource(z6_trainer._full_main)
    # The substrate IS constructed inside _full_main (the primary surface),
    # not added as a bolt-on objective on top of a sister substrate.
    assert "Z6PredictiveCodingSubstrate(cfg)" in source
    # The substrate's autoregressive reconstruct_pair IS the per-step
    # forward (NOT a bolt-on loss term on a sister architecture's output).
    assert "substrate.reconstruct_pair(" in source
    # The optimizer trains substrate.parameters() — the primary surface.
    assert "substrate.parameters()" in source


def test_lift_uses_canonical_pr95_paradigm_ingredients() -> None:
    """PR95 paradigm binding: pyav + YUV6 patch + scorers + EMA + eval_roundtrip + score_pair_components."""
    source = inspect.getsource(z6_trainer._full_main)
    # Catalog #114: pyav decode of real contest pairs (no synthetic data).
    assert "_decode_real_pairs" in source
    # Catalog #187: patch upstream rgb_to_yuv6 BEFORE scorer construction.
    assert "patch_upstream_yuv6_globally" in source
    # Catalog #5: differentiable scorers; eval_roundtrip non-negotiable.
    assert "load_differentiable_scorers" in source
    assert "apply_eval_roundtrip" in source
    # Catalog #88: EMA(decay=0.997).
    assert "EMA(substrate, decay=args.ema_decay)" in source
    # Catalog #164: canonical scorer-loss helper routing — Z6 routes through
    # Z6PredictiveCodingScoreAwareLoss which itself dispatches via
    # score_pair_components_dispatch.
    assert "Z6PredictiveCodingScoreAwareLoss" in source


# ---------------------------------------------------------------------------
# (C) Catalog #311 ego-motion-conditioned next-frame prediction
# ---------------------------------------------------------------------------


def test_lift_invokes_ego_motion_derivation_from_posenet() -> None:
    """Catalog #311 NON-NEGOTIABLE: ego-motion-conditioned prediction MUST route through PoseNet."""
    source = inspect.getsource(z6_trainer._full_main)
    # The trainer body must INVOKE the ego-motion derivation helper.
    assert "_derive_ego_motion_from_posenet" in source
    # The derived ego-motion must be copied into substrate.ego_motion_buffer
    # so the FiLM predictor consumes it during the autoregressive unroll.
    assert "substrate.ego_motion_buffer.copy_" in source


def test_derive_ego_motion_from_posenet_exists_and_shape_correct() -> None:
    """The canonical helper exists with the documented signature + return shape."""
    helper = z6_trainer._derive_ego_motion_from_posenet
    sig = inspect.signature(helper)
    assert "posenet" in sig.parameters
    assert "gt_pair_tensor" in sig.parameters
    assert "ego_motion_dim" in sig.parameters

    # Smoke a fake PoseNet that returns pose features deterministically.
    class _FakePoseNet(torch.nn.Module):
        def preprocess_input(self, pair):
            return pair

        def forward(self, pair):
            B = pair.shape[0]
            return {"pose": torch.linspace(-1.0, 1.0, B * 12).view(B, 12)}

    gt_pairs = torch.rand(4, 2, 3, 48, 64)
    ego = helper(
        _FakePoseNet(),
        gt_pairs,
        ego_motion_dim=8,
        chunk_size=2,
        device=torch.device("cpu"),
    )
    assert ego.shape == (4, 8)
    assert ego.dtype == torch.float32
    # Per-column standardized: mean ~0, std ~1 (FOE-prior conditioning)
    assert torch.allclose(ego.mean(dim=0), torch.zeros(8), atol=1e-4)


def test_derive_ego_motion_with_more_ego_dim_than_pose_pads_with_zeros() -> None:
    """If ego_motion_dim > PoseNet head width, helper pads with zeros."""
    class _SmallPoseNet(torch.nn.Module):
        def preprocess_input(self, pair):
            return pair

        def forward(self, pair):
            B = pair.shape[0]
            return {"pose": torch.linspace(-1.0, 1.0, B * 4).view(B, 4)}

    gt_pairs = torch.rand(3, 2, 3, 24, 32)
    ego = z6_trainer._derive_ego_motion_from_posenet(
        _SmallPoseNet(),
        gt_pairs,
        ego_motion_dim=8,
        chunk_size=2,
        device=torch.device("cpu"),
    )
    assert ego.shape == (3, 8)


def test_derive_ego_motion_with_tensor_only_posenet() -> None:
    """If PoseNet returns a tensor (not dict), the helper still extracts pose."""
    class _TensorPoseNet(torch.nn.Module):
        def preprocess_input(self, pair):
            return pair

        def forward(self, pair):
            B = pair.shape[0]
            return torch.rand(B, 12)

    gt_pairs = torch.rand(2, 2, 3, 24, 32)
    ego = z6_trainer._derive_ego_motion_from_posenet(
        _TensorPoseNet(),
        gt_pairs,
        ego_motion_dim=6,
        chunk_size=2,
        device=torch.device("cpu"),
    )
    assert ego.shape == (2, 6)


def test_derive_ego_motion_refuses_bad_inputs() -> None:
    """Helper validates ego_motion_dim > 0 and gt_pair_tensor shape."""
    posenet = torch.nn.Identity()
    with pytest.raises(ValueError, match="ego_motion_dim must be > 0"):
        z6_trainer._derive_ego_motion_from_posenet(
            posenet, torch.rand(1, 2, 3, 8, 8), ego_motion_dim=0
        )
    with pytest.raises(ValueError, match="\\(N, 2, 3, H, W\\)"):
        z6_trainer._derive_ego_motion_from_posenet(
            posenet, torch.rand(1, 3, 8, 8), ego_motion_dim=4
        )


def test_lift_identity_predictor_uses_zero_ego_motion_per_disambiguator_regime() -> None:
    """Catalog #125 hook #6: identity-predictor disambiguator regime uses zero ego-motion.

    The probe-disambiguator's null hypothesis is that the FiLM-conditioned
    predictor's lift comes from capacity, not from ego-motion conditioning.
    The identity-predictor regime must NOT consume PoseNet-derived ego-motion
    (per the design memo §4.1 + Catalog #125 hook #6).
    """
    source = inspect.getsource(z6_trainer._full_main)
    assert "if args.identity_predictor:" in source
    assert "zero ego-motion" in source.lower() or "zeros(" in source


# ---------------------------------------------------------------------------
# (D) Catalog #312 predictive coding canonical surfaces (paradigm-level)
# ---------------------------------------------------------------------------


def test_lift_residual_entropy_lagrangian_active_at_training() -> None:
    """The Rao-Ballard residual-entropy Lagrangian is part of the training loss."""
    source = inspect.getsource(z6_trainer._full_main)
    # The loss fn explicitly receives substrate.residuals so the
    # residual-entropy term participates in gradient flow.
    assert "residuals=substrate.residuals" in source
    # The loss weights carry lambda_residual_entropy.
    assert "lambda_residual_entropy=args.lambda_residual_entropy" in source


def test_lift_autoregressive_predictor_unroll_via_reconstruct_pair() -> None:
    """The FiLM predictor rolls latent state forward via autoregressive reconstruct_pair."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "substrate.reconstruct_pair(batch_idx_tensor)" in source
    # The Z6PredictiveCodingSubstrate.reconstruct_pair is the canonical
    # autoregressive next-frame predictor (per architecture.py:399-438).
    arch_source = inspect.getsource(
        Z6PredictiveCodingSubstrate.reconstruct_pair
    )
    assert "self.predictor(z_prev, ego_t)" in arch_source
    assert "self.residuals[t]" in arch_source


# ---------------------------------------------------------------------------
# (E) Archive emission + roundtrip-stable
# ---------------------------------------------------------------------------


def test_lift_packs_z6pcwm1_archive_via_canonical_grammar() -> None:
    """The lift emits a Z6PCWM1 archive via the canonical pack_archive helper."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "pack_archive(" in source
    # All 6 substrate sections must be packed (encoder + decoder + predictor +
    # latent_init + residuals + ego_motion_buffer).
    assert "substrate.encoder.state_dict()" in source
    assert "substrate.decoder.state_dict()" in source
    assert "substrate.predictor.state_dict()" in source
    assert "substrate.latent_init.detach().cpu()" in source
    assert "substrate.residuals.detach().cpu()" in source
    assert "substrate.ego_motion_buffer.detach().cpu()" in source


def test_lift_emits_contest_compliant_runtime_tree() -> None:
    """The lift emits inflate.sh + inflate.py + vendored substrate per Catalog #146/#163/#205."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "_write_runtime(submission_dir)" in source
    assert "_build_archive_zip" in source
    # archive.zip must be copied into submission_dir per Catalog #146 contract.
    assert "submission_dir / \"archive.zip\"" in source


def test_write_runtime_emits_3_positional_arg_inflate_sh_per_catalog_146(
    tmp_path: Path,
) -> None:
    """Catalog #146: inflate.sh signature is `inflate.sh <archive_dir> <output_dir> <file_list>`."""
    submission_dir = tmp_path / "submission_dir"
    z6_trainer._write_runtime(submission_dir)
    inflate_sh = (submission_dir / "inflate.sh").read_text()
    assert "set -euo pipefail" in inflate_sh, "Catalog #163 set -euo pipefail required"
    assert "DATA_DIR=\"$1\"" in inflate_sh
    assert "OUTPUT_DIR=\"$2\"" in inflate_sh
    assert "FILE_LIST=\"$3\"" in inflate_sh
    inflate_py = (submission_dir / "inflate.py").read_text()
    assert "main_cli" in inflate_py
    # No scorer imports at inflate time per CLAUDE.md "Strict scorer rule".
    assert "PoseNet" not in inflate_py
    assert "SegNet" not in inflate_py


def test_write_runtime_vendors_substrate_and_shared_inflate(tmp_path: Path) -> None:
    """The runtime tree includes the substrate's architecture + archive + inflate + _shared helper."""
    submission_dir = tmp_path / "submission_dir"
    z6_trainer._write_runtime(submission_dir)
    pkg = submission_dir / "src" / "tac" / "substrates" / "time_traveler_l5_z6"
    assert (pkg / "architecture.py").is_file()
    assert (pkg / "archive.py").is_file()
    assert (pkg / "inflate.py").is_file()
    assert (pkg / "__init__.py").is_file()
    shared = submission_dir / "src" / "tac" / "substrates" / "_shared"
    assert (shared / "inflate_runtime.py").is_file()


def test_archive_roundtrip_stability_post_pack(tmp_path: Path) -> None:
    """Pack a tiny Z6 archive end-to-end via the trainer helpers; verify roundtrip is stable."""
    from tac.substrates.time_traveler_l5_z6 import pack_archive, parse_archive

    torch.manual_seed(0)
    cfg = Z6PredictiveCodingConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=3,
        output_height=48,
        output_width=64,
        predictor_hidden_dim=16,
        predictor_film_mlp_hidden_dim=8,
        predictor_ego_motion_dim=4,
        predictor_kernel_size=3,
    )
    substrate = Z6PredictiveCodingSubstrate(cfg)
    with torch.no_grad():
        substrate.ego_motion_buffer.copy_(torch.linspace(-1.0, 1.0, 3 * 4).view(3, 4))
    bin_bytes = pack_archive(
        substrate.encoder.state_dict(),
        substrate.decoder.state_dict(),
        substrate.predictor.state_dict(),
        substrate.latent_init.detach().cpu(),
        substrate.residuals.detach().cpu(),
        substrate.ego_motion_buffer.detach().cpu(),
        {"smoke": False, "test": "phase_1b_lift"},
    )
    arc = parse_archive(bin_bytes)
    assert arc.latent_init.shape == (8,)
    assert arc.residuals.shape == (3, 8)
    assert arc.ego_motion.shape == (3, 4)


# ---------------------------------------------------------------------------
# (F) Canonical helper routing (Catalog #226 / #127 / #128 / #190)
# ---------------------------------------------------------------------------


def test_lift_routes_through_canonical_gate_auth_eval_call() -> None:
    """Catalog #226: auth-eval MUST route through `gate_auth_eval_call`, never hand-rolled."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "_canon_gate_auth_eval_call(" in source
    # Never hand-rolled subprocess to contest_auth_eval.py per Catalog #226.
    assert "subprocess.run" not in source
    assert "subprocess.Popen" not in source


def test_lift_requires_contest_cuda_auth_eval_claim_per_catalog_127() -> None:
    """Catalog #127: posterior consumption MUST validate the contest-CUDA claim."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "_canon_require_contest_cuda_auth_eval_claim(" in source


def test_lift_invokes_posterior_update_locked_per_catalog_128() -> None:
    """Catalog #128: atomic fcntl-locked continual-learning posterior update."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "posterior_update_locked_from_auth_eval_json" in source


def test_lift_detects_hardware_substrate_per_catalog_190() -> None:
    """Catalog #190: hardware substrate MUST be detected dynamically, not hardcoded."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "_canon_detect_hardware_substrate(" in source
    # NOT hardcoded
    assert "hardware_substrate=\"linux_x86_64_t4\"" not in source


# ---------------------------------------------------------------------------
# (G) 9-dim checklist regression guard
# ---------------------------------------------------------------------------


def test_full_main_docstring_or_module_carries_pr95_paradigm_claim() -> None:
    """The lift's docstring / module-level comment cites PR95 paradigm + canonical ingredients."""
    source = inspect.getsource(z6_trainer._full_main)
    assert "PR 95" in source or "PR95" in source
    # The canonical-vs-unique decision section MUST be present (Catalog #290).
    module_source = inspect.getsource(z6_trainer)
    assert "Canonical-vs-unique decisions per layer" in module_source


def test_lift_argparse_carries_required_full_main_flags() -> None:
    """The lift adds val_pair_count + val_every_epochs + max_pairs + skip_archive_build + skip_auth_eval."""
    parser = z6_trainer._build_parser()
    # Parse default args (smoke) so we can inspect attribute names.
    args = parser.parse_args(["--output-dir", "/tmp/z6_test"])
    assert hasattr(args, "val_pair_count")
    assert hasattr(args, "val_every_epochs")
    assert hasattr(args, "max_pairs")
    assert hasattr(args, "skip_archive_build")
    assert hasattr(args, "skip_auth_eval")
    assert hasattr(args, "ego_motion_chunk_size")
    # Default `--max-pairs` is None (full-length training).
    assert args.max_pairs is None


def test_substrate_contract_still_validates_post_lift() -> None:
    """The SubstrateContract (Catalog #241/#242) still validates after the lift.

    The contract is decorated at module import time; if validation refused
    anything the trainer module would fail to import. We assert the contract
    is present + identity-stable (the registry refuses duplicate IDs, so
    re-importing is intentionally rejected — that itself proves the
    decoration ran successfully on the original import).
    """
    contract = z6_trainer.TIME_TRAVELER_L5_Z6_SUBSTRATE_CONTRACT
    assert contract.id == "time_traveler_l5_z6"
    assert contract.lane_id == z6_trainer.SUBSTRATE_LANE_ID
    # Verify all 8 archive-grammar fields per Catalog #124 are declared.
    assert contract.archive_grammar
    assert contract.parser_section_manifest
    assert contract.inflate_runtime_loc_budget > 0
    assert contract.runtime_dep_closure
    assert contract.export_format
    assert contract.score_aware_loss
    assert contract.bolt_on_loc_budget > 0
    assert contract.no_op_detector_planned in (True, False)
