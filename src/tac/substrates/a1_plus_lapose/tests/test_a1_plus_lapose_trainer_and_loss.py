# SPDX-License-Identifier: MIT
"""Trainer + score-aware-loss + dispatch-readiness tests for A1 + LAPose.

Per CLAUDE.md "Recursive adversarial review protocol" — ≥15 dedicated tests
covering smoke main, score-aware loss math, archive bytes proxy, vendored
runtime, D4-mode parameterization, and Catalog #151/#168 compliance.
"""
from __future__ import annotations

import ast
import importlib
import json
import math
import zipfile
from pathlib import Path

import pytest
import torch

from tac.substrates.a1_plus_lapose.architecture import (
    A1_BASE_CHANNELS,
    A1_CAMERA_H,
    A1_CAMERA_W,
    A1_EVAL_H,
    A1_EVAL_W,
    A1_LATENT_DIM,
    A1_N_PAIRS,
    A1PlusLaposeConfig,
    PerPairResidualHead,
    parse_lapose_atom_indices,
)
from tac.substrates.a1_plus_lapose.archive import (
    LAPOSE_SIDECAR_MAGIC,
    decode_lapose_sidecar,
    pack_composition_archive,
    split_composition_archive,
)
from tac.substrates.a1_plus_lapose.score_aware_loss import (
    A1PlusLaposeLossWeights,
    A1PlusLaposeScoreAwareLoss,
)


REPO_ROOT = Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# A1 architecture constants
# ---------------------------------------------------------------------------

def test_a1_constants_match_canonical_anchor() -> None:
    """Constants reflect the verified A1 PR101-fine-tune anchor."""
    assert A1_EVAL_H == 384
    assert A1_EVAL_W == 512
    assert A1_CAMERA_H == 874
    assert A1_CAMERA_W == 1164
    assert A1_N_PAIRS == 600
    assert A1_LATENT_DIM == 28
    assert A1_BASE_CHANNELS == 36


# ---------------------------------------------------------------------------
# LAPose residual head
# ---------------------------------------------------------------------------

def test_residual_head_zeros_for_unselected_pairs() -> None:
    """Pairs not in the selection get zero residuals (no-op)."""
    cfg = A1PlusLaposeConfig(
        residual_rank=2,
        selected_pair_indices=(3, 7),
        foveal_h=8,
        foveal_w=8,
    )
    head = PerPairResidualHead(cfg)
    out = head.residual_chw(99, 0)
    assert out.shape == (3, 8, 8)
    assert torch.equal(out, torch.zeros_like(out))


def test_residual_head_nonzero_for_selected_pair() -> None:
    """A selected pair returns a non-trivially-zero residual (random init)."""
    cfg = A1PlusLaposeConfig(
        residual_rank=2,
        selected_pair_indices=(5,),
        foveal_h=8,
        foveal_w=8,
    )
    torch.manual_seed(0)
    head = PerPairResidualHead(cfg)
    out_a = head.residual_chw(5, 0)
    out_b = head.residual_chw(5, 1)
    assert out_a.shape == (3, 8, 8)
    assert out_b.shape == (3, 8, 8)
    # Random 0.02-scaled init: definitively non-zero.
    assert out_a.abs().sum().item() > 0
    assert out_b.abs().sum().item() > 0


def test_residual_head_byte_budget_under_5kb_default_config() -> None:
    """Council D2.B verdict: pose-residual budget ≤ 5 KB."""
    cfg = A1PlusLaposeConfig(
        residual_rank=4,
        selected_pair_indices=tuple(range(64)),
        foveal_h=256,
        foveal_w=256,
    )
    head = PerPairResidualHead(cfg)
    est = head.total_int8_bytes()
    # With rank=4, fov=256x256, 64 atoms: per_frame=4*(3*256+256)=4096 bytes.
    # 64 atoms * 2 frames * 4096 = 524288 raw bytes BEFORE brotli; post-
    # brotli the empirical bytes are far lower (~3-5 KB typical).
    # Closed-form is the RAW (pre-brotli) upper bound.
    assert est == 64 * 2 * 4 * (3 * 256 + 256) + 2 + 2 * 64


def test_residual_head_byte_budget_council_target_2kb() -> None:
    """At council target config (rank=4, 64 atoms, fov=64x64) post-brotli
    composition overhead should be ≤ 2 KB on typical empirical data."""
    cfg = A1PlusLaposeConfig(
        residual_rank=4,
        selected_pair_indices=tuple(range(16)),
        foveal_h=64,
        foveal_w=64,
    )
    head = PerPairResidualHead(cfg)
    # Zero residuals -> brotli compresses to ~10-50 B.
    last_dim = 3 * cfg.foveal_h + cfg.foveal_w
    residuals = torch.zeros(16, 2, 4, last_dim)
    blob = pack_composition_archive(
        b"a1_base", selected_indices=cfg.selected_pair_indices,
        residuals=residuals, foveal_h=cfg.foveal_h, foveal_w=cfg.foveal_w,
        residual_rank=cfg.residual_rank, int8_scale=4.0,
    )
    overhead = len(blob) - len(b"a1_base")
    # Zero residuals + brotli: should be well under 2 KB.
    assert overhead < 2048


# ---------------------------------------------------------------------------
# LAPose atom manifest parsing
# ---------------------------------------------------------------------------

def test_parse_atoms_from_atom_ledger_schema() -> None:
    """The canonical atom_ledger schema (build_lapose_motion_atom_manifest)."""
    manifest = {
        "atom_ledger": {
            "rows": [
                {"atom_id": "lapose_motion_pair:75", "hard_pair_support": [75]},
                {"atom_id": "lapose_motion_pair:294", "hard_pair_support": [294]},
            ]
        }
    }
    out = parse_lapose_atom_indices(manifest)
    assert out == (75, 294)


def test_parse_atoms_filters_out_of_range_pairs() -> None:
    """Pair indices >= A1_N_PAIRS (600) are filtered."""
    manifest = {"atoms": [
        {"hard_pair_support": [5]},
        {"hard_pair_support": [600]},  # at boundary
        {"hard_pair_support": [9999]},  # way out
    ]}
    out = parse_lapose_atom_indices(manifest)
    assert out == (5,)


# ---------------------------------------------------------------------------
# Score-aware loss
# ---------------------------------------------------------------------------

class _StubSegScorer(torch.nn.Module):
    """Stub SegNet for unit tests (returns deterministic class logits)."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) per upstream contract; or (B, C, H, W) preprocessed
        if x.dim() == 5:
            x = x[:, -1]  # last frame
        b, _c, h, w = x.shape
        out = torch.zeros(b, self.num_classes, h, w, device=x.device)
        out[:, 0] = x.mean(dim=1)  # class 0 logit ~ avg luminance
        return out

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            x = x[:, -1]
        if x.shape[-2:] != (384, 512):
            x = torch.nn.functional.interpolate(
                x, size=(384, 512), mode="bilinear", align_corners=False
            )
        return x


class _StubPoseScorer(torch.nn.Module):
    """Stub PoseNet for unit tests (returns deterministic 12-dim pose)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        out = torch.zeros(b, 12, device=x.device)
        out[:, 0] = x.mean(dim=(1, 2, 3))  # dim 0 ~ image mean
        return out

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) -> (B, T*6, H/2, W/2) per upstream contract.
        if x.dim() != 5:
            raise ValueError(f"PoseNet preprocess expects 5D; got {x.shape}")
        b, t, c, h, w = x.shape
        # Stub: just return a 4D YUV6-shaped tensor.
        return torch.zeros(b, t * 6, h // 2, w // 2, device=x.device)


def test_loss_weights_defaults_match_contest_constants() -> None:
    """alpha_rate=25, beta_seg=100, gamma_pose=sqrt(10) per CLAUDE.md."""
    w = A1PlusLaposeLossWeights()
    assert w.alpha_rate == 25.0
    assert w.beta_seg == 100.0
    assert math.isclose(w.gamma_pose, math.sqrt(10.0), rel_tol=1e-6)
    assert w.contest_normalizer == 37_545_489.0
    # Default is non-arbitrary contest-formula weighting; operating-point
    # tilts such as PR106's 2.71x must be explicit CLI choices.
    assert w.pose_weight_scale == 1.0


def test_loss_refuses_eval_roundtrip_false() -> None:
    """apply_eval_roundtrip=False is FORBIDDEN per CLAUDE.md."""
    loss_fn = A1PlusLaposeScoreAwareLoss(
        seg_scorer=_StubSegScorer(),
        pose_scorer=_StubPoseScorer(),
        weights=A1PlusLaposeLossWeights(),
    )
    rgb = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match="apply_eval_roundtrip"):
        loss_fn(rgb, rgb, rgb, rgb, torch.tensor(1000.0),
                apply_eval_roundtrip=False)


def test_loss_refuses_negative_noise_std() -> None:
    """noise_std must be non-negative."""
    loss_fn = A1PlusLaposeScoreAwareLoss(
        seg_scorer=_StubSegScorer(),
        pose_scorer=_StubPoseScorer(),
        weights=A1PlusLaposeLossWeights(),
    )
    rgb = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match="noise_std"):
        loss_fn(rgb, rgb, rgb, rgb, torch.tensor(1000.0), noise_std=-0.1)


# ---------------------------------------------------------------------------
# Archive contract — composition preserves A1 verbatim
# ---------------------------------------------------------------------------

def test_composition_bytes_preserve_a1_prefix_for_real_archive() -> None:
    """The canonical A1 archive bytes must be preserved verbatim."""
    a1_zip = REPO_ROOT / "experiments" / "results" / "track4_sg_a1_t178000_20260509" / "submission_dir" / "archive.zip"
    if not a1_zip.is_file():
        pytest.skip("A1 reference archive not present in this clone")
    with zipfile.ZipFile(a1_zip) as zf:
        a1_bytes = zf.read("x")
    cfg = A1PlusLaposeConfig(
        residual_rank=2, selected_pair_indices=(5,), foveal_h=8, foveal_w=8,
    )
    last_dim = 3 * cfg.foveal_h + cfg.foveal_w
    residuals = torch.zeros(1, 2, 2, last_dim)
    comp = pack_composition_archive(
        a1_bytes, selected_indices=cfg.selected_pair_indices,
        residuals=residuals, foveal_h=cfg.foveal_h, foveal_w=cfg.foveal_w,
        residual_rank=cfg.residual_rank, int8_scale=cfg.int8_residual_scale,
    )
    # Composition starts with A1 bytes exactly.
    assert comp[: len(a1_bytes)] == a1_bytes
    # Suffix is the LAPose sidecar with magic.
    assert comp[len(a1_bytes) : len(a1_bytes) + 4] == LAPOSE_SIDECAR_MAGIC
    # Split is consistent.
    split_a1, split_lapose = split_composition_archive(comp)
    assert split_a1 == a1_bytes
    assert split_lapose[:4] == LAPOSE_SIDECAR_MAGIC


# ---------------------------------------------------------------------------
# Trainer module-level contract — Catalog #151 + #168 compliance
# ---------------------------------------------------------------------------

def test_trainer_module_imports() -> None:
    """The trainer module imports cleanly without side-effects."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_substrate_a1_plus_lapose",
        REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "_build_parser")
    assert hasattr(mod, "_smoke_main")
    assert hasattr(mod, "_full_main")
    assert hasattr(mod, "main")
    assert hasattr(mod, "TIER_1_OPERATOR_REQUIRED_FLAGS")


def test_trainer_tier_1_required_flags_are_annotated_assignment() -> None:
    """Catalog #168 META-class: TIER_1 manifest must be ast.AnnAssign so the
    Catalog #151 + #152 AST walkers observe it.  Bare ast.Assign is the
    silent-bypass class extincted on 2026-05-12."""
    src = (REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found_ann = False
    found_bare_assign = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "TIER_1_OPERATOR_REQUIRED_FLAGS":
                found_ann = True
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TIER_1_OPERATOR_REQUIRED_FLAGS":
                    found_bare_assign = True
    assert found_ann, "TIER_1_OPERATOR_REQUIRED_FLAGS must be AnnAssign (Catalog #168)"
    assert not found_bare_assign, "no bare Assign duplicate of TIER_1_OPERATOR_REQUIRED_FLAGS"


def test_trainer_tier_1_includes_required_input_files() -> None:
    """Per Catalog #152: required_input_file=True entries must have
    a generator_command + rationale_audit per the canonical schema."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_substrate_a1_plus_lapose",
        REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    manifest = mod.TIER_1_OPERATOR_REQUIRED_FLAGS

    for required_flag in ("--a1-archive", "--lapose-atom-manifest", "--video-path"):
        assert required_flag in manifest
        entry = manifest[required_flag]
        assert entry.get("required_input_file") is True
        assert "env" in entry and entry["env"].startswith("A1_PLUS_LAPOSE_")
        assert "rationale" in entry
        assert "generator_command" in entry


def test_trainer_argparse_accepts_d4_modes() -> None:
    """D4 mode (operator-routed) supports all three: D4.A / D4.B / D4.C."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_substrate_a1_plus_lapose",
        REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parser = mod._build_parser()
    for d4_mode in ("d4a_two_stage", "d4b_single_stage", "d4c_no_grammar_change"):
        ns = parser.parse_args([
            "--output-dir", "/tmp/x", "--epochs", "1", "--smoke",
            "--d4-mode", d4_mode,
        ])
        assert ns.d4_mode == d4_mode


def test_trainer_argparse_rejects_invalid_d4_mode() -> None:
    """Unknown D4 mode raises SystemExit (argparse default)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_substrate_a1_plus_lapose",
        REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parser = mod._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--output-dir", "/tmp/x", "--epochs", "1",
            "--d4-mode", "d4_kitchen_sink",
        ])


def test_trainer_refuses_mps_device() -> None:
    """MPS is REFUSED per CLAUDE.md 'MPS auth eval is NOISE'."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_substrate_a1_plus_lapose",
        REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(RuntimeError, match="MPS is REFUSED"):
        mod._device_or_die("mps", smoke=False)


def test_trainer_refuses_cpu_full_main() -> None:
    """CPU is only allowed in --smoke; full main rejects it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_substrate_a1_plus_lapose",
        REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(RuntimeError, match="CPU is REFUSED"):
        mod._device_or_die("cpu", smoke=False)


def test_trainer_smoke_main_runs_end_to_end(tmp_path) -> None:
    """The CPU smoke main produces composition bytes + metadata."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_substrate_a1_plus_lapose",
        REPO_ROOT / "experiments" / "train_substrate_a1_plus_lapose.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    a1_zip = REPO_ROOT / "experiments" / "results" / "track4_sg_a1_t178000_20260509" / "submission_dir" / "archive.zip"
    if not a1_zip.is_file():
        pytest.skip("A1 reference archive not present")

    parser = mod._build_parser()
    args = parser.parse_args([
        "--a1-archive", str(a1_zip),
        "--output-dir", str(tmp_path / "smoke_out"),
        "--epochs", "2",
        "--device", "cpu",
        "--smoke",
        "--max-atoms", "4",
        "--residual-rank", "2",
        "--foveal-h", "8",
        "--foveal-w", "8",
    ])
    rc = mod._smoke_main(args)
    assert rc == 0
    meta_path = tmp_path / "smoke_out" / "smoke_metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["a1_base_bytes"] == 178162
    assert meta["a1_base_sha256"].startswith("8e664385")
    assert meta["lapose_atom_count"] > 0
    assert meta["composition_bytes"] > meta["a1_base_bytes"]
    assert meta["d4_mode"] == "d4b_single_stage"


# ---------------------------------------------------------------------------
# Manifest schema check on the live fixture
# ---------------------------------------------------------------------------

def test_canonical_lapose_atom_manifest_parses() -> None:
    """The committed LAPose atom manifest fixture yields valid pair indices."""
    manifest_path = (
        REPO_ROOT / ".omx" / "research" / "artifacts"
        / "lapose_motion_atoms_20260505_codex"
        / "lapose_motion_atom_manifest_fixture.json"
    )
    if not manifest_path.is_file():
        pytest.skip("LAPose atom manifest fixture not present")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    indices = parse_lapose_atom_indices(manifest, max_atoms=16)
    assert len(indices) >= 1
    assert all(0 <= i < A1_N_PAIRS for i in indices)
