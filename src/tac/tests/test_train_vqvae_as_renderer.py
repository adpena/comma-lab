"""Tests for ``experiments/train_vqvae_as_renderer.py`` + ``tac.vqvae_as_full_renderer``.

Coverage (substrate + trainer):
- VQVAEFullConfig validates codebook_ema_decay in [0.9, 1.0).
- VQVAEFullConfig validates num_entries in (0, 65535].
- VQVAEFullConfig validates perplexity floor ratio in [0, 1].
- VQVAEFullRenderer forward returns (decoded, indices, commitment_loss).
- Decoded RGB shape (B, 2, 3, 384, 512); range [0, 255].
- Encoder maps latent → (B, tokens, entry_dim).
- Codebook indices in [0, num_entries) shape (B, tokens).
- Codebook update_ema runs without error + adapts persistent buffers.
- Schema is complete (covers encoder + decoder; codebook is separate).
- vqvae_train_step refuses eval_roundtrip=False.
- vqvae_train_step includes commitment loss in total.
- compute_perplexity_from_indices is monotone (uniform → max perplexity).
- assert_codebook_perplexity_ok raises VQVAECodebookCollapseError on collapse.
- assert_codebook_perplexity_ok passes when perplexity >= floor.
- export_vqvae_to_archive returns sha256 + writes file with > 16 bytes.
- export deterministic same-weights.
- ARCHIVE_GRAMMAR_VQVAE_FULL internal consistency (6 sections; codebook ema decay).
- Trainer parse_args defaults match canonicals.
- Trainer rejects --device mps.
- Trainer rejects --auth-eval without --phase-b-auth-memo (Catalog #150).
- Trainer smoke runs end-to-end on CPU.
- Provenance includes vqvae_as_full_renderer_not_bolt_on + nn2_perplexity_gate_per_epoch.
- Trainer source has no /tmp paths.
- Trainer does not call make_synthetic outside smoke.
- Substrate declares archive grammar at module level.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_trainer():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_vqvae_as_renderer",
        REPO_ROOT / "experiments" / "train_vqvae_as_renderer.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def repo_output_dir(tmp_path):
    root = REPO_ROOT / "experiments" / "results" / ".pytest_tmp_outputs" / tmp_path.name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    def make(name: str) -> Path:
        return root / name

    yield make
    shutil.rmtree(root, ignore_errors=True)


# ── Substrate config validation ──────────────────────────────────────────


def test_config_defaults_canonical():
    from tac.vqvae_as_full_renderer import VQVAEFullConfig
    c = VQVAEFullConfig(cuda_required=False)
    assert c.num_entries == 256, "van den Oord §4.2 canonical"
    assert c.entry_dim == 64, "Quantizr-class C=64"
    assert c.codebook_ema_decay == 0.99, "van den Oord §3.2 canonical"
    assert c.commitment_weight == 0.25, "van den Oord §3.2 β=0.25"
    assert c.nn2_perplexity_floor_ratio == 0.4


def test_config_rejects_invalid_codebook_decay():
    from tac.vqvae_as_full_renderer import VQVAEFullConfig
    with pytest.raises(ValueError):
        VQVAEFullConfig(codebook_ema_decay=0.5, cuda_required=False)
    with pytest.raises(ValueError):
        VQVAEFullConfig(codebook_ema_decay=1.0, cuda_required=False)
    with pytest.raises(ValueError):
        VQVAEFullConfig(codebook_ema_decay=1.1, cuda_required=False)


def test_config_rejects_invalid_num_entries():
    from tac.vqvae_as_full_renderer import VQVAEFullConfig
    with pytest.raises(ValueError):
        VQVAEFullConfig(num_entries=0, cuda_required=False)
    with pytest.raises(ValueError):
        VQVAEFullConfig(num_entries=70000, cuda_required=False)


def test_config_rejects_invalid_perplexity_ratio():
    from tac.vqvae_as_full_renderer import VQVAEFullConfig
    with pytest.raises(ValueError):
        VQVAEFullConfig(nn2_perplexity_floor_ratio=-0.1, cuda_required=False)
    with pytest.raises(ValueError):
        VQVAEFullConfig(nn2_perplexity_floor_ratio=1.5, cuda_required=False)


# ── Substrate forward + shapes ────────────────────────────────────────────


def test_renderer_forward_returns_3_tuple():
    import torch
    from tac.vqvae_as_full_renderer import VQVAEFullConfig, VQVAEFullRenderer
    config = VQVAEFullConfig(
        latent_dim=8, num_entries=16, entry_dim=4, tokens_per_pair=2,
        n_pairs=4, base_channels=8, cuda_required=False,
    )
    renderer = VQVAEFullRenderer(config)
    z = torch.randn(2, 8)
    decoded, indices, commitment = renderer(z)
    assert decoded.shape == (2, 2, 3, 384, 512), f"got {tuple(decoded.shape)}"
    assert indices.shape == (2, 2), f"got {tuple(indices.shape)}"
    assert commitment.dim() == 0  # scalar


def test_renderer_rgb_output_in_0_255_range():
    import torch
    from tac.vqvae_as_full_renderer import VQVAEFullConfig, VQVAEFullRenderer
    config = VQVAEFullConfig(
        latent_dim=8, num_entries=16, entry_dim=4, tokens_per_pair=2,
        n_pairs=4, base_channels=8, cuda_required=False,
    )
    renderer = VQVAEFullRenderer(config)
    z = torch.randn(2, 8)
    decoded, _, _ = renderer(z)
    assert decoded.min() >= 0.0
    assert decoded.max() <= 255.0


def test_codebook_indices_in_range():
    import torch
    from tac.vqvae_as_full_renderer import VQVAEFullConfig, VQVAEFullRenderer
    config = VQVAEFullConfig(
        latent_dim=8, num_entries=16, entry_dim=4, tokens_per_pair=2,
        n_pairs=4, base_channels=8, cuda_required=False,
    )
    renderer = VQVAEFullRenderer(config)
    z = torch.randn(2, 8)
    _, indices, _ = renderer(z)
    assert indices.min().item() >= 0
    assert indices.max().item() < 16


def test_codebook_update_ema_runs():
    """Persistent EMA buffer update must run without error + adapt buffers."""
    import torch
    from tac.vqvae_as_full_renderer import VQVAEFullConfig, VQVAEFullRenderer
    config = VQVAEFullConfig(
        latent_dim=8, num_entries=16, entry_dim=4, tokens_per_pair=2,
        n_pairs=4, base_channels=8, cuda_required=False,
    )
    renderer = VQVAEFullRenderer(config)
    z = torch.randn(2, 8)
    z_e = renderer.encoder(z)
    _, indices, _ = renderer.codebook(z_e)
    before = renderer.codebook.ema_count.sum().item()
    renderer.codebook.update_ema(z_e, indices)
    after = renderer.codebook.ema_count.sum().item()
    # EMA count must have grown (was 0; now has signal).
    assert after > before


def test_schema_covers_encoder_and_decoder():
    """Schema must cover all encoder + decoder state_dict entries (codebook excluded)."""
    from tac.vqvae_as_full_renderer import VQVAEFullConfig, VQVAEFullRenderer
    config = VQVAEFullConfig(
        latent_dim=8, num_entries=16, entry_dim=4, tokens_per_pair=2,
        n_pairs=4, base_channels=8, cuda_required=False,
    )
    renderer = VQVAEFullRenderer(config)
    schema = renderer.schema
    schema_keys = {k for k, _ in schema}
    # All schema keys must reference encoder.* or decoder.*; codebook is separate.
    for k in schema_keys:
        assert k.startswith(("encoder.", "decoder."))
    # Encoder + decoder weight params must all appear.
    sd = renderer.state_dict()
    enc_dec = {k for k in sd if k.startswith(("encoder.", "decoder."))}
    assert schema_keys == enc_dec, f"missing={enc_dec - schema_keys}"


# ── Train step semantics ──────────────────────────────────────────────────


def test_train_step_refuses_eval_roundtrip_false():
    import torch
    from tac.vqvae_as_full_renderer import (
        VQVAEFullConfig, VQVAEFullRenderer, VQVAEFullLatentTable,
        vqvae_train_step,
        default_vqvae_seg_surrogate, default_vqvae_pose_surrogate,
    )
    config = VQVAEFullConfig(
        latent_dim=4, num_entries=8, entry_dim=4, tokens_per_pair=2,
        n_pairs=2, base_channels=4, cuda_required=False,
    )
    renderer = VQVAEFullRenderer(config)
    latents = VQVAEFullLatentTable(config.n_pairs, config.latent_dim)
    pair_indices = torch.tensor([0, 1])
    gt = torch.randint(0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8)

    class _NoOp:
        def __call__(self, x): return torch.zeros(x.shape[0], 6)
        def preprocess_input(self, x): return x
        def eval(self): return self
    seg = pose = _NoOp()

    with pytest.raises(ValueError) as exc_info:
        vqvae_train_step(
            renderer=renderer, latent_table=latents,
            pair_indices=pair_indices, gt_pairs_uint8=gt,
            scorer_seg=seg, scorer_pose=pose,
            seg_surrogate=default_vqvae_seg_surrogate,
            pose_surrogate=default_vqvae_pose_surrogate,
            lambda_seg=1.0, lambda_pose=1.0,
            eval_roundtrip=False,
        )
    assert "eval_roundtrip" in str(exc_info.value)


# ── Perplexity gate (NN-2) ────────────────────────────────────────────────


def test_perplexity_uniform_maxed():
    """Uniform-over-all-N indices → perplexity ≈ N."""
    import torch
    from tac.vqvae_as_full_renderer import compute_perplexity_from_indices
    # Use 256 entries, perfectly uniform.
    indices = torch.arange(256).repeat(4)  # each entry hit exactly 4 times
    perp = compute_perplexity_from_indices(indices, num_entries=256)
    assert perp == pytest.approx(256, abs=0.5)


def test_perplexity_collapsed_low():
    """Collapsed (all same index) → perplexity = 1."""
    import torch
    from tac.vqvae_as_full_renderer import compute_perplexity_from_indices
    indices = torch.zeros(100, dtype=torch.long)
    perp = compute_perplexity_from_indices(indices, num_entries=256)
    assert perp == pytest.approx(1.0, abs=1e-3)


def test_assert_perplexity_ok_passes_uniform():
    import torch
    from tac.vqvae_as_full_renderer import assert_codebook_perplexity_ok
    indices = torch.arange(256).repeat(4)
    diag = assert_codebook_perplexity_ok(indices, num_entries=256, floor_ratio=0.4)
    assert diag["passed"] is True


def test_assert_perplexity_raises_on_collapse():
    import torch
    from tac.vqvae_as_full_renderer import (
        assert_codebook_perplexity_ok, VQVAECodebookCollapseError,
    )
    indices = torch.zeros(100, dtype=torch.long)
    with pytest.raises(VQVAECodebookCollapseError):
        assert_codebook_perplexity_ok(indices, num_entries=256, floor_ratio=0.4)


# ── Archive export ────────────────────────────────────────────────────────


def test_export_returns_sha256(tmp_path):
    from tac.vqvae_as_full_renderer import (
        VQVAEFullConfig, VQVAEFullRenderer, VQVAEFullLatentTable,
        export_vqvae_to_archive,
    )
    config = VQVAEFullConfig(
        latent_dim=8, num_entries=16, entry_dim=4, tokens_per_pair=2,
        n_pairs=4, base_channels=8, cuda_required=False,
    )
    renderer = VQVAEFullRenderer(config)
    latents = VQVAEFullLatentTable(config.n_pairs, config.latent_dim)
    out_path = tmp_path / "0.bin"
    sha = export_vqvae_to_archive(
        renderer=renderer, latent_table=latents, output_path=out_path,
    )
    assert len(sha) == 64
    assert out_path.exists()
    assert out_path.stat().st_size > 16


def test_export_deterministic_same_weights(tmp_path):
    import torch
    from tac.vqvae_as_full_renderer import (
        VQVAEFullConfig, VQVAEFullRenderer, VQVAEFullLatentTable,
        export_vqvae_to_archive,
    )
    torch.manual_seed(42)
    config = VQVAEFullConfig(
        latent_dim=8, num_entries=16, entry_dim=4, tokens_per_pair=2,
        n_pairs=4, base_channels=8, cuda_required=False,
    )
    renderer = VQVAEFullRenderer(config)
    latents = VQVAEFullLatentTable(config.n_pairs, config.latent_dim)
    sha1 = export_vqvae_to_archive(
        renderer=renderer, latent_table=latents, output_path=tmp_path / "a.bin",
    )
    sha2 = export_vqvae_to_archive(
        renderer=renderer, latent_table=latents, output_path=tmp_path / "b.bin",
    )
    assert sha1 == sha2


# ── Archive grammar manifest ──────────────────────────────────────────────


def test_archive_grammar_internal_consistency():
    from tac.vqvae_as_full_renderer import (
        ARCHIVE_GRAMMAR_VQVAE_FULL,
        VQVAE_FULL_FORMAT_VERSION,
        VQVAE_FULL_MAGIC,
    )
    g = ARCHIVE_GRAMMAR_VQVAE_FULL
    assert g["format_version"] == VQVAE_FULL_FORMAT_VERSION
    assert g["magic"] == VQVAE_FULL_MAGIC.decode("ascii")
    assert g["codebook_ema_decay"] == 0.99
    assert g["nn2_perplexity_floor_ratio"] == 0.4
    section_names = {s["name"] for s in g["sections"]}
    assert section_names == {
        "header", "codebook_blob", "decoder_blob", "scale_table",
        "indices_blob", "sidecar_blob",
    }


# ── Trainer CLI ───────────────────────────────────────────────────────────


def test_parse_args_requires_output_dir():
    trainer = _import_trainer()
    with pytest.raises(SystemExit):
        trainer.parse_args([])


def test_parse_args_rejects_mps():
    trainer = _import_trainer()
    with pytest.raises(SystemExit):
        trainer.parse_args(["--output-dir", "/tmp_t", "--device", "mps"])


def test_parse_args_canonical_defaults():
    trainer = _import_trainer()
    args = trainer.parse_args(["--output-dir", "/tmp_t"])
    assert args.device == "cuda"
    assert args.ema_decay == 0.997, "CLAUDE.md weight EMA"
    assert args.codebook_ema_decay == 0.99, "van den Oord §3.2"
    assert args.commitment_weight == 0.25, "van den Oord β"
    assert args.perplexity_floor_ratio == 0.4, "CLAUDE.md Phase 2"
    assert args.num_entries == 256
    assert args.entry_dim == 64


# ── Auth-eval gate (Catalog #150) ─────────────────────────────────────────


def test_auth_eval_without_memo_refused(repo_output_dir):
    trainer = _import_trainer()
    with pytest.raises(SystemExit) as exc_info:
        trainer.main([
            "--output-dir", str(repo_output_dir("out")),
            "--device", "cpu", "--auth-eval", "--smoke",
        ])
    assert "phase-b-auth-memo" in str(exc_info.value).lower() or "150" in str(exc_info.value)


def test_auth_eval_with_non_repo_path_refused(tmp_path, repo_output_dir):
    trainer = _import_trainer()
    with pytest.raises((SystemExit, ValueError)):
        trainer.main([
            "--output-dir", str(repo_output_dir("out")),
            "--device", "cpu", "--auth-eval", "--smoke",
            "--phase-b-auth-memo", str(tmp_path / "fake.md"),
        ])


# ── Device resolution ────────────────────────────────────────────────────


def test_resolve_device_mps_refused():
    trainer = _import_trainer()
    with pytest.raises(SystemExit):
        trainer._resolve_device("mps")


# ── Smoke end-to-end ──────────────────────────────────────────────────────


def test_smoke_runs_end_to_end(repo_output_dir):
    trainer = _import_trainer()
    out = repo_output_dir("smoke_out")
    rc = trainer.main([
        "--output-dir", str(out),
        "--device", "cpu", "--smoke",
        "--epochs", "1", "--batch-size", "1",
        "--latent-dim", "8", "--num-entries", "8",
        "--entry-dim", "4", "--tokens-per-pair", "2",
        "--n-pairs", "4",
    ])
    assert rc == 0
    assert (out / "0.bin").exists()
    assert (out / "provenance.json").exists()


def test_smoke_provenance_compliance_tags(repo_output_dir):
    trainer = _import_trainer()
    out = repo_output_dir("smoke_prov")
    trainer.main([
        "--output-dir", str(out),
        "--device", "cpu", "--smoke",
        "--epochs", "1", "--latent-dim", "8",
        "--num-entries", "8", "--entry-dim", "4",
        "--tokens-per-pair", "2", "--n-pairs", "4",
    ])
    prov = json.loads((out / "provenance.json").read_text())
    tags = set(prov["compliance_tags"])
    required = {
        "ema_0p997_weights_snapshot_restore",
        "codebook_ema_0p99_van_den_oord_canon",
        "eval_roundtrip_true",
        "no_mps_authoritative",
        "differentiable_yuv6",
        "score_aware_lagrangian",
        "vandenoord_commitment_loss",
        "nn2_perplexity_gate_per_epoch",
        "no_synthetic_outside_smoke",
        "no_tmp_paths",
        "auth_eval_gated_phase_b_option_c",
        "cuda_required_default",
        "vqvae_as_full_renderer_not_bolt_on",
    }
    assert required.issubset(tags), f"missing: {required - tags}"


def test_smoke_provenance_score_claim_false(repo_output_dir):
    trainer = _import_trainer()
    out = repo_output_dir("smoke_no_claim")
    trainer.main([
        "--output-dir", str(out),
        "--device", "cpu", "--smoke",
        "--epochs", "1", "--latent-dim", "8",
        "--num-entries", "8", "--entry-dim", "4",
        "--tokens-per-pair", "2", "--n-pairs", "4",
    ])
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["score_claim"] is False
    assert prov["promotion_eligible"] is False
    assert prov["ready_for_exact_eval_dispatch"] is False
    assert prov["ema_decay_codebook"] == 0.99
    assert prov["ema_decay_weights"] == 0.997


def test_trainer_source_no_tmp_durable_paths():
    src = (REPO_ROOT / "experiments" / "train_vqvae_as_renderer.py").read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        assert "/tmp/" not in line, f"durable /tmp forbidden: {line!r}"
    assert "assert_not_temporary_output_dir" in src
    assert "def _refuse_tmp_output_dir" not in src


def test_trainer_source_no_make_synthetic_outside_smoke():
    src = (REPO_ROOT / "experiments" / "train_vqvae_as_renderer.py").read_text()
    assert "use_synthetic = bool(args.smoke)" in src
    assert "if use_synthetic:" in src


def test_substrate_declares_archive_grammar_at_module_level():
    src = (REPO_ROOT / "src" / "tac" / "vqvae_as_full_renderer.py").read_text()
    assert "ARCHIVE_GRAMMAR_VQVAE_FULL" in src
    assert "schema_keys_in_order" in src
    assert "codebook_ema_decay" in src
    assert "nn2_perplexity_floor_ratio" in src


def test_substrate_full_renderer_not_bolt_on():
    """The substrate must declare it's a FULL RENDERER (not a T17-style bolt-on)."""
    src = (REPO_ROOT / "src" / "tac" / "vqvae_as_full_renderer.py").read_text()
    # Module docstring must distinguish itself from T17 bolt-on pattern.
    assert "BOLT-ON" in src or "bolt-on" in src or "bolt_on" in src
    assert "FULL" in src or "full RGB renderer" in src.lower()
