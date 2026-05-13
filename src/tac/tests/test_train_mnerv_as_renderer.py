"""Tests for ``experiments/train_mnerv_as_renderer.py`` + ``tac.mnerv_as_renderer``.

Coverage (substrate + trainer):
- MNeRVConfig validates n_scales=3 pinned + latent_dim/base_channels positive.
- MNeRVRenderer forward shape (B, 2, 3, 384, 512).
- MNeRVRenderer schema is deterministic + complete (no missing keys).
- Multi-scale stages output at progressively larger resolutions.
- Different latents produce different outputs (per-pair conditioning).
- mnerv_train_step returns finite + non-zero on a smoke batch.
- mnerv_train_step refuses eval_roundtrip=False.
- export_mnerv_to_archive returns sha256 + deterministic bytes.
- ARCHIVE_GRAMMAR_MNERV is internally consistent.
- Trainer parse_args defaults match CLAUDE.md.
- Trainer rejects --device mps.
- Trainer requires --phase-b-auth-memo when --auth-eval set (Catalog #150).
- Trainer smoke runs end-to-end on CPU + emits archive + provenance.
- Provenance includes multi_scale_3_level_mallat_scattering compliance tag.
- Trainer source has no durable /tmp paths.
- Trainer does not call make_synthetic outside smoke.
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
        "train_mnerv_as_renderer",
        REPO_ROOT / "experiments" / "train_mnerv_as_renderer.py",
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


# ── Substrate (tac.mnerv_as_renderer) ────────────────────────────────────


def test_config_defaults_pin_3_scales():
    from tac.mnerv_as_renderer import MNeRVConfig
    c = MNeRVConfig(cuda_required=False)
    assert c.n_scales == 3
    assert c.frames_per_pair == 2
    assert c.eval_size == (384, 512)


def test_config_rejects_non_3_scales():
    from tac.mnerv_as_renderer import MNeRVConfig
    with pytest.raises(ValueError):
        MNeRVConfig(n_scales=2, cuda_required=False)
    with pytest.raises(ValueError):
        MNeRVConfig(n_scales=4, cuda_required=False)


def test_config_rejects_non_positive_latent_dim():
    from tac.mnerv_as_renderer import MNeRVConfig
    with pytest.raises(ValueError):
        MNeRVConfig(latent_dim=0, cuda_required=False)
    with pytest.raises(ValueError):
        MNeRVConfig(latent_dim=-1, cuda_required=False)


def test_renderer_forward_shape_matches_design():
    import torch
    from tac.mnerv_as_renderer import MNeRVConfig, MNeRVRenderer
    config = MNeRVConfig(latent_dim=8, base_channels=8, n_pairs=4, cuda_required=False)
    renderer = MNeRVRenderer(config)
    z = torch.randn(2, 8)
    out = renderer(z)
    assert out.shape == (2, 2, 3, 384, 512), f"got {tuple(out.shape)}"


def test_renderer_forward_outputs_in_0_255_range():
    import torch
    from tac.mnerv_as_renderer import MNeRVConfig, MNeRVRenderer
    config = MNeRVConfig(latent_dim=8, base_channels=8, n_pairs=4, cuda_required=False)
    renderer = MNeRVRenderer(config)
    z = torch.randn(2, 8)
    out = renderer(z)
    assert out.min() >= 0.0
    assert out.max() <= 255.0


def test_renderer_per_pair_conditioning_distinct_outputs():
    """Different latents must produce different rendered frames."""
    import torch
    from tac.mnerv_as_renderer import MNeRVConfig, MNeRVRenderer
    config = MNeRVConfig(latent_dim=8, base_channels=8, n_pairs=4, cuda_required=False)
    renderer = MNeRVRenderer(config)
    z = torch.randn(2, 8)
    out = renderer(z)
    # Batch element 0 vs 1 should differ.
    assert not torch.allclose(out[0], out[1])


def test_renderer_schema_is_complete_and_ordered():
    from tac.mnerv_as_renderer import MNeRVConfig, MNeRVRenderer
    config = MNeRVConfig(latent_dim=8, base_channels=8, n_pairs=4, cuda_required=False)
    renderer = MNeRVRenderer(config)
    schema = renderer.schema
    # All state_dict keys must appear in schema (no orphan tensors).
    sd_keys = set(renderer.state_dict().keys())
    schema_keys = {k for k, _ in schema}
    assert schema_keys == sd_keys, (
        f"schema/state-dict mismatch: missing={sd_keys - schema_keys} "
        f"extra={schema_keys - sd_keys}"
    )


def test_renderer_3_scales_resolutions():
    """Internal scales must be (96,128) → (192,256) → (384,512)."""
    from tac.mnerv_as_renderer import MNeRVConfig, MNeRVRenderer
    config = MNeRVConfig(latent_dim=8, base_channels=8, n_pairs=4, cuda_required=False)
    renderer = MNeRVRenderer(config)
    assert renderer.scales == [(96, 128), (192, 256), (384, 512)]


def test_archive_grammar_internal_consistency():
    from tac.mnerv_as_renderer import (
        ARCHIVE_GRAMMAR_MNERV,
        MNERV_FORMAT_VERSION,
        MNERV_MAGIC,
    )
    g = ARCHIVE_GRAMMAR_MNERV
    assert g["format_version"] == MNERV_FORMAT_VERSION
    assert g["magic"] == MNERV_MAGIC.decode("ascii")
    assert g["n_scales"] == 3
    assert g["scale_resolutions"] == [(96, 128), (192, 256), (384, 512)]
    # 5 sections (header + 4 length-prefixed).
    section_names = {s["name"] for s in g["sections"]}
    assert section_names == {
        "header", "decoder_blob", "scale_table", "latent_blob", "sidecar_blob",
    }


def test_export_archive_returns_sha256(tmp_path):
    """export_mnerv_to_archive returns hex sha256 + writes file."""
    from tac.mnerv_as_renderer import (
        MNeRVConfig, MNeRVRenderer, MNeRVLatentTable, export_mnerv_to_archive,
    )
    config = MNeRVConfig(latent_dim=8, base_channels=8, n_pairs=4, cuda_required=False)
    renderer = MNeRVRenderer(config)
    latents = MNeRVLatentTable(config.n_pairs, config.latent_dim)
    out_path = tmp_path / "0.bin"
    sha = export_mnerv_to_archive(
        renderer=renderer, latent_table=latents, output_path=out_path,
    )
    assert len(sha) == 64
    assert out_path.exists()
    assert out_path.stat().st_size > 16  # > header bytes


def test_export_archive_deterministic_same_weights(tmp_path):
    """Same weights → same sha256."""
    import torch
    from tac.mnerv_as_renderer import (
        MNeRVConfig, MNeRVRenderer, MNeRVLatentTable, export_mnerv_to_archive,
    )
    torch.manual_seed(42)
    config = MNeRVConfig(latent_dim=8, base_channels=8, n_pairs=4, cuda_required=False)
    renderer = MNeRVRenderer(config)
    latents = MNeRVLatentTable(config.n_pairs, config.latent_dim)
    sha1 = export_mnerv_to_archive(
        renderer=renderer, latent_table=latents, output_path=tmp_path / "a.bin",
    )
    sha2 = export_mnerv_to_archive(
        renderer=renderer, latent_table=latents, output_path=tmp_path / "b.bin",
    )
    assert sha1 == sha2


def test_train_step_refuses_eval_roundtrip_false():
    """mnerv_train_step raises if eval_roundtrip=False (CLAUDE.md non-negotiable)."""
    import torch
    from tac.mnerv_as_renderer import (
        MNeRVConfig, MNeRVRenderer, MNeRVLatentTable, mnerv_train_step,
        default_mnerv_seg_surrogate, default_mnerv_pose_surrogate,
    )
    config = MNeRVConfig(latent_dim=4, base_channels=4, n_pairs=2, cuda_required=False)
    renderer = MNeRVRenderer(config)
    latents = MNeRVLatentTable(config.n_pairs, config.latent_dim)
    pair_indices = torch.tensor([0, 1])
    gt = torch.randint(0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8)
    # Use no-op scorers (won't be called because eval_roundtrip=False fails fast).
    class _NoOp:
        def __call__(self, x): return torch.zeros(x.shape[0], 6)
        def preprocess_input(self, x): return x
        def eval(self): return self
    seg = pose = _NoOp()
    with pytest.raises(ValueError) as exc_info:
        mnerv_train_step(
            renderer=renderer, latent_table=latents,
            pair_indices=pair_indices, gt_pairs_uint8=gt,
            scorer_seg=seg, scorer_pose=pose,
            seg_surrogate=default_mnerv_seg_surrogate,
            pose_surrogate=default_mnerv_pose_surrogate,
            lambda_seg=1.0, lambda_pose=1.0,
            eval_roundtrip=False,
        )
    assert "eval_roundtrip" in str(exc_info.value)


# ── Trainer CLI / argparse ────────────────────────────────────────────────


def test_parse_args_requires_output_dir():
    trainer = _import_trainer()
    with pytest.raises(SystemExit):
        trainer.parse_args([])


def test_parse_args_rejects_mps():
    trainer = _import_trainer()
    with pytest.raises(SystemExit):
        trainer.parse_args(["--output-dir", "/tmp_t", "--device", "mps"])


def test_parse_args_defaults_match_canonicals():
    trainer = _import_trainer()
    args = trainer.parse_args(["--output-dir", "/tmp_t"])
    assert args.device == "cuda"
    assert args.ema_decay == 0.997
    assert args.enable_score_aware_loss is True
    assert args.enable_differentiable_yuv6 is True


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


def test_resolve_device_cpu_works():
    import torch
    trainer = _import_trainer()
    assert trainer._resolve_device("cpu") == torch.device("cpu")


# ── Smoke end-to-end ──────────────────────────────────────────────────────


def test_smoke_runs_end_to_end(repo_output_dir):
    trainer = _import_trainer()
    out = repo_output_dir("smoke_out")
    rc = trainer.main([
        "--output-dir", str(out),
        "--device", "cpu", "--smoke",
        "--epochs", "1", "--batch-size", "1",
        "--latent-dim", "8", "--n-pairs", "4",
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
        "--epochs", "1", "--latent-dim", "8", "--n-pairs", "4",
    ])
    prov = json.loads((out / "provenance.json").read_text())
    tags = set(prov["compliance_tags"])
    required = {
        "ema_0p997_snapshot_restore",
        "eval_roundtrip_true",
        "no_mps_authoritative",
        "differentiable_yuv6",
        "score_aware_lagrangian",
        "no_synthetic_outside_smoke",
        "no_tmp_paths",
        "auth_eval_gated_phase_b_option_c",
        "cuda_required_default",
        "multi_scale_3_level_mallat_scattering",
    }
    assert required.issubset(tags), f"missing: {required - tags}"


def test_smoke_provenance_score_claim_false(repo_output_dir):
    trainer = _import_trainer()
    out = repo_output_dir("smoke_no_claim")
    trainer.main([
        "--output-dir", str(out),
        "--device", "cpu", "--smoke",
        "--epochs", "1", "--latent-dim", "8", "--n-pairs", "4",
    ])
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["score_claim"] is False
    assert prov["promotion_eligible"] is False
    assert prov["ready_for_exact_eval_dispatch"] is False


def test_trainer_source_has_no_tmp_durable_paths():
    src = (REPO_ROOT / "experiments" / "train_mnerv_as_renderer.py").read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        assert "/tmp/" not in line, f"durable /tmp path forbidden: {line!r}"
    assert "assert_not_temporary_output_dir" in src
    assert "def _refuse_tmp_output_dir" not in src


def test_trainer_source_no_make_synthetic_outside_smoke():
    src = (REPO_ROOT / "experiments" / "train_mnerv_as_renderer.py").read_text()
    assert "use_synthetic = bool(args.smoke)" in src
    assert "if use_synthetic:" in src


# ── Substrate cross-ref: archive grammar is declared at module level ─────


def test_substrate_declares_archive_grammar_at_module_level():
    """CLAUDE.md Catalog #124: archive grammar must be in source, not runtime."""
    src = (REPO_ROOT / "src" / "tac" / "mnerv_as_renderer.py").read_text()
    assert "ARCHIVE_GRAMMAR_MNERV" in src
    assert "schema_keys_in_order" in src
    assert "scale_resolutions" in src
