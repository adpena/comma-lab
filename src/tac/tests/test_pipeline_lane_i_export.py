"""Verify `experiments/pipeline.py` `step_export` correctly dispatches on
the Lane I (Cool-Chic / C3 residual) variants.

This is the Phase 2 coverage for the Lane I work (2026-04-27). Without it,
a CoolChic / C3 training run would produce a checkpoint that step_export
would silently route through the canonical AsymmetricPairGenerator FP4A
path, hit a strict load_state_dict mismatch, and waste the GPU dispatch.

Test approach:
  * Build a small CoolChic (or C3) PairGenerator, save a fake `.pt`
    checkpoint with `__meta__["variant"]` = the right value plus the
    matching architecture knobs, then invoke `step_export` and assert the
    resulting `.bin` has the right magic bytes (CCh1 / C3R1).
  * Round-trip via `_load_renderer` to confirm the binary is consumable.
  * Verify the `.done_export` marker carries the variant + residual_quant_bits
    so downstream cache invalidation logic can reason about Lane I exports
    distinctly from FP4A exports.
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR.parent))


def _save_lane_i_checkpoint(
    out_path: Path,
    model: torch.nn.Module,
    *,
    variant: str,
    arch_overrides: dict | None = None,
):
    """Mimic train_renderer.py's saved-checkpoint format: state_dict + __meta__.

    The trainer stores a fp32 .pt at `out_dir/renderer_<tag>_best_fp32.pt`
    with `model_state_dict` + `__meta__` (full arch + variant). We replicate
    the relevant subset so step_export's `_arch(...)` lookup picks the
    right values.
    """
    meta: dict = {
        "variant": variant,
        "fp4_codebook": "default",
        "fp4_robust_scale": False,
        # Architecture (minimum needed for Lane I dispatch).
        "num_classes": 5,
        "embed_dim": int(model.renderer.class_embed_dim if variant == "coolchic_renderer"
                          else model.renderer.base_renderer.class_embed_dim),
        "latent_ch": int(
            model.renderer.latent_ch if variant == "coolchic_renderer"
            else model.renderer.base_renderer.latent_ch
        ),
        "base_ch": int(
            model.renderer.hidden if variant == "coolchic_renderer"
            else model.renderer.base_renderer.hidden
        ),
        "motion_hidden": 16,
        "latent_shapes": [list(s) for s in (
            model.renderer.latent_shapes if variant == "coolchic_renderer"
            else model.renderer.base_renderer.latent_shapes
        )],
        "blend_mode": getattr(model, "blend_mode", "scalar"),
        "noise_mode": getattr(model, "noise_mode", "deterministic"),
    }
    if variant == "c3_residual_renderer":
        meta.update({
            "residual_hidden": int(model.renderer.residual_hidden),
            "residual_layers": int(model.renderer.residual_layers),
            "residual_scale": float(model.renderer.residual_scale),
        })
    if arch_overrides:
        meta.update(arch_overrides)
    torch.save({
        "model_state_dict": model.state_dict(),
        "__meta__": meta,
    }, str(out_path))


def _build_coolchic():
    from tac.contrib.coolchic_renderer import build_coolchic_renderer
    torch.manual_seed(11)
    m = build_coolchic_renderer(
        num_classes=5, embed_dim=4, latent_ch=4, hidden=16,
        motion_hidden=16, latent_shapes=((2, 3), (4, 6)),
    )
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    m.eval()
    return m


def _build_c3():
    from tac.contrib.coolchic_renderer import build_c3_residual_renderer
    torch.manual_seed(11)
    m = build_c3_residual_renderer(
        num_classes=5, embed_dim=4, latent_ch=2, hidden=12,
        motion_hidden=16, residual_hidden=16, residual_layers=2,
        residual_scale=10.0, latent_shapes=((2, 3), (4, 6)),
    )
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    m.eval()
    return m


def _make_cfg(tmp_path: Path, ckpt_path: Path, **overrides):
    """Build a minimum PipelineConfig for step_export. The actual cfg has
    many more fields (training/QAT/pose-TTO knobs we don't exercise here);
    we splat the dataclass defaults and then overlay step_export-relevant
    fields."""
    from experiments.pipeline import PipelineConfig
    defaults = {f.name: f.default for f in fields(PipelineConfig)}
    defaults["video"] = ""
    defaults["checkpoint"] = str(ckpt_path)
    defaults["output_dir"] = str(tmp_path / "out")
    defaults["device"] = "cpu"
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestStepExportDispatchesCoolChic:
    def test_coolchic_meta_routes_to_CCh1(self, tmp_path: Path):
        from experiments.pipeline import step_export

        model = _build_coolchic()
        ckpt = tmp_path / "renderer_coolchic_best_fp32.pt"
        _save_lane_i_checkpoint(ckpt, model, variant="coolchic_renderer")

        cfg = _make_cfg(tmp_path, ckpt, variant="coolchic_renderer")
        bin_path = step_export(cfg, iteration=0)

        assert bin_path.exists()
        assert bin_path.read_bytes()[:4] == b"CCh1", \
            "step_export with variant=coolchic_renderer must produce CCh1 magic"

    def test_done_export_marker_records_variant(self, tmp_path: Path):
        from experiments.pipeline import step_export

        model = _build_coolchic()
        ckpt = tmp_path / "renderer_coolchic_best_fp32.pt"
        _save_lane_i_checkpoint(ckpt, model, variant="coolchic_renderer")

        cfg = _make_cfg(tmp_path, ckpt, variant="coolchic_renderer")
        step_export(cfg, iteration=0)

        done = Path(cfg.output_dir) / "iter_0" / ".done_export"
        assert done.exists()
        meta = json.loads(done.read_text())
        assert meta.get("variant") == "coolchic_renderer", \
            ".done_export must record variant for cache-invalidation reasoning"


class TestStepExportDispatchesC3Residual:
    def test_c3_meta_routes_to_C3R1(self, tmp_path: Path):
        from experiments.pipeline import step_export

        model = _build_c3()
        ckpt = tmp_path / "renderer_c3_best_fp32.pt"
        _save_lane_i_checkpoint(ckpt, model, variant="c3_residual_renderer")

        cfg = _make_cfg(tmp_path, ckpt, variant="c3_residual_renderer")
        bin_path = step_export(cfg, iteration=0)
        assert bin_path.read_bytes()[:4] == b"C3R1"

    def test_c3_with_residual_quant_bits_8(self, tmp_path: Path):
        """Phase 3 wiring: cfg.residual_quant_bits=8 must thread through to
        the exporter, producing a smaller-residual-error binary."""
        from experiments.pipeline import step_export

        model = _build_c3()
        ckpt = tmp_path / "renderer_c3_best_fp32.pt"
        _save_lane_i_checkpoint(ckpt, model, variant="c3_residual_renderer")

        cfg = _make_cfg(
            tmp_path, ckpt,
            variant="c3_residual_renderer",
            residual_quant_bits=8,
        )
        bin_path = step_export(cfg, iteration=0)

        # Binary loads cleanly. The header MUST record residual_quant_bits=8
        # — the loader uses it to dispatch the residual_net layers to the
        # int dequantizer instead of the FP4 path.
        from tac.renderer_export import load_c3_residual_renderer
        loaded = load_c3_residual_renderer(bin_path)
        # Smoke-forward: verify the loaded model produces finite output.
        mask_t = torch.randint(0, 5, (1, 12, 18))
        mask_t1 = torch.randint(0, 5, (1, 12, 18))
        with torch.no_grad():
            out = loaded(mask_t, mask_t1)
        assert torch.isfinite(out).all()

        done = Path(cfg.output_dir) / "iter_0" / ".done_export"
        meta = json.loads(done.read_text())
        assert meta.get("residual_quant_bits") == 8


class TestStepExportLegacyPathStillWorks:
    """The Lane I dispatch must NOT regress the canonical AsymmetricPairGenerator
    FP4A path (the existing 0.90 baseline + every training run before
    Lane I). The dispatch is gated on cfg.variant == "" / "coolchic_renderer"
    / "c3_residual_renderer" — anything else falls through to the legacy
    branch."""

    def test_empty_variant_routes_to_FP4A_legacy(self, tmp_path: Path):
        from experiments.pipeline import step_export
        from tac.renderer import build_renderer

        torch.manual_seed(31)
        model = build_renderer(
            num_classes=5, embed_dim=6,
            base_ch=8, mid_ch=12, motion_hidden=8, depth=1,
            pose_dim=0, use_dsconv=False,
        )
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.05)
        model.eval()

        ckpt = tmp_path / "renderer_legacy_best_fp32.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "__meta__": {
                # No `variant` key → falls to cfg.variant.
                "fp4_codebook": "default",
                "fp4_robust_scale": False,
                "embed_dim": 6, "base_ch": 8, "mid_ch": 12,
                "motion_hidden": 8, "depth": 1, "pose_dim": 0,
                "use_dsconv": False, "padding_mode": "zeros",
                "use_dilation": False, "use_zoom_flow": False,
            },
        }, str(ckpt))

        cfg = _make_cfg(
            tmp_path, ckpt,
            # Legacy archs have padding_mode="zeros" by default; Pipeline
            # default is "replicate" which would force a build_renderer
            # rebuild with the wrong arg. Override here so the rebuilt
            # state_dict matches.
            padding_mode="zeros",
            base_ch=8, mid_ch=12, motion_hidden=8,
        )
        bin_path = step_export(cfg, iteration=0)
        # Legacy FP4A magic — NOT CCh1 or C3R1.
        assert bin_path.read_bytes()[:4] == b"FP4A"
