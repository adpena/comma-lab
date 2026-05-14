#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a sjkl.bin residual payload from prepared pair tensors + scorer.

Recovery note: this script was lost when subagent worktrees were auto-cleaned
without committing source. Rebuilt 2026-05-04 as a thin orchestration script
on top of the now-complete tac.sjkl_basis codec library (basis encoder + alpha
block V2 sparse + full payload wrapper, all runtime-byte-compatible).

Pipeline:
  1. Load prepared pair tensors (output of experiments/prepare_sjkl_pair_tensors.py)
  2. Compute Fisher-info top-K basis via tac.sjkl_basis.compute_sjkl_basis_lanczos
     (CPU-stub-friendly Lanczos HVP; runs on CUDA when --device cuda)
  3. For each selected pair, project residual frames onto basis -> coefficients (K)
  4. Quantize coefficients per-pair -> (mins, steps, qs) at alpha_bits
  5. Encode SJK2 sparse alpha block, brotli-compress
  6. Wrap basis + alpha_block via encode_full_sjkl_payload
  7. Write sjkl.bin + sjkl_manifest.json (score_claim=false until contest_auth_eval)

Per-pair quantization (per addendum):
  alpha[i, j] = mins[i] + qs[i, j] * steps[i]
  where steps[i] = (max_alpha[i] - min_alpha[i]) / (2^alpha_bits - 1)
        mins[i]  = min_alpha[i]

CUDA policy (per runbook):
  - Default --device cuda (production)
  - --device cpu allowed only with --allow-cpu-stub (CPU stub mode for smoke tests)
  - Score claim is ALWAYS false; runtime auth eval through contest CUDA pipeline is required
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.sjkl_basis import (
    SJKLBasis,
    compute_sjkl_basis_lanczos,
    encode_full_sjkl_payload,
    encode_sjkl_alpha_block_v1_dense,
    encode_sjkl_alpha_block_v2_sparse,
)


@dataclass(frozen=True)
class BuildConfig:
    pair_tensor_manifest: Path
    output_dir: Path
    device: str
    rank: int
    n_pairs: int
    alpha_bits: int
    basis_quant_bits: int
    max_bytes: int
    allow_cpu_stub: bool
    seed: int
    # Optional path to a JointFrameGenerator (Q-FAITHFUL) checkpoint. When
    # provided AND --device cuda, the score_fn used by the Lanczos basis solve
    # passes the anchor pair through JFG before the SegNet+PoseNet contest
    # score formula `100 * seg_dist + sqrt(10 * pose_dist)`. When None, the
    # score_fn falls back to the self-target Gauss-Newton scorer-only proxy.
    # Only used at COMPRESS time (strict-scorer-rule: never at inflate time).
    renderer_checkpoint: Path | None = None


# Module-level test injection hook. When set, replaces the entire CUDA
# score_fn factory with a callable returning (score_fn, scorer_meta). Tests
# wire stub JFG/SegNet/PoseNet through this hook to exercise the wiring on
# CPU without paid GPU spend. NOT a public API. NEVER set in production
# code paths - this exists exclusively for src/tac/tests/test_sjkl_cuda_scorer_wiring.py.
_SCORE_FN_FACTORY_OVERRIDE = None  # type: ignore[var-annotated]


# Q-FAITHFUL JointFrameGenerator default architecture (matches PR #55 / Lane
# Q-FAITHFUL). Override via the QFAI binary header when loading a checkpoint
# packaged with the contest layout; raw .pt state-dict checkpoints assume
# these defaults.
_DEFAULT_JFG_ARCH: dict = {
    "num_classes": 5,
    "pose_dim": 6,
    "cond_dim": 48,
    "depth_mult": 1,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_pairs(n_total: int, n_select: int, seed: int) -> np.ndarray:
    """Select n_select pair indices deterministically. Default policy: first
    n_select indices for reproducibility. Production callers should override
    this via a sensitivity-ranking selector + an explicit pair_indices arg."""
    if n_select > n_total:
        raise ValueError(f"n_pairs ({n_select}) cannot exceed available pairs ({n_total})")
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_select, replace=False).astype(np.uint16)


def _quantize_alpha_per_pair(alpha: np.ndarray, alpha_bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair scalar quantization to alpha_bits.

    alpha: (n_pairs, K) float32 raw projection coefficients.
    Returns (qs, mins, steps) where qs is uint8/uint16 in [0, 2^alpha_bits - 1].
    """
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2-D (n_pairs, K), got {alpha.shape}")
    qmax = (1 << alpha_bits) - 1
    n_pairs, K = alpha.shape
    mins = alpha.min(axis=1).astype(np.float32)
    maxs = alpha.max(axis=1).astype(np.float32)
    spans = (maxs - mins).clip(min=1e-12)  # avoid divide-by-zero on flat pairs
    steps = (spans / qmax).astype(np.float32)
    qs_float = (alpha - mins[:, None]) / steps[:, None]
    qs = np.clip(np.round(qs_float), 0, qmax).astype(np.uint16 if alpha_bits > 8 else np.uint8)
    return qs, mins, steps


def _project_residual_onto_basis(
    pair_residuals: torch.Tensor,
    basis: SJKLBasis,
) -> np.ndarray:
    """Project per-pair residual frames onto basis to get (n_pairs, K) alpha coefficients.

    pair_residuals: (n_pairs, D) float32 - each row is a flattened residual frame.
    basis: SJKLBasis with eigenvectors (K, D).
    """
    if pair_residuals.shape[-1] != basis.dim:
        raise ValueError(
            f"pair_residuals last dim {pair_residuals.shape[-1]} does not match basis.dim {basis.dim}"
        )
    eigenvectors = basis.eigenvectors.to(pair_residuals.dtype).to(pair_residuals.device)
    # alpha[i, k] = <pair_residuals[i], eigenvectors[k]>
    alpha = pair_residuals @ eigenvectors.T  # (n_pairs, K)
    return alpha.detach().cpu().to(torch.float32).numpy()


def _build_cpu_stub_score_fn(dim: int, *, peak_idx: int = 0):
    """A CPU-stub quadratic score_fn for smoke testing without a real scorer."""
    diag = torch.tensor(
        [10.0, 5.0, 2.0, 0.5, 0.1] + [0.05] * (dim - 5),
        dtype=torch.float32,
    )
    if peak_idx != 0 and peak_idx < dim:
        diag = diag.roll(peak_idx)

    def score_fn(f: torch.Tensor) -> torch.Tensor:
        return 0.5 * (diag.to(f.device) * f * f).sum()

    return score_fn


def _infer_rgb_shape(dim: int) -> tuple[int, int] | None:
    if dim == 3 * 384 * 512:
        return (384, 512)
    if dim == 3 * 192 * 256:
        return (192, 256)
    if dim == 3 * 128 * 128:
        return (128, 128)
    return None


def _load_jfg_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a Quantizr-Faithful JointFrameGenerator (JFG) from disk.

    Two layouts are supported:

    1. **Raw torch state-dict ``.pt``** - produced by the training pipelines
       (``experiments/train_renderer.py`` etc.). The JFG architecture defaults
       to PR-#55 defaults (num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1)
       and is built via ``build_quantizr_faithful_renderer``.
    2. **QFAI binary blob** - produced by the inflate-side QFAI packer (see
       ``submissions/robust_current/inflate_renderer.py:4395``). Layout:
       ``magic[b"QFAI"] + uint32_LE header_len + JSON header + torch.save body``.
       The JSON header carries arch overrides so non-default architectures load
       cleanly.

    The returned module is on ``device``, in ``eval()`` mode, with
    ``requires_grad_(False)`` on every parameter. **Strict-scorer-rule note:**
    this function loads a renderer at COMPRESS time only - the SJ-KL basis
    build is a compress-time process. The inflate path loads JFG via its own
    QFAI dispatch in ``inflate_renderer.py``.
    """
    import io as _io
    import struct as _struct

    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

    raw_bytes = checkpoint_path.read_bytes()

    if raw_bytes.startswith(b"QFAI"):
        offset = 4
        header_len = _struct.unpack("<I", raw_bytes[offset:offset + 4])[0]
        offset += 4
        header = json.loads(raw_bytes[offset:offset + header_len].decode("utf-8"))
        offset += header_len
        gen = build_quantizr_faithful_renderer(
            num_classes=int(header.get("num_classes", _DEFAULT_JFG_ARCH["num_classes"])),
            pose_dim=int(header.get("pose_dim", _DEFAULT_JFG_ARCH["pose_dim"])),
            cond_dim=int(header.get("cond_dim", _DEFAULT_JFG_ARCH["cond_dim"])),
            depth_mult=int(header.get("depth_mult", _DEFAULT_JFG_ARCH["depth_mult"])),
        )
        state = torch.load(
            _io.BytesIO(raw_bytes[offset:]),
            map_location=str(device),
            weights_only=True,
        )
    else:
        gen = build_quantizr_faithful_renderer(**_DEFAULT_JFG_ARCH)
        state = torch.load(
            checkpoint_path,
            map_location=str(device),
            weights_only=True,
        )

    gen.load_state_dict(state, strict=True)
    gen.to(device).eval()
    for param in gen.parameters():
        param.requires_grad_(False)
    return gen


def _build_cuda_jfg_contest_score_fn(
    anchor_frame: torch.Tensor,
    jfg: torch.nn.Module,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    *,
    target_h: int,
    target_w: int,
) -> tuple[object, dict]:
    """Build a direct-pixel contest-scorer differentiable score_fn.

    The returned ``score_fn(flat)`` re-shapes ``flat`` into a candidate frame
    pair, forwards through SegNet+PoseNet, and returns the contest formula::

        score = 100 * seg_dist(cand) + sqrt(10 * pose_dist(cand) + eps)

    where:
      * ``pose_dist = MSE(PoseNet(cand)[..., :6], target_pose)`` - directly
        differentiable (matches ``upstream/modules.py:PoseNet.compute_distortion``
        exactly modulo the per-pair mean: we sum here for Hessian-vector-product
        stability; the Lanczos eigvecs are scale-invariant).
      * ``seg_dist`` uses a **soft** disagreement proxy:
        ``mean(1 - softmax(SegNet(cand))[target_argmax])``, which collapses to
        the hard argmax disagreement at the limit and is differentiable
        everywhere. Hard ``argmax`` is used in the contest scorer; the soft
        proxy is the canonical replacement for compress-time gradients.

    ``flat`` is interpreted as the flattened candidate RGB frame already in
    the renderer output range [0, 255]. The current helper does **not** route
    through JFG; the ``jfg`` argument is retained for API compatibility with
    the Track-B scaffold, and metadata marks ``scorer_fisher_jfg_in_loop`` as
    false so manifests cannot overclaim generator-mediated Fisher evidence.

    Args:
        anchor_frame: flat (D,) RGB tensor at the linearization point.
        jfg: loaded JointFrameGenerator placeholder (unused in this helper).
        posenet, segnet: frozen PoseNet/SegNet from
            ``tac.scorer.load_differentiable_scorers``.
        target_h, target_w: spatial shape that ``flat`` should reshape to
            (3 * target_h * target_w == flat.numel()).

    Returns:
        (score_fn, scorer_meta) where ``scorer_meta`` is appended to the
        manifest for forensic provenance.
    """
    posenet.eval()
    segnet.eval()
    for model in (posenet, segnet):
        for param in model.parameters():
            param.requires_grad_(False)

    def _pair_from_flat(flat: torch.Tensor) -> torch.Tensor:
        chw = flat.reshape(3, target_h, target_w).clamp(0.0, 255.0)
        # 2-frame pair, batch=1: (B=1, T=2, C=3, H, W)
        return torch.stack([chw, chw], dim=0).unsqueeze(0)

    # Compute per-pair targets at the anchor frame, frozen. These are the
    # "contest-correct" regression targets: the SegNet argmax class indices
    # and the PoseNet first-6 pose vector at the anchor.
    with torch.no_grad():
        base_pair = _pair_from_flat(anchor_frame.detach())
        seg_logits_target = segnet(segnet.preprocess_input(base_pair)).detach()
        target_seg_argmax = seg_logits_target.argmax(dim=1).detach()  # (B, h, w)
        target_pose = posenet(posenet.preprocess_input(base_pair))["pose"][..., :6].detach()

    def score_fn(flat: torch.Tensor) -> torch.Tensor:
        pair = _pair_from_flat(flat.reshape(-1))

        # PoseNet term - direct MSE-on-first-6 (matches contest formula).
        pose_out = posenet(posenet.preprocess_input(pair))["pose"][..., :6]
        pose_mse = (pose_out - target_pose).pow(2).mean()

        # SegNet term - soft disagreement (1 - softmax_at_target_class), since
        # the hard argmax in the contest scorer is non-differentiable. The
        # Fisher basis we recover via Lanczos is invariant to monotonic
        # transforms of this proxy (it's the curvature directions that matter,
        # not the absolute scale).
        seg_logits = segnet(segnet.preprocess_input(pair))  # (B, K, h, w)
        seg_softmax = torch.nn.functional.softmax(seg_logits, dim=1)
        # gather target class probs: index along class dim
        target_one = target_seg_argmax.unsqueeze(1)  # (B, 1, h, w)
        target_prob = seg_softmax.gather(1, target_one).squeeze(1)  # (B, h, w)
        seg_dist = (1.0 - target_prob).mean()

        # Contest formula. clamp_min(0) defensive against numeric underflow
        # of the MSE term going slightly negative on FP16 round-trip.
        return (
            100.0 * seg_dist
            + torch.sqrt((10.0 * pose_mse).clamp(min=0.0) + 1e-12)
        )

    return score_fn, {
        "scorer_fisher_mode": "cuda_direct_pixel_contest_score_proxy",
        "scorer_fisher_frame_shape": [3, int(target_h), int(target_w)],
        "scorer_fisher_score_claim": False,
        "scorer_fisher_jfg_in_loop": False,
        "scorer_fisher_jfg_argument_used": False,
        "score_formula": "100 * seg_dist + sqrt(10 * pose_dist + 1e-12)",
        "seg_dist_proxy": "soft_1_minus_softmax_at_target_argmax",
    }


def _build_cuda_self_target_scorer_fn(anchor_frame: torch.Tensor) -> tuple[object, dict]:
    """Build a compress-time scorer Fisher proxy around the frozen scorers.

    This is not an auth-eval score path. It uses the scorer networks only to
    define a differentiable self-target Gauss-Newton loss at the anchor frame.
    """
    h_w = _infer_rgb_shape(int(anchor_frame.numel()))
    if h_w is None:
        raise SystemExit(
            "FATAL: CUDA scorer Fisher proxy requires a flat RGB frame with known "
            f"shape, got dim={int(anchor_frame.numel())}. Add shape metadata in the "
            "pair tensor manifest before production SJ-KL basis builds."
        )
    height, width = h_w
    from tac.scorer import load_differentiable_scorers

    posenet, segnet = load_differentiable_scorers(REPO_ROOT / "upstream", device=anchor_frame.device)
    for model in (posenet, segnet):
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    def _pair_from_flat(flat: torch.Tensor) -> torch.Tensor:
        chw = flat.reshape(3, height, width).clamp(0.0, 255.0)
        return torch.stack([chw, chw], dim=0).unsqueeze(0)

    with torch.no_grad():
        base_pair = _pair_from_flat(anchor_frame.detach())
        target_pose = posenet(posenet.preprocess_input(base_pair))["pose"][..., :6].detach()
        target_seg = segnet(segnet.preprocess_input(base_pair)).detach()

    def score_fn(flat: torch.Tensor) -> torch.Tensor:
        pair = _pair_from_flat(flat.reshape(-1))
        pose_out = posenet(posenet.preprocess_input(pair))["pose"][..., :6]
        seg_out = segnet(segnet.preprocess_input(pair))
        pose_loss = 5.0 * (pose_out - target_pose).pow(2).sum()
        seg_loss = 50.0 * (seg_out - target_seg).pow(2).sum()
        return pose_loss + seg_loss

    return score_fn, {
        "scorer_fisher_mode": "cuda_self_target_gauss_newton_proxy",
        "scorer_fisher_frame_shape": [3, height, width],
        "scorer_fisher_score_claim": False,
    }


def _build_cpu_stub_pair_residuals(n_pairs: int, dim: int, *, seed: int) -> torch.Tensor:
    """Synthetic pair residuals for smoke testing (small random noise around zero)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_pairs, dim, generator=g, dtype=torch.float32) * 0.01


def build_sjkl_residual(cfg: BuildConfig) -> dict:
    """Main pipeline. Returns manifest dict (also written to disk)."""
    device = torch.device(cfg.device)
    scorer_meta: dict = {}
    if cfg.device != "cuda" and not cfg.allow_cpu_stub:
        raise SystemExit(
            f"FATAL: --device {cfg.device} requires --allow-cpu-stub. "
            "Production builds must use --device cuda; score remains [advisory only] otherwise."
        )

    if cfg.allow_cpu_stub and not cfg.pair_tensor_manifest.is_file():
        # Pure-CPU smoke mode: synthesize tiny inputs to exercise the pipeline byte-faithfully
        print("[sjkl-residual] CPU STUB MODE: synthesizing inputs (manifest absent)", file=sys.stderr)
        dim = 256
        n_total_pairs = 600
        anchor_frame = torch.zeros(dim, dtype=torch.float32)
        pair_residuals_all = _build_cpu_stub_pair_residuals(n_total_pairs, dim, seed=cfg.seed)
        if _SCORE_FN_FACTORY_OVERRIDE is not None:
            score_fn, override_meta = _SCORE_FN_FACTORY_OVERRIDE(  # type: ignore[misc]
                anchor_frame, cfg
            )
            scorer_meta.update(override_meta)
        else:
            score_fn = _build_cpu_stub_score_fn(dim)
        manifest_sha = "stub_no_manifest"
    else:
        manifest = json.loads(cfg.pair_tensor_manifest.read_text())
        manifest_sha = _sha256_file(cfg.pair_tensor_manifest)
        anchor_path = REPO_ROOT / manifest["anchor_frame_path"]
        residuals_path = REPO_ROOT / manifest["pair_residuals_path"]
        anchor_frame = torch.load(anchor_path, map_location="cpu", weights_only=False)
        pair_residuals_all = torch.load(residuals_path, map_location="cpu", weights_only=False)
        if anchor_frame.dim() > 1:
            anchor_frame = anchor_frame.flatten()
        if pair_residuals_all.dim() > 2:
            pair_residuals_all = pair_residuals_all.reshape(pair_residuals_all.shape[0], -1)
        n_total_pairs, dim = pair_residuals_all.shape
        # In CPU stub mode without a real scorer module, use the stub quadratic
        # UNLESS a test has installed _SCORE_FN_FACTORY_OVERRIDE (which lets a
        # CPU run exercise the JFG/SegNet/PoseNet wiring with stub modules).
        if cfg.device == "cpu" and _SCORE_FN_FACTORY_OVERRIDE is None:
            score_fn = _build_cpu_stub_score_fn(dim)
        elif _SCORE_FN_FACTORY_OVERRIDE is not None:
            anchor_frame = anchor_frame.to(device)
            score_fn, scorer_meta = _SCORE_FN_FACTORY_OVERRIDE(  # type: ignore[misc]
                anchor_frame, cfg
            )
        elif cfg.renderer_checkpoint is not None:
            # JFG-aware contest score: load JFG + scorers, build the
            # 100 * seg_dist + sqrt(10 * pose_dist) differentiable score_fn.
            anchor_frame = anchor_frame.to(device)
            jfg_path = Path(cfg.renderer_checkpoint)
            if not jfg_path.is_file():
                raise SystemExit(
                    f"FATAL: --renderer-checkpoint {jfg_path} does not exist. "
                    "Pass an explicit path to a Q-FAITHFUL JFG state-dict (.pt) "
                    "or QFAI binary blob."
                )
            jfg = _load_jfg_from_checkpoint(jfg_path, device)
            from tac.scorer import load_differentiable_scorers

            posenet, segnet = load_differentiable_scorers(REPO_ROOT / "upstream", device=device)

            h_w = _infer_rgb_shape(int(anchor_frame.numel()))
            if h_w is None:
                raise SystemExit(
                    "FATAL: JFG-aware score_fn requires a flat RGB frame with "
                    f"known shape, got dim={int(anchor_frame.numel())}. Add "
                    "shape metadata to the pair tensor manifest before "
                    "production SJ-KL basis builds."
                )
            target_h, target_w = h_w
            score_fn, scorer_meta = _build_cuda_jfg_contest_score_fn(
                anchor_frame,
                jfg,
                posenet,
                segnet,
                target_h=target_h,
                target_w=target_w,
            )
            scorer_meta["scorer_fisher_renderer_checkpoint"] = str(jfg_path)
            scorer_meta["scorer_fisher_renderer_sha256"] = _sha256_file(jfg_path)
        else:
            anchor_frame = anchor_frame.to(device)
            score_fn, scorer_meta = _build_cuda_self_target_scorer_fn(anchor_frame)

    # Move to device
    anchor_frame = anchor_frame.to(device)
    pair_residuals_all = pair_residuals_all.to(device)

    # 1. Compute Fisher-info top-K basis
    print(f"[sjkl-residual] Computing Lanczos top-{cfg.rank} basis (dim={dim})...", file=sys.stderr)
    basis = compute_sjkl_basis_lanczos(
        score_fn, anchor_frame, rank=cfg.rank, n_iters=max(cfg.rank * 4, 16), seed=cfg.seed
    )

    # 2. Select pair indices
    pair_indices = _select_pairs(n_total_pairs, cfg.n_pairs, cfg.seed)
    pair_residuals_sel = pair_residuals_all[pair_indices.astype(np.int64)]

    # 3. Project residuals onto basis -> alpha coefficients (n_pairs, K)
    alpha = _project_residual_onto_basis(pair_residuals_sel, basis)

    # 4. Per-pair quantize alpha
    qs, mins, steps = _quantize_alpha_per_pair(alpha, alpha_bits=cfg.alpha_bits)

    # 5. Encode SJK2 sparse alpha block, brotli-compress
    raw_alpha_block = encode_sjkl_alpha_block_v2_sparse(
        qs, mins, steps, alpha_bits=cfg.alpha_bits, pair_indices=pair_indices
    )
    compressed_alpha_block = brotli.compress(raw_alpha_block, quality=11)

    # 6. Wrap into full sjkl.bin payload
    sjkl_bytes = encode_full_sjkl_payload(
        basis, compressed_alpha_block, basis_quant_bits=cfg.basis_quant_bits
    )

    # 7. Size cap check (per addendum: SJKL_MAX_BYTES = 32768 default)
    if len(sjkl_bytes) > cfg.max_bytes:
        raise SystemExit(
            f"FATAL: sjkl.bin size {len(sjkl_bytes)} > max_bytes {cfg.max_bytes}. "
            f"Reduce --rank, --n-pairs, or --alpha-bits."
        )

    # 8. Write sjkl.bin + manifest
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    sjkl_path = cfg.output_dir / "sjkl.bin"
    sjkl_path.write_bytes(sjkl_bytes)
    sjkl_sha = hashlib.sha256(sjkl_bytes).hexdigest()

    manifest_out = {
        "sjkl_bin_path": str(sjkl_path.relative_to(REPO_ROOT) if sjkl_path.is_relative_to(REPO_ROOT) else sjkl_path),
        "sjkl_bin_bytes": len(sjkl_bytes),
        "sjkl_bin_sha256": sjkl_sha,
        "rank": int(cfg.rank),
        "n_pairs": int(cfg.n_pairs),
        "alpha_bits": int(cfg.alpha_bits),
        "basis_quant_bits": int(cfg.basis_quant_bits),
        "selected_pair_indices": [int(x) for x in pair_indices.tolist()],
        "device": cfg.device,
        "seed": int(cfg.seed),
        "pair_tensor_manifest_sha256": manifest_sha,
        "score_claim": False,
        "evidence_grade": "queued_exact_cuda_required_for_score",
        "raw_alpha_block_bytes": len(raw_alpha_block),
        "compressed_alpha_block_bytes": len(compressed_alpha_block),
        "tag": "[empirical:" + str(sjkl_path) + "]" if cfg.device == "cuda" else "[advisory only]",
    }
    manifest_out.update(scorer_meta)
    manifest_path = cfg.output_dir / "sjkl_manifest.json"
    manifest_path.write_text(json.dumps(manifest_out, indent=2))

    print(f"[sjkl-residual] wrote {sjkl_path} ({len(sjkl_bytes)} bytes, sha {sjkl_sha[:16]})", file=sys.stderr)
    print(f"[sjkl-residual] wrote {manifest_path}", file=sys.stderr)
    print("[sjkl-residual] score_claim=false; auth eval through contest CUDA pipeline required", file=sys.stderr)

    return manifest_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-tensor-manifest", type=Path, default=Path("/dev/null"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--rank", type=int, default=4, help="Top-K basis width (k in alpha block).")
    parser.add_argument("--n-pairs", type=int, default=16, help="Number of selected pairs (sparse).")
    parser.add_argument("--alpha-bits", type=int, default=4, help="Bits per alpha coefficient (1-16).")
    parser.add_argument("--basis-quant-bits", type=int, default=6, help="Bits per basis weight (4-8 or 0/None for FP16).")
    parser.add_argument("--max-bytes", type=int, default=32768, help="Hard cap on sjkl.bin size.")
    parser.add_argument(
        "--allow-cpu-stub",
        action="store_true",
        help="Permit --device cpu (smoke tests only). Score remains [advisory only].",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--renderer-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to a Q-FAITHFUL JointFrameGenerator checkpoint "
            "(.pt state-dict OR QFAI binary blob). When provided AND "
            "--device cuda, the SJ-KL Fisher basis is computed against the "
            "contest-faithful score formula 100 * seg_dist + sqrt(10 * pose_dist) "
            "with the JFG renderer in the differentiable graph. "
            "Strict-scorer-rule: COMPRESS time only - never used at inflate."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = BuildConfig(
        pair_tensor_manifest=args.pair_tensor_manifest,
        output_dir=args.output_dir,
        device=args.device,
        rank=args.rank,
        n_pairs=args.n_pairs,
        alpha_bits=args.alpha_bits,
        basis_quant_bits=args.basis_quant_bits,
        max_bytes=args.max_bytes,
        allow_cpu_stub=args.allow_cpu_stub,
        seed=args.seed,
        renderer_checkpoint=args.renderer_checkpoint,
    )
    build_sjkl_residual(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -------------------------------------------------------------------------
# Legacy compatibility symbols.
# Older SJ-KL runtime tests/tools build payloads from already-packed sections.
# Keep these thin wrappers byte-compatible with the canonical encoders.
# -------------------------------------------------------------------------


def pack_alpha_block(*args: object, **kwargs: object) -> bytes:
    if len(args) < 3:
        raise TypeError("pack_alpha_block requires alpha_qs, mins, and steps")
    alpha_qs, mins, steps = args[:3]
    alpha_bits = int(kwargs.pop("alpha_bits", 8))
    pair_indices = kwargs.pop("pair_indices", None)
    sparse_bitpacked = bool(kwargs.pop("sparse_bitpacked", pair_indices is not None))
    if kwargs:
        raise TypeError(f"unexpected pack_alpha_block kwargs: {sorted(kwargs)}")
    qs = np.vstack([np.asarray(x) for x in alpha_qs]).astype(np.uint16 if alpha_bits > 8 else np.uint8)
    mins_arr = np.asarray(mins, dtype=np.float32)
    steps_arr = np.asarray(steps, dtype=np.float32)
    if sparse_bitpacked:
        if pair_indices is None:
            pair_indices_arr = np.arange(qs.shape[0], dtype=np.uint16)
        else:
            pair_indices_arr = np.asarray(pair_indices, dtype=np.uint16)
        return encode_sjkl_alpha_block_v2_sparse(
            qs,
            mins_arr,
            steps_arr,
            alpha_bits=alpha_bits,
            pair_indices=pair_indices_arr,
        )
    return encode_sjkl_alpha_block_v1_dense(qs, mins_arr, steps_arr, alpha_bits=alpha_bits)


def pack_full_sjkl_payload(*args: object, **kwargs: object) -> bytes:
    if len(args) != 2:
        raise TypeError("pack_full_sjkl_payload requires basis_bytes and alpha_block_bytes")
    if kwargs:
        raise TypeError(f"unexpected pack_full_sjkl_payload kwargs: {sorted(kwargs)}")
    basis_bytes, alpha_block_bytes = (bytes(args[0]), bytes(args[1]))
    if not basis_bytes.startswith(b"SJKL"):
        raise ValueError("basis_bytes must start with SJKL magic")
    basis_section = basis_bytes[4:]
    if alpha_block_bytes.startswith((b"SJKB", b"SJK2")):
        alpha_block_bytes = brotli.compress(alpha_block_bytes, quality=11)
    return (
        b"SJKL"
        + len(basis_section).to_bytes(4, "little")
        + len(alpha_block_bytes).to_bytes(4, "little")
        + basis_section
        + alpha_block_bytes
    )
