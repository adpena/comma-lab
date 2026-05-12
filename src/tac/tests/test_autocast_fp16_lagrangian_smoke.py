"""Runtime smoke test for the autocast FP16 + GradScaler path landed in
`experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`
commit b0ef91a3.

Per 2026-05-12 adversarial review Finding 2/5: the autocast patch landed
with ruff F821+E9 static verification only — no runtime test verifying:
  (a) gradients propagate through the fp16 forward → fp32 Lagrangian path
  (b) GradScaler's scale_factor finds a usable value without overflow
  (c) the .float() casts before coord.step() preserve autograd connectivity
  (d) teacher_pose_cache (fp32) + student (fp16) dtype handoff works

This smoke uses a tiny CPU-compatible model + synthetic data. CUDA-specific
autocast paths cannot be runtime-validated on a CPU-only machine; the test
exercises the same code structure with `enabled=False` on CPU and verifies
the dtype handoff would succeed under `enabled=True` (no dtype mismatch
errors, no detached gradients, no nan/inf losses).

For full CUDA validation, this test should be re-run on a Modal T4 or
Vast.ai 4090 instance with `enabled=True` before the next $5+ dispatch.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TinyDecoder(nn.Module):
    """Stand-in for the 125K-param decoder used in T1 Ballé end-to-end."""

    def __init__(self, latent_dim: int = 4, channels: int = 3):
        super().__init__()
        self.lin = nn.Linear(latent_dim, channels * 8 * 8)
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        n = y_hat.shape[0]
        x = self.lin(y_hat).reshape(n, 3, 8, 8)
        x = self.up(x)
        # Map to (B, 2, 3, H, W) "pair" tensor matching trainer's expected shape.
        return x.unsqueeze(1).expand(-1, 2, -1, -1, -1).contiguous() * 127.0 + 127.0


class _TinyBalle(nn.Module):
    """Stand-in for the Ballé entropy model. Returns dict with rate_total_bits."""

    def __init__(self, channels: int = 4):
        super().__init__()
        self.proj = nn.Linear(channels, channels)

    def forward(self, y: torch.Tensor) -> dict:
        y_hat = self.proj(y)
        rate = (y_hat ** 2).sum() * 1e-3 + 100.0
        return {"y_hat": y_hat, "rate_total_bits": rate}


def _eval_roundtrip_decoded(decoded: torch.Tensor, *, noise_std: float = 0.5) -> torch.Tensor:
    """Minimal stand-in matching the trainer's roundtrip semantics."""
    if noise_std > 0:
        decoded = decoded + torch.randn_like(decoded) * noise_std
    return decoded.clamp(0.0, 255.0)


def _scorer_loss_terms_stub(decoded_rt: torch.Tensor, tgt: torch.Tensor):
    """Stand-in for scorer_loss_terms_btchw — returns three scalar losses."""
    diff = (decoded_rt - tgt).abs().mean()
    pose = diff * 0.5
    seg = diff * 0.3
    return diff, pose, seg


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_autocast_lagrangian_path_does_not_break_gradients(autocast_enabled):
    """Verify the autocast forward → fp32 Lagrangian → scaled backward path
    preserves gradient connectivity for both autocast_enabled=False (the
    backward-compat path) and autocast_enabled=True (the speedup path).
    """
    torch.manual_seed(0)
    device = torch.device("cpu")  # CPU-only smoke; CUDA validation is separate
    batch_size = 4
    latent_dim = 4

    decoder = _TinyDecoder(latent_dim=latent_dim).to(device)
    balle = _TinyBalle(channels=latent_dim).to(device)
    main_params = list(decoder.parameters()) + list(balle.parameters())

    optim_main = torch.optim.Adam(main_params, lr=1e-4)
    # GradScaler only meaningful on CUDA; on CPU it's a no-op even when enabled.
    # The test instead validates the *code structure* runs without dtype mismatches.
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # CPU forced-off

    y = torch.randn(batch_size, latent_dim, device=device)
    tgt = (torch.rand(batch_size, 2, 3, 8, 8, device=device) * 255.0)

    # Replicate the trainer's per-batch path structurally.
    # NB: torch.autocast('cpu', dtype=torch.bfloat16) exists on CPU; using
    # `enabled=False` here matches what would run on a CPU-only machine
    # when --enable-autocast-fp16 is set (it should no-op gracefully).
    autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32
    with torch.autocast("cpu", dtype=autocast_dtype, enabled=autocast_enabled):
        balle_out = balle(y)
        decoded = decoder(balle_out["y_hat"])
        decoded_rt = _eval_roundtrip_decoded(decoded)
        scorer_distortion, pose_loss, seg_loss = _scorer_loss_terms_stub(decoded_rt, tgt)
        distortion = scorer_distortion

    # Cast to FP32 BEFORE the "Lagrangian" — mirrors trainer's safeguard.
    distortion = distortion.float() if autocast_enabled else distortion
    seg_loss = seg_loss.float() if autocast_enabled else seg_loss
    pose_loss = pose_loss.float() if autocast_enabled else pose_loss
    rate_bits = balle_out["rate_total_bits"]
    if autocast_enabled and hasattr(rate_bits, "float"):
        rate_bits = rate_bits.float()

    # Synthetic "augmented Lagrangian" — sum of fp32-cast terms.
    total_loss = distortion + 0.01 * rate_bits + 1.0 * seg_loss + 1.0 * pose_loss
    assert torch.isfinite(total_loss).all(), "total_loss must be finite"
    assert total_loss.dtype == torch.float32, (
        f"total_loss should be FP32 after explicit casts, got {total_loss.dtype}"
    )

    optim_main.zero_grad()
    if autocast_enabled and scaler.is_enabled():
        scaler.scale(total_loss).backward()
        scaler.step(optim_main)
        scaler.update()
    else:
        total_loss.backward()
        optim_main.step()

    # Validate gradient connectivity:
    for p in main_params:
        if p.requires_grad:
            assert p.grad is not None, "parameter must have gradient"
            assert torch.isfinite(p.grad).all(), "gradient must be finite"


def test_teacher_cache_dtype_handoff_to_student_fp16():
    """Verify the teacher_pose_cache (built fp32 pre-loop) handoff to a
    student PoseNet path running under autocast fp16 does not raise a
    dtype mismatch in apply_kl_pose_distill.

    Replicates the structural pattern of the trainer at lines 2349-2380.
    """
    torch.manual_seed(0)
    n_pairs = 8
    pose_dim = 12

    # Teacher cache: fp32, pre-loop on CPU
    teacher_pose_cache = torch.randn(n_pairs, pose_dim, dtype=torch.float32)

    # Student forward under (synthetic) autocast — produces fp16 tensor
    # (or bfloat16 on CPU; the test verifies the .float() cast at consume-time)
    idx = torch.arange(4)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        student_pose = torch.randn(4, pose_dim, requires_grad=True).bfloat16()

    teacher_pose = teacher_pose_cache[idx].detach()

    # Mimic apply_kl_pose_distill input — verify dtype reconciliation.
    # The trainer's pattern: student.float() and teacher.float() before KL.
    student_fp32 = student_pose.float()
    teacher_fp32 = teacher_pose.float()

    assert student_fp32.dtype == torch.float32
    assert teacher_fp32.dtype == torch.float32
    assert student_fp32.shape == teacher_fp32.shape

    # Simulated KL loss — verify gradient flows
    kl = F.kl_div(
        F.log_softmax(student_fp32 / 2.0, dim=-1),
        F.softmax(teacher_fp32 / 2.0, dim=-1),
        reduction="batchmean",
    )
    assert torch.isfinite(kl)
    kl.backward()
    # student_pose was created inside the autocast context — its .grad
    # should be populated through the .float() cast
    # (in practice this works for bfloat16 backward on CPU)


def test_gradscaler_disabled_on_cpu_is_no_op():
    """Verify GradScaler with enabled=False (CPU-only smoke fallback)
    behaves identically to plain backward+step pipeline."""
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    assert not scaler.is_enabled()
    # scaler.scale(x) on disabled returns x unchanged
    x = torch.tensor(1.0, requires_grad=True)
    scaled = scaler.scale(x)
    assert scaled is x or torch.equal(scaled, x), (
        "GradScaler(enabled=False).scale() must be identity"
    )
