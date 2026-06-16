# SPDX-License-Identifier: MIT
"""KD-warm-start actuator — distill a CONVERGED vendored-taper basin (teacher) into a
re-tapered (solved-taper, possibly +FiLM) decoder (student) so the student inherits the
basin's knowledge in a SHORT warm-up phase, instead of re-running the months-long PR95
29k-epoch from-scratch curriculum on MPS.

WHY a separate primitive (the wall-clock resolution).  ``warm_start_dir`` warm-starts a
decoder by STRICT-loading the saved ``best_ema_decoder.pt`` into a freshly built decoder.
That works when the architecture matches (same taper), but a RE-TAPER changes the decoder
channel shapes, so the strict decoder load FAILS — the converged basin cannot be carried
into the solved-taper student that way.  The LATENTS, however, are ``(n_pairs, 28)`` —
INDEPENDENT of the taper (only decoder channels change) — so they CAN be carried directly
(the existing ``_load_warm_start_into`` latent path already does this).  The decoder
knowledge is transferred by DISTILLATION: a frozen teacher (the basin, built at the basin's
vendored taper) renders the target frames, and the student (the solved taper) is trained to
match those frames on the SAME latents.  After the warm-up the normal score-aware
curriculum (oomph + FiLM + QAT + Muon + …) continues from the distilled student.

The distillation objective is a clean FRAME-MSE (per-pixel L2 between the student pair
frames and the teacher pair frames, both ``(B, 2, 3, H, W)`` in [0, 255]).  We searched the
repo for an existing decoder→decoder frame-KD to reuse:
``tac.archive.scorer_distill`` distills a renderer's features into tiny SCORER heads
(renderer→scorer-proxy), and ``experiments/train_distill.py`` trains a renderer from a
teacher's RENDERED-FRAME targets but is a monolithic CLI bound to the renderer/SegMap
stack — neither is a decoder-architecture frame-KD for an HNeRV warm-start.  So this is a
focused, documented frame-MSE KD (the simplest objective that makes the re-tapered student
reproduce the basin's rendered pairs); the score-aware curriculum that follows supplies the
SegNet/PoseNet awareness.

NO-FAKE: the teacher is FROZEN (``requires_grad_(False)`` + ``eval()`` + rendered under
``torch.no_grad()``); the KD step ACTUALLY lowers the student-vs-teacher frame distortion
(proved by the ``kd_warm_start`` tests, which run a few KD steps and assert the frame-MSE
measurably DROPS toward the teacher — not a constant, the real distillation behavior).

Authority: torch-CPU TRUSTED (CLAUDE.md "local CPU + MLX GPU good"); NO MPS for the exact
metric.  The KD warm-up is a TRAIN-TIME priming pass — it makes NO score claim; the exact
d_seg/d_pose that pick BEST still run the full scorer on the CPU authority on the
byte-closed archive.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_EVAL_H, _EVAL_W = 384, 512


def load_kd_warm_start_latents(kd_dir: Path | str, *, n_pairs: int, latent_dim: int) -> torch.Tensor:
    """Load the converged basin's ``best_ema_latents.pt`` as a bare tensor (the stage-0
    latent init for the re-tapered student).  The latents are taper-INDEPENDENT
    ``(n_pairs, latent_dim)`` so they carry directly into ANY taper.

    Tolerates the bare-tensor save (the driver's canonical layout) AND a dict-wrapped
    variant defensively.  Raises on a missing file or a shape mismatch (a misconfigured
    warm-start, never a silent wrong-shape load)."""
    kd_dir = Path(kd_dir)
    lat_path = kd_dir / "best_ema_latents.pt"
    if not lat_path.exists():
        raise FileNotFoundError(
            f"kd_warm_start_dir={kd_dir} must hold best_ema_latents.pt (the converged-basin "
            f"latents, taper-independent (n_pairs, latent_dim)); missing best_ema_latents.pt"
        )
    lat = torch.load(lat_path, map_location="cpu", weights_only=False)
    if isinstance(lat, dict):
        if not lat:
            raise ValueError(f"kd_warm_start latents dict at {lat_path} is empty")
        lat = lat.get("latents", next(iter(lat.values())))
    lat = lat.detach()
    if int(lat.shape[0]) != int(n_pairs) or int(lat.shape[1]) != int(latent_dim):
        raise ValueError(
            f"kd_warm_start latents shape {tuple(lat.shape)} != (n_pairs={n_pairs}, "
            f"latent_dim={latent_dim}); cannot warm-start a different basis"
        )
    return lat.clone()


def build_frozen_teacher(
    kd_dir: Path | str,
    *,
    vendored_decoder_cls: type,
    latent_dim: int,
    base_channels: int,
    device: torch.device | str = "cpu",
    eval_size: tuple[int, int] = (_EVAL_H, _EVAL_W),
) -> nn.Module:
    """Build the FROZEN teacher decoder from the converged basin's ``best_ema_decoder.pt``.

    The teacher is the basin's OWN architecture — the VENDORED taper (the basin was trained
    with NO ``taper_channels``), so it is the plain ``vendored_decoder_cls`` at the basin's
    ``base_channels``/``latent_dim``.  The saved EMA shadow state_dict is STRICT-loaded into
    it (a shape/key mismatch = a misconfigured KD source, never a silent partial load).  The
    teacher is set to ``eval()`` and ALL its params get ``requires_grad_(False)`` so the KD
    step can never train it (the NO-FAKE frozen-teacher contract).

    Returns the frozen teacher decoder on ``device``."""
    kd_dir = Path(kd_dir)
    dec_path = kd_dir / "best_ema_decoder.pt"
    if not dec_path.exists():
        raise FileNotFoundError(
            f"kd_warm_start_dir={kd_dir} must hold best_ema_decoder.pt (the converged-basin "
            f"EMA decoder shadow); missing best_ema_decoder.pt"
        )
    sd = torch.load(dec_path, map_location="cpu", weights_only=False)
    # A bare HNeRV state_dict has param-name keys (e.g. "rgb_1.weight") — never a literal
    # "state_dict" key — so this unwrap only fires on a dict-wrapped save.
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    teacher = vendored_decoder_cls(
        latent_dim=int(latent_dim),
        base_channels=int(base_channels),
        eval_size=tuple(eval_size),
    ).to(device)
    # STRICT load — the teacher MUST be the exact converged basin.
    teacher.load_state_dict({k: v.to(device) for k, v in sd.items()})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _render_pairs(decoder: nn.Module, latents: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Render a batch of pairs from a (bare, non-FiLM) decoder → ``(B, 2, 3, H, W)`` in
    [0, 255].  Used for BOTH the teacher (under no_grad by the caller) and the student
    (with grad).  The bare vendored/taper decoder forward takes only the latents."""
    return decoder(latents[idx])


def kd_frame_mse(student_frames: torch.Tensor, teacher_frames: torch.Tensor) -> torch.Tensor:
    """The KD distillation objective: per-pixel L2 between the student and teacher pair
    frames (both ``(B, 2, 3, H, W)`` in [0, 255]).  A plain ``F.mse_loss`` mean — the
    simplest objective that makes the student reproduce the teacher's rendered pairs.  The
    teacher tensor MUST be detached (the caller renders it under ``no_grad``)."""
    return F.mse_loss(student_frames, teacher_frames.detach())


def kd_warm_up_decoder(
    *,
    student: nn.Module,
    teacher: nn.Module,
    latents: nn.Parameter,
    n_pairs: int,
    epochs: int,
    batch_size: int,
    lr: float,
    train_latents: bool = True,
    latent_lr_mult: float = 10.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
    pose_film_enabled: bool = False,
) -> dict[str, float]:
    """Run the KD WARM-UP: distill the FROZEN ``teacher`` into the (trainable) ``student``
    on the SAME ``latents`` via the frame-MSE objective, for ``epochs`` epochs.

    The teacher renders the target pairs (under ``no_grad``); the student is trained to
    match.  Optionally the latents are co-trained (``train_latents``) — they are shared by
    teacher and student, so co-training them lets the student adapt the code to its own
    taper while still chasing the teacher frames (the teacher's frames are recomputed each
    step from the CURRENT latents, so the target tracks the latents — a consistent joint
    objective).  The student decoder is the re-tapered (solved-taper) decoder, possibly a
    FiLM wrapper; when ``pose_film_enabled`` the student forward needs the per-pair index
    (the FiLM looks up the stored pose), and the teacher (no FiLM) is rendered WITHOUT idx.

    Returns ``{"first_loss", "last_loss", "epochs"}`` for telemetry (the frame-MSE at the
    first and last epoch — ``last < first`` is the proof the distillation actually ran).
    This is a TRAIN-TIME priming pass; it makes NO score claim."""
    device = torch.device(device) if isinstance(device, str) else device
    student.train()
    teacher.eval()

    params: list[dict] = [{"params": [p for p in student.parameters() if p.requires_grad], "lr": lr}]
    if train_latents and latents.requires_grad:
        params.append({"params": [latents], "lr": lr * latent_lr_mult})
    opt = torch.optim.AdamW(params, weight_decay=0.0)

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    first_loss = float("nan")
    last_loss = float("nan")
    for ep in range(int(epochs)):
        perm = torch.randperm(int(n_pairs), generator=g).to(device)
        ep_loss = 0.0
        nb = 0
        for bs in range(0, int(n_pairs), int(batch_size)):
            idx = perm[bs : bs + int(batch_size)]
            with torch.no_grad():
                teacher_frames = _render_pairs(teacher, latents, idx)
            # FiLM-wrapped student forward needs the per-pair idx (looks up the stored
            # pose); the teacher (no FiLM) is rendered WITHOUT idx.
            student_frames = (
                student(latents[idx], idx)
                if pose_film_enabled
                else _render_pairs(student, latents, idx)
            )
            loss = kd_frame_mse(student_frames, teacher_frames)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        mean = ep_loss / max(nb, 1)
        if ep == 0:
            first_loss = mean
        last_loss = mean
    return {"first_loss": first_loss, "last_loss": last_loss, "epochs": float(epochs)}


__all__ = [
    "build_frozen_teacher",
    "kd_frame_mse",
    "kd_warm_up_decoder",
    "load_kd_warm_start_latents",
]
