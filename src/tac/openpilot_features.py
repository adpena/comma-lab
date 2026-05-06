"""Lane DI: compress-time openpilot supercombo feature extraction.

Supercombo is allowed only before archive creation.  This module extracts a
per-pair penultimate feature vector when a session is available and falls back
to zero embeddings only when the caller explicitly requests a smoke fallback.
"""
from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SupercomboFeatureUnavailable", "extract_supercombo_features"]


class SupercomboFeatureUnavailable(RuntimeError):
    """Raised when openpilot features are requested but no model/session exists."""


def _n_pairs(video: torch.Tensor) -> int:
    return int(video.shape[0] // 2)


def _feature_output_name(session: Any) -> str | None:
    try:
        outputs = session.get_outputs()
    except Exception:
        return None
    if not outputs:
        return None
    for out in outputs:
        name = getattr(out, "name", "") or ""
        if "feature" in name.lower():
            return name
    return getattr(outputs[0], "name", None)


def _input_specs(session: Any) -> list[Any]:
    try:
        return list(session.get_inputs())
    except Exception:
        return []


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            dims.append(1)
    return tuple(dims)


def _pair_to_supercombo_input(frame_prev: torch.Tensor, frame_curr: torch.Tensor) -> np.ndarray:
    from tac.openpilot_seeding import _frames_to_supercombo_yuv

    inp = _frames_to_supercombo_yuv(frame_curr.cpu(), frame_prev.cpu())
    return inp.detach().cpu().numpy().astype(np.float32, copy=False)


def _zero_features(
    video: torch.Tensor,
    *,
    feature_dim: int,
    reason: str,
) -> torch.Tensor:
    warnings.warn(
        f"supercombo features unavailable ({reason}); returning zero embedding",
        RuntimeWarning,
        stacklevel=3,
    )
    return torch.zeros(_n_pairs(video), feature_dim, dtype=torch.float32, device=video.device)


def extract_supercombo_features(
    video: torch.Tensor,
    *,
    supercombo_path: str | Path | None = None,
    session: Any | None = None,
    require_cuda: bool = False,
    feature_dim: int = 512,
    allow_zero_fallback: bool = False,
) -> torch.Tensor:
    """Extract one supercombo feature vector per non-overlapping frame pair.

    Args:
        video: ``(N, H, W, 3)`` uint8 RGB frames.
        supercombo_path: optional ONNX path.  If omitted, the openpilot default
            path is tried; absence falls back to zero embeddings.
        session: optional injected ONNX-like session for tests or callers that
            manage model loading themselves.
        require_cuda: reject CPU tensors when CUDA-only production behavior is
            requested.
        feature_dim: fallback feature width when no session is available.
        allow_zero_fallback: if true, return warning-tagged zero features when
            openpilot runtime/model loading fails. Defaults to fail-closed.
    """
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"video must have shape (N, H, W, 3); got {tuple(video.shape)}")
    if require_cuda and video.device.type != "cuda":
        raise RuntimeError("CUDA is required for supercombo feature extraction")

    pairs = _n_pairs(video)
    if pairs == 0:
        return torch.zeros(0, feature_dim, dtype=torch.float32, device=video.device)

    if session is None:
        try:
            from tac.openpilot_seeding import (
                OPENPILOT_SUPERCOMBO_DEFAULT_PATH,
                SupercomboUnavailable,
                load_supercombo_model,
            )

            path = Path(supercombo_path or OPENPILOT_SUPERCOMBO_DEFAULT_PATH)
            session = load_supercombo_model(path, video.device)
        except Exception as exc:
            unavailable = getattr(sys.modules.get("tac.openpilot_seeding"), "SupercomboUnavailable", None)
            if unavailable is not None and isinstance(exc, unavailable):
                if allow_zero_fallback:
                    return _zero_features(video, feature_dim=feature_dim, reason=str(exc))
                raise SupercomboFeatureUnavailable(str(exc)) from exc
            if allow_zero_fallback:
                return _zero_features(video, feature_dim=feature_dim, reason=str(exc))
            raise SupercomboFeatureUnavailable(str(exc)) from exc

    specs = _input_specs(session)
    input_name = getattr(specs[0], "name", "input_imgs") if specs else "input_imgs"
    output_name = _feature_output_name(session)
    output_names = [output_name] if output_name else None

    features: list[torch.Tensor] = []
    for i in range(pairs):
        frame_prev = video[2 * i]
        frame_curr = video[2 * i + 1]
        feed: dict[str, np.ndarray] = {}
        main = _pair_to_supercombo_input(frame_prev, frame_curr)
        for spec in specs:
            name = getattr(spec, "name", "")
            if name == input_name:
                feed[name] = main
            else:
                shape = _shape_tuple(getattr(spec, "shape", (1,)))
                feed[name] = np.zeros(shape, dtype=np.float32)
        if not feed:
            feed[input_name] = main

        result = session.run(output_names, feed)
        arr = np.asarray(result[0], dtype=np.float32).reshape(1, -1)
        feat = torch.from_numpy(arr[0]).to(device=video.device, dtype=torch.float32)
        features.append(feat)

    return torch.stack(features, dim=0)


class SceneEmbeddingDistiller(nn.Module):
    """Small MLP that distills supercombo features into a scene embedding."""

    def __init__(
        self,
        *,
        input_dim: int = 512,
        output_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dim": self.hidden_dim,
                "state_dict": self.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "SceneEmbeddingDistiller":
        payload = torch.load(Path(path), map_location=map_location)
        model = cls(
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 128)),
        )
        model.load_state_dict(payload["state_dict"])
        return model


def train_scene_embedding_distiller(
    features: torch.Tensor,
    poses: torch.Tensor,
    masks: torch.Tensor,
    *,
    output_dim: int = 32,
    steps: int = 50,
    lr: float = 1e-3,
) -> SceneEmbeddingDistiller:
    """Train a lightweight distiller against pose and mask summary targets."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D; got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("training requires at least 2 feature rows")
    if poses.shape[0] != features.shape[0] or masks.shape[0] != features.shape[0]:
        raise ValueError("features, poses, and masks must share the leading dimension")

    model = SceneEmbeddingDistiller(input_dim=features.shape[1], output_dim=output_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    target_parts = [poses.float().reshape(features.shape[0], -1)]
    mask_summary = masks.float().reshape(features.shape[0], -1).mean(dim=1, keepdim=True)
    target_parts.append(mask_summary)
    target = torch.cat(target_parts, dim=1)
    if target.shape[1] < output_dim:
        target = F.pad(target, (0, output_dim - target.shape[1]))
    else:
        target = target[:, :output_dim]

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(features), target)
        loss.backward()
        opt.step()
    return model


def _install_scene_embedding_distiller_module() -> None:
    name = "tac.scene_embedding_distiller"
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__file__ = __file__
        sys.modules[name] = module
    module.SceneEmbeddingDistiller = SceneEmbeddingDistiller
    module.train_scene_embedding_distiller = train_scene_embedding_distiller

    tac_pkg = sys.modules.get("tac")
    if tac_pkg is not None:
        setattr(tac_pkg, "scene_embedding_distiller", module)


_install_scene_embedding_distiller_module()
