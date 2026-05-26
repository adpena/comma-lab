# SPDX-License-Identifier: MIT
"""ib_loss_mine — MINE-based IB regularizer + sparse-Laplacian for J=MDL-IBPS.

Phase 1 CC-J-4 unwind: MINE (Belghazi 2018) replaces variational KL upper
bound used in C6 v1 ``mdl_loss.py``. MINE provides tight LOWER bound on
I(z; frames) via Donsker-Varadhan dual representation, avoiding the loose-
bound problem of variational KL upper bound (loose by ~10x for deep nets).

Phase 1 CC-J-3 unwind: beta is operator-tunable per-arm; default sweep
{1e-5, 1e-4, 1e-3, 1e-2} per Higgins-memorial T3 v2 verdict.

Phase 1 CC-J-2 unwind: discrete-categorical posterior (NOT continuous
Gaussian as in C6 v1); the IB-side regularizer operates on the per-pair
one-hot encoded categorical indices.

Phase 1 Path B5 MacKay-influence: sparse-Laplacian regularizer on FiLM
modulation matrices to encourage categorical-index sparsity.

Per CLAUDE.md FORBIDDEN_PATTERNS:
- No scorer load (this module routes the IB-regularizer; sister
  ``score_aware_loss.py`` is the canonical Catalog #164 routing)
- No /tmp paths
- File reviewable in 30 seconds per HNeRV parity L12
"""

from __future__ import annotations

import torch
from torch import nn

from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
    CATEGORICAL_G,
    CATEGORICAL_K,
    DEFAULT_LAMBDA_SPARSE,
    MINE_HIDDEN_DIM,
)


class MINECritic(nn.Module):
    """MINE critic network (Belghazi 2018 DV representation).

    Two-layer MLP critic T_theta(z, frames) that approximates the
    Donsker-Varadhan dual function for I(z; frames).

    Per Belghazi 2018: training alternates between substrate parameters
    and critic parameters; we run the substrate gradient step while
    keeping the critic's parameters fixed within each batch.

    Args:
        z_dim: dimensionality of latent z (per-pair flattened one-hot;
            CATEGORICAL_G * CATEGORICAL_K).
        frames_feat_dim: dimensionality of frames feature vector
            (compact summary of pair frames; e.g. SegNet's pre-classifier
            features compressed via global pool).
        hidden_dim: critic hidden width (default MINE_HIDDEN_DIM=128).
    """

    def __init__(
        self,
        z_dim: int = CATEGORICAL_G * CATEGORICAL_K,
        frames_feat_dim: int = 256,
        hidden_dim: int = MINE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if z_dim <= 0 or frames_feat_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"all dimensions must be positive; got z={z_dim}, "
                f"frames={frames_feat_dim}, hidden={hidden_dim}"
            )
        self.z_dim = z_dim
        self.frames_feat_dim = frames_feat_dim
        self.hidden_dim = hidden_dim
        in_dim = z_dim + frames_feat_dim
        self.layer_1 = nn.Linear(in_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(
        self, z: torch.Tensor, frames_features: torch.Tensor
    ) -> torch.Tensor:
        if z.dim() != 2 or frames_features.dim() != 2:
            raise ValueError(
                f"both inputs must be 2D; got z={z.shape}, "
                f"frames={frames_features.shape}"
            )
        if z.shape[0] != frames_features.shape[0]:
            raise ValueError(
                f"batch size mismatch; z={z.shape[0]}, "
                f"frames={frames_features.shape[0]}"
            )
        h = torch.cat([z, frames_features], dim=-1)
        h = self.activation(self.layer_1(h))
        h = self.activation(self.layer_2(h))
        return self.layer_out(h).squeeze(-1)


def mine_lower_bound(
    critic: MINECritic,
    z_joint: torch.Tensor,
    frames_joint: torch.Tensor,
    z_marginal: torch.Tensor,
) -> torch.Tensor:
    """Compute MINE Donsker-Varadhan lower bound on I(z; frames).

    Formula: I(z; frames) >= E_p(z,f)[T(z,f)] - log E_p(z) p(f) [exp T(z,f)]

    Args:
        critic: MINECritic instance.
        z_joint: ``(B, z_dim)`` joint samples (z drawn from same batch as frames).
        frames_joint: ``(B, frames_feat_dim)`` joint sample frames.
        z_marginal: ``(B, z_dim)`` SHUFFLED z from another batch (marginal sample).

    Returns:
        scalar lower bound on I(z; frames) (in nats; differentiable w.r.t.
        substrate parameters that produced z_joint).
    """
    t_joint = critic(z_joint, frames_joint)
    t_marginal = critic(z_marginal, frames_joint)
    # Numerically-stable log-sum-exp
    max_marginal = t_marginal.detach().max()
    log_mean_exp = max_marginal + torch.log(
        torch.exp(t_marginal - max_marginal).mean()
    )
    return t_joint.mean() - log_mean_exp


def sparse_laplacian_l1(matrices: list[torch.Tensor]) -> torch.Tensor:
    """Sparse-Laplacian L1 regularizer (MacKay 2003 ch. 28).

    Args:
        matrices: list of weight tensors (e.g. FiLM proj weights).

    Returns:
        scalar L1 norm sum (in nats per analogous interpretation).
    """
    if not matrices:
        return torch.tensor(0.0)
    return sum(m.abs().sum() for m in matrices)


class MDLIBPSJIBLoss(nn.Module):
    """Composite IB loss: MINE-based lower bound + sparse-Laplacian regularizer.

    Total IB loss:
        L_IB = -beta * mine_lower_bound(critic, z, frames, z_shuffled)
                + lambda_sparse * sum(|W_film|_1)

    Note the NEGATIVE sign on the MINE term: we MAXIMIZE the MINE lower
    bound on I(z; frames) which is equivalent to MINIMIZING the negative.
    Then beta scales this term in the joint substrate + IB optimization.

    Per CC-J-3 unwind: beta is operator-tunable (default 1e-3; sweep
    {1e-5, 1e-4, 1e-3, 1e-2}).

    Per Path B5 MacKay influence: sparse-Laplacian on FiLM matrices.
    """

    def __init__(
        self,
        beta: float = 1e-3,
        lambda_sparse: float = DEFAULT_LAMBDA_SPARSE,
    ) -> None:
        super().__init__()
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0; got {beta}")
        if lambda_sparse < 0.0:
            raise ValueError(f"lambda_sparse must be >= 0; got {lambda_sparse}")
        self.beta = float(beta)
        self.lambda_sparse = float(lambda_sparse)

    def forward(
        self,
        critic: MINECritic,
        z_joint: torch.Tensor,
        frames_joint: torch.Tensor,
        z_marginal: torch.Tensor,
        film_matrices: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward composite IB loss.

        Returns:
            (loss_term, parts_dict) where:
            - loss_term: scalar composite loss.
            - parts_dict: {"mine_lower_bound", "sparse_l1", "loss_ib"}.
        """
        mi_lb = mine_lower_bound(critic, z_joint, frames_joint, z_marginal)
        sparse_l1 = sparse_laplacian_l1(film_matrices or [])
        # MAXIMIZE MI -> MINIMIZE negative; scaled by beta
        loss = -self.beta * mi_lb + self.lambda_sparse * sparse_l1
        parts = {
            "mine_lower_bound": mi_lb.detach(),
            "sparse_l1": sparse_l1.detach() if isinstance(sparse_l1, torch.Tensor)
                        else torch.tensor(float(sparse_l1)),
            "loss_ib": loss.detach(),
            "beta": torch.tensor(self.beta),
            "lambda_sparse": torch.tensor(self.lambda_sparse),
        }
        return loss, parts


__all__ = [
    "MDLIBPSJIBLoss",
    "MINECritic",
    "mine_lower_bound",
    "sparse_laplacian_l1",
]
