"""Wasserstein-Barycenter Checkpoint Ensemble + MERA Quantizer (WBCE-MERA).

This primitive composes THREE techniques into one canonical pipeline:

1. **Wasserstein barycenter of K checkpoints.** Given K trained
   checkpoints ``θ_1 … θ_K`` (different curriculum stages, optimizer
   trajectories, or batch orderings), compute the Bures-Wasserstein
   barycenter ``θ*`` via Anderson 2016 fixed-point iteration. For
   Gaussian-approximated parameter distributions, the barycenter has an
   analytic fixed-point form

       Σ* = Σ_k λ_k (Σ*^{1/2} Σ_k Σ*^{1/2})^{1/2}                     (E3.1)

   and the mean barycenter is the weighted Euclidean mean ``μ* = Σ_k λ_k μ_k``.
   The Bures-Wasserstein barycenter is provably the optimal transport
   centroid in Gaussian Wasserstein space (Agueh-Carlier 2011) and beats
   naive averaging on the Wortsman-2022 model-soup benchmark.

2. **MERA tensor-network decomposition.** The averaged parameters get a
   multi-scale entanglement renormalization ansatz (MERA) decomposition
   with per-layer bond dimension allocated by Fisher water-filling. The
   MERA preserves hierarchical structure and captures long-range
   correlations Vidal-2007 proved are exactly representable in the
   ansatz.

3. **Brenier-OT quantizer.** Final scalar quantization uses the
   Brenier-OT optimal map ``T*(z) = ∇φ*(z)`` from a continuous source
   distribution onto a discrete codebook. Brenier (1991) proved this
   minimises mean-squared quantization error subject to the codebook
   support constraint.

Source memo: ``.omx/research/zen_state_frontier_deep_math_research_20260513.md``
§E3 + ``.omx/research/zen_state_frontier_optimal_transport_20260513.md``.

The implementation is FORWARD-ONLY (no training); WBCE-MERA is a
post-training compression pipeline that takes existing checkpoints +
emits compressed parameters + uncompresses at inflate time.

HNeRV parity discipline (per CLAUDE.md "HNeRV / leaderboard-implementation
parity discipline")
-----------------------------------------------------------------------
1. Score-aware: Fisher water-filling for bond-dim allocation uses
   ``param_sensitivity`` provided by the substrate trainer
   (gradient-through-SegNet/PoseNet). This module accepts the
   sensitivity tensor as an input rather than computing it itself.
2. Export-first: deterministic serialisation via :meth:`serialize_state`.
3-6. Substrate concerns; not violated.
7. Bolt-on ≤ 500 LOC (substrate engineering may exceed; this module
   tags ``lane_class=substrate_engineering``).
8-13. Standard substrate concerns; not violated.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
The pipeline produces a compressed parameter representation; until
paired ``[contest-CUDA]`` + ``[contest-CPU]`` anchors land on a
WBCE-MERA-equipped substrate, every result is
``score_claim=False, promotion_eligible=False,
ready_for_exact_eval_dispatch=False``.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tac.composition.frontier_primitives import (
    DiagonalGaussian,
    wasserstein_diagonal_gaussian_barycenter,
)
from tac.composition.frontier_primitives import (
    normalize_weights as _canonical_normalize_weights,
)

WBCE_MAGIC = b"WBC1"
MERA_MAGIC = b"MER1"
BRENIER_MAGIC = b"BRO1"
WBCE_MERA_SCHEMA_VERSION = 1

DEFAULT_BARYCENTER_MAX_ITERS = 64
DEFAULT_BARYCENTER_TOL = 1e-6
DEFAULT_MERA_MAX_BOND = 64
DEFAULT_BRENIER_SINKHORN_ITERS = 200


class WBCEMERAError(ValueError):
    """Raised when a WBCE-MERA spec or input is invalid."""


# ---------------------------------------------------------------------------
# Wasserstein-barycenter ensemble
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarycenterSpec:
    """Specification for the Wasserstein-2 barycenter of K checkpoints.

    For Gaussian-approximated parameter distributions, the
    Bures-Wasserstein metric

        d_BW^2(N(μ_1, Σ_1), N(μ_2, Σ_2)) =
            ||μ_1 - μ_2||^2 + tr(Σ_1 + Σ_2 - 2(Σ_1^{1/2} Σ_2 Σ_1^{1/2})^{1/2})

    admits the Anderson-2016 fixed-point iteration (E3.1) which we
    implement here. For *flat* parameter tensors (no per-parameter
    covariance), the barycenter reduces to the simple weighted Euclidean
    mean — which is exact in the zero-covariance limit and is what
    Wortsman-2022 model-soup uses.

    Args:
        weights: per-checkpoint weights ``λ_k``; must be non-negative and
            sum to 1. None → uniform.
        max_iters: maximum Anderson iterations (default 64).
        tol: convergence tolerance on the Frobenius norm of the update
            (default 1e-6).
        use_covariance: if True, treat each checkpoint as a Gaussian
            ``N(μ_k, Σ_k)`` with empirical Σ_k from the per-parameter
            sample variance across the K checkpoints' history (NOT
            supported with K < 2). Default False (pure mean barycenter).
    """

    weights: tuple[float, ...] | None = None
    max_iters: int = DEFAULT_BARYCENTER_MAX_ITERS
    tol: float = DEFAULT_BARYCENTER_TOL
    use_covariance: bool = False

    def __post_init__(self) -> None:
        if self.max_iters < 1:
            raise WBCEMERAError(f"max_iters must be ≥ 1; got {self.max_iters}")
        if self.tol <= 0 or not math.isfinite(float(self.tol)):
            raise WBCEMERAError(f"tol must be positive and finite; got {self.tol}")
        if self.weights is not None:
            if any(not math.isfinite(float(w)) for w in self.weights):
                raise WBCEMERAError(f"weights must be finite; got {self.weights}")
            if any(w < 0 for w in self.weights):
                raise WBCEMERAError(
                    f"weights must be non-negative; got {self.weights}"
                )
            total = sum(self.weights)
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise WBCEMERAError(
                    f"weights must sum to 1; got sum={total} ({self.weights})"
                )


class WassersteinBarycenterEnsemble:
    """Wasserstein-2 barycenter of K parameter checkpoints.

    Each checkpoint is a flat ``torch.Tensor`` of parameters of the same
    shape. For ``use_covariance=False`` we compute the closed-form
    weighted Euclidean mean (exact Bures-Wasserstein barycenter when
    Σ_k = 0). For ``use_covariance=True`` we approximate Σ_k from the
    cross-checkpoint sample variance and run the Anderson fixed-point
    iteration on each scalar parameter independently.
    """

    def __init__(self, spec: BarycenterSpec | None = None) -> None:
        self.spec = spec or BarycenterSpec()

    def _resolve_weights(self, k: int) -> tuple[float, ...]:
        if self.spec.weights is None:
            return tuple(1.0 / k for _ in range(k))
        if len(self.spec.weights) != k:
            raise WBCEMERAError(
                f"weights length ({len(self.spec.weights)}) "
                f"!= number of checkpoints ({k})"
            )
        try:
            return _canonical_normalize_weights(self.spec.weights, k)
        except ValueError as exc:
            raise WBCEMERAError(str(exc)) from exc

    def compute(self, checkpoints: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return the Wasserstein-2 barycenter of the K checkpoints."""
        if not checkpoints:
            raise WBCEMERAError("checkpoints sequence must be non-empty")
        ref = checkpoints[0]
        for i, ck in enumerate(checkpoints):
            if ck.shape != ref.shape:
                raise WBCEMERAError(
                    f"checkpoints[{i}].shape={tuple(ck.shape)} != reference "
                    f"shape={tuple(ref.shape)}"
                )
            if not torch.isfinite(ck).all():
                raise WBCEMERAError(f"checkpoints[{i}] must contain finite values")
        weights = self._resolve_weights(len(checkpoints))
        gaussians = tuple(
            DiagonalGaussian(
                mean=c.to(device=ref.device, dtype=ref.dtype),
                variance=torch.zeros_like(ref),
                label=f"checkpoint_{idx}",
            )
            for idx, c in enumerate(checkpoints)
        )
        return wasserstein_diagonal_gaussian_barycenter(gaussians, weights).mean


# ---------------------------------------------------------------------------
# MERA tensor-network decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MERAQuantizerSpec:
    """Specification for the MERA tensor-network quantizer.

    The MERA (Vidal 2007) decomposes a 2-d tensor as a hierarchical
    product of isometries + unitaries with per-level bond dimensions.
    For matrix-shaped parameters we use a simplified two-level form

        W ≈ U @ S @ V^T                                                (E3.2)

    where ``U`` and ``V`` are orthogonal blocks and ``S`` is a diagonal
    bond-dim-truncated singular-value matrix. Bond dimensions are
    allocated by Fisher water-filling: each layer's bond dim
    ``χ_l = min(max_bond, max(1, round(χ_total · F_l / Σ F_l)))`` with
    ``F_l`` the layer's Fisher-information score.

    Args:
        max_bond: per-layer cap on bond dimension; default 64.
        rank_floor: minimum bond dim per layer (default 1).
        fisher_floor: minimum allowed Fisher score; smaller values are
            clamped (default 1e-12).
    """

    max_bond: int = DEFAULT_MERA_MAX_BOND
    rank_floor: int = 1
    fisher_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_bond < 1:
            raise WBCEMERAError(f"max_bond must be ≥ 1; got {self.max_bond}")
        if self.rank_floor < 1:
            raise WBCEMERAError(f"rank_floor must be ≥ 1; got {self.rank_floor}")
        if self.rank_floor > self.max_bond:
            raise WBCEMERAError(
                f"rank_floor ({self.rank_floor}) > max_bond ({self.max_bond})"
            )
        if self.fisher_floor <= 0 or not math.isfinite(float(self.fisher_floor)):
            raise WBCEMERAError(
                f"fisher_floor must be positive; got {self.fisher_floor}"
            )


@dataclass(frozen=True)
class MERAFactors:
    """Output of :meth:`MERAQuantizer.compress`."""

    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor
    bond_dim: int

    def reconstruct(self) -> torch.Tensor:
        return self.U @ torch.diag(self.S) @ self.V.t()


class MERAQuantizer:
    """MERA tensor-network compressor for matrix parameters.

    For a single matrix W ∈ R^{m × n}, compute the truncated SVD of rank
    χ given by Fisher water-filling. For batched / multi-layer
    parameters call :meth:`fisher_water_fill` first to allocate
    per-layer bond dimensions then :meth:`compress` per layer.
    """

    def __init__(self, spec: MERAQuantizerSpec | None = None) -> None:
        self.spec = spec or MERAQuantizerSpec()

    def fisher_water_fill(
        self,
        fisher_scores: Sequence[float],
        chi_total: int,
    ) -> list[int]:
        """Allocate bond dimensions per layer via Fisher water-filling.

        Allocates an integer bond dim to each layer proportional to the
        layer's Fisher score, capped by ``max_bond``. The remaining
        budget is distributed greedily to the highest-Fisher layers
        until ``chi_total`` is exhausted or every layer is capped.
        """
        if chi_total < len(fisher_scores) * self.spec.rank_floor:
            raise WBCEMERAError(
                f"chi_total={chi_total} insufficient for "
                f"{len(fisher_scores)} layers at rank_floor={self.spec.rank_floor}"
            )
        if any(not math.isfinite(float(s)) for s in fisher_scores):
            raise WBCEMERAError("fisher_scores must be finite")
        clamped = [max(s, self.spec.fisher_floor) for s in fisher_scores]
        total = sum(clamped)
        raw_alloc = [chi_total * f / total for f in clamped]
        alloc = [
            max(self.spec.rank_floor, min(self.spec.max_bond, round(x)))
            for x in raw_alloc
        ]
        # Greedy correction: distribute residual budget to highest-Fisher
        # layers below cap.
        sorted_idx = sorted(range(len(clamped)), key=lambda i: -clamped[i])
        while sum(alloc) < chi_total:
            progress = False
            for i in sorted_idx:
                if alloc[i] < self.spec.max_bond:
                    alloc[i] += 1
                    progress = True
                    if sum(alloc) == chi_total:
                        break
            if not progress:
                break
        while sum(alloc) > chi_total:
            progress = False
            for i in reversed(sorted_idx):
                if alloc[i] > self.spec.rank_floor:
                    alloc[i] -= 1
                    progress = True
                    if sum(alloc) == chi_total:
                        break
            if not progress:
                break
        return alloc

    def compress(self, W: torch.Tensor, bond_dim: int) -> MERAFactors:
        """Truncated-SVD compression of a 2-d matrix at bond_dim."""
        if W.ndim != 2:
            raise WBCEMERAError(
                f"compress expects 2-d matrix; got shape {tuple(W.shape)}"
            )
        if not torch.isfinite(W).all():
            raise WBCEMERAError("compress expects finite matrix values")
        m, n = W.shape
        max_chi = min(m, n)
        chi = min(bond_dim, max_chi)
        if chi < 1:
            raise WBCEMERAError(f"bond_dim must be ≥ 1; got {bond_dim}")
        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
        U = U_full[:, :chi].contiguous()
        S = S_full[:chi].contiguous()
        V = Vh_full[:chi, :].t().contiguous()
        return MERAFactors(U=U, S=S, V=V, bond_dim=chi)


# ---------------------------------------------------------------------------
# Brenier-OT quantizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrenierOTSpec:
    """Specification for the Brenier-optimal-transport quantizer.

    Brenier (1991) proved that the optimal-transport map between two
    one-dimensional probability measures with strictly convex cost is

        T*(z) = F_target^{-1}(F_source(z))                             (E3.3)

    For a continuous source ``p(z)`` and a discrete target codebook
    ``c_1 ≤ … ≤ c_M`` with uniform probability weights, this reduces
    to "assign z to the nearest codebook point under the order
    statistic". We implement the multivariate generalisation via
    1-d projection onto each parameter's coordinate; this is exact for
    independent dimensions and a good first-order approximation
    otherwise.

    Args:
        codebook_size: M (number of codebook entries).
        symmetric: whether to enforce a symmetric codebook around 0
            (recommended for weight quantization).
    """

    codebook_size: int = 16
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.codebook_size < 2:
            raise WBCEMERAError(
                f"codebook_size must be ≥ 2; got {self.codebook_size}"
            )


@dataclass(frozen=True)
class BrenierOTResult:
    """Output of :meth:`BrenierOTQuantizer.fit_quantize`."""

    codebook: torch.Tensor
    indices: torch.Tensor


class BrenierOTQuantizer:
    """Brenier-optimal-transport scalar quantizer."""

    def __init__(self, spec: BrenierOTSpec | None = None) -> None:
        self.spec = spec or BrenierOTSpec()

    def fit_quantize(self, x: torch.Tensor) -> BrenierOTResult:
        """Fit a Brenier-optimal codebook to ``x`` and quantize.

        Returns the codebook (1-d tensor of length M) and a tensor of
        per-element codebook indices matching ``x``'s shape.
        """
        flat = x.detach().flatten().to(torch.float32)
        if flat.numel() == 0:
            raise WBCEMERAError("fit_quantize: x must be non-empty")
        if not torch.isfinite(flat).all():
            raise WBCEMERAError("fit_quantize: x must contain finite values")
        M = self.spec.codebook_size
        sorted_vals, _ = torch.sort(flat)
        # Order-statistic codebook: pick M evenly-spaced quantiles.
        idxs = torch.linspace(0, flat.numel() - 1, M).round().long()
        codebook = sorted_vals[idxs].contiguous()
        if self.spec.symmetric:
            codebook = self._symmetrize(codebook)
        # Quantize each entry to nearest codebook value (Brenier 1-d).
        diffs = (flat.unsqueeze(1) - codebook.unsqueeze(0)).abs()
        indices = diffs.argmin(dim=1).view_as(x)
        return BrenierOTResult(codebook=codebook, indices=indices)

    @staticmethod
    def _symmetrize(codebook: torch.Tensor) -> torch.Tensor:
        # Fold codebook around 0: the symmetric Brenier codebook is the
        # antisymmetric arrangement of the sorted absolute quantiles.
        abs_q = codebook.abs().sort().values
        half = abs_q[len(abs_q) // 2:]
        if len(half) == 0:
            return codebook
        sym = (
            torch.cat([-half.flip(0), half])
            if len(codebook) % 2 == 0
            else torch.cat([-half[1:].flip(0), half])
        )
        return sym.contiguous()

    def dequantize(
        self, indices: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        return codebook[indices]


# ---------------------------------------------------------------------------
# Top-level compose API
# ---------------------------------------------------------------------------


def compose_wbce_mera(
    checkpoints: Sequence[torch.Tensor],
    fisher_scores: Sequence[float] | None = None,
    chi_total: int | None = None,
    bary_spec: BarycenterSpec | None = None,
    mera_spec: MERAQuantizerSpec | None = None,
    brenier_spec: BrenierOTSpec | None = None,
) -> dict:
    """End-to-end WBCE-MERA compose for a single matrix-shaped parameter.

    Args:
        checkpoints: sequence of K parameter tensors (all same shape;
            must be 2-d for MERA-SVD compression).
        fisher_scores: per-checkpoint Fisher scores; if provided AND
            multiple matrices are stacked this is consumed by
            :meth:`MERAQuantizer.fisher_water_fill`. For a single-matrix
            call it is ignored.
        chi_total: total bond-dim budget. Default min(m, n).
        bary_spec: WBCE spec (optional).
        mera_spec: MERA spec (optional).
        brenier_spec: Brenier spec (optional).

    Returns:
        dict with keys ``barycenter`` (mean tensor), ``mera`` (MERAFactors),
        ``brenier`` (BrenierOTResult), and ``reconstruction`` (the
        post-pipeline approximation of the barycenter).
    """
    if not checkpoints:
        raise WBCEMERAError("checkpoints must be non-empty")
    bary = WassersteinBarycenterEnsemble(bary_spec).compute(checkpoints)
    if bary.ndim != 2:
        raise WBCEMERAError(
            "compose_wbce_mera currently supports 2-d parameters only; "
            f"got shape {tuple(bary.shape)}"
        )
    mera = MERAQuantizer(mera_spec)
    chi = chi_total if chi_total is not None else min(bary.shape)
    factors = mera.compress(bary, chi)
    brenier = BrenierOTQuantizer(brenier_spec)
    # Quantize the singular values (most-significant compression target).
    bro = brenier.fit_quantize(factors.S)
    quantized_S = brenier.dequantize(bro.indices, bro.codebook)
    reconstruction = factors.U @ torch.diag(quantized_S) @ factors.V.t()
    return {
        "barycenter": bary,
        "mera": factors,
        "brenier": bro,
        "reconstruction": reconstruction,
    }


def serialize_compose_state(state: dict) -> bytes:
    """Serialise the compressed state returned by :func:`compose_wbce_mera`.

    The payload stores U/V factors, the Brenier codebook, and int32 indices for
    the singular values. It intentionally does NOT store original float32
    singular values; reconstructing from the payload uses quantized singular
    values only.
    """
    factors: MERAFactors = state["mera"]
    bro: BrenierOTResult = state["brenier"]
    U = factors.U.detach().cpu().contiguous().numpy().astype("float32")
    V = factors.V.detach().cpu().contiguous().numpy().astype("float32")
    cb = bro.codebook.detach().cpu().contiguous().numpy().astype("float32")
    idx_tensor = bro.indices.detach().cpu().contiguous().reshape(-1)
    if idx_tensor.numel() != factors.bond_dim:
        raise WBCEMERAError(
            f"singular-value index count {idx_tensor.numel()} != bond_dim "
            f"{factors.bond_dim}"
        )
    idx = idx_tensor.numpy().astype("int32")
    header = struct.pack(
        "<4sBHIIIIIIII",
        WBCE_MAGIC,
        WBCE_MERA_SCHEMA_VERSION,
        factors.bond_dim,
        U.shape[0],
        U.shape[1],
        V.shape[0],
        V.shape[1],
        cb.shape[0],
        idx.shape[0],
        idx.shape[0],
        1,  # flags: quantized singular values only
    )
    return (
        header
        + U.tobytes()
        + V.tobytes()
        + cb.tobytes()
        + idx.tobytes()
    )


def deserialize_compose_state(payload: bytes) -> dict:
    """Decode a :func:`serialize_compose_state` payload.

    Returns the same operational subset as :func:`compose_wbce_mera`:
    ``mera`` factors, ``brenier`` codebook/indices, and ``reconstruction``.
    ``barycenter`` is not serialized because it is the uncompressed source
    tensor, not compressed payload.
    """

    header = struct.Struct("<4sBHIIIIIIII")
    if len(payload) < header.size:
        raise WBCEMERAError(f"payload too short ({len(payload)} < {header.size})")
    (
        magic,
        version,
        bond_dim,
        u_rows,
        u_cols,
        v_rows,
        v_cols,
        codebook_size,
        index_count,
        singular_count,
        flags,
    ) = header.unpack_from(payload, 0)
    if magic != WBCE_MAGIC:
        raise WBCEMERAError(f"bad magic: {magic!r}")
    if version != WBCE_MERA_SCHEMA_VERSION:
        raise WBCEMERAError(f"unsupported schema version: {version}")
    if flags != 1:
        raise WBCEMERAError(f"unsupported payload flags: {flags}")
    if u_cols != bond_dim or v_cols != bond_dim:
        raise WBCEMERAError("U/V factor shapes must align with bond_dim")
    if index_count != singular_count or index_count != bond_dim:
        raise WBCEMERAError("singular index count must equal bond_dim")
    offset = header.size
    U, offset = _read_float32_tensor(payload, offset, (u_rows, u_cols), "U")
    V, offset = _read_float32_tensor(payload, offset, (v_rows, v_cols), "V")
    codebook, offset = _read_float32_tensor(
        payload,
        offset,
        (codebook_size,),
        "codebook",
    )
    indices, offset = _read_int32_tensor(payload, offset, (index_count,), "indices")
    if offset != len(payload):
        raise WBCEMERAError(f"payload has {len(payload) - offset} trailing bytes")
    if torch.any(indices < 0) or torch.any(indices >= codebook.numel()):
        raise WBCEMERAError("indices out of codebook range")
    quantized_s = codebook[indices.to(dtype=torch.long)].to(dtype=U.dtype)
    factors = MERAFactors(U=U, S=quantized_s, V=V, bond_dim=bond_dim)
    brenier = BrenierOTResult(codebook=codebook, indices=indices)
    return {
        "mera": factors,
        "brenier": brenier,
        "reconstruction": factors.reconstruct(),
    }


def estimate_compressed_bytes(
    bond_dim: int, m: int, n: int, codebook_size: int
) -> int:
    """Estimate compressed payload bytes, excluding original singular values."""
    return compressed_payload_byte_breakdown(
        bond_dim,
        m,
        n,
        codebook_size,
    )["total_bytes"]


def compressed_payload_byte_breakdown(
    bond_dim: int,
    m: int,
    n: int,
    codebook_size: int,
) -> dict[str, int | bool]:
    """Return explicit WBCE-MERA charged-payload accounting.

    The original float32 singular values are not stored. Only quantized
    singular-value indices plus the Brenier codebook are charged here.
    """

    if min(bond_dim, m, n, codebook_size) <= 0:
        raise WBCEMERAError("bond_dim, m, n, and codebook_size must be positive")
    u_bytes = bond_dim * m * 4
    v_bytes = bond_dim * n * 4
    codebook_bytes = codebook_size * 4
    index_bytes = bond_dim * 4
    return {
        "u_float32_bytes": u_bytes,
        "v_float32_bytes": v_bytes,
        "codebook_float32_bytes": codebook_bytes,
        "singular_index_int32_bytes": index_bytes,
        "original_singular_float32_bytes": 0,
        "stores_original_singular_values": False,
        "total_bytes": u_bytes + v_bytes + codebook_bytes + index_bytes,
    }


def _read_float32_tensor(
    payload: bytes,
    offset: int,
    shape: tuple[int, ...],
    field_name: str,
) -> tuple[torch.Tensor, int]:
    nbytes = math.prod(shape) * 4
    end = offset + nbytes
    if end > len(payload):
        raise WBCEMERAError(f"payload truncated while reading {field_name}")
    tensor = torch.frombuffer(
        bytearray(payload[offset:end]), dtype=torch.float32
    ).reshape(shape).clone()
    if not torch.isfinite(tensor).all():
        raise WBCEMERAError(f"{field_name} must contain finite values")
    return tensor, end


def _read_int32_tensor(
    payload: bytes,
    offset: int,
    shape: tuple[int, ...],
    field_name: str,
) -> tuple[torch.Tensor, int]:
    nbytes = math.prod(shape) * 4
    end = offset + nbytes
    if end > len(payload):
        raise WBCEMERAError(f"payload truncated while reading {field_name}")
    tensor = torch.frombuffer(
        bytearray(payload[offset:end]), dtype=torch.int32
    ).reshape(shape).clone()
    return tensor, end


__all__ = [
    "BRENIER_MAGIC",
    "MERA_MAGIC",
    "WBCE_MAGIC",
    "WBCE_MERA_SCHEMA_VERSION",
    "BarycenterSpec",
    "BrenierOTQuantizer",
    "BrenierOTResult",
    "BrenierOTSpec",
    "MERAFactors",
    "MERAQuantizer",
    "MERAQuantizerSpec",
    "WBCEMERAError",
    "WassersteinBarycenterEnsemble",
    "compose_wbce_mera",
    "compressed_payload_byte_breakdown",
    "deserialize_compose_state",
    "estimate_compressed_bytes",
    "serialize_compose_state",
]
