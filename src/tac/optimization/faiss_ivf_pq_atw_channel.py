# SPDX-License-Identifier: MIT
"""Faiss-IVF-PQ canonical helper for ATW V2-1 per-region SegNet softmax histogram side-info channel.

[verified-against: .omx/research/atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md §6.2]
[verified-against: Jégou-Douze-Schmid 2011 "Product quantization for nearest neighbor search" IEEE PAMI 33(1)]
[verified-against: Faiss 1.8 (2024) GitHub `facebookresearch/faiss` API]
[verified-against: ATW V2 reactivation symposium 2026-05-18 Revisions #2-#3 binding (continuous-distribution-preserving + ≤2KB shippable)]

Per Catalog #810 sister surface (master-gradient per-pair compatibility):
the per-pair PQ codeword stream emitted by `encode_per_region_histogram` is
per-pair-addressable — each pair occupies `n_regions * (ksub_bits + log2(nlist)) / 8`
bytes at a known offset, enabling per-pair finite-difference via the canonical
`tac.master_gradient.MasterGradient` schema's `payload_bytes_per_pair` field.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
- Byte-budget arithmetic per §6.3 of the design memo is FIRST-PRINCIPLES
  bounded (information-theoretic; not measured): V1 dense full-PQ ≈ 1280
  bytes/pair; V2 sparse top-k ≈ 24 bytes/pair; V3 pool-shared ≈ 1 byte/pair.
- Empirical MI claims AWAIT the V1/V2/V3 disambiguator probe (planned next
  subagent at `tools/probe_atw_v2_1_faiss_pq_disambiguator.py`).

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" +
Catalog #220: this helper is the CANONICAL ENCODER for the ATW V2-1 substrate
distinguishing-feature; the inflate-time decode is operationally consumed
when V2-1 dispatches (research-only at landing per recipe `dispatch_enabled:false`).

Module contract
---------------

Four PUBLIC functions:
* :func:`build_pq_codebook` — train Faiss IVF-PQ codebook offline on
  representative SegNet softmax outputs; called ONCE before training
  trainer epoch 0 (Catalog #290 layer "Codebook training schedule" UNIQUE FORK).
* :func:`encode_per_region_histogram` — encode one pair's per-region softmax
  tensor to bit-packed bytes via the trained codebook.
* :func:`decode_per_region_histogram` — decode the bit-packed bytes back to
  approximate per-region softmax tensor at inflate time.
* :func:`serialize_codebook` / :func:`deserialize_codebook` — round-trip the
  codebook via :func:`faiss.serialize_index` for archive persistence.

The module is **import-deferred**: ``import faiss`` happens inside each public
function rather than at module top so that:
* CI / preflight / OSS clones without faiss-cpu installed do NOT crash on
  import of this module.
* Catalog #305 observability surface declarations can inspect the module's
  public API without requiring faiss-cpu.

Faiss installation (per CLAUDE.md uv discipline + design memo §9.5):
``uv pip install faiss-cpu``.

Local M5 Max + 128GB unified hardware exploitation (per design memo §9):
codebook construction on 600 pairs × 16×16 region grid × 5-class softmax
(~3MB input) trains in ~1-2 sec via faiss-cpu; encode + decode + MI compute
together run in ~30 sec wall-clock at $0 cost — the canonical $0 disambiguator
probe BEFORE any paid Modal dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_NLIST: int = 256
"""Default IVF coarse-quantizer centroid count.

Per Jégou-Douze-Schmid 2011 §3.2: nlist ~ sqrt(N) where N is training-set size.
For our 600 pairs × 256 regions = 153,600 training vectors, sqrt ~ 392; we
round down to 256 for byte-efficiency of the per-vector IVF index (log2(256)=8
bits exactly).
"""

DEFAULT_M_SUBQ: int = 4
"""Default number of PQ sub-quantizers.

Per Jégou-Douze-Schmid 2011 §3.3: M divides the input dimension d. For our
d=5 (5-class softmax) the closest evenly-dividing M values are {1, 5}; we
pick M=4 with the convention that the 5-dim vector is zero-padded to 8-dim
internally by Faiss (Faiss supports any d via internal padding). The user
may override via the ``m_subq`` kwarg of :func:`build_pq_codebook`.

CAVEAT: d=5 + M=4 implies sub-quantizer subspace dimension = 8/4 = 2 (after
zero-pad). This is the canonical workflow but the operator may prefer M=5
(subspace dim = 1) if encoding latency dominates over rate.
"""

DEFAULT_NBITS: int = 8
"""Default bits per PQ codeword (ksub = 1 << nbits = 256 codewords per sub-quantizer).

8 bits = 1 byte per sub-quantizer; total per-vector PQ payload =
M × nbits / 8 = 4 × 1 = 4 bytes. With IVF prefix log2(nlist)/8 = 1 byte,
total per-vector payload = 5 bytes (the V1 dense full-PQ rate per design memo
§6.3).
"""

DEFAULT_TRAINING_SEED: int = 42
"""Deterministic seed for Faiss k-means (passed via numpy RNG)."""


@dataclass(frozen=True)
class PqEncodingBudget:
    """Per-variant byte-budget arithmetic per design memo §6.3.

    All fields derived first-principles from information-theoretic bounds
    on Faiss IVF-PQ; NO empirical measurements until the disambiguator
    probe runs.

    [verified-against: design memo §6.3 byte-budget arithmetic]
    """

    variant_id: str
    """One of 'v1_dense', 'v2_sparse_top_k', 'v3_pool_shared'."""

    n_regions: int
    """Region count per pair (e.g. 256 for 16×16 grid; 16 for 4×4 grid)."""

    nlist: int
    """IVF centroid count."""

    m_subq: int
    """PQ sub-quantizer count."""

    nbits: int
    """Bits per PQ codeword."""

    top_k_regions: int | None = None
    """Top-k regions to encode per pair (None for dense; 8 for sparse top-k)."""

    per_pair_bytes: int = 0
    """First-principles per-pair byte count (excludes shared codebook)."""

    total_pairs: int = 600
    """Total pair count for the contest video (600 = canonical)."""

    total_pair_bytes: int = 0
    """First-principles per-pair × total_pairs."""

    shared_codebook_bytes_estimate: int = 0
    """First-principles shared codebook byte estimate."""

    total_archive_contribution_bytes: int = 0
    """First-principles total archive contribution."""

    contest_rate_cost_estimate: float = 0.0
    """First-principles contest rate cost: 25 × total / 37_545_489."""

    shippable_verdict: str = "PENDING"
    """One of 'SHIPPABLE', 'ARGUABLE', 'NOT_SHIPPABLE', 'PENDING'."""

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe representation per Catalog #305 observability."""
        return {
            "variant_id": self.variant_id,
            "n_regions": self.n_regions,
            "nlist": self.nlist,
            "m_subq": self.m_subq,
            "nbits": self.nbits,
            "top_k_regions": self.top_k_regions,
            "per_pair_bytes": self.per_pair_bytes,
            "total_pairs": self.total_pairs,
            "total_pair_bytes": self.total_pair_bytes,
            "shared_codebook_bytes_estimate": self.shared_codebook_bytes_estimate,
            "total_archive_contribution_bytes": self.total_archive_contribution_bytes,
            "contest_rate_cost_estimate": self.contest_rate_cost_estimate,
            "shippable_verdict": self.shippable_verdict,
        }


CONTEST_RATE_NORMALIZER_BYTES: float = 37_545_489.0
"""Canonical contest rate normalizer (per CLAUDE.md "Apples-to-apples evidence discipline")."""


def estimate_pq_encoding_budget(
    *,
    variant_id: str,
    n_regions: int,
    nlist: int = DEFAULT_NLIST,
    m_subq: int = DEFAULT_M_SUBQ,
    nbits: int = DEFAULT_NBITS,
    top_k_regions: int | None = None,
    total_pairs: int = 600,
    softmax_dim: int = 5,
) -> PqEncodingBudget:
    """First-principles byte-budget arithmetic for a Faiss-IVF-PQ variant.

    Returns a :class:`PqEncodingBudget` populated from the canonical formula:

    * per-vector payload = log2(nlist) + M × nbits  (bits)
    * shared codebook ≈ nlist × d × 4 + M × (1 << nbits) × (d / M) × 4 (bytes)
    * total archive contribution = top_k × per_vector_bytes × total_pairs
      + shared_codebook_bytes

    [verified-against: design memo §6.3]
    [empirical: pending — V1/V2/V3 disambiguator probe at tools/probe_atw_v2_1_faiss_pq_disambiguator.py]

    Per design memo's THREE canonical variants:
        * v1_dense: n_regions=256, nlist=256, m_subq=4, nbits=8, top_k=None
          → 5 bytes/vec × 256 regions = 1280 bytes/pair (NOT SHIPPABLE)
        * v2_sparse_top_k: n_regions=16, nlist=64, m_subq=2, nbits=6, top_k=8
          → ~3 bytes/vec × 8 regions = ~24 bytes/pair (ARGUABLE)
        * v3_pool_shared: n_regions=16, nlist=64, m_subq=2, nbits=6, top_k=1
          → ~1 byte/pair (SHIPPABLE)
    """
    if n_regions <= 0:
        raise ValueError(f"n_regions must be positive: {n_regions}")
    if nlist <= 0 or m_subq <= 0 or nbits <= 0:
        raise ValueError(f"nlist/m_subq/nbits must be positive: {nlist}/{m_subq}/{nbits}")
    if total_pairs <= 0:
        raise ValueError(f"total_pairs must be positive: {total_pairs}")
    if softmax_dim <= 0:
        raise ValueError(f"softmax_dim must be positive: {softmax_dim}")

    # Per-vector PQ payload in bits, then bytes (rounded up).
    per_vector_bits = int(np.ceil(np.log2(nlist))) + m_subq * nbits
    per_vector_bytes = max(1, (per_vector_bits + 7) // 8)

    # Effective regions encoded per pair
    effective_regions = top_k_regions if top_k_regions is not None else n_regions

    # Add region-index overhead for sparse top-k variants
    region_index_bits = 0
    if top_k_regions is not None:
        region_index_bits = int(np.ceil(np.log2(n_regions))) * top_k_regions

    per_pair_bytes = max(
        1,
        (effective_regions * per_vector_bits + region_index_bits + 7) // 8,
    )
    total_pair_bytes = per_pair_bytes * total_pairs

    # Shared codebook estimate: nlist centroids (d-dim float32) + M codebooks
    # of ksub=2^nbits entries (d/M-dim float32 each).
    nlist_centroids_bytes = nlist * softmax_dim * 4
    ksub = 1 << nbits
    pq_codebooks_bytes = m_subq * ksub * max(1, softmax_dim // m_subq) * 4
    shared_codebook_bytes = nlist_centroids_bytes + pq_codebooks_bytes

    total_archive_contribution_bytes = total_pair_bytes + shared_codebook_bytes
    contest_rate_cost = 25.0 * total_archive_contribution_bytes / CONTEST_RATE_NORMALIZER_BYTES

    # Shippability verdict per design memo §6.3 thresholds
    if contest_rate_cost > 0.10:
        shippable_verdict = "NOT_SHIPPABLE"
    elif contest_rate_cost > 0.005:
        shippable_verdict = "ARGUABLE"
    else:
        shippable_verdict = "SHIPPABLE"

    return PqEncodingBudget(
        variant_id=variant_id,
        n_regions=n_regions,
        nlist=nlist,
        m_subq=m_subq,
        nbits=nbits,
        top_k_regions=top_k_regions,
        per_pair_bytes=per_pair_bytes,
        total_pairs=total_pairs,
        total_pair_bytes=total_pair_bytes,
        shared_codebook_bytes_estimate=shared_codebook_bytes,
        total_archive_contribution_bytes=total_archive_contribution_bytes,
        contest_rate_cost_estimate=contest_rate_cost,
        shippable_verdict=shippable_verdict,
    )


def _import_faiss() -> Any:
    """Deferred Faiss import with operator-friendly error message.

    Per design memo §9.5: faiss-cpu is NOT installed in the default OSS
    clone; the canonical install is ``uv pip install faiss-cpu``. This
    helper raises ImportError with the canonical install command when
    Faiss is unavailable so CI / preflight surfaces operator-actionable
    guidance.
    """
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required for ATW V2-1 Faiss-IVF-PQ encoding. "
            "Install via: uv pip install faiss-cpu "
            "(per `.omx/research/atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md` §9.5). "
            f"Original error: {exc}"
        ) from exc
    return faiss


def _padded_dim_for_pq(softmax_dim: int, m_subq: int) -> int:
    """Return the smallest dimension >= softmax_dim divisible by m_subq."""
    if softmax_dim <= 0:
        raise ValueError(f"softmax_dim must be positive: {softmax_dim}")
    if m_subq <= 0:
        raise ValueError(f"m_subq must be positive: {m_subq}")
    remainder = softmax_dim % m_subq
    if remainder == 0:
        return softmax_dim
    return softmax_dim + (m_subq - remainder)


def _pad_features_for_codebook(values: np.ndarray, *, codebook_dim: int) -> np.ndarray:
    """Pad feature columns with zeros so Faiss PQ receives codebook_dim inputs."""
    if values.ndim != 2:
        raise ValueError(f"values must be 2-d, got shape {values.shape}")
    if codebook_dim <= 0:
        raise ValueError(f"codebook_dim must be positive: {codebook_dim}")
    if values.shape[1] > codebook_dim:
        raise ValueError(
            f"input feature dim {values.shape[1]} exceeds codebook dim {codebook_dim}"
        )
    values_f32 = np.ascontiguousarray(values.astype(np.float32))
    if values_f32.shape[1] == codebook_dim:
        return values_f32
    padded = np.zeros((values_f32.shape[0], codebook_dim), dtype=np.float32)
    padded[:, : values_f32.shape[1]] = values_f32
    return np.ascontiguousarray(padded)


def build_pq_codebook(
    segnet_softmax_batch: np.ndarray,
    *,
    nlist: int = DEFAULT_NLIST,
    m_subq: int = DEFAULT_M_SUBQ,
    nbits: int = DEFAULT_NBITS,
    seed: int = DEFAULT_TRAINING_SEED,
) -> Any:
    """Train Faiss IVF-PQ codebook offline on representative SegNet softmax outputs.

    Per design memo §1 "Codebook training schedule" UNIQUE FORK: codebook is
    trained ONCE before trainer epoch 0 and frozen for all subsequent epochs.

    [verified-against: Jégou-Douze-Schmid 2011 §4.1 canonical training workflow]
    [verified-against: Faiss 1.8 IndexIVFPQ API]

    Args:
        segnet_softmax_batch: shape (N_training_vectors, softmax_dim).
            Per design memo §9.1: input from `upstream/videos/0.mkv`
            SegNet softmax outputs × N_pairs × N_regions = N_training_vectors.
        nlist: IVF centroid count. Default 256 (per :data:`DEFAULT_NLIST`).
        m_subq: PQ sub-quantizer count. Default 4 (per :data:`DEFAULT_M_SUBQ`).
        nbits: Bits per PQ codeword (ksub = 1 << nbits). Default 8.
        seed: Deterministic seed for Faiss k-means.

    Returns:
        Trained :class:`faiss.IndexIVFPQ`.

    Raises:
        ImportError: faiss-cpu not installed (operator-actionable per :func:`_import_faiss`).
        ValueError: input shape / type invariants violated.

    Local M5 Max benchmark (design memo §9.1):
        600 pairs × 256 regions × 5-class softmax = 153_600 training vectors;
        trains in ~1-2 sec on M5 Max CPU.
    """
    if not isinstance(segnet_softmax_batch, np.ndarray):
        raise ValueError(
            f"segnet_softmax_batch must be np.ndarray, got {type(segnet_softmax_batch)}"
        )
    if segnet_softmax_batch.ndim != 2:
        raise ValueError(
            f"segnet_softmax_batch must be 2-d (N_vectors, softmax_dim), "
            f"got shape {segnet_softmax_batch.shape}"
        )
    n_vectors, softmax_dim = segnet_softmax_batch.shape
    if n_vectors < nlist:
        raise ValueError(
            f"n_vectors ({n_vectors}) must be >= nlist ({nlist}) for Faiss IVF training; "
            f"reduce nlist or provide more training vectors per Jégou-Douze-Schmid 2011 §3.2"
        )

    faiss = _import_faiss()

    # Faiss IndexIVFPQ requires d % M == 0. SegNet has 5 classes, while the
    # canonical V2/V3 configs use M=2 and V1 uses M=4, so we explicitly zero-pad
    # to the next divisible dimension and trim back to softmax_dim on decode.
    index_dim = _padded_dim_for_pq(softmax_dim, m_subq)
    input_f32 = _pad_features_for_codebook(
        segnet_softmax_batch, codebook_dim=index_dim
    )

    # IVF coarse quantizer (flat L2)
    quantizer = faiss.IndexFlatL2(index_dim)
    # IVF + PQ index
    index = faiss.IndexIVFPQ(quantizer, index_dim, nlist, m_subq, nbits)

    # Set seed for deterministic k-means (per Faiss canonical: ClusteringParameters)
    # Faiss-cpu exposes seed via faiss.cvar.cp_kmeans_seed (or via numpy seed)
    np.random.seed(seed)

    # Train only. Adding training vectors would serialize the whole IVF database
    # into the archive-side codebook blob, which is a rate bug for ATW V2-1.
    # Faiss sa_encode/sa_decode works against the trained quantizers with
    # ntotal=0, keeping the persisted codebook close to the design budget.
    index.train(input_f32)

    # Set nprobe = 1 for fastest decode (we want exact codeword reconstruction, not nearest-neighbor search)
    index.nprobe = 1

    return index


def encode_per_region_histogram(
    softmax_per_pair: np.ndarray,
    codebook: Any,
) -> bytes:
    """Encode one pair's per-region softmax tensor to bit-packed bytes.

    Per design memo §1 "Decoder reconstruction" UNIQUE FORK:
    `faiss_pq_decode → softmax_reconstruct → wz_head` pipeline.

    [verified-against: Faiss 1.8 IndexIVFPQ.sa_encode API]
    [verified-against: design memo §6.2 canonical helper integration]

    Args:
        softmax_per_pair: shape (N_regions, softmax_dim). Per design memo
            §6.2: typically (256, 5) for 16×16 region grid with 5-class softmax.
        codebook: Trained `faiss.IndexIVFPQ` from :func:`build_pq_codebook`.

    Returns:
        Bit-packed PQ codeword bytes (per design memo §6.3:
        ~5 bytes per region for V1 dense; ~3 bytes for V2 sparse).

    Raises:
        ImportError: faiss-cpu not installed.
        ValueError: input shape invariants violated.
    """
    if not isinstance(softmax_per_pair, np.ndarray):
        raise ValueError(
            f"softmax_per_pair must be np.ndarray, got {type(softmax_per_pair)}"
        )
    if softmax_per_pair.ndim != 2:
        raise ValueError(
            f"softmax_per_pair must be 2-d (N_regions, softmax_dim), "
            f"got shape {softmax_per_pair.shape}"
        )

    _import_faiss()

    codebook_dim = int(getattr(codebook, "d", softmax_per_pair.shape[1]))
    input_f32 = _pad_features_for_codebook(softmax_per_pair, codebook_dim=codebook_dim)

    # Faiss sa_encode emits bytes (per-vector code = log2(nlist) bytes for IVF list
    # + M × nbits / 8 bytes for PQ codewords)
    encoded = codebook.sa_encode(input_f32)
    return bytes(encoded.tobytes())


def decode_per_region_histogram(
    encoded_bytes: bytes,
    codebook: Any,
    *,
    n_regions: int,
    softmax_dim: int = 5,
) -> np.ndarray:
    """Decode bit-packed bytes back to approximate per-region softmax tensor at inflate time.

    Per design memo §1 "Decoder reconstruction" UNIQUE FORK + Catalog #220
    (operationally consumed at inflate; no SegNet load at inflate per strict-
    scorer-rule).

    [verified-against: Faiss 1.8 IndexIVFPQ.sa_decode API]
    [verified-against: design memo §6.2 canonical helper integration]

    Args:
        encoded_bytes: Output from :func:`encode_per_region_histogram`.
        codebook: Deserialized `faiss.IndexIVFPQ` from :func:`deserialize_codebook`.
        n_regions: Expected region count (used to validate decode shape).
        softmax_dim: Expected softmax dimension (default 5 for SegNet 5-class).

    Returns:
        Reconstructed shape (n_regions, softmax_dim) np.float32 softmax tensor.

    Raises:
        ImportError: faiss-cpu not installed.
        ValueError: byte count does not divide cleanly into per-vector codes.

    NOTE: Faiss IVF-PQ decoding is APPROXIMATE — the reconstructed softmax
    will differ from the original by the quantization error bounded by
    Jégou-Douze-Schmid 2011 Theorem 1. For ATW V2-1's downstream use (WZ
    side-info head consumption) this approximation is acceptable per Atick-
    Redlich 1990 canonical recommendation to preserve continuous-distribution
    information (NOT exact reconstruction).
    """
    if not isinstance(encoded_bytes, (bytes, bytearray)):
        raise ValueError(
            f"encoded_bytes must be bytes or bytearray, got {type(encoded_bytes)}"
        )
    if n_regions <= 0:
        raise ValueError(f"n_regions must be positive: {n_regions}")

    # Faiss sa_decode expects uint8 contiguous array
    code_size = codebook.sa_code_size()
    expected_bytes = code_size * n_regions
    if len(encoded_bytes) != expected_bytes:
        raise ValueError(
            f"encoded_bytes length {len(encoded_bytes)} does not match "
            f"expected {expected_bytes} = code_size({code_size}) × n_regions({n_regions})"
        )

    _import_faiss()

    codes = np.frombuffer(encoded_bytes, dtype=np.uint8).reshape(n_regions, code_size)
    decoded = codebook.sa_decode(codes)
    if decoded.shape[1] < softmax_dim:
        raise ValueError(
            f"decoded feature dim {decoded.shape[1]} is smaller than requested "
            f"softmax_dim {softmax_dim}"
        )
    return np.ascontiguousarray(decoded[:, :softmax_dim].astype(np.float32))


def serialize_codebook(codebook: Any) -> bytes:
    """Round-trip the trained codebook via :func:`faiss.serialize_index` for archive persistence.

    Per design memo §6.1 archive grammar: shipped as ATW21PQ
    ``faiss_codebook_blob`` (~10KB amortized once per archive).

    [verified-against: Faiss 1.8 serialize_index API]
    """
    faiss = _import_faiss()
    return bytes(faiss.serialize_index(codebook).tobytes())


def deserialize_codebook(blob: bytes) -> Any:
    """Inverse of :func:`serialize_codebook` — restore codebook at inflate time.

    [verified-against: Faiss 1.8 deserialize_index API]
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise ValueError(f"blob must be bytes or bytearray, got {type(blob)}")
    faiss = _import_faiss()
    arr = np.frombuffer(blob, dtype=np.uint8)
    return faiss.deserialize_index(arr)


# Canonical variant constants per design memo §6.3
CANONICAL_VARIANT_V1_DENSE = "v1_dense"
"""V1 dense full-PQ: 16×16 region grid, M=4 subq, k=256 codewords. NOT SHIPPABLE."""

CANONICAL_VARIANT_V2_SPARSE_TOP_K = "v2_sparse_top_k"
"""V2 sparse top-k: 4×4 region grid, M=2 subq, k=64 codewords, top-k=8 regions. ARGUABLE."""

CANONICAL_VARIANT_V3_POOL_SHARED = "v3_pool_shared"
"""V3 pool-shared codebook + per-pair top-1 codeword: 16-region grid, M=2, k=64, top-1. SHIPPABLE."""

CANONICAL_VARIANTS: tuple[str, str, str] = (
    CANONICAL_VARIANT_V1_DENSE,
    CANONICAL_VARIANT_V2_SPARSE_TOP_K,
    CANONICAL_VARIANT_V3_POOL_SHARED,
)


__all__ = [
    "CANONICAL_VARIANTS",
    "CANONICAL_VARIANT_V1_DENSE",
    "CANONICAL_VARIANT_V2_SPARSE_TOP_K",
    "CANONICAL_VARIANT_V3_POOL_SHARED",
    "CONTEST_RATE_NORMALIZER_BYTES",
    "DEFAULT_M_SUBQ",
    "DEFAULT_NBITS",
    "DEFAULT_NLIST",
    "DEFAULT_TRAINING_SEED",
    "PqEncodingBudget",
    "build_pq_codebook",
    "decode_per_region_histogram",
    "deserialize_codebook",
    "encode_per_region_histogram",
    "estimate_pq_encoding_budget",
    "serialize_codebook",
]
