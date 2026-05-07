"""Dynamic-learning derivers for the PR101 split-Brotli + byte-map codec.

Replaces PR101's hardcoded constants — `DECODER_STORAGE_ORDER`,
`DECODER_STREAM_ENDS`, `CONV4_STORAGE_PERMS`, `LATENT_DIM_ORDER`, and
`SIDECAR_DELTAS_X100` — with substrate-adaptive derivations.

Motivation: Op 1 empirical (commit `c18b664b` + Round 2/3 fixes
`b71b0288`/`34e69f01`) measured PR101's split-Brotli + byte-maps on PR106
as **-241 bytes** vs the predicted **-7,963 bytes**. The shortfall traces to
PR101's constants being tuned for PR101's own fine-tuned weights — they are
not portable across substrates. Per-substrate derivation is the correct
non-arbitrariness fix; the audit at
``.omx/research/pr_top3_non_arbitrariness_paper_cross_reference_20260507_claude.md``
spec'd the 5 derivers landed here.

Each deriver function is paired in §1-§5 of that audit and grounded in
literature:

* `derive_storage_order` — agglomerative clustering on tensor byte-frequency
  cosine similarity. Reference: Alakuijala et al. "Brotli: A General-Purpose
  Data Compressor" (ACM TOMS 2018) §3 context-modeling; standard hierarchical
  spectral clustering (Witten/Neal/Cleary "Arithmetic Coding for Data
  Compression" CACM 1987 on probability-model accuracy from grouping).

* `derive_stream_ends` — DP over candidate split-sets minimizing total brotli
  output. Reference: standard interval-DP; rate-distortion (Cover & Thomas
  Ch. 13) on per-segment-entropy lower bound.

* `derive_conv4_perms` — exhaustive 24-permutation brotli minimization per
  conv4-shape tensor. Reference: Selfcomp PR #56 HWOI permutation +5% gain
  on segmap weights (memory:
  `reference_pr56_selfcomp_blob_byte_layout_proper_reverse_engineering_20260501.md`);
  general autocorrelation-maximization principle in DL inference layouts
  (TVM, ONNX HWOI/IOHW debates).

* `derive_latent_dim_order` — descending per-dimension variance permutation.
  Reference: Ballé, Minnen, Singh, Hwang, Johnston "Variational Image
  Compression with a Scale Hyperprior" (ICLR 2018) — high-variance dims
  dominate rate; ordering by variance helps LZMA's match-finder for
  near-Gaussian latents.

* `derive_sidecar_codebook` — Lloyd-Max iterative scalar quantization on the
  delta distribution. Reference: Lloyd "Least Squares Quantization in PCM"
  (IEEE TIT 1982 reprint of 1957 work); Max "Quantizing for Minimum
  Distortion" (IRE TIT 1960).

Strict-scorer-rule: this module loads NO scorer weights and has zero
MPS/CUDA dependency. CPU-only, deterministic, and pure side-effect-free
arithmetic on bytes / numpy arrays. Each deriver returns a value with the
same type as the constant it replaces in
``tac.pr101_split_brotli_codec``.

Tag discipline: per `forbidden_empirical_claim_without_evidence_tag`, any
predicted-byte numbers in this module's docstrings or comments are tagged
``[predicted]`` until measured under the canonical demo
(``experiments/demo_pr101_dynamic_derivers.py``). Empirical measurements
are tagged ``[empirical:<artifact-path>]``.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch

from tac.pr101_split_brotli_codec import (
    FIXED_STATE_SCHEMA,
    LATENT_DIM,
    Pr101SplitBrotliCodecError,
    _build_per_tensor_payload,
    _quantize_tensor,
    pack_brotli_stream,
)

# PR101 packs its sidecar codebook as ``np.ndarray[np.int8]`` of length 16.
# The constant lives in PR101's source (codec.py:68-71); we mirror it here as
# the deriver's Lloyd-Max initializer and as the empty-input fallback.
SIDECAR_DELTAS_X100_PR101_DEFAULT: np.ndarray = np.array(
    [-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
    dtype=np.int8,
)
"""PR101 sidecar codebook (length 16, asymmetric, skips 0). Used as the
Lloyd-Max initializer in :func:`derive_sidecar_codebook`."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §1. Storage order — agglomerative clustering on byte-histogram cosine sim
# ---------------------------------------------------------------------------


def _check_state_dict_complete(state_dict: dict[str, torch.Tensor]) -> None:
    """Contrarian Round-1 fix: defensive check — fail loud if the input
    state_dict is missing any schema-required tensor. The derivers all
    assume the full 28-tensor schema; partial inputs would silently produce
    invalid permutations."""
    schema_names = [name for name, _ in FIXED_STATE_SCHEMA]
    missing = [n for n in schema_names if n not in state_dict]
    if missing:
        raise Pr101SplitBrotliCodecError(
            f"state_dict is missing {len(missing)} schema tensor(s); first 3: {missing[:3]}"
        )


def _quantized_byte_histogram(state_dict: dict[str, torch.Tensor], idx: int) -> np.ndarray:
    """Compute a length-256 byte-frequency histogram of tensor ``idx`` after
    PR101-style quantization + zigzag mapping.

    The histogram is the substrate-level signal we cluster on: tensors with
    similar histograms compress well when packed adjacent in a brotli stream.
    """
    name, _ = FIXED_STATE_SCHEMA[idx]
    qt = _quantize_tensor(name, state_dict[name])
    # We don't apply byte_map / conv4 perms here — the histogram is a coarse
    # similarity signal; the downstream encoder will handle per-tensor
    # transforms. Use the raw int8 reinterpreted as uint8 (twos-complement)
    # for a transform-independent histogram.
    flat_u8 = qt.q_i8.reshape(-1).view(np.uint8)
    hist = np.bincount(flat_u8, minlength=256).astype(np.float64)
    return hist


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def derive_storage_order(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    """Replace PR101's hardcoded ``DECODER_STORAGE_ORDER`` with a per-substrate
    permutation derived from agglomerative clustering on tensor
    byte-frequency cosine similarity.

    Algorithm (§1 of the audit memo):

    1. Compute the length-256 byte-frequency histogram of each schema tensor
       after symmetric INT8 quantization.
    2. Build a 28x28 cosine-similarity matrix between histograms.
    3. Run greedy single-linkage agglomerative clustering: at each step,
       merge the two clusters whose closest members have the highest
       similarity. Each cluster is represented as an ordered list; merging
       concatenates them so adjacent elements stay adjacent.
    4. Return the DFS-order traversal (which is just the final concatenated
       list).

    The resulting permutation places tensors with similar byte distributions
    adjacent, which is what brotli's context model exploits. Reference: the
    audit memo §1 cross-references Alakuijala et al. (2018) "Brotli" §3 on
    context-modeling and Witten/Neal/Cleary (1987) on probability-model
    accuracy from symbol grouping.

    Args:
        state_dict: HNeRVDecoder state dict keyed by FIXED_STATE_SCHEMA names.

    Returns:
        Length-28 tuple permuting ``range(28)`` such that adjacent indices
        are histogram-similar.

    Determinism: the algorithm is fully deterministic for a given input; tied
    similarities are broken by lower-index-first to ensure reproducibility.

    Complexity: O(n^3) with n = 28 schema size — ~22k single-linkage
    iterations + n^2 = 784 cosine similarities. Wall-clock <0.1s on a modern
    laptop. NOT suitable for n > ~100; the spec only contemplates n=28.

    Limitation (Shannon Round-1 caveat): single-linkage clustering can
    produce "chained" orderings that bunch dissimilar tensors at the
    extremes. Empirically on PR106 (see ``demo_pr101_dynamic_derivers.py``)
    storage_order alone slightly REGRESSES (+48 bytes); the win comes from
    composition with stream_ends + conv4_perms. Future work: try
    complete-linkage / average-linkage as alternative deriver variants.
    """
    _check_state_dict_complete(state_dict)
    n = len(FIXED_STATE_SCHEMA)
    hists = np.stack([_quantized_byte_histogram(state_dict, i) for i in range(n)], axis=0)

    # Pairwise cosine similarity (n x n, symmetric).
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = _cosine_similarity(hists[i], hists[j])
            sim[i, j] = s
            sim[j, i] = s

    # Greedy single-linkage agglomerative clustering. Each cluster is a list;
    # we merge the pair of clusters whose maximum cross-cluster similarity is
    # highest. Tie-break: smaller (min-index, sorted) cluster first.
    clusters: list[list[int]] = [[i] for i in range(n)]

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_sim = -np.inf
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                # Single-linkage: max over cross-cluster pairs.
                cross_sim = max(sim[i, j] for i in clusters[a] for j in clusters[b])
                # Tie-break key: prefer earlier-min-index merges for reproducibility.
                if cross_sim > best_sim or (
                    cross_sim == best_sim
                    and best_pair is not None
                    and (min(clusters[a]), min(clusters[b]))
                    < (min(clusters[best_pair[0]]), min(clusters[best_pair[1]]))
                ):
                    best_sim = cross_sim
                    best_pair = (a, b)
        if best_pair is None:  # pragma: no cover — n >= 2 always
            break
        a, b = best_pair
        # Merge: concatenate b into a (keeping adjacency by traversal).
        merged = clusters[a] + clusters[b]
        new_clusters = [merged] + [c for k, c in enumerate(clusters) if k not in (a, b)]
        clusters = new_clusters

    order = tuple(int(x) for x in clusters[0])
    if sorted(order) != list(range(n)):
        raise Pr101SplitBrotliCodecError(
            f"derive_storage_order produced non-permutation: {order}"
        )
    return order


# ---------------------------------------------------------------------------
# §2. Stream ends — DP over split-points minimizing total brotli output
# ---------------------------------------------------------------------------


def _build_payloads_in_order(
    state_dict: dict[str, torch.Tensor],
    storage_order: tuple[int, ...],
) -> list[bytes]:
    """Build per-tensor on-disk payloads in storage_order. Used by the DP +
    by the round-trip end-to-end demo."""
    quantized = [
        _quantize_tensor(name, state_dict[name])
        for name, _shape in FIXED_STATE_SCHEMA
    ]
    return [
        _build_per_tensor_payload(quantized[storage_idx], storage_idx)
        for storage_idx in storage_order
    ]


def derive_stream_ends(
    state_dict: dict[str, torch.Tensor],
    storage_order: tuple[int, ...],
    *,
    brotli_quality: int = 11,
    max_streams: int = 7,
) -> tuple[int, ...]:
    """Replace PR101's hardcoded ``DECODER_STREAM_ENDS`` with a per-substrate
    split-point set derived via DP over candidate splits minimizing total
    brotli output.

    Algorithm (§2 of the audit memo):

    Let ``parts[k]`` be the on-disk per-tensor payload at storage_order
    position ``k``. We want to partition ``[0, N)`` into ``s`` non-empty
    contiguous windows whose summed brotli compressed size is minimum,
    where ``s ∈ {1, ..., max_streams}``.

    Define ``cost(i, j)`` = ``len(brotli(concat(parts[i:j])))`` for
    ``0 <= i < j <= N``. Then for a fixed number of streams ``s``:

        dp[i][s] = min over j < i of dp[j][s-1] + cost(j, i)
        base:    dp[0][0] = 0; dp[i][0] = +inf for i > 0
                 dp[i][1] = cost(0, i)

    Final answer: ``min over s of dp[N][s]``. The split set is recovered by
    backtracking through argmins. Length ≤ ``max_streams``; PR101 uses 7,
    we mirror that cap.

    Cost: O(N² × brotli_time) for the brotli-cost lookup table + O(N² × s)
    for the DP, where N = 28 and s ≤ 7. Wall-clock: ~30s CPU on a modern
    laptop at brotli quality 11 (matches PR101's encoder setting).

    Args:
        state_dict: HNeRVDecoder state dict.
        storage_order: permutation of range(28) — pass the result of
            :func:`derive_storage_order` for the joint-optimal pipeline.
        brotli_quality: must match the encoder's quality (default 11).
        max_streams: cap on the number of streams. PR101 uses 7; the
            DP can return fewer if more streams cost more.

    Returns:
        Length-``s`` tuple of cumulative end-positions into ``storage_order``,
        with the last element equal to ``len(storage_order)``. Compatible
        wire-format with PR101's ``DECODER_STREAM_ENDS``.
    """
    _check_state_dict_complete(state_dict)
    parts = _build_payloads_in_order(state_dict, storage_order)
    n = len(parts)

    # Pre-compute cost(i, j) = len(brotli(concat(parts[i:j]))). O(n^2) brotli runs.
    INF = float("inf")
    cost = np.full((n + 1, n + 1), INF, dtype=np.float64)
    # Cache the cumulative bytes so concat is O(1) slice; brotli still O(window).
    for i in range(n):
        running = bytearray()
        for j in range(i + 1, n + 1):
            running.extend(parts[j - 1])
            cost[i, j] = float(len(pack_brotli_stream(bytes(running), quality=brotli_quality)))

    # DP over (position, num_streams_used).
    s_max = min(max_streams, n)
    dp = np.full((n + 1, s_max + 1), INF, dtype=np.float64)
    parent = np.full((n + 1, s_max + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for s in range(1, s_max + 1):
        for i in range(1, n + 1):
            for j in range(0, i):
                if dp[j, s - 1] == INF:
                    continue
                candidate = dp[j, s - 1] + cost[j, i]
                if candidate < dp[i, s]:
                    dp[i, s] = candidate
                    parent[i, s] = j

    # Find the optimal stream count.
    best_s = int(np.argmin(dp[n, 1 : s_max + 1])) + 1
    # Backtrack to recover split points.
    ends: list[int] = []
    i = n
    s = best_s
    while s > 0:
        ends.append(i)
        i = int(parent[i, s])
        s -= 1
    ends.reverse()
    return tuple(int(e) for e in ends)


# ---------------------------------------------------------------------------
# §3. Conv4 storage perms — exhaustive 4!-permutation brotli minimization
# ---------------------------------------------------------------------------


import itertools as _itertools


_CONV4_PERMS_CACHE: list[tuple[int, int, int, int]] = [
    p for p in _itertools.permutations(range(4))
]
"""All 24 4D-axis permutations. Hotz Round-1 simplification: itertools is
clearer than the nested-loop version. Output is identical."""


def derive_conv4_perms(
    state_dict: dict[str, torch.Tensor],
    *,
    brotli_quality: int = 11,
) -> dict[int, tuple[int, int, int, int]]:
    """Replace PR101's hardcoded ``CONV4_STORAGE_PERMS`` with a per-substrate
    dict derived via exhaustive 4! = 24 permutation search per conv4 tensor.

    Algorithm (§3 of the audit memo):

    For each tensor ``idx`` whose schema shape is 4D (Conv2d weights; the
    HNeRV decoder has 13 of them at indices 2,4,6,...,26 — even indices only):

    1. Quantize symmetric-INT8.
    2. For each of the 24 axis permutations, transpose the q_i8 array,
       flatten, zigzag-encode, append fp16 scale, brotli-compress.
    3. Pick the permutation that produces the smallest brotli output.

    Reference: Selfcomp PR #56 found HWOI permutation gave +5% xz savings
    on his SegMap weights vs the default OHWI; the same principle applies to
    brotli — the permutation that maximizes intra-byte autocorrelation
    minimizes 0-th order entropy, which brotli exploits via its context
    model.

    Args:
        state_dict: HNeRVDecoder state dict.
        brotli_quality: must match the encoder's quality (default 11).

    Returns:
        Dict keyed by tensor schema index (only conv4 indices appear).
        Compatible wire-format with PR101's ``CONV4_STORAGE_PERMS``.

    Determinism: tied permutations are broken by lexicographic order
    (smallest perm wins) for reproducibility.
    """
    _check_state_dict_complete(state_dict)
    quantized = [
        _quantize_tensor(name, state_dict[name])
        for name, _shape in FIXED_STATE_SCHEMA
    ]
    perms_dict: dict[int, tuple[int, int, int, int]] = {}
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        if len(shape) != 4:
            continue
        qt = quantized[idx]
        best_perm: tuple[int, int, int, int] | None = None
        best_bytes: int | None = None
        scale_bytes = np.array([qt.scale], dtype=np.float16).tobytes()
        # Zigzag encode under each permutation. We use the default 'zig'
        # byte_map here; the per-tensor byte_map decision is owned by
        # ``auto_select_byte_maps`` (already landed in commit 34e69f01).
        for perm in _CONV4_PERMS_CACHE:
            permuted = np.transpose(qt.q_i8, perm).copy()
            flat = permuted.reshape(-1)
            zz = np.where(
                flat.astype(np.int32) >= 0,
                2 * flat.astype(np.int32),
                -2 * flat.astype(np.int32) - 1,
            ).astype(np.uint8)
            payload = zz.tobytes() + scale_bytes
            n_bytes = len(pack_brotli_stream(payload, quality=brotli_quality))
            if best_bytes is None or n_bytes < best_bytes or (
                n_bytes == best_bytes and best_perm is not None and perm < best_perm
            ):
                best_bytes = n_bytes
                best_perm = perm
        assert best_perm is not None  # 24 perms always non-empty
        perms_dict[idx] = best_perm
    return perms_dict


# ---------------------------------------------------------------------------
# §4. Latent dim order — descending-variance permutation
# ---------------------------------------------------------------------------


def derive_latent_dim_order(latents: torch.Tensor) -> tuple[int, ...]:
    """Replace PR101's hardcoded ``LATENT_DIM_ORDER`` with a per-substrate
    permutation ordering latent dimensions by descending variance across
    the 600 frames.

    Algorithm (§4 of the audit memo):

    1. Compute per-dimension variance ``var_d = Var_n[latents[n, d]]`` over
       the 600 frame indices ``n`` for each of the 28 dimensions ``d``.
    2. Return ``argsort(-var_d)`` — descending variance.

    Reference: Ballé et al. (ICLR 2018) "Variational Image Compression with
    a Scale Hyperprior" — high-variance dimensions dominate the rate term
    of any near-Gaussian latent; LZMA's distance-matching heuristic benefits
    when high-energy dimensions are stored adjacent (more match
    opportunities, more aggressive RLE / LZ77 compression). HNeRV latents
    are nearly-Gaussian by construction.

    Args:
        latents: tensor of shape ``(N_PAIRS, LATENT_DIM)`` (600, 28). The
            function does not mutate this tensor.

    Returns:
        Length-``LATENT_DIM`` tuple permuting ``range(LATENT_DIM)`` in
        descending-variance order. Tied variances are broken by lower-dim
        first.

    Determinism: numpy's stable argsort + sign-trick guarantees reproducibility.
    """
    if latents.dim() != 2:
        raise Pr101SplitBrotliCodecError(
            f"derive_latent_dim_order requires 2D tensor, got dim={latents.dim()}"
        )
    if latents.shape[1] != LATENT_DIM:
        raise Pr101SplitBrotliCodecError(
            f"derive_latent_dim_order expected dim-1={LATENT_DIM}, got {latents.shape[1]}"
        )
    arr = latents.detach().cpu().float().numpy()
    var_d = arr.var(axis=0, ddof=0)
    # ``np.argsort`` is ascending; negate for descending. ``kind='stable'``
    # ensures tied variances break by lower-dim-first.
    order = np.argsort(-var_d, kind="stable")
    return tuple(int(x) for x in order)


# ---------------------------------------------------------------------------
# §5. Sidecar codebook — Lloyd-Max scalar quantization
# ---------------------------------------------------------------------------


def derive_sidecar_codebook(
    deltas: np.ndarray | Iterable[float],
    *,
    n_levels: int = 16,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Replace PR101's hardcoded ``SIDECAR_DELTAS_X100`` with a per-substrate
    16-value codebook derived via Lloyd-Max iterative scalar quantization on
    the actual delta distribution.

    Algorithm (§5 of the audit memo + Lloyd 1957 / Max 1960):

    1. Initialize centroids with PR101's default codebook
       (``[-10, -8, ..., 8, 10]`` with 0 excluded).
    2. Iterate until convergence:
       a. Cell boundaries = midpoints between consecutive centroids.
       b. Centroids = mean of input samples falling in each cell.
    3. Return centroids quantized to ``int8`` (matches PR101's wire format).

    The PR101 default skips 0 (16 levels for non-zero deltas). We preserve
    that semantic: input ``deltas`` should be non-zero values; the deriver
    operates on them as-is.

    Reference: Lloyd "Least Squares Quantization in PCM" (IEEE TIT 1982
    reprint of 1957 work); Max "Quantizing for Minimum Distortion" (IRE TIT
    1960). For a known input distribution, the optimal N-level codebook
    minimizes mean-squared error via fixed-point iteration on cell
    boundaries and centroids.

    Args:
        deltas: 1D array (or iterable) of integer-valued delta samples
            (×100 scale, matching PR101's wire format). Typical input:
            the per-frame delta values from a training-set pass over the
            sidecar correction stream.
        n_levels: codebook size. PR101 uses 16; this is the wire-format
            constraint. Other values produce non-PR101-compatible codebooks.
        max_iters: Lloyd iteration cap.
        tol: convergence threshold on max centroid movement.

    Returns:
        Length-``n_levels`` ``np.ndarray[np.int8]`` codebook in ascending
        order. Matches the dtype/shape of ``SIDECAR_DELTAS_X100``.

    Edge cases:

    * If ``deltas`` is empty, returns the PR101 default unchanged.
    * If a Lloyd cell is empty during iteration, that centroid keeps its
      previous value (avoids div-by-zero; standard fix).
    * Tied samples between cells are assigned to the lower cell
      (deterministic).
    """
    deltas_arr = np.asarray(list(deltas), dtype=np.float64)
    if deltas_arr.size == 0 or n_levels < 2:
        # Empty / degenerate — fall back to PR101 default, sliced/padded.
        if n_levels == len(SIDECAR_DELTAS_X100_PR101_DEFAULT):
            return SIDECAR_DELTAS_X100_PR101_DEFAULT.copy()
        # Generic linspace fallback — preserves int8 dtype.
        return np.linspace(-10, 10, n_levels).round().astype(np.int8)

    # Initialize centroids with PR101's default if size matches; else linspace.
    if n_levels == len(SIDECAR_DELTAS_X100_PR101_DEFAULT):
        centroids = SIDECAR_DELTAS_X100_PR101_DEFAULT.astype(np.float64).copy()
    else:
        lo, hi = float(deltas_arr.min()), float(deltas_arr.max())
        centroids = np.linspace(lo, hi, n_levels)

    for _ in range(max_iters):
        # Step 1: cell boundaries = midpoints of adjacent centroids.
        sorted_centroids = np.sort(centroids)
        boundaries = (sorted_centroids[:-1] + sorted_centroids[1:]) / 2.0
        # Step 2: assign each sample to its nearest cell (via boundaries).
        # ``np.searchsorted`` returns indices in [0, n_levels].
        cell_idx = np.searchsorted(boundaries, deltas_arr)
        # Recompute centroids = within-cell means; preserve previous if empty.
        new_centroids = sorted_centroids.copy()
        for k in range(n_levels):
            mask = cell_idx == k
            if mask.any():
                new_centroids[k] = float(deltas_arr[mask].mean())
        movement = float(np.max(np.abs(new_centroids - sorted_centroids)))
        centroids = new_centroids
        if movement < tol:
            break

    # Round to int8 to match PR101's wire format. Sort ascending for
    # deterministic ordering. Clip to int8 range (-128, 127); PR101's range is
    # tight (-10..10) so clipping is a no-op for typical inputs.
    out = np.clip(np.round(np.sort(centroids)), -128, 127).astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Convenience: derive ALL constants in one call (used by encoder's
# `auto_derive_all=True` flag and by the demo CLI).
# ---------------------------------------------------------------------------


def derive_all(
    state_dict: dict[str, torch.Tensor],
    latents: torch.Tensor | None = None,
    sidecar_deltas: np.ndarray | None = None,
    *,
    brotli_quality: int = 11,
) -> dict[str, object]:
    """Convenience wrapper running all 5 derivers on the supplied substrate.

    Returns a dict with keys:

    * ``storage_order`` — output of :func:`derive_storage_order`
    * ``stream_ends`` — output of :func:`derive_stream_ends`
    * ``conv4_perms`` — output of :func:`derive_conv4_perms`
    * ``latent_dim_order`` — output of :func:`derive_latent_dim_order` (if
      ``latents`` is given), else PR101 default tuple
    * ``sidecar_codebook`` — output of :func:`derive_sidecar_codebook` (if
      ``sidecar_deltas`` is given), else PR101 default array
    """
    storage_order = derive_storage_order(state_dict)
    stream_ends = derive_stream_ends(
        state_dict, storage_order, brotli_quality=brotli_quality
    )
    conv4_perms = derive_conv4_perms(state_dict, brotli_quality=brotli_quality)
    if latents is not None:
        lat_order: tuple[int, ...] = derive_latent_dim_order(latents)
    else:
        lat_order = tuple(range(LATENT_DIM))
    if sidecar_deltas is not None:
        codebook = derive_sidecar_codebook(sidecar_deltas)
    else:
        codebook = SIDECAR_DELTAS_X100_PR101_DEFAULT.copy()
    return {
        "storage_order": storage_order,
        "stream_ends": stream_ends,
        "conv4_perms": conv4_perms,
        "latent_dim_order": lat_order,
        "sidecar_codebook": codebook,
    }


__all__ = [
    "SIDECAR_DELTAS_X100_PR101_DEFAULT",
    "derive_all",
    "derive_conv4_perms",
    "derive_latent_dim_order",
    "derive_sidecar_codebook",
    "derive_storage_order",
    "derive_stream_ends",
]
