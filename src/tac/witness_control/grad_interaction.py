"""Per-term gradient INTERACTION telemetry (task #312 Phase A).

The synergy/antagonism matrix, live: at stage/octave boundaries (and every N verdict epochs)
the trainer computes, over a small K-pair sample, the per-loss-term gradient w.r.t. the
witness parameters, then this module builds the term×term cosine-similarity matrix, the
grad-norm (dominance) shares, and the conflict pairs (cos < threshold). Two loss terms whose
gradients point the same way are SYNERGISTIC; opposed gradients are ANTAGONISTIC (the pair is
fighting the optimizer). This is the interaction layer on top of the existing per-term loss
STATE decomposition (``LOSS_TERM_KEYS``).

SAFETY (binding — this is a MEASUREMENT pass, never a training update):
  * the per-term backward passes run under a SEPARATE ``value_and_grad`` on a COPY of the
    forward; the optimizer / EMA are never touched (the caller passes grad-only closures);
  * #312 GradNorm discipline: this module only OBSERVES gradient balance — it emits
    recommendations at stage boundaries, it NEVER rescales a live per-step gradient
    (per-step gradient balancing muted a canary once);
  * the trainer wraps the call in an RNG snapshot/restore + a byte-identity assertion so the
    seeded stream is provably unperturbed (``rng_fingerprint`` + ``assert_rng_unperturbed``).

Core math (flatten / cosine / shares / row) is pure-numpy and unit-tested; MLX helpers
(tree flatten, RNG snapshot/restore) are thin adapters. Every row is score-neutral
observability; pointer 0.19110 UNMOVED.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

DEFAULT_CONFLICT_THRESHOLD = -0.2  # cos below this = antagonistic pair (task #312 tag)
AXIS_TAG = "[macOS advisory] NON-PROMOTABLE"


# ─────────────────────────── pure-numpy interaction math ─────────────────────
def _unit(v: np.ndarray) -> tuple[np.ndarray, float]:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.sqrt(np.dot(v, v)))
    if n <= 1e-30:
        return v, 0.0
    return v / n, n


def cosine_matrix(vectors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray, dict[str, float]]:
    """Return ``(names, cos_matrix, norms)`` for the per-term gradient vectors.

    ``cos_matrix[i,j]`` = cosine similarity of term i's and term j's gradient (diagonal 1.0,
    or 0.0 for a zero-gradient term). Names are the input keys in insertion order. ``norms``
    maps name -> L2 gradient norm (before normalization) for the dominance shares."""
    names = list(vectors)
    units: list[np.ndarray] = []
    norms: dict[str, float] = {}
    ref_len: int | None = None
    for name in names:
        u, n = _unit(vectors[name])
        if ref_len is None:
            ref_len = u.size
        elif u.size != ref_len:
            raise ValueError(f"gradient vector length mismatch for {name!r}: "
                             f"{u.size} != {ref_len}")
        units.append(u)
        norms[name] = n
    k = len(names)
    mat = np.eye(k, dtype=np.float64)
    for i in range(k):
        if norms[names[i]] == 0.0:
            mat[i, i] = 0.0
        for j in range(i + 1, k):
            c = float(np.clip(np.dot(units[i], units[j]), -1.0, 1.0))
            mat[i, j] = mat[j, i] = c
    return names, mat, norms


def upper_triangle(names: list[str], mat: np.ndarray) -> list[dict]:
    """Flatten the strict upper triangle to ``[{pair:[a,b], cos: c}, ...]``."""
    out: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out.append({"pair": [names[i], names[j]], "cos": round(float(mat[i, j]), 4)})
    return out


def conflict_pairs(names: list[str], mat: np.ndarray,
                   threshold: float = DEFAULT_CONFLICT_THRESHOLD) -> list[dict]:
    """Antagonistic pairs: cos < threshold (default −0.2), most-negative first."""
    pairs = [p for p in upper_triangle(names, mat) if p["cos"] < threshold]
    pairs.sort(key=lambda p: p["cos"])
    return pairs


def dominance_shares(norms: dict[str, float]) -> dict[str, float]:
    """Per-term gradient-norm share = ||g_i|| / Σ||g_j|| (the dominance fraction). Sums to 1.0
    (or {} when every gradient is zero)."""
    total = float(sum(norms.values()))
    if total <= 1e-30:
        return {k: 0.0 for k in norms}
    return {k: round(v / total, 6) for k, v in norms.items()}


def grad_interaction_row(vectors: dict[str, np.ndarray], *, stage: str, ep: int,
                         conflict_threshold: float = DEFAULT_CONFLICT_THRESHOLD,
                         k_pairs: int, cadence: str = "boundary",
                         emit_reason: str = "") -> dict:
    """Build the canonical ``{stage: "grad_interactions"}`` telemetry row: the term×term
    cosine upper triangle, conflict pairs, dominance shares, and the dominant term. Pure /
    MLX-free / unit-tested. ``k_pairs`` records the sample size; ``cadence`` tags why it fired
    (``boundary`` / ``periodic``). Score-neutral observability — no ΔS, no actuation."""
    # keep only terms that HAVE a gradient this emission (drop inactive levers so the matrix is
    # about the live optimization, not padded zeros)
    active = {k: v for k, v in vectors.items()
             if float(np.sqrt(np.dot(np.asarray(v, np.float64).ravel(),
                                     np.asarray(v, np.float64).ravel()))) > 1e-30}
    names, mat, norms = cosine_matrix(active) if active else ([], np.zeros((0, 0)), {})
    shares = dominance_shares(norms)
    dom = max(shares, key=shares.get) if shares else None
    confs = conflict_pairs(names, mat, conflict_threshold)
    row: dict[str, object] = {
        "stage": "grad_interactions", "ep": int(ep), "seg_stage": str(stage),
        "cadence": str(cadence), "k_pairs": int(k_pairs), "n_active_terms": len(names),
        "terms": names,
        "cosine_upper_triangle": upper_triangle(names, mat),
        "conflict_pairs": confs, "conflict_threshold": float(conflict_threshold),
        "n_conflicts": len(confs),
        "dominance_shares": shares, "dominant_term": dom,
        "axis": AXIS_TAG, "score_neutral": True,
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if emit_reason:
        row["emit_reason"] = str(emit_reason)
    return row


# ─────────────────────────── MLX adapters (thin) ────────────────────────────
def flatten_grad_tree(tree) -> np.ndarray:
    """Flatten an MLX/nested param-gradient pytree into ONE 1-D float64 numpy vector in a
    DETERMINISTIC key order (sorted). Handles dict / list / tuple / mx.array / np.ndarray /
    scalars recursively. A ``None`` leaf (a param with no gradient) contributes nothing.
    The deterministic order is what makes two flattened trees dot-comparable."""
    parts: list[np.ndarray] = []

    def _walk(node) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            for kk in sorted(node):
                _walk(node[kk])
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)
            return
        arr = np.asarray(node, dtype=np.float64)  # mx.array -> np via __array__
        parts.append(arr.ravel())

    _walk(tree)
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def rng_fingerprint() -> dict:
    """Capture the numpy + MLX global RNG state as a comparable fingerprint. Used to PROVE a
    measurement pass left the seeded stream untouched (the score-neutrality guard)."""
    fp: dict = {}
    st = np.random.get_state()
    # position + a hash of the key buffer (the whole tuple is large; pos+key-hash is exact enough)
    fp["np_pos"] = int(st[2])
    fp["np_key_hash"] = int(np.asarray(st[1], dtype=np.uint64).sum(dtype=np.uint64)) & ((1 << 63) - 1)
    try:
        import mlx.core as mx
        fp["mx_state"] = [np.asarray(a).tolist() for a in mx.random.state]
    except Exception:
        fp["mx_state"] = None
    return fp


def assert_rng_unperturbed(before: dict, after: dict, *, where: str = "grad_interaction") -> None:
    """Raise AssertionError if the RNG fingerprint changed across a measurement pass — the
    binding byte-identity guard: a telemetry pass that advances the seeded stream would make
    the run non-reproducible, so it fails CLOSED."""
    if before != after:
        raise AssertionError(
            f"[{where}] RNG STREAM PERTURBED by the measurement pass — "
            f"before={before} after={after}. A telemetry pass MUST NOT touch the training RNG "
            f"(snapshot/restore or fork the key).")


class MxRngGuard:
    """Context manager for a score-neutral measurement pass.

    * numpy stream: snapshotted on enter and RESTORED on exit (``set_state`` round-trips
      exactly), so a measurement that samples pairs via ``np.random`` is rolled back.
    * MLX global stream: MLX's default RNG has internal counter state NOT exposed by
      ``mx.random.state`` (verified: state assignment does not reproduce the next draw), so it
      cannot be restored. The binding contract is therefore that the measurement must NOT draw
      from the GLOBAL mx stream — it must use an EXPLICIT forked key (``mx.random.split``) or
      only deterministic ops (the witness render/loss is deterministic: no dropout, no implicit
      mx.random). On a clean exit the guard ASSERTS the global mx state is unchanged and FAILS
      CLOSED otherwise (a measurement that touched the global stream would make the run
      non-reproducible — that is a bug to surface, not silently mis-restore).
    """

    def __init__(self, where: str = "grad_interaction") -> None:
        self.where = where
        self._np = None
        self._fp_before: dict | None = None

    def __enter__(self) -> "MxRngGuard":
        self._np = np.random.get_state()
        self._fp_before = rng_fingerprint()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._np is not None:
            np.random.set_state(self._np)  # numpy stream rolled back exactly
        # only assert clean round-trip when the block itself did not raise
        if exc_type is None:
            assert_rng_unperturbed(self._fp_before, rng_fingerprint(), where=self.where)
