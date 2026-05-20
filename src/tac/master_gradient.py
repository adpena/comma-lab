# SPDX-License-Identifier: MIT
"""Canonical master-gradient helper — diagnostic score-response tensor at a measured operating point.

The *master gradient*[^master-gradient] is an internal nickname for a diagnostic
score-response tensor at a specific operating point. It decomposes the additive
scorer `S = 100*d_seg + sqrt(10*d_pose) + 25*R` across a parser-known payload
domain so grammar-aware interventions can be prioritized for real packet-valid
probes.

Authority boundary: this module is not a raw archive-byte optimizer. Promotion
and rank/kill authority must route through `CandidateModificationSpec` rows with
`grammar_aware_operator` coordinates and packet proofs; see
`tac.master_gradient_operator_plan`.

[^master-gradient]: Internal nickname; not a published term. Symposium memo
    .omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md §3.

Eight in-training / in-design uses per symposium §3.6:
1. Score-aware loss term at byte-grain     -> tac.losses.master_gradient_term (TODO)
2. Per-pixel/per-byte attention reweighting -> tac.losses.u_die_kl consumer (TODO)
3. Bit allocator hook (Catalog #125 hook 3) -> tac.optimization.bit_allocator (TODO)
4. Architecture search discriminator        -> autopilot composition matrix (TODO)
5. Score-aware QAT FP4 codebook             -> tac.quantization.lsq_step_size (TODO)
6. Pareto facets feed Dykstra (Catalog #296) -> tac.optimization.dykstra_feasibility (TODO)
7. Continual-learning posterior for autopilot -> tac.autopilot_rudin_daubechies.master_gradient_lens (THIS PR)
8. Magic codec[^magic-codec] per-stream selection -> tac.packet_compiler.score_aware_selector (TODO)

[^magic-codec]: Internal nickname for the per-stream optimal-entropy-coder
    auto-selector in tac.packet_compiler.* — see docs/pr_writeups/cpu_frontier_fec6_20260517.md Glossary §12.1.

The canonical 4-layer pattern mirrors Catalog #245 (Modal call_id ledger):
- Layer 1 (this module): MasterGradient dataclass + fcntl-locked ledger
- Layer 2: tools/extract_master_gradient.py CLI (TODO follow-on)
- Layer 3: STRICT preflight gate refusing stale gradient citations (TODO follow-on)
- Layer 4: autopilot rerank wire-in via tac.autopilot_rudin_daubechies.master_gradient_lens

The tensor is OPERATING-POINT-LOCAL and often measured on a sample subset. Such
rows are diagnostic unless the ledger records full charged-archive custody,
sample count, hardware substrate, and an authoritative axis label.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import socket
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy unavailable in some minimal envs
    np = None  # type: ignore[assignment]


__all__ = [
    "AGGREGATE_GRADIENT_TENSOR_KIND",
    "CONTEST_RATE_DENOM_BYTES",
    "LEGAL_GRADIENT_TENSOR_KINDS",
    "MASTER_GRADIENT_LEDGER_PATH",
    "PER_PAIR_GRADIENT_TENSOR_KIND",
    "MasterGradient",
    "MasterGradientLedgerCorruptError",
    "OperatingPoint",
    "append_anchor_locked",
    "append_score_axis_dominance_backfill",
    "build_score_axis_dominance_backfill_anchor",
    "compute_marginal_coefficients",
    "contest_axis_authority_violation_reason",
    "effective_anchor_sort_key",
    "is_authoritative_axis_anchor",
    "is_authoritative_contest_axis_anchor",
    "is_usable_planning_anchor",
    "latest_anchor_for_archive",
    "latest_rejected_contest_axis_anchor_for_archive",
    "load_anchors_lenient",
    "load_anchors_strict",
    "predict_delta_s",
    "predict_delta_s_per_pair",
    "query_anchors_by_archive",
    "score_axis_dominance_summary",
    "unresolved_contest_axis_authority_violations",
    "update_from_anchor",
]

# Per upstream/evaluate.py:92
CONTEST_RATE_DENOM_BYTES: int = 37_545_489

MASTER_GRADIENT_LEDGER_PATH = Path(".omx/state/master_gradient_anchors.jsonl")
_LEDGER_LOCK_PATH = Path(".omx/state/.master_gradient.lock")

AGGREGATE_GRADIENT_TENSOR_KIND = "aggregate_per_byte_v1"
PER_PAIR_GRADIENT_TENSOR_KIND = "per_pair_per_byte_v1"
LEGAL_GRADIENT_TENSOR_KINDS = frozenset(
    {AGGREGATE_GRADIENT_TENSOR_KIND, PER_PAIR_GRADIENT_TENSOR_KIND}
)
AUTHORITATIVE_CONTEST_AXES = frozenset({"[contest-CPU]", "[contest-CUDA]"})
SCORE_AXIS_LABELS = ("seg", "pose", "rate")


def _is_contest_axis_label(axis: object) -> bool:
    text = str(axis or "").strip().lower().replace("-", "_")
    return text in {
        "[contest_cpu]",
        "contest_cpu",
        "[contest_cuda]",
        "contest_cuda",
    }


def _normalized_contest_axis(axis: object) -> str | None:
    text = str(axis or "").strip().lower().replace("-", "_")
    if text in {"[contest_cpu]", "contest_cpu"}:
        return "[contest-CPU]"
    if text in {"[contest_cuda]", "contest_cuda"}:
        return "[contest-CUDA]"
    return None


class MasterGradientLedgerCorruptError(RuntimeError):
    """Raised when load_anchors_strict fails to parse a row [verified-against: Catalog #138]."""


@dataclass(frozen=True)
class OperatingPoint:
    """The (d_seg, d_pose, R, score) tuple at which a gradient was measured."""

    d_seg: float
    d_pose: float
    rate: float
    score: float

    def __post_init__(self) -> None:
        if not (self.d_seg >= 0 and self.d_pose > 0 and self.rate >= 0):
            raise ValueError(
                f"operating point invalid: d_seg={self.d_seg}, d_pose={self.d_pose}, R={self.rate}; "
                "d_pose must be > 0 (master-gradient PoseNet marginal 5/sqrt(10*d_pose) is undefined at 0)"
            )

    def as_dict(self) -> dict[str, float]:
        return {"d_seg": self.d_seg, "d_pose": self.d_pose, "rate": self.rate, "score": self.score}


def compute_marginal_coefficients(op: OperatingPoint) -> tuple[float, float, float]:
    """Return (∂S/∂d_seg, ∂S/∂d_pose, ∂S/∂byte) at the operating point.

    Derivation (per symposium §1; corrected from the original 922 to 5/sqrt(10*d_pose) per op-routable #7):
        S = 100 * d_seg + sqrt(10 * d_pose) + 25 * R
        ∂S/∂d_seg  = 100  (constant)
        ∂S/∂d_pose = 5 / sqrt(10 * d_pose)  (hyperbolic; diverges as d_pose -> 0)
        ∂S/∂R      = 25 (per unit rate; equivalently 25 / CONTEST_RATE_DENOM_BYTES per byte)
    """
    seg_marginal = 100.0
    pose_marginal = 5.0 / math.sqrt(10.0 * op.d_pose)
    rate_marginal_per_byte = 25.0 / CONTEST_RATE_DENOM_BYTES
    return seg_marginal, pose_marginal, rate_marginal_per_byte


def score_axis_dominance_summary(
    gradient_tensor,
    operating_point: OperatingPoint,
    *,
    axis_dominance_threshold: float = 0.7,
    max_chunk_entries: int = 1_000_000,
) -> dict[str, object]:
    """Summarize which score axis dominates a master-gradient tensor.

    This is producer-side metadata for downstream planning tools. It records
    score-axis dominance derived from the gradient tensor and the contest-score
    marginal coefficients, without granting raw byte-mutation authority.
    Large per-pair tensors are processed in chunks to avoid allocating an
    additional full-size fp64 tensor during extraction.
    """
    if np is None:
        raise RuntimeError("numpy required to summarize master-gradient dominance")
    if not (0.0 <= axis_dominance_threshold <= 1.0):
        raise ValueError("axis_dominance_threshold must be in [0, 1]")
    if max_chunk_entries <= 0:
        raise ValueError("max_chunk_entries must be > 0")

    arr = np.asarray(gradient_tensor)
    if arr.ndim not in (2, 3) or arr.shape[-1] != 3:
        raise ValueError(
            "gradient_tensor must have shape (N_bytes, 3) or "
            f"(N_bytes, N_pairs, 3); got {arr.shape}"
        )

    coeffs = np.asarray(compute_marginal_coefficients(operating_point), dtype=np.float64)
    entries_per_byte = int(arr.shape[1]) if arr.ndim == 3 else 1
    bytes_per_chunk = max(1, int(max_chunk_entries) // max(1, entries_per_byte))
    entry_count = int(np.prod(arr.shape[:-1], dtype=np.int64))
    zero_contribution_count = 0
    dominant_counts_arr = np.zeros(3, dtype=np.int64)
    threshold_counts_arr = np.zeros(3, dtype=np.int64)
    total_abs_score_contribution_l1_arr = np.zeros(3, dtype=np.float64)
    sum_axis_share_arr = np.zeros(3, dtype=np.float64)

    for start in range(0, int(arr.shape[0]), bytes_per_chunk):
        chunk = np.asarray(arr[start : start + bytes_per_chunk], dtype=np.float64)
        flat = (np.abs(chunk) * coeffs).reshape(-1, 3)
        denom = flat.sum(axis=1)
        shares = np.zeros_like(flat)
        np.divide(flat, denom[:, None], out=shares, where=denom[:, None] > 0)
        nonzero = denom > 0
        zero_contribution_count += int((~nonzero).sum())
        dominant = shares.argmax(axis=1)
        for axis_idx in range(3):
            dominant_counts_arr[axis_idx] += int(((dominant == axis_idx) & nonzero).sum())
            threshold_counts_arr[axis_idx] += int(
                ((shares[:, axis_idx] >= axis_dominance_threshold) & nonzero).sum()
            )
            total_abs_score_contribution_l1_arr[axis_idx] += float(flat[:, axis_idx].sum())
            sum_axis_share_arr[axis_idx] += float(shares[:, axis_idx].sum())

    dominant_counts = {
        label: int(dominant_counts_arr[axis_idx])
        for axis_idx, label in enumerate(SCORE_AXIS_LABELS)
    }
    threshold_counts = {
        label: int(threshold_counts_arr[axis_idx])
        for axis_idx, label in enumerate(SCORE_AXIS_LABELS)
    }
    total_abs_score_contribution_l1 = {
        label: float(total_abs_score_contribution_l1_arr[axis_idx])
        for axis_idx, label in enumerate(SCORE_AXIS_LABELS)
    }
    mean_axis_share = {
        label: float(sum_axis_share_arr[axis_idx] / entry_count) if entry_count else 0.0
        for axis_idx, label in enumerate(SCORE_AXIS_LABELS)
    }

    payload: dict[str, object] = {
        "schema": "master_gradient_score_axis_dominance_v1",
        "formula": (
            "abs(gradient_axis) * contest_score_marginal / "
            "sum_axes(abs(gradient_axis) * contest_score_marginal)"
        ),
        "axis_labels": list(SCORE_AXIS_LABELS),
        "marginal_coefficients": {
            label: float(coeffs[axis_idx])
            for axis_idx, label in enumerate(SCORE_AXIS_LABELS)
        },
        "axis_dominance_threshold": float(axis_dominance_threshold),
        "tensor_shape": [int(dim) for dim in arr.shape],
        "entry_count": entry_count,
        "zero_contribution_count": int(zero_contribution_count),
        "dominant_axis_counts": dominant_counts,
        "threshold_dominant_axis_counts": threshold_counts,
        "mean_axis_share": mean_axis_share,
        "total_abs_score_contribution_l1": total_abs_score_contribution_l1,
        "promotion_authority": False,
        "raw_archive_byte_authority": False,
        "authority_boundary": (
            "diagnostic score-axis dominance metadata; packet-valid mutation "
            "still requires grammar-aware CandidateModificationSpec rows, "
            "packet proofs, and exact eval"
        ),
    }
    if arr.ndim == 2:
        payload["n_bytes"] = int(arr.shape[0])
        payload["pose_axis_dominant_byte_count"] = int(threshold_counts["pose"])
    else:
        payload["n_bytes"] = int(arr.shape[0])
        payload["n_pairs"] = int(arr.shape[1])
        payload["pose_axis_dominant_pair_entry_count"] = int(threshold_counts["pose"])
    return payload


@dataclass(frozen=True)
class MasterGradient:
    """Diagnostic score-response tensor measured at a specific operating point.

    The default aggregate gradient tensor is stored as a sidecar .npy file
    (shape (N_bytes, 3), dtype float32, columns [seg, pose, rate_bytes_delta]).
    `N_bytes` is the gradient subject byte domain, not necessarily the charged
    contest ZIP size. For one-member ZIP submissions this is commonly the inner
    payload stream; charged ZIP custody is recorded separately.

    Per-pair extraction uses a sister tensor kind with shape
    (N_bytes, N_pairs, 3). It must stay explicitly typed; silently treating a
    per-pair tensor as an aggregate gradient would erase exactly the pair-wise
    cancellation signal this artifact exists to preserve.

    See symposium §3 + §3.6 for the eight in-training / in-design uses.
    """

    archive_sha256: str
    operating_point: OperatingPoint
    gradient_array_path: str
    n_bytes: int
    measurement_method: str  # "autograd_per_parameter_projected" | "finite-difference bit flip" | ...
    measurement_axis: str  # "[contest-CPU]" / "[contest-CUDA]"
    measurement_hardware: str  # e.g. "linux_x86_64_modal_cpu"
    measurement_call_id: str | None
    measurement_utc: str
    pareto_facets: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    rashomon_disagreement_score: float | None = None
    gradient_tensor_kind: str = AGGREGATE_GRADIENT_TENSOR_KIND
    n_pairs: int | None = None
    scored_archive_sha256: str | None = None
    scored_archive_bytes: int | None = None
    gradient_subject_sha256: str | None = None
    gradient_subject_bytes: int | None = None
    gradient_byte_domain: str = "scored_archive_bytes"
    n_pairs_used: int | None = None
    n_pairs_total: int | None = None
    score_axis_dominance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.archive_sha256, str) or len(self.archive_sha256) < 16:
            raise ValueError("archive_sha256 must be a hex sha256 (>=16 chars)")
        if self.n_bytes <= 0:
            raise ValueError("n_bytes must be > 0")
        if not self.measurement_axis.startswith("["):
            raise ValueError(
                f"measurement_axis {self.measurement_axis!r} must be lane-tagged per CLAUDE.md "
                "'Apples-to-apples evidence discipline' (e.g. '[contest-CPU]', '[contest-CUDA]')"
            )
        if self.gradient_tensor_kind not in LEGAL_GRADIENT_TENSOR_KINDS:
            raise ValueError(
                f"gradient_tensor_kind={self.gradient_tensor_kind!r} not in "
                f"{sorted(LEGAL_GRADIENT_TENSOR_KINDS)!r}"
            )
        if self.gradient_tensor_kind == AGGREGATE_GRADIENT_TENSOR_KIND:
            if self.n_pairs is not None:
                raise ValueError(
                    "aggregate master-gradient anchors must not set n_pairs; "
                    "use gradient_tensor_kind='per_pair_per_byte_v1' for "
                    "(N_bytes, N_pairs, 3) tensors"
                )
        elif self.gradient_tensor_kind == PER_PAIR_GRADIENT_TENSOR_KIND and (
            isinstance(self.n_pairs, bool)
            or not isinstance(self.n_pairs, int)
            or self.n_pairs <= 0
        ):
            raise ValueError(
                "per-pair master-gradient anchors must set positive integer n_pairs"
            )
        for field_name, value in (
            ("scored_archive_sha256", self.scored_archive_sha256),
            ("gradient_subject_sha256", self.gradient_subject_sha256),
        ):
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdefABCDEF" for ch in value)
            ):
                raise ValueError(f"{field_name} must be a 64-char hex sha256 when set")
        for field_name, value in (
            ("scored_archive_bytes", self.scored_archive_bytes),
            ("gradient_subject_bytes", self.gradient_subject_bytes),
            ("n_pairs_used", self.n_pairs_used),
            ("n_pairs_total", self.n_pairs_total),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer when set")
        if (
            self.n_pairs_used is not None
            and self.n_pairs_total is not None
            and self.n_pairs_used > self.n_pairs_total
        ):
            raise ValueError("n_pairs_used cannot exceed n_pairs_total")
        if self.gradient_subject_bytes is not None and self.gradient_subject_bytes != self.n_bytes:
            raise ValueError("gradient_subject_bytes must equal n_bytes for the sidecar tensor")
        if self.score_axis_dominance is not None and not isinstance(
            self.score_axis_dominance, Mapping
        ):
            raise ValueError("score_axis_dominance must be a mapping when set")

    def load_gradient(self):
        """Load the (N_bytes, 3) float32 array; requires numpy."""
        if np is None:
            raise RuntimeError("numpy required to load master-gradient arrays")
        if self.gradient_tensor_kind != AGGREGATE_GRADIENT_TENSOR_KIND:
            raise ValueError(
                f"gradient_tensor_kind={self.gradient_tensor_kind!r} is not an "
                "aggregate (N_bytes, 3) tensor; use load_per_pair_gradient() "
                "for per-pair anchors"
            )
        arr = np.load(self.gradient_array_path)
        if arr.shape != (self.n_bytes, 3):
            raise ValueError(
                f"loaded array shape {arr.shape} != declared ({self.n_bytes}, 3)"
            )
        return arr

    def load_per_pair_gradient(self):
        """Load the (N_bytes, N_pairs, 3) per-pair array; requires numpy."""
        if np is None:
            raise RuntimeError("numpy required to load master-gradient arrays")
        if self.gradient_tensor_kind != PER_PAIR_GRADIENT_TENSOR_KIND:
            raise ValueError(
                f"gradient_tensor_kind={self.gradient_tensor_kind!r} is not a "
                "per-pair (N_bytes, N_pairs, 3) tensor; use load_gradient() "
                "for aggregate anchors"
            )
        arr = np.load(self.gradient_array_path)
        expected = (self.n_bytes, self.n_pairs, 3)
        if arr.shape != expected:
            raise ValueError(f"loaded array shape {arr.shape} != declared {expected}")
        return arr

    def coefficients(self) -> tuple[float, float, float]:
        return compute_marginal_coefficients(self.operating_point)

    def predict_delta_s(self, byte_modifications: Mapping[int, float]) -> float:
        """Diagnostic ΔS for a local byte-value perturbation.

        This helper is intentionally diagnostic: `byte_modifications` uses
        sidecar-domain byte coordinates and is not a packet-valid archive
        mutation. Score-lowering candidates must be materialized as
        `CandidateModificationSpec` / `grammar_aware_operator` response rows
        that rebuild ZIP metadata and measure `rate_bytes_delta` explicitly.
        """
        arr = self.load_gradient()
        coeffs = self.coefficients()
        # coeffs = (seg_marginal, pose_marginal, rate_per_byte)
        # G[i, :] = (∂d_seg/∂byte_i, ∂d_pose/∂byte_i, ∂R/∂byte_i)
        # predicted ΔS = sum_i delta_i * (G[i, 0]*seg_marginal + G[i, 1]*pose_marginal + G[i, 2]*rate_per_byte)
        total = 0.0
        for byte_idx, delta in byte_modifications.items():
            if not (0 <= byte_idx < self.n_bytes):
                raise ValueError(f"byte_idx {byte_idx} out of range [0, {self.n_bytes})")
            g = arr[byte_idx]
            total += float(delta) * (
                g[0] * coeffs[0] + g[1] * coeffs[1] + g[2] * coeffs[2]
            )
        return total

    def predict_delta_s_per_pair(self, byte_modifications: Mapping[int, float]):
        """Predicted ΔS vector for each pair using a per-pair gradient tensor.

        Returns an ndarray of shape (N_pairs,). This intentionally does not
        collapse to a scalar; callers that want the averaged scalar must make
        that aggregation explicit so sign-cancellation cannot be hidden.
        """
        arr = self.load_per_pair_gradient()
        coeffs = self.coefficients()
        total = np.zeros((self.n_pairs,), dtype=np.float64)
        for byte_idx, delta in byte_modifications.items():
            if not (0 <= byte_idx < self.n_bytes):
                raise ValueError(f"byte_idx {byte_idx} out of range [0, {self.n_bytes})")
            g = arr[byte_idx]
            total += float(delta) * (
                g[:, 0] * coeffs[0] + g[:, 1] * coeffs[1] + g[:, 2] * coeffs[2]
            )
        return total


def predict_delta_s(
    gradient: MasterGradient, byte_modifications: Mapping[int, float]
) -> float:
    """Convenience top-level alias for MasterGradient.predict_delta_s."""
    return gradient.predict_delta_s(byte_modifications)


def predict_delta_s_per_pair(
    gradient: MasterGradient, byte_modifications: Mapping[int, float]
):
    """Convenience alias for MasterGradient.predict_delta_s_per_pair."""
    return gradient.predict_delta_s_per_pair(byte_modifications)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effective_anchor_sort_key(row: Mapping[str, object]) -> tuple[str, str]:
    """Sort key for append-only anchor corrections.

    Measurement time identifies the empirical tensor. ``written_at_utc`` breaks
    ties so later append-only metadata corrections for the same tensor become
    the effective row without falsifying the original measurement timestamp.
    """

    return (
        str(row.get("measurement_utc") or ""),
        str(row.get("written_at_utc") or ""),
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_append(target: Path, line: str) -> None:
    """Atomic append via tmp+rename per Catalog #245 sister discipline."""
    _ensure_parent(target)
    existing = target.read_bytes() if target.exists() else b""
    payload = existing + line.encode() + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(target.parent),
        prefix=target.name,
        suffix=f".tmp.{uuid.uuid4().hex[:12]}",
        delete=False,
    ) as f:
        f.write(payload)
        tmp = Path(f.name)
    os.replace(tmp, target)


def append_anchor_locked(
    gradient: MasterGradient,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """fcntl-locked append per Catalog #128 / #131 / #245.

    After the atomic append succeeds, fires the canonical cathedral-consumer
    post-anchor hook (sister of Catalog #343
    :func:`tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`
    pattern). Auto-discovers consumers under ``src/tac/cathedral_consumers/``
    that opt-in via module-level ``CONSUMES_MASTER_GRADIENT_ANCHORS = True``
    and calls each consumer's ``update_from_anchor(anchor_row)`` with the
    just-appended JSONL row (parsed dict).

    Per CLAUDE.md "Subagent coherence-by-default" maximum-signal-preservation
    + the Catalog #343 sister contract: per-consumer exceptions are caught
    + warning-logged so the ledger write (which already succeeded) is not
    blocked by a buggy downstream consumer. The ledger is canonical state;
    the consumer hook is an observability-only downstream signal.

    Closes the orphan-signal gap from commit ``7b9d5e280``
    (PER-BYTE-METHODOLOGY-FOLLOWUP) which landed the structural consumer
    package but explicitly deferred this runtime wire-in per Catalog
    #110/#113 APPEND-ONLY discipline.
    """
    target = path or MASTER_GRADIENT_LEDGER_PATH
    lock = lock_path or _LEDGER_LOCK_PATH
    _ensure_parent(lock)
    payload = asdict(gradient)
    # asdict converts nested dataclass OperatingPoint correctly
    payload["written_at_utc"] = _utc_now()
    payload["written_pid"] = os.getpid()
    payload["written_host"] = socket.gethostname()
    payload["schema_version"] = "master_gradient_anchor_v1"
    line = json.dumps(payload, sort_keys=True)
    with open(lock, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            _atomic_write_append(target, line)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    # WAVE-3-AUTO-TRIGGER-RUNTIME-WIRE-IN: cathedral consumer post-anchor
    # hook (sister of Catalog #343 dispatch-outcome auto-refresh pattern).
    # Fail-quiet per the canonical contract: ledger write succeeded; the
    # downstream consumer surface must not block the canonical write path.
    try:
        _fire_post_anchor_consumer_hooks(payload)
    except Exception:  # noqa: BLE001 — fail-quiet per the canonical contract
        pass


def _fire_post_anchor_consumer_hooks(anchor_row: dict) -> None:
    """Invoke ``update_from_anchor`` on every opt-in cathedral consumer.

    Discovers compliant consumers via
    :func:`tools.cathedral_autopilot_autonomous_loop.discover_compliant_consumer_modules`
    (sister of Catalog #335 canonical-contract auto-discovery), filters to
    those declaring module-level ``CONSUMES_MASTER_GRADIENT_ANCHORS = True``,
    and calls each consumer's ``update_from_anchor(anchor_row)``.

    Per-consumer exceptions are caught + warning-logged so a single buggy
    consumer cannot block sister consumers from receiving the anchor.

    Sister of Catalog #343 pattern at
    :func:`tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Lazy import — keep master_gradient module import-time lightweight and
    # avoid circular imports with cathedral_consumers packages that may
    # themselves import from tac.master_gradient at module top-level.
    try:
        # The canonical discovery helper lives in tools/ — load via importlib
        # spec to keep the master_gradient → tools dependency one-directional.
        import importlib.util
        spec = importlib.util.find_spec(
            "tools.cathedral_autopilot_autonomous_loop"
        )
        if spec is None:
            # tools/ not on PYTHONPATH (e.g. installed-package surface);
            # fail-quiet — the hook is observability-only.
            return
        loop_module = importlib.import_module(
            "tools.cathedral_autopilot_autonomous_loop"
        )
        discover = getattr(
            loop_module, "discover_compliant_consumer_modules", None
        )
        if discover is None:
            return
        modules = discover()
    except Exception as exc:  # noqa: BLE001 — fail-quiet
        logger.warning(
            "master_gradient.append_anchor_locked: consumer discovery "
            "failed (%s); skipping post-anchor hook fan-out",
            exc,
        )
        return

    for mod in modules:
        if not getattr(mod, "CONSUMES_MASTER_GRADIENT_ANCHORS", False):
            continue
        hook = getattr(mod, "update_from_anchor", None)
        if hook is None or not callable(hook):
            continue
        try:
            hook(anchor_row)
        except Exception as exc:  # noqa: BLE001 — fail-quiet per consumer
            logger.warning(
                "master_gradient.append_anchor_locked: consumer %s "
                "update_from_anchor raised %s (%s); ledger write "
                "preserved; sister consumers continue",
                getattr(mod, "__name__", "<unknown>"),
                type(exc).__name__,
                exc,
            )


def load_anchors_lenient(path: Path | None = None) -> list[dict]:
    """Skip malformed rows; return parsed list."""
    target = path or MASTER_GRADIENT_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict] = []
    for line in target.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_anchors_strict(path: Path | None = None) -> list[dict]:
    """Raise + quarantine on parse failure per Catalog #138."""
    target = path or MASTER_GRADIENT_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict] = []
    for idx, line in enumerate(target.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            quarantine = target.with_suffix(f".corrupt.{_utc_now()}")
            target.rename(quarantine)
            raise MasterGradientLedgerCorruptError(
                f"master_gradient_anchors.jsonl line {idx + 1}: {exc}; quarantined to {quarantine}"
            ) from exc
        if not isinstance(row, dict):
            quarantine = target.with_suffix(f".corrupt.{_utc_now()}")
            target.rename(quarantine)
            raise MasterGradientLedgerCorruptError(
                f"master_gradient_anchors.jsonl line {idx + 1}: non-dict root; quarantined to {quarantine}"
            )
        rows.append(row)
    return rows


def query_anchors_by_archive(
    archive_sha256: str, *, path: Path | None = None
) -> list[dict]:
    return [
        r for r in load_anchors_lenient(path) if r.get("archive_sha256") == archive_sha256
    ]


def contest_axis_authority_violation_reason(
    row: Mapping[str, object],
) -> str | None:
    """Return why a row cannot carry contest-axis master-gradient authority.

    Historical advisory/subset rows may carry an over-strong axis label. Treat
    those as diagnostic even if ``measurement_axis`` says ``[contest-CPU]`` or
    ``[contest-CUDA]`` so downstream consumers fail closed instead of promoting
    stale local probes.
    """
    axis = row.get("measurement_axis")
    normalized_axis = _normalized_contest_axis(axis)
    if normalized_axis is None:
        return None

    hardware = str(row.get("measurement_hardware", "")).lower()
    if any(token in hardware for token in ("advisory", "darwin", "macos", "mps")):
        return "contest axis uses advisory/local/proxy hardware"
    if normalized_axis == "[contest-CPU]":
        if "linux" not in hardware or "cpu" not in hardware:
            return "contest-CPU axis requires linux CPU hardware"
    elif normalized_axis == "[contest-CUDA]":
        cuda_tokens = ("cuda", "gpu", "t4", "a10", "a100", "h100", "l40", "rtx", "4090")
        if not any(token in hardware for token in cuda_tokens):
            return "contest-CUDA axis requires CUDA/GPU hardware"

    method = str(row.get("measurement_method", "")).lower()
    if "subset" in method:
        return "contest axis uses subset measurement method"
    if not str(row.get("measurement_call_id") or "").strip():
        return "contest axis missing measurement call/runtime custody"

    n_pairs_used = row.get("n_pairs_used")
    n_pairs_total = row.get("n_pairs_total")
    if n_pairs_used is None or n_pairs_total is None:
        return "contest axis missing pair-count custody"
    try:
        if int(n_pairs_used) != int(n_pairs_total):
            return "contest axis uses pair subset"
    except (TypeError, ValueError):
        return "contest axis has unparsable pair counts"

    archive_sha = str(row.get("archive_sha256") or "")
    scored_archive_sha = str(row.get("scored_archive_sha256") or "")
    if not scored_archive_sha:
        return "contest axis missing scored archive SHA custody"
    if archive_sha and scored_archive_sha.lower() != archive_sha.lower():
        return "contest axis scored archive SHA does not match anchor archive"
    scored_archive_bytes = row.get("scored_archive_bytes")
    if isinstance(scored_archive_bytes, bool):
        return "contest axis has invalid scored archive byte custody"
    try:
        if int(scored_archive_bytes) <= 0:
            return "contest axis has invalid scored archive byte custody"
    except (TypeError, ValueError):
        return "contest axis has invalid scored archive byte custody"
    return None


def is_authoritative_axis_anchor(row: Mapping[str, object]) -> bool:
    """Backward-compatible alias for :func:`is_usable_planning_anchor`.

    Non-contest diagnostic/advisory rows return True here because they do not
    claim contest-axis authority. Use
    :func:`is_authoritative_contest_axis_anchor` when a caller specifically
    needs a contest-CPU/contest-CUDA source row.
    """
    return is_usable_planning_anchor(row)


def is_usable_planning_anchor(row: Mapping[str, object]) -> bool:
    """Return whether a row is safe for diagnostic/planning consumption.

    Diagnostic/advisory axes are usable planning signal. Contest axes are usable
    only when their hardware/method/pair-count/runtime custody is complete.
    """
    return contest_axis_authority_violation_reason(row) is None


def is_authoritative_contest_axis_anchor(row: Mapping[str, object]) -> bool:
    """Return whether a row has authoritative contest-axis custody."""
    return (
        _normalized_contest_axis(row.get("measurement_axis")) is not None
        and contest_axis_authority_violation_reason(row) is None
    )


def unresolved_contest_axis_authority_violations(
    rows: Iterable[Mapping[str, object]],
) -> list[tuple[Mapping[str, object], str]]:
    """Return effective contest-axis authority violations after append-only corrections.

    The ledger is append-only historical state. A stale row can be corrected by a
    later row for the same archive + tensor artifact that re-labels the operating
    point as diagnostic/advisory. This helper therefore reports false-authority
    rows only when they remain the latest effective row for that exact gradient
    artifact. A later CUDA row, per-pair row, or different sidecar cannot hide an
    older false CPU row.
    """
    materialized = list(rows)
    latest_by_key: dict[tuple[str, str, str, str, str], Mapping[str, object]] = {}
    for row in materialized:
        archive = str(row.get("archive_sha256") or "")
        if not archive:
            continue
        sidecar_or_call = str(row.get("gradient_array_path") or "")
        if not sidecar_or_call:
            sidecar_or_call = str(row.get("measurement_call_id") or "")
        key = (
            archive,
            str(row.get("gradient_tensor_kind") or AGGREGATE_GRADIENT_TENSOR_KIND),
            sidecar_or_call,
            str(row.get("gradient_subject_sha256") or ""),
            str(row.get("scored_archive_sha256") or ""),
        )
        current = latest_by_key.get(key)
        if current is None or str(row.get("measurement_utc", "")) >= str(
            current.get("measurement_utc", "")
        ):
            latest_by_key[key] = row

    violations: list[tuple[Mapping[str, object], str]] = []
    for row in latest_by_key.values():
        reason = contest_axis_authority_violation_reason(row)
        if reason is not None:
            violations.append((row, reason))
    return violations


def latest_anchor_for_archive(
    archive_sha256: str,
    *,
    path: Path | None = None,
    axis: str | None = None,
) -> dict | None:
    """Latest gradient anchor for this archive (optionally filtered by axis)."""
    rows = query_anchors_by_archive(archive_sha256, path=path)
    if axis is not None:
        rows = [r for r in rows if r.get("measurement_axis") == axis]
        if _is_contest_axis_label(axis):
            rows = [r for r in rows if is_authoritative_axis_anchor(r)]
    else:
        rows = [r for r in rows if is_authoritative_axis_anchor(r)]
    if not rows:
        return None
    return max(rows, key=effective_anchor_sort_key)


def latest_rejected_contest_axis_anchor_for_archive(
    archive_sha256: str,
    *,
    path: Path | None = None,
    axis: str,
) -> tuple[Mapping[str, object], str] | None:
    """Latest same-axis row that was rejected by the contest-axis authority filter."""
    if axis not in AUTHORITATIVE_CONTEST_AXES:
        raise ValueError(f"axis must be one of {sorted(AUTHORITATIVE_CONTEST_AXES)!r}")
    rows = [
        r
        for r in query_anchors_by_archive(archive_sha256, path=path)
        if r.get("measurement_axis") == axis
    ]
    rejected: list[tuple[Mapping[str, object], str]] = []
    for row in rows:
        reason = contest_axis_authority_violation_reason(row)
        if reason is not None:
            rejected.append((row, reason))
    if not rejected:
        return None
    return max(rejected, key=lambda item: item[0].get("measurement_utc", ""))


def _operating_point_from_anchor(anchor: Mapping[str, object]) -> OperatingPoint:
    op = anchor["operating_point"]
    if isinstance(op, dict):
        return OperatingPoint(
            d_seg=float(op["d_seg"]),
            d_pose=float(op["d_pose"]),
            rate=float(op.get("rate", op.get("R", 0.0))),
            score=float(op["score"]),
        )
    if isinstance(op, OperatingPoint):
        return op
    raise TypeError(f"operating_point must be dict or OperatingPoint, got {type(op).__name__}")


def build_score_axis_dominance_backfill_anchor(
    anchor: Mapping[str, object],
    *,
    axis_dominance_threshold: float = 0.7,
) -> dict[str, object]:
    """Return an append-only anchor correction with producer-side dominance metadata."""

    if np is None:
        raise RuntimeError("numpy required to backfill score-axis dominance")
    if not isinstance(anchor, Mapping):
        raise TypeError("anchor must be a mapping")
    gradient_path_value = anchor.get("gradient_array_path")
    if not isinstance(gradient_path_value, str) or not gradient_path_value:
        raise ValueError("anchor gradient_array_path missing")
    gradient_path = Path(gradient_path_value)
    if not gradient_path.is_absolute():
        gradient_path = Path.cwd() / gradient_path
    if not gradient_path.is_file():
        raise FileNotFoundError(f"gradient array not found: {gradient_path}")

    op_obj = _operating_point_from_anchor(anchor)
    tensor = np.load(gradient_path)
    summary = score_axis_dominance_summary(
        tensor,
        op_obj,
        axis_dominance_threshold=axis_dominance_threshold,
    )
    summary.update(
        {
            "backfill_schema": "master_gradient_score_axis_dominance_backfill_v1",
            "backfill_reason": "producer_side_score_axis_dominance_metadata_closure",
            "derived_by": "tac.master_gradient.build_score_axis_dominance_backfill_anchor",
            "source_anchor_row_canonical_json_sha256": _canonical_json_sha256(anchor),
        }
    )
    corrected = dict(anchor)
    corrected["score_axis_dominance"] = summary
    return corrected


def append_score_axis_dominance_backfill(
    anchor: Mapping[str, object],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    axis_dominance_threshold: float = 0.7,
) -> dict[str, object]:
    """Append a locked metadata-correction anchor with score-axis dominance.

    This intentionally appends a new row instead of editing historical JSONL
    state. The corrected row keeps the original ``measurement_utc``; readers use
    ``written_at_utc`` as the tie-breaker via :func:`effective_anchor_sort_key`.
    """

    corrected = build_score_axis_dominance_backfill_anchor(
        anchor,
        axis_dominance_threshold=axis_dominance_threshold,
    )
    update_from_anchor(corrected, path=path, lock_path=lock_path)
    return corrected


def update_from_anchor(
    anchor: dict,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Catalog #265 canonical-contract alias.

    Accepts a dict-shape anchor (e.g. from continual-learning sister); persists
    via the canonical helper."""
    op_obj = _operating_point_from_anchor(anchor)
    grad = MasterGradient(
        archive_sha256=anchor["archive_sha256"],
        operating_point=op_obj,
        gradient_array_path=anchor["gradient_array_path"],
        n_bytes=int(anchor["n_bytes"]),
        measurement_method=anchor["measurement_method"],
        measurement_axis=anchor["measurement_axis"],
        measurement_hardware=anchor["measurement_hardware"],
        measurement_call_id=anchor.get("measurement_call_id"),
        measurement_utc=anchor.get("measurement_utc", _utc_now()),
        pareto_facets=tuple(tuple(p) for p in anchor.get("pareto_facets", ())),
        rashomon_disagreement_score=anchor.get("rashomon_disagreement_score"),
        gradient_tensor_kind=anchor.get(
            "gradient_tensor_kind", AGGREGATE_GRADIENT_TENSOR_KIND
        ),
        n_pairs=(
            int(anchor["n_pairs"]) if anchor.get("n_pairs") is not None else None
        ),
        scored_archive_sha256=anchor.get("scored_archive_sha256"),
        scored_archive_bytes=(
            int(anchor["scored_archive_bytes"])
            if anchor.get("scored_archive_bytes") is not None
            else None
        ),
        gradient_subject_sha256=anchor.get("gradient_subject_sha256"),
        gradient_subject_bytes=(
            int(anchor["gradient_subject_bytes"])
            if anchor.get("gradient_subject_bytes") is not None
            else None
        ),
        gradient_byte_domain=str(
            anchor.get("gradient_byte_domain", "scored_archive_bytes")
        ),
        n_pairs_used=(
            int(anchor["n_pairs_used"])
            if anchor.get("n_pairs_used") is not None
            else None
        ),
        n_pairs_total=(
            int(anchor["n_pairs_total"])
            if anchor.get("n_pairs_total") is not None
            else None
        ),
        score_axis_dominance=(
            anchor["score_axis_dominance"]
            if isinstance(anchor.get("score_axis_dominance"), Mapping)
            else None
        ),
    )
    append_anchor_locked(grad, path=path, lock_path=lock_path)
