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
import json
import math
import os
import socket
import tempfile
import uuid
from collections.abc import Mapping
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
    "compute_marginal_coefficients",
    "latest_anchor_for_archive",
    "load_anchors_lenient",
    "load_anchors_strict",
    "predict_delta_s",
    "predict_delta_s_per_pair",
    "query_anchors_by_archive",
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
    """fcntl-locked append per Catalog #128 / #131 / #245."""
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
    if not rows:
        return None
    return max(rows, key=lambda r: r.get("measurement_utc", ""))


def update_from_anchor(anchor: dict, *, path: Path | None = None) -> None:
    """Catalog #265 canonical-contract alias.

    Accepts a dict-shape anchor (e.g. from continual-learning sister); persists
    via the canonical helper."""
    op = anchor["operating_point"]
    if isinstance(op, dict):
        op_obj = OperatingPoint(
            d_seg=float(op["d_seg"]),
            d_pose=float(op["d_pose"]),
            rate=float(op.get("rate", op.get("R", 0.0))),
            score=float(op["score"]),
        )
    elif isinstance(op, OperatingPoint):
        op_obj = op
    else:
        raise TypeError(f"operating_point must be dict or OperatingPoint, got {type(op).__name__}")
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
    )
    append_anchor_locked(grad, path=path)
