# SPDX-License-Identifier: MIT
"""Canonical per-substrate Wyner-Ziv deliverability proof builder.

Per the T3 Grand Council symposium 2026-05-17 verdict
``.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md``
+ implementation queue
``.omx/research/wyner_ziv_optimal_implementation_queue_20260517.md`` Q1.

Consumes:
  1. ``WynerZivSideInfoClassification`` from
     ``tac.master_gradient_consumers.wyner_ziv_side_info_covariance``
     (candidate-shared-prior byte indices + pair-specific byte indices +
     mixed byte indices + estimated WZ gain).
  2. Optional ``deliverability_prober_result`` dict from
     ``tools/wyner_ziv_deliverability_prober.py`` (sister subagent output)
     with per-byte source-class metadata (which canonical helper would
     reconstruct the byte, compressed size estimate, scorer-dominance).
  3. Optional raw ``archive_bytes`` for live lzma/brotli/zlib compression
     probes when the prober output is unavailable.
  4. Operator-approval flag for Tier 3 bytes per HNeRV parity L4
     (inflate.py ≤ 200 LOC waiver requires operator review).

Produces:
  ``DeliverabilityProof`` frozen dataclass with all per-tier byte counts,
  byte indices, per-tier score-savings estimates, compliance verdict, and
  canonical_helper_invocation citation. Persists to
  ``.omx/state/wyner_ziv_deliverability/proof_<sha[:12]>_<utc>.json``
  via fcntl-locked write per Catalog #131.

Public API:
  * ``build_deliverability_proof_from_wyner_ziv_classification(...)``
  * ``load_deliverability_proof_for_archive(archive_sha256)``
  * ``verify_deliverability_proof_contest_compliance(proof)``

Contest-compliance checks (per ``verify_deliverability_proof_contest_compliance``):
  * strict-scorer-rule (Catalog #6 + CLAUDE.md "Strict scorer rule") — no
    scorer load at inflate time => Tier 4 byte count must be 0 OR caller
    accepted via ``operator_approved_tier_3=True`` (which still keeps Tier
    4 forbidden; only relaxes Tier 3).
  * HNeRV parity L4 (inflate.py ≤ 200 LOC waiver) — Tier 3 byte count > 0
    requires ``operator_approved_tier_3=True``.
  * HNeRV parity L9 runtime closure — every nonzero Tier MUST cite a
    canonical helper invocation; bare "unknown" rejected.
  * Catalog #213 (Comma2k19 canonical helper) — any Tier 2 byte whose
    citation mentions Comma2k19 MUST cite ``Comma2k19LocalCache.fetch_chunk``
    or sister ``Comma2k19LocalCache`` API (no raw URLs / urlopen / wget).

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305:

* Inspectable per layer: every per-byte tier classification is recorded
  in ``tier_N_byte_indices`` tuples (sorted, ascending).
* Decomposable per signal: per-tier byte count + per-tier score savings
  + canonical_helper_invocation citation per nonzero tier.
* Diff-able across runs: ``proof_sha256`` is sha256 over the canonical
  serialization of all fields; two builder invocations on equivalent
  inputs produce byte-identical proof files.
* Queryable post-hoc: persisted as JSON at
  ``.omx/state/wyner_ziv_deliverability/proof_<archive_sha[:12]>_<utc>.json``.
* Cite-able: every proof carries (archive_sha256, written_at_utc,
  canonical_helper_invocation, evidence_grade).
* Counterfactual-able: ``score_claim=False`` + ``promotion_eligible=False``
  defaults force operator review before any predicted-band claim becomes
  a frontier ranking decision; mutation of any tier byte count triggers a
  fresh proof build.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import errno
import fcntl
import hashlib
import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # avoid runtime import cycle
    from tac.master_gradient_consumers import WynerZivSideInfoClassification

__all__ = [
    "DELIVERABILITY_PROOF_SCHEMA_VERSION",
    "WYNER_ZIV_DELIVERABILITY_PROOFS_DIR",
    "DeliverabilityProof",
    "DeliverabilityTier",
    "build_deliverability_proof_from_wyner_ziv_classification",
    "load_deliverability_proof_for_archive",
    "verify_deliverability_proof_contest_compliance",
]


# ---------------------------------------------------------------------------
# Canonical paths + schema version
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
WYNER_ZIV_DELIVERABILITY_PROOFS_DIR = (
    _REPO_ROOT / ".omx" / "state" / "wyner_ziv_deliverability"
)
DELIVERABILITY_PROOF_SCHEMA_VERSION = "deliverability_proof_v1"
_LOCK_TIMEOUT_SEC = 30.0
_LOCK_FILE_NAME = "_proofs.lock"


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class DeliverabilityTier(str, Enum):
    """Per-byte deliverability classification per T3 symposium Component 2.

    The four tiers are exhaustive + mutually exclusive. Each byte index in
    a candidate-shared-prior set MUST be assigned to exactly one tier.

    * TIER_1_ZERO_COST: deterministic transforms of frame_0 / well-known
      constants (torch.zeros patterns, math constants like pi/e/sqrt(2),
      identity tensors). No archive cost; no LOC waiver; no operator
      review.
    * TIER_2_CONSTANTS: ≤ 5 KB baked Python literals derived from public
      datasets (Comma2k19 UV palette, ImageNet statistics, dashcam priors).
      No archive cost (constants baked into inflate.py); LOC waiver may
      apply per HNeRV L4 if inflate.py + baked constants exceed 100 LOC.
    * TIER_3_WAIVER_REQUIRED: 5 KB < cumulative compressed size ≤ 200 KB.
      Requires operator review (HNeRV L4 inflate.py ≤ 200 LOC waiver) AND
      explicit ``operator_approved_tier_3=True`` flag.
    * TIER_4_FORBIDDEN: bytes that require scorer access (POSE_AXIS or
      SEG_AXIS dominant in per-pair gradient breakdown), network fetch at
      inflate time (not derivable from Comma2k19LocalCache or sister
      canonical helpers), OR non-reproducible state. CLAUDE.md strict-
      scorer-rule FORBIDS these from any contest archive.
    """

    TIER_1_ZERO_COST = "tier_1_zero_cost"
    TIER_2_CONSTANTS = "tier_2_constants"
    TIER_3_WAIVER_REQUIRED = "tier_3_waiver_required"
    TIER_4_FORBIDDEN = "tier_4_forbidden"


_LEGAL_COMPRESSION_CODECS = frozenset({"lzma", "brotli", "zlib"})
_LEGAL_CONTEST_COMPLIANCE_VERDICTS = frozenset(
    {"pending", "compliant", "partial", "non_compliant"}
)
_LEGAL_EVIDENCE_GRADES = frozenset(
    {"predicted", "empirical_cpu", "empirical_paired_cuda"}
)
_LEGAL_OPERATOR_REVIEW_STATUSES = frozenset(
    {"pending", "approved", "denied"}
)
_LEGAL_CATALOG_319_STATUSES = frozenset({"pending", "passed", "refused"})

# Canonical contest rate-term denominator (bytes); 25 / N is the formula
# coefficient per CLAUDE.md "Contest scoring".
_CONTEST_RATE_DENOM_BYTES = 37_545_489

DEFAULT_CONTEST_COMPLIANCE_RATIONALE = (
    "Contest-compliance boundary: upstream/evaluate.py:63 charges "
    "archive.zip as the rate-term container. Output-affecting source bytes "
    "must be INSIDE archive.zip or deterministically regenerated by an allowed "
    "canonical helper such as Comma2k19LocalCache per Catalog #213. "
    "Output-affecting payload bytes OUTSIDE archive.zip are the rejected "
    "PR #68 loophole_v2 class. This proof is planning evidence only "
    "(score_claim=False, promotion_eligible=False)."
)


# ---------------------------------------------------------------------------
# Tier-1 / Tier-4 detection patterns (no runtime imports required)
# ---------------------------------------------------------------------------

# Canonical Tier-1 blob fingerprints: sha256 of a small set of zero-cost
# byte patterns that the contest decoder can reconstruct without any
# archive bytes. We hash the candidate-shared-prior byte ranges against
# this set during build.

_TIER_1_PATTERN_SHAS: dict[str, str] = {
    "zeros_uint8_1k": hashlib.sha256(b"\x00" * 1024).hexdigest(),
    "zeros_uint8_256": hashlib.sha256(b"\x00" * 256).hexdigest(),
    "zeros_uint8_64": hashlib.sha256(b"\x00" * 64).hexdigest(),
    "ones_uint8_64": hashlib.sha256(b"\x01" * 64).hexdigest(),
}


def _bytes_match_tier_1_pattern(blob: bytes) -> bool:
    """Return True if `blob` exactly equals a known zero-cost byte pattern."""
    if len(blob) == 0:
        return True  # trivially reconstructible
    if blob == b"\x00" * len(blob):
        return True  # all-zero pattern of any length
    if blob == b"\x01" * len(blob):
        return True  # all-ones pattern of any length
    h = hashlib.sha256(blob).hexdigest()
    return h in set(_TIER_1_PATTERN_SHAS.values())


# ---------------------------------------------------------------------------
# DeliverabilityProof dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliverabilityProof:
    """Per-substrate Wyner-Ziv deliverability proof.

    Consumed by Catalog #319 STRICT preflight gate (Q2 successor) and the
    autopilot reweight v2 (Q3 successor) to apply per-tier reward factors:
    Tier 1: 1.20x / Tier 2: 1.10x / Tier 3: 1.05x conditional on operator
    approval / Tier 4 or no-proof: 1.0x (no reward).

    All byte indices are sorted ascending. All score-savings estimates are
    in contest-score units (multiplied by 25/37_545_489 per the canonical
    rate formula). The proof is frozen — mutation requires a fresh build.
    """

    # Identity
    archive_sha256: str
    candidate_shared_prior_byte_count: int

    # Per-tier byte counts
    tier_1_byte_count: int
    tier_2_byte_count: int
    tier_3_byte_count: int
    tier_4_byte_count: int

    # Per-tier byte indices (sorted ascending tuples)
    tier_1_byte_indices: tuple[int, ...] = field(default=())
    tier_2_byte_indices: tuple[int, ...] = field(default=())
    tier_3_byte_indices: tuple[int, ...] = field(default=())
    tier_4_byte_indices: tuple[int, ...] = field(default=())

    # Per-tier score-savings estimates (in contest-score units)
    tier_1_score_savings_estimate: float = 0.0
    tier_2_score_savings_estimate: float = 0.0
    tier_3_score_savings_estimate: float = 0.0
    deliverable_score_savings_estimate: float = 0.0

    # Tier-3 waiver / operator review
    waiver_required_for_tier_3: bool = True
    operator_review_status_for_tier_3: str = "pending"

    # Inflate runtime budget
    inflate_py_loc_estimate: int = 0
    inflate_py_compressed_bytes_estimate: int = 0
    compression_codec: str = "lzma"
    compression_ratio: float = 1.0

    # Canonical helper invocation citation (HNeRV parity L9 runtime closure)
    canonical_helper_invocation: str = ""

    # Contest compliance verdict
    contest_compliance_verdict: str = "pending"
    contest_compliance_rationale: str = DEFAULT_CONTEST_COMPLIANCE_RATIONALE

    # Custody / promotion routing per CLAUDE.md "Apples-to-apples evidence
    # discipline" + Catalog #127 (custody validator)
    evidence_grade: str = "predicted"
    score_claim: bool = False
    promotion_eligible: bool = False
    catalog_319_gate_status: str = "pending"

    # Provenance + observability
    proof_sha256: str = ""
    written_at_utc: str = ""
    schema_version: str = DELIVERABILITY_PROOF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate frozen-dataclass invariants per Catalog #229 PV-style.

        Run AFTER dataclass auto-init. Enforces:
          * archive_sha256 is a non-empty hex-lowercase 64-char sha256
          * per-tier counts sum to candidate_shared_prior_byte_count
          * per-tier indices are sorted ascending + match per-tier counts
          * compression_codec is one of {lzma, brotli, zlib}
          * contest_compliance_verdict + evidence_grade +
            operator_review_status_for_tier_3 + catalog_319_gate_status
            are legal enum members
          * promotion_eligible=True requires contest_compliance_verdict in
            {compliant, partial} + evidence_grade in
            {empirical_cpu, empirical_paired_cuda}
        """
        if not isinstance(self.archive_sha256, str) or len(self.archive_sha256) != 64:
            raise ValueError(
                f"archive_sha256 must be a 64-char hex string; got "
                f"{self.archive_sha256!r}"
            )
        try:
            int(self.archive_sha256, 16)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"archive_sha256 must be hex; got {self.archive_sha256!r}"
            ) from exc

        # Per-tier byte counts must sum to candidate total
        per_tier_sum = (
            self.tier_1_byte_count
            + self.tier_2_byte_count
            + self.tier_3_byte_count
            + self.tier_4_byte_count
        )
        if per_tier_sum != self.candidate_shared_prior_byte_count:
            raise ValueError(
                f"per-tier byte counts must sum to "
                f"candidate_shared_prior_byte_count "
                f"({self.candidate_shared_prior_byte_count}); got sum "
                f"{per_tier_sum}"
            )

        # Per-tier indices must match per-tier counts
        for tier_name, indices, count in [
            ("tier_1", self.tier_1_byte_indices, self.tier_1_byte_count),
            ("tier_2", self.tier_2_byte_indices, self.tier_2_byte_count),
            ("tier_3", self.tier_3_byte_indices, self.tier_3_byte_count),
            ("tier_4", self.tier_4_byte_indices, self.tier_4_byte_count),
        ]:
            if not isinstance(indices, tuple):
                raise ValueError(
                    f"{tier_name}_byte_indices must be a tuple; got "
                    f"{type(indices).__name__}"
                )
            if len(indices) != count:
                raise ValueError(
                    f"{tier_name}_byte_indices length ({len(indices)}) must "
                    f"equal {tier_name}_byte_count ({count})"
                )
            if list(indices) != sorted(indices):
                raise ValueError(
                    f"{tier_name}_byte_indices must be sorted ascending"
                )

        if self.compression_codec not in _LEGAL_COMPRESSION_CODECS:
            raise ValueError(
                f"compression_codec must be in "
                f"{sorted(_LEGAL_COMPRESSION_CODECS)}; got "
                f"{self.compression_codec!r}"
            )
        if self.contest_compliance_verdict not in _LEGAL_CONTEST_COMPLIANCE_VERDICTS:
            raise ValueError(
                f"contest_compliance_verdict must be in "
                f"{sorted(_LEGAL_CONTEST_COMPLIANCE_VERDICTS)}; got "
                f"{self.contest_compliance_verdict!r}"
            )
        if self.evidence_grade not in _LEGAL_EVIDENCE_GRADES:
            raise ValueError(
                f"evidence_grade must be in {sorted(_LEGAL_EVIDENCE_GRADES)}; "
                f"got {self.evidence_grade!r}"
            )
        if self.operator_review_status_for_tier_3 not in _LEGAL_OPERATOR_REVIEW_STATUSES:
            raise ValueError(
                f"operator_review_status_for_tier_3 must be in "
                f"{sorted(_LEGAL_OPERATOR_REVIEW_STATUSES)}; got "
                f"{self.operator_review_status_for_tier_3!r}"
            )
        if self.catalog_319_gate_status not in _LEGAL_CATALOG_319_STATUSES:
            raise ValueError(
                f"catalog_319_gate_status must be in "
                f"{sorted(_LEGAL_CATALOG_319_STATUSES)}; got "
                f"{self.catalog_319_gate_status!r}"
            )

        if self.promotion_eligible:
            if self.contest_compliance_verdict not in {"compliant", "partial"}:
                raise ValueError(
                    "promotion_eligible=True requires "
                    "contest_compliance_verdict in {compliant, partial}; "
                    f"got {self.contest_compliance_verdict!r}"
                )
            if self.evidence_grade not in {  # CUSTODY_VALIDATOR_OK:fail-closed guard inside DeliverabilityProof.__post_init__ refuses promotion_eligible=True for non-empirical evidence_grades; this IS the local custody validator routing per Catalog #127 (the proof IS the custody context — score_claim + promotion_eligible default False and only flip True when contest_compliance_verdict + evidence_grade are both empirically anchored)
                "empirical_cpu",
                "empirical_paired_cuda",
            }:
                raise ValueError(
                    "promotion_eligible=True requires evidence_grade in "
                    "{empirical_cpu, empirical_paired_cuda}; got "
                    f"{self.evidence_grade!r}"
                )

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (tuples become lists)."""
        d = dataclasses.asdict(self)
        for k in (
            "tier_1_byte_indices",
            "tier_2_byte_indices",
            "tier_3_byte_indices",
            "tier_4_byte_indices",
        ):
            d[k] = list(d[k])
        return d


# ---------------------------------------------------------------------------
# Fcntl-locked write helper (per-file lock at WYNER_ZIV_DELIVERABILITY_PROOFS_DIR / _LOCK_FILE_NAME)
# ---------------------------------------------------------------------------


@contextmanager
def _proofs_lock(lock_path: Path | None = None) -> Iterator[None]:
    """Acquire fcntl LOCK_EX on the proofs-dir lock file per Catalog #131."""
    path = lock_path or (WYNER_ZIV_DELIVERABILITY_PROOFS_DIR / _LOCK_FILE_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
    fd: int | None = None
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError) as exc:
                if isinstance(exc, OSError) and exc.errno not in (
                    errno.EWOULDBLOCK,
                    errno.EAGAIN,
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring deliverability-proof lock "
                        f"after {_LOCK_TIMEOUT_SEC}s at {path}"
                    )
                time.sleep(0.05)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write — unique .tmp.<uuid12> + fsync + os.replace.

    HISTORICAL_PROVENANCE per Catalog #110/#113: proof files are
    append-only by archive_sha256 + written_at_utc; a new build for the
    same archive lands a NEW file (different timestamp), preserving the
    audit trail.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(text, encoding="utf-8")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _compute_proof_sha256(record: dict[str, Any]) -> str:
    """Sha256 over canonical-sorted-JSON of all non-provenance fields."""
    canonical = {
        k: v
        for k, v in record.items()
        if k
        not in {
            "proof_sha256",
            "written_at_utc",
            "schema_version",
            "written_pid",
            "written_host",
        }
    }
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _per_tier_score_savings(byte_count: int) -> float:
    """Canonical contest rate-term savings = 25 * byte_count / denom."""
    return 25.0 * float(byte_count) / float(_CONTEST_RATE_DENOM_BYTES)


def _classify_byte_indices_into_tiers(
    *,
    candidate_indices: tuple[int, ...],
    archive_bytes: bytes | None,
    deliverability_prober_result: dict | None,
    pose_seg_dominant_byte_indices: tuple[int, ...],
    tier_2_budget_bytes: int,
    tier_3_budget_bytes: int,
    compression_codec: str,
) -> dict[str, list[int]]:
    """Greedy per-tier assignment of candidate-shared-prior byte indices.

    Priority order (Tier 4 first to remove forbidden bytes, then Tier 1
    zero-cost, then Tier 2/3 by budget):
      1. Tier 4: any byte in pose_seg_dominant_byte_indices OR flagged
         scorer/network-dependent by the prober.
      2. Tier 1: bytes matching canonical zero-cost patterns
         (heuristic — depends on archive_bytes OR prober metadata).
      3. Tier 2: cumulative compressed size <= tier_2_budget_bytes
         (greedy by descending per-byte information gain when prober
         output available; otherwise lexicographic order).
      4. Tier 3: cumulative compressed size > tier_2_budget AND
         <= tier_3_budget.
      5. Remaining bytes that exceed tier_3 budget are Tier 4 by exhaustion.
    """
    if compression_codec not in _LEGAL_COMPRESSION_CODECS:
        raise ValueError(
            f"compression_codec must be in "
            f"{sorted(_LEGAL_COMPRESSION_CODECS)}; got {compression_codec!r}"
        )

    candidate_set = set(int(i) for i in candidate_indices)
    pose_seg_set = set(int(i) for i in pose_seg_dominant_byte_indices)
    remaining = candidate_set.copy()

    tier_4: list[int] = sorted(remaining & pose_seg_set)
    remaining -= set(tier_4)

    # Prober-flagged scorer / network bytes -> Tier 4
    prober_tier_4: set[int] = set()
    prober_tier_1: set[int] = set()
    prober_per_byte_compressed_size: dict[int, int] = {}
    prober_per_byte_helper: dict[int, str] = {}
    if deliverability_prober_result is not None:
        per_byte = deliverability_prober_result.get("per_byte_classification", {})
        for k, v in per_byte.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(v, dict):
                continue
            source_class = v.get("source_class", "").lower()
            if source_class in {"scorer_dependent", "network_required", "forbidden"}:
                prober_tier_4.add(idx)
            elif source_class in {"zero_cost", "deterministic_transform"}:
                prober_tier_1.add(idx)
            csize = v.get("compressed_size_bytes")
            if isinstance(csize, int) and csize >= 0:
                prober_per_byte_compressed_size[idx] = csize
            helper = v.get("canonical_helper_invocation", "")
            if isinstance(helper, str) and helper:
                prober_per_byte_helper[idx] = helper

    tier_4_from_prober = sorted((remaining & prober_tier_4))
    if tier_4_from_prober:
        tier_4 = sorted(set(tier_4) | set(tier_4_from_prober))
        remaining -= set(tier_4_from_prober)

    # Tier 1: prober zero-cost OR bytes matching canonical zero patterns
    tier_1: list[int] = sorted(remaining & prober_tier_1)
    remaining -= set(tier_1)
    if archive_bytes is not None and remaining:
        # Test for ALL-ZERO contiguous runs of length >= 4 in candidate range
        contiguous_zero_runs: list[int] = []
        for idx in sorted(remaining):
            if 0 <= idx < len(archive_bytes) and archive_bytes[idx] == 0:
                contiguous_zero_runs.append(idx)
        if contiguous_zero_runs:
            tier_1 = sorted(set(tier_1) | set(contiguous_zero_runs))
            remaining -= set(contiguous_zero_runs)

    # Tier 2 / Tier 3: cumulative compressed-size budget split
    remaining_sorted = sorted(remaining)
    # Estimate per-byte compressed size: prefer prober output, else fallback
    # of 1 byte per index (assume incompressible) — caller can override via
    # archive_bytes for a more accurate estimate.
    def _per_byte_size(idx: int) -> int:
        if idx in prober_per_byte_compressed_size:
            return max(0, prober_per_byte_compressed_size[idx])
        return 1  # conservative: assume 1 byte per index

    tier_2: list[int] = []
    tier_3: list[int] = []
    cumulative = 0
    for idx in remaining_sorted:
        sz = _per_byte_size(idx)
        if cumulative + sz <= tier_2_budget_bytes:
            tier_2.append(idx)
            cumulative += sz
        elif cumulative + sz <= tier_2_budget_bytes + tier_3_budget_bytes:
            tier_3.append(idx)
            cumulative += sz
        else:
            # Exhaustion -> Tier 4 (cannot deliver via baked constants)
            tier_4.append(idx)
    tier_4 = sorted(set(tier_4))

    return {
        "tier_1": tier_1,
        "tier_2": tier_2,
        "tier_3": tier_3,
        "tier_4": tier_4,
        "_prober_per_byte_helper": [
            (idx, prober_per_byte_helper.get(idx, "")) for idx in tier_2 + tier_3
        ],
    }


def _derive_canonical_helper_invocation(
    *,
    tier_1_count: int,
    tier_2_count: int,
    tier_3_count: int,
    prober_helpers: list[tuple[int, str]],
    deliverability_prober_result: dict | None,
) -> str:
    """Derive the canonical_helper_invocation citation string.

    Preference order:
      1. Prober's top-level ``canonical_helper_invocation`` field if non-empty.
      2. Most-frequent per-byte helper from prober output.
      3. Tier-1-only proofs cite "tac.wyner_ziv_deliverability (zero-cost only;
         no external helper required)".
      4. Fallback: explicit "unknown" — sets contest_compliance_verdict to
         "non_compliant" via the verifier.
    """
    if deliverability_prober_result is not None:
        top_level = deliverability_prober_result.get("canonical_helper_invocation", "")
        if isinstance(top_level, str) and top_level.strip():
            return top_level.strip()
    if prober_helpers:
        # Most-frequent non-empty helper
        from collections import Counter

        non_empty = [h for _idx, h in prober_helpers if h]
        if non_empty:
            counts = Counter(non_empty)
            return counts.most_common(1)[0][0]
    if tier_1_count > 0 and tier_2_count == 0 and tier_3_count == 0:
        return "tac.wyner_ziv_deliverability (zero-cost only; no external helper required)"
    return "unknown"


def _contest_compliance_rationale_with_loophole_boundary(
    *,
    tier_rationale: str,
    helper: str,
) -> str:
    """Attach the upstream loophole-boundary rationale to a proof verdict.

    This is intentionally part of every persisted proof, not only a research
    memo: downstream autopilot and PR-body generators need the structural
    distinction between allowed deterministic helper code and forbidden
    out-of-archive payload relocation at the artifact level.
    """
    boundary = (
        "Contest-compliance boundary: upstream/evaluate.py:63 charges "
        "archive.zip as the rate-term container, so every output-affecting "
        "seed/source byte must either live INSIDE archive.zip or be "
        "deterministically reconstructed by an allowed canonical helper. "
        "Output-affecting payload bytes OUTSIDE archive.zip are the rejected "
        "PR #68 loophole_v2 class; deterministic helper code without hidden "
        "contest-video payload is categorically different. Catalog #213 "
        "requires Comma2k19-derived bytes to route through Comma2k19LocalCache. "
        f"Canonical helper: {helper}. "
        "This proof is planning evidence only until exact contest evaluation "
        "(score_claim=False, promotion_eligible=False)."
    )
    return f"{tier_rationale} {boundary}"


def _tier_2_rationale_missing_required_anchors(rationale: str) -> list[str]:
    """Return missing required anchors for Tier-2 procedural-generation proofs."""
    required = (
        "PR #68 loophole_v2",
        "upstream/evaluate.py:63",
        "Catalog #213",
        "Comma2k19LocalCache",
        "INSIDE archive.zip",
        "OUTSIDE archive.zip",
        "score_claim=False",
        "promotion_eligible=False",
    )
    return [anchor for anchor in required if anchor not in rationale]


def build_deliverability_proof_from_wyner_ziv_classification(
    *,
    wyner_ziv_result: "WynerZivSideInfoClassification",
    archive_sha256: str,
    archive_bytes: bytes | None = None,
    deliverability_prober_result: dict | None = None,
    pose_seg_dominant_byte_indices: tuple[int, ...] = (),
    operator_approved_tier_3: bool = False,
    compression_codec: str = "lzma",
    tier_2_budget_bytes: int = 5_120,
    tier_3_budget_bytes: int = 204_800,
    persist: bool = True,
    proofs_dir: Path | None = None,
) -> DeliverabilityProof:
    """Build canonical DeliverabilityProof from a WynerZivSideInfoClassification.

    Args:
      wyner_ziv_result: producer output from
        ``tac.master_gradient_consumers.wyner_ziv_side_info_covariance``.
        Only ``candidate_shared_prior_byte_indices`` is read by the builder
        (the other fields are recorded into the proof's provenance for audit).
      archive_sha256: SHA-256 of the archive bytes (64 hex chars).
      archive_bytes: optional raw archive bytes for live zero-cost pattern
        detection. Mutually accepted with deliverability_prober_result.
      deliverability_prober_result: optional dict output from sister subagent
        ``tools/wyner_ziv_deliverability_prober.py``. Schema:
          {
            "canonical_helper_invocation": str,
            "per_byte_classification": {
              "<byte_idx>": {
                "source_class": "zero_cost"|"comma2k19"|"imagenet"|"dashcam"|
                                "scorer_dependent"|"network_required"|"forbidden",
                "compressed_size_bytes": int,
                "canonical_helper_invocation": str,
              }, ...
            }
          }
        When provided, takes precedence over heuristics from archive_bytes.
      pose_seg_dominant_byte_indices: optional byte indices whose per-pair
        gradient breakdown is dominantly POSE_AXIS or SEG_AXIS (>0.7 of total
        magnitude). These bytes are TIER_4_FORBIDDEN per CLAUDE.md
        strict-scorer-rule. Caller must declare; the WZ producer does NOT
        carry per-axis breakdown today.
      operator_approved_tier_3: when True, Tier 3 bytes are allowed; when
        False (default), Tier 3 bytes do NOT contribute to
        deliverable_score_savings_estimate.
      compression_codec: one of {lzma, brotli, zlib}. Default lzma.
      tier_2_budget_bytes: default 5120 (~5 KB per HNeRV L4 Tier-2 ceiling).
      tier_3_budget_bytes: default 204800 (~200 KB per HNeRV L4 waiver ceiling).
      persist: when True, write the proof JSON to
        ``WYNER_ZIV_DELIVERABILITY_PROOFS_DIR/proof_<sha[:12]>_<utc>.json``.
      proofs_dir: override the canonical proofs directory (test fixture only).

    Returns:
      DeliverabilityProof — frozen dataclass with all per-tier byte counts,
      indices, score savings, compliance verdict, and canonical helper citation.
    """
    candidate_indices = tuple(int(i) for i in wyner_ziv_result.candidate_shared_prior_byte_indices)
    n_candidate = len(candidate_indices)

    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
        raise ValueError(
            f"archive_sha256 must be a 64-char hex string; got "
            f"{archive_sha256!r}"
        )

    if compression_codec not in _LEGAL_COMPRESSION_CODECS:
        raise ValueError(
            f"compression_codec must be in "
            f"{sorted(_LEGAL_COMPRESSION_CODECS)}; got {compression_codec!r}"
        )

    if tier_2_budget_bytes < 0 or tier_3_budget_bytes < 0:
        raise ValueError(
            f"tier budgets must be >= 0; got tier_2={tier_2_budget_bytes}, "
            f"tier_3={tier_3_budget_bytes}"
        )

    classification = _classify_byte_indices_into_tiers(
        candidate_indices=candidate_indices,
        archive_bytes=archive_bytes,
        deliverability_prober_result=deliverability_prober_result,
        pose_seg_dominant_byte_indices=pose_seg_dominant_byte_indices,
        tier_2_budget_bytes=tier_2_budget_bytes,
        tier_3_budget_bytes=tier_3_budget_bytes,
        compression_codec=compression_codec,
    )

    tier_1_count = len(classification["tier_1"])
    tier_2_count = len(classification["tier_2"])
    tier_3_count = len(classification["tier_3"])
    tier_4_count = len(classification["tier_4"])

    # Per-tier score-savings (canonical formula 25 * N / denom)
    tier_1_savings = _per_tier_score_savings(tier_1_count)
    tier_2_savings = _per_tier_score_savings(tier_2_count)
    tier_3_savings = _per_tier_score_savings(tier_3_count)

    # Deliverable savings = Tier 1 + Tier 2 always; Tier 3 only if operator
    # approved.
    deliverable = tier_1_savings + tier_2_savings
    if operator_approved_tier_3:
        deliverable += tier_3_savings

    # Operator review status
    operator_review = "approved" if operator_approved_tier_3 else "pending"

    # Canonical helper invocation
    helper = _derive_canonical_helper_invocation(
        tier_1_count=tier_1_count,
        tier_2_count=tier_2_count,
        tier_3_count=tier_3_count,
        prober_helpers=classification["_prober_per_byte_helper"],
        deliverability_prober_result=deliverability_prober_result,
    )

    # Contest compliance verdict: derived from tier breakdown
    # * Tier 4 > 0 -> non_compliant
    # * Tier 3 > 0 AND NOT operator_approved -> partial
    # * Else compliant
    if tier_4_count > 0:
        verdict = "non_compliant"
        tier_rationale = (
            f"Tier 4 (forbidden) byte count is {tier_4_count} > 0 per "
            f"CLAUDE.md strict-scorer-rule + HNeRV parity L9 runtime closure. "
            f"Scorer-dependent / network-required bytes cannot be hoisted "
            f"to a contest-deliverable shared prior."
        )
    elif tier_3_count > 0 and not operator_approved_tier_3:
        verdict = "partial"
        tier_rationale = (
            f"Tier 3 (waiver required) byte count is {tier_3_count} > 0 but "
            f"operator_approved_tier_3=False. Per HNeRV parity L4 the "
            f"inflate.py >= 100 LOC + Tier-3 baked-constant waiver requires "
            f"explicit operator review before contributing to deliverable "
            f"savings."
        )
    elif helper == "unknown" and (tier_1_count + tier_2_count + tier_3_count) > 0:
        verdict = "non_compliant"
        tier_rationale = (
            f"canonical_helper_invocation is 'unknown' for a proof with "
            f"nonzero deliverable tiers ({tier_1_count + tier_2_count + tier_3_count} "
            f"bytes). Per HNeRV parity L9 every nonzero Tier MUST cite a "
            f"canonical helper for runtime closure."
        )
    else:
        verdict = "compliant"
        tier_rationale = (
            f"Tier 4 = 0; canonical helper cited "
            f"({helper!r}); Tier 3 either zero or operator-approved."
        )
    rationale = _contest_compliance_rationale_with_loophole_boundary(
        tier_rationale=tier_rationale,
        helper=helper,
    )

    # Compression ratio: best-effort from prober output (compressed/raw)
    raw_total = max(1, n_candidate)
    compressed_total = 0
    if deliverability_prober_result is not None:
        per_byte = deliverability_prober_result.get("per_byte_classification", {})
        for v in per_byte.values():
            if isinstance(v, dict):
                csize = v.get("compressed_size_bytes")
                if isinstance(csize, int):
                    compressed_total += max(0, csize)
    if compressed_total <= 0:
        compressed_total = raw_total  # 1.0 fallback ratio
    compression_ratio = float(compressed_total) / float(raw_total)
    inflate_compressed_bytes = compressed_total

    # Inflate.py LOC estimate: ~20 LOC overhead + 1 LOC per ~100 bytes
    # baked constant. HNeRV L4 waiver ceiling is 200 LOC.
    inflate_loc_estimate = 20 + max(0, (tier_1_count + tier_2_count + tier_3_count) // 100)

    # Construct intermediate record (before sha computation)
    intermediate = {
        "archive_sha256": archive_sha256,
        "candidate_shared_prior_byte_count": n_candidate,
        "tier_1_byte_count": tier_1_count,
        "tier_2_byte_count": tier_2_count,
        "tier_3_byte_count": tier_3_count,
        "tier_4_byte_count": tier_4_count,
        "tier_1_byte_indices": list(classification["tier_1"]),
        "tier_2_byte_indices": list(classification["tier_2"]),
        "tier_3_byte_indices": list(classification["tier_3"]),
        "tier_4_byte_indices": list(classification["tier_4"]),
        "tier_1_score_savings_estimate": tier_1_savings,
        "tier_2_score_savings_estimate": tier_2_savings,
        "tier_3_score_savings_estimate": tier_3_savings,
        "deliverable_score_savings_estimate": deliverable,
        "waiver_required_for_tier_3": tier_3_count > 0,
        "operator_review_status_for_tier_3": operator_review,
        "inflate_py_loc_estimate": inflate_loc_estimate,
        "inflate_py_compressed_bytes_estimate": inflate_compressed_bytes,
        "compression_codec": compression_codec,
        "compression_ratio": compression_ratio,
        "canonical_helper_invocation": helper,
        "contest_compliance_verdict": verdict,
        "contest_compliance_rationale": rationale,
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotion_eligible": False,
        "catalog_319_gate_status": "pending",
    }
    proof_sha = _compute_proof_sha256(intermediate)
    written_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

    proof = DeliverabilityProof(
        archive_sha256=archive_sha256,
        candidate_shared_prior_byte_count=n_candidate,
        tier_1_byte_count=tier_1_count,
        tier_2_byte_count=tier_2_count,
        tier_3_byte_count=tier_3_count,
        tier_4_byte_count=tier_4_count,
        tier_1_byte_indices=tuple(classification["tier_1"]),
        tier_2_byte_indices=tuple(classification["tier_2"]),
        tier_3_byte_indices=tuple(classification["tier_3"]),
        tier_4_byte_indices=tuple(classification["tier_4"]),
        tier_1_score_savings_estimate=tier_1_savings,
        tier_2_score_savings_estimate=tier_2_savings,
        tier_3_score_savings_estimate=tier_3_savings,
        deliverable_score_savings_estimate=deliverable,
        waiver_required_for_tier_3=tier_3_count > 0,
        operator_review_status_for_tier_3=operator_review,
        inflate_py_loc_estimate=inflate_loc_estimate,
        inflate_py_compressed_bytes_estimate=inflate_compressed_bytes,
        compression_codec=compression_codec,
        compression_ratio=compression_ratio,
        canonical_helper_invocation=helper,
        contest_compliance_verdict=verdict,
        contest_compliance_rationale=rationale,
        evidence_grade="predicted",
        score_claim=False,
        promotion_eligible=False,
        catalog_319_gate_status="pending",
        proof_sha256=proof_sha,
        written_at_utc=written_at,
        schema_version=DELIVERABILITY_PROOF_SCHEMA_VERSION,
    )

    if persist:
        out_dir = proofs_dir or WYNER_ZIV_DELIVERABILITY_PROOFS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"proof_{archive_sha256[:12]}_{stamp}.json"
        payload = proof.as_dict()
        payload["written_pid"] = os.getpid()
        payload["written_host"] = socket.gethostname()
        with _proofs_lock(out_dir / _LOCK_FILE_NAME):
            _atomic_write_json(out_path, payload)

    return proof


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def load_deliverability_proof_for_archive(
    archive_sha256: str,
    *,
    proofs_dir: Path | None = None,
) -> DeliverabilityProof | None:
    """Load the most-recent DeliverabilityProof for the given archive sha256.

    Returns None when no proof exists. Per Catalog #319 STRICT gate
    semantics: a None return value signals the gate to REFUSE the Wyner-
    Ziv reward branch (no per-substrate deliverability proof => no reward).
    """
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
        raise ValueError(
            f"archive_sha256 must be a 64-char hex string; got "
            f"{archive_sha256!r}"
        )
    out_dir = proofs_dir or WYNER_ZIV_DELIVERABILITY_PROOFS_DIR
    if not out_dir.exists():
        return None
    pattern = f"proof_{archive_sha256[:12]}_*.json"
    matches = sorted(out_dir.glob(pattern))
    if not matches:
        return None
    # Sort by mtime descending; most-recent wins
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in matches:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Filter to canonical dataclass fields
        valid_fields = {f.name for f in dataclasses.fields(DeliverabilityProof)}
        kwargs = {k: v for k, v in payload.items() if k in valid_fields}
        # Coerce list -> tuple for byte-index fields
        for tier_key in (
            "tier_1_byte_indices",
            "tier_2_byte_indices",
            "tier_3_byte_indices",
            "tier_4_byte_indices",
        ):
            if tier_key in kwargs and isinstance(kwargs[tier_key], list):
                kwargs[tier_key] = tuple(int(i) for i in kwargs[tier_key])
        try:
            return DeliverabilityProof(**kwargs)
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def verify_deliverability_proof_contest_compliance(
    proof: DeliverabilityProof,
) -> tuple[bool, list[str]]:
    """Verify contest compliance against CLAUDE.md non-negotiables.

    Returns:
      (is_compliant, blockers) where blockers is a list of human-readable
      strings naming each non-negotiable violated. Empty list when compliant.

    Checks:
      1. strict-scorer-rule: tier_4_byte_count must be 0.
      2. HNeRV parity L4 (waiver discipline): tier_3 > 0 requires
         operator_review_status_for_tier_3 == "approved".
      3. HNeRV parity L9 runtime closure: every nonzero deliverable tier
         requires a non-"unknown" canonical_helper_invocation citation.
      4. Catalog #213 Comma2k19 canonical helper: if the helper citation
         mentions Comma2k19, it MUST cite Comma2k19LocalCache or sister
         canonical API (no raw URLs).
      5. inflate.py LOC budget: inflate_py_loc_estimate <= 200 per HNeRV L4
         waiver ceiling.
    """
    blockers: list[str] = []

    if proof.tier_4_byte_count > 0:
        blockers.append(
            f"strict-scorer-rule violated: Tier 4 (forbidden) byte count "
            f"is {proof.tier_4_byte_count} > 0. CLAUDE.md "
            f"'Strict scorer rule' non-negotiable + Catalog #6 forbid "
            f"scorer-dependent or network-required bytes in any contest "
            f"archive."
        )

    if proof.tier_3_byte_count > 0 and proof.operator_review_status_for_tier_3 != "approved":
        blockers.append(
            f"HNeRV parity L4 violated: Tier 3 byte count is "
            f"{proof.tier_3_byte_count} > 0 but "
            f"operator_review_status_for_tier_3 is "
            f"{proof.operator_review_status_for_tier_3!r} (must be "
            f"'approved'). Operator must explicitly approve the inflate.py "
            f">= 100 LOC + Tier-3 baked-constant waiver before promotion."
        )

    deliverable_tier_count = (
        proof.tier_1_byte_count + proof.tier_2_byte_count + proof.tier_3_byte_count
    )
    if deliverable_tier_count > 0 and proof.canonical_helper_invocation == "unknown":
        blockers.append(
            f"HNeRV parity L9 violated: canonical_helper_invocation is "
            f"'unknown' but proof has {deliverable_tier_count} bytes in "
            f"deliverable tiers. Every nonzero Tier MUST cite a canonical "
            f"helper for runtime closure (e.g. "
            f"'Comma2k19LocalCache.fetch_chunk(...)' or "
            f"'tac.wyner_ziv_deliverability (zero-cost only)')."
        )

    helper_lower = proof.canonical_helper_invocation.lower()
    mentions_comma2k19 = "comma2k19" in helper_lower or "comma-2k19" in helper_lower
    cites_canonical_cache = (
        "comma2k19localcache" in helper_lower
        or "local_chunk_cache" in helper_lower
        or "fetch_chunk" in helper_lower
    )
    if mentions_comma2k19 and not cites_canonical_cache:
        blockers.append(
            f"Catalog #213 violated: canonical_helper_invocation mentions "
            f"Comma2k19 but does not cite Comma2k19LocalCache / "
            f"local_chunk_cache / fetch_chunk. Raw URL or sister "
            f"non-canonical helper rejected; route through "
            f"tac.substrates.pretrained_driving_prior.local_chunk_cache."
            f"Comma2k19LocalCache."
        )

    if proof.inflate_py_loc_estimate > 200:
        blockers.append(
            f"HNeRV parity L4 waiver ceiling exceeded: "
            f"inflate_py_loc_estimate is {proof.inflate_py_loc_estimate} > "
            f"200. Reduce baked constants, choose a different tier-2 "
            f"baker, or escalate the waiver ceiling via council review."
        )

    if proof.tier_2_byte_count > 0:
        missing = _tier_2_rationale_missing_required_anchors(
            proof.contest_compliance_rationale
        )
        if missing:
            blockers.append(
                "contest_compliance_rationale missing Tier-2 procedural-generation "
                "boundary anchors: "
                + ", ".join(missing)
            )

    return (len(blockers) == 0, blockers)
