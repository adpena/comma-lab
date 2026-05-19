# SPDX-License-Identifier: MIT
"""Canonical reader for per-byte sensitivity payloads from the master-gradient ledger.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE" + Catalog #125
6-hook wire-in non-negotiable + Catalog #335 cathedral auto-ingest paradigm
shift + Catalog #327 master-gradient consumers + the cathedral consumer
template per ``feedback_master_gradient_consumer_cathedral_wire_in_landed_
20260519.md`` (commit ``418698227``).

The master-gradient extractor (Catalog #327, ``tac.master_gradient``) emits
per-pair AND per-byte sensitivity signals into
``.omx/state/master_gradient_anchors.jsonl``. The 8 Cable D master-gradient
consumers wire per-pair signals into cathedral_autopilot via the canonical
``CathedralConsumerContract`` (Catalog #335). The per-byte sensitivity
signal sits in the same anchors with ``gradient_tensor_kind="aggregate_per_byte_v1"``
+ ``gradient_array_path`` pointing at an (N_bytes, 3) ``.npy`` array but no
cathedral consumer reads it. THIS helper is the canonical reader the new
per-byte cathedral consumer (``src/tac/cathedral_consumers/
per_byte_sensitivity_consumer/``) uses.

Per-byte sensitivity informs (per CLAUDE.md):

* Hook #1 sensitivity-map contribution (which bytes carry the score weight)
* Hook #3 bit-allocator (per-byte sensitivity → allocation priority order)
* Hook #4 cathedral autopilot dispatch (per-byte signal in candidate ranking)

Per CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden empirical-
claim-without-evidence-tag (the docstring-overstatement trap)" the canonical
``PerByteSensitivityPayload`` is observability-only: it carries the typed
contract that downstream consumers (cathedral autopilot ranker / Pareto
solver / bit-allocator) interpret. Score-claim adjustments derived from
per-byte sensitivity ALONE are not permitted (Catalog #287 / #323 phantom-
score discipline); the consumer's ``consume_candidate`` returns
``predicted_delta_adjustment=0.0`` always.

Anchor schema (``master_gradient_anchor_v1`` per ``tac.master_gradient``):

* ``archive_sha256`` (str)
* ``gradient_array_path`` (str; absolute or repo-relative path to .npy)
* ``gradient_tensor_kind`` ("aggregate_per_byte_v1" required here)
* ``n_bytes`` (int; matches loaded array first-axis length)
* ``operating_point`` (dict with d_seg/d_pose/rate/score)
* ``measurement_axis``, ``measurement_hardware``, ``measurement_method``
* ``measurement_utc``, ``written_at_utc``

Sister of:

* :mod:`tac.master_gradient` — producer of anchors + canonical loader
  (``latest_anchor_for_archive`` / ``load_anchors_strict`` / etc.)
* :mod:`tac.master_gradient_consumers` — per-pair consumer surfaces (Cable D)
* :mod:`tac.provenance.builders` — canonical Provenance contract per
  Catalog #323 (``build_provenance_for_predicted``)
* :mod:`tac.cathedral_consumers.per_byte_sensitivity_consumer` — the
  cathedral consumer that ingests payloads from this helper

Strict-load semantics (per Catalog #138):
``load_per_byte_sensitivity_for_archive(..., strict=True)`` raises
:class:`MasterGradientPerByteCorruptError` on JSONL parse failure or shape
mismatch between declared ``n_bytes`` and loaded array first axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance


class MasterGradientPerByteCorruptError(Exception):
    """Raised when a per-byte sensitivity payload cannot be loaded.

    Sister of :class:`tac.master_gradient.MasterGradientLedgerCorruptError`
    per Catalog #138 strict-load discipline. Distinct exception class so the
    caller can differentiate ledger-level corruption from per-byte payload-
    level corruption (shape mismatch / array-load failure / missing path).
    """


# Canonical grain taxonomy (mirrors ``MasterGradient.gradient_byte_domain``).
# Post-decompress grains operate at the CORRECT locality basis for
# entropy-coded archives (sister of slot 15
# ``MUTATION_GRAIN_POST_BROTLI_DECOMPRESS`` at
# ``tac.master_gradient_post_brotli_decompress``). Raw-byte grains carry
# entropy-cascade smearing risk per Catalog #318 + codex op7 finding
# 2026-05-19 (one raw-archive-byte mutation invalidates the entire
# downstream entropy stream so the per-byte gradient is NOT a local
# sensitivity at the raw-byte grain).
GRAIN_RAW_ARCHIVE_BYTE = "scored_archive_bytes"
GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD = "zip_inner_member_payload"
GRAIN_POST_BROTLI_DECOMPRESS = "post_brotli_decompress_decoder_weight_bytes"
GRAIN_POST_ARITHMETIC_DECOMPRESS = "post_arithmetic_decompress_decoder_weight_bytes"
GRAIN_POST_DECOMPRESS_GENERIC = "post_decompress_decoder_weight_bytes"
GRAIN_DIAGNOSTIC = "diagnostic_grade"

# Grains whose locality is preserved through the inflate path (a single
# byte mutation in this domain changes ONE downstream weight byte). The
# consumer prefers these when both raw-byte and post-decompress anchors
# exist for an archive.
POST_DECOMPRESS_GRAINS: tuple[str, ...] = (
    GRAIN_POST_BROTLI_DECOMPRESS,
    GRAIN_POST_ARITHMETIC_DECOMPRESS,
    GRAIN_POST_DECOMPRESS_GENERIC,
)

# Raw-archive-byte grains: entropy-cascade-smeared per Catalog #318 +
# codex op7 finding. Consumer falls back to these only when no
# post-decompress anchor exists AND ``fallback_to_raw_byte=True``; the
# verdict carries an explicit ``cascade_smearing_risk`` warning.
RAW_BYTE_GRAINS: tuple[str, ...] = (
    GRAIN_RAW_ARCHIVE_BYTE,
    GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD,
)


@dataclass(frozen=True)
class PerByteSensitivityPayload:
    """Typed per-byte sensitivity payload extracted from a master-gradient anchor.

    Frozen for immutability (no in-place mutation after extraction). All
    fields are pure-Python so the dataclass is JSON-serializable for ledger
    persistence (no numpy arrays held by-reference here; the consumer reads
    summary stats and computes top-K indices on demand).

    Fields
    ------
    archive_sha256
        Archive this payload describes. Must be lowercase hex.
    gradient_array_path
        Resolved absolute path to the (N_bytes, 3) ``.npy`` array.
    n_bytes
        Declared byte count; matches array first-axis length.
    measurement_axis
        Canonical axis tag (e.g. ``[contest-CPU]`` / ``[macOS-CPU advisory]``).
        Honored downstream by the cathedral consumer's axis routing.
    measurement_hardware
        Canonical hardware substrate (``darwin_arm64_*`` / ``linux_x86_64_*``).
    measurement_method
        Canonical method tag (``autograd_per_parameter_*`` / etc.).
    measurement_utc
        ISO-UTC timestamp when the anchor was extracted.
    top_k_sensitivity_indices
        Pre-computed top-K byte indices ordered by descending absolute
        sensitivity (L1-sum-of-abs across 3 score axes per the canonical
        ``_compute_aggregate_per_byte_importance`` in master_gradient_consumers).
        Empty tuple when ``top_k=0`` or array load fails.
    aggregate_l1_importance_sum
        Sum of L1-importance across ALL bytes (diagnostic; lets consumers
        normalize without re-loading the array).
    n_bytes_above_zero
        Count of bytes with non-zero aggregate importance (sparsity proxy).
    gradient_byte_domain
        Canonical grain identifier mirroring
        ``MasterGradient.gradient_byte_domain``. One of the canonical
        GRAIN_* constants. Defaults to GRAIN_RAW_ARCHIVE_BYTE for backward
        compatibility with pre-grain anchors.
    cascade_smearing_risk
        True when ``gradient_byte_domain`` is a raw-byte grain
        (entropy-coded archive). False when post-decompress. Operator-
        facing observability flag per Catalog #305 + #318 — does NOT
        change the verdict ``predicted_delta_adjustment`` (Catalog
        #287/#323) but WARNS the consumer that the per-byte sensitivity
        is subject to entropy-decoder cascade and should not be treated
        as a true local derivative without a paired post-decompress
        anchor cross-check.
    """

    archive_sha256: str
    gradient_array_path: str
    n_bytes: int
    measurement_axis: str
    measurement_hardware: str
    measurement_method: str
    measurement_utc: str
    top_k_sensitivity_indices: tuple[int, ...] = field(default_factory=tuple)
    aggregate_l1_importance_sum: float = 0.0
    n_bytes_above_zero: int = 0
    gradient_byte_domain: str = GRAIN_RAW_ARCHIVE_BYTE
    cascade_smearing_risk: bool = False

    def __post_init__(self) -> None:
        if not self.archive_sha256:
            raise ValueError("archive_sha256 must be non-empty")
        if not self.gradient_array_path:
            raise ValueError("gradient_array_path must be non-empty")
        if self.n_bytes <= 0:
            raise ValueError(f"n_bytes must be positive, got {self.n_bytes}")
        if not isinstance(self.top_k_sensitivity_indices, tuple):
            raise TypeError(
                "top_k_sensitivity_indices must be a tuple, got "
                f"{type(self.top_k_sensitivity_indices).__name__}"
            )
        for i, idx in enumerate(self.top_k_sensitivity_indices):
            if not isinstance(idx, int):
                raise TypeError(
                    f"top_k_sensitivity_indices[{i}] must be int, got "
                    f"{type(idx).__name__}"
                )
            if not (0 <= idx < self.n_bytes):
                raise ValueError(
                    f"top_k_sensitivity_indices[{i}]={idx} out of range "
                    f"[0, {self.n_bytes})"
                )
        if self.aggregate_l1_importance_sum < 0:
            raise ValueError(
                "aggregate_l1_importance_sum must be non-negative, got "
                f"{self.aggregate_l1_importance_sum}"
            )
        if not (0 <= self.n_bytes_above_zero <= self.n_bytes):
            raise ValueError(
                f"n_bytes_above_zero={self.n_bytes_above_zero} must be in "
                f"[0, {self.n_bytes}]"
            )
        if not isinstance(self.gradient_byte_domain, str) or not self.gradient_byte_domain:
            raise ValueError(
                "gradient_byte_domain must be a non-empty string"
            )
        if not isinstance(self.cascade_smearing_risk, bool):
            raise TypeError(
                "cascade_smearing_risk must be a bool, got "
                f"{type(self.cascade_smearing_risk).__name__}"
            )
        # Invariant: cascade_smearing_risk MUST match grain classification.
        # Raw-byte / unknown grains carry cascade risk; post-decompress
        # grains do not. Inconsistent pair is a hard error so the consumer
        # never lies about the source's locality.
        expected_risk = self.gradient_byte_domain not in POST_DECOMPRESS_GRAINS
        if self.cascade_smearing_risk != expected_risk:
            raise ValueError(
                f"cascade_smearing_risk={self.cascade_smearing_risk} inconsistent "
                f"with gradient_byte_domain={self.gradient_byte_domain!r}: "
                f"expected {expected_risk} per POST_DECOMPRESS_GRAINS "
                f"({POST_DECOMPRESS_GRAINS})"
            )


def _aggregate_l1_importance(arr) -> "object":
    """Canonical L1-sum-of-abs aggregator (sister of master_gradient_consumers).

    Mirrors ``_compute_aggregate_per_byte_importance`` per the Cable D pattern:
    input shape (N_bytes, 3) → output shape (N_bytes,) via L1-sum-of-abs.
    """
    import numpy as np  # local import per consumer pattern

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"per-byte gradient array must be shape (N_bytes, 3), got {arr.shape}"
        )
    return np.abs(arr).sum(axis=1)


def top_k_sensitive_byte_indices(
    payload: PerByteSensitivityPayload, k: int = 100
) -> list[int]:
    """Return top-K byte indices ranked by absolute sensitivity (descending).

    Re-loads the gradient array from ``payload.gradient_array_path`` and
    applies the canonical L1-sum-of-abs aggregator. Use the pre-computed
    ``payload.top_k_sensitivity_indices`` for the in-memory shortcut; this
    function is the on-demand re-computation path when the caller needs a
    different K than the one captured at payload-extraction time.

    Per Catalog #138 strict-load: returns ``[]`` if the array file is
    missing rather than raising, so the caller can route to the no-sensitivity
    branch. Raises :class:`MasterGradientPerByteCorruptError` only on shape
    mismatch (the array exists but is structurally broken).
    """
    if k <= 0:
        return []
    path = Path(payload.gradient_array_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return []
    try:
        import numpy as np
    except ImportError:
        return []
    try:
        arr = np.load(path)
    except (OSError, ValueError):
        return []
    if arr.shape[0] != payload.n_bytes:
        raise MasterGradientPerByteCorruptError(
            f"per-byte gradient array first-axis length {arr.shape[0]} != "
            f"declared n_bytes {payload.n_bytes} (path={path})"
        )
    importance = _aggregate_l1_importance(arr)
    # argsort descending; bounded by available bytes
    k_effective = min(int(k), int(arr.shape[0]))
    if k_effective <= 0:
        return []
    # np.argsort returns ascending; take last k_effective + reverse
    ascending = np.argsort(importance)
    top = ascending[-k_effective:][::-1]
    return [int(idx) for idx in top]


def load_per_byte_sensitivity_for_archive(
    archive_sha256: str,
    *,
    path: Path | None = None,
    axis: str | None = None,
    top_k: int = 100,
    strict: bool = False,
    prefer_grain: str = "post_decompress",
    fallback_to_raw_byte: bool = True,
) -> PerByteSensitivityPayload | None:
    """Load the latest per-byte sensitivity payload for an archive.

    Routes through :func:`tac.master_gradient.query_anchors_by_archive`
    (canonical loader; per Catalog #138 + #245 sister) filtering on
    ``gradient_tensor_kind="aggregate_per_byte_v1"`` + ``gradient_array_path``
    present. Picks the most-recent row by ``measurement_utc``, loads the
    declared (N_bytes, 3) numpy array, computes top-K + aggregate stats,
    and returns the typed :class:`PerByteSensitivityPayload`.

    Grain-aware routing (slot 6 + slot 10 grain awareness landing
    2026-05-19): sister of slot 15
    ``tac.master_gradient_post_brotli_decompress``. When BOTH raw-byte
    and post-decompress anchors exist for the same archive, the
    post-decompress anchor is the CORRECT locality basis (a single
    decompressed-byte flip changes ONE downstream weight byte). Raw-byte
    anchors are entropy-cascade-smeared per Catalog #318 + codex op7
    finding. The ``prefer_grain`` kwarg controls grain selection:

    * ``prefer_grain="post_decompress"`` (default) — pick the most-recent
      anchor in the post-decompress grain class first; fall back to
      raw-byte only when ``fallback_to_raw_byte=True`` AND no
      post-decompress anchor exists. The returned payload's
      ``cascade_smearing_risk`` flag mirrors the chosen grain.
    * ``prefer_grain="raw_byte"`` — pick raw-byte first (back-compat for
      callers that explicitly want the entropy-cascade-smeared signal).
    * ``prefer_grain="any"`` — pick the most-recent anchor regardless of
      grain (pre-grain-aware behavior).

    Returns ``None`` if no matching anchor exists (or if the gradient array
    file is missing and ``strict=False``). Returns ``None`` if numpy is not
    available (consumer is observability-only; refuses to fail dispatch
    just because numpy isn't installed in the caller's environment).

    Per Catalog #138 strict-load: when ``strict=True``, raises
    :class:`MasterGradientPerByteCorruptError` on shape mismatch (the
    array exists but is structurally broken).

    Per CLAUDE.md "Apples-to-apples evidence discipline": the ``axis``
    parameter filters by ``measurement_axis`` if set; default loads the
    most-recent row regardless of axis (so the consumer can surface the
    available signal with the correct ``axis_tag`` rather than fail-closed
    when only advisory anchors exist).
    """
    if not archive_sha256:
        return None

    try:
        from tac.master_gradient import query_anchors_by_archive
    except ImportError:  # pragma: no cover (always available in repo)
        return None

    if prefer_grain not in ("post_decompress", "raw_byte", "any"):
        raise ValueError(
            f"prefer_grain={prefer_grain!r} must be one of "
            "{'post_decompress', 'raw_byte', 'any'}"
        )

    rows = query_anchors_by_archive(archive_sha256, path=path)
    # Filter to per-byte aggregate anchors only.
    rows = [
        r
        for r in rows
        if r.get("gradient_tensor_kind", "aggregate_per_byte_v1")
        == "aggregate_per_byte_v1"
        and r.get("gradient_array_path")
    ]
    if axis is not None:
        rows = [r for r in rows if r.get("measurement_axis") == axis]
    if not rows:
        return None

    # Grain-aware row selection per the prefer_grain cascade.
    def _row_grain(row: dict) -> str:
        return str(row.get("gradient_byte_domain") or GRAIN_RAW_ARCHIVE_BYTE)

    post_decompress_rows = [
        r for r in rows if _row_grain(r) in POST_DECOMPRESS_GRAINS
    ]
    raw_byte_rows = [r for r in rows if _row_grain(r) in RAW_BYTE_GRAINS]
    other_rows = [
        r
        for r in rows
        if _row_grain(r) not in POST_DECOMPRESS_GRAINS
        and _row_grain(r) not in RAW_BYTE_GRAINS
    ]

    if prefer_grain == "post_decompress":
        if post_decompress_rows:
            candidate_rows = post_decompress_rows
        elif fallback_to_raw_byte and (raw_byte_rows or other_rows):
            candidate_rows = raw_byte_rows + other_rows
        else:
            return None
    elif prefer_grain == "raw_byte":
        if raw_byte_rows:
            candidate_rows = raw_byte_rows
        elif post_decompress_rows:
            candidate_rows = post_decompress_rows
        else:
            candidate_rows = other_rows or rows
    else:  # "any"
        candidate_rows = rows

    if not candidate_rows:
        return None

    # Latest by measurement_utc (chronological).
    latest = max(candidate_rows, key=lambda r: str(r.get("measurement_utc", "")))

    gradient_array_path_raw = str(latest.get("gradient_array_path") or "")
    if not gradient_array_path_raw:
        return None
    array_path = Path(gradient_array_path_raw)
    if not array_path.is_absolute():
        array_path = Path.cwd() / array_path

    n_bytes = int(latest.get("n_bytes", 0) or 0)
    if n_bytes <= 0:
        return None

    measurement_axis = str(latest.get("measurement_axis") or "[predicted]")
    measurement_hardware = str(latest.get("measurement_hardware") or "unknown")
    measurement_method = str(latest.get("measurement_method") or "unknown")
    measurement_utc = str(latest.get("measurement_utc") or "")

    # Compute aggregate stats + top-K (best-effort; payload still returns
    # when the array file is missing so the consumer can route via metadata).
    top_indices: tuple[int, ...] = ()
    aggregate_sum = 0.0
    n_above_zero = 0
    if array_path.is_file():
        try:
            import numpy as np  # local import per consumer pattern
        except ImportError:
            np = None  # type: ignore[assignment]
        if np is not None:
            try:
                arr = np.load(array_path)
            except (OSError, ValueError) as exc:
                if strict:
                    raise MasterGradientPerByteCorruptError(
                        f"failed to load gradient array {array_path}: {exc}"
                    ) from exc
                arr = None
            if arr is not None:
                if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] != n_bytes:
                    if strict:
                        raise MasterGradientPerByteCorruptError(
                            f"gradient array shape {arr.shape} != expected "
                            f"({n_bytes}, 3) (path={array_path})"
                        )
                else:
                    importance = np.abs(arr).sum(axis=1)
                    aggregate_sum = float(importance.sum())
                    n_above_zero = int((importance > 0).sum())
                    k_effective = max(0, min(int(top_k), n_bytes))
                    if k_effective > 0:
                        ascending = np.argsort(importance)
                        top = ascending[-k_effective:][::-1]
                        top_indices = tuple(int(idx) for idx in top)

    grain = str(latest.get("gradient_byte_domain") or GRAIN_RAW_ARCHIVE_BYTE)
    cascade_risk = grain not in POST_DECOMPRESS_GRAINS

    return PerByteSensitivityPayload(
        archive_sha256=archive_sha256,
        gradient_array_path=str(array_path),
        n_bytes=n_bytes,
        measurement_axis=measurement_axis,
        measurement_hardware=measurement_hardware,
        measurement_method=measurement_method,
        measurement_utc=measurement_utc,
        top_k_sensitivity_indices=top_indices,
        aggregate_l1_importance_sum=aggregate_sum,
        n_bytes_above_zero=n_above_zero,
        gradient_byte_domain=grain,
        cascade_smearing_risk=cascade_risk,
    )


def payload_provenance(payload: PerByteSensitivityPayload) -> "Provenance":
    """Build a canonical Provenance for this payload per Catalog #323.

    Always emits ``PREDICTED_FROM_MODEL`` grade — per-byte sensitivity is a
    derived diagnostic, not a contest-axis score claim. The consumer's
    ``consume_candidate`` uses this Provenance to populate the verdict's
    optional ``provenance`` field per the canonical CathedralConsumerContract.
    """
    from tac.provenance.builders import build_provenance_for_predicted

    return build_provenance_for_predicted(
        model_id="tac.master_gradient_per_byte_consumer.load_per_byte_sensitivity_for_archive",
        inputs_sha256=payload.archive_sha256,
        measurement_axis="[predicted]",
        hardware_substrate=payload.measurement_hardware,
        captured_at_utc=payload.measurement_utc or None,
    )


def summarize_payload(payload: PerByteSensitivityPayload) -> dict[str, object]:
    """Human-readable summary dict (for the cathedral consumer's rationale).

    Pure-Python dict; safe to embed in the consumer's verdict rationale
    string OR persist alongside the consumer's ConsumerVerdict.notes field.
    """
    return {
        "archive_sha256_prefix": payload.archive_sha256[:12],
        "n_bytes": payload.n_bytes,
        "n_bytes_above_zero": payload.n_bytes_above_zero,
        "sparsity_pct": (
            100.0 * (1.0 - payload.n_bytes_above_zero / payload.n_bytes)
            if payload.n_bytes > 0
            else 0.0
        ),
        "aggregate_l1_importance_sum": payload.aggregate_l1_importance_sum,
        "top_k_count": len(payload.top_k_sensitivity_indices),
        "measurement_axis": payload.measurement_axis,
        "measurement_hardware": payload.measurement_hardware,
        "measurement_utc": payload.measurement_utc,
        "gradient_byte_domain": payload.gradient_byte_domain,
        "cascade_smearing_risk": payload.cascade_smearing_risk,
    }


def available_grains_for_archive(
    archive_sha256: str,
    *,
    path: Path | None = None,
) -> dict[str, list[str]]:
    """Inventory available grains for an archive across the ledger.

    Returns a dict with three keys:

    * ``post_decompress`` — list of grain strings in POST_DECOMPRESS_GRAINS
      that have at least one anchor for this archive.
    * ``raw_byte`` — list of grain strings in RAW_BYTE_GRAINS with anchors.
    * ``other`` — any non-canonical grain strings (diagnostic / unknown).

    Used by:

    * Slot 10 xray ``--grain compare_both`` mode to decide whether the
      side-by-side cascade-smearing comparison plot can be emitted.
    * Cathedral consumer rationale to surface "BOTH grains available;
      using post_decompress" vs "ONLY raw_byte available;
      cascade_smearing_risk WARNED" so the operator sees the routing
      reasoning explicitly.

    Returns empty lists when no anchors exist for the archive (no
    exception).
    """
    if not archive_sha256:
        return {"post_decompress": [], "raw_byte": [], "other": []}
    try:
        from tac.master_gradient import query_anchors_by_archive
    except ImportError:  # pragma: no cover
        return {"post_decompress": [], "raw_byte": [], "other": []}
    rows = query_anchors_by_archive(archive_sha256, path=path)
    rows = [
        r
        for r in rows
        if r.get("gradient_tensor_kind", "aggregate_per_byte_v1")
        == "aggregate_per_byte_v1"
        and r.get("gradient_array_path")
    ]
    post: list[str] = []
    raw: list[str] = []
    other: list[str] = []
    for r in rows:
        grain = str(r.get("gradient_byte_domain") or GRAIN_RAW_ARCHIVE_BYTE)
        if grain in POST_DECOMPRESS_GRAINS:
            if grain not in post:
                post.append(grain)
        elif grain in RAW_BYTE_GRAINS:
            if grain not in raw:
                raw.append(grain)
        else:
            if grain not in other:
                other.append(grain)
    return {"post_decompress": post, "raw_byte": raw, "other": other}


__all__ = (
    "GRAIN_DIAGNOSTIC",
    "GRAIN_POST_ARITHMETIC_DECOMPRESS",
    "GRAIN_POST_BROTLI_DECOMPRESS",
    "GRAIN_POST_DECOMPRESS_GENERIC",
    "GRAIN_RAW_ARCHIVE_BYTE",
    "GRAIN_RAW_ZIP_INNER_MEMBER_PAYLOAD",
    "MasterGradientPerByteCorruptError",
    "POST_DECOMPRESS_GRAINS",
    "PerByteSensitivityPayload",
    "RAW_BYTE_GRAINS",
    "available_grains_for_archive",
    "load_per_byte_sensitivity_for_archive",
    "payload_provenance",
    "summarize_payload",
    "top_k_sensitive_byte_indices",
)
