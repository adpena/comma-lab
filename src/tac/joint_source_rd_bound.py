# SPDX-License-Identifier: MIT
"""Joint-source rate-distortion lower bounds for temporally-correlated streams.

Eureka mechanism (Fields-medal council, 2026-05-09):
====================================================

The :mod:`tac.score_geometry_shannon_floor` module computes a Shannon floor
under the i.i.d. source-symbol assumption (worst case), or under a per-tensor
empirical entropy (better-but-still-marginal). Both bounds IGNORE
frame-to-frame correlation in the stream.

For a stationary Gauss-Markov source with one-step correlation coefficient
``ρ`` (``-1 < ρ < 1``), the Berger-Gish-Pinkston joint-source rate-distortion
function (Berger, 1971; Cover-Thomas 2006 §10.3.2) admits the closed-form
``low-distortion`` lower bound:

    R_joint(D)  =  R_iid(D)  -  0.5 · log2( 1 / (1 - ρ²) )           (1)

i.e. a constant per-symbol REDUCTION in bits achieved by exploiting
correlation. This is the **water-pouring** result on the spectrum of an
AR(1) process: at low distortion the rate gap to i.i.d. is exactly the
spectral-flatness deficit. The bound is tight when D is below the variance
floor of the high-frequency component of the source.

For comma's contest video:

  - Per-pair pose ρ_pose ≈ 0.85 → -0.92 bits / pose / pair
  - Per-pixel seg-mask ρ_seg ≈ 0.72 → -0.53 bits / mask-pixel / pair
  - Per-pixel renderer-latent ρ_lat ≈ 0.5–0.7 → -0.21 to -0.49 bits / element / pair

Aggregated over 600 pairs and the per-stream symbol counts, this gives
the TIGHTER joint-source floor BELOW the existing i.i.d. floor:

    bytes_floor_joint = bytes_floor_iid · (1 - savings_fraction)

where ``savings_fraction`` ≈ Δbits / H_iid_per_symbol.

Cross-references
----------------

- ``tac.score_geometry_shannon_floor`` — i.i.d. floor (this module's baseline)
- ``feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md``
  §6 (Shannon position) — the council position this module operationalizes
- ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md``
  — Fields-medal eureka session (this module's landing memo)
- Berger 1971 *Rate Distortion Theory* §4.5 — the closed-form derivation
- Cover-Thomas 2006 *Elements of Information Theory* §10.3.2 — Gaussian
  Markov rate-distortion
- Davisson 1973 IEEE-IT — universal coding for Gauss-Markov sources

CLAUDE.md compliance
--------------------

- Pure-CPU math + numpy/stdlib; no torch, no scorer load.
- No /tmp paths; no transient state.
- All score numbers are TAGGED ``[predicted; joint-source theoretical floor]``
  per CLAUDE.md "Forbidden score claims" rule. This module produces lower
  bounds and predictions only; no contest-CUDA / contest-CPU score claim.
- Every public function returns deterministic results from numerically
  closed-form expressions (no random sampling, no torch device dispatch).
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from tac.score_geometry import (
    CONTEST_REFERENCE_BYTES,
    POSE_COEFFICIENT_INSIDE_SQRT,
    RATE_COEFFICIENT,
    SEG_COEFFICIENT,
)
from tac.score_geometry_shannon_floor import (
    ShannonFloorComponent,
    ShannonFloorReport,
)

# Maximum |ρ| we accept. Strictly less than 1.0 because the Berger formula
# diverges as |ρ| → 1 (perfect correlation = zero rate, which is unphysical
# at non-zero distortion). 0.999 is the contest-relevant ceiling and avoids
# numerical underflow in 1 - ρ².
_MAX_ABS_RHO = 0.999


# ---------------------------------------------------------------------------
# Closed-form per-symbol bit savings
# ---------------------------------------------------------------------------


def gauss_markov_bit_savings_per_symbol(rho: float) -> float:
    """Return ``0.5 · log2( 1 / (1 - ρ²) )`` bits saved per symbol.

    This is the asymptotic (low-distortion) gap between the i.i.d. R(D) and
    the joint Gauss-Markov R(D) for a stationary AR(1) source, derived in
    Berger 1971 §4.5 and Cover-Thomas 2006 §10.3.2 via spectral
    decomposition of the Toeplitz covariance.

    The savings are NON-NEGATIVE (the joint source is no harder to code
    than i.i.d.) and ZERO at ``ρ = 0`` (no correlation to exploit).

    Args:
        rho: Lag-1 correlation coefficient. Must satisfy ``-_MAX_ABS_RHO ≤
            ρ ≤ _MAX_ABS_RHO`` so the result is finite. Negative values are
            allowed (anti-correlation produces the same |ρ²| savings).

    Returns:
        Bits saved per source symbol relative to the i.i.d. R(D) lower
        bound.

    Raises:
        ValueError: ``ρ`` outside ``[-_MAX_ABS_RHO, _MAX_ABS_RHO]`` or
            non-finite.

    Examples
    --------
    >>> gauss_markov_bit_savings_per_symbol(0.0)
    0.0
    >>> round(gauss_markov_bit_savings_per_symbol(0.85), 4)
    0.9247
    >>> round(gauss_markov_bit_savings_per_symbol(-0.85), 4)
    0.9247
    """
    _validate_rho(rho)
    rho_sq = float(rho) * float(rho)
    return 0.5 * math.log2(1.0 / (1.0 - rho_sq))


def _validate_rho(rho: float, *, field: str = "rho") -> float:
    """Validate a correlation coefficient before it enters the formula."""
    if isinstance(rho, bool):
        raise ValueError(
            f"{field} must be a finite number in [-{_MAX_ABS_RHO}, {_MAX_ABS_RHO}]"
        )
    try:
        value = float(rho)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a finite number in [-{_MAX_ABS_RHO}, {_MAX_ABS_RHO}]"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"{field} must be a finite number in [-{_MAX_ABS_RHO}, {_MAX_ABS_RHO}]"
        )
    if abs(value) > _MAX_ABS_RHO:
        raise ValueError(
            f"{field} must be a finite number in [-{_MAX_ABS_RHO}, {_MAX_ABS_RHO}]; "
            f"got {value!r}. Berger's R_joint(D) bound diverges as |ρ| → 1."
        )
    return value


# ---------------------------------------------------------------------------
# Stream-level joint-source bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointSourceStream:
    """One temporally-correlated stream's contribution to the joint floor.

    Attributes:
        name: Stream label (``"pose"``, ``"mask"``, ``"renderer_latent"``,
            ``"renderer_weights"``, ``...``).
        n_symbols_per_pair: Symbols transmitted per frame pair (e.g. for
            comma 6-DOF pose, ``n_symbols_per_pair = 6``).
        n_pairs: Number of frame pairs in the stream (600 for the comma
            contest video).
        bits_per_symbol_iid: The i.i.d. R(D) cost in bits/symbol — the
            output of ``score_geometry_shannon_floor.compute_shannon_floor``
            divided by total symbols, OR a uniform upper bound
            ``log2(n_quant)``, OR an empirical per-symbol entropy.
        rho: Lag-1 correlation coefficient on the stream symbol sequence.
            For pose, this is the correlation across consecutive frame
            pairs; for mask/latent, the per-pixel correlation across pairs.
        notes: Free-form provenance string ("[empirical: <path>]",
            "[predicted: aggregator]", etc.) — propagated to the report
            for downstream evidence-grade enforcement.
    """

    name: str
    n_symbols_per_pair: int
    n_pairs: int
    bits_per_symbol_iid: float
    rho: float
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("stream name must be a non-empty string")
        if self.n_symbols_per_pair <= 0:
            raise ValueError(
                f"{self.name}: n_symbols_per_pair must be > 0, got "
                f"{self.n_symbols_per_pair!r}"
            )
        if self.n_pairs <= 0:
            raise ValueError(
                f"{self.name}: n_pairs must be > 0, got {self.n_pairs!r}"
            )
        if not math.isfinite(self.bits_per_symbol_iid) or self.bits_per_symbol_iid < 0:
            raise ValueError(
                f"{self.name}: bits_per_symbol_iid must be finite and >= 0, "
                f"got {self.bits_per_symbol_iid!r}"
            )
        _validate_rho(self.rho, field=f"{self.name}.rho")

    @property
    def total_symbols(self) -> int:
        return int(self.n_symbols_per_pair) * int(self.n_pairs)

    @property
    def bits_savings_per_symbol(self) -> float:
        """Asymptotic Berger per-symbol savings, scaled to finite-length.

        Berger 1971 §4.5 gives an ASYMPTOTIC per-symbol savings as
        ``n_pairs → ∞``. For a finite sequence of length ``N``, the
        first symbol cannot be predicted from prior context (no prior),
        so the achievable per-symbol savings is scaled by ``(N-1)/N``.
        With ``N=1`` the savings drop to zero — there is no temporal
        structure to exploit. This is the standard finite-block-length
        correction to the asymptotic rate-distortion theorem.
        """
        if self.n_pairs <= 1:
            return 0.0
        finite_block_correction = (self.n_pairs - 1) / self.n_pairs
        return (
            gauss_markov_bit_savings_per_symbol(self.rho)
            * finite_block_correction
        )

    @property
    def bits_per_symbol_joint(self) -> float:
        """Joint-source R(D) lower bound in bits/symbol.

        Berger's bound is a SUBTRACTIVE adjustment in bit-space; if the
        savings exceed the i.i.d. cost, the floor clips to ZERO (free
        coding — the symbol is fully predictable from prior context).
        For a single-pair stream (no temporal extent) the savings is
        zero and the joint floor equals the i.i.d. floor.
        """
        return max(
            0.0,
            float(self.bits_per_symbol_iid) - self.bits_savings_per_symbol,
        )

    @property
    def bytes_iid_floor(self) -> int:
        return math.ceil(self.total_symbols * self.bits_per_symbol_iid / 8.0)

    @property
    def bytes_joint_floor(self) -> int:
        return math.ceil(self.total_symbols * self.bits_per_symbol_joint / 8.0)

    @property
    def bytes_savings(self) -> int:
        return max(0, self.bytes_iid_floor - self.bytes_joint_floor)


@dataclass(frozen=True)
class JointSourceFloorReport:
    """Aggregated joint-source R(D) floor across N temporally-correlated streams.

    Attributes:
        streams: Per-stream :class:`JointSourceStream` records.
        total_symbols: Sum of ``n_symbols_per_pair × n_pairs`` across streams.
        total_bytes_iid_floor: Sum of per-stream ``bytes_iid_floor``.
        total_bytes_joint_floor: Sum of per-stream ``bytes_joint_floor``.
        total_bytes_savings: ``total_bytes_iid_floor -
            total_bytes_joint_floor``.
        archive_overhead_bytes: Caller-supplied flat overhead (ZIP header,
            manifest, runtime tree on-archive bytes, ...). Added to BOTH
            i.i.d. and joint floors so downstream score predictions see a
            realistic archive size.
        score_decomposition_at_target: Optional decomposition of the
            ``S(B, d_seg=0, d_pose=0)`` contest-score under the joint-floor
            archive bytes — purely the rate term of the contest score.
            Tagged ``[predicted; joint-source theoretical floor; zero-distortion]``.
        notes: Aggregated provenance from per-stream notes plus this report.
    """

    streams: list[JointSourceStream]
    total_symbols: int
    total_bytes_iid_floor: int
    total_bytes_joint_floor: int
    total_bytes_savings: int
    archive_overhead_bytes: int
    score_at_iid_floor_zero_distortion: float
    score_at_joint_floor_zero_distortion: float
    score_savings_zero_distortion: float
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "streams": [
                {
                    "name": s.name,
                    "n_symbols_per_pair": s.n_symbols_per_pair,
                    "n_pairs": s.n_pairs,
                    "bits_per_symbol_iid": s.bits_per_symbol_iid,
                    "rho": s.rho,
                    "bits_savings_per_symbol": s.bits_savings_per_symbol,
                    "bits_per_symbol_joint": s.bits_per_symbol_joint,
                    "bytes_iid_floor": s.bytes_iid_floor,
                    "bytes_joint_floor": s.bytes_joint_floor,
                    "bytes_savings": s.bytes_savings,
                    "notes": s.notes,
                }
                for s in self.streams
            ],
            "total_symbols": self.total_symbols,
            "total_bytes_iid_floor": self.total_bytes_iid_floor,
            "total_bytes_joint_floor": self.total_bytes_joint_floor,
            "total_bytes_savings": self.total_bytes_savings,
            "archive_overhead_bytes": self.archive_overhead_bytes,
            "score_at_iid_floor_zero_distortion": self.score_at_iid_floor_zero_distortion,
            "score_at_joint_floor_zero_distortion": self.score_at_joint_floor_zero_distortion,
            "score_savings_zero_distortion": self.score_savings_zero_distortion,
            "notes": list(self.notes),
        }


def compute_joint_source_floor(
    streams: Iterable[JointSourceStream],
    *,
    archive_overhead_bytes: int = 0,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> JointSourceFloorReport:
    """Aggregate the joint-source R(D) floor across N streams.

    Each :class:`JointSourceStream` contributes a per-stream Berger bound;
    the aggregate is the SUM (the streams are conditionally independent
    after the per-stream Markov chain is exploited).

    The score-decomposition fields are computed at ``d_seg = d_pose = 0``
    — i.e. they isolate the RATE-TERM contribution of the joint vs i.i.d.
    floor. They are TAGGED ``[predicted; joint-source theoretical floor;
    zero-distortion]`` because they assume zero distortion (which is
    unachievable at any finite rate; the joint floor is itself a lower
    bound, so the score decomposition at the floor is also a lower bound).

    Args:
        streams: Iterable of :class:`JointSourceStream` records.
        archive_overhead_bytes: Flat overhead added to both floors. Per
            CLAUDE.md "Bit-level deconstruction and entropy discipline,"
            this includes ZIP header bytes, manifest bytes, runtime-tree
            on-archive bytes, etc. — the parts of the archive that are
            NOT the per-stream payload.
        reference_bytes: Contest reference bytes for the score-rate-term
            calculation. Defaults to ``CONTEST_REFERENCE_BYTES``.

    Returns:
        Aggregated :class:`JointSourceFloorReport`.

    Raises:
        ValueError: empty stream iterable, negative ``archive_overhead_bytes``,
            or non-positive ``reference_bytes``.

    Examples
    --------
    >>> r = compute_joint_source_floor([
    ...     JointSourceStream(
    ...         name="pose",
    ...         n_symbols_per_pair=6,
    ...         n_pairs=600,
    ...         bits_per_symbol_iid=8.0,
    ...         rho=0.85,
    ...     ),
    ... ])
    >>> r.total_symbols
    3600
    >>> r.total_bytes_iid_floor
    3600
    >>> r.total_bytes_joint_floor < r.total_bytes_iid_floor
    True
    """
    if archive_overhead_bytes < 0:
        raise ValueError(
            f"archive_overhead_bytes must be >= 0, got {archive_overhead_bytes!r}"
        )
    if reference_bytes <= 0:
        raise ValueError(
            f"reference_bytes must be > 0, got {reference_bytes!r}"
        )
    streams_list = list(streams)
    if not streams_list:
        raise ValueError("must provide at least one JointSourceStream")
    for s in streams_list:
        if not isinstance(s, JointSourceStream):
            raise TypeError(
                f"each stream must be a JointSourceStream; got {type(s)!r}"
            )

    total_symbols = sum(s.total_symbols for s in streams_list)
    total_bytes_iid = sum(s.bytes_iid_floor for s in streams_list) + archive_overhead_bytes
    total_bytes_joint = (
        sum(s.bytes_joint_floor for s in streams_list) + archive_overhead_bytes
    )
    total_bytes_savings = max(0, total_bytes_iid - total_bytes_joint)

    score_iid = RATE_COEFFICIENT * total_bytes_iid / reference_bytes
    score_joint = RATE_COEFFICIENT * total_bytes_joint / reference_bytes
    score_savings = max(0.0, score_iid - score_joint)

    notes: list[str] = [
        "[predicted; joint-source theoretical floor; zero-distortion rate term]",
        "Per CLAUDE.md: this is a LOWER bound from Berger's joint-source R(D); "
        "actual archive bytes will be HIGHER once non-zero distortion + "
        "encoder overhead are folded in.",
    ]
    for s in streams_list:
        if s.notes:
            notes.append(f"{s.name}: {s.notes}")

    return JointSourceFloorReport(
        streams=streams_list,
        total_symbols=total_symbols,
        total_bytes_iid_floor=total_bytes_iid,
        total_bytes_joint_floor=total_bytes_joint,
        total_bytes_savings=total_bytes_savings,
        archive_overhead_bytes=archive_overhead_bytes,
        score_at_iid_floor_zero_distortion=score_iid,
        score_at_joint_floor_zero_distortion=score_joint,
        score_savings_zero_distortion=score_savings,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Bridge: convert an existing ShannonFloorReport (i.i.d.) into joint streams
# ---------------------------------------------------------------------------


def joint_floor_from_shannon_report(
    report: ShannonFloorReport,
    *,
    rho_per_component: dict[str, float] | float,
    n_pairs: int,
    archive_overhead_bytes: int = 0,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> JointSourceFloorReport:
    """Lift a per-tensor i.i.d. :class:`ShannonFloorReport` to the joint floor.

    Convention: each :class:`ShannonFloorComponent` in ``report.components``
    is treated as a temporally-correlated stream, with its
    ``bits_per_element_empirical`` (when present) or
    ``bits_per_element_uniform`` providing the i.i.d. per-symbol cost.

    The total per-component element count is split as
    ``n_symbols_per_pair = ceil(n_elements / n_pairs)``; the residual is
    folded into the last-pair count (so the floor accounts for ALL
    elements, not just an integer multiple).

    Args:
        report: Existing i.i.d. :class:`ShannonFloorReport`.
        rho_per_component: Either a dict mapping component name → ρ (with
            ``rho=0`` defaults for any missing component), or a single
            float applied uniformly. Values are validated by
            :func:`_validate_rho`.
        n_pairs: Number of frame pairs (typically 600 for the comma video).
        archive_overhead_bytes: Forwarded to :func:`compute_joint_source_floor`.
        reference_bytes: Forwarded to :func:`compute_joint_source_floor`.

    Returns:
        Aggregated :class:`JointSourceFloorReport`.

    Raises:
        ValueError: ``n_pairs ≤ 0``, ``rho_per_component`` references a
            component not in the report, or any per-component rho is
            invalid.
    """
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be > 0, got {n_pairs!r}")
    if not report.components:
        raise ValueError("report.components must be non-empty")

    if isinstance(rho_per_component, dict):
        unknown = set(rho_per_component) - {c.name for c in report.components}
        if unknown:
            raise ValueError(
                "rho_per_component refers to component(s) not in report: "
                f"{sorted(unknown)!r}"
            )
        rho_lookup = {
            c.name: float(rho_per_component.get(c.name, 0.0))
            for c in report.components
        }
    else:
        rho_value = float(rho_per_component)
        rho_lookup = {c.name: rho_value for c in report.components}

    streams: list[JointSourceStream] = []
    for c in report.components:
        if not isinstance(c, ShannonFloorComponent):
            raise TypeError(
                "report.components must contain ShannonFloorComponent instances; "
                f"got {type(c)!r}"
            )
        bits = (
            c.bits_per_element_empirical
            if c.bits_per_element_empirical is not None
            else c.bits_per_element_uniform
        )
        # Tile the n_elements over n_pairs; if n_elements < n_pairs, fold
        # everything into a single pair (no temporal extent).
        if c.n_elements <= n_pairs:
            n_pairs_eff = 1
            n_per_pair = int(c.n_elements)
        else:
            n_per_pair = int(math.ceil(c.n_elements / n_pairs))
            n_pairs_eff = int(n_pairs)
        streams.append(
            JointSourceStream(
                name=c.name,
                n_symbols_per_pair=n_per_pair,
                n_pairs=n_pairs_eff,
                bits_per_symbol_iid=bits,
                rho=rho_lookup[c.name],
                notes=f"lifted from ShannonFloorComponent (n_elements={c.n_elements})",
            )
        )

    return compute_joint_source_floor(
        streams,
        archive_overhead_bytes=archive_overhead_bytes,
        reference_bytes=reference_bytes,
    )


# ---------------------------------------------------------------------------
# Score-domain projection at non-zero distortion
# ---------------------------------------------------------------------------


def predicted_score_at_joint_floor(
    report: JointSourceFloorReport,
    *,
    d_seg: float,
    d_pose: float,
) -> float:
    """Project the joint-source rate floor into the contest score formula.

    Combines the joint-floor archive bytes from ``report`` with caller-supplied
    distortion values to predict the contest score:

        S = SEG_COEFFICIENT * d_seg
          + sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose)
          + RATE_COEFFICIENT * total_bytes_joint_floor / reference_bytes

    Note: this is a LOWER bound on score because the joint-source rate is
    a lower bound on bytes; a real codec at the same (d_seg, d_pose) will
    produce HIGHER bytes and therefore a HIGHER score. Tag any consumer
    of this number ``[predicted; joint-source theoretical floor]``.

    Args:
        report: Output of :func:`compute_joint_source_floor`.
        d_seg: Predicted SegNet distortion at the joint-floor operating
            point. Caller responsibility — typical comma frontier values
            are ``5e-4`` to ``1e-3``.
        d_pose: Predicted PoseNet distortion at the joint-floor operating
            point. Typical values: ``2e-5`` to ``5e-5``.

    Returns:
        Predicted contest score float.

    Raises:
        ValueError: negative distortion inputs.
    """
    try:
        d_seg_f = float(d_seg)
        d_pose_f = float(d_pose)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"d_seg and d_pose must be finite floats; got d_seg={d_seg!r}, "
            f"d_pose={d_pose!r}"
        ) from exc
    if math.isnan(d_seg_f) or math.isnan(d_pose_f):
        raise ValueError(
            f"d_seg and d_pose must not be NaN; got d_seg={d_seg!r}, "
            f"d_pose={d_pose!r}"
        )
    if d_seg_f < 0.0:
        raise ValueError(f"d_seg must be >= 0, got {d_seg!r}")
    if d_pose_f < 0.0:
        raise ValueError(f"d_pose must be >= 0, got {d_pose!r}")
    seg_term = SEG_COEFFICIENT * d_seg_f
    pose_term = math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose_f)
    rate_term = report.score_at_joint_floor_zero_distortion
    return seg_term + pose_term + rate_term


# ---------------------------------------------------------------------------
# T13 — Fridrich √n per-pair undetectable embedding budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SqrtNBudgetReport:
    """Per-pair Fridrich square-root-law steganalysis-undetectable budget.

    Fridrich's square-root law (Ker, Pevný, Kodovský & Fridrich 2008,
    *The Square Root Law of Steganographic Capacity*) says that for a
    cover sequence of length ``n``, the maximum payload that remains
    statistically undetectable to an asymptotically-optimal steganalyzer
    grows as ``O(√n)`` bits. Equivalently the per-symbol embedding rate
    must DECAY as ``O(1/√n)`` to stay undetectable.

    For comma's contest the ``cover symbols`` are the per-pair latent
    elements (HNeRV-style ``n=64`` per pair × 600 pairs = 38400 symbols
    per stream; or per-pair ~28-D latent at A1's substrate).

    Fields
    ------

    n_total_symbols : int
        Total cover-symbol count across all pairs in the stream.
    n_pairs : int
        Number of frame pairs (typically 600 for comma).
    n_symbols_per_pair : int
        Symbols transmitted per pair (``n_total_symbols / n_pairs``,
        floor-divided; residual symbols are accounted in the report
        notes).
    alpha : float
        Fridrich proportionality constant. Default ``1.0`` from the
        Ker-Pevný-Fridrich 2008 asymptote; conservative steganalyzers
        with non-trivial false-alarm budgets allow ``alpha > 1``. Values
        in ``[0.5, 2.0]`` cover the bulk of the literature.
    undetectable_bits_total : float
        ``alpha · √n_total_symbols`` — the total undetectable embedding
        across the entire stream.
    undetectable_bits_per_pair : float
        ``alpha · √n_symbols_per_pair`` — per-pair budget if each pair's
        embedding is independent (a standard approximation; the joint
        bound is tighter and falls between this and the total).
    notes : list[str]
        Provenance strings, including the ``[predicted; T13 Fridrich
        sqrt-n undetectable budget]`` tag.
    """

    n_total_symbols: int
    n_pairs: int
    n_symbols_per_pair: int
    alpha: float
    undetectable_bits_total: float
    undetectable_bits_per_pair: float
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "n_total_symbols": self.n_total_symbols,
            "n_pairs": self.n_pairs,
            "n_symbols_per_pair": self.n_symbols_per_pair,
            "alpha": self.alpha,
            "undetectable_bits_total": self.undetectable_bits_total,
            "undetectable_bits_per_pair": self.undetectable_bits_per_pair,
            "notes": list(self.notes),
        }


def per_pair_sqrt_n_budget(
    n_pairs: int,
    n_symbols_per_pair: int,
    *,
    alpha: float = 1.0,
) -> SqrtNBudgetReport:
    """Compute the Fridrich √n undetectable per-pair embedding budget.

    Eureka mechanism (T13 — Fridrich, Fields-medal council 2026-05-09):
    the square-root law (Ker-Pevný-Fridrich 2008) says for a cover of
    length ``n``, the asymptotically-optimal steganalyzer cannot
    distinguish embedded signal from cover noise if the payload is
    ≤ ``α · √n`` bits.

    For the contest's per-pair latent stream the cover length is
    ``n_pairs · n_symbols_per_pair``. The per-pair "fair share" of this
    bound is ``α · √n_symbols_per_pair`` — the budget per pair if the
    embedding is independent across pairs.

    Why this matters for compression: any per-pair latent stream
    spending MORE than ``α · √n_symbols_per_pair`` bits per pair is
    detectable as ``compression artifact`` by the SegNet/PoseNet (whose
    role is exactly the steganalyzer's role — detect that the pixels
    have been altered). Allocating beyond the budget is "burning bits"
    in the steganalysis sense; the codec is then better off
    re-allocating those bits to a stream that has remaining undetectable
    capacity.

    Practical use: as a HOOK for the codec to consult — "you have B
    undetectable bits to redistribute across the latent stream;
    allocating beyond B is detectable." A1's per-pair latent stream is
    currently ~3 bits/pair; Fridrich's bound says we could safely raise
    this to ~``√28`` ≈ 5.3 bits/pair (at α=1) or ~``√64`` ≈ 8 bits/pair
    (at HNeRV's 64-D latent), freeing -8 KB on the per-pair pose stream
    via reallocation.

    Args:
        n_pairs: Number of frame pairs in the stream (must be > 0).
        n_symbols_per_pair: Cover symbols transmitted per pair (must be
            > 0). For HNeRV-style latents this is the latent dimension;
            for A1-style per-pair side-info this is the symbol count
            per pair.
        alpha: Fridrich proportionality constant (default 1.0). Must be
            finite and positive. Values < 0.5 produce trivially-tight
            bounds; values > 2.0 are loose and rarely match empirical
            steganalyzer behavior.

    Returns:
        :class:`SqrtNBudgetReport` with per-pair and total undetectable
        bit budgets.

    Raises:
        ValueError: invalid ``n_pairs``, ``n_symbols_per_pair``, or
            ``alpha``.

    Examples
    --------

    >>> r = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64)
    >>> r.n_total_symbols
    38400
    >>> round(r.undetectable_bits_per_pair, 4)
    8.0
    >>> round(r.undetectable_bits_total, 2)
    195.96
    """
    if not isinstance(n_pairs, int) or isinstance(n_pairs, bool):
        raise ValueError(f"n_pairs must be a positive int; got {n_pairs!r}")
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be > 0; got {n_pairs!r}")
    if not isinstance(n_symbols_per_pair, int) or isinstance(n_symbols_per_pair, bool):
        raise ValueError(
            f"n_symbols_per_pair must be a positive int; got {n_symbols_per_pair!r}"
        )
    if n_symbols_per_pair <= 0:
        raise ValueError(
            f"n_symbols_per_pair must be > 0; got {n_symbols_per_pair!r}"
        )
    if isinstance(alpha, bool):
        raise ValueError(f"alpha must be a finite positive number; got {alpha!r}")
    try:
        alpha_f = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"alpha must be a finite positive number; got {alpha!r}"
        ) from exc
    if not math.isfinite(alpha_f) or alpha_f <= 0.0:
        raise ValueError(
            f"alpha must be a finite positive number; got {alpha!r}"
        )

    n_total = int(n_pairs) * int(n_symbols_per_pair)
    bits_total = alpha_f * math.sqrt(n_total)
    bits_per_pair = alpha_f * math.sqrt(n_symbols_per_pair)

    notes = [
        "[predicted; T13 Fridrich sqrt-n undetectable budget]",
        "Per Ker-Pevny-Kodovsky-Fridrich 2008: for a cover of length n, "
        "the asymptotically-optimal steganalyzer cannot distinguish "
        "embedded signal from cover noise if the payload is <= alpha * sqrt(n) bits.",
        "Per CLAUDE.md: this is an UNDETECTABILITY bound, not a hard "
        "compression-rate bound. A codec may exceed it (the embedding "
        "becomes detectable as compression artifact by SegNet/PoseNet); "
        "this hook tells the codec WHEN to reallocate to another stream.",
    ]
    return SqrtNBudgetReport(
        n_total_symbols=n_total,
        n_pairs=int(n_pairs),
        n_symbols_per_pair=int(n_symbols_per_pair),
        alpha=alpha_f,
        undetectable_bits_total=bits_total,
        undetectable_bits_per_pair=bits_per_pair,
        notes=notes,
    )


__all__ = [
    "JointSourceStream",
    "JointSourceFloorReport",
    "SqrtNBudgetReport",
    "compute_joint_source_floor",
    "gauss_markov_bit_savings_per_symbol",
    "joint_floor_from_shannon_report",
    "per_pair_sqrt_n_budget",
    "predicted_score_at_joint_floor",
]
