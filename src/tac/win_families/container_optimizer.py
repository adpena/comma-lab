# SPDX-License-Identifier: MIT
"""F3 -- the CONTAINER / RE-ENCODE TRANSFORM OPTIMIZER, at the ARCHIVE layer.

THE FAMILY, AND WHAT IT IS WORTH
--------------------------------
Three arms won bytes without touching a single distortion term::

    ddm_ck2   -657 B   2-plane carrier interleave, parameter-free, receiver cost 0 counted bytes
    ddm_to1   -105 B   tail override (ma1's within-miss law spliced onto the live body)
    ddm_up3      0 B   absorbed a +48 B brotli cost so a pose re-solve shipped at DB = 0

762 B at zero distortion is **-5.074e-04 S** at 25/37,545,489 per byte.  That is the
family: the payload is fixed, and the only question is how the container carries it.

THE LAW THAT MAKES THIS ITS OWN FAMILY
--------------------------------------
**Archive delta is not payload delta.**  ``ddm_up3`` re-solved 600 carrier coefficients,
which moved the Rice payload by **+7 bits** -- under one byte -- and the archive grew
**+48 B**.  Nothing about the payload explains that; brotli's parse of a different byte
sequence does.  The same arm then recovered the whole 48 B by turning the CK2 interleave
OFF and dropping to ``q10/lgwin16``, landing at **DB = 0** on a strictly better payload.

So a rate claim measured on payload length is not a rate claim.  The counted quantity is
``archive.zip``'s own size, and this module refuses to report anything else.

THE LAUNDERING RISK, AND THE THREE THINGS THAT CONTAIN IT
---------------------------------------------------------
``ddm_up3`` named the risk in its own memo: searching container parameters until the
number improves is how a rate claim gets laundered.  A search over a space you are free
to grow after seeing results has no honest baseline.  The containment is structural here:

1. **The space is DECLARED and SEALED.**  :class:`ContainerSpace` hashes its option list
   at construction and every receipt carries that ``seal_digest``.  Growing the space
   changes the digest, so a later, larger search cannot be quietly compared against an
   earlier claim.
2. **Ties go to the incumbent.**  The shipped container shape is index 0 and wins every
   tie, so noise can never manufacture a "win" over the object that already ships.
3. **A candidate must PARSE BACK before it may win.**  Every config is verified to decode
   to exactly the payload it claims to carry.  A container that loses information is not
   a smaller archive, it is a broken one.

None of the three is a comment: each is enforced in :func:`search_container_space`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tac import contest_score

__all__ = [
    "CONTEST_UNCOMPRESSED_BYTES",
    "ContainerCandidate",
    "ContainerConfig",
    "ContainerDeltaReport",
    "ContainerOptimizerError",
    "ContainerSearchResult",
    "ContainerSpace",
    "archive_delta_report",
    "bytes_to_score",
    "search_container_space",
]

#: Re-exported from the canonical helper so no second denominator exists in the repo.
CONTEST_UNCOMPRESSED_BYTES = contest_score.UNCOMPRESSED_SIZE_BYTES


class ContainerOptimizerError(RuntimeError):
    """A container-search precondition failed.  Always fail closed."""


def bytes_to_score(delta_bytes: int | float) -> float:
    """Convert an ARCHIVE byte delta into score units.  Negative bytes = negative score.

    Delegates to :func:`tac.contest_score.rate_term`, which owns the weight, the
    denominator, and upstream's exact binary64 association order.  ``rate_term`` takes a
    non-negative size, so the sign is carried outside the magnitude rather than by
    re-deriving the arithmetic here.
    """
    magnitude = contest_score.rate_term(abs(float(delta_bytes)))
    return -magnitude if delta_bytes < 0 else magnitude


# ---------------------------------------------------------------------------
# The declared option space.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContainerConfig:
    """One point in the container transform space.

    Every field is an ENCODER-ONLY choice: it changes how the same payload is written,
    never what the payload is.  That is what makes the family distortion-free by
    construction rather than by measurement.
    """

    #: CK2 2-plane carrier interleave (``reserved & 0x4``).  ``ddm_ck2`` won -657 B by
    #: turning it ON for the semantic body; ``ddm_up3`` won 48 B back by turning it OFF
    #: for a re-solved carrier.  Same knob, opposite sign, different payload -- which is
    #: exactly why it must be searched per body and never carried as a constant.
    interleave: bool = True
    brotli_quality: int = 11
    brotli_lgwin: int = 24
    #: Optional explicit section order.  ``None`` keeps the shipped order.
    section_order: tuple[str, ...] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.brotli_quality <= 11:
            raise ContainerOptimizerError(
                f"brotli quality must be in [0, 11], got {self.brotli_quality}"
            )
        if not 10 <= self.brotli_lgwin <= 24:
            raise ContainerOptimizerError(
                f"brotli lgwin must be in [10, 24], got {self.brotli_lgwin}"
            )
        if self.section_order is not None and len(set(self.section_order)) != len(
            self.section_order
        ):
            raise ContainerOptimizerError(
                f"section_order repeats a section: {self.section_order}"
            )

    def describe(self) -> str:
        name = self.label or "cfg"
        order = "shipped" if self.section_order is None else "/".join(self.section_order)
        return (
            f"{name}[interleave={'on' if self.interleave else 'off'} "
            f"q={self.brotli_quality} lgwin={self.brotli_lgwin} order={order}]"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "interleave": self.interleave,
            "brotli_quality": self.brotli_quality,
            "brotli_lgwin": self.brotli_lgwin,
            "section_order": list(self.section_order) if self.section_order else None,
            "label": self.label,
        }


#: The option list ``ddm_up3`` declared and searched.  Index 0 is the shipped shape.
UP3_DECLARED_OPTIONS: tuple[ContainerConfig, ...] = (
    ContainerConfig(True, 11, 24, label="shipped"),
    ContainerConfig(True, 10, 16),
    ContainerConfig(True, 9, 16),
    ContainerConfig(False, 11, 24),
    ContainerConfig(False, 11, 10),
    ContainerConfig(False, 10, 16),
    ContainerConfig(False, 10, 10),
    ContainerConfig(False, 9, 16),
)


class ContainerSpace:
    """A DECLARED, SEALED option list.  Index ``incumbent_index`` is the shipped shape.

    The seal is the anti-laundering device.  ``seal_digest`` is the sha256 of the declared
    configs, so a receipt carrying it proves which space produced the winner.  A later,
    larger search has a different digest and therefore cannot be presented as the same
    experiment.
    """

    def __init__(
        self,
        configs: Sequence[ContainerConfig],
        *,
        incumbent_index: int = 0,
        name: str = "container_space",
    ) -> None:
        if not configs:
            raise ContainerOptimizerError("a container space needs at least one config")
        if not 0 <= incumbent_index < len(configs):
            raise ContainerOptimizerError(
                f"incumbent_index {incumbent_index} is outside the {len(configs)}-config space"
            )
        payloads = [json.dumps(c.to_json(), sort_keys=True) for c in configs]
        if len(set(payloads)) != len(payloads):
            raise ContainerOptimizerError(
                "the declared space contains duplicate configs; a duplicate silently "
                "doubles a config's chance of winning a tie-break"
            )
        self.configs = tuple(configs)
        self.incumbent_index = incumbent_index
        self.name = name
        self._digest = hashlib.sha256(
            json.dumps({"name": name, "configs": payloads}, sort_keys=True).encode()
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.configs)

    @property
    def seal_digest(self) -> str:
        """sha256 of the declared option list.  Growing the space changes this."""
        return self._digest

    @property
    def incumbent(self) -> ContainerConfig:
        return self.configs[self.incumbent_index]

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seal_digest": self.seal_digest,
            "size": len(self.configs),
            "incumbent_index": self.incumbent_index,
            "configs": [c.to_json() for c in self.configs],
        }


# ---------------------------------------------------------------------------
# Candidates and results.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContainerCandidate:
    """One compiled container, with its MEASURED archive size and its parse-back verdict."""

    config: ContainerConfig
    index: int
    archive_bytes: int
    archive_sha256: str
    parse_back_exact: bool
    error: str | None = None

    @property
    def admissible(self) -> bool:
        """A candidate may win only if it compiled and round-tripped exactly."""
        return self.error is None and self.parse_back_exact

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "config": self.config.to_json(),
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "parse_back_exact": self.parse_back_exact,
            "admissible": self.admissible,
            "error": self.error,
        }


@dataclass(frozen=True)
class ContainerSearchResult:
    """The search outcome, carrying the seal that proves which space produced it."""

    space_name: str
    seal_digest: str
    candidates: tuple[ContainerCandidate, ...]
    winner_index: int
    incumbent_index: int
    tie_broken_to_incumbent: bool

    @property
    def winner(self) -> ContainerCandidate:
        return self.candidates[self.winner_index]

    @property
    def incumbent(self) -> ContainerCandidate:
        return self.candidates[self.incumbent_index]

    @property
    def delta_archive_bytes(self) -> int:
        """MEASURED archive bytes saved vs the incumbent.  Negative = smaller = better."""
        return self.winner.archive_bytes - self.incumbent.archive_bytes

    @property
    def delta_score(self) -> float:
        """The rate-leg score delta.  Negative = better."""
        return bytes_to_score(self.delta_archive_bytes)

    @property
    def admissible_count(self) -> int:
        return sum(1 for candidate in self.candidates if candidate.admissible)

    def to_json(self) -> dict[str, Any]:
        return {
            "space_name": self.space_name,
            "seal_digest": self.seal_digest,
            "searched": len(self.candidates),
            "admissible": self.admissible_count,
            "winner": self.winner.to_json(),
            "incumbent_archive_bytes": self.incumbent.archive_bytes,
            "delta_archive_bytes": self.delta_archive_bytes,
            "delta_score": self.delta_score,
            "tie_broken_to_incumbent": self.tie_broken_to_incumbent,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "measurement_note": (
                "delta_archive_bytes is the MEASURED size of archive.zip, never a payload "
                "length. ddm_up3 measured a +7-bit payload change costing +48 archive bytes."
            ),
            "axis": "[macOS-CPU exact byte measurement]",
            "score_claim": False,
        }


# ---------------------------------------------------------------------------
# The search.
# ---------------------------------------------------------------------------


def search_container_space(
    space: ContainerSpace,
    *,
    compile_fn: Callable[[ContainerConfig], bytes],
    parse_back_fn: Callable[[bytes], Any],
    expected_payload: Any,
    payload_equal: Callable[[Any, Any], bool] | None = None,
    strict: bool = True,
) -> ContainerSearchResult:
    """Compile every declared config, verify parse-back, and pick the smallest.

    Args:
        space: the SEALED option list.
        compile_fn: build ``archive.zip`` bytes under one config.
        parse_back_fn: decode a built archive back to its payload, through the RECEIVER.
        expected_payload: what every candidate must decode to.
        payload_equal: equality test; defaults to ``==``.  Pass ``np.array_equal`` for
            array payloads, where ``==`` returns an array and is not a verdict.
        strict: when True (default) a compile failure is recorded and the config is
            simply inadmissible; the search continues.  The incumbent failing is always
            fatal, because then there is nothing honest to compare against.

    Returns:
        The result, with ties broken to the incumbent.

    Raises:
        ContainerOptimizerError: the incumbent failed, or nothing was admissible.
    """
    equal = payload_equal or (lambda a, b: bool(a == b))
    candidates: list[ContainerCandidate] = []

    for index, config in enumerate(space.configs):
        try:
            blob = compile_fn(config)
            if not isinstance(blob, (bytes, bytearray)):
                raise ContainerOptimizerError(
                    f"compile_fn returned {type(blob).__name__}, expected bytes"
                )
            blob = bytes(blob)
            recovered = parse_back_fn(blob)
            exact = bool(equal(expected_payload, recovered))
            candidates.append(
                ContainerCandidate(
                    config=config,
                    index=index,
                    archive_bytes=len(blob),
                    archive_sha256=hashlib.sha256(blob).hexdigest(),
                    parse_back_exact=exact,
                )
            )
        except Exception as error:
            if not strict and index == space.incumbent_index:
                raise
            candidates.append(
                ContainerCandidate(
                    config=config,
                    index=index,
                    archive_bytes=-1,
                    archive_sha256="",
                    parse_back_exact=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )

    incumbent = candidates[space.incumbent_index]
    if not incumbent.admissible:
        raise ContainerOptimizerError(
            f"the incumbent container {incumbent.config.describe()} did not compile and "
            f"round-trip ({incumbent.error}); without a working incumbent there is no "
            "honest baseline to measure a saving against"
        )
    admissible = [candidate for candidate in candidates if candidate.admissible]
    if not admissible:  # pragma: no cover - unreachable while the incumbent is checked
        raise ContainerOptimizerError("no admissible container candidate")

    # Ties go to the incumbent: strictly-smaller wins, and the incumbent's own index
    # breaks any remaining tie by sorting first.
    best = min(
        admissible,
        key=lambda candidate: (
            candidate.archive_bytes,
            0 if candidate.index == space.incumbent_index else 1,
            candidate.index,
        ),
    )
    tie_broken = (
        best.index != space.incumbent_index
        and best.archive_bytes == incumbent.archive_bytes
    )
    return ContainerSearchResult(
        space_name=space.name,
        seal_digest=space.seal_digest,
        candidates=tuple(candidates),
        winner_index=best.index,
        incumbent_index=space.incumbent_index,
        tie_broken_to_incumbent=tie_broken,
    )


# ---------------------------------------------------------------------------
# The archive-vs-payload law.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContainerDeltaReport:
    """The measured gap between what the payload did and what the archive did."""

    payload_delta_bits: int
    archive_delta_bytes: int

    @property
    def payload_delta_bytes_ceiling(self) -> int:
        """The payload change expressed in whole bytes, rounded away from zero."""
        bits = self.payload_delta_bits
        whole = abs(bits) // 8 + (1 if abs(bits) % 8 else 0)
        return whole if bits >= 0 else -whole

    @property
    def container_attributable_bytes(self) -> int:
        """Archive bytes NOT explained by the payload.  This is the container's own term."""
        return self.archive_delta_bytes - self.payload_delta_bytes_ceiling

    @property
    def law_holds(self) -> bool:
        """True when the container term is non-zero -- i.e. the two deltas differ.

        ``ddm_up3``'s measurement is the anchor: +7 payload bits, +48 archive bytes.
        """
        return self.container_attributable_bytes != 0

    def to_json(self) -> dict[str, Any]:
        return {
            "payload_delta_bits": self.payload_delta_bits,
            "payload_delta_bytes_ceiling": self.payload_delta_bytes_ceiling,
            "archive_delta_bytes": self.archive_delta_bytes,
            "container_attributable_bytes": self.container_attributable_bytes,
            "archive_delta_score": bytes_to_score(self.archive_delta_bytes),
            "law": (
                "archive delta is not payload delta; only the archive.zip stat is the "
                "counted quantity (ddm_up3: +7 payload bits cost +48 archive bytes)"
            ),
        }


def archive_delta_report(
    *, payload_delta_bits: int, archive_delta_bytes: int
) -> ContainerDeltaReport:
    """Attribute an archive delta between its payload term and its container term."""
    return ContainerDeltaReport(
        payload_delta_bits=int(payload_delta_bits),
        archive_delta_bytes=int(archive_delta_bytes),
    )


# ---------------------------------------------------------------------------
# The three controls, as typed results.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlResult:
    """One executable falsifier of a container build."""

    control: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"control": self.control, "passed": self.passed, **self.detail}


def identity_control(
    *,
    compile_fn: Callable[[ContainerConfig], bytes],
    shipped_config: ContainerConfig,
    expected_sha256: str,
) -> ControlResult:
    """Re-compile the SHIPPED payload under the SHIPPED config; bytes must be identical.

    This is the control that makes every later saving mean something.  If the builder
    cannot reproduce the object that already ships, a smaller number from the same
    builder is not a saving -- it is a different object.
    """
    blob = compile_fn(shipped_config)
    observed = hashlib.sha256(bytes(blob)).hexdigest()
    return ControlResult(
        control="identity_shipped_config_reproduces_shipped_archive",
        passed=observed == expected_sha256,
        detail={
            "expected_sha256": expected_sha256,
            "observed_sha256": observed,
            "observed_bytes": len(blob),
        },
    )


def determinism_control(
    *, compile_fn: Callable[[ContainerConfig], bytes], config: ContainerConfig
) -> ControlResult:
    """Compile twice; the bytes must be identical (deterministic reproducibility)."""
    first = bytes(compile_fn(config))
    second = bytes(compile_fn(config))
    return ControlResult(
        control="double_compile_determinism",
        passed=first == second,
        detail={
            "first_sha256": hashlib.sha256(first).hexdigest(),
            "second_sha256": hashlib.sha256(second).hexdigest(),
            "bytes": len(first),
        },
    )
