# SPDX-License-Identifier: MIT
"""Pure terminal costate-skip eligibility decisions.

The #396 exact-metric Monte-Carlo finisher is an accept/reject search, not a
gradient estimator.  SPSA/ES are kept distinct and may skip the exact teacher
only under a deterministic effective-dimension certificate of at most two.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

REQUIRED_N_PAIRS = 600
MAX_GRADIENT_FREE_EFFECTIVE_DIMENSION = 2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRANSIENT_PREFIXES = (
    "/tmp",
    "/private/tmp",
    "/var/tmp",
    "/private/var/tmp",
    "/var/folders",
    "/private/var/folders",
)
TRUSTED_TERMINAL_RECEIPT_SHA256S = frozenset(
    {"17574857da5ff862e520140977e988197962f009d6870d23fe3071c398112a9c"}
)
# No SPSA/ES effective-dimension certificate has met the n600 admission bar.
TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S = frozenset()


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character sha256")


def _resolve_durable_file(path: str | Path, field_name: str) -> Path:
    """Resolve symlinks before rejecting all transient storage roots."""

    try:
        source = Path(path).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{field_name} must be a durable file") from exc
    normalized = str(source)
    if not source.is_file() or any(
        normalized == prefix or normalized.startswith(f"{prefix}/")
        for prefix in _TRANSIENT_PREFIXES
    ):
        raise ValueError(f"{field_name} must be a durable file")
    return source


class TerminalMethod(StrEnum):
    EXACT_METRIC_MC_396 = "exact_metric_mc_396"
    SPSA = "spsa"
    ES = "es"


class TerminalAction(StrEnum):
    SKIP_COSTATE_EXACT_METRIC_MC = "skip_costate_exact_metric_mc"
    SKIP_COSTATE_DIMENSION_CERTIFIED = "skip_costate_dimension_certified"
    EXACT_METRIC_MC_ORDINARY_ROUTE = "exact_metric_mc_ordinary_route"
    FULL_TEACHER_OR_396_ORDINARY_ROUTE = "full_teacher_or_396_ordinary_route"


@dataclass(frozen=True)
class TerminalReceiptIdentity:
    path: str
    sha256: str
    status: str
    n_pairs: int
    objective_sha256: str
    scorer_sha256: str
    content_verified: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_path(
        cls, path: str | Path, *, expected_sha256: str
    ) -> TerminalReceiptIdentity:
        """Load only a receipt whose bytes match an externally pinned root hash."""

        _require_sha256(expected_sha256, "expected receipt sha256")
        if expected_sha256 not in TRUSTED_TERMINAL_RECEIPT_SHA256S:
            raise ValueError("expected receipt sha256 is not a code-reviewed trust root")
        source = _resolve_durable_file(path, "terminal receipt")
        raw = source.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("terminal receipt sha256 does not match the pinned root")
        payload = json.loads(raw)
        required = ("status", "n_pairs", "objective_sha256", "scorer_sha256")
        if any(key not in payload for key in required):
            raise ValueError("terminal receipt schema is missing identity fields")
        identity = cls(
            path=str(source),
            sha256=actual_sha256,
            status=payload["status"],
            n_pairs=payload["n_pairs"],
            objective_sha256=payload["objective_sha256"],
            scorer_sha256=payload["scorer_sha256"],
        )
        object.__setattr__(identity, "content_verified", True)
        return identity

    def matches(self, expected: TerminalReceiptIdentity) -> bool:
        return (
            bool(self.path.strip())
            and self.status == "completed"
            and self.n_pairs == REQUIRED_N_PAIRS
            and self.content_verified is True
            and expected.content_verified is True
            and self._bytes_still_match_identity()
            and expected._bytes_still_match_identity()
            and _SHA256_RE.fullmatch(self.sha256) is not None
            and _SHA256_RE.fullmatch(self.objective_sha256) is not None
            and _SHA256_RE.fullmatch(self.scorer_sha256) is not None
            and self == expected
        )

    def _bytes_still_match_identity(self) -> bool:
        """Close the load-to-authorize window by re-reading the durable receipt."""

        try:
            source = _resolve_durable_file(self.path, "terminal receipt")
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.sha256:
                return False
            payload = json.loads(raw)
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and all(
            payload.get(field) == expected
            for field, expected in {
                "status": self.status,
                "n_pairs": self.n_pairs,
                "objective_sha256": self.objective_sha256,
                "scorer_sha256": self.scorer_sha256,
            }.items()
        )


@dataclass(frozen=True)
class EffectiveDimensionCertificate:
    effective_dimension: int
    deterministic: bool
    artifact_path: str
    artifact_sha256: str
    content_verified: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_path(
        cls, path: str | Path, *, expected_sha256: str
    ) -> EffectiveDimensionCertificate:
        """Load only a certificate whose bytes match an external pinned root hash."""

        _require_sha256(expected_sha256, "expected dimension certificate sha256")
        if expected_sha256 not in TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S:
            raise ValueError(
                "expected dimension certificate sha256 is not a code-reviewed trust root"
            )
        source = _resolve_durable_file(path, "dimension certificate")
        raw = source.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("dimension certificate sha256 does not match the pinned root")
        payload = json.loads(raw)
        if "effective_dimension" not in payload or "deterministic" not in payload:
            raise ValueError("dimension certificate schema is incomplete")
        certificate = cls(
            effective_dimension=payload["effective_dimension"],
            deterministic=payload["deterministic"],
            artifact_path=str(source),
            artifact_sha256=actual_sha256,
        )
        object.__setattr__(certificate, "content_verified", True)
        return certificate

    def admits(self) -> bool:
        return (
            not isinstance(self.effective_dimension, bool)
            and isinstance(self.effective_dimension, int)
            and 0 <= self.effective_dimension <= MAX_GRADIENT_FREE_EFFECTIVE_DIMENSION
            and self.deterministic is True
            and self.content_verified is True
            and bool(self.artifact_path.strip())
            and _SHA256_RE.fullmatch(self.artifact_sha256) is not None
            and self._bytes_still_match_identity()
        )

    def _bytes_still_match_identity(self) -> bool:
        """Close the load-to-authorize window for effective-dimension evidence."""

        try:
            source = _resolve_durable_file(self.artifact_path, "dimension certificate")
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.artifact_sha256:
                return False
            payload = json.loads(raw)
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and (
            payload.get("effective_dimension") == self.effective_dimension
            and payload.get("deterministic") is self.deterministic
        )


@dataclass(frozen=True)
class TerminalSkipDecision:
    action: TerminalAction
    costate_required: bool
    reason: str


def decide_terminal_costate_skip(
    *,
    method: TerminalMethod,
    receipt: TerminalReceiptIdentity | None,
    expected_receipt: TerminalReceiptIdentity,
    expected_receipt_sha256: str,
    dimension_certificate: EffectiveDimensionCertificate | None = None,
    expected_dimension_certificate_sha256: str | None,
) -> TerminalSkipDecision:
    """Authorize a skip only against explicit external receipt/certificate roots."""

    receipt_root_matches = (
        isinstance(expected_receipt_sha256, str)
        and _SHA256_RE.fullmatch(expected_receipt_sha256) is not None
        and expected_receipt_sha256 in TRUSTED_TERMINAL_RECEIPT_SHA256S
        and expected_receipt.sha256 == expected_receipt_sha256
    )
    receipt_matches = (
        receipt_root_matches
        and receipt is not None
        and receipt.sha256 == expected_receipt_sha256
        and receipt.matches(expected_receipt)
    )
    if method is TerminalMethod.EXACT_METRIC_MC_396:
        if receipt_matches:
            return TerminalSkipDecision(
                TerminalAction.SKIP_COSTATE_EXACT_METRIC_MC,
                False,
                "#396 uses exact n600 accept/reject metrics and no gradient",
            )
        return TerminalSkipDecision(
            TerminalAction.EXACT_METRIC_MC_ORDINARY_ROUTE,
            True,
            "#396 handoff receipt identity is absent or mismatched",
        )

    certificate_root_matches = (
        dimension_certificate is not None
        and isinstance(expected_dimension_certificate_sha256, str)
        and _SHA256_RE.fullmatch(expected_dimension_certificate_sha256) is not None
        and expected_dimension_certificate_sha256
        in TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S
        and dimension_certificate.artifact_sha256 == expected_dimension_certificate_sha256
    )
    if (
        receipt_matches
        and certificate_root_matches
        and dimension_certificate is not None
        and dimension_certificate.admits()
    ):
        return TerminalSkipDecision(
            TerminalAction.SKIP_COSTATE_DIMENSION_CERTIFIED,
            False,
            "SPSA/ES effective dimension is deterministically certified <=2",
        )
    return TerminalSkipDecision(
        TerminalAction.FULL_TEACHER_OR_396_ORDINARY_ROUTE,
        True,
        "SPSA/ES lacks matching n600 custody or a deterministic effective-dimension <=2 certificate",
    )


__all__ = [
    "MAX_GRADIENT_FREE_EFFECTIVE_DIMENSION",
    "REQUIRED_N_PAIRS",
    "TRUSTED_EFFECTIVE_DIMENSION_CERTIFICATE_SHA256S",
    "TRUSTED_TERMINAL_RECEIPT_SHA256S",
    "EffectiveDimensionCertificate",
    "TerminalAction",
    "TerminalMethod",
    "TerminalReceiptIdentity",
    "TerminalSkipDecision",
    "decide_terminal_costate_skip",
]
