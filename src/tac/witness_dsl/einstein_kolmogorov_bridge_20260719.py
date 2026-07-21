"""Typed contract for the Einstein--Kolmogorov cross-checkpoint xi bridge.

The contract has two deliberately disjoint execution classes: a zero-cost local
diagnostic capped at 24 pairs, and an operator-authorized full-n600 run.  Neither
class can request exact evaluation or a frontier-pointer update.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "einstein_kolmogorov_xi_bridge_config.v3"
EXPECTED_N_PAIRS = 600
DIAGNOSTIC_MAX_PAIRS = 24
DIAGNOSTIC_MODE = "diagnostic"
GOVERNED_FULL_MODE = "governed_full"
LANE_ID = "lane_einstein_kolmogorov_crux_20260719"
DXI_SCALE = 1.0
POSE_CARRIER_S_T = 0.16
POSE_CARRIER_S_R = 0.0
POSE_CARRIER_PITCH = 0.0
XI_CODER = "delta_ar"
XI_Q_LEVELS = 4096
BIT_EXACT_PAIRS = 2

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


class EinsteinKolmogorovBridgeConfigError(ValueError):
    """Raised when the bridge contract is not explicit and safely contained."""


def _absolute(value: str, field: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise EinsteinKolmogorovBridgeConfigError(f"{field} must be an absolute path")
    return str(path.resolve(strict=False))


def _sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise EinsteinKolmogorovBridgeConfigError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class EinsteinKolmogorovXiBridgeConfig:
    """Closed, serializable configuration for one contained bridge run."""

    generator_checkpoint_dir: str
    generator_npz_path: str
    generator_npz_sha256: str
    donor_r1_npz_path: str
    donor_r1_npz_sha256: str
    gt_cache_path: str
    gt_cache_sha256: str
    packet_output_dir: str
    result_json_path: str
    max_pairs: int = DIAGNOSTIC_MAX_PAIRS
    execution_mode: str = DIAGNOSTIC_MODE
    operator_authorization_path: str | None = None
    operator_authorization_sha256: str | None = None
    governed_claim_job_id: str | None = None
    governed_claim_platform: str | None = None
    declared_max_cost_usd: float | None = None
    # Historical v2 receipts carried these fields.  V3 can retain them as
    # non-authorizing provenance, but neither field is execution authority.
    operator_authorization_token: str | None = None
    declared_minimum_free_bytes: int | None = None
    failure_manifest_path: str | None = None
    backend_resume_receipt_path: str | None = None
    backend_resume_receipt_sha256: str | None = None
    expected_n_pairs: int = EXPECTED_N_PAIRS
    dxi_scale: float = DXI_SCALE
    pose_carrier_s_t: float = POSE_CARRIER_S_T
    pose_carrier_s_r: float = POSE_CARRIER_S_R
    pose_carrier_pitch: float = POSE_CARRIER_PITCH
    xi_coder: str = XI_CODER
    xi_q_levels: int = XI_Q_LEVELS
    bit_exact_pairs: int = BIT_EXACT_PAIRS
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise EinsteinKolmogorovBridgeConfigError(f"unsupported schema {self.schema!r}")
        for field in (
            "generator_checkpoint_dir",
            "generator_npz_path",
            "donor_r1_npz_path",
            "gt_cache_path",
            "packet_output_dir",
            "result_json_path",
        ):
            object.__setattr__(self, field, _absolute(str(getattr(self, field)), field))
        if self.failure_manifest_path is None:
            result = Path(self.result_json_path)
            object.__setattr__(
                self,
                "failure_manifest_path",
                str(result.with_name(f"{result.stem}.failure.json")),
            )
        else:
            object.__setattr__(
                self,
                "failure_manifest_path",
                _absolute(str(self.failure_manifest_path), "failure_manifest_path"),
            )
        if (self.operator_authorization_path is None) != (self.operator_authorization_sha256 is None):
            raise EinsteinKolmogorovBridgeConfigError(
                "operator authorization path and SHA-256 must be supplied together"
            )
        if self.operator_authorization_path is not None:
            object.__setattr__(
                self,
                "operator_authorization_path",
                _absolute(
                    str(self.operator_authorization_path),
                    "operator_authorization_path",
                ),
            )
            object.__setattr__(
                self,
                "operator_authorization_sha256",
                _sha256(
                    str(self.operator_authorization_sha256),
                    "operator_authorization_sha256",
                ),
            )
        if (self.backend_resume_receipt_path is None) != (self.backend_resume_receipt_sha256 is None):
            raise EinsteinKolmogorovBridgeConfigError(
                "backend resume receipt path and SHA-256 must be supplied together"
            )
        if self.backend_resume_receipt_path is not None:
            object.__setattr__(
                self,
                "backend_resume_receipt_path",
                _absolute(str(self.backend_resume_receipt_path), "backend_resume_receipt_path"),
            )
            object.__setattr__(
                self,
                "backend_resume_receipt_sha256",
                _sha256(str(self.backend_resume_receipt_sha256), "backend_resume_receipt_sha256"),
            )
        for field in (
            "generator_npz_sha256",
            "donor_r1_npz_sha256",
            "gt_cache_sha256",
        ):
            object.__setattr__(self, field, _sha256(str(getattr(self, field)), field))

        generator_npz = Path(self.generator_npz_path)
        if generator_npz.parent != Path(self.generator_checkpoint_dir):
            raise EinsteinKolmogorovBridgeConfigError(
                "generator_npz_path must be a direct child of generator_checkpoint_dir"
            )
        if generator_npz.suffix != ".npz":
            raise EinsteinKolmogorovBridgeConfigError("generator_npz_path must end in .npz")
        if Path(self.donor_r1_npz_path).suffix != ".npz":
            raise EinsteinKolmogorovBridgeConfigError("donor_r1_npz_path must end in .npz")
        if Path(self.gt_cache_path).suffix != ".npz":
            raise EinsteinKolmogorovBridgeConfigError("gt_cache_path must end in .npz")

        packet_dir = Path(self.packet_output_dir)
        if not any(_is_within(packet_dir, root) for root in _SSD_ROOTS):
            raise EinsteinKolmogorovBridgeConfigError(
                "packet_output_dir must be under /Volumes/VertigoDataTier/pact or "
                "/Volumes/APDataStore/pact; local/tmp bulky output is forbidden"
            )
        for field in ("result_json_path", "failure_manifest_path"):
            result_path = Path(str(getattr(self, field)))
            if not _is_within(result_path, _REPO_ROOT):
                raise EinsteinKolmogorovBridgeConfigError(f"{field} must be inside repository {_REPO_ROOT}")
            if result_path.suffix != ".json":
                raise EinsteinKolmogorovBridgeConfigError(f"{field} must end in .json")
        if self.result_json_path == self.failure_manifest_path:
            raise EinsteinKolmogorovBridgeConfigError("result_json_path and failure_manifest_path must be distinct")

        if isinstance(self.max_pairs, bool) or not isinstance(self.max_pairs, int):
            raise EinsteinKolmogorovBridgeConfigError("max_pairs must be an integer")
        if self.execution_mode == DIAGNOSTIC_MODE:
            if not 1 <= self.max_pairs <= DIAGNOSTIC_MAX_PAIRS:
                raise EinsteinKolmogorovBridgeConfigError(
                    f"diagnostic max_pairs must be in [1, {DIAGNOSTIC_MAX_PAIRS}]"
                )
            if self.operator_authorization_token is not None:
                raise EinsteinKolmogorovBridgeConfigError(
                    "diagnostic mode must not carry legacy operator authorization tokens"
                )
            if self.operator_authorization_path is not None:
                raise EinsteinKolmogorovBridgeConfigError(
                    "diagnostic mode must not carry a full-run operator authorization"
                )
            if self.governed_claim_job_id is not None or self.governed_claim_platform is not None:
                raise EinsteinKolmogorovBridgeConfigError("diagnostic mode must not carry a governed dispatch claim")
            if self.declared_max_cost_usd is not None:
                raise EinsteinKolmogorovBridgeConfigError(
                    "diagnostic mode must not carry a full-run cost authorization"
                )
            if self.backend_resume_receipt_path is not None:
                raise EinsteinKolmogorovBridgeConfigError(
                    "diagnostic mode must not carry a full-backend resume receipt"
                )
        elif self.execution_mode == GOVERNED_FULL_MODE:
            if self.max_pairs != EXPECTED_N_PAIRS:
                raise EinsteinKolmogorovBridgeConfigError("governed_full mode requires max_pairs=600")
            if self.operator_authorization_token is not None:
                raise EinsteinKolmogorovBridgeConfigError(
                    "legacy OPERATOR-GO prefixes are not authorization; governed_full requires "
                    "a committed, hash-bound operator authorization file"
                )
            if self.operator_authorization_path is None:
                raise EinsteinKolmogorovBridgeConfigError(
                    "governed_full requires operator_authorization_path and SHA-256"
                )
            auth_path = Path(self.operator_authorization_path)
            auth_root = _REPO_ROOT / ".omx" / "research" / "operator_authorizations"
            if not _is_within(auth_path, auth_root) or auth_path.suffix != ".json":
                raise EinsteinKolmogorovBridgeConfigError(
                    "operator_authorization_path must be a JSON file under canonical "
                    ".omx/research/operator_authorizations"
                )
            for field in ("governed_claim_job_id", "governed_claim_platform"):
                value = getattr(self, field)
                if not isinstance(value, str) or not value.strip() or value != value.strip():
                    raise EinsteinKolmogorovBridgeConfigError(f"governed_full requires non-empty canonical {field}")
            if (
                isinstance(self.declared_max_cost_usd, bool)
                or not isinstance(self.declared_max_cost_usd, (int, float))
                or not 0.0 <= float(self.declared_max_cost_usd) < float("inf")
            ):
                raise EinsteinKolmogorovBridgeConfigError("governed_full requires finite declared_max_cost_usd >= 0")
        else:
            raise EinsteinKolmogorovBridgeConfigError(
                f"execution_mode must be {DIAGNOSTIC_MODE!r} or {GOVERNED_FULL_MODE!r}"
            )
        if self.declared_minimum_free_bytes is not None and (
            isinstance(self.declared_minimum_free_bytes, bool)
            or not isinstance(self.declared_minimum_free_bytes, int)
            or self.declared_minimum_free_bytes <= 0
        ):
            raise EinsteinKolmogorovBridgeConfigError(
                "declared_minimum_free_bytes must be a positive integer when supplied"
            )
        fixed = (
            self.expected_n_pairs == EXPECTED_N_PAIRS
            and self.dxi_scale == DXI_SCALE
            and self.pose_carrier_s_t == POSE_CARRIER_S_T
            and self.pose_carrier_s_r == POSE_CARRIER_S_R
            and self.pose_carrier_pitch == POSE_CARRIER_PITCH
            and self.xi_coder == XI_CODER
            and self.xi_q_levels == XI_Q_LEVELS
            and self.bit_exact_pairs == BIT_EXACT_PAIRS
        )
        if not fixed:
            raise EinsteinKolmogorovBridgeConfigError(
                "bridge constants are frozen: expected_n_pairs=600, dxi_scale=1, "
                "pose_carrier_s_t=0.16, pose_carrier_s_r=0, pitch=0, xi_coder=delta_ar, "
                "xi_q_levels=4096, bit_exact_pairs=2"
            )

    @property
    def generator_npz_name(self) -> str:
        return Path(self.generator_npz_path).name

    @property
    def derived_minimum_free_bytes(self) -> int:
        """Conservative receiver budget: four raw copies plus a 1 GiB reserve."""

        raw_bytes = self.max_pairs * 2 * 874 * 1164 * 3
        return 4 * raw_bytes + (1 << 30)

    @property
    def required_free_bytes(self) -> int:
        declared = self.declared_minimum_free_bytes or 0
        return max(self.derived_minimum_free_bytes, declared)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EinsteinKolmogorovXiBridgeConfig:
        fields = cls.__dataclass_fields__
        unknown = set(payload) - set(fields)
        missing = {
            name
            for name, field in fields.items()
            if name not in payload and field.default is MISSING and field.default_factory is MISSING
        }
        if unknown or missing:
            raise EinsteinKolmogorovBridgeConfigError(
                f"config keys differ (unknown={sorted(unknown)}, missing={sorted(missing)})"
            )
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path) -> EinsteinKolmogorovXiBridgeConfig:
        config_path = Path(path)
        if not config_path.is_absolute():
            raise EinsteinKolmogorovBridgeConfigError("config path must be absolute")
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EinsteinKolmogorovBridgeConfigError("config is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise EinsteinKolmogorovBridgeConfigError("config JSON must be an object")
        return cls.from_dict(payload)
