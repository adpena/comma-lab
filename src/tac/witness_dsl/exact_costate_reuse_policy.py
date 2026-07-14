# SPDX-License-Identifier: MIT
"""Typed, argv-inert policy for event-controlled exact-costate reuse.

A content-verified n600 receipt may admit the *measurement*.  It cannot admit
live trainer activation: the current trainer has no current-costate provider
seam and this DSL lever deliberately compiles no argv.  Keeping those two
authorities separate prevents a good offline receipt from manufacturing a live
integration claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.witness_dsl.curriculum_dsl import Lever

POLICY_NAME = "exact_costate_reuse_k2_guarded"
ADMISSION_VERDICT = "ADMIT_K2_GUARDED_REUSE"
FALLBACK = "full_teacher_refresh"
REQUIRED_K_MAX = 2
REQUIRED_N_PAIRS = 600
RECEIPT_SCHEMA = "p0_costate_reuse_k2_n600.v2"
PAIR_SCHEMA = "p0_costate_reuse_k2_pair.v2"
STAGE_SCHEMA = "p0_costate_reuse_k2_stage.v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRANSIENT_PREFIXES = (
    "/tmp",
    "/private/tmp",
    "/var/tmp",
    "/private/var/tmp",
    "/var/folders",
    "/private/var/folders",
)


def _valid_sha256(value: str) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _path_error(value: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "receipt path is missing"
    raw = Path(value)
    if ".." in raw.parts:
        return "receipt path contains parent traversal"
    resolved = str(raw.resolve(strict=False))
    if any(resolved == prefix or resolved.startswith(f"{prefix}/") for prefix in _TRANSIENT_PREFIXES):
        return "receipt path is transient"
    return None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return _sha256_bytes(encoded)


def _read_json_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} bytes are unavailable: {exc}") from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    return raw, payload


def _safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path escapes the receipt directory")
    return path


@dataclass(frozen=True)
class TemporalFidelityReceiptCustody:
    """Content-bound receipt loaded from durable storage."""

    path: str
    sha256: str
    status: str
    admission_verdict: str
    n_pairs: int
    objective_sha256: str
    scorer_sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> TemporalFidelityReceiptCustody:
        """Load actual bytes and derive custody; no caller-supplied hash is trusted."""

        path_text = str(path)
        error = _path_error(path_text)
        if error is not None:
            raise ValueError(error)
        raw, payload = _read_json_bytes(Path(path_text), label="receipt")
        required = {
            "status",
            "admission_verdict",
            "n_pairs",
            "objective_sha256",
            "scorer_sha256",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"receipt JSON is missing fields: {', '.join(missing)}")
        return cls(
            path=path_text,
            sha256=_sha256_bytes(raw),
            status=payload["status"],
            admission_verdict=payload["admission_verdict"],
            n_pairs=payload["n_pairs"],
            objective_sha256=payload["objective_sha256"],
            scorer_sha256=payload["scorer_sha256"],
        )

    def _stage_manifest_errors(
        self,
        *,
        stage_row: Any,
        receipt_dir: Path,
        run_contract_sha256: str,
    ) -> tuple[tuple[str, ...], set[int]]:
        """Re-hash one manifest and every pair record it names."""

        if not isinstance(stage_row, dict):
            return ("receipt stage-manifest custody entry is not an object",), set()
        checkpoint_name = stage_row.get("checkpoint_name")
        if not isinstance(checkpoint_name, str) or not checkpoint_name:
            return ("receipt stage-manifest checkpoint name is missing",), set()
        path_value = stage_row.get("path")
        if not isinstance(path_value, str) or not path_value:
            return ("stage manifest path is missing",), set()
        path_error = _path_error(path_value)
        if path_error is not None:
            return (path_error,), set()
        stage_path = Path(path_value).resolve()
        if (
            stage_path.parent != receipt_dir
            or stage_path.name != f"stage_{checkpoint_name}_complete.json"
        ):
            return ("stage manifest path does not match checkpoint name",), set()
        errors: list[str] = []
        try:
            raw, manifest = _read_json_bytes(stage_path, label=f"stage manifest {checkpoint_name}")
        except ValueError as exc:
            return (str(exc),), set()
        if stage_row.get("bytes") != len(raw):
            errors.append(f"stage manifest {checkpoint_name} byte count mismatch")
        if stage_row.get("sha256") != _sha256_bytes(raw):
            errors.append(f"stage manifest {checkpoint_name} sha256 mismatch")
        if manifest.get("schema") != STAGE_SCHEMA:
            errors.append(f"stage manifest {checkpoint_name} schema mismatch")
        if manifest.get("checkpoint_name") != checkpoint_name:
            errors.append(f"stage manifest {checkpoint_name} checkpoint mismatch")
        if manifest.get("run_contract_sha256") != run_contract_sha256:
            errors.append(f"stage manifest {checkpoint_name} run-contract mismatch")
        records = manifest.get("records")
        if not isinstance(records, list):
            return (*errors, f"stage manifest {checkpoint_name} records are missing"), set()
        if manifest.get("state_count") != len(records) or stage_row.get("state_count") != len(records):
            errors.append(f"stage manifest {checkpoint_name} state count mismatch")
        if manifest.get("tree_sha256") != _canonical_sha256(records):
            errors.append(f"stage manifest {checkpoint_name} tree sha256 mismatch")
        if stage_row.get("tree_sha256") != manifest.get("tree_sha256"):
            errors.append(f"stage manifest {checkpoint_name} receipt tree binding mismatch")

        pair_indices: set[int] = set()
        for record in records:
            if not isinstance(record, dict):
                errors.append(f"stage manifest {checkpoint_name} has non-object pair record")
                continue
            pair_index = record.get("pair_index")
            if isinstance(pair_index, bool) or not isinstance(pair_index, int):
                errors.append(f"stage manifest {checkpoint_name} pair index is invalid")
                continue
            if pair_index in pair_indices:
                errors.append(f"stage manifest {checkpoint_name} repeats pair {pair_index}")
                continue
            pair_indices.add(pair_index)
            try:
                pair_relative = _safe_relative_path(record.get("path"), label="pair record")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            expected_pair_relative = Path("pairs") / f"pair_{pair_index:04d}.json"
            if pair_relative != expected_pair_relative:
                errors.append(f"pair {pair_index} path is not canonical")
                continue
            pair_path = receipt_dir / pair_relative
            try:
                pair_raw, pair = _read_json_bytes(pair_path, label=f"pair record {pair_index}")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if record.get("bytes") != len(pair_raw):
                errors.append(f"pair {pair_index} byte count mismatch")
            if record.get("sha256") != _sha256_bytes(pair_raw):
                errors.append(f"pair {pair_index} sha256 mismatch")
            unsigned = {key: value for key, value in pair.items() if key != "record_content_sha256"}
            if pair.get("record_content_sha256") != _canonical_sha256(unsigned):
                errors.append(f"pair {pair_index} self-hash mismatch")
            assignment = pair.get("assignment")
            if pair.get("schema") != PAIR_SCHEMA:
                errors.append(f"pair {pair_index} schema mismatch")
            if pair.get("run_contract_sha256") != run_contract_sha256:
                errors.append(f"pair {pair_index} run-contract mismatch")
            if not isinstance(assignment, dict) or assignment.get("pair_index") != pair_index:
                errors.append(f"pair {pair_index} assignment mismatch")
            elif assignment.get("checkpoint_name") != checkpoint_name:
                errors.append(f"pair {pair_index} checkpoint assignment mismatch")
        return tuple(errors), pair_indices

    def _content_errors(self) -> tuple[str, ...]:
        path_error = _path_error(self.path)
        if path_error is not None:
            return (path_error,)
        try:
            raw, payload = _read_json_bytes(Path(self.path), label="receipt")
        except ValueError as exc:
            message = str(exc)
            if message.startswith("receipt bytes are unavailable"):
                return ("receipt bytes are unavailable",)
            return (message,)
        errors: list[str] = []
        if _sha256_bytes(raw) != self.sha256:
            errors.append("receipt bytes sha256 mismatch")
        expected = {
            "status": self.status,
            "admission_verdict": self.admission_verdict,
            "n_pairs": self.n_pairs,
            "objective_sha256": self.objective_sha256,
            "scorer_sha256": self.scorer_sha256,
        }
        for field, expected_value in expected.items():
            if payload.get(field) != expected_value:
                errors.append(f"receipt content {field} mismatch")
        if payload.get("schema") != RECEIPT_SCHEMA:
            errors.append("receipt schema is not the hardened v2 contract")
        run_contract = payload.get("run_contract")
        admission_content = payload.get("admission_content")
        measurement = payload.get("measurement")
        stage_custody = payload.get("stage_manifest_custody")
        gate = (payload.get("fidelity_gate") or {}).get("calibration_admission_gate")
        if not isinstance(run_contract, dict) or not _valid_sha256(run_contract.get("sha256")):
            errors.append("receipt run contract custody is missing")
        if not isinstance(admission_content, dict):
            errors.append("receipt admission content is missing")
        else:
            if _canonical_sha256(admission_content) != payload.get("admission_content_sha256"):
                errors.append("receipt admission content sha256 mismatch")
            bindings = {
                "objective_sha256": self.objective_sha256,
                "scorer_sha256": self.scorer_sha256,
                "admission_verdict": self.admission_verdict,
            }
            if isinstance(run_contract, dict):
                bindings["run_contract_sha256"] = run_contract.get("sha256")
            if isinstance(measurement, dict):
                bindings["aggregate_sha256"] = _canonical_sha256(measurement)
            for field, expected_value in bindings.items():
                if admission_content.get(field) != expected_value:
                    errors.append(f"receipt admission binding {field} mismatch")
            if admission_content.get("stage_manifest_custody") != stage_custody:
                errors.append("receipt stage-manifest admission binding mismatch")
        if not isinstance(stage_custody, list) or len(stage_custody) != 3:
            errors.append("receipt requires three checkpoint stage manifests")
        elif isinstance(run_contract, dict) and _valid_sha256(run_contract.get("sha256")):
            all_pairs: set[int] = set()
            for row in stage_custody:
                stage_errors, pair_indices = self._stage_manifest_errors(
                    stage_row=row,
                    receipt_dir=Path(self.path).resolve().parent,
                    run_contract_sha256=run_contract["sha256"],
                )
                errors.extend(stage_errors)
                overlap = all_pairs & pair_indices
                if overlap:
                    errors.append("stage manifests repeat pair indices")
                all_pairs.update(pair_indices)
            if len(all_pairs) != REQUIRED_N_PAIRS:
                errors.append("stage manifests do not cover exactly n600 unique pair records")
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            errors.append("receipt calibration admission gate did not pass")
        return tuple(errors)

    def validation_errors(
        self,
        *,
        expected_objective_sha256: str,
        expected_scorer_sha256: str,
        expected_receipt_sha256: str,
    ) -> tuple[str, ...]:
        errors = list(self._content_errors())
        if not _valid_sha256(self.sha256):
            errors.append("receipt sha256 is invalid")
        if not _valid_sha256(expected_receipt_sha256):
            errors.append("trusted expected receipt sha256 is missing or invalid")
        elif self.sha256 != expected_receipt_sha256:
            errors.append("receipt sha256 does not match trusted expected receipt sha256")
        if self.status != "completed":
            errors.append("receipt is not completed")
        if self.admission_verdict != ADMISSION_VERDICT:
            errors.append("receipt does not admit guarded K2 reuse")
        if isinstance(self.n_pairs, bool) or self.n_pairs != REQUIRED_N_PAIRS:
            errors.append(f"receipt n_pairs must equal {REQUIRED_N_PAIRS}")
        if not _valid_sha256(self.objective_sha256):
            errors.append("receipt objective sha256 is invalid")
        elif self.objective_sha256 != expected_objective_sha256:
            errors.append("receipt objective sha256 mismatch")
        if not _valid_sha256(self.scorer_sha256):
            errors.append("receipt scorer sha256 is invalid")
        elif self.scorer_sha256 != expected_scorer_sha256:
            errors.append("receipt scorer sha256 mismatch")
        return tuple(errors)


@dataclass(frozen=True)
class ExactCostateReusePolicy:
    """Sealed event-controlled ``K_max=2`` policy, live activation refused."""

    enabled: bool = False
    k_max: int = REQUIRED_K_MAX
    n_pairs: int = REQUIRED_N_PAIRS
    objective_sha256: str = ""
    scorer_sha256: str = ""
    receipt: TemporalFidelityReceiptCustody | None = None
    expected_receipt_sha256: str = ""
    fallback: str = FALLBACK
    provider_current: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if self.k_max != REQUIRED_K_MAX:
            raise ValueError(f"exact-costate reuse K_max is sealed to {REQUIRED_K_MAX}")
        if self.n_pairs != REQUIRED_N_PAIRS:
            raise ValueError(f"exact-costate reuse admission is sealed to n={REQUIRED_N_PAIRS}")
        if self.fallback != FALLBACK:
            raise ValueError(f"fallback is sealed to {FALLBACK!r}")
        if self.provider_current is not False:
            raise ValueError("current-costate provider is not integrated")
        if self.expected_receipt_sha256 and not _valid_sha256(self.expected_receipt_sha256):
            raise ValueError("expected_receipt_sha256 must be empty or a lowercase sha256")

    def measurement_errors(self) -> tuple[str, ...]:
        """Validate offline n600 evidence without implying trainer activation."""

        errors: list[str] = []
        if not _valid_sha256(self.objective_sha256):
            errors.append("expected objective sha256 is missing or invalid")
        if not _valid_sha256(self.scorer_sha256):
            errors.append("expected scorer sha256 is missing or invalid")
        if self.receipt is None:
            errors.append("completed temporal-fidelity receipt custody is missing")
        elif _valid_sha256(self.objective_sha256) and _valid_sha256(self.scorer_sha256):
            # Authorization deliberately re-reads bytes rather than trusting an
            # earlier process-local receipt object (TOCTOU fail-closed).
            errors.extend(
                self.receipt.validation_errors(
                    expected_objective_sha256=self.objective_sha256,
                    expected_scorer_sha256=self.scorer_sha256,
                    expected_receipt_sha256=self.expected_receipt_sha256,
                )
            )
        return tuple(errors)

    def trainer_activation_errors(self) -> tuple[str, ...]:
        """Return live refusal reasons; receipt admission cannot erase wiring debt."""

        errors = list(self.measurement_errors())
        if not self.enabled:
            errors.append("policy is default-off")
        if not self.provider_current:
            errors.append("current-costate provider is unavailable")
        errors.append("live trainer argv is empty")
        return tuple(errors)

    def compile_measurement_contract(self) -> dict[str, Any]:
        """Compile immutable offline evidence requirements; no blind cadence."""

        return {
            "policy": POLICY_NAME,
            "enabled": self.enabled,
            "K_max": self.k_max,
            "n_pairs": self.n_pairs,
            "pattern": [
                "exact_anchor_with_payload_and_full_facet_baseline",
                "at_most_one_changed_frame_reuse_attempt_in_same_event_scope",
            ],
            "accept_guard": {
                "ce": "candidate_strictly_less_than_anchor",
                "d_seg": "candidate_less_than_or_equal_to_anchor",
                "d_pose": "candidate_less_than_or_equal_to_anchor",
            },
            "forced_refresh_boundaries": ["event", "stage", "custody_change"],
            "fallback": self.fallback,
            "receipt": asdict(self.receipt) if self.receipt is not None else None,
            "expected_receipt_sha256": self.expected_receipt_sha256,
            "objective_sha256": self.objective_sha256,
            "scorer_sha256": self.scorer_sha256,
            "live_trainer_argv": [],
            "provider_current": self.provider_current,
            "score_claim": False,
        }

    def compile_activation_contract(self) -> dict[str, Any]:
        """Expose measurement and live-trainer authorities as separate gates."""

        measurement_errors = self.measurement_errors()
        trainer_errors = self.trainer_activation_errors()
        return {
            **self.compile_measurement_contract(),
            "measurement_admitted": not measurement_errors,
            "measurement_errors": list(measurement_errors),
            "measurement_authority": (
                "completed_content_verified_n600_temporal_fidelity_receipt"
                if not measurement_errors
                else "REFUSED"
            ),
            "trainer_activation_admitted": False,
            "trainer_activation_errors": list(trainer_errors),
            "trainer_activation_authority": "REFUSED_NO_PROVIDER_OR_ARGV",
        }


def exact_costate_reuse_k2_lever(
    policy: ExactCostateReusePolicy | None = None,
) -> Lever:
    """Return the named default-off DSL leg with empty argv overrides."""

    compiled = (policy or ExactCostateReusePolicy()).compile_activation_contract()
    measurement = "ADMITTED" if compiled["measurement_admitted"] else "REFUSED"
    trainer = "ADMITTED" if compiled["trainer_activation_admitted"] else "REFUSED"
    reasons = "; ".join(compiled["trainer_activation_errors"])
    return Lever(
        name=POLICY_NAME,
        overrides={},
        epochs_delta=0,
        notes=(
            f"argv-inert typed policy; measurement={measurement}; trainer={trainer}; "
            f"{reasons}; explicit provider and main review remain owed"
        ),
    )


__all__ = [
    "ADMISSION_VERDICT",
    "FALLBACK",
    "POLICY_NAME",
    "REQUIRED_K_MAX",
    "REQUIRED_N_PAIRS",
    "ExactCostateReusePolicy",
    "TemporalFidelityReceiptCustody",
    "exact_costate_reuse_k2_lever",
]
