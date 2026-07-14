# SPDX-License-Identifier: MIT
"""Typed, argv-inert policy for the corrected exact-costate-reuse NO-GO.

The code-reviewed n600 wrapper may verify offline measurement custody.  Its
corrected verdict is ``NOT_ADMITTED``, and it can authorize neither the policy
nor live trainer activation.  The current trainer also has no current-costate
provider seam, so this DSL lever deliberately compiles no argv.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.witness_dsl.curriculum_dsl import Lever

POLICY_NAME = "exact_costate_reuse_k2_guarded"
CORRECTED_ADMISSION_VERDICT = "NOT_ADMITTED"
FALLBACK = "full_teacher_refresh"
REQUIRED_K_MAX = 2
REQUIRED_N_PAIRS = 600
RECEIPT_SCHEMA = "p0_costate_reuse_k2_corrected_adjudication.v1"
PAIR_SCHEMA = "p0_costate_reuse_k2_pair.v2"
STAGE_SCHEMA = "p0_costate_reuse_k2_stage.v2"
COMPLETE_SCHEMA = "p0_costate_reuse_k2_complete.v2"
TRUSTED_CORRECTED_WRAPPER_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/results/p0_costate_reuse_k2_n600_v3_20260713"
    / "corrected_adjudication_receipt.json"
)
TRUSTED_CORRECTED_WRAPPER_SHA256S = frozenset({"2102912bc8bd9711f00869746414fb21ea723729bcd26e612274547c6ca73d59"})
TRUSTED_ADJUDICATION_CONTENT_SHA256 = "4f7f2a6ef95f9989b734cd6e785d8b55dca7a77d7d9f03c693ff18287dea6e6e"
TRUSTED_SOURCE_ROOTS = {
    "measurement_receipt_sha256": "4c84c1f80ae7fc1b4ee76d28395405834e3eecd439155e4ebd79d4e81530506c",
    "complete_sha256": "45ccbccee780d26bf350442ddf5551d62d483957c591b706fe5eb746dfbea34c",
    "run_contract_sha256": "e9c4a6629bcbc91876d2476b0bef051dfe56fe27d93076fa79f7225a5b62d56f",
    "objective_sha256": "af5ae342f3987b82c2d3ee5bdb12dcfca1ecab07631fd545a9e723c15cb7c9e7",
    "scorer_sha256": "584f711dfb85163c38caf8976ebeda87698baefb45f9f5979539a8c176b6b73e",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRANSIENT_PREFIXES = (
    "/tmp",
    "/private/tmp",
    "/var/tmp",
    "/private/var/tmp",
    "/var/folders",
    "/private/var/folders",
)
_FROM_PATH_AUTHORITY_TOKEN = object()


@dataclass(frozen=True)
class _FileSnapshot:
    relative_path: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return _sha256_bytes(encoded)


def _read_json_no_follow(root: Path, relative: Path, *, label: str) -> tuple[bytes, dict[str, Any], _FileSnapshot]:
    """Read one regular JSON file through no-follow directory descriptors."""

    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} path escapes the receipt directory")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow | cloexec
    file_flags = os.O_RDONLY | nofollow | cloexec
    descriptors: list[int] = []
    try:
        descriptors.append(os.open(root, directory_flags))
        for component in relative.parts[:-1]:
            descriptors.append(os.open(component, directory_flags, dir_fd=descriptors[-1]))
        descriptor = os.open(relative.parts[-1], file_flags, dir_fd=descriptors[-1])
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"{label} path contains a symlink") from exc
        raise ValueError(f"{label} bytes are unavailable: {exc}") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(raw) != after.st_size:
        raise ValueError(f"{label} changed while being read")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    snapshot = _FileSnapshot(
        relative_path=relative.as_posix(),
        device=after.st_dev,
        inode=after.st_ino,
        mode=after.st_mode,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
        sha256=_sha256_bytes(raw),
    )
    return raw, payload, snapshot


def _record_snapshot(
    snapshots: dict[str, _FileSnapshot],
    snapshot: _FileSnapshot,
    *,
    label: str,
) -> list[str]:
    prior = snapshots.get(snapshot.relative_path)
    if prior is not None and prior != snapshot:
        return [f"{label} changed during first custody pass"]
    snapshots[snapshot.relative_path] = snapshot
    return []


def _before_final_snapshot_verify() -> None:
    """Test seam immediately before the fail-closed second custody pass."""


def _verify_snapshot_unchanged(root: Path, snapshots: dict[str, _FileSnapshot]) -> tuple[str, ...]:
    errors: list[str] = []
    for relative_text, expected in sorted(snapshots.items()):
        try:
            _, _, observed = _read_json_no_follow(root, Path(relative_text), label=relative_text)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if observed != expected:
            errors.append(f"{relative_text} changed between custody passes")
    return tuple(errors)


def _safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path escapes the receipt directory")
    return path


@dataclass(frozen=True)
class TemporalFidelityReceiptCustody:
    """Code-reviewed corrected wrapper plus recursively verified source custody."""

    path: str
    sha256: str
    status: str
    admission_verdict: str
    n_pairs: int
    objective_sha256: str
    scorer_sha256: str
    run_contract_sha256: str
    _authority_token: object | None = field(default=None, init=False, repr=False, compare=False)
    _wrapper_snapshot: _FileSnapshot | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_path(cls, path: str | Path) -> TemporalFidelityReceiptCustody:
        """Load only the code-reviewed wrapper path; caller hashes have no authority."""

        candidate = Path(path)
        path_text = str(candidate)
        error = _path_error(path_text)
        if error is not None:
            raise ValueError(error)
        if candidate.absolute() != TRUSTED_CORRECTED_WRAPPER_PATH.absolute():
            raise ValueError("corrected wrapper path is not code-reviewed")
        root = candidate.absolute().parent
        raw, payload, snapshot = _read_json_no_follow(root, Path(candidate.name), label="corrected wrapper")
        source = payload.get("source_adjudication")
        measurement = payload.get("measurement_rederived_from_sealed_rows")
        if not isinstance(source, dict) or not isinstance(measurement, dict):
            raise ValueError("corrected wrapper is missing source or measurement content")
        custody = cls(
            path=path_text,
            sha256=_sha256_bytes(raw),
            status=payload["status"],
            admission_verdict=payload["corrected_admission_verdict"],
            n_pairs=measurement["state_count"],
            objective_sha256=source["objective_sha256"],
            scorer_sha256=source["scorer_sha256"],
            run_contract_sha256=source["run_contract_sha256"],
        )
        object.__setattr__(custody, "_authority_token", _FROM_PATH_AUTHORITY_TOKEN)
        object.__setattr__(custody, "_wrapper_snapshot", snapshot)
        return custody

    def public_custody(self) -> dict[str, Any]:
        """Return serializable fields without private construction authority."""

        return {
            "path": self.path,
            "sha256": self.sha256,
            "status": self.status,
            "admission_verdict": self.admission_verdict,
            "n_pairs": self.n_pairs,
            "objective_sha256": self.objective_sha256,
            "scorer_sha256": self.scorer_sha256,
            "run_contract_sha256": self.run_contract_sha256,
        }

    @staticmethod
    def _custody_file(
        root: Path,
        entry: Any,
        snapshots: dict[str, _FileSnapshot],
        *,
        label: str,
    ) -> tuple[list[str], bytes | None, dict[str, Any] | None]:
        if not isinstance(entry, dict):
            return [f"{label} custody entry is not an object"], None, None
        try:
            relative = _safe_relative_path(entry.get("path"), label=label)
        except ValueError as exc:
            return [str(exc)], None, None
        try:
            raw, payload, snapshot = _read_json_no_follow(root, relative, label=label)
        except ValueError as exc:
            return [str(exc)], None, None
        errors = _record_snapshot(snapshots, snapshot, label=label)
        if entry.get("bytes") != len(raw):
            errors.append(f"{label} byte count mismatch")
        if entry.get("sha256") != _sha256_bytes(raw):
            errors.append(f"{label} sha256 mismatch")
        return errors, raw, payload

    def _content_errors(
        self,
    ) -> tuple[tuple[str, ...], Path, dict[str, _FileSnapshot]]:
        path_error = _path_error(self.path)
        root = Path(self.path).absolute().parent
        snapshots: dict[str, _FileSnapshot] = {}
        if path_error is not None:
            return (path_error,), root, snapshots
        try:
            wrapper_relative = Path(self.path).absolute().relative_to(root)
            raw, payload, wrapper_snapshot = _read_json_no_follow(root, wrapper_relative, label="corrected wrapper")
        except ValueError as exc:
            message = str(exc)
            if "bytes are unavailable" in message:
                return ("corrected wrapper bytes are unavailable",), root, snapshots
            return (message,), root, snapshots
        errors = _record_snapshot(snapshots, wrapper_snapshot, label="corrected wrapper")
        if self._authority_token is not _FROM_PATH_AUTHORITY_TOKEN:
            errors.append("corrected wrapper custody was not established by from_path")
        if self._wrapper_snapshot is None or self._wrapper_snapshot != wrapper_snapshot:
            errors.append("corrected wrapper from_path snapshot mismatch")
        if _sha256_bytes(raw) != self.sha256:
            errors.append("corrected wrapper bytes sha256 mismatch")
        source = payload.get("source_adjudication")
        measurement = payload.get("measurement_rederived_from_sealed_rows")
        wrapper_fields = {
            "status": payload.get("status"),
            "admission_verdict": payload.get("corrected_admission_verdict"),
            "n_pairs": measurement.get("state_count") if isinstance(measurement, dict) else None,
            "objective_sha256": source.get("objective_sha256") if isinstance(source, dict) else None,
            "scorer_sha256": source.get("scorer_sha256") if isinstance(source, dict) else None,
            "run_contract_sha256": source.get("run_contract_sha256") if isinstance(source, dict) else None,
        }
        for field_name, wrapper_value in wrapper_fields.items():
            if getattr(self, field_name) != wrapper_value:
                errors.append(f"corrected wrapper instance {field_name} does not match bytes")
        unsigned = {k: v for k, v in payload.items() if k != "adjudication_content_sha256"}
        if payload.get("adjudication_content_sha256") != _canonical_sha256(unsigned):
            errors.append("corrected wrapper content sha256 mismatch")
        if payload.get("adjudication_content_sha256") != TRUSTED_ADJUDICATION_CONTENT_SHA256:
            errors.append("corrected wrapper content root is not reviewed")
        if payload.get("schema") != RECEIPT_SCHEMA:
            errors.append("corrected wrapper schema mismatch")
        roots = source.get("reviewed_source_roots") if isinstance(source, dict) else None
        if roots != TRUSTED_SOURCE_ROOTS:
            errors.append("corrected wrapper reviewed source roots mismatch")
        if not isinstance(source, dict) or source.get("original_admission_verdict_status") != (
            "SUPERSEDED_INVALID_FALLBACK_CHARGE"
        ):
            errors.append("original receipt supersession label is missing")
        gate = payload.get("corrected_admission_gate")
        if not isinstance(gate, dict) or gate.get("passed") is not False:
            errors.append("corrected gate must preserve measured non-admission")
        if payload.get("corrected_admission_verdict") != CORRECTED_ADMISSION_VERDICT:
            errors.append("corrected verdict is not NOT_ADMITTED")
        authority = payload.get("authority")
        required_false = (
            "score_claim",
            "promotion_eligible",
            "pointer_moved",
            "live_trainer_activation",
            "runtime_exact_gradient_access",
        )
        if not isinstance(authority, dict) or any(authority.get(k) is not False for k in required_false):
            errors.append("corrected wrapper carries false authority")

        custody = payload.get("source_custody")
        if not isinstance(custody, dict):
            return (
                (*errors, "corrected wrapper source custody is missing"),
                root,
                snapshots,
            )
        source_tree = custody.get("source_tree")
        if not isinstance(source_tree, list) or len(source_tree) != 606:
            errors.append("source tree must contain exactly 606 files")
            source_tree = []
        elif custody.get("source_tree_sha256") != _canonical_sha256(source_tree):
            errors.append("source tree sha256 mismatch")
        tree_by_path: dict[str, dict[str, Any]] = {}
        for entry in source_tree:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                errors.append("source tree entry is invalid")
                continue
            if entry["path"] in tree_by_path:
                errors.append("source tree repeats a path")
            tree_by_path[entry["path"]] = entry
            item_errors, _, _ = self._custody_file(root, entry, snapshots, label=entry["path"])
            errors.extend(item_errors)
        expected_paths = {
            "run_contract.json",
            "measurement_receipt.json",
            "complete.json",
            *{f"pairs/pair_{i:04d}.json" for i in range(REQUIRED_N_PAIRS)},
            *{
                f"stage_{row['checkpoint_name']}_complete.json"
                for row in custody.get("stage_manifests", [])
                if isinstance(row, dict) and isinstance(row.get("checkpoint_name"), str)
            },
        }
        if set(tree_by_path) != expected_paths:
            errors.append("source tree inventory is not exact")

        for label, key, expected_sha in (
            ("run contract", "run_contract", None),
            ("measurement receipt", "measurement_receipt", TRUSTED_SOURCE_ROOTS["measurement_receipt_sha256"]),
            ("completion seal", "complete", TRUSTED_SOURCE_ROOTS["complete_sha256"]),
        ):
            entry = custody.get(key)
            item_errors, _, item = self._custody_file(root, entry, snapshots, label=label)
            errors.extend(item_errors)
            if isinstance(entry, dict) and expected_sha is not None and entry.get("sha256") != expected_sha:
                errors.append(f"{label} is not the reviewed byte root")
            if key == "run_contract" and isinstance(item, dict):
                contract_unsigned = item.get("payload")
                semantic_contract = (
                    {k: v for k, v in contract_unsigned.items() if k != "git_head_at_launch"}
                    if isinstance(contract_unsigned, dict)
                    else contract_unsigned
                )
                if item.get("sha256") != _canonical_sha256(semantic_contract):
                    errors.append("run contract semantic hash mismatch")
                if item.get("sha256") != TRUSTED_SOURCE_ROOTS["run_contract_sha256"]:
                    errors.append("run contract semantic root mismatch")
            if (
                key == "complete"
                and isinstance(item, dict)
                and (
                    item.get("schema") != COMPLETE_SCHEMA
                    or item.get("receipt_sha256") != TRUSTED_SOURCE_ROOTS["measurement_receipt_sha256"]
                )
            ):
                errors.append("completion seal does not bind the reviewed receipt")

        stages = custody.get("stage_manifests")
        pairs = custody.get("pair_records")
        if not isinstance(stages, list) or len(stages) != 3:
            errors.append("corrected wrapper requires exactly three stages")
            stages = []
        if not isinstance(pairs, list) or len(pairs) != REQUIRED_N_PAIRS:
            errors.append("corrected wrapper requires exactly n600 pair custody rows")
            pairs = []
        pair_custody = {row.get("pair_index"): row for row in pairs if isinstance(row, dict)}
        all_pairs: set[int] = set()
        for stage in stages:
            checkpoint = stage.get("checkpoint_name") if isinstance(stage, dict) else None
            item_errors, _, manifest = self._custody_file(
                root,
                stage,
                snapshots,
                label=f"stage manifest {checkpoint}",
            )
            errors.extend(item_errors)
            if not isinstance(manifest, dict):
                continue
            records = manifest.get("records")
            if (
                manifest.get("schema") != STAGE_SCHEMA
                or manifest.get("run_contract_sha256") != self.run_contract_sha256
                or not isinstance(records, list)
                or len(records) != 200
                or manifest.get("state_count") != 200
                or manifest.get("tree_sha256") != _canonical_sha256(records)
                or stage.get("tree_sha256") != manifest.get("tree_sha256")
            ):
                errors.append(f"stage manifest {checkpoint} semantic custody mismatch")
                continue
            for record in records:
                pair_index = record.get("pair_index") if isinstance(record, dict) else None
                if isinstance(pair_index, bool) or not isinstance(pair_index, int):
                    errors.append(f"stage manifest {checkpoint} has invalid pair index")
                    continue
                if pair_index in all_pairs:
                    errors.append("stage manifests repeat pair indices")
                all_pairs.add(pair_index)
                entry = pair_custody.get(pair_index)
                if not isinstance(entry, dict) or any(
                    entry.get(k) != record.get(k) for k in ("path", "bytes", "sha256")
                ):
                    errors.append(f"pair {pair_index} custody does not match stage manifest")
                    continue
                item_errors, _, pair = self._custody_file(
                    root,
                    entry,
                    snapshots,
                    label=f"pair {pair_index}",
                )
                errors.extend(item_errors)
                if not isinstance(pair, dict):
                    continue
                unsigned_pair = {k: v for k, v in pair.items() if k != "record_content_sha256"}
                assignment = pair.get("assignment")
                if (
                    pair.get("schema") != PAIR_SCHEMA
                    or pair.get("run_contract_sha256") != self.run_contract_sha256
                    or pair.get("record_content_sha256") != _canonical_sha256(unsigned_pair)
                    or entry.get("record_content_sha256") != pair.get("record_content_sha256")
                    or not isinstance(assignment, dict)
                    or assignment.get("pair_index") != pair_index
                    or assignment.get("checkpoint_name") != checkpoint
                ):
                    errors.append(f"pair {pair_index} semantic custody mismatch")
        if all_pairs != set(range(REQUIRED_N_PAIRS)):
            errors.append("stage manifests do not cover exactly pair indices 0..599")
        if len(snapshots) != 607:
            errors.append("custody snapshot must cover wrapper plus exactly 606 source files")
        return tuple(errors), root, snapshots

    def validation_errors(
        self,
        *,
        expected_objective_sha256: str,
        expected_scorer_sha256: str,
    ) -> tuple[str, ...]:
        content_errors, root, snapshots = self._content_errors()
        errors = list(content_errors)
        if not _valid_sha256(self.sha256):
            errors.append("corrected wrapper sha256 is invalid")
        elif self.sha256 not in TRUSTED_CORRECTED_WRAPPER_SHA256S:
            errors.append("corrected wrapper sha256 is not code-reviewed")
        if self.status != "completed_source_revalidated":
            errors.append("corrected wrapper is not completed")
        if self.admission_verdict != CORRECTED_ADMISSION_VERDICT:
            errors.append("corrected wrapper verdict changed from NOT_ADMITTED")
        if isinstance(self.n_pairs, bool) or self.n_pairs != REQUIRED_N_PAIRS:
            errors.append(f"receipt n_pairs must equal {REQUIRED_N_PAIRS}")
        if not _valid_sha256(self.objective_sha256):
            errors.append("receipt objective sha256 is invalid")
        elif self.objective_sha256 != expected_objective_sha256:
            errors.append("receipt objective sha256 mismatch")
        if not _valid_sha256(self.scorer_sha256):
            errors.append("receipt scorer sha256 is invalid")
        elif self.scorer_sha256 != expected_scorer_sha256:
            errors.append("corrected wrapper scorer sha256 mismatch")
        if self.run_contract_sha256 != TRUSTED_SOURCE_ROOTS["run_contract_sha256"]:
            errors.append("corrected wrapper run contract sha256 mismatch")
        _before_final_snapshot_verify()
        errors.extend(_verify_snapshot_unchanged(root, snapshots))
        return tuple(errors)


@dataclass(frozen=True)
class _ActivationValidationTransaction:
    """One immutable custody verdict shared by all compiled authority gates."""

    measurement_errors: tuple[str, ...]
    trainer_errors: tuple[str, ...]


@dataclass(frozen=True)
class ExactCostateReusePolicy:
    """Sealed event-controlled ``K_max=2`` policy, live activation refused."""

    enabled: bool = False
    k_max: int = REQUIRED_K_MAX
    n_pairs: int = REQUIRED_N_PAIRS
    objective_sha256: str = TRUSTED_SOURCE_ROOTS["objective_sha256"]
    scorer_sha256: str = TRUSTED_SOURCE_ROOTS["scorer_sha256"]
    receipt: TemporalFidelityReceiptCustody | None = None
    fallback: str = FALLBACK
    provider_current: bool = False

    def __post_init__(self) -> None:
        self._exact_receipt_or_none()
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

    def _exact_receipt_or_none(self) -> TemporalFidelityReceiptCustody | None:
        """Reject subclasses and duck types before dispatching receipt methods."""

        receipt = self.receipt
        if receipt is not None and type(receipt) is not TemporalFidelityReceiptCustody:
            raise TypeError(
                "receipt must have exact type TemporalFidelityReceiptCustody; subclasses and duck types are refused"
            )
        return receipt

    def _measurement_errors_transaction(self) -> tuple[str, ...]:
        """Perform one complete offline n600 custody validation transaction."""

        errors: list[str] = []
        if not _valid_sha256(self.objective_sha256):
            errors.append("expected objective sha256 is missing or invalid")
        if not _valid_sha256(self.scorer_sha256):
            errors.append("expected scorer sha256 is missing or invalid")
        receipt = self._exact_receipt_or_none()
        if receipt is None:
            errors.append("completed temporal-fidelity receipt custody is missing")
        elif _valid_sha256(self.objective_sha256) and _valid_sha256(self.scorer_sha256):
            # Authorization deliberately re-reads bytes rather than trusting an
            # earlier process-local receipt object (TOCTOU fail-closed).
            errors.extend(
                receipt.validation_errors(
                    expected_objective_sha256=self.objective_sha256,
                    expected_scorer_sha256=self.scorer_sha256,
                )
            )
        return tuple(errors)

    def _trainer_errors_from_measurement(
        self,
        measurement_errors: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Derive trainer refusal from one immutable measurement verdict."""

        errors = list(measurement_errors)
        if not self.enabled:
            errors.append("policy is default-off")
        if not self.provider_current:
            errors.append("current-costate provider is unavailable")
        errors.append("live trainer argv is empty")
        return tuple(errors)

    def _activation_validation_transaction(self) -> _ActivationValidationTransaction:
        measurement_errors = self._measurement_errors_transaction()
        return _ActivationValidationTransaction(
            measurement_errors=measurement_errors,
            trainer_errors=self._trainer_errors_from_measurement(measurement_errors),
        )

    def measurement_errors(self) -> tuple[str, ...]:
        """Validate offline n600 evidence without implying trainer activation."""

        return self._measurement_errors_transaction()

    def trainer_activation_errors(self) -> tuple[str, ...]:
        """Return live refusal reasons; receipt admission cannot erase wiring debt."""

        return self._activation_validation_transaction().trainer_errors

    def compile_measurement_contract(self) -> dict[str, Any]:
        """Compile immutable offline evidence requirements; no blind cadence."""

        receipt = self._exact_receipt_or_none()
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
            "receipt": receipt.public_custody() if receipt is not None else None,
            "objective_sha256": self.objective_sha256,
            "scorer_sha256": self.scorer_sha256,
            "live_trainer_argv": [],
            "provider_current": self.provider_current,
            "score_claim": False,
        }

    def compile_activation_contract(self) -> dict[str, Any]:
        """Expose measurement and live-trainer authorities as separate gates."""

        validation = self._activation_validation_transaction()
        measurement_errors = validation.measurement_errors
        trainer_errors = validation.trainer_errors
        return {
            **self.compile_measurement_contract(),
            "measurement_verified": not measurement_errors,
            "measurement_admitted": False,
            "measurement_errors": list(measurement_errors),
            "measurement_authority": (
                "OFFLINE_N600_TRAINING_SIGNAL_ONLY_NOT_ADMITTED" if not measurement_errors else "REFUSED"
            ),
            "corrected_admission_verdict": CORRECTED_ADMISSION_VERDICT,
            "trainer_activation_admitted": False,
            "trainer_activation_errors": list(trainer_errors),
            "trainer_activation_authority": "REFUSED_NO_PROVIDER_OR_ARGV",
        }


def exact_costate_reuse_k2_lever(
    policy: ExactCostateReusePolicy | None = None,
) -> Lever:
    """Return the named default-off DSL leg with empty argv overrides."""

    compiled = (policy or ExactCostateReusePolicy()).compile_activation_contract()
    measurement = "VERIFIED_NO_GO" if compiled["measurement_verified"] else "REFUSED"
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
    "CORRECTED_ADMISSION_VERDICT",
    "FALLBACK",
    "POLICY_NAME",
    "REQUIRED_K_MAX",
    "REQUIRED_N_PAIRS",
    "ExactCostateReusePolicy",
    "TemporalFidelityReceiptCustody",
    "exact_costate_reuse_k2_lever",
]
