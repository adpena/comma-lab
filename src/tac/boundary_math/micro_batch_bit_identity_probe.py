"""Micro-batch (--micro-batch-pairs B>1) bit-identity DECOMPOSITION probe.

CRUX-ENGINEERING FINDING (2026-07-08, MEASURED): the trajectory-A/B gate on the
``--micro-batch-pairs`` 2-4x speed lever CANNOT be dissolved by fixed-order reduction
engineering, because the DOMINANT divergence between the batched twin and the serial
accumulation enters at the FROZEN-SCORER FORWARD KERNEL (EfficientNet-B2 SegNet /
FastViT PoseNet), which is batch-DEPENDENT on the real MLX backends:

    device   segnet max|Δlogit|   argmax px flipped   posenet max|Δ|
    ------   ------------------   -----------------   --------------
    GPU      2.3e-2               11 / 196608         7.7e-3
    CPU      7.1e-5               0                   2.0e-6

(measured with the real upstream adapter over K=4 random 384x512 frames; see
``tools/micro_batch_bit_identity_probe.py`` to reproduce.) This is UPSTREAM of any loss
or gradient reduction: ``segnet(f1_batch)[k] != segnet(f1_batch[k:k+1])[0]`` bit-for-bit,
so the per-pair loss ``L_k`` computed from the batched forward already differs from the
serial per-pair ``L_k`` before any accumulation happens. It is a GPU/CPU conv/matmul
kernel tiling property (sister of the #348 ``mlx_gpu_crossprocess_nondeterminism_v1``
family), NOT a reordering we can control.

The reduction/accumulation-ORDER sources (what the operator's fix (a)/(b) anticipated)
are SECONDARY and only surface where the scorer IS batch-invariant. This module MEASURES
them in isolation with a batch-INVARIANT mock scorer (a linear per-pixel/per-frame op,
whose batched forward+backward are provably 0.0 batch-dependent), so the residual is
PURELY the loss/grad reduction order:

* the batched twin builds ``L = mean_k L_k`` and takes ONE ``value_and_grad`` -> MLX's
  backward accumulates the K per-pair contributions into the SHARED witness params
  (out_tex / in_proj / code ...) in a graph-internal order that differs from the serial
  explicit left-fold ``accum = g0; accum += g1; ...`` -> ~1e-3..1e-2 max|Δ| on the grad
  tree (hidden by the global-L2 ~1e-7 metric the equivalence tests use).

CONSEQUENCE (the measured diagnostic): full bit-identity of B>1 to the serial path at any
speedup > 1x is IMPOSSIBLE on the real MLX scorer (GPU or CPU), because the entire win is
the batched scorer forward (GPU ~1.56x / CPU ~1.75x at K=8) which is the exact op that is
not batch-invariant; a bit-identical construction requires a per-pair (batch-1) scorer
forward == the serial path == 1.0x. Operator directive 2026-07-12 explicitly WAIVES
bit identity for the TRAINING trajectory: the runtime bar is functional loss/gradient parity
plus measured wall-clock improvement. The drift remains telemetry. This module intentionally
cannot mint authoritative functional parity or runtime admission from persisted JSON because
those bytes cannot attest execution. Disk rows establish only that *reported* metrics are
schema-valid and within the registered diagnostic policy.
The waiver never grants score authority; exact byte-closed evaluator replay is unchanged.

MEANS, not ends. Nothing here moves the canonical ``reports/latest.md`` contest-CPU pointer. Only
a byte-closed n600 ``upstream/evaluate.py`` row does.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# MEASURED empirical anchors (2026-07-08, real upstream adapter, K=4, 384x512).
# Recorded for provenance + as the classification reference; the CLI re-measures live.
# ─────────────────────────────────────────────────────────────────────────────
MEASURED_SCORER_FWD_GPU_SEG_MAXABS = 2.259e-2
MEASURED_SCORER_FWD_GPU_ARGMAX_FLIPS = 11        # of 384*512 = 196608 px
MEASURED_SCORER_FWD_GPU_POSE_MAXABS = 7.728e-3
MEASURED_SCORER_FWD_CPU_SEG_MAXABS = 7.105e-5
MEASURED_SCORER_FWD_CPU_ARGMAX_FLIPS = 0
MEASURED_SCORER_FWD_CPU_POSE_MAXABS = 2.027e-6
# scorer-forward microbench speedup (K=8, ONE batched fwd vs K per-pair fwds).
MEASURED_SCORER_FWD_SPEEDUP_GPU = 1.56
MEASURED_SCORER_FWD_SPEEDUP_CPU = 1.75
# argmax-invariance is the load-bearing sub-property: SegNet d_seg is an argmax rate.
# CPU flips 0 px; GPU flips 11 px (0.006%). The reduction-order grad drift on a
# batch-INVARIANT mock scorer (source B, isolated), observed max|Δ| on the grad tree:
MEASURED_REDUCTION_ORDER_GRAD_MAXABS_MOCK = 3.9e-3  # K=4, ce, out_tex leaf


# Admission policy is deliberately module-owned. Evidence may report these values, but it
# cannot select them. Keeping the policy as immutable tuples also prevents a caller from
# weakening coverage by passing an alternate mapping into the admission classifier.
REQUIRED_FUNCTIONAL_PARITY_LEVERS = ("chroma", "phase", "temporal", "area", "full_v9")
CANONICAL_CONFIG_ID = "v9_cgauge_432"
CANONICAL_FUNCTIONAL_DEVICE = "gpu"
CANONICAL_SCORER_SURFACE = "real_frozen_v9"
CANONICAL_MINIMUM_SPEEDUP = 1.0
_CANONICAL_PARITY_POLICY = (
    # lever, loss_abs, loss_rel, grad_rel_l2, grad_maxabs, backend
    ("chroma", 1e-4, 1e-4, 1e-4, 1e-2, "metal"),
    ("phase", 1e-4, 1e-4, 1e-4, 1e-2, "metal"),
    ("temporal", 1e-4, 1e-4, 1e-4, 1e-2, "metal"),
    ("area", 1e-4, 1e-4, 1e-4, 1e-2, "mlx_vectorized"),
    ("full_v9", 1e-4, 1e-4, 1e-4, 1e-2, "metal+mlx"),
)


def _canonical_parity_policy(lever: str) -> tuple[float, float, float, float, str]:
    for name, loss_abs, loss_rel, grad_rel, grad_maxabs, backend in _CANONICAL_PARITY_POLICY:
        if lever == name:
            return loss_abs, loss_rel, grad_rel, grad_maxabs, backend
    raise ValueError(f"unknown functional-parity lever: {lever!r}")


@dataclass(frozen=True)
class ReductionOrderDrift:
    """Pure reduction/accumulation-ORDER drift (source B) measured with a batch-INVARIANT
    mock scorer, so the scorer-forward kernel (source A) contributes exactly 0.0 and the
    residual is only the loss/grad accumulation order.

    ``grad_maxabs`` is the max absolute per-leaf difference between the batched twin grad
    and the serial left-fold mean-of-per-pair grad (the trajectory-relevant metric — the
    global-L2 ``grad_rel_l2`` hides it). ``loss_abs`` is the loss-scalar difference.
    """

    K: int
    seg_form: str
    grad_maxabs: float
    grad_rel_l2: float
    loss_abs: float
    worst_leaf: str


@dataclass(frozen=True)
class FunctionalParityReceipt:
    """Reusable measured receipt for one routed micro-batch loss leg."""

    lever: str
    K: int
    loss_abs: float
    loss_rel: float
    grad_rel_l2: float
    grad_maxabs: float
    loss_rel_tolerance: float
    grad_rel_tolerance: float
    grad_maxabs_tolerance: float
    passed: bool
    backend_receipt: str
    config_id: str = ""
    device: str = ""
    scorer_surface: str = ""
    faithful_scale: bool = False
    measurement_artifact: str = ""
    measurement_artifact_sha256: str = ""
    scorer_fingerprint_sha256: str = ""
    loss_abs_tolerance: float = 1e-4

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TrainingAdmissionReceipt:
    """Fail-closed training verdict plus functional and timing telemetry.

    A JSON timing file cannot attest that the claimed benchmark actually executed. Therefore
    this module has no persisted-evidence path to ``training_throughput_admitted=True``.
    """

    required_levers: tuple[str, ...]
    covered_levers: tuple[str, ...]
    missing_levers: tuple[str, ...]
    failed_levers: tuple[str, ...]
    invalid_context_levers: tuple[str, ...]
    reported_end_to_end_speedup: float
    minimum_speedup: float
    required_config_id: str
    required_device: str
    required_scorer_surface: str
    functional_parity_passed: bool
    training_throughput_admitted: bool
    reported_functional_metrics_within_tolerance: bool = False
    functional_telemetry_valid: bool = False
    timing_telemetry_valid: bool = False
    runtime_admission_evidence_present: bool = False
    admission_blocker: str = (
        "persisted timing JSON is schema-validated telemetry only and cannot attest execution"
    )
    training_only: bool = True
    no_score_authority: bool = True

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ArtifactBinding:
    """Byte custody for a durable artifact; a digest-shaped string is insufficient."""

    path: str
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CompiledConfigIdentity:
    """Identity recomputed from the canonical typed V9+MicroBatch compiler."""

    config_id: str
    micro_batch_pairs: int
    typed_config_hash: str
    compiled_argv_sha256: str
    flag_manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ScorerFingerprint:
    """Fingerprint of the two canonical scorer content files, never caller-selected files."""

    segnet_sha256: str
    posenet_sha256: str
    aggregate_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class EndToEndTimingTelemetry:
    """Schema-validated *reported* full-step timing values; never execution authority."""

    reported_serial_seconds: float
    reported_micro_batch_seconds: float
    reported_end_to_end_speedup: float
    device: str
    benchmark_surface: str
    faithful_scale: bool
    serial_measured_steps: int
    micro_batch_measured_steps: int
    warmup_steps: int
    clock: str
    config_identity: CompiledConfigIdentity
    measurement_artifact: ArtifactBinding

    def as_dict(self) -> dict[str, Any]:
        return {
            "reported_serial_seconds": self.reported_serial_seconds,
            "reported_micro_batch_seconds": self.reported_micro_batch_seconds,
            "reported_end_to_end_speedup": self.reported_end_to_end_speedup,
            "device": self.device,
            "benchmark_surface": self.benchmark_surface,
            "faithful_scale": self.faithful_scale,
            "serial_measured_steps": self.serial_measured_steps,
            "micro_batch_measured_steps": self.micro_batch_measured_steps,
            "warmup_steps": self.warmup_steps,
            "clock": self.clock,
            "config_identity": self.config_identity.as_dict(),
            "measurement_artifact": self.measurement_artifact.as_dict(),
        }


_VALIDATION_TOKEN = object()


class SchemaValidatedFunctionalParityTelemetry:
    """Durable, schema-validated *reported* metrics with no execution attestation."""

    __slots__ = ("config_identity", "measurement_artifact", "receipt", "scorer_fingerprint")

    def __init__(self, receipt, measurement_artifact, config_identity, scorer_fingerprint, *, _token):
        if _token is not _VALIDATION_TOKEN:
            raise TypeError(
                "use build_schema_validated_functional_parity_telemetry or telemetry loader")
        self.receipt = receipt
        self.measurement_artifact = measurement_artifact
        self.config_identity = config_identity
        self.scorer_fingerprint = scorer_fingerprint

    def as_dict(self) -> dict[str, Any]:
        reported = self.receipt.as_dict()
        reported.pop("passed", None)
        reported["reported_metrics_within_tolerance"] = _parity_passed(self.receipt)
        return {
            "reported_metrics": reported,
            "measurement_artifact": self.measurement_artifact.as_dict(),
            "config_identity": self.config_identity.as_dict(),
            "scorer_fingerprint": self.scorer_fingerprint.as_dict(),
            "telemetry_only": True,
            "execution_attested": False,
            "can_establish_functional_parity": False,
            "can_authorize_training": False,
        }


class SchemaValidatedEndToEndTimingTelemetry:
    """Schema-validated timing *telemetry*, never an admission-capable execution receipt.

    Disk bytes can prove only what the JSON says, not that the benchmark ran. The wrapper is
    useful for comparison/provenance and is intentionally rejected as training authority.
    """

    __slots__ = ("receipt",)

    def __init__(self, receipt: EndToEndTimingTelemetry, *, _token):
        if _token is not _VALIDATION_TOKEN:
            raise TypeError("use build_schema_validated_timing_telemetry or telemetry loader")
        self.receipt = receipt

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.receipt.as_dict(),
            "telemetry_only": True,
            "can_authorize_training": False,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _durable_artifact_roots() -> tuple[Path, ...]:
    """Allowlisted evidence roots.

    Repo results and the two governed SSD Pact tiers are durable campaign surfaces.  Arbitrary
    home caches, source-tree fixtures, Downloads, and scratch aliases are not evidence custody.
    Keeping this as a helper (rather than a constant) lets tests replace ``_repo_root`` without
    retaining a stale import-time path.
    """
    return (
        (_repo_root() / "experiments" / "results").resolve(),
        Path("/Volumes/VertigoDataTier/pact").resolve(),
        Path("/Volumes/APDataStore/pact").resolve(),
    )


def _artifact_binding(path_like: str | Path) -> ArtifactBinding:
    try:
        path = Path(path_like).expanduser().resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("admission artifact path is malformed") from exc
    if not any(_under(path, root) for root in _durable_artifact_roots()):
        roots = ", ".join(str(root) for root in _durable_artifact_roots())
        raise ValueError(
            f"telemetry artifact must be under an allowlisted durable result root; "
            f"got {path}; allowed roots: {roots}"
        )
    if not path.is_file():
        raise ValueError(f"admission artifact does not exist as a file: {path}")
    return ArtifactBinding(path=str(path), bytes=path.stat().st_size, sha256=_sha256_file(path))


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _require_field(mapping: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _require_exact_fields(
    mapping: dict[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(mapping)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{context} fields do not match schema; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_finite_number(
    value: Any, *, context: str, positive: bool = False, nonnegative: bool = False
) -> float:
    # JSON booleans are integers in Python; exact type checks prevent True from becoming 1.0.
    if type(value) not in (int, float):
        raise ValueError(f"{context} must be a JSON number, not {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{context} must be finite")
    if positive and not out > 0.0:
        raise ValueError(f"{context} must be positive")
    if nonnegative and out < 0.0:
        raise ValueError(f"{context} must be non-negative")
    return out


def _require_hex_digest(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(f"{context} must be a 64-character hexadecimal digest")
    return value.lower()


def _compiled_identity_from_mapping(value: Any, *, context: str) -> CompiledConfigIdentity:
    row = _require_mapping(value, context=context)
    _require_exact_fields(
        row,
        {"config_id", "micro_batch_pairs", "typed_config_hash", "compiled_argv_sha256",
         "flag_manifest_sha256"},
        context=context,
    )
    if not isinstance(row["config_id"], str):
        raise ValueError(f"{context} config_id must be a string")
    if type(row["micro_batch_pairs"]) is not int:
        raise ValueError(f"{context} micro_batch_pairs must be an integer")
    for field in ("typed_config_hash", "compiled_argv_sha256", "flag_manifest_sha256"):
        _require_hex_digest(row[field], context=f"{context} {field}")
    try:
        return CompiledConfigIdentity(**row)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is malformed") from exc


def _scorer_fingerprint_from_mapping(value: Any, *, context: str) -> ScorerFingerprint:
    row = _require_mapping(value, context=context)
    _require_exact_fields(
        row, {"segnet_sha256", "posenet_sha256", "aggregate_sha256"}, context=context)
    for field in ("segnet_sha256", "posenet_sha256", "aggregate_sha256"):
        _require_hex_digest(row[field], context=f"{context} {field}")
    try:
        return ScorerFingerprint(**row)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is malformed") from exc


def _artifact_binding_from_mapping(value: Any, *, context: str) -> ArtifactBinding:
    row = _require_mapping(value, context=context)
    _require_exact_fields(row, {"path", "bytes", "sha256"}, context=context)
    try:
        binding = ArtifactBinding(**row)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} is malformed") from exc
    if not isinstance(binding.path, str) or not binding.path:
        raise ValueError(f"{context} path must be a non-empty string")
    if type(binding.bytes) is not int or binding.bytes < 0:
        raise ValueError(f"{context} bytes must be a non-negative integer")
    _require_hex_digest(binding.sha256, context=f"{context} sha256")
    return binding


def _verify_artifact(binding: ArtifactBinding) -> bool:
    try:
        actual = _artifact_binding(binding.path)
    except (OSError, TypeError, ValueError):
        return False
    return (actual.bytes == int(binding.bytes)
            and actual.sha256 == str(binding.sha256).lower())


def canonical_compiled_config_identity(micro_batch_pairs: int) -> CompiledConfigIdentity:
    """Recompile the one canonical V9 configuration; do not trust a caller manifest."""
    from tac.witness_dsl.curriculum_dsl import MicroBatch
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config

    K = int(micro_batch_pairs)
    compiled = compile_v9_cgauge_432_launch_config()
    program = compiled.typed.to_program().with_lever(MicroBatch(K))
    violations = program.validate()
    if violations:
        raise ValueError(f"canonical V9+MicroBatch({K}) compile invalid: {violations[:4]}")
    argv = program.compile_trainer_argv()
    flag_manifest = program.flag_dict()
    return CompiledConfigIdentity(
        config_id="v9_cgauge_432", micro_batch_pairs=K,
        typed_config_hash=compiled.typed.typed_config_hash(),
        compiled_argv_sha256=_canonical_json_sha256(argv),
        flag_manifest_sha256=_canonical_json_sha256(flag_manifest),
    )


def _verify_config_identity(identity: CompiledConfigIdentity) -> bool:
    try:
        return identity == canonical_compiled_config_identity(identity.micro_batch_pairs)
    except (TypeError, ValueError):
        return False


def canonical_scorer_fingerprint() -> ScorerFingerprint:
    """Hash the canonical scorer bytes at their fixed repo paths."""
    model_root = _repo_root() / "upstream" / "models"
    seg = _sha256_file(model_root / "segnet.safetensors")
    pose = _sha256_file(model_root / "posenet.safetensors")
    aggregate = _canonical_json_sha256({"segnet.safetensors": seg, "posenet.safetensors": pose})
    return ScorerFingerprint(segnet_sha256=seg, posenet_sha256=pose,
                             aggregate_sha256=aggregate)


def _verify_scorer_fingerprint(fingerprint: ScorerFingerprint) -> bool:
    try:
        return fingerprint == canonical_scorer_fingerprint()
    except OSError:
        return False


def make_functional_parity_receipt(
    *,
    lever: str,
    K: int,
    batched_loss: float,
    serial_mean_loss: float,
    grad_rel_l2: float,
    grad_maxabs: float,
    backend_receipt: str,
    config_id: str = "",
    device: str = "",
    scorer_surface: str = "",
    faithful_scale: bool = False,
    measurement_artifact: str = "",
    measurement_artifact_sha256: str = "",
    scorer_fingerprint_sha256: str = "",
) -> FunctionalParityReceipt:
    """Classify diagnostic loss/gradient parity under module-owned per-lever thresholds."""
    lever = str(lever)
    (loss_abs_tolerance, loss_rel_tolerance, grad_rel_tolerance,
     grad_maxabs_tolerance, _expected_backend) = _canonical_parity_policy(lever)
    loss_abs = abs(float(batched_loss) - float(serial_mean_loss))
    loss_rel = loss_abs / (abs(float(serial_mean_loss)) + 1e-12)
    metrics = (loss_abs, loss_rel, float(grad_rel_l2), float(grad_maxabs))
    tolerances = (float(loss_abs_tolerance), float(loss_rel_tolerance),
                  float(grad_rel_tolerance), float(grad_maxabs_tolerance))
    passed = (
        all(math.isfinite(value) and value >= 0.0 for value in metrics + tolerances)
        and loss_abs <= float(loss_abs_tolerance)
        and loss_rel <= float(loss_rel_tolerance)
        and float(grad_rel_l2) <= float(grad_rel_tolerance)
        and float(grad_maxabs) <= float(grad_maxabs_tolerance)
    )
    return FunctionalParityReceipt(
        lever=lever, K=int(K), loss_abs=loss_abs, loss_rel=loss_rel,
        grad_rel_l2=float(grad_rel_l2), grad_maxabs=float(grad_maxabs),
        loss_rel_tolerance=float(loss_rel_tolerance),
        grad_rel_tolerance=float(grad_rel_tolerance),
        grad_maxabs_tolerance=float(grad_maxabs_tolerance), passed=bool(passed),
        backend_receipt=str(backend_receipt),
        config_id=str(config_id), device=str(device), scorer_surface=str(scorer_surface),
        faithful_scale=bool(faithful_scale), measurement_artifact=str(measurement_artifact),
        measurement_artifact_sha256=str(measurement_artifact_sha256),
        scorer_fingerprint_sha256=str(scorer_fingerprint_sha256),
        loss_abs_tolerance=float(loss_abs_tolerance),
    )


def build_schema_validated_functional_parity_telemetry(
    measurement_artifact: str | Path,
) -> SchemaValidatedFunctionalParityTelemetry:
    """Validate reported functional metrics and custody without attesting execution.

    A digest proves which JSON bytes were read. It cannot prove that Metal, the scorer, or even
    the named benchmark executed. The returned object is telemetry only and can establish at most
    ``reported_metrics_within_tolerance``; it can never establish functional parity or authorize
    training.
    """
    artifact = _artifact_binding(measurement_artifact)
    try:
        payload = json.loads(Path(artifact.path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("functional measurement artifact must be valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "micro_batch_functional_telemetry.v1":
        raise ValueError("functional measurement artifact has wrong/missing schema")
    context = "functional measurement artifact"
    _require_exact_fields(
        payload,
        {
            "schema", "lever", "K", "batched_loss", "serial_mean_loss", "grad_rel_l2",
            "grad_maxabs", "loss_abs_tolerance", "loss_rel_tolerance",
            "grad_rel_tolerance", "grad_maxabs_tolerance", "reported_backend",
            "reported_device", "reported_scorer_surface", "reported_spatial_scale",
            "config_identity", "scorer_fingerprint",
        },
        context=context,
    )
    lever = _require_field(payload, "lever", context=context)
    if not isinstance(lever, str):
        raise ValueError("functional measurement lever must be a string")
    (loss_abs_tol, loss_rel_tol, grad_rel_tol, grad_maxabs_tol,
     expected_backend) = _canonical_parity_policy(lever)
    canonical_tolerances = {
        "loss_abs_tolerance": loss_abs_tol,
        "loss_rel_tolerance": loss_rel_tol,
        "grad_rel_tolerance": grad_rel_tol,
        "grad_maxabs_tolerance": grad_maxabs_tol,
    }
    for field, expected in canonical_tolerances.items():
        actual = _require_field(payload, field, context=context)
        if type(actual) not in (int, float) or not math.isfinite(float(actual)):
            raise ValueError(f"functional measurement {field} must be a finite number")
        if float(actual) != expected:
            raise ValueError(
                f"functional measurement {field} must equal module-owned canonical "
                f"threshold {expected} for {lever}"
            )
    K = _require_field(payload, "K", context=context)
    if type(K) is not int or K < 2:
        raise ValueError("functional measurement K must be an integer >= 2")
    backend = _require_field(payload, "reported_backend", context=context)
    device = _require_field(payload, "reported_device", context=context)
    scorer_surface = _require_field(payload, "reported_scorer_surface", context=context)
    faithful_scale = _require_field(payload, "reported_spatial_scale", context=context)
    if backend != expected_backend:
        raise ValueError(f"functional measurement backend must be canonical {expected_backend!r}")
    if device != CANONICAL_FUNCTIONAL_DEVICE:
        raise ValueError(
            f"functional measurement device must be canonical {CANONICAL_FUNCTIONAL_DEVICE!r}"
        )
    if scorer_surface != CANONICAL_SCORER_SURFACE:
        raise ValueError(
            f"functional measurement scorer surface must be canonical {CANONICAL_SCORER_SURFACE!r}"
        )
    if faithful_scale is not True:
        raise ValueError("functional measurement reported_spatial_scale must be true")
    identity = _compiled_identity_from_mapping(
        _require_field(payload, "config_identity", context=context),
        context="functional config_identity",
    )
    scorer = _scorer_fingerprint_from_mapping(
        _require_field(payload, "scorer_fingerprint", context=context),
        context="functional scorer_fingerprint",
    )
    try:
        batched_loss = _require_finite_number(
            _require_field(payload, "batched_loss", context=context),
            context="functional measurement batched_loss",
        )
        serial_mean_loss = _require_finite_number(
            _require_field(payload, "serial_mean_loss", context=context),
            context="functional measurement serial_mean_loss",
        )
        grad_rel_l2 = _require_finite_number(
            _require_field(payload, "grad_rel_l2", context=context),
            context="functional measurement grad_rel_l2", nonnegative=True,
        )
        grad_maxabs = _require_finite_number(
            _require_field(payload, "grad_maxabs", context=context),
            context="functional measurement grad_maxabs", nonnegative=True,
        )
        receipt = make_functional_parity_receipt(
            lever=lever,
            K=K,
            batched_loss=batched_loss,
            serial_mean_loss=serial_mean_loss,
            grad_rel_l2=grad_rel_l2,
            grad_maxabs=grad_maxabs,
            backend_receipt=backend,
            config_id=identity.config_id,
            device=device,
            scorer_surface=scorer_surface,
            faithful_scale=True,
        )
    except (OverflowError, TypeError) as exc:
        raise ValueError("functional measurement metrics are malformed") from exc
    if not _verify_config_identity(identity):
        raise ValueError("functional receipt config identity is not the canonical compiled V9 argv")
    if identity.micro_batch_pairs != receipt.K:
        raise ValueError("functional receipt K does not match compiled config identity")
    if not _verify_scorer_fingerprint(scorer):
        raise ValueError("functional receipt scorer fingerprint does not match canonical scorer bytes")
    return SchemaValidatedFunctionalParityTelemetry(
        receipt, artifact, identity, scorer, _token=_VALIDATION_TOKEN)


def build_schema_validated_timing_telemetry(
    measurement_artifact: str | Path,
) -> SchemaValidatedEndToEndTimingTelemetry:
    """Validate timing JSON as telemetry only.

    This proves schema, arithmetic, config identity, and byte custody. It cannot prove that the
    claimed benchmark executed, so its result is categorically non-admitting.
    """
    artifact = _artifact_binding(measurement_artifact)
    try:
        payload = json.loads(Path(artifact.path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("timing measurement artifact must be valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "micro_batch_e2e_timing.v1":
        raise ValueError("timing measurement artifact has wrong/missing schema")
    context = "timing measurement artifact"
    _require_exact_fields(
        payload,
        {
            "schema", "serial_seconds", "micro_batch_seconds", "device", "benchmark_surface",
            "faithful_scale", "serial_measured_steps", "micro_batch_measured_steps",
            "warmup_steps", "clock", "config_identity",
        },
        context=context,
    )
    serial = _require_finite_number(
        _require_field(payload, "serial_seconds", context=context),
        context="timing serial_seconds", positive=True)
    batched = _require_finite_number(
        _require_field(payload, "micro_batch_seconds", context=context),
        context="timing micro_batch_seconds", positive=True)
    config_identity = _compiled_identity_from_mapping(
        _require_field(payload, "config_identity", context=context),
        context="timing config_identity",
    )
    if not _verify_config_identity(config_identity):
        raise ValueError("timing receipt config identity is not the canonical compiled V9 argv")
    device = _require_field(payload, "device", context=context)
    surface = _require_field(payload, "benchmark_surface", context=context)
    faithful_scale = _require_field(payload, "faithful_scale", context=context) is True
    serial_steps_raw = _require_field(payload, "serial_measured_steps", context=context)
    batched_steps_raw = _require_field(payload, "micro_batch_measured_steps", context=context)
    warmup_steps_raw = _require_field(payload, "warmup_steps", context=context)
    clock = _require_field(payload, "clock", context=context)
    if any(type(value) is not int for value in (
        serial_steps_raw, batched_steps_raw, warmup_steps_raw
    )):
        raise ValueError("timing step counts must be integers")
    serial_steps = serial_steps_raw
    batched_steps = batched_steps_raw
    warmup_steps = warmup_steps_raw
    if not (
        config_identity.micro_batch_pairs >= 2
        and device == CANONICAL_FUNCTIONAL_DEVICE
        and surface == "full_v9_training_step"
        and faithful_scale
        and serial_steps == batched_steps > 0
        and warmup_steps > 0
        and clock == "time.perf_counter"
    ):
        raise ValueError("timing artifact is not a faithful GPU full-V9 equal-step benchmark")
    receipt = EndToEndTimingTelemetry(
        reported_serial_seconds=serial, reported_micro_batch_seconds=batched,
        reported_end_to_end_speedup=serial / batched,
        device=device, benchmark_surface=surface, faithful_scale=faithful_scale,
        serial_measured_steps=serial_steps, micro_batch_measured_steps=batched_steps,
        warmup_steps=warmup_steps, clock=clock,
        config_identity=config_identity, measurement_artifact=artifact)
    return SchemaValidatedEndToEndTimingTelemetry(receipt, _token=_VALIDATION_TOKEN)


def _functional_from_telemetry_row(
    row: dict[str, Any],
) -> SchemaValidatedFunctionalParityTelemetry:
    row = _require_mapping(row, context="functional receipt row")
    _require_exact_fields(
        row,
        {
            "reported_metrics", "measurement_artifact", "config_identity", "scorer_fingerprint",
            "telemetry_only", "execution_attested", "can_establish_functional_parity",
            "can_authorize_training",
        },
        context="functional telemetry row",
    )
    if not (
        row["telemetry_only"] is True
        and row["execution_attested"] is False
        and row["can_establish_functional_parity"] is False
        and row["can_authorize_training"] is False
    ):
        raise ValueError("functional telemetry authority labels must remain fail-closed")
    artifact = _artifact_binding_from_mapping(
        _require_field(row, "measurement_artifact", context="functional receipt row"),
        context="functional receipt artifact binding",
    )
    if not _verify_artifact(artifact):
        raise ValueError("functional receipt artifact bytes/SHA-256 do not match")
    validated = build_schema_validated_functional_parity_telemetry(artifact.path)
    if validated.as_dict() != row:
        raise ValueError("functional receipt wrapper does not match validated artifact payload")
    return validated


def load_functional_parity_telemetry(
    path: str | Path,
) -> tuple[SchemaValidatedFunctionalParityTelemetry, ...]:
    """Load reported functional rows with schema/custody checks and no execution authority."""
    bundle = _artifact_binding(path)
    try:
        raw = json.loads(Path(bundle.path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("functional parity receipt bundle must be valid JSON") from exc
    rows = raw.get("functional_parity_telemetry", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("functional parity receipt JSON must be a list or keyed list")
    return tuple(_functional_from_telemetry_row(row) for row in rows)


def load_timing_telemetry(path: str | Path) -> SchemaValidatedEndToEndTimingTelemetry:
    """Load schema-validated timing telemetry; this can never authorize training."""
    bundle = _artifact_binding(path)
    try:
        raw = json.loads(Path(bundle.path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("timing telemetry receipt bundle must be valid JSON") from exc
    row = _require_mapping(raw, context="timing telemetry receipt row")
    artifact = _artifact_binding_from_mapping(
        _require_field(row, "measurement_artifact", context="timing telemetry receipt row"),
        context="timing telemetry artifact binding",
    )
    if not _verify_artifact(artifact):
        raise ValueError("timing artifact bytes/SHA-256 do not match")
    validated = build_schema_validated_timing_telemetry(artifact.path)
    if validated.as_dict() != row:
        raise ValueError("timing receipt wrapper does not match validated artifact payload")
    return validated


def _parity_passed(receipt: FunctionalParityReceipt) -> bool:
    """Recompute the verdict; never trust the serialized ``passed`` boolean."""
    try:
        metrics = (receipt.loss_abs, receipt.loss_rel, receipt.grad_rel_l2, receipt.grad_maxabs)
        tolerances = (receipt.loss_abs_tolerance, receipt.loss_rel_tolerance,
                      receipt.grad_rel_tolerance, receipt.grad_maxabs_tolerance)
        return bool(
            all(math.isfinite(float(value)) and float(value) >= 0.0 for value in metrics)
            and all(math.isfinite(float(value)) and float(value) >= 0.0 for value in tolerances)
            and float(receipt.loss_abs) <= float(receipt.loss_abs_tolerance)
            and float(receipt.loss_rel) <= float(receipt.loss_rel_tolerance)
            and float(receipt.grad_rel_l2) <= float(receipt.grad_rel_tolerance)
            and float(receipt.grad_maxabs) <= float(receipt.grad_maxabs_tolerance)
        )
    except (AttributeError, OverflowError, TypeError, ValueError):
        return False


def _functional_telemetry_matches_artifact(validated: Any) -> bool:
    """Rebuild from evidence bytes so even a stolen private token cannot forge custody."""
    if not isinstance(validated, SchemaValidatedFunctionalParityTelemetry):
        return False
    try:
        rebuilt = build_schema_validated_functional_parity_telemetry(
            validated.measurement_artifact.path)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return rebuilt.as_dict() == validated.as_dict()


def _timing_telemetry_matches_artifact(validated: Any) -> bool:
    if not isinstance(validated, SchemaValidatedEndToEndTimingTelemetry):
        return False
    try:
        rebuilt = build_schema_validated_timing_telemetry(
            validated.receipt.measurement_artifact.path)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return rebuilt.as_dict() == validated.as_dict()


def _timing_telemetry_is_valid(
    timing: SchemaValidatedEndToEndTimingTelemetry | None,
    *,
    shared_config_identity: CompiledConfigIdentity | None,
) -> bool:
    """Validate telemetry context without treating disk JSON as execution authority."""
    if not isinstance(timing, SchemaValidatedEndToEndTimingTelemetry):
        return False
    try:
        row = timing.receipt
        actual = float(row.reported_serial_seconds) / float(row.reported_micro_batch_seconds)
        return bool(
            shared_config_identity is not None
            and math.isfinite(float(row.reported_serial_seconds))
            and float(row.reported_serial_seconds) > 0.0
            and math.isfinite(float(row.reported_micro_batch_seconds))
            and float(row.reported_micro_batch_seconds) > 0.0
            and math.isfinite(float(row.reported_end_to_end_speedup))
            and float(row.reported_end_to_end_speedup) == actual
            and row.device == CANONICAL_FUNCTIONAL_DEVICE
            and row.benchmark_surface == "full_v9_training_step"
            and row.faithful_scale is True
            and row.serial_measured_steps == row.micro_batch_measured_steps > 0
            and row.warmup_steps > 0
            and row.clock == "time.perf_counter"
            and row.config_identity == shared_config_identity
            and _timing_telemetry_matches_artifact(timing)
            and _verify_config_identity(row.config_identity)
            and _verify_artifact(row.measurement_artifact)
        )
    except (AttributeError, OverflowError, TypeError, ValueError, ZeroDivisionError):
        return False


def classify_training_admission(
    receipts: (list[SchemaValidatedFunctionalParityTelemetry]
               | tuple[SchemaValidatedFunctionalParityTelemetry, ...]),
    *,
    timing_receipt: SchemaValidatedEndToEndTimingTelemetry | None = None,
    reported_end_to_end_speedup: float | None = None,
) -> TrainingAdmissionReceipt:
    """Classify parity and fail closed on runtime training admission.

    Per-lever reported rows remain useful diagnostics, but cannot establish functional parity or
    authorize the
    training path. Chroma/phase/temporal must prove that their new Metal forward actually fired;
    area is the vectorized MLX leg; ``full_v9`` proves the combined live-state/render/scorer surface.

    The required lever set and all context/tolerance/backend expectations are module-owned.
    Persisted timing JSON is telemetry only: it cannot attest actual benchmark execution, so this
    function deliberately has no path to either ``functional_parity_passed=True`` or
    ``training_throughput_admitted=True``. A future runtime
    admission surface must consume an in-process benchmark result with non-forgeable execution
    custody; that mechanism is not implemented here.
    """
    required_levers = REQUIRED_FUNCTIONAL_PARITY_LEVERS
    # A bare scalar is retained only as legacy telemetry. It can never authorize training.
    by_lever: dict[str, list[SchemaValidatedFunctionalParityTelemetry]] = {}
    unchecked_receipt_present = False
    for validated in receipts:
        if not isinstance(validated, SchemaValidatedFunctionalParityTelemetry):
            unchecked_receipt_present = True
            continue
        try:
            lever = validated.receipt.lever
        except AttributeError:
            unchecked_receipt_present = True
            continue
        if lever not in required_levers:
            unchecked_receipt_present = True
            continue
        by_lever.setdefault(lever, []).append(validated)
    missing = tuple(lever for lever in required_levers if lever not in by_lever)
    failed = tuple(
        lever for lever in required_levers
        if lever in by_lever and not all(_parity_passed(item.receipt)
                                         for item in by_lever[lever])
    )
    try:
        canonical_scorer = canonical_scorer_fingerprint() if by_lever else None
    except OSError:
        canonical_scorer = None

    all_validated = [item for group in by_lever.values() for item in group]
    identities = [
        item.config_identity
        for item in all_validated
        if isinstance(getattr(item, "config_identity", None), CompiledConfigIdentity)
    ]
    shared_config_identity = (
        identities[0]
        if identities and len(identities) == len(all_validated)
        and all(identity == identities[0] for identity in identities)
        else None
    )
    try:
        shared_identity_ok = bool(
            shared_config_identity is not None
            and _verify_config_identity(shared_config_identity)
            and shared_config_identity.config_id == CANONICAL_CONFIG_ID
            and shared_config_identity.micro_batch_pairs >= 2
            and all(
                item.config_identity == shared_config_identity
                and shared_config_identity.micro_batch_pairs == item.receipt.K
                for item in all_validated
            )
        )
    except (AttributeError, TypeError, ValueError):
        shared_identity_ok = False

    def _telemetry_context_valid(validated: SchemaValidatedFunctionalParityTelemetry) -> bool:
        try:
            if not shared_identity_ok or shared_config_identity is None:
                return False
            receipt = validated.receipt
            (*_thresholds, expected_backend) = _canonical_parity_policy(receipt.lever)
            return bool(
                shared_config_identity.micro_batch_pairs == receipt.K
                and receipt.device == CANONICAL_FUNCTIONAL_DEVICE
                and receipt.scorer_surface == CANONICAL_SCORER_SURFACE
                and receipt.faithful_scale is True
                and validated.config_identity == shared_config_identity
                and _functional_telemetry_matches_artifact(validated)
                and _verify_artifact(validated.measurement_artifact)
                and validated.scorer_fingerprint == canonical_scorer
                and receipt.backend_receipt == expected_backend
            )
        except (AttributeError, TypeError, ValueError):
            return False

    invalid = tuple(
        lever for lever in required_levers
        if lever in by_lever
        and not all(_telemetry_context_valid(receipt) for receipt in by_lever[lever])
    )
    timing_valid = _timing_telemetry_is_valid(
        timing_receipt,
        shared_config_identity=shared_config_identity if shared_identity_ok else None,
    )
    if timing_receipt is not None and not timing_valid:
        invalid = tuple(dict.fromkeys((*invalid, *(lever for lever in required_levers
                                                  if lever in by_lever))))
    reported_within_policy = (
        not missing and not failed and not invalid and not unchecked_receipt_present)
    if timing_valid and timing_receipt is not None:
        speedup = float(timing_receipt.receipt.reported_end_to_end_speedup)
    elif reported_end_to_end_speedup is not None:
        try:
            speedup = float(reported_end_to_end_speedup)
        except (TypeError, ValueError):
            speedup = 0.0
        if not math.isfinite(speedup):
            speedup = 0.0
    else:
        speedup = 0.0
    # Disk-loaded rows cannot attest actual execution. They may be internally consistent and
    # within policy, but authoritative functional parity and runtime admission both remain REFUSE.
    authoritative_parity = False
    admitted = False
    return TrainingAdmissionReceipt(
        required_levers=tuple(required_levers), covered_levers=tuple(sorted(by_lever)),
        missing_levers=missing, failed_levers=failed, invalid_context_levers=invalid,
        reported_end_to_end_speedup=float(speedup),
        minimum_speedup=CANONICAL_MINIMUM_SPEEDUP, required_config_id=CANONICAL_CONFIG_ID,
        required_device=CANONICAL_FUNCTIONAL_DEVICE,
        required_scorer_surface=CANONICAL_SCORER_SURFACE,
        functional_parity_passed=authoritative_parity,
        training_throughput_admitted=admitted,
        reported_functional_metrics_within_tolerance=reported_within_policy,
        functional_telemetry_valid=bool(by_lever) and not invalid and not unchecked_receipt_present,
        timing_telemetry_valid=timing_valid,
    )


THETA_RELEVANT_VJP_INPUTS = {
    "chroma": ("frame_rgb",),
    "phase": ("signed_margin",),
    "temporal": ("g1", "g0_warped"),
    "area": ("logits",),
}


def _canonical_v9_synthetic_probe_context(
    *, K: int, height: int, width: int, seed: int
) -> dict[str, Any]:
    """Build deterministic synthetic arrays under the canonical typed V9 lever context.

    The spatial shape, value domains, active classes, and scalar weights come from production
    semantics/the typed DSL. Pixels remain synthetic: this helper does not run the frozen scorer,
    build real GT providers, or exercise the temporal warp.
    """
    import numpy as np

    from tac.witness_dsl.curriculum_dsl import MicroBatch
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config

    compiled = compile_v9_cgauge_432_launch_config()
    program = compiled.typed.to_program().with_lever(MicroBatch(int(K)))
    violations = program.validate()
    if violations:
        raise ValueError(f"canonical V9 synthetic probe config invalid: {violations[:4]}")
    flags = program.flag_dict()
    rng = np.random.default_rng(int(seed))

    frame_rgb = rng.uniform(0.0, 255.0, (K, height, width, 3)).astype(np.float32)
    gt_rgb = rng.uniform(0.0, 255.0, (K, height, width, 3)).astype(np.float32)
    gt_luma = (
        0.299 * gt_rgb[..., 0:1]
        + 0.587 * gt_rgb[..., 1:2]
        + 0.114 * gt_rgb[..., 2:3]
    )
    target_chroma = (gt_rgb - gt_luma).astype(np.float32)
    signed = rng.standard_normal((K, height, width)).astype(np.float32)
    direction = rng.integers(0, 2, (K, height, width)).astype(np.float32)
    phase_reference = rng.random((K, height, width)).astype(np.float32)
    def ground_probabilities() -> Any:
        logits = rng.standard_normal((K, height, width, 5)).astype(np.float32)
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return (exp / np.sum(exp, axis=-1, keepdims=True))[..., 0:3].astype(np.float32)

    g1_probability = ground_probabilities()
    g0_warped = ground_probabilities()

    def binary_weight() -> Any:
        value = (rng.random((K, height, width)) < 0.25).astype(np.float32)
        value.reshape(-1)[0] = 1.0
        return value

    temporal_classes = {
        int(item) for item in str(flags["--seg-temporal-screw-classes"]).split(",")
        if item.strip()
    }
    area_classes = tuple(
        int(item) for item in str(flags["--area-constraint-classes"]).split(",")
        if item.strip()
    )
    return {
        "program": program,
        "frame_rgb": frame_rgb,
        "target_chroma": target_chroma,
        "signed": signed,
        "direction": direction,
        "phase_reference": phase_reference,
        "g1_probability": g1_probability,
        "g0_warped": g0_warped,
        "chroma_weight_map": binary_weight(),
        "phase_weight_map": binary_weight(),
        "temporal_weight_map": binary_weight(),
        "temporal_class_mask": np.asarray(
            [1.0 if cls in temporal_classes else 0.0 for cls in (0, 1, 2)],
            dtype=np.float32,
        ),
        "area_classes": area_classes,
        "lever_weights": {
            "chroma": float(flags["--seg-chroma-boundary-weight"]),
            "phase": float(flags["--seg-phase-advect-weight"]),
            "temporal": float(flags["--seg-temporal-screw-weight"]),
        },
    }


def measure_v9_synthetic_map_parity(
    *,
    K: int = 2,
    seed: int = 1975,
) -> dict[str, Any]:
    """Measure four routed map primitives at the faithful 384x512 spatial surface.

    Chroma, phase, and temporal compare the fused Metal forward plus every theta-relevant VJP
    against their pure-MLX references. In particular temporal checks both ``g1`` and
    ``g0_warped``. Area compares one vectorized ``B=K`` expression against the independent mean
    of ``K`` per-pair expressions. Inputs are deterministic and use canonical V9 weights/classes
    and production numeric domains, but remain synthetic.

    This is an in-process primitive diagnostic, not canonical full-V9 functional parity and not
    an admission token. It raises ``RuntimeError`` unless a
    real Metal dispatch occurs for every fused lever, and the returned dictionary explicitly
    refuses training authority. Persisting the dictionary cannot turn it into execution custody.
    No frozen scorer or temporal warp executes here; the combined full-V9/scorer/warp step remains
    a separate owed runtime gate.
    """
    if type(K) is not int or K < 2:
        raise ValueError(f"V9 synthetic map parity requires integer K >= 2, got {K!r}")

    import mlx.core as mx
    import numpy as np

    from tac.boundary_math.levelset_micro_batch_loss import (
        LeverConfig,
        _batched_v9_map_terms,
        _single_area_constraint,
    )
    from tac.local_acceleration import metal_micro_batch_v9_levers as kernels
    from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

    H, W = 384, 512
    context = _canonical_v9_synthetic_probe_context(
        K=K, height=H, width=W, seed=int(seed))
    rng = np.random.default_rng(int(seed) + 1)

    def _metric_receipt(lever: str, fused_value: Any, fused_grad: Any,
                        reference_value: Any, reference_grad: Any,
                        backend_receipt: str) -> FunctionalParityReceipt:
        fused_v = float(np.asarray(fused_value).reshape(()))
        reference_v = float(np.asarray(reference_value).reshape(()))
        fused_g = np.asarray(fused_grad, dtype=np.float64)
        reference_g = np.asarray(reference_grad, dtype=np.float64)
        delta = fused_g - reference_g
        grad_maxabs = float(np.max(np.abs(delta))) if delta.size else 0.0
        grad_rel_l2 = float(
            np.linalg.norm(delta.reshape(-1))
            / (np.linalg.norm(reference_g.reshape(-1)) + 1e-12)
        )
        return make_functional_parity_receipt(
            lever=lever,
            K=K,
            batched_loss=fused_v,
            serial_mean_loss=reference_v,
            grad_rel_l2=grad_rel_l2,
            grad_maxabs=grad_maxabs,
            backend_receipt=backend_receipt,
            config_id=CANONICAL_CONFIG_ID,
            device=CANONICAL_FUNCTIONAL_DEVICE,
            scorer_surface="synthetic_v9_map_primitive_no_scorer_no_warp",
            faithful_scale=True,
        )

    with temporary_mlx_device(CANONICAL_FUNCTIONAL_DEVICE):
        if not kernels.metal_micro_batch_v9_available():
            raise RuntimeError(
                "V9 synthetic map parity REFUSE: initialized Metal device unavailable"
            )

        rgb = mx.array(context["frame_rgb"])
        target = mx.array(context["target_chroma"])
        signed = mx.array(context["signed"])
        direction = mx.array(context["direction"])
        reference = mx.array(context["phase_reference"])
        g1_probability = mx.array(context["g1_probability"])
        g0 = mx.array(context["g0_warped"])
        class_mask = mx.array(context["temporal_class_mask"])
        chroma_annulus = mx.array(context["chroma_weight_map"])
        phase_weight = mx.array(context["phase_weight_map"])
        temporal_annulus = mx.array(context["temporal_weight_map"])
        cases: tuple[
            tuple[
                str,
                Callable[..., Any],
                Callable[..., Any],
                tuple[Any, ...],
                Any,
                float,
            ],
            ...,
        ] = (
            (
                "chroma",
                kernels.chroma_squared_map,
                kernels.chroma_squared_map_reference,
                (rgb, target),
                chroma_annulus, float(context["lever_weights"]["chroma"]),
            ),
            (
                "phase",
                kernels.phase_squared_map,
                kernels.phase_squared_map_reference,
                (signed, direction, reference),
                phase_weight, float(context["lever_weights"]["phase"]),
            ),
            (
                "temporal",
                kernels.temporal_squared_map,
                kernels.temporal_squared_map_reference,
                (g1_probability, g0, class_mask),
                temporal_annulus, float(context["lever_weights"]["temporal"]),
            ),
        )
        receipts: list[FunctionalParityReceipt] = []
        backend_details: dict[str, Any] = {}
        for lever, fused, pure_reference, args, pixel_weight, lever_weight in cases:
            fused_scalar: Callable[[Any], Any]
            reference_scalar: Callable[[Any], Any]
            if lever == "temporal":
                # Both tensors are theta-bearing in production. Packing them into one differentiated
                # argument ensures the diagnostic cannot accidentally certify only the g1 transpose.
                live_pair = mx.stack([args[0], args[1]], axis=0)

                def _fused_temporal(
                    live: Any,
                    _fused: Callable[..., Any] = fused,
                    _mask: Any = args[-1],
                    _weight: Any = pixel_weight,
                    _lever_weight: float = lever_weight,
                ) -> Any:
                    squared = _fused(live[0], live[1], _mask, use_metal=True)
                    numer = mx.sum(squared * _weight, axis=(1, 2))
                    denom = mx.sum(_weight, axis=(1, 2)) + 1e-6
                    return _lever_weight * mx.mean(numer / denom)

                def _reference_temporal(
                    live: Any,
                    _reference: Callable[..., Any] = pure_reference,
                    _mask: Any = args[-1],
                    _weight: Any = pixel_weight,
                    _lever_weight: float = lever_weight,
                ) -> Any:
                    rows = []
                    for pair_index in range(K):
                        squared = _reference(
                            live[0, pair_index:pair_index + 1],
                            live[1, pair_index:pair_index + 1],
                            _mask,
                        )
                        weight = _weight[pair_index:pair_index + 1]
                        rows.append(mx.sum(squared * weight) / (mx.sum(weight) + 1e-6))
                    return _lever_weight * mx.mean(mx.stack(rows))

                differentiated = live_pair
                fused_scalar = _fused_temporal
                reference_scalar = _reference_temporal
            else:
                def _fused_non_temporal(
                    first: Any,
                    _fused: Callable[..., Any] = fused,
                    _args: tuple[Any, ...] = args,
                    _weight: Any = pixel_weight,
                    _lever_weight: float = lever_weight,
                ) -> Any:
                    squared = _fused(first, *_args[1:], use_metal=True)
                    numer = mx.sum(squared * _weight, axis=(1, 2))
                    denom = mx.sum(_weight, axis=(1, 2)) + 1e-6
                    return _lever_weight * mx.mean(numer / denom)

                def _reference_non_temporal(
                    first: Any,
                    _reference: Callable[..., Any] = pure_reference,
                    _args: tuple[Any, ...] = args,
                    _weight: Any = pixel_weight,
                    _lever_weight: float = lever_weight,
                ) -> Any:
                    rows = []
                    for pair_index in range(K):
                        per_pair_args = tuple(
                            arg if tuple(getattr(arg, "shape", ())) == (3,)
                            else arg[pair_index:pair_index + 1]
                            for arg in _args[1:]
                        )
                        squared = _reference(
                            first[pair_index:pair_index + 1], *per_pair_args)
                        weight = _weight[pair_index:pair_index + 1]
                        rows.append(mx.sum(squared * weight) / (mx.sum(weight) + 1e-6))
                    return _lever_weight * mx.mean(mx.stack(rows))

                differentiated = args[0]
                fused_scalar = _fused_non_temporal
                reference_scalar = _reference_non_temporal

            fused_value, fused_grad = mx.value_and_grad(
                lambda value, _scalar=fused_scalar: _scalar(value)
            )(differentiated)
            reference_value, reference_grad = mx.value_and_grad(
                lambda value, _scalar=reference_scalar: _scalar(value)
            )(differentiated)
            # The Metal helper owns truthful lazy-execution custody. It evaluates the exact pending
            # forward/VJP arrays before upgrading metal_planned -> metal.
            backend_details[lever] = kernels.verify_v9_lever_backend_execution(lever)
            mx.eval(fused_value, fused_grad, reference_value, reference_grad)
            backend = kernels.v9_lever_backend_receipt().get(lever, "")
            if backend != "metal":
                raise RuntimeError(
                    f"V9 synthetic map parity REFUSE: {lever} backend was {backend!r}"
                )
            receipts.append(_metric_receipt(
                lever, fused_value, fused_grad, reference_value, reference_grad, backend))

        logits = mx.array(rng.standard_normal((K, H, W, 5)).astype(np.float32))
        labels = rng.integers(0, 5, (K, H, W))
        one_hot = mx.array(np.eye(5, dtype=np.float32)[labels])
        # Lambda magnitudes are synthetic because production derives them live from GT global areas;
        # the active classes themselves are the canonical typed V9 {1,3} surface.
        area_config = LeverConfig(area_lambda={
            int(cls): (2.0 if index == 0 else 0.7)
            for index, cls in enumerate(context["area_classes"])
        })
        dummy_rgb = mx.zeros((K, H, W, 3), dtype=mx.float32)

        def area_batched(z):
            return _batched_v9_map_terms(
                None,
                z,
                z,
                dummy_rgb,
                None,
                one_hot,
                list(range(K)),
                area_config,
                include_area=True,
            )

        def area_per_pair(z):
            rows = [
                _single_area_constraint(
                    z[pair_index:pair_index + 1],
                    one_hot[pair_index:pair_index + 1],
                    area_config,
                )
                for pair_index in range(K)
            ]
            return mx.mean(mx.stack(rows))

        batched_value, batched_grad = mx.value_and_grad(area_batched)(logits)
        per_pair_value, per_pair_grad = mx.value_and_grad(area_per_pair)(logits)
        mx.eval(batched_value, batched_grad, per_pair_value, per_pair_grad)
        receipts.append(_metric_receipt(
            "area", batched_value, batched_grad, per_pair_value, per_pair_grad,
            "mlx_vectorized"))

    return {
        "schema": "micro_batch_v9_synthetic_map_measurement.v1",
        "K": K,
        "height": H,
        "width": W,
        "seed": int(seed),
        "config_identity": canonical_compiled_config_identity(K).as_dict(),
        "in_process_synthetic_map_receipts": [receipt.as_dict() for receipt in receipts],
        "reported_map_metrics_within_tolerance": all(receipt.passed for receipt in receipts),
        "authoritative_functional_parity_established": False,
        "measured_levers": tuple(receipt.lever for receipt in receipts),
        "theta_relevant_vjps_checked": THETA_RELEVANT_VJP_INPUTS,
        "backend_execution_details": backend_details,
        "scope": "synthetic_per_lever_map_primitives_at_faithful_spatial_scale",
        "synthetic_inputs": True,
        "frozen_scorer_executed": False,
        "temporal_warp_executed": False,
        "canonical_full_v9_parity_established": False,
        "full_v9_runtime_validation_owed": True,
        "training_throughput_admitted": False,
        "timing_authority": "none; persisted JSON cannot attest execution",
        "no_score_authority": True,
    }


@dataclass(frozen=True)
class BitIdentityVerdict:
    """Honest classification of whether B>1 can be bit-identical + at what speedup."""

    device: str
    scorer_fwd_seg_maxabs: float
    scorer_fwd_argmax_flips: int
    scorer_fwd_pose_maxabs: float
    reduction_order_grad_maxabs: float
    scorer_fwd_speedup: float
    # derived
    scorer_forward_is_batch_invariant: bool
    argmax_is_batch_invariant: bool
    bit_identical_at_speedup_possible: bool
    surviving_speedup_at_bit_identity: float
    dominant_source: str            # "scorer_forward" | "reduction_order" | "none"
    admission_path: str
    reported_functional_parity_supplied: bool
    training_throughput_admitted: bool
    authoritative_functional_parity_established: bool = False
    training_only: bool = True
    score_authority: bool = False
    no_score_authority: bool = True
    operator_override: str = "max_throughput_over_bit_identity_operator_override_20260712"

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "scorer_fwd_seg_maxabs": self.scorer_fwd_seg_maxabs,
            "scorer_fwd_argmax_flips": self.scorer_fwd_argmax_flips,
            "scorer_fwd_pose_maxabs": self.scorer_fwd_pose_maxabs,
            "reduction_order_grad_maxabs": self.reduction_order_grad_maxabs,
            "scorer_fwd_speedup": self.scorer_fwd_speedup,
            "scorer_forward_is_batch_invariant": self.scorer_forward_is_batch_invariant,
            "argmax_is_batch_invariant": self.argmax_is_batch_invariant,
            "bit_identical_at_speedup_possible": self.bit_identical_at_speedup_possible,
            "surviving_speedup_at_bit_identity": self.surviving_speedup_at_bit_identity,
            "dominant_source": self.dominant_source,
            "admission_path": self.admission_path,
            "reported_functional_parity_supplied": self.reported_functional_parity_supplied,
            "authoritative_functional_parity_established": (
                self.authoritative_functional_parity_established),
            "training_throughput_admitted": self.training_throughput_admitted,
            "training_only": self.training_only,
            "score_authority": self.score_authority,
            "no_score_authority": self.no_score_authority,
            "operator_override": self.operator_override,
            "pointer": ("reports/latest.md canonical pointer UNMOVED (means; only a byte-closed "
                        "n600 evaluate.py row moves it)"),
        }


# Tolerance below which a scorer FORWARD delta is treated as bit-invariant (fp32
# machine-eps scale). The real scorer clears this on NEITHER GPU (2.3e-2) nor CPU
# (7e-5) — both exceed it — so batching the scorer forward is never bit-invariant.
_SCORER_FWD_BIT_INVARIANT_TOL = 1e-6


def classify_micro_batch_bit_identity(
    *,
    device: str,
    scorer_fwd_seg_maxabs: float,
    scorer_fwd_argmax_flips: int,
    scorer_fwd_pose_maxabs: float,
    reduction_order_grad_maxabs: float,
    scorer_fwd_speedup: float,
    reported_functional_parity_supplied: bool = False,
    scorer_fwd_bit_invariant_tol: float = _SCORER_FWD_BIT_INVARIANT_TOL,
) -> BitIdentityVerdict:
    """Classify the micro-batch bit-identity situation from MEASURED inputs.

    The load-bearing logic (NO hidden assumptions): a batched-scorer construction can be
    bit-identical to serial ONLY IF the scorer forward is batch-invariant to fp32 eps.
    The entire measured speedup is the batched scorer forward, so if that forward is not
    bit-invariant, the surviving speedup at bit-identity collapses to 1.0x (the per-pair
    forward == serial). The reduction-order grad drift is a SECONDARY source that only
    matters where the scorer is invariant.
    """
    scorer_fwd_invariant = (
        max(float(scorer_fwd_seg_maxabs), float(scorer_fwd_pose_maxabs))
        <= float(scorer_fwd_bit_invariant_tol)
    )
    argmax_invariant = int(scorer_fwd_argmax_flips) == 0

    if scorer_fwd_invariant:
        # Where the scorer IS bit-invariant, the only residual is the reduction order,
        # which is controllable (fixed-order left-fold) -> bit-identity achievable AND
        # the batched-forward speedup survives.
        bit_possible = True
        surviving = float(scorer_fwd_speedup)
        dominant = "reduction_order" if reduction_order_grad_maxabs > 0.0 else "none"
        admission = "bit-identical at the batched speedup; diagnostic clears without A/B"
    else:
        # The real scorer case: forward is NOT batch-invariant -> bit-identity requires a
        # per-pair (batch-1) forward == serial == 1.0x. The speedup is inseparable from the
        # non-bit-invariant batched forward.
        bit_possible = False
        surviving = 1.0
        dominant = "scorer_forward"
        admission = (
            "bit-identity impossible at speedup>1x; bounded A/B remains useful drift telemetry "
            "but is not a training gate under the 2026-07-12 operator override"
        )

    # This classifier sees only the isolated scorer-forward microbenchmark. That number is useful
    # diagnosis but is NOT the full training-step wall-clock receipt, so it must never admit the
    # training path on its own. ``classify_training_admission`` validates persisted rows as
    # telemetry but deliberately cannot infer execution authority from JSON.
    training_admitted = False
    admission += (
        "; DIAGNOSTIC ONLY: caller reported functional parity, but that telemetry and persisted "
        "functional/timing "
        "receipts cannot attest runtime execution; classify_training_admission therefore "
        "remains REFUSE; no score authority"
        if reported_functional_parity_supplied else
        "; TRAINING REFUSE: functional-parity runtime evidence is absent and persisted timing "
        "cannot attest execution; no score authority"
    )

    return BitIdentityVerdict(
        device=str(device),
        scorer_fwd_seg_maxabs=float(scorer_fwd_seg_maxabs),
        scorer_fwd_argmax_flips=int(scorer_fwd_argmax_flips),
        scorer_fwd_pose_maxabs=float(scorer_fwd_pose_maxabs),
        reduction_order_grad_maxabs=float(reduction_order_grad_maxabs),
        scorer_fwd_speedup=float(scorer_fwd_speedup),
        scorer_forward_is_batch_invariant=bool(scorer_fwd_invariant),
        argmax_is_batch_invariant=bool(argmax_invariant),
        bit_identical_at_speedup_possible=bool(bit_possible),
        surviving_speedup_at_bit_identity=float(surviving),
        dominant_source=dominant,
        admission_path=admission,
        reported_functional_parity_supplied=bool(reported_functional_parity_supplied),
        training_throughput_admitted=training_admitted,
        authoritative_functional_parity_established=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reduction-order isolation (source B) — self-contained tiny witness + a batch-INVARIANT
# mock scorer (so source A == 0 by construction). Imports MLX lazily so the module is
# importable/inspectable without MLX.
# ─────────────────────────────────────────────────────────────────────────────
def _build_tiny_env(K: int, seed: int = 0):
    """A tiny witness + a linear (batch-INVARIANT) mock SegNet/PoseNet + K random pairs.
    Returns the arg bundle the twin's ``batched_realized_loss`` / ``single_realized_loss``
    consume. The mock scorer is linear per-pixel/per-frame so ``segnet(batch)[k]`` is
    bit-identical to ``segnet(batch[k:k+1])[0]`` -> isolates the reduction order."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    class _TinyWitness(nn.Module):
        def __init__(self, feat_dim, mod_dim, hidden, n_frames, n_classes=5, s=0):
            super().__init__()
            self.n_hidden = 1
            self.hidden_dim = hidden
            self.in_proj = nn.Linear(feat_dim, hidden)
            self.film = nn.Linear(mod_dim, self.n_hidden * 2 * hidden)
            self.hidden = [nn.Linear(hidden, hidden)]
            self.out_sdf = nn.Linear(hidden, n_classes)
            self.out_tex = nn.Linear(hidden, 3)
            rng = np.random.default_rng(s)
            self.code = mx.array(rng.standard_normal((n_frames, mod_dim)).astype(np.float32) * 0.5)

        def _trunk(self, cf, code_idx):
            h = nn.relu(self.in_proj(cf))
            film = mx.reshape(self.film(self.code[code_idx]), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                h = nn.relu(layer(h) * (1.0 + film[li, 0]) + film[li, 1])
            return h

        def sdf(self, cf, code_idx):
            return self.out_sdf(self._trunk(cf, code_idx))

        def __call__(self, cf, code_idx):
            return mx.sigmoid(self.out_tex(self._trunk(cf, code_idx))) * 255.0

    class _MockAdapter:
        def __init__(self, s=1):
            rng = np.random.default_rng(s)
            self.seg_w = mx.array(rng.standard_normal((3, 5)).astype(np.float32))
            self.seg_b = mx.array(rng.standard_normal((5,)).astype(np.float32))
            self.pose_w = mx.array(rng.standard_normal((12, 6)).astype(np.float32) * 0.01)

        def segnet(self, f):
            return f @ self.seg_w + self.seg_b

        def posenet(self, yuv):
            return {"pose": mx.mean(yuv, axis=(1, 2)) @ self.pose_w}

    rh, rw = 8, 12
    n_px = rh * rw
    feat_dim, mod_dim, hidden = 7, 4, 6
    rng = np.random.default_rng(seed + 7)
    model = _TinyWitness(feat_dim, mod_dim, hidden, 2 * K, s=seed)
    adapter = _MockAdapter(s=seed + 1)
    cf = mx.array(rng.standard_normal((n_px, feat_dim)).astype(np.float32))
    cf_list = [cf for _ in range(K)]
    c0_list = [2 * p + 0 for p in range(K)]
    c1_list = [2 * p + 1 for p in range(K)]
    oh_list, mg_list, pt_list = [], [], []
    for _ in range(K):
        arg = rng.integers(0, 5, size=(rh, rw))
        oh = np.eye(5, dtype=np.float32)[arg].reshape(1, rh, rw, 5)
        mg = rng.random((1, rh, rw)).astype(np.float32) * 0.5
        oh_list.append(mx.array(oh))
        mg_list.append(mx.array(mg))
        pt_list.append(mx.array(rng.standard_normal(6).astype(np.float32) * 0.01))
    mx.eval(model.parameters(), cf, *oh_list, *mg_list, *pt_list)
    return {
        "model": model,
        "adapter": adapter,
        "rh": rh,
        "rw": rw,
        "cf_list": cf_list,
        "c0_list": c0_list,
        "c1_list": c1_list,
        "oh_list": oh_list,
        "mg_list": mg_list,
        "pt_list": pt_list,
    }


def _zero_eikonal_length(phi_pk, rh, rw, **kw):
    """Trivial batch-invariant eikonal/length stub (returns constant zeros). Used only with
    eik_w == len_w == 0 so it contributes nothing to the loss/grad; keeps the probe
    self-contained (no trainer/scipy import)."""
    import mlx.core as mx

    z = mx.zeros(())
    return z, z, None


def _zero_nuclear(code, **kw):
    import mlx.core as mx

    return mx.zeros(())


def _render_fn(model, cf, code_idx, rh, rw):
    import mlx.core as mx

    return mx.reshape(model(cf, int(code_idx)), (1, rh, rw, 3))


def measure_reduction_order_drift(K: int = 4, seg_form: str = "ce", seed: int | None = None,
                                  *, w_seg: float = 100.0, w_pose: float = 1.0,
                                  hinge: float = 4.0, mtgt: float = 0.5) -> ReductionOrderDrift:
    """Measure the PURE reduction/accumulation-order drift (source B): the batched twin's
    grad vs the serial explicit left-fold mean-of-per-pair grad, using a batch-INVARIANT
    mock scorer so the scorer-forward kernel contributes exactly 0.0.

    Run on MLX CPU (deterministic) — the caller sets the device. Returns the max absolute
    per-leaf grad difference (the trajectory-relevant metric) + the global-L2 rel err +
    the loss-scalar difference.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten, tree_map

    from tac.boundary_math.levelset_micro_batch_loss import (
        LeverConfig,
        batched_realized_loss,
        single_realized_loss,
    )

    env = _build_tiny_env(K, seed=(K if seed is None else seed))
    model = env["model"]
    lc = LeverConfig(seg_loss_default=seg_form, tau_use=0.3, l7_thr_use=0.42, l7_mult=4.0,
                     score_domain=True, pose_eps=1e-2,
                     eikonal_length=_zero_eikonal_length, nuclear_norm_smooth=_zero_nuclear)

    def _bfn(m):
        return batched_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"], env["c0_list"], env["c1_list"],
            env["oh_list"], env["mg_list"], env["pt_list"],
            w_seg, w_pose, hinge, mtgt, seg_form, 0.0, 0.0, lc)

    lb, gb = nn.value_and_grad(model, _bfn)(model)
    mx.eval(lb, gb)

    def _sfn(m, k):
        return single_realized_loss(
            m, env["adapter"], _render_fn, env["rh"], env["rw"],
            env["cf_list"][k], env["c0_list"][k], env["c1_list"][k],
            env["oh_list"][k], env["mg_list"][k], env["pt_list"][k],
            w_seg, w_pose, hinge, mtgt, seg_form, 0.0, 0.0, lc)

    accum = None
    lsum = 0.0
    for k in range(K):
        ls, gs = nn.value_and_grad(model, _sfn)(model, k)
        mx.eval(ls, gs)
        lsum += float(ls)
        accum = gs if accum is None else tree_map(lambda a, b: a + b, accum, gs)
        mx.eval(accum)
    mean_grad = tree_map(lambda g, c=float(K): g / c, accum)
    mx.eval(mean_grad)

    fb = dict(tree_flatten(gb))
    fm = dict(tree_flatten(mean_grad))
    grad_maxabs = 0.0
    worst = ""
    for key in fb:
        d = float(np.max(np.abs(np.asarray(fb[key], np.float64) - np.asarray(fm[key], np.float64))))
        if d >= grad_maxabs:
            grad_maxabs = d
            worst = key
    diff = np.concatenate([(np.asarray(fb[k], np.float64) - np.asarray(fm[k], np.float64)).ravel()
                           for k in fb])
    ref = np.concatenate([np.asarray(fm[k], np.float64).ravel() for k in fm])
    grad_rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12))
    loss_abs = abs(float(lb) - lsum / K)
    return ReductionOrderDrift(K=int(K), seg_form=str(seg_form), grad_maxabs=grad_maxabs,
                               grad_rel_l2=grad_rel_l2, loss_abs=loss_abs, worst_leaf=worst)
