"""Apple Neural Engine (ANE) SCREENING backend for the frozen contest scorers.

The ANE is fp16-only.  Under the same law CLAUDE.md applies to MPS -- *a
gradient device, never a score* -- the ANE is a **screening** device: it may
RANK or PRESCREEN candidates, and it may never emit a number that leaves an
instrument.  Every pair a screening backend ADOPTS is re-measured on
``cpu_torch`` fp32 before any value is reported.  That invariant is enforced
here (:func:`assert_cpu_confirm_contract`), not left to the caller's memory.

Why the contract has teeth, MEASURED (ddm_ane1, 2026-09-05, and the 2026-07-13
parent lane ``ane_unlock_correction_20260713``):

* SegNet (smp Unet / EfficientNet-B2) argmax is decided by sub-1e-3 logit
  margins.  CoreML fp16 flips ~2.5% of pixels against 1-thread CPU-torch fp32 --
  about 750x above the 3.3e-5 authority bar.  **No fp16 backend may ever read
  d_seg.**
* PoseNet (FastViT-T12) emits a 6-dim regression whose contest term is
  ``sqrt(10 * MSE)``.  Its fp16 perturbation is a *relative* one on a smooth
  output, so the axis that reads it tolerates far more drift than argmax does.

Both facts are the same law seen on two architecture classes: **precision drift
is read by the AXIS, not by the model** (sister of
``mps_drift_architecture_class_dependent_v1``).

Nothing in this module imports ``coremltools`` at module scope: the repo's main
virtualenv does not carry it, and importing ``tac.ane_screening`` must never
break a CPU-torch caller.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AUTHORITY_BACKEND",
    "BACKEND_AXIS_VERDICTS",
    "SCORER_BACKENDS",
    "SCREENING_BACKENDS",
    "SEG_AUTHORITY_FLIP_BAR",
    "AneScreeningError",
    "CoreMLPoseBackend",
    "assert_backend_admissible_for_axis",
    "assert_backend_name",
    "assert_cpu_confirm_contract",
    "backend_axis_verdict",
    "backend_is_authority",
    "load_pose_backend",
    "mlpackage_provenance",
    "screening_receipt",
    "sha256_tree",
]


class AneScreeningError(RuntimeError):
    """Raised when a screening backend is used outside its contract."""


#: Every backend the pose-axis instrument family accepts.
SCORER_BACKENDS: tuple[str, ...] = ("cpu_torch", "coreml_cpu_fp32", "ane_fp16_screen")

#: The ONLY backend whose numbers may leave an instrument.
AUTHORITY_BACKEND = "cpu_torch"

#: Backends that may rank/screen and must be confirmed on :data:`AUTHORITY_BACKEND`.
SCREENING_BACKENDS: tuple[str, ...] = ("coreml_cpu_fp32", "ane_fp16_screen")

#: Operator-set SegNet argmax flip-rate bar for a label-grade backend
#: (2026-07-13 lane preregistration: <= 0.0033%).
SEG_AUTHORITY_FLIP_BAR = 3.3e-5


#: MEASURED per-backend verdicts, so no caller rediscovers them and none picks a
#: backend that has already been measured unfit for the axis it is being asked
#: to read.  "off" is a tracked state with a reason, never a silent default.
#:
#: Every row is a MEASUREMENT with its artifact, not a policy opinion.  ``ok``
#: means the backend was measured fit FOR THAT AXIS; it never means authority --
#: ``cpu_torch`` remains the only backend whose numbers may leave an instrument.
BACKEND_AXIS_VERDICTS: dict[tuple[str, str], dict[str, Any]] = {
    ("coreml_cpu_fp32", "pose_rank"): {
        "ok": True,
        "measured": "argmin agreement 38/39 = 97.44%, Kendall tau-b median 1.00",
        "adoption_gain": "+1.2080e-04 -- reproduces pr1's CPU sweep gain exactly",
        "speedup_end_to_end": 1.94,
        "arm": "ddm_ane2",
        "artifact": (
            "/Volumes/VertigoDataTier/pact/ddm_ane2_precision/replay/"
            "ane2_selector_replay_coreml_cpu_fp32_validate39.json"
        ),
    },
    ("ane_fp16_screen", "pose_rank"): {
        "ok": False,
        "measured": "argmin agreement 4/39 = 10.26% against a 12.5% chance rate; "
        "Kendall tau-b median 0.0714",
        "adoption_gain": "-4.728e-02 -- the wrong direction by 391x",
        "arm": "ddm_ane1",
        "artifact": (
            "/Volumes/VertigoDataTier/pact/ddm_ane1_ane_screening/replay/"
            "ane1_selector_replay_validate39.json"
        ),
    },
    ("ane_fp16_screen", "pose_value"): {
        "ok": False,
        "measured": "self-MSE 1.6630e-03 = 214x the exact d_pose (7.77e-06); "
        "6.32x of that is the ANE's own arithmetic, not fp16 -- the identical "
        "package under CPU_ONLY measures 2.6334e-04",
        "arm": "ddm_ane2",
        "artifact": (
            "/Volumes/VertigoDataTier/pact/ddm_ane2_precision/stage2/"
            "units_posenet_fp16.json"
        ),
    },
    ("ane_fp16_screen", "seg_argmax"): {
        "ok": False,
        "measured": "flip rate 4.4039e-05 = 1.33x the 3.3e-05 bar on a generated "
        "decode; 9.7097e-04 = 29.4x the bar on GT frames -- the INPUT moves this "
        "22.0x, more than any precision split does",
        "arm": "ddm_ane2",
        "artifact": (
            "/Volumes/VertigoDataTier/pact/ddm_ane2_precision/stage3/"
            "ladder_segnet_generated_n120.json"
        ),
    },
}


def backend_axis_verdict(name: str, axis: str) -> dict[str, Any] | None:
    """The MEASURED verdict for ``(backend, axis)``, or ``None`` if unmeasured.

    ``None`` is not permission: it means no arm has measured this pairing, which
    is a duty to measure, not a licence to assume.
    """
    return BACKEND_AXIS_VERDICTS.get((assert_backend_name(name), axis))


def assert_backend_admissible_for_axis(name: str, axis: str) -> str:
    """Refuse a backend an arm has already MEASURED unfit for this axis.

    Raises with the measured number and its artifact, so the refusal teaches
    rather than merely blocks.  An unmeasured pairing is allowed through --
    this guard extincts rediscovery of known negatives, it does not pretend to
    know verdicts nobody has taken.
    """
    verdict = backend_axis_verdict(name, axis)
    if verdict is not None and not verdict["ok"]:
        raise AneScreeningError(
            f"backend {name!r} was MEASURED unfit for axis {axis!r} by "
            f"{verdict['arm']}: {verdict['measured']}. Evidence: "
            f"{verdict['artifact']}"
        )
    return name


def backend_is_authority(name: str) -> bool:
    """True only for the CPU-torch fp32 backend."""
    return assert_backend_name(name) == AUTHORITY_BACKEND


def assert_backend_name(name: str) -> str:
    """Return ``name`` if it is a known backend, else fail closed."""
    if name not in SCORER_BACKENDS:
        raise AneScreeningError(
            f"unknown scorer backend {name!r}; expected one of {SCORER_BACKENDS}"
        )
    return name


def sha256_tree(path: Path) -> str:
    """Content hash of a file, or of a directory tree (an ``.mlpackage``).

    Directory hashing walks in sorted relative-path order and folds each
    entry's path and bytes, so the digest is stable across hosts and is a real
    custody statement about the package -- not about its mtimes.
    """
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise AneScreeningError(f"cannot hash missing path: {path}")
    for entry in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(entry.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with entry.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def mlpackage_provenance(path: Path) -> dict[str, Any]:
    """Custody record for one ``.mlpackage``: path, tree sha256, tool version.

    Every receipt a screening backend writes carries this, so a number can
    always be traced back to the exact converted graph that produced it.
    """
    path = Path(path)
    record: dict[str, Any] = {
        "mlpackage": str(path),
        "mlpackage_sha256": sha256_tree(path),
        "coremltools_version": None,
    }
    try:  # pragma: no cover - depends on the caller's virtualenv
        import coremltools  # type: ignore[import-not-found]

        record["coremltools_version"] = str(coremltools.__version__)
    except Exception as exc:  # pragma: no cover - main venv has no coremltools
        record["coremltools_version_error"] = repr(exc)
    return record


@dataclass
class CoreMLPoseBackend:
    """A drop-in stand-in for the frozen torch PoseNet, backed by CoreML.

    ``experiments/ddm_up2_shipping_pose_solve.pose_from_frames`` needs exactly
    two things from its ``posenet`` argument: ``preprocess_input(pair)`` and
    ``model(prepared)["pose"]``.  The preprocess (bilinear resize + YUV6) stays
    in torch fp32 for EVERY backend, so the only thing this class substitutes is
    the FastViT-T12 trunk -- which is precisely the term being screened.

    ``mode`` is the compute-unit request that produced ``mlmodel``; it is
    recorded, never inferred, and it is NOT by itself a placement proof.
    """

    mlmodel: Any
    torch_posenet: Any
    mlpackage_path: Path
    mode: str
    input_name: str
    output_name: str
    pose_dims: int = 12
    calls: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def preprocess_input(self, pair):
        """Delegate verbatim to the frozen torch preprocess (fp32, exact)."""
        return self.torch_posenet.preprocess_input(pair)

    def __call__(self, prepared) -> dict[str, Any]:
        import numpy as np
        import torch

        array = prepared.detach().to(torch.float32).cpu().numpy()
        out = self.mlmodel.predict({self.input_name: array})
        pose = np.asarray(out[self.output_name], dtype=np.float32)
        if pose.ndim == 1:
            pose = pose[None, :]
        self.calls += 1
        return {"pose": torch.from_numpy(pose)}

    def receipt(self) -> dict[str, Any]:
        record = {
            "backend": "coreml",
            "compute_units_requested": self.mode,
            "calls": self.calls,
            "score_claim": False,
            "promotable": False,
        }
        record.update(self.provenance)
        return record


def load_pose_backend(
    name: str,
    *,
    torch_posenet: Any,
    mlpackage: Path | None = None,
    batch_size: int | None = None,
) -> Any:
    """Return an object usable wherever the frozen torch PoseNet is used.

    ``cpu_torch`` returns ``torch_posenet`` unchanged, so the authority path is
    byte-for-byte the path it has always been -- this module cannot perturb it.
    """
    assert_backend_name(name)
    if name == AUTHORITY_BACKEND:
        return torch_posenet
    if mlpackage is None:
        raise AneScreeningError(f"backend {name!r} needs --pose-mlpackage")
    import coremltools as ct  # type: ignore[import-not-found]

    mode = "CPU_ONLY" if name == "coreml_cpu_fp32" else "CPU_AND_NE"
    units = getattr(ct.ComputeUnit, mode)
    mlmodel = ct.models.MLModel(str(mlpackage), compute_units=units)
    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    backend = CoreMLPoseBackend(
        mlmodel=mlmodel,
        torch_posenet=torch_posenet,
        mlpackage_path=Path(mlpackage),
        mode=mode,
        input_name=input_name,
        output_name=output_name,
        provenance=mlpackage_provenance(Path(mlpackage)),
    )
    backend.provenance["scorer_backend"] = name
    backend.provenance["batch_size_converted_for"] = batch_size
    return backend


def assert_cpu_confirm_contract(
    *,
    backend: str,
    adopted: Sequence[int],
    confirmed: Sequence[int] | None,
) -> dict[str, Any]:
    """Fail closed unless every ADOPTED pair was re-measured on ``cpu_torch``.

    This is the whole reason a screening backend is admissible at all.  A
    screening run that adopts pairs it never confirmed is not a cheaper
    measurement, it is an unmeasured claim -- so it raises here rather than
    emitting a receipt a reader could mistake for a measurement.
    """
    assert_backend_name(backend)
    adopted_set = {int(p) for p in adopted}
    if backend == AUTHORITY_BACKEND:
        return {
            "scorer_backend": backend,
            "cpu_confirm_required": False,
            "cpu_confirm_satisfied": True,
            "adopted_pairs": len(adopted_set),
            "confirmed_pairs": len(adopted_set),
        }
    confirmed_set = {int(p) for p in (confirmed or ())}
    missing = sorted(adopted_set - confirmed_set)
    if missing:
        raise AneScreeningError(
            f"backend {backend!r} adopted {len(adopted_set)} pairs but "
            f"{len(missing)} were never re-measured on {AUTHORITY_BACKEND}: "
            f"{missing[:16]}{'...' if len(missing) > 16 else ''}"
        )
    return {
        "scorer_backend": backend,
        "cpu_confirm_required": True,
        "cpu_confirm_satisfied": True,
        "adopted_pairs": len(adopted_set),
        "confirmed_pairs": len(confirmed_set),
    }


def screening_receipt(
    *,
    backend: str,
    screen_values: dict[int, float],
    confirm_values: dict[int, float],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Both numbers, the backend name, and ``score_claim=false`` -- always.

    The screened value is retained beside the confirmed one so the reader can
    see the drift that was accepted, instead of a single number whose backend
    has been forgotten.
    """
    assert_backend_name(backend)
    adopted = sorted(screen_values)
    confirm_verdict = assert_cpu_confirm_contract(
        backend=backend, adopted=adopted, confirmed=sorted(confirm_values)
    )
    rows = []
    for pair in adopted:
        screened = float(screen_values[pair])
        confirmed = float(confirm_values[pair])
        rows.append(
            {
                "pair": int(pair),
                "screened_value": screened,
                "confirmed_value": confirmed,
                "abs_drift": abs(screened - confirmed),
                "rel_drift": (
                    abs(screened - confirmed) / abs(confirmed)
                    if confirmed != 0.0
                    else float("inf")
                ),
            }
        )
    return {
        "schema": "tac.ane_screening.receipt.v1",
        "scorer_backend": backend,
        "authority_backend": AUTHORITY_BACKEND,
        "score_claim": False,
        "promotable": False,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet confirm]",
        "cpu_confirm": confirm_verdict,
        "pairs": rows,
        "provenance": dict(provenance or {}),
    }


def rank_agreement(screened: Sequence[float], confirmed: Sequence[float]) -> dict[str, Any]:
    """Agreement of an argmin/ordering read between two backends.

    Returns the exact argmin match plus Kendall-tau-b, computed without SciPy so
    the helper is importable from the repo's main virtualenv.
    """
    a = [float(x) for x in screened]
    b = [float(x) for x in confirmed]
    if len(a) != len(b) or not a:
        raise AneScreeningError("rank_agreement needs two equal, non-empty sequences")
    concordant = discordant = tied_a = tied_b = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0.0 and db == 0.0:
                tied_a += 1
                tied_b += 1
            elif da == 0.0:
                tied_a += 1
            elif db == 0.0:
                tied_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = len(a) * (len(a) - 1) / 2
    denominator = ((n0 - tied_a) * (n0 - tied_b)) ** 0.5
    tau = (concordant - discordant) / denominator if denominator > 0 else float("nan")
    return {
        "n": len(a),
        "argmin_screened": min(range(len(a)), key=lambda i: a[i]),
        "argmin_confirmed": min(range(len(b)), key=lambda i: b[i]),
        "argmin_agrees": min(range(len(a)), key=lambda i: a[i])
        == min(range(len(b)), key=lambda i: b[i]),
        "kendall_tau_b": tau,
        "concordant": concordant,
        "discordant": discordant,
    }


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Atomically persist a receipt and return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return hashlib.sha256(blob).hexdigest()
