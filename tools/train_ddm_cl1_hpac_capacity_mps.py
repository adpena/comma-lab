#!/usr/bin/env python3
"""MPS throughput port of the identity-pinned RX2 HPAC trainer.

The reference trainer remains untouched.  This wrapper imports it by pinned
content hash, changes only the admitted device/epoch envelope, and delegates
the model, STE, loss, optimizer, EMA, evaluation, checkpoint, and result-JSON
paths to that reference implementation.

``parity-cpu`` and ``parity-mps`` are the matched six-epoch instrument pair.
With ``qat_fraction=0.5``, QAT starts at epoch four, so the run covers three
QAT epochs and emits QAT metrics at epochs four and six.  ``full-mps`` is the
60-epoch race instrument.  MPS fallback must remain disabled; completing a
parity/full run therefore verifies that every operation actually exercised by
the model, STE, backward, optimizer, and EMA path has an MPS kernel.

MPS training is seeded but is not claimed bit-reproducible run-to-run.  The
retained, SHA-256-inventoried checkpoint is its reproducibility anchor.  CPU
packing remains serialization authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TRAINER = REPO_ROOT / "tools/train_ddm_cl1_hpac_capacity.py"
REFERENCE_TRAINER_SHA256 = "8392a9b9f2d303698de59e627fa489a792ab0b0b38170cebd425f9310162059e"
PORT_MODES = {
    "parity-cpu": {"device": "cpu", "epochs": 6},
    "parity-mps": {"device": "mps", "epochs": 6},
    "full-mps": {"device": "mps", "epochs": 60},
    # Law-derived extension (wc2 memo §5/§5a boundary protocol, 2026-08-15):
    # N=480 from the fitted exp-floor descent laws on the completed 60-epoch
    # run (continuous saturates ~ep75-120; QAT tau ~40 ep wants ~240 QAT
    # epochs). Same sealed envelope in every other field.
    "full-mps-e480": {"device": "mps", "epochs": 480},
}


class MPSPortError(RuntimeError):
    """Fail-closed error for a violated HPAC MPS-port contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference() -> ModuleType:
    observed = _sha256_file(REFERENCE_TRAINER)
    if observed != REFERENCE_TRAINER_SHA256:
        raise MPSPortError(
            "reference trainer changed; port delta must be re-audited: "
            f"expected {REFERENCE_TRAINER_SHA256}, observed {observed}"
        )
    spec = importlib.util.spec_from_file_location("_rx2_hpac_reference_trainer", REFERENCE_TRAINER)
    if spec is None or spec.loader is None:
        raise MPSPortError(f"cannot import reference trainer: {REFERENCE_TRAINER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_port_args(argv: Sequence[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port-mode", choices=tuple(PORT_MODES), required=True)
    parsed, remaining = parser.parse_known_args(list(argv))
    return parsed.port_mode, remaining


def _validate_mode_args(reference: ModuleType, port_mode: str, argv: Sequence[str]) -> argparse.Namespace:
    args = reference._build_parser().parse_args(list(argv))
    expected = PORT_MODES[port_mode]
    observed = {
        "profile": args.profile,
        "device": args.device,
        "epochs": args.epochs,
        "qat_fraction": args.qat_fraction,
        "eval_every": args.eval_every,
        "stop_after_epoch": args.stop_after_epoch,
    }
    required = {
        "profile": "rx2_mc36",
        "device": expected["device"],
        "epochs": expected["epochs"],
        "qat_fraction": 0.5,
        "eval_every": 2,
        "stop_after_epoch": None,
    }
    if observed != required:
        raise MPSPortError(
            f"{port_mode} invocation violates its sealed envelope: expected {required!r}, observed {observed!r}"
        )
    if port_mode.startswith("parity-"):
        qat_start = max(1, int(args.epochs * (1.0 - args.qat_fraction)) + 1)
        if args.epochs - qat_start + 1 < 2:
            raise MPSPortError("parity run must include at least two QAT epochs")
    return args


def _configure_reference(
    reference: ModuleType,
    *,
    port_mode: str,
    raw_launch_argv: Sequence[str],
) -> None:
    mode = PORT_MODES[port_mode]
    admitted = dict(reference.RX2_PREREGISTERED_CONFIG)
    admitted["device"] = mode["device"]
    admitted["epochs"] = mode["epochs"]
    reference.RX2_PREREGISTERED_CONFIG = admitted
    reference.PREREGISTERED_CONFIG_BY_PROFILE = {
        **reference.PREREGISTERED_CONFIG_BY_PROFILE,
        "rx2_mc36": admitted,
    }

    original_run_identity = reference._run_identity
    wrapper_path = Path(__file__).resolve()
    wrapper_sha256 = _sha256_file(wrapper_path)

    def ported_run_identity(*args: Any, **kwargs: Any) -> dict[str, Any]:
        identity = original_run_identity(*args, **kwargs)
        reference_sha256 = identity["trainer_sha256"]
        identity.update(
            {
                "trainer": str(wrapper_path),
                "trainer_sha256": wrapper_sha256,
                "reference_trainer": str(REFERENCE_TRAINER.resolve()),
                "reference_trainer_sha256": reference_sha256,
                "port_mode": port_mode,
                "mps_trained": mode["device"] == "mps",
                "raw_launch_argv": list(raw_launch_argv),
                "port_delta": (
                    "admission/provenance only: device and scaled-parity epoch envelope; "
                    "reference model, STE, loss, optimizer, EMA, evaluation, checkpoints, "
                    "and result JSON are imported unchanged"
                ),
                "training_reproducibility": {
                    "seeded": True,
                    "run_to_run_bit_reproducible": mode["device"] != "mps",
                    "anchor": "retained_checkpoint_plus_manifest_sha256",
                    "serialization_authority": "cpu_pack_path",
                },
            }
        )
        source_identity = {
            "trainer_sha256": wrapper_sha256,
            "reference_trainer_sha256": reference_sha256,
            "intake_source_sha256": identity["intake_source_sha256"],
            "local_causal_source_sha256": identity["local_causal_source_sha256"],
            "port_mode": port_mode,
        }
        identity["trainer_source_identity_sha256"] = reference._canonical_json_sha256(source_identity)
        return identity

    reference._run_identity = ported_run_identity


def main(argv: Sequence[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    port_mode, reference_argv = _split_port_args(raw)
    reference = _load_reference()
    _validate_mode_args(reference, port_mode, reference_argv)
    _configure_reference(
        reference,
        port_mode=port_mode,
        raw_launch_argv=[sys.executable, str(Path(__file__).resolve()), *raw],
    )
    sys.argv = [str(Path(__file__).resolve()), *reference_argv]
    reference.main()


if __name__ == "__main__":
    main()
