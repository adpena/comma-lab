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
    # QAT power-tail continuation (wc2 §5i, 2026-08-15): the e480b ENDPOINT
    # refit flipped the QAT law to a power tail (alpha 0.14, asymptote
    # 118,147 B) — descent not floored, ~1.2 KB/doubling. Resumes the ep480
    # checkpoint for 480 more QAT epochs. REQUIRES --resume-from (fail-closed)
    # and holds LR at eta_min past the resume boundary: the restored cosine
    # scheduler carries T_max=480, so un-patched forward epochs would ride the
    # cosine past pi and RAISE the LR — a regime change; the measured tail was
    # fitted near eta_min, so eta_min-hold is the within-regime continuation.
    "full-mps-e960": {
        "device": "mps",
        "epochs": 960,
        "resume_required": True,
        "lr_floor_hold_after_epoch": 480,
        # Hold the PARENT run's EMA geometry: the parent decay was derived for
        # a 480-epoch horizon, which is exactly this continuation's FORWARD
        # window — within-regime (the tail law was measured under it) and
        # horizon-correct. Also makes the trainer's ema_policy equality and
        # EMA-decay restore checks pass natively instead of being bypassed.
        "ema_geometry_epochs": 480,
        # Epoch-extension continuation: the reference's resume-identity gate is
        # designed for same-config crash recovery; the wrapper reconciles the
        # comparison explicitly (see _install_continuation_adapter) with a
        # closed allowlist and an epochs-only training_config rule.
        "continuation_of_epochs": 480,
    },
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
    if expected.get("resume_required") and args.resume_from is None:
        raise MPSPortError(f"{port_mode} is a continuation mode: --resume-from is required")
    return args


def _install_lr_floor_hold(reference: ModuleType, boundary_epoch: int) -> None:
    """Hold LR at eta_min once the scheduler passes ``boundary_epoch``.

    The reference restores the checkpoint's CosineAnnealingLR state including
    its T_max; forward epochs past T_max would otherwise mirror the cosine and
    RAISE the LR.  The e480b tail law was measured near eta_min, so the
    continuation holds eta_min exactly (process-scoped class patch; inert for
    epochs at or below the boundary).
    """
    import torch

    scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
    if getattr(scheduler_cls, "_rx2_lr_floor_hold_after", None) == boundary_epoch:
        return
    original_get_lr = scheduler_cls.get_lr

    def held_get_lr(self):  # type: ignore[no-untyped-def]
        if self.last_epoch > boundary_epoch:
            return [self.eta_min for _ in self.optimizer.param_groups]
        return original_get_lr(self)

    scheduler_cls.get_lr = held_get_lr
    scheduler_cls._rx2_lr_floor_hold_after = boundary_epoch


_CONTINUATION_ALLOWED_DRIFT = frozenset(
    {
        # Sanctioned by the reference's own --resume-allow-trainer-drift class
        # (the wrapper file's self-hash changed since the parent run).
        "trainer_sha256",
        "trainer_source_identity_sha256",
        # Launch provenance: describes THIS invocation, never behavior.
        "raw_launch_argv",
        "port_mode",
        "port_delta",
        # The intended extension itself (epochs-only; deep-checked below) and
        # its pure derivative (schedule hash includes epochs).
        "training_config",
        "seed_schedule_identity_sha256",
        # Volatile per the reference's own comparison.
        "launch_git_sha",
        # Observer-dependent probe metadata (the reference compares only a
        # behavioral subset; the wrapper compares none of it).
        "hardware",
        "software",
    }
)


def _install_continuation_adapter(
    reference: ModuleType,
    *,
    resume_path: Path,
    continuation_of_epochs: int,
    port_mode: str,
) -> dict[str, Any]:
    """Wrap ``reference.torch.load`` to stash the resume payload for the
    identity reconciliation performed in ``ported_run_identity``.

    The reference's resume-identity gate guards same-config crash recovery;
    an epoch extension intentionally drifts a closed set of keys.  The adapter
    verifies every OTHER key strictly (cache/init/intake/source shas, config
    minus epochs, ema_policy) and fails closed on anything outside the
    allowlist, then rewrites the in-memory comparison reference to the current
    identity while appending a typed continuation record to the checkpoint's
    resume_lineage (which the trainer persists into every new save).  The
    parent file on disk is never touched.
    """
    state: dict[str, Any] = {"resume": None}
    resolved_resume = resume_path.resolve()
    original_load = reference.torch.load

    def stashing_load(path, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        payload = original_load(path, *args, **kwargs)
        try:
            matches = Path(path).resolve() == resolved_resume
        except (TypeError, OSError):
            matches = False
        if matches and isinstance(payload, dict) and "run_identity" in payload:
            state["resume"] = payload
        return payload

    reference.torch.load = stashing_load
    state["continuation_of_epochs"] = int(continuation_of_epochs)
    state["port_mode"] = port_mode
    return state


def _reconcile_continuation_identity(
    state: dict[str, Any],
    current_identity: dict[str, Any],
) -> None:
    """Fail-closed epoch-extension reconciliation (see adapter docstring)."""
    import copy

    resume = state.get("resume")
    if resume is None:
        raise MPSPortError(
            "continuation adapter: resume checkpoint was not loaded before "
            "run-identity computation; reference call order changed — re-audit"
        )
    observed = resume.get("run_identity") or {}
    drifting = sorted(
        k
        for k in set(observed) | set(current_identity)
        if observed.get(k) != current_identity.get(k)
    )
    illegal = [k for k in drifting if k not in _CONTINUATION_ALLOWED_DRIFT]
    if illegal:
        raise MPSPortError(
            f"continuation refused: non-continuation identity drift {illegal} "
            f"(full drift set: {drifting})"
        )
    if "training_config" in drifting:
        parent_cfg = observed.get("training_config") or {}
        current_cfg = current_identity.get("training_config") or {}
        cfg_drift = sorted(
            k
            for k in set(parent_cfg) | set(current_cfg)
            if parent_cfg.get(k) != current_cfg.get(k)
        )
        if cfg_drift != ["epochs"]:
            raise MPSPortError(
                f"continuation refused: training_config drift {cfg_drift} "
                "(only 'epochs' may change in a continuation)"
            )
    if observed.get("ema_policy") != current_identity.get("ema_policy"):
        raise MPSPortError(
            "continuation refused: ema_policy drifted despite parent-geometry "
            "hold — resolve_ema_policy patch did not take effect"
        )
    record = {
        "event": "wrapper_epoch_extension_continuation",
        "port_mode": state.get("port_mode"),
        "continuation_of_epochs": state.get("continuation_of_epochs"),
        "parent_run_identity_sha256": _canonical_sha256_of(observed),
        "drifting_keys_accepted": drifting,
    }
    lineage = resume.setdefault(
        "resume_lineage",
        {"schema": "ddm_cl1_hpac_capacity_resume_lineage.v1", "entries": []},
    )
    # The reference stores lineage as a dict-with-entries on fresh runs but a
    # bare list in saved checkpoints; append to whichever shape is present.
    if isinstance(lineage, list):
        lineage.append(record)
    else:
        lineage.setdefault("entries", []).append(record)
    resume["run_identity"] = copy.deepcopy(current_identity)
    print(
        "[continuation] epoch-extension identity reconciled: accepted drift "
        f"{drifting}; parent identity recorded in resume_lineage",
        flush=True,
    )


def _canonical_sha256_of(payload: dict[str, Any]) -> str:
    import json

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _configure_reference(
    reference: ModuleType,
    *,
    port_mode: str,
    raw_launch_argv: Sequence[str],
    resume_from: Path | None = None,
) -> None:
    mode = PORT_MODES[port_mode]
    lr_hold_boundary = mode.get("lr_floor_hold_after_epoch")
    if lr_hold_boundary is not None:
        _install_lr_floor_hold(reference, int(lr_hold_boundary))
    ema_geometry_epochs = mode.get("ema_geometry_epochs")
    if ema_geometry_epochs is not None:
        original_resolve = reference.resolve_ema_policy
        geometry_ratio = int(ema_geometry_epochs), int(mode["epochs"])

        def held_resolve_ema_policy(updates_per_run, **kwargs):  # type: ignore[no-untyped-def]
            if updates_per_run is not None:
                held = int(updates_per_run) * geometry_ratio[0] // geometry_ratio[1]
                print(
                    f"[continuation] EMA geometry held at parent horizon: "
                    f"updates_per_run {updates_per_run} -> {held}",
                    flush=True,
                )
                updates_per_run = held
            return original_resolve(updates_per_run, **kwargs)

        reference.resolve_ema_policy = held_resolve_ema_policy
    continuation_state: dict[str, Any] | None = None
    if mode.get("continuation_of_epochs") is not None:
        if resume_from is None:
            raise MPSPortError(f"{port_mode} requires --resume-from for the continuation adapter")
        continuation_state = _install_continuation_adapter(
            reference,
            resume_path=resume_from,
            continuation_of_epochs=int(mode["continuation_of_epochs"]),
            port_mode=port_mode,
        )
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
                    + (
                        f"; lr_floor_hold_after_epoch={mode['lr_floor_hold_after_epoch']}"
                        " (eta_min held past the resume boundary — within-regime"
                        " continuation of the measured power tail)"
                        if mode.get("lr_floor_hold_after_epoch") is not None
                        else ""
                    )
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
        if continuation_state is not None:
            _reconcile_continuation_identity(continuation_state, identity)
        return identity

    reference._run_identity = ported_run_identity


def main(argv: Sequence[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    port_mode, reference_argv = _split_port_args(raw)
    reference = _load_reference()
    validated = _validate_mode_args(reference, port_mode, reference_argv)
    _configure_reference(
        reference,
        port_mode=port_mode,
        raw_launch_argv=[sys.executable, str(Path(__file__).resolve()), *raw],
        resume_from=validated.resume_from,
    )
    sys.argv = [str(Path(__file__).resolve()), *reference_argv]
    reference.main()


if __name__ == "__main__":
    main()
