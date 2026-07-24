# SPDX-License-Identifier: MIT
"""Typed WitnessProgram target for the DDM event-continuation engine."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from tac.witness_dsl.lawref import (
    LADDER_DERIVED_AT_CONFIG,
    InputRef,
    LawRef,
    resolve,
)

if TYPE_CHECKING:
    from tac.optimization.ddm_event_continuation import DDMEventContinuationV1

SCHEMA: Final = "DDMWitnessProgramV1"
OBJECTIVE: Final = "100*d_seg_R+sqrt(10*d_pose_YUV6_R)+25*archive_bytes/37545489"


class DDMWitnessProgramError(ValueError):
    """Fail-closed invalid compile target or source/value custody."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bound_source(repo_root: Path, path: str, expected_sha256: str) -> dict[str, Any]:
    candidate = repo_root / path
    if not candidate.is_file() or candidate.is_symlink():
        raise DDMWitnessProgramError(f"bound source is unavailable: {path}")
    payload = candidate.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise DDMWitnessProgramError(
            f"bound source SHA differs for {path}: {actual} != {expected_sha256}"
        )
    return {"path": path, "sha256": actual, "bytes": len(payload)}


@dataclass(frozen=True, slots=True)
class MetricSelectorV1:
    selector_id: str
    acceptance_metric: str = OBJECTIVE
    proposal_metric: str = "scorer_recursive_rank4_fisher_corrected_J"
    authority_roundtrip: str = "camera_Q8->uint8->R->frozen_SegNet_PoseNet->exact_archive_bytes"

    def __post_init__(self) -> None:
        if not self.selector_id or self.acceptance_metric != OBJECTIVE:
            raise DDMWitnessProgramError("metric selector lacks the exact contest functional")
        if "scorer_recursive" not in self.proposal_metric or "corrected_J" not in self.proposal_metric:
            raise DDMWitnessProgramError("proposal metric is not scorer-recursive corrected-J")

    def to_payload(self) -> dict[str, str]:
        return {
            "selector_id": self.selector_id,
            "acceptance_metric": self.acceptance_metric,
            "proposal_metric": self.proposal_metric,
            "authority_roundtrip": self.authority_roundtrip,
        }


@dataclass(frozen=True, slots=True)
class SolveHookV1:
    hook_id: str
    event: str
    implementation: str
    execution_enabled: bool
    required_receipts: tuple[str, ...]
    blocker: str | None = None

    def __post_init__(self) -> None:
        if not self.hook_id or not self.event or not self.implementation:
            raise DDMWitnessProgramError("solve hook identities must be nonempty")
        if not self.required_receipts:
            raise DDMWitnessProgramError(f"solve hook {self.hook_id} lacks receipt contract")
        if not self.execution_enabled and not self.blocker:
            raise DDMWitnessProgramError(f"disabled solve hook {self.hook_id} lacks blocker")

    def to_payload(self) -> dict[str, Any]:
        return {
            "hook_id": self.hook_id,
            "event": self.event,
            "implementation": self.implementation,
            "execution_enabled": self.execution_enabled,
            "required_receipts": list(self.required_receipts),
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True)
class DDMWitnessProgramV1:
    program_id: str
    event_continuation: DDMEventContinuationV1
    metric_selector: MetricSelectorV1
    solve_hooks: tuple[SolveHookV1, ...]
    ticket_path: str
    source_bindings: Mapping[str, str]
    beta2: float
    ema_decay: float
    inference_shadow: Literal["ema"]
    execution_allowed: bool
    op_gc1_5_execution_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.program_id or not self.ticket_path:
            raise DDMWitnessProgramError("program and ticket identities must be nonempty")
        if not 0.0 < self.beta2 < 1.0:
            raise DDMWitnessProgramError("beta2 must be in (0,1)")
        if not math.isclose(self.ema_decay, 0.997, rel_tol=0.0, abs_tol=1.0e-15):
            raise DDMWitnessProgramError("EMA decay must be exactly the operator-directed 0.997")
        if self.inference_shadow != "ema":
            raise DDMWitnessProgramError("DDM verdict/inference must consume the EMA shadow")
        if self.op_gc1_5_execution_enabled:
            raise DDMWitnessProgramError("OP-GC1-5 must remain preregistered and execution-disabled")
        hook_ids = {hook.hook_id for hook in self.solve_hooks}
        required = {"fork_head_solve", "head_offset_solver", "ms2_terminal_solve", "mc_finisher"}
        if hook_ids != required:
            raise DDMWitnessProgramError(f"solve-hook set differs: {sorted(hook_ids)}")
        if self.execution_allowed != self.event_continuation.execution_allowed:
            raise DDMWitnessProgramError("program and event-graph execution authority differ")
        required_sources = {
            "launcher",
            "consumer",
            "event_engine",
            "dm4_adapter",
            "dm4_constructor",
        }
        if set(self.source_bindings) != required_sources:
            raise DDMWitnessProgramError("DDM program source-binding set differs")

    def _lawrefs(self) -> tuple[LawRef, LawRef]:
        budget_updates = self.event_continuation.budget_caps.maximum_receiver_verdicts
        beta_ref = LawRef(
            equation_id="adam_v_variance_warmup_length_v1",
            inputs={
                "beta2": InputRef.literal(
                    self.beta2,
                    "DDM selected beta2 fixed within a continuation segment",
                ),
                "steps_per_epoch": InputRef.literal(
                    1,
                    "DDM continuation clock: one receiver-verdict update per accepted-state epoch",
                ),
                "c": InputRef.literal(
                    2.0,
                    "RAdam variance-rectification two-memory-window guard",
                ),
            },
            ladder_class=LADDER_DERIVED_AT_CONFIG,
        )
        warmup_fraction = 2.0 / ((1.0 - self.ema_decay) * budget_updates)
        ema_ref = LawRef(
            equation_id="ema_decay_run_geometry_v1",
            inputs={
                "mode": InputRef.literal(
                    2,
                    "mode code 2 = decay_from_warmup_fraction",
                ),
                "updates_per_run": InputRef.literal(
                    budget_updates,
                    "explicit receiver-verdict safety cap; resource cap only",
                ),
                "warmup_fraction": InputRef.literal(
                    warmup_fraction,
                    "operator-directed 0.997 shadow geometry expressed as 2/((1-d)*U)",
                ),
            },
            ladder_class=LADDER_DERIVED_AT_CONFIG,
        )
        return beta_ref, ema_ref

    def resolve_constants(self, *, repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        beta_ref, ema_ref = self._lawrefs()
        beta_resolved = resolve(beta_ref, repo_root=repo_root)
        ema_resolved = resolve(ema_ref, repo_root=repo_root)
        if beta_resolved.fallback_used or ema_resolved.fallback_used:
            raise DDMWitnessProgramError("DDM beta2/EMA LawRef fallback is forbidden")
        if not math.isclose(float(ema_resolved.value), self.ema_decay, rel_tol=0.0, abs_tol=1.0e-15):
            raise DDMWitnessProgramError("EMA LawRef resolution differs from selected decay")
        constants = {
            "adam_beta2": self.beta2,
            "beta2_rewarmup_receiver_verdicts": int(beta_resolved.value),
            "ema_decay": float(ema_resolved.value),
            "inference_shadow": self.inference_shadow,
        }
        manifest = {
            "adam_beta2_rewarmup": beta_resolved.to_dict(),
            "ema_decay": ema_resolved.to_dict(),
        }
        return constants, manifest

    def compile_trainer_argv_with_constants(
        self,
        *,
        repo_root: Path,
        out_dir: str,
        mode: Literal["dry-run", "bounded-smoke", "full-run"] = "dry-run",
        resume_from: str | None = None,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """Compile only real launcher argv; never invent a trainer flag."""

        if mode == "full-run" and not self.execution_allowed:
            raise DDMWitnessProgramError("full-run compile refused: execution_allowed=false")
        sources = {
            name: _bound_source(repo_root, path, self.source_bindings[name])
            for name, path in {
                "launcher": "tools/launch_ddm_joint_descent.py",
                "consumer": "src/tac/optimization/direct_description_joint_descent.py",
                "event_engine": "src/tac/optimization/ddm_event_continuation.py",
                "dm4_adapter": "src/tac/optimization/ddm_dm4_j5_adapter.py",
                "dm4_constructor": "src/tac/optimization/ddm_dm4_targeted_realization_cures.py",
            }.items()
        }
        constants, lawref_manifest = self.resolve_constants(repo_root=repo_root)
        mode_flag = {
            "dry-run": "--dry-run",
            "bounded-smoke": "--bounded-smoke",
            "full-run": "--full-run",
        }[mode]
        argv = (
            "python3",
            "tools/launch_ddm_joint_descent.py",
            "--ticket",
            self.ticket_path,
            "--out-dir",
            out_dir,
            mode_flag,
        )
        if resume_from is not None:
            argv = (*argv, "--resume-from", resume_from)
        program_payload = self.to_payload()
        manifest = {
            "schema": "ddm_witness_program_compile.v1",
            "program_id": self.program_id,
            "program_semantic_hash": _sha256_bytes(_canonical_bytes(program_payload)),
            "event_graph_semantic_hash": self.event_continuation.semantic_hash,
            "argv": list(argv),
            "argv_sha256": _sha256_bytes(_canonical_bytes(list(argv))),
            "constants": constants,
            "lawrefs": lawref_manifest,
            "source_bindings": sources,
            "metric_selector": self.metric_selector.to_payload(),
            "causal_event_marks": {
                "schema": "pact.causal_manifest.v1",
                "row_kind": "event_mark",
                "stable_event_id": "sha256(canonical_event_identity)",
                "resume_semantics": "append_only_deduplicated",
            },
            "solve_hooks": [hook.to_payload() for hook in self.solve_hooks],
            "op_gc1_5": {
                "arm_id": "ddm_gc1_op5_d_first_v14_falsifier_v1",
                "preregistered": True,
                "execution_enabled": False,
            },
            "execution_allowed": self.execution_allowed,
        }
        manifest["typed_config_hash"] = _sha256_bytes(_canonical_bytes(manifest))
        return argv, manifest

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "program_id": self.program_id,
            "event_continuation": self.event_continuation.to_payload(),
            "metric_selector": self.metric_selector.to_payload(),
            "solve_hooks": [hook.to_payload() for hook in self.solve_hooks],
            "ticket_path": self.ticket_path,
            "source_bindings": dict(sorted(self.source_bindings.items())),
            "beta2": self.beta2,
            "ema_decay": self.ema_decay,
            "inference_shadow": self.inference_shadow,
            "execution_allowed": self.execution_allowed,
            "op_gc1_5_execution_enabled": self.op_gc1_5_execution_enabled,
        }


__all__ = [
    "OBJECTIVE",
    "SCHEMA",
    "DDMWitnessProgramError",
    "DDMWitnessProgramV1",
    "MetricSelectorV1",
    "SolveHookV1",
]
