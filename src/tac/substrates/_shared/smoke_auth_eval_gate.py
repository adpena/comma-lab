# SPDX-License-Identifier: MIT
"""Canonical smoke/full auth-eval gate for substrate trainers.

Per CLAUDE.md "Auth eval EVERYWHERE" non-negotiable + HNeRV parity discipline
lesson L13 + the Path B council-grade decision documented in
``feedback_pr95plus_smoke_archive_completion_landed_20260513.md``: smoke
mode is INFRASTRUCTURE-validation (non-promotable;
``smoke_validation_contract=training_artifact_v1``) and MUST NOT invoke
``experiments/contest_auth_eval.py`` because smoke archives often do not
render a valid contest ``.raw`` output (the substrate is L0 SCAFFOLD per
the same lesson).

Anchors (each substrate previously open-coded the same gate; this module
makes the gate the single source of truth across substrates):

* PR95++: ``experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py``
  (Path B fix at HEAD ``54e4a306e``).
* Time-traveler: ``experiments/train_substrate_time_traveler_l5_autonomy.py``
  (``full_cpu_active`` + smoke-skip pattern).
* Wyner-Ziv: ``experiments/train_substrate_wyner_ziv_cooperative_receiver.py``
  (``full_cpu_active`` + smoke-skip pattern).
* L2 SAR: ``experiments/train_substrate_sar_coherent_pose_pairs.py``
  (subset of the gate; coordinated with the SAR-TRAINER-DEBUG sister fix).

The helper accepts an ``argparse.Namespace`` plus the archive/runtime
paths and the contest auth-eval script path. It returns ``None`` when
the gate refuses auth eval (smoke / explicit ``--skip-auth-eval`` /
non-CUDA device); it returns a dict of ``auth_eval_*`` keys when auth
eval actually runs and produces a valid contest-CUDA score claim. The
canonical refusal banners are written to stderr-equivalent ``print(...)``
so the operator sees the contract enforced loud-and-clear inside the
training log.

Per CLAUDE.md "Forbidden score claims" non-negotiable: implicit local CPU
coercion is still refused, but the training device and auth-eval device are
separate. Modal can train on CUDA while explicitly running auth eval on CPU
to avoid NVDEC/DALI. A Linux x86_64 CPU auth-eval run may be valid
``[contest-CPU]`` evidence, but it must never flow through CUDA claim fields.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable:
the helper runs explicit ``auth_eval_device=cpu`` / ``AUTH_EVAL_DEVICE=cpu``
requests and preserves their JSON custody on the payload's own score axis.
Legacy CUDA-claim callers receive ``None`` for non-CUDA results by default;
new CPU-aware callers can opt in with ``return_non_cuda_result=True``.

Returns the canonical claim dict on success::

    {
        "auth_eval_json_path": str(output_json),
        "auth_eval_cuda_score": float(claim.score),
        "auth_eval_score_axis": "contest_cuda",
        "auth_eval_lane_tag": claim.lane_tag,
        "auth_eval_score_claim_valid": True,
        "auth_eval_exact_cuda_complete": True,
    }

Returns ``None`` on refusal AND sets ``args.auth_eval_skipped_reason``
(if the namespace is writable) so the caller can persist the reason
into ``manifest.json`` / ``provenance.json`` / posterior anchors. The
caller is responsible for the manifest write; this helper only
formats and emits the reason string.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "AuthEvalGateError",
    "AuthEvalGateRefusal",
    "format_smoke_skip_banner",
    "gate_auth_eval_call",
]


class AuthEvalGateRefusal(str):
    """Sentinel-string subclass that surfaces the refusal reason.

    Returned via ``args.auth_eval_skipped_reason`` so downstream manifest
    writers can record the canonical refusal token alongside other
    provenance.
    """

    __slots__ = ()


class AuthEvalGateError(RuntimeError):
    """Raised when auth eval runs but does not produce a valid score claim."""


# Refusal reason tokens (stable strings; consumed by autopilot harvest and
# Catalog #127 custody validator downstream). Adding a new token requires
# updating the union of accepted reasons in
# `tac.continual_learning.AUTHORITATIVE_TAGS`-adjacent validators.
SMOKE_REFUSAL_REASON = (
    "smoke_validation_contract=training_artifact_v1; "
    "auth-eval refused at smoke per Path B council decision "
    "(score_claim=false, evidence_grade=training-only)"
)
CPU_REFUSAL_REASON = (
    "device_type=cpu; contest_auth_eval CUDA-axis path requires "
    "device.type=='cuda'. Linux x86_64 CPU auth eval is a separate "
    "operator-routed workflow per CLAUDE.md `Submission auth eval` "
    "non-negotiable; refusing inline CPU coercion."
)
EXPLICIT_NON_CUDA_AUTH_EVAL_RESULT_REASON = (
    "auth_eval_device is explicit and non-CUDA; contest_auth_eval result "
    "was produced on its own score_axis and must not be consumed as a "
    "contest-CUDA claim. Legacy CUDA-claim callers receive None unless "
    "return_non_cuda_result=True."
)
SKIP_FLAG_REASON = (
    "--skip-auth-eval set; caller explicitly disabled the auth eval path"
)
FULL_CPU_REFUSAL_REASON = (
    "full_cpu_advisory_only=true; macOS-CPU is NOT 1:1 contest-compliant "
    "per CLAUDE.md `Submission auth eval — BOTH CPU AND CUDA, ON 1:1 "
    "CONTEST-COMPLIANT HARDWARE` non-negotiable; use the sister smoke "
    "harness for [macOS-CPU advisory only] proxy signal"
)


def format_smoke_skip_banner(substrate_tag: str) -> str:
    """Return the canonical loud banner for smoke-mode skip."""

    return (
        f"[{substrate_tag}] [WARNING] smoke mode does NOT run "
        "contest_auth_eval; archive non-promotable per "
        "smoke_validation_contract=training_artifact_v1 "
        "(manifest.json + non-empty archive.zip only; no score-axis "
        "claim). Full-mode auth eval lands behind the Phase 2 design "
        "memo per HNeRV parity discipline lesson L13."
    )


def _record_refusal(
    args: argparse.Namespace,
    reason: str,
    *,
    record_field: str = "auth_eval_skipped_reason",
) -> None:
    """Stamp the refusal reason onto the namespace for downstream manifest.

    Best-effort; if the namespace is locked / read-only, the caller can
    still rely on the function's return value (``None``) and the printed
    banner.
    """

    try:
        setattr(args, record_field, AuthEvalGateRefusal(reason))
    except (AttributeError, TypeError):  # frozen namespace / dataclass-mode
        pass


def _detect_device_type(device: Any) -> str:
    """Best-effort device.type extraction from common shapes.

    Accepts:
    * ``torch.device`` instance -> ``device.type``
    * ``str`` -> the string itself (canonical: ``"cuda"`` or ``"cpu"``)
    * Anything else -> ``str(device).lower()`` with the common cuda:n
      prefix collapsed to ``"cuda"``.
    """

    if device is None:
        return "cpu"
    # torch.device path (avoid importing torch at module level for cheap
    # tests / smoke environments)
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str) and device_type:
        return device_type.lower()
    text = str(device).strip().lower()
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("cpu"):
        return "cpu"
    if text.startswith("mps"):
        return "mps"
    return text or "cpu"


def _explicit_auth_eval_device(auth_eval_device: str | None) -> str | None:
    """Return the explicit auth-eval device, preferring arg over environment."""

    if auth_eval_device is not None:
        text = str(auth_eval_device).strip()
        return text or None
    text = os.environ.get("AUTH_EVAL_DEVICE", "").strip()
    return text or None


# Canonical device-suffix tokens recognized in auth-eval output filenames.
# A filename containing one of these tokens is interpreted as device-claimed; if
# the actual ``auth_eval_device_type`` does not match, the filename is rewritten
# so the on-disk artifact name matches the on-disk content's score axis.
_AUTH_EVAL_DEVICE_FILENAME_TOKENS = ("cuda", "cpu", "mps")


def _redirect_output_json_to_match_device(
    output_json: Path, auth_eval_device_type: str, *, substrate_tag: str
) -> Path:
    """Rewrite ``output_json`` to a device-correct filename when needed.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden
    component-aliasing for baselines" non-negotiables: an artifact's filename
    MUST NOT make a claim its content cannot honor. When a caller passes
    ``contest_auth_eval_cuda.json`` and the actual ``auth_eval_device_type`` is
    ``cpu``, this helper rewrites the filename to ``contest_auth_eval_cpu.json``
    and emits a loud stderr-equivalent warning so the caller can be migrated to
    a device-aware pattern at the source.

    The historical lie surface (Z3 v2 FULL 2026-05-15:
    ``contest_auth_eval_cuda.json`` containing ``device=cpu``) is structurally
    extincted: future dispatches with the same caller code path land at
    ``contest_auth_eval_cpu.json`` and ``contest_auth_eval_cpu_work/`` so the
    filename matches the content. Catalog #249 is the structural sister gate.
    """

    stem = output_json.stem
    suffix = output_json.suffix
    parent = output_json.parent

    # Case A: filename has no device suffix — preserve as-is (device-agnostic
    # callers are honest by construction).
    matched_token: str | None = None
    for token in _AUTH_EVAL_DEVICE_FILENAME_TOKENS:
        # Match `_cuda.json` / `_cpu.json` / `_mps.json` style suffixes.
        # Word-boundary check: token must be preceded by `_` or be at the start.
        if stem.endswith(f"_{token}"):
            matched_token = token
            break
    if matched_token is None:
        return output_json

    # Case B: filename's device token matches the actual device — no lie.
    if matched_token == auth_eval_device_type:
        return output_json

    # Case C: filename's device token DOES NOT match the actual device. Rewrite.
    new_stem = stem[: -(len(matched_token) + 1)] + f"_{auth_eval_device_type}"
    new_path = parent / f"{new_stem}{suffix}"
    print(
        f"[{substrate_tag}] [WARNING] [phantom-score-fix] caller passed "
        f"output_json={output_json.name!r} but auth_eval_device_type="
        f"{auth_eval_device_type!r}; rewriting output to {new_path.name!r} "
        "to prevent phantom-score directory-name lie. Update the caller to "
        "construct the filename from auth_eval_device. See Catalog #249 + "
        "CLAUDE.md FORBIDDEN PATTERNS 'Forbidden misleading-directory-name "
        "(the phantom-score directory trap)'.",
        flush=True,
    )
    return new_path


def gate_auth_eval_call(
    *,
    args: argparse.Namespace,
    archive_zip: Path,
    inflate_sh: Path,
    upstream_dir: Path,
    output_json: Path,
    contest_auth_eval_script: Path,
    substrate_tag: str,
    device: Any = None,
    full_cpu_active: bool = False,
    required_score_axis: str = "contest_cuda",
    require_component_recompute: bool = True,
    auth_eval_device: str | None = None,
    return_non_cuda_result: bool = False,
    extra_argv: tuple[str, ...] | None = None,
) -> dict[str, object] | None:
    """Canonical gate: skip ``contest_auth_eval`` at smoke; run at full.

    Args:
        args: argparse.Namespace from the trainer. Must carry ``smoke`` and
            ``skip_auth_eval`` boolean attributes (any falsy default is OK).
        archive_zip: Path to the materialized ``archive.zip``.
        inflate_sh: Path to the substrate's ``inflate.sh`` runtime entry.
        upstream_dir: Path to the upstream snapshot (passed verbatim to
            ``contest_auth_eval.py``).
        output_json: Where ``contest_auth_eval.py`` writes its result JSON.
            Must be under durable operator/provider storage so the result
            survives the lifecycle. Per CLAUDE.md the canonical location is
            ``args.output_dir / "contest_auth_eval_cuda.json"``.
        contest_auth_eval_script: Path to ``experiments/contest_auth_eval.py``.
        substrate_tag: Short tag used in printed banners (e.g. ``"pr95plus"``,
            ``"sane_hnerv"``, ``"wyner_ziv"``, ``"time_traveler_l5"``).
        device: ``torch.device`` / str / None. ``device.type != "cuda"`` is
            refused per CLAUDE.md "Submission auth eval" non-negotiable.
        full_cpu_active: When True the trainer is on the ``--full-cpu``
            advisory-only path (time-traveler L5; macOS-CPU); auth eval is
            refused with ``FULL_CPU_REFUSAL_REASON``.
        required_score_axis: Forwarded to ``parse_auth_eval_score_claim``.
        require_component_recompute: Forwarded to
            ``parse_auth_eval_score_claim``; default True matches
            ``[contest-CUDA]`` custody requirements.
        auth_eval_device: Optional explicit auth-eval device. When omitted,
            ``AUTH_EVAL_DEVICE`` is honored; when neither is set, the trainer
            ``device`` is used for backward-compatible implicit CUDA gating.
        return_non_cuda_result: When False, explicit CPU auth eval still runs
            and writes ``output_json`` but returns ``None`` so legacy callers
            that treat non-None as a CUDA claim cannot accidentally consume a
            CPU score. CPU-aware callers may opt in to the structured result.
        extra_argv: Optional extra positional args appended to the
            ``contest_auth_eval.py`` invocation (per-substrate flags).

    Returns:
        ``None`` when the gate refuses auth eval. The caller's reason is
        also stamped onto ``args.auth_eval_skipped_reason`` (best-effort).

        A dict with ``auth_eval_*`` keys on success (auth eval ran AND
        produced a valid contest-CUDA score claim).

    Raises:
        ``AuthEvalGateError`` when auth eval ran but did not produce a
        valid claim (silent-success refused per CLAUDE.md).
        ``RuntimeError`` when ``contest_auth_eval.py`` exits non-zero.
    """

    smoke = bool(getattr(args, "smoke", False))
    skip_auth_eval = bool(getattr(args, "skip_auth_eval", False))
    training_device_type = _detect_device_type(device)
    explicit_eval_device = _explicit_auth_eval_device(auth_eval_device)
    auth_eval_device_type = (
        _detect_device_type(explicit_eval_device)
        if explicit_eval_device is not None
        else training_device_type
    )

    # 1. Smoke ALWAYS skips (defense-in-depth even if --skip-auth-eval not set).
    if smoke:
        print(format_smoke_skip_banner(substrate_tag), flush=True)
        _record_refusal(args, SMOKE_REFUSAL_REASON)
        return None

    # 2. Full-CPU advisory-only path (time-traveler L5 lineage).
    if full_cpu_active:
        print(
            f"[{substrate_tag}-full-cpu] auth eval SKIPPED "
            "(macOS-CPU is not 1:1 contest-compliant; use sister smoke "
            "harness for [macOS-CPU advisory only] proxy signal).",
            flush=True,
        )
        _record_refusal(args, FULL_CPU_REFUSAL_REASON)
        return None

    # 3. Explicit caller opt-out via --skip-auth-eval.
    if skip_auth_eval:
        print(
            f"[{substrate_tag}] auth eval SKIPPED (--skip-auth-eval set).",
            flush=True,
        )
        _record_refusal(args, SKIP_FLAG_REASON)
        return None

    # 4. Non-CUDA training device: refuse implicit local/advisory coercion.
    # Explicit AUTH_EVAL_DEVICE=cpu is handled below as a real auth-eval run
    # with its own score axis. This keeps CPU/GPU differences measurable
    # without letting CPU evidence masquerade as CUDA evidence.
    if explicit_eval_device is None and training_device_type != "cuda":
        print(
            f"[{substrate_tag}] auth eval SKIPPED "
            f"(device.type={training_device_type!r}; "
            "contest CUDA axis requires device.type='cuda').",
            flush=True,
        )
        _record_refusal(args, CPU_REFUSAL_REASON)
        return None

    # 5. Explicit auth-eval devices may be CPU or CUDA. MPS and other devices
    # remain refused here because they are proxy axes, not contest auth-eval
    # axes, and several provider paths cannot run them deterministically.
    if auth_eval_device_type not in {"cpu", "cuda"}:
        print(
            f"[{substrate_tag}] auth eval SKIPPED "
            f"(auth_eval_device={auth_eval_device_type!r}; only cpu/cuda "
            "are canonical auth-eval devices).",
            flush=True,
        )
        _record_refusal(args, CPU_REFUSAL_REASON)
        return None

    # 6. Run the canonical contest_auth_eval.py invocation.
    #
    # The evaluator's custody guard rejects score-grade evidence whose work
    # dir is silently deleted or parked under the process temp root. Keep the
    # exact-eval work tree next to the requested JSON by default so trainer
    # artifacts remain inspectable after harvest. Modal training wrappers are
    # a narrow exception: they intentionally force AUTH_EVAL_DEVICE=cpu and
    # MODAL_AUTH_EVAL_ADVISORY_ONLY=1 because the training container does not
    # provide NVDEC. That path is diagnostic/advisory only, so the evaluator's
    # explicit temp bypass is appropriate and keeps the failure mode visible.
    #
    # PHANTOM-SCORE BUG CLASS PERMANENT FIX (Catalog #249, 2026-05-15):
    # Trainers historically hardcoded ``output_json=output_dir / "contest_auth_eval_cuda.json"``
    # regardless of the actual ``auth_eval_device``. When AUTH_EVAL_DEVICE=cpu
    # was injected by the Modal dispatcher, the CPU eval result landed in a
    # file named ``*_cuda.json`` and a directory named ``*_cuda_work/``. Parent
    # agents then quoted the filename as evidence of a CUDA score that did not
    # exist. Empirical anchor: Z3 v2 FULL 2026-05-15 dispatch produced
    # ``contest_auth_eval_cuda.json`` containing ``device=cpu`` /
    # ``score_axis=diagnostic_cpu`` — the file's CONTENT was honest, but the
    # FILENAME lied. The fix re-derives the output filename + work_dir to match
    # the actual ``auth_eval_device_type`` and emits a LOUD warning if the
    # caller passed a misleading path so the upstream caller can be migrated.
    output_json = _redirect_output_json_to_match_device(
        output_json, auth_eval_device_type, substrate_tag=substrate_tag
    )
    work_dir = output_json.parent / f"{output_json.stem}_work"
    modal_cpu_advisory = (
        auth_eval_device_type == "cpu"
        and os.environ.get("MODAL_AUTH_EVAL_ADVISORY_ONLY", "").strip() == "1"
    )
    cmd = [
        sys.executable,
        str(contest_auth_eval_script),
        "--archive",
        str(archive_zip),
        "--inflate-sh",
        str(inflate_sh),
        "--upstream-dir",
        str(upstream_dir),
        "--device",
        auth_eval_device_type,
        "--inflate-device",
        auth_eval_device_type,
        "--json-out",
        str(output_json),
        "--keep-work-dir",
        "--work-dir",
        str(work_dir),
    ]
    if modal_cpu_advisory:
        cmd.append("--allow-temp-work-dir")
    if extra_argv:
        cmd.extend(str(a) for a in extra_argv)
    print(f"[{substrate_tag}-auth-eval] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[{substrate_tag}] contest_auth_eval.py failed "
            f"rc={proc.returncode}; stdout_tail={proc.stdout[-2000:]}; "
            f"stderr_tail={proc.stderr[-2000:]}"
        )

    # 7. Parse + validate the result. CUDA goes through the strict claim
    # parser; CPU keeps its own axis and defaults to a None return for legacy
    # callers that only understand CUDA claim dicts.
    from tac.auth_eval_result import parse_auth_eval_score_claim, parse_finite_auth_eval_score

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if auth_eval_device_type != "cuda":
        if modal_cpu_advisory and (
            payload.get("score_axis") == "contest_cpu"
            or payload.get("cpu_leaderboard_reproduction_eligible") is True
        ):
            raise AuthEvalGateError(
                f"[{substrate_tag}] Modal advisory CPU auth eval produced a "
                "contest-CPU-authority payload; expected diagnostic demotion "
                "from contest_auth_eval.py"
            )
        parsed = parse_finite_auth_eval_score(
            payload,
            require_component_recompute=require_component_recompute,
        )
        if parsed is None:
            raise AuthEvalGateError(
                f"[{substrate_tag}] contest_auth_eval.py completed on "
                f"{auth_eval_device_type} but did not produce a finite, "
                "component-coherent score payload"
            )
        result = {
            "auth_eval_json_path": str(output_json),
            "auth_eval_score": float(parsed.score),
            "auth_eval_cpu_score": (
                float(parsed.score) if auth_eval_device_type == "cpu" else None
            ),
            "auth_eval_device": auth_eval_device_type,
            "auth_eval_score_axis": str(
                payload.get("score_axis") or f"diagnostic_{auth_eval_device_type}"
            ),
            "auth_eval_lane_tag": str(payload.get("lane_tag") or ""),
            "auth_eval_evidence_grade": str(payload.get("evidence_grade") or ""),
            "auth_eval_score_claim": payload.get("score_claim") is True,
            "auth_eval_score_claim_valid": (
                payload.get("score_claim_valid") is True
            ),
            "auth_eval_promotion_eligible": (
                payload.get("promotion_eligible") is True
            ),
            "auth_eval_exact_cuda_complete": (
                payload.get("exact_cuda_eval_complete") is True
            ),
            "auth_eval_cpu_leaderboard_reproduction_eligible": (
                payload.get("cpu_leaderboard_reproduction_eligible") is True
            ),
        }
        _record_refusal(args, EXPLICIT_NON_CUDA_AUTH_EVAL_RESULT_REASON)
        return result if return_non_cuda_result else None

    claim = parse_auth_eval_score_claim(
        payload,
        required_score_axis=required_score_axis,
        require_component_recompute=require_component_recompute,
    )
    if claim is None:
        raise AuthEvalGateError(
            f"[{substrate_tag}] contest_auth_eval.py completed but did "
            f"not produce a valid {required_score_axis} score claim; "
            "refusing silent success per CLAUDE.md `Forbidden score "
            "claims` non-negotiable"
        )
    return {
        "auth_eval_json_path": str(output_json),
        "auth_eval_cuda_score": float(claim.score),
        "auth_eval_score_axis": required_score_axis,
        "auth_eval_lane_tag": claim.lane_tag,
        "auth_eval_score_claim_valid": True,
        "auth_eval_exact_cuda_complete": True,
    }
