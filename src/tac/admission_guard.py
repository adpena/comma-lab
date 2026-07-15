"""#254 governor admission-bypass GUARD — refuse raw heavy launches that skip the P0 gate.

The P0 machine-crash rule (a concurrent >128 GB run CRASHED the box) says every HEAVY train/
measurement launch MUST go through the governed admission path (``tools/launch_witness_run.py``
or ``tools/safe_run.py``), which registers the projected footprint with the system memory
governor BEFORE spawning. The failure mode this guard extincts: someone runs the heavy trainer
RAW (``.venv/bin/python experiments/train_...py ...``), bypassing admission → uncapped RSS →
crash. ``tools/system_memory_governor.py`` owns the admission decision; this module is the
RUNTIME assertion the heavy entrypoint calls at startup so a raw launch fails CLOSED.

Contract:
  * A governed launcher calls :func:`mark_admitted_env` on the child's env before spawning; the
    child then sees ``TAC_GOVERNED_ADMISSION=1`` and passes.
  * A heavy entrypoint calls :func:`assert_governed_admission` at the TOP of ``main()``. When
    admission is ENFORCED (``TAC_ADMISSION_ENFORCE=1`` or ``.omx/state/admission_enforce.flag``
    — the SAME predicate the governor uses) and the marker is absent, it REFUSES (exit 7) with
    a clear message pointing at the governed launchers. Default ADVISORY (warn only) until
    enforce is armed — mirrors the governor's strict-flip discipline (enforce ⊆ advisory).
  * A reviewed raw exception sets ``TAC_ADMISSION_BYPASS_OK=<reason>`` (non-empty rationale;
    placeholder literals rejected), which admits + logs the bypass for audit.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GOVERNED_MARKER_ENV = "TAC_GOVERNED_ADMISSION"       # set by governed launchers on the child env
ADMISSION_ENFORCE_ENV = "TAC_ADMISSION_ENFORCE"      # per-invocation enforce override (mirrors governor)
BYPASS_OVERRIDE_ENV = "TAC_ADMISSION_BYPASS_OK"      # reviewed raw-run exception (non-empty rationale)
EXIT_ADMISSION_REFUSED = 7
EXIT_DSL_COMPILE_REFUSED = 8

_TRUTHY = {"1", "true", "yes", "on"}
# Placeholder rationales that must NOT satisfy the bypass (so the docstring example cannot self-waive).
_PLACEHOLDER_RATIONALES = {"", "<reason>", "<rationale>", "reason", "rationale", "todo", "tbd", "x"}
# Durable, project-scoped enforce flag — the SAME path system_memory_governor uses. Absolute so it is
# cwd-independent (the guard runs from experiments/ / tools/ / repo-root alike).
_ENFORCE_FLAG = Path(__file__).resolve().parents[2] / ".omx" / "state" / "admission_enforce.flag"


def admission_enforcing(env: dict | None = None) -> bool:
    """True iff the admission gate is in ENFORCE mode. MIRRORS
    ``tools/system_memory_governor.admission_enforcing`` (env var OR durable flag file, truthy
    first line). Kept in sync by :func:`test_admission_guard` parity check."""
    e = os.environ if env is None else env
    if (e.get(ADMISSION_ENFORCE_ENV, "0") or "").strip().lower() in _TRUTHY:
        return True
    try:
        if _ENFORCE_FLAG.is_file():
            first = (_ENFORCE_FLAG.read_text().splitlines() or [""])[0].strip().lower()
            return first in _TRUTHY
    except OSError:
        return False
    return False


def bypass_rationale(env: dict | None = None) -> str | None:
    """The reviewed raw-run bypass rationale, if a NON-PLACEHOLDER one is set; else None."""
    e = os.environ if env is None else env
    raw = (e.get(BYPASS_OVERRIDE_ENV, "") or "").strip()
    return raw if raw and raw.lower() not in _PLACEHOLDER_RATIONALES and len(raw) >= 4 else None


def is_admitted(env: dict | None = None) -> bool:
    """True iff this process was launched through a governed path (marker present) OR carries a
    reviewed bypass rationale."""
    e = os.environ if env is None else env
    if (e.get(GOVERNED_MARKER_ENV, "") or "").strip().lower() in _TRUTHY:
        return True
    return bypass_rationale(e) is not None


def mark_admitted_env(env: dict) -> dict:
    """Stamp ``env`` (a child-process env dict a governed launcher is about to spawn with) as
    admitted. Idempotent; returns the same dict for chaining."""
    env[GOVERNED_MARKER_ENV] = "1"
    return env


def _requires_dsl_compile_hash(label: str) -> bool:
    """The witness trainer family is the Catalog #406 protected surface."""

    return "witness" in str(label).lower()


def dsl_compile_admission_status(
    label: str,
    env: dict | None = None,
    *,
    process_argv: list[str] | tuple[str, ...] | None = None,
) -> tuple[bool, str]:
    """Recompute the DSL binding and match it to this trainer's actual argv."""

    if not _requires_dsl_compile_hash(label):
        return True, f"dsl_compile_hash not required for non-witness entrypoint {label!r}"
    e = os.environ if env is None else env
    from tac.v9_provenance_gates import (
        DSL_COMPILE_HASH_ENV,
        DSL_LAUNCH_SH_PATH_ENV,
        DSL_PROVENANCE_PATH_ENV,
        verify_dsl_provenance_artifacts,
        verify_dsl_provenance_document,
    )

    carried = (e.get(DSL_COMPILE_HASH_ENV, "") or "").strip()
    launch_sh = (e.get(DSL_LAUNCH_SH_PATH_ENV, "") or "").strip()
    provenance_path = (e.get(DSL_PROVENANCE_PATH_ENV, "") or "").strip()
    if not carried or not launch_sh or not provenance_path:
        return False, (
            "DSL COMPILE REFUSED: missing TAC_DSL_COMPILE_HASH / provenance / launch.sh "
            "custody; a governed marker alone cannot admit a witness trainer"
        )
    ok, detail = verify_dsl_provenance_artifacts(
        launch_sh,
        provenance_path=provenance_path,
        expected_hash=carried,
    )
    if not ok:
        return False, f"DSL COMPILE REFUSED: artifact recomputation failed: {detail}"
    if process_argv is not None:
        try:
            document = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"DSL COMPILE REFUSED: provenance reread failed: {exc}"
        recorded_argv = list(document.get("resolved_argv") or ())
        if not recorded_argv:
            return False, "DSL COMPILE REFUSED: provenance has no resolved interpreter argv"
        # Python does not expose the interpreter in sys.argv.  Reuse only the
        # already governor-verified recorded interpreter, then compare the
        # actual script+flags byte-for-byte (apart from declared volatile values).
        running = [str(recorded_argv[0]), *[str(token) for token in process_argv]]
        ok, detail = verify_dsl_provenance_document(
            document,
            launch_argv=running,
            expected_hash=carried,
        )
        if not ok:
            return False, f"DSL COMPILE REFUSED: running argv mismatch: {detail}"
    return True, f"DSL compile admission OK for {label!r}: {detail}"


def admission_status(
    label: str,
    env: dict | None = None,
    *,
    process_argv: list[str] | tuple[str, ...] | None = None,
) -> tuple[bool, str]:
    """Return (ok, message) WITHOUT exiting — the testable core of the guard. ``ok`` is False
    only when enforcement is ON and the process is neither governed nor bypass-approved."""
    e = os.environ if env is None else env
    dsl_ok, dsl_detail = dsl_compile_admission_status(
        label, e, process_argv=process_argv
    )
    if not dsl_ok:
        return False, dsl_detail
    enforcing = admission_enforcing(e)
    admitted = is_admitted(e)
    if admitted:
        why = "bypass-approved" if bypass_rationale(e) else "governed"
        return True, f"admission OK ({why}) for {label!r}; {dsl_detail}"
    if not enforcing:
        return True, (f"admission ADVISORY: {label!r} was NOT launched through the governed path "
                      f"(tools/launch_witness_run.py or tools/safe_run.py) — allowed because "
                      f"{ADMISSION_ENFORCE_ENV} is not armed. Route heavy launches through the "
                      f"governor to avoid the P0 machine-crash risk.")
    return False, (f"REFUSED: {label!r} bypassed the P0 memory admission gate. Heavy train/"
                   f"measurement launches MUST go through tools/launch_witness_run.py or "
                   f"tools/safe_run.py (they register the footprint with the system memory "
                   f"governor before spawning; raw launches can crash the box). For a reviewed "
                   f"raw exception set {BYPASS_OVERRIDE_ENV}=<non-empty rationale>.")


def assert_governed_admission(label: str, *, env: dict | None = None,
                              on_refuse: str = "exit") -> bool:
    """Call at the TOP of a heavy entrypoint's ``main()``. Prints the status; when REFUSED
    (enforce armed + not governed + no bypass) either exits ``EXIT_ADMISSION_REFUSED`` (default)
    or raises ``PermissionError`` (``on_refuse="raise"``). Returns True when admitted/advisory."""
    ok, msg = admission_status(label, env, process_argv=sys.argv)
    stream = sys.stderr if not ok else sys.stdout
    print(f"[admission-guard] {msg}", file=stream)
    if ok:
        return True
    if on_refuse == "raise":
        raise PermissionError(msg)
    exit_code = EXIT_DSL_COMPILE_REFUSED if "DSL COMPILE REFUSED" in msg else EXIT_ADMISSION_REFUSED
    sys.exit(exit_code)


__all__ = [
    "ADMISSION_ENFORCE_ENV",
    "BYPASS_OVERRIDE_ENV",
    "EXIT_ADMISSION_REFUSED",
    "EXIT_DSL_COMPILE_REFUSED",
    "GOVERNED_MARKER_ENV",
    "admission_enforcing",
    "admission_status",
    "assert_governed_admission",
    "bypass_rationale",
    "dsl_compile_admission_status",
    "is_admitted",
    "mark_admitted_env",
]
