# SPDX-License-Identifier: MIT
"""Run the Modal auth-eval wrappers' own guards LOCALLY, before any dispatch.

Why this module exists (measured 2026-08-10, three refusals in one session):

``modal run <file>::<entrypoint>`` hydrates the app -- which BUILDS the image --
*before* it invokes the ``@app.local_entrypoint()``. So a guard that lives inside
``main()`` still costs a full image build (~5 minutes) to fire, even though it
never touches the network. Every refusal in the lc2 dispatch chain was that
shape: one guard per round trip, five minutes each.

Importing the wrapper module costs 0.10 s. Its guards are ordinary Python. This
module imports the wrapper, resolves the entrypoint against
``app.registered_entrypoints``, checks every forwarded flag against the
entrypoint's real signature, and calls the wrapper's OWN runtime-tree validator
-- all locally, all before a Modal app is created.

Design rules this module honors:
  * It invents no flags and no validation logic. Every check reads the wrapper's
    actual signature or calls the wrapper's actual validator function.
  * A refusal here must be a refusal there. The point is to move the SAME
    failure earlier, never to add a new opinion about what is valid.
  * Checks that cannot be run locally are reported as UNCHECKED, never as pass.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]

# The token that separates ``modal run [modal-flags] FILE::ENTRYPOINT`` from the
# wrapper's own flags. Everything after the spec belongs to the entrypoint.
ENTRYPOINT_SPEC_SEPARATOR = "::"


@dataclass(frozen=True)
class PreflightRefusal:
    """One locally-reproduced refusal, named so the operator can act on it."""

    kind: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.kind}] {self.detail}"


@dataclass(frozen=True)
class AxisCommandParse:
    """The structural read of one ``modal run`` axis command."""

    wrapper_path: str
    entrypoint_name: str
    spec_index: int
    flags: dict[str, str | bool]
    unknown_tokens: tuple[str, ...]


def _flag_to_param(flag: str) -> str:
    """Map a Modal CLI flag to its entrypoint parameter name.

    Modal's CLI derives ``--expected-archive-sha256`` from the parameter
    ``expected_archive_sha256``. This is the inverse of that derivation.
    """

    return flag.removeprefix("--").replace("-", "_")


def find_entrypoint_spec(command: list[str]) -> int:
    """Index of the ``FILE::ENTRYPOINT`` token, or -1 when absent.

    A bare ``FILE`` (no ``::``) is deliberately NOT matched: that is the
    ambiguous form that fails at dispatch when the app exposes more than one
    local entrypoint, and detecting it is one of this module's jobs.
    """

    for index, token in enumerate(command):
        if ENTRYPOINT_SPEC_SEPARATOR in token and token.endswith(".py") is False:
            head, _, tail = token.partition(ENTRYPOINT_SPEC_SEPARATOR)
            if head.endswith(".py") and tail:
                return index
    return -1


def _find_bare_wrapper_index(command: list[str]) -> int:
    """Index of a bare ``*.py`` positional target (the ambiguous form)."""

    for index, token in enumerate(command):
        if token.endswith(".py") and ENTRYPOINT_SPEC_SEPARATOR not in token:
            return index
    return -1


def load_wrapper_module(wrapper_path: str | Path, *, repo_root: Path = REPO_ROOT) -> Any:
    """Import a Modal wrapper file without building anything.

    Modal image definitions are declarative; only ``run``/``.remote()`` triggers
    a build. Measured import cost for the auth-eval wrappers: ~0.10 s.
    """

    path = Path(wrapper_path)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Modal wrapper not found: {path}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_name = f"_tac_preflight_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load Modal wrapper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_axis_command(
    command: list[str], *, repo_root: Path = REPO_ROOT
) -> tuple[AxisCommandParse | None, list[PreflightRefusal]]:
    """Split one axis command into wrapper spec + forwarded flags.

    Bool-vs-value flags are decided from the entrypoint's real signature, not
    from a heuristic about the next token.
    """

    refusals: list[PreflightRefusal] = []

    spec_index = find_entrypoint_spec(command)
    if spec_index < 0:
        bare_index = _find_bare_wrapper_index(command)
        if bare_index < 0:
            refusals.append(
                PreflightRefusal(
                    "ENTRYPOINT_SPEC_MISSING",
                    "no FILE::ENTRYPOINT (and no *.py target) in the command; "
                    f"cannot preflight: {' '.join(command)}",
                )
            )
            return None, refusals
        refusals.append(
            PreflightRefusal(
                "ENTRYPOINT_SPEC_BARE",
                f"target {command[bare_index]!r} names no entrypoint. "
                "'modal run <file>' only auto-selects when the app exposes "
                "exactly one local entrypoint; name it explicitly as "
                f"'{command[bare_index]}::<entrypoint>'.",
            )
        )
        return None, refusals

    wrapper_path, _, entrypoint_name = command[spec_index].partition(
        ENTRYPOINT_SPEC_SEPARATOR
    )

    try:
        module = load_wrapper_module(wrapper_path, repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001 - report, never swallow
        refusals.append(
            PreflightRefusal(
                "WRAPPER_IMPORT_FAILED",
                f"{wrapper_path}: {type(exc).__name__}: {exc}",
            )
        )
        return None, refusals

    entrypoint = getattr(module, entrypoint_name, None)
    if entrypoint is None:
        refusals.append(
            PreflightRefusal(
                "ENTRYPOINT_MISSING",
                f"{wrapper_path} defines no attribute {entrypoint_name!r}",
            )
        )
        return None, refusals

    app = getattr(module, "app", None)
    registered = dict(getattr(app, "registered_entrypoints", {}) or {})
    if registered and entrypoint_name not in registered:
        refusals.append(
            PreflightRefusal(
                "ENTRYPOINT_NOT_REGISTERED",
                f"{entrypoint_name!r} is not a registered local entrypoint of "
                f"{wrapper_path} (registered: {sorted(registered)})",
            )
        )
        return None, refusals

    signature = _entrypoint_signature(entrypoint)
    if signature is None:
        # Never turn "I could not read the signature" into "every flag is
        # invalid" -- that is a false-refusal storm wearing a real gate's
        # clothes. Report the blind spot and check nothing flag-shaped.
        refusals.append(
            PreflightRefusal(
                "UNCHECKED_ENTRYPOINT_SIGNATURE",
                f"could not resolve the signature of {wrapper_path}::"
                f"{entrypoint_name}; forwarded flags are NOT validated.",
            )
        )
        return (
            AxisCommandParse(
                wrapper_path=wrapper_path,
                entrypoint_name=entrypoint_name,
                spec_index=spec_index,
                flags={},
                unknown_tokens=(),
            ),
            refusals,
        )
    params = signature.parameters

    flags: dict[str, str | bool] = {}
    unknown: list[str] = []
    index = spec_index + 1
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = _flag_to_param(token)
        param = params.get(name)
        if param is None:
            unknown.append(token)
            index += 1
            continue
        if param.annotation is bool or isinstance(param.default, bool):
            flags[name] = True
            index += 1
            continue
        if index + 1 < len(command):
            flags[name] = command[index + 1]
            index += 2
        else:
            unknown.append(token)
            index += 1

    for token in unknown:
        refusals.append(
            PreflightRefusal(
                "UNKNOWN_FLAG",
                f"{token} is not a parameter of {wrapper_path}::{entrypoint_name} "
                "(dead-flag class: grep the signature before emitting a flag)",
            )
        )

    parse = AxisCommandParse(
        wrapper_path=wrapper_path,
        entrypoint_name=entrypoint_name,
        spec_index=spec_index,
        flags=flags,
        unknown_tokens=tuple(unknown),
    )
    return parse, refusals


def _is_varargs_only(signature: inspect.Signature) -> bool:
    """True when a signature carries no named parameters we can check against.

    ``modal.app.LocalEntrypoint`` is itself callable through a synchronizer
    wrapper, so ``inspect.signature(entrypoint)`` SUCCEEDS and returns
    ``(*args, **kwargs)``. Accepting that silently made every real flag look
    unknown -- caught by the positive control, 2026-08-10.
    """

    return all(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )


def _entrypoint_signature(entrypoint: Any) -> inspect.Signature | None:
    """Signature of a Modal local entrypoint, unwrapping Modal's wrapper object.

    The UNDECORATED function is authoritative: Modal stores it at
    ``LocalEntrypoint.info.raw_f``. It is tried FIRST, before the callable
    wrapper, because the wrapper's own signature is varargs-only and would
    otherwise shadow the real parameter names.
    """

    info = getattr(entrypoint, "info", None)
    candidates = (
        getattr(info, "raw_f", None),
        getattr(entrypoint, "raw_f", None),
        getattr(entrypoint, "__wrapped__", None),
        entrypoint,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            signature = inspect.signature(candidate)
        except (TypeError, ValueError):
            continue
        if _is_varargs_only(signature):
            continue
        return signature
    return None


def check_runtime_tree_expectation(
    module: Any, parse: AxisCommandParse, *, repo_root: Path = REPO_ROOT
) -> list[PreflightRefusal]:
    """Call the wrapper's OWN runtime-tree validator, locally.

    This is the guard that cost a five-minute image build on 2026-08-10: the CPU
    wrapper structurally refuses a concrete tree hash on the uploaded
    ``--submission-dir`` axis (the r9m deadlock), and the dispatcher was
    resolving ``auto`` into one.
    """

    validator = getattr(module, "_validate_uploaded_runtime_tree_expectation", None)
    if validator is None:
        return [
            PreflightRefusal(
                "UNCHECKED_RUNTIME_TREE",
                f"{parse.wrapper_path} exposes no "
                "_validate_uploaded_runtime_tree_expectation; this axis is "
                "NOT preflighted for runtime-tree custody.",
            )
        ]

    submission_dir = parse.flags.get("submission_dir")
    if not submission_dir:
        return []
    submission_path = Path(str(submission_dir))
    if not submission_path.is_absolute():
        submission_path = repo_root / submission_path

    try:
        validator(
            expected_runtime_tree_sha256=str(
                parse.flags.get("expected_runtime_tree_sha256", "") or ""
            ),
            submission_dir_path=submission_path,
            inflate_sh_rel=str(parse.flags.get("inflate_sh", "") or ""),
        )
    except SystemExit as exc:
        return [PreflightRefusal("RUNTIME_TREE_EXPECTATION", str(exc))]
    except Exception as exc:  # noqa: BLE001 - a validator crash is a refusal too
        return [
            PreflightRefusal(
                "RUNTIME_TREE_VALIDATOR_ERROR",
                f"{type(exc).__name__}: {exc}",
            )
        ]
    return []


def preflight_axis_command(
    command: list[str], *, repo_root: Path = REPO_ROOT
) -> list[PreflightRefusal]:
    """Reproduce, locally, every wrapper guard we can reach for one axis."""

    parse, refusals = parse_axis_command(command, repo_root=repo_root)
    if parse is None:
        return refusals

    module = load_wrapper_module(parse.wrapper_path, repo_root=repo_root)
    refusals.extend(check_runtime_tree_expectation(module, parse, repo_root=repo_root))
    return refusals


def preflight_axis_commands(
    commands: dict[str, list[str]], *, repo_root: Path = REPO_ROOT
) -> dict[str, list[PreflightRefusal]]:
    """Preflight every axis of a paired plan. Empty lists mean clean."""

    return {
        axis: preflight_axis_command(command, repo_root=repo_root)
        for axis, command in commands.items()
    }


__all__ = [
    "ENTRYPOINT_SPEC_SEPARATOR",
    "AxisCommandParse",
    "PreflightRefusal",
    "check_runtime_tree_expectation",
    "find_entrypoint_spec",
    "load_wrapper_module",
    "parse_axis_command",
    "preflight_axis_command",
    "preflight_axis_commands",
]
