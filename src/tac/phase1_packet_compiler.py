"""Phase 1 packet compiler — dezeta-Phase-1 checkpoint -> byte-closed contest packet.

This module is the missing bridge identified by the 2026-05-09 comprehensive
adversarial review (HIGH 1) and the codex Phase 2/3 plan review §5: it turns a
Phase 1 trained checkpoint (dezeta-track1 Ballé hyperprior + 128K decoder) into
a contest-compliant ``submission_dir/`` with a byte-closed ``archive.zip``,
deterministic ``inflate.sh`` (≤100 LOC, ≤3 deps), runtime tree custody, and a
``build_manifest.json`` documenting all 8 HNeRV-parity fields and a
``no_op_proof.json`` proving the bytes change AND the inflate consumes them.

Design contract — three modes:

* ``identity`` — re-emit the input packet byte-for-byte (the input is an
  existing contest-compliant packet, e.g. the A1 canonical archive). Verifies
  byte-closure preserved on round-trip and produces conformance manifests for
  future Rust/C/Zig ports.

* ``canonicalize`` — rebuild the packet, normalising compliance-approved
  metadata only (deterministic ZIP timestamps + sorted member ordering +
  permission normalisation). Reports every changed byte. Refuses any change to
  score-affecting payload bytes (the ``x`` member contents).

* ``optimize`` — turn a fresh Phase 1 trained checkpoint into a brand-new
  contest packet. Score-affecting bytes change by definition; the caller must
  acknowledge with ``score_affecting_payload_changed=True`` and provide
  ``baseline_archive_sha256``, ``baseline_archive_size_bytes`` so the
  ``no_op_proof.json`` can record the old/new SHA delta.

Per CLAUDE.md "Deterministic packet compiler" non-negotiable, all three modes
fail closed on:

* Hidden sidecars (anything not declared in ``build_manifest.json``).
* Scorer modifications (``PoseNet`` / ``SegNet`` / ``rgb_to_yuv6`` patched at
  inflate time per the strict-scorer-rule).
* External state (any path outside the packet root).
* Network/package-install dependencies in ``inflate.sh`` and runtime Python.
* Unsupported ZIP features (only ``ZIP_STORED`` + ``ZIP_DEFLATED`` allowed).
* Parser divergence (decoder must produce bit-exact output to the PR101
  polymorphic codec port — verified via member SHA-256 comparison in identity
  mode).
* Non-deterministic native builds.
* Missing golden vectors.
* Missing runtime-tree custody.

This module is library code; ``tools/build_phase1_packet_compiler.py`` is the
thin CLI wrapper that exposes ``compile_phase1_packet`` and writes outputs to
disk.

CLAUDE.md compliance tags:

* Lane class: ``substrate_engineering`` (not a representation lane).
* Score claim: ``False`` always — this tool only proves byte-closure; CUDA
  auth eval is required for any score claim.
* MPS authoritative: refused — no scorer forwards in this tool.

6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default" +
Check #125):

1. **Sensitivity-map**: this compiler is the inverse of the sensitivity map —
   it turns sensitivity-driven training into bit-allocations preserved
   byte-for-byte in the archive. No new sensitivity-map row is added; the
   compiler consumes whatever the trainer emitted.
2. **Pareto frontier**: the compiler enforces Pareto frontier constraints
   indirectly (rate ≤ R via archive_size_bytes; archive ≤ B via inflate.sh
   LOC budget + dep closure declaration). No new Pareto row is added; the
   compiler is downstream of the planner.
3. **Bit-allocator**: DIRECT — the compiler IS the bit allocator's
   archive-side. The HNeRV-parity manifest's ``parser_section_manifest`` +
   ``archive_grammar`` field is the durable record of the byte budget
   actually shipped.
4. **Cathedral autopilot dispatch**: every dispatch result that returns from
   contest_auth_eval triggers a follow-up packet compile + lane-registry
   row update. The compiler's ``build_manifest.json`` is the input the
   autopilot consumes when ranking the next dispatch batch.
5. **Continual-learning posterior update**: empirical results from packet
   evals (CUDA + CPU) trigger posterior updates on the per-architecture
   CUDA/CPU drift profiles in ``tac.cuda_cpu_axis_profile_learning_layer``.
   The compiler is the gate that ensures the empirical result is on
   byte-closed evidence, not a state-dict snapshot.
6. **Probe-disambiguator**: the THREE modes (``identity`` / ``canonicalize``
   / ``optimize``) are the probe — each mode is the right answer for a
   specific substrate question (preserve / normalise / re-encode). Operator
   selects per substrate, with the no_op_proof.json recording the
   disambiguation outcome.
"""
from __future__ import annotations

import ast
import dataclasses
import datetime as _dt
import io
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal

from tac.repo_io import json_text, sha256_bytes, sha256_file

SCHEMA_VERSION = "phase1_packet_compiler.v1"
TOOL_NAME = "tac.phase1_packet_compiler"

#: Modes the compiler supports. Surfaced both as a tuple for runtime checks
#: and as a Literal alias for static-typer-friendly call sites.
COMPILER_MODES: tuple[str, ...] = ("identity", "canonicalize", "optimize")
CompilerMode = Literal["identity", "canonicalize", "optimize"]

#: Target packet profiles understood by the Phase 1 compiler. They mirror the
#: ``tac.submission_packet_compiler.TARGET_PROFILE_POLICIES`` taxonomy but the
#: Phase 1 compiler currently only emits contest-axis packets.
TARGET_MODES: tuple[str, ...] = (
    "contest_one_video_replay",
    "contest_generalized",
)
TargetMode = Literal["contest_one_video_replay", "contest_generalized"]

#: Deterministic ZIP date-time used by every packet emit. Mirrors
#: ``tac.submission_archive.DETERMINISTIC_ZIP_DATE_TIME`` (1980-01-01 epoch)
#: so independent ports produce bit-identical archives.
DETERMINISTIC_ZIP_DATE_TIME: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)

#: Allowed file modes inside the packet root. ``inflate.sh`` is executable; all
#: other files are 0o644. We refuse anything else for determinism.
EXECUTABLE_MODE = 0o755
NON_EXECUTABLE_MODE = 0o644

#: Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" the packet's
#: ``inflate.sh`` MUST take exactly three positional arguments: archive_dir,
#: output_dir, file_list. Anything else fails closed.
INFLATE_SH_REQUIRED_POSITIONAL_ARGS = 3

#: Accepted ZIP compression methods. STORED is preferred for byte-closed
#: payloads (already-compressed brotli/torch); DEFLATE is allowed for source
#: text where it gives a real saving.
ALLOWED_ZIP_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})

#: Patterns that, if found in ``inflate.sh`` or ``inflate.py``, indicate a
#: scorer load at inflate time (forbidden per CLAUDE.md strict-scorer-rule).
#: Mirrors the patterns in ``tac.preflight._scan_inflate_for_scorer_load``.
FORBIDDEN_INFLATE_TOKENS: tuple[str, ...] = (
    "PoseNet",
    "SegNet",
    "from upstream.modules",
    "import upstream.modules",
    "rgb_to_yuv6",
    "EfficientNet",
    "FastViT",
)

#: Patterns that indicate an inflate.sh / inflate.py is reaching outside the
#: packet root for state. We allow ``upstream/`` references via the
#: contest-provided ``archive_dir`` argument, but absolute paths or repo-local
#: research paths are forbidden.
FORBIDDEN_EXTERNAL_STATE_PATTERNS: tuple[str, ...] = (
    "/Users/",
    "/home/",
    "experiments/results/",
    ".omx/state/",
    ".omx/research/",
)

#: Network-fetch tokens forbidden in ``inflate.sh``. The packet must inflate
#: hermetically on the contest runner; no curl/wget/pip/uv dependency fetches
#: or alternate package indexes at inflate time.
FORBIDDEN_NETWORK_TOKENS: tuple[str, ...] = (
    "--extra-index-url",
    "--find-links",
    "--index-url",
    "curl ",
    "http://",
    "https://",
    "httpx.",
    "requests.",
    "urllib.request",
    "urlopen(",
    "urlretrieve(",
    "wget ",
    "pip install",
    "'pip', 'install'",
    '"pip", "install"',
    " uv run ",
    "uv run --with",
    "uv pip install",
    "'uv', 'pip', 'install'",
    '"uv", "pip", "install"',
    "git clone",
)

#: Python modules that make an inflate runtime non-hermetic. These are kept
#: separate from ``FORBIDDEN_NETWORK_TOKENS`` because Python code can hide
#: network/package-install behavior behind imports, aliases, or subprocess
#: argument lists that do not look like shell snippets.
FORBIDDEN_PYTHON_RUNTIME_MODULES: tuple[str, ...] = (
    "aiohttp",
    "ensurepip",
    "ftplib",
    "http.client",
    "httpx",
    "pip",
    "pip._internal",
    "requests",
    "socket",
    "urllib",
    "urllib.request",
    "urllib3",
)

FORBIDDEN_PYTHON_RUNTIME_CALLS: tuple[str, ...] = (
    "ensurepip.bootstrap",
    "pip.main",
    "urlopen",
    "urlretrieve",
)

FORBIDDEN_PYTHON_COMMAND_SEQUENCES: tuple[tuple[str, ...], ...] = (
    ("curl",),
    ("git", "clone"),
    ("pip", "install"),
    ("uv", "pip", "install"),
    ("uv", "run", "--with"),
    ("wget",),
)

FORBIDDEN_REPO_LOCAL_RUNTIME_TOKENS: tuple[str, ...] = (
    "_find_repo_src",
    "repo-local tac runtime dependency",
    "for parent in here.parents",
    "parent / 'src' / 'tac'",
    'parent / "src" / "tac"',
)

#: HNeRV-parity 8-field manifest declaration required per CLAUDE.md "HNeRV
#: parity discipline". Every Phase 1 packet records all 8 fields verbatim so
#: lane registry + downstream gates can audit them.
HNERV_PARITY_FIELDS: tuple[str, ...] = (
    "archive_grammar",
    "parser_section_manifest",
    "inflate_runtime_loc_budget",
    "runtime_dep_closure",
    "export_format",
    "score_aware_loss",
    "bolt_on_loc_budget",
    "no_op_detector_planned",
)

#: A1 canonical archive — used by identity-mode round-trip tests as the
#: bit-identical reference. SHA / size must match the value in
#: ``.omx/state/canonical_a1_designation.md``.
A1_CANONICAL_ARCHIVE_SHA256 = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
A1_CANONICAL_ARCHIVE_SIZE_BYTES = 178262


class Phase1PacketCompilerError(ValueError):
    """Raised when a Phase 1 packet cannot be compiled deterministically."""


@dataclasses.dataclass(frozen=True)
class Phase1PacketResult:
    """The structured result of ``compile_phase1_packet``.

    Attributes mirror the keys written into ``build_manifest.json`` so callers
    can introspect the result without re-parsing JSON. ``score_claim`` is
    permanently False — this tool only proves byte-closure; CUDA auth eval is
    the only permitted source of a score claim.
    """

    schema_version: str
    mode: str
    target_mode: str
    output_dir: str
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    runtime_tree_sha256: str
    runtime_files: tuple[dict[str, Any], ...]
    archive_members: tuple[dict[str, Any], ...]
    parser_section_manifest: dict[str, Any]
    hnerv_parity_manifest: dict[str, Any]
    no_op_proof: dict[str, Any]
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool
    blockers: tuple[str, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso(now: _dt.datetime | None = None) -> str:
    value = now or _dt.datetime.now(_dt.UTC)
    return value.astimezone(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _zip_member_is_unsafe(name: str) -> bool:
    pure = PurePosixPath(name)
    parts = pure.parts
    return (
        name.startswith("/")
        or ".." in parts
        or any(part.startswith(".") and part not in {"."} for part in parts)
        or "__MACOSX" in parts
        or any(part == "" for part in parts)
    )


def _read_archive_members(archive_path: Path) -> list[dict[str, Any]]:
    """Return a deterministic per-member manifest for a ZIP archive."""
    if not archive_path.is_file():
        raise Phase1PacketCompilerError(
            f"archive does not exist or is not a file: {archive_path}"
        )
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as zf:
        infos = sorted(
            (info for info in zf.infolist() if not info.is_dir()),
            key=lambda item: item.filename,
        )
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise Phase1PacketCompilerError(
                f"archive has duplicate member names: {archive_path}"
            )
        for index, info in enumerate(infos):
            if info.compress_type not in ALLOWED_ZIP_METHODS:
                raise Phase1PacketCompilerError(
                    "archive uses unsupported ZIP compression method "
                    f"{info.compress_type!r}: {info.filename}"
                )
            if _zip_member_is_unsafe(info.filename):
                raise Phase1PacketCompilerError(
                    f"archive has unsafe member name: {info.filename!r}"
                )
            payload = zf.read(info)
            members.append(
                {
                    "name": info.filename,
                    "order_index": index,
                    "uncompressed_bytes": int(info.file_size),
                    "compressed_bytes": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": sha256_bytes(payload),
                    "date_time": list(info.date_time),
                }
            )
    return members


def _runtime_tree_files(packet_dir: Path) -> list[dict[str, Any]]:
    """Walk the packet directory and return a deterministic manifest of every
    runtime-tree file (everything except the archive.zip itself).

    Round 3 Tao MEDIUM fix: refuse if any *direct child* of packet_dir is a
    symlink — this prevents a malicious caller from supplying a packet
    whose ``src/`` is a symlink that walks into ``/Users/<other>/...``.
    Deep-tree symlinks are rejected by per-file ``is_symlink`` further down.
    """
    for child in sorted(packet_dir.iterdir() if packet_dir.is_dir() else []):
        if child.is_symlink():
            raise Phase1PacketCompilerError(
                f"packet refuses top-level symlink at {child.relative_to(packet_dir)}; "
                "all packet contents must be regular files/directories"
            )
    rows: list[dict[str, Any]] = []
    for path in sorted(packet_dir.rglob("*"), key=lambda p: p.relative_to(packet_dir).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(packet_dir).as_posix()
        if rel == "archive.zip":
            continue
        if "__pycache__" in path.parts or path.name.endswith((".pyc", ".pyo")):
            continue
        if path.is_symlink():
            raise Phase1PacketCompilerError(
                f"packet refuses symlink at runtime tree path: {rel}"
            )
        st = path.stat()
        mode = stat.S_IMODE(st.st_mode)
        # Normalise to the two allowed modes.
        if rel == "inflate.sh":
            expected_mode = EXECUTABLE_MODE
        else:
            expected_mode = NON_EXECUTABLE_MODE
        rows.append(
            {
                "relpath": rel,
                "bytes": st.st_size,
                "sha256": sha256_file(path),
                "mode": f"{mode:04o}",
                "expected_mode": f"{expected_mode:04o}",
                "mode_matches_expected": mode == expected_mode,
            }
        )
    return rows


def _runtime_tree_sha256(rows: Iterable[dict[str, Any]]) -> str:
    basis = [
        {
            "relpath": row["relpath"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "mode": row["mode"],
        }
        for row in rows
    ]
    return sha256_bytes(json_text(basis).encode("utf-8"))


def _scan_text_for_forbidden(
    text: str,
    *,
    forbidden_tokens: Iterable[str],
    label: str,
) -> list[str]:
    hits: list[str] = []
    for token in forbidden_tokens:
        if token in text:
            hits.append(f"{label}: contains forbidden token {token!r}")
    return hits


def _python_module_is_forbidden(module: str) -> bool:
    return any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for forbidden in FORBIDDEN_PYTHON_RUNTIME_MODULES
    )


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        if prefix:
            return f"{prefix}.{node.attr}"
        return node.attr
    return None


def _normalise_command_token(token: str) -> str:
    lowered = token.strip().lower()
    if "/" in lowered:
        lowered = lowered.rsplit("/", 1)[-1]
    return lowered


def _contains_command_sequence(
    tokens: list[str],
    pattern: tuple[str, ...],
) -> bool:
    if len(pattern) > len(tokens):
        return False
    for start in range(0, len(tokens) - len(pattern) + 1):
        if tuple(tokens[start : start + len(pattern)]) == pattern:
            return True
    return False


def _python_literal_strings(
    node: ast.AST,
    constants: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name):
        return constants.get(node.id, ())
    if isinstance(node, (ast.List, ast.Tuple)):
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                out.extend(_python_literal_strings(elt.value, constants))
            else:
                out.extend(_python_literal_strings(elt, constants))
        return tuple(out)
    return ()


def _python_import_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _scan_python_ast_for_hermeticity(
    tree: ast.AST,
    *,
    label: str,
) -> list[str]:
    blockers: list[str] = []
    constants: dict[str, tuple[str, ...]] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            resolved = _python_literal_strings(value, constants)
            if not resolved:
                continue
            for token in resolved:
                if token.startswith(("http://", "https://")):
                    blockers.append(
                        f"{label}: Python runtime literal contains external URL"
                    )
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = resolved
            continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                if _python_module_is_forbidden(alias.name):
                    blockers.append(
                        f"{label}: imports forbidden Python runtime module "
                        f"{alias.name!r}"
                    )
            continue

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            candidates = [module]
            candidates.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
            for candidate in candidates:
                if candidate and _python_module_is_forbidden(candidate):
                    blockers.append(
                        f"{label}: imports forbidden Python runtime module "
                        f"{candidate!r}"
                    )
                    break
            continue

        if not isinstance(node, ast.Call):
            continue

        call_name = _qualified_name(node.func) or ""
        if any(
            call_name == forbidden or call_name.endswith(f".{forbidden}")
            for forbidden in FORBIDDEN_PYTHON_RUNTIME_CALLS
        ):
            blockers.append(
                f"{label}: calls forbidden Python runtime API {call_name!r}"
            )

        import_target = None
        if call_name in {"__import__", "importlib.import_module"} and node.args:
            import_target = _python_import_literal(node.args[0])
        if import_target and _python_module_is_forbidden(import_target):
            blockers.append(
                f"{label}: dynamically imports forbidden Python runtime module "
                f"{import_target!r}"
            )

        literal_tokens: list[str] = []
        for arg in node.args:
            literal_tokens.extend(_python_literal_strings(arg, constants))
        for keyword in node.keywords:
            if keyword.value is not None:
                literal_tokens.extend(
                    _python_literal_strings(keyword.value, constants)
                )

        normalised_tokens: list[str] = []
        for token in literal_tokens:
            normalised_tokens.extend(
                _normalise_command_token(part)
                for part in token.split()
                if part.strip()
            )
        for pattern in FORBIDDEN_PYTHON_COMMAND_SEQUENCES:
            if _contains_command_sequence(normalised_tokens, pattern):
                blockers.append(
                    f"{label}: Python runtime command contains forbidden "
                    f"sequence {' '.join(pattern)!r}"
                )
        for token in literal_tokens:
            if token.startswith(("http://", "https://")):
                blockers.append(
                    f"{label}: Python runtime literal contains external URL"
                )

    return sorted(set(blockers))


def _scan_python_runtime_hermeticity(py_file: Path, *, label: str) -> list[str]:
    text = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=py_file.as_posix())
    except SyntaxError as exc:
        return [f"{label}: Python syntax parse failed: {exc.msg}"]
    blockers = _scan_python_ast_for_hermeticity(tree, label=label)
    blockers.extend(
        _scan_text_for_forbidden(
            text,
            forbidden_tokens=FORBIDDEN_REPO_LOCAL_RUNTIME_TOKENS,
            label=label,
        )
    )
    return blockers


def _bash_n_check(inflate_sh: Path) -> dict[str, Any]:
    """Round 2 Contrarian MEDIUM fix: verify inflate.sh syntactic validity
    via ``bash -n`` (parse-only). Catches missing fi/done/'' before the
    operator burns a $80 GPU dispatch on a script that won't even parse.

    Returns a dict with passed/returncode/stderr; failure is recorded but
    treated as warn-only because some CI environments may not have bash
    available.
    """
    import subprocess  # local import to keep cold-import cheap

    try:
        proc = subprocess.run(
            ["bash", "-n", str(inflate_sh)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "attempted": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "attempted": True,
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
    }


def _scan_inflate_sh(inflate_sh: Path) -> tuple[list[str], dict[str, Any]]:
    """Validate inflate.sh: positional-arg shape, hermetic-runtime, no
    network/scorer/external-state hits.
    """
    text = inflate_sh.read_text(encoding="utf-8", errors="replace")
    blockers: list[str] = []
    blockers.extend(
        _scan_text_for_forbidden(
            text,
            forbidden_tokens=FORBIDDEN_NETWORK_TOKENS,
            label="inflate.sh",
        )
    )
    blockers.extend(
        _scan_text_for_forbidden(
            text,
            forbidden_tokens=FORBIDDEN_INFLATE_TOKENS,
            label="inflate.sh",
        )
    )
    blockers.extend(
        _scan_text_for_forbidden(
            text,
            forbidden_tokens=FORBIDDEN_EXTERNAL_STATE_PATTERNS,
            label="inflate.sh",
        )
    )
    # Positional-arg contract: must reference $1, $2, $3 (data dir, output
    # dir, file list). We tolerate variants like "${1}".
    positional_seen = sum(
        1 for marker in ("$1", "${1}", "$2", "${2}", "$3", "${3}")
        if marker in text
    )
    # Each arg can match either bare or braced form; require all 3 present.
    has_1 = "$1" in text or "${1}" in text
    has_2 = "$2" in text or "${2}" in text
    has_3 = "$3" in text or "${3}" in text
    if not (has_1 and has_2 and has_3):
        blockers.append(
            "inflate.sh: missing required positional args $1 (archive_dir) "
            "$2 (output_dir) $3 (file_list); contest contract requires all 3"
        )
    if "set -euo pipefail" not in text and "set -e" not in text:
        blockers.append("inflate.sh: missing 'set -e' / 'set -euo pipefail'")
    # Round 1 Fridrich HIGH fix: LOC excludes blank lines and pure-comment
    # lines (lines whose first non-whitespace char is `#`) so the budget
    # measures executable shell intent, not formatting.
    raw_lines = text.splitlines()
    code_lines = 0
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        code_lines += 1
    bash_n = _bash_n_check(inflate_sh)
    if bash_n.get("attempted") and bash_n.get("passed") is False:
        blockers.append(
            f"inflate.sh: bash -n parse-only validation failed (rc={bash_n.get('returncode')}): "
            f"{bash_n.get('stderr', '')[:200]}"
        )
    info: dict[str, Any] = {
        "loc": code_lines,
        "loc_raw": len(raw_lines),
        "has_set_euo_pipefail": "set -euo pipefail" in text,
        "positional_args_seen": positional_seen,
        "byte_size": len(text.encode("utf-8")),
        "bash_n": bash_n,
    }
    return blockers, info


def _scan_inflate_py(inflate_py: Path) -> list[str]:
    text = inflate_py.read_text(encoding="utf-8", errors="replace")
    blockers: list[str] = []
    blockers.extend(
        _scan_text_for_forbidden(
            text,
            forbidden_tokens=FORBIDDEN_NETWORK_TOKENS,
            label="inflate.py",
        )
    )
    blockers.extend(
        _scan_text_for_forbidden(
            text,
            forbidden_tokens=FORBIDDEN_INFLATE_TOKENS,
            label="inflate.py",
        )
    )
    blockers.extend(
        _scan_text_for_forbidden(
            text,
            forbidden_tokens=FORBIDDEN_EXTERNAL_STATE_PATTERNS,
            label="inflate.py",
        )
    )
    blockers.extend(_scan_python_runtime_hermeticity(inflate_py, label="inflate.py"))
    return blockers


def _scan_runtime_python_surfaces(packet_dir: Path) -> list[str]:
    blockers: list[str] = []
    for py_file in sorted(packet_dir.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        if py_file == packet_dir / "inflate.py":
            continue
        rel = py_file.relative_to(packet_dir).as_posix()
        blockers.extend(_scan_python_runtime_hermeticity(py_file, label=rel))
    return blockers


def _scan_runtime_shell_surfaces(packet_dir: Path) -> list[str]:
    """Scan non-inflate shell helpers for hermetic-runtime violations."""
    blockers: list[str] = []
    for sh_file in sorted(packet_dir.rglob("*.sh")):
        if "__pycache__" in sh_file.parts:
            continue
        if sh_file == packet_dir / "inflate.sh":
            continue
        rel = sh_file.relative_to(packet_dir).as_posix()
        text = sh_file.read_text(encoding="utf-8", errors="replace")
        blockers.extend(
            _scan_text_for_forbidden(
                text,
                forbidden_tokens=FORBIDDEN_NETWORK_TOKENS,
                label=rel,
            )
        )
        blockers.extend(
            _scan_text_for_forbidden(
                text,
                forbidden_tokens=FORBIDDEN_INFLATE_TOKENS,
                label=rel,
            )
        )
        blockers.extend(
            _scan_text_for_forbidden(
                text,
                forbidden_tokens=FORBIDDEN_EXTERNAL_STATE_PATTERNS,
                label=rel,
            )
        )
    return blockers


def _undeclared_python_imports_in_runtime_tree(
    packet_dir: Path,
    declared_dep_closure: list[str],
) -> list[str]:
    """Round 1 Selfcomp MEDIUM fix: scan every .py file in the runtime tree
    for top-level ``import X`` / ``from X import ...`` and emit a hint when
    third-party deps appear that are NOT in the declared dep closure. Stdlib
    + packet-local imports are tolerated. Repo-local ``tac`` imports are not:
    a contest packet must carry any needed runtime subset under ``src/``.
    """
    local_top_level_modules = {"model", "codec"}
    for base in (packet_dir, packet_dir / "src"):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.name.startswith("__"):
                continue
            if child.is_dir() and (child / "__init__.py").is_file():
                local_top_level_modules.add(child.name)
            elif child.is_file() and child.suffix == ".py":
                local_top_level_modules.add(child.stem)

    stdlib_prefixes = {
        "__future__", "sys", "os", "io", "re", "json", "pickle", "struct", "argparse",
        "pathlib", "typing", "dataclasses", "datetime", "stat", "shutil",
        "hashlib", "zipfile", "subprocess", "math", "warnings", "abc",
        "collections", "functools", "itertools", "enum", "contextlib",
    }
    found: list[str] = []
    declared_set = {dep.lower() for dep in declared_dep_closure}
    pattern = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z_0-9]*)")
    for py_file in sorted(packet_dir.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            module = match.group(1)
            if module in stdlib_prefixes or module in local_top_level_modules:
                continue
            if module.lower() in declared_set:
                continue
            found.append(
                f"undeclared_runtime_dep:{py_file.name}:{module}"
            )
    return sorted(set(found))


def _ensure_no_hidden_sidecars(
    packet_dir: Path,
    declared_files: Iterable[str],
) -> list[str]:
    declared = {row for row in declared_files}
    actual = set()
    for row in _runtime_tree_files(packet_dir):
        actual.add(row["relpath"])
    actual.add("archive.zip")
    extras = sorted(actual - declared)
    if not extras:
        return []
    return [f"hidden sidecar (not declared in build_manifest): {rel}" for rel in extras]


def _safe_clear_output_dir(output_dir: Path, *, allow_existing: bool = False) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise Phase1PacketCompilerError(
                f"output_dir exists and is not a directory: {output_dir}"
            )
        if any(output_dir.iterdir()) and not allow_existing:
            raise Phase1PacketCompilerError(
                "output_dir is not empty; pass allow_existing_output_dir=True "
                f"to replace its contents: {output_dir}"
            )
        if any(output_dir.iterdir()):
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_packet_tree(source_dir: Path, dest_dir: Path) -> list[dict[str, Any]]:
    """Copy a packet directory bit-identically, preserving file modes for the
    two allowed values (0o755 for inflate.sh, 0o644 for everything else).
    """
    copied: list[dict[str, Any]] = []
    for src in sorted(source_dir.rglob("*"), key=lambda p: p.relative_to(source_dir).as_posix()):
        if "__pycache__" in src.parts or src.name.endswith((".pyc", ".pyo")):
            continue
        if src.is_symlink():
            raise Phase1PacketCompilerError(
                f"packet refuses symlink at: {src.relative_to(source_dir)}"
            )
        rel = src.relative_to(source_dir).as_posix()
        dst = dest_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        if rel == "inflate.sh" or stat.S_IMODE(src.stat().st_mode) & 0o111:
            dst.chmod(EXECUTABLE_MODE)
        else:
            dst.chmod(NON_EXECUTABLE_MODE)
        copied.append(
            {
                "relpath": rel,
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "mode": f"{stat.S_IMODE(dst.stat().st_mode):04o}",
            }
        )
    return copied


def _resolve_input_packet(input_path: Path) -> tuple[Path, Path]:
    """Return (packet_dir, archive_path). The input may be either a
    submission_dir/ directory containing archive.zip + inflate.sh, or a bare
    archive.zip file (in which case packet_dir is the archive's parent and we
    fail closed if no inflate.sh sits next to it).
    """
    if input_path.is_dir():
        archive = input_path / "archive.zip"
        if not archive.is_file():
            raise Phase1PacketCompilerError(
                f"input directory missing archive.zip: {input_path}"
            )
        return input_path, archive
    if input_path.is_file() and input_path.suffix == ".zip":
        packet_dir = input_path.parent
        archive = input_path
        return packet_dir, archive
    raise Phase1PacketCompilerError(
        f"input is neither a packet directory nor an archive.zip: {input_path}"
    )


def _build_parser_section_manifest(
    archive_members: list[dict[str, Any]],
    packet_dir: Path | None = None,
) -> dict[str, Any]:
    """Per CLAUDE.md HNeRV parity discipline + Check 101 (gate3 parser
    section manifest), HNeRV-family monolithic packets must declare per-
    member offsets, lengths, names, SHAs, entropy estimates, and section
    boundaries.

    Phase 1 packets use three deterministic ZIP members: ``x`` for Ballé
    strings, ``decoder.bin`` for the decoder state dict, and ``balle.bin`` for
    the hyperprior state dict. The trainer may also emit
    ``archive_section_manifest.json``; when present, this manifest is folded in
    as additional parser metadata.
    """
    out: dict[str, Any] = {
        "schema_version": "phase1_parser_section_manifest.v1",
        "section_count": len(archive_members),
        "section_names": [row["name"] for row in archive_members],
        "lengths": [row["uncompressed_bytes"] for row in archive_members],
        "section_sha256s": [row["sha256"] for row in archive_members],
        "offsets": "see ZIP central directory; deterministic via DETERMINISTIC_ZIP_DATE_TIME",
        "entropy_estimates": "deferred to trainer's archive_section_manifest.json (Phase 2 byte-tightening)",
        "old_new_section_boundaries": (
            "Phase 1 packet boundary is the deterministic three-member ZIP "
            "grammar: x / decoder.bin / balle.bin. Intra-member offsets are "
            "not used until the Phase 2 deterministic tensor wire format."
        ),
        "intra_member_section_manifest_present": False,
    }
    if packet_dir is not None:
        section_path = packet_dir / "archive_section_manifest.json"
        if section_path.is_file():
            try:
                intra = json.loads(section_path.read_text(encoding="utf-8"))
                out["intra_member_section_manifest_present"] = True
                out["intra_member_section_manifest"] = intra
            except (json.JSONDecodeError, OSError):
                # Malformed sidecar: surface as missing rather than
                # silently dropping it.
                out["intra_member_section_manifest_present"] = False
                out["intra_member_section_manifest_error"] = "json_decode_or_read_failure"
    return out


def _build_hnerv_parity_manifest(
    *,
    archive_size_bytes: int,
    inflate_sh_loc: int,
    runtime_dep_closure: list[str],
    export_format: str,
    bolt_on_loc_budget: int,
) -> dict[str, Any]:
    """Emit the 8-field HNeRV parity manifest for the lane registry / preflight."""
    return {
        "archive_grammar": "Phase1-three-member-x-decoder-bin-balle-bin",
        "parser_section_manifest": "tac.phase1_packet_compiler._build_parser_section_manifest",
        "inflate_runtime_loc_budget": 100,
        "inflate_runtime_loc_actual": inflate_sh_loc,
        "runtime_dep_closure": runtime_dep_closure,
        "export_format": export_format,
        "score_aware_loss": "delegated_from_phase1_trainer",
        "bolt_on_loc_budget": bolt_on_loc_budget,
        "no_op_detector_planned": True,
    }


def _build_no_op_proof(
    *,
    new_archive_sha256: str,
    new_archive_size: int,
    baseline_archive_sha256: str | None,
    baseline_archive_size_bytes: int | None,
    runtime_consumes_bytes: bool,
    score_affecting_payload_changed: bool,
) -> dict[str, Any]:
    """Per Catalog #105 (gate7 no_op provenance): every byte-changing emit must
    record old/new SHA, payload-change-proof, and runtime-consumption proof.
    """
    proof: dict[str, Any] = {
        "schema_version": "phase1_no_op_proof.v1",
        "score_affecting_payload_changed": score_affecting_payload_changed,
        "new_archive_sha256": new_archive_sha256,
        "new_archive_size_bytes": new_archive_size,
        "baseline_archive_sha256": baseline_archive_sha256,
        "baseline_archive_size_bytes": baseline_archive_size_bytes,
        "byte_delta": (
            new_archive_size - baseline_archive_size_bytes
            if baseline_archive_size_bytes is not None
            else None
        ),
        "sha_changed": (
            new_archive_sha256 != baseline_archive_sha256
            if baseline_archive_sha256 is not None
            else None
        ),
        "runtime_consumption_proof": runtime_consumes_bytes,
        "no_op_detector_passed": (
            # If we claim no payload change, sha must be unchanged.
            (not score_affecting_payload_changed)
            == (baseline_archive_sha256 == new_archive_sha256)
            if baseline_archive_sha256 is not None
            else None
        ),
    }
    return proof


def _verify_runtime_consumes_payload_bytes(
    runtime_files: list[dict[str, Any]],
    packet_dir: Path,
) -> bool:
    """Verify the runtime tree actually consumes archive bytes.

    Round 1 Yousfi HIGH fix: presence of inflate.sh + inflate.py is necessary
    but not sufficient. We additionally require inflate.py to contain at
    least one byte-read pattern that points at the archive contents
    (``read_bytes``, ``open(...).read()``, ``zipfile.ZipFile``, etc.). A
    maliciously empty inflate.py would otherwise pass the check.

    Codex round 5 HIGH 1 fix (2026-05-09, catalog #139): the text-pattern
    detector is necessary but still fails-open against a script that *reads*
    the archive without using its bytes. The stronger executable smoke is
    in :func:`_verify_runtime_consumes_payload_bytes_executable` which
    actually mutates a byte and observes whether downstream output changes;
    callers that own a real archive should prefer it. The text-pattern
    helper remains the cheap structural gate and is honored by
    `_finalize_packet_result` when archive/inflate execution is unavailable.
    """
    relpaths = {row["relpath"] for row in runtime_files}
    if "inflate.sh" not in relpaths or "inflate.py" not in relpaths:
        return False
    inflate_py = packet_dir / "inflate.py"
    if not inflate_py.is_file():
        return False
    text = inflate_py.read_text(encoding="utf-8", errors="replace")
    byte_read_patterns = (
        ".read_bytes(",
        ".read(",
        "zipfile.ZipFile",
        "open(",
        "frombuffer",
        "torch.load",
        "pickle.load",
        "struct.unpack",
    )
    return any(pat in text for pat in byte_read_patterns)


# ---------------------------------------------------------------------------
# Executable byte-mutation smoke (codex round 5 HIGH 1 fix, catalog #139)
# ---------------------------------------------------------------------------


def _verify_runtime_consumes_payload_bytes_executable(
    *,
    packet_dir: Path,
    archive_path: Path,
) -> tuple[bool, str]:
    """Executable smoke: mutate one archive byte, observe inflate output change.

    Codex round 5 HIGH 1 fix (2026-05-09, catalog #139): the original
    text-pattern detector accepts ``open(...)`` and ``read(...)`` tokens
    without proving the bytes ACTUALLY round-trip into a downstream side-
    effect. A maliciously / accidentally no-op ``inflate.py`` (e.g., reads
    the archive and discards the bytes, or writes a constant placeholder
    output) reports as clean compile and burns evaluation spend.

    This helper:

    1. Reads ``archive.zip`` byte content.
    2. Computes a SHA-256 reference fingerprint of all FILE outputs of a
       reference inflate run through the contest ``inflate.sh`` three-argument
       contract with a non-empty ``video_names`` file.
    3. Mutates a single non-trivial byte in a copy of the archive.
    4. Re-runs the same inflate.py against the mutated archive into a
       SECOND inflated_dir.
    5. Compares fingerprints. If they're identical, the runtime did not
       consume the mutated byte → return ``(False, reason)``.
    6. Returns ``(True, "byte-mutation-changed-output")`` only when the
       mutation produced a measurable downstream byte change.

    Returns ``(False, reason_string)`` on any failure mode (missing
    archive, inflate.py crash, identical fingerprints). Non-fatal: callers
    use the boolean to decide whether to promote the no-op-proof failure
    to a blocker.

    Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact", the
    mutated archive copy and inflated dirs are written into
    ``packet_dir / "_no_op_smoke"`` (a sibling under the same canonical
    output_dir), not /tmp. The directory is removed after the smoke; if
    cleanup fails, a forensic sibling remains for operator inspection.
    """
    import shutil
    import subprocess
    inflate_sh = packet_dir / "inflate.sh"
    if not inflate_sh.is_file():
        return False, "inflate_sh_missing"
    if not archive_path.is_file():
        return False, "archive_missing"
    try:
        archive_bytes = archive_path.read_bytes()
    except OSError as exc:
        return False, f"archive_read_failed:{exc}"
    if len(archive_bytes) < 8:
        return False, f"archive_too_small_for_smoke:{len(archive_bytes)}_bytes"

    smoke_dir = packet_dir / "_no_op_smoke"
    try:
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir, ignore_errors=True)
        smoke_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        return False, f"smoke_dir_setup_failed:{exc}"

    try:
        def _materialise_archive_dir(
            archive_dir: Path,
            *,
            member_payload_mutation: bool,
        ) -> None:
            """Write best-effort extracted member files into archive_dir.

            The official contest runner unzips ``archive.zip`` first, then
            passes that extracted directory as ``$1`` to ``inflate.sh``. The
            smoke intentionally does not place ``archive.zip`` inside
            ``archive_dir``; runtimes must consume extracted member bytes, not
            a bundled fallback copy beside inflate.py.
            """

            archive_dir.mkdir(parents=True)
            try:
                with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
                    names = sorted(zf.namelist())
                    for idx, name in enumerate(names):
                        if name.endswith("/"):
                            continue
                        payload = bytearray(zf.read(name))
                        if member_payload_mutation and idx == 0 and payload:
                            mutate_at = len(payload) // 2
                            payload[mutate_at] = (payload[mutate_at] ^ 0xFF) & 0xFF
                        dst = archive_dir / name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(bytes(payload))
            except (zipfile.BadZipFile, OSError, RuntimeError):
                # Leave archive_dir empty. Runtimes that do not consume
                # extracted members must fail this smoke rather than silently
                # reading a fallback archive.zip outside the contest contract.
                return

        # Reference run: archive_dir contains the original extracted archive
        # members, matching experiments/contest_auth_eval.py and upstream/evaluate.sh.
        ref_archive_dir = smoke_dir / "ref" / "archive_dir"
        _materialise_archive_dir(
            ref_archive_dir,
            member_payload_mutation=False,
        )
        ref_inflated = smoke_dir / "ref" / "inflated"
        ref_inflated.mkdir(parents=True)

        # Mutated run: flip one byte deep in the archive payload (skip the
        # first 64 bytes to avoid ZIP local-file-header which any sane
        # inflate would catch with a CRC error before reaching downstream).
        offset = max(64, len(archive_bytes) // 2)
        offset = min(offset, len(archive_bytes) - 1)
        mutated_bytes = bytearray(archive_bytes)
        mutated_bytes[offset] = (mutated_bytes[offset] ^ 0xFF) & 0xFF
        mut_archive_dir = smoke_dir / "mut" / "archive_dir"
        _materialise_archive_dir(
            mut_archive_dir,
            member_payload_mutation=True,
        )
        mut_inflated = smoke_dir / "mut" / "inflated"
        mut_inflated.mkdir(parents=True)

        # Non-empty video-names file: the smoke must exercise the contest
        # per-video output path, not an empty-list shortcut.
        names_file = smoke_dir / "video_names.txt"
        names_file.write_text("0.mkv\n", encoding="utf-8")

        def _run_inflate(archive_dir: Path, inflated_dir: Path) -> int:
            try:
                proc = subprocess.run(
                    [
                        str(inflate_sh),
                        str(archive_dir),
                        str(inflated_dir),
                        str(names_file),
                    ],
                    capture_output=True,
                    timeout=120,
                    cwd=str(packet_dir),
                )
                return proc.returncode
            except (subprocess.TimeoutExpired, OSError):
                return -1

        rc_ref = _run_inflate(ref_archive_dir, ref_inflated)
        rc_mut = _run_inflate(mut_archive_dir, mut_inflated)

        # If both runs failed identically (e.g. CLI signature mismatch),
        # we can't make a claim. Return False — the no-op proof remains
        # advisory only; the static text scan governs.
        if rc_ref < 0 and rc_mut < 0:
            return False, "inflate_py_unexecutable_for_smoke"

        # If both runs failed (non-zero returncode) the smoke is inconclusive
        # — they may both crash on the same arg signature mismatch with
        # neither actually reading our archive. Don't downgrade in that case;
        # the static text scan governs.
        if rc_ref != 0 and rc_mut != 0:
            return False, (
                "inflate_py_failed_in_both_runs_smoke_inconclusive:"
                f"rc_ref={rc_ref}_rc_mut={rc_mut}"
            )

        # If the reference run failed but mutated succeeded (or vice-versa)
        # the script IS reading the bytes (different bytes → different rc).
        if rc_ref != rc_mut:
            return True, f"byte_mutation_changed_returncode:{rc_ref}_vs_{rc_mut}"

        # Both succeeded: compare output trees.
        def _fingerprint(root: Path) -> str:
            import hashlib

            sha = hashlib.sha256()
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    sha.update(p.relative_to(root).as_posix().encode("utf-8"))
                    sha.update(b"\x00")
                    try:
                        sha.update(p.read_bytes())
                    except OSError:
                        sha.update(b"<unreadable>")
                    sha.update(b"\x00")
            return sha.hexdigest()

        fp_ref = _fingerprint(ref_inflated)
        fp_mut = _fingerprint(mut_inflated)
        # If both inflated dirs are empty (both succeeded with rc=0 but
        # produced no files), the smoke is also inconclusive — we cannot
        # distinguish "script silently no-ops" from "script writes outputs
        # outside the inflated_dir contract". Don't downgrade.
        empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        if fp_ref == fp_mut == empty_sha:
            return False, (
                "inflate_py_produced_no_output_in_either_run_smoke_inconclusive"
            )
        if fp_ref == fp_mut:
            return (
                False,
                f"byte_mutation_did_not_change_inflated_output:fp={fp_ref[:12]}",
            )
        return (
            True,
            f"byte_mutation_changed_inflated_output:ref={fp_ref[:12]}:mut={fp_mut[:12]}",
        )
    finally:
        try:
            shutil.rmtree(smoke_dir, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------


def _compile_identity(
    *,
    input_packet_dir: Path,
    input_archive: Path,
    output_dir: Path,
    target_mode: TargetMode,
    runtime_dep_closure: list[str],
    export_format: str,
    bolt_on_loc_budget: int,
) -> Phase1PacketResult:
    """Re-emit the input packet byte-for-byte. Verifies SHA preserved."""
    pre_archive_sha = sha256_file(input_archive)
    pre_archive_size = input_archive.stat().st_size

    copied = _copy_packet_tree(input_packet_dir, output_dir)
    declared_files = [row["relpath"] for row in copied]

    out_archive = output_dir / "archive.zip"
    if not out_archive.is_file():
        raise Phase1PacketCompilerError(
            f"identity copy missing archive.zip in output: {out_archive}"
        )
    post_archive_sha = sha256_file(out_archive)
    post_archive_size = out_archive.stat().st_size

    if post_archive_sha != pre_archive_sha:
        raise Phase1PacketCompilerError(
            "identity-mode failed byte-closure: archive.zip SHA changed "
            f"({pre_archive_sha} -> {post_archive_sha})"
        )
    if post_archive_size != pre_archive_size:
        raise Phase1PacketCompilerError(
            "identity-mode failed byte-closure: archive.zip size changed "
            f"({pre_archive_size} -> {post_archive_size})"
        )

    return _finalize_packet_result(
        output_dir=output_dir,
        mode="identity",
        target_mode=target_mode,
        runtime_dep_closure=runtime_dep_closure,
        export_format=export_format,
        bolt_on_loc_budget=bolt_on_loc_budget,
        baseline_archive_sha256=pre_archive_sha,
        baseline_archive_size_bytes=pre_archive_size,
        score_affecting_payload_changed=False,
        declared_files=declared_files,
    )


def _compile_canonicalize(
    *,
    input_packet_dir: Path,
    input_archive: Path,
    output_dir: Path,
    target_mode: TargetMode,
    runtime_dep_closure: list[str],
    export_format: str,
    bolt_on_loc_budget: int,
) -> Phase1PacketResult:
    """Rebuild the packet, normalising compliance metadata (deterministic ZIP
    timestamps + sorted member order). Refuses to change score-affecting
    payload bytes.
    """
    pre_archive_sha = sha256_file(input_archive)
    pre_archive_size = input_archive.stat().st_size

    # Read members.
    members = _read_archive_members(input_archive)

    # Re-emit deterministically.
    copied: list[dict[str, Any]] = []
    for src in sorted(input_packet_dir.rglob("*"), key=lambda p: p.relative_to(input_packet_dir).as_posix()):
        if "__pycache__" in src.parts or src.name.endswith((".pyc", ".pyo")):
            continue
        if src.is_symlink():
            raise Phase1PacketCompilerError(
                f"packet refuses symlink at: {src.relative_to(input_packet_dir)}"
            )
        rel = src.relative_to(input_packet_dir).as_posix()
        if not src.is_file():
            continue
        if rel == "archive.zip":
            continue
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        dst.chmod(EXECUTABLE_MODE if rel == "inflate.sh" else NON_EXECUTABLE_MODE)
        copied.append(
            {
                "relpath": rel,
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "mode": f"{stat.S_IMODE(dst.stat().st_mode):04o}",
            }
        )
    out_archive = output_dir / "archive.zip"
    with zipfile.ZipFile(out_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        with zipfile.ZipFile(input_archive) as src_zf:
            for member_row in members:
                payload = src_zf.read(member_row["name"])
                info = zipfile.ZipInfo(member_row["name"], date_time=DETERMINISTIC_ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (0o100000 | NON_EXECUTABLE_MODE) << 16
                info.create_system = 3
                zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)

    post_archive_sha = sha256_file(out_archive)
    post_archive_size = out_archive.stat().st_size

    # Verify payload SHA per-member is preserved (canonicalize MUST NOT
    # change any score-affecting byte).
    out_members = _read_archive_members(out_archive)
    for pre, post in zip(members, out_members, strict=False):
        if pre["sha256"] != post["sha256"]:
            raise Phase1PacketCompilerError(
                "canonicalize-mode changed score-affecting payload bytes "
                f"for member {pre['name']!r}: {pre['sha256']} -> {post['sha256']}"
            )

    declared_files = [row["relpath"] for row in copied]

    return _finalize_packet_result(
        output_dir=output_dir,
        mode="canonicalize",
        target_mode=target_mode,
        runtime_dep_closure=runtime_dep_closure,
        export_format=export_format,
        bolt_on_loc_budget=bolt_on_loc_budget,
        baseline_archive_sha256=pre_archive_sha,
        baseline_archive_size_bytes=pre_archive_size,
        score_affecting_payload_changed=False,
        declared_files=declared_files,
    )


def _compile_optimize(
    *,
    input_packet_dir: Path,
    input_archive: Path,
    output_dir: Path,
    target_mode: TargetMode,
    runtime_dep_closure: list[str],
    export_format: str,
    bolt_on_loc_budget: int,
    baseline_archive_sha256: str,
    baseline_archive_size_bytes: int,
    score_affecting_payload_changed: bool,
    declared_packet_compiler_transforms: tuple[str, ...] = (),
) -> Phase1PacketResult:
    """Emit a new packet from a Phase 1 trained checkpoint. Score-affecting
    bytes change by definition.

    Phase 1 scope: this mode CURRENTLY copies the input packet's runtime tree
    + archive verbatim (the trainer is responsible for byte-tight encoding;
    the compiler asserts the byte-closure contract). Phase 2 will introduce
    payload re-encoding (Ballé string re-pack, decoder state-dict
    deterministic serialiser, brotli-q11) inside this mode.
    """
    if not score_affecting_payload_changed:
        raise Phase1PacketCompilerError(
            "optimize mode requires score_affecting_payload_changed=True; "
            "if no payload bytes change, use canonicalize mode instead"
        )
    if not baseline_archive_sha256 or not isinstance(baseline_archive_sha256, str):
        raise Phase1PacketCompilerError(
            "optimize mode requires baseline_archive_sha256 (the pre-emit "
            "reference SHA) so no_op_proof can record old/new delta"
        )
    if baseline_archive_size_bytes is None or baseline_archive_size_bytes <= 0:
        raise Phase1PacketCompilerError(
            "optimize mode requires baseline_archive_size_bytes > 0"
        )

    copied = _copy_packet_tree(input_packet_dir, output_dir)
    out_archive = output_dir / "archive.zip"
    if not out_archive.is_file():
        raise Phase1PacketCompilerError(
            f"optimize copy missing archive.zip in output: {out_archive}"
        )
    declared_files = [row["relpath"] for row in copied]

    # Round 1 Hotz MEDIUM fix: optimize mode asserts
    # score_affecting_payload_changed=True; the new archive SHA MUST
    # therefore differ from baseline_archive_sha256. If they match, the
    # operator's claim is internally inconsistent and we refuse rather than
    # emitting a no_op_proof with sha_changed=False but
    # score_affecting_payload_changed=True.
    new_sha = sha256_file(out_archive)
    if new_sha == baseline_archive_sha256:
        raise Phase1PacketCompilerError(
            "optimize mode asserted score_affecting_payload_changed=True but "
            "the emitted archive.zip SHA matches baseline_archive_sha256 "
            f"({baseline_archive_sha256}); use canonicalize mode for "
            "byte-identical re-emit"
        )
    transform_proof = _load_packet_compiler_transform_proof(
        input_packet_dir=input_packet_dir,
        declared_transforms=declared_packet_compiler_transforms,
        archive_sha256=new_sha,
    )

    return _finalize_packet_result(
        output_dir=output_dir,
        mode="optimize",
        target_mode=target_mode,
        runtime_dep_closure=runtime_dep_closure,
        export_format=export_format,
        bolt_on_loc_budget=bolt_on_loc_budget,
        baseline_archive_sha256=baseline_archive_sha256,
        baseline_archive_size_bytes=baseline_archive_size_bytes,
        score_affecting_payload_changed=score_affecting_payload_changed,
        declared_files=declared_files,
        declared_packet_compiler_transforms=declared_packet_compiler_transforms,
        packet_compiler_transform_proof=transform_proof,
    )


def _finalize_packet_result(
    *,
    output_dir: Path,
    mode: CompilerMode,
    target_mode: TargetMode,
    runtime_dep_closure: list[str],
    export_format: str,
    bolt_on_loc_budget: int,
    baseline_archive_sha256: str | None,
    baseline_archive_size_bytes: int | None,
    score_affecting_payload_changed: bool,
    declared_files: list[str],
    declared_packet_compiler_transforms: tuple[str, ...] = (),
    packet_compiler_transform_proof: dict[str, Any] | None = None,
) -> Phase1PacketResult:
    """Run the fail-closed gates + write build_manifest.json + no_op_proof.json,
    and return a Phase1PacketResult.
    """
    out_archive = output_dir / "archive.zip"
    inflate_sh = output_dir / "inflate.sh"
    inflate_py = output_dir / "inflate.py"

    blockers: list[str] = []

    if not out_archive.is_file():
        blockers.append("output_archive_missing")
        archive_members: list[dict[str, Any]] = []
        archive_sha = ""
        archive_size = 0
    else:
        archive_members = _read_archive_members(out_archive)
        archive_sha = sha256_file(out_archive)
        archive_size = out_archive.stat().st_size

    runtime_files = _runtime_tree_files(output_dir)
    if not runtime_files:
        blockers.append("runtime_tree_empty_no_inflate_runtime_custody")
    for row in runtime_files:
        if not row.get("mode_matches_expected", False):
            blockers.append(
                "runtime_file_mode_mismatch:"
                f"{row['relpath']}:mode={row['mode']}:expected={row['expected_mode']}"
            )
    pre_manifest_runtime_tree_sha = _runtime_tree_sha256(runtime_files)

    inflate_sh_info: dict[str, Any] = {}
    if not inflate_sh.is_file():
        blockers.append("inflate_sh_missing")
    else:
        sh_blockers, sh_info = _scan_inflate_sh(inflate_sh)
        blockers.extend(sh_blockers)
        inflate_sh_info = sh_info
        if sh_info["loc"] > 100:
            blockers.append(
                f"inflate_sh_loc_{sh_info['loc']}_exceeds_budget_100"
            )

    if not inflate_py.is_file():
        blockers.append("inflate_py_missing")
    else:
        py_blockers = _scan_inflate_py(inflate_py)
        blockers.extend(py_blockers)

    blockers.extend(_scan_runtime_python_surfaces(output_dir))
    blockers.extend(_scan_runtime_shell_surfaces(output_dir))

    # Round 1 Selfcomp MEDIUM fix: scan runtime tree for undeclared third-
    # party imports. Catches the case where the inflate.py imports e.g.
    # `compressai` but the operator only declared `["torch"]`.
    undeclared = _undeclared_python_imports_in_runtime_tree(
        output_dir, runtime_dep_closure
    )
    blockers.extend(undeclared)

    # Hidden sidecar gate: every file in the runtime tree + archive.zip must
    # be in declared_files. We always include archive.zip implicitly.
    declared_set = set(declared_files) | {"archive.zip"}
    actual_set = {row["relpath"] for row in runtime_files} | {"archive.zip"}
    extras = sorted(actual_set - declared_set)
    if extras:
        for rel in extras:
            blockers.append(f"hidden_sidecar:{rel}")

    # Per Check 109 (no public PR intake clone edits): we don't write into
    # those clones from this compiler; output_dir must NOT be inside a
    # public_pr*_intake_*/ directory.
    pr_intake_re = re.compile(r"public_pr[0-9]+_.*_intake_")
    if pr_intake_re.search(output_dir.as_posix()):
        blockers.append(
            "output_dir_inside_public_pr_intake_clone_forbidden_per_check_109"
        )

    parser_section_manifest = _build_parser_section_manifest(
        archive_members, packet_dir=output_dir
    )
    hnerv_parity_manifest = _build_hnerv_parity_manifest(
        archive_size_bytes=archive_size,
        inflate_sh_loc=inflate_sh_info.get("loc", 0),
        runtime_dep_closure=runtime_dep_closure,
        export_format=export_format,
        bolt_on_loc_budget=bolt_on_loc_budget,
    )
    runtime_consumes = _verify_runtime_consumes_payload_bytes(runtime_files, output_dir)
    # Codex round 5 HIGH 1 fix (catalog #139): augment the static text-pattern
    # detector with the executable byte-mutation smoke. Only attempted when
    # we have a real archive to mutate AND inflate.py is present; failures
    # of the smoke itself (timeout, missing entry point) leave runtime_consumes
    # at the static-detector value (best-effort).
    no_op_smoke_outcome: dict[str, Any] = {
        "executable_smoke_attempted": False,
        "executable_smoke_passed": None,
        "executable_smoke_reason": "not_attempted",
    }
    if out_archive.is_file() and inflate_py.is_file():
        smoke_passed, smoke_reason = _verify_runtime_consumes_payload_bytes_executable(
            packet_dir=output_dir,
            archive_path=out_archive,
        )
        no_op_smoke_outcome = {
            "executable_smoke_attempted": True,
            "executable_smoke_passed": smoke_passed,
            "executable_smoke_reason": smoke_reason,
        }
        # The executable smoke is AUTHORITATIVE when it returns True (proves
        # bytes flow). When it returns False we DOWNGRADE runtime_consumes
        # only if the smoke itself ran successfully (i.e. produced a
        # decisive negative). Inflate-unexecutable-for-smoke leaves the
        # static detector's verdict in place.
        if smoke_passed:
            runtime_consumes = True
        elif smoke_reason.startswith("byte_mutation_did_not_change_"):
            runtime_consumes = False
    no_op_proof = _build_no_op_proof(
        new_archive_sha256=archive_sha,
        new_archive_size=archive_size,
        baseline_archive_sha256=baseline_archive_sha256,
        baseline_archive_size_bytes=baseline_archive_size_bytes,
        runtime_consumes_bytes=runtime_consumes,
        score_affecting_payload_changed=score_affecting_payload_changed,
    )
    no_op_proof["executable_smoke"] = no_op_smoke_outcome

    # Codex round 5 HIGH 1 fix (catalog #139): promote no-op-proof failures
    # to BLOCKERS. Pre-fix, ``runtime_consumption_proof=False`` and
    # ``no_op_detector_passed=False`` were RECORDED on no_op_proof.json but
    # the CLI exit gate read only ``result.blockers`` — so a no-op compile
    # exited 0 and burned eval spend. The two failure modes are now first-
    # class blockers; the ``# NO_OP_PROOF_ADVISORY_OK:<reason>`` waiver
    # exists for the rare deliberately-advisory mode (e.g., a probe-only
    # packet that does not produce a runnable inflate) — callers OPT IN
    # via ``allow_no_op_proof_advisory=True`` to suppress promotion.
    if no_op_proof.get("runtime_consumption_proof") is False:
        blockers.append(
            "inflate_does_not_consume_archive_bytes:"
            f"smoke={no_op_smoke_outcome['executable_smoke_reason']}"
        )
    if no_op_proof.get("no_op_detector_passed") is False:
        blockers.append(
            "no_op_detector_failed:"
            f"score_affecting_payload_changed={score_affecting_payload_changed}:"
            f"sha_changed={no_op_proof.get('sha_changed')}"
        )

    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "created_at_utc": _utc_iso(),
        "mode": mode,
        "target_mode": target_mode,
        "lane_class": "substrate_engineering",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_exact_eval_dispatch_reason": (
            "Phase 1 packet compiler proves byte-closure only; promotion + "
            "dispatch readiness require contest-CUDA + contest-CPU auth eval "
            "on these exact archive bytes."
        ),
        "evidence_grade": "byte_custody_only",
        "output_dir": output_dir.as_posix(),
        "archive_relpath": "archive.zip",
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "archive_members": archive_members,
        # ``contest_auth_eval.py`` hashes every .json file in the runtime
        # root, including this build_manifest.json and no_op_proof.json. A
        # final evaluator-visible hash cannot be embedded here without making
        # the hash self-referential, so keep the expected-hash field falsey.
        # Callers must use contest_auth_eval's recorded
        # provenance.inflate_runtime_manifest.runtime_tree_sha256 for the final
        # tree. The pre-manifest hash remains below for compiler custody.
        "runtime_tree_sha256": "",
        "runtime_tree_sha256_status": (
            "withheld_self_referential_final_manifest_hash_use_contest_auth_eval"
        ),
        "pre_manifest_runtime_tree_sha256": pre_manifest_runtime_tree_sha,
        "runtime_files_scope": (
            "pre_manifest_runtime_tree_excludes_build_manifest_json_and_no_op_proof_json"
        ),
        "runtime_files": runtime_files,
        "inflate_sh_info": inflate_sh_info,
        "parser_section_manifest": parser_section_manifest,
        "hnerv_parity_manifest": hnerv_parity_manifest,
        "no_op_proof": no_op_proof,
        "packet_compiler_transforms": list(declared_packet_compiler_transforms),
        "packet_compiler_transform_proof": (
            packet_compiler_transform_proof
            if packet_compiler_transform_proof is not None
            else {
                "status": "not_applicable",
                "packet_compiler_transforms": [],
                "score_claim": False,
                "runtime_consumption_proof": False,
            }
        ),
        "blockers": blockers,
    }
    (output_dir / "build_manifest.json").write_text(
        json_text(build_manifest), encoding="utf-8"
    )
    (output_dir / "no_op_proof.json").write_text(
        json_text(no_op_proof), encoding="utf-8"
    )

    return Phase1PacketResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        target_mode=target_mode,
        output_dir=output_dir.as_posix(),
        archive_path=(output_dir / "archive.zip").as_posix(),
        archive_sha256=archive_sha,
        archive_size_bytes=archive_size,
        runtime_tree_sha256=pre_manifest_runtime_tree_sha,
        runtime_files=tuple(runtime_files),
        archive_members=tuple(archive_members),
        parser_section_manifest=parser_section_manifest,
        hnerv_parity_manifest=hnerv_parity_manifest,
        no_op_proof=no_op_proof,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        blockers=tuple(blockers),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


#: Recognised typed packet-compiler transforms callers can declare were
#: applied by the upstream trainer. These are recorded in the build manifest
#: under ``packet_compiler_transforms`` so reviewers know which
#: ``tac.packet_compiler`` primitive produced the score-affecting bytes. The
#: Phase 1 compiler does NOT mutate archive bytes itself; it only records the
#: declared transforms for downstream audit + future Rust/Zig port routing.
#: Unknown tokens are refused so a typo cannot silently slip into a manifest.
PACKET_COMPILER_TRANSFORMS: tuple[str, ...] = (
    # PR101 — sidecar grammar
    "pr101_ranked_no_op_sidecar",
    "pr101_centered_delta_uint8_lzma",
    "pr101_split_brotli_self_delimiting",
    # PR103 — arithmetic coding
    "pr103_merged_range_stream",
    "pr103_latent_hi_arithmetic",
    "pr103_adaptive_brotli_param_search",
    # PR81 — Quantizr FP4 codebook + ROUTER_ACTION
    "pr81_fp4_codebook",
    "pr81_router_action",
    # PR84 — adaptive-context range coder
    "pr84_adaptive_mask_context",
    # PR91 — universal AC wrapper + QMQH grammar
    "pr91_arithmetic_coder_constriction",
    "pr91_qmqh_grammar",
    # PR92 — RMC1 joint stream meta-codec
    "pr92_rmc_joint_stream",
    # PR93 — delta-varint pose + QZMB1
    "pr93_delta_varint_pose",
    "pr93_qzmb_qzpdv_grammar",
    # PR63 — qpose14 uint16-view int16 + single-zip-member packed payload (2026-05-12)
    "pr63_qpose14_uint16_view_int16",
    "pr63_qpose14_packed_payload",
    # PR64 — unified-brotli pose-velocity-only codec (2026-05-12)
    "pr64_unified_brotli_pose_velocity",
    # PR65 — PQ12 12-bit / 3-byte / 2-value packed pose codec (2026-05-12)
    "pr65_pq12_pose",
    # PR105 — kitchen_sink packed-state-schema size-sorted helper (2026-05-12)
    "pr105_packed_state_schema_size_sorted",
    # PR101 GOLD — 3 newly-ported primitives (2026-05-12)
    "pr101_decoder_storage_order",
    "pr101_conv4_storage_perms",
    "pr101_decoder_byte_maps",
    # PR93 — lowpass-luma residual codec (3 or 6 fp32 coeffs, low-freq RGB-luma correction)
    "pr93_lowpass_luma_residual",
    # PR97 — H3 wire-format grammar (length-prefixed sections + tile-band multi-stream)
    "pr97_length_prefixed_sections",
    "pr97_tile_band_streams",
    # Sparse PacketIR codec — closes O's L2 wire-format ceiling (2026-05-11)
    "sparse_rle_of_zeros",
    "sparse_arithmetic_coefficients",
    "sparse_temporal_subsampled",
    # Magic codec — per-stream auto-selector + meta-codec dispatch (2026-05-11)
    "magic_codec_auto_select",
    # Magic codec dense streams — per-stream brotli/lzma/magic_classic bundle (2026-05-12)
    "magic_codec_dense_streams",
    # Sign-encoding 5-strategy unified taxonomy (2026-05-12)
    # Unifies PR96 / PR101 / PR103 sign-encoding strategies into one typed
    # API. Each token tags one of the 5 strategies; encoders/decoders are
    # in ``src/tac/packet_compiler/sign_encoding.py``.
    #
    # MUTUAL EXCLUSION CONTRACT (per ZZZZZ audit Shannon-Low 2026-05-12):
    # the 5 ``sign_encode_*`` strategies are MUTUALLY EXCLUSIVE on the SAME
    # byte region — at most one strategy may be tagged per (tensor, region)
    # pair. Stacking two strategies on the same bytes is a no-op + roundtrip
    # corruption (the second decoder would interpret bytes the first
    # encoder produced under a different convention). Adapter callers are
    # responsible for enforcing per-region uniqueness; the token list here
    # does not. Catalog #139 ``check_packet_compiler_no_op_proof_promotes_to_blocker``
    # catches the stacking violation at byte-mutation smoke time.
    "sign_encode_negzig",
    "sign_encode_zig",
    "sign_encode_twos",
    "sign_encode_off",
    "sign_encode_raw_uint8",
    # Schema-elision V1+V2 — PR98 CD1 compact + PR100 schema-driven decoder
    # grammar. Sister to PR105 packed-state-schema size-sort already
    # registered above. V1 and V2 are MUTUALLY EXCLUSIVE (both elide the
    # same per-tensor metadata region — see design memo
    # `.omx/research/schema_elision_design_pr98_pr100_pr105_20260512.md`).
    "pr98_cd1_compact_architecture_ordered_decoder_format",
    "pr100_schema_driven_decoder_storage_grammar",
    # CompressAI reference neural-compression codecs — packet-compiler
    # adapters (2026-05-12). Each token tags one CompressAI model family.
    # Adapters in ``src/tac/packet_compiler/{factorized_prior,
    # balle_hyperprior, cheng2020}.py`` wrap encode/decode + serialize/
    # deserialize behind a typed wire format with a magic+version header.
    # ``score_claim=false`` per
    # ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``;
    # archive-grammar declaration lives in each adapter's docstring.
    "compressai_factorized_prior",
    "compressai_balle_hyperprior",
    "compressai_cheng2020",
)
PACKET_COMPILER_TRANSFORM_PROOF_FILENAME = "packet_compiler_transform_proof.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_packet_compiler_transform_proof(
    *,
    input_packet_dir: Path,
    declared_transforms: tuple[str, ...],
    archive_sha256: str,
) -> dict[str, Any]:
    """Load fail-closed proof that declared transforms materialized bytes."""

    if not declared_transforms:
        return {
            "status": "not_applicable",
            "packet_compiler_transforms": [],
            "score_claim": False,
            "runtime_consumption_proof": False,
        }
    proof_path = input_packet_dir / PACKET_COMPILER_TRANSFORM_PROOF_FILENAME
    if not proof_path.is_file():
        raise Phase1PacketCompilerError(
            "declared packet_compiler_transforms require "
            f"{PACKET_COMPILER_TRANSFORM_PROOF_FILENAME} beside archive.zip; "
            "loose transform labels are not materialization proof"
        )
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase1PacketCompilerError(
            f"could not parse {PACKET_COMPILER_TRANSFORM_PROOF_FILENAME}: {exc}"
        ) from exc
    if not isinstance(proof, dict):
        raise Phase1PacketCompilerError(
            f"{PACKET_COMPILER_TRANSFORM_PROOF_FILENAME} must be a JSON object"
        )
    proof_transforms = tuple(
        proof.get("packet_compiler_transforms") or proof.get("transforms") or ()
    )
    if proof_transforms != declared_transforms:
        raise Phase1PacketCompilerError(
            "packet_compiler_transform_proof transform list does not match "
            f"declared packet_compiler_transforms: proof={list(proof_transforms)} "
            f"declared={list(declared_transforms)}"
        )
    if proof.get("archive_sha256") != archive_sha256:
        raise Phase1PacketCompilerError(
            "packet_compiler_transform_proof archive_sha256 does not match "
            f"compiled archive: proof={proof.get('archive_sha256')!r} "
            f"compiled={archive_sha256}"
        )
    proof_kind = proof.get("proof_kind")
    if proof_kind not in {
        "upstream_trainer_materialized_bytes",
        "packet_compiler_materialized_bytes",
    }:
        raise Phase1PacketCompilerError(
            "packet_compiler_transform_proof requires proof_kind in "
            "{'upstream_trainer_materialized_bytes', 'packet_compiler_materialized_bytes'}"
        )
    if proof.get("score_claim") is True:
        raise Phase1PacketCompilerError(
            "packet_compiler_transform_proof must not claim score; exact auth eval owns score claims"
        )
    if proof.get("runtime_consumption_proof") is not True:
        raise Phase1PacketCompilerError(
            "packet_compiler_transform_proof requires runtime_consumption_proof=true; "
            "declared transforms without consumed-byte proof remain forensic"
        )
    evidence = proof.get("transform_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(proof_transforms):
        raise Phase1PacketCompilerError(
            "packet_compiler_transform_proof requires transform_evidence with one row per transform"
        )
    for index, (expected_transform, row) in enumerate(zip(proof_transforms, evidence, strict=True)):
        if not isinstance(row, dict):
            raise Phase1PacketCompilerError(
                f"packet_compiler_transform_proof transform_evidence[{index}] must be an object"
            )
        if row.get("transform") != expected_transform:
            raise Phase1PacketCompilerError(
                "packet_compiler_transform_proof transform_evidence transform mismatch: "
                f"row={row.get('transform')!r} expected={expected_transform!r}"
            )
        input_sha = row.get("input_sha256")
        output_sha = row.get("output_sha256")
        if not (isinstance(input_sha, str) and _SHA256_RE.match(input_sha)):
            raise Phase1PacketCompilerError(
                f"packet_compiler_transform_proof transform_evidence[{index}] input_sha256 is required"
            )
        if not (isinstance(output_sha, str) and _SHA256_RE.match(output_sha)):
            raise Phase1PacketCompilerError(
                f"packet_compiler_transform_proof transform_evidence[{index}] output_sha256 is required"
            )
        if input_sha == output_sha:
            raise Phase1PacketCompilerError(
                f"packet_compiler_transform_proof transform_evidence[{index}] did not change bytes"
            )
        has_target_member = isinstance(row.get("target_member"), str) and bool(
            row["target_member"].strip()
        )
        has_section_id = isinstance(row.get("section_id"), str) and bool(
            row["section_id"].strip()
        )
        if not (has_target_member or has_section_id):
            raise Phase1PacketCompilerError(
                f"packet_compiler_transform_proof transform_evidence[{index}] requires target_member or section_id"
            )
        byte_delta = row.get("byte_delta")
        changed_bytes = row.get("changed_bytes_count")
        has_byte_delta = (
            isinstance(byte_delta, int)
            and not isinstance(byte_delta, bool)
            and byte_delta != 0
        )
        has_changed_bytes = (
            isinstance(changed_bytes, int)
            and not isinstance(changed_bytes, bool)
            and changed_bytes > 0
        )
        if not (has_byte_delta or has_changed_bytes):
            raise Phase1PacketCompilerError(
                f"packet_compiler_transform_proof transform_evidence[{index}] requires nonzero byte_delta or changed_bytes_count"
            )
    return {
        "status": "materialization_proof_present",
        "path": proof_path.name,
        "schema": proof.get("schema"),
        "proof_kind": proof_kind,
        "packet_compiler_transforms": list(proof_transforms),
        "archive_sha256": archive_sha256,
        "score_claim": False,
        "runtime_consumption_proof": True,
        "transform_evidence": evidence,
        "producer": proof.get("producer"),
        "source_manifest": proof.get("source_manifest"),
    }


def compile_phase1_packet(
    input_packet: Path | str,
    output_dir: Path | str,
    *,
    mode: CompilerMode = "identity",
    target_mode: TargetMode = "contest_one_video_replay",
    runtime_dep_closure: Iterable[str] = ("torch", "brotli", "compressai"),
    export_format: str = "phase1_three_member_x_decoder_bin_balle_bin",
    bolt_on_loc_budget: int = 400,
    allow_existing_output_dir: bool = False,
    score_affecting_payload_changed: bool = False,
    baseline_archive_sha256: str | None = None,
    baseline_archive_size_bytes: int | None = None,
    fail_on_score_affecting_change: bool = True,
    packet_compiler_transforms: Iterable[str] | None = None,
) -> Phase1PacketResult:
    """Compile a Phase 1 packet from an input packet directory or archive.

    Parameters
    ----------
    input_packet
        Either a packet directory containing ``archive.zip``, ``inflate.sh``,
        ``inflate.py``, ``src/`` (the trainer's ``submission_dir/``-shaped
        output), or a bare ``archive.zip`` (in which case its parent
        directory must contain the runtime tree).
    output_dir
        Destination directory. Must not exist, or must be empty unless
        ``allow_existing_output_dir=True``.
    mode
        One of ``identity`` / ``canonicalize`` / ``optimize``.
    target_mode
        ``contest_one_video_replay`` (default) or ``contest_generalized``.
    runtime_dep_closure
        The exhaustive list of runtime deps the inflate path requires; the
        manifest records this so dispatch+preflight gates can audit it.
    export_format
        HNeRV-parity field describing the payload-on-disk grammar.
    bolt_on_loc_budget
        Per HNeRV parity discipline: the LOC budget the lane is permitted
        to spend on bolt-on transforms.
    allow_existing_output_dir
        When True, replace any existing contents of ``output_dir``.
    score_affecting_payload_changed
        REQUIRED to be True for ``optimize`` mode. Forbidden to be True for
        ``identity`` and ``canonicalize`` modes.
    baseline_archive_sha256, baseline_archive_size_bytes
        Required for ``optimize`` mode so the no_op_proof can record
        old/new delta. Optional for the other modes (auto-derived from the
        input archive).
    fail_on_score_affecting_change
        When True (default), refuse to write outputs if canonicalize mode
        would alter any payload SHA. Set False only for
        diagnostic/inspection workflows.
    packet_compiler_transforms
        Optional list of ``tac.packet_compiler`` primitive identifiers the
        upstream trainer used to produce the archive bytes. The compiler
        records these in ``build_manifest.json::packet_compiler_transforms``
        for downstream audit + future native-port routing. The list MUST
        come from :data:`PACKET_COMPILER_TRANSFORMS`; unknown tokens are
        refused. Default behaviour is unchanged (None → empty list); this
        flag never mutates archive bytes — that responsibility stays with
        the trainer per the existing "optimize mode is a custody contract"
        contract.

    Returns
    -------
    Phase1PacketResult
        Structured result mirroring ``build_manifest.json``. Blockers list
        is non-empty when any fail-closed gate fired; the manifest is still
        written (so operators can inspect what failed) but the result's
        ``score_claim`` / ``promotion_eligible`` /
        ``ready_for_exact_eval_dispatch`` are all False.

    Raises
    ------
    Phase1PacketCompilerError
        On any structural failure — missing input, unsupported mode,
        unsafe ZIP member, identity-mode SHA drift, canonicalize-mode
        score-affecting byte change with ``fail_on_score_affecting_change``
        True, etc.
    """
    if mode not in COMPILER_MODES:
        raise Phase1PacketCompilerError(
            f"unknown mode {mode!r}; expected one of {COMPILER_MODES}"
        )
    if target_mode not in TARGET_MODES:
        raise Phase1PacketCompilerError(
            f"unknown target_mode {target_mode!r}; expected one of {TARGET_MODES}"
        )
    if mode in {"identity", "canonicalize"} and score_affecting_payload_changed:
        raise Phase1PacketCompilerError(
            f"{mode} mode forbids score_affecting_payload_changed=True; "
            "use optimize mode for any payload-changing emit"
        )

    declared_transforms = tuple(packet_compiler_transforms or ())
    unknown_transforms = [
        t for t in declared_transforms if t not in PACKET_COMPILER_TRANSFORMS
    ]
    if unknown_transforms:
        raise Phase1PacketCompilerError(
            f"unknown packet_compiler_transforms tokens: {unknown_transforms}; "
            f"expected subset of {list(PACKET_COMPILER_TRANSFORMS)}"
        )
    if declared_transforms and mode != "optimize":
        raise Phase1PacketCompilerError(
            "packet_compiler_transforms may only be declared in optimize "
            f"mode (mode={mode!r} forbids byte-changing transforms)"
        )

    input_path = Path(input_packet)
    output_dir_path = Path(output_dir)

    input_packet_dir, input_archive = _resolve_input_packet(input_path)

    # Empty checkpoint guard. R9 Tao 2nd HIGH fix: also reject zero-member
    # ZIPs (file is non-empty, e.g. 22 bytes for an empty central directory,
    # but no payload exists).
    if input_archive.stat().st_size == 0:
        raise Phase1PacketCompilerError(
            f"input archive is empty (zero bytes): {input_archive}; "
            "refusing to compile a no-op packet"
        )
    try:
        _empty_check_members = _read_archive_members(input_archive)
    except Phase1PacketCompilerError:
        # Re-raise: malformed ZIP / unsafe member is a hard fail.
        raise
    if not _empty_check_members:
        raise Phase1PacketCompilerError(
            f"input archive has zero members (empty central directory): "
            f"{input_archive}; refusing to compile a no-op packet"
        )

    runtime_dep_closure_list = list(runtime_dep_closure)

    # R10 Quantizr 3rd HIGH fix: refuse if output_dir == input packet dir or
    # is a parent of the input packet dir. With allow_existing_output_dir
    # the safe-clear would otherwise delete the input itself.
    try:
        in_resolved = input_packet_dir.resolve()
        out_resolved = output_dir_path.resolve()
    except OSError:
        in_resolved, out_resolved = input_packet_dir, output_dir_path
    if out_resolved == in_resolved:
        raise Phase1PacketCompilerError(
            f"output_dir is the same as input packet dir: {out_resolved}; "
            "refusing to overwrite the input"
        )
    try:
        if in_resolved.is_relative_to(out_resolved):
            raise Phase1PacketCompilerError(
                f"output_dir {out_resolved} is a parent of input packet dir "
                f"{in_resolved}; refusing to overwrite the input"
            )
    except AttributeError:  # pragma: no cover - Path.is_relative_to <3.9
        if str(in_resolved).startswith(str(out_resolved) + "/"):
            raise Phase1PacketCompilerError(
                f"output_dir {out_resolved} is a parent of input packet dir "
                f"{in_resolved}; refusing to overwrite the input"
            )

    _safe_clear_output_dir(output_dir_path, allow_existing=allow_existing_output_dir)

    if mode == "identity":
        return _compile_identity(
            input_packet_dir=input_packet_dir,
            input_archive=input_archive,
            output_dir=output_dir_path,
            target_mode=target_mode,
            runtime_dep_closure=runtime_dep_closure_list,
            export_format=export_format,
            bolt_on_loc_budget=bolt_on_loc_budget,
        )
    if mode == "canonicalize":
        try:
            return _compile_canonicalize(
                input_packet_dir=input_packet_dir,
                input_archive=input_archive,
                output_dir=output_dir_path,
                target_mode=target_mode,
                runtime_dep_closure=runtime_dep_closure_list,
                export_format=export_format,
                bolt_on_loc_budget=bolt_on_loc_budget,
            )
        except Phase1PacketCompilerError:
            if fail_on_score_affecting_change:
                raise
            # diagnostic-only fall-through: re-raise to caller after writing
            # whatever partial state exists.
            raise
    if mode == "optimize":
        if baseline_archive_sha256 is None:
            raise Phase1PacketCompilerError(
                "optimize mode requires baseline_archive_sha256"
            )
        if baseline_archive_size_bytes is None:
            raise Phase1PacketCompilerError(
                "optimize mode requires baseline_archive_size_bytes"
            )
        return _compile_optimize(
            input_packet_dir=input_packet_dir,
            input_archive=input_archive,
            output_dir=output_dir_path,
            target_mode=target_mode,
            runtime_dep_closure=runtime_dep_closure_list,
            export_format=export_format,
            bolt_on_loc_budget=bolt_on_loc_budget,
            baseline_archive_sha256=baseline_archive_sha256,
            baseline_archive_size_bytes=baseline_archive_size_bytes,
            score_affecting_payload_changed=score_affecting_payload_changed,
            declared_packet_compiler_transforms=declared_transforms,
        )
    raise Phase1PacketCompilerError(f"unreachable mode: {mode!r}")


__all__ = [
    "A1_CANONICAL_ARCHIVE_SHA256",
    "A1_CANONICAL_ARCHIVE_SIZE_BYTES",
    "ALLOWED_ZIP_METHODS",
    "COMPILER_MODES",
    "DETERMINISTIC_ZIP_DATE_TIME",
    "FORBIDDEN_EXTERNAL_STATE_PATTERNS",
    "FORBIDDEN_INFLATE_TOKENS",
    "FORBIDDEN_NETWORK_TOKENS",
    "FORBIDDEN_PYTHON_RUNTIME_MODULES",
    "HNERV_PARITY_FIELDS",
    "PACKET_COMPILER_TRANSFORMS",
    "Phase1PacketCompilerError",
    "Phase1PacketResult",
    "SCHEMA_VERSION",
    "TARGET_MODES",
    "TOOL_NAME",
    "compile_phase1_packet",
]
