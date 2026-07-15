"""Executable V9 provenance and NO-FAKE preflight gates.

These checks deliberately derive their verdicts from the live V9 witness DSL,
the trainer's real argparse parser, and structured evidence receipts.  A prose
marker or a previously emitted ``PASS`` receipt is never accepted as proof that
the corresponding runtime edge exists.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXACT_LANGUAGE_RE = re.compile(
    r"(?:\bexact(?:-authority| authority| score| eval(?:uation)?)?\b|"
    r"\bauthoritative\b|\bpromotion(?:-grade| eligible)?\b|"
    r"\bpointer\s+(?:moved|improved)\b|\[contest-(?:CPU|CUDA)\])",
    re.IGNORECASE,
)
_ADVISORY_AXES = {"mps", "mlx", "macos-cpu advisory", "advisory", "proxy", "synthetic"}
_CONTEST_AXES = {"contest-cpu", "contest-cuda"}
_RUNG_ALIASES = {
    "measured_anchor",
    "derived_at_config",
    "derived_live",
    "hardcoded_waiver",
}
_PRINT_LIMIT = 25


class V9ProvenanceGateError(RuntimeError):
    """Raised when a V9 provenance gate is requested in strict mode."""


def _stable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _stable_value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(_stable_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _semantically_equal(raw_value: Any, runtime_value: Any) -> bool:
    """Compare a DSL value with argparse normalization without erasing either type."""

    if raw_value == runtime_value:
        return True
    if isinstance(raw_value, str) and isinstance(runtime_value, (int, float)):
        try:
            return float(raw_value) == float(runtime_value)
        except ValueError:
            return False
    return False


@dataclass(frozen=True)
class FlagProvenanceBinding:
    """One semantic flag's independently checkable custody edges."""

    flag: str
    raw_value: Any
    raw_type: str
    raw_tokens: tuple[str, ...]
    runtime_dest: str | None
    runtime_value: Any
    runtime_type: str | None
    lever_owners: tuple[str, ...]
    lawref_equation_ids: tuple[str, ...]
    compiler_record_equation_ids: tuple[str, ...]
    provenance_rung: str | None
    consumer_locations: tuple[str, ...]
    runtime_receipt_schemas: tuple[str, ...]


@dataclass(frozen=True)
class ConfigBijectionSnapshot:
    """Canonical, timestamp-free snapshot of one compiled V9 factory."""

    program: str
    bindings: tuple[FlagProvenanceBinding, ...]
    semantic_order: tuple[str, ...]
    emitted_order: tuple[str, ...]
    lawref_flags: tuple[str, ...]
    compiler_record_flags: tuple[str, ...]
    provenance_flags: tuple[str, ...]

    @property
    def bijection_hash(self) -> str:
        return _hash_payload(asdict(self))


@dataclass(frozen=True)
class FakeClaim:
    """A normalized claim checked against executable V9 custody."""

    claim_id: str
    vehicle: str = "v9_cgauge"
    active_basis_label: str | None = None
    basis_implementation: str | None = None
    basis_is_spatially_localized: bool | None = None
    pose_selected: bool = False
    pose_receiver_authenticated: bool = False
    pose_parseback_byte_effect_sha256: str | None = None
    d_pose_source: str | None = None
    numeric_d_pose: float | None = None
    self_orient_claim: bool | None = None
    compiled_self_orient: bool | None = None
    claimed_percent_reduction: float | None = None
    receipt_vehicle: str | None = None
    receipt_self_orient: bool | None = None
    receipt_scope: str | None = None
    waiver: str | None = None


@dataclass(frozen=True)
class EvidenceClaim:
    """A score/authority statement plus the custody capable of supporting it."""

    claim_id: str
    language: str
    evidence_axis: str
    archive_sha256: str | None = None
    archive_bytes: int | None = None
    evaluator: str | None = None
    pairs: int | None = None
    promotion_claim: bool = False
    requested_exact_authority: bool = False
    score_claim: bool = False
    waiver: str | None = None


@dataclass(frozen=True)
class FourierBasisAuditViolation:
    """One executable Fourier-basis or malformed FFT-tool waiver finding."""

    path: str
    line: int
    signature: str
    line_text: str
    kind: str
    rationale: str


_FOURIER_BASIS_SIGNATURE_RE = re.compile(
    r"\b("
    r"legacy_fourier_ab_control|LegacyFourierABControl|"
    r"polar_fourier|polar_directional_fourier(?:_[A-Za-z0-9_]+)?|"
    r"self_oriented_fourier(?:_[A-Za-z0-9_]+)?|"
    r"directional_fourier(?:_[A-Za-z0-9_]+)?|fourier_features?(?:_[A-Za-z0-9_]+)?|"
    r"fourier_basis|n_fourier|fourier_sigma|n_dir_freqs|"
    r"global_polar_directional_fourier_plane_waves"
    r")\b",
    re.IGNORECASE,
)
_FOURIER_TOKEN_RE = re.compile(r"fourier", re.IGNORECASE)
_FFT_TOOL_RE = re.compile(
    r"\b(?:np|numpy|torch|mx|scipy)\.fft\b|"
    r"\b(?:fft|fft2|ifft|ifft2|rfft|irfft|dct|dctn|idct|idctn)\s*\(",
    re.IGNORECASE,
)
_VALID_FFT_TOOL_WAIVER_RE = re.compile(r"#\s*FFT_TOOL_USE_OK:(?P<reason>.+)$")
_PLACEHOLDER_WAIVER_RE = re.compile(r"^\s*(?:todo|tbd|placeholder|ok|n/a|none)\s*$", re.IGNORECASE)
_SOURCE_ROOTS = ("src/tac", "experiments")


def _iter_python_sources(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for rel_root in _SOURCE_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        paths.extend(path for path in base.rglob("*.py") if path.is_file())
    return tuple(sorted(paths))


def _docstring_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not body or not isinstance(body, list):
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start = getattr(first, "lineno", None)
            end = getattr(first, "end_lineno", start)
            if start is not None and end is not None:
                lines.update(range(int(start), int(end) + 1))
    return lines


def _pure_comment_lines(source: str) -> set[int]:
    """Lines whose first non-whitespace token is a comment.

    Inline comments do not make the executable prefix disappear; a previous
    implementation treated any line containing a COMMENT token as non-code and
    could therefore hide ``fourier_features(...)  # comment`` from the gate.
    """

    return {
        lineno
        for lineno, line in enumerate(source.splitlines(), start=1)
        if line.lstrip().startswith("#")
    }


def _non_executable_lines(source: str) -> set[int]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _pure_comment_lines(source)
    return _docstring_lines(tree) | _pure_comment_lines(source)


_REPLACEMENT_PROOF_SCOPES = {
    "src/tac/boundary_math/compact_shearlet_frame.py": frozenset(
        {
            ("ClassDef", "ShearletCertificate"),
            ("FunctionDef", "shearlet_certificate"),
        }
    ),
    "src/tac/boundary_math/windowed_curvelet_frame.py": frozenset(
        {
            ("ClassDef", "LocalizationCertificate"),
            ("FunctionDef", "localization_certificate"),
        }
    ),
}
_GATE_APPARATUS_SOURCE_PATHS = frozenset(
    {
        "src/tac/tests/test_check_pose_basis_fit_kill.py",
        "src/tac/tests/test_v9_provenance_gates.py",
        "src/tac/v9_provenance_gates.py",
    }
)


def _replacement_proof_scope_lines(relative_path: str, source: str) -> set[int]:
    """Return exact top-level certificate/proof scopes for canonical frame paths.

    The exemption is derived from the parsed AST rather than source annotations.
    It cannot cover the feature constructors because their names are deliberately
    absent from the allowlist, and a same-basename file outside the canonical path
    receives no exemption.
    """

    scopes = _REPLACEMENT_PROOF_SCOPES.get(relative_path)
    if scopes is None:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in tree.body:
        if (type(node).__name__, getattr(node, "name", None)) not in scopes:
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if start is not None and end is not None:
            lines.update(range(int(start), int(end) + 1))
    return lines


def _gate_apparatus_source(relative_path: str) -> bool:
    """The signature table is enforcement apparatus, not a witness representation."""

    return relative_path in _GATE_APPARATUS_SOURCE_PATHS


def _valid_fft_tool_waiver(line_text: str) -> bool:
    match = _VALID_FFT_TOOL_WAIVER_RE.search(line_text)
    if not match:
        return False
    reason = match.group("reason").strip()
    return len(reason) >= 16 and not _PLACEHOLDER_WAIVER_RE.fullmatch(reason)


def audit_no_fourier_basis_in_witness_representation(
    repo_root: Path | None = None,
    *,
    paths: Sequence[Path] | None = None,
) -> list[FourierBasisAuditViolation]:
    """Pure source audit for executable Fourier witness-basis signatures.

    This intentionally does not ban generic words such as ``frequency``,
    ``spectral``, ``sin``, or ``cos``.  The strict flip is gated on the
    operator-GO n600 byte-closed realized-through-R A/B proving that the
    curvelet treatment has no d_seg regression versus the explicit legacy
    Fourier A/B control, so the production preflight wrapper is warn-only for
    now.
    """

    root = Path(repo_root or _REPO_ROOT).resolve()
    source_paths = tuple(paths) if paths is not None else _iter_python_sources(root)
    violations: list[FourierBasisAuditViolation] = []
    for source_path in source_paths:
        path = Path(source_path)
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if path.is_absolute():
            try:
                rel = path.resolve().relative_to(root).as_posix()
            except ValueError:
                rel = path.resolve().as_posix()
        else:
            rel = path.as_posix()
        non_exec = _non_executable_lines(text)
        replacement_proof_scope = _replacement_proof_scope_lines(rel, text)
        exempt_source = _gate_apparatus_source(rel)
        for lineno, line_text in enumerate(text.splitlines(), start=1):
            if lineno in non_exec or exempt_source:
                continue
            fft_match = _FFT_TOOL_RE.search(line_text)
            if fft_match:
                if "FFT_TOOL_USE_OK:" in line_text and not _valid_fft_tool_waiver(line_text):
                    violations.append(
                        FourierBasisAuditViolation(
                            path=rel,
                            line=lineno,
                            signature=fft_match.group(0),
                            line_text=line_text.strip(),
                            kind="MALFORMED_FFT_TOOL_WAIVER",
                            rationale="explicit FFT/DCT/rFFT tool-use waiver lacks a substantive rationale",
                        )
                    )
                elif "FFT_TOOL_USE_OK:" not in line_text:
                    violations.append(
                        FourierBasisAuditViolation(
                            path=rel,
                            line=lineno,
                            signature=fft_match.group(0),
                            line_text=line_text.strip(),
                            kind="FFT_TOOL_USE_UNWAIVED",
                            rationale="explicit FFT/DCT/rFFT use must be labeled as a tool transform",
                        )
                    )
            signature_matches = tuple(_FOURIER_BASIS_SIGNATURE_RE.finditer(line_text))
            if lineno not in replacement_proof_scope:
                for match in signature_matches:
                    violations.append(
                        FourierBasisAuditViolation(
                            path=rel,
                            line=lineno,
                            signature=match.group(1),
                            line_text=line_text.strip(),
                            kind="FOURIER_BASIS_SIGNATURE",
                            rationale="executable witness representation still exposes Fourier-basis semantics",
                        )
                    )
            if not signature_matches:
                token_match = _FOURIER_TOKEN_RE.search(line_text)
                if (
                    token_match
                    and lineno not in replacement_proof_scope
                    and not _valid_fft_tool_waiver(line_text)
                ):
                    violations.append(
                        FourierBasisAuditViolation(
                            path=rel,
                            line=lineno,
                            signature=token_match.group(0),
                            line_text=line_text.strip(),
                            kind="FOURIER_SEMANTIC_UNWAIVED",
                            rationale=(
                                "executable Fourier semantic is not a known basis signature and lacks "
                                "a substantive FFT-tool waiver"
                            ),
                        )
                    )
    return violations


def _canonical_factories() -> Mapping[str, Any]:
    from tac.witness_dsl.spec_v9_cgauge import (
        compile_v9_cgauge_432_launch_config,
        compile_v9_cgauge_ideal_launch_config,
        compile_v9_cgauge_ideal_mod19_launch_config,
        compile_v9_cgauge_ideal_mod32_launch_config,
    )

    return {
        "v9_cgauge_432": compile_v9_cgauge_432_launch_config,
        "v9_cgauge_truly_optimal_core": compile_v9_cgauge_ideal_launch_config,
        "v9_cgauge_ideal_mod19": compile_v9_cgauge_ideal_mod19_launch_config,
        "v9_cgauge_ideal_mod32": compile_v9_cgauge_ideal_mod32_launch_config,
    }


def _trainer_consumers(trainer_path: Path, repo_root: Path) -> Mapping[str, tuple[str, ...]]:
    """Map argparse dests to executable ``args.<dest>`` source locations."""

    tree = ast.parse(trainer_path.read_text())
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def scope_name(node: ast.AST) -> str:
        cur: ast.AST | None = node
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = parents.get(cur)
        return "<module>"

    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        dest: str | None = None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
        ):
            dest = node.attr
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "args"
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            dest = node.args[1].value
        if dest:
            found.setdefault(dest, set()).add(f"{trainer_path.relative_to(repo_root)}:{scope_name(node)}:{node.lineno}")
    return {dest: tuple(sorted(locations)) for dest, locations in found.items()}


def _lawrefs_for_flag(lever: Any, flag: str) -> tuple[Any, ...]:
    refs = getattr(lever, "constant_refs", None)
    if refs is None:
        return ()
    if isinstance(refs, Mapping):
        value = refs.get(flag, ())
        if isinstance(value, (tuple, list, set)):
            return tuple(value)
        return () if value is None else (value,)
    return tuple(ref for ref in refs if getattr(ref, "flag", None) == flag or getattr(ref, "flag_name", None) == flag)


def _receipt_schemas_for_flag(lever: Any, flag: str) -> tuple[str, ...]:
    schemas = getattr(lever, "runtime_receipt_schema", None)
    if schemas is None:
        schemas = getattr(lever, "runtime_receipt_schemas", None)
    value = schemas.get(flag, ()) if isinstance(schemas, Mapping) else schemas or ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _emitted_order(flag_dict: Mapping[str, Any], argv: Sequence[str]) -> tuple[str, ...]:
    """Verify and normalize the compiler token stream without reparsing it heuristically."""

    emitted: list[str] = []
    cursor = 2
    for flag, value in flag_dict.items():
        if cursor >= len(argv):
            break
        expected = flag
        if value is False:
            expected = flag.replace("--", "--no-", 1)
        token = argv[cursor]
        normalized = flag if token == expected else token
        emitted.append(normalized)
        cursor += 1 if isinstance(value, bool) else 2
    if cursor != len(argv):
        emitted.extend(argv[cursor:])
    return tuple(emitted)


def collect_live_v9_bijection_snapshots(repo_root: Path | None = None) -> tuple[ConfigBijectionSnapshot, ...]:
    """Compile the closed V9 factory set and walk its real ownership graph."""

    from tac.witness_dsl.curriculum_dsl import TRAINER_REL, build_real_trainer_parser
    from tac.witness_dsl.spec_v9_cgauge import V9_CGAUGE_432_PROVENANCE, V9_CGAUGE_PROVENANCE

    root = Path(repo_root or _REPO_ROOT).resolve()
    trainer_path = root / TRAINER_REL
    parser = build_real_trainer_parser(trainer_path)
    consumers = _trainer_consumers(trainer_path, root)
    actions = {option: action for action in parser._actions for option in action.option_strings}
    snapshots: list[ConfigBijectionSnapshot] = []

    for program_name, factory in _canonical_factories().items():
        provenance = dict(V9_CGAUGE_PROVENANCE)
        if program_name == "v9_cgauge_432":
            provenance.update(V9_CGAUGE_432_PROVENANCE)
        launch = factory()
        program = launch.typed.to_program()
        flag_dict = program.flag_dict()
        argv, constants_manifest = program.compile_trainer_argv_with_constants(repo_root=root)
        compiler_records: dict[str, list[Mapping[str, Any]]] = {}
        for key, record in launch.constants_manifest.items():
            normalized = key if key.startswith("--") else f"--{key.replace('_', '-')}"
            compiler_records.setdefault(normalized, []).append(record)
        for key, record in constants_manifest.items():
            bucket = compiler_records.setdefault(key, [])
            if record not in bucket:
                bucket.append(record)
        namespace = parser.parse_args(argv[2:])
        bindings: list[FlagProvenanceBinding] = []
        for flag, raw_value in flag_dict.items():
            emitted_option = flag.replace("--", "--no-", 1) if raw_value is False else flag
            action = actions.get(emitted_option) or actions.get(flag)
            runtime_dest = getattr(action, "dest", None)
            runtime_value = getattr(namespace, runtime_dest) if runtime_dest else None
            owners = tuple(lever.name for lever in program.levers if flag in lever.overrides)
            refs = tuple(ref for lever in program.levers for ref in _lawrefs_for_flag(lever, flag))
            law_ids = tuple(str(getattr(ref, "equation_id", "")) for ref in refs)
            compiler_ids = tuple(str(record.get("equation_id", "")) for record in compiler_records.get(flag, ()))
            receipt_schemas = tuple(
                schema
                for lever in program.levers
                if flag in lever.overrides
                for schema in _receipt_schemas_for_flag(lever, flag)
            )
            raw_tokens = (emitted_option,) if isinstance(raw_value, bool) else (flag, str(raw_value))
            bindings.append(
                FlagProvenanceBinding(
                    flag=flag,
                    raw_value=_stable_value(raw_value),
                    raw_type=type(raw_value).__name__,
                    raw_tokens=raw_tokens,
                    runtime_dest=runtime_dest,
                    runtime_value=_stable_value(runtime_value),
                    runtime_type=type(runtime_value).__name__ if runtime_dest else None,
                    lever_owners=owners,
                    lawref_equation_ids=law_ids,
                    compiler_record_equation_ids=compiler_ids,
                    provenance_rung=(provenance.get(flag) or {}).get("rung"),
                    consumer_locations=consumers.get(runtime_dest or "", ()),
                    runtime_receipt_schemas=receipt_schemas,
                )
            )
        snapshots.append(
            ConfigBijectionSnapshot(
                program=program_name,
                bindings=tuple(bindings),
                semantic_order=tuple(flag_dict),
                emitted_order=_emitted_order(flag_dict, argv),
                lawref_flags=tuple(sorted(binding.flag for binding in bindings if binding.lawref_equation_ids)),
                compiler_record_flags=tuple(sorted(compiler_records)),
                provenance_flags=tuple(sorted(provenance)),
            )
        )
    return tuple(snapshots)


def _coverage_detail(expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    stale = sorted(actual - expected)
    return (
        f"missing_count={len(missing)} missing_sample={missing[:8]} stale_count={len(stale)} stale_sample={stale[:8]}"
    )


def audit_config_flag_provenance_bijection(
    snapshots: Sequence[ConfigBijectionSnapshot],
) -> list[str]:
    """Return every missing, stale, duplicate, reordered, or mismatched edge."""

    violations: list[str] = []
    for snapshot in snapshots:
        prefix = snapshot.program
        bindings = snapshot.bindings
        flags = tuple(binding.flag for binding in bindings)
        flag_set = set(flags)
        if len(flags) != len(flag_set):
            violations.append(f"{prefix}: duplicate semantic flags")
        if flags != snapshot.semantic_order:
            violations.append(f"{prefix}: binding order differs from semantic flag order")
        if snapshot.emitted_order != snapshot.semantic_order:
            violations.append(f"{prefix}: compiled argv flags are missing, stale, or reordered")
        if set(snapshot.lawref_flags) != flag_set:
            violations.append(
                f"{prefix}: LawRef coverage mismatch {_coverage_detail(flag_set, set(snapshot.lawref_flags))}"
            )
        if set(snapshot.compiler_record_flags) != flag_set:
            violations.append(
                f"{prefix}: compiler-record coverage mismatch "
                f"{_coverage_detail(flag_set, set(snapshot.compiler_record_flags))}"
            )
        if set(snapshot.provenance_flags) != flag_set:
            violations.append(
                f"{prefix}: provenance-table coverage mismatch "
                f"{_coverage_detail(flag_set, set(snapshot.provenance_flags))}"
            )
        for binding in bindings:
            bp = f"{prefix}:{binding.flag}"
            if not binding.raw_tokens or binding.raw_tokens[0] not in {
                binding.flag,
                binding.flag.replace("--", "--no-", 1),
            }:
                violations.append(f"{bp}: missing raw DSL token custody")
            if not binding.raw_type:
                violations.append(f"{bp}: missing raw DSL type custody")
            if not binding.runtime_dest or not binding.runtime_type:
                violations.append(f"{bp}: missing argparse-normalized runtime custody")
            elif not _semantically_equal(binding.raw_value, binding.runtime_value):
                violations.append(
                    f"{bp}: raw/runtime mismatch {binding.raw_type}={binding.raw_value!r} "
                    f"!= {binding.runtime_type}={binding.runtime_value!r}"
                )
            if len(binding.lever_owners) != 1:
                violations.append(f"{bp}: expected exactly one Lever owner, got {binding.lever_owners}")
            if len(binding.lawref_equation_ids) != 1 or not binding.lawref_equation_ids[0]:
                violations.append(
                    f"{bp}: expected exactly one Lever.constant_refs LawRef, got {binding.lawref_equation_ids}"
                )
            if len(binding.compiler_record_equation_ids) != 1:
                violations.append(
                    f"{bp}: expected exactly one canonical compiler record, got {binding.compiler_record_equation_ids}"
                )
            elif binding.compiler_record_equation_ids != binding.lawref_equation_ids:
                violations.append(f"{bp}: compiler record does not match owning LawRef")
            if binding.provenance_rung not in _RUNG_ALIASES:
                violations.append(f"{bp}: missing or unknown value-provenance rung {binding.provenance_rung!r}")
            if not binding.consumer_locations:
                violations.append(f"{bp}: no executable trainer-consumer location")
            if len(binding.runtime_receipt_schemas) != 1 or not binding.runtime_receipt_schemas[0]:
                violations.append(
                    f"{bp}: expected exactly one runtime receipt schema, got {binding.runtime_receipt_schemas}"
                )
    return violations


def check_config_flag_provenance_bijection_complete(
    repo_root: Path | None = None,
    *,
    strict: bool = False,
    verbose: bool = True,
    snapshots: Sequence[ConfigBijectionSnapshot] | None = None,
) -> list[str]:
    """Require a deterministic one-to-one V9 flag provenance graph."""

    first = tuple(snapshots) if snapshots is not None else collect_live_v9_bijection_snapshots(repo_root)
    violations = audit_config_flag_provenance_bijection(first)
    if snapshots is None:
        second = collect_live_v9_bijection_snapshots(repo_root)
        first_hashes = {item.program: item.bijection_hash for item in first}
        second_hashes = {item.program: item.bijection_hash for item in second}
        if first_hashes != second_hashes:
            violations.append(
                "identical V9 factory compilation produced non-deterministic bijection hashes: "
                f"{first_hashes} != {second_hashes}"
            )
    if violations and verbose:
        print(f"[v9-provenance-bijection] WARN: {len(violations)} violation(s)")
        for violation in violations[:_PRINT_LIMIT]:
            print(f"  - {violation}")
        if len(violations) > _PRINT_LIMIT:
            print(f"  - ... {len(violations) - _PRINT_LIMIT} additional violation(s)")
    elif verbose:
        hashes = {item.program: item.bijection_hash for item in first}
        print(f"[v9-provenance-bijection] OK: {len(first)} factories {hashes}")
    if strict and violations:
        raise V9ProvenanceGateError("V9 config provenance bijection incomplete:\n" + "\n".join(violations))
    return violations


def _valid_historical_waiver(waiver: str | None) -> bool:
    if not waiver or not waiver.startswith("HISTORICAL_NON_AUTHORIZING:"):
        return False
    reason = waiver.split(":", 1)[1].strip()
    return len(reason) >= 12 and reason.lower() not in {"todo", "tbd", "placeholder"}


def audit_v9_fake_claims(claims: Iterable[FakeClaim]) -> list[str]:
    """Refuse the four fake-claim families catalog #351 names."""

    violations: list[str] = []
    for claim in claims:
        prefix = claim.claim_id
        waiver_ok = _valid_historical_waiver(claim.waiver)
        label = (claim.active_basis_label or "").lower()
        implementation = (claim.basis_implementation or "").lower()
        if ("curvelet" in label or "shearlet" in label) and (
            claim.basis_is_spatially_localized is not True or "global" in implementation or "fourier" in implementation
        ):
            violations.append(
                f"{prefix}: ACTIVE {claim.active_basis_label!r} label is incompatible with "
                f"unlocalized implementation {claim.basis_implementation!r}"
            )
        if claim.pose_selected:
            if not claim.pose_receiver_authenticated:
                violations.append(f"{prefix}: selected-pose marker lacks authenticated receiver custody")
            if not _SHA256_RE.fullmatch(claim.pose_parseback_byte_effect_sha256 or ""):
                violations.append(f"{prefix}: selected-pose marker lacks parse-back byte-effect SHA-256")
        if (
            claim.numeric_d_pose is not None
            and (claim.d_pose_source or "").lower() != "live_posenet"
            and not (waiver_ok and not claim.pose_selected)
        ):
            violations.append(
                f"{prefix}: numeric d_pose={claim.numeric_d_pose} came from non-live source {claim.d_pose_source!r}"
            )
        if claim.self_orient_claim is not None:
            if claim.compiled_self_orient is None or claim.self_orient_claim != claim.compiled_self_orient:
                violations.append(
                    f"{prefix}: self-orient claim disagrees with compiled vehicle flag "
                    f"({claim.self_orient_claim!r} != {claim.compiled_self_orient!r})"
                )
            if claim.receipt_self_orient is None or claim.self_orient_claim != claim.receipt_self_orient:
                violations.append(
                    f"{prefix}: self-orient claim disagrees with scoped receipt "
                    f"({claim.self_orient_claim!r} != {claim.receipt_self_orient!r})"
                )
        if claim.claimed_percent_reduction is not None:
            if claim.receipt_vehicle != claim.vehicle:
                violations.append(
                    f"{prefix}: percentage claim vehicle {claim.vehicle!r} != receipt {claim.receipt_vehicle!r}"
                )
            if claim.receipt_scope not in {"full-n600-realized-through-R", "full-n600-byte-closed"}:
                violations.append(f"{prefix}: percentage claim lacks full-n600 realized-through-R receipt scope")
        if claim.waiver and not waiver_ok:
            violations.append(f"{prefix}: malformed or placeholder fake-claim waiver")
    return violations


def collect_live_v9_fake_claims(repo_root: Path | None = None) -> tuple[FakeClaim, ...]:
    """Derive basis/pose/self-orient claims from the live V9 compiled surface."""

    from tac.witness_control.pose_verdict_gate import check_pose_verdict_fallback_is_live_or_refused
    from tac.witness_dsl.basis_control import LEGACY_FOURIER_AB_CONTROL, normalize_basis_family

    root = Path(repo_root or _REPO_ROOT).resolve()
    basis_source = root / "src/tac/boundary_math/lever_b_levelset_generator.py"
    source = basis_source.read_text()
    global_fourier_alias = (
        bool(re.search(r"curvelet_directional_B\s*=\s*polar_directional_fourier_B", source))
        and "not spatially localized" in source
    )
    pose_receipt = check_pose_verdict_fallback_is_live_or_refused(gate_on=False)
    claims: list[FakeClaim] = []
    for program_name, factory in _canonical_factories().items():
        launch = factory()
        flags = launch.typed.to_program().flag_dict()
        compiled_basis = normalize_basis_family(flags.get("--basis", LEGACY_FOURIER_AB_CONTROL))
        localized = compiled_basis == "windowed_curvelet"
        claims.append(
            FakeClaim(
                claim_id=f"{program_name}:compiled-surface",
                vehicle=program_name,
                active_basis_label=compiled_basis,
                basis_implementation=(
                    "tac.boundary_math.windowed_curvelet_frame.WindowedCurveletConfig"
                    if localized
                    else (
                        "global_polar_directional_fourier_plane_waves"
                        if global_fourier_alias
                        else "source-unclassified"
                    )
                ),
                basis_is_spatially_localized=True if localized else (False if global_fourier_alias else None),
                pose_selected=False,
                d_pose_source=str(pose_receipt.get("score_path_d_pose_source", "live_posenet")),
                compiled_self_orient=bool(flags.get("--curvelet-self-orient", False)),
            )
        )
    return tuple(claims)


def check_v9_fake_claim_guards(
    repo_root: Path | None = None,
    *,
    strict: bool = False,
    verbose: bool = True,
    claims: Iterable[FakeClaim] | None = None,
) -> list[str]:
    """Run the executable V9 fake-claim guards; never trust a PASS marker."""

    checked = tuple(claims) if claims is not None else collect_live_v9_fake_claims(repo_root)
    violations = audit_v9_fake_claims(checked)
    if violations and verbose:
        print(f"[v9-fake-claims] WARN: {len(violations)} violation(s)")
        for violation in violations[:_PRINT_LIMIT]:
            print(f"  - {violation}")
        if len(violations) > _PRINT_LIMIT:
            print(f"  - ... {len(violations) - _PRINT_LIMIT} additional violation(s)")
    elif verbose:
        print(f"[v9-fake-claims] OK: {len(checked)} live claim surface(s)")
    if strict and violations:
        raise V9ProvenanceGateError("V9 fake claim guard failed:\n" + "\n".join(violations))
    return violations


def check_no_fourier_basis_in_witness_representation(
    repo_root: Path | None = None,
    *,
    strict: bool = False,
    verbose: bool = True,
    paths: Sequence[Path] | None = None,
) -> list[str]:
    """Warn on Fourier-as-witness-basis until the owed no-regression A/B exists.

    Strict mode is intentionally gated on an operator-GO n600 byte-closed,
    realized-through-R curvelet-vs-legacy-control verdict proving no d_seg
    regression.  Today this gate is apparatus only: it reports executable
    Fourier-basis debt and malformed FFT-tool waivers, but preflight wires it
    warn-only.
    """

    findings = audit_no_fourier_basis_in_witness_representation(repo_root, paths=paths)
    violations = [
        f"{item.path}:{item.line}: {item.kind} {item.signature!r}: {item.rationale}"
        for item in findings
    ]
    if violations and verbose:
        print(f"[no-fourier-basis] WARN: {len(violations)} violation(s)")
        for violation in violations[:_PRINT_LIMIT]:
            print(f"  - {violation}")
        if len(violations) > _PRINT_LIMIT:
            print(f"  - ... {len(violations) - _PRINT_LIMIT} additional violation(s)")
    elif verbose:
        print("[no-fourier-basis] OK: no executable Fourier witness-basis signatures")
    if strict and violations:
        raise V9ProvenanceGateError(
            "Fourier witness-basis structural ban is not clean:\n" + "\n".join(violations)
        )
    return violations


def _valid_advisory_waiver(waiver: str | None) -> bool:
    if not waiver or not waiver.startswith("ADVISORY_ONLY:"):
        return False
    reason = waiver.split(":", 1)[1].strip()
    return len(reason) >= 12 and reason.lower() not in {"todo", "tbd", "placeholder"}


def audit_evidence_authority_claims(claims: Iterable[EvidenceClaim]) -> list[str]:
    """Refuse exact-authority language unless every authority edge is custodied."""

    violations: list[str] = []
    for claim in claims:
        axis = claim.evidence_axis.strip().lower()
        language_requests_exact = bool(_EXACT_LANGUAGE_RE.search(claim.language))
        requests_exact = language_requests_exact or claim.requested_exact_authority or claim.score_claim
        waiver_ok = _valid_advisory_waiver(claim.waiver)
        if claim.waiver and not waiver_ok:
            violations.append(f"{claim.claim_id}: malformed or placeholder advisory waiver")
        if not requests_exact:
            continue
        custody: list[str] = []
        if axis not in _CONTEST_AXES:
            custody.append(f"axis={claim.evidence_axis!r}")
        if axis in _ADVISORY_AXES or "advisory" in axis:
            custody.append("advisory evidence")
        if not _SHA256_RE.fullmatch(claim.archive_sha256 or ""):
            custody.append("archive SHA-256")
        if not isinstance(claim.archive_bytes, int) or claim.archive_bytes <= 0:
            custody.append("positive archive bytes")
        if claim.evaluator != "upstream/evaluate.py":
            custody.append("frozen upstream/evaluate.py")
        if claim.pairs != 600:
            custody.append("full n600")
        if not claim.promotion_claim:
            custody.append("promotion_claim=true")
        if custody:
            suffix = " (waiver cannot grant exact authority)" if waiver_ok else ""
            violations.append(f"{claim.claim_id}: exact-authority language lacks {', '.join(custody)}{suffix}")
    return violations


def collect_live_v9_evidence_claims(repo_root: Path | None = None) -> tuple[EvidenceClaim, ...]:
    """Read the structured V9 F6 receipt as evidence, not as a trusted verdict."""

    root = Path(repo_root or _REPO_ROOT).resolve()
    receipt_path = root / ".omx/research/v9_cgauge_fake_extinction_audit_receipt_20260714.json"
    if not receipt_path.exists():
        return ()
    payload = json.loads(receipt_path.read_text())
    rows = payload.get("checks", payload.get("fake_class_audits", payload.get("audits", [])))
    claims: list[EvidenceClaim] = []
    for row in rows:
        if row.get("fake_id") != "F6_R1_EVIDENCE_AUTHORITY":
            continue
        evidence = row.get("evidence", {})
        claims.append(
            EvidenceClaim(
                claim_id="F6_R1_EVIDENCE_AUTHORITY",
                language=str(evidence.get("effective_label", row.get("ledger_claim", ""))),
                evidence_axis=str(evidence.get("measurement_axis", "macOS-CPU advisory")),
                archive_sha256=evidence.get("archive_zip_sha256"),
                archive_bytes=evidence.get("archive_zip_bytes"),
                evaluator="upstream/evaluate.py" if evidence.get("exact_eval_upstream_evaluate_ran") else None,
                pairs=evidence.get("pairs"),
                promotion_claim=bool(evidence.get("promotion_claim", False)),
                requested_exact_authority=bool(evidence.get("requested_exact_authority", False)),
                score_claim=False,
                waiver=(
                    f"ADVISORY_ONLY:{evidence['waiver_effective_label']}"
                    if evidence.get("waiver_effective_label")
                    else None
                ),
            )
        )
    return tuple(claims)


def check_evidence_authority_claims_are_custodied(
    repo_root: Path | None = None,
    *,
    strict: bool = False,
    verbose: bool = True,
    claims: Iterable[EvidenceClaim] | None = None,
) -> list[str]:
    """Require complete contest custody for every exact-authority statement."""

    checked = tuple(claims) if claims is not None else collect_live_v9_evidence_claims(repo_root)
    violations = audit_evidence_authority_claims(checked)
    if violations and verbose:
        print(f"[v9-evidence-authority] WARN: {len(violations)} violation(s)")
        for violation in violations[:_PRINT_LIMIT]:
            print(f"  - {violation}")
        if len(violations) > _PRINT_LIMIT:
            print(f"  - ... {len(violations) - _PRINT_LIMIT} additional violation(s)")
    elif verbose:
        print(f"[v9-evidence-authority] OK: {len(checked)} structured evidence claim(s)")
    if strict and violations:
        raise V9ProvenanceGateError("V9 evidence authority claim lacks custody:\n" + "\n".join(violations))
    return violations


__all__ = [
    "ConfigBijectionSnapshot",
    "EvidenceClaim",
    "FakeClaim",
    "FlagProvenanceBinding",
    "FourierBasisAuditViolation",
    "V9ProvenanceGateError",
    "audit_config_flag_provenance_bijection",
    "audit_evidence_authority_claims",
    "audit_no_fourier_basis_in_witness_representation",
    "audit_v9_fake_claims",
    "check_config_flag_provenance_bijection_complete",
    "check_evidence_authority_claims_are_custodied",
    "check_no_fourier_basis_in_witness_representation",
    "check_v9_fake_claim_guards",
    "collect_live_v9_bijection_snapshots",
    "collect_live_v9_evidence_claims",
    "collect_live_v9_fake_claims",
]
