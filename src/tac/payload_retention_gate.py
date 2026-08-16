"""Payload-retention gate — refuse the measure-and-discard anti-pattern.

Operator binding 2026-08-09 (P0, DEF CON 1000): *"You should be constantly keeping
payloads. You shouldn't be running anything that doesn't keep the payload."*

THE DEFECT (anchor: ``ans_real_n600.py`` on the PR130 base, 2026-08-09)::

    rng = len(enc.get_compressed().tobytes()); del enc     # :37  range payload
    an  = len(ans.get_compressed().tobytes())              # :41  ANS payload
    open(D + '/ans_vs_range_n600_result.json', 'w').write(json.dumps(res))   # :47  scalars only

681 s of encode over n600; both coder payloads discarded; only lengths survived. Cost:
two full re-encodes to recreate bytes already produced once — and the recovered ANS
payload then measured -2,120 B against the shipped range coder, so the discard also
delayed a real rate win.

THE SIGNATURE IS NOT ``del``. ``del`` was incidental (line 41 has none and is equally
defective). The signature is: **a payload-producing expression is reduced to a scalar
(``len(...)``) and the producing object never reaches a byte-persisting call.**

DETECTOR-DOES-NOT-ZERO-ON-THE-CURE (per the standing law): applying the cure — adding a
real byte-write of the payload — is exactly what clears the finding, and adding a write
of some *unrelated* object does not, because persistence is matched against the ROOT
BINDING of the measured expression, not against "any write anywhere".

SCOPE OF THE DUTY (2026-08-16 census correction, ddm_kp2). The rule binds a **run** that
materializes a payload, not every expression that touches one. A scope that hands its
bytes onward — ``return blob``, ``out.write(blob)`` into any handle or ``io.BytesIO``
container, ``parts.append(blob)`` later joined and written — has not discarded anything;
the persistence duty moved to the caller. The first census measured 1,824 findings over
890 files, of which 1,219 (66.8%) were this shape: 530 returned as bytes, 543 flowed into
another binding, 146 written to a handle the gate did not model (the commonest single
idiom being ``out.write(struct.pack("<I", len(blob))); out.write(blob)`` — a length
PREFIX in a container format, i.e. correct codec design flagged as a discard). A gate
firing 2-in-3 false is unactionable and trains readers to ignore it, so escape is now
modelled explicitly.

The distinction that keeps the gate strong is REDUCTION: a name reaching a sink as bytes
escapes; a name reduced to a scalar first (``len(blob)``, ``sha256(blob).hexdigest()``)
does not. That is exactly why the anchor stays caught — its surviving artifact is
``json.dumps({"range": rng, ...})`` where ``rng`` is an int, never the bytes.

KNOWN RESIDUAL FALSE NEGATIVE (stated, not hidden). ``.write(...)`` on an in-memory
buffer counts as escape without proving the buffer itself is later persisted, so
``buf = io.BytesIO(); buf.write(blob); print(len(buf.getvalue()))`` reads as retained
when it is not. Modelling that would require whole-program flow; the trade was taken
deliberately because the same relaxation removes 146 measured false positives on the
dominant, correct ``length-prefix + payload into a returned container`` idiom. The gate
is therefore SOUND-LEANING-PERMISSIVE at the container boundary and exact at the scalar
boundary, which is where the anchor's defect lives.

Waiver: same-line ``# MEASURE_ONLY_OK:<rationale>`` for a genuine scalar-only probe.
Placeholder rationales (``<rationale>``, ``TBD``, ``reason``) are rejected, per the
Catalog #287 sister discipline.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BYTE_PERSISTERS",
    "PAYLOAD_PRODUCERS",
    "WAIVER_TOKEN",
    "PayloadDiscardFinding",
    "PayloadRetentionPopulation",
    "audit_measure_and_discard_payload",
    "check_no_measure_and_discard_payload",
    "scan_paths",
    "scan_source",
]

#: Calls whose result is a real byte payload. Deliberately NARROW: these are
#: compressed/serialized byte producers, not generic ``encode``/``dumps`` (which would
#: sweep in ordinary string work and drown the signal).
PAYLOAD_PRODUCERS: frozenset[str] = frozenset(
    {
        "get_compressed",  # constriction RangeEncoder / AnsCoder
        "tobytes",         # numpy / memoryview materialization
        "tofile",          # numpy direct write (also a persister; see below)
        "compress",        # brotli / zlib / lzma / bz2 / zstd
        "flush",           # incremental compressobj finalization
        "getbuffer",       # io.BytesIO
        "serialize",       # our codec surfaces
        "pack_model",      # tm1-style packers
        "build_archive",   # archive builders
    }
)

#: Calls that durably persist bytes.
#:
#: NOTE the deliberate absence of bare ``write``. A caught false negative (2026-08-09,
#: by this module's own test): treating every ``.write(...)`` as persistence let
#: ``open(path, 'w').write(str(len(payload)))`` — line 47 of the ANCHOR ITSELF, a TEXT
#: write of a scalar — launder the discarded payload. ``write`` now only clears a root
#: when the module also opens a handle in binary write mode (see ``_persisted_roots``
#: and ``_binary_write_targets``).
BYTE_PERSISTERS: frozenset[str] = frozenset(
    {
        "write_bytes",  # pathlib.Path
        "tofile",       # numpy
        "save",         # np.save / torch.save
        "savez",
        "savez_compressed",
        "dump",         # pickle.dump / joblib.dump
        "writestr",     # zipfile, explicit bytes
        "retain_payload",  # canonical/local checked retention helper
    }
)

WAIVER_TOKEN = "MEASURE_ONLY_OK"

_PLACEHOLDER_RATIONALES = frozenset(
    {"", "<rationale>", "<reason>", "rationale", "reason", "tbd", "todo", "placeholder", "n/a"}
)


@dataclass(frozen=True)
class PayloadDiscardFinding:
    """One measure-and-discard site."""

    path: str
    line: int
    root: str
    producer: str
    snippet: str
    payload_line: int
    payload_column: int

    def render(self) -> str:
        return (
            f"{self.path}:{self.line}: payload materialized at "
            f"{self.payload_line}:{self.payload_column + 1} by "
            f"`{self.root}.{self.producer}(...)` is reduced "
            f"to a scalar and never persisted — `{self.snippet}`.\n"
            f"    RULE: CLAUDE.md 'ALWAYS KEEP THE PAYLOAD' (P0, operator 2026-08-09). Every run "
            f"that materializes a payload MUST persist it; a scalar-only artifact is forbidden.\n"
            f"    FIX: write the bytes to the SSD tier "
            f"(/Volumes/VertigoDataTier/pact/<arm>/retained/) and record sha256 + byte count in "
            f"the result JSON, e.g. `Path(out).write_bytes(payload)`.\n"
            f"    WAIVE (genuine scalar-only probe only): append "
            f"`# {WAIVER_TOKEN}:<substantive rationale>` on line {self.line}."
        )


@dataclass(frozen=True)
class PayloadRetentionPopulation:
    """One bounded census of the gate's static detector population."""

    findings: tuple[PayloadDiscardFinding, ...]
    python_files_discovered: int
    python_files_examined: int
    candidate_files_parsed: int
    unreadable_files: int

    @property
    def files_with_findings(self) -> int:
        return len({finding.path for finding in self.findings})


def _waiver_rationale(line_text: str) -> str | None:
    """Return the waiver rationale on this source line, or None if unwaived."""
    marker = f"# {WAIVER_TOKEN}:"
    idx = line_text.find(marker)
    if idx < 0:
        # Tolerate `#MEASURE_ONLY_OK:` without the space.
        marker = f"#{WAIVER_TOKEN}:"
        idx = line_text.find(marker)
        if idx < 0:
            return None
    return line_text[idx + len(marker) :].strip()


def _waiver_is_valid(rationale: str | None) -> bool:
    if rationale is None:
        return False
    return rationale.strip().lower() not in _PLACEHOLDER_RATIONALES and len(rationale.strip()) >= 4


def _root_binding(node: ast.AST) -> str | None:
    """Walk an attribute/call chain down to its root ``Name``.

    ``enc.get_compressed().tobytes()`` -> ``"enc"``; ``brotli.compress(x)`` -> ``"brotli"``.
    """
    cur: ast.AST = node
    while True:
        if isinstance(cur, ast.Call):
            cur = cur.func
        elif isinstance(cur, (ast.Attribute, ast.Subscript)):
            cur = cur.value
        elif isinstance(cur, ast.Name):
            return cur.id
        else:
            return None


@dataclass(frozen=True)
class _PayloadCall:
    root: str
    producer: str
    line: int
    column: int

    @property
    def identity(self) -> tuple[int, int, str]:
        return (self.line, self.column, self.producer)


def _producer_name(node: ast.Call) -> str | None:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else (
        func.id if isinstance(func, ast.Name) else None
    )
    return name if name in PAYLOAD_PRODUCERS else None


def _payload_call(node: ast.Call) -> _PayloadCall | None:
    """Describe one payload-producing call without counting its ingredients."""
    name = _producer_name(node)
    if name is None:
        return None
    root = _root_binding(node.func)
    if name in {"compress", "serialize", "pack_model", "build_archive"} and node.args:
        arg_root = _root_binding(node.args[0])
        if arg_root is not None:
            root = arg_root
    if root is None:
        root = name
    return _PayloadCall(
        root=root,
        producer=name,
        line=getattr(node, "lineno", 0),
        column=getattr(node, "col_offset", 0),
    )


def _outer_payload_calls(node: ast.AST) -> list[_PayloadCall]:
    """Return the outer payload(s) reduced by an expression.

    A producer consumes any nested producers into ONE new payload.  Therefore
    ``zlib.compress(q.tobytes() + scale.tobytes())`` is one zlib payload, not
    three findings.  Sibling producers under a non-producer expression remain
    distinct payloads.
    """
    if isinstance(node, ast.Call):
        payload = _payload_call(node)
        if payload is not None:
            return [payload]
    found: list[_PayloadCall] = []
    for child in ast.iter_child_nodes(node):
        found.extend(_outer_payload_calls(child))
    return found


#: Calls that reduce a payload to a scalar. A name that reaches a sink only through one
#: of these has NOT escaped as bytes — this is the whole hinge of the escape model, and
#: what keeps ``open(p, 'w').write(str(len(blob)))`` from laundering a discard.
_SCALAR_REDUCERS: frozenset[str] = frozenset(
    {
        "len",
        "print",
        "repr",
        "str",
        "format",
        "hash",
        "id",
        "bool",
        "int",
        "float",
        "type",
        "sha256",
        "sha1",
        "md5",
        "blake2b",
        "hexdigest",
        "digest",
        "entropy",
    }
)


def _unreduced_names(node: ast.AST) -> set[str]:
    """Names appearing in ``node`` that are NOT wrapped in a scalar reduction.

    ``out.write(blob)`` -> ``{"out", "blob"}``; ``len(blob)`` -> ``set()``. Callers use
    this to ask "did the bytes themselves reach this sink, or only a number about them?"
    """
    out: set[str] = set()

    def walk(node_: ast.AST, reduced: bool) -> None:
        if isinstance(node_, ast.Name):
            if not reduced:
                out.add(node_.id)
            return
        child_reduced = reduced
        if isinstance(node_, ast.Call):
            func = node_.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _SCALAR_REDUCERS:
                child_reduced = True
        for child in ast.iter_child_nodes(node_):
            walk(child, child_reduced)

    walk(node, False)
    return out


def _bound_payload_calls(node: ast.AST) -> list[_PayloadCall]:
    """Payload calls in an assignment RHS that survive as bytes.

    Distinct from :func:`_outer_payload_calls`, which is applied to a ``len()`` argument
    and must see through the reduction. Here the reduction is decisive: ``rng =
    len(enc.get_compressed().tobytes())`` binds an INT, so ``rng`` must not be recorded
    as a payload carrier — otherwise a later ``write(json.dumps({"n": rng}))`` would read
    as persistence and clear the anchor itself.
    """
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in _SCALAR_REDUCERS:
            return []
        payload = _payload_call(node)
        if payload is not None:
            return [payload]
    found: list[_PayloadCall] = []
    for child in ast.iter_child_nodes(node):
        found.extend(_bound_payload_calls(child))
    return found


_LEXICAL_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _position(node: ast.AST) -> tuple[int, int]:
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def _scope_nodes(scope: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    """Nodes owned by one lexical scope plus its immediately nested scopes.

    Module-wide binding was the original gate's second counting bug: a local named
    ``blob`` in function A could be joined to ``len(blob)`` in function B.  Walking
    each lexical scope independently keeps a materialization tied to the name that can
    actually reference it.
    """
    stack = (
        [scope.body]
        if isinstance(scope, ast.Lambda)
        else list(reversed(getattr(scope, "body", ())))
    )
    owned: list[ast.AST] = []
    nested: list[ast.AST] = []
    while stack:
        node = stack.pop()
        if isinstance(node, _LEXICAL_SCOPES):
            nested.append(node)
            continue
        owned.append(node)
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return owned, nested


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _latest_binding(
    bindings: dict[str, list[tuple[tuple[int, int], list[_PayloadCall]]]],
    name: str,
    before: tuple[int, int],
) -> list[_PayloadCall]:
    eligible = [entry for entry in bindings.get(name, ()) if entry[0] < before]
    if not eligible:
        return []
    # One statement can bind several payloads at the SAME position — its own
    # (``comp = lzma.compress(body)``) plus everything it inherits from its operands.
    # Taking a single ``max`` entry dropped the inherited half and re-reported retained
    # bytes as discarded, so the newest position contributes ALL of its payloads.
    newest = max(position for position, _ in eligible)
    return [
        payload
        for position, payloads in eligible
        if position == newest
        for payload in payloads
    ]


def _scope_bindings(
    nodes: Sequence[ast.AST],
) -> dict[str, list[tuple[tuple[int, int], list[_PayloadCall]]]]:
    bindings: dict[str, list[tuple[tuple[int, int], list[_PayloadCall]]]] = {}
    for node in nodes:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        payloads = _bound_payload_calls(node.value)
        if payloads:
            bindings.setdefault(target.id, []).append((_position(node), payloads))
    _propagate_carriers(nodes, bindings)
    return bindings


_CARRIER_APPENDS = frozenset({"append", "extend", "add", "insert"})


def _container_names(nodes: Sequence[ast.AST]) -> set[str]:
    """Names used as accumulators (``parts.append(...)``, ``buf.extend(...)``)."""
    names: set[str] = set()
    for node in nodes:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _CARRIER_APPENDS
        ):
            container = _root_binding(node.func.value)
            if container is not None:
                names.add(container)
    return names


def _resolve_payloads(
    bindings: dict[str, list[tuple[tuple[int, int], list[_PayloadCall]]]],
    containers: set[str],
    name: str,
    before: tuple[int, int],
) -> list[_PayloadCall]:
    """Payloads ``name`` holds at ``before``.

    Rebinding replaces (``x = a; x = b`` -> only ``b``), but an accumulator ACCUMULATES:
    ``body_chunks`` filled by six appends carries all six payloads into
    ``b"".join(body_chunks)``. Taking only the latest entry there dropped five of six
    retained payloads on the floor and re-reported them as discards (measured on
    ``src/tac/renderer_export.py``: 22 such findings, every one of them retained).
    """
    if name not in containers:
        return _latest_binding(bindings, name, before)
    return [
        payload
        for position, payloads in bindings.get(name, ())
        if position < before
        for payload in payloads
    ]


def _record(
    bindings: dict[str, list[tuple[tuple[int, int], list[_PayloadCall]]]],
    name: str,
    position: tuple[int, int],
    payloads: Sequence[_PayloadCall],
) -> bool:
    """Attach payloads to ``name`` at ``position``; True when anything was new."""
    entries = bindings.setdefault(name, [])
    known = {payload.identity for _, existing in entries for payload in existing}
    fresh = [payload for payload in payloads if payload.identity not in known]
    if not fresh:
        return False
    entries.append((position, fresh))
    return True


def _propagate_carriers(
    nodes: Sequence[ast.AST],
    bindings: dict[str, list[tuple[tuple[int, int], list[_PayloadCall]]]],
) -> None:
    """Follow a payload through the names that carry it onward, to a fixpoint.

    ``blob = brotli.compress(raw)`` / ``parts.append(blob)`` / ``packet = b"".join(parts)``
    / ``Path(p).write_bytes(packet)`` is ONE retained payload across four statements. The
    dedup on payload identity bounds the iteration.
    """
    containers = _container_names(nodes)
    for _ in range(len(nodes) + 1):
        changed = False
        for node in nodes:
            position = _position(node)
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                inherited: list[_PayloadCall] = []
                for name in _unreduced_names(node.value):
                    if name != target.id:
                        inherited.extend(_resolve_payloads(bindings, containers, name, position))
                if inherited:
                    changed |= _record(bindings, target.id, position, inherited)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _CARRIER_APPENDS
            ):
                container = _root_binding(node.func.value)
                if container is None:
                    continue
                inherited = []
                for arg in node.args:
                    for name in _unreduced_names(arg):
                        inherited.extend(_resolve_payloads(bindings, containers, name, position))
                if inherited:
                    changed |= _record(bindings, container, position, inherited)
        if not changed:
            return


def _escaped_payloads(
    nodes: Sequence[ast.AST],
    bindings: dict[str, list[tuple[tuple[int, int], list[_PayloadCall]]]],
) -> set[tuple[int, int, str]]:
    """Materialization identities whose BYTES leave this scope.

    Three sinks discharge the scope's retention duty:

    * a byte persister (``write_bytes``, ``np.save``, ``retain_payload``, ...);
    * ``.write(bytes)`` on any handle — a file opened without ``with``, an
      ``io.BytesIO`` archive container, ``sys.stdout.buffer``;
    * ``return`` / ``yield`` of the bytes, which hands the duty to the caller.

    A name that arrives at a sink only under a scalar reduction has NOT escaped, so
    ``open(p, 'w').write(json.dumps({"n": len(blob)}))`` still leaves ``blob`` discarded.
    """
    escaped: set[tuple[int, int, str]] = set()
    containers = _container_names(nodes)
    for node in nodes:
        args: list[ast.AST]
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in BYTE_PERSISTERS:
                args = list(node.args) + [keyword.value for keyword in node.keywords]
            elif name == "write" and isinstance(node.func, ast.Attribute):
                args = list(node.args)
            else:
                continue
        elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
            if node.value is None:
                continue
            args = [node.value]
        else:
            continue
        for arg in args:
            for touched in _unreduced_names(arg):
                escaped.update(
                    payload.identity
                    for payload in _resolve_payloads(
                        bindings, containers, touched, _position(node)
                    )
                )
    return escaped


def scan_source(source: str, path: str = "<string>") -> list[PayloadDiscardFinding]:
    """Findings for one module's source text."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # not our bug class

    lines = source.splitlines()
    findings: list[PayloadDiscardFinding] = []
    seen: set[tuple[int, int, str]] = set()
    scopes: list[ast.AST] = [tree]
    while scopes:
        nodes, nested = _scope_nodes(scopes.pop())
        scopes.extend(nested)
        bindings = _scope_bindings(nodes)
        persisted = _escaped_payloads(nodes, bindings)
        for node in nodes:
            # The defect shape: len(<payload expression>) — inline OR via a bound name.
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"
                and node.args
            ):
                continue
            arg = node.args[0]
            payloads = _outer_payload_calls(arg)
            if not payloads and isinstance(arg, ast.Name):
                payloads = [
                    _PayloadCall(
                        root=arg.id,
                        producer=payload.producer,
                        line=payload.line,
                        column=payload.column,
                    )
                    for payload in _latest_binding(bindings, arg.id, _position(node))
                ]
            for payload in payloads:
                if payload.identity in persisted or payload.identity in seen:
                    continue
                seen.add(payload.identity)
                line_no = getattr(node, "lineno", 0)
                line_text = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
                if _waiver_is_valid(_waiver_rationale(line_text)):
                    continue
                findings.append(
                    PayloadDiscardFinding(
                        path=path,
                        line=line_no,
                        root=payload.root,
                        producer=payload.producer,
                        snippet=line_text.strip()[:160],
                        payload_line=payload.line,
                        payload_column=payload.column,
                    )
                )
    return findings


def scan_paths(paths: Iterable[Path]) -> list[PayloadDiscardFinding]:
    """Findings across a set of ``.py`` files. Unreadable files are skipped."""
    out: list[PayloadDiscardFinding] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        out.extend(scan_source(text, str(p)))
    return out


def _scan_population(
    paths: Iterable[Path],
    *,
    candidate_paths: set[Path] | None = None,
) -> PayloadRetentionPopulation:
    findings: list[PayloadDiscardFinding] = []
    discovered = 0
    examined = 0
    parsed = 0
    unreadable = 0
    producer_needles = tuple(PAYLOAD_PRODUCERS)
    for path in paths:
        discovered += 1
        examined += 1
        if candidate_paths is not None and path not in candidate_paths:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            unreadable += 1
            continue
        # Cheap lossless prefilter: the AST detector cannot fire without a len call
        # and one of the deliberately narrow producer names.  This keeps preflight
        # under its wall-clock budget while preserving the census denominator.
        if "len" not in source or not any(needle in source for needle in producer_needles):
            continue
        parsed += 1
        findings.extend(scan_source(source, str(path)))
    return PayloadRetentionPopulation(
        findings=tuple(findings),
        python_files_discovered=discovered,
        python_files_examined=examined,
        candidate_files_parsed=parsed,
        unreadable_files=unreadable,
    )


def _rg_paths(args: Sequence[str]) -> set[Path] | None:
    try:
        completed = subprocess.run(
            ["rg", *args],
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if completed.returncode not in {0, 1}:
        return None
    return {
        Path(raw.decode("utf-8", errors="surrogateescape"))
        for raw in completed.stdout.split(b"\0")
        if raw
    }


def _rg_population_paths(
    roots: Sequence[Path],
    *,
    excluded_parts: Sequence[str] = (),
) -> tuple[set[Path], set[Path]] | None:
    """Use ripgrep's indexed walker to enumerate and prefilter a large corpus."""
    exclusions = (
        "-g", "*.py",
        "-g", "!**/.git/**",
        "-g", "!**/.venv/**",
        "-g", "!**/node_modules/**",
        "-g", "!**/__pycache__/**",
        "-g", "!**/upstream/**",
        "-g", "!**/*_intake_*/**",
        *(item for part in excluded_parts for item in ("-g", f"!**/{part}/**")),
    )
    root_args = tuple(str(root.resolve()) for root in roots if root.exists())
    if not root_args:
        return set(), set()
    all_paths = _rg_paths(("--files", "-0", "-uu", *exclusions, *root_args))
    len_paths = _rg_paths(("-l", "-0", "-uu", *exclusions, r"\blen\s*\(", *root_args))
    producer_pattern = rf"\b(?:{'|'.join(re.escape(name) for name in PAYLOAD_PRODUCERS)})\s*\("
    producer_paths = _rg_paths(
        ("-l", "-0", "-uu", *exclusions, producer_pattern, *root_args)
    )
    if all_paths is None or len_paths is None or producer_paths is None:
        return None
    candidates = all_paths & len_paths & producer_paths
    return all_paths, candidates


def _iter_python(
    roots: Sequence[Path],
    *,
    excluded_parts: Sequence[str] = (),
) -> Iterator[Path]:
    skip = {".git", ".venv", "node_modules", "__pycache__", "upstream", *excluded_parts}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in skip or "_intake_" in part for part in p.parts):
                continue
            yield p


def check_no_measure_and_discard_payload(
    repo_root: Path | str = ".",
    strict: bool = False,
    roots: Sequence[str] = ("experiments", "tools", "src/tac", "scripts"),
) -> list[PayloadDiscardFinding]:
    """Refuse the measure-and-discard payload class.

    Returns the findings. In ``strict`` mode raises ``RuntimeError`` when any survive.
    """
    report = audit_measure_and_discard_payload(repo_root=repo_root, roots=roots)
    findings = list(report.findings)
    if strict and findings:
        detail = "\n".join(f.render() for f in findings)
        raise RuntimeError(
            f"ALWAYS KEEP THE PAYLOAD (P0, operator 2026-08-09): "
            f"{len(findings)} measure-and-discard site(s) found.\n{detail}"
        )
    return findings


def audit_measure_and_discard_payload(
    repo_root: Path | str = ".",
    roots: Sequence[str] = ("experiments", "tools", "src/tac", "scripts"),
    excluded_parts: Sequence[str] = (),
) -> PayloadRetentionPopulation:
    """Return findings and explicit file denominators for one bounded census."""
    base = Path(repo_root)
    scope_roots = [base / root for root in roots]
    indexed = _rg_population_paths(scope_roots, excluded_parts=excluded_parts)
    if indexed is not None:
        paths, candidates = indexed
        return _scan_population(sorted(paths), candidate_paths=candidates)
    return _scan_population(_iter_python(scope_roots, excluded_parts=excluded_parts))
