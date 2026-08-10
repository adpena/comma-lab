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

Waiver: same-line ``# MEASURE_ONLY_OK:<rationale>`` for a genuine scalar-only probe.
Placeholder rationales (``<rationale>``, ``TBD``, ``reason``) are rejected, per the
Catalog #287 sister discipline.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = [
    "PayloadDiscardFinding",
    "PAYLOAD_PRODUCERS",
    "BYTE_PERSISTERS",
    "WAIVER_TOKEN",
    "scan_source",
    "scan_paths",
    "check_no_measure_and_discard_payload",
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

    def render(self) -> str:
        return (
            f"{self.path}:{self.line}: payload from `{self.root}.{self.producer}(...)` is reduced "
            f"to a scalar and never persisted — `{self.snippet}`.\n"
            f"    RULE: CLAUDE.md 'ALWAYS KEEP THE PAYLOAD' (P0, operator 2026-08-09). Every run "
            f"that materializes a payload MUST persist it; a scalar-only artifact is forbidden.\n"
            f"    FIX: write the bytes to the SSD tier "
            f"(/Volumes/VertigoDataTier/pact/<arm>/retained/) and record sha256 + byte count in "
            f"the result JSON, e.g. `Path(out).write_bytes(payload)`.\n"
            f"    WAIVE (genuine scalar-only probe only): append "
            f"`# {WAIVER_TOKEN}:<substantive rationale>` on line {self.line}."
        )


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
        elif isinstance(cur, ast.Attribute):
            cur = cur.value
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        elif isinstance(cur, ast.Name):
            return cur.id
        else:
            return None


def _producers_in(node: ast.AST) -> list[tuple[str, str]]:
    """Every (root_binding, producer_name) pair reachable inside ``node``."""
    found: list[tuple[str, str]] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )
        if name not in PAYLOAD_PRODUCERS:
            continue
        # For `brotli.compress(payload)` the interesting root is the ARGUMENT (the
        # payload), not the module. Prefer an argument root when the callee root is a
        # bare module-looking name and an argument resolves to a Name.
        root = _root_binding(func)
        if name in {"compress", "serialize", "pack_model", "build_archive"} and sub.args:
            arg_root = _root_binding(sub.args[0])
            if arg_root is not None:
                root = arg_root
        if root is not None:
            found.append((root, name))
    return found


def _names_touched(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _persisted_roots(tree: ast.AST) -> set[str]:
    """Root bindings that reach a byte-persisting call anywhere in the module.

    Deliberately module-scoped (not flow-sensitive): a persistence anywhere clears the
    binding. False negatives here are acceptable; false positives are not, because this
    gate is STRICT-eligible and must never refuse an honest run.
    """
    persisted: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )

        # `open(path, 'wb').write(payload)` / `with open(path,'wb') as f: f.write(payload)`
        if name == "open":
            mode = ""
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if "b" in mode and ("w" in mode or "a" in mode or "x" in mode):
                # Binary-write handle: everything written through it counts. Approximate
                # by clearing every root that appears in a `.write(...)` in the module.
                persisted.add("__BINARY_HANDLE__")
            continue

        if name not in BYTE_PERSISTERS:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            persisted |= _names_touched(arg)
        # `payload.write_bytes(...)`-style: also clear the receiver.
        if isinstance(func, ast.Attribute):
            recv = _root_binding(func.value)
            if recv is not None:
                persisted.add(recv)
    return persisted


def _binary_write_targets(tree: ast.AST) -> set[str]:
    """Names passed to any ``.write(...)`` — paired with a binary ``open`` elsewhere."""
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "write":
                for arg in node.args:
                    targets |= _names_touched(arg)
    return targets


def scan_source(source: str, path: str = "<string>") -> list[PayloadDiscardFinding]:
    """Findings for one module's source text."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # not our bug class

    lines = source.splitlines()
    persisted = _persisted_roots(tree)
    if "__BINARY_HANDLE__" in persisted:
        persisted |= _binary_write_targets(tree)

    # BIND-THEN-MEASURE: `payload = enc.get_compressed().tobytes()` ... `len(payload)`.
    # Without this map the detector only saw the inline `len(<producer expr>)` form and
    # was blind to the more common two-line shape (caught by this module's own test,
    # 2026-08-09). The bound NAME becomes the tracked root, because that is the name a
    # cure would persist.
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                produced = _producers_in(node.value)
                if produced:
                    bound[target.id] = produced[0][1]

    findings: list[PayloadDiscardFinding] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        # The defect shape: len(<payload expression>) — inline OR via a bound name.
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "len" and node.args):
            continue
        arg = node.args[0]
        pairs = _producers_in(arg)
        if not pairs and isinstance(arg, ast.Name) and arg.id in bound:
            pairs = [(arg.id, bound[arg.id])]
        for root, producer in pairs:
            if root in persisted:
                continue  # the cure was applied — detector correctly goes quiet
            line_no = getattr(node, "lineno", 0)
            # ONE discarded payload = ONE finding, keyed on the root binding, not on the
            # number of `len()` sites that measure it. Counting per-site would report one
            # fact N times (the #821 law); the population is payloads, not call sites.
            if root in seen:
                continue
            seen.add(root)
            line_text = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
            if _waiver_is_valid(_waiver_rationale(line_text)):
                continue
            findings.append(
                PayloadDiscardFinding(
                    path=path,
                    line=line_no,
                    root=root,
                    producer=producer,
                    snippet=line_text.strip()[:160],
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


def _iter_python(roots: Sequence[Path]) -> Iterator[Path]:
    skip = {".git", ".venv", "node_modules", "__pycache__", "upstream"}
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
    base = Path(repo_root)
    findings = scan_paths(_iter_python([base / r for r in roots]))
    if strict and findings:
        detail = "\n".join(f.render() for f in findings)
        raise RuntimeError(
            f"ALWAYS KEEP THE PAYLOAD (P0, operator 2026-08-09): "
            f"{len(findings)} measure-and-discard site(s) found.\n{detail}"
        )
    return findings
