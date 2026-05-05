#!/usr/bin/env python3
"""Orphan .pyc -> .py recovery harness, v3.

Differences from v2:
  - For files where pycdc output is not AST-parseable, write a STUB that DOES
    parse, with the original pycdc output preserved inside an r-string. This
    is necessary because the global pre-commit preflight scans every .py file
    for SyntaxError, even ones not in the commit.
  - For 314 pycs that pycdc cannot read, write a stub pointing to the spec.
"""
from __future__ import annotations

import ast
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path("/Users/adpena/Projects/pact")
PYCDC = "/tmp/pycdc/pycdc"
PY312 = "/Users/adpena/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12"
PY313 = "/Users/adpena/.local/share/uv/python/cpython-3.13-macos-aarch64-none/bin/python3.13"
PY314 = "/Users/adpena/.local/share/uv/python/cpython-3.14-macos-aarch64-none/bin/python3.14"

ALREADY_RECOVERED = {
    "scripts/probe_fastest_chip.py",
    "tools/argparse_dryrun.py",
    "tools/lane_magic_registry.py",
}


def find_pyc(orphan_rel: str) -> Optional[tuple[Path, str]]:
    p = REPO_ROOT / orphan_rel
    parent = p.parent
    modname = p.stem
    cache_dir = parent / "__pycache__"
    if not cache_dir.exists():
        return None
    candidates = [
        (cache_dir / f"{modname}.cpython-312.pyc", "312"),
        (cache_dir / f"{modname}.cpython-313.pyc", "313"),
        (cache_dir / f"{modname}.cpython-314.pyc", "314"),
    ]
    for pytest_ver in ("9.0.3",):
        candidates.append((cache_dir / f"{modname}.cpython-312-pytest-{pytest_ver}.pyc", "pytest-312"))
        candidates.append((cache_dir / f"{modname}.cpython-313-pytest-{pytest_ver}.pyc", "pytest-313"))
        candidates.append((cache_dir / f"{modname}.cpython-314-pytest-{pytest_ver}.pyc", "pytest-314"))
    for pyc, tag in candidates:
        if pyc.exists():
            return pyc, tag
    return None


def py_for_tag(tag: str) -> str:
    if "312" in tag:
        return PY312
    if "313" in tag:
        return PY313
    if "314" in tag:
        return PY314
    return PY312


def run_pycdc(pyc_path: Path) -> tuple[str, str, int]:
    try:
        r = subprocess.run([PYCDC, str(pyc_path)], capture_output=True, text=True, timeout=120)
        return r.stdout, r.stderr, r.returncode
    except Exception as e:
        return "", f"pycdc subprocess error: {e}", -1


def ast_check(source: str) -> tuple[bool, str]:
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.lineno}:{e.offset}: {e.msg}"
    except Exception as e:
        return False, str(e)


SPEC_HELPER = r"""
import sys, marshal, struct, dis, json
from io import StringIO

p = sys.argv[1]
with open(p, 'rb') as f:
    magic = f.read(4)
    flags = struct.unpack('<I', f.read(4))[0]
    if flags & 1:
        f.read(8)
    else:
        f.read(4); f.read(4)
    try:
        code = marshal.load(f)
    except Exception as e:
        print(json.dumps({'error': f'marshal: {e}'}))
        sys.exit(0)

def safe_repr(c, depth=0):
    if depth > 4: return '<deep>'
    if c is None or isinstance(c, (bool, int, float, str)):
        return c
    if isinstance(c, bytes):
        return {'__bytes_hex__': c.hex()[:400]}
    if isinstance(c, tuple):
        return [safe_repr(x, depth+1) for x in c]
    if isinstance(c, frozenset):
        return {'__frozenset__': [safe_repr(x, depth+1) for x in c]}
    if hasattr(c, 'co_name'):
        return {
            '__code__': c.co_name,
            'qualname': getattr(c, 'co_qualname', c.co_name),
            'argcount': c.co_argcount,
            'varnames': list(c.co_varnames),
            'names': list(c.co_names),
            'freevars': list(c.co_freevars),
            'cellvars': list(c.co_cellvars),
        }
    return repr(c)[:300]

spec = {
    'pyc_path': p,
    'names': list(code.co_names),
    'varnames': list(code.co_varnames),
    'consts': [safe_repr(c) for c in code.co_consts],
    'freevars': list(code.co_freevars),
    'cellvars': list(code.co_cellvars),
    'doc': code.co_consts[0] if code.co_consts and isinstance(code.co_consts[0], str) else None,
    'child_codes': [],
}

sio = StringIO()
try:
    dis.dis(code, file=sio, depth=0)
    spec['module_dis'] = sio.getvalue()
except Exception as e:
    spec['module_dis_error'] = str(e)

def walk(c):
    for const in c.co_consts:
        if hasattr(const, 'co_name'):
            child = {
                'qualname': getattr(const, 'co_qualname', const.co_name),
                'name': const.co_name,
                'argcount': const.co_argcount,
                'kwonlyargcount': const.co_kwonlyargcount,
                'varnames': list(const.co_varnames),
                'names': list(const.co_names),
                'freevars': list(const.co_freevars),
                'cellvars': list(const.co_cellvars),
                'doc': const.co_consts[0] if const.co_consts and isinstance(const.co_consts[0], str) else None,
            }
            sio2 = StringIO()
            try:
                dis.dis(const, file=sio2, depth=0)
                child['dis'] = sio2.getvalue()
            except Exception as e:
                child['dis_error'] = str(e)
            spec['child_codes'].append(child)
            walk(const)

walk(code)
print(json.dumps(spec, indent=2, default=str))
"""


def extract_recovery_spec(pyc_path: Path, py_version_tag: str) -> dict:
    py = py_for_tag(py_version_tag)
    helper_path = Path("/tmp/_extract_pyc_spec.py")
    helper_path.write_text(SPEC_HELPER)
    try:
        r = subprocess.run([py, str(helper_path), str(pyc_path)], capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError as e:
                return {"error": f"json decode: {e}", "raw_stdout_head": r.stdout[:500], "stderr": r.stderr[:500]}
        return {"error": f"helper rc={r.returncode}", "stderr": r.stderr[:500]}
    except Exception as e:
        return {"error": str(e)}


def make_stub_for_partial(orphan_rel: str, pycdc_out: str, ast_err: str) -> str:
    spec_name = Path(orphan_rel).stem + ".recovery_spec.json"
    safe = pycdc_out.replace('"""', '\\"\\"\\"')
    stub = (
        f'"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.\n'
        f'\n'
        f'This file was decompiled from a .pyc orphan whose .py source was never\n'
        f'committed. pycdc 3.12 produces substantially-complete output but trips on\n'
        f'@dataclass/@property decorators, complex lambdas, and walrus operators,\n'
        f'so the raw output does not parse: ``{ast_err}``.\n'
        f'\n'
        f'The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``\n'
        f'below. The companion ``{spec_name}`` contains co_names, co_consts,\n'
        f'co_varnames, and dis() output for every code object - the structural\n'
        f'ground-truth a hand-rehydrator should consult.\n'
        f'\n'
        f'This stub itself is a no-op; importing it just exposes the partial\n'
        f'output as a string. Replace the stub with hand-rewritten Python once\n'
        f'rehydration is done.\n'
        f'"""\n'
        f'from __future__ import annotations\n'
        f'\n'
        f'__recovery_status__ = "partial"\n'
        f'__recovery_orphan__ = {orphan_rel!r}\n'
        f'__recovery_spec__ = {spec_name!r}\n'
        f'__recovery_ast_error__ = {ast_err!r}\n'
        f'\n'
        f'_PYCDC_PARTIAL_OUTPUT = r"""\n'
        f'{safe}\n'
        f'"""\n'
    )
    try:
        ast.parse(stub)
        return stub
    except SyntaxError:
        # Fallback: encode pycdc output as JSON string (always parses)
        stub = (
            f'"""RECOVERY STUB (fallback) - pycdc output as JSON string."""\n'
            f'from __future__ import annotations\n'
            f'import json as _json\n'
            f'__recovery_status__ = "partial"\n'
            f'__recovery_orphan__ = {orphan_rel!r}\n'
            f'__recovery_spec__ = {spec_name!r}\n'
            f'__recovery_ast_error__ = {ast_err!r}\n'
            f'_PYCDC_PARTIAL_OUTPUT = _json.loads({json.dumps(pycdc_out)!r})\n'
        )
        return stub


def make_stub_for_unloadable_pyc(orphan_rel: str, py_version_tag: str) -> str:
    spec_name = Path(orphan_rel).stem + ".recovery_spec.json"
    return (
        f'"""RECOVERY STUB - pycdc cannot load this .pyc (Python {py_version_tag}).\n'
        f'\n'
        f'Only the structural disassembly is available; see ``{spec_name}`` for\n'
        f'co_names / co_consts / dis() output.\n'
        f'"""\n'
        f'from __future__ import annotations\n'
        f'__recovery_status__ = "spec_only"\n'
        f'__recovery_orphan__ = {orphan_rel!r}\n'
        f'__recovery_spec__ = {spec_name!r}\n'
        f'__recovery_pyc_python__ = {py_version_tag!r}\n'
    )


def attempt_recover(orphan_rel: str) -> dict:
    result: dict = {
        "orphan_rel": orphan_rel,
        "pyc_exists": False,
        "pyc_path": None,
        "pyc_python_tag": None,
        "decompiled_ok": False,
        "ast_parse_ok": False,
        "stub_written": False,
        "ast_error": "",
        "spec_written": False,
        "py_path": "",
        "spec_path": "",
        "warnings": [],
    }
    found = find_pyc(orphan_rel)
    if found is None:
        return result
    pyc, tag = found
    result["pyc_exists"] = True
    result["pyc_path"] = str(pyc)
    result["pyc_python_tag"] = tag

    py_out = REPO_ROOT / orphan_rel
    py_out.parent.mkdir(parents=True, exist_ok=True)
    spec_path = py_out.with_suffix(".recovery_spec.json")

    # Always extract spec
    spec = extract_recovery_spec(pyc, tag)
    spec_path.write_text(json.dumps(spec, indent=2, default=str))
    result["spec_written"] = True
    result["spec_path"] = str(spec_path)

    stdout, stderr, rc = run_pycdc(pyc)
    if rc == -1 or "Bad MAGIC" in stdout or "Bad MAGIC" in stderr:
        # pycdc cannot load (likely 314); write stub
        py_out.write_text(make_stub_for_unloadable_pyc(orphan_rel, tag))
        result["py_path"] = str(py_out)
        result["stub_written"] = True
        result["warnings"].append(f"pycdc bad MAGIC for {tag}-tagged pyc")
        return result

    if "WARNING: Decompyle incomplete" in stdout:
        result["warnings"].append("pycdc emitted 'Decompyle incomplete'")
    if stderr.strip():
        result["warnings"].append(f"pycdc stderr: {stderr.strip()[:200]}")

    ast_ok, ast_err = ast_check(stdout)
    result["ast_parse_ok"] = ast_ok
    result["ast_error"] = ast_err
    result["decompiled_ok"] = ast_ok and "WARNING: Decompyle incomplete" not in stdout

    if ast_ok:
        # Use the pycdc output directly (parses cleanly)
        py_out.write_text(stdout)
        result["py_path"] = str(py_out)
    else:
        # Write a stub that quarantines the partial pycdc output
        stub = make_stub_for_partial(orphan_rel, stdout, ast_err)
        py_out.write_text(stub)
        result["py_path"] = str(py_out)
        result["stub_written"] = True
    return result


def main() -> int:
    orphans = json.loads(Path("/tmp/orphan_not_in_git.json").read_text())
    orphans = [o for o in orphans if o not in ALREADY_RECOVERED]
    print(f"Recovering {len(orphans)} orphans (skipped {len(ALREADY_RECOVERED)})")

    results = []
    for i, o in enumerate(orphans):
        r = attempt_recover(o)
        results.append(r)
        if not r.get("pyc_exists"):
            status = "NO-PYC"
        elif r["decompiled_ok"]:
            status = "OK"
        elif r["ast_parse_ok"]:
            status = "AST-OK-WARN"
        elif r.get("stub_written"):
            status = "STUB"
        else:
            status = "?"
        tag = r.get("pyc_python_tag") or "-"
        print(f"[{i+1:3d}/{len(orphans)}] {status:11s} [{tag:9s}] {o}")
    Path("/tmp/recovery_results_v3.json").write_text(json.dumps(results, indent=2))

    # Verify EVERY .py file we wrote parses cleanly (no SyntaxError gates)
    print()
    print("== AST RE-VERIFICATION ==")
    failed = []
    for r in results:
        if r.get("py_path"):
            try:
                ast.parse(Path(r["py_path"]).read_text())
            except SyntaxError as e:
                failed.append((r["orphan_rel"], str(e)))
    if failed:
        print(f"  FAILED: {len(failed)} files do NOT parse:")
        for rel, err in failed:
            print(f"    {rel}: {err}")
    else:
        print(f"  All {sum(1 for r in results if r.get('py_path'))} written .py files parse cleanly.")

    n_total = len(results)
    n_no_pyc = sum(1 for r in results if not r.get("pyc_exists"))
    n_ok = sum(1 for r in results if r["decompiled_ok"])
    n_ast_ok = sum(1 for r in results if r["ast_parse_ok"] and not r["decompiled_ok"])
    n_stub = sum(1 for r in results if r.get("stub_written"))
    print()
    print("== SUMMARY ==")
    print(f"  total:           {n_total}")
    print(f"  no-pyc:          {n_no_pyc}")
    print(f"  fully OK:        {n_ok}")
    print(f"  AST-OK warn:     {n_ast_ok}")
    print(f"  stub (partial):  {n_stub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
