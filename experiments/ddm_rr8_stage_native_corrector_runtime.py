#!/usr/bin/env python3
"""ddm_rr8 - stage a candidate runtime tree whose free corrector is the C port.

WHY.  ``ddm_cd1`` MEASURED the corrector at **917.929 s = 71.7%** of the shipped jg5 token
stage on a contest T4, and pre-registered a break-even band a port must clear:
**2.03--2.22x** for frame B, **2.77--3.08x** for frame A (original cd1 anchor through the
jg5/noise re-anchor).  ``ddm_rr7`` had already ported the OTHER 21% -- the integer
HPAC model -- and MEASURED a 15.3% regression, because that port moved work off the T4's GPU
onto its weak container vCPUs.  The corrector is ALREADY on those vCPUs in numpy, so this
port changes the language and not the processor.

WHAT IT CHANGES, and the discipline it borrows.  Two files ADDED, two files REWRITTEN, every
rewrite recorded as a textual transformation with an asserted match count (``ddm_wc2c`` /
``ddm_cd1``), so a drifted base REFUSES to stage rather than being silently patched:

  ADD      runtime/f26_corrector_native.c      the port
  ADD      runtime/native_free_corrector.py    the ctypes binding + config-drift refusal
  REWRITE  runtime/residual_archive.py         one selector call, plus its helper
  REWRITE  inflate.sh                          a FAIL-CLOSED build of the corrector library

FAIL-CLOSED IS THE POINT, and it is ``ddm_rr6`` §2's lesson paid forward.  ``inflate.sh``
runs under ``set -euo pipefail``, so a bare ``cc`` failure on unknown contest silicon would
abort the inflate and turn a wall-clock WARN into a ZERO.  Both attempts here sit inside an
``if`` CONDITION -- which errexit does not apply to -- and a failure leaves
``F26_CORRECTOR_NATIVE_LIBRARY`` unset, which the selector reads as "use the shipped Python
corrector".  A hostile toolchain must cost the SPEEDUP, never the submission.

An explicitly-named-but-missing library still raises.  Operator misconfiguration and a
hostile toolchain are different classes and are treated differently.

WHAT IT DOES NOT DO.  It does not touch the sealed jg5 tree, does not fire anything, and
claims no score.  The archive is untouched, so ``evaluate.py`` -- which reads only ``0.raw``
and ``archive.zip`` -- must return the same score by construction.  A T4 row that disagrees
FALSIFIES the identity proof; it does not reprice the candidate.

COMPOSES WITH ``ddm_cd1``.  Point ``--base`` at cd1's instrumented tree and the result is a
tree that is both ported AND self-decomposing, which is what prices the port's own scope on
the shipping axis rather than inferring it by subtraction.  The anchors below survive cd1's
rewrites intact, and the stager asserts that rather than assuming it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO / "submissions" / "robust_current" / "jg5_sub015_runtime" / "runtime"
SOURCE_DIR = REPO / "runtime-rs" / "native" / "f26-corrector"

PY_TARGET = "runtime/residual_archive.py"
SH_TARGET = "inflate.sh"
ADDED = ("runtime/f26_corrector_native.c", "runtime/native_free_corrector.py")

# --- rewrite 1: the selector helper -----------------------------------------------------
HELPER_ANCHOR = '''def decode_production_tokens(
    parts: ResidualArchiveParts,
'''

HELPER_NEW = '''# --- ddm_rr8 NATIVE FREE CORRECTOR SELECTION --------------------------------------------
# The corrector is 71.7% of the shipped token stage on a contest T4 (ddm_cd1, MEASURED), so
# it is lowered to C.  The C reproduces the frozen SHIPPED_CONFIG only, and refuses to bind
# if that config has drifted -- a refusal costs the speedup, a silent mismatch costs the
# submission (ddm_rr2 scored S = 27.83 on a desynchronised decoder and it read as a model
# failure rather than what it was).
def _rr8_corrector_kind(corrector) -> str:
    """Name the corrector that ACTUALLY ran, into the token report.

    Without this the identity gate is VACUOUS.  If the native build fails, the selector
    falls back to the shipped Python corrector -- correctly -- and a full-field identity run
    then compares the shipped decoder against itself and PASSES, having proved nothing about
    the port.  Absence of a fallback message on stderr is not evidence either.  This field is
    what lets the gate fail, and it is what tells a paid T4 row whether it measured the port
    or the fallback.
    """
    return type(corrector).__name__


def _rr8_select_corrector(plane: int):
    """The C corrector when the runtime built one; otherwise the shipped Python corrector.

    No try/except: ``load_native_corrector`` returns None when no library was built (the
    fail-closed path inflate.sh takes on a hostile toolchain) and RAISES when one was named
    but is missing, has the wrong ABI, or was compiled against a different configuration.
    Swallowing that second class would run a corrector nobody chose.
    """
    from .free_corrector import FreeCorrector
    from .native_free_corrector import load_native_corrector

    native = load_native_corrector(plane)
    if native is not None:
        return native
    return FreeCorrector(plane)


def decode_production_tokens(
    parts: ResidualArchiveParts,
'''

# --- rewrite 2: the construction --------------------------------------------------------
CONSTRUCT_OLD = '''    corrector = FreeCorrector(runtime.EVAL_H * runtime.EVAL_W)
'''

CONSTRUCT_NEW = '''    corrector = _rr8_select_corrector(runtime.EVAL_H * runtime.EVAL_W)
'''

# --- rewrite 3: the fail-closed build ---------------------------------------------------
BUILD_OLD = '''export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"
'''

BUILD_NEW = '''export CPR1_RC64_LIBRARY="$BUILD_DIR/rc64_backend.so"

# ddm_rr8 -- the float64 free corrector, lowered to C.  Both attempts sit inside the `if`
# CONDITION so `set -e` does NOT apply to them: a compiler that cannot build this file must
# cost the speedup, never the submission.  Failure leaves the variable unset and the decoder
# falls back to the proven Python corrector.
#
# -ffp-contract=off is LOAD-BEARING, not hygiene.  FMA contraction fuses a multiply and an
# add into a single rounding step, which would change the emitted probabilities and
# desynchronise the arithmetic decoder.
if [[ -n "${F26_CORRECTOR_NATIVE_LIBRARY:-}" ]]; then
  [[ -f "$F26_CORRECTOR_NATIVE_LIBRARY" ]] || { echo "missing F26 corrector library" >&2; exit 69; }
elif "${CC:-cc}" -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \\
       "$HERE/runtime/f26_corrector_native.c" -lm \\
       -o "$BUILD_DIR/f26_corrector_native.so" 2>/dev/null; then
  export F26_CORRECTOR_NATIVE_LIBRARY="$BUILD_DIR/f26_corrector_native.so"
else
  echo "f26 corrector native build unavailable; using the python corrector" >&2
fi
'''

# --- rewrite 3: the receipt -------------------------------------------------------------
# Anchored on the bit-position line alone, which appears exactly once in the base tree AND
# exactly once in ddm_cd1's instrumented rewrite of the same block -- so this stager composes
# onto either without a stale-anchor refusal.
RECEIPT_OLD = '''        "decoder_bit_position": decoder.bit_position,
'''

RECEIPT_NEW = '''        "decoder_bit_position": decoder.bit_position,
        "free_corrector": _rr8_corrector_kind(corrector),
'''

PY_REWRITES = (
    ("corrector selector helper", HELPER_ANCHOR, HELPER_NEW),
    ("corrector construction", CONSTRUCT_OLD, CONSTRUCT_NEW),
    ("corrector receipt", RECEIPT_OLD, RECEIPT_NEW),
)
SH_REWRITES = (("fail-closed corrector build", BUILD_OLD, BUILD_NEW),)


class StagingError(RuntimeError):
    """A base-tree assumption or a produced-tree invariant failed."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _apply(text: str, old: str, new: str, label: str) -> str:
    """Apply one recorded rewrite, refusing on any count but exactly one."""
    found = text.count(old)
    if found != 1:
        raise StagingError(
            f"{label}: expected exactly 1 occurrence in the base tree, found {found}; "
            "the base tree has drifted and this transformation is stale"
        )
    return text.replace(old, new)


def _tree_sha256(root: Path) -> str:
    """Order-stable hash over every staged file's relative path and content.

    Byte-compatible with ``ddm_wc2c`` / ``ddm_cd1``'s definition on purpose: these trees are
    all compared against the same jg5 base, and a divergent definition would make their
    manifests incomparable.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name.startswith("._"):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _census(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.startswith("._")
    }


def _assert_expected_delta(base: Path, output: Path) -> dict[str, list[str]]:
    """Refuse unless the staged tree differs from its base in EXACTLY the declared way.

    Two files added, two rewritten, nothing else touched and nothing removed.  Stated as an
    invariant the stager checks rather than a claim the memo makes.
    """
    before, after = _census(base), _census(output)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(name for name in before if name in after and before[name] != after[name])

    if removed:
        raise StagingError(f"staging removed files, which it never does: {removed}")
    if added != sorted(ADDED):
        raise StagingError(f"expected exactly {sorted(ADDED)} added, got {added}")
    if changed != sorted((PY_TARGET, SH_TARGET)):
        raise StagingError(
            f"expected exactly {sorted((PY_TARGET, SH_TARGET))} changed, got {changed}"
        )
    return {"added": added, "changed": changed}


def stage(base: Path, output: Path, *, force: bool = False) -> dict[str, Any]:
    base = base.resolve()
    output = output.resolve()
    if not (base / PY_TARGET).is_file():
        raise StagingError(f"no candidate runtime tree at {base}")
    for name in ADDED:
        if (base / name).exists():
            raise StagingError(
                f"{name} already exists in the base tree; this stager only ADDS it, so a "
                "pre-existing copy means the base has already been ported"
            )
    source_c = SOURCE_DIR / "f26_corrector_native.c"
    source_py = SOURCE_DIR / "native_free_corrector.py"
    for path in (source_c, source_py):
        if not path.is_file():
            raise StagingError(f"missing port source: {path}")

    if output.exists():
        if not force:
            raise StagingError(f"refusing to overwrite existing tree: {output}")
        shutil.rmtree(output)

    base_sha = _tree_sha256(base)
    shutil.copytree(base, output, ignore=shutil.ignore_patterns("__pycache__", "._*"))

    shutil.copyfile(source_c, output / ADDED[0])
    shutil.copyfile(source_py, output / ADDED[1])

    py_path = output / PY_TARGET
    text = py_path.read_text()
    for label, old, new in PY_REWRITES:
        text = _apply(text, old, new, label)
    py_path.write_text(text)
    # The staged module must at least PARSE before anything downstream trusts it; a
    # SyntaxError discovered on a paid runner costs the dispatch, not the edit.
    compile(text, str(py_path), "exec")
    compile(source_py.read_text(), str(output / ADDED[1]), "exec")

    sh_path = output / SH_TARGET
    sh_text = sh_path.read_text()
    for label, old, new in SH_REWRITES:
        sh_text = _apply(sh_text, old, new, label)
    sh_path.write_text(sh_text)
    sh_path.chmod(0o755)

    delta = _assert_expected_delta(base, output)

    manifest = {
        "schema": "ddm_rr8_stage_native_corrector.v1",
        "base_tree": str(base),
        "base_tree_sha256": base_sha,
        "output_tree": str(output),
        "output_tree_sha256": _tree_sha256(output),
        "added_files": delta["added"],
        "changed_files": delta["changed"],
        "port_sources": {
            str(source_c.relative_to(REPO)): _sha256_file(source_c),
            str(source_py.relative_to(REPO)): _sha256_file(source_py),
        },
        "staged_files": {name: _sha256_file(output / name) for name in ADDED}
        | {
            PY_TARGET: _sha256_file(py_path),
            SH_TARGET: _sha256_file(sh_path),
        },
        "rewrites_applied": [label for label, _, _ in PY_REWRITES + SH_REWRITES],
        "port": {
            "scope": "group_state + coding_row + observe (ddm_cd1 port_scope_seconds)",
            "measured_scope_seconds_t4": 917.929,
            # Legacy scalar keys stay additive-compatible for existing manifest
            # readers; the adjacent band is the decision surface (rvf1 F2/F3).
            "break_even_frame_b": 2.03,
            "break_even_frame_b_band": {
                "original_cd1": 2.03,
                "jg5_noise_reanchor": 2.22,
            },
            "break_even_frame_a": 2.77,
            "break_even_frame_a_band": {
                "original_cd1": 2.77,
                "jg5_noise_reanchor": 3.08,
            },
            "fallback": "F26_CORRECTOR_NATIVE_LIBRARY unset -> shipped Python corrector",
            "cannot_move_a_byte_because": (
                "the C reproduces the same IEEE-754 correctly rounded operations in the "
                "same order; the falsifier is the full-field identity run, not this claim"
            ),
        },
        "owed_before_any_exact_eval": [
            "full-field n600 identity run against this staged tree (0.raw + the four token anchors)",
            "a fresh candidate seal measured from this tree -- never a hand-typed digest",
        ],
    }
    _atomic_json(output.parent / f"{output.name}_stage_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest = stage(args.base, args.output, force=args.force)
    except StagingError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
