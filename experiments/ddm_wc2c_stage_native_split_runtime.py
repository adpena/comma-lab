#!/usr/bin/env python3
"""ddm_wc2c - stage a candidate runtime tree with ``native-hpac`` re-enabled.

WHY A STAGER AND NOT A HAND EDIT.  The two files that carry the refusal --
``runtime/f26_inflate.py`` and ``runtime/residual_archive.py`` -- have **no
byte-identical source anywhere in this repository**.  A census of the jg5
candidate tree (34 files) found 24 with no repo source at all and 6 that are
import-rewritten derivatives of an ``experiments/`` module; the shipping
decoder therefore lives only on an external SSD, unversioned.  Editing those
files by hand would make the change unreviewable and unreproducible.  So the
change is expressed as a RECORDED TEXTUAL TRANSFORMATION with asserted match
counts: if the base tree drifts, the assertion fires and the stage refuses,
rather than silently applying a patch to text that no longer means what it did.

WHAT IT CHANGES, exactly four things:

1. Copies ``experiments/ddm_wc2c_split_token_decoder.py`` in as
   ``runtime/f26_split_token_decoder.py``, BYTE-IDENTICALLY.  It already uses
   package-relative imports, so unlike the ``ddm_fx1`` staging path it needs no
   rewrite -- and byte-identity is a stronger property than an asserted rewrite
   count, so the stager checks for it.
2. Replaces the ``native-hpac`` refusal in ``f26_inflate.py`` with a
   requirement that the identity proof exists.
3. Routes the ``native-hpac`` branch to ``decode_split_tokens``.
4. Adds the split decoder and the C source to the token-decoder fingerprint, so
   a changed decoder cannot reuse a token checkpoint made by a different one.

WHAT IT DOES NOT DO.  It does not touch the jg5 tree in place, does not fire
anything, and does not claim a score.  It emits a NEW tree plus a manifest of
every source SHA it consumed and every SHA it produced.

THE OUTPUT IS NOT SUBMITTABLE UNTIL IT IS RE-PROVEN.  Staging changes the
runtime tree SHA, which the harness's runtime-tree validators enforce, and it
changes the token-decoder fingerprint.  A staged tree owes a fresh full-field
identity run before any exact eval.
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
SPLIT_SOURCE = REPO / "experiments" / "ddm_wc2c_split_token_decoder.py"
C_SOURCE = REPO / "runtime-rs" / "native" / "f26-hpac" / "f26_hpac_native.c"
STAGED_SPLIT = "runtime/f26_split_token_decoder.py"
STAGED_C = "runtime/f26_hpac_native.c"

REFUSAL_OLD = '''    if token_decoder != "python":
        raise InflationError(
            "this generation wires the ddm_rr2 free probability corrector into the "
            "python token decoder only; the native-hpac path is unpatched and would "
            "decode a different field, so it is refused rather than trusted"
        )
'''

REFUSAL_NEW = '''    if token_decoder == "native-hpac" and not os.environ.get(
        "F26_HPAC_NATIVE_LIBRARY", ""
    ).strip():
        raise InflationError(
            "native-hpac requires F26_HPAC_NATIVE_LIBRARY; the split decoder keeps "
            "the free probability corrector in numpy and lowers only the integer "
            "model, and it is admissible only against a built library whose "
            "full-field identity has been proven (ddm_wc2c)"
        )
'''

ROUTE_OLD = '''        if token_decoder == "native-hpac":
            if checkpoint_dir is None:
                raise InflationError("native token decode requires checkpoint_dir")
            from .f26_hpac_native import decode_native_tokens

            native_progress = checkpoint_dir.resolve() / "native_hpac_progress"
            tokens, token_report = decode_native_tokens(
                parts,
                renderer,
                renderer_dir,
                device,
                frame_limit=pair_count,
                output_path=native_progress / "tokens_partial.u8",
                checkpoint_dir=native_progress / "checkpoints",
            )
'''

ROUTE_NEW = '''        if token_decoder == "native-hpac":
            if checkpoint_dir is None:
                raise InflationError("native token decode requires checkpoint_dir")
            from .f26_split_token_decoder import decode_split_tokens

            native_progress = checkpoint_dir.resolve() / "native_hpac_progress"
            tokens, token_report = decode_split_tokens(
                parts,
                renderer,
                renderer_dir,
                device,
                checkpoint_dir=native_progress,
            )
'''

FINGERPRINT_OLD = '''        files.extend(
            [
                runtime_dir / "f26_hpac_native.py",
                runtime_dir / "f26_hpac_native.c",
            ]
        )
'''

# The OpenMP-free build line.  The C no longer uses OpenMP, so requiring
# `brew --prefix libomp` on Darwin and `-fopenmp` on Linux at DECODE TIME is a
# dependency that can only fail -- and its failure mode is the accelerator
# silently unavailable inside the 30-minute wall.  Both branches collapse to one
# portable command that links libc only.  `-ffp-contract=off -fno-fast-math` are
# load-bearing, not decoration: FMA contraction and reassociation both change
# rounding, which desynchronises the decoder.
BUILD_OLD = '''    case "$(uname -s)" in
      Darwin)
        LIBOMP_PREFIX="$(brew --prefix libomp)"
        "${CC:-cc}" -O3 -mcpu=native -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \\
          -Xpreprocessor -fopenmp -I"$LIBOMP_PREFIX/include" \\
          "$HERE/runtime/f26_hpac_native.c" -L"$LIBOMP_PREFIX/lib" -lomp \\
          -Wl,-rpath,"$LIBOMP_PREFIX/lib" -lm -o "$BUILD_DIR/f26_hpac_native.so" ;;
      *)
        "${CC:-cc}" -O3 -march=native -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \\
          -fopenmp "$HERE/runtime/f26_hpac_native.c" -lm -o "$BUILD_DIR/f26_hpac_native.so" ;;
    esac
'''

BUILD_NEW = '''    case "$(uname -s)" in
      Darwin) NATIVE_ARCH_FLAG="-mcpu=native" ;;
      *)      NATIVE_ARCH_FLAG="-march=native" ;;
    esac
    # The arch flag is a build-host optimisation, never the ISA gate: the gate is
    # __builtin_cpu_supports at runtime.  So a compiler that rejects it falls back
    # to the portable build rather than turning the accelerator off.
    if ! "${CC:-cc}" -O3 $NATIVE_ARCH_FLAG -std=c11 -shared -fPIC \\
        -ffp-contract=off -fno-fast-math \\
        "$HERE/runtime/f26_hpac_native.c" -lm \\
        -o "$BUILD_DIR/f26_hpac_native.so" 2>/dev/null
    then
      "${CC:-cc}" -O3 -std=c11 -shared -fPIC -ffp-contract=off -fno-fast-math \\
        "$HERE/runtime/f26_hpac_native.c" -lm -o "$BUILD_DIR/f26_hpac_native.so"
    fi
'''

FINGERPRINT_NEW = '''        files.extend(
            [
                runtime_dir / "f26_hpac_native.py",
                runtime_dir / "f26_hpac_native.c",
                runtime_dir / "f26_split_token_decoder.py",
                runtime_dir / "free_corrector.py",
            ]
        )
'''


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
    """Order-stable hash over every staged file's relative path and content."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name.startswith("._"):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def stage(base: Path, output: Path, *, force: bool = False) -> dict[str, Any]:
    base = base.resolve()
    output = output.resolve()
    if not (base / "runtime" / "f26_inflate.py").is_file():
        raise StagingError(f"no candidate runtime tree at {base}")
    if output.exists():
        if not force:
            raise StagingError(f"refusing to overwrite existing tree: {output}")
        shutil.rmtree(output)
    for required in (SPLIT_SOURCE, C_SOURCE):
        if not required.is_file():
            raise StagingError(f"missing repo source: {required}")

    shutil.copytree(base, output, ignore=shutil.ignore_patterns("__pycache__", "._*"))

    # 1. Byte-identical copy of the split decoder.
    split_destination = output / STAGED_SPLIT
    shutil.copyfile(SPLIT_SOURCE, split_destination)
    if _sha256_file(SPLIT_SOURCE) != _sha256_file(split_destination):
        raise StagingError("staged split decoder is not byte-identical to its source")

    # The C source ships inside the tree because inflate.sh compiles it there.
    c_destination = output / STAGED_C
    shutil.copyfile(C_SOURCE, c_destination)
    if _sha256_file(C_SOURCE) != _sha256_file(c_destination):
        raise StagingError("staged C source is not byte-identical to its source")

    # 2-4. Recorded rewrites of the inherited inflate entry point.
    inflate_path = output / "runtime" / "f26_inflate.py"
    text = inflate_path.read_text()
    text = _apply(text, REFUSAL_OLD, REFUSAL_NEW, "native-hpac refusal")
    text = _apply(text, ROUTE_OLD, ROUTE_NEW, "native-hpac decode route")
    text = _apply(text, FINGERPRINT_OLD, FINGERPRINT_NEW, "token decoder fingerprint")
    inflate_path.write_text(text)

    shell_path = output / "inflate.sh"
    shell = shell_path.read_text()
    shell = _apply(shell, BUILD_OLD, BUILD_NEW, "OpenMP-free native build line")
    shell_path.write_text(shell)

    manifest = {
        "schema": "ddm_wc2c_stage_native_split.v1",
        "base_tree": str(base),
        "base_tree_sha256": _tree_sha256(base),
        "output_tree": str(output),
        "output_tree_sha256": _tree_sha256(output),
        "repo_sources": {
            "split_decoder": {
                "path": str(SPLIT_SOURCE.relative_to(REPO)),
                "sha256": _sha256_file(SPLIT_SOURCE),
            },
            "native_c": {
                "path": str(C_SOURCE.relative_to(REPO)),
                "sha256": _sha256_file(C_SOURCE),
            },
        },
        "staged_files": {
            STAGED_SPLIT: _sha256_file(split_destination),
            STAGED_C: _sha256_file(c_destination),
            "runtime/f26_inflate.py": _sha256_file(inflate_path),
            "inflate.sh": _sha256_file(shell_path),
        },
        "rewrites_applied": [
            "native-hpac refusal",
            "native-hpac decode route",
            "token decoder fingerprint",
            "OpenMP-free native build line",
        ],
        "owed_before_any_exact_eval": [
            "full-field identity run against this staged tree",
            "re-pin the harness runtime-tree SHA validators",
        ],
    }
    _atomic_json(output.parent / f"{output.name}_stage_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_jg5/candidate_runtime_jg5"),
    )
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
