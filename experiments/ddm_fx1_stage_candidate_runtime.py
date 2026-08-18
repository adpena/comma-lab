"""ddm_fx1 - generate the fx1 receiver runtime tree, receipted and re-measured.

WHY THIS IS A SCRIPT AND NOT A SEQUENCE OF COPIES.

``ddm_rr2`` proved byte-exactness locally and was then refused on the contest T4
at ``S = 27.83``.  The lesson that outlived the specific bug is STAGING
INFIDELITY: the tree that ships drifts from the code that was measured, and
nothing in a hand-assembled directory records which is which.  A hand-staged
runtime is a claim with no receipt behind it.

So this generator is the only sanctioned way to produce an fx1 candidate tree.
It is deterministic (same inputs -> byte-identical tree), it records the sha256
of every staged file, and -- the part that actually closes the hazard -- it can
RE-MEASURE the full n600 code length THROUGH THE STAGED MODULE and refuse to
emit a receipt unless the staged tree reproduces the raced number exactly.  A
tree that cannot reproduce its own measurement is not a candidate.

WHAT IT STAGES.  The base tree already carries the shipped rr4 corrector at
``runtime/free_corrector.py`` and consumes it at
``runtime/residual_archive.py:535`` as ``FreeCorrector(EVAL_H * EVAL_W)``.  That
is exactly the frozen drop-in signature ``ddm_fx1_logistic_mixer_corrector``
exposes, so the integration needs no change to the decode path at all:

* ``runtime/rr4_free_corrector.py``  <- experiments/ddm_rr4_free_corrector_v2.py,
  byte-identical.  The mixer inherits its transport and its constants from this
  module, so it is staged unmodified rather than inlined; a copy that drifted
  from the shipped law would be a second source of truth.
* ``runtime/free_corrector.py``      <- experiments/ddm_fx1_logistic_mixer_corrector.py
  with ONE mechanical transformation: the absolute import of the rr4 module
  becomes the package-relative import of the staged copy.  The transformation is
  asserted to apply an exact expected number of times, so a refactor upstream
  that changes the import shape fails the staging loudly instead of silently
  producing a tree that imports nothing.
* ``inflate.py``                     <- archive sha256 / byte pins rewritten.
* ``archive.zip``                    <- the byte-closed fx1 archive.

RULE 118.  Everything staged here is generic receiver CODE and is free.  No
learned or video-derived content enters the tree: the mixer's weights are not
stored, they start at a fixed integer and regenerate from already-decoded
symbols.  The only counted bytes are ``archive.zip`` itself.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_BASE_TREE = Path(
    "/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime"
)
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_fx1/candidate_runtime")
DEFAULT_ARCHIVE = Path("/Volumes/APDataStore/pact/ddm_fx1/byteclose_r2/retained/archive.zip")

RR4_SOURCE = REPO / "experiments" / "ddm_rr4_free_corrector_v2.py"
FX1_SOURCE = REPO / "experiments" / "ddm_fx1_logistic_mixer_corrector.py"

STAGED_RR4 = "runtime/rr4_free_corrector.py"
STAGED_FX1 = "runtime/free_corrector.py"

IMPORT_FROM = "from experiments.ddm_rr4_free_corrector_v2 import"
IMPORT_TO = "from .rr4_free_corrector import"
EXPECTED_IMPORT_REWRITES = 2

# Excluded from the copy so the tree is a deterministic function of its inputs.
SKIP_DIRS = {"__pycache__", ".omx"}
SKIP_PREFIXES = ("._",)
SKIP_NAMES = {
    "archive.zip",
    "GENERATION_RECEIPT.json",
    "RECEIVER_PARSEBACK.json",
    "FX1_STAGING_RECEIPT.json",
}

LEAK_MARKERS = ("/Volumes/", "/Users/", "/home/", "/tmp/", "/private/")
"""Absolute-path prefixes that must never appear inside the shipped tree.

The shipped rr4 receipts carry 63 local absolute paths inside the hashed runtime
tree.  That is a packet-hygiene defect and a disclosure defect at once: it pins a
private filesystem layout into a public artifact and makes the tree
irreproducible anywhere else.  This generator refuses to emit such a tree rather
than relying on a later scrub, because a scrub changes the hashed tree and
therefore invalidates every measurement taken against it."""

BANNED_ATTRS = {"log", "log2", "log10", "exp", "exp2", "expm1", "log1p", "power", "float_power", "pow"}

CC_ANCHOR = '"${CC:-cc}" -O3 -std=c11 -shared -fPIC \\\n'
"""The unguarded RC64 compile in the inherited inflate.sh."""

CC_GUARD = """# --- fail-closed C toolchain guard (ddm_fx1) --------------------------------
# The RC64 backend is compiled from source at decode time, which makes a working
# C compiler a HARD runtime dependency of this receiver.  The inherited script
# compiled it unconditionally: on a host without a toolchain the failure surfaced
# as a raw compiler error from inside `set -e`, not as a named missing
# dependency.  Brotli already fails closed by name here; the compiler gets the
# same honesty.  Existence is not enough -- a stub or broken `cc` passes
# `command -v` -- so the guard PROBES a real compile before trusting it.
CC_BIN="${CC:-cc}"
if ! command -v "$CC_BIN" >/dev/null 2>&1; then
  echo "F26 decode requires a C compiler to build the RC64 backend." >&2
  echo "  missing: ${CC_BIN} (set CC=<compiler> or install build tools)" >&2
  exit 69
fi
printf 'int main(void){return 0;}\\n' > "$BUILD_DIR/cc_probe.c"
if ! "$CC_BIN" -O0 -std=c11 "$BUILD_DIR/cc_probe.c" -o "$BUILD_DIR/cc_probe" 2>"$BUILD_DIR/cc_probe.log"; then
  echo "F26 decode requires a WORKING C compiler; ${CC_BIN} failed a trivial compile:" >&2
  sed -n '1,20p' "$BUILD_DIR/cc_probe.log" >&2
  exit 69
fi

"""

RUNTIME_DEPENDENCIES = [
    {"name": "python", "kind": "interpreter", "guard": "implicit (inflate.sh invokes it)"},
    {"name": "numpy", "kind": "python package", "guard": "import-time"},
    {"name": "torch", "kind": "python package", "guard": "import-time"},
    {"name": "Brotli==1.2.0", "kind": "python package",
     "guard": "fail-closed, self-installs via uv, exit 69 if uv absent (inherited)"},
    {"name": "cc (C compiler)", "kind": "build toolchain",
     "guard": "fail-closed probe added by ddm_fx1, exit 69 with named dep"},
    {"name": "uv", "kind": "tool",
     "guard": "required only when Brotli 1.2.0 is absent (inherited)"},
]
"""Declared decode-time dependencies.  The e4 precedent declares brotli; this
list exists so the compiler is declared with the same honesty rather than being
an undeclared hard dependency discovered on a host that lacks it."""

RACED_CODE_BITS = 879609.6594294705
"""The n600 code length this arm raced and MAIN byte-closed.  The staged tree
must reproduce it exactly or the staging is infidelitous."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_label(path: Path, keep: int = 2) -> str:
    """A location-free label for a local path: its last ``keep`` components."""
    parts = Path(path).resolve().parts
    return "/".join(parts[-keep:]) if len(parts) >= keep else str(Path(path).name)


def assert_tree_is_path_clean(output: Path) -> dict[str, object]:
    """Refuse a staged tree that embeds a local absolute path anywhere.

    Applied to every staged file, not only to the receipts, because the property
    the submission needs is that the PACKET is portable -- and a hardcoded
    /Volumes path in a runtime source file breaks that just as thoroughly as one
    in a receipt.
    """
    offenders: dict[str, list[str]] = {}
    text_scanned = 0
    binary_scanned = 0
    skipped_applefile = 0
    skipped_non_regular = 0
    byte_markers = [marker.encode() for marker in LEAK_MARKERS]
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if not path.is_file():
            # Broken symlinks, fifos and the like: invisible to this gate and to
            # the digest, yet `cp -R` ships them. Counted so the denominator is
            # not silently short.
            if not path.is_dir():
                skipped_non_regular += 1
            continue
        if _is_applefile(relative):
            skipped_applefile += 1
            continue
        # EVERY physically present file, including the ones the manifest excludes
        # (archive.zip, the receipt) -- what ships is what is on disk, not what
        # the manifest chose to hash.
        blob = path.read_bytes()
        try:
            blob.decode()
        except UnicodeDecodeError:
            binary_scanned += 1
        else:
            text_scanned += 1
        # Searched as BYTES so binary files are covered on the same footing.
        # The prior version skipped them on the claim that "binary payloads
        # carry no paths to leak".  That claim is false and it cost us: CPython
        # writes the absolute source path into every .pyc it emits, so two
        # decode-time bytecode caches sat in the staged tree carrying the drive
        # name and the arm codename, under a green hygiene report that had
        # simply never opened them.
        hits = sorted(
            {
                marker
                for marker, raw in zip(LEAK_MARKERS, byte_markers, strict=True)
                if raw in blob
            }
        )
        if hits:
            offenders[str(relative)] = hits
    if offenders:
        raise SystemExit(
            "staged tree leaks local absolute paths (packet-hygiene refusal): "
            + json.dumps(offenders, indent=2)
        )
    return {
        # The DENOMINATOR is reported, not just the verdict.  A scan that says
        # only "no hits" is indistinguishable from a scan that opened nothing.
        # The skip counts are MEASURED, never asserted: an earlier version of
        # this dict hardcoded files_skipped_unscanned=0 while the same receipt
        # reported 35 skipped sidecars -- the vacuity bug reappearing inside its
        # own cure.
        "text_files_scanned": text_scanned,
        "binary_files_scanned": binary_scanned,
        "total_files_scanned": text_scanned + binary_scanned,
        "files_skipped_unscanned": skipped_applefile + skipped_non_regular,
        "files_skipped_applefile": skipped_applefile,
        "files_skipped_non_regular": skipped_non_regular,
        "receipt_excluded_from_own_scan": True,
        "scan_note": (
            "Counted at receipt-write time, when the receipt does not yet exist. "
            "A later --verify-only run over the same untouched tree therefore "
            "reports exactly one more file. That difference is expected, not drift."
        ),
        "offending_files": len(offenders),
        # Reported WITHOUT their slashes on purpose: echoing the literal prefixes
        # would embed matchable absolute-path strings in the very receipt that
        # certifies the tree carries none, tripping any downstream scanner and
        # making this function refuse its own output.
        "leak_marker_labels": [marker.strip("/") for marker in LEAK_MARKERS],
    }


def assert_no_stray_files(output: Path) -> dict[str, object]:
    """Refuse a tree carrying files the digest EXCLUDES but the packet ships.

    Scope, stated precisely because an earlier docstring overclaimed it: this
    gate re-derives the manifest with the same ``_skip`` predicate that built it,
    so it catches only files ``_skip`` DROPS -- ``__pycache__`` and friends.  It
    cannot see a manifest file that was edited or replaced, because membership is
    then true by construction.  ``assert_tree_matches_receipt`` covers that
    second class, against the pinned receipt.  Neither gate subsumes the other.

    The tree digest cannot see this class by construction: it is computed over
    the files ``_skip`` keeps, so anything ``_skip`` drops is invisible to it no
    matter how loudly it leaks.  "Excluded from the hash" is not "absent from
    the packet" -- a judge copies the DIRECTORY, and gets whatever is in it.

    The empirical anchor is this arm's own tree.  Staging emitted a clean
    31-file manifest; the parse-back decode then imported two staged modules and
    CPython wrote 60,282 B of ``__pycache__`` bytecode back into the tree, each
    file embedding its absolute source path.  Every gate still read green: the
    digest excluded them, and the hygiene scan skipped them as binary.

    So this check runs over what is PHYSICALLY present and is exposed via
    ``--verify-only``, because the defect is created AFTER staging finishes.  A
    gate that can only run before the thing it guards against is a gate that
    cannot fire.
    """
    manifest: list[str] = []
    deliberate: list[str] = []
    sidecars: list[str] = []
    strays: dict[str, int] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        name = str(relative)
        if _is_applefile(relative):
            sidecars.append(name)
        elif not _skip(relative):
            manifest.append(name)
        elif name in SKIP_NAMES:
            deliberate.append(name)
        else:
            strays[name] = path.stat().st_size
    if strays:
        raise SystemExit(
            "staged tree carries files absent from the hashed manifest "
            "(stray-file refusal; these ship if the directory is copied):\n"
            + json.dumps(strays, indent=2)
        )
    return {
        "manifest_files": len(manifest),
        "deliberate_exclusions": sorted(deliberate),
        "stray_files": len(strays),
        "applefile_sidecars": len(sidecars),
        "applefile_sidecar_note": (
            "AppleDouble metadata written by the ExFAT staging volume, not "
            "packet content. They are counted rather than refused because the "
            "volume recreates them, and named here so the packaging step cannot "
            "forget them: strip with `find <tree> -name '._*' -delete` and "
            "re-run --verify-only before assembling the PR."
        ),
    }


def assert_receipt_is_path_clean(receipt: dict) -> None:
    """Hold the receipt to the SAME rule as the tree it certifies.

    ``assert_tree_is_path_clean`` walks the staged tree, but ``_skip`` excludes
    this receipt -- so without this second check the one file most likely to
    contain local paths would be the one file never inspected.  A checker that
    skips its own output is the vacuity failure: it passes by construction.
    """
    text = json.dumps(receipt)
    hits = sorted({marker for marker in LEAK_MARKERS if marker in text})
    if hits:
        raise SystemExit(
            f"staging receipt itself leaks local absolute paths: {hits}"
        )


def _git_commit() -> str:
    """The staging commit, or a refusal. An empty string is lost provenance."""
    completed = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        raise SystemExit(
            "cannot resolve the staging git commit; refusing to emit a receipt "
            f"with empty provenance (git rc={completed.returncode})"
        )
    return commit


def _manifest_of(output: Path) -> dict[str, str]:
    """The {relative path: sha256} map the tree digest is taken over."""
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and not _skip(path.relative_to(output))
    }


def _digest_of(manifest: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(sorted(manifest.items()))).encode()
    ).hexdigest()


def assert_tree_matches_receipt(output: Path) -> dict[str, object]:
    """Refuse a tree that has DRIFTED from the manifest its own receipt pins.

    ``assert_no_stray_files`` catches files the manifest never had.  It cannot
    catch a manifest file that was EDITED, REPLACED or REMOVED after staging,
    because it re-derives the manifest with the same predicate that built it --
    membership is then true by construction.  That is the staging-infidelity
    class this generator exists to close, so it is closed against the pinned
    receipt rather than against a re-derivation of the present state.

    Also re-checks the SHIPPED archive against ``archive_pins``.  ``repin_inflate``
    hashes the SOURCE archive it was handed, and ``archive.zip`` is excluded from
    the tree digest, so without this nothing in the receipt covers the bytes that
    actually ship -- a truncated copy would surface only at decode time, on paid
    contest hardware.
    """
    receipt_path = output / "FX1_STAGING_RECEIPT.json"
    if not receipt_path.is_file():
        raise SystemExit(
            f"no staging receipt in {output}: there is nothing to verify against. "
            "An empty or wrong directory must refuse, not certify itself clean."
        )
    receipt = json.loads(receipt_path.read_text())
    missing = [
        key for key in ("files", "runtime_tree_sha256", "archive_pins")
        if key not in receipt
    ]
    if missing:
        # A malformed receipt must refuse BY NAME, not KeyError out of a gate.
        raise SystemExit(
            f"staging receipt is missing required keys {missing}; it cannot "
            "certify this tree"
        )
    pinned = {name: entry["sha256"] for name, entry in receipt["files"].items()}
    physical = _manifest_of(output)

    added = sorted(set(physical) - set(pinned))
    removed = sorted(set(pinned) - set(physical))
    changed = sorted(k for k in set(pinned) & set(physical) if pinned[k] != physical[k])
    if added or removed or changed:
        raise SystemExit(
            "staged tree has DRIFTED from the manifest its receipt pins "
            "(staging-infidelity refusal):\n"
            + json.dumps(
                {"added": added, "removed": removed, "changed": changed}, indent=2
            )
        )

    digest = _digest_of(physical)
    if digest != receipt["runtime_tree_sha256"]:
        raise SystemExit(
            f"runtime tree digest mismatch: receipt pins {receipt['runtime_tree_sha256']}, "
            f"tree re-derives {digest}"
        )

    archive = output / "archive.zip"
    if not archive.is_file():
        raise SystemExit(f"staged tree carries no archive.zip: {output}")
    shipped_sha = sha256_file(archive)
    shipped_bytes = archive.stat().st_size
    pins = receipt["archive_pins"]
    if shipped_sha != pins["archive_sha256"] or shipped_bytes != pins["archive_bytes"]:
        raise SystemExit(
            "SHIPPED archive does not match the receipt pins:\n"
            f"  receipt {pins['archive_bytes']} B {pins['archive_sha256']}\n"
            f"  shipped {shipped_bytes} B {shipped_sha}"
        )
    return {
        "manifest_files_pinned": len(pinned),
        "manifest_files_present": len(physical),
        "added": added,
        "removed": removed,
        "changed": changed,
        "runtime_tree_sha256_rederived": digest,
        "shipped_archive_bytes": shipped_bytes,
        "shipped_archive_sha256": shipped_sha,
        "shipped_archive_matches_pins": True,
    }


def _is_applefile(path: Path) -> bool:
    """AppleDouble sidecar written by the ExFAT staging volume, not content."""
    return path.name.startswith(SKIP_PREFIXES)


def _skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    # SKIP_NAMES is anchored to the TREE ROOT, matched against the full relative
    # path rather than the basename.  Matching a bare name at any depth would let
    # a nested `runtime/archive.zip` be silently not-copied AND classified
    # "deliberate exclusion" instead of "stray" -- shipping un-hashed, unflagged.
    return path.name.startswith(SKIP_PREFIXES) or str(path) in SKIP_NAMES


def copy_base_tree(base: Path, output: Path) -> list[str]:
    """Copy every file that is not regenerated, preserving relative layout."""
    copied: list[str] = []
    for source in sorted(base.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(base)
        if _skip(relative):
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(str(relative))
    return copied


def stage_corrector(output: Path) -> dict[str, object]:
    """Stage both corrector modules; rewrite exactly one import shape."""
    rr4_destination = output / STAGED_RR4
    rr4_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RR4_SOURCE, rr4_destination)

    if sha256_file(RR4_SOURCE) != sha256_file(rr4_destination):
        # Reported-but-unenforced is how a false lands in a receipt and ships.
        raise SystemExit("staged rr4 copy is not byte-identical to its source")

    source_text = FX1_SOURCE.read_text()
    rewrites = source_text.count(IMPORT_FROM)
    if rewrites != EXPECTED_IMPORT_REWRITES:
        raise SystemExit(
            f"expected {EXPECTED_IMPORT_REWRITES} occurrences of {IMPORT_FROM!r} in "
            f"{FX1_SOURCE.name}, found {rewrites}; the staging transformation is stale"
        )
    staged_text = source_text.replace(IMPORT_FROM, IMPORT_TO)
    if IMPORT_FROM in staged_text:
        raise SystemExit("import rewrite left an absolute import behind")
    (output / STAGED_FX1).write_text(staged_text)

    return {
        "rr4_source": str(RR4_SOURCE.relative_to(REPO)),
        "rr4_source_sha256": sha256_file(RR4_SOURCE),
        "rr4_staged_sha256": sha256_file(rr4_destination),
        "rr4_byte_identical": sha256_file(RR4_SOURCE) == sha256_file(rr4_destination),
        "fx1_source": str(FX1_SOURCE.relative_to(REPO)),
        "fx1_source_sha256": sha256_file(FX1_SOURCE),
        "fx1_staged_sha256": sha256_file(output / STAGED_FX1),
        "import_rewrites": rewrites,
        "import_from": IMPORT_FROM,
        "import_to": IMPORT_TO,
    }


def assert_no_transcendental(path: Path) -> None:
    """The rr2 desync class must stay structurally impossible in the SHIPPED copy.

    The repo module is gated by its own test; the staged copy is a different file
    and is therefore gated again here.  Auditing only the source would leave the
    artifact that actually ships unchecked.
    """
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            offenders.append(f"line {node.lineno}: ** operator")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            offenders.append(f"line {node.lineno}: .{node.attr}")
        if isinstance(node, ast.Name) and node.id in BANNED_ATTRS:
            offenders.append(f"line {node.lineno}: {node.id}")
    if offenders:
        raise SystemExit(f"staged corrector reintroduced a transcendental: {offenders}")


def guard_c_toolchain(output: Path) -> dict[str, object]:
    """Insert the fail-closed compiler probe ahead of the RC64 compile."""
    path = output / "inflate.sh"
    text = path.read_text()
    occurrences = text.count(CC_ANCHOR)
    if occurrences != 1:
        raise SystemExit(
            f"expected exactly 1 RC64 compile anchor in inflate.sh, found {occurrences}; "
            "the toolchain guard would land in the wrong place"
        )
    path.write_text(text.replace(CC_ANCHOR, CC_GUARD + CC_ANCHOR))
    guarded = path.read_text()
    if 'command -v "$CC_BIN"' not in guarded or "cc_probe.c" not in guarded:
        raise SystemExit("toolchain guard did not land")
    path.chmod(0o755)
    return {
        "anchor_occurrences": occurrences,
        "guard_present": True,
        "probes_a_real_compile": True,
        "exit_code_on_missing": 69,
    }


def repin_inflate(output: Path, archive: Path) -> dict[str, object]:
    """Rewrite the archive identity pins the receiver fails closed on."""
    path = output / "inflate.py"
    text = path.read_text()
    digest = sha256_file(archive)
    size = archive.stat().st_size

    lines, replaced_sha, replaced_bytes = [], 0, 0
    for line in text.splitlines(keepends=True):
        if line.startswith("ARCHIVE_SHA256 = "):
            lines.append(f'ARCHIVE_SHA256 = "{digest}"\n')
            replaced_sha += 1
        elif line.startswith("ARCHIVE_BYTES = "):
            lines.append(f"ARCHIVE_BYTES = {size:_}\n")
            replaced_bytes += 1
        else:
            lines.append(line)
    if replaced_sha != 1 or replaced_bytes != 1:
        raise SystemExit(
            f"expected exactly one pin of each kind, rewrote {replaced_sha} sha / "
            f"{replaced_bytes} bytes; inflate.py layout changed"
        )
    path.write_text("".join(lines))
    return {"archive_sha256": digest, "archive_bytes": size}


def verify_staged_replay(output: Path) -> dict[str, object]:
    """Re-measure the full n600 code length THROUGH THE STAGED MODULE.

    This is the anti-staging-infidelity gate.  It imports the staged file itself
    -- not the repo source -- and refuses the tree unless it reproduces the raced
    code length exactly.
    """
    script = f"""
import importlib.util, sys, json
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, "/Volumes/APDataStore/pact/ddm_me1")
# Give the staged relative import a package to resolve against.
import types
pkg = types.ModuleType("staged_runtime"); pkg.__path__ = [{str(output / "runtime")!r}]
sys.modules["staged_runtime"] = pkg
spec = importlib.util.spec_from_file_location(
    "staged_runtime.rr4_free_corrector", {str(output / STAGED_RR4)!r})
rr4 = importlib.util.module_from_spec(spec); sys.modules[spec.name] = rr4
spec.loader.exec_module(rr4)
spec = importlib.util.spec_from_file_location(
    "staged_runtime.free_corrector", {str(output / STAGED_FX1)!r})
fx1 = importlib.util.module_from_spec(spec); sys.modules[spec.name] = fx1
spec.loader.exec_module(fx1)
from assets import ASSETS
from tac.micro_edit.coder_replay import replay_code_length
result = replay_code_length(ASSETS, label="staged", corrector_factory=fx1.FreeCorrector)
print(json.dumps({{"code_bits": result.code_bits, "code_bytes": result.code_bytes}}))
"""
    # -B: never write bytecode.  Without it this very subprocess imports the two
    # staged correctors and CPython drops a __pycache__ directory INSIDE the tree
    # -- files that embed their absolute source path and that the tree digest
    # cannot see, because it excludes them.  The staging step would then create
    # the exact leak the staging step just certified absent.
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode != 0:
        raise SystemExit(f"staged replay failed:\n{completed.stderr[-3000:]}")
    lines = completed.stdout.strip().splitlines()
    try:
        measured = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        # rc=0 with unparseable output is a staging failure, not a crash site.
        raise SystemExit(
            f"staged replay returned rc=0 but no parseable result line: {exc}\n"
            f"stdout tail:\n{completed.stdout[-2000:]}"
        ) from exc
    delta = measured["code_bits"] - RACED_CODE_BITS
    if delta != 0.0:
        raise SystemExit(
            f"STAGING INFIDELITY: staged tree measures {measured['code_bits']!r} bits, "
            f"raced {RACED_CODE_BITS!r}, delta {delta:+.6f}"
        )
    return {
        "code_bits": measured["code_bits"],
        "code_bytes": measured["code_bytes"],
        "raced_code_bits": RACED_CODE_BITS,
        "delta_bits": delta,
        "bit_identical_to_raced": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tree", type=Path, default=DEFAULT_BASE_TREE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--skip-replay", action="store_true",
                        help="Skip the code-length replay; for iterating on the "
                             "staging only. The receipt is STILL written, marked "
                             "fidelity_verified=false. Never fire such a tree.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Re-check an EXISTING staged tree for path leaks and "
                             "stray files, then exit. Stages nothing. Run this "
                             "after any decode and immediately before packaging: "
                             "the decode writes __pycache__ back into the tree, "
                             "which staging-time checks are too early to catch.")
    args = parser.parse_args()

    if args.verify_only:
        if not args.output.is_dir():
            raise SystemExit(f"missing staged tree: {args.output}")
        verdict = {
            "packet_hygiene": assert_tree_is_path_clean(args.output),
            "packet_strays": assert_no_stray_files(args.output),
            "receipt_fidelity": assert_tree_matches_receipt(args.output),
        }
        print(json.dumps(verdict, indent=2))
        return 0

    for required in (args.base_tree, args.archive, RR4_SOURCE, FX1_SOURCE):
        if not required.exists():
            raise SystemExit(f"missing input: {required}")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    copied = copy_base_tree(args.base_tree, args.output)
    corrector = stage_corrector(args.output)
    # BOTH staged correctors: the rr2 desync-class constants live in the rr4 half,
    # and the argument for gating the staged copy applies to it identically.
    assert_no_transcendental(args.output / STAGED_FX1)
    assert_no_transcendental(args.output / STAGED_RR4)
    shutil.copyfile(args.archive, args.output / "archive.zip")
    pins = repin_inflate(args.output, args.archive)
    toolchain = guard_c_toolchain(args.output)

    # The replay EXECUTES staged code, so it runs BEFORE the digest is taken --
    # the digest must describe the tree as it finally stands, not a state a later
    # step could still invalidate.  This is the same ordering error the hygiene
    # gates already had.
    replay = None if args.skip_replay else verify_staged_replay(args.output)

    tree = {
        str(path.relative_to(args.output)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and not _skip(path.relative_to(args.output))
    }
    tree_digest = _digest_of({k: v["sha256"] for k, v in tree.items()})

    receipt = {
        "schema": "ddm_fx1_staging_receipt.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": "experiments/ddm_fx1_stage_candidate_runtime.py",
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "archive_source_label": portable_label(args.archive, keep=3),
        "git_commit": _git_commit(),
        "base_tree_label": portable_label(args.base_tree),
        "files_copied_from_base": len(copied),
        "corrector_staging": corrector,
        "archive_pins": pins,
        "toolchain_guard": toolchain,
        "runtime_dependencies": RUNTIME_DEPENDENCIES,
        "archive_is_a_parameter": {
            "note": (
                "The tree is archive-AGNOSTIC by construction: --archive is a "
                "parameter and both inflate.py pins are DERIVED from that file, "
                "never hardcoded here. Swapping a section inside the archive "
                "(e.g. a re-solved dxi/pose section of identical byte length) "
                "therefore needs a new encoder run and a re-stage with --archive, "
                "and NO change to this generator or to the staged corrector."
            ),
            "compose_path": [
                "1. encoder byte-close with the fx1 corrector + the swapped section",
                "2. re-run this generator with --archive <new archive.zip>",
                "3. re-run the parse-back falsifier against the new tree",
            ],
        },
        "runtime_tree_sha256": tree_digest,
        "runtime_tree_digest_excludes": [
            "archive.zip -- the COUNTED artifact, pinned separately in archive_pins",
            "FX1_STAGING_RECEIPT.json -- carries a timestamp, so including it would "
            "make the digest non-deterministic and destroy the re-run check",
        ],
        "file_count": len(tree),
        "files": tree,
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
    }
    # An unverified tree must SAY so in a field a consumer reads, not merely omit
    # a key: absence-of-key is far too weak a signal for a tree headed to paid
    # contest hardware.
    receipt["fidelity_verified"] = replay is not None
    receipt["staged_replay_verification"] = (
        replay if replay is not None else {"skipped": True, "reason": "--skip-replay"}
    )

    # LAST, after every step that can touch the tree -- the replay included.
    # Running these earlier certified a state that later steps then invalidated,
    # which is how two leaking bytecode files shipped under a green report.
    receipt["packet_hygiene"] = assert_tree_is_path_clean(args.output)
    receipt["packet_strays"] = assert_no_stray_files(args.output)

    assert_receipt_is_path_clean(receipt)
    (args.output / "FX1_STAGING_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(f"staged {len(tree)} files -> {args.output}")
    print(f"  runtime_tree_sha256 {tree_digest}")
    print(f"  archive             {pins['archive_bytes']} B  {pins['archive_sha256'][:16]}...")
    if not args.skip_replay:
        print(f"  staged replay       {receipt['staged_replay_verification']['code_bits']!r} bits"
              f"  delta {receipt['staged_replay_verification']['delta_bits']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
