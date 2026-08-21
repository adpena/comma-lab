# SPDX-License-Identifier: MIT
"""Canonical submission chain: state -> archive.zip -> inflate -> evaluate -> receipt.

Operator binding 2026-08-03: *"Anything that is necessary for full, byte closed,
archived submission should not be ad hoc or manual or exist in temporary probe
scripts or anything like that ... it should be canonicalized for our own frontier
work."*

This module is the ONE vehicle-agnostic implementation of the four steps that
every own-vehicle frontier row must pass through.  It does not build payloads --
that is the vehicle's job (``tac.optimization.ddm_ix2_archive_container`` for the
ix2/v4d container line) -- it takes a finished payload and closes it:

``build_byte_ledger``
    The standing per-section byte ledger for the ix2/v4d container -- the format
    the live own-vehicle frontier actually ships.  Before this existed, every
    section size in every memo about that archive was re-derived BY HAND by
    parsing the live zip, which is the "measured once, banked nowhere" class.

    TWO SIBLING LEDGERS ALREADY EXIST and are deliberately NOT duplicated here
    (anti-duplicate-SoT #533 requires disclosing them rather than pretending to
    be first):

    * ``tac.optimization.ddm_tr1_runtime.section_ledger`` -- ``{name, bytes,
      sha256, consumer}`` for the TR1 four-section PACKET DIRECTORY format;
    * ``tac.boundary_math.integer_plane_emitter_byte_close.archive_receipt`` --
      ``{name, uncompressed_bytes, compressed_bytes, sha256}`` plus overhead and
      a custody refusal, for the C2 integer-plane archive.

    Neither parses the ix2 two-tier payload (one stored bulk section + one
    jointly-coded group), which is why a third exists; the field names here are
    taken FROM the C2 sibling so the three agree on vocabulary.  If the ix2
    container is ever retired, this ledger retires with it.

    The ledger CLOSES exactly or raises.  The closure is deliberately
    NON-CIRCULAR: the ZIP framing is PREDICTED independently by the container's
    own ``zip_framing_overhead`` and compared against the measured remainder, and
    the payload is RE-ENCODED and compared byte-for-byte.  An earlier draft
    computed the framing by subtraction, which made the residual identically
    zero by algebra -- a guard that could never fire, i.e. the exact
    vacuity-equals-pass class this module exists to refuse.  It also separates
    the *raw* section size from the *counted* coded-group size, so a reader can
    never mistake a pre-coder number for a rate cost.

``audit_runtime_tree``
    Custody for the vendored contest runtime.  Each staged file is hashed and
    matched against its repo counterpart (which may carry a DIFFERENT basename:
    the shipped ``inflate_runner.py`` is ``experiments/inflate_runner_v4d.py``),
    and the import graph is closed TRANSITIVELY from the ``inflate.sh`` entry
    point.  The transitivity is load-bearing: ``repair_entropy_coder_runtime_
    adapters.py`` is NOT imported by ``inflate_runner`` and reads as dead weight
    in a one-level scan, but ``ddm_r7_token_coder`` imports it, so deleting it
    breaks inflate.  A one-level reachability audit would have shipped that bug.

``run_inflate`` / ``run_upstream_evaluate``
    Foreground subprocess execution with the return code CAPTURED and REFUSED on
    non-zero.  ``ddm_si1`` (task #929) measured that the inflate script itself
    was never the liar -- run in the foreground the pre-fix bare-``python`` form
    returns 127 correctly -- and that the real amplifier is a BACKGROUNDING
    launcher whose rc belongs to the launcher, not the job.  This module
    therefore never backgrounds, and additionally refuses a run that returns
    rc=0 while producing no output (vacuity != pass, ``m50``).

``ChainReceipt``
    The typed, provenance-stamped result: git hash, upstream snapshot sha, seed,
    archive sha256 + size, the byte ledger, the custody audit, and the score
    components RECOMPUTED from parts via ``tac.contest_score``.  Never the
    rounded ``final_score`` field alone.

Axis discipline is unchanged and enforced here: MPS is REFUSED outright, a
non-Linux-x86_64 CPU row is labelled ``[macOS-CPU advisory]`` and never
``[contest-CPU]``, and every receipt carries ``score_claim=False`` unless the
caller is on contest-compliant hardware.  Evidence produced by this module is
advisory on this host; only ``upstream/evaluate.py`` on contest hardware over the
exact shipped bytes is a score.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from tac.process_group_kill import run_in_process_group

__all__ = [
    "CANONICAL_UNCOMPRESSED_SIZE",
    "ArchiveByteLedger",
    "ChainPaths",
    "ChainReceipt",
    "EvaluateResult",
    "InflateResult",
    "RuntimeFileCustody",
    "RuntimeTreeCustody",
    "SectionRow",
    "SubmissionChainError",
    "ZipMemberRow",
    "advisory_axis_label",
    "audit_runtime_tree",
    "axis_and_authority",
    "build_byte_ledger",
    "git_hash",
    "refuse_transient_path",
    "run_inflate",
    "run_upstream_evaluate",
    "sha256_bytes",
    "sha256_file",
    "stage_submission",
    "verify_archive_identity",
]

CANONICAL_UNCOMPRESSED_SIZE: Final = 37_545_489

# The contest entry point is a shell script taking (archive_dir, out_dir, names).
_INFLATE_ENTRY: Final = "inflate.sh"

# upstream/evaluate.py prints this block; a missing REQUIRED field raises rather
# than defaulting, because a fabricated component is a fabricated score.
_EVAL_REPORT_PATTERNS: Final = {
    "d_pose": r"Average PoseNet Distortion:\s*([0-9.eE+\-]+)",
    "d_seg": r"Average SegNet Distortion:\s*([0-9.eE+\-]+)",
    "rate": r"Compression Rate:\s*([0-9.eE+\-]+)",
    "final_score": r"Final score[^=]*=\s*([0-9.eE+\-]+)",
    "n_samples": r"Evaluation results over\s*([0-9]+)\s*samples",
}


class SubmissionChainError(RuntimeError):
    """Raised when a chain step does not close exactly.  Always fail CLOSED."""


# --------------------------------------------------------------------------- #
# small shared primitives (thin wrappers so this module stays importable when
# vendored flat into a contest runtime tree, where ``tac`` is not installed)
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of an in-memory payload."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, streamed."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_hash(repo_root: str | Path) -> str:
    """Current ``HEAD`` sha, or ``"unknown"`` when git cannot answer.

    NOTE the asymmetry: an unknown git hash degrades provenance but must not
    abort a measurement, whereas an unknown *return code* aborts everything.
    """

    try:
        out = subprocess.run(  # GROUP_KILL_OK: `git rev-parse` is a single leaf binary — it spawns nothing, so there is no grandchild for a group kill to reach.
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def refuse_transient_path(path: str | Path, field_name: str) -> Path:
    """Refuse ``/tmp``-rooted durable evidence paths (CLAUDE.md non-negotiable).

    A ``/tmp`` path does not survive a fresh checkout and cannot be verified by
    another agent, so citing one as evidence produces a phantom receipt.
    """

    resolved = Path(path).expanduser()
    parts = resolved.resolve().parts if resolved.is_absolute() else resolved.parts
    if parts[:2] in (("/", "tmp"), ("/", "private")) and "tmp" in parts[:3]:
        raise SubmissionChainError(
            f"{field_name}={resolved} is under a transient tmp root; durable "
            "evidence paths only (CLAUDE.md 'Forbidden /tmp paths in any "
            "persisted artifact')."
        )
    return resolved


def _is_linux_x86_64() -> bool:
    return platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")


def advisory_axis_label() -> str:
    """Axis label for a CPU row on THIS host.

    ``[contest-CPU]`` is reserved for Linux x86_64 (the contest CI family).
    Apple Silicon / macOS CPU is ``[macOS-CPU advisory]`` and NON-PROMOTABLE.
    """

    return "[contest-CPU]" if _is_linux_x86_64() else "[macOS-CPU advisory]"


def axis_and_authority(device: str) -> tuple[str, str]:
    """``(score_axis, authority)`` from the ACTUAL device first, host second.

    CPU and CUDA are separate evidence spaces and neither may be inferred from
    the other.  MPS is never an authority anywhere.
    """

    dev = device.lower()
    if dev == "mps":
        raise SubmissionChainError(
            "MPS is NEVER a score authority (CLAUDE.md 'MPS auth eval is NOISE'). "
            "Use --device cpu or --device cuda."
        )
    if dev == "cuda":
        return "[contest-CUDA]", "authority"
    if dev == "cpu":
        axis = advisory_axis_label()
        return axis, ("authority" if axis == "[contest-CPU]" else "advisory")
    raise SubmissionChainError(f"unknown eval device {device!r}; expected cpu or cuda")


# --------------------------------------------------------------------------- #
# 1. the standing BYTE LEDGER
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ZipMemberRow:
    """One ZIP member as the rate term actually sees it.

    ``compressed_bytes`` is what the archive pays; ``uncompressed_bytes`` is what
    the receiver reads.  They differ for any non-STORED member, and only the
    former is a rate cost.
    """

    name: str
    uncompressed_bytes: int
    compressed_bytes: int
    compress_type: int
    sha256: str


@dataclass(frozen=True)
class SectionRow:
    """One payload section.

    ``raw_bytes`` is the section's size BEFORE the joint group coder.  It is not
    a rate cost on its own: the counted cost of the joint group is
    ``ArchiveByteLedger.joint_coded_bytes``.  Keeping the two named apart is the
    point -- quoting a raw section size as a byte cost is the single easiest way
    to publish a number the archive does not honour.
    """

    index: int
    name: str
    raw_bytes: int
    sha256: str
    magic: str | None
    counted: bool


@dataclass(frozen=True)
class ArchiveByteLedger:
    """Per-section byte accounting that CLOSES exactly on the archive bytes."""

    archive_path: str
    archive_sha256: str
    archive_bytes: int
    zip_members: tuple[ZipMemberRow, ...]
    zip_framing_bytes: int
    payload_bytes: int
    payload_header_bytes: int
    bulk_bytes: int
    bulk_sha256: str
    joint_count_byte: int
    joint_coded_bytes: int
    joint_raw_bytes: int
    joint_coder_saving_bytes: int
    sections: tuple[SectionRow, ...]
    predicted_framing_bytes: int
    accounted_bytes: int
    residual_bytes: int
    payload_reencodes_identically: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def closes(self) -> bool:
        """Both independent checks must hold.

        ``residual_bytes`` compares the PREDICTED zip framing against the
        measured remainder (so a comment, an extra field, a renamed member, or a
        non-STORED member all move it off zero), and
        ``payload_reencodes_identically`` re-runs the canonical encoder over the
        parsed parts.  Neither is derivable from the other.
        """

        return self.residual_bytes == 0 and self.payload_reencodes_identically


# Section names for the ix2/v4d joint group, in shipped order.
#
# The first four are the base v4d group.  ``frame0_pose_repair`` is the OPTIONAL
# fifth section (ddm_fz1's F0PR1 int16 DCT frame_0 pose-repair stream): archives
# that do not carry it simply have four sections, and the extra name is unused,
# so naming it here cannot change any existing ledger.  It is named rather than
# left to the ``section_{i}`` fallback because an unnamed counted section is a
# section no reader can attribute bytes to -- the ledger would still CLOSE while
# reporting "section_4", which is exactly the silent-instrument shape.
_IX2_JOINT_NAMES: Final = (
    "config",
    "renderer",
    "selector",
    "pose_warp",
    "frame0_pose_repair",
)


def _section_magic(blob: bytes) -> str | None:
    """Return a printable leading magic when the section declares one."""

    head = blob[:8]
    if len(head) == 8 and all(32 <= b < 127 for b in head):
        return head.decode("ascii")
    if blob[:1] == b"{":
        return "json"
    return None


def build_byte_ledger(
    archive_path: str | Path,
    *,
    joint_names: tuple[str, ...] = _IX2_JOINT_NAMES,
    payload_member: str = "0.bin",
    require_stored: bool = True,
) -> ArchiveByteLedger:
    """Parse ``archive.zip`` into a closing per-section byte ledger.

    Raises ``SubmissionChainError`` when the accounting does not close, which is
    the whole value of the artifact: a ledger that silently absorbs a residual
    is a ledger that cannot detect a format change.
    """

    from tac.optimization.ddm_ix2_archive_container import (
        build_payload,
        parse_payload,
        zip_framing_overhead,
    )

    path = Path(archive_path)
    if not path.exists():
        raise SubmissionChainError(f"archive not found: {path}")
    archive_bytes = path.stat().st_size

    members: list[ZipMemberRow] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            blob = zf.read(info.filename)
            members.append(
                ZipMemberRow(
                    name=info.filename,
                    uncompressed_bytes=len(blob),
                    compressed_bytes=int(info.compress_size),
                    compress_type=int(info.compress_type),
                    sha256=sha256_bytes(blob),
                )
            )
        if payload_member not in zf.namelist():
            raise SubmissionChainError(
                f"archive {path} has no {payload_member!r} member "
                f"(members: {zf.namelist()})"
            )
        payload = zf.read(payload_member)

    bulk, sections = parse_payload(payload)

    # payload = <I bulk_len><B coder_id> bulk <B count> coded_joint_block
    header_bytes = 5
    count_byte = 1
    joint_coded = len(payload) - header_bytes - len(bulk) - count_byte
    if joint_coded < 0:
        raise SubmissionChainError(
            f"payload accounting went negative (payload={len(payload)}, "
            f"bulk={len(bulk)}); the container format changed under this ledger."
        )

    rows: list[SectionRow] = []
    for i, sec in enumerate(sections):
        rows.append(
            SectionRow(
                index=i,
                name=joint_names[i] if i < len(joint_names) else f"section_{i}",
                raw_bytes=len(sec),
                sha256=sha256_bytes(sec),
                magic=_section_magic(sec),
                counted=True,
            )
        )
    joint_raw = sum(r.raw_bytes for r in rows)

    # The canonical packer (``build_single_member_zip``) writes ZIP_STORED, and
    # ``zip_framing_overhead`` models the STORED layout.  A non-STORED member
    # therefore means the archive did NOT come from our packer -- a repack, a
    # third-party rezip, or a different grammar -- and its framing prediction is
    # not valid.  Refuse rather than emit a ledger with an unmodelled layout.
    if require_stored:
        bad = [m.name for m in members if m.compress_type != zipfile.ZIP_STORED]
        if bad:
            raise SubmissionChainError(
                f"archive {path} has non-STORED member(s) {bad}; the canonical "
                "ix2 packer emits ZIP_STORED only, so this archive was not "
                "produced by it and its byte accounting is not modelled here."
            )

    # INDEPENDENT closure. The framing is PREDICTED from the member names by the
    # container's own accounting function -- never back-computed by subtraction,
    # which would make the residual identically zero by algebra.
    predicted_framing = zip_framing_overhead(
        [m.name for m in members], [m.compressed_bytes for m in members]
    )
    accounted = predicted_framing + sum(m.compressed_bytes for m in members)
    residual = archive_bytes - accounted

    # Second, independent check: the canonical encoder must reproduce the exact
    # payload bytes from the parsed parts.
    try:
        reencodes = build_payload(bulk, list(sections)) == payload
    except Exception:  # any encoder failure is itself a non-closure
        reencodes = False

    ledger = ArchiveByteLedger(
        archive_path=str(path),
        archive_sha256=sha256_file(path),
        archive_bytes=archive_bytes,
        zip_members=tuple(members),
        zip_framing_bytes=archive_bytes - sum(m.compressed_bytes for m in members),
        payload_bytes=len(payload),
        payload_header_bytes=header_bytes,
        bulk_bytes=len(bulk),
        bulk_sha256=sha256_bytes(bulk),
        joint_count_byte=count_byte,
        joint_coded_bytes=joint_coded,
        joint_raw_bytes=joint_raw,
        joint_coder_saving_bytes=joint_raw - joint_coded,
        sections=tuple(rows),
        predicted_framing_bytes=predicted_framing,
        accounted_bytes=accounted,
        residual_bytes=residual,
        payload_reencodes_identically=reencodes,
    )
    if not ledger.closes():
        raise SubmissionChainError(
            f"byte ledger did not close: archive={archive_bytes} accounted="
            f"{accounted} residual={residual} (predicted framing "
            f"{predicted_framing}, measured {ledger.zip_framing_bytes}), "
            f"payload_reencodes_identically={reencodes}.  Refusing to emit a "
            "ledger whose own arithmetic is wrong."
        )
    return ledger


# --------------------------------------------------------------------------- #
# 2. vendored runtime custody
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuntimeFileCustody:
    """One staged runtime file: its hash, its repo counterpart, its verdict."""

    staged_name: str
    staged_sha256: str
    staged_bytes: int
    repo_path: str | None
    repo_sha256: str | None
    verdict: str  # IDENTICAL | DIVERGED | UNMAPPED
    reached: bool
    divergence_reason: str | None = None


@dataclass(frozen=True)
class RuntimeTreeCustody:
    """Custody verdict for the whole vendored contest runtime tree."""

    staged_dir: str
    entry_script: str
    entry_module: str | None
    files: tuple[RuntimeFileCustody, ...]
    reached_modules: tuple[str, ...]
    unreached_files: tuple[str, ...]
    identical_count: int
    diverged_count: int
    unmapped_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entry_module_from_inflate_sh(script: Path) -> str | None:
    """Extract the python entry module name invoked by ``inflate.sh``."""

    text = script.read_text()
    mobj = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\.py", text)
    return mobj.group(1) if mobj else None


def _module_imports(path: Path) -> set[str]:
    """Top-level module names imported by a python source file (AST, not regex).

    Regex over import lines misses ``from x import (`` continuations and counts
    strings; the AST does not.
    """

    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):  # pragma: no cover - malformed staged file
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _reachable_modules(staged_dir: Path, entry_module: str) -> set[str]:
    """TRANSITIVE import closure over modules that exist IN the staged tree.

    Transitivity is the point.  ``repair_entropy_coder_runtime_adapters`` is not
    imported by the entry module at all -- it is pulled in by
    ``ddm_r7_token_coder`` -- so a one-level scan reports it unreached and a
    reader concludes it is deletable counted-tree dead weight.  It is not.
    """

    local = {p.stem for p in staged_dir.glob("*.py")}
    if entry_module not in local:
        return set()
    reached: set[str] = set()
    frontier = [entry_module]
    while frontier:
        mod = frontier.pop()
        if mod in reached:
            continue
        reached.add(mod)
        for dep in _module_imports(staged_dir / f"{mod}.py"):
            if dep in local and dep not in reached:
                frontier.append(dep)
    return reached


def audit_runtime_tree(
    staged_dir: str | Path,
    *,
    repo_root: str | Path,
    repo_map: dict[str, str] | None = None,
    entry_script: str = _INFLATE_ENTRY,
) -> RuntimeTreeCustody:
    """Hash every staged runtime file and classify it against its repo source.

    ``repo_map`` maps staged basename -> repo-relative path, because the shipped
    name is not always the repo name (``inflate_runner.py`` is the repo's
    ``experiments/inflate_runner_v4d.py``).  An unmapped file is reported as
    ``UNMAPPED``, never silently treated as clean.
    """

    staged = Path(staged_dir)
    root = Path(repo_root)
    if not staged.is_dir():
        raise SubmissionChainError(f"staged runtime tree not found: {staged}")
    script = staged / entry_script
    if not script.exists():
        raise SubmissionChainError(f"entry script missing: {script}")

    entry_module = _entry_module_from_inflate_sh(script)
    reached = _reachable_modules(staged, entry_module) if entry_module else set()

    repo_map = dict(repo_map or {})
    rows: list[RuntimeFileCustody] = []
    for item in sorted(staged.iterdir()):
        if not item.is_file() or item.suffix not in (".py", ".sh"):
            continue
        staged_sha = sha256_file(item)
        rel = repo_map.get(item.name)
        repo_sha: str | None = None
        verdict = "UNMAPPED"
        reason: str | None = None
        if rel:
            candidate = root / rel
            if candidate.exists():
                repo_sha = sha256_file(candidate)
                if repo_sha == staged_sha:
                    verdict = "IDENTICAL"
                else:
                    verdict = "DIVERGED"
                    reason = (
                        "staged bytes differ from the repo counterpart; the "
                        "shipped receiver is a PINNED vendored copy and repo "
                        "HEAD has moved. Re-staging from HEAD would ship a "
                        "different receiver."
                    )
            else:
                reason = f"mapped repo path does not exist: {rel}"
        rows.append(
            RuntimeFileCustody(
                staged_name=item.name,
                staged_sha256=staged_sha,
                staged_bytes=item.stat().st_size,
                repo_path=rel,
                repo_sha256=repo_sha,
                verdict=verdict,
                reached=(item.stem in reached) or item.name == entry_script,
                divergence_reason=reason,
            )
        )

    unreached = tuple(
        r.staged_name for r in rows if not r.reached and r.staged_name.endswith(".py")
    )
    return RuntimeTreeCustody(
        staged_dir=str(staged),
        entry_script=entry_script,
        entry_module=entry_module,
        files=tuple(rows),
        reached_modules=tuple(sorted(reached)),
        unreached_files=unreached,
        identical_count=sum(1 for r in rows if r.verdict == "IDENTICAL"),
        diverged_count=sum(1 for r in rows if r.verdict == "DIVERGED"),
        unmapped_count=sum(1 for r in rows if r.verdict == "UNMAPPED"),
    )


# --------------------------------------------------------------------------- #
# 3. path configuration (no hard-coded SSD roots)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChainPaths:
    """Every path the chain needs, resolved from explicit args or environment.

    The probe-script predecessor hard-coded five absolute ``/Volumes/...`` paths,
    so it ran on exactly one machine with exactly one SSD mounted and failed
    opaquely otherwise.  Every field here is overridable and every one is
    existence-checked BEFORE any expensive step starts.
    """

    repo_root: Path
    upstream_dir: Path
    runtime_src: Path
    videos_dir: Path
    video_names_file: Path
    work_dir: Path

    @staticmethod
    def from_env(
        *,
        repo_root: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        runtime_src: str | Path | None = None,
        videos_dir: str | Path | None = None,
        video_names_file: str | Path | None = None,
        work_dir: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> ChainPaths:
        """Resolve with precedence: explicit arg > environment > repo default."""

        env = dict(env if env is not None else os.environ)

        def pick(arg: Any, var: str, default: Any) -> Path:
            if arg is not None:
                return Path(arg)
            if env.get(var):
                return Path(env[var])
            return Path(default)

        root = Path(repo_root or env.get("PACT_REPO_ROOT") or _default_repo_root())
        return ChainPaths(
            repo_root=root,
            upstream_dir=pick(upstream_dir, "PACT_UPSTREAM_DIR", root / "upstream"),
            runtime_src=pick(runtime_src, "PACT_RUNTIME_SRC", root / "submissions" / "robust_current"),
            videos_dir=pick(videos_dir, "PACT_VIDEOS_DIR", root / "upstream" / "videos"),
            video_names_file=pick(
                video_names_file,
                "PACT_VIDEO_NAMES_FILE",
                root / "upstream" / "public_test_video_names.txt",
            ),
            work_dir=pick(work_dir, "PACT_CHAIN_WORK_DIR", root / "experiments" / "results" / "submission_chain"),
        )

    def preflight(self, *, require_eval_inputs: bool = True) -> list[str]:
        """Return the list of missing required paths (empty == ready).

        Returns rather than raises so a caller can report EVERY missing path at
        once; a preflight that dies on the first miss makes the operator
        discover the set one round-trip at a time.
        """

        missing: list[str] = []
        checks: list[tuple[str, Path]] = [
            ("repo_root", self.repo_root),
            ("runtime_src", self.runtime_src),
        ]
        if require_eval_inputs:
            checks += [
                ("upstream_dir", self.upstream_dir),
                ("videos_dir", self.videos_dir),
                ("video_names_file", self.video_names_file),
            ]
        for name, path in checks:
            refuse_transient_path(path, name)
            if not path.exists():
                missing.append(f"{name}={path}")
        if require_eval_inputs and self.upstream_dir.exists() and not (
            self.upstream_dir / "evaluate.py"
        ).exists():
            missing.append(f"upstream_dir/evaluate.py={self.upstream_dir / 'evaluate.py'}")
        refuse_transient_path(self.work_dir, "work_dir")
        return missing


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# 4. stage / verify / inflate / evaluate
# --------------------------------------------------------------------------- #
def stage_submission(
    payload: bytes,
    *,
    dest: str | Path,
    runtime_src: str | Path,
    runtime_files: tuple[str, ...],
    payload_member: str = "0.bin",
) -> Path:
    """Write ``dest/archive.zip`` from ``payload`` and copy the runtime tree.

    Uses the SAME deterministic single-member STORED packer the encoder uses, so
    a re-stage of identical inputs is byte-identical by construction.
    """

    from tac.optimization.ddm_ix2_archive_container import build_single_member_zip

    dest_dir = refuse_transient_path(dest, "dest")
    src = Path(runtime_src)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in runtime_files:
        source = src / name
        if not source.exists():
            raise SubmissionChainError(f"runtime file missing from {src}: {name}")
        shutil.copy2(source, dest_dir / name)
    archive = dest_dir / "archive.zip"
    archive.write_bytes(build_single_member_zip(payload, name=payload_member))
    return archive


def verify_archive_identity(
    archive_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Byte-identity control against a known frontier archive.

    Returns the comparison instead of raising, so a DIVERGENCE is reportable
    data ("which stage diverges and why") rather than an opaque abort.
    """

    path = Path(archive_path)
    got_sha = sha256_file(path)
    got_bytes = path.stat().st_size
    result: dict[str, Any] = {
        "archive_path": str(path),
        "sha256": got_sha,
        "bytes": got_bytes,
        "expected_sha256": expected_sha256,
        "expected_bytes": expected_bytes,
        "sha_matches": None if expected_sha256 is None else got_sha == expected_sha256,
        "bytes_match": None if expected_bytes is None else got_bytes == int(expected_bytes),
    }
    result["byte_identical"] = bool(result["sha_matches"]) and bool(result["bytes_match"])
    return result


@dataclass(frozen=True)
class InflateResult:
    returncode: int
    out_dir: str
    raw_files: int
    raw_bytes: int
    seconds: float
    stdout_tail: str
    stderr_tail: str


def run_inflate(
    submission_dir: str | Path,
    *,
    archive_dir: str | Path,
    out_dir: str | Path,
    video_names_file: str | Path,
    python_bin: str | None = None,
    timeout: int = 3600,
) -> InflateResult:
    """Run the SHIPPED ``inflate.sh`` in the FOREGROUND and refuse on failure.

    Three failure modes are refused, not just one:

    1. non-zero return code (the ordinary case);
    2. rc=0 with NO output produced -- the vacuity-equals-pass class (``m50``):
       an empty output directory is not a successful decode;
    3. rc=0 with zero total output bytes.

    ``ddm_si1`` measured that the shipped script returns rc correctly in the
    foreground; the historical "exit 0 while failing" came from a BACKGROUNDING
    launcher whose rc was the launcher's.  This function therefore never
    backgrounds -- that is the structural fix, not a flag.
    """

    import time

    sub = Path(submission_dir)
    script = sub / _INFLATE_ENTRY
    if not script.exists():
        raise SubmissionChainError(f"inflate entry script missing: {script}")
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # The shipped inflate.sh may invoke a bare `python`, which need not exist on
    # a python3-only host.  Prepending the running interpreter's bin directory
    # lets the SHIPPED script run byte-identically instead of editing the
    # counted runtime tree to suit this host.
    bin_dir = str(Path(python_bin or sys.executable).parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

    started = time.time()
    # GROUP-scoped timeout, not child-scoped. `subprocess.run(..., timeout=)` kills only
    # `bash`; the decoder underneath is a GRANDCHILD and survives. Measured here, on this
    # exact call (ddm_cpu1, 2026-08-20): the timeout fired at 1799.99997045 s, and the
    # decoder ran on to 4,369.600210089 s and wrote a complete 600-pair report for a run
    # this function had already raised on. Now the whole tree goes down with the wall.
    proc = run_in_process_group(
        ["bash", str(script), str(Path(archive_dir)), str(out), str(Path(video_names_file))],
        cwd=str(sub),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.time() - started
    produced = sorted(p for p in out.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in produced)

    if proc.returncode != 0:
        raise SubmissionChainError(
            f"inflate FAILED rc={proc.returncode}\n"
            f"stdout tail:\n{(proc.stdout or '')[-2000:]}\n"
            f"stderr tail:\n{(proc.stderr or '')[-2000:]}"
        )
    if not produced:
        raise SubmissionChainError(
            f"inflate returned rc=0 but produced NO files in {out}. "
            "An empty decode is not a pass (vacuity != pass)."
        )
    if total_bytes == 0:
        raise SubmissionChainError(
            f"inflate returned rc=0 and produced {len(produced)} files totalling "
            "0 bytes. An empty decode is not a pass (vacuity != pass)."
        )
    return InflateResult(
        returncode=proc.returncode,
        out_dir=str(out),
        raw_files=len(produced),
        raw_bytes=total_bytes,
        seconds=elapsed,
        stdout_tail=(proc.stdout or "")[-2000:],
        stderr_tail=(proc.stderr or "")[-2000:],
    )


def parse_evaluate_report(text: str) -> dict[str, Any]:
    """Parse ``upstream/evaluate.py``'s report block into typed components.

    A missing REQUIRED field raises: never fabricate a score component.
    ``n_samples`` is optional-by-absence (older report formats omit the line);
    absence is recorded as ``None`` and is NOT a partial-sample claim.
    """

    out: dict[str, Any] = {}
    for key, pat in _EVAL_REPORT_PATTERNS.items():
        mobj = re.search(pat, text)
        if mobj is None:
            if key == "n_samples":
                out[key] = None
                continue
            raise SubmissionChainError(
                f"evaluate.py report missing {key!r} (pattern {pat!r}); refusing "
                "to fabricate a score component (NO-FAKE).\n" + text[:2000]
            )
        out[key] = int(mobj.group(1)) if key == "n_samples" else float(mobj.group(1))
    return out


@dataclass(frozen=True)
class EvaluateResult:
    ran: bool
    device: str
    score_axis: str
    authority: str
    d_seg: float
    d_pose: float
    rate_from_evaluate: float
    evaluate_final_score: float
    recomputed_score: float
    recomputed_vs_evaluate_delta: float
    n_samples: int | None
    archive_bytes_scored: int
    report_path: str
    score_claim: bool = False
    promotion_eligible: bool = False


def run_upstream_evaluate(
    submission_dir: str | Path,
    *,
    upstream_dir: str | Path,
    videos_dir: str | Path,
    video_names_file: str | Path,
    archive_bytes: int,
    device: str = "cpu",
    batch_size: int = 16,
    num_threads: int | None = 2,
    require_n600: bool = True,
    timeout: int = 24 * 3600,
    python_bin: str | None = None,
    report_path: str | Path | None = None,
    preserve_existing_report: bool = True,
) -> EvaluateResult:
    """Run the REAL contest scorer on the exact staged bytes and recompute S.

    The returned score is ALWAYS recomputed from ``d_seg``/``d_pose``/bytes via
    ``tac.contest_score.compute_contest_score``; ``evaluate.py``'s own printed
    ``Final score`` is carried only as a cross-check, because the printed field
    is rounded and a rounded field has been quoted as truth before.
    """

    from tac.contest_score import compute_contest_score

    axis, authority = axis_and_authority(device)
    sub = Path(submission_dir)
    up = Path(upstream_dir)
    evaluate_py = up / "evaluate.py"
    if not evaluate_py.exists():
        raise SubmissionChainError(f"upstream/evaluate.py not found at {evaluate_py}")
    if not (sub / "archive.zip").exists():
        raise SubmissionChainError(f"missing {sub / 'archive.zip'} (NO-FAKE).")

    # A submission dir is usually a DURABLE evidence directory that already holds
    # the report of the run that produced the row.  Writing straight to
    # ``report.txt`` silently destroys that artifact -- measured live on
    # 2026-08-04, when this function was about to overwrite the ddm_pu2 report
    # that ddm_si1 had cited hours earlier.  Default: side-step the existing file
    # rather than clobber it.
    report_path = Path(report_path) if report_path else (sub / "report.txt")
    if preserve_existing_report and report_path.exists() and report_path == sub / "report.txt":
        from datetime import UTC, datetime

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report_path = sub / f"report_{stamp}.txt"
    cmd = [
        python_bin or sys.executable,
        str(evaluate_py),
        "--submission-dir", str(sub.resolve()),
        "--uncompressed-dir", str(Path(videos_dir).resolve()),
        "--video-names-file", str(Path(video_names_file).resolve()),
        "--device", device,
        "--report", str(report_path.resolve()),
        "--batch-size", str(int(batch_size)),
    ]
    if num_threads is not None:
        cmd += ["--num-threads", str(int(num_threads))]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(up.resolve()) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    # Sister of run_inflate above: `upstream/evaluate.py` spawns DataLoader workers, so a
    # child-scoped timeout orphans them mid-scoring. Group-scoped so the wall is real.
    proc = run_in_process_group(
        cmd, cwd=str(up), env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise SubmissionChainError(
            f"upstream/evaluate.py FAILED rc={proc.returncode}\n{combined[-4000:]}"
        )

    text = report_path.read_text() if report_path.exists() else combined
    parsed = parse_evaluate_report(text)
    n_samples = parsed.get("n_samples")
    if require_n600 and n_samples is not None and int(n_samples) != 600:
        raise SubmissionChainError(
            f"evaluate.py reports n_samples={int(n_samples)} != 600; refusing to "
            "record a partial-sample count as an exact row (n600 or not evidence)."
        )
    d_seg = float(parsed["d_seg"])
    d_pose = float(parsed["d_pose"])
    recomputed = compute_contest_score(d_seg, d_pose, archive_bytes)
    return EvaluateResult(
        ran=True,
        device=device,
        score_axis=axis,
        authority=authority,
        d_seg=d_seg,
        d_pose=d_pose,
        rate_from_evaluate=float(parsed["rate"]),
        evaluate_final_score=float(parsed["final_score"]),
        recomputed_score=recomputed,
        recomputed_vs_evaluate_delta=abs(recomputed - float(parsed["final_score"])),
        n_samples=n_samples,
        archive_bytes_scored=int(archive_bytes),
        report_path=str(report_path),
        score_claim=(authority == "authority"),
        promotion_eligible=False,
    )


# --------------------------------------------------------------------------- #
# 5. the typed receipt
# --------------------------------------------------------------------------- #
@dataclass
class ChainReceipt:
    """The single durable artifact a byte-close emits."""

    schema: str = "tac_submission_chain_receipt.v1"
    utc: str = ""
    git_hash: str = "unknown"
    seed: int | None = None
    host_platform: str = ""
    python_version: str = ""
    upstream_dir: str = ""
    upstream_tree_sha256: str | None = None
    archive_path: str = ""
    archive_sha256: str = ""
    archive_bytes: int = 0
    byte_ledger: dict[str, Any] = field(default_factory=dict)
    runtime_custody: dict[str, Any] = field(default_factory=dict)
    byte_identity: dict[str, Any] = field(default_factory=dict)
    inflate: dict[str, Any] | None = None
    evaluate: dict[str, Any] | None = None
    score_axis: str = ""
    score_claim: bool = False
    promotion_eligible: bool = False
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=False) + "\n"

    def write(self, path: str | Path) -> Path:
        out = refuse_transient_path(path, "receipt_path")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json())
        return out


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
