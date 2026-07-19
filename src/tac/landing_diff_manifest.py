"""Typed, deterministic custody receipt for a terminal ``BASE..HEAD`` landing.

The manifest is deliberately conservative: every path starts UNACCOUNTED and
only an explicit per-path declaration can change that state.  The receipt is
derived from Git objects, not from a mutable working-tree snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

SCHEMA = "pact.landing_diff_manifest.v1"
GENERATOR = "tac.landing_diff_manifest.v1"
_HEX40 = frozenset("0123456789abcdef")
_PLACEHOLDERS = frozenset({"", "none", "none:", "n/a", "na", "tbd", "todo", "<reason>", "<consumer>"})
_DECLARATION_KEYS = frozenset({"disposition", "status", "reason", "named_consumer", "consumer"})
_BLOCKER_KEYS = frozenset({"code", "path", "detail"})
_PATH_KEYS = frozenset(
    {
        "path",
        "git_status",
        "old_path",
        "base_sha256",
        "head_sha256",
        "gitignored_at_head",
        "findings_or_memo",
        "disposition",
        "reason",
        "named_consumer",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "generator",
        "base_sha",
        "head_sha",
        "tracked_diff_sha256",
        "path_count",
        "complete",
        "blockers",
        "paths",
    }
)


class LandingDiffManifestError(ValueError):
    """The requested Git range or serialized receipt is invalid."""


class PathDisposition(StrEnum):
    MERGED = "merged"
    INTENTIONALLY_DROPPED = "intentionally-dropped"
    DEFERRED = "deferred"
    UNACCOUNTED = "UNACCOUNTED"


@dataclass(frozen=True)
class DispositionDeclaration:
    disposition: PathDisposition
    reason: str | None = None
    named_consumer: str | None = None

    @classmethod
    def from_value(cls, value: object) -> DispositionDeclaration:
        if isinstance(value, str):
            raw: Mapping[str, object] = {"disposition": value}
        elif isinstance(value, Mapping):
            raw = value
        else:
            raise LandingDiffManifestError("disposition declaration must be a string or object")
        _reject_unknown_keys(raw, _DECLARATION_KEYS, "disposition declaration")
        if "disposition" in raw and "status" in raw:
            raise LandingDiffManifestError("declaration cannot contain both disposition and status")
        if "named_consumer" in raw and "consumer" in raw:
            raise LandingDiffManifestError(
                "declaration cannot contain both named_consumer and consumer"
            )
        status = raw.get("disposition", raw.get("status"))
        try:
            disposition = PathDisposition(str(status))
        except (TypeError, ValueError) as exc:
            raise LandingDiffManifestError(f"unknown path disposition: {status!r}") from exc
        reason = _optional_text(raw.get("reason"), "reason")
        consumer = _optional_text(raw.get("named_consumer", raw.get("consumer")), "named_consumer")
        return cls(disposition=disposition, reason=reason, named_consumer=consumer)

    def to_dict(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "reason": self.reason,
            "named_consumer": self.named_consumer,
        }


@dataclass(frozen=True)
class ManifestBlocker:
    code: str
    path: str | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "path": self.path, "detail": self.detail}

    @classmethod
    def from_dict(cls, raw: object) -> ManifestBlocker:
        if not isinstance(raw, Mapping):
            raise LandingDiffManifestError("manifest blocker must be an object")
        _reject_unknown_keys(raw, _BLOCKER_KEYS, "manifest blocker")
        code = _required_text(raw.get("code"), "blocker.code")
        detail = _required_text(raw.get("detail"), "blocker.detail")
        path_raw = raw.get("path")
        path = None if path_raw is None else _normalize_path(str(path_raw))
        return cls(code=code, path=path, detail=detail)


@dataclass(frozen=True)
class PathChange:
    path: str
    git_status: str
    old_path: str | None
    base_sha256: str | None
    head_sha256: str | None
    gitignored_at_head: bool
    findings_or_memo: bool
    disposition: PathDisposition
    reason: str | None
    named_consumer: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "git_status": self.git_status,
            "old_path": self.old_path,
            "base_sha256": self.base_sha256,
            "head_sha256": self.head_sha256,
            "gitignored_at_head": self.gitignored_at_head,
            "findings_or_memo": self.findings_or_memo,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "named_consumer": self.named_consumer,
        }

    @classmethod
    def from_dict(cls, raw: object) -> PathChange:
        if not isinstance(raw, Mapping):
            raise LandingDiffManifestError("path record must be an object")
        _reject_unknown_keys(raw, _PATH_KEYS, "path record")
        path = _normalize_path(_required_text(raw.get("path"), "path.path"))
        old_raw = raw.get("old_path")
        old_path = None if old_raw is None else _normalize_path(str(old_raw))
        git_status = _required_text(raw.get("git_status"), "path.git_status")
        try:
            disposition = PathDisposition(str(raw.get("disposition")))
        except ValueError as exc:
            raise LandingDiffManifestError(f"unknown path disposition for {path}: {raw.get('disposition')!r}") from exc
        return cls(
            path=path,
            git_status=git_status,
            old_path=old_path,
            base_sha256=_optional_sha(raw.get("base_sha256"), "base_sha256"),
            head_sha256=_optional_sha(raw.get("head_sha256"), "head_sha256"),
            gitignored_at_head=_required_bool(raw.get("gitignored_at_head"), "gitignored_at_head"),
            findings_or_memo=_required_bool(raw.get("findings_or_memo"), "findings_or_memo"),
            disposition=disposition,
            reason=_optional_text(raw.get("reason"), "reason"),
            named_consumer=_optional_text(raw.get("named_consumer"), "named_consumer"),
        )


@dataclass(frozen=True)
class LandingDiffManifest:
    base_sha: str
    head_sha: str
    tracked_diff_sha256: str
    paths: tuple[PathChange, ...]
    blockers: tuple[ManifestBlocker, ...]
    complete: bool
    schema: str = SCHEMA
    generator: str = GENERATOR

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "generator": self.generator,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "tracked_diff_sha256": self.tracked_diff_sha256,
            "path_count": len(self.paths),
            "complete": self.complete,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "paths": [path.to_dict() for path in self.paths],
        }

    def to_json_bytes(self) -> bytes:
        return (json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")

    @classmethod
    def from_dict(cls, raw: object) -> LandingDiffManifest:
        if not isinstance(raw, Mapping):
            raise LandingDiffManifestError("landing manifest must be a JSON object")
        _reject_unknown_keys(raw, _MANIFEST_KEYS, "landing manifest")
        if raw.get("schema") != SCHEMA:
            raise LandingDiffManifestError(f"unsupported manifest schema: {raw.get('schema')!r}")
        if raw.get("generator") != GENERATOR:
            raise LandingDiffManifestError(f"unsupported manifest generator: {raw.get('generator')!r}")
        paths_raw = raw.get("paths")
        blockers_raw = raw.get("blockers")
        if not isinstance(paths_raw, list) or not isinstance(blockers_raw, list):
            raise LandingDiffManifestError("manifest paths and blockers must be arrays")
        paths = tuple(PathChange.from_dict(value) for value in paths_raw)
        if tuple(sorted(paths, key=lambda value: value.path)) != paths:
            raise LandingDiffManifestError("manifest paths must be sorted by normalized path")
        names = [value.path for value in paths]
        if len(names) != len(set(names)):
            raise LandingDiffManifestError("manifest contains duplicate paths")
        path_count = raw.get("path_count")
        if type(path_count) is not int or path_count != len(paths):
            raise LandingDiffManifestError("manifest path_count does not match paths")
        blockers = tuple(ManifestBlocker.from_dict(value) for value in blockers_raw)
        expected_blockers = tuple(_semantic_blockers(paths))
        if blockers != expected_blockers:
            raise LandingDiffManifestError("manifest blocker list does not match path semantics")
        complete = _required_bool(raw.get("complete"), "complete")
        if complete != (not blockers):
            raise LandingDiffManifestError("manifest complete claim contradicts blockers")
        return cls(
            base_sha=_required_sha(raw.get("base_sha"), "base_sha"),
            head_sha=_required_sha(raw.get("head_sha"), "head_sha"),
            tracked_diff_sha256=_required_sha(raw.get("tracked_diff_sha256"), "tracked_diff_sha256", length=64),
            paths=paths,
            blockers=blockers,
            complete=complete,
        )


def _required_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise LandingDiffManifestError(f"{label} must be a boolean")
    return value


def _reject_unknown_keys(raw: Mapping[object, object], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise LandingDiffManifestError(f"{label} contains unknown fields: {unknown!r}")


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LandingDiffManifestError(f"{label} must be a string or null")
    return value.strip() or None


def _required_text(value: object, label: str) -> str:
    text = _optional_text(value, label)
    if text is None:
        raise LandingDiffManifestError(f"{label} must be non-empty")
    return text


def _required_sha(value: object, label: str, *, length: int = 40) -> str:
    text = _required_text(value, label)
    if text != text.lower():
        raise LandingDiffManifestError(f"{label} must use lowercase hex")
    if len(text) != length or any(char not in _HEX40 for char in text):
        raise LandingDiffManifestError(f"{label} must be a {length}-character lowercase hex digest")
    return text


def _optional_sha(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_sha(value, label, length=64)


def _real_text(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in _PLACEHOLDERS


def _normalize_path(raw: str) -> str:
    if not raw or "\x00" in raw:
        raise LandingDiffManifestError(f"invalid repository path: {raw!r}")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise LandingDiffManifestError(f"path escapes or is not normalized: {raw!r}")
    normalized = pure.as_posix()
    if normalized != raw:
        raise LandingDiffManifestError(f"path is not normalized: {raw!r}")
    return normalized


def _run_git(repo: Path, args: Sequence[str], *, input_bytes: bytes | None = None) -> bytes:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            input=input_bytes,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise LandingDiffManifestError(f"could not execute git: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise LandingDiffManifestError(f"git {' '.join(args[:3])} failed with rc={proc.returncode}: {detail}")
    return proc.stdout


def _resolve_commit(repo: Path, revision: str) -> str:
    output = _run_git(repo, ["rev-parse", "--verify", f"{revision}^{{commit}}"])
    return _required_sha(output.decode("ascii").strip(), "resolved revision")


def _object_bytes(repo: Path, commit: str, path: str) -> bytes:
    """Stable bytes for a path, including gitlinks as well as ordinary blobs."""
    object_id = _run_git(repo, ["rev-parse", "--verify", f"{commit}:{path}"]).decode("ascii").strip()
    object_type = _run_git(repo, ["cat-file", "-t", object_id]).decode("ascii").strip()
    if object_type == "blob":
        return _run_git(repo, ["cat-file", "blob", object_id])
    return f"git-object:{object_type}:{object_id}\n".encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_name_status(raw: bytes) -> list[tuple[str, str | None, str]]:
    if not raw:
        return []
    tokens = raw.split(b"\0")
    if tokens[-1] == b"":
        tokens.pop()
    out: list[tuple[str, str | None, str]] = []
    index = 0
    while index < len(tokens):
        status = tokens[index].decode("ascii", errors="strict")
        index += 1
        if not status:
            raise LandingDiffManifestError("empty Git status token")
        if status[0] in {"R", "C"}:
            if index + 1 >= len(tokens):
                raise LandingDiffManifestError("truncated rename/copy record")
            old_path = _normalize_path(tokens[index].decode("utf-8", errors="surrogateescape"))
            path = _normalize_path(tokens[index + 1].decode("utf-8", errors="surrogateescape"))
            index += 2
        else:
            if index >= len(tokens):
                raise LandingDiffManifestError("truncated Git path record")
            old_path = None
            path = _normalize_path(tokens[index].decode("utf-8", errors="surrogateescape"))
            index += 1
        out.append((status, old_path, path))
    return out


def _is_findings_or_memo(path: str) -> bool:
    pure = PurePosixPath(path)
    in_research = len(pure.parts) >= 3 and pure.parts[:2] == (".omx", "research")
    stem_tokens = set(filter(None, re.split(r"[^a-z0-9]+", pure.stem.lower())))
    return (in_research and pure.suffix.lower() == ".md") or bool(
        {"finding", "findings", "memo"} & stem_tokens
    )


def _ignored_paths_at_commit(repo: Path, commit: str, paths: Sequence[str]) -> set[str]:
    """Evaluate ignore rules from ``commit``, independent of the live checkout.

    ``git check-ignore`` has no tree argument.  Materialize only the committed
    ``.gitignore`` files into a tiny temporary repository, disable the user's
    global excludes file, and ask Git itself to evaluate the paths.  The
    temporary directory is context-managed and therefore cannot become
    orphaned bulk.
    """
    if not paths:
        return set()
    tree_paths_raw = _run_git(repo, ["ls-tree", "-r", "-z", "--name-only", commit, "--"])
    tree_paths = [
        _normalize_path(token.decode("utf-8", errors="surrogateescape"))
        for token in tree_paths_raw.split(b"\0")
        if token
    ]
    ignore_paths = [path for path in tree_paths if PurePosixPath(path).name == ".gitignore"]
    with tempfile.TemporaryDirectory(prefix="pact-landing-ignore-") as temporary:
        scratch = Path(temporary)
        _run_git(scratch, ["init", "-q"])
        for ignore_path in ignore_paths:
            destination = scratch / ignore_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_object_bytes(repo, commit, ignore_path))
        for path in paths:
            (scratch / path).parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(scratch),
                "-c",
                "core.excludesFile=/dev/null",
                "check-ignore",
                "--no-index",
                "-z",
                "--stdin",
            ],
            input=b"\0".join(path.encode("utf-8", errors="surrogateescape") for path in paths) + b"\0",
            capture_output=True,
            check=False,
        )
        if proc.returncode not in {0, 1}:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()
            raise LandingDiffManifestError(f"git check-ignore failed: {detail}")
        return {
            _normalize_path(token.decode("utf-8", errors="surrogateescape"))
            for token in proc.stdout.split(b"\0")
            if token
        }


def _semantic_blockers(paths: Sequence[PathChange]) -> list[ManifestBlocker]:
    blockers: list[ManifestBlocker] = []
    for record in paths:
        if record.disposition is PathDisposition.UNACCOUNTED:
            blockers.append(ManifestBlocker("unaccounted_path", record.path, "changed path lacks a disposition"))
        if record.disposition is PathDisposition.INTENTIONALLY_DROPPED and not _real_text(record.reason):
            blockers.append(
                ManifestBlocker(
                    "dropped_reason_missing",
                    record.path,
                    "intentionally-dropped requires a real reason",
                )
            )
        if record.disposition is PathDisposition.DEFERRED and not _real_text(record.named_consumer):
            blockers.append(
                ManifestBlocker(
                    "deferred_consumer_missing",
                    record.path,
                    "deferred requires a named consumer",
                )
            )
        if (
            record.findings_or_memo
            and record.disposition is not PathDisposition.UNACCOUNTED
            and not _real_text(record.named_consumer)
        ):
            blockers.append(
                ManifestBlocker(
                    "findings_consumer_missing",
                    record.path,
                    "findings/memo artifact requires a named consumer or explicit deferral",
                )
            )
        if record.gitignored_at_head:
            blockers.append(
                ManifestBlocker(
                    "gitignored_changed_path",
                    record.path,
                    "changed path is ignored by the HEAD commit's Git rules",
                )
            )
    return blockers


def parse_declarations(raw: object) -> dict[str, DispositionDeclaration]:
    if isinstance(raw, Mapping) and "dispositions" in raw:
        raw = raw["dispositions"]
    if not isinstance(raw, Mapping):
        raise LandingDiffManifestError("declarations JSON must be an object keyed by path")
    out: dict[str, DispositionDeclaration] = {}
    for path_raw, value in raw.items():
        if not isinstance(path_raw, str):
            raise LandingDiffManifestError("declaration path keys must be strings")
        path = _normalize_path(path_raw)
        if path in out:
            raise LandingDiffManifestError(f"duplicate declaration path: {path}")
        out[path] = DispositionDeclaration.from_value(value)
    return out


def build_manifest(
    repo: str | os.PathLike[str],
    base: str,
    head: str,
    declarations: Mapping[str, DispositionDeclaration | Mapping[str, object] | str] | None = None,
    *,
    global_consumer: str | None = None,
) -> LandingDiffManifest:
    repo_path = Path(repo).resolve()
    if not repo_path.is_dir():
        raise LandingDiffManifestError(f"repository path is not a directory: {repo_path}")
    base_sha = _resolve_commit(repo_path, base)
    head_sha = _resolve_commit(repo_path, head)
    raw_status = _run_git(repo_path, ["diff", "--name-status", "-z", "-M", "-C", base_sha, head_sha, "--"])
    changes = _parse_name_status(raw_status)
    changed_names = {path for _, _, path in changes}
    parsed: dict[str, DispositionDeclaration] = {}
    for path_raw, value in (declarations or {}).items():
        path = _normalize_path(str(path_raw))
        if path in parsed:
            raise LandingDiffManifestError(f"duplicate declaration path: {path}")
        parsed[path] = value if isinstance(value, DispositionDeclaration) else DispositionDeclaration.from_value(value)
    extras = sorted(set(parsed) - changed_names)
    if extras:
        raise LandingDiffManifestError(f"declarations reference paths outside BASE..HEAD: {extras!r}")
    consumer = _optional_text(global_consumer, "global_consumer")
    head_paths = [path for status, _, path in changes if status[0] != "D"]
    ignored_at_head = _ignored_paths_at_commit(repo_path, head_sha, head_paths)
    records: list[PathChange] = []
    for status, old_path, path in changes:
        kind = status[0]
        base_path = old_path if kind in {"R", "C"} else path
        base_bytes = None if kind == "A" else _object_bytes(repo_path, base_sha, base_path or path)
        head_bytes = None if kind == "D" else _object_bytes(repo_path, head_sha, path)
        declaration = parsed.get(path, DispositionDeclaration(PathDisposition.UNACCOUNTED))
        is_memo = _is_findings_or_memo(path)
        named_consumer = declaration.named_consumer
        if is_memo and named_consumer is None and consumer is not None:
            named_consumer = consumer
        records.append(
            PathChange(
                path=path,
                git_status=status,
                old_path=old_path,
                base_sha256=None if base_bytes is None else _sha256(base_bytes),
                head_sha256=None if head_bytes is None else _sha256(head_bytes),
                gitignored_at_head=path in ignored_at_head,
                findings_or_memo=is_memo,
                disposition=declaration.disposition,
                reason=declaration.reason,
                named_consumer=named_consumer,
            )
        )
    ordered = tuple(sorted(records, key=lambda value: value.path))
    diff_bytes = _run_git(
        repo_path,
        [
            "-c",
            "color.ui=false",
            "-c",
            "diff.algorithm=myers",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            base_sha,
            head_sha,
            "--",
        ],
    )
    blockers = tuple(_semantic_blockers(ordered))
    return LandingDiffManifest(
        base_sha=base_sha,
        head_sha=head_sha,
        tracked_diff_sha256=_sha256(diff_bytes),
        paths=ordered,
        blockers=blockers,
        complete=not blockers,
    )


def _strict_json_load(data: bytes, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in pairs:
            if key in out:
                raise LandingDiffManifestError(f"{label} contains duplicate JSON key {key!r}")
            out[key] = value
        return out

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LandingDiffManifestError(f"could not parse {label}: {exc}") from exc


def load_manifest_bytes(data: bytes, *, label: str = "landing manifest") -> LandingDiffManifest:
    return LandingDiffManifest.from_dict(_strict_json_load(data, label))


def load_manifest(path: str | os.PathLike[str]) -> LandingDiffManifest:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise LandingDiffManifestError(f"could not read landing manifest {path}: {exc}") from exc
    return load_manifest_bytes(data, label=f"landing manifest {path}")


def write_manifest(path: str | os.PathLike[str], manifest: LandingDiffManifest) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(manifest.to_json_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def verify_manifest(repo: str | os.PathLike[str], manifest: LandingDiffManifest) -> tuple[ManifestBlocker, ...]:
    declarations = {
        record.path: DispositionDeclaration(
            record.disposition, reason=record.reason, named_consumer=record.named_consumer
        )
        for record in manifest.paths
    }
    rebuilt = build_manifest(repo, manifest.base_sha, manifest.head_sha, declarations)
    if rebuilt.to_dict() == manifest.to_dict():
        return ()
    return (
        ManifestBlocker(
            "receipt_git_mismatch",
            None,
            "receipt does not match the current Git objects for its BASE..HEAD range",
        ),
    )


def _load_declarations(path: str | None) -> dict[str, DispositionDeclaration]:
    if path is None:
        return {}
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise LandingDiffManifestError(f"could not read declarations {path}: {exc}") from exc
    return parse_declarations(_strict_json_load(data, f"declarations {path}"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--declarations-json")
    parser.add_argument("--global-consumer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(
            args.repo,
            args.base,
            args.head,
            _load_declarations(args.declarations_json),
            global_consumer=args.global_consumer,
        )
        write_manifest(args.output, manifest)
    except LandingDiffManifestError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        f"landing_diff_manifest paths={len(manifest.paths)} "
        f"blockers={len(manifest.blockers)} complete={str(manifest.complete).lower()} "
        f"output={args.output}"
    )
    return 0 if manifest.complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
