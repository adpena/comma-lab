#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Append-only resumable G29 compile/discovery/authority-preflight runner.

This runner intentionally stops before authority execution whenever any F0 is
open.  On non-contest hardware it produces a strict execution-readiness blocker
receipt; it never relabels macOS work as contest-CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

from tac.witness_dsl.taskspace_public_auth_eval_closure import (  # noqa: E402
    AuthClosureCheckpointArtifactV1,
    AuthClosureExecutionReadinessReceiptV1,
    AuthClosureStageCheckpointV1,
    AuthClosureStageV1,
    CompiledPublicRuntimeV1,
    GenericSourceAuditReceiptV1,
    InterpreterDistributionABIClosureV1,
    PayloadPlacementManifestV1,
    PublicAuthClosureError,
    PublicRuntimeCompileReceiptV1,
    RuntimeDependencyDiscoveryReceiptV1,
    assess_auth_eval_execution_readiness,
    compile_lvpg2_public_runtime,
    discover_public_runtime_dependencies,
    require_exact_public_runtime_tree,
)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise PublicAuthClosureError(f"refusing to overwrite drifted retained artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise PublicAuthClosureError(f"stale partial artifact blocks atomic write: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        temporary.unlink()
        raise PublicAuthClosureError(f"refusing to overwrite retained artifact: {path}") from exc
    temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(
        path,
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii"),
    )


def _artifact(run_dir: Path, path: Path, kind: str) -> AuthClosureCheckpointArtifactV1:
    return AuthClosureCheckpointArtifactV1(
        artifact_kind=kind,
        relative_path=path.relative_to(run_dir).as_posix(),
        content_sha256=_sha256_file(path),
        nbytes=path.stat().st_size,
    )


def _write_cleanup_certification(
    *,
    run_dir: Path,
    run_id: str,
    stage: AuthClosureStageV1,
    ordinal: int,
    source_paths: tuple[Path, ...],
    stage_artifacts: tuple[AuthClosureCheckpointArtifactV1, ...],
    invocation_argv: tuple[str, ...],
) -> Path:
    """Certify that this bounded stage created no bulky or deleted artifact."""

    runtime_files = tuple(sorted(path for path in (run_dir / "runtime").glob("*") if path.is_file()))
    destination = run_dir / "receipts" / f"{ordinal:03d}_cleanup_certification.json"
    _atomic_json(
        destination,
        {
            "cold_store_destination": None,
            "deletions_performed": [],
            "generated_runtime_files": [
                {
                    "bytes": path.stat().st_size,
                    "path": path.relative_to(run_dir).as_posix(),
                    "sha256": _sha256_file(path),
                }
                for path in runtime_files
            ],
            "invocation_argv": list(invocation_argv),
            "large_artifacts_created": False,
            "no_cleanup_required_reason": (
                "bounded compile/discovery/preflight created only retained source, runtime, and JSON receipts"
            ),
            "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "rebuildable_reason": (
                "runtime and receipts deterministically rebuild from exact source archive, renderer, and argv"
            ),
            "research_only": True,
            "run_id": run_id,
            "schema": "tac.taskspace_auth_closure_cleanup_certification.v1",
            "source_inputs": [
                {
                    "bytes": path.stat().st_size,
                    "path": path.as_posix(),
                    "sha256": _sha256_file(path),
                }
                for path in source_paths
            ],
            "stage": stage.value,
            "stage_artifacts": [item.to_dict() for item in stage_artifacts],
            "stage_ordinal": ordinal,
        },
    )
    return destination


def _checkpoint(
    *,
    run_dir: Path,
    run_id: str,
    stage: AuthClosureStageV1,
    ordinal: int,
    previous: AuthClosureStageCheckpointV1 | None,
    artifacts: tuple[AuthClosureCheckpointArtifactV1, ...],
    cleanup_certification_path: Path,
    blockers: tuple[str, ...] = (),
) -> AuthClosureStageCheckpointV1:
    cleanup_artifact = _artifact(run_dir, cleanup_certification_path, "cleanup.certification")
    retained_artifacts = (*artifacts, cleanup_artifact)
    checkpoint = AuthClosureStageCheckpointV1(
        run_id=run_id,
        stage=stage,
        stage_ordinal=ordinal,
        previous_checkpoint_sha256=None if previous is None else previous.identity_sha256,
        artifacts=tuple(sorted(retained_artifacts, key=lambda item: (item.artifact_kind, item.relative_path))),
        blockers=tuple(sorted(set(blockers))),
        completed=not blockers,
        research_only=True,
        cleanup_certification_sha256=cleanup_artifact.content_sha256,
    )
    checkpoint_path = run_dir / "checkpoints" / f"{ordinal:03d}_{stage.value.lower()}.json"
    if checkpoint_path.exists():
        raise PublicAuthClosureError(f"refusing to overwrite retained stage checkpoint: {checkpoint_path}")
    checkpoint.write_atomic(checkpoint_path)
    return checkpoint


def _verify_resume_chain(
    *,
    run_dir: Path,
    resume_path: Path,
    run_id: str,
) -> tuple[AuthClosureStageCheckpointV1, ...]:
    checkpoints: dict[str, tuple[Path, AuthClosureStageCheckpointV1]] = {}
    for path in sorted((run_dir / "checkpoints").glob("*.json")):
        checkpoint = AuthClosureStageCheckpointV1.from_receipt_bytes(path.read_bytes())
        if checkpoint.identity_sha256 in checkpoints:
            raise PublicAuthClosureError("duplicate checkpoint identity in resume directory")
        checkpoints[checkpoint.identity_sha256] = (path, checkpoint)
    retained = AuthClosureStageCheckpointV1.from_receipt_bytes(resume_path.read_bytes())
    if retained.run_id != run_id:
        raise PublicAuthClosureError("--resume-from belongs to another run_id")
    if retained.identity_sha256 not in checkpoints:
        raise PublicAuthClosureError("--resume-from is not an immutable checkpoint in run-dir")
    reversed_chain: list[AuthClosureStageCheckpointV1] = []
    visited: set[str] = set()
    current = retained
    while True:
        if current.identity_sha256 in visited:
            raise PublicAuthClosureError("resume checkpoint chain contains a cycle")
        visited.add(current.identity_sha256)
        if current.run_id != run_id:
            raise PublicAuthClosureError("resume chain crosses run IDs")
        for artifact in current.artifacts:
            path = run_dir / artifact.relative_path
            if (
                not path.is_file()
                or path.stat().st_size != artifact.nbytes
                or _sha256_file(path) != artifact.content_sha256
            ):
                raise PublicAuthClosureError(f"resume artifact failed retained byte identity: {artifact.relative_path}")
        reversed_chain.append(current)
        if current.previous_checkpoint_sha256 is None:
            break
        parent = checkpoints.get(current.previous_checkpoint_sha256)
        if parent is None:
            raise PublicAuthClosureError("resume chain parent checkpoint is missing")
        current = parent[1]
    chain = tuple(reversed(reversed_chain))
    ordinals = tuple(item.stage_ordinal for item in chain)
    if ordinals != tuple(range(len(chain))):
        raise PublicAuthClosureError("resume checkpoint ordinals are not a contiguous zero-based history")
    if set(checkpoints) != visited:
        raise PublicAuthClosureError("--resume-from must name the unique tip of the retained history")
    return chain


def _artifact_path(
    run_dir: Path,
    checkpoint: AuthClosureStageCheckpointV1,
    *,
    artifact_kind: str,
) -> Path:
    matches = [run_dir / item.relative_path for item in checkpoint.artifacts if item.artifact_kind == artifact_kind]
    if len(matches) != 1:
        raise PublicAuthClosureError(f"checkpoint must retain exactly one {artifact_kind!r} artifact")
    return matches[0]


def _load_compiled_from_checkpoint(
    *,
    run_dir: Path,
    checkpoint: AuthClosureStageCheckpointV1,
) -> CompiledPublicRuntimeV1:
    if checkpoint.stage is not AuthClosureStageV1.COMPILE_PUBLIC_RUNTIME or checkpoint.completed is not True:
        raise PublicAuthClosureError("resume compile checkpoint is not complete")
    compile_receipt = PublicRuntimeCompileReceiptV1.from_receipt_bytes(
        _artifact_path(run_dir, checkpoint, artifact_kind="compile.receipt").read_bytes()
    )
    abi = InterpreterDistributionABIClosureV1.from_receipt_bytes(
        _artifact_path(run_dir, checkpoint, artifact_kind="decoder.abi.receipt").read_bytes()
    )
    placement = PayloadPlacementManifestV1.from_receipt_bytes(
        _artifact_path(run_dir, checkpoint, artifact_kind="placement.receipt").read_bytes()
    )
    source_audit_paths = sorted(
        run_dir / item.relative_path
        for item in checkpoint.artifacts
        if item.artifact_kind == "generic.source.audit.receipt"
    )
    source_audits = tuple(
        GenericSourceAuditReceiptV1.from_receipt_bytes(path.read_bytes()) for path in source_audit_paths
    )
    if (
        tuple(sorted(item.identity_sha256 for item in source_audits)) != compile_receipt.source_audit_receipt_sha256s
        or placement.identity_sha256 != compile_receipt.placement_identity_sha256
        or abi.identity_sha256 != compile_receipt.abi_identity_sha256
    ):
        raise PublicAuthClosureError("retained compile parents drifted from compile receipt")
    runtime_dir = run_dir / "runtime"
    require_exact_public_runtime_tree(runtime_dir, complete=True)
    for item in compile_receipt.runtime_files:
        path = runtime_dir / item.relative_path
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.nbytes
            or _sha256_file(path) != item.content_sha256
        ):
            raise PublicAuthClosureError(f"retained public runtime drifted after compile: {item.relative_path}")
    return CompiledPublicRuntimeV1(
        compile_receipt=compile_receipt,
        placement=placement,
        abi_closure=abi,
        source_audits=source_audits,
    )


def _load_discovery_from_checkpoint(
    *,
    run_dir: Path,
    checkpoint: AuthClosureStageCheckpointV1,
) -> RuntimeDependencyDiscoveryReceiptV1:
    if checkpoint.stage is not AuthClosureStageV1.DEPENDENCY_DISCOVERY or checkpoint.completed is not True:
        raise PublicAuthClosureError("resume dependency-discovery checkpoint is not complete")
    return RuntimeDependencyDiscoveryReceiptV1.from_receipt_bytes(
        _artifact_path(
            run_dir,
            checkpoint,
            artifact_kind="dependency.discovery.receipt",
        ).read_bytes()
    )


def _select_retained_readiness(
    chain: tuple[AuthClosureStageCheckpointV1, ...],
) -> AuthClosureStageCheckpointV1 | None:
    attempts = tuple(item for item in chain if item.stage is AuthClosureStageV1.EXECUTION_PREFLIGHT)
    if not attempts:
        return None
    successful = tuple(item for item in attempts if item.completed)
    if len(successful) > 1:
        raise PublicAuthClosureError("resume history contains duplicate successful readiness stages")
    selected = successful[0] if successful else attempts[-1]
    if chain[-1] is not selected:
        raise PublicAuthClosureError("resume history continued after a successful readiness stage")
    return selected


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
            "ep725_population_global_recode_v2_20260726_r2/"
            "ep725_population_global_recode_v2.not_a_candidate.zip"
        ),
    )
    parser.add_argument(
        "--lvls1-runtime",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="g29.public.auth.closure")
    parser.add_argument(
        "--stop-after",
        choices=("compile", "discover", "readiness"),
        default="readiness",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--retry-blocked-stage",
        action="store_true",
        help="Append a fresh attempt after a retained blocked stage; completed stages are still never rerun.",
    )
    parser.add_argument(
        "--attest-generic-lineage",
        action="store_true",
        help="Required explicit attestation that emitted Python sources are generic, not video-derived.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    invocation_argv = tuple(sys.argv if argv is None else (Path(__file__).name, *argv))
    args = _parse_args(argv)
    if not args.attest_generic_lineage:
        raise SystemExit("--attest-generic-lineage is required; syntax cannot infer provenance")
    repo_root = args.repo_root.resolve(strict=True)
    archive = args.archive if args.archive.is_absolute() else repo_root / args.archive
    renderer = args.lvls1_runtime if args.lvls1_runtime.is_absolute() else repo_root / args.lvls1_runtime
    run_dir = args.run_dir.resolve(strict=False)
    runtime_dir = run_dir / "runtime"
    run_dir.mkdir(parents=True, exist_ok=True)
    chain: tuple[AuthClosureStageCheckpointV1, ...] = ()
    previous: AuthClosureStageCheckpointV1 | None = None
    next_ordinal = 0
    if args.resume_from is not None:
        chain = _verify_resume_chain(
            run_dir=run_dir,
            resume_path=args.resume_from.resolve(strict=True),
            run_id=args.run_id,
        )
        previous = chain[-1]
        next_ordinal = previous.stage_ordinal + 1
    elif any((run_dir / "checkpoints").glob("*.json")):
        raise SystemExit("existing checkpoints require --resume-from; refusing history overwrite")

    if chain:
        compile_checkpoint = chain[0]
        compiled = _load_compiled_from_checkpoint(
            run_dir=run_dir,
            checkpoint=compile_checkpoint,
        )
        compile_path = _artifact_path(
            run_dir,
            compile_checkpoint,
            artifact_kind="compile.receipt",
        )
        checkpoint = previous
        assert checkpoint is not None
    else:
        compiled = compile_lvpg2_public_runtime(
            archive_path=archive,
            lvls1_runtime_source_path=renderer,
            runtime_dir=runtime_dir,
            lineage_attested_generic=True,
        )
        compile_path = run_dir / "receipts" / "compile.json"
        decoder_abi_path = run_dir / "receipts" / "decoder_abi.json"
        placement_path = run_dir / "receipts" / "placement.json"
        compile_path.parent.mkdir(parents=True, exist_ok=True)
        for path, payload in (
            (compile_path, compiled.compile_receipt.to_receipt_bytes()),
            (decoder_abi_path, compiled.abi_closure.to_receipt_bytes()),
            (placement_path, compiled.placement.to_receipt_bytes()),
        ):
            _atomic_bytes(path, payload)
        source_audit_paths: list[Path] = []
        for index, audit in enumerate(sorted(compiled.source_audits, key=lambda item: item.source_name)):
            path = run_dir / "receipts" / f"source_audit_{index:03d}.json"
            _atomic_bytes(path, audit.to_receipt_bytes())
            source_audit_paths.append(path)
        compile_artifacts = tuple(
            sorted(
                (
                    _artifact(run_dir, compile_path, "compile.receipt"),
                    _artifact(run_dir, decoder_abi_path, "decoder.abi.receipt"),
                    _artifact(run_dir, placement_path, "placement.receipt"),
                    *(_artifact(run_dir, path, "generic.source.audit.receipt") for path in source_audit_paths),
                    *(
                        _artifact(
                            run_dir,
                            runtime_dir / item.relative_path,
                            "runtime.file",
                        )
                        for item in compiled.compile_receipt.runtime_files
                    ),
                ),
                key=lambda item: (item.artifact_kind, item.relative_path),
            )
        )
        compile_cleanup = _write_cleanup_certification(
            run_dir=run_dir,
            run_id=args.run_id,
            stage=AuthClosureStageV1.COMPILE_PUBLIC_RUNTIME,
            ordinal=next_ordinal,
            source_paths=(archive.resolve(strict=True), renderer.resolve(strict=True)),
            stage_artifacts=compile_artifacts,
            invocation_argv=invocation_argv,
        )
        checkpoint = _checkpoint(
            run_dir=run_dir,
            run_id=args.run_id,
            stage=AuthClosureStageV1.COMPILE_PUBLIC_RUNTIME,
            ordinal=next_ordinal,
            previous=None,
            artifacts=compile_artifacts,
            cleanup_certification_path=compile_cleanup,
        )
        next_ordinal += 1
    if args.stop_after == "compile":
        print(compile_path)
        return 0
    downstream_source_paths = (
        tuple(runtime_dir / item.relative_path for item in compiled.compile_receipt.runtime_files)
        if chain
        else (archive.resolve(strict=True), renderer.resolve(strict=True))
    )

    discovery: RuntimeDependencyDiscoveryReceiptV1 | None = None
    completed_discovery_checkpoints = tuple(
        item for item in chain if item.stage is AuthClosureStageV1.DEPENDENCY_DISCOVERY and item.completed
    )
    if completed_discovery_checkpoints:
        if len(completed_discovery_checkpoints) != 1:
            raise PublicAuthClosureError("resume history contains duplicate completed discovery stages")
        discovery = _load_discovery_from_checkpoint(
            run_dir=run_dir,
            checkpoint=completed_discovery_checkpoints[0],
        )
        if (
            discovery.compile_receipt_sha256 != compiled.compile_receipt.identity_sha256
            or discovery.decoder_abi_identity_sha256 != compiled.abi_closure.identity_sha256
        ):
            raise PublicAuthClosureError("retained discovery drifted from resumed compile")
    if args.stop_after == "discover" and discovery is not None:
        print(
            _artifact_path(
                run_dir,
                completed_discovery_checkpoints[0],
                artifact_kind="dependency.discovery.receipt",
            )
        )
        return 0

    selected_readiness = _select_retained_readiness(chain)
    if selected_readiness is not None:
        retained_readiness = AuthClosureExecutionReadinessReceiptV1.from_receipt_bytes(
            _artifact_path(
                run_dir,
                selected_readiness,
                artifact_kind="execution.readiness.receipt",
            ).read_bytes()
        )
        if (
            retained_readiness.compile_receipt_sha256 != compiled.compile_receipt.identity_sha256
            or retained_readiness.dependency_discovery_receipt_sha256
            != (None if discovery is None else discovery.identity_sha256)
        ):
            raise PublicAuthClosureError("retained readiness drifted from resumed parents")
        readiness_path = _artifact_path(
            run_dir,
            selected_readiness,
            artifact_kind="execution.readiness.receipt",
        )
        if retained_readiness.ready_to_execute or not args.retry_blocked_stage:
            print(readiness_path)
            return 0 if retained_readiness.ready_to_execute else 4

    discovery_blockers: tuple[str, ...] = ()
    if discovery is None:
        discovery_artifacts: list[AuthClosureCheckpointArtifactV1] = []
        discovery_ordinal = next_ordinal
        try:
            discovery = discover_public_runtime_dependencies(
                repo_root=repo_root,
                runtime_dir=runtime_dir,
                compiled=compiled,
            )
            discovery_path = run_dir / "receipts" / f"{discovery_ordinal:03d}_dependency_discovery.json"
            _atomic_bytes(discovery_path, discovery.to_receipt_bytes())
            discovery_artifacts.append(_artifact(run_dir, discovery_path, "dependency.discovery.receipt"))
        except (PublicAuthClosureError, ValueError, OSError) as exc:
            blocker_path = run_dir / "receipts" / f"{discovery_ordinal:03d}_dependency_discovery_blocker.json"
            _atomic_json(
                blocker_path,
                {
                    "blocker": "STRICT_RUNTIME_DEPENDENCY_DISCOVERY_REFUSED",
                    "detail": str(exc),
                    "research_only": True,
                    "schema": "tac.taskspace_dependency_discovery_blocker.v1",
                },
            )
            discovery_artifacts.append(_artifact(run_dir, blocker_path, "dependency.discovery.blocker"))
            discovery_blockers = ("STRICT_RUNTIME_DEPENDENCY_DISCOVERY_REFUSED",)
        discovery_artifact_tuple = tuple(discovery_artifacts)
        discovery_cleanup = _write_cleanup_certification(
            run_dir=run_dir,
            run_id=args.run_id,
            stage=AuthClosureStageV1.DEPENDENCY_DISCOVERY,
            ordinal=next_ordinal,
            source_paths=downstream_source_paths,
            stage_artifacts=discovery_artifact_tuple,
            invocation_argv=invocation_argv,
        )
        checkpoint = _checkpoint(
            run_dir=run_dir,
            run_id=args.run_id,
            stage=AuthClosureStageV1.DEPENDENCY_DISCOVERY,
            ordinal=next_ordinal,
            previous=checkpoint,
            artifacts=discovery_artifact_tuple,
            cleanup_certification_path=discovery_cleanup,
            blockers=discovery_blockers,
        )
        next_ordinal += 1
    if args.stop_after == "discover":
        print(run_dir / "checkpoints" / f"{checkpoint.stage_ordinal:03d}_dependency_discovery.json")
        return 0 if discovery is not None else 3

    readiness = assess_auth_eval_execution_readiness(
        repo_root=repo_root,
        runtime_dir=runtime_dir,
        compiled=compiled,
        discovery=discovery,
    )
    readiness_path = run_dir / "receipts" / f"{next_ordinal:03d}_execution_readiness.json"
    _atomic_bytes(readiness_path, readiness.to_receipt_bytes())
    readiness_artifacts = (_artifact(run_dir, readiness_path, "execution.readiness.receipt"),)
    readiness_cleanup = _write_cleanup_certification(
        run_dir=run_dir,
        run_id=args.run_id,
        stage=AuthClosureStageV1.EXECUTION_PREFLIGHT,
        ordinal=next_ordinal,
        source_paths=downstream_source_paths,
        stage_artifacts=readiness_artifacts,
        invocation_argv=invocation_argv,
    )
    readiness_checkpoint = _checkpoint(
        run_dir=run_dir,
        run_id=args.run_id,
        stage=AuthClosureStageV1.EXECUTION_PREFLIGHT,
        ordinal=next_ordinal,
        previous=checkpoint,
        artifacts=readiness_artifacts,
        cleanup_certification_path=readiness_cleanup,
        blockers=readiness.preflight_blockers,
    )
    _ = readiness_checkpoint
    print(readiness_path)
    return 0 if readiness.ready_to_execute else 4


if __name__ == "__main__":
    raise SystemExit(main())
