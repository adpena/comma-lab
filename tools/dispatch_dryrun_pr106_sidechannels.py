#!/usr/bin/env python3
"""Read-only dispatch dry-run for PR106 sidechannel builder readiness.

This is a $0 local guard. It does not dispatch, does not touch provider state,
does not require CUDA, and does not build archives. It validates only the
surfaces that can fail before money or GPU time is spent:

  1. The PR106 sidechannel builder/profiler scripts exist and parse as Python.
  2. Their expected argparse flags are present.
  3. Their `--help` surfaces execute successfully.
  4. Their build metadata paths include `score_claim: false` in source.
  5. yshift/lrl1 real modes are still fail-closed behind NotImplementedError.
  6. If `--production-readiness` is selected, the caller must provide existing
     manifest and sister archive paths; those manifests must report
     `score_claim=false`.

Default dry-run mode intentionally does not require PR106 anchor artifacts,
build manifests, or sister archives. Those are production-custody inputs, so
missing paths fail only when `--production-readiness` is explicit.

Usage:
  .venv/bin/python tools/dispatch_dryrun_pr106_sidechannels.py
  .venv/bin/python tools/dispatch_dryrun_pr106_sidechannels.py --json
  .venv/bin/python tools/dispatch_dryrun_pr106_sidechannels.py \
      --production-readiness \
      --pr106-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
      --latent-sister-archive experiments/results/lane_pr106_latent_sidecar_cpu_smoke_20260505/sidecar_archive.zip \
      --yshift-sister-archive experiments/results/lane_pr106_yshift_sidechannel_smoke_20260505T065831Z_advisory/pr106_yshift_sidechannel_archive.zip \
      --lrl1-sister-archive experiments/results/lane_pr106_lrl1_sidechannel_smoke_20260505T072638Z_advisory/pr106_lrl1_sidechannel_archive.zip \
      --wavelet-sister-archive experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/hnerv_wavelet_sidechannel_candidate.zip \
      --latent-manifest experiments/results/lane_pr106_latent_sidecar_cpu_smoke_20260505/build_metadata.json \
      --yshift-manifest experiments/results/lane_pr106_yshift_sidechannel_smoke_20260505T065831Z_advisory/build_metadata.json \
      --lrl1-manifest experiments/results/lane_pr106_lrl1_sidechannel_smoke_20260505T072638Z_advisory/build_metadata.json \
      --wavelet-manifest experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/manifest.json \
      --stacked-manifest experiments/results/lane_pr106_stacked_3sister_cpu_smoke_20260505T140325Z/build_metadata.json
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.audit_contract import AuditReport
from tac.repo_io import json_text, read_json

REPO = Path(__file__).resolve().parents[1]
REAL_SIDECHANNEL_MODES = {"gradient", "brute_force"}
LATENT_OUTER_MAGIC = 0xFE
YSHIFT_OUTER_MAGIC = 0xFC
LRL1_OUTER_MAGIC = 0xFB
WAVELET_OUTER_MAGIC = 0xFA
WAVELET_OUTER_VERSION = 1


@dataclass(frozen=True)
class BuilderSpec:
    name: str
    relpath: str
    expected_flags: tuple[str, ...]
    help_tokens: tuple[str, ...]
    real_mode_guard_functions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductionInputs:
    pr106_archive: Path | None = None
    latent_sister_archive: Path | None = None
    yshift_sister_archive: Path | None = None
    lrl1_sister_archive: Path | None = None
    wavelet_sister_archive: Path | None = None
    latent_manifest: Path | None = None
    yshift_manifest: Path | None = None
    lrl1_manifest: Path | None = None
    wavelet_manifest: Path | None = None
    stacked_manifest: Path | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class DryRunReport:
    ok: bool
    checks: tuple[CheckResult, ...]
    warnings: tuple[str, ...]
    score_claim: bool
    dispatch_attempted: bool
    gpu_required: bool
    provider_state_free: bool

    def audit_report(self, repo: Path) -> AuditReport:
        failed_checks = tuple(c.name for c in self.checks if not c.ok)
        return AuditReport(
            audit="pr106_sidechannel_dispatch_dryrun",
            readiness_key="ready_for_local_readiness",
            ready=self.ok,
            blockers=failed_checks,
            summary={
                "check_count": len(self.checks),
                "failed_check_count": len(failed_checks),
                "warning_count": len(self.warnings),
            },
            metadata={
                "checks": [
                    {"name": c.name, "ok": c.ok, "detail": c.detail}
                    for c in self.checks
                ],
                "gpu_required": self.gpu_required,
                "ok": self.ok,
                "provider_state_free": self.provider_state_free,
                "repo": str(repo),
                "schema": "pr106_sidechannel_dispatch_dryrun_v1",
                "warnings": list(self.warnings),
            },
        )

    def to_jsonable(self, repo: Path) -> dict[str, Any]:
        return self.audit_report(repo).to_dict()


BUILDER_SPECS: tuple[BuilderSpec, ...] = (
    BuilderSpec(
        name="latent",
        relpath="experiments/build_pr106_latent_sidecar.py",
        expected_flags=(
            "--source-archive",
            "--output-dir",
            "--device",
            "--smoke",
            "--top-k",
            "--search-mode",
        ),
        help_tokens=(
            "Build PR106 + per-pair latent-correction sidecar archive",
            "--source-archive",
            "--output-dir",
            "--search-mode",
            "Stage 3",
        ),
    ),
    BuilderSpec(
        name="yshift",
        relpath="experiments/build_pr106_yshift_sidechannel.py",
        expected_flags=(
            "--pr106-archive",
            "--out-dir",
            "--search-mode",
            "--score-table-npy",
            "--score-table-manifest",
            "--candidate-radius",
            "--score-step",
            "--n-pairs",
        ),
        help_tokens=(
            "Build a pr106_yshift_sidechannel archive",
            "--pr106-archive",
            "--out-dir",
            "--search-mode",
            "score_table",
            "gradient/brute_force",
        ),
        real_mode_guard_functions=(
            "_gradient_search_stub",
            "_brute_force_search_stub",
        ),
    ),
    BuilderSpec(
        name="yshift_score_table",
        relpath="experiments/build_pr106_yshift_score_table.py",
        expected_flags=(
            "--pr106-archive",
            "--out-dir",
            "--candidate-radius",
            "--score-step",
            "--n-pairs",
            "--device",
            "--batch-pairs",
            "--candidate-batch-size",
            "--claims-path",
            "--lane-id",
            "--instance-job-id",
            "--dry-run-plan",
        ),
        help_tokens=(
            "Build a scorer-backed PR106 yshift candidate table",
            "--dry-run-plan",
            "--instance-job-id",
            "score_table",
        ),
    ),
    BuilderSpec(
        name="lrl1",
        relpath="experiments/build_pr106_lrl1_sidechannel.py",
        expected_flags=(
            "--pr106-archive",
            "--out-dir",
            "--search-mode",
            "--K",
            "--low-h",
            "--low-w",
            "--basis-step",
            "--coeff-step",
            "--n-pairs",
        ),
        help_tokens=(
            "Build a pr106_lrl1_sidechannel archive",
            "--pr106-archive",
            "--out-dir",
            "--search-mode",
            "gradient/brute_force",
        ),
        real_mode_guard_functions=(
            "_gradient_search_stub",
            "_brute_force_search_stub",
        ),
    ),
    BuilderSpec(
        name="stacked",
        relpath="experiments/build_pr106_stacked.py",
        expected_flags=(
            "--pr106-archive",
            "--latent",
            "--yshift",
            "--lrl1",
            "--wavelet",
            "--output-dir",
        ),
        help_tokens=(
            "Compose a pr106_stacked archive",
            "--pr106-archive",
            "--latent",
            "--yshift",
            "--lrl1",
            "--wavelet",
        ),
    ),
)


def _rel(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _check(name: str, checks: list[CheckResult], condition: bool, detail: str) -> None:
    checks.append(CheckResult(name=name, ok=condition, detail=detail))


def _load_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _argparse_flags(tree: ast.AST) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            value = _literal_string(arg)
            if value and value.startswith("--"):
                flags.add(value)
    return flags


def _has_score_claim_value(tree: ast.AST, expected: bool) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if _literal_string(key) != "score_claim":
                continue
            if isinstance(value, ast.Constant) and value.value is expected:
                return True
    return False


def _functions_that_raise_notimplemented(tree: ast.AST) -> set[str]:
    guarded: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Raise):
                continue
            exc = child.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                if exc.func.id == "NotImplementedError":
                    guarded.add(node.name)
            elif isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                guarded.add(node.name)
    return guarded


def _run_help(path: Path, repo: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTHONHASHSEED", "0")
    return subprocess.run(  # subprocess-no-check-OK: dryrun probe inspects returncode at call sites; --help non-zero is informative not fatal
        [sys.executable, str(path), "--help"],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _check_builder_surface(
    repo: Path,
    spec: BuilderSpec,
    checks: list[CheckResult],
    *,
    run_help: bool,
) -> ast.Module | None:
    path = repo / spec.relpath
    if not path.is_file():
        _check(
            f"{spec.name}:file",
            checks,
            False,
            f"missing required builder {spec.relpath}",
        )
        return None
    _check(
        f"{spec.name}:file",
        checks,
        True,
        f"{spec.relpath} present",
    )

    try:
        tree = _load_ast(path)
    except SyntaxError as exc:
        _check(
            f"{spec.name}:ast",
            checks,
            False,
            f"syntax error in {spec.relpath}: {exc}",
        )
        return None
    _check(f"{spec.name}:ast", checks, True, f"{spec.relpath} parses")

    flags = _argparse_flags(tree)
    missing_flags = sorted(set(spec.expected_flags) - flags)
    _check(
        f"{spec.name}:argparse-flags",
        checks,
        not missing_flags,
        (
            f"all expected flags present: {', '.join(spec.expected_flags)}"
            if not missing_flags
            else f"missing argparse flags: {missing_flags}"
        ),
    )

    score_claim_false = _has_score_claim_value(tree, False)
    score_claim_true = _has_score_claim_value(tree, True)
    _check(
        f"{spec.name}:score-claim-marker",
        checks,
        score_claim_false and not score_claim_true,
        "builder metadata includes score_claim=false"
        if score_claim_false
        else "builder metadata must explicitly include score_claim=false",
    )

    if spec.real_mode_guard_functions:
        guarded = _functions_that_raise_notimplemented(tree)
        missing_guards = sorted(set(spec.real_mode_guard_functions) - guarded)
        _check(
            f"{spec.name}:real-mode-notimplemented-guards",
            checks,
            not missing_guards,
            (
                "real search stubs raise NotImplementedError"
                if not missing_guards
                else f"real search stubs do not fail closed: {missing_guards}"
            ),
        )

    if run_help:
        try:
            proc = _run_help(path, repo)
        except subprocess.TimeoutExpired:
            _check(f"{spec.name}:help", checks, False, "--help timed out")
        else:
            help_output = proc.stdout + proc.stderr
            missing_tokens = [token for token in spec.help_tokens if token not in help_output]
            ok = proc.returncode == 0 and not missing_tokens
            detail = (
                "--help returned 0 and contains expected readiness tokens"
                if ok
                else (
                    f"--help failed rc={proc.returncode}; missing tokens={missing_tokens}; "
                    f"stderr={proc.stderr.strip()[:300]}"
                )
            )
            _check(f"{spec.name}:help", checks, ok, detail)

    return tree


def _check_selected_modes(
    checks: list[CheckResult],
    *,
    latent_search_mode: str,
    yshift_search_mode: str,
    lrl1_search_mode: str,
) -> None:
    _check(
        "latent:selected-mode",
        checks,
        latent_search_mode == "heuristic",
        (
            "latent mode heuristic is CPU-safe dry-run scaffolding"
            if latent_search_mode == "heuristic"
            else f"latent mode {latent_search_mode!r} is not dry-run safe"
        ),
    )
    for lane, mode in (
        ("yshift", yshift_search_mode),
        ("lrl1", lrl1_search_mode),
    ):
        if mode in REAL_SIDECHANNEL_MODES:
            _check(
                f"{lane}:selected-mode",
                checks,
                False,
                (
                    f"selected real mode {mode!r} is NotImplemented in the local builder; "
                    "do not dispatch from this dry-run"
                ),
            )
        else:
            _check(
                f"{lane}:selected-mode",
                checks,
                mode == "zero",
                f"{lane} mode zero is CPU-safe wire-format dry-run"
                if mode == "zero"
                else f"{lane} mode {mode!r} is unknown",
            )


def _json_manifest_contract(
    path: Path,
    *,
    archive_path: Path | None,
    repo: Path,
    require_dispatch_blockers: bool = False,
) -> tuple[bool, str]:
    try:
        data = read_json(path)
    except ValueError as exc:
        return False, f"invalid JSON: {exc}"
    if not isinstance(data, dict):
        return False, "manifest is not a JSON object"
    if data.get("score_claim") is not False:
        return False, f"score_claim is {data.get('score_claim')!r}, expected false"
    if data.get("dispatch_attempted") is True:
        return False, "dispatch_attempted=true belongs to post-dispatch evidence, not readiness"
    if data.get("remote_jobs_dispatched") is True:
        return False, "remote_jobs_dispatched=true belongs to post-dispatch evidence, not readiness"
    if require_dispatch_blockers:
        if data.get("ready_for_exact_eval_dispatch") is not False:
            return False, "ready_for_exact_eval_dispatch must be explicitly false before exact CUDA eval"
        blockers = data.get("dispatch_blockers")
        if not isinstance(blockers, list) or not blockers:
            return False, "dispatch_blockers must be a non-empty list"
    if archive_path is not None:
        archive_field = (
            data.get("archive_path")
            or data.get("candidate_archive_path")
        )
        if not isinstance(archive_field, str) or not archive_field:
            return False, "manifest must record archive_path or candidate_archive_path"
        recorded_path = Path(archive_field)
        if not recorded_path.is_absolute():
            recorded_path = repo / recorded_path
        if recorded_path.resolve(strict=False) != archive_path.resolve(strict=False):
            return (
                False,
                "manifest archive path does not match supplied archive: "
                f"metadata={_rel(recorded_path, repo)} supplied={_rel(archive_path, repo)}",
            )
        recorded_bytes = (
            data.get("archive_zip_bytes")
            or data.get("archive_size_bytes")
            or data.get("candidate_archive_bytes")
        )
        if not isinstance(recorded_bytes, int) or isinstance(recorded_bytes, bool):
            return False, "manifest must record integer archive byte count"
        actual_bytes = archive_path.stat().st_size
        if recorded_bytes != actual_bytes:
            return (
                False,
                "manifest archive byte count does not match supplied archive: "
                f"metadata={recorded_bytes} supplied={actual_bytes}",
            )
    return True, "manifest matches score/dispatch/archive custody contract"


def _check_path_required(
    checks: list[CheckResult],
    *,
    name: str,
    path: Path | None,
    repo: Path,
    kind: str,
) -> None:
    if path is None:
        _check(name, checks, False, f"production readiness requires --{kind}")
        return
    _check(
        name,
        checks,
        path.is_file(),
        f"{_rel(path, repo)} present" if path.is_file() else f"missing {_rel(path, repo)}",
    )


def _check_zip_has_zero_bin(
    checks: list[CheckResult],
    *,
    name: str,
    path: Path | None,
    repo: Path,
) -> None:
    if path is None or not path.is_file():
        return
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except zipfile.BadZipFile as exc:
        _check(name, checks, False, f"{_rel(path, repo)} is not a valid ZIP: {exc}")
        return
    _check(
        name,
        checks,
        "0.bin" in names,
        f"{_rel(path, repo)} contains 0.bin"
        if "0.bin" in names
        else f"{_rel(path, repo)} missing 0.bin (members={names})",
    )


def _read_zero_bin(path: Path) -> bytes:
    with zipfile.ZipFile(path) as z:
        return z.read("0.bin")


def _read_single_member_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if len(infos) != 1 or infos[0].is_dir():
            raise ValueError("expected exactly one non-directory ZIP member")
        return z.read(infos[0].filename)


def _check_embedded_pr106_payload(
    checks: list[CheckResult],
    *,
    name: str,
    archive: Path | None,
    pr106_payload: bytes | None,
    repo: Path,
    kind: str,
) -> None:
    if archive is None or not archive.is_file() or pr106_payload is None:
        return
    try:
        payload = _read_single_member_payload(archive) if kind == "wavelet" else _read_zero_bin(archive)
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        _check(name, checks, False, f"{_rel(archive, repo)} cannot be parsed: {exc}")
        return
    try:
        if kind == "latent":
            ok = len(payload) >= 6 and payload[0] == LATENT_OUTER_MAGIC
            embedded_len = int.from_bytes(payload[2:6], "little") if ok else -1
            embedded = payload[6:6 + embedded_len] if ok else b""
        elif kind in {"yshift", "lrl1"}:
            expected_magic = YSHIFT_OUTER_MAGIC if kind == "yshift" else LRL1_OUTER_MAGIC
            ok = len(payload) >= 4 and payload[0] == expected_magic
            embedded_len = int.from_bytes(payload[1:4], "little") if ok else -1
            embedded = payload[4:4 + embedded_len] if ok else b""
        elif kind == "wavelet":
            ok = (
                len(payload) >= 5
                and payload[0] == WAVELET_OUTER_MAGIC
                and payload[1] == WAVELET_OUTER_VERSION
            )
            embedded_len = int.from_bytes(payload[2:5], "little") if ok else -1
            embedded = payload[5:5 + embedded_len] if ok else b""
        else:  # pragma: no cover - internal callsite bug.
            raise AssertionError(f"unsupported sidechannel kind: {kind}")
    except IndexError as exc:
        _check(name, checks, False, f"{_rel(archive, repo)} truncated: {exc}")
        return
    _check(
        name,
        checks,
        ok and embedded == pr106_payload,
        f"{_rel(archive, repo)} embeds the selected PR106 0.bin payload"
        if ok and embedded == pr106_payload
        else (
            f"{_rel(archive, repo)} is not anchored to the selected PR106 payload "
            f"(kind={kind}, embedded_len={embedded_len}, expected_len={len(pr106_payload)})"
        ),
    )


def _check_zip_has_single_file(
    checks: list[CheckResult],
    *,
    name: str,
    path: Path | None,
    repo: Path,
) -> None:
    if path is None or not path.is_file():
        return
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
    except zipfile.BadZipFile as exc:
        _check(name, checks, False, f"{_rel(path, repo)} is not a valid ZIP: {exc}")
        return
    ok = len(infos) == 1 and not infos[0].is_dir()
    _check(
        name,
        checks,
        ok,
        f"{_rel(path, repo)} contains one file member"
        if ok
        else f"{_rel(path, repo)} must contain exactly one file member",
    )


def _check_manifest(
    checks: list[CheckResult],
    *,
    name: str,
    path: Path | None,
    repo: Path,
    production_readiness: bool,
    warnings: list[str],
) -> None:
    if path is None:
        if production_readiness:
            _check(name, checks, False, f"production readiness requires --{name.replace(':', '-')}")
        return
    if not path.is_file():
        detail = f"missing {_rel(path, repo)}"
        if production_readiness:
            _check(name, checks, False, detail)
        else:
            warnings.append(f"{name}: {detail}; ignored outside --production-readiness")
        return
    ok, detail = _json_manifest_contract(
        path,
        archive_path=None,
        repo=repo,
        require_dispatch_blockers=False,
    )
    _check(name, checks, ok, f"{_rel(path, repo)}: {detail}")


def _check_manifest_for_archive(
    checks: list[CheckResult],
    *,
    name: str,
    path: Path | None,
    archive_path: Path | None,
    repo: Path,
    production_readiness: bool,
    warnings: list[str],
    require_dispatch_blockers: bool = False,
) -> None:
    if path is None:
        if production_readiness:
            _check(name, checks, False, f"production readiness requires --{name.replace(':', '-')}")
        return
    if not path.is_file():
        detail = f"missing {_rel(path, repo)}"
        if production_readiness:
            _check(name, checks, False, detail)
        else:
            warnings.append(f"{name}: {detail}; ignored outside --production-readiness")
        return
    ok, detail = _json_manifest_contract(
        path,
        archive_path=archive_path,
        repo=repo,
        require_dispatch_blockers=require_dispatch_blockers,
    )
    _check(name, checks, ok, f"{_rel(path, repo)}: {detail}")


def _iter_optional_paths(inputs: ProductionInputs) -> Iterable[tuple[str, Path | None]]:
    yield "pr106_archive", inputs.pr106_archive
    yield "latent_sister_archive", inputs.latent_sister_archive
    yield "yshift_sister_archive", inputs.yshift_sister_archive
    yield "lrl1_sister_archive", inputs.lrl1_sister_archive
    yield "wavelet_sister_archive", inputs.wavelet_sister_archive
    yield "latent_manifest", inputs.latent_manifest
    yield "yshift_manifest", inputs.yshift_manifest
    yield "lrl1_manifest", inputs.lrl1_manifest
    yield "wavelet_manifest", inputs.wavelet_manifest
    yield "stacked_manifest", inputs.stacked_manifest


def _check_production_inputs(
    checks: list[CheckResult],
    warnings: list[str],
    *,
    repo: Path,
    production_readiness: bool,
    inputs: ProductionInputs,
) -> None:
    if not production_readiness:
        for name, path in _iter_optional_paths(inputs):
            if path is not None and not path.is_file():
                warnings.append(
                    f"{name}: missing {_rel(path, repo)}; ignored outside --production-readiness"
                )
        for manifest_name, path in (
            ("latent:manifest", inputs.latent_manifest),
            ("yshift:manifest", inputs.yshift_manifest),
            ("lrl1:manifest", inputs.lrl1_manifest),
            ("wavelet:manifest", inputs.wavelet_manifest),
            ("stacked:manifest", inputs.stacked_manifest),
        ):
            _check_manifest(
                checks,
                name=manifest_name,
                path=path,
                repo=repo,
                production_readiness=False,
                warnings=warnings,
            )
        return

    required_paths = (
        ("production:pr106-archive", inputs.pr106_archive, "pr106-archive"),
        ("production:latent-sister-archive", inputs.latent_sister_archive, "latent-sister-archive"),
        ("production:yshift-sister-archive", inputs.yshift_sister_archive, "yshift-sister-archive"),
        ("production:lrl1-sister-archive", inputs.lrl1_sister_archive, "lrl1-sister-archive"),
    )
    for name, path, kind in required_paths:
        _check_path_required(checks, name=name, path=path, repo=repo, kind=kind)

    for name, path in (
        ("production:pr106-archive-zero-bin", inputs.pr106_archive),
        ("production:latent-sister-zero-bin", inputs.latent_sister_archive),
        ("production:yshift-sister-zero-bin", inputs.yshift_sister_archive),
        ("production:lrl1-sister-zero-bin", inputs.lrl1_sister_archive),
    ):
        _check_zip_has_zero_bin(checks, name=name, path=path, repo=repo)
    _check_zip_has_single_file(
        checks,
        name="production:wavelet-sister-single-member",
        path=inputs.wavelet_sister_archive,
        repo=repo,
    )

    pr106_payload: bytes | None = None
    if inputs.pr106_archive is not None and inputs.pr106_archive.is_file():
        try:
            pr106_payload = _read_zero_bin(inputs.pr106_archive)
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            _check(
                "production:pr106-payload-read",
                checks,
                False,
                f"{_rel(inputs.pr106_archive, repo)} cannot read 0.bin: {exc}",
            )
        else:
            _check(
                "production:pr106-payload-magic",
                checks,
                bool(pr106_payload) and pr106_payload[0] == 0xFF,
                "PR106 0.bin has packed archive magic 0xFF"
                if bool(pr106_payload) and pr106_payload[0] == 0xFF
                else "PR106 0.bin must start with packed archive magic 0xFF",
            )

    for name, archive, kind in (
        ("production:latent-embeds-pr106", inputs.latent_sister_archive, "latent"),
        ("production:yshift-embeds-pr106", inputs.yshift_sister_archive, "yshift"),
        ("production:lrl1-embeds-pr106", inputs.lrl1_sister_archive, "lrl1"),
        ("production:wavelet-embeds-pr106", inputs.wavelet_sister_archive, "wavelet"),
    ):
        _check_embedded_pr106_payload(
            checks,
            name=name,
            archive=archive,
            pr106_payload=pr106_payload,
            repo=repo,
            kind=kind,
        )

    for manifest_name, path, archive in (
        ("latent:manifest", inputs.latent_manifest, inputs.latent_sister_archive),
        ("yshift:manifest", inputs.yshift_manifest, inputs.yshift_sister_archive),
        ("lrl1:manifest", inputs.lrl1_manifest, inputs.lrl1_sister_archive),
    ):
        _check_manifest_for_archive(
            checks,
            name=manifest_name,
            path=path,
            archive_path=archive,
            repo=repo,
            production_readiness=True,
            warnings=warnings,
        )
    _check_manifest_for_archive(
        checks,
        name="stacked:manifest",
        path=inputs.stacked_manifest,
        archive_path=None,
        repo=repo,
        production_readiness=True,
        warnings=warnings,
        require_dispatch_blockers=True,
    )
    if inputs.wavelet_sister_archive is not None or inputs.wavelet_manifest is not None:
        _check_manifest_for_archive(
            checks,
            name="wavelet:manifest",
            path=inputs.wavelet_manifest,
            archive_path=inputs.wavelet_sister_archive,
            repo=repo,
            production_readiness=True,
            warnings=warnings,
            require_dispatch_blockers=True,
        )


def run_dryrun(
    *,
    repo: Path = REPO,
    run_help: bool = True,
    production_readiness: bool = False,
    production_inputs: ProductionInputs | None = None,
    latent_search_mode: str = "heuristic",
    yshift_search_mode: str = "zero",
    lrl1_search_mode: str = "zero",
) -> DryRunReport:
    checks: list[CheckResult] = []
    warnings: list[str] = []
    repo = repo.resolve()
    inputs = production_inputs or ProductionInputs()

    for spec in BUILDER_SPECS:
        _check_builder_surface(repo, spec, checks, run_help=run_help)

    _check_selected_modes(
        checks,
        latent_search_mode=latent_search_mode,
        yshift_search_mode=yshift_search_mode,
        lrl1_search_mode=lrl1_search_mode,
    )
    _check_production_inputs(
        checks,
        warnings,
        repo=repo,
        production_readiness=production_readiness,
        inputs=inputs,
    )

    ok = all(c.ok for c in checks)
    return DryRunReport(
        ok=ok,
        checks=tuple(checks),
        warnings=tuple(sorted(set(warnings))),
        score_claim=False,
        dispatch_attempted=False,
        gpu_required=False,
        provider_state_free=True,
    )


def _print_text_report(report: DryRunReport, repo: Path) -> None:
    for check in report.checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    for warning in report.warnings:
        print(f"[WARN] {warning}")
    print("")
    print(f"score_claim={str(report.score_claim).lower()}")
    print(f"dispatch_attempted={str(report.dispatch_attempted).lower()}")
    print(f"gpu_required={str(report.gpu_required).lower()}")
    print(f"provider_state_free={str(report.provider_state_free).lower()}")
    print(f"repo={repo}")
    if report.ok:
        print("PR106 sidechannel dry-run PASSED: local readiness surfaces are intact.")
    else:
        failed = sum(1 for c in report.checks if not c.ok)
        print(f"PR106 sidechannel dry-run FAILED: {failed} check(s) failed.")
        print("Do NOT dispatch from this state.")


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print deterministic JSON report.")
    parser.add_argument(
        "--skip-help-subprocess",
        action="store_true",
        help="Skip target `--help` subprocess checks. Intended only for constrained test environments.",
    )
    parser.add_argument(
        "--production-readiness",
        action="store_true",
        help="Fail closed unless PR106 anchor, sister archives, and build manifests are provided and valid.",
    )
    parser.add_argument("--pr106-archive", type=str, default=None)
    parser.add_argument("--latent-sister-archive", type=str, default=None)
    parser.add_argument("--yshift-sister-archive", type=str, default=None)
    parser.add_argument("--lrl1-sister-archive", type=str, default=None)
    parser.add_argument("--wavelet-sister-archive", type=str, default=None)
    parser.add_argument("--latent-manifest", type=str, default=None)
    parser.add_argument("--yshift-manifest", type=str, default=None)
    parser.add_argument("--lrl1-manifest", type=str, default=None)
    parser.add_argument("--wavelet-manifest", type=str, default=None)
    parser.add_argument("--stacked-manifest", type=str, default=None)
    parser.add_argument(
        "--latent-search-mode",
        choices=("heuristic",),
        default="heuristic",
        help="Selected latent builder mode to preflight; only heuristic is dry-run safe.",
    )
    parser.add_argument(
        "--yshift-search-mode",
        choices=("zero", "gradient", "brute_force"),
        default="zero",
        help="Selected yshift mode to preflight. gradient/brute_force fail closed here.",
    )
    parser.add_argument(
        "--lrl1-search-mode",
        choices=("zero", "gradient", "brute_force"),
        default="zero",
        help="Selected lrl1 mode to preflight. gradient/brute_force fail closed here.",
    )
    args = parser.parse_args(argv)

    inputs = ProductionInputs(
        pr106_archive=_optional_path(args.pr106_archive),
        latent_sister_archive=_optional_path(args.latent_sister_archive),
        yshift_sister_archive=_optional_path(args.yshift_sister_archive),
        lrl1_sister_archive=_optional_path(args.lrl1_sister_archive),
        wavelet_sister_archive=_optional_path(args.wavelet_sister_archive),
        latent_manifest=_optional_path(args.latent_manifest),
        yshift_manifest=_optional_path(args.yshift_manifest),
        lrl1_manifest=_optional_path(args.lrl1_manifest),
        wavelet_manifest=_optional_path(args.wavelet_manifest),
        stacked_manifest=_optional_path(args.stacked_manifest),
    )
    report = run_dryrun(
        repo=REPO,
        run_help=not args.skip_help_subprocess,
        production_readiness=args.production_readiness,
        production_inputs=inputs,
        latent_search_mode=args.latent_search_mode,
        yshift_search_mode=args.yshift_search_mode,
        lrl1_search_mode=args.lrl1_search_mode,
    )
    if args.json:
        print(json_text(report.to_jsonable(REPO.resolve())), end="")
    else:
        _print_text_report(report, REPO.resolve())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
