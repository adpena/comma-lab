# SPDX-License-Identifier: MIT
"""WAVE-3 PR101 GOLD master-gradient-null-byte REMOVAL smoke (LOCAL macOS-CPU).

Per PROCEDURAL-CODEBOOK BUILD landing memo (commit ``1dd8569de``) Top-3
op-routable #2 + null-byte probe matrix (commit ``82c1b3bac``) +
MAGIC-CODEC pair #2 META-LESSON (commit ``8e2134edc``).

NEW ARCHITECTURAL APPROACH distinct from substitution paradigm:
**REMOVE master-gradient-null bytes from charged surface + RECONSTRUCT
arbitrary constants at inflate**. Tests 4 variants of byte modification
on PR101 GOLD fec6 archive's 16,292 master-gradient-null byte indices.

Tests 3 hypotheses on LOCAL macOS-CPU (NEVER promotable per Catalog
#192 + #127 + #323):

* H1: bytes truly score-irrelevant (V_ZERO == V_HALF == V_RANDOM ==
  baseline within 1e-4 of 0.19205); BUILD justified (predicted dS
  -0.01086 per probe matrix).
* H2: bytes partially relevant (variants diverge but not catastrophic);
  predicted dS overestimated.
* H3: bytes opaque-to-scorer-but-not-bytes (substitution changes inflate
  decoded output catastrophically); paradigm needs rescope.

Pair #2 META-LESSON cascade pivot context: pair #2 EMPIRICALLY FALSIFIED
substitution paradigm at 101x residual zscore because null-byte VALUES
are near-random uniform-uint8 (NOT zero). REMOVAL paradigm is structurally
distinct: instead of trying to PREDICT byte values via codebook, test if
the bytes are score-irrelevant entirely.

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval - BOTH
CPU AND CUDA" non-negotiables: macOS-CPU smoke is observability-only;
contest-CPU paired Linux x86_64 anchor required for any promotion.

Catalog #270 tool dispatch scope per "tac stays clean" + canonical
dispatch optimization protocol.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.provenance import build_provenance_for_macos_cpu_advisory  # noqa: E402

# Canonical contest constants per CLAUDE.md "Auth eval EVERYWHERE" +
# canonical equation registry per Catalog #344.
CANONICAL_RATE_DENOM_BYTES = 37_545_489
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_SEG_MULTIPLIER = 100.0
CANONICAL_POSE_SQRT_INNER = 10.0
EPSILON_GRADIENT_NULL = 1e-9
FEC6_FRONTIER_SHA256 = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)
FEC6_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
FEC6_INFLATE_SH = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.sh"
)
MASTER_GRADIENT_NPY = (
    REPO_ROOT
    / "experiments/results/master_gradient_per_archive_fp64_extraction_wave_20260519T012404Z/master_gradient_pr101_fec6_frontier_macos_cpu_advisory_8pair_fp64_20260518.npy"
)
CONTEST_AUTH_EVAL = REPO_ROOT / "experiments/contest_auth_eval.py"

# Hypothesis classification thresholds (per task description; bounded
# observability-only).
HYPOTHESIS_H1_THRESHOLD = 1.0e-4  # all variants within this of baseline
HYPOTHESIS_H2_THRESHOLD = 0.05  # variants diverge but not catastrophic


class NullByteRemovalSmokeError(RuntimeError):
    """Raised when the smoke pipeline cannot complete."""


@dataclass(frozen=True)
class VariantResult:
    """One variant's empirical smoke result."""

    variant_name: str
    fill_byte_strategy: str
    archive_sha256: str
    archive_bytes: int
    score: float
    seg_distortion: float
    pose_distortion: float
    rate_term: float
    delta_s_vs_baseline: float | None
    eval_returncode: int
    eval_wallclock_seconds: float
    work_dir: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_master_gradient_null_indices() -> np.ndarray:
    """Load master-gradient .npy and return indices where row norm < EPSILON."""

    if not MASTER_GRADIENT_NPY.is_file():
        raise NullByteRemovalSmokeError(
            f"master_gradient .npy not found at {MASTER_GRADIENT_NPY} "
            "(per task description fixture path)"
        )
    mg = np.load(MASTER_GRADIENT_NPY)
    if mg.ndim != 2 or mg.shape[1] != 3:
        raise NullByteRemovalSmokeError(
            f"master_gradient shape unexpected: {mg.shape}; expected (N, 3) "
            "for (seg, pose, rate) components"
        )
    norms = np.linalg.norm(mg, axis=1)
    null_indices = np.where(norms < EPSILON_GRADIENT_NULL)[0]
    if len(null_indices) != 16_292:
        raise NullByteRemovalSmokeError(
            f"master_gradient null-byte count {len(null_indices)} != "
            "expected 16292 per probe matrix anchor (commit 82c1b3bac); "
            "fixture may have changed"
        )
    return null_indices


def _read_fec6_archive() -> tuple[bytes, bytes, str]:
    """Read fec6 frontier archive + verify sha256.

    Returns: (archive_zip_bytes, inner_member_bytes, member_name).
    """

    if not FEC6_FRONTIER_ARCHIVE.is_file():
        raise NullByteRemovalSmokeError(
            f"fec6 frontier archive not found at {FEC6_FRONTIER_ARCHIVE}"
        )
    archive_bytes = FEC6_FRONTIER_ARCHIVE.read_bytes()
    actual_sha = hashlib.sha256(archive_bytes).hexdigest()
    if actual_sha != FEC6_FRONTIER_SHA256:
        raise NullByteRemovalSmokeError(
            f"fec6 archive sha mismatch: {actual_sha} != "
            f"{FEC6_FRONTIER_SHA256} (canonical_frontier_pointer.json)"
        )
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise NullByteRemovalSmokeError(
                f"fec6 archive expected 1 member, got {len(names)}: {names}"
            )
        member_name = names[0]
        inner = zf.read(member_name)
    return archive_bytes, inner, member_name


def _derive_random_bytes_for_indices(
    n_indices: int, seed: bytes = b"pr101_gold_null_byte_removal_smoke_2026_05_20"
) -> bytes:
    """Deterministic pseudo-random bytes derived from seed via numpy PCG64."""

    seed_int = int.from_bytes(hashlib.sha256(seed).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed_int))
    return bytes(rng.integers(0, 256, size=n_indices, dtype=np.uint8))


def _build_variant_archive(
    inner_bytes: bytes,
    member_name: str,
    null_indices: np.ndarray,
    variant_name: str,
    fill_byte_strategy: str,
) -> tuple[bytes, str, int]:
    """Apply variant byte modification + repack archive.

    Variants:
    * V_BASELINE: original inner_bytes (control)
    * V_ZERO: 0x00 at every null-index
    * V_HALF: 0x80 at every null-index
    * V_RANDOM: deterministic pseudo-random bytes via PCG64
    """

    mutated = bytearray(inner_bytes)
    n_null = len(null_indices)

    if variant_name == "V_BASELINE":
        pass
    elif variant_name == "V_ZERO":
        for idx in null_indices:
            mutated[idx] = 0x00
    elif variant_name == "V_HALF":
        for idx in null_indices:
            mutated[idx] = 0x80
    elif variant_name == "V_RANDOM":
        random_bytes = _derive_random_bytes_for_indices(n_null)
        for i, idx in enumerate(null_indices):
            mutated[idx] = random_bytes[i]
    else:
        raise NullByteRemovalSmokeError(
            f"unknown variant {variant_name!r}; expected V_BASELINE / "
            "V_ZERO / V_HALF / V_RANDOM"
        )

    mutated_bytes = bytes(mutated)

    # Repack as deterministic ZIP per CLAUDE.md "Beauty, simplicity, and
    # developer experience" + canonical contest packet form (single
    # member, fixed timestamp, no extra fields).
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(filename=member_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 0
        zf.writestr(info, mutated_bytes)
    archive_zip_bytes = buf.getvalue()
    archive_sha = hashlib.sha256(archive_zip_bytes).hexdigest()
    return archive_zip_bytes, archive_sha, len(archive_zip_bytes)


def _run_contest_auth_eval_macos_cpu(
    archive_zip_bytes: bytes,
    variant_name: str,
    output_root: Path,
) -> tuple[float, float, float, float, int, float, Path]:
    """Run experiments/contest_auth_eval.py with --device cpu locally.

    Returns: (score, seg, pose, rate, returncode, wallclock_seconds, work_dir).
    """

    variant_dir = output_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    archive_path = variant_dir / "archive.zip"
    archive_path.write_bytes(archive_zip_bytes)
    work_dir = variant_dir / "auth_eval_work"
    work_dir.mkdir(exist_ok=True)
    json_out = variant_dir / "auth_eval_result.json"

    cmd = [
        sys.executable,
        str(CONTEST_AUTH_EVAL),
        "--archive",
        str(archive_path),
        "--inflate-sh",
        str(FEC6_INFLATE_SH),
        "--device",
        "cpu",
        "--work-dir",
        str(work_dir),
        "--json-out",
        str(json_out),
        "--keep-work-dir",
    ]
    # Thread .venv python through PACT_PYTHON_BIN so the canonical fec6
    # inflate.sh (which falls through to bare `python`) finds brotli +
    # other dependencies. Per CLAUDE.md "Tooling - non-negotiable" use
    # the uv-managed venv binary.
    import os as _os
    env = dict(_os.environ)
    env["PACT_PYTHON_BIN"] = sys.executable
    t0 = _dt.datetime.now()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    elapsed = (_dt.datetime.now() - t0).total_seconds()

    # Capture stderr tail for diagnostic surfacing even on inflate failure.
    stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
    (variant_dir / "auth_eval_stderr_tail.txt").write_text(stderr_tail)

    if proc.returncode != 0 or not json_out.is_file():
        # IMPORTANT: per task description hypothesis H3, an inflate failure
        # is a CANONICAL EMPIRICAL OUTCOME (the bytes are essential for
        # parsing even though master-gradient is zero). Return sentinel
        # values + nonzero rc so the smoke can complete the 4-variant
        # comparison rather than crashing. The hypothesis classifier
        # treats inflate failures as H3 by construction (score=inf
        # equivalent / catastrophic divergence).
        return (
            float("nan"),  # score
            float("nan"),  # seg
            float("nan"),  # pose
            float("nan"),  # rate
            proc.returncode,
            elapsed,
            work_dir,
        )
    result = json.loads(json_out.read_text())
    # Prefer canonical_score (full precision) over final_score (rounded to
    # 2 decimals per contest evaluate.py display). Canonical formula:
    # 100*seg + sqrt(10*pose) + 25*archive_bytes/37_545_489.
    score = float(
        result.get(
            "canonical_score",
            result.get("score_recomputed_from_components", result.get("final_score", float("nan"))),
        )
    )
    seg = float(result.get("avg_segnet_dist", float("nan")))
    pose = float(result.get("avg_posenet_dist", float("nan")))
    rate = float(result.get("score_rate_contribution", float("nan")))
    return score, seg, pose, rate, proc.returncode, elapsed, work_dir


def _classify_hypothesis(
    baseline_score: float,
    v_zero_delta: float,
    v_half_delta: float,
    v_random_delta: float,
    *,
    n_inflate_failures: int = 0,
) -> tuple[str, str]:
    """Return (hypothesis_label, rationale).

    If any variant inflate fails (n_inflate_failures > 0), the verdict is
    H3_OPAQUE_TO_SCORER by construction — the bytes are bit-essential
    for parsing even though master-gradient is zero (a HARD-EARNED
    empirical finding distinct from scalar-score divergence).
    """

    if n_inflate_failures > 0:
        return (
            "H3_OPAQUE_TO_SCORER",
            f"{n_inflate_failures}/3 modified variants FAILED to inflate "
            "(crashed during contest auth_eval inflate step); the master-"
            "gradient-null bytes are bit-essential for archive parser "
            "(e.g. PR101 magic header / Huffman table headers / FEC6 "
            "format metadata) even though they have zero gradient leverage "
            "on score. This is the same META class as pair #2 falsification "
            "but at a DEEPER surface: pair #2 bytes broke residual-codec "
            "predictor-empirical match; H3 here breaks inflate parser "
            "directly. REMOVAL paradigm needs RESCOPE to exclude parser-"
            "essential null-gradient regions.",
        )

    max_abs_delta = max(abs(v_zero_delta), abs(v_half_delta), abs(v_random_delta))
    if max_abs_delta < HYPOTHESIS_H1_THRESHOLD:
        return (
            "H1_SCORE_IRRELEVANT",
            f"max(|dS|)={max_abs_delta:.6f} < {HYPOTHESIS_H1_THRESHOLD} threshold; "
            "bytes truly score-irrelevant; BUILD justified per probe matrix "
            "predicted dS -0.01086",
        )
    if max_abs_delta < HYPOTHESIS_H2_THRESHOLD:
        return (
            "H2_PARTIALLY_RELEVANT",
            f"max(|dS|)={max_abs_delta:.6f} in [{HYPOTHESIS_H1_THRESHOLD}, "
            f"{HYPOTHESIS_H2_THRESHOLD}); bytes partially relevant; predicted "
            "dS overestimated; cascade pivot recommended",
        )
    return (
        "H3_OPAQUE_TO_SCORER",
        f"max(|dS|)={max_abs_delta:.6f} >= {HYPOTHESIS_H2_THRESHOLD}; bytes "
        "opaque-to-scorer-but-not-bytes; paradigm needs rescope; substitution "
        "changes inflate decoded output catastrophically (same META class as "
        "pair #2 falsification)",
    )


def _write_smoke_result_json(
    output_dir: Path,
    variants: list[VariantResult],
    hypothesis_label: str,
    hypothesis_rationale: str,
    null_index_count: int,
) -> Path:
    """Emit smoke_result.json with full provenance per Catalog #323."""

    baseline = next(v for v in variants if v.variant_name == "V_BASELINE")
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=baseline.archive_sha256,
        source_path=str(output_dir.relative_to(REPO_ROOT)),
    )
    payload = {
        "schema_version": "pr101_gold_master_gradient_null_byte_removal_smoke_v1_20260520",
        "smoke_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "fec6_frontier_archive_sha256": FEC6_FRONTIER_SHA256,
        "fec6_frontier_archive_bytes": baseline.archive_bytes,
        "master_gradient_null_index_count": null_index_count,
        "master_gradient_null_fraction_pct": (
            null_index_count / 178_417 * 100
        ),
        "variants": [v.as_dict() for v in variants],
        "hypothesis_label": hypothesis_label,
        "hypothesis_rationale": hypothesis_rationale,
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "provenance": {
            "artifact_kind": str(provenance.artifact_kind.value if hasattr(provenance.artifact_kind, "value") else provenance.artifact_kind),
            "evidence_grade": str(provenance.evidence_grade.value if hasattr(provenance.evidence_grade, "value") else provenance.evidence_grade),
            "measurement_axis": provenance.measurement_axis,
            "hardware_substrate": provenance.hardware_substrate,
            "source_sha256": provenance.source_sha256,
            "source_path": provenance.source_path,
            "captured_at_utc": provenance.captured_at_utc,
            "score_claim_valid": provenance.score_claim_valid,
            "promotion_eligible": provenance.promotion_eligible,
        },
        "catalog_disciplines_honored": [
            "#125 6-hook wire-in",
            "#127 axis x hardware x evidence_grade custody",
            "#185 META drift",
            "#192 macOS-CPU non-promotable",
            "#272 byte-mutation smoke",
            "#287 placeholder-rationale rejection",
            "#318 master-gradient null-space surface",
            "#323 canonical Provenance umbrella",
            "#344 canonical equation cross-ref",
        ],
        "canonical_equation_cross_ref": (
            "procedural_codebook_from_seed_compression_savings_v1 (Catalog #344 "
            "registry #26); NEW IN-DOMAIN context candidate "
            "'master_gradient_null_byte_removal_with_constant_reconstruction' "
            "anchored ONLY if H1 verdict"
        ),
        "cascade_context": {
            "parent_landing_commit": "1dd8569de",
            "probe_matrix_commit": "82c1b3bac",
            "pair_2_falsification_commit": "8e2134edc",
            "pair_2_meta_lesson": (
                "pair #2 EMPIRICALLY FALSIFIED substitution paradigm at 101x "
                "residual zscore because null-byte VALUES are near-random "
                "uniform-uint8 (NOT zero); REMOVAL paradigm tested HERE is "
                "structurally distinct"
            ),
        },
    }
    out_path = output_dir / "smoke_result.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def _write_smoke_result_md(
    output_dir: Path,
    variants: list[VariantResult],
    hypothesis_label: str,
    hypothesis_rationale: str,
    null_index_count: int,
) -> Path:
    """Emit human-readable smoke_result.md."""

    baseline = next(v for v in variants if v.variant_name == "V_BASELINE")
    lines = [
        "<!-- HISTORICAL_SCORE_LITERAL_OK:macos_cpu_advisory_smoke_not_score_truth_pr101_gold_null_byte_removal_2026-05-20 -->",
        "# PR101 GOLD master-gradient-null-byte REMOVAL smoke",
        "",
        f"- **fec6 frontier archive sha256**: `{FEC6_FRONTIER_SHA256[:16]}...`",
        f"- **fec6 frontier archive bytes**: {baseline.archive_bytes:,}",
        f"- **master-gradient null-byte indices**: {null_index_count:,} "
        f"({null_index_count/178_417*100:.2f}% null fraction)",
        f"- **smoke_at_utc**: {_dt.datetime.now(tz=_dt.timezone.utc).isoformat()}",
        f"- **axis tag**: `[macOS-CPU advisory]` (NEVER promotable per Catalog #192)",
        f"- **$ spent**: $0 (LOCAL macOS-CPU)",
        "",
        "## 4-variant smoke results",
        "",
        "| variant | fill byte strategy | archive bytes | score | seg | pose | rate | dS vs baseline | wallclock_s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in variants:
        ds_str = f"{v.delta_s_vs_baseline:+.6f}" if v.delta_s_vs_baseline is not None else "(baseline)"
        lines.append(
            f"| `{v.variant_name}` | {v.fill_byte_strategy} | "
            f"{v.archive_bytes:,} | {v.score:.6f} | {v.seg_distortion:.6f} | "
            f"{v.pose_distortion:.6f} | {v.rate_term:.6f} | {ds_str} | "
            f"{v.eval_wallclock_seconds:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Hypothesis verdict",
            "",
            f"**Verdict**: `{hypothesis_label}`",
            "",
            f"**Rationale**: {hypothesis_rationale}",
            "",
            "## Cascade context",
            "",
            "- Parent landing commit: `1dd8569de` (PROCEDURAL-CODEBOOK BUILD)",
            "- Probe matrix commit: `82c1b3bac` (16,292 null-byte indices identified)",
            "- Pair #2 falsification commit: `8e2134edc` (substitution paradigm EMPIRICALLY FALSIFIED at 101x residual zscore)",
            "- This smoke tests REMOVAL paradigm (structurally distinct from substitution)",
            "",
            "## Provenance (Catalog #323)",
            "",
            "- `score_claim`: False",
            "- `promotion_eligible`: False",
            "- `rank_or_kill_eligible`: False",
            "- `ready_for_exact_eval_dispatch`: False",
            "- `axis_tag`: `[macOS-CPU advisory]`",
            "- `evidence_grade`: `macOS-CPU-advisory`",
            "",
            "## Operator-routable next-actions",
            "",
        ]
    )

    if hypothesis_label == "H1_SCORE_IRRELEVANT":
        lines.extend(
            [
                "1. **BUILD** ~200-400 LOC archive surgery + inflate constant-reconstruction; predicted dS -0.01086 per probe matrix (Catalog #344 anchor candidate)",
                "2. **Catalog #325** per-substrate symposium BEFORE paid dispatch",
                "3. **Catalog #324** post-training Tier-C validation on landed archive",
            ]
        )
    elif hypothesis_label == "H2_PARTIALLY_RELEVANT":
        lines.extend(
            [
                "1. **DEFER** REMOVAL paradigm pending mechanism investigation",
                "2. **PIVOT** to DP1 procedural codebook smoke (different bytes class)",
                "3. **AMEND** canonical equation #26 domain_of_validity to mark REMOVAL as INDETERMINATE pending further probes",
            ]
        )
    else:  # H3
        lines.extend(
            [
                "1. **DEFER-PENDING-RESCOPE** REMOVAL paradigm: same META class as pair #2 substitution falsification (bytes are SCORE-OPAQUE but BYTE-OPAQUE)",
                "2. **PIVOT** to substrate-level architectural changes (NSCS06 v8, ATW V2 codec quantizer LUT) which are IN-DOMAIN per canonical equation #26",
                "3. **AMEND** canonical equation #26 to EXCLUDE `master_gradient_null_byte_removal_with_constant_reconstruction` (per Catalog #344 domain_refined event); BUILD investment AVOIDED ~$200-400 LOC",
            ]
        )
    return _write_path_with_lines(output_dir / "smoke_result.md", lines)


def _write_path_with_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PR101 GOLD master-gradient-null-byte REMOVAL smoke (LOCAL "
            "macOS-CPU). Tests 4 variants of byte modification on fec6 "
            "frontier 16,292 null-byte indices + disambiguates H1/H2/H3 "
            "hypotheses. Catalog #270 tool dispatch scope."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/results/pr101_gold_master_gradient_null_byte_removal_smoke_<utc>/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not run contest_auth_eval (verify fixture paths + master-gradient indices)",
    )
    args = parser.parse_args(argv)

    null_indices = _load_master_gradient_null_indices()
    archive_zip_bytes, inner_bytes, member_name = _read_fec6_archive()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "null_index_count": int(len(null_indices)),
                    "fec6_archive_bytes": len(archive_zip_bytes),
                    "fec6_member_bytes": len(inner_bytes),
                    "fec6_member_name": member_name,
                    "fec6_sha256_verified": FEC6_FRONTIER_SHA256[:16] + "...",
                    "planned_variants": ["V_BASELINE", "V_ZERO", "V_HALF", "V_RANDOM"],
                },
                indent=2,
            )
        )
        return 0

    utc = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        REPO_ROOT
        / f"experiments/results/pr101_gold_master_gradient_null_byte_removal_smoke_{utc}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    variants_spec = [
        ("V_BASELINE", "original inner_bytes (control)"),
        ("V_ZERO", "0x00 at every null-index"),
        ("V_HALF", "0x80 at every null-index"),
        ("V_RANDOM", "deterministic PCG64 pseudo-random bytes"),
    ]
    variants: list[VariantResult] = []
    baseline_score: float | None = None

    for variant_name, fill_strategy in variants_spec:
        print(f"[{variant_name}] building variant archive...")
        variant_zip_bytes, variant_sha, variant_bytes = _build_variant_archive(
            inner_bytes,
            member_name,
            null_indices,
            variant_name,
            fill_strategy,
        )
        print(
            f"[{variant_name}] archive sha={variant_sha[:16]}... bytes={variant_bytes}; "
            f"running contest_auth_eval --device cpu..."
        )
        score, seg, pose, rate, rc, elapsed, work_dir = _run_contest_auth_eval_macos_cpu(
            variant_zip_bytes, variant_name, output_dir
        )
        delta = None if baseline_score is None else (score - baseline_score)
        if variant_name == "V_BASELINE":
            baseline_score = score
        variants.append(
            VariantResult(
                variant_name=variant_name,
                fill_byte_strategy=fill_strategy,
                archive_sha256=variant_sha,
                archive_bytes=variant_bytes,
                score=score,
                seg_distortion=seg,
                pose_distortion=pose,
                rate_term=rate,
                delta_s_vs_baseline=delta,
                eval_returncode=rc,
                eval_wallclock_seconds=elapsed,
                work_dir=str(work_dir.relative_to(REPO_ROOT)),
            )
        )
        print(
            f"[{variant_name}] score={score:.6f} "
            f"dS_vs_baseline={'(baseline)' if delta is None else f'{delta:+.6f}'} "
            f"({elapsed:.1f}s)"
        )

    assert baseline_score is not None
    v_zero = next(v for v in variants if v.variant_name == "V_ZERO")
    v_half = next(v for v in variants if v.variant_name == "V_HALF")
    v_random = next(v for v in variants if v.variant_name == "V_RANDOM")
    import math as _math
    n_inflate_failures = sum(
        1
        for v in (v_zero, v_half, v_random)
        if v.eval_returncode != 0 or _math.isnan(v.score)
    )
    hypothesis_label, hypothesis_rationale = _classify_hypothesis(
        baseline_score,
        v_zero.delta_s_vs_baseline if v_zero.delta_s_vs_baseline is not None and not _math.isnan(v_zero.delta_s_vs_baseline) else 0.0,
        v_half.delta_s_vs_baseline if v_half.delta_s_vs_baseline is not None and not _math.isnan(v_half.delta_s_vs_baseline) else 0.0,
        v_random.delta_s_vs_baseline if v_random.delta_s_vs_baseline is not None and not _math.isnan(v_random.delta_s_vs_baseline) else 0.0,
        n_inflate_failures=n_inflate_failures,
    )

    json_path = _write_smoke_result_json(
        output_dir, variants, hypothesis_label, hypothesis_rationale, len(null_indices)
    )
    md_path = _write_smoke_result_md(
        output_dir, variants, hypothesis_label, hypothesis_rationale, len(null_indices)
    )

    print("")
    print(f"=== SMOKE RESULT ===")
    print(f"Hypothesis: {hypothesis_label}")
    print(f"Rationale: {hypothesis_rationale}")
    print(f"JSON: {json_path.relative_to(REPO_ROOT)}")
    print(f"MD: {md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
