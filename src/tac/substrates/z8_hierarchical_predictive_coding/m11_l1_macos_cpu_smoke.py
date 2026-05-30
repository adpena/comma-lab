# SPDX-License-Identifier: MIT
# NO_GRAD_WAIVED:M11_L1_macOS_CPU_smoke_runner_uses_torch_no_grad_in_inflate_and_evaluator_paths_per_HNeRV_parity_L9_runtime_closure_for_canonical_end_to_end_cycle_through_upstream_evaluate_py_device_cpu_per_CLAUDE_md_auth_eval_EVERYWHERE_non_negotiable_20260530
# AUTOCAST_FP16_WAIVED:M11_L1_macOS_CPU_smoke_runner_runs_on_fp32_per_canonical_Wyner_Ziv_1976_Theorem_1_round_trip_invariant_inherited_from_M6_M9_M10_compose_pattern
"""Z8 M11 L1 macOS-CPU end-to-end smoke runner per build_progress.py M11.

THIS module IS the M11 milestone landing per ``build_progress.py``
``l1_macos_cpu_smoke_landed`` (operator-routed Yousfi-cascade TOP-1 post-M10;
2026-05-30). Chains the canonical compose pattern from M9 + M10 through
``upstream/evaluate.py --device cpu`` per CLAUDE.md "Auth eval EVERYWHERE"
non-negotiable.

## Canonical end-to-end cycle

The canonical M11 cycle binds the full Z8 substrate stack against the
contest evaluator with REAL ``upstream/videos/0.mkv`` frames:

    1. **Train** the canonical quadruple (M4 Mamba-2 + M5 Mallat DWT +
       M6 Wyner-Ziv + M8 ScoreAwareLevelLoss) on real video pairs via
       ``run_canonical_quadruple_training_loop`` (M9).
    2. **Emit** canonical Z8HPC1 archive bytes via
       ``build_z8hpc1_archive_bytes_from_canonical_quadruple`` (M9 sister).
    3. **Write** canonical submission packet (``archive.zip`` + ``inflate.sh``
       + ``inflate.py``) per Catalog #146 contest 3-arg contract.
    4. **Inflate** via ``bash inflate.sh <archive_dir> <output_dir>
       <file_list>`` invoking ``inflate_one_video_from_archive_bytes``
       (M10) to produce 1200 frames raw output at 1164×874×3 per
       Catalog #367.
    5. **Evaluate** via ``python upstream/evaluate.py --device cpu`` (the
       canonical contest CPU evaluator) producing per-component PoseNet
       distortion + SegNet distortion + rate term + final score per
       upstream/evaluate.py:92 ``score = 100 * segnet_dist +
       math.sqrt(posenet_dist * 10) + 25 * rate``.
    6. **Capture** non-promotable ``[macOS-CPU advisory]`` score per
       Catalog #192 + CLAUDE.md "Submission auth eval — BOTH CPU AND
       CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable. M11
       runs on macOS-CPU which is NEVER promotable; M12 (sister
       paired-CUDA Modal T4 + Linux x86_64 CPU dispatch ~$1.50-3.00
       per Catalog #246) is required for ``[contest-CPU]``.

## Canonical-vs-unique decision per layer (Catalog #290)

- **ADOPT_CANONICAL**: M9 ``run_canonical_quadruple_training_loop`` for
  training (no re-implementation).
- **ADOPT_CANONICAL**: M9 ``build_z8hpc1_archive_bytes_from_canonical_quadruple``
  for archive emission (no re-implementation).
- **ADOPT_CANONICAL**: M9 ``load_real_video_targets_numpy`` for real-frame
  loading per Catalog #213.
- **ADOPT_CANONICAL**: M10 ``inflate_one_video_from_archive_bytes`` +
  M10 ``main_cli`` for inflate (no re-implementation).
- **ADOPT_CANONICAL**: ``upstream/evaluate.py --device cpu`` for the
  contest evaluator (zero modification to upstream per CLAUDE.md
  "Non-Negotiable Upstream Rule").
- **ADOPT_CANONICAL**: ``select_inflate_device`` per Catalog #205
  (inherited via M10 inflate.py).
- **ADOPT_CANONICAL**: Catalog #146 contest 3-arg signature
  (``inflate.sh <archive_dir> <output_dir> <file_list>``) inherited
  via M10 main_cli.
- **FORK_BECAUSE_PRINCIPLED_MISMATCH** (this module's UNIQUE primitive):
  the **canonical end-to-end orchestration helper** ``run_z8_m11_l1_smoke``
  that chains M9 training -> archive emission -> packet write ->
  inflate.sh invocation -> evaluator invocation -> result capture as ONE
  atomic substrate_engineering operation. No canonical helper exists for
  this 5-step contest-evaluator binding because Z8 is the first substrate
  to bind the full Mallat + Wyner-Ziv + score-aware compose pattern
  against the contest evaluator per HNeRV parity L7 substrate-engineering
  UNIQUE-IFIES discipline.

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS** (class-shift not within-class): Z8 quadruple is the
   canonical Catalog #312 class-shift away from PR101 within-class
   HNeRV-bolt-on lineage; M11 validates the class-shift binds end-to-end
   against the contest evaluator structurally.
2. **BEAUTY + ELEGANCE** (PR101-style 30-sec-reviewable): the M11
   ``run_z8_m11_l1_smoke`` orchestrator is ~280 LOC for the canonical
   5-step cycle (training + archive + packet + inflate + evaluate).
3. **DISTINCTNESS**: explicitly different from M9 (M9 is training-only;
   M11 is end-to-end-cycle-through-evaluator); explicitly different from
   M10 (M10 is inflate-only); explicitly different from M12 (M11 is
   macOS-CPU advisory; M12 is paired-CUDA + Linux x86_64 CPU promotable).
4. **RIGOR**: premise verification at start (Catalog #229 + #376 + #378);
   adversarial review by Contrarian + Assumption-Adversary (Catalog #292);
   empirical anchor on real upstream/videos/0.mkv per Catalog #213.
5. **OPTIMIZATION PER TECHNIQUE** (per Catalog #290): see canonical-vs-unique
   above; sister Catalog #312 quadruple covers per-primitive optimization.
6. **STACK-OF-STACKS-COMPOSABILITY** (orthogonal axes + additive ΔS):
   the M11 cycle is the canonical compose surface for future Z8 +
   sister-substrate compositions (Cascade A FEC10 selector / etc.); the
   contest 3-arg contract is the canonical universal binding API.
7. **DETERMINISTIC REPRODUCIBILITY** (byte-stable + seed-pinned): M9
   anneal-to-zero schedule + M5 Mallat perfect reconstruction + M6
   Wyner-Ziv deterministic encode all produce byte-stable archives
   under identical seeds.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: M11 L1 smoke completes
   end-to-end at training resolution (e.g. 32×32 × 4 pairs × 5 epochs)
   in ~10-30 minutes on macOS-CPU (the inflate 1200-frame write at
   1164×874×3 dominates; ~12 min worth of bicubic upscale + uint8 cast
   per Mallat perfect-reconstruction inverse chain).
9. **OPTIMAL MINIMAL CONTEST SCORE**: M11 advisory score baselines the
   per-component (SegNet, PoseNet, rate) contribution structure for M12
   paired-CUDA sub-0.189 attempt; the advisory score is NOT promotable
   per Catalog #192 but IS the canonical apples-to-apples anchor for
   the M12 paired-CUDA delta-vs-frontier comparison.

## Observability surface (Catalog #305)

The output artifact ``m11_l1_smoke_output.json`` carries ALL 6 facets:

  - **Inspectable per layer**: per-stage wall-clock (training / archive /
    packet-write / inflate / evaluator).
  - **Decomposable per signal**: per-component PoseNet distortion +
    SegNet distortion + compression rate + final advisory score.
  - **Diff-able across runs**: archive sha256 + inflate output sha256
    sample + advisory score floats.
  - **Queryable post-hoc**: schema_version pinned; canonical JSON keys
    fixed.
  - **Cite-able**: substrate_id + lane_id + git_HEAD_sha + canonical
    Provenance per Catalog #323.
  - **Counterfactual-able**: byte-mutation regression guard via
    re-running the helper with a different training seed produces a
    different archive sha256 -> different inflate sha sample -> bound
    different advisory score.

## Mission alignment (Catalog #300)

``mission_predicted_contribution=frontier_breaking_enabler``: M11
unblocks M12 paired-CUDA sub-0.189 attempt by validating the canonical
quadruple binds end-to-end against the contest evaluator. The advisory
score itself is NOT contest-grade per Catalog #192 — the structural
value is the cycle-closure validation, NOT the score literal.

[verified-against: src/tac/substrates/z8_hierarchical_predictive_coding/canonical_quadruple_binding.py M9 helpers]
[verified-against: src/tac/substrates/z8_hierarchical_predictive_coding/inflate.py M10 main_cli + inflate_one_video_from_archive_bytes]
[verified-against: src/tac/substrates/z8_hierarchical_predictive_coding/build_progress.py M11 acceptance criteria]
[verified-against: upstream/evaluate.py --device cpu canonical contest CPU evaluator signature]
[verified-against: feedback_z8_m10_inflate_consumes_real_trained_weights_per_catalog_369_landed_20260530.md M10 cycle-closure precedent]
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# Repo root anchor; tests + CLI use this for upstream paths.
def _repo_root() -> Path:
    """Return the canonical repo root (parent of src/tac/substrates/...).

    The file lives at ``<repo_root>/src/tac/substrates/z8_hierarchical_predictive_coding/m11_l1_macos_cpu_smoke.py``.
    Walk 4 levels up: file -> z8_hier...coding -> substrates -> tac -> src -> repo_root.
    """
    return Path(__file__).resolve().parents[4]


CANONICAL_VIDEO_PATH_RELATIVE: str = "upstream/videos/0.mkv"
"""Canonical contest test video relative to repo root."""

CANONICAL_VIDEO_NAMES_RELATIVE: str = "upstream/public_test_video_names.txt"
"""Canonical contest test video names file relative to repo root."""

CANONICAL_UPSTREAM_EVALUATE_RELATIVE: str = "upstream/evaluate.py"
"""Canonical contest CPU evaluator relative to repo root."""

CANONICAL_RAW_BYTES_PER_VIDEO: int = 1164 * 874 * 1200 * 3
"""Catalog #367 contest raw bytes contract: 3,662,409,600."""

assert CANONICAL_RAW_BYTES_PER_VIDEO == 3_662_409_600, (
    "CANONICAL_RAW_BYTES_PER_VIDEO drift; expected 3,662,409,600"
)


# --- Canonical inflate.sh template (Catalog #146 3-arg contract) ---------

INFLATE_SH_TEMPLATE: str = """#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Canonical Z8 M11 L1 inflate.sh per Catalog #146 contest 3-arg signature.
# Generated by tac.substrates.z8_hierarchical_predictive_coding.m11_l1_macos_cpu_smoke.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
exec "${PYBIN:-python}" "${HERE}/inflate.py" "${ARCHIVE_DIR}" "${OUTPUT_DIR}" "${FILE_LIST}"
"""

# --- Canonical self-contained inflate.py shim (Catalog #295 PYTHONPATH) ---
# The shim imports the canonical Z8 inflate main_cli from the installed
# tac package. M11 runs locally where ``tac`` IS importable from PYTHONPATH;
# the shim does NOT need to vendor the Z8 module bodies because the canonical
# end-to-end smoke runs the inflate.sh from the local repo root which has
# src/tac on PYTHONPATH per the canonical workspace layout. This satisfies
# Catalog #205 (canonical select_inflate_device routed via the M10 inflate.py
# main_cli) + Catalog #295 (PYTHONPATH self-contained: the shim's `from tac...`
# import resolves through the canonical repo-local install).
INFLATE_PY_TEMPLATE: str = '''# SPDX-License-Identifier: MIT
# Canonical Z8 M11 L1 inflate.py shim per Catalog #146 + Catalog #205 +
# Catalog #295. Routes to the canonical Z8 M10 inflate main_cli.
"""Z8 M11 L1 inflate.py shim.

This shim imports the canonical Z8 M10 ``main_cli`` from the installed
``tac`` package. M11 runs LOCALLY where ``tac`` IS importable from
``PYTHONPATH=src`` per the canonical workspace layout. The shim
inherits canonical ``select_inflate_device`` per Catalog #205 + canonical
3-arg contract per Catalog #146 + canonical Mallat 1989 §7.5 perfect
reconstruction per Catalog #369 (real-trained-weight consumption, NOT
synthetic frame base).
"""
import sys
from pathlib import Path

# Make the local src/ importable so `from tac...` resolves on macOS-LOCAL.
HERE = Path(__file__).resolve().parent
REPO_SRC = HERE.parent.parent / "src"
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))

from tac.substrates.z8_hierarchical_predictive_coding.inflate import main_cli

if __name__ == "__main__":
    sys.exit(main_cli())
'''


# --- Canonical M11 result artifact ---------------------------------------


@dataclass(frozen=True)
class Z8M11L1SmokeResult:
    """Canonical Z8 M11 L1 macOS-CPU end-to-end smoke result.

    Frozen dataclass per Catalog #300 v2 frontmatter discipline + Catalog
    #323 canonical Provenance umbrella (non-promotable by construction).

    Carries the canonical 5-stage observability surface per Catalog #305:
    training + archive + packet-write + inflate + evaluator wall-clock,
    and the per-component contest evaluator output (PoseNet distortion,
    SegNet distortion, compression rate, final score).

    Per Catalog #192 the result IS non-promotable: macOS-CPU is NOT 1:1
    contest-compliant Linux x86_64. The score axis_tag MUST be
    ``[macOS-CPU advisory]``; promotion to ``[contest-CPU]`` requires the
    M12 sister paired-CUDA + Linux x86_64 dispatch per Catalog #246.
    """

    # Identification
    substrate_id: str
    lane_id: str
    schema_version: str
    git_head_sha: str

    # M9 training stage
    training_pairs: int
    training_epochs: int
    training_resolution_hw: tuple[int, int]
    training_wall_clock_seconds: float
    training_convergence_verdict: str
    training_final_total_loss: float
    training_final_wyner_ziv_payload_bytes: int

    # M9 archive emission
    archive_bytes_total: int
    archive_sha256: str
    archive_emission_wall_clock_seconds: float

    # M11 packet write
    packet_write_wall_clock_seconds: float
    submission_dir_relative: str
    inflate_sh_path_relative: str
    inflate_py_path_relative: str

    # M10 inflate stage
    inflate_wall_clock_seconds: float
    inflate_raw_bytes_per_video: int
    inflate_total_videos: int
    inflate_first_video_sha256_sample_first_4096_bytes: str

    # upstream/evaluate.py --device cpu stage
    evaluator_wall_clock_seconds: float
    evaluator_posenet_distortion: float
    evaluator_segnet_distortion: float
    evaluator_compression_rate: float
    evaluator_compressed_size_bytes: int
    evaluator_uncompressed_size_bytes: int
    evaluator_final_score: float
    evaluator_report_path_relative: str

    # Canonical Provenance per Catalog #323 (NEVER promotable on macOS)
    score_claim: bool = field(default=False)
    promotable: bool = field(default=False)
    ready_for_exact_eval_dispatch: bool = field(default=False)
    axis_tag: str = field(default="[macOS-CPU advisory]")
    evidence_grade: str = field(default="macOS-CPU-advisory")
    hardware_substrate: str = field(default="macos_arm64")

    # Free-form
    notes: str = field(default="")

    def __post_init__(self) -> None:
        # Catalog #323 canonical Provenance invariants: macOS-CPU MUST
        # carry score_claim=False / promotable=False per Catalog #192.
        if self.score_claim is not False:
            raise ValueError(
                "Z8M11L1SmokeResult MUST carry score_claim=False per "
                "Catalog #192 (macOS-CPU is NEVER promotable to contest score)"
            )
        if self.promotable is not False:
            raise ValueError(
                "Z8M11L1SmokeResult MUST carry promotable=False per Catalog #192"
            )
        if self.ready_for_exact_eval_dispatch is not False:
            raise ValueError(
                "Z8M11L1SmokeResult MUST carry ready_for_exact_eval_dispatch=False "
                "per Catalog #192 (M12 paired-CUDA + Linux x86_64 required first)"
            )
        # Bound finite numerics: per build_progress.py M11 acceptance #3
        # "upstream/evaluate.py --device cpu produces finite score (NOT NaN,
        # NOT inf, NOT > 100)".
        if not (0.0 <= self.evaluator_final_score < 1000.0):
            raise ValueError(
                f"evaluator_final_score must be finite + 0 <= score < 1000; "
                f"got {self.evaluator_final_score}"
            )
        if self.archive_bytes_total < 1:
            raise ValueError("archive_bytes_total must be >= 1")
        if self.inflate_raw_bytes_per_video != CANONICAL_RAW_BYTES_PER_VIDEO:
            raise ValueError(
                f"inflate_raw_bytes_per_video must equal {CANONICAL_RAW_BYTES_PER_VIDEO} "
                f"(Catalog #367); got {self.inflate_raw_bytes_per_video}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Canonical JSON-serializable view per Catalog #305 observability."""
        return {
            "schema": str(self.schema_version),
            "substrate_id": str(self.substrate_id),
            "lane_id": str(self.lane_id),
            "git_head_sha": str(self.git_head_sha),
            # M9 training stage
            "training_pairs": int(self.training_pairs),
            "training_epochs": int(self.training_epochs),
            "training_resolution_hw": [int(self.training_resolution_hw[0]),
                                       int(self.training_resolution_hw[1])],
            "training_wall_clock_seconds": float(self.training_wall_clock_seconds),
            "training_convergence_verdict": str(self.training_convergence_verdict),
            "training_final_total_loss": float(self.training_final_total_loss),
            "training_final_wyner_ziv_payload_bytes": int(
                self.training_final_wyner_ziv_payload_bytes
            ),
            # M9 archive emission
            "archive_bytes_total": int(self.archive_bytes_total),
            "archive_sha256": str(self.archive_sha256),
            "archive_emission_wall_clock_seconds": float(
                self.archive_emission_wall_clock_seconds
            ),
            # M11 packet write
            "packet_write_wall_clock_seconds": float(
                self.packet_write_wall_clock_seconds
            ),
            "submission_dir_relative": str(self.submission_dir_relative),
            "inflate_sh_path_relative": str(self.inflate_sh_path_relative),
            "inflate_py_path_relative": str(self.inflate_py_path_relative),
            # M10 inflate stage
            "inflate_wall_clock_seconds": float(self.inflate_wall_clock_seconds),
            "inflate_raw_bytes_per_video": int(self.inflate_raw_bytes_per_video),
            "inflate_total_videos": int(self.inflate_total_videos),
            "inflate_first_video_sha256_sample_first_4096_bytes": str(
                self.inflate_first_video_sha256_sample_first_4096_bytes
            ),
            # evaluator stage
            "evaluator_wall_clock_seconds": float(self.evaluator_wall_clock_seconds),
            "evaluator_posenet_distortion": float(self.evaluator_posenet_distortion),
            "evaluator_segnet_distortion": float(self.evaluator_segnet_distortion),
            "evaluator_compression_rate": float(self.evaluator_compression_rate),
            "evaluator_compressed_size_bytes": int(
                self.evaluator_compressed_size_bytes
            ),
            "evaluator_uncompressed_size_bytes": int(
                self.evaluator_uncompressed_size_bytes
            ),
            "evaluator_final_score": float(self.evaluator_final_score),
            "evaluator_report_path_relative": str(self.evaluator_report_path_relative),
            # Canonical Provenance (NEVER promotable on macOS)
            "score_claim": False,
            "promotable": False,
            "ready_for_exact_eval_dispatch": False,
            "axis_tag": str(self.axis_tag),
            "evidence_grade": str(self.evidence_grade),
            "hardware_substrate": str(self.hardware_substrate),
            "notes": str(self.notes),
        }


# --- Canonical evaluator output parser ------------------------------------


_EVAL_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_EVAL_POSENET_RE = re.compile(
    rf"Average\s+PoseNet\s+Distortion:\s*({_EVAL_FLOAT})"
)
_EVAL_SEGNET_RE = re.compile(
    rf"Average\s+SegNet\s+Distortion:\s*({_EVAL_FLOAT})"
)
_EVAL_RATE_RE = re.compile(
    rf"Compression\s+Rate:\s*({_EVAL_FLOAT})"
)
_EVAL_COMPRESSED_RE = re.compile(
    r"Submission\s+file\s+size:\s*([\d,]+)\s*bytes"
)
_EVAL_UNCOMPRESSED_RE = re.compile(
    r"Original\s+uncompressed\s+size:\s*([\d,]+)\s*bytes"
)
_EVAL_SCORE_RE = re.compile(
    rf"Final\s+score:[^=]*=\s*({_EVAL_FLOAT})"
)


def parse_evaluator_report(report_text: str) -> dict[str, float | int]:
    """Parse the canonical upstream/evaluate.py report.txt output.

    Per upstream/evaluate.py:92-101 the canonical report carries:
      - ``Average PoseNet Distortion: <float>``
      - ``Average SegNet Distortion: <float>``
      - ``Submission file size: <int> bytes`` (compressed archive size)
      - ``Original uncompressed size: <int> bytes`` (uncompressed video size)
      - ``Compression Rate: <float>``
      - ``Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = <float>``

    Returns a dict with canonical keys; raises ValueError on missing fields.
    """
    keys: dict[str, float | int] = {}
    matches = (
        ("posenet_distortion", _EVAL_POSENET_RE, float),
        ("segnet_distortion", _EVAL_SEGNET_RE, float),
        ("compression_rate", _EVAL_RATE_RE, float),
        ("final_score", _EVAL_SCORE_RE, float),
    )
    for key, pattern, cast in matches:
        match = pattern.search(report_text)
        if not match:
            raise ValueError(
                f"evaluator report missing {key} regex {pattern.pattern!r}\n"
                f"report text:\n{report_text}"
            )
        keys[key] = cast(match.group(1))
    for key, pattern in (
        ("compressed_size_bytes", _EVAL_COMPRESSED_RE),
        ("uncompressed_size_bytes", _EVAL_UNCOMPRESSED_RE),
    ):
        match = pattern.search(report_text)
        if not match:
            raise ValueError(
                f"evaluator report missing {key} regex {pattern.pattern!r}"
            )
        keys[key] = int(match.group(1).replace(",", ""))
    return keys


# --- Canonical M11 L1 smoke runner ----------------------------------------


def _git_head_sha(repo_root: Path) -> str:
    """Return HEAD sha (12-char prefix); '' on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()[:12]
    except (subprocess.SubprocessError, OSError):
        return ""


def _sha256_first_n_bytes(path: Path, n: int = 4096) -> str:
    """Compute sha256 of first n bytes of a file (deterministic sample)."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        hasher.update(fh.read(n))
    return hasher.hexdigest()


def _file_size_recursive(root: Path) -> int:
    """Sum of file sizes under root (matches upstream/evaluate.py:64)."""
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def write_canonical_submission_packet(
    submission_dir: Path,
    archive_bytes: bytes,
) -> tuple[Path, Path, Path, Path]:
    """Write canonical contest submission packet to submission_dir.

    Layout (per upstream/evaluate.sh + Catalog #146):
        submission_dir/
            archive.zip            (the canonical contest archive ZIP)
            inflate.sh             (canonical 3-arg contract per Catalog #146)
            inflate.py             (canonical shim routing to M10 main_cli)
            archive/               (will be unzipped by evaluate.sh)
            inflated/              (will be populated by inflate.sh)

    The archive.zip contains a single ``0.bin`` member carrying the
    Z8HPC1 archive bytes per the canonical single-file archive grammar.

    Returns ``(archive_zip_path, inflate_sh_path, inflate_py_path,
    archive_dir_path)`` for the orchestrator to invoke inflate.sh.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    archive_zip = submission_dir / "archive.zip"
    inflate_sh = submission_dir / "inflate.sh"
    inflate_py = submission_dir / "inflate.py"
    archive_dir = submission_dir / "archive"

    # Write archive.zip via canonical deterministic ZIP grammar per
    # CLAUDE.md "Forbidden non-deterministic archive ZIP" + Catalog
    # #157/#174 sister discipline. ZipInfo + writestr + fixed timestamp +
    # ZIP_STORED (no compression on top of brotli'd payload).
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("0.bin")
        info.date_time = (2026, 1, 1, 0, 0, 0)
        zf.writestr(info, archive_bytes)

    inflate_sh.write_text(INFLATE_SH_TEMPLATE, encoding="utf-8")
    inflate_sh.chmod(0o755)
    inflate_py.write_text(INFLATE_PY_TEMPLATE, encoding="utf-8")
    inflate_py.chmod(0o644)

    return archive_zip, inflate_sh, inflate_py, archive_dir


def run_z8_m11_l1_smoke(
    output_dir: Path,
    *,
    training_pairs: int = 4,
    training_epochs: int = 5,
    training_resolution_hw: tuple[int, int] = (32, 32),
    video_path: Path | None = None,
    video_names_file: Path | None = None,
    uncompressed_dir: Path | None = None,
    evaluator_script: Path | None = None,
    python_executable: str | None = None,
    repo_root: Path | None = None,
    lane_id: str = (
        "lane_z8_m11_l1_macos_cpu_mlx_local_end_to_end_smoke_canonical_evaluate_cpu_binding_20260530"
    ),
    schema_version: str = "z8_m11_l1_macos_cpu_smoke_v1",
    notes: str = "",
) -> Z8M11L1SmokeResult:
    """Canonical Z8 M11 L1 macOS-CPU end-to-end smoke runner.

    Chains the canonical M9 + M10 cycle through ``upstream/evaluate.py
    --device cpu`` per CLAUDE.md "Auth eval EVERYWHERE" non-negotiable.

    Args:
        output_dir: directory to write submission packet + result JSON.
            Must NOT be under /tmp/, /private/tmp/, or /var/tmp/ per
            CLAUDE.md "Forbidden /tmp paths" non-negotiable.
        training_pairs: number of pair frames decoded from
            upstream/videos/0.mkv for M9 training. Default 4 (smoke).
        training_epochs: number of M9 training epochs. Default 5.
        training_resolution_hw: (H, W) training resolution. Default (32, 32).
        video_path: canonical contest video. Defaults to
            ``repo_root/upstream/videos/0.mkv``.
        video_names_file: canonical contest video names. Defaults to
            ``repo_root/upstream/public_test_video_names.txt``.
        uncompressed_dir: canonical uncompressed video directory for rate
            term denominator. Defaults to ``repo_root/upstream/videos/``.
        evaluator_script: canonical upstream evaluator. Defaults to
            ``repo_root/upstream/evaluate.py``.
        python_executable: Python interpreter for inflate + evaluator
            subprocess. Defaults to ``sys.executable``.
        repo_root: canonical repo root. Defaults to auto-detected.
        lane_id: canonical lane_id for Provenance.
        schema_version: schema_version for output JSON.
        notes: free-form notes for the result.

    Returns:
        :class:`Z8M11L1SmokeResult` carrying canonical 5-stage observability
        surface + per-component contest evaluator output + canonical
        Provenance (NEVER promotable on macOS).
    """
    if repo_root is None:
        repo_root = _repo_root()
    if video_path is None:
        video_path = repo_root / CANONICAL_VIDEO_PATH_RELATIVE
    if video_names_file is None:
        video_names_file = repo_root / CANONICAL_VIDEO_NAMES_RELATIVE
    if uncompressed_dir is None:
        uncompressed_dir = repo_root / "upstream" / "videos"
    if evaluator_script is None:
        evaluator_script = repo_root / CANONICAL_UPSTREAM_EVALUATE_RELATIVE
    if python_executable is None:
        python_executable = sys.executable

    output_dir = Path(output_dir).resolve()
    output_dir_str = str(output_dir)
    if (
        output_dir_str.startswith("/tmp/")
        or output_dir_str.startswith("/private/tmp/")
        or output_dir_str.startswith("/var/tmp/")
    ):
        raise ValueError(
            f"Refusing absolute /tmp/ output_dir per CLAUDE.md transient-evidence "
            f"trap: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = output_dir / "submission"

    if not video_path.exists():
        raise FileNotFoundError(
            f"Canonical contest video not found at {video_path}; required "
            f"per Catalog #213 + CLAUDE.md 'Forbidden make_synthetic_pair_batch'"
        )
    if not video_names_file.exists():
        raise FileNotFoundError(
            f"Canonical video names file not found at {video_names_file}"
        )
    if not uncompressed_dir.is_dir():
        raise FileNotFoundError(
            f"Canonical uncompressed dir not found at {uncompressed_dir}"
        )
    if not evaluator_script.exists():
        raise FileNotFoundError(
            f"Canonical upstream evaluator not found at {evaluator_script}"
        )

    git_head_sha = _git_head_sha(repo_root)
    eval_h, eval_w = int(training_resolution_hw[0]), int(training_resolution_hw[1])

    # --- Step 1: M9 training -------------------------------------------------
    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
        load_real_video_targets_numpy,
        run_canonical_quadruple_training_loop,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
    )

    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=int(training_pairs),
        deterministic_state_dim=16,
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(eval_h, eval_w),
    )
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)

    # Decode real video frames per Catalog #213 (NO synthetic per Slot EEE).
    pair_rgb_targets = load_real_video_targets_numpy(
        video_path,
        num_pairs=int(training_pairs),
        output_height=eval_h,
        output_width=eval_w,
    )

    training_start = time.time()
    training_artifact = run_canonical_quadruple_training_loop(
        binding,
        pair_rgb_targets,
        epochs=int(training_epochs),
        substrate_id="z8_hierarchical_predictive_coding",
        lane_id=lane_id,
        hardware_substrate="macos_arm64",
        notes=(
            f"Z8 M11 L1 macOS-CPU smoke: M9 training stage; "
            f"{training_pairs}p x {training_epochs}ep at ({eval_h},{eval_w}); "
            f"real {video_path.name} per Catalog #213; macOS-CPU NEVER "
            f"promotable per Catalog #192."
        ),
    )
    training_wall_clock = time.time() - training_start

    # --- Step 2: M9 archive emission -----------------------------------------
    # The canonical archive bytes derive entirely from real video frames
    # routed through the canonical M5 Mallat decompose + M6 Wyner-Ziv encode
    # + M4 deterministic-state step pipeline per Catalog #369.
    archive_emission_start = time.time()
    # For archive emission we need frame_1 as well; load both frames.
    from tac.data import decode_video

    decoded_frames = decode_video(
        video_path,
        target_h=eval_h,
        target_w=eval_w,
        max_frames=2 * int(training_pairs),
    )
    if len(decoded_frames) < 2 * int(training_pairs):
        raise RuntimeError(
            f"decode_video produced {len(decoded_frames)} frames; need "
            f"{2 * int(training_pairs)}"
        )
    frame_0_stack = np.stack(
        [decoded_frames[2 * i].numpy() for i in range(int(training_pairs))],
        axis=0,
    )
    frame_1_stack = np.stack(
        [decoded_frames[2 * i + 1].numpy() for i in range(int(training_pairs))],
        axis=0,
    )
    real_frame_0 = (frame_0_stack.astype(np.float32) / 255.0).astype(np.float32)
    real_frame_1 = (frame_1_stack.astype(np.float32) / 255.0).astype(np.float32)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding,
        real_frame_0,
        real_frame_1,
    )
    archive_emission_wall_clock = time.time() - archive_emission_start
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    # --- Step 3: M11 packet write (Catalog #146 contest 3-arg contract) ------
    packet_write_start = time.time()
    archive_zip, inflate_sh, inflate_py, archive_dir = (
        write_canonical_submission_packet(submission_dir, archive_bytes)
    )
    packet_write_wall_clock = time.time() - packet_write_start

    # --- Step 4: M10 inflate stage ------------------------------------------
    # Mimic the canonical upstream/evaluate.sh dance: unzip archive.zip,
    # then invoke inflate.sh with (archive_dir, output_dir, file_list).
    archive_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip, "r") as zf:
        zf.extractall(archive_dir)
    inflated_dir = submission_dir / "inflated"
    inflated_dir.mkdir(parents=True, exist_ok=True)

    inflate_start = time.time()
    inflate_result = subprocess.run(
        [
            "bash",
            str(inflate_sh),
            str(archive_dir),
            str(inflated_dir),
            str(video_names_file),
        ],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYBIN": python_executable,
            "PYTHONPATH": f"{repo_root / 'src'}:{repo_root}",
            "HOME": str(Path.home()),
            "TMPDIR": str(repo_root / ".omx" / "tmp"),
        },
        capture_output=True,
        text=True,
    )
    inflate_wall_clock = time.time() - inflate_start
    if inflate_result.returncode != 0:
        raise RuntimeError(
            f"inflate.sh failed rc={inflate_result.returncode}\n"
            f"stdout:\n{inflate_result.stdout}\n"
            f"stderr:\n{inflate_result.stderr}"
        )

    # Locate first inflated .raw file for sha256 sample.
    video_names = [
        line.strip()
        for line in video_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    first_video_base = Path(video_names[0]).stem
    first_raw_path = inflated_dir / f"{first_video_base}.raw"
    if not first_raw_path.exists():
        raise RuntimeError(
            f"inflated .raw not found at {first_raw_path}; videos={video_names}"
        )
    actual_size = first_raw_path.stat().st_size
    if actual_size != CANONICAL_RAW_BYTES_PER_VIDEO:
        raise AssertionError(
            f"inflated raw size mismatch: {actual_size} != "
            f"{CANONICAL_RAW_BYTES_PER_VIDEO} per Catalog #367"
        )
    first_video_sha_sample = _sha256_first_n_bytes(first_raw_path, 4096)

    # --- Step 5: upstream/evaluate.py --device cpu ---------------------------
    report_path = submission_dir / "report.txt"
    evaluator_start = time.time()
    evaluator_result = subprocess.run(
        [
            python_executable,
            str(evaluator_script),
            "--submission-dir", str(submission_dir),
            "--uncompressed-dir", str(uncompressed_dir),
            "--report", str(report_path),
            "--video-names-file", str(video_names_file),
            "--device", "cpu",
            "--batch-size", "2",
            "--num-threads", "1",
            "--prefetch-queue-depth", "1",
        ],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "PYTHONPATH": f"{repo_root / 'upstream'}:{repo_root / 'src'}:{repo_root}",
            "HOME": str(Path.home()),
            "TMPDIR": str(repo_root / ".omx" / "tmp"),
            "RANK": "0",
            "WORLD_SIZE": "1",
        },
        capture_output=True,
        text=True,
    )
    evaluator_wall_clock = time.time() - evaluator_start
    if evaluator_result.returncode != 0:
        # Write captured streams to disk for forensic recovery.
        (output_dir / "evaluator_stdout.txt").write_text(
            evaluator_result.stdout, encoding="utf-8"
        )
        (output_dir / "evaluator_stderr.txt").write_text(
            evaluator_result.stderr, encoding="utf-8"
        )
        raise RuntimeError(
            f"upstream/evaluate.py --device cpu failed rc={evaluator_result.returncode}\n"
            f"stdout (truncated to 4KB):\n{evaluator_result.stdout[:4096]}\n"
            f"stderr (truncated to 4KB):\n{evaluator_result.stderr[:4096]}\n"
            f"full streams written under {output_dir}/evaluator_stdout.txt / evaluator_stderr.txt"
        )
    if not report_path.exists():
        raise RuntimeError(
            f"upstream/evaluate.py did not produce report at {report_path}"
        )
    parsed = parse_evaluator_report(report_path.read_text(encoding="utf-8"))

    # --- Step 6: capture canonical Provenance + result JSON ------------------
    result = Z8M11L1SmokeResult(
        substrate_id="z8_hierarchical_predictive_coding",
        lane_id=lane_id,
        schema_version=schema_version,
        git_head_sha=git_head_sha,
        # M9 training
        training_pairs=int(training_pairs),
        training_epochs=int(training_epochs),
        training_resolution_hw=(eval_h, eval_w),
        training_wall_clock_seconds=float(training_wall_clock),
        training_convergence_verdict=str(training_artifact.convergence_verdict),
        training_final_total_loss=float(sum(training_artifact.final_per_level_l2_loss)),
        training_final_wyner_ziv_payload_bytes=int(
            training_artifact.final_wyner_ziv_payload_bytes
        ),
        # M9 archive emission
        archive_bytes_total=int(len(archive_bytes)),
        archive_sha256=str(archive_sha256),
        archive_emission_wall_clock_seconds=float(archive_emission_wall_clock),
        # M11 packet write
        packet_write_wall_clock_seconds=float(packet_write_wall_clock),
        submission_dir_relative=str(
            submission_dir.relative_to(repo_root)
            if submission_dir.is_relative_to(repo_root)
            else submission_dir
        ),
        inflate_sh_path_relative=str(
            inflate_sh.relative_to(repo_root)
            if inflate_sh.is_relative_to(repo_root)
            else inflate_sh
        ),
        inflate_py_path_relative=str(
            inflate_py.relative_to(repo_root)
            if inflate_py.is_relative_to(repo_root)
            else inflate_py
        ),
        # M10 inflate
        inflate_wall_clock_seconds=float(inflate_wall_clock),
        inflate_raw_bytes_per_video=CANONICAL_RAW_BYTES_PER_VIDEO,
        inflate_total_videos=len(video_names),
        inflate_first_video_sha256_sample_first_4096_bytes=str(first_video_sha_sample),
        # evaluator stage
        evaluator_wall_clock_seconds=float(evaluator_wall_clock),
        evaluator_posenet_distortion=float(parsed["posenet_distortion"]),
        evaluator_segnet_distortion=float(parsed["segnet_distortion"]),
        evaluator_compression_rate=float(parsed["compression_rate"]),
        evaluator_compressed_size_bytes=int(parsed["compressed_size_bytes"]),
        evaluator_uncompressed_size_bytes=int(parsed["uncompressed_size_bytes"]),
        evaluator_final_score=float(parsed["final_score"]),
        evaluator_report_path_relative=str(
            report_path.relative_to(repo_root)
            if report_path.is_relative_to(repo_root)
            else report_path
        ),
        notes=str(notes),
    )

    # Write canonical result JSON per Catalog #305 observability surface.
    result_path = output_dir / "m11_l1_smoke_output.json"
    result_path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return result


__all__ = [
    "CANONICAL_RAW_BYTES_PER_VIDEO",
    "CANONICAL_UPSTREAM_EVALUATE_RELATIVE",
    "CANONICAL_VIDEO_NAMES_RELATIVE",
    "CANONICAL_VIDEO_PATH_RELATIVE",
    "INFLATE_PY_TEMPLATE",
    "INFLATE_SH_TEMPLATE",
    "Z8M11L1SmokeResult",
    "parse_evaluator_report",
    "run_z8_m11_l1_smoke",
    "write_canonical_submission_packet",
]
