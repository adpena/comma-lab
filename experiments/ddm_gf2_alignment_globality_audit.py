#!/usr/bin/env python3
"""Certify the GF2 rigid-static ceiling without trusting coordinate descent.

For every allowed integer translation, this audit records the class histogram
seen on the common interior of the 512x384 lattice.  If two frames are rendered
from one shared categorical field, their combined errors are at least the
total-variation distance between those two histograms.  Summing that inequality
over any disjoint pairing of the 600 frames gives a lower bound on the errors of
*every* shared field and every choice of per-frame translations in the declared
[-12,+12]^2 family.

The retained GF2 fit is therefore used only as an observed upper bound.  This
runner supplies the missing global lower bound needed to decide whether the
charter's 292,264-mismatch optimistic ceiling can be refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

SCHEMA: Final = "ddm_gf2_alignment_globality_audit.v1"
MANIFEST_SCHEMA: Final = "ddm_gf2_alignment_globality_manifest.v1"
AXIS: Final = "[macOS-CPU scorer-free exact field measurement, n600]"
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
NUM_CLASSES: Final = 5
FIELD_SHAPE: Final = (N_PAIRS, HEIGHT, WIDTH)
FIELD_BYTES: Final = int(np.prod(FIELD_SHAPE))
FIELD_SHA256: Final = (
    "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
)
DEFAULT_FIELD: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/"
    "converged_v3/retained/source_afr1_jbp1_field.u8"
)
OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/"
    "globality_audit_v1"
)
SEARCH_RADIUS: Final = 12
MAX_CEILING_COMPATIBLE_MISMATCHES: Final = 292_264
MINIMUM_FREE_BYTES: Final = 1 << 30


class GlobalityAuditError(RuntimeError):
    """A source, theorem-input, retention, or resource invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fact_matches(fact: object) -> bool:
    if not isinstance(fact, dict) or "path" not in fact:
        return False
    path = Path(str(fact["path"]))
    return bool(
        path.is_file()
        and path.stat().st_size == int(fact.get("bytes", -1))
        and sha256_file(path) == fact.get("sha256")
    )


def atomic_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return file_fact(path)


def atomic_array(path: Path, array: np.ndarray) -> dict[str, object]:
    return atomic_bytes(path, np.ascontiguousarray(array).tobytes())


def atomic_json(path: Path, value: object) -> dict[str, object]:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return atomic_bytes(path, payload)


def peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(observed if sys.platform == "darwin" else observed * 1024)


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def storage_preflight(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(output)
    free = int(stat.f_bavail * stat.f_frsize)
    row = {
        "path": str(output),
        "observed_free_bytes": free,
        "required_free_bytes": MINIMUM_FREE_BYTES,
        "status": "PASS" if free >= MINIMUM_FREE_BYTES else "REFUSED",
        "retention_policy": "certify-or-block; generated audit payloads are retained",
    }
    atomic_json(output / "stage_checkpoints/00_storage_preflight.json", row)
    if free < MINIMUM_FREE_BYTES:
        raise GlobalityAuditError(f"Vertigo storage preflight refused: {row}")
    return row


def validate_source(path: Path) -> dict[str, object]:
    fact = file_fact(path)
    if fact["bytes"] != FIELD_BYTES or fact["sha256"] != FIELD_SHA256:
        raise GlobalityAuditError(f"exact field identity mismatch: {fact}")
    return fact


def ordered_translations(radius: int) -> np.ndarray:
    rows = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    return np.asarray(rows, dtype="<i2")


def shifted_interior_histograms(
    target: np.ndarray, *, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact class counts for every frame and allowed interior shift."""

    if target.shape != FIELD_SHAPE:
        raise GlobalityAuditError(
            f"field shape {target.shape} does not match declared {FIELD_SHAPE}"
        )
    translations = ordered_translations(radius)
    top = radius + translations[:, 0].astype(np.int64)
    bottom = HEIGHT - radius + translations[:, 0].astype(np.int64)
    left = radius + translations[:, 1].astype(np.int64)
    right = WIDTH - radius + translations[:, 1].astype(np.int64)
    histograms = np.empty(
        (N_PAIRS, translations.shape[0], NUM_CLASSES), dtype="<i4"
    )
    for pair in range(N_PAIRS):
        frame = np.asarray(target[pair])
        if np.any(frame >= NUM_CLASSES):
            raise GlobalityAuditError(f"frame {pair} carries a class outside [0,4]")
        for label in range(NUM_CLASSES):
            integral = np.zeros((HEIGHT + 1, WIDTH + 1), dtype=np.int32)
            integral[1:, 1:] = np.cumsum(
                np.cumsum(frame == label, axis=0, dtype=np.int32),
                axis=1,
                dtype=np.int32,
            )
            histograms[pair, :, label] = (
                integral[bottom, right]
                - integral[top, right]
                - integral[bottom, left]
                + integral[top, left]
            )
    expected_sites = (HEIGHT - 2 * radius) * (WIDTH - 2 * radius)
    if not np.all(histograms.sum(axis=2, dtype=np.int64) == expected_sites):
        raise GlobalityAuditError("shifted histogram totals do not equal the interior area")
    return histograms, translations


def extreme_pairing(values: np.ndarray) -> np.ndarray:
    """Pair opposite ends of a scalar ordering, with frame id as the tie-break."""

    if values.shape != (N_PAIRS,) or N_PAIRS % 2:
        raise GlobalityAuditError("pairing requires one scalar per frame and even N_PAIRS")
    frames = np.arange(N_PAIRS, dtype=np.int32)
    order = np.lexsort((frames, values))
    return np.stack((order[: N_PAIRS // 2], order[: N_PAIRS // 2 - 1 : -1]), axis=1)


def candidate_pairings(histograms: np.ndarray, translations: np.ndarray) -> dict[str, np.ndarray]:
    if N_PAIRS % 2:
        raise GlobalityAuditError("globality proof requires an even number of frames")
    zero_rows = np.flatnonzero(np.all(translations == 0, axis=1))
    if zero_rows.size != 1:
        raise GlobalityAuditError("translation list does not contain exactly one zero shift")
    center = histograms[:, int(zero_rows[0]), :]
    pairings = {
        "fixed_half_offset": np.stack(
            (
                np.arange(N_PAIRS // 2, dtype=np.int32),
                np.arange(N_PAIRS // 2, N_PAIRS, dtype=np.int32),
            ),
            axis=1,
        )
    }
    for label in range(NUM_CLASSES):
        pairings[f"extreme_center_class_{label}"] = extreme_pairing(center[:, label])
    return pairings


def interval_pair_lower_bound(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
) -> int:
    separated = np.maximum(left_min - right_max, right_min - left_max)
    return int(np.maximum(separated, 0).max())


def pairing_interval_lower_bound(histograms: np.ndarray, pairs: np.ndarray) -> int:
    minima = histograms.min(axis=1).astype(np.int64)
    maxima = histograms.max(axis=1).astype(np.int64)
    return sum(
        interval_pair_lower_bound(minima[a], maxima[a], minima[b], maxima[b])
        for a, b in pairs.tolist()
    )


def minimum_histogram_tv(left: np.ndarray, right: np.ndarray) -> int:
    """Exact minimum histogram TV across both frames' allowed translations."""

    left64 = left.astype(np.int64, copy=False)
    right64 = right.astype(np.int64, copy=False)
    l1 = np.abs(left64[:, None, :] - right64[None, :, :]).sum(axis=2)
    if np.any(l1 % 2):
        raise GlobalityAuditError("equal-area categorical histograms have odd L1 distance")
    return int(l1.min() // 2)


def exact_pairing_lower_bounds(histograms: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    bounds = np.empty(pairs.shape[0], dtype="<i4")
    for row, (left, right) in enumerate(pairs.tolist()):
        bounds[row] = minimum_histogram_tv(histograms[left], histograms[right])
    return bounds


def choose_and_bound_pairing(
    histograms: np.ndarray, translations: np.ndarray
) -> dict[str, object]:
    pairings = candidate_pairings(histograms, translations)
    interval_rows = {
        name: pairing_interval_lower_bound(histograms, pairs)
        for name, pairs in pairings.items()
    }
    selected_name = min(
        interval_rows,
        key=lambda name: (-interval_rows[name], name),
    )
    selected_pairs = pairings[selected_name]
    exact_bounds = exact_pairing_lower_bounds(histograms, selected_pairs)
    return {
        "candidate_interval_lower_bounds": interval_rows,
        "selected_pairing": selected_name,
        "selected_pairs": selected_pairs.tolist(),
        "per_pair_exact_histogram_tv_lower_bounds": exact_bounds.tolist(),
        "certified_global_mismatch_lower_bound": int(exact_bounds.sum(dtype=np.int64)),
    }


def completed_result(output: Path, source: dict[str, object]) -> dict[str, object] | None:
    result_path = output / "RESULT.json"
    manifest_path = output / "MANIFEST.json"
    if not result_path.is_file() or not manifest_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA or result.get("source_field") != source:
        raise GlobalityAuditError("retained result belongs to a different source or schema")
    runner = file_fact(Path(__file__).resolve())
    if result.get("provenance", {}).get("runner") != runner:
        raise GlobalityAuditError("retained result was produced by different runner bytes")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or any(
        not fact_matches(fact) for fact in manifest.get("entries", [])
    ):
        raise GlobalityAuditError("retained manifest is incomplete or stale")
    return result


def inventory(output: Path, manifest_path: Path) -> list[dict[str, object]]:
    return [
        file_fact(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path
    ]


def run(output: Path, field_path: Path, resume_from: Path) -> dict[str, object]:
    if output.resolve() != OUTPUT.resolve():
        raise GlobalityAuditError(f"output must be exactly {OUTPUT}")
    if resume_from.resolve() != output.resolve():
        raise GlobalityAuditError("--resume-from must name the output stage-checkpoint root")
    started = time.time()
    source = validate_source(field_path)
    output.mkdir(parents=True, exist_ok=True)
    prior = completed_result(output, source)
    if prior is not None:
        return prior
    storage = storage_preflight(output)
    runner = file_fact(Path(__file__).resolve())
    target = np.memmap(field_path, mode="r", dtype=np.uint8, shape=FIELD_SHAPE)

    hist_path = output / "retained/shifted_interior_histograms.i32le"
    repeat_path = output / "retained/shifted_interior_histograms.repeat.i32le"
    translations_path = output / "retained/ordered_translations.i16le"
    histogram_checkpoint_path = output / "stage_checkpoints/01_histograms_complete.json"
    if histogram_checkpoint_path.is_file():
        checkpoint = json.loads(histogram_checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("schema") != SCHEMA or any(
            not fact_matches(checkpoint.get(key))
            for key in ("histograms", "histograms_repeat", "translations")
        ):
            raise GlobalityAuditError("histogram checkpoint is incomplete or stale")
        if checkpoint.get("runner") != runner or checkpoint.get("source_field") != source:
            raise GlobalityAuditError("histogram checkpoint belongs to different inputs")
        histogram_fact = checkpoint["histograms"]
        repeat_fact = checkpoint["histograms_repeat"]
        translations_fact = checkpoint["translations"]
        shift_count = (2 * SEARCH_RADIUS + 1) ** 2
        histograms = np.fromfile(hist_path, dtype="<i4").reshape(
            N_PAIRS, shift_count, NUM_CLASSES
        )
        repeated = np.fromfile(repeat_path, dtype="<i4").reshape(
            N_PAIRS, shift_count, NUM_CLASSES
        )
        translations = np.fromfile(translations_path, dtype="<i2").reshape(shift_count, 2)
    else:
        histograms, translations = shifted_interior_histograms(
            target, radius=SEARCH_RADIUS
        )
        repeated, repeated_translations = shifted_interior_histograms(
            target, radius=SEARCH_RADIUS
        )
        if not np.array_equal(translations, repeated_translations):
            raise GlobalityAuditError("determinism repeat changed translation order")
        if not np.array_equal(histograms, repeated):
            raise GlobalityAuditError("determinism repeat changed histogram bytes")
        histogram_fact = atomic_array(hist_path, histograms)
        repeat_fact = atomic_array(repeat_path, repeated)
        translations_fact = atomic_array(translations_path, translations)
        atomic_json(
            histogram_checkpoint_path,
            {
                "schema": SCHEMA,
                "source_field": source,
                "runner": runner,
                "histograms": histogram_fact,
                "histograms_repeat": repeat_fact,
                "translations": translations_fact,
                "byte_identical_repeat": histogram_fact["sha256"] == repeat_fact["sha256"],
                "interior_shape": [
                    HEIGHT - 2 * SEARCH_RADIUS,
                    WIDTH - 2 * SEARCH_RADIUS,
                ],
            },
        )

    bounds = choose_and_bound_pairing(histograms, translations)
    repeat_bounds = choose_and_bound_pairing(repeated, translations)
    if bounds != repeat_bounds:
        raise GlobalityAuditError("determinism repeat changed the certified lower bound")
    bounds_fact = atomic_json(output / "retained/pairing_lower_bounds.json", bounds)
    bounds_repeat_fact = atomic_json(
        output / "retained/pairing_lower_bounds.repeat.json", repeat_bounds
    )
    certified = int(bounds["certified_global_mismatch_lower_bound"])
    decision = (
        "CEILING-REFUSED"
        if certified > MAX_CEILING_COMPATIBLE_MISMATCHES
        else "CEILING-INCONCLUSIVE"
    )
    bound_checkpoint = {
        "schema": SCHEMA,
        "source_field": source,
        "runner": runner,
        "pairing_lower_bounds": bounds_fact,
        "pairing_lower_bounds_repeat": bounds_repeat_fact,
        "byte_identical_repeat": bounds_fact["sha256"] == bounds_repeat_fact["sha256"],
        "certified_global_mismatch_lower_bound": certified,
        "maximum_ceiling_compatible_mismatches": MAX_CEILING_COMPATIBLE_MISMATCHES,
        "decision": decision,
    }
    atomic_json(output / "stage_checkpoints/02_global_bound_complete.json", bound_checkpoint)

    rss = peak_rss_bytes()
    if rss > 20_000_000_000:
        raise GlobalityAuditError(f"peak RSS {rss} exceeded the 20 GB charter ceiling")
    result = {
        "schema": SCHEMA,
        "axis": AXIS,
        "source_field": source,
        "decision": decision,
        "verdict_scope": (
            "FORMULATION: one categorical field shared by all 600 frames, with arbitrary "
            "per-frame integer translations dy,dx in [-12,+12] and any residual charged "
            "at the charter's optimistic 0.2909 B/site arithmetic"
        ),
        "theorem": {
            "common_interior": [
                SEARCH_RADIUS,
                HEIGHT - SEARCH_RADIUS,
                SEARCH_RADIUS,
                WIDTH - SEARCH_RADIUS,
            ],
            "common_interior_sites_per_frame": (
                (HEIGHT - 2 * SEARCH_RADIUS) * (WIDTH - 2 * SEARCH_RADIUS)
            ),
            "statement": (
                "For paired frames i,j and any shared categorical field S and allowed "
                "translations a,b, errors_i+errors_j on the common interior are at "
                "least TV(hist(T_i shifted by a), hist(T_j shifted by b)). The exact "
                "minimum over all 625x625 shift pairs is summed over 300 disjoint pairs."
            ),
            "selection_validity": (
                "Every candidate is a disjoint pairing; selecting one from exact retained "
                "histograms does not weaken the inequality because it holds for every pairing."
            ),
            "excluded_errors": "all border errors are ignored, so the result is a lower bound",
        },
        "histogram_payloads": {
            "primary": histogram_fact,
            "repeat": repeat_fact,
            "translations": translations_fact,
            "byte_identical_repeat": histogram_fact["sha256"] == repeat_fact["sha256"],
        },
        "pairing_payloads": {
            "primary": bounds_fact,
            "repeat": bounds_repeat_fact,
            "byte_identical_repeat": bounds_fact["sha256"] == bounds_repeat_fact["sha256"],
        },
        "certified_global_mismatch_lower_bound": certified,
        "maximum_ceiling_compatible_mismatches": MAX_CEILING_COMPATIBLE_MISMATCHES,
        "certified_excess_mismatches": certified - MAX_CEILING_COMPATIBLE_MISMATCHES,
        "global_optimum_claim": False,
        "interpretation": (
            "The exact optimum mismatch count was not solved; the retained bound is sufficient "
            "to prove that no member of the declared rigid-static family can meet the ceiling."
        ),
        "constraints": {
            "scorer_invoked": False,
            "modal_invoked": False,
            "metal_or_mps_invoked": False,
            "upstream_modified": False,
            "submission_modified": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        "resource": {
            "elapsed_seconds": time.time() - started,
            "peak_rss_bytes": rss,
            "peak_rss_limit_bytes": 20_000_000_000,
            "within_peak_rss_limit": True,
        },
        "provenance": {
            "runner": runner,
            "git_head_before_commit": git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "argv": sys.argv,
            "resume_from": str(resume_from),
            "storage_preflight": storage,
        },
    }
    atomic_json(output / "RESULT.json", result)
    manifest_path = output / "MANIFEST.json"
    entries = inventory(output, manifest_path)
    atomic_json(
        manifest_path,
        {
            "schema": MANIFEST_SCHEMA,
            "entry_count": len(entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "entries": entries,
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.output, args.field, args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
