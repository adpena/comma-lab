#!/usr/bin/env python3
"""Adjudicate GF3 GOP fields with GF2's certified histogram-TV theorem.

The physical GF3 runner retains one achievable field, its global offsets, and
its real-coded correction stream for every chartered GOP length.  Its first
cellwise lower bound is valid but deliberately grants a different translation
to every lattice cell, so it is too weak for the family verdict.  This audit
supersedes that proof surface without overwriting its evidence.

Within every GOP, each candidate partitions the frames into disjoint pairs.
For each pair, the exact minimum class-histogram total variation over all
625x625 allowed translation pairs lower-bounds the two frames' combined errors
against any shared categorical field.  Summing disjoint pairs is valid, and
taking the maximum across several valid pairings remains a valid lower bound.
No fitted field, scorer, trainer, Metal/MPS path, or Modal service is invoked.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_gf2_alignment_globality_audit as gf2
from experiments import ddm_gf3_time_conditioned_factorized_field_pricing as gf3
from experiments import ddm_up2_shipping_pose_solve as up2

SCHEMA: Final = "ddm_gf3_gop_histogram_globality_audit.v1"
MANIFEST_SCHEMA: Final = "ddm_gf3_gop_histogram_globality_manifest.v1"
AXIS: Final = "[macOS-CPU scorer-free exact field measurement, n600]"
OUTPUT: Final = gf3.OUTPUT / "certified_globality_v2"
PHYSICAL_ROOT: Final = gf3.OUTPUT
GF2_AUDIT_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/globality_audit_v1")
GF2_RESULT: Final = GF2_AUDIT_ROOT / "RESULT.json"
HISTOGRAM_SHA256: Final = "6c13293602abc5868f2198da4928111e64b56fd0b7dad51ff27474cdcc106818"
TRANSLATIONS_SHA256: Final = "6b28d52741726bc162e96f66932e54c2ef578b5db9614d4c7fb616287a3bce51"
REPAIR_RATE_NUMERATOR: Final = 2_909
REPAIR_RATE_DENOMINATOR: Final = 10_000


class GF3GlobalityError(RuntimeError):
    """A theorem input, retained-row identity, or proof invariant failed."""


def peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(observed if sys.platform == "darwin" else observed * 1024)


def candidate_pairings(frame_ids: Sequence[int], center_histograms: np.ndarray) -> dict[str, list[list[int]]]:
    """Return deterministic disjoint pairings, leaving one odd frame free."""

    ids = [int(frame) for frame in frame_ids]
    half = len(ids) // 2
    if not ids or center_histograms.ndim != 2 or center_histograms.shape[1] != gf3.NUM_CLASSES:
        raise GF3GlobalityError("candidate pairing inputs have invalid geometry")
    if any(frame < 0 or frame >= center_histograms.shape[0] for frame in ids):
        raise GF3GlobalityError("candidate pairing contains an invalid frame id")
    pairings = {"fixed_extremes": [[left, right] for left, right in zip(ids[:half], ids[-half:], strict=True)]}
    for label in range(gf3.NUM_CLASSES):
        order = sorted(ids, key=lambda frame: (int(center_histograms[frame, label]), frame))
        pairings[f"extreme_center_class_{label}"] = [
            [left, right] for left, right in zip(order[:half], reversed(order[-half:]), strict=True)
        ]
    for name, pairs in pairings.items():
        flat = [frame for pair in pairs for frame in pair]
        if len(pairs) != half or len(flat) != len(set(flat)) or not set(flat).issubset(ids):
            raise GF3GlobalityError(f"{name} is not a disjoint within-GOP pairing")
    return pairings


def exact_pairing_row(histograms: np.ndarray, pairs: Sequence[Sequence[int]]) -> dict[str, object]:
    per_pair = []
    for pair in pairs:
        if len(pair) != 2:
            raise GF3GlobalityError("pairing row does not contain two frame ids")
        left, right = (int(pair[0]), int(pair[1]))
        per_pair.append(
            {
                "frames": [left, right],
                "minimum_histogram_tv": gf2.minimum_histogram_tv(histograms[left], histograms[right]),
            }
        )
    return {
        "pairs": per_pair,
        "lower_bound": sum(int(row["minimum_histogram_tv"]) for row in per_pair),
    }


def exact_gop_certificate(histograms: np.ndarray, translations: np.ndarray, gop_length: int) -> dict[str, object]:
    if histograms.shape != (gf3.N_PAIRS, translations.shape[0], gf3.NUM_CLASSES):
        raise GF3GlobalityError("histogram geometry differs from the exact n600 theorem input")
    if gop_length <= 0 or gf3.N_PAIRS % gop_length:
        raise GF3GlobalityError("GOP length must be positive and divide the exact frame count")
    zero_rows = np.flatnonzero(np.all(translations == 0, axis=1))
    if zero_rows.size != 1:
        raise GF3GlobalityError("translation table lacks exactly one zero shift")
    center = histograms[:, int(zero_rows[0]), :]
    gop_rows = []
    for start in range(0, gf3.N_PAIRS, gop_length):
        frame_ids = list(range(start, start + gop_length))
        candidates = {
            name: exact_pairing_row(histograms, pairs) for name, pairs in candidate_pairings(frame_ids, center).items()
        }
        selected_name = max(candidates, key=lambda name: (int(candidates[name]["lower_bound"]), name))
        gop_rows.append(
            {
                "start_frame": start,
                "stop_frame_exclusive": start + gop_length,
                "candidate_pairings": candidates,
                "selected_pairing": selected_name,
                "selected_lower_bound": int(candidates[selected_name]["lower_bound"]),
            }
        )
    return {
        "gop_length": gop_length,
        "gop_count": gf3.N_PAIRS // gop_length,
        "paired_frames_per_gop": 2 * (gop_length // 2),
        "unpaired_frames_per_gop": gop_length % 2,
        "gops": gop_rows,
        "certified_lower_bound_mismatches": sum(int(row["selected_lower_bound"]) for row in gop_rows),
    }


def validate_gf2_inputs() -> tuple[np.memmap, np.memmap, np.ndarray, dict[str, object]]:
    result = json.loads(GF2_RESULT.read_text(encoding="utf-8"))
    if result.get("source_field", {}).get("sha256") != gf3.FIELD_SHA256:
        raise GF3GlobalityError("GF2 theorem result belongs to a different exact field")
    payloads = result["histogram_payloads"]
    for key in ("primary", "repeat", "translations"):
        if not gf3.fact_matches(payloads[key]):
            raise GF3GlobalityError(f"GF2 {key} theorem payload failed custody validation")
    if (
        payloads["primary"]["sha256"] != HISTOGRAM_SHA256
        or payloads["repeat"]["sha256"] != HISTOGRAM_SHA256
        or payloads["translations"]["sha256"] != TRANSLATIONS_SHA256
    ):
        raise GF3GlobalityError("GF2 theorem payload identity differs from the recalled source")
    shift_count = (2 * gf3.SEARCH_RADIUS + 1) ** 2
    primary = np.memmap(
        payloads["primary"]["path"],
        mode="r",
        dtype="<i4",
        shape=(gf3.N_PAIRS, shift_count, gf3.NUM_CLASSES),
    )
    repeated = np.memmap(
        payloads["repeat"]["path"],
        mode="r",
        dtype="<i4",
        shape=(gf3.N_PAIRS, shift_count, gf3.NUM_CLASSES),
    )
    translations = np.fromfile(payloads["translations"]["path"], dtype="<i2").reshape(shift_count, 2)
    if not np.array_equal(translations, gf2.ordered_translations(gf3.SEARCH_RADIUS)):
        raise GF3GlobalityError("GF2 translation table differs from the declared [-12,+12]^2 grid")
    expected_sites = (gf3.HEIGHT - 2 * gf3.SEARCH_RADIUS) * (gf3.WIDTH - 2 * gf3.SEARCH_RADIUS)
    if not np.all(primary.sum(axis=2, dtype=np.int64) == expected_sites):
        raise GF3GlobalityError("GF2 primary histogram totals differ from the common interior")
    if not np.all(repeated.sum(axis=2, dtype=np.int64) == expected_sites):
        raise GF3GlobalityError("GF2 repeat histogram totals differ from the common interior")
    return primary, repeated, translations, result


def validate_physical_row(gop_length: int) -> tuple[dict[str, object], dict[str, object]]:
    result_path = PHYSICAL_ROOT / f"L_{gop_length:03d}/RESULT.json"
    manifest_path = PHYSICAL_ROOT / f"L_{gop_length:03d}/MANIFEST.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise GF3GlobalityError(f"physical L={gop_length} row is incomplete")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    physical_runner = result.get("provenance", {}).get("runner")
    if (
        result.get("schema") != gf3.SCHEMA
        or result.get("gop_length") != gop_length
        or result.get("source_field", {}).get("sha256") != gf3.FIELD_SHA256
        or result.get("constraints", {}).get("all_materialized_payloads_retained") is not True
        or result.get("packet", {}).get("fields_and_offsets_parseback_exact") is not True
        or result.get("domain_matched_residual_to_target", {}).get("receiver_verified") is not True
        or not gf3.fact_matches(physical_runner)
        or physical_runner != gf3.file_fact(Path(gf3.__file__).resolve())
    ):
        raise GF3GlobalityError(f"physical L={gop_length} row failed its receiver contract")
    if manifest.get("schema") != gf3.MANIFEST_SCHEMA or any(
        not gf3.fact_matches(fact) for fact in manifest.get("entries", [])
    ):
        raise GF3GlobalityError(f"physical L={gop_length} manifest failed rehash")
    return result, {
        "result": gf3.file_fact(result_path),
        "manifest": gf3.file_fact(manifest_path),
        "entry_count": manifest["entry_count"],
        "total_bytes": manifest["total_bytes"],
    }


def price_row(certificate: Mapping[str, object], physical: Mapping[str, object]) -> dict[str, object]:
    lower_bound = int(certificate["certified_lower_bound_mismatches"])
    packet_bytes = int(physical["pricing"]["packet_bytes"])
    observed_residual_bytes = int(physical["pricing"]["residual_bytes"])
    excess = max(0, lower_bound - gf3.MISMATCH_TARGET)
    charge_numerator = excess * REPAIR_RATE_NUMERATOR
    optimistic_charge = charge_numerator / REPAIR_RATE_DENOMINATOR
    direct_admits = lower_bound <= gf3.MISMATCH_TARGET and packet_bytes <= gf3.PACKET_CAP_BYTES
    replacement_admits = packet_bytes + optimistic_charge <= gf3.REPLACEMENT_CAP_BYTES
    disposition = "BOUND-ADMITS-BUILD" if direct_admits or replacement_admits else "PRICING-REFUSED"
    return {
        "gop_length": int(certificate["gop_length"]),
        "certified_histogram_tv_lower_bound_mismatches": lower_bound,
        "bound_denominator": gf3.N_PAIRS * (gf3.HEIGHT - 2 * gf3.SEARCH_RADIUS) * (gf3.WIDTH - 2 * gf3.SEARCH_RADIUS),
        "physical_packet_bytes": packet_bytes,
        "observed_fit_mismatches": int(physical["observed_fit_upper_bound"]["post_sweep_mismatches"]),
        "observed_fit_minus_bound": int(physical["observed_fit_upper_bound"]["post_sweep_mismatches"]) - lower_bound,
        "physical_domain_residual_bytes_to_46804": observed_residual_bytes,
        "physical_packet_plus_residual_bytes": packet_bytes + observed_residual_bytes,
        "charter_optimistic_repair_charge": {
            "excess_mismatches_above_46804": excess,
            "rate_bytes_per_site": "0.2909",
            "exact_numerator": charge_numerator,
            "exact_denominator": REPAIR_RATE_DENOMINATOR,
            "bytes": optimistic_charge,
            "packet_plus_charge_bytes": packet_bytes + optimistic_charge,
            "physical_coder_claim": False,
        },
        "direct_bound_gate_passes": direct_admits,
        "replacement_charge_gate_passes": replacement_admits,
        "disposition": disposition,
    }


def inventory(root: Path, manifest_path: Path) -> list[dict[str, object]]:
    return [gf3.file_fact(path) for path in sorted(root.rglob("*")) if path.is_file() and path != manifest_path]


def run(output: Path, resume_from: Path) -> dict[str, object]:
    if output.resolve() != OUTPUT.resolve() or resume_from.resolve() != output.resolve():
        raise GF3GlobalityError(f"--output and --resume-from must both equal {OUTPUT}")
    result_path = output / "RESULT.json"
    manifest_path = output / "MANIFEST.json"
    runner = gf3.file_fact(Path(__file__).resolve())
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("schema") != SCHEMA or result.get("provenance", {}).get("runner") != runner:
            raise GF3GlobalityError("completed audit result is stale")
        gf3.atomic_json_once(
            output / "stage_checkpoints/02_final_result_complete.json",
            {"schema": SCHEMA, "result": gf3.file_fact(result_path), "runner": runner},
        )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            entries = inventory(output, manifest_path)
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "entry_count": len(entries),
                "total_bytes": sum(int(entry["bytes"]) for entry in entries),
                "entries": entries,
            }
            gf3.atomic_json_once(manifest_path, manifest)
        if manifest.get("schema") != MANIFEST_SCHEMA or any(
            not gf3.fact_matches(fact) for fact in manifest.get("entries", [])
        ):
            raise GF3GlobalityError("completed audit manifest is stale")
        return result
    if manifest_path.exists():
        raise GF3GlobalityError("audit manifest exists without its result")

    started = time.time()
    storage = gf3.storage_preflight(output, label="globality_v2")
    source = gf3.validate_source(gf3.DEFAULT_FIELD)
    lineage = up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
    histograms, repeated, translations, gf2_result = validate_gf2_inputs()
    rows = []
    for gop_length in gf3.GOP_LENGTHS:
        stage = output / f"L_{gop_length:03d}"
        certificate = exact_gop_certificate(histograms, translations, gop_length)
        repeat = exact_gop_certificate(repeated, translations, gop_length)
        if certificate != repeat:
            raise GF3GlobalityError(f"L={gop_length} independent theorem repeat differs")
        primary_fact = gf3.atomic_json_once(stage / "lower_bound_certificate.json", certificate)
        repeat_fact = gf3.atomic_json_once(stage / "lower_bound_certificate.repeat.json", repeat)
        if primary_fact["sha256"] != repeat_fact["sha256"]:
            raise GF3GlobalityError(f"L={gop_length} certificate bytes differ across repeats")
        physical, physical_custody = validate_physical_row(gop_length)
        row = price_row(certificate, physical)
        row.update(
            {
                "certificate": primary_fact,
                "certificate_repeat": repeat_fact,
                "certificate_byte_identical_repeat": True,
                "physical_row_custody": physical_custody,
            }
        )
        gf3.atomic_json_once(stage / "RESULT.json", row)
        gf3.atomic_json_once(
            output / f"stage_checkpoints/01_L_{gop_length:03d}_complete.json",
            {
                "schema": SCHEMA,
                "gop_length": gop_length,
                "certificate": primary_fact,
                "certificate_repeat": repeat_fact,
                "physical_row": physical_custody,
                "runner": runner,
            },
        )
        rows.append(row)

    admitting = [row for row in rows if row["disposition"] == "BOUND-ADMITS-BUILD"]
    rss = peak_rss_bytes()
    if rss > gf3.PEAK_RSS_LIMIT_BYTES:
        raise GF3GlobalityError(f"peak RSS {rss} exceeds {gf3.PEAK_RSS_LIMIT_BYTES}")
    result = {
        "schema": SCHEMA,
        "axis": AXIS,
        "source_field": source,
        "lineage_gate": lineage,
        "rows": rows,
        "decision": {
            "disposition": "BOUND-ADMITS-BUILD" if admitting else "PRICING-CLOSED",
            "admitting_gop_lengths": [row["gop_length"] for row in admitting],
            "verdict_scope": (
                "FORMULATION: the discrete L={2,3,5,10,20,50} family of one full categorical field per "
                "GOP plus one global integer translation in [-12,+12]^2 per frame, with generic repairs "
                "charged at the charter's optimistic 0.2909 B/site comparison rate"
            ),
            "not_closed": (
                "time-conditioned low-rank factors, non-rigid warps, parametric boundary atoms, learned "
                "residual probability models, and any GOP length outside the chartered grid"
            ),
            "scorer_invoked": False,
            "trainer_built": False,
            "pointer_moved": False,
        },
        "theorem": {
            "statement": (
                "For each disjoint frame pair inside a GOP, combined shared-field errors are at least the "
                "minimum class-histogram TV over all 625x625 allowed translation pairs. Pair bounds sum "
                "within a disjoint pairing; the maximum of six valid pairings is retained per GOP."
            ),
            "excluded_errors": "all border errors and one frame in every odd-length GOP are discarded",
            "global_optimum_claim": False,
            "gf2_registered_ancestor": gf3.file_fact(GF2_RESULT),
            "gf2_certified_global_bound": gf2_result["certified_global_mismatch_lower_bound"],
        },
        "prior_law": {
            "prediction_L10_at_least_400000": False,
            "measured_L10_bound": next(
                row["certified_histogram_tv_lower_bound_mismatches"] for row in rows if row["gop_length"] == 10
            ),
            "qualitative_rate_accuracy_tradeoff": "confirmed on the chartered grid",
        },
        "storage_preflight": storage,
        "resource": {
            "elapsed_seconds": time.time() - started,
            "peak_rss_bytes": rss,
            "peak_rss_limit_bytes": gf3.PEAK_RSS_LIMIT_BYTES,
        },
        "provenance": {
            "runner": runner,
            "gf3_physical_runner": gf3.file_fact(Path(gf3.__file__).resolve()),
            "git_head_before_commit": gf3.git_head(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "argv": sys.argv,
            "seed": None,
            "determinism": "exact integer histogram arithmetic; no RNG",
            "execution_env": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "constraints": {
            "all_materialized_payloads_retained": True,
            "scorer_invoked": False,
            "training_invoked": False,
            "modal_invoked": False,
            "metal_or_mps_invoked": False,
            "upstream_modified": False,
            "submission_modified": False,
            "score_claim": False,
            "pointer_moved": False,
        },
    }
    gf3.atomic_json_once(result_path, result)
    gf3.atomic_json_once(
        output / "stage_checkpoints/02_final_result_complete.json",
        {"schema": SCHEMA, "result": gf3.file_fact(result_path), "runner": runner},
    )
    entries = inventory(output, manifest_path)
    gf3.atomic_json_once(
        manifest_path,
        {
            "schema": MANIFEST_SCHEMA,
            "entry_count": len(entries),
            "total_bytes": sum(int(entry["bytes"]) for entry in entries),
            "entries": entries,
        },
    )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run(args.output, args.resume_from), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
