#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations.gap_decomposition_against_floor_20260802 import (  # noqa: E402
    GapDecomposition,
    MeasuredScoreTriple,
    demonstrated_floor_pr130,
    seg_rate_exchange_bytes_per_flip,
)
from tac.derived_upstream_refresh import (  # noqa: E402
    BASE_SHA_FORMULA_INVARIANT,
    BASE_SHA_PRESENT,
    DISPOSITION_CURRENT,
    DISPOSITION_EXACT_INVARIANT,
    DISPOSITION_QUEUED_FIBER_INPUT_BLOCKED,
    DISPOSITION_QUEUED_HEAVY_REFRESH,
    DISPOSITION_REFRESHED_SCORER_FREE,
    ROUTE_ALREADY_CURRENT,
    ROUTE_EXACT_INVARIANT,
    ROUTE_FIBER_TRANSPORT,
    ROUTE_FULL_RECOMPUTE,
    ROUTE_SCORER_FREE_DERIVATION,
    RefreshRegistryError,
    RefreshRegistryRow,
    registry_denominators,
    require_fresh_for_consumption,
    write_refresh_registry_jsonl,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / ".omx" / "research" / "ddm_uf1_20260805"
RATE_DENOMINATOR_BYTES = 37_545_489

QO1_ARCHIVE_SHA256 = "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a"
QO1_ARCHIVE_BYTES = 357_836
QO1_D_SEG = 0.00431179
QO1_D_POSE = 0.00071459
QO1_AXIS = "[macOS-CPU advisory]"
QO1_SOURCE = ".omx/research/ddm_sb1_20260804/sb1_rows.jsonl:1"

FZ4_ARCHIVE_SHA256 = "ad5dd0e4fbe5b13ab53a5995a6d77cc558c25f40b63f894ea50ad336bd50fb66"

PF2_RECEIPT = (
    ".omx/research/ddm_pf2_dimension_conditioned_two_type_20260724T020205Z/"
    "ddm_pf2_dimension_conditioned_two_type_receipt.json"
)
MS_RECEIPTS = (
    ".omx/research/ddm_ms3_metric_custody_bundle_20260724T035249Z/BUNDLE-PARTIAL.json",
    ".omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/BUNDLE-COMPLETE.json",
    ".omx/research/ddm_ms6_receiver_support_measurement_20260724T052034Z/"
    "ddm_ms6_receiver_support_measurement_receipt.json",
)
G2_RECEIPT = ".omx/research/ddm_g2_solve_diff_op_mining_n600_20260722T194000Z/receipt.json"
G3_RECEIPT = ".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_receipt.json"
G4_RECEIPT = (
    ".omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z/"
    "ddm_g4_spatial_stationarity_receipt.json"
)
R9M_RECEIPT = ".omx/research/ddm_r9m_first_contest_cpu_row_20260804/FIRST_OWN_VEHICLE_CONTEST_CPU_ROW_20260804.md"
NG1_RECEIPT = ".omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md"
FT1_MODULE = "src/tac/canonical_equations/ddm_ft1_photometric_beta_commutator_20260802.py"
MARGIN_SALIENCY_SOURCES = (
    "src/tac/margin_saliency_map.py",
    "src/tac/logit_margin_sensitivity_weighted.py",
    "src/tac/learnable_saliency_threshold.py",
)
SENSITIVITY_BITALLOC_SOURCES = (
    "src/tac/witness_sensitivity_bitalloc.py",
    "src/tac/frontier_exact_bitalloc.py",
    "src/tac/canonical_duckdb/per_byte_sensitivity_ext.py",
)
G2F_TRUST_REGION_SOURCES = (
    G2_RECEIPT,
    "src/tac/scorer_surrogate/costate_trust_region.py",
    "src/tac/master_gradient_trust_region.py",
)


@dataclass(frozen=True)
class WrittenOutputs:
    output_dir: Path
    registry_path: Path
    summary_path: Path
    m66_path: Path
    queued_path: Path
    transport_path: Path
    receipt_path: Path
    next_path: Path


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _sha256_file(path: str | Path) -> str:
    p = _resolve(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundle_sha256(paths: tuple[str, ...]) -> str:
    members: list[dict[str, str]] = []
    for path in paths:
        members.append({"path": path, "sha256": _sha256_file(path)})
    payload = json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _formula_sha256(label: str, payload: dict[str, Any]) -> str:
    body = json.dumps({"label": label, **payload}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _receipt_row(
    *,
    quantity_id: str,
    description: str,
    evidence_paths: tuple[str, ...],
    consumers: tuple[str, ...],
    trigger: str,
    owner: str,
    notes: str,
    validity_radius_status: str = "UNKNOWN",
    validity_radius_derive_route: str = "derive from same-base n600 refresh before use",
) -> RefreshRegistryRow:
    return RefreshRegistryRow(
        quantity_id=quantity_id,
        description=description,
        base_identity_kind="source_receipt_bundle_sha256",
        computed_at_base_sha256=_bundle_sha256(evidence_paths),
        base_sha_status=BASE_SHA_PRESENT,
        base_age="STALE_BY_CADENCE_OR_BASE_MOVE",
        current_base_sha256=_bundle_sha256(evidence_paths),
        consumers=consumers,
        validity_radius_status=validity_radius_status,
        validity_radius_derive_route=validity_radius_derive_route,
        refresh_route=ROUTE_FULL_RECOMPUTE,
        trigger=trigger,
        owner=owner,
        disposition=DISPOSITION_QUEUED_HEAVY_REFRESH,
        evidence_paths=evidence_paths,
        score_claim=False,
        promotion_eligible=False,
        notes=notes,
    )


def build_refresh_rows() -> tuple[RefreshRegistryRow, ...]:
    w_sha = _formula_sha256(
        "W_bytes_per_argmax_flip",
        {
            "rate_denominator_bytes": RATE_DENOMINATOR_BYTES,
            "scored_pairs": 600,
            "height": 384,
            "width": 512,
            "formula": "4*DEN/(600*512*384)",
        },
    )
    rows = [
        _receipt_row(
            quantity_id="pf2_1200_row_atlas",
            description="PF2 1200-row dimension-conditioned atlas",
            evidence_paths=(PF2_RECEIPT,),
            consumers=("tools/tac.bit_allocator", "tac.cathedral_autopilot"),
            trigger="new qo1 live base or cadence use before bit-allocation launch",
            owner="queue:atlas_refresh_on_scorer_slot",
            notes="Heavy atlas refresh queued; no scorer run in UF1.",
        ),
        _receipt_row(
            quantity_id="ms3_ms6_metric_bundle",
            description="MS3-MS6 metric custody bundle and receiver support measurements",
            evidence_paths=MS_RECEIPTS,
            consumers=(
                "tac.optimization.ddm_metric_custody_bundle:load_metric_custody_bundle",
                "tac.optimization.ddm_min_description_contract:build_minimum_description_headline",
            ),
            trigger="consumer attempts metric admission after qo1 pointer move",
            owner="queue:metric_bundle_refresh",
            notes="Full metric recompute is scorer/atlas heavy and remains queued behind sq2.",
        ),
        _receipt_row(
            quantity_id="margin_saliency_maps_141",
            description="#141 margin and saliency map surfaces",
            evidence_paths=MARGIN_SALIENCY_SOURCES,
            consumers=("margin adaptive map builders", "score-aware loss planners"),
            trigger="candidate base or scorer target changes before margin-map consumption",
            owner="queue:margin_saliency_refresh",
            notes="Source surface inventoried; real map refresh requires the scorer/GT cache path.",
        ),
        _receipt_row(
            quantity_id="sensitivity_bitalloc_maps_157_336",
            description="#157/#336 sensitivity bit-allocation maps",
            evidence_paths=SENSITIVITY_BITALLOC_SOURCES,
            consumers=("sensitivity weighted bit allocators", "per-byte sensitivity consumers"),
            trigger="bit allocator consumes a stale sensitivity surface after qo1",
            owner="queue:sensitivity_bitalloc_refresh",
            notes="Queued as heavy; do not consume without a same-base receipt.",
        ),
        _receipt_row(
            quantity_id="g3_score_atlas",
            description="G3 score atlas",
            evidence_paths=(G3_RECEIPT,),
            consumers=("tac.optimization.ddm_g4_spatial_stationarity", "stationarity consumers"),
            trigger="new base, new hard-pair registry, or cadence refresh before atlas use",
            owner="queue:g3_atlas_refresh",
            notes="Heavy n600 atlas refresh queued; sq2 owns scorer slot.",
        ),
        _receipt_row(
            quantity_id="g4_stationarity_maps",
            description="G4 spatial stationarity maps",
            evidence_paths=(G4_RECEIPT,),
            consumers=("spatial stationarity consumers", "trajectory/stopping controllers"),
            trigger="new G3 atlas or qo1 base before stationarity admission",
            owner="queue:g4_stationarity_refresh",
            notes="Heavy stationarity refresh queued, not rerun in UF1.",
        ),
        _receipt_row(
            quantity_id="g2f_trust_regions",
            description="G2F trust-region and costate trust-region surfaces",
            evidence_paths=G2F_TRUST_REGION_SOURCES,
            consumers=("trust-region policy", "costate consumers", "trajectory stopping"),
            trigger="trust-region consumer sees a base or atlas mismatch",
            owner="queue:g2f_trust_region_refresh",
            notes="Queued because same-base scorer atlas inputs are required.",
        ),
        RefreshRegistryRow(
            quantity_id="r9m_advisory_to_contest_cpu_calibration_prior",
            description="r9m advisory-to-contest-CPU transfer prior measured on fz4",
            base_identity_kind="archive_zip_sha256",
            computed_at_base_sha256=FZ4_ARCHIVE_SHA256,
            base_sha_status=BASE_SHA_PRESENT,
            base_age="STALE_FOR_QO1_ARCHIVE",
            current_base_sha256=QO1_ARCHIVE_SHA256,
            consumers=("contest-axis projection writers", "candidate promotion triage"),
            validity_radius_status="UNKNOWN",
            validity_radius_derive_route="paired contest-CPU row on the consumed archive",
            refresh_route=ROUTE_FULL_RECOMPUTE,
            trigger="archive sha moved fz4 -> qo1; projection is not consumed unchecked",
            owner="queue:next_modal_contest_cpu_row",
            disposition=DISPOSITION_QUEUED_HEAVY_REFRESH,
            evidence_paths=(R9M_RECEIPT, ".omx/research/ddm_fz3_20260804/fz3_sub_final_eval_receipt.json"),
            score_claim=False,
            promotion_eligible=False,
            notes="UF1 guard refuses qo1 consumption of the fz4-only +6.55e-6 prior.",
        ),
        RefreshRegistryRow(
            quantity_id="W_bytes_per_flip_exchange",
            description="Exact archive-byte exchange rate per SegNet argmax flip",
            base_identity_kind="rate_denominator_formula_sha256",
            computed_at_base_sha256=w_sha,
            base_sha_status=BASE_SHA_FORMULA_INVARIANT,
            base_age="EXACT_INVARIANT_GIVEN_RATE_DENOMINATOR",
            current_base_sha256=w_sha,
            consumers=("gap/pricing tables", "bit allocation break-even calculators"),
            validity_radius_status="NOT_APPLICABLE",
            validity_radius_derive_route="linear formula W=4*DEN/(600*512*384)",
            refresh_route=ROUTE_EXACT_INVARIANT,
            trigger="only upstream/videos denominator or scored lattice changes",
            owner="uf1",
            disposition=DISPOSITION_EXACT_INVARIANT,
            evidence_paths=("upstream/evaluate.py",),
            score_claim=False,
            promotion_eligible=False,
            notes="No archive-size dependence; scorer-free carry-forward.",
        ),
        RefreshRegistryRow(
            quantity_id="m66_gap_decomposition_inputs_qo1",
            description="m66 gap decomposition inputs re-derived against qo1",
            base_identity_kind="archive_zip_sha256",
            computed_at_base_sha256=QO1_ARCHIVE_SHA256,
            base_sha_status=BASE_SHA_PRESENT,
            base_age="CURRENT_QO1",
            current_base_sha256=QO1_ARCHIVE_SHA256,
            consumers=("gap tables", "axis prioritization", "main_hot_state gap line"),
            validity_radius_status="KNOWN",
            validity_radius_derive_route="valid only for exact qo1 components and PR130 floor",
            refresh_route=ROUTE_SCORER_FREE_DERIVATION,
            trigger="own-vehicle pointer archive/components change",
            owner="uf1",
            disposition=DISPOSITION_REFRESHED_SCORER_FREE,
            evidence_paths=(QO1_SOURCE, ".omx/state/canonical_frontier_pointer.json"),
            score_claim=False,
            promotion_eligible=False,
            notes="Executed in UF1 without scorer use.",
        ),
        RefreshRegistryRow(
            quantity_id="prefix_bias_ratios_931",
            description="#931 pose prefix/population ratios re-derived by ng1",
            base_identity_kind="gt_cache_and_pose_trace_sha256",
            computed_at_base_sha256=_bundle_sha256((NG1_RECEIPT,)),
            base_sha_status=BASE_SHA_PRESENT,
            base_age="CURRENT_FOR_GT_CACHE_SHA",
            current_base_sha256=_bundle_sha256((NG1_RECEIPT,)),
            consumers=("negative-verdict regrade", "prefix sample triage", "pose rerun queue"),
            validity_radius_status="KNOWN",
            validity_radius_derive_route="valid for gt_n600 sha and d2 partial trace shas in ng1",
            refresh_route=ROUTE_ALREADY_CURRENT,
            trigger="gt_n600 sha or d2 trace sha changes",
            owner="ng1-resolved; uf1-registry-consumer",
            disposition=DISPOSITION_CURRENT,
            evidence_paths=(NG1_RECEIPT,),
            score_claim=False,
            promotion_eligible=False,
            notes=(
                "n24 2.5354755796492157, n48 2.6401816891545127, "
                "n64 2.647768849998471, n96 4.206770932037033."
            ),
        ),
        RefreshRegistryRow(
            quantity_id="fiber_transport_ab_rows_891",
            description="#891 fiber-transport cheap refresh candidates",
            base_identity_kind="fiber_transport_formula_source_sha256",
            computed_at_base_sha256=_sha256_file(FT1_MODULE),
            base_sha_status=BASE_SHA_PRESENT,
            base_age="FORMULA_CURRENT_INPUTS_MISSING",
            current_base_sha256=_sha256_file(FT1_MODULE),
            consumers=("coordinate-fit refreshers", "stale fit ladder extensions"),
            validity_radius_status="UNKNOWN",
            validity_radius_derive_route="requires stored H_ab and mixed partial inputs",
            refresh_route=ROUTE_FIBER_TRANSPORT,
            trigger="data-complete stale (a,b) row appears; otherwise full recompute",
            owner="queue:fiber_transport_input_custody",
            disposition=DISPOSITION_QUEUED_FIBER_INPUT_BLOCKED,
            evidence_paths=(FT1_MODULE,),
            score_claim=False,
            promotion_eligible=False,
            notes="UF1 found the formula but no data-complete stale row to transport in scope.",
        ),
    ]
    return tuple(rows)


def recompute_m66_gap() -> dict[str, Any]:
    ours = MeasuredScoreTriple(
        d_seg=QO1_D_SEG,
        d_pose=QO1_D_POSE,
        archive_bytes=QO1_ARCHIVE_BYTES,
        rate_denominator_bytes=RATE_DENOMINATOR_BYTES,
        source_artifact=QO1_SOURCE,
        axis_tag=QO1_AXIS,
    )
    floor = demonstrated_floor_pr130(RATE_DENOMINATOR_BYTES)
    gap = GapDecomposition(ours=ours, floor=floor)
    w = seg_rate_exchange_bytes_per_flip(RATE_DENOMINATOR_BYTES)
    return {
        "schema": "ddm_uf1_m66_gap_refresh.v1",
        "axis": QO1_AXIS,
        "n_samples": 600,
        "source_artifact": QO1_SOURCE,
        "archive_sha256": QO1_ARCHIVE_SHA256,
        "ours": {
            "d_seg": ours.d_seg,
            "d_pose": ours.d_pose,
            "archive_bytes": ours.archive_bytes,
            "rate_denominator_bytes": ours.rate_denominator_bytes,
            "seg_contribution": ours.seg_contribution,
            "pose_contribution": ours.pose_contribution,
            "rate_contribution": ours.rate_contribution,
            "S": ours.total,
        },
        "floor": {
            "source_artifact": floor.source_artifact,
            "axis": floor.axis_tag,
            "d_seg": floor.d_seg,
            "d_pose": floor.d_pose,
            "archive_bytes": floor.archive_bytes,
            "S": floor.total,
        },
        "gap": {
            "seg": gap.seg_gap,
            "pose": gap.pose_gap,
            "rate": gap.rate_gap,
            "total": gap.total_gap,
            "shares": gap.shares(),
            "rank_by_gap": list(gap.rank_by_gap()),
            "bytes_per_percent_of_gap": gap.bytes_per_percent_of_gap(),
            "W_bytes_per_argmax_flip": w.value,
        },
        "cross_axis_warning": gap.cross_axis_warning(),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ensure_no_tmp_citation(output_dir: Path) -> None:
    transient_token = "/" + "tmp"
    offenders: list[str] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        if transient_token in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(output_dir)))
    if offenders:
        raise RefreshRegistryError(
            f"persisted UF1 evidence cites {transient_token}: {offenders}"
        )


def _guard_receipt(rows: tuple[RefreshRegistryRow, ...]) -> dict[str, Any]:
    row_by_id = {row.quantity_id: row for row in rows}
    refused: dict[str, str] = {}
    for quantity_id, consumer in (
        ("r9m_advisory_to_contest_cpu_calibration_prior", "contest-axis projection writers"),
        ("m66_gap_decomposition_inputs_qo1", "gap tables"),
        ("W_bytes_per_flip_exchange", "gap/pricing tables"),
    ):
        try:
            require_fresh_for_consumption(
                row_by_id[quantity_id],
                current_base_sha256=QO1_ARCHIVE_SHA256,
                consumer=consumer,
            )
        except RefreshRegistryError as exc:
            refused[quantity_id] = str(exc)
    return {
        "schema": "ddm_uf1_freshness_guard_receipt.v1",
        "current_qo1_archive_sha256": QO1_ARCHIVE_SHA256,
        "consumption_refusals": refused,
        "passed_current_or_invariant": [
            "m66_gap_decomposition_inputs_qo1",
            "W_bytes_per_flip_exchange",
        ],
        "score_claim": False,
        "promotion_eligible": False,
    }


def _receipt_markdown(summary: dict[str, int], m66: dict[str, Any], guard: dict[str, Any]) -> str:
    shares = m66["gap"]["shares"]
    refusal = guard["consumption_refusals"].get(
        "r9m_advisory_to_contest_cpu_calibration_prior", "NOT_REFUSED"
    )
    return f"""# ddm_uf1 refresh registry receipt

Date: 2026-08-05. Arm: `uf1`. Scorer use: none. Protected files touched: none.

## Denominators

- quantities found: {summary["quantities_found"]}
- with consumers: {summary["with_consumers"]}
- with triggers: {summary["with_triggers"]}
- with known validity radius: {summary["with_known_validity_radius"]}
- scorer-free refreshed rows: {summary["scorer_free_refreshed"]}
- exact invariant rows: {summary["exact_invariants"]}
- heavy refreshes queued: {summary["heavy_refreshes_queued"]}
- fiber input blockers queued: {summary["fiber_input_blockers_queued"]}

Typed registry: `refresh_registry.jsonl`.

## Executed scorer-free refresh

m66 gap decomposition was re-derived against qo1, using `{QO1_SOURCE}` and archive sha
`{QO1_ARCHIVE_SHA256}`.

| component | value |
|---|---:|
| qo1 S | {m66["ours"]["S"]:.16f} |
| PR130 floor S | {m66["floor"]["S"]:.16f} |
| total gap | {m66["gap"]["total"]:.16f} |
| seg gap | {m66["gap"]["seg"]:.16f} |
| pose gap | {m66["gap"]["pose"]:.16f} |
| rate gap | {m66["gap"]["rate"]:.16f} |
| seg share | {shares["seg"]:.6f} |
| pose share | {shares["pose"]:.6f} |
| rate share | {shares["rate"]:.6f} |
| W bytes/flip | {m66["gap"]["W_bytes_per_argmax_flip"]:.16f} |

Axis warning: `{m66["cross_axis_warning"]}`.

## Freshness-at-consumption guard

The qo1 consumer guard refused the fz4-only r9m advisory-to-contest calibration prior:

`{refusal}`

Current/invariant rows consumed by the guard: `m66_gap_decomposition_inputs_qo1`,
`W_bytes_per_flip_exchange`.

## Queued follow-ons

Heavy scorer/atlas refreshes are in `queued_refreshes.jsonl`. #891 fiber-transport input custody is
in `transport_refreshes.jsonl`; UF1 found no data-complete stale row with stored H_ab and mixed
partial inputs, so no transport was executed.

No score claim. No promotion eligibility. Own-vehicle frontier unchanged:
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
"""


def _next_if_resumed() -> str:
    return """# NEXT-IF-RESUMED - ddm_uf1

1. Before any consumer reads `r9m_advisory_to_contest_cpu_calibration_prior` for qo1 or a later archive,
   fire a paired contest-CPU row for that exact archive, or keep the projection labeled queued.
2. When the scorer slot is free, refresh PF2, MS3-MS6, G3, G4, G2F, margin/saliency, and
   sensitivity-bitalloc rows only if a named consumer is about to use them.
3. For #891, first locate durable H_ab and mixed-partial inputs for a stale `(a,b)` row. If they are not
   available, route the row to full recompute; do not invent transport inputs.
4. If `gt_n600.npz` or the d2 pose trace sha changes, re-run the #931 prefix ratio derivation before
   using prefix negatives for pose-family decisions.
5. Preserve the scorer-free boundary: UF1 did not run `upstream/evaluate.py`, atlas jobs, or Modal jobs.
"""


def build_outputs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> WrittenOutputs:
    rows = build_refresh_rows()
    summary = registry_denominators(rows)
    m66 = recompute_m66_gap()
    guard = _guard_receipt(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = output_dir / "refresh_registry.jsonl"
    summary_path = output_dir / "refresh_summary.json"
    m66_path = output_dir / "m66_gap_decomposition_qo1.json"
    queued_path = output_dir / "queued_refreshes.jsonl"
    transport_path = output_dir / "transport_refreshes.jsonl"
    receipt_path = output_dir / "UF1_RECEIPT.md"
    next_path = output_dir / "NEXT-IF-RESUMED.md"

    write_refresh_registry_jsonl(rows, registry_path)
    _json_dump(
        summary_path,
        {
            "schema": "ddm_uf1_refresh_summary.v1",
            "current_qo1_archive_sha256": QO1_ARCHIVE_SHA256,
            "current_qo1_score": m66["ours"]["S"],
            "denominators": summary,
            "freshness_guard": guard,
            "score_claim": False,
            "promotion_eligible": False,
        },
    )
    _json_dump(m66_path, m66)
    _jsonl_dump(
        queued_path,
        [
            row.to_json_obj()
            for row in rows
            if row.disposition == DISPOSITION_QUEUED_HEAVY_REFRESH
        ],
    )
    _jsonl_dump(
        transport_path,
        [
            {
                **row.to_json_obj(),
                "uf1_action": "NO_TRANSPORT_EXECUTED_INPUT_BLOCKED",
            }
            for row in rows
            if row.refresh_route == ROUTE_FIBER_TRANSPORT
        ],
    )
    receipt_path.write_text(_receipt_markdown(summary, m66, guard), encoding="utf-8")
    next_path.write_text(_next_if_resumed(), encoding="utf-8")
    _ensure_no_tmp_citation(output_dir)
    return WrittenOutputs(
        output_dir=output_dir,
        registry_path=registry_path,
        summary_path=summary_path,
        m66_path=m66_path,
        queued_path=queued_path,
        transport_path=transport_path,
        receipt_path=receipt_path,
        next_path=next_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    outputs = build_outputs(args.output_dir)
    print(f"wrote UF1 refresh registry to {outputs.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
