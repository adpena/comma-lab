# SPDX-License-Identifier: MIT
"""Three more anti-patterns from 2026-09-11/12 (ddm_cons3): two custody, one training state.

1. ``undived_reserve_constant_transferred_across_volume_regimes_v1`` (ddm_sr5, unit 2).
   ``src/comma_lab/storage_tiers.py`` carries ``DEFAULT_RESERVE_FREE_GB = 40.0`` with no comment,
   no measurement and no rationale.  The only 40 in the tree that carries a stated reason is a
   DIFFERENT constant on a DIFFERENT volume: ``tools/launch_detached_process.py``'s
   ``BOOT_MIN_FREE_GIB = 40.0``, derived from the 2026-09-04 ENOSPC where macOS swap reached
   72 GiB -- and swap lives on the boot volume, not on ``/Volumes/VertigoDataTier`` or
   ``/Volumes/APDataStore``.  ddm_cons3 re-derived both from source rather than quoting sr5: the
   comment is present on one and absent on the other, exactly as the memo says.  The uncontrolled-
   growth term that justifies 40 GiB on the boot volume is ABSENT on the SSDs, so the number has
   no support there; sr5's component sum lands at ~21 GiB, ~24 GiB with a 1.15x multiple.  The
   reserve is not passive: it refused a **3 KB** copy on a volume with 35.3 GiB genuinely free.
   **Nothing was changed.**  This anti-pattern records the transfer, not a new number.

2. ``display_unit_label_read_as_the_true_unit_v1`` (ddm_sr5, unit 1).
   ``df -h`` on this host prints the 10^9 value under a ``Gi`` label, and sr5's charter targets
   were written from that display.  The difference is 2^30/10^9 = 1.0737x: Vertigo's free space
   read 37,965,692,928 B, which is **35.358 GiB** and **37.97 GB** -- so a target written off the
   display names 7.4 % more bytes than it appears to.  The cure is to report ``statvfs``
   ``f_bavail x f_frsize`` in true GiB with the GB value alongside, which is what every sr5 table
   does.  This is the UNITS x LEVEL x AGGREGATION genus at its cheapest and most invisible: a
   correct number under a wrong label.

3. ``warm_start_init_dropped_the_quantization_state_v1`` (MEASURED by MAIN 2026-09-12; the price
   row is ddm_dpi1's and is PENDING).
   The reference refit law warm-starts from an EMA init that carries **zero** ``*.bit_depth``
   keys, while the source checkpoint it was cut from carries ten.  The trainer registers every
   row at ``init_bits = 8.0``, calls ``load_state_dict(strict=False)`` and explicitly tolerates
   the missing keys -- so every refit under this law starts its per-row depths at 8 bits and has
   30 QAT epochs to descend.  The consequence was MEASURED by hpr1 without the cause being named:
   retrained priors sit at mean row depth 5.888-5.890 bits against the shipped prior's 4.116,
   IDENTICALLY across geometries, which hpr1 read as "a 60-epoch training-budget effect".  The
   packed price of that inflation on move 47's prior was **+351 B** (12,262 against 11,911 B).
   A silent default in the harmful direction that no error reports: the confound signature.

The frontier is UNMOVED by this module: ``composition S 0.13638261682704697 @ 179,111 B
[contest-CUDA T4 n600] (move 48)``.  No reserve was changed, no training was run, no price is
claimed for the cure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tac.canonical_anti_patterns.anti_pattern import (
    INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
    INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
    PARADIGM_DATA_SOURCE,
    PARADIGM_OBSERVABILITY,
    PARADIGM_PROVENANCE,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_OBSERVED_HIGH,
    SEVERITY_OBSERVED_MEDIUM,
    AntiPattern,
    EmpiricalFalsification,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

_UTC = "2026-09-12T00:00:00Z"

_RESERVE_MEMO = ".omx/research/ddm_sr5_reserve_derivation_20260911.md"
_SR5_LEDGER = ".omx/research/ddm_sr5_certified_rebuildable_deletion_20260911.md"
_DPI1_CHARTER = ".omx/research/ddm_dpi1_hpac_depth_state_restored_warm_start_charter_20260912.md"
_HPR1_SHAPE = ".omx/research/ddm_hpr1_hpac_receptive_field_shape_rung_20260911.md"
_TIERS = "src/comma_lab/storage_tiers.py"
_LAUNCHER = "tools/launch_detached_process.py"

# MEASURED, re-derived from source by ddm_cons3 (not quoted from the memo).
SSD_RESERVE_GIB = 40.0            # storage_tiers.py:26, NO derivation comment
BOOT_RESERVE_GIB = 40.0           # launch_detached_process.py:396, WITH a derivation comment
BOOT_SWAP_PEAK_GIB = 72.0         # the 2026-09-04 ENOSPC that derived the boot figure
SINGLE_ARM_WORST_FOOTPRINT_GIB = 10.429
PROPOSED_RESERVE_GIB = 21.0
PROPOSED_RESERVE_WITH_MARGIN_GIB = 24.0
CONCURRENCY_ASSUMED = 2
ARMS_HARDCODING_FORTY = 6
REFUSED_COPY_BYTES = 3_000        # a 3 KB copy
FREE_AT_REFUSAL_GIB = 35.3

# MEASURED (units): the same free-space reading in both units.
VERTIGO_FREE_BYTES_BEFORE = 37_965_692_928
VERTIGO_FREE_GIB_BEFORE = 35.358
VERTIGO_FREE_GB_BEFORE = 37.97
VERTIGO_FREE_BYTES_AFTER = 65_536_303_104
VERTIGO_FREE_GIB_AFTER = 61.036
VERTIGO_FREE_GB_AFTER = 65.54
GIB_PER_GB = 2**30 / 10**9

# MEASURED (warm start): the dropped state and its packed price.
INIT_BIT_DEPTH_KEYS = 0
SOURCE_BIT_DEPTH_TENSORS = 10
INIT_BITS_DEFAULT = 8.0
SHIPPED_MEAN_ROW_DEPTH_BITS = 4.116
RETRAINED_MEAN_ROW_DEPTH_BITS = (5.888, 5.890)
HPAC_BYTES_SHIPPED = 11_911
HPAC_BYTES_RETRAINED = 12_262
HPAC_BYTES_RETRAINED_AFTER_EVEN_ROUNDING = 11_629
PACKED_PRICE_OF_INFLATION_BYTES = HPAC_BYTES_RETRAINED - HPAC_BYTES_SHIPPED
INIT_SHA_PREFIX = "cf411127"
SOURCE_CKPT_SHA_PREFIX = "5007beae"


def _provenance(sidecar: str, reactivation: str, axis: str, substrate: str) -> Provenance:
    return build_provenance_for_research_sidecar(
        sidecar_path=sidecar,
        reactivation_criteria=reactivation,
        measurement_axis=axis,
        hardware_substrate=substrate,
        captured_at_utc=_UTC,
    )


def build_undived_reserve_transferred_across_volume_regimes_v1() -> AntiPattern:
    """A floor derived for one volume, copied onto volumes whose failure mode does not exist."""
    anti_pattern_id = "undived_reserve_constant_transferred_across_volume_regimes_v1"
    provenance = _provenance(
        _RESERVE_MEMO,
        (
            "this anti-pattern records a TRANSFER, not a replacement number. sr5's ~21/~24 GiB "
            "rests on a concurrency factor of 2 that is a JUDGEMENT, not a measurement: four "
            "simultaneous cold decodes would need ~42 GiB and would VINDICATE 40. Before any "
            "value changes, measure the true distribution of simultaneously-decoding arms and "
            "take a direct high-water mark of one cold n600 decode plus encode, and make the six "
            "arms that hardcode 40 * 1024**3 import the canonical constant -- until they do, "
            "changing the canonical value changes nothing"
        ),
        "[macOS filesystem custody; scorer-free]",
        "macOS boot volume + external SSD tiers (VertigoDataTier, APDataStore)",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="sr5_ssd_reserve_is_a_boot_swap_floor_20260911",
        measurement_method=(
            "read every reserve constant in the tree and ask each one for its derivation; "
            "assemble what the SSD reserve must actually cover from measured component sizes; "
            "record what the enforced number refused on the day"
        ),
        empirical_artifact_path=_RESERVE_MEMO,
        empirical_output={
            "canonical_ssd_reserve": {
                "site": f"{_TIERS}:26 DEFAULT_RESERVE_FREE_GB",
                "value_gib": SSD_RESERVE_GIB,
                "derivation_present": False,
                "re_derived_by_cons3_from_source": True,
            },
            "the_only_forty_with_a_reason": {
                "site": f"{_LAUNCHER}:396 BOOT_MIN_FREE_GIB",
                "value_gib": BOOT_RESERVE_GIB,
                "derivation": (
                    f"2026-09-04 ENOSPC; macOS swap reached {BOOT_SWAP_PEAK_GIB} GiB, and swap "
                    "lives on /System/Volumes/Data"
                ),
                "applies_to": "the boot volume only",
            },
            "why_it_does_not_transfer": (
                "neither SSD tier hosts swap, hosts the OS, or takes Time Machine local snapshots "
                "(tmutil reported none on Vertigo at 22:05Z). Their only writers are arms, and "
                "arms size their own payloads -- the uncontrolled-growth term is absent"
            ),
            "what_the_reserve_must_cover_gib": {
                "serializer_fallback_bundle": 0.0154,
                "one_cold_n600_raw": 3.411,
                "its_in_progress_twin": 3.411,
                "its_token_stage_file": 0.110,
                "one_twin_encode_or_candidate": 3.26,
                "scorer_input_caches": 0.220,
                "candidate_runtime_tree": 0.002,
                "single_arm_worst_simultaneous": SINGLE_ARM_WORST_FOOTPRINT_GIB,
            },
            "proposed_not_applied_gib": {
                "derived": PROPOSED_RESERVE_GIB,
                "with_1p15x_margin": PROPOSED_RESERVE_WITH_MARGIN_GIB,
                "concurrency_factor_assumed": CONCURRENCY_ASSUMED,
            },
            "what_the_enforced_number_refused": {
                "a_3kb_copy_with_free_gib": FREE_AT_REFUSAL_GIB,
                "three_instruments_agreed": (
                    "df 35.32 GiB, diskutil apfs 37,921,574,912 B, tmutil no snapshots "
                    "(MEASURED by ddm_tmx1 at 22:05Z and relayed)"
                ),
                "a_10p9_gib_certified_move": "destination would land at 38.2 GiB, 1.8 under",
                "arm_side_statement": "'under 11 GiB usable above its 40 GiB reserve'",
                "what_it_manufactured": (
                    "sr5 deleted 28.247 GiB of certified-rebuildable payload and relocated a "
                    "6.27 GB tar to keep two tiers above the floor"
                ),
            },
            "not_a_single_point_of_control": (
                f"{ARMS_HARDCODING_FORTY} arms re-hardcode 40 * 1024**3 instead of importing the "
                "canonical constant (bnd2, bnd3, gdc1, dwc1, ntb2, rbf1)"
            ),
            "nothing_was_changed": True,
        },
        # DERIVED: how far the enforced floor sits above the largest derived need.
        falsification_residual=(
            (SSD_RESERVE_GIB - PROPOSED_RESERVE_WITH_MARGIN_GIB)
            / PROPOSED_RESERVE_WITH_MARGIN_GIB
        ),
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_MEDIUM,
        operator_routable_unwind_path=(
            "give the constant a derivation comment naming its measured components, make every "
            "call site import it, and treat the reserve as DERIVED OUTPUT -- never change a "
            "reserve in the minute after it refuses you"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A guard constant derived for one storage regime is copied onto another where the "
            "failure mode it protects against does not exist, and nobody re-derives it at the new "
            "scope. The SSD tiers enforce a 40 GiB floor whose only stated derivation is a BOOT "
            "volume's swap growth (72 GiB during a 2026-09-04 near-OOM); neither SSD hosts swap, "
            "the OS, or local snapshots. Measured cost: a 3 KB copy refused with 35.3 GiB free, a "
            "certified 10.9 GiB move refused, and 28.2 GiB of certified payload deleted to "
            "manufacture headroom under a floor roughly 1.7-1.9x its own job."
        ),
        forbidden_pattern_predicate=(
            "reserve_constant.value == another_regime.constant.value AND "
            "reserve_constant.derivation_comment is absent AND "
            "the justifying failure mode (OS swap / uncontrolled growth) is absent at this scope"
        ),
        falsification_band={
            "enforced_ssd_reserve_gib": SSD_RESERVE_GIB,
            "derived_need_gib": PROPOSED_RESERVE_GIB,
            "derived_need_with_margin_gib": PROPOSED_RESERVE_WITH_MARGIN_GIB,
            "single_arm_worst_footprint_gib": SINGLE_ARM_WORST_FOOTPRINT_GIB,
            "free_space_at_a_refused_3kb_copy_gib": FREE_AT_REFUSAL_GIB,
            "call_sites_hardcoding_the_value": float(ARMS_HARDCODING_FORTY),
        },
        recurrence_conditions=(
            "a constant is introduced with a derivation on one volume and reused on another",
            "the reused site carries no comment, so the transfer is invisible to grep",
            "arms hardcode the literal rather than importing the canonical constant",
            "a guard refuses real work and the reflex is to lower it rather than derive it",
        ),
        canonical_source_anchor=f"{_RESERVE_MEMO} (whole memo); constants at {_TIERS}:26 and {_LAUNCHER}:396",
        canonical_unwind_path=(
            "Re-derive the number at the scope where it binds: enumerate what the reserve must "
            "cover that a job CANNOT size in advance, measure each component, and write the sum "
            "into a derivation comment the way BOOT_MIN_FREE_GIB already does. Make every call "
            "site import the canonical constant so there is one point of control. Treat the "
            "reserve as derived OUTPUT, re-measured whenever the decode or candidate shape "
            "changes. Propose, never apply in the minute a guard refuses you."
        ),
        canonical_producers=(_TIERS, _LAUNCHER, "tools/vertigo_certify_move.py"),
        canonical_consumers=(
            _RESERVE_MEMO,
            _SR5_LEDGER,
            ".omx/research/ddm_tmx1_refit_tail_mixer_tc1_on_current_field_20260911.md",
            "experiments/ddm_rbf1_boundary_probe.py",
        ),
        paradigm_class=PARADIGM_PROVENANCE,
        severity=SEVERITY_MEDIUM,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_display_unit_label_read_as_the_true_unit_v1() -> AntiPattern:
    """A tool prints one unit under another unit's label, and a target is written from the label."""
    anti_pattern_id = "display_unit_label_read_as_the_true_unit_v1"
    provenance = _provenance(
        _SR5_LEDGER,
        (
            "re-check whenever a free-space, size or quota target is written from a human-readable "
            "display rather than from a byte count. The cure is structural: report bytes, then "
            "both units, and never let a charter target inherit a label"
        ),
        "[macOS filesystem custody; scorer-free]",
        "macOS df(1) / statvfs on external SSD tiers",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="sr5_df_h_prints_gb_under_a_gi_label_20260911",
        measurement_method=(
            "read free space with statvfs (f_bavail x f_frsize) and print the true GiB (2^30) "
            "value beside the GB (10^9) value, then compare both against what df -h displays"
        ),
        empirical_artifact_path=_SR5_LEDGER,
        empirical_output={
            "vertigo_before": {
                "bytes": VERTIGO_FREE_BYTES_BEFORE,
                "true_gib": VERTIGO_FREE_GIB_BEFORE,
                "gb": VERTIGO_FREE_GB_BEFORE,
                "what_df_h_showed_under_Gi": VERTIGO_FREE_GB_BEFORE,
            },
            "vertigo_after": {
                "bytes": VERTIGO_FREE_BYTES_AFTER,
                "true_gib": VERTIGO_FREE_GIB_AFTER,
                "gb": VERTIGO_FREE_GB_AFTER,
            },
            "inflation_factor_derived": GIB_PER_GB,
            "consequence": (
                "a target written from the display names ~7.4 % more bytes than it appears to; "
                "sr5's charter targets were written that way and the memo says so in the table "
                "that corrects them"
            ),
            "why_it_is_invisible": (
                "the number is correct and the LABEL is wrong, so every cross-check against the "
                "same display agrees with itself"
            ),
        },
        falsification_residual=GIB_PER_GB - 1.0,
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
        severity_observed=SEVERITY_OBSERVED_MEDIUM,
        operator_routable_unwind_path=(
            "quote bytes first and both units after; never take a threshold from a -h display"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A measurement is taken from a human-readable display whose unit LABEL does not match "
            "the unit it prints. On this host df -h prints the 10^9 value under a 'Gi' label, so "
            "a free-space target written from the display is 2^30/10^9 = 1.0737x off -- 37.97 "
            "shown where 35.358 GiB is true. The value is right and the label is wrong, so every "
            "cross-check against the same display confirms it."
        ),
        forbidden_pattern_predicate=(
            "threshold_or_target.source == human_readable_display AND "
            "display.unit_label != display.unit_actually_printed"
        ),
        falsification_band={
            "gib_per_gb": GIB_PER_GB,
            "relative_error": GIB_PER_GB - 1.0,
            "measured_bytes": float(VERTIGO_FREE_BYTES_BEFORE),
            "true_gib": VERTIGO_FREE_GIB_BEFORE,
            "displayed_value_under_a_Gi_label": VERTIGO_FREE_GB_BEFORE,
        },
        recurrence_conditions=(
            "a charter or target is written from a terminal display rather than from bytes",
            "a tool's -h / --human output is used as a data source",
            "two instruments are cross-checked and both read the same display",
            "GiB and GB are used interchangeably in prose near a threshold",
        ),
        canonical_source_anchor=f"{_SR5_LEDGER} section 'FREED, WITH df BEFORE AND AFTER'",
        canonical_unwind_path=(
            "Take free space from statvfs (f_bavail x f_frsize) in BYTES, then render true GiB "
            "(2^30) with the GB (10^9) value alongside so the reader cannot confuse them, and "
            "state which one any threshold is written in."
        ),
        canonical_producers=(
            "tools/sr5_certify_delete_derived.py",
            "tools/sr5_certify_hardlink_dedup.py",
        ),
        canonical_consumers=(_SR5_LEDGER, _RESERVE_MEMO),
        paradigm_class=PARADIGM_OBSERVABILITY,
        severity=SEVERITY_MEDIUM,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_warm_start_init_dropped_the_quantization_state_v1() -> AntiPattern:
    """A warm-start checkpoint that silently lost the state every refit then relearns from 8 bits."""
    anti_pattern_id = "warm_start_init_dropped_the_quantization_state_v1"
    provenance = _provenance(
        _DPI1_CHARTER,
        (
            "the PRICE of the cure is not established and is not claimed here: ddm_dpi1 owns it "
            "and its row is PENDING. Reactivate this anti-pattern's band when dpi1 reports the "
            "epoch-0 histogram, the terminal mean row depth, and the archive bytes of a refit "
            "warm-started from an initializer that carries the depths -- and note that a depth "
            "restoration could also LOSE, exactly as tmx1's low-capacity refit did"
        ),
        "[MEASURED defect receipts; the cure's price is PENDING]",
        "macOS Metal training host + real coder (the price leg is not yet measured)",
    )
    incident = EmpiricalFalsification(
        anti_pattern_id=anti_pattern_id,
        falsification_id="main_hpac_warm_start_init_carries_zero_bit_depth_keys_20260912",
        measurement_method=(
            "zip-scan the warm-start initializer's data.pkl for *.bit_depth keys, scan the source "
            "checkpoint it was cut from for the same keys, and read the trainer's load path for "
            "how it treats them missing; then compare the shipped prior's mean row depth against "
            "every retrained prior's"
        ),
        empirical_artifact_path=_DPI1_CHARTER,
        empirical_output={
            "initializer": {
                "sha_prefix": INIT_SHA_PREFIX,
                "bit_depth_keys": INIT_BIT_DEPTH_KEYS,
            },
            "source_checkpoint_it_was_cut_from": {
                "sha_prefix": SOURCE_CKPT_SHA_PREFIX,
                "bit_depth_tensors": SOURCE_BIT_DEPTH_TENSORS,
            },
            "trainer_behaviour": (
                "registers every row's bit depth at init_bits = 8.0, then load_state_dict("
                "strict=False) and explicitly tolerates missing .bit_depth keys (allowed_missing)"
                " -- so the depths are not restored and nothing reports it"
            ),
            "measured_consequence": {
                "shipped_prior_mean_row_depth_bits": SHIPPED_MEAN_ROW_DEPTH_BITS,
                "retrained_prior_mean_row_depth_bits": list(RETRAINED_MEAN_ROW_DEPTH_BITS),
                "identical_across_geometries": True,
                "what_the_producing_arm_called_it": (
                    "'a 60-epoch training-budget effect' -- the consequence was measured, the "
                    "cause was not found"
                ),
            },
            "packed_price_of_the_inflation_bytes": PACKED_PRICE_OF_INFLATION_BYTES,
            "hpac_member_bytes": {
                "shipped": HPAC_BYTES_SHIPPED,
                "retrained": HPAC_BYTES_RETRAINED,
                "retrained_after_even_rounding_move48": (
                    HPAC_BYTES_RETRAINED_AFTER_EVEN_ROUNDING
                ),
            },
            "why_it_is_a_confound_and_not_a_bug_report": (
                "silent x default-in-the-harmful-direction x measurement-corrupting: the refit "
                "still produced -887 B and became pointer move 47, so nothing failed loudly and "
                "the model leg's +351 B was attributed to training budget"
            ),
            "sister_trap": (
                "the flag and the state disagree silently -- init_bits stays 8.0 as a FLAG while "
                "a checkpoint's state is what should override it"
            ),
            "the_cure_is_not_priced_here": True,
        },
        falsification_residual=None,
        captured_at_utc=_UTC,
        canonical_provenance=provenance,
        incident_classification=INCIDENT_RATIFICATION_AT_NEW_SUBSTRATE,
        severity_observed=SEVERITY_OBSERVED_HIGH,
        operator_routable_unwind_path=(
            "rebuild the initializer from the source checkpoint's own state and assert at epoch 0 "
            "that the loaded depths reproduce the shipped mean row depth before any epoch runs"
        ),
    )
    return AntiPattern(
        anti_pattern_id=anti_pattern_id,
        description=(
            "A warm-start initializer is cut from a checkpoint and silently loses state the "
            "training law depends on -- here the ten per-row bit_depth tensors -- while a tolerant "
            "load path (strict=False plus an allowed-missing list) makes the loss unobservable. "
            "Every refit then restarts that state from a hardcoded default (8 bits) with only the "
            "QAT epochs to descend, so retrained sections sit above the shipped one's depth "
            "(5.888-5.890 against 4.116 bits, identically across geometries) and the inflation is "
            "attributed to training budget. Packed price on one section: +351 B."
        ),
        forbidden_pattern_predicate=(
            "warm_start_init.state_dict is missing keys the law's own source checkpoint carries "
            "AND loader.strict is False AND those keys are on an allowed-missing list "
            "AND the corresponding runtime value falls back to a hardcoded default"
        ),
        falsification_band={
            "bit_depth_keys_in_initializer": float(INIT_BIT_DEPTH_KEYS),
            "bit_depth_tensors_in_source_checkpoint": float(SOURCE_BIT_DEPTH_TENSORS),
            "fallback_init_bits": INIT_BITS_DEFAULT,
            "shipped_mean_row_depth_bits": SHIPPED_MEAN_ROW_DEPTH_BITS,
            "retrained_mean_row_depth_bits_low": RETRAINED_MEAN_ROW_DEPTH_BITS[0],
            "retrained_mean_row_depth_bits_high": RETRAINED_MEAN_ROW_DEPTH_BITS[1],
            "packed_price_bytes": float(PACKED_PRICE_OF_INFLATION_BYTES),
        },
        recurrence_conditions=(
            "an EMA or averaged checkpoint is cut from a live one and keeps only parameter rows",
            "a loader tolerates missing keys by an allowlist rather than by an explicit contract",
            "the missing state has a hardcoded default that is valid but wrong",
            "a downstream arm measures the consequence and names a plausible cause (budget, "
            "schedule, seed) that no receipt supports",
        ),
        canonical_source_anchor=f"{_DPI1_CHARTER} -- 'The measured defect this rung cures'",
        canonical_unwind_path=(
            "Build an OWNED initializer: the EMA state plus the state keys the law's source "
            "checkpoint carries, with the same keys and shapes, asserting that every compressible "
            "row has one. Prove it at epoch 0 by the trainer's own histogram -- the loaded depths "
            "must reproduce the shipped section's mean row depth before a single epoch runs -- "
            "and record both source hashes and the new file's. Then change NOTHING else, so the "
            "price of the restoration is measured alone."
        ),
        canonical_producers=("tools/train_ddm_cl1_hpac_capacity.py",),
        canonical_consumers=(_DPI1_CHARTER, _HPR1_SHAPE),
        paradigm_class=PARADIGM_DATA_SOURCE,
        severity=SEVERITY_HIGH,
        provenance=provenance,
        empirical_falsifications=(incident,),
        last_recalibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )
