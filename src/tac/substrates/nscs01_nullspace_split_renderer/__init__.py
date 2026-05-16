# SPDX-License-Identifier: MIT
"""NSCS01 nullspace-split-renderer substrate (assumptions-challenge audit
NSCS01 — exploits SegNet last-frame-only structural assumption).

Per ``.omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json``
entry SA02_segnet_only_last_frame: ``upstream/modules.py:108`` SegNet's
``preprocess_input`` slices ``x[:, -1, ...]`` and so frame[0] is in the
SegNet structural nullspace. NSCS01 exploits this with a TWO-HEAD renderer
where:

* ``frame_0_head`` is small (~30K params, 4-bit packed) — ONLY trained
  against the PoseNet axis (frame_0 is in SegNet's nullspace).
* ``frame_1_head`` is full (~150K params, 8-bit packed) — trained against
  both SegNet (last-frame slice) AND PoseNet (frame-1 contribution).

This is the ASSUMPTIONS-CHALLENGE-AUDIT NSCS01 hypothesis: by paying full
RGB-rate for frame[0] when only PoseNet sees it, current
single-renderer substrates waste ~7-15% of archive bytes on the wrong
distortion. The split exploits the asymmetry.

Distinction from sister D4 ``tac.substrates.d4_wyner_ziv_frame_0``
-----------------------------------------------------------------

D4 derives frame[0] from frame[1] via Wyner-Ziv (motion + residual). NSCS01
RENDERS both frames from the SAME per-pair latent but with DIFFERENT
per-head architectures and bit-widths. The two are STRUCTURALLY DISTINCT
exploits of the same SegNet nullspace and could compose: a future variant
could use NSCS01's ``frame_1_head`` + D4's frame_0 derivation.

UNIQUE-AND-COMPLETE-PER-METHOD (per the standing directive)
-----------------------------------------------------------

Per ``feedback_consolidate_everything_into_meta_layer_or_canonical_helpers_standing_directive_20260515.md``
the new operating mode is FORK by default. NSCS01 forks at the layers where
the structural insight cannot be expressed inside a shared canonical
helper, and ADOPTS canonical at every layer where the canonical pattern is
hygiene. Decision matrix in the design memo §7.

Catalog #124 archive-grammar 8 fields (declared inline so the AST walker observes them)
---------------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` (substrate-engineering)
- ``parser_section_manifest``: NSP1 header + (a) HEAD0_BLOB
  (frame_0_head weights at HEAD0_BITS=4; brotli) + (b) HEAD1_BLOB
  (frame_1_head weights at HEAD1_BITS=8; brotli) + (c) LATENT_BLOB
  (per-pair latents at LATENT_BITS=12; brotli) + (d) META_BLOB
  (sorted-keys JSON manifest with shapes + scales)
- ``inflate_runtime_loc_budget``: ≤ 200 LOC substrate-engineering waiver
  (full split-renderer reconstruction + per-pair forward + raw output)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 ≤ 2 deps)
- ``export_format``: NSP1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``NullspaceSplitScoreAwareLoss`` routes through
  ``tac.losses.scorer_loss_terms_btchw`` (the same low-level helper that
  ``score_pair_components_dispatch`` calls); BOTH scorers' ``preprocess_input``
  is invoked per Catalog #164. The split gradient routing is a structural
  forking of the canonical dispatch helper documented in design memo §4.
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity
  L7); the per-head bit-width split + per-pair latent contract is
  substrate engineering.
- ``no_op_detector_planned``: byte-mutation smoke per Catalog #139 — mutate
  HEAD0 byte → frame_0 changes; mutate HEAD1 byte → frame_1 changes;
  mutate LATENT byte → both frames change. Test verifies all three
  mutations produce different output bytes.

target_modes: ``research_substrate`` until trainer/recipe/auth-eval custody land
lane_class: ``substrate_engineering``
research_only: true (export-first design is present, but the promoted
  trainer, operator recipe, remote driver, byte-closed archive export, and
  paired auth-eval custody are not yet all present)
canary_status: ``independent_substrate`` (structurally distinct from
  HNeRV-family AND from sister D4 frame-0-derivation)

Predicted score band: unranked until
``tools/probe_nscs01_head0_arch_disambiguator.py`` records frame-0/frame-1
PoseNet sensitivity and head0 CNN-vs-MLP evidence.

Smoke envelope: TBD after probe; recipe is research-only/dispatch-disabled.

Cross-references
----------------

- Design memo: ``.omx/research/nscs01_nullspace_split_renderer_design_20260515.md``
- Audit anchor: ``.omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json``
  (entry SA02)
- Sister D4: ``tac.substrates.d4_wyner_ziv_frame_0`` (different exploit;
  could compose)
- Canonical scorer contract: ``tac.substrates.score_aware_common`` +
  ``tac.losses.scorer_loss_terms_btchw``
- Canonical inflate runtime helpers: ``tac.substrates._shared.inflate_runtime``
- Canonical trainer skeleton: ``tac.substrates._shared.trainer_skeleton``

Lane: ``lane_nscs01_nullspace_split_renderer_20260515``
"""

from tac.substrates.nscs01_nullspace_split_renderer.architecture import (
    CAMERA_H,
    CAMERA_W,
    NUM_PAIRS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    NullspaceSplitConfig,
    NullspaceSplitRenderer,
)
from tac.substrates.nscs01_nullspace_split_renderer.archive import (
    NSP1_HEADER_FMT,
    NSP1_HEADER_SIZE,
    NSP1_MAGIC,
    NSP1_SCHEMA_VERSION,
    NSP1_SECTION_ROLES,
    NullspaceSplitArchive,
    deserialize_head_state_dicts,
    deserialize_latents,
    pack_archive,
    parse_archive,
)
from tac.substrates.nscs01_nullspace_split_renderer.registered_substrate import (
    NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT,
)
from tac.substrates.nscs01_nullspace_split_renderer.score_aware_loss import (
    CONTEST_NORMALIZER,
    NullspaceSplitLossWeights,
    NullspaceSplitScoreAwareLoss,
)

__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "CONTEST_NORMALIZER",
    "NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT",
    "NSP1_HEADER_FMT",
    "NSP1_HEADER_SIZE",
    "NSP1_MAGIC",
    "NSP1_SCHEMA_VERSION",
    "NSP1_SECTION_ROLES",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "NullspaceSplitArchive",
    "NullspaceSplitConfig",
    "NullspaceSplitLossWeights",
    "NullspaceSplitRenderer",
    "NullspaceSplitScoreAwareLoss",
    "deserialize_head_state_dicts",
    "deserialize_latents",
    "pack_archive",
    "parse_archive",
]
