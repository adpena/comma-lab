# SPDX-License-Identifier: MIT
"""Canonical ALASKA inverse-steganalysis patterns (Yousfi et al. 2019).

Ports the **conceptual** patterns from
``github.com/YassineYousfi/alaska`` (Copyright (c) 2019 DDE Lab, Binghamton
University; redistributed under the educational/research/non-profit license
in :file:`external/alaska_yousfi/LICENSE.md`) to the comma-video
contest substrate surface.

Origin paper
------------
::

    @inproceedings{Yousfi2019Alaska,
      author    = {Yousfi, Yassine and Butora, Jan and Fridrich, Jessica and Giboulot, Quentin},
      title     = {Breaking ALASKA: Color Separation for Steganalysis in JPEG Domain},
      booktitle = {Proceedings of the ACM Workshop on Information Hiding and Multimedia Security},
      series    = {IH\&MMSec'19},
      year      = {2019},
      pages     = {138--149},
      doi       = {10.1145/3335203.3335727},
    }

Operator binding
----------------
Operator 2026-05-30 verbatim: *"what is yousfi's alaska repo, what is it, what
does it do, spawn a deep research and copy agent"* + *"all of yousfi's
recommendations are approved"*. Per CLAUDE.md "Exact scorer architectures":
Yousfi was Fridrich's PhD student at Binghamton's DDE Lab; the canonical
comma SegNet (``smp.Unet('tu-efficientnet_b2')``) + PoseNet (``FastViT-T12``)
contest scorers are designed against the same canonical inverse-steganalysis
intuitions Yousfi used to win ALASKA #1 in 2019. Mining alaska for canonical
patterns is the **canonical knowledge transfer** path for our pose-axis attack
cascade.

5-axis adaptation taxonomy
--------------------------
Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable + 15-item-audit
1:1-fidelity standing directive: every helper here documents the canonical
ALASKA source location + 5-axis classification:

* **Axis A (contest)** — JPEG steganalysis -> comma-video lossy compression
  (different cover statistics: JPEG quantization noise vs YUV6 contest scorer)
* **Axis B (problem space)** — embedding 0.4 bpac payload -> minimising
  ``d_seg + sqrt(10*d_pose) + 25*rate``
* **Axis C (math)** — 5-class softmax over {cover, EBS, JUNI, NSF5, UED} ->
  per-pair carrier (cover, perturbed)
* **Axis D (data)** — BOSSBase JPEG QF95 256x256 -> ``upstream/videos/0.mkv``
  1164x874x1200 frames per Catalog #213
* **Axis E (video)** — single-image scoring -> per-pair (frame_0, frame_1)
  scoring with shared latent

Public surface
--------------
* :class:`ColorBranchStrategy` — canonical YCrCb/Y/CrCb/Cr/Cb channel slicing
  per Yousfi's `branch_to_slice` (jpeg_utils.py:50-62 upstream).
* :class:`ColorBranchSliceStrategy` — sister enum for the comma YUV6 scorer
  contract (Y/UV/YUV6/Y_UV_separate per CLAUDE.md PoseNet design).
* :class:`PairConstraintBatchStrategy` — canonical cover-stego pair-constraint
  batching per Yousfi's `gen_train`/`gen_valid` (tf_utils.py:55-95 upstream).
* :class:`MultiSchemeDirichletPrior` — canonical {EBS:0.15, JUNI:0.4, NSF5:0.15,
  UED:0.3} categorical prior per Yousfi's `priors` (tf_fine_tune_branch.ipynb
  upstream); generalized to comma's per-pair perturbation modes.
* :func:`build_alaska_canonical_patterns_inventory` — operator-facing
  introspection helper returning the canonical pattern registry.

Sister landings
---------------
* Slot FF ``tac.composition.uniward_canonical_inverse_steganalysis_holub_fridrich_denemark_2014`` — UNIWARD per Holub-Fridrich-Denemark 2014.
* Slot YY ``tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014`` — HILL per Li-Wang-Li-Huang 2014.
* Slot AAA ``tac.composition.mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016`` — MiPOD per Sedighi-Cogranne-Fridrich 2016.
* Slot CCC ``tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010`` — HUGO per Pevny-Filler-Bas 2010.
* THIS ``tac.composition.alaska_inverse_steganalysis_patterns`` — Yousfi 2019 ALASKA-#1-winning canonical patterns (color separation + pair constraint + multi-scheme prior + SRNet feature extractor; sister of FF/YY/AAA/CCC at the **detector-architecture** surface; FF/YY/AAA/CCC are at the **cost-function** surface).

Cross-references
----------------
* CLAUDE.md "Yousfi's repos" (cataloged 2026-04-21).
* CLAUDE.md "Exact scorer architectures" (canonical SegNet + PoseNet design
  vendored from Yousfi's `comma10k-baseline` repo).
* CLAUDE.md "Fridrich inverse steganalysis" (UNIWARD + detector-informed
  embedding = canonical TTO approach).
* Catalog #109 (vendored intake clones pristine discipline; the
  ``external/alaska_yousfi/`` clone is read-only).
* Catalog #213 (canonical Comma2k19 cache routing for dataset bytes).
* Catalog #213 sister: ``upstream/videos/0.mkv`` is the canonical contest
  video; alaska canonical helpers route via PyAV per the CLAUDE.md
  "Forbidden ``make_synthetic_pair_batch`` calls" non-negotiable.
"""

from __future__ import annotations

from tac.composition.alaska_inverse_steganalysis_patterns.canonical_color_separation import (
    ColorBranchStrategy,
    ColorBranchSliceStrategy,
    branch_to_yuv6_channel_slice,
    SRNET_BRANCH_ORDER,
    YUV6_CHANNEL_LAYOUT,
)
from tac.composition.alaska_inverse_steganalysis_patterns.canonical_pair_constraint_batch import (
    PairConstraintBatchStrategy,
    PairConstraintBatchConfig,
    build_pair_constraint_batch,
    PairConstraintBatchError,
)
from tac.composition.alaska_inverse_steganalysis_patterns.canonical_multi_scheme_prior import (
    MultiSchemeDirichletPrior,
    MultiSchemePriorConfig,
    sample_perturbation_scheme,
    canonical_alaska_1_priors,
    canonical_comma_pair_perturbation_priors,
    MultiSchemePriorError,
)
from tac.composition.alaska_inverse_steganalysis_patterns.canonical_detector_aware_iterative_training import (
    DetectorAwareIterativeTrainingStrategy,
    DetectorAwareTrainingConfig,
    DetectorAwareTrainingError,
)
from tac.composition.alaska_inverse_steganalysis_patterns.canonical_cmd_per_image_discrimination import (
    CMDDiscriminationStrategy,
    CMDDiscriminationConfig,
    compute_cmd_per_image_score,
    CMDDiscriminationError,
)
from tac.composition.alaska_inverse_steganalysis_patterns.canonical_pattern_inventory import (
    AlaskaCanonicalPatternRow,
    build_alaska_canonical_patterns_inventory,
    ALASKA_REPO_ATTRIBUTION,
    ALASKA_ORIGIN_PAPER_CITATION,
)

__all__ = (
    "ColorBranchStrategy",
    "ColorBranchSliceStrategy",
    "branch_to_yuv6_channel_slice",
    "SRNET_BRANCH_ORDER",
    "YUV6_CHANNEL_LAYOUT",
    "PairConstraintBatchStrategy",
    "PairConstraintBatchConfig",
    "build_pair_constraint_batch",
    "PairConstraintBatchError",
    "MultiSchemeDirichletPrior",
    "MultiSchemePriorConfig",
    "sample_perturbation_scheme",
    "canonical_alaska_1_priors",
    "canonical_comma_pair_perturbation_priors",
    "MultiSchemePriorError",
    "DetectorAwareIterativeTrainingStrategy",
    "DetectorAwareTrainingConfig",
    "DetectorAwareTrainingError",
    "CMDDiscriminationStrategy",
    "CMDDiscriminationConfig",
    "compute_cmd_per_image_score",
    "CMDDiscriminationError",
    "AlaskaCanonicalPatternRow",
    "build_alaska_canonical_patterns_inventory",
    "ALASKA_REPO_ATTRIBUTION",
    "ALASKA_ORIGIN_PAPER_CITATION",
)
