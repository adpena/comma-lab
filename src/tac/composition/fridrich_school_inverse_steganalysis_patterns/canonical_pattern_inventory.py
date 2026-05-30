# SPDX-License-Identifier: MIT
"""Canonical Fridrich-school pattern inventory + operator-facing cross-reference.

Operator-facing introspection helper returning the registered patterns +
their HARD-EARNED-vs-CARGO-CULTED classification per CLAUDE.md Catalog
#303 sister discipline + Yousfi-Fridrich-DDELab cascade cross-reference.

Sister of ``tac.composition.alaska_inverse_steganalysis_patterns.canonical_pattern_inventory``
at the BROADER-Fridrich-school surface; alaska covers the ALASKA-2019-paper
canonical patterns; this inventory covers Yousfi's POST-alaska recent repos
(autostego / deepsteganalysis / OneHotConv / comma10k-baseline) + Fridrich's
other students (Filler STC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = (
    "FridrichSchoolCanonicalPatternRow",
    "build_fridrich_school_canonical_patterns_inventory",
    "FRIDRICH_SCHOOL_ATTRIBUTION",
    "YOUSFI_GITHUB_HOMEPAGE",
    "FRIDRICH_GROUP_ATTRIBUTION",
)


YOUSFI_GITHUB_HOMEPAGE: str = "https://github.com/YassineYousfi"
"""Yousfi's GitHub homepage (research engineer at comma.ai per
https://yassineyousfi.github.io/)."""


FRIDRICH_SCHOOL_ATTRIBUTION: str = (
    "Patterns derived from canonical repos by Yassine Yousfi (research "
    "engineer at comma.ai; PhD Fridrich DDE Lab Binghamton) and the DDE "
    "Lab organization (https://github.com/DDELab). Sister patterns "
    "from Tomas Filler (Fridrich PhD student; canonical STC author) + "
    "Vojtech Holub (UNIWARD author) + Vahid Sedighi (MiPOD author) + "
    "Tomas Pevny (HUGO author). All upstream code redistributed under "
    "the respective licenses (DDE Lab educational/research/non-profit "
    "license + MIT for downstream PyTorch Lightning ports)."
)


FRIDRICH_GROUP_ATTRIBUTION: str = (
    "Fridrich PhD students cited in this inventory (chronological): "
    "Tomas Filler (STC; canonical syndrome-trellis coding), Tomas Pevny "
    "(HUGO; co-author with Filler-Bas 2010 + author of SRM with Bas-Fridrich "
    "2011), Vojtech Holub (UNIWARD; canonical universal distortion function "
    "with Fridrich-Denemark 2014), Vahid Sedighi (MiPOD; Sedighi-Cogranne-"
    "Fridrich 2016), Mehdi Boroumand (SRNet; Boroumand-Chen-Fridrich 2019), "
    "Jan Kodovsky (KV+HV models; ensemble classifier), Tomas Denemark "
    "(content-adaptive embedding co-author), Jan Butora (ALASKA co-author), "
    "Quentin Giboulot (ALASKA co-author), Yassine Yousfi (ALASKA + OneHot + "
    "comma10k-baseline + autostego + currently research engineer at comma.ai)."
)


@dataclass(frozen=True)
class FridrichSchoolCanonicalPatternRow:
    """One canonical pattern row.

    Attributes
    ----------
    pattern_id
        Canonical name (e.g. ``"alice_vs_eve_adversarial_loop"``).
    upstream_source
        Upstream repo + file:line reference if applicable.
    upstream_author
        Primary author + Fridrich-school lineage (e.g. ``"Yousfi (Fridrich)"``).
    upstream_paper_or_repo
        Paper DOI or GitHub URL.
    last_updated_or_published
        ISO date of last meaningful update.
    tac_module
        Where the pattern lives in tac.
    five_axis_classification
        Mapping ``{"contest"/"problem_space"/"math"/"data"/"video"}`` ->
        HARD-EARNED-1:1 / DOCUMENTED-ADAPTATION / N/A per Catalog #303.
    cross_reference_yousfi_fridrich_axis
        Which Yousfi-Fridrich-cascade axis this slots into.
    pr_score_lowering_ev_estimate
        Operator-facing predicted ΔS band per Catalog #296 +
        ``FORMALIZATION_PENDING`` per Catalog #344 if not yet bounded.
    """

    pattern_id: str
    upstream_source: str
    upstream_author: str
    upstream_paper_or_repo: str
    last_updated_or_published: str
    tac_module: str
    five_axis_classification: Mapping[str, str]
    cross_reference_yousfi_fridrich_axis: str
    pr_score_lowering_ev_estimate: str


def build_fridrich_school_canonical_patterns_inventory() -> tuple[
    FridrichSchoolCanonicalPatternRow, ...
]:
    """Return canonical 7-pattern inventory + cross-reference matrix.

    Returns
    -------
    tuple[FridrichSchoolCanonicalPatternRow, ...]
        7 canonical patterns extracted from Yousfi's recent repos + sister
        Fridrich-school authors.
    """
    return (
        FridrichSchoolCanonicalPatternRow(
            pattern_id="alice_vs_eve_adversarial_loop",
            upstream_source="github.com/YassineYousfi/autostego (README + eve.py + program_eve.md)",
            upstream_author="Yousfi (Fridrich; currently comma.ai)",
            upstream_paper_or_repo="https://github.com/YassineYousfi/autostego",
            last_updated_or_published="2026-03-11",
            tac_module=(
                "tac.composition.fridrich_school_inverse_steganalysis_patterns."
                "canonical_alice_vs_eve_adversarial_loop"
            ),
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:JPEG-to-comma-video",
                "problem_space": "DOCUMENTED-ADAPTATION:round-robin-to-fixed-eve",
                "math": "HARD-EARNED-1:1:minimax-Alice-min-Eve-max",
                "data": "DOCUMENTED-ADAPTATION:BOSSbase-to-Comma2k19",
                "video": "DOCUMENTED-ADAPTATION:per-image-to-per-pair",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 0 (foundational; canonical adversarial game-theoretic "
                "framing for every substrate attack the Slot FF/YY/AAA/CCC + "
                "Yousfi-Tier-1 pose-axis canonical helpers compose)"
            ),
            pr_score_lowering_ev_estimate=(
                "[+0.000, -0.001] DISPATCH-BUDGET-EV-NOT-SCORE-EV per "
                "Catalog #344; the EV is in framing not direct score; "
                "operationalized via Slot FF/YY/AAA/CCC dispatch budget "
                "via canonical minimax scoring rule"
            ),
        ),
        FridrichSchoolCanonicalPatternRow(
            pattern_id="lclsmr_linear_steganalysis_detector",
            upstream_source=(
                "github.com/YassineYousfi/autostego/blob/eve/steganalysis/lclsmr.py + "
                "_lclsmr.py (LSMR solver)"
            ),
            upstream_author="Yousfi (Fridrich; currently comma.ai)",
            upstream_paper_or_repo="https://github.com/YassineYousfi/autostego",
            last_updated_or_published="2026-03-11",
            tac_module=(
                "tac.composition.fridrich_school_inverse_steganalysis_patterns."
                "canonical_lclsmr_linear_steganalysis_detector"
            ),
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:detector-to-vulnerability-regression",
                "problem_space": "DOCUMENTED-ADAPTATION:binary-to-regression",
                "math": "HARD-EARNED-1:1:LSMR-Fong-Saunders-2011",
                "data": "DOCUMENTED-ADAPTATION:BOSSbase-features-to-pair-latent",
                "video": "DOCUMENTED-ADAPTATION:per-image-to-per-pair",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 10 (NEW; canonical lightweight linear detector that "
                "every Slot RR pose-axis null projection + sister Yousfi-Tier-1 "
                "pose-vulnerability map per-pair regression can adopt; faster "
                "than SRNet and more accurate than naive linear regression on "
                "ill-conditioned feature matrices)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.002, +0.000] FORMALIZATION_PENDING per Catalog #344; "
                "the EV is in canonical pose-vulnerability regression speed "
                "+ accuracy; compounds with Slot RR + Yousfi-Tier-1 pose-axis "
                "canonical helpers"
            ),
        ),
        FridrichSchoolCanonicalPatternRow(
            pattern_id="efficientnet_steganalysis_surgery",
            upstream_source="github.com/DDELab/deepsteganalysis (PyTorch Lightning)",
            upstream_author="DDELab (Yousfi-Fridrich group)",
            upstream_paper_or_repo="https://github.com/DDELab/deepsteganalysis",
            last_updated_or_published="2025-05-01",
            tac_module=(
                "tac.composition.fridrich_school_inverse_steganalysis_patterns."
                "canonical_efficientnet_steganalysis_surgery"
            ),
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:detector-surgery-to-attack-vector",
                "problem_space": "DOCUMENTED-ADAPTATION:5-class-softmax-to-per-pair-score",
                "math": "HARD-EARNED-1:1:stride-preservation-Nyquist-decimation",
                "data": "DOCUMENTED-ADAPTATION:256x256-JPEG-to-contest-resolution",
                "video": "DOCUMENTED-ADAPTATION:single-image-to-per-pair",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 11 (NEW; canonical ATTACK-VECTOR taxonomy at the "
                "scorer-architecture surface; the canonical contest SegNet "
                "blind spot below (256,192) is Slot RR pose-axis null "
                "projection canonical target per Slot GGG empirical anchor)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.005, +0.000] FORMALIZATION_PENDING per Catalog #344; "
                "the EV requires per-substrate empirical anchor on whether "
                "perturbations restricted to the stride-2 blind spot can "
                "lower SegNet d_seg without raising other axes"
            ),
        ),
        FridrichSchoolCanonicalPatternRow(
            pattern_id="onehot_jpeg_steganalysis",
            upstream_source="github.com/YassineYousfi/OneHotConv (LitModel.py + SRNet.py)",
            upstream_author="Yousfi-Fridrich",
            upstream_paper_or_repo=(
                "doi:10.1109/LSP.2020.2993959 (IEEE Signal Processing Letters "
                "vol 27 pp 830-834)"
            ),
            last_updated_or_published="2021-06-17",
            tac_module=(
                "tac.composition.fridrich_school_inverse_steganalysis_patterns."
                "canonical_onehot_jpeg_steganalysis"
            ),
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:DCT-to-YUV6-pixel",
                "problem_space": "DOCUMENTED-ADAPTATION:2-class-to-per-pair-score",
                "math": "HARD-EARNED-1:1:one-hot-encoding-Yousfi-Fridrich-2020-Section-III",
                "data": "DOCUMENTED-ADAPTATION:BOSSbase-JPEG-to-upstream-videos",
                "video": "DOCUMENTED-ADAPTATION:single-image-to-per-pair",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 12 (NEW; canonical input-range-dominates-signal "
                "RESOLUTION; sister of CLAUDE.md 'Fridrich inverse steganalysis' "
                "item 4 CNN blind spots)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.003, +0.000] FORMALIZATION_PENDING per Catalog #344; "
                "memory cost ~2049x; only viable for small spatial dims or "
                "sparse representations; per-substrate anchor required"
            ),
        ),
        FridrichSchoolCanonicalPatternRow(
            pattern_id="comma10k_baseline_lineage",
            upstream_source="github.com/YassineYousfi/comma10k-baseline (LitModel + train script)",
            upstream_author="Yousfi (canonical road-segmentation baseline)",
            upstream_paper_or_repo="https://github.com/YassineYousfi/comma10k-baseline",
            last_updated_or_published="2023-07-06",
            tac_module=(
                "tac.composition.fridrich_school_inverse_steganalysis_patterns."
                "canonical_comma10k_baseline_lineage"
            ),
            five_axis_classification={
                "contest": "HARD-EARNED-1:1:contest-native-resolution-874x1164",
                "problem_space": "HARD-EARNED-1:1:5-class-semantic-segmentation",
                "math": "HARD-EARNED-1:1:smp-Unet-framework",
                "data": "DOCUMENTED-ADAPTATION:comma10k-to-contest-video",
                "video": "DOCUMENTED-ADAPTATION:per-frame-to-per-pair",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 0 (foundational; this is the DIRECT ANCESTRY of the "
                "contest SegNet; every substrate attacker MUST understand the "
                "Yousfi-comma transfer to avoid cargo-culting incompatible "
                "framework choices per Catalog #290 UNIQUE-AND-COMPLETE-PER-METHOD)"
            ),
            pr_score_lowering_ev_estimate=(
                "[+0.000, +0.000] INFRASTRUCTURE-ONLY per Catalog #344; "
                "this canonical helper is operator-facing knowledge transfer "
                "not direct score lowering; sister of CLAUDE.md 'Exact scorer "
                "architectures' SegNet documentation"
            ),
        ),
        FridrichSchoolCanonicalPatternRow(
            pattern_id="syndrome_trellis_coding_filler",
            upstream_source=(
                "Filler-Judas-Fridrich 2011 IEEE TIFS canonical paper + "
                "http://dde.binghamton.edu/download/syndrome/ canonical C++/Matlab"
            ),
            upstream_author="Filler-Judas-Fridrich (Tomas Filler PhD with Fridrich)",
            upstream_paper_or_repo=(
                "Filler T., Judas J., Fridrich J. 'Minimizing Additive "
                "Distortion in Steganography Using Syndrome-Trellis Codes', "
                "IEEE TIFS vol 6 no 3 pp 920-935 2011"
            ),
            last_updated_or_published="2011 (canonical paper)",
            tac_module=(
                "tac.composition.fridrich_school_inverse_steganalysis_patterns."
                "canonical_syndrome_trellis_coding_filler"
            ),
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:steganographic-embedding-to-perturbation-allocation",
                "problem_space": "DOCUMENTED-ADAPTATION:minimize-distortion-to-minimize-rate",
                "math": "HARD-EARNED-1:1:Viterbi-trellis-DP-Filler-2011",
                "data": "DOCUMENTED-ADAPTATION:BOSSbase-to-per-pair-latent",
                "video": "DOCUMENTED-ADAPTATION:per-image-to-per-pair",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 13 (NEW; canonical OPTIMAL embedding-cost-allocation "
                "algorithm; sister of Slot FF UNIWARD + Slot YY HILL + Slot "
                "AAA MiPOD + Slot CCC HUGO which provide the cost rho_i; STC "
                "provides the OPTIMAL allocation given rho_i)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.005, -0.001] FORMALIZATION_PENDING per Catalog #344; "
                "STC gets within 1-3% of rate-distortion bound; canonical "
                "cost-function-to-archive-bytes optimal allocation primitive; "
                "compounds with Slot FF/YY/AAA/CCC + canonical equation "
                "dykstra_pareto_polytope_intersection_compounding_v1 per Catalog #372"
            ),
        ),
        FridrichSchoolCanonicalPatternRow(
            pattern_id="fusion_detector_ensemble",
            upstream_source=(
                "github.com/YassineYousfi/autostego/blob/eve/steganalysis/fusion.py + "
                "eve.py run_fusion_detector"
            ),
            upstream_author="Yousfi (canonical multi-detector ensemble)",
            upstream_paper_or_repo="https://github.com/YassineYousfi/autostego",
            last_updated_or_published="2026-03-11",
            tac_module=(
                "tac.composition.fridrich_school_inverse_steganalysis_patterns."
                "canonical_fusion_detector_ensemble"
            ),
            five_axis_classification={
                "contest": "HARD-EARNED-1:1:linear-weighted-fusion-matches-contest-scoring",
                "problem_space": "DOCUMENTED-ADAPTATION:detector-ensemble-to-axis-score-fusion",
                "math": "HARD-EARNED-1:1:linear-weighted-sum-100-d_seg-sqrt-10-d_pose-25-rate",
                "data": "DOCUMENTED-ADAPTATION:pre-trained-detector-scores-to-live-scorer-outputs",
                "video": "DOCUMENTED-ADAPTATION:per-image-to-per-pair",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 14 (NEW; canonical SCORE-LEVEL FUSION; sister of "
                "Catalog #379 canonical 3-metric trichotomy orthogonality + "
                "CLAUDE.md 'Exact scorer architectures' canonical 100*d_seg "
                "+ sqrt(10*d_pose) + 25*rate weighting)"
            ),
            pr_score_lowering_ev_estimate=(
                "[+0.000, +0.000] INFRASTRUCTURE-ONLY per Catalog #344; "
                "this is canonical fusion math not direct score; ensures every "
                "downstream Slot FF/YY/AAA/CCC trainer uses the canonical "
                "contest score weighting and does not silently re-invent it"
            ),
        ),
    )
