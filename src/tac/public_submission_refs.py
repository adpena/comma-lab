# SPDX-License-Identifier: MIT
"""Canonical public submission references for contest provenance.

These identifiers are metadata only. They let training, analysis, deploy, and
release tooling cite public pull requests without turning PR numbers into
runtime behavior branches or score claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

CONTEST_REPO = "commaai/comma_video_compression_challenge"
CONTEST_PULL_URL = "https://github.com/commaai/comma_video_compression_challenge/pull/{number}"


@dataclass(frozen=True)
class PublicSubmissionRef:
    key: str
    pr_number: int
    family: str
    role: str
    url: str


PUBLIC_SUBMISSION_REFS: dict[str, PublicSubmissionRef] = {
    "PR85": PublicSubmissionRef(
        key="PR85",
        pr_number=85,
        family="adaptive_masking_joint_frame_model",
        role="current contest-faithful exact-eval anchor family",
        url=CONTEST_PULL_URL.format(number=85),
    ),
    "PR86": PublicSubmissionRef(
        key="PR86",
        pr_number=86,
        family="hpac_token_entropy_model",
        role="HPAC source-code and probability-contract reference",
        url=CONTEST_PULL_URL.format(number=86),
    ),
    "PR90": PublicSubmissionRef(
        key="PR90",
        pr_number=90,
        family="stbm1br_lossless_mask_recode",
        role="STBM1BR lossless mask-recode reference",
        url=CONTEST_PULL_URL.format(number=90),
    ),
    "PR91": PublicSubmissionRef(
        key="PR91",
        pr_number=91,
        family="hpm1_hpac_hybrid_mask",
        role="HPM1 mask-only byte target and fail-closed parity target",
        url=CONTEST_PULL_URL.format(number=91),
    ),
    "PR92": PublicSubmissionRef(
        key="PR92",
        pr_number=92,
        family="rmb1_randomized_mask_byte_recode",
        role="RMB1 pure-rate recode reference and replay-runtime target",
        url=CONTEST_PULL_URL.format(number=92),
    ),
    "PR94": PublicSubmissionRef(
        key="PR94",
        pr_number=94,
        family="mps_only_public_floor_forensics",
        role="MPS/static-score forensic reference; not promotion evidence",
        url=CONTEST_PULL_URL.format(number=94),
    ),
    "PR95": PublicSubmissionRef(
        key="PR95",
        pr_number=95,
        family="hnerv_muon_single_member_codec",
        role="HNeRV/Muon public frontier and exact-replay runtime reference",
        url=CONTEST_PULL_URL.format(number=95),
    ),
    "PR96": PublicSubmissionRef(
        key="PR96",
        pr_number=96,
        family="latest_public_frontier_forensics",
        role="latest public frontier intake target pending local exact replay",
        url=CONTEST_PULL_URL.format(number=96),
    ),
    "PR97": PublicSubmissionRef(
        key="PR97",
        pr_number=97,
        family="vibe_coder_final_boss_h3_sidecar",
        role="latest public frontier intake target pending local exact replay",
        url=CONTEST_PULL_URL.format(number=97),
    ),
    "PR98": PublicSubmissionRef(
        key="PR98",
        pr_number=98,
        family="hnerv_muon_finetuned_from_pr95",
        role="HNeRV/Muon QAT-finetuned public frontier exact-replay target",
        url=CONTEST_PULL_URL.format(number=98),
    ),
    "PR99": PublicSubmissionRef(
        key="PR99",
        pr_number=99,
        family="hnerv_muon_lc_latent_correction",
        role="HNeRV/Muon latent-correction public frontier exact-replay target",
        url=CONTEST_PULL_URL.format(number=99),
    ),
    "PR100": PublicSubmissionRef(
        key="PR100",
        pr_number=100,
        family="hnerv_muon_lc_v2_latent_correction",
        role="HNeRV/Muon LC-v2 public frontier exact-replay target",
        url=CONTEST_PULL_URL.format(number=100),
    ),
    "PR101": PublicSubmissionRef(
        key="PR101",
        pr_number=101,
        family="hnerv_ft_microcodec",
        role="late HNeRV microcodec public frontier exact-replay target",
        url=CONTEST_PULL_URL.format(number=101),
    ),
    "PR102": PublicSubmissionRef(
        key="PR102",
        pr_number=102,
        family="hnerv_lc_v2_scale095_rplus1",
        role="late HNeRV LC-v2 scaled public frontier exact-replay target",
        url=CONTEST_PULL_URL.format(number=102),
    ),
    "PR103": PublicSubmissionRef(
        key="PR103",
        pr_number=103,
        family="hnerv_lc_arithmetic_coding",
        role="late HNeRV latent-correction arithmetic-code exact-replay target",
        url=CONTEST_PULL_URL.format(number=103),
    ),
    "PR104": PublicSubmissionRef(
        key="PR104",
        pr_number=104,
        family="qhnerv_ft_best",
        role="late quantized HNeRV fine-tune exact-replay target",
        url=CONTEST_PULL_URL.format(number=104),
    ),
    "PR105": PublicSubmissionRef(
        key="PR105",
        pr_number=105,
        family="kitchen_sink_hnerv_training_pipeline",
        role="late kitchen-sink HNeRV pipeline exact-replay and source-study target",
        url=CONTEST_PULL_URL.format(number=105),
    ),
    "PR106": PublicSubmissionRef(
        key="PR106",
        pr_number=106,
        family="belt_and_suspenders_hnerv_training_pipeline",
        role="late belt-and-suspenders HNeRV pipeline exact-replay and source-study target",
        url=CONTEST_PULL_URL.format(number=106),
    ),
}


def normalize_public_pr_key(value: str) -> str:
    text = str(value).strip().upper()
    if text.isdigit():
        text = f"PR{text}"
    if not (text.startswith("PR") and text[2:].isdigit()):
        raise ValueError(f"unsupported public PR reference: {value!r}")
    if text not in PUBLIC_SUBMISSION_REFS:
        raise ValueError(
            f"unknown public PR reference {text}; supported: "
            + ",".join(sorted(PUBLIC_SUBMISSION_REFS))
        )
    return text


def public_submission_refs_for_manifest(values: Iterable[str]) -> dict[str, dict[str, object]]:
    normalized = [normalize_public_pr_key(value) for value in values if str(value).strip()]
    return {key: asdict(PUBLIC_SUBMISSION_REFS[key]) for key in dict.fromkeys(normalized)}


def parse_public_pr_refs_csv(value: str | None) -> dict[str, dict[str, object]]:
    if value is None or not str(value).strip():
        return {}
    return public_submission_refs_for_manifest(value.split(","))
