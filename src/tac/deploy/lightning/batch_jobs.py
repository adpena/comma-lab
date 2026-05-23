# SPDX-License-Identifier: MIT
"""Auditable Lightning Batch Jobs helpers.

This module is intentionally separate from ``lightning_dispatch.py``. The
existing dispatcher talks to a mutable Studio over SSH/tmux; this module wraps
the official Lightning SDK ``Job.run`` API so promotion-grade runs can have a
job record, snapshot path, artifact path, immutable command, and local state.

Score authority still comes only from the job's ``contest_auth_eval.json``.
No human logs are parsed for scores here.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Lightning tooling is POSIX in practice.
    fcntl = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[4]
LIGHTNING_BATCH_STATE = REPO_ROOT / ".omx/state/lightning_batch_jobs.json"
ARTIFACT_METADATA = "lightning_queue_metadata.json"
ARTIFACT_ARCHIVE = "archive.zip"
ARTIFACT_VALIDATION = "lightning_artifact_mirror_validation.json"
ARTIFACT_RUNNER_PREFLIGHT = "lightning_runner_preflight.json"
ARTIFACT_INFLATE_RUNTIME_BOOTSTRAP = "lightning_inflate_runtime_bootstrap.json"
ARTIFACT_INFLATE_RUNTIME_STATIC_PREFLIGHT = "lightning_inflate_runtime_static_preflight.json"
ARTIFACT_DALI_BOOTSTRAP = "lightning_dali_bootstrap.json"
ARTIFACT_DALI_REQUIREMENTS = "lightning_dali_requirements.txt"
ARTIFACT_SUPPLY_CHAIN_SCAN_PRE = "lightning_supply_chain_scan_pre.json"
ARTIFACT_SUPPLY_CHAIN_SCAN = "lightning_supply_chain_scan.json"
ARTIFACT_COMPONENT_RESPONSE_INPUTS = "official_component_response_inputs.json"
ARTIFACT_COMPONENT_RESPONSE_SUMMARY = "official_component_response_summary.json"
ARTIFACT_COMPONENT_RESPONSE_VALIDATION = "official_component_response_artifact_validation.json"
ARTIFACT_COMPONENT_RESPONSE_LOG = "official_component_response.log"
ARTIFACT_COMPONENT_SENSITIVITY_INPUTS = "diagnostic_component_sensitivity_inputs.json"
ARTIFACT_COMPONENT_SENSITIVITY_RUN = "diagnostic_component_sensitivity_run.json"
ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION = "diagnostic_component_sensitivity_artifact_validation.json"
ARTIFACT_COMPONENT_SENSITIVITY_LOG = "diagnostic_component_sensitivity.log"
ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY = "component_sensitivity_profile_summary.json"
ARTIFACT_COMPONENT_TRACE = "component_trace.json"
ARTIFACT_COMPONENT_TRACE_LOG = "component_trace.log"
ARTIFACT_COMPONENT_TRACE_STATUS = "component_trace_status.json"
ARTIFACT_INFRA_FAILURE = "lightning_artifact_infra_failure.json"
LIGHTNING_EMPTY_ARTIFACT_INFRA_TERMINAL_CLASS = "empty_lightning_artifact_dir_infra"
LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS = "exact_eval_missing_score_json"
COMPONENT_RESPONSE_COMPONENTS = ("posenet", "segnet", "combined")
COMPONENT_RESPONSE_CURVE_FILES = tuple(
    f"{component}_official_response_curve.json"
    for component in COMPONENT_RESPONSE_COMPONENTS
)
COMPONENT_SENSITIVITY_MAP_FILES = tuple(
    f"{component}_sensitivity_map.pt"
    for component in COMPONENT_RESPONSE_COMPONENTS
)
COMPONENT_SENSITIVITY_HOLDOUT_MAP_FILES = tuple(
    f"{component}_holdout_sensitivity_map.pt"
    for component in COMPONENT_RESPONSE_COMPONENTS
)
COMPONENT_SENSITIVITY_CURVE_FILES = tuple(
    f"{component}_response_curve.json"
    for component in COMPONENT_RESPONSE_COMPONENTS
)
COMPONENT_SENSITIVITY_PROFILE_FILES = (
    ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY,
    "sample_plan.json",
    "stability.json",
    "perturbation_basis_v1.json",
    *COMPONENT_SENSITIVITY_MAP_FILES,
    *COMPONENT_SENSITIVITY_HOLDOUT_MAP_FILES,
    *COMPONENT_SENSITIVITY_CURVE_FILES,
)
COMPONENT_SENSITIVITY_DIAGNOSTIC_SOURCES = {
    "fisher_proxy",
    "direct_renderer_cuda_finite_difference_component_response",
}
CANONICAL_ARTIFACT_FILES = (
    ARTIFACT_METADATA,
    "contest_auth_eval.json",
    ARTIFACT_ARCHIVE,
    "eval_provenance.json",
    "report.txt",
    ARTIFACT_DALI_BOOTSTRAP,
    ARTIFACT_DALI_REQUIREMENTS,
    ARTIFACT_RUNNER_PREFLIGHT,
    ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
    ARTIFACT_SUPPLY_CHAIN_SCAN,
    "adjudication_provenance.json",
    "contest_auth_eval.adjudicated.json",
)
CANONICAL_CPU_EXACT_EVAL_ARTIFACT_FILES = tuple(
    name
    for name in CANONICAL_ARTIFACT_FILES
    if name not in {ARTIFACT_DALI_BOOTSTRAP, ARTIFACT_DALI_REQUIREMENTS}
)
EXACT_AUTH_EVAL_ROLES = frozenset({"exact_cuda_eval", "exact_cpu_eval"})
LIGHTNING_STUDIO_MACHINE_CLASS_PAIRS = {
    "g4dn.xlarge": "T4_SMALL",
    "g4dn.2xlarge": "T4",
    "g4dn.12xlarge": "T4_X_4",
    "g6e.4xlarge": "L40S",
    "g7e.4xlarge": "RTXP_6000",
    "g7e.12xlarge": "RTXP_6000_X_2",
}
LIGHTNING_STUDIO_SYMBOLIC_MACHINE_CLASSES = frozenset(
    {"T4_SMALL", "T4", "T4_X_4"}
)
LIGHTNING_STUDIO_SYMBOLIC_MACHINE_SUGGESTIONS = {
    "H100": "use a concrete Studio-compatible provider class, or submit an image-backed job on a matching cloud account",
    "H200": "use a concrete Studio-compatible provider class, or submit an image-backed job on a matching cloud account",
    "A100": "use a concrete Studio-compatible provider class, or submit an image-backed job on a matching cloud account",
    "A100_SXM4": "use a concrete Studio-compatible provider class, or submit an image-backed job on a matching cloud account",
    "L40S": "use g6e.4xlarge for the current Studio-backed AWS L40S route",
    "RTXP_6000": "use g7e.4xlarge for the current Studio-backed RTX PRO route",
    "RTXP_6000_X_2": "use g7e.12xlarge for the current Studio-backed dual RTX PRO route",
}
_LIGHTNING_STUDIO_PROVIDER_MACHINE_RE = re.compile(r"^[a-z][a-z0-9-]*\.[a-z0-9]+$")
LIGHTNING_STUDIO_CLOUD_ACCOUNT_MISMATCH_FRAGMENT = (
    "Studio cloud account does not match provided cloud account"
)
LIGHTNING_STUDIO_CLOUD_ACCOUNT_MISMATCH_TERMINAL_CLASS = (
    "studio_cloud_account_namespace_mismatch"
)


class LightningStudioCloudAccountMismatchError(RuntimeError):
    """Project diagnostic for Studio/env cloud-account namespace mismatches."""

    terminal_class = LIGHTNING_STUDIO_CLOUD_ACCOUNT_MISMATCH_TERMINAL_CLASS

    def __init__(
        self,
        *,
        job_name: str,
        studio: str | None,
        cloud_account: str | None,
        machine: str,
        original_error_type: str,
        original_message: str,
    ) -> None:
        self.job_name = job_name
        self.studio = studio
        self.cloud_account = cloud_account
        self.machine = machine
        self.original_error_type = original_error_type
        self.original_message = original_message
        super().__init__(
            "Lightning Studio submit blocked: terminal_class="
            f"{self.terminal_class}; job_name={job_name!r}; studio={studio!r}; "
            f"machine={machine!r}; cloud_account={cloud_account!r}. "
            "The selected Studio environment belongs to a different Lightning "
            "cloud-account namespace than the explicit --cloud-account route. "
            "Do not retry the same Studio-backed submit; use a Studio/env in "
            "that cloud account, omit --cloud-account for the default Studio "
            "account route, or submit an image-backed job on the target cloud "
            f"account. SDK said {original_error_type}: {original_message}"
        )


def _is_studio_cloud_account_mismatch(exc: BaseException) -> bool:
    return LIGHTNING_STUDIO_CLOUD_ACCOUNT_MISMATCH_FRAGMENT in str(exc)


def _supported_studio_machine_pairs_text() -> str:
    return ", ".join(
        f"{machine}/{machine_class}"
        for machine, machine_class in sorted(LIGHTNING_STUDIO_MACHINE_CLASS_PAIRS.items())
    )


def validate_studio_machine_class_pair(machine: str, *, cloud_account: str | None = None) -> None:
    """Fail closed on Studio machine classes known not to exist for this workflow."""

    del cloud_account  # reserved for future cloud-account-specific allowlists
    value = str(machine or "").strip()
    if not value:
        raise ValueError("Lightning Studio machine is required")
    key = value.upper()
    if key in LIGHTNING_STUDIO_SYMBOLIC_MACHINE_CLASSES:
        return
    suggestion = LIGHTNING_STUDIO_SYMBOLIC_MACHINE_SUGGESTIONS.get(key)
    if suggestion:
        raise ValueError(
            "unsupported symbolic Lightning Studio accelerator "
            f"{value!r}: {suggestion}"
        )
    if value.lower() in LIGHTNING_STUDIO_MACHINE_CLASS_PAIRS:
        return
    if _LIGHTNING_STUDIO_PROVIDER_MACHINE_RE.fullmatch(value.lower()):
        raise ValueError(
            "unsupported Lightning Studio machine/class pair: "
            f"{value!r}. Supported Studio pairs: {_supported_studio_machine_pairs_text()}"
        )
OPTIONAL_ARTIFACT_FILES = (
    "auth_eval.log",
    "adjudication.log",
    ARTIFACT_INFLATE_RUNTIME_BOOTSTRAP,
    ARTIFACT_INFLATE_RUNTIME_STATIC_PREFLIGHT,
    ARTIFACT_COMPONENT_TRACE,
    ARTIFACT_COMPONENT_TRACE_LOG,
    ARTIFACT_COMPONENT_TRACE_STATUS,
)
SSH_AUTH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "KbdInteractiveAuthentication=no",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "TCPKeepAlive=yes",
    "-o",
    "ConnectionAttempts=3",
)
COMPONENT_RESPONSE_CANONICAL_ARTIFACT_FILES = (
    ARTIFACT_METADATA,
    ARTIFACT_COMPONENT_RESPONSE_INPUTS,
    ARTIFACT_COMPONENT_RESPONSE_SUMMARY,
    *COMPONENT_RESPONSE_CURVE_FILES,
    ARTIFACT_DALI_BOOTSTRAP,
    ARTIFACT_DALI_REQUIREMENTS,
    ARTIFACT_RUNNER_PREFLIGHT,
    ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
    ARTIFACT_SUPPLY_CHAIN_SCAN,
)
COMPONENT_RESPONSE_OPTIONAL_ARTIFACT_FILES = (
    ARTIFACT_COMPONENT_RESPONSE_LOG,
    ARTIFACT_COMPONENT_RESPONSE_VALIDATION,
)
COMPONENT_SENSITIVITY_CANONICAL_ARTIFACT_FILES = (
    ARTIFACT_METADATA,
    ARTIFACT_COMPONENT_SENSITIVITY_INPUTS,
    ARTIFACT_COMPONENT_SENSITIVITY_RUN,
    *COMPONENT_SENSITIVITY_PROFILE_FILES,
    ARTIFACT_DALI_BOOTSTRAP,
    ARTIFACT_DALI_REQUIREMENTS,
    ARTIFACT_RUNNER_PREFLIGHT,
    ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
    ARTIFACT_SUPPLY_CHAIN_SCAN,
)
COMPONENT_SENSITIVITY_OPTIONAL_ARTIFACT_FILES = (
    ARTIFACT_COMPONENT_SENSITIVITY_LOG,
    ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION,
)
COMPONENT_RESPONSE_EVAL_EVIDENCE_NAMES = {
    "contest_auth_eval.json",
    "provenance.json",
    "report.txt",
    "contest_auth_eval.stdout.log",
    "contest_auth_eval.stderr.log",
}
DALI_BOOTSTRAP_VERSION = "1.52.0"
DALI_BOOTSTRAP_WHEELS = {
    "common": [
        {
            "name": "absl-py",
            "version": "2.3.1",
            "url": "https://files.pythonhosted.org/packages/8f/aa/ba0014cc4659328dc818a28827be78e6d97312ab0cb98105a770924dc11e/absl_py-2.3.1-py3-none-any.whl",
            "sha256": "eeecf07f0c2a93ace0772c92e596ace6d3d3996c042b2128459aaae2a76de11d",
        },
        {
            "name": "astunparse",
            "version": "1.6.3",
            "url": "https://files.pythonhosted.org/packages/2b/03/13dde6512ad7b4557eb792fbcf0c653af6076b81e5941d36ec61f7ce6028/astunparse-1.6.3-py2.py3-none-any.whl",
            "sha256": "c2652417f2c8b5bb325c885ae329bdf3f86424075c4fd1a128674bc6fba4b8e8",
        },
        {
            "name": "attrs",
            "version": "25.4.0",
            "url": "https://files.pythonhosted.org/packages/3a/2a/7cc015f5b9f5db42b7d48157e23356022889fc354a2813c15934b7cb5c0e/attrs-25.4.0-py3-none-any.whl",
            "sha256": "adcf7e2a1fb3b36ac48d97835bb6d8ade15b8dcce26aba8bf1d14847b57a3373",
        },
        {
            "name": "gast",
            "version": "0.6.0",
            "url": "https://files.pythonhosted.org/packages/a3/61/8001b38461d751cd1a0c3a6ae84346796a5758123f3ed97a1b121dfbf4f3/gast-0.6.0-py3-none-any.whl",
            "sha256": "52b182313f7330389f72b069ba00f174cfe2a06411099547288839c6cbafbd54",
        },
        {
            "name": "makefun",
            "version": "1.16.0",
            "url": "https://files.pythonhosted.org/packages/b7/c0/4bc973defd1270b89ccaae04cef0d5fa3ea85b59b108ad2c08aeea9afb76/makefun-1.16.0-py2.py3-none-any.whl",
            "sha256": "43baa4c3e7ae2b17de9ceac20b669e9a67ceeadff31581007cca20a07bbe42c4",
        },
        {
            "name": "packaging",
            "version": "25.0",
            "url": "https://files.pythonhosted.org/packages/20/12/38679034af332785aac8774540895e234f4d07f7545804097de4b666afd8/packaging-25.0-py3-none-any.whl",
            "sha256": "29572ef2b1f17581046b3a2227d5c611fb25ec70ca1ba8554b24b0e69331a484",
        },
        {
            "name": "six",
            "version": "1.17.0",
            "url": "https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl",
            "sha256": "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
        },
        {
            "name": "wheel",
            "version": "0.45.1",
            "url": "https://files.pythonhosted.org/packages/0b/2c/87f3254fd8ffd29e4c02732eee68a83a1d3c346ae39bc6822dcbcb697f2b/wheel-0.45.1-py3-none-any.whl",
            "sha256": "708e7481cc80179af0e556bbf0cc00b8444c7321e2700b8d8580231d13017248",
        },
    ],
    "py310": [
        {
            "name": "dm-tree",
            "version": "0.1.9",
            "url": "https://files.pythonhosted.org/packages/7c/79/ba0f7274164eb6bd06a36c2f8cb21b0debc32fd9ba8e73a7c9e50c90041b/dm_tree-0.1.9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "831699d2c60a1b38776a193b7143ae0acad0a687d87654e6d3342584166816bc",
        },
        {
            "name": "nvtx",
            "version": "0.2.13",
            "url": "https://files.pythonhosted.org/packages/34/15/0b56e9b3020613d7d167bc4cdee3ba8686f6320c6aa62e85ed17b54c4dcb/nvtx-0.2.13-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "7874534af889ab7c2c63554c73119d193d2beb7671b551b7f43de5b97ceb5971",
        },
        {
            "name": "wrapt",
            "version": "2.0.0",
            "url": "https://files.pythonhosted.org/packages/28/8d/d5df2af58ae479785473607a3b25726c295640cdcaee830847cee339eff9/wrapt-2.0.0-cp310-cp310-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
            "sha256": "b6a18c813196e18146b8d041e20875bdb0cb09b94ac1d1e1146e0fa87b2deb0d",
        },
    ],
    "py311": [
        {
            "name": "dm-tree",
            "version": "0.1.9",
            "url": "https://files.pythonhosted.org/packages/e8/46/939fbf81177c7cb3b1e5ddebd696237b3be9520769cce882f064de497103/dm_tree-0.1.9-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "294dc1cecf87552a45cdd5ddb215e7f5295a5a47c46f1f0a0463c3dd02a527d7",
        },
        {
            "name": "nvtx",
            "version": "0.2.13",
            "url": "https://files.pythonhosted.org/packages/1d/55/e1e43201959dd854005c72b8a13ec86b775c349cdcb1d23423d841bbad58/nvtx-0.2.13-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "5640ca4b8be2c19a8fc4ca8403d3c2598165ea27541940b4897138a7b0a717fe",
        },
        {
            "name": "wrapt",
            "version": "2.0.0",
            "url": "https://files.pythonhosted.org/packages/70/c3/c82263503f554715aa1847e85dc75a69631a54e9d7ab0f1a55e34a22d44a/wrapt-2.0.0-cp311-cp311-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
            "sha256": "f460e1eb8e75a17c3918c8e35ba57625721eef2439ef0bcf05304ac278a65e1d",
        },
    ],
    "py312": [
        {
            "name": "dm-tree",
            "version": "0.1.9",
            "url": "https://files.pythonhosted.org/packages/86/52/27607a275c12858b979b8e943d2bd3bd0f9028503bb7079d5830a8b3cac0/dm_tree-0.1.9-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "2334cfe9d2ed4293f9f1c7aefba0657deaab9ea74b5fadd966f6d01d9b6b42d9",
        },
        {
            "name": "nvtx",
            "version": "0.2.13",
            "url": "https://files.pythonhosted.org/packages/12/ab/762da984e7671f7c34ae87e5b70523c3eeb4563759268bfaea07c97f32a6/nvtx-0.2.13-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "453d838dd1424a04303281ee57a73e2b8dca0e03039bc609a945861b8fe7d7d9",
        },
        {
            "name": "wrapt",
            "version": "2.0.0",
            "url": "https://files.pythonhosted.org/packages/ff/0c/0f565294897a72493dbafe7b46229b5f09f3776795a894d6b737e98387de/wrapt-2.0.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
            "sha256": "43dc0550ae15e33e6bb45a82a5e1b5495be2587fbaa996244b509921810ee49f",
        },
    ],
    "py313": [
        {
            "name": "dm-tree",
            "version": "0.1.9",
            "url": "https://files.pythonhosted.org/packages/e5/0a/f4d72ffb64ab3edc1fa66261f81ee3b4142ab14cd8aa1dfc7bbeca5ee4ba/dm_tree-0.1.9-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "f68b0efad76703dd4648586c75618a48cdd671b68c3266fe980e323c15423607",
        },
        {
            "name": "nvtx",
            "version": "0.2.13",
            "url": "https://files.pythonhosted.org/packages/14/4b/21e975997def8a387543ba2bbe227551ad466781c39fc67f37f53555f37e/nvtx-0.2.13-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sha256": "edd7b729ed0211350258a21dd13422f59bc521de2b2fd21feb6c177af492f4e1",
        },
        {
            "name": "wrapt",
            "version": "2.0.0",
            "url": "https://files.pythonhosted.org/packages/e4/5f/e4eabd0cc6684c5b208c2abc5c3459449c4d15be1694a9bbcf51e0e135fd/wrapt-2.0.0-cp313-cp313-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl",
            "sha256": "db2eea83c43f84e4e41dbbb4c1de371a53166e55f900a6b130c3ef51c6345c1a",
        },
    ],
    "cu120": [
        {
            "name": "nvidia-dali-cuda120",
            "version": "1.52.0",
            "url": "https://pypi.nvidia.com/nvidia-dali-cuda120/nvidia_dali_cuda120-1.52.0-py3-none-manylinux_2_28_x86_64.whl",
            "sha256": "52310878e2c6ced901c8e9fde8f8ac79b65537abc2a290a1cbf1f53f44072206",
        },
        {
            "name": "nvidia-nvcomp-cu12",
            "version": "5.0.0.6",
            "url": "https://files.pythonhosted.org/packages/e3/b5/1c3b2493dbcd0f436715b3faba4bef4747743dbb1828ce7823e61423863c/nvidia_nvcomp_cu12-5.0.0.6-py3-none-manylinux_2_28_x86_64.whl",
            "sha256": "b5b8a9a3ed33c4a87bb9103fbe59c83b28e4559234447ad40cab11896557c031",
        },
        {
            "name": "nvidia-nvimgcodec-cu12",
            "version": "0.6.1.37",
            "url": "https://files.pythonhosted.org/packages/13/f4/36906056947347350a4b7441f7431355ff3fbc747c4ae3f9ec9d8e18cffc/nvidia_nvimgcodec_cu12-0.6.1.37-py3-none-manylinux2014_x86_64.whl",
            "sha256": "3b72bc65cfd113ef8a599082ca7f80ef53df0577f13e8489d065ba31d2df78f7",
        },
        {
            "name": "nvidia-nvjpeg",
            "version": "13.0.1.86",
            "url": "https://files.pythonhosted.org/packages/b6/e6/9b96fcead7f78a5e2d5d9e726f0d0b1517faf2895146d571e006793180d7/nvidia_nvjpeg-13.0.1.86-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            "sha256": "f658a05c48702219b3db52e194c2ea1ff0ccd3fbf2b75127b4a573074cc22fd5",
        },
        {
            "name": "nvidia-nvjpeg2k-cu12",
            "version": "0.9.0.43",
            "url": "https://files.pythonhosted.org/packages/5b/6a/45a73ce34c5a1e81ada2c125d22c0599458775aad3d90c1abc539653a34f/nvidia_nvjpeg2k_cu12-0.9.0.43-py3-none-manylinux2014_x86_64.whl",
            "sha256": "4f7a561cb74512de5832a674118e8d46c362d042917fc84b1e6d007fe5e15ab8",
        },
        {
            "name": "nvidia-nvtiff-cu12",
            "version": "0.5.1.75",
            "url": "https://files.pythonhosted.org/packages/68/7e/e0f521101ef7d4127b2da7d24904249d5b300b625b37b3a54c76cc0f0115/nvidia_nvtiff_cu12-0.5.1.75-py3-none-manylinux2014_x86_64.whl",
            "sha256": "99dffbe2eac4a34326a79cabdf24b154790f5c4a25a2fefc97f952d2fd51d9a2",
        },
    ],
    "cu130": [
        {
            "name": "nvidia-dali-cuda130",
            "version": "1.52.0",
            "url": "https://pypi.nvidia.com/nvidia-dali-cuda130/nvidia_dali_cuda130-1.52.0-py3-none-manylinux_2_28_x86_64.whl",
            "sha256": "37369fb30e9c66f710b29836688c90abc36793bbe757cd3ad699fac76ba07119",
        },
        {
            "name": "nvidia-nvcomp-cu13",
            "version": "5.0.0.6",
            "url": "https://files.pythonhosted.org/packages/56/c9/f8a1b957f949ab4c4dc29e7f56316a2224b92c78cad9a66aeab2b36f8857/nvidia_nvcomp_cu13-5.0.0.6-py3-none-manylinux_2_28_x86_64.whl",
            "sha256": "91a4e4b1dc15b0f38e54a3353c917086c99c9f415e1ad79a57d5f28d62b68a4d",
        },
        {
            "name": "nvidia-nvimgcodec-cu13",
            "version": "0.6.1.37",
            "url": "https://files.pythonhosted.org/packages/2a/86/5a89168b14c335891e446575b04bf807c23dac37509a755d032f7e62aec7/nvidia_nvimgcodec_cu13-0.6.1.37-py3-none-manylinux2014_x86_64.whl",
            "sha256": "c9cd2f917a3b4c248296c963fa3220e544c06a7d2679c82454ba0836a352981c",
        },
        {
            "name": "nvidia-nvjpeg",
            "version": "13.0.1.86",
            "url": "https://files.pythonhosted.org/packages/b6/e6/9b96fcead7f78a5e2d5d9e726f0d0b1517faf2895146d571e006793180d7/nvidia_nvjpeg-13.0.1.86-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            "sha256": "f658a05c48702219b3db52e194c2ea1ff0ccd3fbf2b75127b4a573074cc22fd5",
        },
        {
            "name": "nvidia-nvjpeg2k-cu13",
            "version": "0.9.0.43",
            "url": "https://files.pythonhosted.org/packages/ea/ed/24134897e077340eefa0e9ae61b58620b13e2f2118f896010aa7fedd6168/nvidia_nvjpeg2k_cu13-0.9.0.43-py3-none-manylinux2014_x86_64.whl",
            "sha256": "554761474cf2e0f85d04644d08e7949a4ffd2f4be10a5318de85e46f86ad8f07",
        },
        {
            "name": "nvidia-nvtiff-cu13",
            "version": "0.5.1.75",
            "url": "https://files.pythonhosted.org/packages/19/eb/20c5b6d9e60b6252b3d72857d6401b7ea94aa3a29098fc49f9916aabaacf/nvidia_nvtiff_cu13-0.5.1.75-py3-none-manylinux2014_x86_64.whl",
            "sha256": "f71ee243369e0e65d5a55ccf27eb7c76c6b77aafb7caabeb74bab5037a3d3d09",
        },
    ],
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _command_sha256(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _ensure_remote_uv_command(*, output_dir: str) -> str:
    out = _quote(output_dir)
    return "\n".join(
        [
            "if [ ! -f scripts/ensure_remote_uv.sh ]; then",
            "  echo 'FATAL: scripts/ensure_remote_uv.sh missing; remote source bundle incomplete' >&2",
            "  exit 31",
            "fi",
            f"UV_BOOTSTRAP_LOG={out}/uv_bootstrap.log",
            "export UV_BOOTSTRAP_LOG",
            "UV_BIN=$(bash scripts/ensure_remote_uv.sh --symlink-system)",
            "export UV_BIN",
            'export PATH="$(dirname "$UV_BIN"):$PATH"',
            'test -x "$UV_BIN"',
        ]
    )


def lightning_sdk_job_name(name: str) -> str:
    """Return the path/name form Lightning SDK uses for Job artifacts."""

    return name.replace("_", "-").lower()


def lightning_sdk_artifact_path(name: str) -> str:
    """Return the SDK-reported artifact view for a submitted Studio job."""

    return f"/teamspace/jobs/{lightning_sdk_job_name(name)}/artifacts"


def lightning_sdk_persisted_studio_output_dir(
    *,
    sdk_artifact_path: str,
    remote_output_dir: str,
) -> str | None:
    """Map a Studio path to the SDK artifact mirror path when possible."""

    prefix = "/teamspace/studios/this_studio/"
    remote = str(remote_output_dir).rstrip("/")
    if not remote.startswith(prefix):
        return None
    rel = remote[len(prefix):].strip("/")
    if not rel:
        return None
    return f"{str(sdk_artifact_path).rstrip('/')}/{rel}"


def default_exact_eval_output_dir(*, repo_dir: str, job_name: str) -> str:
    """Return a writable default output directory inside the Studio workspace.

    Lightning SDK reports Studio job artifacts under ``/teamspace/jobs/...``,
    but live r2 evidence showed that path is read-only inside the running job.
    Write into the Studio workspace instead; mirror/harvest can validate the
    resulting JSON/artifacts after the job exits.
    """

    repo = str(repo_dir).rstrip("/")
    if not repo:
        raise ValueError("repo_dir is required for default exact-eval output dir")
    return f"{repo}/experiments/results/lightning_batch/{job_name}"


def default_exact_eval_local_artifact_dir(*, job_name: str) -> str:
    """Return the deterministic local mirror directory for exact eval artifacts."""

    name = str(job_name).strip()
    if not name:
        raise ValueError("job_name is required for default exact-eval local artifact dir")
    return f"experiments/results/lightning_batch/{name}"


def _validate_writable_output_dir(output_dir: str | Path) -> None:
    out = str(output_dir).rstrip("/")
    if not out:
        raise ValueError("exact eval output_dir is required")
    if out == "/teamspace/jobs" or out.startswith("/teamspace/jobs/"):
        raise ValueError(
            "Lightning exact eval output_dir must be writable; /teamspace/jobs/... "
            "is the SDK artifact view and is read-only inside Studio jobs"
        )


def _validate_sha256(value: str | None, *, field: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise ValueError(f"{field} must be a 64-character hex SHA-256")


def _validate_expected_archive(expected_sha256: str | None, expected_size_bytes: int | None) -> None:
    _validate_sha256(expected_sha256, field="expected_archive_sha256")
    if expected_size_bytes is not None and expected_size_bytes <= 0:
        raise ValueError("expected_archive_size_bytes must be positive")
    if (expected_sha256 is None) != (expected_size_bytes is None):
        raise ValueError(
            "expected_archive_sha256 and expected_archive_size_bytes must be provided together"
        )


def _require_expected_archive(expected_sha256: str | None, expected_size_bytes: int | None) -> None:
    _validate_expected_archive(expected_sha256, expected_size_bytes)
    if expected_sha256 is None or expected_size_bytes is None:
        raise ValueError(
            "exact eval jobs require expected_archive_sha256 and expected_archive_size_bytes; "
            "use --infer-expected-archive or pass the archive identity explicitly"
        )


def _normalise_heredoc_marker(fragment: str) -> tuple[str | None, bool]:
    """Return a shell heredoc marker from a ``<<...`` fragment."""

    if fragment.startswith("<<-"):
        raw = fragment[3:].strip().split(None, 1)[0] if fragment[3:].strip() else ""
        strip_tabs = True
    elif fragment.startswith("<<"):
        raw = fragment[2:].strip().split(None, 1)[0] if fragment[2:].strip() else ""
        strip_tabs = False
    else:
        return None, False
    if not raw:
        return None, strip_tabs
    if (raw[0], raw[-1:]) in {("'", "'"), ('"', '"')} and len(raw) >= 2:
        raw = raw[1:-1]
    return raw or None, strip_tabs


def _python_stdin_heredoc_prefix(prefix: str) -> bool:
    tokens = prefix.strip().split()
    command_index = None
    for idx, token in enumerate(tokens):
        if re.search(r"(?:^|/)python(?:\d+(?:\.\d+)?)?$", token):
            command_index = idx
            break
    if command_index is None:
        return False
    command = tokens[command_index].rsplit("/", 1)[-1]
    if command not in {"python", "python3"} and not command.startswith("python3."):
        return False
    return "-" in tokens[command_index + 1 :]


def _validate_python_stdin_heredocs(command: str, *, job_name: str) -> None:
    """Compile Python code embedded via ``python - <<'PY'`` before dispatch.

    ``bash -n`` does not parse heredoc bodies. A malformed generic Lightning
    training command can therefore pass shell syntax checks and fail only after
    paid GPU allocation. Compile stdin Python heredocs locally so those jobs
    fail before provider submission.
    """

    lines = command.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        marker_at = line.find("<<")
        if marker_at < 0:
            idx += 1
            continue
        prefix = line[:marker_at]
        marker, strip_tabs = _normalise_heredoc_marker(line[marker_at:])
        if marker is None or not _python_stdin_heredoc_prefix(prefix):
            idx += 1
            continue
        body_start = idx + 1
        body_end = body_start
        while body_end < len(lines):
            candidate = lines[body_end].lstrip("\t") if strip_tabs else lines[body_end]
            if candidate.strip() == marker:
                break
            body_end += 1
        if body_end >= len(lines):
            raise ValueError(f"{job_name}: unterminated Python heredoc marker {marker!r}")
        source = "\n".join(lines[body_start:body_end]) + "\n"
        try:
            compile(source, f"<LightningBatchJobSpec {job_name} heredoc {marker}>", "exec")
        except SyntaxError as exc:
            raise ValueError(
                f"{job_name}: embedded Python heredoc {marker!r} fails local compile: {exc.msg}"
            ) from exc
        idx = body_end + 1


def _validate_optional_number(
    value: float | None,
    *,
    field: str,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"adjudication {field} must be numeric")
    finite = value == value and abs(value) != float("inf")
    if not finite:
        raise ValueError(f"adjudication {field} must be finite")
    if minimum is not None:
        if strict_minimum and value <= minimum:
            raise ValueError(f"adjudication {field} must be > {minimum}")
        if not strict_minimum and value < minimum:
            raise ValueError(f"adjudication {field} must be >= {minimum}")


def _normalise_eval_device(eval_device: str) -> str:
    device = str(eval_device).strip().lower()
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"exact eval device must be 'cuda' or 'cpu', got {eval_device!r}")
    return device


def _exact_eval_role(eval_device: str) -> str:
    return f"exact_{_normalise_eval_device(eval_device)}_eval"


def _exact_eval_axis(eval_device: str) -> str:
    return f"contest_{_normalise_eval_device(eval_device)}"


def _exact_eval_score_tag(eval_device: str) -> str:
    return "[contest-CUDA]" if _normalise_eval_device(eval_device) == "cuda" else "[contest-CPU]"


def _exact_eval_artifact_files_for_role(role: object) -> tuple[str, ...]:
    return (
        CANONICAL_CPU_EXACT_EVAL_ARTIFACT_FILES
        if role == "exact_cpu_eval"
        else CANONICAL_ARTIFACT_FILES
    )


def _normalise_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    out: dict[str, Any] = {}
    for key, value in sorted(metadata.items()):
        if not isinstance(key, str) or not key.strip():
            raise ValueError("queue metadata keys must be non-empty strings")
        try:
            json.dumps(value)
        except TypeError as exc:
            raise ValueError(f"queue metadata {key!r} is not JSON serializable") from exc
        out[key] = value
    return out


def archive_identity(path: str | Path) -> dict[str, Any]:
    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    return {
        "archive_sha256": _sha256(archive),
        "archive_size_bytes": archive.stat().st_size,
    }


def _require_finite_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"contest_auth_eval.json missing numeric {key!r}")
    out = float(value)
    if not (out == out and abs(out) != float("inf")):
        raise ValueError(f"contest_auth_eval.json {key!r} is not finite: {value!r}")
    return out


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json_replace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    path.chmod(0o644)


def _validate_supply_chain_scan_artifact(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json_object(path, label=label)
    violations: list[str] = []
    if payload.get("tool") != "scripts/scan_lightning_supply_chain.py":
        violations.append(f"tool={payload.get('tool')!r}")
    if payload.get("strict") is not True:
        violations.append(f"strict={payload.get('strict')!r}")
    if payload.get("status") != "OK":
        violations.append(f"status={payload.get('status')!r}")
    if payload.get("violation_count") != 0:
        violations.append(f"violation_count={payload.get('violation_count')!r}")
    if payload.get("violations") != []:
        violations.append("violations must be an empty list")
    if not isinstance(payload.get("package_versions"), dict):
        violations.append("package_versions must be recorded")
    if violations:
        raise ValueError(f"{label} failed validation: " + "; ".join(violations))
    return payload


def _validate_dali_requirements_artifact(
    path: Path,
    *,
    selected_wheels: list[dict[str, Any]] | None,
) -> list[str]:
    text = path.read_text()
    if "--index-url" in text or "--extra-index-url" in text or "\n-i " in text:
        raise ValueError("DALI requirements artifact must use direct wheel URLs, not package indexes")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError("DALI requirements artifact is empty")
    for line in lines:
        parts = line.split()
        if len(parts) != 2 or not parts[0].startswith(("https://", "http://")):
            raise ValueError(f"DALI requirements line is not a direct wheel URL with one hash: {line!r}")
        if not parts[0].endswith(".whl"):
            raise ValueError(f"DALI requirements line does not point to a wheel: {line!r}")
        hash_prefix = "--hash=sha256:"
        if not parts[1].startswith(hash_prefix):
            raise ValueError(f"DALI requirements line missing sha256 hash: {line!r}")
        digest = parts[1][len(hash_prefix):]
        _validate_sha256(digest, field="DALI requirements sha256")

    if selected_wheels is not None:
        expected = {
            f"{wheel['url']} --hash=sha256:{wheel['sha256']}"
            for wheel in selected_wheels
            if isinstance(wheel, dict) and "url" in wheel and "sha256" in wheel
        }
        missing = sorted(expected.difference(lines))
        if missing:
            raise ValueError(
                "DALI requirements artifact does not match bootstrap selected_wheels; "
                f"missing {missing[:3]}"
            )
    return lines


def _validate_dali_bootstrap_artifact(
    bootstrap_path: Path,
    requirements_path: Path,
) -> dict[str, Any]:
    payload = _load_json_object(bootstrap_path, label="DALI bootstrap artifact")
    violations: list[str] = []
    selected_package = payload.get("selected_package")
    selected_requirement = payload.get("selected_requirement")
    if payload.get("tool") != "lightning_exact_eval_dali_bootstrap":
        violations.append(f"tool={payload.get('tool')!r}")
    if payload.get("schema_version") != 1:
        violations.append(f"schema_version={payload.get('schema_version')!r}")
    if payload.get("required_dali_version") != DALI_BOOTSTRAP_VERSION:
        violations.append(
            f"required_dali_version={payload.get('required_dali_version')!r}, "
            f"expected {DALI_BOOTSTRAP_VERSION!r}"
        )
    if selected_package not in {"nvidia-dali-cuda120", "nvidia-dali-cuda130"}:
        violations.append(f"selected_package={selected_package!r}")
    expected_requirement = f"{selected_package}=={DALI_BOOTSTRAP_VERSION}" if isinstance(selected_package, str) else None
    if selected_requirement != expected_requirement:
        violations.append(f"selected_requirement={selected_requirement!r}, expected {expected_requirement!r}")
    if payload.get("bootstrap_action") not in {"already_exact", "install_hash_pinned_wheels"}:
        violations.append(f"bootstrap_action={payload.get('bootstrap_action')!r}")
    if payload.get("final_probe_violations") != []:
        violations.append("final_probe_violations must be an empty list")

    selected_wheels_raw = payload.get("selected_wheels")
    if not isinstance(selected_wheels_raw, list) or not selected_wheels_raw:
        violations.append("selected_wheels must be a non-empty list")
        selected_wheels: list[dict[str, Any]] | None = None
    else:
        selected_wheels = []
        has_selected_package_wheel = False
        for idx, wheel in enumerate(selected_wheels_raw):
            if not isinstance(wheel, dict):
                violations.append(f"selected_wheels[{idx}] is not an object")
                continue
            name = wheel.get("name")
            url = wheel.get("url")
            sha256 = wheel.get("sha256")
            version = wheel.get("version")
            if name == selected_package:
                has_selected_package_wheel = True
            if not all(isinstance(value, str) and value for value in (name, url, sha256, version)):
                violations.append(f"selected_wheels[{idx}] missing name/version/url/sha256")
                continue
            if not str(url).startswith(("https://", "http://")) or not str(url).endswith(".whl"):
                violations.append(f"selected_wheels[{idx}] URL is not a direct wheel URL")
            try:
                _validate_sha256(str(sha256), field=f"selected_wheels[{idx}].sha256")
            except ValueError as exc:
                violations.append(str(exc))
            selected_wheels.append(wheel)
        if not has_selected_package_wheel:
            violations.append(f"selected_wheels does not include selected_package={selected_package!r}")

    install_command = payload.get("install_command")
    if payload.get("installed") is True:
        if not isinstance(install_command, list):
            violations.append("installed=True but install_command is not recorded as a list")
        else:
            cmd_tokens = [str(item) for item in install_command]
            for token in ("pip", "install", "--require-hashes", "--no-deps", "--only-binary", ":all:", "--strict", "-r"):
                if token not in cmd_tokens:
                    violations.append(f"install_command missing {token!r}")
            if "--index-url" in cmd_tokens or "--extra-index-url" in cmd_tokens:
                violations.append("install_command uses a package index")

    final_probe = payload.get("final_probe")
    if not isinstance(final_probe, dict):
        violations.append("final_probe must be recorded")
    else:
        installed = final_probe.get("installed_distributions")
        if final_probe.get("dali_version") != DALI_BOOTSTRAP_VERSION:
            violations.append(f"final_probe.dali_version={final_probe.get('dali_version')!r}")
        if final_probe.get("nvidia_dali_fn_module") != "nvidia.dali.fn":
            violations.append(
                f"final_probe.nvidia_dali_fn_module={final_probe.get('nvidia_dali_fn_module')!r}"
            )
        if (
            isinstance(installed, dict)
            and isinstance(selected_package, str)
            and installed.get(selected_package) != DALI_BOOTSTRAP_VERSION
        ):
            violations.append(
                f"final_probe.installed_distributions[{selected_package!r}]="
                f"{installed.get(selected_package)!r}"
            )

    if violations:
        raise ValueError("DALI bootstrap artifact failed validation: " + "; ".join(violations))
    _validate_dali_requirements_artifact(requirements_path, selected_wheels=selected_wheels)
    return payload


def _validate_runner_preflight_artifact(
    path: Path,
    *,
    required_device: str = "cuda",
) -> dict[str, Any]:
    required_device = _normalise_eval_device(required_device)
    payload = _load_json_object(path, label="runner preflight artifact")
    violations: list[str] = []
    if payload.get("tool") != "lightning_exact_eval_runner_preflight":
        violations.append(f"tool={payload.get('tool')!r}")
    requested_device = payload.get("requested_device") or "cuda"
    if requested_device != required_device:
        violations.append(f"requested_device={requested_device!r}, expected {required_device!r}")
    if required_device == "cuda":
        if payload.get("cuda_available") is not True:
            violations.append(f"cuda_available={payload.get('cuda_available')!r}")
        device_count = payload.get("device_count")
        if not isinstance(device_count, int) or device_count <= 0:
            violations.append(f"device_count={device_count!r}")
        if payload.get("nvidia_dali_fn_module") != "nvidia.dali.fn":
            violations.append(f"nvidia_dali_fn_module={payload.get('nvidia_dali_fn_module')!r}")
        if not payload.get("device_name"):
            violations.append("device_name must be recorded")
    else:
        if payload.get("torch_import_ok") is not True:
            violations.append(f"torch_import_ok={payload.get('torch_import_ok')!r}")
        if payload.get("cpu_contest_platform") is not True:
            violations.append(f"cpu_contest_platform={payload.get('cpu_contest_platform')!r}")
        if payload.get("platform_system") != "Linux":
            violations.append(f"platform_system={payload.get('platform_system')!r}")
        machine = str(payload.get("platform_machine") or "").lower()
        if machine not in {"x86_64", "amd64"}:
            violations.append(f"platform_machine={payload.get('platform_machine')!r}")
    if violations:
        raise ValueError("runner preflight artifact failed validation: " + "; ".join(violations))
    return payload


@contextlib.contextmanager
def _state_file_lock(path: Path):
    """Serialize state-file read/modify/write sequences across processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_state_unlocked(path: Path = LIGHTNING_BATCH_STATE) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_state_unlocked(records: list[dict[str, Any]], path: Path = LIGHTNING_BATCH_STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        path.chmod(0o644)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_state(path: Path = LIGHTNING_BATCH_STATE) -> list[dict[str, Any]]:
    with _state_file_lock(path):
        return _load_state_unlocked(path)


def _save_state(records: list[dict[str, Any]], path: Path = LIGHTNING_BATCH_STATE) -> None:
    with _state_file_lock(path):
        _save_state_unlocked(records, path)


def _mutate_state(
    path: Path,
    mutator: Callable[[list[dict[str, Any]]], Any],
) -> Any:
    """Run a state mutation under one lock and atomically persist the result."""

    with _state_file_lock(path):
        records = _load_state_unlocked(path)
        result = mutator(records)  # type: ignore[operator]
        _save_state_unlocked(records, path)
        return result


def _safe_getattr(obj: object, name: str) -> Any:
    try:
        value = getattr(obj, name)
    except AttributeError:
        return None
    except Exception as exc:  # pragma: no cover - defensive around SDK objects
        return f"<error:{exc!r}>"
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


@dataclasses.dataclass(frozen=True)
class LightningAdjudicationSpec:
    """Structured arguments for ``scripts/adjudicate_contest_auth_eval.py``."""

    baseline_score: float
    predicted_band_low: float
    predicted_band_high: float
    regression_threshold: float
    baseline_archive_size_bytes: int | None = None
    max_posenet_dist: float | None = None
    max_segnet_dist: float | None = None
    baseline_posenet_dist: float | None = None
    baseline_segnet_dist: float | None = None
    max_posenet_relative: float | None = None
    max_segnet_relative: float | None = None
    component_reference_label: str = "baseline"
    delta_key: str = "score_delta_vs_baseline"
    max_sane_score: float = 10.0
    required_samples: int = 600
    required_device: str = "cuda"
    result_copy_name: str = "contest_auth_eval.adjudicated.json"
    provenance_name: str = "adjudication_provenance.json"
    allow_component_gate_forensic_success: bool = False
    allow_sane_score_forensic_success: bool = False

    def validate(self) -> None:
        for field in ("baseline_score", "predicted_band_low", "predicted_band_high", "regression_threshold"):
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"adjudication {field} must be numeric")
        if self.predicted_band_low > self.predicted_band_high:
            raise ValueError("adjudication predicted band low must be <= high")
        if self.baseline_archive_size_bytes is not None and self.baseline_archive_size_bytes <= 0:
            raise ValueError("adjudication baseline_archive_size_bytes must be positive")
        _validate_optional_number(self.max_posenet_dist, field="max_posenet_dist", minimum=0.0)
        _validate_optional_number(self.max_segnet_dist, field="max_segnet_dist", minimum=0.0)
        _validate_optional_number(
            self.baseline_posenet_dist,
            field="baseline_posenet_dist",
            minimum=0.0,
            strict_minimum=True,
        )
        _validate_optional_number(
            self.baseline_segnet_dist,
            field="baseline_segnet_dist",
            minimum=0.0,
            strict_minimum=True,
        )
        _validate_optional_number(
            self.max_posenet_relative,
            field="max_posenet_relative",
            minimum=0.0,
            strict_minimum=True,
        )
        _validate_optional_number(
            self.max_segnet_relative,
            field="max_segnet_relative",
            minimum=0.0,
            strict_minimum=True,
        )
        if self.max_posenet_relative is not None and self.baseline_posenet_dist is None:
            raise ValueError("adjudication max_posenet_relative requires baseline_posenet_dist")
        if self.max_segnet_relative is not None and self.baseline_segnet_dist is None:
            raise ValueError("adjudication max_segnet_relative requires baseline_segnet_dist")
        if not isinstance(self.component_reference_label, str) or not self.component_reference_label.strip():
            raise ValueError("adjudication component_reference_label must be a non-empty string")
        _normalise_eval_device(self.required_device)
        if self.required_samples <= 0:
            raise ValueError("adjudication required_samples must be positive")
        if "/" in self.result_copy_name or "/" in self.provenance_name:
            raise ValueError("adjudication artifact names must be local file names")

    def asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LightningBatchJobSpec:
    """Serializable specification for one Lightning Batch Job."""

    name: str
    machine: str
    command: str
    studio: str | None = None
    image: str | None = None
    teamspace: str | None = None
    org: str | None = None
    user: str | None = None
    cloud_account: str | None = None
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    interruptible: bool = False
    max_runtime: int | None = None
    reuse_snapshot: bool = True
    path_mappings: dict[str, str] | None = None
    scratch_disks: dict[str, int] | None = None
    role: str = "generic"
    expected_archive_sha256: str | None = None
    expected_archive_size_bytes: int | None = None
    queue_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    local_artifact_dir: str | None = None
    remote_output_dir: str | None = None
    adjudication: LightningAdjudicationSpec | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("job name is required")
        if not self.machine:
            raise ValueError("machine is required")
        if not self.command:
            raise ValueError("command is required")
        _validate_python_stdin_heredocs(self.command, job_name=self.name)
        if self.studio and self.image:
            raise ValueError("studio and image are mutually exclusive")
        if self.studio:
            validate_studio_machine_class_pair(self.machine, cloud_account=self.cloud_account)
        _validate_expected_archive(self.expected_archive_sha256, self.expected_archive_size_bytes)
        _normalise_metadata(self.queue_metadata)
        if self.adjudication is not None:
            self.adjudication.validate()
        if self.role in EXACT_AUTH_EVAL_ROLES:
            role_device = "cuda" if self.role == "exact_cuda_eval" else "cpu"
            _require_expected_archive(self.expected_archive_sha256, self.expected_archive_size_bytes)
            if self.adjudication is None:
                raise ValueError("exact eval jobs require adjudication provenance")
            if _normalise_eval_device(self.adjudication.required_device) != role_device:
                raise ValueError(
                    "exact eval job role and adjudication required_device disagree: "
                    f"role={self.role!r} required_device={self.adjudication.required_device!r}"
                )
            if self.interruptible:
                raise ValueError("exact eval jobs must not be interruptible")
            if f"--device {role_device}" not in self.command:
                raise ValueError(
                    "exact eval job command must run contest_auth_eval.py "
                    f"--device {role_device}"
                )
            other_device = "cpu" if role_device == "cuda" else "cuda"
            if f"--device {other_device}" in self.command:
                raise ValueError(
                    "exact eval job command mixes device axes: "
                    f"role={self.role!r} also contains --device {other_device}"
                )
            if "contest_auth_eval.json" not in self.command:
                raise ValueError("exact eval job command must preserve contest_auth_eval.json")
            if "scripts/scan_lightning_supply_chain.py" not in self.command:
                raise ValueError("exact eval job command must run Lightning supply-chain scan")
            if role_device == "cuda":
                if "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" not in self.command:
                    raise ValueError("exact eval job command must run CUDA runner preflight")
                if "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" not in self.command:
                    raise ValueError("exact eval job command must run DALI runner preflight")
                if "--require-hashes" not in self.command or "--no-deps" not in self.command:
                    raise ValueError("exact eval DALI bootstrap must use hash-pinned no-deps install")
                if "--index-url" in self.command or "--extra-index-url" in self.command:
                    raise ValueError("exact eval DALI bootstrap must use direct wheel URLs, not package indexes")
            else:
                if "LIGHTNING_RUNNER_CPU_PREFLIGHT_OK" not in self.command:
                    raise ValueError("exact CPU eval job command must run CPU runner preflight")
                if "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" in self.command:
                    raise ValueError("exact CPU eval job command must not run CUDA runner preflight")
                if "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" in self.command:
                    raise ValueError("exact CPU eval job command must not run DALI runner preflight")
                if "INFLATE_REQUIRE_CUDA=1" in self.command:
                    raise ValueError("exact CPU eval job command must not require CUDA inflate")
            if "LIGHTNING_INFLATE_RUNTIME_STATIC_PREFLIGHT_OK" not in self.command:
                raise ValueError("exact eval job command must run inflate runtime static preflight")
            if self.remote_output_dir is not None:
                _validate_writable_output_dir(self.remote_output_dir)
            if "/teamspace/jobs/" in self.command:
                raise ValueError(
                    "exact eval job command must not target /teamspace/jobs/...; "
                    "Lightning exposes that as a read-only artifact view inside Studio jobs"
                )
        if self.role == "alpha_geo0_exact_eval":
            if self.adjudication is None:
                raise ValueError("Alpha-Geo-0 exact eval jobs require adjudication provenance")
            if self.interruptible:
                raise ValueError("Alpha-Geo-0 exact eval jobs must not be interruptible")
            if "experiments/alpha_geo0_pose_regen.py" not in self.command:
                raise ValueError("Alpha-Geo-0 exact eval job must run experiments/alpha_geo0_pose_regen.py")
            if "--device cuda" not in self.command:
                raise ValueError("Alpha-Geo-0 exact eval job command must require --device cuda")
            if "contest_auth_eval.json" not in self.command:
                raise ValueError("Alpha-Geo-0 exact eval job command must preserve contest_auth_eval.json")
            if "scripts/scan_lightning_supply_chain.py" not in self.command:
                raise ValueError("Alpha-Geo-0 exact eval job must run Lightning supply-chain scan")
            if "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" not in self.command:
                raise ValueError("Alpha-Geo-0 exact eval job must run CUDA runner preflight")
            if "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" not in self.command:
                raise ValueError("Alpha-Geo-0 exact eval job must run DALI runner preflight")
            if "--require-hashes" not in self.command or "--no-deps" not in self.command:
                raise ValueError("Alpha-Geo-0 DALI bootstrap must use hash-pinned no-deps install")
            if "--index-url" in self.command or "--extra-index-url" in self.command:
                raise ValueError("Alpha-Geo-0 DALI bootstrap must use direct wheel URLs, not package indexes")
            if self.remote_output_dir is not None:
                _validate_writable_output_dir(self.remote_output_dir)
            if "/teamspace/jobs/" in self.command:
                raise ValueError(
                    "Alpha-Geo-0 exact eval job command must not target /teamspace/jobs/...; "
                    "Lightning exposes that as a read-only artifact view inside Studio jobs"
                )
        if self.role == "official_component_response":
            if self.interruptible:
                raise ValueError("official component-response jobs must not be interruptible")
            if "experiments/profile_component_sensitivity_official.py" not in self.command:
                raise ValueError("official component-response job must run the official profiler")
            if "--device cuda" not in self.command:
                raise ValueError("official component-response job must run with --device cuda")
            if ARTIFACT_COMPONENT_RESPONSE_SUMMARY not in self.command:
                raise ValueError("official component-response job must preserve the summary JSON")
            if "scripts/scan_lightning_supply_chain.py" not in self.command:
                raise ValueError("official component-response job must run Lightning supply-chain scan")
            if "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" not in self.command:
                raise ValueError("official component-response job must run CUDA runner preflight")
            if "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" not in self.command:
                raise ValueError("official component-response job must run DALI runner preflight")
            if "--require-hashes" not in self.command or "--no-deps" not in self.command:
                raise ValueError("official component-response DALI bootstrap must use hash-pinned no-deps install")
            if "--index-url" in self.command or "--extra-index-url" in self.command:
                raise ValueError("official component-response DALI bootstrap must use direct wheel URLs, not package indexes")
            if self.remote_output_dir is not None:
                _validate_writable_output_dir(self.remote_output_dir)
            if "/teamspace/jobs/" in self.command:
                raise ValueError(
                    "official component-response job command must not target /teamspace/jobs/...; "
                    "Lightning exposes that as a read-only artifact view inside Studio jobs"
                )
        if self.role == "diagnostic_component_sensitivity":
            if self.interruptible:
                raise ValueError("diagnostic component-sensitivity jobs must not be interruptible")
            if "experiments/profile_component_sensitivity.py" not in self.command:
                raise ValueError("diagnostic component-sensitivity job must run the diagnostic profiler")
            if "--device cuda" not in self.command:
                raise ValueError("diagnostic component-sensitivity job must run with --device cuda")
            if "--manifest-output" in self.command:
                raise ValueError("diagnostic component-sensitivity job must not assemble a promotion manifest")
            if ARTIFACT_COMPONENT_SENSITIVITY_INPUTS not in self.command:
                raise ValueError("diagnostic component-sensitivity job must record input custody")
            if ARTIFACT_COMPONENT_SENSITIVITY_RUN not in self.command:
                raise ValueError("diagnostic component-sensitivity job must record command/env metadata")
            if ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY not in self.command:
                raise ValueError("diagnostic component-sensitivity job must preserve the profile summary")
            for member in ("renderer.bin", "masks.mkv", "optimized_poses.bin"):
                if member not in self.command:
                    raise ValueError(
                        "diagnostic component-sensitivity job must zip-slip-safely extract "
                        f"{member}"
                    )
            if "scripts/scan_lightning_supply_chain.py" not in self.command:
                raise ValueError("diagnostic component-sensitivity job must run Lightning supply-chain scan")
            if "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK" not in self.command:
                raise ValueError("diagnostic component-sensitivity job must run CUDA runner preflight")
            if "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK" not in self.command:
                raise ValueError("diagnostic component-sensitivity job must run DALI runner preflight")
            if "--require-hashes" not in self.command or "--no-deps" not in self.command:
                raise ValueError("diagnostic component-sensitivity DALI bootstrap must use hash-pinned no-deps install")
            if "--index-url" in self.command or "--extra-index-url" in self.command:
                raise ValueError("diagnostic component-sensitivity DALI bootstrap must use direct wheel URLs, not package indexes")
            if self.remote_output_dir is not None:
                _validate_writable_output_dir(self.remote_output_dir)
            if "/teamspace/jobs/" in self.command:
                raise ValueError(
                    "diagnostic component-sensitivity job command must not target /teamspace/jobs/...; "
                    "Lightning exposes that as a read-only artifact view inside Studio jobs"
                )

    def asdict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["env"] = dict(sorted(self.env.items()))
        out["queue_metadata"] = _normalise_metadata(self.queue_metadata)
        if self.adjudication is not None:
            out["adjudication"] = self.adjudication.asdict()
        return out


def _metadata_payload(
    *,
    job_name: str | None,
    role: str,
    expected_archive_sha256: str | None,
    expected_archive_size_bytes: int | None,
    queue_metadata: dict[str, Any] | None,
    adjudication: LightningAdjudicationSpec | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_name": job_name,
        "role": role,
        "expected_archive_sha256": expected_archive_sha256,
        "expected_archive_size_bytes": expected_archive_size_bytes,
        "queue_metadata": _normalise_metadata(queue_metadata),
        "adjudication": adjudication.asdict() if adjudication is not None else None,
        "score_source": "contest_auth_eval.json:score_recomputed_from_components",
        "status_source": "lightning_sdk_job_attributes",
    }


def _write_metadata_command(
    *,
    python_bin: str,
    output_dir: str,
    payload: dict[str, Any],
) -> str:
    py = _quote(python_bin)
    out = _quote(output_dir)
    payload_arg = _quote(json.dumps(payload, sort_keys=True))
    return (
        f"{py} - {out}/{ARTIFACT_METADATA} {payload_arg} <<'PY'\n"
        "import json, sys, time\n"
        "payload = json.loads(sys.argv[2])\n"
        "payload['artifact_recorded_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())\n"
        "open(sys.argv[1], 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "PY"
    )


def _archive_preflight_command(
    *,
    python_bin: str,
    archive_path: str,
    expected_archive_sha256: str | None,
    expected_archive_size_bytes: int | None,
) -> str:
    py = _quote(python_bin)
    archive = _quote(archive_path)
    expected_sha = _quote(expected_archive_sha256 or "")
    expected_bytes = _quote(str(expected_archive_size_bytes or ""))
    return (
        f"{py} - {archive} {expected_sha} {expected_bytes} <<'PY'\n"
        "import hashlib, json, pathlib, sys\n"
        "archive = pathlib.Path(sys.argv[1])\n"
        "expected_sha = sys.argv[2] or None\n"
        "expected_bytes = int(sys.argv[3]) if sys.argv[3] else None\n"
        "h = hashlib.sha256()\n"
        "with archive.open('rb') as f:\n"
        "    for chunk in iter(lambda: f.read(1024 * 1024), b''):\n"
        "        h.update(chunk)\n"
        "actual_sha = h.hexdigest()\n"
        "actual_bytes = archive.stat().st_size\n"
        "if expected_sha is not None and actual_sha != expected_sha:\n"
        "    raise SystemExit(f'FATAL: archive sha256 mismatch: expected={expected_sha} actual={actual_sha}')\n"
        "if expected_bytes is not None and actual_bytes != expected_bytes:\n"
        "    raise SystemExit(f'FATAL: archive bytes mismatch: expected={expected_bytes} actual={actual_bytes}')\n"
        "print('LIGHTNING_ARCHIVE_PREFLIGHT_JSON_OK')\n"
        "print(json.dumps({'archive_sha256': actual_sha, 'archive_size_bytes': actual_bytes}, sort_keys=True))\n"
        "PY"
    )


def _dali_bootstrap_command(
    *,
    python_bin: str,
    output_dir: str,
) -> str:
    py = _quote(python_bin)
    out = _quote(output_dir)
    wheel_matrix = _quote(json.dumps(DALI_BOOTSTRAP_WHEELS, sort_keys=True))
    return (
        f"{py} - {out}/{ARTIFACT_DALI_BOOTSTRAP} "
        f"{out}/{ARTIFACT_DALI_REQUIREMENTS} {wheel_matrix} <<'PY'\n"
        "import importlib.metadata as metadata\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import shutil\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "\n"
        "DALI_VERSION = '1.52.0'\n"
        "DALI_CUDA120_REQUIREMENT = 'nvidia-dali-cuda120==1.52.0'\n"
        "DALI_CUDA130_REQUIREMENT = 'nvidia-dali-cuda130==1.52.0'\n"
        "BOOTSTRAP_JSON = pathlib.Path(sys.argv[1])\n"
        "REQUIREMENTS_TXT = pathlib.Path(sys.argv[2])\n"
        "WHEEL_MATRIX = json.loads(sys.argv[3])\n"
        "\n"
        "def version_or_none(name):\n"
        "    try:\n"
        "        return metadata.version(name)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        return None\n"
        "\n"
        "try:\n"
        "    import torch\n"
        "except Exception as exc:\n"
        "    raise SystemExit(f'FATAL: torch import failed while selecting DALI package: {exc!r}')\n"
        "cuda_version = getattr(torch.version, 'cuda', None)\n"
        "cuda_major_s = str(cuda_version or '').split('.', 1)[0]\n"
        "if cuda_major_s not in {'12', '13'}:\n"
        "    raise SystemExit(f'FATAL: unsupported torch CUDA version for exact eval DALI bootstrap: {cuda_version!r}')\n"
        "cuda_major = int(cuda_major_s)\n"
        "cuda_family = 'cu130' if cuda_major == 13 else 'cu120'\n"
        "expected_package = 'nvidia-dali-cuda130' if cuda_major == 13 else 'nvidia-dali-cuda120'\n"
        "unexpected_package = 'nvidia-dali-cuda120' if cuda_major == 13 else 'nvidia-dali-cuda130'\n"
        "selected_requirement = DALI_CUDA130_REQUIREMENT if cuda_major == 13 else DALI_CUDA120_REQUIREMENT\n"
        "py_tag = f'py{sys.version_info.major}{sys.version_info.minor}'\n"
        "if py_tag not in WHEEL_MATRIX:\n"
        "    raise SystemExit(f'FATAL: no hash-pinned DALI dependency wheels registered for {py_tag}')\n"
        "selected_wheels = WHEEL_MATRIX['common'] + WHEEL_MATRIX[py_tag] + WHEEL_MATRIX[cuda_family]\n"
        "\n"
        "def _probe():\n"
        "    code = \"\"\"\n"
        "import importlib.metadata as metadata, json, sys\n"
        "import nvidia.dali\n"
        "import nvidia.dali.fn as dali_fn\n"
        "def version_or_none(name):\n"
        "    try:\n"
        "        return metadata.version(name)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        return None\n"
        "print(json.dumps({\n"
        "    'python': sys.executable,\n"
        "    'dali_version': getattr(nvidia.dali, '__version__', None),\n"
        "    'nvidia_dali_fn_module': getattr(dali_fn, '__name__', None),\n"
        "    'installed_distributions': {\n"
        "        name: version_or_none(name)\n"
        "        for name in ['nvidia-dali-cuda120', 'nvidia-dali-cuda130']\n"
        "    },\n"
        "}, sort_keys=True))\n"
        "\"\"\"\n"
        "    return subprocess.run([sys.executable, '-c', code], check=False, capture_output=True, text=True, timeout=30)\n"
        "\n"
        "def _parse_probe(result):\n"
        "    if result.returncode != 0:\n"
        "        return None\n"
        "    try:\n"
        "        return json.loads(result.stdout)\n"
        "    except Exception:\n"
        "        return None\n"
        "\n"
        "def _probe_violations(result):\n"
        "    if result.returncode != 0:\n"
        "        return [f'import failed: {result.stderr[-1000:]}']\n"
        "    data = _parse_probe(result)\n"
        "    if not isinstance(data, dict):\n"
        "        return [f'probe did not emit JSON: {result.stdout[-1000:]}']\n"
        "    installed = data.get('installed_distributions') or {}\n"
        "    violations = []\n"
        "    if data.get('dali_version') != DALI_VERSION:\n"
        "        violations.append(f\"nvidia.dali.__version__={data.get('dali_version')!r}, expected {DALI_VERSION!r}\")\n"
        "    if installed.get(expected_package) != DALI_VERSION:\n"
        "        violations.append(f\"{expected_package}={installed.get(expected_package)!r}, expected {DALI_VERSION!r}\")\n"
        "    if installed.get(unexpected_package) is not None:\n"
        "        violations.append(f\"unexpected CUDA-family DALI package installed: {unexpected_package}={installed.get(unexpected_package)!r}\")\n"
        "    if data.get('nvidia_dali_fn_module') != 'nvidia.dali.fn':\n"
        "        violations.append(f\"nvidia_dali_fn_module={data.get('nvidia_dali_fn_module')!r}\")\n"
        "    return violations\n"
        "\n"
        "def _write_payload(payload):\n"
        "    BOOTSTRAP_JSON.parent.mkdir(parents=True, exist_ok=True)\n"
        "    BOOTSTRAP_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "\n"
        "def _write_requirements():\n"
        "    REQUIREMENTS_TXT.parent.mkdir(parents=True, exist_ok=True)\n"
        "    lines = [f\"{wheel['url']} --hash=sha256:{wheel['sha256']}\" for wheel in selected_wheels]\n"
        "    REQUIREMENTS_TXT.write_text('\\n'.join(lines) + '\\n')\n"
        "\n"
        "payload = {\n"
        "    'schema_version': 1,\n"
        "    'tool': 'lightning_exact_eval_dali_bootstrap',\n"
        "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "    'python': sys.executable,\n"
        "    'required_dali_version': DALI_VERSION,\n"
        "    'torch_version': getattr(torch, '__version__', None),\n"
        "    'torch_cuda_version': cuda_version,\n"
        "    'cuda_family': cuda_family,\n"
        "    'selected_package': expected_package,\n"
        "    'selected_requirement': selected_requirement,\n"
        "    'unexpected_package': unexpected_package,\n"
        "    'requirements_path': str(REQUIREMENTS_TXT),\n"
        "    'selected_wheels': selected_wheels,\n"
        "    'installed': False,\n"
        "    'installer_bootstrap_action': None,\n"
        "}\n"
        "_write_requirements()\n"
        "\n"
        "initial = _probe()\n"
        "payload['initial_probe_returncode'] = initial.returncode\n"
        "payload['initial_probe_stdout'] = initial.stdout.strip()\n"
        "payload['initial_probe_stderr_tail'] = initial.stderr[-2000:]\n"
        "payload['initial_probe'] = _parse_probe(initial)\n"
        "initial_violations = _probe_violations(initial)\n"
        "payload['initial_probe_violations'] = initial_violations\n"
        "if not initial_violations:\n"
        "    payload['bootstrap_action'] = 'already_exact'\n"
        "elif initial.returncode == 0:\n"
        "    payload['bootstrap_action'] = 'fail_wrong_preinstalled_dali'\n"
        "    _write_payload(payload)\n"
        "    raise SystemExit('FATAL: preinstalled DALI is not the exact expected package/version: ' + '; '.join(initial_violations))\n"
        "else:\n"
        "    payload['bootstrap_action'] = 'install_hash_pinned_wheels'\n"
        "    uv_env = os.environ.get('UV_BIN')\n"
        "    uv = uv_env if uv_env and pathlib.Path(uv_env).is_file() else shutil.which('uv')\n"
        "    if not uv:\n"
        "        raise SystemExit('FATAL: uv is required to install pinned DALI into this pip-less runner env; run scripts/ensure_remote_uv.sh first')\n"
        "    payload['installer_bootstrap_action'] = 'uv_provided_by_shell_ensure_remote_uv' if uv_env else 'uv_already_available'\n"
        "    payload['installer_uv_path'] = uv\n"
        "    cmd = [\n"
        "        uv,\n"
        "        'pip',\n"
        "        'install',\n"
        "        '--python',\n"
        "        sys.executable,\n"
        "        '--require-hashes',\n"
        "        '--no-deps',\n"
        "        '--only-binary',\n"
        "        ':all:',\n"
        "        '--strict',\n"
        "        '-r',\n"
        "        str(REQUIREMENTS_TXT),\n"
        "    ]\n"
        "    payload['install_command'] = cmd\n"
        "    install = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=900)\n"
        "    payload['install_returncode'] = install.returncode\n"
        "    payload['install_stdout_tail'] = install.stdout[-4000:]\n"
        "    payload['install_stderr_tail'] = install.stderr[-4000:]\n"
        "    if install.returncode != 0:\n"
        "        _write_payload(payload)\n"
        "        raise SystemExit(f'FATAL: hash-pinned DALI install failed with returncode={install.returncode}')\n"
        "    payload['installed'] = True\n"
        "\n"
        "final = _probe()\n"
        "payload['final_probe_returncode'] = final.returncode\n"
        "payload['final_probe_stdout'] = final.stdout.strip()\n"
        "payload['final_probe_stderr_tail'] = final.stderr[-2000:]\n"
        "payload['final_probe'] = _parse_probe(final)\n"
        "final_violations = _probe_violations(final)\n"
        "payload['final_probe_violations'] = final_violations\n"
        "for package in ('nvidia-dali-cuda120', 'nvidia-dali-cuda130'):\n"
        "    try:\n"
        "        payload[f'{package}_version'] = metadata.version(package)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        payload[f'{package}_version'] = None\n"
        "_write_payload(payload)\n"
        "if final_violations:\n"
        "    raise SystemExit('FATAL: DALI exact preflight failed: ' + '; '.join(final_violations))\n"
        "print('LIGHTNING_RUNNER_DALI_PREFLIGHT_OK')\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "PY"
    )


def _runner_preflight_command(
    *,
    python_bin: str,
    output_dir: str,
    eval_device: str = "cuda",
) -> str:
    eval_device = _normalise_eval_device(eval_device)
    py = _quote(python_bin)
    out = _quote(output_dir)
    if eval_device == "cpu":
        return "\n".join(
            [
                (
                    f"{py} scripts/scan_lightning_supply_chain.py "
                    f"--json-out {out}/{ARTIFACT_SUPPLY_CHAIN_SCAN_PRE} "
                    "--quiet --strict"
                ),
                (
                    f"{py} scripts/scan_lightning_supply_chain.py "
                    f"--json-out {out}/{ARTIFACT_SUPPLY_CHAIN_SCAN} "
                    "--quiet --strict"
                ),
                (
                    f"{py} - {out}/{ARTIFACT_RUNNER_PREFLIGHT} <<'PY'\n"
                    "import json, platform, sys, time\n"
                    "try:\n"
                    "    import torch\n"
                    "    torch_import_ok = True\n"
                    "    torch_version = getattr(torch, '__version__', None)\n"
                    "    cuda_available = bool(torch.cuda.is_available())\n"
                    "    device_count = int(torch.cuda.device_count()) if cuda_available else 0\n"
                    "except Exception as exc:\n"
                    "    raise SystemExit(f'FATAL: torch import failed before exact CPU eval: {exc!r}')\n"
                    "system = platform.system()\n"
                    "machine = platform.machine().lower()\n"
                    "cpu_contest_platform = system == 'Linux' and machine in {'x86_64', 'amd64'}\n"
                    "payload = {\n"
                    "    'schema_version': 1,\n"
                    "    'tool': 'lightning_exact_eval_runner_preflight',\n"
                    "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
                    "    'requested_device': 'cpu',\n"
                    "    'python': sys.executable,\n"
                    "    'torch_import_ok': torch_import_ok,\n"
                    "    'torch_version': torch_version,\n"
                    "    'platform_system': system,\n"
                    "    'platform_machine': platform.machine(),\n"
                    "    'cpu_contest_platform': cpu_contest_platform,\n"
                    "    'cuda_available': cuda_available,\n"
                    "    'device_count': device_count,\n"
                    "}\n"
                    "if not cpu_contest_platform:\n"
                    "    open(sys.argv[1], 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
                    "    raise SystemExit('FATAL: exact CPU eval requires Linux x86_64/amd64 contest-compatible hardware')\n"
                    "open(sys.argv[1], 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
                    "print('LIGHTNING_RUNNER_CPU_PREFLIGHT_OK')\n"
                    "print(json.dumps(payload, sort_keys=True))\n"
                    "PY"
                ),
            ]
        )
    return "\n".join(
        [
            (
                f"{py} scripts/scan_lightning_supply_chain.py "
                f"--json-out {out}/{ARTIFACT_SUPPLY_CHAIN_SCAN_PRE} "
                "--quiet --strict"
            ),
            "LIGHTNING_VENV_LOCK=.omx/state/lightning_exact_eval_venv.lock",
            "LIGHTNING_VENV_LOCK_ACQUIRED=0",
            "for _ in $(seq 1 600); do",
            '  if mkdir "$LIGHTNING_VENV_LOCK" 2>/dev/null; then',
            "    LIGHTNING_VENV_LOCK_ACQUIRED=1",
            '    printf "%s\\n" "$$" > "$LIGHTNING_VENV_LOCK/pid"',
            "    break",
            "  fi",
            "  sleep 2",
            "done",
            'if [ "$LIGHTNING_VENV_LOCK_ACQUIRED" != "1" ]; then',
            "  echo 'FATAL: timed out waiting for Lightning exact-eval venv lock' >&2",
            "  exit 43",
            "fi",
            'trap \'rm -rf "$LIGHTNING_VENV_LOCK"\' EXIT',
            _dali_bootstrap_command(python_bin=python_bin, output_dir=output_dir),
            (
                f"{py} scripts/scan_lightning_supply_chain.py "
                f"--json-out {out}/{ARTIFACT_SUPPLY_CHAIN_SCAN} "
                "--quiet --strict"
            ),
            'rm -rf "$LIGHTNING_VENV_LOCK"',
            "trap - EXIT",
            (
                f"{py} - {out}/{ARTIFACT_RUNNER_PREFLIGHT} <<'PY'\n"
                "import json, shutil, subprocess, sys, time\n"
                "try:\n"
                "    import torch\n"
                "except Exception as exc:\n"
                "    raise SystemExit(f'FATAL: torch import failed before exact eval: {exc!r}')\n"
                "try:\n"
                "    import nvidia.dali.fn as dali_fn\n"
                "except Exception as exc:\n"
                "    raise SystemExit(f'FATAL: nvidia.dali import failed before exact eval: {exc!r}')\n"
                "cuda_available = bool(torch.cuda.is_available())\n"
                "device_count = int(torch.cuda.device_count()) if cuda_available else 0\n"
                "if not cuda_available or device_count <= 0:\n"
                "    raise SystemExit('FATAL: CUDA is not visible before exact eval')\n"
                "device_name = torch.cuda.get_device_name(0)\n"
                "nvidia_smi = shutil.which('nvidia-smi')\n"
                "nvidia_smi_head = None\n"
                "if nvidia_smi:\n"
                "    probe = subprocess.run(\n"
                "        [nvidia_smi, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],\n"
                "        check=False,\n"
                "        capture_output=True,\n"
                "        text=True,\n"
                "        timeout=20,\n"
                "    )\n"
                "    nvidia_smi_head = (probe.stdout or probe.stderr).strip().splitlines()[:4]\n"
                "payload = {\n"
                "    'schema_version': 1,\n"
                "    'tool': 'lightning_exact_eval_runner_preflight',\n"
                "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
                "    'python': sys.executable,\n"
                "    'torch_version': getattr(torch, '__version__', None),\n"
                "    'nvidia_dali_fn_module': getattr(dali_fn, '__name__', None),\n"
                "    'cuda_available': cuda_available,\n"
                "    'device_count': device_count,\n"
                "    'device_name': device_name,\n"
                "    'gpu_t4_match': 'T4' in device_name,\n"
                "    'nvidia_smi': nvidia_smi,\n"
                "    'nvidia_smi_head': nvidia_smi_head,\n"
                "}\n"
                "open(sys.argv[1], 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
                "print('LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK')\n"
                "print(json.dumps(payload, sort_keys=True))\n"
                "PY"
            ),
        ]
    )


def _inflate_runtime_bootstrap_command(
    *,
    python_bin: str,
    output_dir: str,
    env: dict[str, str] | None,
) -> str:
    py = _quote(python_bin)
    out = _quote(output_dir)
    specs = []
    for key in (
        "INFLATE_TORCH_SPEC",
        "INFLATE_TORCHVISION_SPEC",
        "INFLATE_BROTLI_SPEC",
        "INFLATE_AV_SPEC",
        "INFLATE_NUMPY_SPEC",
    ):
        value = str((env or {}).get(key, "")).strip()
        if value:
            specs.append(value)
    specs_json = _quote(json.dumps(specs, sort_keys=True))
    return (
        f"{py} - {out}/{ARTIFACT_INFLATE_RUNTIME_BOOTSTRAP} {specs_json} {py} <<'PY'\n"
        "import json, os, shutil, subprocess, sys, time\n"
        "out_path = sys.argv[1]\n"
        "specs = json.loads(sys.argv[2])\n"
        "python_bin = sys.argv[3]\n"
        "payload = {\n"
        "    'schema_version': 1,\n"
        "    'tool': 'lightning_exact_eval_inflate_runtime_bootstrap',\n"
        "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "    'python': python_bin,\n"
        "    'requested_specs': specs,\n"
        "    'installed': False,\n"
        "    'install_command': None,\n"
        "    'install_returncode': None,\n"
        "    'stdout_tail': '',\n"
        "    'stderr_tail': '',\n"
        "}\n"
        "if specs:\n"
        "    uv = os.environ.get('UV_BIN') or shutil.which('uv')\n"
        "    if not uv:\n"
        "        raise SystemExit('FATAL: optional inflate runtime deps requested but uv is unavailable')\n"
        "    cmd = [uv, 'pip', 'install', '--python', python_bin, *specs]\n"
        "    payload['install_command'] = cmd\n"
        "    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600)\n"
        "    payload['install_returncode'] = result.returncode\n"
        "    payload['stdout_tail'] = result.stdout[-4000:]\n"
        "    payload['stderr_tail'] = result.stderr[-4000:]\n"
        "    payload['installed'] = result.returncode == 0\n"
        "    if result.returncode != 0:\n"
        "        open(out_path, 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "        raise SystemExit('FATAL: optional inflate runtime dependency install failed')\n"
        "open(out_path, 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "print('LIGHTNING_INFLATE_RUNTIME_BOOTSTRAP_OK')\n"
        "print(json.dumps({'requested_specs': specs, 'installed': payload['installed']}, sort_keys=True))\n"
        "PY"
    )


def _inflate_runtime_static_preflight_command(
    *,
    python_bin: str,
    output_dir: str,
) -> str:
    py = _quote(python_bin)
    out = _quote(output_dir)
    return (
        f"{py} - {out}/{ARTIFACT_INFLATE_RUNTIME_STATIC_PREFLIGHT} <<'PY'\n"
        "import inspect, json, py_compile, sys, time\n"
        "from pathlib import Path\n"
        "files = [\n"
        "    'submissions/robust_current/inflate_renderer.py',\n"
        "    'submissions/robust_current/unpack_renderer_payload.py',\n"
        "    'submissions/robust_current/apply_qzs3_postprocess.py',\n"
        "]\n"
        "compiled = []\n"
        "for rel in files:\n"
        "    path = Path(rel)\n"
        "    if path.is_file():\n"
        "        py_compile.compile(str(path), doraise=True)\n"
        "        compiled.append(rel)\n"
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('inflate_renderer_static_preflight', files[0])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "assert spec is not None and spec.loader is not None\n"
        "spec.loader.exec_module(module)\n"
        "signature = inspect.signature(module._generate_and_write)\n"
        "if 'pr81_router_actions' not in signature.parameters:\n"
        "    raise SystemExit('FATAL: inflate_renderer._generate_and_write lacks pr81_router_actions parameter')\n"
        "payload = {\n"
        "    'schema_version': 1,\n"
        "    'tool': 'lightning_exact_eval_inflate_runtime_static_preflight',\n"
        "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "    'python': sys.executable,\n"
        "    'compiled': compiled,\n"
        "    'generate_and_write_parameters': list(signature.parameters),\n"
        "    'no_router_qpost_guard': True,\n"
        "}\n"
        "Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "print('LIGHTNING_INFLATE_RUNTIME_STATIC_PREFLIGHT_OK')\n"
        "print(json.dumps({'compiled': compiled, 'no_router_qpost_guard': True}, sort_keys=True))\n"
        "PY"
    )


def _adjudication_command(
    *,
    python_bin: str,
    output_dir: str,
    adjudication: LightningAdjudicationSpec,
) -> str:
    py = _quote(python_bin)
    out = _quote(output_dir)
    cmd = [
        f"{py} -u scripts/adjudicate_contest_auth_eval.py",
        f"--contest-json {out}/contest_auth_eval.json",
        f"--provenance {out}/{_quote(adjudication.provenance_name)}",
        f"--archive {out}/{ARTIFACT_ARCHIVE}",
        f"--result-copy {out}/{_quote(adjudication.result_copy_name)}",
        f"--baseline-score {_quote(str(adjudication.baseline_score))}",
        (
            "--predicted-band "
            f"{_quote(str(adjudication.predicted_band_low))} "
            f"{_quote(str(adjudication.predicted_band_high))}"
        ),
        f"--regression-threshold {_quote(str(adjudication.regression_threshold))}",
        f"--delta-key {_quote(adjudication.delta_key)}",
        f"--component-reference-label {_quote(adjudication.component_reference_label)}",
        f"--required-device {_quote(_normalise_eval_device(adjudication.required_device))}",
        f"--required-samples {_quote(str(adjudication.required_samples))}",
        f"--max-sane-score {_quote(str(adjudication.max_sane_score))}",
    ]
    if adjudication.baseline_archive_size_bytes is not None:
        cmd.append(f"--baseline-archive-bytes {_quote(str(adjudication.baseline_archive_size_bytes))}")
    if adjudication.max_posenet_dist is not None:
        cmd.append(f"--max-posenet-dist {_quote(str(adjudication.max_posenet_dist))}")
    if adjudication.max_segnet_dist is not None:
        cmd.append(f"--max-segnet-dist {_quote(str(adjudication.max_segnet_dist))}")
    if adjudication.baseline_posenet_dist is not None:
        cmd.append(f"--baseline-posenet-dist {_quote(str(adjudication.baseline_posenet_dist))}")
    if adjudication.baseline_segnet_dist is not None:
        cmd.append(f"--baseline-segnet-dist {_quote(str(adjudication.baseline_segnet_dist))}")
    if adjudication.max_posenet_relative is not None:
        cmd.append(f"--max-posenet-relative {_quote(str(adjudication.max_posenet_relative))}")
    if adjudication.max_segnet_relative is not None:
        cmd.append(f"--max-segnet-relative {_quote(str(adjudication.max_segnet_relative))}")
    if adjudication.allow_component_gate_forensic_success:
        cmd.append("--allow-component-gate-forensic-success")
    if adjudication.allow_sane_score_forensic_success:
        cmd.append("--allow-sane-score-forensic-success")
    return " ".join(cmd) + f" 2>&1 | tee {out}/adjudication.log"


def exact_cuda_eval_command(
    *,
    repo_dir: str,
    archive_path: str,
    upstream_dir: str,
    output_dir: str,
    python_bin: str = ".venv/bin/python",
    inflate_sh: str = "submissions/robust_current/inflate.sh",
    job_name: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    queue_metadata: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    adjudication: LightningAdjudicationSpec | None = None,
    component_trace: bool = False,
    component_trace_top_k: int = 80,
    eval_device: str = "cuda",
) -> str:
    """Build a fail-closed exact auth-eval command for a Lightning job.

    The historical function name is retained for compatibility. Pass
    ``eval_device="cpu"`` only for an explicit [contest-CPU] replay axis.
    """

    eval_device = _normalise_eval_device(eval_device)
    role = _exact_eval_role(eval_device)
    axis = _exact_eval_axis(eval_device)
    ok_marker = (
        "LIGHTNING_EXACT_CUDA_EVAL_JSON_OK"
        if eval_device == "cuda"
        else "LIGHTNING_EXACT_CPU_EVAL_JSON_OK"
    )
    _validate_writable_output_dir(output_dir)
    _require_expected_archive(expected_archive_sha256, expected_archive_size_bytes)
    if adjudication is None:
        raise ValueError("exact auth eval command requires adjudication provenance")
    if _normalise_eval_device(adjudication.required_device) != eval_device:
        raise ValueError(
            "exact auth eval command device and adjudication required_device disagree: "
            f"eval_device={eval_device!r} required_device={adjudication.required_device!r}"
        )
    if eval_device == "cpu" and component_trace:
        raise ValueError("component_trace is CUDA-only; run CPU exact eval without component_trace")
    adjudication.validate()
    expected_runtime_tree_sha256 = None
    if isinstance(queue_metadata, dict):
        raw_runtime_sha = queue_metadata.get("expected_runtime_tree_sha256")
        if raw_runtime_sha is not None:
            expected_runtime_tree_sha256 = str(raw_runtime_sha)
    runtime_tree_arg = (
        f"--expected-runtime-tree-sha256 {_quote(expected_runtime_tree_sha256)} "
        if expected_runtime_tree_sha256
        else ""
    )
    repo = _quote(repo_dir)
    archive = _quote(archive_path)
    upstream = _quote(upstream_dir)
    out = _quote(output_dir)
    py = _quote(python_bin)
    inflate = _quote(inflate_sh)
    cuda_env_export = "export INFLATE_REQUIRE_CUDA=1\n" if eval_device == "cuda" else ""
    command = "\n".join(
        [
            "set -euo pipefail",
            f"cd {repo}",
            "test -f env.sh && source env.sh || true",
            f"mkdir -p {out}",
            f"rm -rf {out}/eval_work {out}/uv_project_env",
            (
                f"rm -f {out}/{ARTIFACT_ARCHIVE} "
                f"{out}/contest_auth_eval.json "
                f"{out}/contest_auth_eval.adjudicated.json "
                f"{out}/eval_provenance.json "
                f"{out}/report.txt "
                f"{out}/auth_eval.log "
                f"{out}/{ARTIFACT_COMPONENT_TRACE} "
                f"{out}/{ARTIFACT_COMPONENT_TRACE_LOG} "
                f"{out}/{ARTIFACT_COMPONENT_TRACE_STATUS} "
                f"{out}/adjudication.log "
                f"{out}/adjudication_provenance.json "
                f"{out}/{ARTIFACT_METADATA} "
                f"{out}/{ARTIFACT_DALI_BOOTSTRAP} "
                f"{out}/{ARTIFACT_DALI_REQUIREMENTS} "
                f"{out}/{ARTIFACT_RUNNER_PREFLIGHT} "
                f"{out}/{ARTIFACT_INFLATE_RUNTIME_BOOTSTRAP} "
                f"{out}/{ARTIFACT_INFLATE_RUNTIME_STATIC_PREFLIGHT} "
                f"{out}/{ARTIFACT_SUPPLY_CHAIN_SCAN_PRE} "
                f"{out}/{ARTIFACT_SUPPLY_CHAIN_SCAN}"
            ),
            _ensure_remote_uv_command(output_dir=output_dir),
            _write_metadata_command(
                python_bin=python_bin,
                output_dir=output_dir,
                payload=_metadata_payload(
                    job_name=job_name,
                    role=role,
                    expected_archive_sha256=expected_archive_sha256,
                    expected_archive_size_bytes=expected_archive_size_bytes,
                    queue_metadata=queue_metadata,
                    adjudication=adjudication,
                ),
            ),
            _runner_preflight_command(
                python_bin=python_bin,
                output_dir=output_dir,
                eval_device=eval_device,
            ),
            _inflate_runtime_bootstrap_command(
                python_bin=python_bin,
                output_dir=output_dir,
                env=env,
            ),
            _inflate_runtime_static_preflight_command(
                python_bin=python_bin,
                output_dir=output_dir,
            ),
            f"cp {archive} {out}/{ARTIFACT_ARCHIVE}",
            _archive_preflight_command(
                python_bin=python_bin,
                archive_path=f"{output_dir}/{ARTIFACT_ARCHIVE}",
                expected_archive_sha256=expected_archive_sha256,
                expected_archive_size_bytes=expected_archive_size_bytes,
            ),
            (
                f"export UV_PROJECT_ENVIRONMENT={out}/uv_project_env\n"
                "export UV_LINK_MODE=${UV_LINK_MODE:-copy}\n"
                f"{cuda_env_export}"
                f"export PATH=$(dirname {py}):$PATH\n"
                f"{py} -u experiments/contest_auth_eval.py "
                f"--archive {out}/{ARTIFACT_ARCHIVE} "
                f"--inflate-sh {inflate} "
                f"--upstream-dir {upstream} "
                f"--device {eval_device} "
                "--keep-work-dir "
                f"--work-dir {out}/eval_work "
                f"{runtime_tree_arg}"
                f"2>&1 | tee {out}/auth_eval.log"
            ),
            f"test -f {out}/eval_work/contest_auth_eval.json",
            f"cp {out}/eval_work/contest_auth_eval.json {out}/contest_auth_eval.json",
            f"cp {out}/eval_work/provenance.json {out}/eval_provenance.json",
            f"cp {out}/eval_work/report.txt {out}/report.txt",
            (
                f"{py} - {out}/contest_auth_eval.json <<'PY'\n"
                "import json, math, sys\n"
                "payload = json.load(open(sys.argv[1]))\n"
                "prov = payload.get('provenance') or {}\n"
                "assert payload.get('n_samples') == 600, payload.get('n_samples')\n"
                f"assert prov.get('device') == {eval_device!r}, prov.get('device')\n"
                f"assert payload.get('score_axis') == {axis!r}, payload.get('score_axis')\n"
                "score = payload.get('score_recomputed_from_components')\n"
                "assert isinstance(score, (int, float)) and math.isfinite(score), score\n"
                f"expected_sha = {expected_archive_sha256!r}\n"
                f"expected_bytes = {expected_archive_size_bytes!r}\n"
                "if expected_sha is not None:\n"
                "    assert prov.get('archive_sha256') == expected_sha, prov.get('archive_sha256')\n"
                "if expected_bytes is not None:\n"
                "    assert payload.get('archive_size_bytes') == expected_bytes, payload.get('archive_size_bytes')\n"
                f"print({ok_marker!r})\n"
                "print(json.dumps({'score_recomputed_from_components': score, "
                f"'score_axis': {axis!r}, "
                "'archive_sha256': prov.get('archive_sha256'), "
                "'archive_size_bytes': payload.get('archive_size_bytes'), "
                "'gpu_model': prov.get('gpu_model'), "
                "'gpu_t4_match': prov.get('gpu_t4_match')}, sort_keys=True))\n"
                "PY"
            ),
        ]
    )
    if component_trace:
        command = "\n".join(
            [
                command,
                "set +e",
                (
                    f"{py} -u experiments/contest_component_trace.py "
                    f"--submission-dir {out}/eval_work "
                    f"--upstream-dir {upstream} "
                    f"--uncompressed-dir {upstream}/videos "
                    f"--video-names-file {upstream}/public_test_video_names.txt "
                    "--device cuda "
                    f"--contest-auth-eval-json {out}/contest_auth_eval.json "
                    f"--output-json {out}/{ARTIFACT_COMPONENT_TRACE} "
                    f"--top-k {_quote(str(component_trace_top_k))} "
                    f"2>&1 | tee {out}/{ARTIFACT_COMPONENT_TRACE_LOG}"
                ),
                "component_trace_rc=${PIPESTATUS[0]}",
                "set -e",
                (
                    f"{py} - {out}/{ARTIFACT_COMPONENT_TRACE_STATUS} "
                    f"{out}/{ARTIFACT_COMPONENT_TRACE} \"$component_trace_rc\" <<'PY'\n"
                    "import json, math, sys, time\n"
                    "from pathlib import Path\n"
                    "status_path = Path(sys.argv[1])\n"
                    "trace_path = Path(sys.argv[2])\n"
                    "exit_code = int(sys.argv[3])\n"
                    "status = {\n"
                    "    'schema_version': 1,\n"
                    "    'tool': 'exact_cuda_eval_command.component_trace_status',\n"
                    "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
                    "    'score_claim': False,\n"
                    "    'evidence_grade': 'diagnostic_component_trace_status',\n"
                    "    'component_trace_exit_code': exit_code,\n"
                    "    'component_trace_json': str(trace_path),\n"
                    "}\n"
                    "errors = []\n"
                    "summary = {}\n"
                    "if exit_code != 0:\n"
                    "    errors.append(f'component trace command exited {exit_code}')\n"
                    "elif not trace_path.is_file():\n"
                    "    errors.append('component_trace.json was not written')\n"
                    "else:\n"
                    "    try:\n"
                    "        payload = json.loads(trace_path.read_text())\n"
                    "        if payload.get('score_claim') is not False:\n"
                    "            errors.append('component_trace.json score_claim is not false')\n"
                    "        if payload.get('evidence_grade') != 'diagnostic_component_trace':\n"
                    "            errors.append('component_trace.json evidence_grade mismatch')\n"
                    "        if payload.get('n_samples') != 600:\n"
                    "            errors.append(f\"component_trace.json n_samples={payload.get('n_samples')!r}\")\n"
                    "        cross = payload.get('contest_auth_eval_cross_check') or {}\n"
                    "        if not isinstance(cross, dict) or cross.get('all_match') is not True:\n"
                    "            errors.append('component_trace.json cross-check did not match contest_auth_eval.json')\n"
                    "        score = payload.get('score_recomputed_from_components')\n"
                    "        if not isinstance(score, (int, float)) or not math.isfinite(score):\n"
                    "            errors.append(f'component_trace.json score is not finite: {score!r}')\n"
                    "        summary = {\n"
                    "            'n_samples': payload.get('n_samples'),\n"
                    "            'avg_posenet_dist': payload.get('avg_posenet_dist'),\n"
                    "            'avg_segnet_dist': payload.get('avg_segnet_dist'),\n"
                    "            'score_recomputed_from_components': score,\n"
                    "        }\n"
                    "    except Exception as exc:\n"
                    "        errors.append(f'component_trace.json validation error: {exc}')\n"
                    "status['status'] = 'ok' if not errors else 'diagnostic_unavailable_or_invalid'\n"
                    "status['errors'] = errors\n"
                    "status['summary'] = summary\n"
                    "status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + '\\n')\n"
                    "print('LIGHTNING_COMPONENT_TRACE_STATUS')\n"
                    "print(json.dumps({'status': status['status'], 'errors': errors, **summary}, sort_keys=True))\n"
                    "PY"
                ),
            ]
        )
    command = "\n".join(
        [
            command,
            _adjudication_command(
                python_bin=python_bin,
                output_dir=output_dir,
                adjudication=adjudication,
            ),
            f"test -f {out}/{_quote(adjudication.provenance_name)}",
            f"test -f {out}/{_quote(adjudication.result_copy_name)}",
        ]
    )
    return command


def make_exact_eval_spec(
    *,
    name: str,
    archive_path: str,
    repo_dir: str,
    upstream_dir: str,
    output_dir: str | None = None,
    machine: str = "T4",
    studio: str | None = None,
    image: str | None = None,
    python_bin: str = ".venv/bin/python",
    inflate_sh: str = "submissions/robust_current/inflate.sh",
    max_runtime: int | None = 3 * 60 * 60,
    env: dict[str, str] | None = None,
    teamspace: str | None = None,
    org: str | None = None,
    user: str | None = None,
    cloud_account: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    queue_metadata: dict[str, Any] | None = None,
    local_artifact_dir: str | None = None,
    adjudication: LightningAdjudicationSpec | None = None,
    component_trace: bool = False,
    component_trace_top_k: int = 80,
    eval_device: str = "cuda",
) -> LightningBatchJobSpec:
    """Create a strict exact-eval job spec."""

    eval_device = _normalise_eval_device(eval_device)
    role = _exact_eval_role(eval_device)
    out = output_dir or default_exact_eval_output_dir(repo_dir=repo_dir, job_name=name)
    command = exact_cuda_eval_command(
        repo_dir=repo_dir,
        archive_path=archive_path,
        upstream_dir=upstream_dir,
        output_dir=out,
        python_bin=python_bin,
        inflate_sh=inflate_sh,
        job_name=name,
        expected_archive_sha256=expected_archive_sha256,
        expected_archive_size_bytes=expected_archive_size_bytes,
        queue_metadata=queue_metadata,
        env=env,
        adjudication=adjudication,
        component_trace=component_trace,
        component_trace_top_k=component_trace_top_k,
        eval_device=eval_device,
    )
    local_out = local_artifact_dir or default_exact_eval_local_artifact_dir(job_name=name)
    spec = LightningBatchJobSpec(
        name=name,
        machine=machine,
        command=command,
        studio=studio,
        image=image,
        teamspace=teamspace,
        org=org,
        user=user,
        cloud_account=cloud_account,
        env=dict(env or {}),
        interruptible=False,
        max_runtime=max_runtime,
        reuse_snapshot=False,
        role=role,
        expected_archive_sha256=expected_archive_sha256,
        expected_archive_size_bytes=expected_archive_size_bytes,
        queue_metadata=dict(queue_metadata or {}),
        local_artifact_dir=local_out,
        remote_output_dir=out,
        adjudication=adjudication,
    )
    spec.validate()
    return spec


def _diagnostic_component_sensitivity_input_preflight_command(
    *,
    python_bin: str,
    output_dir: str,
    baseline_archive_path: str,
    expected_baseline_archive_sha256: str | None,
    expected_baseline_archive_size_bytes: int | None,
    pair_weights_path: str | None,
) -> str:
    py = _quote(python_bin)
    out_path = _quote(f"{output_dir}/{ARTIFACT_COMPONENT_SENSITIVITY_INPUTS}")
    extract_dir = _quote(f"{output_dir}/extracted")
    baseline = _quote(baseline_archive_path)
    expected_sha = _quote(expected_baseline_archive_sha256 or "")
    expected_bytes = _quote(str(expected_baseline_archive_size_bytes or ""))
    pair_weights = _quote(pair_weights_path or "")
    return (
        f"{py} - {out_path} {extract_dir} {baseline} {expected_sha} {expected_bytes} {pair_weights} <<'PY'\n"
        "import hashlib, json, pathlib, shutil, sys, time, zipfile\n"
        "\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "extract_dir = pathlib.Path(sys.argv[2])\n"
        "baseline_archive = pathlib.Path(sys.argv[3])\n"
        "expected_sha = sys.argv[4] or None\n"
        "expected_bytes = int(sys.argv[5]) if sys.argv[5] else None\n"
        "pair_weights_arg = sys.argv[6]\n"
        "pair_weights = pathlib.Path(pair_weights_arg) if pair_weights_arg else None\n"
        "required_members = ('renderer.bin', 'masks.mkv', 'optimized_poses.bin')\n"
        "\n"
        "def sha256(path):\n"
        "    h = hashlib.sha256()\n"
        "    with path.open('rb') as f:\n"
        "        for chunk in iter(lambda: f.read(1024 * 1024), b''):\n"
        "            h.update(chunk)\n"
        "    return h.hexdigest()\n"
        "\n"
        "def file_meta(path, *, label):\n"
        "    path = path.resolve()\n"
        "    if not path.is_file():\n"
        "        raise SystemExit(f'FATAL: {label} not found: {path}')\n"
        "    size = path.stat().st_size\n"
        "    if size <= 0:\n"
        "        raise SystemExit(f'FATAL: {label} is empty: {path}')\n"
        "    return {'path': str(path), 'bytes': size, 'sha256': sha256(path)}\n"
        "\n"
        "archive_meta = file_meta(baseline_archive, label='baseline archive')\n"
        "if expected_sha is not None and archive_meta['sha256'] != expected_sha:\n"
        "    raise SystemExit(f\"FATAL: baseline archive sha256 mismatch: expected={expected_sha} actual={archive_meta['sha256']}\")\n"
        "if expected_bytes is not None and archive_meta['bytes'] != expected_bytes:\n"
        "    raise SystemExit(f\"FATAL: baseline archive bytes mismatch: expected={expected_bytes} actual={archive_meta['bytes']}\")\n"
        "\n"
        "extract_root = extract_dir.resolve()\n"
        "if extract_root.exists():\n"
        "    shutil.rmtree(extract_root)\n"
        "extract_root.mkdir(parents=True, exist_ok=True)\n"
        "seen = set()\n"
        "infos = {}\n"
        "member_names = []\n"
        "try:\n"
        "    with zipfile.ZipFile(baseline_archive, 'r') as zf:\n"
        "        for info in zf.infolist():\n"
        "            name = info.filename\n"
        "            pure = pathlib.PurePosixPath(name)\n"
        "            parts = pure.parts\n"
        "            if not name or '\\\\' in name or pure.is_absolute() or '..' in parts:\n"
        "                raise SystemExit(f'FATAL: zip-slip member in baseline archive: {name!r}')\n"
        "            if name in seen:\n"
        "                raise SystemExit(f'FATAL: duplicate zip member in baseline archive: {name!r}')\n"
        "            seen.add(name)\n"
        "            if any(part in {'__MACOSX', '.DS_Store'} or part.startswith('._') for part in parts):\n"
        "                raise SystemExit(f'FATAL: hidden/resource-fork zip member in baseline archive: {name!r}')\n"
        "            member_names.append(name)\n"
        "            infos[name] = info\n"
        "        missing = [name for name in required_members if name not in infos]\n"
        "        if missing:\n"
        "            raise SystemExit('FATAL: baseline archive missing required members: ' + ', '.join(missing))\n"
        "        extracted = {}\n"
        "        for name in required_members:\n"
        "            info = infos[name]\n"
        "            if info.is_dir():\n"
        "                raise SystemExit(f'FATAL: required archive member is a directory: {name!r}')\n"
        "            target = (extract_root / name).resolve()\n"
        "            if target.parent != extract_root:\n"
        "                raise SystemExit(f'FATAL: unsafe extraction target for member: {name!r}')\n"
        "            with zf.open(info, 'r') as src, target.open('wb') as dst:\n"
        "                shutil.copyfileobj(src, dst)\n"
        "            meta = file_meta(target, label=f'extracted {name}')\n"
        "            meta.update({\n"
        "                'member': name,\n"
        "                'zip_crc': int(info.CRC),\n"
        "                'zip_file_size': int(info.file_size),\n"
        "                'zip_compress_size': int(info.compress_size),\n"
        "            })\n"
        "            extracted[name] = meta\n"
        "except zipfile.BadZipFile as exc:\n"
        "    raise SystemExit(f'FATAL: baseline archive is not a readable zip: {baseline_archive}: {exc}') from exc\n"
        "pair_weights_meta = file_meta(pair_weights, label='pair weights') if pair_weights else None\n"
        "archive_meta.update({\n"
        "    'zip_member_count': len(member_names),\n"
        "    'required_members': list(required_members),\n"
        "    'member_names': member_names,\n"
        "})\n"
        "payload = {\n"
        "    'schema_version': 1,\n"
        "    'tool': 'lightning_diagnostic_component_sensitivity_input_preflight',\n"
        "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "    'baseline_archive': archive_meta,\n"
        "    'extracted_dir': str(extract_root),\n"
        "    'extracted_members': extracted,\n"
        "    'pair_weights': pair_weights_meta,\n"
        "    'diagnostic': True,\n"
        "    'score_claim': False,\n"
        "    'promotion_eligible': False,\n"
        "    'promotion_blockers': [\n"
        "        'profile_component_sensitivity.py emits diagnostic direct-renderer/Fisher artifacts, not canonical archive.zip -> inflate.sh -> upstream/evaluate.py component-response evidence'\n"
        "    ],\n"
        "}\n"
        "out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "print('LIGHTNING_DIAGNOSTIC_COMPONENT_SENSITIVITY_INPUT_PREFLIGHT_OK')\n"
        "print(json.dumps({'baseline_sha256': archive_meta['sha256'], 'baseline_bytes': archive_meta['bytes'], 'extracted_members': list(required_members)}, sort_keys=True))\n"
        "PY"
    )


def _diagnostic_component_sensitivity_run_metadata_command(
    *,
    python_bin: str,
    output_dir: str,
    profile_argv: list[str],
) -> str:
    py = _quote(python_bin)
    out_path = _quote(f"{output_dir}/{ARTIFACT_COMPONENT_SENSITIVITY_RUN}")
    argv_json = _quote(json.dumps(profile_argv, sort_keys=True))
    return (
        f"{py} - {out_path} {argv_json} <<'PY'\n"
        "import json, os, sys, time\n"
        "profile_argv = json.loads(sys.argv[2])\n"
        "env_keys = [\n"
        "    'CUDA_VISIBLE_DEVICES',\n"
        "    'LIGHTNING_CLOUD_PROJECT_ID',\n"
        "    'LIGHTNING_DISPATCHED_JOB_NAME',\n"
        "    'UV_LINK_MODE',\n"
        "    'UV_PROJECT_ENVIRONMENT',\n"
        "]\n"
        "payload = {\n"
        "    'schema_version': 1,\n"
        "    'tool': 'lightning_diagnostic_component_sensitivity_run_metadata',\n"
        "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "    'role': 'diagnostic_component_sensitivity',\n"
        "    'profile_argv': profile_argv,\n"
        "    'environment': {key: os.environ.get(key) for key in env_keys},\n"
        "    'diagnostic': True,\n"
        "    'score_claim': False,\n"
        "    'promotion_eligible': False,\n"
        "    'promotion_blockers': [\n"
        "        'not canonical component-response custody; output is diagnostic only'\n"
        "    ],\n"
        "}\n"
        "open(sys.argv[1], 'w').write(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "print('LIGHTNING_DIAGNOSTIC_COMPONENT_SENSITIVITY_RUN_METADATA_OK')\n"
        "PY"
    )


def _diagnostic_component_sensitivity_validation_command(
    *,
    python_bin: str,
    output_dir: str,
) -> str:
    py = _quote(python_bin)
    out = _quote(output_dir)
    expected_files = _quote(json.dumps(COMPONENT_SENSITIVITY_PROFILE_FILES, sort_keys=True))
    allowed_sources = _quote(json.dumps(sorted(COMPONENT_SENSITIVITY_DIAGNOSTIC_SOURCES)))
    components = _quote(json.dumps(COMPONENT_RESPONSE_COMPONENTS))
    return (
        f"{py} - {out} {expected_files} {allowed_sources} {components} <<'PY'\n"
        "import json, math, pathlib, sys, time\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "expected_files = json.loads(sys.argv[2])\n"
        "allowed_sources = set(json.loads(sys.argv[3]))\n"
        "components = json.loads(sys.argv[4])\n"
        "summary_path = root / 'component_sensitivity_profile_summary.json'\n"
        "inputs_path = root / 'diagnostic_component_sensitivity_inputs.json'\n"
        "run_path = root / 'diagnostic_component_sensitivity_run.json'\n"
        "for path in (summary_path, inputs_path, run_path):\n"
        "    if not path.is_file():\n"
        "        raise SystemExit(f'FATAL: missing diagnostic component-sensitivity artifact: {path}')\n"
        "summary = json.loads(summary_path.read_text())\n"
        "inputs = json.loads(inputs_path.read_text())\n"
        "run = json.loads(run_path.read_text())\n"
        "from tac.sensitivity_map import load_sensitivity_map\n"
        "if inputs.get('score_claim') is not False or inputs.get('promotion_eligible') is not False:\n"
        "    raise SystemExit('FATAL: diagnostic component-sensitivity inputs must be non-score and non-promotable')\n"
        "if run.get('score_claim') is not False or run.get('promotion_eligible') is not False:\n"
        "    raise SystemExit('FATAL: diagnostic component-sensitivity run metadata must be non-score and non-promotable')\n"
        "if summary.get('tool') != 'experiments/profile_component_sensitivity.py':\n"
        "    raise SystemExit('FATAL: profile summary has unexpected tool')\n"
        "if summary.get('device') != 'cuda':\n"
        "    raise SystemExit(f\"FATAL: profile summary device={summary.get('device')!r}, expected 'cuda'\")\n"
        "if summary.get('score_claim') is not False:\n"
        "    raise SystemExit('FATAL: diagnostic profile summary must have score_claim=false')\n"
        "if summary.get('promotion_eligible') is not False:\n"
        "    raise SystemExit('FATAL: diagnostic profile summary must have promotion_eligible=false')\n"
        "if summary.get('official_component_response') is not False:\n"
        "    raise SystemExit('FATAL: diagnostic profile summary must not claim official_component_response')\n"
        "if summary.get('canonical_scorer_path') is not False:\n"
        "    raise SystemExit('FATAL: diagnostic profile summary must not claim canonical_scorer_path')\n"
        "if 'diagnostic' not in str(summary.get('evidence_grade', '')):\n"
        "    raise SystemExit('FATAL: diagnostic profile summary has non-diagnostic evidence_grade')\n"
        "sensitivity_source = summary.get('sensitivity_source')\n"
        "if sensitivity_source not in allowed_sources:\n"
        "    raise SystemExit(f'FATAL: unsupported diagnostic sensitivity_source={sensitivity_source!r}')\n"
        "profile_argv = run.get('profile_argv')\n"
        "if not isinstance(profile_argv, list) or '--device' not in profile_argv:\n"
        "    raise SystemExit('FATAL: diagnostic run metadata must record profile argv including --device')\n"
        "if sensitivity_source == 'direct_renderer_cuda_finite_difference_component_response':\n"
        "    if summary.get('promotion_requested') is not True:\n"
        "        raise SystemExit('FATAL: direct finite-difference summary must record promotion_requested=true')\n"
        "    if '--promotion-finite-difference' not in profile_argv or '--all-pairs' not in profile_argv:\n"
        "        raise SystemExit('FATAL: direct finite-difference run metadata must include --promotion-finite-difference and --all-pairs')\n"
        "    if summary.get('n_pairs_total') != 600:\n"
        "        raise SystemExit('FATAL: direct finite-difference summary must cover all 600 contest pairs')\n"
        "    finite_eps = summary.get('finite_difference_epsilon')\n"
        "    if isinstance(finite_eps, bool) or not isinstance(finite_eps, (int, float)) or float(finite_eps) <= 0.0:\n"
        "        raise SystemExit('FATAL: direct finite-difference summary must record positive finite_difference_epsilon')\n"
        "elif '--promotion-finite-difference' in profile_argv:\n"
        "    raise SystemExit('FATAL: fisher_proxy diagnostic run must not include --promotion-finite-difference')\n"
        "sample_plan = json.loads((root / 'sample_plan.json').read_text())\n"
        "seen_pairs = set()\n"
        "for group_name in ('calibration_pairs', 'holdout_pairs'):\n"
        "    rows = sample_plan.get(group_name)\n"
        "    if not isinstance(rows, list):\n"
        "        raise SystemExit(f'FATAL: sample_plan missing {group_name}')\n"
        "    for idx, row in enumerate(rows):\n"
        "        if not isinstance(row, dict):\n"
        "            raise SystemExit(f'FATAL: sample_plan.{group_name}[{idx}] must be an object')\n"
        "        pair_index = row.get('pair_index')\n"
        "        if isinstance(pair_index, bool) or not isinstance(pair_index, int):\n"
        "            raise SystemExit(f'FATAL: sample_plan.{group_name}[{idx}].pair_index must be int')\n"
        "        if row.get('video') != 0 or row.get('t') != 2 * pair_index or row.get('t1') != 2 * pair_index + 1:\n"
        "            raise SystemExit(f'FATAL: sample_plan.{group_name}[{idx}] does not use absolute contest pair ids')\n"
        "        if pair_index in seen_pairs:\n"
        "            raise SystemExit(f'FATAL: duplicate sample_plan pair_index={pair_index}')\n"
        "        seen_pairs.add(pair_index)\n"
        "sample_plan_validation = {\n"
        "    'pair_count': len(seen_pairs),\n"
        "    'full_600_pair_coverage': seen_pairs == set(range(600)),\n"
        "    'split_hash': sample_plan.get('split_hash'),\n"
        "}\n"
        "fd_shard = summary.get('finite_difference_shard')\n"
        "fd_is_partial_shard = isinstance(fd_shard, dict) and fd_shard.get('is_shard') is True\n"
        "fd_merge = summary.get('finite_difference_merge')\n"
        "fd_merged = isinstance(fd_merge, dict) and fd_merge.get('schema') == 'component_sensitivity_direct_fd_merge_v1'\n"
        "missing = [name for name in expected_files if not (root / name).is_file()]\n"
        "if missing:\n"
        "    raise SystemExit('FATAL: missing profile output artifacts: ' + ', '.join(missing))\n"
        "elapsed = summary.get('elapsed_s')\n"
        "if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)):\n"
        "    raise SystemExit('FATAL: profile summary elapsed_s must be finite')\n"
        "curves = {}\n"
        "map_metadata_by_component = {}\n"
        "for component in components:\n"
        "    map_path = root / f'{component}_sensitivity_map.pt'\n"
        "    _map_values, map_metadata = load_sensitivity_map(map_path)\n"
        "    map_metadata_by_component[component] = map_metadata\n"
        "    if map_metadata.get('device') != 'cuda':\n"
        "        raise SystemExit(f\"FATAL: {component} sensitivity map device={map_metadata.get('device')!r}, expected 'cuda'\")\n"
        "    if map_metadata.get('component') != component and map_metadata.get('scorer_target') != component:\n"
        "        raise SystemExit(f'FATAL: {component} sensitivity map metadata does not identify the component')\n"
        "    if map_metadata.get('score_claim') is not False or map_metadata.get('promotion_eligible') is not False:\n"
        "        raise SystemExit(f'FATAL: {component} sensitivity map must be non-score and non-promotable')\n"
        "    if map_metadata.get('official_component_response') is not False:\n"
        "        raise SystemExit(f'FATAL: {component} sensitivity map must not claim official_component_response')\n"
        "    if map_metadata.get('canonical_scorer_path') is not False:\n"
        "        raise SystemExit(f'FATAL: {component} sensitivity map must not claim canonical_scorer_path')\n"
        "    if map_metadata.get('sensitivity_source') != sensitivity_source:\n"
        "        raise SystemExit(f'FATAL: {component} sensitivity map sensitivity_source does not match summary')\n"
        "    if map_metadata.get('finite_difference_shard') != fd_shard:\n"
        "        raise SystemExit(f'FATAL: {component} sensitivity map shard metadata does not match summary')\n"
        "    _holdout_values, holdout_metadata = load_sensitivity_map(root / f'{component}_holdout_sensitivity_map.pt')\n"
        "    if holdout_metadata.get('split') != 'holdout':\n"
        "        raise SystemExit(f'FATAL: {component} holdout sensitivity map must record split=holdout')\n"
        "    if holdout_metadata.get('finite_difference_shard') != fd_shard:\n"
        "        raise SystemExit(f'FATAL: {component} holdout sensitivity map shard metadata does not match summary')\n"
        "    curve_path = root / f'{component}_response_curve.json'\n"
        "    curve = json.loads(curve_path.read_text())\n"
        "    curves[component] = curve\n"
        "    if curve.get('component') != component:\n"
        "        raise SystemExit(f\"FATAL: {component} response curve component={curve.get('component')!r}\")\n"
        "    if curve.get('device') != 'cuda':\n"
        "        raise SystemExit(f\"FATAL: {component} response curve device={curve.get('device')!r}, expected 'cuda'\")\n"
        "    if curve.get('score_claim') is not False or curve.get('promotion_eligible') is not False:\n"
        "        raise SystemExit(f'FATAL: {component} response curve must be non-score and non-promotable')\n"
        "    if curve.get('official_component_response') is not False:\n"
        "        raise SystemExit(f'FATAL: {component} response curve must not claim official_component_response')\n"
        "    if curve.get('canonical_scorer_path') is not False:\n"
        "        raise SystemExit(f'FATAL: {component} response curve must not claim canonical_scorer_path')\n"
        "    if curve.get('sensitivity_source') != sensitivity_source:\n"
        "        raise SystemExit(f'FATAL: {component} response curve sensitivity_source does not match summary')\n"
        "    if curve.get('component_response_path') != 'direct_renderer_tensor_inprocess_scorer':\n"
        "        raise SystemExit(f'FATAL: {component} response curve must remain diagnostic direct-renderer response')\n"
        "certification_handoff_eligible = bool(\n"
        "    sensitivity_source == 'direct_renderer_cuda_finite_difference_component_response'\n"
        "    and sample_plan_validation['full_600_pair_coverage'] is True\n"
        "    and not fd_is_partial_shard\n"
        "    and (not isinstance(fd_shard, dict) or fd_shard.get('merge_required_for_certification_handoff') is False)\n"
        ")\n"
        "if certification_handoff_eligible and isinstance(fd_shard, dict) and fd_shard.get('merged_from_shards') is True:\n"
        "    certification_handoff_eligible = fd_merged\n"
        "payload = {\n"
        "    'schema_version': 1,\n"
        "    'validated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "    'role': 'diagnostic_component_sensitivity',\n"
        "    'artifact_dir': str(root),\n"
        "    'baseline_archive_sha256': inputs['baseline_archive']['sha256'],\n"
        "    'baseline_archive_size_bytes': inputs['baseline_archive']['bytes'],\n"
        "    'device': 'cuda',\n"
        "    'evidence_grade': summary.get('evidence_grade'),\n"
        "    'sensitivity_source': summary.get('sensitivity_source'),\n"
        "    'component_response_path': summary.get('component_response_path'),\n"
        "    'planning_eligible': True,\n"
        "    'sample_plan_validation': sample_plan_validation,\n"
        "    'certification_handoff_eligible': certification_handoff_eligible,\n"
        "    'promotion_eligible': False,\n"
        "    'score_claim': False,\n"
        "    'diagnostic': True,\n"
        "    'input_preflight': inputs,\n"
        "    'run_metadata': run,\n"
        "    'summary': summary,\n"
        "    'curves': curves,\n"
        "    'map_metadata': map_metadata_by_component,\n"
        "    'preserved_files': expected_files,\n"
        "}\n"
        "(root / 'diagnostic_component_sensitivity_artifact_validation.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "print('LIGHTNING_DIAGNOSTIC_COMPONENT_SENSITIVITY_ARTIFACTS_OK')\n"
        "print(json.dumps({'baseline_sha256': payload['baseline_archive_sha256'], 'promotion_eligible': False, 'score_claim': False}, sort_keys=True))\n"
        "PY"
    )


def diagnostic_component_sensitivity_command(
    *,
    repo_dir: str,
    baseline_archive_path: str,
    upstream_dir: str,
    output_dir: str,
    python_bin: str = ".venv/bin/python",
    video_mkv: str | None = None,
    pair_weights_path: str | None = None,
    job_name: str | None = None,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    queue_metadata: dict[str, Any] | None = None,
    top_k_pairs: int = 64,
    pair_batch: int = 2,
    response_top_k: int = 16,
    response_epsilons: str = "-0.002,-0.001,-0.0005,0.0,0.0005,0.001,0.002",
    split_seed: int = 20260430,
    holdout_fraction: float = 0.2,
    aggregate: str = "sum",
    promotion_finite_difference: bool = False,
    finite_difference_epsilon: float = 0.001,
    finite_difference_shard_index: int = 0,
    finite_difference_shard_count: int = 1,
) -> str:
    """Build a CUDA diagnostic component-sensitivity profiler command."""

    _validate_writable_output_dir(output_dir)
    _validate_expected_archive(
        expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes,
    )
    if aggregate not in {"sum", "mean", "max"}:
        raise ValueError("aggregate must be one of: sum, mean, max")
    for field, value in (
        ("top_k_pairs", top_k_pairs),
        ("pair_batch", pair_batch),
        ("response_top_k", response_top_k),
    ):
        if int(value) <= 0:
            raise ValueError(f"{field} must be positive")
    _validate_optional_number(float(holdout_fraction), field="holdout_fraction", minimum=0.0, strict_minimum=True)
    if float(holdout_fraction) >= 1.0:
        raise ValueError("holdout_fraction must be < 1.0")
    _validate_optional_number(
        float(finite_difference_epsilon),
        field="finite_difference_epsilon",
        minimum=0.0,
        strict_minimum=True,
    )
    if promotion_finite_difference and pair_weights_path:
        raise ValueError("promotion_finite_difference requires --all-pairs; do not pass pair_weights_path")
    if int(finite_difference_shard_count) <= 0:
        raise ValueError("finite_difference_shard_count must be positive")
    if int(finite_difference_shard_index) < 0 or int(finite_difference_shard_index) >= int(finite_difference_shard_count):
        raise ValueError("finite_difference_shard_index must be within shard count")
    if not promotion_finite_difference and (
        int(finite_difference_shard_index) != 0 or int(finite_difference_shard_count) != 1
    ):
        raise ValueError("finite-difference shard flags require promotion_finite_difference")

    repo = _quote(repo_dir)
    out = _quote(output_dir)
    video = video_mkv or f"{str(upstream_dir).rstrip('/')}/videos/0.mkv"
    profile_argv = [
        python_bin,
        "-u",
        "experiments/profile_component_sensitivity.py",
        "--checkpoint",
        f"{output_dir}/extracted/renderer.bin",
        "--video",
        video,
        "--masks-mkv",
        f"{output_dir}/extracted/masks.mkv",
        "--poses",
        f"{output_dir}/extracted/optimized_poses.bin",
        "--upstream",
        upstream_dir,
        "--output-dir",
        output_dir,
        "--top-k-pairs",
        str(int(top_k_pairs)),
        "--pair-batch",
        str(int(pair_batch)),
        "--response-top-k",
        str(int(response_top_k)),
        f"--response-epsilons={response_epsilons}",
        "--split-seed",
        str(int(split_seed)),
        "--holdout-fraction",
        str(float(holdout_fraction)),
        "--aggregate",
        aggregate,
        "--device",
        "cuda",
    ]
    if promotion_finite_difference:
        profile_argv.extend(
            [
                "--promotion-finite-difference",
                "--finite-difference-epsilon",
                str(float(finite_difference_epsilon)),
                "--finite-difference-shard-index",
                str(int(finite_difference_shard_index)),
                "--finite-difference-shard-count",
                str(int(finite_difference_shard_count)),
            ]
        )
    if pair_weights_path:
        profile_argv.extend(["--pair-weights", pair_weights_path])
    else:
        profile_argv.append("--all-pairs")
    profile_command = " ".join(_quote(part) for part in profile_argv)

    metadata_payload = {
        "schema_version": 1,
        "job_name": job_name,
        "role": "diagnostic_component_sensitivity",
        "expected_archive_sha256": expected_baseline_archive_sha256,
        "expected_archive_size_bytes": expected_baseline_archive_size_bytes,
        "expected_baseline_archive_sha256": expected_baseline_archive_sha256,
        "expected_baseline_archive_size_bytes": expected_baseline_archive_size_bytes,
        "queue_metadata": _normalise_metadata(queue_metadata),
        "adjudication": None,
        "score_source": "none:diagnostic_component_sensitivity_non_promotable",
        "score_claim": False,
        "promotion_eligible": False,
        "finite_difference_shard_index": int(finite_difference_shard_index),
        "finite_difference_shard_count": int(finite_difference_shard_count),
        "status_source": "lightning_sdk_job_attributes",
    }

    files_to_remove = [
        ARTIFACT_METADATA,
        ARTIFACT_COMPONENT_SENSITIVITY_INPUTS,
        ARTIFACT_COMPONENT_SENSITIVITY_RUN,
        ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION,
        ARTIFACT_COMPONENT_SENSITIVITY_LOG,
        ARTIFACT_DALI_BOOTSTRAP,
        ARTIFACT_DALI_REQUIREMENTS,
        ARTIFACT_RUNNER_PREFLIGHT,
        ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
        ARTIFACT_SUPPLY_CHAIN_SCAN,
        *COMPONENT_SENSITIVITY_PROFILE_FILES,
    ]
    command = "\n".join(
        [
            "set -euo pipefail",
            f"cd {repo}",
            "test -f env.sh && source env.sh || true",
            f"mkdir -p {out}",
            f"rm -rf {out}/extracted {out}/uv_project_env",
            " ".join(["rm", "-f", *[f"{out}/{_quote(name)}" for name in files_to_remove]]),
            _ensure_remote_uv_command(output_dir=output_dir),
            _write_metadata_command(
                python_bin=python_bin,
                output_dir=output_dir,
                payload=metadata_payload,
            ),
            _runner_preflight_command(python_bin=python_bin, output_dir=output_dir),
            _diagnostic_component_sensitivity_input_preflight_command(
                python_bin=python_bin,
                output_dir=output_dir,
                baseline_archive_path=baseline_archive_path,
                expected_baseline_archive_sha256=expected_baseline_archive_sha256,
                expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
                pair_weights_path=pair_weights_path,
            ),
            (
                f"export UV_PROJECT_ENVIRONMENT={out}/uv_project_env\n"
                "export UV_LINK_MODE=${UV_LINK_MODE:-copy}\n"
                + _diagnostic_component_sensitivity_run_metadata_command(
                    python_bin=python_bin,
                    output_dir=output_dir,
                    profile_argv=profile_argv,
                )
            ),
            profile_command + f" 2>&1 | tee {out}/{ARTIFACT_COMPONENT_SENSITIVITY_LOG}",
            f"test -f {out}/{ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY}",
            *[
                f"test -f {out}/{_quote(name)}"
                for name in COMPONENT_SENSITIVITY_PROFILE_FILES
            ],
            _diagnostic_component_sensitivity_validation_command(
                python_bin=python_bin,
                output_dir=output_dir,
            ),
            f"test -f {out}/{ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION}",
        ]
    )
    return command


def make_diagnostic_component_sensitivity_spec(
    *,
    name: str,
    baseline_archive_path: str,
    repo_dir: str,
    upstream_dir: str,
    output_dir: str | None = None,
    machine: str = "T4",
    studio: str | None = None,
    image: str | None = None,
    python_bin: str = ".venv/bin/python",
    max_runtime: int | None = 6 * 60 * 60,
    env: dict[str, str] | None = None,
    teamspace: str | None = None,
    org: str | None = None,
    user: str | None = None,
    cloud_account: str | None = None,
    video_mkv: str | None = None,
    pair_weights_path: str | None = None,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    queue_metadata: dict[str, Any] | None = None,
    local_artifact_dir: str | None = None,
    top_k_pairs: int = 64,
    pair_batch: int = 2,
    response_top_k: int = 16,
    response_epsilons: str = "-0.002,-0.001,-0.0005,0.0,0.0005,0.001,0.002",
    split_seed: int = 20260430,
    holdout_fraction: float = 0.2,
    aggregate: str = "sum",
    promotion_finite_difference: bool = False,
    finite_difference_epsilon: float = 0.001,
    finite_difference_shard_index: int = 0,
    finite_difference_shard_count: int = 1,
) -> LightningBatchJobSpec:
    """Create a non-promotable diagnostic component-sensitivity Batch Job spec."""

    out = output_dir or default_exact_eval_output_dir(repo_dir=repo_dir, job_name=name)
    command = diagnostic_component_sensitivity_command(
        repo_dir=repo_dir,
        baseline_archive_path=baseline_archive_path,
        upstream_dir=upstream_dir,
        output_dir=out,
        python_bin=python_bin,
        video_mkv=video_mkv,
        pair_weights_path=pair_weights_path,
        job_name=name,
        expected_baseline_archive_sha256=expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
        queue_metadata=queue_metadata,
        top_k_pairs=top_k_pairs,
        pair_batch=pair_batch,
        response_top_k=response_top_k,
        response_epsilons=response_epsilons,
        split_seed=split_seed,
        holdout_fraction=holdout_fraction,
        aggregate=aggregate,
        promotion_finite_difference=promotion_finite_difference,
        finite_difference_epsilon=finite_difference_epsilon,
        finite_difference_shard_index=finite_difference_shard_index,
        finite_difference_shard_count=finite_difference_shard_count,
    )
    spec = LightningBatchJobSpec(
        name=name,
        machine=machine,
        command=command,
        studio=studio,
        image=image,
        teamspace=teamspace,
        org=org,
        user=user,
        cloud_account=cloud_account,
        env=dict(env or {}),
        interruptible=False,
        max_runtime=max_runtime,
        reuse_snapshot=False,
        role="diagnostic_component_sensitivity",
        expected_archive_sha256=expected_baseline_archive_sha256,
        expected_archive_size_bytes=expected_baseline_archive_size_bytes,
        queue_metadata=dict(queue_metadata or {}),
        local_artifact_dir=local_artifact_dir,
        remote_output_dir=out,
        adjudication=None,
    )
    spec.validate()
    return spec


def _component_response_input_preflight_command(
    *,
    python_bin: str,
    output_dir: str,
    baseline_archive_path: str,
    perturbation_plan_path: str,
    baseline_contest_auth_eval_json: str | None,
    expected_baseline_archive_sha256: str | None,
    expected_baseline_archive_size_bytes: int | None,
) -> str:
    py = _quote(python_bin)
    out_path = _quote(f"{output_dir}/{ARTIFACT_COMPONENT_RESPONSE_INPUTS}")
    baseline = _quote(baseline_archive_path)
    plan = _quote(perturbation_plan_path)
    baseline_json = _quote(baseline_contest_auth_eval_json or "")
    expected_sha = _quote(expected_baseline_archive_sha256 or "")
    expected_bytes = _quote(str(expected_baseline_archive_size_bytes or ""))
    return (
        f"{py} - {out_path} {baseline} {plan} {baseline_json} {expected_sha} {expected_bytes} <<'PY'\n"
        "import hashlib, json, pathlib, sys, time, zipfile\n"
        "\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "baseline_archive = pathlib.Path(sys.argv[2])\n"
        "plan_path = pathlib.Path(sys.argv[3])\n"
        "baseline_json_arg = sys.argv[4]\n"
        "baseline_json = pathlib.Path(baseline_json_arg) if baseline_json_arg else None\n"
        "expected_sha = sys.argv[5] or None\n"
        "expected_bytes = int(sys.argv[6]) if sys.argv[6] else None\n"
        "\n"
        "def sha256(path):\n"
        "    h = hashlib.sha256()\n"
        "    with path.open('rb') as f:\n"
        "        for chunk in iter(lambda: f.read(1024 * 1024), b''):\n"
        "            h.update(chunk)\n"
        "    return h.hexdigest()\n"
        "\n"
        "def file_meta(path, *, label, archive=False):\n"
        "    path = path.resolve()\n"
        "    if not path.is_file():\n"
        "        raise SystemExit(f'FATAL: {label} not found: {path}')\n"
        "    meta = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}\n"
        "    if meta['bytes'] <= 0:\n"
        "        raise SystemExit(f'FATAL: {label} is empty: {path}')\n"
        "    if archive:\n"
        "        names = []\n"
        "        seen = set()\n"
        "        try:\n"
        "            with zipfile.ZipFile(path, 'r') as zf:\n"
        "                for info in zf.infolist():\n"
        "                    name = info.filename\n"
        "                    parts = pathlib.PurePosixPath(name).parts\n"
        "                    if not name or name.startswith('/') or '..' in parts:\n"
        "                        raise SystemExit(f'FATAL: zip-slip member in {label}: {name!r}')\n"
        "                    if name in seen:\n"
        "                        raise SystemExit(f'FATAL: duplicate zip member in {label}: {name!r}')\n"
        "                    seen.add(name)\n"
        "                    if any(part in {'__MACOSX', '.DS_Store'} or part.startswith('._') for part in parts):\n"
        "                        raise SystemExit(f'FATAL: hidden/resource-fork zip member in {label}: {name!r}')\n"
        "                    names.append(name)\n"
        "        except zipfile.BadZipFile as exc:\n"
        "            raise SystemExit(f'FATAL: {label} is not a readable zip: {path}: {exc}') from exc\n"
        "        if not names:\n"
        "            raise SystemExit(f'FATAL: {label} archive has no members: {path}')\n"
        "        meta['zip_member_count'] = len(names)\n"
        "    return meta\n"
        "\n"
        "def load_json(path, *, label):\n"
        "    try:\n"
        "        payload = json.loads(path.read_text())\n"
        "    except json.JSONDecodeError as exc:\n"
        "        raise SystemExit(f'FATAL: {label} is invalid JSON: {path}: {exc}') from exc\n"
        "    return payload\n"
        "\n"
        "def resolve(value, *, root):\n"
        "    path = pathlib.Path(str(value))\n"
        "    return path.resolve() if path.is_absolute() else (root / path).resolve()\n"
        "\n"
        "baseline_meta = file_meta(baseline_archive, label='baseline archive', archive=True)\n"
        "if expected_sha is not None and baseline_meta['sha256'] != expected_sha:\n"
        "    raise SystemExit(f\"FATAL: baseline archive sha256 mismatch: expected={expected_sha} actual={baseline_meta['sha256']}\")\n"
        "if expected_bytes is not None and baseline_meta['bytes'] != expected_bytes:\n"
        "    raise SystemExit(f\"FATAL: baseline archive bytes mismatch: expected={expected_bytes} actual={baseline_meta['bytes']}\")\n"
        "plan_meta = file_meta(plan_path, label='perturbation plan')\n"
        "plan = load_json(plan_path, label='perturbation plan')\n"
        "if isinstance(plan, list):\n"
        "    points = plan\n"
        "    top_baseline_json = None\n"
        "elif isinstance(plan, dict):\n"
        "    points = plan.get('points')\n"
        "    top_baseline_json = plan.get('baseline_contest_auth_eval_json')\n"
        "else:\n"
        "    raise SystemExit('FATAL: perturbation plan must be a JSON object or list')\n"
        "if not isinstance(points, list) or not points:\n"
        "    raise SystemExit('FATAL: perturbation plan points must be a non-empty list')\n"
        "root = plan_path.resolve().parent\n"
        "if baseline_json is None and top_baseline_json is not None:\n"
        "    baseline_json = resolve(top_baseline_json, root=root)\n"
        "baseline_json_meta = file_meta(baseline_json, label='baseline contest_auth_eval_json') if baseline_json else None\n"
        "point_records = []\n"
        "nonzero_count = 0\n"
        "for index, raw in enumerate(points):\n"
        "    if not isinstance(raw, dict):\n"
        "        raise SystemExit(f'FATAL: points[{index}] must be an object')\n"
        "    try:\n"
        "        epsilon = float(raw.get('epsilon'))\n"
        "    except Exception as exc:\n"
        "        raise SystemExit(f'FATAL: points[{index}].epsilon must be numeric') from exc\n"
        "    archive_value = raw.get('archive')\n"
        "    if archive_value is None and abs(epsilon) <= 1e-12:\n"
        "        archive = baseline_archive\n"
        "    elif archive_value is None:\n"
        "        raise SystemExit(f'FATAL: points[{index}].archive is required for nonzero epsilon')\n"
        "    else:\n"
        "        archive = resolve(archive_value, root=root)\n"
        "    nonzero_count += int(abs(epsilon) > 1e-12)\n"
        "    record = {\n"
        "        'index': index,\n"
        "        'epsilon': epsilon,\n"
        "        'archive': file_meta(archive, label=f'points[{index}].archive', archive=True),\n"
        "    }\n"
        "    json_value = raw.get('contest_auth_eval_json')\n"
        "    if json_value is not None:\n"
        "        record['contest_auth_eval_json'] = file_meta(resolve(json_value, root=root), label=f'points[{index}].contest_auth_eval_json')\n"
        "    point_records.append(record)\n"
        "if nonzero_count <= 0:\n"
        "    raise SystemExit('FATAL: perturbation plan must contain at least one nonzero response point')\n"
        "payload = {\n"
        "    'schema_version': 1,\n"
        "    'tool': 'lightning_official_component_response_input_preflight',\n"
        "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "    'baseline_archive': baseline_meta,\n"
        "    'baseline_contest_auth_eval_json': baseline_json_meta,\n"
        "    'perturbation_plan': plan_meta,\n"
        "    'point_count': len(point_records),\n"
        "    'nonzero_point_count': nonzero_count,\n"
        "    'points': point_records,\n"
        "}\n"
        "out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "print('LIGHTNING_COMPONENT_RESPONSE_INPUT_PREFLIGHT_OK')\n"
        "print(json.dumps({'point_count': len(point_records), 'nonzero_point_count': nonzero_count, 'baseline_sha256': baseline_meta['sha256']}, sort_keys=True))\n"
        "PY"
    )


def _component_response_cleanup_command(*, python_bin: str, output_dir: str) -> str:
    py = _quote(python_bin)
    out = _quote(output_dir)
    return (
        f"{py} - {out} <<'PY'\n"
        "import json, pathlib, shutil, sys, time\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "removed = []\n"
        "for point in sorted((root / 'evals').glob('*')):\n"
        "    if not point.is_dir():\n"
        "        continue\n"
        "    for name in ('inflated', 'extracted'):\n"
        "        target = point / name\n"
        "        if target.exists():\n"
        "            shutil.rmtree(target)\n"
        "            removed.append(str(target))\n"
        "    archive = point / 'archive.zip'\n"
        "    if archive.exists():\n"
        "        archive.unlink()\n"
        "        removed.append(str(archive))\n"
        "payload = {\n"
        "    'schema_version': 1,\n"
        "    'tool': 'lightning_official_component_response_cleanup',\n"
        "    'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n"
        "    'removed_count': len(removed),\n"
        "    'removed_paths': removed[:200],\n"
        "}\n"
        "(root / 'official_component_response_cleanup.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')\n"
        "print('LIGHTNING_COMPONENT_RESPONSE_CLEANUP_OK')\n"
        "print(json.dumps({'removed_count': len(removed)}, sort_keys=True))\n"
        "PY"
    )


def official_component_response_command(
    *,
    repo_dir: str,
    baseline_archive_path: str,
    perturbation_plan_path: str,
    upstream_dir: str,
    output_dir: str,
    python_bin: str = ".venv/bin/python",
    baseline_contest_auth_eval_json: str | None = None,
    inflate_sh: str = "submissions/robust_current/inflate.sh",
    video_names_file: str = "upstream/public_test_video_names.txt",
    job_name: str | None = None,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    queue_metadata: dict[str, Any] | None = None,
    max_relative_error: float = 0.35,
    zero_repro_tolerance: float = 1e-7,
    min_observed_delta: float = 1e-12,
    allow_directional: bool = False,
    require_passed: bool = False,
) -> str:
    """Build a fail-closed CUDA official component-response command."""

    _validate_writable_output_dir(output_dir)
    _validate_expected_archive(
        expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes,
    )
    for field, value in (
        ("max_relative_error", max_relative_error),
        ("zero_repro_tolerance", zero_repro_tolerance),
        ("min_observed_delta", min_observed_delta),
    ):
        _validate_optional_number(float(value), field=field, minimum=0.0)

    repo = _quote(repo_dir)
    out = _quote(output_dir)
    py = _quote(python_bin)
    baseline = _quote(baseline_archive_path)
    plan = _quote(perturbation_plan_path)
    upstream = _quote(upstream_dir)
    inflate = _quote(inflate_sh)
    video_names = _quote(video_names_file)
    profile_parts = [
        f"{py} -u experiments/profile_component_sensitivity_official.py",
        f"--baseline-archive {baseline}",
        f"--perturbation-plan {plan}",
        f"--output-dir {out}",
        "--contest-auth-eval-script experiments/contest_auth_eval.py",
        f"--inflate-sh {inflate}",
        f"--upstream {upstream}",
        f"--video-names-file {video_names}",
        "--device cuda",
        f"--max-relative-error {_quote(str(max_relative_error))}",
        f"--zero-repro-tolerance {_quote(str(zero_repro_tolerance))}",
        f"--min-observed-delta {_quote(str(min_observed_delta))}",
    ]
    if baseline_contest_auth_eval_json:
        profile_parts.append(
            f"--baseline-contest-auth-eval-json {_quote(baseline_contest_auth_eval_json)}"
        )
    if allow_directional:
        profile_parts.append("--allow-directional")
    if require_passed:
        profile_parts.append("--require-passed")

    validation_parts = [
        f"{py} -u scripts/launch_lightning_batch_job.py validate-component-response-artifacts",
        f"--artifact-dir {out}",
    ]
    if expected_baseline_archive_sha256 is not None:
        validation_parts.extend(
            [
                f"--expected-baseline-archive-sha256 {_quote(expected_baseline_archive_sha256)}",
                f"--expected-baseline-archive-size-bytes {_quote(str(expected_baseline_archive_size_bytes))}",
            ]
        )
    if require_passed:
        validation_parts.append("--require-passed")

    metadata_payload = {
        "schema_version": 1,
        "job_name": job_name,
        "role": "official_component_response",
        "expected_archive_sha256": expected_baseline_archive_sha256,
        "expected_archive_size_bytes": expected_baseline_archive_size_bytes,
        "expected_baseline_archive_sha256": expected_baseline_archive_sha256,
        "expected_baseline_archive_size_bytes": expected_baseline_archive_size_bytes,
        "queue_metadata": _normalise_metadata(queue_metadata),
        "adjudication": None,
        "score_source": "official_component_response_summary.json:contest_auth_eval_json_components",
        "status_source": "lightning_sdk_job_attributes",
    }

    files_to_remove = [
        ARTIFACT_METADATA,
        ARTIFACT_COMPONENT_RESPONSE_INPUTS,
        ARTIFACT_COMPONENT_RESPONSE_SUMMARY,
        ARTIFACT_COMPONENT_RESPONSE_VALIDATION,
        ARTIFACT_COMPONENT_RESPONSE_LOG,
        "official_component_response_cleanup.json",
        "official_component_response_artifact_validation.log",
        ARTIFACT_DALI_BOOTSTRAP,
        ARTIFACT_DALI_REQUIREMENTS,
        ARTIFACT_RUNNER_PREFLIGHT,
        ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
        ARTIFACT_SUPPLY_CHAIN_SCAN,
        *COMPONENT_RESPONSE_CURVE_FILES,
    ]
    command = "\n".join(
        [
            "set -euo pipefail",
            f"cd {repo}",
            "test -f env.sh && source env.sh || true",
            f"mkdir -p {out}",
            f"rm -rf {out}/evals {out}/uv_project_env",
            " ".join(["rm", "-f", *[f"{out}/{_quote(name)}" for name in files_to_remove]]),
            _ensure_remote_uv_command(output_dir=output_dir),
            _write_metadata_command(
                python_bin=python_bin,
                output_dir=output_dir,
                payload=metadata_payload,
            ),
            _runner_preflight_command(python_bin=python_bin, output_dir=output_dir),
            _component_response_input_preflight_command(
                python_bin=python_bin,
                output_dir=output_dir,
                baseline_archive_path=baseline_archive_path,
                perturbation_plan_path=perturbation_plan_path,
                baseline_contest_auth_eval_json=baseline_contest_auth_eval_json,
                expected_baseline_archive_sha256=expected_baseline_archive_sha256,
                expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
            ),
            (
                f"export UV_PROJECT_ENVIRONMENT={out}/uv_project_env\n"
                "export UV_LINK_MODE=${UV_LINK_MODE:-copy}\n"
                + " ".join(profile_parts)
                + f" 2>&1 | tee {out}/{ARTIFACT_COMPONENT_RESPONSE_LOG}"
            ),
            f"test -f {out}/{ARTIFACT_COMPONENT_RESPONSE_SUMMARY}",
            *[
                f"test -f {out}/{curve_name}"
                for curve_name in COMPONENT_RESPONSE_CURVE_FILES
            ],
            _component_response_cleanup_command(python_bin=python_bin, output_dir=output_dir),
            " ".join(validation_parts) + f" 2>&1 | tee {out}/official_component_response_artifact_validation.log",
            f"test -f {out}/{ARTIFACT_COMPONENT_RESPONSE_VALIDATION}",
        ]
    )
    return command


def make_official_component_response_spec(
    *,
    name: str,
    baseline_archive_path: str,
    perturbation_plan_path: str,
    repo_dir: str,
    upstream_dir: str,
    output_dir: str | None = None,
    machine: str = "T4",
    studio: str | None = None,
    image: str | None = None,
    python_bin: str = ".venv/bin/python",
    max_runtime: int | None = 6 * 60 * 60,
    env: dict[str, str] | None = None,
    teamspace: str | None = None,
    org: str | None = None,
    user: str | None = None,
    cloud_account: str | None = None,
    baseline_contest_auth_eval_json: str | None = None,
    inflate_sh: str = "submissions/robust_current/inflate.sh",
    video_names_file: str = "upstream/public_test_video_names.txt",
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    queue_metadata: dict[str, Any] | None = None,
    local_artifact_dir: str | None = None,
    max_relative_error: float = 0.35,
    zero_repro_tolerance: float = 1e-7,
    min_observed_delta: float = 1e-12,
    allow_directional: bool = False,
    require_passed: bool = False,
) -> LightningBatchJobSpec:
    """Create a strict official component-response Batch Job spec."""

    out = output_dir or default_exact_eval_output_dir(repo_dir=repo_dir, job_name=name)
    command = official_component_response_command(
        repo_dir=repo_dir,
        baseline_archive_path=baseline_archive_path,
        perturbation_plan_path=perturbation_plan_path,
        upstream_dir=upstream_dir,
        output_dir=out,
        python_bin=python_bin,
        baseline_contest_auth_eval_json=baseline_contest_auth_eval_json,
        inflate_sh=inflate_sh,
        video_names_file=video_names_file,
        job_name=name,
        expected_baseline_archive_sha256=expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
        queue_metadata=queue_metadata,
        max_relative_error=max_relative_error,
        zero_repro_tolerance=zero_repro_tolerance,
        min_observed_delta=min_observed_delta,
        allow_directional=allow_directional,
        require_passed=require_passed,
    )
    spec = LightningBatchJobSpec(
        name=name,
        machine=machine,
        command=command,
        studio=studio,
        image=image,
        teamspace=teamspace,
        org=org,
        user=user,
        cloud_account=cloud_account,
        env=dict(env or {}),
        interruptible=False,
        max_runtime=max_runtime,
        reuse_snapshot=False,
        role="official_component_response",
        expected_archive_sha256=expected_baseline_archive_sha256,
        expected_archive_size_bytes=expected_baseline_archive_size_bytes,
        queue_metadata=dict(queue_metadata or {}),
        local_artifact_dir=local_artifact_dir,
        remote_output_dir=out,
        adjudication=None,
    )
    spec.validate()
    return spec


def _queue_record(spec: LightningBatchJobSpec, *, queued_at_utc: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "queue_name": "official_lightning_batch_jobs",
        "queued_at_utc": queued_at_utc,
        "job_name": spec.name,
        "role": spec.role,
        "machine": spec.machine,
        "cloud_account": spec.cloud_account,
        "interruptible": spec.interruptible,
        "reuse_snapshot": spec.reuse_snapshot,
        "expected_archive_sha256": spec.expected_archive_sha256,
        "expected_archive_size_bytes": spec.expected_archive_size_bytes,
        "local_artifact_dir": spec.local_artifact_dir,
        "remote_output_dir": spec.remote_output_dir,
        "predicted_sdk_artifact_path": lightning_sdk_artifact_path(spec.name),
        "sdk_artifact_path": lightning_sdk_artifact_path(spec.name),
        "command_sha256": _command_sha256(spec.command),
        "queue_metadata": _normalise_metadata(spec.queue_metadata),
        "adjudication": spec.adjudication.asdict() if spec.adjudication is not None else None,
    }


def job_status_snapshot(job: object) -> dict[str, Any]:
    """Snapshot official Batch Job attributes without reading human logs."""

    fields = (
        "id",
        "name",
        "status",
        "link",
        "snapshot_path",
        "artifact_path",
        "machine",
        "total_cost",
        "created_at",
        "started_at",
        "completed_at",
        "failed_at",
        "failure_reason",
    )
    return {
        "schema_version": 1,
        "refreshed_at_utc": _utc_now(),
        "source": "lightning_sdk_job_attributes",
        **{field: _safe_getattr(job, field) for field in fields},
    }


def _extract_job_status(snapshot: dict[str, Any]) -> str | None:
    value = snapshot.get("status")
    if value is None or (isinstance(value, str) and value.startswith("<error:")):
        return None
    return str(value)


_LIGHTNING_STATUS_RANKS = {
    "submitted": 10,
    "queued": 20,
    "pending": 20,
    "provisioning": 30,
    "starting": 35,
    "running": 40,
    "completed": 100,
    "complete": 100,
    "succeeded": 100,
    "success": 100,
    "failed": 100,
    "error": 100,
    "cancelled": 100,
    "canceled": 100,
    "stopped": 100,
    "terminated": 100,
    "timeout": 100,
    "timed_out": 100,
}

_LIGHTNING_TERMINAL_STATUSES = {
    "completed",
    "complete",
    "succeeded",
    "success",
    "failed",
    "error",
    "cancelled",
    "canceled",
    "stopped",
    "terminated",
    "timeout",
    "timed_out",
}

_REMOTE_STATUS_RECONCILIATION_REQUIRED = "REMOTE_STATUS_RECONCILIATION_REQUIRED"
_LIGHTNING_FAIL_CLOSED_ROLES = {
    "alpha_geo0_exact_eval",
    "exact_cuda_eval",
    "exact_cpu_eval",
    "official_component_response",
    "diagnostic_component_sensitivity",
}


def _normalise_lightning_status(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower().replace("-", "_").replace(" ", "_")


def _lightning_status_transition_anomaly(
    previous_status: object,
    current_status: object,
) -> dict[str, Any] | None:
    previous_key = _normalise_lightning_status(previous_status)
    current_key = _normalise_lightning_status(current_status)
    if previous_key is None or current_key is None or previous_key == current_key:
        return None
    previous_rank = _LIGHTNING_STATUS_RANKS.get(previous_key)
    current_rank = _LIGHTNING_STATUS_RANKS.get(current_key)
    if previous_rank is None or current_rank is None:
        return None
    if previous_key in _LIGHTNING_TERMINAL_STATUSES and current_key not in _LIGHTNING_TERMINAL_STATUSES:
        anomaly_type = "terminal_status_reopened"
    elif current_rank < previous_rank:
        anomaly_type = "nonterminal_status_regression"
    else:
        return None
    return {
        "type": anomaly_type,
        "previous_status": str(previous_status),
        "current_status": str(current_status),
        "previous_rank": previous_rank,
        "current_rank": current_rank,
    }


def _lightning_status_anomaly_key(anomaly: dict[str, Any]) -> tuple[object, ...]:
    return (
        anomaly.get("type"),
        anomaly.get("previous_status"),
        anomaly.get("current_status"),
        anomaly.get("recorded_at_utc"),
        anomaly.get("status_history_index"),
    )


def _attach_lightning_status_history_anomalies(record: dict[str, Any]) -> None:
    history = [
        dict(entry)
        for entry in record.get("status_history") or []
        if isinstance(entry, dict)
    ]
    anomalies = [
        dict(entry)
        for entry in record.get("status_anomalies") or []
        if isinstance(entry, dict)
    ]
    seen = {_lightning_status_anomaly_key(entry) for entry in anomalies}
    for index in range(1, len(history)):
        previous = history[index - 1].get("observed_status", history[index - 1].get("status"))
        current = history[index].get("observed_status", history[index].get("status"))
        anomaly = _lightning_status_transition_anomaly(previous, current)
        if anomaly is None:
            continue
        anomaly = {
            **anomaly,
            "recorded_at_utc": history[index].get("recorded_at_utc"),
            "source": history[index].get("source"),
            "status_history_index": index,
        }
        key = _lightning_status_anomaly_key(anomaly)
        if key in seen:
            continue
        seen.add(key)
        anomalies.append(anomaly)
        history[index]["anomaly"] = anomaly
    if anomalies:
        record["status_anomalies"] = anomalies
        record["status_reconciliation_required"] = True
    record["status_history"] = history


def _lightning_refresh_requires_fail_closed_status(
    record: dict[str, Any],
    anomaly: dict[str, Any] | None,
    observed_status: object,
) -> bool:
    if anomaly is None or bool(record.get("dry_run")):
        return False
    observed_key = _normalise_lightning_status(observed_status)
    if observed_key in _LIGHTNING_TERMINAL_STATUSES:
        return False
    spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
    role = spec.get("role")
    return role in _LIGHTNING_FAIL_CLOSED_ROLES


def validate_local_artifact_dir(
    artifact_dir: str | Path,
    *,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    require_adjudication: bool = False,
) -> dict[str, Any]:
    """Validate a locally harvested Lightning artifact directory via JSON/files only."""

    artifact_root = Path(artifact_dir)
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"artifact dir not found: {artifact_root}")

    metadata_path = artifact_root / ARTIFACT_METADATA
    contest_json_path = artifact_root / "contest_auth_eval.json"
    archive_path = artifact_root / ARTIFACT_ARCHIVE
    dali_bootstrap_path = artifact_root / ARTIFACT_DALI_BOOTSTRAP
    dali_requirements_path = artifact_root / ARTIFACT_DALI_REQUIREMENTS
    runner_preflight_path = artifact_root / ARTIFACT_RUNNER_PREFLIGHT
    supply_chain_pre_path = artifact_root / ARTIFACT_SUPPLY_CHAIN_SCAN_PRE
    supply_chain_post_path = artifact_root / ARTIFACT_SUPPLY_CHAIN_SCAN
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Lightning artifact dir missing required files: {[str(metadata_path)]}")
    metadata = _load_json_object(metadata_path, label="Lightning queue metadata")
    role = metadata.get("role")
    if role not in EXACT_AUTH_EVAL_ROLES:
        raise ValueError(
            f"Lightning queue metadata role={role!r}, "
            "expected 'exact_cuda_eval' or 'exact_cpu_eval'"
        )
    eval_device = "cuda" if role == "exact_cuda_eval" else "cpu"
    required = [
        metadata_path,
        contest_json_path,
        archive_path,
        artifact_root / "eval_provenance.json",
        artifact_root / "report.txt",
        runner_preflight_path,
        supply_chain_pre_path,
        supply_chain_post_path,
    ]
    if eval_device == "cuda":
        required.extend([dali_bootstrap_path, dali_requirements_path])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Lightning artifact dir missing required files: {missing}")

    if metadata.get("score_source") != "contest_auth_eval.json:score_recomputed_from_components":
        raise ValueError("Lightning queue metadata score_source is not the exact contest JSON component recompute")
    metadata_expected_sha = metadata.get("expected_archive_sha256")
    metadata_expected_bytes = metadata.get("expected_archive_size_bytes")
    if metadata_expected_sha is None or metadata_expected_bytes is None:
        raise ValueError("Lightning queue metadata missing expected archive sha256/bytes")
    _validate_expected_archive(metadata_expected_sha, metadata_expected_bytes)
    if expected_archive_sha256 is not None and expected_archive_sha256 != metadata_expected_sha:
        raise ValueError(
            "explicit expected archive sha256 does not match Lightning metadata: "
            f"explicit={expected_archive_sha256!r} metadata={metadata_expected_sha!r}"
        )
    if expected_archive_size_bytes is not None and expected_archive_size_bytes != metadata_expected_bytes:
        raise ValueError(
            "explicit expected archive bytes does not match Lightning metadata: "
            f"explicit={expected_archive_size_bytes!r} metadata={metadata_expected_bytes!r}"
        )
    expected_sha = expected_archive_sha256 if expected_archive_sha256 is not None else metadata_expected_sha
    expected_bytes = (
        expected_archive_size_bytes
        if expected_archive_size_bytes is not None
        else metadata_expected_bytes
    )
    _require_expected_archive(expected_sha, expected_bytes)

    supply_chain_pre = _validate_supply_chain_scan_artifact(
        supply_chain_pre_path,
        label="pre-DALI supply-chain scan",
    )
    supply_chain_post = _validate_supply_chain_scan_artifact(
        supply_chain_post_path,
        label="post-DALI supply-chain scan",
    )
    dali_bootstrap = (
        _validate_dali_bootstrap_artifact(dali_bootstrap_path, dali_requirements_path)
        if eval_device == "cuda"
        else None
    )
    runner_preflight = _validate_runner_preflight_artifact(
        runner_preflight_path,
        required_device=eval_device,
    )

    payload = _load_json_object(contest_json_path, label="contest_auth_eval.json")
    score = _require_finite_number(payload, "score_recomputed_from_components")
    _require_finite_number(payload, "final_score")
    _require_finite_number(payload, "avg_posenet_dist")
    _require_finite_number(payload, "avg_segnet_dist")
    n_samples = payload.get("n_samples")
    if n_samples != 600:
        raise ValueError(f"contest_auth_eval.json n_samples={n_samples!r}, expected 600")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("contest_auth_eval.json missing provenance object")
    device = provenance.get("device")
    if device != eval_device:
        raise ValueError(f"contest_auth_eval provenance.device={device!r}, expected {eval_device!r}")
    expected_score_axis = _exact_eval_axis(eval_device)
    if payload.get("score_axis") != expected_score_axis:
        raise ValueError(
            f"contest_auth_eval score_axis={payload.get('score_axis')!r}, "
            f"expected {expected_score_axis!r}"
        )

    actual_identity = archive_identity(archive_path)
    payload_bytes = payload.get("archive_size_bytes")
    if payload_bytes != actual_identity["archive_size_bytes"]:
        raise ValueError(
            "contest_auth_eval archive_size_bytes does not match artifact archive: "
            f"json={payload_bytes!r} actual={actual_identity['archive_size_bytes']}"
        )
    payload_sha = provenance.get("archive_sha256") or payload.get("archive_sha256")
    if payload_sha != actual_identity["archive_sha256"]:
        raise ValueError(
            "contest_auth_eval archive_sha256 does not match artifact archive: "
            f"json={payload_sha!r} actual={actual_identity['archive_sha256']}"
        )
    if expected_sha is not None and payload_sha != expected_sha:
        raise ValueError(
            f"artifact archive sha256 {payload_sha!r} does not match expected {expected_sha!r}"
        )
    if expected_bytes is not None and payload_bytes != expected_bytes:
        raise ValueError(
            f"artifact archive bytes {payload_bytes!r} does not match expected {expected_bytes!r}"
        )

    adjudication_provenance: dict[str, Any] | None = None
    adjudication_lane_status: str | None = None
    adjudication_component_gate_triggered = False
    adjudication_regression_triggered = False
    adjudication_sane_score_gate_triggered = False
    adjudication_promotion_eligible: bool | None = None
    adjudication_meta = metadata.get("adjudication")
    if not isinstance(adjudication_meta, dict):
        raise ValueError(
            "Lightning exact-eval artifact metadata must include adjudication; "
            "promotion-grade artifacts require adjudication provenance"
        )
    if adjudication_meta or require_adjudication:
        provenance_name = "adjudication_provenance.json"
        result_copy_name = "contest_auth_eval.adjudicated.json"
        if isinstance(adjudication_meta, dict):
            provenance_name = str(adjudication_meta.get("provenance_name") or provenance_name)
            result_copy_name = str(adjudication_meta.get("result_copy_name") or result_copy_name)
        adjudication_path = artifact_root / provenance_name
        result_copy_path = artifact_root / result_copy_name
        if not adjudication_path.is_file():
            raise FileNotFoundError(f"adjudication provenance not found: {adjudication_path}")
        if not result_copy_path.is_file():
            raise FileNotFoundError(f"adjudicated contest JSON copy not found: {result_copy_path}")
        adjudication_provenance = _load_json_object(
            adjudication_path,
            label="adjudication provenance",
        )
        adjudicated_payload = _load_json_object(
            result_copy_path,
            label="adjudicated contest JSON copy",
        )
        if adjudicated_payload != payload:
            raise ValueError("adjudicated contest JSON copy does not match contest_auth_eval.json")
        adj_sha = adjudication_provenance.get(f"{expected_score_axis}_archive_sha256")
        adj_bytes = adjudication_provenance.get(f"{expected_score_axis}_archive_bytes")
        if adj_sha != actual_identity["archive_sha256"]:
            raise ValueError("adjudication archive sha256 does not match artifact archive")
        if adj_bytes != actual_identity["archive_size_bytes"]:
            raise ValueError("adjudication archive bytes do not match artifact archive")
        if adjudication_provenance.get(f"{expected_score_axis}_score_source") != (
            "contest_auth_eval.json:score_recomputed_from_components"
        ):
            raise ValueError("adjudication provenance score source is not contest_auth_eval.json")
        if adjudication_provenance.get(f"{expected_score_axis}_device") != eval_device:
            raise ValueError(f"adjudication provenance device is not {eval_device}")
        if eval_device == "cpu" and adjudication_provenance.get("promotion_eligible") is True:
            raise ValueError("exact CPU adjudication must not be promotion_eligible")
        if eval_device == "cpu" and adjudication_provenance.get("score_tag") != "[contest-CPU]":
            raise ValueError("exact CPU adjudication must carry [contest-CPU] score_tag")
        adjudication_lane_status = adjudication_provenance.get("lane_status")
        adjudication_component_gate_triggered = bool(
            adjudication_provenance.get("component_gate_triggered")
        )
        adjudication_regression_triggered = bool(
            adjudication_provenance.get("regression_triggered")
        )
        adjudication_sane_score_gate_triggered = bool(
            adjudication_provenance.get("sane_score_gate_triggered")
        )
        if "promotion_eligible" in adjudication_provenance:
            adjudication_promotion_eligible = (
                adjudication_provenance.get("promotion_eligible") is True
            )

    component_trace_validation: dict[str, Any] | None = None
    component_trace_path = artifact_root / ARTIFACT_COMPONENT_TRACE
    component_trace_status_path = artifact_root / ARTIFACT_COMPONENT_TRACE_STATUS
    component_trace_status: dict[str, Any] | None = None
    if component_trace_status_path.is_file():
        component_trace_status = _load_json_object(
            component_trace_status_path,
            label="component_trace_status.json",
        )
        if component_trace_status.get("score_claim") is not False:
            raise ValueError("component_trace_status.json must remain score_claim=false")
        if component_trace_status.get("evidence_grade") != "diagnostic_component_trace_status":
            raise ValueError("component_trace_status.json has unexpected evidence_grade")
    if component_trace_path.is_file():
        component_trace_payload = _load_json_object(
            component_trace_path,
            label="component_trace.json",
        )
        if component_trace_payload.get("score_claim") is not False:
            raise ValueError("component_trace.json must remain score_claim=false")
        if component_trace_payload.get("evidence_grade") != "diagnostic_component_trace":
            raise ValueError("component_trace.json has unexpected evidence_grade")
        if component_trace_payload.get("n_samples") != 600:
            raise ValueError("component_trace.json n_samples is not 600")
        cross_check = component_trace_payload.get("contest_auth_eval_cross_check")
        if not isinstance(cross_check, dict) or cross_check.get("all_match") is not True:
            raise ValueError("component_trace.json cross-check did not match contest_auth_eval.json")
        component_trace_validation = {
            "path": str(component_trace_path),
            "sha256": _sha256(component_trace_path),
            "n_samples": component_trace_payload.get("n_samples"),
            "avg_posenet_dist": component_trace_payload.get("avg_posenet_dist"),
            "avg_segnet_dist": component_trace_payload.get("avg_segnet_dist"),
            "score_recomputed_from_components": component_trace_payload.get(
                "score_recomputed_from_components"
            ),
            "contest_auth_eval_cross_check": cross_check,
        }

    promotion_eligible = (
        adjudication_promotion_eligible
        if adjudication_promotion_eligible is not None
        else (
            adjudication_provenance is not None
            and provenance.get("gpu_t4_match") is True
        )
    )
    promotion_eligible = (
        promotion_eligible
        and not adjudication_component_gate_triggered
        and not adjudication_regression_triggered
        and not adjudication_sane_score_gate_triggered
    )

    return {
        "schema_version": 1,
        "validated_at_utc": _utc_now(),
        "artifact_dir": str(artifact_root),
        "archive_sha256": actual_identity["archive_sha256"],
        "archive_size_bytes": actual_identity["archive_size_bytes"],
        "score_recomputed_from_components": score,
        "score_axis": expected_score_axis,
        "n_samples": n_samples,
        "device": device,
        "gpu_model": provenance.get("gpu_model"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
        "metadata": metadata,
        "dali_bootstrap": dali_bootstrap,
        "runner_preflight": runner_preflight,
        "supply_chain_pre": supply_chain_pre,
        "supply_chain_post": supply_chain_post,
        "adjudication_provenance": adjudication_provenance,
        "adjudication_lane_status": adjudication_lane_status,
        "adjudication_component_gate_triggered": adjudication_component_gate_triggered,
        "adjudication_regression_triggered": adjudication_regression_triggered,
        "adjudication_sane_score_gate_triggered": adjudication_sane_score_gate_triggered,
        "component_trace": component_trace_validation,
        "component_trace_status": component_trace_status,
        "promotion_eligible": promotion_eligible,
        "score_source": "contest_auth_eval.json:score_recomputed_from_components",
    }


def _optional_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    if out != out or abs(out) == float("inf"):
        return None
    return out


def _reconstruct_missing_adjudication_artifacts_from_metadata(
    artifact_root: Path,
) -> list[str]:
    """Recover adjudication copies after a remote artifact-copy race.

    Some Studio-backed jobs finish ``scripts/adjudicate_contest_auth_eval.py``
    and flush ``adjudication.log`` before the JSON copies are visible in the
    persisted artifact mirror. If the exact contest JSON, queue metadata,
    provenance, and archive are present, we can deterministically regenerate
    the two adjudication files from machine-readable inputs instead of
    downgrading a valid exact eval to an infra failure.
    """

    metadata_path = artifact_root / ARTIFACT_METADATA
    contest_json_path = artifact_root / "contest_auth_eval.json"
    archive_path = artifact_root / ARTIFACT_ARCHIVE
    if not (metadata_path.is_file() and contest_json_path.is_file() and archive_path.is_file()):
        return []

    metadata = _load_json_object(metadata_path, label="Lightning queue metadata")
    adjudication_meta = metadata.get("adjudication")
    if not isinstance(adjudication_meta, dict):
        return []
    provenance_name = str(adjudication_meta.get("provenance_name") or "adjudication_provenance.json")
    result_copy_name = str(adjudication_meta.get("result_copy_name") or "contest_auth_eval.adjudicated.json")
    provenance_path = artifact_root / provenance_name
    result_copy_path = artifact_root / result_copy_name
    missing_names = [
        name
        for name, path in (
            (provenance_name, provenance_path),
            (result_copy_name, result_copy_path),
        )
        if not path.is_file()
    ]
    if not missing_names:
        return []

    payload = _load_json_object(contest_json_path, label="contest_auth_eval.json")
    score = _require_finite_number(payload, "score_recomputed_from_components")
    final_score = _require_finite_number(payload, "final_score")
    avg_posenet = _require_finite_number(payload, "avg_posenet_dist")
    avg_segnet = _require_finite_number(payload, "avg_segnet_dist")
    n_samples = payload.get("n_samples")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return []
    actual_identity = archive_identity(archive_path)
    if payload.get("archive_size_bytes") != actual_identity["archive_size_bytes"]:
        return []
    payload_sha = provenance.get("archive_sha256") or payload.get("archive_sha256")
    if payload_sha != actual_identity["archive_sha256"]:
        return []
    required_device = str(adjudication_meta.get("required_device") or "cuda")
    required_device = _normalise_eval_device(required_device)
    required_samples = int(adjudication_meta.get("required_samples") or 600)
    device = provenance.get("device")
    if device != required_device or n_samples != required_samples:
        return []
    score_axis = _exact_eval_axis(required_device)
    if payload.get("score_axis") != score_axis:
        return []

    def component_gate(
        *,
        component: str,
        metric: str,
        observed: float,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        max_abs = _optional_finite_float(adjudication_meta.get(f"max_{component}_dist"))
        reference = _optional_finite_float(adjudication_meta.get(f"baseline_{component}_dist"))
        max_relative = _optional_finite_float(adjudication_meta.get(f"max_{component}_relative"))
        if max_abs is None and max_relative is None:
            return None, []
        reference_label = str(adjudication_meta.get("component_reference_label") or "baseline")
        gate: dict[str, Any] = {
            "component": component,
            "metric": metric,
            "observed": observed,
            "max_absolute": max_abs,
            "reference": reference,
            "reference_label": reference_label,
            "max_relative": max_relative,
            "relative_to_reference": None,
            "passed": True,
        }
        violations: list[dict[str, Any]] = []
        if max_abs is not None and observed > max_abs:
            gate["passed"] = False
            violations.append(
                {
                    "component": component,
                    "metric": metric,
                    "observed": observed,
                    "max_absolute": max_abs,
                    "reason": "absolute_component_gate",
                }
            )
        if max_relative is not None and reference is not None and reference > 0.0:
            relative = observed / reference
            gate["relative_to_reference"] = relative
            if relative > max_relative:
                gate["passed"] = False
                violations.append(
                    {
                        "component": component,
                        "metric": metric,
                        "observed": observed,
                        "reference": reference,
                        "reference_label": reference_label,
                        "relative_to_reference": relative,
                        "max_relative": max_relative,
                        "reason": "relative_component_gate",
                    }
                )
        return gate, violations

    component_gates: list[dict[str, Any]] = []
    component_violations: list[dict[str, Any]] = []
    for gate, violations in (
        component_gate(component="posenet", metric="avg_posenet_dist", observed=avg_posenet),
        component_gate(component="segnet", metric="avg_segnet_dist", observed=avg_segnet),
    ):
        if gate is not None:
            component_gates.append(gate)
        component_violations.extend(violations)

    baseline_score = _optional_finite_float(adjudication_meta.get("baseline_score"))
    predicted_low = _optional_finite_float(adjudication_meta.get("predicted_band_low"))
    predicted_high = _optional_finite_float(adjudication_meta.get("predicted_band_high"))
    regression_threshold = _optional_finite_float(adjudication_meta.get("regression_threshold"))
    score_delta_vs_baseline = None if baseline_score is None else score - baseline_score
    regression_triggered = (
        False
        if score_delta_vs_baseline is None or regression_threshold is None
        else score_delta_vs_baseline > regression_threshold
    )
    max_sane_score = _optional_finite_float(adjudication_meta.get("max_sane_score")) or 10.0
    sane_score_gate_triggered = not (0.0 < score < max_sane_score)
    component_gate_triggered = bool(component_violations)
    if regression_triggered:
        lane_status = "REGRESSION_REVIEW_REQUIRED"
    elif predicted_low is not None and predicted_high is not None and predicted_low <= score <= predicted_high:
        lane_status = "IN_PREDICTED_BAND"
    elif predicted_low is not None and predicted_high is not None:
        lane_status = "OUT_OF_PREDICTED_BAND"
    else:
        lane_status = "ADJUDICATION_ARTIFACT_RECOVERED_REVIEW_REQUIRED"
    if component_gate_triggered:
        lane_status = "COMPONENT_GATE_REVIEW_REQUIRED"
    if sane_score_gate_triggered:
        lane_status = "SANE_SCORE_REVIEW_REQUIRED"

    scientific_score_eligible = (
        not regression_triggered
        and not component_gate_triggered
        and not sane_score_gate_triggered
    )
    if required_device == "cuda":
        contest_equivalent_hardware = provenance.get("gpu_t4_match") is True
        promotion_eligible = scientific_score_eligible and contest_equivalent_hardware
        score_claim_valid = promotion_eligible
        evidence_grade = "A++ contest T4" if contest_equivalent_hardware else "A score-grade"
        paper_claim_grade = (
            evidence_grade
            if promotion_eligible
            else (
                "A score-grade; T4/equivalent promotion required"
                if scientific_score_eligible
                else "A-negative scoped forensic"
            )
        )
        allowed_use = (
            ["promotion_review", "rank_frontier_candidate"]
            if promotion_eligible
            else ["forensic", "no_rank_frontier", "no_promotion"]
        )
    else:
        contest_equivalent_hardware = (
            payload.get("evidence_grade") == "contest-CPU"
            and payload.get("score_claim_valid") is True
            and payload.get("cpu_leaderboard_reproduction_eligible") is True
        )
        promotion_eligible = False
        score_claim_valid = scientific_score_eligible and contest_equivalent_hardware
        evidence_grade = "contest-CPU" if contest_equivalent_hardware else "CPU advisory"
        paper_claim_grade = (
            "contest-CPU public leaderboard reproduction"
            if score_claim_valid
            else (
                "contest-CPU replay requires Linux x86_64 full-sample custody"
                if scientific_score_eligible
                else "A-negative scoped CPU forensic"
            )
        )
        allowed_use = (
            [
                "cpu_axis_score_claim",
                "public_leaderboard_reproduction",
                "cpu_cuda_drift_diagnosis",
                "no_cuda_promotion",
            ]
            if score_claim_valid
            else ["forensic", "no_rank_frontier", "no_cuda_promotion"]
        )
    axis_fields: dict[str, Any] = {
        f"{score_axis}_score": score,
        f"{score_axis}_score_recomputed": score,
        f"{score_axis}_score_reported_rounded": final_score,
        f"{score_axis}_score_source": "contest_auth_eval.json:score_recomputed_from_components",
        f"{score_axis}_avg_posenet_dist": avg_posenet,
        f"{score_axis}_avg_segnet_dist": avg_segnet,
        f"{score_axis}_result_json": str(result_copy_path),
        f"{score_axis}_n_samples": n_samples,
        f"{score_axis}_archive_sha256": actual_identity["archive_sha256"],
        f"{score_axis}_archive_bytes": actual_identity["archive_size_bytes"],
        f"{score_axis}_device": device,
    }
    if required_device == "cuda":
        axis_fields.update(
            {
                "contest_cuda_gpu_model": provenance.get("gpu_model"),
                "contest_cuda_gpu_t4_match": provenance.get("gpu_t4_match"),
            }
        )
    else:
        axis_fields.update(
            {
                "contest_cpu_platform_system": provenance.get("platform_system"),
                "contest_cpu_platform_machine": provenance.get("platform_machine"),
                "contest_cpu_leaderboard_reproduction_eligible": payload.get(
                    "cpu_leaderboard_reproduction_eligible"
                ),
            }
        )
    recovered_provenance = {
        "completed_at_utc": _utc_now(),
        "artifact_recovery": "reconstructed_missing_adjudication_artifacts_from_metadata",
        "stacked_archive_bytes": actual_identity["archive_size_bytes"],
        "final_archive_bytes": actual_identity["archive_size_bytes"],
        "baseline_archive_bytes": adjudication_meta.get("baseline_archive_size_bytes"),
        "archive_delta_bytes": (
            actual_identity["archive_size_bytes"] - int(adjudication_meta["baseline_archive_size_bytes"])
            if isinstance(adjudication_meta.get("baseline_archive_size_bytes"), int)
            else None
        ),
        **axis_fields,
        "score_axis": score_axis,
        "contest_equivalent_hardware": contest_equivalent_hardware,
        "evidence_grade": evidence_grade,
        "promotion_eligible": promotion_eligible,
        "score_claim_valid": score_claim_valid,
        "scientific_score_eligible": scientific_score_eligible,
        "hardware_promotion_gate_triggered": (
            scientific_score_eligible
            and required_device == "cuda"
            and not contest_equivalent_hardware
        ),
        "paper_claim_grade": paper_claim_grade,
        "allowed_use": allowed_use,
        "score_tag": _exact_eval_score_tag(required_device),
        "result_tag": _exact_eval_score_tag(required_device),
        "score_delta_vs_baseline": score_delta_vs_baseline,
        str(adjudication_meta.get("delta_key") or "score_delta_vs_baseline"): score_delta_vs_baseline,
        "regression_threshold": regression_threshold,
        "regression_threshold_mode": "delta_vs_baseline",
        "regression_triggered": regression_triggered,
        "regression_scope": "measured_implementation_config_only_pending_review",
        "sane_score_gate_triggered": sane_score_gate_triggered,
        "sane_score_gate_violation": None,
        "max_sane_score": max_sane_score,
        "component_gates": component_gates,
        "component_gate_violations": component_violations,
        "component_gate_triggered": component_gate_triggered,
        "distillation_policy_active": False,
        "distillation_policy_gate_violations": [],
        "distillation_policy_gate_triggered": False,
        "distillation_policy_sha256": None,
        "distillation_policy_sha256_expected": None,
        "lane_status": lane_status,
    }
    _write_json_replace(result_copy_path, payload)
    _write_json_replace(provenance_path, recovered_provenance)
    return missing_names


def mirror_local_artifact_dir(
    source_dir: str | Path,
    mirror_dir: str | Path,
    *,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    require_adjudication: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Mirror a local artifact directory and validate the mirror."""

    source = Path(source_dir)
    mirror = Path(mirror_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source artifact dir not found: {source}")
    if mirror.exists() and any(mirror.iterdir()) and not overwrite:
        raise FileExistsError(f"mirror dir is not empty: {mirror}")
    mirror.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        dest = mirror / child.name
        if child.is_dir():
            shutil.copytree(child, dest, dirs_exist_ok=overwrite)
        else:
            shutil.copy2(child, dest)
    validation = validate_local_artifact_dir(
        mirror,
        expected_archive_sha256=expected_archive_sha256,
        expected_archive_size_bytes=expected_archive_size_bytes,
        require_adjudication=require_adjudication,
    )
    _write_json_replace(mirror / ARTIFACT_VALIDATION, validation)
    return validation


def _validate_component_sensitivity_sample_plan_full(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, label="component-sensitivity sample plan")
    calibration = payload.get("calibration_pairs")
    holdout = payload.get("holdout_pairs")
    if not isinstance(calibration, list) or not isinstance(holdout, list):
        raise ValueError("component-sensitivity sample_plan missing calibration/holdout pairs")
    seen: set[int] = set()
    for group_name, rows in (("calibration_pairs", calibration), ("holdout_pairs", holdout)):
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"sample_plan.{group_name}[{index}] must be an object")
            pair_index = row.get("pair_index")
            if isinstance(pair_index, bool) or not isinstance(pair_index, int):
                raise ValueError(f"sample_plan.{group_name}[{index}].pair_index must be an integer")
            if row.get("video") != 0 or row.get("t") != 2 * pair_index or row.get("t1") != 2 * pair_index + 1:
                raise ValueError(f"sample_plan.{group_name}[{index}] does not use absolute contest pair ids")
            if pair_index in seen:
                raise ValueError(f"sample_plan duplicate pair_index={pair_index}")
            seen.add(pair_index)
    missing = set(range(600)) - seen
    extra = seen - set(range(600))
    return {
        "schema_version": 1,
        "path": str(path),
        "calibration_count": len(calibration),
        "holdout_count": len(holdout),
        "pair_count": len(seen),
        "full_600_pair_coverage": not missing and not extra,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "split_hash": payload.get("split_hash"),
    }


def validate_local_component_response_artifact_dir(
    artifact_dir: str | Path,
    *,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    """Validate official component-response Lightning artifacts via JSON/files."""

    artifact_root = Path(artifact_dir)
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"component-response artifact dir not found: {artifact_root}")
    _validate_expected_archive(
        expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes,
    )

    required = [
        artifact_root / name
        for name in COMPONENT_RESPONSE_CANONICAL_ARTIFACT_FILES
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"component-response artifact dir missing required files: {missing}")

    metadata = _load_json_object(artifact_root / ARTIFACT_METADATA, label="Lightning queue metadata")
    if metadata.get("role") != "official_component_response":
        raise ValueError(
            f"Lightning queue metadata role={metadata.get('role')!r}, "
            "expected 'official_component_response'"
        )
    if metadata.get("score_source") != "official_component_response_summary.json:contest_auth_eval_json_components":
        raise ValueError("component-response metadata score_source is not the official summary/components source")

    supply_chain_pre = _validate_supply_chain_scan_artifact(
        artifact_root / ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
        label="pre-DALI supply-chain scan",
    )
    supply_chain_post = _validate_supply_chain_scan_artifact(
        artifact_root / ARTIFACT_SUPPLY_CHAIN_SCAN,
        label="post-DALI supply-chain scan",
    )
    dali_bootstrap = _validate_dali_bootstrap_artifact(
        artifact_root / ARTIFACT_DALI_BOOTSTRAP,
        artifact_root / ARTIFACT_DALI_REQUIREMENTS,
    )
    runner_preflight = _validate_runner_preflight_artifact(artifact_root / ARTIFACT_RUNNER_PREFLIGHT)

    inputs = _load_json_object(
        artifact_root / ARTIFACT_COMPONENT_RESPONSE_INPUTS,
        label="official component-response input preflight",
    )
    if inputs.get("tool") != "lightning_official_component_response_input_preflight":
        raise ValueError("component-response input preflight has unexpected tool")
    baseline_input = inputs.get("baseline_archive")
    if not isinstance(baseline_input, dict):
        raise ValueError("component-response input preflight missing baseline_archive")
    baseline_sha = baseline_input.get("sha256")
    baseline_bytes = baseline_input.get("bytes")
    if expected_baseline_archive_sha256 is not None and baseline_sha != expected_baseline_archive_sha256:
        raise ValueError(
            "baseline archive sha256 does not match expected: "
            f"actual={baseline_sha!r} expected={expected_baseline_archive_sha256!r}"
        )
    if expected_baseline_archive_size_bytes is not None and baseline_bytes != expected_baseline_archive_size_bytes:
        raise ValueError(
            "baseline archive bytes do not match expected: "
            f"actual={baseline_bytes!r} expected={expected_baseline_archive_size_bytes!r}"
        )
    if not isinstance(inputs.get("points"), list) or int(inputs.get("nonzero_point_count") or 0) <= 0:
        raise ValueError("component-response input preflight has no nonzero response points")

    summary = _load_json_object(
        artifact_root / ARTIFACT_COMPONENT_RESPONSE_SUMMARY,
        label="official component-response summary",
    )
    if summary.get("format") != "official_component_response_summary_v1":
        raise ValueError(f"component-response summary has unexpected format={summary.get('format')!r}")
    if summary.get("device") != "cuda":
        raise ValueError(f"component-response summary device={summary.get('device')!r}, expected 'cuda'")
    summary_baseline = summary.get("baseline_archive")
    if not isinstance(summary_baseline, dict):
        raise ValueError("component-response summary missing baseline_archive")
    if summary_baseline.get("sha256") != baseline_sha:
        raise ValueError("component-response summary baseline sha256 does not match input preflight")
    if summary_baseline.get("bytes") != baseline_bytes:
        raise ValueError("component-response summary baseline bytes do not match input preflight")
    external_baseline_required = summary.get("external_baseline_contest_auth_eval_json") is not None

    response_curve_paths = summary.get("response_curve_paths")
    if not isinstance(response_curve_paths, dict):
        raise ValueError("component-response summary missing response_curve_paths")
    curves: dict[str, Any] = {}
    failed_components: list[str] = []
    for component in COMPONENT_RESPONSE_COMPONENTS:
        curve_path = artifact_root / f"{component}_official_response_curve.json"
        curve = _load_json_object(curve_path, label=f"{component} official response curve")
        curves[component] = curve
        if curve.get("format") != "official_component_response_curves_v1":
            raise ValueError(f"{component} response curve has unexpected format={curve.get('format')!r}")
        if curve.get("component") != component:
            raise ValueError(f"{component} response curve component={curve.get('component')!r}")
        if curve.get("device") != "cuda":
            raise ValueError(f"{component} response curve device={curve.get('device')!r}, expected 'cuda'")
        if curve.get("official_component_response") is not True:
            raise ValueError(f"{component} response curve is not marked official_component_response")
        if curve.get("canonical_scorer_path") is not True:
            raise ValueError(f"{component} response curve is not marked canonical_scorer_path")
        if curve.get("component_response_path") != "archive_zip_inflate_sh_upstream_evaluate_py":
            raise ValueError(f"{component} response curve has non-canonical component_response_path")
        points = curve.get("points")
        if not isinstance(points, list) or len(points) <= 1:
            raise ValueError(f"{component} response curve must include baseline plus response points")
        if Path(str(response_curve_paths.get(component, ""))).name != curve_path.name:
            raise ValueError(f"summary response_curve_paths[{component!r}] does not point at {curve_path.name}")
        gate_results = curve.get("gate_results")
        if external_baseline_required and (
            not isinstance(gate_results, dict)
            or gate_results.get("external_baseline_repro") is not True
        ):
            failed_components.append(component)
        if curve.get("passed") is not True:
            failed_components.append(component)
    failed_components = sorted(set(failed_components))

    promotion_eligible = bool(summary.get("promotion_eligible")) and not failed_components
    if require_passed and not promotion_eligible:
        raise ValueError(f"official component-response gates did not pass: failed_components={failed_components}")

    validation = {
        "schema_version": 1,
        "validated_at_utc": _utc_now(),
        "artifact_dir": str(artifact_root),
        "role": "official_component_response",
        "baseline_archive_sha256": baseline_sha,
        "baseline_archive_size_bytes": baseline_bytes,
        "point_count": inputs.get("point_count"),
        "nonzero_point_count": inputs.get("nonzero_point_count"),
        "device": "cuda",
        "gpu_model": runner_preflight.get("device_name"),
        "gpu_t4_match": runner_preflight.get("gpu_t4_match"),
        "metadata": metadata,
        "input_preflight": inputs,
        "summary": summary,
        "curves": curves,
        "external_baseline_repro_required": external_baseline_required,
        "dali_bootstrap": dali_bootstrap,
        "runner_preflight": runner_preflight,
        "supply_chain_pre": supply_chain_pre,
        "supply_chain_post": supply_chain_post,
        "failed_components": failed_components,
        "promotion_eligible": promotion_eligible,
        "score_source": "official_component_response_summary.json:contest_auth_eval_json_components",
    }
    return validation


def validate_local_component_sensitivity_artifact_dir(
    artifact_dir: str | Path,
    *,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate diagnostic component-sensitivity Lightning artifacts.

    These artifacts are useful for design and exact-response planning, but they
    are deliberately non-promotable: the profiler emits Fisher/direct-renderer
    diagnostics, not canonical archive.zip -> inflate.sh -> upstream/evaluate.py
    component-response measurements.
    """

    artifact_root = Path(artifact_dir)
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"component-sensitivity artifact dir not found: {artifact_root}")
    _validate_expected_archive(
        expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes,
    )

    required = [
        artifact_root / name
        for name in COMPONENT_SENSITIVITY_CANONICAL_ARTIFACT_FILES
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"component-sensitivity artifact dir missing required files: {missing}")

    metadata = _load_json_object(artifact_root / ARTIFACT_METADATA, label="Lightning queue metadata")
    if metadata.get("role") != "diagnostic_component_sensitivity":
        raise ValueError(
            f"Lightning queue metadata role={metadata.get('role')!r}, "
            "expected 'diagnostic_component_sensitivity'"
        )
    if metadata.get("score_source") != "none:diagnostic_component_sensitivity_non_promotable":
        raise ValueError("component-sensitivity metadata score_source is not the diagnostic non-promotable source")
    if metadata.get("score_claim") is not False or metadata.get("promotion_eligible") is not False:
        raise ValueError("component-sensitivity metadata must record score_claim=false and promotion_eligible=false")

    supply_chain_pre = _validate_supply_chain_scan_artifact(
        artifact_root / ARTIFACT_SUPPLY_CHAIN_SCAN_PRE,
        label="pre-DALI supply-chain scan",
    )
    supply_chain_post = _validate_supply_chain_scan_artifact(
        artifact_root / ARTIFACT_SUPPLY_CHAIN_SCAN,
        label="post-DALI supply-chain scan",
    )
    dali_bootstrap = _validate_dali_bootstrap_artifact(
        artifact_root / ARTIFACT_DALI_BOOTSTRAP,
        artifact_root / ARTIFACT_DALI_REQUIREMENTS,
    )
    runner_preflight = _validate_runner_preflight_artifact(artifact_root / ARTIFACT_RUNNER_PREFLIGHT)

    inputs = _load_json_object(
        artifact_root / ARTIFACT_COMPONENT_SENSITIVITY_INPUTS,
        label="diagnostic component-sensitivity input preflight",
    )
    if inputs.get("tool") != "lightning_diagnostic_component_sensitivity_input_preflight":
        raise ValueError("component-sensitivity input preflight has unexpected tool")
    if inputs.get("score_claim") is not False or inputs.get("promotion_eligible") is not False:
        raise ValueError("component-sensitivity input preflight must be explicitly non-promotable")
    baseline_input = inputs.get("baseline_archive")
    if not isinstance(baseline_input, dict):
        raise ValueError("component-sensitivity input preflight missing baseline_archive")
    baseline_sha = baseline_input.get("sha256")
    baseline_bytes = baseline_input.get("bytes")
    if expected_baseline_archive_sha256 is not None and baseline_sha != expected_baseline_archive_sha256:
        raise ValueError(
            "baseline archive sha256 does not match expected: "
            f"actual={baseline_sha!r} expected={expected_baseline_archive_sha256!r}"
        )
    if expected_baseline_archive_size_bytes is not None and baseline_bytes != expected_baseline_archive_size_bytes:
        raise ValueError(
            "baseline archive bytes does not match expected: "
            f"actual={baseline_bytes!r} expected={expected_baseline_archive_size_bytes!r}"
        )
    extracted = inputs.get("extracted_members")
    if not isinstance(extracted, dict):
        raise ValueError("component-sensitivity input preflight missing extracted_members")
    for member in ("renderer.bin", "masks.mkv", "optimized_poses.bin"):
        meta = extracted.get(member)
        if not isinstance(meta, dict) or not meta.get("sha256") or not meta.get("bytes"):
            raise ValueError(f"component-sensitivity input preflight missing extracted member custody: {member}")

    run = _load_json_object(
        artifact_root / ARTIFACT_COMPONENT_SENSITIVITY_RUN,
        label="diagnostic component-sensitivity run metadata",
    )
    if run.get("role") != "diagnostic_component_sensitivity":
        raise ValueError("component-sensitivity run metadata has unexpected role")
    if run.get("score_claim") is not False or run.get("promotion_eligible") is not False:
        raise ValueError("component-sensitivity run metadata must be explicitly non-promotable")
    if not isinstance(run.get("profile_argv"), list) or "--device" not in run.get("profile_argv", []):
        raise ValueError("component-sensitivity run metadata must record profile argv including --device")
    profile_argv = run.get("profile_argv", [])

    summary = _load_json_object(
        artifact_root / ARTIFACT_COMPONENT_SENSITIVITY_SUMMARY,
        label="diagnostic component-sensitivity profile summary",
    )
    if summary.get("tool") != "experiments/profile_component_sensitivity.py":
        raise ValueError("component-sensitivity summary has unexpected tool")
    if summary.get("device") != "cuda":
        raise ValueError(f"component-sensitivity summary device={summary.get('device')!r}, expected 'cuda'")
    if summary.get("score_claim") is not False:
        raise ValueError("component-sensitivity summary must have score_claim=false")
    if summary.get("promotion_eligible") is not False:
        raise ValueError("component-sensitivity summary must have promotion_eligible=false")
    if summary.get("official_component_response") is not False:
        raise ValueError("component-sensitivity summary must not claim official_component_response")
    if summary.get("canonical_scorer_path") is not False:
        raise ValueError("component-sensitivity summary must not claim canonical_scorer_path")
    if "diagnostic" not in str(summary.get("evidence_grade", "")):
        raise ValueError("component-sensitivity summary evidence_grade must be diagnostic")
    sensitivity_source = summary.get("sensitivity_source")
    if sensitivity_source not in COMPONENT_SENSITIVITY_DIAGNOSTIC_SOURCES:
        raise ValueError(
            "component-sensitivity summary sensitivity_source must be one of "
            f"{sorted(COMPONENT_SENSITIVITY_DIAGNOSTIC_SOURCES)}"
        )
    if sensitivity_source == "direct_renderer_cuda_finite_difference_component_response":
        if summary.get("promotion_requested") is not True:
            raise ValueError("direct finite-difference sensitivity summary must record promotion_requested=true")
        if "--promotion-finite-difference" not in profile_argv:
            raise ValueError("direct finite-difference run metadata must include --promotion-finite-difference")
        if "--all-pairs" not in profile_argv:
            raise ValueError("direct finite-difference run metadata must include --all-pairs")
        if summary.get("n_pairs_total") != 600:
            raise ValueError("direct finite-difference sensitivity summary must cover all 600 contest pairs")
        finite_eps = summary.get("finite_difference_epsilon")
        if isinstance(finite_eps, bool) or not isinstance(finite_eps, (int, float)) or float(finite_eps) <= 0.0:
            raise ValueError("direct finite-difference sensitivity summary must record positive finite_difference_epsilon")
    elif "--promotion-finite-difference" in profile_argv:
        raise ValueError("fisher_proxy sensitivity run metadata must not include --promotion-finite-difference")
    sample_plan_validation = _validate_component_sensitivity_sample_plan_full(
        artifact_root / "sample_plan.json"
    )
    fd_shard = summary.get("finite_difference_shard")
    fd_is_partial_shard = isinstance(fd_shard, dict) and fd_shard.get("is_shard") is True
    fd_merge = summary.get("finite_difference_merge")
    fd_merged = isinstance(fd_merge, dict) and fd_merge.get("schema") == "component_sensitivity_direct_fd_merge_v1"

    curves: dict[str, Any] = {}
    map_metadata: dict[str, Any] = {}
    from tac.sensitivity_map import load_sensitivity_map

    for component in COMPONENT_RESPONSE_COMPONENTS:
        map_path = artifact_root / f"{component}_sensitivity_map.pt"
        if map_path.stat().st_size <= 0:
            raise ValueError(f"{component} sensitivity map is empty")
        try:
            map_values, metadata_for_map = load_sensitivity_map(map_path)
        except Exception as exc:
            raise ValueError(f"{component} sensitivity map is not a valid tac sensitivity map: {exc}") from exc
        if not map_values:
            raise ValueError(f"{component} sensitivity map contains no tensors")
        if metadata_for_map.get("device") != "cuda":
            raise ValueError(
                f"{component} sensitivity map metadata device={metadata_for_map.get('device')!r}, "
                "expected 'cuda'"
            )
        if metadata_for_map.get("component") != component and metadata_for_map.get("scorer_target") != component:
            raise ValueError(f"{component} sensitivity map metadata does not identify the component")
        if metadata_for_map.get("score_claim") is not False:
            raise ValueError(f"{component} sensitivity map metadata must have score_claim=false")
        if metadata_for_map.get("promotion_eligible") is not False:
            raise ValueError(f"{component} sensitivity map metadata must have promotion_eligible=false")
        if metadata_for_map.get("official_component_response") is not False:
            raise ValueError(f"{component} sensitivity map metadata must not claim official_component_response")
        if metadata_for_map.get("canonical_scorer_path") is not False:
            raise ValueError(f"{component} sensitivity map metadata must not claim canonical_scorer_path")
        if metadata_for_map.get("sensitivity_source") != sensitivity_source:
            raise ValueError(f"{component} sensitivity map metadata sensitivity_source does not match summary")
        if metadata_for_map.get("finite_difference_shard") != fd_shard:
            raise ValueError(f"{component} sensitivity map metadata finite_difference_shard does not match summary")
        map_metadata[component] = metadata_for_map
        holdout_map_path = artifact_root / f"{component}_holdout_sensitivity_map.pt"
        if holdout_map_path.stat().st_size <= 0:
            raise ValueError(f"{component} holdout sensitivity map is empty")
        try:
            holdout_values, holdout_metadata = load_sensitivity_map(holdout_map_path)
        except Exception as exc:
            raise ValueError(f"{component} holdout sensitivity map is invalid: {exc}") from exc
        if not holdout_values:
            raise ValueError(f"{component} holdout sensitivity map contains no tensors")
        if holdout_metadata.get("split") != "holdout":
            raise ValueError(f"{component} holdout sensitivity map metadata must record split=holdout")
        if holdout_metadata.get("finite_difference_shard") != fd_shard:
            raise ValueError(f"{component} holdout sensitivity map shard metadata does not match summary")
        curve_path = artifact_root / f"{component}_response_curve.json"
        curve = _load_json_object(curve_path, label=f"{component} diagnostic response curve")
        curves[component] = curve
        if curve.get("component") != component:
            raise ValueError(f"{component} response curve component={curve.get('component')!r}")
        if curve.get("device") != "cuda":
            raise ValueError(f"{component} response curve device={curve.get('device')!r}, expected 'cuda'")
        if curve.get("score_claim") is not False:
            raise ValueError(f"{component} response curve must have score_claim=false")
        if curve.get("promotion_eligible") is not False:
            raise ValueError(f"{component} response curve must have promotion_eligible=false")
        if curve.get("official_component_response") is not False:
            raise ValueError(f"{component} response curve must not claim official_component_response")
        if curve.get("canonical_scorer_path") is not False:
            raise ValueError(f"{component} response curve must not claim canonical_scorer_path")
        if curve.get("sensitivity_source") != sensitivity_source:
            raise ValueError(f"{component} response curve sensitivity_source does not match summary")
        if curve.get("component_response_path") != "direct_renderer_tensor_inprocess_scorer":
            raise ValueError(f"{component} response curve must remain diagnostic direct-renderer response")
        points = curve.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"{component} diagnostic response curve must include points")

    certification_handoff_eligible = (
        sensitivity_source == "direct_renderer_cuda_finite_difference_component_response"
        and sample_plan_validation["full_600_pair_coverage"] is True
        and not fd_is_partial_shard
        and (
            fd_shard is None
            or fd_shard.get("merge_required_for_certification_handoff") is False
        )
    )
    if certification_handoff_eligible and isinstance(fd_shard, dict) and fd_shard.get("merged_from_shards") is True:
        certification_handoff_eligible = fd_merged

    validation = {
        "schema_version": 1,
        "validated_at_utc": _utc_now(),
        "artifact_dir": str(artifact_root),
        "role": "diagnostic_component_sensitivity",
        "baseline_archive_sha256": baseline_sha,
        "baseline_archive_size_bytes": baseline_bytes,
        "device": "cuda",
        "gpu_model": runner_preflight.get("device_name"),
        "gpu_t4_match": runner_preflight.get("gpu_t4_match"),
        "metadata": metadata,
        "input_preflight": inputs,
        "run_metadata": run,
        "summary": summary,
        "sensitivity_source": sensitivity_source,
        "sample_plan_validation": sample_plan_validation,
        "planning_eligible": True,
        "certification_handoff_eligible": certification_handoff_eligible,
        "certification_candidate": (
            sensitivity_source == "direct_renderer_cuda_finite_difference_component_response"
        ),
        "map_metadata": map_metadata,
        "curves": curves,
        "dali_bootstrap": dali_bootstrap,
        "runner_preflight": runner_preflight,
        "supply_chain_pre": supply_chain_pre,
        "supply_chain_post": supply_chain_post,
        "promotion_eligible": False,
        "score_claim": False,
        "score_source": "none:diagnostic_component_sensitivity_non_promotable",
    }
    return validation


def _copy_component_response_evidence_files(source: Path, mirror: Path) -> list[str]:
    copied: list[str] = []
    for name in COMPONENT_RESPONSE_CANONICAL_ARTIFACT_FILES:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(f"component-response artifact missing required file: {src}")
        dest = mirror / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(name)
    for name in COMPONENT_RESPONSE_OPTIONAL_ARTIFACT_FILES:
        src = source / name
        if not src.is_file():
            continue
        dest = mirror / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(name)
    evals = source / "evals"
    if evals.is_dir():
        for src in sorted(evals.rglob("*")):
            if not src.is_file() or src.name not in COMPONENT_RESPONSE_EVAL_EVIDENCE_NAMES:
                continue
            rel = src.relative_to(source)
            dest = mirror / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel.as_posix())
    return copied


def mirror_local_component_response_artifact_dir(
    source_dir: str | Path,
    mirror_dir: str | Path,
    *,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    require_passed: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Mirror compact official component-response artifacts and validate them."""

    source = Path(source_dir)
    mirror = Path(mirror_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source component-response artifact dir not found: {source}")
    if mirror.exists() and any(mirror.iterdir()):
        if not overwrite:
            raise FileExistsError(f"mirror dir is not empty: {mirror}")
        shutil.rmtree(mirror)
    mirror.mkdir(parents=True, exist_ok=True)
    copied = _copy_component_response_evidence_files(source, mirror)
    validation = validate_local_component_response_artifact_dir(
        mirror,
        expected_baseline_archive_sha256=expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
        require_passed=require_passed,
    )
    validation["copied_files"] = copied
    _write_json_replace(mirror / ARTIFACT_COMPONENT_RESPONSE_VALIDATION, validation)
    return validation


def _copy_component_sensitivity_evidence_files(source: Path, mirror: Path) -> list[str]:
    copied: list[str] = []
    for name in COMPONENT_SENSITIVITY_CANONICAL_ARTIFACT_FILES:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(f"component-sensitivity artifact missing required file: {src}")
        dest = mirror / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(name)
    for name in COMPONENT_SENSITIVITY_OPTIONAL_ARTIFACT_FILES:
        src = source / name
        if not src.is_file():
            continue
        dest = mirror / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(name)
    return copied


def mirror_local_component_sensitivity_artifact_dir(
    source_dir: str | Path,
    mirror_dir: str | Path,
    *,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Mirror compact diagnostic component-sensitivity artifacts and validate them."""

    source = Path(source_dir)
    mirror = Path(mirror_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source component-sensitivity artifact dir not found: {source}")
    if mirror.exists() and any(mirror.iterdir()):
        if not overwrite:
            raise FileExistsError(f"mirror dir is not empty: {mirror}")
        shutil.rmtree(mirror)
    mirror.mkdir(parents=True, exist_ok=True)
    copied = _copy_component_sensitivity_evidence_files(source, mirror)
    validation = validate_local_component_sensitivity_artifact_dir(
        mirror,
        expected_baseline_archive_sha256=expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
    )
    validation["copied_files"] = copied
    _write_json_replace(mirror / ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION, validation)
    return validation


def _validate_ssh_artifact_source(ssh_target: str, remote_dir: str | Path) -> tuple[str, str]:
    target = str(ssh_target).strip()
    remote = str(remote_dir).strip().rstrip("/")
    if not target or any(ch in target for ch in "\r\n\0"):
        raise ValueError("ssh_target must be non-empty and must not contain control characters")
    if target == "ssh.lightning.ai":
        raise ValueError(
            "ssh_target must be a ~/.ssh/config alias or user-qualified target, not bare ssh.lightning.ai"
        )
    if any(ch.isspace() for ch in target):
        raise ValueError("ssh_target must not contain whitespace")
    if not remote or not remote.startswith("/") or any(ch in remote for ch in "\r\n\0"):
        raise ValueError("remote artifact dir must be an absolute path without control characters")
    return target, remote


def _ssh_transfer_options(connect_timeout: int | None) -> list[str]:
    options = list(SSH_AUTH_OPTIONS)
    if connect_timeout is not None:
        if connect_timeout <= 0:
            raise ValueError("ssh_connect_timeout must be positive")
        options.extend(["-o", f"ConnectTimeout={int(connect_timeout)}"])
    return options


def _ssh_command(
    ssh_bin: str,
    target: str,
    remote_command: str,
    *,
    connect_timeout: int | None,
) -> list[str]:
    return [ssh_bin, *_ssh_transfer_options(connect_timeout), target, remote_command]


def _scp_command(
    scp_bin: str,
    source: str,
    dest: str | Path,
    *,
    connect_timeout: int | None,
) -> list[str]:
    return [scp_bin, *_ssh_transfer_options(connect_timeout), "-p", source, str(dest)]


def _ssh_top_level_file_names(
    *,
    ssh_target: str,
    remote_dir: str,
    ssh_bin: str,
    ssh_connect_timeout: int | None,
) -> list[str]:
    listing = subprocess.run(
        _ssh_command(
            ssh_bin,
            ssh_target,
            "find " + shlex.quote(remote_dir) + " -maxdepth 1 -type f -print",
            connect_timeout=ssh_connect_timeout,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = remote_dir.rstrip("/") + "/"
    names: list[str] = []
    for raw_line in listing.stdout.splitlines():
        value = raw_line.strip()
        if not value:
            continue
        value = value[len(prefix) :] if value.startswith(prefix) else PurePosixPath(value).name
        if value:
            names.append(value)
    return sorted(set(names))


def classify_empty_ssh_exact_eval_artifact_dir(
    *,
    ssh_target: str,
    remote_dir: str | Path,
    mirror_dir: str | Path,
    job_name: str,
    sdk_job_name: str | None = None,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    ssh_bin: str = "ssh",
    ssh_connect_timeout: int | None = 15,
) -> dict[str, Any] | None:
    """Return an infra diagnostic when a remote exact-eval artifact dir is empty."""

    target, remote = _validate_ssh_artifact_source(ssh_target, remote_dir)
    exists_probe = subprocess.run(
        _ssh_command(ssh_bin, target, "test -d " + shlex.quote(remote), connect_timeout=ssh_connect_timeout),
        check=False,
        capture_output=True,
        text=True,
    )
    if exists_probe.returncode != 0:
        return {
            "schema_version": 1,
            "classified_at_utc": _utc_now(),
            "status": "ARTIFACT_NOT_READY",
            "failure_class": "remote_artifact_dir_missing_or_not_yet_persisted",
            "reason": (
                "remote Lightning exact-eval artifact directory is not present; "
                "the job may still be pending/running or provider artifact "
                "persistence may not have completed"
            ),
            "job_name": job_name,
            "sdk_job_name": sdk_job_name,
            "ssh_source": {
                "ssh_target": target,
                "remote_dir": remote,
                "mirror_dir": str(mirror_dir),
            },
            "expected_archive_sha256": expected_archive_sha256,
            "expected_archive_size_bytes": expected_archive_size_bytes,
            "score_claim": False,
            "method_evidence": False,
            "promotion_eligible": False,
            "evidence_grade": "invalid",
            "score_source": "none:artifact_not_ready",
            "recommended_action": "Refresh job status and retry harvest after artifacts are present.",
        }
    top_level_files = _ssh_top_level_file_names(
        ssh_target=target,
        remote_dir=remote,
        ssh_bin=ssh_bin,
        ssh_connect_timeout=ssh_connect_timeout,
    )
    if top_level_files:
        return None
    return {
        "schema_version": 1,
        "classified_at_utc": _utc_now(),
        "status": "ARTIFACT_INFRA_FAILURE",
        "terminal_class": LIGHTNING_EMPTY_ARTIFACT_INFRA_TERMINAL_CLASS,
        "failure_class": "provider_or_artifact_transport_infrastructure",
        "reason": (
            "remote Lightning exact-eval artifact directory exists but contains "
            "no top-level files; no contest_auth_eval.json or archive.zip was produced"
        ),
        "job_name": job_name,
        "sdk_job_name": sdk_job_name,
        "ssh_source": {
            "ssh_target": target,
            "remote_dir": remote,
            "mirror_dir": str(mirror_dir),
            "top_level_files": [],
        },
        "expected_archive_sha256": expected_archive_sha256,
        "expected_archive_size_bytes": expected_archive_size_bytes,
        "score_claim": False,
        "method_evidence": False,
        "promotion_eligible": False,
        "evidence_grade": "invalid",
        "score_source": "none:empty_lightning_artifact_dir",
        "recommended_action": (
            "Treat as provider/infrastructure failure, not PR84/PR82 method "
            "evidence. Reroute or retry only after the orchestrator claims a "
            "fresh lane/job."
        ),
    }


def _safe_remote_rel(path: str) -> str:
    rel = str(path).strip()
    pure = Path(rel)
    if not rel or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe remote evidence path: {path!r}")
    return rel


def mirror_ssh_artifact_dir(
    *,
    ssh_target: str,
    remote_dir: str | Path,
    mirror_dir: str | Path,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    require_adjudication: bool = False,
    overwrite: bool = False,
    ssh_bin: str = "ssh",
    scp_bin: str = "scp",
    ssh_connect_timeout: int | None = 15,
) -> dict[str, Any]:
    """Mirror a Lightning Studio artifact directory over SSH and validate it.

    This keeps terminal-job harvest reproducible: the remote source path,
    local mirror, archive identity, and adjudication checks are all captured in
    one stateful operation instead of relying on ad hoc operator copies.
    """

    target, remote = _validate_ssh_artifact_source(ssh_target, remote_dir)
    mirror = Path(mirror_dir)
    if mirror.exists() and any(mirror.iterdir()):
        if not overwrite:
            raise FileExistsError(f"mirror dir is not empty: {mirror}")
        shutil.rmtree(mirror)
    mirror.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        _ssh_command(ssh_bin, target, "test -d " + shlex.quote(remote), connect_timeout=ssh_connect_timeout),
        check=True,
        capture_output=True,
        text=True,
    )
    top_level_files = _ssh_top_level_file_names(
        ssh_target=target,
        remote_dir=remote,
        ssh_bin=ssh_bin,
        ssh_connect_timeout=ssh_connect_timeout,
    )
    top_level_file_set = set(top_level_files)
    copied_files: list[str] = []
    missing_required_files: list[str] = []
    artifact_file_names = CANONICAL_ARTIFACT_FILES
    if ARTIFACT_METADATA in top_level_file_set:
        subprocess.run(
            _scp_command(
                scp_bin,
                f"{target}:{remote}/{ARTIFACT_METADATA}",
                mirror / ARTIFACT_METADATA,
                connect_timeout=ssh_connect_timeout,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        copied_files.append(ARTIFACT_METADATA)
        try:
            metadata = _load_json_object(mirror / ARTIFACT_METADATA, label="Lightning queue metadata")
            artifact_file_names = _exact_eval_artifact_files_for_role(metadata.get("role"))
        except Exception:
            artifact_file_names = CANONICAL_ARTIFACT_FILES
    for name in artifact_file_names:
        if name == ARTIFACT_METADATA and name in copied_files:
            continue
        if name not in top_level_file_set:
            missing_required_files.append(name)
            continue
        subprocess.run(
            _scp_command(
                scp_bin,
                f"{target}:{remote}/{name}",
                mirror / name,
                connect_timeout=ssh_connect_timeout,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        copied_files.append(name)
    for name in OPTIONAL_ARTIFACT_FILES:
        exists = subprocess.run(
            _ssh_command(
                ssh_bin,
                target,
                "test -f " + shlex.quote(f"{remote}/{name}"),
                connect_timeout=ssh_connect_timeout,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if exists.returncode != 0:
            continue
        subprocess.run(
            _scp_command(
                scp_bin,
                f"{target}:{remote}/{name}",
                mirror / name,
                connect_timeout=ssh_connect_timeout,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        copied_files.append(name)
    if missing_required_files:
        recovered_files = _reconstruct_missing_adjudication_artifacts_from_metadata(mirror)
        if recovered_files:
            copied_files.extend(f"{name}:reconstructed" for name in recovered_files)
            missing_required_files = [
                name for name in missing_required_files if not (mirror / name).is_file()
            ]
    if missing_required_files:
        diagnostic: dict[str, Any] = {
            "schema_version": 1,
            "classified_at_utc": _utc_now(),
            "status": "ARTIFACT_INFRA_FAILURE",
            "terminal_class": LIGHTNING_MISSING_EXACT_EVAL_JSON_TERMINAL_CLASS,
            "failure_class": "runtime_or_harness_failure_before_score_json",
            "reason": (
                "remote Lightning exact-eval artifact directory contains partial "
                "artifacts but is missing required exact-score JSON/report files"
            ),
            "missing_required_files": missing_required_files,
            "job_artifact_files": top_level_files,
            "expected_archive_sha256": expected_archive_sha256,
            "expected_archive_size_bytes": expected_archive_size_bytes,
            "score_claim": False,
            "method_evidence": False,
            "promotion_eligible": False,
            "evidence_grade": "invalid",
            "score_source": "none:missing_contest_auth_eval_json",
            "ssh_source": {
                "ssh_target": target,
                "remote_dir": remote,
                "mirror_dir": str(mirror),
                "copied_files": copied_files,
                "ssh_connect_timeout": ssh_connect_timeout,
            },
            "recommended_action": (
                "Preserve the partial artifacts and logs as a harness/runtime "
                "failure. Do not rank, retire, promote, or claim score without "
                "contest_auth_eval.json and adjudication artifacts."
            ),
        }
        archive_path = mirror / ARTIFACT_ARCHIVE
        if archive_path.is_file():
            actual_identity = archive_identity(archive_path)
            diagnostic["archive_identity"] = actual_identity
            if (
                expected_archive_sha256 is not None
                and actual_identity["archive_sha256"] != expected_archive_sha256
            ):
                raise ValueError(
                    "partial artifact archive sha256 does not match expected: "
                    f"actual={actual_identity['archive_sha256']!r} "
                    f"expected={expected_archive_sha256!r}"
                )
            if (
                expected_archive_size_bytes is not None
                and actual_identity["archive_size_bytes"] != expected_archive_size_bytes
            ):
                raise ValueError(
                    "partial artifact archive bytes does not match expected: "
                    f"actual={actual_identity['archive_size_bytes']} "
                    f"expected={expected_archive_size_bytes}"
                )
        _write_json_replace(mirror / ARTIFACT_INFRA_FAILURE, diagnostic)
        _write_json_replace(mirror / ARTIFACT_VALIDATION, diagnostic)
        return diagnostic
    validation = validate_local_artifact_dir(
        mirror,
        expected_archive_sha256=expected_archive_sha256,
        expected_archive_size_bytes=expected_archive_size_bytes,
        require_adjudication=require_adjudication,
    )
    validation["ssh_source"] = {
        "ssh_target": target,
        "remote_dir": remote,
        "mirror_dir": str(mirror),
        "copied_files": copied_files,
        "ssh_connect_timeout": ssh_connect_timeout,
    }
    _write_json_replace(mirror / ARTIFACT_VALIDATION, validation)
    return validation


def mirror_ssh_component_response_artifact_dir(
    *,
    ssh_target: str,
    remote_dir: str | Path,
    mirror_dir: str | Path,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    require_passed: bool = False,
    overwrite: bool = False,
    ssh_bin: str = "ssh",
    scp_bin: str = "scp",
    ssh_connect_timeout: int | None = 15,
) -> dict[str, Any]:
    """Mirror compact official component-response artifacts over SSH and validate."""

    target, remote = _validate_ssh_artifact_source(ssh_target, remote_dir)
    mirror = Path(mirror_dir)
    if mirror.exists() and any(mirror.iterdir()):
        if not overwrite:
            raise FileExistsError(f"mirror dir is not empty: {mirror}")
        shutil.rmtree(mirror)
    mirror.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        _ssh_command(ssh_bin, target, "test -d " + shlex.quote(remote), connect_timeout=ssh_connect_timeout),
        check=True,
        capture_output=True,
        text=True,
    )
    copied_files: list[str] = []
    for name in COMPONENT_RESPONSE_CANONICAL_ARTIFACT_FILES:
        subprocess.run(
            _scp_command(
                scp_bin,
                f"{target}:{remote}/{name}",
                mirror / name,
                connect_timeout=ssh_connect_timeout,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        copied_files.append(name)
    for name in COMPONENT_RESPONSE_OPTIONAL_ARTIFACT_FILES:
        exists = subprocess.run(
            _ssh_command(
                ssh_bin,
                target,
                "test -f " + shlex.quote(f"{remote}/{name}"),
                connect_timeout=ssh_connect_timeout,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if exists.returncode != 0:
            continue
        subprocess.run(
            _scp_command(
                scp_bin,
                f"{target}:{remote}/{name}",
                mirror / name,
                connect_timeout=ssh_connect_timeout,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        copied_files.append(name)

    find_cmd = (
        "cd "
        + shlex.quote(remote)
        + " && find evals -type f \\( "
        + " -o ".join(
            f"-name {shlex.quote(name)}"
            for name in sorted(COMPONENT_RESPONSE_EVAL_EVIDENCE_NAMES)
        )
        + " \\) -print 2>/dev/null || true"
    )
    found = subprocess.run(
        _ssh_command(ssh_bin, target, find_cmd, connect_timeout=ssh_connect_timeout),
        check=True,
        capture_output=True,
        text=True,
    )
    for line in found.stdout.splitlines():
        rel = _safe_remote_rel(line)
        dest = mirror / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            _scp_command(
                scp_bin,
                f"{target}:{remote}/{rel}",
                dest,
                connect_timeout=ssh_connect_timeout,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        copied_files.append(rel)

    validation = validate_local_component_response_artifact_dir(
        mirror,
        expected_baseline_archive_sha256=expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
        require_passed=require_passed,
    )
    validation["ssh_source"] = {
        "ssh_target": target,
        "remote_dir": remote,
        "mirror_dir": str(mirror),
        "copied_files": copied_files,
        "ssh_connect_timeout": ssh_connect_timeout,
    }
    _write_json_replace(mirror / ARTIFACT_COMPONENT_RESPONSE_VALIDATION, validation)
    return validation


def mirror_ssh_component_sensitivity_artifact_dir(
    *,
    ssh_target: str,
    remote_dir: str | Path,
    mirror_dir: str | Path,
    expected_baseline_archive_sha256: str | None = None,
    expected_baseline_archive_size_bytes: int | None = None,
    overwrite: bool = False,
    ssh_bin: str = "ssh",
    scp_bin: str = "scp",
    ssh_connect_timeout: int | None = 15,
) -> dict[str, Any]:
    """Mirror compact diagnostic component-sensitivity artifacts over SSH and validate."""

    target, remote = _validate_ssh_artifact_source(ssh_target, remote_dir)
    mirror = Path(mirror_dir)
    if mirror.exists() and any(mirror.iterdir()):
        if not overwrite:
            raise FileExistsError(f"mirror dir is not empty: {mirror}")
        shutil.rmtree(mirror)
    mirror.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        _ssh_command(ssh_bin, target, "test -d " + shlex.quote(remote), connect_timeout=ssh_connect_timeout),
        check=True,
        capture_output=True,
        text=True,
    )
    copied_files: list[str] = []
    for name in COMPONENT_SENSITIVITY_CANONICAL_ARTIFACT_FILES:
        subprocess.run(
            _scp_command(
                scp_bin,
                f"{target}:{remote}/{name}",
                mirror / name,
                connect_timeout=ssh_connect_timeout,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        copied_files.append(name)
    for name in COMPONENT_SENSITIVITY_OPTIONAL_ARTIFACT_FILES:
        exists = subprocess.run(
            _ssh_command(
                ssh_bin,
                target,
                "test -f " + shlex.quote(f"{remote}/{name}"),
                connect_timeout=ssh_connect_timeout,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if exists.returncode != 0:
            continue
        subprocess.run(
            _scp_command(
                scp_bin,
                f"{target}:{remote}/{name}",
                mirror / name,
                connect_timeout=ssh_connect_timeout,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        copied_files.append(name)

    validation = validate_local_component_sensitivity_artifact_dir(
        mirror,
        expected_baseline_archive_sha256=expected_baseline_archive_sha256,
        expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
    )
    validation["ssh_source"] = {
        "ssh_target": target,
        "remote_dir": remote,
        "mirror_dir": str(mirror),
        "copied_files": copied_files,
        "ssh_connect_timeout": ssh_connect_timeout,
    }
    _write_json_replace(mirror / ARTIFACT_COMPONENT_SENSITIVITY_VALIDATION, validation)
    return validation


class LightningBatchJobsClient:
    """Small wrapper around ``lightning_sdk.Job.run`` with local state."""

    def __init__(self, *, state_path: Path = LIGHTNING_BATCH_STATE, job_cls: object | None = None) -> None:
        self.state_path = state_path
        self._job_cls = job_cls

    @staticmethod
    def _import_job_cls() -> object:
        # `lightning_sdk.__init__` performs a PyPI version check on import by
        # default. Keep promotion tooling deterministic and avoid unnecessary
        # network touches unless an operator explicitly opts back in.
        os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")
        try:
            from lightning_sdk import Job
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "lightning_sdk is required for non-dry-run Lightning Batch Jobs"
            ) from exc
        return Job

    def list_records(self) -> list[dict[str, Any]]:
        return _load_state(self.state_path)

    def record(self, record: dict[str, Any]) -> None:
        def append_record(records: list[dict[str, Any]]) -> None:
            records.append(record)

        _mutate_state(self.state_path, append_record)

    def _replace_record(self, index: int, record: dict[str, Any]) -> None:
        def replace_record(records: list[dict[str, Any]]) -> None:
            records[index] = record

        _mutate_state(self.state_path, replace_record)

    def replace_latest_record_for_job(
        self,
        job_name: str,
        updater: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        """Atomically update and persist the newest state record for a job."""

        def update_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
            for idx in range(len(records) - 1, -1, -1):
                record = records[idx]
                spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
                queue = record.get("queue") if isinstance(record.get("queue"), dict) else {}
                job = record.get("job") if isinstance(record.get("job"), dict) else {}
                if job_name not in {spec.get("name"), queue.get("job_name"), job.get("name")}:
                    continue
                updated = updater(dict(record))
                if updated is None:
                    return None
                records[idx] = updated
                return updated
            raise KeyError(f"Lightning Batch Job record not found: {job_name}")

        return _mutate_state(self.state_path, update_record)

    def _find_record_index(self, job_name: str) -> int:
        records = self.list_records()
        for idx in range(len(records) - 1, -1, -1):
            record = records[idx]
            spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
            job = record.get("job") if isinstance(record.get("job"), dict) else {}
            if spec.get("name") == job_name or job.get("name") == job_name:
                return idx
        raise KeyError(f"Lightning Batch Job record not found: {job_name}")

    def submit(self, spec: LightningBatchJobSpec, *, dry_run: bool = False) -> dict[str, Any]:
        spec.validate()
        recorded_at = _utc_now()
        record: dict[str, Any] = {
            "schema_version": 2,
            "recorded_at_utc": recorded_at,
            "dry_run": dry_run,
            "queue": _queue_record(spec, queued_at_utc=recorded_at),
            "spec": spec.asdict(),
        }
        if dry_run:
            record["status"] = "DRY_RUN"
            self.record(record)
            return record

        job_cls = self._job_cls or self._import_job_cls()
        try:
            job = job_cls.run(
                name=spec.name,
                machine=spec.machine,
                command=spec.command,
                studio=spec.studio,
                image=spec.image,
                teamspace=spec.teamspace,
                org=spec.org,
                user=spec.user,
                cloud_account=spec.cloud_account,
                env=spec.env or None,
                interruptible=spec.interruptible,
                max_runtime=spec.max_runtime,
                reuse_snapshot=spec.reuse_snapshot,
                path_mappings=spec.path_mappings,
                scratch_disks=spec.scratch_disks,
            )
        except Exception as exc:
            terminal_class = None
            if spec.studio and spec.cloud_account and _is_studio_cloud_account_mismatch(exc):
                terminal_class = LIGHTNING_STUDIO_CLOUD_ACCOUNT_MISMATCH_TERMINAL_CLASS
            submit_error: dict[str, Any] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if terminal_class:
                submit_error["terminal_class"] = terminal_class
            record.update(
                {
                    "status": "SUBMIT_FAILED",
                    "submit_failed_at_utc": _utc_now(),
                    "submit_error": submit_error,
                    "status_history": [
                        {"recorded_at_utc": recorded_at, "status": "SUBMIT_ATTEMPTED"},
                        {"recorded_at_utc": _utc_now(), "status": "SUBMIT_FAILED"},
                    ],
                }
            )
            if terminal_class:
                record["terminal_class"] = terminal_class
            self.record(record)
            if terminal_class == LIGHTNING_STUDIO_CLOUD_ACCOUNT_MISMATCH_TERMINAL_CLASS:
                raise LightningStudioCloudAccountMismatchError(
                    job_name=spec.name,
                    studio=spec.studio,
                    cloud_account=spec.cloud_account,
                    machine=spec.machine,
                    original_error_type=type(exc).__name__,
                    original_message=str(exc),
                ) from exc
            raise
        snapshot = job_status_snapshot(job)
        record.update(
            {
                "status": "SUBMITTED",
                "job": snapshot,
                "status_history": [{"recorded_at_utc": recorded_at, "status": "SUBMITTED"}],
            }
        )
        self.record(record)
        return record

    def refresh_status_from_job(self, *, job_name: str, job: object) -> dict[str, Any]:
        """Refresh a local record from SDK job attributes; never reads logs."""

        snapshot = job_status_snapshot(job)

        def apply_snapshot(record: dict[str, Any]) -> dict[str, Any]:
            observed_status = _extract_job_status(snapshot)
            previous_status = record.get("status")
            anomaly = _lightning_status_transition_anomaly(previous_status, observed_status)
            accepted_status = observed_status
            if _lightning_refresh_requires_fail_closed_status(record, anomaly, observed_status):
                accepted_status = _REMOTE_STATUS_RECONCILIATION_REQUIRED
            if observed_status is not None:
                record["status"] = accepted_status
                record["remote_observed_status"] = observed_status
                record["remote_status_accepted"] = accepted_status == observed_status
            if snapshot.get("id") in (None, ""):
                record["identity_confidence"] = "name_only"
                record["identity_reconciliation_required"] = True
            else:
                record["identity_confidence"] = "sdk_job_id"
                record["identity_reconciliation_required"] = False
            record["job"] = snapshot
            history = list(record.get("status_history") or [])
            history_entry = {
                "recorded_at_utc": snapshot["refreshed_at_utc"],
                "previous_status": previous_status,
                "observed_status": observed_status,
                "accepted_status": record.get("status"),
                "status": record.get("status"),
                "source": snapshot["source"],
                "job_snapshot": snapshot,
                "identity_confidence": record.get("identity_confidence"),
            }
            if anomaly is not None:
                anomaly = {
                    **anomaly,
                    "recorded_at_utc": snapshot["refreshed_at_utc"],
                    "source": snapshot["source"],
                    "status_history_index": len(history),
                    "accepted_status": record.get("status"),
                }
                anomalies = list(record.get("status_anomalies") or [])
                anomalies.append(anomaly)
                record["status_anomalies"] = anomalies
                record["status_reconciliation_required"] = True
                history_entry["anomaly"] = anomaly
            history.append(history_entry)
            record["status_history"] = history
            _attach_lightning_status_history_anomalies(record)
            return record

        updated = self.replace_latest_record_for_job(job_name, apply_snapshot)
        if updated is None:
            raise KeyError(f"Lightning Batch Job record not found: {job_name}")
        return updated

    def harvest_local_artifacts(
        self,
        *,
        job_name: str,
        artifact_dir: str | Path,
        mirror_dir: str | Path | None = None,
        expected_archive_sha256: str | None = None,
        expected_archive_size_bytes: int | None = None,
        require_adjudication: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Validate or mirror a local artifact dir and attach the result to state."""

        index = self._find_record_index(job_name)
        records = self.list_records()
        record = dict(records[index])
        spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
        expected_sha = expected_archive_sha256 or spec.get("expected_archive_sha256")
        expected_bytes = expected_archive_size_bytes or spec.get("expected_archive_size_bytes")
        if mirror_dir is None:
            validation = validate_local_artifact_dir(
                artifact_dir,
                expected_archive_sha256=expected_sha,
                expected_archive_size_bytes=expected_bytes,
                require_adjudication=require_adjudication,
            )
        else:
            validation = mirror_local_artifact_dir(
                artifact_dir,
                mirror_dir,
                expected_archive_sha256=expected_sha,
                expected_archive_size_bytes=expected_bytes,
                require_adjudication=require_adjudication,
                overwrite=overwrite,
            )
        harvests = list(record.get("harvests") or [])
        harvests.append(validation)
        record["harvests"] = harvests
        record["status"] = "HARVESTED"
        record.pop("terminal_class", None)
        history = list(record.get("status_history") or [])
        history.append(
            {
                "recorded_at_utc": validation["validated_at_utc"],
                "status": "HARVESTED",
                "source": "local_artifact_validation",
            }
        )
        record["status_history"] = history
        self._replace_record(index, record)
        return validation

    def harvest_ssh_artifacts(
        self,
        *,
        job_name: str,
        ssh_target: str,
        remote_dir: str | Path | None = None,
        mirror_dir: str | Path | None = None,
        expected_archive_sha256: str | None = None,
        expected_archive_size_bytes: int | None = None,
        require_adjudication: bool = False,
        overwrite: bool = False,
        ssh_bin: str = "ssh",
        scp_bin: str = "scp",
        ssh_connect_timeout: int | None = 15,
    ) -> dict[str, Any]:
        """Mirror a remote Lightning artifact dir over SSH, validate, and record."""

        index = self._find_record_index(job_name)
        records = self.list_records()
        record = dict(records[index])
        spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
        queue = record.get("queue") if isinstance(record.get("queue"), dict) else {}
        job = record.get("job") if isinstance(record.get("job"), dict) else {}
        persisted_remote = None
        if isinstance(queue, dict):
            sdk_artifact_path = job.get("artifact_path") or queue.get("sdk_artifact_path")
            spec_remote_output = spec.get("remote_output_dir")
            if isinstance(sdk_artifact_path, str) and isinstance(spec_remote_output, str):
                persisted_remote = lightning_sdk_persisted_studio_output_dir(
                    sdk_artifact_path=sdk_artifact_path,
                    remote_output_dir=spec_remote_output,
                )
        resolved_remote = remote_dir or persisted_remote or spec.get("remote_output_dir")
        resolved_mirror = mirror_dir or spec.get("local_artifact_dir")
        if not resolved_remote:
            raise ValueError("remote artifact dir is required; state record has no remote_output_dir")
        if not resolved_mirror:
            raise ValueError("mirror dir is required; state record has no local_artifact_dir")
        expected_sha = expected_archive_sha256 or spec.get("expected_archive_sha256")
        expected_bytes = expected_archive_size_bytes or spec.get("expected_archive_size_bytes")
        sdk_job_name = job.get("name") if isinstance(job.get("name"), str) else lightning_sdk_job_name(job_name)
        infra_failure = classify_empty_ssh_exact_eval_artifact_dir(
            ssh_target=ssh_target,
            remote_dir=resolved_remote,
            mirror_dir=resolved_mirror,
            job_name=job_name,
            sdk_job_name=sdk_job_name,
            expected_archive_sha256=expected_sha if isinstance(expected_sha, str) else None,
            expected_archive_size_bytes=expected_bytes if isinstance(expected_bytes, int) else None,
            ssh_bin=ssh_bin,
            ssh_connect_timeout=ssh_connect_timeout,
        )
        if infra_failure is not None:
            if infra_failure.get("status") == "ARTIFACT_NOT_READY":
                return infra_failure
            mirror_path = Path(resolved_mirror)
            mirror_path.mkdir(parents=True, exist_ok=True)
            _write_json_replace(mirror_path / ARTIFACT_INFRA_FAILURE, infra_failure)
            failures = list(record.get("artifact_failures") or [])
            failures.append(infra_failure)
            record["artifact_failures"] = failures
            record["status"] = "ARTIFACT_INFRA_FAILURE"
            record["terminal_class"] = infra_failure["terminal_class"]
            history = list(record.get("status_history") or [])
            history.append(
                {
                    "recorded_at_utc": infra_failure["classified_at_utc"],
                    "status": "ARTIFACT_INFRA_FAILURE",
                    "terminal_class": infra_failure["terminal_class"],
                    "source": "ssh_artifact_preharvest_classification",
                }
            )
            record["status_history"] = history
            self._replace_record(index, record)
            return infra_failure
        validation = mirror_ssh_artifact_dir(
            ssh_target=ssh_target,
            remote_dir=resolved_remote,
            mirror_dir=resolved_mirror,
            expected_archive_sha256=expected_sha,
            expected_archive_size_bytes=expected_bytes,
            require_adjudication=require_adjudication,
            overwrite=overwrite,
            ssh_bin=ssh_bin,
            scp_bin=scp_bin,
            ssh_connect_timeout=ssh_connect_timeout,
        )
        harvests = list(record.get("harvests") or [])
        if validation.get("status") == "ARTIFACT_INFRA_FAILURE":
            failures = list(record.get("artifact_failures") or [])
            failures.append(validation)
            record["artifact_failures"] = failures
            record["status"] = "ARTIFACT_INFRA_FAILURE"
            if isinstance(validation.get("terminal_class"), str):
                record["terminal_class"] = validation["terminal_class"]
            history = list(record.get("status_history") or [])
            history.append(
                {
                    "recorded_at_utc": validation.get("classified_at_utc", _utc_now()),
                    "status": "ARTIFACT_INFRA_FAILURE",
                    "terminal_class": validation.get("terminal_class"),
                    "source": "ssh_artifact_partial_failure_classification",
                }
            )
            record["status_history"] = history
            self._replace_record(index, record)
            return validation
        harvests.append(validation)
        record["harvests"] = harvests
        record["status"] = "HARVESTED"
        record.pop("terminal_class", None)
        history = list(record.get("status_history") or [])
        history.append(
            {
                "recorded_at_utc": validation["validated_at_utc"],
                "status": "HARVESTED",
                "source": "ssh_artifact_validation",
            }
        )
        record["status_history"] = history
        self._replace_record(index, record)
        return validation

    def harvest_ssh_component_response_artifacts(
        self,
        *,
        job_name: str,
        ssh_target: str,
        remote_dir: str | Path | None = None,
        mirror_dir: str | Path | None = None,
        expected_baseline_archive_sha256: str | None = None,
        expected_baseline_archive_size_bytes: int | None = None,
        require_passed: bool = False,
        overwrite: bool = False,
        ssh_bin: str = "ssh",
        scp_bin: str = "scp",
        ssh_connect_timeout: int | None = 15,
    ) -> dict[str, Any]:
        """Mirror component-response artifacts over SSH, validate, and record."""

        index = self._find_record_index(job_name)
        records = self.list_records()
        record = dict(records[index])
        spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
        queue = record.get("queue") if isinstance(record.get("queue"), dict) else {}
        job = record.get("job") if isinstance(record.get("job"), dict) else {}
        persisted_remote = None
        if isinstance(queue, dict):
            sdk_artifact_path = job.get("artifact_path") or queue.get("sdk_artifact_path")
            spec_remote_output = spec.get("remote_output_dir")
            if isinstance(sdk_artifact_path, str) and isinstance(spec_remote_output, str):
                persisted_remote = lightning_sdk_persisted_studio_output_dir(
                    sdk_artifact_path=sdk_artifact_path,
                    remote_output_dir=spec_remote_output,
                )
        resolved_remote = remote_dir or persisted_remote or spec.get("remote_output_dir")
        resolved_mirror = mirror_dir or spec.get("local_artifact_dir")
        if not resolved_remote:
            raise ValueError("remote artifact dir is required; state record has no remote_output_dir")
        if not resolved_mirror:
            raise ValueError("mirror dir is required; state record has no local_artifact_dir")
        expected_sha = (
            expected_baseline_archive_sha256
            or spec.get("expected_baseline_archive_sha256")
            or spec.get("expected_archive_sha256")
        )
        expected_bytes = (
            expected_baseline_archive_size_bytes
            or spec.get("expected_baseline_archive_size_bytes")
            or spec.get("expected_archive_size_bytes")
        )
        validation = mirror_ssh_component_response_artifact_dir(
            ssh_target=ssh_target,
            remote_dir=resolved_remote,
            mirror_dir=resolved_mirror,
            expected_baseline_archive_sha256=expected_sha,
            expected_baseline_archive_size_bytes=expected_bytes,
            require_passed=require_passed,
            overwrite=overwrite,
            ssh_bin=ssh_bin,
            scp_bin=scp_bin,
            ssh_connect_timeout=ssh_connect_timeout,
        )
        harvests = list(record.get("harvests") or [])
        harvests.append(validation)
        record["harvests"] = harvests
        record["status"] = "HARVESTED"
        history = list(record.get("status_history") or [])
        history.append(
            {
                "recorded_at_utc": validation["validated_at_utc"],
                "status": "HARVESTED",
                "source": "ssh_component_response_artifact_validation",
            }
        )
        record["status_history"] = history
        self._replace_record(index, record)
        return validation

    def harvest_ssh_component_sensitivity_artifacts(
        self,
        *,
        job_name: str,
        ssh_target: str,
        remote_dir: str | Path | None = None,
        mirror_dir: str | Path | None = None,
        expected_baseline_archive_sha256: str | None = None,
        expected_baseline_archive_size_bytes: int | None = None,
        overwrite: bool = False,
        ssh_bin: str = "ssh",
        scp_bin: str = "scp",
        ssh_connect_timeout: int | None = 15,
    ) -> dict[str, Any]:
        """Mirror diagnostic component-sensitivity artifacts over SSH, validate, and record."""

        index = self._find_record_index(job_name)
        records = self.list_records()
        record = dict(records[index])
        spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
        queue = record.get("queue") if isinstance(record.get("queue"), dict) else {}
        job = record.get("job") if isinstance(record.get("job"), dict) else {}
        persisted_remote = None
        if isinstance(queue, dict):
            sdk_artifact_path = job.get("artifact_path") or queue.get("sdk_artifact_path")
            spec_remote_output = spec.get("remote_output_dir")
            if isinstance(sdk_artifact_path, str) and isinstance(spec_remote_output, str):
                persisted_remote = lightning_sdk_persisted_studio_output_dir(
                    sdk_artifact_path=sdk_artifact_path,
                    remote_output_dir=spec_remote_output,
                )
        resolved_remote = remote_dir or persisted_remote or spec.get("remote_output_dir")
        resolved_mirror = mirror_dir or spec.get("local_artifact_dir")
        if not resolved_remote:
            raise ValueError("remote artifact dir is required; state record has no remote_output_dir")
        if not resolved_mirror:
            raise ValueError("mirror dir is required; state record has no local_artifact_dir")
        expected_sha = (
            expected_baseline_archive_sha256
            or spec.get("expected_baseline_archive_sha256")
            or spec.get("expected_archive_sha256")
        )
        expected_bytes = (
            expected_baseline_archive_size_bytes
            or spec.get("expected_baseline_archive_size_bytes")
            or spec.get("expected_archive_size_bytes")
        )
        validation = mirror_ssh_component_sensitivity_artifact_dir(
            ssh_target=ssh_target,
            remote_dir=resolved_remote,
            mirror_dir=resolved_mirror,
            expected_baseline_archive_sha256=expected_sha,
            expected_baseline_archive_size_bytes=expected_bytes,
            overwrite=overwrite,
            ssh_bin=ssh_bin,
            scp_bin=scp_bin,
            ssh_connect_timeout=ssh_connect_timeout,
        )
        harvests = list(record.get("harvests") or [])
        harvests.append(validation)
        record["harvests"] = harvests
        record["status"] = "HARVESTED"
        history = list(record.get("status_history") or [])
        history.append(
            {
                "recorded_at_utc": validation["validated_at_utc"],
                "status": "HARVESTED",
                "source": "ssh_component_sensitivity_artifact_validation",
            }
        )
        record["status_history"] = history
        self._replace_record(index, record)
        return validation
