# SPDX-License-Identifier: MIT
"""Shared Modal runtime contracts for contest CUDA work.

Lane-specific Modal actuators should own lane parameters, custody metadata,
and recovery policy. They should not each rediscover the scorer/runtime
dependency closure. This module keeps the reusable Modal image contract in the
deploy layer so experiment files stay thin.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

CONTEST_CUDA_APT_PACKAGES: tuple[str, ...] = (
    "bash",
    "build-essential",
    "curl",
    "ffmpeg",
    "git",
    "libgl1",
    "libglib2.0-0",
    "unzip",
    "wget",
)

CONTEST_CUDA_PIP_PACKAGES: tuple[str, ...] = (
    "torch==2.5.1",
    "torchvision",
    "safetensors",
    "einops",
    "segmentation-models-pytorch",
    "av",
    "brotli",
    "click",
    "nvidia-dali-cuda120==1.52.0",
    "tqdm",
    "timm",
    "scipy",
    "numpy<2.0",
    "Pillow",
    "pydantic>=2.0",
)

CONTEST_SCORER_IMPORT_PROBE_MODULES: tuple[str, ...] = (
    "safetensors.torch",
    "segmentation_models_pytorch",
    "upstream.modules",
    "tac.scorer",
)

DALI_DISABLE_NVML_VALUE = "1"
PYTORCH_CUDA_ALLOC_CONF_VALUE = "expandable_segments:True"
# Catalog #244 (META-CONSOLIDATION-CRITICAL-PLUS-POC 2026-05-15) sister of
# Catalog #224 (`check_modal_training_image_includes_hard_runtime_deps`).
# CUBLAS_WORKSPACE_CONFIG=:4096:8 silences the deterministic-CUDA matmul
# warning emitted by upstream evaluate.py + scorer forwards on T4/A100. D1
# incident anchor 2026-05-15 commit 611495f26 added it to the D1 lane script
# alongside DALI_DISABLE_NVML and PYTORCH_CUDA_ALLOC_CONF. Now consolidated
# here so the substrate driver generator + Modal env block can import a
# single canonical value.
CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"
NVIDIA_DALI_EXTRA_INDEX_URL = "https://pypi.nvidia.com"


def build_contest_cuda_base_image(
    modal_module: Any,
    *,
    python_version: str = "3.11",
    extra_pip_packages: Iterable[str] = (),
    install_uv: bool = True,
):
    """Return a Modal image with the shared contest-CUDA scorer runtime.

    ``modal_module`` is passed in by the actuator instead of imported here so
    this module remains cheap to import in local tests and OSS tooling.
    """

    image = (
        modal_module.Image.debian_slim(python_version=python_version)
        .apt_install(*CONTEST_CUDA_APT_PACKAGES)
        .pip_install(
            *CONTEST_CUDA_PIP_PACKAGES,
            *tuple(extra_pip_packages),
            extra_index_url=NVIDIA_DALI_EXTRA_INDEX_URL,
        )
    )
    if install_uv:
        image = image.run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
        )
    return image
