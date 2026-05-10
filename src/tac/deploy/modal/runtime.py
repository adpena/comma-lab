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
