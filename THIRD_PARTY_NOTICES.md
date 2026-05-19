# Third-Party Notices

This file lists external dependencies that `tac` (the Task-Aware Compression
library distributed in this repository) interoperates with, links against, or
vendors, along with their respective licenses. This document is maintained in
the spirit of the comma.ai openpilot project's third-party notices conventions.

The `tac` source code in this repository is licensed under the MIT License (see
`LICENSE`). The licenses below apply to the corresponding upstream projects as
distributed by their authors; we do not relicense them.

## Hard runtime dependencies

These ship by default with `pip install tac` and `uv pip install tac`. All are
permissive (MIT / Apache-2.0 / BSD / PSF-equivalent):

| Project | License | Use |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | Neural network training and inference |
| [pydantic](https://github.com/pydantic/pydantic) | MIT | Typed data models and validation |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | Array math |
| [Click](https://github.com/pallets/click) | BSD-3-Clause | CLI |
| [brotli](https://github.com/google/brotli) | MIT | Archive payload compression |
| [constriction](https://github.com/bamler-lab/constriction) | MIT OR Apache-2.0 OR BSL-1.0 | Range / ANS entropy coding (PR86/HPAC-family wire format) |
| [cryptography](https://github.com/pyca/cryptography) | Apache-2.0 OR BSD-3-Clause | Ed25519 attestation signatures for the Lane C compliance gate |
| [cmaes](https://github.com/CyberAgentAILab/cmaes) | MIT | Black-box CPU-planning optimization |
| [Optuna](https://github.com/optuna/optuna) | MIT | TPE / NSGA-II / multi-objective optimization |

## Optional runtime extras

Extras opt-in to specific capabilities. The default install (`pip install tac`)
does NOT pull these.

| Extra | Project | License | Use |
|---|---|---|---|
| `[pr86_replay]` | [pyppmd](https://github.com/miurahr/pyppmd) | **LGPL-2.1-or-later** | PPMd decode for third-party PR86 / PR91 / HPAC-family archives. Users opting in accept the LGPL obligation. |
| `[mlx]` | [MLX](https://github.com/ml-explore/mlx) | MIT | Apple Silicon array framework (advisory / dev-loop scoring only; never authoritative per CLAUDE.md "MPS auth eval is NOISE") |
| `[viz]` | [Plotly](https://github.com/plotly/plotly.py) | MIT | Interactive plotting |
| `[viz]` | [matplotlib](https://github.com/matplotlib/matplotlib) | matplotlib (PSF-equivalent BSD-compatible) | Static plotting |
| `[viz]` | [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU (HPND-Markdown / permissive) | Image I/O |
| `[viz]` | [imageio](https://github.com/imageio/imageio) | BSD-2-Clause | Video and image I/O |
| `[analysis]` | [Dask](https://github.com/dask/dask) | BSD-3-Clause | Parallel array analytics |
| `[analysis]` | [Polars](https://github.com/pola-rs/polars) | MIT | Columnar dataframe analytics |
| `[notebooks]` | [marimo](https://github.com/marimo-team/marimo) | Apache-2.0 | Reactive notebooks |
| `[runtime]` | [PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause | Video decode (pyav ffmpeg bindings) |
| `[runtime]` | [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | Tensor serialization |
| `[runtime]` | [opencv-python](https://github.com/opencv/opencv-python) | Apache-2.0 | Frame-level utilities |
| `[runtime]` | [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | Pre-trained vision models |
| `[runtime]` | [einops](https://github.com/arogozhnikov/einops) | MIT | Tensor reshape primitives |
| `[runtime]` | [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) | MIT | SegNet architecture (used by the contest scorer) |
| `[dev]` | [pytest](https://github.com/pytest-dev/pytest) | MIT | Test runner |
| `[dev]` | [hypothesis](https://github.com/HypothesisWorks/hypothesis) | MPL-2.0 | Property-based testing |
| `[dev]` | [ruff](https://github.com/astral-sh/ruff) | MIT | Linter and formatter |
| `[dev]` | [mypy](https://github.com/python/mypy) | MIT | Type checker |
| `[dev]` | [scipy](https://github.com/scipy/scipy) | BSD-3-Clause | Numerical routines (Rodrigues / Rotation reference) |
| `[dev]` | [PyWavelets](https://github.com/PyWavelets/pywt) | MIT | Wavelet oracle for tests |
| `[cloud]` | [Lightning AI SDK](https://github.com/Lightning-AI/lightning-sdk) | Apache-2.0 | Lightning batch jobs |
| `[cloud]` | [Modal](https://github.com/modal-labs/modal-client) | Apache-2.0 | Modal Labs ephemeral compute |
| `[cloud]` | [vastai](https://github.com/vast-ai/vast-python) | MIT | Vast.ai instance orchestration |
| `[cloud]` | [kaggle](https://github.com/Kaggle/kaggle-api) | Apache-2.0 | Kaggle CLI |

## Upstream / referenced projects

| Project | License | Use |
|---|---|---|
| [comma.ai comma_video_compression_challenge](https://github.com/commaai/comma_video_compression_challenge) | MIT | Pinned upstream snapshot under `upstream/` is the authoritative contest scorer; we do not modify it. The challenge frames our task. |
| [comma.ai openpilot](https://github.com/commaai/openpilot) | MIT | Reference for OSS posture, codebase conventions, and the comma.ai camera geometry that informs our renderer design. |
| [Yeachan-Heo/oh-my-codex](https://github.com/Yeachan-Heo/oh-my-codex) | MIT | Conceptual reference for the OMX / Codex Ralph-style workflow. |
| [iannuttall/ralph](https://github.com/iannuttall/ralph) | MIT | Conceptual reference for file-based loop iteration. |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | MIT | Conceptual inspiration for constrained experiment loops. |
| [DSPy](https://github.com/stanfordnlp/dspy) / [GEPA](https://github.com/stanfordnlp/gepa) | MIT / Apache-2.0 | Conceptual inspiration for evolving prompts and other text-shaped artifacts. |
| [NousResearch/hermes-agent-self-evolution](https://github.com/NousResearch) | MIT | Conceptual inspiration for persistent optimization loops. |
| Modular Mojo documentation | Modular EULA | Referenced for the optional experimental Mojo lane (`mojo/`). |

## Pinned upstream snapshot

`upstream/` contains a pinned snapshot of the comma.ai contest scorer and
evaluator. It is read-only as far as this repository is concerned: per the
project's non-negotiable rule we do NOT modify `upstream/*`, `submissions/exact_current/inflate.py`,
`submissions/exact_current/inflate.sh`, or `start.sh` without explicit
operator approval. The upstream license applies to those files as
distributed by comma.ai.

## License obligations summary

- **Default `pip install tac`**: permissive only (MIT / Apache-2.0 / BSD).
  No copyleft obligations beyond preserving copyright notices.
- **`pip install tac[pr86_replay]`**: pulls `pyppmd` (LGPL-2.1-or-later).
  Users opting in must comply with LGPL terms when redistributing.
- **`pip install tac[dev]`**: pulls `hypothesis` (MPL-2.0). MPL applies only
  if modified hypothesis source is redistributed; using it as a test
  dependency carries no obligation on `tac` itself.

Please review the licenses of any third-party code you clone or install
separately. If you find an entry that is incorrect or missing, please open
an issue.
