# SPDX-License-Identifier: MIT
"""P2 torch-vehicle adapter for the vendored PR95 HNeRV-Muon trainer.

The vendored PR95 trainer
(``experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/
source/submissions/hnerv_muon/src``) is the ONLY Modal-runnable, basin-proven
vehicle for the approved $100 n600 capstone (the MLX capstone is Apple-only).
But the vendored ``train.py`` is a single ~50-hour from-scratch script with NO
resume, NO telemetry, and a HARDCODED ``base_channels=36`` — it cannot run the
``base_ch=20`` rate-win config, cannot survive a SIGURG/OOM/reboot without
losing the whole run, and emits no observable trajectory.

This package is the THIN ADAPTER that brings the torch vehicle to MLX parity
WITHOUT editing the pristine vendored source (CLAUDE.md "Forbidden in-place
edits to public PR intake clones"). It imports the vendored
``HNeRVDecoder`` / ``Muon`` / loss fns / ``precompute_targets`` / ``build_archive``
/ ``parse_archive`` / ``evaluate_decoder`` / ``compute_score`` as a library and
adds, in the adapter only:

* ``base_channels`` + ``latent_dim`` threaded through the curriculum (the
  vendored loop hardcodes 36/28 in 3 places);
* complete stage+epoch checkpoint/resume (params + Muon momentum + AdamW
  state + EMA shadow + curriculum position + cosine/sched state), atomic
  write, done-marker-on-exit (the "Durable detached daemons" +
  "LONG RESUMABLE SATURATION SWEEPS" non-negotiables);
* BEST-checkpoint-by-canonical-score tracking (the EMA shadow is the
  inference/export bytes — the EMA non-negotiable);
* per-stage/epoch d_seg / d_pose / rate / loss / lr telemetry to durable
  JSONL + a summary (the "Max observability" non-negotiable);
* export-path verification (the vendored ``codec_stage`` -> ``0.bin`` ->
  ``archive.zip`` -> ``inflate.sh`` -> ``inflate.py`` chain).

Authority: this is infrastructure + a local advisory loop. $0, local,
torch-CPU (TRUSTED per CLAUDE.md "local CPU + MLX GPU good"); NO MPS. The
in-loop d_seg/d_pose are ``[macOS-CPU advisory]`` until the byte-closed archive
is run through ``upstream/evaluate.py``. The actual $100 Modal run is
post-symposium.
"""

from tac.torch_vehicle.checkpoint import (
    CHECKPOINT_VERSION,
    TorchCheckpointPosition,
    checkpoint_exists,
    is_done,
    load_checkpoint,
    read_manifest,
    save_checkpoint,
    write_done_marker,
)
from tac.torch_vehicle.vendored_imports import (
    VENDORED_SRC,
    import_vendored,
    resolve_challenge_root,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "VENDORED_SRC",
    "TorchCheckpointPosition",
    "checkpoint_exists",
    "import_vendored",
    "is_done",
    "load_checkpoint",
    "read_manifest",
    "resolve_challenge_root",
    "save_checkpoint",
    "write_done_marker",
]
