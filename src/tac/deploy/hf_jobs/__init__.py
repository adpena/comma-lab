# SPDX-License-Identifier: MIT
"""Canonical Hugging Face Jobs dispatch helpers (Catalog #342 + Item #878).

Mirrors :mod:`tac.deploy.modal` 4-layer canonical pattern per CLAUDE.md
Catalog #245 (modal_call_id_ledger canonical exemplar) for HF Jobs paid
remote-GPU dispatch:

- Layer 1: ``tac.deploy.hf_jobs.job_id_ledger`` — fcntl-locked JSONL +
  append-only HISTORICAL_PROVENANCE per Catalog #110 / #113 / #128 / #131 /
  #138.
- Layer 2: ``tools/dispatch_hf_jobs_vision_training.py`` — operator-facing
  CLI wrapping ``huggingface_hub.HfApi().run_uv_job(...)`` per the
  ``huggingface-skills:hugging-face-vision-trainer`` plugin canonical
  pattern.
- Layer 3: STRICT preflight gates (sister to Catalog #270 dispatch protocol;
  this surface is platform="hf_jobs" so the existing platform-scoped
  gates apply transparently).
- Layer 4: ``tools/operator_authorize.py::_dispatch_hf_jobs`` runtime
  wire-in (deferred; Phase 8 of the HF-DATASET-PREP-AND-JOBS landing).

Sister of :mod:`tac.deploy.modal.call_id_ledger`.
"""

from __future__ import annotations

from tac.deploy.hf_jobs.job_id_ledger import (
    HF_JOBS_CALL_ID_LEDGER_PATH,
    EVENT_DISPATCHED,
    EVENT_HARVESTED,
    EVENT_FAILED,
    EVENT_STALE,
    register_dispatched_hf_jobs_id,
    update_hf_jobs_outcome,
    query_by_hf_jobs_id,
    query_by_lane,
)

__all__ = (
    "HF_JOBS_CALL_ID_LEDGER_PATH",
    "EVENT_DISPATCHED",
    "EVENT_HARVESTED",
    "EVENT_FAILED",
    "EVENT_STALE",
    "register_dispatched_hf_jobs_id",
    "update_hf_jobs_outcome",
    "query_by_hf_jobs_id",
    "query_by_lane",
)
