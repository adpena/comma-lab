"""
Legacy Modal deployment scripts — preserved for reference.

These scripts are superseded by the canonical asymmetric warp deploy pipeline.
They contain useful patterns (resume logic, volume mounting, precompute upload)
that may be harvested for future deploys.

Moved here 2026-04-12 during Modal infrastructure canonicalization.

Dead techniques (do NOT revive):
  - modal_dilated_kl_hardframe_deploy.py — KL distill is structurally dead

Historical value:
  - modal_nuclear_deploy.py — good resume/checkpoint pattern
  - modal_renderer_smoke_deploy.py — renderer smoke test pattern
"""
