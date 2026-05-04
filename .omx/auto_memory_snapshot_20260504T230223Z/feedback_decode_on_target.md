---
name: Decode on Target — No Precomputed Frames Across Environments
description: Every deployment target decodes from raw video natively. Precomputed frames never cross environment boundaries. Eliminates DALI/PyAV mismatch by construction.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Every deployment target (Lightning, Kaggle, Modal) decodes from raw video using the native GPU decode path (DALI). Precomputed frames NEVER cross environment boundaries.

**Why:** The 29x PoseNet divergence was caused by DALI vs PyAV decode differences. Precomputing frames locally (PyAV) and uploading to GPU targets (DALI scorer) introduces a systematic distribution shift that no amount of validation can fully eliminate. Decoding on the target machine using the same pipeline as the scorer eliminates this by construction.

**How to apply:**
- Lightning/Kaggle/Modal: decode from raw .hevc/.mkv using DALI, extract masks, train — all on the same machine
- Local (M5 Max): decode via PyAV for development iteration only. NEVER trust local scores as authoritative.
- Do NOT upload experiments/precomputed_local to any GPU target
- The --precomputed flag is for local development ONLY
- Each training run starts with a ~2 minute decode preamble — negligible vs 48h training
- This applies to ALL deployment targets, not just Lightning
