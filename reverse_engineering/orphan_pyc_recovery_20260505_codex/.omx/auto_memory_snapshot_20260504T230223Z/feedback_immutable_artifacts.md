---
name: Immutable Artifacts — Never Overwrite Checkpoints or Archives
description: postfilter_int8.pt was silently overwritten with wrong checkpoint, causing all evals to show wrong score. Hash-verify everything.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
CRITICAL LESSON (2026-04-12): The 1.33 score was REAL but we couldn't reproduce it because postfilter_int8.pt in submissions/robust_current/ had been silently overwritten with a different (worse) checkpoint. The correct checkpoint was at reports/raw/modal_artifacts/postfilter_dilated_h64_long1000_modal_best_int8.pt (md5 62e4b2dc). The wrong one was md5 1d9d1128.

This caused:
- Hours of debugging thinking local scorer disagreed with auth
- False DALI vs PyAV theory
- False AllNorm regression analysis (2.15 was partly wrong checkpoint + wrong exploits)
- Multiple wasted eval runs

**Rules:**
- NEVER overwrite postfilter_int8.pt directly. Always version with timestamp.
- EVERY eval must log the checkpoint md5 hash
- runner.py experiment records capture this (config_fingerprint + checkpoint_md5)
- promoted_result.json must include checkpoint md5
- Before any eval: verify checkpoint hash matches promoted_result.json
- The archive.zip MUST contain the same checkpoint that's referenced in promoted_result
