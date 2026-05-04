---
name: Complete Experiment Records — Non-Negotiable
description: Every eval run must auto-capture full config, hashes, flags, git state. results.jsonl must be complete. No manual recording.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
RULE: Every evaluation run MUST automatically capture and record ALL of the following. No manual entry. No missing fields. The runner does this, not the human.

**Required fields in every results.jsonl entry:**
- Full config.env snapshot (every encode parameter: CRF, codec, color_matrix, scale, keyint, film-grain, preset)
- Checkpoint md5 hash
- Archive md5 hash and byte count
- Git commit hash at time of evaluation
- Machine/platform identifier
- All inflate flags (brightness_shift, chroma_smooth, deblock, multi_pass, tto_steps)
- Exact inflate command used
- Scorer device (cpu/cuda/mps)
- All score components (pose, seg, rate, total)
- Timestamp
- Run directory path

**Why:** The 1.33 result had only 3 config fields recorded. When we tried to reproduce it, we couldn't — the archive had been overwritten by a re-encode with different settings. We spent hours debugging a 2.07 score that was simply a different archive.

**How to apply:**
- runner.py must auto-capture all fields into run_dir/state.json AND append to results.jsonl
- compress.sh must NEVER overwrite archive.zip — write to timestamped path, symlink as archive.zip
- Every run_dir must contain: archive.zip copy, config.env snapshot, checkpoint copy or symlink
- The promoted_result.json must include archive md5
