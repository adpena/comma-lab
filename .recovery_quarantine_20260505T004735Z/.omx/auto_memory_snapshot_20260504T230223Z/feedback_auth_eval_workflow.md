---
name: Auth Eval Workflow — Must Re-Inflate Before Eval
description: evaluate.py reads stale inflated/ dir. Must run inflate.sh with new checkpoint FIRST.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
The upstream evaluate.py reads from `submission_dir/inflated/` — it does NOT
call inflate.sh or re-inflate. If the inflated directory contains frames from
a previous checkpoint, the eval scores that old checkpoint silently.

**Correct workflow:**
1. Copy new postfilter_int8.pt to submissions/robust_current/
2. Rebuild archive.zip (or swap the .pt inside)
3. Unzip archive to temp dir
4. Run inflate.sh on the temp dir → submissions/robust_current/inflated/
5. THEN run evaluate.py

**Why:** We wasted 15 minutes on a stale auth eval that returned baseline numbers
because the inflated/ directory still had the old dilated model's output.

**How to apply:** Always delete inflated/ before auth eval. Or check the mtime
of inflated/0.raw against the postfilter_int8.pt timestamp.
