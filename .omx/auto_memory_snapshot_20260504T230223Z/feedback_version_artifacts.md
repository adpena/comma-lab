---
name: Version All Artifacts — Never Overwrite
description: compress.sh overwrites archive.zip, destroying previous versions. Every experiment must preserve its exact artifacts.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
CRITICAL LESSON (2026-04-12): The 1.33 archive (864,167 bytes) was overwritten by re-encoding. All eval attempts today used a DIFFERENT archive (888,824 bytes) producing score 2.07. The original 1.33 archive was only found because it survived in the upstream repo copy.

**Why:** compress.sh writes to a fixed path (submissions/robust_current/archive.zip). Every re-encode overwrites the previous version. There's no versioning, no backup, no way to know which archive produced which score.

**How to apply:**
- NEVER overwrite archive.zip directly. Write to a timestamped path first.
- Before any compress: `cp archive.zip archive_$(date +%Y%m%dT%H%M%S).zip`
- Each experiment run must have its own directory with: archive.zip, config.env snapshot, checkpoint, report
- The runner.py eval_runs/ pattern is correct — extend it to compress too
- Record the archive md5 hash in every eval report
- The promoted_result.json should include the archive md5
